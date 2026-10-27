"""Mainline acceptance for one recursive CUDA Z3 vertical slice."""

from __future__ import annotations

import pytest
import torch
import tvm_ffi

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.testing.ir import collect_nodes
from tvm import tirx
from tvm.ir.instrument import pass_instrument


_CONFIG = "tl.enable_z3_pipeline_planning"
_PREPARED = "tl.cuda_z3_recursive_prepared"
_CLOCKS = "tl.cuda_z3_prepared_clock_map"
_EPOCHS = "tl.cuda_z3_prepared_buffer_epochs"
_REQUIRED_FFI = (
    "tl.cuda.transform.z3_schedule_loop_python",
    "tl.cuda.transform.NormalizeZ3ControlFlow",
    "tl.cuda.transform.MaterializeZ3ScheduleUnits",
    "tl.cuda.transform.Z3AnalyzeRecursiveLifetimes",
    "tl.cuda.transform.Z3BuildWSSchedule",
    "tl.cuda.transform.LowerZ3ScheduleUnits",
    "tl.transform.RestoreWhileLoops",
)


@pytest.fixture(autouse=True)
def _run_owned_tilelang_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "tilelang-cache"))


def _require_recursive_library():
    missing = [name for name in _REQUIRED_FFI if tvm_ffi.get_global_func(name, allow_missing=True) is None]
    if missing:
        pytest.skip(f"recursive mainline requires the centrally rebuilt TileLang library: {missing}")


def _nested_fixed_stage_kernel():
    """Slice E's nested fixed-stage/two-version example as a runnable kernel."""

    @T.prim_func
    def main(B: T.Tensor((8,), "int32")):
        with T.Kernel(1, threads=1):
            shared = T.alloc_shared((8,), "int32", scope="shared")
            T.annotate_buffer_versions({shared: (2, "iteration")})
            for _o in T.serial(2):
                for i in T.serial(
                    4,
                    annotations={
                        "num_stages": 2,
                        "tl_z3_pipeline_enable_offset": True,
                        "tl_z3_pipeline_resource_flags": [1, 1],
                        "tl_z3_pipeline_buffer_vars": [shared.data],
                        "tl_z3_pipeline_buffer_sizes": [32],
                        "tl_z3_pipeline_memory_groups": [[64, 0]],
                    },
                ):
                    with T.Stage(0, schema="recursive"), T.Task(latency=2, ii=1, schema="recursive"):
                        shared[i] = i
                    with T.Stage(1, schema="recursive"), T.Task(latency=2, ii=1, schema="recursive"):
                        B[i] = shared[i] + 1

    return main


def _ints(values):
    return [int(value) for value in values]


def _loops(module):
    return collect_nodes(module["main"].body, tirx.For)


def _root(module):
    return next(
        node
        for node in collect_nodes(module["main"].body, tirx.SBlock)
        if node.name_hint == "tilelang_root"
    )


@pass_instrument
class _RecursiveAudit:
    def __init__(self):
        self.events = []
        self.before = {}
        self.after = {}

    def run_before_pass(self, mod, info):
        name = str(info.name)
        self.events.append(name)
        self.before[name] = mod

    def run_after_pass(self, mod, info):
        self.after[str(info.name)] = mod


def _assert_subsequence(events, expected):
    position = 0
    for event in events:
        if position < len(expected) and event == expected[position]:
            position += 1
    assert position == len(expected), f"missing recursive pass order {expected!r} in {events!r}"


def test_recursive_mainline_compile_only_order_and_before_after_audit():
    _require_recursive_library()
    target = tvm.target.Target(
        {
            "kind": "cuda",
            "arch": "sm_120a",
            "max_shared_memory_per_block": 1 << 20,
        }
    )
    audit = _RecursiveAudit()
    with target, tvm.transform.PassContext(config={_CONFIG: True}, instruments=[audit]):
        artifact = tilelang.lower(
            _nested_fixed_stage_kernel(),
            target=target,
            enable_device_compile=False,
        )
    assert artifact.kernel_source

    _assert_subsequence(
        audit.events,
        [
            "tl.UnrollLoopSkipZ3Atomic",
            "tl.cuda.NormalizeZ3ControlFlow",
            "tl.cuda.MaterializeZ3ScheduleUnits",
            "tl.Z3PipelinePlanning",
            "tl.cuda.BuildZ3WSSchedule",
            "tl.cuda.LowerZ3ScheduleUnits",
            "tl.PipelinePlanning",
            "tl.InjectSoftwarePipeline",
            "tl.RestoreWhileLoops",
        ],
    )
    assert "tl.NormalizeZ3ScheduleUnits" not in audit.events
    assert "tl.NormalizeZ3BufferVersions" not in audit.events
    assert "tl.Z3OwnerRingPlanning" not in audit.events
    assert "tl.MaterializeWSSchedule" not in audit.events

    planned = audit.after["tl.Z3PipelinePlanning"]
    _outer_planned, inner_planned = sorted(_loops(planned), key=lambda loop: int(loop.extent))
    assert int(inner_planned.annotations["tl_z3_pipeline_audit_ii"]) == 2
    assert _ints(inner_planned.annotations["tl_z3_pipeline_audit_child_starts"]) == [0, 3]
    assert _ints(inner_planned.annotations["tl_z3_pipeline_audit_child_stages"]) == [0, 1]

    # Immediately before stock planning, Slice E has restored physical
    # producer/consumer order and published its exact prepared-loop contract.
    lowered = audit.before["tl.PipelinePlanning"]
    outer_lowered, inner_lowered = sorted(_loops(lowered), key=lambda loop: int(loop.extent))
    assert _ints(inner_lowered.annotations["tl_pipeline_order"]) == [0, 1]
    assert _ints(inner_lowered.annotations["tl_pipeline_stage"]) == [0, 1]
    assert int(inner_lowered.annotations["num_stages"]) == 2
    assert bool(inner_lowered.annotations[_PREPARED]) is True
    inner_stores = collect_nodes(inner_lowered.body, tirx.BufferStore)
    assert [str(store.buffer.name) for store in inner_stores] == ["shared", "B"]
    shared = next(buffer for buffer in _root(lowered).alloc_buffers if buffer.name == "shared")
    assert int(inner_lowered.annotations["tl_pipeline_buffer_versions"][shared.data]) == 2
    prepared_root = _root(lowered)
    clock = prepared_root.annotations[_CLOCKS][shared.data]
    clock_vars = collect_nodes(clock, tirx.Var)
    assert any(var.same_as(inner_lowered.loop_var) for var in clock_vars)
    assert any(var.same_as(outer_lowered.loop_var) for var in clock_vars)
    epoch_plan = prepared_root.annotations[_EPOCHS]
    assert len(epoch_plan) == 1
    assert str(epoch_plan[0]["clock"]) == "iteration"
    assert int(epoch_plan[0]["versions"]) == 2

    stock = audit.after["tl.PipelinePlanning"]
    outer_stock, inner_stock = sorted(_loops(stock), key=lambda loop: int(loop.extent))
    assert _ints(outer_stock.annotations["software_pipeline_order"]) == [0]
    assert _ints(outer_stock.annotations["software_pipeline_stage"]) == [0]
    assert _ints(inner_stock.annotations["software_pipeline_order"]) == [0, 1]
    assert _ints(inner_stock.annotations["software_pipeline_stage"]) == [0, 1]
    assert "software_pipeline_async_stages" not in outer_stock.annotations
    assert "software_pipeline_async_stages" not in inner_stock.annotations

    # After injection, the single leading ring dimension uses the flattened
    # nested lexical clock, not an immediate-inner-loop modulo fallback.
    injected = audit.after["tl.InjectSoftwarePipeline"]
    injected_root = _root(injected)
    injected_shared = next(buffer for buffer in injected_root.alloc_buffers if buffer.data.same_as(shared.data))
    assert [int(dim) for dim in injected_shared.shape] == [2, 8]
    injected_nodes = collect_nodes(injected["main"].body, tirx.BufferLoad) + collect_nodes(
        injected["main"].body, tirx.BufferStore
    )
    accesses = [
        node
        for node in injected_nodes
        if node.buffer.data.same_as(shared.data) and len(node.indices) == 2
    ]
    assert accesses
    for access in accesses:
        version_vars = collect_nodes(access.indices[0], tirx.Var)
        assert any(var.same_as(outer_lowered.loop_var) for var in version_vars), access


@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_sm120_recursive_mainline_exact_output():
    _require_recursive_library()
    target = tvm.target.Target(
        {
            "kind": "cuda",
            "arch": "sm_120a",
            "max_shared_memory_per_block": int(torch.cuda.get_device_properties(0).shared_memory_per_block_optin),
        }
    )
    kernel = tilelang.compile(
        _nested_fixed_stage_kernel(),
        target=target,
        out_idx=None,
        execution_backend="tvm_ffi",
        pass_configs={_CONFIG: True},
    )
    expected = torch.tensor([1, 2, 3, 4, -7, -7, -7, -7], dtype=torch.int32, device="cuda")
    for _ in range(4):
        output = torch.full((8,), -7, dtype=torch.int32, device="cuda")
        kernel(output)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
