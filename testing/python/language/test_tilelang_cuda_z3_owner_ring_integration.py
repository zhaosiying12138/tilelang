"""Function-level Z3 owner-ring integration and SM120A numerics."""

from __future__ import annotations

import json

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.language.annotations import z3_pipeline_annotations
from tilelang.testing.ir import collect_nodes


def _owner_loop_range(owner, extent, variant, A, private_bound):
    if owner != 0:
        return 0, extent
    if variant == "constant_bounds":
        return 2, 2 + extent
    if variant == "symbolic_bounds":
        return A.shape[0], A.shape[0] + A.shape[0] + 1
    if variant == "thread_min":
        tx = T.get_thread_binding()
        return tx, tx + extent
    if variant == "thread_extent":
        return 0, extent + T.get_thread_binding()
    if variant == "private_min":
        return private_bound[0], private_bound[0] + extent
    if variant == "private_extent":
        return 0, private_bound[0]
    return 0, extent


def _owner_kernel(depth: int, debug_dir: str, variant: str | None = None):
    extents = (3, 5)

    def owner_bodies(owner: int):
        return {
            "Producer": [
                T.WSSync.producer_acquire("ring"),
                f"produce{owner}",
                T.WSSync.producer_commit("ring"),
            ],
            "Consumer": [
                T.WSSync.consumer_wait("ring"),
                f"consume{owner}",
                T.WSSync.consumer_release("ring"),
                f"output{owner}",
            ],
        }

    @T.prim_func
    def kernel(A: T.Tensor((T.dynamic("n"),), T.float32), B: T.Tensor((64,), T.float32)):
        with T.Kernel(1, threads=128):
            ring = T.alloc_shared((4,), T.float32, scope="shared")
            frag = T.alloc_fragment((4,), T.float32)
            private_bound = T.alloc_local((1,), T.int32) if variant in ("private_min", "private_extent") else None
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8),
                    ],
                    pipelines=[
                        T.WSPipeline(
                            "ring",
                            [ring],
                            depth=depth,
                            owner_scopes=("owner0", "owner1"),
                        )
                    ],
                    scopes=[
                        T.WSScope("owner0", owner_bodies(0)),
                        T.WSScope("owner1", owner_bodies(1)),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {
                                "Producer": ["owner0", "owner1"],
                                "Consumer": ["owner0", "owner1"],
                            },
                        ),
                    ],
                )
            )
            for owner, extent in enumerate(extents):
                annotations = z3_pipeline_annotations(
                    latencies=[4],
                    iis=[1],
                    resource_flags=[1],
                    owner_exclusion_deps=[(0, 1, 1, 2)],
                    enable_offset=False,
                    verbose=True,
                    debug_dir=debug_dir,
                    owner_ring=owner,
                )
                for i in T.Pipelined(
                    *_owner_loop_range(owner, extent, variant, A, private_bound),
                    num_stages=depth,
                    annotations={**annotations, "tl.ws_op_id": f"owner{owner}"},
                ):
                    T.fill(ring, i + 16 * owner, annotations={T.WSID: f"produce{owner}"})
                    T.copy(ring, frag, annotations={T.WSID: f"consume{owner}"})
                    T.copy(frag, B[(owner * 5 + i) * 4], annotations={T.WSID: f"output{owner}"})

    return kernel


def _function_pass(mod, config_key):
    with tvm.transform.PassContext(config={config_key: True}):
        return tilelang.transform.Z3OwnerRingPlanning()(mod)


def _launch_thread_var(func):
    (bind,) = [
        node
        for node in collect_nodes(func, tvm.tirx.Bind)
        if isinstance(node.value, tvm.tirx.Call)
        and node.value.op.name == "tl.launch_thread_idx"
        and int(node.value.args[0]) == 0
    ]
    return bind.var


def _stamp_owner_step(func, step):
    def add_step(stmt):
        if isinstance(stmt, tvm.tirx.For) and stmt.annotations.get(T.WSID) == "owner0":
            return tvm.tirx.For(
                stmt.loop_var,
                stmt.min,
                stmt.extent,
                stmt.kind,
                stmt.body,
                stmt.thread_binding,
                stmt.annotations,
                step=step,
            )
        return None

    return func.with_body(tvm.tirx.stmt_functor.ir_transform(func.body, None, add_step, ["tirx.For"]))


def _run_host_function_pass(func):
    config_key = tilelang.PassConfigKey.TL_ENABLE_Z3_PIPELINE_PLANNING
    target = tvm.target.Target(
        {
            "kind": "cuda",
            "arch": "sm_100a",
            "max_shared_memory_per_block": 1 << 20,
        }
    )
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    return _function_pass(mod, config_key)


def test_function_level_owner_validation_and_materialization(tmp_path):
    config_key = tilelang.PassConfigKey.TL_ENABLE_Z3_PIPELINE_PLANNING
    target = tvm.target.Target(
        {
            "kind": "cuda",
            "arch": "sm_100a",
            "max_shared_memory_per_block": 1 << 20,
        }
    )
    func = _owner_kernel(2, str(tmp_path)).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = _function_pass(mod, config_key)
    validated = [
        block
        for block in collect_nodes(mod["main"], tvm.tirx.SBlock)
        if "tl_z3_owner_ring_validated" in block.annotations
    ]
    assert len(validated) == 1
    for loop in collect_nodes(mod["main"], tvm.tirx.For):
        # Metadata is retained for audit; the owner-specific pass skips generic planning.
        assert "tl_z3_owner_ring" in loop.annotations
        assert "tl_z3_pipeline" in loop.annotations
        assert "num_stages" in loop.annotations
    mod = tilelang.cuda.transform.MaterializeWSSchedule()(mod)
    schedule_files = list(tmp_path.glob("schedule_*/schedule_info.json"))
    assert len(schedule_files) == 1
    schedule = json.loads(schedule_files[0].read_text())
    assert schedule["fallback"] is False
    assert schedule["minimal_II"] == 3
    left, right = schedule["start_times"]
    assert left % 3 == 0
    assert right % 3 == 1


@pytest.mark.parametrize("variant", ["thread_min", "thread_extent", "private_min", "private_extent"])
def test_function_level_owner_rejects_nonuniform_loop_bounds(variant, tmp_path):
    func = _owner_kernel(2, str(tmp_path), variant=variant)
    with pytest.raises(Exception, match="role-uniform"):
        _run_host_function_pass(func)
    assert not list(tmp_path.glob("schedule_*/schedule_info.json"))


def test_function_level_owner_rejects_thread_dependent_step(tmp_path):
    func = _owner_kernel(2, str(tmp_path))
    stepped = _stamp_owner_step(func, _launch_thread_var(func) + 1)
    with pytest.raises(Exception, match="role-uniform"):
        _run_host_function_pass(stepped)
    assert not list(tmp_path.glob("schedule_*/schedule_info.json"))


def test_function_level_owner_still_rejects_nonunit_step(tmp_path):
    stepped = _stamp_owner_step(_owner_kernel(2, str(tmp_path)), tvm.tirx.IntImm("int32", 2))
    with pytest.raises(Exception, match="non-unit loop step"):
        _run_host_function_pass(stepped)
    assert not list(tmp_path.glob("schedule_*/schedule_info.json"))


@pytest.mark.parametrize("variant", ["constant_bounds", "symbolic_bounds"])
def test_function_level_owner_accepts_role_uniform_loop_bounds(variant, tmp_path):
    mod = _run_host_function_pass(_owner_kernel(2, str(tmp_path), variant=variant))
    (owner0,) = [
        loop
        for loop in collect_nodes(mod["main"], tvm.tirx.For)
        if loop.annotations.get(T.WSID) == "owner0"
    ]
    if variant == "constant_bounds":
        assert int(owner0.min) == 2
        assert int(owner0.extent) == 3
    else:
        assert isinstance(owner0.min, tvm.tirx.Var) and owner0.min.name == "n"
        extent = tvm.arith.Analyzer().simplify(owner0.extent)
        tvm.ir.assert_structural_equal(extent, owner0.min + 1)
    validated = [
        block
        for block in collect_nodes(mod["main"], tvm.tirx.SBlock)
        if "tl_z3_owner_ring_validated" in block.annotations
    ]
    assert len(validated) == 1


@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_sm120_owner_ring_forward_wraparound_and_reuse(depth, tmp_path):
    config_key = tilelang.PassConfigKey.TL_ENABLE_Z3_PIPELINE_PLANNING
    target = tvm.target.Target(
        {
            "kind": "cuda",
            "arch": "sm_120a",
            "max_shared_memory_per_block": int(
                torch.cuda.get_device_properties(0).shared_memory_per_block_optin
            ),
        }
    )
    kernel = tilelang.compile(
        _owner_kernel(depth, str(tmp_path)),
        target=target,
        out_idx=None,
        execution_backend="tvm_ffi",
        pass_configs={config_key: True},
    )
    a = torch.arange(64, dtype=torch.float32, device="cuda")
    b = torch.full((64,), -7.0, dtype=torch.float32, device="cuda")
    expected = b.clone()
    for owner, extent in enumerate((3, 5)):
        for index in range(extent):
            output = (owner * 5 + index) * 4
            expected[output : output + 4] = index + 16 * owner
    for _ in range(4):
        kernel(a, b)
        torch.testing.assert_close(b, expected, rtol=0, atol=0)
    schedule_files = list(tmp_path.glob("schedule_*/schedule_info.json"))
    assert len(schedule_files) == 1
    schedule = json.loads(schedule_files[0].read_text())
    assert schedule["fallback"] is False
    assert schedule["minimal_II"] == 3
