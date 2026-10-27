"""Host IR coverage for the recursive-Z3 to stock-pipeline vertical slice."""

from __future__ import annotations

import pytest
import tvm_ffi

import tilelang as tl
import tilelang.testing
from tilelang import language as T  # noqa: F401  # Register the real solver FFI.
from tilelang import tvm
from tilelang.testing.ir import collect_nodes
from tvm import tirx


_SOLVER = "tl.cuda.transform.z3_schedule_loop_python"
_CONFIG = "tl.enable_z3_pipeline_planning"
_PREPARED = "tl.cuda_z3_recursive_prepared"
_CLOCKS = "tl.cuda_z3_prepared_clock_map"
_EPOCHS = "tl.cuda_z3_prepared_buffer_epochs"
_SM80 = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})


def _unit(body, source_id, stage, *, latency=2, ii=1):
    marker = tirx.AttrStmt(
        {"latency": latency, "ii": ii}, "tl.cuda_task", 1, body
    )
    return tirx.AttrStmt(
        {"stage": stage, "source_id": source_id}, "tl.schedule_unit", 1, marker
    )


def _control(loop_var, extent, children, source_id, *, annotations=None, stage=-1):
    loop = tirx.For(
        loop_var,
        0,
        extent,
        tirx.ForKind.SERIAL,
        tirx.SeqStmt(children) if len(children) > 1 else children[0],
        annotations=annotations or {},
    )
    return tirx.AttrStmt(
        {"stage": stage, "source_id": source_id}, "tl.schedule_unit", 1, loop
    )


def _ints(values):
    return [int(value) for value in values]


def _module(root):
    func = tirx.PrimFunc([], root).with_attr("global_symbol", "main")
    return tvm.tirx.transform.BindTarget(_SM80)(tvm.IRModule.from_expr(func))


def _loops(module):
    return collect_nodes(module["main"].body, tirx.For)


def _root(module):
    return next(
        node
        for node in collect_nodes(module["main"].body, tirx.SBlock)
        if node.name_hint == "tilelang_root"
    )


def _nested_fixed_stage_module(shared, out, inner, outer):
    # This is Slice E's nested fixed-stage/two-version example. Its restored
    # inner list will be physical order (producer, consumer), as in source.
    producer = _unit(tirx.BufferStore(shared, inner, [inner]), 1, 0)
    consumer = _unit(
        tirx.BufferStore(out, shared[inner] + 1, [inner]), 2, 1
    )
    inner_loop = _control(
        inner,
        4,
        [producer, consumer],
        10,
        annotations={
            "num_stages": tirx.IntImm("int64", 2),
            "tl_z3_pipeline_enable_offset": True,
            "tl_z3_pipeline_resource_flags": [
                tirx.IntImm("int64", 1),
                tirx.IntImm("int64", 1),
            ],
            "tl_z3_pipeline_buffer_vars": [shared.data],
            "tl_z3_pipeline_buffer_sizes": [tirx.IntImm("int64", 32)],
            "tl_z3_pipeline_memory_groups": [
                [tirx.IntImm("int64", 64), tirx.IntImm("int64", 0)]
            ],
        },
    )
    outer_loop = _control(outer, 2, [inner_loop], 20)
    root = tirx.SBlock(
        [],
        [],
        [],
        "tilelang_root",
        outer_loop,
        alloc_buffers=[shared, out],
        annotations={
            "tl.cuda_z3_buffer_version_requests": {
                shared.data: tirx.IntImm("int64", 2)
            },
            "tl.cuda_z3_buffer_version_mode_requests": {
                shared.data: "iteration"
            },
        },
    )
    return _module(root)


def test_nested_fixed_stage_projection_uses_prepared_affine_clock():
    if (
        tvm_ffi.get_global_func(_SOLVER, allow_missing=True) is None
        or tvm_ffi.get_global_func(
            "tl.cuda.transform.LowerZ3ScheduleUnits", allow_missing=True
        )
        is None
    ):
        pytest.skip("recursive projection requires the rebuilt TileLang library")

    shared = tirx.decl_buffer((8,), "int32", name="shared", scope="shared")
    out = tirx.decl_buffer((8,), "int32", name="out", scope="global")
    inner = tirx.Var("i", "int32")
    outer = tirx.Var("o", "int32")

    with tvm.transform.PassContext(config={_CONFIG: True}):
        planned = tl.transform.Z3PipelinePlanning()(
            _nested_fixed_stage_module(shared, out, inner, outer)
        )
    lower = tvm_ffi.get_global_func("tl.cuda.transform.LowerZ3ScheduleUnits")
    lowered = lower()(planned)

    # Before stock planning: exact Slice E projection contract.
    outer_lowered, inner_lowered = sorted(
        _loops(lowered), key=lambda loop: int(loop.extent)
    )
    assert _ints(inner_lowered.annotations["tl_pipeline_order"]) == [0, 1]
    assert _ints(inner_lowered.annotations["tl_pipeline_stage"]) == [0, 1]
    assert int(inner_lowered.annotations["num_stages"]) == 2
    assert bool(inner_lowered.annotations[_PREPARED]) is True
    inner_stores = collect_nodes(inner_lowered.body, tirx.BufferStore)
    assert [store.buffer.name for store in inner_stores] == ["shared", "out"]
    versions = inner_lowered.annotations["tl_pipeline_buffer_versions"]
    assert int(versions[shared.data]) == 2

    prepared_root = _root(lowered)
    clocks = prepared_root.annotations[_CLOCKS]
    clock = clocks[shared.data]
    clock_vars = collect_nodes(clock, tirx.Var)
    assert any(var.same_as(inner) for var in clock_vars)
    assert any(var.same_as(outer) for var in clock_vars)
    epoch_plan = prepared_root.annotations[_EPOCHS]
    assert len(epoch_plan) == 1
    assert str(epoch_plan[0]["clock"]) == "iteration"
    assert int(epoch_plan[0]["versions"]) == 2

    # PipelinePlanning must visit the annotated child before the annotated
    # parent. Prepared markers suppress only legacy stage-level async fallback.
    stock = tl.transform.PipelinePlanning()(lowered)
    outer_stock, inner_stock = sorted(
        _loops(stock), key=lambda loop: int(loop.extent)
    )
    assert _ints(outer_stock.annotations["software_pipeline_order"]) == [0]
    assert _ints(outer_stock.annotations["software_pipeline_stage"]) == [0]
    assert _ints(inner_stock.annotations["software_pipeline_order"]) == [0, 1]
    assert _ints(inner_stock.annotations["software_pipeline_stage"]) == [0, 1]
    assert "software_pipeline_async_stages" not in outer_stock.annotations
    assert "software_pipeline_async_stages" not in inner_stock.annotations

    # After stock injection: one leading ring dimension and no immediate-i
    # modulo fallback. Every dynamic shared access uses the flattened nested
    # lexical clock, which retains the outer loop variable in this example.
    injected = tl.transform.InjectSoftwarePipeline()(stock)
    injected_root = _root(injected)
    injected_shared = next(
        buffer
        for buffer in injected_root.alloc_buffers
        if buffer.data.same_as(shared.data)
    )
    assert [int(dim) for dim in injected_shared.shape] == [2, 8]
    shared_loads = collect_nodes(injected["main"].body, tirx.BufferLoad)
    shared_stores = collect_nodes(injected["main"].body, tirx.BufferStore)
    shared_accesses = [
        node
        for node in shared_loads + shared_stores
        if node.buffer.data.same_as(shared.data) and len(node.indices) == 2
    ]
    assert shared_accesses
    for access in shared_accesses:
        version_vars = collect_nodes(access.indices[0], tirx.Var)
        assert any(var.same_as(outer) for var in version_vars), access


if __name__ == "__main__":
    tilelang.testing.main()
