"""Host-only recursive CUDA Z3 planning/projection vertical slice.

The test uses the real Z3 solver and no GPU. It skips until the central shared
library is rebuilt with the Slice E implementation.
"""

from __future__ import annotations

import pytest
import tvm_ffi

import tilelang as tl
from tilelang import language as T  # noqa: F401  # Register the real solver FFI.
from tilelang import tvm
from tilelang.testing.ir import collect_nodes
from tvm import tirx


_SOLVER = "tl.cuda.transform.z3_schedule_loop_python"
_CONFIG = "tl.enable_z3_pipeline_planning"
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


def _main_module(root):
    func = tirx.PrimFunc([], root).with_attr("global_symbol", "main")
    return tvm.tirx.transform.BindTarget(_SM80)(tvm.IRModule.from_expr(func))


def _loops(module):
    return collect_nodes(module["main"].body, tirx.For)


def test_nested_fixed_stage_ring_projects_to_stock_pipeline():
    previous = tvm_ffi.get_global_func(_SOLVER, allow_missing=True)
    lower_factory = tvm_ffi.get_global_func(
        "tl.cuda.transform.LowerZ3ScheduleUnits", allow_missing=True
    )
    if previous is None or lower_factory is None:
        pytest.skip("recursive planning requires the centrally rebuilt TileLang library")

    calls = []

    def recording_solver(*args):
        calls.append(args)
        return previous(*args)

    tvm_ffi.register_global_func(_SOLVER, f=recording_solver, override=True)
    try:
        shared = tirx.decl_buffer((8,), "int32", name="shared", scope="shared")
        out = tirx.decl_buffer((8,), "int32", name="out", scope="global")
        inner = tirx.Var("i", "int32")
        outer = tirx.Var("o", "int32")

        # Normalized source schedule: produce at stage 0, consume at stage 1,
        # with an exact two-version public request for the shared ring.
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

        with tvm.transform.PassContext(config={_CONFIG: True}):
            planned = tl.transform.Z3PipelinePlanning()(_main_module(root))

        # Real solver calls are bottom-up: inner children, outer Control, root.
        assert [len(call[1]) for call in calls] == [2, 1, 1]
        assert _ints(calls[0][1]) == [2, 2]
        assert list(calls[0][12]) == [0, 1]
        assert bool(calls[0][13]) is True

        outer_planned, inner_planned = sorted(
            _loops(planned), key=lambda loop: int(loop.extent)
        )
        assert int(inner_planned.annotations["tl_z3_pipeline_audit_ii"]) == 2
        assert _ints(inner_planned.annotations["tl_z3_pipeline_audit_child_starts"]) == [0, 3]
        assert _ints(inner_planned.annotations["tl_z3_pipeline_audit_child_stages"]) == [0, 1]
        assert _ints(inner_planned.annotations["tl_z3_pipeline_audit_child_residues"]) == [0, 1]
        assert int(inner_planned.annotations["tl_z3_pipeline_audit_first_span"]) == 5
        assert int(inner_planned.annotations["tl_z3_pipeline_audit_last_iter_offset"]) == 6
        assert int(outer_planned.annotations["tl_z3_pipeline_audit_first_span"]) == 11
        assert int(outer_planned.annotations["tl_z3_pipeline_audit_last_iter_offset"]) == 11

        planned_root = next(
            node
            for node in collect_nodes(planned["main"].body, tirx.SBlock)
            if node.name_hint == "tilelang_root"
        )
        selected = planned_root.annotations["tl.cuda_z3_buffer_versions_selected"]
        assert int(selected[shared.data]) == 2

        lowered = lower_factory()(planned)
        inner_lowered = next(loop for loop in _loops(lowered) if int(loop.extent) == 4)
        schedule_units = [
            node
            for node in collect_nodes(lowered["main"].body, tirx.AttrStmt)
            if node.attr_key == "tl.schedule_unit"
        ]
        assert schedule_units == []

        # Physical order is producer(stage 0), then consumer(stage 1). Since
        # the restored list is already physical, stock order is dense identity.
        assert _ints(inner_lowered.annotations["tl_pipeline_order"]) == [0, 1]
        assert _ints(inner_lowered.annotations["tl_pipeline_stage"]) == [0, 1]
        assert int(inner_lowered.annotations["num_stages"]) == 2
        versions = inner_lowered.annotations["tl_pipeline_buffer_versions"]
        assert int(versions[shared.data]) == 2
        assert bool(inner_lowered.annotations["tl.cuda_z3_recursive_prepared"]) is True

        prepared_root = next(
            node
            for node in collect_nodes(lowered["main"].body, tirx.SBlock)
            if node.name_hint == "tilelang_root"
        )
        clocks = prepared_root.annotations["tl.cuda_z3_prepared_clock_map"]
        assert shared.data in clocks
        clock_vars = collect_nodes(clocks[shared.data], tirx.Var)
        assert any(var.same_as(inner) for var in clock_vars)
        assert any(var.same_as(outer) for var in clock_vars)
        epoch_plan = prepared_root.annotations["tl.cuda_z3_prepared_buffer_epochs"]
        assert len(epoch_plan) == 1
        assert str(epoch_plan[0]["clock"]) == "iteration"
        assert int(epoch_plan[0]["versions"]) == 2

        # Run the complete restored tree through stock planning. Prepared
        # markers must survive on both Controls and suppress legacy fallback.
        stock = tl.transform.PipelinePlanning()(lowered)
        outer_stock, inner_stock = sorted(
            _loops(stock), key=lambda loop: int(loop.extent)
        )
        assert bool(outer_stock.annotations["tl.cuda_z3_recursive_prepared"]) is True
        assert bool(inner_stock.annotations["tl.cuda_z3_recursive_prepared"]) is True
        assert _ints(outer_stock.annotations["software_pipeline_order"]) == [0]
        assert _ints(outer_stock.annotations["software_pipeline_stage"]) == [0]
        assert _ints(inner_stock.annotations["software_pipeline_order"]) == [0, 1]
        assert _ints(inner_stock.annotations["software_pipeline_stage"]) == [0, 1]
        assert "software_pipeline_async_stages" not in outer_stock.annotations
        assert "software_pipeline_async_stages" not in inner_stock.annotations

        # Injection must consume the root's prepared nested clock. Retaining
        # the outer Var in every dynamic version index rules out immediate-i
        # modulo fallback and proves the map reached PipelineBodyRewriter.
        injected = tl.transform.InjectSoftwarePipeline()(stock)
        injected_root = next(
            node
            for node in collect_nodes(injected["main"].body, tirx.SBlock)
            if node.name_hint == "tilelang_root"
        )
        injected_shared = next(
            buffer
            for buffer in injected_root.alloc_buffers
            if buffer.data.same_as(shared.data)
        )
        assert [int(dim) for dim in injected_shared.shape] == [2, 8]
        shared_accesses = [
            node
            for node in (
                collect_nodes(injected["main"].body, tirx.BufferLoad)
                + collect_nodes(injected["main"].body, tirx.BufferStore)
            )
            if node.buffer.data.same_as(shared.data) and len(node.indices) == 2
        ]
        assert shared_accesses
        for access in shared_accesses:
            version_vars = collect_nodes(access.indices[0], tirx.Var)
            assert any(var.same_as(outer) for var in version_vars), access
    finally:
        if tvm_ffi.get_global_func(_SOLVER, allow_missing=True) is not None:
            tvm_ffi.remove_global_func(_SOLVER)
        tvm_ffi.register_global_func(_SOLVER, f=previous, override=True)
