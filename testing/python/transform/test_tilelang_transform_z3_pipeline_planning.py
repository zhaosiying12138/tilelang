"""Host-only tests for the opt-in CUDA Z3 loop pipeline adapter."""

from __future__ import annotations

import pytest
import tvm_ffi

import tilelang as tl
from tilelang import tvm
import tilelang.language as T  # noqa: F401  # Keep the language surface loaded.
from tilelang.language.annotations import z3_pipeline_annotations
from tilelang.transform.pass_config import normalize_pass_configs
import tilelang.testing
from tilelang.testing.ir import collect_calls, collect_nodes
from tvm.tirx.stmt_functor import post_order_visit


_FFI_NAME = "tl.cuda.transform.z3_schedule_loop_python"
_CONFIG = "tl.enable_z3_pipeline_planning"
_AUTO_WS_CONFIG = "tl.enable_auto_warp_specialization"
_SM80 = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})


@pytest.fixture(autouse=True)
def _restore_solver_ffi():
    previous = tvm_ffi.get_global_func(_FFI_NAME, allow_missing=True)
    yield
    if tvm_ffi.get_global_func(_FFI_NAME, allow_missing=True) is not None:
        tvm_ffi.remove_global_func(_FFI_NAME)
    if previous is not None:
        tvm_ffi.register_global_func(_FFI_NAME, f=previous, override=True)


def _install_solver(result):
    calls = []

    def solver(
        num_stages,
        latencies,
        iis,
        resource_flags,
        data_deps,
        resource_deps,
        owner_exclusion_deps,
        pipe_order_deps,
        buffer_sizes,
        memory_groups,
        stage_order_deps,
        enable_offset,
        manual_stages,
        manual_schedule,
        resource_bits,
        seed,
        recalculate_buffer_versions,
        verbose,
        debug_dir,
    ):
        call = {
            "num_stages": num_stages,
            "latencies": latencies,
            "iis": iis,
            "resource_flags": resource_flags,
            "data_deps": data_deps,
            "resource_deps": resource_deps,
            "owner_exclusion_deps": owner_exclusion_deps,
            "pipe_order_deps": pipe_order_deps,
            "buffer_sizes": buffer_sizes,
            "memory_groups": memory_groups,
            "stage_order_deps": stage_order_deps,
            "enable_offset": enable_offset,
            "manual_stages": manual_stages,
            "manual_schedule": manual_schedule,
            "resource_bits": resource_bits,
            "seed": seed,
            "recalculate_buffer_versions": recalculate_buffer_versions,
            "verbose": verbose,
            "debug_dir": debug_dir,
        }
        calls.append(call)
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            return result(call)
        return result

    tvm_ffi.register_global_func(_FFI_NAME, f=solver, override=True)
    return calls


def _annotations(**overrides):
    arguments = {
        "latencies": [1, 1, 1, 1],
        "iis": [1, 1, 1, 1],
        "resource_flags": [0, 0, 0, 0],
    }
    arguments.update(overrides)
    return z3_pipeline_annotations(**arguments)


def _make_loop(annotations, *, num_stages=4, order=None, stage=None, replayable_bind=False, dependent=False):
    tir = tvm.tirx
    buffers = [tir.decl_buffer((8,), "int32", name=name) for name in ("A", "B", "C", "D")]
    if callable(annotations):
        annotations = annotations(buffers)
    annotations = dict(annotations)
    annotations["num_stages"] = tir.IntImm("int32", num_stages)
    if order is not None:
        annotations["tl_pipeline_order"] = order
    if stage is not None:
        annotations["tl_pipeline_stage"] = stage

    k = tir.Var("k", "int32")
    value = k
    statements = []
    if replayable_bind:
        value = tir.Var("replayed", "int32")
        statements.append(tir.Bind(value, k + 1))
    for index, buffer in enumerate(buffers):
        destination = buffers[0] if dependent else buffer
        statements.append(tir.BufferStore(destination, value + index, [k]))
    loop = tir.For(
        k,
        0,
        4,
        tir.ForKind.SERIAL,
        tir.SeqStmt(statements),
        annotations=annotations,
    )
    func = tir.PrimFunc(
        [buffer.data for buffer in buffers],
        loop,
        buffer_map={buffer.data: buffer for buffer in buffers},
    )
    return func.with_attr("global_symbol", "main"), buffers


def _make_statement_func(statements, buffers, annotations, loop_var, *, num_stages=4):
    tir = tvm.tirx
    annotations = dict(annotations)
    annotations["num_stages"] = tir.IntImm("int32", num_stages)
    loop = tir.For(
        loop_var,
        0,
        4,
        tir.ForKind.SERIAL,
        tir.SeqStmt(statements) if len(statements) > 1 else statements[0],
        annotations=annotations,
    )
    func = tir.PrimFunc(
        [buffer.data for buffer in buffers],
        loop,
        buffer_map={buffer.data: buffer for buffer in buffers},
    )
    return func.with_attr("global_symbol", "main")


def _clone_plan(loop):
    key = "tl_pipeline_scalar_clone_plan"
    return list(loop.annotations[key]) if key in loop.annotations else []


def _make_buffer_bind_func(annotations):
    tir = tvm.tirx
    a = tir.decl_buffer((8,), "int32", name="clone_A")
    b = tir.decl_buffer((8,), "int32", name="clone_B")
    k = tir.Var("k", "int32")
    value = tir.Var("cloned_value", "int32")
    statements = [
        tir.BufferStore(a, 1, [0]),
        tir.Bind(value, tir.BufferLoad(a, [0]) + 1),
        tir.BufferStore(b, value, [k]),
    ]
    if callable(annotations):
        annotations = annotations((a, b))
    return _make_statement_func(statements, [a, b], annotations, k, num_stages=3), (a, b)


def _bound_module(func):
    mod = tvm.IRModule.from_expr(func)
    return tvm.tirx.transform.BindTarget(_SM80)(mod)


def _run_z3(func, *, config=None):
    mod = _bound_module(func)
    pass_config = {_CONFIG: True} if config is None else config
    with tvm.transform.PassContext(config=pass_config):
        return tl.transform.Z3PipelinePlanning()(mod)


def _project_and_inject(planned):
    with tvm.transform.PassContext(config={_CONFIG: True}):
        stock = tl.transform.PipelinePlanning()(planned)
        injected = tl.transform.InjectSoftwarePipeline()(stock)
    return stock, injected


def _find_steady_pipeline_loop(func):
    loops = [
        loop
        for loop in collect_nodes(func, tvm.tirx.For)
        if "tl_pipelined_num_stages" in loop.annotations
    ]
    assert len(loops) == 1
    return loops[0]


def _node_position(nodes, target):
    matches = [index for index, node in enumerate(nodes) if node.same_as(target)]
    assert len(matches) == 1
    return matches[0]


def _binding_for_var(bindings, var):
    matches = [binding for binding in bindings if binding.var.same_as(var)]
    assert len(matches) == 1
    return matches[0]


def _expr_var(expr, name):
    matches = [var for var in collect_nodes(expr, tvm.tirx.Var) if str(var.name) == name]
    assert len(matches) == 1
    return matches[0]


def _assert_clone_loads_reach_async_metadata(root, clone_binds):
    clone_loads = [
        load
        for binding in clone_binds
        for load in collect_nodes(binding.value, tvm.tirx.BufferLoad)
    ]
    assert clone_loads
    # The materializer checks the clone SBlock reads directly before emission.
    # These fixtures then unwrap the Bind to preserve scalar scope. Their only
    # synchronous cross-stage readers are the clones, so the emitted wait proves
    # that stock async analysis consumed those rebuilt read regions.
    assert collect_calls(root, op="tirx.ptx_wait_group")


def _find_loop(func):
    loops = []

    def visit(node):
        if isinstance(node, tvm.tirx.For):
            loops.append(node)

    post_order_visit(func.body, visit)
    assert len(loops) == 1
    return loops[0]


def _ints(values):
    return [int(value) for value in values]


def _nested_ints(values):
    return [_ints(group) for group in values]


def _body_store_names(loop):
    statements = list(loop.body.seq) if isinstance(loop.body, tvm.tirx.SeqStmt) else [loop.body]
    return [str(statement.buffer.name) for statement in statements]


def test_complete_problem_projection_and_stock_pipeline_planning():
    annotations = _annotations(
        latencies=[1, 2, 3, 4],
        iis=[2, 3, 4, 5],
        resource_flags=[1, 2, 4, 8],
        data_deps=[(0, 1, 0, 1)],
        resource_deps=[(0, 1)],
        pipe_order_deps=[(1, 2)],
        resource_bits=7,
        enable_offset=True,
        seed=17,
        recalculate_buffer_versions=False,
        verbose=True,
        debug_dir="mock-z3-debug",
    )
    func, _ = _make_loop(annotations, num_stages=3)
    calls = _install_solver(([0, 2, 3, 4], [], 8))

    mod = _run_z3(func)
    assert len(calls) == 1
    call = calls[0]
    assert int(call["num_stages"]) == 3
    assert _ints(call["latencies"]) == [1, 2, 3, 4]
    assert _ints(call["iis"]) == [2, 3, 4, 5]
    assert _ints(call["resource_flags"]) == [1, 2, 4, 8]
    assert _nested_ints(call["data_deps"]) == [[0, 1, 0, 1]]
    assert _nested_ints(call["resource_deps"]) == [[0, 1]]
    assert _nested_ints(call["owner_exclusion_deps"]) == []
    assert _nested_ints(call["pipe_order_deps"]) == [[1, 2]]
    assert _nested_ints(call["stage_order_deps"]) == []
    assert _ints(call["buffer_sizes"]) == []
    assert _nested_ints(call["memory_groups"]) == []
    assert call["enable_offset"] is True
    assert _ints(call["manual_stages"]) == []
    assert call["manual_schedule"] is False
    assert int(call["resource_bits"]) == 7
    assert int(call["seed"]) == 17
    assert call["recalculate_buffer_versions"] is False
    assert call["verbose"] is True
    assert call["debug_dir"] == "mock-z3-debug"

    loop = _find_loop(mod["main"])
    assert "tl_z3_pipeline" not in loop.annotations
    assert _ints(loop.annotations["tl_pipeline_order"]) == [0, 1, 2, 3]
    assert _ints(loop.annotations["tl_pipeline_stage"]) == [0, 0, 0, 0]
    assert int(loop.annotations["tl_z3_pipeline_audit_ii"]) == 8
    assert _ints(loop.annotations["tl_z3_pipeline_audit_starts"]) == [0, 2, 3, 4]

    with tvm.transform.PassContext(config={_CONFIG: True}):
        projected = tl.transform.PipelinePlanning()(mod)
    projected_loop = _find_loop(projected["main"])
    assert "tl_pipeline_order" not in projected_loop.annotations
    assert "tl_pipeline_stage" not in projected_loop.annotations
    assert _ints(projected_loop.annotations["software_pipeline_order"]) == [0, 1, 2, 3]
    assert _ints(projected_loop.annotations["software_pipeline_stage"]) == [0, 0, 0, 0]
    assert int(projected_loop.annotations["tl_z3_pipeline_audit_ii"]) == 8


def test_output_depth_covers_solver_stages_and_buffer_versions():
    def annotations(buffers):
        return _annotations(
            buffer_sizes=[32],
            buffer_vars=[buffers[0].data],
            memory_groups=[[128, 0]],
        )

    func, buffers = _make_loop(annotations, num_stages=2)
    _install_solver(([1, 2, 6, 8], [3], 2))

    loop = _find_loop(_run_z3(func)["main"])
    assert int(loop.annotations["num_stages"]) == 5
    versions = loop.annotations["tl_pipeline_buffer_versions"]
    assert int(versions[buffers[0].data]) == 3


def test_cross_stage_tie_uses_larger_stage_then_source_index_and_inverse_order():
    func, _ = _make_loop(_annotations())
    _install_solver(([0, 50, 151, 152], [], 152))

    loop = _find_loop(_run_z3(func)["main"])
    # Residues are [0, 50, 151, 0]. Physical order is [3, 0, 1, 2],
    # so the annotation stores its inverse, indexed by source statement.
    assert _ints(loop.annotations["tl_pipeline_stage"]) == [0, 0, 0, 1]
    assert _ints(loop.annotations["tl_pipeline_order"]) == [1, 2, 3, 0]
    assert _body_store_names(loop) == ["A", "B", "C", "D"]
    assert _ints(loop.annotations["tl_z3_pipeline_audit_starts"]) == [0, 50, 151, 152]


def test_audit_ii_and_starts_preserve_int64_values():
    large_ii = 3_000_000_000
    func, _ = _make_loop(_annotations())
    _install_solver(([0, large_ii, 0, 0], [], large_ii))

    loop = _find_loop(_run_z3(func)["main"])
    assert _ints(loop.annotations["tl_pipeline_stage"]) == [0, 1, 0, 0]
    assert _ints(loop.annotations["tl_pipeline_order"]) == [1, 0, 2, 3]
    assert int(loop.annotations["tl_z3_pipeline_audit_ii"]) == large_ii
    assert _ints(loop.annotations["tl_z3_pipeline_audit_starts"]) == [0, large_ii, 0, 0]


def test_manual_offset_and_minimum_stage_normalization():
    annotations = _annotations(
        enable_offset=True,
        manual_schedule=True,
        manual_stages=[1, 1, 1, 3],
        seed=23,
    )
    func, _ = _make_loop(annotations)
    calls = _install_solver(([152, 202, 303, 456], [], 152))

    loop = _find_loop(_run_z3(func)["main"])
    assert calls[0]["manual_schedule"] is True
    assert calls[0]["enable_offset"] is True
    assert _ints(calls[0]["manual_stages"]) == [1, 1, 1, 3]
    assert int(calls[0]["seed"]) == 23
    assert _ints(loop.annotations["tl_pipeline_stage"]) == [0, 0, 0, 2]
    assert _ints(loop.annotations["tl_pipeline_order"]) == [1, 2, 3, 0]


def test_flat_non_identity_schedule_reorders_statements_and_uses_identity_order():
    func, _ = _make_loop(_annotations())
    _install_solver(([30, 0, 20, 10], [], 100))

    loop = _find_loop(_run_z3(func)["main"])
    assert _body_store_names(loop) == ["B", "D", "C", "A"]
    assert _ints(loop.annotations["tl_pipeline_stage"]) == [0, 0, 0, 0]
    assert _ints(loop.annotations["tl_pipeline_order"]) == [0, 1, 2, 3]
    assert _ints(loop.annotations["tl_z3_pipeline_audit_starts"]) == [30, 0, 20, 10]


def test_flat_non_identity_schedule_rejects_buffer_dependencies():
    func, _ = _make_loop(_annotations(), dependent=True)
    _install_solver(([30, 0, 20, 10], [], 100))

    with pytest.raises(Exception, match="unsafe flat reorder"):
        _run_z3(func)


def test_buffer_versions_map_uses_ordered_buffer_data_vars_and_intimm():
    def annotations(buffers):
        return _annotations(
            buffer_sizes=[16, 32],
            buffer_vars=buffers[:2],
            memory_groups=[[128, 0, 1]],
        )

    func, buffers = _make_loop(annotations)
    calls = _install_solver(([0, 0, 0, 0], [2, 3], 8))

    loop = _find_loop(_run_z3(func)["main"])
    versions = loop.annotations["tl_pipeline_buffer_versions"]
    assert len(versions) == 2
    assert int(versions[buffers[0].data]) == 2
    assert int(versions[buffers[1].data]) == 3
    assert all(isinstance(value, tvm.tirx.IntImm) for _, value in versions.items())
    assert _ints(calls[0]["buffer_sizes"]) == [16, 32]
    assert _nested_ints(calls[0]["memory_groups"]) == [[128, 0, 1]]


def test_replayable_bind_is_not_a_solver_task():
    func, _ = _make_loop(_annotations(), replayable_bind=True)
    calls = _install_solver(([0, 0, 0, 0], [], 8))

    loop = _find_loop(_run_z3(func)["main"])
    assert len(calls) == 1
    assert _ints(loop.annotations["tl_pipeline_order"]) == [0, 1, 2, 3]


def test_task_count_must_match_scheduled_body_statements():
    func, _ = _make_loop(_annotations(latencies=[1, 1, 1], iis=[1, 1, 1], resource_flags=[0, 0, 0]))
    calls = _install_solver(([0, 0, 0, 0], [], 8))

    with pytest.raises(Exception, match="does not match body scheduled statement count"):
        _run_z3(func)
    assert calls == []


def test_unexpressible_dependencies_fail_before_solver():
    func, _ = _make_loop(_annotations(owner_exclusion_deps=[(0, 1, 1, 1)]))
    calls = _install_solver(([0, 0, 0, 0], [], 8))

    with pytest.raises(Exception, match="owner_exclusion_deps"):
        _run_z3(func)
    assert calls == []


def test_stage_order_is_forwarded_and_checked_after_projection():
    func, _ = _make_loop(_annotations(stage_order_deps=[(0, 1)]))
    calls = _install_solver(([0, 2, 0, 0], [], 2))

    loop = _find_loop(_run_z3(func)["main"])
    assert _nested_ints(calls[0]["stage_order_deps"]) == [[0, 1]]
    assert _ints(loop.annotations["tl_pipeline_stage"]) == [0, 1, 0, 0]


def test_stage_order_rejects_a_solver_result_that_violates_it():
    func, _ = _make_loop(_annotations(stage_order_deps=[(0, 1)]))
    _install_solver(([2, 0, 0, 0], [], 2))

    with pytest.raises(Exception, match="violates stage_order_deps"):
        _run_z3(func)


def test_stage_order_materializes_scheduled_bind_clones():
    tir = tvm.tirx
    a = tir.decl_buffer((8,), "int32", name="bind_A")
    b = tir.decl_buffer((8,), "int32", name="bind_B")
    k = tir.Var("k", "int32")
    value = tir.Var("scheduled_value", "int32")
    statements = [
        tir.BufferStore(a, 1, [k]),
        tir.Bind(value, tir.BufferLoad(a, [k]) + 1),
        tir.BufferStore(b, value, [k]),
    ]
    annotations = _annotations(latencies=[1, 1, 1], iis=[1, 1, 1], resource_flags=[0, 0, 0], stage_order_deps=[(1, 2)])
    annotations["num_stages"] = tir.IntImm("int32", 2)
    loop = tir.For(k, 0, 4, tir.ForKind.SERIAL, tir.SeqStmt(statements), annotations=annotations)
    func = tir.PrimFunc([a.data, b.data], loop, buffer_map={a.data: a, b.data: b}).with_attr("global_symbol", "main")
    def solver_result(call):
        starts_by_tasks = {3: [1, 2, 5], 4: [1, 2, 5, 4]}
        return starts_by_tasks[len(call["latencies"])], [], 4

    calls = _install_solver(solver_result)
    planned = _run_z3(func)
    loop = _find_loop(planned["main"])
    assert len(calls) == 2
    plan = _clone_plan(loop)
    assert len(plan) == 1
    assert int(plan[0]["definition_task"]) == 1
    assert int(plan[0]["consumer_task"]) == 2
    assert _ints(loop.annotations["tl_z3_pipeline_audit_starts"]) == [1, 2, 5]

    # Planner support is complete, but fresh Vars and consumer rewriting still
    # have to be materialized by InjectSoftwarePipeline. Keep that integration
    # contract visible rather than silently treating metadata as cloned IR.
    with tvm.transform.PassContext(config={_CONFIG: True}):
        stock = tl.transform.PipelinePlanning()(planned)
        stock_loop = _find_loop(stock["main"])
        assert _ints(stock_loop.annotations["software_pipeline_replayable_scalar_binds"]) == [0, 0, 0]
        injected = tl.transform.InjectSoftwarePipeline()(stock)
    bindings = []
    post_order_visit(
        injected["main"].body,
        lambda node: bindings.append(node) if isinstance(node, tvm.tirx.Bind) else None,
    )
    assert len(bindings) >= 2
    assert not bindings[0].var.same_as(bindings[1].var)
    assert any(not binding.var.same_as(value) for binding in bindings)


def test_clone_order_survives_independent_post_consumer_overwrite():
    tir = tvm.tirx
    a = tir.decl_buffer((8,), "int32", name="ordered_A")
    b = tir.decl_buffer((8,), "int32", name="ordered_B")
    k = tir.Var("k", "int32")
    value = tir.Var("ordered_value", "int32")
    statements = [
        tir.BufferStore(a, 1, [k]),
        tir.Bind(value, tir.BufferLoad(a, [k]) + 1),
        tir.BufferStore(b, value, [k]),
        # This independent task follows the consumer in the original body but
        # is scheduled between the clone and consumer in the augmented order.
        tir.BufferStore(a, 99, [k]),
    ]
    annotations = _annotations(latencies=[1] * 4, iis=[1] * 4, resource_flags=[0] * 4)
    func = _make_statement_func(statements, [a, b], annotations, k)

    def solver_result(call):
        starts = {
            # The pre-clone scalar WAR edge (2, 1, 1, 1) requires
            # start(consumer) <= start(definition) + II - latency.
            4: [0, 1, 16, 17],
            5: [0, 1, 20, 19, 18],
        }
        return starts[len(call["latencies"])], [], 16

    calls = _install_solver(solver_result)
    planned = _run_z3(func)
    loop = _find_loop(planned["main"])
    assert len(calls) == 2
    plan = _clone_plan(loop)
    assert len(plan) == 1
    clone_task = 4
    assert [clone_task, 3, 0, 1] in _nested_ints(calls[1]["data_deps"])
    assert int(plan[0]["schedule_position"]) == 2
    assert int(plan[0]["consumer_schedule_position"]) == 4
    assert _ints(loop.annotations["tl_pipeline_order"]) == [0, 1, 4, 3]

    _, injected = _project_and_inject(planned)
    steady = _find_steady_pipeline_loop(injected["main"])
    bindings = collect_nodes(steady.body, tvm.tirx.Bind)
    stores = collect_nodes(steady.body, tvm.tirx.BufferStore)
    (consume,) = [store for store in stores if store.buffer.data.same_as(b.data)]
    (overwrite,) = [
        store
        for store in stores
        if store.buffer.data.same_as(a.data)
        and isinstance(store.value, tvm.tirx.IntImm)
        and int(store.value) == 99
    ]
    assert isinstance(consume.value, tvm.tirx.Var)
    clone = _binding_for_var(bindings, consume.value)
    assert not clone.var.same_as(value)
    _assert_clone_loads_reach_async_metadata(steady, [clone])

    events = []
    post_order_visit(
        steady.body,
        lambda node: events.append(node)
        if isinstance(node, (tvm.tirx.Bind, tvm.tirx.BufferStore))
        else None,
    )
    # The old consumer-prefix materialization emitted overwrite -> clone ->
    # consumer here, reversing the solver's clone-to-overwrite WAR edge.
    assert _node_position(events, clone) < _node_position(events, overwrite)
    assert _node_position(events, overwrite) < _node_position(events, consume)


def test_flat_declarations_are_mapped_before_clone_materialization():
    tir = tvm.tirx
    scratch = tir.decl_buffer((8,), "int32", name="flat_scratch")
    declared_view = tir.decl_buffer((8,), "int32", name="flat_declared_view")
    b = tir.decl_buffer((8,), "int32", name="flat_B")
    k = tir.Var("k", "int32")
    value = tir.Var("flat_value", "int32")
    statements = [
        tir.AllocBuffer(scratch),
        tir.DeclBuffer(declared_view),
        tir.BufferStore(scratch, 1, [k]),
        tir.Bind(value, tir.BufferLoad(scratch, [k]) + 1),
        tir.BufferStore(b, value, [k]),
    ]
    annotations = _annotations(latencies=[1] * 3, iis=[1] * 3, resource_flags=[0] * 3)
    annotations["num_stages"] = tir.IntImm("int32", 2)
    loop = tir.For(k, 0, 4, tir.ForKind.SERIAL, tir.SeqStmt(statements), annotations=annotations)
    func = tir.PrimFunc([b.data], loop, buffer_map={b.data: b}).with_attr("global_symbol", "main")

    def solver_result(call):
        starts = {3: [1, 2, 5], 4: [1, 2, 5, 4]}
        return starts[len(call["latencies"])], [], 4

    calls = _install_solver(solver_result)
    planned = _run_z3(func)
    loop = _find_loop(planned["main"])
    assert len(calls) == 2
    assert _ints(loop.annotations["tl_pipeline_scalar_clone_original_flat_indices"]) == [2, 3, 4]
    plan = _clone_plan(loop)
    assert len(plan) == 1
    assert int(plan[0]["definition_stmt"]) == 3
    assert int(plan[0]["consumer_stmt"]) == 4

    _, injected = _project_and_inject(planned)
    steady = _find_steady_pipeline_loop(injected["main"])
    bindings = collect_nodes(steady.body, tvm.tirx.Bind)
    (consume,) = [
        store
        for store in collect_nodes(steady.body, tvm.tirx.BufferStore)
        if store.buffer.data.same_as(b.data)
    ]
    clone = _binding_for_var(bindings, consume.value)
    assert not clone.var.same_as(value)
    _assert_clone_loads_reach_async_metadata(steady, [clone])


def test_zero_latency_clone_tie_uses_topological_order_and_virtual_reads():
    def annotations(buffers):
        return _annotations(
            latencies=[0, 0, 0],
            iis=[1, 1, 1],
            resource_flags=[0, 0, 0],
            buffer_sizes=[4],
            buffer_vars=[buffers[0].data],
        )

    func, _ = _make_buffer_bind_func(annotations)

    def solver_result(call):
        starts = {3: [0, 0, 4], 4: [0, 0, 4, 4]}
        return starts[len(call["latencies"])], [1], 4

    calls = _install_solver(solver_result)
    loop = _find_loop(_run_z3(func)["main"])
    assert len(calls) == 2
    clone_task = 3
    assert any(
        dep[0] == 0 and dep[1] == clone_task and dep[2] == -1
        for dep in _nested_ints(calls[1]["data_deps"])
    ), "consumer-site clone reads must retain the buffer-version distance"
    assert any(
        dep[0] == clone_task and dep[1] == 0 and dep[2] == -1
        for dep in _nested_ints(calls[1]["data_deps"])
    ), "the clone read must also retain its loop-carried WAR hand-off"

    plan = _clone_plan(loop)
    assert len(plan) == 1
    assert len(plan[0]["virtual_reads"]) == 1
    # Augmented source order would put task 2 before appended task 3. The
    # zero-latency scalar edge requires the opposite topological order.
    assert int(plan[0]["schedule_position"]) < int(plan[0]["consumer_schedule_position"])
    assert len(loop.annotations["tl_pipeline_order"]) == 3
    assert _ints(loop.annotations["tl_z3_pipeline_audit_starts"]) == [0, 0, 4]


def test_chained_lets_converge_to_a_consumer_site_clone_chain():
    tir = tvm.tirx
    a = tir.decl_buffer((8,), "int32", name="chain_A")
    b = tir.decl_buffer((8,), "int32", name="chain_B")
    k = tir.Var("k", "int32")
    x = tir.Var("chain_x", "int32")
    y = tir.Var("chain_y", "int32")
    statements = [
        tir.BufferStore(a, 1, [k]),
        tir.Bind(x, tir.BufferLoad(a, [k]) + 1),
        tir.Bind(y, x + 1),
        tir.BufferStore(b, y, [k]),
    ]
    annotations = _annotations(latencies=[1] * 4, iis=[1] * 4, resource_flags=[0] * 4)
    func = _make_statement_func(statements, [a, b], annotations, k)

    def solver_result(call):
        starts = {
            4: [1, 2, 5, 20],
            5: [1, 2, 5, 20, 16],
            6: [1, 2, 5, 20, 17, 16],
        }
        return starts[len(call["latencies"])], [], 16

    calls = _install_solver(solver_result)
    planned = _run_z3(func)
    loop = _find_loop(planned["main"])
    assert len(calls) == 3
    assert int(loop.annotations["tl_z3_pipeline_audit_clone_rounds"]) == 3
    assert int(loop.annotations["tl_pipeline_scalar_clone_original_task_count"]) == 4
    assert _ints(loop.annotations["tl_pipeline_scalar_clone_original_flat_indices"]) == [0, 1, 2, 3]
    assert _ints(loop.annotations["tl_z3_pipeline_audit_starts"]) == [1, 2, 5, 20]

    plan = _clone_plan(loop)
    assert len(plan) == 2
    by_definition = {int(entry["definition_task"]): entry for entry in plan}
    assert set(by_definition) == {1, 2}
    assert all(int(entry["consumer_task"]) == 3 for entry in plan)
    assert all(int(entry["stage"]) == 1 for entry in plan)
    assert _ints(by_definition[2]["scalar_inputs"]) == [5]
    assert _ints(by_definition[1]["scalar_inputs"]) == []
    assert int(by_definition[1]["schedule_position"]) < int(by_definition[2]["schedule_position"])

    _, injected = _project_and_inject(planned)
    steady = _find_steady_pipeline_loop(injected["main"])
    bindings = collect_nodes(steady.body, tvm.tirx.Bind)
    (consume,) = [
        store
        for store in collect_nodes(steady.body, tvm.tirx.BufferStore)
        if store.buffer.data.same_as(b.data)
    ]
    clone_y = _binding_for_var(bindings, consume.value)
    clone_x = _binding_for_var(bindings, _expr_var(clone_y.value, "chain_x"))
    assert not clone_x.var.same_as(x)
    assert not clone_y.var.same_as(y)
    _assert_clone_loads_reach_async_metadata(steady, [clone_x])

    events = []
    post_order_visit(
        steady.body,
        lambda node: events.append(node)
        if isinstance(node, (tvm.tirx.Bind, tvm.tirx.BufferStore))
        else None,
    )
    assert _node_position(events, clone_x) < _node_position(events, clone_y)
    assert _node_position(events, clone_y) < _node_position(events, consume)


def test_diamond_lets_share_one_clone_per_consumer_chain():
    tir = tvm.tirx
    a = tir.decl_buffer((8,), "int32", name="diamond_A")
    b = tir.decl_buffer((8,), "int32", name="diamond_B")
    k = tir.Var("k", "int32")
    x = tir.Var("diamond_x", "int32")
    y = tir.Var("diamond_y", "int32")
    z = tir.Var("diamond_z", "int32")
    w = tir.Var("diamond_w", "int32")
    statements = [
        tir.BufferStore(a, 1, [k]),
        tir.Bind(x, tir.BufferLoad(a, [k]) + 1),
        tir.Bind(y, x + 1),
        tir.Bind(z, x + 2),
        tir.Bind(w, y + z),
        tir.BufferStore(b, w, [k]),
    ]
    annotations = _annotations(latencies=[1] * 6, iis=[1] * 6, resource_flags=[0] * 6)
    func = _make_statement_func(statements, [a, b], annotations, k)

    def solver_result(call):
        starts = {
            6: [1, 4, 8, 12, 16, 65],
            7: [1, 4, 8, 12, 16, 65, 64],
            8: [1, 4, 8, 12, 16, 66, 65, 64, 64],
            9: [1, 4, 8, 12, 16, 67, 66, 65, 65],
            10: [1, 4, 8, 12, 16, 67, 66, 65, 65, 64],
        }
        return starts[len(call["latencies"])], [], 64

    calls = _install_solver(solver_result)
    planned = _run_z3(func)
    loop = _find_loop(planned["main"])
    assert len(calls) == 4
    plan = _clone_plan(loop)
    assert len(plan) == 4
    by_definition = {int(entry["definition_task"]): entry for entry in plan}
    assert set(by_definition) == {1, 2, 3, 4}
    assert sum(int(entry["definition_task"]) == 1 for entry in plan) == 1
    assert all(int(entry["consumer_task"]) == 5 for entry in plan)
    assert set(_ints(by_definition[4]["scalar_inputs"])) == {7, 8}
    assert _ints(by_definition[2]["scalar_inputs"]) == [9]
    assert _ints(by_definition[3]["scalar_inputs"]) == [9]
    assert int(by_definition[1]["schedule_position"]) < int(by_definition[2]["schedule_position"])
    assert int(by_definition[1]["schedule_position"]) < int(by_definition[3]["schedule_position"])

    _, injected = _project_and_inject(planned)
    steady = _find_steady_pipeline_loop(injected["main"])
    bindings = collect_nodes(steady.body, tvm.tirx.Bind)
    (consume,) = [
        store
        for store in collect_nodes(steady.body, tvm.tirx.BufferStore)
        if store.buffer.data.same_as(b.data)
    ]
    clone_w = _binding_for_var(bindings, consume.value)
    clone_y = _binding_for_var(bindings, _expr_var(clone_w.value, "diamond_y"))
    clone_z = _binding_for_var(bindings, _expr_var(clone_w.value, "diamond_z"))
    x_from_y = _expr_var(clone_y.value, "diamond_x")
    x_from_z = _expr_var(clone_z.value, "diamond_x")
    assert x_from_y.same_as(x_from_z)
    clone_x = _binding_for_var(bindings, x_from_y)
    for clone, original in ((clone_x, x), (clone_y, y), (clone_z, z), (clone_w, w)):
        assert not clone.var.same_as(original)
    # Both diamond branches reference the one fresh x definition; no second
    # clone of the shared subtree is introduced during materialization.
    assert _binding_for_var(bindings, x_from_z).var.same_as(clone_x.var)
    _assert_clone_loads_reach_async_metadata(steady, [clone_x])

    events = []
    post_order_visit(
        steady.body,
        lambda node: events.append(node)
        if isinstance(node, (tvm.tirx.Bind, tvm.tirx.BufferStore))
        else None,
    )
    x_position = _node_position(events, clone_x)
    y_position = _node_position(events, clone_y)
    z_position = _node_position(events, clone_z)
    w_position = _node_position(events, clone_w)
    assert x_position < y_position and x_position < z_position
    assert y_position < w_position and z_position < w_position
    assert w_position < _node_position(events, consume)


@pytest.mark.parametrize(
    ("recalculate", "capacity", "error"),
    [(True, 8, None), (False, 8, "under-versioned buffer"), (True, 7, "memory-group capacity")],
)
def test_clone_augmentation_recomputes_versions_and_rechecks_capacity(recalculate, capacity, error):
    def annotations(buffers):
        return _annotations(
            latencies=[30, 2, 1],
            iis=[1, 1, 1],
            resource_flags=[0, 0, 0],
            buffer_sizes=[4],
            buffer_vars=[buffers[0].data],
            memory_groups=[[capacity, 0]],
            recalculate_buffer_versions=recalculate,
        )

    func, buffers = _make_buffer_bind_func(annotations)

    def solver_result(call):
        starts = {3: [1, 3, 34], 4: [1, 3, 34, 32]}
        # Deliberately return a prefix-only/stale version. Recompute mode must
        # repair it after seeing the clone WAR edge; strict mode must reject it.
        return starts[len(call["latencies"])], [1], 32

    calls = _install_solver(solver_result)
    if error is not None:
        with pytest.raises(Exception, match=error):
            _run_z3(func)
        assert len(calls) == 2
        return
    loop = _find_loop(_run_z3(func)["main"])
    assert len(calls) == 2
    assert int(loop.annotations["tl_pipeline_buffer_versions"][buffers[0].data]) == 2
    assert len(_clone_plan(loop)) == 1


def test_clone_plan_round_limit_reports_nonconvergence():
    annotations = _annotations(latencies=[0, 0, 0], iis=[1, 1, 1], resource_flags=[0, 0, 0])
    annotations["tl_z3_pipeline_max_clone_rounds"] = tvm.tirx.IntImm("int64", 1)
    func, _ = _make_buffer_bind_func(annotations)
    calls = _install_solver(([0, 0, 4], [], 4))

    with pytest.raises(Exception, match="clone-plan fixed point did not converge"):
        _run_z3(func)
    assert len(calls) == 1


def test_async_producer_annotations_reject_cross_stage_clone_augmentation():
    annotations = _annotations(latencies=[0, 0, 0], iis=[1, 1, 1], resource_flags=[0, 0, 0])
    annotations["software_pipeline_async_producers"] = [
        tvm.tirx.IntImm("int64", value) for value in [1, 0, 0]
    ]
    func, _ = _make_buffer_bind_func(annotations)
    calls = _install_solver(([0, 0, 4], [], 4))
    with pytest.raises(Exception, match="async producer/consumer ownership"):
        _run_z3(func)
    assert len(calls) == 1


@pytest.mark.parametrize("effect", ["extern", "pure_extern", "read_state"])
def test_noncloneable_scalar_effects_require_same_stage(effect):
    tir = tvm.tirx
    b = tir.decl_buffer((4,), "int32", name="effect_B")
    k = tir.Var("k", "int32")
    value = tir.Var("effect_value", "int32")
    if effect == "extern":
        expression = tir.call_extern("int32", "opaque_effect")
    elif effect == "pure_extern":
        expression = tir.call_pure_extern("int32", "pure_but_abi_opaque")
    else:
        expression = tir.call_intrin("int32", "tirx.tvm_context_id")
    func = _make_statement_func(
        [tir.Bind(value, expression), tir.BufferStore(b, value, [k])],
        [b],
        _annotations(latencies=[1, 1], iis=[1, 1], resource_flags=[0, 0]),
        k,
    )
    calls = _install_solver(([0, 1], [], 4))
    loop = _find_loop(_run_z3(func)["main"])
    assert len(calls) == 1
    assert _clone_plan(loop) == []
    stage_pairs = _nested_ints(calls[0]["stage_order_deps"])
    assert [0, 1] in stage_pairs and [1, 0] in stage_pairs


def test_noncloneable_cross_stage_solver_result_is_rejected():
    tir = tvm.tirx
    b = tir.decl_buffer((4,), "int32", name="opaque_B")
    k = tir.Var("k", "int32")
    value = tir.Var("opaque_value", "int32")
    func = _make_statement_func(
        [tir.Bind(value, tir.call_extern("int32", "opaque_effect")), tir.BufferStore(b, value, [k])],
        [b],
        _annotations(latencies=[1, 1], iis=[1, 1], resource_flags=[0, 0]),
        k,
    )
    _install_solver(([2, 5], [], 4))
    with pytest.raises(Exception, match="violates stage_order_deps"):
        _run_z3(func)


def test_mandatory_scalar_raw_dedup_keeps_maximum_latency_for_explicit_problem():
    annotations = _annotations(
        latencies=[2, 7, 3],
        iis=[1, 1, 1],
        resource_flags=[0, 0, 0],
        data_deps=[(1, 2, 0, 1)],
    )
    func, _ = _make_buffer_bind_func(annotations)
    calls = _install_solver(([0, 7, 14], [], 16))
    _run_z3(func)

    matching = [dep for dep in _nested_ints(calls[0]["data_deps"]) if dep[:3] == [1, 2, 0]]
    assert matching == [[1, 2, 0, 7]]
    assert [2, 1, 1, 3] in _nested_ints(calls[0]["data_deps"])
    assert len(calls) == 1


@pytest.mark.parametrize("recalculate", [True, False])
def test_versions_are_recomputed_or_under_versions_rejected(recalculate):
    def annotations(buffers):
        return _annotations(
            buffer_sizes=[4],
            buffer_vars=[buffers[0].data],
            memory_groups=[[8, 0]],
            data_deps=[(0, 1, -1, 30)],
            recalculate_buffer_versions=recalculate,
        )

    func, buffers = _make_loop(annotations, num_stages=3)
    _install_solver(([15, 16, 0, 0], [1], 16))
    if recalculate:
        loop = _find_loop(_run_z3(func)["main"])
        assert int(loop.annotations["tl_pipeline_buffer_versions"][buffers[0].data]) == 2
    else:
        with pytest.raises(Exception, match="under-versioned buffer 0"):
            _run_z3(func)


def test_recomputed_versions_recheck_memory_group_capacity():
    def annotations(buffers):
        return _annotations(
            buffer_sizes=[4],
            buffer_vars=[buffers[0].data],
            memory_groups=[[7, 0]],
            data_deps=[(0, 1, -1, 30)],
            recalculate_buffer_versions=True,
        )

    func, _ = _make_loop(annotations, num_stages=3)
    _install_solver(([15, 16, 0, 0], [1], 16))
    with pytest.raises(Exception, match="memory-group capacity"):
        _run_z3(func)


def test_manual_schedule_conflicts_with_handwritten_order_stage():
    annotations = _annotations(manual_schedule=True, manual_stages=[0, 0, 0, 0])
    func, _ = _make_loop(annotations, order=[0, 1, 2, 3], stage=[0, 0, 0, 0])
    calls = _install_solver(([0, 0, 0, 0], [], 8))

    with pytest.raises(Exception, match="handwritten pipeline order/stage"):
        _run_z3(func)
    assert calls == []


def test_auto_warp_specialization_conflict_is_explicit():
    func, _ = _make_loop(_annotations())
    calls = _install_solver(([0, 0, 0, 0], [], 8))

    with pytest.raises(Exception, match="automatic warp specialization"):
        _run_z3(func, config={_CONFIG: True, _AUTO_WS_CONFIG: "role_based"})
    assert calls == []


def test_existing_manual_warp_specialization_is_not_silently_skipped():
    func, buffers = _make_loop(_annotations())
    body = tvm.tirx.AttrStmt(0, "warp_specialize", 1, func.body)
    func = tvm.tirx.PrimFunc(
        func.params,
        body,
        buffer_map={buffer.data: buffer for buffer in buffers},
    ).with_attr("global_symbol", "main")
    calls = _install_solver(([0, 0, 0, 0], [], 8))

    with pytest.raises(Exception, match="existing warp-specialization schedule"):
        _run_z3(func)
    assert calls == []


def test_cuda_pipeline_rejects_auto_ws_before_any_scheduler_runs():
    from tilelang.cuda.pipeline import CUDAPassPipelineBodyPrologue

    func, _ = _make_loop(_annotations())
    mod = _bound_module(func)
    calls = _install_solver(([0, 0, 0, 0], [], 8))

    with tvm.transform.PassContext(config={_CONFIG: True, _AUTO_WS_CONFIG: "role_based"}), pytest.raises(
        ValueError, match="cannot schedule the same annotated module"
    ):
        CUDAPassPipelineBodyPrologue(mod, _SM80)
    assert calls == []


def test_main_pipeline_rejects_task_analysis_without_z3_planning():
    from tilelang.cuda.pipeline import CUDAPassPipelineBodyPrologue

    func, _ = _make_loop(_annotations())
    calls = _install_solver(([0, 0, 0, 0], [], 8))
    with tvm.transform.PassContext(config={"tl.enable_z3_task_analysis": True}), pytest.raises(
        ValueError, match="requires tl.enable_z3_pipeline_planning"
    ):
        CUDAPassPipelineBodyPrologue(_bound_module(func), _SM80)
    assert calls == []


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (([0, 0, 0], [], 8), "start times"),
        (([0, 0, 0, 0], [1], 8), "buffer versions"),
        (([0, 0, 0, 0], [], 0), "non-positive II"),
        (([-1, 0, 0, 0], [], 8), "negative start"),
    ],
)
def test_invalid_solver_returns_are_rejected(result, message):
    func, _ = _make_loop(_annotations())
    _install_solver(result)

    with pytest.raises(Exception, match=message):
        _run_z3(func)


def test_non_positive_buffer_version_is_rejected():
    def annotations(buffers):
        return _annotations(buffer_sizes=[16, 32], buffer_vars=buffers[:2])

    func, _ = _make_loop(annotations)
    _install_solver(([0, 0, 0, 0], [0, 2], 8))

    with pytest.raises(Exception, match="non-positive version"):
        _run_z3(func)


@pytest.mark.parametrize(
    "error",
    [
        tl.cuda.transform.Z3ScheduleInfeasibleError("injected UNSAT"),
        tl.cuda.transform.Z3ScheduleUnknownError("injected unknown at rlimit"),
    ],
)
def test_unsat_and_unknown_propagate_without_fallback(error):
    func, _ = _make_loop(_annotations())
    _install_solver(error)

    with pytest.raises(Exception, match="UNSAT|unknown"):
        _run_z3(func)


def test_no_annotation_is_strict_noop_even_when_enabled():
    func, _ = _make_loop({})
    before = _bound_module(func)
    calls = _install_solver(RuntimeError("must not be called"))

    with tvm.transform.PassContext(config={_CONFIG: True}):
        after = tl.transform.Z3PipelinePlanning()(before)
    assert after["main"].same_as(before["main"])
    assert _find_loop(after["main"]).annotations == _find_loop(before["main"]).annotations
    assert calls == []


def test_pass_config_defaults_off_and_preserves_annotated_loop():
    assert tl.PassConfigKey.TL_ENABLE_Z3_PIPELINE_PLANNING.value == _CONFIG
    assert _CONFIG not in normalize_pass_configs({})
    func, _ = _make_loop(_annotations())
    before = _bound_module(func)
    calls = _install_solver(RuntimeError("must not be called"))

    with tvm.transform.PassContext(config={}):
        after = tl.transform.Z3PipelinePlanning()(before)
    assert after["main"].same_as(before["main"])
    assert "tl_z3_pipeline" in _find_loop(after["main"]).annotations
    assert "tl_pipeline_order" not in _find_loop(after["main"]).annotations
    assert calls == []


def test_python_helper_validates_per_task_and_manual_shapes():
    with pytest.raises(ValueError, match="equal lengths"):
        _annotations(iis=[1])
    with pytest.raises(ValueError, match="requires one manual stage"):
        _annotations(manual_schedule=True)
    with pytest.raises(ValueError, match="buffer_sizes and buffer_vars"):
        _annotations(buffer_sizes=[1])


if __name__ == "__main__":
    tilelang.testing.main()
