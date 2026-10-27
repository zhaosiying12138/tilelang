"""Host oracles for public CUDA Z3 Task/Stage schedule units (P01/P02)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

import tilelang
from tilelang import tvm
import tilelang.language as T
from tilelang.language.annotations import z3_pipeline_annotations
from tvm.tirx.stmt_functor import post_order_visit


_TASK = "tl.cuda_z3_task"
_STAGE = "tl.cuda_z3_stage"
_ANALYSIS_CONFIG = "tl.enable_z3_task_analysis"


def _imms(values):
    return [tvm.tirx.IntImm("int64", value) for value in values]


def _annotations(task_count, *, enable_offset=True, resource_flags=None):
    if resource_flags is None:
        resource_flags = [0] * task_count
    return z3_pipeline_annotations(
        latencies=[2] * task_count,
        iis=[1] * task_count,
        resource_flags=resource_flags,
        enable_offset=enable_offset,
    )


def _task(body, *, latency=None, ii=None):
    metadata = {}
    if latency is not None:
        metadata["latency"] = tvm.tirx.IntImm("int64", latency)
    if ii is not None:
        metadata["ii"] = tvm.tirx.IntImm("int64", ii)
    return tvm.tirx.AttrStmt(metadata, _TASK, tvm.tirx.IntImm("int32", 1), body)


def _stage(stage, body):
    return tvm.tirx.AttrStmt(stage, _STAGE, tvm.tirx.IntImm("int32", 1), body)


def _make_module(
    annotations: dict,
    body_factory: Callable[[tvm.tirx.Var, tvm.tirx.Buffer], tvm.tirx.Stmt | list[tvm.tirx.Stmt]],
    *,
    num_stages=2,
):
    tir = tvm.tirx
    output = tir.decl_buffer((32,), "int32", name="output")
    k = tir.Var("k", "int32")
    body = body_factory(k, output)
    if isinstance(body, list):
        body = tir.SeqStmt(body)
    annotations = dict(annotations)
    if num_stages is not None:
        annotations["num_stages"] = tir.IntImm("int32", num_stages)
    loop = tir.For(k, 0, 4, tir.ForKind.SERIAL, body, annotations=annotations)
    func = tir.PrimFunc(
        [output.data],
        loop,
        buffer_map={output.data: output},
    ).with_attr("global_symbol", "main")
    return tvm.IRModule.from_expr(func)


def _normalize(mod):
    return tilelang.transform.NormalizeZ3ScheduleUnits()(mod)


def _z3_loop(func_or_mod):
    func = func_or_mod["main"] if isinstance(func_or_mod, tvm.IRModule) else func_or_mod
    loops = []

    def visit(node):
        if isinstance(node, tvm.tirx.For) and "tl_z3_pipeline" in node.annotations:
            loops.append(node)

    post_order_visit(func.body, visit)
    assert len(loops) == 1
    return loops[0]


def _flat(loop):
    if isinstance(loop.body, tvm.tirx.SeqStmt):
        return list(loop.body.seq)
    return [loop.body]


def _ints(values):
    return [int(value) for value in values]


def _attrs(stmt, key):
    result = []

    def visit(node):
        if isinstance(node, tvm.tirx.AttrStmt) and node.attr_key == key:
            result.append(node)

    post_order_visit(stmt, visit)
    return result


@pytest.mark.parametrize(
    "kwargs",
    [
        {"latency": True},
        {"latency": -1},
        {"latency": 1.0},
        {"latency": "3"},
        {"ii": False},
        {"ii": 0},
        {"ii": -1},
        {"ii": 1.5},
        {"latency": 1, "ii": 2},
    ],
)
def test_task_frontend_rejects_invalid_costs(kwargs):
    with pytest.raises(ValueError, match="Task"):
        T.Task(**kwargs)


@pytest.mark.parametrize("stage", [True, -1, 1.5, "2"])
def test_stage_frontend_rejects_invalid_values(stage):
    with pytest.raises(ValueError, match="Stage"):
        T.Stage(stage)


def test_public_frontend_emits_exact_marker_shapes():
    @T.prim_func
    def main(A: T.Buffer((1,), "int32")):
        with T.Stage(2), T.Task(latency=3, ii=1):
            A[0] = 1

    stages = _attrs(main.body, _STAGE)
    tasks = _attrs(main.body, _TASK)
    assert len(stages) == len(tasks) == 1
    assert int(stages[0].node) == 2
    assert stages[0].node.dtype == "int64"
    assert int(stages[0].value) == 1
    assert set(tasks[0].node) == {"latency", "ii"}
    assert int(tasks[0].node["latency"]) == 3
    assert int(tasks[0].node["ii"]) == 1
    assert tasks[0].node["latency"].dtype == "int64"
    assert int(tasks[0].value) == 1
    assert stages[0].body.same_as(tasks[0])


def test_unmarked_and_legacy_flat_modules_are_structural_noops():
    for annotations in (
        _annotations(1),
        z3_pipeline_annotations(
            latencies=[2],
            iis=[1],
            resource_flags=[0],
            manual_schedule=True,
            manual_stages=[0],
        ),
    ):
        before = _make_module(annotations, lambda k, out: tvm.tirx.BufferStore(out, k, [k]))
        after = _normalize(before)
        tvm.ir.assert_structural_equal(after["main"], before["main"])


def test_task_is_one_atomic_statement_and_internal_bind_does_not_leak():
    def body(k, out):
        tir = tvm.tirx
        value = tir.Var("task_value", "int32")
        internal = tir.SeqStmt(
            [
                tir.Bind(value, k + 1),
                tir.BufferStore(out, value, [k]),
                tir.BufferStore(out, value + 1, [k + 1]),
            ]
        )
        return [_task(internal), tir.BufferStore(out, k, [k + 2])]

    loop = _z3_loop(_normalize(_make_module(_annotations(2), body)))
    statements = _flat(loop)
    assert len(statements) == 2
    assert statements[0].attr_key == _TASK
    assert isinstance(statements[0].body, tvm.tirx.SeqStmt)
    assert len(statements[0].body.seq) == 3
    assert not any(isinstance(stmt, tvm.tirx.Bind) for stmt in statements)
    assert list(loop.annotations["tl_z3_pipeline_execution_domains"]) == ["scalar", "scalar"]


def test_copy_tile_op_is_one_copy_domain_task():
    annotations = _annotations(1)

    @T.prim_func
    def main(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        for _k in T.Pipelined(4, num_stages=2, annotations=annotations):
            with T.Task():
                T.copy(A, B)

    loop = _z3_loop(_normalize(tvm.IRModule.from_expr(main)))
    assert list(loop.annotations["tl_z3_pipeline_execution_domains"]) == ["copy"]
    statements = _flat(loop)
    assert len(statements) == 1
    assert statements[0].attr_key == _TASK
    assert isinstance(statements[0].body, tvm.tirx.Evaluate)


def test_copy_and_scalar_compute_cannot_share_one_task():
    annotations = _annotations(1)

    @T.prim_func
    def main(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        for k in T.Pipelined(4, num_stages=2, annotations=annotations):
            with T.Task():
                T.copy(A, B)
                B[k] = B[k] + 1

    with pytest.raises(Exception, match="exactly one execution_domain"):
        _normalize(tvm.IRModule.from_expr(main))


def test_unknown_task_operation_is_not_assigned_a_guessed_domain():
    mod = _make_module(
        _annotations(1),
        lambda k, out: _task(tvm.tirx.Evaluate(tvm.tirx.call_extern("int32", "opaque_domain"))),
    )
    with pytest.raises(Exception, match="unknown tile operation or opaque call"):
        _normalize(mod)


@pytest.mark.parametrize("num_stages,expected_bound", [(None, 3), (0, 3), (1, 1), (8, 8)])
def test_partial_and_source_opposed_stages_normalize_depth_and_defaults(num_stages, expected_bound):
    def body(k, out):
        tir = tvm.tirx
        return [
            _stage(1, _task(tir.BufferStore(out, k, [k]))),
            tir.BufferStore(out, k + 1, [k + 1]),
            _stage(2, _task(tir.BufferStore(out, k + 2, [k + 2]))),
        ]

    mod = _make_module(_annotations(3), body, num_stages=num_stages)
    loop = _z3_loop(_normalize(mod))
    assert loop.annotations["tl_z3_pipeline_manual_schedule"] is True
    assert _ints(loop.annotations["tl_z3_pipeline_manual_stages"]) == [1, 0, 2]
    assert int(loop.annotations["num_stages"]) == expected_bound
    assert not _attrs(loop.body, _STAGE)
    assert len(_attrs(loop.body, _TASK)) == 2
    assert "tl_z3_pipeline_pipe_order_deps" not in loop.annotations


def test_replayable_bind_does_not_consume_a_manual_stage():
    def body(k, out):
        tir = tvm.tirx
        value = tir.Var("replayed", "int32")
        return [
            tir.Bind(value, k + 1),
            _stage(1, _task(tir.BufferStore(out, value, [k]))),
            tir.BufferStore(out, value + 1, [k + 1]),
        ]

    loop = _z3_loop(_normalize(_make_module(_annotations(2), body)))
    assert len(_flat(loop)) == 3
    assert _ints(loop.annotations["tl_z3_pipeline_manual_stages"]) == [1, 0]


def test_stage_containing_only_a_replayable_bind_is_rejected():
    value = tvm.tirx.Var("only_replayed", "int32")
    mod = _make_module(
        _annotations(1),
        lambda k, out: _stage(0, tvm.tirx.Bind(value, k + 1)),
    )
    with pytest.raises(Exception, match="markers do not contain a scheduled task"):
        _normalize(mod)


def test_cross_domain_task_is_rejected_before_cost_or_solver_fallback():
    def body(k, out):
        tir = tvm.tirx
        p = tir.Var("p", "int32")
        parallel = tir.For(p, 0, 2, tir.ForKind.PARALLEL, tir.BufferStore(out, p, [p]))
        return _task(tir.SeqStmt([parallel, tir.BufferStore(out, k, [k])]))

    with pytest.raises(Exception, match="exactly one execution_domain"):
        _normalize(_make_module(_annotations(1), body))


def test_nonzero_stage_requires_offset_and_zero_stage_does_not():
    zero = _make_module(
        _annotations(1, enable_offset=False),
        lambda k, out: _stage(0, _task(tvm.tirx.BufferStore(out, k, [k]))),
        num_stages=None,
    )
    assert int(_z3_loop(_normalize(zero)).annotations["num_stages"]) == 1

    nonzero = _make_module(
        _annotations(1, enable_offset=False),
        lambda k, out: _stage(1, _task(tvm.tirx.BufferStore(out, k, [k]))),
    )
    with pytest.raises(Exception, match="enable_offset=True"):
        _normalize(nonzero)


@pytest.mark.parametrize(
    "conflict",
    ["manual_schedule", "manual_stages", "tl_pipeline_order", "tl_pipeline_stage", "software_pipeline_order", "software_pipeline_stage", "pipe_order"],
)
def test_stage_rejects_old_and_handwritten_schedule_schemas(conflict):
    annotations = _annotations(1)
    if conflict == "manual_schedule":
        annotations["tl_z3_pipeline_manual_schedule"] = True
    elif conflict == "manual_stages":
        annotations["tl_z3_pipeline_manual_stages"] = _imms([0])
    elif conflict == "pipe_order":
        annotations["tl_z3_pipeline_pipe_order_deps"] = [_imms([0, 0])]
    else:
        annotations[conflict] = _imms([0])
    mod = _make_module(annotations, lambda k, out: _stage(0, _task(tvm.tirx.BufferStore(out, k, [k]))))
    with pytest.raises(Exception, match="cannot be combined|derives per-pipe"):
        _normalize(mod)


def test_task_can_use_the_legacy_manual_schema_without_stage_markers():
    annotations = z3_pipeline_annotations(
        latencies=[2],
        iis=[1],
        resource_flags=[0],
        manual_schedule=True,
        manual_stages=[0],
    )
    mod = _make_module(annotations, lambda k, out: _task(tvm.tirx.BufferStore(out, k, [k])))
    loop = _z3_loop(_normalize(mod))
    assert loop.annotations["tl_z3_pipeline_manual_schedule"] is True
    assert _ints(loop.annotations["tl_z3_pipeline_manual_stages"]) == [0]


@pytest.mark.parametrize(
    "body_factory,diagnostic",
    [
        (lambda k, out: _stage(0, _stage(0, _task(tvm.tirx.BufferStore(out, k, [k])))), "nested T.Stage"),
        (lambda k, out: _task(_stage(0, tvm.tirx.BufferStore(out, k, [k]))), "nested T.Task/T.Stage"),
        (lambda k, out: _task(_task(tvm.tirx.BufferStore(out, k, [k]))), "nested T.Task/T.Stage"),
        (lambda k, out: _stage(-1, _task(tvm.tirx.BufferStore(out, k, [k]))), "non-negative constant"),
        (lambda k, out: _task(tvm.tirx.Evaluate(0)), "executable statement"),
    ],
)
def test_invalid_task_stage_nesting_and_empty_tasks_are_errors(body_factory, diagnostic):
    with pytest.raises(Exception, match=diagnostic):
        _normalize(_make_module(_annotations(1), body_factory))


def test_nonconstant_stage_is_rejected():
    stage = tvm.tirx.Var("dynamic_stage", "int32")
    mod = _make_module(
        _annotations(1),
        lambda k, out: _stage(stage, _task(tvm.tirx.BufferStore(out, k, [k]))),
    )
    with pytest.raises(Exception, match="non-negative constant"):
        _normalize(mod)


@pytest.mark.parametrize("marker", [_TASK, _STAGE])
def test_markers_inside_parallel_task_bodies_are_rejected(marker):
    def body(k, out):
        tir = tvm.tirx
        p = tir.Var("p", "int32")
        store = tir.BufferStore(out, p, [p])
        nested = _task(store) if marker == _TASK else _stage(0, store)
        return tir.For(p, 0, 2, tir.ForKind.PARALLEL, nested)

    with pytest.raises(Exception, match="nested T.Task/T.Stage"):
        _normalize(_make_module(_annotations(1), body))


@pytest.mark.parametrize("marker", [_TASK, _STAGE])
def test_markers_outside_an_opted_in_z3_loop_are_rejected(marker):
    annotations = _annotations(1)
    annotations.pop("tl_z3_pipeline")

    def body(k, out):
        store = tvm.tirx.BufferStore(out, k, [k])
        return _task(store) if marker == _TASK else _stage(0, store)

    with pytest.raises(Exception, match="only legal inside"):
        _normalize(_make_module(annotations, body))


def test_normalization_is_idempotent():
    mod = _make_module(
        _annotations(2),
        lambda k, out: [
            _stage(1, _task(tvm.tirx.BufferStore(out, k, [k]))),
            tvm.tirx.BufferStore(out, k + 1, [k + 1]),
        ],
    )
    once = _normalize(mod)
    twice = _normalize(once)
    tvm.ir.assert_structural_equal(once["main"], twice["main"])


def test_stage_mode_infers_per_pipe_source_order_during_analysis():
    annotations = _annotations(2, resource_flags=[1, 1])
    annotations.pop("tl_z3_pipeline_resource_deps")
    mod = _make_module(
        annotations,
        lambda k, out: [
            _stage(0, _task(tvm.tirx.BufferStore(out, k, [k]))),
            _stage(0, _task(tvm.tirx.BufferStore(out, k + 1, [k + 1]))),
        ],
    )
    normalized = _normalize(mod)
    with tvm.transform.PassContext(config={_ANALYSIS_CONFIG: True}):
        inferred = tilelang.transform.InferZ3PipelineProblem()(normalized)
    loop = _z3_loop(inferred)
    assert [_ints(pair) for pair in loop.annotations["tl_z3_pipeline_pipe_order_deps"]] == [[0, 1]]
    assert [_ints(pair) for pair in loop.annotations["tl_z3_pipeline_resource_deps"]] == [[0, 1]]
