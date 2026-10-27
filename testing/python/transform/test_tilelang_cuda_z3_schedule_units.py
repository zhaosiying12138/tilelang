"""Host-only tests for recursive CUDA Z3 schedule-unit materialization."""

from __future__ import annotations

import pytest
import tvm_ffi

import tilelang  # noqa: F401  # Load the TileLang shared library.
from tilelang import language as T
from tilelang import tvm
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit


def _ffi(name):
    function = tvm_ffi.get_global_func(name, allow_missing=True)
    if function is None:
        pytest.skip(f"{name} requires the centrally rebuilt TileLang library")
    return function


def _materialize(module):
    return _ffi("tl.cuda.transform.MaterializeZ3ScheduleUnits")()(module)


def _task(body, *, latency=None, ii=None, key="tl.cuda_task", metadata=None):
    payload = {} if metadata is None else dict(metadata)
    if latency is not None:
        payload["latency"] = tirx.IntImm("int64", latency)
    if ii is not None:
        payload["ii"] = tirx.IntImm("int64", ii)
    return tirx.AttrStmt(payload, key, 1, body)


def _stage(stage, body):
    return tirx.AttrStmt(stage, "tl.cuda_stage", 1, body)


def _per_role(body):
    return tirx.AttrStmt(0, "tl.cuda_per_role_task", 1, body)


def _kernel(body, *, buffers=()):
    root = tirx.SBlock([], [], [], "tilelang_root", body, alloc_buffers=list(buffers))
    func = tirx.PrimFunc([], tirx.SBlockRealize([], True, root)).with_attr("global_symbol", "main")
    return tvm.IRModule({"main": func})


def _root_body(module):
    return module["main"].body.block.body


def _flat(body):
    return list(body.seq) if isinstance(body, tirx.SeqStmt) else [body]


def _nodes(ir, kind):
    if isinstance(ir, tvm.IRModule):
        ir = ir["main"].body
    elif isinstance(ir, tirx.PrimFunc):
        ir = ir.body
    found = []
    post_order_visit(ir, lambda node: found.append(node) if isinstance(node, kind) else None)
    return found


def _units(ir):
    return [node for node in _nodes(ir, tirx.AttrStmt) if node.attr_key == "tl.schedule_unit"]


def _stages(units):
    return [int(unit.node["stage"]) for unit in units]


def _assert_source_ids_unique(module):
    source_ids = [int(unit.node["source_id"]) for unit in _units(module)]
    assert all(source_id >= 0 for source_id in source_ids)
    assert len(source_ids) == len(set(source_ids))


def _region(buffer, mask):
    indices = [0] * len(buffer.shape)
    return tirx.Call(
        "handle",
        tvm.ir.Op.get("tl.region"),
        [buffer[tuple(indices)], mask, *buffer.shape],
    )


def _copy(src, dst):
    return tirx.Evaluate(
        tirx.Call(
            "handle",
            tvm.ir.Op.get("tl.tileop.copy"),
            [_region(src, 1), _region(dst, 2)],
        )
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"latency": True},
        {"latency": -1},
        {"latency": 1.0},
        {"ii": False},
        {"ii": 0},
        {"ii": -1},
        {"latency": 1, "ii": 2},
    ],
)
def test_recursive_task_frontend_rejects_invalid_costs(kwargs):
    with pytest.raises(ValueError, match="Task"):
        T.Task(schema="recursive", **kwargs)


@pytest.mark.parametrize("stage", [True, -1, 1.5, "2"])
def test_recursive_stage_frontend_rejects_invalid_values(stage):
    with pytest.raises(ValueError, match="Stage"):
        T.Stage(stage, schema="recursive")


def test_schema_selection_is_explicit_and_flat_remains_default():
    with pytest.raises(ValueError, match="schema"):
        T.Task(schema="infer")

    @T.prim_func
    def flat(A: T.Buffer((1,), "int32")):
        with T.Stage(2), T.Task(latency=3, ii=1):
            A[0] = 1

    @T.prim_func
    def recursive(A: T.Buffer((1,), "int32")):
        with T.Stage(2, schema="recursive"), T.Task(latency=3, ii=1, schema="recursive"):
            A[0] = 1

    flat_attrs = _nodes(flat, tirx.AttrStmt)
    recursive_attrs = _nodes(recursive, tirx.AttrStmt)
    assert {node.attr_key for node in flat_attrs} == {"tl.cuda_z3_stage", "tl.cuda_z3_task"}
    assert {node.attr_key for node in recursive_attrs} == {"tl.cuda_stage", "tl.cuda_task"}
    task = next(node for node in recursive_attrs if node.attr_key == "tl.cuda_task")
    stage = next(node for node in recursive_attrs if node.attr_key == "tl.cuda_stage")
    assert int(task.node["latency"]) == 3
    assert int(task.node["ii"]) == 1
    assert int(stage.node) == 2
    assert stage.body.same_as(task)


def test_per_role_frontend_emits_only_recursive_marker():
    @T.prim_func
    def main(A: T.Buffer((1,), "int32")):
        with T.PerRoleTask(), T.Task(schema="recursive"):
            A[0] = 1

    markers = [node for node in _nodes(main, tirx.AttrStmt) if node.attr_key.startswith("tl.cuda")]
    assert {node.attr_key for node in markers} == {"tl.cuda_per_role_task", "tl.cuda_task"}
    group = next(node for node in markers if node.attr_key == "tl.cuda_per_role_task")
    task = next(node for node in markers if node.attr_key == "tl.cuda_task")
    assert int(group.value) == 1
    assert int(group.node) == 0
    assert group.body.same_as(task)


def test_if_else_becomes_complementary_guarded_siblings():
    output = tirx.decl_buffer((2,), "int32", name="output")
    condition = tirx.Var("condition", "bool")
    body = tirx.IfThenElse(
        condition,
        tirx.BufferStore(output, 1, [0]),
        tirx.BufferStore(output, 2, [1]),
    )

    after = _materialize(_kernel(body, buffers=[output]))
    units = _flat(_root_body(after))
    assert len(units) == 2
    assert _stages(units) == [-1, -1]
    then_guard, else_guard = [unit.body for unit in units]
    assert isinstance(then_guard, tirx.IfThenElse) and isinstance(else_guard, tirx.IfThenElse)
    assert then_guard.condition.same_as(condition)
    tvm.ir.assert_structural_equal(else_guard.condition, tirx.Not(condition))
    assert then_guard.then_case.attr_key == else_guard.then_case.attr_key == "tl.cuda_task"
    _assert_source_ids_unique(after)


def test_declarations_and_stage_modes_preserve_sequence_and_nested_independence():
    output = tirx.decl_buffer((8,), "int32", name="output")
    root_local = tirx.decl_buffer((1,), "int32", name="root_local", scope="local.var")
    loop_local = tirx.decl_buffer((1,), "int32", name="loop_local", scope="local.var")
    i = tirx.Var("i", "int32")
    nested = tirx.For(
        i,
        0,
        2,
        tirx.ForKind.SERIAL,
        tirx.SeqStmt(
            [
                tirx.AllocBuffer(loop_local),
                _stage(3, _task(tirx.BufferStore(output, i, [i]))),
                tirx.BufferStore(output, i + 1, [i]),
            ]
        ),
    )
    body = tirx.SeqStmt(
        [
            tirx.AttrStmt(None, "tl.ws_op_id", "root_declaration", tirx.DeclBuffer(root_local)),
            _stage(2, _task(tirx.BufferStore(output, 0, [0]))),
            tirx.BufferStore(output, 1, [0]),
            _stage(1, nested),
        ]
    )

    after = _materialize(_kernel(body, buffers=[output, root_local, loop_local]))
    top = _flat(_root_body(after))
    assert _stages(top) == [-1, 2, 0, 1]
    declaration = top[0].body
    assert declaration.attr_key == "tl.cuda_task"
    assert isinstance(declaration.body, tirx.DeclBuffer)
    assert int(declaration.node["declaration"]) == 1
    assert int(declaration.node["solver_eligible"]) == 0

    nested_units = _flat(top[-1].body.body)
    assert _stages(nested_units) == [-1, 3, 0]
    assert isinstance(nested_units[0].body.body, tirx.AllocBuffer)
    _assert_source_ids_unique(after)


def test_nested_auto_list_does_not_inherit_parent_manual_mode():
    output = tirx.decl_buffer((4,), "int32", name="output")
    i = tirx.Var("i", "int32")
    control = tirx.For(
        i,
        0,
        2,
        tirx.ForKind.SERIAL,
        tirx.SeqStmt([tirx.BufferStore(output, i, [i]), tirx.BufferStore(output, i + 1, [i])]),
    )
    body = tirx.SeqStmt([_stage(1, _task(tirx.BufferStore(output, 0, [0]))), control])
    after = _materialize(_kernel(body, buffers=[output]))
    top = _flat(_root_body(after))
    assert _stages(top) == [1, 0]
    assert _stages(_flat(top[1].body.body)) == [-1, -1]


def test_compound_worker_task_remains_atomic():
    output = tirx.decl_buffer((2,), "int32", name="output")
    value = tirx.Var("value", "int32")
    body = _task(
        tirx.SeqStmt(
            [
                tirx.Bind(value, 1),
                tirx.BufferStore(output, value, [0]),
                tirx.BufferStore(output, value + 1, [1]),
            ]
        ),
        latency=3,
        ii=1,
    )
    after = _materialize(_kernel(body, buffers=[output]))
    (unit,) = _flat(_root_body(after))
    assert unit.body.attr_key == "tl.cuda_task"
    assert isinstance(unit.body.body, tirx.SeqStmt)
    assert len(unit.body.body.seq) == 3
    assert int(unit.body.node["latency"]) == 3
    assert int(unit.body.node["ii"]) == 1


def test_cross_role_compound_task_is_rejected():
    source = tirx.decl_buffer((4,), "int32", name="source")
    shared = tirx.decl_buffer((4,), "int32", name="shared", scope="shared.dyn")
    body = _task(tirx.SeqStmt([_copy(source, shared), tirx.BufferStore(shared, 1, [0])]))
    with pytest.raises(Exception, match="one CUDA role-capability family"):
        _materialize(_kernel(body, buffers=[source, shared]))


@pytest.mark.parametrize("explicit", [True, False])
def test_explicit_and_inferred_per_role_candidates_keep_original_guards(explicit):
    output = tirx.decl_buffer((2,), "int32", name="output")
    condition = tirx.Var("condition", "bool")
    then_case = tirx.BufferStore(output, 1, [0])
    else_case = tirx.BufferStore(output, 2, [1])
    if explicit:
        then_case = _task(then_case)
        else_case = _task(else_case)
    body = _per_role(tirx.IfThenElse(condition, then_case, else_case))

    after = _materialize(_kernel(body, buffers=[output]))
    (unit,) = _flat(_root_body(after))
    group = unit.body
    assert group.attr_key == "tl.cuda_per_role_task"
    assert isinstance(group.body, tirx.IfThenElse)
    assert group.body.condition.same_as(condition)
    candidates = [node for node in _nodes(group, tirx.AttrStmt) if node.attr_key == "tl.cuda_task"]
    assert len(candidates) == 2
    # The candidate-dispatch If is inside the group, not flattened into sibling units.
    assert len(_units(after)) == 1


def test_dynamic_snapshot_guard_remains_user_exactly_once_contract():
    pred = tirx.decl_buffer((1,), "int32", name="pred", scope="shared.dyn")
    output = tirx.decl_buffer((1,), "int32", name="output")
    condition = tirx.Var("__cond_0", "bool")
    body = _per_role(
        tirx.SeqStmt(
            [
                tirx.Bind(condition, pred[0] > 0),
                tirx.IfThenElse(condition, tirx.BufferStore(output, 1, [0]), None),
            ]
        )
    )
    after = _materialize(_kernel(body, buffers=[pred, output]))
    (unit,) = _flat(_root_body(after))
    assert unit.body.attr_key == "tl.cuda_per_role_task"
    # No static exhaustiveness proof is required for a data-dependent guard.
    assert len([node for node in _nodes(unit, tirx.IfThenElse)]) == 1


def test_static_per_role_zero_multiple_and_nonexhaustive_cases_are_rejected():
    output = tirx.decl_buffer((1,), "int32", name="output")
    condition = tirx.Var("condition", "bool")
    store = tirx.BufferStore(output, 1, [0])
    cases = [
        (_per_role(tirx.IfThenElse(False, _task(store), None)), "unreachable"),
        (_per_role(tirx.SeqStmt([_task(store), _task(store)])), "pairwise exclusive"),
        (_per_role(tirx.IfThenElse(condition, _task(store), None)), "collectively exhaustive"),
    ]
    for body, diagnostic in cases:
        with pytest.raises(Exception, match=diagnostic):
            _materialize(_kernel(body, buffers=[output]))


def test_per_role_rejects_mismatched_role_and_access_signatures():
    source = tirx.decl_buffer((1,), "int32", name="source")
    shared = tirx.decl_buffer((1,), "int32", name="shared", scope="shared.dyn")
    output = tirx.decl_buffer((1,), "int32", name="output")
    condition = tirx.Var("condition", "bool")

    role_mismatch = _per_role(
        tirx.IfThenElse(condition, _task(_copy(source, shared)), _task(tirx.BufferStore(output, 1, [0])))
    )
    with pytest.raises(Exception, match="same CUDA role-capability family"):
        _materialize(_kernel(role_mismatch, buffers=[source, shared, output]))

    signature_mismatch = _per_role(
        tirx.IfThenElse(
            condition,
            _task(tirx.BufferStore(output, source[0], [0])),
            _task(tirx.BufferStore(output, 1, [0])),
        )
    )
    with pytest.raises(Exception, match="direction|signatures"):
        _materialize(_kernel(signature_mismatch, buffers=[source, output]))


def test_unresolved_opaque_per_role_side_effect_is_rejected():
    condition = tirx.Var("condition", "bool")
    opaque = tirx.Evaluate(tirx.call_extern("int32", "opaque_side_effect"))
    body = _per_role(tirx.IfThenElse(condition, _task(opaque), _task(opaque)))
    with pytest.raises(Exception, match="unresolved opaque side effect"):
        _materialize(_kernel(body))


@pytest.mark.parametrize(
    "body,diagnostic",
    [
        (lambda store: _task(_task(store)), "Nested T.Task"),
        (lambda store: _task(_stage(0, store)), "T.Stage cannot appear inside T.Task"),
        (lambda store: _stage(0, _stage(0, _task(store))), "Nested T.Stage"),
        (lambda store: _per_role(_per_role(_task(store))), "Nested T.PerRoleTask"),
        (lambda store: _task(_per_role(_task(store))), "PerRoleTask cannot be nested inside T.Task"),
    ],
)
def test_nested_atomic_and_stage_markers_are_rejected(body, diagnostic):
    output = tirx.decl_buffer((1,), "int32", name="output")
    with pytest.raises(Exception, match=diagnostic):
        _materialize(_kernel(body(tirx.BufferStore(output, 1, [0])), buffers=[output]))


def test_flat_and_recursive_markers_are_never_silently_combined():
    output = tirx.decl_buffer((1,), "int32", name="output")
    body = tirx.SeqStmt(
        [
            _task(tirx.BufferStore(output, 0, [0]), key="tl.cuda_z3_task"),
            _task(tirx.BufferStore(output, 1, [0])),
        ]
    )
    with pytest.raises(Exception, match="flat schema"):
        _materialize(_kernel(body, buffers=[output]))


def test_public_compiler_owned_task_fields_and_nonconstant_stage_are_rejected():
    output = tirx.decl_buffer((1,), "int32", name="output")
    store = tirx.BufferStore(output, 1, [0])
    with pytest.raises(Exception, match="compiler-owned"):
        _materialize(_kernel(_task(store, metadata={"candidate_roles_lo": 1}), buffers=[output]))

    dynamic_stage = tirx.Var("dynamic_stage", "int32")
    with pytest.raises(Exception, match="non-negative constant integer"):
        _materialize(_kernel(_stage(dynamic_stage, _task(store)), buffers=[output]))


def test_stage_requires_a_complete_task_and_duplicate_ws_ids_are_rejected():
    output = tirx.decl_buffer((1,), "int32", name="output")
    local = tirx.decl_buffer((1,), "int32", name="local", scope="local.var")
    with pytest.raises(Exception, match="complete scheduler task"):
        _materialize(_kernel(_stage(0, tirx.AllocBuffer(local)), buffers=[local]))

    first = tirx.AttrStmt(None, "tl.ws_op_id", "duplicate", _task(tirx.BufferStore(output, 0, [0])))
    second = tirx.AttrStmt(None, "tl.ws_op_id", "duplicate", _task(tirx.BufferStore(output, 1, [0])))
    with pytest.raises(Exception, match="Duplicate tl.ws_op_id"):
        _materialize(_kernel(tirx.SeqStmt([first, second]), buffers=[output]))


def test_bare_while_and_unknown_guard_attribute_fail_with_integration_diagnostics():
    pred = tirx.decl_buffer((1,), "int32", name="pred", scope="shared.dyn")
    bare = tirx.While(pred[0] > 0, tirx.BufferStore(pred, 0, [0]))
    with pytest.raises(Exception, match="NormalizeZ3ControlFlow"):
        _materialize(_kernel(bare, buffers=[pred]))

    unknown = tirx.AttrStmt(0, "example.unknown_guard", 1, tirx.BufferStore(pred, 1, [0]))
    with pytest.raises(Exception, match="example.unknown_guard.*root"):
        _materialize(_kernel(unknown, buffers=[pred]))
