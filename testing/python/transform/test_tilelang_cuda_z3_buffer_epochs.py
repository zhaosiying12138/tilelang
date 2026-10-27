"""Host-only CUDA Z3 clock, counter-group, and epoch-domain tests.

These tests inspect logical plans only. Counter task insertion, physical buffer
expansion, WS sync emission, and stock-pipeline clock lowering belong to later
slices. No build is performed here; tests skip until the central library exports
the Slice D inspection functions.
"""

from __future__ import annotations

import pytest
import tvm_ffi

import tilelang  # noqa: F401
from tilelang import language as T
from tilelang import tvm
from tvm import tirx


def _ffi(name):
    function = tvm_ffi.get_global_func(name, allow_missing=True)
    if function is None:
        pytest.skip(f"{name} requires the centrally rebuilt TileLang library")
    return function


def _roles():
    return [T.WSRole("Worker", warps_lo=0, warps_hi=4)]


def _unit(body, source_id, ws_op_id, *, guard=None, stage=0):
    metadata = {"ws_op_id": ws_op_id} if ws_op_id else {}
    body = tirx.AttrStmt(metadata, "tl.cuda_task", 1, body)
    if guard is not None:
        body = tirx.IfThenElse(guard, body, None)
    return tirx.AttrStmt(
        {"stage": stage, "source_id": source_id}, "tl.schedule_unit", 1, body
    )


def _control(loop_var, extent, children, source_id, ws_op_id, *, claims=(), stage=0):
    annotations = {"tl.ws_op_id": ws_op_id}
    if claims:
        annotations["multi_buffer_eligible"] = list(claims)
    body = children[0] if len(children) == 1 else tirx.SeqStmt(children)
    loop = tirx.For(
        loop_var,
        0,
        extent,
        tirx.ForKind.SERIAL,
        body,
        annotations=annotations,
    )
    return tirx.AttrStmt(
        {"stage": stage, "source_id": source_id}, "tl.schedule_unit", 1, loop
    )


def _root(body, buffers, versions, modes=None):
    annotations = {"tl.cuda_z3_selected_buffer_versions": versions}
    if modes:
        annotations["tl.buffer_version_mode"] = modes
    return tirx.SBlock(
        [],
        [],
        [],
        "tilelang_root",
        body,
        alloc_buffers=list(buffers),
        annotations=annotations,
    )


def _buffers():
    shared = tirx.decl_buffer((8,), "int32", name="shared", scope="shared")
    out = tirx.decl_buffer((8,), "int32", name="out", scope="global")
    return shared, out


def _ring_tasks(shared, out, iv, *, guard=None, second_guard=None, stages=(0, 0), prefix=1):
    return [
        _unit(
            tirx.BufferStore(shared, iv, [0]),
            prefix,
            "",
            guard=guard,
            stage=stages[0],
        ),
        _unit(
            tirx.BufferStore(out, shared[0], [0]),
            prefix + 1,
            "",
            guard=guard if second_guard is None else second_guard,
            stage=stages[1],
        ),
    ]


def _plan(root, roles=None):
    summary = _ffi("tl.cuda.transform.Z3BufferEpochs")(
        root, roles or _roles(), 0, '{"kind":"cuda","arch":"sm_90"}'
    )
    return {str(item["storage"]): item for item in summary}


def _registry(root):
    return _ffi("tl.cuda.transform.Z3EpochDomains")(
        root, _roles(), 0, '{"kind":"cuda","arch":"sm_90"}'
    )


@pytest.mark.parametrize("versions", [1, 3], ids=["single-version", "ring"])
def test_affine_single_owner_uses_iteration_and_retains_owner_semantics(versions):
    shared, out = _buffers()
    iv = tirx.Var("i", "int32")
    owner = _control(iv, 4, _ring_tasks(shared, out, iv), 10, "owner")
    plan = _plan(_root(owner, [shared, out], {shared.data: versions}))["shared"]
    assert plan["clock"] == "iteration"
    assert plan["counter_group"] == 0
    assert plan["counter"] == ""
    assert len(plan["owners"]) == 1
    assert plan["versions"] == versions


@pytest.mark.parametrize("versions", [1, 3], ids=["single-version", "ring"])
def test_guarded_single_owner_uses_counter_for_any_depth(versions):
    shared, out = _buffers()
    iv = tirx.Var("i", "int32")
    guard = iv % 2 == 0
    owner = _control(iv, 4, _ring_tasks(shared, out, iv, guard=guard), 10, "owner")
    plan = _plan(_root(owner, [shared, out], {shared.data: versions}))["shared"]
    assert plan["clock"] == "counter"
    assert plan["counter_group"] > 0
    assert plan["counter"]
    assert len(plan["owners"]) == 1
    assert tvm.arith.Analyzer().can_prove_equal(plan["owners"][0]["guard"], guard)


def test_explicit_counter_is_not_silently_changed_to_iteration():
    shared, out = _buffers()
    iv = tirx.Var("i", "int32")
    owner = _control(iv, 4, _ring_tasks(shared, out, iv), 10, "owner")
    root = _root(
        owner,
        [shared, out],
        {shared.data: 2},
        {shared.data: "counter"},
    )
    assert _plan(root)["shared"]["clock"] == "counter"


def test_explicit_iteration_rejects_guarded_or_multiple_owners():
    for owners, guarded in [(1, True), (2, False)]:
        shared, out = _buffers()
        controls = []
        for index in range(owners):
            iv = tirx.Var(f"i{index}", "int32")
            guard = iv % 2 == 0 if guarded else None
            controls.append(
                _control(
                    iv,
                    4,
                    _ring_tasks(
                        shared, out, iv, guard=guard, prefix=100 + index * 10
                    ),
                    10 + index,
                    f"owner{index}",
                )
            )
        body = controls[0] if owners == 1 else tirx.SeqStmt(controls)
        root = _root(
            body,
            [shared, out],
            {shared.data: 2},
            {shared.data: "iteration"},
        )
        with pytest.raises(Exception, match="iteration|exactly one owner|unconditional"):
            _plan(root)


def test_three_sibling_owners_share_one_continuous_counter_plan():
    shared, out = _buffers()
    predicate = tirx.Var("predicate", "bool")
    controls = []
    for index in range(3):
        iv = tirx.Var(f"i{index}", "int32")
        controls.append(
            _control(
                iv,
                4,
                _ring_tasks(shared, out, iv, guard=predicate, prefix=100 + index * 10),
                10 + index,
                f"owner{index}",
            )
        )
    root = _root(tirx.SeqStmt(controls), [shared, out], {shared.data: 2})
    plan = _plan(root)["shared"]
    assert plan["clock"] == "counter"
    assert [str(owner["owner"]) for owner in plan["owners"]] == [
        "owner0",
        "owner1",
        "owner2",
    ]
    # One group/counter is the plan-level contract for one reset and exactly
    # one guarded advance per owner in the later materialization slices.
    assert plan["counter_group"] > 0
    assert plan["counter"]


def test_non_affine_loop_break_requires_counter():
    shared, out = _buffers()
    iv = tirx.Var("i", "int32")
    body = _ring_tasks(shared, out, iv)
    body.append(_unit(tirx.Evaluate(T.loop_break()), 20, "break"))
    owner = _control(iv, 4, body, 10, "owner")
    plan = _plan(_root(owner, [shared, out], {shared.data: 2}))["shared"]
    assert plan["clock"] == "counter"


def test_cross_stage_auto_falls_back_to_lexical_but_explicit_counter_fails():
    for mode in ("auto", "counter"):
        shared, out = _buffers()
        iv = tirx.Var("i", "int32")
        guard = iv % 2 == 0
        owner = _control(
            iv,
            4,
            _ring_tasks(shared, out, iv, guard=guard, stages=(0, 1)),
            10,
            "owner",
        )
        modes = {shared.data: mode}
        root = _root(owner, [shared, out], {shared.data: 2}, modes)
        if mode == "auto":
            plan = _plan(root)["shared"]
            assert plan["clock"] == "iteration"
            assert bool(tvm.arith.Analyzer().can_prove(plan["owners"][0]["guard"]))
        else:
            with pytest.raises(Exception, match="multiple stages|common stage|counter"):
                _plan(root)


def test_cross_stage_multiple_owners_cannot_fall_back_to_iteration():
    shared, out = _buffers()
    controls = []
    for index in range(2):
        iv = tirx.Var(f"i{index}", "int32")
        controls.append(
            _control(
                iv,
                4,
                _ring_tasks(
                    shared, out, iv, stages=(0, 1), prefix=100 + index * 10
                ),
                10 + index,
                f"owner{index}",
            )
        )
    with pytest.raises(Exception, match="cross-stage|counter|common stage"):
        _plan(_root(tirx.SeqStmt(controls), [shared, out], {shared.data: 2}))


def test_late_guard_definition_uses_lexical_auto_and_rejects_explicit_counter():
    for mode in ("auto", "counter"):
        shared, out = _buffers()
        iv = tirx.Var("i", "int32")
        p = tirx.Var("p", "bool")
        q = tirx.Var("q", "bool")
        late = tirx.Var("late", "bool")
        body = _ring_tasks(shared, out, iv, guard=p, second_guard=late)
        body.insert(1, _unit(tirx.Bind(late, q), 30, "late_definition"))
        body.insert(
            2,
            _unit(
                tirx.BufferStore(shared, iv + 1, [0]),
                31,
                "late_write",
                guard=late,
            ),
        )
        owner = _control(iv, 4, body, 10, "owner")
        root = _root(owner, [shared, out], {shared.data: 2}, {shared.data: mode})
        if mode == "auto":
            plan = _plan(root)["shared"]
            assert plan["clock"] == "iteration"
            assert bool(tvm.arith.Analyzer().can_prove(plan["owners"][0]["guard"]))
        else:
            with pytest.raises(Exception, match="defined at or after|dominate|counter"):
                _plan(root)


def _grouping_fixture(relation):
    x = tirx.decl_buffer((8,), "int32", name="x", scope="shared")
    y = tirx.decl_buffer((8,), "int32", name="y", scope="shared")
    out = tirx.decl_buffer((8,), "int32", name="out", scope="global")
    pred = tirx.decl_buffer((1,), "int32", name="pred", scope="global")
    p = tirx.Var("p", "int32")
    q = tirx.Var("q", "int32")
    iv = tirx.Var("i", "int32")
    first = p > 0
    second = p > 0 if relation == "same" else p >= 1 if relation == "equivalent" else q > 0
    prefix = []
    middle = []
    if relation == "mutated-memory":
        first = tirx.Var("before_write", "bool")
        second = tirx.Var("after_write", "bool")
        prefix = [_unit(tirx.Bind(first, pred[0] > 0), 1, "before")]
        middle = [
            _unit(tirx.BufferStore(pred, 1, [0]), 10, "mutate"),
            _unit(tirx.Bind(second, pred[0] > 0), 11, "after"),
        ]
    body = [
        *prefix,
        *_ring_tasks(x, out, iv, guard=first, prefix=20),
        *middle,
        *_ring_tasks(y, out, iv, guard=second, prefix=30),
    ]
    owner = _control(iv, 4, body, 40, "owner")
    root = _root(
        owner,
        [x, y, out, pred],
        {x.data: 2, y.data: 3},
        {x.data: "counter", y.data: "counter"},
    )
    return root


@pytest.mark.parametrize(
    "relation,shared_group",
    [("same", True), ("equivalent", True), ("different", False), ("mutated-memory", False)],
)
def test_counter_group_guard_equivalence_and_independent_depths(relation, shared_group):
    plan = _plan(_grouping_fixture(relation))
    x, y = plan["x"], plan["y"]
    assert (x["counter_group"] == y["counter_group"]) == shared_group
    assert (x["counter"] == y["counter"]) == shared_group
    assert x["versions"] == 2 and y["versions"] == 3


def _projection_fixture():
    shared, out = _buffers()
    p = tirx.Var("p", "bool")
    q = tirx.Var("q", "bool")
    first_iv = tirx.Var("i", "int32")
    second_iv = tirx.Var("j", "int32")
    first = _control(
        first_iv,
        4,
        _ring_tasks(shared, out, first_iv, guard=p, prefix=10),
        1,
        "first",
    )
    second = _control(
        second_iv,
        4,
        _ring_tasks(shared, out, second_iv, guard=tirx.And(p, q), prefix=20),
        2,
        "second",
    )
    return _root(tirx.SeqStmt([first, second]), [shared, out], {shared.data: 2})


def test_epoch_registry_projects_guards_but_never_counter_to_lexical():
    summary = _registry(_projection_fixture())
    domains = {int(item["id"]): item for item in summary["domains"]}
    storage = {
        (str(item["storage"]), str(item["scope"])): int(item["domain"])
        for item in summary["storage_domains"]
    }
    first = storage["shared", "first"]
    second = storage["shared", "second"]
    root = storage["shared", "<root>"]
    assert domains[first]["kind"] == "counter"
    assert domains[second]["kind"] == "counter"
    assert domains[root]["kind"] == "lexical"
    projections = {
        (str(item["scope"]), int(item["source"]), int(item["target"]))
        for item in summary["projections"]
    }
    assert ("<root>", first, first) in projections
    assert ("<root>", first, second) in projections  # (p && q) implies p
    assert ("<root>", second, first) not in projections
    assert ("<root>", first, root) not in projections
    assert ("<root>", second, root) not in projections


def test_compatible_storages_intern_one_counter_domain_and_ids_are_deterministic():
    root = _grouping_fixture("equivalent")
    first = _registry(root)
    second = _registry(root)
    storage = {
        (str(item["storage"]), str(item["scope"])): int(item["domain"])
        for item in first["storage_domains"]
    }
    assert storage["x", "owner"] == storage["y", "owner"]
    signature = lambda summary: [
        (int(item["id"]), str(item["kind"]), str(item["owner"]), str(item["counter"]))
        for item in summary["domains"]
    ]
    assert signature(first) == signature(second)


def test_role_private_or_unavailable_guard_is_rejected_conservatively():
    shared, out = _buffers()
    src = tirx.decl_buffer((8,), "int32", name="src", scope="global")
    private = tirx.decl_buffer((1,), "int32", name="private", scope="local")
    iv = tirx.Var("i", "int32")
    guard = private[0] > 0
    body = [
        _unit(tirx.Evaluate(T.copy(src, shared)), 1, "load", guard=guard),
        _unit(tirx.Evaluate(T.copy(shared, out)), 2, "store", guard=guard),
    ]
    owner = _control(iv, 4, body, 10, "owner")
    root = _root(
        owner,
        [src, shared, out, private],
        {shared.data: 2},
        {shared.data: "counter"},
    )
    roles = [
        T.WSRole("Worker", warps_lo=0, warps_hi=4),
        T.WSRole("Load", warps_lo=4, warps_hi=5),
        T.WSRole("Store", warps_lo=5, warps_hi=6),
    ]
    with pytest.raises(Exception, match="guard|available|role|uniform"):
        _plan(root, roles=roles)
