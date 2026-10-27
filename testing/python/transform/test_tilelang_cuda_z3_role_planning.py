"""Host-only CUDA Z3 role candidate and execution-demand tests.

These tests inspect native metadata only. They do not compile or execute a GPU
kernel and skip until the central shared library includes z3_role_planning.cc.
"""

from __future__ import annotations

import pytest
import tvm_ffi

import tilelang  # noqa: F401
from tilelang import language as T
from tilelang import tvm
from tvm import tirx


WORKER = 1
LOAD = 2
MMA = 4
STORE = 8


def _ffi(name):
    function = tvm_ffi.get_global_func(name, allow_missing=True)
    if function is None:
        pytest.skip(f"{name} requires the centrally rebuilt TileLang library")
    return function


def _roles():
    return [
        T.WSRole("Worker", warps_lo=0, warps_hi=4),
        T.WSRole("Load", warps_lo=4, warps_hi=5),
        T.WSRole("MMA", warps_lo=5, warps_hi=6),
        T.WSRole("Store", warps_lo=6, warps_hi=7),
    ]


def _unit(body, source_id, ws_op_id, *, guard=None, per_role=False):
    body = tirx.AttrStmt(
        {"ws_op_id": ws_op_id},
        "tl.cuda_per_role_task" if per_role else "tl.cuda_task",
        1,
        body,
    )
    if guard is not None:
        body = tirx.IfThenElse(guard, body, None)
    return tirx.AttrStmt(
        {"stage": -1, "source_id": source_id}, "tl.schedule_unit", 1, body
    )


def _control(loop_var, extent, children, source_id, ws_op_id):
    loop = tirx.For(
        loop_var,
        0,
        extent,
        tirx.ForKind.SERIAL,
        children[0] if len(children) == 1 else tirx.SeqStmt(children),
        annotations={"tl.ws_op_id": ws_op_id},
    )
    return tirx.AttrStmt(
        {"stage": -1, "source_id": source_id}, "tl.schedule_unit", 1, loop
    )


def _root(body, buffers, annotations=None):
    return tirx.SBlock(
        [], [], [], "tilelang_root", body, alloc_buffers=list(buffers), annotations=annotations or {}
    )


def _copy(src, dst):
    return tirx.Evaluate(T.copy(src, dst))


def _plan(root, *, resolve=True, roles=None, active=0):
    summary = _ffi("tl.cuda.transform.Z3RolePlanning")(
        root, roles or _roles(), active, '{"kind":"cuda","arch":"sm_90"}', resolve
    )
    return {str(item["task"]): dict(item) for item in summary}


def _task(summary, name):
    if name in summary:
        return summary[name]
    matches = [value for key, value in summary.items() if key.endswith("/" + name)]
    assert len(matches) == 1, (name, list(summary))
    return matches[0]


def _header(summary, loop_name):
    matches = [value for key, value in summary.items() if key.endswith("@header") and loop_name in key]
    assert len(matches) == 1, (loop_name, list(summary))
    return matches[0]


def _buffers():
    src = tirx.decl_buffer((16, 64), "float16", name="src", scope="global")
    dst = tirx.decl_buffer((16, 64), "float16", name="dst", scope="global")
    shared = tirx.decl_buffer((16, 64), "float16", name="shared", scope="shared")
    return src, dst, shared


def test_copy_instruction_selection_and_per_role_single_role_resolution():
    src, dst, shared = _buffers()
    load = _unit(_copy(src, shared), 1, "load")
    store = _unit(_copy(shared, dst), 2, "store")
    candidates = _plan(_root(tirx.SeqStmt([load, store]), [src, dst, shared]), resolve=False)
    assert _task(candidates, "load")["candidate"] == LOAD
    assert _task(candidates, "store")["candidate"] == STORE

    choose_load = tirx.Var("choose_load", "int32")
    alternatives = tirx.SeqStmt(
        [
            tirx.IfThenElse(choose_load != 0, _copy(src, shared), None),
            tirx.IfThenElse(choose_load == 0, _copy(shared, dst), None),
        ]
    )
    group = _unit(alternatives, 3, "group", per_role=True)
    resolved = _plan(_root(group, [src, dst, shared]))
    assert _task(resolved, "group")["candidate"] == LOAD | STORE
    assert _task(resolved, "group")["resolved"] == LOAD


def test_layout_ineligible_tma_copy_falls_back_to_worker():
    from tilelang.layout import Layout

    src, _, shared = _buffers()
    padded = Layout([16, 64], lambda i, j: i * 72 + j)
    load = _unit(_copy(src, shared), 1, "load")
    root = _root(load, [src, shared], annotations={"layout_map": {shared: padded}})
    summary = _plan(root, resolve=False)
    assert _task(summary, "load")["candidate"] == WORKER


def test_role_private_guard_rejects_incompatible_role_and_names_task():
    src, _, shared = _buffers()
    private = tirx.decl_buffer((1,), "int32", name="private", scope="local")
    out = tirx.decl_buffer((1,), "int32", name="out", scope="global")
    definition = _unit(
        tirx.SeqStmt(
            [tirx.BufferStore(out, 1, [0]), tirx.BufferStore(private, 1, [0])]
        ),
        1,
        "private_def",
    )
    load = _unit(_copy(src, shared), 2, "guarded_load", guard=private[0] > 0)
    with pytest.raises(Exception, match="guarded_load|unavailable|empty|Cannot place"):
        _plan(_root(tirx.SeqStmt([definition, load]), [src, shared, private, out]))


def test_global_scalar_replicates_but_external_effect_does_not_broadcast():
    src, _, shared = _buffers()
    predicate_buffer = tirx.decl_buffer((1,), "int32", name="predicate_buffer", scope="global")
    out = tirx.decl_buffer((1,), "int32", name="out", scope="global")
    predicate = tirx.Var("predicate", "int32")
    bind = _unit(tirx.Bind(predicate, predicate_buffer[0]), 1, "bind")
    load = _unit(_copy(src, shared), 2, "load", guard=predicate > 0)
    effect = _unit(tirx.BufferStore(out, predicate, [0]), 3, "effect")
    summary = _plan(
        _root(tirx.SeqStmt([bind, load, effect]), [src, shared, predicate_buffer, out])
    )
    assert _task(summary, "bind")["resolved"] == WORKER | LOAD
    assert _task(summary, "load")["resolved"] == LOAD
    assert _task(summary, "effect")["resolved"] == WORKER


def test_private_register_chain_and_ancestor_bound_fixed_points():
    src, _, shared = _buffers()
    private = tirx.decl_buffer((1,), "int32", name="private", scope="local")
    out = tirx.decl_buffer((16, 64), "float16", name="out", scope="global")
    x = tirx.Var("x", "int32")
    bound = tirx.Var("bound", "int32")
    i = tirx.Var("i", "int32")
    init = _unit(tirx.BufferStore(private, 1, [0]), 1, "init")
    chain = _unit(tirx.Bind(x, private[0]), 2, "chain")
    bind_bound = _unit(tirx.Bind(bound, 2), 3, "bind_bound")
    load = _unit(_copy(src, shared), 4, "load", guard=x > 0)
    worker = _unit(_copy(shared, out), 5, "worker")
    # A plain shared/global compound statement is Worker work; use a direct store
    # to avoid making the fixed-point fixture depend on another TMA classification.
    worker = _unit(tirx.BufferStore(out, shared[0, 0], [0, 0]), 5, "worker")
    loop = _control(i, bound, [load, worker], 6, "loop")
    summary = _plan(
        _root(tirx.SeqStmt([init, chain, bind_bound, loop]), [src, shared, private, out])
    )
    assert _task(summary, "init")["resolved"] == LOAD
    assert _task(summary, "chain")["resolved"] == LOAD
    assert _task(summary, "bind_bound")["resolved"] == WORKER | LOAD
    assert _header(summary, "loop")["resolved"] == WORKER | LOAD


def test_pure_fallback_is_deterministic_with_and_without_worker():
    value = tirx.Var("value", "int32")
    bind = _unit(tirx.Bind(value, 1), 1, "unused")
    assert _task(_plan(_root(bind, [])), "unused")["resolved"] == WORKER

    load_only_roles = [
        T.WSRole("Load", warps_lo=0, warps_hi=1),
        T.WSRole("Store", warps_lo=1, warps_hi=2),
    ]
    summary = _plan(_root(bind, []), roles=load_only_roles)
    assert _task(summary, "unused")["resolved"] == 1


def test_loop_exit_includes_later_third_role_and_all_control_producers():
    src, dst, shared = _buffers()
    out = tirx.decl_buffer((1,), "int32", name="out", scope="global")
    bound = tirx.Var("bound", "int32")
    predicate = tirx.Var("predicate", "int32")
    i = tirx.Var("i", "int32")
    bind_bound = _unit(tirx.Bind(bound, 2), 1, "bind_bound")
    load = _unit(_copy(src, shared), 2, "load")
    worker = _unit(tirx.BufferStore(out, i, [0]), 3, "worker")
    bind_predicate = _unit(tirx.Bind(predicate, tirx.Cast("int32", i >= 0)), 4, "predicate")
    break_task = _unit(
        tirx.Evaluate(T.loop_break()), 5, "break", guard=predicate > 0
    )
    loop = _control(i, bound, [load, worker, bind_predicate, break_task], 6, "loop")
    later_store = _unit(_copy(shared, dst), 7, "later_store", guard=predicate > 0)
    summary = _plan(
        _root(tirx.SeqStmt([bind_bound, loop, later_store]), [src, dst, shared, out])
    )
    required = WORKER | LOAD | STORE
    assert _task(summary, "predicate")["resolved"] == required
    assert _task(summary, "break")["resolved"] == required
    assert _task(summary, "bind_bound")["resolved"] == required
    assert _header(summary, "loop")["resolved"] == required


def test_role_specific_early_exit_around_shared_loop_is_rejected():
    src, _, shared = _buffers()
    out = tirx.decl_buffer((1,), "int32", name="out", scope="global")
    i = tirx.Var("i", "int32")
    load = _unit(_copy(src, shared), 1, "load")
    worker_break = _unit(
        tirx.SeqStmt(
            [tirx.BufferStore(out, i, [0]), tirx.Evaluate(T.loop_break())]
        ),
        2,
        "worker_break",
    )
    loop = _control(i, 2, [load, worker_break], 3, "loop")
    with pytest.raises(Exception, match="worker_break|role demand|broadcast"):
        _plan(_root(loop, [src, shared, out]))


def _counter_fixture(predicate):
    src, _, shared = _buffers()
    counter = tirx.decl_buffer((1,), "int32", name="counter", scope="local")
    i = tirx.Var("i", "int32")
    j = tirx.Var("j", "int32")
    epoch = _unit(tirx.Bind(predicate, 1), 1, "epoch")
    counter_init = _unit(tirx.BufferStore(counter, 0, [0]), 2, "counter_init")
    worker = _unit(tirx.BufferStore(shared, src[0, 0], [0, 0]), 3, "worker")
    first = _control(i, 1, [worker], 4, "first")
    load = _unit(_copy(src, shared), 5, "load")
    second = _control(j, 1, [load], 6, "second")
    root = _root(
        tirx.SeqStmt([epoch, counter_init, first, second]), [src, shared, counter]
    )
    return root


def test_counter_guards_and_state_join_execution_demand():
    predicate = tirx.Var("predicate", "int32")
    root = _counter_fixture(predicate)
    summary_list = _ffi("tl.cuda.transform.Z3CounterRolePlanning")(
        root,
        _roles(),
        0,
        '{"kind":"cuda","arch":"sm_90"}',
        "shared",
        "counter",
        ["first", "second"],
        [predicate > 0, predicate > 0],
    )
    summary = {str(item["task"]): dict(item) for item in summary_list}
    assert _task(summary, "epoch")["resolved"] == WORKER | LOAD
    assert _task(summary, "counter_init")["resolved"] == WORKER | LOAD


def test_counter_guard_private_to_one_protocol_role_is_rejected():
    src, _, shared = _buffers()
    counter = tirx.decl_buffer((1,), "int32", name="counter", scope="local")
    private = tirx.decl_buffer((1,), "int32", name="private", scope="local")
    out = tirx.decl_buffer((1,), "int32", name="out", scope="global")
    i = tirx.Var("i", "int32")
    definition = _unit(
        tirx.SeqStmt(
            [tirx.BufferStore(out, 1, [0]), tirx.BufferStore(private, 1, [0])]
        ),
        1,
        "private_def",
    )
    worker = _unit(tirx.BufferStore(shared, src[0, 0], [0, 0]), 2, "worker")
    first = _control(i, 1, [worker], 3, "first")
    load = _unit(_copy(src, shared), 4, "load")
    second = _control(i, 1, [load], 5, "second")
    root = _root(
        tirx.SeqStmt([definition, first, second]), [src, shared, counter, private, out]
    )
    with pytest.raises(Exception, match="guard|load|empty|Cannot"):
        _ffi("tl.cuda.transform.Z3CounterRolePlanning")(
            root,
            _roles(),
            0,
            '{"kind":"cuda","arch":"sm_90"}',
            "shared",
            "counter",
            ["first", "second"],
            [private[0] > 0, private[0] > 0],
        )
