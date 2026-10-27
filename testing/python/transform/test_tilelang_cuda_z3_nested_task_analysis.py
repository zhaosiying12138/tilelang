"""Host-only nested CUDA Z3 access and dependency analysis tests.

The inspection FFI becomes available after the central build includes
src/cuda/transform/z3_dependency_analysis.cc. No CUDA kernel is compiled or run.
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


def _unit(body, source_id, *, ws_op_id=None, stage=-1, guard=None, assume=None):
    metadata = {} if ws_op_id is None else {"ws_op_id": ws_op_id}
    body = tirx.AttrStmt(metadata, "tl.cuda_task", 1, body)
    if assume is not None:
        body = tirx.AttrStmt(assume, "tl.assume", 1, body)
    if guard is not None:
        body = tirx.IfThenElse(guard, body, None)
    return tirx.AttrStmt(
        {"stage": stage, "source_id": source_id}, "tl.schedule_unit", 1, body
    )


def _control(loop_var, extent, children, source_id, ws_op_id, *, minimum=0, step=None):
    loop = tirx.For(
        loop_var,
        minimum,
        extent,
        tirx.ForKind.SERIAL,
        children[0] if len(children) == 1 else tirx.SeqStmt(children),
        annotations={"tl.ws_op_id": ws_op_id},
        step=step,
    )
    return tirx.AttrStmt(
        {"stage": -1, "source_id": source_id}, "tl.schedule_unit", 1, loop
    )


def _root(body, buffers=(), annotations=None):
    return tirx.SBlock(
        [], [], [], "tilelang_root", body, alloc_buffers=list(buffers), annotations=annotations or {}
    )


def _analyze(root):
    return [dict(item) for item in _ffi("tl.cuda.transform.Z3NestedTaskAnalysis")(root)]


def _edges(deps, prod=None, cons=None, *, scope=None, distance=None, storage=None, kind=None):
    result = []
    for dep in deps:
        if prod is not None and str(dep["prod"]) != prod:
            continue
        if cons is not None and str(dep["cons"]) != cons:
            continue
        if scope is not None and str(dep["scope"]) != scope:
            continue
        if distance is not None and int(dep["distance"]) != distance:
            continue
        if storage is not None and str(dep["storage"]) != storage:
            continue
        if kind is not None and int(dep["kind"]) != kind:
            continue
        result.append(dep)
    return result


def _with_pair(deps, producer, consumer):
    result = []
    for dep in deps:
        pairs = [str(value) for value in dep["task_pairs"]]
        if any(producer in pairs[i] and consumer in pairs[i + 1] for i in range(0, len(pairs), 2)):
            result.append(dep)
    return result


def test_nested_raw_war_waw_and_guard_only_reads_reach_exact_tasks():
    data = tirx.decl_buffer((8,), "int32", name="data", scope="shared")
    flags = tirx.decl_buffer((1,), "int32", name="flags", scope="shared")
    out = tirx.decl_buffer((8,), "int32", name="out", scope="shared")
    i = tirx.Var("i", "int32")
    flag_write = _unit(tirx.BufferStore(flags, 1, [0]), 1, ws_op_id="flag_write")
    first = _unit(tirx.BufferStore(data, i, [i]), 2, ws_op_id="first")
    guarded = _unit(
        tirx.BufferStore(out, data[i], [i]),
        3,
        ws_op_id="guarded",
        guard=flags[0] > 0,
    )
    second = _unit(tirx.BufferStore(data, 2, [i]), 4, ws_op_id="second")
    loop = _control(i, 8, [first, guarded, second], 5, "inner")
    deps = _analyze(_root(tirx.SeqStmt([flag_write, loop]), [data, flags, out]))

    assert _with_pair(_edges(deps, "first", "guarded", scope="inner", distance=0), "first", "guarded")
    assert _edges(deps, "first", "second", scope="inner", distance=0)  # WAW
    assert _edges(deps, "guarded", "second", scope="inner", distance=0)  # WAR
    guard_edges = _with_pair(
        _edges(deps, "flag_write", "inner", scope="root", distance=0),
        "flag_write",
        "guarded",
    )
    assert guard_edges and all(str(dep["storage"]) == "flags" for dep in guard_edges)


def test_mutated_bound_snapshot_and_control_header_scalar_dependency():
    limits = tirx.decl_buffer((1,), "int32", name="limits", scope="shared")
    out = tirx.decl_buffer((8,), "int32", name="out", scope="shared")
    bound = tirx.Var("bound", "int32")
    i = tirx.Var("i", "int32")
    write_limit = _unit(tirx.BufferStore(limits, 4, [0]), 1, ws_op_id="write_limit")
    snapshot = _unit(tirx.Bind(bound, limits[0]), 2, ws_op_id="snapshot")
    child = _unit(tirx.BufferStore(out, i, [i]), 3, ws_op_id="child")
    loop = _control(i, bound, [child], 4, "bounded_loop")
    deps = _analyze(_root(tirx.SeqStmt([write_limit, snapshot, loop]), [limits, out]))

    assert _with_pair(
        _edges(deps, "write_limit", "snapshot", scope="root"),
        "write_limit",
        "snapshot",
    )
    header = _with_pair(
        _edges(deps, "snapshot", "bounded_loop", scope="root", distance=0),
        "snapshot",
        "bounded_loop (control header)",
    )
    assert header, deps


def test_aliased_views_conflict_by_storage_identity():
    storage = tirx.decl_buffer((8,), "int32", name="storage", scope="shared")
    view = tirx.decl_buffer((8,), "int32", name="view", data=storage.data, scope="shared")
    writer = _unit(tirx.BufferStore(storage, 1, [0]), 1, ws_op_id="writer")
    reader = _unit(tirx.BufferStore(view, view[0] + 1, [1]), 2, ws_op_id="reader")
    deps = _analyze(_root(tirx.SeqStmt([writer, reader]), [storage]))
    edges = _with_pair(_edges(deps, "writer", "reader", distance=0), "writer", "reader")
    assert edges and all(str(dep["storage"]) == "storage" for dep in edges)


def test_opaque_side_effect_is_rejected_even_with_pointer_transport():
    data = tirx.decl_buffer((8,), "int32", name="data", scope="shared")
    opaque = tirx.Evaluate(
        tirx.call_extern("int32", "opaque_side_effect", T.address_of(data[0]))
    )
    root = _root(_unit(opaque, 1, ws_op_id="opaque"), [data])
    with pytest.raises(Exception, match="opaque|stateful"):
        _analyze(root)


def test_loop_break_fences_only_its_direct_loop_siblings():
    i = tirx.Var("i", "int32")
    j = tirx.Var("j", "int32")
    break_task = _unit(tirx.Evaluate(T.loop_break()), 1, ws_op_id="break")
    sibling = _unit(tirx.Evaluate(0), 2, ws_op_id="sibling")
    inner = _control(j, 4, [break_task, sibling], 3, "inner")
    outer_sibling = _unit(tirx.Evaluate(0), 4, ws_op_id="outer_sibling")
    outer = _control(i, 3, [inner, outer_sibling], 5, "outer")
    deps = _analyze(_root(outer))

    assert _with_pair(
        _edges(deps, "break", "sibling", scope="inner", distance=0), "break", "sibling"
    )
    assert not _with_pair(
        _edges(deps, "inner", "outer_sibling", scope="outer"), "break", "outer_sibling"
    )


def test_mbarrier_state_and_buffer_contract_cross_control_boundary():
    barrier = tirx.decl_buffer((1,), "int32", name="barrier", scope="shared")
    i = tirx.Var("i", "int32")
    arrive = _unit(tirx.Evaluate(T.ptx_arrive_barrier(barrier[0])), 1, ws_op_id="arrive")
    inner = _control(i, 2, [arrive], 2, "inner")
    wait = _unit(tirx.Evaluate(T.mbarrier_wait_parity(barrier[0], 0)), 3, ws_op_id="wait")
    deps = _analyze(_root(tirx.SeqStmt([inner, wait]), [barrier]))

    data_edges = _with_pair(
        _edges(deps, "inner", "wait", scope="root", distance=0, storage="barrier"),
        "arrive",
        "wait",
    )
    state_edges = _with_pair(
        _edges(deps, "inner", "wait", scope="root", distance=0, storage=""),
        "arrive",
        "wait",
    )
    assert data_edges and state_edges


def test_scalar_chain_and_diamond_cover_bounds_guards_and_nested_consumers():
    out = tirx.decl_buffer((8,), "int32", name="out", scope="shared")
    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    c = tirx.Var("c", "int32")
    d = tirx.Var("d", "int32")
    i = tirx.Var("i", "int32")
    bind_a = _unit(tirx.Bind(a, 1), 1, ws_op_id="bind_a")
    bind_b = _unit(tirx.Bind(b, a + 1), 2, ws_op_id="bind_b")
    bind_c = _unit(tirx.Bind(c, a + 2), 3, ws_op_id="bind_c")
    bind_d = _unit(tirx.Bind(d, b + c), 4, ws_op_id="bind_d")
    consumer = _unit(
        tirx.BufferStore(out, d, [i]), 5, ws_op_id="consumer", guard=c > 0
    )
    loop = _control(i, b, [consumer], 6, "loop")
    deps = _analyze(_root(tirx.SeqStmt([bind_a, bind_b, bind_c, bind_d, loop]), [out]))

    assert _with_pair(_edges(deps, "bind_a", "bind_b"), "bind_a", "bind_b")
    assert _with_pair(_edges(deps, "bind_a", "bind_c"), "bind_a", "bind_c")
    assert _with_pair(_edges(deps, "bind_d", "loop"), "bind_d", "consumer")
    assert _with_pair(
        _edges(deps, "bind_c", "loop"), "bind_c", "consumer"
    )  # guard-only scalar use
    assert _with_pair(
        _edges(deps, "bind_b", "loop"), "bind_b", "loop (control header)"
    )


def test_manual_ring_checks_beyond_distance_one():
    data = tirx.decl_buffer((16,), "int32", name="data", scope="shared")
    out = tirx.decl_buffer((1,), "int32", name="out", scope="shared")
    i = tirx.Var("i", "int32")
    writer = _unit(tirx.BufferStore(data, i, [i]), 1, ws_op_id="writer")
    reader = _unit(tirx.BufferStore(out, data[i - 2], [0]), 2, ws_op_id="reader")
    loop = _control(i, 8, [writer, reader], 3, "loop", minimum=2)
    root = _root(
        loop,
        [data, out],
        annotations={"tl.buffer_versions_map": {data.data: 3}},
    )
    deps = _analyze(root)
    assert _with_pair(
        _edges(deps, "writer", "reader", scope="loop", distance=2), "writer", "reader"
    )


def test_duplicate_scalar_identity_and_use_before_definition_are_rejected():
    value = tirx.Var("value", "int32")
    duplicate = tirx.SeqStmt(
        [
            _unit(tirx.Bind(value, 1), 1, ws_op_id="first"),
            _unit(tirx.Bind(value, 2), 2, ws_op_id="second"),
        ]
    )
    with pytest.raises(Exception, match="Duplicate|duplicate|scalar"):
        _analyze(_root(duplicate))

    out = tirx.decl_buffer((1,), "int32", name="out", scope="shared")
    use_before_def = tirx.SeqStmt(
        [
            _unit(tirx.BufferStore(out, value, [0]), 3, ws_op_id="reader"),
            _unit(tirx.Bind(value, 1), 4, ws_op_id="late"),
        ]
    )
    with pytest.raises(Exception, match="before|lexical|scalar"):
        _analyze(_root(use_before_def, [out]))
