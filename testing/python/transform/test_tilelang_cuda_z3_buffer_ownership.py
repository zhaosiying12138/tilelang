"""Host-only CUDA Z3 write-first ownership and complete-frontier tests.

The native inspection entry point is intentionally independent of the CUDA
compilation pipeline. Tests skip until the central shared library is rebuilt
with z3_buffer_ownership.cc; no GPU or kernel compilation is required.
"""

from __future__ import annotations

import pytest
import tvm_ffi

import tilelang  # noqa: F401
from tilelang import tvm
from tvm import tirx


def _ffi(name):
    function = tvm_ffi.get_global_func(name, allow_missing=True)
    if function is None:
        pytest.skip(f"{name} requires the centrally rebuilt TileLang library")
    return function


def _unit(body, source_id, ws_op_id, *, guard=None, stage=-1):
    metadata = {"ws_op_id": ws_op_id} if ws_op_id else {}
    body = tirx.AttrStmt(metadata, "tl.cuda_task", 1, body)
    if guard is not None:
        body = tirx.IfThenElse(guard, body, None)
    return tirx.AttrStmt(
        {"stage": stage, "source_id": source_id}, "tl.schedule_unit", 1, body
    )


def _control(loop_var, extent, children, source_id, ws_op_id, *, claims=(), stage=-1):
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


def _root(body, buffers, annotations=None):
    return tirx.SBlock(
        [],
        [],
        [],
        "tilelang_root",
        body,
        alloc_buffers=list(buffers),
        annotations=annotations or {},
    )


def _buffers():
    shared = tirx.decl_buffer((8,), "int32", name="shared", scope="shared")
    out = tirx.decl_buffer((8,), "int32", name="out", scope="global")
    return shared, out


def _write(buffer, value, index=0):
    indices = list(index) if isinstance(index, (tuple, list)) else [index]
    return tirx.BufferStore(buffer, value, indices)


def _read_into(buffer, out, index=0):
    return tirx.BufferStore(out, buffer[index], [index])


def _claims(root):
    summary = _ffi("tl.cuda.transform.Z3BufferOwnership")(root)
    return {
        str(item["storage"]): {
            "owners": [str(owner) for owner in item["owners"]],
            "broadcasts": [str(task) for task in item["broadcast_tasks"]],
        }
        for item in summary
    }


def _ring_body(
    shared, out, iv, *, write_guard=None, read_guard=None, stages=(-1, -1), source=10
):
    return [
        _unit(_write(shared, iv), source, "", guard=write_guard, stage=stages[0]),
        _unit(_read_into(shared, out), source + 1, "", guard=read_guard, stage=stages[1]),
    ]


@pytest.mark.parametrize(
    "case,expected",
    [
        ("write-first", True),
        ("read-first", False),
        ("read-modify-write", False),
        ("same-guard", True),
        ("disjoint-guards", False),
        ("exhaustive-writes", True),
        ("manual-stage-order", True),
    ],
)
def test_write_first_lattice_controls_automatic_claim(case, expected):
    shared, out = _buffers()
    enabled = tirx.Var("enabled", "bool")
    iv = tirx.Var("i", "int32")
    if case == "read-first":
        body = [
            _unit(_read_into(shared, out), 1, "read"),
            _unit(_write(shared, iv), 2, "write"),
        ]
    elif case == "read-modify-write":
        body = [_unit(_write(shared, shared[0] + 1), 1, "rmw")]
    elif case == "same-guard":
        body = _ring_body(shared, out, iv, write_guard=enabled, read_guard=enabled)
    elif case == "disjoint-guards":
        body = _ring_body(
            shared, out, iv, write_guard=enabled, read_guard=tirx.Not(enabled)
        )
    elif case == "exhaustive-writes":
        body = [
            _unit(_write(shared, iv), 1, "write_true", guard=enabled),
            _unit(_write(shared, iv + 1), 2, "write_false", guard=tirx.Not(enabled)),
            _unit(_read_into(shared, out), 3, "read"),
        ]
    elif case == "manual-stage-order":
        body = _ring_body(shared, out, iv, stages=(0, 1))
        body.reverse()
        body[0] = _unit(_read_into(shared, out), 1, "read", stage=1)
        body[1] = _unit(_write(shared, iv), 2, "write", stage=0)
    else:
        body = _ring_body(shared, out, iv)
    root = _root(_control(iv, 4, body, 20, "owner"), [shared, out])
    claims = _claims(root)
    assert ("shared" in claims) == expected
    if expected:
        assert claims["shared"]["owners"] == ["owner"]


def test_three_disjoint_siblings_form_one_ordered_frontier():
    shared, out = _buffers()
    loops = []
    for index in range(3):
        iv = tirx.Var(f"i{index}", "int32")
        loops.append(
            _control(
                iv,
                2,
                _ring_body(shared, out, iv, source=100 + index * 10),
                30 + index,
                f"owner{index}",
            )
        )
    claims = _claims(_root(tirx.SeqStmt(loops), [shared, out]))
    assert claims["shared"]["owners"] == ["owner0", "owner1", "owner2"]


def test_aliases_and_row_write_whole_read_promote_to_outer_owner():
    base = tirx.decl_buffer((4, 4), "int32", name="base", scope="shared")
    alias = tirx.decl_buffer((16,), "int32", name="alias", data=base.data)
    out = tirx.decl_buffer((16,), "int32", name="out", scope="global")
    outer = tirx.Var("outer", "int32")
    row = tirx.Var("row", "int32")
    inner = _control(
        row,
        4,
        [_unit(_write(base, row, [outer, row]), 1, "row_write")],
        2,
        "rows",
    )
    whole_read = _unit(tirx.BufferStore(out, alias[0], [0]), 3, "whole_read")
    owner = _control(outer, 4, [inner, whole_read], 4, "outer")
    claims = _claims(_root(owner, [base, out]))
    assert claims["base"]["owners"] == ["outer"]


def test_explicit_claim_suppresses_automatic_claim_and_requires_complete_coverage():
    shared, out = _buffers()
    first_iv = tirx.Var("i", "int32")
    second_iv = tirx.Var("j", "int32")
    first = _control(
        first_iv,
        2,
        _ring_body(shared, out, first_iv, source=40),
        1,
        "explicit",
        claims=[shared.data],
    )
    second = _control(
        second_iv, 2, _ring_body(shared, out, second_iv, source=50), 2, "automatic"
    )
    with pytest.raises(Exception, match="outside its owner|complete|frontier"):
        _claims(_root(tirx.SeqStmt([first, second]), [shared, out]))


@pytest.mark.parametrize(
    "versions,mode,expected",
    [(1, None, False), (2, None, False), (1, "auto", True), (1, "counter", True)],
)
def test_requested_version_policy_controls_automatic_eligibility(versions, mode, expected):
    shared, out = _buffers()
    iv = tirx.Var("i", "int32")
    annotations = {"tl.buffer_versions_map": {shared.data: versions}}
    if mode is not None:
        annotations["tl.buffer_version_mode"] = {shared.data: mode}
    root = _root(
        _control(iv, 2, _ring_body(shared, out, iv), 1, "owner"),
        [shared, out],
        annotations,
    )
    claims = _claims(root)
    assert ("shared" in claims) == expected


def test_bare_single_version_opt_out_removes_explicit_claims():
    shared, out = _buffers()
    iv = tirx.Var("i", "int32")
    owner = _control(
        iv,
        2,
        _ring_body(shared, out, iv),
        1,
        "owner",
        claims=[shared.data],
    )
    root = _root(owner, [shared, out], {"tl.buffer_versions_map": {shared.data: 1}})
    assert "shared" not in _claims(root)


def test_nested_explicit_owners_fail_with_storage_diagnostic():
    shared, out = _buffers()
    outer = tirx.Var("outer", "int32")
    inner = tirx.Var("inner", "int32")
    child = _control(
        inner,
        2,
        [_unit(_write(shared, inner), 1, "write")],
        2,
        "inner",
        claims=[shared.data],
    )
    parent = _control(outer, 2, [child], 3, "outer", claims=[shared.data])
    with pytest.raises(Exception, match="nested owner|disjoint"):
        _claims(_root(parent, [shared, out]))


def test_owner_external_ordinary_read_is_rejected():
    shared, out = _buffers()
    iv = tirx.Var("i", "int32")
    owner = _control(
        iv,
        2,
        [_unit(_write(shared, iv), 1, "write")],
        2,
        "owner",
        claims=[shared.data],
    )
    escaped = _unit(_read_into(shared, out), 3, "escaped_read")
    with pytest.raises(Exception, match="outside its owner|owner-external"):
        _claims(_root(tirx.SeqStmt([owner, escaped]), [shared, out]))


def test_unnormalized_buffer_dependent_bound_is_rejected():
    shared, out = _buffers()
    outer = tirx.Var("outer", "int32")
    inner = tirx.Var("inner", "int32")
    child = _control(
        inner,
        shared[0],
        [_unit(_read_into(shared, out), 1, "read")],
        2,
        "child",
    )
    parent = _control(
        outer,
        2,
        [_unit(_write(shared, 1), 3, "initialize"), child],
        4,
        "outer",
        claims=[shared.data],
    )
    with pytest.raises(Exception, match="unnormalized loop bound|normalize"):
        _claims(_root(parent, [shared, out]))


def _region(buffer, mask, shape=None):
    return tirx.Call(
        "handle",
        tvm.ir.Op.get("tl.region"),
        [buffer[tuple(0 for _ in buffer.shape)], mask, *(shape or buffer.shape)],
    )


def _fill(buffer, value, *, self_read=False):
    return tirx.Evaluate(
        tirx.Call(
            "handle",
            tvm.ir.Op.get("tl.tileop.fill"),
            [_region(buffer, 2), buffer[0] if self_read else value],
        )
    )


def test_owner_external_fill_is_recognized_for_every_version():
    shared, out = _buffers()
    iv = tirx.Var("i", "int32")
    fill = _unit(_fill(shared, 0), 1, "fill")
    owner = _control(
        iv,
        2,
        [_unit(_write(shared, iv), 2, "write")],
        3,
        "owner",
        claims=[shared.data],
    )
    root = _root(
        tirx.SeqStmt([fill, owner]),
        [shared, out],
        {"tl.cuda_z3_selected_buffer_versions": {shared.data: 3}},
    )
    claims = _claims(root)
    assert claims["shared"]["broadcasts"] == ["fill"]


def test_self_reading_external_fill_is_not_a_broadcast_exception():
    shared, out = _buffers()
    iv = tirx.Var("i", "int32")
    fill = _unit(_fill(shared, 0, self_read=True), 1, "bad_fill")
    owner = _control(
        iv,
        2,
        [_unit(_write(shared, iv), 2, "write")],
        3,
        "owner",
        claims=[shared.data],
    )
    with pytest.raises(Exception, match="outside its owner|non-self-reading|fill"):
        _claims(_root(tirx.SeqStmt([fill, owner]), [shared, out]))
