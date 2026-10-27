"""Host-only oracles for recursive CUDA Z3 control-flow normalization.

The FFI passes require a central build that includes the new CUDA sources. These
tests never compile or launch a kernel and require no GPU.
"""

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


def _normalize(module):
    return _ffi("tl.cuda.transform.NormalizeZ3ControlFlow")()(module)


def _restore(module):
    return _ffi("tl.transform.RestoreWhileLoops")()(module)


def _kernel(body, *, buffers=()):
    root = tirx.SBlock([], [], [], "tilelang_root", body, alloc_buffers=list(buffers))
    func = tirx.PrimFunc([], tirx.SBlockRealize([], True, root)).with_attr("global_symbol", "main")
    return tvm.IRModule({"main": func})


def _root_body(module):
    return module["main"].body.block.body


def _nodes(ir, kind):
    if isinstance(ir, tvm.IRModule):
        ir = ir["main"].body
    elif isinstance(ir, tirx.PrimFunc):
        ir = ir.body
    found = []
    post_order_visit(ir, lambda node: found.append(node) if isinstance(node, kind) else None)
    return found


def _flat(body):
    return list(body.seq) if isinstance(body, tirx.SeqStmt) else [body]


@pytest.mark.parametrize("control", ["condition", "loop-bound"])
def test_mutable_control_read_is_snapshotted_before_mutation(control):
    pred = tirx.decl_buffer((1,), "int32", name="pred", scope="shared.dyn")
    write = tirx.BufferStore(pred, 0, [0])
    if control == "condition":
        body = tirx.IfThenElse(pred[0] > 0, write, tirx.BufferStore(pred, 1, [0]))
    else:
        body = tirx.For(tirx.Var("i", "int32"), 0, pred[0], tirx.ForKind.SERIAL, write)

    after = _normalize(_kernel(body, buffers=[pred]))
    (snapshot,) = _nodes(after, tirx.Bind)
    (load,) = _nodes(snapshot.value, tirx.BufferLoad)
    assert load.buffer.same_as(pred)

    statements = _flat(_root_body(after))
    assert statements[0].same_as(snapshot)
    if control == "condition":
        (branch,) = _nodes(after, tirx.IfThenElse)
        assert branch.condition.same_as(snapshot.var)
        assert isinstance(branch.then_case, tirx.BufferStore)
        assert isinstance(branch.else_case, tirx.BufferStore)
    else:
        (loop,) = _nodes(after, tirx.For)
        assert loop.extent.same_as(snapshot.var)
        assert statements[-1].same_as(loop)


@pytest.mark.parametrize("field", ["min", "extent", "step"])
def test_every_buffer_reading_serial_loop_bound_is_snapshotted(field):
    pred = tirx.decl_buffer((4,), "int32", name="pred", scope="shared.dyn")
    values = {"min": 0, "extent": 3, "step": 1}
    values[field] = pred[0] + 1
    loop_var = tirx.Var("i", "int32")
    body = tirx.For(
        loop_var,
        values["min"],
        values["extent"],
        tirx.ForKind.SERIAL,
        tirx.BufferStore(pred, 0, [0]),
        step=values["step"],
    )

    after = _normalize(_kernel(body, buffers=[pred]))
    (snapshot,) = _nodes(after, tirx.Bind)
    (loop,) = _nodes(after, tirx.For)
    assert _flat(_root_body(after))[0].same_as(snapshot)
    actual = loop.step if field == "step" else getattr(loop, field)
    assert actual.same_as(snapshot.var)
    assert any(load.buffer.same_as(pred) for load in _nodes(snapshot.value, tirx.BufferLoad))


def test_nested_while_normalizes_and_restores_bottom_up_without_dead_counters():
    pred = tirx.decl_buffer((2,), "int32", name="pred", scope="shared.dyn")
    output = tirx.decl_buffer((4,), "int32", name="output")
    stop = tirx.Var("stop", "bool")
    inner = tirx.While(
        pred[1] > 0,
        tirx.SeqStmt(
            [
                tirx.IfThenElse(stop, tirx.Evaluate(T.loop_break()), None),
                tirx.BufferStore(output, 1, [0]),
            ]
        ),
    )
    outer = tirx.While(pred[0] > 0, inner)

    normalized = _normalize(_kernel(outer, buffers=[pred, output]))
    synthetic = [
        loop for loop in _nodes(normalized, tirx.For) if "synthetic_while" in loop.annotations
    ]
    assert len(synthetic) == 2
    assert all("num_stages" not in loop.annotations for loop in synthetic)
    assert all(int(loop.extent) >= 2 for loop in synthetic)
    # The newly introduced `not cond` guards are also snapshots.
    assert len(_nodes(normalized, tirx.Bind)) >= 2

    restored = _restore(normalized)
    assert not [
        loop for loop in _nodes(restored, tirx.For) if "synthetic_while" in loop.annotations
    ]
    assert len(_nodes(restored, tirx.While)) == 2
    assert not _nodes(restored, tirx.AllocBuffer)
    assert any(
        isinstance(call.op, tvm.ir.Op) and call.op.name == "tl.loop_break"
        for call in _nodes(restored, tirx.Call)
    )


def test_synthetic_while_bound_is_checked_lcm_and_not_pipelined():
    i = tirx.Var("i", "int32")
    j = tirx.Var("j", "int32")
    body = tirx.SeqStmt(
        [
            tirx.For(i, 0, 2, tirx.ForKind.SERIAL, tirx.Evaluate(0), annotations={"num_stages": 3}),
            tirx.For(j, 0, 2, tirx.ForKind.SERIAL, tirx.Evaluate(0), annotations={"num_stages": 4}),
        ]
    )
    normalized = _normalize(_kernel(tirx.While(True, body)))
    (loop,) = [node for node in _nodes(normalized, tirx.For) if "synthetic_while" in node.annotations]
    assert int(loop.extent) == 12
    assert "num_stages" not in loop.annotations

    overflowing = tirx.For(
        i,
        0,
        2,
        tirx.ForKind.SERIAL,
        tirx.Evaluate(0),
        annotations={"num_stages": 48},
    )
    with pytest.raises(Exception, match="overflows int64"):
        _normalize(_kernel(tirx.While(True, overflowing)))


@pytest.mark.parametrize("opaque", ["parallel", "block", "task"])
def test_while_inside_opaque_leaf_is_not_rewritten(opaque):
    pred = tirx.decl_buffer((1,), "int32", name="pred", scope="shared.dyn")
    loop_var = tirx.Var("p", "int32")
    while_loop = tirx.While(pred[0] > 0, tirx.BufferStore(pred, 0, [0]))
    if opaque == "parallel":
        body = tirx.For(loop_var, 0, 2, tirx.ForKind.PARALLEL, while_loop)
    elif opaque == "block":
        body = tirx.SBlock([], [], [], "opaque_block", while_loop)
    else:
        body = tirx.AttrStmt({}, "tl.cuda_task", 1, while_loop)

    before = _kernel(body, buffers=[pred])
    after = _normalize(before)
    tvm.ir.assert_structural_equal(after["main"], before["main"])
    assert not [loop for loop in _nodes(after, tirx.For) if "synthetic_while" in loop.annotations]


def _restore_synthetic(body, loop_var, output=None):
    if output is None:
        output = body.buffer
    synthetic = tirx.For(
        loop_var,
        3,
        8,
        tirx.ForKind.SERIAL,
        body,
        annotations={"synthetic_while": 1},
    )
    func = tirx.PrimFunc(
        [output.data],
        synthetic,
        buffer_map={output.data: output},
    ).with_attr("global_symbol", "main")
    return _restore(tvm.IRModule.from_expr(func))["main"].body


def test_restore_preserves_residual_loop_var_with_initialized_counter():
    output = tirx.decl_buffer((16,), "int32", name="output")
    loop_var = tirx.Var("synthetic", "int32")
    body = _restore_synthetic(tirx.BufferStore(output, loop_var, [loop_var]), loop_var)

    assert isinstance(body, tirx.SeqStmt)
    allocation, initialize, loop = body.seq
    assert isinstance(allocation, tirx.AllocBuffer)
    assert isinstance(initialize, tirx.BufferStore)
    assert int(initialize.value) == 3
    assert isinstance(loop, tirx.While)

    restored_store, increment = loop.body.seq
    assert isinstance(restored_store.value, tirx.BufferLoad)
    assert restored_store.value.buffer.same_as(allocation.buffer)
    assert isinstance(restored_store.indices[0], tirx.BufferLoad)
    assert restored_store.indices[0].buffer.same_as(allocation.buffer)
    assert isinstance(increment, tirx.BufferStore)
    assert increment.buffer.same_as(allocation.buffer)


def test_restore_avoids_counter_when_loop_var_is_dead():
    output = tirx.decl_buffer((1,), "int32", name="output")
    loop_var = tirx.Var("synthetic", "int32")
    body = _restore_synthetic(tirx.BufferStore(output, 7, [0]), loop_var)
    assert isinstance(body, tirx.While)
    assert isinstance(body.body, tirx.BufferStore)


def test_restore_simplifies_version_indices_before_allocating_counter():
    output = tirx.decl_buffer((2,), "int32", name="output")
    loop_var = tirx.Var("synthetic", "int32")
    inner_var = tirx.Var("inner", "int32")
    version_index = tirx.indexmod(loop_var * 4 + inner_var, 2)
    scheduled = tirx.For(
        inner_var,
        0,
        4,
        tirx.ForKind.SERIAL,
        tirx.BufferStore(output, 7, [version_index]),
    )

    body = _restore_synthetic(scheduled, loop_var, output)
    assert isinstance(body, tirx.While)
    assert isinstance(body.body, tirx.For)
    tvm.ir.assert_structural_equal(body.body.body.indices[0], tirx.indexmod(inner_var, 2))
