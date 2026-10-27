import tilelang
import pytest
from tilelang import tvm
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.tirx.stmt_functor import post_order_visit


def test_unroll_decl_buffer_defs_are_fresh():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((16,), "float32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                A_flat = T.decl_buffer((16,), "float32", data=A.data)
                A_flat[0] = T.float32(i)

    after = tilelang.transform.UnrollLoop()(Before)
    body = after["main"].body

    decls = [stmt for stmt in body.seq if isinstance(stmt, tvm.tirx.DeclBuffer)]
    stores = [stmt for stmt in body.seq if isinstance(stmt, tvm.tirx.BufferStore)]

    assert len(decls) == 2
    assert len(stores) == 2
    assert not decls[0].buffer.same_as(decls[1].buffer)
    assert decls[0].buffer.data.same_as(decls[1].buffer.data)
    assert stores[0].buffer.same_as(decls[0].buffer)
    assert stores[1].buffer.same_as(decls[1].buffer)


def test_unroll_explicit_loops_only():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((4,), "int32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                A[i] = i
            for j in T.unroll(2, annotations={"pragma_unroll_explicit": False}):
                A[j + 2] = j

    after = tilelang.transform.UnrollLoop()(Before)
    body = after["main"].body

    assert isinstance(body, tvm.tirx.SeqStmt)
    assert isinstance(body.seq[0], tvm.tirx.BufferStore)
    assert isinstance(body.seq[1], tvm.tirx.BufferStore)
    assert isinstance(body.seq[2], tvm.tirx.For)
    assert body.seq[2].kind == tvm.tirx.ForKind.UNROLLED


def test_unroll_explicit_loops_preserves_dynamic_and_factor_loops():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(n: T.int32, A: T.Buffer((16,), "int32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                A[i] = i
            for j in T.unroll(n):
                A[j] = j
            for k in T.unroll(8, annotations={"pragma_unroll_factor": 4}):
                A[k + 8] = k

    after = tilelang.transform.UnrollLoop()(Before)
    loops = []
    post_order_visit(
        after["main"].body,
        lambda node: loops.append(node) if isinstance(node, tvm.tirx.For) else None,
    )

    assert [loop.loop_var.name for loop in loops] == ["j", "k"]
    assert loops[0].kind == tvm.tirx.ForKind.UNROLLED
    assert int(loops[1].annotations["pragma_unroll_factor"]) == 4


def test_explicit_unroll_rejects_break_targeting_loop():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((2,), "int32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                A[i] = i
                if A[i] != 0:
                    T.evaluate(T.call_intrin("handle", tvm.tirx.op.Op.get("tl.loop_break")))

    with pytest.raises(ValueError, match="cannot be fully expanded"):
        tilelang.transform.UnrollLoop()(Before)


def test_non_explicit_unroll_allows_break():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((2,), "int32")):
            for i in T.unroll(2):
                A[i] = i
                if A[i] != 0:
                    T.evaluate(T.call_intrin("handle", tvm.tirx.op.Op.get("tl.loop_break")))

    after = tilelang.transform.UnrollLoop()(Before)
    assert isinstance(after["main"].body, tvm.tirx.For)
    assert after["main"].body.kind == tvm.tirx.ForKind.UNROLLED


def test_explicit_unroll_allows_break_targeting_nested_loop():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((4,), "int32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                for j in T.serial(2):
                    A[i * 2 + j] = j
                    if A[i * 2 + j] != 0:
                        T.evaluate(T.call_intrin("handle", tvm.tirx.op.Op.get("tl.loop_break")))

    after = tilelang.transform.UnrollLoop()(Before)
    loops = []
    post_order_visit(
        after["main"].body,
        lambda node: loops.append(node) if isinstance(node, tvm.tirx.For) else None,
    )

    assert len(loops) == 2
    assert all(loop.kind == tvm.tirx.ForKind.SERIAL for loop in loops)


def _z3_skip_unroll_module(*, kind, explicit, extent=2, step=None, nested=False):
    tir = tvm.tirx
    output = tir.decl_buffer((16,), "int32", name="skip_unroll_output")
    i = tir.Var("i", "int32")
    annotations = {} if explicit is None else {"pragma_unroll_explicit": explicit}
    inner = tir.For(i, 1 if step is not None else 0, extent, kind, tir.BufferStore(output, i, [i]), annotations=annotations, step=step)
    body = inner
    if nested:
        j = tir.Var("retained_outer", "int32")
        body = tir.For(j, 0, 2, tir.ForKind.SERIAL, inner)
    func = tir.PrimFunc([output.data], body, buffer_map={output.data: output}).with_attr("global_symbol", "main")
    return tvm.IRModule.from_expr(func)


@pytest.mark.parametrize("explicit", [True, False, None], ids=["true", "false", "absent"])
@pytest.mark.parametrize("kind", [tvm.tirx.ForKind.SERIAL, tvm.tirx.ForKind.UNROLLED], ids=["serial", "unrolled"])
def test_skip_z3_atomic_expands_only_explicit_unrolled_loops(kind, explicit):
    before = _z3_skip_unroll_module(kind=kind, explicit=explicit)
    after = tilelang.transform.UnrollLoopSkipZ3Atomic()(before)
    should_expand = kind == tvm.tirx.ForKind.UNROLLED and explicit is True
    if should_expand:
        assert isinstance(after["main"].body, tvm.tirx.SeqStmt)
        assert [int(stmt.value) for stmt in after["main"].body.seq] == [0, 1]
    else:
        tvm.ir.assert_structural_equal(after["main"], before["main"])


def test_skip_z3_atomic_nested_explicit_and_nonunit_step():
    before = _z3_skip_unroll_module(
        kind=tvm.tirx.ForKind.UNROLLED,
        explicit=True,
        extent=5,
        step=tvm.tirx.IntImm("int32", 2),
        nested=True,
    )
    after = tilelang.transform.UnrollLoopSkipZ3Atomic()(before)
    outer = after["main"].body
    assert isinstance(outer, tvm.tirx.For)
    assert outer.kind == tvm.tirx.ForKind.SERIAL
    assert isinstance(outer.body, tvm.tirx.SeqStmt)
    assert [int(stmt.value) for stmt in outer.body.seq] == [1, 3, 5]


@pytest.mark.parametrize("explicit", [False, None], ids=["false", "absent"])
def test_skip_z3_atomic_dynamic_nonexplicit_loop_is_retained(explicit):
    before = _z3_skip_unroll_module(kind=tvm.tirx.ForKind.UNROLLED, explicit=explicit, extent=tvm.tirx.Var("n", "int32"))
    after = tilelang.transform.UnrollLoopSkipZ3Atomic()(before)
    tvm.ir.assert_structural_equal(after["main"], before["main"])


def test_skip_z3_atomic_dynamic_explicit_loop_is_an_error():
    before = _z3_skip_unroll_module(kind=tvm.tirx.ForKind.UNROLLED, explicit=True, extent=tvm.tirx.Var("n", "int32"))
    with pytest.raises(Exception, match="constant loop extent"):
        tilelang.transform.UnrollLoopSkipZ3Atomic()(before)


def test_skip_z3_atomic_task_body_is_structurally_unchanged():
    tir = tvm.tirx
    output = tir.decl_buffer((16,), "int32", name="atomic_output")
    i = tir.Var("atomic_i", "int32")
    inner = tir.For(
        i,
        0,
        2,
        tir.ForKind.UNROLLED,
        tir.BufferStore(output, i, [i]),
        annotations={"pragma_unroll_explicit": True},
    )
    task = tir.AttrStmt({}, "tl.cuda_z3_task", tir.IntImm("int32", 1), inner)
    func = tir.PrimFunc([output.data], task, buffer_map={output.data: output}).with_attr("global_symbol", "main")
    before = tvm.IRModule.from_expr(func)
    after = tilelang.transform.UnrollLoopSkipZ3Atomic()(before)
    tvm.ir.assert_structural_equal(after["main"], before["main"])


def test_skip_z3_atomic_freshens_buffer_and_var_definitions():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((16,), "int32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                value: T.int32 = i + 1
                A_flat = T.decl_buffer((16,), "int32", data=A.data)
                A_flat[i] = value

    after = tilelang.transform.UnrollLoopSkipZ3Atomic()(Before)
    binds = [stmt for stmt in after["main"].body.seq if isinstance(stmt, tvm.tirx.Bind)]
    decls = [stmt for stmt in after["main"].body.seq if isinstance(stmt, tvm.tirx.DeclBuffer)]
    assert len(binds) == len(decls) == 2
    assert not binds[0].var.same_as(binds[1].var)
    assert not decls[0].buffer.same_as(decls[1].buffer)


if __name__ == "__main__":
    test_unroll_decl_buffer_defs_are_fresh()
    test_unroll_explicit_loops_only()
    test_unroll_explicit_loops_preserves_dynamic_and_factor_loops()
