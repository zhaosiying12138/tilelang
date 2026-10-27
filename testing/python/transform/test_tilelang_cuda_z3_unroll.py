"""PASS-025 host oracles for explicit unrolling at the CUDA Z3 boundary."""

from __future__ import annotations

from collections.abc import Callable

import pytest

import tilelang
from tilelang import tvm
from tilelang.cuda.pipeline import CUDAPassPipelineBodyPrologue
import tilelang.language as T  # noqa: F401  # Register the tile-op language surface.
from tilelang.language.annotations import z3_pipeline_annotations
from tvm.ir.instrument import pass_instrument
from tvm.tirx.stmt_functor import post_order_visit


_CONFIG = "tl.enable_z3_pipeline_planning"
_ANALYSIS_CONFIG = "tl.enable_z3_task_analysis"
_INFER = "tl.InferZ3PipelineProblem"
_PLAN = "tl.Z3PipelinePlanning"
_UNROLL = "tl.UnrollLoopSkipZ3Atomic"
_NORMALIZE = "tl.NormalizeZ3ScheduleUnits"
_SIMPLIFY = "tl.Simplify"
_SM80 = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})


@pass_instrument
class _Z3InputRecorder:
    """Capture the actual module consumed by each relevant native pass."""

    def __init__(self):
        self.events: list[str] = []
        self.z3_inputs: dict[str, tvm.tirx.Stmt] = {}

    def run_before_pass(self, mod, info):
        name = str(info.name)
        self.events.append(name)
        if name in (_INFER, _PLAN):
            self.z3_inputs[name] = mod["main"].body


def _make_module(
    task_count: int,
    body_factory: Callable[[tvm.tirx.Var, tvm.tirx.Buffer], tvm.tirx.Stmt],
):
    tir = tvm.tirx
    output = tir.decl_buffer((32,), "int32", name="output")
    k = tir.Var("k", "int32")
    annotations = z3_pipeline_annotations(
        latencies=[1] * task_count,
        iis=[1] * task_count,
        resource_flags=[0] * task_count,
        manual_schedule=True,
        manual_stages=list(range(task_count)),
        enable_offset=task_count > 1,
    )
    annotations["num_stages"] = tir.IntImm("int32", 2)
    loop = tir.For(
        k,
        0,
        4,
        tir.ForKind.SERIAL,
        body_factory(k, output),
        annotations=annotations,
    )
    func = tir.PrimFunc(
        [output.data],
        loop,
        buffer_map={output.data: output},
    ).with_attr("global_symbol", "main")
    return tvm.tirx.transform.BindTarget(_SM80)(tvm.IRModule.from_expr(func))


def _run_prologue(mod):
    recorder = _Z3InputRecorder()
    config = {_CONFIG: True, _ANALYSIS_CONFIG: True}
    with _SM80, tvm.transform.PassContext(config=config, instruments=[recorder]):
        lowered = CUDAPassPipelineBodyPrologue(mod, _SM80)
    assert _INFER in recorder.z3_inputs
    assert _PLAN in recorder.z3_inputs
    return recorder, lowered


def _z3_loop(body):
    matches = []

    def visit(node):
        if isinstance(node, tvm.tirx.For) and "tl_z3_pipeline" in node.annotations:
            matches.append(node)

    post_order_visit(body, visit)
    assert len(matches) == 1
    return matches[0]


def _top_level_tasks(loop):
    """Mirror Z3's SeqStmt-only flattening for these declaration-free bodies."""
    def flatten(stmt):
        if isinstance(stmt, tvm.tirx.SeqStmt):
            return [task for child in stmt.seq for task in flatten(child)]
        return [stmt]

    return flatten(loop.body)


def _nested_loops(stmt):
    loops = []
    post_order_visit(
        stmt,
        lambda node: loops.append(node) if isinstance(node, tvm.tirx.For) else None,
    )
    return loops


def test_explicit_unroll_materializes_tasks_before_z3_inference_and_planning():
    tir = tvm.tirx

    def body(k, output):
        i = tir.Var("i", "int32")
        return tir.For(
            i,
            0,
            2,
            tir.ForKind.UNROLLED,
            tir.BufferStore(output, i, [k * 2 + i]),
            annotations={"pragma_unroll_explicit": True},
        )

    recorder, _ = _run_prologue(_make_module(2, body))

    normalize = recorder.events.index(_NORMALIZE)
    unroll = recorder.events.index(_UNROLL)
    simplify = recorder.events.index(_SIMPLIFY, unroll)
    infer = recorder.events.index(_INFER)
    plan = recorder.events.index(_PLAN)
    assert normalize < unroll < simplify < infer < plan

    inference_loop = _z3_loop(recorder.z3_inputs[_INFER])
    planning_loop = _z3_loop(recorder.z3_inputs[_PLAN])
    for loop in (inference_loop, planning_loop):
        tasks = _top_level_tasks(loop)
        assert len(tasks) == 2
        assert all(isinstance(task, tir.BufferStore) for task in tasks)
        assert [int(task.value) for task in tasks] == [0, 1]
        assert _nested_loops(loop.body) == []
        assert len(loop.annotations["tl_z3_pipeline_latencies"]) == 2
    tvm.ir.assert_structural_equal(inference_loop.body, planning_loop.body)


@pytest.mark.parametrize("explicit", [False, None], ids=["disabled", "absent"])
def test_non_explicit_unroll_remains_one_atomic_z3_task(explicit):
    tir = tvm.tirx
    original = []

    def body(k, output):
        i = tir.Var("i", "int32")
        annotations = {} if explicit is None else {"pragma_unroll_explicit": explicit}
        loop = tir.For(
            i,
            0,
            2,
            tir.ForKind.UNROLLED,
            tir.BufferStore(output, i, [k * 2 + i]),
            annotations=annotations,
        )
        original.append(loop)
        return loop

    recorder, _ = _run_prologue(_make_module(1, body))

    for pass_name in (_INFER, _PLAN):
        loop = _z3_loop(recorder.z3_inputs[pass_name])
        tasks = _top_level_tasks(loop)
        assert len(tasks) == 1
        retained = tasks[0]
        assert isinstance(retained, tir.For)
        assert retained.kind == tir.ForKind.UNROLLED
        assert len(loop.annotations["tl_z3_pipeline_latencies"]) == 1
        if explicit is None:
            assert "pragma_unroll_explicit" not in retained.annotations
        else:
            assert not retained.annotations["pragma_unroll_explicit"]
        tvm.ir.assert_structural_equal(retained, original[0])


def test_pre_z3_unroll_preserves_explicit_loop_inside_atomic_parallel_region():
    """A parallel region is one schedule task; its interior is not pre-Z3 unrolled."""
    tir = tvm.tirx
    original = []

    def body(_k, output):
        p = tir.Var("p", "int32")
        i = tir.Var("i", "int32")
        inner = tir.For(
            i,
            0,
            2,
            tir.ForKind.UNROLLED,
            tir.BufferStore(output, i, [p * 2 + i]),
            annotations={"pragma_unroll_explicit": True},
        )
        original.append(inner)
        return tir.For(p, 0, 2, tir.ForKind.PARALLEL, inner)

    recorder, _ = _run_prologue(_make_module(1, body))

    for pass_name in (_INFER, _PLAN):
        loop = _z3_loop(recorder.z3_inputs[pass_name])
        tasks = _top_level_tasks(loop)
        assert len(tasks) == 1
        region = tasks[0]
        assert isinstance(region, tir.For)
        assert region.kind == tir.ForKind.PARALLEL
        assert len(loop.annotations["tl_z3_pipeline_latencies"]) == 1
        assert isinstance(region.body, tir.For), (
            f"explicit unroll entered an atomic parallel region before {pass_name}"
        )
        tvm.ir.assert_structural_equal(region.body, original[0])
