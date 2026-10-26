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


def _bound_module(func):
    mod = tvm.IRModule.from_expr(func)
    return tvm.tirx.transform.BindTarget(_SM80)(mod)


def _run_z3(func, *, config=None):
    mod = _bound_module(func)
    pass_config = {_CONFIG: True} if config is None else config
    with tvm.transform.PassContext(config=pass_config):
        return tl.transform.Z3PipelinePlanning()(mod)


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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"owner_exclusion_deps": [(0, 1, 1, 1)]}, "owner_exclusion_deps"),
        ({"stage_order_deps": [(0, 1)]}, "stage_order_deps"),
    ],
)
def test_unexpressible_dependencies_fail_before_solver(overrides, message):
    func, _ = _make_loop(_annotations(**overrides))
    calls = _install_solver(([0, 0, 0, 0], [], 8))

    with pytest.raises(Exception, match=message):
        _run_z3(func)
    assert calls == []


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

    with tvm.transform.PassContext(config={_CONFIG: True, _AUTO_WS_CONFIG: "role_based"}):
        with pytest.raises(ValueError, match="cannot schedule the same annotated module"):
            CUDAPassPipelineBodyPrologue(mod, _SM80)
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
