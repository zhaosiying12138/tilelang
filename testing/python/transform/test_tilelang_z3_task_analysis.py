"""Host-only tests for CUDA Z3 task/cost inference.

No CUDA kernel is compiled or launched. Native-pass tests use the host TIR and
Python FFI only; the integration test calls the real CPU Z3 solver.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tvm_ffi

import tilelang as tl
from tilelang import tvm
import tilelang.language as T  # noqa: F401 - load the language surface.
from tilelang.cuda import task_cost
from tilelang.transform.pass_config import normalize_pass_configs
import tilelang.testing
from tvm.tirx.stmt_functor import post_order_visit


_CONFIG = "tl.enable_z3_task_analysis"
_Z3_CONFIG = "tl.enable_z3_pipeline_planning"
_ESTIMATOR = "tl.cuda.estimate_task"
_CAPACITIES = "tl.cuda.z3_memory_group_capacities"
_SOLVER = "tl.cuda.transform.z3_schedule_loop_python"
_SM80 = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})


def _path(**overrides):
    value = {
        "base_cycles": 2,
        "bandwidth_bytes_per_cycle": 16,
        "descriptor_cycles": 1,
        "issue_cycles": 1,
        "min_ii": 1,
        "transaction_bytes": 16,
        "strided_bandwidth_bytes_per_cycle": 8,
        "unknown_bandwidth_bytes_per_cycle": 4,
    }
    value.update(overrides)
    return value


_HARDWARE = {
    "schema": "tilelang.cuda.hardware-spec",
    "device": {
        "compute_capability": "12.0",
        "shared_memory_per_block_optin_bytes": 131072,
    },
}
_CALIBRATION = {
    "status": "measured",
    "arch": "sm120",
    "clock_hz": 1_000_000_000,
    "sm_count": 1,
    "memory_paths": {
        "global->shared": _path(),
        "global->register": _path(),
        "shared->register": _path(),
        "register->shared": _path(),
        "register->global": _path(),
        "shared->global": _path(),
    },
    "gemm_base_cycles": 2,
    "gemm_min_ii": 1,
    "gemm_pipeline_depth": 1,
    "gemm_throughput_ops_per_cycle": {
        "f16": 1024,
        "f16:f32": 1024,
        "bf16": 1024,
        "bf16:f32": 1024,
    },
    "vector_throughput_ops_per_cycle": {
        "add": 16,
        "sub": 16,
        "mul": 16,
        "fma": 16,
        "div": 4,
        "mod": 4,
        "min_max": 16,
        "cmp": 16,
        "logic": 16,
        "special": 8,
        "exp2": 16,
        "unknown": 1,
    },
    "vector_bandwidth_bytes_per_cycle": 64,
    "scalar_cycles": 1,
}


@pytest.fixture(autouse=True)
def _restore_ffi():
    names = (_ESTIMATOR, _CAPACITIES, _SOLVER)
    previous = {name: tvm_ffi.get_global_func(name, allow_missing=True) for name in names}
    yield
    for name in names:
        if tvm_ffi.get_global_func(name, allow_missing=True) is not None:
            tvm_ffi.remove_global_func(name)
        if previous[name] is not None:
            tvm_ffi.register_global_func(name, f=previous[name], override=True)


def _imms(values):
    return [tvm.tirx.IntImm("int64", value) for value in values]


def _nested(groups):
    return [_imms(group) for group in groups]


def _complete_annotations(latencies, iis=None, resource_flags=None):
    n = len(latencies)
    if iis is None:
        iis = [1] * n
    if resource_flags is None:
        resource_flags = [0] * n
    return {
        "tl_z3_pipeline": True,
        "tl_z3_pipeline_latencies": _imms(latencies),
        "tl_z3_pipeline_iis": _imms(iis),
        "tl_z3_pipeline_resource_flags": _imms(resource_flags),
        "tl_z3_pipeline_data_deps": [],
        "tl_z3_pipeline_resource_deps": [],
        "tl_z3_pipeline_owner_exclusion_deps": [],
        "tl_z3_pipeline_pipe_order_deps": [],
        "tl_z3_pipeline_stage_order_deps": [],
        "tl_z3_pipeline_buffer_sizes": [],
        "tl_z3_pipeline_memory_groups": [],
        "tl_z3_pipeline_buffer_vars": [],
    }


def _make_store_func(statements, buffers, annotations, *, extra_params=(), loop_var=None):
    tir = tvm.tirx
    k = tir.Var("k", "int32") if loop_var is None else loop_var
    loop = tir.For(
        k,
        0,
        4,
        tir.ForKind.SERIAL,
        tir.SeqStmt(statements) if len(statements) > 1 else statements[0],
        annotations={**annotations, "num_stages": tir.IntImm("int32", 2)},
    )
    func = tir.PrimFunc(
        [*extra_params, *[buffer.data for buffer in buffers]],
        loop,
        buffer_map={buffer.data: buffer for buffer in buffers},
    )
    return func.with_attr("global_symbol", "main")


def _bound_module(func):
    return tvm.tirx.transform.BindTarget(_SM80)(tvm.IRModule.from_expr(func))


def _infer_pass():
    return tvm_ffi.get_global_func("tl.transform.InferZ3PipelineProblem")()


def _run_infer(func_or_mod, *, config=None):
    mod = func_or_mod if isinstance(func_or_mod, tvm.IRModule) else _bound_module(func_or_mod)
    pass_config = {_CONFIG: True} if config is None else config
    with tvm.transform.PassContext(config=pass_config):
        return _infer_pass()(mod)


def _find_pipeline_loop(func):
    loops = []

    def visit(node):
        if isinstance(node, tvm.tirx.For) and "num_stages" in node.annotations:
            loops.append(node)

    post_order_visit(func.body, visit)
    assert len(loops) == 1
    return loops[0]


def _ints(values):
    return [int(value) for value in values]


def _nested_ints(values):
    return [_ints(group) for group in values]


def _install_model_mock():
    calls = []

    def estimator(features, hardware, calibration, require_measured=False):
        calls.append(features)
        assert hardware == "mock-hardware.json"
        assert calibration == "mock-calibration.json"
        return task_cost.estimate_task(features, _HARDWARE, _CALIBRATION, require_measured)

    tvm_ffi.register_global_func(_ESTIMATOR, f=estimator, override=True)
    tvm_ffi.register_global_func(
        _CAPACITIES,
        f=lambda hardware: task_cost.memory_group_capacities(_HARDWARE),
        override=True,
    )
    return calls


def _mock_paths():
    return {
        "tl_z3_pipeline_hardware": "mock-hardware.json",
        "tl_z3_pipeline_calibration": "mock-calibration.json",
    }


def _write_specs(tmp_path: Path):
    hardware = tmp_path / "hardware.json"
    calibration = tmp_path / "calibration.json"
    hardware.write_text(json.dumps(_HARDWARE), encoding="utf-8")
    calibration.write_text(json.dumps(_CALIBRATION), encoding="utf-8")
    return hardware, calibration


def test_default_noop_and_complete_problem_is_idempotent():
    tir = tvm.tirx
    a = tir.decl_buffer((4,), "float32", name="A")
    annotations = _complete_annotations([3])
    k = tir.Var("k", "int32")
    func = _make_store_func([tir.BufferStore(a, 1.0, [k])], [a], annotations, loop_var=k)
    before = _bound_module(func)

    tvm_ffi.register_global_func(
        _ESTIMATOR,
        f=lambda *args: (_ for _ in ()).throw(RuntimeError("must not estimate")),
        override=True,
    )
    after = _run_infer(before, config={})
    assert after["main"].same_as(before["main"])
    assert _CONFIG not in normalize_pass_configs({})

    after = _run_infer(before)
    assert after["main"].same_as(before["main"])
    without_marker = dict(annotations)
    without_marker.pop("tl_z3_pipeline")
    unannotated = _make_store_func([tir.BufferStore(a, 1.0, [k])], [a], without_marker, loop_var=k)
    before_unannotated = _bound_module(unannotated)
    after_unannotated = _run_infer(before_unannotated)
    assert after_unannotated["main"].same_as(before_unannotated["main"])


def test_missing_ii_is_filled_without_overwriting_explicit_fields():
    tir = tvm.tirx
    a = tir.decl_buffer((4,), "float32", name="A")
    k = tir.Var("k", "int32")
    annotations = _complete_annotations([77], resource_flags=[2])
    annotations.pop("tl_z3_pipeline_iis")
    annotations.update(_mock_paths())
    func = _make_store_func([tir.BufferStore(a, 1.0, [k])], [a], annotations, loop_var=k)
    calls = []

    def estimator(features, hardware, calibration, require_measured=False):
        calls.append(features)
        return {
            "latency": 9,
            "ii": 4,
            "terms": {"resource_flags": 16},
            "resource_flags": 16,
            "reliability": "estimated",
            "fallback_used": ["mock-fallback"],
        }

    tvm_ffi.register_global_func(_ESTIMATOR, f=estimator, override=True)
    inferred = _run_infer(func)
    loop = _find_pipeline_loop(inferred["main"])
    assert _ints(loop.annotations["tl_z3_pipeline_latencies"]) == [77]
    assert _ints(loop.annotations["tl_z3_pipeline_iis"]) == [4]
    assert _ints(loop.annotations["tl_z3_pipeline_resource_flags"]) == [2]
    assert list(loop.annotations["tl_z3_pipeline_cost_reliability"]) == ["estimated"]
    assert [list(items) for items in loop.annotations["tl_z3_pipeline_cost_fallback_used"]] == [["mock-fallback"]]
    assert len(calls) == 1

    again = _run_infer(inferred)
    assert again["main"].same_as(inferred["main"])
    assert len(calls) == 1


def test_nested_replayability_is_dependency_closed_and_scalar_raw_war_are_inferred():
    tir = tvm.tirx
    a = tir.decl_buffer((8,), "int32", name="closure_A")
    readonly = tir.decl_buffer((8,), "int32", name="closure_readonly")
    b = tir.decl_buffer((8,), "int32", name="closure_B")
    k = tir.Var("k", "int32")
    scheduled = tir.Var("scheduled_scalar", "int32")
    nested = tir.Var("nested_scalar", "int32")
    literal = tir.Var("literal_scalar", "int32")
    read_only_value = tir.Var("read_only_scalar", "int32")
    affine = tir.Var("affine_scalar", "int32")
    statements = [
        tir.BufferStore(a, 1, [k]),
        # This Bind is scheduled because A is pipeline-written.
        tir.Bind(scheduled, tir.BufferLoad(a, [k]) + 1),
        # Although locally pure, this must not be replayed without its def.
        tir.Bind(nested, scheduled + 1),
        # Literal/affine and known read-only chains remain replayable.
        tir.Bind(literal, k + 1),
        tir.Bind(read_only_value, tir.BufferLoad(readonly, [k])),
        tir.Bind(affine, read_only_value + literal),
        tir.BufferStore(b, nested + affine, [k]),
    ]
    annotations = _complete_annotations([2, 3, 4, 5])
    annotations.pop("tl_z3_pipeline_data_deps")
    func = _make_store_func(statements, [a, readonly, b], annotations, loop_var=k)

    loop = _find_pipeline_loop(_run_infer(func)["main"])
    assert _ints(loop.annotations["tl_z3_pipeline_latencies"]) == [2, 3, 4, 5]
    dependencies = _nested_ints(loop.annotations["tl_z3_pipeline_data_deps"])
    assert [1, 2, 0, 3] in dependencies
    assert [2, 1, 1, 4] in dependencies
    assert [2, 3, 0, 4] in dependencies
    assert [3, 2, 1, 5] in dependencies
    # Replayable definitions have no solver task and therefore no scalar edge.
    assert not any(dep[0] > 3 or dep[1] > 3 for dep in dependencies)


def test_raw_war_waw_fixed_dependencies_follow_source_order():
    tir = tvm.tirx
    a = tir.decl_buffer((8,), "float32", name="A")
    b = tir.decl_buffer((8,), "float32", name="B")
    k = tir.Var("k", "int32")
    statements = [
        tir.BufferStore(a, 1.0, [k]),
        tir.BufferStore(b, tir.BufferLoad(a, [k]), [k]),
        tir.BufferStore(a, 2.0, [k]),
    ]
    annotations = _complete_annotations([3, 4, 5])
    annotations.pop("tl_z3_pipeline_data_deps")
    func = _make_store_func(statements, [a, b], annotations, loop_var=k)

    loop = _find_pipeline_loop(_run_infer(func)["main"])
    dependencies = _nested_ints(loop.annotations["tl_z3_pipeline_data_deps"])
    same_iteration = [dependency for dependency in dependencies if dependency[2] == 0]
    assert same_iteration == [
        [0, 1, 0, 3],  # RAW: A writer -> A reader.
        [0, 2, 0, 3],  # WAW: first A writer -> second A writer.
        [1, 2, 0, 4],  # WAR: A reader -> later A writer.
    ]
    assert dependencies == same_iteration


@pytest.mark.parametrize("index_mode", ["increasing", "decreasing"])
def test_distance_two_conflict_is_not_dropped_by_distance_one_reasoning(index_mode):
    tir = tvm.tirx
    a = tir.decl_buffer((8,), "float32", name="A")
    k = tir.Var("k", "int32")
    statements = [
        tir.BufferStore(a, 1.0, [k]),
        tir.BufferStore(a, 2.0, [k + 2 if index_mode == "increasing" else 6 - k]),
    ]
    annotations = _complete_annotations([3, 4])
    annotations.pop("tl_z3_pipeline_data_deps")
    func = _make_store_func(statements, [a], annotations, loop_var=k)

    loop = _find_pipeline_loop(_run_infer(func)["main"])
    dependencies = _nested_ints(loop.annotations["tl_z3_pipeline_data_deps"])
    assert [0, 1, 1, 3] in dependencies


def test_copy_gemm_parallel_vector_and_scalar_features_and_resources():
    annotations = {"tl_z3_pipeline": True, **_mock_paths()}

    @T.prim_func
    def main(
        A: T.Tensor((16, 16), "float16"),
        B: T.Tensor((16, 16), "float16"),
        Out: T.Tensor((16, 16), "float32"),
    ):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((16, 16), "float16")
            b_shared = T.alloc_shared((16, 16), "float16")
            c_fragment = T.alloc_fragment((16, 16), "float32")
            for _ in T.Pipelined(1, num_stages=2, annotations=annotations):
                T.copy(A, a_shared, coalesced_width=16)
                T.gemm(a_shared, b_shared, c_fragment)
                for i, j in T.Parallel(16, 16):
                    Out[i, j] = T.exp2(c_fragment[i, j]) + 1.0

    calls = _install_model_mock()
    loop = _find_pipeline_loop(_run_infer(main)["main"])
    assert len(calls) == 3
    copy = calls[0]["copies"][0]
    assert (copy["src_scope"], copy["dst_scope"]) == ("global", "shared")
    assert copy["bytes"] == 512
    assert copy["geometry"] == "coalesced"
    assert copy["width"] == 32
    gemm = calls[1]["gemms"][0]
    assert (gemm["M"], gemm["N"], gemm["K"], gemm["dtype"], gemm["acc"]) == (16, 16, 16, "f16", "f32")
    vector = calls[2]["vector"]
    assert vector["add"] == 256
    assert vector["exp2"] == 256
    assert vector["read_bytes"] == 1024
    assert _ints(loop.annotations["tl_z3_pipeline_resource_flags"]) == [1, 8, 50]
    assert _ints(loop.annotations["tl_z3_pipeline_buffer_sizes"]) == [512]
    assert len(loop.annotations["tl_z3_pipeline_buffer_vars"]) == 1
    assert _nested_ints(loop.annotations["tl_z3_pipeline_memory_groups"]) == [[131072, 0]]

    tir = tvm.tirx
    a = tir.decl_buffer((4,), "float32", name="scalar_A")
    b = tir.decl_buffer((4,), "float32", name="scalar_B")
    k = tir.Var("k", "int32")
    scalar_annotations = {"tl_z3_pipeline": True, **_mock_paths()}
    scalar_func = _make_store_func(
        [tir.BufferStore(b, tir.BufferLoad(a, [k]) + 1.0, [k])],
        [a, b],
        scalar_annotations,
        loop_var=k,
    )
    scalar_calls = _install_model_mock()
    scalar_loop = _find_pipeline_loop(_run_infer(scalar_func)["main"])
    assert scalar_calls[0]["scalar"] == 1
    assert _ints(scalar_loop.annotations["tl_z3_pipeline_resource_flags"]) == [67]


def test_resource_pairs_are_derived_from_effective_masks():
    tir = tvm.tirx
    a = tir.decl_buffer((4,), "float32", name="A")
    k = tir.Var("k", "int32")
    annotations = _complete_annotations([1, 1, 1], resource_flags=[1, 3, 2])
    annotations.pop("tl_z3_pipeline_resource_deps")
    func = _make_store_func(
        [tir.BufferStore(a, 1.0, [k]), tir.BufferStore(a, 2.0, [k]), tir.BufferStore(a, 3.0, [k])],
        [a],
        annotations,
        loop_var=k,
    )
    loop = _find_pipeline_loop(_run_infer(func)["main"])
    assert _nested_ints(loop.annotations["tl_z3_pipeline_resource_deps"]) == [[0, 1], [1, 2]]


def test_non_static_versioned_buffer_size_is_an_error_not_zero():
    tir = tvm.tirx
    n = tir.Var("n", "int32")
    shared = tir.decl_buffer((n,), "float32", name="dynamic_shared", scope="shared")
    k = tir.Var("k", "int32")
    annotations = _complete_annotations([1])
    annotations.pop("tl_z3_pipeline_buffer_sizes")
    annotations.pop("tl_z3_pipeline_buffer_vars")
    annotations.pop("tl_z3_pipeline_memory_groups")
    func = _make_store_func(
        [tir.BufferStore(shared, 1.0, [k])], [shared], annotations, extra_params=[n], loop_var=k
    )
    with pytest.raises(Exception, match="buffer shape extent"):
        _run_infer(func)


def test_opaque_side_effect_is_not_silently_dropped_from_dependencies():
    tir = tvm.tirx
    a = tir.decl_buffer((4,), "float32", name="A")
    k = tir.Var("k", "int32")
    annotations = _complete_annotations([1])
    annotations.pop("tl_z3_pipeline_data_deps")
    opaque = tir.Evaluate(tir.call_extern("int32", "opaque_side_effect"))
    func = _make_store_func([opaque, tir.BufferStore(a, 1.0, [k])], [a], annotations, loop_var=k)
    with pytest.raises(Exception, match="opaque|stateful"):
        _run_infer(func)


def _async_annotations(manual, stages=(0, 0)):
    annotations = _complete_annotations([1, 1])
    annotations.pop("tl_z3_pipeline_data_deps")
    annotations["software_pipeline_async_producers"] = _imms([1, 0])
    annotations["software_pipeline_async_producer_groups"] = _imms([0, -1])
    if manual:
        annotations["tl_z3_pipeline_manual_schedule"] = True
        annotations["tl_z3_pipeline_manual_stages"] = _imms(list(stages))
    return annotations


def _async_func(manual, stages=(0, 0)):
    tir = tvm.tirx
    shared = tir.decl_buffer((4,), "float32", name="async_shared", scope="shared")
    out = tir.decl_buffer((4,), "float32", name="async_out")
    k = tir.Var("k", "int32")
    return _make_store_func(
        [
            tir.BufferStore(shared, 1.0, [k]),
            tir.BufferStore(out, tir.BufferLoad(shared, [k]), [k]),
        ],
        [shared, out],
        _async_annotations(manual, stages),
        loop_var=k,
    )


def test_async_dependencies_allow_automatic_and_ordered_manual_stages():
    automatic = _find_pipeline_loop(_run_infer(_async_func(False))["main"])
    assert _nested_ints(automatic.annotations["tl_z3_pipeline_async_same_stage_deps"]) == [[0, 1]]
    same_stage = _find_pipeline_loop(_run_infer(_async_func(True))["main"])
    assert _nested_ints(same_stage.annotations["tl_z3_pipeline_async_same_stage_deps"]) == [[0, 1]]
    forward = _find_pipeline_loop(_run_infer(_async_func(True, (0, 1)))["main"])
    assert _nested_ints(forward.annotations["tl_z3_pipeline_async_same_stage_deps"]) == [[0, 1]]
    with pytest.raises(Exception, match="must not occur after consumer"):
        _run_infer(_async_func(True, (1, 0)))


def test_task_cost_ffi_is_strict_json_and_measured_only_is_controllable(tmp_path):
    hardware, calibration = _write_specs(tmp_path)
    features = {
        "copies": [
            {
                "src_scope": "global",
                "dst_scope": "shared",
                "bytes": 64,
                "geometry": "coalesced",
                "width": 0,
            }
        ]
    }
    ffi = tvm_ffi.get_global_func(_ESTIMATOR)
    result = ffi(features, str(hardware), str(calibration), False)
    assert set(result) == {"latency", "ii", "terms", "resource_flags", "reliability", "fallback_used"}
    assert (result["latency"], result["ii"], result["resource_flags"]) == (7, 5, 1)
    assert result["reliability"] == "measured"
    assert result["fallback_used"] == []
    assert json.loads(json.dumps(task_cost._plain(result), allow_nan=False))["terms"]["resource_flags"] == 1
    direct = task_cost.estimate_task(features, str(hardware), str(calibration), False)
    assert json.loads(json.dumps(direct, allow_nan=False))["terms"]["resource_flags"] == 1
    assert ffi(features, str(hardware), str(calibration), True)["latency"] == 7

    unmeasured = tmp_path / "unmeasured.json"
    data = dict(_CALIBRATION)
    data["status"] = "provided"
    unmeasured.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(Exception, match="Measured"):
        ffi(features, str(hardware), str(unmeasured), True)


def test_hardware_capacity_requires_explicit_optin_field():
    hardware = {
        "schema": "tilelang.cuda.hardware-spec",
        "device": {"compute_capability": "12.0"},
    }
    with pytest.raises(ValueError, match="shared_memory_per_block_optin_bytes"):
        task_cost.memory_group_capacities(hardware)


def test_inferred_scalar_and_buffer_dependencies_support_clone_augmented_scheduling():
    tir = tvm.tirx
    a = tir.decl_buffer((8,), "int32", name="inferred_clone_A")
    b = tir.decl_buffer((8,), "int32", name="inferred_clone_B")
    k = tir.Var("k", "int32")
    value = tir.Var("inferred_clone_value", "int32")
    statements = [
        tir.BufferStore(a, 1, [k]),
        tir.Bind(value, tir.BufferLoad(a, [k]) + 1),
        tir.BufferStore(b, value, [k]),
    ]
    annotations = _complete_annotations([1, 1, 1])
    annotations.pop("tl_z3_pipeline_data_deps")
    annotations.update(
        {
            "tl_z3_pipeline_manual_schedule": True,
            "tl_z3_pipeline_manual_stages": _imms([0, 0, 1]),
            "tl_z3_pipeline_enable_offset": True,
        }
    )
    func = _make_store_func(statements, [a, b], annotations, loop_var=k)
    inferred = _run_infer(func)

    solver = tvm_ffi.get_global_func(_SOLVER)
    calls = []
    results = []

    def tracked_solver(*args):
        calls.append(args)
        result = solver(*args)
        results.append(result)
        return result

    tvm_ffi.register_global_func(_SOLVER, f=tracked_solver, override=True)
    with tvm.transform.PassContext(config={_Z3_CONFIG: True}):
        planned = tl.transform.Z3PipelinePlanning()(inferred)
    loop = _find_pipeline_loop(planned["main"])
    assert len(calls) == 2
    assert _ints(calls[1][12]) == [0, 0, 1, 1]
    assert _ints(loop.annotations["tl_z3_pipeline_audit_starts"]) == _ints(results[-1][0])[:3]
    assert len(loop.annotations["tl_z3_pipeline_audit_starts"]) == 3
    plan = list(loop.annotations["tl_pipeline_scalar_clone_plan"])
    assert len(plan) == 1
    assert int(plan[0]["definition_task"]) == 1
    assert int(plan[0]["consumer_task"]) == 2
    assert int(plan[0]["stage"]) == 1


@pytest.mark.parametrize("alias_storage", [True, False])
def test_aliased_views_keep_clone_raw_war_constraints_and_versions(alias_storage):
    """Cross-iteration view aliases must not disappear from inference or clones."""
    tir = tvm.tirx
    written = tir.decl_buffer((8,), "int32", name="dependency_written_view", scope="shared")
    if alias_storage:
        read = tir.decl_buffer(
            (8,), "int32", name="dependency_aliased_view", data=written.data, scope="shared"
        )
    else:
        read = tir.decl_buffer((8,), "int32", name="dependency_disjoint_view")
    out = tir.decl_buffer((8,), "int32", name="dependency_view_out")
    k = tir.Var("k", "int32")
    value = tir.Var("dependency_view_value", "int32")
    statements = [
        tir.BufferStore(written, 1, [k]),
        # Reading the exact written Buffer keeps this Bind scheduled in both
        # cases. Its same-object W[k] accesses are disjoint across iterations,
        # so only the second view can introduce cross-iteration RAW/WAR here.
        tir.Bind(value, tir.BufferLoad(read, [k]) + tir.BufferLoad(written, [k])),
        tir.BufferStore(out, value, [k]),
    ]
    annotations = _complete_annotations([2, 3, 4])
    annotations.pop("tl_z3_pipeline_data_deps")
    annotations.pop("tl_z3_pipeline_buffer_sizes")
    annotations.pop("tl_z3_pipeline_buffer_vars")
    annotations.update(
        {
            "tl_z3_pipeline_manual_schedule": True,
            "tl_z3_pipeline_manual_stages": _imms([0, 0, 1]),
            "tl_z3_pipeline_enable_offset": True,
        }
    )
    buffers = [written, out] if alias_storage else [written, read, out]
    inferred = _run_infer(_make_store_func(statements, buffers, annotations, loop_var=k))
    inferred_loop = _find_pipeline_loop(inferred["main"])
    assert len(inferred_loop.annotations["tl_z3_pipeline_buffer_vars"]) == 1
    assert inferred_loop.annotations["tl_z3_pipeline_buffer_vars"][0].same_as(written.data)
    dependencies = _nested_ints(inferred_loop.annotations["tl_z3_pipeline_data_deps"])
    original_raw_war = ([0, 1, -1, 2], [1, 0, -1, 3])
    for dependency in original_raw_war:
        assert (list(dependency) in dependencies) == alias_storage

    solver = tvm_ffi.get_global_func(_SOLVER)
    calls = []

    def tracked_solver(*args):
        calls.append(args)
        return solver(*args)

    tvm_ffi.register_global_func(_SOLVER, f=tracked_solver, override=True)
    with tvm.transform.PassContext(config={_Z3_CONFIG: True}):
        planned = tl.transform.Z3PipelinePlanning()(inferred)
    loop = _find_pipeline_loop(planned["main"])
    assert len(calls) == 2
    assert _ints(calls[-1][12]) == [0, 0, 1, 1]
    augmented_dependencies = _nested_ints(calls[-1][4])
    clone_raw_war = ([0, 3, -1, 2], [3, 0, -1, 3])
    for dependency in (*original_raw_war, *clone_raw_war):
        assert (list(dependency) in augmented_dependencies) == alias_storage

    plan = list(loop.annotations["tl_pipeline_scalar_clone_plan"])
    assert len(plan) == 1
    assert int(plan[0]["definition_task"]) == 1
    assert int(plan[0]["consumer_task"]) == 2
    versions = int(loop.annotations["tl_pipeline_buffer_versions"][written.data])
    if alias_storage:
        assert versions >= 2
    else:
        assert versions == 1


def test_real_z3_integration_after_inference(tmp_path):
    hardware, calibration = _write_specs(tmp_path)
    annotations = {
        "tl_z3_pipeline": True,
        "tl_z3_pipeline_hardware": str(hardware),
        "tl_z3_pipeline_calibration": str(calibration),
        "tl_z3_pipeline_manual_schedule": True,
        "tl_z3_pipeline_manual_stages": _imms([0, 1]),
        "tl_z3_pipeline_enable_offset": True,
    }

    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((16,), "float32")
            for k in T.Pipelined(4, num_stages=2, annotations=annotations):
                T.copy(A[k * 16 : (k + 1) * 16], a_shared, coalesced_width=16)
                for i in T.Parallel(16):
                    B[k * 16 + i] = a_shared[i] + 1.0

    inferred = _run_infer(main)
    with tvm.transform.PassContext(config={_Z3_CONFIG: True}):
        planned = tl.transform.Z3PipelinePlanning()(inferred)
    loop = _find_pipeline_loop(planned["main"])
    assert "tl_z3_pipeline" not in loop.annotations
    assert _ints(loop.annotations["tl_pipeline_stage"]) == [0, 1]
    assert len(loop.annotations["tl_z3_pipeline_audit_starts"]) == 2
    assert int(loop.annotations["tl_z3_pipeline_audit_ii"]) > 0
    assert len(loop.annotations["tl_pipeline_buffer_versions"]) == 1


def test_regular_package_import_registers_task_cost_ffi_without_explicit_import():
    import subprocess
    import sys

    code = (
        "import tilelang, tvm_ffi; "
        "assert tvm_ffi.get_global_func('tl.cuda.estimate_task'); "
        "assert tvm_ffi.get_global_func('tl.cuda.z3_memory_group_capacities')"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=60)


def test_main_cuda_pipeline_infers_before_z3_planning(tmp_path):
    hardware, calibration = _write_specs(tmp_path)
    annotations = {
        "tl_z3_pipeline": True,
        "tl_z3_pipeline_hardware": str(hardware),
        "tl_z3_pipeline_calibration": str(calibration),
        "tl_z3_pipeline_manual_schedule": True,
        "tl_z3_pipeline_manual_stages": _imms([0, 1]),
        "tl_z3_pipeline_enable_offset": True,
    }

    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((16,), "float32")
            for k in T.Pipelined(4, num_stages=2, annotations=annotations):
                T.copy(A[k * 16 : (k + 1) * 16], a_shared, coalesced_width=16)
                for i in T.Parallel(16):
                    B[k * 16 + i] = a_shared[i] + 1.0

    from tilelang.cuda.pipeline import CUDAPassPipelineBodyPrologue

    events = []
    estimator = tvm_ffi.get_global_func(_ESTIMATOR)
    solver = tvm_ffi.get_global_func(_SOLVER)

    def tracked_estimator(*args):
        events.append("estimate")
        return estimator(*args)

    def tracked_solver(*args):
        events.append("solve")
        return solver(*args)

    tvm_ffi.register_global_func(_ESTIMATOR, f=tracked_estimator, override=True)
    tvm_ffi.register_global_func(_SOLVER, f=tracked_solver, override=True)
    config = {_CONFIG: True, _Z3_CONFIG: True}
    with _SM80, tvm.transform.PassContext(config=config):
        lowered = CUDAPassPipelineBodyPrologue(_bound_module(main), _SM80)

    assert events[:2] == ["estimate", "estimate"]
    assert "solve" in events
    assert set(events[events.index("solve") :]) == {"solve"}
    audits = []

    def visit(node):
        if isinstance(node, tvm.tirx.For) and "tl_z3_pipeline_audit_starts" in node.annotations:
            audits.append(node)
        assert not isinstance(node, tvm.tirx.For) or "tl_z3_pipeline" not in node.annotations

    post_order_visit(lowered["main"].body, visit)
    assert audits
    assert int(audits[0].annotations["tl_z3_pipeline_audit_ii"]) > 0


if __name__ == "__main__":
    tilelang.testing.main()
