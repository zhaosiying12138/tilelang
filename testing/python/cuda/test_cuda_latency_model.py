"""CPU-only formula tests for the CUDA task latency model.

The module is loaded by path so these tests never import native TileLang, TVM,
torch, CUDA, or a GPU backend.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import importlib.util
import json
from pathlib import Path
import sys

import pytest


_MODULE_PATH = Path(__file__).parents[3] / "tilelang" / "cuda" / "latency_model.py"
_SPEC = importlib.util.spec_from_file_location("_tilelang_cuda_latency_model_under_test", _MODULE_PATH)
lm = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = lm
_SPEC.loader.exec_module(lm)


def _path(**overrides):
    value = {
        "base_cycles": 10,
        "bandwidth_bytes_per_cycle": 64,
        "descriptor_cycles": 2,
        "issue_cycles": 1,
        "min_ii": 3,
        "transaction_bytes": 16,
        "strided_bandwidth_bytes_per_cycle": 0,
        "unknown_bandwidth_bytes_per_cycle": 4,
    }
    value.update(overrides)
    return value


def _calibration_data():
    return {
        "status": "measured",
        "arch": "sm120",
        "clock_hz": 1_000_000_000,
        "sm_count": 1,
        "memory_paths": {
            "global->shared": _path(),
            "shared->register": _path(
                base_cycles=2,
                bandwidth_bytes_per_cycle=32,
                descriptor_cycles=1,
                issue_cycles=0,
                min_ii=1,
            ),
        },
        "gemm_base_cycles": 5,
        "gemm_min_ii": 1,
        "gemm_pipeline_depth": 1,
        "gemm_throughput_ops_per_cycle": {
            "bf16": 1024,
            "bf16:f32": 1024,
            "f16": 1024,
            "f16:f32": 1024,
            "tf32": 1024,
            "tf32:f32": 1024,
        },
        "vector_throughput_ops_per_cycle": {
            "add": 4,
            "sub": 4,
            "mul": 2,
            "fma": 4,
            "div": 1,
            "mod": 1,
            "min_max": 4,
            "cmp": 4,
            "logic": 4,
            "special": 2,
            "exp2": 8,
            "unknown": 1,
        },
        "vector_bandwidth_bytes_per_cycle": 64,
        "scalar_cycles": 1,
    }


def _params(**changes):
    data = _calibration_data()
    data.update(changes)
    return lm.Calibration.from_json(data).params


def _instruction_params(instruction, path="global->shared", **overrides):
    data = _calibration_data()
    data["memory_paths"][f"{path}@{instruction}"] = _path(**overrides)
    return _params(**data)


def _copy(payload=130, **kwargs):
    kwargs.setdefault("instruction", "sync")
    return lm.MemoryAccess("global", "shared", payload, **kwargs)


def test_resource_bits_match_z3_seven_bit_contract():
    resources = (
        lm.Resource.GLOBAL_LOAD,
        lm.Resource.GLOBAL_STORE,
        lm.Resource.SHARED_MEMORY,
        lm.Resource.TENSOR_CORE,
        lm.Resource.CUDA_CORE,
        lm.Resource.SFU,
        lm.Resource.SCALAR,
    )
    assert tuple(int(resource) for resource in resources) == (1, 2, 4, 8, 16, 32, 64)
    assert int(lm.Resource.ALL) == 127


def test_memory_completion_and_independent_ii_formula():
    cost = lm.estimate_task(lm.TaskFeatures(copies=[_copy(130)]), _params())
    # completion = 10 + ceil(130/64) + 2 = 15;
    # II = max(ceil(130/64), 3) + issue(1) = 4, without completion overhead.
    assert (cost.latency, cost.ii) == (15, 4)
    assert cost.terms["memory_latency"] == 15
    assert cost.terms["memory_ii"] == 4
    assert cost.terms["memory_accesses"][0]["transfer_cycles"] == 3
    assert cost.terms["resource_flags"] == int(lm.Resource.GLOBAL_LOAD)
    assert cost.reliability == lm.Reliability.MEASURED
    assert cost.fallback_used == ()


def test_memory_completion_sums_accesses_while_ii_takes_path_max():
    accesses = [
        _copy(130),
        lm.MemoryAccess("shared", "register", 65, instruction="sync"),
    ]
    cost = lm.estimate_task(lm.TaskFeatures(copies=accesses), _params())
    # Global completion 15; shared completion = 2 + ceil(65/32) + 1 = 6.
    # Their path IIs are 4 and 3 respectively.
    assert (cost.latency, cost.ii) == (21, 4)
    assert cost.terms["memory_path_iis"] == {"global->shared": 4, "shared->register": 3}
    assert cost.terms["resource_flags"] == int(lm.Resource.GLOBAL_LOAD | lm.Resource.SHARED_MEMORY)


def test_descriptor_count_formula_and_lanes_never_multiply_total_payload():
    params = _instruction_params("tma")

    def estimate(lanes):
        access = _copy(130, instruction="tma", descriptor_count=3, lanes=lanes)
        return lm.estimate_task(lm.TaskFeatures(copies=[access]), params, require_measured=True)

    cost = estimate(64)
    # completion = 3*(10+2) + ceil(130/64) = 39.
    # path II = 3*1 + max(ceil(130/64), 3*3) = 12.
    assert (cost.latency, cost.ii) == (39, 12)
    detail = cost.terms["memory_accesses"][0]
    assert detail["descriptor_completion_cycles"] == 36
    assert detail["transfer_cycles"] == 3
    assert detail["descriptor_count"] == 3
    assert detail["lanes"] == 64
    assert detail["bytes"] == detail["total_bytes"] == 130
    assert detail["calibration_path"] == "global->shared@tma"
    assert cost.fallback_used == ()
    # lanes documents/proves partitioning; bytes is already the total payload.
    one_lane = estimate(1)
    assert (one_lane.latency, one_lane.ii) == (cost.latency, cost.ii)
    assert one_lane.terms["memory_accesses"][0]["total_bytes"] == 130


def test_instruction_path_ii_sums_within_and_takes_max_across_paths():
    accesses = [
        _copy(64, instruction="tma", descriptor_count=2),
        _copy(64, instruction="tma"),
        _copy(64, instruction="sync"),
    ]
    cost = lm.estimate_task(
        lm.TaskFeatures(copies=accesses), _instruction_params("tma"), require_measured=True
    )
    # TMA completions are 25 and 13; generic synchronous completion is 13.
    # TMA II is (2+6)+(1+3)=12; synchronous II is a separate four-cycle path.
    assert (cost.latency, cost.ii) == (51, 12)
    assert cost.terms["memory_path_iis"] == {"global->shared@tma": 12, "global->shared": 4}


def test_source_destination_payload_origin_and_role_binding_are_audited():
    accesses = [
        _copy(130, payload_origin="source", role="scale_a", binding="gemm0/sfa"),
        _copy(64, payload_origin="destination", role="accumulator", binding="gemm0/acc"),
    ]
    features = lm.TaskFeatures(copies=accesses)
    cost = lm.estimate_task(features, _params(), require_measured=True)
    # Native extraction has already selected each total payload. Different
    # origins do not multiply, infer, or replace those byte counts in Python.
    assert (cost.latency, cost.ii) == (28, 8)
    details = cost.terms["memory_accesses"]
    assert [item["bytes"] for item in details] == [130, 64]
    assert [item["payload_origin"] for item in details] == ["source", "destination"]
    assert [item["role"] for item in details] == ["scale_a", "accumulator"]
    assert [item["binding"] for item in details] == ["gemm0/sfa", "gemm0/acc"]
    encoded = features.to_dict()["copies"]
    assert encoded[0]["payload_origin"] == "source"
    assert encoded[1]["role"] == "accumulator"
    assert encoded[1]["binding"] == "gemm0/acc"


@pytest.mark.parametrize("instruction", ["cp_async", "tma", "tmem_load", "tmem_store"])
def test_instruction_specific_calibration_keys_are_independent(instruction):
    params = _instruction_params(instruction)
    access = _copy(64, instruction=instruction, descriptor_count=2)
    cost = lm.estimate_task(lm.TaskFeatures(copies=[access]), params, require_measured=True)
    assert (cost.latency, cost.ii) == (25, 8)
    detail = cost.terms["memory_accesses"][0]
    assert detail["instruction"] == instruction
    assert detail["calibration_path"] == f"global->shared@{instruction}"
    assert detail["aggregation_path"] == f"global->shared@{instruction}"
    assert cost.reliability == lm.Reliability.MEASURED
    assert cost.fallback_used == ()


def test_instruction_specific_generic_fallback_is_explicit_and_unmeasured():
    access = _copy(64, instruction="cp_async", descriptor_count=2)
    cost = lm.estimate_task(lm.TaskFeatures(copies=[access]), _params())
    assert (cost.latency, cost.ii) == (25, 8)  # Nonfree generic estimate.
    assert cost.reliability == lm.Reliability.UNKNOWN
    assert "memory_instruction:cp_async" in cost.fallback_used
    detail = cost.terms["memory_accesses"][0]
    assert detail["calibration_path"] == "global->shared"
    assert detail["aggregation_path"] == "global->shared@cp_async"
    with pytest.raises(lm.CalibrationError, match="memory_instruction:cp_async"):
        lm.estimate_task(lm.TaskFeatures(copies=[access]), _params(), require_measured=True)


@pytest.mark.parametrize("instruction", ["auto", "unknown"])
def test_unresolved_or_unknown_instruction_is_nonfree_and_unmeasured(instruction):
    access = _copy(64, instruction=instruction)
    cost = lm.estimate_task(lm.TaskFeatures(copies=[access]), _params())
    assert (cost.latency, cost.ii) == (13, 4)
    assert cost.reliability == lm.Reliability.UNKNOWN
    assert f"memory_instruction:{instruction}" in cost.fallback_used
    with pytest.raises(lm.CalibrationError, match=f"memory_instruction:{instruction}"):
        lm.estimate_task(lm.TaskFeatures(copies=[access]), _params(), require_measured=True)


def test_legacy_feature_mapping_without_instruction_keeps_measured_generic_contract():
    legacy_copy = {
        "src_scope": "global",
        "dst_scope": "shared",
        "bytes": 64,
        "geometry": "coalesced",
        "width": 0,
    }
    features = lm.TaskFeatures.from_dict({"copies": [legacy_copy]})
    assert features.copies[0].instruction == "sync"
    cost = lm.estimate_task(features, _params(), require_measured=True)
    assert cost.reliability == lm.Reliability.MEASURED
    assert cost.fallback_used == ()
    assert cost.terms["memory_accesses"][0]["calibration_path"] == "global->shared"

    explicit_auto = lm.TaskFeatures.from_dict({
        "copies": [dict(legacy_copy, instruction="auto")],
    })
    assert explicit_auto.copies[0].instruction == "auto"
    unresolved = lm.estimate_task(explicit_auto, _params())
    assert unresolved.reliability == lm.Reliability.UNKNOWN
    assert "memory_instruction:auto" in unresolved.fallback_used


def test_missing_instruction_and_generic_paths_remain_nonfree():
    access = lm.MemoryAccess(
        "tmem", "register", 16, instruction="tmem_load", descriptor_count=2
    )
    cost = lm.estimate_task(lm.TaskFeatures(copies=[access]), _params())
    assert cost.latency > 0 and cost.ii > 0
    assert "memory_instruction:tmem_load" in cost.fallback_used
    assert "memory_path:tmem->register@tmem_load" in cost.fallback_used
    assert cost.reliability == lm.Reliability.UNKNOWN


def test_dynamic_descriptor_and_lane_extents_are_bounded_and_checked():
    access = _copy(64, descriptor_count=lm.Extent(2), lanes=lm.Extent(32))
    cost = lm.estimate_task(lm.TaskFeatures(copies=[access]), _params())
    assert (cost.latency, cost.ii) == (25, 8)
    assert cost.reliability == lm.Reliability.UPPER_BOUND


def test_composed_serial_branch_uses_instruction_costs_and_keeps_fallbacks():
    params = _instruction_params("tma")
    sync_leaf = {"kind": "leaf", "features": {"copies": [_copy(64).to_dict()]}}
    tma_leaf = {
        "kind": "leaf",
        "features": {
            "copies": [_copy(64, instruction="tma", descriptor_count=2).to_dict()]
        },
    }
    tree = {"kind": "serial", "children": [sync_leaf, {"kind": "branch", "then": tma_leaf, "else": sync_leaf}]}
    cost = lm.estimate_composed_task(tree, params, require_measured=True)
    # sync=(13,4), TMA=(25,8), branch=(25,8), serial=(max(13,4+25),4+8).
    assert (cost.latency, cost.ii) == (29, 12)
    composition = cost.terms["composition"]
    branch_features = composition["children"][1]["then"]["features"]["copies"][0]
    assert branch_features["instruction"] == "tma"
    assert branch_features["descriptor_count"] == 2

    fallback_leaf = {
        "kind": "leaf",
        "features": {"copies": [_copy(64, instruction="cp_async").to_dict()]},
    }
    fallback_tree = {"kind": "serial", "children": [sync_leaf, fallback_leaf]}
    fallback = lm.estimate_composed_task(fallback_tree, _params())
    assert "memory_instruction:cp_async" in fallback.fallback_used
    with pytest.raises(lm.CalibrationError, match="memory_instruction:cp_async"):
        lm.estimate_composed_task(fallback_tree, _params(), require_measured=True)


def test_zero_bandwidth_is_explicit_fallback_not_free():
    data = _calibration_data()
    data["memory_paths"]["global->shared"]["bandwidth_bytes_per_cycle"] = 0
    cost = lm.estimate_task(lm.TaskFeatures(copies=[_copy(64)]), _params(**data))
    # Unknown bandwidth uses the marked 1 B/cycle placeholder, not zero cycles.
    assert cost.terms["memory_accesses"][0]["transfer_cycles"] == 64
    assert cost.latency == 76
    assert "memory_paths.global->shared.bandwidth_bytes_per_cycle" in cost.fallback_used
    assert cost.reliability == lm.Reliability.UNKNOWN
    with pytest.raises(lm.CalibrationError, match="Measured task estimate"):
        lm.estimate_task(lm.TaskFeatures(copies=[_copy(64)]), _params(**data), require_measured=True)


def test_unknown_memory_path_has_nonfree_explicit_fallback():
    cost = lm.estimate_task(
        lm.TaskFeatures(copies=[lm.MemoryAccess("global", "register", 10)]),
        _params(),
    )
    assert cost.latency == 10
    assert cost.terms["memory_accesses"][0]["transfer_cycles"] == 10
    assert "memory_path:global->register" in cost.fallback_used
    assert "memory_paths.global->register.bandwidth_bytes_per_cycle" in cost.fallback_used
    assert cost.reliability == lm.Reliability.UNKNOWN


@pytest.mark.parametrize(
    ("width", "expected_bw", "expected_transfer"),
    [(8, 32, 4), (24, 48, 3)],
)
def test_strided_geometry_uses_transaction_width(width, expected_bw, expected_transfer):
    cost = lm.estimate_task(
        lm.TaskFeatures(copies=[_copy(128, geometry="strided", width=width)]),
        _params(),
    )
    access = cost.terms["memory_accesses"][0]
    # rate = 64 * width / (16 * ceil(width/16)); payload is 128 B.
    assert access["bandwidth_bytes_per_cycle"] == expected_bw
    assert access["transfer_cycles"] == expected_transfer
    coalesced = lm.estimate_task(lm.TaskFeatures(copies=[_copy(128)]), _params())
    assert coalesced.terms["memory_accesses"][0]["transfer_cycles"] == 2


def test_unknown_geometry_and_width_are_marked():
    unknown_geometry = lm.estimate_task(
        lm.TaskFeatures(copies=[_copy(64, geometry="mystery", width=0)]),
        _params(),
    )
    # The measured unknown-width/path fallback is 4 B/cycle.
    assert unknown_geometry.terms["memory_accesses"][0]["transfer_cycles"] == 16
    assert any(item.startswith("memory_geometry[0]") for item in unknown_geometry.fallback_used)

    unknown_width = lm.estimate_task(
        lm.TaskFeatures(copies=[_copy(64, geometry="strided", width=0)]),
        _params(),
    )
    assert unknown_width.terms["memory_accesses"][0]["transfer_cycles"] == 16
    assert unknown_width.terms["memory_accesses"][0]["bandwidth_bytes_per_cycle"] == 4
    assert "copies[0].width" in unknown_width.fallback_used
    assert unknown_width.reliability == lm.Reliability.UNKNOWN


def test_sm120_bf16_mma_sync_atom_rounding_and_gemm_ii():
    feature = lm.GemmFeature(M=17, N=9, K=17, dtype="bf16", acc="f32")
    cost = lm.estimate_task(lm.TaskFeatures(gemms=[feature]), _params())
    # sm120 bf16 atom=(16,8,16): rounded shape=(32,16,32),
    # ops=2*32*16*32=32768, throughput=ceil(32768/1024)=32.
    detail = cost.terms["gemms"][0]
    assert (detail["rounded_M"], detail["rounded_N"], detail["rounded_K"]) == (32, 16, 32)
    assert detail["operations"] == 32768
    assert detail["atom"] == [16, 8, 16]
    assert (cost.latency, cost.ii) == (37, 32)  # Base completion 5 is not II.
    assert cost.terms["resource_flags"] == int(lm.Resource.TENSOR_CORE)


def test_sm120_tf32_k_atom_is_distinct_from_generic_mma_sync():
    feature = lm.GemmFeature(M=16, N=8, K=9, dtype="tf32", acc="f32")
    sm120 = lm.estimate_task(lm.TaskFeatures(gemms=[feature]), _params(arch="sm120"))
    sm80 = lm.estimate_task(lm.TaskFeatures(gemms=[feature]), _params(arch="sm80"))
    assert sm120.terms["gemms"][0]["atom"] == [16, 8, 4]
    assert sm120.terms["gemms"][0]["rounded_K"] == 12
    assert sm80.terms["gemms"][0]["atom"] == [16, 8, 8]
    assert sm80.terms["gemms"][0]["rounded_K"] == 16
    assert (sm120.latency, sm80.latency) == (8, 9)


def test_multiple_gemms_sum_completion_but_take_max_ii():
    feature = lm.GemmFeature(M=17, N=9, K=17, dtype="bf16", acc="f32")
    cost = lm.estimate_task(lm.TaskFeatures(gemms=[feature, feature]), _params())
    assert (cost.latency, cost.ii) == (69, 32)  # One base 5, two 32-cycle terms.


def test_unsupported_gemm_dtype_is_explicit_fallback():
    feature = lm.GemmFeature(M=16, N=8, K=16, dtype="f32", acc="f32")
    cost = lm.estimate_task(lm.TaskFeatures(gemms=[feature]), _params())
    assert cost.latency == 5 + 2 * 16 * 8 * 16
    assert "gemm_dtype:f32" in cost.fallback_used
    assert "gemm_throughput_ops_per_cycle.f32:f32" in cost.fallback_used
    assert cost.reliability == lm.Reliability.UNKNOWN
    with pytest.raises(lm.CalibrationError):
        lm.estimate_task(lm.TaskFeatures(gemms=[feature]), _params(), require_measured=True)


def test_vector_operation_sfu_bandwidth_and_resource_max():
    vector = lm.VectorOps(add=10, mul=3, div=5, special=3, read_bytes=130)
    cost = lm.estimate_task(lm.TaskFeatures(vector=vector), _params())
    # CUDA-core ops: ceil(10/4)+ceil(3/2)+ceil(5/1)=3+2+5=10.
    # SFU: ceil(3/2)=2; shared reads: ceil(130/64)=3.
    # Compute and vector II are max(max(10,3),2)=10, not their sum.
    assert cost.terms["vector_op_cycles"] == {"add": 3, "mul": 2, "div": 5, "special": 2}
    assert cost.terms["vector_bandwidth_cycles"] == 3
    assert (cost.latency, cost.ii) == (10, 10)
    assert cost.terms["resource_flags"] == int(lm.Resource.CUDA_CORE | lm.Resource.SFU)


def test_zero_vector_bandwidth_is_not_free():
    vector = lm.VectorOps(read_bytes=65)
    cost = lm.estimate_task(lm.TaskFeatures(vector=vector), _params(vector_bandwidth_bytes_per_cycle=0))
    assert cost.latency == 65
    assert cost.ii == 65
    assert "vector_bandwidth_bytes_per_cycle" in cost.fallback_used
    assert cost.reliability == lm.Reliability.UNKNOWN


def test_scalar_and_mixed_compute_use_resource_max():
    scalar = lm.estimate_task(lm.TaskFeatures(scalar=3), _params(scalar_cycles=2))
    assert (scalar.latency, scalar.ii) == (6, 6)
    mixed = lm.estimate_task(
        lm.TaskFeatures(copies=[_copy(130)], vector=lm.VectorOps(add=1), scalar=3),
        _params(scalar_cycles=2),
    )
    # Memory completion 15 + max(vector=1, scalar=6); II remains max(4,1,6).
    assert (mixed.latency, mixed.ii) == (21, 6)


def test_scope_alias_normalizes_memory_path_and_provenance():
    data = _calibration_data()
    data["memory_paths"]["global->shared.dyn"] = data["memory_paths"].pop("global->shared")
    params = lm.Calibration.from_json(data).params
    assert "global->shared" in params.memory_paths
    task = lm.TaskFeatures(copies=[lm.MemoryAccess("global", "shared.dyn", 64, instruction="sync")])
    cost = lm.estimate_task(task, params, require_measured=True)
    assert (cost.latency, cost.ii) == (13, 4)


def test_dynamic_extent_upper_bound_and_unknown_reliability():
    bounded = lm.estimate_task(
        lm.TaskFeatures(copies=[_copy(lm.Extent(65))]),
        _params(),
    )
    assert bounded.latency == 14  # 10 + ceil(65/64) + 2
    assert bounded.reliability == lm.Reliability.UPPER_BOUND

    unreliable = lm.estimate_task(
        lm.TaskFeatures(copies=[_copy(64, reliability=False)]),
        _params(),
    )
    assert unreliable.reliability == lm.Reliability.UNKNOWN

    with pytest.raises(ValueError, match="finite dynamic extent upper_bound"):
        lm.estimate_task(lm.TaskFeatures(copies=[_copy(lm.Extent(None))]), _params())


def test_latency_and_ii_overrides_are_independent_and_validated():
    overridden = lm.estimate_task(
        lm.TaskFeatures(latency_override=9, ii_override=4),
        lm.CudaLatencyParams(),
        require_measured=True,
    )
    assert (overridden.latency, overridden.ii) == (9, 4)
    assert overridden.fallback_used == ()

    latency_only = lm.estimate_task(
        lm.TaskFeatures(copies=[_copy(130)], latency_override=20), _params()
    )
    assert (latency_only.latency, latency_only.ii) == (20, 4)
    assert latency_only.terms["estimated_latency"] == 15

    ii_only = lm.estimate_task(lm.TaskFeatures(copies=[_copy(130)], ii_override=2), _params())
    assert (ii_only.latency, ii_only.ii) == (15, 2)

    with pytest.raises(ValueError, match="latency>=ii>0"):
        lm.estimate_task(lm.TaskFeatures(latency_override=3, ii_override=4))
    with pytest.raises(ValueError):
        lm.estimate_task(lm.TaskFeatures(latency_override=0, ii_override=1))
    with pytest.raises(ValueError):
        lm.estimate_task(lm.TaskFeatures(ii_override=0))
    with pytest.raises(ValueError, match="Final task cost"):
        lm.estimate_task(lm.TaskFeatures(copies=[_copy(130)], latency_override=1), _params())


def test_serial_repeat_and_branch_formulas():
    a = lm.TaskCost(20, 4, {"name": "A"})
    b = lm.TaskCost(30, 3, {"name": "B"})
    combined = lm.serial(a, b)
    assert (combined.latency, combined.ii) == (34, 7)  # max(20, 4+30), 4+3

    repeated = lm.repeat(a, 3)
    assert (repeated.latency, repeated.ii) == (28, 12)  # 20+(3-1)*4, 3*4
    repeated_once = lm.repeat(a, 1)
    assert (repeated_once.latency, repeated_once.ii) == (a.latency, a.ii)
    empty = lm.repeat(a, 0)
    assert (empty.latency, empty.ii) == (0, 0)

    selected = lm.branch(a, b)
    assert (selected.latency, selected.ii) == (30, 4)
    assert lm.compose_serial([]).latency == 0


def test_per_core_phase_max_cross_wait_add_and_protocol_ii():
    a = lm.TaskCost(20, 4)
    b = lm.TaskCost(30, 3)
    c = lm.TaskCost(10, 2)
    d = lm.TaskCost(12, 1)
    one_phase = lm.aggregate_per_core([[a, b]], cross_wait=False)
    assert (one_phase.latency, one_phase.ii) == (30, 4)

    phased = lm.aggregate_per_core([[a, b], [c, d]], cross_wait=True)
    # max(20,30)+max(10,12)=42; an explicit wait makes protocol II=latency.
    assert (phased.latency, phased.ii) == (42, 42)
    assert phased.terms["cross_wait"] is True

    repeated = lm.repeat_per_core(one_phase, 3)
    assert (repeated.latency, repeated.ii) == (90, 12)
    repeated_boundary = lm.repeat_per_core(phased, 2, has_cross_wait=True)
    assert (repeated_boundary.latency, repeated_boundary.ii) == (84, 84)


def test_capacity_independent_completion_and_issue_intervals_differ():
    # Completion-only base/descriptor terms affect latency, not issue occupancy.
    memory = lm.estimate_task(lm.TaskFeatures(copies=[_copy(1)]), _params())
    gemm = lm.estimate_task(
        lm.TaskFeatures(gemms=[lm.GemmFeature(16, 8, 16, "bf16", "f32")]), _params()
    )
    assert memory.latency > memory.ii
    assert gemm.latency > gemm.ii
    assert memory.terms["memory_latency"] == 13
    assert memory.ii == 4
    assert gemm.terms["gemm_latency"] == 9
    assert gemm.ii == 4


def test_descriptor_completion_arithmetic_is_checked_for_overflow():
    access = _copy(1, instruction="tma", descriptor_count=lm.INT64_MAX)
    with pytest.raises(OverflowError, match="memory descriptor completion"):
        lm.estimate_task(lm.TaskFeatures(copies=[access]), _instruction_params("tma"))


@pytest.mark.parametrize(
    "changes",
    [
        {"instruction": "dense"},
        {"instruction": ""},
        {"instruction": 1},
        {"payload_origin": "source_bytes"},
        {"payload_origin": 1},
        {"role": "scale"},
        {"role": 1},
        {"binding": 1},
        {"descriptor_count": 0},
        {"descriptor_count": -1},
        {"descriptor_count": True},
        {"descriptor_count": lm.Extent(None)},
        {"lanes": 0},
        {"lanes": -1},
        {"lanes": True},
        {"lanes": lm.Extent(None)},
    ],
)
def test_invalid_copy_instruction_geometry_and_payload_metadata_are_rejected(changes):
    with pytest.raises((TypeError, ValueError)):
        access = _copy(64, **changes)
        lm.estimate_task(lm.TaskFeatures(copies=[access]), _params())


@pytest.mark.parametrize(
    "path",
    [
        "global->shared@",
        "global->shared@auto",
        "global->shared@sync",
        "global->shared@unknown",
        "global->shared@TMA",
        "global->shared@tma@cp_async",
    ],
)
def test_invalid_instruction_calibration_suffix_is_rejected(path):
    with pytest.raises(ValueError, match="instruction suffix"):
        lm.CudaLatencyParams(memory_paths={path: lm.MemoryPathParams()})


def test_duplicate_canonical_instruction_path_is_rejected():
    with pytest.raises(ValueError, match="Duplicate canonical memory path"):
        lm.CudaLatencyParams(memory_paths={
            "global->shared@tma": lm.MemoryPathParams(),
            "global->shared.dyn@tma": lm.MemoryPathParams(),
        })


def test_integer_overflow_and_invalid_values_are_rejected():
    with pytest.raises(OverflowError):
        lm.serial(lm.TaskCost(lm.INT64_MAX, lm.INT64_MAX), lm.TaskCost(1, 1))
    with pytest.raises(OverflowError):
        lm.repeat(lm.TaskCost(lm.INT64_MAX, lm.INT64_MAX), 2)
    with pytest.raises(ValueError):
        lm.estimate_task(lm.TaskFeatures(copies=[_copy(-1)]), _params())
    with pytest.raises(ValueError):
        lm.estimate_task(lm.TaskFeatures(gemms=[lm.GemmFeature(0, 8, 16)]), _params())
    with pytest.raises(ValueError):
        lm.aggregate_per_core([[lm.TaskCost(2, 1)], [lm.TaskCost(2, 1)]], cross_wait=False)
    with pytest.raises(ValueError, match="field_sources"):
        lm.CudaLatencyParams(field_sources={"arch": "not-a-source"})


def test_dataclasses_and_results_are_strict_json_serializable():
    features = lm.TaskFeatures(copies=[_copy(lm.Extent(65))])
    cost = lm.estimate_task(features, _params())
    encoded = cost.to_json()
    decoded = json.loads(encoded)
    restored = lm.TaskCost.from_dict(decoded)
    assert restored == cost
    assert json.dumps(asdict(cost), allow_nan=False)
    assert is_dataclass(cost) and is_dataclass(features)
    assert json.loads(features.to_json())["copies"][0]["bytes"]["upper_bound"] == 65
    assert json.loads(_params().to_json())["arch"] == "sm120"


def test_calibration_converts_ns_and_per_second_units_explicitly():
    data = _calibration_data()
    data["memory_paths"]["global->shared"] = {
        "base_ns": 8,
        "bandwidth_bytes_per_second": 64_000_000_000,
        "descriptor_ns": 2,
        "issue_ns": 1,
        "min_ii": 3,
        "transaction_bytes": 16,
        "strided_bandwidth_bytes_per_cycle": 0,
        "unknown_bandwidth_bytes_per_cycle": 4,
    }
    data.pop("gemm_base_cycles")
    data["gemm_base_ns"] = 5
    data["gemm_throughput_ops_per_cycle"]["bf16"] = {"ops_per_second": 1024_000_000_000}
    params = lm.Calibration.from_json(data).params
    assert params.gemm_base_cycles == 5
    assert params.gemm_throughput_ops_per_cycle["bf16"] == 1024
    cost = lm.estimate_task(lm.TaskFeatures(copies=[_copy(64)]), params)
    assert (cost.latency, cost.ii) == (11, 4)  # 8ns + 1 cycle + 2ns at 1GHz.


def test_gdn_calibration_schema_is_normalized_per_sm_per_cycle():
    gdn = {
        "measurements": {
            "device": {"sm_count": 2, "clock_peak_khz": 1_000_000},
            "cp_async_16b": {"gbps": 128},
            "smem_read_bw_gbps": {"gbps": 256},
            "gmem_read_l2resident": {"gbps": 64},
            "gmem_read_dram": {"gbps": 32},
            "mma_bf16_m16n8k16": {"flops": 4096e9},
            "sfu_ex2_f32": {"elem_per_s": 8e9},
        }
    }
    calibration = lm.Calibration.from_json(gdn)
    params = calibration.params
    assert params.memory_paths["global->shared"].bandwidth_bytes_per_cycle == 64
    assert params.gemm_throughput_ops_per_cycle["bf16:f32"] == 2048
    assert params.vector_throughput_ops_per_cycle["exp2"] == 4
    assert params.vector_bandwidth_bytes_per_cycle == 128
    assert calibration.fallback_used  # Completion bases and unmeasured paths remain explicit.


def test_require_measured_rejects_defaults_and_unmeasured_inputs():
    with pytest.raises(lm.CalibrationError):
        lm.Calibration.from_json("{}", require_measured=True)
    lm.Calibration.from_json(_calibration_data(), require_measured=True)

    measured_task = lm.TaskFeatures(copies=[_copy(64)])
    lm.estimate_task(measured_task, _params(), require_measured=True)

    unmeasured_data = _calibration_data()
    unmeasured_data.pop("status")
    unmeasured = lm.Calibration.from_json(unmeasured_data)
    assert unmeasured.reliability == lm.Reliability.ESTIMATED
    cost = lm.estimate_task(measured_task, unmeasured)
    assert cost.reliability == lm.Reliability.ESTIMATED
    with pytest.raises(lm.CalibrationError):
        lm.estimate_task(measured_task, unmeasured, require_measured=True)


def test_default_arch_and_ii_parameters_are_explicit_fallbacks():
    direct = lm.CudaLatencyParams(
        memory_paths={
            "global->shared": lm.MemoryPathParams(
                base_cycles=10,
                bandwidth_bytes_per_cycle=64,
                descriptor_cycles=2,
                issue_cycles=1,
            )
        },
        gemm_base_cycles=5,
        gemm_throughput_ops_per_cycle={"bf16:f32": 1024},
    )
    memory = lm.estimate_task(lm.TaskFeatures(copies=[_copy(64)]), direct)
    assert "memory_paths.global->shared.min_ii" in memory.fallback_used

    gemm = lm.estimate_task(
        lm.TaskFeatures(gemms=[lm.GemmFeature(16, 8, 16, "bf16", "f32")]), direct
    )
    assert {"arch", "gemm_min_ii", "gemm_pipeline_depth"} <= set(gemm.fallback_used)
    assert gemm.reliability == lm.Reliability.UNKNOWN
    with pytest.raises(lm.CalibrationError):
        lm.estimate_task(lm.TaskFeatures(gemms=[lm.GemmFeature(16, 8, 16)]), direct, require_measured=True)


def test_empty_task_gets_minimal_schedulable_cost():
    cost = lm.estimate_task(lm.TaskFeatures(), _params())
    assert (cost.latency, cost.ii) == (1, 1)
    assert cost.terms["units"]["latency"] == "cycles"
