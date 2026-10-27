"""CPU-only tests for strict recursive CUDA task-cost composition."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


_ROOT = Path(__file__).parents[3]
_LM_PATH = _ROOT / "tilelang" / "cuda" / "latency_model.py"
_LM_SPEC = importlib.util.spec_from_file_location("_tilelang_cuda_composed_latency_under_test", _LM_PATH)
lm = importlib.util.module_from_spec(_LM_SPEC)
sys.modules[_LM_SPEC.name] = lm
_LM_SPEC.loader.exec_module(lm)

_TC_PATH = _ROOT / "tilelang" / "cuda" / "task_cost.py"
_TC_SPEC = importlib.util.spec_from_file_location("_tilelang_cuda_composed_task_cost_under_test", _TC_PATH)
tc = importlib.util.module_from_spec(_TC_SPEC)
sys.modules[_TC_SPEC.name] = tc
_TC_SPEC.loader.exec_module(tc)


def _leaf(latency, ii, **feature_changes):
    features = {"latency_override": latency, "ii_override": ii}
    features.update(feature_changes)
    return {"kind": "leaf", "features": features}


def _serial(*children):
    return {"kind": "serial", "children": list(children)}


def _repeat(trips, body):
    return {"kind": "repeat", "trips": trips, "body": body}


def _branch(then, otherwise):
    return {"kind": "branch", "then": then, "else": otherwise}


def _copy_resource():
    return {
        "copies": [{"src_scope": "global", "dst_scope": "shared", "bytes": 64}],
    }


def _measured_calibration():
    return {
        "status": "measured",
        "arch": "sm120",
        "clock_hz": 1_000_000_000,
        "sm_count": 1,
        "memory_paths": {
            "global->shared": {
                "base_cycles": 10,
                "bandwidth_bytes_per_cycle": 64,
                "descriptor_cycles": 2,
                "issue_cycles": 1,
                "min_ii": 3,
                "transaction_bytes": 16,
                "strided_bandwidth_bytes_per_cycle": 0,
                "unknown_bandwidth_bytes_per_cycle": 4,
            },
        },
        "gemm_base_cycles": 5,
        "gemm_min_ii": 1,
        "gemm_pipeline_depth": 1,
        "gemm_throughput_ops_per_cycle": {},
        "vector_throughput_ops_per_cycle": {"add": 4, "unknown": 1},
        "vector_bandwidth_bytes_per_cycle": 64,
        "scalar_cycles": 1,
    }


def _hardware():
    return {
        "schema": tc.HARDWARE_SCHEMA,
        "device": {"compute_capability": "12.0"},
    }


def test_nested_serial_repeat_branch_matches_independent_oracle():
    a = _leaf(20, 4)
    b = _leaf(30, 3)
    c = _leaf(11, 2)
    # branch(B,C)=(30,3); repeat twice=(33,6).
    # serial(A,repeated)=(max(20,4+33),4+6)=(37,10).
    tree = _branch(_serial(a, _repeat(2, _branch(b, c))), c)
    cost = lm.estimate_composed_task(tree, lm.CudaLatencyParams(), require_measured=True)
    assert (cost.latency, cost.ii) == (37, 10)
    assert cost.reliability == lm.Reliability.MEASURED
    assert cost.fallback_used == ()
    assert cost.terms["composition"]["kind"] == "branch"
    repeated = cost.terms["composition"]["then"]["children"][1]
    assert repeated["kind"] == "repeat"
    assert repeated["body"]["kind"] == "branch"
    leaves = repeated["body"]
    assert leaves["then"]["features"]["latency_override"] == 30
    assert leaves["else"]["features"]["ii_override"] == 2
    # Canonical feature dictionaries retain all leaf data, not just cost pairs.
    assert "copies" in leaves["then"]["features"]
    assert "vector" in leaves["then"]["features"]


def test_zero_one_many_trips_and_empty_identity_keep_structural_resources():
    body = _leaf(20, 4, scalar=1)
    zero = lm.estimate_composed_task(_repeat(0, body))
    once = lm.estimate_composed_task(_repeat(1, body))
    many = lm.estimate_composed_task(_repeat(3, body))
    assert (zero.latency, zero.ii) == (0, 0)
    assert (once.latency, once.ii) == (20, 4)
    assert (many.latency, many.ii) == (28, 12)
    for cost in (zero, once, many):
        assert cost.terms["resource_flags"] == int(lm.Resource.SCALAR)

    empty = lm.estimate_composed_task(_serial())
    assert (empty.latency, empty.ii) == (0, 0)
    assert empty.terms["resource_flags"] == 0
    nested = lm.estimate_composed_task(_serial(_leaf(9, 2), _repeat(0, _leaf(20, 4, scalar=1))))
    assert (nested.latency, nested.ii) == (9, 2)
    assert nested.terms["resource_flags"] == int(lm.Resource.SCALAR)


def test_branch_max_is_independent_from_resource_union():
    global_copy = _leaf(20, 4, **_copy_resource())
    vector = _leaf(30, 3, vector={"add": 1})
    scalar = _leaf(11, 2, scalar=1)
    selected = lm.estimate_composed_task(_branch(global_copy, vector))
    assert (selected.latency, selected.ii) == (30, 4)
    assert selected.terms["resource_flags"] == int(lm.Resource.GLOBAL_LOAD | lm.Resource.CUDA_CORE)
    assert selected.terms["resources"] == ["GLOBAL_LOAD", "CUDA_CORE"]

    nested = lm.estimate_composed_task(_repeat(2, _serial(_branch(global_copy, vector), scalar)))
    # serial=(max(30,4+11),4+2)=(30,6); repeat=(30+6,12).
    assert (nested.latency, nested.ii) == (36, 12)
    assert nested.terms["resource_flags"] == int(
        lm.Resource.GLOBAL_LOAD | lm.Resource.CUDA_CORE | lm.Resource.SCALAR
    )


def test_dynamic_trip_extent_reliability_and_canonical_terms():
    body = _leaf(20, 4)
    bounded = lm.estimate_composed_task(_repeat(lm.Extent(3), body))
    assert (bounded.latency, bounded.ii) == (28, 12)
    assert bounded.reliability == lm.Reliability.UPPER_BOUND
    assert bounded.terms["composition"]["trips"] == {
        "upper_bound": 3,
        "reliability": "upper_bound",
    }

    estimated = lm.estimate_composed_task(
        _repeat({"upper_bound": 2, "reliability": "estimated"}, body)
    )
    assert (estimated.latency, estimated.ii) == (24, 8)
    assert estimated.reliability == lm.Reliability.ESTIMATED
    with pytest.raises(lm.CalibrationError, match="Measured composed task estimate"):
        lm.estimate_composed_task(
            _repeat({"upper_bound": 2, "reliability": "estimated"}, body),
            require_measured=True,
        )
    with pytest.raises(ValueError, match="finite dynamic extent upper_bound"):
        lm.estimate_composed_task(_repeat(lm.Extent(None), body))
    with pytest.raises(ValueError, match="finite dynamic extent upper_bound"):
        lm.estimate_composed_task(_repeat({"upper_bound": None}, body))


def test_fallback_and_reliability_union_is_sorted_and_deterministic():
    unknown_copy = {
        "kind": "leaf",
        "features": {
            "copies": [{"src_scope": "global", "dst_scope": "register", "bytes": 8}],
        },
    }
    unknown_vector = {"kind": "leaf", "features": {"vector": {"add": 3}}}
    tree = _repeat(2, _branch(unknown_copy, unknown_vector))
    first = lm.estimate_composed_task(tree, lm.CudaLatencyParams())
    second = lm.estimate_composed_task(tree, lm.CudaLatencyParams())
    assert first.reliability == lm.Reliability.UNKNOWN
    assert first.fallback_used == tuple(sorted(set(first.fallback_used)))
    assert "memory_path:global->register" in first.fallback_used
    assert "vector_throughput_ops_per_cycle.add" in first.fallback_used
    assert first.to_json() == second.to_json()
    assert first.terms == second.terms


def test_unknown_vector_work_is_nonfree_unknown_not_dense_or_zero():
    tree = {"kind": "leaf", "features": {"vector": {"unknown": 7}}}
    cost = lm.estimate_composed_task(tree, lm.CudaLatencyParams())
    assert (cost.latency, cost.ii) == (7, 7)
    assert cost.reliability == lm.Reliability.UNKNOWN
    assert "vector_unknown_ops" in cost.fallback_used
    assert "vector_throughput_ops_per_cycle.unknown" in cost.fallback_used
    assert cost.terms["resource_flags"] == int(lm.Resource.CUDA_CORE)
    assert not cost.terms["resource_flags"] & int(lm.Resource.TENSOR_CORE)
    with pytest.raises(lm.CalibrationError):
        lm.estimate_composed_task(tree, lm.CudaLatencyParams(), require_measured=True)


def test_composed_arithmetic_overflow_is_checked():
    largest = _leaf(lm.INT64_MAX, lm.INT64_MAX)
    with pytest.raises(OverflowError, match="serial latency|serial II"):
        lm.estimate_composed_task(_serial(largest, _leaf(1, 1)))
    with pytest.raises(OverflowError, match="repeat latency|repeat II"):
        lm.estimate_composed_task(_repeat(2, largest))
    with pytest.raises(OverflowError):
        lm.estimate_composed_task(_repeat({"upper_bound": lm.INT64_MAX}, _leaf(2, 2)))


@pytest.mark.parametrize(
    "tree",
    [
        {},
        {"kind": "unknown"},
        {"kind": 1},
        {"kind": "leaf"},
        {"kind": "leaf", "features": {}, "extra": 0},
        {"kind": "leaf", "features": []},
        {"kind": "serial", "children": {}},
        {"kind": "serial", "children": [], "extra": 0},
        {"kind": "repeat", "trips": 1},
        {"kind": "repeat", "trips": True, "body": _serial()},
        {"kind": "repeat", "trips": -1, "body": _serial()},
        {"kind": "repeat", "trips": 1.5, "body": _serial()},
        {"kind": "repeat", "trips": {"upper": 2}, "body": _serial()},
        {"kind": "repeat", "trips": {"upper_bound": 2, "max": 2}, "body": _serial()},
        {"kind": "branch", "then": _serial()},
        {"kind": "branch", "then": _serial(), "else": _serial(), "extra": 0},
    ],
)
def test_strict_tree_schema_and_invalid_trips_are_rejected(tree):
    with pytest.raises((TypeError, ValueError)):
        lm.estimate_composed_task(tree)


def test_non_object_tree_and_invalid_options_are_rejected():
    with pytest.raises(TypeError, match="tree must be a JSON object"):
        lm.estimate_composed_task([])
    with pytest.raises(TypeError, match="require_measured must be bool"):
        lm.estimate_composed_task(_serial(), require_measured=1)
    with pytest.raises(TypeError, match="params must be"):
        lm.estimate_composed_task(_serial(), params={})


def test_task_cost_internal_json_api_matches_composed_ffi_contract():
    tree = _branch(
        _leaf(20, 4, **_copy_resource()),
        _leaf(30, 3, vector={"add": 1}),
    )
    result = tc.estimate_composed_task(
        json.dumps(tree),
        _hardware(),
        _measured_calibration(),
        require_measured=True,
    )
    assert set(result) == {"latency", "ii", "terms", "resource_flags", "reliability", "fallback_used"}
    assert (result["latency"], result["ii"]) == (30, 4)
    assert result["resource_flags"] == int(lm.Resource.GLOBAL_LOAD | lm.Resource.CUDA_CORE)
    assert result["reliability"] == "measured"
    assert result["fallback_used"] == []
    assert result["terms"]["composition"]["kind"] == "branch"
    assert result["terms"]["composition"]["then"]["features"]["copies"][0]["bytes"] == 64
    assert json.loads(json.dumps(result, allow_nan=False, sort_keys=True)) == result


def test_task_cost_composed_ffi_global_round_trip():
    if tc._tvm_ffi is None:
        pytest.skip("tvm_ffi is unavailable")
    tree = _branch(
        _leaf(20, 4, **_copy_resource()),
        _leaf(30, 3, vector={"add": 1}),
    )
    ffi = tc._tvm_ffi.get_global_func("tl.cuda.estimate_composed_task")
    result = ffi(json.dumps(tree), _hardware(), _measured_calibration(), True)
    assert (result["latency"], result["ii"]) == (30, 4)
    assert result["resource_flags"] == int(lm.Resource.GLOBAL_LOAD | lm.Resource.CUDA_CORE)
    assert result["terms"]["composition"]["kind"] == "branch"
    assert json.loads(json.dumps(tc._plain(result), allow_nan=False, sort_keys=True)) == tc._plain(result)


def test_task_cost_json_api_rejects_dynamic_unmeasured_trips():
    with pytest.raises(tc._lm.CalibrationError, match="Measured composed task estimate"):
        tc.estimate_composed_task(
            _repeat({"upper_bound": 2}, _leaf(20, 4)),
            _hardware(),
            _measured_calibration(),
            require_measured=True,
        )
