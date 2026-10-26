"""Versioned calibration JSON, rate aggregation, and measured-only gates."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

try:
    from .environment import selected_clock_hz, utc_now
    from .instruction_audit import AUDITED_BENCHMARKS
    from .specs import CORE_BENCHMARKS, get_spec, units_for, validate_resolved_config, workload_for
    from .stats import summarize_samples
except ImportError:  # Direct ``python calibrate.py`` and path-based CPU tests.
    from environment import selected_clock_hz, utc_now
    from instruction_audit import AUDITED_BENCHMARKS
    from specs import CORE_BENCHMARKS, get_spec, units_for, validate_resolved_config, workload_for
    from stats import summarize_samples

SCHEMA_VERSION = "tilelang.cuda.calibration.v2"
LATENCY_MODEL_VIEW_VERSION = "tilelang.cuda.CudaLatencyParams.v1"
CUDA_COST_MODEL_SCHEMA = "tilelang.cuda.kernel-cost-calibration"
CUDA_COST_RATE_UNITS = {
    "clock_hz": "Hz",
    "mma_f16_flops_per_second": "FLOP/s",
    "mma_bf16_flops_per_second": "FLOP/s",
    "mma_tf32_flops_per_second": "FLOP/s",
    "cuda_fp32_flops_per_second": "FLOP/s",
    "dram_bytes_per_second": "B/s",
    "l2_bytes_per_second": "B/s",
    "shared_bytes_per_second": "B/s",
    "lsu_warp_instructions_per_second": "warp-instruction/s",
    "sfu_exp2_operations_per_second": "exp2/s",
    "barrier_ns": "ns",
    "launch_ns": "ns",
}

RATE_DEFINITIONS = {
    "dram_copy_bytes_per_second": ("dram_copy", "byte/second"),
    "l2_copy_bytes_per_second": ("l2_resident_copy", "byte/second"),
    "shared_read_bytes_per_second": ("shared_read", "byte/second"),
    "shared_write_bytes_per_second": ("shared_write", "byte/second"),
    "bf16_gemm_flops_per_second": ("bf16_mma_gemm", "FLOP/second"),
    "f16_gemm_flops_per_second": ("f16_mma_gemm", "FLOP/second"),
    "tf32_gemm_flops_per_second": ("tf32_mma_gemm", "FLOP/second"),
    "cuda_fp32_fma_flops_per_second": ("cuda_fp32_fma", "FLOP/second"),
    "sfu_exp2_operations_per_second": ("sfu_exp2", "operation/second"),
    "barrier_latency_ns": ("barrier", "nanosecond/(barrier*CTA)"),
    "event_overhead_ns": ("empty_launch", "nanosecond/event-pair"),
    "empty_launch_overhead_ns": ("empty_launch", "nanosecond/launch-including-events"),
    "launch_overhead_ns": ("empty_launch", "nanosecond/launch-marginal-to-event-pair"),
}
BENCHMARK_RATE_KEYS = {
    benchmark: tuple(key for key, (source, _) in RATE_DEFINITIONS.items() if source == benchmark)
    for benchmark in CORE_BENCHMARKS
}


class CalibrationSchemaError(ValueError):
    pass


def assert_finite_json(value: Any, path: str = "$") -> None:
    """Reject non-finite numbers and non-JSON values before any file is written."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CalibrationSchemaError(f"Non-finite JSON number at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CalibrationSchemaError(f"Non-string JSON key at {path}: {key!r}")
            assert_finite_json(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_json(item, f"{path}[{index}]")
        return
    raise CalibrationSchemaError(f"Unsupported JSON value at {path}: {type(value).__name__}")


def successful_result(result: Any) -> bool:
    basic = bool(
        isinstance(result, Mapping)
        and result.get("status") == "complete"
        and isinstance(result.get("correctness"), Mapping)
        and result["correctness"].get("passed") is True
        and isinstance(result.get("compile"), Mapping)
        and result["compile"].get("status") == "compiled"
        and isinstance(result.get("provenance"), Mapping)
        and result["provenance"].get("independent_measurement") is True
        and result["provenance"].get("ratio_derived") is False
        and result["provenance"].get("fit_from_formal_candidates") is False
    )
    if not basic or result.get("name") not in AUDITED_BENCHMARKS:
        return basic
    audit = result.get("instruction_audit")
    return bool(
        isinstance(audit, Mapping)
        and audit.get("passed") is True
        and audit.get("benchmark") == result.get("name")
        and audit.get("artifact_kind") in {"ptx", "sass"}
    )


def completion_status(results: Mapping[str, Any], required: tuple[str, ...] = CORE_BENCHMARKS) -> str:
    """No empty/subset/failed result set can accidentally become complete."""

    if not required:
        return "partial"
    return "complete" if all(successful_result(results.get(name)) for name in required) else "partial"


def _environment_with_device(environment: Mapping[str, Any], results: Mapping[str, Any]) -> dict[str, Any]:
    env = deepcopy(dict(environment))
    if not isinstance(env.get("device"), Mapping):
        for name in CORE_BENCHMARKS:
            result = results.get(name)
            if isinstance(result, Mapping):
                device = result.get("environment", {}).get("device")
                if isinstance(device, Mapping):
                    env["device"] = deepcopy(dict(device))
                    break
    return env


def _rate_value(rates: Mapping[str, Any], key: str) -> float | None:
    rate = rates.get(key)
    if not isinstance(rate, Mapping):
        return None
    value = rate.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


def _latency_model_view(status: str, environment: Mapping[str, Any], rates: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    device = environment.get("device", {})
    clock_hz = selected_clock_hz(dict(environment))
    arch = str(device.get("arch", "unknown"))
    sm_count = device.get("sm_count", 1)
    if isinstance(sm_count, bool) or not isinstance(sm_count, int) or sm_count <= 0:
        sm_count = 1
    view: dict[str, Any] = {
        "status": "measured" if status == "complete" else "partial",
        "measured": status == "complete",
        "source": "benchmark/z3_cost_model/calibration (microbenchmarks only)",
        "description": LATENCY_MODEL_VIEW_VERSION + "; validate the outer versioned document before consuming this view",
        "arch": arch,
        "clock_hz": clock_hz,
        "sm_count": sm_count,
        "device": {
            "name": device.get("name", environment.get("initial_gpu", {}).get("raw", {}).get("name")),
            "clock_basis": "nvidia-smi clocks.max.sm, falling back to observed clocks.sm",
        },
        "units": {
            "clock_hz": "Hz",
            "memory_paths.*.bandwidth_bytes_per_second": "byte/second",
            "gemm_throughput_ops_per_cycle.*.ops_per_second": "FLOP/second",
            "vector_throughput_ops_per_cycle.*.ops_per_second": "operation/second",
        },
    }
    if clock_hz <= 0:
        view["memory_paths"] = {}
        view["gemm_throughput_ops_per_cycle"] = {}
        view["vector_throughput_ops_per_cycle"] = {}
        return view, False

    # These are effective streaming rates, not claims about pure unidirectional
    # DRAM or asynchronous-copy descriptor costs.  See model_assumptions.
    path_sources = {
        "global->shared": "dram_copy_bytes_per_second",
        "global->register": "l2_copy_bytes_per_second",
        "shared->global": "dram_copy_bytes_per_second",
        "register->global": "dram_copy_bytes_per_second",
        "shared->register": "shared_read_bytes_per_second",
        "register->shared": "shared_write_bytes_per_second",
    }
    paths = {}
    for path, key in path_sources.items():
        value = _rate_value(rates, key)
        if value is not None and value > 0:
            paths[path] = {"bandwidth_bytes_per_second": value}
    view["memory_paths"] = paths

    gemm_rates = {}
    gemm_sources = {
        "bf16": "bf16_gemm_flops_per_second",
        "f16": "f16_gemm_flops_per_second",
        "tf32": "tf32_gemm_flops_per_second",
    }
    measured_gemm_rates = {}
    for dtype, rate_name in gemm_sources.items():
        value = _rate_value(rates, rate_name)
        measured_gemm_rates[dtype] = value
        if value is not None and value > 0:
            for key in (dtype, f"{dtype}:f32"):
                gemm_rates[key] = {"ops_per_second": value}
    view["gemm_throughput_ops_per_cycle"] = gemm_rates

    vector_rates = {}
    exp2 = _rate_value(rates, "sfu_exp2_operations_per_second")
    if exp2 is not None and exp2 > 0:
        # ``special`` is an explicit exp2 proxy; it is not a calibration of all SFU ops.
        vector_rates = {key: {"ops_per_second": exp2} for key in ("exp2", "special")}
    fp32_fma = _rate_value(rates, "cuda_fp32_fma_flops_per_second")
    if fp32_fma is not None and fp32_fma > 0:
        vector_rates["fma"] = {"ops_per_second": fp32_fma}
    view["vector_throughput_ops_per_cycle"] = vector_rates
    shared_read = _rate_value(rates, "shared_read_bytes_per_second")
    if shared_read is not None and shared_read > 0:
        view["vector_bandwidth_bytes_per_second"] = shared_read

    ready = bool(
        status == "complete"
        and arch != "unknown"
        and len(paths) == len(path_sources)
        and all(measured_gemm_rates[dtype] is not None for dtype in gemm_sources)
        and fp32_fma is not None
        and exp2 is not None
        and shared_read is not None
    )
    return view, ready


def _cuda_cost_model_view(
    status: str,
    environment: Mapping[str, Any],
    rates: Mapping[str, Any],
    core_ready: bool,
) -> dict[str, Any]:
    """Build the exact manifest understood by tilelang.cuda.cost_model.

    Missing rates stay null.  ``status=measured`` describes a complete core
    microbenchmark run; CudaCostModel still rejects any null it requires.
    """

    clock_hz = selected_clock_hz(dict(environment))
    shared_read = _rate_value(rates, "shared_read_bytes_per_second")
    shared_write = _rate_value(rates, "shared_write_bytes_per_second")
    shared = None
    lsu = None
    if shared_read is not None and shared_read > 0 and shared_write is not None and shared_write > 0:
        # Read L bytes and write L bytes: total traffic is 2L and total time
        # is L/read + L/write, hence harmonic-mean bandwidth 2RW/(R+W).
        shared = 2.0 * shared_read * shared_write / (shared_read + shared_write)
        # One 32-lane FP32 access moves 128 B per warp instruction.  For an
        # equal instruction mix, the same derivation gives shared/128.
        lsu = shared / 128.0

    values = {
        "clock_hz": clock_hz if clock_hz > 0 else None,
        "mma_f16_flops_per_second": _rate_value(rates, "f16_gemm_flops_per_second"),
        "mma_bf16_flops_per_second": _rate_value(rates, "bf16_gemm_flops_per_second"),
        "mma_tf32_flops_per_second": _rate_value(rates, "tf32_gemm_flops_per_second"),
        "cuda_fp32_flops_per_second": _rate_value(rates, "cuda_fp32_fma_flops_per_second"),
        "dram_bytes_per_second": _rate_value(rates, "dram_copy_bytes_per_second"),
        "l2_bytes_per_second": _rate_value(rates, "l2_copy_bytes_per_second"),
        "shared_bytes_per_second": shared,
        "lsu_warp_instructions_per_second": lsu,
        "sfu_exp2_operations_per_second": _rate_value(rates, "sfu_exp2_operations_per_second"),
        "barrier_ns": _rate_value(rates, "barrier_latency_ns"),
        "launch_ns": _rate_value(rates, "launch_overhead_ns"),
    }
    native_rates = {}
    field_sources = {}
    for name, unit in CUDA_COST_RATE_UNITS.items():
        value = values[name]
        if value is None or value <= 0:
            native_rates[name] = None
            field_sources[f"rates.{name}"] = "missing"
        else:
            # Derived shared/LSU values are still measured-microbenchmark rates,
            # never formal-candidate fits.  Their formulas remain in the outer JSON.
            native_rates[name] = {"value": value, "unit": unit, "source": "measured"}
            field_sources[f"rates.{name}"] = "measured"
    return {
        "schema": CUDA_COST_MODEL_SCHEMA,
        "version": 1,
        "status": "measured" if core_ready else "partial",
        "allow_partial": any(value is None for value in native_rates.values()),
        "rates": native_rates,
        "field_sources": field_sources,
    }


def aggregate_calibration(
    results: Mapping[str, Any],
    *,
    environment: Mapping[str, Any] | None = None,
    run: Mapping[str, Any] | None = None,
    allow_partial: bool = True,
) -> dict[str, Any]:
    if not isinstance(allow_partial, bool):
        raise TypeError("allow_partial must be bool")
    unknown = set(results) - set(CORE_BENCHMARKS)
    if unknown:
        raise CalibrationSchemaError(f"Unknown benchmark results: {sorted(unknown)}")
    benchmarks = {name: deepcopy(results.get(name)) for name in CORE_BENCHMARKS}
    status = completion_status(benchmarks)
    if status != "complete" and not allow_partial:
        missing = [name for name in CORE_BENCHMARKS if not successful_result(benchmarks[name])]
        raise CalibrationSchemaError("Partial calibration is not allowed; missing/invalid: " + ", ".join(missing))

    rates: dict[str, Any] = {}
    for key, (benchmark, unit) in RATE_DEFINITIONS.items():
        result = benchmarks[benchmark]
        value = result.get("rates", {}).get(key) if successful_result(result) else None
        if value is not None:
            if not isinstance(value, Mapping) or value.get("unit") != unit:
                raise CalibrationSchemaError(f"Invalid rate object for {key}")
            value = deepcopy(dict(value))
        rates[key] = value

    env = _environment_with_device(environment or {}, benchmarks)
    latency_model, model_ready = _latency_model_view(status, env, rates)
    cost_model = _cuda_cost_model_view(status, env, rates, model_ready)
    model_ready = bool(model_ready and all(value is not None for value in cost_model["rates"].values()))
    cost_model["status"] = "measured" if model_ready else "partial"
    missing = [name for name in CORE_BENCHMARKS if not successful_result(benchmarks[name])]
    document = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "allow_partial": allow_partial,
        "model_ready": model_ready,
        "created_at_utc": utc_now(),
        "required_benchmarks": list(CORE_BENCHMARKS),
        "selected_benchmarks": [name for name in CORE_BENCHMARKS if benchmarks[name] is not None],
        "missing_benchmarks": missing,
        "benchmarks": benchmarks,
        "rates": rates,
        "cost_model": cost_model,
        "calibration": deepcopy(cost_model),
        "latency_model": latency_model,
        "environment": env,
        "run": deepcopy(dict(run or {})),
        "event_launch_overhead": {
            "event_ns": _rate_value(rates, "event_overhead_ns"),
            "empty_launch_ns": _rate_value(rates, "empty_launch_overhead_ns"),
            "marginal_launch_ns": _rate_value(rates, "launch_overhead_ns"),
            "correction": "Other rates subtract empty-launch median when supplied; raw and net times are both retained.",
        },
        "model_assumptions": [
            "Rates come only from dedicated microbenchmarks; no GEMM4096/Conv/FA/GDN formal candidate timings are inputs or fit targets.",
            "DRAM/L2 copy bandwidth counts read+write payload bytes and is an effective streaming proxy, not pure-load or cp.async descriptor calibration.",
            "Shared-memory rates use this access pattern only; bank conflicts, vector width, synchronization, and address arithmetic in another kernel can change throughput.",
            "For CudaCostModel, shared B/s is 2*read*write/(read+write) for an equal-byte mix; LSU warp-instruction/s is shared/128 for an equal-instruction mix of this FP32 access pattern.",
            "BF16, FP16, and TF32 mma.sync rates are independently compiled, checked, timed, and compiled-artifact audited; no cross-dtype ratio is used.",
            "FP32 CUDA-core FLOP/s comes from independent scalar T.fma chains at two FLOPs per FMA, with TensorCore instructions forbidden by its artifact audit.",
            "The barrier value is per CTA and is not multiplied by concurrent blocks; residency and multi-wave execution limit that interpretation.",
            "Single-stream event timing does not calibrate cross-stream overlap. Resource times must not be assumed fully additive or fully overlapped.",
            "Empty-launch/event correction is an estimate; raw samples remain authoritative and uncorrected values are retained by each worker.",
        ],
        "provenance": {
            "method": "torch.cuda.Event, isolated one-process workers, median/IQR",
            "fit_from_formal_candidates": False,
            "used_nsys": False,
            "used_cupti": False,
            "used_injection": False,
            "reference_philosophy": "G3-style dedicated microbenchmarks and environment capture; no legacy numeric constants imported",
            "instruction_evidence": "GEMM and FP32 FMA rates require exported PTX or SASS audit in the per-task run cache; source/function names alone are insufficient",
        },
    }
    validate_calibration(document)
    return document


def _validate_success_item(name: str, item: Mapping[str, Any]) -> None:
    required = {"name", "status", "config", "shape", "dtype", "work", "units", "warmup", "repeat", "sample_count", "raw_samples_ms", "median_ms", "iqr_ms", "correctness", "compile", "provenance", "environment", "rates"}
    absent = required - set(item)
    if absent:
        raise CalibrationSchemaError(f"{name} lacks required fields: {sorted(absent)}")
    if item["name"] != name or item["shape"] != list(item["config"].get("shape", [])):
        raise CalibrationSchemaError(f"Identity/shape mismatch in {name}")
    if item["dtype"] != get_spec(name).dtype:
        raise CalibrationSchemaError(f"Unexpected dtype for {name}")
    validate_resolved_config(item["config"])
    expected_work = workload_for(item["config"])
    if item["work"] != expected_work:
        raise CalibrationSchemaError(f"Workload formula mismatch for {name}: {item['work']} != {expected_work}")
    if item["units"] != units_for(name):
        raise CalibrationSchemaError(f"Units mismatch for {name}")
    stats = summarize_samples(item["raw_samples_ms"])
    for key in ("median_ms", "iqr_ms", "q1_ms", "q3_ms"):
        if key in item and not math.isclose(float(item[key]), stats[key], rel_tol=1e-12, abs_tol=1e-12):
            raise CalibrationSchemaError(f"{key} is inconsistent with raw samples for {name}")
    if item["sample_count"] != item["repeat"] or item["sample_count"] != len(item["raw_samples_ms"]):
        raise CalibrationSchemaError(f"repeat/sample count mismatch for {name}")
    if not isinstance(item["warmup"], int) or isinstance(item["warmup"], bool) or item["warmup"] < 0:
        raise CalibrationSchemaError(f"Invalid warmup for {name}")
    if not isinstance(item["repeat"], int) or isinstance(item["repeat"], bool) or item["repeat"] <= 0:
        raise CalibrationSchemaError(f"Invalid repeat for {name}")
    if item["correctness"].get("passed") is not True:
        raise CalibrationSchemaError(f"A complete item must pass correctness: {name}")
    if not isinstance(item["environment"], Mapping):
        raise CalibrationSchemaError(f"Missing environment for {name}")
    if item["compile"].get("status") != "compiled":
        raise CalibrationSchemaError(f"Missing successful compile provenance for {name}")
    provenance = item["provenance"]
    if provenance.get("independent_measurement") is not True or provenance.get("ratio_derived") is not False or provenance.get("fit_from_formal_candidates") is not False:
        raise CalibrationSchemaError(f"Invalid independent-measurement provenance for {name}")
    if name in AUDITED_BENCHMARKS:
        compile_info = item["compile"]
        source_hash = compile_info.get("source_sha256")
        if not isinstance(source_hash, str) or len(source_hash) != 64 or any(char not in "0123456789abcdef" for char in source_hash):
            raise CalibrationSchemaError(f"Missing compiled-source SHA256 for {name}")
        if not isinstance(compile_info.get("source_path"), str) or not compile_info["source_path"]:
            raise CalibrationSchemaError(f"Missing compiled-source path for {name}")
        audit = item.get("instruction_audit")
        if not isinstance(audit, Mapping) or audit.get("passed") is not True or audit.get("benchmark") != name:
            raise CalibrationSchemaError(f"Missing compiled-instruction audit for {name}")
        kind = audit.get("artifact_kind")
        if kind not in {"ptx", "sass"} or audit.get("export_method") != f"export_{kind}":
            raise CalibrationSchemaError(f"Invalid instruction artifact kind/export provenance for {name}")
        artifact_dir = item["config"].get("artifact_dir")
        if not isinstance(artifact_dir, str) or not Path(artifact_dir).is_absolute() or item["config"].get("instruction_artifact_kind") != kind:
            raise CalibrationSchemaError(f"Instruction audit is not tied to the resolved task artifact directory for {name}")
        expected_source = (Path(artifact_dir) / f"{name}.cuda").resolve(strict=False)
        expected_artifact = (Path(artifact_dir) / f"{name}.{kind}").resolve(strict=False)
        if Path(compile_info["source_path"]).resolve(strict=False) != expected_source:
            raise CalibrationSchemaError(f"Compiled source path does not match the task artifact directory for {name}")
        raw_artifact_path = audit.get("artifact_path")
        if not isinstance(raw_artifact_path, str) or not raw_artifact_path:
            raise CalibrationSchemaError(f"Missing instruction artifact path for {name}")
        if Path(raw_artifact_path).resolve(strict=False) != expected_artifact:
            raise CalibrationSchemaError(f"Instruction artifact path does not match the task artifact directory for {name}")
        artifact_hash = audit.get("artifact_sha256")
        if not isinstance(artifact_hash, str) or len(artifact_hash) != 64 or any(char not in "0123456789abcdef" for char in artifact_hash):
            raise CalibrationSchemaError(f"Missing instruction artifact SHA256 for {name}")
        if not isinstance(audit.get("artifact_path"), str) or not audit["artifact_path"] or not audit.get("matched_required"):
            raise CalibrationSchemaError(f"Missing instruction artifact evidence for {name}")
        if audit.get("missing_required") or audit.get("forbidden_matches"):
            raise CalibrationSchemaError(f"Instruction audit contains missing/forbidden evidence for {name}")
    expected_rates = set(BENCHMARK_RATE_KEYS[name])
    if set(item["rates"]) != expected_rates:
        raise CalibrationSchemaError(f"Rate keys mismatch for {name}")
    for key in expected_rates:
        rate = item["rates"][key]
        if not isinstance(rate, Mapping) or rate.get("unit") != RATE_DEFINITIONS[key][1]:
            raise CalibrationSchemaError(f"Invalid unit for {key}")
        value = rate.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise CalibrationSchemaError(f"Invalid value for {key}")
        if name != "empty_launch" and value <= 0:
            raise CalibrationSchemaError(f"Measured throughput/latency must be positive for {key}")


def validate_calibration(document: Mapping[str, Any], *, require_complete: bool = False) -> None:
    if not isinstance(document, Mapping):
        raise CalibrationSchemaError("Calibration must be a JSON object")
    assert_finite_json(document)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise CalibrationSchemaError(f"Expected schema_version={SCHEMA_VERSION!r}")
    status = document.get("status")
    if status not in {"partial", "complete"}:
        raise CalibrationSchemaError("status must be partial or complete")
    if not isinstance(document.get("allow_partial"), bool):
        raise CalibrationSchemaError("allow_partial must be bool")
    if status == "partial" and not document["allow_partial"]:
        raise CalibrationSchemaError("A partial document must explicitly set allow_partial=true")
    if require_complete and status != "complete":
        raise CalibrationSchemaError("Measured-only CudaCostModel consumption requires status=complete")
    if document.get("required_benchmarks") != list(CORE_BENCHMARKS):
        raise CalibrationSchemaError("Core benchmark list/version mismatch")
    benchmarks = document.get("benchmarks")
    if not isinstance(benchmarks, Mapping) or set(benchmarks) != set(CORE_BENCHMARKS):
        raise CalibrationSchemaError("benchmarks must contain every core key, using null for missing items")
    for name, item in benchmarks.items():
        if item is None:
            if status == "complete":
                raise CalibrationSchemaError(f"Complete calibration has null item {name}")
            continue
        if not isinstance(item, Mapping):
            raise CalibrationSchemaError(f"Benchmark {name} must be an object or null")
        if successful_result(item):
            _validate_success_item(name, item)
        elif status == "complete":
            raise CalibrationSchemaError(f"Complete calibration contains unsuccessful {name}")
    expected_status = completion_status(benchmarks)
    if status != expected_status:
        raise CalibrationSchemaError(f"status={status} contradicts benchmark results ({expected_status})")
    missing = [name for name in CORE_BENCHMARKS if not successful_result(benchmarks[name])]
    if document.get("missing_benchmarks") != missing:
        raise CalibrationSchemaError("missing_benchmarks is inconsistent with benchmark results")
    rates = document.get("rates")
    if not isinstance(rates, Mapping) or set(rates) != set(RATE_DEFINITIONS):
        raise CalibrationSchemaError("rates must contain all global keys, with null for unavailable values")
    for key, (benchmark, unit) in RATE_DEFINITIONS.items():
        rate = rates[key]
        if not successful_result(benchmarks[benchmark]):
            if rate is not None:
                raise CalibrationSchemaError(f"Unavailable {benchmark} must leave {key}=null")
        else:
            if not isinstance(rate, Mapping) or rate.get("unit") != unit:
                raise CalibrationSchemaError(f"Invalid global rate {key}")
            if rate != benchmarks[benchmark]["rates"].get(key):
                raise CalibrationSchemaError(f"Global rate {key} differs from its benchmark measurement")
    if document.get("selected_benchmarks") != [name for name in CORE_BENCHMARKS if benchmarks[name] is not None]:
        raise CalibrationSchemaError("selected_benchmarks is inconsistent with benchmark results")
    if not isinstance(document.get("environment"), Mapping) or not isinstance(document.get("run"), Mapping):
        raise CalibrationSchemaError("environment and run must be objects")
    expected_latency, expected_ready = _latency_model_view(status, document["environment"], rates)
    expected_cost_model = _cuda_cost_model_view(status, document["environment"], rates, expected_ready)
    expected_ready = bool(expected_ready and all(value is not None for value in expected_cost_model["rates"].values()))
    expected_cost_model["status"] = "measured" if expected_ready else "partial"
    if document.get("cost_model") != expected_cost_model or document.get("calibration") != expected_cost_model:
        raise CalibrationSchemaError("cost_model/calibration views are not the canonical derivation from rates/environment")
    if document.get("latency_model") != expected_latency:
        raise CalibrationSchemaError("latency_model view is not the canonical derivation from rates/environment")
    if not isinstance(document.get("model_ready"), bool):
        raise CalibrationSchemaError("model_ready must be bool")
    if document.get("model_ready") != expected_ready:
        raise CalibrationSchemaError("model_ready is inconsistent with status, device, clock, and rates")
    if require_complete and document.get("model_ready") is not True:
        raise CalibrationSchemaError("CudaCostModel model view is incomplete (clock/device/rate metadata missing)")


def validate_for_cost_model(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the native view after the core gate; consumers still reject nulls."""

    validate_calibration(document, require_complete=True)
    view = document.get("cost_model")
    if not isinstance(view, Mapping) or view.get("status") != "measured" or view.get("schema") != CUDA_COST_MODEL_SCHEMA:
        raise CalibrationSchemaError("Complete calibration lacks a measured CudaCostModel view")
    return deepcopy(dict(view))


def json_dumps(document: Mapping[str, Any], *, indent: int = 2) -> str:
    assert_finite_json(document)
    return json.dumps(document, indent=indent, sort_keys=True, allow_nan=False) + "\n"


def write_json(path: str | os.PathLike[str], document: Mapping[str, Any]) -> None:
    """Atomically publish a validated artifact; never publish an invalid one."""

    validate_calibration(document)
    destination = Path(path)
    if not destination.parent.is_dir():
        raise CalibrationSchemaError(f"Output parent does not exist: {destination.parent}")
    payload = json_dumps(document)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
