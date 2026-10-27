"""CPU-only tests for the calibration schema and orchestration safeguards.

No torch, TileLang, CUDA driver, GPU, nvidia-smi, build, or benchmark execution is
used.  Synthetic complete results exist only in memory to test the state machine.
"""

from __future__ import annotations

import builtins
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmark.z3_cost_model.calibration import calibrate, environment, instruction_audit, paths, schema, specs, stats  # noqa: E402


def _device_environment():
    return {
        "initial_gpu": {"max_clock_mhz": 1500, "clock_mhz": 1200, "raw": {"name": "Synthetic CPU test device"}},
        "device": {"name": "Synthetic CPU test device", "arch": "sm_90", "sm_count": 4},
    }


def _rates_for_result(name, work, median_ms):
    rates = {}
    for key in schema.BENCHMARK_RATE_KEYS[name]:
        unit = schema.RATE_DEFINITIONS[key][1]
        if key == "barrier_latency_ns":
            value = median_ms * 1e6 / work["barriers_per_cta"]
            rate_work = work["barriers_per_cta"]
        elif key == "event_overhead_ns":
            value, rate_work = 100.0, 1
        elif key == "empty_launch_overhead_ns":
            value, rate_work = 500.0, 1
        elif key == "launch_overhead_ns":
            value, rate_work = 400.0, 1
        else:
            if key.endswith("flops_per_second"):
                amount = work["flops"]
            elif key.endswith("bytes_per_second"):
                amount = work["bytes"]
            else:
                amount = work["operations"]
            value = amount / (median_ms / 1000.0)
            rate_work = amount
        rates[key] = {"value": value, "unit": unit, "work": rate_work, "time_ms": median_ms, "source": name, "basis": "synthetic schema test"}
    return rates


def _result(name, samples=(1.0, 2.0, 3.0, 4.0)):
    config = specs.resolve_config(name, sm_count=4, l2_bytes=16 * specs.MIB)
    if name in instruction_audit.AUDITED_BENCHMARKS:
        config["artifact_dir"] = "/tmp/synthetic"
        config["instruction_artifact_kind"] = "ptx"
    work = specs.workload_for(config)
    summary = stats.summarize_samples(samples)
    result = {
        "name": name,
        "status": "complete",
        "config": config,
        "shape": config["shape"],
        "dtype": config["dtype"],
        "accum_dtype": config["accum_dtype"],
        "work": work,
        "units": specs.units_for(name),
        "warmup": 2,
        "repeat": len(samples),
        "correctness": {"passed": True, "check": "synthetic independent check"},
        "compile": {
            "status": "compiled",
            "target": "cuda",
            "execution_backend": "tvm_ffi",
            "source_path": f"/tmp/synthetic/{name}.cuda",
            "source_sha256": "a" * 64,
        },
        "provenance": {
            "measurement": "dedicated_microbenchmark",
            "independent_measurement": True,
            "ratio_derived": False,
            "fit_from_formal_candidates": False,
            "timing": "torch.cuda.Event",
        },
        "environment": {"device": _device_environment()["device"], "worker": name},
        "rates": _rates_for_result(name, work, summary["median_ms"]),
    }
    result.update(summary)
    if name in instruction_audit.AUDITED_BENCHMARKS:
        result["instruction_audit"] = {
            "benchmark": name,
            "artifact_kind": "ptx",
            "export_method": "export_ptx",
            "artifact_path": f"/tmp/synthetic/{name}.ptx",
            "artifact_sha256": "b" * 64,
            "artifact_bytes": 128,
            "passed": True,
            "matched_required": [[{"pattern": "synthetic", "count": 1, "minimum": 1}]],
            "missing_required": [],
            "forbidden_matches": [],
            "match_counts": {"synthetic": 1},
        }
    if name == "empty_launch":
        result["event_pair"] = stats.summarize_samples([0.001, 0.001, 0.001, 0.001])
    return result


def _all_results():
    return {name: _result(name) for name in specs.CORE_BENCHMARKS}


class FormulaTests(unittest.TestCase):
    def test_copy_bytes_include_read_and_write_and_iterations(self):
        self.assertEqual(specs.copy_bytes(1024, "float32", 3), 2 * 1024 * 4 * 3)
        self.assertEqual(specs.copy_bytes(1024, "bfloat16", 1), 4096)

    def test_gemm_flops_are_logical_not_per_thread_warp_count(self):
        self.assertEqual(specs.gemm_flops(16, 8, 16), 4096)
        self.assertEqual(specs.gemm_flops(96, 80, 48, blocks=7, inner_iterations=11), 7 * 11 * 2 * 96 * 80 * 48)
        # The old per-thread failure mode would make this 32 times larger.
        self.assertNotEqual(specs.gemm_flops(16, 8, 16), 32 * 4096)

    def test_shared_sfu_and_barrier_formulas(self):
        expected = 3 * 256 * 17 * 16 * 4
        self.assertEqual(specs.shared_access_bytes(3, 256, 17, 16), expected)
        self.assertEqual(specs.sfu_exp2_operations(3, 256, 17, 4), 3 * 256 * 17 * 4)
        self.assertEqual(specs.fp32_fma_flops(3, 256, 17, 4), 2 * 3 * 256 * 17 * 4)
        self.assertEqual(specs.barrier_counts(3, 19), {"barriers_per_cta": 19, "total_barriers": 57})

    def test_every_resolved_workload_matches_independent_formulas(self):
        for name in specs.CORE_BENCHMARKS:
            with self.subTest(name=name):
                config = specs.resolve_config(name, sm_count=4, l2_bytes=16 * specs.MIB)
                specs.validate_resolved_config(config)
                work = specs.workload_for(config)
                self.assertEqual(work["launches"], 1)
                self.assertLess(config["estimated_device_memory_bytes"], 2 * specs.GIB)
                if name in {"dram_copy", "l2_resident_copy"}:
                    self.assertEqual(work["bytes"], 2 * config["shape"][0] * 4 * config["inner_iterations"])
                if name in {"bf16_mma_gemm", "f16_mma_gemm", "tf32_mma_gemm"}:
                    self.assertEqual(work["flops"], config["blocks"] * config["inner_iterations"] * 2 * 96 * 80 * 48)
                if name == "cuda_fp32_fma":
                    self.assertEqual(work["flops"], 2 * work["operations"])
                    self.assertEqual(work["operations"], config["blocks"] * config["threads"] * config["inner_iterations"] * config["unroll"])

    def test_calibration_gemm_shape_is_not_a_formal_shape(self):
        for name in ("bf16_mma_gemm", "f16_mma_gemm", "tf32_mma_gemm"):
            shape = specs.get_spec(name).shape
            self.assertNotIn(shape, {(4096, 4096, 4096), (64, 64, 64), (128, 128, 128)})
            self.assertEqual(shape, (96, 80, 48))

    def test_tf32_requires_fp32_storage_and_explicit_path(self):
        config = specs.resolve_config("tf32_mma_gemm", sm_count=4, l2_bytes=16 * specs.MIB)
        self.assertEqual(config["storage_dtype"], "float32")
        self.assertEqual(config["dtype"], "tfloat32")
        self.assertTrue(config["allow_tf32"])
        config["allow_tf32"] = False
        with self.assertRaisesRegex(ValueError, "explicit allow_tf32"):
            specs.validate_resolved_config(config)

    def test_l2_shape_respects_small_l2(self):
        config = specs.resolve_config("l2_resident_copy", sm_count=2, l2_bytes=4 * specs.MIB)
        self.assertLessEqual(2 * config["shape"][0] * 4, (4 * specs.MIB) // 4)
        self.assertEqual(config["shape"][0] % (config["threads"] * config["elements_per_thread"]), 0)


class StatisticsTests(unittest.TestCase):
    def test_median_iqr_and_raw_order(self):
        result = stats.summarize_samples([4.0, 1.0, 3.0, 2.0])
        self.assertEqual(result["raw_samples_ms"], [4.0, 1.0, 3.0, 2.0])
        self.assertEqual(result["median_ms"], 2.5)
        self.assertEqual(result["q1_ms"], 1.75)
        self.assertEqual(result["q3_ms"], 3.25)
        self.assertEqual(result["iqr_ms"], 1.5)

    def test_odd_median_and_rate_units(self):
        result = stats.summarize_samples([9, 1, 5])
        self.assertEqual(result["median_ms"], 5)
        rate = stats.throughput(1000, 2.0, "byte/second")
        self.assertEqual(rate["value"], 500_000)
        latency = stats.latency_per_operation(2.0, 4)
        self.assertEqual(latency["value"], 500_000)

    def test_invalid_samples_are_rejected(self):
        for samples in ([], [-1], [float("nan")], [float("inf")], [True]):
            with self.subTest(samples=samples), self.assertRaises((ValueError, TypeError)):
                stats.summarize_samples(samples)


class SchemaTests(unittest.TestCase):
    def test_empty_or_subset_run_cannot_be_complete(self):
        document = schema.aggregate_calibration({}, environment=_device_environment())
        self.assertEqual(document["status"], "partial")
        self.assertFalse(document["model_ready"])
        self.assertEqual(document["missing_benchmarks"], list(specs.CORE_BENCHMARKS))
        self.assertTrue(all(value is None for value in document["benchmarks"].values()))
        self.assertTrue(all(value is None for value in document["rates"].values()))
        self.assertEqual(schema.completion_status({}), "partial")
        with self.assertRaises(schema.CalibrationSchemaError):
            schema.validate_for_cost_model(document)

    def test_complete_schema_and_native_model_gate(self):
        document = schema.aggregate_calibration(_all_results(), environment=_device_environment(), allow_partial=False)
        self.assertEqual(document["status"], "complete")
        self.assertTrue(document["model_ready"])
        self.assertEqual(document["missing_benchmarks"], [])
        view = schema.validate_for_cost_model(document)
        self.assertEqual(view["status"], "measured")
        self.assertEqual(view["schema"], schema.CUDA_COST_MODEL_SCHEMA)
        self.assertEqual(document["calibration"], view)
        self.assertIsNotNone(view["rates"]["cuda_fp32_flops_per_second"])
        self.assertIsNotNone(view["rates"]["mma_f16_flops_per_second"])
        self.assertIsNotNone(view["rates"]["mma_tf32_flops_per_second"])
        self.assertFalse(view["allow_partial"])
        self.assertTrue(all(value is not None for value in view["rates"].values()))
        self.assertGreater(view["rates"]["lsu_warp_instructions_per_second"]["value"], 0)
        for name in ("f16_mma_gemm", "tf32_mma_gemm", "cuda_fp32_fma"):
            item = document["benchmarks"][name]
            self.assertEqual(item["sample_count"], len(item["raw_samples_ms"]))
            self.assertIsInstance(item["median_ms"], float)
            self.assertIsInstance(item["iqr_ms"], float)
            self.assertEqual(item["compile"]["status"], "compiled")
            self.assertTrue(item["correctness"]["passed"])
            self.assertTrue(item["provenance"]["independent_measurement"])
            self.assertTrue(item["instruction_audit"]["passed"])
        decoded = json.loads(schema.json_dumps(document))
        self.assertEqual(decoded["schema_version"], schema.SCHEMA_VERSION)
        self.assertFalse(document["provenance"]["fit_from_formal_candidates"])

    def test_missing_item_stays_null_and_partial_is_explicit(self):
        results = _all_results()
        del results["dram_copy"]
        document = schema.aggregate_calibration(results, environment=_device_environment(), allow_partial=True)
        self.assertEqual(document["status"], "partial")
        self.assertTrue(document["allow_partial"])
        self.assertIsNone(document["benchmarks"]["dram_copy"])
        self.assertIsNone(document["rates"]["dram_copy_bytes_per_second"])
        self.assertFalse(document["model_ready"])
        with self.assertRaises(schema.CalibrationSchemaError):
            schema.validate_for_cost_model(document)
        with self.assertRaises(schema.CalibrationSchemaError):
            schema.aggregate_calibration(results, environment=_device_environment(), allow_partial=False)

    def test_failed_correctness_invalidates_only_its_rates(self):
        results = _all_results()
        results["bf16_mma_gemm"]["correctness"]["passed"] = False
        document = schema.aggregate_calibration(results, environment=_device_environment())
        self.assertEqual(document["status"], "partial")
        self.assertIsNone(document["rates"]["bf16_gemm_flops_per_second"])
        self.assertIsNotNone(document["rates"]["sfu_exp2_operations_per_second"])

    def test_formula_and_statistic_tampering_are_rejected(self):
        results = _all_results()
        results["dram_copy"]["work"]["bytes"] += 4
        with self.assertRaisesRegex(schema.CalibrationSchemaError, "Workload formula mismatch"):
            schema.aggregate_calibration(results, environment=_device_environment())
        results = _all_results()
        results["shared_read"]["median_ms"] += 1
        with self.assertRaisesRegex(schema.CalibrationSchemaError, "inconsistent with raw samples"):
            schema.aggregate_calibration(results, environment=_device_environment())

    def test_complete_status_tampering_is_rejected(self):
        document = schema.aggregate_calibration({}, environment=_device_environment())
        document["status"] = "complete"
        with self.assertRaises(schema.CalibrationSchemaError):
            schema.validate_calibration(document)

    def test_rate_and_cost_model_tampering_are_rejected(self):
        document = schema.aggregate_calibration(_all_results(), environment=_device_environment())
        document["rates"]["dram_copy_bytes_per_second"]["value"] *= 2
        with self.assertRaisesRegex(schema.CalibrationSchemaError, "differs from its benchmark"):
            schema.validate_calibration(document)
        document = schema.aggregate_calibration(_all_results(), environment=_device_environment())
        document["cost_model"]["clock_hz"] = 1
        with self.assertRaisesRegex(schema.CalibrationSchemaError, "canonical derivation"):
            schema.validate_calibration(document)

    def test_nonfinite_nested_values_never_serialize(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            document = schema.aggregate_calibration(_all_results(), environment=_device_environment())
            document["benchmarks"]["dram_copy"]["environment"]["bad"] = bad
            with self.subTest(bad=bad), self.assertRaises(schema.CalibrationSchemaError):
                schema.validate_calibration(document)
            with self.assertRaises(schema.CalibrationSchemaError):
                schema.json_dumps(document)

    def test_native_shared_and_lsu_rates_use_equal_mix_harmonic_formulas(self):
        results = _all_results()
        results["shared_read"]["rates"]["shared_read_bytes_per_second"]["value"] = 200.0
        results["shared_write"]["rates"]["shared_write_bytes_per_second"]["value"] = 100.0
        document = schema.aggregate_calibration(results, environment=_device_environment())
        rates = document["cost_model"]["rates"]
        expected_shared = 2 * 200.0 * 100.0 / 300.0
        self.assertTrue(math.isclose(rates["shared_bytes_per_second"]["value"], expected_shared))
        self.assertTrue(math.isclose(rates["lsu_warp_instructions_per_second"]["value"], expected_shared / 128.0))

    def test_new_dtype_and_cuda_rates_are_independent_not_ratio_derived(self):
        results = _all_results()
        values = {
            "f16_mma_gemm": ("f16_gemm_flops_per_second", "mma_f16_flops_per_second", 123.0),
            "tf32_mma_gemm": ("tf32_gemm_flops_per_second", "mma_tf32_flops_per_second", 456.0),
            "cuda_fp32_fma": ("cuda_fp32_fma_flops_per_second", "cuda_fp32_flops_per_second", 789.0),
        }
        for benchmark, (rate_key, _, value) in values.items():
            results[benchmark]["rates"][rate_key]["value"] = value
        document = schema.aggregate_calibration(results, environment=_device_environment())
        for _, (rate_key, native_key, value) in values.items():
            self.assertEqual(document["rates"][rate_key]["value"], value)
            self.assertEqual(document["cost_model"]["rates"][native_key]["value"], value)

    def test_instruction_audit_is_required_for_new_measured_rates(self):
        for name in instruction_audit.AUDITED_BENCHMARKS:
            results = _all_results()
            results[name]["instruction_audit"]["passed"] = False
            document = schema.aggregate_calibration(results, environment=_device_environment())
            self.assertEqual(document["status"], "partial")
            self.assertIsNone(document["rates"][schema.BENCHMARK_RATE_KEYS[name][0]])
            with self.assertRaises(schema.CalibrationSchemaError):
                schema.validate_for_cost_model(document)

    def test_instruction_artifact_identity_must_match_benchmark_and_task(self):
        results = _all_results()
        results["f16_mma_gemm"]["instruction_audit"]["artifact_path"] = "/tmp/synthetic/bf16_mma_gemm.ptx"
        with self.assertRaisesRegex(schema.CalibrationSchemaError, "artifact path does not match"):
            schema.aggregate_calibration(results, environment=_device_environment())

    def test_current_pure_python_latency_model_accepts_complete_view(self):
        path = REPO / "tilelang" / "cuda" / "latency_model.py"
        spec = importlib.util.spec_from_file_location("_calibration_latency_model_cpu_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        document = schema.aggregate_calibration(_all_results(), environment=_device_environment())
        view = document["latency_model"]
        calibration = module.Calibration.from_json(view)
        self.assertGreater(calibration.params.gemm_throughput_ops_per_cycle["bf16:f32"], 0)
        self.assertGreater(calibration.params.gemm_throughput_ops_per_cycle["f16:f32"], 0)
        self.assertGreater(calibration.params.gemm_throughput_ops_per_cycle["tf32:f32"], 0)
        self.assertGreater(calibration.params.vector_throughput_ops_per_cycle["fma"], 0)
        self.assertGreater(calibration.params.memory_paths["shared->register"].bandwidth_bytes_per_cycle, 0)
        self.assertGreater(calibration.params.vector_throughput_ops_per_cycle["exp2"], 0)
    def test_latency_model_consumer_schema_accepts_instruction_specific_paths(self):
        path = REPO / "tilelang" / "cuda" / "latency_model.py"
        spec = importlib.util.spec_from_file_location("_calibration_instruction_latency_cpu_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        document = schema.aggregate_calibration(_all_results(), environment=_device_environment())
        # Add synthetic consumer-view entries only in memory. The benchmark
        # producer and checked-in calibration records remain unchanged.
        view = json.loads(json.dumps(document["latency_model"]))
        generic = view["memory_paths"]["global->shared"]
        for instruction in ("cp_async", "tma", "tmem_load", "tmem_store"):
            view["memory_paths"][f"global->shared@{instruction}"] = dict(generic)
        calibration = module.Calibration.from_json(view)
        assert set(calibration.params.memory_paths) >= {
            "global->shared@cp_async",
            "global->shared@tma",
            "global->shared@tmem_load",
            "global->shared@tmem_store",
        }
        key = "memory_paths.global->shared@tma.bandwidth_bytes_per_cycle"
        assert calibration.params.field_sources[key] == "measured"
        access = module.MemoryAccess("global", "shared", 64, instruction="tma")
        cost = module.estimate_task(module.TaskFeatures(copies=[access]), calibration)
        assert cost.terms["memory_accesses"][0]["calibration_path"] == "global->shared@tma"
        assert "memory_instruction:tma" not in cost.fallback_used

    def test_current_cuda_cost_model_accepts_complete_and_rejects_each_new_missing_rate(self):
        path = REPO / "tilelang" / "cuda" / "cost_model.py"
        spec = importlib.util.spec_from_file_location("_calibration_cuda_cost_model_cpu_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        document = schema.aggregate_calibration(_all_results(), environment=_device_environment())
        hardware = {
            "schema": module.HARDWARE_SCHEMA,
            "device": {
                "name": "Synthetic CPU test device",
                "compute_capability": "12.0",
                "multi_processor_count": 4,
                "warp_size": 32,
                "max_threads_per_block": 1024,
                "max_threads_per_multiprocessor": 1024,
                "max_registers_per_multiprocessor": 65536,
                "shared_memory_per_multiprocessor_bytes": 102400,
                "shared_memory_per_block_optin_bytes": 101376,
                "reserved_shared_memory_per_block_bytes": 0,
            },
        }
        shape = {"M": 96, "N": 80, "K": 48, "dtype": "bf16"}
        # All twelve native fields now come from independent microbenchmarks or
        # documented shared/LSU derivations. No synthetic patch-up is needed.
        measured_model = module.CudaCostModel(hardware, document, "gemm", shape)
        self.assertEqual(measured_model.fallback_used, ())

        for benchmark, native_key in (
            ("f16_mma_gemm", "mma_f16_flops_per_second"),
            ("tf32_mma_gemm", "mma_tf32_flops_per_second"),
            ("cuda_fp32_fma", "cuda_fp32_flops_per_second"),
        ):
            with self.subTest(missing=benchmark):
                results = _all_results()
                del results[benchmark]
                partial = schema.aggregate_calibration(results, environment=_device_environment())
                self.assertEqual(partial["status"], "partial")
                self.assertFalse(partial["model_ready"])
                self.assertIsNone(partial["cost_model"]["rates"][native_key])
                with self.assertRaises(module.CalibrationError):
                    module.CudaCostModel(hardware, partial, "gemm", shape)


class InstructionAuditTests(unittest.TestCase):
    def test_ptx_mma_dtypes_are_checked_independently(self):
        examples = {
            "bf16_mma_gemm": "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};",
            "f16_mma_gemm": "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};",
            "tf32_mma_gemm": "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};",
        }
        for name, text in examples.items():
            with self.subTest(name=name):
                report = instruction_audit.require_instruction_audit(name, "// T.gemm source comment\n" + text, "ptx")
                self.assertTrue(report["passed"])
                self.assertEqual(len(report["artifact_sha256"]), 64)
        with self.assertRaises(instruction_audit.InstructionAuditError):
            instruction_audit.require_instruction_audit("bf16_mma_gemm", examples["f16_mma_gemm"], "ptx")
        with self.assertRaises(instruction_audit.InstructionAuditError):
            instruction_audit.require_instruction_audit("tf32_mma_gemm", examples["f16_mma_gemm"], "ptx")

    def test_tf32_k4_and_k8_ptx_are_architecture_alternatives(self):
        for k in (4, 8):
            text = f"mma.sync.aligned.m16n8k{k}.row.col.f32.tf32.tf32.f32;"
            self.assertTrue(instruction_audit.audit_instruction_text("tf32_mma_gemm", text, "ptx")["passed"])

    def test_fp32_fma_requires_four_chains_and_forbids_tensor_core(self):
        fma = "\n".join(["fma.rn.f32 %f1, %f2, %f3, %f4;"] * 4)
        self.assertTrue(instruction_audit.audit_instruction_text("cuda_fp32_fma", fma, "ptx")["passed"])
        packed = "\n".join(["fma.rn.f32x2 %v1, %v2, %v3, %v4;"] * 2)
        self.assertTrue(instruction_audit.audit_instruction_text("cuda_fp32_fma", packed, "ptx")["passed"])
        self.assertFalse(instruction_audit.audit_instruction_text("cuda_fp32_fma", "\n".join(fma.splitlines()[:3]), "ptx")["passed"])
        with_tensor = fma + "\nmma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32;"
        report = instruction_audit.audit_instruction_text("cuda_fp32_fma", with_tensor, "ptx")
        self.assertFalse(report["passed"])
        self.assertTrue(report["forbidden_matches"])

    def test_sass_reaudit_entry_rules(self):
        self.assertTrue(instruction_audit.audit_instruction_text("f16_mma_gemm", "HMMA.16816.F32 R44, R4, R2, R44 ;", "sass")["passed"])
        self.assertFalse(instruction_audit.audit_instruction_text("f16_mma_gemm", "HMMA.16816.F32.BF16 R44, R4, R2, R44 ;", "sass")["passed"])
        self.assertFalse(instruction_audit.audit_instruction_text("f16_mma_gemm", "HMMA.1688.F32.TF32 R72, R124, R4, R72 ;", "sass")["passed"])
        self.assertTrue(instruction_audit.audit_instruction_text("tf32_mma_gemm", "HMMA.1688.F32.TF32", "sass")["passed"])
        self.assertTrue(instruction_audit.audit_instruction_text("cuda_fp32_fma", "FFMA.RN R0, R1, R2, R3", "sass")["passed"])
        self.assertFalse(instruction_audit.audit_instruction_text("cuda_fp32_fma", "FFMA R0, R1, R2, R3\nHMMA.16816.F32.BF16", "sass")["passed"])


class SafeguardTests(unittest.TestCase):
    def test_cache_paths_reject_traversal_absolute_ids_and_bad_names(self):
        root = Path("/tmp/tilelang-calibration-schema-test")
        run = paths.run_cache_path(root, "run-2026.10.27_ok")
        self.assertEqual(run.parent, root)
        for bad in ("", ".", "..", "../escape", "a/b", "a\\b", "/absolute", "white space"):
            with self.subTest(bad=bad), self.assertRaises(paths.CachePathError):
                paths.run_cache_path(root, bad)
        with self.assertRaises(paths.CachePathError):
            paths.run_cache_path("relative/root", "valid")
        for bad in ("../escape", "a/b", ".", "Upper"):
            with self.subTest(bad=bad), self.assertRaises(paths.CachePathError):
                paths.task_cache_path(run, bad)

    def test_task_cache_and_worker_result_stay_inside_run(self):
        run = paths.run_cache_path(Path("/tmp/tilelang-calibration-schema-test"), "safe-run")
        task = paths.task_cache_path(run, "dram_copy")
        self.assertEqual(paths.worker_result_path(run, "dram_copy"), task / "result.json")
        self.assertEqual(paths.artifact_dir_path(run, "dram_copy"), task / "artifacts")
        cache = paths.cache_environment(task)
        self.assertEqual(len(set(cache.values())), 4)
        self.assertEqual(set(cache), {"TILELANG_CACHE_DIR", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR", "TMPDIR"})
        for value in cache.values():
            self.assertTrue(Path(value).is_relative_to(task))

    def test_free_and_compute_process_parsers(self):
        parsed = environment.parse_free_gib("              total used free shared buff/cache available\nMem: 31 2 20 0 8 27\nSwap: 0 0 0\n")
        self.assertTrue(parsed["passed"])
        self.assertEqual(parsed["total_gib"], 31)
        self.assertFalse(environment.parse_free_gib("Mem: 8 1 1 0 6 6\n")["passed"])
        self.assertEqual(environment.parse_compute_apps("\n  \n"), [])
        apps = environment.parse_compute_apps('123,"python, worker",1024 MiB\n')
        self.assertEqual(apps[0]["pid"], "123")
        self.assertEqual(apps[0]["process_name"], "python, worker")

    def test_only_selection_and_cli_list_never_import_gpu_modules(self):
        self.assertEqual(calibrate.parse_only(["sfu_exp2,dram_copy", "barrier"]), ["dram_copy", "sfu_exp2", "barrier"])
        with self.assertRaises(ValueError):
            calibrate.parse_only(["not_a_benchmark"])
        real_import = builtins.__import__

        def no_gpu_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "tilelang"}:
                raise AssertionError(f"--list attempted GPU-module import: {name}")
            return real_import(name, *args, **kwargs)

        output = io.StringIO()
        with mock.patch("builtins.__import__", side_effect=no_gpu_import), mock.patch.object(calibrate, "base_environment", side_effect=AssertionError("GPU preflight called")), mock.patch("sys.stdout", output):
            self.assertEqual(calibrate.main(["--list"]), 0)
        for name in specs.CORE_BENCHMARKS:
            self.assertIn(name, output.getvalue())


if __name__ == "__main__":
    unittest.main()
