"""Pure-CPU synthetic tests for the classic formal cold-run audit."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.z3_cost_model.operators import audit_cold  # noqa: E402


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_run(root, suite, arm, repetition, pid, *, top_k=audit_cold.TOP_K):
    run = root / suite / f"top-k-{top_k:04d}" / f"{arm}__rep-{repetition:04d}"
    run.mkdir(parents=True)
    (run / "cache").mkdir()
    expected = audit_cold.EXPECTED[suite]
    selected = expected["R"] if arm == "stock-raw" else expected["U"] if arm == "z3-legal-exhaustive" else top_k
    cache = run / "cache"
    _write(
        run / "summary.json",
        {
            "status": "ok",
            "arm": arm,
            "operator_run": {"suite": suite, "top_k": top_k, "cache_track": "cold"},
            "experiment": {"arm": arm, "repetition": repetition, "cache_mode": "enabled"},
            "R": expected["R"],
            "U": None if arm == "stock-raw" else expected["U"],
            "selected": selected,
            "compile_attempts": selected,
            "profile": {"attempts": selected, "success": selected},
            "best_latency_ms": 1.0 + repetition / 10,
            "end_to_end_wall_seconds": 2.0 + repetition,
            "cache": {"pipeline_cache_directory": str(cache)},
        },
    )
    _write(
        run / "config.json",
        {
            "suite": suite,
            "arm": arm,
            "top_k": top_k,
            "seed": audit_cold.SEED,
            "warmup": 10,
            "repeat": 50,
            "repetition": repetition,
            "cache_track": "cold",
        },
    )
    _write(
        run / "environment.json",
        {
            "suite": suite,
            "arm": arm,
            "seed": audit_cold.SEED,
            "cache_track": "cold",
            "pid": pid,
            "environment": {
                "TILELANG_DISABLE_CACHE": "0",
                "CUDA_CACHE_DISABLE": "0",
                "TILELANG_CACHE_DIR": str(cache),
            },
        },
    )
    _write(run / "before.json", {"empty": True, "entries": [], "cache_entries": []})
    cache_entry = {"path": "cache", "type": "directory"}
    _write(run / "after.json", {"entries": [cache_entry], "cache_entries": [cache_entry]})
    (run / "events.jsonl").write_text('{"event":"run_end","status":"ok"}\n', encoding="utf-8")
    return run


def _make_root(root):
    runs = {}
    pid = 10000
    for suite in audit_cold.SUITES:
        for repetition in audit_cold.REPETITIONS:
            for arm in audit_cold.ARMS:
                pid += 1
                key = (suite, arm, repetition)
                runs[key] = _make_run(root, suite, arm, repetition, pid)
    if set(runs) != audit_cold.EXPECTED_KEYS or len(runs) != 60:
        raise AssertionError("the synthetic fixture did not construct every formal identity")
    return runs


class ClassicColdAuditTest(unittest.TestCase):
    def test_complete_sixty_run_root_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            runs = _make_root(root)
            self.assertEqual(len(runs), 60)
            self.assertTrue(all((run / "summary.json").is_file() for run in runs.values()))

            report = audit_cold.audit_cold_runs(root)

            self.assertTrue(report["passed"], report["errors"])
            self.assertEqual(report["expected_runs"], 60)
            self.assertEqual(report["observed_runs"], 60)
            self.assertEqual(report["distinct_pids"], 60)
            self.assertEqual(report["ignored_sensitivity_runs"], 0)
            self.assertEqual(report["errors"], [])
            self.assertEqual(len(report["runs"]), 60)
            self.assertEqual(
                {(row["suite"], row["arm"], row["repetition"]) for row in report["runs"]},
                audit_cold.EXPECTED_KEYS,
            )
            self.assertTrue(all(row["cache_entries_after"] == 1 for row in report["runs"]))
            self.assertEqual(report, json.loads(json.dumps(report, allow_nan=False, sort_keys=True)))

            output = Path(directory) / "audit.json"
            with redirect_stdout(io.StringIO()):
                returncode = audit_cold.main(["--root", str(root), "--output", str(output)])
            self.assertEqual(returncode, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)

    def test_pid_reuse_fails_despite_complete_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            runs = _make_root(root)
            keys = list(runs)
            first = json.loads((runs[keys[0]] / "environment.json").read_text(encoding="utf-8"))
            second_path = runs[keys[1]] / "environment.json"
            second = json.loads(second_path.read_text(encoding="utf-8"))
            second["pid"] = first["pid"]
            _write(second_path, second)

            report = audit_cold.audit_cold_runs(root)

            self.assertFalse(report["passed"])
            self.assertEqual(report["observed_runs"], 60)
            self.assertEqual(report["distinct_pids"], 59)
            self.assertTrue(any("process reuse" in error for error in report["errors"]), report["errors"])

    def test_missing_run_writes_audit_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            runs = _make_root(root)
            missing_key = ("gemm_short_k", "budget-random", 1)
            shutil.rmtree(runs[missing_key])
            output = Path(directory) / "incomplete-audit.json"

            with redirect_stdout(io.StringIO()):
                returncode = audit_cold.main(["--root", str(root), "--output", str(output)])

            self.assertEqual(returncode, 1)
            self.assertTrue(output.is_file())
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertEqual(report["expected_runs"], 60)
            self.assertEqual(report["observed_runs"], 59)
            self.assertEqual(report["distinct_pids"], 59)
            self.assertEqual(len(report["runs"]), 59)
            self.assertTrue(
                any("missing runs" in error and repr(missing_key) in error for error in report["errors"]),
                report["errors"],
            )

    def test_pipeline_and_tilelang_cache_escapes_are_both_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            runs = _make_root(root)
            key = ("conv2d_stride_tail", "z3-cm-topk", 2)
            run = runs[key]
            escaped = Path(directory) / "outside-cache"
            summary_path = run / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["cache"]["pipeline_cache_directory"] = str(escaped)
            _write(summary_path, summary)
            environment_path = run / "environment.json"
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
            environment["environment"]["TILELANG_CACHE_DIR"] = str(escaped)
            _write(environment_path, environment)

            report = audit_cold.audit_cold_runs(root)

            self.assertFalse(report["passed"])
            self.assertTrue(any("pipeline cache escapes" in error for error in report["errors"]), report["errors"])
            self.assertTrue(any("TileLang cache identity mismatch" in error for error in report["errors"]), report["errors"])

    def test_sensitivity_summaries_are_ignored_and_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            _make_root(root)
            for arm in audit_cold.ARMS:
                # Reusing a formal pid proves ignored runs do not enter PID accounting.
                _make_run(root, "gemm_square", arm, 0, 10001, top_k=5)

            report = audit_cold.audit_cold_runs(root)

            self.assertTrue(report["passed"], report["errors"])
            self.assertEqual(report["expected_runs"], 60)
            self.assertEqual(report["observed_runs"], 60)
            self.assertEqual(report["distinct_pids"], 60)
            self.assertEqual(report["ignored_sensitivity_runs"], 5)
            self.assertEqual(len(report["runs"]), 60)
            self.assertFalse(any("top-k-0005" in row["path"] for row in report["runs"]))


if __name__ == "__main__":
    unittest.main()
