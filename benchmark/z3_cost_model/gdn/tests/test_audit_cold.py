"""Synthetic tests for the formal cold-run protocol audit."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.z3_cost_model.gdn import audit_cold  # noqa: E402


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_root(root):
    _write(root / "protocol.json", {
        "schema": "tilelang.gdn.cold-fresh-protocol/v1",
        "directions": ["fwd", "bwd"],
        "repetitions": [0],
        "arms": list(audit_cold.ARMS),
        "cache_track": "cold",
        "warmup": 10,
        "repeat": 50,
        "fresh_process_per_arm_repetition": True,
        "script_sha256": "a" * 64,
    })
    pid = 1000
    for direction in audit_cold.DIRECTIONS:
        for arm in audit_cold.ARMS:
            pid += 1
            run = root / direction / f"{arm}__rep-0000"
            expected = audit_cold.EXPECTED[direction]
            selected = expected["R"] if arm == "stock-raw" else expected["U"] if arm == "z3-legal-exhaustive" else expected["top_k"]
            _write(run / "summary.json", {
                "gdn": {"direction": direction, "cache_track": "cold"},
                "arm": arm,
                "experiment": {"repetition": 0},
                "status": "ok",
                "R": expected["R"],
                "U": expected["U"],
                "selected": selected,
                "compile_attempts": selected,
                "profile": {"attempts": 1, "success": 1},
                "best_latency_ms": 1.0,
                "end_to_end_wall_seconds": 2.0,
                "cache": {"pipeline_cache_directory": str(run / "cache")},
            })
            _write(run / "config.json", {"warmup": 10, "repeat": 50, "repetition": 0, "cache_track": "cold"})
            _write(run / "environment.json", {
                "pid": pid,
                "environment": {
                    "TILELANG_DISABLE_CACHE": "0",
                    "CUDA_CACHE_DISABLE": "0",
                    "TILELANG_CACHE_DIR": str(run / "cache"),
                },
            })
            _write(run / "before.json", {"empty": True, "entries": [], "cache_entries": []})
            _write(run / "after.json", {"entries": [], "cache_entries": []})
            _write(run / "events.jsonl", {})
    return pid


class ColdAuditTest(unittest.TestCase):
    def test_complete_ten_run_protocol_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_root(root)
            report = audit_cold.audit_cold_runs(root, repetitions=(0,))
            self.assertTrue(report["passed"], report["errors"])
            self.assertEqual(report["observed_runs"], 10)
            self.assertEqual(report["distinct_pids"], 10)

    def test_pid_reuse_and_changed_warmup_are_reported_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_root(root)
            paths = sorted(root.rglob("environment.json"))
            first = json.loads(paths[0].read_text())
            second_path = paths[1]
            second = json.loads(second_path.read_text())
            second["pid"] = first["pid"]
            _write(second_path, second)
            config_path = paths[0].parent / "config.json"
            config = json.loads(config_path.read_text())
            config["warmup"] = 2
            _write(config_path, config)
            report = audit_cold.audit_cold_runs(root, repetitions=(0,))
            self.assertFalse(report["passed"])
            self.assertTrue(any("process reuse" in error for error in report["errors"]))
            self.assertTrue(any("config warmup" in error for error in report["errors"]))

    def test_missing_run_and_non_cold_cache_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_root(root)
            summary_path = next(root.rglob("summary.json"))
            summary = json.loads(summary_path.read_text())
            summary["gdn"]["cache_track"] = "warm"
            _write(summary_path, summary)
            report = audit_cold.audit_cold_runs(root, repetitions=(0,))
            self.assertFalse(report["passed"])
            self.assertEqual(report["observed_runs"], 9)
            self.assertTrue(any("non-cold" in error for error in report["errors"]))
            self.assertTrue(any("missing runs" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
