"""Cache-track accounting tests for GDN result and prediction aggregation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.z3_cost_model.gdn import analyze_predictions, analyze_results  # noqa: E402


_CONFIG = {
    "direction": "fwd",
    "block_DV": 16,
    "prepare_h_num_stages": 2,
    "group_reduce_block_size": 16,
}


class AnalysisCacheTrackTest(unittest.TestCase):
    def test_result_collection_defaults_to_cold_and_separates_sensitivity_tracks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for track in ("cold", "warm", "no-cache"):
                path = root / track / "summary.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({
                    "gdn": {"direction": "fwd", "cache_track": track},
                    "metadata": {"direction": "fwd", "cache_track": track},
                    "arm": "stock-raw",
                    "experiment": {"repetition": 0},
                    "status": "ok",
                }), encoding="utf-8")
            for track in ("cold", "warm", "no-cache"):
                runs = analyze_results.collect_runs(root, cache_track=track)
                self.assertEqual(len(runs), 1)
                self.assertEqual(runs[0]["cache_track"], track)
            self.assertEqual(analyze_results.collect_runs(root)[0]["cache_track"], "cold")

    def test_prediction_collection_never_mixes_warm_or_no_cache_latencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latencies = {"cold": 0.1, "warm": 0.2, "no-cache": 0.3}
            for track, latency in latencies.items():
                path = root / track / "events.jsonl"
                path.parent.mkdir(parents=True)
                events = [
                    {
                        "event": "run_start",
                        "metadata": {"direction": "fwd", "repetition": 0, "cache_track": track},
                    },
                    {"event": "score", "config": _CONFIG, "config_id": "candidate", "score": 1.0},
                    {"event": "trial", "status": "ok", "config": _CONFIG, "latency_ms": latency},
                ]
                path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            for track, latency in latencies.items():
                predicted, measured = analyze_predictions.collect(root, cache_track=track)
                self.assertEqual(len(predicted), 1)
                self.assertEqual(measured[0]["measured"], latency)
            self.assertEqual(analyze_predictions.collect(root)[1][0]["measured"], 0.1)

    def test_unknown_track_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze_results.collect_runs("unused", cache_track="shared")
        with self.assertRaises(ValueError):
            analyze_predictions.collect("unused", cache_track="shared")


if __name__ == "__main__":
    unittest.main()
