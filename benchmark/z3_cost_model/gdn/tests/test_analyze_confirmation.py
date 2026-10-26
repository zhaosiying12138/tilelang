"""Tests for joining cold counts to independently confirmed GDN latencies."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.z3_cost_model.gdn import analyze_confirmation  # noqa: E402


def _config(stage, group):
    return {
        "direction": "bwd",
        "block_DV": 128,
        "prepare_h_num_stages": stage,
        "group_reduce_block_size": group,
    }


def _cold():
    return {
        "aggregates": [
            {
                "direction": direction,
                "arm": arm,
                "compile_attempts": 4,
                "profile_attempts": 3,
                "end_to_end_wall_seconds": 40.0,
                "search_speedup": 1.25,
            }
            for direction in ("fwd", "bwd")
            for arm in analyze_confirmation.ARMS
        ]
    }


def _summary(direction):
    winners = {
        "stock-raw": _config(2, 8),
        "z3-legal-exhaustive": _config(2, 8),
        "z3-cm-topk": _config(2, 8),
        "budget-raw": _config(1, 8),
        "budget-random": _config(2, 32),
    }
    if direction == "fwd":
        for arm in winners:
            winners[arm] = dict(winners[arm], direction="fwd", block_DV=16, group_reduce_block_size=16, prepare_h_num_stages=2)
    guided_config = winners["z3-cm-topk"]
    comparisons = [{
        "config": guided_config,
        "identical": True,
        "identical_arms": ["stock-raw", "z3-legal-exhaustive", "z3-cm-topk"],
        "alternative_arms": [],
        "guided_median_ms": 4.2,
        "alternative_median_ms": 4.2,
    }]
    if direction == "bwd":
        comparisons.extend([
            {
                "config": winners["budget-raw"],
                "identical": False,
                "identical_arms": [],
                "alternative_arms": ["budget-raw"],
                "alternative_median_ms": 4.4,
            },
            {
                "config": winners["budget-random"],
                "identical": False,
                "identical_arms": [],
                "alternative_arms": ["budget-random"],
                "alternative_median_ms": 4.3,
            },
        ])
    else:
        comparisons[0]["identical_arms"] = list(analyze_confirmation.ARMS)
    return {
        "status": "ok",
        "num_tokens": 8192,
        "rounds": 7,
        "plan": {"selection_rule": "candidate-median", "winners": {arm: {"config": config} for arm, config in winners.items()}},
        "comparisons": comparisons,
        "validations": [{"max_relative_l2": 0.01}],
    }


class ConfirmationAnalysisTest(unittest.TestCase):
    def test_every_arm_has_one_config_and_one_confirmed_latency(self):
        rows, comparisons = analyze_confirmation.confirmed_records(
            _cold(),
            {"fwd": _summary("fwd"), "bwd": _summary("bwd")},
        )
        self.assertEqual(len(rows), 10)
        self.assertEqual(len(comparisons), 4)
        self.assertEqual({(row["direction"], row["method"]) for row in rows}, {
            (direction, arm) for direction in ("fwd", "bwd") for arm in analyze_confirmation.ARMS
        })
        guided = next(row for row in rows if row["direction"] == "bwd" and row["method"] == "z3-cm-topk")
        self.assertEqual(guided["best_latency"], 4.2)
        self.assertEqual(guided["latency_scope"], "confirmed")
        random = next(row for row in rows if row["direction"] == "bwd" and row["method"] == "budget-random")
        self.assertEqual(random["best_latency"], 4.3)

    def test_duplicate_assignment_is_rejected(self):
        summary = _summary("bwd")
        summary["comparisons"].append({
            "config": _config(2, 32),
            "identical": False,
            "identical_arms": [],
            "alternative_arms": ["budget-random"],
            "alternative_median_ms": 4.3,
        })
        with self.assertRaisesRegex(ValueError, "Duplicate confirmed assignment"):
            analyze_confirmation.confirmed_records(_cold(), {"fwd": _summary("fwd"), "bwd": summary})

    def test_summary_serializes_finite_json(self):
        rows, _ = analyze_confirmation.confirmed_records(_cold(), {"fwd": _summary("fwd"), "bwd": _summary("bwd")})
        encoded = json.dumps(rows, allow_nan=False)
        self.assertIn("candidate-median", encoded)


if __name__ == "__main__":
    unittest.main()
