"""CPU tests for frozen-winner selection and ABBA acceptance accounting."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.z3_cost_model.gdn import confirm  # noqa: E402


def _config(direction, *, stage=2, group=8, block=None):
    return {
        "direction": direction,
        "block_DV": 16 if direction == "fwd" and block is None else (128 if block is None else block),
        "prepare_h_num_stages": stage,
        "group_reduce_block_size": 16 if direction == "fwd" else group,
    }


def _analysis(direction):
    runs = []
    for arm in confirm.ARMS:
        for repetition in range(3):
            if direction == "fwd":
                config = _config(direction)
            elif arm in (confirm.GUIDED_ARM, confirm.ORACLE_ARM):
                config = _config(direction, stage=2, group=8)
            elif arm == "budget-raw":
                config = _config(direction, stage=1, group=8)
            else:
                config = _config(direction, stage=2, group=16)
            runs.append({"direction": direction, "arm": arm, "repetition": repetition, "best_config": config})
    return {"runs": runs}


def _measurement(plan, pair_index, round_index, config, median):
    return {
        "pair_index": pair_index,
        "round": round_index,
        "config_id": confirm._canonical(config),
        "median_ms": median,
    }


class ConfirmationTest(unittest.TestCase):
    def test_abba_order_is_balanced_and_deterministic(self):
        self.assertEqual(
            [confirm.abba_order(index) for index in range(7)],
            [(0, 1), (1, 0), (1, 0), (0, 1), (0, 1), (1, 0), (1, 0)],
        )
        with self.assertRaises(ValueError):
            confirm.abba_order(-1)

    def test_modal_winner_uses_votes_before_canonical_tiebreak(self):
        analysis = _analysis("fwd")
        for row in analysis["runs"]:
            if row["arm"] == "stock-raw" and row["repetition"] > 0:
                row["best_config"] = _config("fwd", block=32)
        winners = confirm.modal_winners(analysis, "fwd")
        self.assertEqual(winners["stock-raw"]["config"]["block_DV"], 32)
        self.assertEqual(winners["stock-raw"]["votes"], 2)
        self.assertEqual(winners[confirm.GUIDED_ARM]["config"]["block_DV"], 16)

    def test_bwd_plan_deduplicates_configs_without_losing_control_relations(self):
        plan = confirm.confirmation_plan(_analysis("bwd"), "bwd")
        self.assertEqual(len(plan["candidates"]), 3)
        self.assertEqual(len(plan["comparisons"]), 3)
        identity = next(row for row in plan["comparisons"] if row["identical"])
        self.assertIn(confirm.ORACLE_ARM, identity["identical_arms"])
        random_group = next(row for row in plan["comparisons"] if "budget-random" in row["alternative_arms"])
        self.assertIn("stock-raw", random_group["alternative_arms"])
        self.assertIn("equal_budget_control", random_group["relations"])

    def test_quality_accounting_retains_a_failed_equal_budget_comparison(self):
        plan = confirm.confirmation_plan(_analysis("bwd"), "bwd")
        guided = plan["winners"][confirm.GUIDED_ARM]["config"]
        measurements = []
        for pair_index, comparison in enumerate(plan["comparisons"]):
            if comparison["identical"]:
                continue
            for round_index in range(7):
                # The budget-raw pair deliberately loses 2:1.  The stock pair wins.
                guided_latency = 2.0 if "budget-raw" in comparison["alternative_arms"] else 1.0
                measurements.append(_measurement(plan, pair_index, round_index, guided, guided_latency))
                measurements.append(_measurement(plan, pair_index, round_index, comparison["config"], 1.0))
        results = confirm.evaluate_comparisons(plan, measurements)
        oracle = next(row for row in results if "same_space_exhaustive" in row["relations"])
        self.assertEqual(oracle["quality_ratio_guided_over_alternative"], 1.0)
        self.assertTrue(oracle["passed"])
        budget = next(row for row in results if "budget-raw" in row["alternative_arms"])
        self.assertEqual(budget["quality_ratio_guided_over_alternative"], 2.0)
        self.assertFalse(budget["passed"])
        self.assertFalse(all(row["passed"] for row in results))

    def test_forward_identical_configs_are_not_fake_head_to_head_wins(self):
        plan = confirm.confirmation_plan(_analysis("fwd"), "fwd")
        self.assertEqual(len(plan["candidates"]), 1)
        self.assertEqual(len(plan["comparisons"]), 1)
        comparison = plan["comparisons"][0]
        self.assertTrue(comparison["identical"])
        self.assertEqual(set(comparison["identical_arms"]), set(confirm.ARMS))
        config = plan["winners"][confirm.GUIDED_ARM]["config"]
        records = [_measurement(plan, -1, index, config, 0.16 + index * 0.001) for index in range(7)]
        result = confirm.evaluate_comparisons(plan, records)[0]
        self.assertEqual(result["quality_ratio_guided_over_alternative"], 1.0)
        self.assertTrue(result["passed"])

    def test_analysis_and_json_validation(self):
        with self.assertRaises(ValueError):
            confirm.modal_winners({"runs": []}, "fwd")
        analysis = _analysis("fwd")
        analysis["runs"][0]["best_config"] = None
        with self.assertRaises(ValueError):
            confirm.modal_winners(analysis, "fwd")
        self.assertEqual(json.loads(confirm._canonical({"b": 1, "a": 2})), {"a": 2, "b": 1})
        self.assertEqual(copy.deepcopy(confirm.confirmation_plan(_analysis("fwd"), "fwd"))["direction"], "fwd")


if __name__ == "__main__":
    unittest.main()
