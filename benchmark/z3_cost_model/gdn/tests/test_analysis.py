"""Pure-CPU tests for GDN result and same-repetition prediction analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
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


def _config(direction="fwd", block=16, stages=2, group=16):
    return {
        "direction": direction,
        "block_DV": block,
        "prepare_h_num_stages": stages,
        "group_reduce_block_size": group,
    }


def _canonical(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _config_id(config):
    return hashlib.sha256(_canonical(config).encode("utf-8")).hexdigest()


def _write_run(
    root,
    direction,
    arm,
    repetition,
    *,
    cache_track="cold",
    scores=(),
    trials=(),
    legal_candidates=None,
    score_trials=(),
    run_status="ok",
):
    run_dir = root / cache_track / direction / f"{arm}__rep-{repetition:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    budget = analyze_predictions.ACTUAL_BUDGET[direction]
    events = [{
        "event": "run_start",
        "arm": arm,
        "top_k": budget if arm == analyze_predictions.SCORE_ARM else None,
        "metadata": {
            "direction": direction,
            "arm": arm,
            "repetition": repetition,
            "cache_track": cache_track,
            "shape": {"T": 8192},
        },
    }]
    if legal_candidates is not None:
        for index, config in enumerate(legal_candidates):
            events.append({
                "event": "prune",
                "action": "retained_unknown",
                "index": index,
                "config": config,
                "config_id": _config_id(config),
            })
        events.append({"event": "prune_end", "status": "ok", "audit": {"kept": len(legal_candidates)}})
    for index, (config, score) in enumerate(scores):
        events.append({
            "event": "score",
            "index": index,
            "sequence": 100 + index,
            "config": config,
            "config_id": _config_id(config),
            "score": score,
        })
    all_trials = list(trials) + list(score_trials)
    for index, (config, status, latency) in enumerate(all_trials):
        events.append({
            "event": "trial",
            "index": index,
            "sequence": 200 + index,
            "config": config,
            "config_id": _config_id(config),
            "status": status,
            "latency_ms": latency,
        })
    events.append({"event": "run_end", "status": run_status})
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    selected = len(legal_candidates) if legal_candidates is not None else budget if arm == analyze_predictions.SCORE_ARM else len(trials)
    (run_dir / "summary.json").write_text(json.dumps({
        "arm": arm,
        "status": run_status,
        "R": len(trials),
        "U": len(legal_candidates) if legal_candidates is not None else None,
        "selected": selected,
        "experiment": {"repetition": repetition},
        "metadata": {"direction": direction, "arm": arm, "cache_track": cache_track},
    }), encoding="utf-8")
    return run_dir / "events.jsonl"


def _write_score_and_oracles(
    root,
    direction,
    repetition,
    scores,
    legal_trials=None,
    stock_trials=None,
    *,
    legal_candidates=None,
    score_trials=(),
    cache_track="cold",
):
    _write_run(
        root,
        direction,
        analyze_predictions.SCORE_ARM,
        repetition,
        cache_track=cache_track,
        scores=scores,
        trials=score_trials,
    )
    if legal_trials is not None:
        if legal_candidates is None:
            legal_candidates = [trial[0] for trial in legal_trials]
        _write_run(
            root,
            direction,
            analyze_predictions.LEGAL_ARM,
            repetition,
            cache_track=cache_track,
            trials=legal_trials,
            legal_candidates=legal_candidates,
        )
    if stock_trials is not None:
        _write_run(
            root,
            direction,
            analyze_predictions.STOCK_ARM,
            repetition,
            cache_track=cache_track,
            trials=stock_trials,
        )


def _measured_by_config(group):
    return {row["config_id"]: row["measured"] for row in group["rows"]}


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

    def test_prediction_collection_never_mixes_cache_tracks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latencies = {"cold": 0.1, "warm": 0.2, "no-cache": 0.3}
            for track, latency in latencies.items():
                _write_score_and_oracles(
                    root,
                    "fwd",
                    0,
                    [(_CONFIG, 1.0)],
                    [(_CONFIG, "ok", latency)],
                    cache_track=track,
                )
            for track, latency in latencies.items():
                groups = analyze_predictions.collect(root, cache_track=track, repetitions=(0,), directions=("fwd",))
                self.assertEqual(len(groups), 1)
                self.assertEqual(groups[0]["rows"][0]["measured"], latency)
                self.assertEqual(groups[0]["paired_denominator"], 1)
            default_group = analyze_predictions.collect(root, repetitions=(0,), directions=("fwd",))[0]
            self.assertEqual(default_group["rows"][0]["measured"], 0.1)

    def test_unknown_track_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze_results.collect_runs("unused", cache_track="shared")
        with self.assertRaises(ValueError):
            analyze_predictions.collect("unused", cache_track="shared")


class PredictionPairingTest(unittest.TestCase):
    def test_complete_oracles_never_pool_arms_or_repetitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = _config(block=16), _config(block=32)
            legal_latencies = {0: (10.0, 20.0), 1: (100.0, 50.0)}
            for repetition, (latency_a, latency_b) in legal_latencies.items():
                _write_score_and_oracles(
                    root,
                    "fwd",
                    repetition,
                    [(a, 1.0), (b, 2.0)],
                    [(a, "ok", latency_a), (b, "ok", latency_b)],
                    [(a, "ok", 0.01), (b, "ok", 0.02)],
                    score_trials=[(a, "ok", 900.0), (b, "ok", 901.0)],
                )
            groups = analyze_predictions.collect(root, repetitions=(0, 1), directions=("fwd",))
            self.assertEqual(len(groups), 2)
            for group in groups:
                expected = legal_latencies[group["repetition"]]
                self.assertEqual(_measured_by_config(group), {_config_id(a): expected[0], _config_id(b): expected[1]})
                self.assertEqual(group["paired_denominator"], 2)
                self.assertEqual(group["oracle_kind"], "z3-legal-exhaustive")
                self.assertTrue(group["legal_oracle_complete"])
                self.assertIn(f"rep-{group['repetition']:04d}", group["oracle_events_path"])
                self.assertTrue(all(row["sample_count"] == 1 for row in group["rows"]))

    def test_same_repetition_paired_denominator_is_not_backfilled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b, c = _config(block=16), _config(block=32), _config(block=64)
            scores = [(a, 1.0), (b, 2.0), (c, 3.0)]
            _write_score_and_oracles(root, "fwd", 0, scores, [(a, "ok", 10.0), (b, "ok", 20.0)])
            _write_score_and_oracles(
                root,
                "fwd",
                1,
                scores,
                [(a, "ok", 11.0), (b, "ok", 21.0), (c, "ok", 31.0)],
            )
            groups = analyze_predictions.collect(root, repetitions=(0, 1), directions=("fwd",))
            self.assertEqual([group["paired_denominator"] for group in groups], [2, 3])
            self.assertEqual(groups[0]["missing_measured_config_ids"], [_config_id(c)])
            self.assertNotIn(_config_id(c), _measured_by_config(groups[0]))
            self.assertEqual(_measured_by_config(groups[1])[_config_id(c)], 31.0)
            self.assertEqual([row["sample_count"] for group in groups for row in group["rows"]], [1] * 5)

    def test_canonical_json_breaks_prediction_and_measured_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Canonical JSON orders group=16 before group=8 for the exact score tie.
            a = _config("bwd", block=128, stages=2, group=16)
            b = _config("bwd", block=128, stages=2, group=8)
            c = _config("bwd", block=128, stages=2, group=32)
            scores = [(b, 1.0), (a, 1.0), (c, 2.0)]  # Deliberately reverse the tied canonical order.
            _write_score_and_oracles(
                root,
                "bwd",
                0,
                scores,
                [(b, "ok", 1.0), (a, "ok", 2.0), (c, "ok", 3.0)],
            )
            _write_score_and_oracles(
                root,
                "bwd",
                1,
                scores,
                [(b, "ok", 1.0), (a, "ok", 1.0), (c, "ok", 3.0)],
            )
            groups = analyze_predictions.collect(root, repetitions=(0, 1), directions=("bwd",))
            analysis = analyze_predictions.rank_analysis(groups)
            first, second = analysis["per_repetition"]
            self.assertEqual(first["canonical_prediction_order"], [_config_id(a), _config_id(b), _config_id(c)])
            self.assertEqual(first["canonical_measured_order"], [_config_id(b), _config_id(a), _config_id(c)])
            self.assertEqual(first["top_k_recall"], {"1": 0.0, "2": 1.0, "3": 1.0})
            self.assertEqual(first["top_k_selections"]["1"]["predicted_config_ids"], [_config_id(a)])
            self.assertEqual(first["top_k_selections"]["1"]["measured_config_ids"], [_config_id(b)])
            self.assertEqual(first["oracle_best_config_id"], _config_id(b))
            self.assertEqual(first["oracle_best_prediction_rank"], 2)
            self.assertEqual(first["oracle_best_prediction_average_rank"], 1.5)
            self.assertTrue(first["oracle_best_in_actual_deterministic_top_k"])
            self.assertEqual(len(first["prediction_ties"]), 1)
            self.assertEqual(first["prediction_ties"][0]["config_ids"], [_config_id(a), _config_id(b)])
            self.assertEqual(first["prediction_ties"][0]["canonical_ranks"], [1, 2])
            tied_rows = [row for row in analysis["rows"] if row["repetition"] == 0 and row["config_id"] in {_config_id(a), _config_id(b)}]
            self.assertEqual([row["predicted_average_rank"] for row in tied_rows], [1.5, 1.5])
            self.assertTrue(all(row["prediction_exact_tie"] and row["prediction_tie_size"] == 2 for row in tied_rows))
            # Equal measured latencies also use canonical JSON, not trial input order.
            self.assertEqual(second["canonical_measured_order"], [_config_id(a), _config_id(b), _config_id(c)])
            self.assertEqual(second["top_k_recall"]["1"], 1.0)
            self.assertEqual(second["measured_ties"][0]["config_ids"], [_config_id(a), _config_id(b)])

    def test_incomplete_or_missing_legal_oracle_uses_same_rep_stock_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = _config(block=16), _config(block=32)
            scores = [(a, 1.0), (b, 2.0)]
            _write_score_and_oracles(
                root,
                "fwd",
                0,
                scores,
                [(a, "ok", 100.0), (b, "compile_error", None)],
                [(a, "ok", 7.0), (b, "ok", 8.0)],
            )
            _write_score_and_oracles(root, "fwd", 1, scores, None, [(a, "ok", 17.0), (b, "ok", 18.0)])
            _write_score_and_oracles(
                root,
                "fwd",
                2,
                scores,
                [(a, "ok", 70.0), (b, "ok", 80.0)],
                [(a, "ok", 0.1), (b, "ok", 0.2)],
            )
            groups = analyze_predictions.collect(root, repetitions=(0, 1, 2), directions=("fwd",))
            self.assertEqual([group["legal_oracle_status"] for group in groups], ["incomplete", "unavailable", "complete"])
            self.assertEqual([group["legal_oracle_complete"] for group in groups], [False, False, True])
            self.assertEqual([group["oracle_kind"] for group in groups], [
                "stock-raw-successful-fallback",
                "stock-raw-successful-fallback",
                "z3-legal-exhaustive",
            ])
            self.assertEqual(_measured_by_config(groups[0]), {_config_id(a): 7.0, _config_id(b): 8.0})
            self.assertEqual(_measured_by_config(groups[1]), {_config_id(a): 17.0, _config_id(b): 18.0})
            self.assertEqual(_measured_by_config(groups[2]), {_config_id(a): 70.0, _config_id(b): 80.0})
            self.assertEqual(groups[0]["legal_candidate_count"], 2)
            self.assertEqual(groups[0]["legal_successful_count"], 1)
            self.assertEqual(groups[0]["legal_missing_config_ids"], [_config_id(b)])
            self.assertIn("without successful measurements", groups[0]["fallback_reason"])
            for group in groups[:2]:
                self.assertIn("stock-raw", group["oracle_events_path"])
                self.assertIn(f"rep-{group['repetition']:04d}", group["oracle_events_path"])
                self.assertEqual(group["paired_denominator"], 2)
                self.assertTrue(all(row["sample_count"] == 1 for row in group["rows"]))
            analysis = analyze_predictions.rank_analysis(groups)
            self.assertEqual(analysis["suites"]["fwd"]["legal_oracle_complete"], [False, False, True])
            fallback_rows = [row for row in analysis["rows"] if row["repetition"] == 0]
            self.assertTrue(all(not row["legal_oracle_complete"] for row in fallback_rows))
            self.assertTrue(all(row["oracle_kind"] == "stock-raw-successful-fallback" for row in fallback_rows))

    def test_aggregate_is_median_of_repetition_metrics_not_pooled_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b, c = _config(block=16), _config(block=32), _config(block=64)
            scores = [(a, 1.0), (b, 2.0), (c, 3.0)]
            latencies = ((1.0, 2.0, 3.0), (1.0, 3.0, 2.0), (3.0, 2.0, 1.0))
            for repetition, values in enumerate(latencies):
                _write_score_and_oracles(
                    root,
                    "fwd",
                    repetition,
                    scores,
                    [(a, "ok", values[0]), (b, "ok", values[1]), (c, "ok", values[2])],
                )
            groups = analyze_predictions.collect(root, repetitions=(0, 1, 2), directions=("fwd",))
            analysis = analyze_predictions.rank_analysis(groups)
            aggregate = analysis["suites"]["fwd"]
            per_repetition = aggregate["per_repetition"]
            self.assertEqual([detail["spearman"] for detail in per_repetition], [1.0, 0.5, -1.0])
            self.assertEqual(aggregate["spearman"], 0.5)
            self.assertEqual(
                aggregate["kendall_tau_b"],
                statistics.median(detail["kendall_tau_b"] for detail in per_repetition),
            )
            self.assertEqual([detail["top_k_recall"]["1"] for detail in per_repetition], [1.0, 1.0, 0.0])
            self.assertEqual(aggregate["top_k_recall"]["1"], 1.0)
            self.assertEqual([detail["oracle_best_prediction_rank"] for detail in per_repetition], [1, 1, 3])
            self.assertEqual(aggregate["oracle_best_prediction_rank"], 1)
            self.assertEqual(aggregate["paired_denominators"], [3, 3, 3])
            self.assertEqual(aggregate["per_candidate_sample_counts"], [1, 1, 1])
            self.assertEqual(aggregate["aggregate_repetition_count"], 3)


if __name__ == "__main__":
    unittest.main()
