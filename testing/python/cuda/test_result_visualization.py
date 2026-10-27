"""Synthetic CPU-only tests for result plotting and calculations.

Figures are written only to pytest temporary directories.  They are test
artifacts, not experimental results, and are never used by the design document.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


os.environ.setdefault("MPLBACKEND", "Agg")
pytest.importorskip("matplotlib")

_MODULE_PATH = Path(__file__).parents[3] / "benchmark" / "z3_cost_model" / "visualization.py"
_SPEC = importlib.util.spec_from_file_location("_tilelang_result_visualization_under_test", _MODULE_PATH)
vis = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = vis
_SPEC.loader.exec_module(vis)


def _assert_artifacts(paths, output):
    base = Path(output)
    if base.suffix:
        base = base.with_suffix("")
    assert set(paths) == {"png", "svg", "csv", "json"}
    for suffix in (".png", ".svg", ".csv", ".json"):
        artifact = base.with_suffix(suffix)
        assert artifact.is_file(), artifact
        assert artifact.stat().st_size > 0, artifact
        assert Path(paths[suffix[1:]]) == artifact
    assert base.with_suffix(".png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with base.with_suffix(".json").open(encoding="utf-8") as stream:
        payload = json.load(stream)
    with base.with_suffix(".csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert payload["schema_version"] == 1
    assert payload["rows"]
    assert rows
    return payload, rows


def _loop_schedule():
    return {
        "schedule_kind": "loop",
        "num_stages": 3,
        "num_tasks": 3,
        "minimal_II": 4,
        "latencies": [3, 2, 1],
        "iis": [2, 1, 1],
        "resource_flags": [1, 8, 17],
        "resource_bits": 7,
        "start_times": [0, 4, 2],
        "sorted_indices": [0, 2, 1],
        "makespan": 6,
        "fallback": False,
    }


def test_native_loop_schedule_json_generates_all_artifacts(tmp_path):
    schedule_path = tmp_path / "schedule_info.json"
    schedule_path.write_text(json.dumps(_loop_schedule()), encoding="utf-8")
    output = tmp_path / "figures" / "loop.png"

    payload, rows = _assert_artifacts(vis.plot_schedule_timeline(schedule_path, output), output)

    assert payload["plot"] == "schedule_timeline"
    assert payload["minimal_II"] == 4
    assert payload["makespan"] == 6
    assert [row["stage"] for row in payload["rows"]] == [0, 1, 0]
    assert [row["residue"] for row in payload["rows"]] == [0, 0, 2]
    assert rows[2]["resources"] == "GLOBAL_LOAD+CUDA_CORE"
    assert [int(row["ii"]) for row in rows] == [2, 1, 1]


def test_native_straight_schedule_does_not_invent_minimum_ii_or_stage(tmp_path):
    info = {
        "schedule_kind": "straight-line",
        "num_tasks": 2,
        "start_times": [0, 3],
        "latencies": [3, 2],
        "iis": [2, 1],
        "resource_flags": [0, 2],
        "resource_bits": 7,
        "makespan": 5,
        "fallback": False,
    }
    output = tmp_path / "straight"
    payload, _ = _assert_artifacts(vis.plot_schedule_timeline(info, output), output)

    assert payload["schedule_kind"] == "straight-line"
    assert payload["minimal_II"] is None
    assert [row["stage"] for row in payload["rows"]] == [None, None]


def test_pruning_funnel_preserves_unknown_and_dedup_accounting(tmp_path):
    summary = {
        "raw": 20,
        "sat": 8,
        "unknown": 2,
        "unsat": 7,
        "dedup": 3,
        "compiled": 6,
        "profiled": 4,
    }
    output = tmp_path / "funnel"
    payload, rows = _assert_artifacts(vis.plot_pruning_funnel(summary, output), output)

    assert payload["retained"] == 10
    roles = {row["category"]: row["role"] for row in payload["rows"]}
    assert roles["unknown"] == "retained_solver_outcome_not_unsat"
    assert roles["dedup"] == "duplicate_removed_not_unsat"
    counts = {row["category"]: int(row["count"]) for row in rows}
    assert counts == {"raw": 20, "sat": 8, "unknown": 2, "unsat": 7, "dedup": 3, "other": 0, "retained": 10, "compiled": 6, "profiled": 4}
    assert sum(counts[name] for name in ("sat", "unknown", "unsat", "dedup")) == counts["raw"]


def test_pruning_funnel_rejects_inconsistent_denominator(tmp_path):
    summary = {"raw": 19, "sat": 8, "unknown": 2, "unsat": 7, "dedup": 3, "compiled": 6, "profiled": 4}
    with pytest.raises(ValueError, match="Inconsistent raw denominator"):
        vis.plot_pruning_funnel(summary, tmp_path / "bad")


def _additive_run():
    return {
        "run": "GDN fwd / z3-cm",
        "timing_semantics": "additive",
        "end_to_end_s": 5,
        "times_s": {"solver": 1, "model": 1, "compile": 1, "validation": 0.5, "profile": 0.5},
    }


def _parallel_run():
    return {
        "run": "GDN bwd / parallel",
        "timing_semantics": "parallel_cumulative",
        "end_to_end_s": 5,
        "times_s": {"solver": 4, "model": 3, "compile": 2, "validation": 1, "profile": 2, "other": 0.5},
    }


def test_additive_time_breakdown_derives_only_residual_other(tmp_path):
    output = tmp_path / "time-additive"
    payload, _ = _assert_artifacts(vis.plot_search_time_breakdown([_additive_run()], output), output)

    assert payload["stacked"] is True
    other = next(row for row in payload["rows"] if row["category"] == "other")
    assert other["seconds"] == 1
    assert other["value_origin"] == "derived_residual"
    assert sum(row["seconds"] for row in payload["rows"]) == payload["runs"][0]["end_to_end_s"]


def test_parallel_cumulative_time_uses_independent_bars_and_warns(tmp_path):
    output = tmp_path / "time-parallel.svg"
    with pytest.warns(RuntimeWarning, match="must not be summed"):
        paths = vis.plot_search_time_breakdown([_additive_run(), _parallel_run()], output)
    payload, _ = _assert_artifacts(paths, output)

    assert payload["stacked"] is False
    assert payload["parallel_cumulative_runs"] == ["GDN bwd / parallel"]
    assert "must not be summed" in payload["warning"]
    assert len(payload["rows"]) == 12


def test_additive_time_breakdown_rejects_unreconciled_explicit_other(tmp_path):
    run = _additive_run()
    run["times_s"]["other"] = 2
    with pytest.raises(ValueError, match="do not match end_to_end_s"):
        vis.plot_search_time_breakdown([run], tmp_path / "bad-time")


def test_pure_python_rank_metrics_handle_ties_and_undefined_correlation():
    assert vis.rankdata([10, 20, 20, 30]) == [1, 2.5, 2.5, 4]
    predicted = [1, 2, 3, 4]
    measured = [1, 3, 2, 4]
    assert vis.spearman_rank_correlation(predicted, measured) == pytest.approx(0.8)
    assert vis.kendall_tau_b(predicted, measured) == pytest.approx(2 / 3)
    assert vis.top_k_recall(predicted, measured, 1) == 1
    assert vis.top_k_recall(predicted, measured, 2) == 0.5
    assert vis.spearman_rank_correlation([1, 1], [2, 3]) is None
    assert vis.kendall_tau_b([1, 1], [2, 3]) is None
    with pytest.raises(ValueError, match="exceeds candidate count"):
        vis.top_k_recall(predicted, measured, 5)
    with pytest.raises(ValueError, match="lengths differ"):
        vis.spearman_rank_correlation([1], [1, 2])


def _prediction_records():
    predictions = []
    measured = []
    fwd_predicted = [1, 2, 3, 4]
    fwd_measured = [1, 3, 2, 4]
    for index, (prediction, actual) in enumerate(zip(fwd_predicted, fwd_measured)):
        predictions.append({"suite": "GDN fwd", "candidate": chr(97 + index), "predicted": prediction})
        measured.append({"suite": "GDN fwd", "candidate": chr(97 + index), "confirmed_latency": actual})
    for index, (prediction, actual) in enumerate(zip([1, 1, 2], [3, 2, 1])):
        predictions.append({"suite": "GDN bwd", "candidate": str(index), "predicted": prediction})
        measured.append({"suite": "GDN bwd", "candidate": str(index), "confirmed_latency": actual})
    return predictions, list(reversed(measured))


def test_prediction_rank_displays_suite_metrics_and_joins_candidate_keys(tmp_path):
    predictions, measured = _prediction_records()
    output = tmp_path / "ranks"
    payload, rows = _assert_artifacts(vis.plot_prediction_rank(predictions, measured, output, top_k=(1, 2, 5)), output)

    assert set(payload["suites"]) == {"GDN fwd", "GDN bwd"}
    fwd = payload["suites"]["GDN fwd"]
    assert fwd["spearman"] == pytest.approx(0.8)
    assert fwd["kendall_tau_b"] == pytest.approx(2 / 3)
    assert fwd["top_k_recall"] == {"1": 1, "2": 0.5, "5": None}
    bwd_rows = [row for row in payload["rows"] if row["suite"] == "GDN bwd"]
    assert [row["predicted_rank"] for row in bwd_rows] == [1.5, 1.5, 3]
    assert len(rows) == 7


def test_prediction_rank_rejects_missing_measured_candidates(tmp_path):
    predictions, measured = _prediction_records()
    with pytest.raises(ValueError, match="candidate keys differ"):
        vis.plot_prediction_rank(predictions, measured[1:], tmp_path / "bad-rank")


def _quality_records():
    return [
        {"suite": "GDN fwd", "method": "stock-raw", "search_space": "raw", "candidate": "raw", "search_wall_s": 0, "profile_count": 0, "confirmed_latency": 5, "latency_unit": "ms"},
        {"suite": "GDN fwd", "method": "control", "search_space": "raw", "candidate": "control", "search_wall_s": 1, "profile_count": 1, "confirmed_latency": 4, "latency_unit": "ms"},
        {"suite": "GDN fwd", "method": "z3-legal", "search_space": "legal", "candidate": "legal", "search_wall_s": 2, "profile_count": 2, "confirmed_latency": 3, "latency_unit": "ms"},
        {"suite": "GDN fwd", "method": "z3-cm", "search_space": "legal", "candidate": "c0", "search_wall_s": 1, "profile_count": 1, "confirmed_latency": 4, "latency_unit": "ms"},
        {"suite": "GDN fwd", "method": "z3-cm", "search_space": "legal", "candidate": "c1", "search_wall_s": 2, "profile_count": 2, "confirmed_latency": 2.5, "latency_unit": "ms"},
        {"suite": "GDN fwd", "method": "z3-cm", "search_space": "legal", "candidate": "c2", "search_wall_s": 3, "profile_count": 3, "confirmed_latency": 3, "latency_unit": "ms"},
        {"suite": "GDN fwd", "method": "z3-cm", "search_space": "legal", "candidate": "c3", "search_wall_s": 4, "profile_count": 4, "confirmed_latency": 2.6, "latency_unit": "ms"},
    ]


def test_quality_search_pareto_marks_frontier_and_same_space_oracle(tmp_path):
    oracle = [
        {"suite": "GDN fwd", "search_space": "legal", "oracle_latency": 1.8, "latency_unit": "ms"},
        {"suite": "GDN fwd", "search_space": "raw", "oracle_latency": 3.5, "latency_unit": "ms"},
    ]
    output = tmp_path / "pareto"
    paths = vis.plot_quality_search_pareto(_quality_records(), output, x_axis="profile_count", oracle=oracle)
    payload, _ = _assert_artifacts(paths, output)

    candidates = {row["candidate"]: row for row in payload["rows"] if row["row_kind"] == "candidate"}
    assert candidates["c0"]["is_pareto"] is True
    assert candidates["c1"]["is_pareto"] is True
    assert candidates["c2"]["is_pareto"] is False
    assert candidates["c3"]["is_pareto"] is False
    assert len([row for row in payload["rows"] if row["row_kind"] == "oracle"]) == 2
    assert {row["method"] for row in payload["rows"]} >= {"stock-raw", "z3-legal", "z3-cm", "control"}


def test_quality_search_rejects_cross_space_oracle(tmp_path):
    oracle = [{"suite": "GDN fwd", "search_space": "different", "oracle_latency": 1, "latency_unit": "ms"}]
    with pytest.raises(ValueError, match="no same-suite/same-search-space"):
        vis.plot_quality_search_pareto(_quality_records(), tmp_path / "bad-pareto", oracle=oracle)


def _gdn_records():
    return [
        {"suite": "GDN fwd", "method": "stock-raw", "best_latency": 4, "latency_unit": "ms", "compile_count": 0, "profile_count": 1},
        {"suite": "GDN fwd", "method": "z3-cm", "best_latency": 2.5, "latency_unit": "ms", "compile_count": 4, "profile_count": 3},
        {"suite": "GDN bwd", "method": "stock-raw", "best_latency": 6, "latency_unit": "ms", "compile_count": 0, "profile_count": 1},
        {"suite": "GDN bwd", "method": "z3-cm", "best_latency": 3.5, "latency_unit": "ms", "compile_count": 5, "profile_count": 2},
    ]


def test_gdn_acceptance_has_two_panels_and_exact_input_values(tmp_path):
    output = tmp_path / "gdn.json"
    payload, rows = _assert_artifacts(vis.plot_gdn_acceptance(_gdn_records(), output), output)

    assert {row["direction"] for row in payload["rows"]} == {"fwd", "bwd"}
    assert {(row["panel"], row["method"], row["best_latency"], row["compile_count"], row["profile_count"]) for row in payload["rows"]} == {
        ("GDN fwd", "stock-raw", 4, 0, 1),
        ("GDN fwd", "z3-cm", 2.5, 4, 3),
        ("GDN bwd", "stock-raw", 6, 0, 1),
        ("GDN bwd", "z3-cm", 3.5, 5, 2),
    }
    assert len(rows) == 4
    assert payload["latency_scope"] == "confirmed"


def test_gdn_search_observed_scope_is_explicit(tmp_path):
    records = [dict(record, latency_scope="search_observed") for record in _gdn_records()]
    payload, _ = _assert_artifacts(vis.plot_gdn_acceptance(records, tmp_path / "search"), tmp_path / "search")
    assert payload["latency_scope"] == "search_observed"
    assert {row["latency_scope"] for row in payload["rows"]} == {"search_observed"}
    mixed = [dict(records[0], latency_scope="confirmed"), *records[1:]]
    with pytest.raises(ValueError, match="common latency scope"):
        vis.plot_gdn_acceptance(mixed, tmp_path / "mixed")
    with pytest.raises(ValueError, match="requires non-empty fwd and bwd"):
        vis.plot_gdn_acceptance(_gdn_records()[:2], tmp_path / "bad-gdn")


def test_gdn_confirmation_differences_preserve_rounds_ranges_and_pair_scopes(tmp_path):
    comparisons = [
        {
            "direction": "bwd",
            "alternative_arms": ["budget-raw"],
            "identical": False,
            "config": {"prepare_h_num_stages": 1, "group_reduce_block_size": 8},
            "paired_round_differences_ms": [-0.2, -0.1, -0.15],
            "quality_ratio_guided_over_alternative": 0.96,
            "latency_scope": "confirmed",
        },
        {
            "direction": "bwd",
            "alternative_arms": ["budget-random"],
            "identical": False,
            "config": {"prepare_h_num_stages": 2, "group_reduce_block_size": 32},
            "paired_round_differences_ms": [-0.001, 0.002],
            "quality_ratio_guided_over_alternative": 0.999,
            "latency_scope": "confirmed",
        },
        {"identical": True},
    ]
    output = tmp_path / "paired.json"
    payload, rows = _assert_artifacts(vis.plot_gdn_confirmation_differences(comparisons, output), output)
    assert payload["plot"] == "gdn_confirmation_paired_differences"
    assert payload["pair_count"] == 2
    assert len(rows) == 5
    assert float(rows[0]["median_ms"]) == pytest.approx(-0.15)
    assert float(rows[0]["min_ms"]) == pytest.approx(-0.2)
    assert float(rows[0]["max_ms"]) == pytest.approx(-0.1)
    assert rows[0]["config_id"] == '{"group_reduce_block_size":8,"prepare_h_num_stages":1}'
    assert float(rows[-1]["median_ms"]) == pytest.approx(0.0005)
    assert {row["latency_scope"] for row in rows} == {"confirmed"}


def test_empty_inputs_fail_instead_of_creating_placeholder_figures(tmp_path):
    empty_schedule = _loop_schedule()
    for key in ("start_times", "latencies", "iis", "resource_flags"):
        empty_schedule[key] = []
    empty_schedule["num_tasks"] = 0
    empty_schedule["makespan"] = 0
    with pytest.raises(ValueError, match="at least one task"):
        vis.plot_schedule_timeline(empty_schedule, tmp_path / "empty-schedule")

    empty_funnel = {key: 0 for key in ("raw", "sat", "unknown", "unsat", "dedup", "compiled", "profiled")}
    with pytest.raises(ValueError, match="raw must be > 0"):
        vis.plot_pruning_funnel(empty_funnel, tmp_path / "empty-funnel")
    with pytest.raises(ValueError, match="must not be empty"):
        vis.plot_search_time_breakdown([], tmp_path / "empty-time")
    with pytest.raises(ValueError, match="must not be empty"):
        vis.plot_prediction_rank([], [], tmp_path / "empty-rank")
    with pytest.raises(ValueError, match="must not be empty"):
        vis.plot_quality_search_pareto([], tmp_path / "empty-pareto")
    with pytest.raises(ValueError, match="must not be empty"):
        vis.plot_gdn_acceptance([], tmp_path / "empty-gdn")
    with pytest.raises(ValueError, match="non-identical"):
        vis.plot_gdn_confirmation_differences([{"identical": True}], tmp_path / "empty-paired")
