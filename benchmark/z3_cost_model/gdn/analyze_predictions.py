"""Analyze GDN prediction ranks against same-repetition cold-search oracles.

Scores come only from ``z3-cm-topk``.  Measured latencies come from the same
repetition's complete ``z3-legal-exhaustive`` run, or explicitly from its
``stock-raw`` successful trials when that legal oracle is unavailable or
incomplete.  No winner-confirmation measurements are read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics


DIRECTIONS = ("fwd", "bwd")
REPETITIONS = (0, 1, 2)
SCORE_ARM = "z3-cm-topk"
LEGAL_ARM = "z3-legal-exhaustive"
STOCK_ARM = "stock-raw"
ACTUAL_BUDGET = {"fwd": 2, "bwd": 3}
TOP_K = (1, 2, 3)


def _canonical(config):
    if not isinstance(config, dict):
        raise ValueError(f"Config must be an object, got {config!r}")
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _identity(event, path):
    canonical = _canonical(event.get("config"))
    config_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    recorded = event.get("config_id")
    if recorded is not None and recorded != config_id:
        raise ValueError(f"Non-canonical config_id in {path}: {recorded!r} != {config_id!r}")
    return config_id, canonical, event["config"]


def _event_lines(path):
    with path.open(encoding="utf-8") as stream:
        yield from stream


def _load_run(path, cache_track):
    events = [json.loads(line) for line in _event_lines(path) if line.strip()]
    starts = [event for event in events if event.get("event") == "run_start"]
    if len(starts) != 1:
        raise ValueError(f"Expected exactly one run_start in {path}, found {len(starts)}")
    start = starts[0]
    metadata = start.get("metadata", {})
    track = metadata.get("cache_track")
    if track is None:
        mode = metadata.get("cache_mode")
        track = "cold" if mode == "enabled" else mode
    if track != cache_track:
        return None
    direction = metadata.get("direction")
    arm = metadata.get("arm", start.get("arm"))
    repetition = metadata.get("repetition")
    if direction not in DIRECTIONS or not arm:
        raise ValueError(f"Missing direction/arm identity in {path}")
    if isinstance(repetition, bool) or not isinstance(repetition, int):
        raise ValueError(f"Missing integer repetition in {path}: {repetition!r}")
    summary_path = path.with_name("summary.json")
    summary = None
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as stream:
            summary = json.load(stream)
    return {
        "path": path,
        "events": events,
        "start": start,
        "metadata": metadata,
        "summary": summary,
        "direction": direction,
        "arm": arm,
        "repetition": repetition,
        "cache_track": track,
    }


def _load_runs(root, cache_track):
    runs = {}
    for path in sorted(Path(root).rglob("events.jsonl")):
        run = _load_run(path, cache_track)
        if run is None:
            continue
        key = (run["direction"], run["arm"], run["repetition"])
        if key in runs:
            raise ValueError(f"Duplicate {cache_track} run identity {key}: {path} and {runs[key]['path']}")
        runs[key] = run
    return runs


def _trial_data(run):
    if run is None:
        return {"trials": [], "successful": {}, "configs": {}, "duplicates": []}
    trials = []
    successful = {}
    configs = {}
    duplicates = []
    for event in run["events"]:
        if event.get("event") != "trial":
            continue
        config_id, canonical, config = _identity(event, run["path"])
        if config.get("direction") != run["direction"]:
            raise ValueError(f"Trial direction does not match run in {run['path']}")
        record = {
            "config_id": config_id,
            "canonical_config": canonical,
            "config": config,
            "status": event.get("status"),
            "latency_ms": event.get("latency_ms"),
            "sequence": event.get("sequence"),
        }
        trials.append(record)
        configs[config_id] = record
        if event.get("status") != "ok":
            continue
        latency = event.get("latency_ms")
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
            raise ValueError(f"Invalid successful-trial latency in {run['path']}: {latency!r}")
        record["latency_ms"] = float(latency)
        if config_id in successful:
            duplicates.append(config_id)
        successful.setdefault(config_id, []).append(record)
    return {"trials": trials, "successful": successful, "configs": configs, "duplicates": duplicates}


def _legal_assessment(run):
    if run is None:
        return {
            "complete": False,
            "status": "unavailable",
            "reasons": ["same-repetition z3-legal-exhaustive run is unavailable"],
            "legal_ids": set(),
            "successful": {},
            "candidate_count": 0,
            "successful_count": 0,
            "trial_count": 0,
            "missing_ids": [],
            "unsuccessful_ids": [],
            "path": None,
        }
    data = _trial_data(run)
    retained = {}
    for event in run["events"]:
        if event.get("event") == "prune" and str(event.get("action", "")).startswith("retained"):
            config_id, canonical, config = _identity(event, run["path"])
            retained[config_id] = {"canonical_config": canonical, "config": config}
    legal_ids = set(retained)
    successful = data["successful"]
    trial_ids = {record["config_id"] for record in data["trials"]}
    unsuccessful_ids = sorted({record["config_id"] for record in data["trials"] if record["status"] != "ok"})
    missing_ids = sorted(legal_ids - set(successful))
    reasons = []
    if not legal_ids:
        reasons.append("legal candidate set is unavailable from retained prune events")
    expected_counts = []
    summary = run["summary"] or {}
    for label, value in (("summary.U", summary.get("U")), ("summary.selected", summary.get("selected"))):
        if value is not None:
            expected_counts.append((label, value))
    for event in run["events"]:
        if event.get("event") == "prune_end":
            kept = event.get("audit", {}).get("kept")
            if kept is not None:
                expected_counts.append(("prune_end.audit.kept", kept))
    for label, expected in expected_counts:
        if isinstance(expected, bool) or not isinstance(expected, int) or expected != len(legal_ids):
            reasons.append(f"{label}={expected!r} does not match retained legal candidate count {len(legal_ids)}")
    unexpected_trials = sorted(trial_ids - legal_ids)
    if unexpected_trials:
        reasons.append(f"trials outside retained legal set: {unexpected_trials}")
    missing_trials = sorted(legal_ids - trial_ids)
    if missing_trials:
        reasons.append(f"legal candidates without trial records: {missing_trials}")
    if missing_ids:
        reasons.append(f"legal candidates without successful measurements: {missing_ids}")
    if unsuccessful_ids:
        reasons.append(f"legal candidates with unsuccessful trials: {unsuccessful_ids}")
    if data["duplicates"]:
        reasons.append(f"legal candidates with duplicate successful trials: {sorted(set(data['duplicates']))}")
    run_ends = [event for event in run["events"] if event.get("event") == "run_end"]
    if len(run_ends) != 1 or run_ends[0].get("status") != "ok":
        reasons.append("legal run does not have exactly one successful run_end")
    return {
        "complete": not reasons,
        "status": "complete" if not reasons else "incomplete",
        "reasons": reasons,
        "legal_ids": legal_ids,
        "successful": successful,
        "candidate_count": len(legal_ids),
        "successful_count": len(legal_ids & set(successful)),
        "trial_count": len(data["trials"]),
        "missing_ids": missing_ids,
        "unsuccessful_ids": unsuccessful_ids,
        "path": str(run["path"]),
    }


def _one_success_per_config(data, run, role):
    if data["duplicates"]:
        raise ValueError(f"Duplicate successful {role} trials in {run['path']}: {sorted(set(data['duplicates']))}")
    return {config_id: records[0] for config_id, records in data["successful"].items()}


def _select_oracle(legal_run, stock_run):
    assessment = _legal_assessment(legal_run)
    if assessment["complete"]:
        successful = _one_success_per_config(_trial_data(legal_run), legal_run, "legal-oracle")
        return {
            "kind": LEGAL_ARM,
            "arm": LEGAL_ARM,
            "successful": successful,
            "candidate_count": assessment["candidate_count"],
            "successful_count": assessment["successful_count"],
            "trial_count": assessment["trial_count"],
            "path": assessment["path"],
            "legal": assessment,
            "fallback_reason": None,
        }
    if stock_run is None:
        raise ValueError("Incomplete legal oracle cannot fall back: same-repetition stock-raw run is unavailable")
    stock_data = _trial_data(stock_run)
    successful = _one_success_per_config(stock_data, stock_run, "stock-raw fallback")
    if not successful:
        raise ValueError(f"Incomplete legal oracle and no successful stock-raw trials in {stock_run['path']}")
    reason = "; ".join(assessment["reasons"])
    return {
        "kind": "stock-raw-successful-fallback",
        "arm": STOCK_ARM,
        "successful": successful,
        "candidate_count": len({record["config_id"] for record in stock_data["trials"]}),
        "successful_count": len(successful),
        "trial_count": len(stock_data["trials"]),
        "path": str(stock_run["path"]),
        "legal": assessment,
        "fallback_reason": reason,
    }


def _scores(run):
    scores = {}
    for event in run["events"]:
        if event.get("event") != "score":
            continue
        config_id, canonical, config = _identity(event, run["path"])
        if config.get("direction") != run["direction"]:
            raise ValueError(f"Score direction does not match run in {run['path']}")
        if config_id in scores:
            raise ValueError(f"Duplicate score for {config_id} in {run['path']}")
        value = event.get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid prediction score in {run['path']}: {value!r}")
        scores[config_id] = {
            "config_id": config_id,
            "canonical_config": canonical,
            "config": config,
            "predicted": float(value),
            "finite": math.isfinite(value),
            "score_sequence": event.get("sequence"),
        }
    if not scores:
        raise ValueError(f"No score records in {run['path']}")
    return scores


def _observed_budget(run):
    values = []
    if run["start"].get("top_k") is not None:
        values.append(run["start"]["top_k"])
    if run["summary"] and run["summary"].get("selected") is not None:
        values.append(run["summary"]["selected"])
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"Invalid z3-cm-topk budget in {run['path']}: {values!r}")
    if len(set(values)) > 1:
        raise ValueError(f"Conflicting z3-cm-topk budgets in {run['path']}: {values}")
    return values[0] if values else None


def collect(root, cache_track="cold", repetitions=REPETITIONS, directions=DIRECTIONS):
    """Collect finite score/latency pairs without pooling arms or repetitions."""

    if cache_track not in ("cold", "warm", "no-cache"):
        raise ValueError(f"Unknown cache track: {cache_track}")
    repetitions = tuple(repetitions)
    directions = tuple(directions)
    if not repetitions or not directions:
        raise ValueError("At least one repetition and direction are required")
    if any(direction not in DIRECTIONS for direction in directions):
        raise ValueError(f"Unknown directions: {directions!r}")
    runs = _load_runs(root, cache_track)
    groups = []
    for direction in directions:
        for repetition in repetitions:
            score_run = runs.get((direction, SCORE_ARM, repetition))
            if score_run is None:
                raise ValueError(f"Missing {cache_track} {SCORE_ARM} run for {direction}/rep-{repetition}")
            observed_budget = _observed_budget(score_run)
            if observed_budget is not None and observed_budget != ACTUAL_BUDGET[direction]:
                raise ValueError(
                    f"Actual budget changed for {direction}/rep-{repetition}: "
                    f"expected {ACTUAL_BUDGET[direction]}, got {observed_budget}"
                )
            legal_run = runs.get((direction, LEGAL_ARM, repetition))
            stock_run = runs.get((direction, STOCK_ARM, repetition))
            oracle = _select_oracle(legal_run, stock_run)
            scores = _scores(score_run)
            missing_measured = sorted(set(scores) - set(oracle["successful"]))
            nonfinite = sorted(config_id for config_id, score in scores.items() if not score["finite"])
            rows = []
            for config_id, score in scores.items():
                trial = oracle["successful"].get(config_id)
                if trial is None or not score["finite"]:
                    continue
                rows.append({
                    "suite": direction,
                    "direction": direction,
                    "repetition": repetition,
                    "cache_track": cache_track,
                    "candidate": (
                        f"{direction}:block_DV={score['config'].get('block_DV')},"
                        f"stage={score['config'].get('prepare_h_num_stages')},"
                        f"group={score['config'].get('group_reduce_block_size')}"
                    ),
                    "config_id": config_id,
                    "canonical_config": score["canonical_config"],
                    "config": score["config"],
                    "predicted": score["predicted"],
                    "measured": trial["latency_ms"],
                    "sample_count": 1,
                    "score_sequence": score["score_sequence"],
                    "oracle_trial_sequence": trial["sequence"],
                })
            if not rows:
                raise ValueError(f"No finite same-repetition score/oracle pairs for {direction}/rep-{repetition}")
            legal = oracle["legal"]
            groups.append({
                "suite": direction,
                "direction": direction,
                "repetition": repetition,
                "cache_track": cache_track,
                "num_tokens": score_run["metadata"].get("shape", {}).get("T"),
                "score_arm": SCORE_ARM,
                "score_events_path": str(score_run["path"]),
                "score_count": len(scores),
                "finite_prediction_count": len(scores) - len(nonfinite),
                "nonfinite_score_config_ids": nonfinite,
                "missing_measured_config_ids": missing_measured,
                "paired_denominator": len(rows),
                "per_candidate_sample_count": 1,
                "actual_budget": ACTUAL_BUDGET[direction],
                "oracle_kind": oracle["kind"],
                "oracle_arm": oracle["arm"],
                "oracle_events_path": oracle["path"],
                "oracle_candidate_count": oracle["candidate_count"],
                "oracle_successful_count": oracle["successful_count"],
                "oracle_trial_count": oracle["trial_count"],
                "legal_oracle_complete": legal["complete"],
                "legal_oracle_status": legal["status"],
                "legal_oracle_events_path": legal["path"],
                "legal_candidate_count": legal["candidate_count"],
                "legal_successful_count": legal["successful_count"],
                "legal_trial_count": legal["trial_count"],
                "legal_missing_config_ids": legal["missing_ids"],
                "legal_unsuccessful_config_ids": legal["unsuccessful_ids"],
                "legal_incomplete_reasons": legal["reasons"],
                "fallback_reason": oracle["fallback_reason"],
                "stock_raw_events_path": str(stock_run["path"]) if stock_run is not None else None,
                "rows": rows,
            })
    return groups


def _load_visualization(repo_root):
    path = repo_root / "benchmark/z3_cost_model/visualization.py"
    spec = importlib.util.spec_from_file_location("gdn_prediction_visualization", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tie_records(rows, value_field, average_rank_field, canonical_rank_field):
    grouped = {}
    for row in rows:
        grouped.setdefault(row[value_field], []).append(row)
    records = []
    for value, members in grouped.items():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda row: row["canonical_config"])
        records.append({
            "score" if value_field == "predicted" else "latency_ms": value,
            "size": len(members),
            "average_rank": members[0][average_rank_field],
            "config_ids": [row["config_id"] for row in members],
            "canonical_configs": [row["canonical_config"] for row in members],
            "canonical_ranks": [row[canonical_rank_field] for row in members],
        })
    records.sort(key=lambda record: (record["average_rank"], record["canonical_configs"]))
    return records


def _rank_group(group, viz):
    rows = [dict(row) for row in group["rows"]]
    rows.sort(key=lambda row: (row["predicted"], row["canonical_config"]))
    predicted_order = [row["config_id"] for row in rows]
    measured_rows = sorted(rows, key=lambda row: (row["measured"], row["canonical_config"]))
    measured_order = [row["config_id"] for row in measured_rows]
    canonical_prediction_rank = {config_id: index for index, config_id in enumerate(predicted_order, 1)}
    canonical_measured_rank = {config_id: index for index, config_id in enumerate(measured_order, 1)}
    predicted = [row["predicted"] for row in rows]
    measured = [row["measured"] for row in rows]
    predicted_average = viz.rankdata(predicted)
    measured_average = viz.rankdata(measured)
    for row, predicted_rank, measured_rank in zip(rows, predicted_average, measured_average):
        config_id = row["config_id"]
        row.update({
            "predicted_rank": predicted_rank,
            "actual_rank": measured_rank,
            "predicted_average_rank": predicted_rank,
            "measured_average_rank": measured_rank,
            "canonical_prediction_rank": canonical_prediction_rank[config_id],
            "canonical_measured_rank": canonical_measured_rank[config_id],
        })
    prediction_ties = _tie_records(rows, "predicted", "predicted_average_rank", "canonical_prediction_rank")
    measured_ties = _tie_records(rows, "measured", "measured_average_rank", "canonical_measured_rank")
    prediction_tie_members = {
        config_id: record
        for record in prediction_ties
        for config_id in record["config_ids"]
    }
    measured_tie_members = {
        config_id: record
        for record in measured_ties
        for config_id in record["config_ids"]
    }
    count = len(rows)
    top_k_recalls = {}
    top_k_selections = {}
    selection_specs = [(str(k), k) for k in TOP_K] + [("actual_budget", group["actual_budget"])]
    for label, k in selection_specs:
        if label == "actual_budget" and str(k) in top_k_selections:
            selection = dict(top_k_selections[str(k)])
            selection["label"] = "actual_budget"
        elif k > count:
            selection = {
                "label": label,
                "k": k,
                "available": False,
                "predicted_config_ids": [],
                "measured_config_ids": [],
                "overlap_config_ids": [],
                "recall": None,
            }
        else:
            predicted_top = set(predicted_order[:k])
            measured_top = set(measured_order[:k])
            overlap = predicted_top & measured_top
            selection = {
                "label": label,
                "k": k,
                "available": True,
                "predicted_config_ids": predicted_order[:k],
                "measured_config_ids": measured_order[:k],
                "overlap_config_ids": [config_id for config_id in predicted_order[:k] if config_id in overlap],
                "recall": len(overlap) / k,
            }
        top_k_selections[label] = selection
        if k in TOP_K and str(k) not in top_k_recalls:
            top_k_recalls[str(k)] = selection["recall"]
    actual_selection = top_k_selections["actual_budget"]
    actual_predicted_top = set(actual_selection["predicted_config_ids"])
    oracle_best = measured_rows[0]
    oracle_best_id = oracle_best["config_id"]
    oracle_best_in_actual = oracle_best_id in actual_predicted_top if actual_selection["available"] else False
    metrics = {
        "n": count,
        "paired_denominator": count,
        "spearman": viz.spearman_rank_correlation(predicted, measured),
        "kendall_tau_b": viz.kendall_tau_b(predicted, measured),
        "top_k_recall": top_k_recalls,
        "actual_budget": group["actual_budget"],
        "top_actual_budget_recall": actual_selection["recall"],
        "oracle_best_config_id": oracle_best_id,
        "oracle_best_canonical_config": oracle_best["canonical_config"],
        "oracle_best_measured_latency_ms": oracle_best["measured"],
        "oracle_best_prediction_rank": canonical_prediction_rank[oracle_best_id],
        "oracle_best_prediction_average_rank": oracle_best["predicted_average_rank"],
        "oracle_best_in_actual_deterministic_top_k": oracle_best_in_actual,
    }
    for row in rows:
        config_id = row["config_id"]
        prediction_tie = prediction_tie_members.get(config_id)
        measured_tie = measured_tie_members.get(config_id)
        row.update({
            "paired_denominator": count,
            "score_count": group["score_count"],
            "oracle_kind": group["oracle_kind"],
            "oracle_arm": group["oracle_arm"],
            "oracle_candidate_count": group["oracle_candidate_count"],
            "oracle_successful_count": group["oracle_successful_count"],
            "legal_oracle_complete": group["legal_oracle_complete"],
            "legal_oracle_status": group["legal_oracle_status"],
            "fallback_reason": group["fallback_reason"],
            "prediction_exact_tie": prediction_tie is not None,
            "prediction_tie_size": prediction_tie["size"] if prediction_tie else 1,
            "prediction_tie_config_ids": json.dumps(prediction_tie["config_ids"], separators=(",", ":")) if prediction_tie else "",
            "measured_exact_tie": measured_tie is not None,
            "measured_tie_size": measured_tie["size"] if measured_tie else 1,
            "measured_tie_config_ids": json.dumps(measured_tie["config_ids"], separators=(",", ":")) if measured_tie else "",
            "is_oracle_best": config_id == oracle_best_id,
            "in_actual_deterministic_top_k": config_id in actual_predicted_top,
            "oracle_best_in_actual_deterministic_top_k": oracle_best_in_actual,
            "spearman": metrics["spearman"],
            "kendall_tau_b": metrics["kendall_tau_b"],
            "top_1_recall": top_k_recalls["1"],
            "top_2_recall": top_k_recalls["2"],
            "top_3_recall": top_k_recalls["3"],
            "actual_budget": group["actual_budget"],
            "top_actual_budget_recall": actual_selection["recall"],
            "score_events_path": group["score_events_path"],
            "oracle_events_path": group["oracle_events_path"],
            "num_tokens": group["num_tokens"],
        })
    detail = {key: value for key, value in group.items() if key != "rows"}
    detail.update(metrics)
    detail.update({
        "prediction_ties": prediction_ties,
        "measured_ties": measured_ties,
        "top_k_selections": top_k_selections,
        "canonical_prediction_order": predicted_order,
        "canonical_measured_order": measured_order,
    })
    return detail, rows


def _median_metric(values):
    if not values or any(value is None for value in values):
        return None
    return statistics.median(values)


def rank_analysis(groups, viz=None):
    """Compute per-repetition metrics and median aggregates without pooling rows."""

    if viz is None:
        viz = _load_visualization(Path(__file__).resolve().parents[3])
    details = []
    rows = []
    seen = set()
    for group in groups:
        key = (group["direction"], group["repetition"])
        if key in seen:
            raise ValueError(f"Duplicate collected group: {key}")
        seen.add(key)
        detail, group_rows = _rank_group(group, viz)
        details.append(detail)
        rows.extend(group_rows)
    suites = {}
    for direction in dict.fromkeys(group["direction"] for group in groups):
        selected = [detail for detail in details if detail["direction"] == direction]
        selected.sort(key=lambda detail: detail["repetition"])
        actual_budget = ACTUAL_BUDGET[direction]
        suites[direction] = {
            "suite": direction,
            "repetitions": [detail["repetition"] for detail in selected],
            "aggregate_repetition_count": len(selected),
            "aggregate_rule": "median of per-repetition metrics; no candidate latencies or denominators are pooled",
            "paired_denominators": [detail["paired_denominator"] for detail in selected],
            "per_candidate_sample_counts": [detail["per_candidate_sample_count"] for detail in selected],
            "oracle_kinds": [detail["oracle_kind"] for detail in selected],
            "legal_oracle_complete": [detail["legal_oracle_complete"] for detail in selected],
            "spearman": _median_metric([detail["spearman"] for detail in selected]),
            "kendall_tau_b": _median_metric([detail["kendall_tau_b"] for detail in selected]),
            "top_k_recall": {
                str(k): _median_metric([detail["top_k_recall"][str(k)] for detail in selected])
                for k in TOP_K
            },
            "actual_budget": actual_budget,
            "top_actual_budget_recall": _median_metric([detail["top_actual_budget_recall"] for detail in selected]),
            "oracle_best_prediction_rank": _median_metric([detail["oracle_best_prediction_rank"] for detail in selected]),
            "oracle_best_prediction_average_rank": _median_metric([detail["oracle_best_prediction_average_rank"] for detail in selected]),
            "oracle_best_in_actual_deterministic_top_k_by_repetition": [
                detail["oracle_best_in_actual_deterministic_top_k"] for detail in selected
            ],
            "oracle_best_in_actual_deterministic_top_k_count": sum(
                detail["oracle_best_in_actual_deterministic_top_k"] for detail in selected
            ),
            "per_repetition": selected,
        }
    return {
        "analysis_schema": "gdn-prediction-rank/v2",
        "measurement_scope": "cold-search trials only; no winner-confirmation measurements",
        "score_arm": SCORE_ARM,
        "oracle_policy": (
            "same direction and repetition; complete z3-legal-exhaustive successful trials, "
            "otherwise same-repetition stock-raw successful trials with legal_oracle_complete=false"
        ),
        "config_id_rule": "SHA-256 of canonical compact JSON config",
        "selection_tie_rule": "finite prediction score or measured latency, then canonical compact JSON config",
        "display_rank_rule": "predicted_rank/actual_rank are average tied ranks; canonical_*_rank fields are deterministic",
        "top_k": list(TOP_K),
        "actual_budgets": dict(ACTUAL_BUDGET),
        "suites": suites,
        "per_repetition": details,
        "rows": rows,
    }


def analyze(root, cache_track="cold", repetitions=REPETITIONS, directions=DIRECTIONS, viz=None):
    groups = collect(root, cache_track=cache_track, repetitions=repetitions, directions=directions)
    analysis = rank_analysis(groups, viz=viz)
    analysis["cache_track"] = cache_track
    analysis["repetitions"] = list(repetitions)
    return analysis


def _format_metric(value):
    return "n/a" if value is None else f"{value:.3f}"


def _plot_prediction_rank(analysis, output, viz):
    plt = viz._pyplot()
    directions = [direction for direction in DIRECTIONS if direction in analysis["suites"]]
    repetitions = sorted({detail["repetition"] for detail in analysis["per_repetition"]})
    detail_by_key = {(detail["direction"], detail["repetition"]): detail for detail in analysis["per_repetition"]}
    rows_by_key = {}
    for row in analysis["rows"]:
        rows_by_key.setdefault((row["direction"], row["repetition"]), []).append(row)
    with viz._rc_context(plt):
        fig, axes = plt.subplots(
            len(repetitions),
            len(directions),
            figsize=(8.5 * len(directions), 6.8 * len(repetitions)),
            squeeze=False,
        )
        try:
            for row_index, repetition in enumerate(repetitions):
                for column_index, direction in enumerate(directions):
                    ax = axes[row_index][column_index]
                    key = (direction, repetition)
                    if key not in detail_by_key:
                        ax.axis("off")
                        continue
                    detail = detail_by_key[key]
                    points = rows_by_key[key]
                    aggregate = analysis["suites"][direction]
                    color = "#2563EB" if detail["legal_oracle_complete"] else "#EA580C"
                    for point in points:
                        ax.scatter(
                            point["canonical_prediction_rank"],
                            point["canonical_measured_rank"],
                            s=70,
                            color=color,
                            alpha=0.85,
                            edgecolor="#DC2626" if point["prediction_exact_tie"] else "white",
                            linewidth=1.4 if point["prediction_exact_tie"] else 0.6,
                            zorder=3,
                        )
                        label = (
                            point["candidate"]
                            .replace("block_DV=", "DV")
                            .replace("stage=", "S")
                            .replace("group=", "G")
                            .replace("fwd:", "")
                            .replace("bwd:", "")
                        )
                        right_edge = point["canonical_prediction_rank"] > len(points) * 0.65
                        top_edge = point["canonical_measured_rank"] > len(points) * 0.65
                        ax.annotate(
                            label,
                            (point["canonical_prediction_rank"], point["canonical_measured_rank"]),
                            xytext=(-5 if right_edge else 5, -5 if top_edge else 5),
                            textcoords="offset points",
                            fontsize=7,
                            ha="right" if right_edge else "left",
                            va="top" if top_edge else "bottom",
                            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 0.35},
                            zorder=5,
                        )
                    limit = max(2, len(points) + 1)
                    ax.plot([0.5, limit - 0.5], [0.5, limit - 0.5], "--", color="#6B7280", linewidth=1.1)
                    ax.set_xlim(0.5, limit - 0.5)
                    ax.set_ylim(0.5, limit - 0.5)
                    ax.set_xticks(range(1, len(points) + 1))
                    ax.set_yticks(range(1, len(points) + 1))
                    ax.set_xlabel("Canonical prediction selection rank (1 = lowest score)")
                    ax.set_ylabel("Cold-search measured rank (1 = lowest latency)")
                    oracle_label = "legal complete" if detail["legal_oracle_complete"] else "stock-raw fallback; legal incomplete"
                    ax.set_title(
                        f"{direction} repetition {repetition} | paired n={detail['paired_denominator']} | {oracle_label}",
                        pad=72,
                    )
                    ax.grid(alpha=0.22)
                    ax.set_axisbelow(True)
                    metric_lines = [
                        f"rep Spearman ρ={_format_metric(detail['spearman'])}; "
                        f"Kendall τb={_format_metric(detail['kendall_tau_b'])}",
                        "rep "
                        + "; ".join(f"top-{k}={_format_metric(detail['top_k_recall'][str(k)])}" for k in TOP_K)
                        + f"; budget(k={detail['actual_budget']})={_format_metric(detail['top_actual_budget_recall'])}",
                        f"{aggregate['aggregate_repetition_count']}-rep median ρ={_format_metric(aggregate['spearman'])}; "
                        f"τb={_format_metric(aggregate['kendall_tau_b'])}; "
                        f"oracle-best canonical rank={_format_metric(aggregate['oracle_best_prediction_rank'])}",
                        f"exact prediction-score tie groups={len(detail['prediction_ties'])}; "
                        f"oracle-best in deterministic budget top-k={detail['oracle_best_in_actual_deterministic_top_k']}",
                    ]
                    ax.text(0.0, 1.015, "\n".join(metric_lines), transform=ax.transAxes, va="bottom", fontsize=8, linespacing=1.3)
            fig.suptitle("GDN prediction rank versus cold-search measured rank (same-repetition oracles)", y=0.997, fontsize=15)
            fig.text(
                0.01,
                0.005,
                "Cold-search trials only; winner-confirmation data are not used. "
                "Selection ranks and top-k use score/latency then canonical compact JSON; red edges mark exact score ties. "
                "Each plotted pair has one sample per config per repetition; medians aggregate repetition-level metrics only.",
                fontsize=9,
                color="#374151",
            )
            fig.tight_layout(rect=(0, 0.025, 1, 0.985))
            fields = [
                "suite", "direction", "repetition", "cache_track", "num_tokens", "candidate", "config_id",
                "canonical_config", "predicted", "measured", "sample_count", "paired_denominator", "score_count",
                "oracle_kind", "oracle_arm", "oracle_candidate_count", "oracle_successful_count",
                "legal_oracle_complete", "legal_oracle_status", "fallback_reason", "predicted_rank", "actual_rank",
                "predicted_average_rank", "measured_average_rank", "canonical_prediction_rank", "canonical_measured_rank",
                "prediction_exact_tie", "prediction_tie_size", "prediction_tie_config_ids", "measured_exact_tie",
                "measured_tie_size", "measured_tie_config_ids", "is_oracle_best", "in_actual_deterministic_top_k",
                "oracle_best_in_actual_deterministic_top_k", "spearman", "kendall_tau_b", "top_1_recall",
                "top_2_recall", "top_3_recall", "actual_budget", "top_actual_budget_recall", "score_events_path",
                "oracle_events_path", "score_sequence", "oracle_trial_sequence",
            ]
            payload = {key: value for key, value in analysis.items() if key != "rows"}
            payload.update({
                "plot": "prediction_rank",
                "lower_is_better": True,
                "figure_rank": "canonical deterministic selection ranks; average tied ranks remain in JSON/CSV",
                "tie_rule": "average tied display ranks; deterministic selection uses score/latency then canonical compact JSON config",
            })
            return viz._emit(fig, plt, output, analysis["rows"], fields, payload)
        finally:
            plt.close(fig)


def write_analysis(root, output_dir, cache_track="cold", repetitions=REPETITIONS, directions=DIRECTIONS):
    viz = _load_visualization(Path(__file__).resolve().parents[3])
    analysis = analyze(root, cache_track=cache_track, repetitions=repetitions, directions=directions, viz=viz)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _plot_prediction_rank(analysis, output / "prediction-rank", viz)
    return analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-track", choices=("cold", "warm", "no-cache"), default="cold")
    args = parser.parse_args(argv)
    analysis = write_analysis(args.root, args.output, cache_track=args.cache_track)
    summary = {}
    for direction, aggregate in analysis["suites"].items():
        summary[direction] = {
            key: value for key, value in aggregate.items() if key != "per_repetition"
        }
        summary[direction]["per_repetition"] = [
            {
                "repetition": detail["repetition"],
                "paired_denominator": detail["paired_denominator"],
                "oracle_kind": detail["oracle_kind"],
                "legal_oracle_complete": detail["legal_oracle_complete"],
                "spearman": detail["spearman"],
                "kendall_tau_b": detail["kendall_tau_b"],
                "top_k_recall": detail["top_k_recall"],
                "actual_budget": detail["actual_budget"],
                "top_actual_budget_recall": detail["top_actual_budget_recall"],
                "oracle_best_prediction_rank": detail["oracle_best_prediction_rank"],
                "oracle_best_prediction_average_rank": detail["oracle_best_prediction_average_rank"],
                "oracle_best_in_actual_deterministic_top_k": detail["oracle_best_in_actual_deterministic_top_k"],
                "prediction_ties": detail["prediction_ties"],
            }
            for detail in aggregate["per_repetition"]
        ]
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
