"""Recompute classic operator search evidence from real summaries and JSONL events.

Performance comparisons require complete selected-config trial accounting.  A
collected summary without events, a cache-only result, or missing/duplicate
terminal trials is retained for audit but is not performance evidence.
Prediction/measurement correlations additionally require a complete successful
``z3-legal-exhaustive`` oracle and report their paired-config denominator.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics

if __package__:
    from .autotune import ARMS, DEFAULT_TOP_K, SUITE_NAMES, _strict_json, _write_json
else:
    import sys

    _REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
    if str(_REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPOSITORY_ROOT))
    from benchmark.z3_cost_model.operators.autotune import ARMS, DEFAULT_TOP_K, SUITE_NAMES, _strict_json, _write_json


_TERMINAL_TRIALS = {"ok", "compile_error", "benchmark_error", "timeout"}
_PHASES = ("compile", "validation", "profile")


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive(value) -> bool:
    return _finite(value) and float(value) > 0


def _median(values):
    selected = [float(value) for value in values if _finite(value)]
    return statistics.median(selected) if selected else None


def _ratio(numerator, denominator):
    if not _finite(numerator) or not _positive(denominator):
        return None
    return float(numerator) / float(denominator)


def _load_events(path: Path):
    if not path.is_file():
        return [], "events file is missing"
    events = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError(f"line {line_number} is not an object")
                events.append(event)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return events, f"{type(exc).__name__}: {exc}"
    return events, None


def _event(events, name):
    return [event for event in events if event.get("event") == name]


def _config_id(event):
    value = event.get("config_id")
    if isinstance(value, str) and value:
        return value
    config = event.get("config")
    if isinstance(config, dict):
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        import hashlib

        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return None


def _phase_evidence(events, phase, *, event_phase=None):
    event_name = phase if event_phase is None else event_phase
    starts = _event(events, f"{event_name}_start")
    ends = _event(events, f"{event_name}_end")
    success_status = {"ok", "passed"}
    success = sum(event.get("status") in success_status for event in ends)
    not_run = sum(event.get("status") == "not_run" for event in ends)
    failed = len(ends) - success - not_run
    attempts = len(starts)
    unfinished = max(0, attempts - len(ends))
    issues = []
    if len(ends) != attempts:
        issues.append(f"{phase}: {attempts} starts but {len(ends)} ends")
    wall = 0.0 if attempts == 0 else None
    if attempts and not issues:
        start_times = [event.get("perf_counter") for event in starts]
        end_times = [event.get("perf_counter") for event in ends]
        if all(_finite(value) for value in (*start_times, *end_times)):
            wall = max(0.0, max(end_times) - min(start_times))
        else:
            issues.append(f"{phase}: missing phase timestamps")
    durations = [event.get("wall_seconds") for event in ends]
    cumulative = sum(max(0.0, float(value)) for value in durations) if len(durations) == len(ends) and all(_finite(value) for value in durations) else None
    if cumulative is None and ends:
        issues.append(f"{phase}: missing per-attempt wall_seconds")
    return {
        "attempts": attempts,
        "success": success,
        "fail": failed,
        "not_run": not_run,
        "unfinished": unfinished,
        "wall_seconds": wall,
        "cumulative_seconds": cumulative,
        "issues": issues,
    }


def _interval_wall(events, start_name, end_name):
    starts = _event(events, start_name)
    ends = _event(events, end_name)
    if len(starts) != 1 or len(ends) != 1:
        return None
    start = starts[0].get("perf_counter")
    end = ends[0].get("perf_counter")
    return max(0.0, float(end) - float(start)) if _finite(start) and _finite(end) else None


def _best_from_events(events, run_end):
    successful = []
    for trial in _event(events, "trial"):
        if trial.get("status") == "ok" and _finite(trial.get("latency_ms")):
            successful.append(trial)
    if successful:
        best = min(successful, key=lambda event: (float(event["latency_ms"]), _config_id(event) or ""))
        return float(best["latency_ms"]), best.get("config"), "successful_trials", successful
    if run_end is not None and _finite(run_end.get("best_latency_ms")):
        # Cache hits and partial failed runs may carry a historical/partial best.
        # It remains visible but cannot pass the performance gate below.
        return float(run_end["best_latency_ms"]), run_end.get("best_config"), "run_end_only", successful
    return None, None, "unavailable", successful


def _run_from_summary(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    metadata = summary.get("operator_run", {})
    event_metadata = summary.get("metadata", {})
    experiment = summary.get("experiment", {})
    suite = metadata.get("suite") or event_metadata.get("suite")
    arm = experiment.get("arm") or summary.get("arm")
    if suite not in SUITE_NAMES or arm not in ARMS:
        return None
    events_path = path.parent / "events.jsonl"
    events, event_error = _load_events(events_path)
    run_end_list = _event(events, "run_end")
    run_end = run_end_list[-1] if run_end_list else None
    generated = _event(events, "config_generated")
    raw_ids = [_config_id(event) for event in generated if event.get("phase") == "raw"]
    legal_ids = [_config_id(event) for event in generated if event.get("phase") == "legal"]
    selected_ids = [_config_id(event) for event in generated if event.get("phase") == "selected"]
    generation_ends = _event(events, "config_generation_end")
    generation_end = generation_ends[-1] if generation_ends else None
    trials = _event(events, "trial")
    trial_ids = [_config_id(event) for event in trials]
    status_counts = Counter(str(event.get("status", "unknown")) for event in trials)

    issues = []
    if event_error is not None:
        issues.append(event_error)
    if generation_end is None:
        issues.append("config_generation_end is missing")
    else:
        for field, recomputed in (("raw", len(raw_ids)), ("legal", len(legal_ids)), ("selected", len(selected_ids))):
            observed = generation_end.get(field)
            if field == "legal" and arm == "stock-raw" and observed is None:
                continue
            if observed != recomputed:
                issues.append(f"config generation {field}: event={observed}, recomputed={recomputed}")
    if any(value is None for value in raw_ids + selected_ids + trial_ids):
        issues.append("a raw/selected/trial config has no canonical config_id")
    if arm != "stock-raw" and any(value is None for value in legal_ids):
        issues.append("a legal config has no canonical config_id")
    if len(run_end_list) != 1:
        issues.append(f"expected one run_end, got {len(run_end_list)}")

    selected_counter = Counter(selected_ids)
    trial_counter = Counter(trial_ids)
    trial_accounting_complete = (
        generation_end is not None
        and bool(selected_ids)
        and selected_counter == trial_counter
        and all(status in _TERMINAL_TRIALS for status in status_counts)
        and all(count == 1 for count in trial_counter.values())
    )
    if not trial_accounting_complete:
        issues.append("selected configs and terminal trials do not have a one-to-one mapping")

    phases = {phase: _phase_evidence(events, phase) for phase in _PHASES}
    for phase in _PHASES:
        issues.extend(phases[phase]["issues"])
    end_to_end = _interval_wall(events, "run_start", "run_end")
    if end_to_end is None:
        issues.append("end-to-end wall is unavailable")
    best_latency, best_config, best_source, successful_trials = _best_from_events(events, run_end)
    if run_end is not None and best_source == "successful_trials":
        if not _finite(run_end.get("best_latency_ms")) or float(run_end["best_latency_ms"]) != best_latency:
            issues.append("run_end best latency differs from the minimum successful trial")
        if run_end.get("best_config") != best_config:
            issues.append("run_end best config differs from the minimum successful trial")

    run_status = run_end.get("status") if run_end is not None else summary.get("status")
    performance_evidence = (
        event_error is None
        and run_status == "ok"
        and summary.get("status") == "ok"
        and trial_accounting_complete
        and best_source == "successful_trials"
        and _positive(best_latency)
        and _finite(end_to_end)
        and not issues
    )
    legal_count = len(legal_ids) if arm != "stock-raw" and generation_end is not None else None
    oracle_complete = bool(
        arm == "z3-legal-exhaustive"
        and performance_evidence
        and legal_count is not None
        and legal_count == len(selected_ids)
        and len(successful_trials) == legal_count
        and status_counts == Counter({"ok": legal_count})
        and phases["compile"]["success"] == legal_count
        and phases["validation"]["success"] == legal_count
        and phases["profile"]["success"] == legal_count
    )

    predictions = []
    for event in _event(events, "score"):
        predictions.append(
            {
                "config_id": _config_id(event),
                "config": event.get("config"),
                "score": event.get("score"),
                "rank": event.get("rank"),
                "unknown": bool(event.get("unknown", False)),
            }
        )
    prediction_ids = [record["config_id"] for record in predictions]
    predictions_complete = bool(
        arm == "z3-cm-topk"
        and generation_end is not None
        and generation_end.get("status") == "ok"
        and legal_count is not None
        and len(predictions) == legal_count
        and len(set(prediction_ids)) == len(prediction_ids)
        and None not in prediction_ids
        and Counter(prediction_ids) == Counter(legal_ids)
    )
    solver_records = _event(events, "solver")
    solver_status = Counter(str(event.get("status")) for event in solver_records)
    cache_identity_events = _event(events, "cache_identity")
    cache_identity = cache_identity_events[-1] if cache_identity_events else {}
    cache_hits = _event(events, "cache_hit")
    repetition = experiment.get("repetition", event_metadata.get("repetition"))
    top_k = metadata.get("top_k", event_metadata.get("top_k"))
    seed = metadata.get("seed", event_metadata.get("seed"))
    cache_track = metadata.get("cache_track", event_metadata.get("cache_track", experiment.get("cache_mode")))

    return {
        "path": str(path),
        "events_path": str(events_path),
        "events_available": event_error is None,
        "event_error": event_error,
        "suite": suite,
        "arm": arm,
        "repetition": repetition,
        "top_k": top_k,
        "seed": seed,
        "cache_track": cache_track,
        "status": summary.get("status"),
        "run_end_status": run_status,
        "R": len(raw_ids) if generation_end is not None else None,
        "U": legal_count,
        "selected": len(selected_ids) if generation_end is not None else None,
        "compile_attempts": phases["compile"]["attempts"],
        "compile_success": phases["compile"]["success"],
        "compile_fail": phases["compile"]["fail"],
        "validation_attempts": phases["validation"]["attempts"],
        "validation_success": phases["validation"]["success"],
        "validation_fail": phases["validation"]["fail"],
        "profile_attempts": phases["profile"]["attempts"],
        "profile_success": phases["profile"]["success"],
        "profile_fail": phases["profile"]["fail"],
        "solver_checks": len(solver_records) if generation_end is not None and arm != "stock-raw" else 0,
        "solver_sat": solver_status.get("sat", 0),
        "solver_unsat": solver_status.get("unsat", 0),
        "solver_unknown": solver_status.get("unknown", 0),
        "model_attempts": len(_event(events, "score_start")),
        "model_predictions": len(predictions) if predictions_complete else None,
        "model_unknown": sum(record["unknown"] for record in predictions),
        "solver_wall_seconds": _interval_wall(events, "solver_start", "solver_end"),
        "model_wall_seconds": _phase_evidence(events, "model", event_phase="score")["wall_seconds"] if arm == "z3-cm-topk" else None,
        "compile_wall_seconds": phases["compile"]["wall_seconds"],
        "validation_wall_seconds": phases["validation"]["wall_seconds"],
        "profile_wall_seconds": phases["profile"]["wall_seconds"],
        "compile_cumulative_seconds": phases["compile"]["cumulative_seconds"],
        "validation_cumulative_seconds": phases["validation"]["cumulative_seconds"],
        "profile_cumulative_seconds": phases["profile"]["cumulative_seconds"],
        "end_to_end_wall_seconds": end_to_end,
        "best_latency_ms": best_latency,
        "best_config": best_config,
        "best_source": best_source,
        "reference_latency_ms": summary.get("reference_latency_ms"),
        "cache_identity": cache_identity.get("cache_key"),
        "cache_directory": cache_identity.get("cache_dir"),
        "cache_hit": bool(cache_hits) or run_status == "cache_hit",
        "cache_source": cache_hits[-1].get("source") if cache_hits else None,
        "trials": dict(sorted(status_counts.items())),
        "successful_trials": len(successful_trials),
        "trial_accounting_complete": trial_accounting_complete,
        "performance_evidence": performance_evidence,
        "oracle_complete": oracle_complete,
        "predictions_complete": predictions_complete,
        "accounting_issues": issues,
        "has_recorded_candidate_failures": bool(summary.get("has_recorded_candidate_failures")),
        "_predictions": predictions,
        "_trials": [
            {
                "config_id": _config_id(event),
                "config": event.get("config"),
                "status": event.get("status"),
                "latency_ms": event.get("latency_ms"),
            }
            for event in trials
        ],
    }


def collect_runs(root):
    runs = []
    for path in sorted(Path(root).rglob("summary.json")):
        run = _run_from_summary(path)
        if run is not None:
            runs.append(run)
    if not runs:
        raise ValueError(f"No classic operator summary.json files found under {root}")
    return runs


def _comparison_key(run):
    return (run["suite"], run["repetition"], run["top_k"], run["cache_track"], run["seed"])


def _unique_index(runs, predicate):
    grouped = {}
    for run in runs:
        if predicate(run):
            grouped.setdefault(_comparison_key(run), []).append(run)
    return {key: values[0] for key, values in grouped.items() if len(values) == 1}


def _baseline_key(run):
    return (run["suite"], run["repetition"], DEFAULT_TOP_K, run["cache_track"], run["seed"])


def _same_space(run, baseline):
    if run["R"] != baseline["R"]:
        return False
    baseline_legal = baseline["R"] if baseline["arm"] == "stock-raw" else baseline["U"]
    return run["arm"] == "stock-raw" or run["U"] == baseline_legal


def _baseline(run, index):
    exact = index.get(_comparison_key(run))
    if exact is not None:
        return (exact, False) if _same_space(run, exact) else (None, False)
    if run["top_k"] == DEFAULT_TOP_K:
        return None, False
    shared = index.get(_baseline_key(run))
    if shared is not None and _same_space(run, shared):
        return shared, True
    return None, False


def build_comparisons(runs):
    stock = _unique_index(runs, lambda run: run["arm"] == "stock-raw" and run["performance_evidence"])
    oracle = _unique_index(runs, lambda run: run["arm"] == "z3-legal-exhaustive" and run["oracle_complete"])
    rows = []
    for run in runs:
        row = {key: value for key, value in run.items() if not key.startswith("_")}
        baseline, baseline_shared = _baseline(run, stock)
        reference, oracle_shared = _baseline(run, oracle)
        row["baseline_available"] = baseline is not None
        row["oracle_available"] = reference is not None
        row["baseline_top_k"] = baseline["top_k"] if baseline is not None else None
        row["oracle_top_k"] = reference["top_k"] if reference is not None else None
        row["baseline_shared_with_top10"] = baseline_shared
        row["oracle_shared_with_top10"] = oracle_shared
        row["baseline_path"] = baseline["path"] if baseline is not None else None
        row["oracle_path"] = reference["path"] if reference is not None else None
        row["search_speedup"] = None
        row["compile_reduction"] = None
        row["profile_reduction"] = None
        row["quality_ratio"] = None
        row["regret_ms"] = None
        row["relative_regret"] = None
        row["search_speedup_denominator_seconds"] = None
        row["compile_reduction_denominator"] = None
        row["profile_reduction_denominator"] = None
        row["quality_denominator_ms"] = None
        row["oracle_best_latency_ms"] = reference["best_latency_ms"] if reference is not None else None
        row["oracle_best_config"] = reference["best_config"] if reference is not None else None
        if run["performance_evidence"] and baseline is not None:
            row["search_speedup"] = _ratio(baseline["end_to_end_wall_seconds"], run["end_to_end_wall_seconds"])
            row["search_speedup_denominator_seconds"] = run["end_to_end_wall_seconds"]
            row["compile_reduction"] = (
                1.0 - float(run["compile_attempts"]) / float(baseline["compile_attempts"])
                if _positive(baseline["compile_attempts"])
                else None
            )
            row["profile_reduction"] = (
                1.0 - float(run["profile_attempts"]) / float(baseline["profile_attempts"])
                if _positive(baseline["profile_attempts"])
                else None
            )
            row["compile_reduction_denominator"] = baseline["compile_attempts"] if _positive(baseline["compile_attempts"]) else None
            row["profile_reduction_denominator"] = baseline["profile_attempts"] if _positive(baseline["profile_attempts"]) else None
        if run["performance_evidence"] and reference is not None:
            row["quality_ratio"] = _ratio(run["best_latency_ms"], reference["best_latency_ms"])
            row["regret_ms"] = float(run["best_latency_ms"]) - float(reference["best_latency_ms"])
            row["relative_regret"] = row["quality_ratio"] - 1.0 if row["quality_ratio"] is not None else None
            row["quality_denominator_ms"] = reference["best_latency_ms"]
        rows.append(row)
    return rows


def _consistent(rows, field):
    values = {json.dumps(row[field], sort_keys=True) for row in rows if row[field] is not None}
    if len(values) != 1:
        return None
    return json.loads(values.pop())


def aggregate(comparisons):
    groups = {}
    for row in comparisons:
        key = (row["suite"], row["arm"], row["top_k"], row["cache_track"])
        groups.setdefault(key, []).append(row)
    output = []
    for (suite, arm, top_k, cache_track), rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        evidence = [row for row in rows if row["performance_evidence"]]
        best_values = [row["best_latency_ms"] for row in evidence if _positive(row["best_latency_ms"])]
        best_row = min(evidence, key=lambda row: float(row["best_latency_ms"])) if evidence else None
        output.append(
            {
                "suite": suite,
                "arm": arm,
                "top_k": top_k,
                "cache_track": cache_track,
                "repetitions": len(rows),
                "performance_evidence_repetitions": len(evidence),
                "count_denominator": len(evidence),
                "R": _consistent(rows, "R"),
                "U": _consistent(rows, "U"),
                "selected": _consistent(rows, "selected"),
                "compile_attempts": _median(row["compile_attempts"] for row in evidence),
                "validation_attempts": _median(row["validation_attempts"] for row in evidence),
                "profile_attempts": _median(row["profile_attempts"] for row in evidence),
                "solver_checks": _median(row["solver_checks"] for row in evidence),
                "model_predictions": _median(row["model_predictions"] for row in evidence),
                "solver_wall_seconds": _median(row["solver_wall_seconds"] for row in evidence),
                "model_wall_seconds": _median(row["model_wall_seconds"] for row in evidence),
                "compile_wall_seconds": _median(row["compile_wall_seconds"] for row in evidence),
                "validation_wall_seconds": _median(row["validation_wall_seconds"] for row in evidence),
                "profile_wall_seconds": _median(row["profile_wall_seconds"] for row in evidence),
                "end_to_end_wall_seconds": _median(row["end_to_end_wall_seconds"] for row in evidence),
                "best_latency_ms": _median(best_values),
                "best_latency_denominator": len(best_values),
                "latency_range_ms": [min(best_values), max(best_values)] if best_values else None,
                "best_config": best_row["best_config"] if best_row is not None else None,
                "best_config_repetition": best_row["repetition"] if best_row is not None else None,
                "search_speedup": _median(row["search_speedup"] for row in evidence),
                "compile_reduction": _median(row["compile_reduction"] for row in evidence),
                "profile_reduction": _median(row["profile_reduction"] for row in evidence),
                "quality_ratio": _median(row["quality_ratio"] for row in evidence),
                "regret_ms": _median(row["regret_ms"] for row in evidence),
                "relative_regret": _median(row["relative_regret"] for row in evidence),
                "reference_latency_ms": _median(row["reference_latency_ms"] for row in evidence),
                "oracle_repetitions": sum(row["oracle_available"] for row in evidence),
                "has_recorded_candidate_failures": any(row["has_recorded_candidate_failures"] for row in rows),
            }
        )
    return output


def _ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for offset in range(position, end):
            ranks[order[offset]] = rank
        position = end
    return ranks


def _pearson(xs, ys):
    if len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    dx = [value - mean_x for value in xs]
    dy = [value - mean_y for value in ys]
    denominator = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    return sum(x * y for x, y in zip(dx, dy, strict=True)) / denominator if denominator > 0 else None


def build_correlations(runs):
    oracle = _unique_index(runs, lambda run: run["arm"] == "z3-legal-exhaustive" and run["oracle_complete"])
    output = []
    for run in runs:
        if run["arm"] != "z3-cm-topk":
            continue
        reference, oracle_shared = _baseline(run, oracle)
        record = {
            "suite": run["suite"],
            "arm": run["arm"],
            "repetition": run["repetition"],
            "top_k": run["top_k"],
            "cache_track": run["cache_track"],
            "seed": run["seed"],
            "status": "unavailable",
            "reason": None,
            "prediction_records": len(run["_predictions"]),
            "finite_prediction_records": 0,
            "oracle_measured_configs": 0,
            "paired_configs": 0,
            "denominator": 0,
            "denominator_definition": "unique config_id with a finite prediction and a successful measurement in one complete z3-legal-exhaustive oracle",
            "pearson": None,
            "spearman": None,
            "path": run["path"],
            "oracle_path": reference["path"] if reference is not None else None,
            "oracle_top_k": reference["top_k"] if reference is not None else None,
            "oracle_shared_with_top10": oracle_shared,
        }
        if not run["predictions_complete"]:
            record["reason"] = "guided predictions are incomplete or do not match the legal universe"
            output.append(record)
            continue
        if reference is None:
            record["reason"] = "a unique complete measured oracle is required for the same suite/repetition/top_k/cache/seed"
            output.append(record)
            continue
        measured = {}
        for trial in reference["_trials"]:
            if trial["status"] == "ok" and _finite(trial["latency_ms"]) and trial["config_id"] is not None:
                measured[trial["config_id"]] = float(trial["latency_ms"])
        record["oracle_measured_configs"] = len(measured)
        pairs = []
        finite_predictions = 0
        for prediction in run["_predictions"]:
            if prediction["unknown"] or not _finite(prediction["score"]):
                continue
            finite_predictions += 1
            config_id = prediction["config_id"]
            if config_id in measured:
                pairs.append((float(prediction["score"]), measured[config_id]))
        record["finite_prediction_records"] = finite_predictions
        record["paired_configs"] = len(pairs)
        record["denominator"] = len(pairs)
        if len(pairs) < 2:
            record["reason"] = "at least two paired finite predictions and oracle measurements are required"
            output.append(record)
            continue
        xs = [pair[0] for pair in pairs]
        ys = [pair[1] for pair in pairs]
        record["pearson"] = _pearson(xs, ys)
        record["spearman"] = _pearson(_ranks(xs), _ranks(ys))
        if record["pearson"] is None or record["spearman"] is None:
            record["status"] = "undefined"
            record["reason"] = "a correlation is undefined for constant predictions or measurements"
        else:
            record["status"] = "ok"
        output.append(record)
    return output


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _write_csv(path: Path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def write_analysis(root, output_dir):
    runs = collect_runs(root)
    comparisons = build_comparisons(runs)
    aggregates = aggregate(comparisons)
    correlations = build_correlations(runs)
    output = Path(output_dir).expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "tilelang.classic-operators.search-analysis/v1",
        "evidence_policy": {
            "performance": "run_end=ok, one terminal trial per selected config, complete phase intervals, and a best successful trial are required",
            "oracle": "every legal config in z3-legal-exhaustive must compile, validate, and profile successfully",
            "correlation": "finite predictions joined by config_id only to a unique complete measured oracle; denominator is paired configs",
            "excluded": "cache-only, collected-only, missing/duplicate trial, and failed-run records are not performance evidence",
        },
        "runs": comparisons,
        "aggregates": aggregates,
        "correlations": correlations,
    }
    _write_json(output / "analysis.json", _strict_json(payload))
    _write_csv(output / "runs.csv", comparisons)
    _write_csv(output / "summary.csv", aggregates)
    _write_csv(output / "correlations.csv", correlations)
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Directory containing classic operator run directories")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for analysis.json and CSV files")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    payload = write_analysis(args.root, args.output)
    print(
        json.dumps(
            {
                "runs": len(payload["runs"]),
                "performance_runs": sum(row["performance_evidence"] for row in payload["runs"]),
                "aggregates": len(payload["aggregates"]),
                "correlations": len(payload["correlations"]),
                "output": str(Path(args.output).expanduser().resolve(strict=False)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
