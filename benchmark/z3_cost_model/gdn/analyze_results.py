"""Aggregate real GDN search summaries and generate evidence figures."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics


ARMS = ("stock-raw", "z3-legal-exhaustive", "z3-cm-topk", "budget-raw", "budget-random")


def _median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def _ratio(numerator, denominator):
    return numerator / denominator if denominator not in (None, 0) and numerator is not None else None


def collect_runs(root):
    runs = []
    for path in sorted(Path(root).rglob("summary.json")):
        with path.open(encoding="utf-8") as stream:
            summary = json.load(stream)
        direction = summary.get("gdn", {}).get("direction") or summary.get("metadata", {}).get("direction")
        arm = summary.get("arm") or summary.get("metadata", {}).get("arm")
        if direction not in ("fwd", "bwd") or arm not in ARMS:
            continue
        experiment = summary.get("experiment", {})
        runs.append({
            "path": str(path),
            "direction": direction,
            "repetition": experiment.get("repetition", 0),
            "arm": arm,
            "R": summary.get("R"),
            "U": summary.get("U"),
            "selected": summary.get("selected"),
            "compile_attempts": summary.get("compile_attempts"),
            "compile_success": summary.get("compile_success"),
            "compile_fail": summary.get("compile_fail"),
            "compile_wall_seconds": summary.get("compile_wall_seconds"),
            "profile_attempts": summary.get("profile", {}).get("attempts"),
            "profile_wall_seconds": summary.get("profile_wall_seconds"),
            "validation_attempts": summary.get("validation", {}).get("attempts"),
            "validation_wall_seconds": summary.get("validation_wall_seconds"),
            "solver_checks": summary.get("solver", {}).get("checks"),
            "solver_wall_seconds": summary.get("solver_wall_seconds"),
            "model_attempts": summary.get("model", {}).get("attempts"),
            "model_wall_seconds": summary.get("model_wall_seconds"),
            "end_to_end_wall_seconds": summary.get("end_to_end_wall_seconds"),
            "best_latency_ms": summary.get("best", {}).get("latency_ms", summary.get("best_latency_ms")),
            "best_config": summary.get("best_config", summary.get("best", {}).get("config")),
            "reference_latency_ms": summary.get("reference_latency_ms"),
            "has_recorded_candidate_failures": summary.get("has_recorded_candidate_failures"),
            "status": summary.get("status"),
        })
    if not runs:
        raise ValueError(f"No GDN summary.json files found under {root}")
    return runs


def _by_baseline(runs, arm):
    return {(run["direction"], run["repetition"]): run for run in runs if run["arm"] == arm}


def build_comparisons(runs):
    stock = _by_baseline(runs, "stock-raw")
    oracle = _by_baseline(runs, "z3-legal-exhaustive")
    rows = []
    for run in runs:
        baseline = stock.get((run["direction"], run["repetition"]))
        reference = oracle.get((run["direction"], run["repetition"])) or baseline
        if baseline is None:
            continue
        row = dict(run)
        row.update({
            "search_speedup": _ratio(baseline["end_to_end_wall_seconds"], run["end_to_end_wall_seconds"]),
            "compile_reduction": 1 - _ratio(run["compile_attempts"], baseline["compile_attempts"]),
            "profile_reduction": 1 - _ratio(run["profile_attempts"], baseline["profile_attempts"]),
            "quality_ratio": _ratio(run["best_latency_ms"], reference["best_latency_ms"]),
            "oracle_best_config": reference["best_config"],
        })
        rows.append(row)
    return rows


def aggregate(comparisons):
    output = []
    for direction in ("fwd", "bwd"):
        for arm in ARMS:
            selected = [row for row in comparisons if row["direction"] == direction and row["arm"] == arm]
            if not selected:
                continue
            output.append({
                "direction": direction,
                "arm": arm,
                "repetitions": len(selected),
                "R": selected[0]["R"],
                "U": selected[0]["U"],
                "selected": selected[0]["selected"],
                "best_config": selected[0]["best_config"],
                "best_latency_ms": _median([row["best_latency_ms"] for row in selected]),
                "latency_range_ms": [min(row["best_latency_ms"] for row in selected), max(row["best_latency_ms"] for row in selected)],
                "search_speedup": _median([row["search_speedup"] for row in selected]),
                "compile_reduction": _median([row["compile_reduction"] for row in selected]),
                "profile_reduction": _median([row["profile_reduction"] for row in selected]),
                "quality_ratio": _median([row["quality_ratio"] for row in selected]),
                "compile_attempts": _median([row["compile_attempts"] for row in selected]),
                "profile_attempts": _median([row["profile_attempts"] for row in selected]),
                "validation_attempts": _median([row["validation_attempts"] for row in selected]),
                "solver_checks": _median([row["solver_checks"] for row in selected]),
                "model_attempts": _median([row["model_attempts"] for row in selected]),
                "solver_wall_seconds": _median([row["solver_wall_seconds"] for row in selected]),
                "model_wall_seconds": _median([row["model_wall_seconds"] for row in selected]),
                "compile_wall_seconds": _median([row["compile_wall_seconds"] for row in selected]),
                "profile_wall_seconds": _median([row["profile_wall_seconds"] for row in selected]),
                "validation_wall_seconds": _median([row["validation_wall_seconds"] for row in selected]),
                "end_to_end_wall_seconds": _median([row["end_to_end_wall_seconds"] for row in selected]),
                "reference_latency_ms": _median([row["reference_latency_ms"] for row in selected]),
                "has_recorded_candidate_failures": any(row["has_recorded_candidate_failures"] for row in selected),
                "oracle_best_config": selected[0]["oracle_best_config"],
            })
    return output


def _write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_visualization(repo_root):
    path = repo_root / "benchmark/z3_cost_model/visualization.py"
    spec = importlib.util.spec_from_file_location("gdn_result_visualization", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_analysis(root, output_dir, make_plots=True):
    runs = collect_runs(root)
    comparisons = build_comparisons(runs)
    aggregates = aggregate(comparisons)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "gdn-search-analysis/v1", "runs": comparisons, "aggregates": aggregates}
    (output / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    fields = [
        "direction", "repetition", "arm", "R", "U", "selected", "compile_attempts",
        "compile_success", "compile_fail", "profile_attempts", "validation_attempts", "solver_checks", "model_attempts",
        "solver_wall_seconds", "model_wall_seconds", "compile_wall_seconds",
        "profile_wall_seconds", "validation_wall_seconds", "end_to_end_wall_seconds",
        "best_latency_ms", "search_speedup", "compile_reduction", "profile_reduction",
        "quality_ratio", "best_config", "oracle_best_config", "reference_latency_ms",
        "has_recorded_candidate_failures", "status", "path",
    ]
    _write_csv(output / "runs.csv", comparisons, fields)
    aggregate_fields = [
        "direction", "arm", "repetitions", "R", "U", "selected", "best_config",
        "best_latency_ms", "latency_range_ms", "search_speedup", "compile_reduction",
        "profile_reduction", "quality_ratio", "compile_attempts", "profile_attempts",
        "validation_attempts", "solver_checks", "model_attempts", "solver_wall_seconds",
        "model_wall_seconds", "compile_wall_seconds", "profile_wall_seconds",
        "validation_wall_seconds", "end_to_end_wall_seconds", "reference_latency_ms",
        "has_recorded_candidate_failures", "oracle_best_config",
    ]
    _write_csv(output / "summary.csv", aggregates, aggregate_fields)
    if make_plots:
        viz = _load_visualization(Path(__file__).resolve().parents[3])
        acceptance = [{
            "suite": f"GDN {row['direction']}",
            "direction": row["direction"],
            "method": row["arm"],
            "best_latency": row["best_latency_ms"],
            "latency_unit": "ms",
            "compile_count": round(row["compile_attempts"]),
            "profile_count": round(row["profile_attempts"]),
        } for row in aggregates]
        viz.plot_gdn_acceptance(acceptance, output / "gdn-acceptance")
        time_runs = []
        for row in aggregates:
            components = {
                "solver": row["solver_wall_seconds"] or 0,
                "model": row["model_wall_seconds"] or 0,
                "compile": row["compile_wall_seconds"] or 0,
                "validation": row["validation_wall_seconds"] or 0,
                "profile": row["profile_wall_seconds"] or 0,
            }
            end_to_end = row["end_to_end_wall_seconds"]
            components["other"] = max(0, end_to_end - sum(components.values()))
            time_runs.append({
                "run": f"{row['direction']} {row['arm']}",
                "timing_semantics": "parallel_cumulative",
                "end_to_end_s": end_to_end,
                "times_s": components,
            })
        viz.plot_search_time_breakdown(time_runs, output / "search-time-breakdown")
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Directory containing GDN summary.json files")
    parser.add_argument("--output", required=True, help="Output directory for analysis and figures")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    payload = write_analysis(args.root, args.output, make_plots=not args.no_plots)
    print(json.dumps({"runs": len(payload["runs"]), "aggregates": len(payload["aggregates"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
