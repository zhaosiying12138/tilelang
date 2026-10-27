"""Join cold-search counts with independently confirmed GDN winner latencies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from .analyze_results import ARMS
    from .confirm import _canonical
    from .. import visualization as viz
except ImportError:
    import importlib.util

    from analyze_results import ARMS
    from confirm import _canonical

    path = Path(__file__).resolve().parents[1] / "visualization.py"
    spec = importlib.util.spec_from_file_location("gdn_confirmation_visualization", path)
    viz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viz)


def _load(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def _csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def confirmed_records(cold_analysis, confirmations):
    """Map each arm to exactly one config and one independently measured latency."""

    if set(confirmations) != {"fwd", "bwd"}:
        raise ValueError("Both fwd and bwd confirmations are required")
    cold = {
        (row["direction"], row["arm"]): row
        for row in cold_analysis["aggregates"]
    }
    rows = []
    comparisons = []
    for direction in ("fwd", "bwd"):
        summary = confirmations[direction]
        if summary.get("status") != "ok":
            raise ValueError(f"Confirmation is not complete for {direction}")
        num_tokens = summary["num_tokens"]
        assigned = {}
        for comparison in summary["comparisons"]:
            config_id = _canonical(comparison["config"])
            if comparison["identical"]:
                arms = comparison["identical_arms"]
                latency = comparison["guided_median_ms"]
            else:
                arms = comparison["alternative_arms"]
                latency = comparison["alternative_median_ms"]
            for arm in arms:
                if arm in assigned:
                    raise ValueError(f"Duplicate confirmed assignment for {direction}/{arm}")
                assigned[arm] = (config_id, latency)
            comparisons.append({"direction": direction, **comparison})
        for arm in ARMS:
            if arm not in assigned:
                raise ValueError(f"Missing confirmed assignment for {direction}/{arm}")
            cold_row = cold[(direction, arm)]
            config_id, latency = assigned[arm]
            winner_config = summary["plan"]["winners"][arm]["config"]
            if _canonical(winner_config) != config_id:
                raise ValueError(f"Winner/config mismatch for {direction}/{arm}")
            rows.append({
                "suite": f"GDN {direction} T={num_tokens}",
                "direction": direction,
                "num_tokens": num_tokens,
                "method": arm,
                "config": winner_config,
                "best_latency": latency,
                "latency_unit": "ms",
                "latency_scope": "confirmed",
                "compile_count": cold_row["compile_attempts"],
                "profile_count": cold_row["profile_attempts"],
                "search_wall_seconds": cold_row["end_to_end_wall_seconds"],
                "search_speedup": cold_row["search_speedup"],
                "max_relative_l2": max(item["max_relative_l2"] for item in summary["validations"]),
                "confirmation_rounds": summary["rounds"],
                "selection_rule": summary["plan"]["selection_rule"],
                "summary_path": summary.get("summary_path"),
            })
    return rows, comparisons


def write_analysis(cold_analysis_path, confirmation_root, output_dir):
    cold = _load(cold_analysis_path)
    root = Path(confirmation_root)
    confirmations = {}
    for direction in ("fwd", "bwd"):
        path = root / direction / "summary.json"
        summary = _load(path)
        summary["summary_path"] = str(path)
        confirmations[direction] = summary
    rows, comparisons = confirmed_records(cold, confirmations)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "tilelang.gdn.confirmed-acceptance/v1",
        "rows": rows,
        "comparisons": comparisons,
        "acceptance_passed": all(summary["acceptance"]["passed"] for summary in confirmations.values()),
        "source": {
            "cold_analysis": str(cold_analysis_path),
            "confirmation_root": str(root),
        },
    }
    _write(output / "analysis.json", payload)
    _csv(output / "confirmed-summary.csv", rows, (
        "suite", "direction", "num_tokens", "method", "config", "best_latency",
        "latency_unit", "latency_scope", "compile_count", "profile_count",
        "search_wall_seconds", "search_speedup", "max_relative_l2",
        "confirmation_rounds", "selection_rule", "summary_path",
    ))
    _csv(output / "confirmation-comparisons.csv", comparisons, (
        "direction", "alternative_arms", "identical_arms", "identical", "config",
        "guided_median_ms", "alternative_median_ms", "quality_ratio_guided_over_alternative",
        "paired_round_differences_ms", "requirements", "passed",
    ))
    viz.plot_gdn_acceptance(rows, output / "gdn-confirmation")
    if any(not comparison["identical"] for comparison in comparisons):
        viz.plot_gdn_confirmation_differences(comparisons, output / "gdn-paired-differences")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-analysis", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = write_analysis(args.cold_analysis, args.confirmation_root, args.output)
    print(json.dumps({"rows": len(payload["rows"]), "acceptance_passed": payload["acceptance_passed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
