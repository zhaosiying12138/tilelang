"""Join classic cold-search counts/times with confirmed per-arm latencies.

Every suite/arm receives exactly one frozen winner config and one independently
confirmed latency.  Search-observed counts, times, and latencies remain separate
from confirmation measurements in JSON, CSV, and plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

_OPERATORS_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = _OPERATORS_ROOT.parents[2]
if __package__:
    from .autotune import ARMS, DEFAULT_TOP_K, SUITE_NAMES
    from .confirm import ORACLE_ARM, STOCK_ARM, _canonical
    from .. import visualization as viz
else:
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from benchmark.z3_cost_model.operators.autotune import ARMS, DEFAULT_TOP_K, SUITE_NAMES
    from benchmark.z3_cost_model.operators.confirm import ORACLE_ARM, STOCK_ARM, _canonical
    from benchmark.z3_cost_model import visualization as viz


_METHOD_COLORS = {
    "stock-raw": "#6B7280",
    "z3-legal-exhaustive": "#2563EB",
    "z3-cm-topk": "#EA580C",
    "budget-raw": "#059669",
    "budget-random": "#7C3AED",
}
_SEARCH_SPACE = {
    "stock-raw": "raw-full",
    "z3-legal-exhaustive": "legal",
    "z3-cm-topk": "legal",
    "budget-raw": "legal",
    "budget-random": "legal",
}


def _load(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _csv(path: str | Path, rows, fields) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _positive(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{context} must be a finite positive number, got {value!r}")
    return float(value)


def _nonnegative(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{context} must be finite and non-negative, got {value!r}")
    return float(value)


def _count(value: Any, *, context: str) -> int:
    number = _nonnegative(value, context=context)
    if not number.is_integer():
        raise ValueError(f"{context} must be integer-valued, got {value!r}")
    return int(number)


def _cold_index(cold_analysis: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(cold_analysis, dict) or not isinstance(cold_analysis.get("aggregates"), list):
        raise ValueError("Classic cold analysis must contain an aggregates list")
    index = {}
    for row in cold_analysis["aggregates"]:
        if not isinstance(row, dict) or row.get("top_k") != DEFAULT_TOP_K or row.get("cache_track") != "cold":
            continue
        key = (row.get("suite"), row.get("arm"))
        if key in index:
            raise ValueError(f"Duplicate cold aggregate for {key[0]}/{key[1]}")
        index[key] = row
    return index


def _correctness(summary: dict[str, Any], suite: str) -> dict[str, Any]:
    plan = summary["plan"]
    candidates = plan.get("candidates")
    validations = summary.get("validations")
    if not isinstance(candidates, list) or not candidates or not isinstance(validations, list):
        raise ValueError(f"Confirmation lacks candidates/validations for {suite}")
    expected = set()
    for candidate in candidates:
        config_id = _canonical(candidate.get("config"))
        for phase in ("before", "after"):
            expected.add((config_id, phase))
    observed = set()
    max_abs = 0.0
    max_rel = 0.0
    for validation in validations:
        if not isinstance(validation, dict):
            raise ValueError(f"Invalid validation record for {suite}")
        key = (validation.get("config_id"), validation.get("phase"))
        if key not in expected:
            raise ValueError(f"Unexpected validation assignment for {suite}: {key}")
        if key in observed:
            raise ValueError(f"Duplicate validation assignment for {suite}: {key}")
        observed.add(key)
        if (
            validation.get("status") != "ok"
            or validation.get("passed") is not True
            or validation.get("mismatched_ratio") != 0
            or validation.get("max_mismatched_ratio") != 0
        ):
            raise ValueError(f"Correctness did not pass for {suite}/{key[0]} {key[1]}")
        max_abs = max(max_abs, _nonnegative(validation.get("max_abs_error"), context=f"{suite} max_abs_error"))
        max_rel = max(max_rel, _nonnegative(validation.get("max_rel_error"), context=f"{suite} max_rel_error"))
    if observed != expected:
        raise ValueError(f"Missing before/after correctness validation for {suite}: {sorted(expected - observed)}")
    return {"passed": True, "max_abs_error": max_abs, "max_rel_error": max_rel, "validation_count": len(validations)}


def confirmed_records(cold_analysis: dict[str, Any], confirmations: dict[str, dict[str, Any]]):
    """Assign one config and one confirmed latency to every suite/arm exactly once."""

    if set(confirmations) != set(SUITE_NAMES):
        raise ValueError(f"Confirmations must cover exactly {list(SUITE_NAMES)}")
    cold = _cold_index(cold_analysis)
    rows = []
    comparisons = []
    for suite in SUITE_NAMES:
        summary = confirmations[suite]
        if summary.get("status") != "ok":
            raise ValueError(f"Confirmation is not complete for {suite}")
        if summary.get("suite") != suite:
            raise ValueError(f"Mixed confirmation suite: expected {suite}, got {summary.get('suite')!r}")
        plan = summary.get("plan")
        if not isinstance(plan, dict) or plan.get("selection_rule") != "candidate-median":
            raise ValueError(f"Confirmation for {suite} is not candidate-median selected")
        winners = plan.get("winners")
        if not isinstance(winners, dict) or set(winners) != set(ARMS):
            raise ValueError(f"Confirmation winners must cover every arm exactly once for {suite}")
        correctness = _correctness(summary, suite)
        assigned = {}
        for comparison in summary.get("comparisons", []):
            if not isinstance(comparison, dict) or not isinstance(comparison.get("identical"), bool):
                raise ValueError(f"Invalid comparison for {suite}")
            config = comparison.get("config")
            config_id = _canonical(config)
            if comparison["identical"]:
                arms = comparison.get("identical_arms")
                latency = comparison.get("guided_median_ms")
            else:
                arms = comparison.get("alternative_arms")
                latency = comparison.get("alternative_median_ms")
            if not isinstance(arms, list) or not arms:
                raise ValueError(f"Comparison has no arm assignment for {suite}/{config_id}")
            latency = _positive(latency, context=f"{suite}/{config_id} confirmed latency")
            for arm in arms:
                if arm not in ARMS:
                    raise ValueError(f"Unknown confirmed arm for {suite}: {arm!r}")
                if arm in assigned:
                    raise ValueError(f"Duplicate confirmed assignment for {suite}/{arm}")
                assigned[arm] = (config_id, latency)
            comparisons.append({"suite": suite, "latency_scope": "confirmed", **comparison})
        missing = [arm for arm in ARMS if arm not in assigned]
        if missing:
            raise ValueError(f"Missing confirmed assignment for {suite}: {missing}")
        for arm in ARMS:
            cold_key = (suite, arm)
            if cold_key not in cold:
                raise ValueError(f"Missing cold aggregate for {suite}/{arm}")
            cold_row = cold[cold_key]
            winner = winners[arm]
            if not isinstance(winner, dict) or winner.get("selection_rule") != "candidate-median":
                raise ValueError(f"Invalid frozen winner for {suite}/{arm}")
            winner_config = winner.get("config")
            winner_id = _canonical(winner_config)
            config_id, latency = assigned[arm]
            if winner_id != config_id:
                raise ValueError(f"Winner/config mismatch for {suite}/{arm}")
            compile_count = _count(cold_row.get("compile_attempts"), context=f"{suite}/{arm} compile_count")
            profile_count = _count(cold_row.get("profile_attempts"), context=f"{suite}/{arm} profile_count")
            search_wall = _nonnegative(cold_row.get("end_to_end_wall_seconds"), context=f"{suite}/{arm} search wall")
            search_observed_latency = cold_row.get("best_latency_ms")
            if search_observed_latency is not None:
                search_observed_latency = _positive(search_observed_latency, context=f"{suite}/{arm} search-observed latency")
            legal_complete = plan.get("legal_oracle_complete") is True
            if arm == STOCK_ARM:
                reference_semantics = "stock-raw full raw-space successful-trial reference; not a Z3 legal oracle"
            elif arm == ORACLE_ARM:
                reference_semantics = (
                    "complete Z3 legal oracle" if legal_complete else "incomplete legal exhaustive arm; not a complete oracle"
                )
            else:
                reference_semantics = "Z3 legal-space candidate arm"
            rows.append(
                {
                    "suite": suite,
                    "method": arm,
                    "config": winner_config,
                    "config_id": config_id,
                    "candidate": config_id,
                    "best_latency": latency,
                    "confirmed_latency": latency,
                    "latency_unit": "ms",
                    "latency_scope": "confirmed",
                    "search_observed_latency_ms": search_observed_latency,
                    "search_observed_latency_scope": "search_observed",
                    "compile_count": compile_count,
                    "profile_count": profile_count,
                    "count_scope": "search_observed",
                    "search_wall_seconds": search_wall,
                    "search_wall_s": search_wall,
                    "end_to_end_wall_seconds": search_wall,
                    "search_speedup": cold_row.get("search_speedup"),
                    "search_space": _SEARCH_SPACE[arm],
                    "reference_semantics": reference_semantics,
                    "legal_oracle_complete": legal_complete,
                    "cold_candidate_median_ms": winner.get("median_latency_ms"),
                    "cold_candidate_median_scope": "search_observed",
                    "cold_trial_count": winner.get("trial_count"),
                    "cold_measurement_repetitions": winner.get("measurement_repetitions"),
                    "solver_wall_seconds": cold_row.get("solver_wall_seconds"),
                    "model_wall_seconds": cold_row.get("model_wall_seconds"),
                    "compile_wall_seconds": cold_row.get("compile_wall_seconds"),
                    "validation_wall_seconds": cold_row.get("validation_wall_seconds"),
                    "profile_wall_seconds": cold_row.get("profile_wall_seconds"),
                    "time_scope": "search_observed",
                    "timing_semantics": "independent medians of phase spans; not additive",
                    "correctness_passed": correctness["passed"],
                    "max_abs_error": correctness["max_abs_error"],
                    "max_rel_error": correctness["max_rel_error"],
                    "confirmation_rounds": summary.get("rounds"),
                    "selection_rule": plan["selection_rule"],
                    "summary_path": summary.get("summary_path"),
                }
            )
    return rows, comparisons


def _pyplot():
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise ImportError("matplotlib is required for confirmation visualization") from error
    return plt


def _save_figure(fig, plt, prefix: Path) -> dict[str, str]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {"png": str(prefix.with_suffix(".png")), "svg": str(prefix.with_suffix(".svg"))}
    fig.savefig(paths["png"], dpi=180, bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    plt.close(fig)
    return paths


def _plot_acceptance(rows, prefix: Path) -> dict[str, str]:
    """Plot confirmed bars and cold counts without conflating their scopes."""

    plt = _pyplot()
    with plt.rc_context({"font.family": "sans-serif", "axes.unicode_minus": False, "figure.facecolor": "white"}):
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        max_count = max(max(row["compile_count"], row["profile_count"]) for row in rows)
        for ax, suite in zip(axes.flat, SUITE_NAMES, strict=True):
            suite_rows = sorted((row for row in rows if row["suite"] == suite), key=lambda row: ARMS.index(row["method"]))
            positions = list(range(len(suite_rows)))
            latencies = [row["best_latency"] for row in suite_rows]
            bars = ax.bar(
                positions,
                latencies,
                width=0.64,
                color=[_METHOD_COLORS[row["method"]] for row in suite_rows],
                alpha=0.82,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.bar_label(bars, labels=[f"{value:.6g}" for value in latencies], padding=3, fontsize=8)
            ax.set_xticks(positions, [row["method"] for row in suite_rows], rotation=16, ha="right")
            ax.set_ylabel("Confirmed latency (ms)")
            ax.set_ylim(0, max(latencies) * 1.24)
            ax.set_title(suite)
            ax.grid(axis="y", alpha=0.22)
            ax.set_axisbelow(True)
            count_axis = ax.twinx()
            count_axis.plot(
                positions,
                [row["compile_count"] for row in suite_rows],
                color="#111827",
                marker="o",
                linewidth=1.5,
                label="cold compile count",
            )
            count_axis.plot(
                positions,
                [row["profile_count"] for row in suite_rows],
                color="#DC2626",
                marker="s",
                linewidth=1.5,
                label="cold profile count",
            )
            count_axis.set_ylim(0, max(1, max_count) * 1.3 + 0.5)
            count_axis.set_ylabel("Cold search counts (search_observed)")
        handles = [plt.Rectangle((0, 0), 1, 1, color=_METHOD_COLORS[arm], alpha=0.82, label=arm) for arm in ARMS]
        handles.extend(
            [
                plt.Line2D([0], [0], color="#111827", marker="o", label="cold compile count (right axis)"),
                plt.Line2D([0], [0], color="#DC2626", marker="s", label="cold profile count (right axis)"),
            ]
        )
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=4, frameon=False)
        fig.suptitle("Classic confirmation: bars = confirmed latency; lines = cold search-observed counts", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        return _save_figure(fig, plt, prefix)


def _plot_search_time(rows, prefix: Path) -> dict[str, str]:
    """Plot independent cold phase-span medians without an additive claim."""

    plt = _pyplot()
    categories = (
        ("solver", "solver_wall_seconds", "#2563EB"),
        ("model", "model_wall_seconds", "#7C3AED"),
        ("compile", "compile_wall_seconds", "#0891B2"),
        ("validation", "validation_wall_seconds", "#F59E0B"),
        ("profile", "profile_wall_seconds", "#059669"),
        ("end-to-end", "end_to_end_wall_seconds", "#111827"),
    )
    with plt.rc_context({"font.family": "sans-serif", "axes.unicode_minus": False, "figure.facecolor": "white"}):
        fig, axes = plt.subplots(2, 2, figsize=(19, 12))
        for ax, suite in zip(axes.flat, SUITE_NAMES, strict=True):
            suite_rows = sorted((row for row in rows if row["suite"] == suite), key=lambda row: ARMS.index(row["method"]))
            width = 0.8 / len(categories)
            for category_index, (label, field, color) in enumerate(categories):
                for arm_index, row in enumerate(suite_rows):
                    value = row.get(field)
                    if value is None:
                        if category_index == 0:
                            ax.text(arm_index, 0, "solver n/a", ha="center", va="bottom", fontsize=7, color="#6B7280")
                        continue
                    value = _nonnegative(value, context=f"{suite}/{row['method']} {field}")
                    position = arm_index - 0.4 + width * (category_index + 0.5)
                    ax.bar([position], [value], width=width, color=color, label=label if arm_index == 0 else None)
            ax.set_xticks(range(len(suite_rows)), [row["method"] for row in suite_rows], rotation=16, ha="right")
            ax.set_ylabel("Cold wall/phase span (s), search_observed")
            ax.set_title(f"{suite}: independent medians, not additive")
            ax.grid(axis="y", alpha=0.22)
            ax.set_axisbelow(True)
        handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label) for label, _field, color in categories]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=6, frameon=False)
        fig.suptitle("Cold search timing only; confirmed kernel latencies are plotted separately", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        return _save_figure(fig, plt, prefix)


def _plot_pareto(rows, prefix: Path) -> dict[str, str]:
    projected = [
        {
            "suite": row["suite"],
            "method": row["method"],
            "search_space": row["search_space"],
            "candidate": row["candidate"],
            "search_wall_s": row["search_wall_s"],
            "profile_count": row["profile_count"],
            "confirmed_latency": row["confirmed_latency"],
            "latency_unit": row["latency_unit"],
        }
        for row in rows
    ]
    oracle = [
        {
            "suite": row["suite"],
            "search_space": "legal",
            "oracle_latency": row["confirmed_latency"],
            "latency_unit": "ms",
        }
        for row in rows
        if row["method"] == ORACLE_ARM and row["legal_oracle_complete"]
    ]
    return viz.plot_quality_search_pareto(projected, prefix, x_axis="search_wall_s", oracle=oracle or None)


def write_analysis(cold_analysis_path: str | Path, confirmation_root: str | Path, output_dir: str | Path, *, plots: bool = True):
    cold = _load(cold_analysis_path)
    root = Path(confirmation_root)
    confirmations = {}
    for suite in SUITE_NAMES:
        path = root / suite / "summary.json"
        summary = _load(path)
        if not isinstance(summary, dict):
            raise ValueError(f"Confirmation summary must be an object: {path}")
        summary["summary_path"] = str(path)
        confirmations[suite] = summary
    rows, comparisons = confirmed_records(cold, confirmations)
    output = Path(output_dir).expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "tilelang.classic-operators.confirmed-acceptance/v1",
        "rows": rows,
        "comparisons": comparisons,
        "correctness_passed": all(row["correctness_passed"] for row in rows),
        "acceptance_passed": all(summary["acceptance"]["passed"] is True for summary in confirmations.values()),
        "acceptance_scope": "execution_and_correctness_only",
        "performance_report_only": True,
        "latency_scopes": {
            "best_latency": "confirmed",
            "compile_profile_counts": "search_observed",
            "search_times": "search_observed",
            "cold_candidate_median": "search_observed; used only for frozen winner selection",
        },
        "reference_semantics": {
            "stock-raw": "full raw-space successful-trial reference, not a Z3 legal oracle",
            "z3-legal-exhaustive": "legal oracle only when all three cold runs have oracle_complete=true",
        },
        "source": {"cold_analysis": str(cold_analysis_path), "confirmation_root": str(root)},
        "artifacts": {},
    }
    if plots:
        payload["artifacts"] = {
            "confirmation": _plot_acceptance(rows, output / "classic-confirmation"),
            "quality_pareto": _plot_pareto(rows, output / "classic-quality-pareto"),
            "search_time": _plot_search_time(rows, output / "classic-search-time"),
        }
    _write(output / "analysis.json", payload)
    _csv(
        output / "confirmed-summary.csv",
        rows,
        (
            "suite",
            "method",
            "config",
            "config_id",
            "best_latency",
            "latency_unit",
            "latency_scope",
            "search_observed_latency_ms",
            "search_observed_latency_scope",
            "compile_count",
            "profile_count",
            "count_scope",
            "search_wall_seconds",
            "search_speedup",
            "search_space",
            "reference_semantics",
            "legal_oracle_complete",
            "cold_candidate_median_ms",
            "cold_candidate_median_scope",
            "cold_trial_count",
            "cold_measurement_repetitions",
            "solver_wall_seconds",
            "model_wall_seconds",
            "compile_wall_seconds",
            "validation_wall_seconds",
            "profile_wall_seconds",
            "time_scope",
            "timing_semantics",
            "correctness_passed",
            "max_abs_error",
            "max_rel_error",
            "confirmation_rounds",
            "selection_rule",
            "summary_path",
        ),
    )
    _csv(
        output / "confirmation-comparisons.csv",
        comparisons,
        (
            "suite",
            "alternative_arms",
            "identical_arms",
            "identical",
            "config",
            "config_id",
            "relations",
            "guided_median_ms",
            "alternative_median_ms",
            "quality_ratio_guided_over_alternative",
            "paired_round_differences_ms",
            "paired_difference_statistics",
            "paired_difference_range_ms",
            "guided_round_statistics",
            "alternative_round_statistics",
            "measurement_evidence",
            "legal_oracle_complete",
            "requirements",
            "passed",
            "latency_scope",
        ),
    )
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-analysis", "--analysis", dest="cold_analysis", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = write_analysis(args.cold_analysis, args.confirmation_root, args.output)
    print(
        json.dumps(
            {
                "rows": len(payload["rows"]),
                "correctness_passed": payload["correctness_passed"],
                "acceptance_passed": payload["acceptance_passed"],
                "output": str(Path(args.output).expanduser().resolve(strict=False)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
