"""Strict, CPU-only visualizations for CUDA Z3 search and scheduling results.

Every public plot function accepts either an in-memory JSON-like object or a
JSON file path.  ``output`` is a filename prefix, or one of the generated
filenames.  Each call writes PNG, SVG, CSV, and JSON artifacts.  No function
imports TileLang, TVM, torch, CUDA, SciPy, or NumPy, and no function supplies
experimental values when input data are missing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import csv
import json
import math
import operator
import os
from pathlib import Path
import re
from typing import Any
import warnings


__all__ = [
    "kendall_tau_b",
    "plot_gdn_acceptance",
    "plot_prediction_rank",
    "plot_pruning_funnel",
    "plot_quality_search_pareto",
    "plot_schedule_timeline",
    "plot_search_time_breakdown",
    "rankdata",
    "spearman_rank_correlation",
    "top_k_recall",
]

_MISSING = object()
_DERIVE_OTHER = object()
_TIME_CATEGORIES = ("solver", "model", "compile", "validation", "profile", "other")
_METHOD_ORDER = ("stock-raw", "z3-legal", "z3-cm", "control")
_METHOD_COLORS = {
    "stock-raw": "#6B7280",
    "z3-legal": "#2563EB",
    "z3-cm": "#EA580C",
    "control": "#059669",
}
_RESOURCE_NAMES = (
    (1, "GLOBAL_LOAD", "GLOAD"),
    (2, "GLOBAL_STORE", "GSTORE"),
    (4, "SHARED_MEMORY", "SMEM"),
    (8, "TENSOR_CORE", "TCORE"),
    (16, "CUDA_CORE", "CUDA"),
    (32, "SFU", "SFU"),
    (64, "SCALAR", "SCALAR"),
)
_FONT_FALLBACK = [
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
_RECORD_CONTAINERS = ("records", "runs", "results", "items", "predictions", "measurements", "points")


def _read_source(source: Any, name: str) -> Any:
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"{name} JSON file does not exist: {path}")
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    return source


def _mapping(source: Any, name: str) -> Mapping[str, Any]:
    data = _read_source(source, name)
    if not isinstance(data, Mapping):
        raise TypeError(f"{name} must be a JSON object or a path to one")
    if not data:
        raise ValueError(f"{name} must not be empty")
    return data


def _record_list(source: Any, name: str) -> list[Mapping[str, Any]]:
    data = _read_source(source, name)
    if isinstance(data, Mapping):
        if not data:
            raise ValueError(f"{name} must not be empty")
        containers = [key for key in _RECORD_CONTAINERS if key in data]
        if len(containers) > 1:
            raise ValueError(f"{name} has ambiguous record containers: {containers}")
        if containers:
            data = data[containers[0]]
    if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
        raise TypeError(f"{name} must be a record sequence, a supported JSON wrapper, or a JSON file path")
    if not data:
        raise ValueError(f"{name} must not be empty")
    records = list(data)
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"{name}[{index}] must be an object")
    return records


def _pick(mapping: Mapping[str, Any], names: Sequence[str], label: str, default: Any = _MISSING) -> Any:
    present = [name for name in names if name in mapping]
    if len(present) > 1:
        raise ValueError(f"{label} is ambiguous; specify only one of {present}")
    if not present:
        if default is _MISSING:
            raise ValueError(f"{label} is required (field {names[0]!r})")
        return default
    return mapping[present[0]]


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _identifier(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"{name} must be a string or integer identifier")
    return _text(str(value), name)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer, got {value!r}") from error
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _number(value: Any, name: str, minimum: float | None = None, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be > 0, got {result}")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _output_base(output: str | os.PathLike[str]) -> Path:
    path = Path(output)
    if not path.name:
        raise ValueError("output must name a file prefix")
    if path.suffix.lower() in {".png", ".svg", ".csv", ".json"}:
        return path.with_suffix("")
    if path.suffix:
        raise ValueError("output must be a prefix or end in .png, .svg, .csv, or .json")
    return path


def _pyplot():
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise ImportError("matplotlib is required for result visualization") from error
    return plt


def _rc_context(plt):
    return plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": _FONT_FALLBACK,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "path",
        }
    )


def _write_sidecars(base: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str], payload: dict[str, Any]) -> dict[str, str]:
    csv_path = base.with_suffix(".csv")
    json_path = base.with_suffix(".json")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})
    document = {"schema_version": 1, **payload, "rows": rows}
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {"csv": str(csv_path), "json": str(json_path)}


def _emit(
    fig,
    plt,
    output: str | os.PathLike[str],
    rows: list[dict[str, Any]],
    fieldnames: Sequence[str],
    payload: dict[str, Any],
) -> dict[str, str]:
    base = _output_base(output)
    base.parent.mkdir(parents=True, exist_ok=True)
    png_path = base.with_suffix(".png")
    svg_path = base.with_suffix(".svg")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    paths = {"png": str(png_path), "svg": str(svg_path)}
    paths.update(_write_sidecars(base, rows, fieldnames, payload))
    return paths


def _resource_label(mask: int, short: bool = False) -> str:
    if mask == 0:
        return "NONE"
    labels = []
    position = 2 if short else 1
    for bit, *names in _RESOURCE_NAMES:
        if mask & bit:
            labels.append(names[position - 1])
    known_mask = sum(bit for bit, *_ in _RESOURCE_NAMES)
    unknown = mask & ~known_mask
    for bit in range(unknown.bit_length()):
        if unknown & (1 << bit):
            labels.append(f"BIT{bit}")
    return "+".join(labels)


def _schedule_data(schedule_info: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    info = _mapping(schedule_info, "schedule_info")
    kind = _text(info.get("schedule_kind"), "schedule_kind").lower().replace("_", "-").replace(" ", "-")
    if kind in {"straight", "straightline", "straight-line"}:
        kind = "straight-line"
    elif kind in {"loop", "modulo", "modulo-loop"}:
        kind = "loop"
    else:
        raise ValueError(f"Unsupported schedule_kind {info.get('schedule_kind')!r}; expected loop or straight-line")

    names = ("start_times", "latencies", "iis", "resource_flags")
    arrays = {}
    for name in names:
        value = info.get(name)
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"schedule_info.{name} must be an array")
        arrays[name] = list(value)
    lengths = {name: len(value) for name, value in arrays.items()}
    if not lengths["start_times"]:
        raise ValueError("schedule_info must contain at least one task")
    if len(set(lengths.values())) != 1:
        raise ValueError(f"schedule_info task arrays have inconsistent lengths: {lengths}")
    count = lengths["start_times"]
    if "num_tasks" in info and _integer(info["num_tasks"], "num_tasks") != count:
        raise ValueError(f"num_tasks={info['num_tasks']} does not match {count} task rows")

    minimal_ii = _pick(info, ("minimal_II", "minimal_ii"), "minimal_II", default=None)
    if minimal_ii is not None:
        minimal_ii = _integer(minimal_ii, "minimal_II", 1)
    if kind == "loop" and minimal_ii is None:
        raise ValueError("loop schedule_info requires minimal_II")
    if "resource_bits" in info:
        resource_bits = _integer(info["resource_bits"], "resource_bits", 1)
    else:
        resource_bits = max(1, max((_integer(mask, "resource_flags") for mask in arrays["resource_flags"]), default=0).bit_length())
    supplied_stages = info.get("stages")
    if supplied_stages is not None:
        if isinstance(supplied_stages, (str, bytes)) or not isinstance(supplied_stages, Sequence):
            raise TypeError("schedule_info.stages must be an array")
        if len(supplied_stages) != count:
            raise ValueError("schedule_info.stages must have one entry per task")
    task_names = info.get("task_names")
    if task_names is not None:
        if isinstance(task_names, (str, bytes)) or not isinstance(task_names, Sequence) or len(task_names) != count:
            raise ValueError("schedule_info.task_names must be an array with one entry per task")

    rows = []
    for index in range(count):
        start = _integer(arrays["start_times"][index], f"start_times[{index}]")
        latency = _integer(arrays["latencies"][index], f"latencies[{index}]")
        ii = _integer(arrays["iis"][index], f"iis[{index}]", 1)
        mask = _integer(arrays["resource_flags"][index], f"resource_flags[{index}]")
        if mask >= 1 << resource_bits:
            raise ValueError(f"resource_flags[{index}]={mask} does not fit in resource_bits={resource_bits}")
        if supplied_stages is not None:
            stage = _integer(supplied_stages[index], f"stages[{index}]")
            if kind == "loop" and stage != start // minimal_ii:
                raise ValueError(f"stages[{index}]={stage} contradicts start//minimal_II={start // minimal_ii}")
        elif kind == "loop":
            stage = start // minimal_ii
        else:
            stage = None
        task = f"T{index}" if task_names is None else f"T{index} {_text(task_names[index], f'task_names[{index}]')}"
        rows.append(
            {
                "task_index": index,
                "task": task,
                "resource_mask": mask,
                "resource_bits": resource_bits,
                "resources": _resource_label(mask),
                "start": start,
                "latency": latency,
                "ii": ii,
                "stage": stage,
                "residue": start % minimal_ii if kind == "loop" else None,
                "end": start + latency,
            }
        )
    makespan = max(row["end"] for row in rows)
    if "makespan" in info and _integer(info["makespan"], "makespan") != makespan:
        raise ValueError(f"makespan={info['makespan']} does not match max(start+latency)={makespan}")
    fallback = _boolean(info.get("fallback", False), "fallback")
    metadata = {
        "schedule_kind": kind,
        "minimal_II": minimal_ii,
        "makespan": makespan,
        "makespan_source": "schedule_info" if "makespan" in info else "computed_from_rows",
        "fallback": fallback,
        "units": "cycles",
    }
    return metadata, rows


def plot_schedule_timeline(schedule_info: Any, output: str | os.PathLike[str]) -> dict[str, str]:
    """Plot native CUDA Z3 ``schedule_info.json`` task/resource timelines."""

    _output_base(output)
    metadata, rows = _schedule_data(schedule_info)
    plt = _pyplot()
    with _rc_context(plt):
        height = max(4.0, 0.55 * len(rows) + 2.0)
        fig, ax = plt.subplots(figsize=(14, height))
        try:
            cmap = plt.get_cmap("tab20")
            for y, row in enumerate(rows):
                color = cmap(row["resource_mask"] % 20)
                ax.barh(y, row["latency"], left=row["start"], height=0.72, color=color, edgecolor="black", linewidth=0.6)
                stage = "n/a" if row["stage"] is None else str(row["stage"])
                label = f"start={row['start']}  L={row['latency']}  II={row['ii']}  S={stage}"
                if row["latency"]:
                    ax.text(row["start"] + row["latency"] / 2, y, label, ha="center", va="center", fontsize=7, clip_on=True)
                else:
                    ax.scatter([row["start"]], [y], marker="|", color="black", s=90, zorder=4)
                    ax.text(row["start"], y - 0.38, label, ha="left", va="bottom", fontsize=7)
            labels = [f"{row['task']} | {_resource_label(row['resource_mask'], short=True)}" for row in rows]
            ax.set_yticks(range(len(rows)), labels)
            ax.invert_yaxis()
            ax.set_xlabel("Time (cycles)")
            ax.set_ylabel("Task | resource mask")
            ax.grid(axis="x", alpha=0.25)
            ax.set_axisbelow(True)
            title_kind = "Loop modulo schedule" if metadata["schedule_kind"] == "loop" else "Straight-line schedule"
            title = f"{title_kind} | makespan={metadata['makespan']} cycles"
            if metadata["minimal_II"] is not None:
                title += f" | minimal II={metadata['minimal_II']} cycles"
            if metadata["fallback"]:
                title += " | serial fallback"
            ax.set_title(title)
            grid_note = ""
            if metadata["schedule_kind"] == "loop":
                minimal_ii = metadata["minimal_II"]
                if metadata["makespan"] // minimal_ii <= 128:
                    for boundary in range(minimal_ii, metadata["makespan"] + 1, minimal_ii):
                        ax.axvline(boundary, color="#6B7280", linestyle=":", linewidth=0.8, alpha=0.7)
                else:
                    grid_note = "; stage grid omitted above 128 boundaries"
                note = "Bar width=latency; task II is issue occupancy, not bar width. Stage=start//minimal_II; vertical dots mark II boundaries"
                note += grid_note + "."
            else:
                note = "Bar width=latency; task II is issue occupancy. Straight-line output has no minimal II; stage is shown only when supplied."
            fig.text(0.01, 0.005, note, fontsize=9, color="#374151")
            ax.set_xlim(0, max(1, metadata["makespan"]) * 1.08)
            fig.tight_layout(rect=(0, 0.035, 1, 1))
            fields = ("task_index", "task", "resource_mask", "resource_bits", "resources", "start", "latency", "ii", "stage", "residue", "end")
            return _emit(fig, plt, output, rows, fields, {"plot": "schedule_timeline", **metadata})
        finally:
            plt.close(fig)


def _count(mapping: Mapping[str, Any], aliases: Sequence[str], label: str, default: Any = _MISSING) -> int:
    return _integer(_pick(mapping, aliases, label, default=default), label)


def _pruning_data(summary: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = _mapping(summary, "summary")
    counts = source.get("counts", source)
    if not isinstance(counts, Mapping):
        raise TypeError("summary.counts must be an object")
    raw = _count(counts, ("raw", "raw_candidates"), "raw")
    sat = _count(counts, ("sat", "sat_count", "sat_candidates"), "sat")
    unknown = _count(counts, ("unknown", "unknown_count", "unknown_candidates"), "unknown")
    unsat = _count(counts, ("unsat", "unsat_count", "unsat_candidates"), "unsat")
    dedup = _count(counts, ("dedup", "dedup_count", "deduplicated"), "dedup")
    compiled = _count(counts, ("compiled", "compile_succeeded"), "compiled")
    profiled = _count(counts, ("profiled", "profile_succeeded"), "profiled")
    other = _count(counts, ("other", "other_count"), "other", default=0)
    if raw == 0:
        raise ValueError("pruning summary is empty: raw must be > 0")
    retained = sat + unknown
    accounted = retained + unsat + dedup + other
    if accounted != raw:
        raise ValueError(f"Inconsistent raw denominator: raw={raw}, but sat+unknown+unsat+dedup+other={accounted}")
    if compiled > retained:
        raise ValueError(f"compiled={compiled} exceeds SAT/unknown retained={retained}")
    if profiled > compiled:
        raise ValueError(f"profiled={profiled} exceeds compiled={compiled}")
    metadata = {
        "raw": raw,
        "sat": sat,
        "unknown": unknown,
        "unsat": unsat,
        "dedup": dedup,
        "other": other,
        "retained": retained,
        "compiled": compiled,
        "profiled": profiled,
        "denominator": "raw",
    }
    definitions = (
        ("raw", raw, "input"),
        ("sat", sat, "retained_solver_outcome"),
        ("unknown", unknown, "retained_solver_outcome_not_unsat"),
        ("unsat", unsat, "proven_pruned"),
        ("dedup", dedup, "duplicate_removed_not_unsat"),
        ("other", other, "explicit_other"),
        ("retained", retained, "sat_plus_unknown"),
        ("compiled", compiled, "successful_stage"),
        ("profiled", profiled, "successful_stage"),
    )
    rows = [
        {"category": name, "count": value, "denominator": raw, "fraction_of_raw": value / raw, "role": role}
        for name, value, role in definitions
    ]
    return metadata, rows


def plot_pruning_funnel(summary: Any, output: str | os.PathLike[str]) -> dict[str, str]:
    """Plot raw accounting and the retained/compiled/profiled candidate funnel."""

    _output_base(output)
    metadata, rows = _pruning_data(summary)
    plt = _pyplot()
    with _rc_context(plt):
        fig, (ax, audit) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.25, 1]})
        try:
            raw = metadata["raw"]
            stages = (
                ("Raw", raw, "#475569"),
                ("SAT/unknown retained\n(after UNSAT pruning + dedup)", metadata["retained"], "#2563EB"),
                ("Compiled", metadata["compiled"], "#0891B2"),
                ("Profiled", metadata["profiled"], "#059669"),
            )
            y_positions = list(range(len(stages)))
            for y, (_, count, color) in zip(y_positions, stages):
                ax.barh(y, count, left=(raw - count) / 2, height=0.65, color=color, alpha=0.9)
                text_style = {"color": "white"}
                if count / raw < 0.25:
                    text_style = {
                        "color": "#111827",
                        "bbox": {"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5},
                    }
                ax.text(raw / 2, y, f"{count}  ({count / raw:.1%} of raw)", ha="center", va="center", fontweight="bold", **text_style)
            ax.set_yticks(y_positions, [stage[0] for stage in stages])
            ax.invert_yaxis()
            ax.set_xlim(0, raw)
            ax.set_xlabel("Candidates (common denominator = raw)")
            ax.set_title("Survivor funnel")

            categories = (
                ("SAT retained", metadata["sat"], "#2563EB"),
                ("unknown retained", metadata["unknown"], "#F59E0B"),
                ("UNSAT pruned", metadata["unsat"], "#DC2626"),
                ("deduplicated", metadata["dedup"], "#9333EA"),
                ("explicit other", metadata["other"], "#9CA3AF"),
            )
            left = 0
            for label, count, color in categories:
                if count == 0:
                    continue
                audit.barh([0], [count], left=left, color=color, height=0.5, label=f"{label}: {count}")
                if count / raw >= 0.07:
                    audit.text(left + count / 2, 0, str(count), ha="center", va="center", color="white", fontweight="bold")
                left += count
            audit.set_xlim(0, raw)
            audit.set_yticks([0], ["Raw accounting"])
            audit.set_xlabel(f"All categories partition raw={raw}")
            audit.set_title("Solver outcomes and dedup are disjoint")
            audit.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
            fig.suptitle("Candidate pruning: retained candidates are not a pruning percentage", y=0.985)
            fig.text(
                0.5,
                0.925,
                f"UNSAT pruned={metadata['unsat']}; deduplicated={metadata['dedup']}; unknown and dedup are separate and are never added as UNSAT",
                ha="center",
                fontsize=9,
            )
            fig.tight_layout(rect=(0, 0.04, 1, 0.88))
            fields = ("category", "count", "denominator", "fraction_of_raw", "role")
            return _emit(fig, plt, output, rows, fields, {"plot": "pruning_funnel", **metadata})
        finally:
            plt.close(fig)


def _time_value(run: Mapping[str, Any], nested: Mapping[str, Any], category: str, run_label: str, default: Any = _MISSING) -> Any:
    nested_names = (category, f"{category}_s")
    top_names = (f"{category}_s", f"{category}_time_s")
    nested_present = [name for name in nested_names if name in nested]
    top_present = [name for name in top_names if name in run]
    if nested_present and top_present:
        raise ValueError(f"{run_label}: {category} appears in both times_s and top-level fields")
    if len(nested_present) > 1 or len(top_present) > 1:
        raise ValueError(f"{run_label}: ambiguous aliases for {category}")
    if nested_present:
        return nested[nested_present[0]]
    if top_present:
        return run[top_present[0]]
    if default is _MISSING:
        raise ValueError(f"{run_label}: missing {category} seconds")
    return default


def _search_time_data(run_summaries: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    records = _record_list(run_summaries, "run_summaries")
    runs = []
    rows = []
    seen = set()
    for index, record in enumerate(records):
        label = _text(_pick(record, ("run", "run_id", "name", "suite"), f"run_summaries[{index}].run"), f"run[{index}]")
        if label in seen:
            raise ValueError(f"Duplicate run label {label!r}")
        seen.add(label)
        semantics = _pick(record, ("timing_semantics", "timing_mode"), f"{label}.timing_semantics", default=None)
        parallel = _pick(record, ("parallel", "parallel_cumulative"), f"{label}.parallel", default=None)
        if semantics is not None and parallel is not None:
            raise ValueError(f"{label}: specify timing_semantics or parallel, not both")
        if semantics is None:
            if parallel is None:
                raise ValueError(f"{label}: timing_semantics='additive' or 'parallel_cumulative' is required")
            semantics = "parallel_cumulative" if _boolean(parallel, f"{label}.parallel") else "additive"
        semantics = _text(semantics, f"{label}.timing_semantics").lower().replace("-", "_").replace(" ", "_")
        if semantics in {"parallel", "cumulative", "parallel_accumulated"}:
            semantics = "parallel_cumulative"
        if semantics not in {"additive", "parallel_cumulative"}:
            raise ValueError(f"{label}: unsupported timing semantics {semantics!r}")
        end_to_end = _number(
            _pick(record, ("end_to_end_s", "end_to_end", "wall_time_s"), f"{label}.end_to_end_s"),
            f"{label}.end_to_end_s",
            minimum=0,
        )
        nested = record.get("times_s", record.get("timings_s", {}))
        if not isinstance(nested, Mapping):
            raise TypeError(f"{label}.times_s must be an object")
        values = {}
        origins = {}
        for category in _TIME_CATEGORIES:
            default = _DERIVE_OTHER if category == "other" and semantics == "additive" else _MISSING
            raw = _time_value(record, nested, category, label, default=default)
            if raw is _DERIVE_OTHER:
                continue
            values[category] = _number(raw, f"{label}.{category}_s", minimum=0)
            origins[category] = "input"
        if "other" not in values:
            residual = end_to_end - sum(values.values())
            tolerance = max(1e-9, abs(end_to_end) * 1e-9)
            if residual < -tolerance:
                raise ValueError(f"{label}: additive components exceed end_to_end_s by {-residual:.6g} seconds")
            values["other"] = max(0.0, residual)
            origins["other"] = "derived_residual"
        total = sum(values.values())
        if semantics == "additive" and not math.isclose(total, end_to_end, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{label}: additive components+other={total} do not match end_to_end_s={end_to_end}")
        runs.append({"run": label, "timing_semantics": semantics, "end_to_end_s": end_to_end, "times_s": values})
        for category in _TIME_CATEGORIES:
            rows.append(
                {
                    "run": label,
                    "timing_semantics": semantics,
                    "category": category,
                    "seconds": values[category],
                    "end_to_end_s": end_to_end,
                    "value_origin": origins[category],
                }
            )
    parallel_runs = [run["run"] for run in runs if run["timing_semantics"] == "parallel_cumulative"]
    metadata = {
        "stacked": not parallel_runs,
        "parallel_cumulative_runs": parallel_runs,
        "warning": None,
    }
    if parallel_runs:
        message = (
            "Parallel-cumulative component timings can overlap and must not be summed to end-to-end wall time; "
            "all runs are shown as independent bars. Affected runs: " + ", ".join(parallel_runs)
        )
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        metadata["warning"] = message
    return metadata, rows, runs


def plot_search_time_breakdown(run_summaries: Any, output: str | os.PathLike[str]) -> dict[str, str]:
    """Plot additive wall-time stacks or non-additive parallel timings safely."""

    _output_base(output)
    metadata, rows, runs = _search_time_data(run_summaries)
    plt = _pyplot()
    with _rc_context(plt):
        fig, ax = plt.subplots(figsize=(max(11, 2.1 * len(runs)), 6.5))
        try:
            colors = ("#2563EB", "#7C3AED", "#0891B2", "#F59E0B", "#059669", "#6B7280")
            x_positions = list(range(len(runs)))
            if metadata["stacked"]:
                bottoms = [0.0] * len(runs)
                for category, color in zip(_TIME_CATEGORIES, colors):
                    values = [run["times_s"][category] for run in runs]
                    ax.bar(x_positions, values, bottom=bottoms, width=0.68, label=category, color=color)
                    bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
                ax.set_ylabel("Wall time (s); additive stack")
                ax.set_title("Search time breakdown reconciled to end-to-end")
                note = "Explicit or residual 'other' makes each additive stack equal end-to-end. Black diamonds are measured end-to-end wall time."
            else:
                width = 0.8 / len(_TIME_CATEGORIES)
                for category_index, (category, color) in enumerate(zip(_TIME_CATEGORIES, colors)):
                    positions = [x - 0.4 + width * (category_index + 0.5) for x in x_positions]
                    values = [run["times_s"][category] for run in runs]
                    ax.bar(positions, values, width=width, label=category, color=color)
                ax.set_ylabel("Reported component time (s); independent bars")
                ax.set_title("Search component timings (no additive stack)")
                note = "At least one run uses parallel-cumulative timings. Components can overlap: do not sum them or compare their sum with end-to-end."
            end_to_end = [run["end_to_end_s"] for run in runs]
            ax.scatter(x_positions, end_to_end, marker="D", color="black", s=55, label="end-to-end", zorder=5)
            for x, value in zip(x_positions, end_to_end):
                ax.annotate(f"{value:.6g}s", (x, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
            ax.set_xticks(x_positions, [run["run"] for run in runs], rotation=15, ha="right")
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
            fig.text(0.01, 0.01, note, fontsize=9, color="#374151")
            fig.tight_layout(rect=(0, 0.09, 1, 1))
            fields = ("run", "timing_semantics", "category", "seconds", "end_to_end_s", "value_origin")
            return _emit(fig, plt, output, rows, fields, {"plot": "search_time_breakdown", **metadata, "runs": runs})
        finally:
            plt.close(fig)


def _paired_values(predicted: Sequence[Any], measured: Sequence[Any]) -> tuple[list[float], list[float]]:
    if len(predicted) != len(measured):
        raise ValueError(f"predicted and measured lengths differ: {len(predicted)} != {len(measured)}")
    if not predicted:
        raise ValueError("ranking values must not be empty")
    x = [_number(value, f"predicted[{index}]") for index, value in enumerate(predicted)]
    y = [_number(value, f"measured[{index}]") for index, value in enumerate(measured)]
    return x, y


def rankdata(values: Sequence[Any]) -> list[float]:
    """Return one-based ascending ranks with average ranks for ties."""

    if not values:
        raise ValueError("rankdata values must not be empty")
    numbers = [_number(value, f"values[{index}]") for index, value in enumerate(values)]
    order = sorted(range(len(numbers)), key=lambda index: (numbers[index], index))
    ranks = [0.0] * len(numbers)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and numbers[order[end]] == numbers[order[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def spearman_rank_correlation(predicted: Sequence[Any], measured: Sequence[Any]) -> float | None:
    """Compute Spearman rho from average ranks; return None when undefined."""

    x, y = _paired_values(predicted, measured)
    if len(x) < 2:
        return None
    x_ranks = rankdata(x)
    y_ranks = rankdata(y)
    x_mean = sum(x_ranks) / len(x_ranks)
    y_mean = sum(y_ranks) / len(y_ranks)
    x_centered = [value - x_mean for value in x_ranks]
    y_centered = [value - y_mean for value in y_ranks]
    denominator = math.sqrt(sum(value * value for value in x_centered) * sum(value * value for value in y_centered))
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(x_centered, y_centered)) / denominator


def kendall_tau_b(predicted: Sequence[Any], measured: Sequence[Any]) -> float | None:
    """Compute Kendall tau-b, including tie correction; return None when undefined."""

    x, y = _paired_values(predicted, measured)
    count = len(x)
    if count < 2:
        return None
    concordant = 0
    discordant = 0
    for left in range(count):
        for right in range(left + 1, count):
            x_direction = (x[left] > x[right]) - (x[left] < x[right])
            y_direction = (y[left] > y[right]) - (y[left] < y[right])
            product = x_direction * y_direction
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    pairs = count * (count - 1) / 2
    x_ties = sum(size * (size - 1) / 2 for size in Counter(x).values())
    y_ties = sum(size * (size - 1) / 2 for size in Counter(y).values())
    denominator = math.sqrt((pairs - x_ties) * (pairs - y_ties))
    if denominator == 0:
        return None
    return (concordant - discordant) / denominator


def top_k_recall(predicted: Sequence[Any], measured: Sequence[Any], k: int) -> float:
    """Recall of measured's lowest-k items in predicted's lowest-k items.

    Lower values are better.  Exact ties are resolved by original input order.
    """

    x, y = _paired_values(predicted, measured)
    k = _integer(k, "k", 1)
    if k > len(x):
        raise ValueError(f"k={k} exceeds candidate count {len(x)}")
    predicted_top = set(sorted(range(len(x)), key=lambda index: (x[index], index))[:k])
    measured_top = set(sorted(range(len(y)), key=lambda index: (y[index], index))[:k])
    return len(predicted_top & measured_top) / k


def _scored_records(source: Any, name: str, scalar_field: str) -> list[Mapping[str, Any]]:
    data = _read_source(source, name)
    if isinstance(data, Mapping) and data and not any(key in data for key in _RECORD_CONTAINERS):
        expanded = []
        for suite, candidates in data.items():
            if not isinstance(candidates, Mapping):
                raise TypeError(f"{name}.{suite} must map candidate IDs to values or objects")
            for candidate, value in candidates.items():
                record = dict(value) if isinstance(value, Mapping) else {scalar_field: value}
                if "suite" in record and record["suite"] != suite:
                    raise ValueError(f"{name}.{suite}.{candidate}: conflicting suite field")
                if "candidate" in record and str(record["candidate"]) != str(candidate):
                    raise ValueError(f"{name}.{suite}.{candidate}: conflicting candidate field")
                record.setdefault("suite", suite)
                record.setdefault("candidate", candidate)
                expanded.append(record)
        if not expanded:
            raise ValueError(f"{name} must not be empty")
        return expanded
    return _record_list(data, name)


def _prediction_points(predictions: Any, measured: Any) -> dict[str, list[dict[str, Any]]]:
    prediction_records = _scored_records(predictions, "predictions", "predicted")
    measured_records = _scored_records(measured, "measured", "measured")

    def parse(records, value_aliases, role):
        result = {}
        order = []
        for index, record in enumerate(records):
            suite = _text(_pick(record, ("suite", "suite_name", "benchmark"), f"{role}[{index}].suite"), f"{role}[{index}].suite")
            candidate = _identifier(
                _pick(record, ("candidate", "candidate_id", "id"), f"{role}[{index}].candidate"),
                f"{role}[{index}].candidate",
            )
            value = _number(
                _pick(record, value_aliases, f"{role}[{index}] value"),
                f"{role}[{suite}/{candidate}]",
                minimum=0 if role == "measured" else None,
            )
            key = (suite, candidate)
            if key in result:
                raise ValueError(f"Duplicate {role} record for {suite}/{candidate}")
            result[key] = value
            order.append(key)
        return result, order

    predicted, order = parse(
        prediction_records,
        ("predicted", "prediction", "score", "cost", "predicted_latency", "predicted_cost"),
        "predictions",
    )
    actual, measured_order = parse(
        measured_records,
        ("measured", "latency", "confirmed_latency", "measured_latency", "actual_latency"),
        "measured",
    )
    if set(predicted) != set(actual):
        missing_measured = sorted(set(predicted) - set(actual))
        missing_predicted = sorted(set(actual) - set(predicted))
        raise ValueError(
            "Prediction/measured candidate keys differ; "
            f"missing measured={missing_measured}, missing predictions={missing_predicted}"
        )
    if set(measured_order) != set(order):
        raise ValueError("Prediction/measured candidate accounting is inconsistent")
    grouped = {}
    for suite in dict.fromkeys(suite for suite, _ in order):
        keys = [key for key in order if key[0] == suite]
        grouped[suite] = [
            {"suite": suite, "candidate": candidate, "predicted": predicted[(suite, candidate)], "measured": actual[(suite, candidate)]}
            for _, candidate in keys
        ]
    return grouped


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def plot_prediction_rank(
    predictions: Any,
    measured: Any,
    output: str | os.PathLike[str],
    *,
    top_k: int | Sequence[int] = (1, 5, 10),
) -> dict[str, str]:
    """Plot per-suite predicted/actual ranks and rank-quality metrics."""

    _output_base(output)
    grouped = _prediction_points(predictions, measured)
    if isinstance(top_k, int) and not isinstance(top_k, bool):
        top_k = (top_k,)
    if isinstance(top_k, (str, bytes)) or not isinstance(top_k, Sequence) or not top_k:
        raise ValueError("top_k must be a positive integer or a non-empty sequence of integers")
    k_values = tuple(_integer(value, "top_k", 1) for value in top_k)
    if len(set(k_values)) != len(k_values):
        raise ValueError("top_k values must be unique")

    rows = []
    suite_metrics = {}
    for suite, points in grouped.items():
        predicted = [point["predicted"] for point in points]
        actual = [point["measured"] for point in points]
        predicted_ranks = rankdata(predicted)
        actual_ranks = rankdata(actual)
        metrics = {
            "n": len(points),
            "spearman": spearman_rank_correlation(predicted, actual),
            "kendall_tau_b": kendall_tau_b(predicted, actual),
            "top_k_recall": {str(k): top_k_recall(predicted, actual, k) if k <= len(points) else None for k in k_values},
        }
        suite_metrics[suite] = metrics
        for point, predicted_rank, actual_rank in zip(points, predicted_ranks, actual_ranks):
            row = {**point, "predicted_rank": predicted_rank, "actual_rank": actual_rank}
            row.update({"spearman": metrics["spearman"], "kendall_tau_b": metrics["kendall_tau_b"]})
            row.update({f"top_{k}_recall": metrics["top_k_recall"][str(k)] for k in k_values})
            rows.append(row)

    plt = _pyplot()
    with _rc_context(plt):
        suites = list(grouped)
        columns = min(2, len(suites))
        nrows = math.ceil(len(suites) / columns)
        fig, axes = plt.subplots(nrows, columns, figsize=(7 * columns, 6 * nrows), squeeze=False)
        try:
            for suite_index, suite in enumerate(suites):
                ax = axes[suite_index // columns][suite_index % columns]
                points = [row for row in rows if row["suite"] == suite]
                metrics = suite_metrics[suite]
                x = [point["predicted_rank"] for point in points]
                y = [point["actual_rank"] for point in points]
                ax.scatter(x, y, s=55, color="#2563EB", alpha=0.8, edgecolor="white", linewidth=0.5)
                limit = max(2, len(points) + 1)
                ax.plot([0.5, limit - 0.5], [0.5, limit - 0.5], "--", color="#6B7280", label="ideal rank agreement")
                ax.set_xlim(0.5, limit - 0.5)
                ax.set_ylim(0.5, limit - 0.5)
                if len(points) <= 16:
                    for point in points:
                        label = (
                            point["candidate"]
                            .replace("block_DV=", "DV")
                            .replace("stage=", "S")
                            .replace("group=", "G")
                            .replace("fwd:", "")
                            .replace("bwd:", "")
                        )
                        right_edge = point["predicted_rank"] > len(points) * 0.65
                        top_edge = point["actual_rank"] > len(points) * 0.65
                        ax.annotate(
                            label,
                            (point["predicted_rank"], point["actual_rank"]),
                            xytext=(-4 if right_edge else 4, -4 if top_edge else 4),
                            textcoords="offset points",
                            fontsize=6,
                            ha="right" if right_edge else "left",
                            va="top" if top_edge else "bottom",
                            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 0.3},
                            zorder=5,
                        )
                ax.set_xlabel("Predicted rank (1 = lowest prediction)")
                ax.set_ylabel("Actual rank (1 = lowest measured latency)")
                ax.set_title(f"{suite} (n={len(points)})", pad=36)
                ax.grid(alpha=0.22)
                metric_lines = [
                    f"Spearman ρ = {_format_metric(metrics['spearman'])}; "
                    f"Kendall τb = {_format_metric(metrics['kendall_tau_b'])}",
                    "; ".join(
                        f"top-{k} recall = {_format_metric(metrics['top_k_recall'][str(k)])}" for k in k_values
                    ),
                ]
                ax.text(
                    0.0,
                    1.01,
                    "\n".join(metric_lines),
                    transform=ax.transAxes,
                    va="bottom",
                    fontsize=8,
                    linespacing=1.25,
                )
            for index in range(len(suites), nrows * columns):
                axes[index // columns][index % columns].axis("off")
            fig.suptitle("Prediction rank versus confirmed measured rank; dashed line = ideal agreement (lower is better)")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fields = ["suite", "candidate", "predicted", "measured", "predicted_rank", "actual_rank", "spearman", "kendall_tau_b"]
            fields.extend(f"top_{k}_recall" for k in k_values)
            payload = {
                "plot": "prediction_rank",
                "lower_is_better": True,
                "tie_rule": "average ranks; top-k ties use prediction input order",
                "top_k": list(k_values),
                "suites": suite_metrics,
            }
            return _emit(fig, plt, output, rows, fields, payload)
        finally:
            plt.close(fig)


def _method_color(method: str) -> str:
    if method in _METHOD_COLORS:
        return _METHOD_COLORS[method]
    palette = ("#BE123C", "#4F46E5", "#0F766E", "#A16207", "#7E22CE", "#0369A1")
    return palette[sum(ord(character) for character in method) % len(palette)]


def _pareto_indices(points: list[dict[str, Any]], x_axis: str) -> list[int]:
    result = []
    for index, point in enumerate(points):
        dominated = False
        for other_index, other in enumerate(points):
            if index == other_index:
                continue
            no_worse = other[x_axis] <= point[x_axis] and other["confirmed_latency"] <= point["confirmed_latency"]
            strictly_better = other[x_axis] < point[x_axis] or other["confirmed_latency"] < point["confirmed_latency"]
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(index)
    return result


def _quality_data(records: Any, x_axis: str, oracle: Any) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    if x_axis not in {"search_wall_s", "profile_count"}:
        raise ValueError("x_axis must be 'search_wall_s' or 'profile_count'")
    source = _record_list(records, "records")
    points = []
    units = {}
    for index, record in enumerate(source):
        prefix = f"records[{index}]"
        suite = _text(_pick(record, ("suite", "suite_name", "benchmark"), f"{prefix}.suite"), f"{prefix}.suite")
        method = _text(record.get("method"), f"{prefix}.method")
        search_space = _text(_pick(record, ("search_space", "space"), f"{prefix}.search_space"), f"{prefix}.search_space")
        unit = _text(_pick(record, ("latency_unit", "unit"), f"{prefix}.latency_unit"), f"{prefix}.latency_unit")
        if suite in units and units[suite] != unit:
            raise ValueError(f"Suite {suite!r} mixes latency units {units[suite]!r} and {unit!r}")
        units[suite] = unit
        wall = _number(_pick(record, ("search_wall_s", "search_wall_time_s"), f"{prefix}.search_wall_s"), f"{prefix}.search_wall_s", minimum=0)
        profile_count = _integer(_pick(record, ("profile_count", "profiles"), f"{prefix}.profile_count"), f"{prefix}.profile_count")
        latency = _number(
            _pick(record, ("confirmed_latency", "best_latency", "latency"), f"{prefix}.confirmed_latency"),
            f"{prefix}.confirmed_latency",
            positive=True,
        )
        candidate = _pick(record, ("candidate", "candidate_id", "id"), f"{prefix}.candidate", default=None)
        points.append(
            {
                "row_kind": "candidate",
                "record_index": index,
                "suite": suite,
                "method": method,
                "search_space": search_space,
                "candidate": "" if candidate is None else _identifier(candidate, f"{prefix}.candidate"),
                "search_wall_s": wall,
                "profile_count": profile_count,
                "confirmed_latency": latency,
                "latency_unit": unit,
                "x_axis": x_axis,
                "is_pareto": None,
            }
        )

    oracle_points = []
    if oracle is not None:
        oracle_records = _record_list(oracle, "oracle")
        seen_oracle = set()
        for index, record in enumerate(oracle_records):
            prefix = f"oracle[{index}]"
            suite = _text(_pick(record, ("suite", "suite_name", "benchmark"), f"{prefix}.suite"), f"{prefix}.suite")
            search_space = _text(_pick(record, ("search_space", "space"), f"{prefix}.search_space"), f"{prefix}.search_space")
            key = (suite, search_space)
            if key in seen_oracle:
                raise ValueError(f"Duplicate oracle for suite={suite!r}, search_space={search_space!r}")
            seen_oracle.add(key)
            matching = [point for point in points if point["suite"] == suite and point["search_space"] == search_space]
            if not matching:
                raise ValueError(f"Oracle has no same-suite/same-search-space candidates: {key}")
            unit = _text(_pick(record, ("latency_unit", "unit"), f"{prefix}.latency_unit"), f"{prefix}.latency_unit")
            if unit != units[suite]:
                raise ValueError(f"Oracle latency unit {unit!r} does not match suite unit {units[suite]!r}")
            latency = _number(
                _pick(record, ("oracle_latency", "confirmed_latency", "best_latency"), f"{prefix}.oracle_latency"),
                f"{prefix}.oracle_latency",
                positive=True,
            )
            oracle_points.append({"suite": suite, "search_space": search_space, "oracle_latency": latency, "latency_unit": unit})

    groups = {}
    for point in points:
        groups.setdefault(point["suite"], []).append(point)
    for suite, suite_points in groups.items():
        method_spaces = sorted({(point["method"], point["search_space"]) for point in suite_points})
        for method, search_space in method_spaces:
            members = [point for point in suite_points if point["method"] == method and point["search_space"] == search_space]
            frontier = set(_pareto_indices(members, x_axis))
            for index, point in enumerate(members):
                point["is_pareto"] = index in frontier
    for point in oracle_points:
        points.append(
            {
                "row_kind": "oracle",
                "record_index": None,
                "suite": point["suite"],
                "method": "oracle",
                "search_space": point["search_space"],
                "candidate": "",
                "search_wall_s": None,
                "profile_count": None,
                "confirmed_latency": point["oracle_latency"],
                "latency_unit": point["latency_unit"],
                "x_axis": x_axis,
                "is_pareto": None,
            }
        )
    metadata = {"x_axis": x_axis, "oracle": oracle_points, "units": units}
    return metadata, points, groups


def plot_quality_search_pareto(
    records: Any,
    output: str | os.PathLike[str],
    *,
    x_axis: str = "search_wall_s",
    oracle: Any = None,
) -> dict[str, str]:
    """Plot search cost versus confirmed latency with same-space oracle lines."""

    _output_base(output)
    metadata, rows, groups = _quality_data(records, x_axis, oracle)
    plt = _pyplot()
    with _rc_context(plt):
        suites = list(groups)
        columns = min(2, len(suites))
        nrows = math.ceil(len(suites) / columns)
        fig, axes = plt.subplots(nrows, columns, figsize=(8 * columns, 6.5 * nrows), squeeze=False)
        try:
            for suite_index, suite in enumerate(suites):
                ax = axes[suite_index // columns][suite_index % columns]
                suite_points = groups[suite]
                method_spaces = sorted({(point["method"], point["search_space"]) for point in suite_points})
                for method, search_space in method_spaces:
                    members = [point for point in suite_points if point["method"] == method and point["search_space"] == search_space]
                    color = _method_color(method)
                    label = f"{method} [{search_space}]"
                    frontier = sorted((point for point in members if point["is_pareto"]), key=lambda point: (point[x_axis], point["confirmed_latency"], point["record_index"]))
                    dominated = [point for point in members if not point["is_pareto"]]
                    if dominated:
                        ax.scatter([point[x_axis] for point in dominated], [point["confirmed_latency"] for point in dominated], color=color, alpha=0.35, s=38)
                    if frontier:
                        ax.plot(
                            [point[x_axis] for point in frontier],
                            [point["confirmed_latency"] for point in frontier],
                            marker="o",
                            markersize=6,
                            linewidth=1.5,
                            color=color,
                            label=label,
                        )
                for oracle_point in metadata["oracle"]:
                    if oracle_point["suite"] != suite:
                        continue
                    ax.axhline(
                        oracle_point["oracle_latency"],
                        linestyle="--",
                        linewidth=1.5,
                        color="#111827",
                        label=f"oracle [{oracle_point['search_space']}]",
                    )
                x_label = "Search wall time (s)" if x_axis == "search_wall_s" else "Confirmed profile count"
                ax.set_xlabel(x_label)
                ax.set_ylabel(f"Confirmed latency ({metadata['units'][suite]})")
                ax.set_title(suite)
                ax.grid(alpha=0.22)
                ax.legend(fontsize=8, frameon=False)
            for index in range(len(suites), nrows * columns):
                axes[index // columns][index % columns].axis("off")
            fig.suptitle("Quality/search Pareto view: lower is better on both axes; frontiers are per method and search space")
            fig.text(0.01, 0.005, "Dashed oracle lines are drawn only for an explicitly matching suite and search_space; no cross-space oracle is inferred.", fontsize=9)
            fig.tight_layout(rect=(0, 0.035, 1, 0.96))
            fields = (
                "row_kind",
                "record_index",
                "suite",
                "method",
                "search_space",
                "candidate",
                "search_wall_s",
                "profile_count",
                "confirmed_latency",
                "latency_unit",
                "x_axis",
                "is_pareto",
            )
            return _emit(fig, plt, output, rows, fields, {"plot": "quality_search_pareto", **metadata})
        finally:
            plt.close(fig)


def _direction(value: Any, name: str) -> str:
    text = _text(value, name).lower()
    aliases = {"fwd": "fwd", "forward": "fwd", "bwd": "bwd", "backward": "bwd"}
    if text not in aliases:
        raise ValueError(f"{name} must identify fwd or bwd")
    return aliases[text]


def _gdn_data(records: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = _record_list(records, "records")
    rows = []
    seen = set()
    units = set()
    for index, record in enumerate(source):
        prefix = f"records[{index}]"
        suite = _text(_pick(record, ("suite", "suite_name", "benchmark"), f"{prefix}.suite"), f"{prefix}.suite")
        direction_value = _pick(record, ("direction", "pass"), f"{prefix}.direction", default=None)
        tokens = re.findall(r"(?:^|[^a-z0-9])(fwd|forward|bwd|backward)(?:$|[^a-z0-9])", suite.lower())
        if len(tokens) > 1:
            raise ValueError(f"{prefix}.suite contains multiple direction tokens")
        if direction_value is None:
            if not tokens:
                raise ValueError(f"{prefix} needs direction=fwd/bwd or a suite name containing fwd/bwd")
            direction = _direction(tokens[0], f"{prefix}.suite")
        else:
            direction = _direction(direction_value, f"{prefix}.direction")
            if tokens and _direction(tokens[0], f"{prefix}.suite") != direction:
                raise ValueError(f"{prefix} direction conflicts with suite name")
        panel = suite if tokens else f"{suite} {direction}"
        method = _text(record.get("method"), f"{prefix}.method")
        key = (panel, direction, method)
        if key in seen:
            raise ValueError(f"Duplicate GDN acceptance record for {panel}/{method}")
        seen.add(key)
        best_latency = _number(
            _pick(record, ("best_latency", "confirmed_latency"), f"{prefix}.best_latency"),
            f"{prefix}.best_latency",
            positive=True,
        )
        unit = _text(_pick(record, ("latency_unit", "unit"), f"{prefix}.latency_unit"), f"{prefix}.latency_unit")
        units.add(unit)
        compile_count = _integer(
            _pick(record, ("compile_count", "compiles"), f"{prefix}.compile_count"),
            f"{prefix}.compile_count",
        )
        profile_count = _integer(
            _pick(record, ("profile_count", "profiles"), f"{prefix}.profile_count"),
            f"{prefix}.profile_count",
        )
        rows.append(
            {
                "panel": panel,
                "direction": direction,
                "suite": suite,
                "method": method,
                "best_latency": best_latency,
                "latency_unit": unit,
                "compile_count": compile_count,
                "profile_count": profile_count,
            }
        )
    directions = {row["direction"] for row in rows}
    if directions != {"fwd", "bwd"}:
        raise ValueError(f"GDN acceptance requires non-empty fwd and bwd panels; found {sorted(directions)}")
    if len(units) != 1:
        raise ValueError(f"GDN acceptance requires one common latency unit, found {sorted(units)}")
    metadata = {"latency_unit": next(iter(units))}
    return metadata, rows


def plot_gdn_acceptance(records: Any, output: str | os.PathLike[str]) -> dict[str, str]:
    """Plot fwd/bwd best latency together with compile and profile counts."""

    _output_base(output)
    metadata, rows = _gdn_data(records)
    plt = _pyplot()
    with _rc_context(plt):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=False)
        try:
            max_count = max(max(row["compile_count"], row["profile_count"]) for row in rows)
            for ax, direction in zip(axes, ("fwd", "bwd")):
                panel_rows = [row for row in rows if row["direction"] == direction]
                panel_rows.sort(key=lambda row: (_METHOD_ORDER.index(row["method"]) if row["method"] in _METHOD_ORDER else len(_METHOD_ORDER), row["method"]))
                panel_max_latency = max(row["best_latency"] for row in panel_rows)
                x_positions = list(range(len(panel_rows)))
                latencies = [row["best_latency"] for row in panel_rows]
                colors = [_method_color(row["method"]) for row in panel_rows]
                bars = ax.bar(x_positions, latencies, width=0.62, color=colors, alpha=0.78, edgecolor="black", linewidth=0.5)
                ax.bar_label(bars, labels=[f"{value:.6g}" for value in latencies], padding=3, fontsize=8)
                ax.set_xticks(x_positions, [row["method"] for row in panel_rows], rotation=15, ha="right")
                ax.set_title(panel_rows[0]["panel"])
                ax.set_ylabel(f"Best confirmed latency ({metadata['latency_unit']})")
                ax.set_ylim(0, panel_max_latency * 1.22)
                ax.grid(axis="y", alpha=0.22)
                ax.set_axisbelow(True)

                count_axis = ax.twinx()
                compile_counts = [row["compile_count"] for row in panel_rows]
                profile_counts = [row["profile_count"] for row in panel_rows]
                count_axis.plot(x_positions, compile_counts, color="#111827", marker="o", linewidth=1.6, label="compile count")
                count_axis.plot(x_positions, profile_counts, color="#DC2626", marker="s", linewidth=1.6, label="profile count")
                count_axis.set_ylim(0, max(1, max_count) * 1.3 + 0.5)
                count_axis.set_ylabel("Attempts / confirmed profiles (count)")
                for x, compile_count, profile_count in zip(x_positions, compile_counts, profile_counts):
                    count_axis.annotate(str(compile_count), (x, compile_count), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
                    count_axis.annotate(str(profile_count), (x, profile_count), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=8, color="#DC2626")
            methods = list(dict.fromkeys(row["method"] for row in rows))
            methods.sort(key=lambda method: (_METHOD_ORDER.index(method) if method in _METHOD_ORDER else len(_METHOD_ORDER), method))
            method_handles = [plt.Rectangle((0, 0), 1, 1, color=_method_color(method), alpha=0.78, label=method) for method in methods]
            metric_handles = [
                plt.Line2D([0], [0], color="#111827", marker="o", label="compile count (right axis)"),
                plt.Line2D([0], [0], color="#DC2626", marker="s", label="profile count (right axis)"),
            ]
            fig.legend(
                handles=method_handles + metric_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.94),
                ncol=4,
                frameon=False,
                columnspacing=1.5,
                handlelength=1.8,
            )
            fig.suptitle("GDN acceptance: bars = best confirmed latency (left axis); lines = exact input counts (right axis)", y=0.995)
            fig.tight_layout(rect=(0, 0, 1, 0.82))
            fields = ("panel", "direction", "suite", "method", "best_latency", "latency_unit", "compile_count", "profile_count")
            return _emit(fig, plt, output, rows, fields, {"plot": "gdn_acceptance", **metadata})
        finally:
            plt.close(fig)
