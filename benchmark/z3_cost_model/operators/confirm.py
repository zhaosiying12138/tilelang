"""Independent candidate-median winner confirmation for classic operators.

The three cold search repetitions are audited and every arm's winner is frozen
before Torch, a candidate factory, or a GPU telemetry query is invoked.  Each
distinct non-guided winner is compared with ``z3-cm-topk`` for seven ABBA rounds.
A round measures 50 warmups followed by 200 individual L2-flushed CUDA-event
samples.  Identical configurations use identity, never fabricated paired data.

This module is CPU-safe on import.  Production compilation and CUDA execution
are explicit in :func:`run_confirmation`; dependency injection is for CPU tests.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

_OPERATORS_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = _OPERATORS_ROOT.parents[2]
if __package__:
    from .autotune import (
        ARMS,
        DEFAULT_REPETITIONS,
        DEFAULT_TOP_K,
        SEED,
        SUITE_NAMES,
        OperatorFixture,
        get_suite,
        make_candidate_factory,
    )
    from ..gdn.autotune import _tool_cache_environment
else:
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from benchmark.z3_cost_model.operators.autotune import (
        ARMS,
        DEFAULT_REPETITIONS,
        DEFAULT_TOP_K,
        SEED,
        SUITE_NAMES,
        OperatorFixture,
        get_suite,
        make_candidate_factory,
    )
    from benchmark.z3_cost_model.gdn.autotune import _tool_cache_environment


GUIDED_ARM = "z3-cm-topk"
ORACLE_ARM = "z3-legal-exhaustive"
STOCK_ARM = "stock-raw"
CONTROL_ARMS = ("budget-raw", "budget-random")
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "evidence/G6/classic-operator-confirmation"
GPU_QUERY = "timestamp,name,driver_version,power.draw,temperature.gpu,clocks.sm,clocks.mem,memory.used"
QUALITY_RULE = (
    "report-only: guided/legal and guided/budget ratios, paired differences and ranges; no GDN-specific performance gate is imposed"
)
QUALITY_RULE_SOURCE = "classic quality is descriptive; execution and correctness are separate from performance comparisons"
_TERMINAL_TRIALS = {"ok", "compile_error", "benchmark_error", "timeout"}
_CACHE_ENVIRONMENT_KEYS = (
    "TILELANG_CACHE_DIR",
    "TILELANG_DISABLE_CACHE",
    "TILELANG_AUTO_TUNING_DISABLE_CACHE",
    "CUDA_CACHE_PATH",
    "CUDA_CACHE_DISABLE",
    "TORCH_EXTENSIONS_DIR",
    "TRITON_CACHE_DIR",
    "XDG_CACHE_HOME",
    "TMPDIR",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(config: dict[str, Any]) -> str:
    if not isinstance(config, dict):
        raise ValueError(f"config must be an object, got {config!r}")
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_analysis(path: str | Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("runs"), list):
        raise ValueError("Classic cold analysis must contain a runs list")
    return value


def _positive_latency(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"Invalid trial latency in {context}: {value!r}; expected a finite positive number")
    return float(value)


def _validated_config(spec, config: Any, *, context: str) -> tuple[dict[str, Any], str]:
    if not isinstance(config, dict):
        raise ValueError(f"Missing or invalid config in {context}: {config!r}")
    if config.get("__pass_configs__") is not None:
        raise ValueError(f"Per-config pass_configs are not allowed in frozen classic suites: {context}")
    normalized = {key: value for key, value in config.items() if key != "__pass_configs__"}
    expected = set(spec.module.CONFIG_KEYS)
    actual = set(normalized)
    if actual != expected:
        raise ValueError(
            f"Config keys in {context} do not match {spec.name}: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    try:
        canonical = _canonical(normalized)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Non-JSON config in {context}: {error}") from error
    return normalized, canonical


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Missing cold-trial events: {path}")
    events = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
            if not isinstance(event, dict):
                raise ValueError(f"Expected an event object in {path}:{line_number}")
            events.append(event)
    if not events:
        raise ValueError(f"Cold-trial events are empty: {path}")
    return events


def _verify_summary_metadata(summary: dict[str, Any], row: dict[str, Any], spec, arm: str, repetition: int, path: Path) -> None:
    if summary.get("status") != "ok":
        raise ValueError(f"Cold summary is not ok for {spec.name}/{arm}/rep-{repetition}: {path}")
    metadata = summary.get("operator_run")
    experiment = summary.get("experiment")
    if not isinstance(metadata, dict) or not isinstance(experiment, dict):
        raise ValueError(f"Cold summary lacks operator_run/experiment metadata: {path}")
    expected = {
        "operator_run.suite": (metadata.get("suite"), spec.name),
        "operator_run.cache_track": (metadata.get("cache_track"), "cold"),
        "operator_run.top_k": (metadata.get("top_k"), DEFAULT_TOP_K),
        "experiment.arm": (experiment.get("arm"), arm),
        "experiment.repetition": (experiment.get("repetition"), repetition),
    }
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ValueError(f"Mixed or inconsistent cold summary {path}: {name}={actual!r}, expected {wanted!r}")
    event_metadata = summary.get("metadata", {})
    if isinstance(event_metadata, dict) and event_metadata.get("suite", spec.name) != spec.name:
        raise ValueError(f"Mixed suite metadata in cold summary {path}")
    if row.get("path") is not None and Path(row["path"]).resolve(strict=False) != path.resolve(strict=False):
        raise ValueError(f"Summary path changed while loading cold evidence: {path}")


def _audit_cold_run(row: dict[str, Any], spec, arm: str, repetition: int) -> list[dict[str, Any]]:
    if row.get("performance_evidence") is not True:
        raise ValueError(f"Cold run lacks complete performance evidence for {spec.name}/{arm}/rep-{repetition}")
    if row.get("cache_hit"):
        raise ValueError(f"Cache-hit run cannot be cold confirmation evidence: {spec.name}/{arm}/rep-{repetition}")
    summary_value = row.get("path")
    if not summary_value:
        raise ValueError(f"Missing summary path for {spec.name}/{arm}/rep-{repetition}")
    summary_path = Path(summary_value)
    if not summary_path.is_file():
        raise ValueError(f"Missing cold summary for {spec.name}/{arm}/rep-{repetition}: {summary_path}")
    summary = _read_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"Cold summary must be an object: {summary_path}")
    _verify_summary_metadata(summary, row, spec, arm, repetition, summary_path)
    events_value = row.get("events_path") or summary_path.parent / "events.jsonl"
    events_path = Path(events_value)
    events = _load_events(events_path)

    run_starts = [event for event in events if event.get("event") == "run_start"]
    run_ends = [event for event in events if event.get("event") == "run_end"]
    generation_ends = [event for event in events if event.get("event") == "config_generation_end"]
    if len(run_starts) != 1 or len(run_ends) != 1 or len(generation_ends) != 1:
        raise ValueError(
            f"Incomplete cold events for {spec.name}/{arm}/rep-{repetition}: "
            f"run_start={len(run_starts)}, config_generation_end={len(generation_ends)}, run_end={len(run_ends)}"
        )
    start = run_starts[0]
    metadata = start.get("metadata")
    if start.get("arm") != arm or not isinstance(metadata, dict):
        raise ValueError(f"Mixed arm or missing run_start metadata in {events_path}")
    expected_metadata = {"suite": spec.name, "cache_track": "cold", "top_k": DEFAULT_TOP_K}
    for key, wanted in expected_metadata.items():
        if metadata.get(key) != wanted:
            raise ValueError(f"Mixed cold run_start metadata in {events_path}: {key}={metadata.get(key)!r}, expected {wanted!r}")
    if "repetition" in metadata and metadata["repetition"] != repetition:
        raise ValueError(f"Mixed repetition in {events_path}: {metadata['repetition']!r} != {repetition}")
    if run_ends[0].get("status") != "ok" or generation_ends[0].get("status") != "ok":
        raise ValueError(f"Cold terminal events are not ok in {events_path}")

    selected = []
    trials = []
    for event in events:
        kind = event.get("event")
        if kind == "config_generated":
            _config, canonical = _validated_config(spec, event.get("config"), context=str(events_path))
            if event.get("phase") == "selected":
                selected.append(canonical)
        elif kind == "score" and arm == GUIDED_ARM:
            _validated_config(spec, event.get("config"), context=str(events_path))
        elif kind == "trial":
            _config, canonical = _validated_config(spec, event.get("config"), context=str(events_path))
            status = event.get("status")
            if status not in _TERMINAL_TRIALS:
                raise ValueError(f"Non-terminal or unknown trial status {status!r} in {events_path}")
            latency = event.get("latency_ms")
            if status == "ok":
                latency = _positive_latency(latency, context=str(events_path))
            elif latency is not None:
                _positive_latency(latency, context=str(events_path))
            trials.append((canonical, status, latency))
    if not selected or Counter(selected) != Counter(canonical for canonical, _status, _latency in trials):
        raise ValueError(f"Selected configs and trial events are not one-to-one in {events_path}")
    if any(count != 1 for count in Counter(selected).values()):
        raise ValueError(f"Duplicate selected config in {events_path}")
    return events


def candidate_median_winners(
    analysis: dict[str, Any],
    suite: str,
    *,
    expected_repetitions=DEFAULT_REPETITIONS,
) -> dict[str, dict[str, Any]]:
    """Freeze all five winners solely from successful cold-search trial events."""

    spec = get_suite(suite)
    if not isinstance(analysis, dict) or not isinstance(analysis.get("runs"), list):
        raise ValueError("Classic cold analysis must contain a runs list")
    expected = tuple(expected_repetitions)
    if len(expected) != 3 or len(set(expected)) != 3 or any(isinstance(value, bool) or not isinstance(value, int) for value in expected):
        raise ValueError("Exactly three distinct integer cold repetitions are required")
    grouped = {arm: {} for arm in ARMS}
    evidence = {arm: [] for arm in ARMS}
    for arm in ARMS:
        rows = [
            row
            for row in analysis["runs"]
            if isinstance(row, dict)
            and row.get("suite") == spec.name
            and row.get("arm") == arm
            and row.get("cache_track") == "cold"
            and row.get("top_k") == DEFAULT_TOP_K
        ]
        by_repetition = {}
        for row in rows:
            repetition = row.get("repetition")
            if isinstance(repetition, bool) or not isinstance(repetition, int):
                raise ValueError(f"Invalid cold repetition for {spec.name}/{arm}: {repetition!r}")
            if repetition in by_repetition:
                raise ValueError(f"Duplicate cold repetition for {spec.name}/{arm}/rep-{repetition}")
            by_repetition[repetition] = row
        if set(by_repetition) != set(expected):
            raise ValueError(f"Expected cold repetitions {list(expected)} for {spec.name}/{arm}, got {sorted(by_repetition)}")
        for repetition in expected:
            row = by_repetition[repetition]
            events = _audit_cold_run(row, spec, arm, repetition)
            evidence[arm].append(row)
            observed_successes = set()
            for event in events:
                if event.get("event") != "trial" or event.get("status") != "ok":
                    continue
                config, canonical = _validated_config(spec, event.get("config"), context=row.get("events_path") or row["path"])
                latency = _positive_latency(event.get("latency_ms"), context=row.get("events_path") or row["path"])
                if canonical in observed_successes:
                    raise ValueError(f"Duplicate successful trial for one config in {spec.name}/{arm}/rep-{repetition}")
                observed_successes.add(canonical)
                candidate = grouped[arm].setdefault(
                    canonical,
                    {"config": config, "latencies": [], "repetitions": set()},
                )
                candidate["latencies"].append(latency)
                candidate["repetitions"].add(repetition)

    winners = {}
    for arm in ARMS:
        candidates = grouped[arm]
        if not candidates:
            raise ValueError(f"No successful cold trials for {spec.name}/{arm}")
        ranked = sorted(candidates.items(), key=lambda item: (statistics.median(item[1]["latencies"]), item[0]))
        canonical, candidate = ranked[0]
        arm_evidence = evidence[arm]
        winners[arm] = {
            "config": json.loads(canonical),
            "config_id": canonical,
            "trial_count": len(candidate["latencies"]),
            "measurement_repetitions": len(candidate["repetitions"]),
            "observed_repetitions": sorted(candidate["repetitions"]),
            "median_latency_ms": statistics.median(candidate["latencies"]),
            "candidate_count": len(candidates),
            "cold_repetitions": list(expected),
            "cold_summary_paths": [row["path"] for row in arm_evidence],
            "performance_evidence_repetitions": len(arm_evidence),
            "oracle_complete_repetitions": sum(row.get("oracle_complete") is True for row in arm_evidence),
            "tie_break": "lowest candidate median, then canonical JSON",
            "selection_rule": "candidate-median",
            "latency_scope": "search_observed",
        }
    return winners


def confirmation_plan(analysis: dict[str, Any], suite: str) -> dict[str, Any]:
    """Deduplicate winner configs while retaining every arm/reference relation."""

    spec = get_suite(suite)
    winners = candidate_median_winners(analysis, spec.name)
    groups = {}
    for arm in ARMS:
        canonical = _canonical(winners[arm]["config"])
        groups.setdefault(canonical, []).append(arm)
    guided = _canonical(winners[GUIDED_ARM]["config"])
    candidates = [{"config": winners[GUIDED_ARM]["config"], "arms": [GUIDED_ARM]}]
    comparisons = []
    for canonical, arms in groups.items():
        identical = canonical == guided
        relations = [
            relation
            for relation, members in (
                ("same_space_exhaustive", {ORACLE_ARM}),
                ("equal_budget_control", set(CONTROL_ARMS)),
                ("stock_informational", {STOCK_ARM}),
            )
            if any(arm in members for arm in arms)
        ]
        comparisons.append(
            {
                "alternative_arms": [] if identical else arms,
                "identical_arms": arms if identical else [],
                "config": json.loads(canonical),
                "config_id": canonical,
                "identical": identical,
                "relations": relations,
                "reference_semantics": {
                    "stock-raw": "full raw-space arm winner; not a Z3 legal oracle",
                    "z3-legal-exhaustive": "exhaustive Z3 legal-space arm; legal oracle only when cold oracle evidence is complete",
                },
            }
        )
        if not identical:
            candidates.append({"config": json.loads(canonical), "arms": arms})
    legal_complete = winners[ORACLE_ARM]["oracle_complete_repetitions"] == len(DEFAULT_REPETITIONS)
    return {
        "schema": "tilelang.classic-operators.confirmation-plan/v1",
        "suite": spec.name,
        "guided_arm": GUIDED_ARM,
        "guided_top_k": DEFAULT_TOP_K,
        "winners": winners,
        "candidates": candidates,
        "comparisons": comparisons,
        "selection_rule": "candidate-median",
        "selection_rule_detail": (
            "lowest per-candidate median of valid successful trials across the three actual cold repetitions; "
            "canonical JSON tie-break; confirmation measurements are never selection inputs"
        ),
        "legal_oracle_complete": legal_complete,
        "stock_raw_is_legal_oracle": False,
    }


def abba_order(round_index: int) -> tuple[int, int]:
    """Return guided/alternative positions in AB, BA, BA, AB groups."""

    if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
        raise ValueError("round_index must be a non-negative integer")
    return (0, 1) if round_index % 4 in (0, 3) else (1, 0)


def measurement_schedule(plan: dict[str, Any], rounds: int) -> list[dict[str, Any]]:
    """Expand real ABBA work; an all-identical plan has one seven-round series."""

    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds <= 0:
        raise ValueError("rounds must be a positive integer")
    guided = _canonical(plan["winners"][GUIDED_ARM]["config"])
    schedule = []
    pairs = [(index, row) for index, row in enumerate(plan["comparisons"]) if not row["identical"]]
    if not pairs:
        for round_index in range(rounds):
            schedule.append(
                {
                    "pair_index": -1,
                    "round": round_index,
                    "config_id": guided,
                    "role": "identical",
                    "position": 0,
                }
            )
        return schedule
    for pair_index, comparison in pairs:
        alternative = _canonical(comparison["config"])
        for round_index in range(rounds):
            for position, role in enumerate(abba_order(round_index)):
                schedule.append(
                    {
                        "pair_index": pair_index,
                        "round": round_index,
                        "config_id": guided if role == 0 else alternative,
                        "role": "guided" if role == 0 else "alternative",
                        "position": position,
                    }
                )
    return schedule


def _numeric_stats(values, *, positive: bool, unit: str = "ms") -> dict[str, Any]:
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) or (positive and value <= 0) for value in values):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"statistics require a non-empty sequence of {qualifier} values")
    return {
        "count": len(values),
        f"median_{unit}": statistics.median(values),
        f"mean_{unit}": statistics.fmean(values),
        f"min_{unit}": min(values),
        f"max_{unit}": max(values),
    }


def _sample_stats(values) -> dict[str, Any]:
    return _numeric_stats(values, positive=True)


def _round_stats(records) -> dict[str, Any]:
    by_round = {}
    for record in records:
        key = (record["pair_index"], record["round"])
        by_round.setdefault(key, []).append(record["median_ms"])
    if not by_round or any(len(values) != 1 for values in by_round.values()):
        raise ValueError("each candidate must have exactly one measurement per comparison round")
    medians = [values[0] for _key, values in sorted(by_round.items())]
    rounds = [key[1] for key, _values in sorted(by_round.items())]
    return {**_sample_stats(medians), "rounds": rounds, "round_medians_ms": medians}


def evaluate_comparisons(plan: dict[str, Any], measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute pair-local ratios/ranges and identity-only equal comparisons."""

    guided = _canonical(plan["winners"][GUIDED_ARM]["config"])
    known_configs = {_canonical(entry["config"]) for entry in plan["candidates"]}
    real_pairs = {index for index, row in enumerate(plan["comparisons"]) if not row["identical"]}
    for record in measurements:
        config_id = record.get("config_id")
        if config_id not in known_configs:
            raise ValueError(f"Measurement references an unknown config: {config_id}")
        pair_index = record.get("pair_index")
        if real_pairs:
            if pair_index not in real_pairs:
                raise ValueError(f"Measurement references an invalid/identity pair: {pair_index}")
            allowed = {guided, _canonical(plan["comparisons"][pair_index]["config"])}
            if config_id not in allowed:
                raise ValueError(f"Measurement config {config_id} does not belong to pair {pair_index}")
        elif pair_index != -1 or config_id != guided:
            raise ValueError("An all-identical plan accepts only the single guided identity series")
        _positive_latency(record.get("median_ms"), context="confirmation measurement")
    results = []
    for pair_index, comparison in enumerate(plan["comparisons"]):
        alternative = _canonical(comparison["config"])
        if comparison["identical"]:
            records = [row for row in measurements if row["config_id"] == guided]
            guided_stats = _round_stats(records)
            alternative_stats = copy.deepcopy(guided_stats)
            ratio = 1.0
            paired_differences = [0.0 for _value in guided_stats["round_medians_ms"]]
            evidence = "same canonical config; identity applied without a second sample set"
        else:
            pair_records = [row for row in measurements if row["pair_index"] == pair_index]
            guided_stats = _round_stats([row for row in pair_records if row["config_id"] == guided])
            alternative_stats = _round_stats([row for row in pair_records if row["config_id"] == alternative])
            if guided_stats["rounds"] != alternative_stats["rounds"]:
                raise ValueError(f"Unbalanced or mismatched ABBA rounds for pair {pair_index}")
            ratio = guided_stats["median_ms"] / alternative_stats["median_ms"]
            paired_differences = [
                left - right for left, right in zip(guided_stats["round_medians_ms"], alternative_stats["round_medians_ms"], strict=True)
            ]
            evidence = "pair-local ABBA round medians"
        requirements = {}
        if "same_space_exhaustive" in comparison["relations"]:
            requirements["within_2_percent_of_exhaustive"] = ratio <= 1.02
        if "equal_budget_control" in comparison["relations"]:
            requirements["not_slower_than_equal_budget_control"] = ratio <= 1.0
        difference_stats = _numeric_stats(paired_differences, positive=False)
        results.append(
            {
                **comparison,
                "pair_index": pair_index,
                "guided_median_ms": guided_stats["median_ms"],
                "alternative_median_ms": alternative_stats["median_ms"],
                "guided_round_statistics": guided_stats,
                "alternative_round_statistics": alternative_stats,
                "quality_ratio_guided_over_alternative": ratio,
                "paired_round_differences_ms": paired_differences,
                "paired_difference_statistics": difference_stats,
                "paired_difference_range_ms": [difference_stats["min_ms"], difference_stats["max_ms"]],
                "measurement_evidence": evidence,
                "legal_oracle_complete": plan["legal_oracle_complete"],
                "requirements": requirements,
                "passed": all(requirements.values())
                and (plan["legal_oracle_complete"] or "same_space_exhaustive" not in comparison["relations"]),
                "acceptance_rule_source": QUALITY_RULE_SOURCE,
            }
        )
    return results


def _prepare_run_directory(output_dir: str | Path, suite: str) -> Path:
    spec = get_suite(suite)
    unresolved = Path(output_dir).expanduser() / spec.name
    if unresolved.is_symlink():
        raise ValueError(f"Refusing a symlink confirmation path: {unresolved}")
    run_dir = unresolved.resolve(strict=False)
    shared_cache = (Path.home() / ".tilelang").resolve(strict=False)
    if run_dir == shared_cache or shared_cache in run_dir.parents:
        raise ValueError(f"Refusing to use the shared TileLang cache as a confirmation path: {run_dir}")
    if run_dir.exists():
        if not run_dir.is_dir() or any(run_dir.iterdir()):
            raise ValueError(f"Refusing a non-empty or non-directory confirmation path: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


@contextmanager
def _confirmation_cache(run_dir: str | Path):
    """Own every tool cache below one empty run-local cache root and restore env."""

    run_dir = Path(run_dir)
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError(f"Confirmation cache requires a real run directory: {run_dir}")
    cache = run_dir / "cache"
    if cache.is_symlink() or (cache.exists() and (not cache.is_dir() or any(cache.iterdir()))):
        raise ValueError(f"Refusing a non-empty, non-directory, or symlink cache path: {cache}")
    names = ("TILELANG_CACHE_DIR", "TILELANG_DISABLE_CACHE", "TILELANG_AUTO_TUNING_DISABLE_CACHE")
    previous = {name: os.environ.get(name) for name in names}
    tilelang_cache = cache / "tilelang"
    tilelang_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TILELANG_CACHE_DIR"] = str(tilelang_cache)
    os.environ["TILELANG_DISABLE_CACHE"] = "0"
    os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] = "0"
    try:
        with _tool_cache_environment(run_dir, "cold"):
            yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _gpu_telemetry() -> str:
    command = ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
    return result.stdout.strip()


def _measure_kernel(candidate, inputs, cache, warmup: int, repeat: int, *, torch_module=None):
    """Measure only candidate execution between events; flush before each start."""

    torch = importlib.import_module("torch") if torch_module is None else torch_module
    for _index in range(warmup):
        candidate(*inputs)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _index in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _index in range(repeat)]
    for index in range(repeat):
        cache.zero_()
        starts[index].record()
        candidate(*inputs)
        ends[index].record()
    torch.cuda.synchronize()
    samples = [float(start.elapsed_time(end)) for start, end in zip(starts, ends, strict=True)]
    return _sample_stats(samples), samples


def _json_report(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, dict):
        raise TypeError(f"validate must return ValidationReport or a report dictionary, got {type(value).__name__}")
    report = copy.deepcopy(value)
    nonfinite = []
    for key, item in list(report.items()):
        if isinstance(item, float) and not math.isfinite(item):
            report[key] = None
            nonfinite.append(key)
    if nonfinite:
        report["nonfinite_fields"] = sorted(nonfinite)
    return report


def _validate_candidate(candidate, fixture, inputs, config_id: str, phase: str) -> dict[str, Any]:
    """Run a fresh module FP32 reference and strict validate outside timing."""

    try:
        reference = fixture.reference_program(*inputs)
        actual = candidate(*inputs)
        report = _json_report(fixture.manual_check_prog([actual], [reference]))
        passed = (
            report.get("passed") is True
            and report.get("mismatched_ratio") == 0
            and report.get("max_mismatched_ratio") == 0
            and not report.get("nonfinite_fields")
        )
        return {
            **report,
            "phase": phase,
            "status": "ok" if passed else "error",
            "passed": passed,
            "config_id": config_id,
            "reference": "frozen operator module independent FP32 reference",
            "validation": "frozen module.validate; max_mismatched_ratio=0",
        }
    except Exception as error:
        payload = {}
        report = getattr(error, "report", None)
        if report is not None:
            payload = _json_report(report)
        return {
            **payload,
            "phase": phase,
            "status": "error",
            "passed": False,
            "config_id": config_id,
            "error": f"{type(error).__name__}: {error}",
            "reference": "frozen operator module independent FP32 reference",
            "validation": "frozen module.validate; max_mismatched_ratio=0",
        }


def _compile_candidates(
    spec,
    entries,
    *,
    fixture,
    target,
    pass_configs,
    kernel_factory=None,
    candidate_factory_builder=None,
):
    """Compile every distinct config through the real explicit-signature factory."""

    builder = make_candidate_factory if candidate_factory_builder is None else candidate_factory_builder
    factory = builder(
        spec,
        fixture=fixture,
        target=target,
        pass_configs=pass_configs,
        compile=True,
        kernel_factory=kernel_factory,
    )
    candidates = {}
    for entry in entries:
        config = entry["config"]
        canonical = _canonical(config)
        if canonical in candidates:
            raise ValueError(f"Duplicate candidate in confirmation plan: {canonical}")
        candidate = factory(**config)
        if not callable(candidate):
            raise TypeError(f"Compiled candidate is not callable for {canonical}")
        candidates[canonical] = candidate
    return candidates


def _source_evidence() -> dict[str, Any]:
    paths = [
        Path(__file__),
        Path(__file__).with_name("analyze_confirmation.py"),
        Path(__file__).with_name("autotune.py"),
        Path(__file__).with_name("_common.py"),
        Path(__file__).with_name("gemm.py"),
        Path(__file__).with_name("convolution.py"),
        Path(__file__).with_name("flash_attention.py"),
        REPOSITORY_ROOT / "manifest/experiment_plan.json",
        REPOSITORY_ROOT / "manifest/gdn-selection-v2-plan.json",
    ]
    files = {}
    for path in paths:
        resolved = path.resolve()
        files[str(resolved.relative_to(REPOSITORY_ROOT))] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()
    return {"git_commit": commit, "file_sha256": files}


def _inventory(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"before.json", "after.json"}:
            continue
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return entries


def _append_validation(summary: dict[str, Any], summary_path: Path, report: dict[str, Any]) -> None:
    summary["validations"].append(report)
    _write_json(summary_path, summary)
    if not report["passed"]:
        raise RuntimeError(
            f"Correctness failed for {report['config_id']} during {report['phase']} validation: {report.get('error', 'strict mismatch')}"
        )


def run_confirmation(
    suite: str,
    analysis_path: str | Path,
    output_dir: str | Path,
    *,
    rounds: int = 7,
    warmup: int = 50,
    repeat: int = 200,
    seed: int = SEED,
    device: Any = "cuda",
    target: Any = "auto",
    pass_configs: dict[str, Any] | None = None,
    fixture_factory=OperatorFixture,
    kernel_factory=None,
    candidate_factory_builder=None,
    measure_fn=None,
    telemetry_fn=None,
    torch_module=None,
) -> dict[str, Any]:
    """Run one suite independently; persist failures and never mark partial data ok."""

    spec = get_suite(suite)
    for value, name in ((rounds, "rounds"), (warmup, "warmup"), (repeat, "repeat")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    run_dir = _prepare_run_directory(output_dir, spec.name)
    summary_path = run_dir / "summary.json"
    samples_path = run_dir / "samples.jsonl"
    summary = {
        "schema": "tilelang.classic-operators.confirmation/v1",
        "status": "running",
        "suite": spec.name,
        "operator": spec.operator,
        "shape": spec.shape.to_dict(),
        "rounds": rounds,
        "warmup": warmup,
        "repeat": repeat,
        "seed": seed,
        "device": str(device),
        "target": target,
        "pass_configs": pass_configs,
        "pid": os.getpid(),
        "analysis_path": str(Path(analysis_path).expanduser().resolve(strict=False)),
        "plan": None,
        "source": None,
        "timing": f"median of pair-local per-round medians; each measurement has {repeat} L2-flushed CUDA-event samples after {warmup} warmups",
        "reference_and_compilation_in_timed_region": False,
        "reference_validation_in_timed_region": False,
        "measurements": [],
        "validations": [],
        "telemetry": {"before": None, "after": None, "query": GPU_QUERY},
        "cache": {"root": str(run_dir / "cache"), "run_owned": True, "environment_keys": list(_CACHE_ENVIRONMENT_KEYS)},
    }
    before_path = run_dir / "before.json"
    _write_json(summary_path, summary)
    _write_json(before_path, {"utc": _utc_now(), "pid": os.getpid(), "entries": _inventory(run_dir), "gpu_telemetry": None})
    telemetry = _gpu_telemetry if telemetry_fn is None else telemetry_fn
    try:
        # All cold-data and config failures occur before telemetry, Torch import,
        # fixture construction, candidate compilation, or CUDA initialization.
        summary["source"] = _source_evidence()
        summary["plan"] = confirmation_plan(_read_analysis(analysis_path), spec.name)
        _write_json(summary_path, summary)
        summary["telemetry"]["before"] = telemetry()
        _write_json(
            before_path,
            {
                "utc": _utc_now(),
                "pid": os.getpid(),
                "source": summary["source"],
                "entries": _inventory(run_dir),
                "gpu_telemetry": summary["telemetry"]["before"],
            },
        )
        with _confirmation_cache(run_dir):
            torch = torch_module
            device_context = nullcontext()
            cache = None
            if measure_fn is None:
                torch = importlib.import_module("torch") if torch is None else torch
                selected_device = torch.device(device)
                if selected_device.type != "cuda":
                    raise ValueError(f"Production confirmation requires a CUDA device, got {device!r}")
                device_context = torch.cuda.device(selected_device)
            with device_context:
                fixture = fixture_factory(spec, seed=seed, device=device)
                inputs = [fixture.inputs[name] for name in fixture.input_names]
                candidates = _compile_candidates(
                    spec,
                    summary["plan"]["candidates"],
                    fixture=fixture,
                    target=target,
                    pass_configs=pass_configs,
                    kernel_factory=kernel_factory,
                    candidate_factory_builder=candidate_factory_builder,
                )
                summary["compiled_candidates"] = [
                    {"config_id": canonical, "kernel_type": f"{type(candidate).__module__}.{type(candidate).__qualname__}"}
                    for canonical, candidate in candidates.items()
                ]
                _write_json(summary_path, summary)
                for canonical, candidate in candidates.items():
                    _append_validation(summary, summary_path, _validate_candidate(candidate, fixture, inputs, canonical, "before"))
                measure = _measure_kernel if measure_fn is None else measure_fn
                if measure_fn is None:
                    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device=inputs[0].device)
                with samples_path.open("x", encoding="utf-8", newline="\n") as sample_stream:
                    for record_index, scheduled in enumerate(measurement_schedule(summary["plan"], rounds)):
                        before_telemetry = telemetry()
                        stats, samples = measure(
                            candidates[scheduled["config_id"]],
                            inputs,
                            cache,
                            warmup,
                            repeat,
                            torch_module=torch,
                        )
                        after_telemetry = telemetry()
                        raw = {
                            "record_index": record_index,
                            **scheduled,
                            **stats,
                            "samples_ms": samples,
                            "telemetry_before": before_telemetry,
                            "telemetry_after": after_telemetry,
                        }
                        sample_stream.write(json.dumps(raw, allow_nan=False, separators=(",", ":")) + "\n")
                        sample_stream.flush()
                        compact = {
                            key: value for key, value in raw.items() if key not in ("samples_ms", "telemetry_before", "telemetry_after")
                        }
                        compact["samples"] = f"samples.jsonl#record_index={record_index}"
                        compact["telemetry_before"] = before_telemetry
                        compact["telemetry_after"] = after_telemetry
                        summary["measurements"].append(compact)
                        _write_json(summary_path, summary)
                for canonical, candidate in candidates.items():
                    _append_validation(summary, summary_path, _validate_candidate(candidate, fixture, inputs, canonical, "after"))
        summary["comparisons"] = evaluate_comparisons(summary["plan"], summary["measurements"])
        summary["acceptance"] = {
            "correctness_passed": True,
            "performance_report_only": True,
            "passed": True,
            "passed_scope": "execution_and_correctness_only",
            "rule": QUALITY_RULE,
            "rule_source": QUALITY_RULE_SOURCE,
            "legal_oracle_complete": summary["plan"]["legal_oracle_complete"],
            "stock_raw_semantics": "full raw-space winner/reference, never relabeled as the Z3 legal oracle",
        }
        summary["telemetry"]["after"] = telemetry()
        summary["status"] = "ok"
    except BaseException as error:
        summary["status"] = "error"
        summary["error"] = f"{type(error).__name__}: {error}"
        _write_json(summary_path, summary)
        raise
    finally:
        _write_json(summary_path, summary)
        _write_json(
            run_dir / "after.json",
            {
                "utc": _utc_now(),
                "pid": os.getpid(),
                "status": summary["status"],
                "source": summary["source"],
                "entries": _inventory(run_dir),
                "gpu_telemetry": summary["telemetry"]["after"],
            },
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITE_NAMES, required=True)
    parser.add_argument("--analysis", "--cold-analysis", dest="analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target", default="auto")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_confirmation(
        args.suite,
        args.analysis,
        args.output_dir,
        rounds=args.rounds,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        device=args.device,
        target=args.target,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "suite": summary["suite"],
                "acceptance": summary["acceptance"],
                "summary": str(Path(args.output_dir).expanduser().resolve(strict=False) / summary["suite"] / "summary.json"),
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
