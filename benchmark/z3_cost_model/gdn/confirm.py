"""Independent ABBA confirmation of frozen GDN per-arm winners.

The search is finished before this module runs.  Modal winners are selected from
cold repetitions, with canonical JSON as the deterministic tie-break.  Every
distinct guided/alternative pair is measured for seven rounds using the frozen
50 warmups and 200 L2-flushed CUDA-event samples.  Correctness references and
compilation are outside the timed region.  No result is deleted when a quality
check fails.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

try:
    from .autotune import (
        ARMS,
        BWD_OUTPUTS,
        DEFAULT_TOP_K,
        FWD_OUTPUTS,
        GDNCandidateKernel,
        GDNPipelineAdapter,
        REPOSITORY_ROOT,
        _tool_cache_environment,
    )
    from .config import Direction
except ImportError:
    from autotune import (
        ARMS,
        BWD_OUTPUTS,
        DEFAULT_TOP_K,
        FWD_OUTPUTS,
        GDNCandidateKernel,
        GDNPipelineAdapter,
        REPOSITORY_ROOT,
        _tool_cache_environment,
    )
    from config import Direction


GUIDED_ARM = "z3-cm-topk"
ORACLE_ARM = "z3-legal-exhaustive"
CONTROL_ARMS = ("budget-raw", "budget-random")
GPU_QUERY = "timestamp,name,driver_version,power.draw,temperature.gpu,clocks.sm,clocks.mem,memory.used"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _canonical(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_analysis(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or not isinstance(value.get("runs"), list):
        raise ValueError("GDN analysis must contain a runs list")
    return value


def modal_winners(analysis, direction):
    """Select each arm's modal winner without consulting confirmation latency."""

    selected = Direction(direction).value
    winners = {}
    for arm in ARMS:
        runs = [row for row in analysis["runs"] if row.get("direction") == selected and row.get("arm") == arm]
        if not runs:
            raise ValueError(f"Missing cold-search winner for {selected}/{arm}")
        configs = []
        for row in runs:
            config = row.get("best_config")
            if not isinstance(config, dict) or config.get("direction") != selected:
                raise ValueError(f"Invalid best_config for {selected}/{arm}: {config!r}")
            configs.append(_canonical(config))
        counts = Counter(configs)
        canonical = sorted(counts, key=lambda value: (-counts[value], value))[0]
        winners[arm] = {
            "config": json.loads(canonical),
            "votes": counts[canonical],
            "repetitions": len(runs),
            "tie_break": "descending votes, then canonical JSON",
            "selection_rule": "modal-winner",
        }
    return winners


def candidate_median_winners(analysis, direction, num_tokens=None):
    """Select every arm's best candidate by its cold-trial median."""

    selected = Direction(direction).value
    grouped = {arm: {} for arm in ARMS}
    for row in analysis["runs"]:
        if row.get("direction") != selected or row.get("arm") not in grouped:
            continue
        arm = row["arm"]
        summary_path = row.get("path")
        if not summary_path:
            raise ValueError(f"Missing summary path for {selected}/{arm}")
        events_path = Path(summary_path).parent / "events.jsonl"
        if not events_path.is_file():
            raise ValueError(f"Missing cold-trial events for {selected}/{arm}: {events_path}")
        repetition = row.get("repetition")
        with events_path.open(encoding="utf-8") as stream:
            observed_tokens = None
            for line in stream:
                event = json.loads(line)
                if event.get("event") == "run_start":
                    observed_tokens = event.get("metadata", {}).get("shape", {}).get("T")
                    if num_tokens is not None and observed_tokens != num_tokens:
                        raise ValueError(f"Token shape mismatch in {events_path}: expected T={num_tokens}, got {observed_tokens}")
                if event.get("event") != "trial" or event.get("status") != "ok":
                    continue
                config = event.get("config")
                latency = event.get("latency_ms")
                if not isinstance(config, dict) or config.get("direction") != selected:
                    raise ValueError(f"Invalid successful-trial config in {events_path}")
                if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
                    raise ValueError(f"Invalid successful-trial latency in {events_path}: {latency!r}")
                canonical = _canonical(config)
                candidate = grouped[arm].setdefault(canonical, {"config": config, "latencies": [], "repetitions": set()})
                candidate["latencies"].append(float(latency))
                candidate["repetitions"].add(repetition)
            if num_tokens is not None and observed_tokens != num_tokens:
                raise ValueError(f"Missing or mismatched run_start token shape in {events_path}")
    winners = {}
    for arm in ARMS:
        candidates = grouped[arm]
        if not candidates:
            raise ValueError(f"No successful cold trials for {selected}/{arm}")
        ranked = sorted(
            candidates.items(),
            key=lambda item: (statistics.median(item[1]["latencies"]), item[0]),
        )
        _, candidate = ranked[0]
        winners[arm] = {
            "config": candidate["config"],
            "trial_count": len(candidate["latencies"]),
            "measurement_repetitions": len(candidate["repetitions"]),
            "median_latency_ms": statistics.median(candidate["latencies"]),
            "candidate_count": len(candidates),
            "tie_break": "lowest candidate median, then canonical JSON",
            "selection_rule": "candidate-median",
        }
    return winners


def confirmation_plan(analysis, direction, selection_rule="modal-winner", num_tokens=None):
    """Deduplicate identical configurations and retain every control relation."""

    selected = Direction(direction).value
    if selection_rule == "modal-winner":
        winners = modal_winners(analysis, selected)
        selection_detail = "modal cold-search winner per arm; canonical JSON tie-break; no confirmation measurements used"
    elif selection_rule == "candidate-median":
        winners = candidate_median_winners(analysis, selected, num_tokens=num_tokens)
        selection_detail = "lowest per-candidate cold-trial median per arm; canonical JSON tie-break; no confirmation measurements used"
    else:
        raise ValueError(f"Unknown winner selection rule: {selection_rule}")
    groups = {}
    for arm in ARMS:
        canonical = _canonical(winners[arm]["config"])
        groups.setdefault(canonical, []).append(arm)
    guided = _canonical(winners[GUIDED_ARM]["config"])
    candidates = [{"config": winners[GUIDED_ARM]["config"], "arms": [GUIDED_ARM]}]
    comparisons = []
    for canonical, arms in groups.items():
        identical = canonical == guided
        comparisons.append({
            "alternative_arms": [] if identical else arms,
            "identical_arms": arms if identical else [],
            "config": json.loads(canonical),
            "identical": identical,
            "relations": [
                relation
                for relation, members in (
                    ("same_space_exhaustive", {ORACLE_ARM}),
                    ("equal_budget_control", set(CONTROL_ARMS)),
                    ("stock_informational", {"stock-raw"}),
                )
                if any(arm in members for arm in arms)
            ],
        })
        if not identical:
            candidates.append({"config": json.loads(canonical), "arms": arms})
    return {
        "schema": "tilelang.gdn.confirmation-plan/v1",
        "direction": selected,
        "num_tokens": num_tokens,
        "guided_arm": GUIDED_ARM,
        "guided_top_k": DEFAULT_TOP_K[Direction(selected)],
        "winners": winners,
        "candidates": candidates,
        "comparisons": comparisons,
        "selection_rule": selection_rule,
        "selection_rule_detail": selection_detail,
    }


def abba_order(round_index):
    """Return guided/alternative positions in AB, BA, BA, AB groups."""

    if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
        raise ValueError("round_index must be a non-negative integer")
    return (0, 1) if round_index % 4 in (0, 3) else (1, 0)


def _sample_stats(values):
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("latency samples must be non-empty, finite, and positive")
    return {
        "count": len(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _round_stats(records):
    by_round = {}
    for record in records:
        by_round.setdefault((record["pair_index"], record["round"]), []).append(record["median_ms"])
    if not by_round or any(len(values) != 1 for values in by_round.values()):
        raise ValueError("each candidate must have exactly one measurement per comparison round")
    medians = [values[0] for _, values in sorted(by_round.items())]
    return {**_sample_stats(medians), "round_medians_ms": medians}


def evaluate_comparisons(plan, measurements):
    """Compute ratios from paired rounds; identical configs are exactly equal."""

    guided = _canonical(plan["winners"][GUIDED_ARM]["config"])
    results = []
    for pair_index, comparison in enumerate(plan["comparisons"]):
        alternative = _canonical(comparison["config"])
        if comparison["identical"]:
            records = [row for row in measurements if row["config_id"] == guided]
            guided_stats = _round_stats(records)
            alternative_stats = guided_stats
            ratio = 1.0
            paired_differences = [0.0 for _ in guided_stats["round_medians_ms"]]
        else:
            pair_records = [row for row in measurements if row["pair_index"] == pair_index]
            guided_stats = _round_stats([row for row in pair_records if row["config_id"] == guided])
            alternative_stats = _round_stats([row for row in pair_records if row["config_id"] == alternative])
            ratio = guided_stats["median_ms"] / alternative_stats["median_ms"]
            paired_differences = [
                left - right
                for left, right in zip(guided_stats["round_medians_ms"], alternative_stats["round_medians_ms"], strict=True)
            ]
        requirements = {}
        if "same_space_exhaustive" in comparison["relations"]:
            requirements["within_2_percent_of_exhaustive"] = ratio <= 1.02
        if "equal_budget_control" in comparison["relations"]:
            requirements["not_slower_than_equal_budget_control"] = ratio <= 1.0
        results.append({
            **comparison,
            "pair_index": pair_index,
            "guided_median_ms": guided_stats["median_ms"],
            "alternative_median_ms": alternative_stats["median_ms"],
            "quality_ratio_guided_over_alternative": ratio,
            "paired_round_differences_ms": paired_differences,
            "requirements": requirements,
            "passed": all(requirements.values()),
        })
    return results


@contextmanager
def _confirmation_cache(run_dir):
    names = ("TILELANG_CACHE_DIR", "TILELANG_DISABLE_CACHE", "TILELANG_AUTO_TUNING_DISABLE_CACHE")
    previous = {name: os.environ.get(name) for name in names}
    cache = run_dir / "cache" / "tilelang"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TILELANG_CACHE_DIR"] = str(cache)
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


def _gpu_telemetry():
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result.stdout.strip()


def _measure_public_pipeline(candidate, cache, warmup, repeat):
    import torch

    for _ in range(warmup):
        candidate._execute_public_pipeline()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for index in range(repeat):
        cache.zero_()
        starts[index].record()
        candidate._execute_public_pipeline()
        ends[index].record()
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return _sample_stats(samples), samples


def _relative_l2(actual, expected):
    import torch

    if actual.shape[1] > expected.shape[1]:
        actual = actual[:, : expected.shape[1]]
    error = torch.linalg.vector_norm(actual.double() - expected.double()).item()
    reference = torch.linalg.vector_norm(expected.double()).item()
    if reference == 0:
        if error != 0:
            raise ValueError("nonzero error against a zero reference")
        return 0.0
    return error / reference


def _validate(candidate, reference, phase):
    actual = candidate._execute_public_pipeline()
    candidate.pipeline.manual_check_prog(actual, reference)
    names = FWD_OUTPUTS if candidate.direction is Direction.FWD else BWD_OUTPUTS
    metrics = {}
    for name, actual_value, expected_value in zip(names, actual, reference, strict=True):
        if candidate.pipeline.batch.state_v_first and name in ("final_state", "dh0"):
            actual_value = actual_value.transpose(-1, -2)
        metrics[name] = _relative_l2(actual_value, expected_value)
    return {"phase": phase, "status": "ok", "relative_l2": metrics, "max_relative_l2": max(metrics.values())}


def _source_evidence():
    paths = [
        Path(__file__),
        Path(__file__).with_name("autotune.py"),
        Path(__file__).with_name("runner.py"),
        REPOSITORY_ROOT / "tilelang/cuda/cost_model.py",
        REPOSITORY_ROOT / "manifest/hardware-sm120.json",
        REPOSITORY_ROOT / "manifest/calibration-sm120.json",
    ]
    files = {}
    for path in paths:
        resolved = path.resolve()
        files[str(resolved.relative_to(REPOSITORY_ROOT))] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()
    return {"git_commit": commit, "file_sha256": files}


def _inventory(root):
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "after.json":
            continue
        entries.append({"path": relative, "size_bytes": path.stat().st_size})
    return entries


def run_confirmation(direction, analysis_path, output_dir, *, rounds=7, warmup=50, repeat=200, seed=42, selection_rule="modal-winner", num_tokens=4096):
    for value, name in ((rounds, "rounds"), (warmup, "warmup"), (repeat, "repeat"), (num_tokens, "num_tokens")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    selected = Direction(direction)
    run_dir = Path(output_dir).expanduser().resolve(strict=False) / selected.value
    if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
        raise ValueError(f"Refusing a non-empty or non-directory confirmation path: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    plan = confirmation_plan(_read_analysis(analysis_path), selected, selection_rule=selection_rule, num_tokens=num_tokens)
    summary = {
        "schema": "tilelang.gdn.confirmation/v1",
        "status": "running",
        "direction": selected.value,
        "num_tokens": num_tokens,
        "rounds": rounds,
        "warmup": warmup,
        "repeat": repeat,
        "seed": seed,
        "plan": plan,
        "source": _source_evidence(),
        "timing": f"median of per-round medians; each round has {repeat} L2-flushed CUDA-event samples after {warmup} warmups",
        "reference_and_compilation_in_timed_region": False,
        "measurements": [],
        "validations": [],
    }
    summary_path = run_dir / "summary.json"
    samples_path = run_dir / "samples.jsonl"
    _write_json(run_dir / "before.json", {"empty": True, "entries": [], "utc": _utc_now()})
    _write_json(summary_path, summary)
    try:
        with _confirmation_cache(run_dir):
            import torch

            pipeline = GDNPipelineAdapter(selected, seed=seed, input_kwargs={"num_tokens": num_tokens})
            candidates = {}
            for entry in plan["candidates"]:
                config = entry["config"]
                canonical = _canonical(config)
                candidates[canonical] = GDNCandidateKernel(
                    selected,
                    **{key: value for key, value in config.items() if key != "direction"},
                    pipeline=pipeline,
                )
            reference = pipeline.reference_program()
            for candidate in candidates.values():
                summary["validations"].append(_validate(candidate, reference, "before"))
            cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
            guided = _canonical(plan["winners"][GUIDED_ARM]["config"])
            pairs = [(index, row) for index, row in enumerate(plan["comparisons"]) if not row["identical"]]
            with samples_path.open("x", encoding="utf-8") as sample_stream:
                record_index = 0

                def measure(pair_index, round_index, config_id):
                    nonlocal record_index
                    before = _gpu_telemetry()
                    stats, samples = _measure_public_pipeline(candidates[config_id], cache, warmup, repeat)
                    after = _gpu_telemetry()
                    raw = {
                        "record_index": record_index,
                        "pair_index": pair_index,
                        "round": round_index,
                        "config_id": config_id,
                        "samples_ms": samples,
                        "telemetry_before": before,
                        "telemetry_after": after,
                    }
                    sample_stream.write(json.dumps(raw, allow_nan=False, separators=(",", ":")) + "\n")
                    sample_stream.flush()
                    compact = {key: value for key, value in raw.items() if key not in ("samples_ms", "telemetry_before", "telemetry_after")}
                    compact.update(stats)
                    compact["samples"] = f"samples.jsonl#record_index={record_index}"
                    summary["measurements"].append(compact)
                    record_index += 1
                    _write_json(summary_path, summary)
                    print(json.dumps({"direction": selected.value, **compact}, allow_nan=False), flush=True)

                if pairs:
                    for pair_index, comparison in pairs:
                        alternative = _canonical(comparison["config"])
                        for round_index in range(rounds):
                            for role in abba_order(round_index):
                                measure(pair_index, round_index, guided if role == 0 else alternative)
                else:
                    for round_index in range(rounds):
                        measure(-1, round_index, guided)
            for candidate in candidates.values():
                summary["validations"].append(_validate(candidate, reference, "after"))
        summary["comparisons"] = evaluate_comparisons(plan, summary["measurements"])
        summary["acceptance"] = {
            "passed": all(row["passed"] for row in summary["comparisons"]),
            "rule": "guided/exhaustive <= 1.02 and guided/each equal-budget control <= 1.0; identical configs are exactly equal",
        }
        summary["status"] = "ok"
    except BaseException as error:
        summary["status"] = "error"
        summary["error"] = f"{type(error).__name__}: {error}"
        _write_json(summary_path, summary)
        raise
    finally:
        _write_json(summary_path, summary)
        _write_json(run_dir / "after.json", {"entries": _inventory(run_dir), "utc": _utc_now()})
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("fwd", "bwd"), required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-rule", choices=("modal-winner", "candidate-median"), default="modal-winner")
    parser.add_argument("--num-tokens", type=int, default=4096)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = run_confirmation(
        args.direction,
        args.analysis,
        args.output_dir,
        rounds=args.rounds,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        selection_rule=args.selection_rule,
        num_tokens=args.num_tokens,
    )
    print(json.dumps({"status": summary["status"], "acceptance": summary["acceptance"], "comparisons": summary["comparisons"]}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
