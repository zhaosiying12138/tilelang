"""Audit the formal fresh-process classic-operator cold-search protocol.

Every formal top-k-10 run is checked against its persisted evidence.  Sensitivity
runs with another positive integer top-k are counted and excluded from formal
identity, duplicate, and process-id accounting.  Success is never inferred from
an exit code or a directory name.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


SUITES = ("gemm_square", "gemm_short_k", "conv2d_stride_tail", "flash_attention_causal")
ARMS = ("stock-raw", "z3-legal-exhaustive", "z3-cm-topk", "budget-raw", "budget-random")
REPETITIONS = (0, 1, 2)
TOP_K = 10
SEED = 20261026
WARMUP = 10
REPEAT = 50
CACHE_TRACK = "cold"
EXPECTED = {
    "gemm_square": {"R": 288, "U": 240},
    "gemm_short_k": {"R": 288, "U": 240},
    "conv2d_stride_tail": {"R": 144, "U": 120},
    "flash_attention_causal": {"R": 54, "U": 54},
}
EXPECTED_KEYS = frozenset(
    (suite, arm, repetition)
    for suite in SUITES
    for arm in ARMS
    for repetition in REPETITIONS
)
_SUITE_ORDER = {suite: index for index, suite in enumerate(SUITES)}
_ARM_ORDER = {arm: index for index, arm in enumerate(ARMS)}


def _parse_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite JSON number: {value}")
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_object(path):
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, parse_float=_parse_float, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _write(path, value):
    path = Path(path).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _identity_sort_key(key):
    suite, arm, repetition = key
    return (_SUITE_ORDER[suite], repetition, _ARM_ORDER[arm])


def _read_evidence(run_dir, name, key, errors):
    try:
        return _load_object(run_dir / name)
    except (OSError, UnicodeError, ValueError) as error:
        errors.append(f"unreadable {name} for {key}: {error}")
        return None


def _resolved_path(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        return Path(value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


def audit_cold_runs(root):
    """Return a strict-JSON-compatible audit report for every discoverable run."""

    root = Path(root).expanduser().resolve(strict=False)
    errors = []
    records = []
    pids = {}
    seen = set()
    ignored_sensitivity_runs = 0

    summary_paths = []
    if not root.is_dir():
        errors.append(f"audit root is not a directory: {root}")
    else:
        try:
            summary_paths = sorted(
                root.rglob("summary.json"),
                key=lambda path: path.relative_to(root).as_posix(),
            )
        except (OSError, RuntimeError) as error:
            errors.append(f"unable to enumerate summaries under {root}: {error}")

    for path in summary_paths:
        try:
            summary = _load_object(path)
        except (OSError, UnicodeError, ValueError) as error:
            errors.append(f"unreadable summary.json {path}: {error}")
            continue

        operator_run = summary.get("operator_run")
        operator_run = operator_run if isinstance(operator_run, dict) else {}
        experiment = summary.get("experiment")
        experiment = experiment if isinstance(experiment, dict) else {}
        suite = operator_run.get("suite")
        arm = summary.get("arm")
        repetition = experiment.get("repetition")
        candidate_key = (suite, arm, repetition)
        valid_identity = (
            isinstance(suite, str)
            and suite in EXPECTED
            and isinstance(arm, str)
            and arm in ARMS
            and isinstance(repetition, int)
            and not isinstance(repetition, bool)
            and repetition in REPETITIONS
        )
        if not valid_identity:
            errors.append(f"unexpected run identity {candidate_key}: {path}")
            continue
        key = candidate_key

        top_k = operator_run.get("top_k")
        if _positive_int(top_k) and top_k != TOP_K:
            ignored_sensitivity_runs += 1
            continue
        if not _positive_int(top_k) or top_k != TOP_K:
            errors.append(f"formal top_k for {key}: expected {TOP_K}, got {top_k!r}")
        if key in seen:
            errors.append(f"duplicate run identity {key}: {path}")
            continue
        seen.add(key)

        run_dir = path.parent
        config = _read_evidence(run_dir, "config.json", key, errors)
        environment = _read_evidence(run_dir, "environment.json", key, errors)
        before = _read_evidence(run_dir, "before.json", key, errors)
        after = _read_evidence(run_dir, "after.json", key, errors)

        pid = None
        if environment is not None:
            pid = environment.get("pid")
            if not _positive_int(pid):
                errors.append(f"invalid pid for {key}: {pid!r}")
            elif pid in pids:
                errors.append(f"process reuse between {pids[pid]} and {key}: pid={pid}")
            else:
                pids[pid] = key
            for field, expected in (("suite", suite), ("arm", arm), ("seed", SEED)):
                if environment.get(field) != expected:
                    errors.append(
                        f"environment {field} for {key}: expected {expected!r}, got {environment.get(field)!r}"
                    )
            if "cache_track" in environment and environment["cache_track"] != CACHE_TRACK:
                errors.append(f"environment cache_track for {key}: expected 'cold', got {environment['cache_track']!r}")

        if summary.get("status") != "ok":
            errors.append(f"run status is not ok for {key}: {summary.get('status')!r}")
        if "cache_track" in operator_run and operator_run["cache_track"] != CACHE_TRACK:
            errors.append(f"summary cache_track for {key}: expected 'cold', got {operator_run['cache_track']!r}")
        if "arm" in experiment and experiment["arm"] != arm:
            errors.append(f"experiment arm for {key}: expected {arm!r}, got {experiment['arm']!r}")

        if config is not None:
            for field, expected in (
                ("warmup", WARMUP),
                ("repeat", REPEAT),
                ("repetition", repetition),
                ("cache_track", CACHE_TRACK),
            ):
                if config.get(field) != expected:
                    errors.append(f"config {field} for {key}: expected {expected!r}, got {config.get(field)!r}")
            for field, expected in (("suite", suite), ("arm", arm), ("seed", SEED), ("top_k", TOP_K)):
                if field in config and config[field] != expected:
                    errors.append(f"config {field} for {key}: expected {expected!r}, got {config[field]!r}")

        if before is not None and (
            before.get("empty") is not True or before.get("entries") != [] or before.get("cache_entries") != []
        ):
            errors.append(f"before state is not demonstrably empty for {key}")

        after_cache_entries = None
        if after is not None:
            if not isinstance(after.get("entries"), list) or not isinstance(after.get("cache_entries"), list):
                errors.append(f"after cache inventory is incomplete for {key}")
            if isinstance(after.get("cache_entries"), list):
                after_cache_entries = len(after["cache_entries"])
        if not (run_dir / "events.jsonl").is_file():
            errors.append(f"missing events.jsonl for {key}")

        env = {}
        cache_directory = None
        if environment is not None:
            env = environment.get("environment")
            env = env if isinstance(env, dict) else {}
            if env.get("TILELANG_DISABLE_CACHE") != "0" or env.get("CUDA_CACHE_DISABLE") != "0":
                errors.append(f"cold cache-disable flags are not zero for {key}")
            cache_directory = _resolved_path(env.get("TILELANG_CACHE_DIR"))

        cache = summary.get("cache")
        cache = cache if isinstance(cache, dict) else {}
        pipeline_cache = cache.get("pipeline_cache_directory")
        resolved_pipeline_cache = _resolved_path(pipeline_cache)
        expected_cache = (run_dir / "cache").resolve(strict=False)
        if resolved_pipeline_cache is None or not resolved_pipeline_cache.is_relative_to(expected_cache):
            errors.append(f"pipeline cache escapes run directory for {key}: {pipeline_cache!r}")
        if cache_directory != expected_cache:
            errors.append(f"TileLang cache identity mismatch for {key}: {env.get('TILELANG_CACHE_DIR')!r}")

        expected = EXPECTED[suite]
        if summary.get("R") != expected["R"]:
            errors.append(f"raw denominator changed for {key}: expected {expected['R']}, got {summary.get('R')!r}")
        legal = summary.get("U")
        if arm == "stock-raw":
            if legal is not None and legal != expected["U"]:
                errors.append(f"stock legal denominator changed for {key}: expected null or {expected['U']}, got {legal!r}")
        elif legal != expected["U"]:
            errors.append(f"legal denominator changed for {key}: expected {expected['U']}, got {legal!r}")
        selected = expected["R"] if arm == "stock-raw" else expected["U"] if arm == "z3-legal-exhaustive" else TOP_K
        if summary.get("selected") != selected:
            errors.append(f"selection budget changed for {key}: expected {selected}, got {summary.get('selected')!r}")

        compile_attempts = summary.get("compile_attempts")
        if not _positive_int(compile_attempts):
            errors.append(f"missing compile attempts for {key}: {compile_attempts!r}")
        profile = summary.get("profile")
        profile = profile if isinstance(profile, dict) else {}
        if not _positive_int(profile.get("success")):
            errors.append(f"no successful profile for {key}: {profile.get('success')!r}")
        latency = summary.get("best_latency_ms")
        if (
            not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or not math.isfinite(latency)
            or latency <= 0
        ):
            errors.append(f"invalid best latency for {key}: {latency!r}")

        records.append(
            {
                "suite": suite,
                "arm": arm,
                "repetition": repetition,
                "pid": pid,
                "path": str(path),
                "cache_entries_after": after_cache_entries,
                "compile_attempts": compile_attempts,
                "profile_attempts": profile.get("attempts"),
                "end_to_end_wall_seconds": summary.get("end_to_end_wall_seconds"),
                "best_latency_ms": latency,
            }
        )

    missing = sorted(EXPECTED_KEYS - seen, key=_identity_sort_key)
    if missing:
        errors.append(f"missing runs: {missing}")
    errors.sort()
    records.sort(key=lambda row: _identity_sort_key((row["suite"], row["arm"], row["repetition"])))
    return {
        "schema": "tilelang.classic-operators.cold-protocol-audit/v1",
        "root": str(root),
        "repetitions": list(REPETITIONS),
        "expected_runs": len(EXPECTED_KEYS),
        "observed_runs": len(seen),
        "distinct_pids": len(pids),
        "passed": not errors,
        "errors": errors,
        "runs": records,
        "ignored_sensitivity_runs": ignored_sensitivity_runs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_cold_runs(args.root)
    _write(args.output, report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "runs"},
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
