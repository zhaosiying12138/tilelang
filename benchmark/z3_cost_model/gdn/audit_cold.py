"""Audit the formal fresh-process GDN cold-search protocol.

The audit checks every expected run and reports all violations in one JSON file.
It never infers success from directory names or a zero exit code alone.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


DIRECTIONS = ("fwd", "bwd")
ARMS = ("stock-raw", "z3-legal-exhaustive", "z3-cm-topk", "budget-raw", "budget-random")
EXPECTED = {
    "fwd": {"R": 4, "U": 3, "top_k": 2},
    "bwd": {"R": 9, "U": 6, "top_k": 3},
}


def _load(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def audit_cold_runs(root, repetitions=(0, 1, 2), num_tokens=4096):
    root = Path(root).expanduser().resolve(strict=False)
    if isinstance(num_tokens, bool) or not isinstance(num_tokens, int) or num_tokens <= 0:
        raise ValueError("num_tokens must be a positive integer")
    errors = []
    records = []
    pids = {}
    seen = set()
    expected_keys = {(direction, arm, repetition) for direction in DIRECTIONS for arm in ARMS for repetition in repetitions}
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        errors.append("missing protocol.json")
    else:
        protocol = _load(protocol_path)
        required = {
            "schema": "tilelang.gdn.cold-fresh-protocol/v1",
            "directions": list(DIRECTIONS),
            "repetitions": list(repetitions),
            "arms": list(ARMS),
            "cache_track": "cold",
            "warmup": 10,
            "repeat": 50,
            "fresh_process_per_arm_repetition": True,
        }
        for key, value in required.items():
            if protocol.get(key) != value:
                errors.append(f"protocol {key}: expected {value!r}, got {protocol.get(key)!r}")
        if (num_tokens != 4096 or "num_tokens" in protocol) and protocol.get("num_tokens") != num_tokens:
            errors.append(f"protocol num_tokens: expected {num_tokens}, got {protocol.get('num_tokens')!r}")
        if not isinstance(protocol.get("script_sha256"), str) or len(protocol["script_sha256"]) != 64:
            errors.append("protocol script_sha256 is missing or malformed")

    for path in sorted(root.rglob("summary.json")):
        summary = _load(path)
        gdn = summary.get("gdn", {})
        direction = gdn.get("direction")
        arm = summary.get("arm")
        repetition = summary.get("experiment", {}).get("repetition")
        key = (direction, arm, repetition)
        if direction not in DIRECTIONS or arm not in ARMS or repetition not in repetitions:
            errors.append(f"unexpected run identity {key}: {path}")
            continue
        if gdn.get("cache_track") != "cold":
            errors.append(f"non-cold run in cold root {key}: {path}")
            continue
        if key in seen:
            errors.append(f"duplicate run identity {key}: {path}")
            continue
        seen.add(key)
        run_dir = path.parent
        try:
            config = _load(run_dir / "config.json")
            environment = _load(run_dir / "environment.json")
            before = _load(run_dir / "before.json")
            after = _load(run_dir / "after.json")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"unreadable run evidence {key}: {error}")
            continue
        pid = environment.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            errors.append(f"invalid pid for {key}: {pid!r}")
        elif pid in pids:
            errors.append(f"process reuse between {pids[pid]} and {key}: pid={pid}")
        else:
            pids[pid] = key
        if summary.get("status") != "ok":
            errors.append(f"run status is not ok for {key}: {summary.get('status')}")
        for field, expected in (("warmup", 10), ("repeat", 50), ("repetition", repetition), ("cache_track", "cold")):
            if config.get(field) != expected:
                errors.append(f"config {field} for {key}: expected {expected!r}, got {config.get(field)!r}")
        if config.get("shape", {}).get("T") != num_tokens or gdn.get("shape", {}).get("T") != num_tokens:
            errors.append(f"token shape mismatch for {key}: expected T={num_tokens}")
        if before.get("empty") is not True or before.get("entries") != [] or before.get("cache_entries") != []:
            errors.append(f"before state is not demonstrably empty for {key}")
        if not isinstance(after.get("entries"), list) or not isinstance(after.get("cache_entries"), list):
            errors.append(f"after cache inventory is incomplete for {key}")
        if not (run_dir / "events.jsonl").is_file():
            errors.append(f"missing events.jsonl for {key}")
        env = environment.get("environment", {})
        if env.get("TILELANG_DISABLE_CACHE") != "0" or env.get("CUDA_CACHE_DISABLE") != "0":
            errors.append(f"cold cache-disable flags are not zero for {key}")
        cache = summary.get("cache", {})
        pipeline_cache = cache.get("pipeline_cache_directory")
        expected_cache = (run_dir / "cache").resolve(strict=False)
        if pipeline_cache is None or not Path(pipeline_cache).resolve(strict=False).is_relative_to(expected_cache):
            errors.append(f"pipeline cache escapes run directory for {key}: {pipeline_cache!r}")
        if Path(env.get("TILELANG_CACHE_DIR", "")).resolve(strict=False) != expected_cache:
            errors.append(f"TileLang cache identity mismatch for {key}")
        expected = EXPECTED[direction]
        if summary.get("R") != expected["R"]:
            errors.append(f"raw denominator changed for {key}: {summary.get('R')}")
        if arm != "stock-raw" and summary.get("U") != expected["U"]:
            errors.append(f"legal denominator changed for {key}: {summary.get('U')}")
        if summary.get("selected") != (expected["R"] if arm == "stock-raw" else expected["U"] if arm == "z3-legal-exhaustive" else expected["top_k"]):
            errors.append(f"selection budget changed for {key}: {summary.get('selected')}")
        if not isinstance(summary.get("compile_attempts"), int) or summary["compile_attempts"] <= 0:
            errors.append(f"missing compile attempts for {key}")
        if not isinstance(summary.get("profile", {}).get("success"), int) or summary["profile"]["success"] <= 0:
            errors.append(f"no successful profile for {key}")
        latency = summary.get("best_latency_ms")
        if not isinstance(latency, (int, float)) or isinstance(latency, bool) or not math.isfinite(latency) or latency <= 0:
            errors.append(f"invalid best latency for {key}: {latency!r}")
        records.append({
            "direction": direction,
            "arm": arm,
            "repetition": repetition,
            "pid": pid,
            "path": str(path),
            "cache_entries_after": len(after.get("cache_entries", [])),
            "compile_attempts": summary.get("compile_attempts"),
            "profile_attempts": summary.get("profile", {}).get("attempts"),
            "end_to_end_wall_seconds": summary.get("end_to_end_wall_seconds"),
            "best_latency_ms": latency,
        })

    missing = sorted(expected_keys - seen)
    extra = sorted(seen - expected_keys)
    if missing:
        errors.append(f"missing runs: {missing}")
    if extra:
        errors.append(f"extra runs: {extra}")
    return {
        "schema": "tilelang.gdn.cold-protocol-audit/v1",
        "root": str(root),
        "repetitions": list(repetitions),
        "num_tokens": num_tokens,
        "expected_runs": len(expected_keys),
        "observed_runs": len(seen),
        "distinct_pids": len(pids),
        "passed": not errors,
        "errors": errors,
        "runs": sorted(records, key=lambda row: (row["direction"], row["repetition"], ARMS.index(row["arm"]))),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", default="0,1,2")
    parser.add_argument("--num-tokens", type=int, default=4096)
    args = parser.parse_args(argv)
    repetitions = tuple(int(value) for value in args.repetitions.split(",") if value)
    report = audit_cold_runs(args.root, repetitions=repetitions, num_tokens=args.num_tokens)
    _write(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "runs"}, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
