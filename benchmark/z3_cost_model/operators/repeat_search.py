"""Serial, process-isolated scheduler for classic operator search runs.

No thread/process pool is used: ``subprocess.run`` blocks until the only child
has exited.  Consequently at most one GPU-using autotune child is alive per
scheduler.  The default plan is every arm at repetitions 0, 1, 2 on cold caches.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

_OPERATORS_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = _OPERATORS_ROOT.parents[2]
if __package__:
    from .autotune import (
        ARMS,
        DEFAULT_CALIBRATION_PATH,
        DEFAULT_HARDWARE_PATH,
        DEFAULT_REPETITIONS,
        DEFAULT_TOP_K,
        SEED,
        SUITE_NAMES,
        _strict_json,
        _write_json,
        resolve_top_k,
    )
else:
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from benchmark.z3_cost_model.operators.autotune import (
        ARMS,
        DEFAULT_CALIBRATION_PATH,
        DEFAULT_HARDWARE_PATH,
        DEFAULT_REPETITIONS,
        DEFAULT_TOP_K,
        SEED,
        SUITE_NAMES,
        _strict_json,
        _write_json,
        resolve_top_k,
    )


def _unique_sequence(values, name, allowed=None):
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    if allowed is not None:
        unknown = set(result) - set(allowed)
        if unknown:
            raise ValueError(f"Unknown {name}: {sorted(unknown)}")
    return result


def _parse_repetitions(value: str):
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("repetitions must be comma-separated non-negative integers") from exc
    _unique_sequence(result, "repetitions")
    if any(item < 0 for item in result):
        raise ValueError("repetitions must be non-negative")
    return result


def job_output_root(output_dir, suite, top_k) -> Path:
    return Path(output_dir).expanduser().resolve(strict=False) / suite / f"top-k-{top_k:04d}"


def command_for_job(
    job,
    *,
    output_dir,
    warmup,
    repeat,
    seed,
    timeout,
    hardware_path,
    calibration_path,
    cache_mode,
    device,
    target,
    python_executable=None,
    autotune_path=None,
) -> list[str]:
    """Build one independently executable single-run command."""

    python = sys.executable if python_executable is None else str(python_executable)
    script = _OPERATORS_ROOT / "autotune.py" if autotune_path is None else str(autotune_path)
    output_root = job_output_root(output_dir, job["suite"], job["top_k"])
    return [
        python,
        str(script),
        "--suite",
        job["suite"],
        "--arm",
        job["arm"],
        "--repetition",
        str(job["repetition"]),
        "--top-k",
        str(job["top_k"]),
        "--output-dir",
        str(output_root),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--seed",
        str(seed),
        "--timeout",
        str(timeout),
        "--hardware",
        str(hardware_path),
        "--calibration",
        str(calibration_path),
        "--cache-mode",
        cache_mode,
        "--device",
        str(device),
        "--target",
        str(target),
    ]


def run_jobs(
    *,
    suites=SUITE_NAMES,
    arms=ARMS,
    repetitions=DEFAULT_REPETITIONS,
    output_dir,
    top_k=DEFAULT_TOP_K,
    warmup=10,
    repeat=50,
    seed=SEED,
    timeout=180,
    hardware_path=DEFAULT_HARDWARE_PATH,
    calibration_path=DEFAULT_CALIBRATION_PATH,
    cache_mode="cold",
    device="cuda",
    target="auto",
    python_executable=None,
    autotune_path=None,
    runner=None,
):
    """Run jobs sequentially and persist progress after every child exit."""

    selected_suites = _unique_sequence(suites, "suites", SUITE_NAMES)
    selected_arms = _unique_sequence(arms, "arms", ARMS)
    selected_repetitions = _unique_sequence(repetitions, "repetitions")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in selected_repetitions):
        raise ValueError("repetitions must be non-negative integers")
    # Validate every budget before launching the first GPU child.  In
    # particular, non-default sensitivity is legal only for gemm_square.
    for suite in selected_suites:
        resolve_top_k(suite, top_k)
    if cache_mode not in ("cold", "enabled", "no-cache", "warm"):
        raise ValueError("Unsupported cache_mode")

    output_root = Path(output_dir).expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "scheduler-summary.json"
    jobs = [
        {"suite": suite, "arm": arm, "repetition": repetition, "top_k": top_k}
        for suite in selected_suites
        for repetition in selected_repetitions
        for arm in selected_arms
    ]
    summary = {
        "schema": "tilelang.classic-operators.scheduler/v1",
        "status": "running",
        "execution": "sequential blocking subprocess.run; at most one GPU child alive",
        "cache_mode": cache_mode,
        "warmup": warmup,
        "repeat": repeat,
        "seed": seed,
        "planned_jobs": len(jobs),
        "completed_jobs": 0,
        "runs": [],
    }
    _write_json(summary_path, summary)
    invoke = subprocess.run if runner is None else runner
    environment = os.environ.copy()
    environment.update(
        TILELANG_AUTO_TUNING_CPU_COUNTS="1",
        TILELANG_AUTO_TUNING_MAX_CPU_COUNT="1",
        PYTHONUNBUFFERED="1",
    )

    for job in jobs:
        command = command_for_job(
            job,
            output_dir=output_root,
            warmup=warmup,
            repeat=repeat,
            seed=seed,
            timeout=timeout,
            hardware_path=hardware_path,
            calibration_path=calibration_path,
            cache_mode=cache_mode,
            device=device,
            target=target,
            python_executable=python_executable,
            autotune_path=autotune_path,
        )
        record = copy.deepcopy(job)
        record.update(
            status="running",
            command=command,
            run_directory=str(job_output_root(output_root, job["suite"], top_k) / f"{job['arm']}__rep-{job['repetition']:04d}"),
        )
        summary["runs"].append(record)
        _write_json(summary_path, summary)
        try:
            completed = invoke(command, check=False, env=environment, cwd=REPOSITORY_ROOT)
            returncode = completed.returncode
            record["returncode"] = returncode
            if returncode != 0:
                record["status"] = "error"
                summary["status"] = "error"
                summary["error"] = f"Child exited with status {returncode}: {command}"
                _write_json(summary_path, summary)
                raise subprocess.CalledProcessError(returncode, command)
            record["status"] = "ok"
            summary["completed_jobs"] += 1
            _write_json(summary_path, summary)
        except BaseException as exc:
            if record["status"] != "error":
                record["status"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"
            summary["status"] = "error"
            summary.setdefault("error", f"{type(exc).__name__}: {exc}")
            _write_json(summary_path, summary)
            raise

    summary["status"] = "ok"
    _write_json(summary_path, _strict_json(summary))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=(*SUITE_NAMES, "all"), default="all")
    parser.add_argument("--arm", choices=(*ARMS, "all"), action="append", default=None, help="Repeat to select multiple arms; default: all")
    repetition = parser.add_mutually_exclusive_group()
    repetition.add_argument("--repetition", type=int, help="One repetition id; use for a single-run scheduler command")
    repetition.add_argument("--repetitions", default=None, help="Comma-separated ids (default: 0,1,2)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--hardware", type=Path, default=DEFAULT_HARDWARE_PATH)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--cache-mode", choices=("cold", "enabled", "no-cache", "warm"), default="cold")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target", default="auto")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    suites = SUITE_NAMES if args.suite == "all" else (args.suite,)
    if args.arm is None or "all" in args.arm:
        if args.arm is not None and len(args.arm) != 1:
            raise ValueError("--arm all cannot be combined with other arms")
        arms = ARMS
    else:
        arms = _unique_sequence(args.arm, "arms", ARMS)
    if args.repetition is not None:
        repetitions = (args.repetition,)
    elif args.repetitions is not None:
        repetitions = _parse_repetitions(args.repetitions)
    else:
        repetitions = DEFAULT_REPETITIONS
    summary = run_jobs(
        suites=suites,
        arms=arms,
        repetitions=repetitions,
        output_dir=args.output_dir,
        top_k=args.top_k,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        timeout=args.timeout,
        hardware_path=args.hardware,
        calibration_path=args.calibration,
        cache_mode=args.cache_mode,
        device=args.device,
        target=args.target,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "completed_jobs": summary["completed_jobs"],
                "summary": str(Path(args.output_dir).expanduser().resolve(strict=False) / "scheduler-summary.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
