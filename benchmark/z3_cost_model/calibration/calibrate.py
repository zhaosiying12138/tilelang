#!/usr/bin/env python3
"""Orchestrate isolated calibration workers; ``--list`` never touches a GPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import traceback
import uuid
from typing import Any, Sequence

try:
    from .environment import PreflightError, base_environment, nvidia_snapshot, require_idle_gpu, utc_now
    from .microbenchmarks import failed_result  # Safe module: heavy imports are inside run_benchmark.
    from .paths import artifact_dir_path, cache_environment, run_cache_path, task_cache_path, worker_result_path
    from .schema import aggregate_calibration, assert_finite_json, json_dumps, write_json
    from .specs import CORE_BENCHMARKS, get_spec, list_benchmarks
except ImportError:  # Direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from environment import PreflightError, base_environment, nvidia_snapshot, require_idle_gpu, utc_now
    from microbenchmarks import failed_result
    from paths import artifact_dir_path, cache_environment, run_cache_path, task_cache_path, worker_result_path
    from schema import aggregate_calibration, assert_finite_json, json_dumps, write_json
    from specs import CORE_BENCHMARKS, get_spec, list_benchmarks

DEFAULT_WARMUP = 10
DEFAULT_REPEAT = 50
DEFAULT_TIMEOUT_SECONDS = 240.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List benchmarks and formulas without importing torch/TileLang or probing CUDA")
    parser.add_argument("--only", action="append", metavar="NAME[,NAME...]", help="Run only selected benchmarks; repeat the option or use commas")
    parser.add_argument("--output", default="calibration.json", help="Destination for the versioned aggregate JSON")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help=f"Warmup launches per benchmark (default {DEFAULT_WARMUP})")
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT, help=f"Raw event samples per benchmark (default {DEFAULT_REPEAT})")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="Wall timeout for each single-process worker")
    parser.add_argument("--cache-root", help="Absolute parent for unique per-run caches (default: system temporary directory)")
    parser.add_argument("--device", type=int, default=0, help="Physical NVIDIA GPU index; mapped to device 0 inside each worker")
    parser.add_argument("--instruction-kind", choices=("ptx", "sass"), default="ptx", help="Compiled artifact exported and required for GEMM/FMA instruction audits")
    parser.add_argument("--require-complete", action="store_true", help="Write diagnostics but return nonzero unless every core benchmark is complete")

    # Internal worker protocol.  These options are intentionally not needed by users.
    parser.add_argument("--worker", metavar="NAME", help=argparse.SUPPRESS)
    parser.add_argument("--run-cache", help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    parser.add_argument("--event-overhead-ms", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--empty-launch-overhead-ms", type=float, help=argparse.SUPPRESS)
    return parser


def parse_only(values: Sequence[str] | None) -> list[str]:
    if not values:
        return list(CORE_BENCHMARKS)
    requested = set()
    for value in values:
        for name in value.split(","):
            name = name.strip()
            if not name:
                raise ValueError("--only contains an empty benchmark name")
            get_spec(name)
            requested.add(name)
    if not requested:
        raise ValueError("--only selected no benchmarks")
    return [name for name in CORE_BENCHMARKS if name in requested]


def _validate_common(args: argparse.Namespace) -> None:
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.repeat <= 0:
        raise ValueError("--repeat must be > 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.device < 0:
        raise ValueError("--device must be >= 0")


def _write_worker_payload(path: Path, result: dict[str, Any]) -> None:
    assert_finite_json(result)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_worker(args: argparse.Namespace) -> int:
    """Internal entry; the parent has already performed safety preflights."""

    name = get_spec(args.worker).name
    if not args.run_cache or not args.result:
        raise ValueError("Worker requires --run-cache and --result")
    expected = worker_result_path(args.run_cache, name)
    actual = Path(args.result).resolve(strict=False)
    if actual != expected:
        raise ValueError(f"Worker result must be exactly {expected}, got {actual}")
    task = task_cache_path(args.run_cache, name)
    artifacts = artifact_dir_path(args.run_cache, name)
    for key, value in cache_environment(task).items():
        configured = os.environ.get(key)
        if configured is None or Path(configured).resolve(strict=False) != Path(value).resolve(strict=False):
            raise ValueError(f"Worker cache {key} is not the parent-assigned per-task path")
    try:
        from microbenchmarks import run_benchmark

        result = run_benchmark(
            name,
            warmup=args.warmup,
            repeat=args.repeat,
            physical_device=args.device,
            event_overhead_ms=args.event_overhead_ms,
            empty_launch_overhead_ms=args.empty_launch_overhead_ms,
            artifact_dir=str(artifacts),
            instruction_artifact_kind=args.instruction_kind,
        )
    except BaseException as error:  # A failed item must not discard other items.
        result = failed_result(name, error, args.warmup, args.repeat)
        result["error"]["traceback"] = traceback.format_exc()
    _write_worker_payload(expected, result)
    return 0


def _kill_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_one_process(
    args: argparse.Namespace,
    name: str,
    run_cache: Path,
    overhead: dict[str, float | None],
) -> dict[str, Any]:
    task = task_cache_path(run_cache, name)
    task.mkdir(parents=True, exist_ok=False)
    cache = cache_environment(task)
    for directory in cache.values():
        Path(directory).mkdir(parents=True, exist_ok=False)
    result_path = worker_result_path(run_cache, name)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker", name,
        "--run-cache", str(run_cache),
        "--result", str(result_path),
        "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
        "--timeout", str(args.timeout),
        "--device", str(args.device),
        "--instruction-kind", args.instruction_kind,
    ]
    if overhead.get("event") is not None:
        command += ["--event-overhead-ms", repr(overhead["event"])]
    if overhead.get("empty") is not None:
        command += ["--empty-launch-overhead-ms", repr(overhead["empty"])]
    env = os.environ.copy()
    env.update(cache)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.device),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TILELANG_DEFAULT_TARGET": "cuda",
        "TILELANG_EXECUTION_BACKEND": "tvm_ffi",
    })
    before = nvidia_snapshot(args.device)
    require_idle_gpu(args.device)
    stdout_path = task / "stdout.log"
    stderr_path = task / "stderr.log"
    returncode = None
    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            env=env,
            shell=False,
            start_new_session=os.name == "posix",
        )
        try:
            returncode = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_group(process)
    after = nvidia_snapshot(args.device)
    orchestration = {
        "before": before,
        "after": after,
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout_seconds": args.timeout,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "process_isolation": "one fresh Python process and one fresh process group per benchmark",
    }
    if timed_out:
        error = TimeoutError(f"{name} exceeded {args.timeout} seconds and its process group was killed")
        result = failed_result(name, error, args.warmup, args.repeat)
        result["status"] = "timeout"
    elif not result_path.is_file():
        error = RuntimeError(f"Worker exited with code {returncode} without a result; see {stderr_path}")
        result = failed_result(name, error, args.warmup, args.repeat)
    else:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            assert_finite_json(result)
            if result.get("name") != name:
                raise ValueError("Worker result benchmark identity mismatch")
            if returncode != 0 or not isinstance(result, dict):
                raise ValueError(f"Worker abnormal return code {returncode}")
        except BaseException as error:
            result = failed_result(name, error, args.warmup, args.repeat)
    result.setdefault("environment", {})["orchestration"] = orchestration
    return result


def _new_run_cache(root: str | None) -> tuple[str, Path]:
    if root is None:
        root_path = Path(tempfile.gettempdir()).resolve() / "tilelang-z3-cost-model-calibration"
    else:
        root_path = Path(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{uuid.uuid4().hex[:12]}"
    run_cache = run_cache_path(root_path, run_id)
    run_cache.mkdir(parents=True, exist_ok=False)
    (run_cache / "tasks").mkdir()
    return run_id, run_cache


def list_output() -> str:
    lines = ["CPU-only benchmark listing; no GPU preflight was performed.", ""]
    for item in list_benchmarks():
        lines.append(f"{item['name']}: shape={item['shape']}, dtype={item['dtype']}, threads={item['threads']}, inner={item['inner_iterations']}, unroll={item['unroll']}")
        lines.append(f"  {item['description']}")
        lines.append(f"  {item['formula']}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        if args.worker:
            parser.error("--list cannot be combined with --worker")
        sys.stdout.write(list_output())
        return 0
    try:
        _validate_common(args)
        if args.worker:
            return run_worker(args)
        selected = parse_only(args.only)
    except (ValueError, PreflightError) as error:
        parser.error(str(error))

    output = Path(args.output)
    try:
        environment = base_environment(args.device)
        run_id, run_cache = _new_run_cache(args.cache_root)
    except (PreflightError, OSError, ValueError) as error:
        print(f"Calibration preflight failed; no GPU worker started: {error}", file=sys.stderr)
        return 2

    results: dict[str, Any] = {}
    overhead: dict[str, float | None] = {"event": None, "empty": None}
    ordered = (["empty_launch"] if "empty_launch" in selected else []) + [name for name in selected if name != "empty_launch"]
    started = utc_now()
    for name in ordered:
        try:
            result = _run_one_process(args, name, run_cache, overhead)
        except BaseException as error:
            # Preflight failures after earlier results still produce a partial artifact.
            result = failed_result(name, error, args.warmup, args.repeat)
            result["environment"]["orchestration"] = {"preflight_failure": True}
            results[name] = result
            print(f"{name}: {type(error).__name__}: {error}", file=sys.stderr)
            break
        results[name] = result
        print(f"{name}: {result['status']}", flush=True)
        if name == "empty_launch" and result.get("status") == "complete":
            overhead["event"] = result["event_pair"]["median_ms"]
            overhead["empty"] = result["median_ms"]

    try:
        environment["final_gpu"] = nvidia_snapshot(args.device)
    except BaseException as error:
        environment["final_gpu_error"] = str(error)
    run = {
        "run_id": run_id,
        "cache_dir": str(run_cache),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "selected": ordered,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "timeout_seconds": args.timeout,
        "instruction_artifact_kind": args.instruction_kind,
        "one_process_per_benchmark": True,
    }
    try:
        document = aggregate_calibration(results, environment=environment, run=run, allow_partial=True)
        write_json(output, document)
    except BaseException as error:
        print(f"Failed to aggregate/write calibration: {error}", file=sys.stderr)
        return 2
    print(f"Wrote {document['status']} calibration to {output}; missing={document['missing_benchmarks']}")
    if args.require_complete and document["status"] != "complete":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
