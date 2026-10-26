"""Host/GPU environment probes and preflight checks (no torch import)."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

MIN_HOST_MEMORY_GIB = 16
GPU_FIELDS = (
    "index",
    "name",
    "driver_version",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "clocks.sm",
    "clocks.max.sm",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "pstate",
    "compute_mode",
)


class PreflightError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _command(argv: list[str], timeout: float = 15) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        shell=False,
    )


def parse_free_gib(text: str, minimum_gib: int = MIN_HOST_MEMORY_GIB) -> dict[str, Any]:
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0].rstrip(":").lower() == "mem":
            if len(fields) < 2:
                break
            try:
                total = int(fields[1])
                available = int(fields[6]) if len(fields) >= 7 else None
            except ValueError as error:
                raise PreflightError(f"Cannot parse free -g line: {line!r}") from error
            return {
                "total_gib": total,
                "available_gib": available,
                "minimum_total_gib": minimum_gib,
                "passed": total >= minimum_gib,
                "source": "free -g Mem total",
            }
    raise PreflightError("free -g output has no parseable Mem row")


def host_memory_check(minimum_gib: int = MIN_HOST_MEMORY_GIB) -> dict[str, Any]:
    try:
        result = _command(["free", "-g"])
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PreflightError(f"Unable to execute free -g: {error}") from error
    if result.returncode != 0:
        raise PreflightError(f"free -g failed ({result.returncode}): {result.stderr.strip()}")
    parsed = parse_free_gib(result.stdout, minimum_gib)
    if not parsed["passed"]:
        raise PreflightError(f"Host memory is {parsed['total_gib']} GiB; at least {minimum_gib} GiB is required")
    return parsed


def parse_compute_apps(text: str) -> list[dict[str, str]]:
    apps = []
    for row in csv.reader(line for line in text.splitlines() if line.strip()):
        if not row or not any(cell.strip() for cell in row):
            continue
        apps.append({
            "pid": row[0].strip() if len(row) > 0 else "",
            "process_name": row[1].strip() if len(row) > 1 else "",
            "used_memory": row[2].strip() if len(row) > 2 else "",
        })
    return apps


def compute_processes(device: int = 0) -> list[dict[str, str]]:
    argv = [
        "nvidia-smi", "-i", str(device),
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = _command(argv)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PreflightError(f"Unable to query compute processes: {error}") from error
    if result.returncode != 0:
        raise PreflightError(f"nvidia-smi compute-process query failed: {result.stderr.strip()}")
    return parse_compute_apps(result.stdout)


def require_idle_gpu(device: int = 0) -> dict[str, Any]:
    apps = compute_processes(device)
    if apps:
        raise PreflightError(f"GPU {device} has active compute processes; refusing calibration: {apps}")
    return {"passed": True, "compute_processes": [], "device": device}


def _number(text: str) -> float | int | None:
    value = text.strip()
    try:
        number = float(value)
    except ValueError:
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return int(number) if number.is_integer() else number


def nvidia_snapshot(device: int = 0) -> dict[str, Any]:
    argv = [
        "nvidia-smi", "-i", str(device),
        "--query-gpu=" + ",".join(GPU_FIELDS),
        "--format=csv,noheader,nounits",
    ]
    try:
        result = _command(argv)
        apps = compute_processes(device)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PreflightError(f"Unable to collect nvidia-smi environment: {error}") from error
    if result.returncode != 0:
        raise PreflightError(f"nvidia-smi GPU query failed: {result.stderr.strip()}")
    rows = list(csv.reader(line for line in result.stdout.splitlines() if line.strip()))
    if len(rows) != 1 or len(rows[0]) != len(GPU_FIELDS):
        raise PreflightError(f"Unexpected nvidia-smi row for GPU {device}: {result.stdout!r}")
    raw = {name: value.strip() for name, value in zip(GPU_FIELDS, rows[0])}
    numeric = {name: _number(value) for name, value in raw.items()}
    snapshot = {
        "timestamp_utc": utc_now(),
        "device": device,
        "raw": raw,
        "values": numeric,
        "compute_processes": apps,
    }
    snapshot["clock_mhz"] = numeric.get("clocks.sm")
    snapshot["max_clock_mhz"] = numeric.get("clocks.max.sm")
    snapshot["temperature_c"] = numeric.get("temperature.gpu")
    snapshot["power_w"] = numeric.get("power.draw")
    return snapshot


def source_revision() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[3]
    result: dict[str, Any] = {"repository_root": str(repo)}
    try:
        head = _command(["git", "-C", str(repo), "rev-parse", "HEAD"])
        if head.returncode == 0:
            result["git_head"] = head.stdout.strip()
        status = _command(["git", "-C", str(repo), "status", "--porcelain"])
        if status.returncode == 0:
            result["git_dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        result["git_error"] = str(error)
    return result


def base_environment(device: int = 0) -> dict[str, Any]:
    """Run all mandatory preflights before any worker/CUDA initialization."""

    memory = host_memory_check()
    idle = require_idle_gpu(device)
    snapshot = nvidia_snapshot(device)
    return {
        "captured_at_utc": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pid": os.getpid(),
        "host_memory": memory,
        "preflight": idle,
        "initial_gpu": snapshot,
        "source": source_revision(),
    }


def selected_clock_hz(environment: dict[str, Any]) -> float:
    snapshot = environment.get("initial_gpu", {})
    for key in ("max_clock_mhz", "clock_mhz"):
        value = snapshot.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value) * 1_000_000.0
    return 0.0


def collect_versions(modules: Iterable[Any]) -> dict[str, str | None]:
    return {
        module.__name__: getattr(module, "__version__", None)
        for module in modules
    }
