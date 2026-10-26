"""Lexical and resolved-path defenses for per-run compiler caches."""

from __future__ import annotations

import os
from pathlib import Path
import re

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_BENCHMARK_ID = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


class CachePathError(ValueError):
    pass


def run_cache_path(cache_root: str | os.PathLike[str], run_id: str) -> Path:
    """Resolve root/run_id and reject traversal, absolute IDs, and symlinks out."""

    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise CachePathError(f"Unsafe calibration run ID: {run_id!r}")
    root_text = os.fspath(cache_root)
    if not root_text or not Path(root_text).is_absolute():
        raise CachePathError("Calibration cache root must be a non-empty absolute path")
    root = Path(root_text).resolve(strict=False)
    candidate = root / run_id
    resolved = candidate.resolve(strict=False)
    if resolved.parent != root or resolved.name != run_id:
        raise CachePathError(f"Run cache escapes its root: {candidate}")
    return resolved


def task_cache_path(run_cache: str | os.PathLike[str], benchmark: str) -> Path:
    if not isinstance(benchmark, str) or not _BENCHMARK_ID.fullmatch(benchmark) or benchmark in {".", ".."}:
        raise CachePathError(f"Unsafe benchmark cache name: {benchmark!r}")
    root = Path(run_cache).resolve(strict=False)
    task = (root / "tasks" / benchmark).resolve(strict=False)
    if task.parent != root / "tasks" or task.name != benchmark:
        raise CachePathError("Task cache escaped the run-specific tasks directory")
    return task


def worker_result_path(run_cache: str | os.PathLike[str], benchmark: str) -> Path:
    task = task_cache_path(run_cache, benchmark)
    result = (task / "result.json").resolve(strict=False)
    if result.parent != task:
        raise CachePathError("Worker result path escaped its task cache")
    return result


def artifact_dir_path(run_cache: str | os.PathLike[str], benchmark: str) -> Path:
    """Return the checked per-task directory for exported CUDA/PTX/SASS files."""

    task = task_cache_path(run_cache, benchmark)
    artifacts = (task / "artifacts").resolve(strict=False)
    if artifacts.parent != task:
        raise CachePathError("Instruction artifact directory escaped its task cache")
    return artifacts


def cache_environment(task_cache: str | os.PathLike[str]) -> dict[str, str]:
    """Return per-task cache variables; callers create these directories."""

    task = Path(task_cache).resolve(strict=False)
    paths = {
        "TILELANG_CACHE_DIR": task / "tilelang",
        "CUDA_CACHE_PATH": task / "nvidia",
        "TORCH_EXTENSIONS_DIR": task / "torch-extensions",
        "TMPDIR": task / "tmp",
    }
    return {key: str(value) for key, value in paths.items()}
