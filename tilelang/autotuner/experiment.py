"""Safety-oriented run directories and cache tracks for guided experiments.

This module is intentionally independent of TileLang compiler imports so its
path-safety protocol can be tested on a plain CPU Python installation.  It
never uses TileLang's default ``~/.tilelang`` cache: every execution is placed
under an explicitly created per-arm/per-repetition directory.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import copy
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from typing import Any
import uuid


_ROOT_MANIFEST = ".g4-search-experiment.json"
_RUN_MANIFEST = ".g4-run.json"
_SCHEMA = "tilelang.g4.search-experiment/v1"
_ARM_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ENV_LOCK = threading.Lock()


class ExperimentSafetyError(RuntimeError):
    """A run/cache directory failed an ownership or containment check."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strict_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        _strict_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(payload + "\n")
        output.flush()
        os.fsync(output.fileno())


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ExperimentSafetyError(f"Manifest must contain a JSON object: {path}")
    return value


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _inventory(root: Path, *, exclude_after: bool = False) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    entries = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if exclude_after and relative == "after.json":
            continue
        item: dict[str, Any] = {"path": relative}
        if path.is_symlink():
            item.update({"type": "symlink", "target": os.readlink(path)})
        elif path.is_dir():
            item["type"] = "directory"
        elif path.is_file():
            item.update({"type": "file", "size_bytes": path.stat().st_size})
        else:
            item["type"] = "other"
        entries.append(item)
    return entries


@dataclass
class ExperimentResult:
    """Python result plus the detached pure-JSON summary and evidence paths."""

    result: Any
    summary: dict[str, Any]
    run_directory: Path
    events_path: Path
    before_path: Path
    after_path: Path
    warmup_result: Any = None


class SearchExperiment:
    """Create and optionally clean isolated guided-search experiment runs.

    Every ``run`` uses ``<root>/<arm>__rep-NNNN/`` with separate ``events.jsonl``
    and ``cache/``.  ``before.json`` records the required empty starting state;
    ``after.json`` inventories all resulting entries (except itself).

    Cache tracks:
        * ``enabled``: one cold run with stock caches enabled;
        * ``no-cache``: one run with global and autotune caches disabled;
        * ``warm``: a prewarm run followed by a fresh tuner instance's measured
          run in the same isolated cache directory.

    Environment changes are process-wide, so executions holding this module's
    environment lock are serialized and original values are restored.  No
    method automatically deletes a failed or unfamiliar directory.  Cleanup is
    restricted to immediate child directories carrying this root experiment's
    manifest, a matching experiment id, and a matching canonical run name.
    """

    def __init__(self, root: str | Path):
        self.root = self._safe_root(Path(root).expanduser())
        self.experiment_id = uuid.uuid4().hex
        self._root_ready = False

    @staticmethod
    def _safe_root(path: Path) -> Path:
        resolved = path.resolve(strict=False)
        home = Path.home().resolve()
        default_tilelang = home / ".tilelang"
        anchor = Path(resolved.anchor)
        if resolved in (anchor, home):
            raise ExperimentSafetyError(f"Refusing filesystem root or home as an experiment root: {resolved}")
        if _is_within(resolved, default_tilelang) or _is_within(default_tilelang, resolved):
            raise ExperimentSafetyError(f"Refusing to use or traverse TileLang's default cache: {resolved}")
        return resolved

    @staticmethod
    def _run_name(arm: str, repetition: int) -> str:
        if not isinstance(arm, str) or not _ARM_NAME.fullmatch(arm):
            raise ValueError("arm must be a non-empty filename-safe name")
        if isinstance(repetition, bool) or not isinstance(repetition, int) or repetition < 0:
            raise ValueError("repetition must be a non-negative integer")
        return f"{arm}__rep-{repetition:04d}"

    def _ensure_root(self) -> None:
        if self._root_ready:
            return
        self.root = self._safe_root(self.root)
        if self.root.exists() and (self.root.is_symlink() or not self.root.is_dir()):
            raise ExperimentSafetyError(f"Experiment root is not a real directory: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / _ROOT_MANIFEST
        if manifest_path.exists():
            manifest = _read_json(manifest_path)
            if (
                manifest.get("schema") != _SCHEMA
                or manifest.get("kind") != "root"
                or manifest.get("root") != str(self.root)
                or not isinstance(manifest.get("experiment_id"), str)
            ):
                raise ExperimentSafetyError(f"Unfamiliar or invalid experiment root manifest: {manifest_path}")
            self.experiment_id = manifest["experiment_id"]
        else:
            _write_json(
                manifest_path,
                {
                    "schema": _SCHEMA,
                    "kind": "root",
                    "root": str(self.root),
                    "experiment_id": self.experiment_id,
                    "metadata": {"utc_created": _utc_now()},
                },
            )
        self._root_ready = True

    def _prepare(self, arm: str, repetition: int, cache_mode: str) -> tuple[Path, Path]:
        self._ensure_root()
        name = self._run_name(arm, repetition)
        run_dir = self.root / name
        if run_dir.is_symlink():
            raise ExperimentSafetyError(f"Refusing symlink run directory: {run_dir}")
        if run_dir.exists():
            if not run_dir.is_dir():
                raise ExperimentSafetyError(f"Run path exists and is not a directory: {run_dir}")
            existing = list(run_dir.iterdir())
            if existing:
                raise ExperimentSafetyError(f"Refusing non-empty run directory (familiar or unfamiliar): {run_dir}")
            existed = True
        else:
            run_dir.mkdir()
            existed = False

        # This is evidence about the state before manifest/log/cache creation.
        _write_json(
            run_dir / "before.json",
            {
                "schema": _SCHEMA,
                "kind": "before",
                "run_name": name,
                "existed": existed,
                "entries": [],
                "cache_entries": [],
                "empty": True,
                "metadata": {"perf_counter": time.perf_counter(), "utc_recorded": _utc_now()},
            },
        )
        _write_json(
            run_dir / _RUN_MANIFEST,
            {
                "schema": _SCHEMA,
                "kind": "run",
                "root": str(self.root),
                "experiment_id": self.experiment_id,
                "run_name": name,
                "arm": arm,
                "repetition": repetition,
                "cache_mode": cache_mode,
                "metadata": {"utc_created": _utc_now()},
            },
        )
        cache_dir = run_dir / "cache"
        cache_dir.mkdir()
        return run_dir, cache_dir

    def _after(self, run_dir: Path) -> None:
        entries = _inventory(run_dir, exclude_after=True)
        cache_entries = [entry for entry in entries if entry["path"] == "cache" or entry["path"].startswith("cache/")]
        _write_json(
            run_dir / "after.json",
            {
                "schema": _SCHEMA,
                "kind": "after",
                "run_name": run_dir.name,
                "entries": entries,
                "cache_entries": cache_entries,
                "inventory_excludes": ["after.json"],
                "metadata": {"perf_counter": time.perf_counter(), "utc_recorded": _utc_now()},
            },
        )

    @contextmanager
    def _cache_environment(self, cache_dir: Path, cache_mode: str):
        names = ("TILELANG_CACHE_DIR", "TILELANG_DISABLE_CACHE", "TILELANG_AUTO_TUNING_DISABLE_CACHE")
        with _ENV_LOCK:
            previous = {name: os.environ.get(name) for name in names}
            # Avoid importing TileLang merely to manage paths.  If its environment
            # is already active, also restore programmatic CacheState changes.
            tilelang_env = sys.modules.get("tilelang.env")
            cache_state = getattr(tilelang_env, "CacheState", None) if tilelang_env is not None else None
            previous_state = getattr(cache_state, "_enabled", None) if cache_state is not None else None
            os.environ["TILELANG_CACHE_DIR"] = str(cache_dir)
            disabled = "1" if cache_mode == "no-cache" else "0"
            os.environ["TILELANG_DISABLE_CACHE"] = disabled
            os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] = disabled
            if cache_state is not None:
                if cache_mode == "no-cache":
                    cache_state.disable()
                else:
                    cache_state.enable()
            try:
                yield
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
                if cache_state is not None and previous_state is not None:
                    if previous_state:
                        cache_state.enable()
                    else:
                        cache_state.disable()

    def _validate_manifest(self, run_dir: Path) -> dict[str, Any]:
        self._ensure_root()
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise ExperimentSafetyError(f"Not a real run directory: {run_dir}")
        resolved = run_dir.resolve()
        if resolved.parent != self.root:
            raise ExperimentSafetyError(f"Run directory must be an immediate child of {self.root}: {resolved}")
        manifest = _read_json(run_dir / _RUN_MANIFEST)
        expected_name = self._run_name(manifest.get("arm"), manifest.get("repetition"))
        if (
            manifest.get("schema") != _SCHEMA
            or manifest.get("kind") != "run"
            or manifest.get("root") != str(self.root)
            or manifest.get("experiment_id") != self.experiment_id
            or manifest.get("run_name") != run_dir.name
            or expected_name != run_dir.name
            or manifest.get("cache_mode") not in ("enabled", "no-cache", "warm")
        ):
            raise ExperimentSafetyError(f"Run manifest ownership/name mismatch; refusing cleanup: {run_dir}")
        return manifest

    def cleanup_run(self, run_dir: str | Path) -> Path:
        """Delete one manifest-owned, canonically named run directory only."""
        path = Path(run_dir).expanduser()
        if not path.is_absolute():
            path = self.root / path
        self._validate_manifest(path)
        resolved = path.resolve()
        shutil.rmtree(resolved)
        return resolved

    def cleanup(self) -> list[Path]:
        """Delete owned runs, leaving every unfamiliar entry untouched."""
        self._ensure_root()
        removed = []
        for path in list(self.root.iterdir()):
            if path.is_symlink() or not path.is_dir():
                continue
            manifest_path = path / _RUN_MANIFEST
            if not manifest_path.exists():
                continue
            manifest = _read_json(manifest_path)
            if manifest.get("experiment_id") != self.experiment_id:
                continue
            # An owned id with a mismatched name is evidence of tampering, not
            # permission to remove the directory.
            self._validate_manifest(path)
            removed.append(self.cleanup_run(path))
        return removed

    @staticmethod
    def _experiment_summary(summary: Mapping[str, Any], run_dir: Path, arm: str, repetition: int, cache_mode: str) -> dict:
        result = copy.deepcopy(dict(summary))
        result["experiment"] = {
            "arm": arm,
            "repetition": repetition,
            "cache_mode": cache_mode,
            "run_directory": str(run_dir),
            "before": "before.json",
            "after": "after.json",
            "events": "events.jsonl",
        }
        cache = result.get("cache")
        if isinstance(cache, dict):
            cache["requested_mode"] = cache_mode
            observed_directory = cache.get("directory")
            if observed_directory is not None:
                observed = Path(observed_directory).resolve(strict=False)
                expected_cache = (run_dir / "cache").resolve()
                if not _is_within(observed, expected_cache):
                    raise ExperimentSafetyError(
                        f"Tuner reported a cache outside the experiment directory: {observed}; expected under {expected_cache}"
                    )
        return _strict_json(result)

    def run(
        self,
        tuner_factory: Callable[[], Any],
        *,
        arm: str,
        repetition: int,
        cache_mode: str = "enabled",
        run_kwargs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExperimentResult:
        """Execute one arm/repetition and write summary plus before/after evidence.

        ``tuner_factory`` must return a fresh ``GuidedAutoTuner``-compatible
        object with ``run(**kwargs)`` and ``last_summary``.  A factory rather
        than a tuner instance is required so warm measurements do not reuse the
        prewarm instance's in-memory result.  Exceptions are recorded in
        ``failure.json``/``after.json`` and re-raised, never silently converted
        to a successful arm.
        """
        if not callable(tuner_factory):
            raise TypeError("tuner_factory must be callable")
        if cache_mode not in ("enabled", "no-cache", "warm"):
            raise ValueError("cache_mode must be 'enabled', 'no-cache', or 'warm'")
        kwargs = dict(run_kwargs or {})
        if "jsonl_path" in kwargs:
            raise ValueError("jsonl_path is managed by SearchExperiment and cannot be overridden")
        base_metadata = dict(kwargs.pop("metadata", {}) or {})
        base_metadata.update(dict(metadata or {}))
        base_metadata.update({"arm": arm, "repetition": repetition, "cache_mode": cache_mode})

        run_dir, cache_dir = self._prepare(arm, repetition, cache_mode)
        active_tuner = None
        summary = None
        result = None
        warmup_result = None
        warmup_tuner = None

        def invoke(events_name: str, fresh_from=None):
            nonlocal active_tuner, summary
            active_tuner = None
            candidate_tuner = tuner_factory()
            if fresh_from is not None and candidate_tuner is fresh_from:
                raise ExperimentSafetyError("warm cache measurement requires tuner_factory to return a fresh tuner instance")
            active_tuner = candidate_tuner
            call_kwargs = dict(kwargs)
            call_kwargs["jsonl_path"] = run_dir / events_name
            call_kwargs["metadata"] = base_metadata
            outcome = active_tuner.run(**call_kwargs)
            summary = getattr(active_tuner, "last_summary", None)
            if not isinstance(summary, dict):
                raise RuntimeError("Tuner did not provide a pure-JSON last_summary dictionary")
            return outcome

        try:
            with self._cache_environment(cache_dir, cache_mode):
                if cache_mode == "warm":
                    warmup_result = invoke("warmup.jsonl")
                    warmup_tuner = active_tuner
                    warmup_summary = self._experiment_summary(summary, run_dir, arm, repetition, "warm-prewarm")
                    _write_json(run_dir / "warmup_summary.json", warmup_summary)
                    summary = None
                result = invoke("events.jsonl", fresh_from=warmup_tuner)
            summary = self._experiment_summary(summary, run_dir, arm, repetition, cache_mode)
            _write_json(run_dir / "summary.json", summary)
        except BaseException as exc:
            if summary is None and active_tuner is not None:
                candidate_summary = getattr(active_tuner, "last_summary", None)
                summary = candidate_summary if isinstance(candidate_summary, dict) else None
            if summary is not None:
                try:
                    failure_summary = self._experiment_summary(summary, run_dir, arm, repetition, cache_mode)
                except Exception as summary_error:
                    # Preserve the original failure and after-evidence even when
                    # the tuner's own summary is what violated the safety rule.
                    failure_summary = copy.deepcopy(summary)
                    failure_summary["experiment"] = {
                        "arm": arm,
                        "repetition": repetition,
                        "cache_mode": cache_mode,
                        "run_directory": str(run_dir),
                    }
                    failure_summary["summary_error"] = f"{type(summary_error).__name__}: {summary_error}"
                failure_summary["status"] = "error"
                _write_json(run_dir / "summary.json", _strict_json(failure_summary))
            _write_json(
                run_dir / "failure.json",
                {
                    "schema": _SCHEMA,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "metadata": {"utc_recorded": _utc_now()},
                },
            )
            self._after(run_dir)
            raise
        else:
            self._after(run_dir)

        return ExperimentResult(
            result=result,
            summary=copy.deepcopy(summary),
            run_directory=run_dir,
            events_path=run_dir / "events.jsonl",
            before_path=run_dir / "before.json",
            after_path=run_dir / "after.json",
            warmup_result=warmup_result,
        )


__all__ = ["ExperimentResult", "ExperimentSafetyError", "SearchExperiment"]
