"""Opt-in structured observations for autotuning experiments.

The legacy :class:`TrialReporter` API is intentionally unchanged for callers
that only use ``AutoTuner.run(on_trial=...)``.  The event classes in this module
are used by the explicit guided-tuning API.  Event timestamps come from
:func:`time.perf_counter`; UTC timestamps, when present, are metadata only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any


class ObserverError(RuntimeError):
    """An explicitly requested structured-observer sink failed."""


def _detached_json(value: Any) -> Any:
    """Return strict, detached JSON data and reject NaN/infinity."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _config_id(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EventReporter:
    """Small internal dispatcher which is a no-op without an observer.

    Payload construction and ``perf_counter`` calls happen only when enabled.
    Serialization or sink failures are logged and disable this dispatcher so
    an observer cannot turn a worker-thread reporting problem into a fake
    compile/profile failure.  ``GuidedAutoTuner`` additionally surfaces external
    sink failures as :class:`ObserverError` at the end of the run.
    """

    def __init__(self, sink: Callable[[dict], None] | None, logger: logging.Logger):
        self.sink = sink
        self.logger = logger
        self.errors: list[str] = []

    @property
    def enabled(self) -> bool:
        return self.sink is not None

    def emit(self, event: str, *, index: int | None = None, config: Mapping[str, Any] | None = None, **payload: Any) -> None:
        if self.sink is None:
            return
        try:
            record = {"event": event, "perf_counter": time.perf_counter()}
            if index is not None:
                record["index"] = index
            if config is not None:
                detached_config = _detached_json(config)
                record["config"] = detached_config
                record["config_id"] = _config_id(detached_config)
            record.update(payload)
            record = _detached_json(record)
            self.sink(record)
        except Exception as exc:
            self.errors.append(str(exc))
            self.sink = None
            self.logger.warning("Structured autotune observation failed; disabling this event dispatcher.", exc_info=True)


class JsonlObserver:
    """Thread-safe strict JSON Lines observer.

    Every input event is written as one canonical JSON object with a monotonic
    sequence number.  The file is flushed after every record so a failed run
    leaves useful evidence.  The observer never adds wall-clock timestamps to
    events; ``perf_counter`` remains the timing source.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()
        self._sequence = 0
        self._closed = False

    def __call__(self, event: dict) -> None:
        with self._lock:
            if self._closed:
                raise ObserverError(f"JSONL observer is closed: {self.path}")
            record = dict(event)
            record["sequence"] = self._sequence
            payload = json.dumps(
                _detached_json(record),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            self._file.write(payload + "\n")
            self._file.flush()
            self._sequence += 1

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._file.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class RunRecorder:
    """Aggregate structured events into a pure-JSON run summary.

    Args:
        observer: Optional callable (or object with ``record(event)``) receiving
            detached event dictionaries.  Sink failures are remembered and can
            be raised by ``raise_for_observer_errors``.
        fully_instrumented: False is for adapters around an old, unhooked tuner
            which only supplies legacy ``on_trial`` records.  In that mode
            compile/validation/profile counts and durations remain null with
            explicit reasons instead of being inferred from trial outcomes.
    """

    _PHASES = ("compile", "validation", "profile", "model")

    def __init__(self, observer: Any = None, *, fully_instrumented: bool = True):
        if observer is not None:
            sink = observer if callable(observer) else getattr(observer, "record", None)
            if not callable(sink):
                raise TypeError("observer must be callable or implement record(event)")
            self._observer = sink
        else:
            self._observer = None
        self.fully_instrumented = fully_instrumented
        self.observer_errors: list[str] = []
        self._lock = threading.Lock()
        self._run_start: float | None = None
        self._run_end: float | None = None
        self._status = "running"
        self._context: dict[str, Any] = {}
        self._counts: dict[str, Any] = {}
        self._phase = {name: self._new_phase() for name in self._PHASES}
        self._solver_start: float | None = None
        self._solver_end: float | None = None
        self._solver_audit: dict[str, Any] | None = None
        self._solver_cumulative = 0.0
        self._model_expected = False
        self._score_records = 0
        self._trials: dict[str, int] = {}
        self._observer_lock = threading.Lock()
        self._best_config: dict | None = None
        self._best_config_id: str | None = None
        self._best_latency: float | None = None
        self._cache: dict[str, Any] = {}
        self._cache_hit = False
        self._cache_source: str | None = None
        self._errors: list[dict[str, Any]] = []
        self._errors_omitted = 0

    @staticmethod
    def _new_phase() -> dict[str, Any]:
        return {
            "attempts": 0,
            "success": 0,
            "fail": 0,
            "not_run": 0,
            "ended": 0,
            "first": None,
            "last": None,
            "cumulative": 0.0,
            "open": 0,
        }

    @property
    def cache_hit(self) -> bool:
        return self._cache_hit

    @staticmethod
    def _error_text(error: Any) -> str:
        return str(error)[:4096]

    def _remember_error(self, kind: str, event: Mapping[str, Any]) -> None:
        if event.get("error") is None:
            return
        item = {
            "kind": kind,
            "config_id": event.get("config_id"),
            "index": event.get("index"),
            "error": self._error_text(event["error"]),
        }
        if len(self._errors) < 100:
            self._errors.append(item)
        else:
            self._errors_omitted += 1

    def _phase_start(self, name: str, event: Mapping[str, Any]) -> None:
        phase = self._phase[name]
        now = float(event["perf_counter"])
        phase["attempts"] += 1
        phase["open"] += 1
        if phase["first"] is None:
            phase["first"] = now

    def _phase_end(self, name: str, event: Mapping[str, Any]) -> None:
        phase = self._phase[name]
        now = float(event["perf_counter"])
        status = str(event.get("status", "ok"))
        phase["ended"] += 1
        phase["open"] = max(0, phase["open"] - 1)
        phase["last"] = now if phase["last"] is None else max(phase["last"], now)
        wall = event.get("wall_seconds")
        if isinstance(wall, (int, float)) and not isinstance(wall, bool) and math.isfinite(float(wall)):
            phase["cumulative"] += max(0.0, float(wall))
        if status in ("ok", "passed"):
            phase["success"] += 1
        elif status == "not_run":
            phase["not_run"] += 1
        else:
            phase["fail"] += 1
        self._remember_error(name, event)

    def _record(self, event: dict) -> None:
        name = event.get("event")
        now = event.get("perf_counter")
        if name == "run_start":
            self._run_start = float(now)
            self._context = {key: value for key, value in event.items() if key not in ("event", "perf_counter")}
            self._model_expected = bool(event.get("model_expected", False))
        elif name == "run_end":
            self._run_end = float(now)
            self._status = str(event.get("status", "ok"))
            if event.get("best_config") is not None:
                self._best_config = event["best_config"]
                self._best_config_id = event.get("best_config_id")
            if event.get("best_latency_ms") is not None:
                self._best_latency = float(event["best_latency_ms"])
            self._remember_error("run", event)
        elif name == "config_generation_end":
            self._counts = dict(event)
        elif name == "solver_start":
            self._solver_start = float(now)
        elif name == "solver_end":
            self._solver_end = float(now)
            self._remember_error("solver", event)
        elif name == "solver":
            wall = event.get("wall_seconds")
            if isinstance(wall, (int, float)) and not isinstance(wall, bool) and math.isfinite(float(wall)):
                self._solver_cumulative += max(0.0, float(wall))
        elif name == "prune_end":
            audit = event.get("audit")
            if isinstance(audit, dict):
                self._solver_audit = dict(audit)
            self._remember_error("prune", event)
        elif name == "score":
            self._score_records += 1
        elif name in ("compile_start", "validation_start", "profile_start", "score_start"):
            self._phase_start("model" if name == "score_start" else name[:-6], event)
        elif name in ("compile_end", "validation_end", "profile_end", "score_end"):
            self._phase_end("model" if name == "score_end" else name[:-4], event)
        elif name == "trial":
            status = str(event.get("status", "unknown"))
            self._trials[status] = self._trials.get(status, 0) + 1
            self._remember_error("trial", event)
        elif name == "best":
            self._best_config = event.get("config")
            self._best_config_id = event.get("config_id")
            self._best_latency = event.get("latency_ms")
        elif name == "cache_identity":
            self._cache = dict(event)
        elif name == "cache_hit":
            self._cache_hit = True
            self._cache_source = event.get("source")

    def __call__(self, event: dict) -> None:
        detached = _detached_json(event)
        with self._lock:
            self._record(detached)
        # Sinks such as list.append or a file wrapper need not be thread-safe.
        # Serialize delivery without holding the aggregation lock, so a sink
        # can inspect itself without deadlocking the compile/profile workers.
        with self._observer_lock:
            with self._lock:
                observer = self._observer
            if observer is not None:
                try:
                    observer(detached)
                except Exception as exc:
                    with self._lock:
                        self.observer_errors.append(f"{type(exc).__name__}: {exc}")
                        self._observer = None
                    logging.getLogger(__name__).warning("Structured observer sink failed; it has been disabled.", exc_info=True)

    def record(self, event: dict) -> None:
        """Object-style observer alias for :meth:`__call__`."""
        self(event)

    def raise_for_observer_errors(self) -> None:
        if self.observer_errors:
            raise ObserverError("; ".join(self.observer_errors))

    def _unavailable_phase(self, name: str, reason: str, unavailable: dict[str, str]) -> dict[str, Any]:
        unavailable[name] = reason
        result = {
            "attempts": None,
            "success": None,
            "fail": None,
            "wall_seconds": None,
            "cumulative_seconds": None,
            "unavailable_reason": reason,
        }
        if name == "validation":
            result.update({"passed": None, "failed": None, "not_run": None})
        if name == "model":
            result["predictions"] = None
        return result

    def _phase_summary(self, name: str, unavailable: dict[str, str]) -> dict[str, Any]:
        if not self.fully_instrumented and name != "model":
            return self._unavailable_phase(
                name,
                f"{name} hooks are unavailable from an unhooked legacy tuner",
                unavailable,
            )
        if name == "model" and not self._model_expected:
            return self._unavailable_phase(name, "this arm does not invoke the candidate cost model", unavailable)

        phase = self._phase[name]
        attempts = phase["attempts"]
        unfinished = max(0, attempts - phase["ended"])
        wall = 0.0 if attempts == 0 else None
        reason = None
        if attempts and unfinished == 0 and phase["first"] is not None and phase["last"] is not None:
            wall = max(0.0, phase["last"] - phase["first"])
        elif attempts:
            reason = f"{unfinished} {name} interval(s) did not end; wall time cannot be determined"
            unavailable[f"{name}.wall_seconds"] = reason
        result = {
            "attempts": attempts,
            "success": phase["success"],
            "fail": phase["fail"],
            "unfinished": unfinished,
            "wall_seconds": wall,
            "cumulative_seconds": phase["cumulative"],
            "unavailable_reason": reason,
        }
        if name == "validation":
            result.update(
                {
                    "passed": phase["success"],
                    "failed": phase["fail"],
                    "not_run": phase["not_run"],
                }
            )
        if name == "model":
            if self._counts.get("status") == "ok":
                result["predictions"] = self._score_records
            else:
                prediction_reason = "model scoring did not complete, so the validated prediction count is unavailable"
                result["predictions"] = None
                result["unavailable_reason"] = "; ".join(filter(None, (reason, prediction_reason)))
                unavailable["model.predictions"] = prediction_reason
        return result

    def _solver_summary(self, unavailable: dict[str, str]) -> dict[str, Any]:
        if self._solver_audit is None:
            invoked = self._solver_start is not None
            reason = "this arm does not invoke solver pruning" if self._context.get("arm") == "stock-raw" else "solver pruning did not complete"
            unavailable["solver"] = reason
            wall = None
            if self._solver_start is not None and self._solver_end is not None:
                wall = max(0.0, self._solver_end - self._solver_start)
            return {
                "invoked": invoked,
                "completed": False if invoked else None,
                "checks": None,
                "wall_seconds": wall,
                "cumulative_seconds": None,
                "unavailable_reason": reason,
            }
        audit = self._solver_audit
        wall = audit.get("wall_seconds")
        if wall is None and self._solver_start is not None and self._solver_end is not None:
            wall = max(0.0, self._solver_end - self._solver_start)
        return {
            "invoked": True,
            "completed": True,
            "checks": audit.get("checked"),
            "sat": audit.get("sat"),
            "unsat": audit.get("unsat"),
            "unknown": audit.get("unknown"),
            "timeouts": audit.get("timeouts"),
            "wall_seconds": wall,
            "cumulative_seconds": self._solver_cumulative,
            "unavailable_reason": None,
        }

    def summary(self) -> dict[str, Any]:
        """Return a detached, strict-JSON summary with explicit null reasons."""
        with self._lock:
            unavailable: dict[str, str] = {}
            compile_summary = self._phase_summary("compile", unavailable)
            validation_summary = self._phase_summary("validation", unavailable)
            profile_summary = self._phase_summary("profile", unavailable)
            model_summary = self._phase_summary("model", unavailable)
            solver_summary = self._solver_summary(unavailable)

            raw = self._counts.get("raw", self._context.get("raw"))
            legal = self._counts.get("legal")
            unknown = self._counts.get("unknown")
            duplicates = self._counts.get("duplicates")
            if legal is None:
                reason = "stock-raw does not construct the Z3 legal universe" if self._context.get("arm") == "stock-raw" else "legal-universe count is unavailable because search did not complete"
                legal = None
                unavailable["U"] = reason
            if unknown is None:
                unavailable["unknown"] = "unknown solver verdicts are only defined for a completed pruning pass"
            if duplicates is None:
                duplicates = None
                unavailable["duplicates"] = "duplicate audit is unavailable because config generation did not complete"

            end_to_end = None
            if self._run_start is not None and self._run_end is not None:
                end_to_end = max(0.0, self._run_end - self._run_start)
            else:
                unavailable["end_to_end_wall_seconds"] = "run_start/run_end was not observed"

            cache_key = self._cache.get("cache_key")
            cache_reason = self._cache.get("reason")
            if not self._cache and cache_reason is None:
                cache_reason = "the tuner did not reach cache-key generation"
                unavailable["cache_identity"] = cache_reason
            cache = {
                "identity": cache_key,
                "key": cache_key,
                "directory": self._cache.get("cache_dir"),
                "enabled": self._cache.get("enabled"),
                "hit": self._cache_hit,
                "source": self._cache_source,
                "unavailable_reason": cache_reason,
            }
            best_reason = None
            if self._best_config is None:
                best_reason = "no successful best trial was observed"
                unavailable["best"] = best_reason
            best = {
                "config": self._best_config,
                "config_id": self._best_config_id,
                "latency_ms": self._best_latency,
                "unavailable_reason": best_reason,
            }
            summary = {
                "schema_version": 1,
                "status": self._status,
                "arm": self._context.get("arm"),
                "metadata": self._context.get("metadata", {}),
                "R": raw,
                "U": legal,
                "unknown": unknown,
                "duplicates": duplicates,
                "selected": self._counts.get("selected"),
                "candidate_count": self._counts.get("selected"),
                "compile_attempts": compile_summary["attempts"],
                "compile_success": compile_summary["success"],
                "compile_fail": compile_summary["fail"],
                "compile": compile_summary,
                "validation": validation_summary,
                "profile": profile_summary,
                "solver": solver_summary,
                "model": model_summary,
                "solver_wall_seconds": solver_summary["wall_seconds"],
                "model_wall_seconds": model_summary["wall_seconds"],
                "compile_wall_seconds": compile_summary["wall_seconds"],
                "profile_wall_seconds": profile_summary["wall_seconds"],
                "validation_wall_seconds": validation_summary["wall_seconds"],
                "end_to_end_wall_seconds": end_to_end,
                "best": best,
                "best_config": self._best_config,
                "best_latency_ms": self._best_latency,
                "cache_identity": cache_key,
                "cache": cache,
                "trials": dict(sorted(self._trials.items())),
                "errors": list(self._errors),
                "errors_omitted": self._errors_omitted,
                "observer_errors": list(self.observer_errors),
                "fully_instrumented": self.fully_instrumented,
                "unavailable": dict(sorted(unavailable.items())),
            }
            return _detached_json(summary)


class TrialReporter:
    """Invoke the legacy ``on_trial`` sink synchronously.

    Records contain detached JSON representations of configs, not arbitrary
    objects.  The legacy sink must not mutate/re-enter the tuner.  As before,
    an ordinary legacy sink error disables that sink without failing tuning.
    """

    def __init__(self, sink: Callable[[dict], None] | None, logger: logging.Logger, events: EventReporter | None = None):
        self.sink = sink
        self.logger = logger
        self.events = events

    def emit(self, status, index=None, config=None, latency=None, validation="not_run", error=None):
        if self.events is not None:
            self.events.emit(
                "trial",
                index=index,
                config=config,
                status=status,
                latency_ms=latency,
                validation=validation,
                error=self._error_text(error) if error is not None else None,
            )
        if self.sink is None:
            return
        try:
            record = {
                "status": status,
                "index": index,
                "config": json.loads(json.dumps(config, allow_nan=False)),
                "latency_ms": latency,
                "validation": validation,
                "error": self._error_text(error) if error is not None else None,
            }
            self.sink(record)
        except Exception:
            self.sink = None
            self.logger.warning("Trial reporting failed; disabling the sink for this run.", exc_info=True)

    @staticmethod
    def _error_text(error: Any) -> str:
        return str(error)[:2048]
