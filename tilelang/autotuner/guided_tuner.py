"""Explicit GuidedSearch integration with the stock TileLang autotuner.

``GuidedAutoTuner`` only selects configurations and adds structured evidence.
Compilation, correctness validation, profiling, early-stop validation, and the
stock result cache are still executed by :class:`tilelang.autotuner.AutoTuner`.
The default ``AutoTuner`` API and its exhaustive list-of-configs behavior are
not changed by this module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import copy
import json
import logging
from pathlib import Path
import time
from typing import Any, Literal

from .guided import CandidateCostModel, GuidedSearch, _is_capacity_error
from .report import EventReporter, JsonlObserver, RunRecorder, _config_id
from .space import ConfigSpace, PruneResult, Z3Pruner
from .tuner import AutoTuner


GuidedArm = Literal["stock-raw", "z3-legal-exhaustive", "z3-cm-topk", "budget-raw", "budget-random"]
logger = logging.getLogger(__name__)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


class _ObservedPruner(Z3Pruner):
    """Delegate pruning while exposing the existing audit as event records."""

    def __init__(self, delegate: Z3Pruner, events: EventReporter):
        self.delegate = delegate
        self.events = events

    def prune(self, configs, constraint_fn=None) -> PruneResult:
        self.events.emit("solver_start")
        self.events.emit("prune_start")
        started = time.perf_counter()
        try:
            result = self.delegate.prune(configs, constraint_fn=constraint_fn)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self.events.emit("prune_end", status="error", error=str(exc), wall_seconds=elapsed)
            self.events.emit("solver_end", status="error", error=str(exc), wall_seconds=elapsed)
            raise

        elapsed = time.perf_counter() - started
        for record in result.audit["records"]:
            status = record["status"]
            if status != "duplicate":
                self.events.emit(
                    "solver",
                    index=record["index"],
                    config=record["config"],
                    status=status,
                    reason=record["reason"],
                    wall_seconds=record["wall_seconds"],
                    stats=record["stats"],
                    timeout=record["timeout"],
                    resource_limited=record["resource_limited"],
                    timing_source="Z3Pruner.audit",
                )
            if status == "unsat":
                self.events.emit("prune", index=record["index"], config=record["config"], action="pruned", reason=record["reason"])
            elif status == "unknown":
                self.events.emit("prune", index=record["index"], config=record["config"], action="retained_unknown", reason=record["reason"])
            elif status == "duplicate":
                self.events.emit(
                    "prune",
                    index=record["index"],
                    config=record["config"],
                    action="duplicate",
                    duplicate_of=record["duplicate_of"],
                )

        # Per-config records were already emitted above.  Keeping them out of
        # this aggregate event avoids duplicating a potentially large audit.
        audit = {key: copy.deepcopy(value) for key, value in result.audit.items() if key != "records"}
        self.events.emit("prune_end", status="ok", audit=audit, wall_seconds=elapsed)
        self.events.emit(
            "solver_end",
            status="ok",
            checks=result.audit["checked"],
            wall_seconds=elapsed,
        )
        return result


class _ObservedCostModel:
    """Time model calls without coercing or hiding invalid predictions."""

    def __init__(self, delegate: CandidateCostModel, events: EventReporter):
        self.delegate = delegate
        self.events = events

    def predict(self, config):
        self.events.emit("score_start", config=config)
        started = time.perf_counter()
        try:
            prediction = self.delegate.predict(config)
        except Exception as exc:
            capacity_unknown = _is_capacity_error(exc)
            self.events.emit(
                "score_end",
                config=config,
                status="unknown" if capacity_unknown else "error",
                error=str(exc),
                error_type="capacity" if capacity_unknown else "model",
                wall_seconds=time.perf_counter() - started,
            )
            raise
        self.events.emit(
            "score_end",
            config=config,
            status="ok",
            wall_seconds=time.perf_counter() - started,
        )
        return prediction


class GuidedAutoTuner(AutoTuner):
    """Tune GuidedSearch-selected candidates through the stock AutoTuner path.

    Args:
        fn: TileLang kernel function accepted by :class:`AutoTuner`.
        space: Finite ordered configuration space.
        model: Candidate cost model.  It is called only by ``z3-cm-topk``.
        top_k: Candidate cap for ``z3-cm-topk`` and the default budget for the
            two budget control arms.  ``topK`` is accepted as an explicit alias.
        arm: ``stock-raw`` (R, no solver/model), ``z3-legal-exhaustive`` (U),
            ``z3-cm-topk`` (GuidedSearch.top_k), ``budget-raw`` or
            ``budget-random`` (equal-budget controls from GuidedSearch).
        budget: Optional non-negative control-arm budget; defaults to ``top_k``.
        seed: Seed forwarded to GuidedSearch and the default Z3Pruner.
        pruner: Optional custom Z3Pruner, primarily for backend/test injection.

    Each instance has an isolated in-memory autotune-result cache so separate
    experiment arms/repetitions cannot contaminate one another through the
    process-global memory cache.  The stock disk cache and cache key are reused.
    ``run`` returns the ordinary ``AutotuneResult`` and sets ``last_summary`` to
    a strict JSON-compatible dictionary.  Use ``run_with_summary`` when both
    values are wanted directly.

    Structured observers and ``early_stop=True`` are deliberately incompatible:
    the stock early-stop estimate cannot yet be distinguished from a full
    measured latency.  This explicit tuner always records a summary, so it
    rejects ``early_stop=True`` even when no external observer is supplied.
    """

    ARMS = ("stock-raw", "z3-legal-exhaustive", "z3-cm-topk", "budget-raw", "budget-random")

    def __init__(
        self,
        fn: Callable,
        space: ConfigSpace,
        model: CandidateCostModel,
        top_k: int | None = None,
        *,
        arm: GuidedArm = "z3-cm-topk",
        budget: int | None = None,
        seed: int = 42,
        pruner: Z3Pruner | None = None,
        topK: int | None = None,
    ):
        if not isinstance(space, ConfigSpace):
            raise TypeError("space must be a ConfigSpace")
        if not callable(getattr(model, "predict", None)):
            raise TypeError("model must implement predict(config)")
        if top_k is None:
            top_k = topK
        elif topK is not None:
            raise ValueError("Specify only one of top_k and topK")
        if top_k is None:
            raise ValueError("top_k is required")
        self.top_k = _nonnegative_int(top_k, "top_k")
        self.topK = self.top_k
        self.budget = self.top_k if budget is None else _nonnegative_int(budget, "budget")
        if arm not in self.ARMS:
            raise ValueError(f"arm must be one of {self.ARMS}, got {arm!r}")
        if pruner is not None and not isinstance(pruner, Z3Pruner):
            raise TypeError("pruner must be a Z3Pruner or None")

        self.space = space
        self.model = model
        self.arm = arm
        self.seed = seed
        self.pruner = pruner
        self.search: GuidedSearch | None = None
        self.last_summary: dict[str, Any] | None = None
        self._unknown_counts: dict[str, int | None] | None = None
        super().__init__(fn, configs=[])
        # Assignment intentionally shadows the stock class-level dictionary.
        self._memory_cache = {}

    @classmethod
    def from_kernel(
        cls,
        kernel: Callable,
        space: ConfigSpace,
        model: CandidateCostModel,
        top_k: int | None = None,
        **kwargs,
    ):
        """Named constructor mirroring ``AutoTuner.from_kernel``."""
        return cls(kernel, space=space, model=model, top_k=top_k, **kwargs)

    def _generate_candidates(self, events: EventReporter) -> list[dict[str, Any]]:
        events.emit("config_generation_start", arm=self.arm)
        started = time.perf_counter()
        raw_configs = self.space.raw_configs()
        raw_ids = set()
        for index, config in enumerate(raw_configs):
            raw_ids.add(ConfigSpace.config_id(config))
            events.emit("config_generated", phase="raw", index=index, config=config)

        legal = None
        pruner_unknown = None
        model_unknown = None
        unknown_overlap = None
        unknown_total = None
        duplicates = len(raw_configs) - len(raw_ids)
        if self.arm == "stock-raw":
            selected = raw_configs
            self.search = None
        else:
            base_pruner = self.pruner if self.pruner is not None else Z3Pruner(seed=self.seed)
            observed_pruner = _ObservedPruner(base_pruner, events)
            observed_model = _ObservedCostModel(self.model, events)
            search = GuidedSearch(self.space, observed_model, pruner=observed_pruner, seed=self.seed)
            self.search = search
            legal = search.legal_configs()
            for index, config in enumerate(legal):
                events.emit("config_generated", phase="legal", index=index, config=config)
            pruner_unknown_configs = search.unknown_configs()
            pruner_unknown = len(pruner_unknown_configs)
            duplicates = search.pruning_audit()["duplicates"]

            if self.arm == "z3-legal-exhaustive":
                selected = legal
            elif self.arm == "z3-cm-topk":
                # topK is deliberately produced by GuidedSearch itself, not by
                # re-sorting or partially scoring candidates in this wrapper.
                selected = search.top_k(self.top_k)
                predictions = search.predictions()
                model_unknown_configs = search.model_unknown_configs()
                model_unknown = len(model_unknown_configs)
                pruner_unknown_ids = {ConfigSpace.config_id(config) for config in pruner_unknown_configs}
                model_unknown_ids = {ConfigSpace.config_id(config) for config in model_unknown_configs}
                unknown_overlap = len(pruner_unknown_ids & model_unknown_ids)
                unknown_total = len(pruner_unknown_ids | model_unknown_ids)
                for record in predictions:
                    events.emit(
                        "score",
                        index=record["universe_index"],
                        config=record["config"],
                        rank=record["rank"],
                        score=record["score"],
                        terms=record["terms"],
                        unknown=record.get("unknown", False),
                        error=record.get("error"),
                        error_type=record.get("error_type"),
                    )
            elif self.arm == "budget-raw":
                selected = search.candidates(self.budget, "raw")
            else:
                selected = search.candidates(self.budget, "random")

        for index, config in enumerate(selected):
            events.emit("config_generated", phase="selected", index=index, config=config)
        self._unknown_counts = {
            "pruner": pruner_unknown,
            "model": model_unknown,
            "overlap": unknown_overlap,
            "total": unknown_total,
        }
        events.emit(
            "config_generation_end",
            status="ok",
            raw=len(raw_configs),
            legal=len(legal) if legal is not None else None,
            unknown=pruner_unknown,
            pruner_unknown=pruner_unknown,
            model_unknown=model_unknown,
            unknown_total=unknown_total,
            unknown_sources=copy.deepcopy(self._unknown_counts),
            duplicates=duplicates,
            selected=len(selected),
            wall_seconds=time.perf_counter() - started,
        )
        return selected

    def _recorded_summary(self, recorder: RunRecorder) -> dict[str, Any]:
        """Add source-specific unknown counts unavailable to the generic recorder."""
        summary = recorder.summary()
        counts = self._unknown_counts or {"pruner": None, "model": None, "overlap": None, "total": None}
        summary["pruner_unknown"] = counts["pruner"]
        summary["model_unknown"] = counts["model"]
        summary["unknown_total"] = counts["total"]
        summary["unknown_sources"] = copy.deepcopy(counts)
        summary["model"]["unknown"] = counts["model"]
        if counts["total"] is None:
            summary["unavailable"]["unknown_total"] = (
                "the deduplicated pruner/model unknown total is defined after successful z3-cm-topk scoring"
            )
        return summary

    def run(
        self,
        warmup: int = 25,
        rep: int = 100,
        timeout: int = 180,
        use_pipeline: bool = False,
        enable_grouped_compile: bool = False,
        group_compile_size: int = 2,
        benchmark_devices: list[int] | None = None,
        benchmark_multi_gpu: bool = False,
        early_stop: bool = False,
        early_stop_factor: float = 2.0,
        on_trial: Callable[[dict], None] | None = None,
        *,
        observer: Any = None,
        jsonl_path: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        """Run one explicitly observed guided arm using stock tuning machinery.

        ``observer`` receives structured dictionaries synchronously and may be
        a callable or an object with ``record(event)``.  ``jsonl_path`` writes
        the same events as canonical JSONL.  Observer failures are not silent:
        tuning workers are protected, then ``ObserverError`` is raised after
        the run and recorded in ``last_summary['observer_errors']``.
        """
        jsonl_observer = JsonlObserver(jsonl_path) if jsonl_path is not None else None
        sinks = []
        if jsonl_observer is not None:
            sinks.append(jsonl_observer)
        if observer is not None:
            sink = observer if callable(observer) else getattr(observer, "record", None)
            if not callable(sink):
                if jsonl_observer is not None:
                    jsonl_observer.close()
                raise TypeError("observer must be callable or implement record(event)")
            sinks.append(sink)

        def fanout(event):
            for sink in sinks:
                sink(event)

        recorder = RunRecorder(fanout if sinks else None)
        self._unknown_counts = None
        events = EventReporter(recorder, logger)
        previous_observer = getattr(self, "_event_observer", None)
        self._event_observer = recorder
        self._last_event_reporter = None
        utc_started = datetime.now(timezone.utc).isoformat()
        event_metadata = dict(metadata or {})
        event_metadata["utc_started"] = utc_started
        result = None
        try:
            events.emit(
                "run_start",
                arm=self.arm,
                raw=len(self.space),
                top_k=self.top_k,
                budget=self.budget,
                model_expected=self.arm == "z3-cm-topk",
                metadata=event_metadata,
            )
            self.configs = self._generate_candidates(events)
            result = super().run(
                warmup=warmup,
                rep=rep,
                timeout=timeout,
                use_pipeline=use_pipeline,
                enable_grouped_compile=enable_grouped_compile,
                group_compile_size=group_compile_size,
                benchmark_devices=benchmark_devices,
                benchmark_multi_gpu=benchmark_multi_gpu,
                early_stop=early_stop,
                early_stop_factor=early_stop_factor,
                on_trial=on_trial,
            )
        except BaseException as exc:
            events.emit("run_end", status="error", error=f"{type(exc).__name__}: {exc}")
            recorder.observer_errors.extend(events.errors)
            tuner_events = getattr(self, "_last_event_reporter", None)
            if tuner_events is not None:
                recorder.observer_errors.extend(tuner_events.errors)
            self.last_summary = self._recorded_summary(recorder)
            raise
        else:
            best_config = copy.deepcopy(result.config)
            events.emit(
                "run_end",
                status="cache_hit" if recorder.cache_hit else "ok",
                best_config=best_config,
                best_config_id=_config_id(best_config) if best_config is not None else None,
                best_latency_ms=result.latency,
            )
            recorder.observer_errors.extend(events.errors)
            tuner_events = getattr(self, "_last_event_reporter", None)
            if tuner_events is not None:
                recorder.observer_errors.extend(tuner_events.errors)
            self.last_summary = self._recorded_summary(recorder)
            recorder.raise_for_observer_errors()
            return result
        finally:
            if previous_observer is None:
                self.__dict__.pop("_event_observer", None)
            else:
                self._event_observer = previous_observer
            if jsonl_observer is not None:
                jsonl_observer.close()

    def run_with_summary(self, *args, **kwargs):
        """Return ``(AutotuneResult, detached_summary)`` for experiment drivers."""
        result = self.run(*args, **kwargs)
        return result, copy.deepcopy(self.last_summary)

    def summary_json(self) -> str:
        """Serialize the latest summary as canonical strict JSON text."""
        if self.last_summary is None:
            raise RuntimeError("No guided tuning run has completed yet")
        return json.dumps(self.last_summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

    def write_summary(self, path: str | Path) -> Path:
        """Write the latest pure-JSON summary and return its path."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.summary_json() + "\n", encoding="utf-8", newline="\n")
        return output


__all__ = ["GuidedArm", "GuidedAutoTuner"]
