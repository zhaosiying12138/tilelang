"""Deterministic, backend-neutral model-guided configuration search.

A candidate cost model returns ``(score, terms)`` from :meth:`predict`.  Lower
finite scores rank first.  This module defines no CUDA model and never treats
missing, non-finite, or failed predictions as permission to drop a candidate.
"""

from __future__ import annotations

import copy
import json
import math
import random
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .space import Config, ConfigSpace, PruneResult, Z3Pruner, canonical_json

__all__ = ["CandidateCostModel", "CostModelError", "GuidedSearch"]


@runtime_checkable
class CandidateCostModel(Protocol):
    """Structural interface implemented by backend-specific candidate models.

    Implementations must return a scalar finite score and a dictionary of
    finite numeric itemized terms.  Lower scores are preferred.  ``None``, NaN,
    infinity, and exceptions are invalid predictions, not unknown candidates to
    silently discard.
    """

    def predict(self, config: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
        """Predict one config's score and itemized cost terms."""
        ...


class CostModelError(RuntimeError):
    """A candidate cost model failed or returned an invalid prediction."""


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_float(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CostModelError(f"{description} must be a real numeric scalar, got {value!r}")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise CostModelError(f"{description} cannot be represented as a float: {value!r}") from exc
    if not math.isfinite(normalized):
        raise CostModelError(f"{description} must be finite, got {value!r}")
    return normalized


def _normalize_prediction(prediction: Any) -> tuple[float, dict[str, float]]:
    if not isinstance(prediction, (tuple, list)) or len(prediction) != 2:
        raise CostModelError("Cost model must return a (score, terms) pair")
    raw_score, raw_terms = prediction
    score = _finite_float(raw_score, "score")
    if not isinstance(raw_terms, Mapping):
        raise CostModelError(f"terms must be a mapping, got {type(raw_terms).__name__}")

    terms = {}
    for name, value in raw_terms.items():
        if not isinstance(name, str):
            raise CostModelError(f"term names must be strings, got {name!r}")
        terms[name] = _finite_float(value, f"term {name!r}")
    return score, dict(sorted(terms.items()))


class GuidedSearch:
    """Search a pruned legal universe with deterministic model-based ranking.

    The legal universe ``U`` and model predictions are computed lazily and
    cached.  Every candidate in ``U`` is scored once per ``GuidedSearch``
    instance.  Ranking uses ascending score, then SHA-256 ``config_id``, then
    canonical JSON as a total-order fallback for the theoretical hash-collision
    case.

    Raw and random strategies use the same legal universe and the same budget
    cap as guided selection.  Random selection is without replacement and uses
    a fresh ``random.Random(seed)`` for every call, making repeated calls and
    repeated instances reproducible without mutating global RNG state.
    """

    def __init__(
        self,
        space: ConfigSpace,
        model: CandidateCostModel,
        *,
        pruner: Z3Pruner | None = None,
        seed: int = 42,
    ):
        if not isinstance(space, ConfigSpace):
            raise TypeError("space must be a ConfigSpace")
        if not callable(getattr(model, "predict", None)):
            raise TypeError("model must implement predict(config)")
        if pruner is not None and not isinstance(pruner, Z3Pruner):
            raise TypeError("pruner must be a Z3Pruner or None")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        self.space = space
        self.model = model
        self.pruner = pruner if pruner is not None else Z3Pruner(seed=seed)
        self.seed = seed
        self._prune_result: PruneResult | None = None
        self._predictions: list[dict[str, Any]] | None = None

    def _ensure_universe(self) -> PruneResult:
        if self._prune_result is None:
            self._prune_result = self.pruner.prune(self.space)
        return self._prune_result

    def legal_configs(self) -> list[Config]:
        """Return legal universe ``U`` in deterministic first-seen order."""
        return copy.deepcopy(self._ensure_universe().configs)

    def unknown_configs(self) -> list[Config]:
        """Return the retained subset with unknown/timeout pruning verdicts."""
        return copy.deepcopy(self._ensure_universe().unknown)

    def pruning_audit(self) -> dict[str, Any]:
        """Return a detached JSON-serializable pruning audit."""
        return copy.deepcopy(self._ensure_universe().audit)

    def _score_config(self, config: Config, index: int) -> tuple[float, dict[str, float]]:
        identity = ConfigSpace.config_id(config)
        try:
            prediction = self.model.predict(copy.deepcopy(config))
            return _normalize_prediction(prediction)
        except Exception as exc:
            raise CostModelError(f"Cost model failed for config {identity} at legal index {index}: {exc}") from exc

    def predictions(self) -> list[dict[str, Any]]:
        """Score all of ``U`` and return ranked JSON-serializable records.

        Each record has ``rank`` (one-based), ``universe_index``, ``config``,
        ``config_id``, ``score``, and itemized ``terms``.  No timing or other
        nondeterministic metadata is included, so repeated JSONL output is
        byte-for-byte stable when the model is deterministic.
        """
        if self._predictions is None:
            universe = self._ensure_universe().configs
            ranked = []
            for index, config in enumerate(universe):
                score, terms = self._score_config(config, index)
                ranked.append(
                    {
                        "universe_index": index,
                        "config": copy.deepcopy(config),
                        "config_id": ConfigSpace.config_id(config),
                        "score": score,
                        "terms": terms,
                    }
                )
            ranked.sort(key=lambda record: (record["score"], record["config_id"], canonical_json(record["config"])))
            for rank, record in enumerate(ranked, start=1):
                record["rank"] = rank
            # Validate at the production boundary, not only in tests.
            json.dumps(ranked, ensure_ascii=False, allow_nan=False)
            self._predictions = ranked
        return copy.deepcopy(self._predictions)

    def top_k(self, k: int) -> list[Config]:
        """Return at most ``min(k, len(U))`` best-ranked configurations."""
        count = _nonnegative_int(k, "k")
        return [copy.deepcopy(record["config"]) for record in self.predictions()[:count]]

    def candidates(self, budget: int, strategy: str = "guided") -> list[Config]:
        """Select at most ``budget`` candidates using guided, raw, or random order."""
        count = _nonnegative_int(budget, "budget")
        if strategy not in ("guided", "raw", "random"):
            raise ValueError("strategy must be 'guided', 'raw', or 'random'")
        if strategy == "guided":
            return self.top_k(count)

        universe = self.legal_configs()
        count = min(count, len(universe))
        if strategy == "raw":
            return copy.deepcopy(universe[:count])
        return copy.deepcopy(random.Random(self.seed).sample(universe, count))

    def control_candidates(self, budget: int) -> dict[str, list[Config]]:
        """Return equal-budget raw and seeded-random control candidate lists."""
        count = min(_nonnegative_int(budget, "budget"), len(self._ensure_universe().configs))
        universe = self.legal_configs()
        return {
            "raw": copy.deepcopy(universe[:count]),
            "random": copy.deepcopy(random.Random(self.seed).sample(universe, count)),
        }

    def predictions_jsonl(self) -> str:
        """Serialize all ranked predictions as canonical JSON Lines text."""
        records = self.predictions()
        return "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n" for record in records
        )

    def write_predictions(self, path: str | Path) -> Path:
        """Write ranked predictions as UTF-8 JSONL and return the output path."""
        output_path = Path(path)
        payload = self.predictions_jsonl()
        with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
            output_file.write(payload)
        return output_path
