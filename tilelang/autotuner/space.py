"""Backend-neutral configuration spaces and conservative Z3 pruning.

The module deliberately has no TileLang, compiler, or GPU imports.  A constraint
builder receives ``(config, z3, context)`` and returns one Z3 expression, a
Python boolean, a list/tuple of expressions, or :class:`Indeterminate` when an
external evaluator has already determined UNKNOWN.  Expressions must use the
supplied private context.  Only an explicit UNSAT verdict removes a candidate.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Any, TypeAlias

JSONValue: TypeAlias = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
Config: TypeAlias = dict[str, JSONValue]
ConstraintBuilder: TypeAlias = Callable[[Mapping[str, JSONValue], Any, Any], Any]

__all__ = [
    "Config",
    "ConfigSpace",
    "ConstraintBuildError",
    "ConstraintBuilder",
    "Indeterminate",
    "JSONValue",
    "PruneResult",
    "Z3Pruner",
    "canonical_json",
]


def _normalize_json(value: Any, path: str = "config") -> JSONValue:
    """Return plain JSON data, rejecting lossy or non-deterministic coercions."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{path} contains a non-finite float")
        return normalized
    if isinstance(value, Mapping):
        normalized_dict = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key: {key!r}")
            normalized_dict[key] = _normalize_json(item, f"{path}.{key}")
        return normalized_dict
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize JSON data with sorted keys, compact separators, and finite floats.

    This is the canonical representation used by this module for identity and
    deduplication.  It intentionally rejects non-string mapping keys instead of
    allowing :mod:`json` to coerce them and potentially collide.
    """
    normalized = _normalize_json(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _detached_config(config: Mapping[str, Any]) -> tuple[Config, str, str]:
    if not isinstance(config, Mapping):
        raise TypeError(f"A configuration must be a mapping, got {type(config).__name__}")
    canonical = canonical_json(config)
    normalized = json.loads(canonical)
    if not isinstance(normalized, dict):
        raise TypeError("A configuration must serialize to a JSON object")
    return normalized, canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ConfigSpace:
    """An ordered Cartesian product of finite, discrete JSON-value domains.

    Domain order is preserved.  The last domain varies fastest in
    :meth:`raw_configs`, just like :func:`itertools.product`.  Duplicate values
    inside one domain are rejected rather than silently converted into UNSAT or
    deduplicated candidates.

    Args:
        domains: An ordered mapping or an iterable of ``(name, values)`` pairs.
        constraint_builder: Optional callable consumed by :class:`Z3Pruner`.
            It is called as ``builder(config, z3, private_context)``.
    """

    def __init__(
        self,
        domains: Mapping[str, Iterable[Any]] | Iterable[tuple[str, Iterable[Any]]],
        constraint_builder: ConstraintBuilder | None = None,
    ):
        items = list(domains.items()) if isinstance(domains, Mapping) else list(domains)
        normalized_domains: dict[str, tuple[JSONValue, ...]] = {}
        for name, values in items:
            if not isinstance(name, str):
                raise TypeError(f"Domain names must be strings, got {type(name).__name__}")
            if name in normalized_domains:
                raise ValueError(f"Duplicate domain name: {name!r}")
            if isinstance(values, (str, bytes, bytearray, Mapping, AbstractSet)) or not isinstance(values, Iterable):
                raise TypeError(f"Domain {name!r} must be an ordered iterable of discrete values")
            normalized_values = tuple(json.loads(canonical_json(value)) for value in values)
            identities = [canonical_json(value) for value in normalized_values]
            if len(set(identities)) != len(identities):
                raise ValueError(f"Domain {name!r} contains duplicate canonical JSON values")
            normalized_domains[name] = normalized_values
        if constraint_builder is not None and not callable(constraint_builder):
            raise TypeError("constraint_builder must be callable or None")

        self._domains = normalized_domains
        self._constraint_builder = constraint_builder

    @property
    def domains(self) -> Mapping[str, tuple[JSONValue, ...]]:
        """Return a detached copy of the ordered domains and their values."""
        return copy.deepcopy(self._domains)

    @property
    def constraint_builder(self) -> ConstraintBuilder | None:
        return self._constraint_builder

    @staticmethod
    def canonical_json(config: Mapping[str, Any]) -> str:
        """Return the canonical JSON representation of a configuration."""
        return canonical_json(config)

    @staticmethod
    def config_id(config: Mapping[str, Any]) -> str:
        """Return the SHA-256 hex digest of the configuration's canonical JSON."""
        _, _, identity = _detached_config(config)
        return identity

    @staticmethod
    def config_sha256(config: Mapping[str, Any]) -> str:
        """Explicit alias for :meth:`config_id`."""
        return ConfigSpace.config_id(config)

    def raw_configs(self) -> list[Config]:
        """Enumerate fresh configurations in deterministic Cartesian-product order."""
        names = list(self._domains)
        value_domains = [self._domains[name] for name in names]
        return [copy.deepcopy(dict(zip(names, values))) for values in itertools.product(*value_domains)]

    def __iter__(self):
        return iter(self.raw_configs())

    def __len__(self) -> int:
        return math.prod(len(values) for values in self._domains.values())


@dataclass(frozen=True)
class Indeterminate:
    """An externally determined UNKNOWN that must not invoke a solver.

    Constraint builders return an instance when they cannot produce a
    sufficient SAT/UNSAT constraint but already know the conservative verdict.
    The candidate is retained and audited as unknown; it is never UNSAT and does
    not contribute to the actual solver-check count.
    """

    reason: str = "external_unknown"


@dataclass(frozen=True)
class PruneResult:
    """Conservative pruning output.

    ``configs`` contains unique SAT and indeterminate candidates in first-seen
    order.  ``unknown`` is the subset whose verdict was unknown, including
    external :class:`Indeterminate` results and solver timeouts.  ``audit``
    contains only JSON-serializable values.
    """

    configs: list[Config]
    unknown: list[Config]
    audit: dict[str, Any]


class ConstraintBuildError(RuntimeError):
    """A constraint could not be constructed or added to a solver."""


def _statistics_dict(solver: Any) -> dict[str, JSONValue]:
    raw_stats = solver.statistics()
    if isinstance(raw_stats, Mapping):
        items = list(raw_stats.items())
    else:
        items = [raw_stats[index] for index in range(len(raw_stats))]

    result: dict[str, JSONValue] = {}
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TypeError("Solver statistics must contain (name, value) pairs")
        key, value = item
        key = str(key)
        if isinstance(value, float) and not math.isfinite(value):
            result[key] = str(value)
        elif value is None or isinstance(value, (str, bool, int, float)):
            result[key] = value
        else:
            # Z3 statistics may use arbitrary display objects.  Their string
            # form preserves auditability without weakening JSON guarantees.
            result[key] = str(value)
    return result


def _aggregate_statistics(records: list[dict[str, Any]]) -> dict[str, int | float]:
    aggregate: dict[str, int | float] = {}
    for record in records:
        for key, value in record["stats"].items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            aggregate[key] = aggregate.get(key, 0) + value
    return dict(sorted(aggregate.items()))


def _has_timeout_reason(reason: str) -> bool:
    lowered = reason.lower()
    return any(token in lowered for token in ("timeout", "time out", "timed out"))


def _has_resource_limit_reason(reason: str) -> bool:
    lowered = reason.lower()
    return "resource limit" in lowered or "rlimit" in lowered


class Z3Pruner:
    """Prune only proven-infeasible configurations with isolated Z3 solvers.

    Every determinate non-duplicate candidate gets a fresh solver sharing one
    private context per :meth:`prune` call.  ``random_seed`` and ``rlimit`` are
    fixed on every solver.  Unknown, timeout, and resource-limited results remain
    in the candidate set and are separately identified in the result and audit.
    An :class:`Indeterminate` builder result is audited as unknown without
    constructing a solver or incrementing the solver-check count.

    Args:
        constraint_fn: Optional default constraint builder.  A per-call builder
            overrides this, which in turn overrides a ``ConfigSpace`` builder.
        seed: Z3 ``random_seed`` value.
        rlimit: Positive deterministic Z3 resource limit.  No wall-clock solver
            timeout is used, keeping repeated runs deterministic.
        solver_factory: Optional test/custom injection receiving the private
            context and returning a solver-like object.
        z3_module: Optional module-like injection for custom solver factories.
            The default factory lazily imports :mod:`z3` when needed.
    """

    def __init__(
        self,
        constraint_fn: ConstraintBuilder | None = None,
        *,
        seed: int = 42,
        rlimit: int = 1_000_000,
        solver_factory: Callable[[Any], Any] | None = None,
        z3_module: Any = None,
    ):
        if constraint_fn is not None and not callable(constraint_fn):
            raise TypeError("constraint_fn must be callable or None")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be an integer in [0, 2**32 - 1]")
        if isinstance(rlimit, bool) or not isinstance(rlimit, int) or rlimit <= 0:
            raise ValueError("rlimit must be a positive integer")
        if solver_factory is not None and not callable(solver_factory):
            raise TypeError("solver_factory must be callable or None")

        self.constraint_fn = constraint_fn
        self.seed = seed
        self.rlimit = rlimit
        self.solver_factory = solver_factory
        self.z3_module = z3_module

    def _runtime(self) -> tuple[Any, Any, Callable[[], Any]]:
        if self.solver_factory is None:
            z3 = self.z3_module
            if z3 is None:
                import z3 as imported_z3

                z3 = imported_z3
            context = z3.Context()
            return z3, context, lambda: z3.Solver(ctx=context)

        z3 = self.z3_module
        context = z3.Context() if z3 is not None else None
        return z3, context, lambda: self.solver_factory(context)

    def prune(
        self,
        configs: ConfigSpace | Iterable[Mapping[str, Any]],
        constraint_fn: ConstraintBuilder | None = None,
    ) -> PruneResult:
        """Return retained configs, the unknown subset, and a JSON-safe audit.

        Duplicate canonical configurations are removed before solving and have
        their own ``duplicate`` records.  They are never included in the UNSAT
        count.  Builder/solver errors other than an explicit ``TimeoutError``
        propagate; they are not silently reclassified as UNSAT or unknown.
        """
        started = time.perf_counter()
        builder = constraint_fn if constraint_fn is not None else self.constraint_fn
        if isinstance(configs, ConfigSpace):
            raw_configs = configs.raw_configs()
            if builder is None:
                builder = configs.constraint_builder
        else:
            raw_configs = list(configs)
        if builder is not None and not callable(builder):
            raise TypeError("constraint_fn must be callable or None")

        records: list[dict[str, Any]] = []
        retained: list[Config] = []
        unknown: list[Config] = []
        seen: dict[str, str] = {}
        solver_checks = 0

        z3 = context = None
        new_solver = None
        if builder is not None:
            z3, context, new_solver = self._runtime()

        for index, raw_config in enumerate(raw_configs):
            config, canonical, identity = _detached_config(raw_config)
            if canonical in seen:
                records.append(
                    {
                        "index": index,
                        "config": copy.deepcopy(config),
                        "config_id": identity,
                        "status": "duplicate",
                        "reason": "duplicate",
                        "duplicate_of": seen[canonical],
                        "wall_seconds": 0.0,
                        "stats": {},
                        "timeout": False,
                        "resource_limited": False,
                    }
                )
                continue
            seen[canonical] = identity

            record: dict[str, Any] = {
                "index": index,
                "config": copy.deepcopy(config),
                "config_id": identity,
                "status": "sat",
                "reason": "no_constraints" if builder is None else "sat",
                "wall_seconds": 0.0,
                "stats": {},
                "timeout": False,
                "resource_limited": False,
            }
            if builder is not None:
                assert new_solver is not None
                candidate_started = time.perf_counter()
                try:
                    constraints = builder(copy.deepcopy(config), z3, context)
                    if constraints is None:
                        raise TypeError("Constraint builder returned None instead of expressions")
                    clauses = (
                        None
                        if isinstance(constraints, Indeterminate)
                        else list(constraints)
                        if isinstance(constraints, (list, tuple))
                        else [constraints]
                    )
                except Exception as exc:
                    raise ConstraintBuildError(f"Failed to build constraints for config {identity} at raw index {index}: {exc}") from exc

                if isinstance(constraints, Indeterminate):
                    record["status"] = "unknown"
                    record["reason"] = str(constraints.reason) or "external_unknown"
                    record["wall_seconds"] = time.perf_counter() - candidate_started
                else:
                    assert clauses is not None
                    solver = new_solver()
                    solver.set("random_seed", self.seed)
                    solver.set("rlimit", self.rlimit)
                    try:
                        if clauses:
                            solver.add(*clauses)
                    except Exception as exc:
                        raise ConstraintBuildError(
                            f"Failed to build constraints for config {identity} at raw index {index}: {exc}"
                        ) from exc

                    solver_checks += 1
                    verdict = ""
                    timed_out = False
                    try:
                        verdict = str(solver.check()).lower()
                        reason = verdict
                        if verdict == "sat":
                            status = "sat"
                        elif verdict == "unsat":
                            status = "unsat"
                        elif verdict in ("unknown", "timeout"):
                            status = "unknown"
                            reason = str(solver.reason_unknown()) if verdict == "unknown" else "timeout"
                            if not reason:
                                reason = "unknown"
                        else:
                            raise RuntimeError(f"Unrecognized solver verdict {verdict!r} for config {identity}")
                    except TimeoutError as exc:
                        status = "unknown"
                        reason = f"timeout: {exc}"
                        timed_out = True

                    record["status"] = status
                    record["reason"] = reason
                    record["timeout"] = timed_out or verdict == "timeout"
                    if status == "unknown":
                        record["timeout"] = record["timeout"] or _has_timeout_reason(reason)
                        record["resource_limited"] = _has_resource_limit_reason(reason)
                    record["stats"] = _statistics_dict(solver)
                    record["wall_seconds"] = time.perf_counter() - candidate_started

            records.append(record)
            if record["status"] != "unsat":
                retained.append(config)
            if record["status"] == "unknown":
                unknown.append(config)

        counts = {status: sum(record["status"] == status for record in records) for status in ("sat", "unsat", "unknown", "duplicate")}
        unknown_ids = [record["config_id"] for record in records if record["status"] == "unknown"]
        timeout_ids = [record["config_id"] for record in records if record["timeout"]]
        resource_limited_ids = [record["config_id"] for record in records if record["resource_limited"]]
        audit = {
            "raw": len(raw_configs),
            "unique": len(seen),
            "duplicates": counts["duplicate"],
            "checked": solver_checks,
            "sat": counts["sat"],
            "unsat": counts["unsat"],
            "unknown": counts["unknown"],
            "timeouts": len(timeout_ids),
            "resource_limited": len(resource_limited_ids),
            "kept": len(retained),
            "pruned": counts["unsat"],
            "seed": self.seed,
            "rlimit": self.rlimit,
            "wall_seconds": time.perf_counter() - started,
            "stats": _aggregate_statistics(records),
            "unknown_config_ids": unknown_ids,
            "timeout_config_ids": timeout_ids,
            "resource_limited_config_ids": resource_limited_ids,
            "records": records,
        }
        # Verify the audit contract at the boundary, including model-provided
        # statistics values and all configuration payloads.
        json.dumps(audit, ensure_ascii=False, allow_nan=False)
        return PruneResult(configs=retained, unknown=unknown, audit=audit)
