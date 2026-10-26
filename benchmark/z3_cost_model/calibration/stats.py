"""Small, dependency-free statistics used by workers and CPU tests."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def finite_number(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be > 0")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be >= 0")
    return result


def percentile(sorted_samples: list[float], probability: float) -> float:
    """Linearly interpolated percentile (the NumPy ``linear`` convention)."""

    if not sorted_samples:
        raise ValueError("percentile requires at least one sample")
    if sorted_samples != sorted(sorted_samples):
        raise ValueError("percentile input must be sorted")
    probability = finite_number(probability, "probability")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    position = (len(sorted_samples) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return sorted_samples[lower] * (1.0 - fraction) + sorted_samples[upper] * fraction


def summarize_samples(samples: Iterable[float]) -> dict[str, Any]:
    values = [finite_number(value, "timing sample", nonnegative=True) for value in samples]
    if not values:
        raise ValueError("At least one timing sample is required")
    ordered = sorted(values)
    q1 = percentile(ordered, 0.25)
    median = percentile(ordered, 0.50)
    q3 = percentile(ordered, 0.75)
    return {
        "raw_samples_ms": values,
        "sample_count": len(values),
        "min_ms": ordered[0],
        "q1_ms": q1,
        "median_ms": median,
        "q3_ms": q3,
        "iqr_ms": q3 - q1,
        "max_ms": ordered[-1],
    }


def throughput(work: int, median_ms: float, unit: str) -> dict[str, Any]:
    if isinstance(work, bool) or not isinstance(work, int) or work <= 0:
        raise ValueError("Throughput work must be a positive integer")
    seconds = finite_number(median_ms, "median_ms", positive=True) / 1000.0
    return {
        "value": work / seconds,
        "unit": unit,
        "work": work,
        "time_ms": median_ms,
    }


def latency_per_operation(median_ms: float, operations_per_cta: int) -> dict[str, Any]:
    if isinstance(operations_per_cta, bool) or not isinstance(operations_per_cta, int) or operations_per_cta <= 0:
        raise ValueError("operations_per_cta must be a positive integer")
    elapsed = finite_number(median_ms, "median_ms", positive=True)
    return {
        "value": elapsed * 1_000_000.0 / operations_per_cta,
        "unit": "nanosecond/(barrier*CTA)",
        "work": operations_per_cta,
        "time_ms": median_ms,
        "normalization": "per CTA; concurrent CTAs are not multiplied into this latency",
    }


def with_rate_basis(rate: Mapping[str, Any], source: str, basis: str) -> dict[str, Any]:
    result = dict(rate)
    result["source"] = source
    result["basis"] = basis
    return result
