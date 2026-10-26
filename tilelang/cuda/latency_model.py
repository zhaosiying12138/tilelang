"""Pure-Python CUDA task latency and initiation-interval estimates.

The scheduler consumes integer cycles.  ``CudaLatencyParams`` therefore uses
only cycles, bytes/cycle, and operations/cycle; ``Calibration`` is the explicit
unit-conversion boundary for JSON data in ns, bytes/second, or FLOP/second.
There are no TileLang, TVM, torch, CUDA, or GPU imports in this module.

Formula structure follows Ascend's task model, not its hardware constants:
memory completion is summed, active compute resources take their maximum, and
II is computed independently per resource.  Missing or zero rates use an
explicit unit-rate placeholder so they are never mistaken for free hardware.
Every such use is reported in ``TaskCost.fallback_used`` and degrades
``TaskCost.reliability``.  ``require_measured=True`` turns any non-measured or
fallback result into ``CalibrationError``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, IntFlag
from fractions import Fraction
import json
import math
import operator
import os
from typing import Any

INT64_MAX = (1 << 63) - 1
UNIT_PLACEHOLDER = Fraction(1, 1)  # Deliberately unmeasured, never "free".


class CalibrationError(ValueError):
    """Raised when measured-only estimation needs fallback or unknown data."""


class Reliability(str, Enum):
    """Reliability of both feature extents and numerical calibration."""

    MEASURED = "measured"
    ESTIMATED = "estimated"
    UPPER_BOUND = "upper_bound"
    UNKNOWN = "unknown"


class Resource(IntFlag):
    """Seven-bit CUDA resource mask consumed by the CUDA Z3 scheduler.

    The mask describes contention domains, not an instruction encoding.  TMA
    loads and cp.async can share GLOBAL_LOAD while retaining distinct calibrated
    memory paths in richer callers.
    """

    NONE = 0
    GLOBAL_LOAD = 1
    GLOBAL_STORE = 2
    SHARED_MEMORY = 4
    TENSOR_CORE = 8
    CUDA_CORE = 16
    SFU = 32
    SCALAR = 64

    # Readable aliases used by feature extractors.
    COPY = 1
    CP_ASYNC = 1
    TMA_LOAD = 1
    STORE = 2
    LSU = 4
    LDST = 4
    MMA = 8
    GEMM = 8
    VECTOR = 16
    SPECIAL = 32
    ALL = 127


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite floats are not JSON serializable")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported JSON value of type {type(value).__name__}")


class _JSONMixin:
    def to_dict(self) -> dict[str, Any]:
        result = _json_value(self)
        assert isinstance(result, dict)
        return result

    def to_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("allow_nan", False)
        kwargs.setdefault("sort_keys", True)
        return json.dumps(self.to_dict(), **kwargs)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer, got {value!r}") from error
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    if result > INT64_MAX:
        raise OverflowError(f"{name} exceeds int64: {result}")
    return result


def _checked_add(left: int, right: int, name: str) -> int:
    _integer(left, name)
    _integer(right, name)
    if left > INT64_MAX - right:
        raise OverflowError(f"int64 overflow in {name}: {left} + {right}")
    return left + right


def _checked_mul(left: int, right: int, name: str) -> int:
    _integer(left, name)
    _integer(right, name)
    if left and right > INT64_MAX // left:
        raise OverflowError(f"int64 overflow in {name}: {left} * {right}")
    return left * right


def _bounded(value: int, name: str) -> int:
    if value > INT64_MAX:
        raise OverflowError(f"int64 overflow in {name}: {value}")
    return value


def _rate(value: Any, name: str) -> Fraction | None:
    """Parse a non-negative finite rate; None and zero both mean unknown."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise TypeError(f"{name} must be a real number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    result = value if isinstance(value, Fraction) else Fraction(str(value))
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _ceil_rate(work: int, rate: Fraction, name: str) -> int:
    _integer(work, name)
    if rate <= 0:
        raise ValueError(f"Internal error: non-positive rate for {name}")
    if work == 0:
        return 0
    cycles = (work * rate.denominator + rate.numerator - 1) // rate.numerator
    return _bounded(cycles, name)


def _rate_json(value: Fraction | None) -> int | float | None:
    if value is None:
        return None
    if value.denominator == 1:
        return value.numerator
    return float(value)


def _reliability(value: Reliability | str | bool) -> Reliability:
    if isinstance(value, Reliability):
        return value
    if isinstance(value, bool):
        return Reliability.MEASURED if value else Reliability.UNKNOWN
    aliases = {
        "exact": Reliability.MEASURED,
        "reliable": Reliability.MEASURED,
        "provided": Reliability.ESTIMATED,
        "fallback": Reliability.ESTIMATED,
        "derived": Reliability.ESTIMATED,
        "bound": Reliability.UPPER_BOUND,
        "unreliable": Reliability.UNKNOWN,
    }
    text = str(value).strip().lower()
    if text in aliases:
        return aliases[text]
    try:
        return Reliability(text)
    except ValueError as error:
        raise ValueError(f"Unknown reliability {value!r}") from error


_RELIABILITY_RANK = {
    Reliability.MEASURED: 0,
    Reliability.ESTIMATED: 1,
    Reliability.UPPER_BOUND: 2,
    Reliability.UNKNOWN: 3,
}


def _worst_reliability(*values: Reliability) -> Reliability:
    return max(values, key=_RELIABILITY_RANK.__getitem__, default=Reliability.MEASURED)


def _extent_object(value: Any, name: str = "extent") -> Extent | int:
    if isinstance(value, (Extent, int)) and not isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        upper = value.get("upper_bound", value.get("upper", value.get("max")))
        reliability = value.get("reliability", Reliability.UPPER_BOUND)
        return Extent(upper, reliability)
    if value is None:
        return Extent(None, Reliability.UNKNOWN)
    raise TypeError(f"{name} must be an integer, Extent, mapping, or None")


@dataclass(frozen=True)
class Extent(_JSONMixin):
    """A dynamic extent with a finite upper bound used for integer costing."""

    upper_bound: int | None
    reliability: Reliability | str | bool = Reliability.UPPER_BOUND

    def __post_init__(self) -> None:
        object.__setattr__(self, "reliability", _reliability(self.reliability))
        if self.upper_bound is not None:
            _integer(self.upper_bound, "extent upper_bound")


DynamicExtent = Extent


_MEMORY_PATH_FIELDS = {
    "base_cycles": 0,
    "bandwidth_bytes_per_cycle": 0.0,
    "descriptor_cycles": 0,
    "issue_cycles": 0,
    "min_ii": 1,
    "transaction_bytes": 0,
    "strided_bandwidth_bytes_per_cycle": 0.0,
    "unknown_bandwidth_bytes_per_cycle": 0.0,
}
_MEMORY_PATH_ALIASES = {
    "base_ns": "base_cycles",
    "descriptor_ns": "descriptor_cycles",
    "issue_ns": "issue_cycles",
    "bandwidth_bytes_per_second": "bandwidth_bytes_per_cycle",
    "strided_bandwidth_bytes_per_second": "strided_bandwidth_bytes_per_cycle",
    "unknown_bandwidth_bytes_per_second": "unknown_bandwidth_bytes_per_cycle",
}


def _cycles_from_ns(value: Any, clock_hz: float, name: str) -> int:
    nanoseconds = _rate(value, name)
    if nanoseconds is None or clock_hz <= 0:
        raise ValueError(f"{name} in ns requires a positive measured clock_hz")
    cycles = nanoseconds * Fraction(str(clock_hz)) / 1_000_000_000
    return _bounded((cycles.numerator + cycles.denominator - 1) // cycles.denominator, name)


def _per_cycle(value: Any, clock_hz: float, name: str) -> float:
    per_second = _rate(value, name)
    if per_second is None or clock_hz <= 0:
        raise ValueError(f"{name} per second requires a positive measured clock_hz")
    result = per_second / Fraction(str(clock_hz))
    return float(result)


@dataclass(frozen=True)
class MemoryPathParams(_JSONMixin):
    """Calibrated completion and issue parameters for one physical copy path."""

    base_cycles: int = 0
    bandwidth_bytes_per_cycle: float = 0.0
    descriptor_cycles: int = 0
    issue_cycles: int = 0
    min_ii: int = 1
    transaction_bytes: int = 0
    strided_bandwidth_bytes_per_cycle: float = 0.0
    unknown_bandwidth_bytes_per_cycle: float = 0.0

    def __post_init__(self) -> None:
        for name in ("base_cycles", "descriptor_cycles", "issue_cycles", "min_ii", "transaction_bytes"):
            _integer(getattr(self, name), f"memory path {name}")
        for name in (
            "bandwidth_bytes_per_cycle",
            "strided_bandwidth_bytes_per_cycle",
            "unknown_bandwidth_bytes_per_cycle",
        ):
            _rate(getattr(self, name), f"memory path {name}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], clock_hz: float = 0.0) -> MemoryPathParams:
        if not isinstance(data, Mapping):
            raise TypeError("Memory path calibration must be a mapping")
        unknown = set(data) - set(_MEMORY_PATH_FIELDS) - set(_MEMORY_PATH_ALIASES)
        if unknown:
            raise ValueError(f"Unknown memory path calibration fields: {sorted(unknown)}")
        values: dict[str, Any] = {}
        for name in _MEMORY_PATH_FIELDS:
            if name in data:
                values[name] = data[name]
        for alias, name in _MEMORY_PATH_ALIASES.items():
            if alias not in data:
                continue
            if name in data:
                raise ValueError(f"Specify only one of {name} and {alias}")
            if alias.endswith("_ns"):
                values[name] = _cycles_from_ns(data[alias], clock_hz, alias)
            else:
                values[name] = _per_cycle(data[alias], clock_hz, alias)
        return cls(**values)


def _canonical_scope(scope: str) -> str:
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("Memory scopes must be non-empty strings")
    text = scope.strip().lower().replace("_", ".")
    if text in {"global", "gm", "gmem", "dram"}:
        return "global"
    if text in {"shared", "shared.dyn", "smem"}:
        return "shared"
    if text in {"local", "local.fragment", "register", "registers", "fragment", "reg"}:
        return "register"
    if text in {"tmem", "tensor.memory"}:
        return "tmem"
    return scope.strip().lower()


def _path_name(src_scope: str, dst_scope: str) -> str:
    return f"{_canonical_scope(src_scope)}->{_canonical_scope(dst_scope)}"


def _canonical_dtype(dtype: str) -> str:
    if not isinstance(dtype, str) or not dtype.strip():
        raise ValueError("dtype and acc must be non-empty strings")
    text = dtype.strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "float16": "f16",
        "half": "f16",
        "fp16": "f16",
        "bfloat16": "bf16",
        "float32": "f32",
        "fp32": "f32",
        "float": "f32",
        "float8e4m3": "e4m3",
        "fp8e4m3": "e4m3",
        "float8e5m2": "e5m2",
        "fp8e5m2": "e5m2",
        "float4e2m1": "e2m1",
        "fp4": "e2m1",
        "s8": "int8",
        "int32": "s32",
    }
    return aliases.get(text, text)


def _canonical_arch(arch: str) -> str:
    text = str(arch).strip().lower().replace("_", "").replace(".", "")
    if text.isdigit():
        text = "sm" + text
    return text


def _mapping_item(mapping: Mapping[str, Any], key: str, canonicalizer) -> tuple[str, Any] | None:
    canonical = canonicalizer(key)
    for candidate, value in mapping.items():
        if canonicalizer(str(candidate)) == canonical:
            return str(candidate), value
    return None


def _dtype_key(key: str) -> str:
    parts = key.split(":", 1)
    result = _canonical_dtype(parts[0])
    if len(parts) == 2:
        result += ":" + _canonical_dtype(parts[1])
    return result


@dataclass(frozen=True)
class CudaLatencyParams(_JSONMixin):
    """Cycle-native CUDA calibration and model parameters.

    ``field_sources`` maps a parameter path to ``measured``, ``provided``,
    ``derived``, ``fallback``, ``default``, or ``structural``.  Direct positive
    constructor values are considered provided (not measured); omitted values
    are recognized as defaults when consumed.
    """

    arch: str = "sm120"
    memory_paths: Mapping[str, MemoryPathParams] = field(default_factory=dict)
    gemm_base_cycles: int = 0
    gemm_min_ii: int = 1
    gemm_pipeline_depth: int = 1
    gemm_throughput_ops_per_cycle: Mapping[str, float] = field(default_factory=dict)
    vector_throughput_ops_per_cycle: Mapping[str, float] = field(default_factory=dict)
    vector_bandwidth_bytes_per_cycle: float = 0.0
    scalar_cycles: int = 1
    clock_hz: float = 0.0  # Audit metadata only; task arithmetic is cycle-native.
    sm_count: int = 1  # Audit metadata; rates are already per task/SM.
    field_sources: Mapping[str, str] = field(default_factory=dict, repr=False)
    source: str = "<direct>"

    def __post_init__(self) -> None:
        if not isinstance(self.arch, str) or not self.arch.strip():
            raise ValueError("arch must be a non-empty string")
        paths = {}
        path_names = {}
        for path, value in self.memory_paths.items():
            raw_path = str(path)
            parts = raw_path.split("->")
            if len(parts) != 2:
                raise ValueError(f"Memory path must be 'src->dst', got {raw_path!r}")
            canonical_path = _path_name(parts[0], parts[1])
            if canonical_path in paths:
                raise ValueError(f"Duplicate canonical memory path {canonical_path!r}")
            paths[canonical_path] = value if isinstance(value, MemoryPathParams) else MemoryPathParams.from_dict(value)
            path_names[raw_path] = canonical_path
        object.__setattr__(self, "memory_paths", paths)
        object.__setattr__(self, "gemm_throughput_ops_per_cycle", dict(self.gemm_throughput_ops_per_cycle))
        object.__setattr__(self, "vector_throughput_ops_per_cycle", dict(self.vector_throughput_ops_per_cycle))
        for name in ("gemm_base_cycles", "gemm_min_ii", "gemm_pipeline_depth", "scalar_cycles", "sm_count"):
            _integer(getattr(self, name), name, 1 if name in {"gemm_min_ii", "sm_count"} else 0)
        _rate(self.vector_bandwidth_bytes_per_cycle, "vector_bandwidth_bytes_per_cycle")
        _rate(self.clock_hz, "clock_hz")
        for mapping_name in ("gemm_throughput_ops_per_cycle", "vector_throughput_ops_per_cycle"):
            for key, value in getattr(self, mapping_name).items():
                _rate(value, f"{mapping_name}.{key}")

        sources = {}
        valid_sources = {"measured", "provided", "derived", "fallback", "default", "structural"}
        for key, provenance in self.field_sources.items():
            if not isinstance(key, str) or provenance not in valid_sources:
                raise ValueError(f"Invalid field_sources entry {key!r}: {provenance!r}")
            canonical_key = key
            for raw_path, canonical_path in path_names.items():
                prefix = f"memory_paths.{raw_path}."
                if key.startswith(prefix):
                    canonical_key = f"memory_paths.{canonical_path}." + key[len(prefix):]
                    break
            if canonical_key in sources and sources[canonical_key] != provenance:
                raise ValueError(f"Conflicting provenance for {canonical_key!r}")
            sources[canonical_key] = provenance
        for name, default in (
            ("arch", "sm120"),
            ("gemm_min_ii", 1),
            ("gemm_pipeline_depth", 1),
            ("gemm_base_cycles", 0),
            ("vector_bandwidth_bytes_per_cycle", 0.0),
            ("scalar_cycles", 1),
        ):
            sources.setdefault(name, "default" if getattr(self, name) == default else "provided")
        for path, value in paths.items():
            for name, default in _MEMORY_PATH_FIELDS.items():
                key = f"memory_paths.{path}.{name}"
                sources.setdefault(key, "default" if getattr(value, name) == default else "provided")
        for mapping_name in ("gemm_throughput_ops_per_cycle", "vector_throughput_ops_per_cycle"):
            for key in getattr(self, mapping_name):
                sources.setdefault(f"{mapping_name}.{key}", "provided")
        object.__setattr__(self, "field_sources", sources)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        measured: bool = False,
        source: str = "<json>",
        field_sources: Mapping[str, str] | None = None,
    ) -> CudaLatencyParams:
        if not isinstance(data, Mapping):
            raise TypeError("CUDA latency calibration must be a JSON object")
        known = {
            "arch", "memory_paths", "gemm_base_cycles", "gemm_base_ns", "gemm_min_ii",
            "gemm_pipeline_depth", "gemm_throughput_ops_per_cycle", "vector_throughput_ops_per_cycle",
            "vector_bandwidth_bytes_per_cycle", "vector_bandwidth_bytes_per_second", "scalar_cycles",
            "scalar_ns", "clock_hz", "sm_count", "measured", "status", "source", "description",
            "measured_fields", "fallback_fields", "field_sources", "units", "device",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown CUDA latency calibration fields: {sorted(unknown)}")
        raw_clock_hz = data.get("clock_hz", 0.0)
        if isinstance(raw_clock_hz, bool) or not isinstance(raw_clock_hz, (int, float)):
            raise TypeError("clock_hz must be a real number")
        clock_hz = float(raw_clock_hz)
        if clock_hz < 0 or not math.isfinite(clock_hz):
            raise ValueError("clock_hz must be a non-negative finite number")
        provenance = "measured" if measured else "provided"
        json_sources = data.get("field_sources", {})
        if not isinstance(json_sources, Mapping):
            raise TypeError("field_sources must be a mapping")
        sources = dict(json_sources)
        sources.update(field_sources or {})
        source = str(data.get("source", source))
        values: dict[str, Any] = {"source": source}

        def take(name: str, alias: str | None = None) -> None:
            if name in data:
                values[name] = data[name]
                sources[name] = provenance
            elif alias and alias in data:
                values[name] = data[alias]
                sources[name] = provenance

        take("arch")
        take("gemm_min_ii")
        take("gemm_pipeline_depth")
        take("sm_count")
        take("clock_hz")
        if "gemm_base_cycles" in data and "gemm_base_ns" in data:
            raise ValueError("Specify only one of gemm_base_cycles and gemm_base_ns")
        if "gemm_base_cycles" in data:
            take("gemm_base_cycles")
        elif "gemm_base_ns" in data:
            values["gemm_base_cycles"] = _cycles_from_ns(data["gemm_base_ns"], clock_hz, "gemm_base_ns")
            sources["gemm_base_cycles"] = provenance
        if "scalar_cycles" in data and "scalar_ns" in data:
            raise ValueError("Specify only one of scalar_cycles and scalar_ns")
        if "scalar_cycles" in data:
            take("scalar_cycles")
        elif "scalar_ns" in data:
            values["scalar_cycles"] = _cycles_from_ns(data["scalar_ns"], clock_hz, "scalar_ns")
            sources["scalar_cycles"] = provenance
        if "vector_bandwidth_bytes_per_cycle" in data and "vector_bandwidth_bytes_per_second" in data:
            raise ValueError("Specify vector bandwidth in only one unit")
        if "vector_bandwidth_bytes_per_cycle" in data:
            take("vector_bandwidth_bytes_per_cycle")
        elif "vector_bandwidth_bytes_per_second" in data:
            values["vector_bandwidth_bytes_per_cycle"] = _per_cycle(
                data["vector_bandwidth_bytes_per_second"], clock_hz, "vector_bandwidth_bytes_per_second"
            )
            sources["vector_bandwidth_bytes_per_cycle"] = provenance

        raw_paths = data.get("memory_paths", {})
        if not isinstance(raw_paths, Mapping):
            raise TypeError("memory_paths must be a mapping")
        paths = {}
        for path, raw in raw_paths.items():
            path = str(path)
            paths[path] = MemoryPathParams.from_dict(raw, clock_hz)
            if isinstance(raw, Mapping):
                for key in raw:
                    name = _MEMORY_PATH_ALIASES.get(key, key)
                    sources[f"memory_paths.{path}.{name}"] = provenance
        values["memory_paths"] = paths

        for name in ("gemm_throughput_ops_per_cycle", "vector_throughput_ops_per_cycle"):
            raw_rates = data.get(name, {})
            if not isinstance(raw_rates, Mapping):
                raise TypeError(f"{name} must be a mapping")
            rates = {}
            for key, raw_rate in raw_rates.items():
                if isinstance(raw_rate, Mapping):
                    allowed = {"ops_per_cycle", "ops_per_second"}
                    if set(raw_rate) - allowed or len(raw_rate) != 1:
                        raise ValueError(f"{name}.{key} needs exactly one explicit rate unit")
                    if "ops_per_cycle" in raw_rate:
                        rates[key] = raw_rate["ops_per_cycle"]
                    else:
                        rates[key] = _per_cycle(raw_rate["ops_per_second"], clock_hz, f"{name}.{key}")
                else:
                    rates[key] = raw_rate
                sources[f"{name}.{key}"] = provenance
            values[name] = rates

        sources.update(json_sources)
        sources.update(field_sources or {})
        for key in data.get("measured_fields", []):
            sources[str(key)] = "measured"
        for key in data.get("fallback_fields", []):
            sources[str(key)] = "fallback"
        values["field_sources"] = sources
        return cls(**values)

    @classmethod
    def from_json(cls, data: str | Mapping[str, Any], **kwargs: Any) -> CudaLatencyParams:
        return Calibration.from_json(data, **kwargs).params

    @classmethod
    def load(cls, path: str | os.PathLike[str], **kwargs: Any) -> CudaLatencyParams:
        return Calibration.load(path, **kwargs).params


@dataclass(frozen=True)
class Calibration(_JSONMixin):
    """A parameter bundle plus explicit provenance and fallback diagnostics."""

    params: CudaLatencyParams
    source: str = "<json>"
    reliability: Reliability = Reliability.UNKNOWN
    fallback_used: tuple[str, ...] = ()
    unmeasured_used: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        sources = self.params.field_sources
        performance = {
            key: value for key, value in sources.items()
            if value != "structural" and key not in {"clock_hz", "sm_count"}
        }
        fallback = tuple(sorted(
            key if source != "derived" else f"derived:{key}"
            for key, source in performance.items() if source in {"default", "fallback", "derived"}
        ))
        unmeasured = tuple(sorted(key for key, source in performance.items() if source == "provided"))
        if not performance or any(source in {"default", "fallback"} for source in performance.values()):
            reliability = Reliability.UNKNOWN
        elif any(source in {"provided", "derived"} for source in performance.values()):
            reliability = Reliability.ESTIMATED
        else:
            reliability = Reliability.MEASURED
        object.__setattr__(self, "fallback_used", fallback)
        object.__setattr__(self, "unmeasured_used", unmeasured)
        object.__setattr__(self, "reliability", reliability)
        object.__setattr__(self, "source", self.params.source or self.source)

    def require_measured(self) -> None:
        problems = list(self.fallback_used) + list(self.unmeasured_used)
        if not self.params.field_sources or self.reliability == Reliability.UNKNOWN and not problems:
            problems.append("calibration")
        if problems:
            raise CalibrationError("Measured calibration required; unavailable: " + ", ".join(problems))

    @classmethod
    def from_json(
        cls,
        data: str | Mapping[str, Any],
        *,
        arch: str | None = None,
        require_measured: bool = False,
        source: str = "<json>",
    ) -> Calibration:
        if isinstance(data, str):
            parsed = json.loads(data)
        else:
            parsed = data
        if not isinstance(parsed, Mapping):
            raise TypeError("Calibration JSON must contain an object")
        if isinstance(parsed.get("measurements"), Mapping):
            params = _params_from_gdn(parsed, source=source, arch=arch)
        else:
            native = dict(parsed)
            if arch is not None:
                native["arch"] = arch
            measured = native.get("measured") is True or native.get("status") == "measured"
            params = CudaLatencyParams.from_dict(native, measured=measured, source=source)
        result = cls(params=params, source=source)
        if require_measured:
            result.require_measured()
        return result

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        arch: str | None = None,
        require_measured: bool = False,
    ) -> Calibration:
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
        return cls.from_json(data, arch=arch, require_measured=require_measured, source=os.fspath(path))


def _gdn_number(section: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = section.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
    return None


def _params_from_gdn(data: Mapping[str, Any], source: str, arch: str | None) -> CudaLatencyParams:
    """Import GDN G3-A measured rates without importing its model or constants."""

    root = data["measurements"]
    device = root.get("device", {})
    raw_sm_count = _gdn_number(device, "sm_count")
    if raw_sm_count is None or raw_sm_count <= 0 or not raw_sm_count.is_integer():
        raise CalibrationError("GDN calibration requires a positive integer device.sm_count")
    sm_count = int(raw_sm_count)
    clock_khz = _gdn_number(device, "clock_peak_khz", "clock_khz")
    clock_hz = clock_khz * 1000.0 if clock_khz else 0.0
    if clock_hz <= 0:
        raise CalibrationError("GDN calibration requires device.clock_peak_khz for cycle conversion")
    sources: dict[str, str] = {
        "arch": "provided" if arch is not None else "default",
        "clock_hz": "measured",
        "sm_count": "measured",
    }

    def per_sm_cycle(per_second: float) -> float:
        return float(Fraction(str(per_second)) / Fraction(str(sm_count)) / Fraction(str(clock_hz)))

    paths: dict[str, MemoryPathParams] = {}
    measured_bw = {
        "global->shared": _gdn_number(root.get("cp_async_16b", {}), "gbps"),
        "global->register": _gdn_number(root.get("gmem_read_l2resident", {}), "gbps"),
        "shared->register": _gdn_number(root.get("smem_read_bw_gbps", {}), "gbps"),
    }
    for path, gbps in measured_bw.items():
        if gbps is not None:
            paths[path] = MemoryPathParams(bandwidth_bytes_per_cycle=per_sm_cycle(gbps * 1e9))
            sources[f"memory_paths.{path}.bandwidth_bytes_per_cycle"] = "measured"
    derived_paths = {
        "shared->global": _gdn_number(root.get("gmem_read_dram", {}), "gbps"),
        "register->global": _gdn_number(root.get("gmem_read_dram", {}), "gbps"),
        "register->shared": _gdn_number(root.get("smem_read_bw_gbps", {}), "gbps"),
    }
    for path, gbps in derived_paths.items():
        if gbps is not None:
            paths[path] = MemoryPathParams(bandwidth_bytes_per_cycle=per_sm_cycle(gbps * 1e9))
            sources[f"memory_paths.{path}.bandwidth_bytes_per_cycle"] = "derived"

    gemm_rates: dict[str, float] = {}
    mma = root.get("mma_bf16_m16n8k16", {})
    mma_flops = _gdn_number(mma, "flops", "flops_per_s")
    mma_instr = _gdn_number(mma, "instr_per_s")
    if mma_flops is not None and mma_instr is not None:
        warp_flops = mma_instr / 32.0 * 4096.0
        if mma_flops > warp_flops * 1.5:
            mma_flops = warp_flops
    if mma_flops is not None:
        bf16_rate = per_sm_cycle(mma_flops)
        gemm_rates["bf16"] = bf16_rate
        gemm_rates["bf16:f32"] = bf16_rate
        sources["gemm_throughput_ops_per_cycle.bf16"] = "measured"
        sources["gemm_throughput_ops_per_cycle.bf16:f32"] = "measured"
        for key, rate in (("f16", bf16_rate), ("f16:f32", bf16_rate), ("tf32", bf16_rate * 0.5), ("tf32:f32", bf16_rate * 0.5)):
            gemm_rates[key] = rate
            sources[f"gemm_throughput_ops_per_cycle.{key}"] = "derived"

    vector_rates: dict[str, float] = {}
    ex2 = _gdn_number(root.get("sfu_ex2_f32", {}), "elem_per_s")
    if ex2 is not None:
        vector_rates["exp2"] = per_sm_cycle(ex2)
        sources["vector_throughput_ops_per_cycle.exp2"] = "measured"
        vector_rates["special"] = vector_rates["exp2"]
        sources["vector_throughput_ops_per_cycle.special"] = "derived"
    smem_gbps = measured_bw["shared->register"]
    vector_bw = per_sm_cycle(smem_gbps * 1e9) if smem_gbps is not None else 0.0
    if smem_gbps is not None:
        sources["vector_bandwidth_bytes_per_cycle"] = "measured"
    return CudaLatencyParams(
        arch=arch or "sm120",
        memory_paths=paths,
        gemm_throughput_ops_per_cycle=gemm_rates,
        vector_throughput_ops_per_cycle=vector_rates,
        vector_bandwidth_bytes_per_cycle=vector_bw,
        clock_hz=clock_hz,
        sm_count=int(sm_count),
        field_sources=sources,
        source=source,
    )


@dataclass(frozen=True)
class MemoryAccess(_JSONMixin):
    """One logical descriptor; bytes and width may carry dynamic upper bounds."""

    src_scope: str
    dst_scope: str
    bytes: Extent | int | Mapping[str, Any] | None
    geometry: str = "coalesced"
    width: Extent | int | Mapping[str, Any] | None = 0
    reliability: Reliability | str | bool = Reliability.MEASURED

    def __post_init__(self) -> None:
        _canonical_scope(self.src_scope)
        _canonical_scope(self.dst_scope)
        object.__setattr__(self, "bytes", _extent_object(self.bytes, "memory bytes"))
        object.__setattr__(self, "width", _extent_object(self.width, "memory width"))
        object.__setattr__(self, "reliability", _reliability(self.reliability))
        if not isinstance(self.geometry, str):
            raise TypeError("geometry must be a string")

    @property
    def src(self) -> str:
        return self.src_scope

    @property
    def dst(self) -> str:
        return self.dst_scope


@dataclass(frozen=True)
class GemmFeature(_JSONMixin):
    M: Extent | int | Mapping[str, Any]
    N: Extent | int | Mapping[str, Any]
    K: Extent | int | Mapping[str, Any]
    dtype: str = "bf16"
    acc: str = "f32"
    reliability: Reliability | str | bool = Reliability.MEASURED

    def __post_init__(self) -> None:
        for name in ("M", "N", "K"):
            object.__setattr__(self, name, _extent_object(getattr(self, name), name))
        object.__setattr__(self, "dtype", _canonical_dtype(self.dtype))
        object.__setattr__(self, "acc", _canonical_dtype(self.acc))
        object.__setattr__(self, "reliability", _reliability(self.reliability))


_VECTOR_OP_NAMES = ("add", "sub", "mul", "fma", "div", "mod", "min_max", "cmp", "logic", "special", "exp2", "unknown")


@dataclass(frozen=True)
class VectorOps(_JSONMixin):
    """Aggregated operation counts and on-chip read bytes for a vector task."""

    add: Extent | int = 0
    sub: Extent | int = 0
    mul: Extent | int = 0
    fma: Extent | int = 0
    div: Extent | int = 0
    mod: Extent | int = 0
    min_max: Extent | int = 0
    cmp: Extent | int = 0
    logic: Extent | int = 0
    special: Extent | int = 0
    exp2: Extent | int = 0
    unknown: Extent | int = 0
    read_bytes: Extent | int = 0
    reliability: Reliability | str | bool = Reliability.MEASURED

    def __post_init__(self) -> None:
        for name in (*_VECTOR_OP_NAMES, "read_bytes"):
            object.__setattr__(self, name, _extent_object(getattr(self, name), name))
        object.__setattr__(self, "reliability", _reliability(self.reliability))


@dataclass(frozen=True)
class TaskFeatures(_JSONMixin):
    copies: tuple[MemoryAccess, ...] = ()
    gemms: tuple[GemmFeature, ...] = ()
    vector: VectorOps = field(default_factory=VectorOps)
    scalar: int | bool = False
    latency_override: int | None = None
    ii_override: int | None = None
    reliability: Reliability | str | bool = Reliability.MEASURED

    def __post_init__(self) -> None:
        copies = (self.copies,) if isinstance(self.copies, MemoryAccess) else tuple(self.copies)
        gemms = (self.gemms,) if isinstance(self.gemms, GemmFeature) else tuple(self.gemms)
        object.__setattr__(self, "copies", tuple(
            item if isinstance(item, MemoryAccess) else MemoryAccess(**item) for item in copies
        ))
        object.__setattr__(self, "gemms", tuple(
            item if isinstance(item, GemmFeature) else GemmFeature(**item) for item in gemms
        ))
        if not isinstance(self.vector, VectorOps):
            object.__setattr__(self, "vector", VectorOps(**self.vector))
        if isinstance(self.scalar, bool):
            object.__setattr__(self, "scalar", int(self.scalar))
        _integer(self.scalar, "scalar operation count")
        object.__setattr__(self, "reliability", _reliability(self.reliability))

    @property
    def copy(self) -> tuple[MemoryAccess, ...]:
        return self.copies

    @property
    def gemm(self) -> tuple[GemmFeature, ...]:
        return self.gemms

    @property
    def user_latency(self) -> int | None:
        return self.latency_override

    @property
    def user_ii(self) -> int | None:
        return self.ii_override

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskFeatures:
        values = dict(data)
        if "copy" in values:
            if "copies" in values:
                raise ValueError("Specify only copy or copies")
            values["copies"] = values.pop("copy")
        if "gemm" in values:
            if "gemms" in values:
                raise ValueError("Specify only gemm or gemms")
            values["gemms"] = values.pop("gemm")
        if "user_latency" in values:
            values["latency_override"] = values.pop("user_latency")
        if "user_ii" in values:
            values["ii_override"] = values.pop("user_ii")
        return cls(**values)


@dataclass(frozen=True)
class TaskCost(_JSONMixin):
    latency: int
    ii: int
    terms: Mapping[str, Any] = field(default_factory=dict)
    reliability: Reliability | str = Reliability.MEASURED
    fallback_used: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _integer(self.latency, "task latency")
        _integer(self.ii, "task II")
        if self.latency == 0:
            if self.ii != 0:
                raise ValueError("The empty composition identity must have latency=ii=0")
        elif self.ii < 1 or self.latency < self.ii:
            raise ValueError(f"A schedulable cost requires latency>=ii>0, got {self.latency}>={self.ii}")
        object.__setattr__(self, "reliability", _reliability(self.reliability))
        object.__setattr__(self, "fallback_used", tuple(sorted(set(self.fallback_used))))
        object.__setattr__(self, "terms", dict(self.terms))
        _json_value(self.terms)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskCost:
        return cls(
            latency=data["latency"], ii=data["ii"], terms=data.get("terms", {}),
            reliability=data.get("reliability", Reliability.MEASURED),
            fallback_used=tuple(data.get("fallback_used", ())),
        )


class _EstimateState:
    def __init__(self, params: CudaLatencyParams, reliability: Reliability):
        self.params = params
        self.reliability = reliability
        self.fallback: set[str] = set()
        self.unmeasured: set[str] = set()

    def degrade(self, reliability: Reliability) -> None:
        self.reliability = _worst_reliability(self.reliability, reliability)

    def use(self, key: str) -> None:
        source = self.params.field_sources.get(key, "default")
        if source in {"measured", "structural"}:
            return
        if source == "provided":
            self.unmeasured.add(key)
            self.degrade(Reliability.ESTIMATED)
        elif source == "derived":
            self.fallback.add(f"derived:{key}")
            self.degrade(Reliability.ESTIMATED)
        else:
            self.fallback.add(key)
            self.degrade(Reliability.UNKNOWN)

    def unknown(self, key: str) -> None:
        self.fallback.add(key)
        self.degrade(Reliability.UNKNOWN)


def _resolve_extent(value: Extent | int, name: str, state: _EstimateState, positive: bool = False) -> int:
    if isinstance(value, Extent):
        if value.upper_bound is None:
            raise ValueError(f"{name} needs a finite dynamic extent upper_bound")
        state.degrade(value.reliability)
        result = _integer(value.upper_bound, name, 1 if positive else 0)
    else:
        result = _integer(value, name, 1 if positive else 0)
    return result


def _usable_rate(value: Any, key: str, state: _EstimateState) -> Fraction:
    state.use(key)
    rate = _rate(value, key)
    if rate is None or rate == 0:
        state.unknown(key)
        return UNIT_PLACEHOLDER
    return rate


def _memory_resource(path: str) -> Resource:
    src, dst = path.split("->", 1)
    if src == "global":
        return Resource.GLOBAL_LOAD
    if dst == "global":
        return Resource.GLOBAL_STORE
    return Resource.SHARED_MEMORY


def _geometry(access: MemoryAccess, index: int, state: _EstimateState) -> str:
    text = access.geometry.strip().lower().replace("_", "-")
    if text in {"m", "coalesced", "contiguous", "linear"}:
        return "coalesced"
    if text in {"n", "strided", "row-strided"}:
        return "strided"
    state.unknown(f"memory_geometry[{index}]:{access.geometry}")
    return "unknown"


def _align(extent: int, group: int, name: str) -> int:
    _integer(group, f"{name} atom", 1)
    groups = (extent + group - 1) // group
    return _bounded(_checked_mul(groups, group, name), name)


def _mma_atom(arch: str, dtype: str) -> tuple[tuple[int, int, int], bool, bool]:
    canonical_arch = _canonical_arch(arch)
    known_arch = canonical_arch in {"sm75", "sm80", "sm86", "sm89", "sm90", "sm100", "sm120", "sm121"}
    sm120 = canonical_arch in {"sm120", "sm121"}
    if dtype in {"f16", "bf16"}:
        return (16, 8, 16), True, known_arch
    if dtype == "tf32":
        # GDN's sm120 path uses m16n8k4; older generic mma.sync uses k8.
        return (16, 8, 4 if sm120 else 8), True, known_arch
    if dtype in {"e4m3", "e5m2", "int8"}:
        return (16, 8, 32), True, known_arch
    if dtype == "e2m1":
        return (16, 8, 64), sm120, known_arch
    return (16, 8, 16), False, known_arch


def _supported_acc(dtype: str, acc: str) -> bool:
    if dtype in {"f16", "bf16", "e4m3", "e5m2"}:
        return acc in {"f16", "f32"}
    if dtype in {"tf32", "e2m1"}:
        return acc == "f32"
    if dtype == "int8":
        return acc == "s32"
    return False


def _resource_flags(features: TaskFeatures) -> int:
    mask = Resource.NONE
    for access in features.copies:
        mask |= _memory_resource(_path_name(access.src_scope, access.dst_scope))
    if features.gemms:
        mask |= Resource.TENSOR_CORE
    if any(_static_positive(getattr(features.vector, name)) for name in _VECTOR_OP_NAMES if name not in {"special", "exp2"}):
        mask |= Resource.CUDA_CORE
    if _static_positive(features.vector.read_bytes):
        mask |= Resource.CUDA_CORE
    if _static_positive(features.vector.special) or _static_positive(features.vector.exp2):
        mask |= Resource.SFU
    if features.scalar:
        mask |= Resource.SCALAR
    return int(mask)


def _static_positive(value: Extent | int) -> bool:
    if isinstance(value, Extent):
        return value.upper_bound is None or value.upper_bound > 0
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def estimate_task(
    features: TaskFeatures,
    params: CudaLatencyParams | Calibration | None = None,
    *,
    require_measured: bool = False,
) -> TaskCost:
    """Estimate one leaf task, then apply independent latency/II overrides."""

    if not isinstance(features, TaskFeatures):
        if isinstance(features, Mapping):
            features = TaskFeatures.from_dict(features)
        else:
            raise TypeError("features must be TaskFeatures or a mapping")
    if params is None:
        params = CudaLatencyParams()
    if isinstance(params, Calibration):
        params = params.params
    if not isinstance(params, CudaLatencyParams):
        raise TypeError("params must be CudaLatencyParams, Calibration, or None")
    state = _EstimateState(params, features.reliability)

    if features.latency_override is not None:
        latency_override = _integer(features.latency_override, "latency_override")
    else:
        latency_override = None
    if features.ii_override is not None:
        ii_override = _integer(features.ii_override, "ii_override", 1)
    else:
        ii_override = None

    resource_flags = _resource_flags(features)
    if latency_override is not None and ii_override is not None:
        if latency_override < ii_override:
            raise ValueError("Final override requires latency>=ii>0")
        result = TaskCost(
            latency_override, ii_override,
            {"override": True, "resource_flags": resource_flags},
            state.reliability, tuple(sorted(state.fallback)),
        )
        if require_measured and result.reliability != Reliability.MEASURED:
            raise CalibrationError("Measured task features are required for overridden cost")
        return result

    memory_latency = 0
    memory_ii = 0
    path_iis: dict[str, int] = {}
    memory_details = []
    for index, access in enumerate(features.copies):
        state.degrade(access.reliability)
        payload = _resolve_extent(access.bytes, f"copies[{index}].bytes", state)
        path = _path_name(access.src_scope, access.dst_scope)
        raw_path = params.memory_paths.get(path)
        if raw_path is None:
            item = _mapping_item(params.memory_paths, path, lambda value: value)
            raw_path = item[1] if item else None
        if raw_path is None:
            state.unknown(f"memory_path:{path}")
            raw_path = MemoryPathParams()
        prefix = f"memory_paths.{path}"
        for name in ("base_cycles", "descriptor_cycles", "issue_cycles", "min_ii"):
            state.use(f"{prefix}.{name}")
        base = _integer(raw_path.base_cycles, f"{prefix}.base_cycles")
        descriptor = _integer(raw_path.descriptor_cycles, f"{prefix}.descriptor_cycles")
        issue = _integer(raw_path.issue_cycles, f"{prefix}.issue_cycles")
        min_ii = _integer(raw_path.min_ii, f"{prefix}.min_ii")

        transfer = 0
        effective_bw: Fraction | None = None
        geometry = _geometry(access, index, state)
        if payload:
            bandwidth_key = f"{prefix}.bandwidth_bytes_per_cycle"
            if geometry == "coalesced":
                effective_bw = _usable_rate(raw_path.bandwidth_bytes_per_cycle, bandwidth_key, state)
            else:
                width = _resolve_extent(access.width, f"copies[{index}].width", state)
                direct_key = f"{prefix}.{'strided' if geometry == 'strided' else 'unknown'}_bandwidth_bytes_per_cycle"
                direct_value = getattr(raw_path, f"{'strided' if geometry == 'strided' else 'unknown'}_bandwidth_bytes_per_cycle")
                direct_rate = _rate(direct_value, direct_key)
                if direct_rate is not None and direct_rate > 0:
                    effective_bw = _usable_rate(direct_value, direct_key, state)
                elif geometry == "strided" and width > 0 and raw_path.transaction_bytes > 0:
                    state.use(f"{prefix}.transaction_bytes")
                    base_bw = _usable_rate(raw_path.bandwidth_bytes_per_cycle, bandwidth_key, state)
                    transactions = 1 + (width - 1) // raw_path.transaction_bytes
                    transaction_payload = _checked_mul(
                        raw_path.transaction_bytes, transactions, "memory transaction geometry"
                    )
                    effective_bw = base_bw * Fraction(width, transaction_payload)
                else:
                    if geometry == "strided":
                        if width <= 0:
                            state.unknown(f"copies[{index}].width")
                        else:
                            state.unknown(f"{prefix}.transaction_bytes")
                    unknown_key = f"{prefix}.unknown_bandwidth_bytes_per_cycle"
                    effective_bw = _usable_rate(raw_path.unknown_bandwidth_bytes_per_cycle, unknown_key, state)
            transfer = _ceil_rate(payload, effective_bw, f"copies[{index}].transfer_cycles")

        completion = _checked_add(_checked_add(base, transfer, "memory completion"), descriptor, "memory completion")
        memory_latency = _checked_add(memory_latency, completion, "task memory latency")
        access_ii = _checked_add(max(transfer, min_ii), issue, "memory access II")
        path_iis[path] = _checked_add(path_iis.get(path, 0), access_ii, f"memory path II {path}")
        memory_details.append({
            "path": path, "bytes": payload, "geometry": geometry, "completion_cycles": completion,
            "transfer_cycles": transfer, "ii_cycles": access_ii,
            "bandwidth_bytes_per_cycle": _rate_json(effective_bw),
        })
    if path_iis:
        memory_ii = max(path_iis.values())

    gemm_latency = 0
    gemm_ii = 0
    gemm_details = []
    if features.gemms:
        state.use("arch")
        state.use("gemm_base_cycles")
        state.use("gemm_min_ii")
        state.use("gemm_pipeline_depth")
        gemm_latency = _integer(params.gemm_base_cycles, "gemm_base_cycles")
        gemm_ii = _integer(params.gemm_min_ii, "gemm_min_ii", 1)
        pipeline_depth = _integer(params.gemm_pipeline_depth, "gemm_pipeline_depth")
        gemm_ii = max(gemm_ii, pipeline_depth)
    for index, gemm in enumerate(features.gemms):
        state.degrade(gemm.reliability)
        m = _resolve_extent(gemm.M, f"gemms[{index}].M", state, positive=True)
        n = _resolve_extent(gemm.N, f"gemms[{index}].N", state, positive=True)
        k = _resolve_extent(gemm.K, f"gemms[{index}].K", state, positive=True)
        atom, dtype_supported, arch_supported = _mma_atom(params.arch, gemm.dtype)
        if not dtype_supported:
            state.unknown(f"gemm_dtype:{gemm.dtype}")
        if not arch_supported:
            state.unknown(f"gemm_arch:{params.arch}")
        if not _supported_acc(gemm.dtype, gemm.acc):
            state.unknown(f"gemm_acc:{gemm.dtype}:{gemm.acc}")
        rounded_m = _align(m, atom[0], "GEMM M")
        rounded_n = _align(n, atom[1], "GEMM N")
        rounded_k = _align(k, atom[2], "GEMM K")
        ops = _checked_mul(2, _checked_mul(rounded_m, _checked_mul(rounded_n, rounded_k, "GEMM ops"), "GEMM ops"), "GEMM ops")
        rate_item = _mapping_item(params.gemm_throughput_ops_per_cycle, f"{gemm.dtype}:{gemm.acc}", _dtype_key)
        if rate_item is None:
            rate_item = _mapping_item(params.gemm_throughput_ops_per_cycle, gemm.dtype, _dtype_key)
        if rate_item is None:
            rate_key = f"gemm_throughput_ops_per_cycle.{gemm.dtype}:{gemm.acc}"
            state.unknown(rate_key)
            throughput = UNIT_PLACEHOLDER
        else:
            rate_key = f"gemm_throughput_ops_per_cycle.{rate_item[0]}"
            throughput = _usable_rate(rate_item[1], rate_key, state)
        throughput_cycles = _ceil_rate(ops, throughput, "GEMM throughput cycles")
        gemm_latency = _checked_add(gemm_latency, throughput_cycles, "GEMM latency")
        gemm_ii = max(gemm_ii, throughput_cycles)
        gemm_details.append({
            "M": m, "N": n, "K": k, "rounded_M": rounded_m, "rounded_N": rounded_n,
            "rounded_K": rounded_k, "atom": list(atom), "dtype": gemm.dtype, "acc": gemm.acc,
            "operations": ops, "throughput_cycles": throughput_cycles,
            "dtype_supported": dtype_supported, "arch_supported": arch_supported,
        })

    core_ops_latency = 0
    sfu_latency = 0
    vector_details: dict[str, int] = {}
    state.degrade(features.vector.reliability)
    for name in _VECTOR_OP_NAMES:
        count = _resolve_extent(getattr(features.vector, name), f"vector.{name}", state)
        if not count:
            continue
        if name == "unknown":
            state.unknown("vector_unknown_ops")
        rate_item = _mapping_item(params.vector_throughput_ops_per_cycle, name, str.lower)
        if rate_item is None:
            key = f"vector_throughput_ops_per_cycle.{name}"
            state.unknown(key)
            throughput = UNIT_PLACEHOLDER
        else:
            key = f"vector_throughput_ops_per_cycle.{rate_item[0]}"
            throughput = _usable_rate(rate_item[1], key, state)
        cycles = _ceil_rate(count, throughput, f"vector {name} cycles")
        vector_details[name] = cycles
        if name in {"special", "exp2"}:
            sfu_latency = _checked_add(sfu_latency, cycles, "SFU latency")
        else:
            core_ops_latency = _checked_add(core_ops_latency, cycles, "CUDA core op latency")
    read_bytes = _resolve_extent(features.vector.read_bytes, "vector.read_bytes", state)
    vector_bw_latency = 0
    if read_bytes:
        vector_bw = _usable_rate(params.vector_bandwidth_bytes_per_cycle, "vector_bandwidth_bytes_per_cycle", state)
        vector_bw_latency = _ceil_rate(read_bytes, vector_bw, "vector bandwidth cycles")
    core_latency = max(core_ops_latency, vector_bw_latency)
    vector_latency = max(core_latency, sfu_latency)
    vector_active = bool(vector_details or read_bytes)

    scalar_latency = 0
    if features.scalar:
        state.use("scalar_cycles")
        scalar_latency = _checked_mul(features.scalar, params.scalar_cycles, "scalar latency")

    compute_latency = max(
        gemm_latency if features.gemms else 0,
        vector_latency if vector_active else 0,
        scalar_latency,
    )
    estimated_latency = _checked_add(memory_latency, compute_latency, "task latency")
    estimated_ii = max([
        1,
        memory_ii if features.copies else 0,
        gemm_ii if features.gemms else 0,
        vector_latency if vector_active else 0,
        scalar_latency,
    ])
    if estimated_latency == 0:
        estimated_latency = 1
    final_latency = latency_override if latency_override is not None else estimated_latency
    final_ii = ii_override if ii_override is not None else estimated_ii
    if final_latency < final_ii:
        raise ValueError(f"Final task cost requires latency>=ii>0, got {final_latency}>={final_ii}")

    terms = {
        "memory_latency": memory_latency,
        "compute_latency": compute_latency,
        "memory_ii": memory_ii,
        "gemm_latency": gemm_latency if features.gemms else 0,
        "gemm_ii": gemm_ii if features.gemms else 0,
        "vector_latency": vector_latency,
        "vector_ii": vector_latency if vector_active else 0,
        "scalar_latency": scalar_latency,
        "scalar_ii": scalar_latency,
        "estimated_latency": estimated_latency,
        "estimated_ii": estimated_ii,
        "latency_override": latency_override,
        "ii_override": ii_override,
        "resource_flags": resource_flags,
        "resources": [resource.name for resource in Resource if resource not in {Resource.NONE, Resource.ALL} and resource_flags & resource],
        "memory_accesses": memory_details,
        "memory_path_iis": path_iis,
        "gemms": gemm_details,
        "vector_op_cycles": vector_details,
        "vector_bandwidth_cycles": vector_bw_latency,
        "units": {"latency": "cycles", "ii": "cycles", "bandwidth": "bytes/cycle", "throughput": "ops/cycle"},
    }
    result = TaskCost(final_latency, final_ii, terms, state.reliability, tuple(sorted(state.fallback)))
    if require_measured and (result.reliability != Reliability.MEASURED or result.fallback_used or state.unmeasured):
        reasons = list(result.fallback_used) + sorted(state.unmeasured)
        if not reasons:
            reasons = [f"reliability:{result.reliability.value}"]
        raise CalibrationError("Measured task estimate required; unavailable: " + ", ".join(reasons))
    return result


def _combined_metadata(costs: Iterable[TaskCost]) -> tuple[Reliability, tuple[str, ...]]:
    costs = tuple(costs)
    reliability = _worst_reliability(*(cost.reliability for cost in costs))
    fallback = tuple(sorted({item for cost in costs for item in cost.fallback_used}))
    return reliability, fallback


def _identity(terms: Mapping[str, Any], reliability: Reliability, fallback: tuple[str, ...]) -> TaskCost:
    return TaskCost(0, 0, terms, reliability, fallback)


def compose_serial(costs: Iterable[TaskCost] | TaskCost, *additional: TaskCost) -> TaskCost:
    """Compose A+B as max(A.latency, A.ii+B.latency), II=A.ii+B.ii."""

    items = (costs, *additional) if isinstance(costs, TaskCost) else (*tuple(costs), *additional)
    if not all(isinstance(cost, TaskCost) for cost in items):
        raise TypeError("All composed costs must be TaskCost instances")
    latency = 0
    ii = 0
    for cost in items:
        if cost.ii == 0:
            continue
        if ii == 0:
            latency, ii = cost.latency, cost.ii
        else:
            latency = max(latency, _checked_add(ii, cost.latency, "serial latency"))
            ii = _checked_add(ii, cost.ii, "serial II")
    reliability, fallback = _combined_metadata(items)
    terms = {"composition": "serial", "children": [{"latency": cost.latency, "ii": cost.ii} for cost in items]}
    if ii == 0:
        return _identity(terms, reliability, fallback)
    return TaskCost(latency, ii, terms, reliability, fallback)


def serial(left: TaskCost, right: TaskCost) -> TaskCost:
    return compose_serial((left, right))


def repeat(cost: TaskCost, trips: int) -> TaskCost:
    """Repeat serially: latency=(n-1)*II+latency, II=n*II."""

    if not isinstance(cost, TaskCost):
        raise TypeError("cost must be TaskCost")
    trips = _integer(trips, "repeat trips")
    reliability, fallback = _combined_metadata((cost,))
    terms = {"composition": "repeat", "trips": trips, "body": {"latency": cost.latency, "ii": cost.ii}}
    if trips == 0 or cost.ii == 0:
        return _identity(terms, reliability, fallback)
    latency = _checked_add(_checked_mul(cost.ii, trips - 1, "repeat latency"), cost.latency, "repeat latency")
    ii = _checked_mul(cost.ii, trips, "repeat II")
    return TaskCost(latency, ii, terms, reliability, fallback)


def branch(then_cost: TaskCost, else_cost: TaskCost) -> TaskCost:
    """Mutually exclusive branches take component-wise maxima."""

    if not isinstance(then_cost, TaskCost) or not isinstance(else_cost, TaskCost):
        raise TypeError("Branches must be TaskCost instances")
    reliability, fallback = _combined_metadata((then_cost, else_cost))
    latency = max(then_cost.latency, else_cost.latency)
    ii = max(then_cost.ii, else_cost.ii)
    terms = {
        "composition": "branch",
        "then": {"latency": then_cost.latency, "ii": then_cost.ii},
        "else": {"latency": else_cost.latency, "ii": else_cost.ii},
    }
    if latency == 0:
        return _identity(terms, reliability, fallback)
    return TaskCost(latency, ii, terms, reliability, fallback)


def aggregate_phase(candidates: Iterable[TaskCost]) -> TaskCost:
    """Candidates in one per-core phase run concurrently across cores."""

    items = tuple(candidates)
    if not all(isinstance(cost, TaskCost) for cost in items):
        raise TypeError("Phase candidates must be TaskCost instances")
    reliability, fallback = _combined_metadata(items)
    terms = {"composition": "per_core_phase", "candidates": [{"latency": cost.latency, "ii": cost.ii} for cost in items]}
    if not items:
        return _identity(terms, reliability, fallback)
    return TaskCost(max(cost.latency for cost in items), max(cost.ii for cost in items), terms, reliability, fallback)


def aggregate_per_core(
    phases: Iterable[Iterable[TaskCost] | TaskCost],
    *,
    cross_wait: bool | None = None,
) -> TaskCost:
    """Max within phases, add across cross-wait-delimited phases.

    With a real cross-wait boundary, II equals total latency because the
    explicit protocol state is unversioned.  Without a boundary, candidates'
    II values retain their independent resource meaning.
    """

    raw = tuple(phases)
    if raw and all(isinstance(item, TaskCost) for item in raw):
        raw = (raw,)  # One phase supplied directly.
    if cross_wait is not None and not isinstance(cross_wait, bool):
        raise TypeError("cross_wait must be bool or None")
    boundary = len(raw) > 1 if cross_wait is None else cross_wait
    if len(raw) > 1 and cross_wait is False:
        raise ValueError("Multiple serial phases require cross_wait=True")
    phase_costs = tuple(aggregate_phase(item) if not isinstance(item, TaskCost) else aggregate_phase((item,)) for item in raw)
    reliability, fallback = _combined_metadata(phase_costs)
    latency = 0
    accumulated_ii = 0
    for cost in phase_costs:
        latency = _checked_add(latency, cost.latency, "per-core phase latency")
        accumulated_ii = _checked_add(accumulated_ii, cost.ii, "per-core phase II")
    terms = {
        "composition": "per_core", "cross_wait": boundary,
        "phases": [{"latency": cost.latency, "ii": cost.ii} for cost in phase_costs],
    }
    if latency == 0:
        return _identity(terms, reliability, fallback)
    final_ii = latency if boundary else max(accumulated_ii, 1)
    return TaskCost(latency, final_ii, terms, reliability, fallback)


def repeat_per_core(cost: TaskCost, trips: int, *, has_cross_wait: bool = False) -> TaskCost:
    """Repeat a per-core body; unbounded phases use conservative n*latency."""

    if not isinstance(cost, TaskCost):
        raise TypeError("cost must be TaskCost")
    trips = _integer(trips, "per-core repeat trips")
    if not isinstance(has_cross_wait, bool):
        raise TypeError("has_cross_wait must be bool")
    reliability, fallback = _combined_metadata((cost,))
    terms = {
        "composition": "repeat_per_core", "trips": trips, "cross_wait": has_cross_wait,
        "body": {"latency": cost.latency, "ii": cost.ii},
    }
    if trips == 0 or cost.latency == 0:
        return _identity(terms, reliability, fallback)
    latency = _checked_mul(cost.latency, trips, "per-core repeat latency")
    body_ii = cost.latency if has_cross_wait else cost.ii
    ii = _checked_mul(body_ii, trips, "per-core repeat II")
    return TaskCost(latency, ii, terms, reliability, fallback)


compose_repeat = repeat
compose_branch = branch
per_core_phase = aggregate_phase
per_core = aggregate_per_core


__all__ = [
    "INT64_MAX", "Calibration", "CalibrationError", "CudaLatencyParams", "DynamicExtent", "Extent",
    "GemmFeature", "MemoryAccess", "MemoryPathParams", "Reliability", "Resource", "TaskCost",
    "TaskFeatures", "VectorOps", "aggregate_per_core", "aggregate_phase", "branch", "compose_branch",
    "compose_repeat", "compose_serial", "estimate_task", "per_core", "per_core_phase", "repeat",
    "repeat_per_core", "serial",
]
