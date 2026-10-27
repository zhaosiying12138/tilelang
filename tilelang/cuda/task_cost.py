"""FFI-friendly CUDA task costs backed by :mod:`latency_model`.

``tl.cuda.estimate_task`` accepts JSON-like task features and explicit
hardware/calibration inputs. Hardware supplies the architecture and memory
capacities; calibration alone supplies performance rates. No device probing,
GPU import, architecture default, capacity default, or formula duplication is
performed here. ``require_measured`` is forwarded to both Calibration and the
leaf estimator so fallback/unit-placeholder results cannot masquerade as
measurements.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import operator
import os
from pathlib import Path
import re
from typing import Any

try:  # Normal package import.
    from . import latency_model as _lm
except ImportError:  # Standalone CPU tests loaded directly by file path.
    import importlib.util

    _path = Path(__file__).with_name("latency_model.py")
    _spec = importlib.util.spec_from_file_location("_tilelang_cuda_latency_model_for_task_cost", _path)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Cannot load CUDA latency model from {_path}") from None
    _lm = importlib.util.module_from_spec(_spec)
    import sys

    sys.modules[_spec.name] = _lm
    _spec.loader.exec_module(_lm)

try:  # Keep this module importable in formula-only CPU environments.
    import tvm_ffi as _tvm_ffi
except ImportError:  # pragma: no cover - depends on the host environment.
    _tvm_ffi = None


HARDWARE_SCHEMA = "tilelang.cuda.hardware-spec"


def _register_global(name: str):
    if _tvm_ffi is None:
        return lambda function: function

    def register(function):
        _tvm_ffi.register_global_func(name, f=function, override=True)
        return function

    return register


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key must be str, got {type(key).__name__}")
            result[str(key)] = _plain(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    if isinstance(value, str):
        return str(value)
    value_type = type(value)
    if value_type.__name__ == "IntImm" and value_type.__module__.startswith("tvm."):
        return int(value.value)
    return value


def _mapping(value: Any, description: str) -> tuple[Mapping[str, Any], str, bool]:
    if isinstance(value, Mapping):
        # Strict JSON round-tripping rejects arbitrary objects and non-finite
        # values instead of silently stringifying them for the FFI boundary.
        encoded = json.dumps(_plain(value), allow_nan=False)
        decoded = json.loads(encoded)
        if not isinstance(decoded, dict):
            raise TypeError(f"{description} must be a JSON object")
        return decoded, "<mapping>", False
    if isinstance(value, (str, os.PathLike)):
        path = os.fspath(value)
        if not path:
            raise ValueError(f"{description} path must not be empty")
        with open(path, encoding="utf-8") as stream:
            decoded = json.load(stream)
        if not isinstance(decoded, dict):
            raise TypeError(f"{description} JSON must contain an object")
        return decoded, path, True
    raise TypeError(f"{description} must be a mapping or JSON file path")


def _hardware_parts(hardware: Mapping[str, Any] | str | os.PathLike[str]) -> tuple[dict[str, Any], str, str]:
    spec, source, _ = _mapping(hardware, "hardware")
    if spec.get("schema") != HARDWARE_SCHEMA:
        raise ValueError(f"hardware.schema must be {HARDWARE_SCHEMA!r}")
    device = spec.get("device")
    if not isinstance(device, dict):
        raise ValueError("hardware.device must be an object")
    capability = device.get("compute_capability")
    if not isinstance(capability, str) or re.fullmatch(r"[0-9]+\.[0-9]+", capability) is None:
        raise ValueError("hardware.device.compute_capability must have the form '<major>.<minor>'")
    arch = "sm" + capability.replace(".", "")
    return spec, arch, source


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer, got {value!r}") from error
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


@_register_global("tl.cuda.z3_memory_group_capacities")
def memory_group_capacities(
    hardware: Mapping[str, Any] | str | os.PathLike[str],
) -> dict[str, int]:
    """Return only capacities explicitly present in a hardware spec.

    The Z3 loop solver models automatic shared-buffer versions in one group.
    CUDA's per-block opt-in limit is the relevant explicit capacity. The
    ordinary/static limit is not silently substituted when opt-in is absent.
    Other persistent allocations and downstream storage reuse are outside this
    group; callers must account for them separately when they are present.
    """

    spec, _, _ = _hardware_parts(hardware)
    device = spec["device"]
    if "shared_memory_per_block_optin_bytes" not in device:
        raise ValueError(
            "hardware.device.shared_memory_per_block_optin_bytes is required; "
            "no shared-memory capacity default is defined"
        )
    capacity = _positive_integer(
        device["shared_memory_per_block_optin_bytes"],
        "hardware.device.shared_memory_per_block_optin_bytes",
    )
    return {"shared": capacity}


@_register_global("tl.cuda.estimate_task")
def estimate_task(
    features: _lm.TaskFeatures | Mapping[str, Any] | str,
    hardware: Mapping[str, Any] | str | os.PathLike[str],
    calibration: Mapping[str, Any] | str | os.PathLike[str],
    require_measured: bool = False,
) -> dict[str, Any]:
    """Estimate one task and return a strict-JSON-compatible dictionary.

    Parameters are deliberately explicit. ``hardware`` and ``calibration`` may
    be mappings for CPU use or JSON paths for the C++ FFI. A calibration path is
    loaded with :meth:`Calibration.load`; a mapping uses
    :meth:`Calibration.from_json` with the same architecture and measured-only
    arguments. Returned keys are ``latency``, ``ii``, ``terms``,
    ``resource_flags``, ``reliability``, and ``fallback_used``.
    """

    if not isinstance(require_measured, bool):
        raise TypeError("require_measured must be bool")
    _, arch, _ = _hardware_parts(hardware)
    if isinstance(features, str):
        features = json.loads(features)
    if isinstance(features, Mapping):
        features = _lm.TaskFeatures.from_dict(_plain(features))
    if not isinstance(features, _lm.TaskFeatures):
        raise TypeError("features must be TaskFeatures, a JSON object, or a mapping")

    if isinstance(calibration, (str, os.PathLike)):
        calibrated = _lm.Calibration.load(
            calibration,
            arch=arch,
            require_measured=require_measured,
        )
    else:
        raw_calibration, source, _ = _mapping(calibration, "calibration")
        calibrated = _lm.Calibration.from_json(
            raw_calibration,
            arch=arch,
            require_measured=require_measured,
            source=source,
        )
    cost = _lm.estimate_task(
        features,
        calibrated,
        require_measured=require_measured,
    )
    result = cost.to_dict()
    result["resource_flags"] = int(cost.terms["resource_flags"])
    # Verify the complete FFI result, including nested terms, rather than only
    # its top-level scalar fields.
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # Defensive: cost.to_dict() is an object.
        raise TypeError("Task cost result must be a JSON object")
    return decoded


def estimate_composed_task(
    tree: Mapping[str, Any] | str,
    hardware: Mapping[str, Any] | str | os.PathLike[str],
    calibration: Mapping[str, Any] | str | os.PathLike[str],
    require_measured: bool = False,
) -> dict[str, Any]:
    """Evaluate a composed cost tree for the future native analyzer.

    This intentionally is not registered as an FFI global yet.  The later
    ``tl.cuda.estimate_composed_task`` binding can call this function directly
    after the native ``NativeCostExpr`` extractor is available.  Inputs and
    results otherwise follow the same strict JSON, hardware, calibration, and
    measured-only contracts as :func:`estimate_task`.
    """

    if not isinstance(require_measured, bool):
        raise TypeError("require_measured must be bool")
    _, arch, _ = _hardware_parts(hardware)
    if isinstance(tree, str):
        tree = json.loads(tree)
    if not isinstance(tree, Mapping):
        raise TypeError("tree must be a JSON object, JSON string, or mapping")
    encoded_tree = json.dumps(_plain(tree), allow_nan=False)
    decoded_tree = json.loads(encoded_tree)
    if not isinstance(decoded_tree, dict):
        raise TypeError("tree must be a JSON object")

    if isinstance(calibration, (str, os.PathLike)):
        calibrated = _lm.Calibration.load(
            calibration,
            arch=arch,
            require_measured=require_measured,
        )
    else:
        raw_calibration, source, _ = _mapping(calibration, "calibration")
        calibrated = _lm.Calibration.from_json(
            raw_calibration,
            arch=arch,
            require_measured=require_measured,
            source=source,
        )
    cost = _lm.estimate_composed_task(
        decoded_tree,
        calibrated,
        require_measured=require_measured,
    )
    result = cost.to_dict()
    result["resource_flags"] = int(cost.terms["resource_flags"])
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError("Composed task cost result must be a JSON object")
    return decoded


__all__ = ["HARDWARE_SCHEMA", "estimate_composed_task", "estimate_task", "memory_group_capacities"]
