"""JSON-only CUDA hardware attribute probing and validation.

The probe reports device/driver attributes, not measured performance.  It never
fills an unavailable value from a GPU model table or from another attribute.
All optional dependencies are imported lazily so importing this module neither
initializes CUDA nor requires a built TileLang package.
"""

from __future__ import annotations

import ctypes
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

SCHEMA = "tilelang.cuda.hardware-spec"
SCHEMA_VERSION = 1
COMMAND_TIMEOUT_SECONDS = 10.0

# shared_memory_per_block_bytes is CUDA's ordinary/static block limit.
# shared_memory_per_block_optin_bytes is the opt-in upper bound, not a
# dynamic-only allocation size (static and driver-reserved memory also count).
_DEVICE_FIELDS = (
    "name",
    "compute_capability",
    "total_memory_bytes",
    "multi_processor_count",
    "warp_size",
    "max_threads_per_block",
    "max_threads_per_multiprocessor",
    "max_registers_per_block",
    "max_registers_per_multiprocessor",
    "max_registers_per_thread",
    "shared_memory_per_block_bytes",
    "shared_memory_per_block_optin_bytes",
    "shared_memory_per_multiprocessor_bytes",
    "reserved_shared_memory_per_block_bytes",
    "l2_cache_size_bytes",
    "max_persisting_l2_cache_size_bytes",
    "clock_rate_khz",
    "memory_clock_rate_khz",
    "memory_bus_width_bits",
)

_TORCH_PROPERTIES = {
    "name": ("name",),
    "total_memory_bytes": ("total_memory", "totalMemory"),
    "multi_processor_count": ("multi_processor_count", "multiProcessorCount"),
    "warp_size": ("warp_size", "warpSize"),
    "max_threads_per_block": ("max_threads_per_block", "maxThreadsPerBlock"),
    "max_threads_per_multiprocessor": ("max_threads_per_multi_processor", "max_threads_per_multiprocessor", "maxThreadsPerMultiProcessor"),
    "max_registers_per_block": ("regs_per_block", "max_registers_per_block", "regsPerBlock"),
    "max_registers_per_multiprocessor": ("regs_per_multiprocessor", "regs_per_multi_processor", "regsPerMultiprocessor"),
    # PyTorch normally does not expose this.  Do not derive it from registers
    # per SM divided by threads per SM: that quotient is a different quantity.
    "max_registers_per_thread": ("max_registers_per_thread", "regsPerThread"),
    "shared_memory_per_block_bytes": ("shared_memory_per_block", "sharedMemPerBlock"),
    "shared_memory_per_block_optin_bytes": ("shared_memory_per_block_optin", "sharedMemPerBlockOptin"),
    "shared_memory_per_multiprocessor_bytes": (
        "shared_memory_per_multi_processor",
        "shared_memory_per_multiprocessor",
        "sharedMemPerMultiprocessor",
    ),
    "reserved_shared_memory_per_block_bytes": ("reserved_shared_memory_per_block", "reservedSharedMemPerBlock"),
    "l2_cache_size_bytes": ("L2_cache_size", "l2_cache_size", "l2CacheSize"),
    "max_persisting_l2_cache_size_bytes": ("max_persisting_l2_cache_size", "maxPersistingL2CacheSize"),
    "clock_rate_khz": ("clock_rate", "clockRate"),
    "memory_clock_rate_khz": ("memory_clock_rate", "memoryClockRate"),
    "memory_bus_width_bits": ("memory_bus_width", "memoryBusWidth"),
}

# Numeric values are stable cudaDeviceAttr values from driver_types.h.  They
# are used only by the ctypes CUDA Runtime fallback.  cuda-bindings uses the
# symbolic enum when available.  There is deliberately no numeric fallback for
# max registers/thread because CUDA Runtime does not define a stable public
# cudaDevAttrMaxRegistersPerThread constant.
_RUNTIME_ATTRIBUTES = {
    "multi_processor_count": ("cudaDevAttrMultiProcessorCount", 16),
    "warp_size": ("cudaDevAttrWarpSize", 10),
    "max_threads_per_block": ("cudaDevAttrMaxThreadsPerBlock", 1),
    "max_threads_per_multiprocessor": ("cudaDevAttrMaxThreadsPerMultiProcessor", 39),
    "max_registers_per_block": ("cudaDevAttrMaxRegistersPerBlock", 12),
    "max_registers_per_multiprocessor": ("cudaDevAttrMaxRegistersPerMultiprocessor", 82),
    "max_registers_per_thread": ("cudaDevAttrMaxRegistersPerThread", None),
    "shared_memory_per_block_bytes": ("cudaDevAttrMaxSharedMemoryPerBlock", 8),
    "shared_memory_per_block_optin_bytes": ("cudaDevAttrMaxSharedMemoryPerBlockOptin", 97),
    "shared_memory_per_multiprocessor_bytes": ("cudaDevAttrMaxSharedMemoryPerMultiprocessor", 81),
    "reserved_shared_memory_per_block_bytes": ("cudaDevAttrReservedSharedMemoryPerBlock", 111),
    "l2_cache_size_bytes": ("cudaDevAttrL2CacheSize", 38),
    "max_persisting_l2_cache_size_bytes": ("cudaDevAttrMaxPersistingL2CacheSize", 108),
    "clock_rate_khz": ("cudaDevAttrClockRate", 13),
    "memory_clock_rate_khz": ("cudaDevAttrMemoryClockRate", 36),
    "memory_bus_width_bits": ("cudaDevAttrGlobalMemoryBusWidth", 37),
}

_SOFTWARE_FIELDS = (
    "driver_version",
    "cuda_driver_api_version",
    "cuda_runtime_version",
    "pytorch_version",
    "pytorch_cuda_version",
    "nvcc_version",
    "z3_version",
)

# These are the attributes needed to construct the experiment's hardware
# constraints.  Optional observations (notably max registers/thread and
# reserved shared memory) still have to carry a reason when they are null.
_REQUIRED_DEVICE_FIELDS = (
    "name",
    "compute_capability",
    "total_memory_bytes",
    "multi_processor_count",
    "warp_size",
    "max_threads_per_block",
    "max_threads_per_multiprocessor",
    "max_registers_per_block",
    "max_registers_per_multiprocessor",
    "shared_memory_per_block_bytes",
    "shared_memory_per_block_optin_bytes",
    "shared_memory_per_multiprocessor_bytes",
    "l2_cache_size_bytes",
    "max_persisting_l2_cache_size_bytes",
    "clock_rate_khz",
    "memory_clock_rate_khz",
)
_NONNEGATIVE_DEVICE_FIELDS = {
    "shared_memory_per_block_optin_bytes",
    "l2_cache_size_bytes",
    "max_persisting_l2_cache_size_bytes",
    "reserved_shared_memory_per_block_bytes",
}


class CudaHardwareProbeError(RuntimeError):
    """Raised when a real CUDA device cannot be queried at all."""


class HardwareSpecValidationError(ValueError):
    """Validation failure with core and calibration errors kept separate."""

    def __init__(self, errors: list[str], calibration_errors: list[str] | None = None):
        self.errors = tuple(errors)
        self.calibration_errors = tuple(calibration_errors or ())
        sections = []
        if self.errors:
            sections.append("Hardware spec errors:\n" + "\n".join(f"  - {error}" for error in self.errors))
        if self.calibration_errors:
            sections.append(
                "Calibration errors:\n" + "\n".join(f"  - {error}" for error in self.calibration_errors)
            )
        super().__init__("\n".join(sections) or "Hardware spec validation failed")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_device(device_index: int) -> dict[str, Any]:
    device = {"index": device_index}
    device.update({field: None for field in _DEVICE_FIELDS})
    device["attribute_metadata"] = {
        field: {"source": None, "reason": "Attribute has not been queried yet"} for field in _DEVICE_FIELDS
    }
    return device


def _new_software() -> dict[str, Any]:
    software = {field: None for field in _SOFTWARE_FIELDS}
    software["commands"] = {}
    software["software_metadata"] = {
        field: {"source": None, "reason": "Software version has not been queried yet"}
        for field in _SOFTWARE_FIELDS
    }
    return software


def _record(container: dict[str, Any], metadata_key: str, field: str, value: Any, source: str) -> None:
    container[field] = value
    container[metadata_key][field] = {"source": source, "reason": None}


def _mark_missing(container: dict[str, Any], metadata_key: str, field: str, reason: str) -> None:
    container[field] = None
    entry = container[metadata_key][field]
    old_reason = entry.get("reason")
    reason = str(reason).strip() or "Unavailable for an unspecified reason"
    default_reasons = {"Attribute has not been queried yet", "Software version has not been queried yet"}
    if old_reason and old_reason not in default_reasons and old_reason != reason:
        reason = f"{old_reason}; {reason}"
    entry["source"] = None
    entry["reason"] = reason


def _property_value(properties: Any, names: tuple[str, ...]) -> Any:
    errors = []
    for name in names:
        try:
            value = getattr(properties, name)
        except AttributeError:
            continue
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
        if value is not None:
            return value
    detail = f" ({'; '.join(errors)})" if errors else ""
    raise AttributeError(f"torch.cuda device properties do not expose {'/'.join(names)}{detail}")


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer, not bool")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise ValueError(f"{field} must be a finite integer")
    return int(value)


def _device_name(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Device name must be a non-empty string")
    return value.strip()


def _collect_torch_attributes(device: dict[str, Any], properties: Any) -> None:
    for field, names in _TORCH_PROPERTIES.items():
        try:
            value = _property_value(properties, names)
            if field == "name":
                value = _device_name(value)
            else:
                value = _integer(value, field)
        except Exception as exc:
            _mark_missing(device, "attribute_metadata", field, str(exc))
        else:
            _record(device, "attribute_metadata", field, value, f"torch.cuda.get_device_properties.{names[0]}")

    try:
        major = _integer(_property_value(properties, ("major", "compute_capability_major")), "major")
        minor = _integer(_property_value(properties, ("minor", "compute_capability_minor")), "minor")
        capability = f"{major}.{minor}"
    except Exception as exc:
        _mark_missing(device, "attribute_metadata", "compute_capability", str(exc))
    else:
        _record(
            device,
            "attribute_metadata",
            "compute_capability",
            capability,
            "torch.cuda.get_device_properties.major/minor",
        )


def _error_code(error: Any) -> int:
    value = getattr(error, "value", error)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _unpack_cuda_result(result: Any, operation: str) -> Any:
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError(f"{operation} returned an unexpected result: {result!r}")
    error, value = result
    if _error_code(error) != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {error} (code {_error_code(error)})")
    return value


class _BindingRuntime:
    def __init__(self, module: Any, source: str):
        self.module = module
        self.source = source

    def _attribute(self, enum_name: str, numeric_value: int | None) -> Any:
        enum_type = getattr(self.module, "cudaDeviceAttr", None)
        attribute = getattr(enum_type, enum_name, None) if enum_type is not None else None
        if attribute is None:
            attribute = getattr(self.module, enum_name, None)
        if attribute is None:
            attribute = numeric_value
        if attribute is None:
            raise RuntimeError(f"{self.source} does not expose {enum_name}; no numeric fallback is defined")
        return attribute

    def get_attribute(self, enum_name: str, numeric_value: int | None, device_index: int) -> int:
        attribute = self._attribute(enum_name, numeric_value)
        result = self.module.cudaDeviceGetAttribute(attribute, device_index)
        return _integer(_unpack_cuda_result(result, f"{self.source}.cudaDeviceGetAttribute({enum_name})"), enum_name)

    def _version(self, function_name: str) -> int:
        function = getattr(self.module, function_name)
        return _integer(_unpack_cuda_result(function(), f"{self.source}.{function_name}"), function_name)

    def get_runtime_version(self) -> int:
        return self._version("cudaRuntimeGetVersion")

    def get_driver_version(self) -> int:
        return self._version("cudaDriverGetVersion")


class _CtypesRuntime:
    def __init__(self, library: Any, source: str):
        self.library = library
        self.source = source
        self._get_attribute = library.cudaDeviceGetAttribute
        self._get_attribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        self._get_attribute.restype = ctypes.c_int

    def _version_function(self, name: str) -> Any:
        function = getattr(self.library, name)
        function.argtypes = [ctypes.POINTER(ctypes.c_int)]
        function.restype = ctypes.c_int
        return function

    def _check(self, error: int, operation: str) -> None:
        if error != 0:
            raise RuntimeError(f"{operation} failed with CUDA error code {error}")

    def get_attribute(self, enum_name: str, numeric_value: int | None, device_index: int) -> int:
        if numeric_value is None:
            raise RuntimeError(f"CUDA Runtime has no stable numeric constant for {enum_name}")
        value = ctypes.c_int()
        error = self._get_attribute(ctypes.byref(value), numeric_value, device_index)
        self._check(error, f"{self.source}.cudaDeviceGetAttribute({enum_name})")
        return int(value.value)

    def _version(self, function_name: str) -> int:
        function = self._version_function(function_name)
        value = ctypes.c_int()
        error = function(ctypes.byref(value))
        self._check(error, f"{self.source}.{function_name}")
        return int(value.value)

    def get_runtime_version(self) -> int:
        return self._version("cudaRuntimeGetVersion")

    def get_driver_version(self) -> int:
        return self._version("cudaDriverGetVersion")


def _runtime_library_names() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll", "cudart64_11.dll")
    return ("libcudart.so", "libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so.10.2")


def _load_runtime_backends() -> tuple[list[Any], list[str]]:
    backends = []
    errors = []
    binding_module = None
    binding_source = None
    for module_name, source in (
        ("cuda.bindings.runtime", "cuda.bindings.runtime"),
        ("cuda.cudart", "cuda.cudart"),
    ):
        try:
            binding_module = importlib.import_module(module_name)
            binding_source = source
            break
        except Exception as exc:
            errors.append(f"{module_name} import failed: {exc}")
    if binding_module is not None:
        try:
            backends.append(_BindingRuntime(binding_module, binding_source))
        except Exception as exc:
            errors.append(f"{binding_source} initialization failed: {exc}")

    for library_name in _runtime_library_names():
        try:
            library = ctypes.CDLL(library_name)
            backends.append(_CtypesRuntime(library, f"CUDA Runtime {library_name} via ctypes"))
            break
        except Exception as exc:
            errors.append(f"{library_name} load failed: {exc}")
    return backends, errors


def _runtime_attribute(
    backends: list[Any], backend_errors: list[str], enum_name: str, numeric_value: int | None, device_index: int
) -> tuple[int, str]:
    errors = []
    for backend in backends:
        try:
            return backend.get_attribute(enum_name, numeric_value, device_index), f"{backend.source}.{enum_name}"
        except Exception as exc:
            errors.append(str(exc))
    if not backends:
        errors.extend(backend_errors)
    detail = "; ".join(error for error in errors if error)
    raise RuntimeError(detail or f"No CUDA Runtime backend could query {enum_name}")


def _collect_runtime_attributes(
    device: dict[str, Any], backends: list[Any], backend_errors: list[str], device_index: int
) -> None:
    for field, (enum_name, numeric_value) in _RUNTIME_ATTRIBUTES.items():
        if device[field] is not None:
            continue
        try:
            value, source = _runtime_attribute(backends, backend_errors, enum_name, numeric_value, device_index)
        except Exception as exc:
            _mark_missing(device, "attribute_metadata", field, str(exc))
        else:
            _record(device, "attribute_metadata", field, value, source)

    if device["compute_capability"] is None:
        try:
            major, major_source = _runtime_attribute(
                backends, backend_errors, "cudaDevAttrComputeCapabilityMajor", 75, device_index
            )
            minor, minor_source = _runtime_attribute(
                backends, backend_errors, "cudaDevAttrComputeCapabilityMinor", 76, device_index
            )
        except Exception as exc:
            _mark_missing(device, "attribute_metadata", "compute_capability", str(exc))
        else:
            sources = ", ".join(dict.fromkeys((major_source, minor_source)))
            _record(device, "attribute_metadata", "compute_capability", f"{major}.{minor}", sources)

    for field in _DEVICE_FIELDS:
        if device[field] is None:
            entry = device["attribute_metadata"][field]
            if not entry.get("reason") or entry["reason"] == "Attribute has not been queried yet":
                _mark_missing(
                    device,
                    "attribute_metadata",
                    field,
                    "Neither torch.cuda.get_device_properties nor an available CUDA Runtime API exposed this attribute",
                )


def _format_cuda_version(value: int) -> str:
    # cudaRuntimeGetVersion/cudaDriverGetVersion encode major * 1000 + minor * 10.
    return f"{value // 1000}.{(value % 1000) // 10}"


def _runtime_version(backends: list[Any], backend_errors: list[str], function_name: str) -> tuple[int, str]:
    errors = []
    for backend in backends:
        try:
            return getattr(backend, function_name)(), f"{backend.source}.{function_name}"
        except Exception as exc:
            errors.append(str(exc))
    if not backends:
        errors.extend(backend_errors)
    raise RuntimeError("; ".join(error for error in errors if error) or f"Unable to query {function_name}")


def _readonly_subprocess_environment() -> dict[str, str]:
    # Do not inherit credentials or serialize the environment.  PATH is needed
    # for tool subprocesses and SystemRoot is required by many Windows processes.
    environment = {
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", os.defpath),
    }
    for key in ("SYSTEMROOT", "WINDIR"):
        value = os.environ.get(key)
        if value is not None:
            environment[key] = value
    return environment


def _text_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _command_not_run(command: list[str], reason: str) -> dict[str, Any]:
    return {
        "command": [str(part) for part in command],
        "started_at": None,
        "finished_at": None,
        "duration_seconds": None,
        "timeout_seconds": COMMAND_TIMEOUT_SECONDS,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "error": reason,
    }


def _run_readonly_command(command: list[str], timeout: float = COMMAND_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Run a shell-free read-only command with a bounded, sanitized environment."""
    command = [str(part) for part in command]
    started_at = _utc_timestamp()
    started = time.monotonic()
    record = {
        "command": command,
        "started_at": started_at,
        "finished_at": None,
        "duration_seconds": None,
        "timeout_seconds": float(timeout),
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "error": None,
    }
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            shell=False,
            env=_readonly_subprocess_environment(),
        )
        record.update(
            {
                "returncode": int(completed.returncode),
                "stdout": _text_output(completed.stdout),
                "stderr": _text_output(completed.stderr),
            }
        )
        if completed.returncode != 0:
            record["error"] = f"Command exited with status {completed.returncode}"
    except subprocess.TimeoutExpired as exc:
        record["stdout"] = _text_output(exc.stdout)
        record["stderr"] = _text_output(exc.stderr)
        record["error"] = f"Command timed out after {timeout} seconds"
    except OSError as exc:
        record["error"] = f"Command could not be executed: {exc}"
    record["finished_at"] = _utc_timestamp()
    record["duration_seconds"] = round(max(0.0, time.monotonic() - started), 6)
    return record


def _collect_software(
    torch: Any, device_index: int, backends: list[Any], backend_errors: list[str]
) -> dict[str, Any]:
    software = _new_software()
    for field, value, source in (
        ("pytorch_version", getattr(torch, "__version__", None), "torch.__version__"),
        ("pytorch_cuda_version", getattr(getattr(torch, "version", None), "cuda", None), "torch.version.cuda"),
    ):
        if value is None or not str(value).strip():
            _mark_missing(software, "software_metadata", field, f"{source} is unavailable")
        else:
            _record(software, "software_metadata", field, str(value), source)

    try:
        z3 = importlib.import_module("z3")
        version_function = getattr(z3, "get_version_string")
        z3_version = str(version_function()).strip()
        if not z3_version:
            raise ValueError("z3.get_version_string() returned an empty version")
    except Exception as import_exc:
        try:
            z3_version = metadata.version("z3-solver").strip()
        except Exception as metadata_exc:
            _mark_missing(
                software,
                "software_metadata",
                "z3_version",
                f"Z3 import/query failed: {import_exc}; package metadata failed: {metadata_exc}",
            )
        else:
            _record(software, "software_metadata", "z3_version", z3_version, "importlib.metadata.version(z3-solver)")
    else:
        _record(software, "software_metadata", "z3_version", z3_version, "z3.get_version_string()")

    for field, function_name in (
        ("cuda_driver_api_version", "get_driver_version"),
        ("cuda_runtime_version", "get_runtime_version"),
    ):
        try:
            value, source = _runtime_version(backends, backend_errors, function_name)
        except Exception as exc:
            _mark_missing(software, "software_metadata", field, str(exc))
        else:
            _record(software, "software_metadata", field, _format_cuda_version(value), source)

    nvcc_executable = shutil.which("nvcc")
    nvcc_command = [nvcc_executable or "nvcc", "--version"]
    if nvcc_executable is None:
        nvcc_record = _command_not_run(nvcc_command, "nvcc was not found on PATH; no subprocess was started")
    else:
        nvcc_record = _run_readonly_command(nvcc_command)
    software["commands"]["nvcc"] = nvcc_record
    if nvcc_record["returncode"] == 0:
        match = re.search(r"\brelease\s+([0-9]+(?:\.[0-9]+)+)", nvcc_record["stdout"], re.IGNORECASE)
        if match:
            _record(software, "software_metadata", "nvcc_version", match.group(1), "nvcc --version")
        else:
            _mark_missing(software, "software_metadata", "nvcc_version", "Could not parse a release version from nvcc --version")
    else:
        _mark_missing(
            software,
            "software_metadata",
            "nvcc_version",
            nvcc_record["error"] or "nvcc --version did not return a version",
        )

    query_fields = [
        "driver_version",
        "memory.total",
        "memory.used",
        "memory.free",
        "clocks.current.graphics",
        "clocks.current.memory",
        "power.draw",
        "power.limit",
        "temperature.gpu",
        "pstate",
    ]
    smi_executable = shutil.which("nvidia-smi")
    smi_command = [
        smi_executable or "nvidia-smi",
        f"--id={device_index}",
        f"--query-gpu={','.join(query_fields)}",
        "--format=csv,noheader,nounits",
    ]
    if smi_executable is None:
        smi_record = _command_not_run(smi_command, "nvidia-smi was not found on PATH; no subprocess was started")
    else:
        smi_record = _run_readonly_command(smi_command)
    smi_record["query_fields"] = query_fields
    smi_record["units"] = {
        "memory.*": "MiB as reported by nvidia-smi nounits",
        "clocks.*": "MHz as reported by nvidia-smi nounits",
        "power.*": "W as reported by nvidia-smi nounits",
        "temperature.gpu": "degrees Celsius as reported by nvidia-smi nounits",
    }
    software["commands"]["nvidia_smi"] = smi_record
    if smi_record["returncode"] == 0:
        first_line = next((line for line in smi_record["stdout"].splitlines() if line.strip()), "")
        driver_version = first_line.split(",", 1)[0].strip() if first_line else ""
        if driver_version and driver_version.upper() not in {"N/A", "[N/A]", "UNKNOWN"}:
            _record(
                software,
                "software_metadata",
                "driver_version",
                driver_version,
                "nvidia-smi --query-gpu=driver_version",
            )
        else:
            _mark_missing(software, "software_metadata", "driver_version", "nvidia-smi output did not contain a driver version")
    else:
        _mark_missing(
            software,
            "software_metadata",
            "driver_version",
            smi_record["error"] or "nvidia-smi did not return a driver version",
        )
    return software


def hardware_spec_to_dict(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return an independent, strict-JSON-compatible dictionary.

    Non-finite floats and arbitrary Python objects are rejected rather than
    stringified.  This also verifies that a probe result contains only JSON
    null/boolean/number/string/array/object values.
    """
    if not isinstance(spec, Mapping):
        raise TypeError("Hardware spec must be a mapping")
    encoded = json.dumps(dict(spec), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    result = json.loads(encoded)
    if not isinstance(result, dict):
        raise TypeError("Hardware spec must serialize to a JSON object")
    return result


def probe_cuda_hardware(device_index: int = 0) -> dict[str, Any]:
    """Query one real CUDA device and return a JSON-only hardware spec.

    Every unavailable attribute remains null and has an explanatory entry in
    device.attribute_metadata.  No model-specific lookup table, inferred SM
    count, assumed shared-memory size, or derived register limit is used.

    This function does not run bandwidth/throughput kernels or any other
    microbenchmark.  Its calibration section explicitly says "not_performed".
    """
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError("device_index must be a non-negative integer")
    started_at = _utc_timestamp()
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise CudaHardwareProbeError(
            "PyTorch could not be imported. A CUDA-enabled PyTorch installation and a working NVIDIA driver/device "
            f"are required for hardware probing: {exc}"
        ) from exc
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        raise CudaHardwareProbeError(f"PyTorch could not determine CUDA availability: {exc}") from exc
    if not cuda_available:
        raise CudaHardwareProbeError(
            "CUDA is not available through PyTorch. Check that this is a CUDA-enabled PyTorch build and that an "
            "NVIDIA driver and the requested GPU are visible; no hardware spec was fabricated."
        )
    try:
        properties = torch.cuda.get_device_properties(device_index)
    except Exception as exc:
        raise CudaHardwareProbeError(f"torch.cuda.get_device_properties({device_index}) failed: {exc}") from exc

    device = _new_device(device_index)
    _collect_torch_attributes(device, properties)
    backends, backend_errors = _load_runtime_backends()
    _collect_runtime_attributes(device, backends, backend_errors, device_index)
    software = _collect_software(torch, device_index, backends, backend_errors)
    spec = {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "timestamp": _utc_timestamp(),
        "device": device,
        "software": software,
        "calibration": {
            "status": "not_performed",
            "memory_bandwidth_bytes_per_second": None,
            "throughput_by_operation_operations_per_second": None,
            "reason": (
                "This probe collects reported device attributes only. It did not run memory-bandwidth or "
                "compute-throughput microbenchmarks, so these values must not be treated as measured calibration."
            ),
        },
        "provenance": {
            "probe": "tilelang.cuda.hardware.probe_cuda_hardware",
            "collection_started_at": started_at,
            "collection_finished_at": None,
            "runtime_backends": [backend.source for backend in backends],
            "runtime_backend_errors": backend_errors,
            "microbenchmarks_performed": False,
            "notes": [
                "Device attributes are hardware/driver-reported capabilities and limits, "
                "not measured bandwidth or throughput.",
                "Null attributes are unavailable from the queried APIs; their reasons are recorded per field.",
                "Read-only subprocesses use shell=False, explicit timeouts, and an allowlisted environment; "
                "environment contents are not collected.",
            ],
        },
    }
    spec["provenance"]["collection_finished_at"] = spec["timestamp"]
    return hardware_spec_to_dict(spec)


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    # All Python integers are finite; converting an arbitrarily large one to
    # float merely for math.isfinite can raise OverflowError.
    return isinstance(value, int) or math.isfinite(value)


def validate_hardware_spec(spec: Mapping[str, Any], require_calibration: bool = False) -> None:
    """Validate required experiment attributes, raising on any error.

    Core schema/attribute errors and calibration errors are exposed separately
    as ``HardwareSpecValidationError.errors`` and ``.calibration_errors``.
    Uncalibrated probe output is valid by default.  With require_calibration,
    status must be "measured", memory bandwidth must be a positive number of
    bytes/second, and throughput must be a non-empty mapping from operation
    names to positive operations/second numbers.
    """
    errors = []
    calibration_errors = []
    try:
        spec = hardware_spec_to_dict(spec)
    except (TypeError, ValueError) as exc:
        raise HardwareSpecValidationError([f"Spec is not strict JSON data: {exc}"]) from exc

    if spec.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    version = spec.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
        errors.append(f"version must be integer {SCHEMA_VERSION}")
    if not isinstance(spec.get("timestamp"), str) or not spec["timestamp"].strip():
        errors.append("timestamp must be a non-empty string")

    device = spec.get("device")
    if not isinstance(device, dict):
        errors.append("device must be an object")
        device = {}
        attribute_metadata = {}
    else:
        attribute_metadata = device.get("attribute_metadata")
        if not isinstance(attribute_metadata, dict):
            errors.append("device.attribute_metadata must be an object")
            attribute_metadata = {}
    if isinstance(device.get("index"), bool) or not isinstance(device.get("index"), int) or device["index"] < 0:
        errors.append("device.index must be a non-negative integer")

    for field in _REQUIRED_DEVICE_FIELDS:
        value = device.get(field)
        if value is None:
            errors.append(f"device.{field} is required and must not be null")
            continue
        if field == "name":
            if not isinstance(value, str) or not value.strip():
                errors.append("device.name must be a non-empty string")
        elif field == "compute_capability":
            if not isinstance(value, str) or re.fullmatch(r"[0-9]+\.[0-9]+", value) is None:
                errors.append("device.compute_capability must have the form '<major>.<minor>'")
        elif not _finite_number(value) or isinstance(value, float) and not value.is_integer():
            errors.append(f"device.{field} must be a finite integer")
        elif value < 0 or (value == 0 and field not in _NONNEGATIVE_DEVICE_FIELDS):
            errors.append(f"device.{field} must be {'non-negative' if field in _NONNEGATIVE_DEVICE_FIELDS else 'positive'}")

    for field in _DEVICE_FIELDS:
        if field not in device:
            errors.append(f"device.{field} must be present (null with a reason when unavailable)")
            continue
        entry = attribute_metadata.get(field)
        if not isinstance(entry, dict):
            entry = {}
        if device[field] is None:
            if not isinstance(entry.get("reason"), str) or not entry["reason"].strip():
                errors.append(f"device.attribute_metadata.{field}.reason must explain the null value")
            continue
        if not isinstance(entry.get("source"), str) or not entry["source"].strip():
            errors.append(f"device.attribute_metadata.{field}.source must identify the observed value's source")
        if field not in _REQUIRED_DEVICE_FIELDS and field not in {"name", "compute_capability"}:
            value = device[field]
            if not _finite_number(value) or isinstance(value, float) and not value.is_integer() or value < 0:
                errors.append(f"device.{field} must be a non-negative finite integer when present")

    software = spec.get("software")
    if not isinstance(software, dict):
        errors.append("software must be an object")
    else:
        software_metadata = software.get("software_metadata")
        if not isinstance(software_metadata, dict):
            errors.append("software.software_metadata must be an object")
            software_metadata = {}
        for field in _SOFTWARE_FIELDS:
            if field not in software:
                errors.append(f"software.{field} must be present (null with a reason when unavailable)")
                continue
            entry = software_metadata.get(field)
            if not isinstance(entry, dict):
                entry = {}
            value = software[field]
            if value is None:
                if not isinstance(entry.get("reason"), str) or not entry["reason"].strip():
                    errors.append(f"software.software_metadata.{field}.reason must explain the null value")
                continue
            if not isinstance(value, str) or not value.strip():
                errors.append(f"software.{field} must be a non-empty string or null")
            if not isinstance(entry.get("source"), str) or not entry["source"].strip():
                errors.append(f"software.software_metadata.{field}.source must identify the version's source")
        if not isinstance(software.get("commands"), dict):
            errors.append("software.commands must be an object")
        else:
            for command_name in ("nvcc", "nvidia_smi"):
                if not isinstance(software["commands"].get(command_name), dict):
                    errors.append(f"software.commands.{command_name} must be an object")

    provenance = spec.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")
    elif not isinstance(provenance.get("microbenchmarks_performed"), bool):
        errors.append("provenance.microbenchmarks_performed must be a boolean")

    if require_calibration:
        calibration = spec.get("calibration")
        if not isinstance(calibration, dict):
            calibration_errors.append("calibration must be an object containing measured microbenchmark results")
        else:
            if calibration.get("status") != "measured":
                calibration_errors.append("calibration.status must be 'measured'")
            bandwidth = calibration.get("memory_bandwidth_bytes_per_second")
            if not _finite_number(bandwidth) or bandwidth <= 0:
                calibration_errors.append("calibration.memory_bandwidth_bytes_per_second must be a positive measured number")
            throughput = calibration.get("throughput_by_operation_operations_per_second")
            if not isinstance(throughput, dict) or not throughput:
                calibration_errors.append(
                    "calibration.throughput_by_operation_operations_per_second must be a non-empty measured object"
                )
            else:
                for operation, value in throughput.items():
                    if not isinstance(operation, str) or not operation.strip():
                        calibration_errors.append("calibration throughput operation names must be non-empty strings")
                    if not _finite_number(value) or value <= 0:
                        calibration_errors.append(f"calibration throughput for {operation!r} must be a positive measured number")

    if errors or calibration_errors:
        raise HardwareSpecValidationError(errors, calibration_errors)


__all__ = [
    "CudaHardwareProbeError",
    "HardwareSpecValidationError",
    "hardware_spec_to_dict",
    "probe_cuda_hardware",
    "validate_hardware_spec",
]
