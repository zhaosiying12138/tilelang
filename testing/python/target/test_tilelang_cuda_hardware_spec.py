"""GPU-free tests for the standalone CUDA hardware spec probe."""

import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HARDWARE = _load_module("_tested_cuda_hardware", _REPOSITORY_ROOT / "tilelang" / "cuda" / "hardware.py")
CLI = _load_module("_tested_cuda_hardware_cli", _REPOSITORY_ROOT / "scripts" / "probe_cuda_hardware.py")


class _FakeRuntime:
    source = "mock.cuda.runtime"

    def __init__(self, attributes=None):
        self.attributes = attributes or {}
        self.queries = []

    def get_attribute(self, enum_name, numeric_value, device_index):
        self.queries.append((enum_name, numeric_value, device_index))
        if enum_name not in self.attributes:
            raise RuntimeError(f"mock runtime does not expose {enum_name}")
        return self.attributes[enum_name]

    def get_runtime_version(self):
        return 12080

    def get_driver_version(self):
        return 12080


def _fake_torch(properties):
    torch = types.ModuleType("torch")
    torch.__version__ = "2.7.0+mock"
    torch.version = types.SimpleNamespace(cuda="12.8", hip=None)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda device_index: properties,
    )
    return torch


def _install_probe_mocks(monkeypatch, properties, runtime):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(properties))
    z3 = types.ModuleType("z3")

    def get_z3_version():
        return "4.15.mock"

    z3.get_version_string = get_z3_version
    monkeypatch.setitem(sys.modules, "z3", z3)
    monkeypatch.setattr(HARDWARE, "_load_runtime_backends", lambda: ([runtime], ["mock optional backend unavailable"]))
    monkeypatch.setattr(HARDWARE.shutil, "which", lambda name: f"/mock/{name}")
    monkeypatch.setenv("MOCK_CUDA_PROBE_SECRET", "must-not-be-recorded")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["timeout"] > 0
        assert kwargs["shell"] is False
        assert "MOCK_CUDA_PROBE_SECRET" not in kwargs["env"]
        if command[0].endswith("nvcc"):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="Cuda compilation tools, release 12.8, V12.8.61\n",
                stderr="",
            )
        assert command[0].endswith("nvidia-smi")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="555.42.01, 24576, 128, 24448, 1500, 5000, 100.5, 300.0, 55, P0\n",
            stderr="",
        )

    monkeypatch.setattr(HARDWARE.subprocess, "run", fake_run)
    return calls


def _complete_properties():
    return types.SimpleNamespace(
        name="Mock GPU with non-stock attributes",
        major=9,
        minor=1,
        total_memory=17_179_869_184,
        multi_processor_count=17,
        warp_size=32,
        max_threads_per_block=1024,
        max_threads_per_multi_processor=1536,
        regs_per_block=65536,
        regs_per_multiprocessor=98304,
        shared_memory_per_block=49152,
        shared_memory_per_multi_processor=123904,
        L2_cache_size=12_345_678,
        clock_rate=1_234_000,
    )


def test_probe_is_strict_json_and_uses_only_mocked_observations(monkeypatch):
    runtime = _FakeRuntime(
        {
            "cudaDevAttrMaxSharedMemoryPerBlockOptin": 111_000,
            "cudaDevAttrMaxPersistingL2CacheSize": 4_567_890,
            "cudaDevAttrMemoryClockRate": 7_654_000,
            "cudaDevAttrGlobalMemoryBusWidth": 4096,
        }
    )
    calls = _install_probe_mocks(monkeypatch, _complete_properties(), runtime)

    spec = HARDWARE.probe_cuda_hardware(device_index=3)
    encoded = json.dumps(spec, allow_nan=False)
    assert json.loads(encoded) == spec
    assert HARDWARE.hardware_spec_to_dict(spec) == spec
    independent_copy = HARDWARE.hardware_spec_to_dict(spec)
    independent_copy["device"]["name"] = "changed"
    assert spec["device"]["name"] != "changed"

    assert all(key in spec for key in ("schema", "version", "timestamp", "device", "software", "provenance"))
    device = spec["device"]
    assert device["index"] == 3
    assert device["multi_processor_count"] == 17
    assert device["compute_capability"] == "9.1"
    assert device["shared_memory_per_multiprocessor_bytes"] == 123904
    assert device["shared_memory_per_block_optin_bytes"] == 111000
    assert device["max_persisting_l2_cache_size_bytes"] == 4567890
    assert device["memory_clock_rate_khz"] == 7654000
    assert device["max_registers_per_thread"] is None
    assert device["attribute_metadata"]["max_registers_per_thread"]["reason"]
    assert device["attribute_metadata"]["shared_memory_per_block_optin_bytes"]["source"].startswith(
        "mock.cuda.runtime"
    )

    assert spec["software"]["driver_version"] == "555.42.01"
    assert spec["software"]["cuda_driver_api_version"] == "12.8"
    assert spec["software"]["cuda_runtime_version"] == "12.8"
    assert spec["software"]["nvcc_version"] == "12.8"
    assert spec["software"]["pytorch_version"] == "2.7.0+mock"
    assert spec["software"]["z3_version"] == "4.15.mock"
    assert len(calls) == 2
    for command_name in ("nvcc", "nvidia_smi"):
        command = spec["software"]["commands"][command_name]
        assert command["returncode"] == 0
        assert command["started_at"] and command["finished_at"]
        assert command["duration_seconds"] >= 0
        assert command["timeout_seconds"] > 0
    smi_query = spec["software"]["commands"]["nvidia_smi"]["query_fields"]
    assert {"driver_version", "memory.total", "clocks.current.graphics", "power.draw", "temperature.gpu", "pstate"} <= set(
        smi_query
    )

    assert spec["calibration"]["status"] == "not_performed"
    assert spec["calibration"]["memory_bandwidth_bytes_per_second"] is None
    assert spec["calibration"]["throughput_by_operation_operations_per_second"] is None
    assert spec["provenance"]["microbenchmarks_performed"] is False
    assert "must-not-be-recorded" not in encoded
    HARDWARE.validate_hardware_spec(spec)


def test_probe_never_guesses_missing_device_attributes(monkeypatch):
    properties = types.SimpleNamespace(name="NVIDIA A100 (name only; not a lookup key)")
    _install_probe_mocks(monkeypatch, properties, _FakeRuntime())

    spec = HARDWARE.probe_cuda_hardware()
    device = spec["device"]
    for field in (
        "compute_capability",
        "total_memory_bytes",
        "multi_processor_count",
        "warp_size",
        "max_registers_per_thread",
        "shared_memory_per_block_bytes",
        "shared_memory_per_block_optin_bytes",
        "shared_memory_per_multiprocessor_bytes",
        "l2_cache_size_bytes",
        "max_persisting_l2_cache_size_bytes",
        "clock_rate_khz",
    ):
        assert device[field] is None, field
        metadata = device["attribute_metadata"][field]
        assert metadata["source"] is None
        assert metadata["reason"], field
    assert device["multi_processor_count"] != 82
    assert device["shared_memory_per_multiprocessor_bytes"] != 99 * 1024


def test_hardware_spec_to_dict_rejects_non_json_values():
    with pytest.raises(ValueError):
        HARDWARE.hardware_spec_to_dict({"not_finite": math.nan})
    with pytest.raises(TypeError):
        HARDWARE.hardware_spec_to_dict({"not_json": object()})
    with pytest.raises(TypeError):
        HARDWARE.hardware_spec_to_dict([])


def _valid_spec():
    device = {field: 1 for field in HARDWARE._DEVICE_FIELDS}
    device.update(
        {
            "index": 0,
            "name": "Mock GPU",
            "compute_capability": "9.0",
            "l2_cache_size_bytes": 0,
            "max_persisting_l2_cache_size_bytes": 0,
            "reserved_shared_memory_per_block_bytes": 0,
        }
    )
    device["attribute_metadata"] = {
        field: {"source": "mock", "reason": None} for field in HARDWARE._DEVICE_FIELDS
    }
    software = {field: "mock-version" for field in HARDWARE._SOFTWARE_FIELDS}
    software["software_metadata"] = {
        field: {"source": "mock", "reason": None} for field in HARDWARE._SOFTWARE_FIELDS
    }
    software["commands"] = {"nvcc": {}, "nvidia_smi": {}}
    return {
        "schema": HARDWARE.SCHEMA,
        "version": HARDWARE.SCHEMA_VERSION,
        "timestamp": "2026-01-01T00:00:00Z",
        "device": device,
        "software": software,
        "calibration": {
            "status": "not_performed",
            "memory_bandwidth_bytes_per_second": None,
            "throughput_by_operation_operations_per_second": None,
            "reason": "not measured",
        },
        "provenance": {"microbenchmarks_performed": False},
    }


def test_validation_separates_device_and_calibration_errors():
    spec = _valid_spec()
    assert HARDWARE.validate_hardware_spec(spec) is None
    with pytest.raises(HARDWARE.HardwareSpecValidationError) as calibration_only:
        HARDWARE.validate_hardware_spec(spec, require_calibration=True)
    assert calibration_only.value.errors == ()
    assert len(calibration_only.value.calibration_errors) == 3
    assert "Calibration errors:" in str(calibration_only.value)

    spec["device"]["multi_processor_count"] = None
    spec["device"]["attribute_metadata"]["multi_processor_count"]["reason"] = None
    spec["device"]["attribute_metadata"]["warp_size"]["source"] = None
    with pytest.raises(HARDWARE.HardwareSpecValidationError) as combined:
        HARDWARE.validate_hardware_spec(spec, require_calibration=True)
    assert any("multi_processor_count is required" in error for error in combined.value.errors)
    assert any("reason must explain" in error for error in combined.value.errors)
    assert any("warp_size.source" in error for error in combined.value.errors)
    assert combined.value.calibration_errors

    calibrated = _valid_spec()
    calibrated["calibration"] = {
        "status": "measured",
        "memory_bandwidth_bytes_per_second": 1_000_000_000_000.0,
        "throughput_by_operation_operations_per_second": {"mock_fp16": 2_000_000_000_000.0},
    }
    calibrated["provenance"]["microbenchmarks_performed"] = True
    assert HARDWARE.validate_hardware_spec(calibrated, require_calibration=True) is None


def test_atomic_json_write_replaces_file_and_cleans_failure(tmp_path, monkeypatch):
    destination = tmp_path / "nested" / "spec.json"
    CLI._atomic_write_json(destination, {"array": [1, True, None], "text": "GPU"})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"array": [1, True, None], "text": "GPU"}
    assert list(tmp_path.rglob("*.tmp")) == []

    previous_contents = destination.read_bytes()

    def fail_replace(source, target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(CLI.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        CLI._atomic_write_json(destination, {"new": "must not partially replace old data"})
    assert destination.read_bytes() == previous_contents
    assert list(tmp_path.rglob("*.tmp")) == []


def test_cli_help_does_not_probe(monkeypatch, capsys):
    def unexpected_probe(device_index=0):
        raise AssertionError("--help must not initialize or query CUDA")

    monkeypatch.setattr(CLI, "probe_cuda_hardware", unexpected_probe)
    with pytest.raises(SystemExit) as exit_info:
        CLI.main(["--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--output" in help_text
    assert "--device" in help_text
    assert "--pretty" in help_text
    assert "bandwidth/throughput" in help_text
    assert "microbenchmark" in help_text


def test_cli_pretty_prints_mocked_spec(tmp_path, monkeypatch, capsys):
    spec = _valid_spec()
    observed_device = []

    def fake_probe(device_index=0):
        observed_device.append(device_index)
        return spec

    monkeypatch.setattr(CLI, "probe_cuda_hardware", fake_probe)
    destination = tmp_path / "pretty.json"
    assert CLI.main(["--output", os.fspath(destination), "--device", "2", "--pretty"]) == 0
    assert observed_device == [2]
    assert json.loads(capsys.readouterr().out) == spec
    assert json.loads(destination.read_text(encoding="utf-8")) == spec


def test_cli_non_cuda_error_is_clear_and_nonzero(tmp_path, monkeypatch, capsys):
    def no_cuda(device_index=0):
        raise CLI.CudaHardwareProbeError("CUDA is not available through PyTorch")

    monkeypatch.setattr(CLI, "probe_cuda_hardware", no_cuda)
    destination = tmp_path / "must-not-exist.json"
    assert CLI.main(["--output", os.fspath(destination)]) != 0
    assert "CUDA is not available through PyTorch" in capsys.readouterr().err
    assert not destination.exists()
