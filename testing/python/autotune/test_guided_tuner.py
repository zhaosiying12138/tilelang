"""Pure-CPU integration tests for GuidedAutoTuner and structured evidence.

The real tuner.py, guided_tuner.py, guided.py, space.py, and report.py are
loaded with small dependency stubs.  No TileLang native library, Torch, CUDA,
compiler, or GPU is imported or executed.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
import types

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TILELANG_DIR = _REPOSITORY_ROOT / "tilelang"
_AUTOTUNER_DIR = _TILELANG_DIR / "autotuner"
_ALIAS = "_tilelang_g4_integration_test"
_MISSING = object()


def _package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module
    return module


def _dependency_stubs():
    stubs = {}

    def module(name, package=False):
        value = types.ModuleType(name)
        if package:
            value.__path__ = []
        stubs[name] = value
        return value

    class TensorSupplyType:
        Auto = "auto"

    class Tensor:
        pass

    class Cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def device_count():
            return 0

        @staticmethod
        def set_device(_device):
            return None

    torch = module("torch")
    torch.Tensor = Tensor
    torch.cuda = Cuda()
    torch.device = type("device", (), {})

    class JITImpl:
        pass

    class JITKernel:
        pass

    class PrimFunc:
        pass

    class Var:
        pass

    class Target:
        pass

    tilelang = module("tilelang", package=True)
    tilelang.__version__ = "g4-test"
    tilelang.JITKernel = JITKernel
    tilelang.TensorSupplyType = TensorSupplyType
    tilelang.compile = lambda *args, **kwargs: None

    tvm = module("tvm", package=True)
    tirx = module("tvm.tirx")
    target = module("tvm.target")
    tirx.PrimFunc = PrimFunc
    tirx.Var = Var
    target.Target = Target
    tvm.tirx = tirx
    tvm.target = target
    tilelang.tvm = tvm

    jit = module("tilelang.jit", package=True)
    jit_kernel = module("tilelang.jit.kernel")
    jit.JITImpl = JITImpl
    jit_kernel.JITKernel = JITKernel

    env = module("tilelang.env")
    env.TILELANG_AUTO_TUNING_CPU_UTILITIES = "1.0"
    env.TILELANG_AUTO_TUNING_CPU_COUNTS = "2"
    env.TILELANG_AUTO_TUNING_MAX_CPU_COUNT = "2"
    env.is_cache_enabled = lambda: os.environ.get("TILELANG_DISABLE_CACHE", "0") != "1"
    env.is_autotune_cache_disabled = lambda: os.environ.get("TILELANG_AUTO_TUNING_DISABLE_CACHE", "0") == "1"
    tilelang.env = env

    class _Tqdm:
        def __init__(self, *args, **kwargs):
            self.n = 0
            self.total = kwargs.get("total", 0)

        def update(self, count):
            self.n += count

        def set_postfix(self, _value):
            return None

        def refresh(self):
            return None

        def close(self):
            return None

        @staticmethod
        def write(_message):
            return None

    tqdm = module("tqdm", package=True)
    tqdm_auto = module("tqdm.auto")
    tqdm_auto.tqdm = _Tqdm
    tqdm.auto = tqdm_auto

    @dataclass(frozen=True)
    class CompileArgs:
        out_idx: object = None
        execution_backend: str = "auto"
        target: object = "auto"
        target_host: object = None
        verbose: bool = False
        pass_configs: dict | None = None

        def compile_program(self, _program):
            raise AssertionError("tests inject a fake compiler")

        def __hash__(self):
            return hash((self.out_idx, self.execution_backend, str(self.target), str(self.target_host), self.verbose, str(self.pass_configs)))

    @dataclass(frozen=True)
    class ProfileArgs:
        warmup: int = 25
        rep: int = 100
        timeout: int = 30
        backend: str = "event"
        supply_type: object = TensorSupplyType.Auto
        ref_prog: object = None
        supply_prog: object = None
        rtol: float = 1e-2
        atol: float = 1e-2
        max_mismatched_ratio: float = 0.01
        skip_check: bool = False
        manual_check_prog: object = None
        cache_input_tensors: bool = False

        def __hash__(self):
            return hash((self.warmup, self.rep, self.timeout, self.backend, self.skip_check, self.cache_input_tensors))

    @dataclass
    class AutotuneResult:
        latency: float | None = None
        config: dict | None = None
        ref_latency: float | None = None
        libcode: str | None = None
        func: object = None
        kernel: object = None

        def save_to_disk(self, _path, _verbose=False):
            return None

        @classmethod
        def load_from_disk(cls, _path, _compile_args):
            return None

    param = module("tilelang.autotuner.param")
    param.CompileArgs = CompileArgs
    param.ProfileArgs = ProfileArgs
    param.AutotuneResult = AutotuneResult

    grouped = module("tilelang.autotuner.grouped_compile")
    grouped.compile_grouped_unit_tvm_ffi = lambda **_kwargs: []
    capture = module("tilelang.autotuner.capture")
    capture.get_autotune_inputs = lambda: None

    language = module("tilelang.utils.language")
    language.get_prim_func_name = lambda _prim, default="fake": default
    device = module("tilelang.utils.device")
    device.get_available_cpu_count = lambda: 2
    engine_param = module("tilelang.engine.param")
    engine_param.KernelParam = type("KernelParam", (), {})
    module("tilelang.utils", package=True)
    module("tilelang.engine", package=True)
    module("tilelang.autotuner", package=True)
    return stubs


_package(_ALIAS, _TILELANG_DIR)
_package(f"{_ALIAS}.autotuner", _AUTOTUNER_DIR)
_space = importlib.import_module(f"{_ALIAS}.autotuner.space")
_guided = importlib.import_module(f"{_ALIAS}.autotuner.guided")
_report = importlib.import_module(f"{_ALIAS}.autotuner.report")
_STUBS = _dependency_stubs()
_STUBS["tilelang.autotuner.report"] = sys.modules[f"{_ALIAS}.autotuner.report"]
_SAVED_MODULES = {name: sys.modules.get(name, _MISSING) for name in _STUBS}
try:
    sys.modules.update(_STUBS)
    _tuner = importlib.import_module(f"{_ALIAS}.autotuner.tuner")
    _guided_tuner = importlib.import_module(f"{_ALIAS}.autotuner.guided_tuner")
finally:
    for _name, _saved in _SAVED_MODULES.items():
        if _saved is _MISSING:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _saved

AutoTuner = _tuner.AutoTuner
CompileArgs = _STUBS["tilelang.autotuner.param"].CompileArgs
ProfileArgs = _STUBS["tilelang.autotuner.param"].ProfileArgs
ConfigSpace = _space.ConfigSpace
PruneResult = _space.PruneResult
Z3Pruner = _space.Z3Pruner
GuidedAutoTuner = _guided_tuner.GuidedAutoTuner
CostModelCapacityError = _guided.CostModelCapacityError
CostModelError = _guided.CostModelError
EventReporter = _report.EventReporter
ObserverError = _report.ObserverError
RunRecorder = _report.RunRecorder


class _Prim:
    def __init__(self, x=0):
        self.x = x
        self.params = []
        self.buffer_map = {}


def _kernel(x=None):
    return _Prim(x)


def _reference():
    return None


class _FakePruner(Z3Pruner):
    """Prune x=0, retain x=1 as unknown, and retain x>=2 as SAT."""

    def prune(self, configs, constraint_fn=None):
        raw = configs.raw_configs() if isinstance(configs, ConfigSpace) else list(configs)
        records = []
        retained = []
        unknown = []
        for index, config in enumerate(raw):
            status = "unsat" if config["x"] == 0 else ("unknown" if config["x"] == 1 else "sat")
            record = {
                "index": index,
                "config": dict(config),
                "config_id": ConfigSpace.config_id(config),
                "status": status,
                "reason": status,
                "wall_seconds": 0.001,
                "stats": {"checks": 1},
                "timeout": False,
                "resource_limited": False,
            }
            records.append(record)
            if status != "unsat":
                retained.append(dict(config))
            if status == "unknown":
                unknown.append(dict(config))
        counts = {status: sum(record["status"] == status for record in records) for status in ("sat", "unsat", "unknown")}
        unknown_ids = [record["config_id"] for record in records if record["status"] == "unknown"]
        audit = {
            "raw": len(raw),
            "unique": len(raw),
            "duplicates": 0,
            "checked": len(raw),
            **counts,
            "timeouts": 0,
            "resource_limited": 0,
            "kept": len(retained),
            "pruned": counts["unsat"],
            "seed": 42,
            "rlimit": 1000,
            "wall_seconds": 0.005,
            "stats": {"checks": len(raw)},
            "unknown_config_ids": unknown_ids,
            "timeout_config_ids": [],
            "resource_limited_config_ids": [],
            "records": records,
        }
        return PruneResult(configs=retained, unknown=unknown, audit=audit)


class _Model:
    def __init__(self):
        self.calls = []

    def predict(self, config):
        self.calls.append(config["x"])
        return -float(config["x"]), {"preference": float(config["x"])}


class _Profiler:
    def __init__(self, kernel, controls):
        self.kernel = kernel
        self.controls = controls

    def _get_params(self, with_output=False):
        return []

    def _get_inputs(self, with_output=False):
        return []

    def assert_allclose(self, *_args, **_kwargs):
        x = self.kernel.config["x"]
        self.controls["validation"].append(x)
        if x in self.controls["validation_fail"]:
            raise RuntimeError(f"validation failed for {x}")

    def do_bench(self, program=None, **_kwargs):
        if program is not None:
            return 50.0
        x = self.kernel.config["x"]
        self.controls["profile"].append(x)
        if self.controls["sleep"]:
            time.sleep(self.controls["sleep"])
        if x in self.controls["profile_fail"]:
            raise RuntimeError(f"profile failed for {x}")
        return 10.0 - float(x)


class _Kernel:
    def __init__(self, config, controls):
        self.config = config
        self.controls = controls
        self.prim_func = _Prim(config["x"])
        self.execution_backend = "torch"

    def get_profiler(self, tensor_supply_type=None):
        return _Profiler(self, self.controls)

    def update_tuner_result(self, latency, config, ref_latency):
        self.latency = latency
        self.config = config
        self.ref_latency = ref_latency
        return self

    def get_kernel_source(self):
        return "fake cpu kernel"


def _controls():
    return {
        "compile": [],
        "validation": [],
        "profile": [],
        "compile_fail": set(),
        "validation_fail": set(),
        "profile_fail": set(),
        "sleep": 0.0,
    }


def _configure(tuner, controls, validate=False):
    def compile_fake(**config):
        x = config["x"]
        controls["compile"].append(x)
        if controls["sleep"]:
            time.sleep(controls["sleep"])
        if x in controls["compile_fail"]:
            raise RuntimeError(f"compile failed for {x}")
        return _Kernel({"x": x}, controls)

    tuner.compile_args = CompileArgs(target="llvm", execution_backend="torch")
    tuner.profile_args = ProfileArgs(skip_check=not validate, ref_prog=_reference if validate else None)
    tuner.jit_compile = compile_fake
    tuner.jit_elaborate = lambda **config: _Prim(config.get("x", 0))
    return tuner


@pytest.fixture(autouse=True)
def _isolate_tuner(monkeypatch, tmp_path):
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    monkeypatch.setenv("TILELANG_AUTO_TUNING_DISABLE_CACHE", "1")
    monkeypatch.setattr(_tuner, "_init_logger_handlers", lambda: None)
    monkeypatch.setattr(AutoTuner, "_get_cache_dir", classmethod(lambda cls: tmp_path / "autotuner-cache"))
    AutoTuner._memory_cache = {}


def _guided(arm, controls, space=None, top_k=2, validate=False):
    model = _Model()
    tuner = GuidedAutoTuner(
        _kernel,
        space or ConfigSpace({"x": list(range(5))}),
        model,
        top_k=top_k,
        arm=arm,
        pruner=_FakePruner(),
    )
    _configure(tuner, controls, validate=validate)
    return tuner, model


def test_stock_api_and_default_exhaustive_behavior_are_unchanged():
    expected_parameters = [
        "self",
        "warmup",
        "rep",
        "timeout",
        "use_pipeline",
        "enable_grouped_compile",
        "group_compile_size",
        "benchmark_devices",
        "benchmark_multi_gpu",
        "early_stop",
        "early_stop_factor",
        "on_trial",
    ]
    assert list(inspect.signature(AutoTuner.run).parameters) == expected_parameters

    controls = _controls()
    tuner = _configure(AutoTuner(_kernel, configs=[{"x": 0}, {"x": 1}, {"x": 2}]), controls)
    legacy_events = []
    result = tuner.run(warmup=0, rep=1, timeout=0, on_trial=legacy_events.append)
    assert sorted(controls["compile"]) == [0, 1, 2]
    assert sorted(controls["profile"]) == [0, 1, 2]
    assert result.config == {"x": 2}
    assert len(legacy_events) == 3
    assert {event["status"] for event in legacy_events} == {"ok"}
    assert not hasattr(tuner, "_event_observer")


def test_default_run_does_not_pass_new_hook_kwargs_to_legacy_overrides():
    class LegacyOverrides(AutoTuner):
        def _prepare_compile_execution(self, config_args, grouped_compile_active, group_compile_size, compile_func, elaborate_func):
            return super()._prepare_compile_execution(
                config_args,
                grouped_compile_active,
                group_compile_size,
                compile_func,
                elaborate_func,
            )

        def _benchmark_target(self, jit_kernel, warmup, rep, early_stop_factor, benchmark_state, benchmark_device=None):
            return super()._benchmark_target(
                jit_kernel,
                warmup,
                rep,
                early_stop_factor,
                benchmark_state,
                benchmark_device,
            )

    controls = _controls()
    tuner = _configure(LegacyOverrides(_kernel, configs=[{"x": 1}]), controls)
    result = tuner.run(warmup=0, rep=1, timeout=0)
    assert result.config == {"x": 1}
    assert controls["compile"] == controls["profile"] == [1]


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        ("stock-raw", [0, 1, 2, 3, 4]),
        ("z3-legal-exhaustive", [1, 2, 3, 4]),
        ("z3-cm-topk", [4, 3]),
        ("budget-raw", [1, 2]),
        ("budget-random", [config["x"] for config in random.Random(42).sample([{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}], 2)]),
    ],
)
def test_five_arms_use_expected_candidate_denominators(arm, expected):
    controls = _controls()
    tuner, model = _guided(arm, controls)
    result = tuner.run(warmup=0, rep=1, timeout=0)
    summary = tuner.last_summary

    assert sorted(controls["compile"]) == sorted(expected)
    assert tuner.configs == [{"x": x} for x in expected]
    assert len(controls["profile"]) == len(expected)
    assert result.config == {"x": min(expected, key=lambda x: 10.0 - x)}
    assert summary["R"] == 5
    assert summary["U"] == (None if arm == "stock-raw" else 4)
    assert summary["unknown"] == (None if arm == "stock-raw" else 1)
    assert summary["pruner_unknown"] == summary["unknown"]
    assert summary["model_unknown"] == (0 if arm == "z3-cm-topk" else None)
    assert summary["unknown_total"] == (1 if arm == "z3-cm-topk" else None)
    assert summary["unknown_sources"] == {
        "pruner": summary["pruner_unknown"],
        "model": summary["model_unknown"],
        "overlap": 0 if arm == "z3-cm-topk" else None,
        "total": summary["unknown_total"],
    }
    assert summary["model"]["unknown"] == summary["model_unknown"]
    assert summary["duplicates"] == 0
    assert summary["selected"] == len(expected)
    assert summary["compile_attempts"] == len(expected)
    assert summary["compile_success"] == len(expected)
    assert summary["compile_fail"] == 0
    assert summary["validation"]["attempts"] == len(expected)
    assert summary["validation"]["not_run"] == len(expected)
    assert summary["profile"]["attempts"] == len(expected)
    assert summary["model"]["predictions"] == (4 if arm == "z3-cm-topk" else None)
    assert len(model.calls) == (4 if arm == "z3-cm-topk" else 0)
    json.dumps(summary, allow_nan=False)


def test_capacity_unknown_reaches_compiler_and_summary_deduplicates_sources():
    controls = _controls()
    tuner, _ = _guided("z3-cm-topk", controls, top_k=4)

    class CapacityModel:
        def __init__(self):
            self.calls = []

        def predict(self, config):
            self.calls.append(config["x"])
            if config["x"] in {1, 2}:
                raise CostModelCapacityError(f"shared/register capacity unknown for x={config['x']}")
            return -float(config["x"]), {"preference": float(config["x"])}

    model = CapacityModel()
    tuner.model = model
    events = []
    result = tuner.run(warmup=0, rep=1, timeout=0, observer=events.append)
    records = tuner.search.predictions()
    unknown_records = [record for record in records if record.get("unknown", False)]

    # x=0 is UNSAT; x=1 is both pruner/model unknown; x=2 is only model
    # unknown.  All four legal candidates fit the topK budget and are compiled.
    assert len(records) == 4
    assert [record.get("unknown", False) for record in records] == [False, False, True, True]
    assert all(record["score"] is None for record in unknown_records)
    assert all(record["error_type"] == "capacity" for record in unknown_records)
    assert tuner.configs == [record["config"] for record in records]
    assert set(controls["compile"]) == {1, 2, 3, 4}
    assert {1, 2} <= set(controls["compile"])
    assert result.config == {"x": 4}
    assert len(model.calls) == 4

    summary = tuner.last_summary
    assert summary["R"] == 5
    assert summary["U"] == summary["model"]["predictions"] == 4
    assert summary["selected"] == summary["compile_attempts"] == 4
    assert summary["compile_success"] == 4
    assert summary["unknown"] == summary["pruner_unknown"] == 1
    assert summary["model_unknown"] == summary["model"]["unknown"] == 2
    assert summary["unknown_total"] == 2  # Not 1 + 2: x=1 occurs in both sources.
    assert summary["unknown_sources"] == {"pruner": 1, "model": 2, "overlap": 1, "total": 2}
    assert summary["model"]["attempts"] == 4
    assert summary["model"]["success"] == 2
    assert summary["model"]["fail"] == 2
    assert summary["status"] == "ok"

    score_events = [event for event in events if event["event"] == "score"]
    assert len(score_events) == 4
    assert [event["unknown"] for event in score_events] == [False, False, True, True]
    assert [event["error_type"] for event in score_events] == [None, None, "capacity", "capacity"]
    score_end = [event for event in events if event["event"] == "score_end"]
    assert [event["status"] for event in score_end].count("unknown") == 2
    generation_end = next(event for event in events if event["event"] == "config_generation_end")
    assert generation_end["unknown_sources"] == summary["unknown_sources"]
    assert max(events.index(event) for event in score_events) < events.index(generation_end)
    json.dumps(events, allow_nan=False)
    json.loads(tuner.summary_json())


def test_structured_events_jsonl_and_timing_are_complete(tmp_path):
    controls = _controls()
    tuner, _ = _guided("z3-cm-topk", controls)
    events = []
    output = tmp_path / "events.jsonl"
    tuner.run(warmup=0, rep=1, timeout=0, observer=events.append, jsonl_path=output)

    names = [event["event"] for event in events]
    assert names[0] == "run_start"
    assert names[-1] == "run_end"
    for required in (
        "config_generated",
        "solver_start",
        "solver",
        "prune",
        "prune_end",
        "score_start",
        "score",
        "score_end",
        "compile_start",
        "compile_end",
        "validation_start",
        "validation_end",
        "profile_start",
        "profile_end",
        "trial",
        "best",
        "cache_identity",
    ):
        assert required in names
    assert names.count("compile_start") == names.count("compile_end") == 2
    assert names.count("validation_start") == names.count("validation_end") == 2
    assert names.count("profile_start") == names.count("profile_end") == 2
    assert all(event["config_id"] for event in events if event["event"] == "compile_end")
    assert all("perf_counter" in event for event in events)

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record.pop("sequence") for record in records] == list(range(len(records)))
    assert records == events
    summary = tuner.last_summary
    assert summary["compile_wall_seconds"] <= summary["end_to_end_wall_seconds"]
    assert summary["solver_wall_seconds"] == 0.005
    assert summary["model_wall_seconds"] >= 0
    assert summary["compile"]["cumulative_seconds"] >= 0
    assert summary["best_config"] == {"x": 4}
    assert summary["best_latency_ms"] == 6.0


def test_parallel_compile_cumulative_and_wall_are_distinct():
    controls = _controls()
    controls["sleep"] = 0.03
    tuner, _ = _guided("stock-raw", controls, space=ConfigSpace({"x": [0, 1, 2, 3]}))
    tuner.run(warmup=0, rep=1, timeout=0)
    compile_summary = tuner.last_summary["compile"]
    assert compile_summary["attempts"] == 4
    assert compile_summary["cumulative_seconds"] >= 0.10
    assert compile_summary["wall_seconds"] < compile_summary["cumulative_seconds"]


def test_validation_and_compile_failures_are_counted_not_silently_dropped():
    controls = _controls()
    controls["compile_fail"] = {0}
    controls["validation_fail"] = {2}
    tuner, _ = _guided("stock-raw", controls, space=ConfigSpace({"x": [0, 1, 2]}), validate=True)
    result = tuner.run(warmup=0, rep=1, timeout=0)
    assert result.config == {"x": 1}
    summary = tuner.last_summary
    assert summary["compile_attempts"] == 3
    assert summary["compile_success"] == 2
    assert summary["compile_fail"] == 1
    assert summary["validation"]["attempts"] == 2
    assert summary["validation"]["passed"] == 1
    assert summary["validation"]["failed"] == 1
    assert summary["profile"]["attempts"] == 1
    assert summary["trials"]["compile_error"] == 1
    assert summary["trials"]["benchmark_error"] == 1
    assert len(summary["errors"]) >= 2


def test_all_compile_failures_still_raise_and_leave_error_summary():
    controls = _controls()
    controls["compile_fail"] = {0}
    tuner, _ = _guided("stock-raw", controls, space=ConfigSpace({"x": [0]}))
    with pytest.raises(RuntimeError, match="No configuration successfully compiled"):
        tuner.run(warmup=0, rep=1, timeout=0)
    assert tuner.last_summary["status"] == "error"
    assert tuner.last_summary["compile_fail"] == 1
    assert tuner.last_summary["best_config"] is None
    assert tuner.last_summary["unavailable"]["best"]


def test_early_stop_conflicts_with_legacy_and_structured_observers():
    stock = _configure(AutoTuner(_kernel, configs=[{"x": 0}]), _controls())
    with pytest.raises(ValueError, match="Trial reporting with early_stop"):
        stock.run(early_stop=True, on_trial=lambda _event: None)

    guided, _ = _guided("stock-raw", _controls(), space=ConfigSpace({"x": [0]}))
    with pytest.raises(ValueError, match="Structured observation with early_stop"):
        guided.run(early_stop=True, observer=lambda _event: None)
    assert guided.last_summary["status"] == "error"


def test_memory_cache_hit_has_identity_and_zero_new_attempts(monkeypatch):
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "0")
    monkeypatch.setenv("TILELANG_AUTO_TUNING_DISABLE_CACHE", "0")
    controls = _controls()
    tuner, _ = _guided("stock-raw", controls, space=ConfigSpace({"x": [1]}))
    tuner.run(warmup=0, rep=1, timeout=0)
    first_identity = tuner.last_summary["cache_identity"]
    assert tuner.last_summary["compile_attempts"] == 1

    events = []
    tuner.run(warmup=0, rep=1, timeout=0, observer=events.append)
    summary = tuner.last_summary
    assert summary["status"] == "cache_hit"
    assert summary["cache"]["hit"] is True
    assert summary["cache"]["source"] == "memory"
    assert summary["cache_identity"] == first_identity
    assert summary["compile_attempts"] == 0
    assert summary["profile"]["attempts"] == 0
    assert summary["best_config"] == {"x": 1}
    assert any(event["event"] == "cache_hit" for event in events)
    assert len(controls["compile"]) == 1


def test_solver_failure_is_explicit_and_does_not_fabricate_legal_counts():
    controls = _controls()
    tuner, _ = _guided("z3-legal-exhaustive", controls, space=ConfigSpace({"x": [0, 1]}))

    class BrokenPruner(_FakePruner):
        def prune(self, configs, constraint_fn=None):
            raise RuntimeError("solver unavailable")

    tuner.pruner = BrokenPruner()
    with pytest.raises(RuntimeError, match="solver unavailable"):
        tuner.run(warmup=0, rep=1, timeout=0)
    summary = tuner.last_summary
    assert summary["status"] == "error"
    assert summary["solver"]["invoked"] is True
    assert summary["solver"]["completed"] is False
    assert summary["solver"]["checks"] is None
    assert summary["solver"]["wall_seconds"] >= 0
    assert summary["U"] is None
    assert summary["compile_attempts"] == 0


def test_cost_model_failure_is_not_a_pruned_candidate_or_fake_prediction():
    controls = _controls()
    tuner, _ = _guided("z3-cm-topk", controls, space=ConfigSpace({"x": [0, 1, 2]}))

    class BrokenModel:
        def predict(self, _config):
            return float("nan"), {}

    tuner.model = BrokenModel()
    with pytest.raises(CostModelError, match="score must be finite"):
        tuner.run(warmup=0, rep=1, timeout=0)
    summary = tuner.last_summary
    assert summary["status"] == "error"
    assert summary["model"]["attempts"] == 1
    assert summary["model"]["predictions"] is None
    assert "did not complete" in summary["model"]["unavailable_reason"]
    assert summary["compile_attempts"] == 0
    assert controls["compile"] == []


def test_observer_failure_is_explicit_after_tuning():
    tuner, _ = _guided("stock-raw", _controls(), space=ConfigSpace({"x": [1]}))

    def broken(_event):
        raise RuntimeError("sink broke")

    with pytest.raises(ObserverError, match="sink broke"):
        tuner.run(warmup=0, rep=1, timeout=0, observer=broken)
    assert tuner.last_summary["observer_errors"]
    assert tuner.last_summary["best_config"] == {"x": 1}


def test_unhooked_legacy_summary_uses_null_and_reason_not_fake_timing():
    recorder = RunRecorder(fully_instrumented=False)
    events = EventReporter(recorder, _report.logging.getLogger("g4-test"))
    events.emit("run_start", arm="stock-raw", raw=1, metadata={})
    events.emit("trial", index=0, config={"x": 1}, status="ok", latency_ms=2.0)
    events.emit("run_end", status="ok", best_config={"x": 1}, best_latency_ms=2.0)
    summary = recorder.summary()
    for phase in ("compile", "validation", "profile"):
        assert summary[phase]["attempts"] is None
        assert summary[phase]["wall_seconds"] is None
        assert "unhooked legacy tuner" in summary[phase]["unavailable_reason"]
    assert summary["end_to_end_wall_seconds"] is not None
    assert summary["best_latency_ms"] == 2.0
    json.dumps(summary, allow_nan=False)
