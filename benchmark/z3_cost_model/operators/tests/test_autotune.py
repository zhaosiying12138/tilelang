"""CPU-only tests for the classic operator experiment drivers.

No Torch, CUDA, native TileLang, compiler, profiler, or GPU is imported/executed.
The production classes are replaced only inside these tests; production defaults
in autotune.py continue to resolve the real GuidedAutoTuner/CudaCostModel.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import types

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.z3_cost_model.operators import analyze_results, autotune, repeat_search  # noqa: E402


_SYNTHETIC_PACKAGE = "_classic_operator_autotune_cpu_test"


def _install_package(name, path):
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        package.__package__ = name
        sys.modules[name] = package


_install_package(_SYNTHETIC_PACKAGE, REPOSITORY_ROOT / "tilelang")
_install_package(f"{_SYNTHETIC_PACKAGE}.autotuner", REPOSITORY_ROOT / "tilelang/autotuner")
_install_package(f"{_SYNTHETIC_PACKAGE}.cuda", REPOSITORY_ROOT / "tilelang/cuda")
_experiment = importlib.import_module(f"{_SYNTHETIC_PACKAGE}.autotuner.experiment")
SearchExperiment = _experiment.SearchExperiment
ExperimentSafetyError = _experiment.ExperimentSafetyError
CudaCostModel = importlib.import_module(f"{_SYNTHETIC_PACKAGE}.cuda.cost_model").CudaCostModel


class FakeSpace:
    def __init__(self, spec):
        self.spec = spec
        domains = [spec.module.DOMAINS[key] for key in spec.module.CONFIG_KEYS]
        self.configs = [dict(zip(spec.module.CONFIG_KEYS, values, strict=True)) for values in itertools.product(*domains)]
        self.legal_count = min(12, len(self.configs))

    def raw_configs(self):
        return copy.deepcopy(self.configs)

    def __len__(self):
        return len(self.configs)


class FakeMeasuredCostModel:
    from_files_calls = []

    def __init__(self, operator, shape):
        self.operator = operator
        self.shape = copy.deepcopy(shape)
        self.predicted = []
        self.fallback_used = ()

    @classmethod
    def from_files(cls, hardware, calibration, operator, shape, require_measured=True, **kwargs):
        assert require_measured is True
        assert not kwargs
        cls.from_files_calls.append(
            {
                "hardware": str(hardware),
                "calibration": str(calibration),
                "operator": operator,
                "shape": copy.deepcopy(shape),
            }
        )
        return cls(operator, shape)

    def predict(self, config):
        values = copy.deepcopy(dict(config))
        self.predicted.append(values)
        score = float(values["block_M"] + values["block_N"])
        return score, {"predicted_ms": score}

    def model_card(self):
        return {
            "schema": "mock-measured-model",
            "operator": self.operator,
            "shape": copy.deepcopy(self.shape),
            "calibration": {"status": "measured"},
            "fallback_used": [],
        }


class FakeKernel:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.calls = []
        self.prim_func = object()
        self.execution_backend = "mock"

    def __call__(self, *inputs):
        self.calls.append(inputs)
        return "actual"

    def get_profiler(self, tensor_supply_type=None):
        assert tensor_supply_type is not None
        return FakeProfiler(self)


class FakeProfiler:
    """Mirror the real Profiler callback order without doing any timing work."""

    def __init__(self, kernel):
        self.kernel = kernel

    def _get_params(self, with_output=False):
        assert with_output is False
        return [object(), object()]

    def manual_assert_close(self, reference_program, input_tensors=None, manual_check_prog=None):
        expected = reference_program(*input_tensors)
        actual = self.kernel(*input_tensors)
        manual_check_prog([actual], [expected])

    def do_bench(self, func=None, n_warmup=0, n_repeat=0, input_tensors=None, **kwargs):
        assert input_tensors is not None
        assert (n_warmup, n_repeat) == (2, 3)
        assert kwargs["backend"] == "event"
        if func is not None:
            func(*input_tensors)
            return 4.5
        self.kernel(*input_tensors)
        return float(self.kernel.config["block_M"] / 64 + self.kernel.config["block_N"] / 64)


class FakeGuidedAutoTuner:
    instances = []
    fail_with = None

    def __init__(self, fn, *, space, model, top_k, arm, budget, seed):
        self.fn = fn
        self.space = space
        self.model = model
        self.top_k = top_k
        self.arm = arm
        self.budget = budget
        self.seed = seed
        self.profile_args = {}
        self.compile_args = {}
        self.last_summary = None
        self.selected_configs = None
        self.instances.append(self)

    def set_profile_args(self, **kwargs):
        self.profile_args.update(kwargs)
        return self

    def set_compile_args(self, **kwargs):
        self.compile_args.update(kwargs)
        return self

    def _select(self):
        raw = self.space.raw_configs()
        legal = raw[: self.space.legal_count]
        if self.arm == "stock-raw":
            return raw, None
        if self.arm == "z3-legal-exhaustive":
            return legal, legal
        if self.arm == "z3-cm-topk":
            return sorted(legal, key=lambda config: self.model.predict(config)[0])[: self.top_k], legal
        if self.arm == "budget-raw":
            return legal[: self.budget], legal
        if self.arm == "budget-random":
            return random.Random(self.seed).sample(legal, min(self.budget, len(legal))), legal
        raise AssertionError(self.arm)

    @staticmethod
    def _summary(arm, raw, legal, selected):
        return {
            "schema_version": 1,
            "status": "ok",
            "arm": arm,
            "R": raw,
            "U": legal,
            "selected": selected,
            "compile_attempts": selected,
            "compile_success": selected,
            "compile_fail": 0,
            "validation": {"attempts": selected, "passed": selected, "failed": 0},
            "profile": {"attempts": selected, "success": selected, "fail": 0},
            "best_config": None,
            "best_latency_ms": None,
            "cache": {},
            "trials": {"ok": selected},
            "errors": [],
        }

    def run(self, *, jsonl_path, metadata, warmup, rep, **kwargs):
        assert (warmup, rep) == (2, 3)
        assert kwargs["use_pipeline"] is False
        assert kwargs["enable_grouped_compile"] is False
        assert kwargs["benchmark_multi_gpu"] is False
        assert kwargs["early_stop"] is False
        self.cache_environment = {
            name: os.environ.get(name)
            for name in (
                "TILELANG_CACHE_DIR",
                "TILELANG_DISABLE_CACHE",
                "TILELANG_AUTO_TUNING_DISABLE_CACHE",
                "CUDA_CACHE_PATH",
                "CUDA_CACHE_DISABLE",
                "TORCH_EXTENSIONS_DIR",
                "TRITON_CACHE_DIR",
                "XDG_CACHE_HOME",
                "TMPDIR",
            )
        }
        raw_count = len(self.space)
        legal_count = None if self.arm == "stock-raw" else self.space.legal_count
        events = [{"event": "run_start", "arm": self.arm, "metadata": metadata}]
        if self.fail_with is not None:
            events.append({"event": "run_end", "status": "error", "error": self.fail_with})
            Path(jsonl_path).write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            self.last_summary = self._summary(self.arm, raw_count, legal_count, 0)
            self.last_summary.update(status="error", compile_attempts=0, compile_success=0, trials={})
            self.last_summary["errors"] = [{"kind": "run", "error": self.fail_with}]
            print(self.fail_with)
            print(self.fail_with, file=sys.stderr)
            raise RuntimeError(self.fail_with)

        selected, legal = self._select()
        self.selected_configs = copy.deepcopy(selected)
        summary = self._summary(self.arm, raw_count, None if legal is None else len(legal), len(selected))
        best_latency = math.inf
        best_config = None
        best_candidate = None
        for index, config in enumerate(selected):
            candidate = self.fn(**copy.deepcopy(config))
            profiler = candidate.get_profiler("auto")
            inputs = self.profile_args["supply_prog"](profiler._get_params(with_output=False))
            profiler.manual_assert_close(
                self.profile_args["ref_prog"],
                input_tensors=inputs,
                manual_check_prog=self.profile_args["manual_check_prog"],
            )
            latency = profiler.do_bench(n_warmup=warmup, n_repeat=rep, input_tensors=inputs, backend="event")
            ref_latency = profiler.do_bench(
                self.profile_args["ref_prog"],
                n_warmup=warmup,
                n_repeat=rep,
                input_tensors=inputs,
                backend="event",
            )
            events.append({"event": "trial", "index": index, "status": "ok", "config": config, "latency_ms": latency})
            if latency < best_latency:
                best_latency = latency
                best_config = copy.deepcopy(config)
                best_candidate = candidate
        summary["best_latency_ms"] = best_latency
        summary["best_config"] = best_config
        events.append({"event": "run_end", "status": "ok", "best_config": best_config, "best_latency_ms": best_latency})
        Path(jsonl_path).write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        self.last_summary = summary
        print(f"mock {self.arm}")
        print(f"mock diagnostics {self.arm}", file=sys.stderr)
        return types.SimpleNamespace(latency=best_latency, config=best_config, ref_latency=ref_latency, kernel=best_candidate)


@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeGuidedAutoTuner.instances = []
    FakeGuidedAutoTuner.fail_with = None
    FakeMeasuredCostModel.from_files_calls = []


def _callbacks(spec, records):
    names = {
        "gemm": ("A", "B"),
        "convolution_nhwc_implicit_gemm": ("data", "kernel"),
        "flash_attention_bshd": ("Q", "K", "V"),
    }[spec.operator]

    def input_factory(*, shape, device, seed):
        records["inputs"].append((shape, device, seed))
        return {name: object() for name in names}

    def reference_factory(inputs, *, shape):
        records["reference"].append((inputs, shape))
        return "expected"

    def validate_factory(actual, expected, *, shape):
        records["validate"].append((actual, expected, shape))
        return {"passed": True}

    def kernel_factory(config, *, shape, target, pass_configs, compile):
        record = {
            "config": copy.deepcopy(config),
            "shape": shape,
            "target": target,
            "pass_configs": copy.deepcopy(pass_configs),
            "compile": compile,
        }
        records["kernels"].append(record)
        return FakeKernel(config) if compile else types.SimpleNamespace(params=[], buffer_map={})

    return {
        "input_factory": input_factory,
        "reference_factory": reference_factory,
        "validate_factory": validate_factory,
        "kernel_factory": kernel_factory,
    }


def _fixture(spec, records):
    callbacks = _callbacks(spec, records)
    fixture = autotune.OperatorFixture(
        spec,
        seed=17,
        device="cpu",
        input_factory=callbacks["input_factory"],
        reference_factory=callbacks["reference_factory"],
        validate_factory=callbacks["validate_factory"],
    )
    return fixture, callbacks


def _run_search(root, monkeypatch, records, *, arm="stock-raw", top_k=10, cache_mode="cold", repetition=0):
    spec = autotune.SUITES["gemm_square"]
    callbacks = _callbacks(spec, records)

    def space_factory(*, shape, model):
        records["constraints"].append((shape, model))
        return FakeSpace(spec)

    monkeypatch.setattr(autotune.gemm, "config_space", space_factory)
    return autotune.run_search(
        spec,
        arm,
        top_k,
        root,
        warmup=2,
        repeat=3,
        seed=17,
        cache_mode=cache_mode,
        repetition=repetition,
        timeout=0,
        device="cpu",
        target="mock-target",
        tuner_cls=FakeGuidedAutoTuner,
        experiment_cls=SearchExperiment,
        cost_model_cls=FakeMeasuredCostModel,
        **callbacks,
    )


def _records():
    return {name: [] for name in ("inputs", "reference", "validate", "kernels", "constraints")}


def test_fixed_suites_shapes_and_preregistered_budgets():
    assert autotune.SUITE_NAMES == ("gemm_square", "gemm_short_k", "conv2d_stride_tail", "flash_attention_causal")
    assert [spec.raw_count for spec in autotune.SUITES.values()] == [288, 288, 144, 54]
    assert autotune.SUITES["gemm_short_k"].shape.to_dict() == {"M": 8192, "N": 8192, "K": 256}
    assert autotune.SUITES["conv2d_stride_tail"].shape.to_dict() == {
        "N": 4,
        "C": 64,
        "H": 33,
        "W": 33,
        "F": 96,
        "K": 3,
        "S": 2,
        "D": 1,
        "P": 1,
        "OH": 17,
        "OW": 17,
        "reduction": 576,
    }
    assert autotune.SUITES["flash_attention_causal"].shape.to_dict() == {"B": 1, "H": 16, "S": 2048, "D": 64, "is_causal": True}
    assert autotune.DEFAULT_REPETITIONS == (0, 1, 2)
    for value in (1, 5, 10, 20):
        assert autotune.resolve_top_k("gemm_square", value) == value
    for suite in autotune.SUITE_NAMES:
        assert autotune.resolve_top_k(suite, None) == 10
    with pytest.raises(ValueError, match="top_k"):
        autotune.resolve_top_k("gemm_short_k", 5)
    with pytest.raises(ValueError, match="top_k"):
        autotune.resolve_top_k("gemm_square", 2)


def test_real_cpu_cost_model_names_shapes_and_measured_calibration():
    for spec in autotune.SUITES.values():
        config = {key: values[0] for key, values in spec.module.DOMAINS.items()}
        model = CudaCostModel.from_files(
            autotune.DEFAULT_HARDWARE_PATH,
            autotune.DEFAULT_CALIBRATION_PATH,
            spec.model_operator,
            autotune.model_shape(spec),
            require_measured=True,
        )
        score, terms = model.predict(config)
        assert math.isfinite(score) and score > 0
        assert terms["predicted_ms"] == score
        assert model.fallback_used == ()
        assert model.model_card()["calibration"]["status"] == "measured"
    conv = autotune.model_shape("conv2d_stride_tail")
    assert (conv["K"], conv["R"], conv["S"], conv["stride_h"], conv["pad_h"]) == (96, 3, 3, 2, 1)
    fa = autotune.model_shape("flash_attention_causal")
    assert fa["causal"] is True and fa["dtype"] == fa["output_dtype"] == "f16"


def test_candidate_factory_forwards_exact_shape_config_and_has_no_closure():
    for spec in autotune.SUITES.values():
        records = _records()
        fixture, callbacks = _fixture(spec, records)
        config = {key: values[-1] for key, values in spec.module.DOMAINS.items()}
        factory = autotune.make_candidate_factory(
            spec,
            fixture=fixture,
            target="mock-target",
            kernel_factory=callbacks["kernel_factory"],
        )
        assert factory.__closure__ is None
        assert tuple(inspect.signature(factory).parameters)[: len(spec.module.CONFIG_KEYS)] == spec.module.CONFIG_KEYS
        result = factory(**copy.deepcopy(config))
        assert isinstance(result, FakeKernel)
        assert records["kernels"][-1] == {
            "config": config,
            "shape": spec.shape,
            "target": "mock-target",
            "pass_configs": None,
            "compile": True,
        }
        elaborate = autotune.make_candidate_factory(spec, fixture=fixture, compile=False, kernel_factory=callbacks["kernel_factory"])
        elaborate(**copy.deepcopy(config))
        assert records["kernels"][-1]["compile"] is False
        assert records["kernels"][-1]["shape"] is spec.shape


def test_fixture_shares_input_objects_and_uses_module_reference_and_validate():
    spec = autotune.SUITES["gemm_square"]
    records = _records()
    fixture, _callbacks = _fixture(spec, records)
    first = fixture.supply_prog([object(), object()])
    second = fixture.supply_prog([object(), object()])
    assert first == second
    assert all(actual is fixture.inputs[name] for actual, name in zip(first, fixture.input_names, strict=True))
    assert fixture.reference_program(*first) == "expected"
    assert records["reference"][-1] == (fixture.inputs, spec.shape)
    fixture.manual_check_prog(["actual"], ["expected"])
    assert records["validate"][-1] == ("actual", "expected", spec.shape)
    with pytest.raises(ValueError, match="shared fixed-seed"):
        fixture.reference_program(object(), object())


def test_five_arms_budgets_real_driver_wiring_and_run_evidence(tmp_path, monkeypatch):
    expected = {
        "stock-raw": 288,
        "z3-legal-exhaustive": 12,
        "z3-cm-topk": 10,
        "budget-raw": 10,
        "budget-random": 10,
    }
    for arm, selected in expected.items():
        records = _records()
        result = _run_search(tmp_path / arm, monkeypatch, records, arm=arm)
        summary = result.summary
        tuner = FakeGuidedAutoTuner.instances[-1]
        assert summary["R"] == 288
        assert summary["U"] == (None if arm == "stock-raw" else 12)
        assert summary["selected"] == summary["compile_attempts"] == selected
        assert summary["validation"]["attempts"] == selected
        assert summary["profile"]["attempts"] == selected
        assert summary["reference_latency_ms"] == 4.5
        assert summary["operator_run"]["suite"] == "gemm_square"
        assert summary["operator_run"]["top_k"] == summary["operator_run"]["budget"] == 10
        assert tuner.budget == tuner.top_k == 10
        assert len(tuner.selected_configs) == selected
        assert tuner.profile_args["max_mismatched_ratio"] == 0.0
        assert tuner.profile_args["cache_input_tensors"] is True
        assert tuner.profile_args["skip_check"] is False
        assert tuner.compile_args == {"out_idx": [-1], "target": "mock-target", "pass_configs": None}
        assert len(records["inputs"]) == 1
        assert len(records["validate"]) == selected
        assert len(records["reference"]) == 2 * selected  # validation and separate reference timing
        assert all(call[1] is autotune.gemm.PRIMARY_SHAPE for call in records["reference"])
        assert all(call["shape"] is autotune.gemm.PRIMARY_SHAPE for call in records["kernels"])
        config = json.loads((result.run_directory / "config.json").read_text(encoding="utf-8"))
        assert config["raw_candidate_count"] == 288
        assert config["model_operator"] == "gemm"
        assert config["constraint_model"]["target"] == "sm_120"
        assert config["constraint_model"]["shared_memory_per_block_limit_bytes"] == 101376
        assert config["cost_model"]["calibration"]["status"] == "measured"
        assert config["early_stop"] is False
        for name in ("before.json", "after.json", "events.jsonl", "summary.json", "environment.json", "stdout.log", "stderr.log"):
            assert (result.run_directory / name).is_file(), name
        assert "mock" in (result.run_directory / "stdout.log").read_text(encoding="utf-8")
        assert "diagnostics" in (result.run_directory / "stderr.log").read_text(encoding="utf-8")


def test_cache_modes_are_run_owned_and_warm_uses_fresh_tuners(tmp_path, monkeypatch):
    names = (
        "TILELANG_CACHE_DIR",
        "TILELANG_DISABLE_CACHE",
        "TILELANG_AUTO_TUNING_DISABLE_CACHE",
        "CUDA_CACHE_PATH",
        "CUDA_CACHE_DISABLE",
        "TORCH_EXTENSIONS_DIR",
        "TRITON_CACHE_DIR",
        "XDG_CACHE_HOME",
        "TMPDIR",
    )
    previous = {name: os.environ.get(name) for name in names}
    for mode in ("cold", "enabled", "no-cache", "warm"):
        records = _records()
        result = _run_search(tmp_path / mode, monkeypatch, records, arm="budget-raw", top_k=1, cache_mode=mode)
        expected_tuners = 2 if mode == "warm" else 1
        assert len(FakeGuidedAutoTuner.instances) == expected_tuners
        if mode == "warm":
            assert FakeGuidedAutoTuner.instances[0] is not FakeGuidedAutoTuner.instances[1]
            assert (result.run_directory / "warmup_summary.json").is_file()
        for tuner in FakeGuidedAutoTuner.instances:
            environment = tuner.cache_environment
            assert Path(environment["TILELANG_CACHE_DIR"]) == result.run_directory / "cache"
            for name in ("CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME", "TMPDIR"):
                assert result.run_directory / "cache" in Path(environment[name]).parents
            disabled = "1" if mode == "no-cache" else "0"
            assert environment["CUDA_CACHE_DISABLE"] == disabled
            assert environment["TILELANG_DISABLE_CACHE"] == disabled
            assert environment["TILELANG_AUTO_TUNING_DISABLE_CACHE"] == disabled
        FakeGuidedAutoTuner.instances = []
    assert {name: os.environ.get(name) for name in names} == previous


def test_nonempty_run_conflict_and_failures_are_preserved(tmp_path, monkeypatch):
    records = _records()
    root = tmp_path / "conflict"
    _run_search(root, monkeypatch, records, arm="budget-raw", top_k=1)
    with pytest.raises(ExperimentSafetyError, match="non-empty"):
        _run_search(root, monkeypatch, _records(), arm="budget-raw", top_k=1)

    message = "intentional guided run failure"
    FakeGuidedAutoTuner.fail_with = message
    with pytest.raises(RuntimeError, match=message):
        _run_search(tmp_path / "failure", monkeypatch, _records(), arm="stock-raw")
    run_dir = tmp_path / "failure/stock-raw__rep-0000"
    assert json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))["error"] == message
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["status"] == "error"
    assert (run_dir / "after.json").is_file()
    assert message in (run_dir / "stdout.log").read_text(encoding="utf-8")
    assert message in (run_dir / "stderr.log").read_text(encoding="utf-8")


def test_scheduler_builds_independent_commands_and_runs_serially(tmp_path):
    calls = []

    def runner(command, *, check, env, cwd):
        assert check is False
        assert cwd == REPOSITORY_ROOT
        assert env["TILELANG_AUTO_TUNING_CPU_COUNTS"] == "1"
        assert env["TILELANG_AUTO_TUNING_MAX_CPU_COUNT"] == "1"
        calls.append(command)
        return types.SimpleNamespace(returncode=0)

    summary = repeat_search.run_jobs(
        suites=("gemm_square",),
        arms=("stock-raw", "z3-cm-topk"),
        repetitions=(0, 1, 2),
        output_dir=tmp_path,
        runner=runner,
    )
    assert summary["status"] == "ok"
    assert summary["completed_jobs"] == len(calls) == 6
    assert [command[command.index("--arm") + 1] for command in calls] == ["stock-raw", "z3-cm-topk"] * 3
    assert [command[command.index("--repetition") + 1] for command in calls] == ["0", "0", "1", "1", "2", "2"]
    for command in calls:
        assert command[1] == str(repeat_search._OPERATORS_ROOT / "autotune.py")
        assert command[command.index("--warmup") + 1] == "10"
        assert command[command.index("--repeat") + 1] == "50"
        assert command[command.index("--cache-mode") + 1] == "cold"
        assert command[command.index("--top-k") + 1] == "10"
    with pytest.raises(ValueError, match="top_k"):
        repeat_search.run_jobs(suites=autotune.SUITE_NAMES, output_dir=tmp_path / "invalid", top_k=1, runner=runner)


def test_scheduler_stops_on_child_failure(tmp_path):
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        return types.SimpleNamespace(returncode=7)

    with pytest.raises(Exception, match="returned non-zero exit status 7"):
        repeat_search.run_jobs(
            suites=("gemm_square",),
            arms=autotune.ARMS,
            repetitions=(0,),
            output_dir=tmp_path,
            runner=runner,
        )
    assert len(calls) == 1
    summary = json.loads((tmp_path / "scheduler-summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "error"
    assert summary["runs"][0]["returncode"] == 7


def _event(name, config=None, **payload):
    import hashlib

    event = {"event": name, **payload}
    if config is not None:
        event["config"] = config
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
        event["config_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return event


def _write_analysis_run(root, arm, *, selected_indexes=(0, 1, 2, 3), missing_trial=False, no_events=False, wall=40.0):
    configs = [{"x": index} for index in range(4)]
    latencies = {0: 4.0, 1: 1.0, 2: 2.0, 3: 3.0}
    run_dir = root / f"{arm}__rep-0000"
    run_dir.mkdir(parents=True)
    legal = arm != "stock-raw"
    selected = [configs[index] for index in selected_indexes]
    events = [_event("run_start", perf_counter=1000.0, arm=arm)]
    for phase, values in (("raw", configs), ("legal", configs if legal else []), ("selected", selected)):
        for index, config in enumerate(values):
            events.append(_event("config_generated", config, phase=phase, index=index))
    events.append(
        _event(
            "config_generation_end",
            status="ok",
            raw=len(configs),
            legal=len(configs) if legal else None,
            selected=len(selected),
        )
    )
    if legal:
        events.extend((_event("solver_start", perf_counter=1001.0), _event("solver_end", perf_counter=1002.0, status="ok")))
        for index, config in enumerate(configs):
            events.append(_event("solver", config, index=index, status="sat", wall_seconds=0.01))
    if arm == "z3-cm-topk":
        scores = {0: 40.0, 1: 20.0, 2: 10.0, 3: None}
        for rank, config in enumerate(configs, start=1):
            score = scores[config["x"]]
            events.append(_event("score_start", config, perf_counter=1003.0 + rank * 0.01))
            events.append(
                _event(
                    "score",
                    config,
                    index=config["x"],
                    rank=rank,
                    score=score,
                    unknown=score is None,
                    error_type="capacity" if score is None else None,
                )
            )
            events.append(
                _event(
                    "score_end",
                    config,
                    status="unknown" if score is None else "ok",
                    wall_seconds=0.001,
                    perf_counter=1003.0 + rank * 0.01 + 0.005,
                )
            )
    clock = 1010.0
    for index, config in enumerate(selected):
        for phase, status in (("compile", "ok"), ("validation", "passed"), ("profile", "ok")):
            events.append(_event(f"{phase}_start", config, index=index, perf_counter=clock))
            clock += 0.1
            payload = {"status": status, "wall_seconds": 0.05}
            if phase == "profile":
                payload["latency_ms"] = latencies[config["x"]]
            events.append(_event(f"{phase}_end", config, index=index, perf_counter=clock, **payload))
            clock += 0.1
    measured = selected[:-1] if missing_trial else selected
    for index, config in enumerate(measured):
        events.append(_event("trial", config, index=index, status="ok", validation="passed", latency_ms=latencies[config["x"]]))
    best = min(measured, key=lambda config: latencies[config["x"]])
    events.append(
        _event(
            "run_end",
            status="ok",
            best_config=best,
            best_latency_ms=latencies[best["x"]],
            perf_counter=1000.0 + wall,
        )
    )
    if not no_events:
        (run_dir / "events.jsonl").write_text("".join(json.dumps(event, allow_nan=False) + "\n" for event in events), encoding="utf-8")
    summary = {
        "status": "ok",
        "arm": arm,
        "operator_run": {"suite": "gemm_square", "top_k": 1, "seed": 17, "cache_track": "cold"},
        "experiment": {"arm": arm, "repetition": 0, "cache_mode": "enabled"},
        "reference_latency_ms": 50.0,
        "best_config": best,
        "best_latency_ms": latencies[best["x"]],
        "has_recorded_candidate_failures": False,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, allow_nan=False) + "\n", encoding="utf-8")
    return run_dir


def test_analysis_recomputes_comparisons_and_paired_correlation_denominator(tmp_path):
    _write_analysis_run(tmp_path, "stock-raw", wall=100.0)
    _write_analysis_run(tmp_path, "z3-legal-exhaustive", wall=80.0)
    _write_analysis_run(tmp_path, "z3-cm-topk", selected_indexes=(2,), wall=40.0)
    _write_analysis_run(tmp_path, "budget-raw", selected_indexes=(0, 1), no_events=True, wall=20.0)
    _write_analysis_run(tmp_path, "budget-random", selected_indexes=(0, 1), missing_trial=True, wall=20.0)
    payload = analyze_results.write_analysis(tmp_path, tmp_path / "analysis")
    rows = {(row["arm"], row["repetition"]): row for row in payload["runs"]}
    stock = rows[("stock-raw", 0)]
    oracle = rows[("z3-legal-exhaustive", 0)]
    guided = rows[("z3-cm-topk", 0)]
    collected = rows[("budget-raw", 0)]
    missing = rows[("budget-random", 0)]
    assert stock["R"] == stock["selected"] == stock["compile_attempts"] == 4
    assert oracle["U"] == oracle["successful_trials"] == 4
    assert oracle["oracle_complete"] is True
    assert guided["performance_evidence"] is True
    assert guided["search_speedup"] == 2.5
    assert guided["compile_reduction"] == guided["profile_reduction"] == 0.75
    assert guided["quality_ratio"] == 2.0
    assert guided["regret_ms"] == 1.0
    assert guided["relative_regret"] == 1.0
    assert guided["model_predictions"] == 4
    assert guided["model_unknown"] == 1
    assert guided["model_wall_seconds"] is not None
    assert collected["performance_evidence"] is False
    assert collected["search_speedup"] is None and collected["quality_ratio"] is None
    assert missing["performance_evidence"] is False
    assert missing["trial_accounting_complete"] is False
    assert missing["search_speedup"] is None and missing["quality_ratio"] is None
    correlation = payload["correlations"][0]
    assert correlation["status"] == "ok"
    assert correlation["prediction_records"] == 4
    assert correlation["finite_prediction_records"] == 3
    assert correlation["oracle_measured_configs"] == 4
    assert correlation["paired_configs"] == correlation["denominator"] == 3
    assert correlation["pearson"] is not None and correlation["spearman"] is not None
    for name in ("analysis.json", "runs.csv", "summary.csv", "correlations.csv"):
        assert (tmp_path / "analysis" / name).is_file()
    aggregate_rows = {row["arm"]: row for row in payload["aggregates"]}
    assert aggregate_rows["budget-raw"]["performance_evidence_repetitions"] == 0
    assert aggregate_rows["budget-raw"]["best_latency_ms"] is None
    assert aggregate_rows["z3-cm-topk"]["performance_evidence_repetitions"] == 1


def test_correlation_refuses_incomplete_oracle(tmp_path):
    _write_analysis_run(tmp_path, "z3-cm-topk", selected_indexes=(2,))
    _write_analysis_run(tmp_path, "z3-legal-exhaustive", missing_trial=True)
    runs = analyze_results.collect_runs(tmp_path)
    correlation = analyze_results.build_correlations(runs)[0]
    assert correlation["status"] == "unavailable"
    assert correlation["denominator"] == 0
    assert correlation["pearson"] is None and correlation["spearman"] is None
    assert "complete measured oracle" in correlation["reason"]


def test_real_sm120_models_and_constraints_cover_every_frozen_suite():
    guided = importlib.import_module(f"{_SYNTHETIC_PACKAGE}.autotuner.guided")
    space_module = importlib.import_module(f"{_SYNTHETIC_PACKAGE}.autotuner.space")
    from functools import partial

    constraint_model, _ = autotune.constraint_model_from_hardware(autotune.DEFAULT_HARDWARE_PATH)
    expected_legal = {
        "gemm_square": 256,
        "gemm_short_k": 256,
        "conv2d_stride_tail": 128,
        "flash_attention_causal": 54,
    }
    for name, spec in autotune.SUITES.items():
        space = space_module.ConfigSpace(
            spec.module.DOMAINS,
            constraint_builder=partial(spec.module.constraint_builder, shape=spec.shape, model=constraint_model),
        )
        model = CudaCostModel.from_files(
            autotune.DEFAULT_HARDWARE_PATH,
            autotune.DEFAULT_CALIBRATION_PATH,
            spec.model_operator,
            autotune.model_shape(spec),
            require_measured=True,
        )
        search = guided.GuidedSearch(space, model, seed=42)
        legal = search.legal_configs()
        assert len(space.raw_configs()) == spec.raw_count
        assert len(legal) == expected_legal[name]
        selected = search.top_k(1)
        assert len(selected) == 1 and selected[0] in legal
        assert len(search.predictions()) == len(legal)
