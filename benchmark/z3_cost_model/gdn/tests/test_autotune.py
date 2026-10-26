"""CPU/mock tests for the GDN AutoTuner adapter and search evidence driver.

No Torch, CUDA, FlashQLA, native TileLang, or GPU pipeline is imported/executed.
The real CPU-safe SearchExperiment and CudaCostModel are loaded through synthetic
packages so their package __init__ modules cannot pull in native/GPU paths.
"""

from __future__ import annotations

import copy
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.z3_cost_model.gdn import autotune  # noqa: E402
from benchmark.z3_cost_model.gdn.runner import (  # noqa: E402
    BackwardResult,
    ForwardResult,
)


_SYNTHETIC_PACKAGE = "_gdn_autotune_cpu_test"


def _install_synthetic_package(name, path):
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        package.__package__ = name
        sys.modules[name] = package


_install_synthetic_package(_SYNTHETIC_PACKAGE, REPOSITORY_ROOT / "tilelang")
_install_synthetic_package(
    f"{_SYNTHETIC_PACKAGE}.autotuner",
    REPOSITORY_ROOT / "tilelang/autotuner",
)
_install_synthetic_package(
    f"{_SYNTHETIC_PACKAGE}.cuda",
    REPOSITORY_ROOT / "tilelang/cuda",
)
SearchExperiment = importlib.import_module(f"{_SYNTHETIC_PACKAGE}.autotuner.experiment").SearchExperiment
CudaCostModel = importlib.import_module(f"{_SYNTHETIC_PACKAGE}.cuda.cost_model").CudaCostModel


def _mock_public_fwd():
    pass


def _mock_public_bwd():
    pass


class FakeBatch:
    def __init__(self, seed, options):
        self.seed = seed
        self.options = copy.deepcopy(options)
        self.state_v_first = options["state_v_first"]
        self.cu_seqlens = None


class FakeRunner:
    instances = []
    events = None

    def __init__(self, config, *, forward_block_DV=None):
        self.config = config
        self.forward_block_DV = forward_block_DV
        self.forward_calls = []
        self.backward_calls = []
        self.api = types.SimpleNamespace(
            chunk_gated_delta_rule_fwd=_mock_public_fwd,
            chunk_gated_delta_rule_bwd=_mock_public_bwd,
        )
        self.serial = len(self.instances)
        self.name = f"runner-{self.serial}"
        self.forward_result = ForwardResult(
            o=f"{self.name}-o",
            final_state=f"{self.name}-final_state",
            g=f"{self.name}-g",
            A=f"{self.name}-A",
            h=f"{self.name}-h",
            cp_cache=f"{self.name}-cp_cache",
        )
        self.backward_result = BackwardResult(*(f"{self.name}-{name}" for name in autotune.BWD_OUTPUTS))
        self.instances.append(self)

    def forward(self, batch, *, output_h=False):
        self.forward_calls.append((batch, output_h))
        if self.events is not None:
            self.events.append(("forward", self.name))
        return self.forward_result

    def backward(self, batch, forward=None):
        self.backward_calls.append((batch, forward))
        if self.events is not None:
            self.events.append(("backward", self.name))
        return self.backward_result


class FakeProfiler:
    """Same callback ordering as Profiler.manual_assert_close/do_bench."""

    def __init__(self, *, params, result_idx, supply_type, adapter):
        self.params = params
        self.result_idx = result_idx
        self.supply_type = supply_type
        self.adapter = adapter
        self.manual_calls = []
        self.bench_calls = []

    def _get_params(self, with_output=False):
        del with_output
        return list(self.params)

    def manual_assert_close(
        self,
        reference_program,
        input_tensors=None,
        manual_check_prog=None,
    ):
        assert input_tensors == []
        reference_outputs = reference_program(*input_tensors)
        actual_outputs = self.adapter(*input_tensors)
        self.manual_calls.append((actual_outputs, reference_outputs))
        manual_check_prog(actual_outputs, reference_outputs)

    def do_bench(
        self,
        func=None,
        warmup=25,
        rep=100,
        n_warmup=0,
        n_repeat=0,
        input_tensors=None,
        **kwargs,
    ):
        del warmup, rep, kwargs
        assert input_tensors == []
        target = self.adapter if func is None else func
        self.bench_calls.append((target, n_warmup, n_repeat))
        target(*input_tensors)
        return 1.25 if func is None else 4.5


class FakeSpace:
    def __init__(self, direction):
        self.direction = autotune.Direction(direction)
        if self.direction is autotune.Direction.FWD:
            self.configs = [
                {
                    "direction": "fwd",
                    "block_DV": block,
                    "prepare_h_num_stages": 2,
                    "group_reduce_block_size": 16,
                }
                for block in (16, 32, 64, 128)
            ]
        else:
            self.configs = [
                {
                    "direction": "bwd",
                    "block_DV": 128,
                    "prepare_h_num_stages": stages,
                    "group_reduce_block_size": group,
                }
                for stages in (1, 2, 3)
                for group in (8, 16, 32)
            ]

    def raw_configs(self):
        return copy.deepcopy(self.configs)

    def __len__(self):
        return len(self.configs)


class FakeMeasuredCostModel:
    from_files_calls = []

    def __init__(self, operator, shape):
        self.operator = operator
        self.shape = copy.deepcopy(shape)
        self.predicted_configs = []
        self.fallback_used = ()

    @classmethod
    def from_files(
        cls,
        hardware,
        calibration,
        operator,
        shape,
        require_measured=True,
        **kwargs,
    ):
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
        self.predicted_configs.append(values)
        if self.operator == "gdn_fwd":
            score = float(values["block_DV"])
        else:
            score = float(100 * values["prepare_h_num_stages"] + values["group_reduce_block_size"])
        return score, {"predicted_ms": score}

    def model_card(self):
        return {
            "schema": "mock-measured-model-card",
            "operator": self.operator,
            "shape": copy.deepcopy(self.shape),
            "calibration": {"status": "measured"},
            "fallback_used": [],
        }


class FakeGuidedAutoTuner:
    """Exercise the adapter while mocking only search/tuner machinery."""

    instances = []
    fail_with = None

    def __init__(
        self,
        fn,
        *,
        space,
        model,
        top_k,
        arm,
        budget,
        seed,
    ):
        assert budget == top_k
        self.fn = fn
        self.space = space
        self.model = model
        self.top_k = top_k
        self.arm = arm
        self.seed = seed
        self.profile_args = {}
        self.last_summary = None
        self.selected_configs = None
        self.instances.append(self)

    def set_profile_args(self, **kwargs):
        self.profile_args.update(kwargs)
        return self

    def _select(self):
        raw = self.space.raw_configs()
        legal_count = 3 if self.space.direction is autotune.Direction.FWD else 6
        legal = raw[:legal_count]
        if self.arm == "stock-raw":
            return raw, None
        if self.arm == "z3-legal-exhaustive":
            return legal, legal
        if self.arm == "z3-cm-topk":
            ranked = sorted(
                legal,
                key=lambda config: self.model.predict(config)[0],
            )
            return ranked[: self.top_k], legal
        if self.arm == "budget-raw":
            return legal[: self.top_k], legal
        if self.arm == "budget-random":
            count = min(self.top_k, len(legal))
            return random.Random(self.seed).sample(legal, count), legal
        raise AssertionError(self.arm)

    @staticmethod
    def _write_events(path, events):
        with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
            for event in events:
                stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()

    def _base_summary(self, raw_count, legal_count, selected_count, status="ok"):
        return {
            "schema_version": 1,
            "status": status,
            "arm": self.arm,
            "R": raw_count,
            "U": legal_count,
            "selected": selected_count,
            "candidate_count": selected_count,
            "compile_attempts": 0,
            "compile_success": 0,
            "compile_fail": 0,
            "validation": {
                "attempts": 0,
                "passed": 0,
                "failed": 0,
                "not_run": 0,
            },
            "profile": {"attempts": 0, "success": 0, "fail": 0},
            "best_config": None,
            "best_latency_ms": None,
            "cache": {},
            "trials": {},
            "errors": [],
        }

    def run(self, *, jsonl_path, metadata, warmup, rep, **kwargs):
        del kwargs
        self.cache_environment = {
            name: os.environ.get(name)
            for name in (
                "CUDA_CACHE_PATH",
                "CUDA_CACHE_DISABLE",
                "TORCH_EXTENSIONS_DIR",
                "XDG_CACHE_HOME",
                "TMPDIR",
            )
        }
        raw = self.space.raw_configs()
        legal_count = 3 if self.space.direction is autotune.Direction.FWD else 6
        events = [
            {
                "event": "run_start",
                "arm": self.arm,
                "metadata": metadata,
            }
        ]
        if self.fail_with is not None:
            events.append({"event": "run_end", "status": "error", "error": self.fail_with})
            self._write_events(jsonl_path, events)
            self.last_summary = self._base_summary(len(raw), legal_count, 0, status="error")
            self.last_summary["errors"] = [{"kind": "run", "error": self.fail_with}]
            print(self.fail_with)
            print(self.fail_with, file=sys.stderr)
            raise RuntimeError(self.fail_with)

        selected, legal = self._select()
        self.selected_configs = copy.deepcopy(selected)
        events.append(
            {
                "event": "config_generation_end",
                "raw": len(raw),
                "legal": None if legal is None else len(legal),
                "selected": len(selected),
            }
        )
        summary = self._base_summary(len(raw), None if legal is None else len(legal), len(selected))
        best_candidate = None
        best_config = None
        ref_latency = None
        for index, config in enumerate(selected):
            candidate = self.fn(**copy.deepcopy(config))
            summary["compile_attempts"] += 1
            summary["compile_success"] += 1
            events.append({"event": "compile_end", "index": index, "status": "ok", "config": config})

            profiler = candidate.get_profiler("auto")
            inputs = self.profile_args["supply_prog"](profiler._get_params(with_output=False))
            assert inputs == []
            profiler.manual_assert_close(
                self.profile_args["ref_prog"],
                input_tensors=inputs,
                manual_check_prog=self.profile_args["manual_check_prog"],
            )
            summary["validation"]["attempts"] += 1
            summary["validation"]["passed"] += 1
            events.append(
                {
                    "event": "validation_end",
                    "index": index,
                    "status": "passed",
                    "config": config,
                }
            )

            latency = profiler.do_bench(
                n_warmup=warmup,
                n_repeat=rep,
                input_tensors=inputs,
                backend="event",
            )
            ref_latency = profiler.do_bench(
                self.profile_args["ref_prog"],
                n_warmup=warmup,
                n_repeat=rep,
                input_tensors=inputs,
                backend="event",
            )
            summary["profile"]["attempts"] += 1
            summary["profile"]["success"] += 1
            events.append(
                {
                    "event": "profile_end",
                    "index": index,
                    "status": "ok",
                    "latency_ms": latency,
                    "config": config,
                }
            )
            candidate.update_tuner_result(latency, config, ref_latency)
            if best_candidate is None:
                best_candidate = candidate
                best_config = copy.deepcopy(config)
                summary["best_config"] = copy.deepcopy(config)
                summary["best_latency_ms"] = latency

        summary["trials"] = {"ok": len(selected)}
        events.append(
            {
                "event": "run_end",
                "status": "ok",
                "best_config": best_config,
                "best_latency_ms": summary["best_latency_ms"],
            }
        )
        self._write_events(jsonl_path, events)
        self.last_summary = summary
        print(f"mock run {self.space.direction.value}/{self.arm}")
        print(f"mock diagnostics {self.arm}", file=sys.stderr)
        return types.SimpleNamespace(
            latency=summary["best_latency_ms"],
            config=best_config,
            ref_latency=ref_latency,
            kernel=best_candidate,
        )


class GDNAutotuneCPUTest(unittest.TestCase):
    def setUp(self):
        FakeRunner.instances = []
        FakeRunner.events = []
        FakeGuidedAutoTuner.instances = []
        FakeGuidedAutoTuner.fail_with = None
        FakeMeasuredCostModel.from_files_calls = []
        self.events = FakeRunner.events
        self.input_calls = []
        self.reference_forward_calls = []
        self.reference_backward_calls = []
        self.checks = []

        def input_factory(*, seed, **options):
            self.input_calls.append((seed, copy.deepcopy(options)))
            return FakeBatch(seed, options)

        def reference_forward(batch):
            self.reference_forward_calls.append(batch)
            self.events.append(("reference_forward", batch.seed))
            return "g-ref", "o-ref", "A-ref", "h-ref", "final_state-ref"

        def reference_backward(batch, expected_forward):
            self.reference_backward_calls.append((batch, expected_forward))
            self.events.append(("reference_backward", batch.seed))
            return tuple(f"{name}-ref" for name in autotune.BWD_OUTPUTS)

        def assert_forward(batch, actual, expected, *, output_h=False):
            self.events.append(("manual_forward", batch.seed))
            self.checks.append(
                {
                    "direction": "fwd",
                    "actual": (actual.o, actual.final_state),
                    "reference": (expected[1], expected[4]),
                    "output_h": output_h,
                }
            )

        def assert_backward(batch, actual, expected):
            self.events.append(("manual_backward", batch.seed))
            self.checks.append(
                {
                    "direction": "bwd",
                    "actual": actual.as_tuple(),
                    "reference": tuple(expected),
                }
            )

        self.input_factory = input_factory
        for name, value in (
            ("gdn_reference_forward", reference_forward),
            ("gdn_reference_backward", reference_backward),
            ("gdn_assert_forward", assert_forward),
            ("gdn_assert_backward", assert_backward),
        ):
            patcher = patch.object(autotune, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _pipeline(self, direction, seed=42):
        return autotune.GDNPipelineAdapter(
            direction,
            seed=seed,
            input_factory=self.input_factory,
        )

    def _candidate(self, direction, pipeline=None, **config):
        return autotune.GDNCandidateKernel(
            direction,
            pipeline=self._pipeline(direction) if pipeline is None else pipeline,
            runner_factory=FakeRunner,
            profiler_factory=FakeProfiler,
            **config,
        )

    def _run_search(self, direction, arm, root, **kwargs):
        options = {
            "top_k": 2 if direction == "fwd" else 3,
            "warmup": 2,
            "repeat": 3,
            "seed": 17,
            "tuner_cls": FakeGuidedAutoTuner,
            "experiment_cls": SearchExperiment,
            "cost_model_cls": FakeMeasuredCostModel,
            "input_factory": self.input_factory,
            "runner_factory": FakeRunner,
            "profiler_factory": FakeProfiler,
        }
        options.update(kwargs)
        with patch.object(
            autotune,
            "make_config_space",
            side_effect=lambda current, _hardware: FakeSpace(current),
        ):
            return autotune.run_gdn_search(
                direction,
                arm,
                options.pop("top_k"),
                root,
                options.pop("warmup"),
                options.pop("repeat"),
                options.pop("seed"),
                **options,
            )

    def test_forward_real_profiler_contract_and_callback_order(self):
        candidate = self._candidate("fwd", block_DV=32)
        runner = candidate.runner
        self.assertEqual(len(runner.forward_calls), 1)
        self.assertIsNone(runner.forward_block_DV)

        factory = autotune.make_candidate_factory(
            "fwd",
            pipeline=candidate.pipeline,
            runner_factory=FakeRunner,
            profiler_factory=FakeProfiler,
        )
        self.assertIsNone(factory.__closure__)
        self.assertEqual(
            tuple(inspect_signature(factory).parameters)[:4],
            (
                "direction",
                "block_DV",
                "prepare_h_num_stages",
                "group_reduce_block_size",
            ),
        )

        profiler = candidate.get_profiler("auto")
        self.assertIsInstance(profiler, FakeProfiler)
        self.assertEqual(profiler.params, [])
        self.assertEqual(profiler.result_idx, [])
        self.assertIs(profiler.adapter, candidate.adapter)
        self.assertEqual(autotune._empty_input_supply([]), [])
        with self.assertRaisesRegex(RuntimeError, "empty params"):
            autotune._empty_input_supply([object()])

        profiler.manual_assert_close(
            candidate.ref_prog,
            input_tensors=[],
            manual_check_prog=candidate.manual_check_prog,
        )
        self.assertEqual(
            self.checks[-1],
            {
                "direction": "fwd",
                "actual": (f"{runner.name}-o", f"{runner.name}-final_state"),
                "reference": ("o-ref", "final_state-ref"),
                "output_h": False,
            },
        )
        names = [event[0] for event in self.events]
        self.assertLess(names.index("reference_forward"), names.index("manual_forward"))
        self.assertEqual(names[-3:], ["reference_forward", "forward", "manual_forward"])

        latency = profiler.do_bench(n_warmup=2, n_repeat=3, input_tensors=[])
        ref_latency = profiler.do_bench(
            candidate.ref_prog,
            n_warmup=2,
            n_repeat=3,
            input_tensors=[],
        )
        self.assertEqual((latency, ref_latency), (1.25, 4.5))
        self.assertEqual(len(runner.forward_calls), 3)
        self.assertEqual(len(self.reference_forward_calls), 2)

    def test_backward_artifacts_output_order_and_reference_scope(self):
        candidate = self._candidate(
            "bwd",
            prepare_h_num_stages=1,
            group_reduce_block_size=32,
        )
        runner = candidate.runner
        self.assertEqual(runner.forward_block_DV, 64)
        self.assertEqual(len(runner.forward_calls), 1)
        self.assertEqual(len(runner.backward_calls), 1)
        self.assertIs(candidate.forward_artifacts, runner.forward_result)

        profiler = candidate.get_profiler("auto")
        profiler.manual_assert_close(
            candidate.ref_prog,
            input_tensors=[],
            manual_check_prog=candidate.manual_check_prog,
        )
        expected_actual = tuple(f"{runner.name}-{name}" for name in autotune.BWD_OUTPUTS)
        expected_reference = tuple(f"{name}-ref" for name in autotune.BWD_OUTPUTS)
        self.assertEqual(self.checks[-1]["actual"], expected_actual)
        self.assertEqual(self.checks[-1]["reference"], expected_reference)
        self.assertEqual(len(runner.forward_calls), 1)
        self.assertEqual(len(runner.backward_calls), 2)
        self.assertIs(runner.backward_calls[-1][1], candidate.forward_artifacts)
        self.assertEqual(len(self.reference_forward_calls), 1)

        profiler.do_bench(n_warmup=1, n_repeat=2, input_tensors=[])
        profiler.do_bench(
            candidate.ref_prog,
            n_warmup=1,
            n_repeat=2,
            input_tensors=[],
        )
        self.assertEqual(len(runner.forward_calls), 1)
        self.assertEqual(len(runner.backward_calls), 3)
        self.assertEqual(len(self.reference_forward_calls), 1)
        self.assertEqual(len(self.reference_backward_calls), 2)

    def test_cost_model_config_adapter_and_json_card(self):
        class Delegate:
            fallback_used = ()

            def __init__(self):
                self.configs = []

            def predict(self, config):
                self.configs.append(copy.deepcopy(config))
                return 1.0, {"predicted_ms": 1.0}

            def model_card(self):
                return {"schema": "delegate"}

        fwd_delegate = Delegate()
        fwd = autotune.GDNCandidateCostModel(fwd_delegate, "fwd")
        fwd.predict(
            {
                "direction": "fwd",
                "block_DV": 64,
                "prepare_h_num_stages": 2,
                "group_reduce_block_size": 16,
            }
        )
        self.assertEqual(fwd_delegate.configs, [{"block_DV": 64}])

        bwd_delegate = Delegate()
        bwd = autotune.GDNCandidateCostModel(bwd_delegate, "bwd")
        bwd_config = {
            "direction": "bwd",
            "block_DV": 128,
            "prepare_h_num_stages": 2,
            "group_reduce_block_size": 16,
        }
        bwd.predict(bwd_config)
        self.assertEqual(
            bwd_delegate.configs,
            [{key: value for key, value in bwd_config.items() if key != "direction"}],
        )
        json.dumps(fwd.model_card(), allow_nan=False)
        with self.assertRaisesRegex(ValueError, "cannot score"):
            fwd.predict(bwd_config)

    def test_real_cpu_cost_model_uses_measured_manifest(self):
        for direction, config in (
            (
                "fwd",
                {
                    "direction": "fwd",
                    "block_DV": 64,
                    "prepare_h_num_stages": 2,
                    "group_reduce_block_size": 16,
                },
            ),
            (
                "bwd",
                {
                    "direction": "bwd",
                    "block_DV": 128,
                    "prepare_h_num_stages": 2,
                    "group_reduce_block_size": 16,
                },
            ),
        ):
            with self.subTest(direction=direction):
                shape = autotune.model_shape(direction)
                delegate = CudaCostModel.from_files(
                    autotune.DEFAULT_HARDWARE_PATH,
                    autotune.DEFAULT_CALIBRATION_PATH,
                    f"gdn_{direction}",
                    shape,
                    require_measured=True,
                )
                model = autotune.GDNCandidateCostModel(delegate, direction)
                score, terms = model.predict(config)
                self.assertTrue(math.isfinite(score) and score > 0)
                self.assertEqual(score, terms["predicted_ms"])
                self.assertEqual(delegate.fallback_used, ())
                card = model.model_card()
                self.assertEqual(card["calibration"]["status"], "measured")
                json.dumps(card, allow_nan=False)

    def test_five_arm_denominators_and_strict_run_json(self):
        expected = {
            "fwd": {"R": 4, "U": 3, "capped": 2},
            "bwd": {"R": 9, "U": 6, "capped": 3},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for direction in ("fwd", "bwd"):
                for arm in autotune.ARMS:
                    with self.subTest(direction=direction, arm=arm):
                        result = self._run_search(
                            direction,
                            arm,
                            root / direction / arm,
                        )
                        counts = expected[direction]
                        selected = counts["R"] if arm == "stock-raw" else counts["U"] if arm == "z3-legal-exhaustive" else counts["capped"]
                        summary = result.summary
                        self.assertEqual(summary["R"], counts["R"])
                        self.assertEqual(
                            summary["U"],
                            None if arm == "stock-raw" else counts["U"],
                        )
                        self.assertEqual(summary["selected"], selected)
                        self.assertEqual(summary["compile_attempts"], selected)
                        self.assertEqual(summary["compile_success"], selected)
                        self.assertEqual(summary["validation"]["attempts"], selected)
                        self.assertEqual(summary["validation"]["passed"], selected)
                        self.assertEqual(summary["profile"]["attempts"], selected)
                        self.assertEqual(summary["profile"]["success"], selected)
                        self.assertEqual(summary["reference_latency_ms"], 4.5)
                        self.assertFalse(summary["has_recorded_candidate_failures"])
                        self.assertEqual(summary["gdn"]["cache_track"], "cold")

                        run_dir = result.run_directory
                        for name in (
                            "events.jsonl",
                            "summary.json",
                            "config.json",
                            "environment.json",
                            "stdout.log",
                            "stderr.log",
                            "before.json",
                            "after.json",
                        ):
                            self.assertTrue((run_dir / name).is_file(), name)
                        events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
                        self.assertEqual(events[0]["event"], "run_start")
                        self.assertEqual(events[-1]["event"], "run_end")
                        self.assertEqual(
                            json.loads((run_dir / "summary.json").read_text()),
                            summary,
                        )
                        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
                        self.assertEqual(config["raw_candidate_count"], counts["R"])
                        self.assertTrue(config["require_measured_calibration"])
                        self.assertEqual(
                            config["cost_model"]["calibration"]["status"],
                            "measured",
                        )
                        self.assertEqual(config["shape"]["T"], 4096)
                        environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
                        self.assertEqual(environment["gpu_query"], "not performed by metadata collection")
                        self.assertIn("mock run", (run_dir / "stdout.log").read_text())
                        self.assertIn("mock diagnostics", (run_dir / "stderr.log").read_text())

    def test_warm_track_uses_fresh_tuner_and_preserves_prewarm_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            names = (
                "CUDA_CACHE_PATH",
                "CUDA_CACHE_DISABLE",
                "TORCH_EXTENSIONS_DIR",
                "XDG_CACHE_HOME",
                "TMPDIR",
            )
            previous = {name: os.environ.get(name) for name in names}
            result = self._run_search(
                "fwd",
                "z3-cm-topk",
                Path(directory),
                cache_mode="warm",
                top_k=1,
            )
            self.assertEqual({name: os.environ.get(name) for name in names}, previous)
            self.assertEqual(len(FakeGuidedAutoTuner.instances), 2)
            self.assertIsNot(
                FakeGuidedAutoTuner.instances[0],
                FakeGuidedAutoTuner.instances[1],
            )
            for tuner in FakeGuidedAutoTuner.instances:
                self.assertEqual(
                    tuner.cache_environment["CUDA_CACHE_PATH"],
                    str(result.run_directory / "cache/nvidia"),
                )
                self.assertEqual(
                    tuner.cache_environment["TMPDIR"],
                    str(result.run_directory / "cache/tmp"),
                )
                self.assertEqual(tuner.cache_environment["CUDA_CACHE_DISABLE"], "0")
            self.assertEqual(result.summary["experiment"]["cache_mode"], "warm")
            self.assertTrue((result.run_directory / "warmup.jsonl").is_file())
            self.assertTrue((result.run_directory / "warmup_summary.json").is_file())

    def test_initialization_and_validation_errors_propagate(self):
        shape = autotune.model_shape("fwd", {"num_tokens": 65})
        self.assertEqual(shape["T"], 65)
        self.assertEqual(shape["H"], autotune.DEFAULT_INPUT_KWARGS["num_v_heads"])

        def broken_runner(*args, **kwargs):
            raise RuntimeError("runner initialization failed")

        with self.assertRaisesRegex(RuntimeError, "runner initialization failed"):
            autotune.GDNCandidateKernel(
                "fwd",
                pipeline=self._pipeline("fwd"),
                runner_factory=broken_runner,
                profiler_factory=FakeProfiler,
            )

        candidate = self._candidate("fwd")
        profiler = candidate.get_profiler("auto")
        with (
            patch.object(
                autotune,
                "gdn_assert_forward",
                side_effect=AssertionError("relative L2 failure"),
            ),
            self.assertRaisesRegex(AssertionError, "relative L2 failure"),
        ):
            profiler.manual_assert_close(
                candidate.ref_prog,
                input_tensors=[],
                manual_check_prog=candidate.manual_check_prog,
            )

    def test_failures_are_logged_and_reraised(self):
        message = "intentional candidate factory failure"
        FakeGuidedAutoTuner.fail_with = message
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, message):
                self._run_search("fwd", "stock-raw", root)
            run_dir = root / "stock-raw__rep-0000"
            failure = json.loads((run_dir / "failure.json").read_text())
            self.assertEqual(failure["error_type"], "RuntimeError")
            self.assertIn(message, failure["error"])
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["status"], "error")
            self.assertTrue((run_dir / "after.json").is_file())
            self.assertIn(message, (run_dir / "stdout.log").read_text())
            self.assertIn(message, (run_dir / "stderr.log").read_text())
            events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["status"], "error")

    def test_run_all_arms_is_sequential_with_direction_defaults(self):
        for direction, expected_top_k, tracks in (
            ("fwd", 2, ("cold", "warm")),
            ("bwd", 3, ("cold",)),
        ):
            calls = []

            def fake_run(
                current_direction,
                arm,
                top_k,
                output_dir,
                warmup,
                repeat,
                seed,
                _calls=calls,
                _direction=direction,
                **kwargs,
            ):
                _calls.append(
                    {
                        "direction": autotune.Direction(current_direction).value,
                        "arm": arm,
                        "top_k": top_k,
                        "output_dir": str(output_dir),
                        "warmup": warmup,
                        "repeat": repeat,
                        "seed": seed,
                        "cache_mode": kwargs["cache_mode"],
                    }
                )
                return types.SimpleNamespace(
                    run_directory=Path(output_dir) / f"{arm}__rep-0000",
                    summary={
                        "status": "ok",
                        "R": 4 if _direction == "fwd" else 9,
                        "selected": top_k,
                        "validation": {"attempts": top_k},
                        "profile": {"attempts": top_k},
                    },
                )

            with tempfile.TemporaryDirectory() as directory, patch.object(autotune, "run_gdn_search", side_effect=fake_run):
                results = autotune.run_all_arms(
                    direction,
                    directory,
                    7,
                    11,
                    seed=23,
                    cache_tracks=tracks,
                )
                self.assertEqual(len(results), len(tracks) * len(autotune.ARMS))
                self.assertEqual(
                    [(call["cache_mode"], call["arm"]) for call in calls],
                    [(track, arm) for track in tracks for arm in autotune.ARMS],
                )
                self.assertTrue(all(call["top_k"] == expected_top_k for call in calls))
                self.assertTrue(all(call["warmup"] == 7 for call in calls))
                self.assertTrue(all(call["repeat"] == 11 for call in calls))
                summary_path = Path(directory) / direction / "all_arms_summary.json"
                summary = json.loads(summary_path.read_text())
                self.assertEqual(summary["status"], "ok")
                self.assertEqual(summary["top_k"], expected_top_k)
                self.assertEqual(len(summary["runs"]), len(calls))


def inspect_signature(factory):
    # Local import keeps the module's dependency list visibly stdlib-only.
    import inspect

    return inspect.signature(factory)


if __name__ == "__main__":
    unittest.main()
