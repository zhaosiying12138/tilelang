"""Real GuidedAutoTuner driver for the frozen classic operator suites.

Importing this module is CPU-safe.  A production run explicitly constructs CUDA
inputs and real operator ``JITKernel`` objects; no work is hidden in import-time
code.  Each candidate is compiled by the frozen module's ``kernel_factory`` and
profiled by the stock ``AutoTuner`` through that kernel's real ``Profiler``.

``repeat_search.py`` is the serial multi-process scheduler.  This file's CLI
executes exactly one suite/arm/repetition in the current process.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

_OPERATORS_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = _OPERATORS_ROOT.parents[2]
if __package__:
    from . import convolution, flash_attention, gemm
    from ._common import SEED, ConstraintModel
    from ..gdn.autotune import (
        _capture_output,
        _serial_compile_environment,
        _strict_json,
        _tool_cache_environment,
        _write_json,
    )
else:  # Direct execution: keep every import inside this shared worktree.
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from benchmark.z3_cost_model.operators import convolution, flash_attention, gemm
    from benchmark.z3_cost_model.operators._common import SEED, ConstraintModel
    from benchmark.z3_cost_model.gdn.autotune import (
        _capture_output,
        _serial_compile_environment,
        _strict_json,
        _tool_cache_environment,
        _write_json,
    )


DEFAULT_HARDWARE_PATH = REPOSITORY_ROOT / "manifest/hardware-sm120.json"
DEFAULT_CALIBRATION_PATH = REPOSITORY_ROOT / "manifest/calibration-sm120.json"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "evidence/G6/classic-operator-search"
ARMS = (
    "stock-raw",
    "z3-legal-exhaustive",
    "z3-cm-topk",
    "budget-raw",
    "budget-random",
)
DEFAULT_TOP_K = 10
GEMM_SQUARE_SENSITIVITY_TOP_K = (1, 5, 10, 20)
DEFAULT_REPETITIONS = (0, 1, 2)
_CACHE_MODES = {
    "cold": "enabled",
    "enabled": "enabled",
    "warm": "warm",
    "no-cache": "no-cache",
}
_SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "TILELANG_DEFAULT_TARGET",
    "TILELANG_EXECUTION_BACKEND",
    "TILELANG_AUTO_TUNING_CPU_COUNTS",
    "TILELANG_AUTO_TUNING_MAX_CPU_COUNT",
    "TILELANG_CACHE_DIR",
    "TILELANG_DISABLE_CACHE",
    "TILELANG_AUTO_TUNING_DISABLE_CACHE",
    "CUDA_CACHE_PATH",
    "CUDA_CACHE_DISABLE",
    "TORCH_EXTENSIONS_DIR",
    "TRITON_CACHE_DIR",
    "XDG_CACHE_HOME",
    "TMPDIR",
    "PYTHONPATH",
)


@dataclass(frozen=True)
class SuiteSpec:
    """One fixed benchmark shape and its exact analytical-model registration."""

    name: str
    module: Any
    shape: Any
    model_operator: str
    analytical_shape: dict[str, Any]
    raw_count: int

    @property
    def operator(self) -> str:
        return self.module.OPERATOR


def _gemm_model_shape(shape) -> dict[str, Any]:
    return {**shape.to_dict(), "dtype": "f16", "output_dtype": "f16"}


def _conv_model_shape(shape) -> dict[str, Any]:
    # cost_model.py uses K=output channels, R/S=kernel height/width.  The
    # operator's K/S fields instead mean square kernel extent/stride.
    return {
        "N": shape.N,
        "C": shape.C,
        "H": shape.H,
        "W": shape.W,
        "K": shape.F,
        "R": shape.K,
        "S": shape.K,
        "stride_h": shape.S,
        "stride_w": shape.S,
        "dilation_h": shape.D,
        "dilation_w": shape.D,
        "pad_h": shape.P,
        "pad_w": shape.P,
        "dtype": "f16",
        "output_dtype": "f16",
    }


def _fa_model_shape(shape) -> dict[str, Any]:
    return {
        "B": shape.B,
        "H": shape.H,
        "S": shape.S,
        "D": shape.D,
        "causal": shape.is_causal,
        "dtype": "f16",
        "output_dtype": "f16",
    }


def _make_suites() -> dict[str, SuiteSpec]:
    specs = (
        SuiteSpec("gemm_square", gemm, gemm.PRIMARY_SHAPE, "gemm", _gemm_model_shape(gemm.PRIMARY_SHAPE), gemm.RAW_COUNT),
        SuiteSpec("gemm_short_k", gemm, gemm.SHORT_K_SHAPE, "gemm", _gemm_model_shape(gemm.SHORT_K_SHAPE), gemm.RAW_COUNT),
        SuiteSpec(
            "conv2d_stride_tail",
            convolution,
            convolution.PRIMARY_SHAPE,
            "conv2d_nhwc",
            _conv_model_shape(convolution.PRIMARY_SHAPE),
            convolution.RAW_COUNT,
        ),
        SuiteSpec(
            "flash_attention_causal",
            flash_attention,
            flash_attention.PRIMARY_SHAPE,
            "flash_attention_fwd",
            _fa_model_shape(flash_attention.PRIMARY_SHAPE),
            flash_attention.RAW_COUNT,
        ),
    )
    return {spec.name: spec for spec in specs}


SUITES = _make_suites()
SUITE_NAMES = tuple(SUITES)


def get_suite(suite: str | SuiteSpec) -> SuiteSpec:
    if isinstance(suite, SuiteSpec):
        return suite
    if suite not in SUITES:
        raise ValueError(f"suite must be one of {SUITE_NAMES}, got {suite!r}")
    return SUITES[suite]


def model_shape(suite: str | SuiteSpec) -> dict[str, Any]:
    """Return the exact shape mapping consumed by the registered CM plugin."""

    return copy.deepcopy(get_suite(suite).analytical_shape)


def resolve_top_k(suite: str | SuiteSpec, top_k: int | None) -> int:
    """Apply the pre-registered budget; never adapt it to observed results."""

    selected = get_suite(suite)
    value = DEFAULT_TOP_K if top_k is None else top_k
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("top_k must be an integer")
    allowed = GEMM_SQUARE_SENSITIVITY_TOP_K if selected.name == "gemm_square" else (DEFAULT_TOP_K,)
    if value not in allowed:
        raise ValueError(f"{selected.name} top_k must be one of {allowed}; got {value}")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_hardware(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        hardware = json.load(stream)
    if not isinstance(hardware, dict) or not isinstance(hardware.get("device"), dict):
        raise ValueError(f"Hardware manifest must contain a device object: {path}")
    return hardware


def constraint_model_from_hardware(hardware_path: str | Path) -> tuple[ConstraintModel, dict[str, Any]]:
    """Read explicit resource facts without probing a GPU or guessing atoms."""

    path = Path(hardware_path).expanduser().resolve(strict=False)
    hardware = _read_hardware(path)
    device = hardware["device"]
    capability = device.get("compute_capability")
    target = f"sm_{capability.replace('.', '')}" if isinstance(capability, str) else None
    # ConstraintModel deliberately has no SM120 atom rule.  Passing sm_120 thus
    # retains target-specific questions as unknown instead of pruning them.
    model = ConstraintModel(
        target=target,
        max_threads_per_block=device.get("max_threads_per_block"),
        shared_memory_per_block_limit_bytes=device.get("shared_memory_per_block_optin_bytes"),
        source=f"{path}:device",
    )
    return model, hardware


class OperatorFixture:
    """Fixed-seed inputs and module-owned FP32 reference/strict validation.

    One fixture is shared by every candidate in a tuner.  The supply callback
    returns the same tensor objects in PrimFunc input order.  The reference is
    intentionally evaluated by the module on every stock reference invocation:
    validation and the separate stock reference-timing call therefore never
    confuse a cached output with reference execution time.
    """

    def __init__(
        self,
        spec: str | SuiteSpec,
        *,
        seed: int = SEED,
        device: Any = "cuda",
        input_factory: Any = None,
        reference_factory: Any = None,
        validate_factory: Any = None,
    ):
        self.spec = get_suite(spec)
        self.shape = self.spec.shape
        self.seed = seed
        self.device = device
        module = self.spec.module
        self._reference = module.reference if reference_factory is None else reference_factory
        self._validate = module.validate if validate_factory is None else validate_factory
        factory = module.make_inputs if input_factory is None else input_factory
        self.inputs = factory(shape=self.shape, device=device, seed=seed)
        self.input_names = tuple(self.inputs)

    def _check_shared_inputs(self, args) -> None:
        if len(args) != len(self.input_names):
            raise ValueError(f"Expected {len(self.input_names)} inputs, got {len(args)}")
        for name, actual in zip(self.input_names, args, strict=True):
            if actual is not self.inputs[name]:
                raise ValueError(f"{self.spec.name} input {name!r} does not come from the shared fixed-seed fixture")

    def supply_prog(self, params) -> list[Any]:
        if len(params) != len(self.input_names):
            raise ValueError(f"Profiler supplied {len(params)} parameters; expected {len(self.input_names)}")
        return [self.inputs[name] for name in self.input_names]

    def reference_program(self, *args):
        self._check_shared_inputs(args)
        return self._reference(self.inputs, shape=self.shape)

    def manual_check_prog(self, actual_outputs, reference_outputs):
        if len(actual_outputs) != 1 or len(reference_outputs) != 1:
            raise ValueError("Classic operators expect exactly one actual and one reference output")
        return self._validate(actual_outputs[0], reference_outputs[0], shape=self.shape)


def make_candidate_factory(
    suite: str | SuiteSpec,
    *,
    fixture: OperatorFixture,
    target: Any = "auto",
    pass_configs: dict[str, Any] | None = None,
    compile: bool = True,
    kernel_factory: Any = None,
):
    """Return an explicit-signature factory that calls the frozen module.

    Dependencies live in keyword defaults, not closure cells: stock AutoTuner
    inspects ``fn.__closure__`` and rejects non-scalar captured state.
    """

    spec = get_suite(suite)
    if fixture.spec.name != spec.name:
        raise ValueError("fixture and candidate factory suites differ")

    frozen_pass_configs = copy.deepcopy(pass_configs)
    if spec.module is gemm:

        def candidate_factory(
            block_M,
            block_N,
            block_K,
            num_stages,
            threads,
            enable_rasterization,
            *,
            _fixture=fixture,
            _target=target,
            _pass_configs=frozen_pass_configs,
            _compile=compile,
            _kernel_factory=kernel_factory,
        ):
            config = {
                "block_M": block_M,
                "block_N": block_N,
                "block_K": block_K,
                "num_stages": num_stages,
                "threads": threads,
                "enable_rasterization": enable_rasterization,
            }
            factory = _fixture.spec.module.kernel_factory if _kernel_factory is None else _kernel_factory
            return factory(
                config,
                shape=_fixture.shape,
                target=_target,
                pass_configs=_pass_configs,
                compile=_compile,
            )

    elif spec.module is convolution:

        def candidate_factory(
            block_M,
            block_N,
            block_K,
            num_stages,
            threads,
            *,
            _fixture=fixture,
            _target=target,
            _pass_configs=frozen_pass_configs,
            _compile=compile,
            _kernel_factory=kernel_factory,
        ):
            config = {
                "block_M": block_M,
                "block_N": block_N,
                "block_K": block_K,
                "num_stages": num_stages,
                "threads": threads,
            }
            factory = _fixture.spec.module.kernel_factory if _kernel_factory is None else _kernel_factory
            return factory(
                config,
                shape=_fixture.shape,
                target=_target,
                pass_configs=_pass_configs,
                compile=_compile,
            )

    elif spec.module is flash_attention:

        def candidate_factory(
            block_M,
            block_N,
            num_stages,
            threads,
            *,
            _fixture=fixture,
            _target=target,
            _pass_configs=frozen_pass_configs,
            _compile=compile,
            _kernel_factory=kernel_factory,
        ):
            config = {
                "block_M": block_M,
                "block_N": block_N,
                "num_stages": num_stages,
                "threads": threads,
            }
            factory = _fixture.spec.module.kernel_factory if _kernel_factory is None else _kernel_factory
            return factory(
                config,
                shape=_fixture.shape,
                target=_target,
                pass_configs=_pass_configs,
                compile=_compile,
            )

    else:
        raise ValueError(f"Unsupported operator module: {spec.module!r}")
    return candidate_factory


def _configure_tuner(
    tuner_cls,
    *,
    spec: SuiteSpec,
    fixture: OperatorFixture,
    arm: str,
    top_k: int,
    seed: int,
    space: Any,
    model: Any,
    warmup: int,
    repeat: int,
    target: Any,
    pass_configs: dict[str, Any] | None,
    kernel_factory: Any,
):
    options = {
        "fixture": fixture,
        "target": target,
        "pass_configs": pass_configs,
        "kernel_factory": kernel_factory,
    }
    candidate_factory = make_candidate_factory(spec, compile=True, **options)
    elaborate_factory = make_candidate_factory(spec, compile=False, **options)
    tuner = tuner_cls(
        candidate_factory,
        space=space,
        model=model,
        top_k=top_k,
        arm=arm,
        budget=top_k,
        seed=seed,
    )
    tuner.set_compile_args(out_idx=[-1], target=target, pass_configs=pass_configs)
    tuner.set_profile_args(
        warmup=warmup,
        rep=repeat,
        ref_prog=fixture.reference_program,
        supply_prog=fixture.supply_prog,
        manual_check_prog=fixture.manual_check_prog,
        cache_input_tensors=True,
        backend="event",
        rtol=0.02,
        atol=0.02,
        max_mismatched_ratio=0.0,
        skip_check=False,
    )

    def compile_candidate(**config):
        per_config = config.pop("__pass_configs__", None)
        if per_config is not None:
            raise ValueError("Frozen classic suites do not accept per-config pass_configs")
        return candidate_factory(**config)

    def elaborate_candidate(**config):
        per_config = config.pop("__pass_configs__", None)
        if per_config is not None:
            raise ValueError("Frozen classic suites do not accept per-config pass_configs")
        return elaborate_factory(**config)

    # The real factory returns a compiled JITKernel.  PrimFunc elaboration uses
    # the same frozen factory with compile=False; no replacement IR is built.
    tuner.jit_compile = compile_candidate
    tuner.jit_elaborate = elaborate_candidate
    return tuner


def _environment_payload(spec, arm, seed, cache_track, output_dir):
    packages = {}
    for name in ("tilelang", "torch", "tvm", "z3-solver"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema": "tilelang.classic-operators.autotune-environment/v1",
        "utc_captured": _utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "cwd": os.getcwd(),
        "repository_root": str(REPOSITORY_ROOT),
        "output_dir": str(Path(output_dir).resolve(strict=False)),
        "pid": os.getpid(),
        "suite": spec.name,
        "arm": arm,
        "seed": seed,
        "cache_track": cache_track,
        "environment": {name: os.environ.get(name) for name in _SAFE_ENVIRONMENT_KEYS if os.environ.get(name) is not None},
        "gpu_query": "not performed by metadata collection",
    }


class _ObservedOperatorTuner:
    """Preserve stock events while adding suite/cache/reference evidence."""

    def __init__(self, tuner, *, run_dir, log_paths, run_metadata, cache_mode):
        self.tuner = tuner
        self.run_dir = Path(run_dir)
        self.log_paths = log_paths
        self.run_metadata = copy.deepcopy(run_metadata)
        self.cache_mode = cache_mode
        self.last_summary = None

    def _augment(self, summary, result):
        if not isinstance(summary, dict):
            return summary
        summary = copy.deepcopy(summary)
        summary["operator_run"] = copy.deepcopy(self.run_metadata)
        summary["run_cache_directory"] = str(self.run_dir / "cache")
        cache = summary.get("cache")
        if isinstance(cache, dict):
            cache["pipeline_cache_directory"] = str(self.run_dir / "cache")
            cache["directory_scope"] = (
                "directory/identity concern the AutoTuner result cache; "
                "pipeline_cache_directory is the isolated TileLang/CUDA/Torch cache root"
            )
        summary["reference_latency_ms"] = getattr(result, "ref_latency", None) if result is not None else None
        validation = summary.get("validation", {})
        profile = summary.get("profile", {})
        failures = {
            "compile": summary.get("compile_fail"),
            "validation": validation.get("failed") if isinstance(validation, dict) else None,
            "profile": profile.get("fail") if isinstance(profile, dict) else None,
        }
        summary["candidate_failures"] = failures
        trial_failures = any(
            status not in ("ok", "cache_hit") and isinstance(count, int) and count > 0
            for status, count in summary.get("trials", {}).items()
        )
        summary["has_recorded_candidate_failures"] = (
            any(isinstance(value, int) and value > 0 for value in failures.values()) or trial_failures or bool(summary.get("errors"))
        )
        summary["failure_policy"] = "candidate failures remain in events/summary; factory/run exceptions are re-raised"
        return _strict_json(summary)

    def run(self, **kwargs):
        result = None
        try:
            with _capture_output(*self.log_paths), _tool_cache_environment(self.run_dir, self.cache_mode):
                result = self.tuner.run(**kwargs)
        finally:
            self.last_summary = self._augment(getattr(self.tuner, "last_summary", None), result)
        return result


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def run_search(
    suite,
    arm,
    top_k,
    output_dir,
    warmup=10,
    repeat=50,
    seed=SEED,
    *,
    hardware_path=DEFAULT_HARDWARE_PATH,
    calibration_path=DEFAULT_CALIBRATION_PATH,
    cache_mode="cold",
    repetition=0,
    timeout=180,
    device="cuda",
    target="auto",
    pass_configs=None,
    tuner_cls=None,
    experiment_cls=None,
    cost_model_cls=None,
    fixture_factory=OperatorFixture,
    input_factory=None,
    reference_factory=None,
    validate_factory=None,
    kernel_factory=None,
):
    """Run one real suite/arm/repetition through ``SearchExperiment``.

    Dependency injections are for CPU tests only.  Production defaults resolve
    the actual GuidedAutoTuner, SearchExperiment, CudaCostModel and frozen
    operator factories.  Exceptions are recorded and re-raised.
    """

    spec = get_suite(suite)
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    selected_top_k = resolve_top_k(spec, top_k)
    for value, name in ((warmup, "warmup"), (repeat, "repeat"), (repetition, "repetition"), (timeout, "timeout")):
        _nonnegative_int(value, name)
    _nonnegative_int(seed, "seed")
    if cache_mode not in _CACHE_MODES:
        raise ValueError(f"cache_mode must be one of {tuple(_CACHE_MODES)}")
    experiment_cache_mode = _CACHE_MODES[cache_mode]

    if tuner_cls is None or experiment_cls is None or cost_model_cls is None:
        from tilelang.autotuner import GuidedAutoTuner, SearchExperiment
        from tilelang.cuda.cost_model import CudaCostModel

        tuner_cls = GuidedAutoTuner if tuner_cls is None else tuner_cls
        experiment_cls = SearchExperiment if experiment_cls is None else experiment_cls
        cost_model_cls = CudaCostModel if cost_model_cls is None else cost_model_cls

    output_root = Path(output_dir).expanduser().resolve(strict=False)
    hardware_path = Path(hardware_path).expanduser().resolve(strict=False)
    calibration_path = Path(calibration_path).expanduser().resolve(strict=False)
    run_dir = output_root / f"{arm}__rep-{repetition:04d}"
    log_paths = (run_dir / "stdout.log", run_dir / "stderr.log")
    shape = spec.shape.to_dict()
    analytical_shape = model_shape(spec)
    base_config = {
        "schema": "tilelang.classic-operators.autotune-config/v1",
        "suite": spec.name,
        "operator": spec.operator,
        "model_operator": spec.model_operator,
        "shape": shape,
        "model_shape": analytical_shape,
        "raw_candidate_count": spec.raw_count,
        "configuration_keys": list(spec.module.CONFIG_KEYS),
        "arm": arm,
        "top_k": selected_top_k,
        "budget": selected_top_k,
        "sensitivity_top_k": list(GEMM_SQUARE_SENSITIVITY_TOP_K) if spec.name == "gemm_square" else [DEFAULT_TOP_K],
        "warmup": warmup,
        "repeat": repeat,
        "timeout": timeout,
        "seed": seed,
        "repetition": repetition,
        "device": str(device),
        "target": target,
        "pass_configs": pass_configs,
        "cache_track": cache_mode,
        "hardware_path": str(hardware_path),
        "calibration_path": str(calibration_path),
        "require_measured_calibration": True,
        "reference": "frozen operator module independent FP32 reference",
        "validation": "real Profiler.manual_assert_close -> frozen module.validate; max_mismatched_ratio=0",
        "max_mismatched_ratio": 0.0,
        "shared_fixed_seed_inputs": True,
        "reference_timing": "stock Profiler.do_bench(ref_prog), separate from candidate do_bench",
        "serial_candidate_initialization": True,
        "use_pipeline": False,
        "enable_grouped_compile": False,
        "benchmark_multi_gpu": False,
        "early_stop": False,
    }
    run_metadata = {
        "suite": spec.name,
        "operator": spec.operator,
        "model_operator": spec.model_operator,
        "shape": shape,
        "model_shape": analytical_shape,
        "raw_count": spec.raw_count,
        "top_k": selected_top_k,
        "budget": selected_top_k,
        "seed": seed,
        "cache_track": cache_mode,
        "warmup": warmup,
        "repeat": repeat,
        "reference": base_config["reference"],
        "validation": base_config["validation"],
    }

    def tuner_factory():
        # SearchExperiment has already created an empty run directory.  Keep
        # construction failures and native diagnostics in that same run.
        with _capture_output(*log_paths), _tool_cache_environment(run_dir, experiment_cache_mode):
            _write_json(run_dir / "config.json", base_config)
            _write_json(run_dir / "environment.json", _environment_payload(spec, arm, seed, cache_mode, output_root))
            constraint_model, _hardware = constraint_model_from_hardware(hardware_path)
            space = spec.module.config_space(shape=spec.shape, model=constraint_model)
            raw_count = len(space.raw_configs())
            if raw_count != spec.raw_count or len(space) != spec.raw_count:
                raise RuntimeError(f"{spec.name} raw space changed: expected {spec.raw_count}, got {raw_count}/{len(space)}")
            model = cost_model_cls.from_files(
                hardware_path,
                calibration_path,
                spec.model_operator,
                analytical_shape,
                require_measured=True,
            )
            fixture = fixture_factory(
                spec,
                seed=seed,
                device=device,
                input_factory=input_factory,
                reference_factory=reference_factory,
                validate_factory=validate_factory,
            )
            tuner = _configure_tuner(
                tuner_cls,
                spec=spec,
                fixture=fixture,
                arm=arm,
                top_k=selected_top_k,
                seed=seed,
                space=space,
                model=model,
                warmup=warmup,
                repeat=repeat,
                target=target,
                pass_configs=pass_configs,
                kernel_factory=kernel_factory,
            )
            detailed_config = copy.deepcopy(base_config)
            detailed_config["constraint_model"] = constraint_model.as_dict()
            detailed_config["cost_model"] = model.model_card()
            detailed_config["constraints"] = spec.module.suite_manifest(model=constraint_model)["constraints"]
            _write_json(run_dir / "config.json", detailed_config)
        return _ObservedOperatorTuner(
            tuner,
            run_dir=run_dir,
            log_paths=log_paths,
            run_metadata=run_metadata,
            cache_mode=experiment_cache_mode,
        )

    experiment = experiment_cls(output_root)
    run_kwargs = {
        "warmup": warmup,
        "rep": repeat,
        "timeout": timeout,
        "use_pipeline": False,
        "enable_grouped_compile": False,
        "benchmark_multi_gpu": False,
        "early_stop": False,
    }
    metadata = {
        "suite": spec.name,
        "operator": spec.model_operator,
        "seed": seed,
        "shape": analytical_shape,
        "top_k": selected_top_k,
        "cache_track": cache_mode,
    }
    with _serial_compile_environment():
        return experiment.run(
            tuner_factory,
            arm=arm,
            repetition=repetition,
            cache_mode=experiment_cache_mode,
            run_kwargs=run_kwargs,
            metadata=metadata,
        )


# Descriptive alias for callers that prefer an explicit operator name.
run_operator_search = run_search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITE_NAMES, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--hardware", type=Path, default=DEFAULT_HARDWARE_PATH)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--cache-mode", choices=tuple(_CACHE_MODES), default="cold")
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target", default="auto")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = run_search(
        args.suite,
        args.arm,
        args.top_k,
        args.output_dir,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        hardware_path=args.hardware,
        calibration_path=args.calibration,
        cache_mode=args.cache_mode,
        repetition=args.repetition,
        timeout=args.timeout,
        device=args.device,
        target=args.target,
    )
    print(json.dumps(_strict_json(result.summary), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
