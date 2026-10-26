"""Guided AutoTuner adapter and evidence-producing driver for public GDN pipelines.

The candidate factory in this module deliberately does not elaborate a substitute
PrimFunc.  A candidate owns a :class:`GDNRunner`; constructing it executes the
complete public forward or forward+backward pipeline once, which is the real
FlashQLA/TileLang compilation trigger.  The stock AutoTuner then profiles that
candidate through a genuine ``tilelang.profiler.Profiler`` with empty params and
fixed-seed closure inputs.

Importing this module is CPU-safe.  TileLang, Torch, and the vendored GPU modules
are loaded only when a pipeline, profiler, or search backend is constructed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import copy
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

try:
    from .config import Direction, GDNConfig
    from .correctness import (
        RTOL,
        assert_backward as gdn_assert_backward,
        assert_forward as gdn_assert_forward,
        reference_backward as gdn_reference_backward,
        reference_forward as gdn_reference_forward,
    )
    from .runner import (
        BackwardResult,
        ForwardResult,
        GDNRunner,
        make_fixed_inputs,
    )
    from .search import make_config_space
except ImportError:  # Direct execution with this directory on sys.path.
    from config import Direction, GDNConfig
    from correctness import (
        RTOL,
        assert_backward as gdn_assert_backward,
        assert_forward as gdn_assert_forward,
        reference_backward as gdn_reference_backward,
        reference_forward as gdn_reference_forward,
    )
    from runner import BackwardResult, ForwardResult, GDNRunner, make_fixed_inputs
    from search import make_config_space


GDN_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = GDN_ROOT.parents[2]
DEFAULT_HARDWARE_PATH = REPOSITORY_ROOT / "manifest/hardware-sm120.json"
DEFAULT_CALIBRATION_PATH = REPOSITORY_ROOT / "manifest/calibration-sm120.json"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "evidence/G6/gdn-search"
ARMS = (
    "stock-raw",
    "z3-legal-exhaustive",
    "z3-cm-topk",
    "budget-raw",
    "budget-random",
)
DEFAULT_TOP_K = {Direction.FWD: 2, Direction.BWD: 3}
BACKWARD_FORWARD_BLOCK_DV = 64
DEFAULT_INPUT_KWARGS = {
    "batch_size": 1,
    "num_tokens": 4096,
    "num_k_heads": 4,
    "num_v_heads": 8,
    "varlen": False,
    "cu_seqlens_list": None,
    "use_h0": True,
    "use_dht": True,
    "state_v_first": False,
}
FWD_OUTPUTS = ("o", "final_state")
BWD_OUTPUTS = ("dq", "dk", "dv", "db", "dg", "dh0")
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strict_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _strict_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _default_top_k(direction: Direction | str) -> int:
    return DEFAULT_TOP_K[Direction(direction)]


def _input_kwargs(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is not None and not isinstance(value, dict):
        raise TypeError("input_kwargs must be a dictionary or None")
    overrides = {} if value is None else value
    unknown = set(overrides) - set(DEFAULT_INPUT_KWARGS)
    if unknown:
        raise ValueError(f"Unknown GDN input options: {sorted(unknown)}")
    options = copy.deepcopy(DEFAULT_INPUT_KWARGS)
    options.update(copy.deepcopy(overrides))
    return options


def model_shape(direction: Direction | str, input_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the analytical shape corresponding to the actual fixture closure."""

    selected = Direction(direction)
    options = _input_kwargs(input_kwargs)
    use_h0 = bool(options["use_h0"])
    use_dht = use_h0 if options["use_dht"] is None else bool(options["use_dht"])
    return {
        "B": options["batch_size"],
        "T": options["num_tokens"],
        "Hg": options["num_k_heads"],
        "H": options["num_v_heads"],
        "K": 128,
        "DV": 128,
        "chunk_size": 32,
        "dtype": "bf16",
        "output_dtype": "bf16",
        "use_initial_state": use_h0,
        "use_dht": use_dht if selected is Direction.BWD else False,
        "output_final_state": True,
        "output_h": False,
        "store_o": True,
        "calc_mt": False,
    }


class GDNPipelineAdapter:
    """Own fixed-seed inputs and direction-correct reference/check callables.

    ``reference_program`` and ``manual_check_prog`` are shared by all candidates
    in one tuner.  This is required because stock ``ProfileArgs`` has one reference
    callback per tuner, not one callback per candidate.  Every candidate still has
    its own runner and public-pipeline timing adapter.
    """

    def __init__(
        self,
        direction: Direction | str,
        *,
        seed: int = 42,
        input_kwargs: dict[str, Any] | None = None,
        input_factory: Any = None,
    ):
        self.direction = Direction(direction)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self.seed = seed
        self.input_kwargs = _input_kwargs(input_kwargs)
        factory = make_fixed_inputs if input_factory is None else input_factory
        self.batch = factory(seed=seed, **copy.deepcopy(self.input_kwargs))
        self._expected_forward = None

    def reference_program(self):
        """Return references in exactly the public adapter's output order."""

        if self.direction is Direction.FWD:
            expected = gdn_reference_forward(self.batch)
            # ref_gdr order: g, o, A, h, final_state.
            return expected[1], expected[4]

        # Backward timing starts after forward artifacts exist.  Construct the
        # float64 reference artifacts once (during validation), then time only the
        # reference backward on subsequent calls, including ref_latency.
        if self._expected_forward is None:
            self._expected_forward = gdn_reference_forward(self.batch)
        expected = gdn_reference_backward(self.batch, self._expected_forward)
        return tuple(expected)

    def manual_check_prog(self, actual_outputs, reference_outputs) -> None:
        """Use GDN relative-L2/RTOL=0.02 and padding checks, never allclose."""

        if self.direction is Direction.FWD:
            if len(actual_outputs) != 2 or len(reference_outputs) != 2:
                raise ValueError("fwd validation expects (o, final_state) on both sides")
            actual = ForwardResult(
                o=actual_outputs[0],
                final_state=actual_outputs[1],
                g=None,
                A=None,
                h=None,
                cp_cache=None,
            )
            expected = (
                None,
                reference_outputs[0],
                None,
                None,
                reference_outputs[1],
            )
            gdn_assert_forward(self.batch, actual, expected, output_h=False)
            return

        if len(actual_outputs) != 6 or len(reference_outputs) != 6:
            raise ValueError("bwd validation expects (dq, dk, dv, db, dg, dh0) on both sides")
        gdn_assert_backward(
            self.batch,
            BackwardResult(*actual_outputs),
            tuple(reference_outputs),
        )


class GDNCandidateCostModel:
    """Map direction-specific search configs to ``CudaCostModel`` configs.

    ``make_config_space`` carries explicit fixed fields and ``direction`` for
    auditability.  The analytical GDN plugins accept only real knobs, so this
    adapter removes direction/fixed forward fields without changing any varying
    value.  Prediction exceptions propagate to GuidedSearch.
    """

    def __init__(self, delegate: Any, direction: Direction | str):
        if not callable(getattr(delegate, "predict", None)):
            raise TypeError("delegate must implement predict(config)")
        self.delegate = delegate
        self.direction = Direction(direction)

    def predict(self, config):
        values = copy.deepcopy(dict(config))
        config_direction = Direction(values.pop("direction"))
        if config_direction is not self.direction:
            raise ValueError(f"Model direction {self.direction.value} cannot score {config_direction.value}")
        if self.direction is Direction.FWD:
            if values.pop("prepare_h_num_stages") != 2:
                raise ValueError("fwd prepare_h_num_stages must remain fixed at 2")
            if values.pop("group_reduce_block_size") != 16:
                raise ValueError("fwd group_reduce_block_size must remain fixed at 16")
        return self.delegate.predict(values)

    @property
    def fallback_used(self):
        return self.delegate.fallback_used

    def model_card(self):
        card = copy.deepcopy(self.delegate.model_card())
        card["gdn_config_adapter"] = {
            "direction": self.direction.value,
            "removed_audit_field": "direction",
            "removed_fixed_fwd_fields": (["prepare_h_num_stages", "group_reduce_block_size"] if self.direction is Direction.FWD else []),
        }
        return _strict_json(card)


class GDNCandidateKernel:
    """AutoTuner-compatible handle for one real public GDN pipeline config."""

    def __init__(
        self,
        direction: Direction | str,
        block_DV: int = 128,
        prepare_h_num_stages: int = 2,
        group_reduce_block_size: int = 16,
        *,
        seed: int = 42,
        input_kwargs: dict[str, Any] | None = None,
        pipeline: GDNPipelineAdapter | None = None,
        runner_factory: Any = None,
        profiler_factory: Any = None,
    ):
        self.direction = Direction(direction)
        self.gdn_config = GDNConfig(
            self.direction,
            block_DV=block_DV,
            prepare_h_num_stages=prepare_h_num_stages,
            group_reduce_block_size=group_reduce_block_size,
        )
        self.config: Any = self.gdn_config
        if pipeline is None:
            pipeline = GDNPipelineAdapter(
                self.direction,
                seed=seed,
                input_kwargs=input_kwargs,
            )
        if pipeline.direction is not self.direction:
            raise ValueError("pipeline and candidate directions differ")
        self.pipeline = pipeline
        self.batch = pipeline.batch
        self._profiler_factory = profiler_factory

        factory = GDNRunner if runner_factory is None else runner_factory
        forward_block = BACKWARD_FORWARD_BLOCK_DV if self.direction is Direction.BWD else None
        self.runner = factory(self.gdn_config, forward_block_DV=forward_block)
        self.forward_artifacts = None
        if self.direction is Direction.FWD:
            # A complete public forward triggers cumsum, KKT, and fused-fwd JIT.
            self.runner.forward(self.batch, output_h=False)
        else:
            # This fixed block=64 artifact producer is outside the bwd search and
            # outside bwd timing.  Only the public backward is timed below.
            self.forward_artifacts = self.runner.forward(self.batch, output_h=False)
            self.runner.backward(self.batch, self.forward_artifacts)

        self.adapter = self._execute_public_pipeline
        self.reference_program = self.pipeline.reference_program
        self.ref_prog = self.reference_program
        self.manual_check_prog = self.pipeline.manual_check_prog
        self.params = []
        self.out_idx = []
        self.result_idx = []
        self.prim_func = (
            self.runner.api.chunk_gated_delta_rule_fwd if self.direction is Direction.FWD else self.runner.api.chunk_gated_delta_rule_bwd
        )
        self.execution_backend = "flashqla-public-pipeline"
        self.latency = None
        self.ref_latency = None

    def _execute_public_pipeline(self):
        if self.direction is Direction.FWD:
            result = self.runner.forward(self.batch, output_h=False)
            return result.o, result.final_state
        result = self.runner.backward(self.batch, self.forward_artifacts)
        return result.as_tuple()

    def __call__(self):
        return self.adapter()

    def get_profiler(self, tensor_supply_type=None):
        """Return the real Profiler; no tensor supply or placeholder adapter."""

        factory = self._profiler_factory
        if factory is None:
            from tilelang.profiler import Profiler, TensorSupplyType

            factory = Profiler
            if tensor_supply_type is None:
                tensor_supply_type = TensorSupplyType.Auto
        if tensor_supply_type is None:
            raise ValueError("tensor_supply_type is required with an injected profiler factory")
        return factory(
            params=[],
            result_idx=[],
            supply_type=tensor_supply_type,
            adapter=self.adapter,
        )

    def describe_configuration(self) -> dict[str, Any]:
        return {
            "operator": "FlashQLA chunk_gated_delta_rule public pipeline",
            "direction": self.direction.value,
            "config": self.gdn_config.parameters(),
            "input_shape": model_shape(self.direction, self.pipeline.input_kwargs),
            "seed": self.pipeline.seed,
            "outputs": list(FWD_OUTPUTS if self.direction is Direction.FWD else BWD_OUTPUTS),
            "forward_artifact_block_DV": (BACKWARD_FORWARD_BLOCK_DV if self.direction is Direction.BWD else None),
            "forward_artifacts_in_search": False,
            "public_callable": f"{self.prim_func.__module__}.{self.prim_func.__qualname__}",
            "source_method": "inspect actual public wrappers and their TileLang GDN factories",
        }

    @staticmethod
    def _source_text(value: Any) -> str:
        source = getattr(value, "func_source", None)
        if isinstance(source, str) and source:
            return source
        return inspect.getsource(value)

    def _source_sections(self) -> list[tuple[str, Any]]:
        public = self.prim_func
        if self.direction is Direction.FWD:
            related = (
                ("chunk_local_cumsum", "tilelang_chunk_local_cumsum"),
                ("kkt_solve", "tilelang_kkt_solve"),
                ("fused_gdr_fwd", "tilelang_fused_chunk_gdr_fwd"),
            )
        else:
            related = (
                ("chunk_local_cumsum", "tilelang_chunk_local_cumsum"),
                ("fused_gdr_h", "tilelang_prepare_h"),
                ("fused_gdr_bwd", "tilelang_fused_chunk_gdr_bwd"),
                ("group_reduce_vector", "tilelang_group_reduce_vector"),
            )
        sections = [("public pipeline", public)]
        for wrapper_name, factory_name in related:
            wrapper = public.__globals__.get(wrapper_name)
            if wrapper is None:
                raise RuntimeError(f"Real GDN source wrapper is missing: {wrapper_name}")
            sections.append((f"public wrapper {wrapper_name}", wrapper))
            factory = getattr(wrapper, "__globals__", {}).get(factory_name)
            if factory is None:
                raise RuntimeError(f"Real TileLang GDN factory is missing: {factory_name}")
            sections.append((f"TileLang factory {factory_name}", factory))
        return sections

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        """Return actual GDN public/factory source plus canonical configuration.

        This is source evidence, not a claim that Python orchestration is generated
        CUDA.  Generated CUDA/executor artifacts remain in the isolated TileLang
        cache.  No unrelated placeholder source is ever substituted.
        """

        del kernel_only  # The complete public provenance is useful in either mode.
        header = json.dumps(
            self.describe_configuration(),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        sections = ["# GDN candidate configuration and source provenance\n" + header]
        for label, value in self._source_sections():
            identity = f"{getattr(value, '__module__', '<jit>')}.{getattr(value, '__qualname__', type(value).__name__)}"
            sections.append(f"\n# ---- {label}: {identity} ----\n{self._source_text(value)}")
        return "\n".join(sections)

    @property
    def kernel_source(self) -> str:
        return self.get_kernel_source()

    def get_host_source(self) -> str:
        return self.get_kernel_source(kernel_only=False)

    def update_tuner_result(self, latency: float, config: dict[str, Any], ref_latency: float):
        self.latency = latency
        self.config = copy.deepcopy(dict(config))
        self.ref_latency = ref_latency
        return self

    def get_tuner_result(self) -> dict[str, Any]:
        if self.latency is None:
            raise ValueError("Tuning results are not available yet")
        return {
            "latency": self.latency,
            "config": copy.deepcopy(self.config),
            "ref_latency": self.ref_latency,
        }


def make_candidate_factory(
    direction: Direction | str,
    *,
    seed: int = 42,
    input_kwargs: dict[str, Any] | None = None,
    pipeline: GDNPipelineAdapter | None = None,
    runner_factory: Any = None,
    profiler_factory: Any = None,
    candidate_class: Any = GDNCandidateKernel,
):
    """Build an explicit-signature factory accepted by ``GuidedAutoTuner``.

    All implementation dependencies are defaults rather than closure cells:
    stock AutoTuner rejects non-scalar closure cells while inspecting ``fn``.
    """

    selected = Direction(direction)
    if pipeline is not None and pipeline.direction is not selected:
        raise ValueError("pipeline and factory directions differ")
    options = _input_kwargs(input_kwargs)

    def candidate_factory(
        direction,
        block_DV,
        prepare_h_num_stages,
        group_reduce_block_size,
        *,
        _expected_direction=selected,
        _seed=seed,
        _input_kwargs=options,
        _pipeline=pipeline,
        _runner_factory=runner_factory,
        _profiler_factory=profiler_factory,
        _candidate_class=candidate_class,
    ):
        if Direction(direction) is not _expected_direction:
            raise ValueError("candidate direction escaped its direction-specific space")
        return _candidate_class(
            direction=direction,
            block_DV=block_DV,
            prepare_h_num_stages=prepare_h_num_stages,
            group_reduce_block_size=group_reduce_block_size,
            seed=_seed,
            input_kwargs=copy.deepcopy(_input_kwargs),
            pipeline=_pipeline,
            runner_factory=_runner_factory,
            profiler_factory=_profiler_factory,
        )

    return candidate_factory


class _GDNElaboration:
    """Validation-only config descriptor; supply_prog makes PrimFunc unnecessary."""

    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.params = []
        self.buffer_map = {}


def _elaborate_candidate(**config):
    config.pop("__pass_configs__", None)
    return _GDNElaboration(config)


def _empty_input_supply(params):
    if len(params) != 0:
        raise RuntimeError("GDN Profiler must have empty params; closure inputs are mandatory")
    return []


def _configure_tuner(
    tuner_cls,
    *,
    direction,
    arm,
    top_k,
    seed,
    space,
    model,
    pipeline,
    warmup,
    repeat,
    runner_factory,
    profiler_factory,
):
    candidate_factory = make_candidate_factory(
        direction,
        seed=seed,
        input_kwargs=pipeline.input_kwargs,
        pipeline=pipeline,
        runner_factory=runner_factory,
        profiler_factory=profiler_factory,
    )
    tuner = tuner_cls(
        candidate_factory,
        space=space,
        model=model,
        top_k=top_k,
        arm=arm,
        budget=top_k,
        seed=seed,
    )
    tuner.set_profile_args(
        warmup=warmup,
        rep=repeat,
        ref_prog=pipeline.reference_program,
        supply_prog=_empty_input_supply,
        manual_check_prog=pipeline.manual_check_prog,
        cache_input_tensors=False,
        backend="event",
        rtol=RTOL,
        max_mismatched_ratio=0.0,
    )

    def compile_candidate(**config):
        config.pop("__pass_configs__", None)
        return candidate_factory(**config)

    # Bypass only PrimFunc compilation.  Candidate construction above performs
    # the real public-pipeline JIT trigger; all other stock tuner paths remain.
    tuner.jit_compile = compile_candidate
    tuner.jit_elaborate = _elaborate_candidate
    return tuner


@contextmanager
def _capture_output(stdout_path: Path, stderr_path: Path):
    """Capture Python and native stdout/stderr, restoring process state on error."""

    sys.stdout.flush()
    sys.stderr.flush()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        # Append/text mode handles pytest/caller-owned sys.stdout objects as well
        # as ordinary fd-backed streams; dup2 additionally captures native logs.
        with (
            stdout_path.open("a", encoding="utf-8", buffering=1) as stdout_stream,
            stderr_path.open("a", encoding="utf-8", buffering=1) as stderr_stream,
        ):
            os.dup2(stdout_stream.fileno(), 1)
            os.dup2(stderr_stream.fileno(), 2)
            try:
                with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
                    yield
            finally:
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                finally:
                    os.dup2(saved_stdout, 1)
                    os.dup2(saved_stderr, 2)
    finally:
        os.close(saved_stdout)
        os.close(saved_stderr)


@contextmanager
def _tool_cache_environment(run_dir: Path, cache_mode: str):
    """Isolate CUDA/Torch/XDG/tmp caches inside one prepared run directory."""

    names = (
        "CUDA_CACHE_PATH",
        "CUDA_CACHE_DISABLE",
        "TORCH_EXTENSIONS_DIR",
        "TRITON_CACHE_DIR",
        "XDG_CACHE_HOME",
        "TMPDIR",
    )
    previous = {name: os.environ.get(name) for name in names}
    cache = run_dir / "cache"
    paths = {
        "CUDA_CACHE_PATH": cache / "nvidia",
        "TORCH_EXTENSIONS_DIR": cache / "torch-extensions",
        "TRITON_CACHE_DIR": cache / "triton",
        "XDG_CACHE_HOME": cache / "xdg",
        "TMPDIR": cache / "tmp",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        os.environ[name] = str(path)
    os.environ["CUDA_CACHE_DISABLE"] = "1" if cache_mode == "no-cache" else "0"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _serial_compile_environment():
    """Prevent concurrent candidate initialization from executing GPU pipelines."""

    names = ("TILELANG_AUTO_TUNING_CPU_COUNTS", "TILELANG_AUTO_TUNING_MAX_CPU_COUNT")
    previous = {name: os.environ.get(name) for name in names}
    os.environ["TILELANG_AUTO_TUNING_CPU_COUNTS"] = "1"
    os.environ["TILELANG_AUTO_TUNING_MAX_CPU_COUNT"] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _environment_payload(direction, arm, seed, cache_track, output_dir):
    packages = {}
    for name in ("tilelang", "torch", "tvm", "z3-solver"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema": "tilelang.gdn.autotune-environment/v1",
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
        "direction": Direction(direction).value,
        "arm": arm,
        "seed": seed,
        "cache_track": cache_track,
        "environment": {name: os.environ.get(name) for name in _SAFE_ENVIRONMENT_KEYS if os.environ.get(name) is not None},
        "gpu_query": "not performed by metadata collection",
    }


class _ObservedGDNTuner:
    """Add GDN evidence to the stock summary and capture per-invocation logs."""

    def __init__(self, tuner, *, run_dir, log_paths, gdn_metadata, cache_mode):
        self.tuner = tuner
        self.run_dir = Path(run_dir)
        self.log_paths = log_paths
        self.gdn_metadata = copy.deepcopy(gdn_metadata)
        self.cache_mode = cache_mode
        self.last_summary = None

    def _augment(self, summary, result):
        if not isinstance(summary, dict):
            return summary
        summary = copy.deepcopy(summary)
        summary["gdn"] = copy.deepcopy(self.gdn_metadata)
        cache = summary.get("cache")
        if isinstance(cache, dict):
            cache["pipeline_cache_directory"] = str(self.run_dir / "cache")
            cache["directory_scope"] = (
                "directory/cache identity concern the AutoTuner result cache; "
                "pipeline_cache_directory is the real FlashQLA/TileLang cache root"
            )
        if result is not None:
            summary["reference_latency_ms"] = getattr(result, "ref_latency", None)
        else:
            summary.setdefault("reference_latency_ms", None)
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


def run_gdn_search(
    direction,
    arm,
    top_k,
    output_dir,
    warmup,
    repeat,
    seed,
    *,
    hardware_path=DEFAULT_HARDWARE_PATH,
    calibration_path=DEFAULT_CALIBRATION_PATH,
    cache_mode="cold",
    repetition=0,
    timeout=180,
    input_kwargs=None,
    tuner_cls=None,
    experiment_cls=None,
    cost_model_cls=None,
    pipeline_factory=GDNPipelineAdapter,
    input_factory=None,
    runner_factory=None,
    profiler_factory=None,
):
    """Run one of five arms and return ``SearchExperiment``'s result.

    ``cache_mode='cold'`` maps to a fresh enabled cache, ``'warm'`` performs a
    prewarm plus a fresh-tuner measurement, and ``'no-cache'`` disables caches.
    Every factory/run exception is recorded by SearchExperiment and re-raised.
    Dependency injections exist solely for CPU/mock tests; production defaults
    resolve the real GuidedAutoTuner, SearchExperiment, and CudaCostModel.
    """

    selected_direction = Direction(direction)
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    if top_k is None:
        top_k = _default_top_k(selected_direction)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    for value, name in ((warmup, "warmup"), (repeat, "repeat"), (repetition, "repetition")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
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
    options = _input_kwargs(input_kwargs)
    run_dir = output_root / f"{arm}__rep-{repetition:04d}"
    log_paths = (run_dir / "stdout.log", run_dir / "stderr.log")
    shape = model_shape(selected_direction, options)
    base_config = {
        "schema": "tilelang.gdn.autotune-config/v1",
        "direction": selected_direction.value,
        "arm": arm,
        "top_k": top_k,
        "budget": top_k,
        "warmup": warmup,
        "repeat": repeat,
        "timeout": timeout,
        "seed": seed,
        "repetition": repetition,
        "cache_track": cache_mode,
        "hardware_path": str(hardware_path),
        "calibration_path": str(calibration_path),
        "require_measured_calibration": True,
        "operator": f"gdn_{selected_direction.value}",
        "shape": shape,
        "input_kwargs": options,
        "outputs": list(FWD_OUTPUTS if selected_direction is Direction.FWD else BWD_OUTPUTS),
        "reference": "vendored pure-PyTorch float64 with relative-L2 RTOL=0.02",
        "validation": "Profiler.manual_assert_close + existing GDN padding assertions",
        "forward_artifact_block_DV": (BACKWARD_FORWARD_BLOCK_DV if selected_direction is Direction.BWD else None),
        "forward_artifacts_in_search": False,
        "timing_scope": (
            "public forward only"
            if selected_direction is Direction.FWD
            else "public backward only; fixed block_DV=64 public forward artifacts precomputed"
        ),
        "compile_scope": "one complete public fwd or fwd+bwd initialization per candidate",
        "source_method": "GDNCandidateKernel.get_kernel_source/describe_configuration",
        "serial_candidate_initialization": True,
        "use_pipeline": False,
        "benchmark_multi_gpu": False,
        "early_stop": False,
    }
    gdn_metadata = {
        "direction": selected_direction.value,
        "shape": shape,
        "seed": seed,
        "cache_track": cache_mode,
        "outputs": base_config["outputs"],
        "forward_artifact_block_DV": base_config["forward_artifact_block_DV"],
        "forward_artifacts_in_search": False,
        "timing_scope": base_config["timing_scope"],
        "compile_count_scope": base_config["compile_scope"],
    }

    def tuner_factory():
        # SearchExperiment creates run_dir before invoking this factory.  Capture
        # construction failures as well as the actual tuning invocation.
        with _capture_output(*log_paths), _tool_cache_environment(run_dir, experiment_cache_mode):
            _write_json(run_dir / "config.json", base_config)
            _write_json(
                run_dir / "environment.json",
                _environment_payload(selected_direction, arm, seed, cache_mode, output_root),
            )
            space = make_config_space(selected_direction, hardware_path)
            delegate = cost_model_cls.from_files(
                hardware_path,
                calibration_path,
                f"gdn_{selected_direction.value}",
                shape,
                require_measured=True,
            )
            model = GDNCandidateCostModel(delegate, selected_direction)
            pipeline = pipeline_factory(
                selected_direction,
                seed=seed,
                input_kwargs=options,
                input_factory=input_factory,
            )
            tuner = _configure_tuner(
                tuner_cls,
                direction=selected_direction,
                arm=arm,
                top_k=top_k,
                seed=seed,
                space=space,
                model=model,
                pipeline=pipeline,
                warmup=warmup,
                repeat=repeat,
                runner_factory=runner_factory,
                profiler_factory=profiler_factory,
            )
            detailed_config = copy.deepcopy(base_config)
            detailed_config["raw_candidate_count"] = len(space)
            detailed_config["cost_model"] = model.model_card()
            _write_json(run_dir / "config.json", detailed_config)
        return _ObservedGDNTuner(
            tuner,
            run_dir=run_dir,
            log_paths=log_paths,
            gdn_metadata=gdn_metadata,
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
        "operator": f"gdn_{selected_direction.value}",
        "direction": selected_direction.value,
        "seed": seed,
        "shape": shape,
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


def _all_arms_record(result, arm, cache_track):
    return {
        "arm": arm,
        "cache_track": cache_track,
        "run_directory": str(result.run_directory),
        "status": result.summary.get("status"),
        "R": result.summary.get("R"),
        "U": result.summary.get("U"),
        "selected": result.summary.get("selected"),
        "compile_attempts": result.summary.get("compile_attempts"),
        "validation": result.summary.get("validation"),
        "profile": result.summary.get("profile"),
        "has_recorded_candidate_failures": result.summary.get("has_recorded_candidate_failures"),
    }


def run_all_arms(
    direction,
    output_dir,
    warmup,
    repeat,
    *,
    seed=42,
    cache_tracks=("cold", "warm"),
    hardware_path=DEFAULT_HARDWARE_PATH,
    calibration_path=DEFAULT_CALIBRATION_PATH,
    timeout=180,
):
    """Sequentially run five arms on cold/warm tracks; never parallelize GPUs."""

    selected_direction = Direction(direction)
    tracks = tuple(cache_tracks)
    if not tracks:
        raise ValueError("cache_tracks must not be empty")
    if len(set(tracks)) != len(tracks) or any(track not in ("cold", "warm", "no-cache") for track in tracks):
        raise ValueError("cache_tracks must be unique values from cold, warm, no-cache")
    output_root = Path(output_dir).expanduser().resolve(strict=False)
    direction_root = output_root / selected_direction.value
    summary_path = direction_root / "all_arms_summary.json"
    summary = {
        "schema": "tilelang.gdn.all-arms/v1",
        "status": "running",
        "direction": selected_direction.value,
        "top_k": _default_top_k(selected_direction),
        "warmup": warmup,
        "repeat": repeat,
        "seed": seed,
        "cache_tracks": list(tracks),
        "execution": "sequential; one compile worker and one benchmark device",
        "runs": [],
    }
    results = []
    try:
        for cache_track in tracks:
            for arm in ARMS:
                result = run_gdn_search(
                    selected_direction,
                    arm,
                    _default_top_k(selected_direction),
                    direction_root / cache_track,
                    warmup,
                    repeat,
                    seed,
                    hardware_path=hardware_path,
                    calibration_path=calibration_path,
                    cache_mode=cache_track,
                    timeout=timeout,
                )
                results.append(result)
                summary["runs"].append(_all_arms_record(result, arm, cache_track))
                _write_json(summary_path, summary)
    except BaseException as exc:
        summary["status"] = "error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(summary_path, summary)
        raise
    summary["status"] = "ok"
    _write_json(summary_path, summary)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("fwd", "bwd"), required=True)
    parser.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    parser.add_argument("--top-k", type=int, default=None, help="Defaults to fwd=2, bwd=3")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--hardware", type=Path, default=DEFAULT_HARDWARE_PATH)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument(
        "--cache-mode",
        choices=tuple(_CACHE_MODES),
        default="cold",
        help="Single-arm cache track (default: fresh enabled/cold cache)",
    )
    parser.add_argument(
        "--cache-tracks",
        nargs="+",
        choices=("cold", "warm", "no-cache"),
        default=("cold", "warm"),
        help="Sequential tracks for --arm all (default: cold warm)",
    )
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=None, help="Override only the single-arm input token count (default: 4096)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    common = {
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "hardware_path": args.hardware,
        "calibration_path": args.calibration,
        "timeout": args.timeout,
    }
    if args.arm == "all":
        if args.num_tokens is not None:
            raise ValueError("--num-tokens is supported only for a single-arm run")
        if args.top_k is not None:
            raise ValueError("--top-k is direction-fixed for --arm all (fwd=2, bwd=3)")
        results = run_all_arms(
            args.direction,
            args.output_dir,
            cache_tracks=args.cache_tracks,
            **common,
        )
        payload = {
            "status": "ok",
            "direction": args.direction,
            "runs": [
                _all_arms_record(
                    result,
                    result.summary["experiment"]["arm"],
                    result.summary["gdn"]["cache_track"],
                )
                for result in results
            ],
        }
    else:
        if args.num_tokens is not None and args.num_tokens <= 0:
            raise ValueError("--num-tokens must be positive")
        result = run_gdn_search(
            args.direction,
            args.arm,
            args.top_k,
            args.output_dir,
            args.warmup,
            args.repeat,
            args.seed,
            hardware_path=args.hardware,
            calibration_path=args.calibration,
            cache_mode=args.cache_mode,
            repetition=args.repetition,
            timeout=args.timeout,
            input_kwargs=None if args.num_tokens is None else {"num_tokens": args.num_tokens},
        )
        payload = result.summary
    print(json.dumps(_strict_json(payload), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
