"""Shared, lazy-imported utilities for the classic operator suite.

No function in this module probes a GPU or treats a compiler result as a
constraint.  CUDA properties are supplied explicitly through ConstraintModel.
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

SEED = 20261026
UPSTREAM_REVISION = "80364b3d9758854a9692cc0f14089025f54d3f3f"
SPACE_SOURCE = "tilelang/autotuner/space.py"


def torch_module() -> Any:
    """Import torch only on an input/reference/validation path."""
    return importlib.import_module("torch")


def ceildiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def require_exact_keys(config: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError(f"config must be a mapping, got {type(config).__name__}")
    expected = set(keys)
    actual = set(config)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"configuration keys mismatch: missing={missing}, extra={extra}")
    # Deliberately read every key: no alias or silently ignored compatibility key.
    return {key: config[key] for key in keys}


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return int(value)


def nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return int(value)


def boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool, got {value!r}")
    return value


def make_fp16_inputs(
    specifications: Sequence[tuple[str, tuple[int, ...], float]],
    *,
    device: Any,
    seed: int,
) -> dict[str, Any]:
    """Generate reproducible CPU FP32 normals, round to FP16, then transfer.

    A separate generator per tensor makes adding or reordering other tensors
    unable to change an existing tensor's values.  Values therefore do not
    depend on a CUDA generator or on GPU availability.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    torch = torch_module()
    result = {}
    for offset, (name, shape, std) in enumerate(specifications):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + offset)
        values = torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu")
        values = (values * std).to(dtype=torch.float16).contiguous()
        result[name] = values if device is None else values.to(device=device)
    return result


def check_fp16_inputs(inputs: Mapping[str, Any], expected_shapes: Mapping[str, tuple[int, ...]]) -> None:
    if not isinstance(inputs, Mapping):
        raise TypeError("inputs must be a mapping of named tensors")
    expected_names = set(expected_shapes)
    actual_names = set(inputs)
    if actual_names != expected_names:
        raise ValueError(
            f"input names mismatch: missing={sorted(expected_names - actual_names)}, extra={sorted(actual_names - expected_names)}"
        )
    torch = torch_module()
    for name, expected_shape in expected_shapes.items():
        tensor = inputs[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"inputs[{name!r}] must be a torch.Tensor")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"inputs[{name!r}] has shape {tuple(tensor.shape)}, expected {expected_shape}")
        if tensor.dtype != torch.float16:
            raise TypeError(f"inputs[{name!r}] must have dtype torch.float16, got {tensor.dtype}")


@contextmanager
def ieee_fp32_precision():
    """Disable TF32 for reference matmul/convolution and restore prior flags."""
    torch = torch_module()
    matmul_backend = torch.backends.cuda.matmul
    cudnn_backend = torch.backends.cudnn
    old_matmul = matmul_backend.allow_tf32
    old_cudnn = cudnn_backend.allow_tf32
    matmul_backend.allow_tf32 = False
    cudnn_backend.allow_tf32 = False
    try:
        yield
    finally:
        matmul_backend.allow_tf32 = old_matmul
        cudnn_backend.allow_tf32 = old_cudnn


@dataclass(frozen=True)
class ValidationReport:
    operator: str
    config_id: str | None
    passed: bool
    mismatched_elements: int
    total_elements: int
    mismatched_ratio: float
    max_mismatched_ratio: float
    max_abs_error: float
    max_rel_error: float
    atol: float
    rtol: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "config_id": self.config_id,
            "passed": self.passed,
            "mismatched_elements": self.mismatched_elements,
            "total_elements": self.total_elements,
            "mismatched_ratio": self.mismatched_ratio,
            "max_mismatched_ratio": self.max_mismatched_ratio,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "atol": self.atol,
            "rtol": self.rtol,
        }


class CorrectnessError(AssertionError):
    """Raised after recording a complete strict correctness report."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(
            f"{report.operator} correctness failed: "
            f"{report.mismatched_elements}/{report.total_elements} mismatched "
            f"(ratio={report.mismatched_ratio:.9g}, allowed={report.max_mismatched_ratio}), "
            f"max_abs_error={report.max_abs_error:.9g}, max_rel_error={report.max_rel_error:.9g}"
        )


def canonical_config_id(config: Mapping[str, Any]) -> str:
    # Import the upstream implementation so operator IDs and pruner IDs cannot
    # diverge.  This path is never used while merely importing an operator.
    from tilelang.autotuner.space import ConfigSpace

    return ConfigSpace.config_id(config)


def strict_validate(
    operator: str,
    actual: Any,
    expected: Any,
    *,
    config: Mapping[str, Any] | None,
    atol: float,
    rtol: float,
) -> ValidationReport:
    """Compare every element in FP32; any out-of-tolerance element fails.

    The reference is required to remain FP32 while the kernel output is
    required to be FP16.  Non-finite values are always mismatches, including
    matching infinities.  Relative error uses the same small denominator
    regularization as TileLang's torch_assert_close (1e-12).
    """
    torch = torch_module()
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise TypeError("actual and expected must be torch.Tensor objects")
    if actual.shape != expected.shape:
        raise ValueError(f"output shape mismatch: actual={tuple(actual.shape)}, expected={tuple(expected.shape)}")
    if actual.dtype != torch.float16:
        raise TypeError(f"actual kernel output must be torch.float16, got {actual.dtype}")
    if expected.dtype != torch.float32:
        raise TypeError(f"reference output must be torch.float32, got {expected.dtype}")
    if actual.numel() == 0:
        raise ValueError("empty outputs are outside this benchmark suite")
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be finite and non-negative")

    actual32 = actual.detach().to(dtype=torch.float32)
    expected32 = expected.detach().to(device=actual32.device, dtype=torch.float32)
    difference = (actual32 - expected32).abs()
    relative = difference / (expected32.abs() + 1e-12)
    finite = torch.isfinite(actual32) & torch.isfinite(expected32)
    close = torch.isclose(actual32, expected32, rtol=rtol, atol=atol, equal_nan=False)
    mismatched = ~close | ~finite
    mismatched_elements = int(mismatched.sum().item())
    total_elements = int(actual32.numel())

    # Preserve useful diagnostics even for NaN/Inf failures.
    difference = torch.nan_to_num(difference, nan=float("inf"), posinf=float("inf"), neginf=0.0)
    relative = torch.nan_to_num(relative, nan=float("inf"), posinf=float("inf"), neginf=0.0)
    report = ValidationReport(
        operator=operator,
        config_id=canonical_config_id(config) if config is not None else None,
        passed=mismatched_elements == 0,
        mismatched_elements=mismatched_elements,
        total_elements=total_elements,
        mismatched_ratio=mismatched_elements / total_elements,
        max_mismatched_ratio=0,
        max_abs_error=float(difference.max().item()),
        max_rel_error=float(relative.max().item()),
        atol=float(atol),
        rtol=float(rtol),
    )
    if not report.passed:
        raise CorrectnessError(report)
    return report


def compile_prim_func(prim_func: Any, *, target: Any, pass_configs: Mapping[Any, Any] | None) -> Any:
    """Compile only when explicitly called through a module kernel_factory."""
    tilelang = importlib.import_module("tilelang")
    return tilelang.compile(
        prim_func,
        out_idx=[-1],
        target=target,
        pass_configs=pass_configs,
    )


@dataclass(frozen=True)
class ConstraintModel:
    """Explicit target/resource facts; absent facts remain unknown.

    target may be "sm_80" or "sm_90" for the source-derived stock-pipeline
    GEMM instruction selection rules.  Other/None values retain target-specific
    atom questions instead of guessing.  shared_memory_per_block_limit_bytes is
    a verified usable upper bound for TileLang shared allocations, not a GPU
    model-name guess and not automatically the ordinary 48 KiB static limit.
    """

    target: str | None = None
    max_threads_per_block: int | None = None
    shared_memory_per_block_limit_bytes: int | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if self.target is not None and not isinstance(self.target, str):
            raise TypeError("target must be a string or None")
        if self.source is not None and not isinstance(self.source, str):
            raise TypeError("source must be a string or None")
        if self.max_threads_per_block is not None:
            positive_int(self.max_threads_per_block, "max_threads_per_block")
        if self.shared_memory_per_block_limit_bytes is not None:
            positive_int(self.shared_memory_per_block_limit_bytes, "shared_memory_per_block_limit_bytes")

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "max_threads_per_block": self.max_threads_per_block,
            "shared_memory_per_block_limit_bytes": self.shared_memory_per_block_limit_bytes,
            "source": self.source,
        }


def default_mma_partition_possible(
    block_m: int,
    block_n: int,
    threads: int,
    *,
    n_per_warp: int = 8,
) -> bool:
    """Existence condition from cuda ComputeDefaultWarpPartition."""
    if threads <= 0 or threads % 32 != 0 or block_m % 16 != 0 or block_n % n_per_warp != 0:
        return False
    num_warps = threads // 32
    for m_warp in range(1, num_warps + 1):
        if num_warps % m_warp:
            continue
        n_warp = num_warps // m_warp
        if block_m % (m_warp * 16) == 0 and block_n % (n_warp * n_per_warp) == 0:
            return True
    return False


def wgmma_partition_possible(block_m: int, block_n: int, threads: int) -> bool:
    """Existence condition from cuda ComputeWgmmaWarpPartition."""
    if threads <= 0 or threads % 32 != 0:
        return False
    num_warps = threads // 32
    if num_warps % 4 != 0 or block_m % 16 != 0 or block_n % 8 != 0:
        return False
    for m_warp in range(4, num_warps + 1, 4):
        if num_warps % m_warp:
            continue
        n_warp = num_warps // m_warp
        if block_m % (m_warp * 16) == 0 and block_n % (n_warp * 8) == 0:
            return True
    return False


def stock_cuda_gemm_constraint(
    block_m: int,
    block_n: int,
    block_k: int,
    threads: int,
    target: str | None,
) -> bool | None:
    """Return a provable stock-pipeline atom condition, or None if unknown."""
    if target == "sm_80":
        # FP16 inputs/FP32 accumulator select native mma.sync on Ampere.
        return default_mma_partition_possible(block_m, block_n, threads, n_per_warp=8)
    if target in ("sm_90", "sm_90a"):
        num_warps = threads // 32 if threads > 0 and threads % 32 == 0 else 0
        wgmma_selected = block_m >= 64 and num_warps >= 4 and num_warps % 4 == 0 and block_k % 16 == 0
        if wgmma_selected:
            return wgmma_partition_possible(block_m, block_n, threads)
        return default_mma_partition_possible(block_m, block_n, threads, n_per_warp=8)
    return None


def base_constraint_clauses(
    config: Mapping[str, Any],
    *,
    tile_names: Sequence[str],
    min_trip_count: int,
    model: ConstraintModel,
) -> dict[str, bool]:
    threads = config["threads"]
    stages = config["num_stages"]
    clauses = {
        "positive_tile_extents": all(config[name] > 0 for name in tile_names),
        "nonempty_pipeline_trip": min_trip_count > 0,
        "pipeline_stage_domain": stages >= 0,
        "cuda_warp_threads": threads > 0 and threads % 32 == 0 and threads <= 1024,
    }
    if model.max_threads_per_block is not None:
        clauses["measured_thread_limit"] = threads <= model.max_threads_per_block
    return clauses


def bool_clauses_to_z3(clauses: Mapping[str, bool], z3: Any, context: Any) -> list[Any]:
    # Constant configurations still become real expressions in the private Z3
    # context; Z3Pruner alone decides SAT/UNSAT/unknown.
    return [z3.BoolVal(bool(value), ctx=context) for value in clauses.values()]


def constraints_manifest(
    *,
    trip_count: str,
    shared_lower_bound: str,
    gemm_operations: Sequence[str],
    model: ConstraintModel,
) -> dict[str, Any]:
    unknown = [
        {
            "name": "full_shared_memory",
            "status": "unknown",
            "source": "src/transform/inject_pipeline.cc::PipelineRewriter::ComputeBufferVersions",
            "reason": (
                "Exact post-lowering bytes additionally depend on pipeline stage/order liveness, "
                "shared-memory reuse, alignment/padding, barriers, target and pass configs."
            ),
        },
        {
            "name": "register_usage",
            "status": "unknown",
            "source": None,
            "reason": "No reliable static register formula is available; no register-based pruning is attempted.",
        },
    ]
    if model.shared_memory_per_block_limit_bytes is None:
        unknown.append(
            {
                "name": "usable_shared_memory_limit",
                "status": "unknown",
                "source": None,
                "reason": "No GPU is queried; supply an explicit verified ConstraintModel limit to enable this clause.",
            }
        )
    if model.target not in ("sm_80", "sm_90", "sm_90a"):
        unknown.append(
            {
                "name": "target_specific_gemm_instruction_and_atom",
                "status": "unknown",
                "source": "src/cuda/op/gemm.cc::Gemm::SelectInst",
                "reason": "Target auto/unspecified or unsupported by the conservative selector; no atom-based pruning is attempted.",
            }
        )
    return {
        "policy": "Only an explicit Z3 UNSAT verdict prunes; SAT and unknown are retained. Compiler failures are never inputs to the builder.",
        "model": model.as_dict(),
        "known_clauses": [
            {
                "name": "pipeline_stage_and_trip",
                "rule": "tile extents > 0; num_stages >= 0; reduction trip > 0",
                "trip_count": trip_count,
                "source": "tilelang/language/loop.py::Pipelined and src/transform/inject_pipeline.cc::PipelineRewriter::BuildPipeline",
                "note": "A deeper pipeline than the trip count is retained: guarded prologue/body/epilogue provide no general stage<=trip legality rule.",
            },
            {
                "name": "cuda_threads",
                "rule": "0 < threads <= 1024 and threads % 32 == 0; optional explicit device limit",
                "source": "CUDA warp size and per-block thread limit; tilelang/cuda/hardware.py device fields",
            },
            {
                "name": "stock_cuda_gemm_atom_when_target_known",
                "operations": list(gemm_operations),
                "source": "src/cuda/op/gemm.cc::{AllowWgmma,ComputeDefaultWarpPartition,ComputeWgmmaWarpPartition}",
                "note": "Only local GEMM tile extents and warps are constrained; global M/N/K/reduction tails are not required to be divisible.",
            },
            {
                "name": "necessary_shared_memory_lower_bound",
                "rule": "lower_bound <= explicit usable limit, only when that limit is supplied",
                "lower_bound_bytes": shared_lower_bound,
                "source": "T.alloc_shared declarations and simultaneous GEMM operand liveness; unversioned bytes are a lower bound on versioned storage.",
            },
        ],
        "unknown": unknown,
    }


def dtype_manifest() -> dict[str, str]:
    return {"io": "float16", "accumulator": "float32", "reference": "float32"}


def manifest_json_dumps(manifest: Mapping[str, Any]) -> str:
    """Stable JSON used by tests and by the checked-in aggregate manifest."""
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
