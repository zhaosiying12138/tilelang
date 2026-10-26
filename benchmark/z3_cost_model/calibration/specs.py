"""Benchmark definitions and workload formulas; no accelerator imports.

Every formula describes one timed kernel launch.  In particular, GEMM FLOPs
are logical matrix-multiply FLOPs, not per-thread mma instruction counts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

MIB = 1024**2
GIB = 1024**3
MAX_DEFAULT_MEMORY_BYTES = 2 * GIB
COPY_ELEMENTS_PER_THREAD = 16
COPY_VECTOR_WIDTH = 4
DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2, "tfloat32": 4, "int32": 4}


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    family: str
    shape: tuple[int, ...]
    dtype: str
    threads: int
    inner_iterations: int
    description: str
    formula: str
    accum_dtype: str | None = None
    unroll: int = 1
    blocks_per_sm: int = 0
    fixed_blocks: int = 0
    flush_l2: bool = False
    storage_dtype: str | None = None
    allow_tf32: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["shape"] = list(self.shape)
        return result


# Shapes are deliberately unrelated to GEMM4096, Conv, FA, and the formal GDN
# tiles.  The BF16 triple is valid for a 4-warp mma.sync T.gemm (2x2 warps).
_SPECS = (
    BenchmarkSpec(
        "dram_copy", "copy", (64 * MIB,), "float32", 256, 1,
        "Global FP32 copy with a 512 MiB working set and an untimed L2 flush before each sample.",
        "bytes = 2 * elements * sizeof(dtype) * inner_iterations",
        flush_l2=True,
    ),
    BenchmarkSpec(
        "l2_resident_copy", "copy", (MIB,), "float32", 256, 16,
        "Alternating A->B/B->A copies in an L2-sized working set; no timed cache flush.",
        "bytes = 2 * elements * sizeof(dtype) * inner_iterations",
    ),
    BenchmarkSpec(
        "shared_read", "shared", (8192,), "float32", 256, 1024,
        "Repeated 32-bit shared loads after an untimed-input shared initialization.",
        "bytes = blocks * threads * inner_iterations * unroll * sizeof(dtype)",
        unroll=16, blocks_per_sm=4,
    ),
    BenchmarkSpec(
        "shared_write", "shared", (8192,), "float32", 256, 1024,
        "Repeated 32-bit shared stores with a final full shared-memory checksum image.",
        "bytes = blocks * threads * inner_iterations * unroll * sizeof(dtype)",
        unroll=16, blocks_per_sm=4,
    ),
    BenchmarkSpec(
        "bf16_mma_gemm", "gemm", (96, 80, 48), "bfloat16", 128, 256,
        "BF16 T.gemm with FP32 accumulation, pinned to fragment-output mma.sync lowering and exported-instruction audit.",
        "flops = blocks * inner_iterations * 2 * M * N * K",
        accum_dtype="float32", blocks_per_sm=4,
    ),
    BenchmarkSpec(
        "f16_mma_gemm", "gemm", (96, 80, 48), "float16", 128, 256,
        "FP16 T.gemm with FP32 accumulation, independently timed and exported-instruction audited; never BF16-scaled.",
        "flops = blocks * inner_iterations * 2 * M * N * K",
        accum_dtype="float32", blocks_per_sm=4,
    ),
    BenchmarkSpec(
        "tf32_mma_gemm", "gemm", (96, 80, 48), "tfloat32", 128, 256,
        "TF32 T.gemm using FP32 storage, explicit T.tfloat32/allow_tf32 reference path, independent timing and instruction audit.",
        "flops = blocks * inner_iterations * 2 * M * N * K",
        accum_dtype="float32", blocks_per_sm=4, storage_dtype="float32", allow_tf32=True,
    ),
    BenchmarkSpec(
        "cuda_fp32_fma", "cuda_core", (4,), "float32", 256, 8192,
        "Four independent scalar FP32 T.fma chains per thread; two FLOPs per FMA and no TensorCore path.",
        "flops = blocks * threads * inner_iterations * unroll * 2",
        accum_dtype="float32", unroll=4, blocks_per_sm=4,
    ),
    BenchmarkSpec(
        "sfu_exp2", "sfu", (4,), "float32", 256, 1024,
        "Four independent FP32 exp2 chains per thread with bounded periodic inputs.",
        "operations = blocks * threads * inner_iterations * unroll",
        unroll=4, blocks_per_sm=4,
    ),
    BenchmarkSpec(
        "barrier", "barrier", (1,), "int32", 128, 16384,
        "CTA-wide T.sync_threads round trips with one resident-sized CTA per SM.",
        "barriers_per_cta = inner_iterations; total_barriers = blocks * inner_iterations",
        fixed_blocks=-1,
    ),
    BenchmarkSpec(
        "empty_launch", "launch", (1,), "int32", 32, 1,
        "No-store kernel launch plus a separate back-to-back CUDA-event pair measurement.",
        "launches = 1; bytes = 0; operations = 0",
        fixed_blocks=1,
    ),
)

SPECS = {spec.name: spec for spec in _SPECS}
CORE_BENCHMARKS = tuple(spec.name for spec in _SPECS)


def get_spec(name: str) -> BenchmarkSpec:
    try:
        return SPECS[name]
    except KeyError as error:
        raise ValueError(f"Unknown calibration benchmark {name!r}; choose from {', '.join(CORE_BENCHMARKS)}") from error


def list_benchmarks() -> list[dict[str, Any]]:
    """Return JSON-safe descriptions without importing torch or TileLang."""

    return [get_spec(name).to_dict() for name in CORE_BENCHMARKS]


def _positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def copy_bytes(elements: int, dtype: str = "float32", inner_iterations: int = 1) -> int:
    """Copied payload bytes: one read plus one write per element and iteration."""

    _positive_int(elements, "elements", 0)
    _positive_int(inner_iterations, "inner_iterations")
    return 2 * elements * DTYPE_BYTES[dtype] * inner_iterations


def gemm_flops(m: int, n: int, k: int, blocks: int = 1, inner_iterations: int = 1) -> int:
    """Logical FLOPs; an m16n8k16 warp atom is 4096 FLOPs, never 32*4096."""

    for value, name in ((m, "M"), (n, "N"), (k, "K"), (blocks, "blocks"), (inner_iterations, "inner_iterations")):
        _positive_int(value, name)
    return blocks * inner_iterations * 2 * m * n * k


def shared_access_bytes(blocks: int, threads: int, inner_iterations: int, unroll: int, dtype: str = "float32") -> int:
    for value, name in ((blocks, "blocks"), (threads, "threads"), (inner_iterations, "inner_iterations"), (unroll, "unroll")):
        _positive_int(value, name)
    return blocks * threads * inner_iterations * unroll * DTYPE_BYTES[dtype]


def sfu_exp2_operations(blocks: int, threads: int, inner_iterations: int, unroll: int) -> int:
    for value, name in ((blocks, "blocks"), (threads, "threads"), (inner_iterations, "inner_iterations"), (unroll, "unroll")):
        _positive_int(value, name)
    return blocks * threads * inner_iterations * unroll


def fp32_fma_flops(blocks: int, threads: int, inner_iterations: int, unroll: int) -> int:
    """Two FLOPs per scalar fused multiply-add, with no warp-level multiplier."""

    operations = sfu_exp2_operations(blocks, threads, inner_iterations, unroll)
    return 2 * operations


def barrier_counts(blocks: int, inner_iterations: int) -> dict[str, int]:
    _positive_int(blocks, "blocks")
    _positive_int(inner_iterations, "inner_iterations")
    return {
        "barriers_per_cta": inner_iterations,
        "total_barriers": blocks * inner_iterations,
    }


def resolve_config(name: str, sm_count: int, l2_bytes: int) -> dict[str, Any]:
    """Resolve SM/L2-dependent launch geometry while retaining a JSON-safe spec."""

    spec = get_spec(name)
    _positive_int(sm_count, "sm_count")
    _positive_int(l2_bytes, "l2_bytes", 0)
    config = spec.to_dict()
    config["sm_count"] = sm_count
    config["l2_bytes"] = l2_bytes
    if spec.blocks_per_sm:
        config["blocks"] = sm_count * spec.blocks_per_sm
    elif spec.fixed_blocks == -1:
        config["blocks"] = sm_count
    elif spec.fixed_blocks:
        config["blocks"] = spec.fixed_blocks
    else:
        config["blocks"] = (spec.shape[0] + spec.threads * COPY_ELEMENTS_PER_THREAD - 1) // (spec.threads * COPY_ELEMENTS_PER_THREAD)

    if spec.family == "copy":
        config["elements_per_thread"] = COPY_ELEMENTS_PER_THREAD
        config["vector_width"] = COPY_VECTOR_WIDTH
    if name == "l2_resident_copy" and l2_bytes:
        # Source + destination occupy at most 1/4 of L2.  Keep the default
        # 8 MiB cap and align to the whole CTA copy tile.
        alignment = spec.threads * COPY_ELEMENTS_PER_THREAD
        elements = min(spec.shape[0], l2_bytes // 32)
        if elements < alignment:
            raise ValueError(f"Reported L2 is too small for the calibration copy tile: {l2_bytes} bytes")
        elements = (elements // alignment) * alignment
        config["shape"] = [elements]
        config["blocks"] = (elements + alignment - 1) // alignment
    config["dtype_bytes"] = DTYPE_BYTES[spec.dtype]
    config["estimated_device_memory_bytes"] = estimate_device_memory(config)
    if config["estimated_device_memory_bytes"] >= MAX_DEFAULT_MEMORY_BYTES:
        raise ValueError(f"{name} exceeds the <2 GiB default device-memory budget: {config}")
    return config


def estimate_device_memory(config: Mapping[str, Any]) -> int:
    """Conservative live-tensor estimate for one isolated worker, in bytes."""

    name = config["name"]
    shape = config["shape"]
    blocks = int(config["blocks"])
    dtype_bytes = DTYPE_BYTES[config["dtype"]]
    if name == "dram_copy":
        # A, B, plus a 256 MiB untimed L2 flush buffer.  A is not modified.
        return 2 * shape[0] * dtype_bytes + 256 * MIB
    if name == "l2_resident_copy":
        # A, B, and a small correctness reference for alternating copies.
        return 3 * shape[0] * dtype_bytes
    if name == "shared_read":
        return shape[0] * dtype_bytes + 2 * blocks * config["threads"] * 4
    if name == "shared_write":
        return 2 * shape[0] * dtype_bytes + 2 * blocks * shape[0] * 4
    if name in {"bf16_mma_gemm", "f16_mma_gemm", "tf32_mma_gemm"}:
        m, n, k = shape
        return (m * k + k * n) * dtype_bytes + 3 * blocks * m * n * 4 + m * n * 4
    if name == "cuda_fp32_fma":
        chains = config["unroll"]
        return 6 * blocks * config["threads"] * chains * 4
    if name == "sfu_exp2":
        chains = config["unroll"]
        return 5 * blocks * config["threads"] * chains * 4
    if name == "barrier":
        return blocks * 4
    if name == "empty_launch":
        return 4
    raise ValueError(f"No memory estimate for {name!r}")


def workload_for(config: Mapping[str, Any]) -> dict[str, Any]:
    """Compute auditable work and byte counts from a resolved configuration."""

    name = config["name"]
    shape = config["shape"]
    blocks = int(config["blocks"])
    iterations = int(config["inner_iterations"])
    unroll = int(config["unroll"])
    result: dict[str, Any] = {
        "formula": get_spec(name).formula,
        "bytes": 0,
        "operations": 0,
        "flops": 0,
        "launches": 1,
    }
    if name in {"dram_copy", "l2_resident_copy"}:
        result["bytes"] = copy_bytes(shape[0], config["dtype"], iterations)
        result["operations"] = shape[0] * iterations
    elif name in {"shared_read", "shared_write"}:
        result["bytes"] = shared_access_bytes(blocks, config["threads"], iterations, unroll, config["dtype"])
        result["operations"] = result["bytes"] // DTYPE_BYTES[config["dtype"]]
    elif name in {"bf16_mma_gemm", "f16_mma_gemm", "tf32_mma_gemm"}:
        result["flops"] = gemm_flops(*shape, blocks, iterations)
        result["operations"] = result["flops"]
        # Global I/O is recorded for audit but not mixed into tensor-core FLOPs.
        m, n, k = shape
        result["bytes"] = blocks * (m * k + k * n) * DTYPE_BYTES[config["dtype"]] + blocks * m * n * 4
    elif name == "cuda_fp32_fma":
        result["flops"] = fp32_fma_flops(blocks, config["threads"], iterations, unroll)
        result["operations"] = result["flops"] // 2
        result["bytes"] = 3 * blocks * config["threads"] * unroll * 4
    elif name == "sfu_exp2":
        result["operations"] = sfu_exp2_operations(blocks, config["threads"], iterations, unroll)
        result["bytes"] = 2 * blocks * config["threads"] * unroll * 4
    elif name == "barrier":
        result.update(barrier_counts(blocks, iterations))
        result["operations"] = iterations
    elif name == "empty_launch":
        pass
    else:
        raise ValueError(f"No workload formula for {name!r}")
    for key in ("bytes", "operations", "flops", "launches"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 0:
            raise ValueError(f"Invalid workload {key} for {name}: {result[key]!r}")
    return result


def units_for(name: str) -> dict[str, str]:
    common = {"time": "millisecond", "raw_samples": "millisecond", "bytes": "byte", "shape": "element"}
    specific = {
        "dram_copy": {"work": "copied_element", "rate": "byte/second"},
        "l2_resident_copy": {"work": "copied_element", "rate": "byte/second"},
        "shared_read": {"work": "shared_load", "rate": "byte/second"},
        "shared_write": {"work": "shared_store", "rate": "byte/second"},
        "bf16_mma_gemm": {"work": "floating_point_operation", "rate": "FLOP/second"},
        "f16_mma_gemm": {"work": "floating_point_operation", "rate": "FLOP/second"},
        "tf32_mma_gemm": {"work": "floating_point_operation", "rate": "FLOP/second"},
        "cuda_fp32_fma": {"work": "floating_point_operation", "operation": "fused_multiply_add", "rate": "FLOP/second"},
        "sfu_exp2": {"work": "exp2_operation", "rate": "operation/second"},
        "barrier": {"work": "barrier/CTA", "rate": "nanosecond/(barrier*CTA)"},
        "empty_launch": {"work": "kernel_launch", "rate": "nanosecond"},
    }
    return {**common, **specific[name]}


def validate_resolved_config(config: Mapping[str, Any]) -> None:
    """Fail before compilation if geometry or arithmetic is inconsistent."""

    name = config["name"]
    if config["shape"] != list(get_spec(name).shape) and name != "l2_resident_copy":
        raise ValueError(f"Unexpected shape override for {name}")
    if get_spec(name).family == "copy":
        tile = config["threads"] * config["elements_per_thread"]
        if config["shape"][0] % tile or config["blocks"] != config["shape"][0] // tile:
            raise ValueError(f"Copy shape/grid is not aligned to the {tile}-element CTA tile")
        if config["elements_per_thread"] % config["vector_width"]:
            raise ValueError("Copy elements_per_thread must be divisible by vector_width")
    if name == "tf32_mma_gemm" and (config.get("allow_tf32") is not True or config.get("storage_dtype") != "float32" or config.get("dtype") != "tfloat32"):
        raise ValueError("TF32 calibration requires explicit allow_tf32, FP32 storage, and T.tfloat32 operands")
    workload_for(config)
    estimate = estimate_device_memory(config)
    if estimate != config.get("estimated_device_memory_bytes", estimate):
        raise ValueError("estimated_device_memory_bytes does not match the resolved shape")
    if not math.isfinite(float(estimate)) or estimate >= MAX_DEFAULT_MEMORY_BYTES:
        raise ValueError("Default calibration device memory must be finite and <2 GiB")
