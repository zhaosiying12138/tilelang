"""Analytical whole-kernel candidate costs for SM120 CUDA kernels.

``CudaCostModel.predict`` implements the ``CandidateCostModel`` protocol from
``tilelang.autotuner.guided``: lower finite scores are better and every term is
finite, non-negative, and JSON serializable.  The score and all ``*_ms`` terms
are milliseconds.  Work counters are logical (padded where documented), not
measurements from a compiled kernel.

The module deliberately imports no TileLang, TVM, torch, CUDA, or GPU package.
Calibration rates are device-wide per-second rates; a one-CTA resource time
uses ``rate / SM_count``.  Missing rates never mean free hardware: without
``require_measured`` they use explicit unit placeholders reported in
``fallback_used`` and :meth:`CudaCostModel.model_card`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import copy
import json
import math
import operator as _operator
import os
from pathlib import Path
from typing import Any


HARDWARE_SCHEMA = "tilelang.cuda.hardware-spec"
CALIBRATION_SCHEMA = "tilelang.cuda.kernel-cost-calibration"
MODEL_CARD_SCHEMA = "tilelang.cuda.kernel-cost-model-card"


class CostModelError(ValueError):
    """A model input, plugin result, or prediction is invalid."""


class CostModelCapacityError(CostModelError):
    """A shared/register/thread physical-resource limit prevents a score."""

    # GuidedSearch recognizes this backend-neutral nominal protocol without
    # making this dependency-free module import TileLang or the autotuner.
    _tilelang_cost_model_capacity = True


class CalibrationError(CostModelError):
    """Measured-only construction lacks measured calibration."""


@dataclass(frozen=True)
class PhaseWork:
    """Logical work in one serial source phase, normalized to one CTA.

    Byte counts are traffic, except for ``KernelWork.shared_memory_bytes_per_cta``
    and ``KernelWork.fragment_capacity_bytes_per_cta`` which are capacities.
    Fragment traffic and capacity stay distinct from shared-memory traffic and
    capacity.  ``stages=1`` serializes compute/global/shared resource times;
    two or more stages use the ideal-overlap maximum.  TMA descriptors are
    included in, not added on top of, ``lsu_warp_instructions_per_cta``.
    """

    name: str = "main"
    mma_flops_per_cta: float = 0.0
    cuda_flops_per_cta: float = 0.0
    useful_flops_per_cta: float | None = None
    mma_dtype: str | None = None
    dram_bytes_per_cta: float = 0.0
    l2_bytes_per_cta: float = 0.0
    shared_bytes_per_cta: float = 0.0
    fragment_bytes_per_cta: float = 0.0
    lsu_warp_instructions_per_cta: float = 0.0
    tma_descriptors_per_cta: float = 0.0
    sfu_exp2_operations_per_cta: float = 0.0
    barriers_per_cta: float = 0.0
    stages: int = 1


@dataclass(frozen=True)
class KernelWork:
    """One launch and its serial phases, the public operator-plugin result."""

    name: str
    cta_count: int
    phases: tuple[PhaseWork, ...]
    threads_per_cta: int = 128
    shared_memory_bytes_per_cta: int = 0
    fragment_capacity_bytes_per_cta: int = 0
    registers_per_thread: int | None = None
    launches: int = 1
    fallbacks: tuple[str, ...] = field(default_factory=tuple)


OperatorPlugin = Callable[[Mapping[str, Any], Mapping[str, Any]], KernelWork | Sequence[KernelWork]]


_RATE_UNITS = {
    "clock_hz": "Hz",
    "mma_f16_flops_per_second": "FLOP/s",
    "mma_bf16_flops_per_second": "FLOP/s",
    "mma_tf32_flops_per_second": "FLOP/s",
    "cuda_fp32_flops_per_second": "FLOP/s",
    "dram_bytes_per_second": "B/s",
    "l2_bytes_per_second": "B/s",
    "shared_bytes_per_second": "B/s",
    "lsu_warp_instructions_per_second": "warp-instruction/s",
    "sfu_exp2_operations_per_second": "exp2/s",
    "barrier_ns": "ns",
    "launch_ns": "ns",
}
_TIME_RATES = {"barrier_ns", "launch_ns"}
_MMA_RATES = {
    "f16": "mma_f16_flops_per_second",
    "bf16": "mma_bf16_flops_per_second",
    "tf32": "mma_tf32_flops_per_second",
}


def _load_json(value: Any, description: str) -> tuple[dict[str, Any], str]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value)), "<mapping>"
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise CostModelError(f"{description} JSON must contain an object")
        return data, os.fspath(path)
    raise TypeError(f"{description} must be a mapping or JSON file path")


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CostModelError(f"{name} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise CostModelError(f"{name} must be finite")
    if result < 0 or (positive and result <= 0):
        raise CostModelError(f"{name} must be {'positive' if positive else 'non-negative'}")
    return result


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise CostModelError(f"{name} must be an integer, not bool")
    try:
        result = int(_operator.index(value))
    except TypeError as exc:
        raise CostModelError(f"{name} must be an integer, got {value!r}") from exc
    if result < minimum:
        raise CostModelError(f"{name} must be >= {minimum}, got {result}")
    return result


def _canonical_dtype(value: Any) -> str:
    if not isinstance(value, str):
        raise CostModelError(f"dtype must be a string, got {value!r}")
    key = value.strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "float16": "f16", "half": "f16", "fp16": "f16", "f16": "f16",
        "bfloat16": "bf16", "bf16": "bf16",
        "tensorflow32": "tf32", "tf32": "tf32",
        "float32": "f32", "fp32": "f32", "float": "f32", "f32": "f32",
    }
    if key not in aliases:
        raise CostModelError(f"Unsupported dtype {value!r}")
    return aliases[key]


def _dtype_bytes(dtype: str) -> int:
    return 2 if dtype in {"f16", "bf16"} else 4


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise CostModelError(f"{name} must be bool")
    return value


def _lower_keys(data: Mapping[str, Any], description: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise CostModelError(f"{description} keys must be strings")
        lowered = key.lower()
        if lowered in result:
            raise CostModelError(f"Duplicate case-insensitive {description} key {key!r}")
        result[lowered] = value
    return result


def _take(data: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    present = [name for name in names if name in data]
    if len(present) > 1:
        raise CostModelError(f"Specify only one of {present}")
    return data[present[0]] if present else default


def _shape_int(shape: Mapping[str, Any], *names: str, default: Any = None, minimum: int = 1) -> int:
    value = _take(shape, tuple(name.lower() for name in names), default)
    if value is None:
        raise CostModelError(f"Missing shape dimension {'/'.join(names)}")
    return _integer(value, f"shape.{names[0]}", minimum)


def _config_int(config: Mapping[str, Any], *names: str, default: Any = None, minimum: int = 1) -> int:
    value = _take(config, tuple(name.lower() for name in names), default)
    if value is None:
        raise CostModelError(f"Missing config tile/value {'/'.join(names)}")
    return _integer(value, f"config.{names[0]}", minimum)


def _shape_dtype(shape: Mapping[str, Any], output: bool = False) -> str:
    names = ("output_dtype", "o_dtype") if output else ("dtype", "input_dtype", "qkva_dtype")
    value = _take(shape, names, None)
    if value is None and output:
        return _shape_dtype(shape, False)
    return _canonical_dtype("bf16" if value is None else value)


_RESOURCE_CONFIG_KEYS = {
    "num_stages", "stages", "pipeline", "threads", "num_threads", "block_threads",
    "warps", "num_warps", "registers_per_thread", "num_regs", "registers",
    "copy_kind", "copy_mode", "tma", "use_tma", "tma_store", "l2_resident",
    "output_in_shared", "enable_rasterization", "rasterization",
}
_TILE_CONFIG_KEYS = _RESOURCE_CONFIG_KEYS | {
    "block_m", "block_n", "block_k", "block_pixels", "block_channels", "block_reduce",
    "tile_m", "tile_n", "tile_k", "bm", "bn", "bk", "use_mma",
}
_FA_CONFIG_KEYS = _RESOURCE_CONFIG_KEYS | {
    "block_m", "block_n", "tile_m", "tile_n", "bm", "bn", "store_lse", "causal",
}


def _check_config(config: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise CostModelError(f"Unknown config fields for analytical model: {unknown}")


def _threads(config: Mapping[str, Any], default: int = 128) -> int:
    thread_value = _take(config, ("threads", "num_threads", "block_threads"), None)
    warp_value = _take(config, ("warps", "num_warps"), None)
    if thread_value is not None and warp_value is not None:
        raise CostModelError("Specify threads or warps, not both")
    if warp_value is not None:
        return 32 * _integer(warp_value, "config.num_warps")
    return default if thread_value is None else _integer(thread_value, "config.threads")


def _registers(config: Mapping[str, Any]) -> int | None:
    value = _take(config, ("registers_per_thread", "num_regs", "registers"), None)
    return None if value is None else _integer(value, "config.registers_per_thread")


def _stages(config: Mapping[str, Any], default: int = 1) -> int:
    return _config_int(config, "num_stages", "stages", "pipeline", default=default, minimum=0)


def _copy_kind(config: Mapping[str, Any], default: str = "cp_async") -> str:
    raw = _take(config, ("copy_kind", "copy_mode"), None)
    tma = _take(config, ("tma", "use_tma"), None)
    if raw is not None and tma is not None:
        raise CostModelError("Specify copy_kind or tma, not both")
    if tma is not None:
        if isinstance(tma, str):
            raw = "tma" if tma.lower() in {"on", "true", "tma"} else "cp_async"
        else:
            raw = "tma" if _bool(tma, "config.tma") else "cp_async"
    if raw is None:
        raw = default
    if not isinstance(raw, str):
        raise CostModelError("copy_kind must be 'cp_async' or 'tma'")
    kind = raw.strip().lower().replace(".", "_").replace("-", "_")
    if kind in {"cpasync", "cp_async", "async_copy"}:
        return "cp_async"
    if kind == "tma":
        return "tma"
    raise CostModelError(f"Unsupported copy_kind {raw!r}")


def _traffic(config: Mapping[str, Any], load: float, store: float, compulsory: float) -> tuple[float, float]:
    resident = _bool(config.get("l2_resident", False), "config.l2_resident")
    l2 = load + store
    dram = min(l2, compulsory) if resident else l2
    return dram, l2


def _lsu(
    config: Mapping[str, Any],
    *,
    load: float,
    store: float,
    shared: float,
    kind: str,
    tma_descriptors: int = 0,
    tma_store_descriptors: int = 0,
    scalar_load_bytes: float = 0,
) -> tuple[float, float]:
    # A full cp.async/vector warp instruction is modeled as 32 lanes * 16 B.
    # TMA changes issue count, not logical bytes or DRAM/L2/shared bandwidth.
    vector_bytes = 32 * 16
    shared_issue = math.ceil(shared / vector_bytes) if shared else 0
    if kind in {"cp_async", "vector"}:
        load_issue = math.ceil(load / vector_bytes) if load else 0
        store_issue = math.ceil(store / vector_bytes) if store else 0
        descriptors = 0
    elif kind == "tma":
        load_issue = tma_descriptors
        if scalar_load_bytes:
            load_issue += math.ceil(scalar_load_bytes / (32 * 4))
        use_tma_store = _bool(config.get("tma_store", False), "config.tma_store")
        store_issue = tma_store_descriptors if use_tma_store else (math.ceil(store / vector_bytes) if store else 0)
        descriptors = tma_descriptors + (tma_store_descriptors if use_tma_store else 0)
    else:
        raise CostModelError(f"Unsupported plugin copy kind {kind!r}")
    return float(load_issue + store_issue + shared_issue), float(descriptors)


def _phase(
    config: Mapping[str, Any],
    *,
    name: str,
    mma: float = 0,
    cuda: float = 0,
    useful: float | None = None,
    dtype: str | None = None,
    load: float = 0,
    store: float = 0,
    compulsory: float = 0,
    shared: float = 0,
    fragment: float = 0,
    sfu: float = 0,
    barriers: float = 0,
    stages: int = 1,
    kind: str = "cp_async",
    tma_descriptors: int = 0,
    tma_store_descriptors: int = 0,
    scalar_load_bytes: float = 0,
) -> PhaseWork:
    dram, l2 = _traffic(config, load, store, compulsory)
    lsu, descriptors = _lsu(
        config, load=load, store=store, shared=shared, kind=kind,
        tma_descriptors=tma_descriptors, tma_store_descriptors=tma_store_descriptors,
        scalar_load_bytes=scalar_load_bytes,
    )
    return PhaseWork(
        name=name, mma_flops_per_cta=mma, cuda_flops_per_cta=cuda,
        useful_flops_per_cta=useful, mma_dtype=dtype,
        dram_bytes_per_cta=dram, l2_bytes_per_cta=l2, shared_bytes_per_cta=shared,
        fragment_bytes_per_cta=fragment, lsu_warp_instructions_per_cta=lsu, tma_descriptors_per_cta=descriptors,
        sfu_exp2_operations_per_cta=sfu, barriers_per_cta=barriers, stages=stages,
    )


def _gemm_bytes(specs: Sequence[tuple[int, int, int]], element_bytes: int) -> float:
    # One logical read of each shared operand per GEMM.  Reuse across different
    # GEMMs is visible because every tuple is charged independently.
    return float(sum((m * k + k * n) * element_bytes for m, n, k in specs))


def _tile_config(config: Mapping[str, Any]) -> tuple[int, int, int]:
    bm = _config_int(config, "block_M", "block_pixels", "tile_M", "bm")
    bn = _config_int(config, "block_N", "block_channels", "tile_N", "bn")
    bk = _config_int(config, "block_K", "block_reduce", "tile_K", "bk")
    return bm, bn, bk


def _gemm_like(
    name: str,
    shape: Mapping[str, Any],
    config: Mapping[str, Any],
    m: int,
    n: int,
    k: int,
    compulsory_total: float | None = None,
) -> KernelWork:
    bm, bn, bk = _tile_config(config)
    stages = max(_stages(config), 1)
    dtype = _shape_dtype(shape)
    output_dtype = _shape_dtype(shape, output=True)
    if dtype == "f32":
        raise CostModelError("Use dtype=tf32 for Tensor Core GEMM or use_mma=False for FP32 CUDA cores")
    element_bytes = _dtype_bytes(dtype)
    output_bytes = _dtype_bytes(output_dtype)
    tiles_m = (m + bm - 1) // bm
    tiles_n = (n + bn - 1) // bn
    tiles_k = (k + bk - 1) // bk
    ctas = tiles_m * tiles_n
    mma = 2 * bm * bn * bk * tiles_k
    load = (bm * bk * tiles_k + bk * tiles_k * bn) * element_bytes
    store = bm * bn * output_bytes
    useful_total = 2 * m * n * k
    if compulsory_total is None:
        compulsory_total = (m * k + k * n) * element_bytes + m * n * output_bytes
    else:
        compulsory_total = _finite_number(compulsory_total, f"{name} compulsory bytes")
    output_shared = _bool(config.get("output_in_shared", False), "config.output_in_shared")
    shared = 2 * load + (2 * store if output_shared else 0)
    use_mma = _bool(config.get("use_mma", True), "config.use_mma")
    kind = _copy_kind(config)
    tma_store = _bool(config.get("tma_store", False), "config.tma_store")
    phase = _phase(
        config, name="main", mma=mma if use_mma else 0, cuda=0 if use_mma else mma,
        useful=useful_total / ctas, dtype=dtype if use_mma else None,
        load=load, store=store, compulsory=compulsory_total / ctas, shared=shared,
        barriers=2 * tiles_k + (1 if output_shared else 0), stages=stages, kind=kind,
        tma_descriptors=2 * tiles_k, tma_store_descriptors=1 if tma_store and store else 0,
    )
    capacity = (bm * bk + bk * bn) * element_bytes * stages + (bm * bn * output_bytes if output_shared else 0)
    rasterization = _bool(config.get("enable_rasterization", config.get("rasterization", False)), "config.enable_rasterization")
    fallbacks = ("gemm.rasterization_l2_locality_unmodeled",) if rasterization else ()
    return KernelWork(
        name=name, cta_count=ctas, phases=(phase,), threads_per_cta=_threads(config),
        shared_memory_bytes_per_cta=capacity, registers_per_thread=_registers(config),
        fallbacks=fallbacks,
    )


def _gemm_plugin(shape: Mapping[str, Any], config: Mapping[str, Any]) -> KernelWork:
    _check_config(config, _TILE_CONFIG_KEYS)
    m = _shape_int(shape, "M")
    n = _shape_int(shape, "N")
    k = _shape_int(shape, "K")
    return _gemm_like("gemm", shape, config, m, n, k)


def _conv_plugin(shape: Mapping[str, Any], config: Mapping[str, Any]) -> KernelWork:
    _check_config(config, _TILE_CONFIG_KEYS)
    batch = _shape_int(shape, "N", "batch", "B")
    height = _shape_int(shape, "H")
    width = _shape_int(shape, "W")
    channels = _shape_int(shape, "C")
    outputs = _shape_int(shape, "K", "output_channels")
    kernel_h = _shape_int(shape, "R", "kernel_h")
    kernel_w = _shape_int(shape, "S", "kernel_w")
    stride_h = _shape_int(shape, "stride_h", default=1)
    stride_w = _shape_int(shape, "stride_w", default=stride_h)
    dilation_h = _shape_int(shape, "dilation_h", default=1)
    dilation_w = _shape_int(shape, "dilation_w", default=dilation_h)
    pad_h = _shape_int(shape, "pad_h", "padding", default=0, minimum=0)
    pad_w = _shape_int(shape, "pad_w", "padding", default=pad_h, minimum=0)
    effective_h = dilation_h * (kernel_h - 1) + 1
    effective_w = dilation_w * (kernel_w - 1) + 1
    numerator_h = height + 2 * pad_h - effective_h
    numerator_w = width + 2 * pad_w - effective_w
    if numerator_h < 0 or numerator_w < 0:
        raise CostModelError("Convolution kernel is larger than the padded input")
    out_h = numerator_h // stride_h + 1
    out_w = numerator_w // stride_w + 1
    m = batch * out_h * out_w
    reduction = channels * kernel_h * kernel_w
    dtype = _shape_dtype(shape)
    output_dtype = _shape_dtype(shape, output=True)
    unique = (
        batch * height * width * channels * _dtype_bytes(dtype)
        + outputs * reduction * _dtype_bytes(dtype)
        + m * outputs * _dtype_bytes(output_dtype)
    )
    # NHWC's unique footprint differs from generic GEMM; padded im2col/tile
    # traffic and FLOPs still use the shared tile formula.
    return _gemm_like("conv2d_nhwc", shape, config, m, outputs, reduction, compulsory_total=unique)


def _fa_plugin(shape: Mapping[str, Any], config: Mapping[str, Any]) -> KernelWork:
    _check_config(config, _FA_CONFIG_KEYS)
    batch = _shape_int(shape, "B", "batch")
    heads = _shape_int(shape, "H", "heads")
    seq_q = _shape_int(shape, "S", "T", "seq_len", "seqlen")
    seq_kv = _shape_int(shape, "S_KV", "KV", "kv_len", default=seq_q)
    depth = _shape_int(shape, "D", "head_dim")
    bm = _config_int(config, "block_M", "tile_M", "bm")
    bn = _config_int(config, "block_N", "tile_N", "bn")
    stages = _stages(config)
    dtype = _shape_dtype(shape)
    output_dtype = _shape_dtype(shape, output=True)
    if dtype == "f32":
        raise CostModelError("flash_attention_fwd requires f16/bf16/tf32 storage")
    element_bytes = _dtype_bytes(dtype)
    output_bytes = _dtype_bytes(output_dtype)
    q_tiles = (seq_q + bm - 1) // bm
    kv_tiles = (seq_kv + bn - 1) // bn
    padded_kv = kv_tiles * bn
    ctas = batch * heads * q_tiles
    causal = _bool(config.get("causal", shape.get("causal", False)), "config.causal")
    if causal and seq_q != seq_kv:
        raise CostModelError("The causal upper-bound model currently requires seq_q == seq_kv")
    mma = 4 * bm * padded_kv * depth
    cuda = kv_tiles * (4 * bm * bn + bm * depth) + bm * depth
    if causal:
        useful_total = 2 * batch * heads * depth * seq_q * (seq_q + 1)
    else:
        useful_total = 4 * batch * heads * seq_q * seq_kv * depth
    store_lse = _bool(config.get("store_lse", False), "config.store_lse")
    load = (bm * depth + 2 * padded_kv * depth) * element_bytes
    store = bm * depth * output_bytes + (bm * 4 if store_lse else 0)
    compulsory = (
        bm * depth * element_bytes
        + 2 * seq_kv * depth * element_bytes / q_tiles
        + store
    )
    output_shared = _bool(config.get("output_in_shared", False), "config.output_in_shared")
    # Global->shared writes plus Q reads per KV step and K/V reads per step.
    shared = load + bm * depth * element_bytes * kv_tiles + 2 * padded_kv * depth * element_bytes
    if output_shared:
        shared += 2 * store
    kind = _copy_kind(config)
    tma_store = _bool(config.get("tma_store", False), "config.tma_store")
    phase = _phase(
        config, name="main", mma=mma, cuda=cuda, useful=useful_total / ctas,
        dtype=dtype, load=load, store=store, compulsory=compulsory, shared=shared,
        sfu=bm * padded_kv, barriers=2 * kv_tiles + 1, stages=stages, kind=kind,
        tma_descriptors=1 + 2 * kv_tiles,
        tma_store_descriptors=1 if tma_store and store else 0,
    )
    capacity = (
        bm * depth * element_bytes + stages * 2 * bn * depth * element_bytes
        + (bm * depth * output_bytes if output_shared else 0)
    )
    return KernelWork(
        name="flash_attention_fwd", cta_count=ctas, phases=(phase,),
        threads_per_cta=_threads(config), shared_memory_bytes_per_cta=capacity,
        registers_per_thread=_registers(config),
        fallbacks=("flash_attention_fwd.causal_upper_bound",) if causal else (),
    )


def _gdn_shape(shape: Mapping[str, Any]) -> dict[str, Any]:
    batch = _shape_int(shape, "B", "batch")
    tokens = _shape_int(shape, "T", "S", "num_tokens", "seq_len")
    heads = _shape_int(shape, "H", "num_heads")
    group_heads = _shape_int(shape, "Hg", "HQ", "num_q_heads", default=heads)
    depth_k = _shape_int(shape, "K", "DK", default=128)
    depth_v = _shape_int(shape, "DV", "V", default=128)
    chunk = _shape_int(shape, "chunk_size", "block_S", default=32)
    if chunk != 32 or depth_k != 128 or depth_v != 128:
        raise CostModelError("GDN analytical kernels require chunk_size=32 and K=DV=128")
    if heads % group_heads:
        raise CostModelError("GDN H must be divisible by Hg")
    dtype = _shape_dtype(shape)
    if dtype not in {"f16", "bf16"}:
        raise CostModelError("The SM120 GDN source model supports f16/bf16 qkva storage")
    return {
        "B": batch, "T": tokens, "H": heads, "Hg": group_heads, "K": depth_k,
        "DV": depth_v, "S": chunk, "C": (tokens + chunk - 1) // chunk,
        "dtype": dtype, "e": _dtype_bytes(dtype),
        "out_e": _dtype_bytes(_shape_dtype(shape, output=True)),
        "use_initial_state": _bool(shape.get("use_initial_state", False), "shape.use_initial_state"),
        "output_final_state": _bool(shape.get("output_final_state", True), "shape.output_final_state"),
        "output_h": _bool(shape.get("output_h", False), "shape.output_h"),
        "store_o": _bool(shape.get("store_o", True), "shape.store_o"),
        "use_dht": _bool(shape.get("use_dht", False), "shape.use_dht"),
        "calc_mt": _bool(shape.get("calc_mt", False), "shape.calc_mt"),
    }


def _gdn_config(config: Mapping[str, Any], backward: bool, tokens: int | None = None) -> tuple[int, int, int, int | None]:
    allowed = {
        "block_dv", "prepare_h_num_stages", "group_reduce_block_size",
        "registers_per_thread", "num_regs", "registers", "l2_resident",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise CostModelError(f"Config fields not applicable to the fixed GDN kernels: {unknown}")
    block_dv = _config_int(config, "block_DV", default=128)
    if block_dv > 128 or block_dv % 16 or 128 % block_dv:
        raise CostModelError("block_DV must be a divisor of 128 and a multiple of 16")
    if backward and block_dv != 128:
        raise CostModelError("The SM120 fused backward kernel has no block_DV knob; use 128")
    if "prepare_h_num_stages" in config and not backward:
        raise CostModelError("prepare_h_num_stages applies to gdn_bwd, not gdn_fwd")
    if "group_reduce_block_size" in config and not backward:
        raise CostModelError("group_reduce_block_size applies to gdn_bwd, not gdn_fwd")
    stages = _config_int(config, "prepare_h_num_stages", default=2) if backward else 2
    if backward and stages not in {1, 2, 3}:
        raise CostModelError("prepare_h_num_stages must be in the GDN backward space {1, 2, 3}")
    group_block = _config_int(config, "group_reduce_block_size", default=16) if backward else 16
    if backward and group_block not in {8, 16, 32}:
        raise CostModelError("group_reduce_block_size must be in the GDN backward space {8, 16, 32}")
    if backward and group_block != 16:
        if tokens is None:
            raise CostModelError("GDN backward token count is required for group_reduce_block_size validation")
        if tokens % group_block:
            raise CostModelError(
                f"Non-default group_reduce_block_size={group_block} requires T divisible by the block; got T={tokens}"
            )
    return block_dv, stages, group_block, _registers(config)


def _kkt_kernel(g: Mapping[str, Any], config: Mapping[str, Any], registers: int | None) -> KernelWork:
    s, k, e = g["S"], g["K"], g["e"]
    ctas = g["B"] * g["H"] * g["C"]
    solve_cuda = 2 * 16 * 2 * sum(range(1, 16)) + 2 * (2 * 16**3)
    kkt_shared = 3 * s * k * e + 2 * s * 4 + s * s * 4
    kkt = _phase(
        config, name="kkt", mma=2 * s * s * k, cuda=2 * s * s, dtype=g["dtype"],
        load=s * k * e + s * 4, compulsory=s * k * e * g["Hg"] / g["H"] + s * 4,
        shared=kkt_shared, barriers=1, stages=1, kind="cp_async",
    )
    solve_shared = solve_cuda * 4 + s * s * 4
    solve = _phase(
        config, name="solve", cuda=solve_cuda, store=s * s * e, compulsory=s * s * e,
        shared=solve_shared, barriers=3, stages=1, kind="cp_async",
    )
    capacity = s * k * e + s * 4 + (2 * 17 * 16 + 17 * 16) * 4 + s * s * 4
    return KernelWork(
        "gdn_kkt_solve", ctas, (kkt, solve), threads_per_cta=128,
        shared_memory_bytes_per_cta=capacity, registers_per_thread=registers,
    )


def _gdn_fwd_plugin(shape: Mapping[str, Any], config: Mapping[str, Any]) -> Sequence[KernelWork]:
    g = _gdn_shape(shape)
    block_dv, _, _, registers = _gdn_config(config, backward=False)
    s, k, dv, e, c = g["S"], g["K"], g["DV"], g["e"], g["C"]
    b = block_dv
    vt = dv // b
    ctas = g["B"] * g["H"] * vt
    h_ratio = g["Hg"] / (g["H"] * vt)

    wy_load = c * (s * k * e + s * b * e + 2 * s * 4)
    wy_comp = c * (s * k * e * h_ratio + s * b * e + 2 * s * 4 / vt)
    wy_shared = (
        wy_load
        + c * _gemm_bytes(((s, b, k), (s, b, s)), e)
        + c * (3 * s * b * e + 2 * s * 4)
    )
    wy = _phase(
        config, name="wy", mma=c * (2 * s * k * b + 2 * s * s * b),
        cuda=c * 3 * s * b, dtype=g["dtype"], load=wy_load, compulsory=wy_comp,
        shared=wy_shared, sfu=c * 2 * s, barriers=c * 2, stages=2, kind="tma",
        tma_descriptors=c * 2, scalar_load_bytes=c * 2 * s * 4,
    )

    recurrence_load = k * b * 4 if g["use_initial_state"] else 0
    recurrence_store = (c * k * b * e if g["output_h"] else 0) + (k * b * 4 if g["output_final_state"] else 0)
    recurrence_shared = (
        c * _gemm_bytes(((b, k, s),), e) + c * k * b * e
        + (c * k * b * e if g["output_h"] else 0)
    )
    recurrence = _phase(
        config, name="recurrence", mma=c * 2 * s * k * b, cuda=c * k * b,
        dtype=g["dtype"], load=recurrence_load, store=recurrence_store,
        compulsory=recurrence_load + recurrence_store, shared=recurrence_shared,
        barriers=c * 2, stages=1, kind="tma", scalar_load_bytes=recurrence_load,
    )

    output_load = c * (s * k * e + s * s * e)
    output_store = c * s * b * g["out_e"] if g["store_o"] else 0
    output_comp = c * (s * k * e * h_ratio + s * s * e / vt) + output_store
    output_shared = (
        output_load
        + c * _gemm_bytes(((s, s, k), (s, b, k), (s, b, s)), e)
        + c * s * s * e + (2 * output_store if g["store_o"] else 0)
    )
    output = _phase(
        config, name="output",
        mma=c * (2 * s * s * k + 2 * s * k * b + 2 * s * s * b),
        cuda=c * (5 * s * s + 2 * s * b), dtype=g["dtype"], load=output_load,
        store=output_store, compulsory=output_comp, shared=output_shared,
        sfu=c * (s * (s + 1) // 2), barriers=c * 2, stages=2, kind="tma",
        tma_descriptors=c * 2,
    )
    # Source-lifetime model: o/p/vn alias consumed current q/a buffers; vd and h
    # remain explicit.  This is an estimate, not a ptxas allocation report.
    capacity = (
        4 * s * k * e + 2 * s * b * e + 2 * s * s * e + 4 * s * 4
        + k * b * e + s * b * e + 2 * s * 4
    )
    fused = KernelWork(
        "gdn_fused_fwd", ctas, (wy, recurrence, output), threads_per_cta=512,
        shared_memory_bytes_per_cta=capacity, registers_per_thread=registers,
        fallbacks=("gdn_fwd.shared_alias_estimate", "gdn.fixed_role_phase_model"),
    )
    return (_kkt_kernel(g, config, registers), fused)


def _prepare_h_kernel(g: Mapping[str, Any], config: Mapping[str, Any], stages: int, registers: int | None) -> KernelWork:
    s, k, dv, e, c = g["S"], g["K"], g["DV"], g["e"], g["C"]
    ctas = g["B"] * g["H"]
    output_h = _bool(g.get("prepare_output_h", True), "shape.output_h")
    calc_mt = g["calc_mt"]
    wy_load = c * (s * k * e + s * dv * e + s * s * e + 2 * s * 4)
    wy_comp = c * (s * k * e * g["Hg"] / g["H"] + s * dv * e + s * s * e + 2 * s * 4)
    wy_shared = (
        wy_load + c * _gemm_bytes(((s, k, s), (s, dv, k)), e)
        + c * (s * k * e + s * dv * e + s * 4)
    )
    wy = _phase(
        config, name="wy", mma=c * (2 * s * s * k + 2 * s * k * dv),
        cuda=c * (s * k + 2 * s * dv), dtype=g["dtype"], load=wy_load,
        compulsory=wy_comp, shared=wy_shared, sfu=c * (s + 1), barriers=c * 2,
        stages=stages, kind="tma", tma_descriptors=c * 3, scalar_load_bytes=c * 2 * s * 4,
    )
    initial = k * dv * 4 if g["use_initial_state"] else 0
    final = k * dv * 4 if g["prepare_output_final_state"] else 0
    h_store = c * k * dv * e if output_h else 0
    mt_store = k * k * 4
    recurrence_mma = c * 2 * s * k * dv + (c * 4 * s * k * k if calc_mt else 0)
    recurrence_specs = [(dv, k, s)]
    if calc_mt:
        recurrence_specs.extend(((s, k // 2, k), (s, k // 2, k), (k, k // 2, s), (k, k // 2, s)))
    recurrence_shared = (
        c * _gemm_bytes(recurrence_specs, e) + c * k * dv * e + h_store + mt_store
    )
    recurrence = _phase(
        config, name="recurrence", mma=recurrence_mma,
        cuda=c * k * dv + (k * k if calc_mt else 0), dtype=g["dtype"],
        load=initial, store=h_store + mt_store + final,
        compulsory=initial + h_store + mt_store + final, shared=recurrence_shared,
        sfu=c + (2 if calc_mt else 0), barriers=c * 2, stages=stages, kind="tma",
        scalar_load_bytes=initial,
    )
    # G5 generated-code measurement for the fixed SM120 prepare_h kernel;
    # phase work/traffic above remains analytical and is unchanged.
    return KernelWork(
        "gdn_prepare_h", ctas, (wy, recurrence), threads_per_cta=512,
        shared_memory_bytes_per_cta=57_344 + 18_432 * stages,
        registers_per_thread=registers, fallbacks=("gdn.fixed_role_phase_model",),
    )


def _gqa_reduce_kernel(
    g: Mapping[str, Any], config: Mapping[str, Any], block: int, registers: int | None
) -> KernelWork | None:
    """Model vendor group_reduce_vector for both dq and dk when Hg < H."""
    if g["Hg"] == g["H"]:
        return None
    group = g["H"] // g["Hg"]
    k, e, out_e = g["K"], g["e"], g["out_e"]
    ctas = g["B"] * g["Hg"] * ((g["T"] + block - 1) // block)
    load = group * block * k * e
    store = block * k * out_e
    useful_total = g["B"] * g["T"] * g["H"] * k
    compulsory = (
        g["B"] * g["T"] * g["H"] * k * e
        + g["B"] * g["T"] * g["Hg"] * k * out_e
    ) / ctas
    # Per group: one FP32 buffer-fragment write plus result read/read/write.
    # The initial result clear and final result-fragment read each add one tile.
    fragment = (4 * group + 2) * block * k * 4
    phase = _phase(
        config, name="group_reduce", cuda=group * block * k,
        useful=useful_total / ctas, load=load, store=store, compulsory=compulsory,
        fragment=fragment, stages=1, kind="vector",
    )
    fallbacks = [
        "gdn.fragment_traffic_not_timed",
        "gdn_gqa_reduce.fragment_capacity_lower_bound",
    ]
    if g["T"] % block:
        fallbacks.append("gdn_gqa_reduce.padded_tail")
    return KernelWork(
        "gdn_gqa_reduce", ctas, (phase,), threads_per_cta=128,
        shared_memory_bytes_per_cta=0,
        fragment_capacity_bytes_per_cta=2 * block * k * 4,
        registers_per_thread=registers, launches=2, fallbacks=tuple(fallbacks),
    )


def _reverse_dg_cumsum_kernel(
    g: Mapping[str, Any], config: Mapping[str, Any], registers: int | None
) -> KernelWork:
    """Model the dense public backward reverse=True dg chunk-local cumsum."""
    s, heads = g["S"], g["H"]
    ctas = g["B"] * g["C"]  # H is inside one CTA, not part of this grid.
    global_bytes = s * heads * 4  # public backward asserts dg.dtype == float32
    useful_total = g["B"] * heads * (g["T"] - g["C"])
    compulsory = 2 * g["B"] * g["T"] * heads * 4 / ctas
    phase = _phase(
        config, name="reverse_cumsum", cuda=heads * (s - 1),
        useful=useful_total / ctas, load=global_bytes, store=global_bytes,
        compulsory=compulsory, shared=4 * global_bytes, fragment=8 * global_bytes,
        barriers=2, stages=1, kind="vector",
    )
    fallbacks = [
        "gdn.fragment_traffic_not_timed",
        "gdn.reverse_cumsum.logical_scan_traffic",
        "gdn.reverse_cumsum.zero_initialization_unmodeled",
    ]
    if g["T"] % s:
        fallbacks.append("gdn.reverse_cumsum.padded_tail")
    return KernelWork(
        "gdn_reverse_dg_cumsum", ctas, (phase,), threads_per_cta=128,
        shared_memory_bytes_per_cta=s * (heads + 1) * 4,
        fragment_capacity_bytes_per_cta=2 * global_bytes,
        registers_per_thread=registers, fallbacks=tuple(fallbacks),
    )


def _gdn_bwd_plugin(shape: Mapping[str, Any], config: Mapping[str, Any]) -> Sequence[KernelWork]:
    g = _gdn_shape(shape)
    g["prepare_output_h"] = True
    g["prepare_output_final_state"] = False
    _, stages, group_block, registers = _gdn_config(config, backward=True, tokens=g["T"])
    s, k, dv, e, c = g["S"], g["K"], g["DV"], g["e"], g["C"]
    ctas = g["B"] * g["H"]

    wy_specs = ((s, s, k), (s, dv, k), (s, dv, s))
    dhu_specs = ((k, dv, s), (k, dv, s), (s, k, dv))
    dqk_specs = (
        (s, dv, s), (s, dv, s), (s, k, dv), (s, k, dv),
        (s, k, s), (s, k, s), (s, s, dv), (s, s, dv),
        (s, k, dv), (s, k, s), (s, s, s), (s, s, s), (s, s, k),
    )
    wy_load = c * (2 * s * k * e + s * s * e + s * dv * e + 2 * s * 4)
    wy_comp = c * (2 * s * k * e * g["Hg"] / g["H"] + s * s * e + s * dv * e + 2 * s * 4)
    wy_shared = wy_load + c * _gemm_bytes(wy_specs, e) + c * (3 * s * s + 2 * s * k + 2 * s * dv) * e
    wy = _phase(
        config, name="wy", mma=c * sum(2 * m * n * red for m, n, red in wy_specs),
        cuda=c * (6 * s * s + 2 * s * dv), dtype=g["dtype"], load=wy_load,
        compulsory=wy_comp, shared=wy_shared, sfu=c * (s * (s + 1) // 2 + 2 * s),
        barriers=c * 5, stages=2, kind="tma", tma_descriptors=c * 3,
        scalar_load_bytes=c * (s * dv * e + 2 * s * 4),
    )

    dhu_load = c * k * dv * e + (k * dv * 4 if g["use_dht"] else 0)
    dhu_store = k * dv * 4 if g["use_dht"] else 0
    dhu_shared = c * k * dv * e + c * _gemm_bytes(dhu_specs, e) + c * (2 * k * dv + 2 * s * dv) * e
    dhu = _phase(
        config, name="dhu", mma=c * sum(2 * m * n * red for m, n, red in dhu_specs),
        cuda=c * (3 * k * dv + 2 * s * dv), dtype=g["dtype"], load=dhu_load,
        store=dhu_store, compulsory=dhu_load + dhu_store, shared=dhu_shared,
        barriers=c * 5, stages=2, kind="tma", tma_descriptors=c * 2,
        scalar_load_bytes=k * dv * 4 if g["use_dht"] else 0,
    )

    dqk_load = c * s * dv * e
    dqk_store = c * (2 * s * k * e + s * dv * e + 2 * s * 4)
    dqk_shared = dqk_load + c * _gemm_bytes(dqk_specs, e) + c * (4 * s * k + 3 * s * dv + 4 * s * s) * e
    dqk = _phase(
        config, name="dqkwg", mma=c * sum(2 * m * n * red for m, n, red in dqk_specs),
        cuda=c * (4 * s * dv + 6 * s * k + 17 * s * s), dtype=g["dtype"],
        load=dqk_load, store=dqk_store, compulsory=dqk_load + dqk_store,
        shared=dqk_shared, barriers=c * 8, stages=2, kind="tma", tma_descriptors=c,
    )
    capacity = (
        s * dv * g["out_e"] + s * k * e + s * s * e + (k // 2) * dv * e
        + 4 * s * 4 + s * k * e + 2 * s * 4 + 3 * s * s * e
        + 3 * s * k * e + (k // 2) * dv * e
    )
    fused = KernelWork(
        "gdn_fused_bwd", ctas, (wy, dhu, dqk), threads_per_cta=512,
        shared_memory_bytes_per_cta=capacity, registers_per_thread=registers,
        fallbacks=("gdn.fixed_role_phase_model",),
    )
    works = [_prepare_h_kernel(g, config, stages, registers), fused]
    reduce_work = _gqa_reduce_kernel(g, config, group_block, registers)
    if reduce_work is not None:
        works.append(reduce_work)
    works.append(_reverse_dg_cumsum_kernel(g, config, registers))
    return tuple(works)


_BUILTIN_PLUGINS: dict[str, OperatorPlugin] = {
    "gemm": _gemm_plugin,
    "conv2d_nhwc": _conv_plugin,
    "flash_attention_fwd": _fa_plugin,
    "gdn_fwd": _gdn_fwd_plugin,
    "gdn_bwd": _gdn_bwd_plugin,
}


class _Calibration:
    def __init__(self, data: Mapping[str, Any], require_measured: bool, source: str):
        raw = dict(data)
        if "calibration" in raw and isinstance(raw["calibration"], Mapping):
            outer = raw
            raw = dict(raw["calibration"])
            raw.setdefault("schema", outer.get("calibration_schema"))
        self.source = source
        self.schema = raw.get("schema")
        schema_problem = self.schema != CALIBRATION_SCHEMA
        self.status = raw.get("status", "not_performed")
        rates = raw.get("rates", raw)
        if not isinstance(rates, Mapping):
            raise CalibrationError("calibration.rates must be an object")
        field_sources = raw.get("field_sources", {})
        if not isinstance(field_sources, Mapping):
            raise CalibrationError("calibration.field_sources must be an object")
        self.values: dict[str, float | None] = {}
        self.sources: dict[str, str] = {}
        fallbacks: list[str] = []
        if schema_problem:
            fallbacks.append("calibration.schema:missing-or-unsupported")
        if require_measured and self.status != "measured":
            raise CalibrationError(
                f"Measured calibration required, but calibration.status={self.status!r}; "
                "hardware attributes and status=not_performed are not calibration"
            )
        for name, unit in _RATE_UNITS.items():
            entry = rates.get(name)
            entry_source = None
            if isinstance(entry, Mapping):
                unknown = set(entry) - {"value", "unit", "source"}
                if unknown:
                    raise CalibrationError(f"Unknown calibration fields for {name}: {sorted(unknown)}")
                if entry.get("unit") != unit:
                    raise CalibrationError(f"calibration.{name}.unit must be {unit!r}")
                entry_source = entry.get("source")
                entry = entry.get("value")
            if entry is None:
                self.values[name] = None
                self.sources[name] = "missing"
                fallbacks.append(f"calibration.{name}:unit-placeholder")
                continue
            try:
                value = _finite_number(entry, f"calibration.{name}", positive=True)
            except CostModelError as exc:
                raise CalibrationError(str(exc)) from exc
            provenance = entry_source or field_sources.get(f"rates.{name}", field_sources.get(name))
            if provenance is None:
                provenance = "measured" if self.status == "measured" else "provided"
            if not isinstance(provenance, str):
                raise CalibrationError(f"calibration source for {name} must be a string")
            self.values[name] = value
            self.sources[name] = provenance
            if provenance != "measured":
                fallbacks.append(f"calibration.{name}:{provenance}")
            elif self.status != "measured":
                fallbacks.append(f"calibration.{name}:status={self.status}")
        if require_measured and fallbacks:
            raise CalibrationError("Measured calibration incomplete: " + ", ".join(fallbacks))
        self.fallback_used = tuple(sorted(fallbacks))

    def rate(self, name: str) -> float:
        value = self.values[name]
        if value is not None:
            return value
        # Explicit non-free placeholders.  They are deliberately not hardware
        # estimates and are listed in model_card/fallback_used.
        return 1.0e9 if name in _TIME_RATES else 1.0

    def card(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "expected_schema": CALIBRATION_SCHEMA,
            "status": self.status,
            "source": self.source,
            "units": dict(_RATE_UNITS),
            "rate_scope": "aggregate_device_per_second_except_ns_and_clock",
            "values": dict(self.values),
            "field_sources": dict(self.sources),
        }


class CudaCostModel:
    """Rank whole-kernel configs for one operator and fixed shape.

    Parameters are JSON-compatible mappings.  Use :meth:`from_files` to load
    hardware/calibration/shape JSON.  Custom plugins receive ``(shape, config)``
    and return :class:`KernelWork` or a sequence of launches.  Prediction
    exceptions intentionally propagate to ``GuidedSearch``.
    """

    def __init__(
        self,
        hardware: Mapping[str, Any],
        calibration: Mapping[str, Any],
        operator: str,
        shape: Mapping[str, Any],
        *,
        require_measured: bool = True,
        operator_plugins: Mapping[str, OperatorPlugin] | None = None,
        _hardware_source: str = "<mapping>",
        _calibration_source: str = "<mapping>",
    ):
        if not isinstance(hardware, Mapping) or not isinstance(calibration, Mapping) or not isinstance(shape, Mapping):
            raise TypeError("hardware, calibration, and shape must be mappings")
        if not isinstance(operator, str) or not operator.strip():
            raise CostModelError("operator must be a non-empty string")
        if not isinstance(require_measured, bool):
            raise TypeError("require_measured must be bool")
        self.operator = operator.strip().lower()
        self.shape = copy.deepcopy(dict(shape))
        self.require_measured = require_measured
        self._hardware_source = _hardware_source
        plugins = dict(_BUILTIN_PLUGINS)
        if operator_plugins:
            plugins.update(operator_plugins)
        if self.operator not in plugins:
            raise CostModelError(f"Unknown operator {self.operator!r}; available: {sorted(plugins)}")
        self._plugin = plugins[self.operator]
        self._parse_hardware(hardware)
        self._calibration = _Calibration(calibration, require_measured, _calibration_source)
        self._shape_config = _lower_keys(self.shape, "shape")

    def _parse_hardware(self, hardware: Mapping[str, Any]) -> None:
        if hardware.get("schema", HARDWARE_SCHEMA) != HARDWARE_SCHEMA:
            raise CostModelError(f"hardware.schema must be {HARDWARE_SCHEMA!r}")
        device = hardware.get("device")
        if not isinstance(device, Mapping):
            raise CostModelError("hardware.device must be an object")
        capability = device.get("compute_capability")
        if capability not in {"12.0", "12.1"}:
            raise CostModelError("This mma.sync model supports SM120/SM121 (compute capability 12.0/12.1)")

        def field(name: str, minimum: int = 1) -> int:
            if name not in device or device[name] is None:
                raise CostModelError(f"hardware.device.{name} is required")
            return _integer(device[name], f"hardware.device.{name}", minimum)

        self._sm_count = field("multi_processor_count")
        self._warp_size = field("warp_size")
        self._max_threads_block = field("max_threads_per_block")
        self._max_threads_sm = field("max_threads_per_multiprocessor")
        self._max_registers_sm = field("max_registers_per_multiprocessor")
        self._shared_sm = field("shared_memory_per_multiprocessor_bytes")
        self._shared_block = field("shared_memory_per_block_optin_bytes")
        self._reserved_shared = field("reserved_shared_memory_per_block_bytes", 0)
        self._max_registers_thread = device.get("max_registers_per_thread")
        if self._max_registers_thread is not None:
            self._max_registers_thread = _integer(self._max_registers_thread, "hardware.device.max_registers_per_thread")
        self._max_ctas_sm = _integer(device.get("max_ctas_per_multiprocessor", 16), "hardware.device.max_ctas_per_multiprocessor")
        self._device_card = {
            "name": str(device.get("name", "unknown")),
            "compute_capability": capability,
            "sm_count": self._sm_count,
            "warp_size": self._warp_size,
            "max_ctas_per_sm_structural": self._max_ctas_sm,
            "source": self._hardware_source,
        }

    @classmethod
    def from_files(
        cls,
        hardware: str | os.PathLike[str] | Mapping[str, Any],
        calibration: str | os.PathLike[str] | Mapping[str, Any],
        operator: str | os.PathLike[str],
        shape: str | os.PathLike[str] | Mapping[str, Any],
        require_measured: bool = True,
        **kwargs: Any,
    ) -> CudaCostModel:
        """Construct from manifest/calibration/shape JSON or mappings.

        ``operator`` is normally a built-in name.  It may be a JSON descriptor
        containing ``{"operator": "gemm"}``.  A shape descriptor may wrap its
        dimensions in a top-level ``shape`` object.  ``operator_plugins`` is an
        optional keyword-only mapping for custom operators.
        """
        hardware_data, hardware_source = _load_json(hardware, "hardware")
        calibration_data, calibration_source = _load_json(calibration, "calibration")
        shape_data, _ = _load_json(shape, "shape")
        if "shape" in shape_data:
            wrapped = shape_data["shape"]
            if not isinstance(wrapped, Mapping):
                raise CostModelError("shape descriptor's shape field must be an object")
            shape_data = dict(wrapped)
        if isinstance(operator, os.PathLike) or (isinstance(operator, str) and operator.lower().endswith(".json")):
            operator_data, _ = _load_json(operator, "operator")
            operator_name = operator_data.get("operator", operator_data.get("name"))
        else:
            operator_name = operator
        if not isinstance(operator_name, str):
            raise CostModelError("operator descriptor must contain an operator string")
        return cls(
            hardware_data, calibration_data, operator_name, shape_data,
            require_measured=require_measured, _hardware_source=hardware_source,
            _calibration_source=calibration_source, **kwargs,
        )

    @property
    def fallback_used(self) -> tuple[str, ...]:
        """Calibration-level fallbacks; prediction-only ones are in model_card."""
        return self._calibration.fallback_used

    def model_card(self) -> dict[str, Any]:
        """Return detached pure-JSON provenance, formulas, and limitations."""
        card = {
            "schema": MODEL_CARD_SCHEMA,
            "version": 1,
            "operator": self.operator,
            "shape": copy.deepcopy(self.shape),
            "score": {"term": "predicted_ms", "units": "ms", "direction": "minimize"},
            "purpose": "candidate ranking; predicted absolute milliseconds are unvalidated without isolated measurements",
            "hardware": copy.deepcopy(self._device_card),
            "calibration": self._calibration.card(),
            "require_measured": self.require_measured,
            "fallback_used": list(self.fallback_used),
            "conditional_fallbacks": [
                "config.registers_per_thread missing: register occupancy limit is one CTA/SM",
                "gemm.rasterization_l2_locality_unmodeled",
                "flash_attention_fwd causal=True: full-KV upper-bound work per CTA",
                "GDN fixed-warp roles: conservative serial source-phase composition",
                "GDN forward shared aliases: source-lifetime estimate, not compiled allocation",
                "GDN fragment traffic is counted without inventing a register-file bandwidth time",
                "GDN group-reduce fragment capacity is a logical lower bound, not a ptxas allocation",
                "GDN reverse cumsum includes the TileLang kernel; torch.zeros_like fill and varlen tail zeroing are outside the dense scope",
            ],
            "formula": {
                "occupancy": "min(thread, warp, register, shared, CTA-slot limits); unknown registers give register limit 1",
                "waves": "ceil(cta_count / (SM_count * occupancy_ctas_per_sm))",
                "resource": "one_CTA_ms = work / (aggregate_rate / SM_count) * 1000",
                "global_path_ms": "max(dram_ms, l2_ms)",
                "stage_overlap": "stages=1: sum(compute+cuda+sfu, global_path, shared); stages>=2: max(...)",
                "predicted_ms": "sum_launches(launch_ms + waves * sum_serial_phases(stage_overlap + lsu + barrier))",
            },
            "limits": [
                "Use for deterministic candidate ordering, not a claim of absolute measured latency.",
                "Calibration must be isolated from candidate/timing shapes; no per-config fitted constants are used.",
                "SM120 uses mma.sync-style FP16/BF16/TF32 throughput, not Hopper WGMMA or Blackwell TCGEN05 rates.",
                "GDN models chunk32/K=DV128 fixed producer/S/V/O or K/A/S roles, not a generic warp assignment.",
                "cp.async uses warp issue counts; TMA uses descriptors plus mbarrier phases; logical bytes stay distinct.",
                "Tail predication, cache replacement, bank conflicts, instruction mix, register allocation granularity and tail waves are approximate.",
            ],
            "supported_operators": sorted(_BUILTIN_PLUGINS),
            "plugin": "operator_plugins[name](shape, config) -> KernelWork or sequence[KernelWork]",
        }
        if self.operator == "gdn_bwd":
            card["gdn_scope"] = {
                "direction": "backward",
                "inputs": "A, g, h and other forward artifacts are provided by public forward; KKT is not rerun or charged",
                "kernels_in_program_order": [
                    "prepare_h", "fused_bwd", "gqa_reduce(dq)", "gqa_reduce(dk)",
                    "reverse_dg_cumsum",
                ],
                "gqa_reduce_condition": "the two reduce launches are included only when Hg < H",
                "block_DV": {"value": 128, "tunable": False},
                "prepare_h_num_stages": {"default": 2, "space": [1, 2, 3]},
                "group_reduce_block_size": {
                    "default": 16, "space": [8, 16, 32],
                    "non_default_requires_T_divisible": True,
                },
            }
        elif self.operator == "gdn_fwd":
            card["gdn_scope"] = {
                "direction": "forward",
                "kernels_in_program_order": ["kkt_solve", "fused_fwd"],
                "artifacts": "public forward supplies A/g/forward artifacts consumed by backward",
                "block_DV": {"default": 128, "divisor_of": 128, "multiple_of": 16},
                "unsupported_configs": ["prepare_h_num_stages", "group_reduce_block_size"],
            }
        json.dumps(card, ensure_ascii=False, allow_nan=False)
        return card

    def _occupancy(self, work: KernelWork) -> tuple[int, int, bool]:
        threads = _integer(work.threads_per_cta, f"{work.name}.threads_per_cta")
        if threads > self._max_threads_block:
            raise CostModelCapacityError(f"{work.name} threads_per_cta exceeds hardware block limit")
        capacity = _integer(work.shared_memory_bytes_per_cta, f"{work.name}.shared_memory_bytes_per_cta", 0)
        fragment_capacity = _integer(
            work.fragment_capacity_bytes_per_cta, f"{work.name}.fragment_capacity_bytes_per_cta", 0
        )
        if capacity > self._shared_block:
            raise CostModelCapacityError(
                f"{work.name} shared bytes {capacity} exceed per-block opt-in {self._shared_block}"
            )
        demand_shared = capacity + self._reserved_shared
        if demand_shared > self._shared_sm:
            raise CostModelCapacityError(f"{work.name} shared allocation including reserved bytes exceeds per-SM capacity")
        by_threads = self._max_threads_sm // threads
        warps = (threads + self._warp_size - 1) // self._warp_size
        max_warps = self._max_threads_sm // self._warp_size
        by_warps = max_warps // warps
        by_shared = self._shared_sm // max(1, demand_shared)
        unknown_registers = work.registers_per_thread is None
        if unknown_registers:
            by_registers = 1  # Conservative: never invent an exact register count.
        else:
            registers = _integer(work.registers_per_thread, f"{work.name}.registers_per_thread")
            if self._max_registers_thread is not None and registers > self._max_registers_thread:
                raise CostModelCapacityError(f"{work.name} registers_per_thread exceeds hardware limit")
            demand = threads * registers
            if fragment_capacity > demand * 4:
                raise CostModelCapacityError(
                    f"{work.name} registers_per_thread is below the logical fragment-capacity lower bound"
                )
            if demand > self._max_registers_sm:
                raise CostModelCapacityError(f"{work.name} register demand exceeds one SM")
            by_registers = self._max_registers_sm // demand
        limits = (by_threads, by_warps, by_registers, by_shared, self._max_ctas_sm)
        occupancy = min(limits)
        if occupancy < 1:
            raise CostModelCapacityError(f"{work.name} does not fit one CTA on an SM: limits={limits}")
        ctas = _integer(work.cta_count, f"{work.name}.cta_count")
        waves = (ctas + self._sm_count * occupancy - 1) // (self._sm_count * occupancy)
        return occupancy, waves, unknown_registers

    def _validate_work(self, work: KernelWork) -> None:
        if not isinstance(work, KernelWork):
            raise CostModelError(f"Operator plugin must return KernelWork, got {type(work).__name__}")
        if not isinstance(work.name, str) or not work.name:
            raise CostModelError("KernelWork.name must be non-empty")
        _integer(work.cta_count, f"{work.name}.cta_count")
        _integer(work.launches, f"{work.name}.launches")
        if not work.phases:
            raise CostModelError(f"{work.name} must contain at least one phase")
        for phase in work.phases:
            if not isinstance(phase, PhaseWork):
                raise CostModelError(f"{work.name} phases must be PhaseWork")
            if not isinstance(phase.name, str) or not phase.name:
                raise CostModelError("PhaseWork.name must be non-empty")
            _integer(phase.stages, f"{work.name}.{phase.name}.stages")
            for name in (
                "mma_flops_per_cta", "cuda_flops_per_cta", "dram_bytes_per_cta",
                "l2_bytes_per_cta", "shared_bytes_per_cta", "fragment_bytes_per_cta",
                "lsu_warp_instructions_per_cta", "tma_descriptors_per_cta", "sfu_exp2_operations_per_cta", "barriers_per_cta",
            ):
                _finite_number(getattr(phase, name), f"{work.name}.{phase.name}.{name}")
            if phase.useful_flops_per_cta is not None:
                _finite_number(phase.useful_flops_per_cta, f"{work.name}.{phase.name}.useful_flops_per_cta")
            if phase.mma_flops_per_cta and phase.mma_dtype not in _MMA_RATES:
                raise CostModelError(f"{work.name}.{phase.name} needs mma_dtype in {sorted(_MMA_RATES)}")
        for fallback in work.fallbacks:
            if not isinstance(fallback, str) or not fallback:
                raise CostModelError("KernelWork.fallbacks must contain non-empty strings")

    def predict(self, config: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
        """Return ``(predicted_ms, itemized_terms)`` without catching failures."""
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        normalized = _lower_keys(config, "config")
        raw_work = self._plugin(copy.deepcopy(self._shape_config), copy.deepcopy(normalized))
        works = (raw_work,) if isinstance(raw_work, KernelWork) else tuple(raw_work)
        if not works:
            raise CostModelError("Operator plugin returned no kernels")

        totals: dict[str, float] = {
            "predicted_ms": 0.0, "launch_ms": 0.0, "stage_overlap_ms": 0.0,
            "mma_ms": 0.0, "cuda_ms": 0.0, "dram_ms": 0.0, "l2_ms": 0.0,
            "shared_ms": 0.0, "lsu_ms": 0.0, "sfu_exp2_ms": 0.0, "barrier_ms": 0.0,
            "flops": 0.0, "useful_flops": 0.0, "mma_flops": 0.0, "cuda_flops": 0.0,
            "dram_bytes": 0.0, "l2_bytes": 0.0, "shared_bytes": 0.0, "fragment_bytes": 0.0,
            "lsu_warp_instructions": 0.0, "tma_descriptors": 0.0,
            "sfu_exp2_operations": 0.0, "barrier_count": 0.0,
            "launch_count": 0.0, "cta_count": 0.0, "waves": 0.0,
        }
        stage_totals = {
            name: 0.0
            for name in (
                "kkt", "solve", "wy", "recurrence", "output", "dhu", "dqkwg",
                "group_reduce", "reverse_cumsum",
            )
        }
        kernel_diagnostics = {
            name: {
                "waves": 0.0, "occupancy": 0.0, "shared": 0.0,
                "fragment_capacity": 0.0, "ctas": 0.0, "launches": 0.0,
                "cuda_flops": 0.0, "l2_bytes": 0.0, "shared_traffic": 0.0,
                "fragment_bytes": 0.0,
            }
            for name in ("kkt", "prepare_h", "fused", "group_reduce", "reverse_cumsum")
        }
        occupancies: list[int] = []
        wave_values: list[int] = []
        max_threads = 0
        max_shared_capacity = 0
        max_fragment_capacity = 0
        max_stages = 0
        max_registers = 0
        register_conservative = False
        prediction_fallbacks: set[str] = set()

        for work in works:
            self._validate_work(work)
            occupancy, waves, unknown_registers = self._occupancy(work)
            launches = _integer(work.launches, f"{work.name}.launches")
            occupancies.append(occupancy)
            wave_values.append(waves)
            if self.operator in {"gdn_fwd", "gdn_bwd"}:
                diagnostic_name = {
                    "gdn_kkt_solve": "kkt",
                    "gdn_prepare_h": "prepare_h",
                    "gdn_fused_fwd": "fused",
                    "gdn_fused_bwd": "fused",
                    "gdn_gqa_reduce": "group_reduce",
                    "gdn_reverse_dg_cumsum": "reverse_cumsum",
                }.get(work.name)
                if diagnostic_name is None:
                    raise CostModelError(f"Unknown GDN kernel {work.name!r}")
                diagnostic = kernel_diagnostics[diagnostic_name]
                diagnostic["waves"] += waves * launches
                diagnostic["occupancy"] = occupancy
                diagnostic["shared"] = work.shared_memory_bytes_per_cta
                diagnostic["fragment_capacity"] = work.fragment_capacity_bytes_per_cta
                diagnostic["ctas"] += work.cta_count * launches
                diagnostic["launches"] += launches
            max_threads = max(max_threads, work.threads_per_cta)
            max_shared_capacity = max(max_shared_capacity, work.shared_memory_bytes_per_cta)
            max_fragment_capacity = max(max_fragment_capacity, work.fragment_capacity_bytes_per_cta)
            register_conservative |= unknown_registers
            if work.registers_per_thread is not None:
                max_registers = max(max_registers, work.registers_per_thread)
            else:
                prediction_fallbacks.add(f"{work.name}.registers_per_thread:one-CTA-limit")
            prediction_fallbacks.update(work.fallbacks)
            totals["launch_count"] += launches
            totals["cta_count"] += work.cta_count * launches
            totals["waves"] += waves * launches
            launch_ms = self._calibration.rate("launch_ns") * 1.0e-6
            launch_total = launches * launch_ms
            totals["launch_ms"] += launch_total
            totals["predicted_ms"] += launch_total

            for phase in work.phases:
                max_stages = max(max_stages, phase.stages)
                mma_rate_name = _MMA_RATES.get(phase.mma_dtype or "")
                mma_ms = phase.mma_flops_per_cta / (self._calibration.rate(mma_rate_name) / self._sm_count) * 1000 if mma_rate_name else 0.0
                cuda_ms = phase.cuda_flops_per_cta / (self._calibration.rate("cuda_fp32_flops_per_second") / self._sm_count) * 1000
                dram_ms = phase.dram_bytes_per_cta / (self._calibration.rate("dram_bytes_per_second") / self._sm_count) * 1000
                l2_ms = phase.l2_bytes_per_cta / (self._calibration.rate("l2_bytes_per_second") / self._sm_count) * 1000
                shared_ms = phase.shared_bytes_per_cta / (self._calibration.rate("shared_bytes_per_second") / self._sm_count) * 1000
                lsu_ms = phase.lsu_warp_instructions_per_cta / (self._calibration.rate("lsu_warp_instructions_per_second") / self._sm_count) * 1000
                sfu_ms = phase.sfu_exp2_operations_per_cta / (self._calibration.rate("sfu_exp2_operations_per_second") / self._sm_count) * 1000
                barrier_ms = phase.barriers_per_cta * self._calibration.rate("barrier_ns") * 1.0e-6
                global_ms = max(dram_ms, l2_ms)
                resources = (mma_ms + cuda_ms + sfu_ms, global_ms, shared_ms)
                overlap_ms = max(resources) if phase.stages >= 2 else sum(resources)
                phase_core_ms = overlap_ms + lsu_ms + barrier_ms
                scale = waves * launches

                for name, value in (
                    ("stage_overlap_ms", overlap_ms), ("mma_ms", mma_ms), ("cuda_ms", cuda_ms),
                    ("dram_ms", dram_ms), ("l2_ms", l2_ms), ("shared_ms", shared_ms),
                    ("lsu_ms", lsu_ms), ("sfu_exp2_ms", sfu_ms), ("barrier_ms", barrier_ms),
                ):
                    totals[name] += scale * value
                totals["predicted_ms"] += scale * phase_core_ms
                multiplier = work.cta_count * launches
                mma_flops = phase.mma_flops_per_cta
                cuda_flops = phase.cuda_flops_per_cta
                useful = phase.useful_flops_per_cta
                if useful is None:
                    useful = mma_flops + cuda_flops
                totals["flops"] += multiplier * (mma_flops + cuda_flops)
                totals["useful_flops"] += multiplier * useful
                totals["mma_flops"] += multiplier * mma_flops
                totals["cuda_flops"] += multiplier * cuda_flops
                totals["dram_bytes"] += multiplier * phase.dram_bytes_per_cta
                totals["l2_bytes"] += multiplier * phase.l2_bytes_per_cta
                totals["shared_bytes"] += multiplier * phase.shared_bytes_per_cta
                totals["fragment_bytes"] += multiplier * phase.fragment_bytes_per_cta
                totals["lsu_warp_instructions"] += multiplier * phase.lsu_warp_instructions_per_cta
                totals["tma_descriptors"] += multiplier * phase.tma_descriptors_per_cta
                totals["sfu_exp2_operations"] += multiplier * phase.sfu_exp2_operations_per_cta
                totals["barrier_count"] += multiplier * phase.barriers_per_cta
                if self.operator in {"gdn_fwd", "gdn_bwd"}:
                    if phase.name not in stage_totals:
                        raise CostModelError(f"Unknown GDN stage {phase.name!r}")
                    stage_totals[phase.name] += scale * phase_core_ms
                    diagnostic["cuda_flops"] += multiplier * cuda_flops
                    diagnostic["l2_bytes"] += multiplier * phase.l2_bytes_per_cta
                    diagnostic["shared_traffic"] += multiplier * phase.shared_bytes_per_cta
                    diagnostic["fragment_bytes"] += multiplier * phase.fragment_bytes_per_cta

        totals.update({f"gdn_{name}_ms": value for name, value in stage_totals.items()})
        for name, diagnostic in kernel_diagnostics.items():
            totals[f"gdn_{name}_waves"] = diagnostic["waves"]
            totals[f"gdn_{name}_occupancy_ctas_per_sm"] = diagnostic["occupancy"]
            totals[f"gdn_{name}_shared_memory_bytes_per_cta"] = diagnostic["shared"]
            totals[f"gdn_{name}_fragment_capacity_bytes_per_cta"] = diagnostic["fragment_capacity"]
            totals[f"gdn_{name}_cta_count"] = diagnostic["ctas"]
            totals[f"gdn_{name}_launch_count"] = diagnostic["launches"]
            totals[f"gdn_{name}_cuda_flops"] = diagnostic["cuda_flops"]
            totals[f"gdn_{name}_l2_bytes"] = diagnostic["l2_bytes"]
            totals[f"gdn_{name}_shared_bytes"] = diagnostic["shared_traffic"]
            totals[f"gdn_{name}_fragment_bytes"] = diagnostic["fragment_bytes"]
        totals["occupancy_ctas_per_sm"] = float(min(occupancies))
        totals["max_waves_per_kernel"] = float(max(wave_values))
        totals["threads_per_cta"] = float(max_threads)
        totals["registers_per_thread"] = float(max_registers)
        totals["register_occupancy_conservative"] = float(register_conservative)
        totals["shared_memory_bytes_per_cta"] = float(max_shared_capacity)
        totals["fragment_capacity_bytes_per_cta"] = float(max_fragment_capacity)
        totals["pipeline_stages"] = float(max_stages)
        totals["fallback_count"] = float(len(self.fallback_used) + len(prediction_fallbacks))
        totals["gdn_block_dv"] = float(_config_int(normalized, "block_DV", default=128)) if self.operator in {"gdn_fwd", "gdn_bwd"} else 0.0
        totals["gdn_prepare_h_num_stages"] = float(_config_int(normalized, "prepare_h_num_stages", default=2)) if self.operator == "gdn_bwd" else 0.0
        totals["gdn_group_reduce_block_size"] = float(_config_int(normalized, "group_reduce_block_size", default=16)) if self.operator == "gdn_bwd" else 0.0

        for name, value in totals.items():
            _finite_number(value, f"prediction term {name}")
        score = totals["predicted_ms"]
        if score <= 0:
            raise CostModelError("predicted_ms must be positive; zero would hide missing work or calibration")
        terms = dict(sorted((name, float(value)) for name, value in totals.items()))
        json.dumps(terms, ensure_ascii=False, allow_nan=False)
        return float(score), terms


__all__ = [
    "CALIBRATION_SCHEMA", "CalibrationError", "CostModelCapacityError", "CostModelError", "CudaCostModel",
    "HARDWARE_SCHEMA", "KernelWork", "MODEL_CARD_SCHEMA", "OperatorPlugin", "PhaseWork",
]
