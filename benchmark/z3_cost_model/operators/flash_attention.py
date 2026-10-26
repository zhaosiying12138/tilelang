"""Causal/non-causal BSHD flash attention with a 54-point raw space."""

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from ._common import (
    SEED,
    SPACE_SOURCE,
    UPSTREAM_REVISION,
    ConstraintModel,
    ValidationReport,
    base_constraint_clauses,
    bool_clauses_to_z3,
    boolean,
    ceildiv,
    check_fp16_inputs,
    compile_prim_func,
    constraints_manifest,
    dtype_manifest,
    ieee_fp32_precision,
    make_fp16_inputs,
    positive_int,
    require_exact_keys,
    stock_cuda_gemm_constraint,
    strict_validate,
    torch_module,
)

if TYPE_CHECKING:
    from tilelang.autotuner.space import ConfigSpace

OPERATOR = "flash_attention_bshd"
CONFIG_KEYS = ("block_M", "block_N", "num_stages", "threads")
DOMAINS = {
    "block_M": [64, 128, 256],
    "block_N": [64, 128, 256],
    "num_stages": [1, 2, 3],
    "threads": [128, 256],
}
RAW_COUNT = math.prod(len(values) for values in DOMAINS.values())


@dataclass(frozen=True)
class FAShape:
    B: int = 1
    H: int = 16
    S: int = 2048
    D: int = 64
    is_causal: bool = True

    def __post_init__(self) -> None:
        positive_int(self.B, "B")
        positive_int(self.H, "H")
        positive_int(self.S, "S")
        positive_int(self.D, "D")
        boolean(self.is_causal, "is_causal")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FAConfig:
    block_M: int
    block_N: int
    num_stages: int
    threads: int

    def __post_init__(self) -> None:
        positive_int(self.block_M, "block_M")
        positive_int(self.block_N, "block_N")
        positive_int(self.num_stages, "num_stages")
        positive_int(self.threads, "threads")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "FAConfig":
        return cls(**require_exact_keys(config, CONFIG_KEYS))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRIMARY_SHAPE = FAShape()
SHAPE = PRIMARY_SHAPE


def _trip_bounds(shape: FAShape, config: FAConfig) -> tuple[int, int]:
    maximum = ceildiv(shape.S, config.block_N)
    if not shape.is_causal:
        return maximum, maximum
    minimum = ceildiv(min(shape.S, config.block_M), config.block_N)
    return minimum, maximum


def _shared_memory_lower_bound(shape: FAShape, config: FAConfig) -> int:
    q = 2 * config.block_M * shape.D
    k = 2 * config.block_N * shape.D
    v = 2 * config.block_N * shape.D
    o = 2 * config.block_M * shape.D
    # Q stays live across the KV loop. K and V need not coexist with each
    # other, and O is written only after the loop, so reuse may overlap them.
    return max(q + k, q + v, o)


def constraint_builder(
    config: Mapping[str, Any],
    z3: Any,
    context: Any,
    *,
    shape: FAShape = PRIMARY_SHAPE,
    model: ConstraintModel | None = None,
) -> list[Any]:
    cfg = FAConfig.from_mapping(config)
    values = cfg.to_dict()
    model = model or ConstraintModel()
    min_trip, _ = _trip_bounds(shape, cfg)
    clauses = base_constraint_clauses(
        values,
        tile_names=("block_M", "block_N"),
        min_trip_count=min_trip,
        model=model,
    )
    for name, block_n, block_k in (
        ("qk", cfg.block_N, shape.D),
        ("pv", shape.D, cfg.block_N),
    ):
        atom = stock_cuda_gemm_constraint(cfg.block_M, block_n, block_k, cfg.threads, model.target)
        if atom is not None:
            clauses[f"stock_cuda_gemm_atom_{name}"] = atom
    if model.shared_memory_per_block_limit_bytes is not None:
        clauses["shared_memory_necessary_lower_bound"] = _shared_memory_lower_bound(shape, cfg) <= model.shared_memory_per_block_limit_bytes
    return bool_clauses_to_z3(clauses, z3, context)


def config_space(
    *,
    shape: FAShape = PRIMARY_SHAPE,
    model: ConstraintModel | None = None,
) -> "ConfigSpace":
    from tilelang.autotuner.space import ConfigSpace

    selected_model = model or ConstraintModel()
    return ConfigSpace(
        DOMAINS,
        constraint_builder=partial(constraint_builder, shape=shape, model=selected_model),
    )


def raw_configs() -> list[dict[str, Any]]:
    return config_space().raw_configs()


def _build_prim_func(shape: FAShape, cfg: FAConfig) -> Any:
    import tilelang.language as T

    dtype = T.float16
    accum_dtype = T.float32
    B, H, S, D, is_causal = shape.B, shape.H, shape.S, shape.D, shape.is_causal
    tensor_shape = (B, S, H, D)
    scale = (1.0 / math.sqrt(D)) * math.log2(math.e)

    @T.prim_func
    def main(
        Q: T.Tensor(tensor_shape, dtype),
        K: T.Tensor(tensor_shape, dtype),
        V: T.Tensor(tensor_shape, dtype),
        Output: T.Tensor(tensor_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(S, cfg.block_M), H, B, threads=cfg.threads) as (bx, by, bz):
            Q_shared = T.alloc_shared((cfg.block_M, D), dtype)
            K_shared = T.alloc_shared((cfg.block_N, D), dtype)
            V_shared = T.alloc_shared((cfg.block_N, D), dtype)
            O_shared = T.alloc_shared((cfg.block_M, D), dtype)
            acc_s = T.alloc_fragment((cfg.block_M, cfg.block_N), accum_dtype)
            acc_s_cast = T.alloc_fragment((cfg.block_M, cfg.block_N), dtype)
            acc_o = T.alloc_fragment((cfg.block_M, D), accum_dtype)
            scores_max = T.alloc_fragment((cfg.block_M,), accum_dtype)
            scores_max_prev = T.alloc_fragment((cfg.block_M,), accum_dtype)
            scores_scale = T.alloc_fragment((cfg.block_M,), accum_dtype)
            scores_sum = T.alloc_fragment((cfg.block_M,), accum_dtype)
            logsum = T.alloc_fragment((cfg.block_M,), accum_dtype)

            T.copy(Q[bz, bx * cfg.block_M : (bx + 1) * cfg.block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            loop_range = (
                T.min(T.ceildiv(S, cfg.block_N), T.ceildiv((bx + 1) * cfg.block_M, cfg.block_N)) if is_causal else T.ceildiv(S, cfg.block_N)
            )
            for k in T.Pipelined(loop_range, num_stages=cfg.num_stages):
                T.copy(K[bz, k * cfg.block_N : (k + 1) * cfg.block_N, by, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(cfg.block_M, cfg.block_N):
                        acc_s[i, j] = T.if_then_else(
                            bx * cfg.block_M + i >= k * cfg.block_N + j,
                            0,
                            -T.infinity(acc_s.dtype),
                        )
                else:
                    for i, j in T.Parallel(cfg.block_M, cfg.block_N):
                        acc_s[i, j] = T.if_then_else(
                            k * cfg.block_N + j >= S,
                            -T.infinity(acc_s.dtype),
                            0,
                        )
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(cfg.block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(cfg.block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(cfg.block_M, cfg.block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(cfg.block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(cfg.block_M, D):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, k * cfg.block_N : (k + 1) * cfg.block_N, by, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(cfg.block_M, D):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * cfg.block_M : (bx + 1) * cfg.block_M, by, :])

    return main


def kernel_factory(
    config: Mapping[str, Any],
    *,
    shape: FAShape = PRIMARY_SHAPE,
    pass_configs: Mapping[Any, Any] | None = None,
    target: Any = "auto",
    compile: bool = True,
) -> Any:
    cfg = FAConfig.from_mapping(config)
    prim_func = _build_prim_func(shape, cfg)
    if not compile:
        if pass_configs is not None:
            raise ValueError("pass_configs requires compile=True")
        return prim_func
    return compile_prim_func(prim_func, target=target, pass_configs=pass_configs)


def make_inputs(
    *,
    shape: FAShape = PRIMARY_SHAPE,
    device: Any = "cuda",
    seed: int = SEED,
) -> dict[str, Any]:
    tensor_shape = (shape.B, shape.S, shape.H, shape.D)
    return make_fp16_inputs(
        (
            ("Q", tensor_shape, 1.0),
            ("K", tensor_shape, 1.0),
            ("V", tensor_shape, 1.0),
        ),
        device=device,
        seed=seed,
    )


def reference(inputs: Mapping[str, Any], *, shape: FAShape = PRIMARY_SHAPE) -> Any:
    """FP32 scaled-dot-product attention with an explicit inclusive lower triangle."""
    tensor_shape = (shape.B, shape.S, shape.H, shape.D)
    check_fp16_inputs(inputs, {"Q": tensor_shape, "K": tensor_shape, "V": tensor_shape})
    torch = torch_module()
    q = inputs["Q"].permute(0, 2, 1, 3).float()
    k = inputs["K"].permute(0, 2, 1, 3).float()
    v = inputs["V"].permute(0, 2, 1, 3).float()
    with ieee_fp32_precision():
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(shape.D)
        if shape.is_causal:
            positions = torch.arange(shape.S, device=scores.device)
            # Query row i attends to key column j iff j <= i. The diagonal is
            # included and every row therefore has at least one finite score.
            causal_mask = positions.unsqueeze(0) <= positions.unsqueeze(1)
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        result = probabilities @ v
    return result.permute(0, 2, 1, 3).contiguous()


def validate(
    actual: Any,
    expected: Any | None = None,
    *,
    inputs: Mapping[str, Any] | None = None,
    shape: FAShape = PRIMARY_SHAPE,
    config: Mapping[str, Any] | None = None,
    atol: float = 2e-2,
    rtol: float = 2e-2,
) -> ValidationReport:
    if expected is None:
        if inputs is None:
            raise ValueError("provide expected or inputs for reference computation")
        expected = reference(inputs, shape=shape)
    elif inputs is not None:
        raise ValueError("provide expected or inputs, not both")
    return strict_validate(OPERATOR, actual, expected, config=config, atol=atol, rtol=rtol)


def suite_manifest(*, model: ConstraintModel | None = None) -> dict[str, Any]:
    selected_model = model or ConstraintModel()
    return {
        "schema": "tilelang.z3_cost_model.operator_suite.v1",
        "operator": OPERATOR,
        "shape": PRIMARY_SHAPE.to_dict(),
        "dtype": dtype_manifest(),
        "layout": "BSHD",
        "operation": "softmax((Q@K^T)/sqrt(D) + causal_mask)@V; mask[j,i]=1 iff i<=j",
        "raw_count": RAW_COUNT,
        "configuration_keys": list(CONFIG_KEYS),
        "canonical_id": "sha256(ConfigSpace.canonical_json(config).utf8)",
        "seed": SEED,
        "default_pass_configs": None,
        "constraints": constraints_manifest(
            trip_count="max=ceildiv(S,block_N); causal min=ceildiv(min(S,block_M),block_N)",
            shared_lower_bound="max(Q+K,Q+V,O), where Q=O=2*block_M*D and K=V=2*block_N*D bytes",
            gemm_operations=(
                "QK: M=block_M, N=block_N, K=D, policy=FullRow",
                "PV: M=block_M, N=D, K=block_N, policy=FullRow",
            ),
            model=selected_model,
        ),
        "provenance": {
            "upstream_revision": UPSTREAM_REVISION,
            "sources": ["examples/flash_attention/example_mha_fwd_bshd.py", SPACE_SOURCE],
            "derivation": "Upstream online-softmax schedule and BSHD layout; no fast-math pass is enabled by default. No performance data is inherited.",
        },
    }
