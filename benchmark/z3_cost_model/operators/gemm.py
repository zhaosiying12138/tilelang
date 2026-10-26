"""FP16 GEMM with FP32 accumulation and an exact 288-point raw space."""

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
    nonnegative_int,
    positive_int,
    require_exact_keys,
    stock_cuda_gemm_constraint,
    strict_validate,
    torch_module,
)

if TYPE_CHECKING:
    from tilelang.autotuner.space import ConfigSpace

OPERATOR = "gemm"
CONFIG_KEYS = ("block_M", "block_N", "block_K", "num_stages", "threads", "enable_rasterization")
DOMAINS = {
    "block_M": [64, 128, 256],
    "block_N": [64, 128, 256],
    "block_K": [32, 64],
    "num_stages": [0, 1, 2, 3],
    "threads": [128, 256],
    "enable_rasterization": [False, True],
}
RAW_COUNT = math.prod(len(values) for values in DOMAINS.values())


@dataclass(frozen=True)
class GemmShape:
    M: int = 4096
    N: int = 4096
    K: int = 4096

    def __post_init__(self) -> None:
        positive_int(self.M, "M")
        positive_int(self.N, "N")
        positive_int(self.K, "K")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class GemmConfig:
    block_M: int
    block_N: int
    block_K: int
    num_stages: int
    threads: int
    enable_rasterization: bool

    def __post_init__(self) -> None:
        positive_int(self.block_M, "block_M")
        positive_int(self.block_N, "block_N")
        positive_int(self.block_K, "block_K")
        nonnegative_int(self.num_stages, "num_stages")
        positive_int(self.threads, "threads")
        boolean(self.enable_rasterization, "enable_rasterization")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "GemmConfig":
        return cls(**require_exact_keys(config, CONFIG_KEYS))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRIMARY_SHAPE = GemmShape()
SHORT_K_SHAPE = GemmShape(M=8192, N=8192, K=256)
SHAPE = PRIMARY_SHAPE
SHAPES = (PRIMARY_SHAPE, SHORT_K_SHAPE)


def _shared_memory_lower_bound(config: GemmConfig) -> int:
    # A and B coexist at T.gemm. C is live after the loop and may reuse their
    # storage, so summing all three would not be a sound necessary condition.
    gemm_operands = 2 * config.block_K * (config.block_M + config.block_N)
    output = 2 * config.block_M * config.block_N
    return max(gemm_operands, output)


def constraint_builder(
    config: Mapping[str, Any],
    z3: Any,
    context: Any,
    *,
    shape: GemmShape = PRIMARY_SHAPE,
    model: ConstraintModel | None = None,
) -> list[Any]:
    cfg = GemmConfig.from_mapping(config)
    values = cfg.to_dict()
    model = model or ConstraintModel()
    trip_count = ceildiv(shape.K, cfg.block_K)
    clauses = base_constraint_clauses(
        values,
        tile_names=("block_M", "block_N", "block_K"),
        min_trip_count=trip_count,
        model=model,
    )
    atom = stock_cuda_gemm_constraint(cfg.block_M, cfg.block_N, cfg.block_K, cfg.threads, model.target)
    if atom is not None:
        clauses["stock_cuda_gemm_atom"] = atom
    if model.shared_memory_per_block_limit_bytes is not None:
        clauses["shared_memory_necessary_lower_bound"] = _shared_memory_lower_bound(cfg) <= model.shared_memory_per_block_limit_bytes
    return bool_clauses_to_z3(clauses, z3, context)


def config_space(
    *,
    shape: GemmShape = PRIMARY_SHAPE,
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


def _build_prim_func(shape: GemmShape, cfg: GemmConfig) -> Any:
    import tilelang.language as T

    dtype = T.float16
    accum_dtype = T.float32
    M, N, K = shape.M, shape.N, shape.K

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, cfg.block_N), T.ceildiv(M, cfg.block_M), threads=cfg.threads) as (bx, by):
            A_shared = T.alloc_shared((cfg.block_M, cfg.block_K), dtype)
            B_shared = T.alloc_shared((cfg.block_N, cfg.block_K), dtype)
            C_local = T.alloc_fragment((cfg.block_M, cfg.block_N), accum_dtype)
            C_shared = T.alloc_shared((cfg.block_M, cfg.block_N), dtype)
            T.use_swizzle(panel_size=10, enable=cfg.enable_rasterization)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, cfg.block_K), num_stages=cfg.num_stages):
                T.copy(A[by * cfg.block_M, k * cfg.block_K], A_shared)
                T.copy(B[bx * cfg.block_N, k * cfg.block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * cfg.block_M, bx * cfg.block_N])

    return main


def kernel_factory(
    config: Mapping[str, Any],
    *,
    shape: GemmShape = PRIMARY_SHAPE,
    pass_configs: Mapping[Any, Any] | None = None,
    target: Any = "auto",
    compile: bool = True,
) -> Any:
    """Build a compiled JITKernel, or a PrimFunc with compile=False.

    pass_configs defaults to None (the stock pipeline), including for callers
    that later opt into fast-math or target-specific passes explicitly.
    """
    cfg = GemmConfig.from_mapping(config)
    prim_func = _build_prim_func(shape, cfg)
    if not compile:
        if pass_configs is not None:
            raise ValueError("pass_configs requires compile=True")
        return prim_func
    return compile_prim_func(prim_func, target=target, pass_configs=pass_configs)


def make_inputs(
    *,
    shape: GemmShape = PRIMARY_SHAPE,
    device: Any = "cuda",
    seed: int = SEED,
) -> dict[str, Any]:
    return make_fp16_inputs(
        (
            ("A", (shape.M, shape.K), 0.125),
            ("B", (shape.N, shape.K), 0.125),
        ),
        device=device,
        seed=seed,
    )


def reference(inputs: Mapping[str, Any], *, shape: GemmShape = PRIMARY_SHAPE) -> Any:
    """Independent FP32 C[m,n] = sum_k A[m,k] * B[n,k]."""
    check_fp16_inputs(inputs, {"A": (shape.M, shape.K), "B": (shape.N, shape.K)})
    torch_module()  # Validate the lazy dependency before entering the backend context.
    with ieee_fp32_precision():
        result = inputs["A"].float() @ inputs["B"].float().transpose(0, 1)
    return result.contiguous()


def validate(
    actual: Any,
    expected: Any | None = None,
    *,
    inputs: Mapping[str, Any] | None = None,
    shape: GemmShape = PRIMARY_SHAPE,
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
        "shape_variants": [
            {"name": "main_4096_cubed", "shape": PRIMARY_SHAPE.to_dict()},
            {"name": "short_k_m_n_8192", "shape": SHORT_K_SHAPE.to_dict()},
        ],
        "dtype": dtype_manifest(),
        "operation": "C[M,N] = A[M,K] @ B[N,K]^T",
        "raw_count": RAW_COUNT,
        "configuration_keys": list(CONFIG_KEYS),
        "canonical_id": "sha256(ConfigSpace.canonical_json(config).utf8)",
        "seed": SEED,
        "default_pass_configs": None,
        "constraints": constraints_manifest(
            trip_count="ceildiv(K, block_K)",
            shared_lower_bound="max(2*block_K*(block_M+block_N), 2*block_M*block_N)",
            gemm_operations=("M=block_M, N=block_N, K=block_K, policy=Square",),
            model=selected_model,
        ),
        "provenance": {
            "upstream_revision": UPSTREAM_REVISION,
            "sources": ["examples/gemm/example_gemm_autotune.py", SPACE_SOURCE],
            "derivation": "Upstream tile schedule and Cartesian space; canonical keys threads/enable_rasterization are used directly. No performance data is inherited.",
        },
    }
