"""NHWC implicit-GEMM convolution with an exact 144-point raw space."""

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

OPERATOR = "convolution_nhwc_implicit_gemm"
CONFIG_KEYS = ("block_M", "block_N", "block_K", "num_stages", "threads")
DOMAINS = {
    "block_M": [64, 128, 256],
    "block_N": [64, 128, 256],
    "block_K": [32, 64],
    "num_stages": [0, 1, 2, 3],
    "threads": [128, 256],
}
RAW_COUNT = math.prod(len(values) for values in DOMAINS.values())


@dataclass(frozen=True)
class ConvShape:
    N: int = 4
    C: int = 64
    H: int = 33
    W: int = 33
    F: int = 96
    K: int = 3
    S: int = 2
    D: int = 1
    P: int = 1

    def __post_init__(self) -> None:
        for name in ("N", "C", "H", "W", "F", "K", "S", "D"):
            positive_int(getattr(self, name), name)
        nonnegative_int(self.P, "P")
        if self.OH <= 0 or self.OW <= 0:
            raise ValueError(f"convolution output must be non-empty, got OH={self.OH}, OW={self.OW}")

    @property
    def OH(self) -> int:
        return (self.H + 2 * self.P - self.D * (self.K - 1) - 1) // self.S + 1

    @property
    def OW(self) -> int:
        return (self.W + 2 * self.P - self.D * (self.K - 1) - 1) // self.S + 1

    @property
    def reduction(self) -> int:
        return self.K * self.K * self.C

    def to_dict(self) -> dict[str, int]:
        result = asdict(self)
        result.update(OH=self.OH, OW=self.OW, reduction=self.reduction)
        return result


@dataclass(frozen=True)
class ConvConfig:
    block_M: int
    block_N: int
    block_K: int
    num_stages: int
    threads: int

    def __post_init__(self) -> None:
        positive_int(self.block_M, "block_M")
        positive_int(self.block_N, "block_N")
        positive_int(self.block_K, "block_K")
        nonnegative_int(self.num_stages, "num_stages")
        positive_int(self.threads, "threads")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ConvConfig":
        return cls(**require_exact_keys(config, CONFIG_KEYS))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRIMARY_SHAPE = ConvShape()
SHAPE = PRIMARY_SHAPE


def _shared_memory_lower_bound(config: ConvConfig) -> int:
    gemm_operands = 2 * config.block_K * (config.block_M + config.block_N) * max(1, config.num_stages)
    output = 2 * config.block_M * config.block_N
    return max(gemm_operands, output)


def constraint_builder(
    config: Mapping[str, Any],
    z3: Any,
    context: Any,
    *,
    shape: ConvShape = PRIMARY_SHAPE,
    model: ConstraintModel | None = None,
) -> list[Any]:
    cfg = ConvConfig.from_mapping(config)
    values = cfg.to_dict()
    model = model or ConstraintModel()
    trip_count = ceildiv(shape.reduction, cfg.block_K)
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
    shape: ConvShape = PRIMARY_SHAPE,
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


def _build_prim_func(shape: ConvShape, cfg: ConvConfig) -> Any:
    import tilelang.language as T

    dtype = T.float16
    accum_dtype = T.float32
    N, C, H, W, F, K, S, D, P = (
        shape.N,
        shape.C,
        shape.H,
        shape.W,
        shape.F,
        shape.K,
        shape.S,
        shape.D,
        shape.P,
    )
    KH = KW = K
    OH, OW = shape.OH, shape.OW

    @T.prim_func
    def main(
        data: T.Tensor((N, H, W, C), dtype),
        kernel: T.Tensor((KH, KW, C, F), dtype),
        out: T.Tensor((N, OH, OW, F), dtype),
    ):
        with T.Kernel(
            T.ceildiv(F, cfg.block_N),
            T.ceildiv(N * OH * OW, cfg.block_M),
            threads=cfg.threads,
        ) as (bx, by):
            data_shared = T.alloc_shared((cfg.block_M, cfg.block_K), dtype)
            kernel_shared = T.alloc_shared((cfg.block_K, cfg.block_N), dtype)
            out_local = T.alloc_fragment((cfg.block_M, cfg.block_N), accum_dtype)
            out_shared = T.alloc_shared((cfg.block_M, cfg.block_N), dtype)

            kernel_flat = T.Tensor((KH * KW * C, F), dtype, kernel.data)
            out_flat = T.Tensor((N * OH * OW, F), dtype, out.data)
            T.clear(out_local)
            for k_iter in T.Pipelined(T.ceildiv(KH * KW * C, cfg.block_K), num_stages=cfg.num_stages):
                T.im2col(data, data_shared, by, k_iter, KH, S, D, P)
                T.copy(kernel_flat[k_iter * cfg.block_K, bx * cfg.block_N], kernel_shared)
                T.gemm(data_shared, kernel_shared, out_local)
            T.copy(out_local, out_shared)
            T.copy(out_shared, out_flat[by * cfg.block_M, bx * cfg.block_N])

    return main


def kernel_factory(
    config: Mapping[str, Any],
    *,
    shape: ConvShape = PRIMARY_SHAPE,
    pass_configs: Mapping[Any, Any] | None = None,
    target: Any = "auto",
    compile: bool = True,
) -> Any:
    cfg = ConvConfig.from_mapping(config)
    prim_func = _build_prim_func(shape, cfg)
    if not compile:
        if pass_configs is not None:
            raise ValueError("pass_configs requires compile=True")
        return prim_func
    return compile_prim_func(prim_func, target=target, pass_configs=pass_configs)


def make_inputs(
    *,
    shape: ConvShape = PRIMARY_SHAPE,
    device: Any = "cuda",
    seed: int = SEED,
) -> dict[str, Any]:
    return make_fp16_inputs(
        (
            ("data", (shape.N, shape.H, shape.W, shape.C), 0.125),
            ("kernel", (shape.K, shape.K, shape.C, shape.F), 0.125),
        ),
        device=device,
        seed=seed,
    )


def reference(inputs: Mapping[str, Any], *, shape: ConvShape = PRIMARY_SHAPE) -> Any:
    """Independent FP32 NHWC cross-correlation with stride/dilation/padding."""
    check_fp16_inputs(
        inputs,
        {
            "data": (shape.N, shape.H, shape.W, shape.C),
            "kernel": (shape.K, shape.K, shape.C, shape.F),
        },
    )
    torch = torch_module()
    data_nchw = inputs["data"].permute(0, 3, 1, 2).float()
    kernel_fchw = inputs["kernel"].permute(3, 2, 0, 1).float()
    with ieee_fp32_precision():
        result = torch.nn.functional.conv2d(
            data_nchw,
            kernel_fchw,
            stride=shape.S,
            padding=shape.P,
            dilation=shape.D,
        )
    return result.permute(0, 2, 3, 1).contiguous()


def validate(
    actual: Any,
    expected: Any | None = None,
    *,
    inputs: Mapping[str, Any] | None = None,
    shape: ConvShape = PRIMARY_SHAPE,
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
        "layout": {"data": "NHWC", "kernel": "KH,KW,C,F", "output": "N,OH,OW,F"},
        "operation": "NHWC implicit GEMM; reduction=KH*KW*C, output rows=N*OH*OW",
        "raw_count": RAW_COUNT,
        "configuration_keys": list(CONFIG_KEYS),
        "canonical_id": "sha256(ConfigSpace.canonical_json(config).utf8)",
        "seed": SEED,
        "default_pass_configs": None,
        "constraints": constraints_manifest(
            trip_count="ceildiv(K*K*C, block_K)",
            shared_lower_bound="max(2*block_K*(block_M+block_N)*max(1,num_stages), 2*block_M*block_N)",
            gemm_operations=("M=block_M, N=block_N, K=block_K, policy=Square",),
            model=selected_model,
        ),
        "provenance": {
            "upstream_revision": UPSTREAM_REVISION,
            "sources": ["examples/convolution/example_convolution_autotune.py", SPACE_SOURCE],
            "derivation": "Upstream im2col implicit-GEMM schedule; rasterization is deliberately absent from the 144-point space. No performance data is inherited.",
        },
    }
