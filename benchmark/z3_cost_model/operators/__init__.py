"""Classic GEMM, convolution, and flash-attention benchmark definitions.

GPU compilation is explicit through each module's kernel_factory.  Importing
this package performs no CUDA probing and creates no tensors.
"""

from . import convolution, flash_attention, gemm

OPERATOR_MODULES = (gemm, convolution, flash_attention)


def suite_manifest() -> dict:
    """Return the JSON-safe aggregate represented by manifest.json."""
    return {
        "schema": "tilelang.z3_cost_model.classic_operators.v1",
        "operators": [module.suite_manifest() for module in OPERATOR_MODULES],
    }


__all__ = ["OPERATOR_MODULES", "convolution", "flash_attention", "gemm", "suite_manifest"]
