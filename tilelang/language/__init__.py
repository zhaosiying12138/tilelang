"""Default TileLang language facade.

``tilelang.language`` re-exports the CUDA dialect so that ``import
tilelang.language as T`` yields the common surface plus CUDA extensions. Other
backends are reached explicitly via ``tilelang.<backend>.language`` (which build
on ``tilelang.language.common``).
"""

from __future__ import annotations

from tilelang.cuda.language import *  # noqa: F401,F403
from tilelang.cuda.language import __all__ as __all__  # noqa: F401
from tilelang.language.annotations import PerRoleTask, Stage, Task  # noqa: F401

# Task/Stage default to the legacy flat schema; recursive markers require an
# explicit schema="recursive" opt-in. PerRoleTask exists only in that schema.
__all__ = tuple(dict.fromkeys((*__all__, "PerRoleTask", "Stage", "Task")))

# Imported by name so static type checkers resolve the CUDA-typed signatures
# through this facade (they cannot evaluate the dynamic __all__).
from tilelang.cuda.language import (  # noqa: F401
    Kernel,
    Parallel,
    Unroll,
    atomic_add,
    copy,
    gemm,
    gemm_blockscaled,
    gemm_sp,
    im2col,
    reduce_absmax,
    reduce_max,
    reduce_min,
    unroll,
)

__tilelang_dialect__ = "cuda"
