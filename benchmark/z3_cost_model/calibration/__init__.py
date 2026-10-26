"""Isolated CUDA-event microbenchmarks for the TileLang Z3 cost model.

Importing this package is CPU-only.  Torch and TileLang are imported only by
``microbenchmarks.py`` when a worker is actually asked to execute on a GPU.
"""

from .specs import CORE_BENCHMARKS, get_spec, list_benchmarks

__all__ = ["CORE_BENCHMARKS", "get_spec", "list_benchmarks"]
