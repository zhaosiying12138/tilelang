#!/usr/bin/env bash

TILELANG_REPO_ROOT="${TILELANG_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
TILELANG_CONDA_PREFIX="${TILELANG_CONDA_PREFIX:-$HOME/miniforge3/envs/tilelang-cu133}"
TILELANG_PYTHON="${TILELANG_PYTHON:-$TILELANG_CONDA_PREFIX/bin/python}"

if [[ ! -x "$TILELANG_PYTHON" ]]; then
  printf 'TileLang Python not found: %s\nRun scripts/install_wsl.sh first or set TILELANG_PYTHON.\n' "$TILELANG_PYTHON" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in /usr/local/cuda-13.3 /usr/local/cuda; do
    if [[ -x "$candidate/bin/nvcc" ]]; then
      CUDA_HOME="$candidate"
      break
    fi
  done
fi

if [[ -z "${CUDA_HOME:-}" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  printf 'CUDA toolkit not found; set CUDA_HOME to a toolkit with bin/nvcc.\n' >&2
  return 1 2>/dev/null || exit 1
fi

export TILELANG_REPO_ROOT TILELANG_CONDA_PREFIX TILELANG_PYTHON CUDA_HOME
export CUDAToolkit_ROOT="$CUDA_HOME"
export PATH="/usr/lib/wsl/lib:$TILELANG_CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CMAKE_GENERATOR=Ninja
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export MAX_JOBS="${MAX_JOBS:-$CMAKE_BUILD_PARALLEL_LEVEL}"
