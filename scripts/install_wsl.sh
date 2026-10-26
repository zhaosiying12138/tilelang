#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
TILELANG_CONDA_PREFIX="${TILELANG_CONDA_PREFIX:-$HOME/miniforge3/envs/tilelang-cu133}"
PYTHON="$TILELANG_CONDA_PREFIX/bin/python"

if [[ ! -f "$CONDA_SH" ]]; then
  printf 'Conda activation script not found: %s\nSet CONDA_SH or create TILELANG_CONDA_PREFIX with Python 3.12.\n' "$CONDA_SH" >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  if [[ -e "$TILELANG_CONDA_PREFIX" ]]; then
    printf 'Environment path exists but has no usable Python: %s\n' "$TILELANG_CONDA_PREFIX" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda create --prefix "$TILELANG_CONDA_PREFIX" python=3.12 pip -y
fi

for tool in cmake ninja nvcc; do
  if ! command -v "$tool" >/dev/null 2>&1 && [[ "$tool" != nvcc || ! -x /usr/local/cuda-13.3/bin/nvcc ]]; then
    printf 'Required build tool is missing: %s\n' "$tool" >&2
    exit 1
  fi
done

"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu130 'torch==2.14.1+cu130'
"$PYTHON" -m pip install \
  -r "$REPO_ROOT/requirements-dev.txt" \
  -r "$REPO_ROOT/requirements.txt" \
  pytest pytest-xdist pytest-timeout pytest-durations matplotlib einops ruff markdown-it-py Pygments
"$PYTHON" -m pip check
printf 'Dependencies ready. Build with: %s/scripts/build_tilelang.sh\n' "$REPO_ROOT"
