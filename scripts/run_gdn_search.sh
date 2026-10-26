#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${TILELANG_PYTHON:-}" ]]; then
  PYTHON="${TILELANG_PYTHON}"
elif [[ -x "${HOME}/miniforge3/envs/tilelang-cu133/bin/python" ]]; then
  PYTHON="${HOME}/miniforge3/envs/tilelang-cu133/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

# Keep candidate initialization and benchmarking serial.  SearchExperiment owns
# per-run cache paths; this script never deletes or reuses a shared user cache.
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/3rdparty/tvm/python:${PYTHONPATH:-}"
export TILELANG_AUTO_TUNING_CPU_COUNTS=1
export TILELANG_AUTO_TUNING_MAX_CPU_COUNT=1
export TILELANG_DEFAULT_TARGET="${TILELANG_DEFAULT_TARGET:-cuda}"
export TILELANG_EXECUTION_BACKEND="${TILELANG_EXECUTION_BACKEND:-tvm_ffi}"
export PYTHONUNBUFFERED=1

exec "${PYTHON}" "${REPO_ROOT}/benchmark/z3_cost_model/gdn/autotune.py" "$@"
