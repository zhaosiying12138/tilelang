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

exec "${PYTHON}" "${REPO_ROOT}/benchmark/z3_cost_model/compile_matrix.py" "$@"
