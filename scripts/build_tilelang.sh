#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=./env.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"

BUILD_JOBS="${BUILD_JOBS:-2}"
if [[ ! "$BUILD_JOBS" =~ ^[1-4]$ ]]; then
  printf 'BUILD_JOBS must be an integer from 1 to 4 on this WSL host.\n' >&2
  exit 2
fi
export CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
export MAX_JOBS="$BUILD_JOBS"
export CMAKE_ARGS="-DUSE_CUDA=ON -DUSE_ASCEND=ON -DUSE_ROCM=OFF -DUSE_METAL=OFF -DUSE_LLVM=OFF -DUSE_Z3=ON -DTILELANG_USE_ASCEND_STUBS=ON -DCUDAToolkit_ROOT=$CUDA_HOME -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc"

LOG_FILE="${BUILD_LOG:-$TILELANG_REPO_ROOT/evidence/G0/build-$(date -u +%Y%m%dT%H%M%SZ).log}"
mkdir -p "$(dirname -- "$LOG_FILE")"
"$TILELANG_PYTHON" -m pip install -e "$TILELANG_REPO_ROOT" -v \
  --no-build-isolation --no-deps 2>&1 | tee "$LOG_FILE"
"$TILELANG_PYTHON" -m pip check
