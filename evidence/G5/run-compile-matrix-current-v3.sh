#!/usr/bin/env bash
set -u -o pipefail
ROOT=/home/zhaosiying/codebase/tilelang-cuda-z3
RUN="$ROOT/evidence/G5/compile-matrix-current-v3"
PY=/home/zhaosiying/miniforge3/envs/tilelang-cu133/bin/python

if [[ -e "$RUN" ]]; then
  printf 'Refusing existing output root: %s\n' "$RUN" >&2
  exit 2
fi
mkdir -p "$RUN"
cd "$ROOT" || exit 2
source "$ROOT/scripts/env.sh" || exit 2
export TVM_LIBRARY_PATH="$ROOT/build/lib"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH}"
aggregate_rc=0

run_step() {
  local name=$1
  shift
  printf 'START %s %s\n' "$name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$@" > "$RUN/$name.driver.log" 2>&1
  local rc=$?
  printf 'END %s rc=%d %s\n' "$name" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ $rc -ne 0 ]]; then
    aggregate_rc=1
  fi
}

for index in {0..15}; do
  run_step "batch-$index" env LD_PRELOAD=/usr/lib/wsl/lib/libcuda.so \
    "$ROOT/scripts/run_compile_matrix_batch.sh" \
    "$index" 16 "$ROOT/evidence/G5/full-device-shard-plan-v1.json" "$RUN/batches"
done

run_step merge "$PY" "$ROOT/benchmark/z3_cost_model/merge_matrix.py" \
  --tier full --mode default \
  --manifest "$ROOT/manifest/compile_matrix.json" \
  --output "$RUN/merged" \
  "$RUN"/batches/batch-{0..15} \
  "$ROOT/evidence/G5/compile-matrix-current-v2/cpu"

printf '%d\n' "$aggregate_rc" > "$RUN/aggregate.status"
printf 'AGGREGATE rc=%d %s\n' "$aggregate_rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$aggregate_rc"
