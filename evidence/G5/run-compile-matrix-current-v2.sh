#!/usr/bin/env bash
set -u -o pipefail
ROOT=/home/zhaosiying/codebase/tilelang-cuda-z3
RUN="$ROOT/evidence/G5/compile-matrix-current-v2"
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

run_step host "$ROOT/scripts/run_compile_matrix.sh" \
  --run --tier host --mode default --timeout 300 \
  --output "$RUN/host"

run_step cpu env LD_PRELOAD="$ROOT/build/lib/libstub_cuda.so" \
  "$ROOT/scripts/run_compile_matrix.sh" \
  --run --tier full --mode default --timeout 300 \
  --only testing/python/analysis/test_tilelang_fragment_loop_checker.py \
  --only testing/python/cuda/test_cuda_latency_model.py \
  --only testing/python/kernel/test_tilelang_kernel_fp8_gemm_arch_gate.py \
  --only testing/python/language/test_tilelang_language_fp8.py \
  --only testing/python/language/test_tilelang_language_persistent.py \
  --only testing/python/language/test_tilelang_language_strided_view.py \
  --only testing/python/runtime/test_tilelang_runtime_dynamic_shared_memory.py \
  --only testing/python/transform/test_tilelang_cuda_z3_scheduler.py \
  --only testing/python/transform/test_tilelang_transform_Inject_software_pipeline.py \
  --only testing/python/transform/test_tilelang_transform_im2col_fallback.py \
  --output "$RUN/cpu"

for index in {0..15}; do
  run_step "batch-$index" env LD_PRELOAD=/usr/lib/wsl/lib/libcuda.so \
    "$ROOT/scripts/run_compile_matrix_batch.sh" \
    "$index" 16 "$ROOT/evidence/G5/full-device-shard-plan-v1.json" "$RUN/batches"
done

run_step z3-opt-in env LD_PRELOAD="$ROOT/build/lib/libstub_cuda.so" \
  "$ROOT/scripts/run_compile_matrix.sh" \
  --run --tier full --mode z3-opt-in --timeout 300 \
  --only testing/python/cuda/test_compile_matrix_driver.py::test_z3_opt_in_pass_configs_compile_fixture \
  --output "$RUN/z3-opt-in"

run_step merge "$PY" "$ROOT/benchmark/z3_cost_model/merge_matrix.py" \
  --tier full --mode default \
  --manifest "$ROOT/manifest/compile_matrix.json" \
  --output "$RUN/merged" \
  "$RUN"/batches/batch-{0..15} \
  "$RUN/cpu"

printf '%d\n' "$aggregate_rc" > "$RUN/aggregate.status"
printf 'AGGREGATE rc=%d %s\n' "$aggregate_rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$aggregate_rc"
