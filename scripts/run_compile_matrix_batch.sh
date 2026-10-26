#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TILELANG_REPO_ROOT="$REPO_ROOT"
# shellcheck source=env.sh
source "$REPO_ROOT/scripts/env.sh"

MODE=run
if [[ "${1:-}" == "--print" ]]; then
  MODE=print
  shift
fi
if [[ $# -lt 2 || $# -gt 4 ]]; then
  printf 'Usage: %s [--print] BATCH_INDEX BATCH_COUNT [PLAN_JSON] [OUTPUT_ROOT]\n' "$0" >&2
  exit 2
fi
BATCH_INDEX="$1"
BATCH_COUNT="$2"
PLAN="${3:-$REPO_ROOT/evidence/G5/full-default-shard-plan-v1.json}"
OUTPUT_ROOT="$(realpath -m "${4:-$REPO_ROOT/evidence/G5/compile-matrix-full-run-v1/batches}")"

mapfile -t FILES < <("$TILELANG_PYTHON" - "$PLAN" "$BATCH_INDEX" "$BATCH_COUNT" <<'PY'
import json
from pathlib import Path
import sys

plan_path = Path(sys.argv[1])
index, count = int(sys.argv[2]), int(sys.argv[3])
if count < 1 or count > 64 or not 0 <= index < count:
    raise SystemExit(f"Invalid batch {index}/{count}")
plan = json.loads(plan_path.read_text(encoding="utf-8"))
files = [item["path"] for position, item in enumerate(plan["files"]) if position % count == index]
if not files:
    raise SystemExit(f"Empty batch {index}/{count}")
print("\n".join(files))
PY
)

if [[ "$MODE" == print ]]; then
  printf '%s\n' "${FILES[@]}"
  exit 0
fi
case "$OUTPUT_ROOT" in
  "$REPO_ROOT"/evidence/G5/*) ;;
  *)
    printf 'Output must be under evidence/G5: %s\n' "$OUTPUT_ROOT" >&2
    exit 2
    ;;
esac
mkdir -p "$OUTPUT_ROOT"
OUTPUT="$OUTPUT_ROOT/batch-$BATCH_INDEX"
DRIVER_LOG="$OUTPUT_ROOT/batch-$BATCH_INDEX.driver.log"
if [[ -e "$OUTPUT" || -e "$DRIVER_LOG" ]]; then
  printf 'Refusing an existing matrix batch or log: %s\n' "$OUTPUT" >&2
  exit 2
fi

free -g
"$TILELANG_PYTHON" - <<'PY'
from pathlib import Path

available_kib = int(next(
    line.split()[1]
    for line in Path("/proc/meminfo").read_text().splitlines()
    if line.startswith("MemAvailable:")
))
print(f"MemAvailable={available_kib} KiB; required={16 * 1024 * 1024} KiB")
if available_kib < 16 * 1024 * 1024:
    raise SystemExit("Insufficient host memory for GPU work")
PY
nvidia-smi --query-gpu=name,memory.total,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem --format=csv,noheader,nounits
processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)"
if [[ -n "${processes//[[:space:]]/}" ]]; then
  printf 'Another GPU compute process is active; refusing to start:\n%s\n' "$processes" >&2
  exit 1
fi

ONLY_ARGS=()
for file in "${FILES[@]}"; do
  ONLY_ARGS+=(--only "$file")
done
printf 'START matrix batch=%s/%s files=%s output=%s\n' "$BATCH_INDEX" "$BATCH_COUNT" "${#FILES[@]}" "$OUTPUT"
"$TILELANG_PYTHON" "$REPO_ROOT/benchmark/z3_cost_model/compile_matrix.py" \
  --run --tier full --mode default \
  "${ONLY_ARGS[@]}" \
  --output "$OUTPUT" \
  > "$DRIVER_LOG" 2>&1
printf 'DONE matrix batch=%s/%s\n' "$BATCH_INDEX" "$BATCH_COUNT"
