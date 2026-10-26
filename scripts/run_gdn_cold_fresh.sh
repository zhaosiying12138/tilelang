#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TILELANG_REPO_ROOT="$REPO_ROOT"
# shellcheck source=env.sh
source "$REPO_ROOT/scripts/env.sh"

if [[ $# -ne 3 ]]; then
  printf 'Usage: %s DIRECTION REPETITION OUTPUT_ROOT\n' "$0" >&2
  exit 2
fi
DIRECTION="$1"
REPETITION="$2"
OUTPUT_ROOT="$(realpath -m "$3")"
if [[ "$DIRECTION" != fwd && "$DIRECTION" != bwd ]]; then
  printf 'Direction must be fwd or bwd: %s\n' "$DIRECTION" >&2
  exit 2
fi
if [[ ! "$REPETITION" =~ ^[0-9]+$ ]]; then
  printf 'Repetition must be a non-negative integer: %s\n' "$REPETITION" >&2
  exit 2
fi
case "$OUTPUT_ROOT" in
  "$REPO_ROOT"/evidence/G6/*) ;;
  *)
    printf 'Output must be a run-owned subdirectory of evidence/G6: %s\n' "$OUTPUT_ROOT" >&2
    exit 2
    ;;
esac
mkdir -p "$OUTPUT_ROOT/$DIRECTION"

"$TILELANG_PYTHON" - "$REPO_ROOT" "$OUTPUT_ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

repo, output = map(Path, sys.argv[1:])
path = output / "protocol.json"
script = repo / "scripts/run_gdn_cold_fresh.sh"
script_sha256 = hashlib.sha256(script.read_bytes()).hexdigest()
required = {
    "schema": "tilelang.gdn.cold-fresh-protocol/v1",
    "directions": ["fwd", "bwd"],
    "repetitions": [0, 1, 2],
    "arms": ["stock-raw", "z3-legal-exhaustive", "z3-cm-topk", "budget-raw", "budget-random"],
    "cache_track": "cold",
    "warmup": 10,
    "repeat": 50,
    "fresh_process_per_arm_repetition": True,
    "script_sha256": script_sha256,
}
if path.exists():
    protocol = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in required.items():
        if protocol.get(key) != expected:
            raise SystemExit(f"Protocol mismatch for {key}: {protocol.get(key)!r} != {expected!r}")
else:
    payload = {
        **required,
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "shared_cache_deletion": False,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

preflight() {
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
  local processes
  processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)"
  if [[ -n "${processes//[[:space:]]/}" ]]; then
    printf 'Another GPU compute process is active; refusing to start:\n%s\n' "$processes" >&2
    return 1
  fi
}

for arm in stock-raw z3-legal-exhaustive z3-cm-topk budget-raw budget-random; do
  preflight
  LOG="$OUTPUT_ROOT/$DIRECTION-rep-$REPETITION-$arm.txt"
  if [[ -e "$LOG" ]]; then
    printf 'Refusing to overwrite an existing run log: %s\n' "$LOG" >&2
    exit 2
  fi
  printf 'START direction=%s repetition=%s arm=%s cold fresh-process warmup=10 repeat=50\n' \
    "$DIRECTION" "$REPETITION" "$arm"
  "$TILELANG_PYTHON" "$REPO_ROOT/benchmark/z3_cost_model/gdn/autotune.py" \
    --direction "$DIRECTION" \
    --arm "$arm" \
    --cache-mode cold \
    --repetition "$REPETITION" \
    --warmup 10 \
    --repeat 50 \
    --output-dir "$OUTPUT_ROOT/$DIRECTION" \
    > "$LOG" 2>&1
  printf 'DONE direction=%s repetition=%s arm=%s\n' "$DIRECTION" "$REPETITION" "$arm"
done
