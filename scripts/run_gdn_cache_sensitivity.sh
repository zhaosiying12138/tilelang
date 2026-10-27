#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TILELANG_REPO_ROOT="$REPO_ROOT"
# shellcheck source=env.sh
source "$REPO_ROOT/scripts/env.sh"
OUTPUT_ROOT="${1:-$REPO_ROOT/evidence/G6/gdn-cache-sensitivity-v1}"
export GDN_NUM_TOKENS="${GDN_NUM_TOKENS:-4096}"
if [[ ! "$GDN_NUM_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'GDN_NUM_TOKENS must be a positive integer: %s\n' "$GDN_NUM_TOKENS" >&2
  exit 2
fi

if [[ -e "$OUTPUT_ROOT" ]]; then
  printf 'Refusing an existing cache-sensitivity output path: %s\n' "$OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"

"$TILELANG_PYTHON" - "$REPO_ROOT" "$OUTPUT_ROOT" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

repo, output = map(Path, sys.argv[1:])
script = repo / "scripts/run_gdn_cache_sensitivity.sh"
payload = {
    "schema": "tilelang.gdn.cache-sensitivity-protocol/v1",
    "created": datetime.now(timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
    "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    "directions": ["fwd", "bwd"],
    "tracks": ["no-cache", "warm"],
    "arms": ["stock-raw", "z3-cm-topk"],
    "warmup": 10,
    "repeat": 50,
    "num_tokens": int(os.environ["GDN_NUM_TOKENS"]),
    "fresh_process_per_run": True,
    "shared_cache_deletion": False,
}
(output / "protocol.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

preflight() {
  free -g
  "$TILELANG_PYTHON" - <<'PY'
from pathlib import Path

fields = {
    line.split(":", 1)[0]: int(line.split()[1])
    for line in Path("/proc/meminfo").read_text().splitlines()
    if line.startswith("MemAvailable:")
}
available_kib = fields["MemAvailable"]
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

for direction in fwd bwd; do
  for track in no-cache warm; do
    for arm in stock-raw z3-cm-topk; do
      preflight
      printf 'START direction=%s track=%s arm=%s\n' "$direction" "$track" "$arm"
      "$TILELANG_PYTHON" "$REPO_ROOT/benchmark/z3_cost_model/gdn/autotune.py" \
        --direction "$direction" \
        --arm "$arm" \
        --cache-mode "$track" \
        --num-tokens "$GDN_NUM_TOKENS" \
        --warmup 10 \
        --repeat 50 \
        --output-dir "$OUTPUT_ROOT/$direction/$track" \
        > "$OUTPUT_ROOT/$direction-$track-$arm.txt" 2>&1
      printf 'DONE direction=%s track=%s arm=%s\n' "$direction" "$track" "$arm"
    done
  done
done
