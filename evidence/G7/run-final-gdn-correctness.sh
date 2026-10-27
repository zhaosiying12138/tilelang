#!/usr/bin/env bash
set -euo pipefail
cd /home/zhaosiying/codebase/tilelang-cuda-z3
source scripts/env.sh
python - <<'PY'
import subprocess

memory = subprocess.check_output(["free", "-g"], text=True)
mem_line = next(line for line in memory.splitlines() if line.startswith("Mem:"))
available_gib = int(mem_line.split()[-1])
if available_gib < 16:
    raise SystemExit("Refusing GDN correctness run: less than 16 GiB host memory available")
apps = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
    text=True,
)
if apps.strip():
    raise SystemExit(f"Refusing GDN correctness run while another GPU process is active:\n{apps}")
print(memory, end="")
print("GPU preflight: idle")
PY
python - <<'PY'
import json
from benchmark.z3_cost_model.gdn.config import Direction, GDNConfig
from benchmark.z3_cost_model.gdn.correctness import run_core_correctness

summaries = {
    "fwd": run_core_correctness(
        "fwd",
        config=GDNConfig(Direction.FWD, block_DV=16),
        determinism_iterations=2,
    ),
    "bwd": run_core_correctness(
        "bwd",
        config=GDNConfig(Direction.BWD, prepare_h_num_stages=2, group_reduce_block_size=16),
        determinism_iterations=2,
    ),
}
print(json.dumps({name: summary.as_dict() for name, summary in summaries.items()}, indent=2, sort_keys=True))
PY
