#!/usr/bin/env bash
set -euo pipefail
set -x
cd /home/zhaosiying/codebase/tilelang-cuda-z3
source scripts/env.sh
free -g
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
ROOT=/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G6/gdn-stock-guided-screenshot-v1
test ! -e "$ROOT"
mkdir -p "$ROOT"
python benchmark/z3_cost_model/gdn/autotune.py \
  --direction fwd --arm stock-raw --cache-mode cold \
  --num-tokens 8192 --repetition 0 --warmup 10 --repeat 50 \
  --output-dir "$ROOT/stock" > "$ROOT/stock.log" 2>&1
python benchmark/z3_cost_model/gdn/autotune.py \
  --direction fwd --arm z3-cm-topk --cache-mode cold \
  --num-tokens 8192 --repetition 0 --warmup 10 --repeat 50 \
  --output-dir "$ROOT/guided" > "$ROOT/guided.log" 2>&1
python benchmark/z3_cost_model/gdn/analyze_results.py \
  --root "$ROOT" --output "$ROOT/analysis" --no-plots
cat "$ROOT/analysis/summary.csv"
git rev-parse HEAD
