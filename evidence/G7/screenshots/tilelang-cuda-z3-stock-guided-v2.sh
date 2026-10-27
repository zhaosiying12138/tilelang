#!/usr/bin/env bash
set -euo pipefail
set -x
cd /home/zhaosiying/codebase/tilelang-cuda-z3
source scripts/env.sh
ROOT=/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G6/gdn-stock-guided-screenshot-v1
OUT=/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G7/screenshots/stock-guided-analysis-v2
test -d "$ROOT"
test ! -e "$OUT"
python benchmark/z3_cost_model/gdn/analyze_results.py \
  --root "$ROOT" --output "$OUT" --no-plots
cat "$OUT/summary.csv"
git rev-parse HEAD
