#!/usr/bin/env bash
set -euo pipefail
set -x
cd /home/zhaosiying/codebase/tilelang-cuda-z3
source scripts/env.sh
OUT=/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G7/screenshots/gdn-confirmation-analysis-v1
test ! -e "$OUT"
python benchmark/z3_cost_model/gdn/analyze_confirmation.py \
  --cold-analysis evidence/G6/gdn-heldout-t8192-v1/analysis/analysis.json \
  --confirmation-root evidence/G6/gdn-heldout-t8192-confirmation-v1 \
  --output "$OUT"
cat "$OUT/confirmation-comparisons.csv"
git rev-parse HEAD
