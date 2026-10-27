#!/usr/bin/env bash
set -e
set -x
cd /home/zhaosiying/codebase/tilelang-cuda-z3
source scripts/env.sh
/usr/local/cuda-13.3/bin/cuobjdump --dump-resource-usage evidence/G7/best-gemm-v1/kernel.cubin
/home/zhaosiying/miniforge3/envs/tilelang-cu133/bin/python -c "import json; m=json.load(open('evidence/G7/best-gemm-v1/metadata.json')); print(m['config']); print(m['cubin_sha256']); print(m['source_sha256'])"
git rev-parse HEAD
