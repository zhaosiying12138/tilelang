#!/usr/bin/env bash
set -e
set -x
cd /home/zhaosiying/codebase/tilelang-cuda-z3
source scripts/env.sh
/usr/local/cuda-13.3/bin/cuobjdump --dump-resource-usage evidence/G7/best-gemm-v1/kernel.cubin
sha256sum evidence/G7/best-gemm-v1/source.cu evidence/G7/best-gemm-v1/kernel.cubin
/home/zhaosiying/miniforge3/envs/tilelang-cu133/bin/python -m json.tool evidence/G7/best-gemm-v1/metadata.json
git rev-parse HEAD
