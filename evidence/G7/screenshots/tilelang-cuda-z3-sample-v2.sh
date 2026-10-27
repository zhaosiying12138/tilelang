#!/usr/bin/env bash
set -e
set -x
cd /home/zhaosiying/codebase/tilelang-cuda-z3
source scripts/env.sh
free -g
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem --format=csv,noheader,nounits
/home/zhaosiying/miniforge3/envs/tilelang-cu133/bin/python -m pytest testing/python/cuda/test_result_visualization.py -q
git rev-parse HEAD
