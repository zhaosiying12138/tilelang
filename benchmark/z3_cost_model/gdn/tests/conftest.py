# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Pytest wiring: arch/capability gating.

Capability gating reads the single table in :mod:`arch_utils` instead of spelling arch
names out per test, so a test says *what it needs* (``cp_inter_intra``, ``needs_bwd``,
``multigpu``) rather than *where it runs*.
"""
import os
import sys

import pytest
import torch

# Port adaptation: resolve the vendored package and upstream flat test imports
# without installing flash_qla or replacing any already-imported module.
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(os.path.dirname(TEST_DIR), "vendor")
for _path in (TEST_DIR, os.path.dirname(TEST_DIR), VENDOR_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from arch_utils import ARCH, GPU_AVAILABLE, supports

requires_gpu = pytest.mark.gpu
requires_hopper = pytest.mark.hopper
requires_blackwell = pytest.mark.blackwell
requires_sm120 = pytest.mark.sm120

def pytest_collection_modifyitems(config, items):
    device_count = torch.cuda.device_count()
    for item in items:
        if "gpu" in item.keywords and not GPU_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="CUDA GPU not available"))
        if "hopper" in item.keywords and ARCH != "SM90":
            item.add_marker(pytest.mark.skip(reason="Hopper (SM90) GPU required"))
        if "blackwell" in item.keywords and ARCH not in ["SM100", "SM103"]:
            item.add_marker(pytest.mark.skip(reason="Blackwell (SM100 or SM103) GPU required"))
        if "sm120" in item.keywords and ARCH not in ["SM120", "SM121"]:
            item.add_marker(pytest.mark.skip(reason="Blackwell SM120 GPU required"))
        # Capability-based gating (see arch_utils.ARCH_CAPS).
        if "cp_inter_intra" in item.keywords and not supports("inter_intra"):
            item.add_marker(pytest.mark.skip(
                reason=f"combined inter+intra CP is not implemented on {ARCH}"))
        if "needs_bwd" in item.keywords and not supports("bwd"):
            item.add_marker(pytest.mark.skip(
                reason=f"backward kernels are not implemented on {ARCH}"))
        if "multigpu" in item.keywords and device_count < 2:
            item.add_marker(pytest.mark.skip(
                reason=f"multi-GPU test requires >= 2 GPUs, found {device_count}"))
