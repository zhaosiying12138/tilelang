"""Evidence-backed GDN shared-memory checks with conservative UNKNOWNs.

This module is CPU-only.  It reads an existing hardware manifest and never
queries a GPU, compiles a kernel, or infers register/launch feasibility.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

try:
    from .config import CORE_CONTRACT, CheckResult, CoreContract, Direction, GDNConfig
except ImportError:  # Direct execution with this directory on sys.path.
    from config import CORE_CONTRACT, CheckResult, CoreContract, Direction, GDNConfig

FWD_SHARED_BASE_BYTES = 38_912
FWD_SHARED_PER_BLOCK_DV_BYTES = 576
PREPARE_H_SHARED_BASE_BYTES = 57_344
PREPARE_H_SHARED_PER_STAGE_BYTES = 18_432
HARDWARE_OPTIN_CAP_KEY = "shared_memory_per_block_optin_bytes"

__all__ = [
    "FWD_SHARED_BASE_BYTES",
    "FWD_SHARED_PER_BLOCK_DV_BYTES",
    "PREPARE_H_SHARED_BASE_BYTES",
    "PREPARE_H_SHARED_PER_STAGE_BYTES",
    "GDNResourceChecker",
    "forward_dynamic_shared_bytes",
    "prepare_h_dynamic_shared_bytes",
    "load_hardware_shared_memory_cap",
]


def forward_dynamic_shared_bytes(block_DV: int) -> int:
    """Return the G5-validated fwd dynamic-shared-memory formula in bytes."""

    if isinstance(block_DV, bool) or not isinstance(block_DV, int):
        raise TypeError(f"block_DV must be an integer, got {block_DV!r}")
    if block_DV <= 0:
        raise ValueError(f"block_DV must be positive, got {block_DV}")
    return FWD_SHARED_BASE_BYTES + FWD_SHARED_PER_BLOCK_DV_BYTES * block_DV


def prepare_h_dynamic_shared_bytes(stages: int) -> int:
    """Return the G5-validated bwd prepare_h dynamic shared bytes."""

    if isinstance(stages, bool) or not isinstance(stages, int):
        raise TypeError(f"prepare_h_num_stages must be an integer, got {stages!r}")
    if stages <= 0:
        raise ValueError(f"prepare_h_num_stages must be positive, got {stages}")
    return PREPARE_H_SHARED_BASE_BYTES + PREPARE_H_SHARED_PER_STAGE_BYTES * stages


def load_hardware_shared_memory_cap(hardware_path: str | Path) -> int:
    """Load the opt-in per-block shared-memory cap, failing closed if absent."""

    path = Path(hardware_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("device"), Mapping):
        raise ValueError(f"Hardware manifest {path} has no device object")
    device = payload["device"]
    if HARDWARE_OPTIN_CAP_KEY not in device:
        raise ValueError(f"Hardware manifest {path} lacks device.{HARDWARE_OPTIN_CAP_KEY}; no shared-memory cap fallback is permitted")
    cap = device[HARDWARE_OPTIN_CAP_KEY]
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise ValueError(f"Hardware manifest {path} has invalid device.{HARDWARE_OPTIN_CAP_KEY}: {cap!r}")
    return cap


class GDNResourceChecker:
    """Apply only validated fwd/prepare_h formulas; never invent resource SAT."""

    def __init__(self, hardware_path: str | Path):
        self.hardware_path = Path(hardware_path)
        self.cap_bytes = load_hardware_shared_memory_cap(self.hardware_path)

    def _capacity_result(self, scope: str, dynamic_bytes: int, formula: str) -> CheckResult:
        detail = f"{scope} dynamic_shared_bytes={dynamic_bytes} ({formula}), opt-in cap={self.cap_bytes}"
        if dynamic_bytes > self.cap_bytes:
            return CheckResult.unsat(
                "resource_feasibility",
                f"{detail}: required dynamic shared memory exceeds the cap.",
            )
        return CheckResult.unknown(
            "resource_feasibility",
            f"{detail}: the shared-memory cap is not exceeded, but register usage and other launch feasibility remain UNKNOWN.",
        )

    def check(self, config: GDNConfig, core: CoreContract = CORE_CONTRACT) -> CheckResult:
        direction = Direction(config.direction)
        if core != CORE_CONTRACT:
            return CheckResult.unknown(
                "resource_feasibility",
                "The GDN shared-memory formulas are validated only for the fixed core contract; this core is outside that evidence scope.",
            )

        if direction is Direction.BWD:
            stages = config.prepare_h_num_stages
            if isinstance(stages, bool) or not isinstance(stages, int) or stages <= 0:
                return CheckResult.unknown(
                    "resource_feasibility",
                    f"The prepare_h shared-memory formula requires a positive integer stage count; got {stages!r}. Structural checks decide validity.",
                )
            dynamic_bytes = prepare_h_dynamic_shared_bytes(stages)
            return self._capacity_result(
                "bwd prepare_h",
                dynamic_bytes,
                "57344+18432*prepare_h_num_stages",
            )

        block_DV = config.block_DV
        if isinstance(block_DV, bool) or not isinstance(block_DV, int) or block_DV <= 0:
            return CheckResult.unknown(
                "resource_feasibility",
                f"The fwd shared-memory formula requires a positive integer block_DV; got {block_DV!r}. Structural checks decide validity.",
            )

        dynamic_bytes = forward_dynamic_shared_bytes(block_DV)
        return self._capacity_result("fwd", dynamic_bytes, "38912+576*block_DV")
