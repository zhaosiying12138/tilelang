"""Direction-specific GDN ConfigSpaces connected to conservative Z3 pruning."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

try:
    from .config import (
        BLOCK_DV_DOMAIN,
        GROUP_REDUCE_BLOCK_SIZE_DOMAIN,
        PREPARE_H_NUM_STAGES_DOMAIN,
        Direction,
        GDNConfig,
        Status,
        evaluate_config,
    )
    from .resource import GDNResourceChecker
except ImportError:  # Direct execution with this directory on sys.path.
    from config import (
        BLOCK_DV_DOMAIN,
        GROUP_REDUCE_BLOCK_SIZE_DOMAIN,
        PREPARE_H_NUM_STAGES_DOMAIN,
        Direction,
        GDNConfig,
        Status,
        evaluate_config,
    )
    from resource import GDNResourceChecker

__all__ = ["constraint_builder", "make_config_space", "prune_direction"]


def _space_module():
    """Import the backend-neutral pruner lazily, without importing GPU paths."""

    from tilelang.autotuner import space

    return space


def _domains(direction: Direction) -> dict[str, list[int | str]]:
    if direction is Direction.FWD:
        return {
            "direction": [direction.value],
            "block_DV": list(BLOCK_DV_DOMAIN),
            "prepare_h_num_stages": [2],
            "group_reduce_block_size": [16],
        }
    return {
        "direction": [direction.value],
        "block_DV": [128],
        "prepare_h_num_stages": list(PREPARE_H_NUM_STAGES_DOMAIN),
        "group_reduce_block_size": list(GROUP_REDUCE_BLOCK_SIZE_DOMAIN),
    }


def constraint_builder(
    config: Mapping[str, Any],
    _z3: Any,
    _context: Any,
    *,
    resource_checker: GDNResourceChecker,
) -> Any:
    """Map evaluate_config to False/True/Indeterminate without guessing SAT."""

    values = dict(config)
    values["direction"] = Direction(values["direction"])
    evaluation = evaluate_config(GDNConfig(**values), resource_checker=resource_checker)
    if evaluation.status is Status.UNSAT:
        return False
    if evaluation.status is Status.SAT:
        return True

    unknown_reasons = [
        check.reason for check in (*evaluation.structural_checks, evaluation.resource_check) if check.status is Status.UNKNOWN
    ]
    return _space_module().Indeterminate(" ".join(unknown_reasons) or "GDN configuration feasibility is UNKNOWN.")


def make_config_space(direction: Direction | str, hardware_path: str | Path):
    """Build the fwd-4 or bwd-9 raw space with its resource constraint builder."""

    selected_direction = Direction(direction)
    checker = GDNResourceChecker(hardware_path)
    return _space_module().ConfigSpace(
        _domains(selected_direction),
        constraint_builder=partial(constraint_builder, resource_checker=checker),
    )


def prune_direction(
    direction: Direction | str,
    hardware_path: str | Path,
    *,
    pruner: Any = None,
):
    """Prune one direction; optional pruner injection keeps CPU tests inspectable."""

    space = make_config_space(direction, hardware_path)
    selected_pruner = _space_module().Z3Pruner() if pruner is None else pruner
    return selected_pruner.prune(space)
