"""CPU-only, direction-specific GDN spaces and provable constraints.

Resource feasibility is deliberately deferred.  This module does not import
Torch, TileLang, IR tooling, ptxas, or device libraries, so ``--check-configs``
works on a host without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

BLOCK_DV_DOMAIN = (16, 32, 64, 128)
PREPARE_H_NUM_STAGES_DOMAIN = (1, 2, 3)
GROUP_REDUCE_BLOCK_SIZE_DOMAIN = (8, 16, 32)


class Direction(str, Enum):
    FWD = "fwd"
    BWD = "bwd"


RAW_CONFIGS_PER_DIRECTION = {
    Direction.FWD: len(BLOCK_DV_DOMAIN),
    Direction.BWD: len(PREPARE_H_NUM_STAGES_DOMAIN)
    * len(GROUP_REDUCE_BLOCK_SIZE_DOMAIN),
}


class Status(str, Enum):
    SAT = "SAT"
    UNSAT = "UNSAT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CoreContract:
    """Fixed first-round SM120 core contract; these are not tuning knobs."""

    auto_cp: bool = False
    dtype: str = "bf16"
    head_dim_k: int = 128
    head_dim_v: int = 128
    chunk_size: int = 32
    num_tokens: int = 128
    threads: int = 512
    warp_size: int = 32


CORE_CONTRACT = CoreContract()


@dataclass(frozen=True)
class GDNConfig:
    direction: Direction
    block_DV: int = 128
    prepare_h_num_stages: int = 2
    group_reduce_block_size: int = 16

    def parameters(self) -> dict[str, int | str]:
        return {
            "direction": self.direction.value,
            "block_DV": self.block_DV,
            "prepare_h_num_stages": self.prepare_h_num_stages,
            "group_reduce_block_size": self.group_reduce_block_size,
        }


def default_config(direction: Direction | str) -> GDNConfig:
    """Upstream defaults, with the direction made explicit."""

    return GDNConfig(Direction(direction))


def iter_raw_configs(direction: Direction | str):
    """Yield only real direction-specific knobs, without semantic duplicates."""

    direction = Direction(direction)
    if direction is Direction.FWD:
        for block_DV in BLOCK_DV_DOMAIN:
            yield GDNConfig(direction, block_DV=block_DV)
        return

    for prepare_h_num_stages in PREPARE_H_NUM_STAGES_DOMAIN:
        for group_reduce_block_size in GROUP_REDUCE_BLOCK_SIZE_DOMAIN:
            yield GDNConfig(
                direction,
                prepare_h_num_stages=prepare_h_num_stages,
                group_reduce_block_size=group_reduce_block_size,
            )


def raw_configs(direction: Direction | str) -> tuple[GDNConfig, ...]:
    return tuple(iter_raw_configs(direction))


def raw_config_count(direction: Direction | str) -> int:
    return RAW_CONFIGS_PER_DIRECTION[Direction(direction)]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    reason: str

    @classmethod
    def sat(cls, name: str, reason: str) -> "CheckResult":
        return cls(name, Status.SAT, reason)

    @classmethod
    def unsat(cls, name: str, reason: str) -> "CheckResult":
        return cls(name, Status.UNSAT, reason)

    @classmethod
    def unknown(cls, name: str, reason: str) -> "CheckResult":
        return cls(name, Status.UNKNOWN, reason)


def combine_status(statuses) -> Status:
    """UNSAT dominates; otherwise any UNKNOWN keeps the result unknown."""

    statuses = tuple(statuses)
    if any(status is Status.UNSAT for status in statuses):
        return Status.UNSAT
    if any(status is Status.UNKNOWN for status in statuses):
        return Status.UNKNOWN
    return Status.SAT


class ResourceChecker(Protocol):
    """Extension point for future IR, ptxas, and device-spec resource checks."""

    def check(self, config: GDNConfig, core: CoreContract) -> CheckResult:
        ...


class DeferredResourceChecker:
    """Default checker: make no shared-memory/register/occupancy claims."""

    def check(self, config: GDNConfig, core: CoreContract) -> CheckResult:
        del config, core
        return CheckResult.unknown(
            "resource_feasibility",
            "No IR/ptxas/device-spec resource evaluator is connected; "
            "shared-memory, register, and launch feasibility remain unknown.",
        )


@dataclass(frozen=True)
class ConfigEvaluation:
    config: GDNConfig
    structural_checks: tuple[CheckResult, ...]
    resource_check: CheckResult

    @property
    def structural_status(self) -> Status:
        return combine_status(check.status for check in self.structural_checks)

    @property
    def status(self) -> Status:
        return combine_status(
            [self.structural_status, self.resource_check.status]
        )

    def as_dict(self) -> dict:
        return {
            **self.config.parameters(),
            "structural_status": self.structural_status.value,
            "resource_status": self.resource_check.status.value,
            "status": self.status.value,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "reason": check.reason,
                }
                for check in (*self.structural_checks, self.resource_check)
            ],
        }


def _core_checks(core: CoreContract) -> list[CheckResult]:
    fixed = (
        core.auto_cp is False
        and core.dtype == "bf16"
        and core.head_dim_k == 128
        and core.head_dim_v == 128
        and core.chunk_size == 32
        and core.num_tokens == 128
    )
    core_result = (
        CheckResult.sat(
            "fixed_core_contract",
            "auto_cp=False, bf16, K=DV=128, chunk=32, and T=128 are fixed.",
        )
        if fixed
        else CheckResult.unsat(
            "fixed_core_contract",
            "The first-round core requires auto_cp=False, bf16, K=DV=128, "
            "chunk=32, and T=128.",
        )
    )

    protocol = core.threads == 512 and core.warp_size == 32
    thread_result = (
        CheckResult.sat(
            "thread_protocol",
            "The vendored SM120 warp/mbarrier protocol retains 512 threads "
            "(16 warps of 32 threads).",
        )
        if protocol and core.threads % core.warp_size == 0
        else CheckResult.unsat(
            "thread_protocol",
            "The fixed SM120 protocol requires threads=512 and warp_size=32; "
            "changing that protocol is outside this configuration space.",
        )
    )
    return [core_result, thread_result]


def structural_checks(
    config: GDNConfig, core: CoreContract = CORE_CONTRACT
) -> tuple[CheckResult, ...]:
    """Check only facts established by the fixed shapes and source structure."""

    checks = _core_checks(core)

    valid_block = (
        isinstance(config.block_DV, int)
        and not isinstance(config.block_DV, bool)
        and config.block_DV > 0
        and core.head_dim_v % config.block_DV == 0
    )
    checks.append(
        CheckResult.sat(
            "block_DV_divisibility",
            f"DV={core.head_dim_v} is divisible by block_DV={config.block_DV}.",
        )
        if valid_block
        else CheckResult.unsat(
            "block_DV_divisibility",
            f"block_DV must be a positive integer divisor of DV={core.head_dim_v}; "
            f"got {config.block_DV!r}.",
        )
    )

    valid_stages = (
        isinstance(config.prepare_h_num_stages, int)
        and not isinstance(config.prepare_h_num_stages, bool)
        and config.prepare_h_num_stages >= 1
    )
    checks.append(
        CheckResult.sat(
            "prepare_h_stage_count",
            "prepare_h_num_stages is a positive pipeline-stage count.",
        )
        if valid_stages
        else CheckResult.unsat(
            "prepare_h_stage_count",
            "prepare_h indexes stage buffers and barriers modulo num_stages, "
            "so prepare_h_num_stages must be an integer >= 1.",
        )
    )

    valid_group = (
        isinstance(config.group_reduce_block_size, int)
        and not isinstance(config.group_reduce_block_size, bool)
        and config.group_reduce_block_size > 0
    )
    if not valid_group:
        group_result = CheckResult.unsat(
            "group_reduce_token_tiling",
            "group_reduce_block_size must be a positive integer; "
            f"got {config.group_reduce_block_size!r}.",
        )
    elif (
        config.group_reduce_block_size != 16
        and core.num_tokens % config.group_reduce_block_size != 0
    ):
        group_result = CheckResult.unsat(
            "group_reduce_token_tiling",
            "Non-default group_reduce_block_size requires T divisibility; "
            f"T={core.num_tokens}, block_size={config.group_reduce_block_size}.",
        )
    else:
        group_result = CheckResult.sat(
            "group_reduce_token_tiling",
            "Default 16 preserves upstream behavior, or the non-default "
            f"block size divides T={core.num_tokens}.",
        )
    checks.append(group_result)
    return tuple(checks)


def evaluate_config(
    config: GDNConfig,
    core: CoreContract = CORE_CONTRACT,
    resource_checker: ResourceChecker | None = None,
) -> ConfigEvaluation:
    checker = DeferredResourceChecker() if resource_checker is None else resource_checker
    resource_result = checker.check(config, core)
    if not isinstance(resource_result, CheckResult):
        raise TypeError("ResourceChecker.check must return CheckResult")
    return ConfigEvaluation(config, structural_checks(config, core), resource_result)


def evaluate_configs(
    direction: Direction | str,
    core: CoreContract = CORE_CONTRACT,
    resource_checker: ResourceChecker | None = None,
) -> tuple[ConfigEvaluation, ...]:
    return tuple(
        evaluate_config(config, core, resource_checker)
        for config in iter_raw_configs(direction)
    )


def parameter_consumers(direction: Direction | str) -> dict[str, str]:
    """Describe real consumers and fixed fields without inventing feasibility."""

    direction = Direction(direction)
    if direction is Direction.FWD:
        return {
            "block_DV": "fused_gdr_fwd",
            "prepare_h_num_stages": (
                "fixed at 2; no-CP public forward has no prepare_h consumer"
            ),
            "group_reduce_block_size": "fixed at 16; no forward consumer",
        }
    return {
        "block_DV": "fixed at 128; fused_bwd has no block_DV parameter",
        "prepare_h_num_stages": "fused_gdr_h and cp_preprocess_bwd",
        "group_reduce_block_size": (
            "group_reduce_vector for dq/dk when Hg<H; core GQA uses Hg=2, H=4"
        ),
    }
