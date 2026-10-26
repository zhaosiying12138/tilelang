"""GDN port CLI: host-only config checks and an explicit GPU correctness entry."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

try:
    from .config import (
        BLOCK_DV_DOMAIN,
        CORE_CONTRACT,
        GROUP_REDUCE_BLOCK_SIZE_DOMAIN,
        PREPARE_H_NUM_STAGES_DOMAIN,
        Direction,
        GDNConfig,
        Status,
        evaluate_configs,
        parameter_consumers,
        raw_config_count,
    )
except ImportError:  # python path/to/gdn/cli.py
    from config import (
        BLOCK_DV_DOMAIN,
        CORE_CONTRACT,
        GROUP_REDUCE_BLOCK_SIZE_DOMAIN,
        PREPARE_H_NUM_STAGES_DOMAIN,
        Direction,
        GDNConfig,
        Status,
        evaluate_configs,
        parameter_consumers,
        raw_config_count,
    )


def _raw_domains(direction: Direction) -> dict:
    if direction is Direction.FWD:
        return {"block_DV": list(BLOCK_DV_DOMAIN)}
    return {
        "prepare_h_num_stages": list(PREPARE_H_NUM_STAGES_DOMAIN),
        "group_reduce_block_size": list(GROUP_REDUCE_BLOCK_SIZE_DOMAIN),
    }


def check_configs_payload() -> dict:
    payload = {
        "core": asdict(CORE_CONTRACT),
        "raw_configs_per_direction": {
            direction.value: raw_config_count(direction) for direction in Direction
        },
        "fixed_parameters": {
            "fwd": {
                "prepare_h_num_stages": 2,
                "group_reduce_block_size": 16,
            },
            "bwd": {"block_DV": 128},
        },
        "resource_policy": (
            "Structural SAT is not resource SAT: without IR/ptxas/device-spec "
            "checks, overall feasibility remains UNKNOWN. Fixed fields and "
            "semantic de-duplication are not UNSAT decisions."
        ),
        "directions": {},
    }
    for direction in Direction:
        evaluations = evaluate_configs(direction)
        status_counts = {
            status.value: sum(item.status is status for item in evaluations)
            for status in Status
        }
        structural_counts = {
            status.value: sum(
                item.structural_status is status for item in evaluations
            )
            for status in Status
        }
        payload["directions"][direction.value] = {
            "count": len(evaluations),
            "raw_domains": _raw_domains(direction),
            "status_counts": status_counts,
            "structural_status_counts": structural_counts,
            "parameter_consumers": parameter_consumers(direction),
            "configs": [item.as_dict() for item in evaluations],
        }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--check-configs",
        action="store_true",
        help="Check the real 4-entry fwd and 9-entry bwd spaces (CPU only).",
    )
    action.add_argument(
        "--correctness",
        choices=("fwd", "bwd", "all"),
        help="Run public pipelines against float64 references (requires CUDA GPU).",
    )
    parser.add_argument("--block-dv", type=int, default=128)
    parser.add_argument("--prepare-h-num-stages", type=int, default=2)
    parser.add_argument("--group-reduce-block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--determinism-iters",
        type=int,
        default=0,
        help=(
            "Repeated public-pipeline checks per case; use 1000 to match the "
            "upstream slow determinism tests (default: 0)."
        ),
    )
    return parser


def _correctness_config(args) -> GDNConfig:
    if args.correctness == "fwd":
        if args.prepare_h_num_stages != 2 or args.group_reduce_block_size != 16:
            raise ValueError(
                "fwd fixes prepare_h_num_stages=2 and "
                "group_reduce_block_size=16; only block_DV is tunable."
            )
    if args.correctness == "bwd" and args.block_dv != 128:
        raise ValueError("bwd fixes block_DV=128; fused_bwd has no block_DV parameter.")

    direction = Direction.FWD if args.correctness == "all" else Direction(args.correctness)
    return GDNConfig(
        direction,
        block_DV=args.block_dv,
        prepare_h_num_stages=args.prepare_h_num_stages,
        group_reduce_block_size=args.group_reduce_block_size,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_configs:
        print(json.dumps(check_configs_payload(), indent=2, sort_keys=True))
        return 0

    if args.determinism_iters < 0:
        raise ValueError("--determinism-iters must be >= 0")

    # Deliberately lazy: --check-configs never imports Torch, TileLang, or CUDA.
    try:
        from .correctness import run_core_correctness
    except ImportError:
        from correctness import run_core_correctness

    summary = run_core_correctness(
        args.correctness,
        _correctness_config(args),
        seed=args.seed,
        determinism_iterations=args.determinism_iters,
    )
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
