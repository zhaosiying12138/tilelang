"""Repeat all GDN search arms on independent cold-cache runs."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .autotune import ARMS, DEFAULT_TOP_K, Direction, run_gdn_search
except ImportError:
    from autotune import ARMS, DEFAULT_TOP_K, Direction, run_gdn_search


def run_repetitions(directions, repetitions, output_dir, warmup, repeat, seed):
    output_root = Path(output_dir).expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "repetition-summary.json"
    records = []
    try:
        for direction_name in directions:
            direction = Direction(direction_name)
            for repetition in repetitions:
                for arm in ARMS:
                    result = run_gdn_search(
                        direction,
                        arm,
                        DEFAULT_TOP_K[direction],
                        output_root / direction.value,
                        warmup,
                        repeat,
                        seed,
                        cache_mode="cold",
                        repetition=repetition,
                    )
                    records.append({
                        "direction": direction.value,
                        "repetition": repetition,
                        "arm": arm,
                        "summary": result.summary,
                    })
                    _write_json(summary_path, {
                        "status": "running",
                        "warmup": warmup,
                        "repeat": repeat,
                        "seed": seed,
                        "records": records,
                    })
    except BaseException as exc:
        _write_json(summary_path, {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "warmup": warmup,
            "repeat": repeat,
            "seed": seed,
            "records": records,
        })
        raise
    _write_json(summary_path, {
        "status": "ok",
        "warmup": warmup,
        "repeat": repeat,
        "seed": seed,
        "records": records,
    })
    return records


def _write_json(path, value):
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        import os

        os.fsync(stream.fileno())


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directions", default="fwd,bwd", help="Comma-separated directions: fwd,bwd")
    parser.add_argument("--repetitions", default="1,2", help="Cold repetition ids, e.g. 1,2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    directions = [item.strip() for item in args.directions.split(",") if item.strip()]
    repetitions = [int(item) for item in args.repetitions.split(",") if item.strip()]
    records = run_repetitions(directions, repetitions, args.output_dir, args.warmup, args.repeat, args.seed)
    print(f"Completed {len(records)} cold searches; summary: {Path(args.output_dir) / 'repetition-summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
