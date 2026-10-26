#!/usr/bin/env python3
"""Write a CUDA hardware attribute spec without importing TileLang itself."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_HARDWARE_MODULE_PATH = _REPOSITORY_ROOT / "tilelang" / "cuda" / "hardware.py"


def _load_hardware_module() -> Any:
    # Import the standalone file directly: importing tilelang.cuda.hardware via
    # its package would execute TileLang/CUDA __init__ modules.  Keeping help and
    # non-CUDA error handling independent of a built TileLang is intentional.
    spec = importlib.util.spec_from_file_location("_tilelang_cuda_hardware_standalone", _HARDWARE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the CUDA hardware probe from {_HARDWARE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HARDWARE = _load_hardware_module()
CudaHardwareProbeError = _HARDWARE.CudaHardwareProbeError
hardware_spec_to_dict = _HARDWARE.hardware_spec_to_dict
probe_cuda_hardware = _HARDWARE.probe_cuda_hardware


def _atomic_write_json(output_path: str | os.PathLike[str], spec: dict[str, Any]) -> None:
    """Publish strict JSON by replacing a same-directory temporary file."""
    serializable = hardware_spec_to_dict(spec)
    destination = Path(output_path).expanduser()
    if not destination.name:
        raise ValueError("--output must name a JSON file, not a directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as temporary_file:
            file_descriptor = -1  # The context manager now owns the descriptor.
            json.dump(serializable, temporary_file, ensure_ascii=False, allow_nan=False, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_name, destination)
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Query reported CUDA device attributes and write a JSON hardware spec. "
            "This does not run bandwidth/throughput microbenchmark calibration."
        )
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON path (written atomically)")
    parser.add_argument("--device", type=int, default=0, metavar="INDEX", help="CUDA device index (default: 0)")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Also print the collected spec as indented JSON to stdout after writing it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        spec = probe_cuda_hardware(device_index=args.device)
    except CudaHardwareProbeError as exc:
        print(f"CUDA hardware probe failed: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Invalid CUDA hardware probe argument: {exc}", file=sys.stderr)
        return 2

    try:
        _atomic_write_json(args.output, spec)
    except (OSError, TypeError, ValueError) as exc:
        print(f"Could not write CUDA hardware spec to {args.output}: {exc}", file=sys.stderr)
        return 1

    if args.pretty:
        serializable = hardware_spec_to_dict(spec)
        print(json.dumps(serializable, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
