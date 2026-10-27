"""Export the confirmed GEMM winner's source, cubin, SASS, and resource usage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from benchmark.z3_cost_model.gdn.autotune import _tool_cache_environment
from benchmark.z3_cost_model.operators import gemm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GPU_QUERY = "timestamp,name,driver_version,power.draw,temperature.gpu,clocks.sm,clocks.mem,memory.used"


def _write_json(path, value):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True)
        stream.write("\n")


def _run(command, log_path, **kwargs):
    result = subprocess.run(command, check=True, capture_output=True, text=True, **kwargs)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    return result.stdout


def _inventory(root):
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in ("before.json", "after.json"):
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, required=True, help="gemm_square confirmation summary.json")
    parser.add_argument("--arm", default="z3-cm-topk")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve(strict=False)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Refusing a non-empty or non-directory export path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "before.json", {"empty": True, "entries": []})
    summary = json.loads(args.confirmation.read_text(encoding="utf-8"))
    if summary.get("suite") != "gemm_square" or summary.get("status") != "ok":
        raise ValueError("A complete gemm_square confirmation is required")
    config = summary["plan"]["winners"][args.arm]["config"]
    cache_dir = output / "cache"
    cache_dir.mkdir()
    previous = {
        name: os.environ.get(name) for name in ("TILELANG_CACHE_DIR", "TILELANG_DISABLE_CACHE", "TILELANG_AUTO_TUNING_DISABLE_CACHE")
    }
    os.environ["TILELANG_CACHE_DIR"] = str(cache_dir / "tilelang")
    os.environ["TILELANG_DISABLE_CACHE"] = "0"
    os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] = "0"
    try:
        with _tool_cache_environment(output, "cold"):
            kernel = gemm.kernel_factory(config, shape=gemm.PRIMARY_SHAPE, target="auto")
            source = kernel.get_kernel_source()
            (output / "source.cu").write_text(source, encoding="utf-8")
            compile_flags = list(kernel.compile_flags or [])
            nvcc = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.3")) / "bin/nvcc"
            cubin = output / "kernel.cubin"
            nvcc_command = [
                str(nvcc),
                "-ccbin",
                "/usr/bin/g++",
                "--cubin",
                "-O3",
                "-lineinfo",
                "-arch",
                "sm_120a",
                "-std",
                "c++20",
                f"-I{REPOSITORY_ROOT / 'src'}",
                f"-I{REPOSITORY_ROOT / '3rdparty/cutlass/include'}",
                *compile_flags,
                "-o",
                str(cubin),
                str(output / "source.cu"),
            ]
            _run(nvcc_command, output / "nvcc.log", cwd=REPOSITORY_ROOT)
            _run(["cuobjdump", "--dump-resource-usage", str(cubin)], output / "resource-usage.txt", cwd=REPOSITORY_ROOT)
            _run(["cuobjdump", "--dump-sass", str(cubin)], output / "kernel.sass", cwd=REPOSITORY_ROOT)
            telemetry = subprocess.run(
                ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout.strip()
            metadata = {
                "schema": "tilelang.gemm.winner-artifact/v1",
                "suite": "gemm_square",
                "arm": args.arm,
                "config": config,
                "confirmation": str(args.confirmation.resolve()),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip(),
                "kernel_type": f"{type(kernel).__module__}.{type(kernel).__qualname__}",
                "target": str(kernel.target),
                "compile_flags": compile_flags,
                "nvcc_command": nvcc_command,
                "nvcc_log": "nvcc.log",
                "resource_usage": "resource-usage.txt",
                "sass_log": "kernel.sass",
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "cubin_sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(),
                "cache_root": str(cache_dir),
                "gpu_telemetry": telemetry,
            }
            _write_json(output / "metadata.json", metadata)
            _write_json(output / "after.json", {"entries": _inventory(output), "metadata": metadata})
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    print(json.dumps({"output": str(output), "config": config, "cubin": str(output / "kernel.cubin")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
