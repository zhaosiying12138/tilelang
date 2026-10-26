"""TileLang/torch CUDA workers.  Imported only inside a timed subprocess.

No function in this module is called by ``--list`` or the CPU schema tests.
Every timed sample uses torch.cuda.Event and synchronizes its end event.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Callable, Mapping

try:
    from .environment import collect_versions, nvidia_snapshot, utc_now
    from .instruction_audit import ARTIFACT_KINDS, AUDITED_BENCHMARKS, require_instruction_audit
    from .specs import get_spec, resolve_config, units_for, validate_resolved_config, workload_for
    from .stats import latency_per_operation, summarize_samples, throughput, with_rate_basis
except ImportError:
    from environment import collect_versions, nvidia_snapshot, utc_now
    from instruction_audit import ARTIFACT_KINDS, AUDITED_BENCHMARKS, require_instruction_audit
    from specs import get_spec, resolve_config, units_for, validate_resolved_config, workload_for
    from stats import latency_per_operation, summarize_samples, throughput, with_rate_basis


def _base_result(name: str, config: Mapping[str, Any] | None, warmup: int, repeat: int, environment: Mapping[str, Any] | None = None) -> dict[str, Any]:
    spec = get_spec(name)
    return {
        "name": name,
        "status": "failed",
        "config": dict(config or {}),
        "shape": list((config or spec.to_dict())["shape"]),
        "dtype": spec.dtype,
        "accum_dtype": spec.accum_dtype,
        "work": workload_for(config) if config else None,
        "units": units_for(name),
        "warmup": warmup,
        "repeat": repeat,
        "sample_count": 0,
        "raw_samples_ms": [],
        "min_ms": None,
        "q1_ms": None,
        "median_ms": None,
        "q3_ms": None,
        "iqr_ms": None,
        "max_ms": None,
        "correctness": {"passed": False, "check": "not executed"},
        "compile": {"status": "not_reported"},
        "instruction_audit": None,
        "provenance": {
            "measurement": "dedicated_microbenchmark",
            "independent_measurement": True,
            "ratio_derived": False,
            "fit_from_formal_candidates": False,
            "timing": "torch.cuda.Event",
        },
        "environment": dict(environment or {}),
        "rates": {},
    }


def failed_result(name: str, error: BaseException, warmup: int, repeat: int, environment: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = _base_result(name, None, warmup, repeat, environment)
    result["error"] = {"type": type(error).__name__, "message": str(error)}
    result["rates"] = {key: None for key in _rate_keys(name)}
    return result


def _rate_keys(name: str) -> tuple[str, ...]:
    try:
        from .schema import BENCHMARK_RATE_KEYS
    except ImportError:
        from schema import BENCHMARK_RATE_KEYS
    return BENCHMARK_RATE_KEYS[name]


def _metric(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _close(actual: Any, expected: Any, *, check: str, rtol: float = 0.0, atol: float = 0.0) -> dict[str, Any]:
    import torch

    close = torch.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
    passed = bool(close.all().item())
    with torch.no_grad():
        max_error = (actual.float() - expected.float()).abs().max().item()
    return {
        "passed": passed,
        "check": check,
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": _metric(max_error),
        "elements": actual.numel(),
    }


def _exact(passed: bool, check: str, elements: int) -> dict[str, Any]:
    return {"passed": bool(passed), "check": check, "elements": elements, "max_abs_error": 0.0 if passed else None}


def _time_events(torch: Any, fn: Callable[[], Any] | None, warmup: int, repeat: int, flush: Any = None) -> dict[str, Any]:
    """One raw CUDA-event sample per timed launch, with no cache work in events."""

    if fn is not None:
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        if flush is not None:
            flush.zero_()  # Ordered before start_event on the same stream.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if fn is not None:
            fn()
        end.record()
        end.synchronize()
        elapsed = float(start.elapsed_time(end))
        if not math.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError(f"Invalid CUDA event sample: {elapsed}")
        samples.append(elapsed)
    return summarize_samples(samples)


def _compile(tilelang: Any, func: Any) -> Any:
    return tilelang.compile(
        func,
        target="cuda",
        execution_backend="tvm_ffi",
        pass_configs={"tl.disable_wgmma": True},
    )


def _audit_compiled(compiled: Any, name: str, artifact_dir: str, artifact_kind: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Export and audit real instructions inside the checked per-task cache."""

    if artifact_kind not in ARTIFACT_KINDS:
        raise ValueError(f"Unsupported instruction artifact kind {artifact_kind!r}")
    root = Path(artifact_dir).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    source_path = (root / f"{name}.cuda").resolve(strict=False)
    artifact_path = (root / f"{name}.{artifact_kind}").resolve(strict=False)
    if source_path.parent != root or artifact_path.parent != root:
        raise ValueError("Instruction artifact escaped its per-task directory")
    source = compiled.get_kernel_source()
    source_path.write_text(source, encoding="utf-8")
    export_method = f"export_{artifact_kind}"
    getattr(compiled, export_method)(str(artifact_path))  # Public TileLang API.
    if not artifact_path.is_file():
        raise RuntimeError(f"TileLang {export_method} did not create {artifact_path}")
    artifact_text = artifact_path.read_text(encoding="utf-8")
    report = require_instruction_audit(
        name,
        artifact_text,
        artifact_kind,
        artifact_path=str(artifact_path),
        export_method=export_method,
    )
    compile_info = {
        "status": "compiled",
        "target": str(compiled.target),
        "execution_backend": str(getattr(compiled, "execution_backend", "tvm_ffi")),
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source_bytes": len(source.encode("utf-8")),
    }
    return compile_info, report


def _instruction_details(compile_info: Mapping[str, Any], audit: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    return {"compile": dict(compile_info), "instruction_audit": dict(audit), **extra}


def _copy_runner(torch: Any, tilelang: Any, T: Any, config: Mapping[str, Any]) -> tuple[Callable[[], Any], Callable[[], dict[str, Any]], Any, dict[str, Any]]:
    name = config["name"]
    n = config["shape"][0]
    threads = config["threads"]
    blocks = config["blocks"]
    iterations = config["inner_iterations"]
    dtype = config["dtype"]
    elements_per_thread = config["elements_per_thread"]
    vector_width = config["vector_width"]
    chunks = elements_per_thread // vector_width

    @T.prim_func
    def kernel(A: T.Tensor((T.dynamic("n"),), "float32"), B: T.Tensor((T.dynamic("n"),), "float32")):
        with T.Kernel(blocks, threads=threads) as bx:
            tx = T.get_thread_binding()
            block_base = bx * threads * elements_per_thread
            for iteration in T.unroll(iterations):
                for chunk in T.unroll(chunks):
                    index = block_base + chunk * threads * vector_width + tx * vector_width
                    if iteration % 2 == 0:
                        for lane in T.vectorized(vector_width):
                            B[index + lane] = A[index + lane]
                    else:
                        for lane in T.vectorized(vector_width):
                            A[index + lane] = B[index + lane]

    compiled = _compile(tilelang, kernel)
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.empty_like(a)
    reference = a.clone() if iterations > 1 else a

    def launch() -> None:
        compiled(a, b)

    def correctness() -> dict[str, Any]:
        passed = torch.equal(a, reference) and torch.equal(b, reference)
        return _exact(passed, "A and B exactly equal the original input after alternating copies", 2 * n)

    flush = torch.empty(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.int32) if config["flush_l2"] else None
    return launch, correctness, flush, {"alternating_iterations": iterations, "elements_per_thread": elements_per_thread, "vector_width": vector_width, "l2_flush_bytes": 256 * 1024 * 1024 if flush is not None else 0}


def _shared_read_runner(torch: Any, tilelang: Any, T: Any, config: Mapping[str, Any]):
    size = config["shape"][0]
    threads = config["threads"]
    blocks = config["blocks"]
    iterations = config["inner_iterations"]
    unroll = config["unroll"]

    @T.prim_func
    def kernel(Source: T.Tensor((size,), "float32"), Out: T.Tensor((blocks, threads), "float32")):
        with T.Kernel(blocks, threads=threads) as bx:
            tx = T.get_thread_binding()
            shared = T.alloc_shared((size,), "float32")
            accumulator = T.alloc_local((1,), "float32")
            T.copy(Source, shared)
            T.sync_threads()
            accumulator[0] = T.float32(0)
            for iteration in T.serial(iterations):
                for u in T.unroll(unroll):
                    accumulator[0] += shared[(tx + u * threads + iteration * 64) & (size - 1)]
            Out[bx, tx] = accumulator[0]

    compiled = _compile(tilelang, kernel)
    source = torch.ones(size, device="cuda", dtype=torch.float32)
    out = torch.empty((blocks, threads), device="cuda", dtype=torch.float32)

    def launch() -> None:
        compiled(source, out)

    expected = float(iterations * unroll)
    return launch, lambda: _exact(bool((out == expected).all().item()), "Every shared-read reduction equals iterations*unroll", out.numel()), None, {"initialization": "shared copy and final global store are included in launch time but not shared-read bytes"}


def _shared_write_reference(torch: Any, config: Mapping[str, Any]) -> Any:
    size = config["shape"][0]
    threads = config["threads"]
    iterations = config["inner_iterations"]
    unroll = config["unroll"]
    slots = size // threads
    operations = iterations * unroll
    if size % threads or operations < slots or operations % slots:
        raise ValueError("Shared-write ring requires size=N*threads and a whole number of ring revolutions")
    # Address = (q % slots)*threads + tx.  Each thread owns one column, so
    # independent warps cannot race; consecutive lanes remain bank-conflict-free.
    q = torch.arange(operations - slots, operations, dtype=torch.int64)
    values = (q % 251 + 1).to(torch.float32)
    return values.repeat_interleave(threads).reshape(slots, threads).reshape(size)


def _shared_write_runner(torch: Any, tilelang: Any, T: Any, config: Mapping[str, Any]):
    size = config["shape"][0]
    threads = config["threads"]
    blocks = config["blocks"]
    iterations = config["inner_iterations"]
    unroll = config["unroll"]
    slots = size // threads

    @T.prim_func
    def kernel(Out: T.Tensor((blocks, size), "float32")):
        with T.Kernel(blocks, threads=threads) as bx:
            tx = T.get_thread_binding()
            shared = T.alloc_shared((size,), "float32")
            for iteration in T.serial(iterations):
                for u in T.unroll(unroll):
                    q = iteration * unroll + u
                    shared[(q % slots) * threads + tx] = T.float32(q % 251 + 1)
            T.sync_threads()
            T.copy(shared, Out[bx, :])

    compiled = _compile(tilelang, kernel)
    out = torch.empty((blocks, size), device="cuda", dtype=torch.float32)
    expected = _shared_write_reference(torch, config).cuda()

    def launch() -> None:
        compiled(out)

    return launch, lambda: _exact(bool((out == expected.unsqueeze(0)).all().item()), "Full shared image matches an independent thread-private ring reference", out.numel()), None, {"addressing": "(q % (size/threads))*threads+tx; thread-private columns avoid races and consecutive lanes avoid bank conflicts", "final_drain": "The final shared->global image is included in time but excluded from shared-write bytes"}


def _gemm_runner(torch: Any, tilelang: Any, T: Any, config: Mapping[str, Any]):
    name = config["name"]
    m, n, k = config["shape"]
    blocks = config["blocks"]
    iterations = config["inner_iterations"]
    tilelang_dtypes = {
        "bf16_mma_gemm": T.bfloat16,
        "f16_mma_gemm": T.float16,
        "tf32_mma_gemm": T.tfloat32,
    }
    torch_dtypes = {
        "bf16_mma_gemm": torch.bfloat16,
        "f16_mma_gemm": torch.float16,
        "tf32_mma_gemm": torch.float32,
    }
    tilelang_dtype = tilelang_dtypes[name]
    torch_dtype = torch_dtypes[name]

    @T.prim_func
    def kernel(A: T.Tensor((m, k), tilelang_dtype), B: T.Tensor((k, n), tilelang_dtype), Out: T.Tensor((blocks, m, n), "float32")):
        with T.Kernel(blocks, threads=config["threads"]) as bx:
            a_shared = T.alloc_shared((m, k), tilelang_dtype)
            b_shared = T.alloc_shared((k, n), tilelang_dtype)
            accumulator = T.alloc_fragment((m, n), "float32")
            T.copy(A, a_shared)
            T.copy(B, b_shared)
            T.clear(accumulator)
            for _ in T.serial(iterations):
                T.gemm(a_shared, b_shared, accumulator)
            T.copy(accumulator, Out[bx, :, :])

    compiled = _compile(tilelang, kernel)
    compile_info, instruction_audit = _audit_compiled(compiled, name, config["artifact_dir"], config["instruction_artifact_kind"])
    a_base = ((torch.arange(m * k, device="cuda").reshape(m, k) % 7) - 3) / 16
    b_base = ((torch.arange(k * n, device="cuda").reshape(k, n) % 5) - 2) / 16
    a = a_base.to(torch_dtype)
    b = b_base.to(torch_dtype)
    out = torch.empty((blocks, m, n), device="cuda", dtype=torch.float32)
    reference_allow_tf32 = name == "tf32_mma_gemm"
    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = reference_allow_tf32
    try:
        reference = (a.float() @ b.float()) * iterations
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32

    def launch() -> None:
        compiled(a, b, out)

    details = _instruction_details(
        compile_info,
        instruction_audit,
        lowering="T.gemm with fragment accumulator and tl.disable_wgmma=true; compiled-artifact audit is authoritative",
        source_tl_mma_sync_marker_present="tl::mma_sync" in compiled.get_kernel_source(),
        counting="logical 2*M*N*K FLOPs; no per-thread warp-instruction multiplier and no cross-dtype rate ratio",
        input_storage_dtype=config["storage_dtype"] or config["dtype"],
        tilelang_dtype=str(tilelang_dtype),
        reference_allow_tf32=reference_allow_tf32,
        reference="independent torch matmul with explicit allow_tf32 setting, multiplied by inner_iterations",
    )
    return launch, lambda: _close(out, reference.unsqueeze(0), check=f"{name}: every CTA equals iterations*(A@B) in FP32", rtol=2e-2, atol=2e-2), None, details


def _fp32_fma_runner(torch: Any, tilelang: Any, T: Any, config: Mapping[str, Any]):
    blocks = config["blocks"]
    threads = config["threads"]
    iterations = config["inner_iterations"]
    chains = config["unroll"]

    @T.prim_func
    def kernel(A: T.Tensor((blocks, threads, chains), "float32"), B: T.Tensor((blocks, threads, chains), "float32"), Out: T.Tensor((blocks, threads, chains), "float32")):
        with T.Kernel(blocks, threads=threads) as bx:
            tx = T.get_thread_binding()
            a_local = T.alloc_local((chains,), "float32")
            b_local = T.alloc_local((chains,), "float32")
            accumulator = T.alloc_local((chains,), "float32")
            for u in T.unroll(chains):
                a_local[u] = A[bx, tx, u]
                b_local[u] = B[bx, tx, u]
                accumulator[u] = T.float32(0)
            for iteration in T.serial(iterations):
                for u in T.unroll(chains):
                    factor = a_local[u] + (iteration % 8) * 0.03125
                    accumulator[u] = T.fma(factor, b_local[u], accumulator[u])
            for u in T.unroll(chains):
                Out[bx, tx, u] = accumulator[u]

    compiled = _compile(tilelang, kernel)
    compile_info, instruction_audit = _audit_compiled(compiled, config["name"], config["artifact_dir"], config["instruction_artifact_kind"])
    total = blocks * threads * chains
    a = ((torch.arange(total, device="cuda").reshape(blocks, threads, chains) % 5) - 2) / 16
    b = ((torch.arange(total, device="cuda").reshape(blocks, threads, chains) % 7) - 3) / 32
    out = torch.empty_like(a)
    # Dyadic inputs and bounded numerators make every repeated FMA exact.
    # The independent closed form covers the period-8 input perturbation and
    # prevents the compiler from replacing the loop with one repeated product.
    if iterations % 8:
        raise ValueError("FP32 FMA iterations must be divisible by the input period 8")
    reference = (a * b) * iterations + b * ((iterations // 8) * 0.875)

    def launch() -> None:
        compiled(a, b, out)

    details = _instruction_details(
        compile_info,
        instruction_audit,
        lowering="Explicit scalar T.fma chains on CUDA cores; compiled audit forbids mma.sync/WGMMA/TCGEN05/HMMA as applicable",
        counting="2 FLOPs per scalar FMA; four independent chains per thread; no TensorCore work; periodic input adds and loop overhead are timed but not counted as FMA FLOPs",
        reference="exact independent closed-form elementwise sum for (A+(iteration%8)/32)*B; no torch.matmul, loop reuse, or TensorCore",
    )
    return launch, lambda: _exact(bool((out == reference).all().item()), "Every FP32 FMA chain exactly equals the independent dyadic elementwise reference", total), None, details


def _sfu_runner(torch: Any, tilelang: Any, T: Any, config: Mapping[str, Any]):
    blocks = config["blocks"]
    threads = config["threads"]
    iterations = config["inner_iterations"]
    chains = config["unroll"]

    @T.prim_func
    def kernel(Input: T.Tensor((blocks, threads, chains), "float32"), Out: T.Tensor((blocks, threads, chains), "float32")):
        with T.Kernel(blocks, threads=threads) as bx:
            tx = T.get_thread_binding()
            base = T.alloc_local((chains,), "float32")
            accumulator = T.alloc_local((chains,), "float32")
            for u in T.unroll(chains):
                base[u] = Input[bx, tx, u]
                accumulator[u] = T.float32(0)
            for iteration in T.serial(iterations):
                for u in T.unroll(chains):
                    accumulator[u] += T.exp2(base[u] + (iteration % 8) * 0.125 + u * 0.03125)
            for u in T.unroll(chains):
                Out[bx, tx, u] = accumulator[u]

    compiled = _compile(tilelang, kernel)
    if "exp2" not in compiled.get_kernel_source().lower():
        raise RuntimeError("SFU kernel source has no exp2 marker")
    total = blocks * threads * chains
    values = ((torch.arange(total, device="cuda").reshape(blocks, threads, chains) % 257) / 257.0 - 0.5) * 1.5
    out = torch.empty_like(values)
    reference = torch.zeros_like(values)
    for iteration in range(iterations):
        for u in range(chains):
            reference[:, :, u] += torch.exp2(values[:, :, u] + (iteration % 8) * 0.125 + u * 0.03125)

    def launch() -> None:
        compiled(values, out)

    return launch, lambda: _close(out, reference, check="Each exp2 chain matches torch.exp2 with identical iteration order", rtol=2e-2, atol=2e-2), None, {"operation_count": "Only exp2 calls count as SFU operations; input arithmetic and loop overhead remain in measured time"}


def _barrier_runner(torch: Any, tilelang: Any, T: Any, config: Mapping[str, Any]):
    blocks = config["blocks"]
    iterations = config["inner_iterations"]

    @T.prim_func
    def kernel(Out: T.Tensor((blocks,), "int32")):
        with T.Kernel(blocks, threads=config["threads"]) as bx:
            tx = T.get_thread_binding()
            if tx == 0:
                Out[bx] = 0
            for _ in T.serial(iterations):
                T.sync_threads()
            if tx == 0:
                Out[bx] = 1

    compiled = _compile(tilelang, kernel)
    out = torch.empty(blocks, device="cuda", dtype=torch.int32)
    return lambda: compiled(out), lambda: _exact(bool((out == 1).all().item()), "Every CTA reaches the store after all barriers", blocks), None, {"normalization": "per-CTA barrier latency; blocks are not multiplied into the rate"}


def _empty_runner(torch: Any, tilelang: Any, T: Any, config: Mapping[str, Any]):
    @T.prim_func
    def kernel(Out: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=config["threads"]) as _:
            tx = T.get_thread_binding()
            if tx == 1024:  # Never taken: preserve a launch with no executed store.
                Out[0] = 0

    compiled = _compile(tilelang, kernel)
    out = torch.full((1,), 17, device="cuda", dtype=torch.int32)
    return lambda: compiled(out), lambda: _exact(int(out.item()) == 17, "No-store empty kernel preserves its output sentinel", 1), None, {"semantics": "empty/no-store launch, not a host-wall-clock launch measurement"}


def _rates_for(name: str, work: Mapping[str, Any], median_ms: float, source: str, basis: str) -> dict[str, Any]:
    if name in {"dram_copy", "l2_resident_copy"}:
        key = "dram_copy_bytes_per_second" if name == "dram_copy" else "l2_copy_bytes_per_second"
        rate = throughput(work["bytes"], median_ms, "byte/second")
    elif name == "shared_read":
        key = "shared_read_bytes_per_second"
        rate = throughput(work["bytes"], median_ms, "byte/second")
    elif name == "shared_write":
        key = "shared_write_bytes_per_second"
        rate = throughput(work["bytes"], median_ms, "byte/second")
    elif name in {"bf16_mma_gemm", "f16_mma_gemm", "tf32_mma_gemm"}:
        key = {
            "bf16_mma_gemm": "bf16_gemm_flops_per_second",
            "f16_mma_gemm": "f16_gemm_flops_per_second",
            "tf32_mma_gemm": "tf32_gemm_flops_per_second",
        }[name]
        rate = throughput(work["flops"], median_ms, "FLOP/second")
    elif name == "cuda_fp32_fma":
        key = "cuda_fp32_fma_flops_per_second"
        rate = throughput(work["flops"], median_ms, "FLOP/second")
    elif name == "sfu_exp2":
        key = "sfu_exp2_operations_per_second"
        rate = throughput(work["operations"], median_ms, "operation/second")
    elif name == "barrier":
        key = "barrier_latency_ns"
        rate = latency_per_operation(median_ms, work["barriers_per_cta"])
    else:
        raise ValueError(f"No throughput formula for {name}")
    return {key: with_rate_basis(rate, source, basis)}


def _overhead_rate(value_ms: float, unit: str, source: str) -> dict[str, Any]:
    if not math.isfinite(value_ms) or value_ms < 0:
        raise RuntimeError(f"Invalid overhead median {value_ms}")
    return {"value": value_ms * 1_000_000.0, "unit": unit, "work": 1, "time_ms": value_ms, "source": source, "basis": "raw CUDA-event median"}


def run_benchmark(
    name: str,
    *,
    warmup: int,
    repeat: int,
    physical_device: int = 0,
    event_overhead_ms: float | None = None,
    empty_launch_overhead_ms: float | None = None,
    artifact_dir: str | None = None,
    instruction_artifact_kind: str = "ptx",
) -> dict[str, Any]:
    """Execute exactly one benchmark in the current, isolated worker process."""

    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat <= 0:
        raise ValueError("repeat must be a positive integer")

    # Heavy imports deliberately live below the CLI/list/schema boundary.
    import torch
    import tilelang
    import tilelang.language as T

    before = nvidia_snapshot(physical_device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the calibration worker")
    torch.cuda.set_device(0)  # Parent maps physical_device through CUDA_VISIBLE_DEVICES.
    torch.manual_seed(20261027)
    properties = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    l2_bytes = getattr(properties, "L2_cache_size", getattr(properties, "l2_cache_size", 0))
    device = {
        "name": properties.name,
        "arch": f"sm_{major}{minor}",
        "compute_capability": [major, minor],
        "sm_count": int(properties.multi_processor_count),
        "l2_bytes": int(l2_bytes),
        "total_memory_bytes": int(properties.total_memory),
        "visible_device": 0,
        "physical_device": physical_device,
    }
    config = resolve_config(name, device["sm_count"], device["l2_bytes"])
    if name in AUDITED_BENCHMARKS:
        if artifact_dir is None:
            raise ValueError(f"{name} requires a checked per-task artifact directory")
        if instruction_artifact_kind not in ARTIFACT_KINDS:
            raise ValueError(f"Unsupported instruction artifact kind {instruction_artifact_kind!r}")
        config["artifact_dir"] = artifact_dir
        config["instruction_artifact_kind"] = instruction_artifact_kind
    validate_resolved_config(config)
    environment = {
        "started_at_utc": utc_now(),
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "device": device,
        "versions": {
            **collect_versions((torch, tilelang)),
            "cuda_runtime": torch.version.cuda,
            "nvidia_driver": before.get("raw", {}).get("driver_version"),
        },
        "nvidia_smi_before": before,
        "cache": {key: os.environ.get(key) for key in ("TILELANG_CACHE_DIR", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR", "TMPDIR")},
    }
    result = _base_result(name, config, warmup, repeat, environment)

    runners = {
        "dram_copy": _copy_runner,
        "l2_resident_copy": _copy_runner,
        "shared_read": _shared_read_runner,
        "shared_write": _shared_write_runner,
        "bf16_mma_gemm": _gemm_runner,
        "f16_mma_gemm": _gemm_runner,
        "tf32_mma_gemm": _gemm_runner,
        "cuda_fp32_fma": _fp32_fma_runner,
        "sfu_exp2": _sfu_runner,
        "barrier": _barrier_runner,
        "empty_launch": _empty_runner,
    }
    launch, correctness, flush, details = runners[name](torch, tilelang, T, config)
    details = dict(details)
    result["compile"] = details.pop("compile", {"status": "compiled", "target": "cuda", "execution_backend": "tvm_ffi"})
    if "instruction_audit" in details:
        result["instruction_audit"] = details.pop("instruction_audit")
    launch()
    torch.cuda.synchronize()
    check = correctness()
    result["correctness"] = check
    result["implementation"] = details
    if not check["passed"]:
        result["error"] = {"type": "CorrectnessError", "message": "Independent correctness check failed; no timing/rate accepted"}
        result["rates"] = {key: None for key in _rate_keys(name)}
        result["environment"]["finished_at_utc"] = utc_now()
        result["environment"]["nvidia_smi_after"] = nvidia_snapshot(physical_device)
        return result

    stats = _time_events(torch, launch, warmup, repeat, flush)
    result.update(stats)
    recheck = correctness()
    result["correctness"] = {
        **recheck,
        "checked_before_timing": check["passed"],
        "rechecked_after_timing": recheck["passed"],
    }
    if not recheck["passed"]:
        result["error"] = {"type": "CorrectnessError", "message": "Correctness changed after timed repetition; rates rejected"}
        result["rates"] = {key: None for key in _rate_keys(name)}
        result["environment"]["finished_at_utc"] = utc_now()
        result["environment"]["nvidia_smi_after"] = nvidia_snapshot(physical_device)
        return result
    result["status"] = "complete"
    result["overhead"] = {
        "event_median_ms": event_overhead_ms,
        "empty_launch_median_ms": empty_launch_overhead_ms,
        "correction_applied": False,
        "net_median_ms": None,
    }
    if name == "empty_launch":
        event_stats = _time_events(torch, None, warmup, repeat)
        result["event_pair"] = event_stats
        result["rates"] = {
            "event_overhead_ns": _overhead_rate(event_stats["median_ms"], "nanosecond/event-pair", name),
            "empty_launch_overhead_ns": _overhead_rate(stats["median_ms"], "nanosecond/launch-including-events", name),
            "launch_overhead_ns": _overhead_rate(max(0.0, stats["median_ms"] - event_stats["median_ms"]), "nanosecond/launch-marginal-to-event-pair", name),
        }
    else:
        raw_rates = _rates_for(name, result["work"], stats["median_ms"], name, "raw event median including launch")
        result["uncorrected_rates"] = raw_rates
        rate_ms = stats["median_ms"]
        basis = "raw event median; empty-launch correction unavailable"
        if empty_launch_overhead_ms is not None:
            if not math.isfinite(empty_launch_overhead_ms) or empty_launch_overhead_ms < 0:
                raise ValueError("empty_launch_overhead_ms must be finite and non-negative")
            net_ms = rate_ms - empty_launch_overhead_ms
            # Do not manufacture huge rates from a noise-sized denominator.
            if net_ms <= 0 or net_ms < 0.5 * rate_ms:
                raise RuntimeError(f"{name} is dominated by launch overhead: raw={rate_ms} ms, net={net_ms} ms; increase inner_iterations")
            result["overhead"].update({"correction_applied": True, "net_median_ms": net_ms})
            rate_ms = net_ms
            basis = "raw median minus independently measured empty-launch median"
        result["overhead"]["rate_time_ms"] = rate_ms
        result["rates"] = _rates_for(name, result["work"], rate_ms, name, basis)
    result["environment"]["finished_at_utc"] = utc_now()
    result["environment"]["nvidia_smi_after"] = nvidia_snapshot(physical_device)
    return result
