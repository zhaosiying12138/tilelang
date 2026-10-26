# TileLang CUDA calibration microbenchmarks

This directory contains an **entry point, not a measured calibration result**.
No `calibration.json` is checked in. A file is marked `status="complete"` only
when every required worker really ran, produced finite event samples, and passed
its independent correctness checks before and after timing. GEMM and FP32 FMA
rates additionally require a passing audit of exported PTX or SASS; a
`T.gemm`/`T.fma` name or CUDA-source marker alone is not instruction evidence.

The design follows the useful G3 discipline (dedicated microbenchmarks, CUDA
events, process isolation, robust statistics, and environment capture), but does
**not** import G3 numeric constants or fit rates from GEMM4096, Conv, FA, GDN, or
any other formal candidate result.

## Safety discipline

- The parent runs `free -g` before creating any worker and requires host Mem
  total >= 16 GiB.
- `nvidia-smi --query-compute-apps=...` must be empty before the run and before
  every task. Do not circumvent this on a shared GPU. A worker can appear in its
  own later environment snapshot; that is expected and is not another co-load.
- Tasks are sequential, one fresh Python process/process group per benchmark.
  Each has a wall timeout; timeout kills that process group, not other sessions.
- Every task gets a unique run/task cache. `TILELANG_CACHE_DIR`,
  `CUDA_CACHE_PATH`, `TORCH_EXTENSIONS_DIR`, and `TMPDIR` cannot escape that
  directory. The last also confines temporary CUBIN files used during SASS
  disassembly. `--cache-root` must be absolute. Worker result and
  PTX/SASS/CUDA-source artifact paths are checked as well.
- Only `torch.cuda.Event` timing is used. There is no nsys, CUPTI, CUDA
  injection, CUPTI backend, or profiler-based fitting.
- The largest default live-tensor allocation is DRAM copy: 512 MiB for A+B and
  256 MiB for the untimed L2 flush buffer. All default workers are designed for
  a <2 GiB total tensor-memory budget; isolated processes release allocations
  before the next task.
- `--list` imports neither torch nor TileLang and does not call `free`,
  `nvidia-smi`, or CUDA.

## Core benchmarks

All formulas below are for one timed kernel launch. Warmup launches and timed
raw samples are independent CLI parameters (defaults: 10 and 50). Both copy
kernels assign 16 elements per thread and use 4-element inner vector loops;
resolved tile/grid geometry is recorded in the JSON.

| Name | Default shape / dtype | Work formula and correctness |
|---|---|---|
| `dram_copy` | `[67108864]`, FP32 | `bytes=2*N*4*iterations`; A+B working set is 512 MiB. Exact A/B comparison; an L2 flush is queued before, not inside, each event interval. |
| `l2_resident_copy` | up to `[1048576]`, FP32 | Same copy bytes formula with 16 alternating A->B/B->A iterations. Source+destination are capped at 1/4 of reported L2 and at 8 MiB. Exact comparison to the original input. |
| `shared_read` | 8192 FP32/CTA, 256 threads | `bytes=blocks*threads*1024*16*4`. Every reduction must equal `1024*16`. Initialization and final output are in time but not shared-read bytes. |
| `shared_write` | 8192 FP32/CTA, 256 threads | Same access-count formula for stores. Address `(q%(size/threads))*threads+tx` gives each thread a private ring column, avoids cross-warp races, and keeps consecutive lanes bank-conflict-free. The full final image is checked. Final drain is in time but not shared-write bytes. |
| `bf16_mma_gemm` | `(M,N,K)=(96,80,48)`, BF16 -> FP32 | `FLOPs=blocks*256*2*M*N*K`. Fragment-output `T.gemm`, WGMMA disabled, exported BF16 `mma.sync` instruction required. Every CTA is checked against `256*(A@B)`. Logical GEMM work has no per-thread `*32`. |
| `f16_mma_gemm` | same structure, FP16 -> FP32 | Independently compiled, referenced, timed, and PTX/SASS-audited FP16 `T.gemm`; no BF16 ratio or shared timing samples. |
| `tf32_mma_gemm` | same structure, TF32 -> FP32 | FP32 torch storage with explicit `T.tfloat32` operands. The independent reference explicitly sets/restores `torch.backends.cuda.matmul.allow_tf32=True`; exported TF32 `mma.sync` k4/k8 evidence is required. |
| `cuda_fp32_fma` | four FP32 chains/thread | `FLOPs=blocks*threads*8192*4*2`. Four independent `T.fma` chains with period-8 dyadic inputs, exact closed-form correctness reference, and four scalar-equivalent audited PTX FMAs (four `f32` or two `f32x2`). TensorCore instructions are forbidden in its audit. |
| `sfu_exp2` | four FP32 chains/thread | `operations=blocks*threads*1024*4`. Each chain is checked against `torch.exp2` in identical iteration order. Only exp2 calls count as SFU work; loop/input arithmetic remains in measured time. |
| `barrier` | 128 threads, 16384 barriers/CTA | `total=blocks*16384`, but latency is `median/16384` **per CTA**, without multiplying by concurrent CTAs. Output proves every CTA reached the final store. |
| `empty_launch` | one no-store CTA | Sentinel must remain unchanged. Measures an empty launch and a separate back-to-back event pair. |

The GEMM triple is intentionally different from the formal GEMM4096/Conv/FA/GDN
shapes. BF16, FP16, and TF32 are independently measured and never derived from
one another. FP32 CUDA-core throughput is independently measured with `T.fma`,
not inferred from TensorCore or SFU results. All 11 benchmarks are required for
a complete v2 artifact and have separate task caches even when GEMM shapes match.

## Timing, overhead, and statistics

For each item the JSON stores every `raw_samples_ms`, min/Q1/median/Q3/max,
`iqr_ms`, units, resolved shape/config, work and bytes/FLOP formulas, warmup and
repeat counts, correctness details, compile status/source hash, independent
measurement provenance, and worker environment. Instruction-audited items also
store the artifact kind/path/hash, exporter, matched instructions, and forbidden
matches. Quartiles use the linear interpolation convention at positions
`(n-1)*p`; `IQR=Q3-Q1`.

A full run executes `empty_launch` first. Other rates retain both:

1. `uncorrected_rates`: work / raw event median;
2. `rates`: work / (raw median - empty-launch median), when empty-launch data is
   available.

The correction is rejected if the net time is non-positive or less than half the
raw median, rather than manufacturing a huge rate from noise. A subset run that
does not include `empty_launch` uses raw-median rates, labels that basis, and is
necessarily partial. Event-pair, empty-launch, and marginal launch overhead are
separate global fields. These are CUDA-event estimates, not host wall-clock
launch measurements.

The rates describe only the measured access patterns. Copy bandwidth counts
read+write payload and is an effective streaming proxy, not a pure-load or
`cp.async` descriptor measurement. Shared initialization/drain and loop/address
arithmetic are not magically removed. Single-stream runs do not calibrate
cross-stream overlap: resource times must not be assumed either fully additive
or fully overlapped. Barrier residency, L2 residency, bank conflicts, clock
throttling, and architecture-specific lowering remain explicit validation
concerns.

## Running on a clean machine (not run during implementation)

```bash
cd /home/zhaosiying/codebase/tilelang-cuda-z3

# CPU-only listing; safe on a machine with no GPU.
PYTHONDONTWRITEBYTECODE=1 python3 -B \
  benchmark/z3_cost_model/calibration/calibrate.py --list

# Full run. The destination parent must already exist.
PYTHONDONTWRITEBYTECODE=1 python3 -B \
  benchmark/z3_cost_model/calibration/calibrate.py \
  --output /home/zhaosiying/codebase/tilelang-cuda-z3/benchmark/z3_cost_model/calibration/calibration.json \
  --warmup 10 --repeat 50 --instruction-kind ptx --require-complete

# Diagnostic subset: produces status=partial and null for missing global rates.
PYTHONDONTWRITEBYTECODE=1 python3 -B \
  benchmark/z3_cost_model/calibration/calibrate.py \
  --only dram_copy,l2_resident_copy,bf16_mma_gemm \
  --output /tmp/tilelang-calibration-subset.json \
  --warmup 5 --repeat 20 --timeout 240

# Optional explicit cache parent (must be absolute).
# Add: --cache-root /tmp/tilelang-calibration-cache
# For SASS instead of the default PTX audit, use: --instruction-kind sass

# CPU-only re-audit of an artifact exported by an earlier worker.
PYTHONDONTWRITEBYTECODE=1 python3 -B \
  benchmark/z3_cost_model/calibration/instruction_audit.py \
  /tmp/tilelang-z3-cost-model-calibration/RUN_ID/tasks/f16_mma_gemm/artifacts/f16_mma_gemm.ptx \
  --benchmark f16_mma_gemm --kind ptx
```

The default cache is
`/tmp/tilelang-z3-cost-model-calibration/<run-id>/tasks/<benchmark>/`.
Each task contains its compiler caches, `result.json`, `stdout.log`, and
`stderr.log`; audited tasks additionally contain `artifacts/*.cuda` and
`artifacts/*.ptx` (or `*.sass`). Parent and worker snapshots record Python/TileLang/torch/CUDA and
driver versions, GPU identity, SM count, current/max clocks, temperature,
power, memory/use, process ID, and cache paths. `--require-complete` controls
the process exit status after writing diagnostics; it does not convert a
partial result into a complete one.

## Versioned schema and CudaCostModel boundary

The outer schema is `tilelang.cuda.calibration.v2` (11 required benchmarks and
mandatory compiled-instruction evidence for GEMM/FP32 FMA).

- Missing benchmarks are JSON `null`; their global rates are also `null`.
- Partial artifacts set `status="partial"`, `allow_partial=true`, and
  `model_ready=false`.
- Complete requires all 11 core items, passing correctness, and all required
  instruction audits. Device/clock metadata and every native rate must also be
  usable before `model_ready=true`.
- `validate_for_cost_model(document)` is the outer measured-core gate. It
  rejects partial, missing, non-finite, internally inconsistent, or unusable
  core data and returns the native `cost_model` view only after validation.
- `cost_model` and the duplicate `calibration` manifest use the existing
  `tilelang.cuda.kernel-cost-calibration` schema, so the outer artifact can also
  be passed directly to `CudaCostModel`. The separate `latency_model` member is
  a `CudaLatencyParams` compatibility view.
- Native shared bandwidth is derived as `2*read*write/(read+write)` for an
  equal-byte read/write mix. Native LSU issue throughput is `shared/128` for
  an equal-instruction mix of these 32-lane FP32 accesses. Both formulas and
  their overlap/vector-width limitations are recorded in the artifact.
- FP16, TF32, and FP32 CUDA-core FLOP/s come from their own real microbenchmarks;
  no ratio or placeholder is used. A complete 11-item artifact fills every
  native rate and is accepted by the current `CudaCostModel` with its default
  `require_measured=True`. Deleting or invalidating any one of those three (or
  any other required item) leaves its native rate `null`, forces
  `status=partial`/`model_ready=false`, and is rejected by measured-only use.
  `require_measured=False` exposes explicit non-free fallbacks instead. The
  existing latency model can still reject unmeasured bases or unrelated
  operation types requested by a formal task.

```python
import json
from benchmark.z3_cost_model.calibration.schema import validate_for_cost_model
from tilelang.cuda.cost_model import CudaCostModel

with open("/path/to/calibration.json", encoding="utf-8") as stream:
    document = json.load(stream)
view = validate_for_cost_model(document)  # Partial core input never crosses this gate.
# Formal measured-only use; CalibrationError is raised if any required rate is
# missing. A complete 11-item artifact reaches this constructor with no fallback.
model = CudaCostModel(hardware, view, "gemm", shape)
```

## CPU tests and remaining hardware validation

```bash
cd /home/zhaosiying/codebase/tilelang-cuda-z3
PYTHONDONTWRITEBYTECODE=1 python3 -B \
  testing/python/cuda/test_calibration_schema.py -v

# Equivalent pytest invocation, without .pytest_cache or bytecode:
PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q -p no:cacheprovider \
  testing/python/cuda/test_calibration_schema.py
```

The CPU suite checks schema/version/state transitions, copy/shared/SFU/FLOP
formulas (including two FLOPs per FMA), the warp-count trap, median/IQR, strict
finite JSON, missing and failed items, independent PTX/SASS audit rules,
complete `CudaCostModel` acceptance plus per-new-item rejection, the
`CudaLatencyParams` view, preflight parsers, `--list` import isolation, and cache
traversal defenses.

**Every GPU microbenchmark still requires real-machine validation.** In
particular, validate compilation and correctness on the intended architecture;
actual BF16/FP16/TF32 `mma.sync` and FP32 FMA PTX/SASS evidence; independent
FP16/TF32 timings; TF32 FP32-storage and allow_tf32 behavior; absence of
TensorCore instructions in the FP32 FMA kernel; L2 residency; shared access
patterns; event overhead correction; clocks/temperature/power stability; and
the absence of co-load. CPU tests do not assert hardware rates or claim that a
calibration run occurred.
