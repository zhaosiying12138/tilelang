# GDN Guided AutoTuner adapter and search driver

`autotune.py` connects the direction-specific GDN spaces to the existing
`GuidedAutoTuner` without replacing the stock compilation, validation, profiling,
or event-recording paths with a dummy kernel. It requires the SM120 FlashQLA
contract in `PORTING.md`; importing the module and running its mock tests do not
require a GPU.

## Candidate and profiler contract

`make_candidate_factory(direction, ...)` has the four explicit configuration
arguments produced by `make_config_space`. Its implementation dependencies are
function defaults rather than non-scalar closure cells, so the stock AutoTuner
signature inspection accepts it. The custom `jit_compile` entry constructs a
`GDNCandidateKernel`; it does not call `tilelang.compile` on a substitute
PrimFunc.

Each candidate construction does the following real work:

- **fwd**: construct `GDNRunner` and execute one complete public
  `chunk_gated_delta_rule_fwd` through `GDNRunner.forward`. This triggers the
  public cumsum/KKT/fused-forward JIT path.
- **bwd**: construct `GDNRunner`, execute public forward once, retain its
  artifacts, and execute one complete public `chunk_gated_delta_rule_bwd`.
  This triggers the prepare_h/fused-backward/GQA-reduction/reverse-cumsum path.

`get_profiler()` constructs a real `tilelang.profiler.Profiler` with `params=[]`,
`result_idx=[]`, and the actual bound public-pipeline callable as `adapter`.
`_empty_input_supply` returns `[]` and rejects any non-empty parameter list.
The adapter and reference closures own inputs made by `make_fixed_inputs` with
the run seed, so validation, `do_bench`, and `ref_latency` do not depend on
Profiler's automatic tensor generation. The default fixture is the planned
performance shape `B=1, T=4096, Hg=4, H=8`, bf16, K=DV=128, chunk=32, with h0 and
dht. `config.json` records the exact shape and input options.

### Timing scope and backward forward artifacts

The profiled call is the complete public Python pipeline, including its tensor
allocation and internal launches; Profiler's normal event backend performs the
timing. There is no extra adapter-level synchronize or wall-clock estimator.

- fwd times `runner.forward(batch, output_h=False)` and returns
  `(o, final_state)`.
- bwd times only `runner.backward(batch, saved_forward_artifacts)` and returns
  `(dq, dk, dv, db, dg, dh0)`.

For bwd, the required public forward artifacts are produced once during candidate
initialization with fixed **`forward_block_DV=64`**. That value is a launchability
adaptation, not a backward search dimension. Forward artifact production is not
part of the backward candidate set, backward timing, or backward cost-model
charge. Reference backward follows the same scope: the float64 reference forward
is built once for validation artifacts, and subsequent reference calls execute
only reference backward. Consequently the stock `ref_latency` path times a real
reference implementation rather than a cached no-op.

The default backward search knobs remain `prepare_h_num_stages` and
`group_reduce_block_size`; `block_DV=128` is fixed for fused backward.

## Validation and output ordering

The tuner sets `ref_prog`, `manual_check_prog`, and `supply_prog`, causing stock
`AutoTuner._benchmark_target` to call `Profiler.manual_assert_close`. The shared
`GDNPipelineAdapter` solves the fact that `ProfileArgs` has one reference callback
per tuner while each candidate has its own runner.

The reference callback returns outputs in exactly the adapter order:

| direction | actual and reference order |
| --- | --- |
| fwd | `o`, `final_state` |
| bwd | `dq`, `dk`, `dv`, `db`, `dg`, `dh0` |

The manual checker wraps those sequences in `ForwardResult`/`BackwardResult` and
calls the existing `assert_forward`/`assert_backward`. These use
`gdr_common.assert_relative` (relative L2, **RTOL=0.02**) and the existing
backing-buffer padding assertions. Ordinary `allclose`/`assert_allclose` is not
used as a substitute. Dense search inputs have no varlen tail; when a varlen
fixture is supplied, the same manual path checks its padding exactly.

The source-style float64 forward reference maps `(g, o, A, h, final_state)` to
`(o, final_state)`. Backward keeps the public API's `db, dg` order rather than
the lower-level fused kernel's `dg, db` order.

## Source and configuration evidence

`GDNCandidateKernel.describe_configuration()` returns the real GDN config,
fixture, output order, public callable, and backward artifact policy.
`get_kernel_source()` combines that description with source inspected from the
actual public `chunk_gated_delta_rule_fwd`/`bwd`, its GDN wrappers, and their
TileLang factories (cumsum, KKT/fused-fwd or prepare_h/fused-bwd/GQA reduction).
It never returns unrelated placeholder source.

This source text is provenance for `AutotuneResult.libcode`; it is **not** labeled
as generated CUDA. Generated CUDA and executor artifacts remain in the run-owned
TileLang cache. The callback-based AutoTuner result cache intentionally has no
stable cache key, so warm-track reuse comes from the real pipeline compilation
cache, not from reusing a stale best-result object.

## Five arms and candidate denominators

`run_gdn_search(direction, arm, top_k, output_dir, warmup, repeat, seed)` uses:

1. `make_config_space(direction, hardware_path)`;
2. `CudaCostModel.from_files(..., require_measured=True)` with the measured
   calibration manifest and the actual fixture shape;
3. `GDNCandidateCostModel` to remove audit-only `direction` and fixed fwd fields
   before calling the analytical plugin;
4. `GuidedAutoTuner` and `SearchExperiment`.

The candidate counts are bounded as follows (`K=2` for fwd and `K=3` for bwd by
default):

| arm | fwd denominator | bwd denominator | selection |
| --- | ---: | ---: | --- |
| `stock-raw` | R=4 | R=9 | all raw configs, no solver/model |
| `z3-legal-exhaustive` | U=3 | U=6 | all retained legal/unknown configs |
| `z3-cm-topk` | min(K,U)=2 | min(K,U)=3 | model-ranked GuidedSearch top-K |
| `budget-raw` | min(K,U)=2 | min(K,U)=3 | first K of the same legal universe |
| `budget-random` | min(K,U)=2 | min(K,U)=3 | seeded sample without replacement |

Known invalid raw candidates in `stock-raw` are not silently deleted. Compile,
validation, or profile failures remain in `events.jsonl`, `trials`, `errors`, and
the count fields. `has_recorded_candidate_failures` makes such partial runs
explicit. Factory/tuner exceptions produce `failure.json`/error summaries and
are re-raised; they are not converted into successful results.

`compile_attempts` counts candidate initializations (the real public JIT trigger),
not an inferred number of NVCC invocations. `validation.attempts` and
`profile.attempts` come from the actual stock profiler hooks. Reference timing is
saved separately as `reference_latency_ms`; it is not added to the candidate
profile count.

## Run directories, logs, and cache tracks

Each arm/repetition uses `SearchExperiment` and contains:

- `events.jsonl` and `summary.json`;
- `config.json`, including the measured model card;
- `environment.json`, with an allowlisted environment rather than secrets;
- `stdout.log` and `stderr.log`, capturing Python and native file descriptors;
- `before.json`, `after.json`, run-owned `cache/`, and `failure.json` on errors.

Cold mode creates a fresh enabled cache. Warm mode runs a prewarm followed by a
fresh tuner in the same run-owned cache. In addition to `TILELANG_CACHE_DIR`, the
driver redirects `CUDA_CACHE_PATH`, `TORCH_EXTENSIONS_DIR`, `TRITON_CACHE_DIR`,
`XDG_CACHE_HOME`, and `TMPDIR` below that run's `cache/`; no-cache mode also sets
`CUDA_CACHE_DISABLE=1`. Original process environment values are restored after
each factory/invocation. `no-cache` is available for sensitivity checks. No
command removes or cleans a shared cache. Candidate initialization is forced to
one compile worker, pipelining/multi-GPU/early-stop are disabled, and all arms and
cache tracks execute sequentially.

`run_all_arms(direction, output_dir, warmup, repeat)` runs all five arms on the
cold track and then all five on the warm track, using fwd K=2 or bwd K=3. It
writes `<output_dir>/<direction>/all_arms_summary.json` after every completed arm
and on failure.

## CLI

Run from the repository root (GPU execution is required for a real search):

```bash
# One forward guided run on a fresh cache.
scripts/run_gdn_search.sh \
  --direction fwd --arm z3-cm-topk \
  --output-dir evidence/G6/gdn-search-single \
  --warmup 10 --repeat 50 --seed 42

# All five backward arms, cold then warm, with the default K=3.
scripts/run_gdn_search.sh \
  --direction bwd --arm all \
  --output-dir evidence/G6/gdn-search-all \
  --warmup 10 --repeat 50

# Explicit no-cache sensitivity run.
scripts/run_gdn_search.sh \
  --direction fwd --arm z3-legal-exhaustive --cache-mode no-cache \
  --output-dir evidence/G6/gdn-search-no-cache
```

`scripts/run_gdn_search.sh` only chooses the Python interpreter, sets import and
serial-tuning environment variables, and invokes this CLI. It does not build,
delete caches, or launch parallel workers.

## CPU/mock verification

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python3 benchmark/z3_cost_model/gdn/tests/test_autotune.py -v

# Equivalent pytest invocation without the GPU-detecting shared conftest:
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -q --noconftest -p no:cacheprovider \
  benchmark/z3_cost_model/gdn/tests/test_autotune.py
```

These tests use mock runners/profilers and the CPU-safe experiment machinery.
They do not initialize CUDA, compile a kernel, or execute a GPU pipeline. Real
search results and performance claims are outside the scope of those tests.
