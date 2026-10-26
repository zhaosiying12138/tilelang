# GDN / FlashQLA core port

## Upstream and provenance

- Upstream: `https://github.com/QwenLM/FlashQLA.git`
- Locked commit: `da06429d54b0f577de0a638f451ac8f0b395e0ac` (`v0.1.3`, package version 0.1.3)
- License: MIT, Copyright (c) 2026 Qwen, Alibaba. The unmodified license is at `vendor/LICENSE`; source copyright headers remain in the vendored files.
- `/home/zhaosiying/codebase/FlashQLA` was clean (`git status --porcelain` empty) when copied and when the manifest was written.
- `manifest/gdn_upstream.json` records the revision, remote, license hash, source-clean check, core contract, and direction-specific search spaces.

The complete tracked `flash_qla` package was copied with `git archive`, not reimplemented. This includes all 44 tracked package files (41 Python files plus the autocp README/CSV fixtures), real TileLang forward/backward kernels, public autograd and orchestration, and CP support. No Triton or MLIR substitute is present.

Source mappings:

| FlashQLA source | Destination |
| --- | --- |
| `LICENSE` | `vendor/LICENSE` |
| `flash_qla/**` | `vendor/flash_qla/**` |
| `tests/ref_gdr.py` | `tests/ref_gdr.py` |
| `tests/gdr_common.py` | `tests/gdr_common.py` |
| `tests/test_gdr_unit.py` | `tests/test_gdr_unit.py` |
| `tests/arch_utils.py`, `tests/conftest.py` | same relative paths |
| `pytest.ini` | `pytest.ini` |

`ref_gdr.py`, `gdr_common.py`, `test_gdr_unit.py`, `arch_utils.py`, and `pytest.ini` are byte-identical to upstream. `tests/conftest.py` only adds owned test/package paths so flat upstream imports resolve to this vendor tree without installation.

## Explicit vendored adaptations

Only the following ten files under `vendor/flash_qla` differ from the locked upstream package:

1. `ops/gated_delta_rule/chunk/__init__.py`
   - Adds optional `block_DV` and `prepare_h_num_stages` to the documented forward/backward and high-level autograd APIs.
   - Adds `group_reduce_block_size=16` only to public backward and the high-level autograd path. It is saved in the autograd context and passed to both dq/dk GQA reductions; the additional non-tensor backward return slot keeps autograd arity correct.
   - Validates non-default GQA group tiles before launching backward kernels.
   - Sets `aggregate_card_state = None` in the SM120/SM121 branch. Upstream CP preprocessing imports this symbol before its no-CP early return, although SM120 does not export it. Explicit `None` fixes no-CP import without fabricating the missing kernel; combined inter+intra CP still reaches the existing `NotImplementedError` check.
2. `ops/gated_delta_rule/chunk/cp/preprocess.py`
   - Threads `prepare_h_num_stages` through all three `fused_gdr_h` call sites.
3. `ops/gated_delta_rule/chunk/{hopper,blackwell,blackwell_sm120}/fused_fwd.py`
   - Exposes `block_DV` on the Python wrapper. `None` preserves upstream behavior: the existing grid-based choice on Hopper/Blackwell and 128 on SM120.
4. `ops/gated_delta_rule/chunk/{hopper,blackwell,blackwell_sm120}/prepare_h.py`
   - Exposes `num_stages=2` and passes it to the existing TileLang factory, whose upstream default was already 2.
5. `ops/utils/group_reduce.py`
   - Exposes the factory's existing `block_size=16` as `group_reduce_vector(..., block_size=16)` and passes it through explicitly.
   - Default 16 retains upstream partial-tile behavior. A non-default block size requires `num_tokens % block_size == 0`, otherwise a clear `ValueError` is raised before factory invocation.
6. `ops/gated_delta_rule/chunk/blackwell_sm120/fused_bwd.py`
   - Always writes the computed `dh0`. Upstream guarded the store with `use_dht`, which returned an all-zero `dh0` when `h0` existed but `dht` was absent, contradicting the mathematical reference. The other backward math, 512-thread roles and mbarrier protocol are unchanged.

No `block_DV` parameter was added to `fused_bwd` or its builder: none exists upstream. Apart from the documented `dh0` store fix, no TileLang `prim_func`, 512-thread warp specialization, mbarrier arrival count, or other warp/mbarrier protocol code was changed. There is no monkeypatch; all tuning values use normal public function arguments.

## Runner and fixed core contract

`runner.py` provides `GDNRunner`, `make_runner`, `make_fixed_inputs`, and explicit result containers.

- `forward()` calls public `chunk_gated_delta_rule_fwd`, including cumsum, KKT solve, CP preprocessing, and fused forward orchestration. It returns `o` and `final_state`, plus `g`, `A`, `h`, and `cp_cache` needed by backward.
- `backward()` calls public `chunk_gated_delta_rule_bwd`, including prepare_h, fused backward, GQA reduction, and reverse gate cumsum. It returns `dq, dk, dv, db, dg, dh0`. If necessary it first runs public forward to obtain saved values; it never calls a raw fused kernel itself.
- Both paths explicitly use `auto_cp=False`. Input generation reuses upstream `gdr_common.make_inputs`, with seed 42, bf16 data, and K=DV=128.
- Backward computes forward artifacts with a separate fixed `forward_block_DV=64`, because the nominal 128 configuration exceeds the verified 101376 B shared-memory limit. This is a launchability adaptation, not an additional backward search knob; backward performance scope starts after those forward artifacts are available.
- Core fixtures use `num_k_heads=2` and `num_v_heads=4`, so Hg=2 < H=4 and every backward group-tile config reaches the real dq/dk `group_reduce_vector` path. Forward can share the same GQA input.
- Raw-config correctness uses T=128, making group tiles 8, 16, and 32 all legal. The runner/factory rejects attempts to assign forward-only or backward-fixed knobs to the wrong direction.
- The loader verifies that imported modules come from this directory's vendor/tests trees and raises on a conflicting pre-imported `flash_qla` rather than silently mixing installations.

The fixed contract is `auto_cp=False`, bf16, K=DV=128, chunk=32, T=128 for the search, and the unchanged 512-thread protocol. Other architectures and CP implementations remain vendored but are outside this core correctness claim.

## Direction-specific configuration spaces and feasibility

`config.py` is CPU-only. `GDNConfig` contains `block_DV=128`, `prepare_h_num_stages=2`, and `group_reduce_block_size=16`, but only real direction-specific fields vary:

### Forward: 4 configs

- Variable: `block_DV = [16, 32, 64, 128]`, consumed by `fused_gdr_fwd`.
- Fixed: `prepare_h_num_stages=2`, `group_reduce_block_size=16`.
- Public no-CP forward has no prepare_h consumer, so stage is not a forward search dimension.

### Backward: 9 configs

- Variables: `prepare_h_num_stages=[1,2,3]` and `group_reduce_block_size=[8,16,32]`.
- Fixed: `block_DV=128`; fused_bwd has no such parameter.
- prepare_h stage reaches `fused_gdr_h`/CP preprocessing. Group tile reaches dq/dk `group_reduce_vector` under the core GQA fixture.

The old 12-per-direction cross-products and their semantic duplicates are gone. Fixed fields and semantic de-duplication are not represented as UNSAT.

Only source/shape-provable checks are implemented: positive `block_DV` dividing DV=128, integer `prepare_h_num_stages >= 1`, positive group tile, non-default group-tile divisibility by T, the fixed core contract, and the fixed 512-thread/32-thread-warp protocol. Default group tile 16 does not acquire a new T-divisibility requirement. No old evidence is used to declare `block_DV=128` UNSAT.

Every raw configuration currently has structural status SAT. Overall status remains UNKNOWN because no IR/ptxas/device-spec resource evaluator is connected. `ResourceChecker` is the extension point for later shared-memory, register, and launch feasibility checks; UNSAT dominates UNKNOWN when independent evidence becomes available.

## Correctness coverage

`correctness.py` and `tests/test_runner_core.py` use the vendored pure-PyTorch float64 reference and upstream comparison policy:

- relative L2 error with `RTOL = 0.02`;
- deterministic fixed-seed GQA fixtures;
- source-style repeated public-pipeline checks, with 1000 iterations available for slow determinism runs;
- complete fwd outputs and all six bwd gradients rather than raw-kernel-only checks.

Additional checks cover:

- every element in the varlen backing-buffer padding of `o/dq/dk/dv/db/dg` is exactly zero, not merely NaN-free;
- `final_state` is checked even without `h0`;
- `dh0` is checked when `h0` exists and `dht` is absent, and is `None` when `h0` is absent;
- all 4 forward and 9 backward raw configs, with both backward knobs consumed on every backward case;
- default group tile 16 on non-divisible T and early rejection of non-default group tiles on non-divisible T.

The upstream five test functions in `test_gdr_unit.py` are retained, including long cases and 1000-iteration determinism tests. Six new GPU test functions cover the runner gaps, direction-specific raw spaces, and core no-CP determinism.

## Remaining limitations

- Forward correctness has passed for `block_DV=16/32/64`; `block_DV=128` fails at launch with 112640 B dynamic shared memory, exceeding the 101376 B device opt-in limit. Backward passes dense, h0/no-dht and padded cases after the documented forward-artifact and `dh0` fixes. Performance search and timing have not yet run.
- `--check-configs` is host-only. `--correctness` and pytest GPU entries require a supported CUDA GPU; like upstream, their float64 PyTorch reference runs on the input device.
- The new core correctness entry is scoped to SM120. SM121 and other architecture sources are present but are not validated targets of this first-round port.
- CP and multi-card paths are vendored but not exercised by the core runner; `auto_cp=False` is intentional. Combined SM120 inter+intra CP remains unavailable because there is no `aggregate_card_state` implementation.
- Register and other launch-feasibility checks remain UNKNOWN unless a verified checker supplies evidence. The forward shared-memory rule is derived from generated code and validated on all four block sizes; it does not prove complete kernel correctness or occupancy.
- Slow tests include long upstream inputs, 13 direction-specific raw-config GPU cases, and repeated 1000-iteration checks.

## Reproduction

Run from `/home/zhaosiying/codebase/tilelang-cuda-z3`.

### Host-only checks (executed for this port)

```bash
python3 benchmark/z3_cost_model/gdn/cli.py --check-configs
python3 -m unittest discover \
  -s benchmark/z3_cost_model/gdn/tests -p 'test_config.py' -v

python3 - <<'PY'
import ast
from pathlib import Path
root = Path('benchmark/z3_cost_model/gdn')
paths = sorted(root.rglob('*.py'))
for path in paths:
    ast.parse(path.read_text(), filename=str(path))
print(f'Parsed {len(paths)} Python files')
PY

# Expected differences are exactly the ten adapted vendor files listed above.
diff -qr \
  /home/zhaosiying/codebase/FlashQLA/flash_qla \
  benchmark/z3_cost_model/gdn/vendor/flash_qla \
  --exclude=__pycache__ || true
sha256sum \
  /home/zhaosiying/codebase/FlashQLA/LICENSE \
  benchmark/z3_cost_model/gdn/vendor/LICENSE
```

Observed host results: 22/22 CPU unit tests passed; the CLI reported 4 forward and 9 backward structural-SAT/resource-UNKNOWN configs; all 53 Python files parsed. These are static/host checks only.

### GPU correctness entries (executed as reported above)

```bash
# Public fwd/bwd, dense + h0/no-dht + padded cases, default config.
python3 benchmark/z3_cost_model/gdn/cli.py --correctness fwd
python3 benchmark/z3_cost_model/gdn/cli.py --correctness bwd

# Explicit direction-valid tuning examples; these are correctness, not benchmarks.
python3 benchmark/z3_cost_model/gdn/cli.py \
  --correctness fwd --block-dv 64
python3 benchmark/z3_cost_model/gdn/cli.py \
  --correctness bwd --prepare-h-num-stages 3 --group-reduce-block-size 32

# Source-style repeated determinism checks (slow).
python3 benchmark/z3_cost_model/gdn/cli.py \
  --correctness all --determinism-iters 1000

# New fast gap tests, then all 4+9 raw-config cases (slow).
python3 -m pytest -c benchmark/z3_cost_model/gdn/pytest.ini \
  benchmark/z3_cost_model/gdn/tests/test_runner_core.py -m 'not slow'
python3 -m pytest -c benchmark/z3_cost_model/gdn/pytest.ini \
  benchmark/z3_cost_model/gdn/tests/test_runner_core.py -k 'raw_forward_config or raw_backward_config'

# Preserved upstream suite and core determinism entries (slow where marked).
python3 -m pytest -c benchmark/z3_cost_model/gdn/pytest.ini \
  benchmark/z3_cost_model/gdn/tests/test_gdr_unit.py -m 'not slow'
python3 -m pytest -c benchmark/z3_cost_model/gdn/pytest.ini \
  benchmark/z3_cost_model/gdn/tests/test_runner_core.py -k determinism
```
