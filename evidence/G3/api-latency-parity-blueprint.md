# CUDA Z3 public-API, latency, legalization, and layout implementation blueprint

## 0. How to use this document

This is an implementation plan for the remaining rows named in `evidence/G3/parity-84-reaudit-agent.md`, not a claim that those rows are complete. All paths below are relative to `/home/zhaosiying/codebase/tilelang-cuda-z3`. No implementation or GPU task was run while preparing this plan.

Each patch is independently reviewable and has a bounded owner set. **Land in dependency order.** A patch must retain default-off behavior for unannotated modules, must not fall back to stock scheduling after a Z3 request is malformed or infeasible, and must not turn an unknown operation into a measured/free/dense equivalent. Host tests are required for every patch. Device tests are validation work for the subsequent implementation campaign; none was run for this blueprint.

### 0.1 Source facts that constrain the design

The following are current source behaviors, not proposed semantics.

- Ascend `T.Task`, `T.Stage`, conflict hints, version requests, manual rings, exact addresses, and unlimit scopes are defined in `tilelang/ascend/language/schedule_hint.py` and `tilelang/ascend/language/annotations.py`. Their staged IR is consumed by `src/ascend/transform/materialize_schedule_units.cc`, `normalize_conflict_hints.cc`, `normalize_buffer_version.cc`, and `auto_schedule.cc`.
- Ascend manual scheduling defaults absent stages to zero, derives an absent `num_stages` as `max(manual_stage)+1`, and derives per-pipe issue order from source order (`src/ascend/transform/auto_schedule.cc`). CUDA currently requires a positive `num_stages` in `src/transform/z3_pipeline_planning.cc` and has no `T.Stage` API.
- Ascend dual-copy splitting and RMW legalization happen before scheduling (`src/ascend/transform/rewrite_dual_copy.cc`, `legalize_simd_merging.cc`). OOB clamps and exact padding fills also precede scheduling (`src/ascend/transform/insert_oob_padding.cc`). CUDA `LegalizeSafeMemoryAccess` currently runs after Z3 planning/injection in `tilelang/cuda/pipeline.py`.
- Ascend cost extraction preserves atomic blocks, dynamic region bounds, block-scaled identity, dual-copy per-lane payload, and recursive serial/branch/repeat composition (`src/ascend/transform/estimate_latency.cc`). CUDA `src/transform/z3_task_analysis.cc` currently aggregates static features, sums both branch arms, and rejects `While`.
- CUDA block-scaled GEMM is real generic work: `src/op/gemm_blockscaled.h` defines `GemmBlockScaledNode` **as a subclass of `GemmNode`**, with explicit SFA/SFB regions. Consequently, the current `HandleGemm(const GemmNode *)` path can price it as dense and omit scale cost unless the derived type is handled first. CUDA lowering supports SM100 TCGEN05 and SM120 block-scaled mma.sync in `src/cuda/op/gemm_blockscaled.cc`.
- CUDA already has the required physical mechanisms for generic copy and drain costing: `CopyInst::{kCPAsync,kBulkLoad,kBulkStore,kTMemLoad,kTMemStore}` in `src/cuda/op/copy.h`, instruction selection in `src/cuda/op/copy_analysis.cc`, and TMEM drain plans in `src/cuda/op/copy.cc`.
- CUDA physical TMEM capacity and allocation units are explicit in `src/cuda/transform/lower_shared_tmem.cc`: 128 datapaths, 512 b32 columns, power-of-two allocations of at least 32 columns, and four-column scale-factor alignment. TMEM must therefore use a column capacity group, not a fake byte-offset API.
- CUDA compact sub-byte size logic already exists in `src/transform/common/storage_size.h`. Shared offset assignment and alignment checks exist in `src/transform/merge_shared_memory_allocations.cc`, but there is no public exact-address contract.
- Current injection only expands one annotated layout over a leading version dimension (`ExpandAnnotatedLayoutForMultiVersionedBuffer` in `src/transform/inject_pipeline.cc`). It does not represent independent per-version offsets/alignment.
- Ascend selected versions are merged with `max` across owners, but its own comment says per-owner limits are not authoritative and a global lifetime-aware check is still needed (`MergeSelectedBufferVersion` in `src/ascend/transform/auto_schedule.cc`). Ascend per-version UB padding likewise is not fed back into its solver capacity check (`src/ascend/transform/rewrite_buffer_version_layout.cc`). CUDA must implement the final check rather than copy those limitations.
- CUDA has no global HF32 setter analogous to Ascend. TF32 is a per-GEMM dtype/lowering property. Existing generic stateful operations include `tl.loop_break`, proxy/TCGEN05 fences, WGMMA/TCGEN05 operations, and register-allocation intrinsics. API-011 must analyze those actual operations rather than invent an HF32 mode setter.

### 0.2 Row-to-patch coverage

| Rows | Required patches |
|---|---|
| Z3-012 | P01 |
| Z3-013 | P13, P14, P16, P17; final placement is checked by P18 for exact-addressed shared pools |
| Z3-016 | P13, P14, P16 |
| Z3-017 | P20; expanded determinism matrix reruns after P01-P19 |
| LAT-001 | P01 for atomic boundaries, P06 for richer native features |
| LAT-002 | P04 for dual payload, P05/P06 for dynamic regions/trips, P19 for actual mode-state dataflow |
| LAT-005 | P10 |
| LAT-006 | P06 |
| LAT-007 | P07 |
| LAT-008 | P08; specialized drain/scale paths complete through P09/P10 |
| LAT-009 | P04 and P08 |
| LAT-010 | P08 and P10 |
| LAT-011 | P04 and P09 |
| LAT-013 | P06 |
| LAT-014 | P11 |
| PASS-018 | P13/P14 |
| PASS-019 | P17; P18 proves exact placement survives flattening where requested |
| PASS-020 | P04 |
| PASS-021 | P03 |
| PASS-022 | P05 |
| PASS-025 | P02 |
| API-001 | P01, with composed native cost support in P06 |
| API-002 | P01; atomic unroll interop in P02 |
| API-004 | P12 |
| API-005 | P13/P14 |
| API-006 | P15 |
| API-007 | P16; P17/P18 retain final physical-fit errors |
| API-008 | P18 |
| API-009 | P07 |
| API-011 | P03 and P19 |
| API-013 | P04 |

P11 necessarily lands the enabling guarded-candidate mechanism associated with API-003 even though API-003 was not one of the requested rows. Do not mark API-003 closed unless its full exactly-once, single-pipe, synchronization-migration, and runtime-dispatch contract is also accepted.

### 0.3 Dependency graph

```text
P01 Task/Stage normalization
 ├─ P02 explicit-only atomic unroll
 ├─ P03 RMW legalization ─────────────────────────────┐
 ├─ P04 dual copy ───────────────────────────────┐    │
 ├─ P06 native dynamic/composed costs ──────────┐│    │
 ├─ P07 warp-region override ──────────────────┤│    │
 ├─ P11 per-role candidates/runtime ───────────┤│    │
 └─ P19 stateful dependencies ──────────────────┤│    │
P05 pre-schedule OOB legalization ──────────────┤│    │
P08 copy descriptors/geometry ◄─────────────────┘│    │
 ├─ P09 accumulator drain                       │    │
 └─ P10 block-scaled GEMM + scale copies ◄──────┼────┤
P12 conflict hints ──────────────────────────────┤    │
P13 version request schema                       │    │
 ├─ P14 iteration/counter modes + selected max   │    │
 └─ P15 manual rings                             │    │
P17 per-version physical layouts ◄───────────────┘    │
P16 capacity/unlimit/final proof ◄──── P14/P17 ───────┤
P18 exact shared addresses ◄────────── P13/P16/P17 ───┤
P20 repeated complete-compile determinism ◄────────────┘
```

---

## P01 — Public atomic `Task`/`Stage` markers and manual normalization

**Rows:** Z3-012, LAT-001 (boundary part), API-001, API-002.

### Ownership

- **New:** `tilelang/cuda/language/schedule_hint.py`.
- **Edit:** `tilelang/cuda/language/__init__.py`, `tilelang/transform/__init__.py`, `tilelang/cuda/pipeline.py`.
- **New:** `src/transform/normalize_z3_schedule_units.cc`, `src/transform/common/z3_schedule_hints.h`.
- **Edit:** `src/transform/z3_task_analysis.cc`, `src/transform/z3_pipeline_planning.cc`, `tilelang/language/annotations.py` only to make the old helper reject mixed old/new manual schemas clearly.
- **Tests:** `testing/python/transform/test_tilelang_z3_schedule_units.py` (new), `test_tilelang_transform_z3_pipeline_planning.py`, `test_tilelang_z3_task_analysis.py`, `testing/python/language/test_tilelang_cuda_z3_pipeline.py`.

### Public schema

```python
T.Task(latency: int | None = None, ii: int | None = None)
T.Stage(stage: int)
```

Match Ascend validation: `latency` is a non-negative Python integer, `ii` is positive, bool is rejected, and a pair must satisfy `latency >= ii`. Omitted values are inferred independently. Do not add core masks or resource flags to the public helper.

Emit:

- `AttrStmt(node={"latency"?: int64, "ii"?: int64}, attr_key="tl.cuda_z3_task", value=1, body=...)`.
- `AttrStmt(node=int64_stage, attr_key="tl.cuda_z3_stage", value=1, body=...)`.

`T.Task` is one atomic scheduler task; statements within it issue serially without internal scheduler synchronization. `T.Stage` must wrap complete tasks and is illegal inside a task, parallel tile-op body, or another stage. `with T.Stage(n), T.Task(): ...` is the canonical nesting.

### Implementation

1. `NormalizeZ3ScheduleUnits` walks each `tl_z3_pipeline` loop before `InferZ3PipelineProblem`. It materializes explicit task markers and preserves the existing flat-statement path when no new markers are present.
2. Treat each explicit task as opaque for dependency closure and statement counting. Recurse into it only for feature/cost extraction and for collecting the task's aggregate read/write footprint. A replayable Bind inside a task cannot become a separate solver task or clone outside the marker.
3. Keep tile ops, `ParallelOp` roots, and later warp-region markers atomic. Unsupported nesting, a marker that crosses a pipeline-loop boundary, or a task with no executable statement is an error.
4. Compute an internal `execution_domain` (`copy`, `tensor_core`, `vector`, `scalar`, `async_proxy`) separately from the seven-bit resource mask. One task must have exactly one domain; global load plus shared-memory resource bits in one copy still has domain `copy`. Reject cross-domain task bodies rather than claiming the source single-pipe contract.
5. If any direct child has a stage marker, set `tl_z3_pipeline_manual_schedule=True`; every unannotated sibling defaults to stage zero. Stage markers are removed only after their values are projected to `tl_z3_pipeline_manual_stages` in scheduled-task order.
6. If `num_stages` is zero/absent in this manual mode, set it to `max(manual_stage)+1`. An explicit positive value remains the automatic buffer-version search bound; it does not truncate stage numbers. Final CUDA depth remains the maximum of the version bound, selected versions, and `max(stage)+1`.
7. Manual mode retains current `pipe_order_deps=InferResourcePairs(resource_flags)` source-order behavior. Nonzero stages still require `enable_offset=True`.
8. Reject all ambiguous interoperation: `T.Stage` plus old `manual_stages`, `T.Stage` plus `Pipelined(order=.../stage=...)`, handwritten pipeline order/stage attrs, nested stage markers, negative/nonconstant stage values, and stage/task markers outside an opted-in Z3 loop. Never silently select one schema.

### Conservative rules

- Existing `z3_pipeline_annotations(manual_schedule=True, manual_stages=...)` remains valid alone and behaves identically.
- A task-level explicit latency/II never replaces dependency footprints. If complete regions cannot be inferred, compilation still fails even when both costs are provided.
- A manual infeasibility is fatal; no serial or stock-pipeline fallback.

### Tests

- **Host:** frontend type/range checks; marker shapes; atomic statement counts; serial multi-statement task cost inputs; no internal Bind task leakage; cross-domain rejection; partial stages normalized to zero; source-opposed stages; default depth `max+1`; explicit `num_stages` as version bound only; old/new schema conflicts; per-pipe source order infeasible without fallback; idempotence.
- **Device:** A+B written with `T.Task`; a two-stage load/compute/store pipeline with `T.Stage(0/1)`, source order opposing stage order, and omitted `num_stages`; exact output and stage/order audit. Repeat once with old manual annotations to establish equivalent behavior.

---

## P02 — Explicit-only unrolling that preserves atomic regions

**Rows:** PASS-025, API-002 (interop).

**Depends on:** P01.

### Ownership

- **New:** `src/transform/unroll_loop_skip_z3_atomic.cc`.
- **Edit:** `tilelang/cuda/pipeline.py`, `tilelang/transform/__init__.py`, `src/transform/common/z3_schedule_hints.h`.
- **Tests:** `testing/python/transform/test_tilelang_transform_unroll_loop.py`, `test_tilelang_z3_schedule_units.py`, `test_tilelang_z3_task_analysis.py`.

### Implementation

1. Add `UnrollLoopSkipZ3Atomic`, using the Ascend explicit-unroll algorithm only as a semantic reference, not as a CUDA pass dependency.
2. Expand only `ForKind::kUnrolled` with `pragma_unroll_explicit=True`, constant non-negative extent, preserving non-unit steps and freshening per-iteration definitions. Process nested explicit requests even under retained loops.
3. Return atomic task/warp-region/tile-op bodies without visiting their internals. Do not perform generic auto-unroll, `var_touched_local` expansion, or factor unrolling before Z3 analysis.
4. In `CUDAPassPipelineBodyPrologue`, use this pass instead of generic `UnrollLoop` only for a module with Z3 annotations. Retain the stock pass for other modules and retain the later generic unroll after scheduling.
5. Run Simplify after expansion so the analyzer and planner see identical statements.

### Fallback and tests

- Non-explicit, dynamic, and `pragma_unroll_explicit=False` loops remain loops; an explicitly requested nonconstant extent is an error.
- **Host matrix:** explicit true/false/absent; serial versus unrolled; nested explicit under retained outer; non-unit step; fresh Buffer/Var definitions; atomic task body unchanged; `ParallelOp` body unchanged; task count before/after inference; pass-order assertion proving this pass precedes inference/planning.
- **Device:** two explicitly unrolled copies feeding one atomic compute task; exact result and expected task schedule. A non-explicit loop must produce a different task count but identical numerics.

---

## P03 — Pre-analysis read-modify-write legalization

**Rows:** PASS-021, API-011 (RMW part).

**Depends on:** P01.

### Ownership

- **New:** `src/transform/legalize_z3_read_modify_write.cc`.
- **Edit:** `tilelang/cuda/pipeline.py`, `tilelang/transform/__init__.py`, `src/transform/z3_task_analysis.cc`, `src/transform/z3_pipeline_planning.cc`.
- **Tests:** `testing/python/transform/test_tilelang_z3_rmw_legalization.py` (new), `test_tilelang_z3_task_analysis.py`.

### Implementation

1. Run immediately before P02/inference, after tile-op parsing is available and before any dependency collection.
2. Keep ordinary `BufferStore` RMW as the canonical explicit form: destination is a write, stored value/indices/predicate are visited for reads. Add direct tests rather than rewriting it into a more opaque call.
3. For a recognized value-returning intrinsic with read-modify-write destination semantics, rewrite `dst[i] = op(...)` into a void `Evaluate` whose first operand is `tl.access_ptr(BufferLoad(dst,i), 1, rw_mask=3)`. Reject the same operation used as a pure value with no mutable destination. Preserve a store predicate by wrapping the resulting statement in `IfThenElse`.
4. For tile ops, consume `GetReadBeforeWriteRegions()` in both task collectors in addition to `GetAccessRegions()`. This is mandatory for accumulator and SFA/SFB correctness; do not infer read-before-write solely from a write region.
5. Verify after rewriting that every declared RMW destination has both read and write coverage. Unknown/stateful calls remain conservative errors under P19 rather than receiving guessed RMW semantics.

### Tests

- **Host:** direct store RMW RAW/WAR/WAW matrix; merging-style intrinsic; zeroing/non-RMW intrinsic unchanged; missing mutable destination error; predicated destination preserved; nested predicate and index reads; tile-op accumulator plus block-scaled SFA/SFB read-before-write regions; idempotence and placement before inference.
- **Device:** predicated in-place vector update with untouched neighbor sentinel, and an accumulating GEMM with `clear_accum` true/false to distinguish destination reads.

---

## P04 — Software dual-copy API, role-indexed rewrite, and equivalence

**Rows:** PASS-020, API-013, LAT-002 (dual payload), LAT-009 (dual-warp geometry), LAT-011 (dual drain geometry).

**Depends on:** P01.

### Ownership

- **Edit:** `tilelang/cuda/language/copy_op.py`, `tilelang/cuda/language/__init__.py`, `tilelang/cuda/pipeline.py`, `tilelang/cuda/transform/__init__.py`.
- **New:** `src/cuda/transform/rewrite_dual_copy.cc`.
- **Edit:** `src/transform/z3_task_analysis.cc`, `src/cuda/op/copy_analysis.cc` only for shared descriptor queries used by P08.
- **Tests:** `testing/python/transform/test_tilelang_cuda_rewrite_dual_copy.py` (new), `testing/python/cuda/test_cuda_latency_model.py`, `testing/python/language/test_tilelang_cuda_dual_copy.py` (new).

### Public and staged schema

```python
T.dual_copy(src, dst, *, split: Literal["m", "n"] | None = None,
            annotations: dict | None = None)
```

- Rank-one regions must have an exact 2:1 extent ratio. Rank >=2 permits exactly one trailing-axis M or N split. `split` is an optional disambiguator for dynamic regions; inference from static/allocation shapes is used otherwise.
- The larger region is the full logical tile. Static full extents must be even. Other ranks/ratios and ambiguous axes are errors.
- Stage ordinary CUDA copy annotations with internal keys `cuda_dual_copy_axis` (`0` for M/only axis, `1` for N) and `cuda_dual_copy_source_is_half`. Do not expose Ascend `dual_dst_ctl` or `unit_flag_ctrl` on CUDA.

### Implementation

1. Run before P03/P02, OOB, layout, and task analysis. Reuse an enclosing explicit two-warp role scope; otherwise synthesize a role variable equivalent to `get_warp_idx() & 1` and record that the operation requires exactly two participating warps.
2. Rewrite the full-side region to `min + role*half_extent` and `half_extent`; remove all dual-copy-only annotations. The emitted ordinary copy retains user cache/TMA hints only when legal for the narrowed region.
3. For compiler-owned half-side storage, remap it to per-role storage with a leading two-role dimension and rewrite every alias/access consistently in bit units. This is required for CTA-shared storage, where two warps do not have Ascend-style private UBs.
4. If the half-side buffer is externally supplied, has an unprovable alias/layout, or cannot be expanded without changing its public shape, fail before scheduling with an allocation diagnostic. Do not emit two warps overwriting one shared half tile.
5. If a role scope is absent or does not contain exactly two eligible warps, either materialize a local two-warp scope when legal or reject. There is no single-warp semantic fallback that can preserve the source 2:1 region contract automatically.
6. Record an internal audit marker mapping the original logical copy to the role-indexed ordinary copy so P08 can compare descriptors and payload.

### Cost/region contract

- One logical dual copy becomes one ordinary role-indexed copy task. Total payload and read/write footprints equal the equivalent two role-specific halves; per-role payload is half the full tile.
- Dynamic tails use allocation shapes only to infer the split axis, never to replace actual region extents. If actual dynamic extents cannot be bounded exactly enough for disjoint roles, reject rather than round unevenly.

### Tests

- **Host:** rank-one/M/N matrix; dynamic tail; odd extent; multiple candidate axes; unsupported rank; role extent !=2; source-half and destination-half forms; no unrewritten annotations; per-role allocation and alias rewrite; unexpandable external buffer error; pre/post task read/write and P08 descriptor equality; scheduling occurs only after rewrite.
- **Device:** global->shared and shared->global M/N copies on two warps, including a dynamic tail and CTA-shared per-role storage. Compare exact output with two hand-written role-indexed ordinary copies and check untouched neighbors.

---

## P05 — Materialize OOB clamps and exact padding before scheduling

**Rows:** PASS-022, LAT-002 (dynamic region part).

**Depends on:** P01. P04 ordering is `P04 -> P05` when both are present.

### Ownership

- **New:** `src/transform/legalize_z3_out_of_bounds.cc`.
- **Edit:** `tilelang/cuda/pipeline.py`, `tilelang/transform/__init__.py`, `tilelang/cuda/language/copy_op.py`, `src/transform/z3_task_analysis.cc` (fill feature recognition completed in P06).
- **Tests:** `testing/python/transform/test_tilelang_z3_oob_legalization.py` (new), `test_tilelang_transform_legalize_safe_memory_access.py`, `test_tilelang_z3_task_analysis.py`, `testing/python/language/test_tilelang_cuda_z3_oob.py` (new).

### Public/staged schema

Add an optional typed CUDA copy keyword `pad_value: PrimExpr | None = None`, stored as copy annotation `pad_value`. `None` means zero for materialized padding. Existing `annotate_safe_value` remains the late scalar-load fallback and is not a substitute for a scheduled copy fill.

### Implementation

1. For each global-facing tile copy in an opted-in Z3 region, clamp each source/destination range against its allocation shape using the current analyzer and assumptions. Rebuild the copy with semantic in-bounds ranges and explicit empty-tile guards.
2. Emit `tl.tileop.fill` for exactly the destination padding removed by clamping. Preserve row/column tails independently; do not fill an entire row when only its column tail is OOB. A source fully in bounds does not cause destination padding.
3. Order fills before the clamped copy when they write the same destination. Preserve the original outer predicate and add only the necessary tail predicates. Group paired operand fills without moving them across an intervening writer/consumer.
4. Use actual `pad_value` dtype conversion and destination-side payload. TMA zero-fill may be used only when its descriptor contract matches the requested value; nonzero padding always uses the explicit fill task.
5. Keep the existing late `LegalizeSafeMemoryAccess` as a defensive pass for non-Z3 and residual scalar accesses, but assert that pre-schedule Z3 copies no longer hide clamp/fill work from the analyzer.

### Conservative rules and tests

- Dynamic extents require a finite analyzer/buffer-shape bound. Unknown or opaque physical layouts fail; they are not treated as a one-element region. Explicit latency/II cannot waive an incomplete dependency footprint.
- **Host:** rank-one/two-axis tails; no-OOB no-op; dynamic scalar/local offsets; empty tiles; transpose; nonzero pad; exact fill read/write regions and placement before inference; neighboring storage not included; dual-copy then OOB ordering; idempotence.
- **Device:** rank-one and matrix tails for sync, cp.async, and TMA where legal; transposed GEMM operand tails; zero and nonzero padding; untouched-neighbor sentinel; dynamic offset. Validate numerics and generated schedule contains the fill task.

---

## P06 — Native dynamic extents and recursive task-cost composition

**Rows:** LAT-001 (rich features), LAT-002 (automatic dynamic regions/trips), LAT-006, LAT-013, API-001 (composed task cost).

**Depends on:** P01. P03/P05 feature forms are added as those patches land.

### Ownership

- **Edit:** `src/transform/z3_task_analysis.cc`, `src/transform/common/z3_schedule_hints.h`, `tilelang/cuda/latency_model.py`, `tilelang/cuda/task_cost.py`.
- **New:** `src/transform/common/z3_cost_expression.h`.
- **Tests:** `testing/python/transform/test_tilelang_z3_task_analysis.py`, `testing/python/cuda/test_cuda_latency_model.py`, `test_calibration_schema.py`.

### Internal/FFI schema

Introduce a strict JSON cost tree accepted by a new `tl.cuda.estimate_composed_task` FFI:

```text
{"kind":"leaf", "features": TaskFeatures}
{"kind":"serial", "children":[CostExpr, ...]}
{"kind":"repeat", "trips": int | {"upper_bound":int,"reliability":...},
 "body":CostExpr}
{"kind":"branch", "then":CostExpr, "else":CostExpr}
```

The result remains `{latency, ii, terms, resource_flags, reliability, fallback_used}`. `terms.composition` retains the tree and every leaf's features. Structural resource masks are ORed across all branches, even though branch cost is a maximum.

All native extent fields use either an integer or the existing `Extent` object form. No C++ dynamic value is stringified into a measured integer.

### Implementation

1. Replace branch-blind feature summation with a `NativeCostExpr` tree. A `T.Task` body produces one tree, not multiple scheduler tasks.
2. Serial composition uses `max(A.latency, A.ii+B.latency)` and `A.ii+B.ii`; repeat uses `(n-1)*II+latency` and `n*II`; mutually exclusive branches take component-wise maxima. Empty compositions retain the `0/0` identity only internally; a final schedulable task follows existing validation.
3. For serial/unrolled loops, compose the body then apply repeat. Use analyzer constraints, assumptions, static buffer shape, and explicit trip-bound annotations to produce a finite `Extent`. An unbounded expression raises a named diagnostic instead of becoming a measured extent of one.
4. Handle `While` through the same recursion when a finite trip upper bound is provable. Otherwise require both task-level latency and II overrides and still require complete dependency/state footprints; if either is absent, reject. This does not pretend to implement While normalization/restoration (PASS-001).
5. Evaluate pure loop-invariant scalar Binds once outside the repeat cost when they have no buffer/state read and no dependence on the loop variable. Buffer loads, state reads, and variant expressions remain repeated.
6. Model fill/cast/vector/special/unknown operations explicitly. Unknown work retains `unknown` resource/reliability and a nonfree placeholder; never use dense GEMM or zero cost as its fallback.
7. Python recursively evaluates the tree with the existing checked arithmetic and returns deterministic, sorted fallback metadata.

### Tests

- **Host:** independent serial/repeat/branch oracles; zero/one/many trips; nested branch/loop; bounded While; unbounded While rejection and explicit-override path; invariant Bind counted once; buffer-load Bind repeated; dynamic copy extents from assumptions and static buffer shape; reliability/fallback propagation; resource union across branches; int64 overflow; FFI JSON round trip.
- **Device:** a task containing serial copies and compute, a dynamic-trip serial loop, and a bounded While; exact outputs and expected composed cost annotations. No performance claim is made without the relevant calibration patch.

---

## P07 — Atomic warp/VALU region with independent latency override

**Rows:** LAT-007, API-009.

**Depends on:** P01; cost integration uses P06. P02 recognizes the new marker through the common atomic registry.

### Ownership

- **Edit:** `tilelang/cuda/language/schedule_hint.py`, `tilelang/cuda/language/__init__.py`, `src/transform/common/z3_schedule_hints.h`, `src/transform/z3_task_analysis.cc`, `tilelang/cuda/latency_model.py`.
- **Tests:** `testing/python/transform/test_tilelang_z3_schedule_units.py`, `test_tilelang_z3_task_analysis.py`, `testing/python/cuda/test_cuda_latency_model.py`, `testing/python/language/test_tilelang_cuda_z3_pipeline.py`.

### Public schema

```python
T.WarpRegion(latency: int | None = None)
```

Emit `AttrStmt(node={"latency"?: int64}, attr_key="tl.cuda_z3_warp_region", value=1, body=...)`. `None` invokes the model; an integer is an exact region-latency override. Reject bool and negative values. The region is VALU/vector-only, executes in the enclosing CUDA warp context, and is atomic to unroll and scheduling.

### Implementation

1. Validate no nested warp region, GEMM, asynchronous operation, tile-copy descriptor, or scheduler task boundary inside the marker. Ordinary vector BufferLoad/Store footprints remain visible.
2. Add `region_latency_override: int | None = None` to `TaskFeatures`, distinct from whole-task `latency_override` and `ii_override`.
3. When present, replace only the vector completion and vector-II terms. Continue computing memory, scalar, and other terms, then apply independent whole-task overrides and final `latency>=ii>0` validation.
4. Nested inside `T.Task`, a whole-task latency override does not erase the region-derived II unless an explicit task II is also supplied. Region override does not silently become a whole-task latency.

### Tests

- **Host:** region versus whole-task override independence; region-only, task-latency-only, task-II-only, and all three; invalid mixed region; nested marker; unroll preservation; scalar term retained; final validation; zero explicit region latency rejected when it cannot form a schedulable final pair.
- **Device:** vector region inside a pipelined task with exact output and no internal unrolling; inspect schedule/cost annotations before execution.

---

## P08 — Copy instruction descriptors, layout geometry, and per-warp lanes

**Rows:** LAT-008, LAT-009; required by LAT-010/LAT-011.

**Depends on:** P01 and P06; P04 supplies dual-copy equivalence cases.

### Ownership

- **Edit:** `src/cuda/op/copy.h`, `src/cuda/op/copy_analysis.cc`, `src/cuda/op/copy.cc` (read-only reuse of TMA/TMEM planning), `src/transform/z3_task_analysis.cc`, `tilelang/cuda/pipeline.py`, `tilelang/cuda/latency_model.py`, `tilelang/cuda/task_cost.py`.
- **New:** `src/cuda/op/copy_cost_features.h`.
- **Edit calibration:** `benchmark/z3_cost_model/calibration/`, `testing/python/cuda/test_calibration_schema.py`.
- **Tests:** `testing/python/transform/test_tilelang_z3_task_analysis.py`, `testing/python/cuda/test_cuda_latency_model.py`, `testing/python/language/test_tilelang_cuda_z3_pipeline.py`.

### Feature and calibration schema

Extend `MemoryAccess` without changing the meaning of existing `bytes` (total logical payload):

```text
instruction: "auto" | "sync" | "cp_async" | "tma" |
             "tmem_load" | "tmem_store" | "unknown" = "auto"
descriptor_count: Extent | int = 1
lanes: Extent | int = 1
geometry: existing coalesced/strided/unknown string
width: existing per-row contiguous byte extent
payload_origin: "max" | "source" | "destination" = "max"
role: "generic" | "scale_a" | "scale_b" | "accumulator" = "generic"
binding: string = ""                  # stable scale/accumulator identity
```

Calibration path keys may be suffixed `@cp_async`, `@tma`, `@tmem_load`, or `@tmem_store`. An unsuffixed path is not silently treated as measured instruction-specific calibration; using it as a fallback adds `memory_instruction:<kind>` to fallback provenance and fails `require_measured`.

For `N=descriptor_count`, use checked arithmetic:

```text
completion = N*(base + descriptor) + ceil(total_bytes / effective_bandwidth)
path_II = N*issue + max(ceil(total_bytes / effective_bandwidth), N*min_ii)
```

Aggregate completion over accesses, sum II within one physical path, and take max II across paths. `lanes` documents/proves payload partitioning; it does not multiply the already-total `bytes`.

### Implementation

1. Add a side-effect-free `ExtractCopyCostFeatures` query next to `SelectCopyInstForLowering`; do not duplicate instruction-selection rules in Z3 analysis.
2. For a Z3-annotated module, run `LayoutInference` before task inference so copy selection sees the same shared/fragment layouts as lowering. Keep the later call idempotent and add a structural-equality test proving early inference does not change the final schedule. Unannotated modules retain current pass order.
3. Resolve `CopyInst` to the feature instruction. For TMA, reuse `AnalyzeTMABulkCopy`: `rest_size` is descriptor count, physical box geometry supplies width, and source/destination dtype determines payload. For cp.async, use per-warp vector width and participating warp/lane count; for synchronous copies, preserve the generic path.
4. Infer coalescing from normalized buffer strides/ranges and inferred physical layouts. Prove that the sum of per-role/per-warp payloads equals total payload. Unknown/swizzle/fragment mappings produce `geometry=unknown,width=0`, never a coalesced claim.
5. Explicit `prefer_instruction` is validated by the existing selector. An unresolved `auto` or missing layout uses generic fallback with unknown provenance; it cannot be labeled measured TMA/cp.async.
6. Add independent microbenchmark families for TMA and cp.async descriptor issue plus payload, with resident sizes, narrow/wide rows, and SASS/PTX instruction audit. Record constants per architecture tier rather than copying SM120 values to SM100 or older SMs.

### Tests

- **Host:** descriptor-count formula; completion sum/path-II max; instruction-specific calibration miss/fallback; TMA `rest_size`; cp.async width/lane proof; coalesced/strided/unknown geometry; transposed and padded layouts; early/late layout equality; unrewritten versus rewritten dual-copy feature equality.
- **Device:** synchronous, cp.async, and TMA copies at narrow/wide widths and multiple descriptor counts. Inspect emitted instruction and exact output. Measured rates are accepted only after the corresponding new calibration records are complete.

---

## P09 — Dedicated accumulator drain and destination payload

**Rows:** LAT-011, LAT-008 (drain descriptor part).

**Depends on:** P08; P04 covers dual-role drain equivalence.

### Ownership

- **Edit:** `src/cuda/op/copy_cost_features.h`, `src/cuda/op/copy_analysis.cc`, `src/cuda/op/copy.cc`, `src/transform/z3_task_analysis.cc`, `tilelang/cuda/latency_model.py`, `benchmark/z3_cost_model/calibration/`.
- **Tests:** `testing/python/cuda/test_cuda_latency_model.py`, `testing/python/transform/test_tilelang_z3_task_analysis.py`, `testing/python/language/test_tilelang_cuda_tmem_drain_z3.py` (new).

### Implementation

1. Classify `tmem->fragment` as `tmem_load`, `fragment->tmem` as `tmem_store`, and subsequent fragment->shared/global stores with `role=accumulator`. Use `binding` to associate the drain with the source accumulator storage.
2. Use `Tcgen05CopyPlan`/TMEM layout metadata to obtain actual instruction count, participating datapaths/warps, packed/unpacked row width, and b32/value-column behavior. Do not infer drain cost from total elements alone.
3. Set `payload_origin=destination` for conversion drains, so FP32 accumulator -> BF16/FP16 output is charged using destination payload where the physical path requires it. Retain source, destination, and selected payload in `terms` for audit.
4. Preserve separate architecture paths: SM100 TCGEN05/TMEM drains, SM120 fragment-accumulator mma.sync stores, and generic register/shared paths. No SM tier inherits another tier's measured drain constants.
5. A dual-role drain must have equal pre/post-rewrite total payload, descriptor geometry, and destination writes; narrow rows are charged by physical row width rather than the full accumulator width.

### Tests

- **Host:** destination payload for FP32->BF16/FP16; pack::16b/unpack::16b row widths; sliced packed copy alignment; descriptor count and per-warp lanes; dual-role narrow row; architecture-specific calibration miss; no dense/generic substitution when TMEM metadata is unknown.
- **Device:** SM100 TMEM accumulator drains at two row widths and two destination dtypes; SM120 fragment accumulator store; two-role split drain. Exact output and emitted `tcgen05.ld`/ordinary-store instruction class as applicable, followed by independent drain calibration.

---

## P10 — Block-scaled/MX GEMM and bound scale-factor costs

**Rows:** LAT-005, LAT-010; contributes LAT-002 scale payload and LAT-008 scale descriptors.

**Depends on:** P06 and P08. P03 supplies read-before-write coverage.

### Ownership

- **Edit:** `src/transform/z3_task_analysis.cc`, `tilelang/cuda/latency_model.py`, `tilelang/cuda/task_cost.py`, `src/cuda/op/copy_cost_features.h`.
- **Edit calibration:** `benchmark/z3_cost_model/calibration/`, `testing/python/cuda/test_calibration_schema.py`.
- **Tests:** `testing/python/cuda/test_cuda_latency_model.py`, `testing/python/transform/test_tilelang_z3_task_analysis.py`, `testing/python/language/test_tilelang_language_gemm_blockscaled.py` (host-only additions; numerical additions belong in a new Z3 test).

### Public feature schema

Extend `GemmFeature` with dense-compatible defaults:

```text
kind: "dense" | "blockscaled" = "dense"
sf_a_dtype: str | None = None
sf_b_dtype: str | None = None
sf_a_granularity_k: int | None = None
sf_b_granularity_k: int | None = None
sf_layout: str | None = None
mx_min_mn: int | None = None       # architecture/model metadata, not user IR input
```

A block-scaled feature requires both scale dtypes and positive K granularities. `MemoryAccess.role/binding` from P08 identifies each scale load as `scale_a` or `scale_b` and binds it to the GEMM/operand identity. Unsupported fields on a dense feature are errors.

### Implementation

1. In `TaskFeatureExtractor::VisitExpr_(CallNode)`, test `GemmBlockScaledNode` **before** the inherited `GemmNode` path. Never route block-scaled operations through generic e2m1 dense fallback.
2. Extract M/N/K and actual region bounds as for dense GEMM, plus SFA/SFB dtypes, regions, `sfKStart`, granularity, and `sf_layout`. Emit one `kind=blockscaled` GEMM feature and separate bound scale-copy features using real shared/register/TMEM scopes.
3. Derive CUDA atom/issue rules from the actual selected block-scaled path: SM120 `m16n8k64.kind::mxf4nvf4.block_scale`, and SM100 TCGEN05 block-scaled metadata. `mx_min_mn` is architecture/calibration metadata applied only where that path requires an independent-issue minimum; do not copy Ascend's `cube_mx_min_mn` formula to CUDA.
4. Use throughput keys such as `mxf4nvf4:f32` and `mxf8f6f4:f32`. Missing/unsupported arch, dtype, accumulator, scale layout, or measured throughput is explicit unknown/failure under `require_measured`, never dense `e2m1` pricing.
5. Scale copies use P08 issue+payload descriptors and actual scale-region bytes. Preserve SFA/SFB read regions in dependency analysis and version/capacity analysis; binding metadata is for audit/cost identity and must not merge physically independent TMEM allocations.
6. Add SM120 block-scaled mma.sync and SM100 TCGEN05 microbenchmarks with small/large M,N and independent scale-resident loads. Include scale-copy calibration and instruction audit before promoting reliability to measured.

### Tests

- **Host:** derived-type dispatch is not dense; M/N/K atom rounding; path-specific minimum M/N where required; scale dtypes/granularity/layout validation; SFA/SFB region payload and stable binding; dynamic K/region upper bounds; missing throughput fallback; dense GEMM results unchanged; JSON round trip with scale fields.
- **Device:** SM120 E2M1/UE4M3 block-scaled GEMM and SM100 supported MX path, with independent SFA/SFB values proving scales cannot be dropped. Test scale loads and destination drains, exact/toleranced output against a software reference, then measured throughput calibration.

---

## P11 — Guarded per-role candidates, phase costs, and runtime dispatch

**Rows:** LAT-014. Also the necessary enabling subset of API-003.

**Depends on:** P01 and P06; P04/P09 provide dual-role drain tests; P12 if candidate conflict hints are used.

### Ownership

- **Edit:** `tilelang/cuda/language/schedule_hint.py`, `tilelang/cuda/language/__init__.py`, `src/transform/common/z3_schedule_hints.h`, `src/transform/z3_task_analysis.cc`, `src/transform/inject_pipeline.cc`, `src/cuda/transform/materialize_ws_schedule.cc`, `tilelang/cuda/pipeline.py`, `tilelang/cuda/latency_model.py`, `tilelang/cuda/task_cost.py`.
- **New:** `src/cuda/transform/materialize_z3_per_role_tasks.cc`.
- **Tests:** `testing/python/transform/test_tilelang_cuda_z3_per_role.py` (new), `testing/python/cuda/test_cuda_latency_model.py`, `testing/python/language/test_tilelang_cuda_z3_per_role.py` (new).

### Public and internal schema

```python
T.PerRoleTask()
```

Emit `AttrStmt(attr_key="tl.cuda_z3_per_role_task", value=1)`. The body contains one logical operation with `T.Task` candidate sites under role/warp guards. Candidates must have the same execution domain and logical read/write role. Guards are based on role/warp indices, not optional data; every participating role executes exactly one candidate.

Native cost metadata retains `prefix`, `middle`, `suffix`, candidate count, wait count, and `has_cross_wait`, rather than flattening phases prematurely into one `TaskCost`.

### Implementation

1. Materialize explicit and inferred dependency-bearing candidates before cost estimation. Each candidate must contain a dependency-bearing operation; reject mixed domains, different logical footprints, nested per-role markers, and data-dependent/optional guards.
2. Prove guards mutually exclusive and exhaustive by role. Where static proof is impossible, require a compiler-generated final `else` device assertion or reject; do not assume exactly-once from an `if` with no `else`.
3. Implement Ascend-equivalent phase summaries: max candidates within a phase, add phases separated by real cross-role waits, and reject waits inside candidate-conditional control flow. Without a boundary, repeated iterations use conservative serial repetition.
4. With a boundary, preserve loop-prefix/middle/suffix and charge `suffix+prefix` between iterations. Protocol II is total latency for the current unversioned explicit protocol state. Empty trip counts retain static marker counts for validation.
5. After Z3 planning, materialize each candidate under its original role guard and place the same logical arrive/wait next to every required candidate access. Use existing mbarrier/named-barrier machinery with explicit budget/phase validation; do not enable generic AutoWarpSpecialization as a substitute.
6. If safe barrier allocation, exactly-once dispatch, or a provable wait minimum is unavailable, reject before lowering. Do not emit an unguarded wait or silently remove a candidate.

### Tests

- **Host:** one/multiple phases; boundary-loop prefix/suffix formula; zero/one/many trips; protocol II; marker/wait count equality; branch with boundary rejection; nonexclusive/nonexhaustive guards; mixed pipes and footprints; waits remain under role guards; no generic WS pass invoked; deterministic candidate order.
- **Device:** two warp candidates with different costs, a cross-role wait, and repeated outer iterations; exact output and exactly-once instrumentation. Add the dual-role accumulator drain from P09 to prove cost/runtime equivalence.

---

## P12 — Region/scope/cross/group conflict hints

**Rows:** API-004.

**Depends on:** P01. Required before P15 for manual-ring hint tests.

### Ownership

- **Edit:** `tilelang/cuda/language/schedule_hint.py`, `tilelang/cuda/language/__init__.py`, `src/cuda/op/builtin.h`, `src/cuda/op/builtin.cc`, `tilelang/cuda/pipeline.py`, `tilelang/transform/__init__.py`.
- **New:** `src/transform/normalize_z3_conflict_hints.cc`.
- **Edit:** `src/transform/z3_task_analysis.cc`, `src/transform/z3_pipeline_planning.cc` (reject leftover markers).
- **Tests:** `testing/python/transform/test_tilelang_z3_conflict_hints.py` (new), `test_tilelang_z3_task_analysis.py`.

### Public and normalized schema

Match the source signatures:

```python
T.assume_no_conflict(a, b=None, *, level=None, cross=None, group=None)
T.assume_conflict(a, b=None, *, level=None, cross=None, group=None)
```

`level`: `None` = root plus every common enclosing loop, `-1` = root only, nonnegative = one common loop from outermost. `cross`: `None/-1` = both, `True/1` = cross-iteration, `False/0` = same-iteration. A group has exactly two one-region halves and forbids `b`.

Emit pure statement marker `tl.cuda_z3_conflict_hint(a,b,level,cross,group,is_conflict)`. Normalize to loop/root entries `[a,b,cross,is_conflict]`, where operands are a whole-storage Var or a Simplify-preserved `BufferRegion`. Remove the statement marker during normalization.

### Implementation

1. Port the scope-pairing algorithm without Ascend dependencies: loop chain excludes thread bindings, grouped hints use the longest common prefix, mismatched level/cross and one/three occurrences are errors.
2. Match whole buffers by storage identity and concrete regions by exact Buffer plus structurally equal ranges. Different views/ranks or partial regions do not accidentally match.
3. Apply hints only at the declared sequence and cross mode. Forced conflict wins contradictory declarations, including for distinct buffers. Read/read pairs still create no dependency.
4. `assume_no_conflict` may suppress only inferred buffer-region conservatism. It cannot delete explicit `data_deps`, RMW/state/fence dependencies, manual contracts, or exact-placement overlap errors.
5. Run before inference; any marker surviving into planning or codegen is an error for Z3 modules. Non-Z3 compilation may discard the pure no-op only after confirming no Z3 consumer exists.

### Tests

- **Host:** root/explicit-root/all-loop/numbered levels; same/cross separation; exact versus partial region; aliases; two-scope groups and LCA; group cardinality/mismatch; forced-conflict precedence; distinct-buffer forced edge; no-conflict cannot remove explicit/state dependencies; marker cleanup and idempotence.
- **Device:** dynamically permuted writes with a correct no-conflict declaration, and a forced conflict between two staging buffers. Exact output plus expected wait/schedule count; a contradictory declaration test remains host-only negative validation.

---

## P13 — Public version requests and selected-count normalization

**Rows:** PASS-018, API-005 (count/opt-out/request schema), Z3-016 (two source counts).

**Depends on:** P01.

### Ownership

- **New:** `tilelang/cuda/language/annotations.py`.
- **Edit:** `tilelang/cuda/language/__init__.py`, `tilelang/cuda/pipeline.py`, `src/transform/z3_task_analysis.cc`, `src/transform/z3_pipeline_planning.cc`, `src/transform/inject_pipeline.cc`.
- **New:** `src/transform/common/z3_buffer_versions.h`, `src/transform/normalize_z3_buffer_versions.cc`.
- **Tests:** `testing/python/transform/test_tilelang_z3_buffer_version_annotations.py` (new), `test_tilelang_transform_pipeline_buffer_versions.py`, `test_tilelang_transform_z3_pipeline_planning.py`.

### Public schema

Match Ascend's mapping values:

```text
T.annotate_buffer_versions({
  buf: 1,                    # opt out of automatic eligibility and owner claims
  buf: N,                    # fixed N>=2, automatic indexing mode
  buf: "auto" | "iteration" | "counter",  # infer count, request mode
  buf: (N, "auto" | "iteration" | "counter"),
  buf: (1, "auto"),          # remain eligible with one selected version
})
```

Validate positive Python integers, tuple arity, known modes, and alias consistency by `Buffer.data`. `(1,"iteration"/"counter")` is also eligible, unlike bare `1`.

Staged root keys:

- `tl.cuda_z3_buffer_version_requests: Map<Var,int64>`.
- `tl.cuda_z3_buffer_version_mode_requests: Map<Var,String>`.
- `tl.cuda_z3_buffer_versions_selected: Map<Var,int64>` after scheduling.

The existing loop-scoped `tl_pipeline_buffer_versions` remains compiler-internal physical materialization metadata. It is not the public request schema.

### Implementation

1. Normalize aliases before task inference and reject unknown/out-of-scope requests. Preserve a read-only audit of requests while replacing operational request keys with selected counts.
2. Bare `1` removes the storage from automatic versioning and generates fixed-distance-one dependencies; no physical expansion or owner claim is emitted.
3. Fixed `N` is excluded from solver version variables, uses fixed distance `N`, and deducts `N*physical_slot_size` in P16. If post-solve recurrence requires more than `N`, reject; never raise a fixed request silently.
4. Mode-only/automatic entries use solver variables and the current two-source behavior: model-selected counts or schedule-recalculated counts. Continue recomputing after clone augmentation and reject strict under-versioning.
5. Project selected counts to every owner loop and then to `tl_pipeline_buffer_versions`. Injection consumes/cleans only internal metadata; a second injection/planning pass cannot reinterpret a selected map as a new user request.

### Tests

- **Host:** every mapping form; alias count/mode conflicts; bare-one versus `(1,"auto")`; unknown buffer; fixed count retained exactly; fixed under-version rejected; model versus recalculated sources; clone-augmented recompute; request->selected->internal cleanup; second-pass no-op.
- **Device:** fixed double buffer, inferred count, and bare-one opt-out on recurrent loads; exact output and selected/physical counts.

---

## P14 — Iteration/counter modes and selected-version group maximum

**Rows:** API-005 (modes/selected semantics), PASS-018, Z3-016 (group max/recompute).

**Depends on:** P13; final capacity recheck is P16.

### Ownership

- **Edit:** `src/transform/inject_pipeline.cc`, `src/transform/z3_pipeline_planning.cc`, `src/transform/common/z3_buffer_versions.h`, `tilelang/cuda/pipeline.py`.
- **New:** `src/transform/merge_z3_selected_versions.cc`.
- **Edit:** `src/cuda/transform/materialize_ws_schedule.cc` only to share existing counter/phase utilities without widening owner-ring eligibility.
- **Tests:** `testing/python/transform/test_tilelang_transform_pipeline_buffer_versions.py`, `test_tilelang_transform_z3_pipeline_planning.py`, `testing/python/language/test_tilelang_cuda_z3_version_modes.py` (new).

### Internal schema

- `tl_pipeline_buffer_version_modes: Map<Var,String>` with `auto/iteration/counter` resolved to one concrete mode before injection.
- `tl_pipeline_buffer_counter_map: Map<Var,BufferLoad/Var>` and `tl_pipeline_storage_epoch_guard_map` for counter mode.
- `tl_z3_audit_version_request`, `tl_z3_audit_version_selected`, and `tl_z3_audit_version_required` preserve request/model/recomputed distinctions.

### Implementation

1. `iteration` uses the existing affine logical-iteration ring. Require one consistent affine clock for all accesses and unconditional epoch execution.
2. `counter` allocates a monotonic local counter per physical storage group, initialized before the owner loop and advanced exactly once after a completed active epoch. Replace version indices with `counter % selected_count`, including prologue/body/epilogue and guards.
3. `auto` selects iteration only when its proof succeeds; otherwise selects counter. If active-epoch guards or one-update placement cannot be proven, reject rather than defaulting to an unsafe iteration clock.
4. After all owner loops are planned, merge selected counts per storage/alias group with `max`. Propagate the merged count back to every owner's internal physical map and recompute each owner's required versions against the merged schedule. Explicit fixed counts still cannot be raised.
5. Counter groups shared by disjoint owners use one counter and a compatible active guard. Reject incompatible guard/clock forms or a group with no common owner.
6. Re-run P16 capacity checks after merging and after P17 physical padding. Do not copy Ascend's accepted over-capacity limitation.

### Tests

- **Host:** iteration affine clock; non-affine/conditional counter; one initialization/update per active epoch; depth-one elision only when no live counter use; two owners with different local requirements selecting group max; alias consistency; fixed-count overflow error; post-merge required-version and capacity checks; idempotence.
- **Device:** conditional ring with skipped epochs, two owner loops selecting depths 1/2/3 and requiring the group maximum, repeated exact outputs to expose phase/parity errors.

---

## P15 — Manual handwritten ring declarations without reallocation

**Rows:** API-006.

**Depends on:** P01; P12 is required only for the conflict-hint interaction tests.

### Ownership

- **Edit:** `tilelang/cuda/language/annotations.py`, `tilelang/cuda/language/__init__.py`, `src/transform/common/z3_buffer_versions.h`, `src/transform/z3_task_analysis.cc`, `src/transform/inject_pipeline.cc`.
- **New:** `src/transform/normalize_z3_manual_rings.cc`.
- **Tests:** `testing/python/transform/test_tilelang_z3_manual_rings.py` (new), `test_tilelang_transform_pipeline_buffer_versions.py`, `testing/python/language/test_tilelang_cuda_z3_manual_ring.py` (new).

### Public schema

```python
T.annotate_manual_multi_buffer(buf, {flat_or_subrange_buf: N}, ...)
```

A bare buffer infers `N` from a static leading dimension. A mapping gives `N` explicitly and does not assume a leading version dimension. `N` is a positive compile-time integer used as ring size and maximum searched recurrence distance. The annotation never reshapes or reallocates the buffer.

### Implementation

1. Normalize aliases and attach `tl.cuda_z3_manual_rings: Map<Var,int>` to the scheduled scope. Reject conflicting counts, dynamic inferred leading dimensions, and unknown buffers.
2. For each conflicting access pair, analyze actual index/subrange expressions at distances from `(source<target ? 0 : 1)` through `N`. Use the first proven conflicting distance for that pair, as in the source algorithm; same-iteration dependencies remain separate.
3. If a distance cannot be decided, retain conservative dependencies for all unproven distances rather than dropping a possible recurrence. An opaque dynamic clock may produce an infeasible schedule but never a falsely dependency-free one.
4. Mark manual storage ineligible for automatic version selection. Injection and P17 leave its shape, hand-written clock, and subrange layout untouched; no `tl_pipeline_buffer_versions` entry is created for it.
5. Fixed physical addresses, if requested through P18, still require an explicit count from P13; manual-ring metadata alone does not satisfy exact placement's version-size contract.

### Tests

- **Host:** leading-dimension ping-pong; flat/subrange triple ring; forward/backward distances including `N`; no conflict within `N`; undecidable dynamic index conservative matrix; no allocation/index rewrite; aliases and invalid counts; interaction with forced conflict/no-conflict.
- **Device:** handwritten `buf[l%2]` and flattened `flat[(l%3)*N:...]` rings with exact outputs and unchanged allocation shape.

---

## P16 — Scope capacity groups, fixed deductions, unlimit, and final proof

**Rows:** Z3-013, API-007; required by Z3-016 and PASS-018/019.

**Depends on:** P13/P14 for selected/fixed counts and P17 for physical slot sizes. Basic shared-group conversion can land after P13, but row closure requires the full chain.

### Ownership

- **Edit:** `tilelang/cuda/language/annotations.py`, `tilelang/cuda/task_cost.py`, `src/transform/z3_task_analysis.cc`, `src/transform/z3_pipeline_planning.cc`, `src/transform/common/storage_size.h` (only if shared helpers need generalization), `src/cuda/transform/lower_shared_tmem.cc`, `src/transform/merge_shared_memory_allocations.cc`, `src/transform/storage_rewrite.cc`, `tilelang/cuda/pipeline.py`.
- **New:** `src/transform/validate_z3_physical_capacity.cc`.
- **Edit hardware schema/probe:** `tilelang/cuda/hardware.py`, `testing/python/cuda/test_calibration_schema.py`; update generated manifests only when the corresponding probe is actually run.
- **Tests:** `testing/python/transform/test_tilelang_z3_task_analysis.py`, `test_tilelang_transform_z3_pipeline_planning.py`, `testing/python/cuda/test_cuda_cost_model.py`, `testing/python/autotune/test_guided_search.py`, `testing/python/transform/test_tilelang_z3_physical_capacity.py` (new).

### Public and internal schema

```python
T.annotate_unlimit_memory(*scopes)  # scopes: "shared", "register", "tmem"
```

Unlimit means **skip the solver capacity constraint for this scope**, not prove physical placement. Keep `tl_z3_audit_unlimit_memory_scopes` through final lowering for diagnostics.

Align parallel metadata with existing `memory_groups`:

```text
tl_z3_pipeline_memory_group_scopes: Array<String>  # shared/register/tmem
tl_z3_pipeline_memory_group_units: Array<String>   # bytes/tmem_columns
tl_z3_pipeline_memory_fixed_deductions: Array<int64>
tl_z3_pipeline_memory_group_sources: Array<String> # explicit/structural
```

Existing solver entries remain `[remaining_capacity, buffer_id, ...]`; member sizes are expressed in that group's unit. Unbounded scopes omit the solver inequality rather than substituting a huge numeric capacity.

Hardware fields are explicit: shared opt-in bytes, register bytes per block where available, and TMEM columns/datapaths. TMEM's 512x128 structural contract may be marked `structural`; unknown register capacity stays unknown, never zero or unlimited.

### Implementation

1. Infer all version-eligible on-chip writes, including `shared/shared.dyn`, `local.fragment`/versioned local storage, and `shared.tmem`; global storage is excluded. Preserve alias identity and use the largest physical view.
2. Use `GetBufferStorageSizeBytes` for compact sub-byte payloads, then apply P17's actual per-version layout/alignment. TMEM members use required b32 columns from physical `(datapath,value-column)` shape, with four-column scale-factor and 32-column arena rules.
3. Deduct noneligible and fixed-version allocations before solving, including one-version opt-outs. Apply known fixed reservations and unrelated live allocations. Check minimum usage even when a group has no automatic members; a negative remainder is always an error.
4. For unlimit scopes, omit only the solver constraint. Final validation still runs and reports that the user bypassed the earlier bound. Capacity-unknown remains distinct from unlimit throughout cost model/autotuner behavior.
5. After merging versions and materializing layouts, validate whole-kernel usage:
   - shared: actual arena offsets plus physical allocation/barrier sizes, alignment holes, and unrelated live storage against explicit opt-in/reserved limits;
   - TMEM: actual arena placements against 512 columns and 128 datapaths in `LowerSharedTmem`;
   - register: static IR-level footprint as an upper-bound check in `StorageRewrite`; final register assignment remains subject to the CUDA assembler/compiler and must not be described as proven by Z3.
6. Preserve exact placement errors even for unlimit scopes, positive conflict hints, or disjoint lifetimes.

### Tests

- **Host:** packed FP4/int4 sizes; per-version alignment padding; shared/register/TMEM group units; noneligible/fixed/opt-out deductions; over-capacity with zero automatic members; explicit/structural/unknown provenance; unlimit allows solver but final overfit fails; capacity-unknown remains available to autotune and is not unlimit; TMEM scale/arena boundary 512; whole-kernel shared total with barriers and unrelated buffers.
- **Device:** valid packed/sub-byte and TMEM scale+accumulator kernels compile and produce exact output; an over-capacity unlimit case fails during host lowering, not with a GPU crash. No device execution is needed for negative-capacity tests.

---

## P17 — Independent per-version layout, offset, and alignment remapping

**Rows:** PASS-019; required by Z3-013/API-007 final capacity and API-008 placement.

**Depends on:** P13; counter/mode interaction uses P14.

### Ownership

- **Edit:** `src/transform/inject_pipeline.cc`, `src/transform/common/z3_buffer_versions.h`, `tilelang/cuda/pipeline.py`, `tilelang/transform/__init__.py`.
- **New:** `src/transform/rewrite_z3_buffer_version_layout.cc`.
- **Tests:** `testing/python/transform/test_tilelang_transform_Inject_software_pipeline.py`, `testing/python/transform/test_tilelang_z3_version_layouts.py` (new), `test_tilelang_transform_layout_inference.py`, `testing/python/language/test_tilelang_cuda_z3_version_layouts.py` (new).

### Internal schema

For each selected storage, carry an ordered array of physical-version descriptors:

```text
tl_pipeline_version_layouts: Map<Var, Array<{
  layout: Layout,                 # logical view for this physical version
  offset_bytes: IntImm,           # independent base within physical storage
  slot_bits: IntImm,              # compact logical slot
  aligned_slot_bits: IntImm,      # physical occupied slot
  alignment_bytes: IntImm
}>>
```

The array length equals selected count. Automatic contiguous rings normally use a common aligned stride, but the representation and validator must permit independent offsets/layouts. This is compiler metadata, not a second public layout API.

### Implementation

1. Keep injection's same-layout expansion only as the early logical form needed by tile-op/layout inference. Record compact slot bits and selected-version descriptors alongside it.
2. After `LayoutInference` and before `LowerTileOp`/final `FlattenBuffer`, rewrite versioned storage to its physical extent and remap every access using bit arithmetic:
   `physical = storage_base + version*aligned_stride + inner_offset`, or the explicit per-version offset when supplied.
3. Update annotated layouts/fragments for each physical version rather than dropping an annotation that cannot be expanded by the old leading-dimension helper. Validate TMA/swizzle requirements against each version base.
4. Handle dtype-changing aliases in bits. Rewrite `tvm_access_ptr` offsets and reject any pointer span that can cross a compact/aligned version slot. Unknown dynamic span means conservative rejection, not assumed in-bounds.
5. Leave count-only flat manual rings untouched unless a proven leading version-carrier alias exists. Do not apply automatic padding a second time to an already physical layout.
6. Feed actual aligned physical sizes to P16/P18. The source's choice not to feed UB padding back into capacity is explicitly not copied.

### Tests

- **Host:** independent offsets/alignment and two different annotated layouts; leading-stride legacy case unchanged; packed FP4 and dtype-changing aliases; pointer at slot end and cross-slot rejection; counter and iteration indices; `FlattenBuffer` expected indices; idempotence/no double padding; capacity uses aligned size.
- **Device:** padded FP4 ring, dtype-alias ring, and swizzled/TMA shared ring with exact outputs. Inspect final flattened offsets before execution.

---

## P18 — Exact shared byte placement and placement preservation

**Rows:** API-008; contributes final placement proof to Z3-013/API-007/PASS-019.

**Depends on:** P13 for explicit counts, P16 for capacity, P17 for physical sizes/layouts.

### Ownership

- **Edit:** `tilelang/cuda/language/annotations.py`, `tilelang/cuda/language/__init__.py`, `src/transform/common/z3_buffer_versions.h`, `src/transform/z3_pipeline_planning.cc`, `src/transform/rewrite_z3_buffer_version_layout.cc`, `src/transform/merge_shared_memory_allocations.cc`, `tilelang/cuda/pipeline.py`.
- **New:** `src/transform/common/z3_buffer_addresses.h`.
- **Tests:** `testing/python/transform/test_tilelang_z3_buffer_addresses.py` (new), `test_tilelang_transform_pipeline_buffer_versions.py`, `testing/python/language/test_tilelang_cuda_z3_buffer_addresses.py` (new).

### Public schema

```python
T.annotate_buffer_addresses({shared_or_shared_dyn_buffer: nonnegative_int_offset})
```

- Offsets are bytes relative to the allocation's actual shared/static or shared.dyn pool. Aliases of one `Buffer.data` require the same offset. Partial maps are allowed; an empty map is rejected.
- Supported scopes are `shared` and `shared.dyn` only. TMEM uses columns and register storage is compiler-assigned; neither is accepted by this byte API.
- Every placed storage must also have an explicit positive version count through P13, including `1`. Mode-only and manual-ring-only declarations do not determine physical size.
- Frontend enforces nonnegative integer and the pool's base arena alignment. Final lowering additionally enforces natural, TMA/MMA/swizzle, and per-version alignment; frontend acceptance is not a physical-fit promise.

### Implementation

1. Preserve `tl.cuda_z3_buffer_addresses: Map<Var,int64>` through versioning and layout lowering. Reject unknown/dead/duplicate allocations and conflicting alias remaps rather than dropping their placement.
2. In `MergeSharedMemoryAllocations`, reserve every pinned physical interval before placing other allocations. Compute interval end from P17's full physical versions and padding, not compact logical size.
3. Reject any overlap between pinned intervals or between pinned and unplaced storage, even for disjoint lifetimes or positive conflict hints. Preserve user holes; arena size is the maximum physical end, with final alignment.
4. In a pool containing placements, place unplaced allocations into the first aligned free interval without storage reuse. Pools without placements retain existing reuse behavior.
5. Rewrite flattened BufferLoad/Store, `address_of`, and `tvm_access_ptr` from the same final offset table. Verify every placed storage's final address and all final pool sizes after flattening and barrier allocation.
6. Unlimit and scheduling success never waive address, overlap, alignment, or final capacity errors.

### Tests

- **Host:** exact offsets/holes; reversed declaration/address order; partial first-fit without reuse; overlap only after version expansion/padding; unaligned/natural/TMA alignment; capacity boundary; unknown/dead/duplicate/dynamic size; aliases; static versus dynamic pools; pointer rewrite; final offset survives `FlattenBuffer`; no-contract behavior unchanged.
- **Device:** two pinned buffers plus one unplaced buffer with exact numerics, and a versioned padded pinned ring. Negative overlap/capacity cases are host-only.

---

## P19 — Stateful mode/loop/fence dependencies and role demand

**Rows:** API-011; completes LAT-002's actual mode-state work and API-001 state metadata.

**Depends on:** P01 and P03; cost interactions use P06.

### Ownership

- **New:** `src/transform/common/z3_state_dependencies.h`.
- **Edit:** `src/transform/z3_task_analysis.cc`, `src/transform/z3_pipeline_planning.cc`, `src/cuda/transform/inject_fence_proxy.cc` (share classification only, do not run fence injection early), `src/cuda/transform/materialize_ws_schedule.cc` (role-demand projection only), `tilelang/cuda/latency_model.py` only for explicit per-op mode features.
- **Tests:** `testing/python/transform/test_tilelang_z3_state_dependencies.py` (new), `test_tilelang_z3_task_analysis.py`, `testing/python/language/test_tilelang_cuda_z3_pipeline.py`.

### Internal schema

Use a 64-bit state mask independent of the seven-bit execution-resource mask. Initial states include `LoopControl`, `AsyncProxy`, `MMAAccumulator`, `RegisterBudget`, `StoreMode`, and per-resource pipe-fence bits. Attach per-task `state_reads`/`state_writes` arrays of stable string names in cost/dependency audit metadata.

There is deliberately **no new CUDA HF32 setter**. TF32 remains a per-GEMM dtype/lowering feature. Unsupported custom stateful/opaque calls continue to fail conservatively rather than accepting unverified user mode claims.

### Implementation

1. Classify existing CUDA operations:
   - `tl.loop_break` writes `LoopControl`; every sibling reads it, making it a full sequence fence and a loop-exit dependency.
   - proxy/TCGEN05 fences and WGMMA/TCGEN05 operations read/write the relevant proxy/accumulator state, using the same operation classes as existing fence insertion without duplicating fence placement.
   - register-allocation intrinsics write `RegisterBudget` and contribute role demand; explicit store-mode/atomic operations expose their actual state plus existing buffer RMW regions.
   - GEMM dtype `tf32` is embedded in that GEMM's cost feature; no fictitious inter-task HF32 state edge is added.
2. Reproduce source dependency directions for every state: writer->reader RAW forward, reader->writer WAR reverse/next-iteration, and writer->writer WAW in both source directions with correct latencies. Manual per-pipe order does not replace these edges.
3. For non-LoopControl states, emit edges only between tasks whose role/availability sets intersect. LoopControl fences all siblings. Nested loop-exit writes must order following readers and propagate role demand to the containing scheduled task.
4. Compute a role-demand union/fixpoint for nested state writers and their consumers. If role availability cannot be represented by generic pipeline IR, reject rather than discard the demand (same conservative principle as bare owner exclusions).
5. Keep opaque/stateful unknown calls fatal unless a complete classifier and footprints exist. Explicit task cost overrides cannot bypass missing state dependencies.

### Tests

- **Host:** loop-break RAW/WAR/WAW/fence ordering across nested loops and loop exit; proxy and MMA state order; role-intersecting versus disjoint writers; register-demand propagation; per-op TF32 has no phantom global setter; unknown stateful call remains an error; manual stages cannot bypass state edges.
- **Device:** loop-break nested control and a supported WGMMA/TCGEN05 fence sequence with exact output. Do not create a CUDA test for a nonexistent HF32 mode-switch API.

---

## P20 — Repeated complete-compile and generated-IR determinism gate

**Rows:** Z3-017.

**Depends on:** the base test can land immediately; full row closure reruns the expanded matrix after P01-P19.

### Ownership

- **Tests:** `testing/python/transform/test_tilelang_cuda_z3_compile_determinism.py` (new), `testing/python/language/test_tilelang_cuda_z3_pipeline.py`.
- **Implementation only if a failure is exposed:** `src/transform/z3_pipeline_planning.cc`, `src/transform/inject_pipeline.cc`, `tilelang/cuda/pipeline.py`, `tilelang/cuda/transform/z3_scheduler.py`.

### Implementation/test procedure

1. Build one real Z3 module with a same-start tie, a second with manual stages and fixed versions, and a third with the chained/diamond clone shape. Add new feature cases as P01-P19 land.
2. Run complete lowering repeatedly in one process with identical explicit target/hardware/calibration inputs. Compare module structure after Z3 planning, after injection/layout, and final generated CUDA source byte-for-byte. Solver artifact filenames and cache directories are excluded only from logs, never from source comparison.
3. Run fresh subprocesses under two `PYTHONHASHSEED` values and compare normalized output tuples plus generated-source SHA256. Use separate empty cache directories to avoid reusing compiled artifacts.
4. Include repeated manual and automatic solves, and assert identical selected versions, physical order, stage/order arrays, clone plan, starts, II, final arena offsets, and source. Keep existing solver-lock and cross-entry tests; they do not replace this compile-boundary test.
5. If a difference appears, fix the actual unordered source: use declaration/scheduled order plus explicit stable keys, not pointer addresses or hash iteration. Do not weaken comparison to numeric equivalence.

### Tests

- **Host only:** sequential repeated lowering/compile, two fresh processes, two hash seeds, cold caches, no kernel launch. Assert byte-identical source and structural IR equality at all three boundaries.
- **Device:** no special determinism kernel is required. Existing A+B/owner/multi-async numerical tests guard against a determinism-only tie-break change; any ordering bug fix reruns those subsets.

---

## 1. Hardware-N/A review for the requested rows

**No requested row should be reclassified wholesale as `NOT_APPLICABLE_HARDWARE`.** The source evidence supports only the following narrow hardware-specific exclusions, with the generic work kept in the patches above.

| Narrow source clause | Evidence | Why the row remains implementable |
|---|---|---|
| Ascend L0A/L0B/SF physical issue/binding ABI in LAT-010 | `src/ascend/transform/buffer_version.h` binds L0 data/SF storage groups; `estimate_latency.cc` has L1->L0SF-specific issue costs | CUDA `GemmBlockScaledNode` has actual SFA/SFB regions and shared/TMEM paths. P08/P10 implement those generic scale-copy costs. |
| Ascend FixPipe dual-destination accumulator drain in LAT-011 | `estimate_latency.cc::IsDualCopy` and L0C->UB dual endpoint/row-width formulas | CUDA has distinct `kTMemLoad/kTMemStore` and fragment accumulator paths. P09 implements their descriptor, row-width, and drain costs rather than a fake FixPipe. |
| Ascend HF32/MMAD/atomic global setter forms in LAT-002/API-011 | Ascend `Hf32ModeAnalyzer` and `SpecialRegister` classify those named setters | CUDA TF32 is per-op, and CUDA loop/proxy/MMA/register state exists. P19 implements generic state dependencies without inventing an HF32 API. |
| Ascend `cthread` two-AIV software dual-copy launch mechanism in PASS-020/API-013 | `rewrite_dual_copy.cc` requires a `cthread` extent of two | Rank/2:1 splitting, allocation equivalence, and role-indexed ordinary copies are backend-neutral. P04 uses two CUDA warps and CTA-safe per-role storage. |
| Ascend `MODE_MERGING` intrinsic spelling in PASS-021 | `legalize_simd_merging.cc` matches `tl.simd.*` with that mode | RMW read/write correctness is generic. P03 handles CUDA tile-op read-before-write regions and mutable destinations. |
| Ascend VF block names in PASS-025/API-009 | `unroll_loop_skip_vf.cc` skips `SIMD_VF`/`SIMT_VF` | Atomic-region preservation and independent vector-region latency are generic. P01/P02/P07 use CUDA markers. |

In particular, Z3-013/016, LAT-005/008/009/013/014, PASS-018/019/022/025, and API-005/006/007/008 contain generic solver, metadata, layout, or scheduling work and have no hardware-N/A shortcut.

## 2. Landing gates and evidence discipline

1. **One patch at a time:** run its focused host tests, existing Z3 scheduler/planning/injection tests, and CPU formula tests. Do not batch P13-P18 into one unreviewable storage rewrite.
2. **No blanket status promotion:** a host IR implementation closes only its tested subset. Device numerics, instruction-specific measured calibration, and arbitrary-IR support remain separate evidence planes.
3. **Conservative fallback is not silent:** unknown geometry/cost is marked unknown and nonfree; incomplete dependencies, unsupported role/state contracts, fixed under-versioning, and unprovable physical placement are errors.
4. **Final physical proof is mandatory:** selected-version max, unlimit, padding, and solver memory sums are not final fit proofs. P16/P17/P18 must retain final errors through actual arena/layout lowering.
5. **Do not use hardware differences to skip generic work:** only the named Ascend instruction/ABI clauses above are N/A. Every corresponding generic CUDA algorithm remains assigned to a patch.
6. **Suggested first implementation sequence:** P01 -> P02/P03 -> P06 -> P08 -> P04/P05 -> P13/P14 -> P17/P16/P18, with P07/P09/P10/P11/P12/P15/P19 scheduled against those stable interfaces. P20 runs throughout as a regression gate.
