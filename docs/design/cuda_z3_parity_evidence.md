# CUDA Z3 / cost model: current evidence and parity boundary

Evidence snapshot: mainline commit `4287bacd`. This page records the accepted
focused host/device evidence and the limits of those results. It is not a new
84-row audit and does not claim full Ascend parity. For APIs, pass order, and
the exact reproduction commands, see
[`cuda_z3_pipeline.md`](cuda_z3_pipeline.md#6-focused-reproduction).

## 1. Accepted focused evidence

The suites overlap in implementation coverage but are separate runs; do not add
their test counts into a grand total.

| Evidence | Exact result | What it establishes | What it does not establish |
| --- | --- | --- | --- |
| [`evidence/G3/parity-host-v24.txt`](../../evidence/G3/parity-host-v24.txt) | **562 passed, 2 skipped, 4 deselected** in 7.69s | Focused host regression for flat planning/injection, owner rings, buffer epochs, public Task/Stage, recursive primitives, and the recursive compile-only mainline audit | Device numerics, arbitrary IR, or completion of every feature represented by a test file |
| [`evidence/G3/z3-pipeline-sm120-v7.txt`](../../evidence/G3/z3-pipeline-sm120-v7.txt) | **9 passed** in 16.91s | Flat SM120 default-off/on A+B; multi-async stages at 1/4/5/12 iterations; canonical and alias-view Bind-clone numerics; public Task and Stage examples | General async layouts, arbitrary aliases/Binds, other GPUs, or performance conclusions |
| [`evidence/G3/owner-sm120-v7.txt`](../../evidence/G3/owner-sm120-v7.txt) | **12 passed** in 7.36s | Owner validation/materialization plus the restricted two-owner ring at depths 1/2/3, with four exact-output calls per depth | Twelve independent device cases, automatic ownership, arbitrary roles/owners, or generic bare-owner annotations |
| [`evidence/G3/recursive-mainline-compile-v4.txt`](../../evidence/G3/recursive-mainline-compile-v4.txt) | **1 passed** in 0.95s | Recursive compile-only pass order, prepared-loop metadata, stock projection, and nested lexical ring clock | Device output by itself or general recursive scheduling |
| [`evidence/G3/recursive-mainline-sm120-v1.txt`](../../evidence/G3/recursive-mainline-sm120-v1.txt) | **1 passed** in 2.55s; exact output `[1, 2, 3, 4, -7, -7, -7, -7]` on four calls | The nested fixed-stage/two-version recursive vertical slice on SM120 | Arbitrary nesting/control flow, a complete Task/Control tree, or full PerRole support |
| [`evidence/G3/z3-017-complete-compile-determinism-v1.txt`](../../evidence/G3/z3-017-complete-compile-determinism-v1.txt) | **3 passed** in 27.90s, representing **12 complete compilations and 12 real solves** | Deterministic schedules, solver results, metadata, device TIR, host source, and CUDA across same-process repeats and fresh processes with different hash seeds | Kernel numerics or timing; this test does not launch the compiled kernels |

The deterministic count is intentional: three cases each compile four times.
`default_off` performs no solve, `multi_stage` performs one real solve per
compilation, and `bind_clone` performs two real solves per compilation:
`4*0 + 4*1 + 4*2 = 12`. The stock case also emits no Z3 schedule artifact.
These are complete uncached compilations with run-owned cache directories, not
cache hits or solver-only repetitions.

## 2. Current mechanism boundaries

### Owner ring

Owner exclusion is no longer accurate as a blanket “solver only / runtime
missing” statement. `Z3OwnerRingPlanning` and WS materialization implement a
restricted explicit two-owner/two-role/stage-zero CTA-local shared ring,
including forward and reverse hand-off constraints, wraparound, ring versions,
capacity checks, persistent counters/parity, and full/empty synchronization.
The SM120 subset covers depths 1, 2, and 3.

Generic `Z3PipelinePlanning` still rejects bare owner exclusions. Automatic
owner inference, arbitrary owners/roles, guarded/nested/read-first/stage-offset/
async epochs, and general channel allocation remain open.

### Pure flat scheduled Let/Bind clones

Cross-stage cloning is implemented for pure flat scheduled scalar Binds. The
planner performs clone-augmented solving to a fixed point and rechecks
stage/order, versions, and capacity. Injection creates fresh Vars, substitutes
inputs, maintains defs-before-uses, and rewrites consumers. Chain, diamond,
next-iteration-overwrite, canonical-storage, and alias-view cases are covered.

This does not support arbitrary effectful or control-flow Binds. Noncloneable
Binds retain same-stage/order requirements; async-producer clone augmentation
and Task/Control-tree cloning remain outside the supported subset.

### Multiple async producer stages

The old unconditional multi-stage guard is gone for a restricted cp.async
case. Ordered producer/consumer stages, distinct destination buffers,
stage-local conservative waits, per-consumer coalescing by a provable minimum,
stable physical commit groups, and required ring depths are implemented. The
SM120 evidence uses producer stages 0/1, consumer stage 3, and output stage 4.

Same-buffer multi-stage writers, asynchronous consumers of such inputs,
TMA/mixed/external async forms, conditional or unstable producer groups,
read-modify-write hazards, and unprovable minima still reject. Same-stage
multiple producers and this distinct-stage subset are different contracts.

### Flat Task/Stage and alias-aware injection

Flat `T.Task` defines atomic scheduling units; flat `T.Stage` supplies manual
logical stages, defaults unmarked siblings to stage zero, and can derive the
pipeline depth. It is a distinct schema from legacy manual-stage arrays and
from recursive markers.

Supported static buffer views are tracked by `Buffer.data` storage identity
through dependency analysis, clone planning, version maps, access metadata, and
ring rewriting. A distinct view of the same storage therefore does not silently
remain on unversioned slot zero. This is not an arbitrary pointer-alias or exact
physical-placement proof.

### Composed cost evaluation

Task analysis is not limited to independent flat leaf estimates. Strict
`leaf`/`serial`/`repeat`/`branch` descriptors are evaluated through
`tl.cuda.estimate_composed_task`; all nodes contribute structural resources and
provenance, including zero-trip bodies and both branch arms. Dynamic repeats
require finite upper bounds, and measured-only mode rejects unknown/fallback
results. These are task latency/II estimates, separate from whole-kernel
`CudaCostModel.predict` candidate rankings and from measured device time.

### Recursive vertical slice

The recursive mainline consumes native recursive markers after unroll/Simplify,
performs control-flow normalization and schedule-unit materialization, solves
the child problem, builds/lowers the schedule, and then uses stock pipeline
planning/injection before restoring While loops. The accepted example preserves
a prepared affine clock across two nested loops and produces a two-version ring
with exact SM120 output.

It is one vertical slice, not general recursive or arbitrary-IR parity.

## 3. Default and failure semantics

- G3 planning and task analysis are default-off. Stock CUDA behavior does not
  call the Z3 solver unless the corresponding opt-in path is selected.
- Unsupported inference is an explicit error or requires user annotations;
  unknown work is not silently free.
- Z3 `unknown` is not UNSAT. The automatic flat solver's source-order serial
  fallback occurs only after exhaustive UNSAT; manual and owner infeasibility
  do not use it. The adapter does not catch errors and substitute a stock
  schedule.
- Device numerical success, compile-only success, host IR validation,
  deterministic compilation, analytical cost ranking, and measured performance
  are separate evidence classes.

## 4. Historical 84-row snapshot

[`ascend_z3_cost_model_feature_inventory.md`](ascend_z3_cost_model_feature_inventory.md)
and the earlier evidence appendix describe an older audit baseline. Keep that
material as a **historical snapshot**, not as the current mechanism summary.
Its 84 rows were classified as:

| Historical status | Rows |
| --- | ---: |
| `IMPLEMENTED_HOST_TESTED` | 11 |
| `DEVICE_VALIDATED_SUBSET` | 8 |
| `PARTIAL` | 37 |
| `BLOCKED_EXPLICIT` | 2 |
| `NOT_IMPLEMENTED` | 16 |
| `NOT_APPLICABLE_HARDWARE` | 10 |
| **Total** | **84** |

Thus the snapshot retained 55 rows in the incomplete `PARTIAL`,
`BLOCKED_EXPLICIT`, or `NOT_IMPLEMENTED` categories, plus 10 hardware-specific
N/A rows. The current focused results do not reclassify that entire inventory,
turn hardware N/A into completion, or establish full parity. The inventory is
intentionally not rewritten here; the mechanism boundaries in Section 2 are the
current mainline documentation.

Older 49-solver, 148-host, single-A+B, and failed multi-async/owner/clone logs
remain useful history. They are superseded for the focused current claims by
the accepted logs in Section 1, not deleted or added together as if they were
independent coverage.
