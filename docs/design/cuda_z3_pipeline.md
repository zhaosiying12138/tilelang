# CUDA Z3 pipeline: mainline developer guide

This document describes the current opt-in CUDA implementation on the
`cuda-z3-cost-model` branch. It is a guide to supported mainline paths, not a
claim of arbitrary-IR or full Ascend scheduling parity. The historical 84-row
inventory remains a separate snapshot; see
[`cuda_z3_parity_evidence.md`](cuda_z3_parity_evidence.md).

## 1. Activation and default-off behavior

The loop planner is `tilelang.transform.Z3PipelinePlanning()` and calls the
registered Python solver `tl.cuda.transform.z3_schedule_loop_python`. Enable it
with:

```python
kernel = tilelang.compile(
    func,
    target={"kind": "cuda", "arch": "sm_120a"},
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_Z3_PIPELINE_PLANNING: True},
)
```

The underlying key is `tl.enable_z3_pipeline_planning` and defaults to `False`.
With planning and task analysis disabled, CUDA uses the stock unroll, pipeline,
and warp-specialization paths and does not call the G3 normalizers, owner
planner, or Z3 solver. An unannotated module is also a no-op for the generic
planner when planning is enabled.

`tl.enable_z3_task_analysis=True` enables bounded inference of missing task
costs, dependencies, resource masks, and buffer fields. In the main CUDA
pipeline, analysis and planning must be enabled together for an annotated
module. Direct `InferZ3PipelineProblem` use remains available for offline
inspection. Explicit fields are authoritative; unsupported IR is rejected or
requires overrides rather than silently becoming zero-cost, dependency-free
work.

An annotated G3 module cannot also enable automatic warp specialization. G3
skips the preceding automatic/legacy WS rewrite block for the whole module so
that two schedulers cannot claim the same loop. The explicit owner-ring path is
a narrow exception: its typed `T.WSSchedule` is materialized only after owner
validation.

## 2. Public APIs

### 2.1 Flat annotations, `T.Task`, and `T.Stage`

`z3_pipeline_annotations` builds the legacy flat loop annotations. Without task
markers, nested top-level `SeqStmt` nodes are flattened, buffer declarations
and replayable scalar Binds are removed from the task stream, and every other
top-level statement is one task. Dependency indices use this normalized source
order.

Flat `T.Task` makes its complete body one atomic scheduling task. Flat
`T.Stage` wraps complete tasks and selects manual logical stages; unmarked
sibling tasks default to stage zero. If `num_stages` is omitted or zero, the
stage schema derives `max(stage) + 1`. Nonzero stages require
`enable_offset=True`.

```python
from tilelang.language.annotations import z3_pipeline_annotations

annotations = z3_pipeline_annotations(
    latencies=[4, 4, 2, 2],
    iis=[1, 1, 1, 1],
    resource_flags=[1, 1, 16, 2],
    data_deps=[(0, 2, 0, 4), (1, 2, 0, 4), (2, 3, 0, 2)],
    resource_deps=[(0, 1)],
    enable_offset=True,
)

for k in T.Pipelined(4, annotations=annotations):  # depth two is inferred
    with T.Stage(0), T.Task():
        T.copy(A[k * block : (k + 1) * block], a_shared)
    with T.Stage(0), T.Task():
        T.copy(B[k * block : (k + 1) * block], b_shared)
    with T.Stage(1), T.Task():
        for i in T.Parallel(block):
            c_fragment[i] = a_shared[i] + b_shared[i]
    with T.Stage(1), T.Task():
        T.copy(c_fragment, C[k * block : (k + 1) * block])
```

`T.Task(latency=..., ii=...)` supplies per-task cost overrides. A compound Task
is valid only when its body has exactly one execution domain; do not combine a
copy with scalar/vector compute in one Task. Nested Task/Stage markers, a Stage
without a complete Task, nonconstant stages, and mixing Stage markers with the
legacy manual-stage or handwritten order/stage schemas are explicit errors.
The default schema is `schema="flat"`; flat and recursive markers are not
silently mixed.

### 2.2 Recursive schema: one supported vertical slice

The recursive schema is selected explicitly. The accepted mainline example is
a nested fixed-stage loop with a two-version shared ring and an `iteration`
clock:

```python
T.annotate_buffer_versions({shared: (2, "iteration")})
for _o in T.serial(2):
    for i in T.serial(
        4,
        annotations={
            "num_stages": 2,
            "tl_z3_pipeline_enable_offset": True,
            "tl_z3_pipeline_resource_flags": [1, 1],
            "tl_z3_pipeline_buffer_vars": [shared.data],
            "tl_z3_pipeline_buffer_sizes": [32],
            "tl_z3_pipeline_memory_groups": [[64, 0]],
        },
    ):
        with T.Stage(0, schema="recursive"), T.Task(latency=2, ii=1, schema="recursive"):
            shared[i] = i
        with T.Stage(1, schema="recursive"), T.Task(latency=2, ii=1, schema="recursive"):
            B[i] = shared[i] + 1
```

The complete function is in
`testing/python/language/test_tilelang_cuda_z3_recursive_pipeline.py`. This
slice runs a real child solve, lowers prepared schedule units into the stock
pipeline contract, and flattens the nested lexical clock for ring indexing. It
is not a general Task/Control-tree implementation, arbitrary control-flow
scheduler, or completed PerRole scheduling API.

### 2.3 Explicit owner-ring annotations

The dedicated owner path uses `owner_ring=0` and `owner_ring=1` on two ordered
sibling loops, one synthetic group task per owner, and one `(0, 1)` owner
exclusion. A matching typed `T.WSSchedule` supplies two fixed roles, one ring
pipeline, and two ordered `owner_scopes`. See the complete public example in
`testing/python/language/test_tilelang_cuda_z3_owner_ring_integration.py`.

Bare `owner_exclusion_deps` in generic `Z3PipelinePlanning` are still rejected.
They are accepted only by `Z3OwnerRingPlanning` under the restricted contract
in Section 4.4.

### 2.4 Composed task-cost evaluation

Multi-statement task analysis can evaluate a strict `leaf` / `serial` /
`repeat` / `branch` descriptor through `tl.cuda.estimate_composed_task`; the
Python integration points are `tilelang.cuda.task_cost.estimate_composed_task`
and `tilelang.cuda.latency_model.estimate_composed_task`.

```python
from tilelang.cuda.latency_model import estimate_composed_task

cost = estimate_composed_task(
    {
        "kind": "serial",
        "children": [
            {"kind": "leaf", "features": {"latency_override": 20, "ii_override": 4}},
            {
                "kind": "repeat",
                "trips": 2,
                "body": {"kind": "leaf", "features": {"latency_override": 30, "ii_override": 3}},
            },
        ],
    },
    require_measured=True,
)
assert (cost.latency, cost.ii) == (37, 10)
```

Every node is evaluated, including zero-trip bodies and both branch arms, so
resource flags and provenance are retained. Dynamic repeats need finite upper
bounds; `require_measured=True` rejects unknown or fallback results. These are
per-task cycle/II estimates for scheduling, not whole-kernel `predicted_ms`,
device timing, pipeline IR, or proof of an optimal CUDA schedule.

## 3. Current pass order

Only the G3-relevant order is shown. Square brackets denote opt-in or
schema-dependent passes.

**Flat loop annotations and flat Task/Stage:**

```text
NormalizeZ3BufferVersions
-> NormalizeZ3ScheduleUnits
-> UnrollLoopSkipZ3Atomic
-> Simplify
-> [InferZ3PipelineProblem]
-> Z3OwnerRingPlanning
-> Z3PipelinePlanning
-> [MaterializeWSSchedule for a validated owner ring]
-> PipelinePlanning
-> InjectSoftwarePipeline
-> Simplify
```

**Recursive schema:**

```text
UnrollLoopSkipZ3Atomic
-> Simplify
-> NormalizeZ3ControlFlow
-> MaterializeZ3ScheduleUnits
-> [InferZ3PipelineProblem]
-> Z3PipelinePlanning
-> BuildZ3WSSchedule
-> LowerZ3ScheduleUnits
-> [MaterializeWSSchedule if a WS schedule was produced]
-> PipelinePlanning
-> InjectSoftwarePipeline
-> RestoreWhileLoops
-> Simplify
```

The recursive path deliberately does not run the flat buffer/version or
Task/Stage normalizers and does not run `Z3OwnerRingPlanning`. The accepted
fixed-stage vertical slice does not emit `MaterializeWSSchedule`; its audit
checks the exact order above before and after each pass.

## 4. Supported mechanisms and boundaries

### 4.1 Solver projection and buffer versions

For source task `i`, the adapter derives `k[i] = floor(starts[i] / II)`,
normalizes stages by `min(k)`, and computes residues. It sorts by residue,
then descending stage, then source index, and emits the inverse permutation as
`tl_pipeline_order` plus normalized `tl_pipeline_stage`. Raw starts and II are
audit metadata, not absolute hardware-time promises.

`tl_pipeline_buffer_versions` maps each ordered solver `Buffer.data` Var to its
required positive version count. Stock injection consumes this contract and
rejects under-provisioned exact overrides. Output depth covers the requested
stages, solver stages, and buffer versions.

`unknown` is not UNSAT and is never converted into a serial schedule. The
automatic flat solver has an explicit source-order serial fallback only after
exhaustive UNSAT; manual and owner infeasibility do not use that fallback. The
C++ adapter does not catch solver errors and silently substitute stock
planning. A returned `(starts, versions, II)` tuple does not itself carry a
SAT/optimality verdict.

### 4.2 Pure scheduled Let/Bind cloning and alias-aware rings

The flat planner supports consumer-site cloning for pure, scheduled scalar
Binds. It solves the original problem, augments it with clone tasks and virtual
buffer dependencies to a fixed point, and rechecks stage/order, versions, and
capacity. Injection creates fresh Vars, substitutes clone inputs, places
definitions before uses, and rewrites consumers. Chains, diamonds, zero-latency
ties, and next-iteration source overwrites have focused host coverage; canonical
and alias-view SM120 cases have exact numerical coverage.

Versioning and access rewriting use storage identity (`Buffer.data`) for
supported static views, rather than `Buffer` Python/TIR object identity. Thus a
load, store, access region, or clone through a distinct view of the same
storage uses the same versioned allocation and leading ring index. This is not
an arbitrary pointer-alias proof.

Noncloneable, opaque, or effectful Binds must remain same-stage and correctly
ordered. Async-producer clone augmentation, fixed-point overflow, arbitrary
control-flow cloning, and Task/Control-tree cloning remain unsupported or
explicit errors.

### 4.3 Multiple async producer stages

A synchronous consumer may read distinct buffers produced at different async
stages. Task analysis permits automatic schedules and ordered manual stages
(`producer_stage <= consumer_stage`). Injection computes conservative per-stage
wait counts, coalesces them at each consumer using a provable minimum, preserves
physical commit-group mappings, and derives the required ring depths. The old
blanket rejection of every multi-async-stage dependency no longer applies.

The validated SM120 shape uses producer stages 0/1, consumer stage 3, output
stage 4, and checks exact numerics for 1, 4, 5, and 12 iterations. Same-buffer
multi-stage writers, asynchronous consumers of multi-stage inputs, TMA/mixed/
external async forms, conditional producers, split or unstable commit groups,
read-modify-write hazards, and unprovable wait minima remain rejected.

### 4.4 Restricted explicit owner ring

`Z3OwnerRingPlanning` implements exactly two ordered sibling owners and two
fixed warp roles around one CTA-local shared ring. It validates:

- owner indices 0/1, one synthetic task per owner, and one ordered exclusion;
- matching `WSSchedule` owner scopes and a single pipeline depth;
- forward/reverse hand-off and wraparound solver constraints;
- stage-zero, synchronous epochs and matching arrival/body contracts;
- role-uniform loop bounds and unit steps, with no thread- or role-private trip
  counts;
- ring versions, aligned shared capacity, counters/parity, and acquire/commit/
  wait/release materialization.

SM120 evidence covers depths 1, 2, and 3 with repeated exact outputs. Generic
bare-owner annotations, automatic owner inference, additional owners/roles,
nested/guarded/read-first/stage-offset/async owner epochs, non-CTA-local rings,
and general channel allocation are not implemented by this path.

### 4.5 Recursive mainline slice

For the example in Section 2.2, the child solve produces starts `[0, 3]`,
stages `[0, 1]`, and II 2. `LowerZ3ScheduleUnits` restores physical
producer/consumer order and publishes `tl_pipeline_order=[0, 1]`,
`tl_pipeline_stage=[0, 1]`, two versions, and the prepared clock/epoch contract.
Stock planning and injection then produce the two-version ring whose leading
index uses both nested loop variables.

The SM120 test checks the exact vector `[1, 2, 3, 4, -7, -7, -7, -7]` on four
calls. This closes only that nested fixed-stage/two-version vertical slice.

## 5. Flat annotation reference

Arrays use normalized source task order unless stated otherwise.

| Key | Default / requirement | Meaning |
| --- | --- | --- |
| `tl_z3_pipeline` | Required marker, `True` | Select a legacy flat loop for G3. |
| `num_stages` | Positive on legacy loops | Buffer-version bound and downstream depth input; flat Stage markers can derive it. |
| `tl_z3_pipeline_latencies` | Required, length `N` | Non-negative task latencies. |
| `tl_z3_pipeline_iis` | Required, length `N` | Positive per-task initiation intervals. |
| `tl_z3_pipeline_resource_flags` | Required, length `N` | Non-negative resource bitmasks. |
| `tl_z3_pipeline_data_deps` | `[]` | `(source, target, distance, latency)`; negative distance `-b-1` selects solver buffer `b`. |
| `tl_z3_pipeline_resource_deps` | `[]` | Resource-exclusion `(left, right)` pairs. |
| `tl_z3_pipeline_owner_exclusion_deps` | `[]` | `(left, right, latency_lr, latency_rl)`; generic bare use rejects, dedicated owner ring only. |
| `tl_z3_pipeline_pipe_order_deps` | `[]` | `(previous, current)` residue-order constraints. |
| `tl_z3_pipeline_stage_order_deps` | `[]` | `(upstream, downstream)` stage constraints; checked against returned stages and clone-augmented plans. |
| `tl_z3_pipeline_buffer_sizes` | `[]` | Solver buffer sizes, length `B`. |
| `tl_z3_pipeline_buffer_vars` | `[]` | Ordered buffer-id to `Buffer.data` mapping, length `B`. |
| `tl_z3_pipeline_memory_groups` | `[]` | `[capacity, buffer_id, ...]` groups. |
| `tl_z3_pipeline_resource_bits` | `7` | Number of resource-mask bits. |
| `tl_z3_pipeline_enable_offset` | `False` | Permit solver stage offsets. |
| `tl_z3_pipeline_manual_stages` | `[]` | Legacy manual stage per task; do not combine with public Stage markers. |
| `tl_z3_pipeline_manual_schedule` | `False` | Enable legacy manual-stage constraints. |
| `tl_z3_pipeline_seed` | `42` | Non-negative solver seed. |
| `tl_z3_pipeline_recalculate_buffer_versions` | `True` | Recompute required versions and recheck capacity. |
| `tl_z3_pipeline_verbose` | `False` | Emit solver diagnostics/artifacts. |
| `tl_z3_pipeline_debug_dir` | Unset | Optional solver artifact directory. |
| `tl_z3_owner_ring` | Unset | Owner index 0 or 1 for the dedicated restricted owner path. |

## 6. Focused reproduction

Use the built branch and its WSL environment. Device commands require an
SM120 GPU; the deterministic compile test targets SM120 but does not launch the
compiled kernels.

```bash
cd /home/zhaosiying/codebase/tilelang-cuda-z3
source scripts/env.sh
PYTHON="$HOME/miniforge3/envs/tilelang-cu133/bin/python"
```

### 6.1 Host suite: 562 passed, 2 skipped, 4 deselected

```bash
"$PYTHON" -m pytest -q \
  testing/python/transform/test_tilelang_cuda_z3_buffer_epochs.py \
  testing/python/transform/test_tilelang_cuda_z3_buffer_ownership.py \
  testing/python/transform/test_tilelang_cuda_z3_control_flow.py \
  testing/python/transform/test_tilelang_cuda_z3_epoch_lifetimes.py \
  testing/python/transform/test_tilelang_cuda_z3_nested_task_analysis.py \
  testing/python/transform/test_tilelang_cuda_z3_owner_ring.py \
  testing/python/transform/test_tilelang_cuda_z3_physical_plan.py \
  testing/python/transform/test_tilelang_cuda_z3_pipeline_projection.py \
  testing/python/transform/test_tilelang_cuda_z3_recursive_planning.py \
  testing/python/transform/test_tilelang_cuda_z3_role_planning.py \
  testing/python/transform/test_tilelang_cuda_z3_schedule_ir.py \
  testing/python/transform/test_tilelang_cuda_z3_schedule_units.py \
  testing/python/transform/test_tilelang_cuda_z3_unroll.py \
  testing/python/transform/test_tilelang_cuda_z3_ws_recursive.py \
  testing/python/transform/test_tilelang_transform_z3_pipeline_planning.py \
  testing/python/transform/test_tilelang_z3_buffer_version_annotations.py \
  testing/python/transform/test_tilelang_z3_schedule_units.py \
  testing/python/transform/test_tilelang_z3_task_analysis.py \
  testing/python/transform/test_tilelang_transform_pipeline_planning.py \
  testing/python/transform/test_tilelang_transform_Inject_software_pipeline.py \
  testing/python/transform/test_tilelang_transform_pipeline_buffer_versions.py \
  testing/python/transform/test_tilelang_transform_unroll_loop.py \
  testing/python/transform/test_tilelang_transform_materialize_ws_schedule.py \
  testing/python/language/test_tilelang_cuda_z3_owner_ring_integration.py \
  testing/python/language/test_tilelang_cuda_z3_recursive_pipeline.py \
  --deselect 'testing/python/language/test_tilelang_cuda_z3_owner_ring_integration.py::test_sm120_owner_ring_forward_wraparound_and_reuse[1]' \
  --deselect 'testing/python/language/test_tilelang_cuda_z3_owner_ring_integration.py::test_sm120_owner_ring_forward_wraparound_and_reuse[2]' \
  --deselect 'testing/python/language/test_tilelang_cuda_z3_owner_ring_integration.py::test_sm120_owner_ring_forward_wraparound_and_reuse[3]' \
  --deselect 'testing/python/language/test_tilelang_cuda_z3_recursive_pipeline.py::test_sm120_recursive_mainline_exact_output'
```

### 6.2 SM120 device and deterministic-compile suites

```bash
# Flat integration: 9 passed.
"$PYTHON" -m pytest -q \
  testing/python/language/test_tilelang_cuda_z3_pipeline.py

# Owner integration: 12 passed, including host checks and depths 1/2/3 on SM120.
"$PYTHON" -m pytest -q \
  testing/python/language/test_tilelang_cuda_z3_owner_ring_integration.py

# Recursive mainline exact-output case: 1 passed.
"$PYTHON" -m pytest -q \
  testing/python/language/test_tilelang_cuda_z3_recursive_pipeline.py::test_sm120_recursive_mainline_exact_output

# Three pytest cases: 12 complete compilations and 12 real solves in total.
"$PYTHON" -m pytest -q \
  testing/python/language/test_tilelang_cuda_z3_determinism.py
```

For determinism, each of three cases compiles four times: two same-process
repeats and two fresh processes with different `PYTHONHASHSEED` values. The
stock case performs zero solves; multi-stage performs one per compilation;
Bind cloning performs two per compilation. Thus the totals are 12 complete
compilations and `0 + 4*1 + 4*2 = 12` real solves. Schedules, solver results,
metadata, device TIR, host source, and generated CUDA are compared after
normalizing only run-owned paths, timestamps, and schedule directory names.

The accepted logs and their exact evidence boundaries are indexed in
[`cuda_z3_parity_evidence.md`](cuda_z3_parity_evidence.md).
