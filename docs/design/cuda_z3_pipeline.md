# CUDA Z3 loop pipeline adapter (G3)

## Scope and activation

G3 connects an **explicit, user-supplied loop scheduling problem** to the
registered CUDA Python solver
`tl.cuda.transform.z3_schedule_loop_python`. It does not infer latencies,
dependencies, resource usage, or buffer sizes from CUDA IR.

The transform is exposed as `tilelang.transform.Z3PipelinePlanning()` and is
controlled by:

```python
PassConfigKey.TL_ENABLE_Z3_PIPELINE_PLANNING
# value: "tl.enable_z3_pipeline_planning"; default: False
```

The CUDA pipeline invokes it immediately before stock `PipelinePlanning` only
when this config is true. The C++ pass checks the same config, so direct uses
are also disabled by default. With the config false, or with no
`tl_z3_pipeline` loop marker, the pass is a no-op.

A module containing G3 loops must not also enable
`tl.enable_auto_warp_specialization`. The CUDA pipeline rejects this
combination before automatic warp specialization can rewrite the loops; the
C++ pass has the same guard for direct use. This prevents two schedulers from
claiming the same loop.

Even when no auto-scheduler is named, `ProducerConsumerWarpSpecialized` can
consume `num_stages`. Consequently, the official CUDA pipeline skips the whole
preceding automatic/legacy WS rewrite block for a G3-owned annotated module.
This ownership decision is module-wide: other loops in that module are not
auto-warp-specialized in the same opt-in run. Unannotated modules keep the
stock WS path. An existing handwritten/materialized WS schedule is an explicit
error rather than being silently skipped.

## Python helper

The helper is intentionally imported from its defining module:

```python
from tilelang.language.annotations import z3_pipeline_annotations

annotations = z3_pipeline_annotations(
    latencies=[20, 40],
    iis=[4, 8],
    resource_flags=[1, 2],
    data_deps=[(0, 1, 0, 20)],
)

for k in T.Pipelined(32, num_stages=3, annotations=annotations):
    copy_task()
    compute_task()
```

`num_stages` remains the existing `T.Pipelined` argument and is passed to the solver as the automatic buffer-version bound. After solving, the emitted CUDA pipeline depth is normalized to `max(input_num_stages, max(stage)+1, max(buffer_versions))`, so a deeper solver schedule cannot outrun downstream barrier/wait depth. The helper validates integer/boolean types, tuple arities, task and buffer indices, per-task lengths, memory-group members, manual-stage shape, resource masks, and duplicate buffer Vars.

## Input annotation schema

All G3 input keys are flat For annotations. Arrays use source task order unless
stated otherwise.

| Key | Type | Required / default | Meaning |
| --- | --- | --- | --- |
| `tl_z3_pipeline` | `bool` | required marker, must be `True` | Select this loop for G3. |
| `num_stages` | `IntImm` | required, positive | Solver buffer-version bound and existing pipeline depth input. Normally written by `T.Pipelined`. |
| `tl_z3_pipeline_latencies` | `Array<IntImm>` | required, length `N` | Non-negative task latencies. |
| `tl_z3_pipeline_iis` | `Array<IntImm>` | required, length `N` | Positive per-task initiation intervals. |
| `tl_z3_pipeline_resource_flags` | `Array<IntImm>` | required, length `N` | Non-negative resource bitmasks. |
| `tl_z3_pipeline_data_deps` | `Array<Array<IntImm>>` | `[]` | `(source, target, distance, latency)`. Negative distance `-b-1` selects solver buffer `b`. |
| `tl_z3_pipeline_resource_deps` | `Array<Array<IntImm>>` | `[]` | `(left, right)` resource exclusion pairs. |
| `tl_z3_pipeline_owner_exclusion_deps` | `Array<Array<IntImm>>` | `[]` | `(left, right, latency_lr, latency_rl)`. **Currently rejected by G3.** |
| `tl_z3_pipeline_pipe_order_deps` | `Array<Array<IntImm>>` | `[]` | `(previous, current)` residue-order constraints. |
| `tl_z3_pipeline_stage_order_deps` | `Array<Array<IntImm>>` | `[]` | `(upstream, downstream)`. **Currently rejected conservatively by G3.** |
| `tl_z3_pipeline_buffer_sizes` | `Array<IntImm>` | `[]` | Solver buffer sizes, length `B`. |
| `tl_z3_pipeline_buffer_vars` | `Array<Var>` | `[]` | Ordered solver buffer-id to `Buffer.data` mapping, length `B`. |
| `tl_z3_pipeline_memory_groups` | `Array<Array<IntImm>>` | `[]` | Solver-native `[capacity, buffer_id, ...]` groups. |
| `tl_z3_pipeline_resource_bits` | `IntImm` | `7` | Number of resource-mask bits. |
| `tl_z3_pipeline_enable_offset` | `bool` | `False` | Solver offset mode. |
| `tl_z3_pipeline_manual_stages` | `Array<IntImm>` | `[]` | Manual stage per task; length `N` when manual mode is active. |
| `tl_z3_pipeline_manual_schedule` | `bool` | `False` | Enable solver manual-stage constraints. |
| `tl_z3_pipeline_seed` | `IntImm` | `42` | Non-negative solver random seed. |
| `tl_z3_pipeline_recalculate_buffer_versions` | `bool` | `True` | Solver buffer-version recalculation mode. |
| `tl_z3_pipeline_verbose` | `bool` | `False` | Solver debug logging/artifacts. |
| `tl_z3_pipeline_debug_dir` | `String` | unset / solver default | Optional solver debug artifact directory. |

The C++ adapter forwards all 19 solver FFI arguments explicitly, including the
optional debug directory.

### Task-to-statement mapping

Before solving, G3 normalizes the body in the same way as stock pipeline
planning:

1. flatten nested top-level `SeqStmt` nodes;
2. remove `AllocBuffer` / `DeclBuffer` declarations from the task stream;
3. remove replayable scalar `Bind` statements from the task stream.

Every remaining statement is one task. `N` must equal this scheduled statement
count, and every per-task array has length `N`. A mismatch is an error before
the solver is called. Dependency indices refer to this filtered source task
stream, not to unfiltered top-level statement positions.

## Solver result validation and projection

The solver must return `(starts, versions, II)`. G3 rejects:

- a result of the wrong tuple/container shape;
- `len(starts) != N` or `len(versions) != B`;
- negative starts;
- non-positive `II` or any non-positive buffer version.

Python solver errors, including UNSAT and `unknown`, propagate to the caller.
There is no silent serial/topological fallback in G3.

For each source task `i`, G3 computes:

```text
k[i]       = floor(starts[i] / II)
stage[i]   = k[i] - min(k)
residue[i] = starts[i] - k[i] * II
```

It sorts source indices by `(residue ascending, stage descending, source index
ascending)`. If `physical_order[position] = source`, the emitted order is the
inverse permutation:

```text
tl_pipeline_order[source] = position
tl_pipeline_stage[source] = stage[source]
```

The raw solver starts and II are retained for audit as:

- `tl_z3_pipeline_audit_starts`: source-order `Array<IntImm>`;
- `tl_z3_pipeline_audit_ii`: `IntImm`.

These are solver time units for auditing, not promises about absolute CUDA
hardware timing. Current CUDA pipeline IR expresses stages and relative order;
it does not encode the absolute start times.

### Cross-stage tie example

For `starts = [0, 50, 151, 152]` and `II = 152`:

| Task | `k` / normalized stage | Residue | Physical position | Emitted order |
| --- | --- | --- | --- | --- |
| 0 | 0 / 0 | 0 | 1 | 1 |
| 1 | 0 / 0 | 50 | 2 | 2 |
| 2 | 0 / 0 | 151 | 3 | 3 |
| 3 | 1 / 1 | 0 | 0 | 0 |

Physical order is `[3, 0, 1, 2]`; the inverse annotation is `[1, 2, 3, 0]`.
Task 3 precedes task 0 at residue zero because the larger stage wins the tie.

## Buffer-version contract

G3 emits:

```text
tl_pipeline_buffer_versions: Map<Var, IntImm>
```

For every solver buffer id `b`, the key is exactly
`buffer_vars[b]` (the associated `Buffer.data` Var) and the value is
`versions[b]`. Duplicate Vars, mapping-length mismatches, and non-positive
versions are errors. G3 also refuses to overwrite an existing
`tl_pipeline_buffer_versions` annotation.

This adapter only produces the shared contract. Consumption/multi-version
rewriting belongs to the downstream pipeline implementation and is not claimed
as completed by this G3 file alone.

## Flat schedules and statement reordering

If every normalized stage is zero and physical order is non-identity, retaining
the source body would lose the schedule: downstream flat pipelines do not
consume a non-identity order. G3 therefore physically reorders the statements
and emits identity `tl_pipeline_order`.

This reorder is deliberately conservative. It requires:

- no scalar `Bind` or standalone buffer declaration in the flat body;
- no opaque/non-buffer side effects;
- no inversion of statements that may read/write the same buffer storage.

Otherwise G3 raises an explicit error instead of silently dropping the order.
An already identity flat schedule needs no such reorder. Non-flat schedules
retain source body order and use the inverse order annotation for downstream
pipeline rewriting.

## Conflicts and current limitations

- Existing handwritten `tl_pipeline_order`, `tl_pipeline_stage`,
  `software_pipeline_order`, or `software_pipeline_stage` is an error on a G3
  loop. Manual Z3 mode never overwrites a handwritten schedule.
- `owner_exclusion_deps` is rejected before solving. Stage/order annotations do
  not preserve owner lifetime, direction, and wraparound semantics.
- All `stage_order_deps` are currently rejected before solving. This is
  conservative: downstream Let cloning and the associated stage-order parity
  are not implemented, and G3 does not distinguish clones that might be safe
  from clones that are not.
- Flat non-identity reorder has the safety restrictions listed above; more
  general Let/declaration motion is not implemented.
- The annotation problem is authoritative. G3 does not prove that a user has
  supplied every real IR dependency or an accurate performance model.
- Automatic latency/dependency extraction, owner-ring lowering, Let clone
  materialization, absolute-time CUDA IR, and full Ascend scheduling parity are
  future work, not completed G3 features.

After a successful solve, G3 removes its input-problem annotations, preserves
`num_stages` and unrelated annotations, and emits the audit, order/stage, and
buffer-version annotations. Stock `PipelinePlanning` can then project the
standard `tl_pipeline_order/stage` keys to `software_pipeline_order/stage`.

## Host test coverage

`testing/python/transform/test_tilelang_transform_z3_pipeline_planning.py`
uses direct TIR construction and a mock FFI callback. It covers full argument
projection, stock planning composition, the cross-stage tie and inverse order,
flat physical reorder and rejection, buffer maps, manual/offset normalization,
replayable-Bind mapping, task-count mismatch, unsupported dependencies,
scheduler conflicts, invalid results, UNSAT/unknown propagation, strict no-op,
and default-off behavior. No GPU or real Z3 solve is needed by these tests.
