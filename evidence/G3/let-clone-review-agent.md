# Independent Let/Bind clone correctness review

Static, read-only review of the current uncommitted implementation. I did not edit the implementation, build, or run GPU tests. Findings are ordered by correctness impact.

## 1. High: consumer-prefix materialization can reverse the solver's clone-to-overwrite WAR order

**Evidence**

- Clone reads participate in same-iteration and loop-carried WAR dependencies in `src/transform/z3_pipeline_planning.cc:747-770`.
- The solver's physical order is computed with clones as independent tasks in `src/transform/z3_pipeline_planning.cc:506-560` and `src/transform/z3_pipeline_planning.cc:1201-1203`. Original tasks are then extracted by filtering clones out in `src/transform/z3_pipeline_planning.cc:1204-1215`.
- The clone and consumer positions are saved separately in `src/transform/z3_pipeline_planning.cc:1306-1308`, but injection does not read `consumer_schedule_position` and does not give the clone its own scheduled block. Instead, `MaterializeScheduledBindClones` groups entries by consumer in `src/transform/inject_pipeline.cc:3786-3797` and prefixes all of them to that consumer in `src/transform/inject_pipeline.cc:3822-3849`.

Consequently, an augmented order `clone X -> original O -> consumer C` becomes original order `O -> C`, and materialization emits `O -> (X, C)`. Any constraint that required X to precede O has been reversed.

A minimal relative timing accepted by the augmented checks is: X, O, and C in one raw stage, II = 16, `start(X)=16`, `start(O)=17`, `start(C)=18`, and clone latency 1. If X reads A and O later overwrites A, the virtual same-iteration WAR edge X -> O holds (`17-16 >= 1`), as does the scalar edge X -> C (`18-16 >= 1`). Topological order is X, O, C by residue. The emitted order is nevertheless O, X, C, so X observes O's overwrite instead of its source snapshot. Multi-versioning does not inherently fix this case: O and X have the same stage and logical iteration, hence select the same ring slot.

This is not excluded by `ValidatePipelineBody`: the clone has been folded into C, and the consumer block's access annotations are not rebuilt (finding 5). Even with rebuilt annotations, an O -> C RAW edge would appear valid in the new O-before-C order; it would not recover the lost X-before-O requirement.

**Required direction:** preserve clones as independently ordered schedule units, or constrain/solve the model under the actual consumer-prefix emission rule and prove that no conflicting original task can occur between a clone and its consumer. Add a test with an independent post-consumer overwrite task, not only an overwrite inside C itself.

## 2. High: replayability treats two views of overwritten storage as unrelated

**Evidence**

- `BufferSet` is keyed by `Buffer` object identity in `src/transform/common/bind_utils.h:32-35`.
- Replay safety checks `write_buffers.count(read->buffer)` in `src/transform/common/bind_utils.h:163-180`.
- The write sets are populated with the exact `BufferRegion.buffer` objects in `src/transform/z3_pipeline_planning.cc:345-368` and similarly in task analysis and stock planning.
- Elsewhere, the dependency implementation explicitly recognizes that different `Buffer` objects can share `Buffer.data`: `src/transform/common/pipeline_utils.h:121-128`.

For two distinct views V and W with the same `Buffer.data`, a pipeline write through W does not make a Bind reading V non-replayable. For example:

1. write W;
2. `x = load(V) + 1`;
3. overwrite W;
4. consume x.

The Bind can be removed from the scheduled task stream and replayed at step 4, where it reads the overwrite. This defeats the dependency-closed replay analysis and can silently change the value. It affects all users of `AnalyzeScalarBinds`, not just the Z3 adapter.

**Required direction:** key mutation/stability membership by storage identity (`Buffer.data`), with conservative handling for aliases. Add a replay test using two Buffer objects over one allocation.

## 3. High: cross-iteration RAW/WAR is dropped for aliased Buffer views, so version recomputation cannot repair the schedule

**Evidence**

- `PipelineDistinctIterationsDisjoint` immediately returns true when `left->buffer` and `right->buffer` are not the same object in `src/transform/common/bind_utils.h:195-201`.
- `PipelineLoopRegionsConflict` uses that result for every cross-iteration query in `src/transform/common/bind_utils.h:294-302`.
- Virtual clone RAW/WAR dependencies rely on this query in `src/transform/z3_pipeline_planning.cc:747-770`.
- Task inference contains the same object-identity early return in its duplicate helper at `src/transform/z3_task_analysis.cc:788-795`.
- This differs from same-iteration conflict handling, which compares `Buffer.data` and conservatively treats different views of one allocation as conflicting (`src/transform/common/pipeline_utils.h:121-128`).

Thus, if a clone reads V and a different iteration writes W, where V and W are distinct Buffer objects over the same storage, the cross-iteration conflict is reported as disjoint and the RAW/WAR edge disappears. `ValidateSchedule` can only recalculate versions from dependencies present in `problem.data_deps` (`src/transform/z3_pipeline_planning.cc:875-903`), so the missing edge also means no additional version is required. The returned schedule and version count can both validate while the emitted clone reads overwritten storage.

**Required direction:** return disjoint early only when storage identity differs. If storage is shared but Buffer views differ, remain conservative unless physical alias/disjointness is proven. Fix both the shared helper and the duplicate task-analysis helper.

## 4. Medium: flat declaration indices are used against a declaration-stripped materialization array

**Evidence**

- Z3 flattens the complete body and excludes declarations only when forming scheduled tasks in `src/transform/z3_pipeline_planning.cc:345-384`.
- Clone `definition_stmt` and `consumer_stmt` are emitted from the resulting full-flat indices in `src/transform/z3_pipeline_planning.cc:1275-1276` and `src/transform/z3_pipeline_planning.cc:1301-1303`. These indices therefore include preceding `AllocBuffer`/`DeclBuffer` statements.
- Injection strips those declarations before constructing `original_order` in `src/transform/inject_pipeline.cc:4118-4131`.
- Materialization nevertheless uses the full-flat indices directly in `original_order` in `src/transform/inject_pipeline.cc:3750-3762`, `src/transform/inject_pipeline.cc:3790-3804`, and `src/transform/inject_pipeline.cc:3840-3845`.
- Although Z3 emits `tl_pipeline_scalar_clone_original_flat_indices` in `src/transform/z3_pipeline_planning.cc:1313-1320`, injection never reads that annotation.

A loop-local declaration before a cloned definition shifts every subsequent index. Depending on array length and statement types, this causes an out-of-range rejection, a non-Bind cloneability rejection, a lookup of the wrong definition's pipeline annotation, or clone insertion into the wrong consumer. Valid declaration-containing clone plans therefore are not replayable by the materializer.

**Required direction:** translate full-flat indices through an explicit flat-to-executable mapping, or use definition/consumer task indices with the scheduled-order mapping. Add a clone integration test with a flat `AllocBuffer` and a `DeclBuffer` before the definition.

## 5. Medium: materialized clone reads are absent from the consumer SBlock access contract

**Evidence**

- Z3 records `virtual_reads` in `src/transform/z3_pipeline_planning.cc:1309-1311`, but there is no consumer of `virtual_reads` in `inject_pipeline.cc`.
- Materialization changes only `consumer->body` in `src/transform/inject_pipeline.cc:3827-3842`; it does not rebuild the SBlock or append the clone reads to `consumer->reads`.
- Subsequent dependency construction reads `block->reads` in `src/transform/inject_pipeline.cc:547-575`, and automatic buffer liveness reads the same stale field in `src/transform/inject_pipeline.cc:1645-1673`.
- `PipelineBodyRewriter` only rewrites already-present regions in `src/transform/inject_pipeline.cc:1106-1117`; it does not infer regions for the new Bind prefix.

The emitted IR and the SBlock metadata therefore describe different dataflow. Clone-only reads are invisible to dependency validation, automatic use-stage/version analysis, and other consumers of block reads. Explicit Z3 version overrides can mask the allocation consequence for listed buffers, but they do not make the metadata correct and do not cover unlisted buffers or independently supplied clone plans.

**Required direction:** rebuild the consumer block from its materialized body (or merge the validated virtual reads) before any dependency, async, or version analysis, and assert that every clone BufferLoad is represented in the resulting reads.

## Host-test false-green patterns

- `testing/python/transform/test_tilelang_transform_z3_pipeline_planning.py:464-500` checks that `virtual_reads` exists in metadata, not that materialization consumes it or that the emitted SBlock/access order is correct. Finding 5 is fully compatible with this test passing.
- The chained and diamond tests at `testing/python/transform/test_tilelang_transform_z3_pipeline_planning.py:503-590` stop after Z3 planning. They do not run stock projection plus `InjectSoftwarePipeline`, so consumer substitution, clone sharing in emitted IR, and materialization ordering are untested for those graphs.
- The only direct clone injection test in that file, `testing/python/transform/test_tilelang_transform_z3_pipeline_planning.py:417-461`, uses one scheduled Bind and checks Bind count/freshness. It has no independent later overwrite, flat declaration, alias view, chain, or diamond and does not check a computed result.
- The CUDA numerical snapshot test overwrites the source inside the same consumer statement (`testing/python/language/test_tilelang_cuda_z3_pipeline.py:199-205`). Consumer-prefix placement trivially handles that shape; it does not exercise finding 1's independent task between clone and consumer. It is also CUDA-only and cannot close host-test coverage.
- No examined host replay/clone test constructs distinct Buffer views sharing one `Buffer.data`, so the identity-based replay and cross-iteration cases in findings 2 and 3 are uncovered.
