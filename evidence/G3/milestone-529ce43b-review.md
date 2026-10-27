# Adversarial review of milestone `529ce43b`

Scope: static read-only review of the committed owner-ring, generic Let clone/alias, and multi-async implementation. The reviewer did not edit files, build, run tests, or use GPU. Follow-up implementation agents received every finding below.

## Verdict

Changes required. Independent clone scheduling, declaration mapping, access metadata, sparse orders, replay alias classification, and static owner counters were substantively improved, but four high-severity residual holes remained outside the existing tests.

## Finding 1: aliased clone reads were not rewritten to selected ring versions

- Solver replay and cross-iteration analysis had moved to `Buffer.data` storage identity in `src/transform/common/bind_utils.h` and `src/transform/z3_task_analysis.cc`.
- Injection still keyed `buffer_remap_`, direct load/store/access-region lookups, and `BufferUsageCollector::AddBuffer` by `Buffer` object identity in `src/transform/inject_pipeline.cc`.
- A producer could therefore write `written_versioned[phase][k]` while a clone reading a distinct view of the same storage continued to read unversioned slot zero.
- Existing alias tests stopped after planning/version metadata and did not prove emitted IR.

Required regression: extend alias planning through stock projection and injection; require every load with matching `buffer.data` to use the versioned allocation and leading phase index. Add distinct values per iteration for numerical stale-slot detection.

## Finding 2: async-only consumers did not activate multi-async protection

- `AnalyzeMultiAsyncStageDependencies` skipped async blocks during consumer discovery.
- `has_multi_async_stage_dependency_` could therefore be enabled only by a synchronous consumer.
- The async-consumer rejection was gated on that flag, and `PopulateWaitCounts` also skipped async statements.
- An asynchronous consumer of multiple asynchronous producer stages could receive neither a wait nor an explicit rejection when no synchronous reader existed.

Required regression: remove the synchronous second consumer (or make it non-reading), retain one async consumer reading both async-produced buffers, and require a deterministic rejection rather than missing synchronization.

## Finding 3: aliased async reads bypassed discovery and waits

- `async_writer_stages`, `AsyncStateGlobal::writes`, and consumer producer-selection maps were keyed by `Buffer` object identity.
- A read through a different `Buffer` view sharing producer `Buffer.data` was treated as unrelated.
- A later synchronous consumer could activate multi-async handling while an earlier alias consumer still received no wait; the later wait could not protect that earlier read.

Required regression: make the first consumer read distinct views of A/B storage while the second consumer reads canonical buffers. Preserve the first consumer's required wait or reject the alias explicitly.

## Finding 4: owner epoch bounds were not proven role-uniform

- Owner structural checks rejected guards/while ancestors but did not reject role-dependent loop bounds.
- Generic uniformity checks did not cover owner loop min/extent/step expressions.
- Each role emitted its own substituted extent while counters advanced once per local close, and deadlock validation used a common synthetic trip count.
- Producer/consumer roles could therefore execute different full/empty epoch counts and deadlock.

Required regression: use launch `threadIdx.x` or another role-private value in an owner extent and require rejection in both Z3 owner planning and WS runtime materialization. Preserve valid static and provably uniform bounds.

## Fixes verified by the review

- Independent clones are separate scheduled `SBlock` units and original tasks retain sparse augmented positions.
- Full-flat declarations are mapped through task/executable lookup rather than used as stripped-array indices.
- Clone and consumer blocks rebuild read metadata; chain and diamond tests reach injection.
- Replay alias classification uses storage identity and has emitted-order tests.
- The earlier mixed async-consumer rejection works when a synchronous consumer activates multi-async handling, but Findings 2 and 3 remained.
- Static role-uniform owner counters preserve initialization, owner-boundary continuity, ring indexing, and parity.
- Sparse order/stage unwrapping uses augmented positions as sort keys rather than dense annotation indices.

## Evidence boundary

The committed device tests cover lowered `cp.async`, boundary trip counts, repeated numerics, clone next-iteration overwrites, and static owner depths 1/2/3. They do not cover the four residual cases above. A passing prior test suite is not evidence that these cases are safe.
