# Multi-async producer-stage review

## Verdict

The wait arithmetic is sound for the currently tested, supported shape: unconditional producers, one physical commit group per producer stage, and synchronous consumers. For the device schedule `[0,1,3,4]`, the stage-0 and stage-1 requirements at the stage-3 consumer are 3 and 2 groups respectively, so the minimum is `wait(2)`. In the two-consumer host test, the stage-1 consumer requires `wait(0)` because it consumes the just-committed stage-1 buffer, while the stage-3 consumer requires `wait(2)`; the `[0,2]` expectation is correct.

I found one concrete unsupported-case correctness hole and several material coverage gaps. The scheduled SM120 test is **not sufficient by itself to validate the claimed multi-async synchronization mechanism**.

## Concrete bug: async consumers silently miss required waits

- `AnalyzeMultiAsyncStageDependencies` skips every block annotated async (`src/transform/inject_pipeline.cc:1585-1588`), and `PopulateWaitCounts` again skips every async statement before examining its reads (`src/transform/inject_pipeline.cc:2844-2858`). Consequently, an async block that reads buffers produced by multiple earlier async stages receives no wait and is not rejected.
- Host-IR repro: start from `_make_multi_async_stage_two_consumer_pipeline` (`testing/python/transform/test_tilelang_transform_Inject_software_pipeline.py:272-311`), change `software_pipeline_async_producers` to `[1,1,1,0]` and groups to `[0,0,0,-1]`. Injection succeeds. In the steady loop, `copy_b` and `consume_first` share a commit group, and there is **no wait between them**, even though `consume_first` reads both `a_shared` and `b_shared`; only the later synchronous `consume_second` gets `wait(2)`. With a real asynchronous `copy_b`, `consume_first` can read unfinished data.
- Reject async-on-async buffer dependencies in this mode, or implement the required wait before such blocks. Add a negative host test based on this repro. The scheduled SM120 kernel does not hit this hole because its consumer is synchronous.

## Missing host and negative tests

1. **Prologue/epilogue and boundary bounds.** The implementation has distinct prologue/body/epilogue fragments (`src/transform/inject_pipeline.cc:1341-1351`) and a special no-local-producer tail path (`src/transform/inject_pipeline.cc:2906-2929`), but the main host test asserts only steady-loop waits and allocation shapes (`testing/python/transform/test_tilelang_transform_Inject_software_pipeline.py:320-339`). Add full-function wait/commit placement assertions, including a final drain, and extents 0, 1, `max_stage`, `max_stage+1`, and a longer wraparound case. A host probe at extent 4 (`max_stage+1`) takes the unit-body path and emits epilogue counts `2,1,0`; this boundary behavior is currently unpinned, although the counts are conservative.
2. **Commit-group mappings and wait-count branches.** Current multi-async tests use one group per producer stage. Add cases with multiple groups per stage, buffers in different groups, and consumers ordered before/after producers. These are needed to exercise group validation (`src/transform/inject_pipeline.cc:2759-2815`), the previous-round/current-round selection (`src/transform/inject_pipeline.cc:2921-2929`), and the per-buffer minimum in the wait formula (`src/transform/inject_pipeline.cc:2932-2960`).
3. **Unsupported-case rejection.** There are no focused multi-async negative tests for multiple async stages writing one consumed buffer (`src/transform/inject_pipeline.cc:1600-1605`), TMA or externally managed async/control scopes (`src/transform/inject_pipeline.cc:1622-1629`), conditional producers (`src/transform/inject_pipeline.cc:1630-1634`), or split/unstable/non-contiguous commit groups (`src/transform/inject_pipeline.cc:2776-2815`). Each accepted rejection path should have a small host test.
4. **Buffer-version boundaries.** The implementation correctly computes the automatic requirement before applying an override (`src/transform/inject_pipeline.cc:1691-1711`), but the sole negative test checks only `a_shared=3` versus the required 4 (`testing/python/transform/test_tilelang_transform_Inject_software_pipeline.py:342-346`). Also test `b_shared=2` versus 3 and exact accepted overrides `a_shared=4`, `b_shared=3`; include the output fragment in the four-task shape if overrides are intended to be supported there.

## SM120 device-test assessment

`test_sm120_multi_async_stage_real_solver_and_numerics` usefully proves the real solver schedule `[1,2,6,8]`, II 2, stages `[0,1,3,4]`, and exact numerics for one 12-iteration invocation (`testing/python/language/test_tilelang_cuda_z3_pipeline.py:137-166`). Twelve iterations do exercise ring reuse, and `evidence/G3/multi-async-sm120-v1.txt` records one passing run.

However, the test checks no lowered async property. It does not establish that target lowering emitted pipeline-managed `cp.async`, two commit groups per steady iteration, `wait(2)` before the multi-stage consumer, the expected 4/3/2 buffer rings, or a correct epilogue drain. A synchronous fallback or an over-conservative `wait(0)` could satisfy every current assertion. A single exact-numerics run is also weak evidence against timing-dependent stale reads.

Keep this as an integration smoke test, but add lowered-TIR/CUDA assertions for the actual async groups and waits, plus repeated randomized invocations and at least one boundary trip count. Those additions are necessary before treating the device result as validation of the claimed mechanism rather than only solver scheduling and one successful numerical execution.

## Review validation performed

No builds or GPU tests were run. The two focused host tests pass in the existing `tilelang-cu133` environment. Additional host-only IR probes covered short extents and the async-consumer repro above.
