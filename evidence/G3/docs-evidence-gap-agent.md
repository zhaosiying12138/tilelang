# G3 documentation and evidence gap audit

Audit snapshot: 2026-10-27 19:32 +08:00, repository HEAD `bb50d8654a3cf65a1caffec57618d9c92b688b6a`, with the uncommitted owner-ring, cross-stage scalar-Let clone, and multi-async producer-stage changes present. The shared working tree changed during this read-only audit; the evidence below reflects the newest files observed at this snapshot.

No GPU/test/build command was run, no screenshot was captured, no desktop/window state was inspected or changed, and no existing documentation, article, source, manifest, or script was edited. The only file created by this audit is this report.

## 1. Bottom line and evidence gates

| Mechanism | Current evidence | What can be corrected now | What still must wait |
|---|---|---|---|
| Restricted two-owner ring | Host IR/numerics infrastructure plus `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G3/owner-sm120-v4.txt:57-66` (`4 passed`; depths 1/2/3). The test checks exact outputs on four calls at `/home/zhaosiying/codebase/tilelang-cuda-z3/testing/python/language/test_tilelang_cuda_z3_owner_ring_integration.py:133-166`. | Remove blanket statements that owner-ring/runtime lowering is absent or that every owner annotation is rejected. Document the explicit two-owner/two-role/stage-zero subset and retain the generic bare-owner rejection. | Formal current-tree `DEVICE_VALIDATED_SUBSET` promotion should use a frozen-tree rerun: `/home/zhaosiying/codebase/tilelang-cuda-z3/src/cuda/transform/materialize_ws_schedule.cc` was modified at 19:27:42, after `owner-sm120-v4.txt` completed at 19:23:32. The logs are also untracked, while the inventory definitions require committed evidence. Earlier `owner-sm120-v1.txt` and `owner-sm120-v2.txt` are failed history, not current passing evidence. |
| Multi-async producer stages | Host wait/ring tests plus `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G3/multi-async-sm120-v1.txt:23-31` (`1 passed`). The test requires producer stages 0/1, consumer stage 3, and exact numerics at `/home/zhaosiying/codebase/tilelang-cuda-z3/testing/python/language/test_tilelang_cuda_z3_pipeline.py:95-166`. | Remove the claim that the old multi-stage guard still rejects every such dependency and that only same-stage producers work. Add the new SM120 numeric subset to device-evidence sections. | Keep the unsupported layouts and general sync/epoch gaps open. For a formal frozen-tree final matrix, rerun after the current edits settle and commit/preserve the log; do not infer performance or arbitrary async support. |
| Cross-stage scalar Let/Bind clone | Real planner plus injector implementation and direct chain/diamond/injection tests. `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G3/parity-host-v9.txt:31-38,48` records the latest broader host run (`270 passed, 3 deselected`). A new SM120 next-iteration-overwrite/numerics test now exists at `/home/zhaosiying/codebase/tilelang-cuda-z3/testing/python/language/test_tilelang_cuda_z3_pipeline.py:169-267`. | Replace “no general Let clone,” “cross-stage scheduled Bind is still rejected,” and “clone chains/renaming/defs-before-uses are absent” with the implemented pure, flat, scheduled-scalar subset. A host-only status correction is justified now. | The first combined device attempt `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G3/z3-pipeline-sm120-v1.txt:34,263,283-285` failed the older clone test in `PipelinePlanning` on overlapping writes (`1 failed, 2 passed`). The test was then renamed/restructured as `test_sm120_bind_clone_next_iteration_overwrite_real_solver_and_numerics`. Any clone device claim, and a `DEVICE_VALIDATED_SUBSET` promotion for Z3-010, must wait for a passing matching SM120 rerun on the frozen source; preserve the failed attempt as history. |

Additional current host evidence:

- `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G3/owner-solver-tests-v4.txt:13-32` records **54 passed**, superseding 49 as the current solver-suite result.
- `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G3/parity-host-v9.txt:48` records **270 passed, 3 deselected**. This is a broader/different suite, not a mechanical replacement for every historical “148 passed” or “130 passed” statement.
- `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G3/clone-debug-v4.txt:16-29`, `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G3/clone-debug-diamond-v2.txt:31-44`, `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G3/async-debug-v2.txt:7-16`, and `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G3/owner-integration-debug-v14.txt:18-26` provide focused passing host oracles.

All three mechanisms remain uncommitted implementation/evidence. The formal inventory at `/home/zhaosiying/codebase/tilelang-cuda-z3/docs/design/ascend_z3_cost_model_feature_inventory.md:8-9` defines its host/device statuses in terms of **committed** logs. Either commit/preserve the selected evidence before changing formal status cells, or explicitly revise the audit provenance to label working-tree evidence; do not silently call untracked logs committed.

## 2. Complete stale-statement inventory

Line numbers below refer to the audited working-tree files. “Now” means the false implementation/missing/rejected statement can be corrected immediately. “Clone device” means only the device-backed portion waits for the missing SM120 clone result. “Frozen rerun” means the prose can acknowledge existing results now, but formal final/current-revision status should follow the rerun described above.

### 2.1 Branch README

- `/home/zhaosiying/codebase/tilelang-cuda-z3/README.cuda-z3.md:14` — currently says CUDA owner-ring/runtime synchronization is not closed, the adapter rejects it, and generic Let clone/multi-async remain explicit boundaries. Split this into three precise statements **now**:
  1. bare owner exclusions in generic `Z3PipelinePlanning` still reject, while the explicit restricted two-owner WS path exists and has passing SM120 subset evidence (subject to the frozen-rerun provenance caveat);
  2. pure flat cross-stage scalar Bind clones are implemented/host-tested, with clone SM120 numerics still pending;
  3. distinct-buffer multi-async producer-stage waits are implemented and have a passing SM120 subset log.
  Keep the final warning against arbitrary Ascend end-to-end parity.
- `/home/zhaosiying/codebase/tilelang-cuda-z3/README.md` contains no owner-ring/Let-clone/multi-async parity claim requiring a mechanism update.

### 2.2 Main G3 design documents

#### `/home/zhaosiying/codebase/tilelang-cuda-z3/docs/design/cuda_z3_pipeline.md`

| Lines | Stale content | Exact required update |
|---|---|---|
| 79 | `owner_exclusion_deps` is “Currently rejected by G3.” | Qualify **now**: rejected by the generic planner unless consumed by the dedicated validated `Z3OwnerRingPlanning` path. Cite `/home/zhaosiying/codebase/tilelang-cuda-z3/src/transform/z3_owner_ring_planning.cc` and the generic guard at `/home/zhaosiying/codebase/tilelang-cuda-z3/src/transform/z3_pipeline_planning.cc:1025-1028`. |
| 81 | Stage-order scheduled Bind cases requiring cross-stage cloning are still rejected. | Replace **now** with consumer-site clone planning/materialization for the pure flat subset and cite `test_stage_order_materializes_scheduled_bind_clones`; retain rejection for noncloneable/effectful cases and async-producer clone augmentation. |
| 204-205 | All owner exclusions reject before solving because stage/order cannot preserve lifetime/wraparound. | Restrict this to the bare generic path **now**; describe the separate two-synthetic-task owner solve, forward/reverse hand-off validation, ring versions, capacity, and WS materialization. |
| 208-210 | “No Let clone is invented” and cross-stage consumers remain errors. | Replace **now** with the fixed-point clone-augmented solve and downstream fresh-Var/substitution/defs-before-uses materialization. Retain noncloneable same-stage constraints. |
| 211-212 | More general Let/declaration motion is not implemented. | Keep the flat non-identity reorder safety restriction, but rewrite the wording so it does not contradict the newly supported clone-plan path. Arbitrary control-flow/declaration motion remains open. |
| 217-219 | Owner-ring lowering and general Let clone materialization are not claimed as completed. | Replace with explicit subset status **now**; keep absolute-time CUDA IR and full Ascend parity open. Do not turn this into arbitrary owner or arbitrary Let support. |
| 228-239 | Coverage lists unsupported owner and scheduled-Bind rejection and ends at 148 host tests. | Update direct test names and scope **now**. Add the owner, clone chain/diamond, ordered async, and multi-stage wait tests; cite `parity-host-v9.txt` separately from the historical 148-test log. Add owner/multi SM120 logs, and add clone device evidence only after it passes. |

#### `/home/zhaosiying/codebase/tilelang-cuda-z3/docs/design/cuda_z3_task_analysis.md`

| Lines | Stale content | Exact required update |
|---|---|---|
| 82 | Explicit owner is rejected, stage-order is only the no-clone subset, cross-stage Bind is rejected, and this pass does not extend owner runtime/general clone. | Rewrite **now** to distinguish generic bare-owner rejection from the dedicated owner pass and to describe inferred scalar dependencies that feed clone-augmented planning. Cite the current inferred-clone test at `/home/zhaosiying/codebase/tilelang-cuda-z3/testing/python/transform/test_tilelang_z3_task_analysis.py:546-591`. |
| 141-149 | Section title and prose impose conservative same-stage only: automatic explicit async errors, manual stages must be equal, no general clone, owner always rejected. | Replace the section with ordered producer/consumer stages **now**. Current behavior is automatic acceptance, manual `producer_stage <= consumer_stage`, and rejection only when the producer follows the consumer; direct oracle is `test_async_dependencies_allow_automatic_and_ordered_manual_stages` at `/home/zhaosiying/codebase/tilelang-cuda-z3/testing/python/transform/test_tilelang_z3_task_analysis.py:494-502`. Preserve the true statement that this pass does not synthesize implicit async flags. |
| 153-165 | Host coverage names the old manual-same-stage/automatic-rejection test and reports only the 148-case run. | Name the replacement test and inferred-clone test, then add `owner-solver-tests-v4.txt`/`parity-host-v9.txt` as separate current evidence **now**. Do not imply the 270-case suite is the same 148-case suite. |

### 2.3 Parity evidence appendix — two byte-identical stale copies

The following line numbers and required edits apply to **both**:

- `/home/zhaosiying/codebase/tilelang-cuda-z3/docs/design/cuda_z3_parity_evidence.md`
- `/home/zhaosiying/codebase/tilelang-cuda-z3/article/appendix/parity-evidence.md`

They had identical SHA256 `8a1d4aabe97c42794ef401bce411c792d58064447f2baa483f890cc008d20f80` at the audit snapshot and must be kept synchronized.

| Lines | Stale content | Exact required update |
|---|---|---|
| 7-9 | Audit baseline is old HEAD `4bcc17c4...`; only committed old logs are in scope. | Set the actual HEAD and explicitly describe the dirty implementation/evidence snapshot. Do not claim the new logs are committed. Record the frozen-rerun requirement and retain old failures as history. |
| 19, 25-27 | Current evidence table stops at 49 solver/148 pipeline tests and the single old A+B device pass; row 26 treats the old multi-async failure as the current guard. | Add separate rows for 54 solver, 270 broader host, owner SM120, multi-async SM120, and later clone SM120. Keep old 49/148/A+B rows as historical if desired; do not add overlapping tests into a grand total. Reframe row 26 as historical failure evidence. |
| 46-60 | Owner section says CUDA runtime is unimplemented, nonempty owner always rejects before solving, and planner/ring/counter/sync/channel are absent. | Rewrite **now** around the restricted explicit two-owner implementation and its exact validation/materialization oracles. Keep automatic owner inference, arbitrary roles/owners, guarded/nested/read-first/stage-offset/async epochs, and generic bare-owner IR support open. Add the passing SM120 subset only with the provenance/frozen-rerun caveat. |
| 62-79 | Stage-order section says no Let clone is generated, cross-stage scheduled Bind still fails, and clone chains/renaming/defs-before-uses/snapshot are open. Line 75 and line 79 also name a removed test. | Rewrite **now** for fixed-point clone augmentation and `InjectSoftwarePipeline::MaterializeScheduledBindClones`. Replace `test_stage_order_does_not_bypass_scheduled_bind_validation` with current positive and negative tests at `/home/zhaosiying/codebase/tilelang-cuda-z3/testing/python/transform/test_tilelang_transform_z3_pipeline_planning.py:417-689`. Keep absolute starts, opaque/effectful Bind, arbitrary control flow, and Task/Control trees out of scope. Add device evidence only after the new SM120 clone test passes. |
| 109-116 | Multi-async section says automatic rejects, manual requires same stage, `PopulateWaitCounts` still emits the blanket error, and only same-stage multi-producer tests exist. The old task-analysis test name no longer exists. | Rewrite **now** using ordered-stage inference and per-stage conservative wait counts coalesced by a provable minimum. Name `test_async_dependencies_allow_automatic_and_ordered_manual_stages`, `test_async_pipeline_coalesces_multi_stage_waits_at_each_consumer`, and `test_async_pipeline_preserves_four_task_z3_shape`. Add `multi-async-sm120-v1.txt`; retain the old failure log as history and enumerate still-rejected layouts. |
| 183 | Six-mechanism table says owner is complete only at solver level. | Update **now** to “solver plus restricted explicit two-owner runtime subset”; do not imply full owner parity. |
| 191-203 | Device section recognizes only the old A+B subset and says it cannot support owner/general clone/multi-async; line 203 uses the old failure and guard to prove the current multi-stage gap. | Expand the device matrix with owner and multi-async logs **now**, and clone only after its log exists. Preserve `device-integration-test.txt` as a historical failure, not as proof of a current blanket guard. Keep automatic arbitrary IR, all layouts, and performance conclusions excluded. |
| 222-229 | Hard-gap list says owner planner/ring/sync are open, general Let clone needs implementation, and multi-stage wait/numerics are missing. | Replace solved blanket gaps with precise remaining subsets **now**. Owner inference/general channels, noncloneable/arbitrary-control Let, and unsupported async layouts remain; clone device validation stays listed until its result exists. |
| 232-242 | Static audit claims old counts, 156 valid test references, 19 markers, 11 log patterns, and a tracked diff limited to inventory/appendix. | Rerun the static checks after edits. The old test names are gone, the working tree contains many implementation files and new logs, and all counts/path/marker claims need regeneration rather than hand-preservation. |

### 2.4 Feature inventory — two byte-identical stale copies

The same edits are required in:

- `/home/zhaosiying/codebase/tilelang-cuda-z3/docs/design/ascend_z3_cost_model_feature_inventory.md`
- `/home/zhaosiying/codebase/tilelang-cuda-z3/article/appendix/parity-inventory.md`

They had identical SHA256 `11083717f2a1b3084703b216dfe4bd71d362bd2200c44c8b3b95611dd3c2b275` at the audit snapshot.

| Lines / IDs | Required update |
|---|---|
| 5, 8-9, 16 | Refresh the old HEAD and committed-only audit provenance. Preserve the distinction between host IR, compile-only, device numerics, and committed evidence. |
| 428-432 | Add aliases for `Z3OwnerRingPlanning`, owner materialization, the new owner host/device tests, multi-async device test, clone device test when run, `owner-solver-tests-v4.txt`, `parity-host-v9.txt`, `owner-sm120-v4.txt`, and `multi-async-sm120-v1.txt`. Do not overwrite historical aliases in a way that makes old claims appear to be new results. |
| 440 / Z3-003 | “Let closure ... 未实现” is too broad. Describe inferred flat scalar use-def/reverse-WAR dependencies and clone support; retain special registers and nested timestamps as open. Status can remain PARTIAL. |
| 442 / Z3-005 | The straight-line solver row can remain IMPLEMENTED_HOST_TESTED, but the blanket “CUDA owner runtime/ring/sync 未实现, adapter 明确拒绝” boundary must be narrowed to the generic/bare path. Do not promote this straight-line row merely because the separate loop owner-ring path has device evidence. |
| 446 / Z3-009 | Replace the obsolete `ZP::PlanLoop owner guard`-only mapping and “runtime 未实现” text with the dedicated owner planner/materializer and current tests. After a frozen passing rerun and evidence preservation, this is the direct candidate for IMPLEMENTED_HOST_TESTED -> DEVICE_VALIDATED_SUBSET. |
| 447 / Z3-010 | Replace the no-clone/PARTIAL rationale and deleted test name with the actual clone planner/injector and positive/negative tests. Host-only promotion to IMPLEMENTED_HOST_TESTED is justified now; promote to DEVICE_VALIDATED_SUBSET only after the new SM120 clone snapshot/numerics test passes. |
| 452 / Z3-015 | Remove general clone from the missing list, but keep PARTIAL for ControlNode overall latency and absolute-time IR. |
| 453 / Z3-016 | “multi-owner group max/容量复查未实现” is too blanket. Acknowledge restricted two-owner identical-group/version/capacity checks and clone-augmented version/capacity rechecks; retain generic multi-owner merge/max and final physical-placement gaps. Status can remain PARTIAL. |
| 481 / PASS-004 | Keep BLOCKED_EXPLICIT for automatic write-first/owner-frontier inference, but delete the false statement that explicit owner lifetime cannot enter any CUDA runtime path. The explicit restricted ring is not automatic owner inference. |
| 483 / PASS-006 | Replace “owner/counter epoch ... 未闭环” with the explicit two-owner persistent-counter/parity subset. Keep general counter-group selection, guards, automatic ownership, and full phase protocol PARTIAL. |
| 485 / PASS-008 | Replace same-stage-only/automatic-rejection/blanket-downstream-guard text and the deleted async test name. Add ordered-stage host tests and multi-async SM120 numerics; keep owner epoch closure and unsupported async layouts as reasons for PARTIAL. |
| 486 / PASS-009 | Expand the existing DEVICE_VALIDATED_SUBSET text beyond A+B to owner acquire/release/ring and multi-async subset evidence. Remove the blanket “多 producer-stage/owner 未完成”; retain PerRole/general owner synchronization limits. No status change is needed. |
| 487 / PASS-010 | Add the restricted full/empty barrier pair, arrival counts, parity, allocation lifetime, and channel behavior. Keep general explicit-ID reservation, safe reuse, budgeting, and arbitrary channel allocation PARTIAL. |
| 488 / PASS-011 | Add owner physical ring indices/depth-1 behavior and multi-async version-depth evidence; remove blanket owner-device absence only after the frozen rerun. Keep broadcast/guard clock/elision/full layout matrix gaps. |
| 489 / PASS-012 | Replace “跨 stage Bind 拒绝, general clone 未实现” with actual fresh-Var clone materialization and current tests. Keep role extraction/CoreExtractor, conditional/control Bind, and noncloneable cases PARTIAL. |
| 512 / API-010 | Qualify “not connected to owner channel”: a restricted explicit owner full/empty channel now exists. The public general ID-reservation/caller-managed contract remains unimplemented; do not promote the API row. |
| 521-540 | Recompute family/total status counts only after choosing the final evidence gates, and rewrite the three hard-gap bullets consistently. Do not leave summary counts at 11/8/37 while changing row cells. |
| 693-699 | The “must add” matrix still lists Let clone chains as though absent. Mark their direct host coverage and retain the still-missing overall-latency/nested-timestamp/device combinations. |

Do **not** relabel IR-003 Task/Control trees, PASS-013 epoch separating cuts, API-006 manual-ring declaration, or arbitrary role/owner inference as solved by these mechanisms; those are different contracts.

### 2.5 Other article source files

| File and lines | Required update |
|---|---|
| `/home/zhaosiying/codebase/tilelang-cuda-z3/article/baselines/stock-validation.md:16` | The historical 49-solver baseline is valid only as an old baseline. If presented as current solver evidence, add/update to 54 and link `owner-solver-tests-v4.txt`; do not erase the old log’s provenance. |
| `/home/zhaosiying/codebase/tilelang-cuda-z3/article/mechanisms/01-z3-scheduler.md:141` | Same 49 -> current 54 supplementation. The table at 143-155 can retain solver-scope owner tests and should not claim runtime coverage by itself. |
| `/home/zhaosiying/codebase/tilelang-cuda-z3/article/mechanisms/05-pipeline-adapter.md:3` | Replace the disclaimer that owner-ring/general Let clone are unsupported with scoped implementation statements. |
| `/home/zhaosiying/codebase/tilelang-cuda-z3/article/mechanisms/05-pipeline-adapter.md:21` | Replace the current-coverage story of 148 tests and cross-stage rejection with the new direct host oracles and separate 270-case broader run. |
| `/home/zhaosiying/codebase/tilelang-cuda-z3/article/mechanisms/05-pipeline-adapter.md:61-83` | Keep the original A+B subsection as historical evidence, but add owner and multi-async SM120 subsections/logs and later clone evidence. The 130-case log is a narrower historical suite, not the latest total. |
| `/home/zhaosiying/codebase/tilelang-cuda-z3/article/mechanisms/05-pipeline-adapter.md:85-93` | Preserve the first multi-async failure and quoted old diagnostic as history. Do not delete it. Update line 93’s implication that only a same-stage schedule subsequently completed off/on validation. |
| `/home/zhaosiying/codebase/tilelang-cuda-z3/article/mechanisms/05-pipeline-adapter.md:95-101` | Replace all three current-boundary bullets: owner is no longer always rejected; pure scheduled Bind clones exist; multi-async has ordered-stage lowering and SM120 numerics. Keep arbitrary IR and explicit unsupported subsets. |
| `/home/zhaosiying/codebase/tilelang-cuda-z3/article/results/implementation-status.md:7-15` | Regenerate classification counts from edited inventory rows; do not independently hardcode a conflicting total. |
| `/home/zhaosiying/codebase/tilelang-cuda-z3/article/results/implementation-status.md:25,30,32-35` | Add current 54-solver and 270-host evidence as separate suites and add owner/multi-async device rows. Keep the old A+B and historical multi-async failure clearly labeled. |
| `/home/zhaosiying/codebase/tilelang-cuda-z3/article/results/implementation-status.md:46` | “stock pipeline ... limited combination” remains cautious but should be updated to name the new restricted owner-ring and multi-async combinations rather than implying only the old stock subset. |
| `/home/zhaosiying/codebase/tilelang-cuda-z3/article/results/implementation-status.md:52-59` | Rewrite hard gaps 1-3 using the same restricted owner/clone/multi-async language as the appendices. Keep arbitrary IR, audit metadata, public API, and hardware-specific gaps. |
| `/home/zhaosiying/codebase/tilelang-cuda-z3/article/results/gdn-acceptance.md:10` | Do not continue calling owner-ring/general Let clone unqualified compiler gaps. Say the outer-search result is independent of the remaining general owner inference/arbitrary IR/full-parity limitations; the GDN numbers themselves do not change. |
| `/home/zhaosiying/codebase/tilelang-cuda-z3/article/results/summary.md:51` | Replace the statement that owner runtime/ring/sync, generic Let clone, and multi-async are not closed and always reject with exact subset status and the clone-device gate. |
| `/home/zhaosiying/codebase/tilelang-cuda-z3/article/results/summary.md:53` | Update the 11/8/37 classification counts from the final inventory, not independently. |
| `/home/zhaosiying/codebase/tilelang-cuda-z3/article/results/terminal-evidence.md:15` | The listed final G7 regression counts predate these changes. Rerun the actual final suites before changing the numbers; do not infer new totals by adding 54/270 to old suites. |
| `/home/zhaosiying/codebase/tilelang-cuda-z3/article/results/terminal-evidence.md:25-36` | After an accepted stock/guided capture exists, add its image and sidecar links while preserving the two failed-capture records. The present statement that no acceptable new PNG exists is accurate at this snapshot and must not be changed prematurely. Add the final-confirmation terminal evidence only after its analysis and accepted capture exist. |
| `/home/zhaosiying/codebase/tilelang-cuda-z3/article/appendix/parity-evidence.md` and `/home/zhaosiying/codebase/tilelang-cuda-z3/article/appendix/parity-inventory.md` | Apply every duplicated edit enumerated in sections 2.3 and 2.4. |

### 2.6 Adjacent source documentation drift

- `/home/zhaosiying/codebase/tilelang-cuda-z3/tilelang/language/annotations.py:287-289` still says scheduled Bind cases requiring cloning remain explicit errors. This is false for the new pure flat clone path and should be updated with the same scope/negative constraints. The owner sentences at 281-287 already distinguish bare owner rejection from the restricted ring and should be retained.

## 3. Exact status/count recipe

To avoid indiscriminate promotion, the minimal defensible row transitions are:

1. **Z3-010 now:** PARTIAL -> IMPLEMENTED_HOST_TESTED for pure flat cross-stage scalar Bind clone planning/materialization. If no other row changes yet, totals become **12 host / 8 device / 36 partial / 2 blocked / 16 not implemented / 10 hardware N/A**.
2. **Z3-009 after frozen owner rerun and evidence preservation:** IMPLEMENTED_HOST_TESTED -> DEVICE_VALIDATED_SUBSET. Together with the Z3-010 host promotion, totals become **11 host / 9 device / 36 partial / 2 blocked / 16 not implemented / 10 hardware N/A**.
3. **Z3-010 after a passing frozen clone SM120 log:** IMPLEMENTED_HOST_TESTED -> DEVICE_VALIDATED_SUBSET. Final totals under this minimal recipe become **10 host / 10 device / 36 partial / 2 blocked / 16 not implemented / 10 hardware N/A**. The Z3 family row becomes **9 host / 4 device / 6 partial**, totaling 19.
4. **Multi-async:** expand PASS-008/PASS-009 evidence text, but do not automatically change counts. PASS-008 still has owner epoch/general sync gaps; PASS-009 is already DEVICE_VALIDATED_SUBSET.
5. Keep Z3-005, Z3-015, Z3-016, PASS-004/006/008/010/012, and API-010 at their existing statuses unless their row scope is deliberately rewritten and separately reviewed. Update their false boundary prose only.

These counts are a recommended consistent recipe, not a license to bypass the committed-evidence and frozen-rerun gates. Whatever recipe is chosen must be applied identically to the inventory summary, article implementation-status table, article summary, and parity-evidence static-audit section.

## 4. Screenshot and terminal-evidence gap

### 4.1 What is already accepted/present

- `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G7/screenshots/capture-manifest.json:2-26` approves only the sample capture `tilelang-cuda-z3-sample-v3.png` and explicitly requires individual review of each later image.
- The sample PNG/original/raw log/capture metadata/window binding/crop metadata and the accepted GEMM cubin capture exist. They do not prove stock/guided search or final GDN confirmation visually.
- Real stock/guided execution is already present at `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G6/gdn-stock-guided-screenshot-v1/stock.log` and `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G6/gdn-stock-guided-screenshot-v1/guided.log`; recomputed analysis exists at `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G7/screenshots/stock-guided-analysis-v2/`.
- Formal GDN ABBA confirmation data already exists under `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G6/gdn-heldout-t8192-confirmation-v1/analysis/` and is cited by `/home/zhaosiying/codebase/tilelang-cuda-z3/article/results/gdn-acceptance.md:63-80`. The missing G7 items below are terminal/presentation evidence, **not missing formal GDN numerical confirmation**.

### 4.2 Stock/guided artifacts still missing

Observed failure records:

- `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G7/screenshot-failure-v2/capture-failure.json:1-10` names intended `C:\Users\Admin1\shot\tilelang-cuda-z3-stock-guided-v1.png` and raw log, with `capture_status=failed`, `image_accepted=false`.
- `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G7/screenshot-failure-v3/capture-failure.json:1-10` does the same for `tilelang-cuda-z3-stock-guided-v2.png`.
- `/home/zhaosiying/codebase/tilelang-cuda-z3/article/results/terminal-evidence.md:36` accurately says no acceptable new PNG was produced.

Precise absent artifacts:

- No accepted stock/guided PNG exists in `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G7/screenshots/` for v1, v2, or any later version.
- The Windows shot directory contains raw `.log`, `.png.window-binding.json`, and failure JSON for v1/v2, but **no v1/v2 PNG**, no accepted capture-metadata JSON, no full-screen `-original.png`, and no crop metadata.
- The repo has no accepted stock/guided raw-log/crop/capture-metadata bundle and no stock/guided entry in `capture-manifest.json`.
- `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G7/screenshots/tilelang-cuda-z3-stock-guided-v1.sh:8-10` cannot be rerun unchanged because its ROOT now exists; it also repeats GPU search, which the failure metadata explicitly says not to do merely to repair an image.
- `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G7/screenshots/tilelang-cuda-z3-stock-guided-v2.sh:7-11` is analysis-only but cannot be rerun unchanged because `stock-guided-analysis-v2` now exists and line 9 requires it to be absent.

Required eventual accepted bundle, using a fresh version rather than overwriting failed history:

1. real cropped terminal PNG and uncropped `-original.png`;
2. raw execution `.log` matching visible commands/output and `git rev-parse HEAD`;
3. capture metadata `.json`, `.png.window-binding.json`, and `.crop.json` with matching SHA256/dimensions;
4. fresh analysis-only script/output directory if a visible recomputation is to be captured;
5. individual content/log/window/theme/popup review and a capture-manifest entry or per-capture manifest;
6. article links and regenerated article manifest/package.

No GPU search is needed to fill this gap. Do not rename an old sample or failed sidecar into a stock/guided acceptance artifact.

### 4.3 Final-confirmation artifacts still missing

Only the script `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G7/screenshots/tilelang-cuda-z3-gdn-confirmation-v1.sh` exists. There is no failure record indicating an attempted capture, no Windows shot file matching `*confirmation*`, no repo raw log/PNG/binding/crop bundle, and no output directory `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G7/screenshots/gdn-confirmation-analysis-v1`.

The script at lines 6-12 requires a fresh analysis directory and prints `confirmation-comparisons.csv`. Based on `/home/zhaosiying/codebase/tilelang-cuda-z3/benchmark/z3_cost_model/gdn/analyze_confirmation.py:105-142` and the analogous existing G6 output, the complete missing analysis directory should contain:

- `analysis.json`
- `confirmed-summary.csv`
- `confirmation-comparisons.csv`
- `gdn-confirmation.csv`, `gdn-confirmation.json`, `gdn-confirmation.png`, `gdn-confirmation.svg`
- `gdn-paired-differences.csv`, `gdn-paired-differences.json`, `gdn-paired-differences.png`, `gdn-paired-differences.svg`

Also missing are the raw terminal log and the accepted final-confirmation screenshot bundle (cropped PNG, original PNG, capture metadata, window binding, crop metadata, review/manifest record). The script is analysis-only over existing G6 results; no new GPU confirmation run is required. Because its `test ! -e "$OUT"` currently passes, the first real analysis/capture can use v1, but after a successful run another visible run needs a fresh version/output rather than overwriting evidence.

## 5. Article manifests and build-script implications

### 5.1 Generated outputs must be rebuilt, not hand-edited

The current accepted build manifest `/home/zhaosiying/codebase/tilelang-cuda-z3/article/_build/final-v2/article-manifest.json` matches current article source hashes at this snapshot, but it necessarily becomes stale after the edits above. Its affected source entries are:

- stock baseline and scheduler: lines 12-21;
- pipeline adapter: lines 42-45;
- implementation status: lines 66-69;
- GDN acceptance: lines 78-81;
- terminal evidence: lines 96-99;
- summary: lines 102-105;
- parity inventory/evidence: lines 108-117.

After source and accepted-evidence updates, regenerate the HTML, Zhihu Markdown, formula log, article manifest, article ZIP, PDF/review pages, repository-evidence ZIP, and deliverables in a new reviewed build. `/home/zhaosiying/codebase/tilelang-cuda-z3/article/_build/final-v2/deliverables.json:2-37,58-62` also needs regenerated counts/hashes/visual acceptance; it is not an independent source of truth. Older `_build`, `final`, and `formula-check` trees are historical generated outputs and should not be manually patched to look current.

Newly linked `.txt`, `.log`, `.json`, and `.csv` evidence is automatically collected by the builder. Run the `--check-only` validation described at `/home/zhaosiying/codebase/tilelang-cuda-z3/article/README.md:112-120`, then build and repeat visual/content/hash review. New article pages invalidate the old 103-page visual acceptance record until rerendered/reaccepted.

### 5.2 Plain original-PNG links are not packaged

There is a separate evidence-packaging limitation relevant to the missing screenshot bundles:

- `/home/zhaosiying/codebase/tilelang-cuda-z3/article/build_html.py:733-741` auto-collects ordinary links only for text attachment suffixes; a plain link to an original `.png` is ignored.
- `/home/zhaosiying/codebase/tilelang-cuda-z3/article/build_html.py:327-337` rejects PNG as an `attachments` entry while allowing it only as an image resource.
- Consequently, the existing original sample link at `/home/zhaosiying/codebase/tilelang-cuda-z3/article/results/terminal-evidence.md:15` does not put `tilelang-cuda-z3-sample-v3-original.png` into the final-v2 manifest, even though the cropped image/raw log/window binding are present.

If the accepted stock/guided and final-confirmation **full bundles** are intended to ship inside the evidence ZIP, extend the builder with an explicit non-displayed image-resource/attachment mechanism and add synthetic tests in `/home/zhaosiying/codebase/tilelang-cuda-z3/article/tests/test_build_html.py`, or explicitly document that original PNGs remain repo-side only. Do not claim a complete self-contained capture bundle while only the cropped display PNG and text sidecars are packaged. Update `/home/zhaosiying/codebase/tilelang-cuda-z3/article/README.md:67-79,141-161` if that packaging contract changes.

No mechanism-specific stale prose was found in `/home/zhaosiying/codebase/tilelang-cuda-z3/article/build_html.py`, `/home/zhaosiying/codebase/tilelang-cuda-z3/article/index.json`, `/home/zhaosiying/codebase/tilelang-cuda-z3/scripts/build_tilelang.sh`, or the other native build scripts. They need regeneration and, only for full original-PNG packaging, the explicit builder enhancement above—not owner/clone/async hardcoded text edits.

## 6. Claims that must remain after the updates

- Generic bare `owner_exclusion_deps` still reject at `/home/zhaosiying/codebase/tilelang-cuda-z3/src/transform/z3_pipeline_planning.cc:1025-1028`; only the dedicated explicit owner-ring path bypasses generic planning after validation.
- Owner support is exactly two ordered sibling owners, two fixed roles, stage-zero epochs, CTA-local shared storage, and the validated arrival/body contract. Automatic owner inference, arbitrary owners/roles, guarded/nested/read-first/stage-offset/async epochs, and general channel allocation remain open.
- Clone support is for pure flat scheduled scalar Binds. Noncloneable/opaque/effectful definitions require same-stage placement, async-producer clone augmentation rejects, fixed-point limits remain, and Task/Control-tree or arbitrary-control-flow cloning is not implemented.
- Multi-async support covers distinct buffers from distinct producer stages with stable physical commit-group mappings and provable conservative minima. Same-buffer multi-stage writers, TMA/mixed/external async, conditional producers, split/unstable groups, and unprovable minima still reject at `/home/zhaosiying/codebase/tilelang-cuda-z3/src/transform/inject_pipeline.cc:1572-1635,2754-2949`.
- Preserve `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G3/device-integration-test.txt`, `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G3/owner-sm120-v1.txt`, `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G3/owner-sm120-v2.txt`, `/home/zhaosiying/codebase/tilelang-cuda-z3/evidence/G3/z3-pipeline-sm120-v1.txt`, and both stock/guided capture-failure directories as real history. Passing retries do not justify deleting failures.
- Passing subset numerics do not establish performance, arbitrary-IR correctness, full Ascend parity, or final GDN/classic workload changes.

## 7. Ordered completion checklist

1. Freeze the implementation and run/preserve the missing SM120 cross-stage Bind clone snapshot/numerics result; rerun owner/multi on that same frozen tree for a clean final device matrix.
2. Preserve/commit the selected logs or explicitly change the inventory’s evidence-scope wording before formal status promotion.
3. Apply the source edits above to both duplicated parity files, then recompute all row/family/total counts and static references.
4. Refresh current solver/host/device evidence tables without adding overlapping suite counts; rerun the actual final G7 regression suites before changing terminal-evidence totals.
5. Produce the analysis-only final-confirmation outputs and, independently, accepted stock/guided and final-confirmation capture bundles. Do not rerun GPU work merely to repair screenshots.
6. Decide whether original full-screen PNGs must ship in the ZIP; if so, extend/test the article builder as described.
7. Rebuild article outputs/manifests/packages, rerender review pages, repeat content/log/window/hash review, and regenerate deliverables/visual acceptance.
