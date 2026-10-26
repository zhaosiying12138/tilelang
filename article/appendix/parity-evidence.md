# CUDA Z3 / cost model feature parity 证据附录

## 0. 审计范围与判定原则

本附录配套 `docs/design/ascend_z3_cost_model_feature_inventory.md` 第 6 节。逐项 ID、状态、NV 源码位置和测试位置以该表为准；本文只解释证据和边界，不另建一套 ID 或把硬件不支持项改写成完成。

- 当前源码基线：工作树 HEAD `4bcc17c4e8002ded44464a2b71f21e0a73fff9f8`。
- 本次仅做 CPU 阅读与静态脚本检查；没有运行 GPU、profiler、native build、新测试或后台任务。
- 只引用已经提交且与本任务相关的日志。共享树中其他 agent 的 operators 文件、未提交实验输出和正在进行的 GPU GDN 复测不作为本审计依据；本文不改变实验结果，也不宣称 GDN 或经典 workload 最终验收完成。
- `IMPLEMENTED_HOST_TESTED` 只闭环写明的 host 范围；`DEVICE_VALIDATED_SUBSET` 只闭环写明的设备子集。二者都不是 arbitrary-IR/full-Ascend parity。
- `PARTIAL`、`BLOCKED_EXPLICIT`、`NOT_IMPLEMENTED` 均保留开放工作；`NOT_APPLICABLE_HARDWARE` 只表示直接硬件 ABI/物理机制无对应，不表示相邻通用算法已完成。

## 1. 已提交执行证据核对

以下数字直接来自日志，而不是从任务提示继承。不同日志的套件有重叠，**不得相加成一个总测试数**。

| 证据 | 日志中的结果 | 本审计允许的结论 | 不允许外推的结论 |
|---|---|---|---|
| `evidence/G1/solver-tests.txt` | 49 passed in 1.16s | CUDA Python solver 49 个参数化 host cases 通过 | CUDA runtime、owner ring、任意 IR 或设备正确性 |
| `evidence/G2/latency-model-tests.txt` | 28 passed in 0.16s | pure-Python task latency/II 与组合公式的 host cases 通过 | 所有 task 的真机 absolute cycles |
| `evidence/G2/cost-model-tests.txt` | 31 passed in 0.10s | whole-kernel analytical ranking、容量和 model-card host cases 通过 | predicted ms 等于实测 ms、真机最优配置 |
| `evidence/G2/calibration-tests.txt` | 32 passed, 37 subtests passed | calibration schema、独立公式、单位、来源和 artifact audit 通过 | 每个架构/每种 shape 都有测量 |
| `evidence/G2/calibration-final/calibration.txt` | 11 个 benchmark 均为 complete | SM120 独立 microbenchmark 集合完成 | hardware probe 属性本身就是 throughput calibration |
| `evidence/G2/calibration-summary.json` | `status=measured`；11 个 complete benchmarks；各 rate 有 measured source | 可追溯的 SM120 measured rate 子集 | 全部 latency 参数均为 measured，或模型排名已经真机验证 |
| `evidence/G3/main-pipeline-host-tests-final.txt` | 148 passed in 8.22s | task analysis、Z3 adapter、stock pipeline/buffer-version 的联合 host 回归通过 | 148 个不同 feature 全闭环；device 数值正确性 |
| `evidence/G3/device-integration-test.txt` | 1 failed；consumer 依赖多个 async stages | 保留真实历史失败和 downstream guard 证据 | 删除该失败或宣称 multi-async-stage 已支持 |
| `evidence/G3/device-integration-test-retry.txt` | 1 passed in 4.26s | 当前已提交证据中的单一 SM120 A+B off/on 设备子集通过 | 多 shape、多 IR、owner/general pipeline 或 workload 最终验收 |
| `evidence/G4/guided-search-tests.txt` | 20 passed | pruner/guided ranking/unknown host 语义通过 | solver unknown 被当成 UNSAT |
| `evidence/G4/guided-tuner-tests.txt` | 47 passed | guided tuner 观察、失败计数和 stock API 合同通过 | compile failure 被当成 solver UNSAT |
| `evidence/G4/capacity-unknown-tests.txt` | 85 passed | capacity/guided 联合回归通过，capacity-unknown 保留 | 超容量候选被静默删除或免费通过 |
| `evidence/G4/full-autotune-tests.txt` | 91 passed, 2 skipped | stock+guided autotune 回归结果 | 两个 skip 是 parity 完成证据；或默认 tuner 被静态模型取代 |

### 1.1 SM120 calibration 的实际来源

`manifest/calibration-sm120.json` 同时提供 `latency_model` 和 `cost_model` 两个视图。静态核对显示：

- `latency_model.arch=sm_120`、`status=measured`，并保留 `clock_hz`、SM count、memory-path bandwidth、FP16/BF16/TF32 throughput、FP32 FMA 和 exp2 等来源。
- `cost_model.schema=tilelang.cuda.kernel-cost-calibration`、`status=measured`，12 个 `field_sources` 均为 measured。
- 11 个完成的独立 benchmark 是 `empty_launch`、`dram_copy`、`l2_resident_copy`、`shared_read`、`shared_write`、`bf16_mma_gemm`、`f16_mma_gemm`、`tf32_mma_gemm`、`cuda_fp32_fma`、`sfu_exp2`、`barrier`。
- `model_assumptions` 明确区分 streaming proxy、访问模式相关 shared rate、独立 dtype 测量、单流 event 和 launch/event 修正的近似边界。

`manifest/hardware-sm120.json` 或 `evidence/G2/hardware-probe-complete.txt` 中的设备属性是硬件/driver 报告值；其中 `calibration.status=not_performed` 的 probe 视图不能替代上述 microbenchmark calibration。即便 rate 是 measured，`latency_model` 中未提供或被消费的 default/fallback 字段仍须按 `field_sources`、`fallback_used` 和 `require_measured` 处理。

## 2. Z3 scheduler 与 pipeline 的真实闭环范围

### 2.1 Owner exclusion：solver 已实现，CUDA runtime 未实现

`tilelang/cuda/transform/z3_scheduler.py` 中：

- 直线 `z3_schedule_python` 为 owner pair 建方向 Bool，并分别施加两个方向的 latency exclusion。
- loop `z3_schedule_loop_python` 对 owner pair 施加正向和反向 wraparound 四式，只使用 residue `r`，不是完整 CUDA ring runtime。
- 对应直接 oracle 是 `testing/python/transform/test_tilelang_cuda_z3_scheduler.py::test_straight_line_owner_exclusion_and_pipe_order` 和 `test_loop_owner_forward_and_wraparound_exclusion`，并由 49-case solver 日志覆盖。

但是 `src/transform/z3_pipeline_planning.cc::Z3PipelinePlanner::PlanLoop` 在调用 solver 前执行：

- 读取 `tl_z3_pipeline_owner_exclusion_deps`；
- 要求其为空；
- 非空时明确报错，理由是当前 CUDA pipeline IR 不能表达 owner lifetime/wraparound。

`testing/python/transform/test_tilelang_transform_z3_pipeline_planning.py::test_unexpressible_dependencies_fail_before_solver` 同时证明 guard 在 solver 调用前触发。故 Z3-005/009 只能写 **solver constraints implemented and host tested**。Owner planner、物理 ring、counter/epoch、sync、barrier channel 和最终 runtime 都没有因此完成。

### 2.2 Stage-order 与 Let/Bind：支持无 clone 子集

当前 `src/transform/z3_pipeline_planning.cc`：

- 校验 stage-order task index，并转发给 solver；
- 从返回 starts/II 重建 raw stages 后，再次检查 `raw_stage[upstream] <= raw_stage[downstream]`；
- 不生成 Let clone，也不把绝对 starts 写入 CUDA pipeline IR。

对应 host oracle：

- `test_loop_stage_order_changes_minimum_ii`：solver 约束本身；
- `test_stage_order_is_forwarded_and_checked_after_projection`：adapter 转发和投影；
- `test_stage_order_rejects_a_solver_result_that_violates_it`：伪造 solver 结果也会被拒绝；
- `test_stage_order_does_not_bypass_scheduled_bind_validation`：stage-order 不会绕过 downstream Bind 限制。

`src/transform/inject_pipeline.cc::ValidateScheduledBindDependencies` 要求不可 replay 的 scheduled scalar Bind 与每个 consumer 同 stage 且 producer order 在前；跨 stage 仍明确失败。Replayable Bind 是另一条受限路径，不是 general Let clone。Clone 链、重命名、defs-before-uses 和跨 stage snapshot 仍未闭环。

**文档同步**：`tilelang/language/annotations.py::z3_pipeline_annotations` 与 `docs/design/cuda_z3_task_analysis.md` 已按当前源码统一：owner exclusion 仍拒绝；stage-order 的无 clone 子集已支持，跨 stage scheduled Bind 仍拒绝。

### 2.3 自动 task analysis：有主链，但不是 arbitrary-IR

主链位于 `tilelang/cuda/pipeline.py::CUDAPassPipelineBodyPrologue`：

```text
UnrollLoop -> Simplify
-> InferZ3PipelineProblem (opt-in)
-> Z3PipelinePlanning   (opt-in)
-> PipelinePlanning
-> InjectSoftwarePipeline
```

`src/transform/z3_task_analysis.cc` 和 `src/transform/z3_pipeline_planning.cc` 使用一致的受限 task stream：展开顶层 `SeqStmt`，去掉 buffer declarations 和 replayable scalar Bind，其余顶层 statement 各为一个 task。它不是 Ascend `ScheduledTIR`/`IRStructure` 的 Task/Control tree。

当前自动提取边界包括：

- RAW/WAR/WAW 与 shared resource mask pair 可推导；显式字段存在时不覆盖。
- dense copy、dense GEMM、parallel/scalar expression 有白名单 feature 提取；mixed-input GEMM、非静态 extent、While、未知 pointer base 和无法表示的 opaque/stateful 依赖会报错或要求显式 override。
- 未识别 tile op/extern arithmetic 记为 unknown work，并保守扩大 structural resource mask；这不等于获得精确成本或完整依赖。
- buffer size 非静态时是错误，不退化成 0；自动 shared memory group 也不是整 kernel 最终 allocation proof。

跨迭代测试 `testing/python/transform/test_tilelang_z3_task_analysis.py::test_distance_two_conflict_is_not_dropped_by_distance_one_reasoning` 参数化 `increasing` 和 `decreasing` 两种下标：

- `k` 对 `k+2`；
- `k` 对 `6-k`。

两者都验证只在 distance 2 相遇的冲突不会被 distance-1-only  reasoning 丢弃，并保留保守固定 distance=1 依赖。它证明当前仿射分析同时处理正、负斜率的这一子集，不证明任意符号下标、strided view、控制流或任意 IR。

### 2.4 Multi-async producer-stage：保守同阶段或明确拒绝

当前有两道不同的边界，不能混写为“多 async producer 已支持”：

1. `src/transform/z3_task_analysis.cc::ValidateAsyncSameStage` 对已有 async producer annotations 提取 direct producer/consumer pairs。automatic schedule 遇到需要 same-stage 的 pair 会拒绝；manual schedule 要求每对 producer/consumer stage 相等，并写审计 pairs。`test_async_same_stage_is_audited_for_manual_schedule_and_rejected_otherwise` 覆盖这两种行为。
2. `src/transform/inject_pipeline.cc::PopulateWaitCounts` 在 consumer 的 reads 命中多个 async producer stages 时仍执行 guard：`A dependency on multiple async stages is not supported`。已提交首次设备日志 `evidence/G3/device-integration-test.txt` 就保留了这条真实失败。

`testing/python/transform/test_tilelang_transform_Inject_software_pipeline.py::test_async_pipeline_groups_multiple_copy_producers` 证明的是**同一 stage 的多个 copy producers**及 commit/wait 处理；`test_async_pipeline_waits_for_individual_physical_commit_group` 也不是跨多个 async producer-stage 的任意依赖证明。`src/transform/pipeline_planning.cc::EmitImplicitAsyncAnnotations` 的 `(stage, direct_copy_last_use)` group 和 transitive first-stage producer 归组不能被引述为 multi-stage 支持。

### 2.5 UNSAT、unknown、串行 fallback 与 compiler failure

`tilelang/cuda/transform/z3_scheduler.py::_check_feasibility` 只把 `sat` 转为 true、`unsat` 转为 false；其他 verdict 抛出 `Z3ScheduleUnknownError` 并带 `reason_unknown()`。因此：

- 直线 unknown 不会被报告为 UNSAT；
- loop 在首个或后续 II candidate 遇到 unknown 都立即 abort，不扩大 II、不串行 fallback；
- automatic loop 只有在二分搜索结束、没有 sat model 且所有已检查候选均为 UNSAT 后，才生成 source-order serial starts、单版本和保守 II；
- manual infeasible 使用 dedicated infeasible error，不走 automatic serial fallback。

对应测试包括 `test_straight_line_unknown_is_not_reported_as_unsat`、`test_loop_automatic_serial_fallback_only_after_unsat_exhaustion`、`test_loop_mocked_unknown_never_falls_back`、`test_loop_unknown_after_sat_aborts_instead_of_pruning`。

`src/transform/z3_pipeline_planning.cc` 自身没有 catch 后改走 stock/topological schedule 的路径；solver exception 由 `test_unsat_and_unknown_propagate_without_fallback` 验证为向外传播。需要同时注意：当前 solver 返回 ABI 只有 `(starts, versions, II)`，没有 SAT/UNSAT/fallback verdict 字段。因此“adapter 没有自己的静默 fallback”不等于“调用方能从返回值独立审计每次求解 verdict”；不得把一次成功返回直接写成 SAT/最优性证明。

Guided search 是另一调用链，但 unknown 语义一致保持保守：

- `tilelang/autotuner/space.py::Z3Pruner` 只 prune UNSAT；solver unknown/timeout/resource-limit 保留并单列。
- `tilelang/autotuner/space.py::Indeterminate` 是外部已知 unknown，不构造 solver、不增加 solver-check count、绝非 UNSAT。
- constraint builder error 变成显式 `ConstraintBuildError`，不是 UNSAT。
- `tilelang/autotuner/guided_tuner.py` 的 compile/validation failure 走独立 trial/summary 计数；`test_validation_and_compile_failures_are_counted_not_silently_dropped` 和 `test_all_compile_failures_still_raise_and_leave_error_summary` 证明 compiler failure 不会被伪造成 solver UNSAT。

## 3. Task latency/II 与 whole-kernel ranking 必须分开

### 3.1 `latency_model.py`：task cycles

`tilelang/cuda/latency_model.py::estimate_task` 输出单个 task 的整数 `latency` 和 `ii`，主要结构为：

- 每个 memory access 的 completion 逐项累加；
- compute 资源按 GEMM/vector/scalar 的活跃项取 max；
- memory II 先按 path 聚合，再与其他资源 II 取 max；
- latency/II override 独立，最终仍需满足可调度合同；
- 缺失或零 rate 使用显式非免费 placeholder，并进入 `fallback_used`/reliability；`require_measured=True` 可拒绝。

`compose_serial`、`repeat`、`branch`、`aggregate_phase`、`aggregate_per_core` 和 `repeat_per_core` 是纯成本组合 helper。它们不生成 pipeline IR，不选择 whole-kernel 候选，也不证明设备 absolute cycles。

### 3.2 `cost_model.py`：whole-kernel candidate ranking

`tilelang/cuda/cost_model.py::CudaCostModel.predict` 输出 `(predicted_ms, terms)`，面向同一 operator/shape 的候选排序。它组合 launch、CTA waves、occupancy、serial phases、全局/shared/LSU/SFU/barrier 和 stage-overlap 近似。`model_card()` 明确写着 predicted absolute milliseconds 未经独立同条件测量时不能当成实测 latency。

`tilelang/autotuner/guided.py::GuidedSearch` 对该 score 排序；`tilelang/autotuner/guided_tuner.py::GuidedAutoTuner` 只负责候选选择和结构化证据，编译、正确性验证和实际测量仍由 stock `AutoTuner` 路径完成。静态 task cost、whole-kernel ranking 和实测 tuner 不是同一个机制。

### 3.3 Rasterization L2 locality 显式未建模

`tilelang/cuda/cost_model.py::_gemm_like` 接受 `enable_rasterization`/`rasterization`，但不会虚构 locality gain：

- 工作量不因此获得任意加速系数；
- 记录 `gemm.rasterization_l2_locality_unmodeled`；
- model card 和 prediction fallback count 保留该限制。

`testing/python/cuda/test_cuda_cost_model.py::test_rasterization_is_accepted_and_explicitly_unmodeled` 是直接 oracle。模型同分不代表硬件上 rasterization 无差异。

### 3.4 Capacity-unknown 保留，不是 UNSAT 或免费通过

`tilelang/cuda/cost_model.py::CostModelCapacityError` 表示 shared/register/thread 等物理容量使模型无法给分。`tilelang/autotuner/guided.py::GuidedSearch` 只对这一专用异常生成：

- `score=null`；
- `unknown=true`、`error_type=capacity`；
- 有限诊断 terms，不伪造 predicted ms/occupancy；
- 排在有限 score 之后，但保留 deterministic identity/order；topK 预算覆盖时仍可进入 compiler 和后续验证。

`test_full_stage_capacity_unknown_survives_z3_lower_bound_and_enters_top_k` 还区分了“Z3 必要容量下界 SAT”和“完整 stage capacity unknown”。普通模型错误、NaN/Inf、非法 config 和 compiler failure 不走这个 capacity-unknown 通道。该机制不能用来实现或替代 `annotate_unlimit_memory`，也不能把 capacity-unknown 删除后缩小合法宇宙分母。

## 4. 不得互相冒充的六类机制

| 机制 | 当前位置 | 输出/用途 | 与 parity 的边界 |
|---|---|---|---|
| CUDA Python Z3 scheduler | `tilelang/cuda/transform/z3_scheduler.py` | starts、buffer versions、II；直线 makespan 或 loop modulo constraints | 不是 layout 分数、符号 prover 或实测 tuner；owner 仅 solver 层完成 |
| CUDA task latency model | `tilelang/cuda/latency_model.py`、`tilelang/cuda/task_cost.py` | task latency/II cycles、resource mask、reliability/fallback | 不是 whole-kernel predicted ms，也不替代 schedule runtime |
| CUDA whole-kernel cost model | `tilelang/cuda/cost_model.py` | candidate predicted ms 和 itemized ranking terms | 不是 task latency/II，也不是绝对性能证明 |
| 通用 layout cost model | `src/transform/layout_inference/layout_cost_model.cc` | layout attempt 的 mem/registers 等排名 | 不决定 task start、II、buffer ring 或 sync |
| TVM Z3 prover | `3rdparty/tvm/src/target/z3/z3_prover_on.cc` | 符号表达式/区间可证性 | 不是 scheduler，也不共享其完成证据；另有独立 context/预算语义 |
| Ascend DeepGEMM heuristic | `examples/ascend/deepgemm/_heuristics.pyx` | 特定 DeepGEMM 配置搜索和 ns heuristic | 不是通用 TIR task cost、CUDA calibration 或 Z3 schedule |
| Stock/Guided AutoTuner | `tilelang/autotuner/tuner.py`、`tilelang/autotuner/guided_tuner.py` | 编译、验证、实测和候选选择；guided 增加显式搜索/观察 | 静态模型可指导候选，但不能取代正确性和实际测量 |

## 5. 设备证据的精确范围

`testing/python/language/test_tilelang_cuda_z3_pipeline.py::test_default_off_and_opt_in_real_solver_and_numerics` 是当前允许引用的设备用例：

- 构造两个 global->shared copy、一个 parallel A+B、一个 fragment->global store 的四次迭代 pipeline；
- 显式提供 latencies、IIs、resource flags、data deps、resource deps、`enable_offset=False` 和 verbose debug；
- default-off 编译执行时 solver call 计数必须为空；
- opt-in 编译执行时调用真实 loop solver；
- 两种路径都用 `torch.testing.assert_close(actual, a+b, rtol=0, atol=0)` 检查输出。

`evidence/G3/device-integration-test-retry.txt` 记录该用例 1 passed。它可以支持 PIPE-001、部分 resource/modulo/sync/buffer-ring 和 public annotation 的 **DEVICE_VALIDATED_SUBSET**。它不支持 owner exclusion、general Let clone、multi-async producer-stage、automatic arbitrary-IR、所有 buffer/layout 组合或性能结论。

首次 `evidence/G3/device-integration-test.txt` 的 multi-async-stage failure 必须和 retry 一起保留：retry 证明当前特定用例通过，历史失败和当前 `PopulateWaitCounts` guard 则证明通用 multi-stage 缺口仍存在。

## 6. Ascend 硬件特有条目与替代验证

下表的“替代验证”只验证独立的 NV 通用机制，不是对 Ascend 指令/物理池的伪造 parity。

| ID | 无直接对应的原因 | 可做的独立 NV 验证 | 仍未闭环的边界 |
|---|---|---|---|
| LAT-012 | ND2NZ scatter/post-copy 对应 Ascend ND/NZ fractal 指令和 D-group | 通用 CUDA memory geometry、bandwidth 和 unknown fallback 测试 | 不生成 ND2NZ 同名 pass；CUDA swizzle/layout 另行建模 |
| IR-006 | L0 fractal plane/pitch 是 Ascend 物理存储 ABI | 通用 affine region/disjoint 与 conflict 测试 | CUDA 地址/swizzle 独立性不能沿用 L0 pitch 规则 |
| PASS-017 | hard_event/get_buf/rls_buf、8-slot knapsack 和 32-slot mutex 是 Ascend ABI | mbarrier/named-barrier 的独立分配与 wait 测试 | 不生成伪 mutex；PASS-010 通用 allocator 仍 PARTIAL |
| PASS-023 | InsertNd2Nz 展开的是 Ascend NZ 物理格式 | 普通 CUDA copy/layout lowering 测试 | 通用“所有工作进入调度”仍未由 PASS-020/022 闭环 |
| PASS-024 | Ascend Cube fractal storage、alias 和 pitch 专用 | CUDA 自身 LayoutInference/version-layout 测试 | 不继承 fractal 物理证明；PASS-019 仍 PARTIAL |
| API-012 | 双目的 FixPipe 和 `unit_flag_ctrl` 是 Ascend 指令控制 | 普通 CUDA store、mbarrier 和 async-token 测试 | 普通 store 不是双目的 FixPipe；无同指令 parity |
| HW-001 | AIC/AIV0/AIV1 三核 mask、bank 和 cthread 启动模型专用 | CUDA warp/warpgroup role 的独立 WS 测试与 G3 ownership guard | role 抽象不是 bit mask；PASS-003/007 未实现 |
| HW-002 | L0A/B data 与 SF 的同 owner/version/counter/address 绑定协议专用 | 通用 Buffer.data/version map 和 shared capacity 测试 | 无 L0 SF binding；通用 map 不构成等价 group 证明 |
| HW-003 | AIC+双 AIV 共享 3KiB SS pool 和 packed publication 专用 | CUDA hardware-spec 的显式 shared capacity 与 solver group 测试 | 不假设同容量或同语义物理池 |
| HW-004 | 8 intra、16/AIV cross、sid*16 flag namespace 专用 | CUDA mbarrier phase 和 named barrier ID 的独立测试 | 不复制 8/16/bank；PASS-010 通用分配仍 PARTIAL |

## 7. 尚未闭环的硬缺口（证据索引）

1. **Owner ring/runtime/sync**：solver 测试见 G1；adapter 拒绝见 `src/transform/z3_pipeline_planning.cc` 和 `test_unexpressible_dependencies_fail_before_solver`。owner planner、counter/epoch、ring、sync 和 channel allocation 均不能以 solver 约束代替。
2. **General Let clone**：stage-order 转发/复核已完成 host 子集；`src/transform/inject_pipeline.cc::ValidateScheduledBindDependencies` 仍拒绝跨 stage scheduled Bind。需要真正 clone/substitute/defs-before-uses 及下游 IR 证据。
3. **Multi-async producer-stage**：manual same-stage 审计与 automatic 拒绝见 `src/transform/z3_task_analysis.cc`；多 stage downstream guard 见 `src/transform/inject_pipeline.cc` 和首次 device failure。需要多 producer-stage wait/commit 语义与数值测试，不是再增加一个同 stage 多 producer 测试。
4. **Automatic arbitrary-IR**：当前是 flat/static/whitelist 主链。需要 Task/Control 结构、guard、动态 bounds、Let/special state、嵌套 timestamp、保守 dependency/capacity 证明和 negative/positive IR 矩阵；distance-2 正负斜率仅是子集证据。
5. **硬件特有项**：第 6 节十项保留 `NOT_APPLICABLE_HARDWARE`，任何通用替代验证都不能把这些 ID 改成已实现。
6. **可审计 verdict 与最终集成**：solver 三元素返回不含 verdict/fallback；完整 compile/device failure 诊断、绝对时间保持、exact placement、owner/role/capacity 和全 workload 验证未闭环。
7. **最终 workload 结论**：本审计不使用进行中的 GDN 复测或其他 agent 的 operators 结果提前下结论，也不宣称 GDN/经典最终验收完成。

## 8. 本次静态检查结果

以下检查均只读文本、Python AST、JSON 和 git 状态，没有导入项目或重新执行测试：

- ID/表格：84 行、84 个唯一 ID；分组 `PIPE=2`、`Z3=19`、`LAT=15`、`IR=6`、`PASS=25`、`API=13`、`HW=4`，无缺失、重复或额外 ID；每行六列且状态属于已定义集合。
- 分类计数：`IMPLEMENTED_HOST_TESTED=11`、`DEVICE_VALIDATED_SUBSET=8`、`PARTIAL=37`、`BLOCKED_EXPLICIT=2`、`NOT_IMPLEMENTED=16`、`NOT_APPLICABLE_HARDWARE=10`；逐项重新计数与第 6.1 节每个 family 和总计完全一致。
- 测试引用：以 Python AST 验证 156 个直接测试函数引用，全部存在；37 个 source/test/log alias 目标及两个 CAL 文件（合计 39）均存在。
- 路径/链接：43 个 repo-relative path/glob 引用可解析；唯一不存在的路径是第 0 节明确标注的历史拟建 `testing/python/cuda/auto_schedule`，不计作当前链接。未发现 Markdown relative link 错误或额外缺失路径。
- 关键语义：19 个 owner/stage/Bind/multi-async/unknown/rasterization/capacity 源码标记全部存在。
- 日志：11 个结果 pattern 与已提交日志一致；calibration 另通过 `status=measured`、11 个 complete benchmark、所有 rate `source=measured` 检查。
- 文件范围：tracked diff 仅修改 inventory；新建本附录。没有 commit，也没有修改其他源码、测试或实验结果。
