# Ascend Z3 调度与延迟成本模型：CUDA 功能迁移零遗漏清单

## 0. 审计口径、基线与状态定义

- 仓库：`/home/zhaosiying/codebase/tilelang-cuda-z3`；分支：`cuda-z3-cost-model`；原始 Ascend 审计 base：`80364b3d9758854a9692cc0f14089025f54d3f3f`。本次 parity 核销另对当前工作树 HEAD `4bcc17c4e8002ded44464a2b71f21e0a73fff9f8` 及允许使用的已提交证据复核；不把文章、命名或互联网资料当作源码证据。
- 本文中的“源”指 Ascend 实现，“NV”指 CUDA 当前实现。路径均相对仓库根目录。第 1–5 节保留源行为基线；第 6 节才是当前逐项状态，旧文中的“拟”不再是完成证据。
- 当前状态定义：
  - `IMPLEMENTED_HOST_TESTED`：该栏明确的 NV 实现有直接 host 测试和已提交通过日志；**不隐含 Ascend 全语义、device 或端到端完成**。
  - `DEVICE_VALIDATED_SUBSET`：在 host 证据外，还有已提交设备日志直接覆盖该行写明的子集；未测 IR、shape、架构和性能不在核销范围。
  - `PARTIAL`：已有实现或可复用下游，但条目内关键语义、自动分析或集成仍缺失。
  - `BLOCKED_EXPLICIT`：关键路径未实现，当前源码以明确 guard 拒绝；guard 不是功能完成。
  - `NOT_IMPLEMENTED`：没有可接受的 parity 实现；仅有同名/相邻 CUDA pass 不算完成。
  - `NOT_APPLICABLE_HARDWARE`：仅用于 Ascend 专有指令、ABI、物理池或 bank 无 NV 直接对应的最小单元；通用算法仍须在关联条目单独核销。
- 下文 `TC` 仅保留为原始拟建目录 `testing/python/cuda/auto_schedule` 的历史缩写；第 6 节已替换为当前真实测试路径，不得再由 `TC` 推断文件存在或状态为 PLANNED。
- 已全量枚举 `testing/ascend/auto_schedule`：22 个测试文件、218 个测试函数，另有 `_task_utils.py`、`_scheduled_ir.py`、`_reuse_utils.py` 三个共享 helper。参数化组合的核销规则见文末。
- 本次只做 CPU 源码阅读和静态脚本检查，不运行安装、native build、GPU、profiler 或新测试；执行结果均引用已提交日志并标明覆盖边界。进行中的 GDN/经典实验不作为本清单的提前验收依据。

## 1. 必须分开的六类机制

| 机制 | 本地源位置与符号 | 输入、输出和使用时点 | 与本次迁移的边界 |
|---|---|---|---|
| Ascend Z3 调度器 | `tilelang/ascend/transform/z3_scheduler.py::z3_schedule_python`、`z3_schedule_loop_python`；`src/ascend/transform/auto_schedule.cc::ScheduleBuilder` | 在 `AutoSchedule` 中对已物化 task 求 start、II 和 buffer versions；直线段最小化 makespan，循环段二分最小可行 II | 是本次核心。不能用布局打分、符号 prover、GEMM 配置 heuristic 或实测 tuner 代替 |
| 通用 layout cost model | `src/transform/layout_inference/layout_cost_model.{h,cc}::LayoutCostModel`、`RegisterCountCostModel`、`IOAwareCostModel`；`src/transform/layout_inference/layout_inference.cc` | 对一个连通分量的完整 layout attempt 排名；`AttemptCost{mem,regs}` 字典序。默认 register-count 统计 spill/register，opt-in io-aware 对 fragment↔global copy 和直接访问 GM 的 parallel loop 计算 `max(bw,issue)` | 只选择数据布局，不决定 task 的绝对开始时间、循环 II、同步或 buffer ring；不得复用其分数作为 `T.Task.latency/ii` |
| TVM Z3 prover | `3rdparty/tvm/src/target/z3/z3_prover_on.cc::Z3Prover::Impl`；调用点如 `src/ascend/transform/auto_schedule/dependency_analysis.cc::RegionsMayConflict` | 对符号区间、guard implication、互斥和不可交证明返回是否可证；`CanProve(expr)` 通过检查 `!expr` 是否 unsat。调度依赖中另设 `rlimit=50000` | 是证明器，不是调度优化器。它有自己的 lazy private context、journal replay、确定性 AST slot 管理和 `random_seed=42`；与 Python scheduler 的 context/lock/rlimit 要分别测试。旧 `EnterZ3ContextScope/ExitZ3ContextScope` 已为 no-op，不能用作隔离方案 |
| Ascend DeepGEMM 示例 heuristic | `examples/ascend/deepgemm/_heuristics.pyx::select_gemm_config`、`make_gemm_candidate`、`get_gemm_cost`、`get_gemm_k_cost` | 为特定 DeepGEMM kernel 搜索 block/mad/stage/resident/L2 配置；以 ns 估算 GM/L2/L0/compute/epilogue 并确定性剪枝、细化 | 是应用层专用配置搜索，不是通用 TIR task 成本模型，也不生成 Z3 调度。不得把其中带宽、L2 或 resident 规则默认为 NV `EstimateLatency` |
| 默认 `AutoTuner` | `tilelang/autotuner/tuner.py::AutoTuner`，尤其 `_record_benchmark_result`；`tilelang/autotuner/__init__.py` | 编译多个用户配置并实测 latency，以最小实测值选择 best config/kernel，支持 cache、并发和 early stop | 是运行期 profile/search，不是编译期静态 latency/II 公式。NV 静态模型可指导搜索，但不能声称取代默认 tuner 的实测校准与正确性验证 |
| 现有 CUDA 自动调度路径 | `src/cuda/transform/auto_schedule.cc::AutoWarpSpecialization`、`SchedulerRegistry`；`role_based_scheduler.*::RoleBasedSchedule`；`tilelang/cuda/pipeline.py::CUDAPassPipelineBodyPrologue` | opt-in `tl.enable_auto_warp_specialization="role_based"` 时生成 `WSSchedule`，再做 warp specialization；未设置时 no-op。另有通用 `PipelinePlanning`/`InjectSoftwarePipeline` | 是可复用的 CUDA 插入点和 role/WS 基线，**不是** Ascend 的 Z3 makespan/II 求解器，也不是默认 `AutoTuner`。新 pass 必须明确与这两者的先后和所有权，不得重复调度同一 IR |

## 2. Ascend pipeline 调用链与迁移插入点

### 2.1 从 compile 到 backend pipeline

1. `tilelang/engine/lower.py::lower` 经 `create_backend_context` 进入 `lower_with_context`；`lower_to_host_device_ir` 先做 `PreLowerSemanticCheck`，再调用 `context.lower(mod)`。
2. `tilelang/backend/module.py::create_backend_context` 规范化 target 并由 `_resolve_backend_module` 选择唯一 backend；`BackendModule.lower` 调用 `get_pipeline(target).lower`。
3. `tilelang/ascend/backend.py::BACKEND`（plain AscendC）与 `PTO_BACKEND` 都在 `pipelines={"ascend": pipeline.ascend_pipeline}` 上复用同一 lowering；二者只在 codegen/execution backend 等处分开。
4. `tilelang/backend/pass_pipeline/pipeline.py::PassPipeline.lower` 进入命名 instrumentation 后执行 `tilelang/ascend/pipeline.py::AscendPassPipelineBody`。
5. `tilelang/ascend/pipeline.py::ascend_pipeline = PassPipeline("ascend", AscendPassPipelineBody)` 是注册对象。`tilelang/ascend/__init__.py` 导入 `transform`，`tilelang/ascend/transform/__init__.py` 再导入 `z3_scheduler`，因此两个 `tvm_ffi.register_global_func` 在 C++ 查 FFI 前完成 Python 注册。
6. `allow_autoschedule` 读取 `PassConfigKey.TL_ENABLE_AUTO_SCHEDULE`（默认 true）。false 时保留未自动调度 kernel，并在后段执行 `InferBufferAliases`，不改交 CUDA/其他 scheduler；true 时由 `InsertSync` 产生 alias contract。
7. `should_disable_shared_memory_reuse` 独立控制 UB 复用；禁用它不等于禁用同步或 flag 规划。

### 2.2 标准 pass 顺序（源码顺序不可随意交换）

以下按 `AscendPassPipelineBody` 分组，组内也保持源顺序；括号注明迁移时必须保留的语义原因。

1. **launch 与前置规范化**：`BindTarget` → `MaterializeKernelLaunch(lower_grid_binding=True, lower_thread_binding=False, default_threads=None, launch_dim_tags=["cthread"], unsupported_annotations=["cluster_dims"])` → 可选 `LetInline` → `AddWrapperForSingleBufStore` → `LegalizeNegativeIndex` → 可选 `VerifyParallelLoop` → `InjectAssumes`。
2. **让 task/依赖看到真实读写**：`RewriteDualCopy` → `LegalizeSimdMerging`（merge 目的改为 rw）→ `Simplify` → `CanonicalizeLegacyReducer` → `VerifyReducerEpoch` → `VerifyBufferInit`。
3. **展开和布局**：`UnrollLoopSkipVF`（仅显式 unroll、跳过 VF）→ `Simplify` → `AscendLayoutInference` → `ReducerPlanAndMaterialize` → `LayoutVisual` → `NormalizeAscendFractalStorage`。
4. **在调度前生成全部工作**：`InsertNd2Nz` → `VFChecker` → `AscendInsertOOBPadding`（copy tail clamp 和 GM→L1 fill 都成为可调度的 op）。
5. **自动调度主干，仅 `auto_schedule_enabled=true`**：
   `NormalizeControlFlowForSchedule` → `NormalizeConflictHints` → `MaterializeScheduleUnits` → `AssignCore` → `AnnotateMultiBufferEligible` → `EstimateLatency` → `AutoSchedule` → `PrepareMultiBuffer` → `ResolveCore` → `InsertSync` → `MaterializeMultiBuffer` → `LowerScheduledTIR` → `RestoreWhileLoops`。
   这个顺序同时满足：task 边界先于成本；最宽 core candidate 先于 Z3；owner 先于版本求解；counter/epoch 先于最终 core narrowing 和 sync；物理版本先于 pipeline lowering。
6. **公共后lower**：`Simplify` → `NormalizeBufferVersion` → `AscendSimdVFLowerParallel` → `AscendLowerTileOp` → `VerifyReducerConsumed` → `DecoupleTypeCast` → `LegalizeVectorizedLoop` → `LegalizeSafeMemoryAccess` → `LowerAccessPtr` → `Simplify` → `HoistNonRestrictParams`。
7. **展平、版本 layout 与向量化**：`HoistGlobalBufferAllocations` → `LowerOpaqueBlock` → `Simplify` → `RewriteAscendBufferVersionLayout` → `NarrowDataType(32)` → `FlattenBuffer` → `ConfigIndexBitwidth` → `tirx Simplify` → `VectorizeLoop` → `RewriteFp4ToFp4x2`。
8. **收尾**：`LoopUnswitching` → `UnrollLoop` → `RenormalizeSplitPattern` → `tirx Simplify` → `AscendRemoveNoOp` → `HoistIfThenElse` 首个子 pass → `tirx Simplify` → `AscendRemoveNoOp` → `VerifyMemory` → `AnnotateEntryFunc` → `InferFragment` → `LowerThreadAllreduce` → 可选 `AscendThreadSync("global")` → `AnnotateDeviceRegions` → `MarkScalarDcacheBypass` → `SplitHostDevice` → `AnnotateReadOnlyParams`。
9. **分配与同步 ABI**：仅非自动调度时先 `InferBufferAliases` → `MergeUBAllocations(align_bytes=32, disable_reuse=...)` → `RewriteFlagToBuf` → `AscendThreadSync("shared")` → `AscendThreadSync("shared.dyn")` → `MergeIfStmt` → `MakePackedAPI` → `Simplify` → `LowerDeviceKernelLaunch`。

**NV 插入要求**：在 `tilelang/cuda/pipeline.py` 明确新 Z3 主干与现有 `AutoWarpSpecialization`/`MaterializeWSSchedule`/`ProducerConsumerWarpSpecialized` 及 `PipelinePlanning`/`InjectSoftwarePipeline` 的互斥或组合合同；不能让两个软件流水器静默重写同一 loop。CUDA 的 target、memory scope、pipe role、layout lowering 和 barrier lowering 必须在对应分析之前稳定下来。

## 3. Z3 调度：接口、全部约束、求解与失败语义

### 3.1 FFI 与资源标识

- 两个全局函数：`tl.transform.z3_schedule_python` 和 `tl.transform.z3_schedule_loop_python`，分别在 `z3_schedule_ffi`、`z3_schedule_loop_ffi` 注册。FFI 将 TVM container 转为 Python list，并用 `_convert_int_tuples` 校验 tuple arity；两个 wrapper 都在 `_z3_lock` 内调用求解器。
- 直接调用 Python 函数不经过该锁；`Z3_AVAILABLE=False` 只延迟到 `assert Z3_AVAILABLE`，import 时先打印警告。
- `ResourcePipe` 源值：MTE1=1、MTE2=2、MTE3=4、Cube=8、Vector=16、Fixpipe=32、Scalar=64。`ScheduleBuilder::NodeResourceMask` 还按 `{AIV0,Cube,AIV1}` 将 pipe mask 左移 `8*i`，所以真实 FFI 值可含多个 core 的 7-bit 段；`HbmPort` 另计 AIV/AIC/store 争用。
- `resource_flags` 用于 loop 的 offset 分组和 debug；Python 并不从它自动生成全部 resource pair。`resource_deps` 由 C++ `HasResourceDependency` 生成，后者还要求同 core pipe 相交或 HBM mask 相交。
- `num_stages` 是自动 buffer 的最大版本数；`memory_groups` 每项为 `[capacity, buffer_id...]`；`buffer_sizes` 是单版本字节数。FP4 等 sub-byte size 在 C++ 以 `ceil(product(shape)*bits/8)` 计算，非静态结果变成 0。

### 3.2 直线段 `z3_schedule_python`

变量与目标：

- 对每个 task 建 `start_i ∈ Int`，约束 `start_i >= 0`；建 `makespan ∈ Int`。
- 对所有 i 添加 `makespan >= start_i + latency_i` 和 `makespan >= 0`，用 `z3.Optimize.minimize(makespan)`。
- solver 使用本次调用私有 `z3.Context()`，`rlimit=1000000`。此入口**没有 seed 参数，也没有设置 random_seed**。

| 输入/约束 | 精确语义 |
|---|---|
| `data_deps=(i,j,latency)` | `start_j >= start_i + latency`。C++ 对普通 buffer/let 和 special register 还会用 `GetDependencyLatency` 计算嵌套子树最坏间隔 |
| `resource_deps=(i,j)` | 仅当 `i<j` 建 Bool `O_i_j`；true：`start_j >= start_i + II_i`；false：`start_i >= start_j + II_j`。这是 issue/II 分离，不是完整 latency 互斥 |
| `owner_exclusion_deps=(i,j,Lij,Lji)` | Bool 选择方向；i 先则 `start_j-start_i >= Lij`，j 先则 `start_i-start_j >= Lji`；用于不相交 owner 共享物理存储的寿命排除 |
| `pipe_order_deps=(previous,current)` | `start_previous <= start_current`；manual mode 对每个共享 pipe 按源码顺序生成，包括 Scalar |
| 输出 | 返回 `start_times` 和按 `(start,index)` 排序的索引；FFI 只返回 starts，C++ 再 stable sort，start 相同保持源顺序 |

C++ 侧在调用前把所有节点 `stage=0` 且 start 初始化 0；`n<=1` 直接返回，不进 Python。直线段 manual `T.Stage` 只能是 0，非 0 必须在 `enable_offset=True` 的 serial loop 内。

### 3.3 C++ 生成的依赖不是单一“RAW 列表”

1. **Let/变量**：source order 的 write→read，直线为 `(i,j,lat[i])`，循环为 `(i,j,0,lat[i])`。
2. **Buffer RAW/WAW/WAR**：`AnalyzeDependencies` 分别考察 write→read、write→write、read→write，不创建 read→read；logical storage 以 `Buffer.data` 身份统一 alias。跨 owner 的同类冲突变成 `kOwnerExclusion`，不是强制一个方向。
3. **嵌套时间**：`GetTaskTimestamp` 沿 control path 累加子节点 start；producer 还累加每层 `(trip_count-1)*II_per_iter`，consumer 不加尾迭代。`GetDependencyLatency` 对所有 task pair 取 `max(producer_time+producer.latency-consumer_time,0)`，反向 owner latency 同法。
4. **SpecialRegister**：包括 `Hf32Mode`、`PadValue`、`LoopControl`、`AtomicMode`、七个 pipe register、`MmadDirection`。Cube 默认读 HF32/MmadDirection；padded copy 读 PadValue；GM store 读 AtomicMode；每个节点默认读 LoopControl；sync call 写目标 pipe register。
   - 直线 writer i 对 reader j：i<j 加 `(i,j,lat[i])`，i>j 加 `(j,i,0)`；WAW i<j 加 `(i,j,lat[i])`。除 LoopControl 外要求 core mask 相容。
   - 循环 reader 边：i<j 加 `(i,j,0,lat[i])` 与 `(j,i,1,0)`；i>j 加 `(i,j,1,lat[i])` 与 `(j,i,0,0)`。WAW i<j 加 `(i,j,0,lat[i])` 与 `(j,i,1,lat[j])`。
   - `loop_break` 写 LoopControl，fence 所有同层 sibling；ControlNode 向外聚合时屏蔽该位，避免错误 fence 外层。
5. **Resource**：直线对所有 pipe（含 Scalar）和 HBM 冲突建 resource pair；循环排除 Scalar 的自动 resource pair，但 manual pipe-order 仍包括 Scalar。跨 pipe 的数据/同步正确性不能由 issue-order 推断。
6. **Conflict hints**：`assume_conflict` 强制冲突优先于匹配的 `assume_no_conflict`；bare Buffer/data Var 匹配整个 storage，region 必须同一 logical Buffer 且 footprint 完全相等；`cross` 分别过滤 same/cross。显式冲突可关联不同 storage，但不参与该 storage 的自动版本变量。
7. **区域证明**：`RegionsMayConflict` 只有证明 half-open 区间某一维不相交才去除依赖；索引/endpoint 先提升到 int64，避免 int32 末端溢出。不同 logical view 即使同 storage 也保守冲突。L0A/B/C 在 pitch 不能证明相等时忽略末两维平面信息而保守冲突；pitch 相等仍逐维比较。TVM Analyzer 与 `ana.z3_prover` 都可证明，后者是 prover 而非 scheduler。
8. **循环距离**：root（`loop==nullptr`）不创建 cross-iteration 检查。automatic ring 检查 distance 0 和 -1；负距离重命名 storage-local iteration suffix 并要求至少一个坐标不同，多 owner 物理 ring 则重命名完整 enclosing loop nest 且允许相同坐标表示不同 epoch；-1 在 C++ 中解析成固定版本、固定 annotation 或负编码的 solver buffer variable。manual ring 从 `(i<j?0:1)` 到 N 逐距离找首个冲突，正距离 d 明确约束 consumer 当前坐标等于 producer 坐标+d。不同具体 AIV 的 UB 同 offset 不冲突。
9. **Stage-order/Let closure**：`enable_offset=true` 时，对传递读取 Let 的节点 U 和写该 Let 输入 buffer 的 WAR 节点 W 添加 `(U,W)`，含义 `k_U <= k_W`；随后跨 stage Let 会 clone，clone 链重命名、defs-before-uses 并插到对应 stage 前。

### 3.4 循环段 `z3_schedule_loop_python`：完整约束系统

每次 feasibility check 对固定候选 `II` 新建 `z3.Solver(ctx=同一个私有Context)`，设置 `rlimit=1000000`；`seed is not None` 时设 `random_seed=seed`。FFI 固定传 `seed=42`，Python 直接接口可传 None 或其他整数。

基本表示：

```text
start_i = k_i * II + r_i
k_i >= 0, 0 <= r_i < II, k_i/r_i 均为 Int
buf_b >= 1, buf_b <= num_stages
```

| 约束类别 | 精确公式/行为 |
|---|---|
| 固定距离 data dep | 对 `(u,v,d>=0,latency)`：`start_v-start_u >= latency-II*d` |
| 可变版本 data dep | `d<0` 时 `buffer_id=-d-1`，改为 `start_v-start_u >= latency-II*buf_buffer_id` |
| 内存组 | 每组有成员时：`Sum(buf_b*buffer_sizes[b]) <= capacity`。C++ 先从容量扣 non-eligible 和 fixed-version buffer；`unlimit_memory` 用 `1<<40` |
| resource pair | 对 i<j 建 Bool。i 先：`r_j-r_i >= II_i` 且 `r_i-r_j+II >= II_j`；j 先：`r_i-r_j >= II_j` 且 `r_j-r_i+II >= II_i`。两条同时覆盖正向和回绕 |
| owner exclusion | 对 `(i,j,Lij,Lji)` 同样四式：i 先要求 `r_j-r_i>=Lij`、`r_i-r_j+II>=Lji`；反向对称。只用 r，不用 k，因为物理 slot 在 modulo issue phase 交换 |
| manual pipe order | `r_previous <= r_current`，每条硬件 pipe 独立保持 source issue order |
| stage order | 对 `(u,w)`：`k_u <= k_w`，允许同 stage，配合 Let clone 在写之前读取 |
| manual stages | manual mode 强制 `k_i == manual_stages[i]`；stage 数量必须等于 task 数、均非负；非零 stage 必须 `enable_offset=True` |
| `enable_offset=False` | 按 `resource_flags` 的每个置位 pipe 分组，对任意两任务同时添加 `start_a-start_b < II` 和 `start_b-start_a < II`，即同 pipe 全部任务绝对 start 跨度 `<II`；不是只对 r 加约束 |
| `enable_offset=True` | 不加上述同资源单 II 窗口约束，允许跨 stage offset；其他依赖仍全部生效 |

补充约束来源：`manual_schedule` 时 C++ 拒绝同一 loop 上的 `tl_pipeline_order` 和 `tl_pipeline_stage`，因为 Ascend manual scheduler 使用 `T.Stage` 和 source pipe order，而不是通用 Pipelined 的 order/stage 列表。未显式给 `num_stages` 时 manual 取 `max_manual_stage+1`。

### 3.5 II 二分、结果重建和 buffer versions

1. 搜索区间：`ii_lower=max(iis)`；`ii_upper=max(sum(latencies),sum(iis))+1`；`best_ii=ii_upper`。循环取 `mid=(lower+upper)//2`。
2. `solver.check()==z3.sat` 时保存 model/start/k/buffer vars，并令 `upper=mid` 尝试更小；其他任何结果都令 `lower=mid+1`。这包含 `unknown`（例如 rlimit 耗尽），**没有单独 unknown 状态**。
3. 因为目标是“最小可行 II”，不是每个候选 II 都求最优 start；每个候选只取一个 sat model。二分依赖可行性随 II 单调这一调度结构。
4. 有解时，取 `offset=min(k_i)`，输出 `start_i = model(k_i*bestII+r_i)-offset*bestII`，规范化最早逻辑 iteration 为 0。
5. `recalculate_buffer_versions=True`（Python 默认且 FFI 未暴露开关）时，从 1 开始，对每条负距离 `(u,v,d,latency)` 计算：

   ```text
   versions[b] = max(versions[b],
                     ceil((latency + start_u - start_v) / bestII))
   ```

   false 时直接读取 model 的 `buf_b`。重算路径没有再次用 `num_stages`/memory group 钳制，迁移时不得暗中假设存在二次验证。
6. C++ 将 `raw_k=start_time/II`（因 `0<=r<II` 可恢复 k），自动 mode 在 offset 开启时发布 `raw_k-min_k`，关闭时全 0；manual 保留请求 stage。所有 start 再减全局 `min_start_time`。
7. 排序使用 `phys_time = adjusted_start - stage*II`；稳定排序，phys_time 相同则**大 stage 在前**，大 stage 对应更早逻辑 iteration。跨 stage Let clone 完成后才计算 control cost。
8. ControlNode 的 `overall_latency = max_child(adjusted_start+child.latency) + II*(trip_count-1)`，并把该值同时写到 control 的 latency 和 II，另存 `II_per_iter=II` 供父层时间戳。
9. 多 owner 选择版本时 `MergeSelectedBufferVersion` 取 max，并传播到同一 L0 data/SF group。源码明确不在 merge 后重加各 owner 容量，可能超过物理池，留待下游/runtime；这是当前已知边界，不得误写为全局容量证明。

### 3.6 确定性、锁、rlimit、debug 与 fallback

| 项目 | 源行为，迁移必须逐项保留或显式改变并测试 |
|---|---|
| 私有 context | 两个 Python 入口每次调用各建 `z3.Context()`，避免全局 AST 状态使重复求解结果漂移；循环的多次 II feasibility 共用本次 context，但新建 solver |
| lock | 两个 FFI wrapper 用同一 `_z3_lock` 串行化 Z3；直接 Python API 不在锁内。CUDA 多线程 compile/grouped compile 要测试锁覆盖整个 solve，而非只覆盖输入转换 |
| seed | loop 默认 42、FFI 固定 42，可传 None 直调；直线不设置 seed。不得宣称两个入口都有相同 seed 控制 |
| rlimit | 两者均为 1,000,000，不用 wall-clock timeout 保证确定性；TVM prover 的 50,000 是另一预算。`unknown` 没有独立诊断 |
| 稳定次序 | Python 直接结果按 `(start,index)`；C++ 直线 stable sort，循环按 phys_time/stage 稳定排序；memory scope 用有序 `std::map`。需用重复进程/重复 compile 验证，而不是只比较一次 model |
| verbose | 输入、依赖、II 搜索和结果打印；loop verbose 还写 `debug/schedule_NNN/schedule_info.json`（首个编号空洞），可选 matplotlib 输出 PNG/PDF。JSON 含 timestamp；图像缺包或写文件失败只 warning，不改变调度结果 |
| 直线无 FFI/异常/返回长度错误 | 自动 mode warning 后返回原 scheduled 顺序，日志称“topological sort”，但代码并未另做拓扑排序；manual mode 硬错误。所有节点仍已是 stage 0 |
| 直线求解非 sat | Python 抛 `RuntimeError("...unsat")`，C++ 自动 mode 捕获并走上述原顺序 fallback |
| 循环无 FFI | C++ `LOG(FATAL)`；日志文字虽写 fallback，但实际不会返回一个 fallback schedule |
| 循环所有候选均不可行 | manual 抛带 pipe order/T.Stage/dependencies/resource 提示的 RuntimeError；自动 mode 回退 source-order 串行：`start_0=0`、`start_i=start_{i-1}+latency_{i-1}`、`II=max(sum(latencies),1)`、该求解调用管理的 buffer versions=1（固定 frontend override 随后仍由 C++ 合并） |
| 空输入 | Python 两入口 n<1 都报错；C++ 直线 n<=1 绕过，循环空 children warning 后返回 |

## 4. `EstimateLatency`：task features、动态形状和全部公式

源实现集中在 `src/ascend/transform/estimate_latency.cc`、`estimate_latency.h::AscendLatencyParams` 和 `auto_schedule/task_annotations.h`。它只给 canonical `tl.ascend_task`/`tl.ascend_per_core_task` 写 metadata，不直接求调度。

### 4.1 Feature 提取和动态形状

1. `TaskCostFeatures::CubeShape`：`{m,n,k,input_dtype,hf32,blockscaled}`。GEMM 序列化整数是 padded/static tile 上限；L0 GEMM 的 M/N 从 C region、K 从 A region（考虑 transA）用 `ConstIntBound.max` 收紧，`min(fallback,bound.max)`，无有限 bound 用 fallback。M/N 后续对齐 16，K 按 compute group 对齐。
2. `Hf32ModeAnalyzer` 在每个 GEMM 前做数据流分析：mode 0=disabled，1/2=enabled；条件分支按 guard join，loop backedge 做最多三次 bit-state 迭代；ambiguous 不用 HF32，落回 FP32 成本。SBlock init 与 incoming 取并集。
3. `CopyInfo`：`src_scope,dst_scope,bytes,dual_dst_ctl,geometry,contiguous_bytes_lower_bound,is_nd2nz_post_copy`。geometry 分 M/coalesced、N/strided、unknown；从源/目的 `StridedLayout::FromBufferRange(...).Coalesce` 推断，保持与 `PlanMTECopy` 的行轴、unit-axis normalization 一致。
4. `bytes` 用 `EstimateRegionElementUpperBound`：常量 extent → analyzer finite bound 与 static shape 的较小值 → 有用 analyzer bound → 最后 extent=1 且 `reliable=false`。等形 copy 在两侧可靠时取较小 payload，只一侧可靠可用另一侧收紧；最终按**目的 dtype** 转字节。
5. dual copy 归一成一个物理 AIV 的 payload：`double` 表示 half-source 形式；动态 tail 优先可靠侧，避免无信息 fallback 1 掩盖真实上界。L0C→UB dual 的端点字节再按两个目的地计 2 倍。
6. MTE contiguous width 取源/目的 coalesced row bytes 的下界最小值；动态 tail 无法证明正下界则为 0，进入 unknown-width 带宽。高维、rank-1、显式/隐式 row、N-split 和 unit axis 都必须与 rewrite 后等价。
7. `AIV lanes`：带 `subcore` metadata 的 projected task 为 1，collective task 为 2。AIV MTE 的 latency/II payload 乘 lanes；带宽参数本身按两个 AIV aggregate 校准。
8. `ND2NZ post_copy` 单独记录 `shared.dyn→shared.l1`，rows/cols/element bytes 上界（rows/cols 先取常量或 analyzer 上界，未知回退 1），并强制 M-like endpoint，不套 raw N-strided UB→L1 带宽。
9. `SBlock` 在 resource 和 feature visitor 中是 atomic boundary，避免把 VF 内 ops 同时计入外层 task 两套成本。

### 4.2 override、校验和通用合成

- `T.Task(latency=...,ii=...)` 分别是对**完整 task body** 的精确 override，可只覆盖一个；另一个仍估算。前端拒绝 bool、负数 latency、非正 ii 和 `latency<ii`；C++ 要求常量整数并再次校验最终 `latency>=ii>0`（latency 可为 0，但若 ii 至少 1，最终仍须 latency>=ii）。
- 未 override 且估计 latency=0 时改为 1；显式 latency=0 不享受该修补。II 始终至少 1。
- `TaskAnnotationRewriter` 合并新 latency/II 时保留 subcore、core_mask、broadcast-fill 等其他 metadata。AutoSchedule 入口再检查每个 task II>0，缺失时提示先跑 `EstimateLatency`。
- 对任意 leaf：
  `total_latency = sum(memory_completion_per_access) + max(vector_latency, cube_latency, scalar_latency)`；纯 Scalar 且无 Vector 时 scalar=1。
  `II = max(1, 各活跃 memory path II, cube II, vector II)`，DMA-only/Cube-only/Vector-only/mixed 分支最后都保持这一资源上限语义，而非强制 total latency。
- 所有除法通过 `safe_div_ceil`：分母<=0 时该项为 0；serial/per-core 聚合有非负和 int64 overflow check；lane/带宽乘法有饱和或检查。零带宽/吞吐“贡献 0”是源模型的估算 fallback，不是硬件免费。

### 4.3 Cube：dense、FP32/HF32、FP8、MX blockscaled

定义 `align(x,g)=ceil(x/g)*g`，`ceil_div` 同上：

```text
k_group = 256 / input_bits
FP32: k_group = hf32 ? 8 : 1
M = align(m,16), N = align(n,16)
普通 ops = 2 * M * N * align(k,k_group)
MX ops   = 2 * min(M,N) * max(M,N,cube_mx_min_mn) * align(k,k_group)
throughput_cycles = ceil_div(ops, throughput(dtype,hf32,blockscaled))
```

- dense Cube latency 从 `cube_base_latency` 起，对所有 shapes 累加 `throughput_cycles`；FP32/HF32 将 completion base 提升至 `cube_fp32_base_latency`，MX 提升至 `cube_mx_base_latency`，但每个 leaf 只收一次最终 base。
- Cube II 从 `cube_min_ii` 起，对各 shape 的 `throughput_cycles` 取 **max**（不是求和），再与 `cube_pipeline_depth` 取 max。
- throughput：FP16/BF16=8192，HF32=4096，FP32=512，FP8=16384，MXFP4=32768 ops/cycle；FP16/BF16/FP8/INT8 为 0 时走 `cube_fallback_throughput=8192`，FP32/HF32 必须>0；INT8 默认 0 所以走 fallback。未识别 dtype 也走 fallback。
- 无 shape 的 Cube fallback：`cube_base_latency + ceil(cube_unit_m*cube_unit_k*cube_unit_n / FP16 throughput)`，unit 默认 16³；此处源代码没有普通 `CubeOperations` 的因子 2。
- blockscaled FP4 用 MXFP4 throughput；其他 MX FP8 使用 FP8 throughput。默认 `cube_mx_base_latency=25`、`cube_mx_min_mn=48`。

### 4.4 VALU、特殊函数、VF 和 Scalar

- 若 task 内 `SIMT_VF`/`SIMD_VF` 有 `tl.vf_latency`（前端 `T.SimtVF(latency=...)` 或 `T.SimdVF(latency=...)`），取 block 中最大正 annotation 作为 `vector_latency`，跳过 op-count/带宽估算；缺省 0 走模型。
- 否则 `vector_latency = max(OperationCounter估计, ceil(VALU读取bytes/valu_bandwidth))`。带宽只累计 `local`、`local.fragment`、`shared`、`shared.dyn` 的**读 region**；默认 `valu_bandwidth=256`。
- OperationCounter 对表达式节点计费，并递归子表达式。有效工作量 `effective_factor`：若表达式不含 loop var 则为 1；否则取包含 loop var 的最大深度，并乘所有深度不超过该值的 loop trip count。单次操作成本为 `max(ceil(effective_factor/throughput),1)`；throughput=0 的未知操作固定 1 cycle。
- 动态 serial loop trip count 在 OperationCounter 中：常量 extent/step 用 ceil；无法常量时默认 100，非正 step 用 extent。`GetSerialTripCount` 另在 composed/control 中优先 analyzer 上界、未知 fallback 100，符号正 step 下界按 1 防止低估。
- if/else 在 OperationCounter 中取条件成本加两分支最大增量；loop 逐层影响有效并行因子。
- 默认吞吐：add/sub/mul/min/max/cmp/logic=51；div/mod=13；special func=8。`FloorDiv/FloorMod` 分别复用 div/mod。
- special func 名单：`exp2,log2,exp,log,sin,cos,tan,asin,acos,atan,sinh,cosh,tanh,sqrt,rsqrt,pow,erf,sigmoid`。op 名包含 `copy` 或 `gemm` 不在 VALU 重复计费；其他未知 Op 用 default=0 的 1-cycle fallback。
- `ND2NZ scatter` 是 opaque template，按下节专门公式，不再把 access_ptr 当额外 VALU op；VF 内重复调用乘所有 loop trip count。

### 4.5 MTE/FixPipe：路径、完成延迟与 II

通用 memory access 是 `completion = descriptor_cycles + base_latency(path,geometry) + ceil(payload/bandwidth)`，逐 access 加到 `memory_latency`。II 按物理 path 聚合后取最大；各 descriptor 的特殊 issue floor/overhead 在聚合前累加。

| 路径 | 默认 base/带宽（cycles, bytes/cycle） | II 与几何规则 |
|---|---|---|
| GM→L1 (AIC MTE2) | base 190；BW 100 | 每 descriptor `max(ceil(bytes/100),64)`；测试两个小 copy 也分别收 floor |
| GM→UB (AIV MTE2) | M base 90，N base 75；M BW 100 | aggregate payload=per-AIV bytes×lanes；每 descriptor `max(ceil(payload/有效BW),13)`；N 规则见下 |
| UB→GM (AIV MTE3) | 普通 base 180，narrow/unknown N base 440；M BW 115 | 每 descriptor `max(ceil(payload/有效BW),10)`；N 行宽<128或未知用 narrow base |
| UB→L1 (AIV MTE3) | base 43；BW 256 | M/ND2NZ post 用 256；raw N 行宽>=160 时 `min(256,width)`，0 用 48，其余 `min(256,3*width/4)` |
| L1→L0A / L0B (MTE1) | base 5；各 BW 256 | 每 descriptor `3+ceil(bytes/256)` |
| L1→L0 SF | base 28；BW 32 | 每 descriptor `1+ceil(bytes/32)`，payload 来自实际 region 和 SF dtype |
| L1→BT / FP_BUF | base 5；BW 32 | 普通聚合带宽项；`ClassifyRegion` 还识别 `shared.bt`，`ClassifyCopy` 识别 `shared.fp` |
| L0C→UB (FixPipe) | 非 dual base 58、BW 128；dual base 55 | 非 dual 走普通项；dual 见下 |
| L0C→GM (FixPipe) | base 200；BW 128 | 与 L0C→UB 共用 descriptor FixPipe II 规则 |

AIV N-strided 有效带宽：

```text
transaction_rate(width) =
  peak * width / (transaction_bytes * ceil(width/transaction_bytes))
```

- GM→UB：transaction=128，peak=102；M=100；width<=0 用 48；width<128 用 50；width>=256 用 `min(100,100)`；中间用 `min(100,transaction_rate)`。
- UB→GM：transaction=128，peak=86；M=115；unknown=40；width<128 用 40；width>=512 用 115；256<=width<512 用 `min(115,110)`；中间用 `min(115,transaction_rate)`。
- 普通路径 fallback 也使用上述 per-path base/BW；没有 CopyInfo 时从 read/write regions 和 pipe mask 分类，并将 AIV geometry 视为 M。
- `mte_descriptor_cycles` 默认 0，但属于每次 completion 的独立参数。

FixPipe 特别规则：

- 非 dual L0C→UB/GM 使用 `fixpipe_bandwidth=128` 和相应 base；quant/conversion copy 的 bytes 以 destination payload 为准。
- dual L0C→UB：`copy_bytes=2*per_AIV_payload`；有效 per-AIV row BW 为 `min(fixpipe_dual_bandwidth=256, 2*row_width)`，未知行宽用 128；completion base=55。
- dual/普通 FixPipe II 每 descriptor 取 `max(endpoint_ii, geometry_ii)`：`endpoint_ii = fixpipe_issue_overhead(2)+ceil(ii_bytes/endpoint_BW)`，endpoint BW 普通 128、dual 256；`geometry_ii=ceil(ii_bytes/有效rowBW)`（无 override 时 0）。descriptor overhead 只在 endpoint 收一次，避免窄行重复计费。
- `ClassifyRegion` 对 L0C source 先给 L0C→UB，再由配对 copy 信息精确分类；不能仅凭 scope 把所有 L0C drain 当同一成本。

### 4.6 ND2NZ scatter/post-copy 成本

`ND2NZ scatter` 要求 7 个参数中 rows/cols/src/dst dtype 均为静态；rows/cols>0，支持 float/half/bfloat16_t 源，以及 same dtype、bf16→f32、f32→bf16 三类 profile：

```text
vector_elements = (src == float) ? 64 : 128
passes = ceil(cols/vector_elements)
cycles0 = setup + passes*pass_cycles
cycles  = cycles0 + ceil(rows*passes*row_cycles_x4/4)
latency = max(min_cycles, cycles)
```

| profile | min_cycles | setup | pass_cycles | row_cycles_x4 |
|---|---:|---:|---:|---:|
| same dtype | 62 | 10 | 10 | 8 |
| bf16→f32 | 98 | 40 | 4 | 13 |
| f32→bf16 | 73 | 16 | 4 | 8 |

`ND2NZ post_copy` 本身作为 MTE3 UB→L1 的 M-like copy 计 memory latency/II；scatter 的上述 vector latency 与 post-copy memory latency 在同一 leaf 中按通用公式合成。它不是一个可被通用 layout cost model 隐式覆盖的免费 layout rewrite。

### 4.7 默认参数总表（其余 MTE 参数见 4.5）

迁移时应把下表视作**源模型配置清单**，不是 NV 校准值。NV 必须按 SM 架构重新测量并记录参数来源。

| 参数组 | Ascend 默认值（完整列举） |
|---|---|
| Cube | base=8，fallback throughput=8192，min II=1，pipeline depth=1，unit M/K/N=16/16/16；FP16/BF16=8192，FP32=512，HF32=4096，FP32/HF32 base=67，FP8=16384，MX base=25，MX min larger MN=48，MXFP4=32768，INT8=0 |
| VALU | add/sub/mul=51，div/mod=13，min/max/cmp/logic=51，special=8，unknown/default=0，bandwidth=256 |
| MTE1/SF/FixPipe | MTE1 base=5，issue=3；SF base=28，issue=1；FixPipe base=58，issue=2，L0C→GM base=200；descriptor cycles=0 |
| 其他物理带宽 | L1→L0A/B=256，L1→SF/BT/FP=32，AIC MTE2→L1=100，FixPipe=128，GM→L1 base=190/minII=64 |
| AIV MTE | GM→UB base=90/BW=100/minII=13；UB→GM base=180/BW=115/minII=10；UB→L1 base=43/BW=256 |
| dual FixPipe | dual BW=256，width scale=2，unknown width BW=128，dual base=55 |
| N-strided GM→UB | transaction=128，peak=102，plateau=50，wide=100，unknown=48，N base=75 |
| N-strided UB→GM | transaction=128，peak=86，plateau=40，wide=110，unknown=40，narrow base=440 |
| raw N UB→L1 | full-width=160，width numerator/denominator=3/4，unknown BW=48 |

### 4.8 composed task 与 `PerCoreTask`

**串行 composed task**（Seq/Attr/If/serial-or-unrolled For/While）：

- 两个非空成本 A→B：
  `latency = max(A.latency, A.ii + B.latency)`；
  `II = A.ii + B.ii`。零 II 项是 identity。
- 重复 T 次：`latency = (T-1)*body.ii + body.latency`，`II=T*body.ii`；T<=0 或 body II=0 得 0。
- 分支：condition 纳入上下文，then/else 分别估算，最终 `(max latency, max II)`；`tilelang_assume`、Bind 和 loop var range 会进入 `ConstrSet`。
- While 用动态 trip fallback 100；parallel loop 和 VF block 不当串行结构展开，而按 atomic leaf 估算。
- leaf 会重新跑 resource/access 分析，再套用单任务模型；最终 composed II 至少 1。

**PerCoreTask**：

- 先估每个显式/推断 `T.Task` candidate；每个 candidate 必须至少一个 dependency-bearing buffer access，所有 candidate 与 group 都是同一个且仅一个硬件 pipe。禁止嵌套 PerCoreTask、重复 marker identity、无法匹配 candidate/wait marker。
- 同一 inter-core wait 前的 candidates 跨 core 可并行，`MaxCost` 对 latency 和 II 分别取 max；wait 分隔的 phases 用 `AddCost` 相加。`cross_core_wait_flag` 是 boundary，set 不是。
- 无 boundary 的 serial loop 保守乘 trip count；有 boundary 时按 prefix/middle/suffix 组合，迭代间额外串行 `suffix(i)+prefix(i+1)`，middle 重复 T 次，该部分重复 T-1 次。
- 条件分支取最大且禁止 branch 内出现 inter-core wait；While 用 100，serial/unrolled loop 用 `GetSerialTripCount`。
- 最终 latency 为相位合成结果；若有 boundary，`II=max(latency,1)`，因为 mode-0 explicit flag 是未版本化协议状态；无 boundary 时 `II=max(cost.ii,1)`。
- `T.PerCoreTask` 对外仍是一个逻辑 task，AutoSchedule 只排一次；同步随后在每个 candidate 的原 guard 下展开，而不是把 candidates 当独立自由任务。

## 5. auto_schedule 相关 pass、IR、分析与公开 annotations

### 5.1 Scheduled-TIR 和核心数据结构

| 源路径与符号 | 必须迁移的合同 |
|---|---|
| `src/ascend/transform/attr.h`：`kAscendTask`、`kAscendPerCoreTask`、`kAscendStage`、`kScheduleUnit` | `tl.schedule_unit` 只承载调度 wrapper/stage；语义 task/core 在 `T.Task`/`T.PerCoreTask` 内。stage 在 schedule unit，core ownership 永远在 task（含 compiler sync task） |
| `auto_schedule/scheduled_tir.{h,cc}`：`ScheduledTIR`、`ScheduledTIRMetadata`、encoder/decoder | 每个 leaf schedule unit 恰好一个语义 task；control unit 是 serial/unrolled For，body 是另一 unit list。只允许无 else 的条件 guard 和白名单 assume/assume-runtime-check attribute guard；非法/未物化 IR 必须 fail，不得猜测 |
| 同上 metadata | kernel root shell、buffer_versions（输入 override/输出 selected）、manual versions、version modes、unlimit scopes、alias map+presence bit、AIV count、root conflict hints。codec 清理再写，校验正版本数、auto/iteration/counter、AIV count 1/2、alias 对称性 |
| `auto_schedule/ir_structure.{h,cc}`：`IRStructure`、`TaskNode`、`ControlNode`、`Guard`、`IRExtraInfo` | Task/Control 两节点树；root 和每个 Control 都有 ordered children。保存 parent/index/stage/latency/II、pipe/HBM/special masks、read/write regions+vars、L0 pitch、core mask、outer constraints、PerCore 标记和 pass-local cloneable extra info |
| 同上 `ConditionGuard`/`AttributeGuard` | branch 被展平成 guarded sibling，guard 是最近 surviving structural ancestor 的 delta，不是单独 task；clone/substitute 必须同时改 stmt、region、layout/pitch、guard 和 extra info |
| 同上 `CompareIRStructure`、`CalculateIterationCount` | 跨节点按 root-to-node `(stage,index)` 字典序，祖先关系有特殊返回值；flattened iteration 依次组合每级 `indexdiv(loop_var-min,step)` 和 `ceildiv(extent,step)` |
| `auto_schedule/task_annotations.h::TaskMetadata`、`ResolveTaskCost` | latency/ii/core_mask 是结构化 metadata；core_mask 后端生成而非 `T.Task` 公共参数。其他 compiler key 必须保留 |
| `auto_schedule/kernel_rewriter.h::RewriteTilelangKernels` | 各 pass 只改写 `tilelang_root` kernel segment，携带 outer context/outer sid；多 kernel 仍按 program order 各自调度，不跨 kernel 重排 |

`TaskNode.outer_ctx` 和 `ControlNode.loop_body_ctx_` 不是可选优化：它们保存 loop range、非空条件、Bind、assume、branch guard，供区域冲突、guard 等价、动态 extent 和 core availability 使用。含 condition 的 Bind 只能在其定义域内作为事实，mutable read 必须 freshen，防止把不同迭代的快照误当相同常量。

### 5.2 相关 pass 全清单

| 顺序/路径与符号 | 功能及不可遗漏的边界 |
|---|---|
| `rewrite_dual_copy.cc::RewriteDualCopy` | 在 layout/OOB/scheduling 前将 GM↔UB、UB→L1 software dual copy 改为复用 enclosing cthread sid 的 ordinary copy；auto-mixed kernel 可合成 sid，L0C→UB 保持硬件双目的。成本与依赖必须看到改写前后的等价 payload/region |
| `legalize_simd_merging.cc::LegalizeSimdMerging` | 将 MODE_MERGING 的保留目的 lane 语义改成显式 read-modify-write access，防止依赖分析把 read-before-write 当成纯写 |
| `insert_oob_padding.cc::AscendInsertOOBPadding` | clamp DMA OOB tail，并将 GM→L1 padding 变成带精确写区域的 semantic fill；这些 fill 在 AutoSchedule 前进入 task/dependency，后由 LowerTileOp/codegen 落地 |
| `insert_nd2nz.cc::InsertNd2Nz` | 将布局驱动的 UB ND→NZ 与 UB→L1 转换展开成 SIMD scatter 和 MTE post-copy，使调度和成本能看到所有 op；这是 Ascend NZ 专用 rewrite，NV 不得直接套用 |
| `normalize_fractal_storage.cc::NormalizeAscendFractalStorage` | 在调度前规范 Cube fractal storage、view 和 alias，为 pitch/物理 footprint 提供稳定 IR；属于 Ascend 专用物理布局前置步骤 |
| `unroll_loop_skip_vf.cc::UnrollLoopSkipVF` | 只展开用户显式请求的 unroll 且跳过 VF，使 LayoutInference/AutoSchedule 看到展开后的独立 task；普通 serial 和非显式 hint 不提前展开 |
| `normalize_control_flow_for_schedule.cc::ControlFlowNormalizer` | while→带 `synthetic_while` 的 bounded serial For，extent=`max(2,lcm(1..max_nested_num_stages))`，termination 用 `if !cond loop_break`；复杂 if condition 提升为 Bind；buffer-dependent loop min/extent/step 提升为入口 Bind。非 serial/unrolled 和 SBlock 内保持 atomic |
| `normalize_conflict_hints.cc::ConflictHintNormalizer` | 消费 statement-form `tl.conflict_hint`，按 level=-2/-1/>=0 附到 root/all/指定 loop；group 两半必须恰好两次、level/cross 一致、在共同 loop prefix 配对；规范成 `[a,b,cross,is_conflict]` |
| `materialize_schedule_units.cc::TaskNormalizer`、`ScheduleUnitMaterializer`、validators | 自动给 Evaluate/BufferStore/Bind/While/SBlock/非 serial For 包 Task；branches/attrs 展平到 units，空 list 生成 no-op task；未标 stage 全 -1，一旦同级有 manual stage，其余归 0。校验单 pipe/单 core affinity、Task/Stage 嵌套、PerCore candidates/sync/未标 buffer Bind |
| `assign_core.cc::CoreMaskAssigner`、`auto_schedule/subcore.h::SubcoreProjector` | 先按 pipe/destination/scalar memory 赋最宽 core mask，再按 scalar/register/control 输入 fixed-point 收窄；cross-core flag id 在同一 child list 按 pipe 和可用上下文归核。PIPE_S/PIPE_MTE2 在纯 kernel 可用 kernel/core candidate fallback，在 mixed 或无上下文时必须由 PIPE_V/MTE3 或 PIPE_M/MTE1/FIX 消歧；同一 flag scope 不得同时落 Vector/Cube。Subcore projector 复制 sid-varying vars/buffers/layout/version/address/hint，保留 common storage 和 dual destination，记录 `subcore_origin` |
| `annotate_multi_buffer_eligible.cc::WriteFirstClassifier`、`MultiBufferOwnerPlanner` | 四态 untouched/maybe-write-first/write-first/read-first；按 manual stage 排序并以 guard implication 判断写入覆盖。选择覆盖所有 ordinary access 的最深 disjoint loop frontier，partial descendant coverage 提升到 write-first ancestor；owner 外仅允许独立 broadcast fill |
| `auto_schedule.cc::ScheduleBuilder`、`ScheduleSingleKernel`、`AutoSchedule` | 递归先子后父，root 直线/每个 Control 循环分别求解；设置 UB/L1/L0/SS/per-AIV memory groups、manual versions/conflicts；校验 task costs、L0 data/SF common owner、buffer address fixed version；写 selected versions。未应用的 >1 override 由 `WarnOnUnappliedBufferVersionOverrides` 警告，placement 则严格要求 selected==requested；bound data/SF 没有 common owner 是错误 |
| `prepare_multi_buffer.cc::MultiBufferPlanBuilder`、`AnnotateStorageEpochs`、`MaterializeCounterProtocol` | 校验 disjoint owners、coverage、同一 data/SF owner；选 iteration/counter clock，合并兼容 counter groups，规范 epoch guards，插入 counter init/advance。`needs_counter = 多owner OR 非恒真active_guard OR 非仿射路径`；显式 counter 强制 counter，auto 在 needs_counter 且（counter可用或无法回退单owner iteration）时选 counter。显式 iteration 仅单 owner；counter 要求同 owner 内单一 access stage，auto 单owner在 counter不可用时才回退 iteration。counter group 还要求相同 ordered owners/stages、等价guard及全协议core可求值；active_guard 在同 owner 多 stage 或其定义不支配首次访问时退化为 true；每 group root初始化一次，在每owner最后一个storage access后按active_guard advance |
| `resolve_core.cc::CoreMaskResolver` | 基于实际 var/register/special-register/control/epoch/counter 消费反向传播 execution demand，处理 projected Cube domains 和 loop-break exit demand；再次 fixed-point，fallback 优先 Vector，若已有 Cube 工作则可选 Cube |
| `insert_sync.cc::SyncSiteRegistry`、`SyncAnalyzer`、`FlagAllocator`、`SyncMaterializer` | 建立 task×path×before/after×core×epoch sites；从依赖和 same-pipe order 求 closure、删冗余 sync、规划 ring/flag；生成 acquire/release/barrier/priming/drain；同时产出 buffer alias map。详见 5.3 |
| `materialize_multi_buffer.cc::VersionedBufferRegistry`、`MultiBufferAccessRewriter` | 按真实 allocation pitch 给多版本 buffer 增加 leading version dim/strides；以 `counter%N` 或 flattened iteration%N 重写 access、regions、guard/layout；broadcast fill 覆盖所有版本；单版本 counter group 只有无实际版本时消除 |
| `lower_scheduled_tir.cc::CoreExtractor`、`PipelineLowerer`、`LowerScheduledKernel` | 按 core 抽取、复制 shared scalar、下推 `loop_var-stage*step` 和边界 guards，扩 extent；conditional scalar Bind 转 local.var，handle Bind 保持作用域；生成 CUBE/VECTOR bodies、必要 cthread，迁移 alias/AIV/address contracts |
| `restore_while_loops.cc::WhileLoopRestorer` | synthetic For 恢复 `while(true)`；若 synthetic var 仍被版本/flag 下标使用，生成 local counter init/increment，不得简单置 0 丢 epoch |
| `normalize_buffer_version.cc::BufferVersionNormalizer` | 将 staged `tl.manual_multi_buffer` 转 scoped compiler-internal `tl.buffer_version`，禁止前端伪造 internal attr |
| `rewrite_buffer_version_layout.cc`、pipeline 中 `RewriteAscendBufferVersionLayout` | 在版本物化和 flatten 前重映射每个物理版本的 layout/offset；UB 版本保持 32B alignment，不能仅改首版 |
| `auto_schedule/multi_buffer.h::MultiBufferPlan`、`EpochDomainRegistry` | 一 storage 一 info（包括单版本），多 storage 可共享 counter；lexical/counter domain 分离，domain projection 由 guard implication 和传递闭包决定，不按数字 ID 猜兼容性 |
| `auto_schedule/buffer_lifetime_analysis.h::IterationOrder`、`CanAliasGenerations` | weak issue edge 与 strong completion edge 分离；非负加权最短路，periodic lifetime 用双向 separating cut 证明，互斥 guard 只有 iteration-invariant 时可跨代使用 |
| `auto_schedule/buffer_alias_analysis.h::BufferAliasAnalyzerImpl` | physical may/must access、opaque escape、live-in/out/through、nested/single-trip/composite summary、guard 覆盖与 generation lifetime 共同决定正 alias；任何未知均不复用 |
| `infer_buffer_aliases.cc::SyncAwareLinearizer`、`HappensBeforeGraph`、`BufferReuseAnalysis` | 非自动调度的手工 IR 使用另一套 event/happens-before 分析；per-pipe program order、flag/cross-core handshake、loop-carried 和全 access-pair lifetime separation；同一 pipelined loop 共驻 buffer 禁止复用 |
| `merge_ub_common.h`、`merge_ub_allocations.cc::BuildAliasMemoryPlan`、`BuildPlacedMemoryPlan` | 对称 alias contract 是许可而非推断；默认可复用，disable 后顺序分配；deterministic DSATUR greedy clique、按 alignment/size 布局；exact placement 先保留区间，其余 first-fit，检查所有物理版本/padding/SIMT 32KiB/private AIV |
| `rewrite_flag_to_buf.cc::FlagUsageCollector`、`KnapsackKeep`、`FlagToBufRewriter` | 每个 hard_event 8 slots；只合并 overlapping/contained ranges，adjacent-disjoint 仍分块；稀疏超界但总量<=8 只 compact，超载按不可分割 range block 做 capacity-8 0/1 knapsack，其他 spill 到共享 32-slot get/rls mutex，保留 gemm_l1 两槽 |
| `auto_schedule/memory_detector.h::MemoryAccessDetector` | logical/physical 两种模式，read/write/may/must/opaque footprints、loop-domain relaxation、L0 pitch、pointer/index/layout metadata reads；是依赖和 alias 的共同基础，不可只识别 BufferLoad/Store |

### 5.3 依赖、同步、core、alias 的精确协作

**Resource/access 分析**（`task_analysis.h`、`ascend_pipe.h`）：

- copy 按 scope 映射 MTE1/2/3/Fixpipe/Vector；fill L1→MTE2，其他非 VF fill→Scalar；GEMM/blockscaled→Cube，L1 fused GEMM→MTE1|Cube；VF blocks→Vector；sync calls 对 task pipe 返回 0，由 special register 表达目标 pipe，避免污染 PerCore 数据 pipe。
- 无专用 pipe 的 task 补 Scalar；`T.Task` 将 special-write 也计入 issue-pipe 校验，防止一个原子 task 跨 pipe 或同时有 AIC/AIV HBM affinity。
- HBM：GM↔UB 为 AIV（store 另加 Store），GM→L1 为 AIC，L0C→GM 为 AIC+Store；UB→GM 与 L0C→GM 共享 Store，即使 issue core/pipe 不同仍有 resource dependency。
- `CoreMaskAvailability` 对 scalar/register writer、shared/L1/L0 local memory、region index、guard、loop bound、sync id 和特殊 register 求可用 core；scalar task 还检查自己的读取内存，不允许同时访问 Vector UB 与 Cube L1/L0。bit 编码为 unassigned=0、AIV0=1、Cube=2、AIV1=4、AIV 集合=5、全部=7。

**Storage/capacity 约束**（`auto_schedule.cc::GetSharedMemoryLimit`、`Z3SchedulePythonLoop`）：

- UB 基准为 248KiB；存在 `SIMT_VF` 时再扣 32KiB context，实际为 216KiB。L1=512KiB，L0A=L0B=64KiB，L0C=256KiB，SS=3KiB。`shared.ss` 是 AIC+两个 AIV 的一个共享池。
- 存在 `AivStorageMap` 时建立 `shared.aiv0/1` 组，common UB 同时计入 `shared`、`shared.aiv0`、`shared.aiv1` 三组；projected private UB 只计对应 AIV 组；无该 map 时 common UB 仅计 `shared`。non-eligible buffer 按单份 size 扣容量，fixed-version override 按 `size*versions` 扣容量，其余进入 solver `buf_b` 变量和 memory group。固定分配的总和不是最终物理容量证明，因为下游可能复用。
- 仅当某组含自动版本变量时，才预检扣款后容量非负及“每 buffer 至少一个版本”的 min_usage；非静态 buffer size 在源码中退化为 0，因此该预检也不能单独证明物理可放置。`unlimit_memory` 将容量设为 `1<<40`，公共 API 只允许第 5.4 节列出的五种 scope。

**同步点与事件顺序**（`insert_sync.cc`）：

- `SyncInsertionSite` 包含 path begin/end、before/after、core、pipe、epoch domain、guard。向 loop 内放置 acquire 时必须支配可能为 0 次的 consumer loop；不能穿越 loop_break 或未证明非空的层级。
- owner exclusion 在最终 `(stage,index)` 顺序下拆成双向：`forward_distance=stage(first)-stage(second)`，反向为 `1-forward_distance`。完整 counter ring dependency 必须由相同 counter channel 在两个 owner 的 local closure 覆盖，否则保留 lexical handshake。
- `DepClosure` 饱和正向距离并删已被覆盖 sync；same-pipe issue distance 为 `stage(a)-stage(b)+(a physically after b ? 1:0)` 后截到非负。只有 S/V/M 能把 same-pipe order 当 completion；MTE/FIX issue-order 仍需 barrier/flag。
- same core+same pipe：S/V/M 省同步，MTE/FIX 在 consumer 前插 `PipeBarrier`。跨 core 或跨 pipe：producer 后 set、consumer 前 wait。`MTE2_M`/`M_MTE2`/`MTE3_M`/`M_MTE3` hard event 分别规范到 MTE1 对应事件。
- `unit_flag_ctrl!=0` 且 hard event 含 FIX 时，same-core 由硬件 unit flag 协议承担，跳过自动 same-core sync；caller-managed pairing 不得重复生成。
- flag event id：`num_versions==1` 为 base，否则 `iteration%num_versions+base`。iteration 用 lexical 或 counter clock；counter channel signature 包括 counter group、双向 core/pipe、versions、distance 和 owner 内 ordinal。
- ring 优化：loop trip count不超过版本数时可省 reverse flag；否则用 reverse closure 缩版本并诊断 malformed ring/cycle。跨 iteration flag 需 prefix 初始 token 和 suffix drain，cross-core mode 4 set/wait 按 AIV bank 生成 sid*16 偏移。
- 禁止自动生成 AIV0↔AIV1 依赖同步（包括 mode-1/AIC relay 替代方案）；要求 disjoint regions 或真实 AIC dependency。显式 flag 的 dynamic id 必须能由 loop/Bind/guard/assume 限定区间，否则需要自动 flag 时 fail。
- intra-core flag 每 domain 8 slot，cross-core 每 AIV bank 16 slot；显式 ranges 在所有模式下都保守占用两个 bank，单个 id 按 modulo 16 规范化（负值加回正数），span>=15（覆盖至少16个id）直接占满，auto cross-core 再找所需 versions 的连续空档。只有需要自动生成 cross-core flag 时，unbounded explicit id 才硬错误。auto flags 按 versions 降序分配，只有 lifetime/reuse proof 才共享；multi-owner counter channel 必须独占，不能因普通 reuse 合并。超 intra 8 后由 `RewriteFlagToBuf` 后处理。

**Alias/lifetime**：

- `CanAliasGenerations` 要求所有 generation pair 可分离；非 periodic 恰好一个方向的 all-end-before-begin 成立，periodic 用 `forward+reverse<=1` 的 separating cut。live-through、live-in/out 交叉、opaque access、无法证明 guard 都禁止 alias。
- must-write 与 may-write 不同：condition/while/predicated store/access_ptr 不产生完整定义；loop_break 使 atomic task 内 must-write 全失效；`DetectIterMap` surjective 只用于证明 dense write，不能把 runtime index 的范围当全覆盖。
- automatic scheduled path 的 alias map 在 `InsertSync` 写入；`LowerScheduledTIR` 移到 AttrStmt；`MergeUBAllocations` 消费后移除。manual path 由 `InferBufferAliases` 生成。`disable_shared_memory_reuse` 时仍保留“存在但为空”的合法 contract，继续同步和顺序分配。
- `RemapBufferAliasMap` 在多个旧 storage 合成一个时，只有所有旧组合均可 alias 才保留新边；冲突提示不能替代 lifetime 许可或 address overlap 检查。

### 5.4 公开调度 annotations 与相邻 API

| API/源位置 | 公开语义和校验 |
|---|---|
| `tilelang/ascend/language/schedule_hint.py::Task` | 仅有 `latency`、`ii` 两个公共参数；完整 task 串行、单 pipe/单 core affinity，内部不自动插 sync；`latency>=ii`。不是 enclosing loop II override，也不直接选择 core |
| 同文件 `Stage` | 非负 Python int，任一同级出现即该 child list manual，未标 siblings 归 0；每 pipe 保 source order，不同 pipe 可重排；非零要求 enclosing `enable_offset=True`。必须包完整 task，不可嵌套或放进 Task/PerCore/VF/SBlock/parallel loop |
| 同文件 `PerCoreTask` | 一个逻辑 per-core task 的 candidate sites；显式/推断 `T.Task` 可混合。每次动态调用、每个 block/core 必须恰好一个 candidate，single pipe/相同逻辑角色；完整 dispatch 必须在同一 scope，禁止 sibling scopes 和嵌套 |
| 同文件 `assume_no_conflict`/`assume_conflict` | `a,b` 为精确 region 或 bare buffer；b 默认 a；level=None/-1/非负；cross=None/True/False；group 字符串在两个不同 scope 各写一半。只改变编译期依赖，runtime no-op；forced conflict 不排除 read-read |
| `language/annotations.py::annotate_buffer_versions` | value 可为 1、N>=2、`(N,mode)` 或仅 mode；mode=auto/iteration/counter。bare `{buf:1}` 关闭自动 eligibility，`{buf:(1,mode)}` 保留单版本 ring 语义；alias 必须 count/mode 一致。AutoSchedule 消耗 override 并写 selected count |
| 同文件 `annotate_manual_multi_buffer` | 声明用户已手写 ring index，仅用于依赖距离和 sync，不 reshape/reallocate，不要求 version 在 dim 0。bare buffer 从静态 `shape[0]` 得 N，dict 可显式 N>=1，支持 flattened/subrange ring |
| 同文件 `annotate_unlimit_memory` | 调度容量绕过，公共 scopes 为 `shared`、`shared.l1`、`shared.l0c`、`shared.l0a`、`shared.l0b`；`shared` 同时解除 per-AIV limits。不是物理分配成功证明 |
| 同文件 `annotate_buffer_addresses` | UB/L1/L0A/L0B/L0C/SS exact byte address，非负且分别对齐 32/512/512/512/1024/32；alias 同地址；必须同时显式给版本数（包括 1），manual metadata 或 mode-only 不够。partial maps 可，overlap 永不允许，L0 SF 随 data |
| `tilelang/language/loop.py::Pipelined`、`Persistent`；loop annotations | `num_stages` 是自动版本搜索上界；`annotations={"enable_offset":True}` 允许 stage offset。Ascend manual scheduler 用 `T.Stage`，不能用通用 Pipelined `order/stage` 列表混淆；`T.Stage` 的值本身不被 `num_stages` 限制，manual 未给 num_stages 时才取 max(stage)+1。`T.Persistent` 的 grid/tile 遍历与 Z3 调度是两层 |
| `tilelang/ascend/language/frame.py::SimtVF`、`SimdVF` | `latency=0` 走模型，正数写 `tl.vf_latency` 覆盖 VALU 部分；SIMT threads 1–3 维且全正。VF 是原子 task boundary |
| `tilelang/ascend/language/kernel.py::Kernel`、`MixedKernel` | Kernel 是 1D NPU grid、无 kernel-scope SIMT threads；MixedKernel 返回 `(bx,sid)`，sids=1/2，影响 vector count、projection、同步 banks 和 per-AIV storage |
| `language/copy_op.py::copy`、`dual_copy` | dual M/N split 支持 GM↔UB、UB→L1、L0C→UB；rank-1 也要求 2:1；L0C→UB 是硬件双目的，其他由 `RewriteDualCopy` 变 per-AIV ordinary copy。`unit_flag_ctrl` 控制 FIX unit flag，不等同通用 scheduler flag |
| `language/mode.py` 及 pad/loop/sync API | `set_hf32_mode`、`set_mmad_direction`、`set_atomic(_none)`、copy pad value、`loop_break` 都通过 SpecialRegister 参与顺序；显式 `ascend_set_flag/wait_flag`、`ascend_cross_core_*`、`ascend_pipe_barrier` 的 IDs/pipe 必须纳入 core inference 和 reservation |
| `tilelang/ascend/language/tile_schedule.py::Ascend*TileScheduler` | persistent GEMM tile 选择、group/swizzle/state/assume 是输入 IR 生成器，不是 Z3 cost model；其 while/state 会经 NormalizeControlFlow、task analysis 和 scheduler，不能在 NV 迁移时误当成另一套调度器 |

### 5.5 源实现中必须显式保留或重新评审的边界

1. **Write-first 不是全覆盖证明**：`AnnotateMultiBufferEligible` 源码明确是 storage-granular heuristic；任一 region 写可使 storage write-first，`AssumeForExecutes` 还假定 For 至少执行一次且 maybe-write 最终发生。携带未覆写 region 的 kernel 必须用 `{buf:1}` 或明确 owner，不得把 annotation 当作内存正确性证明。
2. **容量不是全局最终证明**：solver 预检只覆盖自动版本变量的最小需求；fixed allocations 可在下游复用。disjoint owner 合并取 max 后不重新检查全局总量，源码接受可能超池并留待下游/runtime。
3. **动态成本是上界/heuristic 混合**：region 会利用 analyzer/static shape，最后 extent=1；动态 trip fallback 100；未知带宽/吞吐贡献 0；这些只影响估算质量，不得反向用于 correctness proof。
4. **fallback 文案和行为不同**：循环缺 FFI 是 fatal；自动无可行 II 才串行 fallback；直线日志称 topological sort 但实际保持 scheduled/source order。迁移文章必须按代码描述。
5. **Z3 unknown 未独立暴露**：rlimit 导致的 unknown 在二分中按 infeasible 处理，可能扩大 II 或最终 fallback。不能把返回结果都称为数学最优而不记录 solver verdict。
6. **保守 alias 是安全方向**：opaque pointer、不同 view、无法证明 pitch/guard/lifetime 时放弃复用；`assume_no_conflict` 只影响依赖，不会把不安全 alias 变成合法。
7. **性能数字不可跨架构复制**：`test_estimate_latency.py` 中 CANN 9.2/A5/npusim 的精确周期只证明源模型合同。NV 需要独立校准和容差/数值证据，不能把这些常数当成 CUDA 事实。

## 6. Feature parity 总表（当前源码、host 证据与端到端边界）

本表保留原 84 个 ID；源路径和完整源行为见第 1–5 节，表内不塞段落级长文。详细证据和硬缺口解释见 `docs/design/cuda_z3_parity_evidence.md`。所有路径缩写先展开为仓库相对真实路径：

- NV 实现：`CP=tilelang/cuda/pipeline.py`，`ZS=tilelang/cuda/transform/z3_scheduler.py`，`ZP=src/transform/z3_pipeline_planning.cc`，`ZT=src/transform/z3_task_analysis.cc`，`IP=src/transform/inject_pipeline.cc`，`PP=src/transform/pipeline_planning.cc`，`LM=tilelang/cuda/latency_model.py`，`CM=tilelang/cuda/cost_model.py`，`TCO=tilelang/cuda/task_cost.py`，`AN=tilelang/language/annotations.py`，`GS=tilelang/autotuner/guided.py`，`GT=tilelang/autotuner/guided_tuner.py`，`SP=tilelang/autotuner/space.py`，`BV=src/transform/inject_pipeline.cc`，`MA=src/transform/merge_shared_memory_allocations.cc::MergeSharedMemoryAllocations`，`SR=src/transform/storage_rewrite.cc::StorageRewrite`。
- Host 测试：`TS=testing/python/transform/test_tilelang_cuda_z3_scheduler.py`，`TP=testing/python/transform/test_tilelang_transform_z3_pipeline_planning.py`，`TT=testing/python/transform/test_tilelang_z3_task_analysis.py`，`TL=testing/python/cuda/test_cuda_latency_model.py`，`TM=testing/python/cuda/test_cuda_cost_model.py`，`TB=testing/python/transform/test_tilelang_transform_pipeline_buffer_versions.py`，`TI=testing/python/transform/test_tilelang_transform_Inject_software_pipeline.py`，`TQ=testing/python/transform/test_tilelang_transform_pipeline_planning.py`，`TD=testing/python/language/test_tilelang_cuda_z3_pipeline.py`，`TG=testing/python/autotune/test_guided_search.py`，`TU=testing/python/autotune/test_guided_tuner.py`，`TCAL=testing/python/cuda/test_calibration_schema.py`。
- 已提交日志：`E1=evidence/G1/solver-tests.txt`（49 passed），`E2L=evidence/G2/latency-model-tests.txt`（28 passed），`E2C=evidence/G2/cost-model-tests.txt`（31 passed），`E2CAL=evidence/G2/calibration-tests.txt`（32 passed + 37 subtests），`E3=evidence/G3/main-pipeline-host-tests-final.txt`（148 host passed），`E3D=evidence/G3/device-integration-test-retry.txt`（1 device passed），`E3DF=evidence/G3/device-integration-test.txt`（保留的 1 次 multi-async-stage 失败），`E4C=evidence/G4/capacity-unknown-tests.txt`（85 passed），`E4A=evidence/G4/full-autotune-tests.txt`（91 passed + 2 skipped）。`CAL=evidence/G2/calibration-summary.json` 与 `evidence/G2/calibration-final/calibration.txt`（11 个 complete measured benchmarks）。

“无专项”表示没有可接受的本条直接 oracle，不能用联合套件总数反推覆盖。端到端栏中的“A+B”只指 `TD::test_default_off_and_opt_in_real_solver_and_numerics` 的 SM120 显式 annotation、default-off/opt-in、`torch.testing.assert_close(rtol=0, atol=0)` 用例；“校准子集”只指独立 microbenchmark 率，不是 task latency 或 whole-kernel ms 的真机精度验证。

| ID | 源功能（原条目保留） | NV 当前源码位置 | Host 测试 / 已提交证据 | 当前状态 | 端到端状态与边界 |
|---|---|---|---|---|---|
| PIPE-001 | backend 选择、FFI 注册、完整 pipeline 调用链 | `CP::CUDAPassPipelineBody[Prologue]`；`ZP::Z3PipelinePlanning`；`ZT::InferZ3PipelineProblem`；`ZS` 两个 CUDA FFI | `TP::test_complete_problem_projection_and_stock_pipeline_planning`；`TT::test_main_cuda_pipeline_infers_before_z3_planning`；`TD`；`E3`、`E3D` | DEVICE_VALIDATED_SUBSET | 仅 SM120 显式 annotation A+B off/on exact；不代表任意 IR、GDN 或经典验收 |
| PIPE-002 | 默认开关、非自动 alias 路径、reuse 独立开关与 scheduler 互斥 | `CP::{_Z3_PIPELINE_CONFIG,_Z3_TASK_ANALYSIS_CONFIG,_AUTO_WS_CONFIG}` 与 WS skip/guard；`CP::MergeSharedMemoryAllocations`；`ZP` default-off | `TP::{test_pass_config_defaults_off_and_preserves_annotated_loop,test_no_annotation_is_strict_noop_even_when_enabled,test_auto_warp_specialization_conflict_is_explicit,test_existing_manual_warp_specialization_is_not_silently_skipped}`；`E3` | PARTIAL | opt-in ownership/default-off 已有 host+设备子集；Ascend 非自动 alias contract 未迁移，复用开关不等价 |
| Z3-001 | Python/C++ FFI、tuple 转换、锁与 private Context | `ZS::{z3_schedule_ffi,z3_schedule_loop_ffi,_z3_lock}`；两个入口内 `z3.Context()`；`ZP` loop FFI caller | `TS::{test_ffi_registration_conversion_and_return_shapes,test_private_context_loop_seed_and_rlimit,test_repeated_and_threaded_solves_are_deterministic}`；`E1` | IMPLEMENTED_HOST_TESTED | solver/loop adapter host 子集；无独立 straight-line C++ pipeline caller |
| Z3-002 | 直线 start 非负与最小 makespan | `ZS::z3_schedule_python`（Optimize/makespan） | `TS::{test_straight_line_minimizes_makespan_and_stable_tie,test_straight_line_data_latency_and_resource_exclusion}`；`E1` | IMPLEMENTED_HOST_TESTED | 仅 solver 直测；未接入 CUDA 直线 task 自动构建/执行链 |
| Z3-003 | Let/buffer/special/嵌套 latency data edges | `ZS` straight/loop data constraints；`ZT::{TaskBufferRegionCollector,InferDataDependencies}` | `TS::test_straight_line_data_latency_and_resource_exclusion`；`TT::{test_raw_war_waw_fixed_dependencies_follow_source_order,test_distance_two_conflict_is_not_dropped_by_distance_one_reasoning}`；`E1`、`E3` | PARTIAL | buffer RAW/WAR/WAW 与白名单仿射子集可用；Let closure、special register、嵌套 timestamp 未实现 |
| Z3-004 | II resource exclusion、CUDA resource mask 与 load/store 争用 | `ZS` resource pairs；`LM::Resource` 七位 mask；`ZT::InferResourcePairs` | `TS::test_loop_resource_forward_and_wraparound_exclusion`；`TT::test_resource_pairs_are_derived_from_effective_masks`；`TD`；`E1`、`E3`、`E3D` | DEVICE_VALIDATED_SUBSET | A+B 仅验证一对显式 copy resource exclusion；不证明全部 mask/warp role/HBM 场景 |
| Z3-005 | 直线 owner lifetime exclusion 与 per-pipe source order | `ZS::z3_schedule_python`（owner/pipe constraints）；`ZP::PlanLoop` owner guard | `TS::test_straight_line_owner_exclusion_and_pipe_order`；`TP::test_unexpressible_dependencies_fail_before_solver`；`E1`、`E3` | IMPLEMENTED_HOST_TESTED | **仅 solver 约束完成**；CUDA owner runtime/ring/sync 未实现，adapter 明确拒绝，端到端未闭环 |
| Z3-006 | `start=kII+r`、k/r 域与归一化 | `ZS::z3_schedule_loop_python`；`ZP::{raw_stages,residues,min_stage}` | `TS::{test_loop_start_is_k_ii_plus_r,test_loop_manual_stages_are_fixed_then_normalized}`；`TP::test_manual_offset_and_minimum_stage_normalization`；`E1`、`E3`、`E3D` | DEVICE_VALIDATED_SUBSET | A+B 覆盖一种 modulo 投影；CUDA IR 只保留 stage/order，不保留绝对 start |
| Z3-007 | 固定 distance、版本变量 distance 与 ring recurrence | `ZS` fixed/negative-distance constraints；`ZT::DistinctIterationsDisjoint`；`BV` | `TS::{test_loop_fixed_cross_iteration_distance,test_loop_negative_distance_recalculates_and_reads_buffer_versions}`；`TT::test_distance_two_conflict_is_not_dropped_by_distance_one_reasoning`（正/负斜率）；`TB`；`E1`、`E3` | IMPLEMENTED_HOST_TESTED | solver+host IR 子集；自动分析不是任意 IR 通用证明，未做 device ring 矩阵 |
| Z3-008 | modulo resource 正向与回绕 II 约束 | `ZS::z3_schedule_loop_python`（resource 四式） | `TS::test_loop_resource_forward_and_wraparound_exclusion`；`E1` | IMPLEMENTED_HOST_TESTED | solver 直测完成；runtime 仅另见 PIPE/PASS 的有限 pipeline 子集 |
| Z3-009 | owner phase、反向 hand-off 与 wrap（不用 k） | `ZS::z3_schedule_loop_python`（owner 四式）；`ZP::PlanLoop` owner guard | `TS::test_loop_owner_forward_and_wraparound_exclusion`；`TP::test_unexpressible_dependencies_fail_before_solver`；`E1`、`E3` | IMPLEMENTED_HOST_TESTED | **仅 solver 约束完成**；owner ring/wrap runtime 未实现且 adapter 拒绝，禁止端到端核销 |
| Z3-010 | stage-order `k_u<=k_w` 与跨 stage Let clone | `ZS` stage-order；`ZP` forward/result check；`IP::ValidateScheduledBindDependencies` | `TS::test_loop_stage_order_changes_minimum_ii`；`TP::{test_stage_order_is_forwarded_and_checked_after_projection,test_stage_order_rejects_a_solver_result_that_violates_it,test_stage_order_does_not_bypass_scheduled_bind_validation}`；`E1`、`E3` | PARTIAL | 支持无 Let clone 子集并复核 solver；scheduled Bind 跨 stage 仍拒绝，general Let clone 是硬缺口 |
| Z3-011 | `enable_offset=False` 同 pipe `<II` 窗口或允许 offset | `ZS` enable-offset 分支；`AN::z3_pipeline_annotations` | `TS::test_loop_enable_offset_controls_cross_stage_resource_spread`；`TP::test_manual_offset_and_minimum_stage_normalization`；`E1`、`E3` | IMPLEMENTED_HOST_TESTED | solver 与 host 投影已测；设备仅覆盖默认 false 的 A+B，不覆盖任意 offset |
| Z3-012 | manual fixed k、pipe source order、num_stages 与非法 manual | `ZS` manual constraints；`ZP` manual validation/projection；`AN` manual schema | `TS::{test_loop_manual_stages_are_fixed_then_normalized,test_loop_manual_pipe_order_infeasible_does_not_fallback}`；`TP::{test_manual_offset_and_minimum_stage_normalization,test_manual_schedule_conflicts_with_handwritten_order_stage}`；`E1`、`E3` | PARTIAL | fixed stage/order 可用；`num_stages` 仍必须显式为正，未实现 max(manual stage)+1 默认及 `T.Stage` API |
| Z3-013 | 版本上下界、scope 容量、扣款与 sub-byte size | `ZS` buffer/memory-group constraints；`ZT::{InferBuffers,MemoryCapacities}`；`TCO::memory_group_capacities` | `TS::test_loop_memory_group_capacity`；`TT::{test_non_static_versioned_buffer_size_is_an_error_not_zero,test_hardware_capacity_requires_explicit_optin_field}`；`E1`、`E3` | PARTIAL | shared 单组与静态 size 已测；非静态报错非 0；fixed/non-eligible 扣款、reg/tmem 组和最终容量证明未闭环 |
| Z3-014 | II 下/上界与 sat 单调二分最小可行 II | `ZS::z3_schedule_loop_python`（binary search） | `TS::{test_loop_fixed_cross_iteration_distance,test_loop_resource_forward_and_wraparound_exclusion,test_loop_stage_order_changes_minimum_ii}` 的独立 II oracle；`E1` | IMPLEMENTED_HOST_TESTED | solver 算法 host 已测；不表示每次返回都带 SAT/最优性审计元数据 |
| Z3-015 | start/stage/phys-time 重建、control latency 与 Let clones | `ZS` model extraction；`ZP::{raw_stages,residues,physical_order,cuda_depth}` | `TP::{test_cross_stage_tie_uses_larger_stage_then_source_index_and_inverse_order,test_audit_ii_and_starts_preserve_int64_values,test_flat_non_identity_schedule_reorders_statements_and_uses_identity_order}`；`E3` | PARTIAL | stage/order/int64/flat 子集已测；ControlNode overall latency、general clone 与绝对时间 IR 未实现 |
| Z3-016 | version 两来源、重算、group max 与不再钳制 | `ZS::recalculate_buffer_versions`；`ZP::buffer_version_map`；`BV` exact override | `TS::test_loop_negative_distance_recalculates_and_reads_buffer_versions`；`TP::test_output_depth_covers_solver_stages_and_buffer_versions`；`TB::test_explicit_buffer_version_override`；`E1`、`E3` | PARTIAL | solver 两种来源和 downstream exact count 已测；multi-owner group max/容量复查未实现 |
| Z3-017 | private Context、seed、稳定 tie 与重复 compile 隔离 | `ZS::{_z3_lock,Context,seed,sorted}`；`ZP::physical_order` | `TS::{test_private_context_loop_seed_and_rlimit,test_repeated_and_threaded_solves_are_deterministic}`；`TP::test_cross_stage_tie_uses_larger_stage_then_source_index_and_inverse_order`；`E1`、`E3` | PARTIAL | 重复调用/线程与 tie 已测；多进程及重复完整 compile 的确定性证据未闭环 |
| Z3-018 | rlimit/unknown、UNSAT 串行 fallback、manual/fatal 与 adapter 传播 | `ZS::{_check_feasibility,Z3ScheduleUnknownError,Z3ScheduleInfeasibleError}`；`ZP` 无 catch/stock fallback；`SP::Indeterminate`；`GT` compile errors | `TS::{test_straight_line_unknown_is_not_reported_as_unsat,test_loop_automatic_serial_fallback_only_after_unsat_exhaustion,test_loop_mocked_unknown_never_falls_back,test_loop_unknown_after_sat_aborts_instead_of_pruning}`；`TP::test_unsat_and_unknown_propagate_without_fallback`；`TG::test_external_indeterminate_skips_solver_and_preserves_denominators`；`TU::test_all_compile_failures_still_raise_and_leave_error_summary`；`E1`、`E3`、`E4A` | IMPLEMENTED_HOST_TESTED | unknown/timeout/external Indeterminate 均非 UNSAT；自动串行仅全 UNSAT 后；compiler failure 非 UNSAT；三元素返回无 verdict，device 失败审计未闭环 |
| Z3-019 | verbose 日志、编号补洞、JSON/PNG/PDF 与非致命 artifact 错误 | `ZS::{_find_next_schedule_number,_write_verbose_artifacts,_save_verbose_artifacts}` | `TS::{test_verbose_writes_json_in_tmp_path,test_verbose_writes_png_and_pdf_in_tmp_path,test_debug_failure_does_not_change_schedule}`；`E1`；`evidence/G1/visual-review.json` | IMPLEMENTED_HOST_TESTED | host artifact 子集已测；artifact 成功不证明调度数值或 device 正确性 |
| LAT-001 | task feature 提取与 atomic boundary | `ZT::{NativeTaskFeatures,TaskFeatureExtractor}`；`LM::TaskFeatures` | `TT::test_copy_gemm_parallel_vector_and_scalar_features_and_resources`；`E3` | PARTIAL | copy/dense GEMM/vector/scalar 白名单可用；无 Ascend SBlock 等价 atomic 全边界，未知 tile op 不冒充 dense |
| LAT-002 | mode state、动态 region/trip bounds 与 payload | `LM::{Extent,_resolve_extent}`；`ZT::{StaticNonnegative,HandleCopy}` | `TL::test_dynamic_extent_upper_bound_and_unknown_reliability`；`TT::test_non_static_versioned_buffer_size_is_an_error_not_zero`；`E2L`、`E3` | PARTIAL | Python 支持显式 upper-bound；自动 pass 主要要求静态，HF32/TF32 数据流与 dual payload 未迁移 |
| LAT-003 | latency/II 独立 override、校验与 metadata 保留 | `LM::estimate_task` override/final validation；`ZT::InferLoop` 只补缺失字段；`TCO::estimate_task` | `TL::test_latency_and_ii_overrides_are_independent_and_validated`；`TT::test_missing_ii_is_filled_without_overwriting_explicit_fields`；`E2L`、`E3` | IMPLEMENTED_HOST_TESTED | annotation/task-cost host 子集完成；不是公共 `T.Task` 全 API，也不证明估计值等于硬件 latency |
| LAT-004 | Tensor Core shape/dtype、completion 与 II max | `LM::{_mma_atom,_supported_acc,estimate_task}`；`ZT::HandleGemm` | `TL::{test_sm120_bf16_mma_sync_atom_rounding_and_gemm_ii,test_sm120_tf32_k_atom_is_distinct_from_generic_mma_sync,test_multiple_gemms_sum_completion_but_take_max_ii}`；`E2L`；`CAL` | DEVICE_VALIDATED_SUBSET | SM120 FP16/BF16/TF32 mma.sync 率有独立测量；未验证所有 task 绝对 cycles，WGMMA/TCGEN05 率未闭环 |
| LAT-005 | blockscaled/MX shape、throughput 与 scale identity | `LM::_mma_atom` 仅含 `e2m1` 通用 atom/fallback；`ZT::HandleGemm` 只识别 dense `GemmNode` | 无 blockscaled 专项；`TL::test_unsupported_gemm_dtype_is_explicit_fallback` 仅覆盖显式 fallback；`E2L` | PARTIAL | 只有通用 dtype/unknown 路径；无 MX min-MN、scale-factor feature/binding 或实测 throughput |
| LAT-006 | vector op/loop/branch/special/unknown 与带宽模型 | `LM::{VectorOps,estimate_task}`；`ZT::{CountOperation,CountUnknown}` | `TL::{test_vector_operation_sfu_bandwidth_and_resource_max,test_zero_vector_bandwidth_is_not_free}`；`TT::test_copy_gemm_parallel_vector_and_scalar_features_and_resources`；`E2L`、`E3` | PARTIAL | 静态表达式与 unknown 非免费路径已测；动态 loop-invariant 工作量和精确 branch 组合未迁移 |
| LAT-007 | region latency override 与 scalar 成本 | `LM::{estimate_task,TaskFeatures}` 仅 whole-task override 与 scalar term | `TL::{test_scalar_and_mixed_compute_use_resource_max,test_latency_and_ii_overrides_are_independent_and_validated}`；`E2L` | PARTIAL | scalar/whole-task override 已测；无 VALU-only 或 warp-region latency annotation |
| LAT-008 | memory completion、descriptor 与 per-path II | `LM::estimate_task` memory loop；`LM::MemoryPathParams`；`TCO::estimate_task` | `TL::{test_memory_completion_and_independent_ii_formula,test_memory_completion_sums_accesses_while_ii_takes_path_max,test_unknown_memory_path_has_nonfree_explicit_fallback}`；`E2L` | PARTIAL | 通用 global/shared/register/tmem path 已测；TMA/cp.async 独立 task descriptor 校准未完成 |
| LAT-009 | coalesced/strided geometry、transaction width 与 lanes | `LM::{_geometry,estimate_task}`；`ZT::HandleCopy` 仅显式 `coalesced_width` | `TL::{test_strided_geometry_uses_transaction_width,test_unknown_geometry_and_width_are_marked}`；`TT::test_copy_gemm_parallel_vector_and_scalar_features_and_resources`；`E2L`、`E3` | PARTIAL | 几何/unknown 公式已测；自动 coalescing/layout proof、per-warp lanes 与 dual-warp 等价未实现 |
| LAT-010 | operand/scale-factor copy 与 issue+payload | `LM::{_canonical_scope,MemoryPathParams,estimate_task}` 的通用 tmem/register path | `TL::test_scope_alias_normalizes_memory_path_and_provenance`；`E2L`；无 SF 专项 | PARTIAL | 仅通用 issue/payload 可复用；无 L0/SF 对应物，也无 CUDA blockscale storage-group 成本闭环 |
| LAT-011 | accumulator drain、destination payload 与 row width | `LM` 通用 register/tmem→shared/global memory path | `TL::{test_memory_completion_and_independent_ii_formula,test_scope_alias_normalizes_memory_path_and_provenance}`；`E2L`；无 drain 专项 | PARTIAL | 通用 store/drain 路径可算；无专门 accumulator completion、dual row-width 或 SM 分层校准 |
| LAT-012 | ND2NZ scatter profile 与 post-copy MTE 成本 | 无 NV ND2NZ 对应实现；`CP` 不运行 Ascend `InsertNd2Nz`，不得伪造同名 pass | 无等价 parity 测试；`LAT-008/009` 仅独立验证通用 CUDA copy，不是 NZ 替代证明 | NOT_APPLICABLE_HARDWARE | Ascend ND/NZ fractal 指令与 D-group 无直接 NV 对应；CUDA swizzle/layout 须另建条目验证 |
| LAT-013 | serial/repeat/branch/while 与 atomic leaf 成本递归 | `LM::{compose_serial,repeat,branch}`；`ZT::{VisitStmt_(WhileNode),VisitStmt_(IfThenElseNode)}` | `TL::test_serial_repeat_and_branch_formulas`；`E2L`；自动 While guard 见 `ZT` | PARTIAL | Python 三类组合已测；自动 While 明确拒绝，原生 branch 是 conservative 累加上界而非通用 max 递归 |
| LAT-014 | per-core candidates、phase add、boundary loop 与 protocol II | `LM::{aggregate_phase,aggregate_per_core,repeat_per_core}` | `TL::test_per_core_phase_max_cross_wait_add_and_protocol_ii`；`E2L` | PARTIAL | 纯成本组合公式已测；无 PerRole candidate 自动提取、runtime wait 或 role 调度闭环 |
| LAT-015 | 全参数、零分母/溢出与测量校准 | `LM::{CudaLatencyParams,Calibration,_checked_add,_checked_mul,_usable_rate}`；`CM::_Calibration`；`manifest/calibration-sm120.json` | `TL::{test_integer_overflow_and_invalid_values_are_rejected,test_calibration_converts_ns_and_per_second_units_explicitly,test_require_measured_rejects_defaults_and_unmeasured_inputs}`；`TCAL`；`E2L`、`E2CAL`、`CAL` | DEVICE_VALIDATED_SUBSET | SM120 11 项 microbenchmark 及独立 dtype 率真实测量；其余字段仍可 default/fallback，非全 SM/全参数精度证明 |
| IR-001 | schedule-unit grammar、guard whitelist、stage/core 归属 | `ZP::{AnalyzeScheduledStmts,Z3SeqStmtFlattener}`；`ZT::AnalyzeScheduledTasks`；标准 For order/stage annotations | `TP::{test_replayable_bind_is_not_a_solver_task,test_task_count_must_match_scheduled_body_statements}`；`TT::test_default_noop_and_complete_problem_is_idempotent`；`E3` | PARTIAL | 仅 flat top-level statement 合同；无 ScheduledTIR codec、guard 白名单和 role/core 全 grammar |
| IR-002 | versions/modes/manual/unlimit/alias/AIV/conflicts metadata | `ZP` audit/order/stage/`tl_pipeline_buffer_versions`；`BV` | `TP::{test_buffer_versions_map_uses_ordered_buffer_data_vars_and_intimm,test_audit_ii_and_starts_preserve_int64_values}`；`TB`；`E3` | PARTIAL | version/audit/map 子集已测；mode/unlimit/alias/role/conflict metadata 未实现 |
| IR-003 | Task/Control 树、clone/substitute/path/compare/iteration | 无等价 Task/Control tree；`ZP::Z3SeqStmtFlattener` 仅产生 flat array | `TP::test_task_count_must_match_scheduled_body_statements` 只证明 flat 边界；`E3` 不是 tree 覆盖 | NOT_IMPLEMENTED | 无嵌套 ControlNode、clone/extra-info、path compare 或 flattened iteration 实现 |
| IR-004 | resource/HBM/special/access/core availability 分析 | `ZT::{TaskBufferRegionCollector,TaskFeatureExtractor,InferResourcePairs}`；`LM::Resource` | `TT::{test_copy_gemm_parallel_vector_and_scalar_features_and_resources,test_resource_pairs_are_derived_from_effective_masks,test_opaque_side_effect_is_not_silently_dropped_from_dependencies}`；`E3` | PARTIAL | access/resource mask 子集已测；special register、core/warp availability 与 role fixed-point 未实现 |
| IR-005 | logical/physical、may/must/opaque footprints | `ZT::TaskBufferRegionCollector`；`ZP::Z3BufferRegionCollector`；`PP::PipelineRegionsMayConflict` | `TT::{test_opaque_side_effect_is_not_silently_dropped_from_dependencies,test_raw_war_waw_fixed_dependencies_follow_source_order}`；`E3` | PARTIAL | full-region pointer 与保守 opaque guard 可用；无独立 may/must physical-footprint/alias 证明器 |
| IR-006 | L0 pitch 证明与 fractal plane 物理冲突 | 无 NV L0 plane/pitch 对应；`PP::PipelineRegionsMayConflict` 只作独立通用区域判断 | `TT::test_distance_two_conflict_is_not_dropped_by_distance_one_reasoning` 仅验证通用仿射距离，不是 L0 替代测试；`E3` | NOT_APPLICABLE_HARDWARE | Ascend L0 fractal plane/pitch ABI 无直接对应；CUDA 地址/swizzle 仍需 IR-005 的独立证明 |
| PASS-001 | while synthetic、condition/bound snapshot 与恢复 counter | `ZT::TaskFeatureExtractor::VisitStmt_(WhileNode)` 明确 fatal；`CP` 无 Z3 while normalize/restore | 无 While 专项 host oracle；`E3` 仅联合回归，不能视为覆盖；静态 guard 位于 `ZT` | BLOCKED_EXPLICIT | While 自动成本/IR 明确不支持；无 synthetic extent、snapshot 或恢复实现 |
| PASS-002 | task 物化、guard/stage 归一与单 pipe/PerCore 合同 | `ZT::AnalyzeScheduledTasks`；`ZP::AnalyzeScheduledStmts`；仅 task stream 提取，无 materializer | `TP::{test_replayable_bind_is_not_a_solver_task,test_task_count_must_match_scheduled_body_statements}`；`TT::test_default_noop_and_complete_problem_is_idempotent`；`E3` | PARTIAL | flat statement 识别已测；不自动包装任意 IR，不做 guard/stage 全归一或 PerRole 校验 |
| PASS-003 | core/warp role candidate、输入 fixed-point 与 projection | 无 Z3 role assigner；现有 `src/cuda/transform/auto_schedule.cc::AutoWarpSpecialization` 被 `CP/ZP` 与 G3 互斥 | `TP::{test_auto_warp_specialization_conflict_is_explicit,test_cuda_pipeline_rejects_auto_ws_before_any_scheduler_runs}`；`E3` | NOT_IMPLEMENTED | WS 是独立旧路径，不是 Z3 role assignment/projection 证据 |
| PASS-004 | write-first 四态、guard 覆盖与完整 owner frontier | 无 owner planner；`ZT` 默认 owner 列表为空；`ZP::PlanLoop` 对非空 owner 明确拒绝 | `TP::test_unexpressible_dependencies_fail_before_solver`；`E3` | BLOCKED_EXPLICIT | 自动 owner claim/write-first 未实现；显式 owner lifetime 也不能进入当前 CUDA pipeline IR |
| PASS-005 | 递归 AutoSchedule、内存组、版本/address 校验 | `ZP::Z3PipelinePlanner::PlanLoop`；`ZT::Z3TaskAnalyzer::InferLoop` | `TP::{test_complete_problem_projection_and_stock_pipeline_planning,test_output_depth_covers_solver_stages_and_buffer_versions}`；`TT::test_real_z3_integration_after_inference`；`E3` | PARTIAL | 单 loop flat solve+projection 已测；无递归 Control 调度、exact address 或全版本校验 |
| PASS-006 | iteration/counter、epoch/counter group 与 ring protocol | `ZP::buffer_version_map`；`BV`；`IP::BuildPipeline` 的 logical-iteration ring | `TB::test_explicit_buffer_version_override`；`TI::test_inject_software_pipeline_sibling_buffers`；`E3` | PARTIAL | 简单 logical-iteration ring 可用；owner/counter epoch、group init/advance 与 mbarrier phase 协议未闭环 |
| PASS-007 | scalar/special/epoch/counter demand 与 loop exit fixed-point | 无 Z3 demand resolver；现有 `src/cuda/transform/auto_schedule.cc` 仅为互斥 WS 路径 | `TP::test_auto_warp_specialization_conflict_is_explicit` 仅证明互斥；`E3` 不覆盖 demand | NOT_IMPLEMENTED | 无 role demand fixed-point、special/epoch demand 或 loop-exit 分析 |
| PASS-008 | sync site/epoch/closure、RAW+reuse 与冗余消除 | `PP` dependency/async planning；`IP::{ValidatePipelineBody,PopulateWaitCounts}` | `TQ::{test_pipeline_planning_reaches_mixed_dependency_fixed_point,test_pipeline_planning_redirects_copy_anchor_through_copy_producer}`；`TI::test_inject_software_pipeline_checks_cross_thread_raw_reordering`；`TT::test_async_same_stage_is_audited_for_manual_schedule_and_rejected_otherwise`；`E3` | PARTIAL | stock RAW/async 子集可用；无 owner epoch closure，多 async producer-stage 由保守同阶段/下游拒绝处理 |
| PASS-009 | acquire/release/prime/drain、ring 与 PerCore 同步物化 | `IP::{BuildPipeline,CompletePipelineLoopStatements,RelaxTrailingConsumerWaits}` | `TI::{test_async_pipeline_groups_multiple_copy_producers,test_async_pipeline_waits_for_individual_physical_commit_group,test_async_pipeline_relaxes_loop_wait_and_descends_trailing_drain}`；`TD`；`E3`、`E3D`、`E3DF` | DEVICE_VALIDATED_SUBSET | 仅 A+B 简单 commit/wait/ring 数值正确；同 stage 多 producer 可测，多 producer-stage/owner/PerRole 未完成 |
| PASS-010 | barrier ID/phase range、版本分配、safe reuse 与 channel 独占 | `IP` pipeline barrier 创建/phase wait；`src/cuda/transform/lower_shared_barrier.cc::LowerSharedBarrier` | `TQ::test_pipeline_planning_before_after_mbarrier_arrive_wait_plan`；`TI::test_pipeline_explicit_gemm_mbar_uses_logical_epoch`；`E3` | PARTIAL | stock mbarrier 子集已测；无通用显式 ID reservation、owner channel 独占或预算分配器 |
| PASS-011 | 物理版本、pitch/index/guard/layout/broadcast/elision | `BV`；`IP::{PipelineBufferReconciler,ExpandAnnotatedLayoutForMultiVersionedBuffer}` | `TB::{test_explicit_buffer_version_override,test_multiple_buffers_mix_overrides_with_auto_inference}`；`TI::{test_inject_software_pipeline_sibling_buffers,test_inject_software_pipeline_expands_annotated_layout}`；`TD`；`E3`、`E3D` | DEVICE_VALIDATED_SUBSET | A+B shared ring 子集 device exact；owner broadcast、guard clock、counter elision 与全 layout 矩阵未闭环 |
| PASS-012 | scheduled lowering、pipeline boundaries、conditional Bind 与 role bodies | `ZP` order/stage projection；`PP::ValidateScalarDependencies`；`IP::ValidateScheduledBindDependencies`；无 CoreExtractor | `TP::test_stage_order_does_not_bypass_scheduled_bind_validation`；`TI::{test_inject_software_pipeline_replays_scalar_bind_dependencies,test_inject_software_pipeline_schedules_bind_that_reads_pipeline_buffer}`；`E3` | PARTIAL | 标准 lower 与 replayable/same-stage Bind 子集已测；跨 stage Bind 拒绝，role extraction/general clone 未实现 |
| PASS-013 | epoch domain、strong/weak happens-before 与周期 separating cut | 无等价组件；`IP::ComputeBufferVersions` 仅为 downstream liveness heuristic | `TB`/`TI` 仅证明 ring rewrite，不是 lifetime-cut oracle；`E3` 不核销本证明 | NOT_IMPLEMENTED | 无 EpochDomainRegistry、generation strong/weak 边或周期 separating-cut 证明 |
| PASS-014 | 正 alias、live boundaries、physical lifetime 与保守 opaque | 无 Z3 alias producer；`SR`/`MA` 仅为现有下游 storage pass，非等价证明 | 无 parity host oracle；`E3` 不包含 generation-lifetime 证明 | NOT_IMPLEMENTED | 不把 StorageRewrite 或 shared merge 当成 Ascend alias analysis 已完成 |
| PASS-015 | 手工 schedule event/happens-before alias | 无 Z3 manual alias analyzer；`SR` 仍只是通用下游 | `TP` manual-stage 测试不含 alias contract；`E3` 不核销本条 | NOT_IMPLEMENTED | 无手工调度 event/happens-before alias 算法或 contract |
| PASS-016 | alias contract、clique packing、exact placement 与私有 arena | `CP` 调用 `MA`；`MA`；`SR`；无 Z3 exact-offset/placement API | `TD`/`E3D` 只覆盖普通 shared 分配；`testing/python/transform/test_tilelang_transform_disable_memory_reuse.py` 是既有独立路径，未作为本次 148 项证据 | PARTIAL | 通用 shared merge/reuse 可复用；exact offset、overlap/placement 校验、AIV/private arena 与最终容量证明未闭环 |
| PASS-017 | 8-slot compaction/knapsack/32-slot mutex spill | 无 Ascend hard_event/get_buf/rls_buf 对应实现；不得生成 CUDA 伪 mutex | 无等价 ABI 测试；PASS-010 的 mbarrier 测试只验证不同 NV 机制 | NOT_APPLICABLE_HARDWARE | Ascend flag/mutex 指令与预算无直接对应；通用 barrier 分配仍按 PASS-010 PARTIAL |
| PASS-018 | staged manual/physical versions 到 internal scoped attr | `ZP` 生成 `tl_pipeline_buffer_versions`；`BV::ReadBufferVersionOverrides` 校验并由 `IP` 消耗/清理 | `TB::{test_annotation_is_consumed_and_second_pass_does_not_reapply_override,test_nonpositive_version_is_rejected,test_invalid_override_mapping_is_rejected}`；`E3` | PARTIAL | internal count map 合同已测；公共 manual/physical mode normalizer 与 alias/mode 元数据未实现 |
| PASS-019 | 每个物理版本的 layout/offset/alignment 重映射 | `IP::{ExpandAnnotatedLayoutForMultiVersionedBuffer,RewriteAllocBuffer}` | `TI::{test_inject_software_pipeline_expands_annotated_layout,test_inject_software_pipeline_sibling_buffers}`；`E3` | PARTIAL | 同一 layout 的版本扩展/leading stride 已测；无独立 per-version layout/offset/alignment remap |
| PASS-020 | software dual copy 在调度前拆成 role-indexed ordinary copy | 无 CUDA dual-copy split rewrite；`ZT::HandleCopy` 仅处理普通 `CopyNode` | 无专项 host 测试；`TT` copy feature 测试不证明 split 等价 | NOT_IMPLEMENTED | 无 role-indexed M/N split、rank/2:1 校验或 rewrite 前后 cost/region 等价证据 |
| PASS-021 | read-modify-write 显式化并先于依赖分析 | `ZT::TaskBufferRegionCollector::VisitStmt_(BufferStoreNode)` 同时记录 write 并访问 value；`ZP` 同型 collector；无独立 legalization pass | `TT::test_raw_war_waw_fixed_dependencies_follow_source_order` 仅覆盖依赖类型，非专门 RMW legalization oracle；`E3` | PARTIAL | 已写 RMW IR 可被收集读写；无 MODE_MERGING 等效前置 rewrite 和直接专项测试 |
| PASS-022 | OOB clamp 与精确 padding fill 进入调度 | `CP::LegalizeSafeMemoryAccess` 在 Z3/task analysis 之后；无 pre-schedule OOB fill rewrite | 无专项 host 测试；`E3` 不证明 OOB region 进入 solver | NOT_IMPLEMENTED | 动态 tail/padding footprint 未在调度前物化，不能由后段安全访问 pass 补核销 |
| PASS-023 | ND/NZ layout copy 展开为 scatter+post_copy | 无 NV ND2NZ lowering；`CP` 不包含该 Ascend pass | 无等价 parity 测试；PASS-020/022 也未完成，不能称为替代覆盖 | NOT_APPLICABLE_HARDWARE | Ascend ND2NZ 指令和物理 NZ 存储无直接对应；通用“所有 op 可调度”仍有独立缺口 |
| PASS-024 | Cube fractal storage/alias/pitch 前置规范化 | 无 NV fractal storage normalizer；CUDA layout 走自身独立 pass | 无等价 parity 测试；IR-005/PASS-019 不作 fractal 物理证明 | NOT_APPLICABLE_HARDWARE | Ascend Cube fractal storage 无直接对应；CUDA swizzle/layout 必须独立合法化与验证 |
| PASS-025 | 调度前仅显式 unroll，原子区域保持 | `CP` 在 `InferZ3PipelineProblem`/`Z3PipelinePlanning` 前调用 `UnrollLoop`+`Simplify` | `TT::test_main_cuda_pipeline_infers_before_z3_planning` 证明 analysis/planning 次序；无 Z3 unroll 原子边界专项；`E3` | PARTIAL | pass 顺序已接入；缺少显式/非显式 unroll 与原子区域的直接 Z3 oracle |
| API-001 | 原子 task 与 latency/II override 公共 API | `AN::z3_pipeline_annotations` 以 top-level statement 为 task；`ZT::AnalyzeScheduledTasks`；`LM` whole-task override | `TP::test_python_helper_validates_per_task_and_manual_shapes`；`TT::test_missing_ii_is_filled_without_overwriting_explicit_fields`；`TD`；`E3`、`E3D` | PARTIAL | 显式 annotation task/cost 的 A+B 子集可用；无 `tilelang/cuda/language/schedule_hint.py::Task` 原子公共 API |
| API-002 | manual stage/offset/num_stages 与 Pipelined interop | `AN::z3_pipeline_annotations`；`tilelang/language/loop.py::Pipelined`；`ZP` manual/order/stage conflict guards | `TP::{test_manual_offset_and_minimum_stage_normalization,test_manual_schedule_conflicts_with_handwritten_order_stage,test_python_helper_validates_per_task_and_manual_shapes}`；`TD`；`E3`、`E3D` | DEVICE_VALIDATED_SUBSET | A+B 覆盖 Pipelined+默认 auto annotations；manual_schedule/offset 仅 host，无 `T.Stage` 同名 API 或全 interop 验收 |
| API-003 | guarded PerCore/PerRole candidates 与 exactly-once | 无公共 PerRoleTask/candidate API；`LM::aggregate_per_core` 仅成本 helper | `TL::test_per_core_phase_max_cross_wait_add_and_protocol_ii` 只测公式；`E2L` 不核销 API/runtime | NOT_IMPLEMENTED | 无 candidate guard、single-pipe、sync migration 或 runtime dispatch |
| API-004 | region/scope/cross/group conflict hints | 无 CUDA Z3 conflict-hint API/normalizer；`PP::PipelineRegionsMayConflict` 不接受该源合同 | 无专项 host 测试；`TT` 依赖测试不包含 hint | NOT_IMPLEMENTED | 不能显式覆盖/强制冲突，也无 same/cross/group 作用域归一 |
| API-005 | buffer version count/mode/alias/opt-out/selected annotation | `AN` ordered `buffer_vars`；`ZP::buffer_version_map`；`BV` exact count | `TP::test_buffer_versions_map_uses_ordered_buffer_data_vars_and_intimm`；`TB::test_explicit_buffer_version_override`；`E3` | PARTIAL | Buffer.data/count/opt-out 子集已测；无 iteration/counter mode、alias 一致性和 selected/request 全 schema |
| API-006 | 用户手写 ring 声明、距离分析且不重分配 | 无公共 manual-ring declaration；`BV` 只消费 compiler-internal exact count map | `TB` 仅 internal override，不是手写 ring API 测试；`E3` 不核销本条 | NOT_IMPLEMENTED | 无 flattened/subrange ring 声明或 manual clock 依赖合同 |
| API-007 | scope 容量绕过且不作物理保证 | 无 unlimit annotation；`TCO::memory_group_capacities` 要求显式容量；`CM::CostModelCapacityError`；`GS` 保留 capacity-unknown | `TT::test_hardware_capacity_requires_explicit_optin_field`；`TM::test_full_stage_capacity_unknown_survives_z3_lower_bound_and_enters_top_k`；`TG::test_capacity_unknown_is_last_deterministic_and_available_to_top_k`；`E3`、`E4C` | NOT_IMPLEMENTED | 没有容量 bypass；capacity-unknown 被保留而非删除/UNSAT/免费通过，不能据此核销 unlimit API |
| API-008 | exact address+显式版本与 placement 保真 | 无 CUDA exact-offset annotation；`MA` 是通用 allocator 而非该 API | 无专项 host 测试；`TD` 普通 allocation 不证明 exact placement | NOT_IMPLEMENTED | 无地址/版本联合校验、overlap 拒绝或跨版本/layout/flatten 地址保真证据 |
| API-009 | VF/warp-region 原子区域与独立 latency override | 无 region latency API；`AN`/`LM` 仅 whole-task latency/II | `TL::test_latency_and_ii_overrides_are_independent_and_validated` 不是 region 测试；`E2L` 不核销本条 | NOT_IMPLEMENTED | 无 VALU-only/warp-region 原子边界和独立实测 latency annotation |
| API-010 | 显式同步、ID reservation 与 caller-managed contract | 现有 `tilelang/language/allocate.py::alloc_barrier`、`tilelang/language/builtin.py::{mbarrier_arrive,mbarrier_wait_parity,named_barrier_arrive}`；`IP`/`LowerSharedBarrier`；无 Z3 reservation | `TQ::test_pipeline_planning_before_after_mbarrier_arrive_wait_plan`；`TI::test_pipeline_explicit_gemm_mbar_uses_logical_epoch`；`E3` | PARTIAL | CUDA 同步 API/stock 子集存在；未接入 Z3 ID range、unit-flag caller contract 或 owner channel |
| API-011 | stateful mode 的 RAW/WAR/WAW/fence/core demand | 无 special-register 框架；`ZT` 对未知 tile op 记 unknown，对 opaque/stateful 自动依赖路径报错 | `TT::test_opaque_side_effect_is_not_silently_dropped_from_dependencies` 仅证明保守 guard；`E3` 不证明 mode 语义 | NOT_IMPLEMENTED | 无 TF32/MMA/store-mode/loop-exit state dependency 或 role demand |
| API-012 | L0C 双目的与 FixPipe unit flag 控制 | 无 Ascend dual-destination FixPipe/unit_flag_ctrl 对应；普通 CUDA store/mbarrier 不是同指令 | 无等价 ABI 测试；API-010 只验证不同 NV 同步原语 | NOT_APPLICABLE_HARDWARE | Ascend 双目的 FixPipe 与 unit flag ABI 无直接对应；普通 split/store 不得冒充 parity |
| API-013 | M/N software dual copy、rank/2:1 与 role-indexed lowering | 无 CUDA software dual-copy 公共 API；`ZT::HandleCopy` 仅普通 copy | 无专项 host 测试；PASS-020 同缺口 | NOT_IMPLEMENTED | 无 split API、动态 allocation fallback 或 cost/region 等价证据 |
| HW-001 | 三物理 core mask、AIV banks 与 cthread 启动模型 | 无同构 AIC/AIV mask；`src/cuda/transform/auto_schedule.cc` 的 warp role 是独立抽象且与 G3 互斥 | `TP::test_auto_warp_specialization_conflict_is_explicit` 只验证 scheduler ownership，不是 core-mask parity；`E3` | NOT_APPLICABLE_HARDWARE | Ascend AIC/AIV 物理启动/mask 无直接对应；通用 role PASS-003/007 仍未实现 |
| HW-002 | L0 data/SF 同 owner/版本/counter/address group | 无 L0A/B+SF binding ABI；`ZP/BV` 仅通用 buffer version map | `TP::test_buffer_versions_map_uses_ordered_buffer_data_vars_and_intimm` 只证明通用 map，不是 SF group；`E3` | NOT_APPLICABLE_HARDWARE | Ascend L0 scale-factor 绑定无直接对应；通用 Z3-013/PASS-006 不得伪造 group parity |
| HW-003 | AIC+双 AIV 共享 3KiB SS pool | 无同构 SS 物理池；`TCO::memory_group_capacities` 只读显式 CUDA per-block opt-in shared capacity | `TT::test_hardware_capacity_requires_explicit_optin_field`；`manifest/hardware-sm120.json`；`E3` | NOT_APPLICABLE_HARDWARE | Ascend SS packed publication 无直接对应；CUDA shared 容量测试是独立属性/ solver 合同 |
| HW-004 | 8 intra、16/AIV cross 与 sid*16 物理 flag namespace | 无同构 flag bank；CUDA `mbarrier`/named barrier 走 `IP`、`LowerSharedBarrier` 与独立语言 API | `TQ::test_pipeline_planning_before_after_mbarrier_arrive_wait_plan` 只验证 NV barrier，不是 Ascend namespace；`E3` | NOT_APPLICABLE_HARDWARE | 物理 flag 数量/bank/寻址无直接对应；通用分配 PASS-010 仍 PARTIAL |

### 6.1 统计摘要（84 项，非 84 项完成）

| 分组 | IMPLEMENTED_HOST_TESTED | DEVICE_VALIDATED_SUBSET | PARTIAL | BLOCKED_EXPLICIT | NOT_IMPLEMENTED | NOT_APPLICABLE_HARDWARE | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PIPE | 0 | 1 | 1 | 0 | 0 | 0 | 2 |
| Z3 | 10 | 2 | 7 | 0 | 0 | 0 | 19 |
| LAT | 1 | 2 | 11 | 0 | 0 | 1 | 15 |
| IR | 0 | 0 | 4 | 0 | 1 | 1 | 6 |
| PASS | 0 | 2 | 11 | 2 | 7 | 3 | 25 |
| API | 0 | 1 | 3 | 0 | 8 | 1 | 13 |
| HW | 0 | 0 | 0 | 0 | 0 | 4 | 4 |
| **总计** | **11** | **8** | **37** | **2** | **16** | **10** | **84** |

`DEVICE_VALIDATED_SUBSET` 不是 full-device parity；`IMPLEMENTED_HOST_TESTED` 也不是端到端完成。本表不授予任何条目 arbitrary-IR/full-Ascend 闭环结论，也不宣称 GDN 或经典 workload 最终验收完成。

### 6.2 尚未闭环的硬缺口

1. **Owner runtime/ring/sync**：Z3-005/009 只核销 solver 直线/wrap 约束；`ZP` 对非空 owner 列表在 solve 前拒绝。Owner planner、counter/epoch、ring、sync、ID channel 仍未闭环（PASS-004/006/008/009/010/013）。
2. **General Let clone**：Z3-010/015 仅有 stage-order 与无 clone 子集；replayable Bind 和 scheduled same-stage Bind 是不同受限路径。`IP::ValidateScheduledBindDependencies` 仍拒绝跨 stage scheduled Bind，未实现 clone 链、重命名与 defs-before-uses。
3. **Multi-async producer-stage**：`ZT::ValidateAsyncSameStage` 只允许显式 async producer/consumer 在 manual schedule 下同 stage，automatic 路径拒绝；`IP::PopulateWaitCounts` 对 consumer 依赖多个 async stages 仍报 `A dependency on multiple async stages is not supported`。同 stage 多 producer host 测试不等于多 producer-stage 支持。
4. **Automatic arbitrary-IR**：`ZT` 只分析带 marker 的 flat、静态/白名单 IR；opaque/stateful、非静态 extent、While 等保守报错或要求显式 override。distance-2 正/负斜率测试只证明该仿射子集，不能外推任意 IR、Task/Control tree 或通用合法性证明。
5. **硬件特有条目**：LAT-012、IR-006、PASS-017/023/024、API-012、HW-001/002/003/004 为 `NOT_APPLICABLE_HARDWARE`。它们只核销“不得伪造直接映射”的边界；相关通用 copy/layout/barrier/capacity 功能仍按各自 PARTIAL/NOT_IMPLEMENTED 状态保留。
6. **审计元数据与最终验证**：solver 返回 `(starts, versions, II)` 不带 SAT/fallback verdict；capacity-unknown、rasterization-unmodeled 和 measured/fallback provenance 必须继续保留。现有设备证据不覆盖通用性能、正确性或最终 workload 验收。

详细源码/日志解释见 `docs/design/cuda_z3_parity_evidence.md`。

## 7. `testing/ascend/auto_schedule` 逐文件覆盖与拟建 CUDA 对照

本节保留原源测试文件与全部测试函数名，防止只按文件名粗略归类而漏掉末尾回归用例。下文 `TC/...` 是**历史拟建路径**，不是当前真实文件，也不再表示统一 PLANNED 状态；当前 NV 实现、真实 host 测试、执行日志和端到端边界一律以第 6 节为准。目标用例可以按 CUDA 术语重命名，但必须在核销台账保留“源函数+参数组合 → NV 函数+参数组合”的双向映射；Ascend 专用断言转到第 6 节相应 `NOT_APPLICABLE_HARDWARE` 项，不得删除后当作已覆盖。

### 7.1 三个共享 helper（无测试函数，也须迁移测试基础设施）

| 源文件 | 覆盖内容 | 拟建 CUDA 对照 |
|---|---|---|
| `testing/ascend/auto_schedule/_task_utils.py` | `_make_program` 构造显式/隐式 Task copy；`_bind_target`；`_materialize_schedule_units`；收集 latency/ii/core_mask 和 schedule units | `TC/_task_utils.py`：CUDA target binding、role metadata 收集和 canonical task fixture；不能让 helper 顺带跑被测 pass 而掩盖输入合同 |
| `testing/ascend/auto_schedule/_scheduled_ir.py` | `unit` 独立构造 stage/guard/cost/core 的显式输入；`copy_ring` 分别构造 Prepare 前或 Sync/Materialize 前的 iteration/counter、owner、guard、版本 fixture | `TC/_scheduled_ir.py`：独立构造 scheduled IR，注释明确 core/role 编码；保持“不调用被测 pass”这一隔离性 |
| `testing/ascend/auto_schedule/_reuse_utils.py` | 循环 scalar 双 buffer fixture（可 interleave、固定 versions）；`_insert_sync_alias_pairs` 读取 alias contract，支持 disable_reuse | `TC/_reuse_utils.py`：CUDA shared buffer lifetime/alias fixture 和 contract reader |

### 7.2 `test_access_footprints.py`（21 个）

- **源函数全量**：`test_address_of_escape_keeps_unknown_read_footprint_live`、`test_unknown_write_index_does_not_kill_other_elements`、`test_unknown_region_write_does_not_kill_other_elements`、`test_coverage_distinguishes_mutable_index_snapshots`、`test_pointer_footprint_must_fit_the_logical_view`、`test_indirect_intrinsic_read_footprint_is_not_a_compact_region`、`test_raw_storage_pointer_escape_is_not_hidden_by_a_structured_access`、`test_nd2nz_post_copy_uses_its_dma_read_span`、`test_nz_column_slice_keeps_unmodified_rows_live`、`test_nd2nz_scatter_keeps_partial_vector_tail_live`、`test_nd2nz_statement_intrinsics_keep_source_alias_eligibility`、`test_nd2nz_scatter_source_can_alias_a_later_ub_copy_destination`、`test_nd2nz_post_copy_keeps_the_last_required_element_live`、`test_nd2nz_scatter_checks_the_full_vector_read_footprint`、`test_mutable_layout_metadata_does_not_prove_physical_coverage`、`test_layout_metadata_buffer_stays_live_until_the_access`、`test_padded_copy_coverage_uses_physical_rows`、`test_padded_copy_coalesced_aligned_rows_allow_reuse`、`test_compact_l0_drain_preserves_ub_write_coverage`、`test_padded_copy_overflow_excludes_storage_from_reuse`、`test_access_footprints_alias_contract`。
- **覆盖要点**：opaque pointer/index；unknown may-write 不等于 must-write；mutable snapshot；pointer extent/view 边界；gather/scatter；ND2NZ 物理 span；layout metadata；padding；compact L0；最终 alias contract。
- **CUDA 对照**：`TC/test_access_footprints.py`（历史拟名；当前状态见第 6 节）。保留通用 pointer/may-must/opaque/layout 测试；ND2NZ、compact L0 等直接 ABI 用例映射到 IR-006/LAT-012 的 N/A 负面测试，并用 CUDA 对应 strided/swizzle/TMA footprint 另写独立正向用例，不能复用 Ascend 物理公式。

### 7.3 `test_buffer_epochs.py`（3 个）

- **源函数全量**：`test_storage_clock`、`test_iteration_clock_requires_single_owner`、`test_bound_sf_and_data_share_counter_and_union_epoch_guard`。
- **覆盖要点**：iteration/counter/auto、guard、单/多 owner、单/多版本；Prepare 不扩 buffer；counter init/advance 位置和共享；iteration 单 owner；data/SF union epoch guard。
- **CUDA 对照**：`TC/test_buffer_epochs.py`（历史拟名；当前状态见第 6 节），以 ring phase/counter 替换物理 flag ID；L0 data/SF 组合部分归 HW-002，另建 CUDA 实际共享 storage group 用例。

### 7.4 `test_buffer_lifetimes.py`（15 个）

- **源函数全量**：`test_same_pipe_issue_order_does_not_allow_dma_alias`、`test_dma_completion_through_another_pipe_still_allows_alias`、`test_read_modify_write_reduce_keeps_incoming_accumulator_live`、`test_single_trip_writer_and_multi_trip_sibling_reader_keep_overlap`、`test_nested_carried_lifetime_preserves_local_dma_reader`、`test_multi_trip_nested_scopes_use_hierarchical_summary`、`test_loop_varying_mutually_exclusive_phases_keep_distinct_epoch_storage`、`test_preloaded_read_only_storage_stays_live_through_loop`、`test_loop_carried_lifetime_preserves_local_reader`、`test_ring_generation_lifetimes`、`test_local_ring_generations_can_share_whole_allocation`、`test_buffer_lifetimes_alias_contract`、`test_l0_address_reuse_includes_the_bound_sf_lifetime`、`test_preloaded_nested_reader_stays_live_across_parent_iterations`、`test_parent_writer_and_nested_reader_keep_periodic_lifetime`。
- **覆盖要点**：issue≠completion；跨 pipe completion 允许复用；RMW；nested/live-through；guard 的 iteration variance；read-only live；ring generations；alias contract；data/SF 联合 lifetime；周期父子 scope。
- **CUDA 对照**：`TC/test_buffer_lifetimes.py`（历史拟名；当前状态见第 6 节）。以 cp.async/TMA 与 CUDA core/Tensor Core completion 边替代 MTE，保留嵌套和周期 separating-cut oracle；L0/SF 直接部分归 N/A，不跳过普通 group lifetime。

### 7.5 `test_buffer_materialization.py`（8 个）

- **源函数全量**：`test_physical_versions_use_the_prepared_clock`、`test_aliases_use_allocation_pitch`、`test_broadcast_fill_initializes_every_version`、`test_storage_reads_in_guards_follow_the_owner_version`、`test_iteration_clock_counts_logical_trips_across_nested_loops`、`test_counter_group_survives_only_for_physical_versions`、`test_single_version_counter_elision_rejects_live_uses`、`test_bound_data_and_sf_preserve_user_leading_dimensions_under_one_ring`。
- **覆盖要点**：prepared clock；alias allocation pitch/stride；broadcast fill；guard storage read；nested logical trip；counter group 生命周期；非法单版本 elision；用户已有 leading dims 与统一 ring。
- **CUDA 对照**：`TC/test_buffer_materialization.py`（历史拟名；当前状态见第 6 节），逐 IR 检查 version dim、strides、index、guard、layout 和 counter 消除。

### 7.6 `test_buffer_ownership.py`（9 个）

- **源函数全量**：`test_write_first_ownership`、`test_nested_alias_accesses_have_one_complete_owner`、`test_ss_packed_publication_has_one_shared_owner`、`test_explicit_storage_policy_overrides_automatic_claims`、`test_nested_control_reads_belong_to_the_enclosing_owner`、`test_row_writes_and_whole_buffer_read_belong_to_outer_owner`、`test_version_policy_controls_storage_and_alias_eligibility`、`test_l0_sf_group_selects_one_complete_owner`、`test_sf_version_override_applies_to_the_allocation_group`。
- **覆盖要点**：read/write-first、guards、manual stage、VF；完整 owner frontier；SS packed publication；显式/manual policy；loop bound/assume reads；row write+whole read；版本 opt-out；L0 group 和冲突 override。
- **CUDA 对照**：`TC/test_buffer_ownership.py`（历史拟名；当前状态见第 6 节）。SS/L0 专属项按 HW-002/HW-003 核销；其余用 shared/register/tmem owner 与 alias view 逐项验证。

### 7.7 `test_buffer_plan_validation.py`（3 个）

- **源函数全量**：`test_invalid_storage_plans`、`test_external_fill_is_marked_for_broadcast`、`test_bound_data_and_sf_cannot_have_different_owners`。
- **覆盖要点**：nested owner、owner 外 read、自读 fill、L1 broadcast 禁止、未规范化 loop bound；initial/partial/final fill 均可广播；data/SF owner 必须一致。
- **CUDA 对照**：`TC/test_buffer_plan_validation.py`（历史拟名；当前状态见第 6 节），必须保留所有负面诊断；NV 不支持的 pool 不应被静默忽略，而要有明确 scope 错误。

### 7.8 `test_buffer_version_annotations.py`（5 个）

- **源函数全量**：`test_buffer_version_annotations_use_storage_identity`、`test_invalid_buffer_version_annotation`、`test_conflicting_alias_annotations`、`test_placement_requires_an_explicit_version_count`、`test_placement_rejects_unapplied_fixed_versions`。
- **覆盖要点**：data Var 而非 view 身份；count/mode 分解；非法 tuple/mode；alias 冲突；exact placement 需要显式 count；固定版本未应用时报错。
- **CUDA 对照**：`TC/test_buffer_version_annotations.py`（历史拟名；当前状态见第 6 节），覆盖公共 Python API 和 AutoSchedule selected/requested 一致性。

### 7.9 `test_conflict_hints.py`（5 个）

- **源函数全量**：`test_hint_scope_and_region`、`test_assume_no_conflict_overrides_alias_conservatism`、`test_assume_conflict_uses_common_projection_domain`、`test_assume_conflict_keeps_same_and_cross_distances_separate`、`test_forward_loop_carried_dependency_without_hint`。
- **覆盖要点**：root/loop、exact footprint、same/cross；不同 logical view 的 no-conflict；跨 scope group/common projection；same 与 cross 不混；无 hint 的 forward carried dependence。
- **CUDA 对照**：`TC/test_conflict_hints.py`（历史拟名；当前状态见第 6 节），验证 annotation 被消费且 runtime 无残留、forced conflict 优先和距离独立。

### 7.10 `test_control_flow.py`（1 个）

- **源函数全量**：`test_control_read_is_snapshotted_before_body_mutates_it`（condition/loop-bound 两参数）。
- **覆盖要点**：复杂条件和 buffer-dependent loop bound 在 body 写之前生成 Bind snapshot，branch 双分支保留。
- **CUDA 对照**：`TC/test_control_flow.py`（历史拟名；当前状态见第 6 节），另补 while synthetic extent、恢复后 counter 下标和 nested VF 不重写测试。

### 7.11 `test_core_resolution.py`（19 个）

- **源函数全量**：`test_copy_destination_selects_core`、`test_scalar_cannot_mix_cube_and_vector_local_memory`、`test_core_local_condition_cannot_control_other_core`、`test_counter_updates_cover_all_protocol_cores`、`test_control_extent_restricts_flexible_scalar_task`、`test_control_reads_must_be_available_on_child_core`、`test_epoch_guard_producer_must_cover_all_protocol_cores`、`test_ambiguous_cross_core_pipe_uses_available_context`、`test_subblock_guard_is_specialized_before_core_assignment`、`test_scalar_global_store_uses_the_existing_vector_core`、`test_subcore_buffer_expressions_preserve_definitions_and_views`、`test_nested_subcore_projection_reuses_binders_and_views`、`test_subcore_view_metadata_projects_without_cloning_common_storage`、`test_grouped_fixpipe_copies_require_matching_subcore_targets`、`test_fixpipe_route_keeps_its_task_local_scope`、`test_projected_fixpipe_route_rejects_repeated_external_effects`、`test_projected_loop_break_follows_its_cube_branch`、`test_projected_loop_exit_follows_scalar_demand_from_later_consumers`、`test_unit_flag_pairing_between_source_copies_remains_caller_managed`。
- **覆盖要点**：pipe/destination/scalar memory；cross-core 输入可用性；counter/epoch demand；ambig pipe；guard specialization；view/binder projection；Fixpipe routing/replay 限制；loop exit；unit flag。
- **CUDA 对照**：`TC/test_core_resolution.py`（历史拟名；当前状态见第 6 节），改写为 warp role、warpgroup 和 role-local register/shared demand。FixPipe/unit flag/AIV mask 直接断言归 API-012/HW-001，通用“重复副作用必须拒绝”和 loop-exit demand 必须有 NV 正向测试。

### 7.12 `test_cross_core_flags.py`（10 个）

- **源函数全量**：`test_generated_flags_do_not_collide_with_explicit_flags`、`test_unbounded_explicit_flag_is_rejected`、`test_dynamic_nz_prefix_keeps_subcore_write_regions`、`test_explicit_nz_pitch_cannot_hide_overlapping_subcore_writes`、`test_alternating_epilogue_uses_each_aiv_flag_bank`、`test_cross_aiv_memory_dependencies_require_supported_ordering`、`test_only_actual_cube_dependencies_can_order_aiv_accesses`、`test_alternating_aiv_writers_without_cube_ordering_are_rejected`、`test_ss_uses_standard_cross_core_sync`、`test_ss_disjoint_aiv_read_modify_write_needs_no_sync`。
- **覆盖要点**：显式 ID 保留与无界拒绝；NZ 动态 footprint 不能误证 disjoint；AIV bank；AIV↔AIV 不支持路径；真实 AIC order 才能解依赖；SS cross-core 和 disjoint RMW。
- **CUDA 对照**：`TC/test_cross_core_flags.py`（历史拟名；当前状态见第 6 节），使用 warp/mbarrier ID 与跨 role happens-before；不支持的 warp 间情形也必须 fail 而非自动添加未证明 relay。NZ/AIV/SS 直接语义分别归 N/A，CUDA swizzle footprint 另测。

### 7.13 `test_epoch_guards.py`（4 个）

- **源函数全量**：`test_guard_equivalence_preserves_storage_epochs`、`test_nested_condition_cannot_escape_into_outer_epoch`、`test_lexical_epoch_guard_is_available_at_first_access`、`test_cross_stage_storage_cannot_use_counter_clock`。
- **覆盖要点**：结构/证明等价 guard 共享 clock，mutable memory 不共享；nested guard 不泄漏；union/cross-stage/late definition 回退 unconditional；counter 不可跨 stage。
- **CUDA 对照**：`TC/test_epoch_guards.py`（历史拟名；当前状态见第 6 节），保留 guard canonicalization、fresh mutable read 和 fallback domain 测试。

### 7.14 `test_estimate_latency.py`（25 个）

- **源函数全量**：`test_l0_gemm_effective_k_cost`、`test_l0_gemm_mn_cost_rounds_to_fractals`、`test_l0_gemm_k_cost_uses_compute_groups`、`test_l0_gemm_cost_respects_fp32_mode`、`test_l1_gemm_cost_uses_compute_groups`、`test_blockscaled_gemm_measured_compute_cost`、`test_dense_fp8_gemm_keeps_its_cost_model`、`test_estimate_latency_annotates_all_materialized_tasks`、`test_frontend_task_hints_feed_shared_task_metadata`、`test_rewritten_dual_copy_matches_equivalent_aiv_mte_access`、`test_aiv_mte_large_hbm_uses_full_occupancy_ii`、`test_aiv_mte_small_copy_keeps_descriptor_floor_ii`、`test_cthread_symbolic_stride_uses_conservative_unknown_split`、`test_aiv_mte_high_dim_copy_uses_actual_mte_row_axis`、`test_rank_one_dual_copy_keeps_geometry_across_rewrite`、`test_n512_copy_uses_measured_full_width_mte3_rate`、`test_equivalent_strided_regions_share_mte_row_geometry`、`test_sid_partition_does_not_override_strided_mte_geometry`、`test_single_sid_copy_cost`、`test_updated_non_aiv_copy_costs_match_full_path_remeasurement`、`test_mx_sf_copy_cost_uses_region_payload`、`test_fixpipe_quant_cost_uses_destination_payload_bytes`、`test_narrow_fixpipe_dual_uses_physical_trailing_row_width`、`test_fixpipe_dual_ignores_destination_unit_axis_for_row_width`、`test_task_cost_arguments_are_validated`。
- **覆盖要点**：dense/MX effective shape、HF32、dtype groups；所有 task annotation/override；dual 与 rewrite 等价；occupancy 和 descriptor floor；unknown stride/high-dim/rank-1/N512；非 AIV copy 重测值；SF/FixPipe payload；非法参数。
- **CUDA 对照**：`TC/test_estimate_latency.py`（历史拟名；当前状态见第 6 节）。公式结构逐项对照，但 NV 数值必须来自各 SM 的独立 hand calculation/microbenchmark；不能照抄源 `(latency,II)`。`T.Task` override 和 validation 可保持架构无关精确断言。另补 composed/PerCore 数值测试，见第 8 节。

### 7.15 `test_flag_allocation.py`（9 个）

- **源函数全量**：`test_serial_protocols_fit_hardware_flag_budget`、`test_sibling_counter_owners_share_one_release_ring`、`test_distinct_counter_channels_keep_exclusive_rings_and_spill`、`test_flag_spill_preserves_bound_dynamic_ids`、`test_sparse_out_of_range_flags_are_compacted_without_spill`、`test_legal_sparse_flags_remain_unchanged`、`test_cross_core_rings_fit_available_slots`、`test_single_iteration_needs_no_reverse_flag`、`test_extent_one_child_uses_only_its_owner_versions`。
- **覆盖要点**：serial protocol budget；sibling shared release ring；不同 channel 独占和 spill；动态 ID range；sparse compaction/legal unchanged；cross-core banks；单 iteration 无 reverse；child 只用 owner versions。
- **CUDA 对照**：`TC/test_flag_allocation.py`（历史拟名；当前状态见第 6 节），将预算抽象为 NV barrier resource，分别测试分配、复用证明和不足时显式错误；8-slot compact/knapsack/32-slot spill 直接 ABI 归 PASS-017，不得假装 CUDA 也有该 fallback。

### 7.16 `test_reuse_integration.py`（2 个）

- **源函数全量**：`test_alias_contract_lifetime`、`test_fixed_versions_allow_ub_reuse_above_raw_capacity`。
- **覆盖要点**：alias map 在 InsertSync→LowerScheduledTIR→Merge 的生命周期；fixed versions 的 raw size 总和超物理池时仍可按 disjoint lifetime 复用到 128KiB。
- **CUDA 对照**：`TC/test_reuse_integration.py`（历史拟名；当前状态见第 6 节），贯通新 alias producer 与现有/新 shared allocator，区分 solver 容量 heuristic 和最终实际分配字节。

### 7.17 `test_schedule_integration.py`（30 个）

- **源函数全量**：`test_sibling_l0_rings_survive_layout_lowering`、`test_counter_slot_is_compatible_with_l1_layout_lowering`、`test_subcore_roles_survive_complete_lowering`、`test_subcore_gemm_results`、`test_subcore_fixpipe_synchronization`、`test_subcore_loops_can_have_different_extents`、`test_subcore_break_preserves_the_other_cube_stream`、`test_subcore_scalar_loop_exit_reaches_later_cube_consumers`、`test_subcore_strided_view_preserves_physical_storage`、`test_per_core_fixpipe_dispatch_preserves_each_subcore_output`、`test_per_core_vector_dispatch_preserves_each_subcore_output`、`test_subcore_mutable_scalar_is_replicated_with_its_guard`、`test_mixed_kernel_uses_requested_vector_count`、`test_every_kernel_is_scheduled_and_called_in_program_order`、`test_nested_loop_break_is_emitted_on_every_core`、`test_single_version_sibling_protocol_survives_reuse_and_lowering`、`test_mmad_direction_reaches_only_cube_and_precedes_its_gemm`、`test_sf_group_owners_survive_a_plain_gemm_sibling`、`test_scalar_snapshot_precedes_pipeline_update`、`test_pipeline_preserves_addresses_across_versions_layout_remap_and_flatten`、`test_overlap_is_checked_after_automatic_versions_expand`、`test_fractal_padding_is_included_in_the_physical_footprint`、`test_holes_and_versions_cannot_exceed_physical_capacity`、`test_unscheduled_pipeline_rejects_placement_instead_of_dropping_it`、`test_ub_placement_survives_version_and_vf_lowering`、`test_simt_context_limits_placement_after_full_lowering`、`test_ub_placement_keeps_only_active_subcore_storage`、`test_placement_drops_eliminated_allocations`、`test_placement_rewrites_extern_storage_pointers`、`test_ss_placement_survives_cross_core_version_lowering`。
- **覆盖要点**：完整 lowering 后 ring/layout/role/同步/loop-break/输出正确性；PerCore dispatch；mutable scalar；多 kernel program order；mode writer；Let snapshot；exact address 经版本/layout/flatten；物理 padding/capacity/overlap；非自动 pipeline 不丢 placement；SIMT context、active storage、dead alloc、extern pointer、SS。
- **CUDA 对照**：`TC/test_schedule_integration.py`（历史拟名；当前状态见第 6 节）。CPU IR snapshot 与后续 GPU 数值正确性测试分开记录；完整 CUDA pipeline 只由一个 scheduler 所有。L0/AIV/SS/Mmad 直接硬件断言按 parity N/A 核销，剩余版本、layout、alias、placement、控制流、多个 kernel 和副作用测试不得省略。

### 7.18 `test_schedule_units.py`（8 个）

- **源函数全量**：`test_materialize_schedule_units_builds_trivial_guarded_schedule`、`test_materialize_schedule_units_normalizes_partial_manual_stages`、`test_auto_schedule_preserves_explicit_stage_dependencies`、`test_dynamic_tail_regions_preserve_only_required_recurrences`、`test_auto_schedule_requires_its_input_contract`、`test_single_version_sibling_owners_preserve_dependency_kind`、`test_ss_versions_share_capacity_across_aivs`、`test_sf_hazard_uses_the_data_ring_variable_in_the_solver`。
- **覆盖要点**：初始 stage=-1/guard；partial stage 归 0；显式 stage；动态 tail 只保留必要 recurrence 并捕获真实 Z3 输入；missing units/cost/multiple task 错误；单版本 sibling 的 data 与 owner exclusion 区别；SS 共享容量；SF 使用 data ring variable。
- **CUDA 对照**：`TC/test_schedule_units.py`（历史拟名；当前状态见第 6 节），同时保留 monkeypatch 捕获依赖边的白盒 oracle；不能只检查“pass 没报错”。SS/SF 专属部分归 N/A，通用 grouped capacity/variable 必测。

### 7.19 `test_special_registers.py`（8 个）

- **源函数全量**：`test_nested_pipe_wait_orders_the_following_pipe_user`、`test_nested_pad_writer_follows_the_reader_core`、`test_projected_special_register_writers_keep_their_subcore`、`test_shared_special_register_writer_covers_both_cores`、`test_cube_mode_changes_order_nested_and_following_gemms`、`test_cube_mode_schedule_preserves_iteration_order`、`test_manual_stages_break_scalar_pipe_ties_by_iteration`、`test_cube_mode_writers_follow_nested_gemm_core`。
- **覆盖要点**：nested pipe wait；pad/atomic/hf32/mmad/loop-break writer 的 core 和顺序；跨 iteration 固定 model 下 tie-break；Scalar 无自动 resource exclusion 时大 stage 优先；writer 跟随 consumer core。
- **CUDA 对照**：`TC/test_special_registers.py`（历史拟名；当前状态见第 6 节），将 HF32 映射为适用的 TF32 模式，将 MMAD direction 等无对应控制拆为 N/A；保留通用 RAW/WAR/WAW、loop exit、固定 start/II 的排序 oracle。

### 7.20 `test_sync_access_footprints.py`（9 个）

- **源函数全量**：`test_l0_tail_load_accounts_for_source_fractal_padding`、`test_l0_load_matches_explicit_transfer_region_conflicts`、`test_declared_l1_pointer_read_keeps_its_own_footprint`、`test_pointer_store_past_view_keeps_scalar_read_dependency`、`test_expanded_footprint_preserves_explicit_conflict`、`test_no_conflict_keeps_original_logical_region_matching`、`test_unknown_pointer_read_does_not_create_read_read_dependency`、`test_padded_source_hint_survives_loop_scheduling`、`test_physical_pointer_write_orders_control_expression_reads`。
- **覆盖要点**：fractal padding；显式 transfer region；declared pointer read；past-view store；expanded footprint 与 hint 区分；原始 logical region matching；read-read 不建依赖；loop 中 padded hint；pointer write 与 control read。
- **CUDA 对照**：`TC/test_sync_access_footprints.py`（历史拟名；当前状态见第 6 节），保留逻辑/物理 footprint 两层和 conflict-hint 匹配规则；CUDA 地址布局需自己的证明，不能借 L0 规则减少同步。

### 7.21 `test_sync_boundaries.py`（4 个）

- **源函数全量**：`test_acquire_dominates_possibly_empty_consumer_loop`、`test_missing_counter_channel_keeps_lexical_handshake`、`test_narrow_child_handshake_cannot_replace_wider_parent_dependency`、`test_branch_switch_acquires_previous_iteration_before_nested_overwrite`。
- **覆盖要点**：0-trip consumer 仍 acquire；缺 counter channel 时保留 lexical，非法 stage distance 报错；窄 child 不覆盖宽 parent；branch switch 的跨代 acquire/release 位置和 ID。
- **CUDA 对照**：`TC/test_sync_boundaries.py`（历史拟名；当前状态见第 6 节），这是 barrier correctness 必测，不允许以“通常 loop 非空”简化为普通 linear schedule。

### 7.22 `test_sync_dependencies.py`（12 个）

- **源函数全量**：`test_ring_flags_use_the_data_clock`、`test_sync_respects_atomic_task_boundary`、`test_dma_issue_order_is_not_completion_order`、`test_dma_regions_at_int32_limit`、`test_manually_indexed_storage_keeps_the_lexical_clock_under_a_guard`、`test_dma_row_writes_only_serialize_overlapping_regions`、`test_partial_l0_loads_complete_before_gemm_and_order_overlapping_writes`、`test_different_l0_pitches_conflict_within_same_plane`、`test_partial_l0_copies_require_provably_equal_pitches`、`test_l0_copy_pitch_is_instantiated_per_iteration`、`test_atomic_l0_copies_require_a_uniform_pitch`、`test_equivalent_lexical_epoch_serializes_counter_ring`。
- **覆盖要点**：lexical/counter clock；atomic task guard；DMA completion；int64 endpoint；manual ring guard；row overlap；L0 load/pitch；counter 与等价 lexical epoch 的串行化。
- **CUDA 对照**：`TC/test_sync_dependencies.py`（历史拟名；当前状态见第 6 节），保留 async completion、RAW/WAR/WAW、ring clock、int64 和区域 overlap；L0 pitch 专属测试归 IR-006，NV 地址/swizzle 等价和不等价要另证。

### 7.23 `test_tasks.py`（8 个）

- **源函数全量**：`test_invalid_atomic_task`、`test_per_core_candidates_preserve_conditions`、`test_per_core_fixpipe_candidates_keep_separate_routes`、`test_per_core_vector_candidates_project_to_distinct_aivs`、`test_per_core_fixpipe_sync_requires_single_execution`、`test_sync_set_is_migrated_into_each_per_core_candidate`、`test_guarded_per_core_task_keeps_wider_sync_outside_candidates`、`test_fill_outside_vf_synchronizes_on_scalar_pipe`。
- **覆盖要点**：cross-pipe/cross-core/two-sync/nested 错误；显式/推断/混合 candidates 保留 guard；Fixpipe route scope 和重复副作用；vector projection；sync 必须单次执行；candidate 内迁 set；较宽 group guard 的 sync 留外部；非 VF fill 走 Scalar。
- **CUDA 对照**：`TC/test_tasks.py`（历史拟名；当前状态见第 6 节），以 PerRole candidates 保留全部控制流和 exactly-once 合同；同步只能随候选 guard 或留外层的判定必须与源一致，非法多 pipe task 要 fail。

## 8. 必须新增的直测矩阵（现有 22 文件之外）

以下是原始源码审计发现的覆盖空洞或迁移新增边界。部分已有当前真实实现/测试（见第 6 节），本节不得再被解释为“全部未做”；尚未覆盖的参数、诊断和端到端组合仍是开放门禁，不得因已有联合测试总数而删去。

1. **`TC/test_z3_scheduler.py`**：
   - 直线：非平凡最优 makespan、data latency、resource II 两方向、owner 两方向、pipe order、start/index tie、n=0/1、非 sat。
   - 循环：手算最小 II 及二分访问序列；正距离、负 buffer-variable 距离；resource/owner 回绕；`k/r` 域；`enable_offset` 两种；manual stages 和 pipe source order；`stage_order_deps`；memory group；两种 version recalculation。
   - C++ 重建：min-k 和 min-start 归一、raw stage、phys-time tie 大 stage 优先、Let clone 链、overall latency、父子嵌套 producer timestamp。
   - 失败：无 FFI、缺 z3、异常、返回长度错误、manual infeasible、循环无模型串行回退、rlimit/unknown；每个路径检查 warning/error 和实际输出，而非只看日志。
   - 确定性：同输入重复调用、交错不同输入、多线程 compile、必要时多进程；比较 starts/order/II/versions，不只比较 sat。
2. **`TC/test_z3_debug.py`**：verbose 文本、schedule 编号补洞、JSON 全字段和 seed、无 matplotlib 时警告、写文件失败不修改调度、临时目录 artifact；timestamp 不参与确定性比较。
3. **`TC/test_estimate_latency.py` 补充数值 oracle**：composed Seq 两条公式、repeat T=0/1/N、branch max、While fallback、PerCore 无/有 boundary、跨迭代 suffix+prefix、非法多 pipe/candidate/wait conditional；覆盖所有 override 组合和 latency=0/II=1。
4. **`TC/test_latency_params.py`**：第 4 节参数 schema 的每个字段都有至少一个改变结果的用例；零 bandwidth/throughput、未知 op、sub-byte bytes、int64 溢出、动态 bound 无信息和静态 shape cap，防止“参数存在但从未接线”。
5. **`TC/test_pipeline_integration.py`**：验证 CUDA 新 Z3 链与旧 role-based WS、通用 Pipelined/InjectSoftwarePipeline 的互斥或组合顺序，auto-schedule off 的 alias 路径、disable reuse 仍同步、多 kernel program order。
6. **`TC/test_unsupported_ascend_concepts.py`**：逐项覆盖 parity 的 N/A。断言必须覆盖以下可审计结果：文档化不支持、明确诊断，或确认只生成语义不同的 NV 实现；不得把“静默无操作/返回原 IR”判为成功。防止 Ascend intrinsic/mask/pool/预算渗入 CUDA。
7. **TVM prover 独立隔离测试**：不要复用 scheduler 的通过结论。现有 `testing/python/layout/test_tilelang_layout_inference_z3_context.py` 中 fresh-context 两个相关用例可作为 prover 参考；新增直接区域冲突、guard implication、rlimit/unknown 与 scheduler 并发隔离，证明 `Z3Prover` 和 Python scheduler 不共享意外状态。
8. **校准证据测试**：每个 NV latency 参数关联架构、指令、shape/dtype、clock、静态公式和 microbenchmark；静态 cycle 断言与实测性能回归分开，不能以一次机器上的数字宣称所有 SM 正确。

## 9. 零遗漏核销规则（最终门禁）

1. **以最小语义条目建账**。每条必须具有唯一 ID（可采用本文 PIPE/Z3/LAT/IR/PASS/API/HW ID 或新增子 ID）、源位置、源行为、NV 映射、状态和目标测试。一个文件包含多个公式、约束、fallback 或硬件分支时必须拆条，禁止用“已移植某 pass”整包核销。
2. **每条四要素缺一不可**：
   - **源位置**：仓库相对路径 + 具体 symbol（函数/类/字段/公式/annotation）；必要时附源码行范围和输入 IR。
   - **NV 实现位置**：实际 NV 路径 + symbol + 接入 pipeline 的位置。历史“拟”路径已在第 6 节替换为当前真实位置；不存在的组件明确写无实现或 guard，不得写“同上”“类似实现”“应该可复用”。
   - **测试**：具体测试路径 + 测试函数 + 参数组合 ID；必须能反向定位到本条，不接受只有目录级“已测试”。
   - **证据**：独立预期值/手工模型、求解器 model、前后 TIR structural snapshot、诊断文本、分配字节、测试日志，及需要时的 GPU 数值结果和校准数据。证据必须与本条行为直接对应。
3. **N/A 也须完整核销**。N/A 的 NV 实现位置应是“明确拒绝/防止误映射”的实际 guard 或测试位置，而非空白；必须附硬件无对应的源码/架构说明、负面测试和不静默降级证据。通用算法不得因其中一个硬件名词无对应而整条 N/A；应先拆出通用部分并保持 PLANNED。
4. **禁止静默 skip**。不得删除测试、注释断言、以 `pytest.skip`/`xfail`/空 fixture/永远为真的条件代替实现，也不得以“无硬件”“后续再做”“与 CUDA 无关”关闭条目。硬件不可用时，CPU 可做的 solver/TIR 测试仍必须执行；GPU 项标为未执行而不是通过，并保留阻塞原因。
5. **参数化组合逐一枚举**。源函数的每个 `@pytest.mark.parametrize` 维度、显式参数组和 id 都要进入映射表。允许 NV 改硬件参数，但必须一对一说明对应关系；无法对应的组合必须落到具体 N/A 子条目和负面测试，不能只抽查一个代表值。
6. **预期值必须独立**。 latency/II/makespan/version 等先由公式、手算小型模型或独立 oracle 得出，再与实现比较；不得把当前实现的输出复制成 golden 来证明正确。IR 测试须检查 guard、metadata、stage、顺序、物理下标和诊断，不只检查 pass 返回成功。
7. **所有失败和 fallback 都是功能**。缺 z3/FFI、异常、unsat/unknown、rlimit、manual infeasible、空输入、非法 annotation、版本未应用、容量不足、无法证明 alias/core/flag 等路径都必须有测试和实际输出证据。日志出现“fallback”不等于存在 fallback，必须对照控制流。
8. **确定性须跨边界验证**。至少覆盖重复调用、交错求解、多线程和必要的多进程；比较 start/order/II/versions/生成 IR。单独核销 private context、lock、seed、rlimit、stable sort，TVM prover 的确定性不能代替 scheduler 的证明。
9. **pipeline 以最终 IR 为准**。除 pass 单元测试外，每条接入功能还须有完整 pipeline 证据，证明没有被后续 pass 丢弃、重复调度或改变语义。现有 CUDA WS/Pipelined/StorageRewrite/AutoTuner 的存在不能作为新功能完成证据。
10. **成本常数须带校准链**。每个 NV 参数记录架构、指令、dtype/shape、单位、来源、静态推导和 microbenchmark；性能实测与功能正确性分开。Ascend 周期常数、DeepGEMM ns heuristic 和 AutoTuner 实测 latency 不得互相冒充。
11. **新增源码行为先加账再实现**。后续发现新的 helper、依赖类型、resource bit、annotation、动态形状 fallback、诊断或测试参数时，先新增 parity 行和 NV/测试位置，再改实现；禁止代码先行而清单不更新。源 Ascend 测试变化也要同步复核本清单 22 文件/218 函数基线。
12. **完成判定采用闭环审查**。只有“NV 实现已落地 + 对应测试真实执行通过 + 独立证据可复查 + 源/NV/测试三向链接一致”才可核销相应范围。`IMPLEMENTED_HOST_TESTED` 只闭环 host 范围，`DEVICE_VALIDATED_SUBSET` 只闭环写明的设备子集；本文当前不授予任何条目 arbitrary-IR/full-Ascend 端到端完成状态。文章或后续报告引用本清单时必须附范围与证据，不能把设计文档、联合测试总数或单一 A+B 用例当作全量功能完成证明。
