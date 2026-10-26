# 迁移状态：84 项清单与硬件边界

## 1. 全量枚举不等于全量完成

本分支逐项保留 Ascend Z3 scheduler、latency/cost、pass、API 和硬件抽象的 84 个 ID。每项都记录源位置、CUDA 对应源码、直接测试与端到端边界。当前分类如下。

| 状态 | 数量 | 含义 |
|---|---:|---|
| IMPLEMENTED_HOST_TESTED | 11 | 写明范围已有实现和真实 host 通过日志 |
| DEVICE_VALIDATED_SUBSET | 8 | 另有设备或独立校准证据，仅限明确子集 |
| PARTIAL | 37 | 已有实现或可复用组件，关键语义仍未闭环 |
| BLOCKED_EXPLICIT | 2 | 关键路径未实现，源码明确拒绝 |
| NOT_IMPLEMENTED | 16 | 没有可接受的 parity 实现 |
| NOT_APPLICABLE_HARDWARE | 10 | Ascend 专有指令、ABI 或物理池没有直接 NV 对应 |
| **合计** | **84** | **不是 84 项完成** |

完整条目见 [feature inventory](../../docs/design/ascend_z3_cost_model_feature_inventory.md)，机制与证据解释见 [parity appendix](../../docs/design/cuda_z3_parity_evidence.md)。这些 Markdown 文件在仓库中保留，不把它们压缩成“全功能迁移完成”的宣传结论。

## 2. 已执行的证据层级

下表各套件存在重叠，行数不能相加为总测试数。host 公式通过、pipeline host 通过、设备数值通过和独立微基准通过分别回答不同问题。

| 证据集合 | 真实结果 | 直接支持的范围 |
|---|---:|---|
| CUDA Z3 solver | 49 passed | 直线/循环约束、II、版本、unknown 与确定性 |
| task latency/II | 28 passed | memory/compute/composed/per-core 公式 |
| whole-kernel cost model | 31 passed | 解析排序、容量与 model card |
| capacity/guided | 85 passed | UNKNOWN 保留、top-k 与预算控制 |
| stock+guided AutoTuner | 91 passed，2 skipped | 真实 tuner 路径、错误与缓存合同 |
| task/pipeline 联合 host | 148 passed | 自动分析、主链、stage-order 子集、buffer versions |
| 冻结 host 编译矩阵 | 77/77 rows，130 cases passed | host/frontend/IR 的执行覆盖，不含设备数值 |
| SM120 pipeline A+B | 1 passed | 显式 annotation 的 default-off/opt-in 与逐元素精确输出 |
| SM120 独立标定 | 11 个 complete benchmarks | 本机 dtype、DRAM/L2/shared、SFU、LSU、launch 参数 |

对应日志包括 [solver](../../evidence/G1/solver-tests.txt)、[latency](../../evidence/G2/latency-model-tests.txt)、[pipeline host](../../evidence/G3/main-pipeline-host-tests-final.txt)、[host matrix](../../evidence/G5/compile-matrix-host-run-v1/summary.json) 和 [device A+B](../../evidence/G3/device-integration-test-retry.txt)。首次多异步阶段失败仍保存在 [device failure](../../evidence/G3/device-integration-test.txt)，后续特定 A+B 通过不会消除该通用限制。

## 3. Ascend 与 SM120 的资源抽象差异

迁移保留了调度问题，而没有把两套硬件名称做一一替换。CUDA 求解器和成本模型使用自己的资源、容量和校准来源。

| 层面 | Ascend 源机制 | 本机 CUDA/SM120 对应 |
|---|---|---|
| 执行资源 | Cube、MTE1/2/3、Vector、FixPipe、Scalar，另有 AIC/AIV mask | global load/store、shared、Tensor Core、CUDA core、SFU、scalar 七位资源 |
| 存储 | L0A/B/C、L1、UB、SS 与物理 pool/arena | DRAM、L2、shared、register/tmem；per-block opt-in shared 101376 B |
| 矩阵指令 | Cube 及 fractal shape、ND/NZ 物理布局 | 实测 `mma.sync` 路径；不套用 Hopper WGMMA 或 SM100 TCGEN05 参数 |
| 同步与所有权 | owner、counter/epoch、flag bank、跨核 ring | stock pipeline、cp.async/mbarrier、named barrier 的有限组合 |
| task 成本 | Cube/VALU/MTE/FixPipe latency/II 与 copy geometry | task latency/II 独立建模；整 kernel 排名另用 whole-kernel model |
| 硬件标定 | 源侧 cycle 参数与 Ascend shape | RTX 5090 Laptop 独立微基准与 SASS 审核，不继承 Ascend 常数 |

SM120 probe 记录 82 SM、1536 threads/SM、65536 registers/block 与 SM、64 MiB L2 等属性，详见 [hardware manifest](../../manifest/hardware-sm120.json)。吞吐参数来自 [calibration manifest](../../manifest/calibration-sm120.json)，并与正式 workload 分离。

## 4. 仍未闭环的硬缺口

1. **Owner runtime/ring/sync**：直线和 modulo owner 约束已有 solver 测试，CUDA adapter 对非空 owner 列表明确拒绝。owner planner、物理 ring、counter/epoch、同步与 channel allocation 尚未实现。
2. **General Let clone**：stage-order 已支持无 clone 子集，并核验 solver 返回的 raw stage；需要跨 stage clone 的 scheduled Bind 仍被拒绝。replayable Bind 和同 stage Bind 不是通用 clone。
3. **Multi-async producer-stage**：自动分析采用保守同阶段路径，stock `InjectSoftwarePipeline` 对多个 async producer-stage 仍报错。同 stage 多 producer 的通过不能外推到多 stage。
4. **Automatic arbitrary-IR**：自动 task 分析限于 flat、静态、白名单 IR。动态/不透明控制流、Task/Control tree、special state、嵌套 timestamp 和通用容量证明尚未覆盖。
5. **审计与公共 API**：solver 返回 `(starts, versions, II)`，没有独立 SAT/fallback verdict 字段；`T.Stage`、PerRole candidate、manual ring、exact placement 等完整公共 API 尚未对齐。
6. **硬件特有项**：ND2NZ、AIC/AIV、L0A/B/C、SS pool 和 Ascend flag bank 保持硬件不适用。通用 copy/layout/barrier 的独立 CUDA 工作仍需各自验证，不能借 N/A 关闭相邻条目。

因此，后续 GDN 和经典算子的实验结论应准确理解为：**在已实现的 Z3 约束与 CUDA 解析成本模型路径上，验证候选减少、搜索时间变化及最终配置质量**。这些实验支持外层搜索机制的有效性和有限 pipeline 子集的正确性，不补齐上述编译器语义缺口。
