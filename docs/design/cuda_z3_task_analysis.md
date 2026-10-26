# CUDA Z3 自动 task / cost 提取（G3）

实现文件：

- `src/transform/z3_task_analysis.h`
- `src/transform/z3_task_analysis.cc`
- `tilelang/cuda/task_cost.py`

本文只描述当前实现，不声称手动 `z3_pipeline_annotations` schema 已获得任意 TileLang IR 上的全自动通用 parity。

## 1. 入口与默认行为

C++ pass：

```cpp
tvm::transform::Pass tvm::tl::InferZ3PipelineProblem();
```

FFI 名称：`tl.transform.InferZ3PipelineProblem`。Pass 仅在 PassContext 中显式设置

```python
{"tl.enable_z3_task_analysis": True}
```

后工作。默认 config、无 `tl_z3_pipeline` loop、以及输入已经完整的 loop 都是严格 no-op。 Pass 不运行 solver，不生成 pipeline stage/order，也不替代 `Z3PipelinePlanning`。

由于本次改动不修改 `pipeline.py`，主 CUDA pipeline 不会自动插入此 pass。需要显式组合：

```python
from tilelang.cuda import task_cost  # 注册成本/容量 FFI

with tvm.transform.PassContext(config={"tl.enable_z3_task_analysis": True}):
    mod = tvm_ffi.get_global_func("tl.transform.InferZ3PipelineProblem")()(mod)
with tvm.transform.PassContext(config={"tl.enable_z3_pipeline_planning": True}):
    mod = tilelang.transform.Z3PipelinePlanning()(mod)
```

对带 `tl_z3_pipeline` 的 loop，pass 只补充缺失字段；annotation key 已存在时，即使其值是空数组，也视为显式 override，不重算、不覆盖。

## 2. 与 adapter 一致的 task 划分

scheduled statement 的提取规则与 `z3_pipeline_planning.cc` 保持一致：

1. `NormalizePipelineBody` 读取顶层 body；
2. 递归展开 `SeqStmt`，不展开其他 statement；
3. 排除 `AllocBuffer` / `DeclBuffer` declaration；
4. 用相同的 `IsReplayableScalarBind` 和 pipeline write-buffer 集排除可重放 scalar `Bind`；
5. 其余 statement 按 source order 形成一个 task。

`BufferRegion` 收集也沿用 adapter 的 tile op、parallel、`BufferLoad` / `BufferStore`、`address_of`、`tl.access_ptr`、`tvm_access_ptr` 规则。与 adapter 的调度输入检查不同，自动依赖提取遇到未知 pointer base 或带 side effect 的 opaque/stateful call 会报错；不会把未表示的访问当成无依赖。

## 3. 数据依赖和资源对

### 3.1 RAW / WAR / WAW

对 task `i -> j` 分别比较：

- `writes[i]` 与 `reads[j]`：RAW；
- `writes[i]` 与 `writes[j]`：WAW；
- `reads[i]` 与 `writes[j]`：WAR。

只有 `i < j` 时生成 same-iteration 边，distance=0，latency 为 source task 的 effective latency。loop-carried 检查覆盖包括 `i == j` 在内的 task 对；比较时将 consumer 侧的 pipeline loop var 平移一轮。冲突判断复用 `PipelineRegionsMayConflict` 和 thread domains，只有可证明不相交才省略边，symbolic alias / strided view 等情况保守保留。

cross-iteration distance：

- 若 storage 在 effective `buffer_vars` 中，使用 `-buffer_id-1`，由 Z3 buffer version 变量决定；
- 其他 storage 使用固定 distance=1，不假定其会自动多版本化。

相同 `(source,target,distance)` 的边去重；不同 buffer id 的版本约束不会互相合并。此处生成的是保守调度依赖，不是精确指令级 happens-before 证明。

### 3.2 Resource mask 和 resource pairs

mask 与 `latency_model.Resource` 的七位合同一致：global load/store、shared、tensor core、CUDA core、SFU、scalar。缺失 `resource_deps` 时，对 effective masks 有交集的每对 task 生成一个无序 resource pair。`resource_flags` 显式存在时使用显式值，不用重新估计的值替代。

manual schedule 缺失 `pipe_order_deps` 时按共享 resource mask 补 source-order pairs；自动调度补空列表。`owner_exclusion_deps` 和 `stage_order_deps` 在缺失时也只补空列表；显式非空值仍由原 adapter 拒绝。本 pass 不扩展这两种依赖的 downstream 能力。

## 4. TaskFeatures 和 Python FFI

### 4.1 FFI API

`tilelang.cuda.task_cost` 注册：

```text
tl.cuda.estimate_task(features, hardware, calibration, require_measured=False)
tl.cuda.z3_memory_group_capacities(hardware)
```

Python 直接调用为 `tilelang.cuda.task_cost.estimate_task(...)`；C++ 使用上述全局 FFI 名称。`features` 是 JSON-like `TaskFeatures` mapping（也接受 JSON 字符串或 `TaskFeatures`）。hardware/calibration 可为 mapping 或 JSON 路径；C++ annotation 使用路径。

hardware 必须具有：

```json
{
  "schema": "tilelang.cuda.hardware-spec",
  "device": {
    "compute_capability": "12.0",
    "shared_memory_per_block_optin_bytes": 131072
  }
}
```

`compute_capability` 转成 calibration 的 `smXY` arch。容量 helper 只读取明确的 per-block opt-in shared capacity；缺失、非整数或非正值均报错，不回退到静态 shared limit，也不使用设备型号表。

calibration 路径调用现有 `Calibration.load`，mapping 调用等价的 `Calibration.from_json`；随后调用现有 `latency_model.estimate_task`。本模块不复制或修改成本公式。返回严格 JSON-compatible dict：

```text
latency, ii, terms, resource_flags, reliability, fallback_used
```

`require_measured` 同时传给 calibration 和 leaf estimator。非 measured 模式下，原模型的显式 fallback/unit placeholder 仍通过 reliability 和 `fallback_used` 暴露；它不是虚构的默认实测硬件。

C++ 对每个需要成本或 mask 的 task 调 FFI，只填写缺失的 latency/II/resource 字段，并增加（若调用方未提供）：

- `tl_z3_pipeline_cost_reliability`
- `tl_z3_pipeline_cost_fallback_used`

### 4.2 当前特征覆盖

- **dense `T.copy`**：从 src/dst `BufferRegion` extent 和 dtype 计算 payload；src/dst bytes 不同则取较大值并标 upper bound。显式 `coalesced_width` 转成大 model 的 coalesced geometry 和 byte width；未提供时 geometry 为 `unknown`，不声称自动证明 coalescing。
- **dense `T.gemm`**：提取静态 `M/N/K`、输入 dtype 和 accumulator dtype。当前 `TaskFeatures` 只有一种输入 dtype，mixed-input-dtype GEMM 在自动成本路径报错。`GemmSP`、block-scaled 和其他 tile op 不冒充 dense GEMM。
- **parallel / scalar expressions**：静态 loop trip count 下计数 add/sub/mul/div/mod/min-max/compare/logic、`exp2` 和其他 special math；scalar expression 计入 scalar，parallel/vector expression 计入 vector。global/shared load/store 转为 memory accesses；fragment/register load 计入 vector read bytes，register write 不虚构独立 bandwidth term。
- **不支持的 tile op / extern arithmetic**：记录 `vector.unknown`、reliability=unknown，并在结构 mask 上保守 OR `127`，而不是当成零成本零资源。缺少可确定的依赖信息则直接报错。While 成本、非静态 task trip extent 也报错；调用方可显式提供 latency/II/resource override。

分支两侧当前按 conservative upper-bound 累加并标 `upper_bound`，不是精确 branch composition。地址计算、寄存器写带宽、最终 SASS instruction mix、bank conflict 和 layout-dependent coalescing 不在当前自动提取的精确覆盖范围内。

## 5. Buffer size 和 memory group

仅在 `buffer_sizes` 与 `buffer_vars` 都缺失时推断两者；只允许两者同时显式或同时缺失，避免给用户已有 size 数组猜测另一套顺序。推断顺序为 scheduled task 中 shared write buffer 的首次出现顺序，storage key 为 `Buffer.data`。同一 storage 的多个 static view 取最大 allocation bytes，sub-byte dtype 在全部 element bits 汇总后向上取整。

任何需要自动 size 的非静态 shape 都报错；不会变成 0 bytes。若显式 buffer size 已提供，则不会为了推导依赖或 group 而重新要求 shape 静态。

缺失 `memory_groups` 时，只把 effective `buffer_vars` 中 shared/shared.dyn buffer 放入一组，capacity 来自 hardware spec 的 `shared_memory_per_block_optin_bytes`。global 和 register/fragment 不生成 shared memory group。capacity 不自动扣除其他 persistent allocation、reserved bytes 或 downstream storage reuse；存在这些情况时必须显式提供修正后的 memory groups/容量合同，不能把本组当成整 kernel 精确 shared allocation。

## 6. Async producer 的 same-stage 边界

当前 solver 没有独立 same-stage constraint，`stage_order_deps` 又仍被 adapter 拒绝。因此本 pass 不伪装支持自动 same-stage 求解：

- 对已有 `software_pipeline_async_producers` / groups，按 producer write 与 direct consumer read 提取 producer/consumer pairs；同一 group 必须有相同且非空的 direct consumer set。
- `manual_schedule=True` 时要求每对 manual stages 相等，并写 `tl_z3_pipeline_async_same_stage_deps` 审计 pairs。
- 自动 schedule 遇到需要 same-stage 的显式 async producer/consumer 时直接报错，不返回不兼容结果。

本 pass 不新生成 implicit async producer flags；downstream `PipelinePlanning` 的 implicit async 分类和校验仍保持原状。owner exclusion 和 stage-order 语义没有扩展。

## 7. Host 测试范围

`testing/python/transform/test_tilelang_z3_task_analysis.py` 包含：

- 默认 no-op、完整输入幂等、缺字段补全和显式 override；
- RAW/WAR/WAW source-order dependencies、resource pairs；
- copy geometry/bytes、dense GEMM M/N/K/dtype/acc、parallel vector/SFU/read bytes、scalar resource；
- 非静态 shared buffer size 和 opaque side-effect 拒绝；
- manual async same-stage 审计及 automatic 拒绝；
- 真实 `tl.cuda.estimate_task` FFI JSON、measured-only、hardware capacity 缺项；
- 推导后接真实 CPU Z3 solver 的集成。

这些测试不需要 GPU，也不编译 CUDA kernel。原生 pass 测试需要主线先 build 新 C++ 文件；纯 Python `task_cost` 可独立做 CPU 测试。当前文档不把 synthetic calibration 的结果解释为真实设备性能。
