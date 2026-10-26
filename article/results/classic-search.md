# 经典算子：固定搜索空间与公平对照

## 1. Workload 与搜索空间

三个经典 TileLang 算子家族共有四个正式 workload。输入、kernel 输出为 FP16，累加与独立数学参考为 FP32。raw/legal 数字来自同一份 `ConfigSpace` 和本机显式 shared-memory 必要下界；legal 不表示所有配置都能通过后续 lowering、device compile 或 launch。

| suite | 固定问题 | raw | legal | 主预算 |
|---|---|---:|---:|---:|
| gemm_square | FP16 GEMM，M=N=K=4096，B[n,k] 转置 | 288 | 256 | top-10 |
| gemm_short_k | M=N=8192，K=256 | 288 | 256 | top-10 |
| conv2d_stride_tail | NHWC implicit GEMM，33×33→17×17，C64→F96，stride 2、tail | 144 | 128 | top-10 |
| flash_attention_causal | BSHD，B1/H16/S2048/D64，causal | 54 | 54 | top-10 |

GEMM 的空间为 `block_M/N=[64,128,256]`、`block_K=[32,64]`、`num_stages=[0,1,2,3]`、`threads=[128,256]`、rasterization 两值。Conv 删除 kernel 不消费的 rasterization 轴。FA 使用 `block_M/N=[64,128,256]`、stages 1–3 和两种线程数。

方形 GEMM 另做预注册的 top-1/5/20 敏感性。额外预算只运行 guided、`budget-raw` 和 `budget-random`，复用同 suite/repetition/cache/seed 的完整 top-10 stock/legal 基线。复用条件、分母和失败回退见 [sensitivity baseline plan](../../manifest/classic-sensitivity-baseline-plan.json)。主 top-10 的完整 oracle 成本单独保留，不能将基线复用描述为零成本。

## 2. 候选确实经过冻结 kernel factory

每个候选都调用对应模块的 `kernel_factory(config, shape=...)` 得到真实 JITKernel，再由 stock `AutoTuner` 编译、校验和 profile。没有为 cost model 另写替代 PrimFunc，也没有用简单的 fake callable 充当正式性能对象。

```python
return factory(
    config,
    shape=_fixture.shape,
    target=_target,
    pass_configs=_pass_configs,
    compile=_compile,
)
```

生产默认使用 `GuidedAutoTuner`、`SearchExperiment` 和 JITKernel 的真实 `Profiler`。mock 仅用于 CPU 单元测试，不能进入结果表。stock-raw、legal-exhaustive、cost-model top-k、原始顺序等预算、固定 seed 随机等预算五种策略使用同一输入和 correctness 路径。

## 3. 独立 FP32 reference 与 correctness

固定 seed 为 `20261026`。CPU 正态输入转为 FP16 后移动到 CUDA；同一 tuner 的候选共享 tensor 对象，避免数据生成顺序影响比较。

- GEMM 直接计算 FP32 `A @ B.T`；Conv 按 NHWC cross-correlation 定义逐维计算；FA 显式构造 causal mask、FP32 softmax 和 PV 累加。
- matmul/conv 在 reference 期间关闭 TF32 并恢复现场。FA 不调用融合的 SDPA 充当参考。
- actual 与 reference 在 FP32 中比较，默认 `atol=rtol=0.02`，`max_mismatched_ratio=0`。任何一个超差、NaN 或 Inf 都失败，不用允许的 mismatch 比例掩盖局部错误。
- reference timing 与 candidate timing 分开记录，validation/reference 不塞进 candidate latency。`collected`、compile-only 和数值通过分别入账。

CPU 测试与小形状数值测试定义见 [operators README](../../benchmark/z3_cost_model/operators/README.md)。正式大形状仍必须逐个通过真实 kernel 校验，不能用这些前置测试代替。

## 4. 约束只删除可证明的 UNSAT

GEMM/Conv 的 shared 必要下界为 `max(2*block_K*(block_M+block_N), 2*block_M*block_N)`，FA 为 `max(Q+K,Q+V,O)`。这些公式利用操作数同时存活关系，不声称等于最终 post-lowering shared 分配。stage 版本、reuse、padding、barrier、寄存器、TMA 对齐和 target-specific 指令选择在信息不足时保持 UNKNOWN 或不予编码。

SM120 没有照搬 sm_80/sm_90 的 atom 条件，因此 unknown=0 不能被解释为“所有合法性已证明”。`CostModelCapacityError` 也不同于 Z3 UNSAT：容量未知的候选保留在宇宙中并排在有效预测之后，top-k 覆盖时仍可进入编译验证。模型容量 UNKNOWN 与候选 compile/validation/profile failure 分开计数。

## 5. 缓存与统计分母

每次 arm/repetition 使用独立 Python 进程和 run-owned TileLang/CUDA/Torch/XDG/TMP 缓存，搜索 warmup=10、repeat=50，`early_stop=False`，grouped compile 与 multi-GPU 关闭。每轮保存 before/after 清单、config、环境、events 和 summary。

结果聚合要求每个选中配置恰好有一个终态 trial。缺失或重复 trial、cache-only、只收集到节点、错误结束的 run 不进入性能比较。质量比较优先使用完整 legal-oracle；若它不完整，则回退到完整 raw 的实测最优并标记 `quality_reference=stock-raw`。预测相关优先使用完整 legal-oracle，否则只连接 raw 中实际成功的配置，同时标记 `legal_oracle_complete=false` 和实际分母，不把失败候选藏起来。

最终表格同时报告 raw/legal/选中数、compile/profile 次数、各阶段墙钟、搜索内 best、独立确认值、失败候选和预测分母。搜索计数减少与最终配置质量需要分别成立，不能仅凭“少跑了几个候选”宣称模型有效。
