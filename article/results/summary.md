# 结论：哪些工作由 Z3 完成，哪些仍由真实编译与实测决定

## 1. 最终交付的是一条可审计的候选筛选链

本分支没有把 Z3 当成延迟预测器，也没有用解析分数代替真实执行。完整链路是：先用编译前可证明的资源条件缩小空间，再用 SM120 实测标定的解析模型排序，最后仍交给 TileLang 的编译、独立 reference 校验和 CUDA-event profiler。搜索期最优与后续独立确认分开记录，预测排名所需的完整 oracle 成本不隐藏。

对应代码分别位于 [`tilelang/cuda/transform/z3_scheduler.py`](../../tilelang/cuda/transform/z3_scheduler.py)、[`tilelang/cuda/cost_model.py`](../../tilelang/cuda/cost_model.py) 和 [`tilelang/autotuner/guided_tuner.py`](../../tilelang/autotuner/guided_tuner.py)。默认 CUDA pipeline 与默认 AutoTuner 不变；新 pipeline pass 和 guided arm 必须显式启用。task latency/II、整 kernel 排序、layout cost model 和 TVM 通用 Z3 prover 仍按各自接口使用，互不冒充。

## 2. 中心验收：GDN 用更少候选保住了最终配置质量

T8192 的正式实验由 30 个新进程完成，每次 forward/backward、每种策略、每轮都有独立缓存和 before/after 清单。下表的加速比是三轮逐轮 stock-wall/guided-wall 的中位数，不用两个已经取过中位数的 wall 再相除。

| 公开 GDN 路径 | raw compile → guided compile | raw profile → guided profile | stock wall → guided wall | 搜索加速 | wall 节省 | 独立确认 |
|---|---:|---:|---:|---:|---:|---|
| forward | 4 → 2 | 3 → 2 | 34.102 s → 19.410 s | 1.757× | 43.1% | 各策略均为 `block_DV=16`，0.308992 ms |
| backward | 9 → 3 | 6 → 3 | 62.690 s → 49.599 s | 1.258× | 20.5% | guided/oracle 均为 `stage=2, group=8`，4.230792 ms |

forward 输出与 final state、backward 六个梯度均通过计时前后 correctness，最大 relative-L2 分别为 0.004271 和 0.005417，门槛为 0.02。backward guided 对 budget-raw 的确认 latency 比为 0.962739，对 budget-random 为 0.999626。前者说明等预算原始顺序漏掉了更好的配置，后者几乎相同，不能包装成普遍 kernel 提速。

这里的验收结论是“**更少搜索得到相同或不劣于对照的配置**”。T4096 的 backward 旧选择规则曾以 1.000624 的对照比失败；该记录没有被删除。统一改为 candidate-median 后，模型、空间与预算保持冻结，T8192 按预注册作为新 held-out。协议和原始日志见 [GDN 验收章](gdn-acceptance.md)。

## 3. 经典算子给出的不是单一方向的结果

四个经典 workload 各完成三次 cold、五种策略，winner 另有七轮独立 ABBA。下表把搜索加速、搜索期质量和独立确认放在同一口径中；confirmed ratio 是 guided 相对完整穷举或完整 raw 对照的 pair-local 比值，大于 1 表示 guided 更慢。

| workload | top-10 搜索加速 | 搜索期 quality | confirmed guided / exhaustive | 结论 |
|---|---:|---:|---:|---|
| 方形 GEMM | 26.404× | 1.01910 | 1.017253 | 保留了近优候选，优于两个等预算控制 |
| 短 K GEMM | 26.092× | 1.07296 | 1.039685 | 搜索大幅减少，最终慢约 3.97% |
| NHWC Conv2D | 13.218× | 1.19384 | 1.295844 | Z3 容量剪枝有效，top-10 排序明显漏优 |
| causal FlashAttention | 4.321× | 1.01158 | 1.021014 | 减少实测，最终点估计慢约 2.10% |

所有 oracle 最优都未进入 guided top-10：方形 GEMM 的预测名次为 32–33/240，短 K 为 120/240，Conv 为 16/120，FA 为 27/36。方形 GEMM 靠其他近优候选取得较小差距；Conv 的整体 Spearman 虽约 0.80，top-10 recall 却只有 0.6，最终差距在独立确认中扩大。FA 没有完整 legal-oracle：54 个 SAT 候选有 36 个有限预测、30 个成功 raw 实测，配对分母为 30，`legal_oracle_complete=false` 已保留。

方形 GEMM 的 top-1/5/10/20 敏感性中，top-5 的搜索期质量最好；top-20 没有继续改善。top-k 是实测预算，不是质量保证，rasterization 的 L2 locality 也仍未建模。四个 suite 的 20 个 arm 均通过独立 correctness；性能比较不套用 GDN 的专用通过门槛。逐项配置、等预算对照和误差表见 [经典算子结果章](classic-search.md)。

## 4. 约束减少与模型排序要分开评价

Z3 pruning 只接受真正的 UNSAT。`unknown`、timeout、外部 `Indeterminate`、重复配置和编译失败各有独立账目，不能被改写成“非法”。GEMM 与 Conv 的版本化 shared-memory 必要条件为

$$
B_{necessary}=\max\left(2K_{block}(M_{block}+N_{block})\max(1,n_{stages}),\;2M_{block}N_{block}\right).
$$

第一项覆盖 A/B 多版本共存，第二项覆盖输出 shared buffer；二者取最大值而非直接相加。GEMM 的 48 个预测超限与 48 个实际失败一一对应，guided top-10 在约束修正后不变。这证明的是该空间和本机容量的一个必要条件，不是任意 TileLang IR 的通用编译可判定性。

成本模型另做一件事：对保留下来的候选给可解释的 compute、DRAM/L2/shared、LSU、SFU、launch、occupancy/wave 和 overlap 分项。容量未知时保留候选、将有限预测排在未知预测之前，并留下 `score=null`。因此，“约束少删了多少候选”和“排序是否保住好候选”必须分别用合法空间审计与完整 oracle 相关性回答。

## 5. 编译器内迁移的完成范围必须按 84 项清单阅读

求解器、task latency/II、自动 task 分析、pipeline stage/order/buffer-version 投影都有真实实现和 host/device 子集测试。与此同时，owner runtime/ring/sync、通用 Let clone、多 async producer-stage 和任意 IR 自动分析没有闭环；需要这些能力的输入会明确拒绝，而不是静默退回一个看似成功的 stock 调度。

84 项 parity 的实际分类是 11 项 host-tested、8 项 device-validated subset、37 项 partial、2 项 explicit blocked、16 项 not implemented、10 项 Ascend 硬件不适用。ND2NZ、AIC/AIV、L0A/B/C、SS pool 和 flag bank 不因名字替换就变成 CUDA 功能。逐项源码和证据见 [迁移状态](implementation-status.md) 与完整附录。

## 6. 复现时需要保留的实验纪律

1. 先运行 `scripts/probe_cuda_hardware.py` 和独立 calibration；其他 GPU 不得复用 SM120 的吞吐常数。
2. 每个 cold run 使用新进程和事先不存在的 run-owned TileLang/CUDA/Torch/XDG/TMP 目录，保存 before/after 清单。这里的 cold 不是整机 OS cold。
3. 固定 raw 空间、canonical config ID、输入 seed、warmup/repeat 和 top-k；失败候选仍计入尝试。不得用收集到的 pytest 节点充当执行通过。
4. 用 stock-raw、legal-exhaustive、guided-topk 和两个等预算控制分别回答剪枝、排序与“只是少测”三种解释。
5. 最终 winner 使用搜索期 candidate-median 冻结，独立 ABBA 测量不参与选择；相同配置按身份处理，不构造虚假的配对差。
6. 只清理由本次 manifest 明确拥有的缓存，不动共享 `~/.tilelang`、其他项目或未知 GPU 进程。

安装、两并行构建、GDN、四个经典 suite、联合确认、winner cubin 导出与广泛编译矩阵的完整命令见 [`README.cuda-z3.md`](../../README.cuda-z3.md)。文章中的表格可由相应 JSON/CSV 重算，静态 shared、完整动态 shared、compile-only 与数值通过也不混作同一指标。

## 7. 可以带走的工程判断

在本机 RTX 5090 Laptop 和已验证的搜索空间上，Z3 的直接价值是**在真实编译之前删除已证明不可运行的候选**；解析成本模型的价值是**把有限实测预算优先分配给更有希望的合法候选**。GDN 中心验收证明这条链能在公开 forward/backward 上减少构造与 profile，并保住经独立确认的配置质量；经典算子同时证明，模型排序质量随形状而变，较大预算不自动等于更优结果。

因此，本分支适合作为默认 exhaustive AutoTuner 的显式加速路径和 CUDA Z3 pipeline 的受限研究实现，而不是替代所有真实编译、correctness 和 profiling 的黑盒。剩余 parity 缺口、失败记录和未建模项与性能表同样是交付的一部分。
