# GDN 搜索协议与缓存分账

## 1. 固定对象与候选空间

核心验收调用 FlashQLA 的公开 `chunk_gated_delta_rule_fwd` 与 `chunk_gated_delta_rule_bwd`，性能形状固定为 `B=1,T=4096,Hg=4,H=8,K=DV=128`，BF16，chunk=32，512 threads，`auto_cp=False`。反向计时从既有 forward artifacts 开始，artifact producer 固定 `block_DV=64`，不计入反向候选的搜索旋钮或计时区间。

| 方向 | 搜索旋钮 | raw | Z3 后保留 | guided 实测预算 |
|---|---|---:|---:|---:|
| fwd | `block_DV=[16,32,64,128]` | 4 | 3 | top-2 |
| bwd | `prepare_h_num_stages=[1,2,3] × group_reduce_block_size=[8,16,32]` | 9 | 6 | top-3 |

fwd 的动态 shared memory 公式为 $38912+576\times block\_DV$，因此 128 对应的 112640 B 超过本机 opt-in per-block 上限。bwd 的 `prepare_h_num_stages=3` 由 $57344+18432\times stages$ 同样排除。资源之外无法静态证明的属性仍保留为 UNKNOWN，不能由编译失败反推 UNSAT。公式与设备证据见 [fwd shared memory](../../evidence/G5/gdn-fwd-shared-memory.json) 和 [prepare_h shared memory](../../evidence/G5/gdn-prepare-h-shared-memory.json)。

五臂共享同一输入与公开执行路径：`stock-raw` 穷举 raw，`z3-legal-exhaustive` 穷举保留空间，`z3-cm-topk` 在保留空间内按解析模型排序后实测 top-k，`budget-raw` 与 `budget-random` 以相同 k 做原始顺序和固定 seed 随机顺序对照。模型 overhead、solver 检查和所有候选失败都进入事件日志。

## 2. 正式 cold 协议

正式搜索的每一轮都是新的 Python 进程。每个方向执行三次，每次五臂，合计 30 个 run；搜索 warmup=10、repeat=50。入口为：

```bash
for direction in fwd bwd; do
  for repetition in 0 1 2; do
    bash scripts/run_gdn_cold_fresh.sh \
      "$direction" "$repetition" evidence/G6/gdn-cold-fresh-repro
  done
done
```

驱动在每次启动前检查主机可用内存与 GPU 进程，拒绝覆盖已有日志。`SearchExperiment` 创建独立的 TileLang/CUDA/Torch/XDG/TMP 缓存，保存 before/after 文件清单。这里清理的是本次 run 的实验缓存起点，未触碰 `~/.tilelang` 或整机 OS 缓存。

协议审计不能只检查目录是否存在。`benchmark/z3_cost_model/gdn/audit_cold.py` 还检查：30 个身份是否完整且不重复、30 个 PID 是否不同、warmup/repeat 是否均为 10/50、候选预算是否改变、缓存路径是否仍位于 run 内、before 是否为空、是否存在成功 profile 和有限的最佳延迟。任何一项失败都会使审计返回非零。

早期 `repeat_search.py` 在同一进程内调度各臂，并沿用 warmup=2/repeat=5。该数据保留为 pilot，用于观察空间、配置和排序；其 wall time 不进入正式 cold 收益表。正式数据全部来自上述 fresh-process 脚本。这个修正不改变 raw/legal/top-k，也不删除任何 pilot 记录。

## 3. warm 与 no-cache 的独立敏感性

敏感性只做 `stock-raw` 与 `z3-cm-topk`，两个方向各一次，共 8 个独立进程。no-cache 同时禁用 TileLang 和 CUDA 编译缓存；warm 先预热，再用新的 tuner 在同一隔离缓存中测量。warmup run 与 measured run 分别有 JSONL 和 summary，预热成本没有被塞进 measured run 的计时。

下表的 compile 是候选构造尝试次数，warm 命中时不代表重新调用 NVCC。profile wall 是 stock profiler 阶段墙钟，包含其 reference 相关调用，不能解释为纯 kernel 时间。best ms 是该次搜索内最优值，尚未经过七轮独立确认。

| 方向 | 轨道 | arm | compile/profile 次数 | compile wall (s) | profile wall (s) | end-to-end (s) | 搜索内 best (ms) |
|---|---|---|---:|---:|---:|---:|---:|
| fwd | no-cache | stock | 4 / 3 | 29.422 | 1.359 | 31.243 | 0.158595 |
| fwd | no-cache | guided | 2 / 2 | 15.202 | 1.289 | 16.953 | 0.160312 |
| bwd | no-cache | stock | 9 / 6 | 52.156 | 4.031 | 56.712 | 2.051117 |
| bwd | no-cache | guided | 3 / 3 | 40.501 | 3.397 | 44.442 | 2.048730 |
| fwd | warm | stock | 4 / 3 | 0.0029 | 1.356 | 1.385 | 0.161231 |
| fwd | warm | guided | 2 / 2 | 0.0018 | 1.294 | 1.334 | 0.159168 |
| bwd | warm | stock | 9 / 6 | 0.0065 | 3.997 | 4.083 | 2.041240 |
| bwd | warm | guided | 3 / 3 | 0.0027 | 3.414 | 3.506 | 2.038489 |

按 $S=T_{stock}/T_{guided}$ 计算，no-cache 的 fwd/bwd 搜索时间比分别为 **1.843× / 1.276×**；warm 轨分别为 **1.039× / 1.165×**。这些数字来自单次敏感性，不替代三次 cold 的中位数。no-cache 的主要时间差落在候选构造阶段；warm 阶段则主要剩 profiler、校验和少量调度开销。

机器可读来源为 [no-cache summary](../../evidence/G6/gdn-cache-sensitivity-v1/analysis-no-cache/summary.csv)、[warm summary](../../evidence/G6/gdn-cache-sensitivity-v1/analysis-warm/summary.csv) 和 [协议记录](../../evidence/G6/gdn-cache-sensitivity-v1/protocol.json)。审计确认 8 个 PID 全部不同，no-cache 的两个 disable flag 均为 1，warm 均为 0，before 均为空。单次搜索内的延迟差可能来自计时波动，最终配置质量必须由独立交错测量判断。

## 4. 分母与失败保留

`stock-raw` 的 compile attempts 等于 raw 数，但成功 profile 数可能更少：fwd 的 128 与 bwd 的 stage 3 会在 raw 路径触发已记录的资源/启动失败。Z3 的作用是让相同约束在编译前排除这些配置，失败本身不被改写成 UNSAT。guided 与 budget arms 的 compile/profile 计数按实际选中候选计，不能用 raw 数代替。

每个 arm 的 solver、model、compile、validation、profile 和 end-to-end wall 分别保存。`cache_hit`、cache-only、缺失 trial、重复 trial 和收集到的 pytest 节点都不能直接充当性能证据。预测相关性只连接同一配置 ID 的有限预测与完整 oracle 实测，分母随表输出。这样做的目的很具体：少测候选所省的时间必须与最终配置质量同时成立，任何一侧都不能靠改变分母获得表观收益。
