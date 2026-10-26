# GDN 正确性：先让公开前向和反向都值得测量

## 1. 验收对象是完整公开管线

本工作的 GDN 标的来自 `QwenLM/FlashQLA@da06429`（v0.1.3，MIT），不是用 Triton 或 MLIR 重写的替代品。测试通过公开 `chunk_gated_delta_rule_fwd` 和 `chunk_gated_delta_rule_bwd` 执行：

- forward：gate cumsum、KKT/solve、CP preprocessing、fused forward，检查 `o` 和 `final_state`；
- backward：prepare_h、fused backward、GQA 的 dq/dk reduction、reverse gate cumsum，检查 `dq、dk、dv、db、dg、dh0` 六类梯度。

参考实现是 vendored 纯 PyTorch float64 路径，判定使用相对 L2 `RTOL=0.02`。核心输入使用 BF16、`K=DV=128`、chunk 32、`auto_cp=False`，并保留原 512 线程和 mbarrier 协议。搜索不改变这些数学与同步前提。

## 2. 一个必须修复的启动问题

backward 首先需要 public forward 的产物。如果直接沿用 forward 的 `block_DV=128`，会在进入 fused backward 之前失败：

```text
Failed to set the allowed dynamic shared memory size to 112640
```

因此 backward 的 forward artifacts 固定使用已验证可运行的 `block_DV=64`。它不是反向搜索维度，也不会被计入 stage×group 的 9 个配置。该适配只解决启动问题，反向性能范围仍从 prepare_h 和 fused backward 开始。

## 3. `dh0` 全零不是容差问题

第一次 h0/no-dht 校验中，其他梯度已通过，`dh0` 却出现：

```text
DH0_ACTUAL_NORM 0.0
EXPECTED_NORM 7.52098297787412
```

源码显示，`dh_fragment_0/1` 在 `use_dht=False` 时也会被清零并继续参与后续计算，但 upstream 只在 `use_dht=True` 时写回 `dh0`。也就是说，计算已经进行，返回值却被漏写。

修复只改变写回条件：

```python
# dht 只决定终端梯度输入是否为零；
# 已经计算的 dh0 仍必须写回。
if state_v_first:
    for j_v, j_k in T.Parallel(DV, DK // 2):
        dh0[bb, bh, j_v, j_k] = dh_fragment_0[j_v, j_k]
        dh0[bb, bh, j_v, DK // 2 + j_k] = dh_fragment_1[j_v, j_k]
else:
    for j_k, j_v in T.Parallel(DK // 2, DV):
        dh0[bb, bh, j_k, j_v] = dh_fragment_0[j_k, j_v]
        dh0[bb, bh, DK // 2 + j_k, j_v] = dh_fragment_1[j_k, j_v]
```

修复后，dense、h0/no-dht、padded 三类用例全部通过。日志见 [gdn-bwd-correctness-v3.txt](../../evidence/G5/gdn-bwd-correctness-v3.txt)。没有增大容差，也没有跳过该输出。

## 4. 配置级结果与偶发失败的处理

| 检查 | 真实结果 |
|---|---|
| Forward 16/32/64 | dense、h0/no-dht、padded 全通过 |
| Forward 128 | launch 时 shared-memory 112640 B，失败 |
| Backward 默认 stage 2/group 16 | 三组核心用例与六梯度全通过 |
| Backward 9 配置首轮 | 5 通过，3 个 stage 3 容量失败，1 个 stage1/group8 出现 `dv` 不一致 |
| stage1/group8 复测 | 同输入三个 group 的相对误差均约 0.38%，通过 |
| stage1/group8 重复验证 | **1000 次公开 backward 全部通过** |

`dv` 不经过 GQA group-reduce，因此不能把首轮错误简单归因于 group tile。当前证据不能证明存在稳定算法错误，也不允许用一次复测掩盖竞态风险。文章保留 [首轮 9 配置日志](../../evidence/G5/gdn-bwd-all-configs.txt) 和 [1000 次重复日志](../../evidence/G5/gdn-bwd-stage1-group8-determinism.txt)，后续最终入选配置仍要做独立重复验证。

## 5. 为什么先花这些工作量

如果 `dh0` 仍错误，任何“更快搜索”都没有意义；如果默认 tile 不能启动，暴力 AutoTuner 会把大量时间花在不可运行配置上；如果 stage 3 未识别为容量非法，成本模型可能在不可执行解上排序。正确性、资源可行性和搜索效率必须按这个顺序建立。

后续性能章节会在这些门禁之上比较搜索次数和真实延迟。`dh0` 修复是正确性适配，不计入 Z3 的性能贡献；容量剪枝才是搜索成本的一部分，必须计入端到端对照。
