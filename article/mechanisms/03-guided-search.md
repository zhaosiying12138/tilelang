# Guided search：先证明合法性，再花预算测量候选

> 本节对应 `tilelang/autotuner/space.py` 与 `tilelang/autotuner/guided.py`。它解决的是 AutoTuner 的候选空间问题，不是 kernel 内的 II 调度。文中的 9 个配置是真实 Z3 输入，但不是 GPU benchmark。

## 1. 把搜索过程拆成三个集合

一次可审计的搜索需要区分：

- $R$：用户声明的原始笛卡尔积；
- $U$：去重后，Z3 判定为 SAT 或无法确定的候选集合；
- $S$：实际交给编译器和 profiler 的子集。

`ConfigSpace` 固定每个 domain 的顺序，并用 canonical JSON + SHA-256 生成 `config_id`。同一个配置无论字典插入顺序如何，都得到相同身份；重复配置只记为 duplicate，不计入 UNSAT。

`Z3Pruner` 只从 $R$ 中删除被明确证明不可行的配置。`unknown`、timeout 和资源限制全部保留在 $U$ 中，并在 audit 单列。编译失败也不是 UNSAT：它发生在另一阶段，可能暴露模型遗漏，但不能倒写成求解器已经证明非法。

## 2. 一个真实的容量约束例子

示例空间为：

```text
block_M    = [32, 64, 128]
num_stages = [1, 2, 3]
raw_count  = 9
```

约束为 `block_M * num_stages <= 128`。约束构造器不仅检查当前 Python 值，还把配置值固定到私有 Z3 context 中：

```python
def constraint(config, z3, ctx):
    block = z3.Int("block_M", ctx)
    stage = z3.Int("num_stages", ctx)
    return [
        block == config["block_M"],
        stage == config["num_stages"],
        block * stage <= 128,
    ]
```

`(64,3)、(128,2)、(128,3)` 被证明不可行，剩余 6 个候选。实际日志为：

```text
RAW 9 LEGAL 6 UNSAT 3
```

逐配置 verdict、Z3 statistics 和分母保存在 [guided-demo-pruning.json](../../evidence/G4/guided-demo-pruning.json)。

## 3. 成本模型负责排序，不负责宣称硬件事实

`CandidateCostModel.predict(config)` 返回 `(score, terms)`。`GuidedSearch` 对 $U$ 中每个候选评分，按 `(score, config_id, canonical_json)` 稳定排序。低分优先，所有 score 和分项必须是有限数；模型异常不会被静默转换成“丢弃候选”。

本例使用一个仅用于展示接口的合成公式：

$$score=\frac{100}{block_M}+\frac{1}{num\_stages}$$

实际 top-2 与原始顺序 top-2 为：

```text
GUIDED_TOP2 [
  {'block_M': 128, 'num_stages': 1},
  {'block_M': 64, 'num_stages': 2}
]
RAW_TOP2 [
  {'block_M': 32, 'num_stages': 1},
  {'block_M': 32, 'num_stages': 2}
]
```

所有六个预测及分项保存在 [guided-demo-predictions.jsonl](../../evidence/G4/guided-demo-predictions.jsonl)，标准输出见 [guided-demo.txt](../../evidence/G4/guided-demo.txt)。

这个例子说明 API 如何形成 $S$。它并不证明合成公式适合任何 GPU。正式实验使用经过独立微基准标定的 CUDA 模型，并用完整候选实测检验预测排名；等预算 raw/random 对照则用于回答“收益是否只是少测几个配置”。

## 4. 与默认 AutoTuner 的关系

默认 AutoTuner 仍然编译、校验并实测传入的全部配置。GuidedSearch 只负责产生候选列表和审计记录，不替代真实测量，也不把 predicted score 当成实测 latency。

对照时应同时报告：

1. `stock-raw`：完整 $R$；
2. `z3-legal-exhaustive`：完整 $U$，隔离合法剪枝的作用；
3. `z3-cm-topk`：同一个 $U$，只测量模型 top-k；
4. `budget-raw/random`：同一个 $U$、相同 k 的控制组。

如果 guided 只赢完整穷举的搜索时间，却不赢等预算控制组，结论只能是“减少测量有效”，不能证明成本模型具有独立价值。

## 5. 测试证据

主会话独立执行：

```bash
python -m pytest -q testing/python/autotune/test_guided_search.py
```

结果为 **20 passed**，见 [guided-search-tests.txt](../../evidence/G4/guided-search-tests.txt)。测试覆盖 canonical hash、真实 Z3 耦合约束、SAT/UNSAT/unknown、timeout 保留、duplicate 审计、top-k tie、raw/random 等预算、随机种子复现、JSONL、模型异常和空合法集。

后续章节会给出 AutoTuner 集成代码、逐阶段 wall time，以及 GEMM、Conv、FlashAttention 和 GDN 的真实编译/profile 记录。只有那些设备日志才能支持搜索加速和性能保持的结论。
