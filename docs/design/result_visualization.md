# CUDA Z3 结果可视化：输入合同与解读

## 1. 范围与基本原则

实现位于 `benchmark/z3_cost_model/visualization.py`，提供六个生产绘图函数：

- `plot_schedule_timeline`
- `plot_pruning_funnel`
- `plot_search_time_breakdown`
- `plot_prediction_rank`
- `plot_quality_search_pareto`
- `plot_gdn_acceptance`

这些函数只负责把调用者提供的结果转成图和可审计的数据 sidecar。它们不运行 solver、编译、profile 或 GPU，也不把预测值改写成 confirmed latency。本文只定义数据合同和读图方式，不作性能、加速比、剪枝率或搜索质量声明。

所有数值来自输入；例外仅是图中明确标识的派生量，例如 `start // minimal_II`、rank、Pareto 标记，以及 additive 时间分解的 residual `other`。缺失实验记录不会用零、均值或内置示例补齐。空输入、重复键、矛盾分母、单位冲突和跨搜索空间 oracle 会报错。

## 2. 通用调用与输出合同

从仓库根目录可直接导入：

```python
from benchmark.z3_cost_model.visualization import (
    plot_schedule_timeline,
    plot_pruning_funnel,
    plot_search_time_breakdown,
    plot_prediction_rank,
    plot_quality_search_pareto,
    plot_gdn_acceptance,
)
```

每个 `source` 参数均可为对应 JSON 文件路径或内存中的 JSON-like dict/list。`output` 可以是无扩展名前缀，也可以是本次四种产物之一的文件名。例如下面两种写法具有相同输出前缀：

```python
plot_pruning_funnel("run/summary.json", "artifacts/pruning")
plot_pruning_funnel("run/summary.json", "artifacts/pruning.png")
```

每次成功调用均产生：

| 产物 | 内容 |
| --- | --- |
| `<output>.png` | 180 DPI 位图，适合正文或幻灯片 |
| `<output>.svg` | 矢量图；文字转为 path，减少查看环境字体差异 |
| `<output>.csv` | 图中使用的逐行数据；派生字段有明确列名 |
| `<output>.json` | `schema_version=1`、图类型、指标/单位/警告等元数据以及同一组 `rows` |

返回值为 `{"png": ..., "svg": ..., "csv": ..., "json": ...}` 路径字典。函数先完成输入校验再创建输出目录和文件；不会用只有标题的占位图表示空实验。

绘图仅依赖 matplotlib 和 Python 标准库，不导入 TileLang、TVM、torch、CUDA、SciPy 或 NumPy。无显示环境可设置 `MPLBACKEND=Agg`。字体按 Noto Sans CJK SC、Source Han Sans SC、Microsoft YaHei、SimHei、文泉驿、Arial Unicode MS、DejaVu Sans 顺序 fallback，并关闭 Unicode minus 的兼容问题；当前图内标签使用英文，因此缺少中文字体不会改变数值或字段含义。

## 3. Schedule timeline

### 3.1 调用与原生 schema

```python
plot_schedule_timeline(schedule_info, output)
```

`schedule_info` 首选当前 CUDA scheduler 直接写出的 `schedule_info.json`，即 `tilelang/cuda/transform/z3_scheduler.py` 中 `_save_verbose_artifacts` 的对象。不需要先把数组转成 task dict。

| 字段 | 类型 | 要求与含义 |
| --- | --- | --- |
| `schedule_kind` | string | 必填；原生值 `loop` 或 `straight-line` |
| `start_times` | int array | 必填；每个 task 的开始 cycle，非负 |
| `latencies` | int array | 必填；每个 task 的完成 latency，非负 |
| `iis` | int array | 必填；每个 task 的 issue II，至少为 1 |
| `resource_flags` | int array | 必填；每个 task 的非负 resource bitmask |
| `num_tasks` | int | 原生字段；若存在必须等于四个数组的共同长度 |
| `resource_bits` | int | 原生字段；若存在，mask 必须能放入该位宽；缺失时仅按输入 mask 推断展示位宽 |
| `minimal_II` | int | loop 必填且至少为 1；也接受拼写 `minimal_ii` |
| `makespan` | int | 若存在必须等于 `max(start + latency)`；缺失时由这些输入行计算并标为 `computed_from_rows` |
| `stages` | int array | 可选结果字段；loop 中必须等于逐 task 的 `start // minimal_II` |
| `task_names` | string array | 可选展示名；仍保留稳定的 `T<index>` |
| `fallback` | bool | 原生字段；为 true 时标题明确标注 serial fallback |

四个 task 数组必须等长且至少有一项。`sorted_indices`、依赖、buffer 和求解历史等原生字段可保留在输入中，但本图不据此重算或修改 schedule。

Resource bit 与当前 CUDA latency model 对齐：`1=GLOBAL_LOAD`、`2=GLOBAL_STORE`、`4=SHARED_MEMORY`、`8=TENSOR_CORE`、`16=CUDA_CORE`、`32=SFU`、`64=SCALAR`。多 bit mask 用 `+` 连接；零显示 `NONE`，更高位显示 `BIT<n>`，不会被丢弃。

### 3.2 图与 sidecar 解读

- 每行是一个 task，y 轴同时显示 task 和 resource；横条区间是 `[start, start + latency)`。
- 横条文字同时列出 `start`、latency `L`、逐 task `II` 和 stage `S`。latency 是条宽，II 不是条宽，不能把二者互换。
- loop 的 stage 由 `start // minimal_II` 得到，residue 为 `start % minimal_II`；标题显示最小 II，竖直点线表示 II 边界。边界超过 128 个时省略网格并在图下注明，不省略数据。
- straight-line 原生输出没有 `minimal_II`，stage 也未由该 scheduler 定义，因此显示 `n/a`；不会伪造零 stage 或最小 II。若调用者显式提供合法 `stages`，则按输入展示。
- 标题的 makespan 是最后一个 completion endpoint，不是所有 latency 的简单求和。零 latency task 用竖线标记，条宽仍严格为零。
- CSV/JSON 每行含 `task_index`、`task`、mask/resource、`start`、`latency`、`ii`、`stage`、`residue`、`end`，单位均为 cycles。

## 4. Pruning funnel

### 4.1 计数 schema

```python
plot_pruning_funnel(summary, output)
```

`summary` 可直接为计数字典，也可把这些字段放在 `counts` 对象内。计数必须为非负整数。

| 字段 | 必填 | 定义 |
| --- | --- | --- |
| `raw` | 是且大于 0 | 进入本统计合同的 raw candidate 总数，也是所有比例的共同分母 |
| `sat` | 是 | solver 返回 SAT 的 candidate 数 |
| `unknown` | 是 | solver 返回 unknown 而保留的 candidate 数；不是 UNSAT |
| `unsat` | 是 | solver 证明 UNSAT 后剪去的 candidate 数 |
| `dedup` | 是 | 因重复而去除的 candidate 数；不是 UNSAT |
| `other` | 否，缺省为 0 | 其他显式分类；若用它平衡分母，应在输入中明确提供 |
| `compiled` | 是 | 成功完成 compile 的 candidate 数，不是 compile attempt 数 |
| `profiled` | 是 | 成功完成 profile 的 candidate 数，不是 profile attempt 数 |

绘图前强制满足：

```text
retained = sat + unknown
raw = sat + unknown + unsat + dedup + other
0 <= profiled <= compiled <= retained <= raw
```

因此 raw 中的每个 candidate 必须且只能进入 SAT、unknown、UNSAT、dedup、other 之一。若实验还有未求解或错误类别，导出方应先明确归类并计入 `other`，不能靠缩小 raw 或把 residual 塞进 UNSAT 使图闭合。

### 4.2 图与 sidecar 解读

- 左图是 survivor funnel：raw → SAT/unknown retained → compiled → profiled。第一阶段标签明确说明此处已经过 UNSAT pruning 和 dedup。
- 右图是同一 raw 分母下的互斥 accounting：SAT、unknown、UNSAT、dedup 和 optional other。unknown 与 dedup 使用独立颜色、独立行和独立 role。
- 所有百分比均为 `count / raw`，不会把 compiled/profiled 的比例换成另一个隐含分母。
- `retained` 不是“被剪枝数”；UNSAT pruning 数量只看 `unsat`。compile/profile 尚未发生的候选与失败候选不能仅由本 schema 继续区分，导出方不能把它们改名为 UNSAT。
- CSV/JSON 的 `role` 明确标记 `retained_solver_outcome_not_unsat`、`duplicate_removed_not_unsat`、`proven_pruned` 等语义。

## 5. Search time breakdown

### 5.1 run schema

```python
plot_search_time_breakdown(run_summaries, output)
```

输入为非空 run list，或包含 `runs`/`records` list 的 JSON object。每个 run 为：

| 字段 | 类型 | 定义 |
| --- | --- | --- |
| `run` | string | 必填且跨 run 唯一；用于 x 轴，可包含 suite 和机制名 |
| `timing_semantics` | string | 必填；`additive` 或 `parallel_cumulative` |
| `end_to_end_s` | finite number | 必填、非负；独立测得的 end-to-end wall seconds |
| `times_s` | object | 包含下表各类别的 seconds；也可用对应顶层 `<category>_s` 字段 |
| `times_s.solver` | finite number | 必填、非负 |
| `times_s.model` | finite number | 必填、非负 |
| `times_s.compile` | finite number | 必填、非负 |
| `times_s.validation` | finite number | 必填、非负 |
| `times_s.profile` | finite number | 必填、非负 |
| `times_s.other` | finite number | parallel 必填；additive 可省略 |

也可用 `parallel: true/false` 替代 `timing_semantics`，但两者不能同时出现。`true` 对应 `parallel_cumulative`，`false` 对应 `additive`。

Additive 模式要求：

```text
solver + model + compile + validation + profile + other = end_to_end_s
```

比较使用 `rel_tol=1e-9`、`abs_tol=1e-9` 处理浮点计时舍入。若 additive 输入省略 `other`，函数用 `end_to_end_s - 其余五项` 推导 residual，sidecar 的 `value_origin=derived_residual`；residual 明显为负会报错。若显式提供 `other`，其和不匹配会报错，不会悄悄重算覆盖输入。Parallel 模式不允许推导 residual `other`，因为各类别可能重叠。

### 5.2 图与 sidecar 解读

- 全部 run 都是 additive 时，六个类别画 stacked bars，黑色菱形为独立 end-to-end；stack 与菱形应在允许舍入误差内一致。
- 只要有 run 为 `parallel_cumulative`，所有 run 都改用六组独立 bars，不发 stacked 图；同时发出 `RuntimeWarning`，图中和 JSON `warning` 均说明不能求和。
- 平行累计的某个 component 或 component 总和可以大于 end-to-end，这不自动表示输入错误。图中不展示该总和，也不以它替代 wall time。
- CSV 每行是 `(run, category)`，保留 `timing_semantics`、`seconds`、`end_to_end_s` 和 `value_origin`。Mixed run 集合也统一使用 independent bars，避免同一张图出现两种加法语义。

## 6. Prediction rank

### 6.1 两份对齐输入

```python
plot_prediction_rank(predictions, measured, output, top_k=(1, 5, 10))
```

`top_k` 也可为正整数或正整数序列。它是指标口径，不是 candidate 数据。

Canonical prediction record：

| 字段 | 类型 | 定义 |
| --- | --- | --- |
| `suite` | string | 必填，例如输入方可写 `GDN fwd`、`GDN bwd`；图中保留原名 |
| `candidate` | string/int | 必填；suite 内唯一 candidate ID |
| `predicted` | finite number | 必填；cost 或 predicted latency，数值越低代表预测排名越靠前 |

Canonical measured record 使用同样的 `suite` 和 `candidate`，数值字段为 `measured` 或 `confirmed_latency`，必须有限且非负。也支持常见别名 `prediction`/`score`/`cost` 和 `latency`/`measured_latency`/`actual_latency`，但同一条记录不能同时给多个别名。

也接受嵌套映射：

```text
predictions = {suite: {candidate: predicted_value}}
measured    = {suite: {candidate: measured_value}}
```

嵌套 value 也可以是含 canonical 字段的 object。两份输入的 `(suite, candidate)` 键集合必须完全相同；重复键、只在一边出现的 candidate、空输入均报错，不会通过 inner join 静默改变样本量。

### 6.2 排名与指标定义

每个 suite 独立计算，数值方向均为 lower-is-better：

1. `predicted_rank` 和 `actual_rank` 都从 1 开始；ties 使用 average rank。
2. Spearman ρ 是两组 average ranks 的 Pearson correlation。
3. Kendall 使用 tie-corrected tau-b：`(C-D) / sqrt((n0-n1)(n0-n2))`。
4. `top-k recall = | predicted 最低 k 项 ∩ measured 最低 k 项 | / k`。恰好同值时用 prediction 输入顺序稳定打破边界 tie；sidecar 的 `tie_rule` 保留这一口径。
5. 样本少于 2、相关指标分母为零时，对应相关系数为 JSON `null`、图中 `n/a`。`k > n` 时该 top-k recall 也为 `null`/`n/a`，不会缩小 k 后仍称作原 k。

这些计算由公共纯 Python 函数 `rankdata`、`spearman_rank_correlation`、`kendall_tau_b`、`top_k_recall` 提供。公共 `top_k_recall` 对 `k > n` 抛错；绘图函数仅在 `k <= n` 时调用它。

### 6.3 图与 sidecar 解读

- 每个 suite 一个 panel，标题保留输入 suite 名，x 为 predicted rank，y 为 actual rank；灰色虚线是 rank 完全一致的位置，不是拟合线。
- 点偏离虚线表示 rank 不一致；指标框同时列出 Spearman、Kendall 和各 top-k recall。它们描述不同角度，不应互相替代。
- CSV 每 candidate 一行，包含原 prediction、measured、两种 rank，并重复该 suite 的指标便于筛选。JSON `suites` 另存结构化指标。
- 本图不判断 cost 的绝对单位是否等于 latency 单位，因为只比较顺序；confirmed latency 的非负约束和 candidate 对齐仍严格执行。

## 7. Quality/search Pareto

### 7.1 candidate 与 oracle schema

```python
plot_quality_search_pareto(records, output, x_axis="search_wall_s", oracle=None)
```

`x_axis` 只能是 `search_wall_s` 或 `profile_count`。Candidate record：

| 字段 | 类型 | 定义 |
| --- | --- | --- |
| `suite` | string | 必填；每个 suite 一个 panel |
| `method` | string | 必填；`stock-raw`、`z3-legal`、`z3-cm`、`control` 有固定颜色，其他名字按输入保留 |
| `search_space` | string | 必填；frontier 与 oracle 对齐的空间标识 |
| `candidate` | string/int | 可选 sidecar ID，不影响数值 |
| `search_wall_s` | finite number | 必填、非负；搜索 wall seconds |
| `profile_count` | int | 必填、非负；confirmed profile 次数 |
| `confirmed_latency` | finite number | 必填且大于 0；不能使用 predicted cost 代替 |
| `latency_unit` | string | 必填；同一 suite 内必须一致，例如由输入方明确写 `ms` |

两个候选 x 字段都必须提供，使同一数据 sidecar 可复查并切换 x 轴。Oracle 是独立 list/path，每项包含 `suite`、`search_space`、`oracle_latency` 和 `latency_unit`。同一 `(suite, search_space)` 只能有一个 oracle，且必须至少匹配一个同 suite、同 search_space 的 candidate；unit 也必须一致。没有 oracle 输入就不画 oracle，不会从候选最小值或另一个空间推导。

### 7.2 图与 sidecar 解读

- x 为搜索 wall 或 profile 次数，y 始终为 confirmed latency；两个方向都是越低越靠近左下。
- 每个 `(suite, method, search_space)` 单独计算并连接非支配点。点 `b` 支配 `a` 的条件是：两个维度都不更差，且至少一个维度严格更好。完全相同的两点互不支配。
- 非支配点用实线和实心 marker，支配点用较淡散点；sidecar `is_pareto` 与所选 `x_axis` 对应。切换 x 轴应重新调用函数，不能复用另一口径的 frontier。
- 每个匹配空间的 oracle 用黑色水平虚线并标注空间名。不同 search space 的 candidate 不因同图显示而被合并为一个 frontier。
- CSV 同时包含 `row_kind=candidate` 与 `row_kind=oracle`，不会把没有搜索次数的 oracle 补成零次。JSON 另存 oracle 和 suite unit 元数据。

## 8. GDN acceptance

### 8.1 输入 schema

```python
plot_gdn_acceptance(records, output)
```

每个 record 是某 direction、某 mechanism 的验收汇总：

| 字段 | 类型 | 定义 |
| --- | --- | --- |
| `suite` | string | 必填；可直接为 `GDN fwd`/`GDN bwd`，也可为不含方向的 suite 名 |
| `direction` | string | 当 suite 不含方向 token 时必填；`fwd`/`forward` 或 `bwd`/`backward` |
| `method` | string | 必填；同一 panel 内唯一 |
| `best_latency` | finite number | 必填且大于 0；输入方已确认的 best latency，本函数不再从其他记录取 min |
| `latency_unit` | string | 必填；fwd/bwd 所有记录必须使用同一单位，保证两个 panel 可比较 |
| `compile_count` | int | 必填、非负；输入方定义的 compile attempt 总数 |
| `profile_count` | int | 必填、非负；输入方定义的 confirmed profile 总数 |

也可把方向写在 `direction`，令 `suite="GDN"`；panel 标题此时为 `GDN fwd` 或 `GDN bwd`。Suite 与 direction 同时出现但互相矛盾会报错。两个 direction 都必须至少有一条记录；缺失 mechanism 不补零行，重复 `(panel, method)` 也会报错。

### 8.2 图与 sidecar 解读

- 固定为 fwd、bwd 两个 panel；bar 和左 y 轴是 best confirmed latency，两个 panel 共享 latency 尺度。
- 黑色圆点线和红色方点线分别是 compile/profile count，使用右 y 轴；点旁数字和 CSV 保留输入整数。
- Bar 顶部标签是输入 best latency 的紧凑格式。Latency 和 count 是不同单位，不能把两条 y 轴上的高度直接作数值比。
- 图例只说明机制和口径，不引入 baseline improvement 或 acceptance pass/fail 结论。验收阈值、正确性结论及 count 的采集口径应由实验正文另行给出。
- CSV/JSON 逐行保存 `panel`、`direction`、`suite`、`method`、`best_latency`、`latency_unit`、`compile_count`、`profile_count`，没有默认实验行。

## 9. 测试与正文使用边界

测试位于 `testing/python/cuda/test_result_visualization.py`，只使用内存或临时文件中的 synthetic 数据，检查：

- PNG/SVG/CSV/JSON 文件生成；
- 原生 loop/straight schedule 字段转换；
- 分母和 additive 时间闭合错误；
- parallel 警告与 independent bars 分支；
- rank/tie/top-k 指标；
- Pareto 标记与 same-space oracle；
- GDN 双 panel 和输入数值原样传递；
- 空输入不产生占位实验图。

可在无 GPU 环境运行：

```bash
MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -p no:cacheprovider -q testing/python/cuda/test_result_visualization.py
```

Synthetic 图只写入 pytest 的 `tmp_path`，不随文档发布，不能作为实验结果引用。正文应从真实 run 的 JSON/summary 重新生成图，并同时保留对应 CSV/JSON sidecar，以便核对单位、分母、样本集合和派生字段。本文及测试均不据此声明任何机制的性能优劣。
