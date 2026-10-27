# CUDA Z3 scheduler 与解析成本模型分支

本文件对应公开 fork 的 `cuda-z3-cost-model` 分支。上游基线为 `tile-ai/tilelang@80364b3d9758854a9692cc0f14089025f54d3f3f`；递归 submodule 版本见 [`manifest/upstream.json`](manifest/upstream.json)。实现与实验面向本机 **RTX 5090 Laptop / SM120 / WSL Ubuntu**。其他 GPU 必须重新探测和标定，不能直接复用 SM120 吞吐参数。

## 一、交付范围与边界

分支包含 CUDA 直线/循环 Z3 求解、task latency/II、整 kernel 解析排序、保守配置剪枝、五种策略的 GuidedAutoTuner、显式及自动 pipeline problem、逐 buffer 多版本投影、真实硬件标定，以及 GDN 和经典算子的证据驱动。

以下边界必须在复现和引用结果时保留：

- 默认 CUDA pipeline 与默认 AutoTuner 行为保持不变；新 pass 和 guided arm 都需要显式选择。
- Z3 `unknown`、timeout 和外部分析不确定均保留候选。只有真正的 UNSAT 才能剪枝，编译失败不属于 UNSAT。
- `tilelang/cuda/latency_model.py` 为 kernel 内 task 调度提供 latency/II；`tilelang/cuda/cost_model.py` 为外层 AutoTuner 排序候选。二者与 layout cost model、TVM 通用 Z3 prover 不是同一组件。
- owner exclusion 的 solver 约束已有实现，CUDA owner-ring/runtime synchronization 尚未闭环，adapter 明确拒绝。通用 Let 克隆和多 async producer-stage 也有显式边界，不能宣称任意 Ascend 调度已经端到端等价。
- 自动 task 分析覆盖受限 IR 子集；无法证明的依赖、动态尺寸或不透明副作用会保守保留或明确报错。
- `enable_rasterization` 的 L2 locality 未虚构解析系数。容量估计未知的候选保留并排在有效预测之后。
- 性能数字只来自本机日志。`collected`、compile-only、GDN compilation smoke 和数值正确性是四种不同证据。

逐项状态见 [`docs/design/ascend_z3_cost_model_feature_inventory.md`](docs/design/ascend_z3_cost_model_feature_inventory.md)。

## 二、源码、环境与构建

需要 WSL Ubuntu、Python 3.12、CMake、Ninja 和 CUDA toolkit。当前实机环境为 NVCC/CUDA runtime 13.3、driver 610.74、Torch 2.14.1+cu130、Z3 4.15.4。安装脚本默认使用 `$HOME/miniforge3/envs/tilelang-cu133`；已有其他 conda 环境不会被修改。

```bash
git clone --recursive https://github.com/zhaosiying12138/tilelang.git
cd tilelang
git checkout cuda-z3-cost-model
git submodule update --init --recursive

# 安装独立 Python 环境与构建依赖。
bash scripts/install_wsl.sh

# 导出本仓、TVM Python/native 库及 CUDA 路径。
source scripts/env.sh

# 两并行构建；可用 BUILD_JOBS=1 进一步降低内存压力。
BUILD_JOBS=2 bash scripts/build_tilelang.sh
```

构建配置为 `USE_CUDA=ON`、`USE_ASCEND=ON`、`USE_Z3=ON`、`TILELANG_USE_ASCEND_STUBS=ON`，ROCm/Metal/LLVM 关闭。Ascend stubs 只支持源侧 host 分析，不构成 Ascend 真机验证。

常用测试入口：

```bash
# 49 个 CUDA Z3 solver 测试。
python -m pytest -q testing/python/transform/test_tilelang_cuda_z3_scheduler.py

# latency/II 与整 kernel 成本公式。
python -m pytest -q \
  testing/python/cuda/test_cuda_latency_model.py \
  testing/python/cuda/test_cuda_cost_model.py

# 自动 task 分析、Z3 adapter、stock pipeline 和 buffer-version 联合回归。
python -m pytest -q \
  testing/python/transform/test_tilelang_z3_task_analysis.py \
  testing/python/transform/test_tilelang_transform_z3_pipeline_planning.py \
  testing/python/transform/test_tilelang_transform_pipeline_planning.py \
  testing/python/transform/test_tilelang_transform_Inject_software_pipeline.py \
  testing/python/transform/test_tilelang_transform_pipeline_buffer_versions.py

# Guided search、run/cache 安全和结果图。
python -m pytest -q testing/python/autotune
python -m pytest -q testing/python/cuda/test_result_visualization.py

# 不需要 GPU 的 GDN 配置、搜索、聚合与确认逻辑。
python -m pytest -q \
  benchmark/z3_cost_model/gdn/tests/test_config.py \
  benchmark/z3_cost_model/gdn/tests/test_resource_search.py \
  benchmark/z3_cost_model/gdn/tests/test_autotune.py \
  benchmark/z3_cost_model/gdn/tests/test_confirmation.py \
  benchmark/z3_cost_model/gdn/tests/test_analysis.py \
  benchmark/z3_cost_model/gdn/tests/test_audit_cold.py
```

## 三、硬件探测与标定

`manifest/hardware-sm120.json` 保存设备属性和查询来源，`manifest/calibration-sm120.json` 保存 11 个独立微基准的实测吞吐。模型以 `require_measured=True` 加载，缺失或 placeholder 参数会报错。

```bash
python scripts/probe_cuda_hardware.py --output manifest/hardware-sm120.json

python benchmark/z3_cost_model/calibration/calibrate.py \
  --output manifest/calibration-sm120.json \
  --warmup 10 --repeat 50 --timeout 180 \
  --cache-root evidence/G2/calibration-repro/cache \
  --instruction-kind sass --require-complete
```

标定与正式 shape 分离。GEMM/FMA 还要求 SASS 审核，证据位于 `evidence/G2/instruction-artifacts/`。不要把标定速率当作任一正式候选的实测性能。

## 四、GDN 正式 cold 对照

正式协议固定五个 arm、三次 cold repetition、warmup=10、repeat=50。每次 arm/repetition 都由新的 Python 进程执行，缓存目录、before/after 清单、事件流和环境 PID 均独立。以下命令使用新的输出目录；若目录已存在，应选择另一个名字，不能删除已有证据来腾出路径。

```bash
RUN=evidence/G6/gdn-cold-fresh-repro
NUM_TOKENS=4096
test ! -e "$RUN"

for direction in fwd bwd; do
  for repetition in 0 1 2; do
    GDN_NUM_TOKENS="$NUM_TOKENS" bash scripts/run_gdn_cold_fresh.sh \
      "$direction" "$repetition" "$RUN"
  done
done

python benchmark/z3_cost_model/gdn/audit_cold.py \
  --root "$RUN" --output "$RUN/protocol-audit.json" \
  --num-tokens "$NUM_TOKENS"
python benchmark/z3_cost_model/gdn/analyze_results.py \
  --root "$RUN" --output "$RUN/analysis"
python benchmark/z3_cost_model/gdn/analyze_predictions.py \
  --root "$RUN" --output "$RUN/analysis"
```

`stock-raw` 穷举原始空间；`z3-legal-exhaustive` 穷举合法空间；`z3-cm-topk` 在同一合法空间排序后只测 top-k；`budget-raw` 和 `budget-random` 使用相同候选预算。fwd 为 4 raw / 3 legal / top-2，bwd 为 9 raw / 6 legal / top-3。任何失败候选仍留在事件流与汇总中。

`benchmark/z3_cost_model/gdn/repeat_search.py` 是早期同进程 pilot 入口，保留历史 warmup=2/repeat=5 默认值，只用于复现 pilot。正式 cold 数据必须使用上述 shell 驱动。

T4096 的 `modal-winner` 确认曾暴露等预算对照未达标，因此保留为 validation。修订后的 `candidate-median` 对每种策略分别计算各候选在三次 cold 中的实测中位数，选择最低者，canonical JSON 仅处理并列；不增加候选、不改变 top-k，也不使用确认测量来选择 winner。新的 T8192 held-out 在 [`manifest/gdn-selection-v2-plan.json`](manifest/gdn-selection-v2-plan.json) 中预先登记。复现该 held-out 时设置 `NUM_TOKENS=8192`；早期 `modal-winner` 仍保留用于历史对照。

### 七轮独立确认

确认脚本按各策略候选的三次 cold 实测中位数冻结配置，canonical JSON 只用于并列处理。不同配置按 ABBA 交错，warmup=50、每轮 200 个 L2-flushed CUDA-event 样本，共 7 轮。计时前后都校验公开 forward/backward 输出。分析与确认的 token 形状必须一致，否则在计时前报错。

```bash
CONFIRM=evidence/G6/gdn-confirmation-repro
test ! -e "$CONFIRM"

python benchmark/z3_cost_model/gdn/confirm.py \
  --direction fwd --analysis "$RUN/analysis/analysis.json" \
  --output-dir "$CONFIRM" --selection-rule candidate-median \
  --num-tokens "$NUM_TOKENS"
python benchmark/z3_cost_model/gdn/confirm.py \
  --direction bwd --analysis "$RUN/analysis/analysis.json" \
  --output-dir "$CONFIRM" --selection-rule candidate-median \
  --num-tokens "$NUM_TOKENS"
```

脚本退出码表示执行是否完成；质量验收还要读取 `summary.json` 的 `acceptance.passed` 和逐项 ratio。相同配置按身份相等处理，不构造虚假的两组性能差。全部原始样本在 `samples.jsonl`，GPU 状态、源码 SHA 和缓存清单一并保存。

## 五、warm / no-cache 敏感性

敏感性协议使用 fwd、bwd 两个方向和 `stock-raw`、`z3-cm-topk` 两种策略，每轨每次独立进程。输出目录必须不存在。

```bash
CACHE_RUN=evidence/G6/gdn-cache-sensitivity-repro
bash scripts/run_gdn_cache_sensitivity.sh "$CACHE_RUN"

python benchmark/z3_cost_model/gdn/analyze_results.py \
  --root "$CACHE_RUN" --output "$CACHE_RUN/analysis-warm" \
  --cache-track warm --no-plots
python benchmark/z3_cost_model/gdn/analyze_results.py \
  --root "$CACHE_RUN" --output "$CACHE_RUN/analysis-no-cache" \
  --cache-track no-cache --no-plots
```

聚合器默认只统计 cold。warm 和 no-cache 必须显式指定，不能混入主表分母。cold 表示独立的 TileLang/CUDA JIT 实验缓存起点，不表示整台机器的 OS 缓存被清空。

## 六、经典 TileLang 算子搜索

固定 suite 为 `gemm_square`、`gemm_short_k`、`conv2d_stride_tail` 和 `flash_attention_causal`。主预算均为 top-10，三次 cold repetition；方形 GEMM 另外执行预注册的 top-1/5/20 敏感性。每次子进程只运行一个 suite/arm/repetition，父进程阻塞等待上一个子进程退出后再启动下一个。长批次可以重复 `--arm` 拆成 stock 与 non-stock 两组：

```bash
python benchmark/z3_cost_model/operators/repeat_search.py \
  --suite gemm_square \
  --arm z3-legal-exhaustive --arm z3-cm-topk \
  --arm budget-raw --arm budget-random \
  --repetition 0 --top-k 10 --cache-mode cold \
  --output-dir evidence/G6/classic-nonstock-rep-0
```

主预算均为 top-10，三次 cold repetition；方形 GEMM 另外执行预注册的 top-1/5/20 敏感性。

```bash
CLASSIC=evidence/G6/classic-repro
test ! -e "$CLASSIC"

for suite in gemm_square gemm_short_k conv2d_stride_tail flash_attention_causal; do
  for repetition in 0 1 2; do
    python benchmark/z3_cost_model/operators/repeat_search.py \
      --suite "$suite" --arm all --repetition "$repetition" \
      --top-k 10 --cache-mode cold --warmup 10 --repeat 50 \
      --output-dir "$CLASSIC/$suite-rep-$repetition"
  done
done

for top_k in 1 5 20; do
  for repetition in 0 1 2; do
    for arm in z3-cm-topk budget-raw budget-random; do
      python benchmark/z3_cost_model/operators/repeat_search.py \
        --suite gemm_square --arm "$arm" --repetition "$repetition" \
        --top-k "$top_k" --cache-mode cold --warmup 10 --repeat 50 \
        --output-dir "$CLASSIC/sensitivity-k$top_k-rep-$repetition-$arm"
    done
  done
done

python benchmark/z3_cost_model/operators/analyze_results.py \
  --root "$CLASSIC" --output "$CLASSIC/analysis"
python benchmark/z3_cost_model/operators/audit_cold.py \
  --root "$CLASSIC" --output "$CLASSIC/classic-protocol-audit.json"
```

协议审计要求四个 suite × 五个 arm × 三轮恰好 60 个正式 top-10 run，并检查 60 个不同 PID、空缓存起点、run-owned 路径、预算和成功 profile。预注册的其他 top-k 会单独计为敏感性，不混入 60 个正式分母。

四个 suite 的 cold 搜索完成后，分别执行真实 winner 确认，再做联合分析。确认使用真实 JITKernel、固定输入、7 轮 ABBA、50 warmup 和每轮 200 个 L2-flushed CUDA-event 样本；性能只报告比值和分布，不套用 GDN 的专用通过门槛。

```bash
CONFIRM=evidence/G6/classic-confirmation-repro
test ! -e "$CONFIRM"
for suite in gemm_square gemm_short_k conv2d_stride_tail flash_attention_causal; do
  python benchmark/z3_cost_model/operators/confirm.py \
    --suite "$suite" --analysis "$CLASSIC/analysis/analysis.json" \
    --output-dir "$CONFIRM"
done
python benchmark/z3_cost_model/operators/analyze_confirmation.py \
  --cold-analysis "$CLASSIC/analysis/analysis.json" \
  --confirmation-root "$CONFIRM" \
  --output "$CLASSIC/confirmed-analysis"
```

最佳 GEMM winner 的真实 cubin、SASS 与资源使用可这样导出：

```bash
python scripts/export_gemm_winner.py \
  --confirmation "$CONFIRM/gemm_square/summary.json" \
  --output-dir evidence/G7/best-gemm-repro
/usr/local/cuda-13.3/bin/cuobjdump \
  --dump-resource-usage evidence/G7/best-gemm-repro/kernel.cubin
```

导出目录包含 `source.cu`、`kernel.cubin`、`kernel.sass`、`resource-usage.txt`、配置、compile flags、source/cubin SHA256 和 before/after 清单。`cuobjdump` 的 static shared 字段不代表整个 kernel 的动态 shared 总量。

GEMM 两个空间的 raw/legal 为 288/240，Conv 为 144/120，FA 为 54/54。GEMM 的 48 个动态 shared 超限与版本化公式一一对应，修订后 guided top-10 逐项不变；首批证据保留为 validation。SM120 未猜测 stock atom 规则；这些未编码条件与 solver 返回的 UNKNOWN 分开记录，不能用 unknown=0 推断所有编译合法性都已证明。`compile_attempts`、实际成功 profile、失败候选和 cache-only 结果分别入账。质量比较优先使用完整 legal-oracle；若它不完整，则回退到完整 raw 的实测最优并标记 `quality_reference=stock-raw`。预测相关优先使用完整 legal-oracle，否则只连接 raw 中实际成功的配置，同时标记 `legal_oracle_complete=false` 和实际分母，不把失败候选藏起来。

top-1/5/20 与主 top-10 共用同一 raw/oracle 空间，所以额外预算只运行 guided 和两个等预算控制，并复用同 suite/repetition/cache/seed 的完整 top-10 stock/legal 基线。每行结果都记录 `baseline_top_k=10` 和 `baseline_shared_with_top10=true`，主 top-10 的 oracle 成本单独保留，不能称为免费。若完整基线缺失或空间不一致，相关指标保持 unavailable。协议见 [`manifest/classic-sensitivity-baseline-plan.json`](manifest/classic-sensitivity-baseline-plan.json)。

## 七、广泛编译矩阵

冻结清单为 host 77 行、quick 126 行、full 729 行。`--collect` 只验证 pytest 节点和依赖分类，不能写成执行通过。

```bash
bash scripts/run_compile_matrix.sh \
  --collect --tier full --output evidence/G5/compile-matrix-collect-repro
bash scripts/run_compile_matrix.sh \
  --run --tier quick --output evidence/G5/compile-matrix-quick-repro
```

full 执行可使用 `--only` 指定清单中的精确 row/group/file ID，分批完成后再汇总。当前计划将 728 个默认执行 row 分为 84 个文件、16 个交错批次；每个批次使用独立输出，合并器要求所有 row 恰好执行一次：

```bash
# 99 个 CPU-only rows 单独执行，不与设备 shard 重复。
bash scripts/run_compile_matrix_batch.sh \
  0 1 \
  evidence/G5/full-cpu-shard-plan-v1.json \
  evidence/G5/compile-matrix-full-repro/cpu

# 其余 629 个设备 rows 分 16 批串行执行。
for index in {0..15}; do
  bash scripts/run_compile_matrix_batch.sh \
    "$index" 16 \
    evidence/G5/full-device-shard-plan-v1.json \
    evidence/G5/compile-matrix-full-repro/batches
done

python benchmark/z3_cost_model/merge_matrix.py \
  --tier full --mode default \
  --output evidence/G5/compile-matrix-full-repro/merged \
  evidence/G5/compile-matrix-full-repro/batches/batch-{0..15} \
  evidence/G5/compile-matrix-full-repro/cpu/batch-0
```

独立的 opt-in fixture 不混入 default 分母，单独执行：

```bash
python benchmark/z3_cost_model/compile_matrix.py \
  --run --tier full --mode z3-opt-in \
  --only testing/python/cuda/test_compile_matrix_driver.py \
  --output evidence/G5/compile-matrix-z3-opt-in-repro
```

每行结果区分 `FAIL_LOWER`、`FAIL_DEVICE_COMPILE`、`FAIL_LAUNCH`、`FAIL_NUMERIC`、`SKIP_DEP`、`SKIP_ARCH`、`SKIP_PERF` 和真实 pass。`SKIP_PERF` 只用于默认关闭、需要显式 `--run-perf` 的性能测试，与依赖或架构 skip 一样不计为通过。GDN smoke 不计入六梯度数值验收。

## 八、资源与缓存纪律

- 每个 GPU 阶段前检查 `free -g`；可用主机内存低于 16 GiB 不启动。`nvidia-smi` 确认没有其他计算进程，全局最多一个后台 GPU 任务。
- 禁止 nsys 和注入式 profiler。性能测量使用 CUDA event 或 TileLang Profiler。
- 只使用本次 run 明确拥有的缓存目录；不删除 `~/.tilelang`、其他项目、共享 CUDA/XDG 缓存或未知目录。
- 不根据结果删除算子、失败候选或重复轮次。早期失败和协议修正与最终结果一并保留。
- 原始证据中的 `cache/` 是运行产物，不作为源码构建缓存提交。公开日志、JSONL、CSV、图表和 SHA256 manifest 用于独立重算正文数字。

文章章节目录由 [`article/index.json`](article/index.json) 管理；机制说明位于 `article/mechanisms/`，实测结果位于 `article/results/`。

## 九、文章与证据打包

以下命令生成自包含 HTML、知乎 Markdown、公式清单、逐文件 SHA256 manifest 和文章资源 ZIP。构建只读取明确引用的章节、图片与附件，不会把缓存目录隐式打入文章包。

```bash
python article/build_html.py \
  --config article/index.json \
  --output article/_build/tilelang-cuda-z3.html \
  --zhihu-output article/_build/tilelang-cuda-z3-zhihu.md \
  --formula-log article/_build/formulas.json \
  --manifest-output article/_build/article-manifest.json \
  --zip article/_build/tilelang-cuda-z3-article.zip
```

完整源码和已提交原始证据在最终 commit 后另存一份快照。它包含正式 trials、分析 JSON/CSV、截图和矩阵日志压缩包，不包含被忽略的 TileLang/CUDA/Torch 缓存或本地临时文件。

```bash
git archive --format=zip --prefix=tilelang-cuda-z3/ \
  --output article/_build/tilelang-cuda-z3-repository-evidence.zip HEAD
cd article/_build
sha256sum tilelang-cuda-z3.html tilelang-cuda-z3-zhihu.md \
  tilelang-cuda-z3-article.zip tilelang-cuda-z3-repository-evidence.zip \
  > SHA256SUMS
```

`article/_build/` 是生成目录，不作为源码提交。公开 fork 只包含源码、文章源文件和已提交证据；上述命令不在公网发布文章，也不向上游创建 PR。
