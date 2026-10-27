# 广泛编译矩阵：覆盖范围与证据分层

## 1. 冻结分母与执行层级

矩阵固定为 full 729 行：651 个 operator rows、77 个 host rows 和 1 个 Z3 opt-in fixture。quick 为 126 行，host tier 为 77 行。`--collect` 只验证 pytest 节点存在和依赖分类，不能写成执行通过。

| 层级 | 允许证明的内容 | 不允许外推的内容 |
|---|---|---|
| H0 | host/frontend/IR 结构检查 | CUDA device compile 或数值正确性 |
| H1 | lowering、codegen、device compilation | 输出与独立 reference 一致 |
| G | GDN compilation/launch smoke | 六个梯度或 fwd 数值正确性 |
| P | reference-checked numerical parity | 仅因 kernel 能启动就通过 |

失败按 `FAIL_LOWER`、`FAIL_DEVICE_COMPILE`、`FAIL_LAUNCH`、`FAIL_NUMERIC`、`FAIL_ENV`、`FAIL_INFRA` 分类；`SKIP_DEP`、`SKIP_ARCH` 与默认未启用性能测试的 `SKIP_PERF` 不记为通过。NVRTC 错误归入 device compile，动态 shared-memory 设置失败归入 launch，避免误写为 lowering 失败。

## 2. 已完成的 host 与 CPU-only 证据

host tier 的 77 行均实际执行并通过，合计 130 个 pytest case。结果见 [host summary](../../evidence/G5/compile-matrix-host-run-v1/summary.json)。它覆盖 host/frontend/IR，不代替设备执行。

full 中另外 99 个 `requires_device=false` rows 已全部通过，合计 332 个 pytest case，没有失败。结果见 [CPU-only summary](../../evidence/G5/compile-matrix-full-cpu-files-v1/summary.json)。这 99 行使用独立 shard，后续 629 个设备 rows 使用另一份计划，合并时不能重复计数。

| 集合 | 实际执行 | compile-only 通过 | 数值通过 | 失败 |
|---|---:|---:|---:|---:|
| host tier | 77 | 77 | 0（无 P 行） | 0 |
| full CPU-only | 99 | 99 | 0（无 P 行） | 0 |

两组测试没有设备性能含义，也不证明 Z3 新 pipeline 的每个 feature 都已覆盖。84 项 parity 仍需按各自 ID 和直接 oracle 核销。

## 3. 分批执行与缓存身份

默认 full 有 728 个执行 row，另有 1 个专用 opt-in fixture。执行计划覆盖 84 个文件，其中 74 个设备文件、10 个 CPU-only 文件。设备部分分 16 个交错批次，CPU-only 部分独立执行。每个 pytest node 都在独立子进程中运行，使用独立 TileLang/CUDA/Torch/XDG/TMP 缓存和 source/manifest SHA。

```bash
bash scripts/run_compile_matrix_batch.sh \
  "$index" 16 \
  evidence/G5/full-device-shard-plan-v1.json \
  evidence/G5/compile-matrix-full-repro/batches
```

合并器只接受所有应执行 row 恰好出现一次：重复、缺失、manifest/source 不一致、nonempty 输出目录均失败。它不选择多次运行中的最佳结果，也不把 `NOT_RUN` 或 `COLLECTED` 自动转成 pass。

独立的 opt-in fixture 使用 `testing/python/cuda/test_compile_matrix_driver.py::test_z3_opt_in_pass_configs_compile_fixture`，验证 pass config 真正进入编译，而不是只存在 Python 参数中。该 fixture 不混入 default 分母。

## 4. 完整执行结果：覆盖完成不等于全部通过

合并器接受 16 个设备 shard 和既有 CPU-only shard，所有应执行 row 恰好出现一次。最终 `complete=true`、`selected_not_run=0`，默认选中 728 行，另一个 opt-in fixture 独立执行。下表按真实 scope 分开计数；full 是 device 与 CPU-only 的并集，host tier 和 opt-in 不再加进 729 的分母。

| 集合 | 分母 | 默认选中 | attempted | compile-only 通过 | 数值通过 | G smoke（非数值） | skip | 失败 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| host tier（独立） | 77 | 77 | 77 | 77 | 0 | 0 | 0 | 0 |
| full CPU-only | 99 | 99 | 99 | 99 | 0 | 0 | 0 | 0 |
| full device | 629 | 629 | 599 | 240 | 272 | 7 | 82 | 28 |
| **full 合计** | **729** | **728** | **698** | **339** | **272** | **7** | **82** | **28** |
| Z3 opt-in（独立） | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |

full 的 611 个通过行是 339 个 compile-only 与 272 个 numeric 之和；7 个 GDN smoke 不计入数值通过。82 个 skip 包括 74 个 `SKIP_ARCH`、5 个 `SKIP_DEP` 和 3 个默认未启用性能测试的 `SKIP_PERF`，都不是 pass。数值门禁共有 346 个 eligible row，272 个通过、74 个未满足；后者还包括失败与 skip，不能全部称为数值错误。

### 4.1 失败分类

| 状态 | 行数 | 含义 |
|---|---:|---|
| FAIL_LOWER | 17 | lowering、前端或结构性检查失败 |
| FAIL_DEVICE_COMPILE | 6 | NVRTC/device compilation 失败 |
| FAIL_LAUNCH | 3 | 启动或动态 shared-memory 等 launch 失败 |
| FAIL_ENV | 2 | 必需依赖或 import 环境失败 |
| FAIL_NUMERIC | 0 | 没有归入数值不一致的行 |
| FAIL_INFRA | 0 | 最终分类后没有未解释的 infra 失败 |
| **合计** | **28** | **全部保留，不重跑挑选较好结果** |

`FAIL_NUMERIC=0` 不表示所有数值门禁都通过：numeric-gate 的 74 个未满足行中，28 个是上述失败，其余包含架构、依赖或性能 skip。`pytest_case_totals` 也原样保留在 summary 中，但参数化/Hypothesis 展开与 grouped test 和 manifest row 不是同一分母，正文不把 case 数转换成 row 通过率。

全部 28 个失败及其 node、层级、原因和日志路径见 [failures.csv](../../evidence/G5/compile-matrix-full-repro/merged/failures.csv)；84 个文件的逐项账目见 [file-summary.csv](../../evidence/G5/compile-matrix-full-repro/merged/file-summary.csv)，完整 729 行见 [results.jsonl](../../evidence/G5/compile-matrix-full-repro/merged/results.jsonl)。分母规范化后的 full/device/CPU/host/opt-in 对照见 [scoped-summary.json](../../evidence/G5/compile-matrix-full-repro/merged/scoped-summary.json)。

### 4.2 独立 opt-in fixture

`testing/python/cuda/test_compile_matrix_driver.py::test_z3_opt_in_pass_configs_compile_fixture` 以固定 `sm_80` target 真正调用 `tilelang.compile`，确认新 pass config 进入编译且生成非空源码。结果为 1/1 `PASS_COMPILE_ONLY`，没有设备 launch，也不构成 SM80 或 SM120 数值验证。它不计入默认 728 行，结果见 [opt-in summary](../../evidence/G5/compile-matrix-z3-opt-in-v1/summary.json)。

## 5. 分类修订与原始失败保留

首批执行暴露了三类 runner 口径问题：默认关闭的性能测试被留成 `NOT_RUN`；全部 skip 的架构/ROCm 节点因 pytest exit 5 被记成 infra；`fla.ops` 缺失的 collection import failure 需要与 infra 区分。修正后，`SKIP_PERF` 独立计数，明确的 compute-capability/backend skip 记为 `SKIP_ARCH`，必需依赖 import 失败记为 `FAIL_ENV`。

原始批次仍位于 `batches/batch-0/1/2/7`，中间 retry 也未删除；最终合并只使用 `classification-final/batch-0/1/2/7` 和其余正式批次。冻结 manifest、测试节点和源码 SHA 未改变。真实日志回放覆盖六种 arch/perf skip 和两种环境失败，分类、协议与日志打包联合回归为 17 passed。第 7 批的 `fla` 顶层包虽可 import，但 `fla.ops` 不存在且 manifest 将其列为必需依赖，因此保留两个 `FAIL_ENV`，没有擅自安装另一套 FLA 或降级 Torch。

合并所用 17 个 shard 的真实路径、run ID、source SHA 和 manifest SHA 均在 [merged run.json](../../evidence/G5/compile-matrix-full-repro/merged/run.json) 中。`complete=true` 只证明每个应执行 row 已到达可合并终态，不能把 28 个失败或 82 个 skip 改写成通过。

## 6. 原始日志与结果阅读原则

完整日志包保留原始、retry 和 final 路径，cache 目录被排除；ZIP 内 `logs-manifest.json` 逐文件记录字节数和 SHA256，并在写入时检查源文件是否变化。

| 日志包 | 文件数 | 未压缩字节 | 内容 |
|---|---:|---:|---|
| [compile-matrix-logs.zip](../../evidence/G5/compile-matrix-full-repro/compile-matrix-logs.zip) | 32,098 | 4,705,526 | 默认矩阵全部 stdout/stderr/driver 日志 |
| [opt-in-logs.zip](../../evidence/G5/compile-matrix-z3-opt-in-v1/opt-in-logs.zip) | 1,458 | 23,979 | 独立 opt-in fixture 日志 |

真实 P 行才进入 `PASS_NUMERIC`，compile-only、G smoke、dependency/arch/perf skip 和 collected 均不冒充数值通过。没有进入性能实验的 compile-only 行不用于证明 AutoTuner 的时间或配置质量；性能结论仍来自 GDN 与四个正式经典 workload 的独立搜索和确认。广泛矩阵说明默认路径的覆盖范围及现有失败，不补齐 84 项 parity 中明确开放的编译器语义缺口。
