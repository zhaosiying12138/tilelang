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

## 4. 结果阅读原则

后续 full 表会同时列出 selected、attempted、compile-only、numeric、G smoke、dependency/architecture skips 和各类失败。真实 P 行才进入 `PASS_NUMERIC`。任何由依赖缺失产生的 skip 都保留具体包名和版本关系，不通过降级 Torch 或安装另一套 flash-attention/fla 来制造通过。

原始 `results.jsonl`、每个 node 的 stdout/stderr、run metadata、source SHA 和缓存清单用于重算正文。没有进入性能实验的 compile-only 行不会用于证明 AutoTuner 的时间或配置质量；性能结论仍来自 GDN 与四个正式经典 workload 的独立搜索和确认。
