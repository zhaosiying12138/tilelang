# 附录：证据索引与复现入口

## 1. 调度与设备正确性

正文使用的小数推导和受限设备案例可由以下原始记录交叉核对。日志中的 SAT 只覆盖约束求解，数值结论对应各自设备测试。

- 四任务 A+B、公开 Task/Stage、scalar clone 与 multi-async：[SM120 集成日志](../../evidence/G3/z3-pipeline-sm120-v6.txt)；multi-async 的 CPU 端同步与非法输入检查见 [multi-async-host-v2.txt](../../evidence/G3/multi-async-host-v2.txt)。
- 两 owner、两 role ring：[owner-sm120-v6.txt](../../evidence/G3/owner-sm120-v6.txt)；测试源码为 [`test_tilelang_cuda_z3_owner_ring_integration.py`](../../testing/python/language/test_tilelang_cuda_z3_owner_ring_integration.py)。
- 任务成本公式：[latency-model-tests.txt](../../evidence/G2/latency-model-tests.txt)。这些是合成输入的公式检查，不提供设备吞吐结论。
- 早期边界与失败仍保留：[owner 编译容量失败](../../evidence/G3/owner-sm120-v1.txt)、[owner 调用接口失败](../../evidence/G3/owner-sm120-v2.txt)、[clone 早期流水生成失败](../../evidence/G3/z3-pipeline-sm120-v1.txt) 和 [multi-async 旧 guard 失败](../../evidence/G3/device-integration-test.txt)。后续受限用例通过没有改写这些历史记录。

flat 主链源码入口为 [`schedule_hint.py`](../../tilelang/cuda/language/schedule_hint.py)、[`normalize_z3_schedule_units.cc`](../../src/transform/normalize_z3_schedule_units.cc)、[`z3_scheduler.py`](../../tilelang/cuda/transform/z3_scheduler.py) 和 [`z3_pipeline_planning.cc`](../../src/transform/z3_pipeline_planning.cc)。

## 2. 硬件与标定

- 正式实验的设备属性：[hardware-sm120.json](../../manifest/hardware-sm120.json)；模型消费的标定视图：[calibration-sm120.json](../../manifest/calibration-sm120.json)。
- 独立微基准汇总：[calibration-summary.json](../../evidence/G2/calibration-summary.json)。测量协议、单位换算、正确性与指令审核入口见 [`benchmark/z3_cost_model/calibration/README.md`](../../benchmark/z3_cost_model/calibration/README.md)。
- 新设备应重新运行 [`calibrate.py`](../../benchmark/z3_cost_model/calibration/calibrate.py)，并同时更新硬件与标定输入。只替换显卡名称不构成新设备标定。

## 3. GDN held-out 与确认

- 三轮 cold 五臂汇总：[summary.csv](../../evidence/G6/gdn-heldout-t8192-v1/analysis/summary.csv)；预测配对与 oracle rank：[prediction-rank.json](../../evidence/G6/gdn-heldout-t8192-v1/analysis/prediction-rank.json)；进程和缓存协议：[protocol-audit.json](../../evidence/G6/gdn-heldout-t8192-v1/protocol-audit.json)。
- 独立确认：[confirmed-summary.csv](../../evidence/G6/gdn-heldout-t8192-confirmation-v1/analysis/confirmed-summary.csv)；七轮配对差值：[gdn-paired-differences.csv](../../evidence/G6/gdn-heldout-t8192-confirmation-v1/analysis/gdn-paired-differences.csv)。
- 资源公式核对：[forward shared-memory](../../evidence/G5/gdn-fwd-shared-memory.json) 与 [prepare_h shared-memory](../../evidence/G5/gdn-prepare-h-shared-memory.json)。
- validation 与异常记录：[T4096 旧选择规则失败](../../evidence/G6/gdn-confirmation-v2/bwd/summary.json)、[backward 原始全配置记录](../../evidence/G5/gdn-bwd-all-configs.txt) 和 [后续 1000 次重复校验](../../evidence/G5/gdn-bwd-stage1-group8-determinism.txt)。
- 缓存敏感性仅作辅助解释：[no-cache summary](../../evidence/G6/gdn-cache-sensitivity-v1/analysis-no-cache/summary.csv) 与 [warm summary](../../evidence/G6/gdn-cache-sensitivity-v1/analysis-warm/summary.csv)。

驱动协议与参数说明见 [`benchmark/z3_cost_model/gdn/AUTOTUNE.md`](../../benchmark/z3_cost_model/gdn/AUTOTUNE.md)，正式 cold 入口为 [`scripts/run_gdn_cold_fresh.sh`](../../scripts/run_gdn_cold_fresh.sh)。

## 4. 经典算子与预测失误

- 搜索空间、独立参考和必要资源条件：[`benchmark/z3_cost_model/operators/README.md`](../../benchmark/z3_cost_model/operators/README.md)。
- 三轮 cold 主分析：[summary.csv](../../evidence/G6/classic-v2/main-analysis/summary.csv)；top-k 敏感性：[runs.csv](../../evidence/G6/classic-v2/sensitivity-analysis/runs.csv)。
- 独立 winner 确认：[confirmed-summary.csv](../../evidence/G6/classic-v2/confirmed-analysis/confirmed-summary.csv) 与 [confirmation-comparisons.csv](../../evidence/G6/classic-v2/confirmed-analysis/confirmation-comparisons.csv)。

阅读这些 CSV 时，应分别保留 raw、legal、selected、compile attempt 和成功 profile 的分母。FlashAttention 的 `legal_oracle_complete=false` 也应保留，不能把合法穷举写成完整实测参照。

## 5. 已接受捕获包

捕获验收状态与窗口绑定方法见 [capture-acceptance-v1.json](../../evidence/G7/screenshots/capture-acceptance-v1.json) 和 [capture-manifest.json](../../evidence/G7/screenshots/capture-manifest.json)。下列图片仅作为附件链接，不嵌入正文；性能数字仍以对应 CSV/JSON 为准。

- stock/guided 仅分析复算：[裁图][package:stock-guided-crop]、[全屏原图][package:stock-guided-original]、[执行日志](../../evidence/G7/screenshots/tilelang-cuda-z3-stock-guided-v3.log)、[capture metadata](../../evidence/G7/screenshots/tilelang-cuda-z3-stock-guided-v3.json)、[crop metadata](../../evidence/G7/screenshots/tilelang-cuda-z3-stock-guided-v3.crop.json)、[窗口绑定](../../evidence/G7/screenshots/tilelang-cuda-z3-stock-guided-v3.png.window-binding.json)、[analysis.json](../../evidence/G7/screenshots/stock-guided-analysis-v3/analysis.json) 与 [summary.csv](../../evidence/G7/screenshots/stock-guided-analysis-v3/summary.csv)。它复算既有事件，没有新增 GPU 搜索样本；原始事件来自 [stock.log](../../evidence/G6/gdn-stock-guided-screenshot-v1/stock.log) 与 [guided.log](../../evidence/G6/gdn-stock-guided-screenshot-v1/guided.log)。
- GDN 独立确认汇总：[裁图][package:gdn-confirmation-crop]、[全屏原图][package:gdn-confirmation-original]、[执行日志](../../evidence/G7/screenshots/tilelang-cuda-z3-gdn-confirmation-v1.log)、[capture metadata](../../evidence/G7/screenshots/tilelang-cuda-z3-gdn-confirmation-v1.json)、[crop metadata](../../evidence/G7/screenshots/tilelang-cuda-z3-gdn-confirmation-v1.crop.json)、[窗口绑定](../../evidence/G7/screenshots/tilelang-cuda-z3-gdn-confirmation-v1.png.window-binding.json)、[analysis.json](../../evidence/G7/screenshots/gdn-confirmation-analysis-v1/analysis.json) 与 [confirmation-comparisons.csv](../../evidence/G7/screenshots/gdn-confirmation-analysis-v1/confirmation-comparisons.csv)。
- 环境样张：[裁图][package:sample-crop]、[全屏原图][package:sample-original]、[原始日志](../../evidence/G7/screenshots/tilelang-cuda-z3-sample-v3.log) 与 [窗口绑定](../../evidence/G7/screenshots/tilelang-cuda-z3-sample-v3.png.window-binding.json)。它只证明可见命令和终端环境。
- 方形 GEMM winner 资源证据：[终端图][package:best-gemm-capture]、[resource-usage.txt](../../evidence/G7/best-gemm-v1/resource-usage.txt)、[metadata.json](../../evidence/G7/best-gemm-v1/metadata.json)、[kernel.sass](../../evidence/G7/best-gemm-v1/kernel.sass) 与 [终端日志](../../evidence/G7/best-gemm-v1/tilelang-cuda-z3-best-gemm-v3.log)。其中 static shared 字段不能解释为完整动态共享内存总量。

早期未接受捕获仍留档：[第一次失败](../../evidence/G7/screenshot-failure-v2/capture-failure.json) 与 [第二次失败](../../evidence/G7/screenshot-failure-v3/capture-failure.json)。全屏原图和裁图通过 `package:` 声明收入文章 ZIP，正文 HTML 不重复嵌入这些图片。

[package:stock-guided-crop]: ../../evidence/G7/screenshots/tilelang-cuda-z3-stock-guided-v3.png
[package:stock-guided-original]: ../../evidence/G7/screenshots/tilelang-cuda-z3-stock-guided-v3-original.png
[package:gdn-confirmation-crop]: ../../evidence/G7/screenshots/tilelang-cuda-z3-gdn-confirmation-v1.png
[package:gdn-confirmation-original]: ../../evidence/G7/screenshots/tilelang-cuda-z3-gdn-confirmation-v1-original.png
[package:sample-crop]: ../../evidence/G7/screenshots/tilelang-cuda-z3-sample-v3.png
[package:sample-original]: ../../evidence/G7/screenshots/tilelang-cuda-z3-sample-v3-original.png
[package:best-gemm-capture]: ../../evidence/G7/best-gemm-v1/tilelang-cuda-z3-best-gemm-v3.png
