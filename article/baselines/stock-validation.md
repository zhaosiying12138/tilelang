# Stock 验证基线：后续所有比较从这里开始

## 1. 统计口径

本项目的测试记录分为四类：host-only 调度/变换、设备编译、设备数值正确性、性能测量。`collected` 只表示 pytest 能收集到用例；`compile-only` 不证明数值正确；既有 GDN compilation smoke 也不自动等于六梯度校验。性能章节只引用通过相应正确性门禁后的结果。

下表是开始 CUDA Z3 pipeline 集成前及支撑层完成后的真实基线。各集合有重叠，不能把行数简单相加成一个“总通过率”。

| 集合 | 真实结果 | 证据与含义 |
|---|---:|---|
| Ascend auto_schedule host | **706 passed，31 deselected** | [日志](../../evidence/G0/ascend-host-tests.txt)；无 Ascend 设备声明 |
| CUDA pipeline host | **80 passed** | [日志](../../evidence/G0/cuda-pipeline-host-tests-retry.txt)；现有 PipelinePlanning/InjectSoftwarePipeline |
| stock FP16 GEMM cold smoke | **1 passed** | [日志](../../evidence/G0/cuda-gemm-smoke.txt)；独立空 cache |
| GEMM + SM120 TMA | **18 passed，2 skipped** | [日志](../../evidence/G0/broad-stock-gemm-sm120.txt)；多 dtype/layout 和真实 SM120 runtime |
| elementwise/copy/reduce/norm/dynamic | **169 passed，3 failed** | [日志](../../evidence/G0/stock-elementwise-reduce-norm-dynamic.txt)；3 个失败均为 stock lowering |
| CUDA Z3 solver | **49 passed** | [日志](../../evidence/G1/solver-tests.txt)；CPU 约束与调试机制 |
| Guided search | **20 passed** | [日志](../../evidence/G4/guided-search-tests.txt)；纯搜索空间与排名 |
| Hardware probe | **8 passed + 真实 JSON 验证** | [日志](../../evidence/G2/hardware-spec-tests.txt)；不是带宽标定 |
| Task latency/II | **28 passed** | [日志](../../evidence/G2/latency-model-tests.txt)；公式与 fallback，不是设备性能 |
| 可视化工具 | **15 passed** | [日志](../../evidence/G7/visualization-tests.txt)；synthetic 只用于工具测试，不入实证图 |

## 2. 原始 pipeline 基线

运行命令为：

```bash
python -m pytest -q \
  testing/python/transform/test_tilelang_transform_pipeline_planning.py \
  testing/python/transform/test_tilelang_transform_Inject_software_pipeline.py
```

结果为 80 passed。第一次执行在收集阶段失败，因为 `CUDA_VISIBLE_DEVICES=""` 使测试文件顶层的 `determine_target("auto")` 无法找到目标。保留 [原始错误日志](../../evidence/G0/cuda-pipeline-host-tests.txt) 后，允许目标检测并复跑成功。这个差异属于测试环境，不是 pipeline 修复，也不计入优化效果。

后续 default-off 需要证明：未显式启用 Z3 时，solver 调用数为 0，输入 IR 不被新 pass 改写，并继续通过这组原始测试。opt-in 则必须另外证明 solver 确实被调用且 `fallback=False`，不能仅凭 kernel 最终能运行就宣称 Z3 已生效。

## 3. 设备侧 GEMM 与 SM120

完整 GEMM kernel 文件与 SM120 TMA shared-memory 对齐回归在同一冷缓存运行目录中执行，得到 18 passed、2 skipped。测试覆盖 FP16/BF16/FP32/INT8/FP64、不同 transpose、非整 tile、operand layout，以及 SM120 TMA 对齐 codegen 和 host-arch runtime。

这组结果说明 stock 源码能在本机编译并运行代表性 GEMM。它不证明后续 Z3 pipeline 正确，更不支持“性能提升”；优化章必须另列搜索空间、配置和计时。

## 4. 广泛算子中的三个 stock 失败

广泛集合包含 elementwise、copy、reduce、RMSNorm 和动态 shape。最终结果为 169 passed、3 failed、62 warnings。三个失败均为 `FAIL_LOWER`，不是数值误差：

| 用例 | stock 错误 |
|---|---|
| `test_reduce[sum-float16-64x128-f2f-t256-b4]` | `batch=4` 超过每线程输出元素数 `N=2` |
| `test_reduce[sum-bfloat16-64x128-f2f-t256-b4]` | 同上 |
| `test_reduce[min-float16-128x128-f2f-t256-b8]` | `batch=8` 超过每线程输出元素数 `N=4` |

完整错误和上下文在日志中，机器可读分类见 [stock-operator-summary.json](../../evidence/G0/stock-operator-summary.json)。这些用例保留在回归矩阵中。后续若 default 与 opt-in 都以同样方式失败，应记为上游基线未修复；若其中一方改变失败类别，需要单独分析，不能用删除用例制造改善。

## 5. 支撑层结果不能外推

49 个 Z3 测试证明求解器语义，不证明它已接入 CUDA pipeline；20 个 guided-search 测试证明候选空间和排序，不证明 GPU 搜索时间下降；28 个 latency 测试使用合成校准验证公式，不是 5090 吞吐测量；硬件探针只证明属性来源可靠；可视化测试只证明图与数据一致。

因此，后续每一个新机制都必须同时给出：实现位置、针对性测试、真实执行日志和适用边界。性能结论还必须提供同空间、同输入、同缓存口径的对照，以及 GDN 前向和反向的完整公开管线结果。
