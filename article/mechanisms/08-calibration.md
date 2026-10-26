# SM120 标定：让成本模型使用本机吞吐，而不是规格表常数

## 1. 标定与正式评测的边界

本实现把硬件属性查询和吞吐标定分开。`hardware-sm120.json` 只提供 SM 数、shared/L2/寄存器上限等属性；`calibration-sm120.json` 则由 11 个独立 TileLang 微基准产生。校准 shape 与 GEMM、Conv、FlashAttention 和 GDN 的正式 shape 不同，不从候选实测时间反推 per-config 常数。

每项使用独立进程、独立 cache、10 次 warmup、50 次 CUDA event 采样，并保存 raw samples、median/IQR、正确性、编译目标、环境和 SASS 证据。`--require-complete` 要求所有核心标定项完整，否则 measured-only 模型拒绝加载该文件。

核心入口是：

```bash
python benchmark/z3_cost_model/calibration/calibrate.py \
  --output manifest/calibration-sm120.json \
  --warmup 10 --repeat 50 --timeout 180 \
  --cache-root evidence/G2/calibration-final/cache \
  --instruction-kind sass --require-complete
```

## 2. 实测结果

| 微基准 | median ms (50 samples) | 对应实测速率 |
|---|---:|---:|
| BF16 MMA GEMM | 0.5372 | 116.84 TFLOP/s |
| FP16 MMA GEMM | 0.5363 | 117.04 TFLOP/s |
| TF32 MMA GEMM | 1.0687 | 58.33 TFLOP/s |
| FP32 CUDA-core FMA | 0.1595 | 36.17 TFLOP/s |
| DRAM copy | 0.7083 | 765.93 GB/s |
| L2-resident copy | 0.1454 | 972.14 GB/s |
| shared read | 0.3835 | 14.63 TB/s |
| shared write | 0.4225 | 13.25 TB/s |
| SFU exp2 | 0.0239 | 20.77 G exp2/s |
| barrier | 0.1233 | 7.08 ns/逻辑同步轮 |
| empty launch | 0.0073 | 4.75 μs/launch |
| LSU issue（由独立微基准推导） | — | 108.65 G warp-inst/s |

原始样本和机器可读汇总见 [calibration-summary.json](../../evidence/G2/calibration-summary.json) 与 [calibration-sm120.json](../../manifest/calibration-sm120.json)。模型使用的 `shared_bytes_per_second` 聚合值为 13.91 TB/s，由 read/write 的实测速率按工作量口径聚合；不是未经验证的峰值带宽。

## 3. 指令证据比函数名更重要

`T.gemm` 本身不能证明生成的是目标指令。每个 GEMM/FMA 微基准都要求 `export_sass` 的编译产物审核通过：

- BF16：`HMMA.16816.F32.BF16`，示例见 [BF16 SASS](../../evidence/G2/instruction-artifacts/bf16_mma_gemm.sass)；
- FP16：`HMMA.16816.F32`，不是 TF32 的 `HMMA.1688.F32.TF32`，见 [FP16 SASS](../../evidence/G2/instruction-artifacts/f16_mma_gemm.sass)；
- TF32：`HMMA.1688.F32.TF32`，见 [TF32 SASS](../../evidence/G2/instruction-artifacts/tf32_mma_gemm.sass)；
- FP32 FMA：独立的 `FFMA` 路径，禁止把 Tensor Core 指令计入该吞吐，见 [FP32 FMA SASS](../../evidence/G2/instruction-artifacts/cuda_fp32_fma.sass)。

第一次 FP16 审核失败就是因为只按 `F16` 字符串搜索，而真实 SASS 没有该后缀。审核规则随后改为按实际 `HMMA.16816.F32` 形式匹配，并加入 BF16/TF32 交叉反例。没有删除审核或把 TF32 当成 FP16。

## 4. 三类问题的真实修复

正式标定不是一次成功的，过程暴露了三个实际问题：

1. copy kernel 的 `T.Tensor` 注解不能依赖未捕获的 Python 闭包变量，改用 TileLang 动态维度和显式 FP32 dtype 后才通过；
2. FP16 SASS 形式与审核正则不一致，已按真实指令修正；
3. measured-only 模型需要完整的 FP16、TF32 和 FP32 CUDA-core 速率，因此三项独立微基准被补齐，没有使用 BF16 比例推导。

这些失败均保留在本地证据目录；最终使用的是重新运行并通过的完整版本，而非早先的 partial JSON。

## 5. 成本模型如何消费它

`CudaCostModel.from_files(..., require_measured=True)` 要求完整 schema 和每项 measured 来源。主会话用它成功加载标定文件，并对 GDN forward 的 `block_DV=64` 得到可解释的预测分项；该预测仅用于后续排序，不能在独立实测之前当成真实延迟。

标定模型还显式拒绝 `calibration.status=not_performed` 或任一缺失项。后续搜索会把标定成本与校准/验证集分开报告，正式 held-out 的 oracle 数据不反向拟合这些速率。
