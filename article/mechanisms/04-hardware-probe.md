# SM120 硬件探针：每个参数都要能回答“从哪里查到”

> 本文的成本模型同时使用两类数据：设备/driver 报告的属性，以及微基准测得的吞吐。`tilelang/cuda/hardware.py` 只负责第一类，不能把属性查询写成带宽标定。

## 1. 为什么不能按显卡名称填一张固定表

同为 RTX 5090 Laptop，driver、功耗模式、动态 shared-memory opt-in、当前时钟和软件栈都可能不同。把其他机器的 SM 数、带宽或 shared-memory 上限直接写进约束，轻则得到错误排序，重则错误删除可运行配置。

本实现依次尝试 PyTorch `get_device_properties`、`cuda-bindings` 和 CUDA Runtime ctypes 查询。每个字段同时保存 `source` 与 `reason`：查询成功时 reason 为 null；无法查询时值为 null，并记录缺失原因，不按 GPU 型号猜测。

核心原则位于 `hardware.py` 的属性收集与 `_mark_missing`：缺字段不是异常成功的 0，也不是从其他字段推导出的近似值。`max_registers_per_thread` 没有稳定公开查询路径时，就保留 null，而不用 registers/SM 再除以 threads/SM 冒充同一物理量。

## 2. 本机真实属性

以下数据来自 [hardware-sm120.json](../../manifest/hardware-sm120.json)，采集命令为：

```bash
python scripts/probe_cuda_hardware.py \
  --device 0 --output manifest/hardware-sm120.json --pretty
```

| 属性 | 实测查询值 | 来源 |
|---|---:|---|
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU | PyTorch device properties |
| Compute capability | 12.0 | PyTorch major/minor |
| 显存 | 25,650,855,936 B | PyTorch total_memory |
| SM 数 | 82 | PyTorch multi_processor_count |
| Warp size | 32 | PyTorch warp_size |
| 每 block/SM 最大线程 | 1024 / 1536 | CUDA attributes |
| 每 block/SM registers | 65536 / 65536 | CUDA attributes |
| 普通 shared/block | 49152 B | PyTorch shared_memory_per_block |
| opt-in shared/block | 101376 B | PyTorch shared_memory_per_block_optin |
| shared/SM | 102400 B | PyTorch shared_memory_per_multi_processor |
| reserved shared/block | 1024 B | `cudaDevAttrReservedSharedMemoryPerBlock` |
| L2 | 67,108,864 B | PyTorch L2_cache_size |
| persisting L2 上限 | 41,943,040 B | `cudaDevAttrMaxPersistingL2CacheSize` |
| memory bus | 256 bit | PyTorch memoryBusWidth |
| Driver / NVCC | 610.74 / 13.3.73 | nvidia-smi / nvcc `--version` |
| PyTorch / Z3 | 2.14.1+cu130 / 4.15.4 | 包与运行时查询 |

这些数字用于容量和资源约束。`clock_rate_khz`、memory clock 和 nvidia-smi 的当前时钟分别保留，不能把某一时刻的 P8 时钟当成持续负载下的实际频率。

## 3. 一个实际暴露出的复现问题

第一次采集时，device attributes 已完整，但 `driver_version` 为 null。日志给出的原因不是“驱动未知”，而是精简 PATH 中没有 `nvidia-smi`。在 WSL 中，该工具位于 `/usr/lib/wsl/lib/nvidia-smi`。

将 `/usr/lib/wsl/lib` 加入 `scripts/env.sh` 后重新采集，driver、当前显存、时钟、功耗、温度和 pstate 均获得真实值。两次日志分别保留在 [hardware-probe.txt](../../evidence/G2/hardware-probe.txt) 与 [hardware-probe-complete.txt](../../evidence/G2/hardware-probe-complete.txt)，复现者可以看到问题原因和修正后的完整记录。

## 4. 属性查询不等于标定

输出 JSON 中明确写着：

```json
{
  "calibration": {
    "status": "not_performed",
    "memory_bandwidth_bytes_per_second": null,
    "throughput_by_operation_operations_per_second": null
  },
  "provenance": {
    "microbenchmarks_performed": false
  }
}
```

`validate_hardware_spec` 将核心属性错误与 calibration 错误分开。这样可以先验证容量约束需要的字段，同时拒绝把 `not_performed` 的吞吐项送进 measured-only 成本模型。真正的 DRAM/L2/shared、MMA、SFU、barrier 和 launch 数据由独立微基准生成，不能由这张属性表替代。

## 5. 测试证据

主会话运行 `testing/python/target/test_tilelang_cuda_hardware_spec.py`，结果为 **8 passed**，见 [hardware-spec-tests.txt](../../evidence/G2/hardware-spec-tests.txt)。测试覆盖 JSON 可序列化、缺字段不猜测、来源与验证错误、calibration 独立报告、原子写失败清理、CLI 帮助和非 CUDA 环境错误。

主会话还对真实 JSON 额外断言了 82 SM、101376 B opt-in shared memory 和 driver 610.74，输出为 `HARDWARE_SPEC_VALIDATED`。这些检查证明参数来源与 schema 可信；设备性能是否适合某个候选，仍需后续微基准和 kernel 实测回答。
