# CUDA task latency/II 模型：给 Z3 的每一种资源标价

> 本节讨论的是 `tilelang/cuda/latency_model.py`，即为调度器提供 task latency 和 II 的模型。它与后文用于 AutoTuner top-k 的整 kernel `CudaCostModel` 不是同一个分数，也不能把二者混用。

## 1. 为什么 latency 和 II 必须分开

一个操作的完成时间包含固定开销、传输时间和描述符成本；它在流水线上多久允许下一次发射，则由另一条资源占用曲线决定。把二者写成同一个数字，会错误地阻塞独立资源，或者过度乐观地重叠同一资源。

对一条 memory path，本实现使用：

$$completion=base+\left\lceil\frac{bytes}{bandwidth}\right\rceil+descriptor$$

$$access\_ii=\max\left(\left\lceil\frac{bytes}{bandwidth}\right\rceil,min\_ii\right)+issue$$

多个 memory access 的 completion 累加；同一路径的 II 先聚合，不同资源最后取上限。GEMM、CUDA-core、SFU 和 scalar 也分别计算 latency 与 II，再把结果交给 Z3。

真实单元测试 `test_memory_completion_and_independent_ii_formula` 使用 130 字节、64 B/cycle 的输入，得到：

```text
transfer = ceil(130 / 64) = 3 cycles
completion = 10 + 3 + 2 = 15 cycles
II = max(3, 3) + 1 = 4 cycles
```

测试同时断言 `cost.latency == 15`、`cost.ii == 4`，防止未来改动把固定 completion 开销错误塞进 II。

## 2. 资源与 CUDA 的对应关系

七个资源 bit 与 Z3 求解器共享一份契约：

| 位 | CUDA 资源抽象 |
|---:|---|
| 1 | global load，包括 cp.async/TMA 的更细分路径 |
| 2 | global store |
| 4 | shared-memory/LSU |
| 8 | Tensor Core/MMA |
| 16 | CUDA core/vector |
| 32 | SFU/special function |
| 64 | scalar/control |

`Resource` 是 `IntFlag`，因此可直接生成 `resource_flags`。TMA 与 cp.async 可以共享 global-load 竞争域，同时在 `MemoryPathParams` 中拥有不同带宽、固定延迟和发射成本。

## 3. SM120 的 GEMM 不是照搬 Ascend Cube 参数

模型先按目标指令原子对齐 M/N/K，再计算操作数：

$$ops=2\cdot align(M,a_M)\cdot align(N,a_N)\cdot align(K,a_K)$$

$$throughput\_cycles=\left\lceil\frac{ops}{throughput_{dtype}}\right\rceil$$

SM120 的 BF16 `mma.sync` 使用 `(16,8,16)` 原子；TF32 的 K 原子则是 4，与 SM80 路径区分开。核心估算代码位于 `latency_model.py:1173-1207`：

```python
rounded_m = _align(m, atom[0], "GEMM M")
rounded_n = _align(n, atom[1], "GEMM N")
rounded_k = _align(k, atom[2], "GEMM K")
ops = 2 * rounded_m * rounded_n * rounded_k
throughput_cycles = _ceil_rate(ops, throughput, "GEMM throughput cycles")
gemm_latency += throughput_cycles
gemm_ii = max(gemm_ii, throughput_cycles)
```

测试 `test_sm120_bf16_mma_sync_atom_rounding_and_gemm_ii` 输入 `M=17,N=9,K=17`，实际 rounding 为 `(32,16,32)`，操作数是 32768。使用合成校准吞吐 1024 ops/cycle 后，throughput cost 为 32 cycles；固定 completion base 为 5，因此最终是 `latency=37, II=32`。这里的数字只验证公式，不是 RTX 5090 的标定吞吐。

## 4. 向量、SFU 和 composed task

向量模型分别统计 add/mul/div/special 等操作和 shared-memory 读取字节。CUDA-core、SFU 与 scalar 的活跃资源取最大 latency，避免把可以并行的资源简单相加。

在组合任务上：

- 串行 $A+B$：`latency=max(A.latency, A.ii+B.latency)`，`II=A.ii+B.ii`；
- 重复 $n$ 次：`latency=body.ii*(n-1)+body.latency`，`II=body.ii*n`；
- 分支：latency 和 II 分别取分支上限；
- per-core phase：同阶段候选取最大，cross-core wait 分隔的阶段累加，并保留协议对 II 的限制。

这些聚合函数位于 `compose_serial`、`repeat`、`branch`、`aggregate_phase` 和 `aggregate_per_core`。对应测试不仅检查数值，还检查 reliability 和 fallback 会沿组合链传播。

## 5. 未知参数不能变成“免费硬件”

模型允许在公式开发阶段使用显式占位率，但绝不把未知带宽解释成 0 cycles。每次使用占位值都会：

1. 写入 `TaskCost.fallback_used`；
2. 将 reliability 降为 estimated、upper-bound 或 unknown；
3. 在 `require_measured=True` 时抛出 `CalibrationError`。

`test_zero_bandwidth_is_explicit_fallback_not_free` 把带宽设为 0 后，模型没有返回 0-cycle 传输，而是使用带标记的 1 B/cycle 占位率，并在 measured-only 模式下拒绝输出。正式实验将启用 measured-only，不允许用这套占位路径生成性能结论。

## 6. 代码、校准边界和测试

公开数据结构包括 `MemoryAccess`、`GemmFeature`、`VectorOps`、`TaskFeatures`、`TaskCost` 和 `CudaLatencyParams`。`Calibration` 是唯一的单位转换边界：JSON 中的 ns、B/s、FLOP/s 被转换成求解器需要的 cycles、bytes/cycle 和 ops/cycle。动态 extent 只使用明确上界，并把不可靠性带到结果中。

主会话独立运行：

```bash
python -m pytest -q testing/python/cuda/test_cuda_latency_model.py
```

结果为 **28 passed**，完整日志见 [latency-model-tests.txt](../../evidence/G2/latency-model-tests.txt)。覆盖范围包括 memory path、strided transaction、SM120 BF16/TF32 rounding、向量/SFU、串行/重复/分支、per-core 协议、动态上界、override、JSON、整数溢出和 measured-only 拒绝路径。

这一层为 Z3 提供整数周期。后文还会给出真实 SM120 微基准标定，以及这些估算如何进入 pipeline 和候选排序；在标定完成前，不把合成参数下的公式测试写成设备性能证据。
