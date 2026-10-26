# SM120 整 kernel 候选成本模型

实现位置：`tilelang/cuda/cost_model.py`。本文描述当前实现的合同，不把解析模型包装成实测结果。

## 1. 目的与非目的

`CudaCostModel` 面向 `tilelang.autotuner.guided.CandidateCostModel`，对**同一 operator、同一 shape 下的完整 kernel 候选配置**排序：

```python
score, terms = model.predict(config)
```

- `score == terms["predicted_ms"]`，单位为 ms，越小排名越靠前。
- `terms` 是扁平 `dict[str, float]`；所有值都有限、非负、可用 `json.dumps(..., allow_nan=False)` 序列化。
- 模型、operator 插件、非法 config、shared/register 容量和非有限结果的异常均向上传递；不把失败候选静默删除，也不把失败预测改成 0。
- `predicted_ms` 是解析排序量。没有经过独立、同条件实测验证时，**不得据此声称绝对延迟、加速比或真机最优配置**。
- 实现不导入 TileLang/TVM/torch/CUDA，不编译、不探测 GPU，也不从候选实测时间在线拟合系数。

模型成本由 shape、tile、源码阶段和全局标定率决定。正式测试 shape 不参与任何 per-config 常数拟合；源码中的角色数、chunk 大小、GEMM 数量和逐元素操作计数是结构输入，不是拟合参数。

## 2. API

### 2.1 从 JSON 或 mapping 构造

```python
from tilelang.cuda.cost_model import CudaCostModel

model = CudaCostModel.from_files(
    "manifest/hardware-sm120.json",
    "evidence/calibration/sm120-kernel-rates.json",
    "gemm",
    {"M": 4096, "N": 4096, "K": 4096, "dtype": "bf16"},
    require_measured=True,
)
score, terms = model.predict({
    "block_M": 128,
    "block_N": 128,
    "block_K": 32,
    "num_stages": 3,
    "num_warps": 4,
    "registers_per_thread": 96,  # 来自编译指标时才有意义
    "copy_kind": "cp_async",
})
card = model.model_card()
```

`from_files` 的 `hardware`、`calibration`、`shape` 均接受 JSON 路径或 mapping。`shape` 文件可用顶层 `shape` 再包一层；`operator` 通常为字符串，也可为含 `{"operator": "gemm"}` 的 JSON 文件。直接 `CudaCostModel(...)` 使用 mapping，并可选传入 `operator_plugins`。

**当前仓库的 `manifest/hardware-sm120.json` 只有设备属性，且明确写有 `calibration.status=not_performed`。它是 hardware 输入，不是完成的 calibration 输入；将其作为 measured-only calibration 会直接报错。**

### 2.2 `model_card()`

返回每次独立构造的纯 JSON dict，包含：

- operator、shape、硬件与 calibration 来源；
- score 单位与排序方向；
- calibration 每项数值、单位、来源和 `fallback_used`；
- 条件 fallback，包括未知寄存器、causal 上界、GDN 固定角色/alias、fragment 计数及 reverse-cumsum 初始化边界；
- occupancy、waves、overlap 公式及局限。

`model.fallback_used` 是 calibration 级 fallback 的排序元组。预测相关的结构性 fallback 集中列在 `model_card()["conditional_fallbacks"]`，其数量计入本次 `terms["fallback_count"]`。

### 2.3 operator 插件

内置：

- `gemm`
- `conv2d_nhwc`
- `flash_attention_fwd`
- `gdn_fwd`
- `gdn_bwd`

自定义插件签名：

```python
def plugin(shape, config) -> KernelWork | Sequence[KernelWork]:
    ...

model = CudaCostModel(..., operator="mine", operator_plugins={"mine": plugin})
```

一个 `KernelWork` 表示一次 launch（或相同 launch 的重复次数）及其串行 `PhaseWork`。所有工作量是**每 CTA**值；`shared_bytes_per_cta` 与 `fragment_bytes_per_cta` 分别是 shared/register-fragment 流量，`shared_memory_bytes_per_cta` 与 `fragment_capacity_bytes_per_cta` 分别是两类容量，不能混写。插件返回空 launch、空 phase、NaN/Inf、负值或未知 mma dtype 都会报错。

各内置 operator 使用独立 config 白名单；不支持的旋钮不会被静默忽略。例如 FA 没有 `block_K`/`use_mma`，GDN 也不能传入通用 GEMM tile 旋钮。

## 3. Calibration schema 与单位

schema 为 `tilelang.cuda.kernel-cost-calibration`，当前版本 1。推荐格式如下（数值仅展示类型，不是 SM120 标定值）：

```json
{
  "schema": "tilelang.cuda.kernel-cost-calibration",
  "version": 1,
  "status": "measured",
  "rates": {
    "clock_hz": 1000000000.0,
    "mma_f16_flops_per_second": 1000000000.0,
    "mma_bf16_flops_per_second": 1000000000.0,
    "mma_tf32_flops_per_second": 1000000000.0,
    "cuda_fp32_flops_per_second": 1000000000.0,
    "dram_bytes_per_second": 1000000000.0,
    "l2_bytes_per_second": 1000000000.0,
    "shared_bytes_per_second": 1000000000.0,
    "lsu_warp_instructions_per_second": 1000000000.0,
    "sfu_exp2_operations_per_second": 1000000000.0,
    "barrier_ns": 10.0,
    "launch_ns": 1000.0
  }
}
```

也可将单一项写成显式单位/来源对象：

```json
{"value": 10.0, "unit": "ns", "source": "measured"}
```

单位不符、非有限、bool、0 或负数均报错。`field_sources` 可按 `rates.<name>` 或 `<name>` 标记来源；在 `require_measured=True` 时，全局 `status` 必须为 `measured`，且每项来源也必须为 `measured`。

| 字段 | 单位 | 含义 |
|---|---|---|
| `clock_hz` | Hz | 标定窗口的 SM clock 记录；是审计所需项，不再对已经按秒给出的率二次换算 |
| `mma_f16_flops_per_second` | FLOP/s | 全设备 FP16 `mma.sync` 有效吞吐，FMA 计 2 FLOP |
| `mma_bf16_flops_per_second` | FLOP/s | 全设备 BF16 `mma.sync` 有效吞吐 |
| `mma_tf32_flops_per_second` | FLOP/s | 全设备 TF32 `mma.sync` 有效吞吐 |
| `cuda_fp32_flops_per_second` | FLOP/s | 全设备 FP32 CUDA-core 有效吞吐 |
| `dram_bytes_per_second` | B/s | 全设备 DRAM 有效带宽 |
| `l2_bytes_per_second` | B/s | 全设备 L2 有效带宽 |
| `shared_bytes_per_second` | B/s | 全设备 shared-memory 有效带宽总和 |
| `lsu_warp_instructions_per_second` | warp-instruction/s | 全设备 LSU warp-issue 有效吞吐 |
| `sfu_exp2_operations_per_second` | exp2/s | 全设备 SFU `exp2` 有效吞吐 |
| `barrier_ns` | ns | 一个逻辑 barrier/mbarrier 同步轮的标定成本 |
| `launch_ns` | ns | 一次 kernel launch 的标定成本 |

所有 per-second 率都是**全设备聚合值**，不是 per-SM 值。单 CTA 公式显式除以 SM 数，避免把全设备率直接用于一个 CTA。当前 measured-only 合同要求完整 schema 的全部标定项存在且为 measured，宁可拒绝不完整输入，也不根据 operator 悄悄补值。

### 3.1 标定隔离

- 标定 kernel 与候选/正式测试 shape 必须分离；不能用候选实测时间反推 per-config 常数。
- 每项率需有独立 microbenchmark、测量窗口和来源；硬件 API 报告的 clock、带宽参数或 SM 数不等于 measured throughput。
- FP16/BF16/TF32 不互相按固定比例推导，CUDA core 与 Tensor Core 也不互相推导。
- `clock_hz` 只记录标定条件；若未来改用 cycles 标定，应另设显式单位并完成一次转换，不能重复乘 clock。

### 3.2 非 measured 模式的 fallback

`require_measured=False` 只允许显式、可审计的退化：

- 缺失吞吐/带宽使用 `1 unit/s` 占位；缺失 `barrier_ns`/`launch_ns` 使用 `1e9 ns` 占位；绝不把缺失资源当免费。
- schema 缺失/不支持、非 measured 来源及占位都进入 `fallback_used` 和 model card。
- 这些占位不是硬件估计，也不应用于性能结论。

## 4. 统一整 kernel 公式

### 4.1 Occupancy 与 waves

对每个 `KernelWork` 独立计算：

```text
occ_threads = floor(max_threads_per_SM / threads_per_CTA)
occ_warps   = floor((max_threads_per_SM / warp_size) / ceil(threads_per_CTA / warp_size))
occ_shared  = floor(shared_per_SM / (shared_capacity_per_CTA + reserved_shared_per_CTA))
occ_regs    = floor(registers_per_SM / (threads_per_CTA * registers_per_thread))
occ         = min(occ_threads, occ_warps, occ_shared, occ_regs, max_ctas_per_SM)
waves       = ceil(cta_count / (SM_count * occ))
```

- 当前结构 CTA-slot 上限默认 16，可由 hardware 的 `max_ctas_per_multiprocessor` 覆盖；它不是实测标定率。
- shared 或 register 单 CTA 需求超过物理容量、threads 超过 block 上限时直接报错，不通过 `max(1, ...)` 把非法配置伪装成合法。
- manifest 的 `max_registers_per_thread` 可以为 null。未提供 config 的 `registers_per_thread` 时，不假装知道精确寄存器数，而令 `occ_regs=1`，并设置：
  - `registers_per_thread=0`；
  - `register_occupancy_conservative=1`；
  - 对应 conditional fallback。
- 即使提供寄存器数，当前公式也未建模硬件 allocation granularity、不同 warp 角色的寄存器重分配和 spill；它不是 ptxas 精确 occupancy 报告。

多 launch 时，`waves` 是各 launch waves 之和，`occupancy_ctas_per_sm` 是各 launch 的最小值，`shared_memory_bytes_per_cta` 是最大容量。GDN 另有每 kernel 分项，避免这些聚合值遮蔽局部变化。

### 4.2 单 CTA 资源时间

对一个 phase `p`：

```text
t_mma    = mma_flops_p    / (mma_rate(dtype) / SM_count) * 1000
t_cuda   = cuda_flops_p   / (cuda_rate / SM_count) * 1000
t_dram   = dram_bytes_p   / (dram_bw / SM_count) * 1000
t_l2     = l2_bytes_p     / (l2_bw / SM_count) * 1000
t_shared = shared_bytes_p / (shared_bw / SM_count) * 1000
t_lsu    = lsu_warp_ops_p / (lsu_rate / SM_count) * 1000
t_sfu    = exp2_ops_p     / (sfu_rate / SM_count) * 1000
t_bar    = barriers_p * barrier_ns * 1e-6                 [ms]
t_global = max(t_dram, t_l2)
t_compute = t_mma + t_cuda + t_sfu
```

`max(t_dram,t_l2)` 表示层级全局路径的瓶颈近似，不把同一字节的 DRAM/L2 时间重复相加。`t_lsu` 与 barrier 保守地单独相加；MMA、CUDA core 和 SFU 在 phase 内组成 compute 项。

```text
stages = 1: overlap = t_compute + t_global + t_shared
stages >= 2: overlap = max(t_compute, t_global, t_shared)
phase_core = overlap + t_lsu + t_bar
```

两级以上表示理想流水重叠上限，不证明真实依赖一定允许完全重叠；一级表示无 overlap。不同阶段数同时改变 shared 容量，因此可能改变 occupancy 和 waves。

### 4.3 Launch 聚合

```text
T_kernel = launches * launch_ns * 1e-6
           + waves * launches * sum(phase_core for serial phases)
T_model  = sum(T_kernel for program-order kernels)
```

launch 间不重叠。返回的资源 `*_ms` 项已乘对应 `waves * launches`，因此 `stage_overlap_ms` 通常不等于各资源项的简单和；work counters 则乘 `cta_count * launches`。

主要 terms：

- 时间：`launch_ms`、`stage_overlap_ms`、`mma_ms`、`cuda_ms`、`dram_ms`、`l2_ms`、`shared_ms`、`lsu_ms`、`sfu_exp2_ms`、`barrier_ms`。
- 工作：`flops`、`useful_flops`、`mma_flops`、`cuda_flops`、`dram_bytes`、`l2_bytes`、`shared_bytes`、`fragment_bytes`、`lsu_warp_instructions`、`tma_descriptors`、`sfu_exp2_operations`、`barrier_count`。
- 结构：`launch_count`、`cta_count`、`waves`、`max_waves_per_kernel`、`occupancy_ctas_per_sm`、`threads_per_cta`、`registers_per_thread`、`shared_memory_bytes_per_cta`、`fragment_capacity_bytes_per_cta`、`pipeline_stages` 和 fallback/寄存器标志。

`flops` 是 padded tile 工作，`useful_flops` 是不含边界 padding 的逻辑工作；两者可能相同。

## 5. 流量、stage 与 copy issue

- `l2_bytes`：按 tile 计算的逻辑全局流量，包含 tile 重读。
- `dram_bytes`：默认 `l2_resident=False`，保守等于 L2 流量；`l2_resident=True` 时为 `min(l2_bytes, compulsory_unique_bytes)`。唯一 footprint 由 shape 推导，不是拟合命中率。
- `shared_bytes`：global→shared 写、各 GEMM shared operand 读和显式源码 staging 的流量；不同 GEMM 分别计读，避免把跨 GEMM reuse 当成免费。
- `shared_memory_bytes_per_cta`：多版本 buffer 与持久 workspace 的 shared 容量，不能与流量混淆。
- `fragment_bytes`：register fragment 的逻辑读写字节；`fragment_capacity_bytes_per_cta` 是独立 register 容量下界。两者均不写入 shared 字段；没有 measured register-file bandwidth 时只计数，不虚构时间项。

cp.async 与 TMA 的区别在 issue 模型中显式保留：

- cp.async/vector：一个满 warp 指令按 `32*16=512 B` 估算，load/store/shared issue 分别向上取整。
- TMA：load 使用源码/tile 推导的 descriptor 数；直接标量 load、非 TMA store 和 shared issue 另计。`tma_store=True` 时 store 使用 descriptor 数。
- TMA 不会自动增加 DRAM/L2/shared 带宽，也不改变逻辑字节；cp.async 的 `cp.async.wait_group` 与 TMA mbarrier 也不是同一实现。`barrier_ns` 是当前统一逻辑同步轮标定，具体 PTX 指令数仍需编译证据细化。

## 6. 通用 operator 公式

### 6.1 GEMM

令 `BM/BN/BK` 为 config tile，`tm=ceil(M/BM)`、`tn=ceil(N/BN)`、`tk=ceil(K/BK)`，输入/输出元素大小为 `e/e_o`：

```text
CTAs                 = tm * tn
mma_flops_per_CTA    = 2 * BM * BN * BK * tk
useful_flops         = 2 * M * N * K
load_per_CTA         = (BM*BK*tk + BK*tk*BN) * e
store_per_CTA        = BM * BN * e_o
unique_bytes         = (M*K + K*N)*e + M*N*e_o
shared_capacity      = stages * (BM*BK + BK*BN)*e
                       [+ BM*BN*e_o if output_in_shared]
barriers_per_CTA     = 2*tk [+1 if output_in_shared]
TMA descriptors      = 2*tk [+1 if TMA store]
```

默认 shared 流量为 `2*load`；output staging 再加 `2*store`。`use_mma=False` 使用相同 padded FLOP 工作量，但改走 `cuda_fp32_flops_per_second`，`mma_flops=0`。

### 6.2 `conv2d_nhwc`

shape 使用 `N,H,W,C,K,R,S`（K 为输出通道），支持 stride/dilation/padding：

```text
Ho = floor((H + 2*pad_h - dilation_h*(R-1) - 1) / stride_h) + 1
Wo = floor((W + 2*pad_w - dilation_w*(S-1) - 1) / stride_w) + 1
M_gemm = N * Ho * Wo
N_gemm = K
K_gemm = C * R * S
useful_flops = 2 * M_gemm * N_gemm * K_gemm
```

之后复用 GEMM 的 padded tile 公式。`block_M` 是输出像素 tile，`block_N` 是输出通道 tile，`block_K` 是 reduction tile。L2 流量对应 im2col/implicit-GEMM 的逻辑 tile 模型；`l2_resident=True` 的唯一 footprint 则直接使用原始 NHWC 输入、权重和输出，不把 im2col 物化字节误当 compulsory DRAM 流量。它不是某种 direct-convolution SASS 的精确指令模型。

### 6.3 `flash_attention_fwd`

shape 为 `B,H,S,D`，可用 `S_KV` 表示 cross-attention KV 长度；config 为 `block_M=BM`、`block_N=BN`。令 `q_tiles=ceil(S/BM)`、`kv_tiles=ceil(S_KV/BN)`、`P=kv_tiles*BN`：

```text
CTAs              = B * H * q_tiles
mma_per_CTA       = 4 * BM * P * D          # QK^T 与 PV
cuda_per_CTA      = kv_tiles*(4*BM*BN + BM*D) + BM*D
exp2_per_CTA      = BM * P
load_per_CTA      = (BM*D + 2*P*D)*e
store_per_CTA     = BM*D*e_o [+4*BM if store_lse]
```

CUDA-core 项表示 scale/max/subtract/sum、online rescale 和最终 normalize 的结构计数；不含任意拟合 epilogue 系数。Q persistent、K/V 多版本和可选 output staging 决定 shared 容量。

`causal=True` 目前要求 `S==S_KV`，工作量按每个 CTA 的全 KV 上界估计并明确标 fallback；`useful_flops` 仍为精确三角形逻辑工作 `2*B*H*D*S*(S+1)`。这不是精确的逐 Q-tile causal 调度模型。

## 7. GDN：chunk32、K=DV=128 的源码阶段模型

输入 shape 为 `B,T,H,Hg,K,DV`，固定 `S=32`、`K=DV=128`，`C=ceil(T/S)`，元素大小为 `e`。只支持 FP16/BF16 qkva storage。H 必须能被 Hg 整除。

### 7.1 Public pipeline / PORTING 边界

`gdn_fwd` 覆盖 `kkt_solve` + `fused_fwd` 两个 launch。public forward 产生的 A、g、h 及其他 forward artifacts 是 backward 的输入；`gdn_bwd` **不重跑、也不重复收取 KKT**。

`gdn_bwd` 按 public backward 的程序顺序覆盖：

1. `prepare_h`：`output_final_state=False`、`output_h=True`；
2. `fused_bwd`：`use_dht` 默认 `False`，可由 shape 显式覆盖；
3. 仅当 `Hg < H` 时，对 `dq` 和 `dk` 各一次 `group_reduce_vector`，共两个 launch；`Hg==H` 时完全没有该 kernel；
4. 一次 `reverse=True, zero_last_chunk=True` 的 chunk-local `dg` cumsum。

因此 GQA backward 有 5 个 launch，`Hg==H` 时有 3 个。KKT 只存在于 forward scope，不能在 backward 因 A 是输入而重复计费。方向 config 也保持分离：forward 只有 `block_DV`；backward 的 `block_DV=128` 固定，搜索空间是 `prepare_h_num_stages={1,2,3} × group_reduce_block_size={8,16,32}`。

反向 `dg` 的 TileLang cumsum kernel 已建模；`torch.zeros_like` 的 allocator/fill、varlen padding 清零、CP preprocess 和通信不在当前 dense kernel 范围，分别列入 conditional fallback 或需额外 `KernelWork`/插件显式加入。`model_card()["gdn_scope"]` 给出同一方向边界与真实 config 空间。

GDN 阶段时间项为：

```text
gdn_kkt_ms, gdn_solve_ms, gdn_wy_ms, gdn_recurrence_ms,
gdn_output_ms, gdn_dhu_ms, gdn_dqkwg_ms,
gdn_group_reduce_ms, gdn_reverse_cumsum_ms
```

每项是对应 phase 的 `waves*(overlap+LSU+barrier)`，不含 launch，因此同一预测中：

```text
forward predicted_ms = launch_ms + KKT + solve + WY + recurrence + output
backward predicted_ms = launch_ms + WY + recurrence + dhu + dqkwg
                        + group_reduce + reverse_cumsum
```

WY 在 backward 中聚合 `prepare_h` 与 fused recompute 的同名阶段。`gdn_{kkt,prepare_h,fused,group_reduce,reverse_cumsum}_*` 提供每 kernel 的 waves、occupancy、CTA/launch 数、CUDA FLOP、L2/shared/fragment 流量及 shared/fragment 容量诊断。`gdn_group_reduce_cta_count` 包含 dq/dk 两次 launch。

### 7.2 KKT 与 solve

每个 chunk/head CTA 的 KKT：

```text
MMA = 2*S*S*K                    # K @ K^T
CUDA = 2*S*S                     # beta scaling 与 mask/构造
```

32x32 unit-lower solve 按源码两个 16x16 对角块和两块 coupling 计算：

```text
diag_inverse = 2 * 16 * 2 * sum(1..15) = 7680 FLOP
coupling     = 2 * (2*16^3)            = 16384 FLOP
solve_CUDA   = 24064 FLOP per chunk CTA
```

KKT 与 solve 是同一 launch 的串行 phase，不重复收 launch。K 使用 cp.async；solve 的 shared 流量由标量 MAC 操作数与 A32 staging 推导，而不是给整个 GDN 一个总系数。

### 7.3 Forward fused：WY / recurrence / output

`block_DV=b` 必须为 128 的约数且为 16 的倍数。fused grid 为 `B*H*(DV/b)`，每个 CTA 串行处理 C 个 chunk。下表是**每 CTA、每 chunk** 的 MMA；实际 phase work 再乘 C：

| phase | 源码 GEMM | MMA FLOP | CUDA-core | SFU exp2 |
|---|---|---:|---:|---:|
| WY | `U=K@S`、`Vd=A@W` | `2*S*K*b + 2*S*S*b` | `3*S*b` | `2*S` |
| recurrence | `S += K^T@V'` | `2*S*K*b` | `K*b` | 0 |
| output | `P=Q@K^T`、`O=Q@S`、`O+=Pg@Vd` | `2*S*S*K + 2*S*K*b + 2*S*S*b` | `5*S*S + 2*S*b` | `S*(S+1)/2` |

输入 Q/K/A/V/g/b、O/h/h0/ht 字节按各源码阶段计入；Hg 与 DV split 的重复读取影响 L2 和 resident 模式的 compulsory footprint。shared 流量分别列各 GEMM operand 与 W/Vd/Vn/P/O staging。

`prepare_h_num_stages` 和 `group_reduce_block_size` 不属于 forward；传入任一旋钮都会明确报错。

Forward shared 容量使用源码生命周期估计：`o/p/vn` 可复用已消费的当前 q/a buffer，持久 h 与 vd 等另计：

```text
capacity(b) = 4*S*K*e + 3*S*b*e + 2*S*S*e + K*b*e + 24*S
```

该 alias 是 model-card 明示的近似，不冒充 ptxas 分配结果；编译后的确切 shared bytes 应通过插件/后续编译指标替换。

### 7.4 Backward `prepare_h`

`prepare_h_num_stages=p` 只控制 prepare_h 的 K/V/A/g/b 多版本；真实搜索空间为 `p in {1,2,3}`，其他整数直接报错：

```text
versioned = S*K*e + S*DV*e + S*S*e + 2*S*4 = 18688 B (FP16/BF16)
persistent = K*DV*e + S*K*e + S*DV*e + 2*S*(K/2)*e + S*4 = 57472 B
capacity(p) = p*versioned + persistent
```

所以 p=1/2 分别为 76160/94848 B；在当前 SM120 manifest 容量下 p=3 会超容量并报错，而不是被钳制成可运行配置。

每 CTA、每 chunk：

| phase | MMA FLOP | 说明 |
|---|---:|---|
| WY | `2*S*S*K + 2*S*K*DV` | `X=A^T@K`、`U=K@S` 及 X/Y 逐元素操作 |
| recurrence | `2*S*K*DV` | `S += X^T@Y`、state scaling、h/mt 输出 |
| CP correction（仅 `calc_mt=True`） | 额外 `4*S*K*K` | 两个 Z 半块和两个 M 更新半块 |

p 同时控制该 kernel phase 的 overlap；recurrence 的数学依赖仍在文档局限中，不能把 p>1 当成完全消除 serial state dependency。

### 7.5 Backward fused：WY / dhu / dqkwg

fused backward 没有 `block_DV` 旋钮，只接受 128；传其他值直接报错。以下三元组 `(m,n,r)` 各贡献 `2*m*n*r` MMA FLOP，左右半个 DK 的 GEMM 已聚合成完整工作：

| phase | 每 chunk 的 GEMM `(m,n,r)` 列表 | 对应源码工作 |
|---|---|---|
| WY | `(S,S,K), (S,DV,K), (S,DV,S)` | P/QK、U=K@S0、V'=Ag@W 与 mask/WY recompute |
| dhu | `(K,DV,S)` 两次、`(S,K,DV)` 一次 | dS0 的 K/Q 贡献、dV'=K@dSt、dH/dg_last |
| dqkwg | `(S,DV,S)` 两次；`(S,K,DV)` 三次；`(S,K,S)` 三次；`(S,S,DV)` 两次；`(S,S,S)` 两次；`(S,S,K)` 一次 | dV/dK/dQ、dAg/dPg/dA/At 及 dg/db |

逐元素、scale/mask 和 dot-reduction 另计 CUDA-core：

```text
WY:    6*S*S + 2*S*DV
dhu:   3*K*DV + 2*S*DV
dqkwg: 4*S*DV + 6*S*K + 17*S*S
```

这些是源码操作组计数/上界，不是 per-config 拟合系数，也不宣称等于最终 SASS 指令数。Q/K/V/A/do/h/dht 与 dq/dk/dv/dg/db/dh0 流量分别进入对应 phase。

### 7.6 GQA `group_reduce_vector`：dq 与 dk 各一次

仅 `Hg < H` 时建模，且对应 vendor 对 dq、dk 的两次调用；同一 `KernelWork` 用 `launches=2` 表示，不是只算一次。令 `group=H/Hg`、block=`b`：

- `group_reduce_block_size` 默认 16，真实空间为 `{8,16,32}`；其他值报错。
- 非默认值要求 `T % b == 0`。默认 16 保持 vendor `cdiv` 行为，非整除 tail 按完整 block 上界计工作并记录 `gdn_gqa_reduce.padded_tail` fallback。
- 每个 launch `CTAs=B*Hg*ceil(T/b)`，threads=128；`Hg==H` 时 CTAs、launch、成本均为 0。

每 CTA、每次 launch：

```text
global read       = group * b * K * e
global write      = b * K * out_e
CUDA additions    = group * b * K
fragment traffic  = (4*group + 2) * b * K * 4
shared traffic    = 0
shared capacity   = 0
fragment capacity = 2 * b * K * 4   # buffer_fragment + result_fragment, FP32
```

fragment 流量包含每个 group 的 buffer 写、result 的读/读/写，以及初始 clear 和最终 result 读。每次 launch 的 useful additions 为 `B*T*H*K`；返回的 `gdn_group_reduce_cta_count`、流量和成本再包括 dq/dk 两个 launch。

两个 fragment 是 register 容量，不写入 shared 容量，也不把 fragment 流量伪装成 shared 流量。当前没有 measured register-file bandwidth，因此 `fragment_bytes` 是独立工作计数，不虚构 `fragment_ms`；全局、CUDA-core 和 LSU 成本仍照常计算。fragment capacity 是逻辑下界，不是 ptxas 精确分配。occupancy 继续使用 config 的实际 `registers_per_thread`，缺失时保持保守 1 CTA/SM；若已知寄存器容量低于逻辑 fragment 下界则报错。

### 7.7 Reverse `dg` chunk-local cumsum

public backward 在 GQA reduce 后执行一次 `reverse=True, zero_last_chunk=True` 的 cumsum。当前 dense 模型令 `C=ceil(T/32)`：

```text
CTAs              = B*C                  # H 在 CTA 内，不在 grid 上
threads           = 128
global read       = S*H*4 per CTA        # dg 是 FP32
global write      = S*H*4 per CTA
padded additions  = H*(S-1) per CTA
useful additions  = B*H*(T-C) total
shared traffic    = 4*S*H*4 per CTA
fragment traffic  = 8*S*H*4 per CTA
shared capacity   = S*(H+1)*4 per CTA
fragment capacity = 2*S*H*4 per CTA
logical barriers  = 2 per CTA
```

shared 流量分别对应 fragment→shared、shared→fragment 的转置前后读写；fragment 流量单独覆盖全局/fragment、fragment/shared 和 in-place scan 的逻辑读写。非完整最后 chunk 使用 padded 工作并记录 fallback，useful additions 仍按实际 chunk 长度求和。

`zero_last_chunk=True` 还会在 public wrapper 中执行 `torch.zeros_like`。该 allocator/fill 以及 varlen 的额外 tail zeroing 不冒充上述 TileLang kernel 的工作量，已在 `gdn.reverse_cumsum.zero_initialization_unmodeled` 和 model card conditional fallback 中醒目标出；reverse cumsum 本身并非未建模。

### 7.8 固定 warp 角色与 overlap 限制

- KKT/solve 为 128 threads；fused forward 为 512 threads，包含 producer 与固定 S/V/O 三组 128-thread consumer；fused backward 对应 producer 与 S/K/A 角色。
- 模型按这些源码角色生成阶段工作，不把 GDN 当作可任意重排 warps 的普通 GEMM。
- 当前 named phases 采用保守串行组合；phase 内只做上述资源 overlap。它没有精确求解 S/V/O 或 S/K/A 的并发 critical path、每个 mbarrier arrive/wait、寄存器重分配与所有共享 buffer 生命周期。
- 因此阶段项适合比较 `block_DV`、`prepare_h_num_stages`、`group_reduce_block_size` 和字节/工作量变化，不应被解读为固定角色 kernel 的精确绝对 ms。

## 8. SM120 `mma.sync` 与主要局限

- 本模型针对 compute capability 12.0/12.1 的 `mma.sync` 风格吞吐。不能套用 Hopper WGMMA、Blackwell TCGEN05、Ascend Cube 或 GDN 参考模型中的硬件常数。
- TF32 使用独立 measured rate，不由 BF16 乘一个固定比例得到；K 原子、fragment layout 和实际 mma 形状仍需编译证据验证。
- 未精确建模：tail predication、cache replacement、sector/coalescing、shared bank conflict、寄存器 allocation granularity/spill、指令混合、clock/power 变化、最后一个 partial wave 的不均衡，以及同一 wave 内多个 resident CTA 对吞吐的精确争用。
- `stages>=2` 的 max-overlap、L2 resident 的 compulsory 下界、统一逻辑 barrier 成本和 GDN 串行 phase 都是明确近似。它们可能互相抵消或放大误差，不能据此反推真机排名。
- 正确流程是：独立标定 → 解析排序 → 编译取得寄存器/shared/指令证据 → 对候选做独立实测验证。模型不能替代正确性测试或 AutoTuner 实测。

## 9. Synthetic 测试边界

`testing/python/cuda/test_cuda_cost_model.py` 只使用 synthetic hardware/calibration，覆盖：

- guided `(score,terms)` 与严格 JSON、model card；
- measured 缺项、`not_performed`、单位/provenance 和显式 fallback；
- 五类 operator 的有限分项；
- 独立手算 GEMM、cp.async/TMA、FP16/BF16/TF32、stage 和 occupancy；
- GDN forward/backward 阶段差异、`block_DV`、prepare_h stage bytes 与容量；
- 全部 9 个 `prepare_h_num_stages × group_reduce_block_size` backward 组合、GQA reduce 的 CTAs/流量/成本、`Hg==H` 跳过、默认 16 与非法旋钮；
- reverse `dg` cumsum 的 CTA、FLOP、L2/shared/fragment 流量和独立容量，以及 model card 的 public forward/backward scope；
- 非有限插件结果和模型异常不静默。

测试不预设任何真机最快配置，不运行 GPU、不 build，也不验证绝对 ms。
