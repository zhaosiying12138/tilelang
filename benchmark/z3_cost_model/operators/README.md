# 经典 TileLang 评测算子（Z3 cost model）

本目录只定义算子、配置空间、数学参考和保守约束，不包含历史性能数据。三个模块具有相同入口：

- `config_space()`：返回上游 `tilelang.autotuner.space.ConfigSpace`。
- `raw_configs()`：按 domain 顺序返回完整、未裁剪的笛卡尔积。
- `kernel_factory(config)`：构造并编译 TileLang JIT kernel；`compile=False` 时只返回 PrimFunc。
- `make_inputs()`：用固定 seed 在 CPU 生成 FP32 正态输入，转为 FP16 后再移动到指定 device。
- `reference()`：独立 FP32 数学参考，返回 FP32 tensor。
- `validate()`：逐元素严格检查，返回 `ValidationReport`；失败时抛出带 `.report` 的 `CorrectnessError`。
- `suite_manifest()`：返回 JSON-safe manifest，包含 operator、shape、dtype、raw_count、constraints 和 provenance。

`manifest.json` 是三个默认 suite 的聚合静态快照。配置身份直接使用上游定义：`sha256(ConfigSpace.canonical_json(config))`，与 key 顺序无关。

## 1. GEMM

模块：`gemm.py`。

数学定义为：

```text
C[m,n] = sum_k A[m,k] * B[n,k]
A: [M,K] FP16, B: [N,K] FP16, C: [M,N] FP16, accumulator: FP32
```

支持两个 shape：

- 主 shape：`M=N=K=4096`，即 `PRIMARY_SHAPE`。
- 短 K：`M=N=8192, K=256`，即 `SHORT_K_SHAPE`。

raw space 精确为：

```text
block_M=[64,128,256]
block_N=[64,128,256]
block_K=[32,64]
num_stages=[0,1,2,3]
threads=[128,256]
enable_rasterization=[False,True]
```

总数 `3*3*2*4*2*2 = 288`。`enable_rasterization` 直接传给 `T.use_swizzle`，其余键分别控制 tile、pipeline 和 `T.Kernel` threads。

## 2. NHWC implicit-GEMM convolution

模块：`convolution.py`。

默认 shape 为 `N=4,C=64,H=W=33,F=96,K=3,S=2,D=1,P=1`，因此 `OH=OW=17`，implicit GEMM 的 output-row 维为 `N*OH*OW=1156`，reduction 维为 `K*K*C=576`。布局为：

```text
data:   [N,H,W,C] FP16
weight: [KH,KW,C,F] FP16
out:    [N,OH,OW,F] FP16, accumulator: FP32
```

数学参考是无 bias 的二维 cross-correlation：

```text
ih = oh*S + kh*D - P
iw = ow*S + kw*D - P
out[n,oh,ow,f] = sum_{kh,kw,c, 0<=ih<H, 0<=iw<W}
                 data[n,ih,iw,c] * weight[kh,kw,c,f]
```

raw space 是 GEMM space 去掉 rasterization 后的五项，共 `3*3*2*4*2 = 144`。kernel 中没有 rasterization 占位键；多余键会被拒绝，`block_M/N/K`、`num_stages` 和 `threads` 均被实际 IR 使用。

## 3. Flash attention（BSHD）

模块：`flash_attention.py`。

默认 shape 为 `B=1,H=16,S=2048,D=64,is_causal=True`，Q/K/V/Output 均为 `[B,S,H,D] FP16`，online-softmax 累加为 FP32。raw space 为：

```text
block_M=[64,128,256]
block_N=[64,128,256]
num_stages=[1,2,3]
threads=[128,256]
```

总数 `3*3*3*2 = 54`。

FP32 reference 显式执行：

```text
scores[b,h,i,j] = dot(Q[b,i,h,:], K[b,j,h,:]) / sqrt(D)
causal mask: j <= i（包含对角线；j > i 填 -inf）
P = softmax(scores, dim=j)
Output[b,i,h,:] = sum_j P[b,h,i,j] * V[b,j,h,:]
```

kernel 中第二个 GEMM 按上游方法将 P 转为 FP16，PV 累加仍为 FP32。默认 `pass_configs=None`，没有沿用示例中的 fast-math opt-in。

## 输入与 correctness

默认 seed 为 `20261026`，每个输入 tensor 使用独立、确定性的 CPU generator，避免 CUDA generator 或 tensor 生成顺序影响数据。GEMM/Conv 输入标准差为 `0.125`，FA 输入标准差为 `1.0`。

reference 在 matmul/convolution 前将输入转为 FP32，并在调用期间关闭 TF32，随后恢复原 backend flag。FA reference 显式构造 mask 和 softmax，不调用 TileLang，也不调用融合的 SDPA。

`validate` 要求 actual 为 FP16、expected 为 FP32，并在 FP32 中计算误差：

- `max_mismatched_ratio=0`：任何一个超出 `atol/rtol` 的元素都会失败，不允许按比例放过。
- NaN/Inf 总是 mismatch，包括两侧为相同 Inf 的情况。
- report 记录 `mismatched_elements`、`mismatched_ratio`、`max_abs_error` 和 `max_rel_error`。
- 默认 `atol=rtol=0.02`，调用方可以收紧；这不是 bitwise FP32 比较，因为 actual 的接口 dtype 是 FP16。

## 保守 Z3 约束

每个 `config_space()` 都安装了对应 builder，可直接交给上游 `Z3Pruner`。只有显式 UNSAT 才会删除配置；SAT 和 unknown 均保留。builder 不调用编译器，编译失败也不是 UNSAT 证据。

已编码的条件仅有：

1. tile 为正、`num_stages` 非负、reduction trip 非空，CUDA threads 为正、不超过 1024 且能被 32 整除；可再用显式 device thread limit 收紧。
2. `sm_80`/`sm_90` 静态 stock-pipeline 模型下，根据上游 `src/cuda/op/gemm.cc` 的指令选择和 warp partition 存在性编码局部 GEMM atom 条件。FA 分别考虑 QK 与 PV 两个 GEMM。target 未指定或不受支持时，该项为 unknown，不做猜测。
3. 共享内存只编码必要下界，并且只有调用方提供“已核实的可用 per-block 上限”时才启用：
   - GEMM/Conv：`max(2*block_K*(block_M+block_N)*max(1,num_stages), 2*block_M*block_N)` bytes，覆盖有效 pipeline 版本的 A/B 操作数与输出复用后的必要下界；`num_stages=0` 仍按一个 buffer generation 计算。
   - FA：`max(Q+K,Q+V,O)`，其中 `Q=O=2*block_M*D`、`K=V=2*block_N*D` bytes。

这些下界利用 GEMM 操作数的同时存活关系；不是实际 post-lowering shared-memory 预测。pipeline buffer 版本、reuse、alignment/padding、barrier、寄存器和未提供的 device limit 均在 manifest 中列为 unknown 及来源/原因。

特别注意：

- 不添加 `M % block_M == 0`、`N % block_N == 0` 或全局 reduction 整除条件。Conv 的 1156/96 tail 和自定义 GEMM/FA tail 必须保留。
- 上游 pipeline 生成有边界的 prologue/body/epilogue，未找到通用 `num_stages <= trip_count` 合法性条件，因此不把这一启发式用于 UNSAT。trip 公式和该决定记录在 manifest 中。
- `ConstraintModel(target="sm_80"/"sm_90")` 只表示 stock pipeline 的静态来源模型。改变指令选择的 pass configs 需要重新审查模型；默认 auto 模型会保守保留 target-specific 项。

## 使用入口（未在本交付中执行 GPU 部分）

```python
from benchmark.z3_cost_model.operators import gemm

shape = gemm.PRIMARY_SHAPE  # 或 gemm.SHORT_K_SHAPE
config = gemm.raw_configs()[0]

# 默认 stock pipeline；这会请求编译，但不会在 import 时发生。
kernel = gemm.kernel_factory(config, shape=shape)
# 后续如需 opt-in：gemm.kernel_factory(config, shape=shape, pass_configs={...})

inputs = gemm.make_inputs(shape=shape, device="cuda")
actual = kernel(inputs["A"], inputs["B"])
expected = gemm.reference(inputs, shape=shape)
report = gemm.validate(actual, expected, config=config)
print(report.to_dict())
```

Conv 调用为 `kernel(inputs["data"], inputs["kernel"])`，FA 调用为 `kernel(inputs["Q"], inputs["K"], inputs["V"])`；三者都必须让 factory、make_inputs 和 reference 使用同一 shape。`compile=False` 可用于只构造 PrimFunc；此时传 `pass_configs` 会报错，避免配置被静默忽略。

纯 CPU 测试命令：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q -p no:cacheprovider \
  benchmark/z3_cost_model/operators/tests/test_cpu_suite.py
```

测试覆盖 raw 数量/domain、canonical ID、manifest、配置键消费、stock/opt-in 参数转发、默认与目标感知约束、明显合法 tail 配置保留、严格 shared 下界、solver unknown 保留，以及小 shape 的固定输入、独立 FP32 数学参考和 correctness report。测试不编译、不执行 GPU kernel。

## Provenance

基于当前上游 revision `80364b3d9758854a9692cc0f14089025f54d3f3f`：

- `examples/gemm/example_gemm_autotune.py`
- `examples/convolution/example_convolution_autotune.py`
- `examples/flash_attention/example_mha_fwd_bshd.py`
- `tilelang/autotuner/space.py`

约束来源还对应 `src/cuda/op/gemm.cc`、`src/transform/inject_pipeline.cc` 和 `tilelang/language/loop.py`。本目录不修改上游示例，也不修改其他目录的 `__init__`、autotuner 或 tuner。
