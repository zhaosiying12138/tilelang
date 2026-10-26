# CUDA 编译矩阵驱动设计

## 1. 目标与边界

本矩阵把已有 pytest 节点组织成可复现的 host、quick、full 三档集合，用于分别回答：

1. 测试节点是否仍能收集；
2. 默认路径能否完成 lowering、代码生成或设备编译；
3. 有参考断言的节点是否真正通过数值门禁；
4. 显式标注循环能否在新的 Z3 opt-in fixture 中独立编译。

驱动本身不安装依赖、不构建 TileLang、不清缓存，也不把收集、skip、GDN smoke 或 `NOT_RUN` 写成 passed。`--run` 可以执行已有测试体，因此是否启动 GPU kernel 取决于被选择的测试；本文末尾的本次验证只做了 CPU 测试和 `--collect-only`，没有运行 GPU kernel。

实现文件：

- `benchmark/z3_cost_model/compile_matrix.py`：stdlib-only runner；
- `manifest/compile_matrix.json`：冻结节点、分层、arch lane、依赖和门禁；
- `scripts/run_compile_matrix.sh`：仅选择解释器并调用 runner；
- `testing/python/cuda/test_compile_matrix_driver.py`：纯 CPU 驱动测试及独立 Z3 fixture；
- 本文：矩阵和统计合同。

## 2. 三档冻结矩阵

### 2.1 行数口径

一行是一个 manifest 中明确写出的 pytest 节点，不是一个动态 glob，也不是参数化展开后的单个 case。

| tier | operator | host 支撑 | 独立 Z3 fixture | 冻结分母 | 用途 |
|---|---:|---:|---:|---:|---|
| `host` | 0 | 77 | 0 | 77 | Z3 scheduler、latency 公式、software-pipeline 与 loop 结构检查 |
| `quick` | 49 | 77 | 0 | 126 | 代表性算子、动态形状、TMA/cp.async/WS、SM120 与 host 支撑 |
| `full` | 651 = testing 627 + examples 24 | 77 | 1 | 729 | 广泛编译/数值矩阵及独立 opt-in 编译点 |

只读审计中的 L1=126、L2=729 在这里冻结为候选行数；它们不是 passed 数，也不表示本次已经对全部行完成 pytest collection。每次实际收集数量和未运行数量以该次 `summary.json` 为准。

### 2.2 冻结方式

manifest 使用三类 file group：

- `operator_groups`：`path` 加明确的 `nodes`；`quick_nodes` 必须是其中子集；
- `host_groups`：同时进入 host、quick、full；
- `z3_fixture_groups`：只进入 full，并具有显式 `z3_opt_in.pass_configs`。

runner 展开后校验各档行数、节点唯一性、quick 子集关系、路径安全、layer/arch/dependency 名称和数值门禁一致性。命令行只接受精确的行 ID、group ID 或文件 path；不会通过子串、通配符或 `pytest -k` 重新发明一套节点集合。

collect 和 run 都读取同一个 manifest SHA256，并在 `run.json` 记录同一 tier 的 `frozen_node_ids` 以及每个源文件的 SHA256。`--only` 只改变 selected 位，不缩减冻结分母；未选行仍写入 JSONL，状态为 `NOT_RUN`。

### 2.3 覆盖来源

冻结节点覆盖了以下已有源码族，而非仅按文件名推测：

- elementwise/copy：`testing/python/language/test_tilelang_language_copy.py`、`examples/elementwise/`；
- reduce/norm：`test_tilelang_language_reduce.py`、`test_tilelang_language_reducer_v2.py`、`examples/norm/`；
- GEMM 多 dtype/layout：`testing/python/kernel/test_tilelang_kernel_gemm*.py`、WGMMA/TCGEN05、SIMT、SM70/SM75、strided/batched/sliced operand；
- convolution：`examples/convolution/` 与 im2col fallback；
- attention：GQA/MHA fwd/bwd、varlen、block-sparse 和 linear-attention 示例；
- GDN：`examples/gdn/test_example_gdn_compilation.py` 的 8 个节点；
- 量化：FP8/FP4/NVFP4、blockscaled GEMM、scale layout、dequantize/cast；
- 动态 shape：strided/dynamic symbolic profiler、NVRTC dynamic shape、dynamic shared memory；
- TMA、cp.async、WS：TMA load/store/gather/cluster、async-copy、pipeline transitive dependency、no-WS 和 WS producer extent；
- SM120：TMA shared-memory alignment codegen/runtime、NVFP4/SM120 contract 与相应 Blackwell lanes。

数值硬门禁采用保守标记：可从测试及其本地 helper 追踪到 `assert_close`、`assert_allclose`、`allclose` 等等价参考断言时才标 `numeric_gate=true`。示例 wrapper 调用的 `main()` 按实际参考检查归类。只检查生成文本、只编译、只检查输出非空的节点均不标数值门禁。

## 3. H0/H1/G/P 与 arch lane

### 3.1 功能层

| layer | 含义 | 成功状态 |
|---|---|---|
| `H0` | host/frontend/IR 结构检查，不作设备或数值声明 | `PASS_COMPILE_ONLY` |
| `H1` | lowering、代码生成或设备编译，无数值硬门禁 | `PASS_COMPILE_ONLY` |
| `G` | GDN compilation/launch smoke；现有测试没有参考数值断言 | `G_SMOKE`，绝不写成 `PASS_NUMERIC` |
| `P` | 有参考结果的数值 parity；所有收集 case 都必须实际执行并通过 | `PASS_NUMERIC` |

特别是 GDN 的 `test_example_*_compilation` 虽然部分测试体会调用 kernel，但源码没有参考结果断言。因此它们只能是 `G_SMOKE`，不能凭函数名中的 compilation 或 pytest 零退出声称 correctness。

### 3.2 arch lane

`arch_lanes` 包括：

- `host`：不需要当前 GPU；
- `explicit-cuda-target`：测试自己固定目标，例如 `sm_80`，不能替换成当前设备 arch；
- `cuda-any`；
- 精确 lane：`sm70`、`sm75`、`sm80`、`sm89`、`sm90`、`sm100`、`sm120`；
- 范围 lane：`sm80-plus`、`sm89-plus`、`sm90-plus`、`sm100-plus`、`sm120-plus`、`cuda-up-to-sm80`、`cuda-up-to-sm89`。

run 在子进程前检查 lane，不匹配记 `SKIP_ARCH`。可用 `TILELANG_MATRIX_ARCH=sm_120` 显式指定 CI lane；否则 runner 只查询 compute capability，不启动 kernel。collect 不按当前 arch 预先排除节点，因为收集用于检查冻结清单本身。

## 4. 依赖合同

每行分别列出 `dependencies` 和 `optional_dependencies`。probe 只检查 in-tree 文件、module spec 和 distribution metadata，不 import 可选包、不安装包。

FA 与 FLA 是不同依赖：

- FA：`flash-attn` / module `flash_attn`；
- FLA：`flash-linear-attention` / module `fla`。

二者没有替代关系。普通非 varlen FlashAttention 节点不会仅因目录名而被强制声明 FA；实际使用 `flash_attn` 的 varlen 和指定 block-sparse 节点才声明 FA。linear-attention 的 FLA 依赖独立记录。GDN 的 FLA 仅作 optional 参考实现信息，不是 compilation smoke 的强制依赖。

`requirements-test-cuda.txt` 当前同时包含 `flash-attn==2.5.8` 和 `torch<2.14`。在 torch 2.14+ 环境中，runner 记录该冲突；缺包时为 `SKIP_DEP`，不会安装 FA，也不会为了 FA 降级 torch。如果环境中已经存在模块，冲突只作为诊断保留，实际可用性由隔离测试判定。

## 5. 命令与子进程模型

### 5.1 常用命令

```bash
# host 收集；仅这里允许小 worker 并行
scripts/run_compile_matrix.sh \
  --collect --tier host --collect-workers 2 \
  --output /tmp/tl-matrix-host-collect

# quick 收集/运行
scripts/run_compile_matrix.sh --collect --tier quick \
  --output /tmp/tl-matrix-quick-collect
scripts/run_compile_matrix.sh --run --tier quick --timeout 300 \
  --output /tmp/tl-matrix-quick-run

# 精确节点选择，可重复；不接受通配符或模糊 -k
scripts/run_compile_matrix.sh --collect --tier full \
  --only testing/python/cuda/test_compile_matrix_driver.py::test_z3_opt_in_pass_configs_compile_fixture \
  --output /tmp/tl-matrix-one

# 仅专用 fixture 接受 z3-opt-in
scripts/run_compile_matrix.sh --run --tier full --mode z3-opt-in \
  --only testing/python/cuda/test_compile_matrix_driver.py::test_z3_opt_in_pass_configs_compile_fixture \
  --output /tmp/tl-matrix-z3-opt-in
```

`--output` 必填，`--timeout` 是每个节点的秒数。`--collect-workers` 只接受 1–4，且只允许 host collect；quick/full collect 和所有 run 都串行，不会并行使用 GPU。

每个节点启动一个独立 `python -m pytest` 子进程，stdout/stderr 分别写文件。超时终止该节点的进程组并记 `FAIL_INFRA`，不影响后续节点的独立日志和缓存。

### 5.2 collect 与 conftest

仓库 `testing/conftest.py` 的 terminal hook 面向测试执行，会把成功的 `--collect-only` 错判为 “No tests were collected” 并返回 5。为避免修改该公共文件，collect 子进程使用 `--noconftest`；收集不需要 runtime fixture。run 不使用该选项，仍加载正常 conftest。两者使用完全相同的冻结 target，不通过 conftest 或 `-k` 改变选择。

## 6. 缓存与输出

输出目录包含：

```text
run.json          # manifest/source hash、冻结节点、arch probe、运行配置
results.jsonl     # 每个冻结行恰好一条记录，包括 NOT_RUN
summary.json      # 状态聚合和统计口径
logs/<cache identity>.stdout.log
logs/<cache identity>.stderr.log
cache/<cache identity>/
```

每条 JSONL 记录包含：节点、layer、arch lane、依赖、数值门禁、selected/attempted、状态、原因、命令、exit code、duration、pytest case 计数、stdout/stderr 路径、source/manifest hash、`cache_identity` 和独立 cache 路径。

cache identity 由 runner 版本、manifest/source hash、节点、layer、arch lane、数值门禁、mode、显式 pass configs、Python 和 `TILELANG_TARGET` 共同计算。每个实际启动的节点都得到独立目录，并设置：

- `TILELANG_CACHE_DIR`；
- `TORCH_EXTENSIONS_DIR`；
- `TRITON_CACHE_DIR`；
- `CUDA_CACHE_PATH`；
- `XDG_CACHE_HOME` 和节点临时目录。

pytest 共享 cacheprovider 被禁用。runner 不调用 `rmtree`、`rm -rf` 或共享 cache clear，也不允许 cache 路径逃逸到 output 之外或 alias 继承的共享 `TILELANG_CACHE_DIR`。

## 7. 状态和失败分类

| 状态 | 含义 |
|---|---|
| `COLLECTED` | 该节点 collect 成功；不是 compile/pass |
| `PASS_COMPILE_ONLY` | 非数值门禁的测试实际通过 |
| `PASS_NUMERIC` | P 层所有收集 case 实际通过，没有 skip/deselect/xfail |
| `G_SMOKE` | GDN smoke 通过；不是 correctness |
| `SKIP_ARCH` | arch/device 前置检查或 pytest arch skip |
| `SKIP_DEP` | 必需依赖缺失或 pytest dependency skip |
| `FAIL_LOWER` | frontend、TIR、lowering、layout/codegen 阶段失败 |
| `FAIL_DEVICE_COMPILE` | nvcc、ptxas、NVRTC 或设备源码编译失败 |
| `FAIL_LAUNCH` | CUDA launch、illegal access、misaligned address、device assert 等 |
| `FAIL_NUMERIC` | 参考结果断言或数值 mismatch |
| `FAIL_ENV` | 缺模块、共享库、driver/toolchain 环境错误 |
| `FAIL_INFRA` | timeout、spawn、pytest usage/internal/no-tests 等基础设施失败 |
| `NOT_RUN` | 未被 `--only` 选择、被 mode 排除，或无法归入 arch/dep 的 skip/xfail |

分类使用捕获日志的稳定优先级：环境、launch、设备编译、强数值断言、lowering，最后才按该行是否数值门禁处理普通 AssertionError。不能确定时不伪造更具体的成功状态。

一个参数化节点若部分 case passed、部分 skipped/deselected/xfailed，整行不能提升为 `PASS_NUMERIC` 或 `PASS_COMPILE_ONLY`；根据 skip 原因记 `SKIP_ARCH`、`SKIP_DEP` 或 `NOT_RUN`。pytest 零退出但没有实际执行测试也属于 `FAIL_INFRA`，不是 passed。

## 8. 统计口径

`summary.json` 同时保留矩阵行和 pytest case 两套计数：

- `denominator`：该 tier 的冻结 manifest 行数，quick=126、full=729、host=77；
- `selected`：经过 `--only` 和 mode 后实际选择行数；
- `attempted`：真正启动 pytest 子进程的行数；
- `status_counts`：所有状态之和严格等于 denominator；
- `passed`：仅 `PASS_COMPILE_ONLY + PASS_NUMERIC`；
- `g_smoke_not_numeric`：独立列出，不并入数值通过；
- `collected_not_passed`、`skipped_not_passed`、`not_run_not_passed`：防止分母偷换；
- `numeric_gate.selected_eligible/passed/not_satisfied`：所选 P 行的门禁完成情况；collect 或 NOT_RUN 自然不会满足数值门禁，但不等同于 `FAIL_NUMERIC`；
- `pytest_case_totals`：参数化展开后的 collected/passed/skipped 等 case 数，不能替代矩阵行分母。

进程退出码：配置错误为 2，存在 `FAIL_*` 为 1，否则为 0。退出码 0 只表示没有失败状态，并不表示所有数值门禁都已满足；CI 必须同时检查 `numeric_gate` 和 `status_counts`，尤其不能把全 skip 当作全通过。

## 9. default-off 与后续 Z3 fixture

### 9.1 当前结构合同

`tilelang/cuda/pipeline.py` 仅在 `tl.enable_z3_pipeline_planning=True` 时调用 `Z3PipelinePlanning`；native pass 还用 `Bool(false)` 作默认值，并要求循环存在 `tl_z3_pipeline` 标注。不存在可供普通 pytest 假装使用的全局 Z3 环境开关。

`test_z3_default_off_is_a_source_structural_contract` 是纯 CPU 源码结构检查，与专用 opt-in fixture 分离。default run 会排除专用 fixture；显式只选择普通节点并请求 `z3-opt-in` 会直接报配置错误。

当前 fixture 为：

```text
testing/python/cuda/test_compile_matrix_driver.py::test_z3_opt_in_pass_configs_compile_fixture
```

它只有在 `TILELANG_COMPILE_MATRIX_MODE=z3-opt-in` 且收到 manifest 的 JSON `TILELANG_COMPILE_MATRIX_PASS_CONFIGS` 时才导入 TileLang。fixture 显式构造标注循环，把 `pass_configs` 传给 `tilelang.compile`，固定 `sm_80` 目标，只检查编译产物非空，不启动 kernel，也不标数值门禁。普通 pytest 直接执行该文件时，它在导入 TileLang 前 skip。

### 9.2 增加后续 fixture 的步骤

1. **新建专用测试节点**：默认模式下必须在任何 GPU/TileLang 工作前 skip；只接受 runner 的 mode 和 JSON pass configs，不读取一个伪造的全局 Z3 开关。
2. **显式标注与传参**：用 `z3_pipeline_annotations(...)` 标注目标 `T.Pipelined` 循环，并将收到的 dict 传给本次 `tilelang.compile(..., pass_configs=...)`。task 数、dependency index、buffer version 和 `num_stages` 必须按 adapter 合同构造。
3. **独立 default 基线**：若需要 default/opt-in A/B，另写普通 default 节点或结构检查；不要让 opt-in fixture 在 default 下偷偷运行。数值 A/B 必须有真正的参考断言，才可标 P。
4. **加入 `z3_fixture_groups`**：填写唯一节点、layer、arch lane、依赖、`numeric_gate`，以及：
   ```json
   "z3_opt_in": {
     "pass_configs": {"tl.enable_z3_pipeline_planning": true},
     "environment": "TILELANG_COMPILE_MATRIX_PASS_CONFIGS"
   }
   ```
5. **更新冻结计数**：同步修改 `expected_rows.full.z3_fixture/total` 和审计说明；quick 不包含 opt-in fixture。运行 CPU manifest 测试核销节点存在和计数。
6. **保持调度互斥**：同一标注模块不得同时启用 automatic warp specialization；不得把普通已有测试批量设置为 opt-in。
7. **先 collect 再单节点 run**：使用不同 output 目录保存 default 与 opt-in 证据，检查 source hash、pass configs、cache identity、日志和数值门禁，而非只看 pytest 零退出。

## 10. 本次实际验证（2026-10-26）

本次没有 build，没有执行 `--run`，没有启动 GPU kernel。

- 纯 CPU 驱动测试：`31 passed, 1 skipped`；唯一 skip 是未由 runner 唤醒的专用 Z3 fixture。
- host 小样本 collect：4 行 `COLLECTED`，其余 73 行 `NOT_RUN`，`passed=0`；输出 `/tmp/tilelang-compile-matrix-collect-smoke/`。
- quick 小样本 collect：4 行 `COLLECTED`，其余 122 行 `NOT_RUN`，`passed=0`；输出 `/tmp/tilelang-compile-matrix-quick-collect-smoke/`。
- full 小样本 collect：专用 fixture 1 行 `COLLECTED`，其余 728 行 `NOT_RUN`，`passed=0`；输出 `/tmp/tilelang-compile-matrix-full-collect-smoke/`。

因此这些结果只证明 9 个所选节点的真实收集和三档冻结分母/日志合同，不证明任何 operator 的编译或数值通过。quick/full/host 的完整运行以及 Z3 opt-in fixture 编译均仍为未执行项。
