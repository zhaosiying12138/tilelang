# 六、带走什么，怎样复现

## 1. 用五个问题重建整篇文章

读完四任务例子后，可以用五个问题检查是否真正理解机制。

第一，为什么 latency 4 的拷贝可以只有单项 ii 1？因为结果完成时间与发射资源释放时间不同。消费者受前者约束，共享同一资源的下一项任务受后者约束。把二者混用会分别造成过早读取或过度串行。

第二，为什么最小循环 II 是 2，而开始时间却到 8？因为 $II$ 描述周期重复，$start=k\cdot II+r$ 同时保存逻辑轮次坐标和周期内相位。T3 的 start 8 位于 $k=4$，并不表示循环要每 8 个周期才能接下一轮。

第三，为什么 SAT 之后还要看 CUDA 物化？因为约束只覆盖已经编码的语义。owner 双向交接、标量旧迭代快照和异步 commit group 都要求后端同步、依赖和存储版本保持一致。设备数值输出是这条正确性链的最后一环。

第四，为什么资源剪枝与成本模型要分开评价？UNSAT 给出不可行证明，删除它不会改变合法答案集合；预测分数只安排测量顺序，可能把真实最优排在 top-k 之外。前者看证明是否可靠，后者看头部候选覆盖与最终配置质量。

第五，为什么 GDN 的 1.757× 和 1.258× 是搜索加速？因为计时对象包含候选构造、编译、校验和 profiling。guided 与完整穷举选中同一配置时，kernel latency 相同。内部三个正确性案例也没有为这些搜索数字提供性能贡献。

## 2. 机制应该怎样使用

若你正在写一个显式 Z3 流水，先确定任务边界是否适合 flat Task：body 的执行域应清楚，完整读写应能作为同一原子单位分析。需要手工固定逻辑轮次时，再在完整 Task 外使用 Stage。求解后应同时检查 start、II、阶段、顺序、版本和生成代码的数值结果，单独查看最小 II 很容易漏掉数据语义。

若你正在缩小 AutoTuner 空间，先写下能够证明的必要资源条件。无法判断的候选保留为 unknown，让真实编译继续裁决。随后再使用成本模型分配 top-k，并至少保留一个同预算原顺序或随机对照。最终报告应同时包含搜索 wall、真实尝试分母、冻结配置和独立确认 latency。

若模型在一个 workload 上漏掉最优，增大 k 是一种预算选择，本身不构成修复保证。应查看实测最优的预测名次、同分规则和未建模因素，再决定增加预算、修改模型或退回完整穷举。经典 Conv 的 29.58% 回归说明，搜索时间的大幅节省可以伴随明显的配置质量损失。

更换 GPU 时，需要重新查询硬件并标定吞吐。RTX 5090 Laptop / SM120 的速率不能直接套到另一设备；任务周期估算与整 kernel predicted_ms 也要各自检查单位和测量来源。

## 3. 文章自身的最小检查

所有命令从仓库根目录 `/home/zhaosiying/codebase/tilelang-cuda-z3` 执行。文章构建器只做 CPU 文本转换，不运行 TileLang 或 GPU。单元测试命令是：

`PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s article/tests -v`

完整预检命令是：

`python3 article/build_html.py --config article/index.json --check-only`

预检会解析 `article/index.json` 中的六章正文和短附录，检查本地证据、附件、表格和 KaTeX 公式，但不会写出 HTML、PDF 或 ZIP。LaTeX 是公式排版语法，KaTeX 是页面中的公式渲染器。知乎版本中的表格不含 LaTeX；所有公式都由同一 Markdown 源转换。

## 4. 内部调度的设备复现

设备用例需要 compute capability 12.0（SM120）的 GPU、与当前 TileLang 环境兼容的 CUDA Toolkit，以及已配置的 Python 和测试依赖。compute capability 12.0 描述 GPU 架构能力，CUDA Toolkit 版本是另一项软件依赖，不能将两者合并写成“CUDA 12.0”。使用项目环境所选的解释器运行：

`python3 -m pytest -q testing/python/language/test_tilelang_cuda_z3_pipeline.py`

该文件包含公开 A+B Task、公开 Stage、flat 纯 scalar clone 和受限 multi-async 的真实求解与数值用例。重点核对公开/multi-async 四任务的 start `[1,2,6,8]`、最小 II 2，以及 clone 用例的两次求解和两个 source 版本。生成代码中的同步与版本检查应和数值结果一起阅读。

owner-ring 的独立入口是：

`python3 -m pytest -q testing/python/language/test_tilelang_cuda_z3_owner_ring_integration.py`

它只覆盖第四章限定的两个 owner、两个 role 和显式交接结构。因环境而被标记为跳过（skip）与测试通过不同；复现记录应保留实际执行状态，不能把 skip 当成数值验证。

这些设备命令用于正确性复现，不需要运行 GDN 搜索，也不产生第五章的搜索加速结论。

## 5. GDN T8192 搜索与确认

正式协议使用独立进程和运行自有的空缓存。`scripts/run_gdn_cold_fresh.sh` 每次执行一个方向、一个重复轮次编号（repetition）下的五个 arm，并拒绝覆盖已有日志。复现时应选择新的输出目录，例如 `evidence/G6/gdn-repro-t8192-v1`，不要向既有证据目录追加另一轮结果。

forward 的第 0 轮命令为：

`GDN_NUM_TOKENS=8192 scripts/run_gdn_cold_fresh.sh fwd 0 evidence/G6/gdn-repro-t8192-v1`

随后按相同形式运行 repetition 1、2，再把方向换成 `bwd`。脚本固定 warmup 10、repeat 50 和五臂顺序，forward top-2、backward top-3 由对应驱动设置。GPU 应没有其他计算进程，主机内存与缓存路径也要通过脚本预检。

搜索结果的分析入口为：

`python3 benchmark/z3_cost_model/gdn/analyze_results.py --root evidence/G6/gdn-repro-t8192-v1 --output evidence/G6/gdn-repro-t8192-v1/analysis --no-plots`

预测排名可由同目录的 `analyze_predictions.py` 使用相同 root/output 复算，cold 进程与缓存身份由 `audit_cold.py` 审核。分析时保留失败候选和真实分母，不把搜索内 best 当作确认 latency。

冻结 winner 后，forward 的独立确认入口为：

`python3 benchmark/z3_cost_model/gdn/confirm.py --direction fwd --analysis evidence/G6/gdn-repro-t8192-v1/analysis --output-dir evidence/G6/gdn-repro-t8192-confirmation-v1 --rounds 7 --warmup 50 --repeat 200 --selection-rule candidate-median --num-tokens 8192`

backward 使用同一参数并把方向改成 `bwd`。最后用 `analyze_confirmation.py` 的 `--cold-analysis`、`--confirmation-root` 和 `--output` 汇总配对结果。相同配置按身份处理，不同配置查看该配置对自身的比值和逐轮差值。

经典算子的对应入口是 `python3 -m benchmark.z3_cost_model.operators.repeat_search`，参数使用 `--suite all --repetitions 0,1,2 --top-k 10 --warmup 10 --repeat 50` 和一个新的 `--output-dir`。其 `analyze_results.py`、`confirm.py` 与 `analyze_confirmation.py` 按同一职责分离搜索、冻结选择和确认。

新运行的 wall 与 latency 会受时钟、驱动和机器状态影响，复现的目标是核对协议、候选分母、配置身份和统计口径，不要求每个时间数字逐位一致。若硬件或标定文件改变，结果应作为新实验报告，不能继续套用本文 SM120 数值。附录列出本文已接受原始数据、确认样本、失败记录与捕获包，便于逐项交叉检查。
