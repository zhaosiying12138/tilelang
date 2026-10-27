# 三、把同一调度放进 CUDA：周期估算与 flat Task/Stage

## 1. 整数周期从哪里来

第二章把 latency 和单项 ii 当作已知小数。真实编译器需要从任务内容估算它们，否则用户只能为每个任务手工填写所有数字。这里的估算是面向调度约束的整数周期模型，目标是提供一致的相对成本，不构成对设备绝对时钟周期的提前承诺。

本文设备目标是 **SM120**，即 compute capability 12.0 的 CUDA 架构目标；实验设备为 RTX 5090 Laptop GPU。硬件查询提供容量与执行资源上限，独立微基准提供吞吐。**微基准**是专门测量某类操作的小型实验，例如流式拷贝或一组矩阵乘加，它与正式候选的形状分开，避免用待评估候选的答案反推模型参数。

对一条存储路径，完成时间和发射间隔分别估算为：

$$completion=base+\left\lceil\frac{bytes}{bandwidth}\right\rceil+descriptor,$$

$$access\_ii=\max\left(\left\lceil\frac{bytes}{bandwidth}\right\rceil,min\_ii\right)+issue.$$

第一式中，$base$ 是路径固定开销，$bytes/bandwidth$ 是搬运数据所需周期并向上取整，$descriptor$ 是描述符相关开销；三者共同决定结果何时完成。第二式把传输占用、硬件最小发射间隔和发指令开销放进发射侧，决定同一路径多久能接续。

代入一组仅为说明单位的数字：固定开销 2 个周期，128 B 数据，带宽 64 B/周期，描述符开销 1，则完成时间是 $2+\lceil128/64\rceil+1=5$。若最小发射间隔为 1、发指令开销为 1，则访问 ii 是 $\max(2,1)+1=3$。同一份搬运因此有 latency 5、单项 ii 3，等待结果与释放发射资源保持独立。

多个操作的组合也不能把两个量混成一个。若一个调度任务内先后组合 P、Q，模型使用：

$$latency_{P+Q}=\max(P.latency,\;P.ii+Q.latency),$$

$$II_{P+Q}=P.ii+Q.ii.$$

第一式表示组合结果至少要等 P 完成，也要覆盖 P 发射后接上 Q 并完成 Q 的路径；第二式把同一串行组合的发射占用相加。重复和分支有各自组合规则，核心仍是分别传播完成与发射信息。

吞吐参数可以缺失，但缺失不等于免费。模型的 **measured-only** 模式要求所有必需速率都来自已通过的测量，否则拒绝构造；较宽松的模式会显式记录 fallback。这样，一次看似很短的估算不会被误当成经过硬件标定的事实。

## 2. flat Task：把源码边界交给调度器

**IR** 是 intermediate representation，即编译器在中间阶段使用的程序表示。公共标记进入 IR 后，后端需要知道哪些语句共同构成一个任务。当前主链采用 **flat Task/Stage**：flat 指任务位于同一个流水循环的直接子语句层，不再嵌套另一层 Task 或 Stage。

`T.Task(latency=..., ii=...)` 将完整语句体界定为一个原子调度任务。原子表示调度器不会在 body 内另插同步，body 的全部读写共同形成依赖 footprint，并且执行域必须单一。**执行域**是任务的主要执行类别，例如拷贝、向量计算或标量计算；把无法归为一类的操作塞进同一 Task 会被拒绝，需要拆分或补充明确分类。

latency 和 ii 可以分别省略并分别推导。若两者都显式给出，需要满足 latency 非负、ii 为正且 latency 不小于 ii。这个检查拦住明显矛盾的单位，例如“完成比发射间隔还早”。

公共定义位于 [`tilelang/cuda/language/schedule_hint.py`](../../tilelang/cuda/language/schedule_hint.py)，规范化位于 [`src/transform/normalize_z3_schedule_units.cc`](../../src/transform/normalize_z3_schedule_units.cc)。下面保留四任务例子的必要形状，并用测试辅助函数 `z3_pipeline_annotations` 显式展示第二章的成本、依赖和资源输入。`T.Task` body 只界定语句，本身不提供这些 annotation 数值与依赖；缺失信息需要由外层推导流程补全。张量声明与编译参数在这里省略。

```python
annotations = z3_pipeline_annotations(
    latencies=[4, 4, 2, 2], iis=[1, 1, 1, 1],
    resource_flags=[1, 1, 16, 2],
    data_deps=[(0, 2, 0, 4), (1, 2, 0, 4), (2, 3, 0, 2)],
    resource_deps=[(0, 1)],
)
for k in T.Pipelined(iterations, num_stages=2, annotations=annotations):
    with T.Task():
        T.copy(A[k * block:(k + 1) * block], a_shared)
    with T.Task():
        T.copy(B[k * block:(k + 1) * block], b_shared)
    with T.Task():
        for i in T.Parallel(block):
            c_fragment[i] = a_shared[i] + b_shared[i]
    with T.Task():
        T.copy(c_fragment, C[k * block:(k + 1) * block])
# 手工分层时，在完整 Task 外组合 T.Stage(s)。
```

`resource_flags` 依次标记全局读取、全局读取、向量计算和全局写回，`resource_deps=[(0,1)]` 则声明两次读取共享发射资源。

这段代码没有要求求解器改变数学计算。四个 with body 仍然执行两次读取、一次加法和一次写回；标记只把任务边界暴露给显式启用的 Z3 流水。编译器中的一次 IR 变换叫 **pass**。未启用该路径时，stock CUDA pass 保持原行为，公开设备测试也检查了默认路径不会调用求解器。

`T.Stage(stage)` 是手工指定逻辑轮次坐标的入口，必须包住完整 Task。只要同层出现 Stage，未标 Stage 的同级任务默认属于 stage 0；非零 stage 需要允许跨轮次偏移。未给出正的流水深度上界时，上界可以从最大手工 stage 加一推导。显式上界也不会截断用户给出的较大 stage。

这些规则服务于一个简单直觉：Task 决定依赖分析的原子粒度，Stage 决定用户是否已经固定 $k_i$。若 stage 交给求解器，第二章的 $start=k\cdot II+r$ 仍由求解结果决定；若手工给出 stage，求解器只在这个额外约束下检查可行性。

## 3. 从 start/II 到 CUDA 能表达的阶段和顺序

求解器返回整数时间，CUDA 流水后端使用阶段、发射顺序和物理版本。投影发生在 [`src/transform/z3_pipeline_planning.cc`](../../src/transform/z3_pipeline_planning.cc)，第一步仍是对同一个 start 做除法：

$$k_i=\left\lfloor\frac{start_i}{II}\right\rfloor,\qquad stage_i=k_i-\min_j(k_j).$$

第一式取出逻辑轮次坐标，第二式整体减去最小坐标，使最早阶段从 0 开始。整体平移不改变任务间的时间差，只给后端一个规范起点。

代入 `[1,2,6,8]` 和 II=2，得到 `[0,1,3,4]`。T0 位于阶段 0，T1 位于阶段 1，T2 和 T3 分别位于阶段 3、4。两个输入并未被人为压进同一阶段，较长的准备距离保留在阶段坐标中。

相位由另一项给出：

$$residue_i=start_i-k_i\cdot II.$$

它表示任务在周期内的发射位置。本例 residue 为 `[1,0,0,0]`。只有 stage 还不够，因为同一阶段内的多个任务仍需确定谁先发射；只有 residue 也不够，因为 T2、T3 虽同为 0，却处于不同阶段。后端综合相位、阶段和源码位置形成拓扑顺序。**拓扑顺序**是一种让所有依赖都从前指向后的排列。

下游流水需要的数组还有一个方向细节：物理顺序回答“第 $p$ 个发射位置放哪条原语句”，下游 order 回答“原语句被放到哪个位置”。两者互为逆映射：

$$inverse\_order[physical\_order[p]]=p.$$

例如物理位置 2 放原语句 0，则原语句 0 的 order 值就是 2。做逆映射可以避免把“排列里的内容”误当成“内容的位置”，尤其在阶段重排后。

最后，流水深度必须同时覆盖用户上界、派生阶段和物理 buffer 版本：

$$cuda\_depth\geq\max\left(num\_stages,\;\max_i(stage_i)+1,\;\max_b(v_b)\right).$$

它表示实际物化深度至少取三类需求的最大值，输入的 `num_stages` 不会把更大的派生 stage 静默截断。对本例的阶段数组，仅阶段项就要求覆盖 5 层；若某 buffer 需要 3 个版本，版本项也不能低于 3。

这一步把第二章的两个直觉落到了代码结构：$k_i$ 主要变成阶段与跨轮次距离，$r_i$ 参与周期内发射顺序。buffer versions 则按存储身份传到流水注入，避免把“某个数组版本”错配到共享同一名字的另一块存储。

## 4. 为什么物化后仍要做数值验收

投影公式可以检查，生成的 CUDA 也可以编译，但最终验收仍是输入 A、B 后是否精确得到 A+B。公开 Task 用例走过了真实求解、CUDA 编译和设备执行，输出与 PyTorch 参考精确一致；PyTorch 是实验使用的张量计算库。

这些结果覆盖了 flat 公共标记到设备执行的闭环。公开 Stage 用例还检查了手工阶段与旧的手工阶段数组写法在相同意图下的等价数值行为。嵌套任务、混合执行域和无法分类的操作仍需明确拒绝。一次 SAT 只说明整数约束有解，一次仅在 CPU 端完成的结构检查只说明 IR 具备某些形状，设备数值结果才把数据语义纳入验收。

至此，内部调度的分工已经完整：周期模型估算任务，Z3 选择时间，Task/Stage 提供边界，投影生成 CUDA 流水，数值测试检查结果。第五章的成本模型会再次使用硬件吞吐，却服务于另一目标：给多个完整 kernel 配置排序，并决定哪些值得付出真实编译和 profiling 成本。
