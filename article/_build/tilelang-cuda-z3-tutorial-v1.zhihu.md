一、两种搜索：让一次执行正确，让调参少走弯路

1. 从一段已经很熟悉的 CUDA 计算开始

TileLang 是一种用 Python 风格描述分块 GPU 计算的编译框架，CUDA 是 NVIDIA GPU 的编程平台。本文只要求基本 CUDA 经验，不要求接触过约束求解。

考虑对许多个小块重复执行 `C=A+B`。一个 CUDA **kernel** 是交给 GPU 执行的函数；一个 **block** 是一组可以共享片上存储并同步的线程。为了计算一个块，程序通常要把 A、B 从容量较大的全局内存搬到 block 可见的共享内存，再完成向量加法和结果写回。下文把共享内存中的一块临时空间称为 **buffer**。

即使 tile 大小、线程数和流水深度都已固定，仍然有一个调度问题：两次读取能否错开，加法何时获得完整输入，下一轮读取会不会覆盖仍在使用的 buffer。这里的 **tile** 是一次处理的数据块，**流水**则让不同数据块的读取、计算和写回在时间上重叠。

可以把这四步想象成装配线上的四个工位。每个工位都有两件事需要安排：一件产品从进入到离开要多久，以及工位多久能接下一件产品。前一件事关系到半成品何时可用，后一件事关系到产线能否持续接收新任务。编译器还要保证上一工位交付的半成品不会被提前取走或提前覆盖。

在固定配置内部安排工位，是本文的第一条主线。它的输出包括任务开始时间、循环重复节奏和 buffer 版本数，验收标准是生成代码仍然算出 `A+B`。

另一条主线从更外层开始。假设 tile 有四种大小、流水深度有三种选择，应该编译并测量哪一个组合？这种在多个配置中寻找较快组合的过程叫 **自动调优**，英文是 autotuning；执行这项工作的组件叫 **AutoTuner**。每个组合叫一个**候选配置**，测量候选实际运行时间叫 **profiling**。

两个问题会出现在同一份 CUDA 程序里，却有不同的输入、输出和失败方式。

| 对比项 | kernel 内调度 | 外层自动调优 |
| --- | --- | --- |
| 已经给定什么 | 一个配置、任务、依赖和硬件约束 | 一批候选配置及同一数学计算 |
| 需要决定什么 | 何时执行、如何重叠、保留几个版本 | 哪些候选值得编译和实测 |
| 主要输出 | 开始时间、重复间隔、版本和同步 | 排名、实测子集和最终配置 |
| 典型失败 | 数据被提前读取或覆盖，同步错误 | 时间花在无效候选，或漏掉好配置 |
| 最终验收 | 生成 kernel 的数值结果正确 | 搜索成本下降且最终配置质量可接受 |

这张表也划出了后文证据的边界：一个内部调度案例通过数值测试，能够支持该案例的正确性；它不能推出某个外层搜索节省了时间。反过来，一次搜索找到较快配置，也不能证明所有内部调度语义都已覆盖。

2. Z3 在两个位置扮演的角色

**Z3** 是一个约束求解器。使用者把变量和必须满足的条件交给它，由它判断能否同时满足。本文涉及的答案有三种：

- **SAT** 表示存在至少一个解，求解器可以给出变量取值；
- **UNSAT** 表示在当前约束下不存在解；
- **unknown** 表示本次求解没有得出结论，例如资源预算耗尽。

这三种答案必须分开处理。SAT 给出一份候选调度，还要由 CUDA 后端变成真实代码并检查数值。只有 UNSAT 能证明某个候选违反了已编码的必要条件，因此可以安全删除。unknown 没有提供这种证明，对应候选应当保留。

在 kernel 内，约束可能写成“加法必须晚于输入完成”“共享同一发射资源的任务要错开”。求解结果是开始时间和版本数。到了外层，约束可能写成“这个配置需要的共享内存超过 block 上限”。求解结果用于缩小候选集合。两处使用同类工具，解决的问题仍然不同。

外层还有一个 Z3 无法替代的环节：即使两个配置都能运行，约束也不会自动指出哪个更快。为此需要一个**成本模型**，根据工作量、硬件吞吐和重叠假设给候选一个预测分数。分数负责安排实测顺序，真实计时负责裁决结果。

因此，一条完整的外层路径是：保留有合法可能的候选，用预测分数分配有限实测预算，再让原有 AutoTuner 完成编译、数值校验和 profiling。预测毫秒数与调度器使用的整数周期属于不同模型，单位、用途和精度要求都不同。

3. 为什么搜索加速和 kernel 加速要分开说

本文用 **stock** 指仓库中原有、未加入 guided 候选选择的实现路径。假设完整穷举需要编译 9 个候选，guided 搜索只测量其中 3 个。这里的 **guided** 指候选选择受到约束和成本模型引导。若最终两者选中同一个配置，这个配置生成同一个 kernel，那么两次 kernel 执行时间应当相同；节省的是寻找配置的过程。

**搜索墙钟时间**是从候选构造到搜索结束所经历的实际时间，可以包含求解、编译、校验和 profiling。**kernel latency** 是已选 kernel 完成一次计算的时间，本文用毫秒报告。前者缩短 20%，不会自动让后者缩短 20%。

比较搜索质量还需要一个完整参照。本文把对保留空间进行完整实测得到的最优结果称为 **oracle**，可理解为“知道全部实测答案后才能作出的选择”。oracle 用来审计 guided 搜索漏掉了什么，其完整测量成本不能算作免费信息。

等预算对照也很重要。如果 guided 只测 3 个候选，就应同时看原始顺序的前 3 个和固定随机种子的 3 个。它们能帮助区分收益来自候选排序，还是仅仅来自少做了工作。若三种方法最终选择相同，结论应如实写成相同，不能只展示 guided 与完整穷举的比值。

4. 一条连续例子和两类实验

第二、三章始终使用同一个四任务 A+B 流水。数字刻意很小：两次拷贝各需 4 个周期完成，加法和写回各需 2 个周期，每个任务的发射占用都是 1 个周期。由这些数字可以手算验证真实测试采用的一组可行 SAT 开始时间 `[1,2,6,8]`，并确认最小重复间隔为 2。开始时间允许存在其他可行解，这里跟踪的是设备测试实际采用的这一组；随后再看它怎样映射为 CUDA 的任务边界、逻辑阶段和多版本 buffer。

第四章仍围绕“数学上可行，生成代码却可能出错”这一间隙，只讲三个具体难题：两个执行所有者如何交接，标量定义如何保留旧迭代快照，异步拷贝的数据如何等待。每个案例都有明确受限范围和设备数值检查。它们都属于内部调度正确性，不参与外层搜索收益归因。

第五章转到外层，以 GDN 的公开前向和反向管线作为主结果。**GDN** 在本文指 gated delta rule 这一类带门控状态的序列计算；理解实验不需要先掌握它的递推公式，因为调优旋钮是资源与分块配置。经典矩阵乘法、卷积和注意力算子随后用于检验成本模型在别处是否会漏掉最优。

阅读这些实验时，可以一直追问三个问题：候选为什么被删除，最终配置由什么实测选出，独立复测是否仍支持结论。附录只提供原始数据、日志和复现入口；正文所需的数字都会在出现处解释。这样既能看见机制为何存在，也能看见证据恰好证明到哪里。

二、一条 A+B 流水：从四个任务到最小 II

1. 把计算切成四个可讨论的边界

后续例子对长度可变的输入重复下面的循环。每次读取一个 A 块和一个 B 块，在寄存器或局部 fragment 中形成和，再写回 C。**fragment** 在这里指由线程持有的局部数据块；理解调度时只需知道它与前两个共享内存 buffer 是不同的存储位置。

```python
for k in T.Pipelined(iterations):
    T.copy(A[k], a_shared)       # T0：global -> shared
    T.copy(B[k], b_shared)       # T1：global -> shared
    for i in T.Parallel(block):  # T2：vector add
        c_fragment[i] = a_shared[i] + b_shared[i]
    T.copy(c_fragment, C[k])     # T3：写回 global
```

边界决定了求解器能看见什么。T0 和 T1 使用同类全局读取资源，可以互相错开发射；T2 同时依赖两份输入；T3 依赖加法结果。任务内部的线程级并行仍由 CUDA 处理，Z3 安排的是这四个较大粒度任务之间的时间关系。

采用真实设备测试中的整数成本：T0、T1 的 latency 都是 4，T2、T3 的 latency 都是 2；四个任务的每项 II 都是 1。这些数是调度模型的输入，用来解释约束和生成调度，不被当作对所有 A+B kernel 的通用性能预测。

2. latency 与 II：等待时间和接件节奏

任务的 **latency** 是从该任务开始到结果可用所经过的时间。T0 在时刻 1 开始、latency 为 4，则输入在时刻 5 完整可用。依赖它的加法不能仅凭“读取指令已经发出”就提前开始。

任务的 **II** 是 initiation interval 的缩写，本文在单任务成本中表示它占用对应发射资源的间隔。T0 的 II 为 1，表示与它共享资源的下一个任务可以在 1 个周期后接续，而不必等待 4 个周期的完整拷贝尾段。

装配线类比中，latency 是一件产品经过某工位的总时间，II 是工位释放接件口所需的时间。等待内部搬运完成与释放接件口是两件事。把 latency 4 错当成 II 4，会禁止本可发生的重叠；把 II 1 错当成 latency 1，又会让消费者读取尚未完成的数据。

整条循环还有一个最小 II，它是相邻两轮流水之间的重复间隔。为避免混淆，下文用 ii\_i 表示单个任务的发射占用，用大写 II 表示整条循环的候选重复间隔。

3. 数据依赖决定“何时可读”

若任务 u 的结果交给同一轮的任务 v，且结果完成需要 L\_uv，约束写为：

公式：start\_v ≥ start\_u + L\_uv.

它的含义很直接：v 的开始时间至少要等到 u 的开始时间加上结果延迟。等号允许 v 在结果刚可用的时刻开始，大于号则留下额外余量。

真实例子的三条依赖是 T0→T2、T1→T2、T2→T3，延迟分别为 4、4、2。代入后文求得的开始时间：

- T0→T2：6-1=5 ≥ 4，加法开始时 A 已可用；
- T1→T2：6-2=4 ≥ 4，B 恰好完成；
- T2→T3：8-6=2 ≥ 2，写回恰好接上结果。

如果调度只满足前两条而漏掉第三条，加法与写回之间的中间存储就可能被其他轮次改写。依赖分析需要覆盖所有读写 footprint；这里的 **footprint** 是一个任务实际读取和写入的存储集合，包括经别名看到的同一块空间。

4. 资源约束决定“何时能再接一件”

T0 和 T1 共享全局读取资源。直线调度中，可以用布尔变量 O\_01 选择二者顺序，布尔变量就是只取真或假的变量：

公式：O\_01 ⇒ start\_1 ≥ start\_0+ii\_0,

公式：¬ O\_01 ⇒ start\_0 ≥ start\_1+ii\_1.

第一式表示若 T0 排在前面，T1 至少等待 T0 的发射间隔；第二式处理相反顺序。符号 ¬ 表示取反。约束没有要求等待 T0 的全部 latency，因而两个长拷贝仍能紧密发射。

若只执行一轮，可以让 T0 在 0 开始、T1 在 1 开始，T2 在 5 开始、T3 在 7 开始，最后在 9 完成。直线调度的目标常写成：

公式：min makespan, makespan ≥ start\_i+latency\_i.

**makespan** 是所有任务完成时间的上限，最小化它就是缩短这一轮从头到最后结果可用的时间。这里的 i 遍历四个任务。

循环重复执行时，目标发生变化。只看一轮的 9 个周期会漏掉下一轮读取能否提前开始；模调度关注的是稳态吞吐。**模调度**把任务放在一个重复周期里，同时处理周期内的先后和跨过周期边界的先后。

5. start=k·II+r：把“第几层”和“周期内相位”分开

循环中的每个任务开始时间分解为：

公式：start\_i=k\_i · II+r\_i, k\_i ≥ 0, 0 ≤ r\_i&lt;II.

II 是整轮重复间隔，k\_i 是任务所在的逻辑轮次坐标，r\_i 是它在一个周期内的相位。这里的 k\_i 属于调度解，不要与源码遍历输入块的循环变量混为一谈。多个任务可以有相同 r\_i 而处于不同 k\_i，这正体现了跨轮次重叠。

真实求解结果为 start=\[1,2,6,8\]、II=2。直接做整数除法和取余，就得到 k=\[0,1,3,4\]、r=\[1,0,0,0\]。例如 T2 的 6=3 × 2+0，T3 的 8=4 × 2+0。

下表把每个开始时间连同其完成时间一起展开，并在最后两列展示每次再加一个 II 后的时间线。箭头左侧是开始，右侧是结果完成。

| 任务 | latency / 单项 ii | start | k / r | 初始实例 | 向后平移一轮 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T0 读取 A | 4 / 1 | 1 | 0 / 1 | 1 → 5 | 3 → 7 |
| T1 读取 B | 4 / 1 | 2 | 1 / 0 | 2 → 6 | 4 → 8 |
| T2 加法 | 2 / 1 | 6 | 3 / 0 | 6 → 8 | 8 → 10 |
| T3 写回 | 2 / 1 | 8 | 4 / 0 | 8 → 10 | 10 → 12 |

时间线上可以看到两类重叠。时刻 6，T2 的初始实例开始，T1 的另一个实例也可以在 `2+2×2=6` 发射；时刻 8，T3 的初始实例与 T2 的下一轮实例同时落在不同资源上。相位相同没有迫使它们串行，因为资源与数据依赖分别约束。

还要检查 T0、T1 的回绕关系。以相位看，T1 在 0、T0 在 1；从 T0 到下一次 T1 的间隔是 `0+2-1=1`，恰好满足 T0 的单项 ii。只检查 `r` 的一次排序会漏掉这条跨周期边，模调度因此需要同时约束正向和回绕方向。

6. 最小 II 搜索：为什么答案是 2

任何循环 II 都不能小于最大的单项发射占用，所以本例下界是 1。求解器还构造一个有限上界：

公式：II\_hi=max(∑\_i latency\_i, ∑\_i ii\_i)+1.

它把“全部 latency 串行”和“全部发射占用串行”中较大的一个再加 1，作为搜索起点。本例两个和分别为 12 和 4，因此初始搜索区间是 1 到 13，右端不包含在可行候选中。

对固定 II，求解器只需回答当前约束是否 SAT。若可行，就保留解并继续搜索更小值；若 UNSAT，就提高下界。真实查询顺序为：

1. II=7：SAT，暂存一个解，继续向左；
2. II=4：SAT，继续向左；
3. II=2：SAT，再检查是否还能更小；
4. II=1：UNSAT，所以最小可行值是 2。

最后一步可以手算验证。II=1 时所有任务相位都只能是 0。T0 和 T1 共享资源，各自单项 ii 又都是 1；同一周期内无法错开，跨到下一周期又会撞上同相位的另一次发射。两个方向不能同时满足，因此 UNSAT 有直接的资源解释。

II=2 给两个拷贝留下两个不同相位。较大的 latency 仍由 k\_i 吸收：T2 可以位于逻辑轮次坐标 3，而发射资源每 2 个周期就接续一次。流水填充后进入稳态，每 2 个周期启动一轮新迭代，也每 2 个周期完成一轮迭代；启动与完成之间保持固定的流水深度偏移。首轮填充和末轮排空仍由流水的前导与尾部处理。

自动串行 fallback 是另一种结果。**fallback** 指专用优化无法给出解时退回保守实现。只有所有候选 II 都得到明确 UNSAT，自动路径才可退回串行；unknown 会被报告为求解不确定。手工指定的调度若不可行则直接报错，避免悄悄改变用户意图。

7. buffer versions：用存储换跨迭代距离

一份调度还要说明同一逻辑 buffer 同时保留多少份物理拷贝。这个数量叫 **buffer versions**。把这些版本按取模下标循环使用，就形成一个 **ring**：第 0 轮写版本 0，第 1 轮写版本 1，之后再回绕。

这里的取模下标只解释 ring 概念，并不表示本例已经求出具体版本数。本章这份 A+B 注解没有声明 buffer variables，真实求解返回 `versions=[]`。第四章 A 三版本、B 两版本的 ring 来自 multi-async 调度的下游流水物化，不属于本次 A+B 求解的返回数组。

讨论自动版本前，还要区分依赖指向哪一轮数据。**RAW** 是 read after write，即写后读。对固定迭代距离 d，跨迭代 RAW 写为：

公式：start\_v-start\_u ≥ L\_uv-d · II.

d=0 时，它退化为第 3 节同一轮生产者到消费者的 RAW；d ≥ 1 时，v 读取的是 d 轮之前 u 写好的数据，每一个迭代距离提供 II 的时间余量。距离必须来自数据语义，不能只为了让时间不等式成立而添加。

跨迭代依赖也可以用版本数换取时间距离。对自动决定版本数的 buffer b，约束写为：

公式：start\_v-start\_u ≥ latency-II · v\_b.

v\_b 是版本数，II · v\_b 是这些版本能够提供的跨周期时间余量。右端越大，消费者就需要比生产者更晚；增加版本数会降低所需的时间差，让生产者与消费者可以落在更远的逻辑轮次坐标上。

看一个反事实就能理解版本数为何必需。保持 T0 的 latency 为 4、开始时间为 1，并假设把消费者提前到 2，II=2，同时明确消费者读取的是旧迭代已经完成的 A，而非当前这一轮尚未完成的拷贝。若仍要求同一轮 RAW，时间差 2-1=1 小于 latency 4，增加版本也无法改变这项数据语义。只有在依赖明确指向旧迭代时，才能继续检查版本条件：若只有一个版本，约束变成 2-1 ≥ 4-2，即 1 ≥ 2，不成立；有两个版本时变成 1 ≥ 0，成立。第二个版本让消费者读取旧数据时，生产者可以写另一个物理位置。这个提前读取旧迭代的情形用于解释公式，真实解中的 T2 位于 6，并未要求这种提前读取。

版本数还受容量约束：

公式：∑\_b ∈ groupv\_b · size\_b ≤ C\_group.

它表示同一存储组里，各 buffer 的版本数乘单版本字节数，相加后不能超过容量 C\_group。另设一个纯算术例子：A、B 各 512 B，容量为 1536 B，则版本数 2 和 1 恰好占用 1536 B；两者都用两个版本就需要 2048 B，无法装入。调度器必须联合决定时间与存储，单独把版本数调到最大会越界。

8. 用手算检查求解结果

到这里，不运行 Z3 也能审查这份解：

- 三条数据依赖的差值分别为 5、4、2，都达到所需 latency；
- 两个读取任务的发射间隔为 1，循环 II 为 2，回绕方向也正好相接；
- 每个 start 都能分解成唯一的 k · II+r；
- II=1 的资源冲突证明 2 已经是最小值；
- 若选择提前消费或更深跨迭代重叠，还要重新核算版本数和容量。

这组检查解释了求解器为何需要这些变量，也给出后端必须保住的合同。下一章的 CUDA 任务边界、逻辑阶段、发射顺序和 buffer ring 都应保留这些关系；只把 `[1,2,6,8]` 打印出来而没有正确物化，仍不能算完成调度。

三、把同一调度放进 CUDA：周期估算与 flat Task/Stage

1. 整数周期从哪里来

第二章把 latency 和单项 ii 当作已知小数。真实编译器需要从任务内容估算它们，否则用户只能为每个任务手工填写所有数字。这里的估算是面向调度约束的整数周期模型，目标是提供一致的相对成本，不构成对设备绝对时钟周期的提前承诺。

本文设备目标是 **SM120**，即 compute capability 12.0 的 CUDA 架构目标；实验设备为 RTX 5090 Laptop GPU。硬件查询提供容量与执行资源上限，独立微基准提供吞吐。**微基准**是专门测量某类操作的小型实验，例如流式拷贝或一组矩阵乘加，它与正式候选的形状分开，避免用待评估候选的答案反推模型参数。

对一条存储路径，完成时间和发射间隔分别估算为：

公式：completion=base+⌈((bytes)/(bandwidth))⌉+descriptor,

公式：access\_ii=max(⌈((bytes)/(bandwidth))⌉,min\_ii)+issue.

第一式中，base 是路径固定开销，bytes/bandwidth 是搬运数据所需周期并向上取整，descriptor 是描述符相关开销；三者共同决定结果何时完成。第二式把传输占用、硬件最小发射间隔和发指令开销放进发射侧，决定同一路径多久能接续。

代入一组仅为说明单位的数字：固定开销 2 个周期，128 B 数据，带宽 64 B/周期，描述符开销 1，则完成时间是 2+⌈128/64⌉+1=5。若最小发射间隔为 1、发指令开销为 1，则访问 ii 是 max(2,1)+1=3。同一份搬运因此有 latency 5、单项 ii 3，等待结果与释放发射资源保持独立。

多个操作的组合也不能把两个量混成一个。若一个调度任务内先后组合 P、Q，模型使用：

公式：latency\_P+Q=max(P.latency, P.ii+Q.latency),

公式：II\_P+Q=P.ii+Q.ii.

第一式表示组合结果至少要等 P 完成，也要覆盖 P 发射后接上 Q 并完成 Q 的路径；第二式把同一串行组合的发射占用相加。重复和分支有各自组合规则，核心仍是分别传播完成与发射信息。

吞吐参数可以缺失，但缺失不等于免费。模型的 **measured-only** 模式要求所有必需速率都来自已通过的测量，否则拒绝构造；较宽松的模式会显式记录 fallback。这样，一次看似很短的估算不会被误当成经过硬件标定的事实。

2. flat Task：把源码边界交给调度器

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

这些规则服务于一个简单直觉：Task 决定依赖分析的原子粒度，Stage 决定用户是否已经固定 k\_i。若 stage 交给求解器，第二章的 start=k · II+r 仍由求解结果决定；若手工给出 stage，求解器只在这个额外约束下检查可行性。

3. 从 start/II 到 CUDA 能表达的阶段和顺序

求解器返回整数时间，CUDA 流水后端使用阶段、发射顺序和物理版本。投影发生在 [`src/transform/z3_pipeline_planning.cc`](../../src/transform/z3_pipeline_planning.cc)，第一步仍是对同一个 start 做除法：

公式：k\_i=⌊((start\_i)/(II))⌋, stage\_i=k\_i-min\_j(k\_j).

第一式取出逻辑轮次坐标，第二式整体减去最小坐标，使最早阶段从 0 开始。整体平移不改变任务间的时间差，只给后端一个规范起点。

代入 `[1,2,6,8]` 和 II=2，得到 `[0,1,3,4]`。T0 位于阶段 0，T1 位于阶段 1，T2 和 T3 分别位于阶段 3、4。两个输入并未被人为压进同一阶段，较长的准备距离保留在阶段坐标中。

相位由另一项给出：

公式：residue\_i=start\_i-k\_i · II.

它表示任务在周期内的发射位置。本例 residue 为 `[1,0,0,0]`。只有 stage 还不够，因为同一阶段内的多个任务仍需确定谁先发射；只有 residue 也不够，因为 T2、T3 虽同为 0，却处于不同阶段。后端综合相位、阶段和源码位置形成拓扑顺序。**拓扑顺序**是一种让所有依赖都从前指向后的排列。

下游流水需要的数组还有一个方向细节：物理顺序回答“第 p 个发射位置放哪条原语句”，下游 order 回答“原语句被放到哪个位置”。两者互为逆映射：

公式：inverse\_order\[physical\_order\[p\]\]=p.

例如物理位置 2 放原语句 0，则原语句 0 的 order 值就是 2。做逆映射可以避免把“排列里的内容”误当成“内容的位置”，尤其在阶段重排后。

最后，流水深度必须同时覆盖用户上界、派生阶段和物理 buffer 版本：

公式：cuda\_depth ≥ max(num\_stages, max\_i(stage\_i)+1, max\_b(v\_b)).

它表示实际物化深度至少取三类需求的最大值，输入的 `num_stages` 不会把更大的派生 stage 静默截断。对本例的阶段数组，仅阶段项就要求覆盖 5 层；若某 buffer 需要 3 个版本，版本项也不能低于 3。

这一步把第二章的两个直觉落到了代码结构：k\_i 主要变成阶段与跨轮次距离，r\_i 参与周期内发射顺序。buffer versions 则按存储身份传到流水注入，避免把“某个数组版本”错配到共享同一名字的另一块存储。

4. 为什么物化后仍要做数值验收

投影公式可以检查，生成的 CUDA 也可以编译，但最终验收仍是输入 A、B 后是否精确得到 A+B。公开 Task 用例走过了真实求解、CUDA 编译和设备执行，输出与 PyTorch 参考精确一致；PyTorch 是实验使用的张量计算库。

这些结果覆盖了 flat 公共标记到设备执行的闭环。公开 Stage 用例还检查了手工阶段与旧的手工阶段数组写法在相同意图下的等价数值行为。嵌套任务、混合执行域和无法分类的操作仍需明确拒绝。一次 SAT 只说明整数约束有解，一次仅在 CPU 端完成的结构检查只说明 IR 具备某些形状，设备数值结果才把数据语义纳入验收。

至此，内部调度的分工已经完整：周期模型估算任务，Z3 选择时间，Task/Stage 提供边界，投影生成 CUDA 流水，数值测试检查结果。第五章的成本模型会再次使用硬件吞吐，却服务于另一目标：给多个完整 kernel 配置排序，并决定哪些值得付出真实编译和 profiling 成本。

四、求解可行之后：编译器必须接住的三类语义

1. SAT 与可执行代码之间的间隙

第二章的约束能证明时间安排满足已编码条件，却没有自动证明后端能用 CUDA 的同步与存储机制表达这份安排。一个常见间隙是：数学依赖写着“v 等待 u”，生成代码却只有常规 block 同步，而 u、v 实际由不同线程组执行；另一个间隙是：依赖指向旧迭代的值，物化后的代码却重新读取了新迭代的存储。

下面三个案例都从这个间隙出发。每个案例先给出一个会出错的小场景，再说明编译器需要增加什么信息，最后用真实设备数值检查限定范围。它们检验的是内部调度正确性，与下一章的 GDN 搜索加速没有因果关系。

2. 两个 owner 的交接：正向到达还不够

**owner** 在这里指拥有一段流水工作的执行所有者，可以把它理解为一个独立 warp group。**warp group** 是一组协同执行同一角色的 warp，而 warp 是 CUDA 的 32 线程执行单位。不同 owner 不能依赖一条普通的单线程程序顺序自然看见彼此进度，需要显式交接。

已验证的 owner-ring 案例有两个有序 owner：P 负责生产，C 负责消费。这里的 **role** 就是生产或消费职责。二者共享一个 CTA-local 的 buffer ring；CTA-local 表示只在当前 block 内使用。P 取得空版本并提交满版本，C 等待满版本、读取后释放空版本。

```text
P: acquire(empty[r]); 写入 A; commit(full[r])
C: wait(full[r]);    读取 A; release(empty[r])
下一轮 P 必须等 C 释放，形成反向交接
```

若只写 P→C 的正向等待，C 会等到数据，却可能在 P 已回绕并覆盖该版本后才读取。C→P 的释放方向保护的是存储寿命：**lifetime** 指一份数据从可读取到不再需要的区间。正向和反向交接共同保证“数据已到达”和“存储可复用”。

这个受限案例的正向交接延迟为 1，反向为 2。两个任务开始时间为 0 和 1，最小循环 II 可以直接从一周交接看出：

公式：II ≥ h\_P → C+h\_C → P=1+2=3.

它表示从 P 在本轮提交，到 C 允许 P 在后续轮次复用，必须容纳两个方向的交接间隔。II=3 时，P 在 0、C 在 1，下一次 P 在 3；从 C 到下一次 P 的差值正好是 2。II=2 会让反向交接只剩 1，因而不满足约束。

求解结果因此是最小 II=3、residue 分别为 0 和 1。真正物化还需要把 acquire、commit、wait、release 放到与 ring 回绕一致的位置。深度 1、2、3 的设备用例都重复检查精确输出，覆盖了回绕与重复调用。

支持范围保持在这个显式子集：两个有序同级 owner、两个固定 role、block 内共享 ring、规定的 stage-zero epoch 和交接结构。**epoch** 可理解为 ring 完成一轮后推进的逻辑代次。任意 owner 数量、自动推断 owner、带条件或嵌套的交接，以及其他异步 epoch 形式都不能由这组结果外推。

这个案例说明，owner exclusion 约束需要对应真实的双向同步协议。求解器给出 3 个周期的节奏只是起点，交接指令和版本回绕正确才让数值语义成立。

3. 标量快照：消费者需要旧值，重放定义却读到新值

**标量**是单个数值。IR 中的 **Bind** 把一个表达式绑定到变量，例如 `value = source[0] + 1`；可以把它理解为带名字的标量定义。若表达式只读取输入、没有外部副作用，就称其为 **pure**。

考虑 copy producer 在 stage 0 更新 `source`，consumer 在 stage 1 使用 `value`。如果后端只是把 Bind 的表达式复制到 consumer 位置，下面的代码看似保留了定义，实际却改变了读取时刻。

```text
stage 0: source[...] = A[k, ...]
Bind:    value = source[0] + 1
stage 1: B[k, i] = value + i
下一轮 producer 可能先覆盖 source，再执行 stage 1
```

原定义若在原 stage 求值，它能读取第 k 轮快照；重放到 consumer 后，`source[0]` 可能来自下一轮。问题出在读取的对象随时间变化，表达式本身并没有写错。

已验证的处理是在 consumer 位置构造一个 **clone**，即原纯标量定义的独立副本，并给副本新的变量身份。随后编译器增加表示快照关系的虚拟读写依赖，重新求解。虚拟依赖不新增数学输出，它用于告诉调度器：clone 必须读取当前 producer 快照，同时下一次 producer 写入必须等这份快照得到保护。

真实例子的原问题有 3 个任务，clone 加入后变成 4 个，两次求解的 II 都是 31。原开始时间为 `[0,30,32]`；扩充后 producer、原 Bind、clone、consumer 的时间分别为 0、30、31、33，对应阶段 `[0,0,1,1]`。clone 在 31 开始、latency 为 2，consumer 在 33 开始，二者正好衔接。`source` 需要两个版本，用来隔开下一轮覆盖。

这个数值还揭示了一个容易漏掉的点：仅在 consumer 处复制表达式并不充分。若没有两个 source 版本和相应依赖，clone 仍可能读到被覆盖的数据；若只增加版本却不放对 clone 位置，也无法保证它读取预期快照。定义位置、依赖和物理存储需要一起修改。

设备测试覆盖直接 buffer 和共享同一底层存储的别名视图。**别名**是两个名字指向同一底层存储的情况；依赖分析若只比较变量名，就会漏掉这类覆盖。两种形式都重复执行精确输出检查。

当前范围限于 flat、pure、已纳入调度的 scalar Bind。带副作用或无法看清语义的定义不能随意复制，任意控制流和异步 producer 的 clone 也不在这组证据内。不可克隆定义需要更保守地留在同阶段，不能假设重放总能保持值不变。

4. 异步等待：两个 producer stage 需要等谁

同步拷贝可以按“返回后数据可用”理解；**异步拷贝**把发起和完成分开，让执行线程先做其他工作。`cp.async` 是 CUDA 的一条异步拷贝路径。producer 是产生数据的任务，consumer 是读取数据的任务。

异步拷贝通常按 **commit group** 组织：一次 commit 把此前一组异步操作归入可等待的批次。`wait<N>` 的直观含义是等待到最多只剩最近 N 个组仍允许未完成，因此 N 越小，等待越强；`wait<0>` 会排空相关组。等待太弱会读旧数据，等待太强又会消除本来可用的重叠。

继续使用第二章的四任务形状，这个多异步来源（multi-async）例子的开始时间仍为 `[1,2,6,8]`、II 为 2，对应 producer 阶段 0、1，同步 consumer 阶段 3，输出阶段 4。两个 producer 写不同 buffer，消费者同时读取它们。

```text
stage 0: cp.async A; commit
stage 1: cp.async B; commit
stage 3: 对 A、B 所需的来源分别 wait，再计算 C=A+B
stage 4: 写回 C
```

困难在于 consumer 面对两个来自不同阶段的异步来源。只按最近一条拷贝决定等待，可能漏掉较早阶段尚未完成的另一份输入；对所有情况统一 `wait<0>` 虽然保守，却可能等待与本次读取无关的较新组。

实现先要求每个 producer stage 能映射到稳定的物理 commit group。稳定表示生成代码中的提交位置不会随控制路径漂移，编译器能建立来源与批次之间的明确对应。随后分别计算 A、B 完成所需的保守 wait，再取能够满足全部来源的最小 N。取最小值是因为更小的 N 更强，一次等待即可覆盖各个来源的要求，无需连续做多次重复等待。

12 轮案例还检查了下游 multi-async 流水物化形成的物理 ring：A 使用三个版本，B 使用两个版本。第二章 A+B 求解没有声明 buffer variables，返回的是 `versions=[]`；这里的 3/2 属于后续存储物化结果，并非那次求解返回的版本数组。这个差别与两个 producer 所处阶段及后续覆盖距离相呼应，也说明 stage、wait 和版本需要共同验收。只看生成代码里存在 `cp.async`，无法证明 consumer 等到了正确批次；只看 ring 大小，也无法证明同步强度正确。

设备用例覆盖 1、4、5、12 次迭代和重复精确输出。1 次迭代检查前导与尾部，4、5 次迭代跨过不同回绕位置，12 次迭代让较深 ring 和尾部 drain 充分出现。**drain** 是循环结束前等待剩余异步工作完成，防止最后一次输出尚未就绪就退出。

已验证范围是稳定 `cp.async`、同步 consumer、不同 buffer，并且能够证明所需 wait 最小值的情形。异步 consumer 再读异步结果、同一 buffer 的跨阶段多个写入者、其他异步搬运协议、条件 producer 和不稳定 commit group 都需要其他处理或明确拒绝。

5. 三个案例共同改变了什么

三个案例需要的信息各不相同：owner-ring 增加双向交接与回绕协议，scalar clone 增加定义副本和快照依赖，multi-async 增加来源到 commit group 的映射与等待强度。它们共同把“何时执行”推进到“生成代码后，值仍在预期时刻来自预期位置”。

一条可信的内部验收链因此包含约束求解、IR 结构、同步与版本检查、设备数值输出。任何单层结果都不能替代其余层级。下一章出现的 GDN 与经典算子性能数字全部来自外层候选搜索；上述三个案例没有参与那些加速的产生，也不应被用来解释那些数字。

五、从候选到证据：五臂搜索、GDN 与模型的失误

1. 外层问题从一份候选清单开始

第二至四章都假设 kernel 的配置已经固定，调度器负责让这个配置正确重叠。自动调优从更早一步开始：同一个数学计算允许不同的 tile、线程数和流水深度，每个选择组合都是一个候选。候选越多，越可能覆盖好配置，编译和测量成本也越高。

把用户声明的原始候选集合记为 R，即 raw space。去重并经过必要资源条件筛选后保留的集合记为 U，实际送进编译和 profiling 的子集记为 S。三者满足 S ⊆ U ⊆ R；legal 指 U 中通过必要条件或因 unknown 而保留的候选。stock-raw 是一个例外路径，它直接测量原始 R，让真实失败显式发生，用作成本与结果参照。

约束剪枝只删除得到 UNSAT 的候选。unknown、超时和资源预算耗尽都留在 U 中，因为“本次没有证明”不能替代“已经证明不可行”。后续的编译失败也不会倒填成 UNSAT：求解器只对自己编码过的必要条件负责。

GDN 的前向（forward）提供了一个能手算的例子。DV 是输出值向量的特征维度，调优旋钮 `block_DV` 是该维度的块大小，原始取值为 16、32、64、128。**动态共享内存**是 kernel 启动时按配置指定的 block 共享存储；opt-in 上限是显式允许后每个 block 可请求的上限，本机为 101376 B。该配置的必要分配为：

公式：B\_fwd=38912+576 × block\_DV.

它表示固定部分 38912 B 加上随 DV 块线性增长的存储。四个候选分别需要 48128、57344、75776、112640 B。只有 128 对应 112640 B，超过 101376 B，因此 R 中 4 个候选剩为 U 中 3 个。stock-raw 仍会让这个候选走到启动并记录失败，guided 路径则在编译前省去已知无效工作。

GDN 的反向（backward）有两个旋钮：`prepare_h_num_stages` 是 prepare\_h 部分的流水深度，`group_reduce_block_size` 是分组归约使用的块大小。**归约**是把多个值合并为一个结果，例如求和。三个深度与三个块大小形成 9 个原始候选，其中 prepare\_h 的必要分配为：

公式：B\_prepare\_h=57344+18432 × stages.

它表示每增加一层流水需要额外 18432 B。深度 3 需要 112640 B，因此三个对应的 group 配置都被排除，留下 6 个候选。其他没有形成证明的资源需求仍保留为未知，不因这两条公式而被假定为合法。结果部分把 `prepare_h_num_stages` 简写为 stage，把 `group_reduce_block_size` 简写为 group。

2. 成本模型怎样给保留候选排序

合法只回答“能否运行”，排名还要估计“可能多快”。`CudaCostModel` 接收算子、固定形状和一个配置，输出 `predicted_ms` 及资源分项。predicted\_ms 是用于排序的预测毫秒分数，真实 latency 需要后续测量。

模型先把配置展开成若干次 kernel launch。一次 **launch** 是向 GPU 提交一次 kernel 执行；一条公开算子管线可能包含多次 launch。每次 launch 再分成串行 phase，**phase** 指源码中具有明确先后的一段工作。模型统计每段的计算量、全局与共享存储流量、同步次数和流水深度。

对单个 block，一项资源时间可按“工作量除以该资源吞吐”估算。整卡吞吐先按 SM 数分摊；**SM** 是 streaming multiprocessor，即 GPU 上调度并执行 block 的流式多处理器。配置还决定每个 SM 能同时容纳多少 block，模型用 `occupancy_ctas_per_sm` 记录这个 block 数。它不同于 CUDA 通常所说的 active-warp occupancy，后者是活跃 warp 数占硬件最大 warp 数的比例；本文 waves 计算使用的是前者。需要执行的总 block 按批次铺开，批次数叫 **waves**：

公式：waves=⌈((block\_count)/(SM\_count × occupancy\_ctas\_per\_sm))⌉.

它表示总 block 数除以整卡每波可同时容纳的 block 数，再向上取整。若 `occupancy_ctas_per_sm` 因共享内存增加而下降，同样多的 block 就需要更多波次，可能抵消更深流水带来的局部收益。

对单个 phase，模型用简化规则描述流水重叠：

公式：t\_overlap=\[cases\]; ∑\_r t\_r, stages=1,;; max\_r t\_r, stages ≥ 2.;

t\_r 是主要资源 r 的估算时间。单层流水把计算、全局路径和共享路径时间相加；两层及以上用最大资源时间表示理想重叠。额外的发射与同步成本再计入 phase 分数，各 launch 分数相加形成 predicted\_ms。

假设某段计算、全局路径、共享路径分别估计为 6、4、2 ms。单层模型给出 12 ms，多层理想重叠给出 6 ms。这个 6 ms 清楚地表达了模型假设：较慢资源完全遮住其他资源。真实 kernel 若受依赖、缓存替换、bank conflict 或尾部波次影响，实测可能更长。**bank conflict** 是共享存储访问落到冲突银行而降低吞吐的现象，当前模型把它列为近似边界。

标定吞吐来自独立微基准，measured-only 模式拒绝缺失或非测量来源。若物理容量信息不足，候选得到空分数并标记 unknown，仍留在保留集合中；当实测预算覆盖到它时，继续通过真实编译验证。稳定排序使用分数和规范化配置身份处理同分，避免字典插入顺序暗中决定结果。

成本模型的职责因此很窄：给 U 中的候选安排实测优先级。它不会修改第二章的 start、II 或 buffer ring，也不能凭 predicted\_ms 宣称真实 kernel 获得某个延迟。

3. 五臂为什么能拆开收益来源

实验中的每一种候选选择策略叫一个 **arm**。五个 arm 共享候选 kernel 构造函数、数值校验和计时器，差异只限于候选生成与选择。top-k 表示按预测排序后最多取前 k 个，两个等预算控制使用相同的 k。

| arm | 实际选择方式 | 它帮助回答的问题 |
| --- | --- | --- |
| stock-raw | 完整原始集合 | 不筛选时，总成本与实测答案是什么 |
| z3-legal-exhaustive | 完整保留集合 | 只删除已证明无效候选能省多少 |
| z3-cm-topk | 保留集合中的预测前 k 个 | 约束与排序结合后能否进一步省成本 |
| budget-raw | 保留集合按原顺序取前 k 个 | 同样少测候选，原顺序会得到什么 |
| budget-random | 保留集合中固定随机种子的 k 个 | 同样预算下，随机覆盖会得到什么 |

第一臂提供原始基线，第二臂隔离资源剪枝的贡献，第三臂是 guided 路径，后两臂审查“少测就更快”这一平凡收益。若 guided 比 stock 快，却不比两个等预算控制好，最终结果可能主要由预算缩小造成。

选定 S 后，`GuidedAutoTuner` 把候选交回 stock AutoTuner。每个候选仍要真实构造和编译，使用独立数学参考检查输出，再通过 CUDA event profiling。CUDA event 是放在 GPU 工作流中的时间戳，用来测量事件之间提交的 GPU 工作。构造失败、启动失败、数值失败和成功 profile 分开记录，失败候选仍占搜索时间与尝试分母。

**搜索 wall** 使用端到端墙钟时间，包括 guided 的求解和模型调用。求解与模型本身在本例中只占很小比例，但它们没有被放到计时区间之外。不同阶段的并行累计时间不能简单相加替代 wall。

每个 arm 做三次 cold 搜索。cold 指新的 Python 进程和事先为空、由本次运行拥有的实验缓存，清理范围不扩展到共享用户目录或整机系统缓存。搜索时 warmup 为 10、重复测量 50，并关闭 early stop。**warmup** 是正式采样前的预热执行；**early stop** 是看到部分结果后提前结束候选测量的策略，关闭它可让各 arm 的测量口径保持明确。

最终入选配置叫 **winner**。搜索期的选择规则是 candidate-median：对同一 arm 中每个实际测得候选，取它在三次 cold 中的 latency 中位数，再选最小者。搜索结束后冻结配置，之后才做独立确认，确认数据不参与挑选。

独立确认采用 7 轮 ABBA。ABBA 指一轮内按 A、B、B、A 的顺序测两个配置，用对称顺序减弱时钟漂移对某一方的固定偏向。不同配置每轮做 50 次 warmup，再采集 200 个清 L2 后的 CUDA-event 样本；L2 是 GPU 的二级缓存，清理它用于减少前一次测量留下的数据复用。若两个 arm 选择同一配置，就按身份处理，不人为制造两组“对照”样本。

4. GDN T8192：主结果怎样得到

GDN 实验使用公开 FlashQLA 前向与反向管线，输入为 BF16。**BF16** 是一种 16 位浮点格式，保留较宽指数范围、减少尾数位。主形状包含 T=8192 个序列位置，forward 检查输出和最终状态，backward 检查六类梯度。数值参考使用独立的 float64 双精度路径，relative-L2 门槛为 0.02；relative-L2 是整体误差向量长度相对参考向量长度的比例，这里的长度指各元素平方和再开根号。

**held-out** 表示该形状在模型系数、空间和预算冻结前没有用于挑选方案。T4096 先作为 validation，即开发方案的检查形状；T8192 随后作为 held-out 主结果。这样，主结果检验的是既定机制能否迁移到新形状，降低针对同一形状反复调参后报告好成绩的风险。

forward 的预算是 top-2，backward 是 top-3。下表同时给出三次 cold 的搜索成本与后续独立确认。compile/profile 是实际尝试次数；wall 是三次端到端时间的中位数；speedup 是逐次 stock wall 除以对应 arm wall 后再取中位数；confirmed 是冻结配置后的独立测量 latency。

| 方向与 arm | compile / profile | wall（秒） | 搜索加速 | confirmed（ms） |
| --- | ---: | ---: | ---: | ---: |
| fwd stock-raw | 4 / 3 | 34.102 | 1.000× | 0.308992 |
| fwd legal-exhaustive | 3 / 3 | 26.239 | 1.311× | 0.308992 |
| fwd guided top-2 | 2 / 2 | 19.410 | 1.757× | 0.308992 |
| fwd budget-raw | 2 / 2 | 19.362 | 1.765× | 0.308992 |
| fwd budget-random | 2 / 2 | 20.060 | 1.704× | 0.308992 |
| bwd stock-raw | 9 / 6 | 62.690 | 1.000× | 4.230792 |
| bwd legal-exhaustive | 6 / 6 | 56.734 | 1.105× | 4.230792 |
| bwd guided top-3 | 3 / 3 | 49.599 | 1.258× | 4.230792 |
| bwd budget-raw | 3 / 3 | 49.813 | 1.253× | 4.379024 |
| bwd budget-random | 3 / 3 | 55.322 | 1.139× | 4.233904 |

forward 的 guided 编译次数减少一半，成功 profile 从 3 减到 2，按逐轮搜索加速中位数换算的墙钟节省约 43.1%。五个 arm 最终都选择 `block_DV=16`，所以 confirmed latency 完全相同。等预算原始顺序甚至取得 1.765× 的搜索加速，略高于 guided 的 1.757×；这说明本形状的原顺序已经覆盖了好配置，预测排序没有形成额外最终配置优势。

backward 的 guided 编译和成功 profile 分别从 9、6 减到 3、3，按逐轮搜索加速中位数换算的墙钟节省约 20.5%。guided 与合法空间穷举都选择 stage 2、group 8，因此独立确认同为 4.230792 ms。budget-raw 选择 stage 1、group 8，确认 latency 为 4.379024 ms；budget-random 选择 stage 2、group 32，与 guided 很接近。

不同配置对的确认还使用该配置对自身的逐轮中位数计算比值。guided 对 budget-raw 的比值为 0.962739，对 budget-random 为 0.999626。后一个数非常接近 1，逐轮差值有正有负，因此它支持最终配置近似，不支持稳定的 kernel 提速宣称。相同配置的对比按身份处理，比值就是 1，不重复采样制造差异。

这些数字描述的是**搜索过程**加速。forward 和 backward 的 guided 与完整参照若选中同一配置，其 kernel 相同；1.757× 和 1.258× 不能写成该 GDN kernel 的执行加速。第四章的 owner-ring、scalar clone 和 multi-async 也没有参与这些候选的收益归因。

5. 排名质量：总体相关不等于保住第一名

只看最终 winner 会漏掉模型怎样犯错。实验把同一轮的预测顺序与实测顺序配对，计算 Spearman 和 Kendall 排名相关。两者都衡量预测是否倾向于把快候选排在前面，越接近 1 表示顺序越一致；它们不直接保证特定第一名正确。

forward 三轮的两种相关都为 1，候选很少且顺序完全吻合。backward 三轮中位数分别为 0.882735 和 0.787726，整体顺序较好。实测最优在每轮都进入预测 top-3，所以 guided 预算覆盖了它。

backward 的预测第一名却没有命中实测最优。stage 2、group 16 与实测最优 stage 2、group 8 得到相同预测分，规范化配置身份把 group 16 排在前面。这个结果解释了为何 top-3 预算有效，也暴露了同分处理和 top-1 结论的脆弱性。预测排名相关较高与冠军并列或漏排可以同时成立。

**oracle rank** 指实测最优候选在模型预测顺序中的位置，分母是有有限预测并可配对的候选集合。它是检查头部覆盖的直接指标：若 oracle rank 为 16，top-10 无论整体相关系数多高都不可能选中该最优。下一组经典算子用这个指标展示更明显的失误。

6. 经典算子：搜索很快，仍可能漏掉最优

经典实验采用三次 cold 和 top-10 预算，随后同样冻结 winner 并做独立确认。**工作负载**（workload）在这里指特定算子、形状和输入配置。GEMM 是通用矩阵乘法，K 是其归约维长度；Conv2D 是二维卷积，stride 是滑动步长，tail 是边界不足一个整块的部分；causal FlashAttention 是只允许注意力看到当前及之前位置的融合算子。它们用于检验模型在 GDN 之外的排序表现。

下表的 guided/质量参照使用独立确认的配置对配对比值，大于 1 表示 guided 更慢；该比值不要求等于相邻两个各自独立测量的 latency 的简单商。GEMM 与 Conv 使用完整合法空间穷举的质量参照。FlashAttention 的合法穷举仍有失败候选，其参照是完整 stock-raw 中成功候选的实测最优，不能标成完整合法穷举。

| 工作负载 | guided 搜索加速 | guided（ms） | 质量参照（ms） | guided / 质量参照 | 实测最优预测名次 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 方形 GEMM | 26.404× | 1.472920 | 1.448576 | 1.017253 | 32–33 / 240 |
| 短 K GEMM | 26.092× | 0.428928 | 0.410432 | 1.039685 | 120 / 240 |
| Conv2D stride tail | 13.218× | 0.016960 | 0.013088 | 1.295844 | 16 / 120 |
| causal FlashAttention | 4.321× | 0.144040 | 0.140096 | 1.021014 | 27 / 36 |

四个工作负载都大幅缩短搜索，实测最优却都没有进入 guided top-10。方形 GEMM 的最终回归约 1.73%，短 K 约 3.97%，Conv 约 29.58%，FlashAttention 约 2.10%。这些差距与搜索加速处于同一结果中，不能只保留前半句。FlashAttention 的七轮配对差值存在反向，2.10% 的点估计不应被解释为统计上确定的回归。

FlashAttention 的 54 个保留候选中有 36 个有限预测，所以表内最优预测名次以 36 为分母；排名相关另按 30 个成功 stock-raw 实测的配对项计算。两个分母对应不同问题，不能混写。

Conv 尤其说明整体相关的局限：Spearman 中位数为 0.80170，看似能大致排列候选，实测最优却稳定排在预测第 16，top-10 结构性漏掉它。FlashAttention 的 Spearman 中位数甚至为 -0.39058，模型整体存在反向倾向；guided 仍找到接近最优的配置，也没有保住真正最优。

等预算控制进一步限制结论。方形 GEMM guided 优于两个控制，短 K 与随机对照的差异逐轮有正有负，Conv 与随机对照选中同一配置，FlashAttention 也落后于一组控制的共同 winner。模型排序应按 workload 验收，top-k 是实测预算，不构成最优或近优保证。

7. 失败记录怎样约束最终结论

T4096 validation 曾使用旧的 modal-winner 规则，即按重复搜索中最常胜出的配置确定选择；当时 backward guided/control 比值为 1.000624，严格“不更慢”验收失败。统一改用 candidate-median 后，模型系数、候选空间和预算保持冻结，T8192 才作为新的 held-out。旧失败与新主结果属于不同形状和证据集合，不能拼接成一次从未失败的实验。

GDN backward 的 stage 1、group 8 还曾在首轮出现一个梯度输出 `dv` 的数值不一致。原始记录保留；同输入复测和后续 1000 次公开 backward 重复校验通过，降低了稳定算法错误的可能性，却不能证明所有偶发竞态都不存在。性能表沿用真实候选分母，没有通过重跑挑掉这次异常。

缓存也会改变收益构成。no-cache 表示禁用 TileLang 与 CUDA 编译缓存，warm 表示先预热、再用新的 AutoTuner 实例在同一隔离缓存中测量。单次 T4096 敏感性中，no-cache 的 stock/guided 搜索时间比为 forward 1.843×、backward 1.276×，warm 时降为 1.039×、1.165×。原因很直观：编译缓存命中后，少编译候选所省的时间占比下降，校验、profile 和固定开销仍在。正式主结果继续使用 T8192 三次 cold，这组单次敏感性不替代它。

综合主结果与反例，资源约束的价值较清楚：已证明超容量的候选可以提前删除。成本模型的价值则取决于具体 workload：它可能在小预算内保留好配置，也可能把实测最优排到预算之外。可靠的使用方式仍然包括等预算对照、真实数值校验、独立 winner 确认，以及对 oracle rank 的审计。

六、带走什么，怎样复现

1. 用五个问题重建整篇文章

读完四任务例子后，可以用五个问题检查是否真正理解机制。

第一，为什么 latency 4 的拷贝可以只有单项 ii 1？因为结果完成时间与发射资源释放时间不同。消费者受前者约束，共享同一资源的下一项任务受后者约束。把二者混用会分别造成过早读取或过度串行。

第二，为什么最小循环 II 是 2，而开始时间却到 8？因为 II 描述周期重复，start=k · II+r 同时保存逻辑轮次坐标和周期内相位。T3 的 start 8 位于 k=4，并不表示循环要每 8 个周期才能接下一轮。

第三，为什么 SAT 之后还要看 CUDA 物化？因为约束只覆盖已经编码的语义。owner 双向交接、标量旧迭代快照和异步 commit group 都要求后端同步、依赖和存储版本保持一致。设备数值输出是这条正确性链的最后一环。

第四，为什么资源剪枝与成本模型要分开评价？UNSAT 给出不可行证明，删除它不会改变合法答案集合；预测分数只安排测量顺序，可能把真实最优排在 top-k 之外。前者看证明是否可靠，后者看头部候选覆盖与最终配置质量。

第五，为什么 GDN 的 1.757× 和 1.258× 是搜索加速？因为计时对象包含候选构造、编译、校验和 profiling。guided 与完整穷举选中同一配置时，kernel latency 相同。内部三个正确性案例也没有为这些搜索数字提供性能贡献。

2. 机制应该怎样使用

若你正在写一个显式 Z3 流水，先确定任务边界是否适合 flat Task：body 的执行域应清楚，完整读写应能作为同一原子单位分析。需要手工固定逻辑轮次时，再在完整 Task 外使用 Stage。求解后应同时检查 start、II、阶段、顺序、版本和生成代码的数值结果，单独查看最小 II 很容易漏掉数据语义。

若你正在缩小 AutoTuner 空间，先写下能够证明的必要资源条件。无法判断的候选保留为 unknown，让真实编译继续裁决。随后再使用成本模型分配 top-k，并至少保留一个同预算原顺序或随机对照。最终报告应同时包含搜索 wall、真实尝试分母、冻结配置和独立确认 latency。

若模型在一个 workload 上漏掉最优，增大 k 是一种预算选择，本身不构成修复保证。应查看实测最优的预测名次、同分规则和未建模因素，再决定增加预算、修改模型或退回完整穷举。经典 Conv 的 29.58% 回归说明，搜索时间的大幅节省可以伴随明显的配置质量损失。

更换 GPU 时，需要重新查询硬件并标定吞吐。RTX 5090 Laptop / SM120 的速率不能直接套到另一设备；任务周期估算与整 kernel predicted\_ms 也要各自检查单位和测量来源。

3. 文章自身的最小检查

所有命令从仓库根目录 `/home/zhaosiying/codebase/tilelang-cuda-z3` 执行。文章构建器只做 CPU 文本转换，不运行 TileLang 或 GPU。单元测试命令是：

`PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s article/tests -v`

完整预检命令是：

`python3 article/build_html.py --config article/index.json --check-only`

预检会解析 `article/index.json` 中的六章正文和短附录，检查本地证据、附件、表格和 KaTeX 公式，但不会写出 HTML、PDF 或 ZIP。LaTeX 是公式排版语法，KaTeX 是页面中的公式渲染器。知乎版本中的表格不含 LaTeX；所有公式都由同一 Markdown 源转换。

4. 内部调度的设备复现

设备用例需要 compute capability 12.0（SM120）的 GPU、与当前 TileLang 环境兼容的 CUDA Toolkit，以及已配置的 Python 和测试依赖。compute capability 12.0 描述 GPU 架构能力，CUDA Toolkit 版本是另一项软件依赖，不能将两者合并写成“CUDA 12.0”。使用项目环境所选的解释器运行：

`python3 -m pytest -q testing/python/language/test_tilelang_cuda_z3_pipeline.py`

该文件包含公开 A+B Task、公开 Stage、flat 纯 scalar clone 和受限 multi-async 的真实求解与数值用例。重点核对公开/multi-async 四任务的 start `[1,2,6,8]`、最小 II 2，以及 clone 用例的两次求解和两个 source 版本。生成代码中的同步与版本检查应和数值结果一起阅读。

owner-ring 的独立入口是：

`python3 -m pytest -q testing/python/language/test_tilelang_cuda_z3_owner_ring_integration.py`

它只覆盖第四章限定的两个 owner、两个 role 和显式交接结构。因环境而被标记为跳过（skip）与测试通过不同；复现记录应保留实际执行状态，不能把 skip 当成数值验证。

这些设备命令用于正确性复现，不需要运行 GDN 搜索，也不产生第五章的搜索加速结论。

5. GDN T8192 搜索与确认

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

七、附录：证据索引与复现入口

1. 调度与设备正确性

正文使用的小数推导和受限设备案例可由以下原始记录交叉核对。日志中的 SAT 只覆盖约束求解，数值结论对应各自设备测试。

- 四任务 A+B、公开 Task/Stage、scalar clone 与 multi-async：[SM120 集成日志](evidence/G3/z3-pipeline-sm120-v6.txt)；multi-async 的 CPU 端同步与非法输入检查见 [multi-async-host-v2.txt](evidence/G3/multi-async-host-v2.txt)。
- 两 owner、两 role ring：[owner-sm120-v6.txt](evidence/G3/owner-sm120-v6.txt)；测试源码为 [`test_tilelang_cuda_z3_owner_ring_integration.py`](../../testing/python/language/test_tilelang_cuda_z3_owner_ring_integration.py)。
- 任务成本公式：[latency-model-tests.txt](evidence/G2/latency-model-tests.txt)。这些是合成输入的公式检查，不提供设备吞吐结论。
- 早期边界与失败仍保留：[owner 编译容量失败](evidence/G3/owner-sm120-v1.txt)、[owner 调用接口失败](evidence/G3/owner-sm120-v2.txt)、[clone 早期流水生成失败](evidence/G3/z3-pipeline-sm120-v1.txt) 和 [multi-async 旧 guard 失败](evidence/G3/device-integration-test.txt)。后续受限用例通过没有改写这些历史记录。

flat 主链源码入口为 [`schedule_hint.py`](../../tilelang/cuda/language/schedule_hint.py)、[`normalize_z3_schedule_units.cc`](../../src/transform/normalize_z3_schedule_units.cc)、[`z3_scheduler.py`](../../tilelang/cuda/transform/z3_scheduler.py) 和 [`z3_pipeline_planning.cc`](../../src/transform/z3_pipeline_planning.cc)。

2. 硬件与标定

- 正式实验的设备属性：[hardware-sm120.json](manifest/hardware-sm120.json)；模型消费的标定视图：[calibration-sm120.json](manifest/calibration-sm120.json)。
- 独立微基准汇总：[calibration-summary.json](evidence/G2/calibration-summary.json)。测量协议、单位换算、正确性与指令审核入口见 [`benchmark/z3_cost_model/calibration/README.md`](../../benchmark/z3_cost_model/calibration/README.md)。
- 新设备应重新运行 [`calibrate.py`](../../benchmark/z3_cost_model/calibration/calibrate.py)，并同时更新硬件与标定输入。只替换显卡名称不构成新设备标定。

3. GDN held-out 与确认

- 三轮 cold 五臂汇总：[summary.csv](evidence/G6/gdn-heldout-t8192-v1/analysis/summary.csv)；预测配对与 oracle rank：[prediction-rank.json](evidence/G6/gdn-heldout-t8192-v1/analysis/prediction-rank.json)；进程和缓存协议：[protocol-audit.json](evidence/G6/gdn-heldout-t8192-v1/protocol-audit.json)。
- 独立确认：[confirmed-summary.csv](evidence/G6/gdn-heldout-t8192-confirmation-v1/analysis/confirmed-summary.csv)；七轮配对差值：[gdn-paired-differences.csv](evidence/G6/gdn-heldout-t8192-confirmation-v1/analysis/gdn-paired-differences.csv)。
- 资源公式核对：[forward shared-memory](evidence/G5/gdn-fwd-shared-memory.json) 与 [prepare\_h shared-memory](evidence/G5/gdn-prepare-h-shared-memory.json)。
- validation 与异常记录：[T4096 旧选择规则失败](evidence/G6/gdn-confirmation-v2/bwd/summary.json)、[backward 原始全配置记录](evidence/G5/gdn-bwd-all-configs.txt) 和 [后续 1000 次重复校验](evidence/G5/gdn-bwd-stage1-group8-determinism.txt)。
- 缓存敏感性仅作辅助解释：[no-cache summary](evidence/G6/gdn-cache-sensitivity-v1/analysis-no-cache/summary.csv) 与 [warm summary](evidence/G6/gdn-cache-sensitivity-v1/analysis-warm/summary.csv)。

驱动协议与参数说明见 [`benchmark/z3_cost_model/gdn/AUTOTUNE.md`](../../benchmark/z3_cost_model/gdn/AUTOTUNE.md)，正式 cold 入口为 [`scripts/run_gdn_cold_fresh.sh`](../../scripts/run_gdn_cold_fresh.sh)。

4. 经典算子与预测失误

- 搜索空间、独立参考和必要资源条件：[`benchmark/z3_cost_model/operators/README.md`](../../benchmark/z3_cost_model/operators/README.md)。
- 三轮 cold 主分析：[summary.csv](evidence/G6/classic-v2/main-analysis/summary.csv)；top-k 敏感性：[runs.csv](evidence/G6/classic-v2/sensitivity-analysis/runs.csv)。
- 独立 winner 确认：[confirmed-summary.csv](evidence/G6/classic-v2/confirmed-analysis/confirmed-summary.csv) 与 [confirmation-comparisons.csv](evidence/G6/classic-v2/confirmed-analysis/confirmation-comparisons.csv)。

阅读这些 CSV 时，应分别保留 raw、legal、selected、compile attempt 和成功 profile 的分母。FlashAttention 的 `legal_oracle_complete=false` 也应保留，不能把合法穷举写成完整实测参照。

5. 已接受捕获包

捕获验收状态与窗口绑定方法见 [capture-acceptance-v1.json](evidence/G7/screenshots/capture-acceptance-v1.json) 和 [capture-manifest.json](evidence/G7/screenshots/capture-manifest.json)。下列图片仅作为附件链接，不嵌入正文；性能数字仍以对应 CSV/JSON 为准。

- stock/guided 仅分析复算：[裁图](evidence/G7/screenshots/tilelang-cuda-z3-stock-guided-v3.png)、[全屏原图](evidence/G7/screenshots/tilelang-cuda-z3-stock-guided-v3-original.png)、[执行日志](evidence/G7/screenshots/tilelang-cuda-z3-stock-guided-v3.log)、[capture metadata](evidence/G7/screenshots/tilelang-cuda-z3-stock-guided-v3.json)、[crop metadata](evidence/G7/screenshots/tilelang-cuda-z3-stock-guided-v3.crop.json)、[窗口绑定](evidence/G7/screenshots/tilelang-cuda-z3-stock-guided-v3.png.window-binding.json)、[analysis.json](evidence/G7/screenshots/stock-guided-analysis-v3/analysis.json) 与 [summary.csv](evidence/G7/screenshots/stock-guided-analysis-v3/summary.csv)。它复算既有事件，没有新增 GPU 搜索样本；原始事件来自 [stock.log](evidence/G6/gdn-stock-guided-screenshot-v1/stock.log) 与 [guided.log](evidence/G6/gdn-stock-guided-screenshot-v1/guided.log)。
- GDN 独立确认汇总：[裁图](evidence/G7/screenshots/tilelang-cuda-z3-gdn-confirmation-v1.png)、[全屏原图](evidence/G7/screenshots/tilelang-cuda-z3-gdn-confirmation-v1-original.png)、[执行日志](evidence/G7/screenshots/tilelang-cuda-z3-gdn-confirmation-v1.log)、[capture metadata](evidence/G7/screenshots/tilelang-cuda-z3-gdn-confirmation-v1.json)、[crop metadata](evidence/G7/screenshots/tilelang-cuda-z3-gdn-confirmation-v1.crop.json)、[窗口绑定](evidence/G7/screenshots/tilelang-cuda-z3-gdn-confirmation-v1.png.window-binding.json)、[analysis.json](evidence/G7/screenshots/gdn-confirmation-analysis-v1/analysis.json) 与 [confirmation-comparisons.csv](evidence/G7/screenshots/gdn-confirmation-analysis-v1/confirmation-comparisons.csv)。
- 环境样张：[裁图](evidence/G7/screenshots/tilelang-cuda-z3-sample-v3.png)、[全屏原图](evidence/G7/screenshots/tilelang-cuda-z3-sample-v3-original.png)、[原始日志](evidence/G7/screenshots/tilelang-cuda-z3-sample-v3.log) 与 [窗口绑定](evidence/G7/screenshots/tilelang-cuda-z3-sample-v3.png.window-binding.json)。它只证明可见命令和终端环境。
- 方形 GEMM winner 资源证据：[终端图](evidence/G7/best-gemm-v1/tilelang-cuda-z3-best-gemm-v3.png)、[resource-usage.txt](evidence/G7/best-gemm-v1/resource-usage.txt)、[metadata.json](evidence/G7/best-gemm-v1/metadata.json)、[kernel.sass](../../evidence/G7/best-gemm-v1/kernel.sass) 与 [终端日志](evidence/G7/best-gemm-v1/tilelang-cuda-z3-best-gemm-v3.log)。其中 static shared 字段不能解释为完整动态共享内存总量。

早期未接受捕获仍留档：[第一次失败](evidence/G7/screenshot-failure-v2/capture-failure.json) 与 [第二次失败](evidence/G7/screenshot-failure-v3/capture-failure.json)。全屏原图和裁图通过 `package:` 声明收入文章 ZIP，正文 HTML 不重复嵌入这些图片。
