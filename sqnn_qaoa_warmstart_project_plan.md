# SQNN for Large-Scale QAOA Warm-Start 项目计划书

> 文件定位：这是本项目的主计划书和实验记录文件。完整路线、可行/不可行版本、实验日志、指标结果都优先写在这里。`soft_qnn_qaoa_warmstart_summary.md` 是短摘要版，用来快速了解项目当前状态。

## 1. 项目目标

本项目的核心目标是：

\[
\boxed{
\text{用 Soft Quantum Neural Network, SQNN, 为大型 QUBO/QAOA 问题生成高质量 warm-start}
}
\]

我们不把 SQNN 当作最终求解器，而是把它作为一个可训练的先验生成器。SQNN 输出每个二值变量的初始偏置：

\[
p_i = P(x_i = 1),
\]

然后用这些偏置初始化 QAOA、模拟退火、局部搜索或混合求解流程。

项目最终路线是：

\[
\text{Large QUBO}
\rightarrow
\text{sparse graph modeling}
\rightarrow
\text{message-passing SQNN}
\rightarrow
\text{warm-start probabilities}
\rightarrow
\text{QAOA / local search}
\]

## 2. 什么是 QUBO

QUBO 是 Quadratic Unconstrained Binary Optimization 的缩写，即二次无约束二值优化问题。

标准形式为：

\[
\min_{x\in\{0,1\}^n}
E(x)
=
\sum_i Q_{ii}x_i
+
\sum_{i<j}Q_{ij}x_ix_j.
\]

其中：

1. \(x_i\in\{0,1\}\) 是二值决策变量；
2. \(Q_{ii}\) 是变量 \(x_i\) 的线性权重；
3. \(Q_{ij}\) 是变量 \(x_i\) 和 \(x_j\) 之间的二次相互作用；
4. 目标是找到使能量 \(E(x)\) 最小的二值向量。

QUBO 的特点是形式非常统一。很多带约束的组合优化问题都可以通过惩罚项转成 QUBO。

例如，如果希望满足约束：

\[
g(x)=0,
\]

可以加入 penalty：

\[
\lambda g(x)^2.
\]

最终目标变成：

\[
E(x)
=
\text{original objective}
+
\lambda\cdot\text{constraint penalty}.
\]

这样就可以把有约束问题转成无约束二值优化问题。

## 3. QUBO 的现实应用

QUBO 是量子优化、模拟退火和组合优化里非常常见的统一建模形式。典型应用包括：

1. **MaxCut / 图划分**

   给定图 \(G=(V,E)\)，把顶点分成两组，使跨组边权最大。MaxCut 可以自然写成 QUBO，也可以直接映射到 Ising Hamiltonian。

2. **调度问题**

   包括生产排程、任务分配、云计算资源调度、作业车间调度等。变量通常表示“任务 \(i\) 是否在时间 \(t\) 或机器 \(m\) 上执行”。

3. **路径和路由问题**

   例如旅行商问题、车辆路径规划、网络路由、物流配送。变量可以表示“是否选择某条边”或“城市 \(i\) 是否处于路径位置 \(t\)”。

4. **投资组合优化**

   变量表示是否选择某个资产。目标函数结合收益、风险和预算约束，可以写成 QUBO。

5. **图染色 / 分配问题**

   图染色可以用 one-hot 变量 \(x_{v,c}\) 表示“顶点 \(v\) 是否选择颜色 \(c\)”。相邻同色和一个顶点多色都作为 penalty。

6. **机器学习中的特征选择**

   变量表示是否选择某个特征。目标可以综合预测性能、特征数量和特征之间的冗余。

7. **芯片设计和布局布线**

   包括 VLSI placement、routing、逻辑映射。变量表示模块位置、连线选择或局部布线状态。

8. **能源系统优化**

   包括电网调度、机组组合、储能控制。变量表示设备开关状态或离散运行模式。

QUBO 的优势是统一、简单、适合量子算法和启发式算法；缺点是现实问题转成 QUBO 后变量数可能非常大，惩罚系数选择也会显著影响求解难度。

### 3.1 哪些 QUBO 值得优先做

本项目不是所有 QUBO 都同等适合。我们当前技术路线最适合的是：

\[
\text{大规模、稀疏、图结构明显、局部修复有效、固定后 residual 会明显变小的 QUBO}
\]

更具体地说，优先级如下。

| 优先级 | QUBO 类型 | 是否适合当前路线 | 原因 |
|---|---|---|---|
| A | 稀疏 MaxCut / 图划分 / 网络切分 | 很适合 | QUBO 建模自然，图结构直接对应 SQNN message passing，近似比清晰，适合上百/上千变量展示 |
| A | 稀疏 parity / XOR / Ising 约束 QUBO | 很适合 | 有相同/不同两类二次关系，边权可正可负，比 MaxCut 更通用；planted parity 可提供严格 known optimum |
| A- | 稀疏冲突图调度 | 适合 | 任务-资源-时间冲突可以形成稀疏二次 penalty；local repair 能解释为消除冲突 |
| A- | 稀疏资源分配 / 稀疏设施选择 | 适合 | 变量有局部依赖，适合 SQNN 利用边特征给 warm-start |
| B | 稀疏约束满足问题转 QUBO | 有潜力 | 如果 penalty 图稀疏，repair+fix 可能有效；但 penalty 权重要谨慎 |
| B | 投资组合选择 | 有条件适合 | 若协方差矩阵稀疏或经过截断才适合；密集风险项会导致图过密 |
| C | TSP / VRP 的朴素 one-hot QUBO | 暂不优先 | one-hot 和路径约束容易产生稠密 penalty，变量数和边数爆炸 |
| C | 密集 QUBO / 全连接 QUBO | 不适合当前阶段 | SQNN 消息复杂度高，residual 不容易碎成小 active core，QAOA 仍然跑不动 |
| C | penalty 系数极端病态的 QUBO | 高风险 | 模型容易只学 penalty，目标函数被淹没；训练和 repair 都会不稳定 |

当前建议的实用落点是：

1. **稀疏图划分 / 网络分割**

   这是最稳的下一阶段应用方向。原因是图结构天然、QUBO 稀疏、近似比容易定义，并且工程上常见：社区发现、通信网络切分、任务依赖图切分、芯片/计算图 partition 都可以抽象成类似问题。

2. **稀疏 parity / Ising 约束优化**

   这是比 MaxCut 更通用的验证方向。我们已经实现了 `planted_parity` benchmark：它同时包含“变量应相同”和“变量应不同”两类边约束，QUBO 边权有正有负，而且 known optimum 可严格计算。

3. **稀疏调度 / 冲突消解**

   例如任务 \(i\) 与任务 \(j\) 不能同时选择、某资源不能被多个任务同时占用、某些任务组合有协同收益或冲突成本。这类问题如果冲突图稀疏，就非常适合当前的 SQNN warm-start + local repair。

4. **稀疏资源选择 / 设施选址**

   变量表示是否选择某个站点、设备、路线或资源；二次项表示局部冲突、覆盖重叠、协同收益。只要相互作用不是全连接，就适合当前路线。

### 3.2 当前方法对 QUBO 和图形状的要求

当前表现最好的路线是：

\[
\text{hybrid SQNN}
\rightarrow
\text{sample / round}
\rightarrow
\text{local repair}
\rightarrow
\text{repair-calibrated probabilities}
\rightarrow
\text{confidence fixing}
\rightarrow
\text{isolated-variable exact fixing}
\rightarrow
\text{component-wise residual QAOA}
\]

它对 QUBO 有几个实际要求：

1. **稀疏性**

   边数最好近似 \(O(n)\)，例如平均度 4 到 12。当前主要实验在 1000 变量、约 4000 边附近，效果较好。

2. **局部结构**

   SQNN 通过图消息传递利用局部邻域信息。如果 QUBO 是完全随机密集矩阵，局部消息优势会下降。

3. **repair 有用**

   当前 raw SQNN sign 有时会错，真正稳的是 `SQNN confidence + local repair direction`。因此问题需要允许局部翻转搜索快速降低能量。

4. **固定后 residual 要变小**

   大规模 QAOA 的关键不是原图多大，而是 high-confidence fixing 后剩下的 active residual 多大。最理想情况是 residual 只剩少量变量，或者按连通分量拆成小 core。

5. **penalty 权重不要极端失衡**

   如果 constraint penalty 比 objective 大几个数量级，SQNN 训练容易只学会满足 penalty，而不优化原始目标。后续做真实应用时需要做 penalty normalization 或分阶段训练。

对图形状的要求可以总结为：

\[
\text{原图可以大，但 residual graph 必须小、稀疏、可分解}
\]

最喜欢的图形状：

1. 原始图稀疏，平均度低到中等；
2. 局部社区结构明显；
3. 高置信变量固定后，剩余边大量消失；
4. active residual 能拆成小连通分量；
5. 最大连通分量小于 3060 可模拟上限，当前估计约 29 qubits。

不喜欢的图形状：

1. 高度稠密或近似全连接；
2. 固定后仍剩一个很大的 expander-like core；
3. one-hot penalty 导致大量 clique；
4. 权重尺度差异过大；
5. 二次项太多，导致消息传递和 QAOA 都被边数拖垮。

## 4. QUBO 如何建模

QUBO 建模通常分为五步。

### 4.1 定义二值变量

先把实际问题中的决策转成二值变量。

例如：

\[
x_i =
\begin{cases}
1, & \text{选择第 }i\text{ 个方案},\\
0, & \text{不选择第 }i\text{ 个方案}.
\end{cases}
\]

对于图染色：

\[
x_{v,c} =
\begin{cases}
1, & \text{顶点 }v\text{ 使用颜色 }c,\\
0, & \text{否则}.
\end{cases}
\]

### 4.2 写出原始目标函数

例如 MaxCut 可以写为：

\[
\max
\sum_{(i,j)\in E}
w_{ij}
\left(
x_i + x_j - 2x_ix_j
\right).
\]

如果统一为最小化，可以取负号。

### 4.3 把约束变成惩罚项

例如 one-hot 约束：

\[
\sum_c x_{v,c}=1
\]

可以加入：

\[
\lambda
\left(
\sum_c x_{v,c}-1
\right)^2.
\]

相邻顶点不能同色：

\[
x_{u,c}x_{v,c}=0
\]

可以加入：

\[
\lambda
\sum_{(u,v)\in E}
\sum_c
x_{u,c}x_{v,c}.
\]

### 4.4 合并成 QUBO 系数

最后把所有线性项和二次项合并成：

\[
E(x)
=
\sum_i q_i x_i
+
\sum_{i<j} q_{ij}x_ix_j.
\]

在代码里，大型 QUBO 不应使用 dense \(n\times n\) 矩阵，而应该使用稀疏形式：

```text
linear:      [n]
edge_index:  [2, E]
edge_weight: [E]
```

其中 \(E\) 是非零二次项数量。

### 4.5 归一化和 penalty tuning

大型 QUBO 很容易出现不同项尺度差距过大的问题。需要处理：

1. 线性项和二次项归一化；
2. penalty 系数不能太小，否则约束不满足；
3. penalty 系数不能太大，否则优化器只关心约束，忽略原始目标；
4. 对大型问题，最好记录每类 penalty 的贡献，便于调参。

## 5. 什么是 QAOA

QAOA 是 Quantum Approximate Optimization Algorithm，即量子近似优化算法。它是一种变分量子算法，专门用于组合优化问题。

QAOA 的基本思想是：

1. 把优化目标写成 cost Hamiltonian \(H_C\)；
2. 选择一个 mixer Hamiltonian \(H_M\)；
3. 从一个初始态 \(|\psi_0\rangle\) 开始；
4. 交替作用 cost unitary 和 mixer unitary；
5. 通过经典优化器调整参数 \(\gamma,\beta\)，使期望能量最小。

标准 QAOA 状态为：

\[
|\psi(\gamma,\beta)\rangle
=
\prod_{\ell=1}^{p}
e^{-i\beta_\ell H_M}
e^{-i\gamma_\ell H_C}
|\psi_0\rangle.
\]

其中 \(p\) 是 QAOA 深度。

对于 QUBO，可以把二值变量 \(x_i\in\{0,1\}\) 映射为 Pauli-Z：

\[
x_i = \frac{1-Z_i}{2}.
\]

于是 QUBO 目标可以变成：

\[
H_C
=
\sum_i h_i Z_i
+
\sum_{i<j} J_{ij}Z_iZ_j
+
\text{constant}.
\]

这使得 QUBO 可以直接作为 QAOA 的 cost Hamiltonian。

## 6. 大型 QUBO 在 QAOA 中的建模难点

大型 QUBO 进入 QAOA 后会遇到几个关键困难。

### 6.1 变量数大

每个 QUBO 变量通常对应一个 qubit。变量数 \(n\) 大时，所需 qubit 数也大。

\[
n\text{ variables}
\rightarrow
n\text{ qubits}.
\]

这会带来硬件规模和模拟规模的问题。

### 6.2 二次项多

每个非零 \(Q_{ij}\) 对应一个 \(Z_iZ_j\) 相互作用。如果图很密，QAOA 每层需要很多 two-qubit gates。

\[
|E_Q|\text{ nonzero quadratic terms}
\rightarrow
|E_Q|\text{ pair interactions}.
\]

因此大型 QUBO 必须重点关注 sparsity。

### 6.3 硬件连通性限制

如果 QUBO 图中的边和硬件 qubit connectivity 不匹配，需要 SWAP 或 routing，电路深度会增加。

### 6.4 参数优化困难

标准 QAOA 通常从 uniform superposition 开始：

\[
|+\rangle^{\otimes n}.
\]

对于大型问题，这个初态没有利用问题结构，经典优化器很容易需要大量迭代，甚至陷入较差区域。

这正是 warm-start 的动机。

## 7. 什么是 SQNN

SQNN 是 Soft Quantum Neural Network。它的核心思想是把量子神经元中的测量过程软化，使其保持可微分。

普通量子测量会产生硬采样：

\[
0\text{ or }1.
\]

SQNN 使用测量概率或期望值作为连续信号：

\[
p = P(1)
\quad\text{or}\quad
\langle Z\rangle.
\]

这样网络可以用梯度下降训练。

本项目中的 SQNN 有几个重要组件：

1. **soft quantum neuron**

   用 soft measurement 控制后续旋转，保持可微。

2. **multi-basis readout**

   不只读 \(Z\) 基，也读 \(X,Y,Z\) 三个方向：

   \[
   [P_X,P_Y,P_Z].
   \]

   这可以作为三维 Bloch 特征。

3. **edge-conditioned message passing**

   对 QUBO 图上的每条边，根据边特征生成旋转参数：

   \[
   U_{ij}=U_\theta(e_{ij}).
   \]

4. **warm-start readout**

   最终输出：

   \[
   p_i=P(x_i=1).
   \]

SQNN 在这里不是求解器，而是学习从 QUBO 图到变量偏置的映射。

## 8. 大型 QUBO 如何在 SQNN 中建模

大型 QUBO 在 SQNN 中应该建成稀疏图，而不是 dense matrix。

### 8.1 节点表示

每个变量 \(x_i\) 是一个 node。

节点特征可以包括：

1. 线性系数 \(Q_{ii}\)；
2. \(|Q_{ii}|\)；
3. incident edge weight sum；
4. incident absolute edge weight sum；
5. degree。

也就是：

\[
f_i
=
[
Q_{ii},
|Q_{ii}|,
\sum_j Q_{ij},
\sum_j |Q_{ij}|,
d_i
].
\]

### 8.2 边表示

每个非零二次项 \(Q_{ij}\) 是一条 interaction edge。

为了适配有向 SQNN message passing，可以把无向边拆成：

\[
i\rightarrow j,
\quad
j\rightarrow i.
\]

边特征可以包括：

\[
e_{ij}
=
[
Q_{ij},
|Q_{ij}|,
\operatorname{sign}(Q_{ij}),
Q_{ii},
Q_{jj},
d_i,
d_j
].
\]

### 8.3 Edge-conditioned rotation

用共享网络把边特征映射成量子旋转参数：

\[
(\phi_{ij},\theta_{ij},\omega_{ij})
=
g_\theta(e_{ij}).
\]

然后用它作用在源节点的 Bloch 特征上，形成 message：

\[
m_{i\rightarrow j}
=
R(\phi_{ij},\theta_{ij},\omega_{ij})h_i.
\]

### 8.4 聚合和更新

对每个节点聚合邻居消息：

\[
\bar m_j
=
\operatorname{Agg}
\left(
\{m_{i\rightarrow j}: i\in N(j)\}
\right).
\]

然后用 SQNN neuron 更新节点状态：

\[
h_j^{(t+1)}
=
\operatorname{SQNN}
\left(
h_j^{(t)},\bar m_j^{(t)}
\right).
\]

重复多轮后读出：

\[
p_i = \operatorname{Readout}(h_i^{(T)}).
\]

### 8.5 大型问题的关键原则

1. 使用 sparse edge list，不使用 dense \(Q\)；
2. 所有边共享参数 \(U_\theta(e_{ij})\)，避免参数量随边数增长；
3. message passing 复杂度为 \(O(E)\)；
4. 支持 mini-batch 子图训练；
5. 输出 \(p_i\) 后可以只选择高置信变量固定，低置信变量交给 QAOA 或 local search。

## 9. 什么是 Warm Start

Warm start 指的是在求解优化问题前，先给求解器一个较好的初始状态或初始分布，而不是从完全随机或均匀状态开始。

对于 QUBO，warm start 可以是：

1. 一个初始二值解 \(x^{(0)}\)；
2. 每个变量取 1 的概率 \(p_i\)；
3. QAOA 初态角度 \(\theta_i\)；
4. 局部搜索中的变量翻转优先级；
5. 可固定变量和待搜索变量的划分。

在本项目中，SQNN 输出：

\[
p_i=P(x_i=1).
\]

它可以转成 QAOA 初态：

\[
|\psi_0\rangle
=
\bigotimes_i
\left(
\sqrt{1-p_i}|0\rangle
+
\sqrt{p_i}|1\rangle
\right).
\]

对应 \(R_y\) 角度：

\[
\theta_i
=
2\arcsin\sqrt{p_i}.
\]

这比 uniform initialization 更有问题结构信息。

## 10. 为什么 SQNN 适合做 QAOA Warm Start

SQNN 适合做 warm start 的原因有四点。

### 10.1 输出是软概率

QAOA warm-start 需要连续概率 \(p_i\)，而 SQNN 天然输出 soft measurement probability。

### 10.2 可以表达局部相互作用

QUBO 的核心是二次相互作用 \(Q_{ij}x_ix_j\)。message-passing SQNN 可以沿 QUBO 图传播局部信息。

### 10.3 多基读出提供更丰富的状态

三基 Bloch 特征：

\[
[P_X,P_Y,P_Z]
\]

可以保留比单一 \(P_Z\) 更丰富的局部结构信息。

### 10.4 可以服务大型 QAOA

对于大型 QUBO，完整 QAOA 很昂贵。SQNN 可以先给出偏置，帮助：

1. 减少 QAOA 参数搜索难度；
2. 提供更好的初态；
3. 选择变量子集；
4. 指导 classical pre/post-processing。

## 11. 项目计划

项目分为六个阶段。

### 阶段一：QUBO 建模层

目标：建立大型 QUBO 的稀疏表示。

任务：

1. 实现 `QUBOProblem`；
2. 支持 `linear + edge_index + edge_weight`；
3. 支持从 dense matrix 转 sparse QUBO；
4. 支持精确能量计算；
5. 支持期望能量计算；
6. 支持节点特征和边特征生成。

产出：

1. `quantum/warmstart/qubo.py`；
2. QUBO energy / expected energy 测试；
3. 小型 QUBO 示例。

### 阶段二：SQNN Warm-Start 基线模型

目标：实现第一版 QUBO-to-probability 模型。

任务：

1. 每个变量建一个 node state；
2. 每条 QUBO 边建双向 directed edge；
3. 用 edge feature 生成 rotation 参数；
4. 做多轮 message passing；
5. 用 SQNN readout 输出 \(p_i\)。

产出：

1. `quantum/warmstart/qubo_sqnn.py`；
2. `QUBOWarmStartSQNN`；
3. 支持不同规模 QUBO 的 forward。

### 阶段三：Warm-Start Loss 和采样

目标：让 SQNN 可以训练。

任务：

1. 实现 expected QUBO energy loss：

   \[
   \mathcal L_E
   =
   \sum_i Q_{ii}p_i
   +
   \sum_{i<j}Q_{ij}p_ip_j.
   \]

2. 加 entropy regularization；
3. 支持从 \(p_i\) 采样初始解；
4. 支持选择 best-of-N sample；
5. 可选加入 classical solver 产生的 BCE supervision。

产出：

1. `losses.py`；
2. `sampling.py`；
3. training smoke test。

### 阶段四：QAOA 接口

目标：把 SQNN 输出接入 QAOA。

任务：

1. 实现 \(p_i\rightarrow\theta_i\)：

   \[
   \theta_i = 2\arcsin\sqrt{p_i}.
   \]

2. 输出 QAOA 初态参数；
3. 支持 uniform QAOA 和 SQNN warm-start QAOA 对比；
4. 对小型 QUBO 做 exact simulation；
5. 比较 energy convergence。

产出：

1. `qaoa_init.py`；
2. QAOA warm-start demo；
3. 对比实验：uniform vs SQNN warm-start。

### 阶段五：大型 QUBO 扩展

目标：让模型面向大型问题。

任务：

1. 使用 sparse edge list，避免 dense \(Q\)；
2. 支持大图 mini-batch 或 subgraph training；
3. 支持变量置信度：

   \[
   c_i = |p_i - 0.5|.
   \]

4. 高置信变量可固定；
5. 低置信变量交给 QAOA 子问题；
6. 建立 large-QUBO benchmark。

产出：

1. 大型 QUBO 数据生成器；
2. subproblem extraction；
3. SQNN-guided QAOA pipeline。

### 阶段六：应用扩展

目标：从 QUBO 扩展到具体问题。

优先顺序：

1. MaxCut；
2. weighted MaxCut；
3. graph partitioning；
4. graph coloring；
5. scheduling；
6. routing。

每个应用需要：

1. 问题到 QUBO 的编码；
2. penalty tuning；
3. SQNN warm-start；
4. QAOA/local search 后处理；
5. 与 random、greedy、classical heuristic 对比。

## 12. 实验设计

### 12.1 Baselines

需要比较：

1. random initialization；
2. greedy initialization；
3. simulated annealing only；
4. uniform QAOA；
5. classical warm-start QAOA；
6. SQNN warm-start QAOA。

### 12.2 Metrics

核心指标：

1. initial energy；
2. final energy；
3. approximation ratio；
4. iterations to target；
5. QAOA parameter optimization steps；
6. valid solution rate；
7. runtime；
8. memory usage；
9. large-scale transfer ability。

### 12.3 Ablation Study

需要消融：

1. 单基读出 vs 三基读出；
2. feedforward SQNN vs message-passing SQNN；
3. 无边特征 vs edge-conditioned rotation；
4. 不加 entropy vs 加 entropy；
5. 全变量 QAOA vs SQNN 选子问题 QAOA；
6. 不同 message passing rounds；
7. MLP/sigmoid feature map vs quantum-data angle encoding；
8. per-node output logits vs per-node quantum readout rotations。

## 13. 近期最小可行目标

第一阶段先不追求完整大型 QAOA。最小可行实验是：

1. 随机生成稀疏 QUBO；
2. 用 `QUBOProblem` 建模；
3. 用 `QUBOWarmStartSQNN` 输出 \(p_i\)；
4. 用 expected energy loss 训练；
5. 从 \(p_i\) 采样初始解；
6. 和 random sampling 比较 best-of-N energy；
7. 把 \(p_i\) 转成 QAOA 初态角度；
8. 在小规模问题上比较 uniform QAOA 和 warm-start QAOA。

这一阶段的目标不是证明最终量子优势，而是证明：

\[
\boxed{
\text{SQNN 可以学习 QUBO 图结构，并输出比随机更好的初始分布}
}
\]

## 14. 项目风险

### 14.1 Penalty 选择困难

现实问题转 QUBO 后，penalty 系数会影响优化难度。需要设计自动归一化和 penalty sweep。

### 14.2 大型 QAOA 本身受限

大规模 QAOA 受 qubit 数、gate depth 和硬件连接限制。项目中要明确区分：

1. SQNN 预处理大型 QUBO；
2. QAOA 处理完整问题；
3. QAOA 只处理 SQNN 筛选出的子问题。

第三条可能更现实。

### 14.3 SQNN 可能过早坍缩

如果 \(p_i\) 太快接近 0 或 1，模型会失去探索能力。需要 entropy regularization 和 temperature schedule。

### 14.4 泛化问题

如果模型只在固定规模 QUBO 上训练，可能不能迁移到更大图。必须使用参数共享和稀疏 message passing。

## 15. 最终预期成果

项目最终希望形成：

1. 一个大型 QUBO 稀疏建模模块；
2. 一个 dependency-aware SQNN warm-start 模型；
3. 一个 QAOA warm-start 初始化接口；
4. 一套 QUBO / MaxCut / graph coloring benchmark；
5. 一组实验证明 SQNN warm-start 能改善初始能量、收敛速度或最终解质量；
6. 一个可以继续扩展到 scheduling、routing、VLSI 等现实问题的框架。

最终项目标题可以定为：

\[
\boxed{
\text{Dependency-Aware Multi-Basis SQNN for Large-Scale QAOA Warm-Start}
}
\]

中文标题：

\[
\boxed{
\text{面向大型 QAOA 预热的依赖感知多基读出软量子神经网络}
}

## 实验日志追加

- 时间戳: `20260611_191221`
- benchmark: `planted_bipartite_maxcut_n64_d4.0`
- model: `directed`
- 变量数: `64`
- 边数: `133`
- device: `cuda`
- 训练秒数: `2.13`
- no-warm-start random best ratio: `0.628339`
- no-warm-start random+local-search ratio: `0.975495`
- SQNN sampled ratio: `0.643693`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `133`
- QAOA p=2 gates: `266`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 阶段总结：SQNN-QUBO warm-start 当前有效方案

更新时间：2026-06-11。

当前主线已经从“直接用 SQNN 输出作为最终解”调整为：

\[
\text{Large sparse QUBO}
\rightarrow
\text{hybrid SQNN probabilities}
\rightarrow
\text{sampling / rounding}
\rightarrow
\text{greedy local repair}
\rightarrow
\text{confidence-based fixing}
\rightarrow
\text{small residual QUBO for QAOA}
\]

这个路线的关键判断是：大型 QAOA 本体不能直接吃下上百/上千变量，但 SQNN 可以先给出变量偏置，再用局部修复消掉错误，最后把高置信变量固定掉，只把很小的 residual QUBO 交给 QAOA。

### 当前实现文件

1. `quantum/warmstart/qubo.py`：稀疏 QUBO 表示、能量、期望能量、无向边转有向消息边、固定变量后的 residual QUBO。
2. `quantum/warmstart/qubo_sqnn.py`：QUBO 版本 SQNN 模型，包括 node-only、directed、symmetric、instance、hybrid、quantum-data、mean-field baseline。
3. `quantum/warmstart/heuristics.py`：随机 baseline、贪心 rounding、局部搜索、模拟退火。
4. `quantum/warmstart/qaoa_limits.py`：QAOA 层数、双量子门数、3060 显存下 full-state 上限估算。
5. `scripts/run_qubo_warmstart.py`：训练、baseline、近似比、残余 QUBO、QAOA 资源评估主脚本。
6. `scripts/summarize_warmstart_runs.py`：批量汇总实验结果为 CSV/Markdown。
7. `scripts/plot_warmstart_run.py`：单次实验效果图，展示近似比和 residual QUBO 变量数。

### 模型版本与可行性记录

| 版本 | 修改方向 | 结论 | 原因 |
|---|---|---|---|
| V0 node-only SQNN | 只用节点特征，不看 QUBO 边 | 不可行 | 大量节点在 MaxCut 中局部特征相似，不能表达二次耦合，也不能有效破对称 |
| V1 directed SQNN | 把无向 QUBO 边复制成双向消息边 | 单独不可行 | 虽然解决了 SQNN 有向消息与 QUBO 无向耦合的接口问题，但共享参数仍容易停在 \(p_i\approx 0.5\) |
| V2 symmetric SQNN | 强制无向边共享特征、双向一致 | 单独不可行 | 更尊重 QUBO 无向结构，但表达力和破对称能力仍不足 |
| V3 instance embedding SQNN | 给每个变量加 per-instance embedding | 弱可行/不稳定 | 能引入节点身份，但纯 SQNN readout 仍容易坍缩到 0.5 或过度平滑 |
| V4 mean-field baseline | 每个变量直接训练 Bernoulli logit | 强 baseline，但不是 SQNN | planted MaxCut 上极快到达 1.0，说明问题可被直接参数化解掉，但没有保留 SQNN 图消息机制 |
| V5 hybrid SQNN | per-node logit + SQNN 图消息修正 | 当前主线可行 | 能在 1000 变量上产生有用 warm-start，经过局部修复后可把 residual 压到小 QAOA 范围 |
| V6 repair-calibrated SQNN | 保留 SQNN confidence，使用 repair assignment 修正概率方向 | 当前最稳 warm-start 表达 | raw SQNN sign 可能过度自信，repair-calibrated 概率能显著提升采样质量，同时仍可转成 QAOA 初态角 |
| V7 active-residual QAOA | residual 中孤立变量按线性项精确固定，只把有相互作用的 active core 交给 QAOA | 可行且必要 | confidence fixing 后的 residual 往往变量数不小但边很少，孤立变量不需要 QAOA，消元后 25 变量 residual 可变成 8 qubit active QAOA |
| V8 component-wise residual QAOA | active core 按连通分量拆开，每个分量独立 QAOA | 可行，需标注参数语义 | 最大 statevector 规模由 active 总变量数变成最大连通分量变量数；等价于分量独立参数的 QAOA 上层流程 |
| V9 quantum-data SQNN | 去掉 MLP 节点初始化、MLP edge encoder、sigmoid 初态、per-node output logits、非线性 readout；改用 3 维 QUBO 节点量、角度编码、Bloch 旋转、Z 基测量 | 可行但当前 raw warm-start 偏弱 | 结构更接近“量子数据表示”，一个 SQNN 节点只承载三基信息；但去掉 output logits 后破对称能力下降，256 变量 planted parity 上 raw sample+repair 仍能到高质量解，但 confidence fixing 还不能有效压小 residual |
| V10 synchronous local-field SQNN | 一个变量一个三基 SQNN 神经元；\(P_Z\) 读出 \(p_i=P(x_i=1)\)，\(P_X/P_Y\) 作为隐藏相干/相位记忆；每轮用旧状态同步计算 \(F_i=a_i+\sum_j b_{ij}p_j\)，再同步更新所有节点 | 已实现初版，结构干净但效果弱于 hybrid | 避免 GNN 式多维节点特征和人为节点顺序；无向 QUBO 边只作为双向同步 soft influence，不引入有向优化含义；复杂度保持 \(O(n+|E|)\)，区别于全局态 QAOA；当前参数较少，MaxCut 上破对称能力仍不足 |

### V9 quantum-data SQNN 结构记录

V9 的目标是回应“不要用 MLP/sigmoid 把经典特征转成三基概率，而要用量子数据表示节点信息”。实现名为 `QUBOQuantumDataWarmStartSQNN`，命令行为 `--model quantum_data`。

替换关系：

1. 节点初始化：从 5 维 `node_features -> MLP -> sigmoid -> [P_X,P_Y,P_Z]` 改为 3 维 `quantum_node_features -> angle encoding -> Bloch rotations -> [P_X,P_Y,P_Z]`。
2. 边编码：从 `edge_features -> MLP -> [phi,theta,omega]` 改为 `edge_features @ trainable_angle_table -> [phi,theta,omega]`。
3. per-node output logits：删除，改为每个变量一组 `node_readout_angles`，通过节点级读出旋转破对称。
4. 最终读出：从 `QuantumNeuronLayer + sigmoid(logit composition)` 改为 `Bloch state -> readout rotation -> Z-basis measurement`，即 \(p_i=P_Z(1)\)。

保留部分：

1. 无向 QUBO 边仍复制为双向消息边；
2. 消息仍是邻居 Bloch 向量经边条件旋转后聚合；
3. update 仍用 `MultiBasisQuantumNeuronLayer(input_dim=6, output_dim=1)`，它不是 MLP，而是 SQNN 的三基量子神经元读出。

三维节点量：

1. \(a_i\)：QUBO 一次项，表示变量自身偏置；
2. \(\sum_j b_{ij}\)：带符号邻接耦合，表示局部相互作用方向；
3. \(\sum_j |b_{ij}|\)：总耦合强度，表示该变量受邻域约束的强弱。

删除的节点量：

1. \(|a_i|\)：可由 \(a_i\) 推出，作为 MLP shortcut 有用，但不适合严格三基节点编码；
2. degree：已由图结构和消息聚合体现，不再塞进单个 SQNN 神经元状态。

当前判断：

1. 可行：模型能前向、反向、GPU 训练，输出概率合法，smoke test 已加入。
2. 不足：由于删除 per-node output logits，raw SQNN 破对称能力变弱；256 变量 planted parity quick test 中 raw sample ratio 约 0.588，sample+local-search ratio 约 0.936，repair-calibrated sample+local-search 可到 1.0，但 residual 没有被有效压小。
3. 下一步：需要比较 `quantum_data`、`hybrid`、`mean_field` 的同 seed 消融；同时尝试更强的量子式 per-node 参数结构，例如多层局部 readout rotations、Ising \(h_i,J_{ij}\) 专用角度编码、以及 Bloch 球内 mixed-state 初始化。

实现修正：

1. 训练脚本原先按退火后的 `loss = normalized_energy - entropy_weight * entropy` 保存 best checkpoint。由于 entropy weight 会随 epoch 变化，不同 epoch 的 loss 不严格可比，短实验中出现 best epoch 被固定在 0 的问题。
2. 现在改为按 `normalized_energy` 保存 best checkpoint，同时继续记录 loss 和 entropy。这个修正对所有模型都更合理，尤其对 `quantum_data` 这种初始 entropy 较高、训练早期变化慢的结构更重要。

### V10 synchronous local-field SQNN 设计记录

V10 是当前更贴近原始 SQNN 结构的 QUBO 编码版本。它不把 QUBO 节点做成 GNN 特征向量，而是把每个 QUBO 变量对应为一个三基 SQNN 神经元。

QUBO：

\[
E(x)=
\sum_i a_i x_i
+
\sum_{(i,j)} b_{ij}x_ix_j.
\]

状态定义：

1. 每个变量 \(x_i\) 对应一个 SQNN 神经元；
2. 神经元状态用 Bloch/三基读出表示：

   \[
   h_i^t = [P_X^t(i), P_Y^t(i), P_Z^t(i)].
   \]

3. 最终决策只使用 Z 基：

   \[
   p_i^t=P(x_i=1)=P_Z^t(i).
   \]

4. \(P_X,P_Y\) 不再编码 degree、\(|a_i|\)、\(\sum |b_{ij}|\) 等额外 GNN 特征，而是解释为隐藏相干状态、相位记忆和可动性。

无向边处理：

1. QUBO 的 \(b_{ij}x_ix_j\) 是无向相互作用；
2. SQNN 计算消息时可以写成两条方向：

   \[
   j\rightarrow i:\quad b_{ij}p_j^t,
   \]

   \[
   i\rightarrow j:\quad b_{ij}p_i^t.
   \]

3. 这里的方向只表示同步计算流，不表示有向 QUBO，也不引入 src/dst 特征；
4. 两个方向共享同一个 \(b_{ij}\)。

同步更新规则：

\[
p_i^t = P_Z(h_i^t),
\]

\[
G_i^t = \sum_j b_{ij}p_j^t,
\]

\[
F_i^t = a_i + G_i^t.
\]

然后对所有节点同时更新：

\[
h_i^{t+1}
=
\operatorname{SQNNUpdate}
\left(
h_i^t,
F_i^t;\theta_t
\right).
\]

关键要求：

1. 所有 \(p_i^t\) 必须从旧状态读出；
2. 所有 \(F_i^t\) 必须同时计算；
3. 所有 \(h_i^{t+1}\) 必须同时写回；
4. 禁止按节点顺序原地更新，否则会引入人为顺序偏置。

一种具体 Bloch 更新可以写成：

\[
r_i^t =
\left[
1-2P_X^t(i),
1-2P_Y^t(i),
1-2P_Z^t(i)
\right],
\]

\[
\tilde r_i^t =
R_Z(\gamma_t F_i^t) r_i^t,
\]

\[
r_i^{t+1}
=
R_Y(\beta_t)\tilde r_i^t.
\]

其中：

1. \(R_Z(\gamma_tF_i^t)\)：把 QUBO 局部场写入 \(X/Y\) 相位；
2. \(R_Y(\beta_t)\)：把 \(X/Y\) 相干信息转成 Z 基概率变化；
3. \(P_Z\)：仍然是最终 bit 概率读出。

和直接 QAOA 的区别：

1. QAOA 维护全局态 \(|\psi\rangle\in\mathbb C^{2^n}\)，V10 只维护 \(n\) 个三基 SQNN 节点状态；
2. QAOA 的 \(b_{ij}\) 是真实 two-qubit entangling phase，V10 的 \(b_{ij}\) 是 soft influence \(b_{ij}p_j^t\)；
3. QAOA 一层是全局 unitary，V10 是同步局部 SQNN 更新；
4. QAOA statevector 模拟复杂度指数级，V10 复杂度应为 \(O(n+|E|)\)。

诊断量：

1. bit confidence：

   \[
   C_i=|r_{z,i}|.
   \]

2. coherence / mobility：

   \[
   M_i=\sqrt{r_{x,i}^2+r_{y,i}^2}.
   \]

3. edge influence：

   \[
   G_i=\sum_j b_{ij}p_j.
   \]

4. edge dominance ratio：

   \[
   D_i=
   \frac{|G_i|}
   {|a_i|+|G_i|+\epsilon}.
   \]

解释：

1. \(C_i\) 高表示节点当前接近确定；
2. \(M_i\) 高表示节点仍有较强相干/可动性；
3. \(G_i\) 和 \(D_i\) 更适合衡量节点是否主要受边影响，而不是把 \(P_X/P_Y\) 误解释成边强度特征。

实现状态：

1. 已新增 `QUBOSynchronousLocalFieldSQNN`，命令行模型名为 `--model sync_local`；
2. 已接入 `quantum/warmstart/__init__.py` 和 `scripts/run_qubo_warmstart.py`；
3. 已加入 smoke test：检查输出概率合法、`monotone_accept=True` 时 energy trace 不上升、`monotone_accept=False` 时梯度可流向更新参数；
4. 节点数据只使用 \(a_i\)，边数据只使用 \(b_{ij}\)；
5. 训练 loss 仍使用 QUBO 期望能量：

   \[
   \mathcal L =
   \sum_i a_i p_i
   +
   \sum_{(i,j)}b_{ij}p_ip_j.
   \]

能量下降保证：

1. 同步更新本身不天然保证能量下降，因为多个节点同时改变时会相互干扰；
2. 当前实现提供 `monotone_accept=True`：每轮先提出完整同步更新，再计算更新前后的 QUBO 期望能量；
3. 若

   \[
   E[p^{t+1}_{proposal}] \le E[p^t],
   \]

   则接受整轮更新；否则保留旧状态；
4. 因此可以保证模型内部记录的连续期望能量 trace 单调不升；
5. 这个保证只针对 mean-field 期望能量 \(E[p]\)，不保证采样 bitstring、局部搜索后结果、近似比逐轮单调改善。

初步实验：

| run | benchmark | n | raw sampled ratio | sample+local-search ratio | repair-calibrated sample+local-search ratio | 结论 |
|---|---|---:|---:|---:|---:|---|
| `20260615_221318` | planted parity | 64 | 0.714460 | 0.811471 | 0.811471 | raw 高于随机，但高置信固定过早，repair 未救回 |
| `20260615_221342` | planted parity | 256 | 0.634481 | 0.990145 | 0.992586 | 明显优于 random+local-search 0.802523，连续能量 trace 单调下降 |
| `20260615_221406` | planted MaxCut | 256 | 0.539592 | 0.889109 | 0.819859 | 比 random+local-search 0.821803 好，但不如 hybrid；residual 未压小 |

`20260615_221342` 的训练后 energy trace：

\[
[-283.9748,\ -290.9700,\ -299.8148,\ -309.4821,\ -333.1303],
\]

四轮 proposal 均被接受，验证 monotone accept 在该实例上生效。

下一步：

1. 与 V5 hybrid、V9 quantum-data、mean-field baseline 做同 seed 对比；
2. 尝试增加不破坏结构的表达力，例如每层独立 \(\gamma_t,\beta_t\)、轻量 per-node phase memory、温度/阻尼参数；
3. 重点评估 raw sampled ratio、repair 后 ratio、confidence fixing 后 residual 规模。

### V10 规模 sweep：128/256/512/1024 变量

输出目录：

```text
outputs/sync_local_v10_evaluation
```

生成文件：

1. `sync_local_v10_model_notes.md`：完整模型说明、矩阵、局部场演化、迭代、warm-start 状态、QAOA 接口；
2. `metrics.csv` / `metrics.json`：16 次实验指标；
3. `ratio_vs_warmstart_rounds.png`：预热轮次/层数与近似比；
4. `ratio_vs_num_variables.png`：变量数与近似比；
5. `residual_active_vs_num_variables.png`：变量数与 residual active core；
6. `qaoa_gate_estimate_vs_variables.png`：完整 QAOA p=1/2/3 双量子门估计；
7. `training_time_vs_rounds.png`：训练时间与预热轮次。

实验设置：

1. benchmark：`planted_parity`，原因是有已知最优值，可计算严格 approximation ratio；
2. 变量数：128、256、512、1024；
3. SQNN 预热轮次/层数：1、2、4、8；
4. 每次训练 120 epochs；
5. 每次评估 256 samples，local search passes 为 200；
6. GPU：RTX 3060。

最佳结果按变量数汇总：

| n | best rounds | raw sampled ratio | sample+local-search ratio | repair-calibrated sample+local-search ratio | active residual t=0.25 |
|---:|---:|---:|---:|---:|---:|
| 128 | 4 | 0.642713 | 0.990983 | 0.990983 | 11 |
| 256 | 8 | 0.558490 | 0.989430 | 0.823025 | 254 |
| 512 | 1 | 0.573181 | 0.830260 | 0.856066 | 0 |
| 1024 | 8 | 0.672270 | 0.819265 | 0.864227 | 287 |

观察：

1. raw sampled ratio 随预热轮次通常有提升，尤其 1024 变量从 rounds=1 的 0.5585 到 rounds=8 的 0.6723；
2. sample+local-search 在 128/256 变量上可接近 0.99，但 512/1024 变量目前只到约 0.83/0.82；
3. 预热轮次增加会提高 raw ratio，但也可能让 high-confidence fixing 后 residual active 变大，说明“解质量”和“可压缩 residual”之间存在张力；
4. 当前 V10 作为结构干净的 SQNN warm-start 是可运行的，但大规模质量和 residual 压缩仍弱于 V5 hybrid 路线；
5. QAOA 兼容仍应走 residual/component-wise 路线，完整 512/1024 变量 QAOA 不现实。

补充门数图：

`qaoa_gate_estimate_vs_variables.png` 已更新为 full QAOA 与 SQNN warm-start residual QAOA 的同图对比，并保留副本：

```text
outputs/sync_local_v10_evaluation/qaoa_gate_estimate_full_vs_sqnn_residual.png
```

图中实线为完整 QAOA p=1/2/3 的双量子门估计，虚线为 SQNN 预热、local search、t=0.25 confidence fixing、isolated-variable fixing 后 residual active core 所需的 p=1/2/3 门数。

### V10 n=512 prefix rounds 1..100 sweep

输出目录：

```text
outputs/sync_local_v10_n512_rounds_1_100
```

注意：这次不是训练 100 个独立模型，而是训练一个 100-round `sync_local` 模型，然后读取 prefix round 1 到 100 的概率轨迹逐一评估。因此它反映“同一个长程模型在不同截断轮次上的 warm-start 效果”。

生成文件：

1. `metrics.csv` / `metrics.json`：100 个 prefix round 的指标；
2. `n512_ratio_vs_rounds_1_100.png`：近似比随轮次变化；
3. `n512_active_residual_vs_rounds_1_100.png`：active residual 变量数随轮次变化；
4. `n512_residual_qaoa_gates_vs_rounds_1_100.png`：residual QAOA p=1/2/3 门数随轮次变化；
5. `n512_confidence_vs_rounds_1_100.png`：平均置信度随轮次变化；
6. `n512_prefix_rounds_notes.md`：本次 sweep 的短说明。

关键点：

| 指标 | 最佳 round | 数值 | active residual t=0.25 | residual QAOA p=1 gates |
|---|---:|---:|---:|---:|
| raw sampled ratio | 97 | 0.717324 | 157 | 183 |
| sample+local-search ratio | 74 | 0.942139 | 378 | 689 |
| repair-calibrated sample+local-search ratio | 40 | 0.983421 | 499 | 977 |
| 最小 active residual | 95 | active=157 | 157 | 183 |

观察：

1. raw sampled ratio 在后期明显上升，round 97 达到约 0.717；
2. sample+local-search 的最佳点在 round 74，达到约 0.942；
3. repair-calibrated+local-search 的最佳点更早，在 round 40，达到约 0.983；
4. active residual 在 round 60 以前几乎不缩小，之后随平均置信度快速上升而下降，round 95 后约为 157 个 active variables；
5. residual QAOA 门数随 active residual edges 同步下降，p=1 从早期约 977 gates 降到后期约 183 gates；
6. 最优解质量和最小 residual 不在同一个 round，说明实际使用时需要按目标选择截断轮次：要高 ratio 选 40/74 附近，要小 residual 选 95+ 附近。

### 最新关键实验

Benchmark：`planted_bipartite_maxcut_n1000_d8.0`，变量数 1000，边数 4125，GPU 为 NVIDIA GeForce RTX 3060。

| 方法 | 近似比 | 局部搜索翻转数 | 说明 |
|---|---:|---:|---|
| random best | 0.520729 | - | 无 warm-start baseline |
| random + local search | 1.000000 | 约 100+ 到 600+，随设置变化 | 说明局部搜索很强，但从随机点出发成本不稳定 |
| hybrid SQNN sample | 0.864472 | - | SQNN 本身已经明显优于随机采样 |
| hybrid SQNN sample + local repair | 1.000000 | 129 | SQNN 给了更靠近好解的入口，修复成本小 |
| mean-field sample | 1.000000 | 0 | 强 baseline，但不是 SQNN 框架目标 |

QAOA 资源限制记录：

1. 对 1000 变量完整 QAOA，full-state statevector 在 3060 上不可行。
2. 1000 变量、4125 边时，估计双量子门数为：p=1 是 4125，p=2 是 8250，p=3 是 12375。
3. 按 12GB 3060、complex64、0.55 安全系数估算，full-state 可模拟上限约为 29 qubits。
4. 因此真实路线必须是 residual QAOA，而不是完整大规模 QAOA。

最新 hybrid SQNN 的 repair+fix residual 结果：

| 置信阈值 abs(p-0.5) | 固定比例 | 剩余变量 | 剩余边 | residual full-state QAOA |
|---:|---:|---:|---:|---|
| 0.49 | 0.863 | 137 | 165 | 不可行 |
| 0.45 | 0.935 | 65 | 40 | 不可行 |
| 0.40 | 0.953 | 47 | 23 | 不可行 |
| 0.35 | 0.963 | 37 | 12 | 不可行 |
| 0.30 | 0.967 | 33 | 11 | 不可行 |
| 0.25 | 0.975 | 25 | 4 | 可行 |
| 0.20 | 0.981 | 19 | 2 | 可行 |

这说明一个有意义的大规模展示已经成立：原始 1000 变量 QUBO 不能直接做 full-state QAOA，但 hybrid SQNN 生成 warm-start，经过局部修复后，以 0.25 阈值固定 975 个变量，剩 25 个变量和 4 条边，已经可以作为小规模 residual QAOA 输入。

### 当前可行路径

1. 大规模 QUBO 用稀疏图存储，复杂度主要随边数增长。
2. QUBO 无向边通过双向 directed message edges 进入 SQNN，保留无向能量，但允许 SQNN 做有向消息更新。
3. SQNN 输出 \(p_i=P(x_i=1)\)，转成 QAOA warm-start 的 \(R_y\) 初态角：

\[
\theta_i = 2\arcsin \sqrt{p_i}.
\]

4. 先用 SQNN 概率采样或 rounding 得到候选解，再做 greedy local repair。
5. 固定变量时不要直接相信原始概率的符号；应使用 repair 后的 assignment 作为 fixed values，用原始概率只作为 confidence mask。
6. 对 residual QUBO 做 QAOA，而不是对原始 1000 变量做 QAOA。

### 当前不可行或高风险路径

1. 直接完整 1000 变量 QAOA：不可行，statevector 显存指数爆炸，p>1 只会进一步增加优化和门数压力。
2. 纯 shared SQNN 直接求解：不可行，node-only、directed、symmetric 都出现破对称不足，概率停在 0.5 附近。
3. 原始高置信变量直接固定：高风险。早期实验出现过高置信但错误的情况，会把 residual QUBO 锁死。
4. 只看 sampled ratio 不看 local repair flips：不够。warm-start 的价值之一是减少后处理搜索成本，所以必须记录翻转数。
5. mean-field 作为最终方案：不符合项目目标。它是强 baseline 和诊断工具，但没有保留 SQNN 的图消息结构。

### 下一步计划

1. 加入多 seed 批量实验，验证 1000 变量 residual 压缩不是单个 seed 偶然结果。
2. 增加 residual QAOA 小模拟器，只在剩余变量 \(\le 29\) 时启动，用 p=1/2/3 比较 warm-start 初态与普通 \(|+\rangle\) 初态。
3. 做 confidence calibration：用 repair 前后是否翻转来校准 \(p_i\)，减少错误固定。
4. 对 random MaxCut 做 best-observed proxy 评估；没有已知最优解时，近似比需要标注为相对最好观测值。
5. 继续尝试更 SQNN-native 的破对称机制，例如节点噪声特征、Laplacian positional encoding、edge-sign aware phase update，目标是减少对 per-node logit 的依赖。

### residual QAOA 演示记录

新增脚本：`scripts/run_residual_qaoa_demo.py`。

实验源：`outputs/warmstart_runs/20260611_193747_planted_maxcut_hybrid_n1000/metrics.json`。

1. 阈值 0.20 时，hybrid SQNN + repair + fix 后剩余 19 个变量、2 条边，可以在 3060 上运行 statevector QAOA。
2. p=1/p=2 的 residual QAOA 期望近似比约 0.998，exact residual ratio 为 1.0。
3. 在这个 residual 上，普通 plus 初态略好于 SQNN 初态；原因是高置信变量已经被固定，剩下的自由变量本来就是低置信变量，SQNN 概率已经接近 0.5。
4. 阈值 0.25 时剩余 25 个变量、4 条边，显存估算可行，但当前朴素 autograd statevector mixer 在 40 步 p=1 下超时；这条路径标记为“理论可行、当前实现慢”，后续需要更高效的 statevector kernel 或直接调用成熟量子模拟库。

当前判断：SQNN warm-start 对 QAOA 的主要价值不是让 residual 上的初态仍然明显优于 plus，而是先把 1000 变量 QUBO 压缩到 19-25 变量 residual，使小层数 QAOA 有机会实际运行。

追加实验：`outputs/warmstart_runs/20260611_200854_planted_maxcut_hybrid_n1000/metrics.json`。

1. hybrid SQNN sampled ratio 为 0.860030，sampled + local repair ratio 为 1.0，local repair flips 为 137。
2. 阈值 0.25 时剩余 23 个变量、4 条边，已经满足 3060 full-state QAOA 估算上限。
3. 阈值 0.30 时剩余 29 个变量、8 条边，刚好等于当前估算上限。
4. 阈值 0.25 下，repair 后固定值与原始 SQNN rounding 不一致的高置信变量有 132 个，占固定变量约 13.51%。这证明“repair 后再固定”是必要步骤，不能直接按原始概率符号固定。
5. 阈值 0.20 时剩余 18 个变量、3 条边，p=1/p=2 residual QAOA 可运行。plus 初态 best expected ratio 分别约 0.997900/0.998106；SQNN 初态分别约 0.997642/0.997808；exact residual ratio 为 1.0。

校准判断：当前 SQNN 概率的 confidence 可以用来决定固定哪些变量，但 fixed values 应来自局部修复后的 assignment。下一步要研究的是 confidence calibration，而不是盲目提高 SQNN 概率尖锐度。

### sweep 工程记录

新增脚本：`scripts/run_warmstart_sweep.py`。

尝试：256 变量、3 seed、hybrid sweep。结果：第一次运行被外层命令超时打断，并留下子进程继续占用内存/GPU。已确认残留进程都是本项目的 `run_warmstart_sweep.py`、`run_qubo_warmstart.py` 和 25-variable residual QAOA 任务，并已终止。

改进：`run_warmstart_sweep.py` 已加入 `--per-run-timeout`，默认 900 秒。后续某个 seed 卡住时，脚本会自己记录 timeout，避免子进程长期残留。

判断：批量 sweep 是必要路径，但要先保持每个单次实验的输出紧凑、训练时长可控，并设置 per-run timeout；否则容易浪费 3060 资源。

### 1000 变量多 seed 初步稳定性

为了避免 sweep 子进程残留，本轮改为直接单次运行 seed=1 和 seed=2。

配置：`planted_maxcut`，n=1000，average_degree=8，hybrid SQNN，600 epochs，local_search_passes=800。

| seed | 边数 | SQNN sampled ratio | SQNN+repair ratio | repair flips | t0.25 residual vars | t0.25 changed-from-raw |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4002 | 1.000000 | 1.000000 | 0 | 0 | 0 |
| 2 | 3950 | 0.788880 | 1.000000 | 247 | 1 | 248 |

结论：

1. hybrid SQNN 的 raw sampled ratio 对 seed 很敏感；seed=1 直接满分，seed=2 明显偏低。
2. repair+fix residual 压缩更稳定；seed=2 虽然 raw ratio 只有 0.788880，但 repair 后达到 1.0，且 0.25 阈值只剩 1 个变量。
3. changed-from-raw 指标再次说明 raw probability sign 不能直接作为固定值；seed=2 有 248 个高置信固定值来自 repair 翻转。
4. 当前真正可靠的方案应表述为 `hybrid SQNN candidate generation + local repair + confidence mask + residual QAOA`，而不是 `SQNN 直接解 QUBO`。

### V6 repair-calibrated 概率

新增函数：`quantum/warmstart/qaoa_init.py::calibrate_probabilities_with_assignment`。

定义：

\[
c_i = |p_i - 0.5|,
\quad
\tilde p_i =
\begin{cases}
0.5 + c_i, & x_i^{repair}=1,\\
0.5 - c_i, & x_i^{repair}=0.
\end{cases}
\]

也就是说，SQNN 负责给 confidence，局部修复后的 assignment 负责给 sign。这样既不丢掉 SQNN 的软信息，又避免 raw sign 过度自信导致错误固定。

实验：`outputs/warmstart_runs/20260611_201756_planted_maxcut_hybrid_n1000/metrics.json`，seed=2，n=1000，边数 3950。

| 方法 | 近似比 |
|---|---:|
| random best | 0.522779 |
| random + local search | 1.000000 |
| raw hybrid SQNN sample | 0.788665 |
| raw hybrid SQNN sample + repair | 1.000000 |
| repair-calibrated sample | 0.994369 |
| repair-calibrated sample + repair | 1.000000 |

该结果说明 V6 是当前最有展示价值的 warm-start 表达：它不是只报告局部搜索后的满分，而是在不再次局部搜索的情况下，把 raw SQNN sample 从 0.788665 提升到 0.994369。

### V7 active residual QAOA

新增文件：`quantum/warmstart/preprocess.py`。

核心思想：confidence fixing 后得到的 residual QUBO 可能还剩几十个变量，但很多变量已经没有二次边，只剩线性项：

\[
E_i(x_i)=a_i x_i.
\]

这类孤立变量不需要 QAOA，可以精确固定：

\[
x_i^\star =
\begin{cases}
1, & a_i<0,\\
0, & a_i\ge 0.
\end{cases}
\]

然后只把仍有二次相互作用的 active residual 交给 QAOA。这个步骤会显著降低真正需要 full-state 模拟或量子线路处理的 qubit 数。

关键实验：

1. 旧结论：`20260611_193747` 的 1000 变量 hybrid 结果，在阈值 0.25 下 residual 为 25 变量、4 条边；朴素 25-qubit autograd statevector QAOA 曾经超时。
2. 新处理：先消去孤立变量后，同一个 residual 的 active core 只有 8 个变量、4 条边。
3. 结果：active p=1/p=2 residual QAOA 在约数秒内跑完，plus 初态 best expected ratio 约 0.999148/0.999220，exact residual ratio 约 1.0。

这把“25 变量理论可行但当前实现慢”的路径更新为“active-residual QAOA 实际可行”。

新主线因此更新为：

\[
\text{hybrid SQNN}
\rightarrow
\text{repair-calibrated probability}
\rightarrow
\text{confidence fixing}
\rightarrow
\text{isolated-variable exact fixing}
\rightarrow
\text{active residual QAOA}
\]

最新主实验：`outputs/warmstart_runs/20260611_202848_planted_maxcut_hybrid_n1000/metrics.json`。

1. n=1000，边数 3950，seed=2。
2. raw SQNN sampled ratio 为 0.787738。
3. repair-calibrated sampled ratio 为 0.994369。
4. SQNN sampled + repair ratio 为 1.0。
5. t0.30 residual 为 1 变量、0 边；isolated fixing 后 active residual 为 0 变量。
6. t0.49 residual 为 6 变量、0 边；isolated fixing 后 active residual 仍为 0 变量。

判断：对很多经过 repair+fix 的 residual，QAOA 子问题甚至会被孤立变量消元完全吃掉。这不是坏事，反而说明 SQNN warm-start 已经把大规模 QUBO 变成了一个几乎确定的经典后处理问题；只有剩下有边 active core 时才需要 QAOA。

### V8 component-wise residual QAOA

新增功能：

1. `qubo_connected_components`
2. `qubo_component_subproblems`
3. `componentwise_qaoa_resource_summary`
4. `scripts/run_residual_qaoa_demo.py --component-wise`

动机：即使 active residual 仍有几十个变量，只要图分成多个小连通分量，就不需要把全部 active 变量放进一个 statevector。每个连通分量的 QUBO 能量彼此独立，可以分别运行小 QAOA，再把期望能量相加。

注意：component-wise QAOA 使用的是“每个分量独立参数”的工程版本，不等同于全图共享 \(\gamma,\beta\) 的标准 QAOA。它更适合本项目的 hybrid solver 路线：SQNN 负责大规模压缩，QAOA 负责小 active core 的精修。

验证实验：

`outputs/warmstart_runs/20260611_193747_planted_maxcut_hybrid_n1000/residual_qaoa_t0.25.json`

1. 原始问题：1000 变量、4125 边。
2. repair+fix 后 residual：25 变量、4 边。
3. isolated fixing 后 active core：8 变量、4 边。
4. component-wise 后最大连通分量：2 变量。
5. component-wise p=1/p=2 在数秒内跑完。
6. plus 初态 best expected ratio：p=1 为 0.999284，p=2 为 0.999317。
7. SQNN 初态 best expected ratio：p=1 为 0.999013，p=2 为 0.999295。

结论：V8 把 QAOA 支持上限从“active residual 总 qubits”进一步改成“最大连通分量 qubits”。这对大型 QUBO 很关键，因为 warm-start + fixing 后残余图通常非常稀疏。

### planted parity QUBO：第二类严格 benchmark

新增 benchmark：`make_planted_parity_qubo`，对应 CLI 参数：

```powershell
.venv\Scripts\python.exe scripts\run_qubo_warmstart.py --benchmark planted_parity
```

建模方式：每条边给出一个二元 parity 约束，要求：

\[
x_i \oplus x_j = b_{ij}.
\]

其中 \(b_{ij}\) 来自 planted assignment，所以 planted assignment 及其互补解可以满足全部边约束。每条边的 reward 是满足约束的边权，QUBO 最小化目标是负 reward：

\[
E(x)=-\sum_{(i,j)} w_{ij}\mathbf 1[x_i\oplus x_j=b_{ij}].
\]

因此 known optimum 是：

\[
\sum_{(i,j)} w_{ij},
\]

可以严格计算 approximation ratio。这比 random QUBO 更适合做可验证的大规模实验，也比 MaxCut 更通用，因为它同时包含“相同”和“不同”两类二次约束，QUBO 边权有正有负。

#### planted parity 实验结果

256 变量：

`outputs/warmstart_runs/20260611_204401_planted_parity_hybrid_n256/metrics.json`

| 指标 | 数值 |
|---|---:|
| 变量数 | 256 |
| 边数 | 1000 |
| random best ratio | 0.549542 |
| random + local search ratio | 1.000000 |
| raw hybrid SQNN sampled ratio | 0.972300 |
| raw SQNN sampled + repair ratio | 1.000000 |
| repair-calibrated sampled ratio | 1.000000 |
| sampled repair flips | 4 |

1000 变量：

`outputs/warmstart_runs/20260611_204512_planted_parity_hybrid_n1000/metrics.json`

| 指标 | 数值 |
|---|---:|
| 变量数 | 1000 |
| 边数 | 3984 |
| random best ratio | 0.523252 |
| random + local search ratio | 1.000000 |
| raw hybrid SQNN sampled ratio | 0.739505 |
| raw SQNN sampled + repair ratio | 1.000000 |
| repair-calibrated sampled ratio | 0.994300 |
| sampled repair flips | 380 |
| t0.40 residual variables | 2 |
| t0.40 active variables after isolated fixing | 0 |

判断：

1. planted parity 对 raw SQNN 更难，1000 变量时 raw sampled ratio 只有 0.739505。
2. V6 repair-calibrated probabilities 仍然有效，直接把 sampled ratio 提升到 0.994300。
3. 局部 repair 后达到严格最优 ratio 1.0，但需要 380 次翻转，说明 raw SQNN sign 仍不稳定。
4. V7 isolated-variable fixing 之后 t0.40 active residual 为 0，说明这个大规模 QUBO 在 SQNN+repair+fix 后无需再启动 QAOA。
5. 这条 benchmark 支持“项目不只是 MaxCut”的论证：它是一般 parity QUBO，且有严格 known optimum。

### random MaxCut proxy 记录

实验：`random_maxcut_n256_d8.0`，模型为 hybrid SQNN，变量数 256，边数 1002。

因为 random MaxCut 没有已知最优解，本项目脚本现在会把 ratio_reference 标记为 `best_observed_in_run`。这不是严格近似比，只表示相对本次运行中观察到的最好 objective。

结果：

1. random best proxy ratio 为 0.739161。
2. random + local search 达到本次最好观测值，proxy ratio 为 1.0。
3. hybrid SQNN sampled proxy ratio 为 0.999071，sampled + local search 仍为 0.999071，说明该次 SQNN 已经给出局部搜索不再改进的近优入口。
4. 这条结果说明 hybrid SQNN 在一般 random 图上接近强局部搜索 baseline，但还没有证明系统性超过 random+local-search；后续必须做多 seed 批量统计。

追加 random 1000 proxy：

实验：`outputs/warmstart_runs/20260611_202125_random_maxcut_hybrid_n1000/metrics.json`，n=1000，边数 3983，ratio_reference 为 `best_observed_in_run`。

| 方法 | proxy ratio |
|---|---:|
| random best | 0.713924 |
| random + local search | 0.986366 |
| hybrid SQNN sample | 0.998624 |
| hybrid SQNN sample + repair | 1.000000 |
| repair-calibrated sample | 0.999121 |

该实验不是严格近似比，因为没有已知最优值；但它显示在一个 random MaxCut 1000 变量实例上，hybrid SQNN 给出的入口超过了本次 random+local-search baseline，并且 repair 后成为本次最好观测值。residual fixing 在 0.25 阈值下只剩 1 个变量，满足小 QAOA 运行限制。

## 实验日志追加

- 时间戳: `20260611_191322`
- benchmark: `planted_bipartite_maxcut_n64_d4.0`
- model: `instance`
- 变量数: `64`
- 边数: `133`
- device: `cuda`
- 训练秒数: `6.14`
- no-warm-start random best ratio: `0.628339`
- no-warm-start random+local-search ratio: `0.975495`
- SQNN sampled ratio: `0.643693`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `133`
- QAOA p=2 gates: `266`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_191348`
- benchmark: `planted_bipartite_maxcut_n64_d4.0`
- model: `mean_field`
- 变量数: `64`
- 边数: `133`
- device: `cuda`
- 训练秒数: `0.62`
- no-warm-start random best ratio: `0.628339`
- no-warm-start random+local-search ratio: `0.975495`
- SQNN sampled ratio: `0.995362`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `133`
- QAOA p=2 gates: `266`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_191416`
- benchmark: `planted_bipartite_maxcut_n64_d4.0`
- model: `instance`
- 变量数: `64`
- 边数: `133`
- device: `cuda`
- 训练秒数: `15.08`
- no-warm-start random best ratio: `0.628339`
- no-warm-start random+local-search ratio: `0.975495`
- SQNN sampled ratio: `0.643693`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `133`
- QAOA p=2 gates: `266`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_191520`
- benchmark: `planted_bipartite_maxcut_n64_d4.0`
- model: `hybrid`
- 变量数: `64`
- 边数: `133`
- device: `cuda`
- 训练秒数: `15.05`
- no-warm-start random best ratio: `0.628339`
- no-warm-start random+local-search ratio: `0.975495`
- SQNN sampled ratio: `1.000000`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `133`
- QAOA p=2 gates: `266`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_191616`
- benchmark: `planted_bipartite_maxcut_n256_d8.0`
- model: `hybrid`
- 变量数: `256`
- 边数: `1045`
- device: `cuda`
- 训练秒数: `23.02`
- no-warm-start random best ratio: `0.548204`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `1.000000`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `1045`
- QAOA p=2 gates: `2090`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_191818`
- benchmark: `planted_bipartite_maxcut_n1000_d8.0`
- model: `hybrid`
- 变量数: `1000`
- 边数: `4125`
- device: `cuda`
- 训练秒数: `22.56`
- no-warm-start random best ratio: `0.520729`
- no-warm-start random+local-search ratio: `0.898938`
- SQNN sampled ratio: `0.847294`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `4125`
- QAOA p=2 gates: `8250`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_191847`
- benchmark: `planted_bipartite_maxcut_n128_d6.0`
- model: `node_only`
- 变量数: `128`
- 边数: `398`
- device: `cuda`
- 训练秒数: `2.68`
- no-warm-start random best ratio: `0.582847`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `0.583629`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `398`
- QAOA p=2 gates: `796`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_191856`
- benchmark: `planted_bipartite_maxcut_n128_d6.0`
- model: `symmetric`
- 变量数: `128`
- 边数: `398`
- device: `cuda`
- 训练秒数: `11.58`
- no-warm-start random best ratio: `0.582847`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `0.581820`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `398`
- QAOA p=2 gates: `796`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_191857`
- benchmark: `planted_bipartite_maxcut_n128_d6.0`
- model: `directed`
- 变量数: `128`
- 边数: `398`
- device: `cuda`
- 训练秒数: `11.82`
- no-warm-start random best ratio: `0.582847`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `0.581820`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `398`
- QAOA p=2 gates: `796`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_191921`
- benchmark: `planted_bipartite_maxcut_n128_d6.0`
- model: `hybrid`
- 变量数: `128`
- 边数: `398`
- device: `cuda`
- 训练秒数: `9.21`
- no-warm-start random best ratio: `0.582847`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `0.755788`
- SQNN sampled+local-search ratio: `0.755788`
- QAOA p=1 gates: `398`
- QAOA p=2 gates: `796`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_192100`
- benchmark: `planted_bipartite_maxcut_n1000_d8.0`
- model: `hybrid`
- 变量数: `1000`
- 边数: `4125`
- device: `cuda`
- 训练秒数: `75.14`
- no-warm-start random best ratio: `0.520729`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `0.527611`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `4125`
- QAOA p=2 gates: `8250`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_192301`
- benchmark: `planted_bipartite_maxcut_n1000_d8.0`
- model: `hybrid`
- 变量数: `1000`
- 边数: `4125`
- device: `cuda`
- 训练秒数: `76.18`
- no-warm-start random best ratio: `0.520729`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `0.864472`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `4125`
- QAOA p=2 gates: `8250`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_192322`
- benchmark: `planted_bipartite_maxcut_n1000_d8.0`
- model: `mean_field`
- 变量数: `1000`
- 边数: `4125`
- device: `cuda`
- 训练秒数: `3.34`
- no-warm-start random best ratio: `0.520729`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `1.000000`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `4125`
- QAOA p=2 gates: `8250`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_193224`
- benchmark: `planted_bipartite_maxcut_n256_d8.0`
- model: `hybrid`
- 变量数: `256`
- 边数: `1045`
- device: `cuda`
- 训练秒数: `23.72`
- no-warm-start random best ratio: `0.548204`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `1.000000`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `1045`
- QAOA p=2 gates: `2090`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_193452`
- benchmark: `planted_bipartite_maxcut_n1000_d8.0`
- model: `hybrid`
- 变量数: `1000`
- 边数: `4125`
- device: `cuda`
- 训练秒数: `74.54`
- no-warm-start random best ratio: `0.520729`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `0.862477`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `4125`
- QAOA p=2 gates: `8250`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_193747`
- benchmark: `planted_bipartite_maxcut_n1000_d8.0`
- model: `hybrid`
- 变量数: `1000`
- 边数: `4125`
- device: `cuda`
- 训练秒数: `75.15`
- no-warm-start random best ratio: `0.520729`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `0.864472`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `4125`
- QAOA p=2 gates: `8250`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_193943`
- benchmark: `planted_bipartite_maxcut_n1000_d8.0`
- model: `mean_field`
- 变量数: `1000`
- 边数: `4125`
- device: `cuda`
- 训练秒数: `3.25`
- no-warm-start random best ratio: `0.520729`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `1.000000`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `4125`
- QAOA p=2 gates: `8250`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_194349`
- benchmark: `random_maxcut_n256_d8.0`
- model: `hybrid`
- 变量数: `256`
- 边数: `1002`
- device: `cuda`
- 训练秒数: `23.33`
- no-warm-start random best ratio: `0.739161`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `0.999071`
- SQNN sampled+local-search ratio: `0.999071`
- QAOA p=1 gates: `1002`
- QAOA p=2 gates: `2004`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_200259`
- benchmark: `planted_bipartite_maxcut_n128_d6.0`
- model: `hybrid`
- 变量数: `128`
- 边数: `398`
- device: `cuda`
- 训练秒数: `94.49`
- no-warm-start random best ratio: `0.582847`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `0.612275`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `398`
- QAOA p=2 gates: `796`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_200854`
- benchmark: `planted_bipartite_maxcut_n1000_d8.0`
- model: `hybrid`
- 变量数: `1000`
- 边数: `4125`
- device: `cuda`
- 训练秒数: `74.06`
- no-warm-start random best ratio: `0.520729`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `0.860030`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `4125`
- QAOA p=2 gates: `8250`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_201137`
- benchmark: `planted_bipartite_maxcut_n1000_d8.0`
- model: `hybrid`
- 变量数: `1000`
- 边数: `4002`
- device: `cuda`
- 训练秒数: `45.63`
- no-warm-start random best ratio: `0.532702`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `1.000000`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `4002`
- QAOA p=2 gates: `8004`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_201301`
- benchmark: `planted_bipartite_maxcut_n1000_d8.0`
- model: `hybrid`
- 变量数: `1000`
- 边数: `3950`
- device: `cuda`
- 训练秒数: `44.93`
- no-warm-start random best ratio: `0.522779`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `0.788880`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `3950`
- QAOA p=2 gates: `7900`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_201756`
- benchmark: `planted_bipartite_maxcut_n1000_d8.0`
- model: `hybrid`
- 变量数: `1000`
- 边数: `3950`
- device: `cuda`
- 训练秒数: `46.17`
- no-warm-start random best ratio: `0.522779`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `0.788665`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `3950`
- QAOA p=2 gates: `7900`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_202125`
- benchmark: `random_maxcut_n1000_d8.0`
- model: `hybrid`
- 变量数: `1000`
- 边数: `3983`
- device: `cuda`
- 训练秒数: `46.10`
- no-warm-start random best ratio: `0.713924`
- no-warm-start random+local-search ratio: `0.986366`
- SQNN sampled ratio: `0.998624`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `3983`
- QAOA p=2 gates: `7966`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_202848`
- benchmark: `planted_bipartite_maxcut_n1000_d8.0`
- model: `hybrid`
- 变量数: `1000`
- 边数: `3950`
- device: `cuda`
- 训练秒数: `44.99`
- no-warm-start random best ratio: `0.522779`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `0.787738`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `3950`
- QAOA p=2 gates: `7900`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_204401`
- benchmark: `planted_parity_qubo_n256_d8.0`
- model: `hybrid`
- 变量数: `256`
- 边数: `1000`
- device: `cuda`
- 训练秒数: `23.16`
- no-warm-start random best ratio: `0.549542`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `0.972300`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `1000`
- QAOA p=2 gates: `2000`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260611_204512`
- benchmark: `planted_parity_qubo_n1000_d8.0`
- model: `hybrid`
- 变量数: `1000`
- 边数: `3984`
- device: `cuda`
- 训练秒数: `45.43`
- no-warm-start random best ratio: `0.523252`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `0.739505`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `3984`
- QAOA p=2 gates: `7968`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260615_160759`
- benchmark: `planted_parity_qubo_n64_d4.0`
- model: `quantum_data`
- 变量数: `64`
- 边数: `120`
- device: `cuda`
- 训练秒数: `2.10`
- no-warm-start random best ratio: `0.640501`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `0.620883`
- SQNN sampled+local-search ratio: `0.884901`
- QAOA p=1 gates: `120`
- QAOA p=2 gates: `240`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260615_160949`
- benchmark: `planted_parity_qubo_n64_d4.0`
- model: `quantum_data`
- 变量数: `64`
- 边数: `120`
- device: `cuda`
- 训练秒数: `5.39`
- no-warm-start random best ratio: `0.640501`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `0.706876`
- SQNN sampled+local-search ratio: `1.000000`
- QAOA p=1 gates: `120`
- QAOA p=2 gates: `240`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260615_161022`
- benchmark: `planted_parity_qubo_n256_d4.0`
- model: `quantum_data`
- 变量数: `256`
- 边数: `574`
- device: `cuda`
- 训练秒数: `7.84`
- no-warm-start random best ratio: `0.567134`
- no-warm-start random+local-search ratio: `0.802523`
- SQNN sampled ratio: `0.588420`
- SQNN sampled+local-search ratio: `0.936487`
- QAOA p=1 gates: `574`
- QAOA p=2 gates: `1148`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260615_221318`
- benchmark: `planted_parity_qubo_n64_d4.0`
- model: `sync_local`
- 变量数: `64`
- 边数: `120`
- device: `cuda`
- 训练秒数: `2.54`
- no-warm-start random best ratio: `0.640501`
- no-warm-start random+local-search ratio: `1.000000`
- SQNN sampled ratio: `0.714460`
- SQNN sampled+local-search ratio: `0.811471`
- QAOA p=1 gates: `120`
- QAOA p=2 gates: `240`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260615_221342`
- benchmark: `planted_parity_qubo_n256_d4.0`
- model: `sync_local`
- 变量数: `256`
- 边数: `574`
- device: `cuda`
- 训练秒数: `3.65`
- no-warm-start random best ratio: `0.567134`
- no-warm-start random+local-search ratio: `0.802523`
- SQNN sampled ratio: `0.634481`
- SQNN sampled+local-search ratio: `0.990145`
- QAOA p=1 gates: `574`
- QAOA p=2 gates: `1148`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。

## 实验日志追加

- 时间戳: `20260615_221406`
- benchmark: `planted_bipartite_maxcut_n256_d8.0`
- model: `sync_local`
- 变量数: `256`
- 边数: `1023`
- device: `cuda`
- 训练秒数: `2.36`
- no-warm-start random best ratio: `0.551162`
- no-warm-start random+local-search ratio: `0.821803`
- SQNN sampled ratio: `0.539592`
- SQNN sampled+local-search ratio: `0.889109`
- QAOA p=1 gates: `1023`
- QAOA p=2 gates: `2046`
- QAOA full-state possible on 3060 estimate: `False`

记录判断：

- 可行路径：稀疏 QUBO -> SQNN 概率 -> 采样/局部搜索，复杂度随边数线性增长。
- 限制路径：完整大规模 QAOA 不现实；上百/上千变量只能做 warm-start、变量固定或小子问题 QAOA。
- 有向/无向处理：当前模型版本用 directed edge list 承载消息流；`symmetric` 版本强制双向边共享无向特征，`directed` 版本允许方向特征更强表达。
## 15. V10 n=512 expectation-only rounds 1..200 sweep

这次按你的要求，把“从 256 次采样里挑最优 bitstring”从主指标里拿掉，只评估 SQNN 概率态本身的 mean-field QUBO 期望能量：

\[
E[p]=c+\sum_i a_i p_i+\sum_{(i,j)}b_{ij}p_i p_j.
\]

主指标改为：

\[
\text{expected objective ratio}=\frac{-E[p]}{\text{known optimum}}.
\]

解释：这不是 best-shot 指标，而是当前独立 Bernoulli 概率分布如果被测量，平均目标值能达到最优目标值的比例。残余 QAOA 估计仍需要固定高置信变量，所以 residual 部分使用确定性 raw rounding \(p_i\ge 0.5\)，没有采样、没有 best-of-N、没有 repair-calibrated。

输出目录：

```text
outputs/sync_local_v10_n512_expected_rounds_1_200
```

生成文件：

1. `metrics.csv` / `metrics.json`：round 1 到 200 的期望能量、期望 ratio、置信度、确定性 residual 规模、residual QAOA 门数；
2. `model_prefix_trace.pt`：训练后 200-round 模型的 probability trace 和 energy trace；
3. `n512_expected_ratio_vs_rounds_1_200.png`：期望 ratio 随 SQNN 预热轮次变化；
4. `n512_expected_energy_vs_rounds_1_200.png`：期望能量随轮次变化；
5. `n512_expected_active_residual_vs_rounds_1_200.png`：t=0.25 raw fixing 后 residual 变量数；
6. `n512_expected_residual_qaoa_gates_vs_rounds_1_200.png`：残余 QAOA p=1/2/3 双比特门数估计；
7. `n512_expected_confidence_vs_rounds_1_200.png`：平均置信度和高置信比例；
8. `n512_expected_accepted_rounds_1_200.png`：monotone accept 累计接受轮数。

实验设置：

| item | value |
|---|---:|
| benchmark | planted parity QUBO |
| n | 512 |
| average degree | 4 |
| max SQNN rounds | 200 |
| epochs | 120 |
| device | cuda |
| SQNN training seconds | 90.39 |
| total command wall time | about 774.5 s |
| accepted rounds | 96 / 200 |

关键数值：

| round | expected ratio | active residual t=0.25 | residual p=1 gates |
|---:|---:|---:|---:|
| 1 | 0.500000 | 499 | 982 |
| 80 | 0.514222 | 495 | 972 |
| 100 | 0.563289 | 331 | 564 |
| 120 | 0.670094 | 143 | 153 |
| 130 | 0.670853 | 138 | 151 |
| 140 | 0.670853 | 138 | 151 |
| 200 | 0.670853 | 138 | 151 |

最佳点：

1. 最佳 expected ratio：round 130，约 0.670853；
2. 最低 expected energy：round 130，energy 约 -655.502197；
3. 最小 active residual：round 126，active variables 为 132；
4. round 140 到 200 基本没有继续改进，说明当前 200-round 参数化在 130 轮附近已经进入平台。

判断：

1. 期望能量指标比 best-of-256 sampling 更有物理意义：真实量子实验里可以用多次测量估计期望值，但“只报告 shots 里最优的一个”更像后处理竞赛指标。
2. 这次结果说明 V10 的 soft distribution 确实在 100 轮后明显变好：expected ratio 从约 0.50 升到约 0.67，同时 active residual 从约 499 压到约 138。
3. 但是只看期望值，V10 当前还没有达到之前 sample+local-search 指标里的 0.9+，说明高质量 hard solution 主要来自后处理和局部搜索，而不是概率分布本身已经足够尖锐。
4. 运行时间这次主要不在 SQNN 训练，而在每轮每个阈值反复构造 residual QUBO 和 component summary；如果后续只画 expected ratio，可以跳过 residual 统计，大幅加快。
5. 当前更合理的汇报方式应改成两层：主指标用 \(E[p]\) 和 expected ratio；工程接 QAOA 时再报告 deterministic confidence fixing 后的 residual 规模和门数。

## 16. V10 Bloch-X 正半轴检查

根据新的物理约束理解，SQNN 每轮更新中 Bloch 向量的 \(X_i^t\) 分量应保持 \(X_i^t>0\)。如果某些变量的 \(X_i^t<0\)，则该节点的更新方向可以被认为已经翻到错误半轴，局部场 \(F_i\) 对 \(P_Z\) 的推动方向不再可信。

本次只重放已训练好的 SQNN，不重新训练，不接 QAOA，也不做 hard readout。记录 round 0 到 150 的所有变量 Bloch-X：

```text
outputs/sync_local_v10_n512_bloch_x_trace_0_150
```

生成文件：

1. `bloch_x_all_variables_rounds_0_150.png`：512 个变量的 \(X_i^t\) 全量曲线；
2. `bloch_x_summary_quantiles_rounds_0_150.png`：min / p01 / p05 / median / mean / p95 / max；
3. `bloch_x_negative_count_rounds_0_150.png`：每轮 \(X_i^t<0\) 的变量数；
4. `bloch_x_heatmap_rounds_0_150.png`：变量-轮次热图；
5. `bloch_x_summary.csv`：每轮统计；
6. `bloch_x_values_by_variable.csv`：每个变量每轮的 X 值；
7. `bloch_x_report.json` / `bloch_x_trace.pt`：机器可读结果和完整 trace。

关键结论：

| item | value |
|---|---:|
| recorded rounds | 151 |
| variables | 512 |
| first round with \(X<0\) | 102 |
| max negative count | 146 |
| worst min-X round | 115 |
| worst min-X value | -0.990650 |
| final round 150 negative count | 143 |
| final round 150 mean X | 0.358385 |
| final round 150 median X | 0.523796 |

判断：

1. round 0 到 round 101 基本满足 \(X_i^t\ge 0\)；
2. round 102 开始出现 \(X_i^t<0\)；
3. round 110 到 130 之间负 X 变量快速增加，最多达到 146/512；
4. 因此，当前 V10 的后期高轮次结果虽然 \(E[p]\) 继续下降，但已经违反“X 正半轴保证更新方向正确”的结构前提；
5. 下一版模型应加入 \(X\ge 0\) 约束，例如限制旋转角、防止 Bloch-X 穿越零平面，或在每轮 proposal 后进行正半轴投影/拒绝。
## 17. V11 positive-X constrained SQNN 方案

V10 的 Bloch-X 检查说明：round 102 之后开始出现 \(X_i<0\)，最多有 146/512 个变量落入负 X 半轴。根据当前 SQNN-QUBO 的物理解释，局部场更新方向正确的关键不是只看 \(F_i\)，而是看 RY 前的有效 \(X'_i\)：

\[
X'_i=\cos\phi_i^tX_i^t-\sin\phi_i^tY_i^t.
\]

小角度下：

\[
\Delta p_i\approx \frac{\theta_i^tX'_i}{2}.
\]

如果：

\[
\theta_i^t=-\eta_tF_i^t,
\]

那么只有 \(X'_i>0\) 时才有：

\[
F_i^t>0\Rightarrow p_i\downarrow,\qquad F_i^t<0\Rightarrow p_i\uparrow.
\]

因此 V11 的目标是：在每一轮同步更新中，把每个节点保持在 \(X\ge0\) 的正半轴，并尽量保证 RY 前的 \(X'_i>0\)。

### 17.1 一轮更新的物理顺序

第 \(t\) 轮开始时，上一轮已经通过测量/模拟得到：

\[
r_i^t=(X_i^t,Y_i^t,Z_i^t).
\]

先做 phase alignment：

\[
A_i^t=\sqrt{(X_i^t)^2+(Y_i^t)^2},
\]

\[
\delta_i^t=-\operatorname{atan2}(Y_i^t,X_i^t),
\]

\[
R_Z(\delta_i^t)r_i^t=(A_i^t,0,Z_i^t).
\]

这个操作不改变 \(Z_i^t\)，因此不改变：

\[
p_i^t=P(x_i=1)=\frac{1-Z_i^t}{2}.
\]

所以它不会直接改变当前 QUBO 的 \(E[p]\)，只是把 \(X-Y\) 平面的相位积累重新对齐到 \(+X\)。

然后同步计算局部场：

\[
F_i^t=a_i+\sum_jb_{ij}p_j^t.
\]

实现中仍使用归一化局部场：

\[
\hat F_i^t=
\frac{a_i+\sum_jb_{ij}p_j^t}
{|a_i|+\sum_j|b_{ij}|+\epsilon}.
\]

QUBO 相位写入角：

\[
\phi_i^t=\rho_t\hat F_i^t.
\]

RY 概率更新角：

\[
\theta_i^t=\beta_t-\eta_t\hat F_i^t.
\]

由于 phase alignment 本身也是一个 RZ，硬件上不需要真的做三次门：

\[
R_Z(\phi_i^t)R_Z(\delta_i^t)=R_Z(\delta_i^t+\phi_i^t).
\]

所以一轮可以写成：

\[
r_i^{t+1}
=
R_Y(\theta_i^t)R_Z(\delta_i^t+\phi_i^t)r_i^t.
\]

在模拟实现里，为了清楚记录内部量，代码等价地先构造：

\[
(X,Y,Z)\mapsto(\sqrt{X^2+Y^2},0,Z),
\]

再施加 \(R_Z(\phi)\) 和 \(R_Y(\theta)\)。

### 17.2 参数含义

\(\eta_t\)：field step，控制局部场通过 RY 改变 \(Z\) 概率的强度。它是主优化步长。过大时容易把 \(X\) 推到负半轴。

\[
\theta_i^t\approx-\eta_t\hat F_i^t.
\]

\(\rho_t\)：phase step，控制局部场写入 \(X-Y\) 相干平面的强度。它不直接改变 \(p_i\)，但会改变后续 RY 的有效方向和强度。

\[
\phi_i^t=\rho_t\hat F_i^t.
\]

\(\beta_t\)：mixer bias，全局 RY 偏置。为了保持 QUBO local field 的物理意义，V11 默认把它初始化为 0，并限制在很小范围。

\[
|\beta_t|\le0.02.
\]

\(\alpha_t\)：residual blend 系数，控制是否完全接受旋转 proposal：

\[
r_{\text{mix}}^{t+1}
=(1-\alpha_t)r_{\text{aligned}}^t+\alpha_t\tilde r^{t+1}.
\]

如果旧状态和 proposal 都满足 \(X\ge0\)，那么它们的凸组合仍满足 \(X\ge0\)。

### 17.3 衰减策略

V11 使用随轮次衰减的初始 schedule：

\[
\eta_t=\eta_{\min}+(\eta_0-\eta_{\min})\lambda_\eta^t,
\]

\[
\rho_t=\rho_{\min}+(\rho_0-\rho_{\min})\lambda_\rho^t,
\]

\[
\alpha_t=\alpha_{\min}+(\alpha_0-\alpha_{\min})\lambda_\alpha^t.
\]

默认值：

| parameter | value |
|---|---:|
| \(\eta_0\) | 0.12 |
| \(\eta_{\min}\) | 0.02 |
| \(\lambda_\eta\) | 0.97 |
| \(\rho_0\) | 0.04 |
| \(\rho_{\min}\) | 0.005 |
| \(\lambda_\rho\) | 0.97 |
| \(\alpha_0\) | 0.60 |
| \(\alpha_{\min}\) | 0.10 |
| \(\lambda_\alpha\) | 0.97 |

这些 schedule 还带有可训练的非负 scale，因此训练可以调节整体强度，但保持“前期大、后期小”的基本形状。

### 17.4 RZ 裁剪

phase reset 后有：

\[
r_{\text{aligned}}=(A,0,Z),\qquad A\ge0.
\]

再做 QUBO RZ：

\[
X'=\cos\phi\,A.
\]

只要：

\[
|\phi|<\frac{\pi}{2},
\]

就有 \(X'\ge0\)。V11 使用更保守的逐轮衰减裁剪：

\[
\phi_i^t
=
\operatorname{clip}(\rho_t\hat F_i^t,-\phi_{\max}^t,\phi_{\max}^t),
\]

其中：

\[
\phi_{\max}^t
=
\phi_{\min}+(\phi_{\max}^0-\phi_{\min})\lambda_\phi^t.
\]

默认：

| parameter | value |
|---|---:|
| \(\phi_{\max}^0\) | 0.25 |
| \(\phi_{\min}\) | 0.04 |
| \(\lambda_\phi\) | 0.97 |

0.25 rad 约等于 14.3 度，远小于 \(\pi/2\)，因此 RZ 后的 \(X'\) 不会因为相位写入而翻负。

### 17.5 RY 状态相关安全裁剪

RY 后：

\[
X_{\text{out}}=\cos\theta\,X'+\sin\theta\,Z.
\]

即使 \(X'>0\)，如果 \(\theta\) 过大且 \(Z\) 符号不利，仍可能导致 \(X_{\text{out}}<0\)。

V11 先做全局裁剪：

\[
\theta_{\text{clip}}
=
\operatorname{clip}
(\beta_t-\eta_t\hat F_i^t,-\theta_{\max}^t,\theta_{\max}^t).
\]

其中：

\[
\theta_{\max}^t
=
\theta_{\min}+(\theta_{\max}^0-\theta_{\min})\lambda_\theta^t.
\]

默认：

| parameter | value |
|---|---:|
| \(\theta_{\max}^0\) | 0.15 |
| \(\theta_{\min}\) | 0.03 |
| \(\lambda_\theta\) | 0.97 |

然后做状态相关检查。如果：

\[
X_{\text{out}}\le\epsilon,
\]

就把该节点的 \(\theta\) 反复减半。若减半多次后仍不满足，则令该节点：

\[
\theta_i^t=0.
\]

这意味着该节点本轮不做 RY 概率更新，但不会进入负 X 半轴。

默认：

\[
\epsilon=10^{-4},\qquad \text{safety shrink steps}=8.
\]

### 17.6 Bloch ball 安全投影

残差混合后：

\[
r_{\text{mix}}=(1-\alpha)r+\alpha\tilde r.
\]

理论上两个 Bloch ball 内向量的凸组合仍在 Bloch ball 内。为了数值安全，V11 仍执行：

\[
r_{\text{mix}}\leftarrow
\frac{r_{\text{mix}}}{\max(1,\|r_{\text{mix}}\|)}.
\]

这个操作只处理浮点误差，不作为主要物理机制。

### 17.7 monotone accept 的处理

V11 把 phase alignment 视为每轮开始的规范选择。它不改变 \(Z\)，也不改变 \(E[p]\)，所以它总是被保留。

RZ/RY proposal 后，仍然可以使用 V10 的 monotone accept：

1. 计算 proposal 前后的 \(E[p]\)；
2. 如果 \(E[p]\) 不上升，接受 proposal；
3. 如果 \(E[p]\) 上升，拒绝 RZ/RY proposal，但保留 phase-aligned 状态。

这样可以同时保留：

1. \(E[p]\) 的非上升趋势；
2. \(X\ge0\) 的结构约束；
3. 每轮更新角度都由上一轮测量/模拟状态决定。

### 17.8 代码实现状态

已新增模型类：

```python
QUBOPositiveXSynchronousLocalFieldSQNN
```

文件：

```text
quantum/warmstart/qubo_sqnn.py
```

已导出到：

```text
quantum/warmstart/__init__.py
```

已注册命令行模型名：

```text
sync_local_xpos
```

后续运行时可以用：

```text
.venv\Scripts\python.exe scripts\run_qubo_warmstart.py --model sync_local_xpos ...
```

但按照当前要求，本次只完成设计记录和代码实现，暂时不运行实验。

### 17.9 下一步要验证的指标

下一次运行 V11 时必须优先验证：

1. 每轮所有变量是否满足 \(X_i^t\ge0\)；
2. RZ 后 \(X'_i\) 是否始终非负；
3. RY 后 proposal 的 \(X_{\text{out}}\) 是否被安全裁剪；
4. \(E[p]\) 是否仍能下降；
5. hard readout + residual QAOA ratio 是否比 V10 提升；
6. 因为角度更保守，是否需要更多轮数才能达到同等置信度。
## 18. V12 实验实现路线：深电路叠层 vs 每轮重制备

当前 SQNN-QUBO 迭代写成：

\[
r_i^t=(X_i^t,Y_i^t,Z_i^t)
\rightarrow
F_i^t
\rightarrow
R_Z/R_Y
\rightarrow
r_i^{t+1}.
\]

这里有一个真实量子实验必须面对的问题：如果第 \(t\) 轮为了得到 \(X_i^t,Y_i^t,Z_i^t\) 做了测量，那么单次实验中的量子态已经坍缩，不能在同一个量子态上继续接第 \(t+1\) 轮门。因此后续实验实现至少有两条路线。

### 18.1 方案 A：保留相干态，电路逐层叠加

思路：不在中间轮次真正测量并坍缩态，而是把每一轮 SQNN 更新都编译成量子门层，连续接到同一个电路后面：

\[
|\psi^0\rangle
\xrightarrow{U_0}
|\psi^1\rangle
\xrightarrow{U_1}
\cdots
\xrightarrow{U_{T-1}}
|\psi^T\rangle.
\]

如果每轮只包含合并后的 \(R_Z\) 和 \(R_Y\)，那么单变量每轮约为：

\[
R_Z(\delta_i^t+\phi_i^t)R_Y(\theta_i^t)
\]

或根据门顺序写成：

\[
R_Y(\theta_i^t)R_Z(\delta_i^t+\phi_i^t).
\]

优点：

1. 更接近真正的 coherent quantum circuit；
2. 不需要每轮测量后重新制备；
3. 如果所有角度可以提前由经典模拟或训练确定，就能一次性编译成深电路；
4. 概念上更像“SQNN warm-start circuit ansatz”。

问题：

1. 电路深度随轮数线性增长，100 轮就是约 100 层单比特旋转；
2. 如果还要把边相互作用做成受控门，深度和门数会进一步增长；
3. 当前模型的 \(F_i^t\) 依赖上一轮 \(p_j^t\)，如果不测量，真实硬件中无法在电路内部直接知道这些经典概率；
4. 因此 coherent 叠层版本要么只能使用“提前离线算好的角度 schedule”，要么需要中途测量和反馈；
5. 在 NISQ 硬件上，100 层即使只有单比特门也会受噪声影响，若有双比特门更严重。

适合做的版本：

1. 离线训练 SQNN，记录每一轮每个节点的角度：
\[
\delta_i^t,\phi_i^t,\theta_i^t.
\]
2. 把这些角度固定下来，编译成一个 \(T\)-layer warm-start circuit；
3. 最后只测量一次，得到 bitstring 或概率；
4. 与 QAOA 拼接时，把该深电路作为 QAOA initial state preparation。

这个方案的实验重点：

| 指标 | 需要记录 |
|---|---|
| warm-start rounds | \(T\) |
| 单比特门数 | 约 \(2nT\) 或 \(3nT\) |
| 若包含边门 | 约 \(|E|T\) 级别 |
| 最大可承受轮数 | 硬件 coherence / 模拟时间限制 |
| 最终 hard ratio | 最后一次测量/采样得到的 0/1 解 |
| \(E[p]\) | 最后输出分布的期望能量 |

判断：方案 A 物理上更 coherent，但对于 512/1024 变量和 100 轮来说，电路深度很可能过大。它更适合做小规模验证，或作为“离线角度编译”的 warm-start ansatz。

### 18.2 方案 B：每轮测量反馈并重新制备

思路：每一轮结束后，通过多次 shots 估计每个 SQNN 神经元的 Bloch 信息：

\[
X_i^t=\langle\sigma_x\rangle,\quad
Y_i^t=\langle\sigma_y\rangle,\quad
Z_i^t=\langle\sigma_z\rangle.
\]

然后经典计算：

\[
p_i^t=\frac{1-Z_i^t}{2},
\]

\[
F_i^t=a_i+\sum_jb_{ij}p_j^t,
\]

\[
\delta_i^t=-\operatorname{atan2}(Y_i^t,X_i^t),
\]

\[
\phi_i^t=\rho_t\hat F_i^t,\qquad
\theta_i^t=\beta_t-\eta_t\hat F_i^t.
\]

下一轮不在原坍缩态上继续，而是重新制备携带上一轮信息的新初态：

\[
r_i^t=(X_i^t,Y_i^t,Z_i^t)
\]

或 phase-aligned 后的：

\[
\bar r_i^t=(\sqrt{(X_i^t)^2+(Y_i^t)^2},0,Z_i^t).
\]

一个单比特 Bloch 状态可以用两个角制备。若使用纯态近似：

\[
Z=\cos\vartheta,\qquad
X=\sin\vartheta\cos\varphi,\qquad
Y=\sin\vartheta\sin\varphi.
\]

则：

\[
\vartheta_i^t=\arccos(Z_i^t),
\]

\[
\varphi_i^t=\operatorname{atan2}(Y_i^t,X_i^t).
\]

制备门可以写成：

\[
R_Z(\varphi_i^t)R_Y(\vartheta_i^t)|0\rangle.
\]

如果使用 phase-aligned 状态：

\[
Y=0,\quad X\ge0,
\]

则 \(\varphi=0\)，只需要：

\[
R_Y(\arccos Z_i^t)|0\rangle.
\]

优点：

1. 每一轮电路深度很浅，不会随轮数累积；
2. 适合真实硬件做 iterative hybrid loop；
3. 可以自然使用 phase reset，因为 \(X,Y,Z\) 是通过上一轮 tomography/measurement 估计来的；
4. 每轮都可以根据测量结果自适应调整 \(\eta,\rho,\theta,\phi\)；
5. 更接近“SQNN 作为可测量神经元网络”的实验方式。

问题：

1. 每轮需要估计 \(X,Y,Z\)，至少需要 X/Y/Z 三种测量基；
2. shots 开销大，变量数 \(n\) 和轮数 \(T\) 大时，测量成本为 \(O(3nT\times\text{shots})\)；
3. 重新制备会丢失跨轮次的真实相干历史，只保留 Bloch 向量级别的信息；
4. 如果状态是混合态，单纯 \(R_Y/R_Z\) 纯态制备不能完全表示 Bloch 向量长度小于 1 的情况，需要加入噪声/随机化/混合态制备近似；
5. 测量噪声会影响 \(F_i^t\) 和下一轮角度。

适合做的版本：

1. 模拟阶段先用精确 Bloch 向量做 re-prepare loop；
2. 再加入 shots noise，模拟真实测量估计误差；
3. 对 phase-aligned 版本，下一轮只重制备：
\[
(A_i^t,0,Z_i^t)
\]
其中：
\[
A_i^t=\sqrt{1-(Z_i^t)^2}
\]
用于纯态近似，或：
\[
A_i^t=\sqrt{(X_i^t)^2+(Y_i^t)^2}
\]
用于保留测得的混合态 Bloch 长度；
4. 每轮再施加合并相位写入和 RY 更新。

判断：方案 B 更适合当前 SQNN 的神经元解释，也更适合做 100 轮以上的迭代。它牺牲 coherent deep circuit，但换来浅电路、可反馈、可扩展。

### 18.3 两个方案都要试

后续项目应并行保留两条实验路线：

| 路线 | 名称 | 实验含义 | 优先级 |
|---|---|---|---:|
| A | coherent stacked circuit | 把 SQNN 训练出的角度编译成深 warm-start circuit | 中 |
| B | measure-feedback reprepare | 每轮测量 Bloch 向量，重新制备下一轮初态 | 高 |

优先做 B，原因是：

1. 当前模型本来就显式维护每个神经元的 Bloch 向量；
2. phase reset 需要上一轮的 \(X,Y\)，天然适合测量反馈；
3. 大规模 QUBO 的核心瓶颈不是单轮门数，而是全局 QAOA 深度和 statevector 规模；
4. B 可以保持每轮浅电路，适合 512/1024 变量级别的 warm-start。

但 A 也要保留，因为它能回答另一个问题：SQNN warm-start 是否可以被压成一个固定的量子初态制备电路，然后直接接 QAOA。

### 18.4 下一步代码计划

需要新增两个模拟器/评估脚本：

1. `sync_local_xpos_stacked_circuit`：记录 V11 每轮角度，估计 coherent stacked circuit 的门深度、门数、最终 \(E[p]\) 和 hard ratio；
2. `sync_local_xpos_reprepare_loop`：每轮读取/估计 \(X,Y,Z\)，重新制备 phase-aligned 初态，再做下一轮更新。

两者都要记录：

1. 每轮 \(X_i^t\) 是否始终非负；
2. 每轮 \(E[p]\)；
3. hard rounding ratio；
4. hard readout + residual QAOA ratio；
5. shots noise 下的稳定性；
6. 需要的测量次数、门数和最大可支持变量规模。

### 18.5 方案 A 的修正版：测量辅助的电路逐层增长

上面 18.1 的文字容易被误解为“中间完全不测量”。更准确的方案 A 应该是：**每一轮都对当前已构造好的电路做大量测量来估计上一轮输出态，但测量只发生在用于估计的重复 shots 上；得到下一轮规则后，在正式电路结构中移除末端测量模块，并把新的迭代电路接到旧电路后面。**

也就是说，第 \(t\) 轮不是在某一次已经被测量坍缩的量子态上继续演化，而是：

1. 当前已有一个深度为 \(t\) 的电路：
\[
C_t=U_{t-1}\cdots U_1U_0.
\]
2. 为了估计该电路输出的 Bloch 信息，临时运行：
\[
C_t+\text{measurement module}.
\]
3. 通过大量 shots / tomography 得到：
\[
X_i^t,\;Y_i^t,\;Z_i^t.
\]
4. 经典计算：
\[
p_i^t=\frac{1-Z_i^t}{2},
\]
\[
F_i^t=a_i+\sum_jb_{ij}p_j^t,
\]
\[
\delta_i^t=-\operatorname{atan2}(Y_i^t,X_i^t),
\]
\[
\phi_i^t=\rho_t\hat F_i^t,\qquad
\theta_i^t=\beta_t-\eta_t\hat F_i^t.
\]
5. 构造下一层：
\[
U_t=\prod_i R_Y(\theta_i^t)R_Z(\delta_i^t+\phi_i^t).
\]
6. 正式下一轮电路变成：
\[
C_{t+1}=U_tC_t.
\]

这句话可以概括为：

```text
先把当前电路末端接测量模块，用很多 shots 得到下一层规则；
再把测量模块拿掉，在旧电路后面接新的 U_t。
```

因此，方案 A 不是“不测量”，而是“每轮都测量很多次来设计下一层，但最终被增长出来的是一个不含中间测量的深电路”。

这个方案和方案 B 的区别是：

| 对比项 | 方案 A：测量辅助电路增长 | 方案 B：测量反馈重制备 |
|---|---|---|
| 每轮是否测量 | 是，用当前 \(C_t\) 的重复运行估计 \(X,Y,Z\) | 是，用上一轮浅电路估计 \(X,Y,Z\) |
| 下一轮怎么做 | 移除测量模块，把 \(U_t\) 接到旧电路后面 | 重新制备一个携带上一轮信息的新初态 |
| 电路深度 | 随轮数累积，\(C_T=U_{T-1}\cdots U_0\) | 每轮深度基本固定，不累积 |
| 是否保留电路历史 | 保留，历史被编进深电路 | 不保留完整相干历史，只保留 Bloch 信息 |
| 主要成本 | 深电路噪声和门数 | shots/tomography 与重制备误差 |

对当前 SQNN 项目的意义：

1. 方案 A 可以回答：SQNN 迭代是否能被“编译”为一个逐层生长的 warm-start 电路；
2. 方案 B 可以回答：SQNN 是否可以作为真实可测量神经元网络，用浅电路多轮反馈优化；
3. 两者都需要大量测量，但测量的角色不同：A 的测量用于设计下一层深电路，B 的测量用于决定下一轮重新制备的初态；
4. 方案 A 最终电路可以直接接 QAOA，方案 B 更像 classical-quantum iterative optimizer。

后续代码命名应调整为：

1. `sync_local_xpos_growing_circuit`：对应方案 A，测量辅助的逐层增长电路；
2. `sync_local_xpos_reprepare_loop`：对应方案 B，每轮测量反馈并重制备。

方案 A 的评估必须额外记录：

1. 第 \(t\) 轮当前电路深度；
2. 每轮 tomography shots；
3. 累积单比特门数；
4. 如果加入边门，累积双比特门数；
5. 噪声模型下深电路是否仍能保持有效 \(X>0\) 和较低 \(E[p]\)。

### 18.6 V11 positive-X 方向裕量 \(J\) trace 实验

本次回到模型方向性问题：若要保证每个变量每一轮都沿着局部降能方向移动，需要检查
\[
J_i^t=-F_i^t\left(p_{i,\mathrm{proposal}}^{t+1}-p_i^t\right)>0.
\]
其中 \(F_i^t=\partial E[p]/\partial p_i\) 是当前均值场局部场，\(J_i^t>0\) 表示第 \(i\) 个变量在第 \(t\) 轮的概率更新方向与局部降能方向一致。

执行命令：

```powershell
.venv\Scripts\python.exe scripts\evaluate_sync_local_xpos_j_trace.py --device cuda --max-rounds 150 --epochs 120 --n 512 --average-degree 4.0 --seed 17
```

输出目录：

```text
outputs/sync_local_xpos_n512_j_trace_1_150
```

生成文件包括：

1. `j_heatmap_variables_vs_rounds_1_150.png`：横轴 round，纵轴 variable，颜色为 \(J\)；
2. `j_values_by_variable.csv`：每个变量每一轮的 \(J\) 数值；
3. `j_summary_vs_rounds_1_150.png`：\(J\) 的 min / p01 / p05 / median / mean；
4. `j_x_violation_counts_vs_rounds_1_150.png`：\(J<0\)、\(X<0\)、after-RZ \(X<0\) 计数；
5. `xpos_ratio_vs_rounds_1_150.png` 和 `xpos_energy_vs_rounds_1_150.png`：1 到 150 轮优化效果；
6. `metrics.csv` / `metrics.json` / `model_j_trace.pt`。

本次运行环境和耗时：

1. device：`cuda`；
2. GPU：`NVIDIA GeForce RTX 3060`；
3. training seconds：约 `152.62` 秒；
4. 150 轮中 accepted rounds：`121`。

关键结果：

| 指标 | 数值 |
|---|---:|
| best mean-energy ratio | `0.587298` at round `150` |
| best direct-rounding ratio | `0.522522` at round `149` |
| final mean-energy ratio | `0.587298` |
| final mean confidence | `0.322864` |
| any accepted-state \(X<0\) | `False` |
| any after-RZ \(X<0\) | `False` |
| any \(J<0\) | `True` |
| rounds with \(J<0\) | `150 / 150` |
| max \(J<0\) variables in one round | `299 / 512` |
| worst \(J_\min\) | `-0.00022982` at round `85` |
| final \(J<0\) fraction | `0.583984` |

判断：

1. V11 positive-X 机制确实守住了 Bloch 几何中的关键条件：accepted state 的 \(X\) 没有变负，after-RZ 的 \(X\) 也没有变负；
2. 但这还不能推出逐变量方向裕量 \(J_i^t\) 恒正。本次 150 轮里每一轮都有部分变量 \(J<0\)，说明当前模型仍不能保证“每个变量每次迭代方向都不会错误”；
3. 虽然 \(J\) 有负值，整体 mean-field energy 仍从 ratio `0.502210` 提升到 `0.587298`，说明全局 monotone accept 可以保证整体 \(E[p]\) 不恶化，但它不是逐变量方向正确性证明；
4. 下一步如果要严格保证 \(J_i^t>0\)，需要把 \(J\) 作为 hard constraint 或 safety projection，而不只是依赖 positive-X、角度裁剪和全局 monotone accept。

### 18.7 Reset / \(J\) 约束 300 轮 ablation

为了确认 reset 后近似比变差是否只是“优化轮次不够”，做了 300 轮 ablation。所有组使用同一个 benchmark / seed / epoch 设置：

```powershell
.venv\Scripts\python.exe scripts\evaluate_sync_local_reset_ablation_300.py --device cuda --n 512 --max-rounds 300 --epochs 120 --seed 17 --average-degree 4.0 --output-dir outputs\sync_local_reset_ablation_n512_rounds_1_300
```

输出目录：

```text
outputs/sync_local_reset_ablation_n512_rounds_1_300
```

对比组：

1. `no_reset_v10`：保留原 V10 相干相位记忆；
2. `reset_every_round`：每轮开头 reset 到 \(Y=0,X\ge 0\)；
3. `reset_every_5`：每 5 轮 reset 一次；
4. `reset_every_5_after_warmup`：先保留 5 轮相干演化，然后每 5 轮 reset 一次；
5. `x_guard_no_reset`：不清空全部 \(Y\)，只把 after-RZ 的负 \(X\) 修回 \(X\ge\epsilon\)；
6. `no_reset_j_penalty`：完全保留 V10 相干相位，但训练时加入 \(\mathrm{ReLU}(-J)\) 软惩罚。

结果总表：

| case | best mean ratio | best round | final mean ratio | best rounding ratio | accepted | max \(J<0\) vars | final \(J<0\) frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_reset_v10` | `0.665238` | `210` | `0.665238` | `0.774386` | `62` | `312` | `0.259766` |
| `reset_every_round` | `0.500000` | `1` | `0.500000` | `0.501975` | `300` | `0` | `0.000000` |
| `reset_every_5` | `0.500000` | `1` | `0.500000` | `0.501975` | `300` | `0` | `0.000000` |
| `reset_every_5_after_warmup` | `0.500000` | `1` | `0.500000` | `0.501975` | `300` | `0` | `0.000000` |
| `x_guard_no_reset` | `0.500000` | `1` | `0.500000` | `0.501975` | `300` | `0` | `0.000000` |
| `no_reset_j_penalty` | `0.704676` | `299` | `0.704676` | `0.789360` | `187` | `372` | `0.355469` |

额外诊断：

| case | final probability std | final mean confidence |
|---|---:|---:|
| `no_reset_v10` | `0.327036` | `0.303902` |
| `no_reset_j_penalty` | `0.359927` | `0.342798` |
| `reset_every_round` | `0.000000` | `0.000000` |
| `reset_every_5` | `0.000000` | `0.000000` |
| `reset_every_5_after_warmup` | `0.000000` | `0.000000` |
| `x_guard_no_reset` | `0.000000` | `0.000000` |

判断：

1. reset 后变差不是因为 150 轮不够。即使跑到 300 轮，reset / X-guard 组仍停在 \(p_i=0.5\)，概率标准差和平均置信度都为 0；
2. 这个 planted parity benchmark 在 \(p_i=0.5\) 附近存在强对称/局部场停滞点。原 V10 能离开这个点，靠的是多轮 \(X/Y\) 相干相位积累带来的高阶响应，而不是每轮单独的局部贪心方向；
3. 每轮 reset 或把负 \(X\) 修回正半轴，会把这条高阶相干路径剪掉，因此虽然 \(J<0\) 消失了，模型也失去了优化能力；
4. `no_reset_j_penalty` 说明更可行的方向不是 hard reset / hard projection，而是保留 \(X/Y\) 相位记忆，同时用软 \(J\) penalty 限制方向错误的幅度。它把 best mean ratio 从 `0.665238` 提升到 `0.704676`，direct rounding ratio 从 `0.774386` 提升到 `0.789360`；
5. 但 `no_reset_j_penalty` 仍不能保证每个变量 \(J>0\)。它主要降低了严重负 \(J\) 的破坏程度，而不是让负 \(J\) 计数归零。

下一步路线：

1. reset / positive-X hard projection 路线正式放弃，不再作为当前 SQNN warm-start 主线；
2. 应保留相干 \(Y\) 记忆，把 \(J\) 约束改为 soft regularization 或 trust-region；
3. 如果需要可证明的逐变量方向安全，应该只在 proposal 层做小步回退，而不是每轮强制 reset 全部相位。

### 18.8 V12 候选路线：\(J\)-regularized SQNN

基于 18.7 的 ablation，当前更合理的新路线不是 reset，而是：

```text
保留 V10 的相干相位记忆；
不强制 X 始终为正；
不清空 Y；
训练时加入负 J 的软惩罚；
用软约束减少严重错误方向，而不是用 hard reset 切断相干路径。
```

暂定命名：

```text
V12 J-regularized SQNN
```

#### 18.8.1 为什么放弃 reset

reset 的目标原本是让每一轮开始时：

\[
(X_i^t,Y_i^t,Z_i^t)\rightarrow
(\sqrt{(X_i^t)^2+(Y_i^t)^2},0,Z_i^t),
\]

从而让后续小角度 \(R_Y\) 的一阶概率变化满足更稳定的符号关系。

但实验说明，这个机制在 planted parity 上会把模型卡死在：

\[
p_i=0.5.
\]

原因是这个 benchmark 在 \(p=0.5\) 附近有强对称停滞点。原 V10 能离开它，依赖的是多轮 \(X/Y\) 相干相位积累产生的高阶破对称响应。reset 或 positive-X hard projection 会清掉 \(Y\)，也会阻断 \(X<0\) 半轴上的相位路径，因此虽然方向错误看似减少，模型本身却失去优化能力。

因此 reset 不是“训练轮次不够”的问题，而是机制和该类 QUBO 的破对称需求冲突。

#### 18.8.2 \(J\)-regularized SQNN 的核心定义

保留 V10 的状态演化：

1. 每个变量仍然有 Bloch 状态：
\[
(X_i^t,Y_i^t,Z_i^t).
\]
2. 每轮不做 reset；
3. 概率仍从 \(Z\) 读出：
\[
p_i^t=\frac{1-Z_i^t}{2}.
\]
4. 局部场为：
\[
F_i^t=\frac{\partial E[p]}{\partial p_i}.
\]
5. proposal 产生后定义方向裕量：
\[
J_i^t=-F_i^t\left(p_{i,\mathrm{proposal}}^{t+1}-p_i^t\right).
\]

其中 \(J_i^t>0\) 表示该变量的 proposal 沿局部降能方向移动，\(J_i^t<0\) 表示该变量当前 proposal 的局部方向相反。

训练目标从原来的：

\[
\mathcal L
= \frac{E[p]}{n\cdot \mathrm{scale}}
-\lambda_H H[p]
\]

改成：

\[
\mathcal L
= \frac{E[p]}{n\cdot \mathrm{scale}}
-\lambda_H H[p]
+\lambda_J\frac{1}{Tn}\sum_{t,i}\operatorname{ReLU}(-J_i^t).
\]

这里：

1. \(\lambda_H\)：entropy 正则，防止过早坍缩；
2. \(\lambda_J\)：负 \(J\) 软惩罚权重；
3. \(\operatorname{ReLU}(-J)\)：只惩罚 \(J<0\) 的部分；
4. 这个约束是 soft regularization，不是 hard constraint。

需要特别说明：这里的 loss 不是“用梯度下降直接优化每个变量 \(p_i\) 或 \(x_i\)”。

当前有两层过程：

1. **SQNN forward / inference 层**：给定一个 QUBO，变量状态按 SQNN 迭代规则更新：
\[
(X^t,Y^t,Z^t)\rightarrow (X^{t+1},Y^{t+1},Z^{t+1}),
\]
并通过 \(Z\) 读出 \(p_i\)。这一层不是对 \(p_i\) 做梯度下降，而是局部场驱动的 Bloch 旋转和 monotone accept。
2. **参数训练层**：为了让 SQNN 的全局规则更好，会用 PyTorch / AdamW 训练少量可学习参数，例如每轮的 field step、phase step、mixer bias、initial angles 等。这一层才使用 loss。

因此，loss 的作用不是“每一步把变量沿梯度下降推过去”，而是训练 SQNN 的更新规则，让同一套更新规则在 forward 时更倾向于产生低能量概率和较少严重负 \(J\) proposal。

更具体地说：

```text
loss 作用对象：SQNN 参数，例如每轮步长、相位步长、mixer bias。
loss 不直接优化对象：QUBO 变量本身、每个 p_i、每个 x_i。
```

如果未来要做完全无训练版本，则 \(J\) 不能以 training loss 的形式发挥作用，而应该改成 inference-time trust-region：proposal 出来后检查 \(J\)，对严重负 \(J\) 的变量缩小步长或回退。

#### 18.8.3 它和 reset 路线的区别

| 对比项 | reset / positive-X | \(J\)-regularized SQNN |
|---|---|---|
| 是否清空 \(Y\) | 是 | 否 |
| 是否强制 \(X\ge 0\) | 是 | 否 |
| 是否保留相干相位记忆 | 基本不保留 | 保留 |
| 是否保证 \(J>0\) | 不一定；实验中 V11 仍有负 \(J\) | 不保证 |
| 如何处理方向错误 | hard reset / hard projection | soft penalty |
| planted parity 表现 | 卡在 \(p=0.5\) | 明显优于 V10 baseline |
| 风险 | 切断破对称路径 | 仍有部分变量 \(J<0\) |

关键直觉：

```text
reset 是把相位自由度删掉；
J-regularization 是让相位自由度继续存在，但对明显走错方向的 proposal 收费。
```

#### 18.8.4 当前实验结果

在 `n=512, rounds=300, epochs=120, seed=17` 的 ablation 中：

| case | best mean ratio | best rounding ratio | final mean confidence |
|---|---:|---:|---:|
| `no_reset_v10` | `0.665238` | `0.774386` | `0.303902` |
| `no_reset_j_penalty` | `0.704676` | `0.789360` | `0.342798` |
| reset / X-guard 系列 | `0.500000` | `0.501975` | `0.000000` |

这说明：

1. \(J\)-regularized SQNN 保留了 V10 的破对称能力；
2. 它比原始 V10 产生更高置信度的概率分布；
3. 它提升了 mean-field ratio 和 direct rounding ratio；
4. 它没有像 reset 那样卡在 \(p=0.5\)。

同时也要注意：

1. 它没有让所有 \(J\) 都变正；
2. final \(J<0\) fraction 仍为 `0.355469`；
3. max \(J<0\) 变量数仍可能很高；
4. 它主要降低严重负 \(J\) 对整体优化的破坏，并让模型学到更稳的相干路径，而不是给出逐变量单调性证明。

#### 18.8.5 这条路线的研究意义

这条路线更符合当前模型的真实优势：

1. SQNN 不是纯局部贪心优化器；
2. 它靠 \(X/Y/Z\) 三维状态在多轮中积累相干记忆；
3. 某些局部 \(J<0\) 可能是全局耦合协调的一部分；
4. 因此不应该强制每个变量每一步都局部正确；
5. 但可以惩罚严重的局部反向移动，让模型少走“明显坏”的方向。

换句话说，当前更合理的理论表述是：

```text
SQNN 不保证逐变量逐轮方向全对；
SQNN 通过相干相位记忆进行多变量协同；
J-regularization 作为方向正则，减少严重局部反向更新；
全局 monotone accept 保证 E[p] 不恶化。
```

#### 18.8.6 下一步实验

后续优先做以下实验：

1. sweep \(\lambda_J\)：例如 `1, 5, 10, 20, 50, 100`；
2. 比较 penalty 形式：
\[
\operatorname{ReLU}(-J),\quad
\operatorname{ReLU}(-J)^2,\quad
\operatorname{softplus}(-J/\tau);
\]
3. 记录负 \(J\) 的 magnitude，而不仅是 count；
4. 测试 `planted_parity` 的 `n=128/256/512/1024`；
5. 测试 `planted_maxcut`，确认这不是 parity-only 现象；
6. 加入 trust-region 版本：只对严重负 \(J\) 的变量 shrink proposal，而不是 reset 状态；
7. 与 repair-calibrated probability / local search 组合，观察 residual active core 是否继续缩小。

当前主线建议：

```text
V10 sync-local
    -> V12 J-regularized sync-local
    -> V12 + trust-region proposal shrink
    -> V12 + repair-calibrated warm-start / residual QAOA
```

### 18.9 当前阶段聚焦声明

当前阶段先集中推进：

```text
V12 J-regularized SQNN
```

也就是说，后续实验、代码实现和计划书记录都优先围绕这条路线展开：

```text
保留 V10 的 X/Y/Z 相干状态；
不做 reset；
不做 positive-X hard projection；
用 J-regularization 作为方向正则；
继续用 monotone accept 保证 mean-field expected energy 不恶化；
输出 warm-start 概率 p，用于 sampling / rounding / local search / residual QAOA。
```

现阶段暂停推进的路线：

1. reset every round；
2. periodic reset；
3. positive-X hard projection；
4. after-RZ \(X\ge 0\) hard guard；
5. 任何会清空 \(Y\) 相干记忆的方案。

暂停原因：

1. 300 轮 ablation 已经显示 reset / X-guard 系列全部卡在 \(p_i=0.5\)；
2. 这些方案虽然消除了负 \(J\)，但同时消除了模型的破对称能力；
3. 对当前 planted parity benchmark 来说，保留 \(X/Y\) 相干记忆比逐变量方向硬约束更重要。

短期目标：

1. 把 `J-regularized SQNN` 从 ablation 脚本整理成正式模型/训练脚本；
2. 做 \(\lambda_J\) sweep，确认最优方向正则强度；
3. 比较不同 penalty 形式：
\[
\operatorname{ReLU}(-J),\quad
\operatorname{ReLU}(-J)^2,\quad
\operatorname{softplus}(-J/\tau);
\]
4. 在 `planted_parity` 上跑 `n=128/256/512/1024`；
5. 在 `planted_maxcut` 上复验，确认不是 parity-only 现象；
6. 记录 mean ratio、rounding ratio、repair/local-search ratio、负 \(J\) magnitude、confidence 和 residual active core；
7. 之后再考虑 trust-region proposal shrink，作为 V12 的增强版，而不是回到 reset。

当前判断：

```text
V12 的研究问题不再是“如何让每个变量每轮 J 都严格为正”，
而是“如何在保留相干破对称能力的同时，用 J 正则减少严重局部反向更新”。
```

### 18.10 V12 长跑探索任务

用户要求：围绕 `V12 J-regularized SQNN` 系统探索模型潜力，七个方向全部尝试一遍，持续运行至少 8 小时，及时保存结果，并在探索后继续思考改进。

新增脚本：

```text
scripts/explore_j_regularized_sqnn.py
```

正式输出目录：

```text
outputs/j_regularized_exploration_8h
```

探索方向：

1. \(\lambda_J\) sweep；
2. penalty 形式比较：`ReLU(-J)`、`ReLU(-J)^2`、`softplus(-J/tau)`；
3. 轮次权重比较：flat、linear up、sqrt up、linear down、late half；
4. `all proposals` vs `accepted proposals only`；
5. trust-region proposal shrink，不 reset、不清空 \(Y\)；
6. 泛化测试：`planted_parity` / `planted_maxcut`，不同 `n`、seed、average degree；
7. residual QAOA 价值：记录 high-confidence fixing 后的 active variables、active edges、最大连通分量和 local-search / sampling 指标。

记录文件：

1. `summary.csv`：每个 run 的核心结果；
2. `run_status.json`：当前进度和当前最佳配置；
3. `final_report.json`：长跑结束后的总报告；
4. `runs/<run_id>/metrics.json`：单个配置详细结果；
5. `runs/<run_id>/trace_rows.csv`：逐轮指标；
6. `summary_best_ratios.png`：当前 top runs 图。

当前实验原则：

```text
保持模型干净；
不引入 reset；
不引入 positive-X hard projection；
只围绕 J-regularization / round weighting / accepted-only / trust-region shrink 做探索。
```

### 18.11 V12 8 小时探索完成记录

本轮长跑已经完成。执行命令：

```powershell
.venv\Scripts\python.exe scripts\explore_j_regularized_sqnn.py --device cuda --output-dir outputs\j_regularized_exploration_8h --time-budget-hours 8 --min-hours 8 --resume
```

总运行结果：

| 指标 | 数值 |
|---|---:|
| completed runs | `137` |
| elapsed hours | `8.0135` |
| 输出目录 | `outputs/j_regularized_exploration_8h` |
| 分析报告 | `outputs/j_regularized_exploration_8h/analysis_report.md` |

辅助分析脚本：

```text
scripts/analyze_j_regularized_exploration.py
```

分析输出：

```text
outputs/j_regularized_exploration_8h/analysis_report.json
outputs/j_regularized_exploration_8h/analysis_report.md
```

#### 18.11.1 当前最大潜力

本轮探索里，干净 V12 模型的当前最佳结果如下：

| 场景 | best expected ratio | best rounded ratio | round + local search | sample + local search | residual active | max component |
|---|---:|---:|---:|---:|---:|---:|
| `planted_parity`, `n=128`, `d=6`, seed `17` | `0.881953` | `0.988860` | `1.000000` | `1.000000` | `3` | `3` |
| `planted_parity`, `n=512`, trust-region, seed `17` | `0.816512` | `0.817322` | `0.817411` | `0.822768` | `0` | `0` |
| `planted_parity`, `n=512`, adaptive seed `736` | `0.800707` | `0.966370` | `0.988034` | `0.977631` | `48` | `7` |
| `planted_parity`, `n=1024`, seed `17` | `0.697916` | `0.783948` | `0.830844` | `0.782444` | `232` | `163` |
| `planted_parity`, `n=1024`, `d=6`, seed `17` | `0.656934` | `0.799416` | `0.861978` | `0.764150` | `405` | `401` |
| `planted_maxcut`, best observed mean-field | `0.500000` | `0.000000` | case-dependent | up to about `0.68` on `n=512` | large | large |

当前可以给出的潜力判断是：

```text
在 planted_parity / signed sparse QUBO 上，
V12 已经能把 n=512 的 clean mean-field expected ratio 推到约 0.82；
如果接 rounding / local search，多个 n=512 case 可以接近 0.99 或达到 1.0；
n=1024 仍能明显优于随机，但 expected ratio 约停在 0.70 左右，说明扩展能力还有瓶颈；
planted_maxcut 在当前干净 V12 下没有被真正学动，mean-field 仍卡在 0.5。
```

#### 18.11.2 七个方向逐项结论

1. \(\lambda_J\) sweep：

   在 `n=512, planted_parity, seed=17` 上，\(\lambda_J=100\) 的 best expected ratio 达到 `0.711434`，比无 \(J\) 正则 baseline `0.664456` 明显更好。说明 \(J\)-regularization 有真实贡献，但单纯加大 \(\lambda_J\) 不是最终答案。

2. penalty 形式：

   `ReLU(-J)` 仍是最稳的形式。本轮里 `relu_sq` 和 `softplus` 没有超过 `ReLU(-J)` 的最好结果。当前不建议把主线切到 squared 或 softplus penalty。

3. 轮次权重：

   `late_half` 在 `n=512, seed=17` 上把 best expected ratio 推到 `0.719813`，优于 flat sweep 的常规结果。直觉是：早期允许相干态自由破对称，后期再强约束方向错误，可能比全程均匀惩罚更合理。

4. accepted-only：

   accepted-only 变体在 \(\lambda_J=100\) 下达到 `0.714788`。它比 baseline 好，但不如 trust-region 的最好结果。说明只惩罚被全局接受的 proposal 有一定意义，但不是最强信号。

5. trust-region proposal shrink：

   当前最强的 `n=512` expected ratio 来自 trust-region：`trust_shrink=0.0, trust_threshold=1e-4`，best expected ratio 为 `0.816512`，并且 threshold `0.25` 下 residual active variables 为 `0`。这说明“对明显负 \(J\) 的 proposal 做小步回退/冻结”比 reset 更有前途。

   但这里也有一个重要风险：这个配置的 direct rounding / local search 只有约 `0.817`，没有像某些 adaptive seed 一样接近 1.0。也就是说，它让概率分布非常自信、residual 很小，但自信方向未必全对。因此下一步必须做 threshold / calibration 检查。

6. 泛化测试：

   `planted_parity` 泛化明显成立：`n=128/256/512/1024` 都能跑出高于随机的结果，且 `n=512` 上已经很强。`n=1024` 的 best expected ratio 约 `0.697916`，round + local search 最高达到 `0.861978`，说明模型能扩展，但还没有把大规模问题压到很小 residual core。

   `planted_maxcut` 泛化不成立：mean-field expected ratio 基本卡在 `0.5`，direct rounding 常为 `0`。这不是 GPU 或轮次问题，而更像当前全局同构初始化和同步局部规则无法打破 MaxCut 的二分对称性。

7. residual QAOA / local repair 价值：

   在 `planted_parity` 上，V12 的高置信 fixing 能显著缩小 residual。最强 `n=128` case 只剩 `3` 个 active variables；多个 `n=512` adaptive seed 的最大 residual component 在几十以内，round + local search 经常接近 `0.99`。这说明 V12 更适合作为 warm-start / residual reduction 前端，而不是单独作为最终求解器。

#### 18.11.3 当前阶段结论

本轮探索后，路线判断更新为：

```text
reset / positive-X hard projection 路线正式放弃；
J-regularized SQNN 是当前主线；
V12 的优势集中在 signed sparse parity-like QUBO；
trust-region shrink 是最值得继续挖的增强；
late-half J weighting 也值得保留；
planted_maxcut 暂时不能声称被 V12 解决，需要单独处理破对称问题。
```

更具体地说：

1. 当前 clean V12 的最大 mean-field 潜力已经看到 `0.881953`；
2. 当前 clean V12 在 `n=512` 上的最大 mean-field 潜力是 `0.816512`；
3. 当前 pipeline 潜力如果允许 rounding / local search，可以在若干 `planted_parity` case 上达到 `0.99-1.00`；
4. 当前 `n=1024` 仍是瓶颈，best expected ratio 约 `0.70`；
5. 当前 `planted_maxcut` 不是 V12 的成功案例，而是暴露 symmetry breaking 问题的诊断案例。

### 18.12 V12 targeted-improve 计划

8 小时全量探索之后，下一步不再盲目扩大 sweep，而是集中做 targeted improvement。新增脚本模式：

```powershell
.venv\Scripts\python.exe scripts\explore_j_regularized_sqnn.py --targeted-improve --device cuda --output-dir outputs\j_regularized_targeted_improve
```

targeted-improve 只做三类事情，仍保持模型干净：

1. `targeted_trust_n512`：

   围绕当前 `n=512` 最强结果，细扫 `trust_threshold` 和 \(\lambda_J\)，确认 `trust_shrink=0` 的 `0.816512` 是真实可复现增益，还是某个阈值造成的过度自信。

2. `targeted_best_seed_n512`：

   对已经出现高 rounding/local-search ratio 的 seeds `736/732/706`，测试 trust-region 是否能在不损伤最终 assignment 的情况下进一步提高 expected ratio 和缩小 residual。

3. `targeted_scale_n1024` / `targeted_scale_trust_n1024`：

   针对 `n=1024` 的瓶颈，增加 rounds/epochs，比较 \(\lambda_J=50/100\)、`flat/late_half`、以及轻量 trust-region。目标是把 `n=1024` best expected ratio 从约 `0.70` 往上推，并观察 residual max component 能否从上百降下来。

下一步改进原则：

```text
先把 V12 在 planted_parity 上的上限挖透；
优先提高 n=1024 scaling；
谨慎使用 trust-region，避免把错误方向过早固定；
MaxCut 暂时只作为失败诊断，不把破对称结构硬塞进 V12 主线。
```

### 18.13 V12 targeted-improve 完成记录

在 8 小时全量探索之后，又继续完成了一轮 targeted-improve。执行方式为分批 resume：

```powershell
.venv\Scripts\python.exe scripts\explore_j_regularized_sqnn.py --targeted-improve --device cuda --output-dir outputs\j_regularized_targeted_improve --resume
```

输出目录：

```text
outputs/j_regularized_targeted_improve
```

本轮完成：

| 指标 | 数值 |
|---|---:|
| completed targeted runs | `31` |
| targeted 累计运行时间 | 约 `3.5` 小时 |
| 分析报告 | `outputs/j_regularized_targeted_improve/analysis_report.md` |

#### 18.13.1 targeted-improve 后的新上限

| 场景 | 配置 | best expected ratio | rounded | round + local search | sample + local search | active | max component |
|---|---|---:|---:|---:|---:|---:|---:|
| `n=512`, seed `17` | `trust_shrink=0.0`, `threshold=5e-4`, \(\lambda_J=50\) | `0.821887` | `0.819558` | `0.929502` | `0.899585` | `7` | `3` |
| `n=512`, seed `706` | `trust_shrink=0.25`, `threshold=1e-4`, \(\lambda_J=50\) | `0.797521` | `0.827447` | `0.959963` | `0.909972` | `29` | `10` |
| `n=512`, seed `732` | no shrink, \(\lambda_J=50\) | `0.775240` | `0.907880` | `0.992165` | `0.987135` | `84` | `59` |
| `n=1024`, seed `17` | `trust_shrink=0.25`, `threshold=1e-4`, \(\lambda_J=50\) | `0.764526` | `0.801269` | `0.813386` | `0.815897` | `6` | `2` |
| `n=1024`, seed `17` | no shrink, \(\lambda_J=100\), flat | `0.748968` | `0.803216` | `0.835877` | `0.824640` | `106` | `13` |
| `n=1024`, seed `42` | no shrink, \(\lambda_J=100\), flat | `0.744658` | `0.809713` | `0.835192` | `0.828486` | `97` | `33` |
| `n=1024`, seed `23` | no shrink, \(\lambda_J=100\), flat | `0.735232` | `0.809681` | `0.860357` | `0.826536` | `101` | `21` |

因此当前更新后的潜力判断是：

```text
clean V12 在 n=512 planted_parity 上的 best expected ratio 已从 0.816512 推到 0.821887；
clean V12 在 n=1024 planted_parity 上的 best expected ratio 已从 0.697916 推到 0.764526；
n=1024 的 residual max component 最好已经从上百级压到 2；
如果只看 round + local search，n=1024 seed 23 可达到 0.860357；
n=512 的 rounding/local-search 上限仍可接近 0.99。
```

#### 18.13.2 targeted-improve 的机制判断

1. `trust_threshold` 不能太严。

   在 `n=512, seed=17` 上，`threshold=0` 或 `1e-5` 容易回到 `p=0.5` 停滞；`threshold=5e-4` 反而最好。这说明 trust-region 不是越硬越好，它需要允许轻微负 \(J\) 保留相干破对称路径。

2. `trust_shrink` 是 residual 压缩工具，不是所有情况下的 assignment 提升工具。

   对 seed `17`，trust-region 显著提高 expected ratio，并把 residual 压小；但对某些已经 rounding 很强的 seeds，no-shrink 版本的 final assignment 更好。也就是说，trust-region 更适合做 confidence fixing / residual QAOA 前端，不一定总能最大化 direct rounded solution。

3. `n=1024` 上 \(\lambda_J=100\) + flat 是稳定好点。

   对 seeds `17/42/23`，`j_weight=100, round_weight=flat, no shrink` 的表现都较强，best expected ratio 分别约为 `0.748968 / 0.744658 / 0.735232`。这比 8 小时探索阶段的 `0.697916` 有明显提升。

4. `n=1024` 轻量 trust-region 潜力很大，但还不稳定。

   seed `17` 上 `trust_shrink=0.25, threshold=1e-4` 达到 `0.764526`，并且 residual active variables 只剩 `6`、最大分量只剩 `2`；但 seed `42` 同配置只有约 `0.691830`。因此下一步要系统扫 `trust_shrink/threshold`，不能只用一个固定值。

#### 18.13.3 当前最强推荐配置

如果目标是提高 mean-field expected ratio 和压缩 residual：

```text
n=512:
  j_weight = 50
  penalty = relu
  round_weight = flat
  trust_shrink = 0.0
  trust_threshold = 5e-4
  rounds = 360
  epochs = 160

n=1024:
  首选稳健配置:
    j_weight = 100
    penalty = relu
    round_weight = flat
    trust_shrink = 1.0
    rounds = 420
    epochs = 140

  激进 residual 压缩配置:
    j_weight = 50
    penalty = relu
    round_weight = flat
    trust_shrink = 0.25
    trust_threshold = 1e-4
    rounds = 420
    epochs = 140
```

如果目标是最终二值解质量：

```text
不要只看 expected ratio；
需要同时看 rounded ratio、round + local search ratio、sample + local search ratio；
某些 no-shrink 配置虽然 residual 更大，但 local search 后 assignment 更好。
```

#### 18.13.4 下一步真正值得改进的点

下一阶段的改进不应该继续盲目堆轮次，而应该做以下几件事：

1. **confidence calibration**：

   当前 trust-region 能把 residual 压小，但有时会把错误变量也压得很自信。需要加入 fixing threshold sweep，例如 `0.20/0.25/0.30/0.35/0.40`，记录每个 threshold 下的 residual 大小和 repair 后 ratio。

2. **adaptive trust-region**：

   固定 `trust_shrink=0.25` 对 seed `17` 很好，对 seed `42` 不好。下一步应改成按负 \(J\) magnitude 自适应 shrink，而不是统一 shrink。

3. **two-stage V12**：

   第一阶段用 no-shrink / high-\(\lambda_J\) 保持相干破对称，第二阶段再启用温和 trust-region 压 residual。这样可能兼顾 assignment 质量和 residual 压缩。

4. **MaxCut 单独开分支**：

   MaxCut 失败不是简单调参问题，而是破二分对称问题。不要污染 V12 主线；如果要做，应明确作为 `V13 symmetry-breaking SQNN`，例如引入可控的微弱节点级初始扰动或结构特征，而不是继续在 V12 里硬扫。

### 18.14 现实意义路线更新：noisy / weighted signed / MaxCut

用户明确要求：后续模型必须有现实意义，尤其要面向大规模、有真实组合优化价值的问题，并且最终要服务于 QAOA warm-start。因此路线更新为：

```text
V12 主线：
  noisy planted parity
  -> weighted signed graph frustration
  -> large sparse signed QUBO residual compression

V13 分支：
  symmetry-breaking MaxCut
  -> negative-edge-ratio bridge
  -> MaxCut / QAOA warm-start
```

这意味着：

1. clean planted parity 只保留为诊断 benchmark，不再作为最终现实应用；
2. noisy planted parity 用来测试干净结构被破坏后，模型是否还能保留 warm-start / residual compression 能力；
3. weighted signed graph frustration 作为 V12 最重要的现实问题族；
4. MaxCut 必须做，但不能继续用 clean V12，而要单独作为 V13 symmetry-breaking 分支；
5. 用 `negative_ratio` 从 mixed signed graph 逐渐推到 `1.0`，把 signed frustration 和 MaxCut 接起来。

新增 benchmark：

```text
noisy_planted_parity
weighted_signed_frustration
```

对应代码：

```text
quantum/warmstart/benchmarks.py
scripts/run_qubo_warmstart.py
scripts/explore_j_regularized_sqnn.py
```

其中：

```text
noisy_planted_parity:
  先生成 hidden assignment；
  再翻转一部分 same/different 边；
  用于测试抗噪声和 hidden-structure recovery。

weighted_signed_frustration:
  正边希望变量相同；
  负边希望变量不同；
  边权表示约束重要性；
  目标是最大化 satisfied signed-edge weight。

negative_ratio = 1.0:
  所有边都希望不同；
  这就是 signed-edge 形式下的 MaxCut。
```

新增探索模式：

```powershell
.venv\Scripts\python.exe scripts\explore_j_regularized_sqnn.py --realistic-roadmap --device cuda --output-dir outputs\realistic_roadmap_probe
```

这个队列同时覆盖：

1. noisy planted parity；
2. weighted signed frustration；
3. `negative_ratio = 0.5 / 0.7 / 0.9 / 1.0`；
4. V13 random-Z symmetry breaking；
5. adaptive trust-region；
6. two-stage V12；
7. n=512 / n=1024 规模验证。

### 18.15 realistic-roadmap 初步结果

本轮 probe 输出目录：

```text
outputs/realistic_roadmap_probe
```

完成配置数：

```text
34 runs
```

#### 18.15.1 关键结果表

| 问题族 | n | 关键配置 | best expected ratio | round + local search | sample + local search | residual active | max component |
|---|---:|---|---:|---:|---:|---:|---:|
| `planted_maxcut` | 256 | V13 random-Z, strength `0.20` | `0.813585` | `0.990192` | `0.985605` | `5` | `3` |
| `planted_maxcut` | 512 | V13 random-Z, strength `0.20` | `0.756406` | `0.823371` | `0.834898` | `22` | `10` |
| `planted_maxcut` | 1024 | V13 random-Z, strength `0.20` | `0.770585` | `0.827755` | `0.824174` | `6` | `2` |
| `noisy_planted_parity`, noise `0.10` | 512 | two-stage V12, \(\lambda_J=100\) | `0.756189` | `0.819790` | `0.837264` | `23` | `6` |
| `noisy_planted_parity`, noise `0.10` | 1024 | two-stage V12, \(\lambda_J=100\) | `0.773820` | `0.812181` | `0.809950` | `8` | `2` |
| `weighted_signed_frustration`, neg `0.70` | 512 | V13 random-Z, strength `0.12` | `0.782672` | `0.819891` | `0.835408` | `8` | `2` |
| `weighted_signed_frustration`, neg `0.70` | 1024 | V13 random-Z, strength `0.12` | `0.778213` | `0.821614` | `0.822274` | `20` | `4` |
| `weighted_signed_frustration`, neg `1.00` | 512 | V13 random-Z, strength `0.12` | `0.765121` | `0.818686` | `0.828424` | `15` | `5` |
| `weighted_signed_frustration`, neg `1.00` | 1024 | V13 random-Z, strength `0.12` | `0.750179` | `0.818413` | `0.807290` | `37` | `6` |

这里的 ratio 对 `weighted_signed_frustration` 和 `noisy_planted_parity` 是：

```text
satisfied signed-edge weight / total signed-edge weight
```

也就是相对于总边权的满足比例上界，不等同于 frustration optimum 的 approximation ratio。后续如果要和 Aref / Gurobi 精确结果比较，需要把 exact optimum 或 best-known optimum 接入 denominator。

#### 18.15.2 目前最重要的判断

1. **V13 破对称对 MaxCut 是必要的。**

   clean V12 在 MaxCut 上会卡在约 `0.5`。加入 random-Z symmetry breaking + two-stage trust-region 后，`n=256 planted_maxcut` 已经能到：

   ```text
   best expected ratio = 0.813585
   round + local search = 0.990192
   residual active = 5
   max component = 3
   ```

   `n=1024 planted_maxcut` 也能到：

   ```text
   best expected ratio = 0.770585
   round + local search = 0.827755
   residual active = 6
   max component = 2
   ```

2. **weighted signed frustration 大规模下必须用 V13 symmetry breaking。**

   对 `n=512/1024, negative_ratio=0.70`，只用 adaptive trust、不做 symmetry breaking 会回到：

   ```text
   expected ratio ≈ 0.5
   residual active ≈ n
   ```

   加入 V13 random-Z 后，`n=1024, negative_ratio=0.70` 变成：

   ```text
   best expected ratio = 0.778213
   round + local search = 0.821614
   residual active = 20
   max component = 4
   ```

3. **noisy planted parity 是有效桥梁。**

   `noise_rate=0.10` 时，`n=1024` 的 two-stage V12 结果为：

   ```text
   best expected ratio = 0.773820
   round + local search = 0.812181
   residual active = 8
   max component = 2
   ```

   说明从 clean planted parity 走向 noisy constraints 后，模型仍能保留 residual compression 能力。

4. **confidence calibration 已经接入，但还需要强化。**

   目前脚本已经对多个 fixing threshold 做 exact residual completion；当 residual 足够小时，会输出：

   ```text
   best_calibrated_exact_ratio
   best_calibrated_exact_threshold
   best_calibrated_exact_remaining_variables
   ```

   但很多大规模 case residual 仍超过当前 exact 枚举阈值，因此 exact completion 还不是所有 run 都能给出结果。后续要加入 component-wise exact completion，而不是只看总 remaining variable count。

#### 18.15.3 研究路线的阶段性结论

当前更准确的模型定位是：

```text
V12:
  能在 noisy planted parity 上保留大规模 residual compression；
  适合作为 signed sparse QUBO warm-start 前端。

V13:
  在 V12 基础上加入 symmetry breaking；
  是 MaxCut 和高负边比例 signed frustration 的必要分支。

主现实问题:
  weighted signed graph frustration。

量子 QAOA 目标:
  用 SQNN 把 n=512/1024 甚至更大图压成小 residual core；
  再对 residual core 做 component-wise exact / QAOA。
```

下一步优先级：

1. component-wise exact completion / component-wise residual QAOA；
2. V13 symmetry strength sweep for `n=512/1024 MaxCut`；
3. weighted signed frustration 多 seed、多 negative ratio、大规模 sweep；
4. 接入真实 signed-network 数据集或生成 Aref-style benchmark，并用 Gurobi / ILP / local-search baseline 对照；
5. 把 ratio denominator 从 total edge weight 升级为 exact/best-known frustration optimum。

### 18.16 V12/V13 潜力探索：现实任务两小时批处理

本轮探索目标不是继续在 clean planted parity 上刷分，而是检验模型在更有现实意义的稀疏组合优化任务上的潜力：

```text
输出目录:
  outputs/j_regularized_potential_probe_2h

完成 run 数:
  36

自动报告:
  outputs/j_regularized_potential_probe_2h/potential_probe_report.md

图表目录:
  outputs/j_regularized_potential_probe_2h/plots
```

本轮新增了 `random_regular_maxcut` benchmark。`average_degree=3` 时对应 MaxCut-3，也就是 3-正则无权图 MaxCut。需要注意：本轮 `noisy_planted_parity`、`weighted_signed_frustration`、`random_regular_maxcut` 的 ratio 仍然是

```text
satisfied signed-edge weight / total signed-edge weight
或
cut weight / total edge weight
```

因此它是总边权归一化质量，不是相对于 exact optimum 的严格 approximation ratio。后续如果写论文或和 Gurobi/ILP/Aref-style exact baseline 对比，必须把 denominator 升级成 exact/best-known optimum。

后处理算法也已经明确命名：

```text
round + 1-bit greedy QUBO descent
```

含义是：先用 `p_i >= 0.5` 得到二值解；然后每一轮计算所有变量单独翻转的 QUBO 能量增量 \(\Delta E_i\)，选择最负的 \(\Delta E_i\) 翻转；如果所有 \(\Delta E_i >= 0\)，或者达到 pass 上限，就停止。这个不是泛泛的 local search，而是单比特贪心下降。

#### 18.16.1 本轮覆盖的任务强度

```text
V12 noisy planted parity:
  n = 512 / 1024
  noise_rate = 0.00 / 0.05 / 0.10
  plain V12 vs two-stage V12

weighted signed graph frustration:
  n = 512 / 1024
  negative_ratio = 0.30 / 0.50
  V12 adaptive trust-region vs V13 random-Z symmetry breaking

MaxCut-3:
  random 3-regular unweighted graph
  n = 512 / 1024
  V13 random-Z symmetry strength = 0.05 / 0.10 / 0.20 / 0.30
  seed = 17，并补充 seed = 23 的部分点
```

#### 18.16.2 当前最强结果

| 任务 | n | 配置 | best expected | round + 1-bit greedy | sample + 1-bit greedy | residual active | max component |
|---|---:|---|---:|---:|---:|---:|---:|
| MaxCut-3 | 1024 | V13 random-Z, strength `0.20`, seed `17` | `0.835682` | `0.871745` | `0.875651` | `6` | `4` |
| MaxCut-3 | 512 | V13 random-Z, strength `0.05`, seed `23` | `0.806886` | `0.876302` | `0.889323` | `25` | `4` |
| MaxCut-3 | 512 | V13 random-Z, strength `0.30`, seed `17` | `0.829179` | `0.875000` | `0.878906` | `10` | `2` |
| weighted signed frustration, neg `0.30` | 1024 | V12 adaptive trust-region | `0.799594` | `0.827751` | `0.833822` | `13` | `6` |
| weighted signed frustration, neg `0.30` | 1024 | V13 random-Z, strength `0.08` | `0.798465` | `0.833500` | `0.836414` | `12` | `2` |
| weighted signed frustration, neg `0.50` | 1024 | V12 adaptive trust-region | `0.784234` | `0.817966` | `0.820122` | `20` | `3` |
| noisy planted parity, noise `0.10` | 512 | V12 two-stage | `0.755272` | `0.833267` | `0.842291` | `13` | `3` |
| noisy planted parity, noise `0.10` | 1024 | V12 two-stage | `0.745458` | `0.804760` | `0.805727` | `16` | `2` |

#### 18.16.3 关键判断

1. **V13 MaxCut-3 是目前最有希望的 QAOA warm-start 分支。**

   这是本轮最重要的新结果。以前 MaxCut 还只是 `planted_maxcut` 或 signed bridge 的延伸；现在 `random_regular_maxcut, d=3` 直接对应 MaxCut-3。`n=1024` 上 best expected ratio 已经到 `0.835682`，后处理后到 `0.875651`，而 residual active 只有 `6`、最大分量只有 `4`。这说明 V13 不只是破 symmetry，它确实能把随机 3-正则 MaxCut 压成很小的 residual core。

2. **symmetry strength 不是越大越好。**

   `n=1024, seed=17` 的 MaxCut-3：

   ```text
   strength 0.05: expected 0.812494, sample+greedy 0.868490, active 33
   strength 0.10: expected 0.816974, sample+greedy 0.869792, active 42
   strength 0.20: expected 0.835682, sample+greedy 0.875651, active 6
   strength 0.30: expected 0.831641, sample+greedy 0.862630, active 0
   ```

   `0.30` 会把 residual 压到 `0`，但质量下降；这说明过强 symmetry breaking 可能过早锁死次优二值解。当前最值得继续扫的是 `0.15 / 0.18 / 0.20 / 0.22 / 0.25`。

3. **V12 two-stage 是 residual-compression 工具，不一定是最佳二值解工具。**

   在 noisy parity 上，two-stage 会显著压 residual。例如 `n=1024, noise=0.10`：

   ```text
   V12 plain:
     expected 0.684712
     round+greedy 0.816295
     residual active 219
     max component 140

   V12 two-stage:
     expected 0.745458
     round+greedy 0.804760
     residual active 16
     max component 2
   ```

   这说明 two-stage 更适合接 residual QAOA / exact completion；如果目标是直接二值解质量，有时 plain + greedy 反而更高。

4. **weighted signed frustration 在 neg=0.30/0.50 下仍然可做，但需要更强 baseline。**

   `n=1024, neg=0.30` 下，V12 adaptive 和 V13 random-Z 都接近：

   ```text
   V12 adaptive:
     expected 0.799594
     sample+greedy 0.833822
     residual active 13
     max component 6

   V13 random-Z strength 0.08:
     expected 0.798465
     sample+greedy 0.836414
     residual active 12
     max component 2
   ```

   但 `n=512, neg=0.50` 的 V12 adaptive 出现 mean-field 停在 `0.5`、residual 几乎全活跃的失败点，说明 signed frustration 比 MaxCut-3 更依赖任务结构和 symmetry/calibration。后续需要多 seed 和 exact/best-known denominator 才能判断真实竞争力。

#### 18.16.4 下一步改进方向

1. **V13 MaxCut-3 细扫。**
   固定 `n=1024, d=3`，重点扫 `symmetry_strength = 0.15 / 0.18 / 0.20 / 0.22 / 0.25`，并加入 seeds `17/23/42/101/202`。目标不是只追 expected，而是同时看 `sample+greedy`、residual active、max component。

2. **把 residual QAOA 真正接上。**
   当前 `max component <= 4` 的 MaxCut-3 run 已经非常适合做 component-wise exact / small-QAOA。下一步应该把图 3 从 “接单比特贪心” 升级为 “接 component-wise exact / local QAOA”，这样才更贴近最终量子 QAOA 目标。

3. **weighted signed frustration 要引入 exact/best-known baseline。**
   目前 denominator 是 total edge weight。下一步需要至少加入小中规模 exact/ILP 或强 classical heuristic 作为 best-known denominator，否则无法严肃声称 approximation ratio。

4. **noisy parity 保留为诊断桥，而不是最终主应用。**
   它能很好地区分 plain/two-stage 的 residual compression 机制，但现实意义不如 MaxCut-3 和 weighted signed frustration。后续只保留 noise sweep 作为结构破坏诊断。

### 18.17 MaxCut-3 升级：可学习 symmetry strength

用户提出：既然 V13 random-Z symmetry strength 对 MaxCut-3 很关键，是否可以把 strength 直接作为优化变量，而不是手工扫参。

结论：

```text
可以。
```

实现方式已经接入 `scripts/explore_j_regularized_sqnn.py`：

```text
--maxcut3-strength-learn
```

核心参数化为：

```text
strength = strength_max * sigmoid(raw_strength)
```

其中 `raw_strength` 是可训练参数，和 `field_steps / phase_steps / mixer_bias / initial_angles` 一起由 AdamW 更新。这样做的原因是：

1. strength 必须非负；
2. strength 不应该无界增大，否则容易过早锁死二值解；
3. 用 sigmoid 上界以后，仍然保留梯度可训练性。

新增记录字段：

```text
symmetry_strength_trainable
symmetry_strength_max
final_symmetry_strength
```

#### 18.17.1 初步 smoke / paired 结果

输出目录：

```text
outputs/maxcut3_learn_strength_probe
outputs/maxcut3_strength_fixed_pair
```

任务：

```text
random_regular_maxcut
n = 512
d = 3
seed = 17
V13 random-Z
two-stage trust-region
j_weight = 100
rounds = 240
epochs = 90
```

可学习 strength 初步结果：

| init strength | max strength | final strength | best expected | round + 1-bit greedy | sample + 1-bit greedy | residual active | max component |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.05` | `0.30` | `0.042632` | `0.743857` | `0.871094` | `0.876302` | `115` | `17` |
| `0.05` | `0.50` | `0.043806` | `0.500117` | `0.796875` | `0.742188` | `512` | `512` |
| `0.10` | `0.30` | `0.088885` | `0.803732` | `0.854167` | `0.863281` | `31` | `3` |
| `0.10` | `0.50` | `0.086923` | `0.799139` | `0.854167` | `0.865885` | `43` | `7` |

配对固定 strength 对照：

```text
same graph seed = 17
same random-Z seed = 7759
fixed strength = 0.10
```

结果：

| method | strength | best expected | round + 1-bit greedy | sample + 1-bit greedy | residual active | max component |
|---|---:|---:|---:|---:|---:|---:|
| fixed | `0.100000` | `0.547991` | `0.854167` | `0.777344` | `332` | `280` |
| learnable | `0.100000 -> 0.088885` | `0.803732` | `0.854167` | `0.863281` | `31` | `3` |

这个配对实验说明：learnable strength 不只是形式上可训练；在某些 random-Z seed / initial strength 组合下，它能明显修正坏扰动，把 residual 从大核心压到小核心。

#### 18.17.2 当前判断

1. **strength 可以直接作为优化变量。**

   因为 random-Z 向量固定后，初始角度为：

   ```text
   theta_i = theta_base + strength * random_noise_i
   ```

   这对 strength 是可微的，后续 Bloch rotation、expected energy、J penalty 都能把梯度传回 strength。

2. **但 naive joint optimization 不一定自动超过手工 sweep 的全局最优。**

   当前可学习 run 里，strength 往往会变小。例如 `0.10 -> 0.088885`。这对修正坏 seed 有帮助，但它优化的是：

   ```text
   normalized expected energy
   + J penalty
   - entropy term
   ```

   它不是直接优化最终 `sample + 1-bit greedy`，也不是直接优化 residual QAOA 后的结果。因此它可能为了减少 J 负值或保持训练稳定而降低 symmetry strength。

3. **最合理路线不是完全取消 sweep，而是做混合策略。**

   下一步推荐：

   ```text
   outer loop:
     扫少量 init_strength / strength_max / random-Z seed

   inner loop:
     让 strength 可学习
   ```

   也就是把原来的固定 strength sweep 升级成 learnable-strength multi-start，而不是只保留单个 trainable strength。

#### 18.17.3 下一步 MaxCut-3 升级计划

1. 固定 `n=1024, d=3`，跑：

   ```text
   init_strength = 0.10 / 0.15 / 0.20
   strength_max = 0.30 / 0.50
   seed = 17 / 23 / 42 / 101 / 202
   ```

2. 同时记录：

   ```text
   final_symmetry_strength
   best_expected_ratio
   sample + 1-bit greedy
   residual active
   max component
   component-wise exact / QAOA result
   ```

3. 如果 learnable strength 经常向某个区间收敛，例如 `0.08-0.12` 或 `0.18-0.22`，就把这个区间作为 V13 MaxCut-3 的默认初始化区间。

### 18.18 MaxCut-3 residual p=2 QAOA 初步接入结果

用户要求：针对当前 MaxCut-3 表现最好的两组，一个 `n=512`，一个 `n=1024`，在 round 固定变量之后接两层 QAOA，查看最终效果。

本轮计算采用：

```text
固定变量集合:
  |p_i - 0.5| >= threshold

固定变量取值:
  p_i >= 0.5 -> x_i = 1
  p_i <  0.5 -> x_i = 0

QAOA:
  component-wise p=2 QAOA
  每个 residual 连通分量独立优化一套 gamma/beta
  steps = 160
  restarts = 4
```

输出目录：

```text
outputs/maxcut3_residual_p2_qaoa
```

对应脚本：

```text
scripts/run_maxcut3_residual_qaoa_from_exploration.py
```

#### 18.18.1 n=1024 最强 expected run

来源 run：

```text
potential_v13_maxcut3_symmetry_random_regular_maxcut_n1024_d3p0_s17_jw100p0_relu_fc674c86e2
```

原始结果：

```text
rounded ratio          = 0.861328
round + 1-bit greedy   = 0.871745
```

p=2 QAOA threshold sweep：

| threshold | remaining | isolated | active | max comp | p2 QAOA expected | exact residual |
|---:|---:|---:|---:|---:|---:|---:|
| `0.25` | `15` | `9` | `6` | `4` | `0.864790` | `0.865234` |
| `0.30` | `23` | `15` | `8` | `4` | `0.865281` | `0.865885` |
| `0.35` | `31` | `23` | `8` | `4` | `0.866583` | `0.867188` |
| `0.40` | `43` | `27` | `16` | `4` | `0.866863` | `0.867839` |

这里 p2 QAOA 的最好 expected ratio 是：

```text
0.866863
```

但它仍然低于：

```text
round + 1-bit greedy = 0.871745
```

更关键的是，即使 exact residual completion 的上限也只有：

```text
0.867839
```

这说明不是 p=2 QAOA 优化不充分，而是 `round fixing` 本身已经把部分后续应该翻转的变量锁死了。QAOA 只能在固定后的子空间里优化，无法超过这个子空间的 exact 上限。

#### 18.18.2 n=512 最强 binary run

来源 run：

```text
potential_v13_maxcut3_symmetry_random_regular_maxcut_n512_d3p0_s23_jw100p0_relu_762baf65d2
```

原始结果：

```text
rounded ratio          = 0.845052
round + 1-bit greedy   = 0.876302
sample + 1-bit greedy  = 0.889323
```

p=2 QAOA threshold sweep：

| threshold | remaining | isolated | active | max comp | p2 QAOA expected | exact residual |
|---:|---:|---:|---:|---:|---:|---:|
| `0.25` | `44` | `19` | `25` | `4` | `0.859358` | `0.860677` |
| `0.30` | `56` | `24` | `32` | `4` | `0.860038` | `0.863281` |
| `0.35` | `74` | `24` | `50` | `7` | `0.860480` | `0.868490` |
| `0.40` | `97` | `27` | `70` | `19` | `0.860808` | `0.875000` |

这里 p2 QAOA 的最好 expected ratio 是：

```text
0.860808
```

exact residual 上限最高为：

```text
0.875000
```

仍然略低于：

```text
round + 1-bit greedy = 0.876302
```

并明显低于：

```text
sample + 1-bit greedy = 0.889323
```

#### 18.18.3 关键判断

1. **当前 round fixing + p=2 QAOA 没有超过 1-bit greedy。**

   对这两组最强 MaxCut-3 结果，p=2 residual QAOA 的 expected ratio 都低于 `round + 1-bit greedy`。

2. **问题主要出在 fixing，而不是 residual QAOA 本身。**

   `n=1024` 上 exact residual 上限都低于 `round + 1-bit greedy`，这说明固定变量已经锁死了一部分有用翻转。后续 QAOA 再强，也只能在错误固定后的子空间里优化。

3. **QAOA 接法要改。**

   下一步不应该简单地：

   ```text
   high confidence round fixing -> residual QAOA
   ```

   而应该测试：

   ```text
   更保守 fixing threshold，例如 0.45 / 0.48 / 0.49；
   或只固定 greedy 后仍稳定的变量；
   或让 QAOA 接在 residual + selected uncertain shell 上，而不是把所有高置信变量永久锁死。
   ```

4. **这反而是有价值的负结果。**

   它说明 V13 SQNN 当前最强作用是提供很好的二值初始解和 residual 结构诊断；如果要把 QAOA 作为增益模块，必须避免把 QAOA 的可优化空间切得太窄。

### 18.19 当前路线冻结：聚焦 MaxCut / MaxCut-3

用户明确要求：当前阶段先聚焦 MaxCut 问题。`noisy planted parity` 和 `weighted signed graph frustration` 不是放弃，而是先封存，作为后续对照和扩展路线。

#### 18.19.1 主线任务

当前主线改为：

```text
MaxCut
  -> random regular MaxCut-3
  -> large sparse MaxCut warm-start
  -> residual QAOA / local QAOA
```

其中 `MaxCut-3` 指：

```text
random_regular_maxcut
average_degree = 3
unweighted 3-regular graph
objective = cut weight / total edge weight
```

当前最值得主打的结果是：

```text
n = 1024
d = 3
model = V13 random-Z symmetry-breaking J-regularized SQNN
symmetry_strength = 0.20

best expected ratio        = 0.835682
round + 1-bit greedy       = 0.871745
sample + 1-bit greedy      = 0.875651
residual active variables  = 6
max component              = 4
```

另一个二值后处理质量最高的结果是：

```text
n = 512
d = 3
seed = 23
model = V13 random-Z symmetry-breaking J-regularized SQNN
symmetry_strength = 0.05

round + 1-bit greedy       = 0.876302
sample + 1-bit greedy      = 0.889323
residual active variables  = 25
max component              = 4
```

#### 18.19.2 当前主模型

当前主模型命名为：

```text
V13 random-Z symmetry-breaking J-regularized SQNN for MaxCut-3
```

它由以下部分组成：

1. **SQNN message-round dynamics**

   每个变量是一个 Bloch 向量，经过多轮 local-field 驱动的旋转更新。

2. **J-regularized direction constraint**

   每一轮对每个变量计算：

   ```text
   J_i^t = - local_field_i^t * (p_i^{t+1} - p_i^t)
   ```

   用 `ReLU(-J_i^t)` 惩罚负方向，鼓励每轮局部更新方向不要系统性地走反。

3. **two-stage trust-region**

   前半段允许破对称和形成结构；后半段当 `J` 负得超过 threshold 时缩小该变量的 proposal step，避免方向错误扩大。

4. **random-Z symmetry breaking**

   对每个变量加一个固定随机 Z 方向扰动：

   ```text
   theta_i = theta_base + strength * random_noise_i
   ```

   这是 MaxCut-3 必须保留的机制。没有 symmetry breaking 时，MaxCut 容易卡在对称的 `p=0.5` 附近。

5. **后处理**

   当前使用：

   ```text
   round:
     p_i >= 0.5 -> x_i = 1
     p_i <  0.5 -> x_i = 0

   1-bit greedy QUBO descent:
     每轮计算所有单比特翻转的能量增量；
     翻转能量下降最多的变量；
     直到没有单比特翻转能继续降低能量。
   ```

6. **residual QAOA**

   当前已经接入 component-wise `p=2` QAOA，但初步结果显示：如果先做过激 round fixing，QAOA 会被限制在错误固定后的子空间里，无法超过 `round + 1-bit greedy`。因此 QAOA 接法要改成更保守 fixing 或 selected uncertain shell。

#### 18.19.3 当前要优化的变量

当前 MaxCut-3 主线要优化的变量分成三层。

**A. SQNN 内部可训练参数**

```text
field_steps[t]
  每一轮 local field 对 mixer rotation 的步长。

phase_steps[t]
  每一轮 local field 对 phase rotation 的步长。

mixer_bias[t]
  每一轮的全局 mixer 偏置。

initial_angles[3]
  全体变量共享的初始 Bloch 角度。
```

**B. MaxCut symmetry-breaking 参数**

```text
symmetry_strength
  random-Z 破对称强度。
  当前固定扫描中最优区域约在 0.15-0.25，
  n=1024 seed=17 的最好点是 0.20。

raw_symmetry_strength
  可学习版本中实际训练的 unconstrained 参数。

symmetry_strength_max
  可学习 strength 的上界。

symmetry_seed
  random-Z 扰动向量的 seed。
  这不是模型权重，但会显著影响破对称路径，需要 multi-start。
```

当前判断：

```text
固定扫描 strength 更适合找上限；
learnable strength 更适合修正坏配置；
最合理方案是 outer-loop multi-start + inner-loop learnable strength。
```

**C. QAOA / residual 接口变量**

```text
fixing threshold
  当前不能只用 0.25/0.30/0.35/0.40。
  下一步要扫 0.45 / 0.48 / 0.49，
  避免过早锁死后续 QAOA 需要翻转的变量。

QAOA layers
  当前已测 p=2。
  后续要测 p=1/2/3，并比较 expected value 和采样后的二值解。

QAOA initialization
  plus initialization: residual qubit 从 p=0.5 开始；
  SQNN initialization: residual qubit 从 SQNN 概率开始；
  后续还要测试 greedy-shell initialization。

component grouping
  当前使用 component-wise independent QAOA parameters。
  这是 MaxCut-3 residual core 很小、连通块分散时的合理执行方式。
```

#### 18.19.4 暂时封存的路线

以下路线暂时不作为当前主线推进，但保留代码和结果，后续可恢复：

```text
noisy planted parity
```

用途：

```text
作为结构破坏诊断 benchmark；
检验 V12 two-stage 是否还能做 residual compression；
不作为当前论文主应用。
```

```text
weighted signed graph frustration
```

用途：

```text
作为现实 signed-network / frustration-index 扩展方向；
需要 exact/best-known denominator 后才能严肃报告 approximation ratio；
当前先封存，不继续抢占 MaxCut 主线预算。
```

封存不是放弃。当前策略是：

```text
先把 MaxCut-3 主线做深、做强、做清楚；
等 MaxCut/QAOA warm-start 机制稳定后，
再回到 weighted signed frustration 做现实 signed network 扩展。
```

#### 18.19.5 下一轮实验优先级

1. **MaxCut-3 strength fine sweep**

   ```text
   n = 1024
   d = 3
   strength = 0.15 / 0.18 / 0.20 / 0.22 / 0.25
   seed = 17 / 23 / 42 / 101 / 202
   ```

2. **learnable strength multi-start**

   ```text
   init_strength = 0.10 / 0.15 / 0.20 / 0.25
   strength_max = 0.30 / 0.50
   random-Z seed 多起点
   ```

3. **更保守 residual QAOA fixing**

   ```text
   threshold = 0.45 / 0.48 / 0.49
   compare:
     rounded
     round + 1-bit greedy
     p=2/p=3 component-wise QAOA
     exact residual completion
   ```

4. **QAOA 接法改造**

   当前负结果说明：

   ```text
   high-confidence round fixing -> residual QAOA
   ```

   可能太窄。下一步要试：

   ```text
   greedy-stable fixing
   selected uncertain shell QAOA
   residual + boundary variables QAOA
   ```

5. **报告口径**

   当前 MaxCut-3 ratio 是：

   ```text
   cut weight / total edge weight
   ```

   后续如果要写成严格 approximation ratio，需要接入 exact/best-known MaxCut denominator。
