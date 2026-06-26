# Soft QNN for QAOA Warm-Start

> 文件定位：这是项目短摘要/快速阅读版。主计划书、详细实验日志和版本路线请看 `sqnn_qaoa_warmstart_project_plan.md`。这里主要保留概念说明、当前状态和推荐路线，避免每次都翻很长的实验日志。

## 1. 项目定位

本项目专注于：

\[
\boxed{
\text{用 SQNN 为 QAOA、局部搜索和模拟退火生成 warm-start 先验}
}
\]

SQNN 不直接替代组合优化求解器，而是学习一个 problem-aware 的软初始化分布。后续求解器仍负责搜索、约束修复和最终验证。

推荐主线：

\[
\text{problem graph}
\rightarrow
\text{directed message-passing SQNN}
\rightarrow
\text{variable bias / color distribution}
\rightarrow
\text{warm-start QAOA or local search}
\]

## 2. 为什么原始 Feedforward SQNN 不够

当前 SQNN 的基本形式是：

\[
x
\rightarrow
\text{fixed SQNN layers}
\rightarrow
y.
\]

这适合分类任务，但组合优化问题的核心结构是变量之间的约束图。QUBO、Ising、MaxCut、graph coloring 都不是单纯的固定向量分类，而是：

\[
\text{variables}
\leftrightarrow
\text{constraints}
\leftrightarrow
\text{variables}.
\]

因此，warm-start 版本的 SQNN 应该把网络拓扑改成由问题图决定：

1. 每个变量是一个 variable node；
2. 每条非零相互作用或约束形成 directed edges；
3. 多轮消息传递后，每个变量输出一个软决策；
4. 输出结果用于初始化 QAOA、局部搜索或模拟退火。

## 3. 核心结构：Directed Message-Passing SQNN

无向约束图可以拆成两个方向的信息流：

\[
i\rightarrow j,
\quad
j\rightarrow i.
\]

更通用的形式是引入 constraint nodes：

\[
\text{variable}
\rightarrow
\text{constraint}
\rightarrow
\text{variable}.
\]

一轮更新可以写成：

\[
h_a^{(t+1)}
=
F_{V\rightarrow C}
\left(
\{h_i^{(t)}, e_{ia}: i\in N(a)\}
\right),
\]

\[
h_i^{(t+1)}
=
F_{C\rightarrow V}
\left(
h_i^{(t)},
\{h_a^{(t+1)}, e_{ai}: a\in N(i)\}
\right).
\]

这里的 \(h_i\) 和 \(h_a\) 可以是三基 Bloch 特征，也可以是 SQNN neuron 的 soft measurement 输出。

关键是参数共享：

\[
U_{ij}=U_\theta(e_{ij}),
\]

而不是给每条边单独训练一组参数。这样模型才能泛化到不同规模的问题图。

## 4. 三基 Bloch 特征的用法

多基读出给出：

\[
[P_X, P_Y, P_Z].
\]

它更适合作为三维量子特征 embedding，而不是直接当作三个互斥类别概率。原因是：

\[
P_X + P_Y + P_Z
\neq
1
\]

一般不成立。

### 4.1 二值变量

对于 QUBO / Ising / MaxCut，最终需要的是：

\[
p_i=P(x_i=1).
\]

可以使用：

\[
[P_X,P_Y,P_Z]
\rightarrow
\text{readout head}
\rightarrow
p_i.
\]

### 4.2 三染色

对于三染色，真正需要的是：

\[
p_{v,c},
\quad
c\in\{1,2,3\},
\quad
\sum_c p_{v,c}=1.
\]

推荐做法：

\[
[P_X,P_Y,P_Z]_v
\rightarrow
\text{Linear or MLP}
\rightarrow
\text{softmax}
\rightarrow
[p_{v,1},p_{v,2},p_{v,3}].
\]

也可以用 Bloch 球上的三个颜色原型方向：

\[
n_1,n_2,n_3
\]

并令：

\[
p_{v,c}
=
\operatorname{softmax}
\left(
\beta\, r_v\cdot n_c
\right).
\]

其中 \(r_v\) 是顶点的 Bloch 特征，\(n_c\) 是颜色原型。

## 5. QUBO Warm-Start

QUBO 目标：

\[
E(x)
=
\sum_i Q_{ii}x_i
+
\sum_{i<j}Q_{ij}x_ix_j,
\quad
x_i\in\{0,1\}.
\]

SQNN 输出：

\[
p_i=P(x_i=1).
\]

warm-start 用法：

1. 从 \(p_i\) 采样初始解；
2. 用 \(p_i\) 初始化模拟退火或局部搜索；
3. 用 \(p_i\) 构造 QAOA 初态。

QAOA 初态：

\[
|\psi_0\rangle
=
\bigotimes_i
\left(
\sqrt{1-p_i}|0\rangle+
\sqrt{p_i}|1\rangle
\right).
\]

对应的 \(R_y\) 制备角：

\[
\theta_i
=
2\arcsin\sqrt{p_i}.
\]

## 6. QUBO 训练损失

最小可行训练目标是直接优化期望 QUBO 能量：

\[
\mathcal L_E
=
\sum_i Q_{ii}p_i
+
\sum_{i<j}Q_{ij}p_ip_j.
\]

为了避免训练早期过快坍缩到 0 或 1，可以加入 entropy：

\[
\mathcal L
=
\mathcal L_E
-
\tau
\sum_i H(p_i).
\]

其中：

\[
H(p_i)
=
-
p_i\log p_i
-
(1-p_i)\log(1-p_i).
\]

训练策略：

1. 前期较大的 \(\tau\)，鼓励探索；
2. 后期逐渐降低 \(\tau\)，让输出变尖锐；
3. 如果有经典求解器生成的高质量解，可以加入 BCE 监督项。

可选总损失：

\[
\mathcal L
=
\mathcal L_E
+
\lambda_{\text{sup}}\mathcal L_{\text{BCE}}
-
\tau\mathcal H.
\]

## 7. Graph Coloring Warm-Start

输入：

\[
G=(V,E),
\quad
k=3.
\]

SQNN 输出：

\[
p_{v,c}=P(\text{vertex }v\text{ has color }c).
\]

冲突损失：

\[
\mathcal L_{\text{conflict}}
=
\sum_{(u,v)\in E}
\sum_{c=1}^{3}
p_{u,c}p_{v,c}.
\]

这个损失表示相邻顶点被分到同一颜色的概率。

还可以加入 one-hot sharpness 项：

\[
\mathcal L
=
\mathcal L_{\text{conflict}}
-
\tau
\sum_v
H(p_v).
\]

warm-start 用法：

1. 按 \(p_{v,c}\) 采样初始染色；
2. 接 greedy repair 或 local search；
3. 或构造 one-hot-preserving mixer 的 QAOA 初态。

评估指标：

1. initial conflict edges；
2. repair steps；
3. final success rate；
4. iterations to feasible coloring。

## 8. QAOA Warm-Start 接口

### 8.1 二值 QAOA

SQNN 输出：

\[
p_i\in(0,1).
\]

初始化角：

\[
\theta_i
=
2\arcsin\sqrt{p_i}.
\]

初态：

\[
|\psi_0\rangle
=
\bigotimes_i R_y(\theta_i)|0\rangle.
\]

然后运行标准 QAOA：

\[
|\psi(\gamma,\beta)\rangle
=
\prod_{\ell=1}^{p}
U_M(\beta_\ell)
U_C(\gamma_\ell)
|\psi_0\rangle.
\]

### 8.2 三染色 QAOA

三染色通常需要 one-hot 编码：

\[
x_{v,c}\in\{0,1\},
\quad
\sum_c x_{v,c}=1.
\]

SQNN 输出：

\[
p_{v,c}.
\]

可以用它初始化每个顶点的颜色 superposition：

\[
|\psi_v\rangle
=
\sum_{c=1}^3
\sqrt{p_{v,c}}
|c\rangle.
\]

后续 mixer 应该保持 one-hot 子空间，避免 QAOA 在无效颜色编码上浪费搜索。

## 9. 最小可行实验路线

### Version 1：QUBO Warm-Start

1. 生成小规模 QUBO 数据；
2. 把非零 \(Q_{ij}\) 转成 directed edges；
3. 每个变量维护三基 SQNN state；
4. 做 2 到 4 轮 message passing；
5. 输出 \(p_i\)；
6. 比较 random init、classical heuristic init、SQNN warm-start；
7. 指标：initial energy、post-search energy、iterations to target。

### Version 2：MaxCut Warm-Start

1. 输入图 \(G=(V,E)\)；
2. 训练 SQNN 输出每个点分到一侧的概率 \(p_i\)；
3. loss 使用 expected cut value 的负数；
4. 用 \(p_i\) 初始化 QAOA；
5. 比较 uniform QAOA 和 SQNN warm-start QAOA。

### Version 3：Graph Coloring Warm-Start

1. 输入图；
2. SQNN 输出 \(p_{v,c}\)；
3. 采样初始 coloring；
4. 接 repair algorithm；
5. 评估冲突边数量、修复步数和成功率。

## 10. 建议的代码结构

建议新增：

```text
quantum/warmstart/
  __init__.py
  qubo_sqnn.py
  coloring_sqnn.py
  losses.py
  sampling.py
  qaoa_init.py
```

其中：

1. `qubo_sqnn.py`：QUBO / Ising / MaxCut 的 directed message-passing SQNN；
2. `coloring_sqnn.py`：三染色的 vertex color distribution 模型；
3. `losses.py`：expected QUBO energy、conflict loss、entropy；
4. `sampling.py`：从 \(p_i\) 或 \(p_{v,c}\) 采样初始解；
5. `qaoa_init.py`：把 SQNN 输出转成 QAOA 初态参数。

### 10.1 QUBO 建模接口

当前 QUBO 使用稀疏图表示：

```python
from quantum.warmstart import (
    QUBOProblem,
    QUBOWarmStartSQNN,
    entropy_regularized_qubo_loss,
    qaoa_ry_angles_from_probabilities,
)

problem = QUBOProblem.from_terms(
    num_variables=4,
    linear=[1.0, -0.5, 0.2, 0.0],
    edge_index=[[0, 1, 2], [1, 2, 3]],
    edge_weight=[-1.0, 0.7, -0.3],
)

model = QUBOWarmStartSQNN(message_rounds=3)
p = model(problem)
loss = entropy_regularized_qubo_loss(problem, p, entropy_weight=0.01)
theta = qaoa_ry_angles_from_probabilities(p)
```

这里的 \(p_i\) 是 SQNN 为每个 QUBO 变量输出的 warm-start bias，\(\theta_i\) 是对应的 QAOA 初态 \(R_y\) 角度。

## 11. 项目推荐名称

英文：

\[
\boxed{
\text{Dependency-Aware Multi-Basis SQNN for QAOA Warm-Start}
}
\]

中文：

\[
\boxed{
\text{面向 QAOA 预热的依赖感知多基读出软量子神经网络}
}
\]

## 12. 最终路线

\[
\boxed{
\text{Directed message-passing multi-basis SQNN}
+
\text{QAOA warm-start}
}
\]

优先从 QUBO / MaxCut 做最小实验，因为它们最容易把 SQNN 输出的 \(p_i\) 转成 QAOA 初态。三染色可以作为第二阶段，用三基 Bloch 特征作为颜色分布的 embedding，再通过 softmax 读出 \(p_{v,c}\)。

## 13. 当前实现状态

当前项目已经从概念设计推进到可运行的大规模 QUBO warm-start 原型。

已实现：

1. 稀疏 QUBO 建模：`quantum/warmstart/qubo.py`。
2. QUBO-SQNN 模型族：node-only、directed、symmetric、instance、hybrid、quantum-data、mean-field baseline。
3. 大规模 MaxCut benchmark：planted bipartite MaxCut 有已知最优值，可严格计算近似比。
4. 大规模 planted parity QUBO benchmark：同时包含相同/不同二次约束，有已知最优值，可严格计算近似比。
5. baseline：random best、random + greedy local search、mean-field。
6. QAOA 限制评估：3060 上 full-state 约 29 qubits；1000 变量完整 QAOA 不现实。
7. residual QAOA demo：先用 SQNN/repair/fix 把 1000 变量 QUBO 压到 18-25 变量，再运行小 statevector QAOA。
8. V6 repair-calibrated probabilities：保留 SQNN confidence，使用 repair assignment 修正概率方向。
9. V7 active-residual preprocessing：消去 residual 中的孤立变量，只把有二次相互作用的 active core 交给 QAOA。
10. V8 component-wise residual QAOA：把 active core 按连通分量拆分，最大 statevector 规模由最大连通分量决定。
11. V9 quantum-data SQNN：去掉 MLP/sigmoid 节点初始化、MLP 边编码、per-node output logits 和 sigmoid 组合读出，改用 3 维 QUBO 节点量、角度编码、Bloch 旋转和 Z 基测量。
12. V10 synchronous local-field SQNN：已实现初版，命令行模型名 `--model sync_local`；一个变量一个三基 SQNN 神经元，\(P_Z\) 读出 \(p_i=P(x_i=1)\)，\(P_X/P_Y\) 作为隐藏相干/相位记忆；每轮用旧状态同步计算 \(F_i=a_i+\sum_j b_{ij}p_j\)，再同步更新所有节点。

最新关键结果：

1. `planted_bipartite_maxcut_n1000_d8.0` 上，hybrid SQNN raw sampled ratio 可达到约 0.79 到 1.0，随 seed 波动。
2. hybrid SQNN + local repair 在已测 1000 变量 planted 实例上达到 1.0。
3. repair+confidence fixing 可把 1000 变量 residual 压到 0-29 变量范围，进入小 QAOA 可运行区间。
4. seed=2 难例中，raw SQNN sampled ratio 为 0.788665，repair-calibrated sampled ratio 提升到 0.994369。
5. planted parity QUBO n=1000 上，raw SQNN sampled ratio 为 0.739505，repair-calibrated sampled ratio 为 0.994300，repair 后达到严格 ratio 1.0。
6. quantum-data SQNN quick test：n=64 planted parity 上 sample+local-search 到 1.0；n=256 上 raw sample ratio 约 0.588，sample+local-search 约 0.936，repair-calibrated sample+local-search 到 1.0，但 residual 压缩能力弱于 hybrid。

当前推荐主线：

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
\text{residual QAOA}
\]

当前判断：`hybrid SQNN` 仍是效果主线；`quantum_data` 是更接近量子数据表示的结构化路线，适合做可解释性和物理约束消融，但还需要增强破对称能力。

最新建模共识：已实现初版 `synchronous local-field SQNN`。它不再把节点做成 GNN 特征槽；节点只对应变量状态，\(a_i\) 作为 local field，\(b_{ij}\) 作为无向边的 soft influence。更新必须同步：

\[
p_i^t=P_Z(h_i^t),
\quad
F_i^t=a_i+\sum_j b_{ij}p_j^t,
\quad
h_i^{t+1}=\operatorname{SQNNUpdate}(h_i^t,F_i^t).
\]

\(P_X/P_Y\) 用来描述隐藏相干、相位记忆和可动性；节点是否主要受边影响用 \(G_i=\sum_jb_{ij}p_j\) 或 edge dominance ratio 单独记录。

V10 初步结果：

1. n=64 planted parity：raw sampled ratio 0.714460，sample+local-search 0.811471。
2. n=256 planted parity：raw sampled ratio 0.634481，sample+local-search 0.990145，repair-calibrated sample+local-search 0.992586。
3. n=256 planted MaxCut：raw sampled ratio 0.539592，sample+local-search 0.889109。
4. V10 的 `monotone_accept=True` 可以保证内部 mean-field 期望能量 trace 不上升；它不保证采样解或近似比逐轮单调改善。
5. 当前判断：V10 结构最干净，但效果仍弱于 hybrid，尤其 MaxCut 上破对称能力和 residual 压缩能力不足。

V10 规模 sweep 已生成到：

```text
outputs/sync_local_v10_evaluation
```

本次 sweep 使用 128/256/512/1024 变量 planted parity QUBO，以及 1/2/4/8 轮 SQNN 预热。最佳 sample+local-search ratio 分别为：

| n | best rounds | sample+LS ratio |
|---:|---:|---:|
| 128 | 4 | 0.990983 |
| 256 | 8 | 0.989430 |
| 512 | 1 | 0.830260 |
| 1024 | 8 | 0.819265 |

图包括：`ratio_vs_warmstart_rounds.png`、`ratio_vs_num_variables.png`、`residual_active_vs_num_variables.png`、`qaoa_gate_estimate_vs_variables.png`、`training_time_vs_rounds.png`。

`qaoa_gate_estimate_vs_variables.png` 已更新为 full QAOA 与 SQNN warm-start residual QAOA 的门数对比；同目录保留副本 `qaoa_gate_estimate_full_vs_sqnn_residual.png`。

n=512 的 rounds=1..100 prefix sweep 已生成到：

```text
outputs/sync_local_v10_n512_rounds_1_100
```

关键结果：

| 指标 | 最佳 round | 数值 | active residual t=0.25 |
|---|---:|---:|---:|
| raw sampled ratio | 97 | 0.717324 | 157 |
| sample+local-search ratio | 74 | 0.942139 | 378 |
| repair-calibrated sample+local-search ratio | 40 | 0.983421 | 499 |
| 最小 active residual | 95 | active=157 | 157 |

结论：rounds 增加后 raw ratio 和置信度后期会上升，active residual 会下降；但最佳近似比和最小 residual 出现在不同轮次，所以实际接 QAOA 时要按目标选择截断点。

最新改进：25 变量、4 边的 residual 在朴素 statevector QAOA 中较慢；加入孤立变量精确消元后，同一 residual 的 active QAOA core 只有 8 个变量、4 条边，可以快速运行 p=1/p=2 QAOA。

进一步改进：8 变量 active core 还可以按连通分量拆成最大 2 变量的 component-wise QAOA。这个版本使用分量独立参数，适合工程 hybrid solver；需要和标准全图共享参数 QAOA 区分。

## 14. 适合优先做的 QUBO

当前路线最适合：

\[
\text{大规模、稀疏、图结构明显、local repair 有效、固定后 residual 会变小的 QUBO}
\]

优先级最高的是：

1. 稀疏 MaxCut / 图划分 / 网络切分；
2. 稀疏 parity / XOR / Ising 约束 QUBO；
3. 稀疏调度 / 冲突消解；
4. 稀疏资源分配 / 设施选择；
5. 稀疏约束满足问题转 QUBO。

暂不优先：

1. 密集 QUBO；
2. 朴素 TSP / VRP one-hot QUBO；
3. penalty 权重极端失衡的问题；
4. 固定后仍保留大连通 residual core 的图。

一句话：原图可以很大，但 high-confidence fixing 后的 residual graph 必须小、稀疏、可拆成小连通分量。
n=512 expectation-only rounds=1..200 sweep 已生成到：

```text
outputs/sync_local_v10_n512_expected_rounds_1_200
```

这次不使用 Bernoulli sampling，不使用 best-of-N sample，也不把 sample+local-search 当主指标。主指标直接计算：

\[
E[p]=c+\sum_i a_i p_i+\sum_{(i,j)}b_{ij}p_i p_j,\quad
\text{expected ratio}=-E[p]/\text{known optimum}.
\]

关键结果：

| round | expected ratio | active residual t=0.25 |
|---:|---:|---:|
| 1 | 0.500000 | 499 |
| 80 | 0.514222 | 495 |
| 100 | 0.563289 | 331 |
| 120 | 0.670094 | 143 |
| 130 | 0.670853 | 138 |
| 200 | 0.670853 | 138 |

最佳 expected ratio 在 round 130，约 0.670853；最小 active residual 在 round 126，为 132 个变量。round 140 到 200 基本平台化。当前判断：V10 的概率态在 100 轮后确实变好，但只看期望值还没有到 0.9+，之前高 ratio 很大一部分来自 sampling/local-search 后处理。以后汇报 V10 时，主指标应优先用 expected ratio，residual QAOA 再用 deterministic confidence fixing 统计。

Bloch-X 正半轴检查已生成到：

```text
outputs/sync_local_v10_n512_bloch_x_trace_0_150
```

结论：当前 V10 在 round 102 首次出现 \(X_i<0\)，最多有 146/512 个变量的 \(X_i<0\)，最小值出现在 round 115，约 -0.990650。按“\(X>0\) 才保证迭代方向正确”的判据，当前高轮次 V10 已经违反结构前提；下一版需要加入 \(X\ge 0\) 约束或 proposal 拒绝/投影机制。

V11 positive-X constrained SQNN 已写入计划书并完成代码实现，但暂时未运行实验。核心更新：

1. 每轮开始根据上一轮状态做 phase alignment：
\[
(X,Y,Z)\rightarrow(\sqrt{X^2+Y^2},0,Z)
\]
等价于根据上一轮测得的 \(X,Y\) 做 \(R_Z(-\operatorname{atan2}(Y,X))\)，不改变 \(Z\) 和 \(p_i\)。
2. 硬件上把 phase reset 和 QUBO 相位写入合并：
\[
R_Z(\phi)R_Z(\delta)=R_Z(\delta+\phi)
\]
所以每轮仍可理解为一个合并 RZ 加一个 RY。
3. \(\eta,\rho,\alpha\) 使用随轮次衰减的 schedule；\(\beta\) 限制在很小范围。
4. RZ 裁剪保证 \(X'\ge0\)，RY 做状态相关 shrink，若会导致 \(X_{\text{out}}\le\epsilon\)，则缩小 \(\theta\)，必要时令 \(\theta=0\)。
5. 新模型类：`QUBOPositiveXSynchronousLocalFieldSQNN`；命令行模型名：`sync_local_xpos`。

新的实验实现分叉已写入计划书：V12 分成两条路线。

1. measurement-assisted growing circuit：每轮都把当前电路 \(C_t\) 临时接测量模块，用大量 shots/tomography 估计 \(X,Y,Z\)，经典算出下一层 \(U_t\)；然后把测量模块拿掉，把 \(U_t\) 接到旧电路后面得到 \(C_{t+1}=U_tC_t\)。优点是最终得到逐层增长的 coherent warm-start 电路；缺点是电路深度随轮数累积，且每轮都要大量测量来设计下一层。
2. measure-feedback reprepare：每轮测量/估计 \(X,Y,Z\)，经典计算 \(F,\delta,\phi,\theta\)，下一轮重新制备携带上一轮信息的 Bloch 初态。优点是每轮浅电路、适合 phase reset 和大规模迭代；缺点是 shots/tomography 成本高，并且保留的是 Bloch 级别信息而不是完整相干历史。

当前优先级：先做 measure-feedback reprepare，同时保留 measurement-assisted growing circuit 作为小规模 coherent warm-start ansatz 对照。
