# SQNN-QAOA Direct-Readout Project Plan

本文是当前项目的规范主计划。旧版计划书里大量逐轮日志、路线名和临时实验记录已经不再放在主计划里；历史细节以 Git 历史、独立报告和 `outputs/*/report.md` 为准。

当前项目聚焦：

```text
主问题：random 3-regular MaxCut
当前规模：n=512 为主验证集，n=1024 为规模检查
长期目标：构造可解释、可扩展、物理可实验测量的 SQNN-QAOA 组合优化模型
近期目标：用 Z-basis deterministic direct readout 稳定接近或超过论文口径 GW expected baseline
```

定位说明：

```text
当前主线不是“先生成 warm-start，再把主要优化交给 QAOA/贪心”的流程。
历史代码和目录仍保留 warmstart 命名，但模型定位已经转为：
用 SQNN 自身的 cost/mixer-like 交替动力学直接产生高质量 Z-basis bitstring。
QAOA 提供的是 cost/mixer 交替结构的物理启发，而不是当前主线的后续求解器。
```

代码归属规范：

```text
quantum/
  正规、可复用的量子/SQNN/QAOA 模型与算法。

quantum/warmstart/phase_aware_sqnn.py
  当前 MaxCut 主线 V14 / Clean-ZEdge 使用的 PhaseAwareJRegularizedSQNN
  与 MultiHeadPhaseAwareSQNN。

quantum/warmstart/qubo_sqnn.py
  早期可复用 QUBO warm-start SQNN 家族，包括 V10/V11 同步局域场路线。
  目录名保留历史命名；当前主线不再把 warm-start 作为模型定位。

classical/
  CP-SAT、GW-style、random+greedy、指标计算、对比驱动脚本。
  这里可以调用 quantum/ 里的 SQNN 做对比，但不再定义量子模型类。

scripts/
  临时探索、批量实验、rescore、报告生成入口。
  不再把主线量子模型定义在 scripts/ 里。

历史例外：
  scripts/explore_j_regularized_sqnn.py 仍保留 V12/V13 的
  JRegularizedSyncLocalSQNN，用于复现实验和给旧脚本提供训练工具；
  它不是当前 Clean-ZEdge 主线。若 V13 路线重新成为主线，再迁入 quantum/。
```

---

## 0. 模型发展脉络

如果只数“真正改变结构假设”的代表性模型，当前项目可以按 7 个阶段理解：

```text
Early graph warm-start SQNN:
  QUBOWarmStartSQNN / QUBOInstanceEmbeddingWarmStartSQNN /
  QUBOHybridWarmStartSQNN / QUBOQuantumDataWarmStartSQNN。
  这一阶段主要验证 sparse QUBO + message passing + probability readout
  能否给 QAOA 或经典后处理提供 warm-start。

V10 / 基础版 sync-local:
  QUBOSynchronousLocalFieldSQNN。
  这是后续路线的基础版：一变量一 Bloch 向量，从 |+> 出发；
  每轮用旧 p 计算 QUBO local field，先 RZ 写相位，再 RY 改 Z-basis 概率；
  可用 monotone accept 保证 expected QUBO energy 不上升。

V11 / positive-X safety:
  QUBOPositiveXSynchronousLocalFieldSQNN。
  在 V10 上加入 positive-X phase alignment、角度裁剪、步长/残差 schedule，
  主要解决 RZ 后 X' 符号翻转导致 RY 推动方向不稳定的问题。

V12 / J-regularized:
  scripts/explore_j_regularized_sqnn.py 里的 JRegularizedSyncLocalSQNN。
  加入 J_i = -F_i Delta p_i 方向约束、trust region、round weighting，
  目标是让每轮概率变化更一致地朝降低 QUBO energy / 增大 cut 的方向走。

V13 / MaxCut-3 symmetry route:
  在 V12/J-regularized core 上加入 random RY/RZ symmetry breaking、
  two-stage trust、可选 classical warm-start。
  这一阶段开始集中追 MaxCut-3 direct readout，而不是泛 QUBO warm-start。

V14-XY / phase-aware route:
  PhaseAwareJRegularizedSQNN。
  加入 short/long phase memory、XY feedback、z-edge cavity、late collapse、
  final rotation、MultiHeadPhaseAwareSQNN 等机制。
  旧 V14-XY = full-time XY feedback + z-edge cavity + late collapse，
  现在降级为对照路线。

Clean-ZEdge / 当前主线:
  仍属于 V14 phase-aware 家族，但删除 full-time XY feedback；
  使用 short phase memory + directed z-edge anti-correlation + late collapse；
  代码配置名为 clean_edgeboost_mem060。
  当前主指标是可实验测量的 Z-basis deterministic direct readout C_d。
```

当前可以这样称呼：

```text
基础版：V10 sync-local SQNN。
当前版：Clean-ZEdge，也可写成 V14-clean / Clean-ZEdge。
```

---

## 1. 当前主线

### 1.1 任务选择

当前先聚焦无权随机 3-正则图 MaxCut：

```text
G = (V, E)
n = |V|
degree = 3
W = sum_{(i,j) in E} w_ij
无权 random 3-regular 时，w_ij = 1, W = |E| = 3n/2
```

选择这个任务的原因：

```text
1. 它和 QAOA / Google MaxCut-3 文献直接相关；
2. 有清晰 classical baseline：GW expected、GW sampled、CP-SAT / SDP upper bound；
3. 随机 3-正则图足够标准，方便与主流论文对齐；
4. 它比 planted parity 更接近我们最终想做的 QAOA/SQNN 物理可实现组合优化场景。
```

暂时封存但不删除的方向：

```text
noisy planted parity
weighted signed graph frustration
LDPC syndrome decoding QUBO
random Max-kXOR
```

这些方向后续可以作为现实任务扩展，但当前不和 MaxCut 主线混跑。

### 1.2 当前推荐模型

当前推荐从外部 `sqnnqaoa/` 快照吸收来的 clean route 作为下一轮主线：

```text
name: Clean-ZEdge

核心结构：
  short local-field memory
  random RZ+RY symmetry breaking
  directed z-edge anti-correlation message
  late collapse into RY
  stronger z-edge gain schedule

已删除：
  full-time XY feedback

记录规则：
  Clean-ZEdge 主线不再展示已删除机制的关闭值；
  文档只记录 full-time XY feedback removed。
```

核心判断：

```text
1. 十图扫描已经验证：全时 XY feedback 没有稳定收益，当前主线删除它；
2. short phase memory + directed z-edge anti-correlation + late collapse 更干净；
3. 当前收益主要体现在最终 Z-basis deterministic bitstring readout，而不是 expected probability cut；
4. 当前主目标是提升可直接物理测量的 C_d，而不是从固定分布里寻找更强的后处理读出。
```

外部十图扫描结果：

```text
n = 512, degree = 3, seeds = 0..9
baseline = GW expected hyperplane cut

Clean-ZEdge:
  direct gap mean = +0.009560, wins 9/10
  sample gap mean = +0.007216, wins 8/10
  expected gap    = -0.008419
  directgreedy gap = +0.014508
```

这表示 Clean-ZEdge 的 `C_d` 和 `C_s` 已经能超过论文口径 `GW expected`，但 `C[p]` 还没有超过。当前主评价优先看 `C_d`，因为它对应 Z-basis deterministic readout，最接近可直接实验测量的输出。

### 1.3 旧主线的定位

旧 V14-XY route：

```text
V14-XY = full-time XY feedback + z-edge cavity + late collapse
```

现在降级为对照路线，不再作为默认主线。它仍然有参考价值，特别是：

```text
1. 分析 RZ/XY 相位通道；
2. 作为已降级对照，解释为什么主线删除 full-time XY feedback；
3. 复查 full-time XY feedback 对 direct readout 的影响。
```

---

## 2. 问题定义与能量函数

### 2.1 MaxCut 标准目标

令二值变量：

```text
x_i in {0, 1}
s_i = 1 - 2 x_i in {+1, -1}
```

MaxCut cut value：

```text
C(x) = sum_{(i,j) in E} w_ij * 1[x_i != x_j]
     = 1/2 * sum_{(i,j) in E} w_ij * (1 - s_i s_j)
```

项目中的 QUBO energy 使用：

```text
E_QUBO(x) = -C(x)
```

所以优化 MaxCut 等价于最小化 `E_QUBO`。

### 2.2 概率态的 expected cut

SQNN 输出每个变量的概率：

```text
p_i = P(x_i = 1)
```

若按独立 Bernoulli 分布近似，则边 `(i,j)` 被割开的概率为：

```text
P[x_i != x_j] = p_i (1 - p_j) + (1 - p_i) p_j
              = p_i + p_j - 2 p_i p_j
```

因此：

```text
C[p] = sum_{(i,j) in E} w_ij * (p_i + p_j - 2 p_i p_j)
E[p] = -C[p]
```

训练时的主优化目标仍然是 Z-basis / product-distribution expected MaxCut，而不是直接优化一个外部 teacher 或完整向量损失。
但最终要汇报和推进的主输出不是概率分布本体，而是由这个概率态读出的 deterministic Z-basis bitstring `x_d` 及其 `C_d`。

---

## 3. SQNN 模型表达

### 3.1 Bloch 状态与 Z 基读出

每个 MaxCut 变量对应一个 Bloch 向量：

```text
r_i = (X_i, Y_i, Z_i)
```

最终读出只使用 Z 基：

```text
p_i = P(x_i = 1) = (1 - Z_i) / 2
```

反向初始化约定：

```text
如果外部传入 initial_probabilities，它必须表示 P(x_i = 1)，
因此 Bloch 初始化应使用 Z_i = 1 - 2 p_i。
默认主线不传 initial_probabilities，而是从 |+> 开始：
X_i = 1, Y_i = 0, Z_i = 0, p_i = 0.5。
```

这点必须保持清楚：

```text
Z 决定最终概率；
RZ 主要改变 X/Y 相位，不直接改变 p_i；
RY 会把相位信息折回 Z，从而改变 p_i。
```

### 3.2 一轮 SQNN 更新

每轮更新大致分为：

```text
1. 从当前 p 计算 local field F_i；
2. 用 local-field memory 形成 RZ 相位信号；
3. 用 z-edge cavity message 表达相邻节点反相关；
4. 先做 RZ，相位在 X/Y 平面积累；
5. 再做 RY，把 local field 和 relation_signal 折回 Z；
6. 用 J_i 检查更新方向；
7. 可选 monotone accept / trust region。
```

其中 `z-edge cavity` 的含义：

```text
每条无向边 (i,j) 拆成 i -> j 和 j -> i 两条有向消息；
i -> j 根据 i 当前 Z belief 给 j 一个反向 cut 建议；
更新时尽量避免 j -> i 直接回流，形成 non-backtracking / cavity 效果。
```

### 3.3 J 方向约束

局部场：

```text
F_i = d E[p] / d p_i 的局部近似方向
```

每轮概率变化：

```text
Delta p_i = p_i(new) - p_i(old)
```

定义：

```text
J_i = -F_i * Delta p_i
```

解释：

```text
J_i > 0:
  该变量更新大体朝降低 QUBO energy / 增大 cut 的方向走。

J_i < 0:
  该变量更新在局部场意义下方向可疑。
```

J penalty：

```text
J_penalty = mean_i ReLU(-J_i)
```

其中：

```text
ReLU(a) = max(a, 0)
```

所以 `ReLU(-J_i)` 只惩罚 `J_i < 0` 的部分。

---

## 4. 指标规范

这是后续所有报告最重要的命名规则。

### 4.1 基本量

```text
C_value:
  某个 bitstring 的原始 cut value。

W:
  total edge weight。
  无权 random 3-regular 图中 W = |E| = 3n/2。

C_star:
  真实最优 MaxCut 值。

UB:
  certified upper bound，例如 SDP_UB、CP-SAT upper bound、MILP dual bound。

C_best_known:
  当前已知最好的离散 cut value。
```

### 4.2 C/W

```text
C_over_W = C_value / W
```

含义：

```text
cut fraction，割比例。
它表示总边权里被切掉多少比例。
```

注意：

```text
C/W 不是严格 approximation ratio。
历史代码中很多 `ratio` 字段在 MaxCut-3 上实际都是 C/W。
以后报告必须写成 C_over_W 或 cut fraction。
```

### 4.3 C/C*

```text
C_over_Cstar = C_value / C_star
```

含义：

```text
strict approximation ratio。
```

使用条件：

```text
只有 exact solver 证明 OPTIMAL，或者该图最优值已知时，才能报告 C/C*。
```

如果 CP-SAT 只给出 FEASIBLE，不能把 incumbent 当成 `C_star`。

### 4.4 C/UB

```text
C_over_UB = C_value / UB
```

含义：

```text
相对 certified upper bound 的保守近似比下界。
```

因为：

```text
C_star <= UB
```

所以：

```text
C_value / UB <= C_value / C_star
```

也就是说 `C/UB` 不会虚高，是保守的。

### 4.5 C/C_best_known

```text
C_over_best_known = C_value / C_best_known
```

含义：

```text
相对当前最好已知离散解的分数。
```

注意：

```text
如果 C_best_known 还没有被证明等于 C_star，
则 C/C_best_known 不能叫 strict approximation ratio。
```

### 4.6 GW expected

论文口径 GW baseline 指：

```text
GW expected = E_hyperplane[C_GW]
            = sum_{(i,j) in E} arccos(v_i dot v_j) / pi
```

其中 `v_i` 是 GW / SDP-style vector relaxation 的单位向量。

注意：

```text
GW expected 是随机超平面 rounding 的期望 cut value；
它不是 sampled-best；
它也不是 local-search 后处理结果。
```

### 4.7 推荐报告字段

每个主实验至少保存：

```text
graph_id
n
degree
seed
W
C_star              if exact
UB                  if available
C_best_known

SQNN_expected_C
SQNN_direct_C
SQNN_directgreedy_C
SQNN_sample_C
SQNN_bloch_C        if used

GW_expected_C
GW_sampled_best_C   if used

*_C_over_W
*_C_over_Cstar      only if exact
*_C_over_UB         if UB available
*_C_over_best_known if useful
```

---

## 5. SQNN 读出口径

### 5.1 SQNN expected

```text
C_p = C[p]
```

含义：

```text
概率分布本身的 expected cut；
不产生 bitstring；
不采样；
不接 greedy。
```

这是辅助诊断指标，用来观察 product Bernoulli 分布本身的能量形状。
它不是当前最终优化目标，也不是主评价指标。

### 5.2 SQNN direct

```text
x_d,i = 1[p_i >= 0.5]
C_d   = C(x_d)
```

含义：

```text
最干净的 deterministic Z-basis readout。
当前主模型质量优先看 C_d。
它对应按每个变量的 Z-basis 概率做确定性测量/阈值读出，
是当前最重要的物理可实验输出口径。
```

### 5.3 SQNN directgreedy

```text
x_dg = 1-bit-greedy-local-search(x_d)
C_dg = C(x_dg)
```

1-bit greedy local search：

```text
重复寻找一个单 bit flip；
如果翻转它能提高 cut，就执行当前最优正增益 flip；
直到没有任何单 bit flip 能提高 cut。
```

含义：

```text
C_dg 反映 SQNN 给出的初始点 + 经典局部修复的工程效果。
它不能单独代表 SQNN 概率态本体。
```

### 5.4 SQNN sample

```text
x_s^(k) ~ product_i Bernoulli(p_i), k = 1,...,K
x_s     = argmax_k C(x_s^(k))
C_s     = C(x_s)
```

报告要求：

```text
必须写明 K，例如 C_s(K=256) 或 C_s(K=8192)。
如果 sample 后接 greedy，必须写成 C_sg，不能混进 C_s。
```

### 5.5 Bloch hyperplane readout 已封存

用隐藏 Bloch 向量：

```text
r_i = (X_i, Y_i, Z_i)
```

做超平面舍入：

```text
x_i = 1[r_i dot g >= 0]
```

含义：

```text
这是 SQNN-generated embedding 的 correlated readout。
它与 GW 的“向量 + 超平面”机制同构，但向量来自 SQNN 动力学。
```

注意：

```text
Bloch hyperplane readout 不是纯 Z-basis direct readout；
当前主线暂时封存这个方向。
原因一：它主要是在优化出来的分布/隐藏向量固定后，寻找更好的后处理表达；
       这不是对 SQNN 优化结果本身的探索。
原因二：它的物理实验实现不直接，不能作为近期主线的可测量输出。
历史报告中可以保留 C_bloch 作为旧诊断结果，但新的主实验默认不再要求它。
```

---

## 6. 经典 Baseline 规范

### 6.1 主 baseline

当前主 baseline：

```text
GW expected hyperplane cut
```

实现：

```text
classical/maxcut3_compare.py
```

当前代码使用 low-rank Burer-Monteiro style vector relaxation，因此报告时写：

```text
GW-style expected
```

不要写成 certified GW，除非接入真正 SDP solver 并保存证书。

### 6.2 辅助 baseline

```text
Random + 1-bit greedy:
  用于衡量局部搜索本身强度。

GW sampled-best(K):
  与 SQNN sample(K) 对齐。

CP-SAT:
  用于 exact C_star 或 certified UB。

SDP_UB:
  最理想的上界分母，后续需要接入实例级 SDP solver。
```

### 6.3 Baseline 对齐规则

```text
SQNN C_p:
  可作为辅助诊断对比 GW expected，但它不是当前最终优化目标。
  报告时要说明一个是 product probability expected，一个是 vector hyperplane expected。

SQNN C_d:
  对比 GW expected，这是当前最重要的 deterministic readout 对标。
  它是当前主模型最优先的物理可测量指标。

SQNN C_dg:
  可以对比 GW expected，但必须注明带 1-bit greedy 后处理。

SQNN C_s(K):
  对比 GW sampled-best(K)，K 要相同或明确写出。

Bloch hyperplane:
  当前封存；除非专门复现实验或写历史对照，不作为主实验必报指标。
```

---

## 7. 当前已知结论

### 7.1 Clean-ZEdge

外部 `sqnnqaoa/` 十图扫描显示：

```text
n=512, degree=3, seeds=0..9
baseline = GW expected

Clean-ZEdge:
  C_d gap mean = +0.009560
  C_d wins = 9/10
  C_s gap mean = +0.007216
  C_s wins = 8/10
  C_p gap mean = -0.008419
```

解释：

```text
1. 最终 bitstring 质量很强；
2. 概率分布 expected cut 仍未超过 GW expected；
3. 当前主线接受这一点：C[p] 是辅助诊断，最终目标优先是可测量的 C_d。
```

### 7.2 Bloch hyperplane 封存说明

本机旧 V14 / mix route 的五 seed 结果：

```text
direct+greedy mean C/W        = 0.897135
Bernoulli sample mean C/W     = 0.898958
Bloch hyperplane mean C/W     = 0.902604
```

解释：

```text
1. 这些结果说明旧路线的隐藏 Bloch 向量可用于后处理相关读出；
2. 但它主要回答“固定分布/隐藏向量如何读得更好”，不是当前 SQNN 优化机制本身的问题；
3. 它的物理实现路径不直接，当前不作为主线诊断或近期实验目标。
```

### 7.3 n=1024 轻量检查

旧 V14 route 在 `n=1024, seed=17`：

```text
direct+greedy C/W      = 0.888672
Bernoulli sample C/W   = 0.889323
Bloch hyperplane C/W   = 0.889974
```

解释：

```text
1. n=1024 仍可接近 0.89；
2. 旧报告中的 Bloch hyperplane 提升较小，且该方向当前已封存；
3. n=1024 需要用 Clean-ZEdge 重新跑，不能直接沿用旧 V14-XY 结论。
```

---

## 8. 已封存或降级的路线

以下路线目前不作为主线：

```text
reset route:
  已放弃。reset 后近似比变差，不再推进。

full-vector auxiliary loss:
  直接把 full-vector anti-alignment 放进 loss 会伤害 direct/sample readout。

Bloch hyperplane readout:
  当前封存。它是固定隐藏向量后的后处理读出，不是近期主线优化目标；
  物理实验实现也不够直接。

target relation / target-mix:
  多轮扫描负结果，暂时封存。

full-time XY feedback:
  降级为对照，不作为 Clean-ZEdge 主线。

learned node gate:
  参数数量增加明显，收益不稳定，暂缓。

edge_cavity_xy torque directly added to RZ:
  会破坏 Z-edge collapse，暂时封存。

entropy sharpening:
  对当前主目标帮助有限，暂缓。

longer optimization / very large J weight:
  未带来稳定收益，不作为优先方向。
```

---

## 9. 下一步实验计划

### 9.1 复现规范化 baseline

先安装并验证 classical 工具：

```text
requirements-warmstart.txt 已加入 ortools
```

目标：

```text
1. 运行 CP-SAT / GW-style expected baseline；
2. 生成统一字段：C/W、C/UB、C/Cstar；
3. 确认 n=512 seeds=0..9 与外部快照结果一致。
```

### 9.2 复现 Clean-ZEdge

运行：

```text
classical/n512_10_random_graphs.py
model_config = clean_edgeboost_mem060
seeds = 0..9
baseline = GW expected only
```

必须报告：

```text
C_p
C_d
C_dg
C_s(K)
GW_expected
gap_to_GW_expected
```

### 9.3 Direct-readout 物理口径复查

目标：

```text
检查 Clean-ZEdge 的 Z-basis deterministic direct readout 是否稳定。
把 C_d 作为主指标，C_p / C_s(K) 只作为辅助诊断。
```

必须报告：

```text
C_d
C_p
C_s(K)
GW_expected
gap_to_GW_expected
```

### 9.4 n=1024 scale check

用 Clean-ZEdge 在 n=1024 重新跑：

```text
n = 1024
degree = 3
至少 3 个 seed
```

目标：

```text
1. 检查 C_d 是否仍能接近或超过 GW expected；
2. 检查 direct readout 的尺度稳定性；
3. 记录 residual active variables 和 max component。
```

### 9.5 下一轮模型改进

优先改进方向：

```text
1. Tune Clean-ZEdge:
   short-memory strength
   late-collapse strength
   z-edge gain schedule

2. Instance-adaptive route selector:
   用早期 round features 判断当前图适合哪种 gain/collapse schedule。

3. Edge hidden state:
   把 scalar z-edge message 升级为小共享 edge state，
   但必须保持参数共享，避免节点级参数爆炸。

4. Direct-readout improvement:
   优先提升 Z-basis deterministic C_d。
   C[p] 可以辅助观察分布形状，但不是当前最终优化目标。
   模型改进要服务于最终可测 bitstring，而不是只让 product expected objective 变漂亮。
```

---

## 10. 实验记录规范

每次实验必须保存：

```text
summary.csv
report.md
config.json 或等价配置
关键图 png
```

报告必须写清：

```text
1. 任务：
   random 3-regular MaxCut / n / degree / seed

2. 分母：
   W / C_star / UB / C_best_known

3. 指标：
   C_over_W, C_over_Cstar, C_over_UB, C_over_best_known

4. 读出：
   C_d 为主，C_p / C_dg / C_s(K) 按实验需要作为辅助

5. baseline：
   GW expected, GW sampled-best(K), random + greedy

6. 后处理：
   是否使用 1-bit greedy，greedy passes 多少

7. 采样：
   sample K
```

主计划不再记录所有中间失败 sweep。失败路线只在下面两种情况进入主计划：

```text
1. 它改变了主线判断；
2. 它足够重要，需要防止未来重复浪费时间。
```

其余结果放入独立实验报告或 `outputs/*/report.md`。

---

## 11. Classical SA 与 SA-Guided Escape 记录

详细记录见：

```text
docs/reports/maxcut3_sa_baseline_and_escape.md
```

当前判断：

```text
1. Classical simulated annealing 是有价值的强启发式 baseline。
   当前 CPU 通用 QUBO 实现已能快速跑到 n=4096；
   10000 steps x 16 restarts 在 512/1024/2048/4096 上有明显质量提升。

2. SQNN + SA-guided escape 能在 n=64 seed=0 的 exact C*=86 图上达到 C*，
   但它是 hybrid solver，不属于 pure V10/V14 SQNN 指标。

3. full SA-guided escape 太慢：
   单 trial 约 63s。
   更合理的 fast route 是训练阶段保持 monotone SQNN，
   只在最终 forward 的后期触发 SA escape，并限制最多 1 到 3 次。

4. 当前 fast final-only max-3-kick 配置在同一 n=64 exact 图上仍达到 C*=86，
   单 trial 时间降到约 11s。
```

后续优先加速方向：

```text
1. MaxCut-3 专用 incremental SA：
   维护每个点的 flip gain，翻点后只更新邻居，
   避免当前通用 QUBO SA 每步重新扫全部 flip delta。

2. active-set SA：
   只对低置信变量或局部场矛盾变量退火。

3. cascade escape：
   先用便宜 greedy-guided escape，
   无效时再触发 SA-guided escape。

4. cache escape：
   direct assignment 没有明显变化时不重复跑 SA。
```

### 11.1 n=512 15min 对比结论

详细报告见：

```text
docs/reports/maxcut512_15min_classical_vs_sqnn_sa.md
```

本轮设置：

```text
n = 512, degree = 3, seed = 0, W = 768
classical total budget = 900s
SQNN+SA budget = 900s
```

关键结果：

```text
CP-SAT incumbent       C/W = 0.916667  (C = 704)
CP-SAT upper bound     C/W = 0.934896  (UB = 718)
GW + greedy            C/W = 0.912760  (C = 701)
GW sampled-best        C/W = 0.904948  (C = 695)
SQNN+SA best direct    C/W = 0.897135  (C = 689)
```

---

## 12. Q-Tabu Bloch Anneal Probe

Detailed report:

```text
docs/reports/v14_qtabu_anneal_probe.md
```

Main script:

```text
scripts/run_v14_qtabu_anneal_search.py
```

Purpose:

```text
Use tabu-search ideas as Bloch-dynamics control signals, without using tabu or
classical local search as the final optimizer.
```

Mechanisms added:

```text
1. plateau/fixed trigger;
2. gain-aware and conflict-aware active-set selection;
3. qtabu_random small active set instead of full-node perturbation;
4. short no-return memory after RY kicks;
5. branch lookahead, then V14 continues from the selected Bloch state.
```

n=512, degree=3, seed=0 summary:

```text
base V14                 C_dg = 694, C_d = 688, C_exp = 671.374
previous Bloch scan      C_dg = 699, C_d = 695, C_exp = 682.735
Q-tabu best replay       C_dg = 700, C_d = 697, C_exp = 684.380
```

Best replay output:

```text
outputs/v14_qtabu_700_replay_n512_seed0
```

Current judgement:

```text
Q-tabu Bloch anneal is worth keeping as a V14 escape direction.
It improves the Bloch-side best from 699 to 700 and raises direct readout to 697.
It still does not reach the earlier classical tabu/CP-SAT region near 705.

The best route is not hard deterministic gain flipping.
Conflict-biased randomized active sets + mild no-return memory + branch
lookahead work better.  Further manual scanning is likely inefficient; the next
promising step is to make trigger/node/strength selection trainable or adaptive.
```

---

## 13. Soft Global Bloch Anneal Probe

Detailed report:

```text
docs/reports/v14_soft_global_anneal_probe.md
```

Main script:

```text
scripts/run_v14_soft_global_anneal_search.py
```

Purpose:

```text
Test a cleaner dynamical escape mechanism:
global annealing is applied to all nodes, but conflicted/uncertain nodes receive
larger node-dependent annealing strength rho_i.
```

Key distinction from Q-tabu:

```text
Q-tabu uses branch lookahead and selects a branch.
Soft global anneal uses one continuous Bloch trajectory, with no branch
selection and no classical tabu trajectory.
```

n=512, degree=3, seed=0 summary:

```text
base V14                 C_dg = 694, C_d = 688, C_exp = 671.374
previous Bloch scan      C_dg = 699, C_d = 695, C_exp = 682.735
Q-tabu best replay       C_dg = 700, C_d = 697, C_exp = 684.380
Soft global best replay  C_dg = 702, C_d = 701, C_exp = 687.945
```

Best replay output:

```text
outputs/v14_soft_global_702_replay80_n512_seed0
```

Current judgement:

```text
Soft global Bloch anneal is currently the best Bloch-side escape mechanism.
It is also easier to explain theoretically than Q-tabu because it is a single
continuous dynamical trajectory.

It still does not reach 705.  In 80 fixed-parameter replays, C_dg >= 700
occurred 7/80 times and C_dg >= 702 occurred 2/80 times.  The next direction is
to reduce stochasticity by learning/adapting rho_i, trigger time, and
temperature schedule.
```

结论：

```text
当前加速 SQNN+SA 在 n=512 seed0 上还没有超过强经典。
它超过 random+greedy 和本次 active-SA heuristic，
但低于 GW sampled-best、GW+greedy 和 CP-SAT incumbent。
下一步重点应从“加速”转到“提升 n=512 解质量”。
```

---

## 14. Next Routes For Better Escape Quality

Current speed and quality judgement:

```text
Soft global Bloch anneal is fast enough for n=512 exploration.
The fixed best-parameter replay takes about 1.24s per case on CPU.

The main bottleneck is not speed.  The main bottleneck is solution quality and
reliability: in 80 replays, the best result reached C_dg = 702, but no replay
reached the classical heuristic region near 705.
```

Near-term optimization routes:

```text
1. Adaptive anneal controller
   Replace hand-tuned rho_i weights with an adaptive controller.
   Inputs should include confidence, bad-edge count, flip gain, z-edge conflict,
   and local-field magnitude.  Outputs should control node anneal strength,
   RY perturbation direction, and whether to clear phase/memory.

2. Bad-edge cluster coherent anneal
   Move from independent node perturbations to cluster-level coherent rotations.
   The active cluster should be built from connected bad-edge components or
   high-conflict neighborhoods.  A "bad edge" means an edge whose two endpoint
   bits currently sit on the same side of the cut.  The cluster is therefore a
   local frustrated subgraph, not a list of isolated single-bit mistakes.

   The escape operation should anneal or rotate the whole conflicted component
   coherently, so several coupled bits can cross a basin boundary together.
   This is closer to a Bloch/quantum-style cluster anneal than to flipping one
   node at a time.

3. Gain-guided Bloch field
   Inject MaxCut one-flip gain into the continuous Bloch dynamics instead of
   using it only as a ranking signal.  Positive-gain nodes should receive a
   stronger push toward flipping, cheap negative-gain nodes should be allowed
   to cross barriers, and large negative-gain nodes should be protected.

4. Short non-monotone recovery window  [required for next V14 escape]
   During escape, temporarily relax monotone accept for a small number of
   rounds, typically 8 to 16 rounds.  The model should be allowed to get worse
   briefly after the basin jump, then return to monotone accept after recovery.

   This is required because immediate monotone rollback can erase a useful
   basin jump before the V14 dynamics has enough rounds to reorganize the
   Bloch state.

5. Train-time anneal injection
   This should not mean wasting compute by jumping from the beginning.
   Preferred use: inject rare, late, plateau-triggered perturbations during
   training, or train on states sampled from real plateau regions.  The goal is
   to teach V14 how to recover after an escape event, not to disturb every
   normal descent trajectory.
```

Preferred next experiment:

```text
Keep soft global Bloch anneal as the base escape mechanism because it is fast,
continuous, and theoretically easier to analyze than branch-selection Q-tabu.

The next experiment should compare:
1. fixed hand-tuned rho_i;
2. adaptive controller rho_i;
3. cluster coherent anneal;
4. gain-guided Bloch field;
5. gain-guided + non-monotone recovery window.

Primary target:
n = 512, degree = 3, seed = 0, push best C_dg from 702 toward 705.

Secondary target:
test the best mechanism on multiple 512-node random 3-regular graph seeds to
verify that the gain is not a single-graph artifact.
```

---

## 15. Bad-Edge Cluster Bloch Anneal + Non-Monotone Recovery

Detailed report:

```text
docs/reports/v14_cluster_bloch_anneal_probe.md
```

Main script:

```text
scripts/run_v14_cluster_bloch_anneal_search.py
```

Implemented mechanism:

```text
1. trigger near plateau;
2. build bad-edge clusters from uncut MaxCut edges;
3. cap giant clusters so a single escape does not perturb half the graph;
4. apply an alternating cluster RY field in Bloch space;
5. allow an 8 or 16 round non-monotone recovery window;
6. return to ordinary monotone accept after recovery.
```

n=512, degree=3, seed=0 results:

```text
base V14                 C_dg = 694, C_d = 688, C_exp = 671.374
known random RY          C_dg = 697, C_d = 692, C_exp = 682.140
previous soft global     C_dg = 702, C_d = 701, C_exp = 687.945

cluster direct-basis     C_dg = 701, C_d = 699, C_exp = 683.267
cluster greedy-basis     C_dg = 702, C_d = 696, C_exp = 680.815
greedy-basis replay      C_dg = 702, C_d = 688, C_exp = 587.727
```

Time record:

```text
Typical cluster case time: 1.1s to 1.3s on CPU.
Fastest C_dg >= 700: 0.694s.
Fastest C_dg >= 702: 0.809s.
C_dg >= 705: not reached in the current scans.
```

Current judgement:

```text
The non-monotone recovery window is confirmed useful and should stay.
It prevents immediate monotone rollback after a basin jump.

Bad-edge cluster Bloch anneal is fast and can recover 701 to 702-level C_dg,
but it does not yet beat the previous soft-global best and does not reach 705.

Direct-basis clusters preserve the probability-energy state better.
Greedy-basis clusters can hit 702 faster, but they can damage expected cut
badly, so this is not yet a clean SQNN probability-state improvement.

Decision:
keep the non-monotone recovery window as a core escape component.
Do not treat the current bad-edge cluster Bloch anneal as the main replacement
for soft-global anneal yet.  Its current advantage is helping direct+greedy find
a better basin, while the probability-state quality can degrade, especially
with greedy-basis cluster construction.
```

Next refinement:

```text
1. use expected-edge conflict rather than hard direct bad edges;
2. make recovery bounded instead of unconditional accept-all;
3. adapt cluster strength after several recovery rounds;
4. rollback to the best recovery-window state if expected/direct quality does
   not recover after the jump.
```
