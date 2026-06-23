# SQNN-QAOA MaxCut-3 Progress PPT Outline

本文是给进展汇报 PPT 准备的 Markdown 草稿。目标是把模型从灵感来源、V10 基础框架、实现细节、V10 效果，再到 V14 / Clean-ZEdge 的改进逻辑讲清楚。

建议 PPT 主线：

1. 为什么要做这个模型：从 QAOA 的 cost/mixer 交替动力学得到启发。
2. V10 是最干净的基础版：一个变量一个 SQNN/Bloch 节点，用局部场驱动 RZ/RY。
3. V10 的每轮旋转为什么会让状态朝低能量方向演化。
4. V10 的工程细节：monotone accept、破对称、S1/S2/S3 优化策略。
5. V10 的不足：破对称和边耦合表达不足，概率态和最终二值解之间存在差距。
6. V14 / Clean-ZEdge 如何在 V10 上增强边耦合、相位记忆和后期 collapse。
7. 当前结果：Clean-ZEdge 在 n=512 十个随机 3-正则图上，direct readout 相对 GW expected 有稳定提升。

代码参考：

- V10 基础模型：`quantum/warmstart/qubo_sqnn.py` 中 `QUBOSynchronousLocalFieldSQNN`
- V14 / Clean-ZEdge：`quantum/warmstart/phase_aware_sqnn.py` 中 `PhaseAwareJRegularizedSQNN`
- S1/S2/S3 优化：`scripts/run_v10_maxcut3_report.py` 和 `scripts/compare_v10_step_optimizers.py`
- 经典 baseline 与指标：`classical/README.md`、`docs/metrics_and_ratios.md`
- n=512 十图机制扫描：`outputs/n512_mechanism_scan_combined/report.md`

## Slide 1. 汇报标题

标题建议：

```text
SQNN-QAOA for Random 3-Regular MaxCut
从 V10 同步局域场模型到 V14 / Clean-ZEdge
```

要讲的核心：

- 我们研究的是 random 3-regular graph 上的 MaxCut。
- 当前目标不是把 SQNN 只当作 QAOA warm-start，而是让 SQNN 自己通过 cost/mixer-like 动力学直接产生高质量 Z-basis bitstring。
- 评价主线是和论文口径 GW expected baseline 对比。

## Slide 2. 问题定义：MaxCut-3

任务：

```text
给定无权 3-正则图 G=(V,E)，每个点变量 x_i in {0,1}。
目标是让尽量多的边两端取不同值。
```

割数：

```text
C(x) = sum_(i,j in E) 1[x_i != x_j]
```

对于无权 3-正则图：

```text
W = |E| = 3n/2
```

项目里常用 cut fraction：

```text
C/W
```

如果有严格最优值 `C*`，可以报告近似比：

```text
R = C/C*
```

但在 n=512 这类规模上，严格 `C*` 不一定容易证完，所以当前实验主要和 GW expected baseline 做同图对比。

图表建议：

- 放一张小的 3-正则图示意图，边跨 cut 就计 1。
- 旁边写 `W=3n/2`。

## Slide 3. 经典 baseline：为什么用 GW expected

经典对照：

```text
GW expected = Goemans-Williamson 向量解经过随机超平面 rounding 的期望割数
```

注意事项：

- 这里对标的是 expected hyperplane value，不是“采很多次然后挑最好”。
- 不加 greedy，因为论文口径 GW baseline 通常不包含 1-bit greedy 后处理。
- 我们的 SQNN direct 和 GW expected 对比，是为了看 SQNN 的最终可测二值读出是否达到经典强 baseline。

当前指标命名：

```text
SQNN expected: C[p]，不二值化，直接用概率算 expected cut
SQNN direct: C_d，p_i >= 0.5 的直接二值读出
SQNN directgreedy: C_dg，direct 后做 1-bit greedy
SQNN sample: C_s，从 SQNN Bernoulli 分布采样 K 次取最好，不加 greedy
GW expected: C_GW，经典对照主 baseline
```

汇报重点：

- 主指标优先看 `C_d`，因为它对应 Z-basis deterministic readout。
- `C[p]` 是概率分布本身的质量诊断。
- `C_dg` 是工程后处理上限参考，不能混成纯 SQNN direct。
- `C_s` 看 SQNN 分布里是否存在好 bitstring。

## Slide 4. 灵感来源：QAOA 的 cost/mixer 交替思想

QAOA 的启发：

```text
1. cost 部分：根据目标函数给状态写入相位。
2. mixer 部分：把相位和概率重新混合，让状态继续探索。
3. 多轮交替：状态逐步偏向低 cost / 高 cut 的区域。
```

我们借鉴的是这个动力学结构，而不是完整模拟 `2^n` 维量子态。

SQNN 的改写：

```text
QAOA 全局态:        |psi> in C^(2^n)
SQNN 局部状态:      每个变量一个 Bloch 向量 r_i=(X_i,Y_i,Z_i)
QAOA entangling:   真实 two-qubit 相位门
SQNN 边耦合:       用图上的 local field 和 edge message 做近似传播
```

优点：

- 复杂度从指数级状态向量转成约 `O(n+|E|)` 的局部更新。
- 保留 cost/mixer 交替的物理直觉。
- 最终仍然可以用 Z-basis 读出 bitstring。

## Slide 5. SQNN 节点状态：一个变量一个 Bloch 向量

V10 开始，每个变量 `x_i` 对应一个三维 Bloch 向量：

```text
r_i = (X_i, Y_i, Z_i)
```

读出约定：

```text
p_i = P(x_i=1) = (1 - Z_i) / 2
```

三维含义：

```text
Z_i: 直接决定变量取 0/1 的概率
X_i, Y_i: 隐藏相干 / 相位记忆通道，不直接作为最终 bit 概率
```

默认初态：

```text
|+> 状态
X_i = 1, Y_i = 0, Z_i = 0, p_i = 0.5
```

讲稿重点：

- `Z` 是最终要测量的方向。
- `X/Y` 是模型内部用来暂存相位和传播信息的空间。
- 这让模型既有概率读出，又有类似量子相位的隐藏通道。

## Slide 6. V10 基础框架：同步局域场 SQNN

V10 的核心类：

```text
QUBOSynchronousLocalFieldSQNN
```

每一轮从旧概率 `p^t` 出发，先计算每个节点的 QUBO local field：

```text
F_i = a_i + sum_j b_ij p_j
```

直观含义：

```text
F_i 是 expected energy 对 p_i 的局部梯度。
它告诉我们：如果稍微增大 p_i，能量会往上还是往下。
```

V10 每一轮做两步：

```text
1. RZ: 把 local field 写进 X/Y 相位
2. RY: 把 local field 转成 Z 概率变化
```

更新形式：

```text
r_i^(t+1) = RY(theta_i^t) RZ(phi_i^t) r_i^t
```

图表建议：

- 画一个流程箭头：`p^t -> local field F -> RZ phase -> RY probability -> p^(t+1)`。
- 强调所有节点同步更新：每轮都用旧的 `p^t` 算全部 `F_i`，不是边算边改。

## Slide 7. V10 每一轮到底做什么

第 `t` 轮：

```text
输入: 当前 Bloch 状态 r_i^t 和概率 p_i^t
```

步骤 1：计算局部场

```text
F_i^t = a_i + sum_j b_ij p_j^t
```

步骤 2：RZ 相位旋转

```text
phi_i^t = phase_step[t] * F_i^t
```

作用：

```text
RZ 只旋转 X/Y 平面，不直接改变 Z。
所以 RZ 不会立刻改变 p_i。
```

步骤 3：RY 概率旋转

```text
theta_i^t = mixer_bias[t] - field_step[t] * F_i^t
```

作用：

```text
RY 把 X/Z 平面的方向互相转换，直接改变 Z，因此直接改变 p_i。
```

步骤 4：得到候选状态

```text
r_candidate = RY(theta) RZ(phi) r_old
p_candidate = (1 - Z_candidate) / 2
```

步骤 5：可选 monotone accept

```text
如果 E[p_candidate] <= E[p_old]，接受整轮更新。
否则拒绝，保留旧状态。
```

## Slide 8. 为什么 V10 的 RY 会推动能量下降

核心解释：

```text
F_i 是能量 E[p] 对 p_i 的局部梯度。
如果 F_i > 0，增大 p_i 会让能量变差，所以应该减小 p_i。
如果 F_i < 0，增大 p_i 会让能量变好，所以应该增大 p_i。
```

V10 选择：

```text
theta_i = - field_step * F_i    忽略很小的 mixer_bias 时
```

当 RY 前的有效 `X_i' > 0` 且角度较小时：

```text
Delta p_i 约正比于 theta_i * X_i'
```

所以：

```text
Delta p_i 约正比于 - field_step * F_i * X_i'
```

如果 `field_step > 0` 且 `X_i' > 0`：

```text
F_i * Delta p_i <= 0
```

也就是说，一阶近似下每个节点都沿着降低 expected energy 的方向走。

这就是 V10 会朝低能量态演化的关键。

需要强调的限制：

- 这个解释依赖 `X_i'` 仍在正半轴附近。
- 如果多轮 RZ 之后 `X_i'` 翻到负半轴，那么相同的 RY 角可能反而往错误方向推。
- 这也是后续 V11/V14 要处理相位稳定性和 trust region 的原因。

图表建议：

- 画 Bloch 球或 X/Z 平面：RY 把 `X` 分量折到 `Z`，改变 `p_i`。
- 用一句话标注：`theta = -eta F` 是概率空间里的局部反梯度步。

## Slide 9. RZ 的作用：它不改概率，但会改变后续 RY 的效果

RZ 旋转：

```text
RZ(phi): 在 X/Y 平面旋转
Z 不变，所以 p_i 不变
```

为什么还需要 RZ：

```text
1. 它把 local field 写进隐藏相位通道。
2. 它改变 RY 前的有效 X_i'。
3. 多轮迭代后，相位会影响下一轮概率更新的方向和强度。
```

可以这样讲：

```text
RZ 像是“写入相位记忆”；
RY 像是“把相位/局部场折回到最终可测概率”。
```

V10 的问题：

- 如果 RZ 累积过强，`X_i'` 可能变负。
- 一旦 `X_i' < 0`，`theta=-eta F` 的反梯度解释会失效。
- 旧实验中 n=512 高轮次 V10 出现过 `X_i<0`，说明基础 V10 后期不够稳定。

## Slide 10. 实现细节 1：monotone accept

V10 / V14 都支持 `monotone_accept`。

它不是逐节点接受，而是整张图一起接受或拒绝。

流程：

```text
1. 用当前状态算 E_old = E[p_old]
2. 先提出一整轮候选更新，得到 p_candidate
3. 算 E_candidate = E[p_candidate]
4. 如果 E_candidate <= E_old，接受整轮
5. 否则拒绝整轮，Bloch 状态和概率保持旧值
```

因为 MaxCut 中代码使用：

```text
E = -C
```

所以降低能量等价于提高 expected cut。

它保证的是：

```text
C[p] 不下降
```

它不保证：

```text
C_d 每轮单调
C_s 每轮单调
C_dg 每轮单调
```

讲稿重点：

- accept 保护的是 product probability expected objective。
- 最终 direct readout 仍可能波动，因为二值阈值 `p_i>=0.5` 是非连续操作。
- V14 中如果 proposal 被拒绝，主 Bloch 状态会回到旧状态；当前 clean route 默认不回滚辅助消息，这样下一轮不会完全重复同一个 proposal。

## Slide 11. 实现细节 2：随机破对称

为什么必须破对称：

对无权 3-正则 MaxCut，如果所有节点从完全相同的 `|+>` 开始：

```text
p_i = 0.5
```

每个节点的 local field 都一样，甚至在标准归一化下会变成 0：

```text
F_i = -3 + 2 * 3 * 0.5 = 0
```

结果：

```text
所有节点一直同质，模型不知道哪个点该偏向 0，哪个点该偏向 1。
```

破对称方式：

```text
random_ry
random_rz
random_rz_ry
```

破对称强度的含义：

```text
symmetry_strength 是初始随机旋转角度的幅度，不是 bit 翻转概率。
```

代码里会根据 `symmetry_seed` 给不同节点加小的随机 RY/RZ 角。

直观作用：

- 给每个节点一点不同的初始方向。
- 让 local field 和后续边耦合能发展出非平凡结构。
- 多个 symmetry seed 相当于多次从不同初始扰动出发，最后选验证指标最好的结果。

## Slide 12. 实现细节 3：V10 的三种优化策略 S1/S2/S3

V10 主要训练三组逐轮参数：

```text
field_steps[t]: local field 进入 RY 的步长
phase_steps[t]: local field 进入 RZ 的步长
mixer_bias[t]: 每轮共享的 RY 偏置
```

注意：

```text
这些参数是“每一轮共享”，不是每个节点单独一套。
第 t 轮所有节点共用同一个 field_step[t]、phase_step[t]、mixer_bias[t]。
节点差异来自 F_i 和初始破对称，不来自 per-node 参数。
```

S1：full-gradient

```text
直接优化每一轮的 field_steps[t]、phase_steps[t]、mixer_bias[t]。
参数量约为 3T。
优点：自由度最高。
缺点：容易不平滑，可能对单图/单 seed 过拟合。
```

S2：schedule-gradient

```text
不用每轮一个自由参数，而是用少量 control points 生成平滑 schedule。
再用梯度下降优化这些 control points。
```

优点：

- 参数少。
- 轮次变化更平滑。
- 更适合解释“前期探索、后期收缩”的 schedule。

S3：schedule-CEM

```text
仍然使用低维 schedule 参数，但不用梯度。
每代采样一批候选 schedule，评估 expected cut，选 elite 更新分布均值和方差。
```

优点：

- 对 accept/reject 这种非光滑流程更鲁棒。
- 可以作为梯度优化之外的黑盒搜索对照。

缺点：

- 计算更贵。
- 依赖 population/generation 设置。

## Slide 13. V10 基础效果：有效但不够强

V10 的积极结果：

- 模型结构非常干净：一个变量一个 SQNN/Bloch 节点。
- 每轮更新有清晰的局部反梯度解释。
- `monotone_accept=True` 时，内部 expected energy trace 不会上升。
- 在小规模或带后处理的实验中，可以得到不错的 sample/local-search 结果。

旧实验记录中的典型观察：

```text
n=512 V10 expectation-only sweep:
round 1   expected ratio 约 0.500
round 80  expected ratio 约 0.514
round 100 expected ratio 约 0.563
round 120 expected ratio 约 0.670
round 130 expected ratio 约 0.671
round 200 基本平台化
```

解释：

- V10 的概率态确实在多轮后变好。
- 但只看 expected probability cut，还没有达到 GW baseline 附近。
- 之前 sample+local-search 的高值里，有很大一部分来自后处理，而不是概率分布本身已经足够尖锐。

V10 暴露的问题：

```text
1. MaxCut-3 上破对称能力不足。
2. 只用节点 local field，不足以表达相邻节点应该反相关的边结构。
3. RZ 相位累积可能让 X 翻到负半轴，导致 RY 的反梯度解释失效。
4. direct readout 质量不够稳定。
```

所以 V10 是基础框架，不是最终主力模型。

## Slide 14. 从 V10 到 V14：要补什么能力

V10 缺的不是“能不能算局部梯度”，而是以下能力：

```text
1. 相邻节点反相关关系的显式表达
2. 多轮传播里的边消息记忆
3. 对 RY 更新方向的稳定约束
4. 更强的后期二值化 / collapse 能力
5. 更可靠的破对称和 seed 稳定性
```

V14 的思路：

```text
在 V10 的 RZ/RY 框架上，加入 phase-aware 机制。
不改变最终 Z-basis direct readout 的目标，
但让隐藏相位和边消息更好地服务于最终 bitstring。
```

一句话：

```text
V10 是“节点局部场驱动”；
V14 是“节点局部场 + 边反相关消息 + 相位记忆 + 后期 collapse”。
```

## Slide 15. V14 新增机制 1：short phase memory

V14 不只看当前轮 local field，还维护一个短期记忆：

```text
phase_memory_i^t = decay * phase_memory_i^(t-1) + local_field_i^t
```

当前 clean route 中：

```text
phase_memory_decay = 0.60
```

它影响哪里：

```text
phase_signal = phase_memory
RZ angle = phase_step[t] * phase_signal
```

也就是说，phase memory 主要影响 RZ 相位角，而不是直接进入 loss。

直观含义：

- 当前 local field 可能有噪声或来回振荡。
- 短记忆让 RZ 写入的是最近几轮的局部趋势。
- decay 太大时会把旧方向拖太久，decay 太小时又像没有记忆。
- 十图扫描显示 0.60 比 0.80/0.95 这类长记忆更稳。

## Slide 16. V14 新增机制 2：directed z-edge cavity + gain schedule

这是 V14 / Clean-ZEdge 的关键边耦合机制。

每条无向边 `(i,j)` 被拆成两条有向消息：

```text
i -> j
j -> i
```

每条有向边保存一个 `z_edge_message`。

消息含义：

```text
如果 tail 节点更倾向 x=1，
那么 MaxCut 希望 head 节点更倾向 x=0。

所以 tail -> head 的消息大致是：
建议 head 取 tail 的相反方向。
```

这一个机制里包含两部分：

```text
z-edge cavity message:
  保存和更新“边两端应相反”的有向边消息。

z-message gain schedule:
  控制这个边消息有多强，通常前期弱一点，后期强一点。
```

所以汇报里可以把 `z_message_gain schedule` 合并进 `z-edge cavity message`，不需要单独当成一个新机制讲。

代码中的 bit polarity：

```text
bit_polarity_i = 2 p_i - 1
```

含义：

```text
bit_polarity_i > 0: 节点 i 更倾向 x_i=1
bit_polarity_i < 0: 节点 i 更倾向 x_i=0
```

cavity 的作用：

```text
计算 i -> j 的消息时，不直接把 j -> i 的反向消息算回来。
这样避免一条边上来回回声放大。
```

对应代码逻辑：

```text
cavity_tail_belief = incoming[tail] - reverse_message
raw_message = -tanh(gain * tail_belief)
```

负号来自 MaxCut：

```text
边两端应该相反。
```

边消息是不是每轮重新生成：

```text
不是完全独立地每轮重采样。
第一轮如果没有旧消息，就根据当前 p 初始化。
之后每轮都会用当前概率和上一轮 edge_z_message 生成 raw_message，
再做一个带 decay 的更新：

next_message = decay * old_message + (1 - decay) * raw_message
```

因此它更像一个逐轮更新的边状态 / 短期边记忆，而不是每轮从零开始算一个临时量。

当前 Clean-ZEdge 的 gain schedule：

```text
z_message_gain = 1.8
z_message_gain_final = 2.6
```

直观含义：

```text
前期边消息不要太强，避免过早塌缩；
后期增强 MaxCut 反相关约束，帮助 direct readout 形成清晰 0/1 分区。
```

## Slide 17. V14 新增机制 3：late collapse 影响 RY 角度

z-edge cavity 产生两个节点级信号：

```text
node_suggestion: 邻居消息建议当前节点应该偏向哪里
z_edge_error: node_suggestion - 当前 bit_polarity
```

如果当前节点方向和邻居反相关建议不一致，`z_edge_error` 就会变大。

V14 中这个 relation signal 不是直接改 loss，而是在后期加入 RY 角度：

```text
RY angle = mixer_bias[t]
         - field_step[t] * local_field
         + collapse_step[t] * relation_signal
```

注意：

- 它影响的是 RY 旋转角度。
- RY 会直接改变 Z，因此直接影响最终 `p_i` 和 direct readout。
- 它不是直接改损失函数，也不是单独的 greedy 后处理。

为什么叫 late collapse：

```text
前期先让局部场和相位记忆探索；
后期再把边反相关消息折回 Z 概率，
推动概率向更确定的 0/1 方向 collapse。
```

当前 clean route 中：

```text
collapse_init = 0.06
```

## Slide 18. V14 新增机制 4：trust region + monotone accept 双层稳定

V14 中有两层稳定机制。

第一层是节点级 trust region。

它检查单节点概率变化是否符合局部能量下降方向。

定义直觉：

```text
J_i = - F_i * Delta p_i
```

如果 `J_i` 为正，说明该节点变化有助于降低能量。

如果 `J_i` 明显为负，说明本轮 proposal 可能把该节点推错方向。

trust region 做法：

```text
对坏的节点，不是完全接受 raw proposal，
而是把它向旧状态收缩。
```

这解决的问题：

- RZ/RY 多机制叠加后，角度可能偶尔过强。
- 局部 trust 能避免少数节点的坏更新破坏整轮 direct readout。
- 它是节点级 proposal shrink。

第二层是整图级 monotone accept。

```text
如果 trust 修正后的整轮 proposal 让 E[p] 不变差，就接受；
如果 E[p] 变差，就拒绝这一轮主 Bloch 状态。
```

两者关系：

```text
trust region:
  每个节点单独看，先把坏节点的 proposal 缩小。

monotone accept:
  整张图一起看，最后决定这一轮主状态是否接受。
```

## Slide 19. V14 中明确不作为主线的机制

为了让汇报主线清楚，以下机制不要作为 V14 相比 V10 的主要提升来讲。

1. full-time XY feedback：删除

```text
早期 V14-XY 曾经使用 full-time XY feedback。
十图扫描发现它没有稳定收益，并且会干扰 clean Z-edge collapse。
因此当前主线直接删除，不再作为备用主线。
```

2. node step gate：放弃

```text
node step gate 会让不同节点根据 field/confidence 学自己的 RY 步长缩放。
它会显著增加优化空间和解释复杂度。
当前决定完全放弃，不放进 PPT 主线。
```

3. final rotation：不作为 V14 核心提升

```text
final rotation 是 V14 代码里曾经保留过的末端小旋转备用项，
不是 V10 的基础结构，也不是当前 Clean-ZEdge 的关键收益来源。
之前实验中大 final rotation 效果不好。
当前汇报不把它作为 V14 的主要变化。
```

4. neighbor_xy / edge_cavity_xy / phase_diff：实验性保留，不讲主线

```text
这些机制试图用 XY 隐藏相位表达边相关。
目前结论是容易让 RZ 通道复杂化，不如 clean z-edge collapse 稳。
```

当前推荐模型：

```text
Clean-ZEdge / clean_edgeboost_mem060
```

核心配置：

```text
phase_mode = memory_z_edge_cavity_collapse
phase_memory_decay = 0.60
xy_feedback_init = 0.0
collapse_init = 0.06
z_message_gain = 1.8
z_message_gain_final = 2.6
```

讲稿重点：

```text
V14 汇报不要列一堆备用项。
当前干净主线就是：
短相位记忆 + 有向 z-edge 反相关消息及其 gain schedule + 后期 collapse + trust/accept 稳定机制。
```

## Slide 20. V14 / Clean-ZEdge 的完整每轮流程

每一轮：

```text
1. 从当前概率 p_i 计算 local field F_i。

2. 更新 phase memory：
   memory_i = decay * memory_i + F_i

3. 用当前概率和上一轮 edge message 更新 z-edge cavity：
   每条有向边给目标节点一个“取相反值”的建议。

4. 计算 RZ 角：
   RZ angle = phase_step[t] * phase_memory

5. 做 RZ：
   改变 X/Y 相位，不直接改变 p。

6. 计算 RY 角：
   基础项 = mixer_bias[t] - field_step[t] * local_field
   后期加 collapse 项 = collapse_step[t] * z_edge_relation_signal

7. 做 RY：
   改变 Z，因此改变 p_i。

8. trust region：
   对局部方向明显不好的节点收缩 proposal。

9. monotone accept：
   如果整图 expected energy 不变差，接受主状态；
   否则拒绝主状态。clean route 默认辅助消息不回滚。

10. 最终读出：
   direct: p_i >= 0.5
```

图表建议：

用一张横向流程图：

```text
p -> F -> phase memory -> RZ
       -> z-edge cavity -> late collapse -> RY -> trust -> accept -> readout
```

## Slide 21. V14 / Clean-ZEdge 的效果

n=512，random 3-regular，seeds 0..9。

经典 baseline：

```text
GW expected hyperplane cut
```

当前最佳 clean variant：

```text
edge_boost_mem060_no_xy
```

相对 GW expected 的平均 gap：

```text
C_d direct gap mean       = +0.009560, wins 9/10
C_s sample gap mean       = +0.007216, wins 8/10
C[p] expected gap mean    = -0.008419, wins 2/10
C_dg directgreedy gap     = +0.014508, wins 10/10
```

解释：

- 最重要的是 `C_d` 已经在 9/10 个图上超过 GW expected。
- `C_s` 说明 SQNN 分布中也经常能采到优于 GW expected 的 bitstring。
- `C[p]` 还没有超过 GW expected，说明提升主要来自最终二值读出质量，而不是 product Bernoulli expected cut 已经全面强于 GW。
- `C_dg` 更高，但它包含 greedy 后处理，只能作为辅助参考。

图表建议：

- 放 `outputs/n512_mechanism_scan_combined/full_10seed_mean_gap_to_gw.png`
- 放 `outputs/n512_mechanism_scan_combined/best_vs_baselines_direct_gap.png`

## Slide 22. V10 到 V14 的核心提升逻辑

可以用一张对比表：

| 模块 | V10 | V14 / Clean-ZEdge |
|---|---|---|
| 节点状态 | 每个变量一个 Bloch 向量 | 保留 |
| 主动力 | local field -> RZ/RY | 保留 |
| 相位记忆 | 只用当前 local field | short phase memory |
| 边耦合 | 通过 local field 间接体现 | directed z-edge cavity 显式表达反相关，并用 gain schedule 控制强度 |
| 后期二值化 | 依靠 RY 和阈值 | late collapse 把边消息折回 RY |
| 稳定性 | monotone accept | monotone accept + trust region |
| 破对称 | random RY/RZ | 继续使用，多 seed 更重要 |
| 多 seed 使用方式 | 外层多次训练/选择最好 seed | 可做 multi-head ensemble，但不是当前必须主结构 |
| 主指标 | 结构验证 | direct readout 对比 GW expected |

一句话总结：

```text
V10 证明了 SQNN 可以用局部场驱动概率态下降；
V14 进一步让图边上的“相邻节点应相反”这个 MaxCut 结构显式进入 RY collapse，
因此 direct bitstring 更接近高质量 cut。
```

## Slide 23. 当前结论

目前可以汇报的主要结论：

```text
1. V10 是一个可解释的 SQNN-QAOA 基础框架：
   RZ 写相位，RY 改概率，local field 给出低能量方向。

2. V10 的局限主要在 MaxCut-3 的破对称和边反相关表达：
   只靠节点 local field 不够稳定，后期相位还可能破坏 RY 方向。

3. V14 / Clean-ZEdge 在 V10 上加入短相位记忆、有向 z-edge cavity/gain schedule 和 late collapse：
   把“边两端应相反”的 MaxCut 结构直接作用到 RY 概率更新。

4. 当前 n=512 十图结果显示：
   Clean-ZEdge 的 direct readout 平均超过 GW expected，
   并且 9/10 个随机图取胜。

5. 但 SQNN expected C[p] 仍落后 GW expected：
   下一步要提升概率分布本身的质量，而不只提升阈值读出的 bitstring。
```

## Slide 24. 下一步计划

建议下一步：

```text
1. 继续以 C_d direct readout 作为主指标。
2. 在 n=512 更多随机图上验证 Clean-ZEdge 稳定性。
3. 扩展到 n=1024，检查 direct readout 的尺度稳定性。
4. 研究如何让 C[p] 也接近或超过 GW expected。
5. multi-head / symmetry seed ensemble 只作为稳定性增强方案，不和核心机制混在一起讲。
6. 保持模型简洁：XY feedback、node step gate、final rotation 不进入当前主线。
```

可作为收尾的一句话：

```text
当前模型的关键进展不是“加了更多后处理”，
而是把 MaxCut 的边反相关结构嵌入到 SQNN 的 RY 概率演化里，
使最终可测的 Z-basis direct bitstring 开始稳定接近并超过 GW expected baseline。
```

## 附录 A. 可直接放进 PPT 的模型公式

MaxCut：

```text
C(x) = sum_(i,j in E) 1[x_i != x_j]
E(x) = -C(x)
```

概率读出：

```text
p_i = P(x_i=1) = (1 - Z_i)/2
```

局部场：

```text
F_i = a_i + sum_j b_ij p_j
```

V10 旋转：

```text
phi_i^t = phase_step[t] * F_i^t
theta_i^t = mixer_bias[t] - field_step[t] * F_i^t
r_i^(t+1) = RY(theta_i^t) RZ(phi_i^t) r_i^t
```

V14 phase memory：

```text
m_i^t = decay * m_i^(t-1) + F_i^t
phi_i^t = phase_step[t] * m_i^t
```

V14 late collapse：

```text
theta_i^t = mixer_bias[t]
          - field_step[t] * F_i^t
          + collapse_step[t] * relation_signal_i^t
```

direct readout：

```text
x_i = 1[p_i >= 0.5]
```

## 附录 B. 可以直接放进 PPT 的指标表述

```text
C[p]  : SQNN product Bernoulli expected cut
C_d   : SQNN direct readout cut, x_i = 1[p_i >= 0.5]
C_dg  : SQNN direct + 1-bit greedy cut
C_s   : SQNN Bernoulli samples best-of-K cut
C_GW  : GW expected hyperplane cut
```

推荐对比：

```text
C_d  vs GW expected    主对比
C[p] vs GW expected    概率分布质量诊断
C_s  vs GW sampled     可选采样口径；当前主汇报先不作为 baseline
C_dg vs GW expected    后处理上限参考，不能当纯 SQNN direct
```

## 附录 C. 讲给非代码听众的直观版

V10：

```text
每个变量像一个小指南针。
Z 方向决定最后读出 0 还是 1。
RZ 先把局部场写进指南针的隐藏相位。
RY 再把这个信息折回 Z 方向，让概率往降低能量的方向偏。
```

V14：

```text
V10 只知道每个点自己的局部场。
V14 让每条边也保留一个消息：
如果这边的点更像 1，那另一边应该更像 0。
前期先探索，后期把这些边消息折回 RY，
让最终 Z 读出更符合 MaxCut 的“边两端相反”结构。
```

Clean-ZEdge：

```text
当前主线只保留短相位记忆、Z-edge 反相关消息及其强度 schedule、后期 collapse，
再配合 trust region 和 monotone accept 稳定更新。
XY feedback、node step gate 和 final rotation 不作为当前 V14 的核心变化。
```
