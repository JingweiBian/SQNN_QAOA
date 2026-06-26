# V14-UTC: Transition-Conditioned Bloch SQNN

本文档把当前正式算法固定为：

```text
V14 clean_edgeboost_mem060 + UTC-SM-lite v3
```

论文中可以称为：

```text
V14-UTC: Transition-Conditioned Bloch SQNN
```

其中 V14 是基础量子启发 Bloch 动力学模型，UTC-SM-lite v3 是推理阶段的统一相变条件跳盆机制。

## 1. 方法定位

这个方法不是量子线路模拟器，也不是 QAOA statevector。它是一个量子启发的连续态动力学优化器：

- 每个组合变量对应一个 Bloch 向量 `b_i=(X_i,Y_i,Z_i)`。
- `X/Y` 表示相干/相位/隐状态，`Z` 决定二值读出概率。
- MaxCut 目标通过 QUBO 期望能量 `E[p]` 作用在连续概率态 `p` 上。
- 模型通过一组可训练的旋转步长、相位记忆和图消息，在 Bloch 球中迭代演化。
- 最终通过 Z-basis hard readout 得到二值解，再用 direct+greedy 作为报告读出。

本文方法的核心创新不是“枚举很多经典局部搜索路径”，而是：

```text
连续 Bloch 动力学
-> hard readout 相变窗口识别
-> 相变前短窗口 soft-monotone 动力学跳盆
-> 少量候选路径选择
```

## 2. 输入输出

输入：

- 图 `G=(V,E)`，当前实验主任务是 3-regular MaxCut。
- 节点数 `n=512` 为主要诊断规模。
- 图 seed 用于生成随机 3-regular 图。

输出：

- 连续概率态 `p_i=P(x_i=1)`。
- direct readout：`x_i = 1[p_i >= 0.5]`。
- direct+greedy readout：在 direct readout 基础上做 greedy descent，作为主要报告分数。
- 轨迹诊断：`C[p]`、direct cut、direct+greedy cut、Bloch trace、phase trace、J trace。

## 3. Bloch 概率约定

代码中的概率和 Bloch-Z 约定是：

```text
p_i = P(x_i=1) = (1 - Z_i) / 2
Z_i = 1 - 2 p_i
```

所以：

- `Z_i > 0` 倾向 `x_i=0`；
- `Z_i < 0` 倾向 `x_i=1`；
- `Z_i ~= 0` 对应不确定态 `p_i ~= 0.5`；
- 初始无 warm start 时，`b_i=(1,0,0)`，即所有节点在 `|+>` 类似的横向态。

## 4. 基础 V14 模型

正式基础模型使用 `clean_edgeboost_mem060` 配置。

核心配置：

```text
rounds = 280
epochs = 110
phase_mode = memory_z_edge_cavity_collapse
phase_memory_decay = 0.60
xy_feedback_init = 0.0
collapse_init = 0.06
z_message_gain = 1.8
z_message_gain_final = 2.6
z_message_gain_schedule_start = 0.55
monotone_accept = True
rollback_aux_on_reject = False
```

训练配置：

```text
num_samples = 256
local_search_passes = 220
sample_local_search_passes = 80
warm_start_source = none
```

### 4.1 状态变量

每一轮维护：

- `b_i^t=(X_i^t,Y_i^t,Z_i^t)`：节点 Bloch 向量。
- `p_i^t=(1-Z_i^t)/2`：Z-basis 概率。
- `phase_memory_i^t`：局部场相位记忆。
- `edge_message_{i->j}^t`：边上的 XY cavity 消息。
- `edge_z_message_{i->j}^t`：边上的 Z cavity 消息。
- `E[p^t]`：连续概率态的 QUBO 期望能量。

训练后的可学习参数包括：

- `field_steps[t]`：局部场驱动的 RY/mixer 步长。
- `phase_steps[t]`：局部场/记忆诱导的 RZ 相位步长。
- `mixer_bias[t]`：全局 mixer 偏置。
- `collapse_steps[t]`：后期 collapse 通道强度。
- `z_message_gain` / `z_message_gain_final`：Z-edge cavity 影响强度。

### 4.2 每一轮 V14 演化

第 `t` 轮：

1. 从当前概率 `p^t` 计算 QUBO local field：

```text
h_i = linear_i + sum_j Q_ij p_j
```

代码中默认做 degree/weight 归一化。

2. 更新相位记忆：

```text
m_i^t = decay * m_i^{t-1} + h_i
```

正式模型中 `decay=0.60`。

3. 计算边上的 Z cavity 建议。

对于 MaxCut，边希望两端 bit polarity 相反。代码中用 non-backtracking / cavity 形式构造有向边消息：

```text
bit_polarity_i = 2 p_i - 1
raw_message_{i->j} = -tanh(gain * cavity_belief_i)
```

然后聚合到节点，得到：

```text
z_edge_suggestion_i
z_edge_error_i = z_edge_suggestion_i - bit_polarity_i
```

4. 做 RZ 相位旋转。

当前正式 `phase_mode` 主要使用 memory/local-field 通道：

```text
phi_i^t = phase_steps[t] * m_i^t
```

然后对 Bloch 向量施加 RZ 旋转。

5. 做 RY mixer/collapse 旋转。

基础 mixer：

```text
theta_i^t = mixer_bias[t] - field_steps[t] * h_i
```

后期 collapse 通道启动后，加入 Z-edge cavity 方向：

```text
theta_i^t += collapse_steps[t] * z_edge_error_i
```

6. 得到 proposed Bloch 状态 `b_proposed`，计算 `p_proposed` 和 `E[p_proposed]`。

7. Strict monotone accept：

```text
accept iff E[p_proposed] <= E[p_current] + 1e-9
```

如果不接受：

- `b,p,E` 保持原状态；
- 正式配置 `rollback_aux_on_reject=False`，辅助消息不回滚。

8. 记录轨迹：

```text
energy_trace
probability_trace
bloch_trace
accepted_mask
j_trace
raw_j_trace
phase_angle_trace
after_rz_x_trace
```

## 5. 为什么需要 UTC 跳盆

基础 V14 的连续期望目标 `C[p]=-E[p]` 往往很平滑，但 hard readout 会出现离散跃迁：

```text
C[p] smooth
direct readout jumps
many bits flip together
```

我们的解释是：

- 连续态在 Bloch 空间中缓慢靠近 readout boundary；
- 某些窗口中很多变量同时接近边界；
- hard readout 对这些边界穿越非常敏感，于是出现 basin/readout 重排；
- 真正有效的跳盆机会通常在主相变峰前约 `30-60` 轮，而不是完全收敛之后。

所以 UTC 的思想是：

```text
先跑一次基础 V14
检测 direct readout 主相变峰 peak
只在 peak 前固定窗口做短暂 soft-monotone Bloch 扰动
从少数候选路径中选最好结果
```

## 6. 相变窗口检测

先运行一次 baseline V14，得到每轮：

- `expected_cut = -E[p]`
- `direct_cut`
- `direct_greedy_cut`
- `bit_flips_from_prev`
- `abs_d_direct`
- `abs_d_expected`

候选 readout transition 满足：

```text
round >= 60
abs_d_expected <= 2.0
and (
  bit_flips_from_prev >= 12
  or abs_d_direct >= 18
)
```

相邻候选按 `max_cluster_gap=4` 聚成事件。

每个事件记录：

- `start/end`
- `peak_round`
- `peak_readout_jump`
- `max_bit_flips`
- `direct_delta`
- `expected_delta`

主事件选择规则：

```text
优先 direct_delta > 0 的事件
按 direct_delta, peak_abs_d_direct, max_bit_flips 排序
选最大者
```

得到主相变峰：

```text
peak = main_event.peak_round
```

## 7. 正式 UTC-SM-lite v3 跳盆

正式算法使用统一 seed-independent 规则。

候选跳盆起点：

```text
starts = peak - {60, 55, 35, 30}
clip to [20, 220]
deduplicate
```

模板：

```text
template = cosine_stable
```

模板参数：

```text
window = 20
envelope = cosine_cool
temperature = 0.50
guidance = 0.60
noise = 0.08
global_floor = 0.03
transverse_strength = 0.00
z_shrink = 0.02
positive_gain_weight = 1.00
cheap_negative_weight = 0.00
bad_edge_weight = 1.40
low_conf_weight = 0.20
near_best_weight = 0.20
rho_power = 1.00
memory_decay = 0.85
memory_inject = 0.40
memory_strength = 0.04
clear_aux = none
clear_fraction = 0.02
```

Soft-monotone temperature branches：

```text
metropolis_temperature in {template, 0.06, 0.24}
```

其中 `template` 表示保留 `cosine_stable` 默认值，也就是 `0.06`，但保留原始 label/seed 路径。

Repeats：

```text
repeats = 2
```

总候选数约为：

```text
4 starts * 3 temperature branches * 2 repeats = 24 paths
```

有些 seed 因为 start clipping/dedup，实际候选数会略少。

## 8. UTC 跳盆算子

在某个 start 触发后，持续 `window=20` 轮。每个 active round 先做一次 soft global Bloch anneal，再继续 V14 原始 `_propose_round`。

### 8.1 节点评分 rho

对当前 hard readout bits 计算：

- `positive_gain_scale`：翻转该点能带来的正收益归一化。
- `bad_scale`：该点连接坏边的比例。
- `low_conf`：低置信度，即接近 `p=0.5`。
- `near_best`：接近最好 flip gain 的程度。

合成：

```text
score_i =
  1.00 * positive_gain_scale_i
  + 1.40 * bad_scale_i
  + 0.20 * low_conf_i
  + 0.20 * near_best_i
```

归一化后得到：

```text
rho_i = global_floor + (1-global_floor) * normalize(score_i)^rho_power
```

其中 `global_floor=0.03`，所以全图都有很弱的松动，冲突/可改进节点有更强扰动。

### 8.2 跳盆方向

方向来自当前 hard bit 的反方向：

```text
flip_direction_i = -1 if bit_i=1 else +1
```

这不是直接翻 bit，而是在 Bloch 空间沿 RY 方向推概率。

### 8.3 退火 envelope

使用 cosine cooling：

```text
env(s) = 0.5 * (1 + cos(pi * s))
s in [0,1]
```

早期强，后期逐渐冷却。

### 8.4 Anneal memory

UTC 维护一个短期跳盆 memory：

```text
escape_memory_i =
  memory_decay * escape_memory_i
  + memory_inject * env * rho_i * flip_direction_i
```

然后角度：

```text
theta_i =
  temperature * guidance * env * rho_i * flip_direction_i
  + noise * temperature * env * rho_i * Normal(0,1)
  + memory_strength * escape_memory_i
```

正式参数：

```text
temperature=0.50
guidance=0.60
noise=0.08
memory_decay=0.85
memory_inject=0.40
memory_strength=0.04
```

### 8.5 Bloch 更新

构造 RY 旋转：

```text
angles[:, 1] = theta
bloch <- RotY(theta) bloch
```

然后轻微压缩 Z：

```text
Z_i <- Z_i * (1 - z_shrink * env * rho_i)
```

其中 `z_shrink=0.02`。这表示向 readout boundary 轻微松动，但不是主方向。

### 8.6 Soft monotone

UTC active/recovery 阶段允许 soft monotone：

```text
if proposed_energy <= current_energy:
    accept
else:
    accept with probability exp(-(proposed_energy-current_energy)/T)
```

这里：

```text
T = metropolis_temperature * envelope(progress)
```

非 UTC 阶段仍使用基础 V14 的 strict monotone accept。

## 9. Guard / rollback

每次 jump event 会保存 checkpoint。

事件结束后，经历 `guard_recovery_rounds=24` 轮恢复，再检查质量：

```text
expected_ok = post_expected >= pre_expected - 4.0
direct_ok = post_direct >= pre_direct + 1
dg_ok = post_dg >= pre_dg + 1
```

正式 `guard_accept=quality`：

```text
accept_guard iff expected_ok and (direct_ok or dg_ok or post_expected >= pre_expected)
```

如果 guard 不通过，就回滚到 checkpoint。这样 UTC 可以探索，但不会任意破坏已有状态。

在快速扫描中，内部每轮 greedy guard 可跳过：

```text
fast_internal_scan = True
```

但候选路径结束后的最终评分仍使用 direct+greedy。

## 10. 候选路径选择

每个候选路径结束后，计算完整 trace 的：

- best expected cut
- best direct cut
- best direct+greedy cut

正式 selection score：

```text
score = best_direct_greedy_cut
```

最终选择：

```text
best path = argmax over 24 candidates of best_direct_greedy_cut
```

报告结果使用该 best path 的 direct+greedy cut。

## 11. 完整算法伪代码

```text
Algorithm V14-UTC(G, seed):
    Train or load V14 clean_edgeboost_mem060 model for G

    # Baseline dynamics
    state_base = V14.forward(G, return_state=True)
    trace_base = score_trace(state_base)

    # Transition detection
    diagnostics = transition_diagnostics(state_base, trace_base)
    events = detect_gated_readout_events(diagnostics)
    main_event = choose_main_event(events, metric="direct_positive")
    peak = main_event.peak_round

    # Unified transition-conditioned jump candidates
    starts = clip_unique(peak - [60,55,35,30], min=20, max=220)
    temps = [template_default, 0.06, 0.24]
    repeats = [0,1]

    candidates = []
    for start in starts:
        for temp in temps:
            for repeat in repeats:
                config = cosine_stable(start, temp)
                state = run_soft_global_v14(
                    model=V14,
                    G=G,
                    config=config,
                    seed=stable_seed(seed,start,temp,repeat),
                )
                score = best_direct_greedy_score(state)
                candidates.append((score,state,config))

    return candidate with maximum score
```

## 12. 正式实验对照

论文中建议只把以下方法放主表：

1. `base_v14`：基础 V14 动力学。
2. `old_anchor8`：早期固定窗口跳盆 baseline。
3. `V14-UTC`：正式算法，即 UTC-SM-lite v3。
4. `full_tc_sm`：质量上界/重 portfolio，对照而非主方法。
5. Classical heuristic：tabu / breakout / CP-SAT / GW baseline，视实验时间选择。

不要把所有历史策略都塞入主文。历史策略可以放附录或消融。

## 13. 当前 50 seed 结果

Random 50 seed, `n=512`, degree `3`：

| method | mean DG | median DG | min | max | mean time |
| --- | ---: | ---: | ---: | ---: | ---: |
| base V14 | 689.64 | 690.0 | 668 | 702 | 2.11s |
| old anchor8 | 698.06 | 698.5 | 686 | 706 | 17.89s |
| full TC-SM | 700.14 | 701.0 | 686 | 708 | 141.00s |
| V14-UTC | 698.40 | 700.0 | 685 | 706 | 53.14s |

解释：

- V14-UTC 比基础 V14 平均提升 `+8.76`。
- V14-UTC 比 old anchor8 平均提升 `+0.34`。
- full TC-SM 平均比 V14-UTC 高 `+1.74`，但耗时约 `2.65x`。

论文主张：

```text
V14-UTC is the recommended fast transition-conditioned dynamical solver.
full TC-SM is a heavier quality portfolio and serves as an upper-bound ablation.
```

## 14. 论文叙事建议

建议主线：

```text
1. MaxCut is used as a diagnostic benchmark for dynamical combinatorial optimization.
2. V14 represents variables as Bloch vectors and optimizes through trained phase-aware rotations.
3. Direct readout transitions reveal discrete basin rearrangements hidden under smooth C[p].
4. UTC exploits the pre-transition window, not the late locked plateau.
5. A small seed-independent candidate portfolio improves solution quality under a short time budget.
```

避免主张：

```text
Our method universally beats classical MaxCut solvers.
```

更稳妥的主张：

```text
The proposed quantum-inspired Bloch dynamical model exposes interpretable transition windows and supports efficient transition-conditioned basin escape without warm start.
```

## 15. 未来可扩展点

如果后续继续优化，优先方向不是继续无限扫策略，而是：

- 用更好的相变预测器减少候选路径数。
- 学习 `rho_i`，让节点扰动强度由模型预测而非手写特征。
- 把 UTC 训练进模型，让模型内部自发形成 transition-aware escape。
- 迁移到经典算法更弱或结构更动态的组合优化任务。

当前论文阶段，建议冻结正式算法为：

```text
V14 clean_edgeboost_mem060 + UTC-SM-lite v3
```

并把 full TC-SM 保留为附录中的重预算上界。
