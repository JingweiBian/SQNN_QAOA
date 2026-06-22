# V10 Sync-Local SQNN First Version

本文说明项目里的“第一版”主模型：`QUBOSynchronousLocalFieldSQNN`。它是后续 V11/V12/V13/V14/Clean-ZEdge 的基础版。

## 1. 状态与读出

每个二进制变量 `x_i` 对应一个 Bloch 向量：

```text
r_i = (X_i, Y_i, Z_i)
```

默认从 `|+>` 状态出发：

```text
X_i = 1
Y_i = 0
Z_i = 0
p_i = P(x_i = 1) = (1 - Z_i) / 2 = 0.5
```

最终概率只从 Z 基读出：

```text
p_i = P(x_i = 1) = (1 - Z_i) / 2
```

`X/Y` 是隐藏相干/相位通道，不直接作为最终 bit 概率。

## 2. 每一轮做什么

第 `t` 轮从旧概率 `p^t` 开始。

先计算 QUBO expected energy 的局部场：

```text
F_i = a_i + sum_j b_ij p_j
```

其中 QUBO energy 是：

```text
E(x) = constant + sum_i a_i x_i + sum_(i,j) b_ij x_i x_j
```

对于 MaxCut，项目里用：

```text
E_QUBO(x) = -C(x)
```

所以降低 `E` 等价于提高 cut。

然后做两个旋转：

```text
1. RZ:
   phi_i = phase_steps[t] * F_i
   作用：写入相位，只改变 X/Y，不直接改变 p_i。

2. RY:
   theta_i = mixer_bias[t] - field_steps[t] * F_i
   作用：把 X 方向的信息折回 Z，从而改变 p_i。
```

得到候选 Bloch 状态：

```text
r_candidate = RY(theta) RZ(phi) r_old
```

然后读出候选概率：

```text
p_candidate = (1 - Z_candidate) / 2
```

## 3. Monotone Accept

V10 的 `monotone_accept` 是全局整网接受/拒绝，不是逐点接受。

先算旧状态的整图 expected energy：

```text
E_old = E[p_old]
```

再算候选状态的整图 expected energy：

```text
E_candidate = E[p_candidate]
```

接受规则：

```text
if E_candidate <= E_old:
    接受整张图的候选状态 r_candidate
else:
    拒绝候选状态，整张图继续保留 r_old
```

因为 MaxCut 里 `E=-C`，所以这等价于：

```text
if C[p_candidate] >= C[p_old]:
    接受
else:
    拒绝
```

这保证的是 product Bernoulli expected cut `C[p]` 不下降；它不保证 deterministic readout `C_d`、greedy 后处理 `C_dg`、采样 `C_s` 每轮都单调。

代码实现上，候选旋转先存在 `proposed_bloch` 里。只有接受时才执行：

```text
bloch = proposed_bloch
probabilities = proposed_probabilities
current_energy = proposed_energy
```

所以在模拟里不需要“撤回旋转”；拒绝就是不赋值。

## 4. 步长怎么决定

V10 有三组逐轮参数：

```text
field_steps[t]:
  local field 进入 RY 的步长，默认初始化为 0.25。

phase_steps[t]:
  local field 进入 RZ 的步长，默认初始化为 0.10。

mixer_bias[t]:
  每轮共享 RY 偏置，默认初始化为 0。
```

这些不是手写固定 schedule，而是可训练参数。训练时用 AdamW 最小化：

```text
normalized_energy - entropy_weight * entropy
```

其中：

```text
normalized_energy = E[p] / (n * coefficient_scale)
```

训练结束后取训练过程中 `normalized_energy` 最低的参数，再做逐轮读出评估。

## 4.1 随机对称破缺

无权 3-正则 MaxCut 有一个特殊退化：如果 V10 从完全同质的 `|+>` 出发，
则初始 `p_i=0.5`，每个节点的 local field 都是 0：

```text
F_i = -3 + 2 * 3 * 0.5 = 0
```

这时所有节点会保持同质，优化步长也无法产生节点差异。因此 V10 支持可选的
节点级随机初始旋转：

```text
symmetry_breaking = none | random_ry | random_rz | random_rz_ry
symmetry_strength = 随机角度幅度
symmetry_seed     = 随机方向 seed
```

默认仍是 `none`，保持第一版基础行为。对 random unweighted 3-regular
MaxCut，建议使用 `random_rz_ry` 或至少 `random_ry`，并对多个
`symmetry_seed` 分别训练/搜索，最后按验证指标取最好的一条方向。

## 5. 当前评估口径

对 random unweighted 3-regular MaxCut：

```text
n = 512
degree = 3
w_ij = 1
W = |E| = 3n/2 = 768
```

不知道 exact optimum `C*` 时，使用理论上界：

```text
C* <= W
```

所以报告：

```text
R = C / W
```

这是相对理论上界的保守比例，不是严格 `C/C*`。

逐轮报告：

```text
R_d   = C_d / W
R_dg  = C_dg / W
R_s   = C_s(K) / W
R_exp = C[p] / W
R_GW  = GW_expected / W
```

其中 `C_dg` 使用 1-bit greedy local search，`C_s(K)` 是从 SQNN product Bernoulli 分布采样 K 次取最好值，不接 greedy。
