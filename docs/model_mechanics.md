# SQNN 中 `R_Z` 与 `R_Y` 两个旋转操作的作用与改进建议

本文整理了对 SQNN-QUBO 模型中两个 Bloch 向量旋转操作的理解：先做 `R_Z`，再做 `R_Y`。核心结论是：

> `R_Z` 主要改变 `X-Y` 平面中的相干相位，不直接改变变量取 0/1 的概率；`R_Y` 负责把当前的 `X` 分量转成 `Z` 偏置，从而直接影响 `P(x_i=1)`。但多轮迭代时，`R_Z` 会改变后续 `R_Y` 的有效方向，因此需要角度裁剪、步长衰减和残差更新来稳定训练/推理。

当前主线使用这个结构直接产生 Z-basis deterministic readout。这里的 QAOA 关联主要是 cost/mixer 交替思想：局部 cost field 推动概率偏置，mixer-like 旋转保留探索能力；它不是把 SQNN 只当成后续 QAOA 或贪心算法的 warm-start 预处理。

---

## 1. Bloch 向量的基本含义

对每个变量 `x_i`，用一个 Bloch 向量表示其状态：

```text
r_i = [X_i, Y_i, Z_i]^T
```

对应密度矩阵：

```math
\rho_i = \frac{1}{2}\left(I + X_i\sigma_x + Y_i\sigma_y + Z_i\sigma_z\right).
```

三个分量的含义：

```math
X_i = \langle \sigma_x \rangle,\qquad
Y_i = \langle \sigma_y \rangle,\qquad
Z_i = \langle \sigma_z \rangle.
```

其中 `Z_i` 直接对应变量的二进制概率。如果约定：

```math
P(x_i=1)=\frac{1-Z_i}{2},
```

那么：

```math
Z_i>0 \Rightarrow x_i=0 \text{ 更可能},
```

```math
Z_i<0 \Rightarrow x_i=1 \text{ 更可能}.
```

`X_i` 和 `Y_i` 不直接对应 0/1 概率，而是表示 `|0\rangle` 和 `|1\rangle` 之间的相干信息。可以把它们理解成隐藏传播通道或相位记忆通道。

因此，代码中的 `initial_probabilities` 一律应理解为 `P(x_i=1)`。若要从给定概率初始化 Bloch 向量，必须使用：

```math
Z_i = 1 - 2P(x_i=1).
```

如果没有外部初始概率，当前主线默认从 `|+\rangle` 开始，即 `r_i=(1,0,0)`，此时 `Z_i=0` 且 `P(x_i=1)=0.5`。

---

## 2. 两个旋转操作的定义

当前单轮更新可以写成：

```math
r_i^{t+1} = R_Y(\theta_i^t)R_Z(\phi_i^t)r_i^t.
```

其中：

```math
\theta_i^t = \text{mixer\_bias}_t - \text{field\_step}_t F_i^t.
```

这里 `F_i^t` 是当前变量的局部场或局部梯度信息。

### 2.1 `R_Z` 旋转

```math
R_Z(\phi)=
\begin{pmatrix}
\cos\phi & -\sin\phi & 0\\
\sin\phi & \cos\phi & 0\\
0 & 0 & 1
\end{pmatrix}.
```

作用后：

```math
\begin{pmatrix}
X_i'\\
Y_i'\\
Z_i'
\end{pmatrix}
=
\begin{pmatrix}
\cos\phi_i^t X_i^t - \sin\phi_i^t Y_i^t\\
\sin\phi_i^t X_i^t + \cos\phi_i^t Y_i^t\\
Z_i^t
\end{pmatrix}.
```

所以：

```math
Z_i'=Z_i^t.
```

因此，`R_Z` **不会直接改变**：

```math
P(x_i=1)=\frac{1-Z_i}{2}.
```

它只是把 Bloch 向量在 `X-Y` 平面中旋转，相当于调整相干相位或传播方向。

---

### 2.2 `R_Y` 旋转

```math
R_Y(\theta)=
\begin{pmatrix}
\cos\theta & 0 & \sin\theta\\
0 & 1 & 0\\
-\sin\theta & 0 & \cos\theta
\end{pmatrix}.
```

作用在 `R_Z` 后的向量上：

```math
\begin{pmatrix}
X_{i,\mathrm{out}}\\
Y_{i,\mathrm{out}}\\
Z_{i,\mathrm{out}}
\end{pmatrix}
=
R_Y(\theta_i^t)
\begin{pmatrix}
X_i'\\
Y_i'\\
Z_i'
\end{pmatrix}.
```

得到：

```math
X_{i,\mathrm{out}} = \cos\theta_i^t X_i' + \sin\theta_i^t Z_i',
```

```math
Y_{i,\mathrm{out}} = Y_i',
```

```math
Z_{i,\mathrm{out}} = -\sin\theta_i^t X_i' + \cos\theta_i^t Z_i'.
```

因为输出概率由 `Z` 决定，所以真正改变 `P(x_i=1)` 的是：

```math
Z_{i,\mathrm{out}} = -\sin\theta_i^t X_i' + \cos\theta_i^t Z_i^t.
```

---

## 3. 两个旋转合起来的完整矩阵

整体操作为：

```math
r_i^{t+1}=R_Y(\theta_i^t)R_Z(\phi_i^t)r_i^t.
```

矩阵乘积为：

```math
R_Y(\theta)R_Z(\phi)=
\begin{pmatrix}
\cos\theta\cos\phi & -\cos\theta\sin\phi & \sin\theta\\
\sin\phi & \cos\phi & 0\\
-\sin\theta\cos\phi & \sin\theta\sin\phi & \cos\theta
\end{pmatrix}.
```

因此：

```math
\begin{pmatrix}
X_{\mathrm{out}}\\
Y_{\mathrm{out}}\\
Z_{\mathrm{out}}
\end{pmatrix}
=
\begin{pmatrix}
\cos\theta\cos\phi & -\cos\theta\sin\phi & \sin\theta\\
\sin\phi & \cos\phi & 0\\
-\sin\theta\cos\phi & \sin\theta\sin\phi & \cos\theta
\end{pmatrix}
\begin{pmatrix}
X\\
Y\\
Z
\end{pmatrix}.
```

其中最重要的是 `Z` 分量：

```math
Z_{\mathrm{out}}
=
-\sin\theta\cos\phi\,X
+
\sin\theta\sin\phi\,Y
+
\cos\theta\,Z.
```

所以输出概率为：

```math
P_{\mathrm{out}}(x_i=1)
=
\frac{1-Z_{\mathrm{out}}}{2}
```

也就是：

```math
P_{\mathrm{out}}(x_i=1)
=
\frac{
1
+
\sin\theta\cos\phi\,X
-
\sin\theta\sin\phi\,Y
-
\cos\theta Z
}{2}.
```

---

## 4. 对常见初态 `|+>` 的特殊分析

如果初态为：

```math
|+\rangle=\frac{|0\rangle+|1\rangle}{\sqrt{2}},
```

对应：

```math
r_i^0=(1,0,0)^T.
```

先做 `R_Z(\phi)`：

```math
r_i'=(\cos\phi,\sin\phi,0)^T.
```

再做 `R_Y(\theta)`：

```math
r_{i,\mathrm{out}}
=
(\cos\theta\cos\phi,\sin\phi,-\sin\theta\cos\phi)^T.
```

因此：

```math
Z_{i,\mathrm{out}}=-\sin\theta\cos\phi.
```

输出概率为：

```math
P_{\mathrm{out}}(x_i=1)
=
\frac{1+\sin\theta\cos\phi}{2}.
```

这说明 `R_Z` 虽然不直接改变概率，但它通过因子 `cos(phi)` 调节了后续 `R_Y` 改变概率的方向和强度。

---

## 5. 局部场方向是否稳定？

原本希望的方向是：

```math
F_i^t>0 \Rightarrow \theta_i^t<0 \Rightarrow P(x_i=1)\downarrow,
```

```math
F_i^t<0 \Rightarrow \theta_i^t>0 \Rightarrow P(x_i=1)\uparrow.
```

这个结论在 `phi=0` 或 `cos(phi)>0` 时成立。

因为在初态 `r=(1,0,0)^T` 附近：

```math
P(x_i=1)=\frac{1+\sin\theta\cos\phi}{2}.
```

若：

```math
\cos\phi>0,
```

则 `theta` 对概率的推动方向保持正常。

若：

```math
\cos\phi=0,
```

则 `R_Y` 几乎无法把当前相干信息转成 `Z` 概率变化。

若：

```math
\cos\phi<0,
```

则方向会反转。

更一般地，真正决定 `R_Y` 推动方向的是：

```math
X_i' = \cos\phi_i^t X_i^t - \sin\phi_i^tY_i^t.
```

小角度近似下：

```math
\Delta P_i(x_i=1)\approx \frac{\theta_i^t X_i'}{2}.
```

如果 `mixer_bias_t=0`，则：

```math
\theta_i^t=-\eta_tF_i^t,
```

所以：

```math
\Delta P_i(x_i=1)
\approx
-\frac{\eta_tF_i^tX_i'}{2}.
```

因此，局部场对概率的方向不仅取决于 `F_i^t`，还取决于当前 `X_i'` 的符号。

---

## 6. 多轮迭代的影响

多轮迭代时：

```math
r_i^T
=
\left[R_Y(\theta_i^{T-1})R_Z(\phi_i^{T-1})\right]
\cdots
\left[R_Y(\theta_i^0)R_Z(\phi_i^0)\right]
r_i^0.
```

如果角度固定：

```math
\theta_i^t=\theta,\qquad \phi_i^t=\phi,
```

则：

```math
r_i^T = M^T r_i^0,
\qquad
M=R_Y(\theta)R_Z(\phi).
```

由于 `M` 是旋转矩阵，它保持 Bloch 向量长度不变，不会自动收敛。也就是说，纯旋转多轮以后可能只是周期性地在 Bloch 球面上运动，而不是稳定地把概率推向最优解。

如果 `F_i^t` 每轮根据当前状态重新计算，那么系统变成非线性迭代：

```math
r_i^t
\rightarrow
p_i^t
\rightarrow
F_i^t
\rightarrow
\theta_i^t
\rightarrow
r_i^{t+1}.
```

这有利于传播图上多跳邻居信息，但也可能带来：

1. 概率振荡；
2. 方向翻转；
3. 过度自信；
4. 早期错误被放大；
5. 图传播过强导致过平滑。

---

## 7. 改进建议一：角度裁剪

为了避免大角度旋转导致方向翻转，建议对两个角度都做裁剪：

```math
\phi_i^t
=
\operatorname{clip}
\left(
\phi_{\mathrm{raw},i}^t,
-\phi_{\max}^t,
\phi_{\max}^t
\right),
```

```math
\theta_i^t
=
\operatorname{clip}
\left(
\text{mixer\_bias}_t-\eta_t F_i^t,
-\theta_{\max}^t,
\theta_{\max}^t
\right).
```

初始可选：

```math
\phi_{\max}^0\leq \frac{\pi}{4},
\qquad
\theta_{\max}^0\leq \frac{\pi}{4}.
```

原因是当 `|phi| < pi/2` 时，初态附近通常有：

```math
\cos\phi>0,
```

从而 `R_Y` 对概率的推动方向不容易反转。

更稳妥的做法是让角度上限逐轮减小：

```math
\theta_{\max}^t
=
\theta_{\min}
+
(\theta_{\max}^0-\theta_{\min})\lambda_\theta^t,
```

```math
\phi_{\max}^t
=
\phi_{\min}
+
(\phi_{\max}^0-\phi_{\min})\lambda_\phi^t.
```

这样前期允许较大搜索，后期只允许小幅微调。

---

## 8. 改进建议二：field step 衰减

可以让局部场步长前期大、后期小：

```math
\eta_t=\eta_0\lambda_\eta^t,
\qquad 0<\lambda_\eta<1.
```

更推荐加一个下限：

```math
\eta_t
=
\eta_{\min}
+
(\eta_0-\eta_{\min})\lambda_\eta^t.
```

这样前期：

```math
\eta_t \approx \eta_0,
```

模型可以进行较强传播和探索；后期：

```math
\eta_t \approx \eta_{\min},
```

模型逐渐稳定，不会因为局部场波动而一直大幅旋转。

可以理解成优化中的 learning rate decay 或 annealing schedule。

---

## 9. 改进建议三：残差更新

纯旋转更新是：

```math
\tilde r_i^{t+1}=R_Y(\theta_i^t)R_Z(\phi_i^t)r_i^t,
```

```math
r_i^{t+1}=\tilde r_i^{t+1}.
```

残差更新是不完全接受候选状态，而是把旧状态和新状态混合：

```math
r_i^{t+1}
=
(1-\alpha_t)r_i^t
+
\alpha_t\tilde r_i^{t+1}.
```

也可以写成：

```math
r_i^{t+1}
=
r_i^t
+
\alpha_t
\left(
\tilde r_i^{t+1}-r_i^t
\right).
```

其中 `alpha_t` 是接受本轮旋转结果的比例。

如果：

```math
\alpha_t=1,
```

就是完全接受旋转结果。

如果：

```math
\alpha_t=0.2,
```

就是保留 80% 旧状态，只接受 20% 新状态。

残差更新的直观含义是：

> 不要每一轮都猛转，而是朝旋转后的方向小步移动。

建议也让 `alpha_t` 逐轮衰减：

```math
\alpha_t
=
\alpha_{\min}
+
(\alpha_0-\alpha_{\min})\lambda_\alpha^t.
```

这样前期快速传播，后期逐渐稳定。

---

## 10. 是否需要 Normalize？

有两种选择。

### 10.1 强制 normalize

```math
r_i^{t+1}
=
\frac{
(1-\alpha_t)r_i^t+
\alpha_t\tilde r_i^{t+1}
}{
\left\|
(1-\alpha_t)r_i^t+
\alpha_t\tilde r_i^{t+1}
\right\|
}.
```

这样每个变量始终保持在 Bloch 球面上，类似纯态。

优点：结构更接近量子态旋转。

缺点：可能把不确定性重新放大。

### 10.2 不强制 normalize，只投影到 Bloch 球内

```math
r_i^{t+1}
=
(1-\alpha_t)r_i^t+
\alpha_t\tilde r_i^{t+1}.
```

然后做安全投影：

```math
r_i^{t+1}
\leftarrow
\frac{r_i^{t+1}}{\max(1,\|r_i^{t+1}\|)}.
```

由于两个 Bloch 向量的凸组合仍在 Bloch 球内，通常不会超过长度 1。这个版本更像混合态，允许状态长度变短，从而表达“不确定”。

对 SQNN-QUBO 任务，推荐优先使用 **不强制 normalize，只做安全投影** 的版本。

---

## 11. 推荐的完整更新流程

每轮先计算 schedule：

```math
\eta_t
=
\eta_{\min}
+
(\eta_0-\eta_{\min})\lambda_\eta^t,
```

```math
\alpha_t
=
\alpha_{\min}
+
(\alpha_0-\alpha_{\min})\lambda_\alpha^t,
```

```math
\theta_{\max}^t
=
\theta_{\min}
+
(\theta_{\max}^0-\theta_{\min})\lambda_\theta^t,
```

```math
\phi_{\max}^t
=
\phi_{\min}
+
(\phi_{\max}^0-\phi_{\min})\lambda_\phi^t.
```

然后执行：

```math
\phi_i^t
=
\operatorname{clip}
\left(
\phi_{\mathrm{raw},i}^t,
-\phi_{\max}^t,
\phi_{\max}^t
\right),
```

```math
\theta_i^t
=
\operatorname{clip}
\left(
\text{mixer\_bias}_t-\eta_tF_i^t,
-\theta_{\max}^t,
\theta_{\max}^t
\right),
```

```math
\tilde r_i^{t+1}
=
R_Y(\theta_i^t)R_Z(\phi_i^t)r_i^t,
```

```math
r_i^{t+1}
=
(1-\alpha_t)r_i^t+
\alpha_t\tilde r_i^{t+1}.
```

最后做安全投影：

```math
r_i^{t+1}
\leftarrow
\frac{r_i^{t+1}}{\max(1,\|r_i^{t+1}\|)}.
```

---

## 12. 伪代码

```python
for t in range(T):
    # 1. schedules: 前期大，后期小
    eta_t = eta_min + (eta0 - eta_min) * (lambda_eta ** t)
    alpha_t = alpha_min + (alpha0 - alpha_min) * (lambda_alpha ** t)

    theta_clip_t = theta_min + (theta_clip0 - theta_min) * (lambda_theta ** t)
    phi_clip_t = phi_min + (phi_clip0 - phi_min) * (lambda_phi ** t)

    # 2. compute QUBO local field
    F = compute_local_field(r, Q)

    # 3. compute and clip RZ angle
    phi_raw = compute_phi(r, Q)
    phi = clip(phi_raw, -phi_clip_t, phi_clip_t)

    # 4. compute and clip RY angle
    theta_raw = mixer_bias[t] - eta_t * F
    theta = clip(theta_raw, -theta_clip_t, theta_clip_t)

    # 5. proposal update by rotations
    r_proposal = Ry(theta) @ Rz(phi) @ r

    # 6. residual update
    r_new = (1 - alpha_t) * r + alpha_t * r_proposal

    # 7. safety projection into Bloch ball
    norm = norm_of_each_bloch_vector(r_new)
    r_new = r_new / maximum(1.0, norm)

    r = r_new
```

---

## 13. 推荐初始参数

可以先从保守配置开始：

```math
\phi_{\max}^0=\frac{\pi}{4},
\qquad
\phi_{\min}=\frac{\pi}{12},
```

```math
\theta_{\max}^0=\frac{\pi}{4},
\qquad
\theta_{\min}=\frac{\pi}{16},
```

```math
\eta_0=0.5,
\qquad
\eta_{\min}=0.05,
\qquad
\lambda_\eta=0.95,
```

```math
\alpha_0=0.7,
\qquad
\alpha_{\min}=0.1,
\qquad
\lambda_\alpha=0.95.
```

如果收敛太慢，可以提高：

```math
\eta_0,\quad \alpha_0.
```

如果概率震荡明显，可以降低：

```math
\theta_{\max}^0,\quad \phi_{\max}^0,\quad \eta_0,\quad \alpha_0.
```

---

## 14. 对 SQNN-QUBO 路线的总结

这两个旋转的功能分工可以概括为：

```text
R_Z: 在 X-Y 相干平面内编码传播/相位信息；不直接改变二进制概率。

R_Y: 把当前 X' 分量转成 Z 偏置；直接改变 P(x_i=1)。
```

但多轮迭代时，`R_Z` 会影响下一步 `R_Y` 的有效方向。真正决定局部场是否按预期改变概率的是：

```math
X_i' = \cos\phi_i^t X_i^t - \sin\phi_i^tY_i^t.
```

如果：

```math
X_i'>0,
```

则：

```math
F_i^t>0 \Rightarrow P(x_i=1)\downarrow.
```

如果：

```math
X_i'<0,
```

方向会反转。

因此，为了让 SQNN 的多轮传播稳定可控，建议采用：

```text
角度裁剪 + field step 衰减 + 残差更新 + Bloch ball 安全投影
```

最终目标是实现：

```text
前期：大步传播，快速探索图结构；
后期：小步微调，减少振荡并稳定高置信变量。
```

评价上，`C[p]` 只说明 product Bernoulli 概率分布本身的 expected cut 形状；当前主线最终关心的是 `p_i >= 0.5` 得到的 Z-basis deterministic bitstring 和 `C_d`。隐藏 `X/Y/Z` 向量用于解释相位传播，但不再作为近期 hyperplane readout 诊断主线。
