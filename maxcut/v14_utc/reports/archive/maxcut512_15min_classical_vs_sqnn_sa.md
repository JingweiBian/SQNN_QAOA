# MaxCut-3 n=512 15min Classical vs SQNN+SA

Date: 2026-06-23

本报告记录一次限时对比实验。它不是严格 `C/C*` 实验，因为 n=512 seed0 的 `C*` 尚未证明。

## Setup

```text
graph = random 3-regular MaxCut
n = 512
degree = 3
seed = 0
W = |E| = 768
classical total budget = 900s
SQNN+SA budget = 900s
```

输出目录：

```text
outputs/maxcut512_15min_classical_vs_sqnn_sa_seed0
```

关键文件：

```text
summary.csv
cp_sat.json
gw_style.json
sqnn_best_trace.csv
plots/best_c_over_w_compact.png
plots/sqnn_sa_trial_distribution.png
plots/sqnn_sa_trace.png
```

## Classical Side

经典侧是一个 15min portfolio：

```text
1. GW-style low-rank vector relaxation + hyperplane rounding + greedy
2. random + 1-bit greedy
3. active-set SA + greedy heuristic
4. CP-SAT 使用剩余时间，输出 incumbent 和 certified upper bound
```

结果：

| method | C | C/W | seconds | note |
|---|---:|---:|---:|---|
| GW-style expected | 676.509 | 0.880871 | 3.94 | paper-style expected hyperplane cut |
| GW sampled-best | 695 | 0.904948 | 3.94 | 8192 hyperplanes |
| GW + greedy | 701 | 0.912760 | 3.94 | sampled-best then 1-bit greedy |
| random + greedy | 677 | 0.881510 | 30.01 | 2809 random restarts |
| classical active-SA + greedy | 683 | 0.889323 | 150.21 | 604 runs, active size 180 |
| CP-SAT incumbent | 704 | 0.916667 | 716.15 | status FEASIBLE |
| CP-SAT upper bound | 718 | 0.934896 | 716.15 | certified upper bound |

因此当前可证明信息是：

```text
704 <= C_star <= 718
```

对应：

```text
0.916667 <= C_star/W <= 0.934896
```

## SQNN+SA Side

SQNN 侧使用已通过 n=64 exact 守门测试的加速路线：

```text
active-set SA
cascade escape
success-cache only
final-only escape
max_escapes = 3
active_max_fraction = 0.35
```

900s 内完成：

```text
182 trials
total trial seconds ~= 816.11s
```

最好结果：

| SQNN metric | C | C/W |
|---|---:|---:|
| best expected | 687.880 | 0.895676 |
| best direct | 689 | 0.897135 |
| best direct+greedy | 689 | 0.897135 |
| best sample | 689 | 0.897135 |

最佳 trial：

```text
trial = 146
direct C = 689
direct C/W = 0.897135
sa_calls = 2
cascade_hits = 1
active_size_mean = 180
```

## Comparison

排序如下：

| method | C/W |
|---|---:|
| CP-SAT upper bound | 0.934896 |
| CP-SAT incumbent | 0.916667 |
| GW + greedy | 0.912760 |
| GW sampled-best | 0.904948 |
| SQNN+SA best direct | 0.897135 |
| classical active-SA + greedy | 0.889323 |
| random + greedy | 0.881510 |
| GW-style expected | 0.880871 |

当前结论：

```text
1. SQNN+SA 明显强于 random+greedy 和本次 active-SA heuristic。
2. SQNN+SA 仍低于 GW sampled-best、GW+greedy 和 CP-SAT incumbent。
3. 因此当前加速 SQNN+SA 不能声称在 n=512 上超过强经典。
4. 如果用 CP-SAT upper bound 做保守比值：
   SQNN+SA best direct / UB = 689 / 718 = 0.959610。
   CP-SAT incumbent / UB = 704 / 718 = 0.980501。
```

## Google-Ratio Back-Inference

若参考 Google 文章中 random 3-regular 图上 GW expected 的典型近似比约 `0.95-0.96`，本图的 GW expected 为：

```text
C_GW_expected = 676.509
```

则可反推一个非严格估计：

```text
C_star ~= C_GW_expected / 0.96 to C_GW_expected / 0.95
       ~= 704.7 to 712.1
```

这与 CP-SAT 当前证书：

```text
704 <= C_star <= 718
```

是相容的。但这个反推不是实例级证明，不能替代 CP-SAT / SDP / MILP upper bound。

## Next Actions

当前最需要改进的是 SQNN+SA 在 n=512 的解质量，而不是速度：

```text
1. active set 目前固定到 180 个变量，可能太窄或选择规则不够准。
2. SQNN 训练部分仍是 V10-like probe，不是完整 Clean-ZEdge/V14 主模型。
3. final-only escape 很快，但可能错过训练中后期更好的 basin transition。
4. 下一轮应测试：
   active_max_fraction = 0.5 / 0.7
   max_escapes = 5 / 8
   sa_steps = 5000 / 10000
   早期 route selector
   V14-clean + SA escape
```
