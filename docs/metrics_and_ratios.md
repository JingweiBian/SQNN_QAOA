# Metrics And Ratios

本文只规范 MaxCut-3 实验里的指标命名。以后报告、图表、CSV 字段都应尽量沿用这里的名字，避免把 `C/W`、`C/C*` 和 baseline gap 混在一起。

## Basic Quantities

对一个 MaxCut 图：

```text
G = (V, E)
w_ij = edge weight
W = sum_edges w_ij
C(x) = cut value of bitstring x
C_star = true optimum MaxCut value
UB = certified upper bound
C_best_known = best known feasible cut value
```

无权 random 3-regular 图中：

```text
W = |E| = 3n/2
```

## C/W

```text
C_over_W = C(x) / W
```

含义：

```text
cut fraction，割比例。
它表示总边权里有多少比例被切开。
```

注意：

```text
C/W 不是严格 approximation ratio。
历史代码中很多字段名叫 ratio，但在 MaxCut-3 上实际分母是 W。
以后必须写成 C_over_W 或 cut_fraction。
```

## C/C*

```text
C_over_Cstar = C(x) / C_star
```

含义：

```text
strict approximation ratio。
```

只在以下情况使用：

```text
1. exact solver 已证明 OPTIMAL；
2. 或该图的最优值 C_star 已由其他方式严格知道。
```

如果 CP-SAT 只返回 `FEASIBLE`，不能把 incumbent 写成 `C_star`。

## C/UB

```text
C_over_UB = C(x) / UB
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
C(x) / UB <= C(x) / C_star
```

也就是说，`C/UB` 不会虚高，适合在 `C_star` 未证明时做正式对比。

常见 UB：

```text
SDP_UB
CP-SAT upper bound
MILP dual bound
```

## C/C_best_known

```text
C_over_best_known = C(x) / C_best_known
```

含义：

```text
相对当前最好已知离散解的比例。
```

注意：

```text
如果 C_best_known 尚未证明等于 C_star，
C/C_best_known 不能叫 strict approximation ratio。
```

这个指标适合做工程进展比较，但不适合替代严格近似比。

## GW Expected

论文口径 GW baseline 通常指：

```text
GW_expected = E_hyperplane[C_GW]
            = sum_edges arccos(v_i dot v_j) / pi
```

其中：

```text
v_i = vector relaxation 给出的单位向量
```

注意：

```text
GW expected 是随机超平面 rounding 的期望 cut value。
它不是 sampled-best。
它也不是 local-search 后处理结果。
```

## Recommended Field Names

主实验至少保存：

```text
graph_id
n
degree
seed
W
C_star
UB
C_best_known
```

SQNN 结果：

```text
sqnn_expected_C
sqnn_direct_C
sqnn_directgreedy_C
sqnn_sample_C
sqnn_bloch_C
```

baseline 结果：

```text
gw_expected_C
gw_sampled_best_C
random_greedy_C
```

规范化指标：

```text
*_C_over_W
*_C_over_Cstar
*_C_over_UB
*_C_over_best_known
```

## Reporting Rule

以后不要只写：

```text
ratio = 0.90
```

必须写成：

```text
C_over_W = 0.90
```

或：

```text
C_over_UB = 0.95
```

或：

```text
C_over_Cstar = 0.96
```

分母不清楚的 `ratio` 一律视为不规范。
