# Classical Baselines

本文只规范 MaxCut-3 里的经典对比 baseline。指标分母的定义见 [metrics_and_ratios.md](metrics_and_ratios.md)。

## Main Baseline

当前主 baseline 是：

```text
GW expected hyperplane cut
```

定义：

```text
GW_expected = sum_edges arccos(v_i dot v_j) / pi
```

项目当前实现位于：

```text
classical/maxcut3_compare.py
```

当前实现是 low-rank Burer-Monteiro style vector relaxation，所以报告时应写：

```text
GW-style expected
```

不要写成 certified SDP-GW，除非后续接入真正 SDP solver 并保存求解证书。

## Auxiliary Baselines

### Random + 1-bit Greedy

作用：

```text
衡量简单随机初始化加局部搜索本身有多强。
```

它不是主要论文 baseline，但能检查 SQNN 是否只是被 local search 托起来。

### GW Sampled-Best

定义：

```text
从 GW vectors 采 K 个随机 hyperplanes，
取 cut value 最大的 bitstring。
```

用途：

```text
和 SQNN sample(K) 做同口径比较。
```

报告时必须写明：

```text
K = number of sampled hyperplanes
```

### CP-SAT

作用：

```text
1. 小图或中图上尝试证明 exact C_star；
2. 无法证明最优时，给出 incumbent 和 certified upper bound。
```

报告规则：

```text
status = OPTIMAL:
  可以报告 C/C_star。

status = FEASIBLE:
  只能报告 C/UB 和 C/C_best_known。
```

### Simulated Annealing

作用：

```text
1. 作为强 classical heuristic baseline；
2. 作为 SQNN 平台期 escape 的候选模块；
3. 检查 SQNN 的提升是否只是来自 classical post-processing。
```

当前记录：

```text
maxcut/v14_utc/reports/archive/maxcut3_sa_baseline_and_escape.md
```

报告规则：

```text
SA-only 是 classical baseline。
SQNN + SA-guided escape 是 hybrid solver，不计入 pure SQNN 指标。
```

### SDP Upper Bound

作用：

```text
给出 MaxCut 的实例级数学上界 SDP_UB。
```

当前状态：

```text
还没有接入 certified SDP solver。
```

后续如果接入 SDPNAL+、MOSEK、CSDP 等，应保存：

```text
SDP_UB
SDP_solver
SDP_gap
C_GW_rounding
C_GW_rounding / SDP_UB
```

## Which Baseline To Use

### SQNN Expected

```text
C[p]
```

对比：

```text
GW expected
```

注意：

```text
SQNN expected 是 product Bernoulli probability expected cut；
GW expected 是 vector hyperplane expected cut。
两者都是 expected value，但分布族不同。
当前主线里 C[p] 是辅助诊断，不是最终优化目标。
```

### SQNN Direct

```text
C_d = C(1[p_i >= 0.5])
```

对比：

```text
GW expected
```

这是当前最重要的 deterministic readout 对标。
它对应 Z-basis deterministic readout，是当前最接近物理实验测量的主指标。

### SQNN Directgreedy

```text
C_dg = 1-bit greedy applied to direct readout
```

对比：

```text
GW expected
```

但必须注明：

```text
contains 1-bit greedy local search
```

### SQNN Sample

```text
C_s(K) = best of K Bernoulli samples from p
```

对比：

```text
GW sampled-best(K)
```

K 必须写清楚。

### Bloch Hyperplane Readout 已封存

```text
C_bloch(K) = best of K hyperplanes over SQNN Bloch vectors
```

历史复现实验如需使用，建议同时对比：

```text
GW expected
GW sampled-best(K)
```

因为 Bloch hyperplane readout 与 GW 的向量超平面 rounding 机制最接近。

当前主线暂时封存该方向：

```text
1. 它主要是在固定 SQNN 分布/隐藏向量后寻找更强后处理读出；
2. 它不是当前优化结果本身的主评价；
3. 它的物理实验实现路径不直接。
```

## Current Clean Baseline Protocol

n=512 十图主协议：

```text
graph = random 3-regular MaxCut
n = 512
seeds = 0..9
main baseline = GW-style expected
```

报告表至少包含：

```text
GW expected C/W
SQNN expected C/W
SQNN direct C/W
SQNN directgreedy C/W
SQNN sample C/W
gap to GW expected
```

其中 `SQNN direct C/W` 是主指标；`SQNN expected`、`directgreedy` 和
`sample` 是辅助诊断或工程对照。

正式论文式比较再补：

```text
C/UB
C/Cstar, only if exact
```
