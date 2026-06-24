# MaxCut-3 Classical SA And Escape Notes

Date: 2026-06-23

本文记录 classical simulated annealing baseline，以及把 SA 作为 SQNN 平台期 escape 模块时的速度/质量观察。

## Classical SA Baseline

实现位置：

```text
quantum/warmstart/heuristics.py::simulated_annealing
```

当前实现是通用 QUBO simulated annealing，不是 MaxCut-3 专用增量实现。每一步会重新计算 flip delta，因此还有明显优化空间。

下面表格是 CPU 快速规模测试。指标是 `C/W`，不是严格 `C/C*`；除 n=64 exact probe 外，这里没有证明最优值。

| n | graph seed | restarts | steps/restart | SA C/W | SA+greedy C/W | anneal time |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0 | 16 | 2000 | 0.901042 | 0.901042 | 2.08s |
| 256 | 0 | 16 | 2000 | 0.875000 | 0.875000 | 2.14s |
| 512 | 0 | 16 | 2000 | 0.850260 | 0.864583 | 2.27s |
| 1024 | 0 | 16 | 2000 | 0.808594 | 0.858724 | 2.51s |
| 512 | 1 | 64 | 2000 | 0.851562 | 0.867188 | 9.02s |
| 1024 | 1 | 64 | 2000 | 0.809896 | 0.867839 | 10.03s |
| 2048 | 1 | 64 | 2000 | 0.738281 | 0.802734 | 13.68s |
| 4096 | 1 | 64 | 2000 | 0.667318 | 0.723958 | 16.27s |
| 512 | 1 | 16 | 10000 | 0.878906 | 0.878906 | 11.16s |
| 1024 | 1 | 16 | 10000 | 0.865885 | 0.865885 | 12.11s |
| 2048 | 2 | 16 | 10000 | 0.847331 | 0.856120 | 16.45s |
| 4096 | 2 | 16 | 10000 | 0.813151 | 0.837565 | 19.80s |

观察：

```text
1. SA 能轻松跑到 n=4096；当前瓶颈主要是步数/退火深度，不是内存。
2. 2000 steps 对 512/1024 还能给可用解，但对 2048/4096 明显偏浅。
3. 10000 steps 明显改善大图质量，尤其是 2048/4096。
4. 这些结果是 classical heuristic baseline，不应和 GW expected 论文口径混淆。
```

## Exact n=64 Probe

图配置：

```text
n = 64
degree = 3
seed = 0
C_star = 86, CP-SAT OPTIMAL
```

纯 SA 对照：

| method | config | best C | C/C* | time |
|---|---|---:|---:|---:|
| SA | 8 restarts, 2000 steps | 85 | 0.988372 | 1.04s |
| SA | 16 restarts, 2000 steps | 85 | 0.988372 | 2.08s |
| SA | 32 restarts, 2000 steps | 85 | 0.988372 | 4.14s |
| SA | 64 restarts, 2000 steps | 85 | 0.988372 | 8.29s |

在这个图上，纯 SA 明显强于 random+greedy，但 64 次重启仍没有达到 C*=86。

## SQNN + SA-Guided Escape

实验脚本：

```text
scripts/run_maxcut_escape_to_cstar_probe.py
```

核心机制：

```text
SQNN 连续 RZ/RY 演化
-> 平台期检测
-> 从当前 direct readout 出发做短程 SA
-> 把 SA 找到的离散方向写回 Bloch z 状态
-> SQNN 继续演化
```

结果：

| variant | config | best direct C | best direct C/C* | best expected C/C* | kicks | trial seconds |
|---|---|---:|---:|---:|---:|---:|
| monotone SQNN | no escape | 82 | 0.953488 | 0.831226 | 0 | 4.02s |
| full SA-guided escape | SA can run during training + final | 86 | 1.000000 | 0.998353 | 3 | 63.41s |
| fast final-only SA, 1 kick | train monotone, final escape only | 84 | 0.976744 | 0.973872 | 1 | 4.07s |
| fast final-only SA, max 3 kicks | train monotone, final escape only | 86 | 1.000000 | 0.998299 | 2 | 11.19s |
| cascade+success-cache | greedy first, full SA fallback | 86 | 1.000000 | 0.998239 | 3 | 7.01s |
| active-set+cascade+success-cache | greedy first, active SA fallback | 86 | 1.000000 | 0.997948 | 3 | 4.08s |

关键结论：

```text
1. full SA-guided escape 能达到 C*，但太慢。
2. SA 放在每个 epoch 的 forward 里性价比低，因为 SA 是非可微离散模块，对参数梯度贡献有限。
3. 更合理的做法是：训练阶段保持纯 SQNN/monotone，最终评估或少数 late rounds 才触发 SA escape。
4. fast final-only max-3-kick 在该图上保留了 C*=86，同时把单 trial 时间从约 63s 降到约 11s。
5. cascade+success-cache 可继续降到约 7s；active-set+cascade+success-cache 可降到约 4s。
```

最新 active/cascade/cache 消融：

| variant | direct C/C* | expected C/C* | SA calls | cache hits | cascade hits | active size mean | trial seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| active-set SA only | 0.976744 | 0.974358 | 3 | 0 | 0 | 32 | 4.41s |
| cascade+success-cache, full SA fallback | 1.000000 | 0.998239 | 22 | 0 | 1 | 0 | 7.01s |
| active-set+cascade+success-cache | 1.000000 | 0.997948 | 1 | 0 | 2 | 32 | 4.08s |

解释：

```text
1. active-set SA 单独速度很快，但容易因为搜索空间太小而停在 84/86；
2. cascade 单独可行，说明 greedy-guided escape 能过滤掉一部分贵的 SA；
3. cache 只能缓存成功改善 direct 解的 SA 结果，不能缓存失败结果；
4. active-set + cascade 的组合最好：
   greedy 能解决的先用 greedy；
   greedy 解决不了时，只对 32 个活跃变量做 SA。
```

## Speedup Directions

优先级最高的工程改法：

```text
1. SA final-only:
   不在训练 epoch 内反复跑 SA，只在最终 forward/推理阶段触发。

2. late escape:
   只在中后期触发，例如 round >= 40 或 round >= 0.6 * total_rounds。

3. max kicks:
   限制每次 forward 最多 1 到 3 次 SA。

4. cascade:
   先用便宜的 greedy-guided escape；
   如果 direct/readout 仍卡住，再触发 SA-guided escape。
```

更大的算法加速空间：

```text
1. MaxCut-3 incremental SA:
   当前通用 QUBO SA 每步重新扫 flip delta。
   对 3-regular MaxCut 可以维护每个点的 flip gain，
   翻一个点后只更新它和邻居，复杂度从每步 O(|E|) 降到 O(degree)。

2. active-set SA:
   只在低置信变量或局部场矛盾变量上退火，
   高置信变量固定，减少有效 n。

3. batched restarts:
   多个 SA restart 并行，适合 GPU 或向量化 CPU 实现。

4. cache escape:
   如果 direct assignment 没有明显变化，不重复跑 SA。
   注意只缓存成功改善 direct 解的 SA 结果；
   失败结果不能缓存，否则会阻断后续随机重试。
```

当前建议：

```text
把 SA 作为 classical baseline 和 V15 hybrid escape 候选机制保留；
不要把它混入纯 V10/V14 主指标。
若汇报，需要分开写：
  pure SQNN
  classical SA
  SQNN + SA-guided escape
```
