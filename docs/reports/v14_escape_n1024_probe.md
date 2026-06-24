# n=1024 V14 + 跳盆新起点评估

实验对象：`n=1024`、3-regular MaxCut、`seed=0`，总边数 `W=1536`。

本实验继续遵守这条定义：

`V14 readout -> tabu/break 只负责生成新起点 -> 写回 soft probability -> V14 re-evolve -> 再读出`

也就是说，跳盆结果本身不直接当作模型最终结果。

## 结论

1024 上，V14 + tabu 新起点可以从原始 `C_dg=1379` 推到 `1414`。

但和 only heuristic 对比：

- `only heuristic` 随机起点 tabu/break portfolio：最高 `1414`
- `V14 + tabu 新起点 + V14 re-evolve`：最高 `1414`

所以当前结论是：

> 1024 上 V14 起点能让跳盆更快到达同等质量解，但最终 best cut 暂时没有超过 only heuristic。

## 关键数值

| 方法 | C | C/W | 说明 |
|---|---:|---:|---|
| V14 expected | 1333.346 | 0.868064 | 原始 V14 最好 expected |
| V14 direct | 1369 | 0.891276 | 原始 V14 直接二值读出 |
| V14 direct+greedy | 1379 | 0.897786 | 原始 V14 直接读出加 greedy |
| V14 + tabu 新起点 + re-evolve | 1414 | 0.920573 | 多次 tabu 新起点，V14 再演化 |
| only heuristic | 1414 | 0.920573 | 随机 greedy 起点 + tabu/break portfolio |

## 速度对比

这不是严格 benchmark，只是按脚本执行顺序估算的首次命中时间：

| 方法 | 首次到 1414 的时间 |
|---|---:|
| V14 起点 + tabu 新起点 + V14 re-evolve | 约 27.1s |
| only heuristic 随机起点 portfolio | 约 51.6s |

V14 这边对应 case：

`v14_direct_r270_tabu_5s_rep1_conf0.97_physical`

- tabu 新起点 `C=1414`
- round0 写回后 `direct=1414, direct+greedy=1414`
- re-evolve 后仍保持 `direct=1414, direct+greedy=1414, sample=1414`
- expected 从 round0 的 `1336.381` 提升到最高 `1409.940`

因此这里 V14 的作用更像是：

1. 提供较好的初始 basin；
2. tabu 更快找到高质量新起点；
3. V14 re-evolve 保持该离散解，并显著提高 soft expected。

但它没有在离散割数上把 `1414` 推到 `1415+`。

## break 机制表现

随机 break / bad-edge break 在 1024 上暂时不如 tabu：

- raw bad-edge break 会把起点破坏到 `1251-1346` 左右，V14 能恢复到约 `1378-1380`；
- greedy-polished bad-edge break 最高也主要在 `1380-1382`；
- breakout 最高约 `1404`。

所以 1024 当前主线应保留：

`V14 -> short repeated tabu -> high-confidence soft write-back -> V14 re-evolve`

暂时不建议把 raw random break 作为主线跳盆机制。

## 输出位置

- V14 + tabu/break 完整组合：`outputs/maxcut1024_v14_reevolve_escape_seed0_full/`
- V14 + repeated tabu：`outputs/maxcut1024_v14_reevolve_escape_seed0_tabu_repeats/`
- only heuristic baseline：`outputs/maxcut1024_only_heuristic_random_seed0_120s/`
- 1024 V14 训练缓存：`outputs/v14_re_evolve_training_n1024_seed0/`
