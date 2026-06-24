# V14 + 跳盆新起点评估

实验对象：`n=512`、3-regular MaxCut、`seed=0`，总边数 `W=768`。

这次评估不把 tabu / break 的输出直接当最终答案，而是按如下流程：

`V14 readout -> tabu/break 生成新起点 -> 写回 soft probability -> V14 重新演化 -> 再读出`

## 结论

当前实验没有发现 `C>705`。

- `tabu` 作为新起点：可以让 V14 复现或恢复到 `C=705`。
- `random break` / `bad-edge break` 作为新起点：最高只恢复到 `C=703` 或 `C=695` 左右，没有接近 705。
- `bad-edge soften`：把未切边附近节点写成低置信度后，结果反而下降，最高 `704`。

所以目前更准确的说法是：

> tabu 可以给 V14 一个新的强起点，V14 能保持或恢复到 705；但还没有证据表明 V14 从这个新起点继续演化后能突破 705。

## 关键数据

### 1. tabu 新起点 + V14 re-evolve

输出：`outputs/maxcut512_v14_reevolve_escape_seed0_full/summary.csv`

最高结果：

| 起点 | tabu 后 C | V14 best direct | V14 best direct+greedy | V14 best sample | best expected |
|---|---:|---:|---:|---:|---:|
| `v14_direct_r269_tabu_5s_conf0.97_physical` | 705 | 705 | 705 | 705 | 700.875 |
| `v14_direct_r269_tabu_5s_conf0.90_physical` | 705 | 705 | 705 | 705 | 697.407 |
| `v14_direct_r269_tabu_5s_conf0.75_physical` | 705 | 705 | 705 | 705 | 696.602 |

低置信写回时，round 0 并不是好解，但 V14 后续轮次可以把 direct+greedy 拉回 705：

| case | round0 direct | round0 direct+greedy | re-evolve best direct | re-evolve best direct+greedy |
|---|---:|---:|---:|---:|
| `v14_direct_r269_tabu_5s_conf0.55_physical` | 449 | 700 | 705 | 705 |
| `v14_direct_greedy_r270_tabu_2s_conf0.55_physical` | 441 | 696 | 704 | 705 |

### 2. 未切边 softening

输出：`outputs/maxcut512_v14_reevolve_escape_seed0_softbad/summary.csv`

最高：

- `best direct = 704`
- `best direct+greedy = 704`
- `best sample = 704`

这个策略暂时不保留为主线。

### 3. raw random break

输出：`outputs/maxcut512_v14_reevolve_rawbreak_seed0/summary.csv`

最高：

- `best direct = 703`
- `best direct+greedy = 703`
- `best sample = 702`

raw break 可以验证 V14 有一定恢复能力，但不能作为突破 705 的有效路线。

## 判断

tabu 新起点有价值，但它目前更像是：

1. 给出一个强 basin；
2. V14 在这个 basin 内维持/平滑 expected；
3. direct+greedy 最多回到 705。

下一步如果要突破 705，应该让 re-evolve 阶段不是简单复用原 V14 固定参数，而是加入“重演化时的可训练/可扰动自由度”，例如：

- 只在跳盆后继续优化后半段 RY/RZ 参数；
- 对 705 解的未切边邻域做局部可训练 soft block；
- 多个 tabu 起点作为 multi-head warm starts，让 V14 学会融合而不是单起点重跑。
