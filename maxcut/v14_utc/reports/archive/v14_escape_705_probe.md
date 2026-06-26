# V14 + 跳盆机制 512 变量结果记录

实验对象：`n=512`、3-regular MaxCut、`seed=0`，总边数 `W=768`。

## 结论

V14 加跳盆可以到 `C=705`，对应 `C/W=0.917969`。
只使用 V14 起点时，`30s` 搜索预算已经能首次到 `705`；`10s` 预算最高到 `703`。

目前没有把 `705` 往上推到 `706+`。尝试过：

- V14-only tabu / breakout / penalty-breakout portfolio：最高 `705`。
- 15min V14+随机/GW 混合 portfolio：最高 `705`。
- 从 V14-only `705` assignment 出发的 local branching：在半径 `16,24,32,48,64,96` 内均未找到 `706`。
- `dwave-tabu`、`dwave-neal`、大采样 GW+greedy、CP-SAT target feasibility、RC2 MaxSAT：本轮都没有返回 `706+`。

所以当前可复现实验结论是：V14+跳盆能把 `694` 的 V14 direct+greedy 起点推到 `705`，但 `705 -> 706` 是新的硬瓶颈。

## V14-only 速度

| 搜索预算 | 最高 C | 首次到 705 的累计搜索时间 | 触发方法 |
|---:|---:|---:|---|
| 10s | 703 | 未到 | - |
| 30s | 705 | 5.4s | `v14_direct_greedy_r270_tabu_full_t12` |
| 60s | 705 | 50.0s | `v14_direct_r269_breakout35_intense_active45` |
| 120s | 705 | 95.7s | `v14_direct_greedy_r270_breakout35_intense_active45` |

30s 是当前最干净的速度证据：V14 起点 `C_dg=694`，跳盆中一个 `1.35s` 的 tabu slice 找到 `705`，在完整 staged 流程中的累计搜索时间约 `5.4s`。

## 输出位置

- `outputs/maxcut512_v14_only_escape_30s/summary.csv`
- `outputs/maxcut512_v14_only_escape_60s/summary.csv`
- `outputs/maxcut512_v14_only_escape_120s/summary.csv`
- `outputs/maxcut512_v14_escape_portfolio_seed0_penalty_15min/summary.csv`
- `outputs/maxcut512_v14_only_705_local_branching_to706/attempts.csv`
