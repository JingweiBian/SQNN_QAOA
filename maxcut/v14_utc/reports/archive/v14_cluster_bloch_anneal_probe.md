# V14 Bad-Edge Cluster Bloch Anneal Probe

Script:

```text
scripts/run_v14_cluster_bloch_anneal_search.py
```

Purpose:

```text
Test the requested escape mechanism:
1. trigger annealing near a plateau;
2. build bad-edge clusters from currently uncut MaxCut edges;
3. apply a coherent alternating Bloch RY field to a local conflicted cluster;
4. allow a short non-monotone V14 recovery window, typically 8 or 16 rounds;
5. return to ordinary monotone accept after the recovery window.
```

Mechanism:

```text
A bad edge is an edge whose two endpoints currently have the same 0/1 readout.
Bad edges define a local frustrated subgraph.  The script finds connected
components in this bad-edge subgraph, caps the number of active cluster nodes,
and gives each selected cluster an alternating direction field.

The field is written into Bloch space as an RY perturbation.  It is not used as
the final solution.  V14 continues evolving after the perturbation.

The non-monotone recovery window accepts V14 proposals even when expected
energy temporarily worsens.  This prevents immediate monotone rollback from
erasing the jump before the state can reorganize.
```

Important implementation details:

```text
1. Giant bad-edge components are capped.
   A first version accidentally allowed a 469-node bad-edge component to be
   selected as one cluster.  The current version only keeps the highest-priority
   local part of an oversized component.

2. Two cluster bases were tested:
   direct basis: build bad-edge clusters from the current direct readout.
   greedy basis: first greedily repair the direct readout, then build bad-edge
   clusters from that local basin and write the target back into Bloch space.

3. Greedy basis is only used to generate a perturbation direction.
   The reported final trajectory still comes from continued V14 dynamics.
```

Reference results on n=512, degree=3, seed=0:

```text
base V14                 C_dg = 694, C_d = 688, C_exp = 671.374
known random RY          C_dg = 697, C_d = 692, C_exp = 682.140
previous soft global     C_dg = 702, C_d = 701, C_exp = 687.945
```

Cluster Bloch results:

| output dir | trials | best C_dg | best C_d | best C_exp | mean C_dg | C_dg >= 700 | C_dg >= 702 | C_dg >= 705 | mean case time | fastest 700 | fastest 702 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `outputs/v14_cluster_bloch_capped_n512_seed0` | 160 | 701 | 698 | 681.858 | 689.14 | 2 | 0 | 0 | 1.268s | 0.822s | - |
| `outputs/v14_cluster_bloch_best_replay80_n512_seed0` | 80 | 701 | 699 | 683.267 | 693.61 | 2 | 0 | 0 | 1.124s | 0.694s | - |
| `outputs/v14_cluster_bloch_greedybasis_n512_seed0` | 140 | 702 | 696 | 680.815 | 691.70 | 1 | 1 | 0 | 1.120s | 0.779s | 0.815s |
| `outputs/v14_cluster_bloch_greedybasis_702_replay100_n512_seed0` | 100 | 702 | 688 | 587.727 | 689.25 | 2 | 2 | 0 | 1.152s | 0.799s | 0.809s |

Best observed cluster cases:

```text
direct-basis capped:
best C_dg = 701
best C_d = 699
best C_exp = 683.267
fastest C_dg >= 700 = 0.694s

greedy-basis cluster:
best C_dg = 702
fastest C_dg >= 702 = 0.809s
best C_exp is much worse in the 702 replay, showing that this mode improves
direct+greedy basin quality more than it improves the probability-energy state.
```

Plots:

```text
outputs/v14_cluster_bloch_capped_n512_seed0/plots/top_cluster_bloch_cases.png
outputs/v14_cluster_bloch_capped_n512_seed0/plots/best_cluster_bloch_trace.png
outputs/v14_cluster_bloch_capped_n512_seed0/plots/time_to_target.png

outputs/v14_cluster_bloch_greedybasis_n512_seed0/plots/top_cluster_bloch_cases.png
outputs/v14_cluster_bloch_greedybasis_n512_seed0/plots/best_cluster_bloch_trace.png
outputs/v14_cluster_bloch_greedybasis_n512_seed0/plots/time_to_target.png
```

Current judgement:

```text
The requested mechanism is implemented and fast enough.
Typical case time is about 1.1s to 1.3s on CPU.

The non-monotone recovery window is worth keeping.  It allows the model to
retain basin jumps that monotone accept would otherwise erase immediately.

Bad-edge cluster Bloch anneal reaches 701 with a direct basis and 702 with a
greedy basis, but it does not reach 705 in the current scans.  The best 702
cases are not yet satisfying as a pure probability-state improvement because
expected cut can collapse while direct+greedy improves.

Next refinement should make the cluster field less destructive:
1. use expected-edge conflict instead of hard direct bad edges;
2. make non-monotone recovery bounded rather than unconditional;
3. learn or adapt the cluster strength after the first few recovery rounds;
4. only accept the post-recovery state if expected cut or direct cut recovers,
   otherwise roll back to the best state inside the recovery window.
```
