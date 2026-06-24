# V14 Soft Global Bloch Anneal Probe

This probe tests the user's proposed direction:

```text
global annealing, but conflicted/uncertain nodes anneal more strongly
```

The implementation is in:

```text
scripts/run_v14_soft_global_anneal_search.py
```

Unlike Q-tabu anneal, this version does not use branch lookahead or selected
branch continuation.  Every run is a single continuous V14/Bloch trajectory.

## Dynamical Form

At an anneal round, every node receives an additional RY annealing field:

```text
theta_i =
  temperature * envelope(t) * rho_i * directed_flip_i
+ temperature * noise * envelope(t) * rho_i * gaussian_i
+ memory_strength * memory_i
```

where `rho_i` is a continuous escape susceptibility:

```text
rho_i = global_floor + weighted_conflict_score_i
```

The score uses:

```text
bad-edge endpoint strength
positive one-flip gain
cheap negative move indicator
low confidence
near-best gain rank
```

Optional transverse reheating pulls Bloch vectors slightly toward `|+>`, again
with node-dependent strength `rho_i`.

The important conceptual difference from the previous Q-tabu script is:

```text
Q-tabu: perturb candidates, short-run branches, pick one.
Soft global: one continuous trajectory, no branch selection.
```

## Baselines On n=512 Seed 0

| method | best expected C | best C_d | best C_dg |
|---|---:|---:|---:|
| base V14 | 671.374 | 688 | 694 |
| known random RY replay | 682.140 | 692 | 697 |
| previous wider Bloch scan | 682.735 | 695 | 699 |
| Q-tabu Bloch best | 684.380 | 697 | 700 |
| soft global best | 687.945 | 701 | 702 |

## Best Soft Global Configuration

Best replay label:

```text
soft0024_fixed_s145_w8_linear_cool_t0.80_g0.60_n0.20_floor0.06_tr0.02_zs0.00_mem0.85-0.35-0.00
```

Configuration:

| field | value |
|---|---:|
| trigger | fixed round |
| trigger round | 145 |
| anneal window | 8 rounds |
| envelope | linear_cool |
| temperature | 0.80 |
| guidance | 0.60 |
| noise | 0.20 |
| global floor | 0.06 |
| transverse reheating | 0.02 |
| z shrink | 0.00 |
| memory decay | 0.85 |
| memory inject | 0.35 |
| memory angle strength | 0.00 |
| Metropolis temperature | 0.06 |
| clear aux | active top 5% rho nodes |

Result:

| metric | value |
|---|---:|
| best C_dg | 702 |
| best C_d | 701 |
| best expected C | 687.945 |
| event count | 1 |
| case time | about 1.2 s on CPU |

Output directory:

```text
outputs/v14_soft_global_702_replay80_n512_seed0
```

Important plots:

```text
outputs/v14_soft_global_702_replay80_n512_seed0/plots/top_soft_global_cases.png
outputs/v14_soft_global_702_replay80_n512_seed0/plots/best_soft_global_trace.png
outputs/v14_soft_global_702_replay80_n512_seed0/plots/best_by_envelope.png
```

## Search Summary

| output dir | trials | best C_dg | best C_d | best expected C | notes |
|---|---:|---:|---:|---:|---|
| `outputs/v14_soft_global_focused_n512_seed0` | 140 | 702 | 698 | 682.613 | broad scan |
| `outputs/v14_soft_global_refine_n512_seed0` | 160 | 700 | 698 | 681.553 | narrow scan |
| `outputs/v14_soft_global_702_replay80_n512_seed0` | 80 | 702 | 701 | 687.945 | fixed best parameter, different noise seeds |

In the 80-run replay:

```text
mean C_dg      = 694.56
C_dg >= 700    = 7 / 80
C_dg >= 702    = 2 / 80
best C_d       = 701
best C_dg      = 702
```

The best result is stochastic, not deterministic.  The same dynamical rule can
produce a broad range of outcomes depending on the annealing noise realization.

## Interpretation

Soft global anneal is currently the best SQNN/Bloch-side escape mechanism on
the n=512 seed0 test:

```text
base V14        694 C_dg
random Bloch    699 C_dg
Q-tabu Bloch    700 C_dg
soft global     702 C_dg
```

This is also cleaner theoretically than Q-tabu:

```text
no classical tabu trajectory
no branch selection
no direct local-search optimizer
single continuous Bloch trajectory
```

It still has not reached 705.  The main limitation is reliability: high-quality
escapes appear in a minority of noise realizations.  The next step should be to
make the annealing field less hand-random and more adaptive, for example by
learning `rho_i`, the trigger time, and the temperature schedule.
