# V14 Q-Tabu Bloch Anneal Probe

This probe tests whether tabu-search ideas can improve V14 without using
classical local search as the optimizer.  The new script is
`scripts/run_v14_qtabu_anneal_search.py`.

## Mechanism

The Q-tabu anneal controller adds five inference-time mechanisms:

1. plateau/fixed trigger: trigger escape around stagnation or selected rounds;
2. gain-aware active set: score nodes by bad-edge conflict, positive one-flip
   gain, cheap negative gain, and low confidence;
3. bad-edge cluster / qtabu-random selection: perturb a small conflict-biased
   subset instead of all nodes;
4. short no-return memory: after an RY kick, keep a weak bias against immediate
   rollback for a few rounds;
5. branch lookahead: try several Bloch kicks, short-run V14, then continue from
   the best branch state.

The final state is still produced by V14/Bloch evolution.  `C_dg` is only the
diagnostic direct-readout-plus-greedy metric, not the optimization engine.

## Baselines On n=512 Seed 0

| method | best expected C | best C_d | best C_dg |
|---|---:|---:|---:|
| base V14 | 671.374 | 688 | 694 |
| known random RY replay | 682.140 | 692 | 697 |
| previous wider Bloch scan | 682.735 | 695 | 699 |
| Q-tabu best replay | 684.380 | 697 | 700 |

The best replay output is in:

`outputs/v14_qtabu_700_replay_n512_seed0`

Important plots:

* `outputs/v14_qtabu_700_replay_n512_seed0/plots/top_qtabu_cases.png`
* `outputs/v14_qtabu_700_replay_n512_seed0/plots/best_qtabu_trace.png`
* `outputs/v14_qtabu_700_replay_n512_seed0/plots/best_by_selector.png`

## Best Configuration Found

Best label:

`qtabu0008_both_qtabu_random_s145-160_f0.040_t0.80_g1.30_n0.10_nr4x0.14`

Configuration:

| field | value |
|---|---:|
| trigger | both plateau and fixed |
| fixed starts | 145, 160 |
| selector | qtabu_random |
| active fraction | 0.04 |
| active nodes | about 20 / 512 |
| RY temperature | 0.80 |
| guidance | 1.30 |
| noise | 0.10 |
| no-return tenure | 4 rounds |
| no-return strength | 0.14 |
| branch count | 8 |
| branch horizon | 12 rounds |
| branch score | mixed |
| Metropolis temperature | 0.05 |

Result:

| metric | value |
|---|---:|
| best C_dg | 700 |
| best C_d | 697 |
| best expected C | 684.380 |
| events | 2 |
| case time | about 1.9 s on CPU |

## Search Runs

| output dir | trials | best C_dg | best C_d | best expected C | note |
|---|---:|---:|---:|---:|---|
| `outputs/v14_qtabu_focused_n512_seed0` | 80 | 698 | 694 | 679.414 | broad mechanism scan |
| `outputs/v14_qtabu_badcluster_refine_n512_seed0` | 100 | 699 | 695 | 681.266 | cluster/random refinement |
| `outputs/v14_qtabu_random_refine_n512_seed0` | 80 | 700 | 695 | 681.511 | first 700 hit |
| `outputs/v14_qtabu_700_refine_n512_seed0` | 120 | 700 | 697 | 684.380 | refined best |
| `outputs/v14_qtabu_700_replay_n512_seed0` | 8 | 700 | 697 | 684.380 | clean replay |

## Interpretation

The new mechanism improves the best observed V14/Bloch result:

* base V14: 694 `C_dg`
* previous Bloch anneal: 699 `C_dg`
* Q-tabu Bloch anneal: 700 `C_dg`

The improvement is real but still far from the earlier classical tabu target
around 705.  The best direction is not hard deterministic gain flipping.
Deterministic `bad_gain` and `cheap_gain` often damage the continuous SQNN
state.  The strongest route is a conflict-biased randomized active set with
mild no-return memory and branch lookahead.

The same fixed parameter setting is still stochastic.  In the clean replay,
one of eight branch-random seeds reached 700; the others mostly landed between
685 and 695.  This means the mechanism can find better basins, but it is not
yet a reliable deterministic escape rule.

## Current Conclusion

Q-tabu anneal is worth keeping as a research direction.  It gives a
Bloch-side jump from 699 to 700 and raises direct readout to 697, without using
tabu/local search as the final optimizer.

It does not yet reach 705.  Further hand scanning is likely inefficient.  The
next promising step is to make the escape controller trainable or adaptive:
learn when to trigger, which nodes to perturb, and how strong the no-return
memory should be for each instance.
