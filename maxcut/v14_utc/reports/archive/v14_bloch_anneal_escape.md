# V14 Bloch Anneal Escape Probe

Date: 2026-06-24

This report records the first scan of Bloch-space annealing escapes.  These
escapes are different from the previous phase-reset probe: they directly give
selected nodes a chance to change their Z readout basin, then let the trained
V14 dynamics continue.

## Operators tested

- `transverse`: pull selected Bloch vectors toward `|+> = (1,0,0)`.
- `depolarize`: shrink selected Bloch vectors toward the mixed state.
- `ry_kick`: apply a decaying random RY rotation, so selected nodes can cross
  `Z=0`.
- `mixed`: transverse reheating plus RY thermal kick.
- `metropolis`: no direct Bloch perturbation, but temporarily accept some worse
  expected-energy proposals during the anneal window.

Active nodes were selected from low-confidence nodes, bad-edge/low-confidence
nodes, or connected bad-edge clusters.  The final score is still the continued
V14 trajectory, not a classical local-search replacement.

## n=512, seed 0

Base V14:

- best expected cut: `671.374`
- best direct cut: `688`
- best direct+greedy cut: `694`

Focused scan:

- output: `outputs/v14_bloch_anneal_focused_n512_seed0`
- cases: `396`

Best by operator:

| operator | best direct+greedy | best direct | best expected |
|---|---:|---:|---:|
| `mixed` | `697` | `692` | `676.715` |
| `ry_kick` | `697` | `691` | `676.779` |
| `depolarize` | `695` | `688` | `652.834` |
| `transverse` | `694` | `692` | `674.189` |
| `metropolis` | `694` | `688` | `671.959` |

RY refinement:

- output: `outputs/v14_bloch_anneal_ry_refine_n512_seed0`
- cases: `288`
- best direct+greedy: `699`
- best direct: `697`
- best expected: `682.462`

Best case:

- `s160_w20_ry_kick_a0.00_t0.60_m0.10_bad_low_conf_f0.030_none_rep0`
- start round: `160`
- anneal window: `20`
- active selector: `bad_low_conf`
- active fraction: `3%`
- RY temperature: `0.60`
- Metropolis temperature: `0.10`

Top-region refinement:

- output: `outputs/v14_bloch_anneal_ry_topregion_n512_seed0`
- cases: `243`
- best direct+greedy stayed at `699`
- best direct stayed at `695`
- best expected improved slightly to `682.735`

## n=1024, seed 0

Base V14:

- best expected cut: `1333.346`
- best direct cut: `1369`
- best direct+greedy cut: `1379`

RY refinement:

- output: `outputs/v14_bloch_anneal_ry_refine_n1024_seed0`
- cases: `216`
- best direct+greedy: `1385`
- best direct: `1376`
- best expected: `1340.403`

Best direct+greedy case:

- `s160_w20_ry_kick_a0.00_t0.60_m0.15_bad_low_conf_f0.020_none_rep0`
- start round: `160`
- anneal window: `20`
- active selector: `bad_low_conf`
- active fraction: `2%`
- RY temperature: `0.60`
- Metropolis temperature: `0.15`

Best expected/direct case:

- `s160_w20_ry_kick_a0.00_t0.50_m0.15_bad_low_conf_f0.020_none_rep0`
- best direct+greedy: `1383`
- best direct: `1376`
- best expected: `1340.403`

## Interpretation

The useful escape is not phase clearing.  It is controlled RY reheating.

Why:

- RY directly rotates Bloch vectors through the Z axis, so it can change the
  eventual binary basin.
- Small active sets are better than large active sets.  The useful range was
  about `2%-4%` of nodes.
- The best timing is mid-trajectory, around round `155-165` for these V14
  settings.  Later kicks are usually weaker or destructive.
- Metropolis acceptance helps only when paired with a real RY kick.  By itself
  it barely moves the solution.
- Pure depolarization usually damages expected cut too much.
- Pure transverse reheating can improve expected/direct, but did not improve
  direct+greedy in this scan.

Current quality:

- n=512 improved from `694` to `699` direct+greedy.
- n=1024 improved from `1379` to `1385` direct+greedy.
- This is a real SQNN/Bloch-side improvement, but it is still below the strong
  classical tabu/breakout levels previously observed.

## Recommended next step

Keep `RY thermal kick` as the main quantum-driven jump-basin candidate.  The
next version should not be only an evaluation-time perturbation.  It should be
trained with an anneal window so V14 learns how to recover after selected nodes
cross Z=0.  A reasonable default schedule is:

- start around `0.55-0.60` of total rounds;
- window `20-25` rounds;
- active set from bad-edge plus low-confidence nodes;
- active fraction `2%-3%` for n=1024 and `3%` for n=512;
- RY temperature around `0.5-0.6`;
- Metropolis temperature around `0.10-0.15`;
- do not clear all auxiliary edge/phase memory.
