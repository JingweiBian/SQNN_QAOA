# V14 Bloch Anneal Speed and 705 Probe

Date: 2026-06-24

This note answers whether the current Bloch/V14 anneal is fast, whether it can
be accelerated, and whether it can reach the previous 705-level result.

## Speed

Current implementation is serial CPU and evaluates one trajectory at a time.

Observed wall time:

- n=512 single Bloch anneal case: about `0.9-1.3s`.
- n=1024 single Bloch anneal case: about `1.2s`.
- n=512 300-trial random/multi-window search: about `298s`.
- n=512 exact restart probe with 51 cases: about `51s`.
- n=512 direct+greedy-warm restart probe with 75 cases: about `74s`.

This is fast per trajectory, but broad search is slow because cases are still
run serially and scored across all rounds.

## What was tried after the first 699 result

### Multi-window random and hybrid RY search

Output:

- `outputs/v14_bloch_multiwindow_random_search_n512_seed0`

Configuration:

- operators: `random_ry`, `hybrid_ry`
- one, two, or three anneal windows
- start rounds from `135` to `210`
- active fractions from `1.5%` to `5%`
- temperatures from `0.25` to `0.75`
- 300 trials

Result:

- best direct+greedy: `698`
- best direct: `696`
- best expected: `682.991`

Multi-window annealing did not improve on the previous best `699`.

### Guided RY search

Output:

- `outputs/v14_bloch_guided_smoke_n512_seed0`

Operators:

- `gain_ry`: RY direction chosen by positive one-flip gain.
- `bad_ry`: RY direction chosen by bad-edge endpoints.
- `hybrid_ry`: bad-edge guidance plus gain guidance plus noise.

Result:

- guided versions did not beat unguided random RY.
- The deterministic guidance often made the trajectory too biased and less
  exploratory.

### Exact 699 restart from direct readout

Output:

- `outputs/v14_bloch_exact_restart_direct_n512_seed0`

Method:

- reproduce known 699-ish Bloch anneal cases;
- take their best direct readout;
- write it back as a soft initial probability with confidences
  `0.60, 0.75, 0.90, 0.97`;
- run V14 or V14+anneal again.

Result:

- best direct+greedy: `699`
- best direct: `699`
- best expected: `692.132`

Direct warm restart preserves the basin but does not push it past `699`.

### Exact 699 restart from direct+greedy readout

Output:

- `outputs/v14_bloch_exact_restart_dgwarm_n512_seed0`

This uses the greedy-polished readout as warm start, so it is not a pure
quantum/V14-only advantage.  It was tested as a diagnostic.

Result:

- best direct+greedy: `699`
- best direct: `699`
- best expected: `694.861`

Even after writing back the greedy-polished 699 solution, V14+anneal did not
move to 705.

## Current conclusion

The current inference-time Bloch anneal improves V14, but it does not reach
705.

Reliable improvement:

- base V14 direct+greedy: `694`
- best Bloch anneal direct+greedy: `699`
- best Bloch anneal direct: `697`
- best restart direct: `699`

Observed ceiling in these probes:

- `699`, not `705`.

This suggests the current anneal is finding a better nearby basin, but not the
same basin that tabu/breakout found around `705`.

## Acceleration options

Low-risk engineering speedups:

- Run independent anneal cases in parallel processes.  The current search is
  serial; this should scale almost linearly with CPU cores.
- Use fast search scoring: score every 2-5 rounds during broad search, then
  fully rescore only top cases.
- Skip plots and full trace CSVs during random search; write only top summaries
  and rerun selected cases for plots.

Larger speedup:

- Batch multiple Bloch trajectories in one forward loop.  This would require
  making V14 state tensors `[batch, n, 3]` instead of `[n, 3]`, but it is the
  cleanest path to GPU/CPU vectorization.

## How to plausibly reach 705

Inference-time perturbation alone looks insufficient.  More promising routes:

1. Train with anneal injection.
   The model currently sees the kick only at evaluation time.  If anneal windows
   are inserted during training, V14 can learn recovery dynamics after nodes
   cross `Z=0`.

2. Learn the anneal policy.
   Make start round, active fraction, temperature, and possibly node scores
   trainable or scan-optimized on validation seeds.

3. Active-subproblem quantum refinement.
   Freeze high-confidence nodes, take a 20-60 node bad-edge active subproblem,
   and run a small SQNN/QAOA-style anneal on only that subproblem before writing
   it back.  This is still a model-side/quantum-inspired jump, but stronger
   than independent RY kicks.

4. Batched trajectory ensemble.
   Treat RY anneal as a sampling/readout family: run many cheap Bloch anneal
   trajectories in parallel and take the best direct/sample result.  This would
   be closer to the `C_s` metric than pure `C_d`.
