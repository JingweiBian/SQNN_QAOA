# Probabilistic Global Reheating

Date: 2026-06-24

This note records the first integrated version of the user's proposed mechanism:

```text
probability-score nodes -> global reheating toward |+>
-> allow temporary bad states -> bounded V14 recovery
```

Implementation:

```text
maxcut/scripts/run_v14_probabilistic_global_reheating.py
scripts/run_v14_probabilistic_global_reheating.py
```

This runner is different from the first SPBE batch runner.  It runs the V14
trajectory from the beginning and inserts the reheating event inside the V14
loop, so phase memory and edge messages are preserved through the recovery
window.

## Mechanism

At a trigger round, the runner scores the current probability state by soft
MaxCut conflict:

```text
q_ij = P[x_i == x_j]
     = p_i p_j + (1 - p_i)(1 - p_j)
```

For each node:

```text
node_conflict_i = sum_j q_ij / degree_i
entropy_i       = 4 p_i (1 - p_i)
pressure_i      = pressure_guidance * sum_j z_j / degree_i
                + cluster_strength  * sum_j q_ij z_j / degree_i

rho_i = rho_floor
      + (1 - rho_floor) * normalize(
          conflict_weight * node_conflict_i
        + entropy_weight  * entropy_i
        + pressure_weight * |pressure_i|
        )^rho_power
```

The reheating step applies two effects:

```text
1. RY perturbation:
   theta_i = temperature * envelope(t) * rho_i * pressure_i
           + temperature * noise * envelope(t) * rho_i * gaussian_i
           + memory_strength * memory_i

2. global push toward |+>:
   bloch_i <- (1 - alpha_i) * bloch_i + alpha_i * |+>
   alpha_i = plus_strength * temperature * envelope(t) * rho_i
```

After reheating, a short recovery window uses ordinary V14 proposals with a
Metropolis-like non-monotone accept rule.  At the end of the window the runner
keeps the best recovered state only if it recovers the pre-event expected cut
within tolerance or recovers the direct cut; otherwise it rolls back.

## Relation To Previous Soft Global Anneal

They are similar in spirit:

```text
both trigger around plateau regions
both perturb Bloch states globally
both need temporary non-monotone recovery
```

The difference is the signal:

```text
old soft global: hard readout bad edges and flip directions
PGR: probability conflict q_ij and reheating toward |+>
```

So PGR is closer to probability-state dynamics, while the previous soft global
anneal was more readout-driven.

## n=512 Seed 0 Results

Baseline V14:

| metric | value |
|---|---:|
| best C[p] | 671.373 |
| best direct C | 688 |
| best direct+greedy C | 694 |

### Broad Scan

Output:

```text
outputs/v14_prob_global_reheating_n512_seed0_scan1
```

| metric | value |
|---|---:|
| cases | 32 |
| wall time | 72.01s |
| mean case time | 2.15s |
| best direct+greedy C | 696 |
| best direct C | 691 |
| best C[p] | 678.317 |
| cases with C_dg >= 694 | 4 / 32 |
| cases with C_dg >= 696 | 2 / 32 |

### Focused Scan

Output:

```text
outputs/v14_prob_global_reheating_n512_seed0_focused
```

| metric | value |
|---|---:|
| cases | 24 |
| wall time | 54.12s |
| mean case time | 2.13s |
| best direct+greedy C | 698 |
| best direct C | 690 |
| best C[p] | 672.953 |
| cases with C_dg >= 694 | 6 / 24 |
| cases with C_dg >= 696 | 3 / 24 |
| cases with C_dg >= 698 | 1 / 24 |

Best focused label:

```text
pgr0012_both_s230_w4_r48_cosine_cool_t0.45_plus0.20_pg0.20_c0.00_n0.18_floor0.05_rt0.05
```

### Best-Config Replay

Output:

```text
outputs/v14_prob_global_reheating_n512_seed0_replay_best698
```

| metric | value |
|---|---:|
| cases | 24 |
| wall time | 54.86s |
| mean case time | 2.16s |
| best direct+greedy C | 698 |
| best direct C | 691 |
| best C[p] | 668.349 |
| cases with C_dg >= 694 | 6 / 24 |
| cases with C_dg >= 696 | 2 / 24 |
| cases with C_dg >= 698 | 1 / 24 |

## Interpretation

PGR does show real basin movement:

```text
base V14 direct+greedy: 694
PGR broad scan best:    696
PGR focused/replay:     698
```

The key positive result is that the probability-driven global reheating plus
bounded recovery can beat the base V14 readout basin.  The recovery stage is
important; pure SPBE without V14 recovery stayed around 694.

The key limitation is reliability:

```text
best 698 appears in 1 / 24 focused cases
best 698 appears in 1 / 24 replay cases
mean replay C_dg is 690
```

This is not yet competitive with the previous soft global anneal peak of 702.
PGR currently creates basin movement, but it does not reliably land in the
high-quality basin.

## Current Diagnosis

The mechanism is probably too destructive early and too weakly selective late:

```text
early reheating can lower C[p] substantially;
late recovery can repair the state, but only some noise paths land well;
the |+> push increases exploration, but does not yet encode enough coordinated
multi-node direction to target the 700+ basin reliably.
```

The next useful refinement is stricter event acceptance plus learned or adaptive
temperature:

```text
1. trigger only when C[p] is already near the V14 plateau;
2. accept a reheating event only if recovery improves either C[p] or direct C
   relative to a strong checkpoint, not just relative to a weak intermediate;
3. tune or learn rho_i and plus_strength from successful recovery paths;
4. keep the probability conflict q_ij signal, because it is the right
   model-side observable.
```

## Guarded Dynamic Update

The current runner has been updated with the following safeguards:

```text
default trigger mode: plateau
required structure gate:
    max(sharpened conflict_ij) > threshold
 or top-k mean(sharpened conflict_ij) > threshold
rollback reference:
    historical strong checkpoint instead of the immediate weak pre-event state
```

This protects against the failure mode where plateau is detected while
`q_ij` conflict is essentially zero.

Guarded n=512 smoke/scan output:

```text
outputs/v14_dynamic_prob_reheating_n512_seed0_guarded_scan
```

| metric | value |
|---|---:|
| cases | 8 |
| wall time | 19.15s |
| accepted events | 3 |
| base V14 best C[p] | 671.373 |
| guarded best C[p] | 671.401 |
| base V14 best direct C | 687 |
| guarded best direct C | 687 |
| base V14 best direct+greedy C | 694 |
| guarded best direct+greedy C | 694 |

The safeguards fix the early-damage problem, but they also show that the
current dynamic stuck gates are not yet identifying useful intervention nodes.
In the accepted guarded events, structure conflict was real, but
`stuck_high_mean` and `stuck_neutral_mean` were near zero.  The next design
question is therefore not whether high-confidence nodes should be pushed in
general; they should not.  The question is how to identify high-confidence
nodes that are both dynamically stuck and structurally implicated in bad
conflict.

## Cluster-Local Dynamic Reheating

The latest trial adds a local cluster mode:

```text
focus = high / neutral / mixed
seed nodes = top stuck scores for the selected focus
cluster = seeds expanded along sharpened conflict edges
outside cluster = frozen during reheating
recovery = ordinary V14 non-monotone recovery with strong rollback
```

This is intended to test the hypothesis that a basin should be escaped by
reorganizing a small conflicted subgraph, not by pushing one node or heating the
whole graph.

Focused n=512 output:

```text
outputs/v14_cluster_dynamic_reheating_n512_seed0_focused
```

| metric | value |
|---|---:|
| cases | 24 |
| wall time | 66.85s |
| accepted events | 15 |
| base V14 best C[p] | 671.373 |
| cluster best C[p] | 675.557 |
| base V14 best direct C | 687 |
| cluster best direct C | 691 |
| base V14 best direct+greedy C | 694 |
| cluster best direct+greedy C | 694 |

The cluster mechanism is therefore a partial improvement:

```text
C[p] improves by +4.18
raw direct readout improves by +4
direct+greedy does not improve
```

The useful cases are mostly `mixed` focus, but the selected cluster score is
dominated by the neutral channel.  This supports separating the two stuck types:
the current high-confidence branch is not yet helpful, while neutral/mixed
clusters can reshape the probability state.

## Post-Checkpoint Phase Escape

A follow-up tested two additional hypotheses:

```text
1. wait until V14 has reached a strong internal checkpoint, then escape;
2. anneal XY-plane phase and temporarily suppress memory, not only Z/probability.
```

The tested XY modes were:

```text
none, dephase_xplus, rz_noise, xy_reset, dephase_rz, xy_shrink
```

Output:

```text
outputs/v14_post_checkpoint_phase_cluster_n512_seed0_scan1
```

| metric | value |
|---|---:|
| cases | 24 |
| wall time | 58.36s |
| accepted events | 24 |
| base V14 best C[p] | 671.373 |
| post-checkpoint best C[p] | 673.237 |
| base V14 best direct C | 687 |
| post-checkpoint best direct C | 687 |
| base V14 best direct+greedy C | 694 |
| post-checkpoint best direct+greedy C | 694 |

`dephase_rz` was the best XY mode for C[p] in this scan, but no XY/memory
setting improved direct readout or direct+greedy.  This suggests that fully
optimized V14 checkpoints are already too locked in for the current escape
operator.  The in-trajectory late-plateau cluster branch remains more promising
because it improved raw direct C to 691.

## Late-Plateau Phase Escape

A second phase-aware scan moved the intervention earlier, into the V14 late
plateau before the fully optimized checkpoint:

```text
outputs/v14_late_plateau_phase_cluster_n512_seed0_scan1
```

| metric | value |
|---|---:|
| cases | 32 |
| wall time | 67.89s |
| accepted events | 23 |
| base V14 best C[p] | 671.373 |
| late-plateau best C[p] | 676.657 |
| base V14 best direct C | 687 |
| late-plateau best direct C | 690 |
| base V14 best direct+greedy C | 694 |
| late-plateau best direct+greedy C | 694 |

Best C[p] label:

```text
pgr0024_trajectory_plateau_s130_w6_r64_pulse_t0.60_plus0.05_pg1.20_c0.30_n0.12_floor0.00_mixed_xy_reset_rt0.15
```

The event diagnostics show:

```text
trigger round: 262
cluster size: 31 nodes
structure max conflict: 0.164
top-k structure conflict mean: 0.0418
low-response mean at trigger: 0.528
best recovery direct+greedy: 694
```

This supports the current working diagnosis:

```text
late-plateau intervention is better than post-checkpoint intervention;
XY reset / dephase_rz can improve the continuous probability state;
short active-edge memory suppression is useful;
the remaining bottleneck is discrete basin transfer after readout.
```

## Weak Global Background Reheating

The runner now separates two effects that were previously entangled:

```text
local cluster directional push:
    still restricted to the selected conflicted cluster

weak global |+> reheating:
    can use a separate plus_score and is no longer masked out by
    cluster_outside_scale = 0
```

New plus modes:

```text
legacy:
    old ambiguity-weighted behavior

conflict:
    weak global |+> push weighted by sharpened node conflict

locked_conflict:
    weak global |+> push weighted by conflict and low response

high_conflict:
    weak global |+> push weighted by conflict and |z|

uniform:
    whole-graph background push
```

The important implementation change is that non-legacy global reheating no
longer mainly multiplies `ambiguity`.  This avoids spending the |+> push mostly
on nodes that are already near `p = 0.5`.

### Focused Global Scan

Output:

```text
outputs/v14_weak_global_phase_cluster_n512_seed0_scan1
```

| metric | value |
|---|---:|
| cases | 48 |
| best C[p] | 674.425 |
| best direct C | 691 |
| best direct+greedy C | 694 |

Diagnosis:

```text
uniform and high_conflict global pushes are too destructive;
locked_conflict is the safest non-legacy global mode;
alpha around 0.03 to 0.06 is already too large for this trajectory.
```

### Ultraweak Global Scan

Output:

```text
outputs/v14_ultraweak_global_phase_cluster_n512_seed0_scan1
```

| metric | value |
|---|---:|
| cases | 48 |
| best C[p] | 677.080 |
| best direct C | 691 |
| best direct+greedy C | 694 |

Best C[p] case:

```text
locked_conflict plus mode
plus_strength = 0.01
max plus alpha about 0.005
mixed cluster
dephase_rz
```

Interpretation:

```text
very weak global |+> reheating improves the continuous probability state beyond
the previous late-plateau result, but it still does not produce a farther
discrete basin after greedy repair.
```

### Cluster-Plus XY Scope

The runner also supports:

```text
xy_scope = cluster:
    clean XY phase only on the local cluster

xy_scope = cluster_plus:
    clean XY phase on the local cluster plus top plus_score nodes
```

Output:

```text
outputs/v14_cluster_plus_xy_global_phase_n512_seed0_scan1
```

| metric | value |
|---|---:|
| cases | 36 |
| best C[p] | 677.481 |
| best direct C | 691 |
| best direct+greedy C | 694 |

In this scan `cluster_plus` increased XY active nodes from about 25 to about 51
on average, but it did not improve direct+greedy and was slightly worse on mean
C[p] than cluster-only XY.  This suggests that phase cleanup outside the
cluster should be very sparse, not broad.

## Locked-Conflict Boundary Pulse

The runner now also supports a more explicitly boundary-crossing but still
dynamical intervention:

```text
boundary_pulse_mode = locked_conflict / conflict / stuck_high / stuck_neutral
boundary_scope = cluster / global
boundary_fraction = top fraction of selected nodes
boundary_strength = RY pulse scale
boundary_target_z = desired depth across the z=0 readout boundary
boundary_angle_cap = maximum RY pulse angle
boundary_direction_mode = cross / field / hybrid
```

This is not a hard bit flip.  It applies a bounded RY pulse to selected
locked/conflicted nodes and records how many actually cross `z=0`.

Broad scan:

```text
outputs/v14_boundary_pulse_n512_seed0_scan1
```

| metric | value |
|---|---:|
| cases | 60 |
| best C[p] | 677.691 |
| mean C[p] | 659.365 |
| best direct C | 691 |
| best direct+greedy C | 694 |

Focused cluster-only scan:

```text
outputs/v14_boundary_pulse_cluster_focused_n512_seed0_scan1
```

| metric | value |
|---|---:|
| cases | 40 |
| best C[p] | 677.058 |
| mean C[p] | 675.234 |
| best direct C | 691 |
| best direct+greedy C | 694 |

Diagnosis:

```text
global boundary pulses are too destructive;
cluster-scoped pulses preserve the probability state much better;
forcing many nodes to cross z=0 usually hurts C[p];
the best focused cases mostly had zero actual crossings, acting instead as
near-boundary nudges.
```

So the current boundary pulse confirms the hypothesis that stronger
boundary-crossing force is needed, but the current locked/conflict selector is
not yet accurate enough to safely decide which nodes should truly cross.

## Gate Selector Diagnostic

Diagnostic runner:

```text
maxcut/scripts/run_v14_gate_selector_diagnostic.py
scripts/run_v14_gate_selector_diagnostic.py
```

This runner isolates three questions:

```text
1. Does a selector hit hard-readout oracle-useful nodes?
2. If those nodes are flipped as a group, does the greedy basin move?
3. If those nodes receive a continuous Bloch pulse, does V14 recovery keep
   the new basin?
```

Main n=512 seed-0 output:

```text
outputs/v14_gate_selector_diagnostic_n512_seed0_by_start
```

| metric | value |
|---|---:|
| cases | 80 |
| wall time | 130.98s |
| base best C[p] | 671.373 |
| base best direct C | 687 |
| base best direct+greedy C | 694 |
| diagnostic best C[p] | 675.040 |
| diagnostic best direct C | 692 |
| diagnostic best direct+greedy C | 695 |

Selector-level diagnosis:

| selector | mean oracle overlap | max group-flip greedy delta | mean selected positive-gain nodes | max final greedy hamming |
|---|---:|---:|---:|---:|
| bad_edge | 1.000 | 1 | 9.000 | 15 |
| oracle_gain | 0.925 | 1 | 9.000 | 15 |
| locked_conflict | 0.025 | 0 | 0.000 | 2 |
| node_conflict | 0.025 | 0 | 0.000 | 2 |
| random | 0.006 | 1 | 0.375 | 20 |

Interpretation:

```text
The current locked_conflict/node_conflict selectors almost never select the
hard-readout nodes that have positive flip gain.

Oracle/bad-edge selectors improve C[p] and sometimes direct C, but even their
group flips usually recover to the same direct+greedy basin.

The only 695 direct+greedy case in this scan came from random+hybrid.  It moved
the basin slightly, but with much lower C[p], so random noise can escape locally
but is not a controlled optimization mechanism.
```

This means the next modeling bottleneck is gate selection and basin-move
qualification, not merely stronger pulses.

## Readout-Guided Soft Global Retry

The older hard-readout guided soft anneal path was rerun after fixing a CUDA
generator/device mismatch in `scripts/run_v14_bloch_anneal_escape.py`.

Outputs:

```text
outputs/v14_readout_guided_soft_global_retry_n512_seed0
outputs/v14_readout_guided_soft_global_focused2_n512_seed0
```

Combined n=512 seed-0 result:

| metric | value |
|---|---:|
| cases | 160 |
| base V14 direct+greedy C | 694 |
| best readout-guided direct+greedy C | 699 |
| best readout-guided direct C | 694 |
| best readout-guided C[p] | 680.691 |
| cases with direct+greedy C = 699 | 4 |
| cases with direct+greedy C >= 698 | 12 |

Best observed cases:

```text
soft0045 fixed starts 145,160 window 20 cosine:
    direct+greedy 699, direct 693, C[p] 664.565

soft0022 fixed starts 160,230 window 12 linear:
    direct+greedy 699, direct 683, C[p] 667.567

soft0043 both start 175 window 8 linear:
    direct+greedy 698, direct 694, C[p] 678.156

soft0073 fixed starts 145,190 window 20 pulse:
    direct+greedy 695, direct 693, C[p] 680.691
```

Interpretation:

```text
Hard-readout guidance clearly improves basin movement relative to purely
probability/dynamic selectors.

The price is instability: many aggressive events lower C[p] badly, and the
best direct+greedy cases do not necessarily preserve the best probability
state.

So this is a useful performance route and a useful teacher signal, but it
should be wrapped in stronger checkpoint/rollback logic before being treated
as a stable model mechanism.
```

### Guarded Readout-Guided Scan

The readout-guided runner now supports event guarding:

```text
--guard-events
--guard-accept any / expected / quality / strict
--guard-recovery-rounds
--guard-max-expected-drop
--guard-min-direct-gain
--guard-min-dg-gain
```

Implementation detail:

```text
each event saves Bloch/probability/phase-memory checkpoint;
after the recovery window, failed events roll back and their bad trace segment
is overwritten so summary metrics do not count rejected spikes.
```

Guarded output:

```text
outputs/v14_readout_guided_soft_global_guarded_quality_n512_seed0
```

| metric | unguarded 160 cases | guarded 48 cases |
|---|---:|---:|
| best direct+greedy C | 699 | 698 |
| best direct C | 694 | 696 |
| best C[p] | 680.691 | 681.943 |
| direct+greedy >= 698 | 12 / 160 | 2 / 48 |
| direct >= 696 | 0 / 160 | 1 / 48 |
| direct >= 700 | 0 / 160 | 0 / 48 |

Guarded event statistics:

```text
accepted events: 74
rejected events: 9
```

Interpretation:

```text
The guard improves stability and raises the best direct readout from 694 to
696, but it also lowers peak direct+greedy from 699 to 698.

There is still no evidence that this mechanism directly reaches C=700+ by
plain p>=0.5 readout on n=512 seed 0.

The current guard compares against the event trigger state, so early weak
triggers are still too permissive.  The next stability fix should compare
against a historical strong checkpoint, not only the immediate pre-event state.
```

### Strong-Checkpoint Guard

The readout-guided runner now also supports historical checkpoint references:

```text
--guard-reference event / strong_expected / strong_direct / strong_dg / strong_quality
--require-strong-checkpoint
--strong-checkpoint-min-round
--strong-checkpoint-min-expected
```

Smoke behavior:

```text
fixed start 145 with --require-strong-checkpoint
    -> skipped as no_strong_checkpoint
```

This confirms that early events can be blocked until the trajectory has reached
a stronger state.

Corrected late-start scan:

```text
outputs/v14_readout_guided_soft_global_strong_checkpoint_late_corrected_n512_seed0
```

| metric | value |
|---|---:|
| cases | 16 |
| start range | 220-270 |
| best direct+greedy C | 694 |
| best direct C | 691 |
| best C[p] | 678.151 |
| skipped events | 0 |

Mid-window strong-checkpoint scan:

```text
outputs/v14_readout_guided_soft_global_strong_checkpoint_mid_n512_seed0
```

| metric | value |
|---|---:|
| cases | 24 |
| start range | 170-210 |
| best direct+greedy C | 698 |
| best direct C | 695 |
| best C[p] | 677.998 |
| direct+greedy >= 696 | 5 / 24 |
| direct+greedy >= 698 | 1 / 24 |
| direct >= 700 | 0 / 24 |

Interpretation:

```text
145 is too early if the guard reference is only the trigger state.

But 220+ strong-checkpoint starts are too conservative: the V14 trajectory is
already close to locked, so readout-guided annealing mostly preserves the base
694 basin.

The mid-window scan supports the middle-ground hypothesis.  Starts around
170-210 can still move the basin while avoiding the weakest 145-type triggers.
The best observed mid-window case reached direct+greedy 698 and direct 695.

There is still no evidence of direct C >= 700.  Even with readout-guided
annealing and strong-checkpoint guard, the direct readout remains below the
700-level target in these scans.
```

## Useful Commands

Focused scan:

```bash
python scripts/run_v14_probabilistic_global_reheating.py \
  --device cuda:0 \
  --trials 24 \
  --trigger-modes both,fixed \
  --fixed-starts 145,175,210,220,230,240 \
  --fixed-start-counts 1,2 \
  --reheat-windows 3,4,5,6 \
  --recovery-rounds 36,48,64 \
  --envelopes linear_cool,cosine_cool \
  --temperatures 0.25,0.35,0.45 \
  --plus-strengths 0.10,0.15,0.20 \
  --pressure-guidances 0.0,0.10,0.20 \
  --cluster-strengths 0.0,0.2 \
  --noises 0.10,0.18,0.25 \
  --recovery-temperatures 0.05,0.08,0.12 \
  --rho-floors 0.03,0.05,0.08 \
  --rollback-tolerance 0.25 \
  --max-events 2 \
  --score-stride 1 \
  --output-dir outputs/v14_prob_global_reheating_n512_seed0_focused
```

Gate-selector diagnostic:

```bash
python scripts/run_v14_gate_selector_diagnostic.py \
  --device cuda:0 \
  --start-rounds 246,254,262,270 \
  --selectors oracle_gain,bad_edge,locked_conflict,node_conflict,random \
  --directions oracle_flip,hybrid \
  --fractions 0.02 \
  --strengths 0.4,0.8 \
  --angle-caps 0.60 \
  --xy-modes xy_reset \
  --window 6 \
  --recovery-rounds 64 \
  --metropolis-temperature 0.10 \
  --clear-aux active \
  --score-stride 2 \
  --output-dir outputs/v14_gate_selector_diagnostic_n512_seed0_by_start
```

Best-config replay:

```bash
python scripts/run_v14_probabilistic_global_reheating.py \
  --device cuda:0 \
  --trials 24 \
  --trigger-modes both \
  --fixed-starts 230 \
  --fixed-start-counts 1 \
  --min-starts 160 \
  --plateau-rounds 10 \
  --cooldowns 80 \
  --max-events 2 \
  --reheat-windows 4 \
  --recovery-rounds 48 \
  --envelopes cosine_cool \
  --temperatures 0.45 \
  --plus-strengths 0.20 \
  --pressure-guidances 0.20 \
  --cluster-strengths 0.0 \
  --noises 0.18 \
  --rho-floors 0.05 \
  --rho-powers 1.4 \
  --conflict-weights 1.4 \
  --entropy-weights 0.1 \
  --pressure-weights 0.2 \
  --memory-decays 0.85 \
  --memory-injects 0.35 \
  --memory-strengths 0.04 \
  --recovery-temperatures 0.05 \
  --recovery-slacks 0.02 \
  --rollback-tolerance 0.25 \
  --clear-aux none \
  --clear-fractions 0.03 \
  --score-stride 1 \
  --output-dir outputs/v14_prob_global_reheating_n512_seed0_replay_best698
```

## Readout-Guided Timing Scan

The readout-guided soft-global runner now has a deterministic timing scanner:

```text
scripts/run_v14_readout_guided_timing_scan.py
```

The scanner crosses start round, guard mode, template, and anneal repeat.  A
small bug in the first scan methodology was fixed: anneal seeds are now derived
stably from `base_seed + label + repeat`, rather than from the case order.  This
matters because changing the scan list otherwise changes the stochastic
trajectory for the same nominal timing.

### Broad Timing Scan

Output:

```text
outputs/v14_readout_guided_timing_scan_n512_seed0
```

| metric | value |
|---|---:|
| cases | 112 |
| best direct+greedy C | 700 |
| best direct C | 695 |
| best classical tabu C | 700 |
| best classical breakout C | 698 |

Best case:

```text
event_cosine_stable_s130
direct+greedy = 700
direct        = 695
C[p]          = 674.416
```

This scan indicated that the useful intervention window is earlier than the
late plateau.  The late/strong checkpoint window protects quality, but mostly
keeps the system in the same basin.  The early event window can move the basin,
but the result is stochastic.

### Early Stable-Seed Repeat Scan

Output:

```text
outputs/v14_readout_guided_timing_scan_early_repeats_n512_seed0
```

| metric | value |
|---|---:|
| cases | 168 |
| repeats per template/start | 3 |
| best direct+greedy C | 700 |
| best direct C | 696 |
| best classical tabu C in this window | 694 |
| best classical breakout C in this window | 684 |

Best case:

```text
event_late_nudge_s100_r2
direct+greedy = 700
direct        = 696
C[p]          = 686.252
anneal seed   = 1883512151
```

Best-by-start summary:

| start | best SQNN d+g | best SQNN direct | best classical tabu | base readout+greedy |
|---:|---:|---:|---:|---:|
| 70 | 699 | 695 | 694 | 659 |
| 90 | 699 | 693 | 693 | 656 |
| 100 | 700 | 696 | 687 | 656 |
| 140 | 700 | 691 | 690 | 660 |
| 160 | 699 | 692 | 693 | 660 |

Interpretation:

```text
best intervention window: roughly rounds 90-120, with 100 the strongest point
best mechanism: short weak nudge with local auxiliary memory clearing
main gain over classical: the state is perturbed early and then V14 continues
                         evolving; classical local search only sees an early,
                         immature bit string
```

The strongest result is not just greedy repair.  The best case reaches direct
readout 696, above the previous direct ceiling of 695 in these guarded timing
probes.  However direct readout still does not reach 700.

### Focused Manual Refinement

Output:

```text
outputs/v14_readout_guided_timing_scan_focused_n512_seed0
```

This tested small variants around the best early weak-nudge picture:

```text
less noise
more phase/memory clearing
slightly longer window
direct-readout-biased scoring
```

| metric | value |
|---|---:|
| cases | 160 |
| repeats per template/start | 4 |
| best direct+greedy C | 699 |
| best direct C | 693 |
| best focused template | early_nudge_longer_s140 |

The manual variants did not beat the original `late_nudge_s100_r2` result.
This rules out a few simple explanations:

```text
less noise is too conservative;
more clearing stabilizes but washes out useful phase/memory information;
direct-biased scoring improves raw readout a little but reduces basin movement;
longer weak windows can help around round 90, but not around round 100.
```

Current conclusion:

```text
700 is reachable on n=512 seed 0, but it is not yet stable.
The best route is early, short, weak, partly stochastic basin movement followed
by ordinary V14 recovery.  The next improvement should not be a stronger single
kick.  It should either:

1. run a small portfolio of early weak-nudge repeats and keep the best readout;
2. make the model train through these early perturb/recover events so that the
   recovery path becomes less seed-dependent;
3. learn or adapt the event strength from recovery success, instead of hand
   setting noise/clear/memory knobs.
```

## Phase/Z Transition-Window Scan

The next diagnostic records a full V14 trajectory with cumulative phase and
Z-basis information, then scans annealing around detected readout transition
windows.

Implementation:

```text
scripts/run_v14_transition_phase_anneal_scan.py
```

Main outputs:

```text
outputs/v14_transition_phase_anneal_scan_diag_n512_seed0
outputs/v14_transition_phase_anneal_scan_n512_seed0
outputs/v14_transition_phase_anneal_scan_focus_s139_n512_seed0
outputs/v14_transition_phase_anneal_scan_focus_early_triple_n512_seed0
outputs/v14_transition_phase_anneal_combined_n512_seed0
```

The diagnostic records, per round:

```text
cumulative phase:     sum_t phase_angle_trace[t]
phase-step size:      mean |phase_angle_trace[t]|
Z information:        mean |z|, z-step RMS, z-step mean absolute motion
readout softness:     count(|p_i - 0.5| < eps)
readout transition:   bit flips from previous round and direct-cut jumps
```

Detected transition windows on n=512 seed 0:

| window | interpretation |
|---|---|
| 12-20 | first decoding transition / symmetry breaking |
| 52, 72-73, 122 | early small readout rearrangements |
| 169-172 | major basin/readout transition |
| 181-197 | second major basin/readout rearrangement |
| 202-220 | late soft rearrangement, already closer to lock-in |
| 242+ | polarized late regime, mostly too late for useful basin escape |

The phase/Z relation is:

```text
Useful transition windows occur while mean |z| is still small and many nodes
remain near p=0.5.  Late windows can still show direct-readout jumps, but mean
|z| is already rising and near-threshold nodes are disappearing, so annealing
mostly perturbs an already locked basin.
```

Broad transition-window scan:

```text
outputs/v14_transition_phase_anneal_scan_n512_seed0
```

| metric | value |
|---|---:|
| cases | 238 |
| best direct+greedy C | 703 |
| best direct C | 699 |
| best C[p] | 686.495 |

Best single-window path:

```text
start = 139
template = cosine_stable
target transition = event 6, peak around 169
direct+greedy = 703
direct = 697
C[p] = 682.771
```

This is about 30 rounds before the 169-172 basin/readout transition.  It
supports the idea that the useful intervention point is the front edge of a
transition, not the peak after the readout has already rearranged.

Best multi-window direct path:

```text
starts = 10,32,52
template = late_nudge
direct+greedy = 701
direct = 699
C[p] = 686.495
```

Focused upper-bound repeats:

| scan | cases | best direct+greedy C | best direct C |
|---|---:|---:|---:|
| broad transition scan | 238 | 703 | 699 |
| focus around 129-149 before event 6 | 100 | 703 | 698 |
| focus early triple 10,32,52 | 93 | 701 | 699 |

Combined output:

```text
outputs/v14_transition_phase_anneal_combined_n512_seed0
```

Current upper-bound estimate from 431 transition-window anneal paths:

```text
direct+greedy upper bound observed: 703
direct readout upper bound observed: 699
```

Interpretation:

```text
1. Fixed round 100 was useful because it lies in a broad soft-transition
   regime, but it was not the best window.
2. The best single jump is around round 139, about 30 rounds before the
   169-172 readout/basin transition.
3. Repeated annealing is not monotonically better.  Early weak repeated
   annealing can raise direct readout to 699, but repeated mid/late cosine
   annealing often damages C[p].
4. The next model rule should trigger at transition front edges:
      many near-threshold nodes,
      small mean |z|,
      emerging bit-flip/direct-jump spike,
      before late polarization starts.
```
