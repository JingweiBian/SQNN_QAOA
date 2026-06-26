# Dynamic Probabilistic Reheating Design Notes

Date: 2026-06-24

This note organizes the current reasoning about probability-driven basin
escape for V14 MaxCut.

## Current Mechanisms

### Soft Global Bloch Anneal

The older soft global anneal uses hard-readout features:

```text
bits_i = 1[p_i >= 0.5]
bad edge = bits_i == bits_j
```

It then gives selected nodes a clear flip-like direction:

```text
if bit_i = 0, push toward 1
if bit_i = 1, push toward 0
```

So although the perturbation is soft and global, its direction is close to:

```text
push selected nodes across the p=0.5 boundary
```

This is why it can sometimes reach the 702-level basin.

### SPBE v1

SPBE v1 replaces hard bad-edge features with probability conflict:

```text
q_ij = P[x_i == x_j]
     = p_i p_j + (1 - p_i)(1 - p_j)
```

With Bloch convention `p_i = (1 - z_i) / 2`, this is also:

```text
q_ij = (1 + z_i z_j) / 2
```

SPBE v1 is GPU-batched and clean, but it mostly refines the probability state
near the V14 basin.  It does not reliably cross into a new basin because it
lacks a recovery stage and a strong flip direction.

### Probabilistic Global Reheating

PGR adds:

```text
probability conflict q_ij
global push toward |+>
temporary non-monotone recovery
bounded rollback
```

It can move beyond the base V14 basin:

```text
base V14 direct+greedy: 694
PGR best observed:      698
```

But it is still weaker than the old soft global 702 result.

## Why PGR Is Weaker Than Soft Global

The main issue is not only node selection.  The perturbation direction changed.

Soft global:

```text
selected node -> push toward opposite hard-readout side
```

PGR:

```text
selected node -> mostly push toward |+>
```

`|+>` corresponds to:

```text
z = 0
p = 0.5
```

So PGR mainly melts nodes into uncertainty.  This increases exploration, but it
does not specify which side the node should settle on.

The second issue is that raw `q_ij` is too smooth.  Directly averaging it:

```text
node_conflict_i = average_j q_ij
```

causes broad reheating.  In observed runs, `rho_mean` can be around `0.84` to
`0.99`, meaning almost the whole graph is heated.

The third issue is symmetry.  `q_ij` says only whether two endpoints are likely
on the same side.  It does not say which endpoint should move, or in which
direction.

## What q_ij Should And Should Not Do

`q_ij` is a good spatial conflict observable:

```text
q_ij < 0.5  edge likely cut
q_ij = 0.5  uncertain / neutral
q_ij > 0.5  edge likely uncut
```

But it should not directly be used as a node reheating strength.

Instead:

```text
q_ij tells us where the current probability state has conflict.
time dynamics tells us whether that conflict is stuck.
neighbor/local fields tell us which direction to push.
```

## Sharpening q_ij

The raw conflict should be thresholded and sharpened:

```text
conflict_ij = relu((q_ij - tau) / (1 - tau))^gamma
```

Recommended starting values:

```text
tau   = max(0.55, quantile(q_ij, 0.70))
gamma = 2 or 3
```

This does three things:

```text
q_ij <= tau contributes nothing
medium uncertainty is suppressed
high same-side probability is amplified
```

For MaxCut-3, node aggregation should not be a plain mean.  A sharper aggregator
is better:

```text
edge_pressure_i =
    max_j conflict_ij
  + 0.5 * second_largest_j conflict_ij
  + 0.25 * third_largest_j conflict_ij
```

This preserves a single severe bad edge instead of diluting it by degree.

## Direction Is Not In q_ij

Because `q_ij` is symmetric, it cannot define the push direction.  Direction
should use the current Bloch `z` field, weighted by sharpened conflict:

```text
neighbor_field_i = sum_j conflict_ij * z_j
```

If conflicted neighbors mostly have `z_j > 0`, node `i` should move toward
`z_i < 0`.  If neighbors mostly have `z_j < 0`, node `i` should move toward
`z_i > 0`.

Continuous version:

```text
flip_direction_i = -tanh(k * neighbor_field_i)
direction_strength_i = tanh(k * |neighbor_field_i|)
```

Then:

```text
clear neighbor direction -> push opposite across boundary
mixed neighbor direction -> do not force a side; use weak reheating/noise
```

## Dynamics: Stuck Is Different From Conflict

A node can be conflicted without being stuck.  A node can also be stuck near
`p=0.5` without being high-confidence.  These cases need different treatment.

### High-Confidence Stuck

Definition:

```text
|z_i| high
recent velocity low
nearby sharpened conflict high
```

Important caveat:

```text
high |z_i| alone is not a problem.
It may simply be a good, already-decided node.
Only high confidence aligned with bad-edge conflict should be pushed.
```

Treatment:

```text
use conflict-weighted neighbor field
push opposite across the p=0.5 boundary
use little |+> reheating
```

### Neutral Stuck

Definition:

```text
|z_i| near 0
p_i near 0.5
recent velocity low
```

This node is already near `|+>`, so pushing it further toward `|+>` is useless.
It needs symmetry breaking.

Priority order:

```text
1. If neighbors give a clear conflict-weighted direction, use that direction.
2. Else use V14 local field direction.
3. Else use small local noise only on neutral-stuck nodes.
```

### Mixed Neighbor Direction

If:

```text
sum_j conflict_ij * z_j approx 0
```

then the edge conflict does not give a clear side.  Do not hard push.  Use
small reheating/noise and let recovery choose the basin.

## Proposed Next Mechanism

The next version should separate spatial conflict, temporal stuckness, and push
direction:

```text
1. Track recent z history over a window K.

2. Compute sharpened edge conflict:
   conflict_ij = relu((q_ij - tau) / (1 - tau))^gamma

3. Aggregate conflict sharply per node:
   edge_pressure_i = max/top-k over incident conflict_ij

4. Compute stuck gates:
   velocity_i = mean_recent |z_i(t) - z_i(t-1)|
   high_conf_stuck_i = high(|z_i|) * low(velocity_i) * edge_pressure_i
   neutral_stuck_i   = low(|z_i|)  * low(velocity_i)

5. Compute direction:
   neighbor_field_i = sum_j conflict_ij * z_j
   neighbor_direction_i = -tanh(k * neighbor_field_i)
   local_field_direction_i = calibrated sign from V14 local_field_i

6. Apply intervention by type:
   high-confidence stuck:
       strong boundary-crossing RY push using neighbor_direction_i

   neutral stuck with clear neighbors:
       symmetry-breaking push using neighbor_direction_i

   neutral stuck without clear neighbors:
       local-field push, then small noise if local field is weak

   mixed direction:
       weak |+> reheating only, no hard side choice

7. Run bounded V14 recovery and rollback if recovery does not improve from a
   strong checkpoint.
```

## Guarded Trigger Update

The runner now adds three guards before a probabilistic reheating event is
allowed:

```text
1. plateau trigger by default, not a fixed round trigger;
2. sharpened conflict must pass a structure threshold:
      max(conflict_ij) > threshold
   or top-k mean(conflict_ij) > threshold;
3. rollback compares recovery with a historical strong checkpoint, not only
   with the immediate pre-event state.
```

This changes the role of the trigger:

```text
plateau says: the trajectory has stopped improving.
structure says: there is a localized probability conflict worth acting on.
strong rollback says: do not keep reheating damage unless recovery reaches a
state comparable to the best known internal checkpoint.
```

First n=512 guarded scan:

```text
outputs/v14_dynamic_prob_reheating_n512_seed0_guarded_scan
```

Result:

```text
8 cases, 19.15s on cuda:0
base V14 direct+greedy: 694
guarded PGR best direct+greedy: 694
base V14 C[p]: 671.373
guarded PGR best C[p]: 671.401
```

Interpretation:

```text
The guards prevent early destructive reheating and preserve the V14 basin.
They do not yet create a better basin jump.
In the guarded scan, accepted events had real sharpened conflict, but
high_conf_stuck and neutral_stuck were near zero, so the current stuck detector
is too conservative or miscalibrated.
```

## Cluster-Level Local Reheating Trial

The runner now also supports local cluster reheating.  Instead of applying a
soft global perturbation everywhere, each accepted event can:

```text
1. score high-confidence and neutral stuck channels separately;
2. select seed nodes from one focus: high / neutral / mixed;
3. expand the seed set along sharpened conflict edges into a small cluster;
4. apply stronger RY dynamics only on that cluster;
5. keep the rest of the graph fixed except through the later V14 recovery.
```

The stuck score was also softened:

```text
low_response_i =
    drive_strength_i / (drive_strength_i + normalized_velocity_i)
```

so a node is treated as stuck when it has drive but responds weakly, instead of
requiring a hard near-zero velocity threshold.

Focused n=512 result:

```text
outputs/v14_cluster_dynamic_reheating_n512_seed0_focused
```

```text
24 cases, 66.85s on cuda:0
base V14 direct+greedy: 694
cluster PGR best direct+greedy: 694
base V14 direct: 687
cluster PGR best direct: 691
base V14 C[p]: 671.373
cluster PGR best C[p]: 675.557
```

Interpretation:

```text
cluster-local dynamics improves the continuous probability state and raw direct
readout, but it still falls back into the same greedy basin.

mixed/neutral clusters are the useful branch so far.
high-confidence clusters did not help, consistent with the concern that many
high-confidence nodes are already good and should not be pushed.
```

## Post-Checkpoint Phase-Cluster Trial

A second branch tested the idea:

```text
run clean V14 to a strong internal checkpoint
then perform phase-aware cluster escape from that checkpoint
```

This branch also tested XY-plane annealing:

```text
none:
    no explicit XY treatment

dephase_xplus:
    pull cluster XY components toward +X while preserving Z

rz_noise:
    random RZ phase noise on the cluster

xy_reset:
    reset cluster XY phase to +X while preserving Z

dephase_rz:
    dephase toward +X plus RZ phase noise

xy_shrink:
    damp the Y component only
```

It also tested temporary memory suppression:

```text
active memory freeze:
    decay phase_memory / edge_message / edge_z_message on cluster edges
    during reheating and early recovery
```

Focused n=512 result:

```text
outputs/v14_post_checkpoint_phase_cluster_n512_seed0_scan1
```

```text
24 cases, 58.36s on cuda:0
base V14 direct+greedy: 694
post-checkpoint best direct+greedy: 694
base V14 direct: 687
post-checkpoint best direct: 687
base V14 C[p]: 671.373
post-checkpoint best C[p]: 673.237
```

Interpretation:

```text
XY dephasing can improve C[p] slightly, with dephase_rz best in this scan.
However, post-checkpoint escape did not improve raw direct or direct+greedy.
The fully optimized V14 checkpoint appears too rigid; the better window is
probably late plateau before final lock-in, around the earlier in-trajectory
cluster events.
```

## Late-Plateau Phase-Cluster Trial

The next scan moved the same phase-aware cluster escape earlier:

```text
V14 late plateau, before full phase/memory lock-in
local mixed/neutral cluster reheating
explicit XY-plane annealing
temporary active-edge memory suppression
```

Focused n=512 result:

```text
outputs/v14_late_plateau_phase_cluster_n512_seed0_scan1
```

```text
32 cases, 67.89s on cuda:0
base V14 direct+greedy: 694
late-plateau phase-cluster best direct+greedy: 694
base V14 direct: 687
late-plateau phase-cluster best direct: 690
base V14 C[p]: 671.373
late-plateau phase-cluster best C[p]: 676.657
accepted recovery events: 23
```

The best C[p] case used:

```text
mixed cluster focus
xy_reset
active memory freeze for 12 rounds with factor 0.0
trigger around round 262
cluster size about 31 nodes
```

Interpretation:

```text
This is better than post-checkpoint escape and slightly better than the
previous cluster-local scan on C[p].

XY reset and dephase_rz are the useful XY branches in this scan.  RZ noise alone
is weak.  Short active memory suppression helps, while fully waiting for V14
convergence is too late.

However, direct+greedy still stays at 694.  The current escape operator can
improve the continuous probability state and raw direct readout, but the
discrete readout plus greedy repair still returns to the same basin.
```

## Weak Global Plus + Local Phase Cleanup

The next implementation separates the global and local parts of the escape:

```text
weak global |+> reheating:
    a small background push, scored by conflict or low-response conflict

local cluster directional push:
    stronger RY direction only on a conflicted subgraph

XY phase cleanup:
    reset/dephase the local cluster, optionally with a few high plus_score nodes

short memory freeze:
    suppress active-edge memory immediately after the jump
```

The key correction is:

```text
global |+> push should not mainly multiply ambiguity.
Nodes near p = 0.5 are already close to |+>.
The useful weak global push should target conflicted or locked nodes.
```

Implementation knobs:

```text
plus_mode = legacy / conflict / locked_conflict / high_conflict / uniform
plus_alpha_cap = maximum weak global |+> step
xy_scope = cluster / cluster_plus / plus / global
xy_plus_fraction = fraction of top plus_score nodes added to XY cleanup
```

n=512 focused scans:

```text
outputs/v14_weak_global_phase_cluster_n512_seed0_scan1
    best C[p]: 674.425
    best direct: 691
    best direct+greedy: 694

outputs/v14_ultraweak_global_phase_cluster_n512_seed0_scan1
    best C[p]: 677.080
    best direct: 691
    best direct+greedy: 694

outputs/v14_cluster_plus_xy_global_phase_n512_seed0_scan1
    best C[p]: 677.481
    best direct: 691
    best direct+greedy: 694
```

Interpretation:

```text
The model is likely reaching nearby basins in the continuous probability
landscape, but those basins are still close enough that greedy repair maps them
to the same 694 discrete solution.

Ultraweak global reheating helps only when alpha is tiny, around 0.005 to 0.02.
Broader uniform/high-confidence global pushes are destructive.

Cleaning XY phase outside the cluster is possible, but broad cluster_plus
cleanup did not improve over cluster-only cleanup.  The useful regime is likely
sparse phase cleanup on a very small set of locked conflict nodes.
```

## Boundary Pulse Trial

The next trial added a bounded RY pulse intended to cross the hard-readout
boundary without directly assigning bits:

```text
selected locked/conflict nodes
    -> continuous RY pulse
    -> optional XY cleanup
    -> short memory freeze
    -> V14 recovery
```

This explicitly tests whether the missing ingredient is boundary-crossing
strength.

Results:

```text
outputs/v14_boundary_pulse_n512_seed0_scan1
    best C[p]: 677.691
    mean C[p]: 659.365
    best direct: 691
    best direct+greedy: 694

outputs/v14_boundary_pulse_cluster_focused_n512_seed0_scan1
    best C[p]: 677.058
    mean C[p]: 675.234
    best direct: 691
    best direct+greedy: 694
```

Interpretation:

```text
The effect is real but not yet the desired escape.

Global boundary pulses are too destructive.
Cluster-only boundary pulses are much safer.
The best focused cases usually did not actually cross z=0; they behaved as
near-boundary nudges.
Cases that forced many nodes across z=0 often damaged C[p].

Therefore the current problem is not simply "pulse harder".
The model still lacks a sufficiently accurate gate selector for which locked
nodes should truly cross the readout boundary.
```

## Gate Selector Diagnostic

A dedicated diagnostic runner was added:

```text
maxcut/scripts/run_v14_gate_selector_diagnostic.py
scripts/run_v14_gate_selector_diagnostic.py
```

It compares dynamic selectors against hard-readout oracle features, then tests
both a group-flip counterfactual and the actual continuous Bloch pulse.

Key n=512 seed-0 output:

```text
outputs/v14_gate_selector_diagnostic_n512_seed0_by_start
```

Observed selector quality:

```text
bad_edge / oracle_gain:
    oracle overlap about 0.93 to 1.00
    selected positive-gain nodes about 9 / 10
    best C[p] about 675
    best direct+greedy remains mostly 694

locked_conflict / node_conflict:
    oracle overlap about 0.025
    selected positive-gain nodes about 0 / 10
    group-flip greedy delta 0
    final greedy basin almost unchanged

random:
    one case reached direct+greedy 695
    but with much lower C[p], so this is escape noise rather than controlled
    optimization
```

Interpretation:

```text
There are two separate bottlenecks.

1. Dynamic conflict selectors are not yet finding the nodes that hard-readout
   flip-gain diagnostics consider useful.

2. Even oracle-selected useful nodes usually only improve the immediate direct
   cut; after greedy repair they fall back to the same basin.
```

So the next useful search should not simply increase pulse strength.  It should
first learn or design a better gate criterion:

```text
gate_i should predict "this node participates in a coordinated basin move",
not merely "this node has high q_ij" or "this node has low velocity".

Candidate labels for diagnosis:
    group_flip_greedy_delta > 0
    final_greedy_hamming_from_base large but C[p] not destroyed
    recovery improves direct+greedy without large expected-cut damage
```

## Short Takeaway

The right decomposition is:

```text
q_ij / sharpened conflict:
    where is the probability distribution structurally bad?

stuck/velocity history:
    is this bad structure dynamically trapped?

neighbor z / local field:
    in which direction should the state be pushed?

|+> reheating:
    only for loosening ambiguous or mixed-direction regions, not as the main
    flip direction.
```

This keeps the probabilistic model-side picture while restoring the directional
strength that made the older soft global anneal more effective.
