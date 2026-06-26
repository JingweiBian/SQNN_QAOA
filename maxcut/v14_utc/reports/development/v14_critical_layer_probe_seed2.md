# V14 Critical Layer Probe on Seed2

## Question

Seed2 does not naturally show a large V14 direct-readout collective transition.  Can we actively find and create a critical layer?

## New Runner

- `scripts/run_v14_critical_layer_escape_probe.py`

This is an inference-only diagnostic.  It does not train or backpropagate.

The runner:

1. runs trained V14 to a chosen round,
2. detects a locked-conflict cluster from direct-readout bad edges, flip gains, z-locking, and neighbor consistency,
3. locally compresses/dephases that layer in Bloch space,
4. continues V14 and measures near-threshold nodes, bit-flip peaks, direct cut, and direct+greedy cut.

## Outputs

- `outputs/v14_critical_layer_escape_probe_seed2/REPORT.md`
- `outputs/v14_critical_layer_escape_probe_seed2_strong_direction/REPORT.md`

## Baseline

Seed2 base V14:

- best direct+greedy: 683
- best direct: 674
- best expected cut: 664.954

## Main Finding

The detector can find a real locked-conflict layer, and the intervention can create a critical layer signal.

Examples:

- `plus` compression at round 80 produced up to 132 near-threshold nodes.
- `directional` push at round 80 produced up to 38 sampled bit flips.
- stronger `flip_soft` at rounds 140-160 produced about 101-102 sampled bit flips.

So the mechanism can create "many variables moving together"; seed2 is not physically impossible to move.

## But It Is Not Yet a Good Basin Transition

The best first-pass critical-layer result:

- weak/passive probe best direct+greedy: 684
- strong-direction probe best direct+greedy: 688

This is above base V14, but still below the local soft-monotone window result of 694.

Interpretation:

- We can find and disturb a coupled layer.
- We can manufacture near-threshold mass.
- We can manufacture large collective bit flips.
- But the selected layer/direction is not yet the correct collective coordinate for a high-quality basin.

The current detector finds a "movable locked-conflict layer", not necessarily the "right basin-boundary mode".

## Practical Implication

Critical-layer construction is real but incomplete.

The next useful step is not simply stronger push.  Stronger push produced larger flips but not better cuts.

The next step should estimate the direction from a local subproblem or multiple candidate condensations:

- detect locked-conflict layer,
- create the near-threshold layer,
- generate several local condensation directions on that layer,
- let V14 recover briefly,
- select by direct/direct+greedy or expected-cut guard.

This keeps the mechanism dynamical while adding a direction-selection step.
