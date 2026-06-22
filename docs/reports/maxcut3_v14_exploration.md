# MaxCut-3 V14 Exploration Report

## Scope

This report summarizes the current long-run exploration around random 3-regular MaxCut with SQNN/V14-style phase-aware dynamics.

The reported ratios use the current benchmark denominator `W`, so they are cut fractions `C/W`. They are not strict `C/C*` unless an exact optimum is substituted as the denominator.

## Current Best Picture

Across seeds `7, 11, 23, 42, 99`:

```text
direct+1-bit greedy mean C/W        = 0.897135
Bernoulli sample+1-bit greedy mean  = 0.898958
Bloch hyperplane+1-bit greedy mean  = 0.902604
```

Best per-seed Bloch hyperplane readout:

```text
seed=7   C/W = 0.899740
seed=11  C/W = 0.908854
seed=23  C/W = 0.897135
seed=42  C/W = 0.903646
seed=99  C/W = 0.903646
```

Main visualization:

```text
outputs/bloch_readout_overview/bloch_readout_overview.png
outputs/bloch_readout_overview/bloch_readout_overview.md
```

## Main Conclusion

The current model is not limited only by optimization time or final thresholding. The strongest new evidence is:

```text
Bloch hyperplane correlated readout > Bernoulli independent readout > direct rounding
```

This means the hidden Bloch vectors already contain useful correlation information that the scalar probability readout does not fully use.

The closest route to GW is therefore not a larger threshold scan, nor a crude full-vector loss. It is a cleaner SQNN mechanism that makes hidden phase/Bloch vectors more consistently hyperplane-roundable while preserving the Z-basis measurement interpretation.

## Strong Negative Results

These routes should not be mainline:

```text
reset route
pure target relation
target-mix agree/softagree gate
target-mix ramp/decay schedule
larger final global rotation
entropy sharpening
z_message self_mix away from 0.50
z_message decay far from 0.70
longer product-objective optimization
j_weight=50 or j_weight=150
small full-vector auxiliary loss
edge_cavity_xy directly added to z_mix phase dynamics
```

The repeated pattern is that improving the relaxed/product expected objective often does not improve final binary cut quality.

## Current Mainline

For pure SQNN/V14 MaxCut-3:

```text
V14 Z-edge phase dynamics
two-stage trust region
small trainable random RZ/RY symmetry breaking
Z-basis product expected MaxCut objective
J penalty retained
mix025 as weak-seed repair route
gain14 / sched10to14 / sched08to14 as route candidates
Bloch XYZ hyperplane readout as correlated diagnostic/readout
```

## Next Recommended Work

1. Design a measurement-faithful correlated readout story:
   keep Z-basis probabilities as the primary observable, but explicitly justify using hidden Bloch vectors as SQNN-generated embeddings for hyperplane rounding.

2. Build a cleaner phase-correlation mechanism:
   do not directly add full-vector loss or raw edge_cavity torque. Instead, try a separate edge phase memory or readout-only embedding head that does not disturb Z collapse.

3. Improve route selection:
   current best route depends on instance seed. A lightweight instance-adaptive selector or internal adaptive gain mechanism is more promising than a single fixed route.

4. Move from `C/W` to stricter comparisons:
   for small/medium instances, compute or approximate `C*`; for larger instances, compare to best-known classical baselines and GW-style baselines under the same graph distribution.

5. Scale check:
   rerun the current V14 phase-aware model on n=1024 random 3-regular MaxCut and apply Bloch hyperplane readout, because existing n=1024 outputs are mostly from older/non-V14 routes.

## n=1024 Light Scale Check

One lightweight V14 run was added after the main n=512 comparison:

```text
n=1024, degree=3, seed=17
phase = v14_memory_xy_z_edge_gain_schedule_1p0_1p4_collapse
rounds=260, epochs=90
```

Results:

```text
direct+1-bit greedy C/W      = 0.888672
Bernoulli sample+greedy C/W  = 0.889323
Bloch XYZ hyperplane C/W     = 0.889974
residual active variables    = 32
max residual component       = 3
```

This is only a scale smoke test. It suggests the V14 route remains viable at n=1024, but the Bloch hyperplane gain is much smaller than the best n=512 cases. The n=1024 setting needs its own route/gain/schedule tuning.
