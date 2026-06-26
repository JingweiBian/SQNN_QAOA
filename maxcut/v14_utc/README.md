# V14-UTC MaxCut Scheme

This folder is the curated home for the current formal MaxCut line:

```text
V14-UTC = V14 clean_edgeboost_mem060 + UTC-SM-lite v3
```

V14-UTC should be treated as the stable sparse-graph MaxCut demonstration, especially for random regular graphs around `d=3` and `d=4`. It is not the dense-graph main line; the dense direction now moves to `maxcut/v18_dissipative/`.

## Core Idea

The model keeps the V14 Bloch-state dynamics as the main optimizer and adds a transition-conditioned escape layer only around useful readout-transition windows.

- V14 provides the continuous quantum-inspired state evolution over per-variable Bloch vectors.
- Direct readout transitions reveal when the state reorganizes between basins even if expected energy changes smoothly.
- UTC finds a small set of candidate jump windows before the direct-readout transition.
- SM-lite explores a few soft-monotone escape paths in those windows, then selects by direct/direct+greedy readout.

The final policy is deliberately modest: keep V14 strict enough to preserve stable dynamics, then use a small transition-conditioned escape scan instead of repeatedly perturbing the whole trajectory.

## Canonical Files

- [TECHNICAL_ROUTE.md](TECHNICAL_ROUTE.md): V10 -> V14 -> V14-UTC technical route.
- [MODEL_SPEC.md](MODEL_SPEC.md): formal step-by-step model specification.
- [EVALUATION.md](EVALUATION.md): final benchmark tables and conclusions.
- [SCRIPT_INDEX.md](SCRIPT_INDEX.md): scripts to run, merge, and reproduce results.
- [REPORT_INDEX.md](REPORT_INDEX.md): where the old reports went.

## Canonical Outputs

- `outputs/final_v14_utc/random50/method_summary.csv`
- `outputs/final_v14_utc/random50/four_scheme_wide_summary.csv`
- `outputs/final_v14_utc/seed0_9/method_summary.csv`
- `outputs/final_v14_utc/reports/`
- `outputs/report_v10_v14_scale_upper_bound/`
- `outputs/v14_density_sweep_seed0/`

## Current Positioning

For the paper/story, this is the cleanest V14 claim:

1. V10 establishes the first monotone local-field SQNN baseline.
2. V14 improves the dynamical model with memory, edge/cavity corrections, and stable Bloch evolution.
3. V14-UTC adds a transition-conditioned basin escape mechanism based on observed readout transitions.
4. The method is most convincing on sparse MaxCut as a quantum-inspired dynamical optimizer, not as a dense-graph classical-solver replacement.
5. Dense graphs are better handled by V18 dissipative dynamics, which is now the next development target.
