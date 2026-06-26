# V14 Soft Monotone Ablation

## Question

Does "soft monotone / locally relaxed monotone" improve the V14 MaxCut basin-escape path?

Here "soft monotone" means: keep the original V14 strict monotone rule in the ordinary dynamics, but during a short anneal/recovery window allow a proposed round with worse expected energy to be accepted with a Metropolis probability.  This is controlled by `metropolis_temperature` in `run_soft_global_v14`.

This is different from disabling `monotone_accept` globally.  The earlier global-disable ablation was mostly harmful.

## New Runner

- `scripts/run_v14_soft_monotone_ablation.py`

The runner changes only `metropolis_temperature` while keeping V14 weights/proposal dynamics fixed.

## Main Outputs

- `outputs/v14_soft_monotone_ablation_seed2_fast/REPORT.md`
- `outputs/v14_soft_monotone_ablation_seed2_top_repeats/REPORT.md`
- `outputs/v14_soft_monotone_ablation_seed2_top_stride1/REPORT.md`
- `outputs/v14_soft_monotone_ablation_seed0134_sanity/REPORT.md`

## Seed2 Result

Seed2 was the locked seed with weak/no large direct-readout transition.

Base V14:

- best direct+greedy: 683
- best direct: 674
- best expected cut: 664.954

Best local-soft-monotone result, stride-1 verified:

- start round: 160
- template: `cosine_stable`
- best direct+greedy: 694
- best direct: 690
- best expected cut: 675.852

This is a real improvement over base V14 and over the previous seed2 window result around 688-689.

The best repeated seed2 family was:

- start: 160
- template: `cosine_stable`
- `metropolis_temperature`: 0.03, 0.24, or 0.48
- mean direct+greedy across repeated candidates: about 690-691
- best direct+greedy: 694

## Sanity Seeds

A small sanity scan on seeds 0, 1, 3, 4 also improved over base V14, but did not beat the best previous full window-search paths for every seed.

Limited sanity-scan bests:

- seed0: base 694 -> soft 698
- seed1: base 679 -> soft 700
- seed3: base 688 -> soft 693
- seed4: base 692 -> soft 697

Compared with earlier richer window scans, these are not universally new records.  The important point is that local softening can generate useful paths, but it is not a standalone replacement for window selection.

## Interpretation

Soft monotone helps when the jump window is already close to the right transition-preparation region.  It lets the state cross a small expected-energy barrier instead of being immediately rejected by strict monotone.  The guard then rejects paths that damage the checkpoint too much.

It does not solve the whole problem by itself:

- Too much softening is not consistently better.
- Late windows still usually fail.
- The effect depends strongly on the anneal template and start round.
- It should not replace the original monotone rule in normal V14 evolution.

## Recommendation

Keep strict monotone in base V14.

Add local-soft-monotone variants to the basin-escape portfolio:

- `cosine_stable`, start near the selected transition peak minus about 60 rounds
- `metropolis_temperature` in `{0.03, 0.24, 0.48}`
- a few repeats per selected window
- keep guard/rollback enabled

This is worth integrating into the conditioned window scan as an optional portfolio branch.
