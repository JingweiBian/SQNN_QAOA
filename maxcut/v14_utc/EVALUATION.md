# V14-UTC Evaluation

## Random 50-Seed Benchmark

Source: `outputs/final_v14_utc/random50/method_summary.csv`.

| method | seeds | mean DG | median DG | min | max | mean seconds |
|---|---:|---:|---:|---:|---:|---:|
| base_v14 | 50 | 689.64 | 690.0 | 668 | 702 | 2.11 |
| old_anchor8 | 50 | 698.06 | 698.5 | 686 | 706 | 17.89 |
| full_tc_sm | 50 | 700.14 | 701.0 | 686 | 708 | 141.00 |
| utc_sm_lite_v3 | 50 | 698.40 | 700.0 | 685 | 706 | 53.14 |

Interpretation:

- `full_tc_sm` is the highest mean score but too slow for the formal short-time algorithm.
- `utc_sm_lite_v3` keeps most of the gain while being far cheaper than the full transition-conditioned scan.
- `base_v14` remains the clean dynamical baseline.
- `old_anchor8` is competitive but less principled than the transition-conditioned rule.

## Density Sweep, n=512, seed=0

Source: `outputs/v14_density_sweep_seed0/density_vs_classical_summary.csv`.

| degree | edges | V14-UTC DG | V14-UTC frac | GW-style+greedy | GW frac | UTC-GW edges |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 768 | 701 | 0.9128 | 703 | 0.9154 | -2 |
| 4 | 1024 | 872 | 0.8516 | 876 | 0.8555 | -4 |
| 6 | 1536 | 1196 | 0.7786 | 1214 | 0.7904 | -18 |
| 8 | 2048 | 1516 | 0.7402 | 1552 | 0.7578 | -36 |
| 10 | 2560 | 1858 | 0.7258 | 1864 | 0.7281 | -6 |
| 12 | 3072 | 2098 | 0.6829 | 2170 | 0.7064 | -72 |
| 16 | 4096 | 2714 | 0.6626 | 2782 | 0.6792 | -68 |
| 20 | 5120 | 3298 | 0.6441 | 3370 | 0.6582 | -72 |

Conclusion:

- V14-UTC is most appropriate for sparse graphs, especially `d=3` and `d=4`.
- It stays close to GW-style+greedy at low degree, but the dense gap becomes visible.
- Dense graphs should not be the main V14-UTC claim.

## V18 Dense Direction Reference

Source: `outputs/v18_dissipative_dense_probe/v18_vs_v14_gw_density_summary.csv`.

| degree | V14-UTC | V18 dissipative | GW-style | V18-V14 |
|---:|---:|---:|---:|---:|
| 3 | 701 | 689 | 703 | -12 |
| 4 | 872 | 858 | 876 | -14 |
| 6 | 1196 | 1196 | 1214 | 0 |
| 8 | 1516 | 1530 | 1552 | +14 |
| 10 | 1858 | 1860 | 1864 | +2 |
| 12 | 2098 | 2160 | 2170 | +62 |
| 16 | 2714 | 2774 | 2782 | +60 |
| 20 | 3298 | 3366 | 3370 | +68 |

This is why V14-UTC is frozen as the sparse-graph formal method and V18 becomes the dense-graph development path.

## Scale Upper-Bound Reference

Source: `outputs/report_v10_v14_scale_upper_bound/`.

The scale tests support this interpretation:

- original V10 monotone becomes unstable around the 4096-variable level;
- V14 remains usable at much larger size;
- V14 showed useful behavior around the 10k-variable scale, while larger runs became much weaker relative to GW-style baselines.

For reporting, phrase this cautiously as an empirical scale range, not a theorem.

