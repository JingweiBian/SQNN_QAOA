# V14 Soft Monotone Integrated Scan

## Change

`scripts/run_v14_auto_conditioned_window_scan.py` now supports an optional soft-monotone temperature portfolio:

```bash
--metropolis-temperatures 0.03,0.24,0.48
```

If this argument is empty, the script keeps the old behavior and uses each template's default `metropolis_temperature`.

The candidate pool becomes:

```text
jump start x template x metropolis_temperature x repeat
```

This is inference-only.  V14 training and weights are unchanged.

## Evaluation Setup

Seeds: `0,1,2,3,4`

Window pool: previous anchor8 offsets

```text
peak-60, peak-55, peak-45, peak-40, peak-35, peak-30, peak-25, peak-10
```

New soft-monotone extra temperatures:

```text
0.03, 0.24, 0.48
```

The comparison treats soft monotone as an add-on branch, not a replacement for the old anchor8 path.  This is important because the old template-default branch remains a valid candidate.

## Results

| seed | base DG | old anchor8 DG | new soft-extra DG | add-on best DG | gain vs old | gain vs base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 694 | 701 | 699 | 701 | 0 | 7 |
| 1 | 679 | 702 | 699 | 702 | 0 | 23 |
| 2 | 683 | 688 | 690 | 690 | 2 | 7 |
| 3 | 688 | 696 | 697 | 697 | 1 | 9 |
| 4 | 692 | 702 | 701 | 702 | 0 | 10 |

Mean:

- base V14: 687.2
- old anchor8: 697.8
- add-on soft-monotone best: 698.4
- mean gain vs old anchor8: +0.6
- mean gain vs base: +11.2

## Interpretation

Soft monotone is useful as an add-on branch:

- It improved seed2 from 688 to 690 in the automatic anchor8 scan.
- It improved seed3 from 696 to 697.
- It did not beat the old branch on seeds 0, 1, and 4, but keeping the old branch prevents regression.

The seed2 repeat check on `start=160` reached 692 in the integrated runner.  Earlier standalone ablation reached 694 with a different candidate seed path, so the effect is real but path-dependent.

## Recommendation

Use soft monotone as a portfolio branch inside jump-window inference:

```bash
--metropolis-temperatures 0.03,0.06,0.24,0.48
```

Include `0.06` if you want the old `cosine_stable` default branch explicitly included in the same scan.

Do not replace the old branch with only the new temperatures.

## Output Files

- `outputs/v14_auto_conditioned_soft_monotone_anchor8_summary.csv`
- `outputs/v14_auto_conditioned_soft_monotone_anchor8_comparison.csv`
- `outputs/v14_auto_conditioned_soft_monotone_anchor8_seed0/`
- `outputs/v14_auto_conditioned_soft_monotone_anchor8_seed1/`
- `outputs/v14_auto_conditioned_soft_monotone_anchor8_seed2/`
- `outputs/v14_auto_conditioned_soft_monotone_anchor8_seed3/`
- `outputs/v14_auto_conditioned_soft_monotone_anchor8_seed4/`
- `outputs/v14_auto_conditioned_soft_monotone_seed2_start160_repeats/`
