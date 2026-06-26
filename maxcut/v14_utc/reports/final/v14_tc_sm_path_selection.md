# V14 TC-SM Path Selection

## Idea

Transition-Conditioned Soft-Monotone Path Selection:

```text
find jump window
-> generate multiple soft-monotone paths
-> short recovery under V14 dynamics
-> select by direct/direct+greedy
```

This is inference-only.  Training and V14 weights are unchanged.

## Implementation Note

`scripts/run_v14_auto_conditioned_window_scan.py` now supports:

```bash
--metropolis-temperatures template,0.03,0.06,0.24,0.48
```

`template` means "keep the old template-default branch" with the old label/seed path.  This is important: TC-SM should be an add-on portfolio branch, not a replacement for the previous jump path.

## Evaluation

Seeds: `0,1,2,3,4`

Window pool:

```text
peak-60, peak-55, peak-45, peak-40, peak-35, peak-30, peak-25, peak-10
```

Ran:

```bash
--template-names cosine_stable
--metropolis-temperatures 0.03,0.06,0.24,0.48
--coarse-repeats 2
--fast-internal-scan
--score-stride 1
```

The old anchor8 branch is included in the final portfolio comparison as a保底 branch.

## Results

| seed | base DG | old anchor8 DG | TC-SM scan DG | portfolio best DG | gain vs old | gain vs base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 694 | 701 | 701 | 701 | 0 | 7 |
| 1 | 679 | 702 | 701 | 702 | 0 | 23 |
| 2 | 683 | 688 | 692 | 692 | 4 | 9 |
| 3 | 688 | 696 | 698 | 698 | 2 | 10 |
| 4 | 692 | 702 | 703 | 703 | 1 | 11 |

Mean:

- base V14: 687.2
- old anchor8: 697.8
- TC-SM scan alone: 699.0
- old + TC-SM portfolio: 699.2
- gain vs old anchor8: +1.4
- gain vs base V14: +12.0

## Interpretation

TC-SM improves the difficult cases:

- seed2: 688 -> 692
- seed3: 696 -> 698
- seed4: 702 -> 703

seed0 is unchanged relative to old anchor8, and seed1's TC-SM-only best is one point below old anchor8.  Keeping the old branch fixes that.

The best paths are not all from one temperature:

- seed0: `metro=0.24`, start 129
- seed1: `metro=0.24`, start 137
- seed2: `metro=0.06`, start 165
- seed3: `metro=0.06`, start 27
- seed4: `metro=0.03`, start 30

This supports the path-selection picture: temperature is not a universal optimum; it is a candidate-path generator.

## Output Files

- `outputs/v14_tc_sm_path_selection_summary.csv`
- `outputs/v14_tc_sm_path_selection_comparison.csv`
- `outputs/v14_tc_sm_path_selection_portfolio_comparison.csv`
- `outputs/v14_tc_sm_path_selection_seed0/`
- `outputs/v14_tc_sm_path_selection_seed1/`
- `outputs/v14_tc_sm_path_selection_seed2/`
- `outputs/v14_tc_sm_path_selection_seed3/`
- `outputs/v14_tc_sm_path_selection_seed4/`
