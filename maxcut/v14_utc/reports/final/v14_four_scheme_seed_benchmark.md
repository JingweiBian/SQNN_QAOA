# V14 Four-Scheme Seed Benchmark

## Schemes

The benchmark compares four schemes:

| scheme | definition |
| --- | --- |
| `base_v14` | original V14 direct+greedy readout |
| `old_anchor8` | offsets `peak-{60,55,45,40,35,30,25,10}`, template default branch, repeat 1 |
| `full_tc_sm` | same 8 offsets, temperatures `0.03,0.06,0.24,0.48`, repeat 2 |
| `utc_sm_lite_v3` | unified rule: offsets `peak-{60,55,35,30}`, temperatures `template,0.06,0.24`, repeat 2 |

All jump schemes use the same baseline V14 trajectory and the same direct-readout transition detector.

## Trained Seed 0-9 Result

These 10 seeds already had V14 models under `outputs/v14_maxcut3_report_n512_10seeds`, so no new training was needed.

| method | seeds | mean DG | median DG | min DG | max DG | mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_v14` | 10 | 687.4 | 686.5 | 678 | 698 | 2.09 |
| `old_anchor8` | 10 | 696.3 | 696.5 | 688 | 702 | 16.00 |
| `full_tc_sm` | 10 | 699.1 | 700.0 | 692 | 704 | 129.46 |
| `utc_sm_lite_v3` | 10 | 698.0 | 698.0 | 692 | 703 | 44.27 |

Wide per-seed summary:

```text
outputs/v14_four_scheme_seed0_9/four_scheme_wide_summary.csv
```

Method summary:

```text
outputs/v14_four_scheme_seed0_9/method_summary.csv
```

## Interpretation

On seed `0..9`, unified UTC-SM-lite v3 is much faster than full TC-SM:

- full TC-SM: about `129.5s/seed`
- unified v3: about `44.3s/seed`

Quality tradeoff:

- full TC-SM mean DG: `699.1`
- unified v3 mean DG: `698.0`
- unified v3 mean gain over base V14: `+10.6`
- unified v3 mean gain over old anchor8: `+1.7`
- unified v3 mean gap to full TC-SM: `-1.1`

The main weak case for unified v3 is seed `8`:

```text
base=678, old=691, full=700, unified=695
```

This says the unified rule is a strong fast default, but a few seeds still need the broader full TC-SM portfolio.

## Random 50 Seed Setup

The fixed random seed list was generated with:

```text
master_seed = 20260626
range = [10000, 9999999]
exclude = 0..9
count = 50
```

Seed list:

```text
outputs/v14_four_scheme_random50/seed_list.csv
```

The 50-seed list is the first 50 seeds from the earlier 100-seed list, so it can
be extended later without changing the prefix.  The first random seed is
`9678277`. A full train+four-scheme smoke test completed:

```text
seed=9678277 base=682 old=686 full=686 utc=686 total=470.1s
```

That timing includes training a missing V14 model.  Therefore a true random-50 benchmark is expected to take roughly:

```text
50 * 470s / 4 GPUs ~= 1.6 hours
```

## Commands

Simple one-command runner:

```bash
bash scripts/run_v14_four_scheme_random50_all.sh
```

This uses one worker per GPU by default.  If GPU power/utilization is low, run
two workers per GPU:

```bash
WORKERS_PER_GPU=2 bash scripts/run_v14_four_scheme_random50_all.sh
```

The 2-worker mode launches 8 shards across 4 GPUs.  It can be faster when the
job is CPU/Python/greedy-scoring limited, but it may slow down if GPU memory or
CPU cores become the bottleneck.

Manual equivalent, if you prefer four foreground sessions or tmux panes:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_v14_four_scheme_seed_benchmark.py --random-count 50 --random-master-seed 20260626 --exclude-seeds 0,1,2,3,4,5,6,7,8,9 --train-if-missing --shard-count 4 --shard-index 0 --device cuda:0 --output-dir outputs/v14_four_scheme_random50 --v14-training-dir outputs/v14_random50_training
CUDA_VISIBLE_DEVICES=1 python scripts/run_v14_four_scheme_seed_benchmark.py --random-count 50 --random-master-seed 20260626 --exclude-seeds 0,1,2,3,4,5,6,7,8,9 --train-if-missing --shard-count 4 --shard-index 1 --device cuda:0 --output-dir outputs/v14_four_scheme_random50 --v14-training-dir outputs/v14_random50_training
CUDA_VISIBLE_DEVICES=2 python scripts/run_v14_four_scheme_seed_benchmark.py --random-count 50 --random-master-seed 20260626 --exclude-seeds 0,1,2,3,4,5,6,7,8,9 --train-if-missing --shard-count 4 --shard-index 2 --device cuda:0 --output-dir outputs/v14_four_scheme_random50 --v14-training-dir outputs/v14_random50_training
CUDA_VISIBLE_DEVICES=3 python scripts/run_v14_four_scheme_seed_benchmark.py --random-count 50 --random-master-seed 20260626 --exclude-seeds 0,1,2,3,4,5,6,7,8,9 --train-if-missing --shard-count 4 --shard-index 3 --device cuda:0 --output-dir outputs/v14_four_scheme_random50 --v14-training-dir outputs/v14_random50_training
```

Merge shards after completion:

```bash
python scripts/merge_v14_four_scheme_seed_benchmark.py --output-dir outputs/v14_four_scheme_random50
```
