# V14 Unified UTC-SM-Lite

目标：不再为每个 seed 单独设计多阶段扫描，而是使用一套统一的跳盆规则。

## Unified Rule

Baseline V14 先运行一次，自动检测 direct readout 主相变峰 `peak`，然后只在固定相对窗口附近做候选跳盆：

```text
starts = peak - {60, 55, 35, 30}
template = cosine_stable
metropolis_temperature = template, 0.06, 0.24
repeats = 2
fine scan = off
confirm scan = off
selection = direct+greedy at candidate end
```

命令模板：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_v14_auto_conditioned_window_scan.py \
  --seed SEED \
  --device cuda:0 \
  --coarse-offsets=-60,-55,-35,-30 \
  --coarse-repeats 2 \
  --fine-radius=-1 \
  --confirm-top-k 0 \
  --template-names cosine_stable \
  --metropolis-temperatures template,0.06,0.24 \
  --fast-internal-scan \
  --score-stride 1 \
  --output-dir outputs/v14_utc_sm_lite_v3_seedSEED
```

## Results

| seed | base V14 DG | old anchor8 DG | full TC-SM DG | unified v3 DG | unified start | seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 694 | 701 | 701 | 701 | 134 | 48.42 |
| 1 | 679 | 702 | 701 | 702 | 147 | 46.64 |
| 2 | 683 | 688 | 692 | 692 | 160 | 43.64 |
| 3 | 688 | 696 | 698 | 697 | 37 | 35.03 |
| 4 | 692 | 702 | 703 | 703 | 35 | 42.67 |

Mean:

| method | mean DG |
| --- | ---: |
| base V14 | 687.20 |
| old anchor8 | 697.80 |
| full TC-SM | 699.00 |
| unified UTC-SM-lite v3 | 699.00 |

## Interpretation

The first lite version used `{60,45,35,25}` and one repeat. It was fast but missed important starts:

- seed2 needed a second stochastic direction around start 160/165.
- seed3 needed the `peak-30` window, namely start 37.

The unified v3 keeps the mechanism seed-independent but adds just enough coverage:

- `{60,55,35,30}` covers early and mid pre-transition windows.
- `repeats=2` supplies a second direction without doing a full broad scan.
- `template,0.06,0.24` keeps the old branch and one stronger soft-monotone branch.

The only remaining gap is seed3: unified v3 reaches 697, while the full TC-SM broad scan reaches 698. This suggests the unified rule is already near the broad scan, but a rare seed can still benefit from extra temperature/window coverage.

## Output

Detailed comparison:

```text
outputs/v14_utc_sm_lite_v123_comparison.csv
```
