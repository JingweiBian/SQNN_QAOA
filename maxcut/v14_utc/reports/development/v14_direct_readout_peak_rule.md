# V14 Direct-Readout Transition Peak Rule

Date: 2026-06-25

## Update

Transition peaks for basin-jump timing should be detected from the model's own
hard readout, not from greedy-corrected readout.

Use this rule:

1. Find direct-readout transition events:
   - large `|d direct_cut|`
   - or many `bit_flips_from_prev`
2. Gate those events by smooth continuous objective:
   - require small `|d C[p]|`
3. Prefer events whose event-level `direct_delta` is positive.
4. Do not use `direct+greedy` to select the peak.  Greedy can turn a bad
   readout into a good discrete solution, but that is not a clean signal of a
   model-state transition.

## Seed3 Diagnostic

For seed3, a later event around peak 170 looked attractive only after greedy:

- peak 170 event: `direct_delta = -10`, `direct+greedy_delta = +11`
- single-round `|d C[p]| = 0.412`
- event-level `delta C[p] = 2.929`

This is a greedy-rescued readout event, not a direct readout improvement, and
its jump scan was weak (`best DG = 689` from baseline `688`).

The cleaner event was peak 134:

- `direct_delta = +11`
- `max_bit_flips = 89`
- peak `|d C[p]| = 0.060`
- event-level `delta C[p] = 1.541`

Its scan reached `best DG = 696`, with better `C[p]` than the peak-170 scan.

## Time-Optimized Scan

The updated runner uses a lower-cost adaptive scan:

1. Coarse scan candidate starts from the detected peak, default offsets:
   `-45,-40,-35,-30,-25`
2. Fine scan around the coarse best start, default radius `4`, step `2`.
3. Confirm only the top `3` starts, with `2` repeats each.

The runner now also defaults to `--score-stride 4`, so coarse/fine/confirm
paths only run greedy scoring every fourth round.  This keeps the seed3 best
case because its high-quality readout persists for several rounds.

Seed3 speed measurements:

| mode | command changes | jump paths | best DG | seconds |
|---|---|---:|---:|---:|
| exact scoring | `--score-stride 1` | 16 | 696 | 45.20 |
| default fast | `--score-stride 4` | 16 | 696 | 31.43 |
| internal no-greedy | `--fast-internal-scan` | 16 | 696 | 33.73 |
| conservative 12-path | `--fine-radius 2 --confirm-top-k 2` | 12 | 696 | 23.72 |
| aggressive 8-path | `--coarse-offsets=-40,-35,-30 --fine-radius 2 --confirm-top-k 1` | 8 | 696 | 16.91 |
| exploration fast | `--score-stride 4 --confirm-top-k 0 --confirm-repeats 0` | 10 | 696 | 20.43 |
| ultrafast | `--score-stride 4 --fine-radius 0 --confirm-top-k 0 --confirm-repeats 0` | 6 | 696 | 16.98 |
| direct coarse/fine | `--coarse-score-mode direct --fine-score-mode direct --confirm-score-mode dg` | 16 | 694 confirmed / 689 direct-selected | 38.83 |

The direct-score scan was tested because it is conceptually cleaner for
choosing windows.  It did not speed up the current runner because
`run_soft_global_v14` still performs greedy-based guard / best tracking inside
each trajectory.  It also lost the coarse trajectory's `DG=696` because coarse
paths were not DG-scored.  Keep direct-only scoring as a diagnostic option, not
the default production path.

The internal no-greedy mode was then added as `--fast-internal-scan`.  It keeps
external DG scoring, but skips greedy scoring inside `run_soft_global_v14`.
On seed3 it preserved the best path (`DG=696`, start `37`) but did not improve
wall time relative to the latest default fast control; the remaining dominant
cost is path generation plus external trace scoring.  The practical speed win
came from reducing scan granularity.

Recommended usage:

- Use conservative 12-path mode for normal unknown-seed runs when speed matters.
- Use aggressive 8-path mode when quickly mapping many seeds or doing iteration.
- Use exact scoring or add confirmation repeats only for final report numbers.
