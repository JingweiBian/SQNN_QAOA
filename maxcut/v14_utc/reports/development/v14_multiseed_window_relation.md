# V14 Multi-Seed Jump Window Relation

Date: 2026-06-26

## Experiment

We scanned five n=512 MaxCut seeds: `0,1,2,3,4`.

For each seed:

1. run baseline V14 once;
2. detect the direct-readout transition peak;
3. build a single-jump candidate pool around that peak;
4. use the pool best DG as the per-seed reference;
5. simulate cheaper scan policies from the same pool.

Output folder:

- `outputs/v14_multiseed_window_relation_n512_seed0_4`

## Window Relation

Best single-jump offsets relative to the selected direct-positive transition
peak were:

| seed | main peak | largest readout peak | best start | best offset | base DG | best DG |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 169 | 205 | 134 | -35 | 694 | 702 |
| 1 | 177 | 83 | 122 | -55 | 679 | 702 |
| 2 | 220 | 220 | 160 | -60 | 683 | 688 |
| 3 | 67 | 67 | 32 | -35 | 688 | 696 |
| 4 | 65 | 65 | 55 | -10 | 692 | 703 |

Median best offset: `-35` rounds.

The useful window is not a single fixed point.  In this five-seed sample it
ranges from about `peak-60` to `peak-10`, with a strong cluster around
`peak-35`.

## Largest Peak Caveat

The largest raw readout jump is not always the best anchor.  Seed0 has
main peak `169` but largest readout peak `205`; the best jump is `134`, which
is `-35` from the direct-positive peak but `-71` from the largest peak.  Seed1
also has a largest readout peak that is not the useful anchor.  So the timing
rule should prefer the direct-positive / smooth-C[p] event, and use raw largest
peak only as a diagnostic.

## Fastest Same-Value Policies

Using the completed candidate pool, these policies preserved the same per-seed
reference DG on all five seeds:

| policy | offsets | match | mean paths | mean seconds |
|---|---|---:|---:|---:|
| anchor4_observed | `-60,-55,-35,-10` | 5/5 | 3.6 | 6.59 |
| anchor5_mid | `-60,-55,-35,-25,-10` | 5/5 | 4.6 | 8.43 |
| anchor6_robust | `-60,-55,-45,-35,-25,-10` | 5/5 | 5.4 | 9.88 |
| broad_step5_11 | `-60,-55,...,-10` | 5/5 | 10.0 | 18.32 |

`anchor4_observed` is fastest in the completed candidate pool, but it is likely
overfit and it also depends on the candidate-pool anneal seed.  In the actual
auto-scan runner, the same start can get a different anneal perturbation seed
because the label changes.  So the practical default should cover the observed
bands rather than only the exact observed anchors.

Recommended next scan default:

```bash
--coarse-offsets=-60,-55,-45,-40,-35,-30,-25,-10 --fine-radius -1 --confirm-top-k 0
```

This actual command was checked on seed3 and preserved `DG=696` in `14.69s`.
It covers:

- early band: `-60,-55`
- middle band: `-45,-40,-35,-30,-25`
- near-peak fallback: `-10`

Actual runner validation on seeds `0..4`:

| seed | reference DG | anchor8 actual DG | gap | seconds |
|---:|---:|---:|---:|---:|
| 0 | 702 | 701 | -1 | 17.16 |
| 1 | 702 | 702 | 0 | 17.49 |
| 2 | 688 | 688 | 0 | 16.43 |
| 3 | 696 | 696 | 0 | 14.77 |
| 4 | 703 | 702 | -1 | 13.13 |

So this practical command is a very fast near-reference scan: exact on `3/5`
seeds, within `1` DG point on all five.  For strict final numbers, add a small
fine pass around the best coarse start or fall back to `broad_step5_11`.

If a seed returns weak improvement or the best offset hits `-60`, expand earlier
or reconsider the selected transition anchor.
