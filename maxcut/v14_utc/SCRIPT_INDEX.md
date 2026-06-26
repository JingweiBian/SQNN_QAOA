# V14-UTC Script Index

The runnable scripts stay in top-level `scripts/` because many imports and historical commands already point there.

## Formal Benchmark

Run the 50-seed four-scheme comparison:

```bash
WORKERS_PER_GPU=2 bash scripts/run_v14_four_scheme_random50_all.sh
python scripts/merge_v14_four_scheme_seed_benchmark.py --output-dir outputs/v14_four_scheme_random50
```

Curated copied outputs are stored in:

```text
outputs/final_v14_utc/random50/
```

## Single Runner

Main runner:

```text
scripts/run_v14_four_scheme_seed_benchmark.py
```

It supports method selection, density/degree changes, and GPU placement. Use it when adding targeted tests instead of creating a new one-off script.

## Main Dependencies

```text
scripts/run_maxcut3_phase_aware_probe.py
scripts/run_v14_reevolve_from_escape.py
scripts/run_v14_auto_conditioned_window_scan.py
scripts/run_v14_readout_guided_timing_scan.py
scripts/run_v14_transition_phase_anneal_scan.py
scripts/merge_v14_four_scheme_seed_benchmark.py
```

## Archived Exploration Scripts

Old one-off probes live in:

```text
maxcut/v14_utc/scripts_archive/
```

They are useful for tracing the research path, but new work should not add more one-off files unless the experiment genuinely cannot be expressed through the formal runner.

## Model Code

V14 itself is implemented through the MaxCut phase-aware runner and shared quantum utilities. New dense-graph model development should go through:

```text
quantum/warmstart/dissipative_sqnn.py
maxcut/v18_dissipative/
```
