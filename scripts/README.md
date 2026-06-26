# Scripts Index

This folder keeps stable command-line entry points and the V14-UTC dependency stack.

## Current Formal V14-UTC

```text
scripts/README_V14_UTC.md
scripts/run_v14_four_scheme_seed_benchmark.py
scripts/run_v14_four_scheme_random50_all.sh
scripts/merge_v14_four_scheme_seed_benchmark.py
```

Supporting V14-UTC modules that must stay importable from this folder:

```text
scripts/run_maxcut3_phase_aware_probe.py
scripts/run_v14_auto_conditioned_window_scan.py
scripts/run_v14_bloch_anneal_escape.py
scripts/run_v14_bloch_guided_anneal_search.py
scripts/run_v14_manual_schedule_compare.py
scripts/run_v14_quantum_reset_escape.py
scripts/run_v14_readout_guided_timing_scan.py
scripts/run_v14_reevolve_from_escape.py
scripts/run_v14_soft_global_anneal_search.py
scripts/run_v14_transition_phase_anneal_scan.py
```

## Scale And Legacy Baselines

```text
scripts/run_v10_maxcut3_report.py
scripts/run_scale_v10_v14_maxcut3_cuda.sh
scripts/run_scale_v10_v14_maxcut3_cuda_shards.sh
scripts/compute_maxcut3_baselines.py
scripts/explore_j_regularized_sqnn.py
scripts/run_qubo_warmstart.py
```

## Archive

Selected one-off V14 helper scripts live in:

```text
maxcut/v14_utc/scripts_archive/
```

The larger exploratory script archive was deleted. New scripts should either
extend the formal runner or live under the task folder first, then move here
only when they become stable entry points.
