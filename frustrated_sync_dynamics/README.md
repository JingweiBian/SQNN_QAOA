# Frustrated Sync Dynamics

This folder isolates the non-MaxCut basin-stability direction.

The shared SQNN/Bloch primitive is still imported from `quantum.core.layers`.
Task-specific simulator, labels, baselines, and the `SyncBasinSQNN` surrogate
live here so MaxCut experiments can keep their own route.

Main entry points:

```text
python scripts/run_frustrated_sync_basin_benchmark.py
python -m frustrated_sync_dynamics.run_basin_benchmark
```

Reports:

```text
frustrated_sync_dynamics/reports/frustrated_sync_dynamics_plan.md
frustrated_sync_dynamics/reports/frustrated_sync_basin_pilot_results.md
```
