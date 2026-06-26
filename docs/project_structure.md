# Project Structure

Use this as the current repository map.

## Current Entry Points

```text
README.md
PROJECT_PLAN.md
docs/README.md
```

## Core Code

```text
quantum/
```

Reusable SQNN and quantum-inspired model code.

```text
classical/
```

Classical baselines, MaxCut scoring, GW-style/random-greedy comparisons, and report builders.

```text
tools/
```

Small shared utilities.

## Task Folders

```text
maxcut/v14_utc/
```

Frozen formal V14-UTC sparse MaxCut method.

```text
maxcut/v18_dissipative/
```

Active dense-graph MaxCut development direction.

```text
frustrated_sync_dynamics/
```

Separate non-MaxCut task direction.

## Scripts

```text
scripts/
```

Stable runnable entry points and V14-UTC dependency stack. See `scripts/README.md`.

Do not move the V14-UTC dependency scripts casually: several scripts import each other by top-level module name.

## Outputs

```text
outputs/final_v14_utc/
outputs/report_v10_v14_scale_upper_bound/
outputs/v14_density_sweep_seed0/
outputs/v18_dissipative_dense_probe/
```

Current report-worthy outputs.

Raw historical runs, smoke tests, failed branches, and duplicate shards were
pruned from `outputs/`. Keep only report-worthy outputs or compact diagnostics.

## Documentation

```text
docs/
docs/archive/
maxcut/v14_utc/reports/
```

Shared docs live in `docs/`. V14-specific reports live under `maxcut/v14_utc/reports/`. Old root notes live under `docs/archive/root_legacy/`.
