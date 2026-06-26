# MaxCut Experiments

This folder is the task boundary for MaxCut work.

The repository still keeps historical MaxCut scripts in `scripts/`,
`classical/`, and `docs/reports/` because many reports and runners already
refer to those paths.  New MaxCut-specific model changes, experiment notes, and
task wrappers should go under this folder first, then shared utilities can be
promoted to common packages when they are useful outside MaxCut.

Current layout:

```text
maxcut/
  README.md
  v14_utc/
  v18_dissipative/
  reports/
  scripts/
```

Current algorithm split:

- `v14_utc/`: frozen formal V14-UTC sparse MaxCut scheme.
- `v18_dissipative/`: active dense-graph development direction.
- `reports/`: only a compatibility/index location; V14 reports moved under `v14_utc/reports/`.
- `scripts/`: MaxCut-local helper scripts; most runnable historical scripts still live in top-level `scripts/`.

Shared model primitives remain in:

```text
quantum/
```

Task-specific new direction:

```text
frustrated_sync_dynamics/
```

Rule of thumb:

- MaxCut-only changes go in `maxcut/`.
- V14 sparse-graph reporting and formal method details go in `maxcut/v14_utc/`.
- Dense-graph work should now start from `maxcut/v18_dissipative/`.
- Frustrated synchronization changes go in `frustrated_sync_dynamics/`.
- Reusable Bloch/SQNN layers go in `quantum/`.
- Existing historical MaxCut scripts stay where they are until we migrate them
  deliberately.
