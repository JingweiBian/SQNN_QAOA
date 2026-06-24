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
  reports/README.md
  scripts/README.md
```

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
- Frustrated synchronization changes go in `frustrated_sync_dynamics/`.
- Reusable Bloch/SQNN layers go in `quantum/`.
- Existing historical MaxCut scripts stay where they are until we migrate them
  deliberately.
