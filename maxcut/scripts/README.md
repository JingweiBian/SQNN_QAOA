# MaxCut Scripts

Formal runnable MaxCut entry points currently remain in the repository-level
`scripts/` directory to avoid breaking existing commands.

Current V14-UTC entry points:

```text
scripts/README_V14_UTC.md
scripts/run_v14_four_scheme_seed_benchmark.py
scripts/run_v14_four_scheme_random50_all.sh
scripts/merge_v14_four_scheme_seed_benchmark.py
```

Archived V14-only helper scripts that used to be here are now in:

```text
maxcut/v14_utc/scripts_archive/
```

New MaxCut-only runners can be drafted here, but mature shared runners should
be promoted to the top-level `scripts/` directory only when they are intended
as stable entry points.
