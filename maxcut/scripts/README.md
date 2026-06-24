# MaxCut Scripts

Historical MaxCut entry points currently remain in the repository-level
`scripts/` directory to avoid breaking existing reports and commands.

Important current scripts include:

```text
scripts/run_maxcut3_phase_aware_probe.py
scripts/run_v14_bloch_anneal_escape.py
scripts/run_v14_soft_global_anneal_search.py
scripts/run_v14_qtabu_anneal_search.py
scripts/run_maxcut512_classical_vs_sqnn_sa.py
```

New MaxCut-only runners should be created here first, or wrapped here while the
old path remains available for compatibility.
