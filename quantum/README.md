# Quantum Package Layout

The package keeps the reusable SQNN framework:

- `core/`: reusable soft quantum layers and readout primitives.
- `classifiers/`: original SQNN and data-reuploading classification models.
- `encoders/`: step/group encoders used to compress structured inputs.
- `training/`: original classification trainer.
- `warmstart/`: sparse QUBO modeling and SQNN warm-start utilities for QAOA.

Compatibility wrappers remain for the general SQNN paths, such as
`quantum.layers`, `quantum.group_encoder`, `quantum.networkmodels`, and
`quantum.trainer`.

Warm-start experiment entry points:

- `scripts/run_qubo_warmstart.py`: train/evaluate SQNN warm-start models on large QUBO MaxCut benchmarks.
- `scripts/summarize_warmstart_runs.py`: aggregate `metrics.json` files into CSV/Markdown tables.
- `scripts/plot_warmstart_run.py`: generate a compact quality/residual plot for one run.
- `scripts/run_residual_qaoa_demo.py`: run small statevector QAOA on a fixed residual QUBO, including isolated-variable fixing and optional component-wise mode.
- `scripts/run_warmstart_sweep.py`: run controlled multi-seed sweeps with per-run timeout.
- `scripts/smoke_warmstart.py`: run lightweight correctness checks for QUBO preprocessing and residual QAOA.

Supported large-QUBO benchmarks include `planted_maxcut`,
`random_maxcut`, and `planted_parity`. `planted_maxcut` and
`planted_parity` have known optima, so their reported approximation ratios are
strict rather than best-observed proxies.
