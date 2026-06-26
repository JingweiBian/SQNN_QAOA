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

Current experiment entry points:

- `scripts/run_qubo_warmstart.py`: legacy warm-start train/evaluate entry point.
- `scripts/README_V14_UTC.md`: formal V14-UTC MaxCut runner index.
- `maxcut/v14_utc/`: frozen sparse MaxCut method documentation.
- `maxcut/v18_dissipative/`: active dense-graph development direction.

Supported large-QUBO benchmarks include `planted_maxcut`,
`random_maxcut`, and `planted_parity`. `planted_maxcut` and
`planted_parity` have known optima, so their reported approximation ratios are
strict rather than best-observed proxies.
