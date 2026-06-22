# SQNN_QAOA

SQNN-QAOA experiments for MaxCut/QUBO with physically measurable Z-basis direct readout as the current mainline.
Historical warm-start utilities remain under `quantum/warmstart/`.

## Main Documents

- [Project Plan](PROJECT_PLAN.md): current mainline, model direction, next experiments.
- [Metrics And Ratios](docs/metrics_and_ratios.md): `C/W`, `C/C*`, `C/UB`, `C/C_best_known`, `GW expected`.
- [Classical Baselines](docs/classical_baselines.md): GW-style, CP-SAT, random+greedy, sampled baselines.
- [Model Mechanics](docs/model_mechanics.md): Bloch vector, `R_Z`, `R_Y`, and phase dynamics.
- [V14 Exploration Report](docs/reports/maxcut3_v14_exploration.md): archived V14/Bloch readout exploration summary.

## Code Entry Points

- `scripts/run_maxcut3_phase_aware_probe.py`: phase-aware SQNN/V14 experiments.
- `quantum/warmstart/phase_aware_sqnn.py`: reusable V14/Clean-ZEdge phase-aware SQNN models.
- `quantum/warmstart/qubo_sqnn.py`: older reusable QUBO/SQNN model family under the historical warm-start namespace.
- `classical/maxcut3_compare.py`: CP-SAT/GW-style MaxCut-3 baselines plus SQNN comparison driver.
- `classical/n512_10_random_graphs.py`: n=512 ten-graph baseline protocol.

Large experiment outputs are ignored under `outputs/`.
