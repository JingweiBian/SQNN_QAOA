# Frustrated Sync Basin Pilot Results

## MaxCut V15 Decision

The lightweight V15 MaxCut direction is not better than the current V14
baseline under the same small control budget:

- setting: n=512, seeds 0 and 1, 160 rounds, 50 epochs, CPU;
- V14 baseline mean direct gap to GW: -0.028159;
- V15 light edge-memory/soft-J mean direct gap to GW: -0.040528;
- V15 readout-STE mean direct gap to GW: -0.052247;
- V15 variants also lose clearly on expected value and sampled readout.

Only direct+greedy occasionally looks close, but that is a cleanup effect rather
than a stronger model readout.  This V15 MaxCut route should not be treated as
the main line.

## New Direction

Implemented a separate pilot for the plan in
`frustrated_sync_dynamics/reports/frustrated_sync_dynamics_plan.md`:

- script: `frustrated_sync_dynamics/run_basin_benchmark.py`;
- task: first-order frustrated Kuramoto basin-stability prediction;
- label: high-sample vectorized Monte Carlo estimate of recovery probability;
- model: `SyncBasinSQNN`, a Bloch-vector graph surrogate with phase-shifted
  Kuramoto edge messages;
- baselines: low-budget Monte Carlo and a ridge regressor on graph/scenario
  features.

This code is isolated from the MaxCut/QUBO path.

## Transition Pilot

Output directory:

`outputs/frustrated_sync_basin_transition_64_256_1024`

Training:

- train sizes: 32, 64, 128;
- train scenarios: 10 per size;
- train labels: 256 perturbation trajectories per scenario.

Evaluation:

- test sizes: 64, 256, 1024;
- test scenarios: 3 per size;
- truth labels: 512 perturbation trajectories per scenario;
- low-MC baseline: 32 perturbation trajectories per scenario;
- projected high-quality MC reference: 4096 perturbation trajectories.

## Aggregate Results

| n | method | MAE vs 512-MC truth | online seconds | speedup vs 512-MC | projected speedup vs 4096-MC |
|---:|---|---:|---:|---:|---:|
| 64 | low_mc | 0.011719 | 0.043739 | 7.20x | 57.59x |
| 64 | sqnn_basin | 0.162790 | 0.005679 | 55.45x | 443.57x |
| 64 | feature_ridge | 0.333560 | 0.000153 | 2063.61x | 16508.90x |
| 256 | low_mc | 0.014323 | 0.057284 | 9.31x | 74.50x |
| 256 | sqnn_basin | 0.164212 | 0.004768 | 111.89x | 895.10x |
| 256 | feature_ridge | 0.045494 | 0.000159 | 3364.54x | 26916.28x |
| 1024 | low_mc | 0.040365 | 0.172902 | 6.20x | 49.59x |
| 1024 | sqnn_basin | 0.641597 | 0.008608 | 124.49x | 995.95x |
| 1024 | feature_ridge | 0.300903 | 0.000163 | 6579.89x | 52639.16x |

## Interpretation

The SQNN surrogate has the intended online speed shape, especially at n=1024,
where it is about 124x faster than 512-sample MC and roughly 996x faster than a
4096-sample MC estimate by linear projection.

However, this pilot does not yet beat classical baselines on error.  The
32-sample low-budget MC baseline is much more accurate on this test set, and a
simple feature ridge baseline is competitive on n=256.  SQNN handles one 1024
transition case well, but fails badly on two stable 1024 cases, so the current
model does not yet have reliable OOD scale calibration.

Current status:

- good: task pipeline, model, labels, timing comparison, and plots exist;
- good: online speed advantage is real;
- not good enough: accuracy is not competitive with low-budget MC;
- next fix: train on more large-like stable cases or add explicit scale
  calibration/readout features before claiming advantage.
