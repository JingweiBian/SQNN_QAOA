# SQNN Dynamics Plan: Frustrated Oscillator Basin Stability

## 1. Core Position

This direction should not be framed as "SQNN solves a hard combinatorial
optimization problem." The stronger framing is:

> SQNN is a fast amortized surrogate for graph-coupled nonlinear phase dynamics,
> especially for frustrated oscillator networks where basin statistics require
> many nonlinear trajectory simulations.

The target task is not a single trajectory. Single trajectory simulation is a
classical strength. The target task is repeated stability evaluation over many
initial perturbations, fault scenarios, coupling strengths, noise levels, and
graph instances.

## 2. Are There Strong Fast Classical Algorithms?

Yes, but their coverage is limited. We should be explicit:

- Fast ODE/SDE solvers exist for one trajectory. RK4, adaptive Runge-Kutta,
  LSODA-style solvers, and GPU vectorized integrators are strong baselines.
- Spectral, linearized, master-stability, and Dörfler-Bullo-style conditions can
  give fast synchronization certificates for certain Kuramoto/power-grid regimes.
  They are valuable baselines, not weak strawmen.
- Topological proxies such as flow betweenness can correlate with basin
  stability and can be much faster than full Monte Carlo.
- Generic ML surrogates also exist in principle. We must compare against GNN,
  GNN-GRU rollout, Neural ODE/message-passing dynamics, and simple feature
  regressors.

The opening is narrower:

- General basin stability is typically estimated by Monte Carlo sampling of
  nonlinear dynamics; this is computationally expensive.
- Fast analytical conditions are often sufficient/certifying, not full basin
  probability estimators.
- Near transition boundaries, under signed/frustrated couplings, heterogeneous
  frequencies, noise, and multi-attractor dynamics, certificates can become
  inconclusive and direct simulation becomes expensive.

So the claim should be:

> There is no GW-like universal, fast, high-accuracy classical baseline for
> frustrated oscillator basin statistics. The best classical reference is
> expensive multi-start simulation, plus fast but incomplete certificates and
> heuristics.

This is an amortized prediction problem, not a proof of classical intractability.

### Practical Classical Accuracy

The classical reference is strong when the question is a single trajectory:

- A high-quality ODE solver can accurately integrate one disturbance trajectory.
- GPU/vectorized simulation can run many trajectories in parallel.
- If the system is clearly stable or clearly unstable, even low-budget Monte
  Carlo can be enough.

The cost appears when the output is a probability:

```text
B = number of recovered trajectories / number of sampled perturbations
```

For Monte Carlo with `M` samples, the binomial standard error is approximately:

```text
sqrt(B * (1 - B) / M)
```

The transition region is the hardest. If `B ~= 0.5`:

```text
M = 100      -> standard error ~= 0.050
M = 1,000    -> standard error ~= 0.016
M = 10,000   -> standard error ~= 0.005
```

Thus, a reliable probability map over many graph/parameter/fault scenarios can
require thousands of trajectories per scenario. SQNN should not claim to beat
classical solvers on one trajectory. It should aim to beat low-budget Monte
Carlo and generic learned surrogates when many repeated probability queries are
needed.

## 3. Application Framing

Primary application:

- Fast stability screening for power-grid-like oscillator networks.

Secondary applications:

- Design screening for Josephson/laser/spin-torque oscillator arrays.
- Synchronization-risk prediction in coupled neural oscillator models.
- Fast phase-diagram estimation for frustrated XY/Kuramoto-like systems.

The most defensible first application is power-grid-style stability screening,
because basin stability is already used to describe recovery after perturbations.

## 4. Mathematical Task

Start with two model families.

First-order frustrated Kuramoto:

```text
d theta_i / dt = omega_i + sum_j K_ij sin(theta_j - theta_i - alpha_ij) + noise
```

Second-order power-grid Kuramoto:

```text
d^2 theta_i / dt^2 = P_i - D_i d theta_i/dt
                     - sum_j K_ij sin(theta_i - theta_j)
```

For each graph and parameter setting, estimate:

- final synchronization probability;
- final order parameter mean and variance;
- locked fraction;
- basin entropy or multi-attractor label distribution;
- risk class: stable, marginal, unstable, multi-stable.

The expensive label is produced by many initial perturbations:

```text
B(G, params) = fraction of sampled perturbations that recover synchrony
```

The total simulation cost scales roughly as:

```text
cost ~= graph_size * scenario_count * perturbation_samples * integration_steps
```

For sparse graphs, the dominant per-step cost is usually proportional to the
number of edges. A representative large synthetic case:

```text
n = 512
average_degree = 6
edges ~= 1536
M = 5,000 perturbations
T = 10,000 integration steps

edge updates for one scenario ~= 5,000 * 10,000 * 1,536 ~= 7.7e10
```

If a phase diagram or fault study uses hundreds of scenarios, the repeated
simulation cost becomes the bottleneck. This is the intended opening for
amortized prediction.

## 5. Why SQNN Fits

SQNN already has the relevant inductive bias:

- Bloch X/Y coordinates represent phase-like variables.
- RZ is a native phase rotation.
- RY changes population bias / amplitude-like readout.
- Graph message passing naturally represents local coupling.
- Noise layers map cleanly to phase noise, bit-flip, dephasing, or uncertain
  perturbations.
- Multi-round updates are a learned discrete-time dynamical system.

The model should be adapted to predict either:

- direct basin statistics from graph + parameters; or
- a coarse rollout of order parameters and node-level phase confidence.

## 6. Baselines

Classical physics baselines:

- full Monte Carlo trajectory simulation with high sample count;
- low-budget Monte Carlo with equal wall-clock budget;
- spectral / Laplacian / linearized stability features;
- Dörfler-Bullo-style synchronization certificates where applicable;
- flow-betweenness and topology-feature regressors.

Learning baselines:

- MLP on graph-level handcrafted features;
- vanilla GNN;
- GNN-GRU rollout;
- Neural ODE / message-passing ODE;
- Transformer/message-passing model if graph sizes remain moderate.

SQNN must beat at least:

- low-budget Monte Carlo at equal time;
- vanilla GNN on out-of-distribution graph size or frustration strength;
- topology-feature regressors near transition regions.

## 7. Experiment Design

Scale should be chosen so that classical direct simulation is a real bottleneck.
The advantage is unlikely to appear for tiny grids with a few dozen nodes and
only tens of perturbation samples.

Phase 1: small pilot.

- Graph sizes: 32, 64, 128.
- Graphs: random regular, Erdos-Renyi, small-world, power-grid-like synthetic.
- Couplings: positive, signed, and phase-lag/frustrated.
- Labels: 512 to 4096 perturbations per graph-parameter setting.
- Scenarios: 50 to 100 graph/parameter/fault settings.
- Goal: identify regimes where stability probability is neither 0 nor 1 and
  fast analytical baselines are inconclusive.

Phase 2: scaling.

- Train on n <= 128 or n <= 256.
- Test on n = 512, 1024, optionally 2048.
- Labels: 4096 to 10000 perturbations for selected test settings.
- Equal-time Monte Carlo baseline: 32, 64, or 128 perturbations, depending on
  the measured SQNN inference budget.
- Compare inference time and error against direct simulation.

Phase 3: application demo.

- Pick one power-grid-like benchmark family.
- Run high-sample Monte Carlo as ground truth on selected cases.
- Use 100 to 1000 coupling/noise/fault scenarios.
- Show SQNN predicts the stability map over coupling strength and perturbation
  strength at much lower online cost.

Target advantage region:

```text
n >= 512
scenario_count >= 100
perturbation_samples needed for stable labels >= 1,000 to 10,000
```

Small cases remain useful for debugging and ablation, but the main speedup claim
should be made in this large repeated-query setting.

## 8. Main Figures

1. Speed-error curve:
   x-axis = wall-clock time per instance, y-axis = basin-stability error.

2. Stability phase diagram:
   coupling strength vs frustration/noise strength, color = basin stability.
   Show Monte Carlo truth, SQNN prediction, and error map.

3. OOD scaling:
   train small graphs, test larger graphs. Compare SQNN vs GNN vs low-budget MC.

4. Classical certificate coverage:
   show where fast sufficient conditions certify stable/unstable and where they
   are inconclusive. SQNN should target the inconclusive region.

## 9. Success Criteria

Strong success:

- 100x or larger online speedup over high-sample Monte Carlo;
- basin-stability MAE <= 0.03 to 0.05 on held-out graphs;
- better calibration and OOD scaling than vanilla GNN;
- strongest performance in transition/frustrated regimes.

Minimum publishable pilot:

- SQNN beats equal-time low-budget Monte Carlo and vanilla GNN on transition
  regimes;
- clear evidence that physics-structured rotations improve rollout stability or
  basin-statistic prediction.

Stop or pivot if:

- simple spectral/topological features predict the target almost as well;
- low-budget Monte Carlo is already accurate enough at the same wall time;
- SQNN does not outperform ordinary GNN baselines after controlled tuning.

## 10. Immediate Next Steps

1. Implement a simulator for first-order frustrated Kuramoto and second-order
   power-grid Kuramoto.
2. Implement basin-stability label generation with vectorized initial
   perturbations.
3. Build a small benchmark table of direct simulation cost for n=64, 128, 512.
4. Add baseline feature regressors and low-budget Monte Carlo.
5. Adapt SQNN input features from QUBO edges to oscillator edges:
   `K_ij`, `alpha_ij`, node frequency/power, damping, noise strength.
6. Produce the first speed-error and phase-diagram plots.

## 11. References

- Kim, Lee, Holme, "Building blocks of the basin stability of power grids",
  arXiv:1602.01712. The paper states that basin stability for power grids is
  defined by recovery from phase/frequency perturbations and that computing it
  requires Monte Carlo sampling of nonlinear systems, making it expensive.
- Dörfler, Chertkov, Bullo, "Synchronization in Complex Oscillator Networks and
  Smart Grids", arXiv:1208.0045. This provides strong closed-form
  synchronization conditions and should be treated as a serious classical
  baseline/certificate.
- Menara, Baggio, Bassett, Pasqualetti, "Stability Conditions for Cluster
  Synchronization in Networks of Heterogeneous Kuramoto Oscillators",
  arXiv:1806.06083. This supports the existence of useful analytical cluster
  synchronization conditions, again as baselines rather than strawmen.
- Sahabandu, Clark, Bushnell, Poovendran, "Submodular Input Selection for
  Synchronization in Kuramoto Networks", arXiv:2003.12733. Signed and
  heterogeneous Kuramoto synchronization is a known hard regime for control and
  stabilization.
