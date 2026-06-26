# Soft Probabilistic Basin Escape

Date: 2026-06-24

This note records the first GPU-batched implementation of Soft Probabilistic
Basin Escape for V14 MaxCut.

Implementation:

```text
maxcut/scripts/run_v14_soft_probabilistic_basin_escape.py
scripts/run_v14_soft_probabilistic_basin_escape.py
```

The repository-level script is only a compatibility wrapper.  The MaxCut-owned
implementation lives under `maxcut/scripts/`.

## Motivation

The previous soft global Bloch anneal has a clean single-trajectory dynamical
picture, but its node scores are still partly driven by hard readout features:

```text
bits_i = 1[p_i >= 0.5]
bad edge = bits_i == bits_j
```

That is useful, but it does not fully represent the probability distribution
being optimized.  SPBE replaces hard bad-edge features with a soft conflict
probability:

```text
q_ij = P[x_i == x_j]
     = p_i p_j + (1 - p_i)(1 - p_j)
```

For MaxCut, `q_ij` is the probability that edge `(i,j)` is uncut under the
current product distribution.

## Dynamical Field

For each batched trajectory, the script computes:

```text
node_conflict_i = sum_j q_ij / degree_i
entropy_i       = 4 p_i (1 - p_i)
grad_pressure_i = sum_j z_j / degree_i
cluster_i       = sum_j q_ij z_j / degree_i
```

with Bloch convention:

```text
p_i = (1 - z_i) / 2
```

The RY escape field is:

```text
pressure_i = guidance * grad_pressure_i + cluster_strength * cluster_i

rho_i = rho_floor
      + (1 - rho_floor) * normalize(
          conflict_weight * node_conflict_i
        + entropy_weight  * entropy_i
        + pressure_weight * |pressure_i|
        )^rho_power

theta_i =
    temperature * envelope(t) * rho_i * pressure_i
  + temperature * noise * envelope(t) * rho_i * gaussian_i
  + memory_strength * memory_i
```

Then the script applies a batched Bloch RY rotation to `[B, n, 3]`.

This is the main conceptual difference:

```text
old soft global: hard-readout conflict score, one trajectory
SPBE: probability-conflict field, many GPU-batched trajectories
```

There is no tabu, branch lookahead, selected continuation, or local-search
operator inside the SPBE dynamics.  Greedy descent is only used as an auxiliary
score after trajectories are produced.

## GPU Acceleration

The escape stage is vectorized in torch:

```text
state shape: [batch, n, 3]
probability shape: [batch, n]
edge conflict q_ij: [batch, m]
```

Expected cut is evaluated by the existing batched QUBO API:

```text
problem.expected_energy(probabilities)
```

Observed speed on the local A100 machine:

| run | trajectories | wall time | device |
|---|---:|---:|---|
| conservative scan | 1024 | 5.43s | cuda:0 |
| strong scan | 4096 | 6.72s | cuda:0 |

The first sandbox smoke test fell back to CPU because PyTorch could not
initialize CUDA inside the sandbox.  In the approved non-sandbox execution,
PyTorch saw all 4 GPUs and the runner used `cuda:0`.

## n=512 Seed 0 Results

Baseline V14:

| metric | value |
|---|---:|
| best C[p] | 671.373 |
| best direct C | 687 |
| best direct+greedy C | 694 |

Conservative SPBE scan:

```text
outputs/v14_spbe_gpu_n512_seed0_scan1
```

| metric | value |
|---|---:|
| trajectories | 1024 |
| seconds | 5.43 |
| best C[p] | 671.873 |
| best direct C | 694 |
| best sampled C | 688 |

Strong SPBE scan:

```text
outputs/v14_spbe_gpu_n512_seed0_strong_scan
```

| metric | value |
|---|---:|
| trajectories | 4096 |
| seconds | 6.72 |
| best C[p] | 671.522 |
| best direct C | 694 |
| best sampled C | 689 |

## Interpretation

SPBE is currently successful as a GPU-batched probability-state escape probe:

```text
fast batch dynamics
no classical search inside the dynamics
conflict signal comes from q_ij, not hard bad edges
probability state does not collapse in the best conservative cases
```

But this first version does not yet open the 700+ basin:

```text
best direct remains 694
best direct+greedy remains 694 in scored top cases
best sampled cut remains below the previous soft global 702-level result
```

The important diagnostic is that the pure soft probabilistic field behaves more
like a stable distribution refinement around the V14 basin than a basin jump.
This suggests the missing component is not GPU throughput, but a stronger
model-side recovery mechanism after the probability field crosses a basin
boundary.

## Next Model Direction

The next SPBE version should keep the probability-conflict field, but add a
bounded recovery phase:

```text
1. apply GPU-batched SPBE perturbation;
2. keep top probability states by C[p], direct C, and sampled C;
3. run a short V14 recovery window from those soft states;
4. accept recovery only if C[p] or direct C recovers above the pre-escape state.
```

This still keeps the main mechanism model-side and probabilistic.  It avoids
tabu/branch search, but gives the Bloch state time to reorganize after the
soft conflict field pushes it out of the original basin.

## Useful Commands

Small smoke:

```bash
python scripts/run_v14_soft_probabilistic_basin_escape.py \
  --device cuda:0 \
  --trials 1 \
  --batch-size 4 \
  --steps 4 \
  --start-rounds -1 \
  --sample-count 4 \
  --greedy-top-k 4 \
  --output-dir outputs/_debug_spbe_gpu_smoke_cuda_final
```

Conservative scan:

```bash
python scripts/run_v14_soft_probabilistic_basin_escape.py \
  --device cuda:0 \
  --trials 16 \
  --batch-size 64 \
  --sample-count 16 \
  --greedy-top-k 24 \
  --output-dir outputs/v14_spbe_gpu_n512_seed0_scan1
```

Strong scan:

```bash
python scripts/run_v14_soft_probabilistic_basin_escape.py \
  --device cuda:0 \
  --trials 16 \
  --batch-size 256 \
  --start-rounds -1 \
  --steps 24,48,80 \
  --temperatures 0.50,0.80,1.20 \
  --guidances 0.4,0.8,1.2 \
  --cluster-strengths 1.6,2.5,3.5 \
  --noises 0.20,0.50,0.80 \
  --rho-floors 0.02,0.05 \
  --transverse-strengths 0.0,0.04,0.08 \
  --z-shrinks 0.0,0.03 \
  --sample-count 16 \
  --greedy-top-k 16 \
  --output-dir outputs/v14_spbe_gpu_n512_seed0_strong_scan
```
