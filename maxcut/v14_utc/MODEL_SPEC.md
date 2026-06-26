# V14-UTC Model Specification

## Task

For an unweighted graph `G=(V,E)`, solve MaxCut:

```text
C(x) = sum_(i,j in E) 1[x_i != x_j]
```

The model optimizes a product-state probability view and finally reports a deterministic Z-basis bitstring.

## State

Each vertex `v_i` carries a Bloch-like state:

```text
r_i = (x_i, y_i, z_i)
p_i = P(bit_i = 1) = (1 - z_i) / 2
```

Interpretation:

- `z_i` controls the final Z-basis readout bias.
- `(x_i, y_i)` stores phase-plane motion and memory.
- the graph edge structure controls how neighboring states push each other.

## V14 Base Evolution

For each optimization round:

1. Compute the local MaxCut field from neighboring probabilities/states.
2. Apply phase-aware update in the XY plane.
3. Apply short-memory and edge/cavity corrections.
4. Apply directed z-edge anti-correlation pressure.
5. Use late collapse/stronger z-edge gain near the end of optimization.
6. Evaluate expected cut `C[p]`.
7. Keep strict monotone behavior for the base V14 path.

Formal base configuration:

```text
clean_edgeboost_mem060
```

This is the frozen V14 baseline for V14-UTC. The old full-time XY-feedback route is not the formal route.

## Readout Metrics

The method tracks three related but distinct values:

```text
C[p]      expected cut under product probabilities
C_d      deterministic direct readout from sign(z)
C_dg     direct readout followed by a light greedy improvement
```

Important diagnostic fact:

```text
C[p] can be smooth while C_d jumps.
```

Those jumps are treated as readout/basin transition events.

## UTC Window Detection

UTC uses the base V14 trajectory to locate useful escape timing.

1. Record `C_d` over rounds.
2. Find the main direct-readout positive jump.
3. Prefer direct-readout peaks, not direct+greedy peaks, because greedy can hide the actual model transition.
4. Check that the transition happens while `C[p]` is not making a large unstable jump.
5. Generate candidate escape starts before that peak.

The formal lite schedule uses starts around:

```text
peak - 60
peak - 55
peak - 35
peak - 30
```

The exact legal starts are clipped to the available trajectory length.

## SM-Lite Escape

For each candidate window:

1. Restore the V14 state checkpoint near the candidate start.
2. Apply a small soft-monotone escape path.
3. Allow limited non-monotone movement inside the escape path so the state can cross a basin boundary.
4. Run a short recovery segment.
5. Evaluate the candidate after recovery.

Candidate temperatures:

```text
template
0.06
0.24
```

The lite version avoids per-round greedy checks inside the candidate path. This is important for speed: greedy is used at candidate boundaries, not as the internal optimizer.

## Selection

For each candidate, compute:

```text
direct score
direct+greedy score
expected score
runtime
```

The formal reporting score is `C_dg`, with `C_d` and `C[p]` kept for interpretation.

The final selected bitstring is the best candidate under the configured score mode. If no escape path improves the base path, the method can keep the base V14 result.

## What V14-UTC Is Not

V14-UTC is not:

- a pure classical tabu/greedy portfolio;
- a warm-start handed off to another solver;
- a dense-graph-specialized model;
- a repeated late perturbation strategy.

It is a transition-conditioned perturbation of a quantum-inspired Bloch dynamics trajectory.

