# V18 Dissipative MaxCut Direction

This folder is the next development target after freezing V14-UTC.

V18 is not a replacement for the V14-UTC sparse-graph story. It is the dense-graph direction suggested by the current probes.

## Current Evidence

Reference output:

```text
outputs/v18_dissipative_dense_probe/
```

The first sweep shows:

- V14-UTC is better at `d=3` and `d=4`;
- V18 catches up around `d=6`;
- V18 is clearly better than V14-UTC for denser graphs such as `d=12`, `d=16`, and `d=20`;
- V18 is much faster than the full V14-UTC pipeline.

## Development Goal

Turn V18 into a clean dense-graph dynamical model:

1. preserve the dissipative Bloch dynamics as the core mechanism;
2. avoid adding heavy classical portfolios;
3. compare against GW-style and simple greedy baselines at equal time;
4. keep V14-UTC as the sparse benchmark, not as something to keep modifying for dense graphs.

