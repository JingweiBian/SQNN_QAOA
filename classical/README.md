# Classical MaxCut-3 Baselines

This folder contains classical tools for evaluating the current SQNN MaxCut-3
route against two denominators:

- `C/W`: cut fraction, where `W` is the total edge weight. This is the metric
  used by many earlier project tables.
- `C/C*`: strict approximation ratio, where `C*` is the exact MaxCut optimum.

For random 3-regular MaxCut, `W=3n/2` in the unweighted case, but `C*` is not
generally equal to `W`. A value should be called an approximation ratio only
when the exact optimum is known or when the denominator is explicitly named
as a best-known/upper-bound reference.

`maxcut3_compare.py` does three things:

1. Generates the same random 3-regular MaxCut instances used by the SQNN code.
2. Solves a CP-SAT MaxCut model to obtain an exact optimum when possible.
3. Runs three practical GW-style reference values:
   `GW-style expected` is the paper-aligned baseline
   `sum_edges arccos(v_i dot v_j) / pi`; `GW-style sampled-best` is the best
   cut among sampled hyperplanes; `GW-style + 1-bit greedy` additionally
   applies local search.

The GW implementation here is called `GW-style` because it uses a low-rank
Burer-Monteiro relaxation rather than a full certified SDP solve. The expected
hyperplane value is the closer analogue of the grey GW line in Augustino et al.
Figure 4. The sampled-best and `+ 1-bit greedy` values are stronger heuristic
references, but they are not the paper-aligned GW baseline.

SQNN round-trace columns use the following readout names:

- `direct`: deterministic SQNN rounding only, using `p_i >= 0.5`.
- `direct_greedy`: the `direct` bitstring after 1-bit greedy local search.

The first few rounds can have a strong `direct_greedy` score even when
`direct` is still near a random 0.5 cut fraction. In that case the quality is
coming mostly from the greedy post-processing, not from the SQNN state itself.

## Current C* Policy for n=512

For the fixed random 3-regular graph `n=512, degree=3, seed=42`, the current
certified bounds are:

- best known feasible cut: `711`
- CP-SAT certified upper bound: `719`
- therefore: `711 <= C* <= 719`

I also tried two additional proof routes:

- MaxSAT/RC2 encoding: did not finish even on the 256-node validation case
  within the short test budget, so it is not the preferred route here.
- OR-Tools SCIP MIP: validates the 256-node case, but on 512 nodes it returned
  only a `706` incumbent and a loose `734.48` bound after 300 seconds.
- CP-SAT decision test for `cut >= 712`: inconclusive after 300 seconds.

So for `n=512`, do not report `711` as proven `C*`. The safest paper-style
number is the conservative ratio to the certified upper bound:

```text
GW_expected / 719 = 0.9494
```

This already matches the Google-paper-like `0.95` scale. If a single
best-estimate denominator is needed for a discussion table, use either
`best-known` or the random-3-regular theoretical/typical denominator
`0.935~0.937 * |E|`, but label it explicitly and do not call it strict `C*`.

## n=512 Ten-Graph GW-Expected-Only Check

`n512_10_random_graphs.py` evaluates ten random 3-regular graphs at `n=512`
using seeds `0..9`. The report deliberately uses only the Google-paper-style
`GW expected` value as the classical baseline:

```text
GW expected = sum_edges arccos(v_i dot v_j) / pi
```

The plotted SQNN metrics are:

```text
SQNN expected C[p]  probability expected cut, no binary readout
SQNN C_d            direct p_i >= 0.5 readout
SQNN C_dg           direct readout + 1-bit greedy
SQNN C_s            best-of-K SQNN Bernoulli samples
```

Current output folder:

```text
outputs/classical_maxcut3_n512_10seeds
```

The aggregate report is `report.md`, with per-seed traces under
`seed_0` through `seed_9`.

To run the multi-head symmetry ensemble version:

```powershell
.venv\Scripts\python.exe classical\n512_10_random_graphs.py `
  --seeds 0 1 2 3 4 5 6 7 8 9 `
  --head-count 3 `
  --head-seed-stride 7919 `
  --device cpu
```

With `--head-count 3`, the default output folder becomes:

```text
outputs/classical_maxcut3_n512_10seeds_head3
```

This ensemble is still an SQNN readout experiment: each head uses the same
model design but a different symmetry seed, and the final probabilities are
mixed in logit space. It is meant to test whether the current weakness is
mostly instability from symmetry breaking rather than a lack of learning
capacity.

## n=512 Mechanism Scan and Current Clean Route

The mechanism scan output is:

```text
outputs/n512_mechanism_scan_combined
```

Best complete ten-seed candidate:

```text
edge_boost_mem060_no_xy
phase_mode            = memory_z_edge_cavity_collapse
phase_memory_decay    = 0.60
xy_feedback_init      = 0.0
collapse_init         = 0.06
z_message_gain        = 1.8
z_message_gain_final  = 2.6
```

Against GW expected on seeds `0..9`:

```text
C_d gap mean = +0.009560, wins 9/10
C_s gap mean = +0.007216, wins 8/10
expected gap = -0.008419
```

So the recommended clean route removes full-time XY feedback, keeps short
phase memory, and strengthens the MaxCut-specific z-edge/collapse channel.
`classical/n512_10_random_graphs.py` now defaults to this route:

```powershell
.venv\Scripts\python.exe classical\n512_10_random_graphs.py --device cpu
```

To reproduce the older V14 full-time XY route:

```powershell
.venv\Scripts\python.exe classical\n512_10_random_graphs.py `
  --model-config v14_memory_xy_z_edge_gain14 `
  --output-dir outputs/classical_maxcut3_n512_10seeds_v14 `
  --device cpu
```
