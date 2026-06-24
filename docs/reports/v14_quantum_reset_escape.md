# V14 Quantum Reset Escape Probe

Date: 2026-06-24

This note records the first test of a quantum-driven basin escape for V14.
The escape is not a classical local search step.  It directly edits the V14
internal Bloch state at an intermediate round, then lets the same trained V14
parameters continue evolving.

## Reset operators tested

- `phase`: keep each selected node's Bloch Z coordinate, set Y to zero, and put
  the remaining radius back on positive X.  This clears accumulated phase while
  keeping the current Z/readout confidence.
- `partial`: move selected nodes part way back toward `|+>` by shrinking Z, then
  set Y to zero and restore positive X.
- `full`: reset selected nodes to `|+>` exactly, i.e. X=1, Y=0, Z=0.

Selected nodes were chosen from low-confidence nodes, endpoints of currently
uncut edges, or nodes with direct-readout positive flip gain.  I also tested
whether to keep auxiliary messages, clear only active-node incident messages, or
clear all phase/edge memories.

## n=512, 3-regular, seed 0

Base V14:

- best expected cut: `671.3739`
- best direct cut: `688`
- best direct+greedy cut: `694`

Main scan:

- output: `outputs/v14_quantum_reset_escape_n512_seed0`
- cases: `180`
- reset rounds: `120,160,200,240`
- modes: `phase`, `partial rho=0.3/0.6/0.9`, `full`
- selectors: `bad_low_conf`, `gain_low_conf`, `low_conf`
- fractions: `0.02,0.05,0.10`
- auxiliary policy: clear active-node memory only

Best result in main scan:

- best direct+greedy stayed at `694`
- best direct improved to `690`
- best expected improved to `672.713`

Auxiliary-memory scan:

- output: `outputs/v14_quantum_reset_escape_n512_seed0_auxscan`
- cases: `96`
- compared `clear_aux=none` vs `clear_aux=all`

Best focused result:

- `r160_full1.00_bad_low_conf_f0.100_none`
- best expected cut: `672.881`
- best direct cut: `690`
- best direct+greedy cut: `694`

Important observation:

- clearing all auxiliary memory is harmful: best expected falls to about
  `668.68`, and best direct+greedy falls to `692`.
- keeping auxiliary memory while resetting Bloch coordinates works best.

## n=1024, 3-regular, seed 0

Base V14:

- best expected cut: `1333.3463`
- best direct cut: `1369`
- best direct+greedy cut: `1379`

Reset scan:

- output: `outputs/v14_quantum_reset_escape_n1024_seed0`
- cases: `72`
- reset rounds: `160,200,240`
- modes: `phase`, `partial rho=0.9`, `full`
- selectors: `bad_low_conf`, `low_conf`
- fractions: `0.02,0.05`
- auxiliary policy: `none` and `active`

Best direct result:

- `r240_partial0.90_bad_low_conf_f0.050_none`
- best expected cut: `1333.506`
- best direct cut: `1372`
- best direct+greedy cut: `1379`

Best expected result:

- `r200_full1.00_low_conf_f0.050_active`
- best expected cut: `1333.601`
- best direct cut: `1370`
- best direct+greedy cut: `1379`

## Current conclusion

Quantum reset is real but weak in this first implementation:

- It improves the product-distribution expected cut.
- It improves direct readout by a small but repeatable amount.
- It does not yet break the direct+greedy plateau.
- It does not reach the 512-node `705` level found by tabu/breakout escape.
- It does not reach the 1024-node `1414` level found by tabu/breakout escape.

The most useful variant is:

- reset around the middle or late-middle of the trajectory;
- select bad-edge and low-confidence nodes;
- reset about 5%-10% of nodes for n=512, about 5% for n=1024;
- keep auxiliary phase/edge memories, or at most clear only active incident
  messages;
- avoid clearing all memories.

## Interpretation

The result supports the user's hypothesis that phase reset can make the SQNN
leave the exact same smooth trajectory, but it also shows that the trained V14
dynamics pulls the state back toward the same basin after reset.  The edge
messages and phase memory seem to contain useful global coordination; deleting
them makes the model forget too much structure and hurts quality.

This means quantum reset should be treated as a lightweight internal
diversification mechanism, not yet as a full replacement for tabu/breakout.
The next promising version is not "clear more"; it is to add a learned or
scheduled reset gate during training so the model learns how to recover from
resets instead of seeing them only at evaluation time.
