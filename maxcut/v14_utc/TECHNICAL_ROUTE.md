# V10 To V14-UTC Technical Route

## V10 Baseline

V10 is the first stable local-field monotone SQNN MaxCut baseline.

- State: one continuous variable/Bloch-like state per graph vertex.
- Driving signal: local MaxCut field from neighboring variables.
- Guard: monotone accept keeps expected objective from accepting obviously bad updates.
- Strength: stable and interpretable.
- Weakness: when the monotone mechanism becomes too strict, the state can stall. In scale tests the original V10 monotone path became unstable around the 4096-variable level through no-grad/stalled trials, which is useful as a stability upper-bound observation but not as a model improvement route.

The key lesson from V10 is that strict monotonicity protects the trajectory, but can also remove the ability to cross basin boundaries.

## V14 Dynamical Model

V14 keeps the same MaxCut objective but makes the optimizer more dynamical.

- State: per-variable Bloch vector, with `x/y` phase-plane motion and `z` readout bias.
- Field: local edge message plus cavity-style correction.
- Memory: phase and edge-memory terms keep useful directionality without fully freezing the state.
- Readout: direct `z` sign readout, expected objective `C[p]`, and optional direct+greedy evaluation.
- Default formal variant: `clean_edgeboost_mem060`, with memory decay around `0.60`.

V14 improves over V10 because it lets the state accumulate phase/history instead of acting like a purely local scalar relaxation. The important diagnostic discovery was that `C[p]` can look smooth while direct readout jumps suddenly. Those jumps are basin/readout reorganizations.

## Transition Diagnostics

The V14 diagnostics showed three recurring facts:

- Direct readout can have sharp positive jumps while expected energy remains smooth.
- Good escape timing often lies before the readout transition, roughly tens of rounds before the main peak.
- Late perturbations tend to stay inside the same basin because memory/phase have already locked in.

This changed the strategy from "keep perturbing when stuck" to "find the transition window and perturb only around it."

## V14-UTC

UTC means unified transition-conditioned escape.

The formal version used here is `UTC-SM-lite v3`:

- Run or load the V14 base trajectory.
- Detect the main direct-readout transition, preferring direct-readout peaks rather than direct+greedy peaks.
- Keep windows before the transition, typically around `peak-60`, `peak-55`, `peak-35`, and `peak-30`.
- Generate a small number of escape paths with soft monotone temperatures.
- Use a short recovery path after the escape.
- Evaluate candidates by direct/direct+greedy only at candidate boundaries, not every internal round.
- Keep the best candidate under the chosen score mode.

This is not a separate classical heuristic bolted on top. It is a controlled perturbation of the V14 state trajectory: the jump acts on the state, recovery lets the Bloch dynamics re-settle, and readout selects the path.

## Why This Became The Formal V14 Route

Several alternatives were tested: global Bloch anneal, bad-edge cluster anneal, quantum reset, q-tabu, group-targeted escape, full transition-conditioned soft monotone, and dense-specific rescaling. The useful part that survived was the transition-conditioned window plus a light soft-monotone candidate set.

The final route is intentionally conservative:

- preserve V14 as the main dynamical optimizer;
- use only one small escape layer;
- avoid repeated late perturbations;
- avoid heavy candidate portfolios;
- keep the method reproducible and explainable.

