# APFC V2 — source-only audit and refit successor

This is an additive successor to the immutable APFC V1 source screen.  Its
V1 body literals are deliberately deferred until V1 has published a terminal
graph.  It must never modify or retry that root.

## Scope

- GPU0 only after V1 has completed; no target data, no GPU1, and no source
  hyperparameter search.
- Descriptor-validate the completed V1 graph, restore each selected gate
  vector from the selected history row, and replay the exact same two lexical
  validation sessions with both learned and IEEE `+0.0` native gates.
- Recompute the frozen scalar-control conditions and capacity gates.  If no
  capacity arm is eligible, terminally stop.  If one or more are eligible,
  choose the frozen winner rule, reset *only* its gate and Adam state, and
  refit for its selected epoch on all seven source sessions.

## Required evidence

- zero/native exact anchor; DC2 trial-order permutation invariance;
- coordinated versus separate gate algebra (`prediction max abs <=2e-6`,
  `R2 abs <=2e-7`);
- V1 selected parameter vectors exactly restored before replay;
- scalar coefficient negative, both validation deltas nonnegative, and scalar
  mean gain within 0.001 of `+0.0022026004`;
- each capacity arm: delta versus scalar >=.003, both positive, worst >=-.002,
  and delta versus exact zero >=.005.
