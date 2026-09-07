# C2/C3 accelerated full score V8

## Goal

Run the unchanged 252-cell C2/C3 matched score on GPU1 after the accepted V7
engineering-equivalence measurement.

## Engineering parity authority

V7 measured the first within-session C2/M4 cell:

- repeated GPU forward bitwise equal;
- maximum absolute CPU/GPU prediction difference 6.377696990966797e-6;
- absolute CPU/GPU R2 difference 1.043081283569336e-7;
- GPU batch 1024 with no fallback;
- peak CUDA allocation 576029184 bytes.

V8 transparently uses engineering tolerances 1e-5 prediction and 1e-6 R2.
These values were selected after the diagnostic, are not a scientific
preregistration, and are only an execution-equivalence guard. They are at
least four orders of magnitude smaller than the 0.01/0.02 scientific decision
margins. They do not alter any C2/E03 performance gate.

## Unchanged science

The surfaces, 21 sessions, M4/M10/M30 budgets, C2/C3-Const/C3-Real/
C3-Real-q-shuffle arms, inputs, checkpoint states, target masks, governing R2,
row order, paired summaries, and decision gates remain identical to V3.
Target optimizer, backward, and update counts remain zero.

## Drift policy

Execution and review closure are disjoint. Numerical execution drift fails
closed. Work-order/test/comment/review drift is
`ACCEPTED_NON_NUMERIC_DRIFT`, does not affect numerical acceptance, and never
requires restart. Final validation checks the post-execution runtime state and
does not reapply the pre-attempt-only no-Torch condition.

