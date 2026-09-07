# Budget-Matched Posterior CAL-AUG C3 Runtime V1

Date: 2026-08-30

## Purpose

Run the E03 angular-reliability successor immediately after the accepted C2
runtime smoke.  This is a DANDI 000688 sub-C source-only training route.  It
does not open within, external, or formal target data.

## Arms

- `constant`: exact C2 posterior carrier plus a fifth column equal to the
  source q mean.  After source normalization that column is exact zero.
- `real`: exact C2 posterior carrier plus source-normalized posterior angular
  reliability q.

Both arms use the same five-dimensional B3S encoder.  The four-dimensional C2
initial state is strict-loaded first.  The old encoder weights are copied
exactly and only the new q column is initialized to exact zero.  Therefore the
arms have the same initial state; the real arm can learn q while the constant
arm controls for the wider input layer.

The q normalizer uses source rows only.  Rows are concatenated in M4, M10,
M30 order, with the strict source roster and unit order inside each budget.
Each budget contributes the same number of rows.  Population mean/std are
used.  Target refitting is forbidden.

## Runtime smoke

Run 120 optimizer steps per arm using the exact global M30, M10, M4 cycle,
batch size 32, seed 42, Cell-D dropout law, optimizer, LR schedule, source
roster, posterior authority, and initial checkpoint used by C2.  The smoke is
non-performance evidence.  It must prove finite loss/gradients/state, exact
budget counts, identical initial arm state, constant q weight remaining zero,
and real q weight receiving a nonzero update.

## Closure policy

The execution closure contains only code and immutable assets that the live
route imports or executes.  It is strict.  Documentation and tests are kept
in a separate review closure.  Review-only drift is reported as
`ACCEPTED_NON_NUMERIC_DRIFT` and cannot fail or restart a completed numerical
run.  Any execution-closure drift still fails closed.

## Full training and evaluation

After the smoke passes, train both arms for the matched 48 epochs and 1,628,400
optimizer steps.  Score C2, C3-Const, and C3-Real on identical within/external
M4/M10/M30 records.  Run deterministic q-row shuffle only on the trained
C3-Real checkpoint.  The E03 claim requires Real to beat both C2 and Const,
and the same-checkpoint row shuffle to remove the relevant gain.
