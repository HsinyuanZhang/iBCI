# Corrected C2/C3 Posterior-Input Score V1

Score C2, C3-Const, C3-Real, and C3-Real with q rows shuffled on the same
within-6 and external-15 M4/M10/M30 records.  Materialize each target session
once.  Use the frozen deployment support selection: D-optimal first-30 at M4
and chronological at M10/M30.  Fit the source-prior posterior carrier on those
exact selected rows, normalize with the frozen source-only equal-budget
posterior normalizer, and never refit either normalizer on a target.

The old generic C2 score feeds ridge-T4.  Preserve that receipt as a diagnostic
but never call it the E02 result.  The corrected posterior-input score is the
governing E02/E03 result.

C3-Const consumes posterior side4 plus exact q=0.  C3-Real consumes the
source-normalized q.  The q-shuffle cell uses the same trained Real checkpoint
and deterministically permutes only q across the unit axis.  No extra training
is allowed.

Descriptor-reload the accepted T0/C1 deployment receipt and exact-match each
new cell's target, selected support, calibration prefix, window count, and
valid-row count.  Report C2-C1, Real-C2, Real-Const, and Real-q-shuffle paired
session deltas and seed-42 bootstrap intervals.

Before opening the corrected result, tail safety is fixed as a worst-session
R2 delta of at least -0.02 on the promoted external budget.  C2 additionally
requires external and within M30 deltas of at least -0.01.  E03 applies the
same -0.02 tail rule against both C2 and C3-Const.

Publish attempt before target/model access.  Target backward, optimizer, and
update counts are zero.  Execution code is strict.  This work order and its
test are review-only: drift is recorded as `ACCEPTED_NON_NUMERIC_DRIFT` and
cannot invalidate a numerical result.
