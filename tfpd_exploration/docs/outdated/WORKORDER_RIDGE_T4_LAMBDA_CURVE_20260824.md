# Workorder: Short-Prefix Ridge-T4 Lambda Curve

Purpose: complete the low-cost M4/M10 comparison without training.  Evaluate
one frozen normalized-ridge grid on the sealed Cell-D seed-42 model, using the
same within-6/external-15 inputs, last-bin variance-weighted R2, equal-session
aggregation, and ordinary source OLS normalizer as the Phase-1 comparator.

Grid: 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10.  Budgets: M4 and M10.
Both information regimes are reported: B3S M30 with T4 label budget M, and
B3S/T4 both limited to M.  A diagnostic lambda may be selected from within
means only; external values must never select or alter the grid.  This is a
curve/selection diagnostic, not a new governing claim.  No target gradients,
updates, formal evaluation, scheduler changes, or model changes are allowed.

