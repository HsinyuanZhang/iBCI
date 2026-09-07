# Source posterior-q / direction-error audit V2

V2 binds the exact V1 source-materialization failure. V1 assumed every one of
the first 50 rewarded trials had a finite direction. One source session has a
single NaN at fixed chronological position 27.

V2 keeps the exact first-50 pool and never refills from later trials. It uses
an explicit finite-direction mask for each nominal support prefix and for the
disjoint positions 30:50 reference. Every receipt row records nominal and
effective support counts, invalid positions, and mask/input digests. M4 and
M10 retain their exact nominal counts for all 27 sessions. The affected M30
row is explicitly M29-effective and M30 remains descriptive only.

All correlation gates, thresholds, roster, prior, reference estimator,
bootstrap, and source-only boundaries are unchanged. No target, model, Torch,
CUDA, optimizer, backward, or update is allowed. Review-only drift is accepted
without restart and cannot affect numerical acceptance.

