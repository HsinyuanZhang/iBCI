"""Z4 — U-stability diagnostic (RUNBOOK; IMPLEMENTED 2026-08-25).

Status: RUN COMPLETE (pure CPU linear algebra).  Implementation:
``z4_subspace.py`` (same package); receipt:
``results/calibration_gap_v1/z4_subspace.json`` (+ .sha256, 0444).
Result summary: median k=6 theta_max(U_M, U_ref) over the external-15 =
56.1 deg at M4 / 38.4 deg at M10 (thresholds 35/20 deg) -> U starves in the
strict regime; the MEAN principal angle is far lower (23.8/13.8 deg), i.e.
the leading directions stay aligned while the deeper top-k directions rotate
away.  See the receipt's verdict block for the pre-registered reading.

Original runbook (spec only). Question: in the STRICT total-calibration regime the only
target-session activity available for Path-1's neural-covariance subspace U
is the M calibration trials — is U estimated from M4/M10 trials even stable
enough to restrict the carrier's column space?

Protocol (zero labels, zero training, CPU-feasible):
  for each external-15 session:
    1. materialize the full-session binned activity (the decode stream
       already opens it read-only in the diagnostics machinery);
    2. U_ref = top-k principal subspace (k in {4, 6, 8, 10}) of the M30
       calibration-window covariance;
    3. for M in (4, 10): U_M = same estimator on the first-M-trial windows;
       bootstrap 200 resamples of the M-trial windows for a CI;
    4. report per (session, M, k): principal angles between U_M and U_ref,
       effective rank of the M-window sample covariance vs N (units), and
       Ledoit-Wolf-shrunk variant alongside raw.
Reading (pre-registered):
  - median top-subspace angle (k=6) < ~20 deg at M10 and < ~35 deg at M4
    -> U is usable; Path 1 proceed.
  - angles near random (~90 deg) at M4 -> U starves in the strict regime;
    Path 1 requires the Z6 contract answer (transductive evaluation-stream
    activity) or must be re-scoped to M10+.
Coupling (handoff §3 P1): this diagnostic bounds Path 1 BEFORE any build.
"""
