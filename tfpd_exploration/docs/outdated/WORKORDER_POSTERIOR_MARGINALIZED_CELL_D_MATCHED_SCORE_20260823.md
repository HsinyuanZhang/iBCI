# Work Order: PMC-D Matched Score

Date: 2026-08-23 HKT

## Scope

This additive scorer audits the eventual terminal and final-four SWA provenance
for `POSTERIOR_MARGINALIZED_CELL_D_SEED42`.  It does not train, select a
checkpoint, open NWB/target/formal data, initialize CUDA, create a result root,
or publish a receipt.  Until a later root-reviewed capability supplies a valid
full-training terminal, final-four checkpoint map, SWA manifest, and SWA body,
all execution paths fail closed.

## Scientific contract

- Inference is the deterministic ordinary OLS point T4 carrier only.  A
  posterior sample, posterior mean, posterior normalizer, covariance,
  credibility value, or attention/PIRG bias is forbidden at inference.
- The governing estimator is the reviewed
  `tfpd_lane.matched_scorer.session_r2`: fixed bin 49 of every 50-bin window,
  per-session variance-weighted two-coordinate R2, then equal weight per
  session.  The detached contiguous CPU target and validity tensors must
  equal the prepared fixed-bin authority; a dynamically selected final-valid
  bin is not this contract.
- The fixed surfaces are exactly six within sessions and fifteen external
  sessions.  The same materialized input record is paired between PMC-D and
  sealed Cell-D, with zero target optimizer, backward, or update calls.
- Every M30/M10/M4 row carries the per-session PMC-D minus sealed Cell-D delta
  and exact sign.  The surface summary carries the deterministic seed-42
  paired bootstrap interval.
- Within M30 is a safety gate at mean delta >= -0.02.  External is governing:
  external M30 safety is required, while external M4 mean >= +0.03 and at
  least 9/15 positive M4 deltas determine the short-prefix screen.  A failed
  M30 safety is `STOP`; when both safety gates pass and the external M4
  headline/breadth screen passes, a nonnegative external M30 mean is
  `CLEAR_GO` and a negative external M30 mean is the explicitly predeclared
  `PROMISING__M4_SCREEN` verdict.  All remaining cases are
  `HOLD__DESCRIPTIVE_SCREEN`.

The public script is a standard-library dry-plan only.  The scorer source,
script, focused tests, this work order, the reviewed PMC training closure, and
the shared metric/baseline scorer are explicit closure inputs.  The physical
seam additionally binds the exact descriptor-safe `posterior_carrier_v1`
matched-score reader, Cell-D score helper, no-cache T4 helpers, model helper,
metric helper, and the descriptor-bound PMC core/source-adapter helpers by
source SHA; no glob or legacy result discovery is permitted.
