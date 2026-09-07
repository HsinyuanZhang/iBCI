# M2 T4 Activity-Budget Screen V1

## Question

Does the activity-memory half of CDM transfer to FALCON M2 when label budget is
small?

This is an inference-only screen using the already selected seed-42 M2 B3S+T4
checkpoint.  It does not train a new network and it does not update the carrier
from query predictions.

## Frozen inputs

- Task: FALCON M2, 96 channels, 50-bin decoder window.
- Checkpoint SHA-256:
  `25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e`.
- T4 normalization SHA-256:
  `d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e`.
- Support selection: M4 uses a causal D-optimal subset of the non-centre
  directional trials inside the first 30 calibration trials; M10 and M30 use
  the chronological first M trials.  This matches the established short-budget
  rule and avoids pretending that a centre target supplies a direction label.
- T4 estimator: support-only fixed ridge, normalized lambda 0.1, ridge on
  cosine/sine coefficients and an unpenalized intercept.
- Decoder, identity encoder, normalization, and all network weights remain
  frozen.  There are no target gradients or parameter updates.

## Cells

For each budget M in {4, 10, 30}:

1. `ridge_static_mM`: carrier and B3S activity both use the exact selected M
   support trials (D-opt-first30 at M4, chronological at M10/M30).
2. `ridge_activity30_mM`: carrier still uses only `[0:M)`, while B3S activity
   uses `[0:30)`.

At M30 these definitions are identical and only one cell is run.  Therefore
the order per surface is:

1. `ridge_static_m30`
2. `ridge_static_m10`
3. `ridge_activity30_m10`
4. `ridge_static_m4`
5. `ridge_activity30_m4`

## Surfaces

- `within_post30`: all seven held-in calibration sessions, with only complete
  50-bin windows whose first bin is at or after the start of trial 30.
- `external_official_query`: all six held-out M2 query sessions.  Their
  calibration file is separate, so every official query window is after the
  frozen calibration prefix.

Each session is scored by last-bin, variance-weighted two-coordinate R2.  The
surface aggregate is the equal-session mean and median.  The primary contrasts
are activity30 minus static at M10 and M4, paired by session.

## Interpretation

- Positive M4/M10 contrasts support a full M2 CDM successor.
- Near-zero contrasts say the DANDI activity mechanism does not transfer to
  M2 under the frozen checkpoint.
- M30 is a safety anchor, not an optimization target.
- This result is local/mechanistic.  Official online CDM still requires a
  trial-boundary or change-point interface.
