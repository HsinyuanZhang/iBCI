# Work Order: H1 Calibration-Profile FiLM V2

Date: 2026-09-04  
Status: `AUTHORIZED_GPU0_ONCE_AFTER_V1_NUMERICAL_ANCHOR_INCIDENT`

Execute the complete V1 CP-FiLM 2x2 experiment once under the V2 numerical
anchor addendum.  Use a fresh
`tfpd_exploration/h1_series_20260830/results/h1_calibration_profile_film_v2/`
root.  Bind and validate the immutable V1 attempt, training, two checkpoint,
and failure bodies listed in the incident document before reservation.

The only permitted behavior change is
`require_historical_prediction_sha=False` with historical R2 tolerance `1e-7`
and mandatory same-process repeated-prediction byte equality.  No training,
profile, pooling, support, target, scoring, or decision rule may change.
The fold-19250108 trained FiLM state digests must also match the V1 sealed
states exactly; failure to reproduce either state ends V2 without a retry.

All V1 prohibitions remain: source-only H1 date-LODO, logical GPU0 only, no
GPU1 query/touch, no formal held-out/EvalAI, no target updates, no all-source
refit/package/submission, and no modification of the 08:00 LP-R3 timer.
