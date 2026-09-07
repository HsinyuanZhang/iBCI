# M1 support encoder Gate-A numerical correction receipt

**Frozen:** 2026-08-02 (Asia/Hong_Kong), after generation of v1 and before viewing a corrected
v2 result.  
**Scope:** implementation correction only; all scientific arms, folds, features, ridge values,
metrics, practical thresholds, and gates remain those frozen in
`M1_SUPPORT_ENCODER_GATE_A_PROTOCOL.md`.

## Defect in v1

The v1 candidate table exposed overflow for several unregularized inner-fold ridge candidates.
Their scores became non-finite. Python's ordinary `min` initialization could retain a first
non-finite `rate_only` candidate because comparisons against `NaN` are false. Therefore the v1
rate-only model-selection receipt and every decision involving that control are invalid.

The v1 directory is retained unchanged as an audit trail and is **superseded in full**. Its point
estimates, report, and disposition must not be cited as the Gate-A result.

## Frozen correction

1. Prediction and MSE computation may detect overflow but may not upper-clip, impute, or rescue it.
2. A candidate is `invalid_nonfinite` if any inner-fold prediction, MSE, ratio, or aggregate score
   is non-finite.
3. Invalid candidates remain in the JSON with explicit status/reason and null numeric fields, but
   are excluded from `min` selection.
4. Every selected outer model and every primary arm prediction/error must be finite or the script
   fails closed without issuing a decision.
5. At least one finite candidate must exist for E4 and rate-only in every outer fold.
6. The corrected output is a new immutable directory
   `sua_exploration/results/m1_support_encoder_gate_a_v2/` with a new schema/hash manifest.

No additional regularization values, feature forms, clipping constants, folds, resampling seeds,
or decision thresholds are introduced by this correction.
