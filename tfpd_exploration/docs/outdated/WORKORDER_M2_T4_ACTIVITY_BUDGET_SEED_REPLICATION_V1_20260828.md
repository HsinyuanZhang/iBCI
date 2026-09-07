# M2 T4 Activity-Budget Seed Replication V1

## Goal

Replicate the accepted frozen-weight M2 activity-budget screen with the already
trained T4 checkpoints for seeds 42, 43, and 44.  This is inference only.  It
does not train, fine-tune, select a checkpoint on query R2, or update a target
state.

## Matrix

For both seven-session `within_post30` and six-session
`external_official_query`, score the same five cells in the same order:

1. fixed-ridge T4 M30 + B3S M30;
2. fixed-ridge T4 M10 + B3S M10;
3. fixed-ridge T4 M10 + B3S M30;
4. D-opt fixed-ridge T4 M4 + matched four-trial B3S;
5. D-opt fixed-ridge T4 M4 + B3S M30.

Seed42 rows are reused from immutable score SHA
`6bdad93328ba26c12b8aa3afffcbc3490c312dfe22939b3d3bb3c5ec5b9005ce`.
Seeds43/44 rerun the exact query arrays with their frozen checkpoint bytes.
All three checkpoints use the same sealed T4 normalization SHA
`d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e`.

The primary replication statistic is the activity30-minus-static paired delta
for M10 and M4, first per seed and then summarized across the three seed-level
equal-session means.  Every row binds ordered query starts, targets, activity,
T4, predictions, and zero parameter updates.
