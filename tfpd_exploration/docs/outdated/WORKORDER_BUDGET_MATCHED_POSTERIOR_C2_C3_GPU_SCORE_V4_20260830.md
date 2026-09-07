# Accelerated C2/C3 Posterior-Input GPU Score V4

## Goal

Finish the exact V3 C2/C3 matched matrix without waiting for the chunk-32
single-thread CPU reference.  V3 remains untouched and may finish naturally.
V4 is a separate result and does not reinterpret or overwrite V3.

## Numerical contract

- Same immutable C2, C3-Const and C3-Real SWA checkpoints.
- Same posterior input, q normalizer, selected support, target rosters, target
  windows, last-bin mask, comparator rows, R2 formula and promotion gates.
- Same 252-cell order: within then external, session order, M4/M10/M30, then
  C2/C3-Const/C3-Real/C3-Real-q-shuffle.
- Target backward, optimizer and update counts remain zero.

Only the evaluation device and logical decode batch change.  Use physical
GPU1 through `CUDA_VISIBLE_DEVICES=1`, disable TF32, and try logical batch
1024 with deterministic OOM fallback to 512 then 128.

Before the full matrix, materialize the first within session once and compare
the C2/M4 prediction from the frozen CPU batch-32 path with two repeated GPU
forwards.  Continue only when:

- repeated GPU predictions are bitwise identical;
- maximum absolute CPU/GPU prediction difference is at most `2e-6`; and
- absolute R2 difference is at most `2e-7`.

The smoke prediction is reused as the first full-matrix cell; it is not an
extra target selection.  Speed is descriptive and never rescues parity or
scientific gates.

## Lifecycle and drift

Validate V1/V2 failed predecessors, all immutable producers, source authority,
baseline score, canonical data roots, Python namespaces, GPU profile and a
fresh V4 root before attempt publication.  No target/model/checkpoint body is
opened before attempt.

Numerical execution closure drift remains fail-closed.  Work-order and test
drift records `ACCEPTED_NON_NUMERIC_DRIFT` with
`numerical_acceptance_affected=false` and does not trigger rerun.

Canonical result root:

`tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/c2_c3_posterior_gpu_score_v4`

This work order authorizes one V4 GPU1 execution after focused no-CUDA tests,
closure verification, exact environment/profile preflight and root freshness.
