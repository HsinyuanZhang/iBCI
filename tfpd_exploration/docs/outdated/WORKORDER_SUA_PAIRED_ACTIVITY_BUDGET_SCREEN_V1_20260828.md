# Workorder: Paired SUA / pseudo-MUA activity-budget screen V1

Date: 2026-08-28

## Purpose

Complete the lowest-cost missing cross-granularity experiment without rerunning
the accepted 450-cell T4 label-budget curve or the accepted 150-cell classical
controls.

Use the immutable paired-view C1 `shared_t4` checkpoints for seeds 42, 43, and
44. Score the same 15 external sub-M sessions and the same query windows wholly
after rewarded trial 50 in both sorted-SUA and deterministic electrode-summed
pseudo-MUA views.

## Cells

Reuse, do not recompute:

- `ols_m10_activity30_reference`: the accepted M10 label-budget prediction with
  chronological first-30 B3S activity.

Compute exactly three new cells:

- `ols_m10_activity10`: the exact production first-10 grouped-direction OLS T4
  with chronological first-10 B3S activity;
- `ridge_m4_activity4`: four D-optimal cue directions selected inside the fixed
  first-30 candidate pool, fixed normalized ridge lambda 0.1, and only those
  four selected neural trials in B3S;
- `ridge_m4_activity30`: the identical M4 carrier and query, with chronological
  first-30 neural activity in B3S.

This produces 270 new forward rows and 90 immutable reference rows. The primary
contrasts are activity30 minus activity10 at M10 and activity30 minus activity4
at M4, followed by the paired SUA-minus-pseudo-MUA contrast of those gains.

## Invariants

- No target-session gradient, backward call, optimizer step, model update, or
  carrier update.
- Target rows, query starts, checkpoint, normalizers, session order, and metric
  are fixed within every paired contrast.
- Pseudo-MUA trial rates are summed by the frozen unit-to-electrode mapping
  before fitting the carrier. Fitted SUA rows are never pooled after fitting.
- M4 is explicitly cue-budgeted: D-optimal selection may read only the first-30
  candidate cue directions and uses only four trials in the carrier fit.
- The governing metric is last-bin two-coordinate variance-weighted R2 followed
  by equal session and equal seed weighting.
- Cached-identity inference must agree with the ordinary eager path within
  maximum absolute tolerance `1e-5` for every view, seed, and new cell. This
  bounded FP32 allowance covers the observed B1-versus-expanded-batch identity
  reduction difference; it does not relax target equality or the R2 metric.
- Deterministic CUDA execution requires `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and physical GPU1 as the sole visible device.
- GPU1 is the default device. GPU0 and unrelated jobs are not touched.

## Output

Publish one immutable `score.json` and canonical SHA sidecar under
`tfpd_exploration/results/sua_paired_activity_budget_screen_v1`. Include every
per-session/seed row, paired bootstrap intervals, sign counts, input hashes,
cached/eager parity evidence, elapsed time, and peak protocol information.
