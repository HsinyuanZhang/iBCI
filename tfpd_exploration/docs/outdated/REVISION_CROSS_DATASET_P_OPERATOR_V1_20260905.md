# Named revision — P operator, parent, and differentiable ridge

Date: 2026-09-05
Status: **NAMED_REVISION__STAGE1_NOT_GPU_FROZEN**
Parent workorder: `WORKORDER_CROSS_DATASET_FUNCTIONAL_CALIBRATION_V1_20260905.md`
Stage0 root (immutable): `tfpd_exploration/results/cross_dataset_functional_calibration_v1/20260905_113700/`
This revision root: `tfpd_exploration/results/cross_dataset_functional_calibration_v1/20260905_122000/`

This document is the missing authority for workorder §5.3 items 1–3. It does **not**
authorize a GPU pilot. Astra still has to freeze Stage1. No decoder scores were
inspected to choose these operators.

## 1. Parent bytes

| Role | Path | SHA-256 | Verdict |
|---|---|---|---|
| P consumer initialization | `tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/pilot_r3/s_fix/epoch_011.pt` | `7976e0b064fc4d92396b38a8e385aaa78379f0330c52ba7244bb45831b72178a` | **bound parent** |
| Fold-0 source teacher | `.../m1_afc4_source_decoder_fold0/...e1r1_fNone_s42/.../epoch_018.ckpt` | `f2921cabea819fed58b15e169f9cb899472416d30ee5a9b12c4c2087e96cb6be` | lineage only |
| Z-Fix `epoch_011.pt` | fold-local activity-only | `55d7143b…` | rejected |
| All-source `epoch_019.ckpt` | `SPINT-main/logs/train/runs/2026-07-21-19-11-01/...` | `c81a2bbd…` | forbidden as a clean LOSO parent |

Why S-Fix is now unique rather than “a candidate”:

1. It is the only inventoried carrier-aware last-epoch store with independent
   post-`fc_in` `P` of shape `[1024, 4]`.
2. Its Lightning hparams point at the bound fold-0 teacher bytes, not the
   all-source teacher.
3. `source_session_names` / teacher manifest train files are
   `20120926/27/28`. `outer_left_out = ses-20120924`.
4. S-Fix query windows are the legal `[10, 210)` pair. Z-Fix cannot host a
   carrier path. The all-source teacher cannot support a clean LOSO claim.

Claim label: **`CLEAN_OUTER_SESSION_FILE_EXCLUSION`**.

Caveats that remain disclosed, not repaired:

- Teacher hparams use `query_start_trial=0` on **source** sessions. That is
  source-calibration exposure, not 20120924 file leakage. P training still
  uses post-M10 windows.
- S-Fix stored only `epoch_011`. Epochs 000–010 are missing. It is an
  initialization, not a 12-epoch selection archive.
- The Lightning file has `optimizer_states` but no sampler / normalizer /
  basis / RNG fields. It is **not** a P full-ckpt resume. P writes a new
  12-epoch store.
- S-Fix used Adam `wd=0`. The workorder freeze is AdamW `lr=1e-4`, `wd=1e-2`.
  Optimizer state is **not** reused.

Trainable allowlist if this revision is later frozen:

- P-FIX and P-CA: identical consumer (decoder + B3 identity encoder + `P`).
- P-CA only: `RowNormalizedNNMFBasis.raw_dictionary`.
- Preserve the raw 16-D residual head. No hard output projection.

## 2. `f_eta`

Name: `row_normalized_nnmf_nnls_v1`.

```
y_rect(t) = relu(y_S(t))          # existing rectifier
y_scaled(t) = y_rect(t) / s       # source-frozen EMG RMS, floor 1e-8
D = row_l2(relu(D_raw))           # gauge; D ∈ R^{3×16}, nonnegative
z(t) = NNLS(y_scaled(t), D)       # same scipy NNLS as the bank
```

Initialization: copy the existing source-frozen rSyn3 dictionary (already
nonnegative and row-normalized). P-FIX freezes `D_raw`. P-CA trains `D_raw`.

This is the unique layer that satisfies workorder §6.1 without changing rank:
zero basis perturbation must reproduce the existing M10 rSyn3 estimator. A
16→3 linear map, an MLP, or the historical q=8 calibration-aware bottleneck
would break that identity. Those remain other named revisions.

Gauge: row-L2 after ReLU. Coordinate rescaling cannot cheaply evade the ridge
penalty. Source RMS scale is not learned.

Gradient contract: scipy NNLS selects the active set (discrete, detached);
`z_I` is recomputed by `torch.linalg.solve` on that support so `D` receives
gradients. Degenerate Gram is fail-closed, not lambda-repaired.

Lags stay out of this cell. E2 held-trial ΔvwR2 ≈ +0.039 is feasibility only.

## 3. Differentiable ridge

Name: `carrier_solver.solve_ridge`.

Exact bank objective, now in torch float64:

`(X^T X / n + Lambda) beta = X^T rates / n`, `X = [1, z]`,
`Lambda = diag([0, 1, 1, 1])`.

Carrier layout remains `[w1, w2, w3, b]`. H1’s unnormalized `λ=100` ridge is
a different estimator and is not used here. No NumPy cache on the training
path. Intercept stays unpenalized.

## 4. Still not frozen / not authorized

- Carrier normalizer rematerialization after a learned `D` (both arms or
  neither; old source-pooled `[w,b]` stats are not automatically valid).
- Source-selection dates for a 12-epoch store that does not yet exist.
- Disposable GPU smoke / memory profile.
- Correctness suite §6 items 4–9 on a live consumer (unit-permutation,
  dropout sync, target-fit zero backward, full resume).
- GPU P-FIX / P-CA.

## 5. Smallest next step after Astra freeze

If this revision is accepted: rematerialize the source-pooled carrier
normalizer for both arms from the named `D`, write the paired shuffled
manifest, run CPU consumer tests 4–9, then one disposable 100-step profile.
Do not launch the 12-epoch pair before that profile. Do not add E2 lags.
