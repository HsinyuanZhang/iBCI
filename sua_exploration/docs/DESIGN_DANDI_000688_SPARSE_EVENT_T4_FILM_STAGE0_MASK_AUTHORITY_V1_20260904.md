# DANDI 000688 Sparse-Event T4 / FiLM V1 — Stage-0 Mask Authority

Status: `FROZEN_STAGE0_AUTHORITY__SOURCE_ONLY__NO_DECODER_SCORE`

Date: 2026-09-04

This additive authority freezes the only Stage-0 decisions permitted by
`DESIGN_DANDI_000688_SPARSE_EVENT_T4_FILM_V1_20260904.md`.  It does not alter
the parent design, thresholds, windows, label horizon, estimator, profile, or
training law.

## 1. Frozen parents

- design SHA-256:
  `56982085d4cc7c4e06d29d79a3701e8e6f6d93d08955ceff4c09737bef956705`
- work-order SHA-256:
  `18cc4dde507a41b3853cb6b5a6bd6f9f5c3d47b0837e26949433803ce427c6d2`
- source-manifest SHA-256:
  `4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9`

## 2. Canonical Stage-0 witness

Root: `sua_exploration/results/dandi688_sparse_event_t4_v1/stage0`

- attempt body SHA-256:
  `1218e6e7be67d2407f712714c0d7eb1b8e54625b87b83eef20b4d01233dbd533`
- Stage-0 body SHA-256:
  `99ee1c3c64520d503c84ee2be608ae3a2aa9f6279602120de00ded8f56006af8`
- terminal body SHA-256:
  `bc39dd46efed12f4f27353e335a24903a3b98472944567c7139c6650c6dc2caf`
- exact topology: `attempt.json`, `stage0.json`, `terminal.json`, and their
  three `.sha256` sidecars only; all six leaves were regular, mode `0444`, and
  single-link at authority creation.
- process exit status: `0`
- decoder opened: `false`
- GPU opened: `false`
- external/test file opened: `false`

## 3. Additive diagnostic witness

Root:
`sua_exploration/results/dandi688_sparse_event_t4_v1/stage0_supplement_v1`

- attempt body SHA-256:
  `1cf77b38e08b4c5ab4407c5ad1611170b5b2eb1d28df50155f4676900f48fcd4`
- supplement body SHA-256:
  `02e439dd71c320147edd105bfa220645ec64b722f45f754f410826227c755b4b`
- terminal body SHA-256:
  `8167122b71ba430f09b9b8dcdb1488d3c82d3fd66f5b7d7c0fbda9208d031e43`
- exact topology: `attempt.json`, `supplement.json`, `terminal.json`, and their
  three `.sha256` sidecars only; all six leaves were regular, mode `0444`, and
  single-link at authority creation.
- `mask_or_gate_mutated=false`
- primary reliability values and the four-bit mask matched the canonical
  Stage-0 receipt.
- raw spike-rate primitive calls: one concatenated call per source session,
  using eight CPU workers; validation sessions consumed candidate M10 labels
  only, while trials 50–109 audit labels were confined to the 27 training
  sessions.

## 4. Frozen decisions

Column order is exactly `[a_R, c_R, m_R, delta_b]`.

`reliability_mask: [true, true, true, true]`

`estimator_route: OPEN`

`film_route: OPEN`

The canonical source-only evidence was:

| column | split-half Fisher aggregate r | split-half median r | M10/reference Fisher aggregate r | M10/reference median r |
|---|---:|---:|---:|---:|
| `a_R` | 0.8562303409 | 0.8577814651 | 0.8652315855 | 0.8800333351 |
| `c_R` | 0.8429745379 | 0.8620753302 | 0.8649506211 | 0.8496433540 |
| `m_R` | 0.8778093861 | 0.8714471977 | 0.8758405414 | 0.8801302772 |
| `delta_b` | 0.9337433741 | 0.9398033863 | 0.9166602588 | 0.9187001745 |

All four columns had 27/27 finite and positive session correlations in both
gates.  For the estimator gate, POST700 minus whole-trial mean reference
correlation was `+0.1026294746` for `a` and `+0.0844656544` for `c`, with
positive per-session differences in 21/27 and 23/27 training sessions.

## 5. Consequence

GPU Stage 1 may use only the exact mask and route decisions above.  Admission
must revalidate both six-leaf graphs and this document's byte SHA before any
checkpoint load, NWB materialization, or CUDA operation.  No decoder R2,
external session, test split, pseudo-MUA performance, or EvalAI result was
observed in creating this authority.

