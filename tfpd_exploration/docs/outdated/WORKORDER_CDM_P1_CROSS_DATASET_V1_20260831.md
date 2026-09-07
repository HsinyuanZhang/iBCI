# Work Order — CDM + P1 combination, SUA / M2 transfer (V1, revised)

Date: 2026-08-31. Device: companion machine (local GPUs stay with the C2/C3
mainline). Nothing touches running roots or frozen packages.

## 1. Sealed source results (DANDI 000688 sub-C/sub-M)

- **P1 promoted**: support-anchored block-refit carrier; external M4
  +0.0205 over activity-only CDM (0.2459→0.2664); controls all beaten;
  M10/M30 neutral-safe (`results/support_anchored_t4_stage_p_v1/`).
- **C1 mechanism**: prefix-cycle weights, static external M4 +0.0265 / M10
  +0.0350 (`results/cal_aug_v1/`).

## 2. Part A — DANDI full factorial (inference-only)

| Cell | weights | carrier | purpose |
|---|---|---|---|
| F00 | sealed | frozen T4 | P0 baseline (bit-anchor) |
| F10 | C1 | frozen T4 | CDM⊗C1 |
| F01 | sealed | **P1 online** | CDM⊗P1 |
| F11 | C1 | **P1 online** | full stack |

Gates: vs F00 external M4 ≥ +0.01 & ≥10/15; F11 ≥ max(F10,F01) − 0
(additivity); within/M30 ≥ −0.02. P1 hyperparameters verbatim from the
sealed stage-P selection (no re-selection). ~2-4 GPU-h.

## 3. Part B — SUA and M2 transfers

### B1. SUA (same DANDI 000688 loader family, sua signal view)
Closest transfer: same trial boundaries, same direction law, same T4
machinery — port the stage-O/P replay loop onto the SUA surface's frozen
runtime (the sua_exploration matched-scorer assets). Cells F00/F01 first
(P1 needs no training); the C1-weights cells need an SUA prefix-cycle
training pair (separate work order — do not port DANDI hyperparameters as
claims). Gates identical to Part A on the SUA within/external rosters.

### B2. M2 (FALCON M2 track)
Prerequisite audit before any cell: (a) completed-trial boundary
availability under the M2 evaluation contract (if the official eval hides
trial metadata, P1 can only run on the FSU/TTA-legal boundary reconstruction
— freeze and pre-register that law first); (b) T4-equivalent per-unit
cosine fit on M2 (the recent m2_t4_activity_budget lines provide the
assets); (c) direction parametrization for the M2 task. If (a) fails the
causality/contract check, record P1-M2 as not-evaluable under the official
contract and stop that arm honestly.

## 4. Disciplines

Additive packages (`src/cdm_p1_cross_v1/` etc.); attempt-before-data;
0444+sidecar receipts; bit-anchors wherever a same-law sealed row exists;
hyperparameters frozen from sealed selections; no target fitting; nulls
reported as nulls.
