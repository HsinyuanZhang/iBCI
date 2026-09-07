# Work Order — CDM ⊗ CAL-AUG V1: activity-only CDM online memory on the C1 prefix-cycle checkpoint

Date: 2026-08-30
Kind: **inference-only deployment-system combination** (no training; not a
decoder-training variant — outside the §12 stop node's scope).
Authority: operator request 2026-08-30 ("CDM 和 CAL-AUG 不冲突，加入共同实验设计").
Device: this document does not launch the cell. After independent review, use
one root-selected idle RTX 3090 without sharing it with an active training job.

## 0. Evidence update after C2/C3 completion

The C2/C3 GPU score is now terminal. Direct posterior-input C2 and the free
reliability-feature C3 are not promotion candidates. C2 lost to C1 on every
budget and surface; C3's correct q binding did not beat its same-checkpoint
q-shuffle control. These results do not change this work order's primary cell:
`cdm_c1` uses **C1 weights plus activity-only CDM and a frozen ordinary
carrier**.

The paper-level interpretation and any future posterior successor are defined
in `DESIGN_CAUSAL_CALIBRATION_MEMORY_SECOND_INNOVATION_20260830.md`.
Posterior information may only return as a support-anchored residual controller
after the C1 x activity-memory factorial is complete. It must not be inserted
into this V1 cell, and this V1 cell must not be called `CDM+C2`.

## 1. Rationale

The two winning mechanisms are orthogonal:

* **C1** (CAL-AUG) changes WEIGHTS: B3S identity robustness to short
  calibration prefixes (source prefix degradation −0.243 → +0.004; static
  deployment external M4 +0.0265 / M10 +0.0350).
* **CDM activity-only** changes the INFERENCE DATA FLOW: the B3S activity FIFO
  grows with completed trials, re-estimating identity per trial.

CDM's early trials are exactly the short-prefix regime C1 was trained for, so
the gains should stack; C1's only cost (external M30 −0.0220) sits in the
region where the FIFO has already filled to 30 and should dilute.

## 2. Cells

| Arm | Decoder weights | Deployment flow |
|---|---|---|
| `cdm_t0` | T0 SWA (`results/cal_aug_v1/t0_operator_disabled/swa_final4.pt`) | activity-only CDM rollout (A0 law: trial-completed activity FIFO, zero carrier transitions) |
| `cdm_c1` | C1 SWA (`results/cal_aug_v1/c1_prefix_cycle/swa_final4.pt`) | identical rollout law, identical trial order |

Surfaces: within-6 + external-15; budgets M4/M10/M30; governing last-bin
variance-weighted R², equal-session means, paired per-session deltas. Raw
predictions scored (no output filter in the primary rows; the α=0.7 causal EMA
appears ONLY as a predeclared secondary post-processing row, per the frozen
filter-line disposition).

**T0-side shortcut rule (pre-registered):** if `arm_common.state_sha256` of the
T0 SWA equals the sealed Cell-D SWA state digest (`626f65d8…` artifact), then
`cdm_t0` is by definition the sealed activity-only CDM system and the sealed
rows (V8 score / P2' stage-A A0 / the filter-line cdm cache) are its anchor —
only `cdm_c1` runs. If the digests differ, both sides run fresh and the
sealed-number reuse is forbidden.

## 3. Machinery (frozen, imported, never reimplemented)

Reuse the P2' runtime exactly as the sealed drivers use it
(`learnable_output_filter_v1.streams.materialize_cdm` is the proven template:
it reproduced the sealed stage-A A0 rollouts bit-exactly, carrier transitions
asserted zero, model-state digest asserted unchanged). The ONLY addition is a
checkpoint swap of the runtime's model (strict-load the arm SWA into the same
Cell-D graph, mirroring `cal_aug_v1.deployment._swap_runtime_model`) plus the
state-digest equality proofs.

Anchors (hard, fail-closed):
1. `cdm_t0` (if run) reproduces the sealed A0 rows bit-exactly per session
   (matrix R² + raw prediction SHA);
2. every rollout commits ZERO carrier transitions (activity-only law);
3. arm model state digest unchanged before/after;
4. the per-trial B3S stream digests of the two arms differ ONLY through the
   decoder weights (same inputs; assert the input-side digests match).

## 4. Gates (pre-registered)

```text
STACKING_GATE (primary):
  external M4 or M10: cdm_c1 − cdm_t0 mean Δ ≥ +0.01 AND ≥ 10/15 positive
  the other low budget: Δ ≥ 0
  external M30: Δ ≥ −0.02
  within every budget: Δ ≥ −0.02
CHAMPION_READING (reported, not a gate):
  cdm_c1 at M4/M10 vs the sealed activity-only CDM numbers
  (0.2265 / 0.3888 external) — does the combination set a program record?
```

Failures are reported as `STACKING_NULL__NO_CDM_GAIN_FROM_C1_WEIGHTS` (with
the M30-dilution hypothesis explicitly tested by the M30 cell) or
`STACKING_PARTIAL` (passes low budgets, fails an M30/within safety clause).

## 5. Process

Additive package `src/cdm_on_cal_aug_v1/` + driver + tests (synthetic: swap
purity, anchor-comparison logic, gate boundaries with the 1e-12 program
epsilon). Fresh root `results/cdm_on_cal_aug_v1/`; attempt receipt before any
data/model access; sealed-file SHA pinning (P2' runtime, both arm SWAs + their
terminal receipts); atomic terminal-or-failure. Runtime bound: ≤ 90 min GPU
for one arm × 6 rollouts (sealed precedent ~47 s/session-rollout), ≤ 3 h if
both arms run.

## 6. Out of scope

Any carrier-gated CDM variant; any retraining; filter α sweeps beyond the
single predeclared secondary row; target-session fitting; M1/H1 claims (the
mechanism hypothesis may be cited as motivation only, per guidance §10).
