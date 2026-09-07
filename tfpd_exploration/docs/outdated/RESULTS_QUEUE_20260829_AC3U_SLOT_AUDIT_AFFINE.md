# Results Record — 2026-08-29 Queue (AC3-U, SLOT-AUDIT, Affine, CAL-AUG launch)

Authority: `docs/EXECUTION_GUIDANCE_AFTER_AC3_AND_CONTINUITY_AUDIT_20260829.md` §11 queue.
Operator goal directive 2026-08-29. All roots sealed 0444 + sidecars.

## 1. AC3-U — `results/ac3_utility_bridge_v1/` (queue item 1, CLOSED)

Work order: `WORKORDER_AC3_UTILITY_BRIDGE_V1_20260829.md`. M4-only, within-6,
frozen P2' H=5 coherent replay, rows U0/UGE/U2, governing = raw (no filter),
anchor = sealed stage-cop within-M4 O0.

| Row | Direction input | Equal-session raw R² | vs U0 |
|---|---|---:|---:|
| U0 | raw single-view (frozen, bit-anchored to sealed O0) | 0.53556 | — |
| UGE | R-GE four-group circular mean | 0.53291 | **−0.00265** (2/6) |
| U2 | R2 supervised head (grouped-OOF) | 0.54276 | +0.0072 (4/6); vs UGE +0.00985 |

**Disposition: `AC3_U_UTILITY_NULL__CLOSE_AC3_ON_FROZEN_M4_SURFACE`** (§4.4).
UGE fails on the mean AND breadth; U2 rescue fails both margins (needs ≥+0.01
over UGE and over U0). The pre-registered expectation (work order §9: the gate
sits above the within true-direction ceiling O2−O0 = +0.0046) was confirmed;
the ensemble direction is slightly WORSE than the raw single view in coherent
utility, and the supervised head's +0.007 is below every gate margin.

Integrity: U0 anchor bit-exact on all 6 sessions (matrix R², raw/filtered
prediction SHAs, oracle decision records, accept/decision/commit counts,
initial digests) on the SAME physical GPU as the sealed run. Amendment 1
documents the one non-reproducible field (`final_carrier_sha256`: same-card
cross-process float jitter in a late trial's canonical snap; bit-equal
governing predictions make any earlier divergence impossible). Model state
digest unchanged; external roster never opened; 891 s wall.

## 2. SLOT-AUDIT — `results/slot_audit_v1/` (queue item 2, COMPLETE)

Work order: `WORKORDER_SLOT_AUDIT_V1_20260829.md`. 63 CPU decodes
(21 sessions × M4/M10/M30) through the frozen static recipe, every last-bin R²
and prediction SHA bit-anchored to the sealed continuity probe; trajalign
re-measurement reproduced all 24 sealed arms at **max-abs delta 0.0**.
Bin = 20 ms (frozen loader constant; W=50 → 1 s windows; trajalign K=16 delay
= 15 bins = 300 ms; the phrase "zero-lag" stays banned).

Central-question answer (§5.2): **mechanism = covariance reduction from
averaging near-tail redundant estimates + session-dependence; there is NO
systematically better-calibrated earlier slot.**

- external: slot 49 is the best slot at EVERY budget (M4/M30 exactly; M10 best
  s43 within +0.002). Early slots are catastrophically worse (s3/s4 ≈ −0.09 to
  −0.13 mean R²).
- within (source subject): earlier slots beat s49 slightly (M10 s40 +0.026,
  M30 s24 +0.028) but the per-session argmax slot drifts → classification
  `session_dependent` → no deployable fixed slot subset; target selection
  forbidden.
- equal-slot subset averages: last-5/last-10 give small positive gains
  (external M30 last-5 +0.037, M10 +0.018) — consistent with the filter line's
  α=0.7 EMA result; all-50 averaging is catastrophic (−0.22 to −0.43).
- Slot-subset source selection (by within mean, applied to external) selected
  s49/s43/s24 per budget — i.e., the audit endorses the status-quo last-bin at
  M4/M30 and nothing deployable elsewhere.

## 3. Affine diagnostics — `results/affine_diagnostics_v1_r1/` (queue item 5 on the current system)

New package `src/affine_diagnostics_v1/` (8 tests). Per-cell six-vector fits
(closed-form LS), dispersion stats, LOO within-cell r0, and the deployable
CROSS-SURFACE r0 (within-6 mean correction applied to each external session;
zero held-surface labels in the fit).

- **Deployable cross-surface r0 is NEGATIVE everywhere**: CDM external
  M4/M10/M30 = −0.0041 / −0.0084 / −0.0067 (9/15, 5/15, 6/15); static
  −0.0175 / −0.0123 / −0.0067. Session-affine miscalibration is
  SUBJECT-SPECIFIC: the source subject's mean correction does not transfer.
- Full own-affine oracle opportunity confirmed (+0.037…+0.103, shrinking with
  budget) — all rows leakage-labelled, never selection/deployment eligible.
- Within-cell LOO r0: small positive at M30 (+0.019, 6/6) and M10 (+0.008) on
  CDM, negative at M4 (−0.010, 4/6; reproduces the audit's exploratory
  −0.0074 4/6 reference shape).
- §8.3 disposition: the CPU identifiability gate for affine-nuisance training
  FAILS (dispersion large, cross-surface sign flip, r0 negative) — no work
  order will be written for that cell.

## 4. CAL-AUG — smoke PASSED; T0 full arm RUNNING (queue items 3–4)

Work order: `WORKORDER_CAL_AUG_V1_20260829.md`. Successor trainer
`scripts/run_cal_aug_cell_v1.py` (importlib-loads the sealed runner stack after
SHA pinning; training-gated forward-pre-hook `calib_trials[:, :M]`,
`M = (30,10,4)[step % 3]`, zero RNG consumption).

Smoke (`results/cal_aug_v1/smoke/`, 40 steps × both arms on GPU 0):
**SMOKE_PASSED, 0 failed checks** — initial-state SHA equality, optimizer/
schedule equality, first-40 batch-order + session-order digest equality,
first-40 dropout-p digest equality across arms, C1 realized prefix sequence
[30,10,4,…], visible-slice digests, full-block digests unchanged, per-step
encoder `push_trial` counts == declared M (probe), T4 authority fingerprint
unchanged, finiteness, no target path resolved. Four infrastructure failures
during smoke bring-up are preserved as `smoke_failed_*` roots (probe kwarg,
probe counting law, LazyModule numel, import scope).

Throughput probe (in-run, outside the digest-recording window after the
amendment note in `plan.py`): **104.2 steps/s**, pair projection
**8.68 GPU-hours** < 12 h ceiling (sealed anchor 87.99 steps/s; the first
probe reading 16.1 steps/s was polluted by the recording window and was fixed
before the full launch).

T0 launched 19:09 UTC+8 on GPU 0 (`results/cal_aug_v1/t0_operator_disabled/`),
48 epochs, 8 h hard in-process timeout.

**Operator amendment (2026-08-29 19:30, user instruction)**: the work order's
serial-same-GPU clause is superseded — T0 (GPU 0) and C1 (GPU 1) run in
PARALLEL. Both cards are the same model (RTX 3090 24 GB, identical
architecture/SM count), so the guidance §6.4 "second UNLIKE GPU" hardware
confound does not apply; the smoke's cross-arm digests (batch order,
dropout-p stream, prefix sequence) are CPU-RNG quantities and hold on either
card. Measured probes under parallel load: T0 87.9 steps/s, C1 91.4 steps/s
(each within the 12 GPU-hour pair ceiling; the projection is per-arm
2×1,628,400-steps at the measured rate). Bring-up failures are preserved as
`smoke_failed_*`, `t0_interrupted_smoke_cmp_bug`, `c1_interrupted_smoke_cmp_bug`,
`probe_attempt1_killed`, `probe_attempt2_smoke_cmp_bug`.

Both arms restarted 19:41:40 with the smoke-binding comparison fixed (the
values matched bit-exactly; the comparison read the per-arm digest at the
wrong nesting level). Both arms' epoch-000 `smoke_digest_match.all_match` is
now `true`. Mechanism readout (§6.6) and deployment scoring (§6.7) run after
both terminal receipts.

**Rehearsals (scratch, sealed SWA as both arms)**: mechanism module dry-run
(3 sessions) — identical-arm invariance exact (recovery M4/M10 = 0.0, M30
delta = 0.0, gate correctly fails), crossed readout cost 23.5 s / 3 sessions
→ full 27 ≈ 3.5 min; sealed-model prefix degradation M10 −0.066 / M4 −0.194
source R². One dead reference fixed (`schedule.sha256_file_static` →
`receipts.sha256_file`). Deployment scorer rehearsal launched 19:58.

## 4b. CAL-AUG final results (2026-08-30 morning, both arms CELL_TERMINAL)

Both arms re-ran cleanly after the first pair was closed out by the closure
guard (a mid-run edit of `src/cal_aug_v1/mechanism.py` — the scoring-path fix
at 19:55 — drifted the bound closure; the guard fired CORRECTLY, the failed
roots are preserved as `t0/c1_closure_drift_mechanism_edit`). Final arms:
48 epochs, invariant failures empty, launch==final closure, SWA sealed with
strict reload + finite forward smoke. Wall ≈ 5.0 h/arm in parallel.

**Mechanism readout (§6.6, source-27, crossed (B3S=M, T4=train-side M30))**:

| Arm | M30 | M10 | M4 |
|---|---:|---:|---:|
| T0 source R² | 0.7468 | 0.6874 | 0.5041 |
| C1 source R² | 0.7326 | 0.7367 | 0.7367 |
| prefix degradation T0 | — | −0.0594 | −0.2427 |
| prefix degradation C1 | — | **+0.0041** | **+0.0041** |

C1 is essentially **prefix-invariant** (degradation ≤ +0.004 at both short
prefixes; 27/27 sessions positive recovery; recovery M4 = +0.247, M10 =
+0.064). The registration gate FAILED only on its M30 safety clause:
C1−T0 source M30 = **−0.0142 < −0.01** (by 0.004).

**Deployment scoring (§6.7, frozen recipe, within-6 + external-15)**:

| Cell | T0 | C1 | Δ |
|---|---:|---:|---:|
| external M4 | 0.1197 | 0.1462 | **+0.0265** (10/15; LB −0.0057) |
| external M10 | 0.2955 | 0.3304 | **+0.0350** (10/15; LB +0.0058) |
| external M30 | 0.4286 | 0.4066 | **−0.0220** |
| within M4 | 0.3089 | 0.3708 | **+0.0618** |
| within M10 | 0.4677 | 0.4862 | +0.0185 |
| within M30 | 0.5665 | 0.5846 | +0.0181 |

T0 reproduces the sealed static baselines exactly (matched control is
faithful). The lower continuation gate failed ONLY on the external-M30 safety
clause (−0.0220 vs −0.02, by 0.0020); external M10 passed its full clause
(+0.035, 10/15, LB ≥ 0); within is positive at every budget including M30.
**Disposition: `EXTERNAL_CONTINUATION_GATE_FAILED`** (budgets are never
averaged to hide an M30 regression). Per §7/§11: the T4 cell is NOT
authorized and C1 expansion stops; the queue reaches the §12 stop-and-write
node (AC3-U null + C1 below the continuation gate).

Bounded finding to preserve: the single-axis B3S prefix-length cycle
(30/10/4) buys +22% relative R² at external M4 and +20% at within M4, is
prefix-invariant on source, and its entire cost is a −0.022 external-M30
regression (within M30 is POSITIVE +0.018). Any mitigation (milder cycle,
M30-weighted sampling) is a NEW pre-registered cell requiring separate
authorization — not launched here.

**Affine diagnostics on the frozen T0/C1 predictions (§8 item 5,
`results/cal_aug_v1/affine/`)**: T0 cells reproduce the r1 static anchors
exactly. Deployable cross-surface r0: T0 external M4/M10/M30 =
−0.0175/−0.0123/−0.0067 vs **C1 = −0.0036/+0.0011/−0.0010 (neutral)** —
the prefix cycle also shrinks cross-subject affine miscalibration at low
budgets. C1's external-M4 own-affine opportunity is smaller than T0's
(+0.094 vs +0.103): less affine headroom remains. The affine-nuisance
training cell remains unauthorized (cross-surface dispersion still
subject-specific in T0; the C1-side improvement is descriptive only).

## 6. Basis-head reconstruction audit — CLOSED (2026-08-30, CPU-side addendum)

`src/basis_head_audit_v1.py` + `results/basis_head_audit_v1/` (0444). Fixed
orthonormal DCT-II basis over the 50 output positions, coefficients fitted on
the model's own predictions only (structural no-target-leakage), evaluated on
the frozen slot-audit cache (zero decodes). Mean reconstructed-last-bin-R²
delta vs raw:

| cell | K=4 | K=8 | K=16 | K=32 |
|---|---:|---:|---:|---:|
| within M4/M10/M30 | −0.059/−0.088/−0.111 | −0.006/−0.002/+0.007 | ≈0±0.004 | ≈0 |
| external M4/M10/M30 | −0.010/+0.001/+0.014 | +0.002/+0.009/+0.018 | ≈0 | ≈0 |

Closure per §9: no SMALL K is nearly lossless (K=4/8 lose up to −0.11 on the
source surface), and the lossless K (16-32) show no source-held calibration
dispersion improvement. The small external positives at K≤8 replicate the
already-known smoothing/shrinkage mechanism (cf. the α-EMA filter line), not a
parameter-sharing benefit. **The smooth-basis output-head route is closed; no
GPU cell will be written for it.**

## 5. State of the §11 queue

1. AC3-U: **done, NULL, AC3 closed on the frozen M4 surface**.
2. SLOT-AUDIT: **done** (mechanism classified, trajalign bit-reproduced).
3. CAL-AUG smoke: **passed** (14/14 equality items).
4. CAL-AUG full pair: **done, both arms CELL_TERMINAL**; mechanism readout
   (near-perfect prefix invariance, gate failed on the −0.01 source-M30
   clause by 0.004) and deployment scoring (large low-budget wins, gate
   failed on the −0.02 external-M30 clause by 0.002).
5. Affine diagnostics: **done on the current system AND on T0/C1** (T0
   cross-surface r0 negative; C1 neutral — bonus mechanism finding);
   affine-nuisance cell not authorized.
6. Conditional T4 cell: **NOT authorized** (C1 below the continuation gate).
7. Conditional affine/basis-head cells: affine FAILS its CPU gate; basis-head
   reconstruction audit not required for this queue's closure.
8. Stop-and-write node: **REACHED** (AC3-U null + C1 external continuation
   gate failed). No further DANDI Cell-D decoder-training variants are
   launched from this queue; the bounded mechanism finding above is the
   terminal scientific result of the CAL-AUG line.
