# CPU SCREEN LINE CLOSURE: the content axis, the rank fix, and three generalization screens

**Date:** 2026-08-09
**Status:** all screens complete. Readings reported. No route selected.
**Scope:** CPU only, source data, forward passes only. No GPU, no training, no target data.

---

## 1. What this line was about

The `[a,c]` encoding coefficient carries R² by about 0.29 on SUA. Five levers that
tried to estimate it more accurately all closed negative. The remaining question was
whether a different **kind** of number in the carrier slots could hold content that the
activity path cannot express.

The CPU screen line ran seven rounds to answer that. All are now complete.

---

## 2. Round-by-round results

### Round 1 — C1: cross-date reframe (zero compute)

The five-date cross-date mean delta is **+0.056287**, against the same-date fold-0 mean
of **+0.025623**. The honest ratio is **2.2x**, not the earlier claim of 3-4x. The
cross-date effect is real and the same-date effect is a lower bound. 4/5 dates positive,
date-level paired bootstrap 95% CI `[+0.006, +0.097]`.

**Status:** done, stands.

### Round 2 — C2: overlap gate (one forward pass)

The H-C0 activity path is the neutral reference. The overlap residual R² measures how
much of each carrier the activity path already holds. Pooled residuals against H-C0:

| Carrier | Pooled residual R² |
|---|---|
| H1 production (historical preliminary positive control) | 0.868 |
| N4 (negative control) | 0.249 |
| L-A (encoding form) | 0.292 |
| L-C (noise-normalized) | 0.467 |

N4 failed as a whole, so 0.249 is a practical floor. The overlap test **falsifies but
does not confirm**: a residual above zero does not mean useful.

**Status:** historical preliminary screen. Its H1 production number and routing interpretation are
superseded by the exact same-pipeline calibration in round 2b (`H-C=0.804819`).

### Round 2b — decoder calibration of the overlap ruler（2026-08-10 root audit）

The previously missing calibration was subsequently implemented with the exact H1 production
carrier machinery and the same frozen H-C0 activity reference. The within-pipeline primary set is
`{H-C,H-RS,H-LS}`; all decoder gains are relative to separately trained H-C0:

| arm | pooled overlap residual | decoder gain over H-C0 |
|---|---:|---:|
| H-C | `0.804819` | `+0.038895` |
| H-RS | `0.813399` | `-0.027224` |
| H-LS | `0.825789` | `+0.013280` |

The ordering is not monotone. Exact Spearman is `rho=-0.5`, two-sided permutation `p=1.0`
(`n=3`). However, the smallest attainable two-sided exact p at `n=3` is `2/6=0.333`; the three
primary residuals span only `0.020970`, and two points are corruption controls. Adding N4 as an
explicitly heterogeneous cross-pipeline sensitivity point gives `rho=0.0`, `p=1.0` (`n=4`). The
current overlap residual is therefore **not calibrated as a decoder-gain predictor because the
available same-pipeline calibration has no power**. This is not a measured negative relation. It
produces no prediction for L-A, L-C, or the population carrier and remains a one-sided falsifier:
a near-zero residual can reject novelty, while a high residual alone cannot authorize an arm.

Root reproduced the receipt byte-for-byte, verified all four checkpoint file hashes, corrected
three synthetic-test specification errors, and obtained `21 passed`. Receipt SHA-256 is
`624557b1c199f5cfdb6372f4716308c45a9ff9e87af6ccc56ba88c4da5036cb7`, mode `0444`.

**Status:** `INFEASIBLE_TO_CALIBRATE_AT_CURRENT_PROVENANCE`. Do not use this ruler to authorize
L-B or a GPU arm. It may be reopened only if exact same-pipeline SUA instantiation provenance is
recovered and a wider candidate set is frozen before reading gains.

### Round 3 — C-FIX (F1-F4): four corrections

- **F1:** The frozen AFC4 compression rule places `b` in its own slot, never projected
  through U. The old U[8,4] already did this. The F1 fix was a no-op. The b-dominance
  diagnosis was correct: b variance = 240.25, mean W variance = 0.0007, ratio = 326,463.
- **F2:** Separability and drift must be scale-invariant. After unit-RMS normalization,
  L-A ratio = 106.75, L-C ratio = 103.94 — a tie.
- **F3:** H-C is co-adapted; H-C0 (carrier zeroed) is the neutral reference. Bias is
  about 0.006.
- **F4:** All encoding carriers are effectively rank 1. Singular values [26.5, 0.07,
  0.05, 0.03]. First-component fraction 0.9948.

**Status:** CPU ratio screen complete, but the ratio has no validated mapping to decoder R². This
screen neither authorizes nor refutes a whole-pipeline noise-normalized carrier.

### Round 4 — C-FIX2: per-column normalizer

The rank collapse is real and the per-column scale normalizer fixed it:

| Carrier | Before (first-comp fraction) | After |
|---|---|---|
| L-A | 0.9948 | 0.3380 |
| L-C | 0.9942 | 0.3981 |

Four verification checks all passed. H-C0 = [0,0,0,0] stays exactly zero.

The H1 production pipeline uses a **single global RMS scalar**, which cannot balance the
columns. An L-A arm needs the per-column normalizer wired into the production path. This
is a scheduling constraint, not work to do now.

**Status:** done, stands.

### Round 5 — C-FINAL: per-column residuals and decision

The pooled statistic is a mean over per-column R² (not a stacked variance ratio). R² is
scale-invariant, so the per-column normalizer never changed it. Per-column residuals:

| Form | Col 0 | Col 1 | Col 2 | Col 3 |
|---|---|---|---|---|
| L-A | W0: 0.281 | W1: 0.364 | W2: 0.513 | b: **0.001** |
| N4 | mean_rate: **0.001** | Fano: 0.368 | lag1: 0.225 | pop: 0.386 |

The `b` column has near-zero residual (activity path holds it). The three W projections
hold 0.28-0.51. The floor for a W column is not the pooled N4 floor — it is the N4
non-rate columns at 0.225-0.386.

M4 paired sign test: 9/11 positive, p = 0.065 (tie). M3 shows raw W columns (median 0.449)
above U-projected columns (median 0.364), advantage +0.085.

**Outcome: OUTCOME 2.** The encoding form holds content, but U[7,3] discards part of it.
L-A closes in its current form. Lever L-B, the choice of U, is indicated.

**Status:** done. The content axis is not empty.

---

## 3. Generalization screens (P2, P1, P3)

### P2 — kinematic design rank

The M=4 support reaches 90% of kinematic variance in **4 components** (median across 11
recordings). The full recording needs **4**. At 99%: support needs 5, full needs 6.

**Reading:** the support spans the kinematic space. The L-B reading stands. The limit is
not trial diversity.

### P1 — same-checkpoint identity-reliance damage, not attainable headroom

| Task | Same-checkpoint identity intervention | Carrier/system result | Scope warning |
|---|---:|---:|---|
| SUA | large damage values cited (`-0.277/-0.515/-2.298`) | `+0.2528` | heterogeneous historical checkpoints |
| H1 | `+0.903` full-minus-zero damage | `+0.056287` | different endpoint and consumer from other tasks |
| M2 | `+0.165` full-minus-zero damage | official system `+0.116765` | official result also changes selected decoder epoch |
| M1 | later sealed receipt gives `+1.848512` | matched carrier content `-0.006525` | one-fold OOD zero intervention |

These values prove that the trained identity tensors can be load-bearing. They do not measure a
common attainable-headroom quantity: the interventions, checkpoints, endpoints, and decoder
families differ across tasks, and zero identity is out of distribution. Therefore no cross-task
identity-limitedness law or monotone carrier-gain relation is identified from this table.

### P3 — population-structure carrier

A cross-channel PCA carrier fitted on the calibration block. Provenance: `pcs` is
source-frozen (`h1_lag_screen.py:125-127`), never refit per session.

| Carrier | Pooled residual | Sep/Drift ratio | First-comp |
|---|---|---|---|
| H1 production | 0.862 | 81.08 | 0.418 |
| L-A per-column | 0.329 | 106.75 | 0.338 |
| Population structure | **0.190** | 52.76 | 0.257 |
| N4 | 0.273 | 6305 | 0.953 |

The population carrier reaches the **lowest** residual of all forms, but below the N4
floor. Its drift (0.0715) is the largest after canonicalization. The subspace does not
align across sessions.

**Reading:** the lever closes. A target-fitted decomposition holds content but is useless
to a source-trained consumer because its coordinates mean something different each session.

---

## 4. What is settled

| Item | Status |
|---|---|
| C1 cross-date reframe | done, stands |
| Rank collapse | real, per-column normalizer fixed it |
| H-C0 as neutral reference | done, bias quantified at ~0.006 |
| Lever L-C (noise normalization) | CPU proxy complete but uncalibrated; whole-pipeline arm untested |
| Lever L-A (encoding form) | **closed in current form** (U discards content) |
| Lever L-B (choice of U) | **indicated** by C-FINAL outcome 2 |
| P2 kinematic rank | support spans 4 of 7 DOF, L-B stands |
| P1 identity reliance | H1 and M2 diagnostics retained; M1 later became testable via sealed receipt `f419514d...91bf3`, which gives full-minus-zero `+1.848512` on one fold. This proves strong reliance on the trained identity tensor, **not attainable performance headroom**. |
| P3 population structure | **closed** (drift too large, residual below N4 floor) |
| Per-column normalizer in production | scheduling constraint, not work to do now |
| Carrier gain vs identity reliance | heterogeneous damage diagnostics; no valid common-scale law |

## 5. What is open

- **Lever L-B**, the choice of U: C-FINAL outcome 2 indicates it. P2 confirms the support
  spans enough kinematic dimensions. A review decides whether it starts.
- **Lever L-D**, multiplicative consumption: the only untouched structural lever. A review
  decides.
- **M1 boundary condition**: the later sealed checkpoint ablation exists and refutes “identity is
  unused,” but its zero intervention is not a headroom oracle. The realized EMG-AFC4
  content-versus-compact-zero contrast remains negative; no new carrier run is authorized here.
- **External scope**: subject-M same-Dandiset cross-animal evidence is complete; the older sub-C
  one-shot formal slot was consumed without a result and was not rerun. Neither fact replaces the
  other; see `CURRENT_RESULTS.md`.

---

## 6. Receipts

All receipts are immutable (mode 0444 or write-once). SHA-256 verified.

| Receipt | SHA-256 |
|---|---|
| `h1_cross_date_reframe_v1.json` | `a9ab7923...` |
| `h1_overlap_gate_v1.json` | `cbd06ab9...` |
| `h1_lc_screen_v1.json` | `b6fc7f7a...` |
| `h1_content_lever_cfix_v1.json` | `1861ee8b...` |
| `h1_content_lever_cfix2_v1.json` | `e08cd05d...` |
| `h1_content_lever_cfinal_v1.json` | `e3233519...` |
| `h1_generalization_screens_v1.json` | `2288e060...` |

---

## 7. Authorizations and limits

Every screen in this line was CPU-only, on source data. Forward passes used
`map_location="cpu"`, no gradient, no optimizer. Model state hashes were recorded before
and after every forward pass and were identical in every case. No target, minival, formal,
or EvalAI file was opened. No git commit was made. No protected file was edited.
