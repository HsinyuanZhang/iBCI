# DELIVERABLE: Priority A2 weighting control and same-target density dose-response (v2 corrected)

**To:** root
**From:** execution agent
**Date:** 2026-08-11
**Status:** A2a-v2 and A2b-v2 both complete. All previous A2a/A2b numbers invalidated.

> **Invalid receipts (diagnostic only, do not cite):**
> - `priority_a2_weighting_receipt.json` SHA `6364b99f...` — uses un-normalized ridge penalty (sum-objective Gram). The +0.67/+0.82 dense−sparse conclusions from this receipt are invalid.
> - `priority_a2_same_target_density_receipt.json` SHA `0fa3f0e3...` — same solver bug. All dose-response numbers from this receipt are invalid.

---

## 1. Valid receipts

| Receipt | SHA-256 | Status |
|---|---|---|
| `priority_a2_weighting_control_v2_receipt.json` | `b6a080c48d36adc74050f0a6672623585320e32bada45001a962622379bcba58` | Valid (A2a-v2, corrected normalized ridge) |
| `priority_a2_same_target_density_v2_receipt.json` | `0d4cd01b5fea86e2dca4640731f99faa2abde57d3d163bf7dee6ac34ed167361` | Valid (A2b-v2, corrected normalized ridge) |

Both use `sua_exploration/mc_maze/priority_a2_normalized_ridge_v2.py` — a standalone normalized weighted ridge where the Gram is divided by total weight, giving a penalty that is invariant to the absolute scale of sample weights.

---

## 2. A2a-v2 results — dense and direction targets are not interchangeable

### SUA mean R² (15 sessions)

| Budget | Dense-uniform | Dense-equal-trial | Dir-uniform | Dir-equal-trial | D−S (uniform) | D−S (equal) |
|---|---|---|---|---|---|---|
| M=15 | 0.073 | 0.149 | −0.407 | −0.408 | +0.479 | +0.557 |
| M=30 | 0.322 | 0.348 | −0.180 | −0.171 | +0.502 | +0.519 |
| M=50 | 0.418 | 0.429 | −0.109 | −0.081 | +0.527 | +0.510 |

### Pseudo-MUA mean R² (15 sessions)

| Budget | Dense-uniform | Dense-equal-trial | Dir-uniform | Dir-equal-trial | D−S (uniform) | D−S (equal) |
|---|---|---|---|---|---|---|
| M=15 | 0.129 | 0.188 | −0.358 | −0.373 | +0.487 | +0.562 |
| M=30 | 0.341 | 0.358 | −0.144 | −0.123 | +0.486 | +0.481 |
| M=50 | 0.410 | 0.414 | −0.072 | −0.043 | +0.482 | +0.458 |

### Key contrasts

- **The supervision formulation matters**: the dense−direction gap is approximately +0.46 to +0.56 across all budgets, views, and weighting rules.
- **The dense-arm weighting effect is smaller**: equal-trial versus uniform weighting changes mean dense R² by +0.004 to +0.076.
- **The gap is not explained by row weighting alone.** However, dense velocity and trial direction differ in both granularity and target content/semantics, so this comparison is not a pure label-density ablation.

---

## 3. A2b-v2 results — same-target label density affects the ridge readout

The tables below first average each session over mask seeds 42, 43, and 44, then average over the 15 sessions. The `K=all` arm is deterministic and is reused across seeds.

### SUA mean R² (15 sessions, three-seed average)

| Budget | K=1 | K=2 | K=4 | K=8 | K=16 | K=all | all−K1 |
|---|---|---|---|---|---|---|---|
| M=15 | −0.011 | 0.022 | 0.096 | −0.580 | −0.271 | 0.149 | +0.160 |
| M=30 | 0.059 | 0.105 | 0.174 | −0.284 | −0.063 | 0.348 | +0.289 |
| M=50 | 0.100 | 0.159 | 0.228 | −0.148 | 0.081 | 0.429 | +0.328 |

### Pseudo-MUA mean R² (15 sessions, three-seed average)

| Budget | K=1 | K=2 | K=4 | K=8 | K=16 | K=all | all−K1 |
|---|---|---|---|---|---|---|---|
| M=15 | −0.010 | 0.020 | 0.095 | −0.818 | −0.414 | 0.188 | +0.199 |
| M=30 | 0.061 | 0.105 | 0.170 | −0.397 | −0.108 | 0.358 | +0.297 |
| M=50 | 0.102 | 0.156 | 0.217 | −0.216 | 0.067 | 0.414 | +0.312 |

### Mask sensitivity and the exceptional session

The aggregate means at `K=8/16` are dominated by one recording, asset `fee6b912-477a-4fea-ad16-e89e4bd42d25` (`sub-M_ses-CO-20150512`). For every view, budget, and `K=8/16` cell, the other 14 session means are positive; removing this one session makes all twelve `K=8/16` aggregate means positive. The finite-`K` masks are also much more seed-sensitive at `K=8/16`: seeds 42 and 44 give negative aggregates, while seed 43 gives positive aggregates.

This is a real heavy-tailed, session- and mask-sensitive result, but the receipt contains no condition numbers, solve residuals, or precision/primal-dual diagnostics. It therefore does **not** establish floating-point instability or matrix ill-conditioning.

### all−K1 paired statistics

| Budget/View | Mean | Median | Sign | Bootstrap 95% CI | Exact sign-test p |
|---|---|---|---|---|---|
| M=50 SUA | +0.328 | +0.391 | 14+/1− | [+0.239, +0.401] | 0.001 |
| M=30 SUA | +0.289 | +0.360 | 14+/1− | [+0.132, +0.396] | 0.001 |
| M=15 SUA | +0.160 | +0.389 | 13+/2− | [−0.229, +0.396] | 0.007 |
| M=50 MUA | +0.312 | +0.372 | 14+/1− | [+0.238, +0.375] | 0.001 |
| M=30 MUA | +0.297 | +0.349 | 14+/1− | [+0.187, +0.375] | 0.001 |
| M=15 MUA | +0.199 | +0.353 | 13+/2− | [−0.092, +0.380] | 0.007 |

### Key findings

1. **The three-seed aggregate improves from K=1 to K=4** at every budget and view (for example, SUA M=50: 0.100 → 0.159 → 0.228). This does not hold for every individual mask seed at M=15, so it is an aggregate result.

2. **K=8/16 are strongly mask- and session-sensitive.** Their negative grand means are driven by one pathological recording and conflict with positive medians and 14/15 positive session means. The mechanism is unresolved; do not call this a population-wide collapse or numerical ill-conditioning.

3. **K=all has the highest aggregate mean** in all six view/budget cells, with R² 0.15–0.43. The reproduction audit confirms exact R² equality with A2a-v2 `dense_equal_trial` in 90/90 cells and coefficient-SHA equality in 90/90 cells.

4. **all−K1 delta is significant at M=30/M=50**: 14/15 sessions positive (p=0.001, CI excludes 0). At M=15 the CI includes 0.

5. **Do not claim a globally monotone finite-K curve.** The predeclared, interpretable primary contrast is `K=all−K=1`; the intermediate-`K` mean/median conflict must remain visible rather than being used for post-hoc best-K selection.

---

## 4. Reproduction gate

K=all ↔ A2a-v2 `dense_equal_trial`:

- **90/90 cells checked** (15 sessions × 2 views × 3 budgets)
- **R² max absolute difference: 0.0** (exact)
- **Coefficients SHA exact matches: 90/90**
- **Prediction SHA exact matches: 75/90** (15 cells differ in SHA due to batch-ordering float arithmetic, but R² and coefficients are identical)

Gate tolerance: 5e-5. All cells passed at 0.0e+00. The 15 prediction-SHA mismatches are consistent with different matrix-multiplication batching; they do not change coefficients or R². The final receipt seals the combined cell content, but it records the 12 batch filenames rather than their SHA-256 values. The separate post-hoc archival-integrity manifest, SHA `372bb81bffc0daa1417e88bdc81473c987981d082aa61b31f4b9b72363d9dfe8`, records all 12 batch digests and verifies their canonical ordered cell concatenation against the final receipt. It does not prove historical producer lineage.

---

## 5. What the paper can say

> Under a fixed dense 2-D velocity target, fixed 50×N representation, normalized weighted ridge solver (λ=1), equal-trial weighting, and target-blind nested row selection, the three-seed aggregate improves from K=1 to K=4 in both SUA and pseudo-MUA. The all-windows arm gives R² 0.15–0.43 and exceeds K=1 in 14/15 sessions at M=30 and M=50 (exact sign-test p=0.00098; bootstrap intervals exclude zero). At M=15, 13/15 signs are positive but the bootstrap intervals include zero. Separately, the A2a dense−direction gap is approximately +0.46 to +0.53 at M=50 after correcting solver normalization and trial weighting; this is a target-formulation contrast, not a pure label-density contrast.

## 6. What the paper cannot say

- "T4 and K=1 ridge receive the same information" — they use different target semantics, different pretrained priors, and different model structures. T4 vs ridge is a system comparison.
- "Label density monotonically improves ridge at every finite K" — the aggregate curve is non-monotone and becomes strongly session/mask sensitive at K=8/16.
- "K=8/K=16 prove numerical ill-conditioning" — the required condition-number, residual, and precision/primal-dual diagnostics were not recorded.
- "The effect reverses for the population at K≥8" — positive medians and 14/15 positive session means show that the negative aggregate is driven by one exceptional session.
- No post-hoc best-K arm may be selected as a causal baseline.
- A2 does not establish that T4 beats `K=all`, that the systems are information-matched, or that Ridge50's system-level advantage is caused solely by label density.

---

## 7. Non-interference statement

No GPU was used. No process was signalled. No watched directory was written. Thread caps OMP/MKL/OPENBLAS=4, nice -n 10. All output to `sua_exploration/results/trial_level_ridge_v1/`.

---

## 8. Files

| File | Purpose | SHA-256 |
|---|---|---|
| `priority_a2_weighting_control_v2_receipt.json` | A2a-v2 corrected weighting results | `b6a080c4...` |
| `priority_a2_same_target_density_v2_receipt.json` | A2b-v2 corrected dose-response | `0d4cd01b...` |
| `priority_a2_weighting_receipt.json` | **INVALID** old A2a (diagnostic only) | `6364b99f...` |
| `priority_a2_same_target_density_receipt.json` | **INVALID** old A2b (diagnostic only) | `0fa3f0e3...` |
| `a2b_v2_batches/batch_*.json` (12 files) | Per-session batch intermediates | filenames only in final receipt; SHA values in release audit |
| `mc_maze/priority_a2_normalized_ridge_v2.py` | Corrected normalized ridge core | `33436003...` |
| `scripts/run_priority_a2_same_target_density_v2.py` | Receipt-bound A2b-v2 batch/combine runner | `e52b29f9...` |
| `manifests/a2b_v2_posthoc_archival_integrity_20260811.json` | Post-hoc batch/final integrity closure | `372bb81b...` |
| `scripts/verify_priority_a2b_v2_archival_integrity.py` | Read-only archival verifier | versioned with release |
