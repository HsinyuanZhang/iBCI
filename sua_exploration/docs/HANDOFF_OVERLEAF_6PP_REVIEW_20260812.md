# Handoff: Final Corrections for the Six-Page Overleaf Draft

**Date:** 2026-08-12  
**File:** `bci_paper_overleaf/paper_6pp.tex`  
**Current commit:** `9de0608`  
**Review status:** `MINOR REVISION`  
**Scope:** text and equations only. Do not reopen experiments or change sealed results.

## Corrections already accepted

The latest revision correctly:

- removed the unsupported matched-label T4--Ridge `+0.25 R²` claim;
- retained A2b-v2 only as a within-Ridge supervision-density result;
- separated the H1 population carrier from the per-unit OLS carrier family;
- corrected calibration length to `T_cal` and placed trial averaging after `f_pre`;
- introduced the carrier width `d_c`;
- narrowed the full-carrier pooling claim because modulation depth is not additive;
- clarified target-session query-label exclusion;
- kept H1 Context Full, event tags, tag shuffle, and the `0.5165` result out of the paper.

Do not revert these corrections.

## Remaining required corrections

### 1. Remove the dangling attention coefficient

The old statement `sum_i alpha_{c,i}=1` remains even though `alpha_{c,i}` is no longer defined after replacing the single-head equation with MHA.

Use a standard three-input expression:

```text
Q_tilde = Q + Dropout(MHA(LN(Q), LN(H), LN(H)))
```

Then justify permutation invariance directly:

> Multi-head attention is invariant to a shared permutation of its key and value rows; therefore, reordering the unit set leaves the decoded output unchanged.

### 2. Make the pooling proof intercept-consistent

The primary estimator now uses the intercept-augmented matrix `X=[1,Phi]`, but the pooling equation incorrectly returns to `(Phi^T Phi)^-1 Phi^T r`.

Use the complete coefficient vector:

```text
beta_i = X^dagger r_i = [b_i; w_i]
beta_j = X^dagger sum_i r_i = sum_i beta_i
```

This proves that both the intercept and the linear coefficient block are pooling-covariant. Keep the existing qualification that the modulation norm is recomputed after pooling and is not additive. Prefer `X^dagger` in the primary OLS equation as well, matching the implemented least-squares solve without assuming invertibility.

### 3. Remove the unsupported H1 conditioning explanation

Do not say that four H1 trials prove the per-unit OLS is poorly conditioned. The trials are long, and conditioning was not isolated as the reason for using the H1 estimator.

Use:

> H1 uses a separately specified population-level carrier for its dense 7-DoF calibration regime. The estimator projects session activity into a source-frozen neural subspace, regresses velocity in that subspace, back-projects the coefficients to channel rows, and applies empirical-Bayes shrinkage toward a source prior. This is not the per-unit estimator above, and no per-unit OLS pooling-linearity claim is made for it.

Also change the deployment sentence so that it does not say every carrier comes from the per-unit OLS equation:

> At deployment, the descriptor is obtained from the per-unit closed-form estimator for T4/RT or from the separately specified population estimator on H1.

### 4. Make the H1 Ridge comparison explicit and descriptive

Replace `below all decoder arms (0.47--0.53)` and avoid the phrase `same ridge protocol`.

Use:

> On the H1 fold-0 cohort, the fixed four-trial Ridge v2r2 reference reaches pooled R²=0.2582, compared descriptively with 0.4968 for H-S and 0.5255 for H-C on the same post-support query. This comparison does not isolate architecture, supervision density, representation, carrier content, or the cause of the performance difference.

### 5. Report the actual compiled page count

Compile `paper_6pp.tex` in Overleaf and report:

- total manuscript pages;
- whether references are included in the six-page limit;
- any overfull boxes or layout failures.

The source currently has approximately 4,395 words, five figures, and three tables. If it exceeds six pages, shorten the detailed decoder derivation, repeated supervision accounting, and secondary boundary discussion before removing primary evidence.

## Content that must remain unchanged

- No H1 Context Full, event-tag, tag-shuffle, or `0.5165` sparse-context claim.
- No matched-information or matched-label T4--Ridge claim.
- Preserve the valid within-Ridge A2b-v2 result:
  - SUA M30/M50: `+0.2891/+0.3282`;
  - pseudo-MUA M30/M50: `+0.2966/+0.3122`;
  - M50: 14/15 positive sessions in both views.
- Preserve H1 Ridge v2r2 only as a descriptive deployment-boundary reference.
- Preserve the five-date H1 compact-consumer result and organizer-held aggregate with their existing uncertainty and scope qualifications.
- Preserve the distinction among algorithmic target-supervision consumption, human annotation cost, and compute cost.

## Required response

Return:

1. the new commit SHA;
2. exact line references for the five corrections;
3. the compiled page count and any layout warnings;
4. confirmation that all forbidden H1 tag/Context Full and matched-label T4--Ridge language remains absent;
5. final status: `ACCEPT` or a short list of remaining blockers.

