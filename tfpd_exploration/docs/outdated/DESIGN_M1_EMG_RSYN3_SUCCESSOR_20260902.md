# M1 EMG-rSyn3 successor: ReLU nonnegative projection before Syn3

**Date:** 2026-09-02  
**Status:** successor design; does not amend or replace the sealed EMG-Syn3
FAIL  
**Parent design:**
`tfpd_exploration/docs/DESIGN_M1_FUNCTIONAL_CARRIER_MEMORY_20260902.md`  
**Parent design SHA-256:**
`766cb7952913fc0ee57f231bc398b1d50b4d768f4d361e77ca185c61ee6a8259`  
**Parent Stage-0 FAIL:**
`tfpd_exploration/docs/RESULT_M1_EMG_SYN3_STAGE0_SIGNED_REJECT_20260902.md`

This document is the scientific rationale for **EMG-rSyn3**. The parent
Functional Carrier Memory design remains the carrier, injection, CQC and CDM-A
authority. This successor adds exactly one frozen operator and forbids reading
the parent FAIL as a scientific rejection of EMG-Syn3.

## 1. Executive decision

Do not abandon EMG-Syn3. Do not modify or overwrite the sealed Stage-0 FAIL at
`tfpd_exploration/results/m1_emg_syn3_fcm_v1/stage0/`.

That FAIL proves only:

> Stored `preprocessed_emg` cannot be passed to NNMF without a frozen
> nonnegative projection.

It does not prove that three-dimensional EMG synergies, bin-level neural–EMG
encoding, an AFC4-like carrier, independent carrier injection, or
carrier-by-CDM-A interaction failed. Those measurements were not taken.

The successor name is:

> **EMG-rSyn3:** Rectified EMG Synergy-3

“Rectified” here means the algebraic projection below, not a claim that the
FALCON stored series is a physiological rectified envelope.

## 2. Unique preprocessing operator

Freeze one operator, used identically on every source session and every target
support bin:

```text
R(x) = max(x, 0)
x_rect = np.maximum(x_raw, 0.0)
```

Mandatory law:

- elementwise;
- `float64`;
- threshold exactly `0.0`;
- executed before source RMS scaling and before NNMF;
- source sessions and target support use the same operator;
- no learnable parameter;
- no target query is read;
- no session- or channel-specific threshold;
- no sweep;
- no later substitution by `abs`, offset, envelope filter, or smoother.

This is not “learn a correction from the target.” It projects an almost
nonnegative stored view onto the nonnegative cone that NNMF requires.

Negative mass in the sealed stored view is of order `1e-4`. That pattern is
consistent with sparse numerical overshoot, not with untreated bipolar raw EMG.
Paper text still must not call the stored tensor a rectified envelope until
preprocessing provenance is bound.

## 3. What remains unchanged from EMG-Syn3

After `R(x)`, the parent design is reused without relaxation:

- fold 0, seed 42, source `{20120926,20120927,20120928}`, target `20120924`;
- support `[0,10)`, report query `[10,210)`;
- source-only RMS (no mean subtraction);
- rank-3 Frobenius NNMF, coordinate descent, NNDSVDa, seed 42;
- frozen-dictionary NNLS target activations;
- bin-level ridge encoding, intercept unpenalized, `lambda=1`;
- independent post-`fc_in` carrier projection `P`;
- controls Zero4 / rSyn3 / RS4 / LS4 / B4 / PCA3;
- Stage-1 arms `Z-Fix`, `S-Fix`, `S-Acyc`;
- static content gate `S-Fix − Z-Fix >= +0.03 R²`;
- CQC remains forbidden until that gate passes.

Matched PCA3 uses the same `R(x)` so that basis-family contrasts are not
confounded with the rectifier. Historical trial-mean PCA-AFC4 reliability is
disclosed on the same rectified bins and is still not a decoder gate.

## 4. Additive ownership

The successor is a new route. It may read the parent FAIL receipts and the
parent design/work-order bytes. It may not write them. Owned paths are listed
in the successor work order.

## 5. Paper-safe wording now

> A source-frozen EMG-Syn3 Stage-0 audit rejected direct NNMF on stored
> `preprocessed_emg` because a sparse negative overshoot of order 1e-4 left
> the nonnegative cone. That rejection does not evaluate synergy
> constructibility or decoder R². The successor EMG-rSyn3 applies the
> parameter-free projection R(x)=max(x,0) before RMS scaling and NNMF. This
> operator is not claimed to recover a physiological rectified envelope.
