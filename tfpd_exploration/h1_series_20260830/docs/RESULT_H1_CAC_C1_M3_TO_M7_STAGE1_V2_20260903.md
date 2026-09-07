# Result: H1-CAC C1/M3→M7 Frozen-Weight Stage 1 V2

Date: 2026-09-03  
Status: `COMPLETE_H1_CAC_STAGE1_C1_M3_FROZEN_WEIGHT_SCORE`  
Verdict: `PASS_C1_M3_ACTIVITY_CONTENT_BUT_STOP_DEPLOYABLE_CAC`

## 1. Executive result

The C1 consumer has a large and consistent capacity to use causal activity
added after its static M3 calibration identity:

- true-trial content upper bound, `B-TRIAL7 - A-STATIC`:
  **+0.078573 R²**, positive on **5/5 dates and 11/11 recordings**;
- preregistered energy-median deployment arm, `D-EMED7 - A-STATIC`:
  **+0.030115 R²**, nonnegative on **4/5 dates and 10/11 recordings**;
- selection-informed fixed-chunk readout, `C-FIX7 - A-STATIC`:
  **+0.020064 R²**, nonnegative on **3/5 dates and 8/11 recordings**.

The content gate passes strongly. The preregistered deployment gate does not:
the energy-median arm recovers 38.33% of the true-trial headroom, below 50%,
and its worst date is -0.019740, below the -0.010 safety floor. The fixed-chunk
readout also does not pass and remains explicitly post-Stage-0
selection-informed and non-governing.

This result means **activity completion is real under the submitted C1/M3
consumer, but the present boundary-free selectors are not yet reliable enough
to package or submit**. It is not a failure of C1, M3 extrapolation, or the
activity-memory mechanism.

## 2. Governed result table

All values are equal-recording R² on the common strict post-M3 surface.

| Outer date | A-STATIC | B-TRIAL7 | C-FIX7 | D-EMED7 | B−A | C−A | D−A |
|---|---:|---:|---:|---:|---:|---:|---:|
| 19250108 | 0.405181 | 0.495058 | 0.458604 | 0.453462 | +0.089877 | +0.053422 | +0.048281 |
| 19250113 | 0.197022 | 0.239517 | 0.168231 | 0.177283 | +0.042495 | −0.028792 | −0.019740 |
| 19250115 | 0.466369 | 0.550035 | 0.518994 | 0.518530 | +0.083666 | +0.052624 | +0.052160 |
| 19250119 | 0.313423 | 0.384417 | 0.383344 | 0.368144 | +0.070994 | +0.069921 | +0.054721 |
| 19250120 | 0.327243 | 0.433076 | 0.280387 | 0.342397 | +0.105834 | −0.046856 | +0.015155 |
| **Equal-date mean** | **0.341848** | **0.420421** | **0.361912** | **0.371963** | **+0.078573** | **+0.020064** | **+0.030115** |

The per-recording descriptive counts are:

- B−A: 11/11 positive; worst +0.039988;
- C−A: 8/11 positive; worst −0.086958;
- D−A: 10/11 positive; worst −0.065998.

The date-level gates remain governing; recording-level counts are descriptive
and do not replace them.

## 3. Gate audit

### Gate 1: activity content

Required: mean B−A at least +0.010 and at least 4/5 dates positive.

Observed: +0.078573 and 5/5. **PASS.**

This is the principal scientific result. Moving from the frozen M3 identity to
M7 using completed true trials improves every date and every recording. It is
also larger than the earlier H-C/M4 Stage-0 true-trial gain of +0.031999, so
the headroom did not disappear when transferred to the actual C1/M3 consumer.

### Gate 2: preregistered boundary-free detector

Required: D recovery at least 0.50, at least 3/5 dates nonnegative, and worst
date at least -0.010.

Observed: recovery 0.383279, 4/5 dates nonnegative, worst -0.019740. **FAIL.**

The positive mean should be retained as descriptive evidence, but it does not
authorize packaging. The failure is concentrated rather than universal:
19250113 is negative, while four dates are positive.

### Selection-informed C readout

Observed: +0.020064, recovery 0.255356, 3/5 nonnegative, worst -0.046856.
**FAIL.** This arm was retained because C exceeded D in Stage 0; therefore it
cannot replace D or be presented as an independent confirmation.

## 4. Authority and no-update facts

Successful immutable root:

`tfpd_exploration/h1_series_20260830/results/h1_causal_activity_completion_v1/stage1_c1_m3_v2/`

- attempt SHA-256: `84f2ff655d77d851ae28b4fb8105725d2bc16861ca97c032da60c2d3eadf3b38`
- input authority SHA-256: `dfda2e5a5771e067c7059667c3b544417e3e9749ccd2bb8a97970329472dc23a`
- score SHA-256: `7199409652e6d9b8eab190f1db7211daaaf0d0b533dcb986e7838c46a8bd5a3a`
- terminal SHA-256: `bb5858d66bea0162278e6d8a85546132cab58106da482e27d745d9c8cacccd4a`
- fixed source commit: `5b21de415afc35a5f4ad63dd2e8a459d925dbbf7`
- target optimizer, backward, and model-update counts: all zero
- formal held-out, minival, and EvalAI access: all false
- all four arms used the same per-date checkpoint, source-selected carrier
  plan, M3 support carrier, query windows, and targets;
- total recorded frozen-forward wall time: 83.60 seconds on physical GPU0.

Five checkpoints were strict-loaded with zero missing/unexpected keys and exact
terminal state hashes. Their source-only plans were reconstructed only because
the remote upload omitted the NPZ bodies; every array digest, normalizer,
source partition, and final NPZ byte digest matched its frozen authority.

## 5. V1 integration incident

The first attempt wrote only attempt+failure and produced no score. It called
the historical M4-only `fit_frozen_carrier` on M3 and was rejected before
scoring. The immutable incident is documented in
`INCIDENT_H1_CAC_C1_M3_STAGE1_V1_20260903.md`.

The V2 successor changed only the helper to the already reviewed
`fit_deployment_carrier`, which accepts exactly three or four unique trials and
forbids padding or duplication. Checkpoints, plans, surface, arms, and gates
were unchanged.

## 6. Allowed interpretation and next decision

Allowed claim:

> Under five source-grouped C1 date-LODO models, causally extending the H1
> activity identity from M3 to M7 using completed true trials improved
> equal-date R² by 0.0786 and was positive on all dates and recordings.

Not allowed:

- claiming that the current energy-median or fixed-chunk selector is ready for
  EvalAI;
- presenting B as officially deployable;
- relabeling the selection-informed C arm as a preregistered success;
- attributing the gain to a changed carrier or target adaptation.

The next useful H1 cell, if authorized, is a **selector-only source-grouped
successor**: keep the exact frozen C1 checkpoints, M3 carrier, M3→M7 cap, and
decode-before-commit operator; change only the label-free rule deciding which
completed activity chunks enter the four available slots. The target is not
more network capacity—the content upper bound already proves capacity—but a
selector that recovers more of the +0.0786 headroom without the two observed
date-level regressions.
