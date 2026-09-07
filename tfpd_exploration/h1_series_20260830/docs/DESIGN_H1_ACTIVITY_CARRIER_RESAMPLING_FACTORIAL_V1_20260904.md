# Design: H1 Activity/Carrier Resampling Factorial V1

Date: 2026-09-04  
Status: `FROZEN_TRIGGERED_SOURCE_ONLY_MECHANISM_FOLLOWUP`

## 1. Trigger and question

The preregistered H1 support-resampling trigger fired: `LP-R3 - LP-F3` was
`+0.022714 R2` on average and positive on 5/5 outer dates.  The successful
LP-R3 arm resampled activity and its matched H-C carrier together, so the main
experiment cannot say which component caused the gain.

This follow-up completes one 2x2 training factorial.  It is descriptive
mechanism evidence only and cannot change the selected LP-R3 result.

## 2. Factorial

The sealed main experiment already provides:

- `FF = LP-F3`: fixed first-M3 activity + fixed first-M3 carrier;
- `RR = LP-R3`: resampled M3 activity + carrier fitted from that same M3.

This experiment trains only the two missing cells:

- `RF = LP-AR-CF`: resampled M3 activity + fixed first-M3 carrier;
- `FR = LP-AF-CR`: fixed first-M3 activity + carrier fitted from the resampled M3.

The crossed cells are deliberate training-time interventions.  They are not
deployment inputs and are not described as matched biological supports.
Outer-date scoring for every arm uses the legal matched chronological first
M3 activity and carrier.

The new arms inherit the exact LP-R3 contract: sealed C1 fold checkpoint,
frozen decoder/body, only 58,140 identity-branch parameters trainable, late-
pool operator, 12 epochs, batch32, stride4, seed42, Adam `5e-5`, no
distillation, and the same deterministic 50/50 support-index schedule.

## 3. Readings

For each date compute:

- activity-resampling main effect:
  `0.5 * ((RF - FF) + (RR - FR))`;
- carrier-resampling main effect:
  `0.5 * ((FR - FF) + (RR - RF))`;
- interaction:
  `RR - RF - FR + FF`.

A component is called directionally present only when its absolute mean is at
least `0.003 R2` and at least 4/5 dates have the same sign.  These thresholds
classify the mechanism; they do not authorize a new model or EvalAI package.

Interpretation:

- positive activity effect only: trial activity diversity is the useful
  regularizer; carrier should remain canonical first-M3;
- positive carrier effect only: the source benefit comes from exposing the
  identity MLP to uncertainty/variation in the fitted four-dimensional
  carrier;
- both positive with small interaction: approximately additive benefits;
- positive interaction: matched co-resampling matters beyond either marginal
  perturbation;
- neither: the main LP-R3 gain depends on the coupled intervention and should
  not be reduced to one component.

## 4. Boundaries

Only held-in-calib and held-in-minival source/outer-date LODO surfaces are
used.  Training completes and checkpoints are sealed before the corresponding
outer date is opened.  Formal held-out and EvalAI remain closed.  Only GPU0
may be used; GPU1 is not queried or touched.

