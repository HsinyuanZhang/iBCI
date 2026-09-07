# Synthesis: Calibration-Profile FiLM Across M2 and H1

Date: 2026-09-04  
Status: `SUPPORTED_ON_MATCHED_EARLY_POOLING_INTERFACE`

## One-sentence result

Calibration-Profile FiLM improves the native early-pooling calibration encoder
on two distinct FALCON tasks (M2 official +0.0170 R2; H1 source LODO +0.0239
R2), while H1 shows that moving the nonlinear identity projection before the
trial mean is a useful but separate, non-additive pooling strategy.

## Canonical cross-dataset method

The method definition is fixed at the calibration aggregation bottleneck:

```text
activity trials
  -> shared per-trial pre_pool
  -> mean across calibration trials
  -> CP-FiLM(carrier4, task_profile4)
  -> shared nonlinear post_pool
  -> frozen neural decoder
```

Only dimensional width changes with the existing backbone.  The FiLM rank,
context dimension, zero initialization, profile column algebra, `[1,1,0,0]`
mask, frozen decoder rule, and 12-epoch training recipe remain common.

## Pooling interpretation

H1's LP-R3 computes nonlinear post-pool identities per trial and averages them
afterward.  It is a strong H1 baseline (+0.0400 over early zero), but CP-FiLM
adds -0.0002 on top.  Consequently:

- use early/native CP-FiLM as the cross-dataset scientific mechanism;
- retain LP-R3 as the current H1 deployment candidate;
- present late pooling as a crossed architectural ablation and complementary
  H1 inductive bias, not as part of the FiLM definition;
- do not claim that FiLM is invariant to pooling position.

This preserves a globally consistent method without forcing every dataset to
share the same winning product configuration.  Scientific transfer concerns
the FiLM interface; deployment selection may still reflect task-specific
backbone evidence.

## Paper-safe wording

> We condition the calibration encoder through a shared, zero-initialized FiLM
> interface placed after aggregation of per-trial unit features and before the
> nonlinear identity projector.  The same carrier-plus-state-profile operator
> improves M2 on the official hidden evaluation and improves H1 under
> source-date leave-one-out evaluation.  A factorial H1 test shows that late
> pooling is independently beneficial but does not provide additional FiLM
> gain, localizing the transferable mechanism to the pre-projector aggregation
> interface rather than to arbitrary post-MLP pooling.

## Reviewer-facing caveat

M2 and H1 do not expose identical behavioral state labels.  M2 uses hold/reach;
H1 uses low/high speed within the M3 labeled calibration prefix.  The supported
invariance is the two-state conditional-moment descriptor and FiLM interface,
not the literal meaning of each state.  H1 remains source-only evidence until
an official hidden evaluation is authorized and completed.
