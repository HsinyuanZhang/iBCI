# H1 Exactly-M3 Cross-Record Joint Post-Pooling V1 — Result

Date: 2026-09-03  
Status: `COMPLETE_STOP_NO_ALL_SOURCE_RETRAIN_NO_EVALAI`

## Outcome

This five-fold source-only experiment did not pass its preregistered transfer
gate.  Neither trained arm is authorized for all-source retraining or EvalAI.

The experiment used the official held-out deployment support law exactly:
three chronological calibration trials supplied both the frozen H-C carrier
and the activity identity.  Training queries came from a distinct held-in
minival recording, never from the outer-date recording.  Each fold started
from the same sealed C1 epoch-49 checkpoint and ran 12 fixed epochs.

| outer date | frozen C1 | N3-XR12 | J3-XR12 | N3 - frozen | J3 - frozen | J3 - N3 |
|---|---:|---:|---:|---:|---:|---:|
| 19250108 | 0.402106 | 0.420836 | 0.421118 | +0.018729 | +0.019011 | +0.000282 |
| 19250113 | 0.295903 | 0.353226 | 0.350739 | +0.057323 | +0.054836 | -0.002487 |
| 19250115 | 0.584963 | 0.493492 | 0.494541 | -0.091471 | -0.090422 | +0.001049 |
| 19250119 | 0.347527 | 0.345118 | 0.345181 | -0.002409 | -0.002346 | +0.000063 |
| 19250120 | 0.399425 | 0.375944 | 0.373356 | -0.023481 | -0.026068 | -0.002588 |

Equal-date means:

- `N3-XR12 - FROZEN-C1 = -0.008262`, with `2/5` nonnegative dates and
  worst `-0.091471`;
- `J3-XR12 - FROZEN-C1 = -0.008998`, with `2/5` nonnegative dates and
  worst `-0.090422`;
- `J3-XR12 - N3-XR12 = -0.000736`, with `3/5` nonnegative dates and
  worst `-0.002588`.

The registered requirements were mean gain at least `+0.005`, at least `4/5`
nonnegative dates, and worst date at least `-0.010`.  The terminal verdict is
therefore `STOP_H1_M3_CROSS_RECORD_MATCHED_TRAINING`.

## What the result establishes

The failure is not explained by insufficient source optimization.  Source
training loss decreased through epoch 12 in every fold, while cross-record
R2 changed sign across dates.  The unsafe component is therefore the
source-to-recording transfer of the full-model update, not a failure to fit
the source objective.

The post-pooling residual did not reproduce the large post-fusion collapse
seen in the M2 route.  At IEEE `alpha=+0`, N3 and J3 had bitwise-identical
first identities and predictions in every fold, paired RNG consumption was
equal for every update, and the final J3-N3 difference stayed within roughly
`+-0.0026`.  This supports the narrow conclusion that MLP-after-pooling is
architecturally usable on H1 M3, but it does not show a positive incremental
effect.

The two large positive dates (`+0.019` and `+0.055`) show that an exact-M3,
cross-record objective contains useful signal.  The large negative date
(`about -0.09`) means that updating all 10.95 million parameters for 12 epochs
is not a no-regret way to extract it.

## Consequence

Do not rescue this cell by selecting an epoch, excluding a date, increasing
the epoch count, or weakening the gate.  The next independent cell must keep
the sealed C1 decoder intact and use a substantially smaller adaptation
surface.  The first priority is a closed-form, calibration-label readout
correction fitted only from the official M3.  A later identity-only successor
may train a small carrier/post-pool residual, but it must not be described as
confirmation of this failed full-model recipe.

## Immutable authority

- Result root: `tfpd_exploration/h1_series_20260830/results/h1_m3_cross_record_joint_postpool_v1/`
- `attempt.json`: `9ca48b503d1f315bc13617cf2aa9ad6a51df4ad99a5d95afdbdd0c8ae7f79d90`
- `score.json`: `2dd476f62adb4ff293f9fea85d54b0d527ce641ba073218d725b602887bc88fa`
- `terminal.json`: `5decf64d428c8c88eef72138c46c51210931376f4dbc770f497b2249546bdd81`
- Design: `a1361980e217ebd444317a67d5c3b9426a3876313582d1517abccfb998e4e283`
- Work order: `6ebe3cbc38c77ee17b9182998e1b448b8f3c926aa87ef051ba14c68e43478d65`

