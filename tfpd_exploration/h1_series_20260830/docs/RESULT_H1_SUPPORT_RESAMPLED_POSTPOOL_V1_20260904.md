# Result: H1 Support-Resampled Post-Pool V1

Date: 2026-09-04  
Status: `COMPLETE_PASS_LP_R3_FOR_SEPARATE_ALL_SOURCE_WORKORDER`

## 1. Result

The missing branch-only H1 cell succeeded.  Starting from each sealed C1
date-LODO epoch-49 checkpoint, the experiment froze the complete decoder and
trained only the 58,140-parameter MLP-after-pooling identity branch for 12
epochs.  The selected `LP-R3` arm improved over frozen C1 by **+0.040018 R2**
on average, was positive on **5/5 dates**, and had worst-date gain
**+0.005055**.  It passed the preregistered `+0.005 / 4-of-5 /
worst>=-0.010` gate.

| outer date | frozen C1 | LP-F3 | LP-R3 | SRPD | F3-C1 | R3-C1 | SRPD-C1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 19250108 | 0.402106 | 0.425709 | 0.454789 | 0.451214 | +0.023602 | +0.052683 | +0.049108 |
| 19250113 | 0.295903 | 0.388135 | 0.400947 | 0.402748 | +0.092232 | +0.105044 | +0.106845 |
| 19250115 | 0.584963 | 0.581183 | 0.590018 | 0.593382 | -0.003780 | +0.005055 | +0.008419 |
| 19250119 | 0.347527 | 0.337304 | 0.356367 | 0.368309 | -0.010223 | +0.008839 | +0.020782 |
| 19250120 | 0.399425 | 0.384112 | 0.427894 | 0.410311 | -0.015313 | +0.028470 | +0.010887 |
| **mean** | **0.405985** | **0.423289** | **0.446003** | **0.445193** | **+0.017304** | **+0.040018** | **+0.039208** |

`LP-F3` had positive mean but failed stability: only 2/5 dates were
nonnegative and worst was -0.015313.  `SRPD` also passed its primary gate, but
its increment over `LP-R3` was -0.000810 on average and worst -0.017583, so it
failed the distillation safety gate.  The frozen decision law therefore chose
the simpler `LP-R3` arm.

## 2. Main discovery

Support resampling was the decisive factor.  `LP-R3 - LP-F3` was positive on
all five dates:

`[+0.029081, +0.012812, +0.008834, +0.019063, +0.043782]`,

with mean **+0.022714 R2**.  Both arms used exact M3, the same optimizer and
12-epoch budget, and the same late-pool operator.  `LP-R3` differed only by
training on a deterministic 50/50 mixture of chronological first M3 and a
non-first contiguous M3 block, with activity and H-C carrier always fitted
from the same three trials.  Deployment evaluation still used chronological
first M3.

This changes the earlier interpretation.  H1 did not show that MLP-after-
pooling is intrinsically weak.  It showed that a late-pool identity branch
trained on one fixed support realization does not transfer reliably.  Exact-
M3 support diversity, together with a frozen C1 decoder, converted that
unstable solution into a 5/5 positive cross-record result.

The result also explains the prior full-model failure.  On outer date
19250115, full-model matched retraining had lost about 0.09 R2.  Branch-only
`LP-R3` instead gained +0.005055 and `SRPD` gained +0.008419.  The unsafe
component was the 10.95M-parameter decoder/body update, not late pooling
itself.

## 3. What is and is not established

Established:

- late pooling is trainable under the official H1 M3 cardinality;
- freezing the C1 decoder avoids the large cross-date regression seen with
  full-model retraining;
- source-side diversity of matched M3 activity+carrier support is a large and
  consistent regularizer;
- native-identity distillation is unnecessary in this cell.

Not established:

- whether the +0.022714 resampling effect is caused mainly by activity
  diversity, carrier diversity, or their matched interaction;
- whether an all-source model transfers to the formal held-out/EvalAI surface;
- whether query-time continual trial memory is deployable in the H1 evaluator.

The preregistered carrier-extension trigger (`abs mean >=0.003` and common
sign on at least 4/5 dates) passed strongly.  A separate 2x2 source-only
factorial may therefore decompose activity versus carrier resampling.  It is a
mechanism follow-up and cannot alter this result.

## 4. Integrity evidence

- all five folds completed exactly 12 epochs;
- optimizer steps per arm were 1464, 1680, 1680, 1668, and 1680;
- every step had finite nonzero identity-branch gradients;
- LP-R3 and SRPD used identical support selections on every update;
- all three arms were bitwise identical in identity and prediction before the
  first update;
- all frozen decoder/body state hashes were unchanged before versus after
  training;
- epoch-12 branch checkpoints were sealed and strict-reloaded before opening
  each outer date;
- target optimizer, backward, and model-update counts were all zero;
- formal held-out and EvalAI were not opened;
- only GPU0 was used; GPU1 was not queried or touched by this route.

## 5. Immutable authority

- result root: `tfpd_exploration/h1_series_20260830/results/h1_support_resampled_postpool_v1/`
- attempt SHA256: `718da28861258d5461da2d19cc486a55d0cc876bd97e58eb5663045a30bba4ef`
- score SHA256: `a07ab80612de53a625b5dc29ef63bc2c24ce249979d5c3afdc43399fb27851b5`
- terminal SHA256: `c047bbf5bd6e324fcd250ef2702ab507fff54b6a13c6bc1314963983bc298a94`
- design SHA256 at launch: `cb26212d85304bf4070a9bd2fa0c5fe0e8fde342e76b0ea6417ae05185d7a0ff`
- work order SHA256 at launch: `b891108a4116dc33766ab18ac1c843082cbe6052f857d1245e047163c0f88753`

