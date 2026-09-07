# Result: H1 Calibration-Profile FiLM V2

Date: 2026-09-04  
Status: `COMPLETE_FILM_EARLY_REPLICATION_ONLY`

## 1. Outcome

The five-date source-LODO 2x2 experiment completed under the frozen V2
numerical-anchor successor.  CP-FiLM passes on the native early-pooling path
and does not pass on the LP-R3 late-pooling path.

| reading | mean R2 delta | nonnegative dates | worst date | gate |
|---|---:|---:|---:|---|
| EP-FILM - EP-ZERO | **+0.023929** | **4/5** | **-0.008004** | **PASS** |
| LP-FILM - LP-ZERO | -0.000216 | 3/5 | -0.006885 | FAIL |
| LP-ZERO - EP-ZERO | +0.040018 | 5/5 | +0.005055 | descriptive |
| LP-FILM - EP-FILM | +0.015872 | 3/5 | -0.009734 | descriptive |

The predeclared classification is `FILM_EARLY_REPLICATION_ONLY`.  The selected
H1 product remains the already sealed LP-R3 substrate; this experiment does not
authorize an H1 FiLM all-source refit or EvalAI submission.

## 2. Per-date results

| outer date | EP-ZERO | EP-FILM | early gain | LP-ZERO | LP-FILM | late gain |
|---|---:|---:|---:|---:|---:|---:|
| 19250108 | 0.402106 | 0.424035 | +0.021928 | 0.454789 | 0.458081 | +0.003292 |
| 19250113 | 0.295903 | 0.359936 | +0.064033 | 0.400947 | 0.405851 | +0.004904 |
| 19250115 | 0.584963 | 0.601277 | +0.016314 | 0.590018 | 0.591543 | +0.001525 |
| 19250119 | 0.347527 | 0.339524 | -0.008004 | 0.356367 | 0.352449 | -0.003917 |
| 19250120 | 0.399425 | 0.424800 | +0.025376 | 0.427894 | 0.421010 | -0.006885 |
| equal-date mean | 0.405985 | 0.429914 | **+0.023929** | 0.446003 | 0.445787 | **-0.000216** |

The incremental pooling-by-FiLM interaction is -0.024146 R2: the FiLM gain is
materially smaller after the nonlinear late-pooling substrate.  This is a
factorial reading of the two incremental contrasts, not a claim that one
absolute architecture is universally superior.

## 3. What the experiment establishes

The direct M2-to-H1 replication succeeds.  Both datasets use the same
Calibration-Profile FiLM contract:

```text
per-trial pre_pool features
    -> cross-trial mean
    -> carrier4 + state-contrast-profile4 FiLM
    -> nonlinear post_pool
    -> frozen decoder
```

The module is `Linear(8,8) -> ReLU -> Linear(8,2H)`, has an exactly zero final
layer at initialization, uses the frozen profile mask `[1,1,0,0]`, and trains
only the FiLM parameters for 12 epochs at learning rate 3e-4.  H1 uses H=32
(648 parameters); M2 uses H=64 (1,224 parameters).  The width follows the
frozen backbone, while the operator and interface are unchanged.

M2 submission 581801 improved official held-out R2 from 0.303244 to 0.320281,
an increment of +0.017037.  H1 now supplies an independent source-date-LODO
increment of +0.023929 on the architecturally matched early path.  The
magnitudes should not be pooled statistically because one is an official
hidden surface and the other is source LODO; together they support a
cross-dataset mechanism claim, not a common-effect estimate.

The late arm answers a different question.  H1 late pooling is valuable by
itself (+0.040018 over early zero), but adding the same FiLM after moving the
nonlinear `post_pool` inside each trial gives essentially zero further gain
(-0.000216).  Therefore the data do **not** support pooling-position invariance
or the claim that FiLM should be attached after an arbitrary identity MLP.

## 4. Global story to use

The transferable method is not “every dataset gets a different encoder.”  It
is one calibration-conditioned operator at one semantic interface:

> CP-FiLM modulates the aggregated per-unit calibration feature before the
> nonlinear identity projector.  A zero-initialized 8-to-2H FiLM conditioned
> on a four-dimensional carrier and a four-dimensional task-state contrast
> improves the native calibration encoder on both M2 and H1.

Pooling remains an explicit backbone factor.  H1 additionally benefits from a
late-pooling alternative, LP-R3, but that benefit is non-additive with CP-FiLM.
LP-R3 is therefore an H1 product choice and a negative composability result,
not a redefinition of the cross-dataset FiLM method.

The task-state semantics are task-specific but the descriptor construction is
shared: two conditional neural states produce per-unit mean difference,
log-ratio, and two dispersion statistics, robust-z normalized across units;
only the first two columns enter FiLM.  M2 uses hold versus reach, whereas H1
uses low- versus high-speed bins from the labeled M3 calibration prefix.  This
is a change in the meaning of the task states, not in the network API.

## 5. Claims that are not licensed

- Do not say M2 used late pooling; its successful FiLM is early/native.
- Do not say FiLM is pooling-position robust; the late incremental gate failed.
- Do not claim LP-FILM beats LP-R3; its equal-date mean is lower by 0.000216.
- Do not compare the absolute EP-FILM and LP-FILM scores as a clean FiLM effect;
  their sealed zero substrates differ by +0.040018.
- Do not claim official H1 improvement before an authorized H1 EvalAI result.
- Do not attribute the cross-dataset effect to identical behavioral labels;
  only the descriptor algebra and FiLM interface are shared.
- Do not start an epoch/LR/seed/profile-mask sweep from this result.  The frozen
  12-epoch recipe already produced a gated replication on the intended path.

## 6. Training and anchor audit

All five folds completed 12 epochs.  Early-FiLM mean task loss decreased from
epoch 0 to epoch 11 in every fold.  Late-FiLM loss was nearly flat, consistent
with its null incremental score, but this is descriptive rather than an epoch
selection rule.  Every update had a nonzero finite FiLM gradient; frozen C1,
LP-R3, and shared decoder state hashes were unchanged.

The V1 failure was caused by requiring a historical CUDA prediction byte SHA
across processes.  V2 retained exact model, checkpoint, activity, carrier,
profile, query, target, and state authorities; required immediate same-process
prediction repetition to be byte-identical; and required historical R2 error
at most 1e-7.  Across all 11 target records:

- all immediate repeats were byte-identical;
- all EP-ZERO historical prediction SHAs matched;
- LP-ZERO historical SHAs differed, but all R2 discrepancies were within the
  declared numerical tolerance;
- target optimizer, backward, and model-update counts were zero.

The V2 artifact graph contains exactly 23 immutable bodies and 23 matching
sidecars.  There is no failure leaf.  Primary authorities are:

- attempt SHA256: `6ede8bd594ac639cbf2a2d3edce9c1de8f7e00d948278592ed43ad29b19f860a`;
- score SHA256: `8b56b105bf1cd965ffa9d74e323dbce7b5d2e2410a8abfb88b5eaa2151494813`;
- terminal SHA256: `f6432d0832d0173313dc9678078e6e4d348928218396b602d7320ec1b7dcb38d`.

No formal H1 held-out data or EvalAI result was opened.  GPU1 was not queried
or touched, and the scheduled LP-R3 packaging/submission timer was not changed.
