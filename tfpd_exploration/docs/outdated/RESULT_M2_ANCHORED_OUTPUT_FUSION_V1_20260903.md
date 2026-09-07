# Result — M2 Anchored Output Fusion V1

Date: 2026-09-03  
Status: **terminal source-gate failure; no refit, target scoring, packaging, or submission**  
Method: **AOF** (Anchored Output Fusion), one global output-residual scalar

## 1. Question

Pure Post-Fusion retained complementary signal but was substantially worse than
the selected native POOLED comparator. Identity-space scalar anchoring recovered
only a small gain. AOF tested the remaining minimal hypothesis: because the
decoder is nonlinear, interpolate the two frozen predictions after decoding
rather than interpolate their identities before decoding.

```text
y_native = D(x, h_native)
y_post   = D(x, h_post)
y_AOF    = y_native + beta * (y_post - y_native)
```

The backbone, identity encoder, T4 carrier, normalizer, and decoder were frozen.
No optimizer, parameter update, pseudo-label, online state, or hidden/external
target was used. Beta was the declared equal-session MSE-surrogate closed-form
solution on five source sessions, followed by one evaluation on two source
validation sessions.

## 2. Immutable result

Result root:

```text
tfpd_exploration/results/m2_anchored_output_fusion_v1/source_screen
```

The root contains exactly seven immutable body/sidecar pairs:

```text
attempt.json
launch.json
source_authority.json
paired_output_authority.json
fit.json
validation.json
terminal.json
```

There is no `failure.json` and no `all7_refit.json`. Every body and sidecar is
regular, mode `0444`, and link count one. The sidecars exactly match their
body SHA-256.

Key digests:

```text
attempt.json   a624183bcd6d1e21dddbb03320aab19a0b156c22e7a350f49d7adb60e5c6518c
fit.json       b0aba60f798e5364252b7da2b573b02064912f3e20baf0577fb85f5af4f6044a
validation     b0f196cfd554c7c0cb2ead44ca249fc13e4167ec99b5501812448dbb210aab8d
terminal       ab7f864e8922dc0afe7ce445f09e043f206a4eb55b70ce7620e9373ecc8c71cd
closure        28f54813f90563eb779cf76af931b3c9cef4dd1bc1eb34955c372874bfabdd5f
```

The attempt binds design SHA
`48f8f2a1ea7aa99267a230bb08d982e02ee706b955f6ebdac1d717158b998d8e`
and work-order SHA
`1cacd0f6d52ed3475997b5fcff184f502e20fb224e6270028f66908d1de3e1fe`.

## 3. Authority checks

The executed source screen used the exact PIT-cubic first-30 activity authority,
D-optimal four-trial support, frozen ridge T4 carrier, and selected-T4 model.
Every direct native prediction reproduced the immutable
`within_post30 / ridge_activity30_m4` prediction witness.

The official cached EvalAI identity is a CPU artifact, whereas the sealed source
score used a GPU-computed identity. The result therefore records a numerical
bridge rather than falsely claiming cross-device SHA equality. Across all seven
source sessions:

| Bridge statistic | Maximum | Limit |
|---|---:|---:|
| CPU-payload vs GPU identity max-abs | 1.9073486328125e-6 | 2e-6 |
| Corresponding prediction max-abs | 4.0978193283081055e-8 | 2e-6 |
| Corresponding R2 absolute difference | 6.255456042048024e-8 | 2e-7 |

All bridges passed. The source result remains a GPU-screen result; EvalAI
exporter/runtime/container parity was not claimed or exercised.

## 4. Closed-form fit

The five-session equal-session MSE surrogate produced:

```text
beta                 -0.42342679425326024
numerator            -2.058763583040373e-05
denominator           4.862147627362912e-05
conditioning ratio    4.862147627362912e-05
behavior scale        5.0
```

The negative beta says the source fit preferred moving slightly *away* from the
raw Post-Fusion prediction. This is evidence that the Post-Fusion residual is
not aligned with a single globally useful scalar direction.

## 5. Predeclared source validation

| Source validation session | Native R2 | Post R2 | AOF R2 | AOF - native |
|---|---:|---:|---:|---:|
| `ses-2020-10-27-Run2` | 0.6723585730 | 0.5891293305 | 0.6701347349 | -0.0022238381 |
| `ses-2020-10-28-Run1` | 0.7140866539 | 0.6377953862 | 0.7208845501 | +0.0067978962 |

Aggregate result:

```text
mean delta       +0.0022870291   required >= +0.005
positive count    1 / 2          required 2 / 2
worst delta      -0.0022238381   required >= +0.001
```

All three scientific gate components failed. The implementation therefore
correctly stopped without an all-seven refit or downstream evaluation.

## 6. Interpretation

AOF rules out the simplest explanation that Post-Fusion failed merely because
its residual was inserted before a nonlinear decoder. Moving the same residual
to output space changes the outcome from a large negative pure-PF gap to a
small average positive effect, but that effect is not stable across sessions
and is below the predeclared practical threshold.

The correct claim is narrow:

- Post-Fusion contains weak complementary output information.
- One global scalar cannot orient that information consistently across source
  sessions.
- AOF V1 is not a transfer method result and must not be submitted to EvalAI.

The only scientifically coherent PF successor is a separately predeclared,
very-low-dimensional output-residual orientation test. It must use grouped
source out-of-fold evaluation and an exact native anchor; it must not be framed
as a retry of the failed scalar split.

