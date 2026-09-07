# M2 selected QueryAge ext4 error decomposition — development diagnostic

## Scope and conclusion

This is a read-only decomposition of the already selected native QueryAge
exports on the four-session ext4 development surface.  It explains the
observed severe ROUTE result on `ses-2020-11-19-Run1` (R² = -0.238162) in
terms of prediction moments and squared error; it does not fit, calibrate,
lag-shift, alter, re-rank, or select any model.

The direct evidence is most consistent with **primarily dynamic/tracking
error** on that session, with a real but secondary mean-bias component:
10.800% of ROUTE SSE is explained by the constant two-output mean offset and
89.200% remains after centering both prediction and target.  ROUTE's centered
correlations on that session are only 0.471 and 0.217 for outputs 0 and 1.
That supports the descriptive statement “poor centered tracking, accompanied
by mean bias.”  It does **not** establish a causal mechanism (routing,
session identity, input quality, optimization, or data handling), and this
document recommends no corrective recipe.

All four sessions are reported below.  The surface is explicitly local
development evidence, not a pristine holdout, non-inferiority result, or
selection authority.  The frozen source-only selections remain unchanged:
FLAT plain-EMA epoch 002 and ROUTE plain-EMA epoch 020.

## Bound source artifacts and comparability

| Artifact | SHA-256 / status |
| --- | --- |
| Selected ext4 receipt | `queryage_prefix_pair24_selected_ext4_v1/receipt.json`; `0442ed59d740397b621a6c75b53041bca67205ad5e9407096406371d4b966b0e`; `COMPLETE_FIXED_SELECTED_EXT4_DEVELOPMENT_ONLY` |
| FLAT native archive | `FLAT_selected_ext4_native_float64.npz`; `f89c26c641daca798c6cd02c6067ff2cbf48e00cdabfbbecbbf5e54ddec63ffb` |
| ROUTE native archive | `ROUTE_selected_ext4_native_float64.npz`; `0f1397e2f5b2e56c6181499338fbd72a17c98b0e28825b79a0b12633cbe99ad9` |
| Selected FLAT export | epoch 002; `9d0ba29d6f9f3ab7df4880092902c7bac25a319741c65f8007b64dccd41e2b5f` |
| Selected ROUTE export | epoch 020; `182d3b28e4c4668df43ebf4edd8305d7b23ff9c771013fa5d6a1d5740b68fa33` |
| Frozen finalizer receipt | `57c214b12c2097aa6bff74451169d092296f473c1d544dad53344288c23e9dcc` |
| Existing same-start/target baseline addendum | `ext4_e8_spint_dev_pooled_addendum_v1.json`; `a10963c3b87081fa7f401eeed1ab9c6829c613e479abe835acc0ff43405649e0`; marked `DEVELOPMENT_COMPARISON_NOT_UNBIASED_HELDOUT` |

Each selected archive contains finite `float64` `prediction` and `target`
arrays of shape `[2069, 2]`, plus `session` and `start`.  The receipt binds
the following session order/counts and target hashes: 519, 490, 425, and 635
rows for `ses-2020-10-30-Run1`, `ses-2020-10-30-Run2`,
`ses-2020-11-18-Run1`, and `ses-2020-11-19-Run1`, respectively.  Its baseline
authority records `same_starts_targets: true` for Original SPINT on each of
those four session surfaces.

There is no materialized Original-SPINT prediction vector at this exact
2,069-row native-archive endpoint.  The available
`source_minival_e8_spint_m30_replay_v1.npz` (`4c01c6cea089b23a43eff27aa10b14d6d410f7d837e2f588121608052ac29541`)
is a different 1,011-row source minival surface, so it was not mixed into any calculation.  The baseline
authority records only its exact-surface Original prediction hashes and R²
values.  Therefore the mean/std/covariance/SSE decompositions below are
candidate-only; an Original side-by-side decomposition would require an
already existing, hash-bound vector on this same endpoint, and no new
evaluation was run to create one.

## Method and arithmetic check

For one session with `n` rows and two native output coordinates, write target
as `y` and candidate prediction as `p`.  Means, standard deviations, and
centered covariance use the population normalization `1/n` independently for
each output coordinate.  The reported total SSE and its exact decomposition
are

`SSE = Σᵢⱼ (pᵢⱼ − yᵢⱼ)²`

`= n Σⱼ (mean(pⱼ) − mean(yⱼ))² + Σᵢⱼ (((pᵢⱼ − mean(pⱼ)) − (yᵢⱼ − mean(yⱼ)))²)`

The first term is labelled **mean-bias SSE**; the second is **centered/dynamic
SSE**.  “Dynamic” means only residual error after removing each series’ own
constant mean; it is not an inference about temporal dynamics or causality.
`bias share = 100 × mean-bias SSE / SSE`.

All calculations read the frozen `float64` arrays.  As a manual numerical
cross-check, each SSE, bias term, and centered term was independently
re-summed element-by-element; the largest absolute identity residual,
`|SSE − bias − centered|`, was at most `3.4e-15` (the displayed values are
rounded).  The R² recomputed as `1 − SSE / Σᵢⱼ(yᵢⱼ − mean(yⱼ))²` reproduces
the receipt values.

Notation in the tables: vectors are `(output 0, output 1)`; `μ`, `σ`, and
`cov` are in native units; `r` is dimensionless.  Scientific notation is used
because the native values are small.

## Moments and centered association

### FLAT selected epoch 002

| Session | n | target μ | target σ | FLAT μ | FLAT σ | cov(FLAT,target) | r |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| 2020-10-30 R1 | 519 | (5.5193e-04, -6.2465e-04) | (1.0953e-02, 9.8430e-03) | (-6.7242e-04, -7.6719e-04) | (2.5206e-03, 2.5987e-03) | (1.1253e-05, 1.7083e-05) | (0.4076, 0.6679) |
| 2020-10-30 R2 | 490 | (-4.8055e-04, 2.6999e-04) | (5.7235e-03, 1.0641e-02) | (-1.0880e-03, -8.7904e-04) | (2.8351e-03, 2.8501e-03) | (8.4769e-06, 1.9218e-05) | (0.5224, 0.6336) |
| 2020-11-18 R1 | 425 | (-4.1844e-04, -4.2337e-04) | (1.2620e-02, 1.3514e-02) | (-8.7270e-04, -2.1968e-05) | (2.7437e-03, 4.3603e-03) | (1.2016e-05, 2.3157e-05) | (0.3470, 0.3930) |
| 2020-11-19 R1 | 635 | (-5.9457e-04, -6.7971e-04) | (1.0082e-02, 7.7792e-03) | (-5.5706e-04, 4.7585e-05) | (2.6879e-03, 3.3832e-03) | (1.4015e-05, -2.4137e-06) | (0.5172, -0.0917) |
| Pooled | 2069 | (-2.4379e-04, -3.8832e-04) | (1.0099e-02, 1.0361e-02) | (-7.7658e-04, -3.9054e-04) | (2.7025e-03, 3.3489e-03) | (1.1618e-05, 1.2757e-05) | (0.4257, 0.3677) |

### ROUTE selected epoch 020

| Session | n | target μ | target σ | ROUTE μ | ROUTE σ | cov(ROUTE,target) | r |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| 2020-10-30 R1 | 519 | (5.5193e-04, -6.2465e-04) | (1.0953e-02, 9.8430e-03) | (5.9142e-04, 1.1094e-04) | (7.9635e-03, 5.1235e-03) | (6.5153e-05, 3.5904e-05) | (0.7469, 0.7119) |
| 2020-10-30 R2 | 490 | (-4.8055e-04, 2.6999e-04) | (5.7235e-03, 1.0641e-02) | (-3.9941e-04, 8.1860e-04) | (6.8730e-03, 7.7748e-03) | (2.4398e-05, 6.6916e-05) | (0.6202, 0.8088) |
| 2020-11-18 R1 | 425 | (-4.1844e-04, -4.2337e-04) | (1.2620e-02, 1.3514e-02) | (1.2382e-03, 1.4564e-03) | (7.8802e-03, 8.1349e-03) | (6.2492e-05, 7.2434e-05) | (0.6284, 0.6589) |
| 2020-11-19 R1 | 635 | (-5.9457e-04, -6.7971e-04) | (1.0082e-02, 7.7792e-03) | (3.2751e-03, 1.9107e-03) | (9.7667e-03, 5.8333e-03) | (4.6377e-05, 9.8628e-06) | (0.4710, 0.2173) |
| Pooled | 2069 | (-2.4379e-04, -3.8832e-04) | (1.0099e-02, 1.0361e-02) | (1.3133e-03, 1.1073e-03) | (8.4341e-03, 6.7401e-03) | (4.8935e-05, 4.2700e-05) | (0.5745, 0.6114) |

## SSE decomposition and recorded same-surface R² context

The Original-SPINT R² column is copied from the receipt-bound baseline
authority, whose `same_starts_targets` flag is true.  It is shown only as
recorded scalar context, not as a newly recomputed or decomposed Original
result.  Candidate R² is recomputed from the selected archive and agrees with
the selected ext4 receipt.

| Candidate / session | Candidate R² | Recorded Original SPINT R² | SSE | Mean-bias SSE | Centered/dynamic SSE | Bias share | Dynamic share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FLAT / 2020-10-30 R1 | 0.193885 | 0.264641 | 9.0730e-02 | 7.8854e-04 | 8.9941e-02 | 0.869% | 99.131% |
| FLAT / 2020-10-30 R2 | 0.257118 | 0.408982 | 5.3145e-02 | 8.2776e-04 | 5.2318e-02 | 1.558% | 98.442% |
| FLAT / 2020-11-18 R1 | 0.127054 | 0.259449 | 1.2684e-01 | 1.5618e-04 | 1.2668e-01 | 0.123% | 99.877% |
| FLAT / 2020-11-19 R1 | 0.024682 | 0.017300 | 1.0043e-01 | 3.3678e-04 | 1.0009e-01 | 0.335% | 99.665% |
| FLAT / pooled | 0.143064 | 0.229198 | 3.7114e-01 | 5.8732e-04 | 3.7056e-01 | 0.158% | 99.842% |
| ROUTE / 2020-10-30 R1 | 0.516009 | 0.264641 | 5.4474e-02 | 2.8164e-04 | 5.4192e-02 | 0.517% | 99.483% |
| ROUTE / 2020-10-30 R2 | 0.511204 | 0.408982 | 3.4968e-02 | 1.5070e-04 | 3.4817e-02 | 0.431% | 99.569% |
| ROUTE / 2020-11-18 R1 | 0.395749 | 0.259449 | 8.7799e-02 | 2.6682e-03 | 8.5130e-02 | 3.039% | 96.961% |
| ROUTE / 2020-11-19 R1 | **-0.238162** | 0.017300 | 1.2749e-01 | 1.3770e-02 | 1.1372e-01 | **10.800%** | **89.200%** |
| ROUTE / pooled | 0.296396 | 0.229198 | 3.0473e-01 | 9.6443e-03 | 2.9509e-01 | 3.165% | 96.835% |

## What the four-session evidence supports—and does not

- The ROUTE failure on 2020-11-19 R1 is not solely a constant output offset:
  even after removing both output means, its centered SSE is `1.1372e-01`,
  89.200% of its total.  Its second output has weak positive centered
  association (`r = 0.2173`), versus 0.6589 on 2020-11-18 R1 and 0.8088 on
  2020-10-30 R2.  This is evidence of poorer centered tracking on that
  session, not proof of why it occurred.
- A mean-bias component is nevertheless present and larger for ROUTE on this
  failing session than on its other three sessions (10.800% versus 0.431%,
  0.517%, and 3.039%).  The ROUTE means are positive for both outputs while
  that session's target means are negative.  This is a descriptive, fixed
  export property; it does not justify a fitted affine correction.
- Relative to FLAT, ROUTE has stronger centered correlations on the first
  three sessions and the better pooled candidate R², while its fourth-session
  result is much worse.  That heterogeneity is why the pooled result cannot
  be used to erase the fourth-session loss, and the fourth session cannot be
  discarded to imply uniform behavior.
- The evidence is insufficient to attribute the failure to the learned route
  state, a particular feature, a session-specific biological/recording cause,
  or any training decision.  No intervention is proposed here.  Any future
  change would need separate authorization and a new frozen-surface
  evaluation; this diagnostic is not an improvement estimate.
