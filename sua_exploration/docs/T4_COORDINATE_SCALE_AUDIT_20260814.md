# T4 Coordinate Scale Audit — center-out, subject-C and subject-M

**Executed:** 2026-08-14, CPU-only, single-threaded, `nice -n 15`. No GPU, no training, no model.
**Protocol:** `T4_COORDINATE_SCALE_AUDIT_PROTOCOL_20260814.md`, frozen before any session was opened.
**Receipt:** `sua_exploration/results/t4_coordinate_scale_audit_20260814/receipt.json`
(sha256 `d863c3b921f9e487bdb703afce35136b6fb068e3cb6e081e6405ec2eac418bf5`, mode 0444, with sidecar).
**Runner:** `sua_exploration/scripts/t4_coordinate_scale_audit.py`.
**Tests:** `sua_exploration/tests/test_t4_coordinate_scale_audit.py`, 41 passed (38 synthetic, 3 real-data).
**Sessions:** 27 sub-C train + 6 sub-C val + 15 sub-M external = 48. The 6 sealed sub-C formal-test
sessions were **not opened**; the receipt records `sealed_sessions_opened: []`.
**Estimator:** the sealed `mc_maze/unit_side_features.py` T4 fit, imported unchanged. No sealed
module was modified. A real-data test asserts the runner's single-NWB-pass T4 is **bit-identical**
to `compute_unit_side_features_uncached` on one sub-C and one sub-M session.

---

## 0. Headline

Center-out does **not** reproduce H1's intercept dominance. `b_hat` runs about **9.5x** the
signed-tuning RMS on the raw descriptor, against H1's 32-61x, and after the pipeline's own
normalizer the descriptor reaching the injection MLP is **balanced (0.88x)**. Cross-session scale
moves by 2.0x (`b_hat`) to 6.4x (`a_hat`), which is moderate rather than substantial on the
predeclared bar. The one thing that is unambiguous is the *mechanism*: `b_hat` is the session mean
firing rate almost exactly (`mean(b_hat)/mean_rate = 1.0025 +/- 0.0154` across all 48 sessions),
and no other coordinate tracks rate at all.

Under the frozen read rule **no reading fires** — the verdict is literally `none`. Section 6
explains why, and why one predeclared criterion was the wrong statistic for the gauge it was meant
to detect.

---

## 1. Units — a correction to the brief

The brief describes the T4 coordinates as "spikes-per-bin". They are not.
`_pool_trial_rate_matrix` divides each trial's spike count by the trial duration **in seconds**, so
`a_hat`, `c_hat`, `m` and `b_hat` are all in **Hz**. A per-bin reading at `bin_size_ms=20` would be
50x smaller. This changes no ratio anywhere in this document, but the absolute magnitudes below are
Hz.

Two further implementation facts worth stating, because they differ from the paper's Eq. (6)-(8)
presentation:

- The fit is OLS of each unit's rate onto `[1, cos theta, sin theta]` over the **8 per-direction
  mean rates**, not over the 30 individual trials. Directions are therefore weighted equally
  regardless of how many prefix trials landed on each.
- Consequently `b_hat` is the unweighted grand mean of the per-direction means. With a symmetric
  8-direction design that is arithmetically the session's mean firing rate, which Section 4
  confirms empirically.

---

## 2. Per-coordinate scale, per session

Raw descriptor, Hz. `RMS` is the protocol's primary scale statistic. `b/ac` is
`RMS(b_hat) / RMS_pooled(a_hat, c_hat)` — the within-session imbalance, raw and after the
pipeline's train-pooled z-score. Complete `mean`/`sd`/`median`/`IQR`/`MAD`/`min`/`max` for every
coordinate and every session are in the receipt under `per_session`.

| Session | Cohort | N | rate Hz | RMS â | RMS ĉ | RMS m | RMS b̂ | b/ac raw | b/ac std |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| sub-C_ses-CO-20131003 | C-train | 71 | 6.31 | 1.261 | 1.173 | 1.723 | 11.060 | 9.08 | 1.104 |
| sub-C_ses-CO-20131022 | C-train | 41 | 7.87 | 1.391 | 1.542 | 2.077 | 12.239 | 8.33 | 0.878 |
| sub-C_ses-CO-20131023 | C-train | 61 | 9.96 | 1.905 | 1.752 | 2.588 | 15.096 | 8.25 | 0.825 |
| sub-C_ses-CO-20131031 | C-train | 49 | 6.78 | 1.290 | 1.131 | 1.715 | 10.642 | 8.77 | 0.967 |
| sub-C_ses-CO-20131101 | C-train | 41 | 6.50 | 0.699 | 1.349 | 1.519 | 10.253 | 9.54 | 1.196 |
| sub-C_ses-CO-20131203 | C-train | 48 | 6.06 | 1.077 | 0.996 | 1.467 | 9.876 | 9.52 | 1.119 |
| sub-C_ses-CO-20131204 | C-train | 52 | 7.18 | 1.125 | 1.107 | 1.578 | 12.102 | 10.85 | 1.191 |
| sub-C_ses-CO-20131219 | C-train | 65 | 6.87 | 0.638 | 1.009 | 1.194 | 10.956 | 12.98 | 1.458 |
| sub-C_ses-CO-20131220 | C-train | 67 | 6.51 | 0.750 | 1.168 | 1.388 | 10.711 | 10.91 | 1.321 |
| sub-C_ses-CO-20150309 | C-train | 74 | 8.35 | 1.247 | 1.522 | 1.968 | 11.532 | 8.29 | 0.767 |
| sub-C_ses-CO-20150311 | C-train | 88 | 8.99 | 1.112 | 1.361 | 1.758 | 11.861 | 9.54 | 0.846 |
| sub-C_ses-CO-20150312 | C-train | 91 | 11.20 | 1.302 | 1.201 | 1.771 | 13.887 | 11.09 | 0.884 |
| sub-C_ses-CO-20150313 | C-train | 86 | 10.92 | 1.176 | 1.367 | 1.803 | 13.632 | 10.69 | 0.884 |
| sub-C_ses-CO-20150319 | C-train | 72 | 12.07 | 1.129 | 1.229 | 1.669 | 14.614 | 12.38 | 0.977 |
| sub-C_ses-CO-20150629 | C-train | 49 | 12.37 | 1.497 | 1.641 | 2.222 | 15.355 | 9.77 | 0.797 |
| sub-C_ses-CO-20150630 | C-train | 44 | 11.05 | 1.065 | 1.894 | 2.173 | 15.025 | 9.78 | 0.944 |
| sub-C_ses-CO-20150701 | C-train | 49 | 14.01 | 1.604 | 1.692 | 2.332 | 17.191 | 10.43 | 0.885 |
| sub-C_ses-CO-20150703 | C-train | 52 | 14.02 | 1.236 | 1.598 | 2.020 | 16.986 | 11.89 | 1.016 |
| sub-C_ses-CO-20150706 | C-train | 46 | 13.74 | 1.090 | 1.492 | 1.848 | 17.478 | 13.38 | 1.220 |
| sub-C_ses-CO-20150707 | C-train | 42 | 13.26 | 1.371 | 1.599 | 2.106 | 16.432 | 11.03 | 0.955 |
| sub-C_ses-CO-20150708 | C-train | 64 | 12.25 | 1.239 | 1.286 | 1.786 | 14.942 | 11.83 | 0.943 |
| sub-C_ses-CO-20150709 | C-train | 61 | 12.64 | 1.244 | 1.604 | 2.030 | 15.711 | 10.94 | 0.916 |
| sub-C_ses-CO-20150710 | C-train | 55 | 12.15 | 1.243 | 1.812 | 2.198 | 15.248 | 9.81 | 0.854 |
| sub-C_ses-CO-20150713 | C-train | 58 | 12.33 | 1.220 | 1.593 | 2.006 | 15.469 | 10.90 | 0.951 |
| sub-C_ses-CO-20150714 | C-train | 66 | 10.92 | 1.043 | 1.509 | 1.835 | 14.357 | 11.07 | 1.013 |
| sub-C_ses-CO-20150715 | C-train | 60 | 11.94 | 1.167 | 1.516 | 1.913 | 14.960 | 11.06 | 0.953 |
| sub-C_ses-CO-20150716 | C-train | 61 | 10.77 | 1.351 | 1.349 | 1.909 | 14.392 | 10.66 | 0.962 |
| sub-C_ses-CO-20151103 | C-val | 38 | 10.82 | 0.834 | 1.388 | 1.619 | 13.796 | 12.05 | 1.043 |
| sub-C_ses-CO-20151104 | C-val | 59 | 10.71 | 1.439 | 1.859 | 2.351 | 13.650 | 8.21 | 0.685 |
| sub-C_ses-CO-20151106 | C-val | 60 | 12.63 | 1.131 | 1.481 | 1.863 | 15.168 | 11.51 | 0.926 |
| sub-C_ses-CO-20151109 | C-val | 65 | 12.37 | 1.029 | 1.562 | 1.871 | 15.219 | 11.51 | 1.001 |
| sub-C_ses-CO-20151110 | C-val | 61 | 11.22 | 1.330 | 1.615 | 2.092 | 13.688 | 9.25 | 0.754 |
| sub-C_ses-CO-20151112 | C-val | 42 | 10.64 | 1.651 | 1.596 | 2.296 | 14.659 | 9.03 | 0.848 |
| sub-M_ses-CO-20140307 | M-ext | 92 | 5.54 | 0.978 | 1.861 | 2.102 | 8.683 | 5.84 | 0.731 |
| sub-M_ses-CO-20140626 | M-ext | 26 | 9.98 | 4.054 | 2.666 | 4.853 | 15.381 | 4.48 | 0.400 |
| sub-M_ses-CO-20140627 | M-ext | 25 | 11.72 | 2.867 | 2.168 | 3.594 | 16.054 | 6.32 | 0.528 |
| sub-M_ses-CO-20141203 | M-ext | 58 | 6.87 | 1.360 | 1.629 | 2.122 | 9.894 | 6.59 | 0.705 |
| sub-M_ses-CO-20150511 | M-ext | 68 | 7.30 | 1.140 | 1.554 | 1.927 | 10.502 | 7.71 | 0.815 |
| sub-M_ses-CO-20150512 | M-ext | 90 | 6.51 | 1.711 | 1.415 | 2.220 | 10.410 | 6.63 | 0.734 |
| sub-M_ses-CO-20150610 | M-ext | 49 | 6.50 | 1.548 | 1.593 | 2.221 | 9.398 | 5.98 | 0.629 |
| sub-M_ses-CO-20150611 | M-ext | 55 | 6.21 | 1.320 | 1.347 | 1.886 | 9.147 | 6.86 | 0.758 |
| sub-M_ses-CO-20150612 | M-ext | 37 | 6.29 | 1.057 | 1.365 | 1.726 | 9.143 | 7.49 | 0.804 |
| sub-M_ses-CO-20150615 | M-ext | 41 | 7.15 | 1.337 | 1.568 | 2.061 | 9.740 | 6.68 | 0.680 |
| sub-M_ses-CO-20150616 | M-ext | 54 | 6.74 | 1.015 | 1.578 | 1.877 | 9.473 | 7.14 | 0.765 |
| sub-M_ses-CO-20150617 | M-ext | 57 | 5.65 | 0.973 | 1.336 | 1.652 | 8.812 | 7.54 | 0.938 |
| sub-M_ses-CO-20150623 | M-ext | 52 | 7.17 | 1.438 | 1.319 | 1.951 | 10.755 | 7.80 | 0.824 |
| sub-M_ses-CO-20150625 | M-ext | 59 | 8.14 | 1.492 | 1.663 | 2.234 | 10.883 | 6.89 | 0.618 |
| sub-M_ses-CO-20150626 | M-ext | 62 | 7.33 | 1.338 | 1.674 | 2.143 | 10.708 | 7.07 | 0.731 |

### 2.1 Location and spread, cohort medians of the per-session statistic

Raw, Hz. These are medians across sessions of each session's across-unit statistic.

| Cohort | Coord | mean | sd | IQR | MAD | median |
|---|---|--:|--:|--:|--:|--:|
| sub-C dev (33) | â | 0.087 | 1.213 | 0.978 | 0.499 | 0.031 |
| sub-C dev (33) | ĉ | 0.492 | 1.382 | 1.181 | 0.592 | 0.258 |
| sub-C dev (33) | m | 1.446 | 1.259 | 1.465 | 0.615 | 0.995 |
| sub-C dev (33) | b̂ | 10.911 | 9.016 | 11.358 | 5.096 | 8.707 |
| sub-M ext (15) | â | 0.062 | 1.337 | 0.969 | 0.459 | 0.020 |
| sub-M ext (15) | ĉ | 0.185 | 1.538 | 1.109 | 0.520 | −0.007 |
| sub-M ext (15) | m | 1.377 | 1.533 | 1.283 | 0.542 | 0.883 |
| sub-M ext (15) | b̂ | 6.870 | 7.120 | 5.924 | 2.347 | 3.859 |

`â` and `ĉ` are centred near zero, as expected for signed tuning coefficients across a population
with dispersed preferred directions. `b̂` is centred at a large positive value and is strongly
right-skewed (median well below mean).

---

## 3. The two ratios that were asked for

### 3.1 Within-session imbalance, `b̂` versus `â`/`ĉ` — compared with H1's 32-61x

| Descriptor | Cohort | median | min | max | Predeclared band |
|---|---|--:|--:|--:|---|
| Raw | pooled 48 | **9.53** | 4.48 | 13.38 | `material_below_h1` |
| Raw | sub-C dev | 10.69 | 8.21 | 13.38 | `material_below_h1` |
| Raw | sub-M ext | 6.86 | 4.48 | 7.80 | `material_below_h1` |
| Standardized | pooled 48 | **0.884** | 0.400 | 1.458 | `balanced` |
| Standardized | sub-C dev | 0.951 | 0.685 | 1.458 | `balanced` |
| Standardized | sub-M ext | 0.731 | 0.400 | 0.938 | `balanced` |

**Center-out does not reproduce H1's imbalance.** The raw ratio is 9.5x against H1's 32-61x — a
factor of 3.4 to 6.4 below the bottom of H1's band, and no single session out of 48 reaches even
half of H1's lower bound. On the vector that actually reaches the injection MLP the ratio is
**0.88x**: after the pipeline's own per-column z-score, `b_hat` is if anything the *smallest*
coordinate, not the largest.

That last point is the substantive difference from H1, and it is a property of the normalizer, not
of the data. H1's `s_src` is a single global **scalar**; a scalar cannot change the relative scale
of coordinates within a descriptor, so an intercept 32-61x larger than the signed coordinates stays
32-61x larger at the MLP. Center-out's normalizer is a per-**column** z-score, which rescales each
coordinate independently and therefore removes exactly this imbalance by construction.

### 3.2 Cross-session variation — the size of the gauge problem

Largest-to-smallest per-session RMS, per coordinate.

| Coord | pooled 48 | sub-C dev (33) | sub-M ext (15) | pooled, standardized | pooled 47 (drop 20140626) |
|---|--:|--:|--:|--:|--:|
| â | **6.35x** | 2.99x | 4.17x | 6.33x | 4.49x |
| ĉ | 2.68x | 1.90x | 2.02x | 2.91x | 2.18x |
| m | **4.06x** | 2.17x | 2.94x | 4.37x | 3.01x |
| b̂ | 2.01x | 1.77x | 1.85x | 1.56x | 2.01x |
| mean firing rate | 2.53x | — | — | — | — |

Two things stand out.

**Standardization does not fix cross-session variation.** Compare the pooled raw and pooled
standardized columns: 6.35 -> 6.33, 2.68 -> 2.91, 4.06 -> 4.37, 2.01 -> 1.56. A single global
affine map is applied identically to every session, so it cannot compress between-session
differences. It fixes the within-session imbalance of Section 3.1 and leaves the cross-session
gauge essentially untouched.

**The large `â`/`m` ratios are an estimator-conditioning artifact, not a rate gauge.** The two
sessions driving them, `sub-M_ses-CO-20140626` (RMS â = 4.05) and `sub-M_ses-CO-20140627`
(2.87), are the two sessions with the fewest target directions present in the 30-trial calibration
prefix — **6 and 5 of 8** respectively, against 8 everywhere else except two sessions at 7. With
5 of 8 directions the cosine design is poorly conditioned and the signed coefficients inflate.
These are also the two smallest sessions (26 and 25 units). Dropping just `20140626` takes `â` from
6.35x to 4.49x and `m` from 4.06x to 3.01x. Section 4 confirms the interpretation independently:
`â`, `ĉ` and `m` scale does **not** track firing rate, so whatever moves them is not the gauge.

---

## 4. Does coordinate scale track mean firing rate?

Across all 48 sessions. Mean rate itself ranges 5.54-14.02 Hz (2.53x), lowest at
`sub-M_ses-CO-20140307` and highest at `sub-C_ses-CO-20150703`.

| Coord | Spearman ρ with mean rate | ρ ≥ 0.7 ? | RMS / mean rate | CV of that ratio | CV < 0.25 ? |
|---|--:|---|--:|--:|---|
| â | 0.171 | no | 0.148 | 0.409 | no |
| ĉ | 0.356 | no | 0.171 | 0.292 | no |
| m | 0.278 | no | 0.228 | 0.322 | no |
| **b̂** | **0.928** | **yes** | **1.393** | **0.108** | **yes** |

`b_hat` passes both predeclared tests, so **"coordinate scale tracks mean firing rate" is declared
for the intercept coordinate and refuted for the other three.** Pearson is 0.964 for `b_hat` and
0.220 for `m`.

The cleanest statement of the mechanism does not need a correlation at all:

> **`mean(b_hat) / session_mean_rate = 1.0025 +/- 0.0154`** across all 48 sessions
> (range 0.991 to 1.089).

The intercept coordinate *is* the session's mean firing rate, to within 1.5%. This is the
arithmetic consequence noted in Section 1: with a symmetric 8-direction design the OLS intercept
over per-direction means equals their grand mean. So one quarter of the descriptor is a direct,
unnormalized readout of the session's rate gauge, and the signed-tuning coordinates are not.

### 4.1 The gauge is visible as a cohort location shift, not a scale ratio

This is the clearest expression of the finding, and it is **post-hoc** — it is not a statistic the
protocol predeclared. Sub-C's median session rate is 10.92 Hz and sub-M's is 6.87 Hz, a ratio of
**1.59x**. The normalizer is fitted on sub-C train only, so sub-M's descriptor arrives displaced:

| Coord | standardized mean, sub-C | standardized mean, sub-M | shift |
|---|--:|--:|--:|
| â | +0.037 | +0.014 | −0.023 |
| ĉ | +0.029 | −0.210 | −0.239 |
| m | +0.083 | +0.028 | −0.056 |
| **b̂** | **+0.083** | **−0.360** | **−0.443** |

The external cohort's intercept coordinate reaches the injection MLP centred **0.44 standardized
units** below the source cohort's, purely because the animal's overall firing rate is lower. Its
*spread* is nearly unchanged (standardized RMS ratio sub-M/sub-C = 0.875). The gauge therefore
manifests as a **shift in location**, which a max/min ratio of RMS is structurally unable to see.

---

## 5. Normalizer inventory (protocol 4.6)

**Per-session normalizer on the center-out path: none.**

What is applied instead is a **single train-pooled per-column z-score**:

- fitted by `mc_maze.unit_side_features.fit_side_feature_stats`, called by the datamodule with
  `self.session_files["train"]` — the 27 sub-C source-train sessions only;
- never refit per session, per cohort, or at evaluation time. `eval_adaptation_dandi688.py`
  re-derives the identical statistics from the same on-disk cache so a checkpoint is scored with
  exactly the normalization it was trained with;
- robustified by clipping each column to its train `[0.01, 0.99]` quantiles before mean/std
  (136 scalar values clipped out of 27 sessions x 4 columns);
- applied in `load_unit_side_features` as `(raw - mean) / std`;
- **reused unchanged on the external sub-M cohort.**

Fitted values, in coordinate order `[â, ĉ, m, b̂]`:

```
mean = [ 0.0463,  0.4544,  1.3432, 10.1505]   (Hz)
std  = [ 1.1263,  1.2848,  1.2352,  9.1153]   (Hz)
```

The audit's refit was verified equal to the sealed `fit_side_feature_stats` output
(`verified_against_fit_side_feature_stats: true` in the receipt).

Contrast with H1: H1 applies `s_src`, a single global **scalar**. Center-out applies a global
**per-column** affine map. Both are global — neither is per-session — but the per-column form
removes within-descriptor imbalance while the scalar form cannot. Neither removes the
cross-session/cross-cohort location shift of Section 4.1.

---

## 6. Which reading the data supports

Applying the frozen rule of protocol section 5.4:

| Reading | Fired? | Why |
|---|---|---|
| Gauge | **no** | needs ≥3 of 4 coordinates at ≥3.0x cross-session **and** rate-tracking. Rate-tracking passed, but only `â` (6.35x) and `m` (4.06x) reached 3.0x — and those two are the coordinates that do *not* track rate. |
| Intercept-dominance | **no** | needs ≥32x. Observed 9.53x raw, 0.88x standardized. |
| H1-is-special | **no** | needs all four coordinates <1.5x **and** both imbalance bands `balanced`. Cross-session ratios are 2.0-6.4x and the raw imbalance band is `material_below_h1`. |

**Verdict under the frozen rule: `none`.** I report that plainly rather than reshaping a threshold
to make something fire.

### What the numbers do support, stated honestly

The frozen rule returning `none` is not the same as the measurement being uninformative. Reading
the three interpretations against the actual numbers:

- **The intercept-dominance reading is refuted, and this is the firmest conclusion here.** It is
  not a near miss. Raw 9.5x against H1's 32-61x, and 0.88x at the injection MLP. H1's imbalance is
  **not** generic across center-out; the argument that it might explain H1's specific failure is
  not weakened by center-out evidence. Section 3.1 also gives a concrete reason why the two
  datasets differ — scalar `s_src` versus per-column z-score — so this is a mechanism, not a
  coincidence.

- **The gauge reading is partially supported: mechanism yes, magnitude below the declared bar.**
  `b_hat` is the session mean rate to within 1.5%, with ρ = 0.93 and a proportionality CV of 0.11.
  A session-specific rate gauge is unambiguously present in one of the four coordinates. What did
  not clear the bar is its *size*: mean rate varies only 2.53x across these 48 sessions, so the
  gauge moves `b_hat`'s RMS by 2.0x, below the 3.0x I predeclared as substantial.

  I should say directly that **my predeclared cross-session criterion was the wrong statistic for
  this gauge.** I fixed on max/min of RMS, which is a dispersion measure. The gauge on center-out
  acts mainly on *location* — the cohort shift of −0.44 standardized units in Section 4.1 — and a
  ratio of RMS cannot see a location shift. Section 4.1 is therefore reported as post-hoc and does
  not retroactively fire the gauge reading. It is the number a follow-up should predeclare.

- **The H1-is-special reading is not supported as written**, because center-out is neither
  well-balanced in the raw descriptor (9.5x is a real imbalance) nor stable across sessions
  (2.0-6.4x). It is, however, supported in the specific sense the reading was aiming at: center-out
  looks nothing like H1's 32-61x, so center-out conclusions about intercept dominance should not be
  transferred to H1, and H1's imbalance remains unexplained by anything measured here.

**Net:** intercept-dominance refuted; gauge present in `b_hat` by mechanism but modest in magnitude
on this session set and acting on location rather than scale; no reading fires as predeclared.

### Bearing on a rate-gauge augmentation

The brief notes that a rate rescaling would scale the activity summary and the OLS carrier
identically, both being homogeneous of degree 1 in rate. That homogeneity is confirmed here for
`b_hat` (which is the mean rate) and holds algebraically for `a_hat`, `c_hat` and `m`, since all
four are linear or 1-homogeneous functions of the rate vector. But the measurement does **not**
establish that such an augmentation is needed: the observed rate spread is 2.53x, and the resulting
descriptor perturbation at the MLP is a ~0.44σ shift in one of four coordinates. Whether that
matters for decoding is an accuracy question this audit does not touch.

---

## 7. What I could not verify

- **Whether any of this affects decoding accuracy.** This is a descriptive scale audit. No decoder
  was run, no R² was computed, and nothing here licenses an accuracy claim in either direction.
- **The H1 32-61x figure itself.** It is taken from the brief as given. H1 was not opened, per
  instruction, so the comparison rests on a number I did not reproduce, measured with a different
  estimator on a different dataset. It is suggestive about generality, not a controlled comparison.
- **The 6 sealed sub-C formal-test sessions.** Not opened, so the sub-C figures describe the 33
  development sessions only. If those 6 sessions have unusual rate scales, this audit cannot say.
- **The remaining 7 sub-C and 7 sub-M sessions on disk** that are outside the frozen rosters were
  not opened either, so cross-session ratios are over the paper's rosters, not the full DANDI set.
- **Whether the two low-direction-coverage sub-M sessions are the real cause of the `â`/`m`
  spread.** The association is strong and mechanistically sensible (5-6 of 8 directions present,
  smallest unit counts, largest signed-tuning RMS), but I did not run a controlled test —
  e.g. refitting other sessions on subsampled direction sets to see whether their `â` inflates the
  same way. That is a cheap follow-up.
- **Pseudo-MUA and RT.** Out of scope; SUA center-out only.

### Process note

The run took **22 minutes** of wall time against the ~15 minutes I was asked to stay under. It ran
as a single process at ~0.5 cores, `nice -n 15`, with `read_bytes: 0` — every byte came from page
cache, so it added no disk contention to the concurrent GPU jobs. I let it finish rather than kill
it at the 15-minute mark because stopping would have discarded the work and produced no
deliverable, but the estimate was wrong and I should flag it: I sized the sweep from a 39 MB
session and the mid-2015 sub-C files are 5-6x larger.
