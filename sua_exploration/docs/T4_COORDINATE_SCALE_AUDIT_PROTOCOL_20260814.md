# Frozen Protocol: T4 Coordinate Scale Audit (center-out)

**Date predeclared:** 2026-08-14, before any session was opened or any statistic computed.
**Status:** CPU-only, single-threaded, niced. No GPU. No training. No model instantiation.
**Scope:** DANDI 000688 center-out, SUA view only — subject-C development sessions and the frozen
15-session subject-M external roster. **Not** H1, **not** M2, **not** RT, **not** pseudo-MUA.
The paper is not edited by this work.

---

## 1. The question

`T4_i = [a_hat_i, c_hat_i, m_i, b_hat_i]` is the center-out per-unit encoding signature. Three of
its four coordinates (`a_hat`, `c_hat`, `b_hat`, and by construction `m = hypot(a_hat, c_hat)`)
carry session-specific firing-rate units. Only the ratio `atan2(c_hat, a_hat)` is scale-free. The
descriptor is concatenated with an activity summary and mapped to the identity token by an MLP.

**Nobody has measured the numerical scale of these four coordinates on center-out.** A separate
finding on H1 (7-DoF, a different, population-level estimator) measured the intercept coordinate
arriving at the injection MLP at **32-61x** the RMS of the signed-tuning coordinates. This protocol
fixes, in advance, what would count as center-out reproducing that imbalance, and what would count
as a substantial cross-session gauge problem.

---

## 2. Estimator — sealed, reused, not reimplemented

The measurement consumes the existing sealed center-out T4 estimator at
`sua_exploration/mc_maze/unit_side_features.py`. **No sealed module is modified.** The audit imports:

| Symbol | Role |
|---|---|
| `compute_unit_side_features_uncached(..., feature_group="t4")` | authoritative raw T4, `[units, 4]` |
| `list_datamodule_rewarded_trials` | calibration-prefix trial selection |
| `_pool_trial_rate_matrix` | per-unit, per-pool-trial firing rate `[units, trials]` |
| `_unit_tuning_features` | per-unit cosine fit |
| `_nearest_canonical_direction_index`, `CANONICAL_DIRECTIONS_RAD` | canonical direction snapping |
| `_fit_robust_stats` | the train-only z-score statistics used by `fit_side_feature_stats` |

The runner takes a **single NWB pass per session** built from `list_datamodule_rewarded_trials` +
`_pool_trial_rate_matrix` + `_unit_tuning_features`, because the mean-firing-rate statistic
(Section 4.4) needs the same rate matrix the T4 fit consumes. **A required test asserts this
single-pass T4 is bit-identical to the sealed public entry point
`compute_unit_side_features_uncached` on at least two real sessions.** If that test fails the audit
is void.

Frozen estimator settings, taken from the A2 contract
(`a2_matched_subject_shift_v2_core.py`): `pool_size=30` rewarded trials, `bin_size_ms=20`,
`window_size=50`, `trial_result_filter="R"`, `signal_view="sua"`.

**Units.** `_pool_trial_rate_matrix` divides a trial's spike count by the trial duration in seconds,
so every T4 coordinate is in **Hz (spikes per second)**, not spikes per bin. The audit reports Hz
and records this explicitly, since a per-bin reading would differ by a fixed factor of 50 and would
not change any ratio reported here.

---

## 3. Sessions

| Cohort | Roster | n |
|---|---|---:|
| subject-C source train | `configs/subc_co_27_6_strict_train_val_manifest.json` `train` | 27 |
| subject-C development validation | same manifest, `val` | 6 |
| subject-M external | `mc_maze/gpu_contract_common.py` `SUBM_EXTERNAL_SESSIONS` | 15 |

**The 6 subject-C formal-test sessions stay sealed and are never opened.** Their names are recorded
in the receipt as excluded. Any attempt to resolve them to a path is a protocol violation.

Per-session values are reported for all 48 openable sessions. Aggregates never replace them.

---

## 4. What is measured

Everything is computed **per session, across that session's units**, on the **raw** (pre-normalizer)
T4 unless explicitly labelled "standardized".

### 4.1 Per-coordinate scale
For each of `a_hat`, `c_hat`, `m`, `b_hat`: RMS, mean, sd (ddof=0), median, IQR (q75-q25), and MAD
(median absolute deviation about the median, unscaled). Unit count reported alongside.

**Primary scale statistic: RMS.** It is the quantity that determines the magnitude a coordinate
presents at the injection MLP's first layer, and it is the statistic the H1 finding used.

### 4.2 Within-session imbalance
`imbalance_b_over_ac = RMS(b_hat) / RMS_pooled(a_hat, c_hat)`, where the pooled signed-tuning RMS
is taken over the concatenation of the `a_hat` and `c_hat` columns (2N values). Also reported:
`RMS(b_hat)/RMS(a_hat)`, `RMS(b_hat)/RMS(c_hat)`, and `RMS(m)/RMS_pooled(a_hat, c_hat)`.

### 4.3 Cross-session variation
For each coordinate, `max_over_sessions(RMS) / min_over_sessions(RMS)`, reported three ways:
within subject-C (33 sessions), within subject-M (15 sessions), and pooled over all 48.

### 4.4 Firing-rate scale
`session_mean_rate_hz` = mean over units of that unit's mean firing rate across the 30 pool trials,
computed from the same rate matrix the T4 fit consumes. This is deliberately **not** `b_hat`, so
the question "does coordinate scale track firing rate" is not circular.

### 4.5 Standardized descriptor
The center-out path's normalizer is applied by `fit_side_feature_stats`, which the datamodule calls
with **train sessions only**. The audit refits it with the sealed `_fit_robust_stats` on the pooled
27 train sessions' raw T4 and reports the same Section 4.1-4.3 statistics for the standardized
descriptor `(T4 - mean) / std`. This is the vector that actually reaches the injection MLP and is
therefore the correct surface for comparison with H1's "arriving at the injection MLP" figure. Both
raw and standardized numbers are reported; neither is suppressed.

### 4.6 Normalizer inventory
Report, from source, every normalization applied to T4 on the center-out path, and whether any of
it is per-session. Report "none" if none is found.

---

## 5. Predeclared read rule

These thresholds are fixed now and are not revised after seeing any number.

### 5.1 Cross-session variation — the gauge question
Criterion: per-coordinate `max/min` of session RMS, pooled over all 48 sessions.

| Band | Verdict |
|---|---|
| `< 1.5x` | negligible — no gauge problem |
| `1.5x - 3.0x` | moderate |
| `>= 3.0x` | **substantial** — session-specific gauge confirmed |

"Substantial cross-session variation" is declared only if **at least three of the four coordinates**
reach `>= 3.0x` pooled.

### 5.2 Within-session intercept dominance — the H1 comparison
Criterion: median over sessions of `imbalance_b_over_ac` (Section 4.2), evaluated separately on the
raw and the standardized descriptor. H1's reference band is **32-61x**.

| Band | Verdict |
|---|---|
| `>= 32x` | reproduces H1's imbalance; H1 is generic |
| `3x - 32x` | materially imbalanced but well below H1 |
| `< 3x` | balanced; H1's imbalance is not reproduced |

### 5.3 Does scale track firing rate
Two predeclared tests, both across sessions, computed pooled over all 48 sessions:
1. **Rank association.** Spearman rho between `session_mean_rate_hz` and per-session RMS of each
   coordinate. `rho >= 0.7` counts as "tracks".
2. **Proportionality.** Coefficient of variation (sd/mean) of the ratio
   `RMS(coordinate) / session_mean_rate_hz` across sessions. `CV < 0.25` counts as "proportional",
   i.e. the coordinate scale is essentially mean rate times a session-independent constant.

"Coordinate scale tracks mean firing rate" is declared only if **both** tests pass for `b_hat`.

### 5.4 Which reading the data supports
Exactly one of the following is selected, or "none", using only the criteria above.

- **Gauge reading** — 5.1 substantial **and** 5.3 declared. Licenses: the descriptor carries a
  session-specific rate gauge, and a source-training augmentation that randomizes the rate gauge is
  motivated. Note for that augmentation: a rate rescaling scales the activity summary and the OLS
  carrier identically, since both are homogeneous of degree 1 in rate.
- **Intercept-dominance reading** — 5.2 lands in the `>= 32x` band. Licenses: H1's imbalance is
  generic, which weakens the argument that it explains H1's specific failure.
- **H1-is-special reading** — 5.1 negligible **and** 5.2 `< 3x`. Licenses: H1's imbalance is a
  property of that dataset or that estimator; center-out conclusions do not transfer to it.

These are not mutually exclusive by construction. If more than one fires, all are reported with the
numbers that fired them, and no single verdict is manufactured. If none fires, "none" is reported.

---

## 6. Integrity requirements

- **Immutable JSON receipt** with: SHA-256 of every input NWB, SHA-256 of every sealed
  implementation file imported, the full session roster, the excluded sealed roster, the frozen
  estimator settings, a Python/NumPy/environment fingerprint, and every per-session value. Written
  with `O_EXCL` and chmod 0444, plus a `.sha256` sidecar.
- **Per-session values reported**, not only aggregates.
- **Tests pass before the audit runs**, including the sealed-equivalence test of Section 2.
- **Resource cap.** Every process runs `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=` under `nice -n 15`. The machine is concurrently
  running other users' GPU training with ~30 of 32 cores committed.

---

## 7. Interpretation limits fixed in advance

- This is a **descriptive scale audit of the descriptor**. It measures no decoding accuracy and
  licenses no accuracy claim. A gauge finding motivates an augmentation; it does not demonstrate
  that the augmentation helps.
- The H1 comparison is across **different datasets and different estimators** (per-unit OLS here,
  a population-level ridge-and-shrinkage carrier there). A matching or non-matching ratio is
  suggestive about generality, not a controlled comparison.
- `m = hypot(a_hat, c_hat)` is a deterministic function of the other two coordinates, so its scale
  statistics are not independent evidence.
- Sessions are unweighted. A session with 25 units counts the same as one with 92.
