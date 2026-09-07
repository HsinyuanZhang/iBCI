# Decoder error-structure decomposition protocol

**Frozen:** 2026-08-12 (Asia/Hong_Kong)  
**Status:** pre-registered measurement scaffold. Authorizes forward-only CPU scoring on
development/validation sessions only. **Does not authorize GPU runs, training, or opening
sealed formal-test sessions.**  
**Parent item:** `HANDOFF_NEXT_ROUND_DIRECTIONS_20260812.md` item **A13**.  
**House style reference:** `SUA_T4_FACTORIZED_LOGIT_RESIDUAL_PROTOCOL.md`.

---

## 1. Scientific claim under test (precise, not overclaimed)

A cosine-tuning-derived carrier makes a **falsifiable, stratum-specific** prediction: its
benefit should **concentrate** where the carrier is informative (directional phase present,
strong per-unit tuning, poor calibration-direction coverage) and should **vanish** where the
theory says it is not (flat tuning, well-conditioned full direction coverage, speed/magnitude
strata with no phase information). The program has only ever reported scalar session-level
R²; **nobody has checked whether measured gain lands on the strata the theory predicts.**

This protocol does **not** claim the carrier improves aggregate R². It tests **where** any
paired carrier-minus-control improvement appears.

**Concrete anomalies this tool must be able to interrogate (post hoc, not tuned into strata):**

- H1's two recordings (`0.598179` vs `0.314797` under one model): 1.9× spread with no
  published explanation.
- Negative A2b `K=8/16` grand means driven by a single recording (`sub-M_ses-CO-20150512`).

---

## 2. Frozen stratification variables (30 pre-declared strata)

Strata are defined **before** any decomposition receipt exists. Edges and labels are constants
in `mc_maze/decoder_error_structure.py`. **No stratum edge may be recomputed from the windows
being scored.** Post-hoc stratum selection is forbidden.

| Axis | # strata | Labels | Assignment rule |
|---|---:|---|---|
| `target_direction` | 8 | `dir_0`…`dir_7` | Nearest of `CANONICAL_DIRECTIONS_RAD` to window target velocity angle (`arctan2(vy,vx)`). For datasets without native `target_dir`, derive from the scored target vector and record `direction_source: derived_from_target_velocity` in the receipt. |
| `movement_speed` | 4 | `speed_q1`…`speed_q4` | Fixed norm edges on standardized target velocity L2 norm: `[0, 0.35), [0.35, 0.70), [0.70, 1.05), [1.05, ∞)`. |
| `within_trial_phase` | 3 | `phase_early`, `phase_middle`, `phase_late` | Equal thirds of relative bin index within the scored trial window. |
| `active_tuning_strength` | 3 | `tuning_low`, `tuning_mid`, `tuning_high` | Activity-weighted mean unit modulation depth `m_i` with fixed edges `[0, 0.05), [0.05, 0.15), [0.15, ∞)` Hz. |
| `calibration_design_coverage` | 12 | `design_dirs_{2,3_4,5_6,7_8} × design_cond_{rank_deficient,finite_low,finite_high}` | Session-level only: distinct calibration-pool direction count (from `_compute_tuning_features_uncached`) crossed with design rank/condition (`rank<3` vs `rank=3 & cond<10` vs `rank=3 & cond≥10`). |

**Total pre-registered strata:** 8 + 4 + 3 + 3 + 12 = **30** (five independent axes; axes are
**not** Cartesian-producted).

### 2.1 Directional predictions (carrier hypothesis)

| Stratum axis | Strata expected **large** carrier−control gain (Δ primary < 0 for MSE) | Strata expected **no** gain (Δ ≈ 0) |
|---|---|---|
| `target_direction` | Directions with high population tuning heterogeneity and phase ambiguity under activity-only pooling (empirically: directions where calibration coverage is thin — see `calibration_design_coverage`) | None predicted uniformly zero; **uniform gain across all 8 directions falsifies directional specificity** |
| `movement_speed` | Low-to-mid speed quartiles where directional phase matters for decoding transients | Extreme `speed_q4` if errors are dominated by magnitude scaling already captured by activity |
| `within_trial_phase` | `phase_early` and `phase_middle` (movement initiation / acceleration) | `phase_late` near movement offset if velocity transients have decayed |
| `active_tuning_strength` | `tuning_high` (large `m_i` in active units) | `tuning_low` (flat units — carrier phase content uninformative) |
| `calibration_design_coverage` | `design_dirs_2` / `design_dirs_3_4` with `design_cond_rank_deficient` or `design_cond_finite_high` | `design_dirs_7_8` with `design_cond_finite_low` (well-conditioned full direction set) |

**Primary falsifier:** if paired Δ primary is **statistically indistinguishable and numerically
uniform** across all strata on an axis where the table above predicts heterogeneity (especially
`active_tuning_strength` and `calibration_design_coverage`), the mechanism story in
`HANDOFF_NEXT_ROUND_DIRECTIONS_20260812.md` §1.3 requires qualification. A **uniform gain
across all 30 strata** falsifies stratum-specific carrier benefit.

---

## 3. Scoring statistics

### 3.1 Paired windows

Carrier and control arms must be evaluated on **identical** query windows. Each window identity
is `(session_name, trial_index, bin_index)`; SHA-256 hashes are stored in the receipt. The
aggregator's single most important check is **hash equality** across paired receipts.

### 3.2 Primary vs secondary scores

Conditioning on a stratum changes the target variance denominator, so **within-stratum R² is not
comparable across strata or to global session R².**

| Statistic | Definition | Role |
|---|---|---|
| **`common_denominator_mse`** (primary) | Per-window MSE normalized by SS_tot around the **global** query-set target mean (same denominator for every stratum) | Cross-stratum comparability; paired Δ = carrier − control (**negative** = carrier better) |
| `within_stratum_r2` (secondary) | Variance-weighted R² inside the stratum only | Local fit quality; not used for cross-stratum ranking |

### 3.3 Per-stratum receipt fields

For each stratum: `n_windows`, per-arm `within_stratum_r2` and `common_denominator_mse`,
`paired_delta_primary`, `paired_bootstrap_95_ci_primary` (10 000 resamples, seed 42),
`window_primary_deltas`, `sessions_positive_primary_delta` (sessions with mean Δ primary < 0),
residual decomposition (`bias_squared`, `variance`, `mse`), `circular_direction_dependence`
(1 − mean resultant length of error angles in the target-direction frame; **circular**, not
linear), and multiplicity fields under `multiplicity` (§4.3).

---

## 4. Multiple-comparisons discipline

### 4.1 Family definition (implemented)

Each stratification **axis** is one independent multiplicity family. A single query window
receives exactly one stratum label per axis, so the 8 direction strata never compete with the
4 speed strata in the same correction. **Benjamini–Hochberg FDR is applied separately within
each of the 5 axis families** (not across all 30 strata). This matches the scientific
question: "within directions, which directions show gain?" is separate from "within speed
quartiles, which quartiles show gain?"

| Family ID | Strata in family | Correction scope |
|---|---:|---|
| `target_direction` | 8 | BH-FDR among eligible direction strata only |
| `movement_speed` | 4 | BH-FDR among eligible speed strata only |
| `within_trial_phase` | 3 | BH-FDR among eligible phase strata only |
| `active_tuning_strength` | 3 | BH-FDR among eligible tuning strata only |
| `calibration_design_coverage` | 12 | BH-FDR among eligible design strata only |

**Justification:** axes answer distinct mechanistic questions; pooling all 30 into one family
would over-penalize direction tests because design-coverage alone contributes 12 strata.
Per-axis correction is the standard approach when families are pre-declared and non-exchangeable.

### 4.2 Eligibility before correction

Strata are **excluded from the family size m** used by BH (and receive `survives_fdr: false`)
when:

- `n_windows = 0` (`exclusion_reason: empty`), or
- `n_windows < 4` (`too_few_windows`), or
- `n_sessions < 2` (`too_few_sessions`).

Excluded counts are recorded per family (`excluded_counts`). Empty strata remain in the receipt
grid but do not inflate m.

### 4.3 Implemented test and discovery rule

Implemented in `mc_maze/decoder_error_structure.py::apply_axis_family_multiplicity_correction`
and emitted by both `decompose_decoder_error.py` receipts and
`aggregate_decoder_error_structure.py`.

Per eligible stratum the receipt/aggregator emits:

| Field | Meaning |
|---|---|
| `raw_p_value_two_sided` | Sign-flip permutation test on window-level paired primary deltas (5 000 permutations, seed 42) |
| `paired_bootstrap_95_ci_primary` | Unchanged bootstrap interval |
| `bh_adjusted_p_value` | Benjamini–Hochberg adjusted p within the axis family |
| `family` / `family_size_total` / `family_size_tested` | Axis name and m before/after exclusions |
| `ci_excludes_zero` | Both bootstrap bounds share sign |
| `uncorrected_significant` | `raw_p < 0.05` **and** `ci_excludes_zero` |
| `survives_fdr` | `bh_adjusted_p ≤ 0.10` **and** `ci_excludes_zero` |
| `discovery` | Same as `survives_fdr` (individual-stratum claim) |

**FDR threshold:** q = 0.10. **Uncorrected nominal:** α = 0.05 (reported for calibration only).

### 4.4 Uniformity falsifier (separate from FDR)

The **primary mechanism falsifier** remains the pre-declared **uniform-gain test**
(`uniform_gain_diagnostics.uniform_gain_flag` on the direction axis). It asks whether gain is
spread uniformly across strata — a pattern that would **falsify** the carrier mechanism story.
This test is **not** gated by FDR and **not** replaced by it. FDR controls over-interpretation
of individual strata; uniformity tests the global mechanistic prediction.

- **No post-hoc fishing:** full 30-stratum grid always reported.
- **Pre-declared predictions:** §2.1 states expected strata before any run.

---

## 5. Scope limits and sealed-session boundary

- **Allowed sessions:** development and validation splits only.  
- **Forbidden sessions (code must refuse):**  
  `sub-C_ses-CO-20151113`, `sub-C_ses-CO-20151116`, `sub-C_ses-CO-20151117`,
  `sub-C_ses-CO-20151119`, `sub-C_ses-CO-20151120`, `sub-C_ses-CO-20151201`.
- Receipt must record `sealed_test_sessions_opened: false`.
- Real-data paths are gated behind pytest marker `real_data` (skipped by default).
- **No GPU.** Runners set `CUDA_VISIBLE_DEVICES=""` and `torch.device("cpu")`.
- **No training** or large-scale experiment launch.

---

## 6. Artifacts

| File | Purpose |
|---|---|
| `mc_maze/decoder_error_structure.py` | Frozen stratum rules and decomposition core |
| `scripts/decompose_decoder_error.py` | Forward-only CPU receipt writer |
| `scripts/aggregate_decoder_error_structure.py` | Fail-closed multi-receipt aggregator |
| `tests/test_decompose_decoder_error.py` | Synthetic fixtures only |

---

## 7. Design against post-hoc stratum fishing

1. **Freeze before data:** all edges and labels are constants in code, committed in this
   document, and hashed into every receipt (`stratum_definitions`).
2. **Report the full grid:** all 30 strata appear in every receipt; empty strata are explicit
   (`n_windows = 0`), not omitted.
3. **Pre-declared predictions:** §2.1 states which strata should gain before any run.
4. **Falsifier is uniform gain**, not cherry-picked positives.
5. **Aggregator rejects** drift in stratum definitions, incomplete strata, sealed sessions, and
   mismatched query-window identities — the paired delta is void if windows differ.
6. **BH-FDR is computed in code** (§4) — not prose-only; receipts and aggregator output include
   raw and adjusted p-values per family.
