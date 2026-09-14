# M2 MOVE-T4 profile session consistency

This directory is a **read-only descriptive analysis** of the exact deployed M2 Full 582481 carrier: `T`, the fixed source-normalized MOVE-T4 profile. It does not use query labels, train a model, score a decoder, alter a cache, or report decoding performance.

## Result

All 13 sessions and all 78 unordered pairs were retained. Every session has 96 valid channel rows, so every pair has 96 common rows and unit-mask Jaccard `1.00`. The all-pair flattened profile correlation has median `0.57` (IQR `0.45`–`0.67`); its RMSE is median `0.88` fixed-source-normalized T units (IQR `0.75`–`1.07`). The 15,600 fixed-seed electrode-row-shuffle null draws have correlation median `-0.01` (IQR `-0.04`–`0.03`). These are descriptive alignment checks, not independent-pair p-values.

| Pair class | n | Actual flattened r, median [IQR] | RMSE, median [IQR] |
|---|---:|---:|---:|
| same date different run | 5 | 0.90 [0.89, 0.94] | 0.45 [0.41, 0.55] |
| adjacent date | 7 | 0.48 [0.46, 0.59] | 1.12 [0.85, 1.17] |
| other different date | 66 | 0.57 [0.45, 0.66] | 0.88 [0.77, 1.06] |
| all different date (`day_gap > 0`) | 73 | 0.57 [0.45, 0.65] | 0.89 [0.77, 1.08] |

`adjacent date` means **exactly one actual calendar day** of separation (`day_gap == 1`); it contains seven pairs here. The 13-session calendar span is 36 days.


## Files

- `m2_move_t4_profile_session_consistency.pdf`: three panels: 13-session actual correlation heatmap, day-gap versus correlation/RMSE, and actual-alignment versus row-shuffle distributions.
- `m2_move_t4_profile_session_consistency.png`: heatmap page.
- `m2_move_t4_profile_daygap.png`: the day-gap/RMSE PDF page as a standalone PNG.
- `m2_move_t4_profile_actual_vs_shuffle_null.png`: the actual-versus-row-shuffle PDF page as a standalone PNG.
- `profile_rows.csv`: all 13 × 96 × 4 deployed standardized T entries. `raw_coefficient_reconstructed` is an explicit inverse-z-score diagnostic; the figures do **not** label standardized T as raw coefficients.
- `pairs.csv`: all 78 pairwise measurements at full CSV precision, including four per-feature correlations.
- `shuffle_null.csv`: all 78 × 200 row-shuffle null correlations at full precision.
- `sessions.csv`, `summary.json`, `input_sources.json`: roster, numeric summary, and immutable input/source declarations.
- `analyze_m2_move_t4.py`: deterministic analysis script.

## Carrier and channel interpretation

`T[c,:]` contains fixed source-normalized cosine-tuning profile features `(m_cos_phi, m_sin_phi, modulation_magnitude, baseline_rate)`. The original fit uses the first 33 calibration trials and each trial's raw neural bins `[5,30)`; the source-seven pooled normalizer is fixed before target sessions. No per-session centering, rescaling, rotation, or feature alignment occurs in this analysis.

A common M2 row `c=0..95` is supported as a fixed physical electrode/channel coordinate by the NWB electrode mapping and unchanged loader row order. It is **not** proof that row `c` is the same longitudinal neuron across dates. See `btransform_unified_v2/external_baselines_v1/docs/CHANNEL_COORDINATES.md`.

## Split-half limitation

No exact per-trial deployed-T sufficient-statistics cache is present: the artifacts do not retain the raw per-trial spike sums, valid-prefix lengths, and target angles used by the T4 fit. Split-half was intentionally not run. `calib_activity.npy` is available but is not substituted for those exact T4 inputs.
