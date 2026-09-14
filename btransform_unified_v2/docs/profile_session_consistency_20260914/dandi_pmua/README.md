# DANDI PMUA session-profile consistency

This directory is a support-cache-only descriptive analysis of MOVE-T4 profiles across all 18 source and 6 development sessions in the locked DANDI 2015 protocol. It does not read query labels, final caches, model checkpoints, or scoring outputs. Cross-date results describe structure retained relative to a same-pair row-shuffle reference; they make no significance claim.

Run from repository root:

```bash
/home/xinyuan/miniconda3/envs/spint/bin/python btransform_unified_v2/docs/profile_session_consistency_20260914/dandi_pmua/analyze_pmua_profiles.py
```

For each session, the script applies the repository's `estimate_move_t4` to its 33 `carrier_counts` and `carrier_angles`, then applies the frozen PMUA source normalizer from `pretrain_pmua_s42/source_stats.json`. It does not fit a per-session normalizer, rotate features, or align profiles beyond exact matching of canonical electrode indices.

`pairs.csv` has all 276 date pairs. `r_norm_flat` is the Pearson correlation after flattening the four source-normalized T4 columns for exactly shared canonical electrodes. `rmse_norm` is in source-standard-deviation units. `r_raw_*` preserve the four raw-T4 dimension correlations. The shuffled null permutes whole four-dimensional electrode rows within a pair 200 times; it is descriptive and has no p-value. `split_half.csv` uses fixed odd/even support-trial halves; this is within-session estimator-noise context, not a drift correction.

Only PMUA has a defensible cross-session identity here: `channel_indices` maps observed pooled channels to the canonical 96 hardware-electrode table. Sorted SUA unit columns do not establish cross-session unit identity, so this analysis must not be repurposed as a SUA unit trajectory.
