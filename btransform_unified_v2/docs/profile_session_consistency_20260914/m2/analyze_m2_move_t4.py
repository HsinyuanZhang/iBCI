#!/usr/bin/env python3
"""Read-only consistency audit and figures for deployed M2 Full-582481 MOVE-T4.

Uses the exact final payload's precomputed, fixed-source-normalized T arrays.
It neither opens query labels nor trains, scores, changes caches, or changes the
payload. All 13 sessions and all 78 unordered session pairs are retained.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

ROOT = Path('/home/xinyuan/Work_host/SPINT')
OUT = ROOT / 'btransform_unified_v2/docs/profile_session_consistency_20260914/m2'
PAYLOAD = ROOT / ('tfpd_exploration/submissions/evalai_m2_rift_r50_s1formal_flat_e9_v1/'
                  'artifacts/m2_rift_r50_s1formal_flat_e9.pkl')
BANK_SOURCE = ROOT / ('tfpd_exploration/submissions/evalai_m2_small_concat_ort_v1/'
                      'artifacts/m2_small_trf_s42_ema_e08_ext6.pkl')
NORMALIZER = ROOT / 'tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache/move_t4_normalizer.json'
EXT6 = Path('/mnt/data/SPINT_cold_archive/2026-09-11/btransform_unified_v2/results/rift_v1/m2_joint_ext6_raw_m33_v1')
SEED, N_SHUFFLES = 20260914, 200
FEATURES = ('m_cos_phi', 'm_sin_phi', 'modulation_magnitude', 'baseline_rate')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float('nan')
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def parse_tag(tag: str) -> tuple[str, date, str]:
    run, ymd = tag.split('_')
    return (f'{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}', date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:])), run)


def group_name(day_gap: int) -> str:
    if day_gap == 0:
        return 'same_date_different_run'
    if day_gap == 1:
        return 'adjacent_date'
    return 'other_different_date'


def qstats(v: np.ndarray) -> dict[str, float]:
    v = np.asarray(v, dtype=np.float64)
    return {'n': int(v.size), 'median': float(np.median(v)), 'q25': float(np.quantile(v, .25)),
            'q75': float(np.quantile(v, .75)), 'mean': float(np.mean(v)),
            'min': float(np.min(v)), 'max': float(np.max(v))}


def fmt(v: float) -> str:
    return f'{v:.2f}' if np.isfinite(v) else 'NA'


def load_payload(path: Path) -> dict:
    with path.open('rb') as f:
        return pickle.load(f)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def prior_pair_metrics(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """Read prior outputs solely to prove a presentation-only rerun kept core metrics."""
    if not path.is_file():
        return {}
    with path.open(newline='', encoding='utf-8') as f:
        return {(r['left_tag'], r['right_tag']):
                (float(r['flattened_r_actual']), float(r['rmse_fixed_source_normalized_units']))
                for r in csv.DictReader(f)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = prior_pair_metrics(OUT / 'pairs.csv')
    payload = load_payload(PAYLOAD)
    source = load_payload(BANK_SOURCE)
    banks = payload['bank_by_dataset_tag']
    source_banks = source['bank_by_dataset_tag']
    tags = list(banks)
    assert len(tags) == 13 and list(source_banks) == tags
    normalizer = json.loads(NORMALIZER.read_text())
    mean = np.asarray(normalizer['mean'], dtype=np.float64)
    std = np.asarray(normalizer['std'], dtype=np.float64)
    assert mean.shape == std.shape == (4,) and np.all(std > 0)

    sessions = []
    profiles: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for order, tag in enumerate(tags):
        b = banks[tag]
        t = np.asarray(b['T'], dtype=np.float32)
        mask = np.asarray(b['unit_mask'], dtype=bool)
        assert t.shape == (96, 4) and mask.shape == (96,) and np.isfinite(t).all()
        assert np.array_equal(t, np.asarray(source_banks[tag]['T'], dtype=np.float32))
        day_text, day, run = parse_tag(tag)
        profiles[tag], masks[tag] = t, mask
        sessions.append({'session_order': order, 'tag': tag, 'date': day_text, 'run': run,
                         'source_or_public': 'source_held_in' if order < 7 else 'public_held_out_calibration',
                         'n_units': int(mask.sum()), 'n_total_channels': int(mask.size),
                         'coverage': float(mask.mean()), 't_shape': '96x4', 't_dtype': str(t.dtype)})

    profile_rows = []
    for s in sessions:
        tag = s['tag']; t = profiles[tag]; m = masks[tag]
        for channel in range(96):
            for d, feature in enumerate(FEATURES):
                profile_rows.append({'tag': tag, 'date': s['date'], 'run': s['run'],
                                     'source_or_public': s['source_or_public'], 'channel_id': channel,
                                     'unit_mask': bool(m[channel]), 'feature_index': d,
                                     'feature': feature, 'T_source_normalized': float(t[channel, d]),
                                     'raw_coefficient_reconstructed': float(t[channel, d] * std[d] + mean[d])})

    pair_rows, null_rows = [], []
    R = np.eye(13, dtype=float); RMSE = np.zeros((13, 13), dtype=float)
    rng = np.random.default_rng(SEED)
    for i, left in enumerate(sessions):
        for j in range(i + 1, len(sessions)):
            right = sessions[j]
            a, b = profiles[left['tag']], profiles[right['tag']]
            common = masks[left['tag']] & masks[right['tag']]
            ncommon = int(common.sum())
            union = masks[left['tag']] | masks[right['tag']]
            jaccard = float(ncommon / int(union.sum())) if union.any() else float('nan')
            gap = abs((parse_tag(left['tag'])[1] - parse_tag(right['tag'])[1]).days)
            actual = pearson(a[common], b[common]) if ncommon >= 8 else float('nan')
            rmse = float(np.sqrt(np.mean((a[common] - b[common]) ** 2))) if ncommon >= 8 else float('nan')
            features = {f'r_{name}': pearson(a[common, d], b[common, d]) if ncommon >= 8 else float('nan')
                        for d, name in enumerate(FEATURES)}
            null = []
            for rep in range(N_SHUFFLES):
                perm = rng.permutation(96)
                # Whole electrode rows move together: four within-channel dimensions remain linked.
                val = pearson(a[common], b[perm][common]) if ncommon >= 8 else float('nan')
                null.append(val)
                null_rows.append({'left_tag': left['tag'], 'right_tag': right['tag'], 'replicate': rep,
                                  'shuffle_seed': SEED, 'flattened_r_electrode_row_shuffle': val})
            null = np.asarray(null, dtype=float)
            row = {'left_tag': left['tag'], 'right_tag': right['tag'],
                   'left_date': left['date'], 'right_date': right['date'],
                   'left_run': left['run'], 'right_run': right['run'],
                   'left_source_or_public': left['source_or_public'],
                   'right_source_or_public': right['source_or_public'],
                   'day_gap': gap, 'pair_group': group_name(gap), 'n_common_channels': ncommon,
                   'coverage_left': left['coverage'], 'coverage_right': right['coverage'],
                   'jaccard_unit_mask': jaccard, 'insufficient_ncommon_lt8': ncommon < 8,
                   'flattened_r_actual': actual, 'rmse_fixed_source_normalized_units': rmse,
                   'null_r_mean': float(np.nanmean(null)), 'null_r_sd': float(np.nanstd(null, ddof=1)),
                   'null_r_q025': float(np.nanquantile(null, .025)), 'null_r_q50': float(np.nanmedian(null)),
                   'null_r_q975': float(np.nanquantile(null, .975)),
                   **features}
            pair_rows.append(row); R[i, j] = R[j, i] = actual; RMSE[i, j] = RMSE[j, i] = rmse

    if prior:
        now = {(r['left_tag'], r['right_tag']):
               (r['flattened_r_actual'], r['rmse_fixed_source_normalized_units']) for r in pair_rows}
        assert now.keys() == prior.keys()
        for key in now:
            assert now[key] == prior[key], f'core pair metric drift after presentation-only rerun: {key}'

    write_csv(OUT / 'profile_rows.csv', profile_rows,
              ['tag', 'date', 'run', 'source_or_public', 'channel_id', 'unit_mask', 'feature_index',
               'feature', 'T_source_normalized', 'raw_coefficient_reconstructed'])
    pair_fields = list(pair_rows[0])
    write_csv(OUT / 'pairs.csv', pair_rows, pair_fields)
    write_csv(OUT / 'shuffle_null.csv', null_rows,
              ['left_tag', 'right_tag', 'replicate', 'shuffle_seed', 'flattened_r_electrode_row_shuffle'])
    write_csv(OUT / 'sessions.csv', sessions, list(sessions[0]))

    actuals = np.asarray([x['flattened_r_actual'] for x in pair_rows])
    rmses = np.asarray([x['rmse_fixed_source_normalized_units'] for x in pair_rows])
    null_all = np.asarray([x['flattened_r_electrode_row_shuffle'] for x in null_rows])
    summaries = {'all_pairs': qstats(actuals), 'all_pairs_rmse_source_normalized_units': qstats(rmses),
                 'shuffle_null_flattened_r': qstats(null_all)}
    for group in ('same_date_different_run', 'adjacent_date', 'other_different_date'):
        x = np.asarray([p['flattened_r_actual'] for p in pair_rows if p['pair_group'] == group])
        y = np.asarray([p['rmse_fixed_source_normalized_units'] for p in pair_rows if p['pair_group'] == group])
        summaries[group] = {'flattened_r_actual': qstats(x), 'rmse_fixed_source_normalized_units': qstats(y)}
    different_date = [p for p in pair_rows if p['day_gap'] > 0]
    summaries['all_different_date'] = {
        'definition': 'all pairs with actual calendar day_gap > 0',
        'flattened_r_actual': qstats(np.asarray([p['flattened_r_actual'] for p in different_date])),
        'rmse_fixed_source_normalized_units': qstats(
            np.asarray([p['rmse_fixed_source_normalized_units'] for p in different_date])),
    }
    pair_null_means = np.asarray([p['null_r_mean'] for p in pair_rows])
    summaries['pair_level_null_comparison'] = {
        'null_r_mean_across_pairs': qstats(pair_null_means),
        'actual_minus_pair_null_r_mean': qstats(actuals - pair_null_means),
        'interpretation': 'descriptive paired contrast only; no independent-pair p-value',
    }

    labels = [f'{s["date"][5:]}/{s["run"]}' for s in sessions]
    colors = {'same_date_different_run': '#d95f02', 'adjacent_date': '#1b9e77', 'other_different_date': '#7570b3'}
    pdf = OUT / 'm2_move_t4_profile_session_consistency.pdf'
    png = OUT / 'm2_move_t4_profile_session_consistency.png'
    with PdfPages(pdf) as pages:
        fig, ax = plt.subplots(figsize=(10.5, 9.5), constrained_layout=True)
        im = ax.imshow(R, vmin=-1, vmax=1, cmap='coolwarm')
        ax.set_xticks(range(13), labels, rotation=55, ha='right'); ax.set_yticks(range(13), labels)
        for i in range(13):
            for j in range(13):
                ax.text(j, i, f'{R[i,j]:.2f}', ha='center', va='center', fontsize=6,
                        color='white' if abs(R[i,j]) > .5 else 'black')
        fig.colorbar(im, ax=ax, label='Flattened Pearson r: fixed source-normalized MOVE-T4')
        ax.set_title('M2 Full 582481: all 13 sessions, same physical channel coordinate')
        pages.savefig(fig, dpi=220); plt.close(fig)

        fig, axs = plt.subplots(1, 2, figsize=(13, 5.3), constrained_layout=True)
        for group, color in colors.items():
            sub = [p for p in pair_rows if p['pair_group'] == group]
            axs[0].scatter([p['day_gap'] for p in sub], [p['flattened_r_actual'] for p in sub],
                           label=group.replace('_', ' '), color=color, alpha=.8, s=42)
            axs[1].scatter([p['day_gap'] for p in sub], [p['rmse_fixed_source_normalized_units'] for p in sub],
                           label=group.replace('_', ' '), color=color, alpha=.8, s=42)
        axs[0].axhline(0, color='black', lw=.7); axs[0].set(xlabel='Actual calendar day gap', ylabel='Flattened Pearson r', title='Similarity by elapsed days')
        axs[1].set(xlabel='Actual calendar day gap', ylabel='RMSE (fixed source-normalized T units)', title='Difference magnitude by elapsed days')
        axs[0].legend(fontsize=8); axs[1].legend(fontsize=8)
        fig.savefig(OUT / 'm2_move_t4_profile_daygap.png', dpi=260)
        pages.savefig(fig, dpi=220); plt.close(fig)

        fig, axs = plt.subplots(1, 2, figsize=(13, 5.3), constrained_layout=True)
        axs[0].hist(null_all, bins=40, density=True, color='#bdbdbd', label=f'200 row shuffles × 78 pairs (n={null_all.size})')
        axs[0].hist(actuals, bins=15, density=True, color='#1f78b4', alpha=.70, label='78 actual aligned pairs')
        axs[0].axvline(np.median(actuals), color='#1f78b4', ls='--', label=f'actual median={np.median(actuals):.2f}')
        axs[0].axvline(np.median(null_all), color='#555555', ls='--', label=f'null median={np.median(null_all):.2f}')
        axs[0].set(xlabel='Flattened Pearson r', ylabel='Density', title='Actual fixed-channel alignment versus row-shuffle null')
        axs[0].legend(fontsize=8)
        data = [actuals, null_all]
        axs[1].boxplot(data, tick_labels=['actual\n78 pairs', 'row shuffle null\n15,600 draws'], showfliers=False)
        axs[1].set(ylabel='Flattened Pearson r', title='Distribution comparison; no independent-pair p-value')
        for k, vals in enumerate(data, 1):
            axs[1].text(k, np.nanmax(vals), f'median {np.nanmedian(vals):.2f}', ha='center', va='bottom', fontsize=9)
        fig.savefig(OUT / 'm2_move_t4_profile_actual_vs_shuffle_null.png', dpi=260)
        pages.savefig(fig, dpi=220); plt.close(fig)

    # A representative PNG copies the complete first-page heatmap; the PDF contains all panels.
    fig, ax = plt.subplots(figsize=(10.5, 9.5), constrained_layout=True)
    im = ax.imshow(R, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(13), labels, rotation=55, ha='right'); ax.set_yticks(range(13), labels)
    for i in range(13):
        for j in range(13):
            ax.text(j, i, f'{R[i,j]:.2f}', ha='center', va='center', fontsize=6,
                    color='white' if abs(R[i,j]) > .5 else 'black')
    fig.colorbar(im, ax=ax, label='Flattened Pearson r: fixed source-normalized MOVE-T4')
    ax.set_title('M2 Full 582481: all 13 sessions, same physical channel coordinate')
    fig.savefig(png, dpi=260); plt.close(fig)

    exact_support_stats = False
    provenance = {
        'analysis': 'M2 Full 582481 deployed MOVE-T4 session consistency',
        'read_only': True, 'payload_path': str(PAYLOAD), 'payload_sha256': sha256(PAYLOAD),
        'bank_source_payload_path': str(BANK_SOURCE), 'bank_source_payload_sha256': sha256(BANK_SOURCE),
        'normalizer_path': str(NORMALIZER), 'normalizer_sha256': sha256(NORMALIZER),
        'normalizer_mean': normalizer['mean'], 'normalizer_std': normalizer['std'],
        'profile_space': 'fixed source-normalized MOVE-T4; no sessionwise re-normalization, rotation, or alignment',
        'profile_shape_per_session': [96, 4], 'features': list(FEATURES),
        'n_sessions': 13, 'n_unordered_pairs': len(pair_rows), 'n_shuffles_per_pair': N_SHUFFLES,
        'shuffle': 'fixed seed 20260914; a whole 4-D electrode row of the right session is permuted; no pairwise p-values',
        'all_payload_T_byte_equal_to_bank_source': True,
        'source_cache_T_byte_equal_to_payload_for_all_7_heldin': True,
        'heldout_ext6_artifact': str(EXT6),
        'heldout_ext6_note': 'six-session raw-M33 manifest gives public-NWB provenance and seals T equality against query cache',
        'channel_correspondence': 'NWB 13-file audit supports fixed electrode/channel coordinate 0..95; not longitudinal same-neuron identity',
        'exact_per_trial_t4_sufficient_statistics_cached': exact_support_stats,
        'split_half_not_run': 'No cached trial spike sums, valid-prefix lengths, and target angles. Do not treat interpolated calib_activity as exact deployed-T split-half input.',
        'query_labels_opened': False, 'decoder_scored': False,
    }
    (OUT / 'input_sources.json').write_text(json.dumps(provenance, indent=2) + '\n')
    summary = {'provenance': provenance, 'sessions': sessions, 'summary_statistics': summaries,
               'feature_pair_medians': {f: qstats(np.asarray([p[f'r_{f}'] for p in pair_rows])) for f in FEATURES},
               'all_pairs_have_ncommon_at_least_8': bool(all(p['n_common_channels'] >= 8 for p in pair_rows)),
               'all_pairs_unit_mask_jaccard_one': bool(all(p['jaccard_unit_mask'] == 1 for p in pair_rows)),
               'date_span_days': int((max(parse_tag(t)[1] for t in tags) - min(parse_tag(t)[1] for t in tags)).days)}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    readme = f'''# M2 MOVE-T4 profile session consistency

This directory is a **read-only descriptive analysis** of the exact deployed M2 Full 582481 carrier: `T`, the fixed source-normalized MOVE-T4 profile. It does not use query labels, train a model, score a decoder, alter a cache, or report decoding performance.

## Result

All 13 sessions and all 78 unordered pairs were retained. Every session has 96 valid channel rows, so every pair has 96 common rows and unit-mask Jaccard `1.00`. The all-pair flattened profile correlation has median `{fmt(summaries['all_pairs']['median'])}` (IQR `{fmt(summaries['all_pairs']['q25'])}`–`{fmt(summaries['all_pairs']['q75'])}`); its RMSE is median `{fmt(summaries['all_pairs_rmse_source_normalized_units']['median'])}` fixed-source-normalized T units (IQR `{fmt(summaries['all_pairs_rmse_source_normalized_units']['q25'])}`–`{fmt(summaries['all_pairs_rmse_source_normalized_units']['q75'])}`). The 15,600 fixed-seed electrode-row-shuffle null draws have correlation median `{fmt(summaries['shuffle_null_flattened_r']['median'])}` (IQR `{fmt(summaries['shuffle_null_flattened_r']['q25'])}`–`{fmt(summaries['shuffle_null_flattened_r']['q75'])}`). These are descriptive alignment checks, not independent-pair p-values.

| Pair class | n | Actual flattened r, median [IQR] | RMSE, median [IQR] |
|---|---:|---:|---:|
'''
    for g in ('same_date_different_run', 'adjacent_date', 'other_different_date'):
        a, b = summaries[g]['flattened_r_actual'], summaries[g]['rmse_fixed_source_normalized_units']
        readme += f"| {g.replace('_', ' ')} | {a['n']} | {fmt(a['median'])} [{fmt(a['q25'])}, {fmt(a['q75'])}] | {fmt(b['median'])} [{fmt(b['q25'])}, {fmt(b['q75'])}] |\n"
    a, b = summaries['all_different_date']['flattened_r_actual'], summaries['all_different_date']['rmse_fixed_source_normalized_units']
    readme += f"| all different date (`day_gap > 0`) | {a['n']} | {fmt(a['median'])} [{fmt(a['q25'])}, {fmt(a['q75'])}] | {fmt(b['median'])} [{fmt(b['q25'])}, {fmt(b['q75'])}] |\n"
    readme += "\n`adjacent date` means **exactly one actual calendar day** of separation (`day_gap == 1`); it contains seven pairs here. The 13-session calendar span is 36 days.\n"
    readme += '''

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
'''
    (OUT / 'README.md').write_text(readme)
    # Hash finished outputs only. Exclude this manifest and the script to avoid self-reference.
    hashed = ['README.md', 'input_sources.json', 'm2_move_t4_profile_session_consistency.pdf',
              'm2_move_t4_profile_session_consistency.png', 'm2_move_t4_profile_daygap.png',
              'm2_move_t4_profile_actual_vs_shuffle_null.png', 'pairs.csv', 'profile_rows.csv',
              'sessions.csv', 'shuffle_null.csv', 'summary.json']
    (OUT / 'output_hashes.json').write_text(json.dumps({
        'schema': 'm2_move_t4_output_hashes_v1',
        'script': 'analyze_m2_move_t4.py',
        'note': 'hashes generated outputs only; excludes this manifest and the script to avoid self-reference',
        'sha256': {name: sha256(OUT / name) for name in hashed},
    }, indent=2) + '\n')
    print(json.dumps({'summary': summaries, 'out': str(OUT)}, indent=2))

if __name__ == '__main__':
    main()
