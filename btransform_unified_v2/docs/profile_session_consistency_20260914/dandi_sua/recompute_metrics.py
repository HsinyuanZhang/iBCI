#!/usr/bin/env python3
"""Independently recompute and verify every bundled metric from profiles.npz."""
from __future__ import annotations
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ATOL = 2e-12

def rows(name):
    with (HERE / name).open(newline='') as f:
        return list(csv.DictReader(f))

def kernel(a, b, ell):
    d = a[:, None, :] - b[None, :, :]
    return np.exp(-np.sum(d * d, axis=-1) / (2 * ell * ell))

def metrics(a, b, ell):
    kxy = kernel(a, b, ell).mean()
    kxx = kernel(a, a, ell).mean()
    kyy = kernel(b, b, ell).mean()
    raw = float(kxx + kyy - 2 * kxy)
    return float(kxy / np.sqrt(kxx * kyy)), max(0.0, raw), raw

def close(actual, expected, label):
    if not np.isclose(actual, expected, rtol=0, atol=ATOL):
        raise AssertionError(f'{label}: {actual:.17g} != {expected:.17g}; atol={ATOL}')

def main():
    meta = json.loads((HERE / 'summary.json').read_text())
    row_meta = rows('rows.csv')
    pair_csv = rows('pairs.csv')
    half_csv = rows('split_half.csv')
    sessions = {}
    for r in row_meta:
        sessions.setdefault(r['session'], {'split': r['split'], 'date': r['date']})
        if sessions[r['session']] != {'split': r['split'], 'date': r['date']}:
            raise AssertionError('inconsistent session metadata')
    wanted = sorted(sessions)
    if len(wanted) != 24 or sum(sessions[s]['split'] == 'train' for s in wanted) != 18 or len(pair_csv) != 276 or len(half_csv) != 24:
        raise AssertionError('bundle cardinality mismatch')
    with np.load(HERE / 'profiles.npz', allow_pickle=False) as z:
        full = {s: np.asarray(z[f'full/{s}'], dtype=np.float64) for s in wanted}
        odd = {s: np.asarray(z[f'norm_odd/{s}'], dtype=np.float64) for s in wanted}
        even = {s: np.asarray(z[f'norm_even/{s}'], dtype=np.float64) for s in wanted}
        for s in wanted:
            ids, observed = z[f'row_ids/{s}'], z[f'observed_mask/{s}']
            if full[s].shape != (len(ids), 4) or not np.all(observed[ids]):
                raise AssertionError(f'invalid local rows for {s}')
    source = np.concatenate([full[s] for s in wanted if sessions[s]['split'] == 'train'])
    d = np.linalg.norm(source[:, None, :] - source[None, :, :], axis=-1)
    vals = d[np.triu_indices(len(source), 1)]
    ell = float(np.median(vals[vals > 0]))
    close(ell, float(meta['bandwidth']['ell']), 'bandwidth')
    expected_pairs = {}
    for i, a in enumerate(wanted):
        for b in wanted[i + 1:]:
            expected_pairs[(a, b)] = metrics(full[a], full[b], ell)
    for r in pair_csv:
        key = (r['session_a'], r['session_b'])
        if key not in expected_pairs:
            raise AssertionError(f'unexpected pair {key}')
        for name, got in zip(('kernel_cosine_similarity', 'mmd2', 'mmd2_unclipped'), expected_pairs[key]):
            close(got, float(r[name]), f'pair {key} {name}')
    expected_half = {s: metrics(odd[s], even[s], ell) for s in wanted}
    for r in half_csv:
        for name, got in zip(('kernel_cosine_similarity', 'mmd2', 'mmd2_unclipped'), expected_half[r['session']]):
            close(got, float(r[name]), f"half {r['session']} {name}")
    pair_s = [x[0] for x in expected_pairs.values()]
    half_s = [x[0] for x in expected_half.values()]
    for name, got in [('pair_similarity_mean', np.mean(pair_s)), ('pair_similarity_median', np.median(pair_s)), ('split_half_similarity_mean', np.mean(half_s)), ('split_half_similarity_median', np.median(half_s))]:
        close(float(got), float(meta[name]), name)
    print(json.dumps({'verified_pairs': len(expected_pairs), 'verified_split_halves': len(expected_half), 'atol': ATOL, 'ell': ell, 'pair_similarity_mean': float(np.mean(pair_s)), 'split_half_similarity_mean': float(np.mean(half_s))}, indent=2))

if __name__ == '__main__':
    main()
