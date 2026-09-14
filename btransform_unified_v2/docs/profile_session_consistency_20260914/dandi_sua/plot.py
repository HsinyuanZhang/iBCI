#!/usr/bin/env python3
"""Render the DANDI SUA profile-consistency figure from this bundle alone."""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

def read_csv(name):
    with (HERE / name).open(newline='') as f:
        return list(csv.DictReader(f))

def main():
    pairs = read_csv('pairs.csv')
    halves = read_csv('split_half.csv')
    # Each pair appears once; use the date separation so no row identity is implied.
    by_kind = defaultdict(list)
    for x in pairs:
        kind = {'train': 'T', 'dev': 'D'}[x['split_a']] + {'train': 'T', 'dev': 'D'}[x['split_b']]
        if kind == 'DT':
            kind = 'TD'
        by_kind[kind].append((float(x['days']), float(x['kernel_cosine_similarity'])))
    styles = {'TT': ('#1f77b4', 'source/source'), 'TD': ('#9467bd', 'source/development'),
              'DD': ('#ff7f0e', 'development/development')}
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.15), constrained_layout=True)
    ax = axes[0]
    for kind in ('TT', 'TD', 'DD'):
        vals = np.asarray(by_kind[kind])
        if len(vals):
            color, label = styles[kind]
            ax.scatter(vals[:, 0], vals[:, 1], s=23, alpha=.72, color=color,
                       edgecolors='none', label=f'{label} (n={len(vals)})')
    assert {kind: len(by_kind[kind]) for kind in ('TT', 'TD', 'DD')} == {'TT': 153, 'TD': 108, 'DD': 15}
    assert sum(len(by_kind[kind]) for kind in ('TT', 'TD', 'DD')) == len(pairs) == 276
    ax.set(xlabel='Calendar separation (days)', ylabel='Kernel distribution similarity $S$',
           title='Between-session MOVE-T4 profile similarity')
    ax.set_ylim(.90, 1.001)
    ax.grid(alpha=.23)
    ax.legend(frameon=False, fontsize=8, loc='lower left')

    ax = axes[1]
    order = ('train', 'dev')
    data = [np.array([float(x['kernel_cosine_similarity']) for x in halves if x['split'] == s]) for s in order]
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    for body, color in zip(parts['bodies'], ('#1f77b4', '#ff7f0e')):
        body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(.5)
    for i, vals in enumerate(data, 1):
        jitter = np.linspace(-.07, .07, len(vals))
        ax.scatter(np.full(len(vals), i)+jitter, vals, s=17, color=('#1f77b4','#ff7f0e')[i-1], zorder=3)
    ax.set(xticks=[1,2], xticklabels=['Source\n(n=18)', 'Development\n(n=6)'],
           ylabel='Odd/even kernel similarity $S$', title='Within-session split-half reference')
    ax.set_ylim(.90, 1.001)
    ax.grid(axis='y', alpha=.23)
    fig.suptitle('DANDI 2015 SUA: source-normalized empirical carrier profiles', fontsize=13)
    for suffix in ('png', 'pdf'):
        fig.savefig(HERE / f'profile_consistency.{suffix}', dpi=220, bbox_inches='tight')

if __name__ == '__main__':
    main()
