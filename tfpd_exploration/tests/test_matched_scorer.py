"""Adversarial regression for the metric-flattening bug (ledger 11:30).

A predictor that is useless per-output but whose outputs sit on opposite
offsets must score NEGATIVE per-output variance-weighted R², while the
flattened grand-mean variant scores near 1.  If this ever fails again, the
evaluators have reintroduced the element-wise-mask flattening.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tfpd_lane.matched_scorer import session_r2


def _flattened_grand_mean_r2(pred, target):
    p = pred.flatten()
    t = target.flatten()
    return 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()


def test_adversarial_two_dim_negative_vs_flattened_near_one():
    torch.manual_seed(0)
    n = 2000
    # Target with strongly separated per-dim means; predictor is the per-dim
    # mean (zero skill).  Matched per-output R2 is exactly ~0.  A grand-mean
    # flattening absorbs the between-output term into the denominator and
    # scores the SAME useless predictor near 1.
    target = torch.randn(n, 2) + torch.tensor([50.0, -50.0])
    pred = target.mean(dim=0, keepdim=True).expand(n, 2).contiguous()
    matched = session_r2(pred, target)
    flattened = _flattened_grand_mean_r2(pred, target)
    assert abs(matched) < 1e-4, matched  # useless predictor scores exactly 0
    assert flattened > 0.99, flattened   # the bug would have called it ~1


def test_perfect_prediction_scores_one():
    t = torch.randn(500, 2)
    assert abs(session_r2(t.clone(), t) - 1.0) < 1e-6


def test_row_mask_preserves_two_dims():
    p = torch.arange(24.0).reshape(2, 3, 4)
    valid = torch.ones(2, 3, 4).bool().all(dim=-1)
    assert tuple(p[valid].shape) == (6, 4)
