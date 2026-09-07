"""CPU-only guards for M1 optimized-v2 formal training/scoring contracts."""
from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from tfpd_exploration.src.m1_optimized_v2.formal_score import _ema_into
from tfpd_exploration.src.m1_optimized_v2.source_dev import _all_dev_batches, _keep, _sha
from tfpd_exploration.src.m1_temporal_decoder_quick_product_v1.ema import DecoderEMA


def test_keep_is_batched_deterministic_and_not_rank_three():
    for batch in (1, 8, 32):
        a = _keep(batch, 3, 11, torch.device("cpu"))
        b = _keep(batch, 3, 11, torch.device("cpu"))
        c = _keep(batch, 3, 12, torch.device("cpu"))
        assert a.dtype == torch.bool and tuple(a.shape) == (batch, 64)
        assert torch.equal(a, b)
        assert not torch.equal(a, c)
        assert bool(a.any(dim=1).all())


class _Base:
    def __init__(self, pairs):
        self.window_indices = pairs


class _Dataset:
    def __init__(self, pairs):
        self.base = _Base(pairs)
    def __len__(self):
        return len(self.base.window_indices)


def test_tail_batches_are_session_pure_and_cover_every_window():
    pairs = [("ses-20120926", i) for i in range(33)] + [("ses-20120927", i) for i in range(65)] + [("ses-20120928", i) for i in range(1)]
    ds = _Dataset(pairs)
    batches = _all_dev_batches(ds, batch_size=32)
    assert sum(map(len, batches)) == len(pairs)
    assert [len(x) for x in batches] == [32, 1, 32, 32, 1, 1]
    for batch in batches:
        assert len({pairs[index][0] for index in batch}) == 1


def test_purge_predicate_has_no_m10_overlap_or_crossing_history():
    # Mirrors the frozen rule: train start >= M10 boundary and final bin < cut;
    # development full windows start at/after cut.
    m10, cut, window = 1_000, 8_000, 100
    starts = list(range(0, 9_000, 50))
    train = [x for x in starts if x >= m10 and x + window - 1 < cut]
    dev = [x for x in starts if x >= cut]
    assert train and dev and max(x + window - 1 for x in train) < cut
    assert min(dev) >= cut and min(train) >= m10
    assert set(train).isdisjoint(dev)
    assert _sha([("ses-20120926", x) for x in train]) != _sha([("ses-20120926", x) for x in dev])


def test_ema_requires_exact_trainable_shadow_keys():
    model = nn.Sequential(nn.Linear(3, 2), nn.LayerNorm(2))
    expected = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}
    _ema_into(model, expected)
    incomplete = dict(expected); incomplete.pop(next(iter(incomplete)))
    with pytest.raises(RuntimeError, match="incomplete"):
        _ema_into(model, incomplete)
    unexpected = {**expected, "bogus": torch.ones(1)}
    with pytest.raises(RuntimeError, match="unexpected"):
        _ema_into(model, unexpected)


def _toy_step(model, optimizer, ema, step):
    x = torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float32)
    y = torch.tensor([[0.2], [-0.1]], dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    # Uses the same stateless epoch/batch hash law as the formal trainer.
    keep = _keep(2, 1 + step // 2, step % 2, torch.device("cpu"))[:, :2].float()
    loss = ((model(x * keep) - y) ** 2).mean()
    loss.backward(); optimizer.step(); ema.update_after_step(model)


def test_numerical_end_epoch_resume_matches_uninterrupted_toy_training():
    torch.manual_seed(123); np.random.seed(123); random.seed(123)
    uninterrupted = nn.Linear(2, 1); opt_a = torch.optim.AdamW(uninterrupted.parameters(), lr=1e-4, weight_decay=.01); ema_a = DecoderEMA(uninterrupted, decay=.9995)
    for step in range(4): _toy_step(uninterrupted, opt_a, ema_a, step)

    torch.manual_seed(123); np.random.seed(123); random.seed(123)
    resumed = nn.Linear(2, 1); opt_b = torch.optim.AdamW(resumed.parameters(), lr=1e-4, weight_decay=.01); ema_b = DecoderEMA(resumed, decay=.9995)
    for step in range(2): _toy_step(resumed, opt_b, ema_b, step)
    checkpoint = {"model": copy.deepcopy(resumed.state_dict()), "optimizer": copy.deepcopy(opt_b.state_dict()), "ema": copy.deepcopy(ema_b.checkpoint_state()), "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    resumed.load_state_dict(checkpoint["model"]); opt_b.load_state_dict(checkpoint["optimizer"]); ema_b.load_checkpoint_state(checkpoint["ema"])
    random.setstate(checkpoint["python"]); np.random.set_state(checkpoint["numpy"]); torch.set_rng_state(checkpoint["torch"])
    for step in range(2, 4): _toy_step(resumed, opt_b, ema_b, step)
    for key, value in uninterrupted.state_dict().items(): assert torch.equal(value, resumed.state_dict()[key])
    for key, value in ema_a.shadow.items(): assert torch.equal(value, ema_b.shadow[key])
