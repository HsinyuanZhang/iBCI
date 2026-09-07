"""Decoder shape, causality, permutation, and pairing contracts."""

from __future__ import annotations

import torch

from tfpd_exploration.src.m2_dual_track_v1.contracts import SessionBank, make_stub_bank
from tfpd_exploration.src.m2_dual_track_v1 import plan
from tfpd_exploration.src.m2_b_small_stability_v1 import config as cfg
from tfpd_exploration.src.m2_b_small_stability_v1.decoder import (
    SmallTransformerDecoder,
    init_state_sha256,
)
from tfpd_exploration.src.m2_b_small_stability_v1.training import unit_dropout_mask


ATOL = plan.PERM_ATOL
RTOL = plan.PERM_RTOL


def _permute_bank(bank: SessionBank, perm: torch.Tensor) -> SessionBank:
    return SessionBank(
        session_id=bank.session_id,
        support_trial_ids=bank.support_trial_ids,
        raw_trial_ids=bank.raw_trial_ids,
        X_store=bank.X_store,
        target_store=bank.target_store,
        eligible_starts=bank.eligible_starts,
        E0=bank.E0[perm],
        T=bank.T[perm],
        unit_mask=bank.unit_mask[perm],
        provenance=dict(bank.provenance),
        frozen_u=None if bank.frozen_u is None else bank.frozen_u[:, perm],
    )


def test_output_space_and_widths() -> None:
    model = SmallTransformerDecoder(seed=42).eval()
    bank = make_stub_bank(seed=2)
    x = torch.randn(3, cfg.SMALL.window, plan.CHANNELS)
    y = model.forward_last(x, bank, bank.unit_mask)
    assert y.shape == (3, 2)
    assert model.training_target_space == "decoder_raw"
    assert torch.isfinite(y).all()
    hidden = model.forward_hidden(x, bank, bank.unit_mask)
    assert hidden.shape == (3, cfg.SMALL.window, 256)
    scores = model.forward_scores(x, bank, bank.unit_mask)
    assert scores.shape == (3, cfg.SMALL.window, 2)
    pe = model.temporal.pe
    assert pe.shape[-1] == 256
    assert model.frontend.mha.embed_dim == 256
    assert model.frontend.mha.num_heads == 8


def test_s0_s1_init_sha_identical() -> None:
    a = SmallTransformerDecoder(seed=42)
    b = SmallTransformerDecoder(seed=42)
    sha_a = init_state_sha256(a)
    sha_b = init_state_sha256(b)
    assert sha_a == sha_b
    for (n1, p1), (n2, p2) in zip(a.named_parameters(), b.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), n1
    c = SmallTransformerDecoder(seed=43)
    assert init_state_sha256(c) != sha_a


def test_unit_permutation_invariance() -> None:
    model = SmallTransformerDecoder(seed=42).eval()
    bank = make_stub_bank(seed=4)
    x = torch.randn(2, cfg.SMALL.window, plan.CHANNELS)
    perm = torch.randperm(plan.CHANNELS)
    bank_p = _permute_bank(bank, perm)
    y = model.forward_last(x, bank, bank.unit_mask)
    y_p = model.forward_last(x[:, :, perm], bank_p, bank_p.unit_mask)
    assert torch.allclose(y, y_p, atol=ATOL, rtol=RTOL), float((y - y_p).abs().max())


def test_future_bin_does_not_change_prefix() -> None:
    model = SmallTransformerDecoder(seed=42).eval()
    bank = make_stub_bank(seed=5)
    x = torch.randn(2, cfg.SMALL.window, plan.CHANNELS)
    x_future = x.clone()
    x_future[:, 26:] += 4.0
    h = model.forward_hidden(x, bank, bank.unit_mask)
    h_f = model.forward_hidden(x_future, bank, bank.unit_mask)
    assert torch.allclose(h[:, :26], h_f[:, :26], atol=ATOL, rtol=RTOL)
    prefix = model.forward_hidden(x[:, :26], bank, bank.unit_mask)
    assert torch.allclose(prefix, h[:, :26], atol=ATOL, rtol=RTOL)


def test_causal_mask_blocks_future_keys() -> None:
    model = SmallTransformerDecoder(seed=42).eval()
    attn = model.temporal.blocks[0].attn
    width = 8
    qkv = torch.randn(1, width, 3 * 256)
    # Use the same causal helper the stack uses via a future-only perturbation of keys.
    x = torch.randn(1, width, 256)
    y, _ = attn(x)
    x2 = x.clone()
    x2[:, -1] += 10.0
    y2, _ = attn(x2)
    assert torch.allclose(y[:, :-1], y2[:, :-1], atol=1e-5, rtol=1e-4)
    assert not torch.allclose(y[:, -1], y2[:, -1], atol=1e-5)


def test_dropout_masks_pair_across_cells() -> None:
    mask = torch.ones(4, plan.CHANNELS, dtype=torch.bool)
    a = unit_dropout_mask(mask, seed=42, epoch=3, batch_id=17)
    b = unit_dropout_mask(mask, seed=42, epoch=3, batch_id=17)
    c = unit_dropout_mask(mask, seed=42, epoch=3, batch_id=18)
    d = unit_dropout_mask(mask, seed=42, epoch=4, batch_id=17)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert not torch.equal(a, d)
    assert a.shape == mask.shape
    assert a.dtype == torch.bool
    assert int(a.sum()) < int(mask.sum())
