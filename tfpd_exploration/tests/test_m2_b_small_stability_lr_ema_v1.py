"""LR schedule and EMA numeric contracts."""

from __future__ import annotations

import math

import torch
from torch import nn

from tfpd_exploration.src.m2_dual_track_v1 import plan, training as old_training
from tfpd_exploration.src.m2_b_small_stability_v1 import config as cfg
from tfpd_exploration.src.m2_b_small_stability_v1.ema import DecoderEMA
from tfpd_exploration.src.m2_b_small_stability_v1.training import (
    apply_cell_lr,
    lr_at_update,
    make_optimizer,
)


W = 3165
T = 75960
LR_MAX = 3.0e-4
LR_MIN = 3.0e-5


def test_warmup_endpoints_no_jump() -> None:
    s0_1 = lr_at_update("S0-SMALL-LEGACY", t=1)
    s1_1 = lr_at_update("S1-SMALL-COS", t=1)
    assert abs(s0_1 - LR_MAX * 1 / W) < 1e-15
    assert abs(s1_1 - s0_1) < 1e-15
    s0_w = lr_at_update("S0-SMALL-LEGACY", t=W)
    s1_w = lr_at_update("S1-SMALL-COS", t=W)
    assert abs(s0_w - LR_MAX) < 1e-15
    assert abs(s1_w - LR_MAX) < 1e-15
    s0_w1 = lr_at_update("S0-SMALL-LEGACY", t=W + 1)
    s1_w1 = lr_at_update("S1-SMALL-COS", t=W + 1)
    assert abs(s0_w1 - LR_MAX) < 1e-15
    p = 1.0 / (T - W)
    expect_s1 = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * p))
    assert abs(s1_w1 - expect_s1) < 1e-15
    assert s1_w1 < LR_MAX
    s1_t = lr_at_update("S1-SMALL-COS", t=T)
    assert abs(s1_t - LR_MIN) < 1e-15


def test_s0_matches_old_epoch_sequence() -> None:
    # Epoch 12 last update, epoch 13 first, epoch 24 last vs old apply_b_lr.
    opt = torch.optim.AdamW([nn.Parameter(torch.zeros(2))], lr=LR_MAX)
    old_e12 = old_training.apply_b_lr(
        opt, LR_MAX, global_step=12 * W, warmup_steps=W, epoch=12, max_epochs=24
    )
    old_e13 = old_training.apply_b_lr(
        opt, LR_MAX, global_step=12 * W + 1, warmup_steps=W, epoch=13, max_epochs=24
    )
    old_e24 = old_training.apply_b_lr(
        opt, LR_MAX, global_step=T, warmup_steps=W, epoch=24, max_epochs=24
    )
    # S0 uses 1-based t; last update of epoch e is t=e*W.
    assert abs(lr_at_update("S0-SMALL-LEGACY", t=12 * W) - LR_MAX) < 1e-15
    assert abs(lr_at_update("S0-SMALL-LEGACY", t=12 * W) - old_e12) < 1e-15
    e13 = 13
    expect_e13 = LR_MAX * (0.1 + 0.45 * (1.0 + math.cos(math.pi * (e13 - 13) / 11)))
    assert abs(lr_at_update("S0-SMALL-LEGACY", t=12 * W + 1) - expect_e13) < 1e-15
    assert abs(expect_e13 - old_e13) < 1e-15
    expect_e24 = LR_MAX * (0.1 + 0.45 * (1.0 + math.cos(math.pi * (24 - 13) / 11)))
    assert abs(lr_at_update("S0-SMALL-LEGACY", t=T) - expect_e24) < 1e-15
    assert abs(expect_e24 - old_e24) < 1e-15
    assert abs(expect_e24 - 0.1 * LR_MAX) < 1e-15


def test_n_cells_are_exact_one_third_of_s_schedules() -> None:
    for t in (1, W, W + 1, 12 * W, 12 * W + 1, T):
        assert abs(lr_at_update(cfg.CELL_N0, t) - lr_at_update(cfg.CELL_S0, t) / 3.0) < 1e-15
        assert abs(lr_at_update(cfg.CELL_N1, t) - lr_at_update(cfg.CELL_S1, t) / 3.0) < 1e-15
    assert abs(lr_at_update(cfg.CELL_N0, t=W) - cfg.LR_MAX_1E4) < 1e-15
    assert abs(lr_at_update(cfg.CELL_N1, t=T) - cfg.LR_MIN_1E4) < 1e-15


def test_apply_lr_sets_optimizer_before_step() -> None:
    p = nn.Parameter(torch.ones(4))
    opt = make_optimizer([( "w", p)])
    lr = apply_cell_lr(opt, "S1-SMALL-COS", t=1)
    assert abs(lr - LR_MAX / W) < 1e-15
    assert all(abs(g["lr"] - lr) < 1e-15 for g in opt.param_groups)
    lr_w = apply_cell_lr(opt, "S0-SMALL-LEGACY", t=W)
    assert abs(lr_w - LR_MAX) < 1e-15


def test_ema_hand_computed_and_first_copy() -> None:
    raw = nn.Parameter(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = raw

        def trainable_parameters(self):
            return {"w": self.w}

    model = _Tiny()
    shadow = DecoderEMA(model, decay=0.9995)
    assert shadow.n_updates == 0
    # First successful step: copy, not alias.
    with torch.no_grad():
        model.w.add_(1.0)
    shadow.update_after_step(model)
    assert shadow.n_updates == 1
    assert torch.equal(shadow.shadow["w"], model.w.detach())
    assert shadow.shadow["w"].data_ptr() != model.w.data_ptr()
    # Second step: d*EMA + (1-d)*RAW
    with torch.no_grad():
        model.w.fill_(10.0)
    expected = 0.9995 * torch.tensor([2.0, 3.0, 4.0]) + 0.0005 * torch.tensor([10.0, 10.0, 10.0])
    shadow.update_after_step(model)
    assert shadow.n_updates == 2
    torch.testing.assert_close(shadow.shadow["w"], expected, atol=0.0, rtol=0.0)
    assert shadow.shadow["w"].dtype == torch.float32


def test_ema_never_trains_and_checkpoint_not_aliased() -> None:
    model = nn.Linear(3, 2)
    for p in model.parameters():
        p.requires_grad_(True)
    ema = DecoderEMA(model, decay=0.9995)
    ema.update_after_step(model)
    for name, tensor in ema.shadow.items():
        assert not tensor.requires_grad
    payload = ema.checkpoint_state()
    assert payload["shadow"]["weight"].data_ptr() != model.weight.data_ptr()
    payload["shadow"]["weight"].add_(1.0)
    assert not torch.equal(payload["shadow"]["weight"], model.weight.detach())


def test_optimizer_excludes_bias_and_norm() -> None:
    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.norm = nn.LayerNorm(4)

        def trainable_parameters(self):
            return dict(self.named_parameters())

    model = _M()
    opt = make_optimizer(model.trainable_parameters().items())
    decay = [g for g in opt.param_groups if g["weight_decay"] > 0][0]
    nodecay = [g for g in opt.param_groups if g["weight_decay"] == 0][0]
    decay_ids = {id(p) for p in decay["params"]}
    nodecay_ids = {id(p) for p in nodecay["params"]}
    assert id(model.lin.weight) in decay_ids
    assert id(model.lin.bias) in nodecay_ids
    assert id(model.norm.weight) in nodecay_ids
    assert id(model.norm.bias) in nodecay_ids
    assert opt.defaults["betas"] == (0.9, 0.999)
    assert opt.defaults["eps"] == 1.0e-8
    assert cfg.WEIGHT_DECAY == 1.0e-2
    assert cfg.GRAD_CLIP == 1.0
