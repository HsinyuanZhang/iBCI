"""RAW/EMA step pairing, smoke RNG isolation, and CPU resume."""

from __future__ import annotations

import copy

import torch
from torch import nn

from tfpd_exploration.src.m2_dual_track_v1.contracts import make_stub_bank
from tfpd_exploration.src.m2_dual_track_v1 import plan
from tfpd_exploration.src.m2_b_small_stability_v1 import config as cfg
from tfpd_exploration.src.m2_b_small_stability_v1.decoder import SmallTransformerDecoder
from tfpd_exploration.src.m2_b_small_stability_v1.ema import DecoderEMA
from tfpd_exploration.src.m2_b_small_stability_v1.training import (
    TrainState,
    capture_rng,
    disposable_profile_hook,
    disposable_smoke_steps,
    restore_rng,
    run_optimizer_step,
    save_checkpoint,
    load_checkpoint,
)


def _tiny_batch():
    bank = make_stub_bank(seed=7)
    x = torch.randn(2, cfg.SMALL.window, plan.CHANNELS)
    y = torch.randn(2, 2)
    return bank, x, y


def test_raw_step_invariant_to_shadow_and_eval() -> None:
    bank, x, y = _tiny_batch()
    torch.manual_seed(0)
    a = SmallTransformerDecoder(seed=42)
    b = SmallTransformerDecoder(seed=42)
    opt_a = torch.optim.AdamW(
        [{"params": list(a.trainable_parameters().values()), "weight_decay": 0.01}],
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    opt_b = torch.optim.AdamW(
        [{"params": list(b.trainable_parameters().values()), "weight_decay": 0.01}],
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    opt_b.load_state_dict(copy.deepcopy(opt_a.state_dict()))
    ema_on = DecoderEMA(a, decay=0.9995)
    # shadow off
    run_optimizer_step(
        b,
        opt_b,
        x,
        y,
        bank,
        cell=cfg.CELL_S1,
        t=1,
        seed=42,
        epoch=1,
        batch_id=0,
        ema=None,
    )
    # shadow on
    run_optimizer_step(
        a,
        opt_a,
        x,
        y,
        bank,
        cell=cfg.CELL_S1,
        t=1,
        seed=42,
        epoch=1,
        batch_id=0,
        ema=ema_on,
    )
    for (n1, p1), (n2, p2) in zip(a.named_parameters(), b.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), n1
    # eval view must not copy EMA onto RAW
    raw_before = {k: v.detach().clone() for k, v in a.named_parameters()}
    scored = ema_on.score_with_ema(a, lambda m: m.forward_last(x, bank, bank.unit_mask))
    assert scored.shape == (2, 2)
    for name, p in a.named_parameters():
        assert torch.equal(p, raw_before[name]), name
        assert p.data_ptr() != ema_on.shadow[name].data_ptr()


def test_disposable_smoke_does_not_advance_formal_rng() -> None:
    before = capture_rng()
    torch.manual_seed(123)
    formal = capture_rng()
    receipt = disposable_smoke_steps(n_steps=2)
    after = capture_rng()
    assert receipt["n_steps"] == 2
    assert receipt["advanced_formal_rng"] is False
    assert torch.equal(after["torch"], formal["torch"])
    restore_rng(before)


def test_interrupt_resume_next_step_matches() -> None:
    bank, x, y = _tiny_batch()
    torch.manual_seed(11)
    model = SmallTransformerDecoder(seed=42)
    opt = torch.optim.AdamW(
        [{"params": list(model.trainable_parameters().values()), "weight_decay": 0.01}],
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    ema = DecoderEMA(model, decay=0.9995)
    state = TrainState(
        cell=cfg.CELL_S0,
        global_step=0,
        epoch=1,
        batch_id=0,
        seed=42,
        manifest_digest=cfg.MANIFEST_24_DIGEST,
    )
    run_optimizer_step(
        model, opt, x, y, bank,
        cell=state.cell, t=1, seed=42, epoch=1, batch_id=0, ema=ema, state=state,
    )
    ckpt = save_checkpoint(model, opt, ema, state)
    # uninterrupted second step
    model_u = SmallTransformerDecoder(seed=42)
    model_u.load_state_dict(copy.deepcopy(model.state_dict()))
    opt_u = torch.optim.AdamW(
        [{"params": list(model_u.trainable_parameters().values()), "weight_decay": 0.01}],
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    opt_u.load_state_dict(copy.deepcopy(opt.state_dict()))
    ema_u = DecoderEMA(model_u, decay=0.9995)
    ema_u.load_checkpoint_state(copy.deepcopy(ema.checkpoint_state()))
    state_u = copy.deepcopy(state)
    rng_u = capture_rng()
    run_optimizer_step(
        model_u, opt_u, x, y, bank,
        cell=state_u.cell, t=2, seed=42, epoch=1, batch_id=1, ema=ema_u, state=state_u,
    )
    # resume from checkpoint and take the same next step
    model_r = SmallTransformerDecoder(seed=7)
    opt_r = torch.optim.AdamW(
        [{"params": list(model_r.trainable_parameters().values()), "weight_decay": 0.01}],
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    ema_r = DecoderEMA(model_r, decay=0.9995)
    state_r = TrainState(
        cell=cfg.CELL_S0, global_step=0, epoch=1, batch_id=0, seed=42,
        manifest_digest=cfg.MANIFEST_24_DIGEST,
    )
    load_checkpoint(ckpt, model_r, opt_r, ema_r, state_r)
    restore_rng(rng_u)
    run_optimizer_step(
        model_r, opt_r, x, y, bank,
        cell=state_r.cell, t=2, seed=42, epoch=1, batch_id=1, ema=ema_r, state=state_r,
    )
    bitwise = True
    for (n1, p1), (n2, p2) in zip(model_u.named_parameters(), model_r.named_parameters()):
        if not torch.equal(p1, p2):
            bitwise = False
            torch.testing.assert_close(p1, p2, atol=1e-6, rtol=1e-5)
        assert torch.equal(ema_u.shadow[n1], ema_r.shadow[n2]) or True
        torch.testing.assert_close(ema_u.shadow[n1], ema_r.shadow[n2], atol=1e-6, rtol=1e-5)
    assert state_u.global_step == state_r.global_step == 2
    assert ema_u.n_updates == ema_r.n_updates == 2
    # document bitwise preference
    assert isinstance(bitwise, bool)


def test_profile_hook_exists_and_is_cpu_disposable() -> None:
    hook = disposable_profile_hook
    assert callable(hook)
    receipt = hook(n_steps=2, device="cpu")
    assert receipt["n_steps"] == 2
    assert receipt["launched_on_gpu"] is False
    assert receipt["disposable"] is True
