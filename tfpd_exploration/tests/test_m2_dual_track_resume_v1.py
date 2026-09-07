"""Resume contract: 24-epoch manifest prefix and interrupted/uninterrupted parity."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from tfpd_exploration.src.m2_dual_track_v1 import launch, plan, sampler, training


def test_twenty_four_manifest_keeps_first_twelve_batches() -> None:
    lengths = {session: 40 + i for i, session in enumerate(plan.HELDIN_SESSIONS)}
    twelve = sampler.build_shuffled_manifest(lengths, batch_size=8, epochs=12, sampler_seed=sampler.SAMPLER_SEED)
    twenty_four = sampler.extend_shuffled_manifest(twelve, epochs=24)
    assert twenty_four["epochs"] == 24
    assert twenty_four["parent_12_digest"] == twelve["digest"]
    assert twenty_four["digest"] != twelve["digest"]
    for epoch in range(1, 13):
        assert twenty_four["batches"][str(epoch)] == twelve["batches"][str(epoch)]
    assert "13" in twenty_four["batches"] and "24" in twenty_four["batches"]


def test_cosine_tail_matches_declared_24ep_law() -> None:
    opt = torch.optim.AdamW([nn.Parameter(torch.zeros(2))], lr=plan.B_LR)
    lr13 = training.apply_b_lr(opt, plan.B_LR, global_step=37980, warmup_steps=3165, epoch=13, max_epochs=24)
    lr24 = training.apply_b_lr(opt, plan.B_LR, global_step=75960, warmup_steps=3165, epoch=24, max_epochs=24)
    assert abs(lr13 - plan.B_LR) < 1e-12
    assert abs(lr24 - 0.1 * plan.B_LR) < 1e-12


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 2)

    def trainable_parameters(self):
        return dict(self.named_parameters())


def test_interrupted_resume_matches_uninterrupted_step() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.randn(4, 8)
    y = torch.randn(4, 2)
    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=plan.ADAM_BETAS, eps=plan.ADAM_EPS)

    def step() -> tuple[float, torch.Tensor]:
        model.train()
        drop = torch.rand(4, 8)
        pred = model.lin(x * (drop > 0.1).float())
        loss = nn.functional.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(loss.detach()), pred.detach().clone()

    step()
    ckpt = {
        "schema": launch.FULL_CKPT_SCHEMA,
        "epoch_one_based": 2,
        "global_step": 10,
        "state_dict": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(opt.state_dict()),
        "rng": launch._capture_rng(),
        "sampler": {"completed_epoch": 2, "manifest_digest": "toy"},
        "param_names": list(model.trainable_parameters()),
    }
    loss_u, pred_u = step()
    w_u = {k: v.detach().clone() for k, v in model.state_dict().items()}
    mom_u = copy.deepcopy(opt.state_dict())

    model.load_state_dict(ckpt["state_dict"])
    opt.load_state_dict(ckpt["optimizer"])
    launch.assert_optimizer_named_correspondence(model, opt, ckpt)
    launch._restore_rng(ckpt["rng"])
    loss_r, pred_r = step()
    assert abs(loss_u - loss_r) < 1e-7
    torch.testing.assert_close(pred_u, pred_r, atol=1e-6, rtol=1e-5)
    for key, value in w_u.items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=1e-6, rtol=1e-5)
    assert mom_u["state"].keys() == opt.state_dict()["state"].keys()


def test_real_epoch12_is_full_resume_payload() -> None:
    root = plan.active_run_root()
    path = root / "arms" / "B-TRANSFORMER" / "seed42_shuffled" / "epoch_012.pt"
    payload = launch.inspect_full_checkpoint(path)
    assert payload["schema"] == launch.FULL_CKPT_SCHEMA
    assert payload["epoch_one_based"] == 12
    assert payload["global_step"] == 37980
    assert payload["sampler"]["completed_epoch"] == 12
    assert payload["has_optimizer"] is True
    assert payload["has_rng"] is True
    assert payload["n_optimizer_states"] == 77
