"""Stage 0 correctness suite for A-QMEM (query-conditioned prefix memory).

First-round last-bin scoring uses the whole 50-bin live window as the query
feature (workorder §3.2). That is NOT a prefix-causal contract for earlier
decoder timestamps: a single E is produced from X[:, :, i] over all 50 bins
and handed to decode_with_identity; only y[:, -1, :] is scored.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tfpd_exploration.src.m2_dual_track_v1 import plan
from tfpd_exploration.src.m2_dual_track_v1.contracts import (
    CONTRACT_VERSION,
    SessionBank,
    assert_contract_version,
    make_stub_bank,
    make_stub_batch,
)


assert_contract_version(1)


class FakeStudent(nn.Module):
    """Deterministic decode: unit-sum of (neural.permute + identity).

    Matches decode_with_identity I/O: neural [B,W,N], identity [B,N,W] -> [B,W,2].
    Unit-sum is permutation-invariant so set-equivariance tests are meaningful.
    `scale` is an inherited parameter: AQMEM must freeze it (requires_grad False)
    while still letting identity carry input gradients into the residual branch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def decode_with_identity(
        self,
        neural: torch.Tensor,
        identity: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        src = neural.permute(0, 2, 1) + identity
        reduced = src.sum(dim=1) * self.scale
        return torch.stack((reduced, reduced), dim=-1)


def _expand_e0(e0: torch.Tensor, batch: int) -> torch.Tensor:
    return e0.unsqueeze(0).expand(batch, -1, -1).contiguous()


def _bank_with(
    bank: SessionBank,
    *,
    frozen_u: torch.Tensor | None = None,
    e0: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    unit_mask: torch.Tensor | None = None,
) -> SessionBank:
    return SessionBank(
        session_id=bank.session_id,
        support_trial_ids=bank.support_trial_ids,
        raw_trial_ids=bank.raw_trial_ids,
        X_store=bank.X_store,
        target_store=bank.target_store,
        eligible_starts=bank.eligible_starts,
        E0=bank.E0 if e0 is None else e0,
        T=bank.T if t is None else t,
        unit_mask=bank.unit_mask if unit_mask is None else unit_mask,
        provenance=dict(bank.provenance),
        frozen_u=bank.frozen_u if frozen_u is None else frozen_u,
    )


def _make_model(*, seed: int = 0, student: FakeStudent | None = None):
    from tfpd_exploration.src.m2_dual_track_v1.calibration_memory import AQMEM

    torch.manual_seed(seed)
    student = FakeStudent() if student is None else student
    return AQMEM(student)


def _live_w_out(model) -> None:
    torch.nn.init.normal_(model.W_out.weight, mean=0.0, std=0.15)


def _baseline_last(student: FakeStudent, x: torch.Tensor, e0: torch.Tensor) -> torch.Tensor:
    identity = _expand_e0(e0, x.shape[0])
    return student.decode_with_identity(x, identity)[:, -1, :]


def test_contract_version_is_1() -> None:
    assert CONTRACT_VERSION == 1
    assert plan.CONTRACT_VERSION == 1
    assert plan.A_HEADS == 2
    assert plan.A_HEAD_DIM == 16
    assert plan.A_VALUE_DIM == 16
    assert plan.TRAINING_TARGET_SPACE == "decoder_raw"


def test_zero_w_out_predictions_bitwise_equal_e0_baseline() -> None:
    """W_out=0 (init): FP32 same-device same-kernel preds bitwise equal E0-only.

    Signed-zero handling: if Linear(zeros) @ GELU(r) yields -0 where a bin of
    E0 is +0, canonicalize with `y + 0.0` (IEEE +0 + -0 = +0) before
    torch.equal. Documented here so a raw bitwise miss is not treated as a
    residual leak.
    """
    bank = make_stub_bank(num_units=8, k=5, seed=11)
    batch = make_stub_batch(bank, batch_size=3, seed=12)
    model = _make_model(seed=13)
    assert torch.equal(model.W_out.weight, torch.zeros_like(model.W_out.weight))
    assert model.W_out.bias is None

    y = model.forward_last(batch.X, batch.bank, batch.unit_mask)
    y_base = _baseline_last(model.frozen_student, batch.X, batch.bank.E0)
    y_can = y + 0.0
    y_base_can = y_base + 0.0
    assert torch.equal(y_can, y_base_can)
    # Native E0 is used as-is: identities with zero residual equal expanded E0.
    e = model.identities(batch.X, batch.bank, batch.unit_mask)
    e0_exp = _expand_e0(batch.bank.E0, batch.X.shape[0])
    assert torch.equal(e + 0.0, e0_exp + 0.0)


def test_uniform_or_k1_or_identical_memory_residual_is_zero() -> None:
    """K=1 / identical memory rows / forced uniform attention => residual == 0.

    Centered residual: sum_j (w_j - 1/K) v_j. Uniform w or K=1 makes the
    coefficient zero. Identical rows make v = V(0) = 0 (V has no bias).
    FP32 exact: atol 0. If a future kernel is inexact, document 1e-6 here.
    """
    model = _make_model(seed=21)
    _live_w_out(model)

    bank_k1 = make_stub_bank(num_units=6, k=1, seed=22)
    x1 = make_stub_batch(bank_k1, batch_size=2, seed=23).X
    delta_k1 = model.compute_delta(x1, bank_k1, bank_k1.unit_mask)
    assert torch.equal(delta_k1, torch.zeros_like(delta_k1))

    bank = make_stub_bank(num_units=6, k=7, seed=24)
    x = make_stub_batch(bank, batch_size=2, seed=25).X
    identical_u = bank.frozen_u[:1].expand_as(bank.frozen_u).contiguous()
    bank_id = _bank_with(bank, frozen_u=identical_u)
    delta_id = model.compute_delta(x, bank_id, bank.unit_mask)
    assert torch.equal(delta_id, torch.zeros_like(delta_id))

    delta_u = model.compute_delta(
        x, bank, bank.unit_mask, force_uniform_attention=True
    )
    assert torch.equal(delta_u, torch.zeros_like(delta_u))


def test_nonzero_w_out_memory_or_query_changes_output() -> None:
    bank = make_stub_bank(num_units=8, k=5, seed=31)
    batch = make_stub_batch(bank, batch_size=3, seed=32)
    model = _make_model(seed=33)
    _live_w_out(model)

    y0 = model.forward_last(batch.X, batch.bank, batch.unit_mask)
    d0 = model.compute_delta(batch.X, batch.bank, batch.unit_mask)

    mutated_u = batch.bank.frozen_u.clone()
    mutated_u[0] = mutated_u[0] + 1.7
    bank_m = _bank_with(batch.bank, frozen_u=mutated_u)
    y_mem = model.forward_last(batch.X, bank_m, batch.unit_mask)
    assert not torch.equal(y0, y_mem)
    assert not torch.allclose(y0, y_mem, atol=0.0, rtol=0.0)

    x_q = batch.X.clone()
    x_q[:, :, 0] = x_q[:, :, 0] + 2.3
    d_q = model.compute_delta(x_q, batch.bank, batch.unit_mask)
    assert not torch.allclose(d0, d_q, atol=0.0, rtol=0.0)


def test_two_optimizer_updates_w_out_then_qkv_grads() -> None:
    """W_out receives grad on the first backward; Q/K/V after the second update.

    Do not wrap decode in no_grad(). Inherited student params stay frozen
    (requires_grad False) while identity input grads still flow to W_out.
    Real-shaped stub: N=96, K=33, W=50.
    """
    bank = make_stub_bank(num_units=plan.CHANNELS, k=plan.SUPPORT_HORIZON, seed=41)
    batch = make_stub_batch(bank, batch_size=2, seed=42)
    student = FakeStudent()
    model = _make_model(seed=43, student=student)
    model.train()
    assert student.training is False
    assert student.scale.requires_grad is False

    params = model.trainable_parameters()
    opt = torch.optim.AdamW(
        params.values(),
        lr=plan.A_LR,
        betas=plan.ADAM_BETAS,
        eps=plan.ADAM_EPS,
        weight_decay=plan.A_WEIGHT_DECAY,
    )

    def _backward() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        pred = model.forward_last(batch.X, batch.bank, batch.unit_mask)
        loss = torch.mean((pred - batch.last_target) ** 2)
        loss.backward()
        return pred

    _backward()
    w_grad = params["W_out.weight"].grad
    assert w_grad is not None
    assert torch.isfinite(w_grad).all()
    assert float(w_grad.abs().sum()) > 0.0
    assert student.scale.grad is None

    opt.step()
    _backward()

    for key in params:
        if key.split(".")[0] not in {"Q", "K", "V"}:
            continue
        grad = params[key].grad
        assert grad is not None, key
        assert torch.isfinite(grad).all(), key
        assert float(grad.abs().sum()) > 0.0, key

    opt.step()


def test_unit_permutation_leaves_prediction_unchanged() -> None:
    bank = make_stub_bank(num_units=10, k=6, seed=51)
    batch = make_stub_batch(bank, batch_size=3, seed=52)
    model = _make_model(seed=53)
    _live_w_out(model)
    model.eval()

    y = model.forward_last(batch.X, batch.bank, batch.unit_mask)
    perm = torch.randperm(bank.E0.shape[0], generator=torch.Generator().manual_seed(54))
    x_p = batch.X[:, :, perm]
    bank_p = _bank_with(
        bank,
        frozen_u=bank.frozen_u[:, perm, :].contiguous(),
        e0=bank.E0[perm].contiguous(),
        t=bank.T[perm].contiguous(),
        unit_mask=bank.unit_mask[perm].contiguous(),
    )
    y_p = model.forward_last(x_p, bank_p, bank_p.unit_mask)
    torch.testing.assert_close(y, y_p, atol=plan.PERM_ATOL, rtol=plan.PERM_RTOL)


def test_memory_trial_permutation_is_set_invariance_not_negative_control() -> None:
    """Synchronous K/V trial-order permutation must leave the prediction unchanged.

    This is set invariance of attention over the legal prefix. It is not a
    negative control; breaking K/V pairing would be a later ablation.
    """
    bank = make_stub_bank(num_units=8, k=9, seed=61)
    batch = make_stub_batch(bank, batch_size=3, seed=62)
    model = _make_model(seed=63)
    _live_w_out(model)
    model.eval()

    y = model.forward_last(batch.X, batch.bank, batch.unit_mask)
    perm = torch.randperm(bank.frozen_u.shape[0], generator=torch.Generator().manual_seed(64))
    bank_p = _bank_with(bank, frozen_u=bank.frozen_u[perm].contiguous())
    y_p = model.forward_last(batch.X, bank_p, batch.unit_mask)
    torch.testing.assert_close(y, y_p, atol=plan.PERM_ATOL, rtol=plan.PERM_RTOL)


def test_last_bin_query_uses_full_window_no_prefix_causality_claim() -> None:
    """Last-bin contract: score the last timestamp only; query uses all 50 bins.

    First-round A-QMEM does not expose per-timestep causal E. Perturbing any
    bin in the live window (including bin 0 or bin 49) may change y[:, -1]
    through the query projection. We do not claim that earlier decoder
    outputs are prefix-causal. No causal-query helper is implemented in
    this round; if one is added later, test future-bin isolation there.
    """
    bank = make_stub_bank(num_units=6, k=4, seed=71)
    batch = make_stub_batch(bank, batch_size=2, seed=72)
    model = _make_model(seed=73)
    _live_w_out(model)

    d0 = model.compute_delta(batch.X, batch.bank, batch.unit_mask)
    x_early = batch.X.clone()
    x_early[:, 0, :] = x_early[:, 0, :] + 3.1
    x_late = batch.X.clone()
    x_late[:, -1, :] = x_late[:, -1, :] + 3.1
    d_early = model.compute_delta(x_early, batch.bank, batch.unit_mask)
    d_late = model.compute_delta(x_late, batch.bank, batch.unit_mask)
    assert not torch.allclose(d0, d_early)
    assert not torch.allclose(d0, d_late)

    y = model.forward_last(batch.X, batch.bank, batch.unit_mask)
    assert y.shape == (batch.X.shape[0], plan.OUT_DIM)


def test_trainable_allowlist_omits_frozen_student() -> None:
    student = FakeStudent()
    model = _make_model(seed=81, student=student)
    params = model.trainable_parameters()
    roots = {name.split(".")[0] for name in params}
    assert roots == {"query_proj", "Q", "K", "V", "W_out"}
    assert "W_out.weight" in params
    assert "V.weight" in params
    assert all(param is not student.scale for param in params.values())
    assert all(p.requires_grad for p in params.values())
    assert student.scale.requires_grad is False
    assert model.V.bias is None
    assert model.W_out.bias is None
    assert model.name == "A-QMEM"
    assert model.training_target_space == plan.TRAINING_TARGET_SPACE
    for name, value in student.named_parameters():
        assert value.requires_grad is False, name
        assert f"frozen_student.{name}" not in params
        assert name not in params
