from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.tfsr_b3st4_ddrop_v1.model as tfsr_model
from src.tfpd.bilinear_readin import CausalActivityEncoder
from src.tfsr_b3st4_ddrop_v1.model import NormalizedT4Batch, T4Normalizer, TFSRDecoder, ordered_unit_digest


RAW_SHA = "a" * 64
NORMALIZER_SHA = "b" * 64
ROSTER_SHA = "c" * 64


def fixture(n: int = 3, batch: int = 1):
    torch.manual_seed(3)
    x = torch.randn(batch, 50, n)
    calib = torch.randn(batch, 30, 100, n)
    raw = torch.randn(batch, n, 4)
    ids = tuple(f"unit-{index}" for index in range(n))
    normalizer = T4Normalizer(torch.zeros(4), torch.ones(4), RAW_SHA, NORMALIZER_SHA)
    capability = normalizer(raw, roster_digest=ROSTER_SHA, ordered_unit_ids=ids, lineage=("strict-27", "m30"))
    return x, calib, raw, capability, normalizer


def test_uses_exact_shared_activity_encoder_and_no_tables_or_forbidden_modules():
    model = TFSRDecoder()
    assert isinstance(model.activity_encoder, CausalActivityEncoder)
    source = inspect.getsource(tfsr_model)
    assert "class CausalActivityEncoder" not in source
    assert "from src.tfpd.bilinear_readin import CausalActivityEncoder" in source
    assert not any("embedding" in name.lower() or any(word in name.lower() for word in ("session", "subject", "site", "dataset")) for name, _ in model.named_parameters())
    for forbidden in ("nn.Embedding", "torch.cuda", "DataLoader", "pandas", "numpy"):
        assert forbidden not in source


def test_capability_authority_controls_and_joint_permutation():
    x, calib, raw, capability, normalizer = fixture()
    model = TFSRDecoder()
    model.eval()
    aligned = model(x, calib, capability)
    zero = normalizer.zero(capability)
    wrong = normalizer.wrong_pair(capability)
    assert zero.diagnostic_mode == "zero" and wrong.diagnostic_mode == "wrong_pair"
    assert torch.isfinite(model(x, calib, zero)).all()
    assert torch.isfinite(model(x, calib, wrong)).all()
    permutation = torch.tensor([2, 0, 1], dtype=torch.long)
    permuted_capability = capability.joint_permute(permutation)
    assert permuted_capability.diagnostic_mode == "aligned"
    assert permuted_capability.ordered_unit_ids == ("unit-2", "unit-0", "unit-1")
    assert torch.allclose(aligned, model(x[:, :, permutation], calib[:, :, :, permutation], permuted_capability), atol=1e-6, rtol=1e-6)
    assert torch.equal(capability.tensor, (raw - normalizer.mean) / normalizer.std)


def test_exact_cell_d_rng_and_fused_token_mask_once(monkeypatch):
    x, calib, _, capability, _ = fixture(n=3)
    model = TFSRDecoder(0.2, 0.4, capture_diagnostics=True)
    model.train()
    uniform_calls: list[tuple[float, float]] = []
    dropout_calls: list[tuple[tuple[int, ...], float, bool]] = []

    def fake_uniform(low: float, high: float) -> float:
        uniform_calls.append((low, high))
        return 0.25

    def fake_dropout(mask: torch.Tensor, p: float, training: bool) -> torch.Tensor:
        dropout_calls.append((tuple(mask.shape), p, training))
        return mask.new_tensor([[0.0, 1.0 / (1.0 - p), 0.0]])

    monkeypatch.setattr(tfsr_model.random, "uniform", fake_uniform)
    monkeypatch.setattr(tfsr_model.F, "dropout", fake_dropout)
    prediction = model(x, calib, capability)
    assert prediction.shape == (1, 50, 2)
    assert uniform_calls == [(0.2, 0.4)]
    assert dropout_calls == [((1, 3), 0.25, True)]
    assert torch.equal(model.last_dropout_p, torch.tensor(0.25))
    assert torch.equal(model.last_unit_gain_mask, torch.tensor([[0.0, 4.0 / 3.0, 0.0]]))
    assert torch.equal(model.last_unit_survivor_mask, torch.tensor([[False, True, False]]))
    assert torch.equal(model.last_post_mask_tokens, model.last_pre_mask_tokens * model.last_unit_gain_mask[:, None, :, None])
    expected_activity_mass = model.last_activity_features[:, :, 1].abs().sum(dim=-1) / 64
    assert torch.equal(model.last_activity_mass, expected_activity_mass)
    # The single [B,N] gain is broadcast over the complete 50-bin fused-token window.
    assert torch.equal(model.last_post_mask_tokens[:, :, 0], torch.zeros_like(model.last_post_mask_tokens[:, :, 0]))
    assert torch.equal(model.last_post_mask_tokens[:, :, 2], torch.zeros_like(model.last_post_mask_tokens[:, :, 2]))


def test_dropped_unit_never_changes_mass_readin_prediction_or_state(monkeypatch):
    x, calib, raw, capability, normalizer = fixture(n=2)
    model = TFSRDecoder(0.0, 0.0, capture_diagnostics=True)
    model.train()

    def fake_dropout(mask: torch.Tensor, p: float, training: bool) -> torch.Tensor:
        assert mask.shape == (1, 2)
        return mask.new_tensor([[0.0, 1.0]])

    monkeypatch.setattr(tfsr_model.random, "uniform", lambda low, high: 0.0)
    monkeypatch.setattr(tfsr_model.F, "dropout", fake_dropout)
    before, before_states = model(x, calib, capability, return_states=True)
    before_mass, before_readin = model.last_mass.clone(), model.last_readin.clone()
    before_pre_mask_tokens = model.last_pre_mask_tokens.clone()
    changed_x, changed_calib, changed_raw = x.clone(), calib.clone(), raw.clone()
    changed_x[:, :, 0] += 1_000_000
    changed_calib[:, :, :, 0] -= 1_000_000
    changed_raw[:, 0] *= -1_000_000
    changed_capability = normalizer(
        changed_raw,
        roster_digest=ROSTER_SHA,
        ordered_unit_ids=capability.ordered_unit_ids,
        lineage=capability.lineage,
    )
    after, after_states = model(changed_x, changed_calib, changed_capability, return_states=True)
    assert not torch.equal(before_pre_mask_tokens, model.last_pre_mask_tokens)
    assert torch.equal(before_mass, model.last_mass)
    assert torch.equal(before_readin, model.last_readin)
    assert torch.equal(before, after)
    assert torch.equal(before_states, after_states)


def test_prefix_causality_at_several_boundaries_and_state_reset():
    x, calib, _, capability, _ = fixture(n=2)
    model = TFSRDecoder()
    model.eval()
    reference, reference_states = model(x, calib, capability, return_states=True)
    for boundary in (1, 7, 20, 49):
        changed = x.clone()
        changed[:, boundary:] = torch.randn_like(changed[:, boundary:]) * 100
        prediction, states = model(changed, calib, capability, return_states=True)
        assert torch.equal(reference[:, :boundary], prediction[:, :boundary])
        assert torch.equal(reference_states[:, :boundary], states[:, :boundary])
    reset_prediction, reset_states = model(x, calib, capability, return_states=True)
    assert torch.equal(reference, reset_prediction)
    assert torch.equal(reference_states, reset_states)


def test_variable_n_finite_forward_backward_and_eval_draws_no_rng(monkeypatch):
    for n in (1, 4):
        x, calib, _, capability, _ = fixture(n=n)
        model = TFSRDecoder(0.0, 0.0)
        model.train()
        prediction = model(x, calib, capability)
        loss = model.dense_valid_bin_mse(prediction, torch.zeros_like(prediction), torch.ones(1, 50, dtype=torch.bool))
        loss.backward()
        assert torch.isfinite(prediction).all() and torch.isfinite(loss)
        critical = ("b3s.pre_pool.0.weight", "activity_encoder.net.0.weight", "unit_mlp.0.weight", "state_query.weight", "attn.in_proj_weight", "gru.weight_ih_l0", "head.weight")
        named = dict(model.named_parameters())
        assert all(named[name].grad is not None and torch.isfinite(named[name].grad).all() and named[name].grad.abs().sum() > 0 for name in critical)
    x, calib, _, capability, _ = fixture(n=2)
    model = TFSRDecoder()
    model.eval()
    monkeypatch.setattr(tfsr_model.random, "uniform", lambda low, high: pytest.fail("eval must not draw Python RNG"))
    model(x, calib, capability)
    assert model.last_dropout_p is None
    assert torch.equal(model.last_unit_gain_mask, torch.ones_like(model.last_unit_gain_mask))
    assert torch.equal(model.last_unit_survivor_mask, torch.ones_like(model.last_unit_survivor_mask, dtype=torch.bool))


def test_default_diagnostics_are_small_only_and_capture_is_output_neutral():
    x, calib, _, capability, _ = fixture(n=2)
    default = TFSRDecoder()
    captured = TFSRDecoder(capture_diagnostics=True)
    captured.load_state_dict(default.state_dict())
    default.eval()
    captured.eval()
    default_prediction, default_states = default(x, calib, capability, return_states=True)
    captured_prediction, captured_states = captured(x, calib, capability, return_states=True)
    assert torch.equal(default_prediction, captured_prediction)
    assert torch.equal(default_states, captured_states)
    assert default.last_unit_gain_mask is not None and default.last_unit_survivor_mask is not None
    for attribute in (
        "last_pre_mask_tokens", "last_post_mask_tokens", "last_activity_features", "last_activity_mass",
        "last_mass", "last_readin", "last_prediction", "last_states",
    ):
        assert getattr(default, attribute) is None
        assert getattr(captured, attribute) is not None


def test_malformed_capability_input_order_sha_and_config_reject():
    x, calib, raw, capability, normalizer = fixture(n=3)
    model = TFSRDecoder()
    with pytest.raises(ValueError):
        model(x, calib, capability.tensor)
    with pytest.raises(ValueError):
        model(x, calib[:, :, :, :2], capability)
    with pytest.raises(ValueError):
        normalizer(torch.full_like(raw, float("nan")), roster_digest=ROSTER_SHA, ordered_unit_ids=capability.ordered_unit_ids, lineage=capability.lineage)
    with pytest.raises(ValueError):
        T4Normalizer(torch.zeros(4, dtype=torch.long), torch.ones(4), RAW_SHA, NORMALIZER_SHA)
    with pytest.raises(ValueError):
        T4Normalizer(torch.zeros(4), torch.ones(4), "raw", NORMALIZER_SHA)
    with pytest.raises(ValueError):
        NormalizedT4Batch(capability.tensor, RAW_SHA, NORMALIZER_SHA, ROSTER_SHA, "0" * 64, capability.ordered_unit_ids, capability.lineage)
    with pytest.raises(ValueError):
        NormalizedT4Batch(capability.tensor, RAW_SHA, NORMALIZER_SHA, ROSTER_SHA, ordered_unit_digest(("duplicate", "duplicate", "third")), ("duplicate", "duplicate", "third"), capability.lineage)
    with pytest.raises(ValueError):
        NormalizedT4Batch(capability.tensor, RAW_SHA, NORMALIZER_SHA, ROSTER_SHA, capability.ordered_unit_digest, capability.ordered_unit_ids[:-1], capability.lineage)
    with pytest.raises(ValueError):
        capability.joint_permute(torch.tensor([0, 1, 1], dtype=torch.long))
    with pytest.raises(ValueError):
        normalizer.wrong_pair(capability, torch.tensor([0, 1, 2], dtype=torch.long))
    for bounds in ((-0.1, 0.2), (0.8, 0.7), (0.0, 1.1), (float("nan"), 0.1), (True, 0.1)):
        with pytest.raises(ValueError):
            TFSRDecoder(*bounds)
    for invalid_capture in (0, 1, "true", None):
        with pytest.raises(ValueError):
            TFSRDecoder(capture_diagnostics=invalid_capture)


def test_dense_valid_bin_mse_rejects_bad_masks_nonfinite_and_empty_mask():
    prediction = torch.zeros(1, 50, 2)
    target = torch.ones_like(prediction)
    valid = torch.ones(1, 50, dtype=torch.bool)
    assert TFSRDecoder.dense_valid_bin_mse(prediction, target, valid).item() == pytest.approx(1.0)
    for bad_valid in (torch.zeros(1, 50, dtype=torch.bool), torch.full((1, 50), 0.5), torch.full((1, 50), 2, dtype=torch.long), torch.ones(1, 50, dtype=torch.complex64)):
        with pytest.raises(ValueError):
            TFSRDecoder.dense_valid_bin_mse(prediction, target, bad_valid)
    with pytest.raises(ValueError):
        TFSRDecoder.dense_valid_bin_mse(torch.full_like(prediction, float("nan")), target, valid)
    with pytest.raises(ValueError):
        TFSRDecoder.dense_valid_bin_mse(prediction, torch.full_like(target, float("inf")), valid)


def test_accounting_parameter_ceiling_and_latency_preserves_mode():
    x, calib, _, capability, _ = fixture(n=2)
    model = TFSRDecoder()
    accounting = model.accounting()
    assert accounting["trainable_params"] <= 3_600_000
    assert accounting["state_bytes"] == 1024
    assert "analytic_mac_estimate_n" in accounting and "analytic_activation_estimate_bytes" in accounting
    model.train()
    latency = model.cpu_latency_ms(x, calib, capability, warmup=1, repeats=3)
    assert latency >= 0 and model.training
    with pytest.raises(ValueError):
        model.cpu_latency_ms(x, calib, capability, warmup=0, repeats=3)
    with pytest.raises(ValueError):
        model.cpu_latency_ms(x, calib, capability, warmup=1, repeats=2)
