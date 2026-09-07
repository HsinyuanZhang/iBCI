"""Tests for the matched-capacity NON-equivariant control (arm C-prime).

The operator review found the C - A gate confounded by construction: exact
equivariance disconnected/froze 3,203,684 ordinary-consumer parameters, so arm
C trains 492,951 parameters vs sealed Cell D's 3,510,842.  C-prime is the
matched-capacity control.  These tests pin its two defining properties:

- EXACT capacity parity with arm C: same consumer class and parameter count
  (185,793), same state keys/shapes, same frozen inactive modules, same
  trainable set (492,951);
- the rotation property FAILS: the carrier view is the ordinary per-component
  z-scored raw (a, c) (source-fit statistics), which provably does not commute
  with rotation, so f(x, R beta) != R f(x, beta) by a gross margin (the mirror
  of the arm-C commutation/equivariance proofs).

Plus the carrier-view layout, fused-path blindness, attention-logit
non-invariance, smoke-level train-step finiteness, receipt/gate conventions,
and the runner CLI contract.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tfpd_lane import arm_common, equivariant_cell as eq
from src.tfpd_lane import equivariant_control as ctl

CANONICAL = ROOT / "results/admission_arms_v1/canonical_initial_state.pt"
AUTHORITY = ROOT / "results/equivariant_v1/carrier_authority/carrier_authority.pt"


def _canonical_state():
    payload = torch.load(CANONICAL, map_location="cpu", weights_only=False)
    return payload["state_dict"], payload["state_sha256"]


@pytest.fixture(scope="module")
def cprime_model():
    canonical, _ = _canonical_state()
    model = ctl.build_matched_control_model(seed=42)
    eq.load_canonical_prefix(model, canonical)
    eq.freeze_inactive_consumer_modules(model)
    model.eval()
    return model


def _synthetic_batch(batch: int, n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    neural = torch.rand((batch, 50, n), generator=g)
    calib = torch.rand((batch, 30, 100, n), generator=g)
    side = torch.randn((batch, n, 4), generator=g)
    behavior = torch.randn((batch, 50, 2), generator=g)
    behavior[0, 40:] = -1.0
    return neural, behavior, calib, side


def _control_bundle(n: int, seed: int, m_scale: float = 1.3):
    """A synthetic carrier pushed through the ORDINARY per-component view."""
    rng = np.random.default_rng(seed)
    phi = rng.uniform(-np.pi, np.pi, n)
    m = np.abs(rng.normal(1.3, 0.5, n))
    b = rng.normal(10.0, 3.0, n)
    n_invalid = max(1, n // 10)
    m[:n_invalid] = 0.0
    raw = np.stack(
        [m * np.cos(phi) * (m > 0), m * np.sin(phi) * (m > 0), m, b], axis=1
    ).astype(np.float64)
    mean_ac = raw[:, 0:2].mean(axis=0)  # per-component statistics (the trap)
    std_ac = raw[:, 0:2].std(axis=0)
    return ctl.MatchedControlCarrierBundle.from_raw(
        torch.from_numpy(raw),
        torch.from_numpy(mean_ac),
        torch.from_numpy(std_ac),
        torch.from_numpy(rng.normal(0, 1, n).astype(np.float32)),
        torch.from_numpy(rng.normal(0, 1, n).astype(np.float32)),
        m_scale,
    )


# ---- exact capacity parity with arm C ---------------------------------------
def test_cprime_parameter_parity_with_arm_c_exact():
    canonical, canonical_sha = _canonical_state()
    model_c = eq.build_equivariant_model(seed=42)
    eq.load_canonical_prefix(model_c, canonical)
    eq.freeze_inactive_consumer_modules(model_c)
    model_p = ctl.build_matched_control_model(seed=42)
    eq.load_canonical_prefix(model_p, canonical)
    eq.freeze_inactive_consumer_modules(model_p)
    # identical graphs: state keys and shapes match key-for-key
    assert sorted(model_c.state_dict().keys()) == sorted(model_p.state_dict().keys())
    parity = ctl.matched_parameter_parity(model_p, model_c)
    assert parity["state_keys_equal"] and parity["state_shapes_equal"]
    assert parity["consumer_parameters_equal"] is True
    assert parity["total_trainable_equal"] is True
    # the frozen numbers, exactly arm C's
    assert parity["consumer_parameters"] == 185_793
    assert parity["total_trainable"] == 492_951
    assert parity["frozen_canonical_parameters"] == 3_203_684
    assert parity["canonical_tensors_strict_loaded"] == 3_510_842
    assert canonical_sha == _canonical_state()[1]  # artifact untouched
    # both strict-load the SAME canonical bytes
    assert eq.subset_state_sha256(
        model_p.state_dict(), canonical.keys()
    ) == canonical_sha


def test_cprime_uses_the_same_consumer_class():
    consumer = ctl.build_matched_control_model(seed=42).consumer
    assert isinstance(consumer, eq.EquivariantCarrierConsumer)
    assert consumer.num_heads == 2 and consumer.head_key_dim == 64
    assert consumer.iota_dim == 6  # the z-scored pair replaces rho_cos/rho_sin


# ---- the ordinary (non-covariant) carrier view ------------------------------
def test_cprime_carrier_view_layout_and_z_scoring():
    bundle = _control_bundle(36, seed=17)
    raw = bundle.raw.numpy()
    mean_ac, std_ac = bundle.mean_ac.numpy(), bundle.std_ac.numpy()
    expected = (raw[:, 0:2] - mean_ac) / std_ac
    np.testing.assert_allclose(bundle.value_two_vector().numpy(), expected, atol=1e-12)
    features = bundle.invariant_features().numpy()
    assert features.shape == (36, 6)
    # layout: [a_z, c_z, m_norm, b_norm, log1p(|beta|/m_scale), valid]
    np.testing.assert_allclose(features[:, 0:2], expected.astype(np.float32), atol=1e-6)
    np.testing.assert_allclose(
        features[:, 4],
        np.log1p(np.hypot(raw[:, 0], raw[:, 1]) / 1.3).astype(np.float32),
        atol=1e-6,
    )
    assert set(np.unique(features[:, 5])) <= {0.0, 1.0}
    # invalid units keep the validity flag 0 and still carry their z-scores
    assert (~bundle.valid.numpy()).sum() >= 1


def test_cprime_view_does_not_commute_with_rotation():
    """The mirror of the commutation proof: the ordinary z-scored view moves
    under rotation in a NON-covariant way, and the invariant scalars stay."""
    bundle = _control_bundle(50, seed=23)
    base_features = bundle.invariant_features()
    worst_value = 0.0
    for theta in (0.6, -1.4, 2.5):
        rot = eq.rotation_matrix_2d(theta)
        rotated = bundle.rotated(theta)
        expected_covariant = (
            bundle.value_two_vector().numpy() @ rot.T
        )
        worst_value = max(
            worst_value,
            float(
                np.abs(
                    rotated.value_two_vector().numpy() - expected_covariant
                ).max()
            ),
        )
        # the shared invariant scalars (m, b, log-scale, validity) do NOT move
        rotated_features = rotated.invariant_features()
        torch.testing.assert_close(
            rotated_features[:, 2:], base_features[:, 2:], rtol=0, atol=1e-6
        )
        # ... while the z-scored pair does move (and not covariantly)
        assert not torch.equal(rotated_features[:, 0:2], base_features[:, 0:2])
        assert bool(torch.equal(rotated.valid, bundle.valid))
    assert worst_value > 1e-2, worst_value  # z(R beta) != R z(beta), grossly


def test_cprime_is_not_rotation_equivariant(cprime_model):
    """f(x, R beta) != R f(x, beta) by a gross margin on a fixed fixture —
    the deliberate mirror of arm C's 1e-5 equivariance property."""
    model = cprime_model
    neural, _b, calib, side = _synthetic_batch(2, 40, 71)
    bundle = _control_bundle(40, 72)
    model.set_carrier(bundle)
    probe = ctl.nonequivariance_probe(
        model, neural, calib, side, [0.0, 0.9, -1.6, 2.7, 3.1]
    )
    assert probe["identity_rotation_bitwise_equal"] is True  # own baseline
    assert probe["equivariance_claim"] is False
    assert probe["max_violation"] > 0.1, probe["max_violation"]
    assert probe["relative_max_violation"] > 0.5, probe["relative_max_violation"]
    # every nonzero rotation violates (no accidental equivariant direction)
    nonzero = [v for t, v in zip([0.0, 0.9, -1.6, 2.7, 3.1],
                                 probe["per_rotation_violations"]) if t != 0.0]
    assert min(nonzero) > 1e-3, nonzero


def test_cprime_attention_logits_are_not_rotation_invariant(cprime_model):
    """The mirror of arm C's invariant-logits proof: because the z-scored pair
    feeds the keys, the attention weights MOVE under rotation."""
    model = cprime_model
    neural, _b, calib, side = _synthetic_batch(2, 33, 73)
    bundle = _control_bundle(33, 74)
    model.set_carrier(bundle)
    model.eval()
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=torch.zeros_like(side))
        tokens = model.decoder.fc_in(neural.permute(0, 2, 1) + identity)
        queries = model.decoder.fc_in(model.decoder.rep).to(tokens)
        queries = queries.repeat(tokens.size(0), 1, 1)
        w_base = model.consumer.attention_weights(tokens, queries, bundle)
        w_rot = model.consumer.attention_weights(
            tokens, queries, bundle.rotated(1.11)
        )
        assert torch.equal(
            model.consumer.attention_weights(tokens, queries, bundle.rotated(0.0)),
            w_base,
        )
    assert float((w_rot - w_base).abs().max().item()) > 1e-4


def test_cprime_fused_path_blind_to_all_side_columns(cprime_model):
    """The wrapper feeds the B3S encoder zeros_like(side): no normalized side
    column — (a,c) or (m,b) — can move the output by a single bit."""
    model = cprime_model
    neural, _b, calib, side = _synthetic_batch(2, 21, 75)
    model.set_carrier(_control_bundle(21, 76))
    model.eval()
    alt = side.clone()
    alt[..., 0:2] = torch.randn_like(alt[..., 0:2]) * 9.0
    alt2 = side.clone()
    alt2[..., 2:4] = torch.randn_like(alt2[..., 2:4]) * 9.0
    with torch.no_grad():
        base, _ = model(neural, calib_trials=calib, side_features=side)
        out_a, _ = model(neural, calib_trials=calib, side_features=alt)
        out_b, _ = model(neural, calib_trials=calib, side_features=alt2)
    assert torch.equal(base, out_a) and torch.equal(base, out_b)


# ---- smoke-level training ---------------------------------------------------
def test_cprime_train_step_finiteness():
    canonical, _ = _canonical_state()
    model = ctl.build_matched_control_model(seed=42)
    eq.load_canonical_prefix(model, canonical)
    eq.freeze_inactive_consumer_modules(model)
    neural, behavior, calib, side = _synthetic_batch(3, 26, 81)
    model.set_carrier(_control_bundle(26, 82))
    model.train()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=1e-4
    )
    losses = []
    from torch.nn.parameter import UninitializedParameter as _Lazy

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction, _identity = model(neural, calib_trials=calib, side_features=side)
        valid = (behavior != -1.0).all(dim=-1)
        diff2 = ((prediction - behavior) ** 2).sum(dim=-1)
        loss = (diff2 * valid).sum() / (valid.sum() * behavior.shape[-1])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    assert all(math.isfinite(v) for v in losses)
    assert bool(torch.isfinite(prediction).all())
    assert all(
        bool(torch.isfinite(p.detach()).all())
        for p in model.parameters()
        if p.requires_grad and not isinstance(p, _Lazy)
    )
    assert model.consumer.gain.weight.grad is not None
    assert model.consumer.gain.weight.grad.abs().sum() > 0
    # the Z4 fused path keeps W_side at exactly zero gradient, as in arm C
    assert torch.count_nonzero(arm_common.w_side_block(model)).item() == 0
    # after training steps the control STILL violates the rotation property
    probe = ctl.nonequivariance_probe(
        model, neural, calib, side, [0.0, 0.8, -2.2, 3.0]
    )
    assert probe["identity_rotation_bitwise_equal"] is True
    assert probe["relative_max_violation"] > 0.5


# ---- receipt / gate conventions ---------------------------------------------
def test_cprime_integrity_block_and_gates():
    block = ctl.matched_control_integrity_block(
        num_heads=2, canonical_parameters=3_510_842,
        consumer_parameters=185_793, total_trainable=492_951,
        m_scale=1.3573337661619576,
        launch_max_violation=0.167, launch_relative_violation=2.03,
    )
    assert block["symmetry_claim"].startswith("none")
    assert block["carrier_view"]["equivariant_stream"] == "NONE — no complex-unit value stream"
    assert "z-scoring" in block["carrier_view"]["z_scoring"]
    matched = block["matched_to_arm_c"]
    assert matched["consumer_parameters"] == 185_793
    assert matched["total_trainable"] == 492_951
    assert matched["trainable_set_equal_to_arm_c"] is True
    assert block["parameter_disclosure"]["capacity_matched_to_arm_c"] is True
    gates = block["gates"]
    assert gates["primary_reading"]["comparison"] == "C_minus_Cprime_external_governing"
    assert "+0.03" in gates["primary_reading"]["gate"]
    assert "10/15" in gates["primary_reading"]["gate"]
    assert gates["capacity_confound_bound"]["comparison"] == "Cprime_minus_A_external_governing"
    assert gates["also_reported"][0]["comparison"] == "Cprime_minus_B"
    assert any("both granularities" in g for g in gates["granularity"])
    assert "date blocks" in gates["granularity"]
    assert "bootstrap" in gates["statistics"]
    assert gates["eval_policy"] == "plain forward; no augmentation"


def test_cprime_gates_do_not_rewrite_the_launched_gates():
    """The already-launched B/C pre-registrations stay intact: the C-prime
    gates DECOMPOSE the C - A gate, they do not replace it."""
    gates = ctl.MATCHED_CONTROL_GATES
    assert "C_minus_Cprime" in gates["primary_reading"]["comparison"]
    assert "decompose" in gates["note"] or "stay pre-registered" in gates["note"]
    launched = eq.PREREGISTERED_GATES
    assert launched["primary"]["comparison"] == "C_minus_A_external_governing"
    assert launched["primary"]["gate"] == (
        "mean paired delta >= +0.03 AND >= 10/15 sessions positive"
    )


def test_cprime_runner_cli_contract(tmp_path):
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_equivariant_control_cell.py"),
         "--cell", "C_equivariant"],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=120,
    )
    assert proc.returncode == 2
    existing = tmp_path / "cellCprime_matched_control_smoke"
    existing.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_equivariant_control_cell.py"),
         "--smoke", "--device", "cpu", "--output-root", str(tmp_path)],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=120,
    )
    assert proc.returncode == 2
    assert "fresh cell output directory required" in proc.stderr
    source = "\n".join((
        (ROOT / "scripts/run_equivariant_control_cell.py").read_text(),
        (ROOT / "src/tfpd_lane/equivariant_control.py").read_text(),
    ))
    for needle in (
        "C_minus_Cprime_external_governing", "Cprime_minus_A_external_governing",
        "Cprime_minus_B", "matched-capacity", "185,793", "z-scored",
    ):
        assert needle in source
    # the launched runners' bound files stay byte-frozen
    assert eq.PREREGISTERED_GATES is not None
    assert (ROOT / "src/tfpd_lane/equivariant_cell.py").is_file()
    assert (ROOT / "scripts/run_equivariant_cell.py").is_file()


# ---- sealed shared authority -------------------------------------------------
def test_shared_carrier_authority_untouched():
    sidecar = Path(str(AUTHORITY) + ".sha256").read_text().split()[0]
    from src.tfpd_lane import receipt as receipt_mod

    assert receipt_mod.sha256_file(AUTHORITY) == sidecar
    payload = torch.load(AUTHORITY, map_location="cpu", weights_only=False)
    assert payload["kind"] == eq.CARRIER_AUTHORITY_KIND
    assert len(payload["authority"]) == 27
