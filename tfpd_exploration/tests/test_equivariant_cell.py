"""Tests for the SO(2)-equivariant carrier cells (family B: arms B and C).

Covers the frozen equivariant-decoder cell spec:

- equivariance property: for random rotations R, R f(x, beta) == f(x, R beta)
  to 1e-5 across several shapes/seeds (max violation asserted and reported),
  and at R = identity the model is its own bitwise baseline;
- the #1 design trap: the chosen carrier normalization COMMUTES with rotation
  (rotate-then-normalize == normalize-then-rotate) while per-component
  z-scoring of (a,c) provably does NOT (negative control);
- invariant logits: the consumer's softmax attention weights are unchanged
  under rotation;
- structural covariance: an all-invalid carrier gives an exactly-zero output
  (no non-covariant additive term anywhere), and the fused token path is
  carrier-blind (the normalized (a,c) columns cannot move the output);
- arm B: sealed Cell-D graph parity + the joint rotation augmentation law
  (one seeded R per batch, shared across the batch; (m,b)/activity/calib
  invariant; pad rows restored; theta == 0 is a bitwise own baseline; eval
  path is the plain parent forward);
- arm C: canonical-prefix bitwise strict load, inactive-module freeze, and
  the parameter-count disclosure vs canonical Cell D;
- budget guard (33,925 x 48, fail-closed; smoke exempt; SWA window >= 4);
- receipt conventions: transactional no-overwrite writes, integrity-block
  fields, pre-registered gates;
- smoke-level train-step finiteness for both cells (loss/grads/params finite,
  consumer receives gradient, W_side stays exactly zero on the Z4 fused path);
- sealed shared files stay byte-frozen.
"""

from __future__ import annotations

import hashlib
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
from src.tfpd_lane import pop_robust, receipt as receipt_mod

CANONICAL = ROOT / "results/admission_arms_v1/canonical_initial_state.pt"
ARM_A_LAUNCH = ROOT / "results/admission_arms_v1/armA_direct_t4_48/launch_receipt.json"
THETA_ARTIFACT = ROOT / "results/sparsification_theta_authority_v1/theta_authority.pt"
EQUIVARIANCE_TOL = 1e-5


# ---- fixtures / helpers -----------------------------------------------------
def _canonical_state():
    payload = torch.load(CANONICAL, map_location="cpu", weights_only=False)
    return payload["state_dict"], payload["state_sha256"]


@pytest.fixture(scope="module")
def arm_c_model():
    canonical, _ = _canonical_state()
    model = eq.build_equivariant_model(seed=42)
    eq.load_canonical_prefix(model, canonical)
    eq.freeze_inactive_consumer_modules(model)
    model.eval()
    return model


@pytest.fixture(scope="module")
def arm_b_model():
    canonical, _ = _canonical_state()
    model = pop_robust.build_population_robustness_model(seed=42, cell="D")
    model.load_state_dict(canonical, strict=True)
    model.eval()
    return model


def _synthetic_raw(n: int, seed: int):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(-np.pi, np.pi, n)
    m = np.abs(rng.normal(1.3, 0.5, n))
    b = rng.normal(10.0, 3.0, n)
    n_invalid = max(1, n // 10)
    m[:n_invalid] = 0.0
    a = m * np.cos(phi)
    c = m * np.sin(phi)
    a[:n_invalid] = 0.0
    c[:n_invalid] = 0.0
    raw = np.stack([a, c, m, b], axis=1).astype(np.float64)
    m_norm = rng.normal(0.0, 1.0, n).astype(np.float32)
    b_norm = rng.normal(0.0, 1.0, n).astype(np.float32)
    return raw, m_norm, b_norm


def _synthetic_batch(batch: int, n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    neural = torch.rand((batch, 50, n), generator=g)
    calib = torch.rand((batch, 30, 100, n), generator=g)
    side = torch.randn((batch, n, 4), generator=g)
    behavior = torch.randn((batch, 50, 2), generator=g)
    behavior[0, 40:] = -1.0  # pad rows (all-channel fill), as the loader emits
    return neural, behavior, calib, side


def _bundle(n: int, seed: int, m_scale: float = 1.3) -> eq.CarrierBundle:
    raw, m_norm, b_norm = _synthetic_raw(n, seed)
    return eq.CarrierBundle.from_raw(
        torch.from_numpy(raw), torch.from_numpy(m_norm), torch.from_numpy(b_norm),
        m_scale,
    )


# ---- rotation utilities -----------------------------------------------------
def test_rotation_matrix_properties():
    ident = eq.rotation_matrix_2d(0.0)
    assert ident[0, 0] == 1.0 and ident[1, 1] == 1.0
    assert ident[0, 1] == 0.0 and ident[1, 0] == 0.0
    for theta in (0.3, -1.2, 2.71, math.pi):
        rot = eq.rotation_matrix_2d(theta)
        assert np.allclose(rot @ rot.T, np.eye(2), atol=1e-15)
        assert abs(np.linalg.det(rot) - 1.0) < 1e-15
    # group law in float64
    a, b = 0.37, -1.11
    composed = eq.rotation_matrix_2d(a) @ eq.rotation_matrix_2d(b)
    assert np.allclose(composed, eq.rotation_matrix_2d(a + b), atol=1e-14)
    # deterministic probe namespace
    assert eq.probe_thetas(5) == eq.probe_thetas(5)
    assert len(eq.probe_thetas(8)) == 8
    assert all(-math.pi <= t <= math.pi for t in eq.probe_thetas(50))


# ---- the normalization commutation proof (the #1 design trap) ---------------
def test_unit_normalization_commutes_with_rotation():
    """rotate-then-normalize == normalize-then-rotate for beta/|beta| and for
    every rotation-INARIANT scalar normalization (global m_scale included)."""
    rng = np.random.default_rng(7)
    beta = rng.normal(1.0, 2.0, size=(200, 2))  # arbitrary carrier cloud
    m_scale = float(np.hypot(beta[:, 0], beta[:, 1]).mean())
    worst_unit, worst_scale = 0.0, 0.0
    for theta in (0.0, 0.3, 1.1, -2.0, math.pi / 3, 2.999):
        rot = eq.rotation_matrix_2d(theta)
        rotated = beta @ rot.T
        # normalize-then-rotate
        norm_first = beta / np.linalg.norm(beta, axis=1, keepdims=True)
        norm_first_rotated = norm_first @ rot.T
        # rotate-then-normalize
        norm_last = rotated / np.linalg.norm(rotated, axis=1, keepdims=True)
        worst_unit = max(
            worst_unit, float(np.abs(norm_first_rotated - norm_last).max())
        )
        # invariant scalar normalization commutes trivially but prove the
        # magnitudes actually used are invariant: |R beta| == |beta|, and the
        # scaled feature log1p(|beta|/m_scale) is unchanged
        scale_first = np.log1p(np.linalg.norm(beta, axis=1) / m_scale)
        scale_last = np.log1p(np.linalg.norm(rotated, axis=1) / m_scale)
        worst_scale = max(worst_scale, float(np.abs(scale_first - scale_last).max()))
    assert worst_unit <= 1e-12, worst_unit
    assert worst_scale <= 1e-12, worst_scale


def test_per_component_z_scoring_does_not_commute():
    """Negative control (the design trap): z-scoring (a,c) per component with
    population statistics is NOT rotation-equivariant — the rotated z-scored
    carrier differs from the z-scored rotated carrier by a large amount."""
    rng = np.random.default_rng(11)
    beta = rng.normal(1.0, 2.0, size=(500, 2))
    mean, std = beta.mean(axis=0), beta.std(axis=0)  # per-component statistics
    theta = 0.9
    rot = eq.rotation_matrix_2d(theta)
    rotated = beta @ rot.T
    zscore_then_rotate = ((beta - mean) / std) @ rot.T
    rotate_then_zscore = (rotated - mean) / std
    violation = float(np.abs(zscore_then_rotate - rotate_then_zscore).max())
    assert violation > 1e-1, violation  # grossly non-commuting
    # and the choice actually made never touches the z-scored pair
    assert eq.CARRIER_NORMALIZATION_CHOICE.index("z-scored (a,c) is NEVER") > 0


def test_bundle_invariants_recompute_invariant_under_rotation():
    bundle = _bundle(48, seed=3)
    worst_features, worst_unit, worst_valid = 0.0, 0.0, 0.0
    for theta in (0.0, 0.5, -1.7, 2.4):
        rotated = bundle.rotated(theta)
        rot = eq.rotation_matrix_2d(theta)
        # unit vector: rotated bundle's units == R @ base units (float64)
        expected_unit = bundle.unit2.numpy() @ rot.T
        worst_unit = max(
            worst_unit,
            float(np.abs(rotated.unit2.numpy() - expected_unit).max()),
        )
        # invariant features (the float32 tensor the logits consume)
        delta = float(
            (rotated.invariant_features() - bundle.invariant_features())
            .abs().max().item()
        )
        worst_features = max(worst_features, delta)
        worst_valid = max(
            worst_valid,
            float((rotated.valid != bundle.valid).to(torch.float64).max().item()),
        )
    assert worst_unit <= 1e-12, worst_unit
    assert worst_features <= 1e-7, worst_features  # float32 cast of a float64 core
    assert worst_valid == 0.0


def test_bundle_unit_norms_and_invalid_units():
    bundle = _bundle(30, seed=5)
    valid = bundle.valid
    assert int(valid.sum()) >= 2 and int((~valid).sum()) >= 1
    norms = bundle.unit2.norm(dim=-1)
    torch.testing.assert_close(
        norms[valid], torch.ones_like(norms[valid]), rtol=0, atol=1e-12
    )
    assert torch.count_nonzero(bundle.unit2[~valid]).item() == 0
    # rho features exist only for valid units and live in [-1, 1]
    assert torch.count_nonzero(bundle.rho_cos[~valid]).item() == 0
    assert bundle.rho_cos[valid].abs().max() <= 1.0 + 1e-12
    assert bundle.rho_sin[valid].abs().max() <= 1.0 + 1e-12


# ---- the equivariance property ----------------------------------------------
def test_equivariance_random_rotations_shapes_and_seeds(arm_c_model):
    """f(x, R beta) == R f(x, beta) to 1e-5 over random rotations, several
    shapes and carrier/data seeds; identity rotation is a bitwise baseline."""
    model = arm_c_model
    violations = []
    for n_units, batch, seed in (
        (5, 1, 101), (37, 3, 202), (90, 2, 303), (12, 4, 404), (64, 1, 505),
    ):
        neural, behavior, calib, side = _synthetic_batch(batch, n_units, seed)
        bundle = _bundle(n_units, seed + 1)
        model.set_carrier(bundle)
        model.eval()
        with torch.no_grad():
            base, _ = model(neural, calib_trials=calib, side_features=side)
            assert base.shape == (batch, 50, 2)
            assert bool(torch.isfinite(base).all())
            thetas = [0.0] + list(
                np.random.Generator(np.random.PCG64(seed)).uniform(
                    -math.pi, math.pi, 6
                )
            )
            for theta in thetas:
                pred, _ = model(
                    neural, calib_trials=calib, side_features=side,
                    carrier=bundle.rotated(float(theta)),
                )
                if float(theta) == 0.0:
                    assert torch.equal(pred, base), "identity rotation must be bitwise"
                rot = torch.from_numpy(eq.rotation_matrix_2d(float(theta)))
                expected = base.to(torch.float64) @ rot.t()
                violations.append(
                    float((pred.to(torch.float64) - expected).abs().max().item())
                )
    max_violation = max(violations)
    assert max_violation <= EQUIVARIANCE_TOL, max_violation


def test_equivariance_holds_with_a_differently_seeded_head(arm_c_model):
    """The property is architectural: it survives a fresh consumer head init."""
    model = arm_c_model
    torch.manual_seed(9_909)
    model.attach_consumer(
        eq.EquivariantCarrierConsumer()
    )
    try:
        neural, _b, calib, side = _synthetic_batch(2, 23, 77)
        bundle = _bundle(23, 78)
        model.set_carrier(bundle)
        model.eval()
        with torch.no_grad():
            base, _ = model(neural, calib_trials=calib, side_features=side)
            worst = 0.0
            for theta in (0.8, -2.2, 3.0):
                pred, _ = model(
                    neural, calib_trials=calib, side_features=side,
                    carrier=bundle.rotated(theta),
                )
                rot = torch.from_numpy(eq.rotation_matrix_2d(theta))
                worst = max(
                    worst,
                    float(
                        (pred.to(torch.float64) - base.to(torch.float64) @ rot.t())
                        .abs().max().item()
                    ),
                )
        assert worst <= EQUIVARIANCE_TOL, worst
    finally:
        torch.manual_seed(eq.EQUIVARIANT_HEAD_SEED)
        model.attach_consumer(
            eq.EquivariantCarrierConsumer()
        )
        model.eval()


def test_rotation_by_pi_is_exact_negation(arm_c_model):
    model = arm_c_model
    neural, _b, calib, side = _synthetic_batch(2, 15, 88)
    bundle = _bundle(15, 89)
    model.set_carrier(bundle)
    model.eval()
    with torch.no_grad():
        base, _ = model(neural, calib_trials=calib, side_features=side)
        pred, _ = model(
            neural, calib_trials=calib, side_features=side,
            carrier=bundle.rotated(math.pi),
        )
    torch.testing.assert_close(pred, -base, rtol=0, atol=1e-10)


# ---- invariant logits / structural covariance -------------------------------
def test_attention_weights_invariant_under_rotation(arm_c_model):
    model = arm_c_model
    neural, _b, calib, side = _synthetic_batch(3, 40, 61)
    bundle = _bundle(40, 62)
    model.set_carrier(bundle)
    model.eval()
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=torch.zeros_like(side))
        tokens = model.decoder.fc_in(neural.permute(0, 2, 1) + identity)
        queries = model.decoder.fc_in(model.decoder.rep).to(tokens)
        queries = queries.repeat(tokens.size(0), 1, 1)
        w_base = model.consumer.attention_weights(tokens, queries, bundle)
        assert w_base.shape == (3, 2, eq.NUM_HEADS, 40)
        deltas = []
        for theta in (0.0, 0.4, -1.3, 2.9):
            w_rot = model.consumer.attention_weights(
                tokens, queries, bundle.rotated(theta)
            )
            if theta == 0.0:
                assert torch.equal(w_rot, w_base)
            deltas.append(float((w_rot - w_base).abs().max().item()))
    assert max(deltas) <= 1e-7, max(deltas)


def test_all_invalid_carrier_gives_exactly_zero_output(arm_c_model):
    """No non-covariant additive term: with every direction undefined the
    equivariant value stream is identically zero, so the decode is exactly 0."""
    model = arm_c_model
    neural, _b, calib, side = _synthetic_batch(2, 9, 91)
    raw = np.zeros((9, 4), dtype=np.float64)
    raw[:, 3] = 5.0  # b column arbitrary; a=c=m=0 -> all directions invalid
    bundle = eq.CarrierBundle.from_raw(
        torch.from_numpy(raw), torch.zeros(9), torch.zeros(9), 1.0
    )
    model.set_carrier(bundle)
    model.eval()
    with torch.no_grad():
        out, _ = model(neural, calib_trials=calib, side_features=side)
    assert torch.count_nonzero(out).item() == 0
    assert bool(torch.isfinite(out).all())


def test_fused_token_path_is_carrier_blind(arm_c_model):
    """The normalized (a,c) side columns never enter the arm-C graph: changing
    them cannot move the output by a single bit."""
    model = arm_c_model
    neural, _b, calib, side = _synthetic_batch(2, 20, 92)
    bundle = _bundle(20, 93)
    model.set_carrier(bundle)
    model.eval()
    alt = side.clone()
    alt[..., 0:2] = torch.randn_like(alt[..., 0:2]) * 7.0
    with torch.no_grad():
        base, _ = model(neural, calib_trials=calib, side_features=side)
        alt_out, _ = model(neural, calib_trials=calib, side_features=alt)
    assert torch.equal(base, alt_out)


# ---- arm B: sealed Cell D + the augmentation law ----------------------------
def test_arm_b_is_sealed_cell_d_graph(arm_b_model):
    model = arm_b_model
    canonical, canonical_sha = _canonical_state()
    assert arm_common.state_sha256(model) == canonical_sha
    assert model.decoder.transformer.layers[0].cross_attn.num_heads == 2
    assert model.decoder.dynamic_dropout is True
    assert pop_robust.DYNAMIC_DROPOUT_LOW == 0.0 and pop_robust.DYNAMIC_DROPOUT_HIGH == 1.0
    from torch.nn.parameter import UninitializedParameter

    count = sum(
        p.numel() for p in model.parameters()
        if not isinstance(p, UninitializedParameter)
    )
    assert count == 3_510_842


def test_arm_b_rotation_stream_law():
    a, b = eq.RotationStream(), eq.RotationStream()
    values = [a.next() for _ in range(500)]
    assert values == [b.next() for _ in range(500)]  # seeded determinism
    assert a.sha256() == b.sha256()
    stats = a.stats()
    assert stats["n"] == 500
    assert all(-math.pi <= v <= math.pi for v in values)
    c = eq.RotationStream()
    [c.next() for _ in range(10)]
    assert c.sha256() != a.sha256()  # call-order sensitivity
    assert eq.B_ROTATION_SEED == 42_010  # dedicated namespace, frozen


def test_arm_b_joint_rotation_augmentation_law():
    side = torch.randn(4, 30, 4)
    behavior = torch.randn(4, 50, 2)
    behavior[2, 30:] = -1.0  # pad rows
    theta = 1.234
    rot = torch.from_numpy(eq.rotation_matrix_2d(theta)).to(side.dtype)
    rot_side, rot_behavior = eq.rotate_side_behavior(side, behavior, theta)
    # (a,c) columns rotated jointly with the labels
    torch.testing.assert_close(
        rot_side[..., 0:2], side[..., 0:2] @ rot.t(), rtol=0, atol=1e-6
    )
    # (m,b) columns, activity, calibration: bitwise untouched
    assert torch.equal(rot_side[..., 2:4], side[..., 2:4])
    valid = (behavior != -1.0).all(dim=-1)
    torch.testing.assert_close(
        rot_behavior[..., 0:2][valid], behavior[..., 0:2][valid] @ rot.t(),
        rtol=0, atol=1e-6,
    )
    # pad rows restored bitwise so validity masking is unchanged
    assert torch.equal(rot_behavior[~valid], behavior[~valid])
    assert ((rot_behavior != -1.0).all(dim=-1) == valid).all()
    # theta == 0.0: the model's own bitwise baseline
    side0, behavior0 = eq.rotate_side_behavior(side, behavior, 0.0)
    assert torch.equal(side0, side) and torch.equal(behavior0, behavior)
    # a loss computed on rotated labels equals the manual rotated-loss (the
    # supervision really is the jointly rotated pair)
    pred = torch.randn(4, 50, 2)
    diff2 = ((pred - rot_behavior) ** 2).sum(dim=-1)
    loss = (diff2 * valid).sum() / (valid.sum() * 2.0)
    manual = (
        ((pred[valid] - (behavior[..., 0:2][valid] @ rot.t())) ** 2).sum()
        / (valid.sum() * 2.0)
    )
    torch.testing.assert_close(loss, manual, rtol=0, atol=1e-5)


def test_arm_b_eval_forward_is_plain_cell_d_path(arm_b_model):
    model = arm_b_model
    neural, _b, calib, side = _synthetic_batch(2, 18, 94)
    model.eval()
    with torch.no_grad():
        mine, _ = model(neural, calib_trials=calib, side_features=side)
        identity = model.compute_identity(calib, side_features=side)
        src = neural.permute(0, 2, 1) + identity
        src = model.decoder.fc_in(src)
        rep = model.decoder.fc_in(model.decoder.rep).to(src)
        out, _ = model.decoder.transformer(rep.repeat(src.size(0), 1, 1), src)
        parent = model.decoder.fc_out(out).permute(0, 2, 1)
    assert torch.equal(mine, parent)


# ---- arm C: canonical prefix + parameter disclosure -------------------------
def test_arm_c_canonical_prefix_and_parameter_disclosure():
    canonical, canonical_sha = _canonical_state()
    model = eq.build_equivariant_model(seed=42)
    proof = eq.load_canonical_prefix(model, canonical)
    assert proof["strict_load"] and proof["bitwise_equal_all_keys"]
    assert proof["canonical_subset_sha256"] == canonical_sha
    freeze = eq.freeze_inactive_consumer_modules(model)
    from torch.nn.parameter import UninitializedParameter

    for name in freeze["modules"]:
        module = model
        for attr in name.split("."):
            module = getattr(module, attr)
        assert all(
            not p.requires_grad for p in module.parameters()
            if not isinstance(p, UninitializedParameter)
        )
    accounting = eq.parameter_accounting(model)
    assert accounting["canonical_tensors_strict_loaded"] == 3_510_842
    assert accounting["new_consumer_parameters"] == 185_793  # frozen head budget
    assert accounting["frozen_canonical_parameters"] > 0
    assert (
        accounting["total_parameters"]
        == accounting["canonical_tensors_strict_loaded"]
        + accounting["new_consumer_parameters"]
    )
    # the disclosure block tells the truth about the deviation
    block = eq.arm_c_integrity_block(
        num_heads=2,
        canonical_parameters=accounting["canonical_tensors_strict_loaded"],
        consumer_parameters=accounting["new_consumer_parameters"],
        total_trainable=accounting["total_trainable"],
        m_scale=1.0,
        launch_max_violation=0.0,
        normalization_commutes=True,
    )
    d = block["parameter_disclosure"]
    assert d["bitwise_parity_at_init_vs_cell_d_claimed"] is False
    assert d["canonical_tensors_strict_loaded_bitwise"] == 3_510_842
    assert d["new_consumer_parameters"] == 185_793
    assert block["carrier_normalization"]["z_scored_ac_used"] is False
    assert block["carrier_normalization"]["normalization_commutes_with_rotation"] is True
    assert "SO(2)" in block["symmetry_statement"]


def test_consumer_real_coefficients_only():
    """Every consumer parameter is real; nothing can act differently on the
    two carrier components (the architectural equivariance condition)."""
    consumer = eq.EquivariantCarrierConsumer()
    assert all(
        torch.is_floating_point(p) and not torch.is_complex(p)
        for p in consumer.parameters()
    )
    names = sorted(n for n, _ in consumer.named_parameters())
    assert names == [
        "gain.bias", "gain.weight", "head_mix", "k_proj.weight",
        "norm_k.bias", "norm_k.weight", "norm_q.bias", "norm_q.weight",
        "out_scale", "q_proj.weight",
    ]


# ---- budget guard -----------------------------------------------------------
def test_budget_guard():
    ok = eq.assert_training_budget(33_925, 48, smoke=False)
    assert ok["total_optimizer_steps"] == 33_925 * 48
    with pytest.raises(SystemExit):
        eq.assert_training_budget(33_924, 48, smoke=False)
    with pytest.raises(SystemExit):
        eq.assert_training_budget(33_925, 47, smoke=False)
    with pytest.raises(SystemExit):
        eq.assert_training_budget(33_925, 3, smoke=True)  # SWA window needs 4
    smoke = eq.assert_training_budget(12, 4, smoke=True)
    assert smoke["smoke"] is True


# ---- receipt conventions ----------------------------------------------------
def test_receipt_transactional_no_overwrite(tmp_path):
    target = tmp_path / "launch_receipt.json"
    receipt_mod.write_receipt_transactionally(target, {"schema": "x", "v": 1})
    assert target.is_file()
    sidecar = Path(str(target) + ".sha256")
    assert sidecar.is_file()
    with pytest.raises(SystemExit) as exc:
        receipt_mod.write_receipt_transactionally(target, {"schema": "x", "v": 2})
    assert exc.value.code == 2
    assert json.loads(target.read_text())["v"] == 1  # unchanged


def test_integrity_blocks_and_preregistered_gates():
    block_b = eq.arm_b_integrity_block(
        num_heads=2, total_parameters=3_510_842, canonical_parameters=3_510_842
    )
    law = block_b["augmentation_law"]
    assert law["shared_across_batch"] is True
    assert "PCG64(42010)" in law["draw"]
    assert block_b["parameter_count"]["disclosed_deviation"] is False
    assert block_b["num_heads"] == 2
    for block in (block_b, eq.arm_c_integrity_block(
        num_heads=2, canonical_parameters=3_510_842,
        consumer_parameters=185_793, total_trainable=1, m_scale=1.0,
        launch_max_violation=0.0, normalization_commutes=True,
    )):
        gates = block["gates"]
        assert gates["primary"]["comparison"] == "C_minus_A_external_governing"
        assert "+0.03" in gates["primary"]["gate"]
        assert "10/15" in gates["primary"]["gate"]
        assert {g["comparison"] for g in gates["also_reported"]} == {
            "C_minus_B", "B_minus_A"
        }
        assert any(
            "both granularities" in g for g in gates["granularity"]
        )
        assert "date blocks" in gates["granularity"]
        assert "bootstrap" in gates["statistics"]


# ---- smoke-level train-step finiteness --------------------------------------
def _train_steps(model, params, neural, behavior, calib, side, steps=2):
    model.train()
    optimizer = torch.optim.Adam(params, lr=1e-4)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction, _identity = model(
            neural, calib_trials=calib, side_features=side
        )
        valid = (behavior != -1.0).all(dim=-1)
        diff2 = ((prediction - behavior) ** 2).sum(dim=-1)
        loss = (diff2 * valid).sum() / (valid.sum() * behavior.shape[-1])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses, prediction


def test_arm_c_train_step_finiteness_and_zero_w_side():
    canonical, _ = _canonical_state()
    model = eq.build_equivariant_model(seed=42)
    eq.load_canonical_prefix(model, canonical)
    eq.freeze_inactive_consumer_modules(model)
    neural, behavior, calib, side = _synthetic_batch(3, 26, 95)
    bundle = _bundle(26, 96)
    model.set_carrier(bundle)
    losses, prediction = _train_steps(
        model, [p for p in model.parameters() if p.requires_grad],
        neural, behavior, calib, side,
    )
    assert all(math.isfinite(v) for v in losses)
    assert bool(torch.isfinite(prediction).all())
    from torch.nn.parameter import UninitializedParameter as _Lazy

    assert all(
        bool(torch.isfinite(p.detach()).all())
        for p in model.parameters()
        if p.requires_grad and not isinstance(p, _Lazy)
    )
    # the consumer head receives gradient; the Z4 fused path keeps W_side at
    # exactly zero gradient (the carrier enters only through the consumer)
    assert model.consumer.gain.weight.grad is not None
    assert model.consumer.gain.weight.grad.abs().sum() > 0
    w_side = arm_common.w_side_block(model)
    assert torch.count_nonzero(w_side).item() == 0
    # the property survives training steps
    probe = eq.equivariance_probe(
        model, neural, calib, side, [0.0, 0.7, -2.1, 3.05]
    )
    assert probe["identity_rotation_bitwise_equal"] is True
    assert probe["max_violation"] <= EQUIVARIANCE_TOL


def test_arm_b_train_step_finiteness_with_augmentation():
    canonical, _ = _canonical_state()
    model = pop_robust.build_population_robustness_model(seed=42, cell="D")
    model.load_state_dict(canonical, strict=True)
    neural, behavior, calib, side = _synthetic_batch(3, 26, 97)
    stream = eq.RotationStream(seed=123)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses = []
    for _ in range(2):
        theta = stream.next()
        rot_side, rot_behavior = eq.rotate_side_behavior(side, behavior, theta)
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(neural, calib_trials=calib, side_features=rot_side)
        valid = (rot_behavior != -1.0).all(dim=-1)
        diff2 = ((prediction - rot_behavior) ** 2).sum(dim=-1)
        loss = (diff2 * valid).sum() / (valid.sum() * rot_behavior.shape[-1])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    assert all(math.isfinite(v) for v in losses)
    assert bool(torch.isfinite(prediction).all())
    assert stream.stats()["n"] == 2


def test_scoring_no_gradient_no_update(arm_c_model):
    model = arm_c_model
    neural, _b, calib, side = _synthetic_batch(2, 14, 98)
    model.set_carrier(_bundle(14, 99))
    model.eval()
    with torch.no_grad():
        model(neural, calib_trials=calib, side_features=side)
    assert all(
        p.grad is None for p in model.parameters() if hasattr(p, "grad")
    )


# ---- sealed-file freeze -----------------------------------------------------
def test_sealed_files_frozen():
    launch = json.loads(ARM_A_LAUNCH.read_text())
    sealed = launch["source_closure"]["files"]
    for rel in ("scripts/run_admission_arm.py", "src/tfpd/spintshape_module.py"):
        live = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert live == sealed[rel]["sha256"], f"sealed file drifted: {rel}"
    for name in ("spint.py", "streaming_spint.py", "streaming_encoders.py"):
        assert (ROOT.parent / "streaming_calibration_exp/src/models/components" / name).is_file()
    # the canonical artifact is untouched
    sidecar = Path(str(CANONICAL) + ".sha256").read_text().split()[0]
    assert receipt_mod.sha256_file(CANONICAL) == sidecar
    assert receipt_mod.sha256_file(THETA_ARTIFACT) == Path(
        str(THETA_ARTIFACT) + ".sha256"
    ).read_text().split()[0]


# ---- runner CLI contract ----------------------------------------------------
def test_runner_cli_rejects_unknown_cell_and_existing_root(tmp_path):
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_equivariant_cell.py"),
         "--cell", "A_sealed"],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=120,
    )
    assert proc.returncode == 2
    existing = tmp_path / "cellB_rotation_augmentation_smoke"
    existing.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_equivariant_cell.py"),
         "--cell", "B_augmentation", "--smoke", "--device", "cpu",
         "--output-root", str(tmp_path)],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=120,
    )
    assert proc.returncode == 2
    assert "fresh cell output directory required" in proc.stderr
    source = "\n".join((
        (ROOT / "scripts/run_equivariant_cell.py").read_text(),
        (ROOT / "src/tfpd_lane/equivariant_cell.py").read_text(),
    ))
    for needle in (
        "C_minus_A_external_governing", "PREREGISTERED_GATES",
        "identity rotation", "equivariance", "33,925", "fused path",
    ):
        assert needle in source or needle.lower() in source.lower()
