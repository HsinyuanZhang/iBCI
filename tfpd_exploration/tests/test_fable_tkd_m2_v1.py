"""Offline (no-NWB) contracts for FABLE TKD M2 v1 Wave 1.

Mirrors the offline part of stage0 A2/A3: SSM/GRU parity, SHUF/POOL laws,
epsilon=0 static-key cache bit-exactness, Phi_k PV-anchor affine, mu/sigma
closed-form normalizer inverse, PV init end-to-end against an independent
numpy composition, arm-B freezing, and the receipt law.  Run from the repo
root: ``python -m pytest tfpd_exploration/tests/test_fable_tkd_m2_v1.py -q``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "tfpd_exploration" / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from tfpd_exploration.src.fable_tkd_m2_v1 import data, plan, stage0
from tfpd_exploration.src.fable_tkd_m2_v1.model import DiagSSM, TKD, gru_step


def _build(**overrides):
    authority = stage0.synthetic_authority()
    kwargs = dict(
        epsilon="learnable",
        key_mode="identity",
        time_model="ssm",
        init="pv",
        authority_mean=authority["mean"],
        authority_std=authority["std"],
        pv_beta=plan.PV_BETA,
        pv_bin_masses=np.full(plan.N_QUERIES, plan.CHANNELS / plan.N_QUERIES,
                              dtype=np.float32),
        pv_output_scale=plan.PV_INIT_OUTPUT_SCALE,
        pv_readout_bias=plan.PV_READOUT_BIAS_X5,
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    return TKD(**kwargs).eval()


def _synthetic_inputs(seed: int = 7):
    rng = np.random.default_rng(seed)
    raw = np.stack(
        [
            rng.normal(0.0, 0.05, plan.CHANNELS),
            rng.normal(0.0, 0.05, plan.CHANNELS),
            np.abs(rng.normal(0.05, 0.03, plan.CHANNELS)) + 0.01,
            rng.normal(0.1, 0.2, plan.CHANNELS),
        ],
        axis=1,
    ).astype(np.float32)
    authority = stage0.synthetic_authority()
    t = torch.from_numpy(data.normalize_t4(raw, authority["mean"], authority["std"]))
    rho = torch.from_numpy(rng.random(plan.CHANNELS).astype(np.float32))
    x = torch.from_numpy(rng.random((2, plan.WINDOW, plan.CHANNELS)).astype(np.float32))
    return raw, t, rho, x


def test_fable_tkd_shuf_perm_law() -> None:
    perm = np.asarray(plan.SHUF_PERM, dtype=np.int64)
    assert perm.size == plan.CHANNELS
    assert not bool((perm == np.arange(plan.CHANNELS)).all())
    recomputed = np.random.default_rng(plan.SHUFFLE_SEED).permutation(plan.CHANNELS)
    assert bool((perm == recomputed).all())


def test_fable_tkd_shuf_forward_equivalence() -> None:
    _, t, rho, x = _synthetic_inputs()
    model = _build()
    shuffled = _build(key_mode="shuffled")
    with torch.no_grad():
        left = shuffled(x, t, rho)
        right = model(x, t[torch.from_numpy(np.asarray(plan.SHUF_PERM))], rho)
    assert torch.equal(left, right)


def test_fable_tkd_pool_keys_uniform_and_param_counts_equal() -> None:
    _, t, _, _ = _synthetic_inputs()
    pool = _build(key_mode="pool")
    with torch.no_grad():
        keys = pool.compute_key(t)
    assert torch.equal(keys, keys[0:1].expand_as(keys))
    counts = {
        int(sum(p.numel() for p in _build(**arm).parameters()))
        for arm in (
            {"epsilon": "learnable", "key_mode": "identity"},
            {"epsilon": "zero", "key_mode": "identity"},
            {"epsilon": "learnable", "key_mode": "shuffled"},
            {"epsilon": "learnable", "key_mode": "pool"},
        )
    }
    assert len(counts) == 1


def test_fable_tkd_static_key_cache_bitexact() -> None:
    _, t, rho, x = _synthetic_inputs()
    model = _build(epsilon="zero")
    with torch.no_grad():
        static = model.precompute_static(t)
        cached = model(x, t, rho, static=static)
        recomputed = model(x, t, rho)
        recomputed_twice = model(x, t, rho)
    assert torch.equal(cached, recomputed)
    assert torch.equal(recomputed, recomputed_twice)


def test_fable_tkd_key_cos_construction() -> None:
    authority = stage0.synthetic_authority()
    model = _build()
    beta = plan.PV_BETA
    psi = np.asarray(plan.QUERY_DIRECTIONS, dtype=np.float64)
    queries = model.queries.detach().double().numpy()

    angles = np.linspace(0.0, 2.0 * np.pi, plan.CHANNELS, endpoint=False)
    raw_unit = np.stack(
        [np.cos(angles), np.sin(angles), np.ones(plan.CHANNELS),
         np.full(plan.CHANNELS, 0.2)], axis=1,
    ).astype(np.float32)
    t_unit = torch.from_numpy(data.normalize_t4(raw_unit, authority["mean"], authority["std"]))
    with torch.no_grad():
        key_unit = model.compute_key(t_unit).double().numpy()
    dots_unit = (queries @ key_unit.T) / beta
    cos_angle = (
        np.cos(angles)[None, :] * np.cos(psi)[:, None]
        + np.sin(angles)[None, :] * np.sin(psi)[:, None]
    )
    assert float(np.abs(dots_unit - cos_angle).max()) < 1.0e-4

    raw, t, _, _ = _synthetic_inputs()
    with torch.no_grad():
        key = model.compute_key(t).double().numpy()
    dots = (queries @ key.T) / beta
    affine = (
        raw[:, 0][None, :].astype(np.float64) * np.cos(psi)[:, None]
        + raw[:, 1][None, :].astype(np.float64) * np.sin(psi)[:, None]
    )
    assert float(np.abs(dots - affine).max()) < 1.0e-5


def test_fable_tkd_mu_sigma_inverse_bitexact() -> None:
    authority = stage0.synthetic_authority()
    _, t, _, _ = _synthetic_inputs()
    model = _build()
    with torch.no_grad():
        mu, sig = model.compute_mu_sig(t)
    assert torch.equal(mu, t[:, 3] * authority["std"][3] + authority["mean"][3])
    # D12: sigma is the global constant S_POOLED for every unit at PV init.
    assert torch.equal(
        sig,
        torch.clamp(
            torch.full((plan.CHANNELS,), plan.S_POOLED, dtype=torch.float32),
            min=1.0e-3,
        ),
    )
    assert bool((sig == plan.S_POOLED).all())


def test_fable_tkd_psi_causal_ten_bin_mean() -> None:
    model = _build()
    rng = np.random.default_rng(3)
    counts = rng.integers(0, 5, size=(3, plan.WINDOW, plan.CHANNELS)).astype(np.float32)
    with torch.no_grad():
        u = model._psi(torch.from_numpy(counts))
    assert u.shape == (3, plan.WINDOW, plan.CHANNELS, plan.D_V)
    padded = np.concatenate(
        [np.zeros((3, plan.CONV_KERNEL - 1, plan.CHANNELS), np.float32), counts], axis=1
    )
    reference = np.stack(
        [padded[:, index : index + plan.CONV_KERNEL, :].mean(axis=1)
         for index in range(plan.WINDOW)],
        axis=1,
    )
    np.testing.assert_allclose(u[..., 0].numpy(), reference, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(u[..., 1:].numpy(), 0.0, atol=0.0)


def test_fable_tkd_pv_init_matches_independent_composition() -> None:
    raw, t, rho, x = _synthetic_inputs()
    model = _build(epsilon="zero")  # epsilon = 0 at pv init: identical output
    masses = np.full(plan.N_QUERIES, plan.CHANNELS / plan.N_QUERIES, dtype=np.float32)
    beta = plan.PV_BETA
    with torch.no_grad():
        prediction = model(x, t, rho).numpy()
        alpha = model.precompute_static(t)["alpha"].numpy().astype(np.float64)
    assert model.init_kind == "pv"

    counts64 = x.numpy().astype(np.float64)
    rate10 = np.stack(
        [
            np.concatenate(
                [np.zeros((plan.CONV_KERNEL - 1, plan.CHANNELS)), counts64[batch]], axis=0
            )[idx : idx + plan.CONV_KERNEL, :].mean(axis=0)
            for batch in range(counts64.shape[0])
            for idx in range(plan.WINDOW)
        ],
        axis=0,
    ).reshape(counts64.shape[0], plan.WINDOW, plan.CHANNELS)[:, -1, :]
    a, c, m, b = (raw[:, index].astype(np.float64) for index in range(4))
    rho64 = rho.numpy().astype(np.float64)
    # D12: global pooled-RMS value scale (no per-unit 1/m).
    value0 = rho64[None, :] * (rate10 - b[None, :]) / plan.S_POOLED
    psi = np.asarray(plan.QUERY_DIRECTIONS, dtype=np.float64)
    merged = np.zeros((value0.shape[0], 2), dtype=np.float64)
    for ell in range(plan.N_QUERIES):
        z0 = (alpha[ell][None, :] * value0).sum(axis=1)
        merged[:, 0] += np.cos(psi[ell]) * masses[ell] * z0
        merged[:, 1] += np.sin(psi[ell]) * masses[ell] * z0
    expected = (merged * plan.PV_INIT_OUTPUT_SCALE
                + np.asarray(plan.PV_READOUT_BIAS_X5)) / plan.BEHAVIOR_SCALE
    np.testing.assert_allclose(prediction, expected, rtol=1e-4, atol=1e-6)


def test_fable_tkd_pv_init_tracks_depth_weighted_pv_reference() -> None:
    """ADDENDUM-1 A1 reference: y_PV = sum_i rho_i (rate10_i - b_i) [a_i, c_i].

    The PV-initialized model (spec defaults: pv_beta=4, uniform bin masses)
    must track the classic depth-weighted PV on synthetic data.
    """
    raw, t, rho, x = _synthetic_inputs()
    model = _build(epsilon="zero")
    with torch.no_grad():
        prediction = model(x, t, rho).numpy()
    reference = stage0.pv_reference(x.numpy(), raw, rho.numpy())
    flattened = float(np.corrcoef(prediction.reshape(-1), reference.reshape(-1))[0, 1])
    assert flattened > 0.90
    for dim in range(plan.OUT_DIM):
        per_dim = float(np.corrcoef(prediction[:, dim], reference[:, dim])[0, 1])
        assert per_dim > 0.85


def test_fable_tkd_ssm_parity_random() -> None:
    torch.manual_seed(42)
    steps = 800
    z = torch.randn(3, steps, plan.D_H) * 0.1
    layer = DiagSSM(plan.D_H, pv_init=False)
    with torch.no_grad():
        a = torch.rand(plan.D_H) * 0.90 + 0.05  # D15 parametrization a = 0.98*sigmoid
        from tfpd_exploration.src.fable_tkd_m2_v1.model import _SSM_DECAY_CAP

        ratio = (a / _SSM_DECAY_CAP).clamp(1.0e-4, 1.0 - 1.0e-4)
        layer.a_log_raw.copy_(torch.log(ratio / (1.0 - ratio)))
        layer.B.normal_(0.0, 0.1)
        layer.C.normal_(0.0, 0.1)
        layer.glu.weight.normal_(0.0, 0.1)
        layer.glu.bias.zero_()
        layer.gamma_raw.fill_(float(torch.atanh(torch.tensor(0.7))))
        parallel = layer(z)
        hidden = torch.zeros(3, plan.D_H)
        outs = []
        for index in range(steps):
            hidden, out = layer.step(hidden, z[:, index])
            outs.append(out)
        sequential = torch.stack(outs, dim=1)
    assert float((parallel - sequential).abs().max()) <= 1.0e-6


def test_fable_tkd_gru_parity_random() -> None:
    torch.manual_seed(43)
    steps = 800
    gru = torch.nn.GRU(plan.D_H, plan.D_H, num_layers=1, batch_first=True)
    x = torch.randn(3, steps, plan.D_H) * 0.1
    with torch.no_grad():
        full, _ = gru(x)
        state = torch.zeros(1, 3, plan.D_H)
        manual = []
        for index in range(steps):
            out, state = gru_step(gru, x[:, index], state)
            manual.append(out[0])
        manual_full = torch.stack(manual, dim=1)
    assert float((full - manual_full).abs().max()) <= 1.0e-6


def test_fable_tkd_ssm_pv_init_identity_passthrough() -> None:
    torch.manual_seed(5)
    layer = DiagSSM(plan.D_H, pv_init=True)
    z = torch.randn(2, 37, plan.D_H)
    with torch.no_grad():
        out = layer(z)
    assert torch.equal(out, z)  # gamma = 0 -> exact identity regardless of o


def test_fable_tkd_stage0_construction_checks_pass() -> None:
    checks = stage0.construction_checks()
    for key in ("shuf_bitexact", "pool_rows_identical", "param_counts_equal",
                "static_bitexact", "mu_inverse_bitexact", "sig_inverse_bitexact"):
        assert checks[key] is True
    assert checks["key_cos_m_unit_error"] < 1.0e-4
    assert checks["key_cos_general_affine_error"] < 1.0e-5


def test_fable_tkd_arms_forward_shapes_and_frozen_readin() -> None:
    _, t, rho, x = _synthetic_inputs()
    for epsilon, key_mode, time_model in (
        ("learnable", "identity", "ssm"),
        ("zero", "identity", "ssm"),
        ("learnable", "shuffled", "ssm"),
        ("learnable", "pool", "ssm"),
        ("learnable", "identity", "gru"),
    ):
        model = _build(epsilon=epsilon, key_mode=key_mode, time_model=time_model)
        with torch.no_grad():
            y = model(x, t, rho)
        assert y.shape == (x.shape[0], plan.OUT_DIM)
        assert torch.isfinite(y).all()
    frozen = _build(frozen_readin=True)
    trainable = {name for name, p in frozen.named_parameters() if p.requires_grad}
    expected_prefixes = ("ssm_layers.", "readout.")
    assert all(name.startswith(expected_prefixes) for name in trainable)
    assert any(name.startswith("ssm_layers.") for name in trainable)
    assert any(name.startswith("readout.") for name in trainable)
    assert frozen.queries.requires_grad is False
    assert frozen.epsilon.requires_grad is False


def test_fable_tkd_receipt_law(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "receipt.json"
    digest = plan.atomic_receipt(target, {"a": 1})
    assert (target.stat().st_mode & 0o777) == 0o444
    sidecar = target.with_name("receipt.json.sha256")
    assert sidecar.read_text(encoding="utf-8") == f"{digest}  receipt.json\n"
    plan.verify_sidecar(target)
    with pytest.raises(plan.TKDError):
        plan.atomic_receipt(target, {"a": 2}, exclusive=True)


def test_fable_tkd_rho_and_normalizer_helpers() -> None:
    class Dataset:
        pass

    rng = np.random.default_rng(11)
    dataset = Dataset()
    session = "ses-synthetic-Run1"
    theta = np.asarray([0.0, np.pi / 2, np.pi, -np.pi / 2, 0.4, np.nan] * 5,
                       dtype=np.float64)[:30]
    scale = np.linspace(0.05, 1.0, plan.CHANNELS)
    rates = 2.0 + np.cos(theta)[:, None] * scale[None, :] / 10
    rates = rates + np.sin(theta)[:, None] * scale[None, :] / 20
    dataset.calib_trial_target_angles = {session: theta}
    dataset.calib_trial_spike_sums = {session: rates * 100.0}
    dataset.calib_trial_lengths = {session: np.full(30, 100.0)}
    fit = data.m30_t4(dataset, session)
    assert fit.raw.shape == (plan.CHANNELS, 4)
    assert fit.rho.shape == (plan.CHANNELS,)
    assert bool(((fit.rho >= 0) & (fit.rho <= 1)).all())
    assert fit.rho.max() > 0.9  # the synthetic population is perfectly tuned
    authority = stage0.synthetic_authority()
    t = data.normalize_t4(fit.raw, authority["mean"], authority["std"])
    assert t.shape == (plan.CHANNELS, 4) and np.isfinite(t).all()
    with pytest.raises(plan.TKDError):
        data.normalize_t4(np.full((plan.CHANNELS, 4), np.nan, np.float32),
                          authority["mean"], authority["std"])
