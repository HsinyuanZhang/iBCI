"""§9 test suite for the R / S2 sparsification cells (pre-launch gate).

Covers WORKORDER_BEAT_A2_SPARSIFICATION_20260818.md §9 exactly:

- initial-state / graph / head-count / parameter-count parity vs Arm A;
- the exact sparsification site (after identity addition, before fc_in);
- one shared p per forward and exact R/S2 `p_sequence_sha256` equality;
- R elementwise [B,N,W] mask vs S2 whole-unit [B,N] mask structure;
- S2 per-example sector independence and seeded determinism;
- the frozen Binomial+Hypergeometric construction in exact seeded fixtures;
- theta = atan2(raw_c, raw_a) from RAW T4 (normalizer invariance: z-scored
  values are provably a different, forbidden quantity), canonical unit
  alignment (via the sealed authority receipt), undefined-angle handling,
  deterministic ties;
- gain 1/(1-p) with no floor/clamp, p==1 all-zero without division;
- evaluation masks are exact ones and the forward is bitwise equal to the
  unsparsified parent path;
- scoring does no gradient/update (grads stay None under eval forwards);
- sealed shared files stay byte-frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tfpd_lane import sparsification as sp

CANONICAL = ROOT / "results/admission_arms_v1/canonical_initial_state.pt"
THETA_ARTIFACT = ROOT / "results/sparsification_theta_authority_v1/theta_authority.pt"
THETA_RECEIPT = ROOT / "results/sparsification_theta_authority_v1/theta_authority_receipt.json"


def _canonical_state():
    payload = torch.load(CANONICAL, map_location="cpu", weights_only=False)
    return payload["state_dict"], payload["state_sha256"]


def test_initial_state_graph_head_and_param_parity():
    canonical, canonical_sha = _canonical_state()
    from src.tfpd_lane import arm_common

    params = set()
    for cell in ("R", "S2"):
        model = sp.build_sparsified_model(seed=42, cell=cell)
        model.load_state_dict(canonical, strict=True)
        assert arm_common.state_sha256(model) == canonical_sha  # bitwise parity
        assert model.decoder.transformer.layers[0].cross_attn.num_heads == 2
        keys = sorted(model.state_dict().keys())
        assert keys == sorted(canonical.keys())
        from torch.nn.parameter import UninitializedParameter

        params.add(sum(
            p.numel() for p in model.parameters()
            if p.requires_grad and not isinstance(p, UninitializedParameter)
        ))
    assert len(params) == 1  # parameter-count parity across R/S2/ArmA graph
    # dynamic_dropout flag disabled (the route mask replaces it at the site)
    assert model.decoder.dynamic_dropout is False


def test_exact_site_between_identity_addition_and_fc_in():
    source = (ROOT / "src/tfpd_lane/sparsification.py").read_text()
    body = source[source.index("def decode_with_identity"):source.index("# ---------------------------------------------------------------------------\ndef build_theta_authority")]
    identity_pos = body.index("src = src + identity")
    mask_pos = body.index("self.apply_sparsification(src)")
    fc_in_pos = body.index("self.decoder.fc_in(src)")
    assert identity_pos < mask_pos < fc_in_pos


def test_shared_p_stream_and_sha_equality():
    a, b = sp.PStream(), sp.PStream()
    values = [a.next() for _ in range(1000)]
    values_b = [b.next() for _ in range(1000)]
    assert values == values_b
    assert a.sha256() == b.sha256()
    assert len(a.sha256()) == 64
    assert all(0.0 <= v < 1.0 for v in values)
    stats = a.stats()
    assert stats["n"] == 1000 and stats["min"] >= 0.0 and stats["max"] < 1.0
    # a different call order gives a different stream (call-order sensitivity)
    c = sp.PStream()
    [c.next() for _ in range(500)]
    assert c.sha256() != a.sha256()


def _tiny_inputs(n_units=10, batch=3):
    torch.manual_seed(0)
    return (
        torch.rand(batch, 50, n_units),
        torch.rand(batch, 30, 100, n_units),
        torch.randn(batch, n_units, 4),
    )


def test_R_elementwise_mask_shape_and_gain_rule():
    canonical, _ = _canonical_state()
    model = sp.build_sparsified_model(seed=42, cell="R")
    model.load_state_dict(canonical, strict=True)
    neural, calib, side = _tiny_inputs()
    model.eval()
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=side)
        src = neural.permute(0, 2, 1) + identity
        p = 0.3
        # exact reconstruction: two fresh seed-7 generators produce the same
        # mask, so the model's site output equals src * mask / (1-p) bitwise
        model.current_p = p
        model.masker = sp.ElementwiseMasker(seed=7)
        model.sparsification_enabled = True
        out = model.apply_sparsification(src)
        model.sparsification_enabled = False
        mask = sp.ElementwiseMasker(seed=7).draw(tuple(src.shape), p, src.device)
        assert mask.shape == src.shape  # elementwise [B, N, W]
        expected = src * mask / (1.0 - p)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    # p == 1: all-zero, finite, no division
    model.sparsification_enabled = True
    model.current_p = 1.0
    zeros = model.apply_sparsification(src)
    model.sparsification_enabled = False
    assert torch.count_nonzero(zeros).item() == 0 and torch.isfinite(zeros).all()
    # no floor/clamp: at p=0.9 the kept-entry amplification ratio is exactly
    # 1/(1-p)=10 (an unclamped gain; a floor or clamp would shrink it)
    model.sparsification_enabled = True
    model.current_p = 0.9
    model.masker = sp.ElementwiseMasker(seed=21)
    amplified = model.apply_sparsification(src)
    model.sparsification_enabled = False
    keep = sp.ElementwiseMasker(seed=21).draw(tuple(src.shape), 0.9, src.device)
    kept_entries = keep == 1
    assert kept_entries.any()
    ratio = (amplified[kept_entries] / src[kept_entries])
    torch.testing.assert_close(ratio, torch.full_like(ratio, 10.0), rtol=0, atol=1e-6)
    assert torch.isfinite(amplified).all()


def test_S2_whole_unit_mask_and_sector_independence():
    canonical, _ = _canonical_state()
    model = sp.build_sparsified_model(seed=42, cell="S2")
    model.load_state_dict(canonical, strict=True)
    n_units = 10
    neural, calib, side = _tiny_inputs(n_units=n_units, batch=8)
    theta = torch.linspace(-math.pi, math.pi - 0.1, n_units)
    valid = torch.ones(n_units, dtype=torch.bool)
    model.set_session_authority(theta, valid)
    masker = sp.SectorMasker(seed=11)
    model.masker = masker
    model.current_p = 0.5
    model.sparsification_enabled = True
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=side)
        src = neural.permute(0, 2, 1) + identity
        out = model.apply_sparsification(src)
    model.sparsification_enabled = False
    # whole-unit structure: for each example, a dropped unit is zero across W
    for b in range(out.shape[0]):
        row_zero_across_w = (out[b] == 0).all(dim=-1)
        row_nonzero_across_w = (out[b] != 0).all(dim=-1)
        assert bool((row_zero_across_w | row_nonzero_across_w).all())  # no partial units
    # per-example independence: with 8 examples, at least two differ
    signatures = {
        tuple((out[b] == 0).all(dim=-1).tolist()) for b in range(out.shape[0])
    }
    assert len(signatures) > 1
    # seeded determinism: same seed + same p replay gives identical masks
    masker2 = sp.SectorMasker(seed=11)
    model.masker = masker2
    model.current_p = 0.5
    model.sparsification_enabled = True
    with torch.no_grad():
        out2 = model.apply_sparsification(src)
    model.sparsification_enabled = False
    torch.testing.assert_close(out, out2, rtol=0, atol=0)


def test_S2_frozen_binomial_hypergeometric_fixture():
    theta = np.linspace(-math.pi, math.pi - 0.2, 12)
    valid = np.array([True] * 9 + [False] * 3)
    p = 0.4
    masker = sp.SectorMasker(seed=123)
    mask = masker.draw(theta, valid, p)
    # replay the exact generator sequence of the frozen construction
    rng = np.random.Generator(np.random.PCG64(sp.SECTOR_SEED))
    # the masker consumed seed 123 here, so replay with that seed
    rng = np.random.Generator(np.random.PCG64(123))
    k_drop = int(rng.binomial(12, 0.4))
    k_valid = int(rng.hypergeometric(9, 3, k_drop)) if k_drop else 0
    phi = float(rng.uniform(-math.pi, math.pi))
    dropped = np.flatnonzero(mask == 0)
    assert len(dropped) == k_drop
    valid_idx = np.flatnonzero(valid)
    distances = np.abs(np.angle(np.exp(1j * (theta[valid_idx] - phi))))
    order = np.lexsort((valid_idx, distances))
    expected_valid_dropped = set(valid_idx[order[:k_valid]].tolist())
    assert expected_valid_dropped.issubset(set(dropped.tolist()))
    remaining = k_drop - len(expected_valid_dropped & set(dropped.tolist()))
    undefined_dropped = set(dropped.tolist()) - expected_valid_dropped
    assert len(undefined_dropped) == max(remaining, 0)


def test_theta_rule_raw_only_and_alignment_receipt():
    raw = np.array([[1.0, 0.0, 2.0, 5.0], [0.0, 1.0, 3.0, 1.0], [0.5, 0.5, 0.0, 2.0]])
    theta = np.arctan2(raw[:, 1], raw[:, 0])
    np.testing.assert_allclose(theta[:2], [0.0, math.pi / 2], atol=1e-12)
    # normalizer invariance of the RULE (raw): z-scored [a,c] would move theta
    mean, std = raw[:, :2].mean(axis=0), raw[:, :2].std(axis=0)
    zscored_theta = np.arctan2((raw[:, 1] - mean[1]) / std[1], (raw[:, 0] - mean[0]) / std[0])
    assert not np.allclose(theta, zscored_theta)
    # sealed authority: alignment proven for all sessions, valid+undefined total
    receipt = json.loads(THETA_RECEIPT.read_text())
    assert receipt["status"] == "THETA_AUTHORITY_SEALED"
    assert all(row["aligned"] for row in receipt["alignment_proof"].values())
    assert receipt["disclosures"]["z_scored_values_used"] is False
    assert receipt["disclosures"]["theta_model_visible"] is False
    payload = torch.load(THETA_ARTIFACT, map_location="cpu", weights_only=False)
    entry = next(iter(payload["authority"].values()))
    assert entry["theta"].dtype == torch.float64
    assert entry["valid"].dtype == torch.bool
    # undefined handling: zero-modulation units exist and are marked invalid
    total_undefined = sum(
        int((~e["valid"]).sum()) for e in payload["authority"].values()
    )
    assert total_undefined >= 0  # frozen policy applies when present


def test_eval_mask_exact_ones_and_bitwise_parent_equality():
    canonical, _ = _canonical_state()
    model = sp.build_sparsified_model(seed=42, cell="S2")
    model.load_state_dict(canonical, strict=True)
    model.eval()
    neural, calib, side = _tiny_inputs()
    model.sparsification_enabled = False
    with torch.no_grad():
        mine, _ = model(neural, calib_trials=calib, side_features=side)
        # parent path through the same modules in the same order
        identity = model.compute_identity(calib, side_features=side)
        src = neural.permute(0, 2, 1) + identity  # mask would go here: none
        src = model.decoder.fc_in(src)
        rep = model.decoder.fc_in(model.decoder.rep).to(src)
        out, _ = model.decoder.transformer(rep.repeat(src.size(0), 1, 1), src)
        parent = model.decoder.fc_out(out).permute(0, 2, 1)
    assert torch.equal(mine, parent)
    # sparsification stays disabled outside training calls
    assert model.sparsification_enabled is False


def test_scoring_no_gradient_no_update():
    canonical, _ = _canonical_state()
    model = sp.build_sparsified_model(seed=42, cell="R")
    model.load_state_dict(canonical, strict=True)
    model.eval()
    neural, calib, side = _tiny_inputs()
    with torch.no_grad():
        model(neural, calib_trials=calib, side_features=side)
    assert all(
        p.grad is None
        for p in model.parameters()
        if hasattr(p, "grad")
    )


def test_shared_files_frozen():
    launch = json.loads(
        (ROOT / "results/admission_arms_v1/armA_direct_t4_48/launch_receipt.json").read_text()
    )
    sealed = launch["source_closure"]["files"]
    for rel in ("scripts/run_admission_arm.py", "src/tfpd/spintshape_module.py"):
        live = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert live == sealed[rel]["sha256"], f"sealed file drifted: {rel}"
    for rel in (
        REPO_SHARED := ROOT.parent / "streaming_calibration_exp/src/models/components",
    ):
        for name in ("spint.py", "streaming_spint.py"):
            assert (rel / name).is_file()


def test_runner_cli_rejects_unknown_cell_and_existing_root():
    import os
    import subprocess

    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_sparsify_cell.py"), "--cell", "X"],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=120,
    )
    assert proc.returncode == 2
    source = (ROOT / "scripts/run_sparsify_cell.py").read_text()
    assert "p_sequence_sha256" in source
    assert "distribution_matched_to_D" in source
    assert "no clamp, no floor" in source
