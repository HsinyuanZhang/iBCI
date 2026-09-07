"""Gate-0 tests for the three-arm admission trainers (§3/§4/§6/§9/§11).

- LR schedule shape: step 0 = 1e-5, linear through the two-epoch warmup to
  1e-4, cosine strictly decreasing afterwards, exactly 1e-6 at the final step;
- B/C step-equality: B's T4 phase and arm C share epochs, schedule kind,
  schedule parameters, and the LR value at EVERY T4-phase step (enumerated at
  the real steps-per-epoch), while arm A is the 48-epoch baseline;
- §4 admission semantics: 'z4' returns ``zeros_like`` of the aligned
  normalized T4 (bitwise positive zero, same shape/dtype) and 't4' returns the
  exact tensor by identity — admission happens on the normalized tensor, and
  no alpha*raw product exists in the admission path;
- §3 phase-1 zero invariants on the REAL spintshape model: one Z4 step leaves
  grad(W_side) bitwise zero; an Adam step leaves W_side, exp_avg, exp_avg_sq
  elementwise exactly zero;
- §9 fixed-batch diagnostics: post_pool[0] T4/calibration contributions and
  attention entropy run, are finite, and are exactly zero-ratio at W_side=0;
  the attention patch is fully removed afterwards;
- canonical initial state artifact: minted once, strict-reload bitwise equal,
  refuses to be minted twice;
- arm runner preflight (real strict-27 source cache, CPU): passes for arm C
  and seals a preflight receipt without creating the arm run directory.
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

from src.tfpd_lane import arm_common as ac

# The spintshape builder imports src.models.* (owned by streaming_calibration_exp).
# Extend the already-imported ROOT/src package path so both trees resolve in one
# process without shadowing either.
import src as _src_pkg

_STREAMING_SRC = str(ROOT.parent / "streaming_calibration_exp" / "src")
if _STREAMING_SRC not in _src_pkg.__path__:
    _src_pkg.__path__.append(_STREAMING_SRC)

REAL_STEPS_PER_EPOCH = 33_925  # strict-27 full window set, batch 32, drop-partial


# ---- LR schedule -----------------------------------------------------------
def test_warmup_then_cosine_shape():
    spe = 1000
    n = 33
    total = n * spe
    warm = ac.warmup_steps(spe)
    assert warm == 2000
    assert ac.lr_at_step(0, n, spe) == pytest.approx(1e-5, rel=0, abs=0)
    assert ac.lr_at_step(warm - 1, n, spe) == pytest.approx(
        1e-5 + 9e-5 * (warm - 1) / warm
    )
    assert ac.lr_at_step(warm, n, spe) == pytest.approx(1e-4)
    assert ac.lr_at_step(total - 1, n, spe) == pytest.approx(1e-6, abs=1e-12)
    # strictly decreasing across the cosine region
    prev = ac.lr_at_step(warm, n, spe)
    for step in range(warm + 1, total, 97):
        lr = ac.lr_at_step(step, n, spe)
        assert lr < prev
        prev = lr
    # linear inside warmup
    a = ac.lr_at_step(500, n, spe)
    b = ac.lr_at_step(1000, n, spe)
    c = ac.lr_at_step(1500, n, spe)
    assert b - a == pytest.approx(c - b)
    with pytest.raises(ValueError):
        ac.lr_at_step(total, n, spe)
    with pytest.raises(ValueError):
        ac.lr_at_step(-1, n, spe)


def test_schedule_params_match_arm_contract():
    params = ac.schedule_params(33, REAL_STEPS_PER_EPOCH)
    assert params["warmup_epochs"] == 2
    assert params["warmup_steps"] == 2 * REAL_STEPS_PER_EPOCH
    assert params["total_steps"] == 33 * REAL_STEPS_PER_EPOCH
    assert params["lr_at_step_0"] == pytest.approx(1e-5)
    assert params["lr_at_first_cosine_step"] == pytest.approx(1e-4)
    assert params["lr_at_final_step"] == pytest.approx(1e-6, abs=1e-12)


def test_arm_plan_structure_and_totals():
    plan_a = ac.build_arm_plan("A", 15, 33)
    plan_b = ac.build_arm_plan("B", 15, 33)
    plan_c = ac.build_arm_plan("C", 15, 33)
    assert [p["epochs"] for p in plan_a["phases"]] == [48]
    assert [p["epochs"] for p in plan_b["phases"]] == [15, 33]
    assert [p["epochs"] for p in plan_c["phases"]] == [33]
    assert plan_b["phases"][0]["visible_side"] == "z4"
    assert plan_b["phases"][0]["lr_schedule"] == "constant_1e-4"
    assert plan_b["phases"][0]["alpha"] == 0
    for plan in (plan_a, plan_b, plan_c):
        assert plan["total_model_training_epochs"] <= 48
        assert plan["adam_constructor"]["weight_decay"] == 0.0
        assert plan["adam_constructor"]["amsgrad"] is False
    assert plan_b["total_model_training_epochs"] == 48
    assert plan_c["total_model_training_epochs"] == 33
    with pytest.raises(ValueError):
        ac.build_arm_plan("B", 16, 33)  # 16 + 33 != 48
    with pytest.raises(ValueError):
        ac.build_arm_plan("D", 15, 33)


def test_b_phase2_equals_c_at_every_t4_step():
    equality = ac.assert_b_phase2_matches_c_plan(
        ac.build_arm_plan("B", 15, 33), ac.build_arm_plan("C", 15, 33)
    )
    assert equality["t4_phase_epochs_equal"] and equality["lr_schedule_equal"]
    assert equality["adam_constructor_equal"]
    # same function, same phase-local arguments: enumerate the LR value at
    # EVERY T4-phase step of the real schedule plus the boundary regions
    n, spe = 33, REAL_STEPS_PER_EPOCH
    lr_b = lambda step: ac.lr_at_step(step, n, spe)
    lr_c = lambda step: ac.lr_at_step(step, n, spe)
    total = n * spe
    check_steps = list(range(0, 2 * spe + 10)) + list(range(total - 10, total))
    check_steps += list(range(2 * spe + 10, total, 997))
    for step in check_steps:
        assert lr_b(step) == lr_c(step)  # bit-identical float
    # arm A uses the same schedule family with its own 48-epoch horizon
    assert ac.lr_at_step(0, 48, spe) == pytest.approx(1e-5)
    assert ac.lr_at_step(48 * spe - 1, 48, spe) == pytest.approx(1e-6, abs=1e-12)
    # mismatched plans are refused
    with pytest.raises(ValueError):
        ac.assert_b_phase2_matches_c_plan(
            ac.build_arm_plan("B", 14, 34), ac.build_arm_plan("C", 15, 33)
        )


# ---- §4 admission ----------------------------------------------------------
def test_admission_is_post_normalization_and_bitwise():
    torch.manual_seed(0)
    normalized_t4 = torch.randn(3, 7, 4) - 3.0  # arbitrary normalized values
    z4 = ac.admit_side(normalized_t4, "z4")
    assert z4.shape == normalized_t4.shape and z4.dtype == normalized_t4.dtype
    assert torch.count_nonzero(z4).item() == 0
    assert not z4.signbit().any().item()  # bitwise +0.0, never -0.0
    # even a -0.0 / negative input cannot leak through zeros_like
    weird = normalized_t4.clone()
    weird[0, 0, 0] = -0.0
    assert not ac.admit_side(weird, "z4").signbit().any().item()
    # t4 admission returns the exact tensor by identity (never alpha * raw)
    out = ac.admit_side(normalized_t4, "t4")
    assert out is normalized_t4
    with pytest.raises(ValueError):
        ac.admit_side(normalized_t4, "ramp")


# ---- §3 phase-1 zero invariants on the real model ---------------------------
def _real_model():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tfpd_spintshape_module_for_arm_tests",
        ROOT / "src/tfpd/spintshape_module.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_spintshape_model(seed=42)


def test_phase1_z4_step_keeps_w_side_and_moments_exactly_zero():
    torch.manual_seed(0)
    model = _real_model()
    n_units = 12
    neural = torch.rand(2, 50, n_units)
    calib = torch.rand(2, 30, 100, n_units)
    side_t4 = torch.randn(2, n_units, 4)
    side_z4 = ac.admit_side(side_t4, "z4")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    model.train()
    prediction, _ = model(neural, calib_trials=calib, side_features=side_z4)
    loss = ((prediction - torch.zeros_like(prediction)) ** 2).mean()
    loss.backward()
    encoder = model.id_encoder
    hidden, side_dim = encoder.hidden_dim, encoder.side_dim
    grad_block = encoder.post_pool[0].weight.grad[:, hidden : hidden + side_dim]
    assert int(torch.count_nonzero(grad_block).item()) == 0  # grad bitwise zero
    optimizer.step()
    w_after = ac.w_side_block(model)
    assert int(torch.count_nonzero(w_after).item()) == 0
    assert not w_after.signbit().any().item()
    exp_avg = ac.w_side_moment(model, optimizer, "exp_avg")
    exp_avg_sq = ac.w_side_moment(model, optimizer, "exp_avg_sq")
    assert int(torch.count_nonzero(exp_avg).item()) == 0
    assert int(torch.count_nonzero(exp_avg_sq).item()) == 0
    # T4 admission, by contrast, immediately produces a nonzero gradient
    optimizer.zero_grad(set_to_none=True)
    prediction, _ = model(neural, calib_trials=calib, side_features=side_t4)
    ((prediction - torch.zeros_like(prediction)) ** 2).mean().backward()
    grad_t4 = encoder.post_pool[0].weight.grad[:, hidden : hidden + side_dim]
    assert int(torch.count_nonzero(grad_t4).item()) > 0


def test_fixed_batch_diagnostics_and_attention_patch_roundtrip():
    torch.manual_seed(0)
    model = _real_model()
    n_units = 8
    calib = torch.rand(2, 30, 100, n_units)
    side_t4 = torch.randn(2, n_units, 4)
    neural = torch.rand(2, 50, n_units)

    contribution = ac.post_pool_contribution(model, calib, side_t4)
    assert contribution["norm_w_side_times_t4_fixed_batch"] == 0.0  # W_side == 0
    assert contribution["norm_calibration_contribution_fixed_batch"] > 0.0
    assert contribution["ratio_t4_to_calibration_at_post_pool0"] == 0.0

    model.train()
    summary = ac.attention_summary(
        model, neural, calib, ac.admit_side(side_t4, "z4")
    )
    assert summary["n_layers_captured"] >= 1
    assert math.isfinite(summary["mean_attention_entropy"])
    assert 0.0 < summary["mean_max_attention_weight"] <= 1.0
    # the instance-level patch is fully removed and training mode restored
    for layer in model.decoder.transformer.layers:
        assert "forward" not in layer.cross_attn.__dict__
    assert model.training is True
    # entropy is finite under T4 admission too
    summary_t4 = ac.attention_summary(model, neural, calib, side_t4)
    assert math.isfinite(summary_t4["mean_attention_entropy"])


# ---- canonical initial state artifact ---------------------------------------
def _run_cli(script: str, *argv: str, expect: int | None = 0):
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *argv],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
        timeout=900,
    )
    if expect is not None:
        assert proc.returncode == expect, proc.stderr[-3000:]
    return proc


def test_canonical_initial_state_mint_once_and_refuse_twice(tmp_path):
    out = tmp_path / "arms"
    proc = _run_cli(
        "make_canonical_initial_state.py", "--output-dir", str(out)
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "CANONICAL_INITIAL_STATE_SEALED"
    artifact = out / "canonical_initial_state.pt"
    assert artifact.is_file()
    receipt = json.loads((out / "canonical_initial_state_receipt.json").read_text())
    assert receipt["strict_reload_bitwise_equal"] is True
    assert receipt["w_side_exact_positive_zero"] is True
    loaded = torch.load(artifact, map_location="cpu", weights_only=False)
    assert loaded["state_sha256"] == receipt["state_dict_sha256"]
    # immutable: a second mint is refused with exit 2
    _run_cli("make_canonical_initial_state.py", "--output-dir", str(out), expect=2)


def test_arm_runner_preflight_c_passes_and_seals(tmp_path):
    out = tmp_path / "arms"
    _run_cli("make_canonical_initial_state.py", "--output-dir", str(out))
    proc = _run_cli(
        "run_admission_arm.py",
        "--arm", "C",
        "--preflight-only",
        "--device", "cpu",
        "--output-root", str(out),
        "--initial-state", str(out / "canonical_initial_state.pt"),
        "--num-workers", "2",
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ARM_PREFLIGHT_PASSED"
    assert payload["t_pre"] == 15 and payload["e_t4"] == 33
    assert payload["steps_per_epoch"] == REAL_STEPS_PER_EPOCH
    receipt = json.loads((out / "preflight_armC.json").read_text())
    assert receipt["data_contract"]["within_dev_sessions_opened"] is False
    assert receipt["data_contract"]["formal_or_organizer_held_data_opened"] is False
    assert receipt["z4_authority"]["all_bitwise_equal"] is True
    assert receipt["b_phase2_vs_c_equality"]["lr_schedule_equal"] is True
    assert receipt["initial_state"]["strict_load"] is True
    # preflight never creates the arm run directory
    assert not (out / "armC_direct_t4_exposure_matched").exists()


def test_arm_runner_requires_initial_state_and_refuses_existing_dir(tmp_path):
    out = tmp_path / "arms"
    proc = _run_cli(
        "run_admission_arm.py", "--arm", "A", "--device", "cpu",
        "--output-root", str(out),
        "--initial-state", str(out / "not_minted_yet.pt"),
        expect=None,
    )
    assert proc.returncode != 0
    assert "canonical initial state artifact missing" in proc.stderr
