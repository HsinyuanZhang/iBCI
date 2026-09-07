"""TFAP Stage-0 Gate tests (TFAP_CONTRACT_20260817.md §1).

- direction extraction from `target_pos[active_target]` (validity, geometry);
- the closed-form T4 fit is the SAME least-squares cosine model as 000688:
  recovers a known tuning curve, and per-distinct-direction means are not
  reweighted by trial counts;
- normalizer is 000128's own (self-standardization), std-floored;
- P-Z4 mask is `zeros_like` of the STANDARDIZED tensor — exact positive zero,
  independent of raw values, never `0 * raw`;
- budget: steps/epoch = n_windows // 32, 48 epochs frozen;
- interface consistency (real builder): state keys/shapes identical across
  688/128-shaped forwards, strict=True roundtrip bitwise-equal, partial state
  raises (no silent partial load), no teacher anything;
- CLI: the full CPU Stage-0 preflight passes and seals receipt + payload.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tfpd_lane import tfap_stage0 as ts


def _fake_trials(n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        theta = rng.uniform(-math.pi, math.pi)
        radius = rng.uniform(80, 140)
        n_targets = rng.integers(1, 4)
        positions = rng.normal(size=(int(n_targets), 2)) * 30
        active = int(rng.integers(0, int(n_targets)))
        positions[active] = [radius * math.cos(theta), radius * math.sin(theta)]
        rows.append({
            "start_time": float(i), "stop_time": float(i + 2),
            "target_pos": positions.ravel(),
            "active_target": active,
            "split": "train",
        })
    return pd.DataFrame(rows)


def test_trial_directions_geometry_and_validity():
    df = _fake_trials()
    thetas, valid = ts.trial_directions(df)
    assert valid.all() and np.isfinite(thetas).all()
    expected = [
        math.atan2(p.reshape(-1, 2)[a][1], p.reshape(-1, 2)[a][0])
        for p, a in zip(df["target_pos"], df["active_target"])
    ]
    np.testing.assert_allclose(thetas, expected, rtol=0, atol=1e-12)
    # degenerate row: zero-radius target is invalid, not atan2(0,0)
    rows = _fake_trials(seed=1)
    first = rows.iloc[0].copy()
    first["target_pos"] = [0.0, 0.0, 5.0, 5.0]
    first["active_target"] = 0
    degenerate = pd.concat(
        [first.to_frame().T, rows.iloc[1:]], ignore_index=True
    )
    thetas2, valid2 = ts.trial_directions(degenerate)
    assert not bool(valid2[0])
    assert int(valid2.sum()) == len(degenerate) - 1


def test_closed_form_t4_recovers_known_tuning_and_direction_weighting():
    thetas = np.array([0.0, math.pi / 2, math.pi, -math.pi / 2, 0.0, 0.0])
    # unit 0: rate = 5 + 3*cos(theta)  -> [a,c,m,b] = [3, 0, 3, 5]
    rate0 = 5 + 3 * np.cos(thetas)
    # unit 1: zero spikes everywhere -> exact zero fill
    rates = np.stack([rate0, np.zeros_like(rate0)])
    fit = ts.fit_t4_closed_form(rates, thetas)
    t4 = fit["t4"]
    assert fit["n_distinct_directions"] == 4
    assert fit["zero_spike_units"] == 1
    np.testing.assert_allclose(t4[0], [3.0, 0.0, 3.0, 5.0], atol=1e-5)
    np.testing.assert_allclose(t4[1], 0.0, atol=0)
    # per-distinct-direction means: duplicating a direction with its own rate
    # must not change the fit — trial-count reweighting is forbidden
    thetas_dup = np.array([0.0, 0.0, math.pi / 2, math.pi, -math.pi / 2, 0.0])
    rates_dup = np.stack([5 + 3 * np.cos(thetas_dup), np.zeros(6)])
    fit_dup = ts.fit_t4_closed_form(rates_dup, thetas_dup)
    np.testing.assert_allclose(fit_dup["t4"][0], t4[0], atol=1e-6)
    with pytest.raises(ValueError):
        ts.fit_t4_closed_form(rates, np.array([0.3, 0.3]))  # < 2 distinct directions


def test_own_normalizer_and_z4_mask_semantics():
    rng = np.random.default_rng(1)
    t4 = rng.normal(size=(50, 4)).astype(np.float32) * np.array([3, 3, 5, 8], np.float32)
    mean, std = ts.fit_t4_normalizer(t4)
    standardized = ts.standardize_t4(t4, mean, std)
    np.testing.assert_allclose(standardized.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(standardized.std(axis=0), 1.0, atol=1e-4)
    # constant column gets the std floor, never division by ~0
    t4_const = t4.copy()
    t4_const[:, 3] = 2.0
    mean_c, std_c = ts.fit_t4_normalizer(t4_const)
    assert std_c[3] == 1.0
    z4 = ts.mask_z4(standardized)
    assert np.all(z4 == 0) and not np.any(np.signbit(z4))
    assert z4.dtype == standardized.dtype and z4.shape == standardized.shape
    # the mask is zeros_like of the standardized tensor, not 0 * raw
    weird = standardized.copy()
    weird[0, 0] = -0.0
    assert not np.any(np.signbit(ts.mask_z4(weird)))
    # semantic hash is 000688-format and deterministic
    assert ts.normalizer_semantic_sha256(mean, std) == ts.normalizer_semantic_sha256(mean, std)
    assert len(ts.normalizer_semantic_sha256(mean, std)) == 64


def test_budget_freeze_and_steps():
    assert ts.PRETRAIN_EPOCHS == 48
    assert ts.TRAIN_BATCH_SIZE == 32
    assert ts.steps_per_epoch(6400) == 200
    assert ts.steps_per_epoch(6423) == 200  # drop-partial
    assert ts.steps_per_epoch(31) == 0


def test_interface_consistency_single_builder_two_datasets():
    proof = ts.interface_consistency_proof(seed=42)
    assert proof["builder"].endswith("spintshape_module.py:build_spintshape_model")
    assert proof["state_keys_identical_across_688_and_128_batches"] is True
    assert proof["state_shapes_identical_across_688_and_128_batches"] is True
    assert proof["strict_roundtrip_bitwise_equal"] is True
    assert proof["partial_state_raises"] is True  # no silent partial load
    assert proof["forward_finite_688_shaped"] and proof["forward_finite_128_shaped"]
    assert proof["teacher_checkpoint_or_logits_or_loss_used"] is False
    assert proof["output_shapes"] == {"n688": [2, 50, 2], "n128": [2, 50, 2]}


def _run_cli(*argv: str):
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_tfap_stage0.py"), *argv],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=1800,
    )


def test_stage0_cli_full_preflight_passes_and_seals(tmp_path):
    out = tmp_path / "stage0"
    proc = _run_cli("--output-root", str(out))
    assert proc.returncode == 0, proc.stderr[-3000:]
    summary = json.loads(proc.stdout)
    assert summary["status"] == "STAGE0_PREFLIGHT_PASSED"
    assert summary["n_train_units"] == 137
    assert summary["interface_all_green"] is True
    assert summary["n_distinct_directions"] >= 2
    assert summary["steps_per_epoch"] > 0 and summary["total_steps_48ep"] == summary["steps_per_epoch"] * 48
    receipt = json.loads((out / "stage0_receipt.json").read_text())
    assert receipt["p_z4_mask"]["exact_positive_zero"] is True
    assert receipt["disclosures"]["dandi_000688_opened"] is False
    assert receipt["disclosures"]["gpu_used"] is False
    assert (out / "jenkins_derived_payload.npz.sha256").is_file()
    # immutable root: a rerun refuses
    proc2 = _run_cli("--output-root", str(out))
    assert proc2.returncode == 2
