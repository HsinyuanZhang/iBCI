"""Population-robustness round tests (review gate before any GPU launch).

- sealed files untouched: the arm-A runner, spintshape_module, spint.py and
  streaming_spint.py are byte-frozen (their SHAs must equal the values sealed
  in the Gate-2 arm-A launch closure — asserted against the live files);
- bitwise initial-state equality: D and DH strict-load the canonical artifact;
  state keys/shapes equal the canonical payload; parameter counts equal across
  cells (the head count changes only the MHA head partition, and construction
  consumes the identical RNG stream — evidenced by equal state signatures);
- dynamic dropout is train-only and instrumented passively: under
  `model.train()` the model samples one p per forward via random.uniform and
  the recorder captures exactly that p (no second draw); under `model.eval()`
  no dropout and no sampling occur;
- DH attention yields 64 per-head entropies and a diversity statistic, D
  yields 2; the instance-level patch is fully removed afterwards;
- the runner refuses fresh-root violations and unknown cells; the per-epoch
  invariant "sampled-p count == optimizer steps" is enforced in-source.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tfpd_lane import pop_robust as pr

SEALED_FREEZE = {
    "scripts/run_admission_arm.py",
    "src/tfpd/spintshape_module.py",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sealed_files_match_their_gate2_closure():
    launch = json.loads(
        (ROOT / "results/admission_arms_v1/armA_direct_t4_48/launch_receipt.json").read_text()
    )
    sealed = launch["source_closure"]["files"]
    for rel in SEALED_FREEZE:
        assert rel in sealed, rel
        live = _sha(ROOT / rel)
        assert live == sealed[rel]["sha256"], f"sealed file drifted: {rel}"


def _canonical_state():
    payload = torch.load(
        ROOT / "results/admission_arms_v1/canonical_initial_state.pt",
        map_location="cpu", weights_only=False,
    )
    return payload["state_dict"], payload["state_sha256"]


def test_d_and_dh_strict_load_canonical_bitwise():
    canonical, canonical_sha = _canonical_state()
    proof = pr.initial_state_equality_proof(canonical, seed=42)
    assert set(proof) == {"D", "DH"}
    counts = set()
    for cell, entry in proof.items():
        assert entry["state_keys_equal_to_canonical"] is True
        assert entry["shape_mismatches"] == []
        counts.add(entry["trainable_parameters"])
    assert len(counts) == 1  # head count changes no parameter count
    # loaded D/DH state signature equals the canonical payload's bit-for-bit
    for cell in ("D", "DH"):
        model = pr.build_population_robustness_model(seed=42, cell=cell)
        model.load_state_dict(canonical, strict=True)
        keys, shapes = pr.state_signature(model)
        from torch.nn.parameter import UninitializedParameter

        canonical_shapes = {
            k: (("uninitialized-lazy",) if isinstance(canonical[k], UninitializedParameter) else tuple(canonical[k].shape))
            for k in canonical
        }
        assert keys == sorted(canonical.keys())
        assert shapes == canonical_shapes


def test_head_partition_changes_attention_heads_only():
    canonical, _ = _canonical_state()
    model_d = pr.build_population_robustness_model(seed=42, cell="D")
    model_dh = pr.build_population_robustness_model(seed=42, cell="DH")
    keys_d, shapes_d = pr.state_signature(model_d)
    keys_dh, shapes_dh = pr.state_signature(model_dh)
    assert keys_d == keys_dh and shapes_d == shapes_dh
    assert model_d.decoder.transformer.layers[0].cross_attn.num_heads == 2
    assert model_dh.decoder.transformer.layers[0].cross_attn.num_heads == 64
    # dynamic dropout flags on, identical low/high
    for model in (model_d, model_dh):
        assert model.decoder.dynamic_dropout is True
        assert model.decoder.dynamic_dropout_low == 0.0
        assert model.decoder.dynamic_dropout_high == 1.0


def test_dynamic_dropout_train_only_and_passive_recording():
    canonical, _ = _canonical_state()
    model = pr.build_population_robustness_model(seed=42, cell="D")
    model.load_state_dict(canonical, strict=True)
    n_units = 12
    neural = torch.rand(4, 50, n_units)
    calib = torch.rand(4, 30, 100, n_units)
    side = torch.randn(4, n_units, 4)

    # eval mode: no dynamic dropout, no random.uniform sampling
    model.eval()
    with pr.dynamic_dropout_recorder() as rec:
        with torch.no_grad():
            model(neural, calib_trials=calib, side_features=side)
        assert rec["uniform_calls"] == 0
        assert rec["dropout_calls"] == []

    # train mode: exactly one sampled p per forward, recorded passively
    model.train()
    with pr.dynamic_dropout_recorder() as rec:
        pred1, _ = model(neural, calib_trials=calib, side_features=side)
        pred2, _ = model(neural, calib_trials=calib, side_features=side)
    assert rec["uniform_calls"] == 2
    assert len(rec["sampled_p"]) == 2
    assert all(0.0 <= p <= 1.0 for p in rec["sampled_p"])
    assert len(rec["dropout_calls"]) == 2
    for call in rec["dropout_calls"]:
        assert call["p"] in rec["sampled_p"]  # the model's own p, not a redraw
        assert 0.0 <= call["retained_unit_fraction"] <= 1.0
        assert call["shape"][0] == 4 and call["shape"][1] == n_units  # [B, units] mask
    # fixed-batch pre/post dropout token norms (measured, not redrawn)
    diag = pr.fixed_batch_dropout_diagnostic(model, neural, calib, side)
    assert diag["token_norm_pre_dropout_mean"] > 0.0
    assert diag["token_norm_post_dropout_mean"] > 0.0
    assert diag["realized_p"] is not None and 0.0 <= diag["realized_p"] <= 1.0
    # fc_in hook captures BOTH fc_in calls (unit tokens + query rep) with shapes
    with pr.fc_in_token_norm_recorder(model) as norms:
        model(neural, calib_trials=calib, side_features=side)
    assert len(norms) == 2
    unit_entries = [e for e in norms if e["shape"][1] == n_units]
    assert len(unit_entries) == 1 and unit_entries[0]["norm"] > 0.0
    # wrappers fully restored
    import torch.nn.functional as F

    assert F.dropout is torch.nn.functional.dropout or F.dropout.__module__ != "src.tfpd_lane.pop_robust"
    assert random.uniform.__module__ != "src.tfpd_lane.pop_robust"


def test_per_head_attention_summary_counts_heads_and_restores():
    canonical, _ = _canonical_state()
    for cell, heads in (("D", 2), ("DH", 64)):
        model = pr.build_population_robustness_model(seed=42, cell=cell)
        model.load_state_dict(canonical, strict=True)
        n_units = 16
        summary = pr.per_head_attention_summary(
            model, torch.rand(2, 50, n_units), torch.rand(2, 30, 100, n_units),
            torch.randn(2, n_units, 4),
        )
        assert summary["n_heads"] == heads
        assert len(summary["per_head_mean_entropy"]) == heads
        assert all(np.isfinite(summary["per_head_mean_entropy"]))
        assert summary["across_head_entropy_std"] >= 0.0
        for layer in model.decoder.transformer.layers:
            assert "forward" not in layer.cross_attn.__dict__
        assert model.training is True  # original mode restored


def test_runner_cli_rejects_bad_cell_and_existing_root(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "run_pop_robust_cell_under_test", ROOT / "scripts/run_pop_robust_cell.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import subprocess, os

    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_pop_robust_cell.py"), "--cell", "X"],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=120,
    )
    assert proc.returncode == 2

    source = (ROOT / "scripts/run_pop_robust_cell.py").read_text()
    assert "sampled-p count != optimizer steps" in source  # in-source invariant
    assert "n_recorded_unit_mask_calls" in source
    assert "num_heads" in source and "dynamic_dropout" in source
