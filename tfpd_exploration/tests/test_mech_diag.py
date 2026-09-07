"""Focused tests for the §5.7 mechanism diagnostics (Deliverable B).

- checkpoint roundtrip: a synthetic Lightning-style checkpoint rebuilds the
  exact same decoder via state_dict (both Stage-1 model families), without
  instantiating the LightningModule;
- activity-destroyed arm preserves each unit's marginal count sequence,
  changes the input, and is seed-frozen;
- the four diagnostics produce finite R^2 with consistent deltas, scored by
  the house metric (`src.tfpd.synth.r2_score`);
- wrong-pair reuses the model's own frozen `wrong_pair_carrier`;
- CLI: dry-run prints the plan matrix and exits 0; the real-datamodule
  interface refuses without instantiating anything.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.tfpd.bilinear_readin import BilinearTaskFrameDecoder
from src.tfpd.population_vector import LearnedPopulationVectorDecoder
from src.tfpd.synth import generate_session, r2_score
from src.tfpd_lane import mech_diag as md

ROOT = Path(__file__).resolve().parents[1]


def _fake_checkpoint(tmp_path: Path, model_name: str, seed: int) -> Path:
    torch.manual_seed(seed)
    if model_name == "bilinear":
        model = BilinearTaskFrameDecoder(
            window_size=20, carrier_dim=4, feature_dim=16, embed_dim=16,
            hidden_dim=64, latent_dim=32, gru_hidden=64, num_covariates=2,
        )
    else:
        model = LearnedPopulationVectorDecoder(
            window_size=20, carrier_dim=4, hidden_dim=64,
            basis_dim=16, gru_hidden=64, num_covariates=2,
        )
    ckpt = {
        "epoch": 0,
        "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
        "hyper_parameters": {"model_name": model_name, "arm": "t4", "lr": 1e-4, "seed": seed},
    }
    path = tmp_path / f"{model_name}.ckpt"
    torch.save(ckpt, path)
    return path


@pytest.mark.parametrize("model_name", ["bilinear", "population_vector"])
def test_checkpoint_roundtrip(tmp_path, model_name):
    ckpt_path = _fake_checkpoint(tmp_path, model_name, seed=42)
    original = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    reference = (
        BilinearTaskFrameDecoder(
            window_size=20, carrier_dim=4, feature_dim=16, embed_dim=16,
            hidden_dim=64, latent_dim=32, gru_hidden=64, num_covariates=2,
        )
        if model_name == "bilinear"
        else LearnedPopulationVectorDecoder(
            window_size=20, carrier_dim=4, hidden_dim=64,
            basis_dim=16, gru_hidden=64, num_covariates=2,
        )
    )
    reference.load_state_dict(
        {k[len("model."):]: v for k, v in original["state_dict"].items()}
    )
    reference.eval()

    rebuilt = md.load_stage1_model(ckpt_path)
    session = generate_session(seed=1, num_units=32)
    with torch.no_grad():
        out_ref = reference(session.counts.unsqueeze(0), session.carrier.unsqueeze(0))
        out_new = rebuilt(session.counts.unsqueeze(0), session.carrier.unsqueeze(0))
    torch.testing.assert_close(out_new, out_ref)


def test_checkpoint_rejects_unknown(tmp_path):
    path = tmp_path / "bad.ckpt"
    torch.save({"state_dict": {"model.x": torch.zeros(1)}}, path)
    with pytest.raises(ValueError):
        md.load_stage1_model(path)


def test_destroy_activity_preserves_marginals():
    session = generate_session(seed=2, num_units=24)
    destroyed = md.destroy_activity(session.counts, seed=0)
    assert destroyed.shape == session.counts.shape
    assert not torch.equal(destroyed, session.counts)
    # same multiset of counts per unit (time structure alone is destroyed)
    for i in range(session.counts.shape[1]):
        assert torch.equal(torch.sort(session.counts[:, i])[0], torch.sort(destroyed[:, i])[0])
    # frozen seed: identical reproduction
    assert torch.equal(destroyed, md.destroy_activity(session.counts, seed=0))
    assert not torch.equal(destroyed, md.destroy_activity(session.counts, seed=1))


def test_diagnostic_inputs_use_model_wrong_pair():
    torch.manual_seed(0)
    model = BilinearTaskFrameDecoder(window_size=4)
    session = generate_session(seed=3, num_units=32)
    inputs = md.diagnostic_inputs(model, session, wrong_pair_seed=5)
    carrier_b = session.carrier.unsqueeze(0)
    torch.testing.assert_close(
        inputs["wrong_pair"][1], model.wrong_pair_carrier(carrier_b, seed=5)
    )
    torch.testing.assert_close(inputs["zero"][1], torch.zeros_like(carrier_b))
    torch.testing.assert_close(inputs["aligned"][1], carrier_b)
    assert torch.equal(inputs["activity_destroyed"][0][0], md.destroy_activity(session.counts, seed=0))


def test_run_mech_diag_consistent_with_house_metric():
    torch.manual_seed(0)
    model = BilinearTaskFrameDecoder(window_size=4)
    session = generate_session(seed=4, num_units=32)
    result = md.run_mech_diag(model, session, wrong_pair_seed=1, destroy_seed=2)

    assert set(result["per_diagnostic_r2"]) == set(md.DIAGNOSTICS)
    assert all(isinstance(v, float) for v in result["per_diagnostic_r2"].values())
    inputs = md.diagnostic_inputs(model, session, wrong_pair_seed=1, destroy_seed=2)
    with torch.no_grad():
        aligned_pred = model(*inputs["aligned"])[0]
    assert result["per_diagnostic_r2"]["aligned"] == pytest.approx(
        r2_score(aligned_pred, session.behaviour)
    )
    deltas = result["deltas"]
    assert deltas["t4_minus_z4"] == pytest.approx(
        result["per_diagnostic_r2"]["aligned"] - result["per_diagnostic_r2"]["zero"]
    )
    assert deltas["t4_minus_wrong_pair"] == pytest.approx(
        result["per_diagnostic_r2"]["aligned"] - result["per_diagnostic_r2"]["wrong_pair"]
    )


def test_run_mech_diag_cohort_smoke():
    torch.manual_seed(0)
    model = BilinearTaskFrameDecoder(window_size=4)
    sessions = md.synthetic_cohort(seed=6, num_sessions=2, num_units=24)
    result = md.run_mech_diag_cohort(model, sessions)
    assert result["schema"] == "tfpd_mech_diag_v1"
    assert result["num_sessions"] == 2
    assert len(result["per_session"]) == 2
    for name in md.DIAGNOSTICS:
        assert isinstance(result["per_diagnostic_r2_mean"][name], float)


def _run_cli(*argv: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_mech_diag.py"), *argv],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
    )


def test_cli_dry_run_exits_zero_with_plan():
    proc = _run_cli("--dry-run")
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert set(plan["diagnostics"]) == set(md.DIAGNOSTICS)
    assert plan["mode"].startswith("dry-run")


def test_cli_datamodule_config_is_interface_only(tmp_path):
    proc = _run_cli("--datamodule-config", str(tmp_path / "cfg.yaml"))
    assert proc.returncode == 4
    assert "TODO" in proc.stderr


def test_cli_synthetic_smoke(tmp_path):
    out = tmp_path / "mech_diag_smoke"
    proc = _run_cli("--synthetic", "2", "--output-root", str(out))
    assert proc.returncode == 0, proc.stderr
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["status"] == "COMPLETED_FORWARD_ONLY"
    assert receipt["model_source"]["mode"] == "synthetic_random_init"
    # immutable root: rerunning refuses to overwrite
    proc2 = _run_cli("--synthetic", "2", "--output-root", str(out))
    assert proc2.returncode == 2
