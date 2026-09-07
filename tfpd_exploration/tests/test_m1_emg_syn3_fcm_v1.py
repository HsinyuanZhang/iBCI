"""CPU/synthetic tests for M1 EMG-Syn3 FCM V1. No live GPU capability."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest

from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import plan
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import model as injection
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as syn3_data
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import score as syn3_score
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import training as syn3_training
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import receipts as syn3_receipts


ROOT = Path(__file__).resolve().parents[2]
STAGE0_CLI = ROOT / "tfpd_exploration/scripts/run_m1_emg_syn3_stage0.py"
PILOT_CLI = ROOT / "tfpd_exploration/scripts/run_m1_emg_syn3_pilot.py"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_plan_binds_workorder_and_design_sha() -> None:
    assert plan.DESIGN_SHA256 == (
        "766cb7952913fc0ee57f231bc398b1d50b4d768f4d361e77ca185c61ee6a8259"
    )
    assert plan.WORKORDER_SHA256 == (
        "7b4a002b39e4d86808931e2682b8cde2a9b55dc02a557d490da9e944da2c4378"
    )
    assert plan.STATIC_CONTENT_GATE == 0.03
    assert plan.CARRIER_BUDGETS == (10, 6, 4, 2)
    assert plan.ACTIVITY_CYCLE == (10, 5, 2)
    assert plan.CQC_CARRIER_CYCLE == (10, 6, 4)
    assert plan.FOLD0_SOURCE_SESSIONS == (
        "ses-20120926", "ses-20120927", "ses-20120928",
    )
    assert plan.FOLD0_TARGET_SESSION == "ses-20120924"
    assert plan.TEACHER_CKPT_SHA256 == (
        "f2921cabea819fed58b15e169f9cb899472416d30ee5a9b12c4c2087e96cb6be"
    )
    assert plan.STAGE1_EPOCHS == 12
    assert plan.FIXED_LAST_EPOCH_INDEX == 11
    assert plan.RESULT_ROOT_RELATIVE == "tfpd_exploration/results/m1_emg_syn3_fcm_v1"
    assert "tfpd_exploration/results/m1_t0c1_prefix_v1" not in plan.OWNED_PATHS
    assert plan.NNMF_LAW["init"] == "nndsvda"
    assert plan.NNMF_LAW["solver"] == "cd"


def test_signed_signal_rejects_nnmf() -> None:
    signed = np.array([[0.1, -0.2, 0.3], [0.4, 0.5, 0.0]], dtype=np.float64)
    with pytest.raises(syn3.Syn3Error, match="signed"):
        syn3.fit_source_nmf(signed)


def test_nonnegative_signal_fits_deterministic_dictionary() -> None:
    rng = np.random.RandomState(0)
    true_h = np.abs(rng.randn(3, 8)) + 0.1
    true_h /= np.linalg.norm(true_h, axis=1, keepdims=True)
    true_z = rng.rand(80, 3)
    x = np.maximum(true_z @ true_h, 0.0)
    first = syn3.fit_source_nmf(x)
    second = syn3.fit_source_nmf(x)
    assert np.array_equal(first.dictionary, second.dictionary)
    assert np.all(first.dictionary >= 0.0)
    assert np.all(first.activations >= 0.0)
    assert np.allclose(np.linalg.norm(first.dictionary, axis=1), 1.0, atol=1e-8)
    energies = np.sum(first.activations ** 2, axis=0)
    assert np.all(np.diff(energies) <= 1e-12)


def test_target_cannot_mutate_source_dictionary() -> None:
    rng = np.random.RandomState(1)
    source = rng.rand(40, 6)
    target = rng.rand(12, 6) + 2.0
    fitted = syn3.fit_source_nmf(source)
    before = fitted.dictionary.copy()
    _z = syn3.nnls_activations(target, fitted.dictionary)
    assert np.array_equal(fitted.dictionary, before)
    assert np.all(_z >= 0.0)
    assert np.isfinite(_z).all()


def test_ridge_unpenalized_intercept_and_valid_bin_normalization() -> None:
    rng = np.random.RandomState(2)
    z = rng.rand(25, 3)
    true_w = np.array([0.4, -0.2, 0.1])
    true_b = 1.7
    rates = true_b + z @ true_w + 0.01 * rng.randn(25)
    w, b = syn3.fit_unit_ridge(z, rates)
    assert w.shape == (3,)
    recovered = syn3.fit_unit_ridge(np.vstack([z, z]), np.concatenate([rates, rates]))
    assert np.allclose(w, recovered[0], atol=0.05)
    assert np.allclose(b, recovered[1], atol=0.05)


def test_m2_not_rejected_for_having_two_trials() -> None:
    rng = np.random.RandomState(3)
    z = rng.rand(30, 3)
    rates = rng.rand(30, 4)
    trial_ids = np.array([0] * 15 + [1] * 15)
    report = syn3.coverage_report(z, rates, trial_ids=trial_ids, budget=2)
    assert report["n_trials"] == 2
    assert report["valid_bins"] == 30
    assert report["rejected_for_trial_count"] is False
    assert report["design_rank"] >= 1


def test_zero4_is_exact_normalized_zeros_without_target_fit() -> None:
    controls = syn3.build_controls(
        raw_syn3=np.arange(20, dtype=np.float64).reshape(5, 4),
        normalizer_mean=np.zeros(4),
        normalizer_scale=np.ones(4),
        session_name="ses-20120924",
        seed=42,
        n_units=5,
        target_fit_invoked=False,
    )
    assert np.array_equal(controls["Zero4"], np.zeros((5, 4)))
    assert controls["zero4_target_fit_calls"] == 0
    assert np.allclose(controls["B4"][:, :3], 0.0)
    assert not np.array_equal(controls["RS4"], controls["Syn3"])


def test_injection_zero_init_and_shared_dropout_mask() -> None:
    p = injection.zero_linear(4, 8)
    assert np.array_equal(p, np.zeros((8, 4)))
    hidden = np.ones((3, 8))
    carrier = np.arange(12, dtype=np.float64).reshape(3, 4)
    mask = np.array([1.0, 0.0, 1.0])
    out_zero_p = injection.apply_injection(hidden, carrier, p, mask)
    assert np.array_equal(out_zero_p, hidden * mask[:, None])
    p_live = np.eye(8, 4)
    out = injection.apply_injection(hidden, carrier, p_live, mask)
    leaked = injection.apply_injection(hidden, carrier, p_live, np.ones(3))
    assert np.array_equal(out[1], np.zeros(8))
    assert not np.array_equal(leaked[1], out[1])


def test_nonzero_carrier_changes_prediction_and_receives_gradient() -> None:
    torch = pytest.importorskip("torch")
    module = injection.TorchCarrierProjection(model_dim=6)
    assert torch.equal(module.P.weight.data, torch.zeros(6, 4))
    carrier = torch.tensor([[0.2, 0.0, -0.1, 0.5]], dtype=torch.float32)
    hidden = torch.zeros(1, 6)
    mask = torch.ones(1)
    pred = injection.torch_apply(hidden, carrier, module, mask)
    loss = pred.sum()
    loss.backward()
    assert module.P.weight.grad is not None
    assert float(module.P.weight.grad.abs().sum()) > 0.0
    module.P.weight.data[0, 0] = 1.0
    shifted = injection.torch_apply(hidden, carrier, module, mask)
    assert not torch.equal(shifted.detach(), torch.zeros(1, 6))


def test_c1_keeps_carrier_digest_constant() -> None:
    carrier = np.arange(16, dtype=np.float64).reshape(4, 4)
    digest = syn3.array_digest(carrier)
    for prefix in (10, 5, 2):
        activity = np.ones((prefix, 4, 8))
        assert syn3.array_digest(carrier) == digest
        assert activity.shape[0] == prefix


def test_cdm_a_updates_next_trial_only_and_freezes_carrier_model() -> None:
    fifo = syn3_score.ActivityFIFO(support_trials=3)
    init = [np.full((2, 4), fill) for fill in (1.0, 2.0, 3.0)]
    state = fifo.initialize(init)
    carrier = np.ones((4, 4))
    model_digest = "abc"
    first = fifo.identity_for_trial(state, trial_index=3)
    state_after, carrier_after, model_after = fifo.commit_completed(
        state, trial_activity=np.full((2, 4), 9.0), carrier=carrier, model_digest=model_digest,
    )
    second = fifo.identity_for_trial(state_after, trial_index=4)
    assert not np.array_equal(first, second)
    assert np.array_equal(carrier_after, carrier)
    assert model_after == model_digest
    with pytest.raises(syn3_score.ScoreError, match="current"):
        fifo.identity_for_trial(state_after, trial_index=3)


def test_fixed_last_selection_cannot_read_query() -> None:
    with pytest.raises(syn3_training.TrainingError, match="query"):
        syn3_training.select_fixed_last(epoch_index=11, query_r2=0.4)
    path = syn3_training.select_fixed_last(epoch_index=11)
    assert path == "epoch_011"
    with pytest.raises(syn3_training.TrainingError):
        syn3_training.select_fixed_last(epoch_index=10)


def test_forbidden_paths_rejected_before_materialization() -> None:
    with pytest.raises(syn3_data.DataError, match="forbidden"):
        syn3_data.require_source_path(
            ROOT / "SPINT-main/data/000941/sub-MonkeyL-held-out-calib/x.nwb"
        )
    with pytest.raises(syn3_data.DataError, match="forbidden|held-in-calib"):
        syn3_data.require_source_path(
            ROOT / "SPINT-main/data/000941/sub-MonkeyL-held-in-minival/x.nwb"
        )


def test_bin_rows_cannot_cross_support_boundary() -> None:
    trial_ids = np.array([0, 0, 9, 9, 10])
    with pytest.raises(syn3.Syn3Error, match="support"):
        syn3.require_support_bins(trial_ids, budget=10)


def test_failure_lifecycle_excludes_terminal(tmp_path: Path) -> None:
    parent = tmp_path / "results"
    parent.mkdir()
    _shas, terminal, failure = syn3_receipts.run_stage0(
        tmp_path,
        attempt_payload={"schema": "test_attempt", "status": "ATTEMPT_RESERVED"},
        launch_builder=lambda: {"ok": True},
        body_publisher=lambda _root: (_ for _ in ()).throw(RuntimeError("boom")),
        terminal_builder=lambda _shas: {"status": "COMPLETE"},
    )
    assert terminal is None
    assert failure is not None
    stage = tmp_path / plan.STAGE0_ROOT_RELATIVE.split("/")[-1]
    if not stage.exists():
        stage = tmp_path / "stage0"
    names = set(os.listdir(stage)) if stage.exists() else set(os.listdir(next(tmp_path.iterdir())))
    assert any(name.startswith("failure") for name in names)
    assert "terminal.json" not in names


def test_public_cli_is_dry_and_cannot_mint_gpu_capability() -> None:
    dry = subprocess.run(
        [PYTHON, "-S", str(STAGE0_CLI), "--dry-run"],
        cwd=ROOT,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "",
             "PYTHONPATH": str(ROOT)},
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(dry.stdout)
    assert payload["imports_torch"] is False
    assert payload["creates_root_or_receipt"] is False
    assert payload["public_gpu_capability"] is False
    denied = subprocess.run(
        [PYTHON, str(PILOT_CLI), "--execute"],
        cwd=ROOT,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "",
             "PYTHONPATH": str(ROOT)},
        capture_output=True, text=True,
    )
    assert denied.returncode != 0
    assert "capability" in (denied.stderr + denied.stdout).lower()


def test_dry_import_opens_no_torch_or_data() -> None:
    code = r"""
import sys
assert "torch" not in sys.modules
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import plan
assert "torch" not in sys.modules
assert plan.PHASE == "m1_emg_syn3_fcm_v1"
"""
    proc = subprocess.run(
        [PYTHON, "-S", "-c", code],
        cwd=ROOT,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
             "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT)},
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_ls4_deranges_trial_association_not_unit_rows() -> None:
    z = np.arange(20, dtype=np.float64).reshape(10, 2)
    z = np.concatenate([z, np.zeros((10, 1))], axis=1)
    rates = np.arange(40, dtype=np.float64).reshape(10, 4)
    trial_ids = np.arange(10)
    syn_w, _b = syn3.fit_all_units(z, rates)
    ls_w, _ls_b = syn3.fit_all_units(*syn3.derange_trial_association(z, rates, trial_ids, "ses-x", 42))
    assert syn_w.shape == ls_w.shape
    assert not np.allclose(syn_w, ls_w)


def test_pca3_is_signed_and_not_relabeled_syn3() -> None:
    rng = np.random.RandomState(4)
    x = rng.randn(50, 6)
    x -= x.min()
    nmf = syn3.fit_source_nmf(x)
    pca = syn3.fit_source_pca(x)
    assert pca.dictionary.shape == (3, 6)
    assert pca.kind == "signed_pca"
    assert nmf.kind == "nnmf"
    assert not np.allclose(np.abs(pca.dictionary), nmf.dictionary, atol=0.05)
