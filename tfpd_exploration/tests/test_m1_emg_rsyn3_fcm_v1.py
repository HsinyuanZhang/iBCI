"""CPU/synthetic tests for M1 EMG-rSyn3 successor. No live GPU capability."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest

from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import plan
from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import rectify
from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import syn3 as rsyn3
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3 as parent_syn3
from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import training as rsyn3_training
from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import receipts as rsyn3_receipts


ROOT = Path(__file__).resolve().parents[2]
STAGE0_CLI = ROOT / "tfpd_exploration/scripts/run_m1_emg_rsyn3_stage0.py"
PILOT_CLI = ROOT / "tfpd_exploration/scripts/run_m1_emg_rsyn3_pilot.py"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
PARENT_STAGE0 = ROOT / "tfpd_exploration/results/m1_emg_syn3_fcm_v1/stage0"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plan_binds_successor_and_parent_fail_hashes() -> None:
    assert plan.DESIGN_SHA256 == _sha(ROOT / plan.DESIGN_RELATIVE)
    assert plan.WORKORDER_SHA256 == _sha(ROOT / plan.WORKORDER_RELATIVE)
    assert plan.DESIGN_SHA256 == (
        "b5d12b51e74a70d22f50d84ab7078fd0592ba19aae637ed0c193f57497f97cd1"
    )
    assert plan.WORKORDER_SHA256 == (
        "ee9db113f69c80b706436757b0e1221e238acf4cd659d6b9da90bc895c116074"
    )
    assert plan.PARENT_DECISION_SHA256 == (
        "6cc34cdd434917907d8c90b739c3c02003a511c3ae919277baf6de041702bf39"
    )
    assert plan.PARENT_SIGNAL_VIEW_SHA256 == (
        "27e681f86aca6bf7a2d7163cb422ba2c6b1f04c6bb8c5414687335c3453858cf"
    )
    assert plan.PARENT_TERMINAL_SHA256 == (
        "f8436c9e2e1f1e62a3a3fb4f78415cba7bef395c3821437ddb2ee67b99a3f7ba"
    )
    assert plan.RESULT_ROOT_RELATIVE == "tfpd_exploration/results/m1_emg_rsyn3_fcm_v1"
    assert plan.PARENT_STAGE0_ROOT_RELATIVE == (
        "tfpd_exploration/results/m1_emg_syn3_fcm_v1/stage0"
    )
    assert plan.PHASE == "m1_emg_rsyn3_fcm_v1"
    assert plan.RECTIFIER_LAW["name"] == "relu_nonnegative_projection"
    assert plan.RECTIFIER_LAW["threshold"] == 0.0
    assert plan.RECTIFIER_LAW["learnable_parameters"] == 0
    assert plan.STATIC_CONTENT_GATE == 0.03


def test_parent_fail_receipts_remain_byte_identical() -> None:
    assert PARENT_STAGE0.is_dir()
    assert _sha(PARENT_STAGE0 / "decision.json") == plan.PARENT_DECISION_SHA256
    assert _sha(PARENT_STAGE0 / "signal_view.json") == plan.PARENT_SIGNAL_VIEW_SHA256
    assert _sha(PARENT_STAGE0 / "terminal.json") == plan.PARENT_TERMINAL_SHA256
    decision = json.loads((PARENT_STAGE0 / "decision.json").read_text())
    assert decision["decision"] == "FAIL"
    assert "frozen rectifier" in decision["failures"][0]
    assert decision["decoder_r2_computed"] is False
    mode = oct((PARENT_STAGE0 / "decision.json").stat().st_mode & 0o777)
    assert mode == "0o444"


def test_parent_syn3_still_rejects_signed_without_rectifier() -> None:
    signed = np.array([[0.1, -0.2, 0.3], [0.4, 0.5, 0.0]], dtype=np.float64)
    with pytest.raises(parent_syn3.Syn3Error, match="signed"):
        parent_syn3.fit_source_nmf(signed)


def test_relu_is_elementwise_float64_threshold_exact_zero() -> None:
    raw = np.array([[-0.11, 0.0, 1.2], [0.03, -1e-12, 4.0]], dtype=np.float32)
    out = rectify.relu_nonnegative_projection(raw)
    assert out.dtype == np.float64
    assert out.shape == raw.shape
    expected = np.maximum(np.asarray(raw, dtype=np.float64), 0.0)
    assert np.array_equal(out, expected)
    assert float(np.min(out)) >= 0.0
    assert rectify.THRESHOLD == 0.0
    assert rectify.LEARNABLE is False


def test_relu_is_not_abs_offset_envelope_or_smoother() -> None:
    raw = np.array([-0.2, 0.0, 0.5], dtype=np.float64)
    out = rectify.relu_nonnegative_projection(raw)
    assert not np.array_equal(out, np.abs(raw))
    assert out[0] == 0.0
    assert out[2] == 0.5
    offset = raw - raw.min()
    assert not np.array_equal(out, offset)
    source = inspect.getsource(rectify.relu_nonnegative_projection)
    for banned in ("np.abs", "fabs", "envelope", "smooth", "convolve", "clip("):
        assert banned not in source
    assert "np.maximum" in source


def test_rectifier_runs_before_rms_and_nnmf() -> None:
    rng = np.random.RandomState(0)
    nonnegative = np.abs(rng.randn(60, 6)) + 0.05
    signed = nonnegative.copy()
    signed[0, 0] = -0.2
    signed[3, 2] = -0.05
    parent_scale = parent_syn3.positive_scale(signed)
    rectified = rectify.relu_nonnegative_projection(signed)
    relu_scale = parent_syn3.positive_scale(rectified)
    assert not np.allclose(parent_scale, relu_scale)
    fitted = rsyn3.fit_source_nmf(signed)
    assert np.allclose(fitted.scale, relu_scale)
    assert np.all(fitted.dictionary >= 0.0)
    assert np.all(fitted.activations >= 0.0)


def test_same_operator_on_source_and_target_without_query() -> None:
    rng = np.random.RandomState(5)
    source = np.abs(rng.randn(40, 6)) + 0.05
    source[0, 0] = -0.1
    target = np.abs(rng.randn(12, 6)) + 0.02
    target[1, 2] = -0.05
    source_r = rectify.relu_nonnegative_projection(source)
    target_r = rectify.relu_nonnegative_projection(target)
    assert inspect.signature(rectify.relu_nonnegative_projection).parameters.keys() == {"x"}
    assert np.array_equal(source_r, np.maximum(source, 0.0))
    assert np.array_equal(target_r, np.maximum(target, 0.0))
    projected = rsyn3.project_basis(target, rsyn3.fit_source_nmf(source))
    assert np.all(projected >= 0.0)
    assert np.isfinite(projected).all()


def test_rectifier_law_forbids_session_channel_sweep() -> None:
    law = plan.RECTIFIER_LAW
    assert law["per_session_threshold"] == "forbidden"
    assert law["per_channel_threshold"] == "forbidden"
    assert law["sweep"] == "forbidden"
    assert "abs" in law["forbidden_alternatives"]
    assert "offset" in law["forbidden_alternatives"]
    assert "envelope_filter" in law["forbidden_alternatives"]
    assert "smoother" in law["forbidden_alternatives"]
    assert law["query_values_read"] is False
    mass = rectify.mass_report(np.array([[-0.2, 0.1], [0.0, 0.3]], dtype=np.float64))
    assert mass["clipped_count"] == 1
    assert mass["negative_fraction"] > 0.0
    assert mass["rectified_negative_fraction"] == 0.0
    assert mass["maximum_undershoot"] == -0.2


def test_controls_alias_rsyn3_and_zero4_stays_zero() -> None:
    controls = rsyn3.build_controls(
        raw_syn3=np.arange(20, dtype=np.float64).reshape(5, 4),
        normalizer_mean=np.zeros(4),
        normalizer_scale=np.ones(4),
        session_name="ses-20120924",
        seed=42,
        n_units=5,
        target_fit_invoked=False,
    )
    assert np.array_equal(controls["Zero4"], np.zeros((5, 4)))
    assert "rSyn3" in controls
    assert np.array_equal(controls["rSyn3"], controls["Syn3"])
    assert not np.array_equal(controls["RS4"], controls["rSyn3"])


def test_successor_stage0_refuses_parent_result_root(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="parent|sealed|syn3_fcm_v1"):
        rsyn3_receipts.refuse_parent_root(
            ROOT / "tfpd_exploration/results/m1_emg_syn3_fcm_v1"
        )
    rsyn3_receipts.refuse_parent_root(tmp_path / "m1_emg_rsyn3_fcm_v1")


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
    assert payload["phase"] == "m1_emg_rsyn3_fcm_v1"
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
from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import plan
assert "torch" not in sys.modules
assert plan.PHASE == "m1_emg_rsyn3_fcm_v1"
"""
    proc = subprocess.run(
        [PYTHON, "-S", "-c", code],
        cwd=ROOT,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
             "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT)},
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_fixed_last_selection_cannot_read_query() -> None:
    with pytest.raises(rsyn3_training.TrainingError, match="query"):
        rsyn3_training.select_fixed_last(epoch_index=11, query_r2=0.4)
    assert rsyn3_training.select_fixed_last(epoch_index=11) == "epoch_011"


def test_ls4_survives_unequal_trial_lengths() -> None:
    z = np.vstack([
        np.full((3, 3), 1.0),
        np.full((5, 3), 2.0),
        np.full((4, 3), 3.0),
    ])
    rates = np.arange(12 * 2, dtype=np.float64).reshape(12, 2)
    trial_ids = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    remapped, rates_out = rsyn3.derange_trial_association(z, rates, trial_ids, "ses-20120924", 42)
    assert remapped.shape == z.shape
    assert np.array_equal(rates_out, rates)
    assert not np.array_equal(remapped, z)


def test_m2_trial_mean_pca_is_unavailable_not_a_crash() -> None:
    emg_means = np.ones((2, 16), dtype=np.float64)
    rates = np.ones((2, 4), dtype=np.float64)
    report = rsyn3.trial_mean_pca_reliability(emg_means, rates)
    assert report["split"] == "unavailable"
    assert report["weight_flattened_pearson"] is None
    bin_level = np.abs(np.random.RandomState(0).randn(30, 16)) + 0.1
    pca = rsyn3.fit_source_pca(bin_level)
    assert pca.kind == "signed_pca"
    assert pca.dictionary.shape == (3, 16)


def test_verify_parent_fail_rejects_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import stage0 as rstage0

    fake = tmp_path / "stage0"
    fake.mkdir()
    (fake / "decision.json").write_text('{"decision":"PASS"}')
    monkeypatch.setattr(plan, "PARENT_STAGE0_ROOT_RELATIVE", str(fake.relative_to(tmp_path)))
    with pytest.raises(Exception, match="parent|sha|FAIL"):
        rstage0.verify_parent_fail_sealed(tmp_path)
