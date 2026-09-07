from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tfpd_exploration"))

from src.tfsr_b3st4_ddrop_v1.contract import (
    AMIM_RECEIPT_RELATIVE_PATH,
    AMIM_SIDECAR_RELATIVE_PATH,
    HANDOFF_RELATIVE_PATH,
    IMPLEMENTATION_CLOSURE,
    compute_live_closure,
    verify_canonical_evidence,
    verify_live_closure,
)


PHASE_A_SHA = {
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/model.py": "3d4a3a8d4e2a68e9933e84274f2a62308a6eeb5a6978671b53e72af442148fc4",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/__init__.py": "06f3d404f3f73a7bfb69349d71f4adeb24f6306daac99e2a3e7d28a6645f730d",
    "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_v1_stage0.py": "c31f0dfd0190b1a0b09c6fa75ab6c7c3db1045247e161c5af6857bfdb19b92b0",
}


def _copy_evidence_fixture(destination: Path) -> None:
    for relative_path in (HANDOFF_RELATIVE_PATH, AMIM_RECEIPT_RELATIVE_PATH, AMIM_SIDECAR_RELATIVE_PATH):
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative_path, target)
    os.chmod(destination / AMIM_RECEIPT_RELATIVE_PATH, 0o444)
    os.chmod(destination / AMIM_SIDECAR_RELATIVE_PATH, 0o444)


def _copy_closure_fixture(destination: Path) -> None:
    for relative_path in IMPLEMENTATION_CLOSURE:
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative_path, target)


def _preflight_command(*args: str) -> list[str]:
    return [sys.executable, str(ROOT / "tfpd_exploration/scripts/preflight_tfsr_b3st4_ddrop_seed42.py"), *args]


def _preflight_environment(extra_pythonpath: str = "") -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONPATH": ":".join(part for part in (extra_pythonpath, str(ROOT / "tfpd_exploration"), str(ROOT / "tfpd_exploration/src")) if part),
        }
    )
    return environment


def test_canonical_evidence_has_exact_named_semantics_and_sealed_modes():
    evidence = verify_canonical_evidence(ROOT)
    semantics = evidence["amim_receipt"]["semantics"]
    assert semantics["schema"] == "tfpd_subpop_score_v1"
    assert semantics["status"] == "SUBPOP_SCORED"
    assert semantics["matrix_status"] == "matrix_complete"
    assert semantics["matrix_row"] == "JOINT_ABLATION_REQUIRED"
    assert semantics["cells"]["AM"] == {
        "band": "MUCH_LESS_THAN_D", "external_delta": -0.11831592867771784, "external_positive": 2,
        "external_total": 15, "within_delta": -0.08558484415213267, "within_positive": 0, "within_total": 6,
    }
    assert semantics["cells"]["IM"] == {
        "band": "MUCH_LESS_THAN_D", "external_delta": -0.30334921677907306, "external_positive": 0,
        "external_total": 15, "within_delta": -0.07783080140749614, "within_positive": 0, "within_total": 6,
    }
    assert semantics["D_governing_bars"] == {"external": 0.4179362749059995, "within": 0.5696851710478464}


@pytest.mark.parametrize("kind", ("receipt_tamper", "sidecar_tamper", "mode", "symlink"))
def test_canonical_evidence_rejects_tamper_sidecar_mode_and_symlink(tmp_path: Path, kind: str):
    _copy_evidence_fixture(tmp_path)
    receipt = tmp_path / AMIM_RECEIPT_RELATIVE_PATH
    sidecar = tmp_path / AMIM_SIDECAR_RELATIVE_PATH
    if kind == "receipt_tamper":
        os.chmod(receipt, 0o644)
        receipt.write_bytes(receipt.read_bytes() + b" ")
        os.chmod(receipt, 0o444)
    elif kind == "sidecar_tamper":
        os.chmod(sidecar, 0o644)
        sidecar.write_text("0" * 64 + "  subpop_score_receipt.json\n", encoding="ascii")
        os.chmod(sidecar, 0o444)
    elif kind == "mode":
        os.chmod(sidecar, 0o644)
    else:
        receipt.unlink()
        receipt.symlink_to(ROOT / AMIM_RECEIPT_RELATIVE_PATH)
    with pytest.raises(RuntimeError):
        verify_canonical_evidence(tmp_path)


def test_live_closure_freezes_phase_a_and_rejects_drift_missing_extra_and_symlink(tmp_path: Path):
    live = compute_live_closure(ROOT)
    assert set(live["sha256_by_path"]) == set(IMPLEMENTATION_CLOSURE)
    assert {path: live["sha256_by_path"][path] for path in PHASE_A_SHA} == PHASE_A_SHA
    assert verify_live_closure(ROOT, live["sha256_by_path"], live["closure_sha256"]) == live
    _copy_closure_fixture(tmp_path)
    fixture = compute_live_closure(tmp_path)
    assert verify_live_closure(tmp_path, fixture["sha256_by_path"], fixture["closure_sha256"]) == fixture
    drift_path = tmp_path / IMPLEMENTATION_CLOSURE[2]
    drift_path.write_bytes(drift_path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError):
        verify_live_closure(tmp_path, fixture["sha256_by_path"], fixture["closure_sha256"])
    missing_path = tmp_path / IMPLEMENTATION_CLOSURE[3]
    missing_path.unlink()
    with pytest.raises(RuntimeError):
        compute_live_closure(tmp_path)
    _copy_closure_fixture(tmp_path)
    extra = dict(fixture["sha256_by_path"])
    extra["tfpd_exploration/extra.py"] = "0" * 64
    with pytest.raises(RuntimeError):
        verify_live_closure(tmp_path, extra, fixture["closure_sha256"])
    symlink_path = tmp_path / IMPLEMENTATION_CLOSURE[0]
    symlink_path.unlink()
    symlink_path.symlink_to(tmp_path / IMPLEMENTATION_CLOSURE[1])
    with pytest.raises(RuntimeError):
        compute_live_closure(tmp_path)


def test_zero_arg_cli_emits_full_no_data_plan_and_creates_no_result_root():
    result_root = ROOT / "tfpd_exploration/results/tfsr_b3st4_ddrop_v1"
    assert not result_root.exists()
    completed = subprocess.run(
        _preflight_command(), cwd=ROOT, env=_preflight_environment(), check=False, text=True, capture_output=True
    )
    assert completed.returncode == 0, completed.stderr
    assert not result_root.exists()
    plan = json.loads(completed.stdout)
    assert plan["status"] == "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_LAUNCH"
    assert plan["frozen"]["capture_diagnostics"] is False
    assert plan["frozen"]["dimensions"]["attention_dropout"] == 0.0
    assert plan["frozen"]["training"] == {"epochs": 48, "schedule": "warmup_cosine", "swa": "final_4_epochs", "seed": 42, "behavior_scaling_factor": None}
    assert {path: plan["live_implementation_closure"]["sha256_by_path"][path] for path in PHASE_A_SHA} == PHASE_A_SHA
    resource = plan["cpu_no_data_resource_audit"]
    assert resource["gpu"] is False and resource["torch_cuda_called"] is False
    assert resource["synthetic_shape"]["units"] == 128
    assert resource["trainable_params"] <= 3_600_000
    assert resource["analytic_mac_estimate_n128"] > 0
    assert resource["persistent_state_bytes"] == 1024
    assert resource["analytic_one_token_tensor_bytes"] == 50 * 128 * 256 * 4
    assert resource["cpu_latency_protocol"] == {"warmup": 1, "repeats": 3, "statistic": "median", "mode_restored": True}
    assert resource["cpu_forward_latency_median_ms"] >= 0
    assert resource["analytic_gpu_envelope_fp32_B1_N128"]["training_peak"] == "NOT_MEASURED"
    assert resource["analytic_gpu_envelope_fp32_B1_N128"]["required_full_training_launch_blocker"] == (
        "measure_training_peak_during_authorized_source_only_smoke_before_48_epoch_training_launch"
    )


def test_forbidden_cli_argument_exits_before_torch_or_data_side_effect(tmp_path: Path):
    fake_torch = tmp_path / "torch.py"
    fake_torch.write_text("raise RuntimeError('TORCH_WAS_IMPORTED')\n", encoding="utf-8")
    completed = subprocess.run(
        _preflight_command("--forbidden"), cwd=ROOT, env=_preflight_environment(str(tmp_path)), check=False, text=True, capture_output=True
    )
    assert completed.returncode != 0
    assert "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_LAUNCH" in completed.stderr
    assert "TORCH_WAS_IMPORTED" not in completed.stderr + completed.stdout
    assert not (ROOT / "tfpd_exploration/results/tfsr_b3st4_ddrop_v1").exists()
