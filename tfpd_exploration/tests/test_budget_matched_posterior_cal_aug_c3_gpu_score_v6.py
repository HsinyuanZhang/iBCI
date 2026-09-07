from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT / "tfpd_exploration",
    ROOT / "tfpd_exploration/src",
    ROOT / "sua_exploration",
    ROOT / "streaming_calibration_exp/src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from budget_matched_posterior_cal_aug_c3_v1 import score_gpu_v4 as v4
from budget_matched_posterior_cal_aug_c3_v1 import score_gpu_v6 as v6


def test_failed_v5_graph_is_exact() -> None:
    binding = v6.validate_failed_v5(ROOT)
    assert binding["attempt_sha256"] == v6.FAILED_V5_ATTEMPT_SHA256
    assert binding["failure_sha256"] == v6.FAILED_V5_FAILURE_SHA256
    assert binding["exact_leaf_count"] == 4


def test_drift_policy_accepts_review_without_restart() -> None:
    policy = v6.drift_policy()
    assert policy["review_drift"] == "ACCEPTED_NON_NUMERIC_DRIFT"
    assert policy["numerical_acceptance_affected_by_review_drift"] is False
    assert policy["restart_required_for_review_drift"] is False
    assert "tests" in policy["review_only_path_classes"]


def test_execution_and_review_closures_are_disjoint() -> None:
    execution = v6.execution_closure(ROOT)
    review = v6.review_closure(ROOT)
    assert set(execution["files"]).isdisjoint(review["files"])
    assert v6.WORK_ORDER_RELATIVE not in execution["files"]


def test_parity_can_be_measured_without_raising() -> None:
    torch = pytest.importorskip("torch")

    class Inputs:
        surface = "within"
        session = "synthetic"
        last_targets = torch.zeros((2, 2), dtype=torch.float32).numpy()
        last_valid_mask = torch.ones(2, dtype=torch.bool).numpy()

    class Metric:
        @staticmethod
        def session_r2(prediction, target):
            return float(-(prediction - target).square().mean().item())

    cpu = torch.zeros((2, 50, 2), dtype=torch.float32)
    gpu = cpu.clone()
    gpu[:, 49, :] = 1.0e-4
    evidence = v4._parity_evidence(
        cpu, gpu, gpu.clone(), inputs=Inputs(), matched_scorer=Metric(), enforce=False
    )
    assert evidence["passed"] is False
    assert evidence["gpu_repeated_bitwise_equal"] is True
    assert evidence["max_abs_prediction_difference"] == pytest.approx(1.0e-4)
    with pytest.raises(v4.GPUScoreV4Error, match="parity smoke failed"):
        v4._parity_evidence(
            cpu, gpu, gpu.clone(), inputs=Inputs(), matched_scorer=Metric(), enforce=True
        )


def test_static_v6_dry_cli_is_inert() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_c3_gpu_parity_v6.py"
    completed = subprocess.run(
        [sys.executable, "-S", str(script), "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        },
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "DRY_NO_DATA_NO_MODEL_NO_CUDA_NO_WRITE"
    assert payload["full_matrix_authorized"] is False
    assert payload["drift_policy"]["restart_required_for_review_drift"] is False

