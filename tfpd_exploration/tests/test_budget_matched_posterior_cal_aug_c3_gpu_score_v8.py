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
from budget_matched_posterior_cal_aug_c3_v1 import score_gpu_v8 as v8


def test_v7_terminal_is_exact_and_supports_engineering_gate() -> None:
    binding = v8.validate_v7_parity_terminal(ROOT)
    assert binding["exact_leaf_count"] == 6
    assert binding["repeated_gpu_bitwise_equal"] is True
    assert binding["max_abs_prediction_difference"] < v8.ENGINEERING_MAX_ABS_TOLERANCE
    assert binding["absolute_r2_difference"] < v8.ENGINEERING_R2_TOLERANCE


def test_engineering_tolerance_does_not_change_science() -> None:
    assert v8.ENGINEERING_MAX_ABS_TOLERANCE == 1e-5
    assert v8.ENGINEERING_R2_TOLERANCE == 1e-6
    assert v8.ENGINEERING_R2_TOLERANCE <= 1e-4 * 0.01
    assert v8.dry_plan()["engineering_parity_tolerances"]["scientific_gates_unchanged"] is True


def test_parity_accepts_v7_scale_under_v8_tolerance() -> None:
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
    gpu[:, 49, :] = 6.4e-6
    evidence = v4._parity_evidence(
        cpu,
        gpu,
        gpu.clone(),
        inputs=Inputs(),
        matched_scorer=Metric(),
        enforce=True,
        max_abs_tolerance=v8.ENGINEERING_MAX_ABS_TOLERANCE,
        r2_tolerance=v8.ENGINEERING_R2_TOLERANCE,
    )
    assert evidence["passed"] is True
    assert evidence["max_abs_tolerance"] == v8.ENGINEERING_MAX_ABS_TOLERANCE


def test_execution_and_review_closures_are_disjoint() -> None:
    execution = v8.execution_closure(ROOT)
    review = v8.review_closure(ROOT)
    assert set(execution["files"]).isdisjoint(review["files"])
    assert v8.WORK_ORDER_RELATIVE not in execution["files"]


def test_static_v8_dry_cli() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_c3_gpu_score_v8.py"
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
    assert payload["expected_row_count"] == 252
    assert payload["engineering_parity_tolerances"]["scientific_gates_unchanged"] is True

