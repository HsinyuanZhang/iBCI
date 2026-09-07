"""CPU contracts for cross-dataset functional calibration v1. No live NWB/GPU."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import contracts
from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import plan


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "tfpd_exploration/scripts/run_cross_dataset_functional_calibration_v1.py"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
_CLI_ENV = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": str(ROOT),
}


def test_plan_literals_and_owned_paths() -> None:
    assert plan.PHASE == "cross_dataset_functional_calibration_v1"
    assert plan.STAGE0_TIMESTAMP == "20260905_113700"
    assert plan.STAGE0_ROOT_RELATIVE.endswith("20260905_113700")
    assert plan.E1_SEED == 20260905
    assert plan.E1_N_FOCAL == 8
    assert plan.E1_N_MASKS == 16
    assert plan.E1_RETAIN_FRAC == 0.75
    assert plan.E2_LAG_BINS == 5
    assert plan.E2_LAG_MS == 100
    assert plan.M1_RIDGE_LAMBDA == 1.0
    assert plan.H1_RIDGE_LAMBDA == 100.0
    assert "tfpd_exploration/src/m2_dual_track_v1/" in plan.SEALED_FOREIGN_ROOTS
    assert str(CLI.relative_to(ROOT)) in plan.OWNED_PATHS
    assert plan.focal_channel_indices(176) == (0, 22, 44, 66, 88, 110, 132, 154)


def test_workorder_hash_matches_disk() -> None:
    path = ROOT / plan.WORKORDER_RELATIVE
    assert path.is_file()
    assert plan.sha256_bytes(path.read_bytes()) == plan.WORKORDER_SHA256
    plan.verify_bound_documents(ROOT)


def test_named_estimators_are_closed() -> None:
    contracts.require_named_estimator("h1_deployment_m3")
    contracts.require_named_estimator("m1_rsyn3_unit_ridge")
    contracts.require_named_estimator("m1_row_normalized_nnmf_nnls_v1")
    try:
        contracts.require_named_estimator("invented_haufe")
    except plan.PlanError:
        pass
    else:
        raise AssertionError("undocumented estimator must fail")
    assert contracts.UNRESOLVED_P_OPERATORS["f_eta"]["status"] == "NAMED_REVISION"
    assert contracts.UNRESOLVED_P_OPERATORS["parent_bytes"]["status"] == "NAMED_REVISION"
    assert contracts.UNRESOLVED_P_OPERATORS["parent_bytes"]["gpu_eligible"] is False


def test_ridge_objectives_are_distinct() -> None:
    m1 = plan.inherited_ridge_objective()
    h1 = plan.h1_ridge_objective()
    assert m1["normalized_by_n"] is True
    assert " / n" in str(m1["gram"])
    assert h1["normalized_by_n"] is False
    assert m1["lambda"] == 1.0
    assert h1["lambda"] == 100.0
    assert m1["sample_count_changes_lambda"] is False


def test_dry_cli_does_not_open_gpu_or_nwb() -> None:
    payload = plan.dry_cli_payload()
    assert payload["opens_nwb_or_checkpoint"] is False
    assert payload["imports_torch"] is False
    assert payload["initializes_cuda"] is False
    assert payload["public_gpu_capability"] is False
    assert payload["gpu_work_started"] is False
    assert payload["p_stream"]["f_eta_shape"] == [3, 16]
    assert payload["p_stream"]["gpu_eligible"] is False
    completed = subprocess.run(
        [PYTHON, str(CLI), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env=_CLI_ENV,
        cwd=str(ROOT),
    )
    body = json.loads(completed.stdout)
    assert body["phase"] == plan.PHASE
    assert body["gpu_work_started"] is False


def test_execute_gpu_is_rejected() -> None:
    completed = subprocess.run(
        [PYTHON, str(CLI), "--execute-gpu"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
        cwd=str(ROOT),
    )
    assert completed.returncode != 0
    assert "CPU-only" in completed.stderr
