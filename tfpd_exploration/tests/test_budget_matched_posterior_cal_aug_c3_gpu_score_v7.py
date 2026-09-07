from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

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
from budget_matched_posterior_cal_aug_c3_v1 import score_gpu_v7 as v7


def test_failed_v6_graph_is_exact() -> None:
    binding = v7.validate_failed_v6(ROOT)
    assert binding["attempt_sha256"] == v7.FAILED_V6_ATTEMPT_SHA256
    assert binding["failure_sha256"] == v7.FAILED_V6_FAILURE_SHA256
    assert binding["exact_leaf_count"] == 4
    assert binding["disposition"].startswith("FAILED_POST_COMPARISON")


def test_execution_review_split_and_policy() -> None:
    execution = v7.execution_closure(ROOT)
    review = v7.review_closure(ROOT)
    assert set(execution["files"]).isdisjoint(review["files"])
    assert v7.WORK_ORDER_RELATIVE not in execution["files"]
    assert v7.v6.drift_policy()["restart_required_for_review_drift"] is False


def test_post_attempt_runtime_does_not_reapply_torch_absence(monkeypatch, tmp_path) -> None:
    subc = tmp_path / "sub-C"
    subm = tmp_path / "sub-M"
    subc.mkdir()
    subm.mkdir()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("SUBC_DATA_ROOT", str(subc))
    monkeypatch.setenv("SUBM_DATA_ROOT", str(subm))
    monkeypatch.setattr(v4, "_module_origin", lambda name: {"src": "src.py"}.get(name, "model.py"))
    gpu = {"uuid": "GPU-test"}
    monkeypatch.setattr(v4, "_nvidia_smi_profile", lambda: gpu)
    monkeypatch.setitem(sys.modules, "torch", object())
    launch = {
        "cuda_visible_devices": "1",
        "cuda_device_order": "PCI_BUS_ID",
        "cublas_workspace_config": ":4096:8",
        "subc_data_root": str(subc),
        "subm_data_root": str(subm),
        "src_origin": "src.py",
        "streaming_encoder_origin": "model.py",
        "gpu": gpu,
        "torch_imported": False,
    }
    result = v7.post_attempt_runtime_revalidation(ROOT, launch)
    assert result["pre_attempt_only_torch_absence_not_reapplied"] is True
    assert result["torch_imported_after_execution"] is True


def test_v7_source_publishes_comparison_before_final_validation() -> None:
    source = (ROOT / "tfpd_exploration/src/budget_matched_posterior_cal_aug_c3_v1/score_gpu_v7.py").read_text()
    publish = source.index('_publish_pair(result_root, "comparison.json"')
    final = source.index('stage = "final_revalidation"')
    assert publish < final
    assert '"comparison_sha256": comparison_sha' in source
    assert '"numerical_comparison_preserved": comparison_sha is not None' in source


def test_static_v7_dry_cli_is_inert() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_c3_gpu_parity_v7.py"
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
    assert payload["comparison_published_before_final_revalidation"] is True
    assert payload["full_matrix_authorized"] is False

