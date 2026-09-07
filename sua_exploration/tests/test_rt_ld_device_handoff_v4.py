"""Independent v4 handoff checks.  These tests never invoke GPU training."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "sua_exploration/scripts/rt_ld_device_handoff_v4.py"
PLAN = ROOT / "sua_exploration/results/rt_ld_device_handoff_v4/RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v4.json"
SELECTION_TEST = ROOT / "streaming_calibration_exp/tests/test_rt_ld_selection_receipt_contract.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module():
    spec = importlib.util.spec_from_file_location("rt_ld_device_handoff_v4", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v4_plan_is_self_bound_to_v4_runner_and_tests() -> None:
    plan = json.loads(PLAN.read_text())
    assert plan["schema"] == "rt_ld_device_handoff_supplemental_plan_v4"
    assert plan["status"] == "REVIEW_REQUIRED_NOT_ARMED"
    assert plan["runner_path"] == "sua_exploration/scripts/rt_ld_device_handoff_v4.py"
    assert plan["runner_sha256"] == _sha(RUNNER)
    assert plan["test_path"] == "sua_exploration/tests/test_rt_ld_device_handoff_v4.py"
    assert plan["test_sha256"] == _sha(Path(__file__))
    assert plan["selection_contract_test_path"] == "streaming_calibration_exp/tests/test_rt_ld_selection_receipt_contract.py"
    assert plan["selection_contract_test_sha256"] == _sha(SELECTION_TEST)
    assert plan["gpu_launched"] is False and plan["formal_heldout_opened"] is False


def test_relative_run_root_is_resolved_before_directory_actions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.chdir(tmp_path)
    root = module._absolute_run_root(Path("review_runs/rt_ld"))
    assert root.is_absolute() and root == (tmp_path / "review_runs/rt_ld").resolve()


def test_release_probes_reject_replacement_runner_and_accept_two_clean_samples() -> None:
    module = _module()
    clean = ("exited", False, [])
    assert module._eligible_from_probes(clean, clean)
    assert not module._eligible_from_probes(("exited", True, []), clean)
    assert not module._eligible_from_probes(("active", False, []), clean)
    assert not module._eligible_from_probes(("exited", False, ["123"]), clean)
    assert not module._eligible_from_probes(("exited", False, None), clean)


def _synthetic_fit_tree(tmp_path: Path, experiment: str, *, wrong_arm: bool = False) -> Path:
    module = _module()
    spec = module.ARM_SPECS[experiment]
    arm_dir = tmp_path / spec["directory"]
    checkpoint = arm_dir / "checkpoints/best_ckpt/epoch_001.ckpt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"epoch": 1, "global_step": 2}, checkpoint)
    callback = {
        "_target_": "src.callbacks.rt_nested_selection_receipt.RtNestedSelectionReceipt",
        "output_path": str(arm_dir / "rt_nested_selection_receipt.json"),
        "split_manifest_path": str(arm_dir / "split_manifest.json"),
        "config_path": str(arm_dir / ".hydra/config.yaml"),
        "monitor": "val_heldin/r2_mean",
    }
    config = {
        "run_id": "wrong_run" if wrong_arm else spec["run_id"],
        "seed": 42,
        "model": {"rt_ld_arm": spec["model_rt_ld_arm"]},
        "data": {"rt_ld_gain_source": spec["data_rt_ld_gain_source"], "outer_loso_fold": 0},
        "callbacks": {"rt_nested_selection_receipt": callback},
    }
    (arm_dir / ".hydra").mkdir(parents=True)
    (arm_dir / ".hydra/config.yaml").write_text(yaml.safe_dump(config))
    gain = "aligned_full_afc4" if spec["data_rt_ld_gain_source"] == "full" else "strong_xls_v2"
    split = {
        "validation_protocol": "nested_loso",
        "requested_side_feature_group": "afc4_vel",
        "rt_ld": {"identity_carrier": "aligned_full_afc4", "gain_carrier": gain, "identity_never_receives_xls_v2": True},
        "nested_selection": {"checkpoint_metric": "val_heldin/r2_mean", "inner_validation_only_for_checkpoint_selection": True},
    }
    (arm_dir / "split_manifest.json").write_text(json.dumps(split))
    receipt = {
        "schema": "rt_clean_nested_loso_selection_receipt_v1",
        "status": "PASS_FIT_INNER_SELECTION_ONLY",
        "run_id": "wrong_run" if wrong_arm else spec["run_id"],
        "arm": "afc4_vel",
        "outer_loso_fold": 0,
        "seed": 42,
        "selected_by_metric": "val_heldin/r2_mean",
        "selected_metric_scope": "inner_validation_session_only",
        "best_model_path": str(checkpoint.resolve()),
        "run_dir": str(arm_dir.resolve()),
    }
    (arm_dir / "rt_nested_selection_receipt.json").write_text(json.dumps(receipt))
    return arm_dir


def test_synthetic_fit_artifacts_reject_wrong_arm_and_accept_exact_absolute_checkpoint(tmp_path: Path) -> None:
    module = _module()
    experiment = "rt_ld_g_xls_m24_fold0_seed42"
    wrong = _synthetic_fit_tree(tmp_path / "wrong", experiment, wrong_arm=True)
    with pytest.raises(ValueError, match="mapping mismatch|selection receipt mismatch"):
        module._validate_fit_artifacts(wrong, experiment)
    right = _synthetic_fit_tree(tmp_path / "right", experiment)
    assert module._validate_fit_artifacts(right, experiment) == (right / "checkpoints/best_ckpt/epoch_001.ckpt").resolve()


def test_real_hydra_compose_all_three_arms() -> None:
    # The v4 routine invokes `src/train.py --cfg job --resolve` for each arm,
    # which exercises actual Hydra defaults/interpolation without training.
    _module().validate_all_composed()
