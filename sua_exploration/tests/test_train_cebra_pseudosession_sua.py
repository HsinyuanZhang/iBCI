from __future__ import annotations

from pathlib import Path

import pytest

from scripts import train_cebra_pseudosession_sua as subject


def _argv(tmp_path: Path, *, arm: str = "t4", seed: int = 42) -> list[str]:
    root = subject.SUA_ROOT
    return [
        "--teacher_ckpt", str(root / "checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"),
        "--variant", "B3S", "--task", "CO", "--split_counts", "27,6,6",
        "--signal_view", "sua", "--side_features", arm, "--side_feature_pool_size", "30",
        "--calibration_n_trials", "30", "--max_units_exclusive", "100", "--batch_size", "32",
        "--num_workers", "4",
        "--lr", "0.0001", "--max_epochs", "12", "--loss_mode", "task_only",
        "--identity_mode", "calibrated", "--seed", str(seed), "--chronological_calibration",
        "--no_early_stopping", "--checkpoint_every_epoch", "--data_dir",
        str(root / "data/dandi_000688/sub-C"), "--train_val_manifest",
        str(root / "configs/subc_co_27_6_strict_train_val_manifest.json"),
        "--cache_dir", str(root / "cache/dandi688_subc_co_v1"),
        "--out_name", f"{subject.SCREEN_ID}_mix_{arm}_s{seed}",
        "--require_gpu", "--disable_progress_bar",
    ]


def test_exact_stage_p_argv_is_accepted(tmp_path: Path) -> None:
    assert subject.validate_argv(_argv(tmp_path)) == ("t4", 42)
    assert subject.validate_argv(_argv(tmp_path, arm="z4")) == ("z4", 42)


@pytest.mark.parametrize("flag,value", [("--max_epochs", "13"), ("--calibration_n_trials", "29")])
def test_scientific_drift_is_rejected(tmp_path: Path, flag: str, value: str) -> None:
    argv = _argv(tmp_path)
    argv[argv.index(flag) + 1] = value
    with pytest.raises(subject.TrainerContractError):
        subject.validate_argv(argv)


def test_forbidden_limit_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(subject.TrainerContractError):
        subject.validate_argv(_argv(tmp_path) + ["--limit_train_batches", "1"])


def test_constructibility_preflight_is_live() -> None:
    payload = subject.load_preflight()
    assert payload["formal_subc_test_nwb_opened"] is False
    assert payload["pseudo_session_audit"]["schedule_sha256"] == subject.EXPECTED_SEED42_SCHEDULE_SHA256
