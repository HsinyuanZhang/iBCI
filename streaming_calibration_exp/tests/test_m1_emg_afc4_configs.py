"""Static contract for the four isolated, preflight-only M1 AFC4 configs."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "full": "m1_afc4_emg_full_source_loso.yaml",
    "zero4": "m1_afc4_emg_zero4_source_loso.yaml",
    "rs4": "m1_afc4_emg_rs4_source_loso.yaml",
    "b4": "m1_afc4_emg_b4_source_loso.yaml",
}


def test_all_four_configs_are_global_and_preflight_only() -> None:
    for arm, filename in EXPERIMENTS.items():
        text = (ROOT / "configs" / "experiment" / filename).read_text(encoding="utf-8")
        assert "# @package _global_" in text
        assert f"afc4_arm: {arm}" in text
        if arm == "full":
            assert "train: false" in text
            assert "test: true" in text
            assert "monitor: train/loss" in text
            assert "max_epochs: 12" in text
        else:
            assert "- m1_afc4_emg_full_source_loso" in text


def test_full_config_uses_source_task_only_objective() -> None:
    text = (ROOT / "configs" / "experiment" / EXPERIMENTS["full"]).read_text(encoding="utf-8")
    assert "freeze_decoder: true" in text
    assert "teacher_ckpt_path: M1_AFC4_SOURCE_LOSO_CHECKPOINT_REQUIRED" in text
    assert "teacher_ckpt_path: ${paths.m1_teacher_ckpt_path}" not in text
    assert "loss_mode: task_only" in text
    assert "lambda_y: 0.0" in text
    assert "lambda_E: 0.0" in text
