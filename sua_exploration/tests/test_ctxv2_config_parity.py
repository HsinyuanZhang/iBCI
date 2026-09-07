"""Focused CPU tests for CTXV2 Stage A experiment-config parity."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
SPINT = ROOT / "SPINT-main"
PARITY_SCRIPT = SPINT / "scripts/ctxv2_parity_check.py"
LAUNCH_SCRIPT = SPINT / "scripts/ctxv2_stage_a_launch.sh"
CONFIG_DIR = SPINT / "configs/experiment"

NEW_CONFIGS = (
    "h1_ctxv2_context_full_19250108",
    "h1_ctxv2_hse5_full_19250108",
    "h1_ctxv2_zero5_19250108",
)


def load_parity_module():
    spec = importlib.util.spec_from_file_location("ctxv2_parity_check", PARITY_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", NEW_CONFIGS)
def test_stage_a_configs_exist_and_parse(name: str) -> None:
    path = CONFIG_DIR / f"{name}.yaml"
    assert path.is_file()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["pilot"]["fold_date"] == "19250108"
    assert OmegaConf.load(path)


def test_illegal_arm_to_arm_difference_fails_checker() -> None:
    module = load_parity_module()
    left = {"seed": 42, "trainer.max_epochs": 50, "pilot.batch_size": 32}
    right = {"seed": 42, "trainer.max_epochs": 50, "pilot.batch_size": 16}
    result = module.diff_paths(left, right, permitted=module.ARM_TO_ARM_PERMITTED_PATHS)
    assert result["illegal"]
    assert "pilot.batch_size" in result["illegal"]


def test_illegal_date_to_date_difference_fails_checker() -> None:
    module = load_parity_module()
    left = {"task_name": "new", "seed": 42}
    right = {"task_name": "sealed", "seed": 43}
    result = module.diff_paths(left, right, permitted=module.DATE_TO_DATE_PERMITTED_PATHS)
    assert result["illegal"]
    assert "seed" in result["illegal"]


def test_zero_carrier_semantics_on_composed_configs() -> None:
    module = load_parity_module()
    composed = {
        arm: module.compose_experiment(spec["new"])
        for arm, spec in module.STAGE_A_ARMS.items()
    }
    payload = module.check_zero_carrier_semantics(composed)
    assert payload["pass"] is True
    assert composed["zero5"].pilot.zero_carrier is True
    assert composed["context_full"].pilot.zero_carrier is False
    assert composed["hse5_full"].pilot.zero_carrier is False


def test_launch_script_dry_run_prints_three_gpu_assignments_without_launching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/tmux")
    completed = subprocess.run(
        ["bash", str(LAUNCH_SCRIPT), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        cwd=SPINT,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0
    assert "CUDA_VISIBLE_DEVICES=0" in output
    assert "CUDA_VISIBLE_DEVICES=1" in output
    assert output.count("CUDA_VISIBLE_DEVICES=0") >= 2
    assert "ctxv2_ctx" in output
    assert "ctxv2_hse5" in output
    assert "ctxv2_zero" in output
    assert "--dry-run: no tmux sessions started." in output
    assert "tmux new-session" not in output
