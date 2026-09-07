"""Static contracts for the H-U launcher and evaluator gates."""
from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from scripts.h1_carrierid_hu_launcher import (
    RECOMMENDED_FRESH_OUTPUT_ROOT,
    isolated_environment_command,
    prepare_or_launch,
    run_tmux_package_identity_preflight,
)


ROOT = Path(__file__).resolve().parents[1]


def test_launcher_prepare_does_not_launch_and_records_one_cell():
    plan = prepare_or_launch(
        output_root=ROOT / "pilot_artifacts/h1_carrierid_hu/gpu_runs/h32_fold0_hu_v1",
        python_bin=Path("/home/xinyuan/miniconda3/envs/spint/bin/python"),
        execute=False,
    )
    assert plan["launched"] is False
    assert plan["launch_authorized"] is False
    assert plan["cells"] == 1
    assert plan["mode"] == "prepare_print_only"
    assert "experiment=h1_carrierid_hu" in plan["command"]
    assert plan["fixed_contract"]["fold_date"] == "19250101"
    assert plan["fixed_contract"]["seed"] == 42
    assert plan["fixed_contract"]["epochs"] == 50
    assert plan["fixed_contract"]["primary_comparators"] == ["H-C", "H-C0"]
    assert plan["effective_environment"] == {
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": "",
        "CUDA_VISIBLE_DEVICES": "0",
        "python_bin": "/home/xinyuan/miniconda3/envs/spint/bin/python",
        "python_executable_resolved": "/home/xinyuan/miniconda3/envs/spint/bin/python3.10",
        "expected_torch_prefix": "/home/xinyuan/miniconda3/envs/spint",
    }
    assert plan["recommended_fresh_output_root"] == str(RECOMMENDED_FRESH_OUTPUT_ROOT)
    assert plan["tmux_effective_command"].startswith(
        "env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES=0 "
    )


def test_isolated_command_places_environment_inside_tmux_command():
    command = isolated_environment_command(
        command=["/home/xinyuan/miniconda3/envs/spint/bin/python", "src/train.py"], gpu="3"
    )
    assert command == [
        "env",
        "PYTHONNOUSERSITE=1",
        "PYTHONPATH=",
        "CUDA_VISIBLE_DEVICES=3",
        "/home/xinyuan/miniconda3/envs/spint/bin/python",
        "src/train.py",
    ]


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux unavailable")
def test_real_tmux_child_uses_conda_torch_and_never_opens_data_or_trains(tmp_path):
    identity = run_tmux_package_identity_preflight(
        python_bin=Path("/home/xinyuan/miniconda3/envs/spint/bin/python"),
        gpu="0",
        work_dir=tmp_path,
    )
    assert identity["sys_executable"] == "/home/xinyuan/miniconda3/envs/spint/bin/python"
    assert identity["torch_file"].startswith(
        "/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages/torch/"
    )
    assert identity["torch_version"] == "2.5.1.post303"
    assert identity["enable_user_site"] is False
    assert identity["cuda_available"] is True
    assert identity["cuda_device_count"] == 1
    assert identity["cuda_device_name"] == "NVIDIA GeForce RTX 3090"
    assert identity["probe_only_no_training_or_data"] is True


def test_launcher_execute_is_gated_closed_without_both_tokens():
    with pytest.raises(RuntimeError, match="gated closed"):
        prepare_or_launch(
            output_root=ROOT / "pilot_artifacts/h1_carrierid_hu/gpu_runs/unused",
            python_bin=Path("/home/xinyuan/miniconda3/envs/spint/bin/python"),
            execute=True,
            i_have_authorization=False,
        )
    with pytest.raises(RuntimeError, match="gated closed"):
        prepare_or_launch(
            output_root=ROOT / "pilot_artifacts/h1_carrierid_hu/gpu_runs/unused",
            python_bin=Path("/home/xinyuan/miniconda3/envs/spint/bin/python"),
            execute=True,
            i_have_authorization=True,
        )


def test_evaluator_refuses_without_authorization_flag():
    source = (ROOT / "scripts/h1_carrierid_hu_terminal_evaluate.py").read_text(encoding="utf-8")
    assert "--i-have-authorization" in source
    assert "must not be executed in the implementation pass" in source
    assert "load_target_records" in source
    assert "QUERY_SHA = \"665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da\"" in source


def test_existing_hc_experiment_still_points_at_the_frozen_producer():
    text = (ROOT / "configs/experiment/h1_carrierid.yaml").read_text(encoding="utf-8")
    assert "override /data: falcon_h1_m4_eb_normalized_v2" in text
    assert "h1_carrierid_hu" not in text
    data = (ROOT / "configs/data/falcon_h1_m4_eb_normalized_v2.yaml").read_text(encoding="utf-8")
    assert "src.data.h1_m4_eb_normalized_v2.H1M4EBNormalizedV2DataModule" in data
