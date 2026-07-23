"""Tests for revised Gate2 matrix helpers and presets."""
from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.metrics.gate2_matrix import (
    ArtifactValidationError,
    build_matrix_row,
    check_r1_readiness,
    choose_winning_loss,
    decision_status_for_delta,
    evaluate_r1,
    extract_best_epoch,
    find_protocol_control_r2,
    load_matrix,
    loss_overrides,
    refresh_d512_deltas,
    validate_artifact_complete,
    write_matrix,
)
from src.metrics.run_artifacts import make_run_id as artifacts_make_run_id


ANCHOR_DIR = Path(__file__).resolve().parents[1] / "outputs/streaming_calibration/b3_d64_anchor_s42_20260711_011020"


def test_extract_best_epoch_from_checkpoint_manifest():
    manifest = {"source_checkpoint_path": "/tmp/checkpoints/best_ckpt/epoch_036.ckpt"}
    assert extract_best_epoch(manifest) == 36


def test_make_run_id_includes_fold_suffix():
    cfg = OmegaConf.create(
        {
            "run_id": "b3_d64_task_only",
            "seed": 42,
            "model": {"variant": "B3"},
            "data": {"loso_fold": 0},
        }
    )
    run_id = artifacts_make_run_id(cfg)
    assert run_id.startswith("b3_d64_task_only_f0_s42_")


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (-0.005, "green"),
        (-0.015, "amber"),
        (-0.04, "red"),
    ],
)
def test_decision_status_for_delta(delta: float, expected: str):
    assert decision_status_for_delta(delta, green=-0.01, amber=-0.03) == expected


@pytest.mark.skipif(not ANCHOR_DIR.exists(), reason="anchor artifact not present")
def test_build_matrix_row_from_anchor():
    row = build_matrix_row(ANCHOR_DIR, comparison_role="loss_ablation_reference")
    assert row["variant"] == "B3"
    assert row["width_or_RK"] == "D64"
    assert row["loss_mode"] == "task_plus_y_plus_E"
    assert row["validation_session"] == "ses-2020-10-19-Run1"
    assert float(row["R2"]) == pytest.approx(0.63024879, abs=1e-5)
    assert float(row["delta_fixed_B0"]) == pytest.approx(-0.04142343, abs=1e-5)
    assert row["preprocessing"] == "cubic_T100"


def test_validate_artifact_complete_rejects_empty_directory(tmp_path: Path):
    empty = tmp_path / "empty_run"
    empty.mkdir()
    with pytest.raises(ArtifactValidationError):
        validate_artifact_complete(empty)


def test_choose_winning_loss_prefers_higher_r2():
    rows = [
        {"loss_mode": "task_only", "R2": "0.62000000"},
        {"loss_mode": "task_plus_y", "R2": "0.63000000"},
        {"loss_mode": "task_plus_y_plus_E", "R2": "0.62500000"},
    ]
    winner, state, notes = choose_winning_loss(rows)
    assert winner == "task_plus_y"
    assert state == "winner_selected"
    assert notes


def test_choose_winning_loss_tie_returns_no_winner():
    rows = [
        {"loss_mode": "task_only", "R2": "0.63000000"},
        {"loss_mode": "task_plus_y", "R2": "0.63020000"},
        {"loss_mode": "task_plus_y_plus_E", "R2": "0.63010000"},
    ]
    winner, state, notes = choose_winning_loss(rows)
    assert winner is None
    assert state == "tie_requires_seed43"
    assert any("seed=43" in note for note in notes)


def test_check_r1_readiness_requires_all_rows(tmp_path: Path):
    matrix = tmp_path / "gate2_revised_matrix.csv"
    write_matrix(
        matrix,
        [
            {
                "run_id": "anchor",
                "comparison_role": "loss_ablation_reference",
                "fold_id": 0,
                "seed": 42,
                "R2": "0.63000000",
                "delta_fixed_B0": "-0.04000000",
                "loss_mode": "task_plus_y_plus_E",
                "baseline_sha256": "abc",
                "teacher_sha256": "def",
            }
        ],
    )
    readiness = check_r1_readiness(load_matrix(matrix), fold_id=0, seed=42)
    assert readiness.ready is False
    assert any("protocol_control" in item for item in readiness.missing_requirements)


def test_evaluate_r1_not_ready_with_only_anchor(tmp_path: Path):
    matrix = tmp_path / "gate2_revised_matrix.csv"
    write_matrix(
        matrix,
        [
            {
                "run_id": "anchor",
                "comparison_role": "loss_ablation_reference",
                "fold_id": 0,
                "seed": 42,
                "R2": "0.63000000",
                "delta_fixed_B0": "-0.04000000",
                "loss_mode": "task_plus_y_plus_E",
                "baseline_sha256": "abc",
                "teacher_sha256": "def",
            }
        ],
    )
    decision = evaluate_r1(matrix, fold_id=0, seed=42)
    assert decision.r1_ready is False
    assert decision.decision_state == "not_ready"
    assert decision.winning_loss is None


def test_find_protocol_control_r2_requires_matching_seed(tmp_path: Path):
    rows = [
        {
            "run_id": "d512_s43",
            "comparison_role": "protocol_control",
            "fold_id": 0,
            "seed": 43,
            "R2": "0.70000000",
        },
        {
            "run_id": "d512_s42",
            "comparison_role": "protocol_control",
            "fold_id": 0,
            "seed": 42,
            "R2": "0.65000000",
        },
    ]
    assert find_protocol_control_r2(rows, fold_id=0, seed=42) == pytest.approx(0.65)
    assert find_protocol_control_r2(rows, fold_id=0, seed=43) == pytest.approx(0.70)


def test_refresh_d512_deltas_is_scoped_by_fold_and_seed(tmp_path: Path):
    matrix = tmp_path / "gate2_revised_matrix.csv"
    write_matrix(
        matrix,
        [
            {
                "run_id": "d512_f0_s42",
                "comparison_role": "protocol_control",
                "fold_id": 0,
                "seed": 42,
                "R2": "0.65000000",
            },
            {
                "run_id": "b3_f0_s42",
                "comparison_role": "loss_ablation",
                "fold_id": 0,
                "seed": 42,
                "R2": "0.63000000",
            },
            {
                "run_id": "b3_f1_s42",
                "comparison_role": "loss_ablation",
                "fold_id": 1,
                "seed": 42,
                "R2": "0.61000000",
            },
        ],
    )
    refresh_d512_deltas(matrix)
    updated = {row["run_id"]: row for row in load_matrix(matrix)}
    assert updated["b3_f0_s42"]["delta_vs_D512_LOSO"] == pytest.approx("-0.02000000")
    assert updated["b3_f1_s42"]["delta_vs_D512_LOSO"] == ""


def test_evaluate_r1_stop_condition(tmp_path: Path):
    matrix = tmp_path / "gate2_revised_matrix.csv"
    write_matrix(
        matrix,
        [
            {
                "run_id": "d512",
                "comparison_role": "protocol_control",
                "fold_id": 0,
                "seed": 42,
                "R2": "0.60000000",
                "delta_fixed_B0": "-0.04000000",
                "loss_mode": "task_plus_y_plus_E",
                "baseline_sha256": "abc",
                "teacher_sha256": "def",
            },
            {
                "run_id": "task_only",
                "comparison_role": "loss_ablation",
                "fold_id": 0,
                "seed": 42,
                "R2": "0.59000000",
                "delta_fixed_B0": "-0.05000000",
                "loss_mode": "task_only",
                "baseline_sha256": "abc",
                "teacher_sha256": "def",
            },
            {
                "run_id": "task_plus_y",
                "comparison_role": "loss_ablation",
                "fold_id": 0,
                "seed": 42,
                "R2": "0.59500000",
                "delta_fixed_B0": "-0.04500000",
                "loss_mode": "task_plus_y",
                "baseline_sha256": "abc",
                "teacher_sha256": "def",
            },
            {
                "run_id": "anchor",
                "comparison_role": "loss_ablation_reference",
                "fold_id": 0,
                "seed": 42,
                "R2": "0.60000000",
                "delta_fixed_B0": "-0.04000000",
                "loss_mode": "task_plus_y_plus_E",
                "baseline_sha256": "abc",
                "teacher_sha256": "def",
            },
        ],
    )
    decision = evaluate_r1(matrix, fold_id=0, seed=42)
    assert decision.r1_ready is True
    assert decision.stop_architecture_sweep is True
    assert decision.d512_status == "red"


def test_loss_overrides_mapping():
    assert loss_overrides("task_only")["lambda_y"] == 0.0
    assert loss_overrides("task_plus_y")["lambda_E"] == 0.0
    assert loss_overrides("task_plus_y_plus_E")["lambda_E"] == 0.1


@pytest.mark.parametrize(
    "preset",
    [
        "b2_d512_protocol_control",
        "b3_d64_task_only",
        "b3_d64_task_plus_y",
        "b5_ema_r4_loso_probe",
        "b6_fir_r4_k5_loso_probe",
        "b3_d128_gate2",
        "b2_d128_winning_loss",
    ],
)
def test_revised_presets_compose(preset: str):
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(
            config_name="train.yaml",
            overrides=[f"experiment={preset}", "data.loso_fold=0"],
        )
    assert cfg.data.validation_protocol == "loso"
    assert cfg.data.include_heldout_in_fit is False
    assert cfg.data.include_heldout_in_test is False
    assert cfg.model.freeze_decoder is True
    assert cfg.comparison_role is not None
