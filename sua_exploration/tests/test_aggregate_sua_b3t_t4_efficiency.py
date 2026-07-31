from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.aggregate_sua_b3t_t4_efficiency import (
    EPOCHS,
    EXPECTED_COST,
    EXPECTED_VAL_SESSIONS,
    NONINFERIORITY_MARGIN,
    check_aligned_first_gate,
    load_cell,
    parse_seeds,
    summarize,
)


def _write_cell(
    tmp_path,
    *,
    arm: str,
    variant: str,
    side: str,
    seed: int = 42,
    score: float = 0.5,
):
    teacher_path = tmp_path / "teacher.ckpt"
    manifest_path = tmp_path / "strict_manifest.json"
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    epoch_ckpt_dir = tmp_path / "epoch_ckpts"
    epoch_ckpt_dir.mkdir(exist_ok=True)
    for epoch in EPOCHS:
        checkpoint_path = epoch_ckpt_dir / f"epoch_{epoch - 1:03d}.ckpt"
        if not checkpoint_path.exists():
            checkpoint_path.write_bytes(f"checkpoint-{epoch}".encode())
    if not teacher_path.exists():
        teacher_path.write_bytes(b"shared B3T teacher")
    if not manifest_path.exists():
        manifest_path.write_text('{"strict": true}', encoding="utf-8")
    teacher_sha = hashlib.sha256(teacher_path.read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    normalization_sha = "a" * 64
    metadata = {
        "schema_version": 1,
        "variant": variant,
        "seed": seed,
        "task": "CO",
        "signal_view": "sua",
        "split_counts": [27, 6, 6],
        "max_units_exclusive": 100,
        "status": "completed",
        "held_out_test_evaluated": False,
        "encoder_warmstart_path": None,
        "output_dir": str(tmp_path),
        "data_dir": str(data_dir),
        "teacher_checkpoint": str(teacher_path),
        "teacher_sha256": teacher_sha,
        "train_val_manifest": str(manifest_path),
        "train_val_manifest_sha256": manifest_sha,
        "side_features": {
            "group": side,
            "feature_version": 1,
            "pool_size": 30,
            "side_dim": 4,
            "electrode_embed_dim": 0,
            "num_electrodes": 0,
            "uses_equality_only_relation_membership": False,
            "normalization_sha256": normalization_sha,
            "permutation_seed": seed if arm == "b3t_ts4" else None,
        },
        "decoder_architecture": {"mode": "coupled"},
        "fixed_slot": {"enabled": False},
        "session_splits": {
            "train": ["train_session"],
            "val": EXPECTED_VAL_SESSIONS,
            "test": ["sealed_formal_session"],
        },
        "session_unit_counts": {
            "train_session": 64,
            **{session: 64 for session in EXPECTED_VAL_SESSIONS},
        },
        "trainer_fit_validation_loader_contract": {
            "formal_test_sessions_loaded_during_fit": False,
            "loader_0_sessions": EXPECTED_VAL_SESSIONS,
        },
        "session_files": {"test": []},
        "training": {
            "calibration_n_trials": 30,
            "max_epochs": 12,
            "no_early_stopping": True,
            "checkpoint_every_epoch": True,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "loss_mode": "task_only",
            "identity_mode": "calibrated",
            "freeze_decoder": False,
            "freeze_encoder_base": False,
            "deterministic": True,
            "trial_length": 100,
            "window_size": 50,
            "decode_last_timestep_only": True,
            "lambda_y": 1.0,
            "lambda_E": 0.1,
            "epoch_checkpoints_dir": str(epoch_ckpt_dir),
        },
        "encoder_cost_profile_reference": EXPECTED_COST[arm],
    }
    metadata_path = tmp_path / f"{arm}_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    metadata_sha = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    session_scores = {
        session: score + 0.001 * index
        for index, session in enumerate(EXPECTED_VAL_SESSIONS)
    }
    epoch_mean = float(np.mean(list(session_scores.values())))
    artifact = {
        "schema_version": 1,
        "purpose": "epoch_window_deterministic_checkpoint_selection",
        "generated_by": "eval_epoch_window_generic_dandi688.py",
        "run_dir": str(tmp_path),
        "variant": variant,
        "seed": seed,
        "task": "CO",
        "signal_view": "sua",
        "split_counts": [27, 6, 6],
        "max_units_exclusive": 100,
        "data_dir": str(data_dir),
        "epoch_list": EPOCHS,
        "checkpoint_selection_rule": "pre_declared_fixed_epoch_window_no_argmax",
        "no_test_files_evaluated": True,
        "uses_backward_gradients": False,
        "uses_behavior_labels_for_weight_updates": False,
        "calibration_features_use_behavior_labels": True,
        "calibration_trial_selection_uses_behavior_labels": False,
        "calibration_feature_label_scope": "chronological_rewarded_trials[0:30]",
        "protocol": {
            "total_epochs": 12,
            "burn_in_epochs": 4,
            "selection_mode": "first",
            "calibration_n": 30,
            "train_activity_calibration_n": 30,
            "evaluation_forward_calibration_n": 30,
            "label_feature_calibration_n": 30,
            "pool_size": 30,
            "epoch_window": EPOCHS,
        },
        "run_metadata_path": str(metadata_path),
        "run_metadata_sha256": metadata_sha,
        "teacher_ckpt": str(teacher_path),
        "teacher_ckpt_sha256": teacher_sha,
        "train_val_manifest": str(manifest_path),
        "train_val_manifest_sha256": manifest_sha,
        "session_splits": metadata["session_splits"],
        "session_unit_counts": metadata["session_unit_counts"],
        "per_epoch_mean_r2": {str(epoch): epoch_mean for epoch in EPOCHS},
        "variant_score": epoch_mean,
        "per_epoch": {
            str(epoch): {
                "checkpoint_path": str(
                    epoch_ckpt_dir / f"epoch_{epoch - 1:03d}.ckpt"
                ),
                "checkpoint_sha256": hashlib.sha256(
                    (epoch_ckpt_dir / f"epoch_{epoch - 1:03d}.ckpt").read_bytes()
                ).hexdigest(),
                "per_session_r2": session_scores,
                "mean_r2": epoch_mean,
            }
            for epoch in EPOCHS
        },
    }
    artifact_path = tmp_path / f"{arm}_s{seed}.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact_path


def _rebind_metadata(tmp_path, arm: str) -> tuple[Path, dict]:
    metadata_path = tmp_path / f"{arm}_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifact_path = tmp_path / f"{arm}_s42.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["run_metadata_sha256"] = hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact_path, metadata


def test_load_cell_fail_closes_on_measured_cost_drift(tmp_path):
    path = _write_cell(
        tmp_path, arm="b3t_t4", variant="B3TS", side="t4"
    )
    sessions, values, receipt = load_cell(path, "b3t_t4", 42)
    cost = receipt["cost_profile"]
    assert len(sessions) == 6
    assert values.shape == (6,)
    assert cost["parameter_count"] == 12_658
    assert cost["supports_bin_streaming"] is True
    assert cost["trial_buffer_bytes"] == 3_072
    assert cost["peak_live_state_bytes"] == 19_456

    metadata_path = tmp_path / "b3t_t4_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["encoder_cost_profile_reference"]["parameter_count"] += 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["run_metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="cost parameter_count"):
        load_cell(path, "b3t_t4", 42)


def test_load_cell_rejects_non_streaming_b3t_receipt(tmp_path):
    path = _write_cell(
        tmp_path, arm="b3t_t4", variant="B3TS", side="t4"
    )
    metadata_path = tmp_path / "b3t_t4_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["encoder_cost_profile_reference"]["supports_bin_streaming"] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["run_metadata_sha256"] = hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="cost supports_bin_streaming"):
        load_cell(path, "b3t_t4", 42)


def test_load_cell_rejects_wrong_metadata_validation_session(tmp_path):
    path = _write_cell(tmp_path, arm="t4", variant="B3S", side="t4")
    metadata_path = tmp_path / "t4_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["session_splits"]["val"] = ["wrong_session"] * 6
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _rebind_metadata(tmp_path, "t4")

    with pytest.raises(ValueError, match="exact validation sessions"):
        load_cell(path, "t4", 42)


def test_load_cell_rejects_formal_file_receipt(tmp_path):
    path = _write_cell(tmp_path, arm="t4", variant="B3S", side="t4")
    metadata_path = tmp_path / "t4_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["session_files"]["test"] = ["formal_test.nwb"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _rebind_metadata(tmp_path, "t4")

    with pytest.raises(ValueError, match="no formal files opened"):
        load_cell(path, "t4", 42)


def test_load_cell_rejects_variant_score_drift(tmp_path):
    path = _write_cell(tmp_path, arm="t4", variant="B3S", side="t4")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["variant_score"] += 0.01
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="variant_score"):
        load_cell(path, "t4", 42)


def test_load_cell_requires_seeded_ts4_permutation_receipt(tmp_path):
    path = _write_cell(tmp_path, arm="b3t_ts4", variant="B3TS", side="ts4")
    metadata_path = tmp_path / "b3t_ts4_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["side_features"]["permutation_seed"] = None
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _rebind_metadata(tmp_path, "b3t_ts4")

    with pytest.raises(ValueError, match="permutation receipt"):
        load_cell(path, "b3t_ts4", 42)


def test_load_cell_rejects_checkpoint_byte_drift(tmp_path):
    path = _write_cell(tmp_path, arm="t4", variant="B3S", side="t4")
    checkpoint = tmp_path / "epoch_ckpts" / "epoch_004.ckpt"
    checkpoint.write_bytes(b"mutated after evaluation")

    with pytest.raises(ValueError, match="checkpoint bytes"):
        load_cell(path, "t4", 42)


def test_parse_seeds_rejects_non_predeclared_seed():
    assert parse_seeds("42,43,44") == (42, 43, 44)
    with pytest.raises(Exception, match="subset"):
        parse_seeds("42,45")


def test_strict_superiority_requires_three_seeds_not_a_positive_pilot():
    sessions = [f"session_{index}" for index in range(6)]
    one_seed = summarize(
        np.full((1, 6), 0.54),
        np.full((1, 6), 0.50),
        seeds=(42,),
        sessions=sessions,
    )
    assert one_seed["passes_stage0_descriptive_gates"] is True
    assert one_seed["passes_strict_superiority"] is False
    assert one_seed["strict_superiority_gates"]["at_least_three_predeclared_seeds"] is False

    three_seeds = summarize(
        np.full((3, 6), 0.54),
        np.full((3, 6), 0.50),
        seeds=(42, 43, 44),
        sessions=sessions,
    )
    assert three_seeds["passes_strict_superiority"] is True


def _write_aligned_pair(tmp_path, *, fresh_score: float, aligned_score: float):
    _write_cell(tmp_path, arm="t4", variant="B3S", side="t4", score=fresh_score)
    _write_cell(
        tmp_path,
        arm="b3t_t4",
        variant="B3TS",
        side="t4",
        score=aligned_score,
    )


def test_aligned_first_gate_allows_positive_observed_delta(tmp_path):
    _write_aligned_pair(tmp_path, fresh_score=0.50, aligned_score=0.52)
    gate = check_aligned_first_gate(tmp_path)

    assert gate["control_permitted"] is True
    assert gate["b3t_t4_minus_fresh_t4"]["mean_delta_r2"] == pytest.approx(0.02)
    assert (
        gate["cost_profiles"]["fresh_t4"]["parameter_count"]
        == EXPECTED_COST["t4"]["parameter_count"]
    )
    assert (
        gate["cost_profiles"]["b3t_t4"]["supports_bin_streaming"]
        == EXPECTED_COST["b3t_t4"]["supports_bin_streaming"]
    )


def test_aligned_first_gate_rejects_delta_below_margin(tmp_path):
    _write_aligned_pair(tmp_path, fresh_score=0.50, aligned_score=0.469)
    gate = check_aligned_first_gate(tmp_path)

    assert gate["b3t_t4_minus_fresh_t4"]["mean_delta_r2"] == pytest.approx(-0.031)
    assert gate["control_permitted"] is False


def test_aligned_first_gate_accepts_inclusive_margin_boundary(tmp_path):
    _write_aligned_pair(tmp_path, fresh_score=0.50, aligned_score=0.47)
    gate = check_aligned_first_gate(tmp_path)

    assert gate["b3t_t4_minus_fresh_t4"]["mean_delta_r2"] == pytest.approx(
        NONINFERIORITY_MARGIN
    )
    assert gate["control_permitted"] is True


def test_aligned_first_gate_supports_predeclared_replication_seed(tmp_path):
    _write_cell(
        tmp_path,
        arm="t4",
        variant="B3S",
        side="t4",
        seed=43,
        score=0.50,
    )
    _write_cell(
        tmp_path,
        arm="b3t_t4",
        variant="B3TS",
        side="t4",
        seed=43,
        score=0.50,
    )
    gate = check_aligned_first_gate(tmp_path, seed=43)
    assert gate["protocol"]["seed"] == 43
    assert gate["control_permitted"] is True


def test_aligned_first_gate_rejects_cross_arm_teacher_drift(tmp_path):
    _write_aligned_pair(tmp_path, fresh_score=0.50, aligned_score=0.50)
    different_teacher = tmp_path / "different_teacher.ckpt"
    different_teacher.write_bytes(b"different B3T teacher")
    metadata_path = tmp_path / "b3t_t4_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["teacher_checkpoint"] = str(different_teacher)
    metadata["teacher_sha256"] = hashlib.sha256(different_teacher.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    artifact_path, _ = _rebind_metadata(tmp_path, "b3t_t4")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["teacher_ckpt"] = str(different_teacher)
    artifact["teacher_ckpt_sha256"] = metadata["teacher_sha256"]
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="teacher_sha256"):
        check_aligned_first_gate(tmp_path)


def test_aligned_first_gate_rejects_cross_arm_normalization_drift(tmp_path):
    _write_aligned_pair(tmp_path, fresh_score=0.50, aligned_score=0.50)
    metadata_path = tmp_path / "b3t_t4_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["side_features"]["normalization_sha256"] = "b" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _rebind_metadata(tmp_path, "b3t_t4")

    with pytest.raises(ValueError, match="normalization_sha256"):
        check_aligned_first_gate(tmp_path)


def test_full_aggregate_rejects_three_arm_normalization_drift(tmp_path):
    _write_cell(tmp_path, arm="t4", variant="B3S", side="t4")
    _write_cell(tmp_path, arm="b3t_t4", variant="B3TS", side="t4")
    _write_cell(tmp_path, arm="b3t_ts4", variant="B3TS", side="ts4")
    metadata_path = tmp_path / "b3t_ts4_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["side_features"]["normalization_sha256"] = "b" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _rebind_metadata(tmp_path, "b3t_ts4")

    root = Path(__file__).resolve().parents[1]
    aggregate = root / "scripts" / "aggregate_sua_b3t_t4_efficiency.py"
    failed = subprocess.run(
        [
            "/home/xinyuan/miniconda3/envs/spint/bin/python",
            str(aggregate),
            "--result-dir",
            str(tmp_path),
            "--seeds",
            "42",
            "--out",
            str(tmp_path / "aggregate.json"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode != 0
    assert "normalization_sha256" in failed.stderr
    assert not (tmp_path / "aggregate.json").exists()


def test_full_aggregate_refuses_to_overwrite_existing_output(tmp_path):
    root = Path(__file__).resolve().parents[1]
    aggregate = root / "scripts" / "aggregate_sua_b3t_t4_efficiency.py"
    output = tmp_path / "aggregate.json"
    output.write_text('{"immutable": true}\n', encoding="utf-8")
    failed = subprocess.run(
        [
            "/home/xinyuan/miniconda3/envs/spint/bin/python",
            str(aggregate),
            "--result-dir",
            str(tmp_path),
            "--seeds",
            "42",
            "--out",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode != 0
    assert "refusing to overwrite aggregate" in failed.stderr
    assert json.loads(output.read_text(encoding="utf-8")) == {"immutable": True}


def test_aligned_first_gate_missing_result_is_read_only(tmp_path):
    future_results = tmp_path / "future_results"
    with pytest.raises(FileNotFoundError):
        check_aligned_first_gate(future_results)
    assert future_results.exists() is False


def test_b3t_ts4_runner_exposes_and_orders_read_only_gate():
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_sua_b3t_t4_one_cell.sh"
    env = os.environ | {"ARM": "b3t_ts4", "SEED": "42", "GPU": "0"}
    dry_run = subprocess.run(
        ["bash", str(runner), "--dry-run"],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "ALIGNED-FIRST CONTROL GATE (required before --launch):" in dry_run.stdout
    assert "--aligned-only" in dry_run.stdout
    assert "--seeds 42" in dry_run.stdout

    source = runner.read_text(encoding="utf-8")
    launch_block = source.split('[[ -x "$PY" ]]', maxsplit=1)[1]
    assert launch_block.index("--aligned-only") < launch_block.index(
        '[[ ! -e "$RUN_DIR" ]]'
    )
    assert launch_block.index("--aligned-only") < launch_block.index(
        'mkdir -p "$LOGS"'
    )
    assert source.count('[[ ! -e "$RESULT" ]]') == 2
