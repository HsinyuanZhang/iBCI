from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import torch

from scripts.aggregate_sua_residual_film import (
    EXPECTED_HEAD_SHAPES,
    EXPECTED_VAL_SESSIONS,
    aggregate,
    parse_seeds,
)


EPOCHS = list(range(5, 13))
ARMS = {
    "t4_continuation": ("B3S", "t4", 1, 0.50),
    "film": ("B3SCF", "t4cf", 2, 0.52),
    "residual_film": ("B3SCFR", "t4cf_residual", 2, 0.56),
    "residual_shuffle": ("B3SCFRS", "t4cf_residual_shuffled", 2, 0.50),
    "residual_nofilm": ("B3SCFRA", "t4cf_residual", 2, 0.50),
}
SESSIONS = EXPECTED_VAL_SESSIONS


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _anchor_state() -> dict[str, torch.Tensor]:
    state = {}
    for index in range(31):
        state[f"student.decoder.weight_{index}"] = torch.tensor([index], dtype=torch.float32)
    for index in range(8):
        state[f"student.id_encoder.base_{index}"] = torch.tensor([index + 100], dtype=torch.float32)
    return state


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path, *, seed: int = 42) -> tuple[Path, Path, Path, Path]:
    teacher = tmp_path / "teacher.ckpt"
    manifest = tmp_path / "manifest.json"
    teacher.write_bytes(b"teacher")
    manifest.write_bytes(b"manifest")
    anchor_run = tmp_path / "anchor_run"
    anchor_ckpt = anchor_run / "epoch_ckpts" / "epoch_011.ckpt"
    anchor_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": _anchor_state()}, anchor_ckpt)
    anchor_meta = {
        "variant": "B3S", "seed": seed, "status": "completed",
        "held_out_test_evaluated": False, "teacher_sha256": _sha(teacher),
        "teacher_checkpoint": str(teacher), "train_val_manifest_sha256": _sha(manifest),
        "train_val_manifest": str(manifest), "side_features": {"group": "t4", "feature_version": 1, "pool_size": 50},
        "training": {"calibration_n_trials": 30, "max_epochs": 12, "no_early_stopping": True, "checkpoint_every_epoch": True},
    }
    _write_json(anchor_run / "run_metadata.json", anchor_meta)
    return teacher, manifest, anchor_ckpt, tmp_path


def _write_arm(tmp_path: Path, arm: str, *, seed: int = 42, mutate_state: bool = False, mutate_substrate: bool = False, pool: int = 50, permutation_seed: int | None = None, anchor_override: Path | None = None) -> None:
    teacher, manifest, anchor, _ = _fixture(tmp_path, seed=seed) if not (tmp_path / "teacher.ckpt").exists() else (tmp_path / "teacher.ckpt", tmp_path / "manifest.json", tmp_path / "anchor_run" / "epoch_ckpts" / "epoch_011.ckpt", tmp_path)
    if anchor_override is not None:
        anchor = anchor_override
    variant, group, version, score = ARMS[arm]
    run_dir = tmp_path / f"run_{arm}_s{seed}"
    (run_dir / "epoch_ckpts").mkdir(parents=True, exist_ok=True)
    side = {"group": group, "feature_version": version, "pool_size": pool, "side_dim": 4 if arm == "t4_continuation" else 6, "electrode_embed_dim": 0, "num_electrodes": 0, "uses_equality_only_relation_membership": False, "normalization_sha256": "a" * 64, "permutation_seed": permutation_seed}
    training = {"calibration_n_trials": 30, "max_epochs": 12, "no_early_stopping": True, "checkpoint_every_epoch": True, "learning_rate": 1e-4, "batch_size": 32, "loss_mode": "task_only", "identity_mode": "calibrated", "deterministic": True, "trial_length": 100, "window_size": 50, "decode_last_timestep_only": True, "lambda_y": 1.0, "lambda_E": 0.1, "limit_train_batches": None, "limit_val_batches": None, "freeze_decoder": False, "freeze_encoder_base": False}
    metadata = {"schema_version": 1, "status": "completed", "held_out_test_evaluated": False, "variant": variant, "seed": seed, "task": "CO", "signal_view": "sua", "split_counts": [27, 6, 6], "max_units_exclusive": 100, "teacher_checkpoint": str(teacher), "teacher_sha256": _sha(teacher), "train_val_manifest": str(manifest), "train_val_manifest_sha256": _sha(manifest), "encoder_warmstart_path": str(anchor), "encoder_warmstart_sha256": _sha(anchor), "side_features": side, "training": training, "decoder_architecture": {"mode": "coupled"}, "fixed_slot": {"enabled": False}, "session_splits": {"val": SESSIONS}, "session_unit_counts": {session: 64 for session in SESSIONS}, "session_files": {"test": []}, "trainer_fit_validation_loader_contract": {"formal_test_sessions_loaded_during_fit": False, "loader_0_sessions": SESSIONS}, "output_dir": str(run_dir), "encoder_cost_profile_reference": {"parameter_count": 20_000}}
    if arm.startswith("residual_"):
        training.update({"freeze_decoder": True, "freeze_encoder_base": True})
        metadata["confidence_film"] = {"confidence_input_order": ["log_residual_variance", "direction_geometry"], "confidence_mask": [True, False], "additive_only": arm == "residual_nofilm", "parameter_matched_six_wide_context": True, "freeze_encoder_base": True, "freeze_decoder": True, "optimizer_trainable_parameter_names": ["id_encoder.confidence_context.0.weight", "id_encoder.confidence_context.0.bias", "id_encoder.confidence_film.weight", "id_encoder.confidence_film.bias"], "optimizer_trainable_parameter_count": 1208}
    metadata_path = run_dir / "run_metadata.json"
    _write_json(metadata_path, metadata)
    per_epoch = {}
    for epoch in EPOCHS:
        state = _anchor_state()
        if arm.startswith("residual_"):
            for index, (name, shape) in enumerate(sorted(EXPECTED_HEAD_SHAPES.items())):
                state[f"student.id_encoder.{name}"] = torch.full(
                    shape, float(epoch + index)
                )
            if mutate_state and epoch == 5:
                state["student.decoder.weight_0"] = torch.tensor([-1.0])
            if mutate_substrate and epoch == 5:
                state["student.id_encoder.base_0"] = torch.tensor([-1.0])
        checkpoint = run_dir / "epoch_ckpts" / f"epoch_{epoch - 1:03d}.ckpt"
        torch.save({"state_dict": state}, checkpoint)
        scores = {name: score for name in SESSIONS}
        per_epoch[str(epoch)] = {"checkpoint_path": str(checkpoint), "checkpoint_sha256": _sha(checkpoint), "mean_r2": score, "per_session_r2": scores}
    payload = {"schema_version": 1, "generated_by": "eval_epoch_window_generic_dandi688.py", "variant": variant, "seed": seed, "task": "CO", "signal_view": "sua", "split_counts": [27, 6, 6], "max_units_exclusive": 100, "no_test_files_evaluated": True, "uses_backward_gradients": False, "uses_behavior_labels_for_weight_updates": False, "calibration_features_use_behavior_labels": True, "calibration_trial_selection_uses_behavior_labels": False, "calibration_feature_label_scope": "chronological_rewarded_trials[0:50]", "protocol": {"total_epochs": 12, "burn_in_epochs": 4, "selection_mode": "first", "calibration_n": 30, "train_activity_calibration_n": 30, "evaluation_forward_calibration_n": 30, "label_feature_calibration_n": 50, "pool_size": 50, "epoch_window": EPOCHS}, "checkpoint_selection_rule": "pre_declared_fixed_epoch_window_no_argmax", "run_dir": str(run_dir), "run_metadata_path": str(metadata_path), "run_metadata_sha256": _sha(metadata_path), "teacher_ckpt": str(teacher), "teacher_ckpt_sha256": _sha(teacher), "train_val_manifest": str(manifest), "train_val_manifest_sha256": _sha(manifest), "session_splits": {"val": SESSIONS}, "session_unit_counts": metadata["session_unit_counts"], "epoch_list": EPOCHS, "per_epoch": per_epoch, "per_epoch_mean_r2": {str(epoch): score for epoch in EPOCHS}, "variant_score": score}
    _write_json(tmp_path / f"{arm}_m50_s{seed}.json", payload)


def _write_complete_round(tmp_path: Path) -> None:
    _fixture(tmp_path)
    for arm in ARMS:
        _write_arm(tmp_path, arm, permutation_seed=42 if arm == "residual_shuffle" else None)


def test_accepts_frozen_round_and_keeps_three_seed_gate_closed(tmp_path):
    _write_complete_round(tmp_path)
    result = aggregate(tmp_path, (42,))
    assert result["stage0_descriptive_mechanism_pass"] is True
    assert result["formal_effectiveness_eligible"] is False
    assert result["formal_effectiveness_pass"] is False


def test_rejects_actual_checkpoint_decoder_drift(tmp_path):
    _write_complete_round(tmp_path)
    _write_arm(tmp_path, "residual_film", mutate_state=True, permutation_seed=None)
    with pytest.raises(ValueError, match="frozen decoder tensor drifted"):
        aggregate(tmp_path, (42,))


def test_rejects_actual_checkpoint_t4_substrate_drift(tmp_path):
    _write_complete_round(tmp_path)
    _write_arm(tmp_path, "residual_film", mutate_substrate=True, permutation_seed=None)
    with pytest.raises(ValueError, match="frozen T4 substrate tensor drifted"):
        aggregate(tmp_path, (42,))


def test_rejects_m15_and_residual_permutation_drift(tmp_path):
    _write_complete_round(tmp_path)
    _write_arm(tmp_path, "residual_film", pool=15, permutation_seed=None)
    with pytest.raises(ValueError, match="side pool"):
        aggregate(tmp_path, (42,))
    _write_arm(tmp_path, "residual_film", pool=50, permutation_seed=None)
    _write_arm(tmp_path, "residual_shuffle", permutation_seed=43)
    with pytest.raises(ValueError, match="residual permutation seed"):
        aggregate(tmp_path, (42,))


def test_rejects_anchor_or_checkpoint_provenance_drift(tmp_path):
    _write_complete_round(tmp_path)
    path = tmp_path / "residual_nofilm_m50_s42.json"
    payload = json.loads(path.read_text())
    payload["per_epoch"]["5"]["checkpoint_sha256"] = "0" * 64
    _write_json(path, payload)
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        aggregate(tmp_path, (42,))
    _write_complete_round(tmp_path)
    metadata_path = tmp_path / "run_residual_nofilm_s42" / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["encoder_warmstart_sha256"] = "0" * 64
    _write_json(metadata_path, metadata)
    payload = json.loads(path.read_text())
    payload["run_metadata_sha256"] = _sha(metadata_path)
    _write_json(path, payload)
    with pytest.raises(ValueError, match="warm-start SHA-256"):
        aggregate(tmp_path, (42,))


def test_rejects_actual_head_shape_drift(tmp_path):
    _write_complete_round(tmp_path)
    artifact_path = tmp_path / "residual_film_m50_s42.json"
    artifact = json.loads(artifact_path.read_text())
    checkpoint = Path(artifact["per_epoch"]["5"]["checkpoint_path"])
    checkpoint_payload = torch.load(checkpoint, weights_only=True)
    checkpoint_payload["state_dict"][
        "student.id_encoder.confidence_context.0.weight"
    ] = torch.zeros(1)
    torch.save(checkpoint_payload, checkpoint)
    artifact["per_epoch"]["5"]["checkpoint_sha256"] = _sha(checkpoint)
    _write_json(artifact_path, artifact)

    with pytest.raises(ValueError, match="head shape"):
        aggregate(tmp_path, (42,))


def test_rejects_formal_file_or_wrong_session_receipt(tmp_path):
    _write_complete_round(tmp_path)
    metadata_path = tmp_path / "run_residual_film_s42" / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["session_files"]["test"] = ["sealed_formal.nwb"]
    _write_json(metadata_path, metadata)
    artifact_path = tmp_path / "residual_film_m50_s42.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["run_metadata_sha256"] = _sha(metadata_path)
    _write_json(artifact_path, artifact)
    with pytest.raises(ValueError, match="no formal files opened"):
        aggregate(tmp_path, (42,))

    _write_complete_round(tmp_path)
    artifact = json.loads(artifact_path.read_text())
    for record in artifact["per_epoch"].values():
        value = record["per_session_r2"].pop(SESSIONS[-1])
        record["per_session_r2"]["wrong_session"] = value
    _write_json(artifact_path, artifact)
    with pytest.raises(ValueError, match="exact validation sessions"):
        aggregate(tmp_path, (42,))


def test_seed_parser_rejects_unregistered_or_duplicate_seeds():
    assert parse_seeds("42,43,44") == (42, 43, 44)
    with pytest.raises(argparse.ArgumentTypeError, match="subset"):
        parse_seeds("42,45")
    with pytest.raises(argparse.ArgumentTypeError, match="subset"):
        parse_seeds("42,42")


def test_cli_refuses_to_overwrite_aggregate_before_loading_results(tmp_path):
    output = tmp_path / "aggregate.json"
    output.write_text('{"immutable": true}\n', encoding="utf-8")
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "aggregate_sua_residual_film.py"
    )
    failed = subprocess.run(
        [
            "/home/xinyuan/miniconda3/envs/spint/bin/python",
            str(script),
            "--result-dir",
            str(tmp_path / "missing-results"),
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
    assert "Refusing to overwrite aggregate" in failed.stderr
    assert json.loads(output.read_text(encoding="utf-8")) == {"immutable": True}
