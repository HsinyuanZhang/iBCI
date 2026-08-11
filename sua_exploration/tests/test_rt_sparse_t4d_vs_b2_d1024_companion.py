"""Synthetic fail-closed contracts for the RT T4d--B2 companion."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/rt_sparse_t4d_vs_b2_d1024_companion.py"


def module():
    spec = importlib.util.spec_from_file_location("rt_t4d_b2_companion_test", SCRIPT)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def paired_records(deltas=None):
    mod = module()
    if deltas is None:
        deltas = [0.1] * 15
    stage, b2 = {}, {}
    for fold in range(15):
        session = f"ses-{fold:02d}"
        query = (_sha(f"start{fold}"), _sha(f"target{fold}"), _sha(f"both{fold}"))
        stage[fold] = {
            "session": session, "r2": 0.5 + float(deltas[fold]), "query_digests": query,
            "inner_train_sessions": [f"train-{fold}-a", f"train-{fold}-b"],
            "inner_validation_session": f"val-{fold}",
        }
        b2[fold] = {
            "outer": {"outer_target_session": session, "r2_variance_weighted": 0.5},
            "split": {"target_session": session, "inner_train_sessions": [f"train-{fold}-a", f"train-{fold}-b"], "inner_validation_session": f"val-{fold}"},
        }
    return mod, stage, b2


def test_real_happy_path_schema_and_default_prospective_subset():
    mod, stage, b2 = paired_records()
    result = mod.aggregate_companion(stage, b2, reconstructed_query=lambda entry: stage[int(entry["outer"]["outer_target_session"].split("-")[-1])]["query_digests"])
    assert result["status"] == "PASS_TERMINAL_COMPANION_READ_ONLY"
    assert result["full15_descriptive"]["n"] == 15
    assert result["prospective_subset"]["folds"] == list(range(4, 15))
    assert result["prospective_subset"]["statistics"]["n"] == 11
    assert result["prospective_subset"]["statistics"]["gate_mean_positive_median_positive_sign_majority"] is True


def test_missing_fold_fails_closed():
    mod, stage, b2 = paired_records()
    del stage[14]
    with pytest.raises(mod.CompanionError, match="exactly 15"):
        mod.aggregate_companion(stage, b2, reconstructed_query=lambda _: ("x" * 64,) * 3)


def test_query_digest_mismatch_is_not_replaced_by_count():
    mod, stage, b2 = paired_records()
    with pytest.raises(mod.CompanionError, match="query digest"):
        mod.aggregate_companion(stage, b2, reconstructed_query=lambda _: ("0" * 64,) * 3)


def test_positive_mean_negative_median_does_not_pass_gate():
    # One large win cannot make the pre-registered sign/median gate pass.
    mod, stage, b2 = paired_records([15.0] + [-1.0] * 14)
    result = mod.aggregate_companion(stage, b2, reconstructed_query=lambda entry: stage[int(entry["outer"]["outer_target_session"].split("-")[-1])]["query_digests"])
    stats = result["full15_descriptive"]
    assert stats["mean"] > 0.0 and stats["median"] < 0.0
    assert stats["gate_mean_positive_median_positive_sign_majority"] is False


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_stale_legacy_aggregate_hash_fails_closed(tmp_path, monkeypatch):
    mod = module()
    payload = {"schema": "rt_stage_r_rc_vs_rs_b2_d1024_paired_aggregate_v1", "status": "PASS_RT_STAGE_R_RC_VS_RS_B2_D1024_ALL_15_PAIRED", "rows": []}
    path = tmp_path / "legacy.json"
    _write_json(path, payload)
    path.chmod(0o444)
    monkeypatch.setattr(mod, "B2_AGGREGATE_SHA256", "f" * 64)
    with pytest.raises(mod.CompanionError, match="SHA"):
        mod._legacy_b2_rows(path)


def test_fold10_raced_cell_is_rejected_by_immutable_aggregate(tmp_path, monkeypatch):
    mod = module()
    rows = []
    for fold in range(15):
        rows.append({"fold": fold, "r_s_b2_d1024_r2": 0.0, "r_s_files": {}, "r_s_source": "local_3090_supervisor_fold3_14"})
    payload = {"schema": "rt_stage_r_rc_vs_rs_b2_d1024_paired_aggregate_v1", "status": "PASS_RT_STAGE_R_RC_VS_RS_B2_D1024_ALL_15_PAIRED", "rows": rows}
    path = tmp_path / "legacy.json"
    _write_json(path, payload)
    path.chmod(0o444)
    monkeypatch.setattr(mod, "B2_AGGREGATE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(mod.CompanionError, match="fold 10"):
        mod._legacy_b2_rows(path)


def _teacher_witness_path(workspace: Path, root_name: str, fold: int, *, run_name: str | None = None) -> Path:
    run_name = run_name or f"zero4_f{fold}_s42_terminal"
    return (
        workspace
        / "streaming_calibration_exp/outputs/rt_stage_r_b2_local3090"
        / root_name
        / "_artifacts"
        / run_name
        / "teacher_metadata.json"
    )


def test_fold10_selects_race_recovery_witness_over_supervisor_candidate(tmp_path, monkeypatch):
    mod = module()
    monkeypatch.setattr(mod, "WORKSPACE", tmp_path)
    supervisor = _teacher_witness_path(tmp_path, "supervisor_folds03_14_v1", 10)
    recovery = _teacher_witness_path(tmp_path, "fold10_race_recovery_v1", 10)
    _write_json(supervisor, {"teacher_checkpoint_sha256": "a" * 64})
    _write_json(recovery, {"teacher_checkpoint_sha256": "b" * 64})

    assert mod._teacher_metadata_for_fold(10) == recovery


def test_nonfold10_multiple_teacher_witnesses_fail_closed(tmp_path, monkeypatch):
    mod = module()
    monkeypatch.setattr(mod, "WORKSPACE", tmp_path)
    _write_json(_teacher_witness_path(tmp_path, "supervisor_folds03_14_v1", 9), {"teacher_checkpoint_sha256": "a" * 64})
    imported = (
        tmp_path
        / "streaming_calibration_exp/outputs/rt_stage_r_b2_imported_remote/_artifacts"
        / "zero4_f9_s42_terminal"
        / "teacher_metadata.json"
    )
    _write_json(imported, {"teacher_checkpoint_sha256": "b" * 64})

    with pytest.raises(mod.CompanionError, match="ambiguous teacher metadata witnesses"):
        mod._teacher_metadata_for_fold(9)


def test_fold10_missing_race_recovery_witness_fails_closed_even_with_supervisor(tmp_path, monkeypatch):
    mod = module()
    monkeypatch.setattr(mod, "WORKSPACE", tmp_path)
    supervisor = _teacher_witness_path(tmp_path, "supervisor_folds03_14_v1", 10)
    _write_json(supervisor, {"teacher_checkpoint_sha256": "a" * 64})

    with pytest.raises(mod.CompanionError, match="missing freeze-bound race-recovery"):
        mod._teacher_metadata_for_fold(10)


def test_config_mismatch_is_rejected_before_scoring(tmp_path, monkeypatch):
    mod = module()
    session = "ses-RT-20131009"
    nwb = tmp_path / "nwb" / "sub-C_ses-RT-20131009_behavior+ecephys.nwb"
    nwb.parent.mkdir(parents=True); nwb.write_bytes(b"nwb")
    monkeypatch.setattr(mod, "DEFAULT_NWB_ROOT", nwb.parent)
    teacher = tmp_path / "teacher.json"
    _write_json(teacher, {"teacher_checkpoint_sha256": "a" * 64})
    monkeypatch.setattr(mod, "_teacher_metadata_for_fold", lambda _: teacher)
    outer = {"arm": "zero4", "outer_loso_fold": 0, "seed": 42, "status": "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP", "outer_target_session": session, "outer_target_path": str(nwb), "query_windows_evaluated": 5, "r2_variance_weighted": 0.0, "target_backpropagation": False, "optimizer_present": False, "model_training_mode": False, "target_query_labels_used_for_calibration": False, "target_query_labels_used_for_normalization": False, "target_query_labels_used_for_checkpoint_selection": False, "model_state_unchanged": True, "model_state_sha256_before": "b" * 64, "model_state_sha256_after": "b" * 64}
    selection = {"arm": "zero4", "outer_loso_fold": 0, "seed": 42, "status": "PASS_FIT_INNER_SELECTION_ONLY", "selected_by_metric": "val_heldin/r2_mean", "selected_metric_scope": "inner_validation_session_only", "formal_heldout_opened": False, "outer_target_loaded_during_fit": False, "outer_target_query_labels_read_during_fit": False}
    split = {"validation_protocol": "nested_loso", "outer_loso_fold": 0, "requested_side_feature_group": "zero4", "target_session": session, "calibration": {"budget_trials": 24}, "query": {"query_start_trial": 24, "window_size_bins": 50}}
    config = {"seed": 42, "no_early_stopping": True, "optimized_metric": "val_heldin/r2_mean", "data": {"calibration_n_trials": 24, "window_size": 50, "max_trial_length": 100, "session_window_budget": 999, "loso_fold": 0, "outer_loso_fold": 0, "side_feature_group": "zero4", "query_start_trial": 24, "sampler_seed": 42, "random_calibration": False, "smooth_calibration": False}, "model": {"variant": "B2", "id_hidden_dim": 1024, "freeze_decoder": False, "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-4}}, "trainer": {"max_epochs": 35}}
    paths = {"outer": tmp_path / "outer.json", "selection": tmp_path / "selection.json", "split": tmp_path / "split.json", "config": tmp_path / "config.yaml"}
    _write_json(paths["outer"], outer); _write_json(paths["selection"], selection); _write_json(paths["split"], split)
    paths["config"].write_text(yaml.safe_dump(config), encoding="utf-8")
    row = {"r_s_files": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in paths.items()}}
    nwb_rows = {session: {"path": str(nwb), "sha256": hashlib.sha256(nwb.read_bytes()).hexdigest()}}
    with pytest.raises(mod.CompanionError, match="session_window_budget"):
        mod._validate_b2_self(0, row, stage_teacher_sha="a" * 64, nwb_rows=nwb_rows)
