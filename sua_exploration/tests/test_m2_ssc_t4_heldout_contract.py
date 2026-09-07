"""Synthetic, no-NWB tests for the frozen SSC-T4 held-out closure."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "sua_exploration/scripts"
PROTOCOL_SHA = "c769f39703f33e1e1f1f0c02a77d7cc9cd28356e1af2b51ec47fbc214aec24dc"
SESSIONS = [
    "ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1", "ses-2020-11-24-Run1", "ses-2020-11-24-Run2",
]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SOURCE = load("aggregate_m2_ssc_t4_source")
HELDOUT = load("aggregate_m2_ssc_t4_heldout")
COMPLETION = load("validate_m2_clean_teacher_completion")
REPLICATION = load("aggregate_m2_t4_clean_spint_replication")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def audit() -> dict[str, dict[str, object]]:
    return {session: {"support_trials": 24, "query_start_trial": 24, "window_size": 50,
                      "full_window_disjoint": True, "query_trials": 3, "eligible_windows": 9,
                      "raw_query_start_bin": 100, "minimum_window_start_padded_bin": 149}
            for session in SESSIONS}


def config(arm: str, *, test: bool, checkpoint: Path, receipt: Path) -> dict:
    return {
        "seed": 42, "train": not test, "test": True, "no_early_stopping": True,
        "require_baseline_validation": False,
        "trainer": {"max_epochs": 12}, "ckpt_path": str(checkpoint) if test else None,
        "data": {"task": "m2", "loso_fold": 1, "calibration_n_trials": 24,
                 "random_calibration": False, "include_heldout_in_fit": False,
                 "include_heldout_in_test": test, "query_start_trial": 24 if test else 0,
                 "side_feature_group": "t4"},
        "model": {"variant": "B3S", "ssc_t4_prediction_consistency_weight": 0.0 if arm == "ordinary_t4" else 1.0,
                  "require_clean_teacher_receipt": True, "teacher_receipt_path": str(receipt)},
    }


def source_artifact(tmp_path: Path, arm: str, receipt: Path, teacher_sha: str) -> tuple[Path, Path]:
    root = tmp_path / f"source_{arm}"; root.mkdir()
    checkpoint = root / "student.ckpt"; checkpoint.write_bytes(f"student-{arm}".encode())
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(config(arm, test=False, checkpoint=checkpoint, receipt=receipt)))
    split = {"heldout_evaluated_in_fit": False, "heldout_evaluated_in_test": False,
             "native_t4_normalization": {"feature_group": "t4", "sha256": "normalization-sha", "train_sessions": ["a", "b"]}}
    write_json(root / "split_manifest.json", split)
    write_json(root / "checkpoint_manifest.json", {"artifact_checkpoint_path": str(checkpoint), "artifact_checkpoint_sha256": sha(checkpoint), "selected_by_metric": "val_heldin/r2_mean"})
    write_json(root / "teacher_metadata.json", {"teacher_checkpoint_path": "/teacher.ckpt", "teacher_checkpoint_sha256": teacher_sha})
    (root / "metrics_summary.csv").write_text("split,R2_variance_weighted\ntest_heldin,0.25\n")
    return root, checkpoint


def test_source_gate_requires_fixed_heldin_selection_and_test_heldin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    receipt = tmp_path / "teacher_receipt.json"
    write_json(receipt, {"selected_checkpoint": {"sha256": "teacher-sha"}})
    ordinary, _ = source_artifact(tmp_path, "ordinary_t4", receipt, "teacher-sha")
    ssc, _ = source_artifact(tmp_path, "ssc_t4", receipt, "teacher-sha")
    protocol = tmp_path / "protocol.json"; protocol.write_bytes(b"protocol")
    out = tmp_path / "source_gate.json"
    monkeypatch.setattr(SOURCE, "PROTOCOL_RECEIPT_SHA256", sha(protocol))
    monkeypatch.setattr(sys, "argv", ["source", "--ordinary", str(ordinary), "--ssc", str(ssc), "--protocol-receipt", str(protocol), "--out", str(out)])
    SOURCE.main()
    result = json.loads(out.read_text())
    assert result["arms"]["ordinary_t4"]["checkpoint_selection"] == {"selected_by_metric": "val_heldin/r2_mean", "no_early_stopping": True, "max_epochs": 12}
    assert result["arms"]["ordinary_t4"]["teacher"] == result["arms"]["ssc_t4"]["teacher"]
    ignored = result["arms"]["ordinary_t4"]["legacy_baseline_reference_ignored"]
    assert ignored == {"exists": False, "path": None, "sha256": None,
                       "reason": "teacher_sha_mismatch_not_used",
                       "policy": "metric_export_copy_only_not_used_for_training_selection_or_gate"}
    cfg = yaml.safe_load((ordinary / "resolved_config.yaml").read_text())
    cfg["require_baseline_validation"] = True
    (ordinary / "resolved_config.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="explicitly disable incompatible legacy baseline validation"):
        SOURCE.read(ordinary, "ordinary_t4")
    cfg["require_baseline_validation"] = False
    (ordinary / "resolved_config.yaml").write_text(yaml.safe_dump(cfg))
    (ordinary / "metrics_summary.csv").write_text("split,R2_variance_weighted\n")
    with pytest.raises(ValueError, match="test_heldin"):
        SOURCE.read(ordinary, "ordinary_t4")
    # Source receipts cannot be manufactured from a frozen test-only replay.
    (ordinary / "metrics_summary.csv").write_text("split,R2_variance_weighted\ntest_heldin,0.25\n")
    cfg["ckpt_path"] = "/forbidden/test-only.ckpt"
    (ordinary / "resolved_config.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="test-only checkpoint replay"):
        SOURCE.read(ordinary, "ordinary_t4")


def heldout_artifact(tmp_path: Path, arm: str, source_checkpoint: Path, source_gate: Path, receipt: Path, protocol: Path, teacher_sha: str, score_bias: float) -> Path:
    root = tmp_path / f"heldout_{arm}"; root.mkdir()
    artifact_checkpoint = root / "student.ckpt"; artifact_checkpoint.write_bytes(source_checkpoint.read_bytes())
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(config(arm, test=True, checkpoint=source_checkpoint, receipt=receipt)))
    norm = {"feature_group": "t4", "sha256": "normalization-sha", "train_sessions": ["a", "b"]}
    split = {"heldout_evaluated_in_fit": False, "heldout_evaluated_in_test": True, "native_t4_normalization": norm,
             "heldout_query_window_audit": audit()}
    write_json(root / "split_manifest.json", split)
    student_sha = sha(source_checkpoint)
    write_json(root / "checkpoint_manifest.json", {"artifact_checkpoint_path": str(artifact_checkpoint), "artifact_checkpoint_sha256": student_sha,
                                                      "source_checkpoint_sha256": student_sha})
    write_json(root / "teacher_metadata.json", {"teacher_checkpoint_sha256": teacher_sha})
    (root / "metrics_per_session.csv").write_text("split,session,R2_variance_weighted\n" + "".join(f"test_heldout,{s},{0.1 + score_bias + i * 0.001}\n" for i, s in enumerate(SESSIONS)))
    gate = json.loads(source_gate.read_text()); source_arm = gate["arms"][arm]
    provenance = {"arm": arm, "protocol_receipt": {"sha256": sha(protocol)}, "source_gate": {"sha256": sha(source_gate)},
                  "clean_teacher_receipt": {"path": str(receipt), "sha256": sha(receipt), "selected_teacher_checkpoint_sha256": teacher_sha},
                  "frozen_source_student_checkpoint": source_arm["checkpoint"], "source_student_selection": source_arm["checkpoint_selection"],
                  "native_t4_normalization": {"sha256": "normalization-sha", "train_sessions": ["a", "b"]},
                  "six_heldout_calibration_nwbs": [{"session": s, "sha256": f"nwb-{i}", "size_bytes": 1} for i, s in enumerate(SESSIONS)],
                  "support_contract": {"trial_range": [0, 24], "chronological": True}, "query_contract": {"per_session_audit": audit()}}
    write_json(root / "heldout_ssc_t4_provenance.json", provenance)
    return root


def test_heldout_aggregate_requires_provenance_and_keeps_clean_reference_out_of_primary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    receipt = tmp_path / "teacher_receipt.json"; write_json(receipt, {"selected_checkpoint": {"sha256": "teacher-sha"}})
    ordinary_source, ordinary_ckpt = source_artifact(tmp_path, "ordinary_t4", receipt, "teacher-sha")
    ssc_source, ssc_ckpt = source_artifact(tmp_path, "ssc_t4", receipt, "teacher-sha")
    protocol = tmp_path / "protocol.json"; protocol.write_bytes(b"protocol")
    gate = tmp_path / "source_gate.json"
    monkeypatch.setattr(SOURCE, "PROTOCOL_RECEIPT_SHA256", sha(protocol))
    monkeypatch.setattr(sys, "argv", ["source", "--ordinary", str(ordinary_source), "--ssc", str(ssc_source), "--protocol-receipt", str(protocol), "--out", str(gate)])
    SOURCE.main()
    ordinary = heldout_artifact(tmp_path, "ordinary_t4", ordinary_ckpt, gate, receipt, protocol, "teacher-sha", 0.0)
    ssc = heldout_artifact(tmp_path, "ssc_t4", ssc_ckpt, gate, receipt, protocol, "teacher-sha", 0.04)
    reference = tmp_path / "reference.json"
    write_json(reference, {"role": "clean_full_spint_third_reference_not_primary", "protocol_receipt": {"sha256": sha(protocol)}, "source_gate": {"sha256": sha(gate)},
                           "clean_teacher_receipt": json.loads((ordinary / "heldout_ssc_t4_provenance.json").read_text())["clean_teacher_receipt"],
                           "descriptor_contract": {"calibration_trials": 24, "query_start_trial": 24, "window_size_bins": 50, "side_feature_group": "none", "calibration_target_labels_used": False, "backward_gradients": False, "optimizer_updates": False, "checkpoint_selection": False},
                           "six_heldout_calibration_nwbs": json.loads((ordinary / "heldout_ssc_t4_provenance.json").read_text())["six_heldout_calibration_nwbs"],
                           "query_window_audit": audit(), "per_session_r2": {s: 0.3 for s in SESSIONS}, "mean_r2_equal_session": 0.3})
    out = tmp_path / "aggregate.json"
    monkeypatch.setattr(HELDOUT, "PROTOCOL_SHA256", sha(protocol))
    monkeypatch.setattr(sys, "argv", ["heldout", "--ordinary", str(ordinary), "--ssc", str(ssc), "--source-gate", str(gate), "--protocol-receipt", str(protocol), "--clean-spint-reference", str(reference), "--out", str(out)])
    HELDOUT.main()
    result = json.loads(out.read_text())
    assert result["primary_effective"] is True
    assert result["clean_spint_reference"]["mean_r2_equal_session"] == 0.3
    assert result["third_reference_contrasts"]["ordinary_t4_minus_clean_spint"]["decision_role"] == "third_reference_only_not_part_of_primary_gate"
    assert result["third_reference_contrasts"]["ssc_t4_minus_clean_spint"]["positive_sessions"] == 0
    bad_reference = json.loads(reference.read_text())
    bad_reference["mean_r2_equal_session"] = 0.31
    write_json(reference, bad_reference)
    with pytest.raises(ValueError, match="equal-session mean"):
        HELDOUT.read_clean_spint_reference(reference, {**json.loads(gate.read_text()), "_path": str(gate)})


def test_runners_explicitly_request_test_heldin_and_gpu_without_opening_data():
    source = (SCRIPTS / "run_m2_ssc_t4_clean_teacher_one_arm.sh").read_text()
    heldout = (SCRIPTS / "run_m2_ssc_t4_heldout_one_arm.sh").read_text()
    driver = (SCRIPTS / "continue_m2_ssc_heldout_after_source.sh").read_text()
    post_receipt = (SCRIPTS / "continue_m2_ssc_after_receipt.sh").read_text()
    assert '"test=true"' in source
    assert '"data.include_heldout_in_test=false"' in source
    assert '"trainer.accelerator=gpu" "trainer.devices=1"' in heldout
    assert '"train=false" "test=true"' in heldout
    assert '"optimized_metric=null"' in heldout
    assert "source gate forbids opening held-out" in driver
    assert "--arm ordinary --gpu 0" in driver and "--arm ssc --gpu 1" in driver
    assert "write_m2_ssc_t4_heldout_provenance.py" in driver
    assert "eval_clean_teacher_m2_m24_heldout.py" in driver
    assert "validate_m2_clean_teacher_receipt.py" in post_receipt
    assert "finalize_m2_clean_teacher_receipt.py" not in post_receipt
    assert "source arm failure: refusing source gate and held-out" in post_receipt
    assert "continue_m2_ssc_heldout_after_source.sh" in post_receipt
    assert "m2_ssc_t4_v1_ordinary_f1_s42_*" in post_receipt
    assert "m2_ssc_t4_v1_ssc_f1_s42_*" in post_receipt


def test_only_clean_receipt_source_experiments_disable_legacy_baseline_validation():
    config_dir = ROOT / "streaming_calibration_exp/configs/experiment"
    ordinary_clean = yaml.safe_load((config_dir / "b3s_t4_m2_m24_clean_teacher_loso_internal.yaml").read_text())
    ssc_clean = yaml.safe_load((config_dir / "ssc_t4_m2_m24_clean_teacher_loso_internal.yaml").read_text())
    default_train = yaml.safe_load((ROOT / "streaming_calibration_exp/configs/train.yaml").read_text())
    assert ordinary_clean["require_baseline_validation"] is False
    assert ssc_clean["require_baseline_validation"] is False
    assert default_train["require_baseline_validation"] is True


def test_clean_spint_replication_source_is_heldin_only_and_receipt_bound():
    runner = (SCRIPTS / "run_m2_t4_clean_spint_replication_source.sh").read_text()
    validator = (SCRIPTS / "validate_m2_t4_clean_spint_source.py").read_text()
    assert 'PROTOCOL_SHA=c723e8f8e5ca37dd9c24d27eb117d7107bfd9d42a7de392e9224a1c21a9ec3cb' in runner
    assert 'validate_m2_clean_teacher_receipt.py' in runner
    assert '"test=true" "data.include_heldout_in_fit=false" "data.include_heldout_in_test=false"' in runner
    assert '"require_baseline_validation=false"' in runner
    assert 'test_heldin' in validator
    assert "heldout_opened':False" in validator
    assert "selected_by_metric')=='val_heldin/r2_mean'" in validator
    assert 'heldin_performance_not_used_as_gate' in validator
    assert 'pre_heldout_addendum' in validator


def test_clean_spint_replication_heldout_rejects_cross_seed_teacher_receipt_mixups():
    runner = (SCRIPTS / "run_m2_t4_clean_spint_replication_heldout.sh").read_text()
    assert "provided teacher/receipt bytes differ from source receipt" in runner
    assert "provided teacher receipt path differs from source receipt" in runner
    assert "st.get('checkpoint_sha256')!=h(teacher)" in runner
    assert "sr.get('sha256')!=h(receipt)" in runner


def test_clean_replication_primary_reference_matches_t4_provenance_end_to_end(tmp_path: Path):
    source = tmp_path / "source.json"; source.write_text("{}")
    nwb = [{"session": s, "sha256": f"n-{i}", "size_bytes": 1} for i, s in enumerate(sorted(REPLICATION.S))]
    query = audit()
    teacher = {"path": "/teacher.json", "sha256": "receipt-sha", "selected_teacher_checkpoint_sha256": "teacher-sha"}
    provenance = {"clean_teacher_receipt": teacher, "six_heldout_calibration_nwbs": nwb,
                  "query_contract": {"per_session_audit": query}}
    clean = {"role": "clean_full_spint_matched_primary_baseline",
             "protocol_receipt": {"sha256": REPLICATION.PROTO_SHA},
             "source_gate": {"path": str(source.resolve()), "sha256": sha(source)},
             "clean_teacher_receipt": teacher,
             "descriptor_contract": {"calibration_trials": 24, "query_start_trial": 24, "window_size_bins": 50,
                                     "side_feature_group": "none", "calibration_target_labels_used": False,
                                     "backward_gradients": False, "optimizer_updates": False, "checkpoint_selection": False},
             "six_heldout_calibration_nwbs": nwb, "query_window_audit": query,
             "per_session_r2": {s: 0.2 for s in REPLICATION.S}, "mean_r2_equal_session": 0.2,
             "formal_heldout_evaluated": False, "hidden_evalai_evaluated": False}
    assert REPLICATION.check_clean(clean, provenance, None, source, 43) == {s: 0.2 for s in REPLICATION.S}
    clean["clean_teacher_receipt"] = {**teacher, "teacher_checkpoint_sha256": "teacher-sha"}
    with pytest.raises(ValueError, match="T4/clean teacher mismatch"):
        REPLICATION.check_clean(clean, provenance, None, source, 43)


def test_clean_teacher_completion_validator_accepts_only_actual_raw_terminal_record(tmp_path: Path):
    log = tmp_path / "teacher.log"
    marker = COMPLETION.MARKER
    log.write_bytes(b"prefix\rEpoch 34: 100% progress]" + marker + b"\n")
    COMPLETION.validate_completion_log(log)
    log.write_bytes(b"\rEpoch 34: 100% progress]Trainer.fit stopped: `max_epochs=35` reached.\n")
    with pytest.raises(ValueError, match="exactly once"):
        COMPLETION.validate_completion_log(log)
    log.write_bytes(b"\rEpoch 34: 100% progress]`Trainer.fit` stopped: `max_epochs=34` reached.\n")
    with pytest.raises(ValueError, match="exactly once"):
        COMPLETION.validate_completion_log(log)
    log.write_bytes(b"\rEpoch 34: 100% progress]" + marker + b"\n" + b"\rEpoch 34: 100% progress]" + marker + b"\n")
    with pytest.raises(ValueError, match="exactly once"):
        COMPLETION.validate_completion_log(log)
