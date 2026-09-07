#!/usr/bin/env python3
"""Write a non-overwritable provenance receipt for one frozen SSC-T4 replay.

This command intentionally runs *after* a test-only replay has produced its
artifact.  It is the first component allowed to hash the six local
held-out-calibration NWBs; it binds those concrete inputs to the frozen source
gate, clean-teacher receipt, selected student checkpoint, and query audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SHA256 = "c769f39703f33e1e1f1f0c02a77d7cc9cd28356e1af2b51ec47fbc214aec24dc"
EXPECTED_SESSIONS = {
    "ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1", "ses-2020-11-24-Run1", "ses-2020-11-24-Run2",
}
ARM_SSC_WEIGHT = {"ordinary_t4": 0.0, "ssc_t4": 1.0}
M, WINDOW = 24, 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_equal(observed: object, expected: object, what: str) -> None:
    if observed != expected:
        raise ValueError(f"{what}: expected {expected!r}, found {observed!r}")


def expected_heldout_nwbs() -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted((ROOT / "SPINT-main/data/000953").rglob("*held-out-calib*.nwb")):
        session = path.name.split("_")[1].split(".")[0]
        files.append({"session": session, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    if {str(row["session"]) for row in files} != EXPECTED_SESSIONS or len(files) != 6:
        raise ValueError("local held-out input set is not exactly the frozen six M2 sessions")
    return files


def validate_query_audit(audit: object) -> dict[str, dict[str, object]]:
    if not isinstance(audit, dict) or set(audit) != EXPECTED_SESSIONS:
        raise ValueError("expected exactly six held-out query-window audits")
    result: dict[str, dict[str, object]] = {}
    for session, row in audit.items():
        if not isinstance(row, dict):
            raise ValueError(f"{session}: malformed query audit")
        for key, expected in {
            "support_trials": M,
            "query_start_trial": M,
            "window_size": WINDOW,
            "full_window_disjoint": True,
        }.items():
            require_equal(row.get(key), expected, f"{session}.{key}")
        if int(row.get("query_trials", 0)) <= 0 or int(row.get("eligible_windows", 0)) <= 0:
            raise ValueError(f"{session}: empty query after chronological support")
        if row.get("minimum_window_start_padded_bin") != row.get("raw_query_start_bin", -WINDOW) + WINDOW - 1:
            raise ValueError(f"{session}: 50-bin history touches the support boundary")
        result[str(session)] = dict(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARM_SSC_WEIGHT), required=True)
    parser.add_argument("--source-gate", type=Path, required=True)
    parser.add_argument("--teacher-receipt", type=Path, required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    required = ["resolved_config.yaml", "split_manifest.json", "checkpoint_manifest.json", "teacher_metadata.json", "metrics_per_session.csv"]
    if missing := [name for name in required if not (artifact / name).is_file()]:
        raise ValueError(f"{artifact}: missing held-out artifact fields {missing}")
    if sha256(args.protocol_receipt) != PROTOCOL_SHA256:
        raise ValueError("frozen held-out protocol receipt SHA drift")

    source_gate = json.loads(args.source_gate.read_text())
    if source_gate.get("catastrophic_stop") is not False or source_gate.get("next_action") != "one_frozen_local_heldout_replay_required":
        raise ValueError("source gate does not authorize a held-out replay")
    source_arm = source_gate.get("arms", {}).get(args.arm, {})
    source_checkpoint = source_arm.get("checkpoint", {})
    source_teacher = source_arm.get("teacher", {})
    if not isinstance(source_checkpoint, dict) or not isinstance(source_teacher, dict):
        raise ValueError("source gate lacks frozen checkpoint/teacher bindings")
    selection = source_arm.get("checkpoint_selection", {})
    if selection != {"selected_by_metric": "val_heldin/r2_mean", "no_early_stopping": True, "max_epochs": 12}:
        raise ValueError("source gate does not certify held-in-only fixed-budget student selection")

    receipt_path = args.teacher_receipt.resolve()
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    receipt_teacher = receipt.get("selected_checkpoint", {})
    if not isinstance(receipt_teacher, dict) or not receipt_teacher.get("sha256"):
        raise ValueError("clean teacher receipt has no selected checkpoint SHA")
    if source_teacher.get("receipt_sha256") != sha256(receipt_path):
        raise ValueError("source gate was qualified with a different clean teacher receipt")
    if source_teacher.get("checkpoint_sha256") != receipt_teacher["sha256"]:
        raise ValueError("source gate teacher checkpoint does not match clean teacher receipt")

    cfg = yaml.safe_load((artifact / "resolved_config.yaml").read_text())
    data, model = cfg["data"], cfg["model"]
    checks = {
        "data.task": (data.get("task"), "m2"),
        "data.loso_fold": (data.get("loso_fold"), 1),
        "seed": (cfg.get("seed"), 42),
        "data.calibration_n_trials": (data.get("calibration_n_trials"), M),
        "data.random_calibration": (data.get("random_calibration"), False),
        "data.include_heldout_in_fit": (data.get("include_heldout_in_fit"), False),
        "data.include_heldout_in_test": (data.get("include_heldout_in_test"), True),
        "data.query_start_trial": (data.get("query_start_trial"), M),
        "data.side_feature_group": (data.get("side_feature_group"), "t4"),
        "model.variant": (model.get("variant"), "B3S"),
        "model.ssc_t4_prediction_consistency_weight": (model.get("ssc_t4_prediction_consistency_weight"), ARM_SSC_WEIGHT[args.arm]),
        "model.require_clean_teacher_receipt": (model.get("require_clean_teacher_receipt"), True),
        "train": (cfg.get("train"), False),
        "test": (cfg.get("test"), True),
        "ckpt_path": (str(Path(str(cfg.get("ckpt_path"))).resolve()), str(Path(str(source_checkpoint.get("path", ""))).resolve())),
        "teacher_receipt_path": (str(Path(str(model.get("teacher_receipt_path", ""))).resolve()), str(receipt_path)),
    }
    for name, (observed, expected) in checks.items():
        require_equal(observed, expected, name)

    split = json.loads((artifact / "split_manifest.json").read_text())
    require_equal(split.get("heldout_evaluated_in_fit"), False, "heldout_evaluated_in_fit")
    require_equal(split.get("heldout_evaluated_in_test"), True, "heldout_evaluated_in_test")
    audit = validate_query_audit(split.get("heldout_query_window_audit"))
    normalization = split.get("native_t4_normalization", {})
    require_equal(normalization.get("feature_group"), "t4", "T4 normalization group")
    if normalization.get("sha256") != source_arm.get("native_t4_normalization", {}).get("sha256"):
        raise ValueError("test T4 normalization SHA differs from frozen source")
    if normalization.get("train_sessions") != source_arm.get("native_t4_normalization", {}).get("train_sessions"):
        raise ValueError("test T4 normalization train-session set differs from frozen source")

    checkpoint = json.loads((artifact / "checkpoint_manifest.json").read_text())
    actual_checkpoint = Path(str(checkpoint.get("artifact_checkpoint_path", ""))).resolve()
    expected_source = Path(str(source_checkpoint.get("path", ""))).resolve()
    if not actual_checkpoint.is_file() or not expected_source.is_file():
        raise ValueError("student checkpoint artifact/source is missing")
    expected_student_sha = source_checkpoint.get("sha256")
    if sha256(actual_checkpoint) != expected_student_sha or sha256(expected_source) != expected_student_sha:
        raise ValueError("held-out student checkpoint does not byte-match frozen source checkpoint")
    if checkpoint.get("artifact_checkpoint_sha256") != expected_student_sha or checkpoint.get("source_checkpoint_sha256") != expected_student_sha:
        raise ValueError("checkpoint manifest does not bind the frozen source checkpoint")

    teacher_metadata = json.loads((artifact / "teacher_metadata.json").read_text())
    require_equal(teacher_metadata.get("teacher_checkpoint_sha256"), receipt_teacher["sha256"], "test teacher SHA")
    output = artifact / "heldout_ssc_t4_provenance.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    payload = {
        "schema_version": 1,
        "purpose": "M2_M24_SSC_T4_local_heldout_test_only_replay",
        "formal_heldout_evaluated": False,
        "local_heldout_calib_evaluated": True,
        "hidden_evalai_evaluated": False,
        "arm": args.arm,
        "protocol_receipt": {"path": str(args.protocol_receipt.resolve()), "sha256": PROTOCOL_SHA256},
        "source_gate": {"path": str(args.source_gate.resolve()), "sha256": sha256(args.source_gate)},
        "clean_teacher_receipt": {"path": str(receipt_path), "sha256": sha256(receipt_path), "selected_teacher_checkpoint_sha256": receipt_teacher["sha256"]},
        "frozen_source_student_checkpoint": {"path": str(expected_source), "sha256": expected_student_sha},
        "source_student_selection": selection,
        "test_artifact_checkpoint": {"path": str(actual_checkpoint), "sha256": expected_student_sha},
        "native_t4_normalization": {"sha256": normalization["sha256"], "train_sessions": normalization["train_sessions"]},
        "six_heldout_calibration_nwbs": expected_heldout_nwbs(),
        "support_contract": {"trial_range": [0, M], "chronological": True, "activity_labels": "neural only", "t4_labels": "trial-level target-direction metadata only"},
        "query_contract": {"trial_range": [M, None], "window_size_bins": WINDOW, "full_history_after_support_boundary": True, "per_session_audit": audit},
        "no_backward_optimizer_or_checkpoint_selection_on_heldout": True,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
