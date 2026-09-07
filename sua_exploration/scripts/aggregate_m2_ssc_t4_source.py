#!/usr/bin/env python3
"""Freeze the SSC source-only catastrophic gate before local held-out replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml


PROTOCOL_RECEIPT_SHA256 = "c769f39703f33e1e1f1f0c02a77d7cc9cd28356e1af2b51ec47fbc214aec24dc"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def heldin_score(path: Path) -> float:
    for row in csv.DictReader((path / "metrics_summary.csv").open(encoding="utf-8")):
        if row.get("split") == "test_heldin":
            return float(row["R2_variance_weighted"])
    raise ValueError(f"{path}: missing test_heldin R2")


def read(path: Path, arm: str) -> dict:
    required = ["resolved_config.yaml", "split_manifest.json", "metrics_summary.csv", "checkpoint_manifest.json", "teacher_metadata.json"]
    if missing := [name for name in required if not (path / name).is_file()]:
        raise ValueError(f"{path}: incomplete source artifact {missing}")
    config = yaml.safe_load((path / "resolved_config.yaml").read_text())
    split = json.loads((path / "split_manifest.json").read_text())
    checkpoint = json.loads((path / "checkpoint_manifest.json").read_text())
    teacher_metadata = json.loads((path / "teacher_metadata.json").read_text())
    data, model = config["data"], config["model"]
    if config.get("train") is not True or config.get("test") is not True:
        raise ValueError(f"{path}: source must be a train-plus-heldin-test artifact")
    if config.get("ckpt_path") not in (None, "", "null"):
        raise ValueError(f"{path}: source must not be a test-only checkpoint replay")
    expected_weight = 0.0 if arm == "ordinary_t4" else 1.0
    if data.get("task") != "m2" or data.get("calibration_n_trials") != 24 or data.get("random_calibration") is not False:
        raise ValueError(f"{path}: not chronological M24")
    if data.get("include_heldout_in_fit") is not False or data.get("include_heldout_in_test") is not False:
        raise ValueError(f"{path}: source opened held-out")
    if model.get("variant") != "B3S" or data.get("side_feature_group") != "t4":
        raise ValueError(f"{path}: not B3S+T4")
    if config.get("no_early_stopping") is not True or config.get("trainer", {}).get("max_epochs") != 12:
        raise ValueError(f"{path}: source must use fixed 12-epoch, no-early-stopping selection")
    # This clean-teacher source uses a separately validated teacher receipt.  Its
    # teacher intentionally differs from the legacy B0 manifest, so the legacy
    # prerequisite must be explicitly disabled rather than silently bypassed.
    if config.get("require_baseline_validation") is not False:
        raise ValueError(f"{path}: clean-teacher source must explicitly disable incompatible legacy baseline validation")
    if model.get("require_clean_teacher_receipt") is not True or not model.get("teacher_receipt_path"):
        raise ValueError(f"{path}: clean teacher receipt was not required")
    teacher_receipt_path = Path(str(model["teacher_receipt_path"])).resolve()
    if not teacher_receipt_path.is_file():
        raise ValueError(f"{path}: clean teacher receipt is missing")
    teacher_receipt = json.loads(teacher_receipt_path.read_text())
    selected_teacher = teacher_receipt.get("selected_checkpoint", {})
    if not isinstance(selected_teacher, dict) or not selected_teacher.get("sha256"):
        raise ValueError(f"{path}: malformed clean teacher receipt")
    if teacher_metadata.get("teacher_checkpoint_sha256") != selected_teacher["sha256"]:
        raise ValueError(f"{path}: student teacher hash does not match its clean receipt")
    if model.get("ssc_t4_prediction_consistency_weight") != expected_weight:
        raise ValueError(f"{path}: unexpected SSC weight")
    if split.get("heldout_evaluated_in_fit") is not False or split.get("heldout_evaluated_in_test") is not False:
        raise ValueError(f"{path}: split receipt contradicts held-in-only source")
    ckpt = Path(checkpoint.get("artifact_checkpoint_path", ""))
    if not ckpt.is_file() or checkpoint.get("artifact_checkpoint_sha256") != sha256(ckpt):
        raise ValueError(f"{path}: selected student checkpoint hash mismatch")
    if checkpoint.get("selected_by_metric") != "val_heldin/r2_mean":
        raise ValueError(f"{path}: student checkpoint was not selected by held-in R2")
    if "heldout" in str(checkpoint.get("selected_by_metric", "")).lower():
        raise ValueError(f"{path}: forbidden held-out checkpoint selector")
    t4_normalization = split.get("native_t4_normalization", {})
    if not isinstance(t4_normalization, dict) or t4_normalization.get("feature_group") != "t4":
        raise ValueError(f"{path}: source lacks T4 normalization receipt")
    if not t4_normalization.get("sha256") or not isinstance(t4_normalization.get("train_sessions"), list):
        raise ValueError(f"{path}: source T4 normalization receipt is incomplete")
    # train.py's metric export mechanically copies baseline_metrics_path even
    # when validation is disabled.  Preserve that file's provenance without
    # treating it as a comparison, input, or selection signal for this source.
    legacy_reference = path / "baseline_reference.csv"
    legacy_baseline_reference_ignored = {
        "exists": legacy_reference.is_file(),
        "path": str(legacy_reference.resolve()) if legacy_reference.is_file() else None,
        "sha256": sha256(legacy_reference) if legacy_reference.is_file() else None,
        "reason": "teacher_sha_mismatch_not_used",
        "policy": "metric_export_copy_only_not_used_for_training_selection_or_gate",
    }
    return {
        "path": str(path.resolve()),
        "heldin_r2": heldin_score(path),
        "checkpoint": {"path": str(ckpt.resolve()), "sha256": sha256(ckpt)},
        "checkpoint_selection": {"selected_by_metric": checkpoint["selected_by_metric"], "no_early_stopping": True, "max_epochs": 12},
        "teacher": {
            "checkpoint_path": str(Path(str(teacher_metadata.get("teacher_checkpoint_path", ""))).resolve()),
            "checkpoint_sha256": selected_teacher["sha256"],
            "receipt_path": str(teacher_receipt_path),
            "receipt_sha256": sha256(teacher_receipt_path),
        },
        "native_t4_normalization": {
            "sha256": t4_normalization["sha256"],
            "train_sessions": list(t4_normalization["train_sessions"]),
        },
        "legacy_baseline_reference_ignored": legacy_baseline_reference_ignored,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary", type=Path, required=True)
    parser.add_argument("--ssc", type=Path, required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if sha256(args.protocol_receipt) != PROTOCOL_RECEIPT_SHA256:
        raise ValueError("held-out protocol receipt SHA drift")
    ordinary, ssc = read(args.ordinary, "ordinary_t4"), read(args.ssc, "ssc_t4")
    if ordinary["teacher"] != ssc["teacher"]:
        raise ValueError("ordinary and SSC source arms do not use the identical clean teacher/receipt")
    if ordinary["native_t4_normalization"] != ssc["native_t4_normalization"]:
        raise ValueError("ordinary and SSC source arms do not use identical T4 normalization")
    delta = ssc["heldin_r2"] - ordinary["heldin_r2"]
    result = {
        "schema_version": 1, "formal_heldout_evaluated": False,
        "protocol_receipt": str(args.protocol_receipt.resolve()),
        "protocol_receipt_sha256": PROTOCOL_RECEIPT_SHA256,
        "arms": {"ordinary_t4": ordinary, "ssc_t4": ssc},
        "ssc_minus_ordinary_heldin_r2": delta,
        "catastrophic_stop": delta < -0.10,
        "next_action": "stop" if delta < -0.10 else "one_frozen_local_heldout_replay_required",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
