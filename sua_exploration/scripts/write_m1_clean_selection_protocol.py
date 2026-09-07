#!/usr/bin/env python3
"""Freeze the M1 clean-selection protocol before any training is launched.

This receipt binds the chronological support/selection/report windows, the nine
matched cells, and the frozen source configuration hashes.  The report window is
sealed until all training and checkpoint selection has completed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


GROUPS = {"f0": ("B3", "none"), "t4": ("B3S", "t4"), "ts4": ("B3S", "ts4")}
CELLS = {
    "fold1_seed42": (1, 42),
    "fold1_seed43": (1, 43),
    "fold2_seed42": (2, 42),
}
SESSION_BY_FOLD = {1: "ses-20120926", 2: "ses-20120927"}
SUPPORT_END = 10
SELECTION_START = 10
SELECTION_END = 210
REPORT_START = 210
WINDOW = 100
M = 10
AGGREGATE_SHA = "aee9297409ec3accae00819b78c06636c7ca0c0d114667e6560db124dded35c2"
AUDIT_SHA = "d448e8015bf99e244e37b9c3fbf0329f49b74c41c2d2c7a78b60482e4b429a1f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(observed: object, expected: object, name: str) -> None:
    if observed != expected:
        raise ValueError(f"{name}: expected {expected!r}, found {observed!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    aggregate = root / "sua_exploration/results/native_mua_t4_v1/aggregate_m1.json"
    audit = root / "sua_exploration/results/m1_endpoint_infeasibility_v2/audit.json"
    out = args.out or root / "sua_exploration/results/m1_clean_selection_v1/protocol_receipt.json"
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite immutable protocol receipt: {out}")
    require(sha256(aggregate), AGGREGATE_SHA, "aggregate_m1 SHA")
    require(sha256(audit), AUDIT_SHA, "endpoint_infeasibility_v2 audit SHA")
    record = json.loads(aggregate.read_text(encoding="utf-8"))
    artifacts = record.get("artifacts", {}).get("m1", {})
    require(set(artifacts), set(GROUPS), "historical arms")

    frozen: dict[str, dict[str, Any]] = {}
    for group, (variant, side) in GROUPS.items():
        require(set(artifacts[group]), set(CELLS), f"{group} historical cells")
        frozen[group] = {}
        for cell, (fold, seed) in CELLS.items():
            source_artifact = Path(artifacts[group][cell]).resolve()
            needed = ["resolved_config.yaml", "split_manifest.json", "checkpoint_manifest.json"]
            if missing := [name for name in needed if not (source_artifact / name).is_file()]:
                raise ValueError(f"{source_artifact}: missing source artifact files {missing}")
            source_cfg = yaml.safe_load((source_artifact / "resolved_config.yaml").read_text(encoding="utf-8"))
            source_data, source_model = source_cfg["data"], source_cfg["model"]
            require(
                (source_data.get("task"), source_data.get("loso_fold"), source_cfg.get("seed")),
                ("m1", fold, seed),
                f"{group}/{cell} source cell",
            )
            require(
                (source_model.get("variant"), source_data.get("side_feature_group")),
                (variant, side),
                f"{group}/{cell} source arm",
            )
            require(source_data.get("calibration_n_trials"), M, f"{group}/{cell} source M")
            require(source_data.get("random_calibration"), False, f"{group}/{cell} source chronological support")
            require(source_data.get("include_heldout_in_fit"), False, f"{group}/{cell} source heldout fit")
            require(source_data.get("include_heldout_in_test"), False, f"{group}/{cell} source heldout test")
            require(source_data.get("heldin_query_start_trial", 0), 0, f"{group}/{cell} historical heldin query start")
            require(source_data.get("heldin_query_end_trial"), None, f"{group}/{cell} historical heldin query end")
            require(source_cfg.get("trainer", {}).get("max_epochs"), 12, f"{group}/{cell} max_epochs")
            legacy_ckpt = json.loads((source_artifact / "checkpoint_manifest.json").read_text(encoding="utf-8"))
            checkpoint = Path(legacy_ckpt.get("artifact_checkpoint_path", "")).resolve()
            expected_sha = legacy_ckpt.get("artifact_checkpoint_sha256")
            if not checkpoint.is_file() or sha256(checkpoint) != expected_sha:
                raise ValueError(f"{group}/{cell}: frozen checkpoint does not verify")
            frozen[group][cell] = {
                "source_artifact": str(source_artifact),
                "source_resolved_config_sha256": sha256(source_artifact / "resolved_config.yaml"),
                "source_split_manifest_sha256": sha256(source_artifact / "split_manifest.json"),
                "source_checkpoint_manifest_sha256": sha256(source_artifact / "checkpoint_manifest.json"),
                "source_data": source_data,
                "source_data_canonical_sha256": canonical_sha(source_data),
                "source_model": source_model,
                "source_model_canonical_sha256": canonical_sha(source_model),
                "source_split_manifest": json.loads((source_artifact / "split_manifest.json").read_text(encoding="utf-8")),
                "frozen_checkpoint": {"path": str(checkpoint), "sha256": expected_sha},
                "left_out_session": SESSION_BY_FOLD[fold],
            }

    payload = {
        "schema_version": 1,
        "purpose": "M1_internal_LOSO_clean_selection_and_sealed_report",
        "status": "prelaunch_immutable_protocol",
        "report_window_sealed": {
            "statement": (
                "The report window trials [210, end) are sealed until all nine training runs "
                "complete and checkpoint selection on the selection window [10, 210) finishes. "
                "No metric, split adjustment, or hyperparameter change may reference report-window "
                "outcomes before that seal is lifted."
            ),
            "sealed_until": "all_training_and_selection_complete",
        },
        "historical_aggregate_provenance_only": {"path": str(aggregate), "sha256": AGGREGATE_SHA},
        "endpoint_audit_provenance": {"path": str(audit), "sha256": AUDIT_SHA},
        "historical_endpoint_defect": {
            "reason": (
                "frozen best.ckpt files were selected by val_heldin/r2_mean on the contaminated "
                "2-trial held-in-minival endpoint inside calibration support"
            ),
            "forbidden_operations": [
                "reuse contaminated minival scores for selection",
                "select checkpoints on report-window trials",
                "open hidden EvalAI query/test NWB",
            ],
        },
        "chronological_windows": {
            "support": {"start_trial": 0, "end_trial": SUPPORT_END, "purpose": "calibration only"},
            "selection": {
                "start_trial": SELECTION_START,
                "end_trial": SELECTION_END,
                "purpose": "validation signal for checkpoint selection during training",
            },
            "report": {
                "start_trial": REPORT_START,
                "end_trial": None,
                "purpose": "final evaluation only; never consulted during training or selection",
            },
            "per_session_report_trials": {
                "ses-20120926": 409 - REPORT_START,
                "ses-20120927": 376 - REPORT_START,
            },
            "per_session_selection_trials": {
                "ses-20120926": SELECTION_END - SELECTION_START,
                "ses-20120927": SELECTION_END - SELECTION_START,
            },
        },
        "cell_order": list(CELLS),
        "cells": {
            cell: {"fold": fold, "seed": seed, "left_out_session": SESSION_BY_FOLD[fold]}
            for cell, (fold, seed) in CELLS.items()
        },
        "groups": list(GROUPS),
        "frozen_source_arms": frozen,
        "training_contract": {
            "train": True,
            "test": False,
            "max_epochs": 12,
            "calibration_n_trials": M,
            "support_trials": f"[0:{SUPPORT_END}) chronological",
            "heldin_query_start_trial": SELECTION_START,
            "heldin_query_end_trial": SELECTION_END,
            "window_size": WINDOW,
            "full_history_disjoint": True,
            "checkpoint_outputs": ["best.ckpt", "last.ckpt"],
            "hyperparameter_source": "frozen_source_arms[*][*].source_resolved_config_sha256",
            "permitted_runtime_data_differences": [
                "heldin_query_start_trial=0->10",
                "heldin_query_end_trial=null->210",
                "validation endpoint switched from held-in-minival to held-in-calib bounded window",
            ],
            "forbidden": [
                "report-window loading during training",
                "hyperparameter tuning after seeing any number",
                "folds 3/4 or additional seeds",
            ],
        },
        "evaluation_contract": {
            "runs_after": "all_training_complete",
            "heldin_query_start_trial": REPORT_START,
            "heldin_query_end_trial": None,
            "checkpoint_sources": [
                "clean_selection_best",
                "fixed_epoch_last",
                "frozen_native_mua_t4_v1_best",
            ],
            "no_pass_fail_gate": True,
        },
        "session_coverage": {
            "distinct_left_out_sessions": sorted(set(SESSION_BY_FOLD.values())),
            "distinct_left_out_session_count": 2,
            "cells_cover_three_folds_not_four_sessions": True,
            "rule": "fold1_seed42 and fold1_seed43 both leave out ses-20120926; fold2_seed42 leaves out ses-20120927",
        },
        "limitations": {
            "session_level_n_is_two_not_four": True,
            "report_window_shorter_for_ses_20120927": True,
            "direction_labels_half_plane": "held-in M1 directions span 8 directions across 157.5 degrees, not a full circle",
            "development_evidence_only": "native FALCON internal-LOSO development evidence, not hidden EvalAI test",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    print(sha256(out))


if __name__ == "__main__":
    main()
