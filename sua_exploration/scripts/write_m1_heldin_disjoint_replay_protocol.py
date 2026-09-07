#!/usr/bin/env python3
"""Freeze the source/checkpoint contract for the M1 held-in replay correction.

The historical internal-LOSO aggregate is provenance only: it is never an input
score table for this correction.  This receipt binds the nine frozen source
checkpoints and their original held-in source configurations before any new
test-only run is launched.
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
M, WINDOW = 10, 100
AGGREGATE_SHA = "aee9297409ec3accae00819b78c06636c7ca0c0d114667e6560db124dded35c2"
SESSION_BY_FOLD = {1: "ses-20120926", 2: "ses-20120927"}


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
    out = args.out or root / "sua_exploration/results/m1_heldin_disjoint_replay_v1/protocol_receipt.json"
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite immutable protocol receipt: {out}")
    require(sha256(aggregate), AGGREGATE_SHA, "aggregate_m1 SHA")
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
            require((source_data.get("task"), source_data.get("loso_fold"), source_cfg.get("seed")), ("m1", fold, seed), f"{group}/{cell} source cell")
            require((source_model.get("variant"), source_data.get("side_feature_group")), (variant, side), f"{group}/{cell} source arm")
            require(source_data.get("calibration_n_trials"), M, f"{group}/{cell} source M")
            require(source_data.get("random_calibration"), False, f"{group}/{cell} source chronological support")
            require(source_data.get("include_heldout_in_fit"), False, f"{group}/{cell} source heldout fit")
            require(source_data.get("include_heldout_in_test"), False, f"{group}/{cell} source heldout test")
            require(source_data.get("heldin_query_start_trial", 0), 0, f"{group}/{cell} historical heldin query start")
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
                "checkpoint": {"path": str(checkpoint), "sha256": expected_sha},
                "left_out_session": SESSION_BY_FOLD[fold],
            }

    payload = {
        "schema_version": 1,
        "purpose": "M1_internal_LOSO_heldin_calib_post_support_contamination_correction",
        "status": "prelaunch_immutable_protocol",
        "historical_aggregate_provenance_only": {"path": str(aggregate), "sha256": AGGREGATE_SHA},
        "historical_endpoint_withdrawn": {
            "reason": "historical internal-LOSO validation scored the 2-trial held-in-minival file, a bit-exact prefix of calibration support",
            "forbidden_operations": ["overwrite historical artifact", "reuse historical contaminated R2 as corrected evidence"],
        },
        "cell_order": list(CELLS),
        "cells": {cell: {"fold": fold, "seed": seed, "left_out_session": SESSION_BY_FOLD[fold]} for cell, (fold, seed) in CELLS.items()},
        "groups": list(GROUPS),
        "frozen_source_arms": frozen,
        "runtime_contract": {
            "train": False,
            "test": True,
            "calibration_n_trials": M,
            "support_trials": "[0:10] chronological",
            "heldin_query_start_trial": M,
            "window_size": WINDOW,
            "full_history_disjoint": True,
            "minimum_window_start_padded_bin": "raw_query_start_bin + 99",
            "permitted_runtime_data_differences": [
                "heldin_query_start_trial=0_or_missing->10",
            ],
            "permitted_runtime_non_data_differences": ["run_id", "train", "test", "ckpt_path", "optimized_metric", "require_baseline_validation", "trainer accelerator/devices"],
            "forbidden": ["training", "backward pass", "optimizer", "checkpoint selection", "hidden EvalAI query/test NWB"],
        },
        "session_coverage": {
            "distinct_left_out_sessions": sorted(set(SESSION_BY_FOLD.values())),
            "distinct_left_out_session_count": 2,
            "cells_cover_three_folds_not_four_sessions": True,
            "rule": "fold1_seed42 and fold1_seed43 both leave out ses-20120926; fold2_seed42 leaves out ses-20120927",
        },
        "aggregation_contract": {
            "within_cell": "single left-out session per cell",
            "across_cells": "equal mean over the three cells",
            "comparisons": ["T4_minus_F0", "T4_minus_TS4"],
            "no_pass_fail_gate": True,
        },
        "label_information": {
            "F0": "no calibration target label in identity feature",
            "T4": "first-10 chronological trial target-direction labels",
            "TS4": "same first-10 target-direction labels with deterministic side-feature row shuffle",
            "parity_requirement": "T4 and TS4 must share fitted train-session normalization moments",
        },
        "limitations": {
            "checkpoint_selection_contaminated": "best.ckpt was selected by val_heldin/r2_mean on the contaminated 2-trial minival endpoint",
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
