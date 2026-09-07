#!/usr/bin/env python3
"""Freeze the source/checkpoint contract for the M2 M33 replay correction.

The historical M33 local-heldout aggregate is provenance only: it is never an
input score table for this correction.  This receipt binds the nine frozen
source checkpoints and their original held-in source configurations before any
new test-only run is launched.
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
HELDOUT = [
    "ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1", "ses-2020-11-24-Run1", "ses-2020-11-24-Run2",
]
INELIGIBLE = {"ses-2020-11-24-Run1", "ses-2020-11-24-Run2"}
M, WINDOW = 33, 50


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
    legacy = root / "sua_exploration/results/native_mua_heldout_t4_v1/aggregate_heldout.json"
    out = args.out or root / "sua_exploration/results/m2_m33_disjoint_replay_correction_v1/protocol_receipt.json"
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite immutable protocol receipt: {out}")
    record = json.loads(legacy.read_text(encoding="utf-8"))
    task = record.get("tasks", {}).get("m2", {})
    require(task.get("support_policy"), "chronological_first_33", "historical M33 support policy")
    require(task.get("heldout_calibration_sessions"), HELDOUT, "historical held-out ordering")
    legacy_arms = task.get("artifact_manifest", {})
    require(set(legacy_arms), set(GROUPS), "historical arms")

    frozen: dict[str, dict[str, Any]] = {}
    for group, (variant, side) in GROUPS.items():
        require(set(legacy_arms[group]), set(CELLS), f"{group} historical cells")
        frozen[group] = {}
        for cell, (fold, seed) in CELLS.items():
            legacy_artifact = Path(legacy_arms[group][cell]).resolve()
            needed = ["resolved_config.yaml", "split_manifest.json", "checkpoint_manifest.json"]
            if missing := [name for name in needed if not (legacy_artifact / name).is_file()]:
                raise ValueError(f"{legacy_artifact}: missing legacy artifact files {missing}")
            legacy_cfg = yaml.safe_load((legacy_artifact / "resolved_config.yaml").read_text(encoding="utf-8"))
            legacy_data = legacy_cfg["data"]
            require(legacy_cfg.get("train"), False, f"{group}/{cell} historical train")
            require(legacy_cfg.get("test"), True, f"{group}/{cell} historical test")
            require(legacy_data.get("include_heldout_in_fit"), False, f"{group}/{cell} historical heldout fit")
            require(legacy_data.get("include_heldout_in_test"), True, f"{group}/{cell} historical heldout test")
            # Missing is the historical zero/default; it is itself the defect
            # this new replay corrects, so record rather than repair it here.
            require(legacy_data.get("query_start_trial", 0), 0, f"{group}/{cell} historical query start")
            legacy_ckpt = json.loads((legacy_artifact / "checkpoint_manifest.json").read_text(encoding="utf-8"))
            checkpoint = Path(legacy_ckpt.get("source_checkpoint_path", "")).resolve()
            expected_sha = legacy_ckpt.get("source_checkpoint_sha256")
            if not checkpoint.is_file() or sha256(checkpoint) != expected_sha:
                raise ValueError(f"{group}/{cell}: historical frozen checkpoint does not verify")
            source_artifact = checkpoint.parent.parent
            source_needed = ["resolved_config.yaml", "split_manifest.json", "checkpoint_manifest.json"]
            if missing := [name for name in source_needed if not (source_artifact / name).is_file()]:
                raise ValueError(f"{group}/{cell}: source artifact missing {missing}")
            source_cfg = yaml.safe_load((source_artifact / "resolved_config.yaml").read_text(encoding="utf-8"))
            source_data, source_model = source_cfg["data"], source_cfg["model"]
            require((source_data.get("task"), source_data.get("loso_fold"), source_cfg.get("seed")), ("m2", fold, seed), f"{group}/{cell} source cell")
            require((source_model.get("variant"), source_data.get("side_feature_group")), (variant, side), f"{group}/{cell} source arm")
            require(source_data.get("calibration_n_trials"), M, f"{group}/{cell} source M")
            require(source_data.get("random_calibration"), False, f"{group}/{cell} source chronological support")
            require(source_data.get("include_heldout_in_fit"), False, f"{group}/{cell} source heldout fit")
            require(source_data.get("include_heldout_in_test"), False, f"{group}/{cell} source heldout test")
            frozen[group][cell] = {
                "legacy_replay_artifact": str(legacy_artifact),
                "legacy_replay_artifact_sha256": {
                    name: sha256(legacy_artifact / name) for name in needed
                },
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
            }

    payload = {
        "schema_version": 1,
        "purpose": "M2_M33_local_heldout_support_query_contamination_correction",
        "status": "prelaunch_immutable_protocol",
        "historical_aggregate_provenance_only": {"path": str(legacy), "sha256": sha256(legacy)},
        "historical_result_withdrawn": {
            "reason": "historical test-only replay scored from query_start_trial=0, so every score could include chronological support bins",
            "forbidden_operations": ["overwrite historical artifact", "reuse historical R2", "delete historical CSV rows and reaggregate"],
        },
        "cell_order": list(CELLS),
        "cells": {cell: {"fold": fold, "seed": seed} for cell, (fold, seed) in CELLS.items()},
        "groups": list(GROUPS),
        "frozen_source_arms": frozen,
        "runtime_contract": {
            "train": False,
            "test": True,
            "trainer": {"accelerator": "cpu", "devices": 1},
            "calibration_n_trials": M,
            "support_trials": "[0:33] chronological",
            "query_start_trial": M,
            "window_size": WINDOW,
            "full_history_disjoint": True,
            "minimum_window_start_padded_bin": "raw_query_start_bin + 49",
            "allow_empty_heldout_query": True,
            "permitted_runtime_data_differences": [
                "include_heldout_in_test=false->true", "query_start_trial=0_or_missing->33",
                "allow_empty_heldout_query=missing_or_false->true",
            ],
            "permitted_runtime_non_data_differences": ["run_id", "train", "test", "ckpt_path", "optimized_metric", "require_baseline_validation", "trainer accelerator/devices"],
            "forbidden": ["training", "backward pass", "optimizer", "checkpoint selection", "GPU execution"],
        },
        "session_eligibility": {
            "all_historical_heldout_sessions": HELDOUT,
            "ineligible_zero_query_sessions": sorted(INELIGIBLE),
            "eligible_sessions": [session for session in HELDOUT if session not in INELIGIBLE],
            "rule": "A session with total_trials <= 33 has zero query trials and receives no R2; it remains in the audit with an explicit ineligible reason.",
        },
        "aggregation_contract": {
            "name": "M33-eligible four-session subset",
            "within_cell": "equal mean over the four eligible sessions",
            "across_cells": "equal mean over the three cells",
            "comparisons": ["T4_minus_F0", "T4_minus_TS4"],
            "cluster_uncertainty": "four-session cluster bootstrap/sign test; no six-session gate",
            "exact_wilcoxon_disclosure": "For n=4 nonzero paired session clusters, the smallest attainable two-sided exact Wilcoxon p is 0.125; this correction cannot pass the withdrawn six-session p<=0.05 gate.",
        },
        "label_information": {
            "F0": "no calibration target label in identity feature",
            "T4": "first-33 chronological trial target-direction labels",
            "TS4": "same first-33 target-direction labels with deterministic side-feature row shuffle",
            "parity_requirement": "T4 and TS4 must share fitted train-session normalization moments and use the same held-out target-label trial alignment; only attachment/shuffle differs.",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    print(sha256(out))


if __name__ == "__main__":
    main()
