#!/usr/bin/env python3
"""Freeze the M2 held-in post-support precision-calibration protocol.

Stage A replays frozen native_mua_t4_v1 F0/T4 checkpoints on the full
post-support query.  Stage B trains seven LOSO folds for F0 and T4 with a
chronological half-split of the post-support window for checkpoint selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "sua_exploration/results/m2_heldin_postsupport_endpoint_v1/audit.json"
AUDIT_SHA256 = "d6940d156d05cd1cbd43bac95220c1d1db370c50d841de8a51d49a53c9ac70c5"
AGGREGATE = ROOT / "sua_exploration/results/native_mua_t4_v1/aggregate_m2.json"
AGGREGATE_SHA256 = "c6eb1727456040b02f333260f67f47f134cade9e0f556ca912169b0da3fc8613"
OUT = ROOT / "sua_exploration/results/m2_heldin_postsupport_endpoint_v1/protocol_receipt.json"

SUPPORT = 33
WINDOW = 50
CELLS = {
    "fold1_seed42": (1, 42, "ses-2020-10-19-Run2"),
    "fold1_seed43": (1, 43, "ses-2020-10-19-Run2"),
    "fold2_seed42": (2, 42, "ses-2020-10-20-Run1"),
}
FOLDS = {
    0: "ses-2020-10-19-Run1",
    1: "ses-2020-10-19-Run2",
    2: "ses-2020-10-20-Run1",
    3: "ses-2020-10-20-Run2",
    4: "ses-2020-10-27-Run1",
    5: "ses-2020-10-27-Run2",
    6: "ses-2020-10-28-Run1",
}
ARMS = ("f0", "t4")


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


def frozen_checkpoint(group: str, cell: str, artifact_path: Path) -> dict[str, Any]:
    manifest = json.loads((artifact_path / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    ckpt = Path(manifest["artifact_checkpoint_path"])
    return {
        "path": str(ckpt.resolve()),
        "sha256": manifest["artifact_checkpoint_sha256"],
        "source_artifact": str(artifact_path.resolve()),
        "selected_by_metric": manifest.get("selected_by_metric"),
        "selected_metric_value": manifest.get("selected_metric_value"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite immutable protocol receipt: {out}")

    require(sha256(AUDIT), AUDIT_SHA256, "audit SHA")
    require(sha256(AGGREGATE), AGGREGATE_SHA256, "aggregate_m2 SHA")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    aggregate = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    artifacts = aggregate["artifacts"]["m2"]

    frozen_cells: dict[str, dict[str, Any]] = {}
    for cell, (fold, seed, left_out) in CELLS.items():
        frozen_cells[cell] = {}
        for arm in ARMS:
            source = Path(artifacts[arm][cell]).resolve()
            cfg = yaml.safe_load((source / "resolved_config.yaml").read_text(encoding="utf-8"))
            data = cfg["data"]
            frozen_cells[cell][arm] = {
                "frozen_checkpoint": frozen_checkpoint(arm, cell, source),
                "left_out_session": left_out,
                "loso_fold": fold,
                "seed": seed,
                "source_data_canonical_sha256": canonical_sha(data),
            }

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "m2_heldin_postsupport_precision_calibration_f0_t4_only",
        "status": "prelaunch_immutable_protocol",
        "framing": audit["candidate_branch_seal"]["framing"],
        "candidate_branch_seal": audit["candidate_branch_seal"],
        "endpoint_audit_provenance": {
            "path": str(AUDIT.resolve()),
            "sha256": AUDIT_SHA256,
        },
        "historical_aggregate_provenance_only": {
            "path": str(AGGREGATE.resolve()),
            "sha256": AGGREGATE_SHA256,
        },
        "support_window": {
            "start_trial": 0,
            "end_trial": SUPPORT,
            "purpose": "calibration only; identical to frozen native_mua_t4_v1 M2 training",
        },
        "chronological_windows": {
            "support": {
                "start_trial": 0,
                "end_trial": SUPPORT,
                "purpose": "calibration only",
            },
            "stage_a_full_postsupport_query": {
                "start_trial": SUPPORT,
                "end_trial": None,
                "purpose": "frozen-checkpoint inference-only replay on entire post-support window",
            },
            "stage_b_selection": {
                "rule": "per_session_chronological_first_half_of_post_support",
                "start_trial": SUPPORT,
                "end_trial": "33 + floor((n_trials - 33) / 2)",
                "purpose": "validation signal for checkpoint selection during Stage B training",
            },
            "stage_b_report": {
                "rule": "per_session_chronological_second_half_of_post_support",
                "start_trial": "33 + floor((n_trials - 33) / 2)",
                "end_trial": None,
                "purpose": "final evaluation only; never consulted during training or selection",
            },
            "m1_pattern_note": (
                "M1 clean_selection uses fixed trial indices (selection [10,210), report [210,end)). "
                "M2 held-in sessions span 204-339 trials; the shortest session cannot reach trial 210 "
                "as a selection boundary.  Stage B therefore applies the same chronological first/second "
                "half principle per session within the post-support window rather than a shared absolute "
                "trial index."
            ),
        },
        "evaluation_contract": {
            "arms": list(ARMS),
            "aggregation": "equal_session_mean_within_cell_then_equal_cell_mean_across_cells",
            "report_sign_regardless": True,
            "no_pass_fail_gate": True,
            "no_effective_verdict": True,
            "comparators": ["T4_minus_F0"],
            "forbidden_comparators": audit["candidate_branch_seal"]["sealed_branches"],
            "datamodule_requirements": {
                "task": "m2",
                "validation_protocol": "loso",
                "calibration_n_trials": SUPPORT,
                "heldin_query_start_trial": SUPPORT,
                "random_calibration": False,
                "include_heldout_in_fit": False,
                "include_heldout_in_test": False,
                "window_size": WINDOW,
            },
        },
        "stage_a_frozen_checkpoint_replay": {
            "cells": list(CELLS),
            "frozen_cells": frozen_cells,
            "query_window": {
                "start_trial": SUPPORT,
                "end_trial": None,
                "purpose": "frozen-checkpoint inference-only replay on entire post-support window",
            },
            "train": False,
            "test": True,
            "can_establish": [
                "within-session measurement precision on the post-support endpoint",
                "descriptive T4-F0 paired deltas on two distinct left-out sessions",
            ],
            "cannot_establish": [
                "seven-session between-session dispersion",
                "any candidate-branch comparator named in the candidate_branch_seal",
            ],
            "distinct_left_out_sessions": 2,
        },
        "stage_b_seven_fold_training": {
            "folds": FOLDS,
            "arms": list(ARMS),
            "cells_per_arm": 7,
            "seeds": [42],
            "total_training_runs": 14,
            "estimated_wall_clock_hours": audit["power_analysis"]["stage_b_wall_clock_estimate"]["estimated_hours_14_runs"],
            "hyperparameter_source": "frozen native_mua_t4_v1 resolved configs per fold/seed",
            "max_epochs": 12,
            "checkpoint_selection_metric": "val_heldin/r2_mean on stage_b_selection window",
            "can_establish": [
                "seven-session between-session dispersion for F0 and T4",
                "projected MDE calibration on the held-in post-support endpoint",
            ],
            "forbidden": [
                "candidate-branch arms or comparators",
                "held-out-calib scoring",
                "hyperparameter tuning after observing report-window outcomes",
            ],
        },
        "limitations": {
            "development_evidence_only": "native FALCON internal-LOSO development evidence, not hidden EvalAI test",
            "stage_a_checkpoint_selection_contaminated": (
                "frozen best.ckpt files were selected on the contaminated 2-trial held-in-minival endpoint"
            ),
            "directional_trial_note": (
                "roughly half of M2 post-support trials are centre/rest and excluded from the T4 cosine fit; "
                "report both raw and directional trial counts"
            ),
        },
        "explicitly_not_claimed": audit["explicitly_not_claimed"],
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = sha256(out)
    (out.parent / "protocol_receipt.sha256").write_text(f"{digest}  protocol_receipt.json\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
