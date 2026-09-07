#!/usr/bin/env python3
"""Audit common H1 calibration budgets and remaining chronological queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from falcon_challenge.config import FalconTask
from falcon_challenge.dataloaders import load_nwb


BUDGETS = (2, 4, 6, 8)
SCHEMA = "h1_calibration_budget_endpoint_audit_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session_id(path: Path) -> str:
    marker = "_ses-"
    if marker not in path.stem:
        raise ValueError(f"cannot parse session id from {path}")
    return path.stem.split(marker, 1)[1]


def audit(data_dir: Path) -> dict[str, Any]:
    root = data_dir.resolve()
    if "held-out" in str(root).lower():
        raise ValueError("held-out paths are forbidden in the budget audit")
    calib_dir = root / "sub-HumanPitt-held-in-calib"
    files = sorted(calib_dir.glob("*.nwb"))
    if len(files) != 13:
        raise RuntimeError(f"expected 13 H1 held-in calibration files, got {len(files)}")

    rows: list[dict[str, Any]] = []
    for path in files:
        _, _, trial_change, eval_mask = load_nwb(path, FalconTask.h1)
        starts = np.flatnonzero(trial_change)
        sid = session_id(path)
        budget_rows: dict[str, dict[str, Any]] = {}
        for budget in BUDGETS:
            defined = len(starts) >= budget
            has_query = len(starts) > budget
            query_start = int(starts[budget]) if has_query else None
            budget_rows[str(budget)] = {
                "support_defined": bool(defined),
                "post_support_trials": max(int(len(starts)) - budget, 0),
                "post_support_eval_bins": (
                    int(np.asarray(eval_mask[query_start:], dtype=np.int64).sum())
                    if query_start is not None
                    else 0
                ),
            }
        rows.append(
            {
                "session": sid,
                "date": sid[:8],
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "legal_trial_count": int(len(starts)),
                "budgets": budget_rows,
            }
        )

    common = {
        str(budget): {
            "support_defined_recordings": sum(
                row["budgets"][str(budget)]["support_defined"] for row in rows
            ),
            "recordings_with_post_support_query": sum(
                row["budgets"][str(budget)]["post_support_trials"] > 0
                for row in rows
            ),
            "minimum_post_support_trials": min(
                row["budgets"][str(budget)]["post_support_trials"] for row in rows
            ),
        }
        for budget in BUDGETS
    }
    short = [row["session"] for row in rows if row["legal_trial_count"] == 7]
    return {
        "schema": SCHEMA,
        "status": "PASS_READ_ONLY_HELDIN_BUDGET_AUDIT",
        "task": "h1",
        "scope": "public_heldin_calibration_only",
        "formal_heldout_opened": False,
        "budgets_audited": list(BUDGETS),
        "recording_count": len(rows),
        "minimum_legal_trial_count": min(row["legal_trial_count"] for row in rows),
        "seven_trial_recordings": short,
        "common_budget_summary": common,
        "headline_facts": {
            "m8_undefined_recordings": 13 - common["8"]["support_defined_recordings"],
            "m8_recordings_without_future_query": 13
            - common["8"]["recordings_with_post_support_query"],
            "m6_minimum_future_trials": common["6"]["minimum_post_support_trials"],
            "date_19250108_seven_trial_recordings": sum(
                row["date"] == "19250108" and row["legal_trial_count"] == 7
                for row in rows
            ),
            "date_19250108_total_recordings": sum(
                row["date"] == "19250108" for row in rows
            ),
        },
        "interpretation_boundary": {
            "receipt_authorizes_gpu": False,
            "m8_is_common_budget": False,
            "m6_has_uniform_multi_trial_future_query": False,
            "budget_curve_is_descriptive_only": True,
        },
        "recordings": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {output}")
    payload = audit(args.data_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(output, 0o444)
    print(json.dumps({"output": str(output), "status": payload["status"]}))


if __name__ == "__main__":
    main()
