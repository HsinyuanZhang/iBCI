#!/usr/bin/env python3
"""Fail-closed aggregate for the M1 held-in-calib post-support replay correction."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


GROUPS = ("f0", "t4", "ts4")
CELLS = ("fold1_seed42", "fold1_seed43", "fold2_seed42")
CELL_META = {"fold1_seed42": (1, 42), "fold1_seed43": (1, 43), "fold2_seed42": (2, 42)}
SESSION_BY_FOLD = {1: "ses-20120926", 2: "ses-20120927"}
M, WINDOW = 10, 100
SCREEN = "m1_heldin_disjoint_replay_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def eq(observed: object, expected: object, name: str) -> None:
    if observed != expected:
        raise ValueError(f"{name}: expected {expected!r}, found {observed!r}")


def read_scores(path: Path, expected_session: str) -> dict[str, float]:
    with (path / "metrics_per_session.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "test_heldin"]
    if {row.get("session") for row in rows} != {expected_session}:
        raise ValueError(f"{path}: metrics do not contain exactly the left-out session")
    out = {}
    for row in rows:
        eq(int(float(row["M"])), M, f"{path}/{row['session']} M")
        score = float(row["R2_variance_weighted"])
        if not math.isfinite(score):
            raise ValueError(f"{path}/{row['session']}: nonfinite R2")
        out[row["session"]] = score
    return out


def load_arm(protocol: dict, seal_list: dict, group: str, cell: str) -> dict:
    fold, seed = CELL_META[cell]
    expected_session = SESSION_BY_FOLD[fold]
    sealed = seal_list["sealed_artifacts"][group][cell]
    path = Path(sealed["path"]).resolve()
    if not path.is_dir():
        raise ValueError(f"{group}/{cell}: explicitly sealed artifact directory is missing")
    prov_path = path / "m1_heldin_disjoint_replay_correction_provenance.json"
    split_path = path / "split_manifest.json"
    if not prov_path.is_file() or not split_path.is_file():
        raise ValueError(f"{path}: missing correction provenance or split audit")
    provenance = json.loads(prov_path.read_text(encoding="utf-8"))
    eq(sha256(prov_path), sealed["provenance_sha256"], f"{group}/{cell} explicit seal SHA")
    eq(provenance.get("protocol", {}).get("sha256"), seal_list.get("protocol_sha256"), f"{group}/{cell} protocol SHA")
    eq(provenance.get("group"), group, f"{group}/{cell} provenance group")
    eq(provenance.get("cell", {}).get("name"), cell, f"{group}/{cell} provenance cell")
    eq(provenance.get("test_only"), True, f"{group}/{cell} test-only")
    eq(provenance.get("no_backward_optimizer_or_checkpoint_selection_on_heldin_query"), True, f"{group}/{cell} no-selection")
    frozen = protocol["frozen_source_arms"][group][cell]["checkpoint"]
    eq(provenance.get("frozen_source_checkpoint"), frozen, f"{group}/{cell} checkpoint binding")
    audit = provenance.get("query_window_audit", {})
    if set(audit) != {expected_session}:
        raise ValueError(f"{group}/{cell}: audit does not cover the left-out session")
    row = audit[expected_session]
    eq(row.get("support_trials"), M, f"{group}/{cell}/{expected_session} support")
    eq(row.get("query_start_trial"), M, f"{group}/{cell}/{expected_session} query start")
    eq(row.get("window_size"), WINDOW, f"{group}/{cell}/{expected_session} window")
    eq(row.get("full_window_disjoint"), True, f"{group}/{cell}/{expected_session} disjoint")
    if row.get("minimum_window_start_padded_bin") != row.get("raw_query_start_bin") + WINDOW - 1:
        raise ValueError(f"{group}/{cell}/{expected_session}: query window includes support history")
    scores = read_scores(path, expected_session)
    eq(provenance.get("scores"), scores, f"{group}/{cell} score receipt")
    return {
        "path": str(path),
        "scores": scores,
        "session": expected_session,
        "query_trials": row.get("query_trials"),
        "eligible_windows": row.get("eligible_windows"),
        "checkpoint_sha256": frozen["sha256"],
        "split": json.loads(split_path.read_text(encoding="utf-8")),
    }


def report_delta(left: dict, right: dict) -> dict:
    per_cell = {
        cell: left[cell]["scores"][left[cell]["session"]] - right[cell]["scores"][right[cell]["session"]]
        for cell in CELLS
    }
    per_cell_mean = {cell: float(per_cell[cell]) for cell in CELLS}
    return {
        "per_cell_delta_r2": per_cell,
        "equal_cell_mean_delta_r2": float(np.mean(list(per_cell_mean.values()))),
        "paired_note": "Three cells cover two distinct left-out sessions (fold1 x2, fold2 x1); this is descriptive only.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--seal-list", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = root / f"sua_exploration/results/{SCREEN}/protocol_receipt.json"
    protocol_sha = sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    seal_list_path = args.seal_list.resolve()
    seal_list = json.loads(seal_list_path.read_text(encoding="utf-8"))
    eq(seal_list.get("protocol_sha256"), protocol_sha, "seal list protocol SHA")
    if set(seal_list.get("sealed_artifacts", {})) != set(GROUPS):
        raise ValueError("seal list must name exactly F0/T4/TS4")
    for group in GROUPS:
        if set(seal_list["sealed_artifacts"].get(group, {})) != set(CELLS):
            raise ValueError(f"seal list must name exactly three cells for {group}")
    arms = {group: {cell: load_arm(protocol, seal_list, group, cell) for cell in CELLS} for group in GROUPS}
    comparisons = {
        "T4_minus_F0": report_delta(arms["t4"], arms["f0"]),
        "T4_minus_TS4": report_delta(arms["t4"], arms["ts4"]),
    }
    arm_summary = {
        group: {
            "per_cell_r2": {cell: float(list(arms[group][cell]["scores"].values())[0]) for cell in CELLS},
            "per_cell_query_trials": {cell: arms[group][cell]["query_trials"] for cell in CELLS},
            "per_cell_eligible_windows": {cell: arms[group][cell]["eligible_windows"] for cell in CELLS},
            "per_cell_left_out_session": {cell: arms[group][cell]["session"] for cell in CELLS},
            "per_cell_checkpoint_sha256": {cell: arms[group][cell]["checkpoint_sha256"] for cell in CELLS},
        }
        for group in GROUPS
    }
    for summary in arm_summary.values():
        summary["equal_cell_mean_r2"] = float(np.mean(list(summary["per_cell_r2"].values())))
    payload = {
        "schema_version": 1,
        "purpose": "M1_internal_LOSO_heldin_calib_post_support_contamination_correction_aggregate",
        "name": "M1 held-in-calib post-support internal-LOSO development evidence",
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha},
        "explicit_seal_list": {"path": str(seal_list_path), "sha256": sha256(seal_list_path)},
        "historical_contaminated_scores_not_used": True,
        "distinct_left_out_sessions": sorted(set(SESSION_BY_FOLD.values())),
        "distinct_left_out_session_count": 2,
        "aggregation": "single left-out session per cell, then equal mean across the three cells",
        "arms": arm_summary,
        "paired_deltas_r2": comparisons,
        "limitations": {
            **protocol.get("limitations", {}),
            "session_level_n_is_two_not_four": True,
            "no_pass_fail_gate": "This aggregate is descriptive development evidence only.",
        },
        "scope": "test-only inference replay over held-in-calib post-support trials; not hidden EvalAI test.",
    }
    out = args.out or root / f"sua_exploration/results/{SCREEN}/aggregate_heldin.json"
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {out}")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest_path = out.with_suffix(".sha256")
    digest_path.write_text(f"{sha256(out)}  {out.name}\n", encoding="utf-8")
    print(out)
    print(sha256(out))


if __name__ == "__main__":
    main()
