#!/usr/bin/env python3
"""Fail-closed aggregate for M1 clean-selection report-window evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


GROUPS = ("f0", "t4", "ts4")
CELLS = ("fold1_seed42", "fold1_seed43", "fold2_seed42")
CELL_META = {"fold1_seed42": (1, 42), "fold1_seed43": (1, 43), "fold2_seed42": (2, 42)}
SESSION_BY_FOLD = {1: "ses-20120926", 2: "ses-20120927"}
SOURCES = ("clean_best", "fixed_last", "frozen_best")
SCREEN = "m1_clean_selection_v1"
REPORT_START = 210
M = 10


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


def find_eval_artifact(sce: Path, source: str, group: str, fold: int, seed: int) -> Path:
    run_id = f"{SCREEN}_report_{source}_{group}_m1"
    matches = sorted(sce.glob(f"outputs/streaming_calibration/{run_id}_f{fold}_s{seed}_*"))
    if len(matches) != 1:
        raise ValueError(f"expected one report eval artifact for {source}/{group}/f{fold}/s{seed}, found {len(matches)}")
    return matches[0]


def load_cell(protocol: dict, sce: Path, source: str, group: str, cell: str) -> dict[str, Any]:
    fold, seed = CELL_META[cell]
    expected_session = SESSION_BY_FOLD[fold]
    path = find_eval_artifact(sce, source, group, fold, seed)
    cfg = yaml.safe_load((path / "resolved_config.yaml").read_text(encoding="utf-8"))
    data = cfg["data"]
    eq(data.get("heldin_query_start_trial"), REPORT_START, f"{path} heldin_query_start_trial")
    eq(data.get("heldin_query_end_trial"), None, f"{path} heldin_query_end_trial")
    eq(cfg.get("train"), False, f"{path} train flag")
    eq(cfg.get("test"), True, f"{path} test flag")
    split = json.loads((path / "split_manifest.json").read_text(encoding="utf-8"))
    audit = split.get("heldin_query_window_audit", {}).get(expected_session, {})
    ckpt_manifest = json.loads((path / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    ckpt_path = Path(ckpt_manifest["artifact_checkpoint_path"])
    scores = read_scores(path, expected_session)
    frozen = protocol["frozen_source_arms"][group][cell]
    if source == "frozen_best":
        expected_ckpt = Path(frozen["frozen_checkpoint"]["path"])
        expected_sha = frozen["frozen_checkpoint"]["sha256"]
    else:
        train_id = f"{SCREEN}_{group}_m1"
        train_matches = sorted(sce.glob(f"outputs/streaming_calibration/{train_id}_f{fold}_s{seed}_*"))
        if len(train_matches) != 1:
            raise ValueError(f"missing training artifact for {group}/{cell}")
        train_art = train_matches[0]
        ckpt_name = "best.ckpt" if source == "clean_best" else "last.ckpt"
        expected_ckpt = train_art / "checkpoints" / ckpt_name
        expected_sha = sha256(expected_ckpt)
    eq(sha256(ckpt_path), expected_sha, f"{path} checkpoint SHA")
    return {
        "path": str(path),
        "scores": scores,
        "session": expected_session,
        "query_trials": audit.get("query_trials"),
        "eligible_windows": audit.get("eligible_windows"),
        "checkpoint_sha256": expected_sha,
        "checkpoint_path": str(ckpt_path),
        "split": split,
    }


def aggregate_arm(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "equal_cell_mean_r2": sum(cells[c]["scores"][cells[c]["session"]] for c in CELLS) / len(CELLS),
        "per_cell_r2": {c: cells[c]["scores"][cells[c]["session"]] for c in CELLS},
        "per_cell_left_out_session": {c: cells[c]["session"] for c in CELLS},
        "per_cell_query_trials": {c: cells[c]["query_trials"] for c in CELLS},
        "per_cell_eligible_windows": {c: cells[c]["eligible_windows"] for c in CELLS},
        "per_cell_checkpoint_sha256": {c: cells[c]["checkpoint_sha256"] for c in CELLS},
        "per_cell_artifact_path": {c: cells[c]["path"] for c in CELLS},
    }


def report_delta(left: dict, right: dict) -> dict:
    per_cell = {
        cell: left[cell]["scores"][left[cell]["session"]] - right[cell]["scores"][right[cell]["session"]]
        for cell in CELLS
    }
    return {
        "equal_cell_mean_delta_r2": sum(per_cell.values()) / len(per_cell),
        "per_cell_delta_r2": per_cell,
        "paired_note": "Three cells cover two distinct left-out sessions (fold1 x2, fold2 x1); descriptive only.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    sce = root / "streaming_calibration_exp"
    out_dir = args.out or root / f"sua_exploration/results/{SCREEN}"
    out_dir = out_dir.resolve()
    protocol_path = out_dir / "protocol_receipt.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    by_source: dict[str, dict[str, dict[str, Any]]] = {}
    for source in SOURCES:
        by_source[source] = {}
        for group in GROUPS:
            by_source[source][group] = {}
            for cell in CELLS:
                by_source[source][group][cell] = load_cell(protocol, sce, source, group, cell)

    arms = {
        group: {
            source: aggregate_arm(by_source[source][group])
            for source in SOURCES
        }
        for group in GROUPS
    }

    paired = {
        "T4_minus_F0": {
            source: report_delta(by_source[source]["t4"], by_source[source]["f0"])
            for source in SOURCES
        },
        "T4_minus_TS4": {
            source: report_delta(by_source[source]["t4"], by_source[source]["ts4"])
            for source in SOURCES
        },
    }

    payload = {
        "schema_version": 1,
        "purpose": "M1_internal_LOSO_clean_selection_sealed_report_aggregate",
        "name": "M1 clean-selection sealed report-window development evidence",
        "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
        "report_window": {"start_trial": REPORT_START, "end_trial": None},
        "checkpoint_sources": list(SOURCES),
        "distinct_left_out_session_count": 2,
        "distinct_left_out_sessions": sorted(set(SESSION_BY_FOLD.values())),
        "arms": arms,
        "paired_deltas_r2": paired,
        "limitations": protocol.get("limitations", {}),
        "no_pass_fail_gate": True,
    }
    out_path = out_dir / "aggregate_report.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_path)
    print(sha256(out_path))


if __name__ == "__main__":
    main()
