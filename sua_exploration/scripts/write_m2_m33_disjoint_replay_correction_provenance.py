#!/usr/bin/env python3
"""Validate and seal one CPU-only M33 support/query correction replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


GROUPS = {"f0": ("B3", "none"), "t4": ("B3S", "t4"), "ts4": ("B3S", "ts4")}
CELLS = {"fold1_seed42": (1, 42), "fold1_seed43": (1, 43), "fold2_seed42": (2, 42)}
INELIGIBLE = {"ses-2020-11-24-Run1", "ses-2020-11-24-Run2"}
M, WINDOW = 33, 50
NORMALIZATION_ADDEDUM_SHA = "8a103352eb6341152ec18096411563bc968162b7154074f80a958793b0a2d3e8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def eq(observed: object, expected: object, name: str) -> None:
    if observed != expected:
        raise ValueError(f"{name}: expected {expected!r}, found {observed!r}")


def finite_scores(artifact: Path, eligible: set[str]) -> dict[str, float]:
    with (artifact / "metrics_per_session.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "test_heldout"]
    observed = {row.get("session", "") for row in rows}
    eq(observed, eligible, "test-heldout metric session set")
    scores: dict[str, float] = {}
    for row in rows:
        eq(int(float(row.get("M", "nan"))), M, f"{row['session']} metric M")
        value = float(row["R2_variance_weighted"])
        if not math.isfinite(value):
            raise ValueError(f"{row['session']}: nonfinite R2")
        scores[row["session"]] = value
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--expected-protocol-sha", required=True)
    parser.add_argument("--normalization-addendum", required=True, type=Path)
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    parser.add_argument("--cell", choices=sorted(CELLS), required=True)
    args = parser.parse_args()
    artifact, protocol_path = args.artifact.resolve(), args.protocol.resolve()
    if sha256(protocol_path) != args.expected_protocol_sha:
        raise ValueError("protocol receipt SHA drift")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    addendum_path = args.normalization_addendum.resolve()
    if sha256(addendum_path) != NORMALIZATION_ADDEDUM_SHA:
        raise ValueError("normalization-parity addendum SHA drift")
    addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
    eq(addendum.get("protocol", {}).get("sha256"), args.expected_protocol_sha, "normalization addendum protocol binding")
    frozen = protocol["frozen_source_arms"][args.group][args.cell]
    required = ["resolved_config.yaml", "split_manifest.json", "checkpoint_manifest.json", "metrics_per_session.csv"]
    if missing := [name for name in required if not (artifact / name).is_file()]:
        raise ValueError(f"{artifact}: missing {missing}")
    cfg = yaml.safe_load((artifact / "resolved_config.yaml").read_text(encoding="utf-8"))
    data, model = cfg["data"], cfg["model"]
    fold, seed = CELLS[args.cell]
    variant, side = GROUPS[args.group]
    eq((data.get("task"), data.get("loso_fold"), cfg.get("seed")), ("m2", fold, seed), "runtime cell")
    eq((model.get("variant"), data.get("side_feature_group")), (variant, side), "runtime arm")
    expected_data = dict(frozen["source_data"])
    expected_data.update({
        "include_heldout_in_fit": False,
        "include_heldout_in_test": True,
        "query_start_trial": M,
        "allow_empty_heldout_query": True,
    })
    if data != expected_data:
        changed = {
            key: {"source_or_expected": expected_data.get(key), "runtime": data.get(key)}
            for key in sorted(set(data) | set(expected_data)) if data.get(key) != expected_data.get(key)
        }
        raise ValueError(f"runtime data mapping drift outside protocol whitelist: {changed}")
    eq(model, frozen["source_model"], "runtime model mapping")
    eq((cfg.get("train"), cfg.get("test")), (False, True), "test-only flags")
    trainer = cfg.get("trainer", {})
    eq(trainer.get("accelerator"), "cpu", "CPU-only accelerator")
    eq(trainer.get("devices"), 1, "CPU-only device count")
    source = frozen["checkpoint"]
    source_path = Path(source["path"]).resolve()
    if not source_path.is_file() or sha256(source_path) != source["sha256"]:
        raise ValueError("frozen source checkpoint SHA drift")
    eq(str(Path(str(cfg.get("ckpt_path"))).resolve()), str(source_path), "runtime checkpoint path")

    split = json.loads((artifact / "split_manifest.json").read_text(encoding="utf-8"))
    eq(split.get("heldout_evaluated_in_fit"), False, "heldout evaluated in fit")
    eq(split.get("heldout_evaluated_in_test"), True, "heldout evaluated in test")
    eq(split.get("query_start_trial"), M, "query start")
    audits = split.get("heldout_query_window_audit")
    expected_sessions = set(protocol["session_eligibility"]["all_historical_heldout_sessions"])
    if not isinstance(audits, dict) or set(audits) != expected_sessions:
        raise ValueError("heldout query audit does not cover exactly the historical six sessions")
    eligible: set[str] = set()
    for session, row in audits.items():
        for key, value in {"support_trials": M, "query_start_trial": M, "window_size": WINDOW, "full_window_disjoint": True}.items():
            eq(row.get(key), value, f"{session}.{key}")
        if session in INELIGIBLE:
            eq(row.get("total_trials"), M, f"{session}.total_trials")
            eq(row.get("query_trials"), 0, f"{session}.query_trials")
            eq(row.get("eligible_windows"), 0, f"{session}.eligible_windows")
            eq(row.get("ineligible_reason"), "zero_query_trials_after_chronological_support", f"{session}.ineligible_reason")
        else:
            if int(row.get("query_trials", 0)) <= 0 or int(row.get("eligible_windows", 0)) <= 0:
                raise ValueError(f"{session}: expected eligible query windows")
            if row.get("ineligible_reason") is not None:
                raise ValueError(f"{session}: eligible session has ineligible reason")
            eq(row.get("minimum_window_start_padded_bin"), row.get("raw_query_start_bin") + WINDOW - 1, f"{session} full-history boundary")
            eligible.add(session)
    scores = finite_scores(artifact, eligible)
    checkpoint = json.loads((artifact / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    eq(checkpoint.get("source_checkpoint_sha256"), source["sha256"], "source checkpoint receipt SHA")
    eq(checkpoint.get("artifact_checkpoint_sha256"), source["sha256"], "artifact checkpoint receipt SHA")

    t4_norm = split.get("native_t4_normalization")
    if args.group in {"t4", "ts4"}:
        expected = addendum.get("norms", {}).get(args.group, {}).get(args.cell, {})
        source_norm = expected.get("native_t4_normalization")
        if not isinstance(t4_norm, dict) or not isinstance(source_norm, dict):
            raise ValueError("T4/TS4 normalization receipt missing")
        eq(expected.get("source_checkpoint"), source, "normalization addendum source checkpoint")
        eq(t4_norm, source_norm, "addendum-bound train-only T4/TS4 normalization")
        peer_group = "ts4" if args.group == "t4" else "t4"
        peer_norm = addendum["norms"][peer_group][args.cell]["native_t4_normalization"]
        for key in ("mean", "std", "train_sessions"):
            eq(t4_norm.get(key), peer_norm.get(key), f"T4/TS4 same-cell parity {key}")
    elif t4_norm is not None:
        raise ValueError("F0 must not emit a T4 normalization receipt")

    out = artifact / "m33_disjoint_replay_correction_provenance.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    payload = {
        "schema_version": 1,
        "purpose": "M2_M33_local_heldout_support_query_contamination_correction_testonly",
        "protocol": {"path": str(protocol_path), "sha256": args.expected_protocol_sha},
        "normalization_parity_addendum": {"path": str(addendum_path), "sha256": NORMALIZATION_ADDEDUM_SHA},
        "group": args.group,
        "cell": {"name": args.cell, "fold": fold, "seed": seed},
        "test_only": True,
        "uses_cpu_only": True,
        "no_backward_optimizer_or_checkpoint_selection_on_heldout": True,
        "frozen_source_checkpoint": source,
        "eligible_scores": scores,
        "ineligible_sessions": {session: audits[session] for session in sorted(INELIGIBLE)},
        "query_window_audit": audits,
        "t4_ts4_label_normalization_parity_checked": args.group in {"t4", "ts4"},
        "scope": "M33-eligible four-session subset only; not a six-session significance test and not a hidden EvalAI result.",
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
