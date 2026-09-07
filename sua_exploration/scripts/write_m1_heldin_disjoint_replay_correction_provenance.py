#!/usr/bin/env python3
"""Validate and seal one M1 held-in-calib post-support correction replay."""
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
SESSION_BY_FOLD = {1: "ses-20120926", 2: "ses-20120927"}
M, WINDOW = 10, 100
RUNTIME_DATA_WHITELIST_OVERRIDES = {
    "query_start_trial": 0,
    "heldin_query_start_trial": M,
    "allow_empty_heldout_query": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def eq(observed: object, expected: object, name: str) -> None:
    if observed != expected:
        raise ValueError(f"{name}: expected {expected!r}, found {observed!r}")


def finite_scores(artifact: Path, expected_session: str) -> dict[str, float]:
    with (artifact / "metrics_per_session.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "test_heldin"]
    observed = {row.get("session", "") for row in rows}
    eq(observed, {expected_session}, "test-heldin metric session set")
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
    eq((data.get("task"), data.get("loso_fold"), cfg.get("seed")), ("m1", fold, seed), "runtime cell")
    eq((model.get("variant"), data.get("side_feature_group")), (variant, side), "runtime arm")
    expected_data = dict(frozen["source_data"])
    for key, value in RUNTIME_DATA_WHITELIST_OVERRIDES.items():
        expected_data[key] = value
    if data != expected_data:
        changed = {
            key: {"source_or_expected": expected_data.get(key), "runtime": data.get(key)}
            for key in sorted(set(data) | set(expected_data)) if data.get(key) != expected_data.get(key)
        }
        raise ValueError(f"runtime data mapping drift outside protocol whitelist: {changed}")
    eq(model, frozen["source_model"], "runtime model mapping")
    eq((cfg.get("train"), cfg.get("test")), (False, True), "test-only flags")
    source = frozen["checkpoint"]
    source_path = Path(source["path"]).resolve()
    if not source_path.is_file() or sha256(source_path) != source["sha256"]:
        raise ValueError("frozen source checkpoint SHA drift")
    eq(str(Path(str(cfg.get("ckpt_path"))).resolve()), str(source_path), "runtime checkpoint path")

    split = json.loads((artifact / "split_manifest.json").read_text(encoding="utf-8"))
    eq(split.get("heldout_evaluated_in_fit"), False, "heldout evaluated in fit")
    eq(split.get("heldout_evaluated_in_test"), False, "heldout evaluated in test")
    eq(split.get("heldin_query_start_trial"), M, "heldin query start")
    audits = split.get("heldin_query_window_audit")
    expected_session = SESSION_BY_FOLD[fold]
    if not isinstance(audits, dict) or set(audits) != {expected_session}:
        raise ValueError("heldin query audit does not cover exactly the left-out session")
    row = audits[expected_session]
    for key, value in {"support_trials": M, "query_start_trial": M, "window_size": WINDOW, "full_window_disjoint": True}.items():
        eq(row.get(key), value, f"{expected_session}.{key}")
    if int(row.get("query_trials", 0)) <= 0 or int(row.get("eligible_windows", 0)) <= 0:
        raise ValueError(f"{expected_session}: expected eligible query windows")
    eq(row.get("minimum_window_start_padded_bin"), row.get("raw_query_start_bin") + WINDOW - 1, f"{expected_session} full-history boundary")
    scores = finite_scores(artifact, expected_session)
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
        for key in ("feature_group", "mean", "std", "train_sessions"):
            eq(t4_norm.get(key), source_norm.get(key), f"addendum-bound train-only T4/TS4 normalization {key}")
        peer_group = "ts4" if args.group == "t4" else "t4"
        peer_norm = addendum["norms"][peer_group][args.cell]["native_t4_normalization"]
        for key in ("mean", "std", "train_sessions"):
            eq(t4_norm.get(key), peer_norm.get(key), f"T4/TS4 same-cell parity {key}")
    elif t4_norm is not None:
        raise ValueError("F0 must not emit a T4 normalization receipt")

    out = artifact / "m1_heldin_disjoint_replay_correction_provenance.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    payload = {
        "schema_version": 1,
        "purpose": "M1_internal_LOSO_heldin_calib_post_support_contamination_correction_testonly",
        "protocol": {"path": str(protocol_path), "sha256": args.expected_protocol_sha},
        "normalization_parity_addendum": {"path": str(addendum_path), "sha256": sha256(addendum_path)},
        "group": args.group,
        "cell": {"name": args.cell, "fold": fold, "seed": seed, "left_out_session": expected_session},
        "test_only": True,
        "no_backward_optimizer_or_checkpoint_selection_on_heldin_query": True,
        "frozen_source_checkpoint": source,
        "scores": scores,
        "query_window_audit": audits,
        "t4_ts4_label_normalization_parity_checked": args.group in {"t4", "ts4"},
        "limitations": protocol.get("limitations", {}),
        "scope": "M1 internal-LOSO held-in-calib post-support development evidence only; not hidden EvalAI test.",
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
