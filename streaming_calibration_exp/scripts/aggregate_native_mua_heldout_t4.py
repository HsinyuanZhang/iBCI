#!/usr/bin/env python3
"""Fail-closed paired M1/M2 local held-out-calibration T4 aggregator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


CELLS = ((1, 42), (1, 43), (2, 42))
GROUPS = ("f0", "t4", "ts4")
EXPECTED_VARIANT = {"f0": "B3", "t4": "B3S", "ts4": "B3S"}
EXPECTED_FEATURE_GROUP = {"f0": None, "t4": "t4", "ts4": "ts4"}
TASK_CONTRACT = {
    "m1": {
        "prefix": "native_mua_heldout_t4_v1",
        "support": 10,
        "sessions": (
            "ses-20121004",
            "ses-20121017",
            "ses-20121024",
        ),
    },
    "m2": {
        # The completed replacement queue was isolated to physical GPU1.  Keeping
        # this prefix explicit excludes two stale partial M2 artifacts from the
        # first, incorrectly GPU0-visible queue.
        "prefix": "native_mua_heldout_t4_v1_gpu1r1",
        "support": 33,
        "sessions": (
            "ses-2020-10-30-Run1",
            "ses-2020-10-30-Run2",
            "ses-2020-11-18-Run1",
            "ses-2020-11-19-Run1",
            "ses-2020-11-24-Run1",
            "ses-2020-11-24-Run2",
        ),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalization_sha(normalization: dict[str, Any]) -> str:
    encoded = {key: value for key, value in normalization.items() if key != "sha256"}
    payload = json.dumps(encoded, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing required JSON artifact: {path}")
    return json.loads(path.read_text())


def _validate_normalization(
    normalization: Any,
    *,
    artifact: Path,
    expected_feature_group: str,
    source_train_sessions: list[str],
) -> dict[str, Any]:
    if not isinstance(normalization, dict):
        raise ValueError(f"Missing train-only normalization provenance: {artifact}")
    if normalization.get("feature_group") != expected_feature_group:
        raise ValueError(f"Wrong normalization feature group: {artifact}")
    if normalization.get("train_sessions") != source_train_sessions:
        raise ValueError(f"Normalization was not fit on the frozen fold train sessions: {artifact}")
    for key in ("mean", "std"):
        values = normalization.get(key)
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError(f"Normalization {key} must contain four values: {artifact}")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"Non-finite normalization {key}: {artifact}")
    if not all(float(value) > 0.0 for value in normalization["std"]):
        raise ValueError(f"Non-positive normalization std: {artifact}")
    if normalization.get("sha256") != _normalization_sha(normalization):
        raise ValueError(f"Normalization hash mismatch: {artifact}")
    return normalization


def _validate_metrics(
    artifact: Path,
    *,
    task: str,
    group: str,
    fold: int,
    seed: int,
    support: int,
    expected_sessions: tuple[str, ...],
) -> dict[str, float]:
    metrics_path = artifact / "metrics_per_session.csv"
    if not metrics_path.is_file():
        raise ValueError(f"Missing metrics: {metrics_path}")
    rows = list(csv.DictReader(metrics_path.open()))
    heldout_rows = [row for row in rows if row.get("split") == "test_heldout"]
    sessions = [row.get("session", "") for row in heldout_rows]
    if len(sessions) != len(set(sessions)):
        raise ValueError(f"Duplicate held-out metric sessions: {artifact}")
    if set(sessions) != set(expected_sessions):
        raise ValueError(
            f"Wrong held-out sessions for {task}: expected {expected_sessions}, got {sessions}"
        )
    scores: dict[str, float] = {}
    for row in heldout_rows:
        if row.get("run_id") != artifact.name:
            raise ValueError(f"Metric run_id does not match artifact directory: {artifact}")
        if row.get("variant") != EXPECTED_VARIANT[group]:
            raise ValueError(f"Wrong metric variant for {group}: {artifact}")
        if int(row["seed"]) != seed or int(row["fold_id"]) != fold:
            raise ValueError(f"Wrong metric fold/seed: {artifact}")
        if int(float(row["M"])) != support:
            raise ValueError(f"Wrong chronological support in metrics: {artifact}")
        value = float(row["R2_variance_weighted"])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite held-out R2: {artifact}")
        scores[row["session"]] = value
    return {session: scores[session] for session in expected_sessions}


def _validate_cell(
    artifact: Path,
    *,
    frozen_aggregate: dict[str, Any],
    task: str,
    group: str,
    fold: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    contract = TASK_CONTRACT[task]
    support = int(contract["support"])
    expected_sessions = tuple(contract["sessions"])
    cell = f"fold{fold}_seed{seed}"
    provenance = _load_json(artifact / "heldout_t4_provenance.json")
    split = _load_json(artifact / "split_manifest.json")

    expected = {
        "schema_version": 1,
        "task": task,
        "group": group,
        "fold": fold,
        "seed": seed,
        "train": False,
        "test": True,
        "uses_backward_gradients": False,
        "support_trials": support,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(
                f"Bad provenance {key}: expected {value!r}, "
                f"got {provenance.get(key)!r} in {artifact}"
            )
    if split.get("fold_id") != fold or split.get("validation_protocol") != "loso":
        raise ValueError(f"Wrong split fold/protocol: {artifact}")
    if split.get("heldout_evaluated_in_fit") is not False:
        raise ValueError(f"Held-out data was enabled during fit: {artifact}")
    if split.get("heldout_evaluated_in_test") is not True:
        raise ValueError(f"Held-out local calibration data was not evaluated test-only: {artifact}")

    frozen_source = (
        frozen_aggregate.get("artifacts", {})
        .get(task, {})
        .get(group, {})
        .get(cell)
    )
    if not frozen_source:
        raise ValueError(f"Frozen aggregate has no source for {task}/{group}/{cell}")
    source_artifact = Path(provenance.get("source_artifact", "")).resolve()
    if source_artifact != Path(frozen_source).resolve():
        raise ValueError(f"Source artifact is not the frozen aggregate mapping: {artifact}")
    source_checkpoint = Path(provenance.get("source_checkpoint", "")).resolve()
    if not source_checkpoint.is_file():
        raise ValueError(f"Missing frozen source checkpoint: {source_checkpoint}")
    source_manifest = _load_json(source_artifact / "checkpoint_manifest.json")
    if Path(source_manifest.get("artifact_checkpoint_path", "")).resolve() != source_checkpoint:
        raise ValueError(f"Frozen checkpoint path/manifest mismatch: {artifact}")
    checkpoint_sha = _sha256(source_checkpoint)
    if (
        provenance.get("source_checkpoint_sha256") != checkpoint_sha
        or source_manifest.get("artifact_checkpoint_sha256") != checkpoint_sha
    ):
        raise ValueError(f"Frozen checkpoint SHA mismatch: {artifact}")
    source_metadata = _load_json(source_artifact / "run_metadata.json")
    source_split = _load_json(source_artifact / "split_manifest.json")
    if source_metadata.get("seed") != seed or source_metadata.get("fold_id") != fold:
        raise ValueError(f"Frozen source fold/seed mismatch: {artifact}")
    if source_split.get("train_sessions") != split.get("train_sessions"):
        raise ValueError(f"Frozen source/local replay train split mismatch: {artifact}")
    if Path(provenance.get("split_manifest", "")).resolve() != (
        artifact / "split_manifest.json"
    ).resolve():
        raise ValueError(f"Provenance points to a different split manifest: {artifact}")

    audits = provenance.get("heldout_calibration_audit")
    if not isinstance(audits, list):
        raise ValueError(f"Missing held-out calibration audit: {artifact}")
    audit_sessions = [entry.get("session") for entry in audits]
    if len(audit_sessions) != len(set(audit_sessions)):
        raise ValueError(f"Duplicate audit sessions: {artifact}")
    if set(audit_sessions) != set(expected_sessions):
        raise ValueError(f"Wrong audit sessions: {artifact}")
    ordered_audits = []
    by_session = {entry["session"]: entry for entry in audits}
    for session in expected_sessions:
        entry = by_session[session]
        if (
            entry.get("prefix_trials") != support
            or entry.get("rank") != 3
            or int(entry.get("directional_trials", 0)) < 3
            or not math.isfinite(float(entry.get("condition_number", math.inf)))
        ):
            raise ValueError(f"Invalid chronological calibration audit: {artifact}/{session}")
        calibration_nwb = Path(entry.get("calibration_nwb", ""))
        if (
            "held-out-calib" not in calibration_nwb.name
            or session not in calibration_nwb.name
            or not calibration_nwb.is_file()
        ):
            raise ValueError(f"Audit does not identify the expected local calibration NWB: {artifact}")
        ordered_audits.append(entry)

    normalization = provenance.get("train_only_normalization")
    if group == "f0":
        if normalization is not None or split.get("native_t4_normalization") is not None:
            raise ValueError(f"F0 unexpectedly contains T4 normalization: {artifact}")
    else:
        normalization = _validate_normalization(
            normalization,
            artifact=artifact,
            expected_feature_group=EXPECTED_FEATURE_GROUP[group],
            source_train_sessions=source_split["train_sessions"],
        )
        if normalization != split.get("native_t4_normalization"):
            raise ValueError(f"Provenance/split normalization mismatch: {artifact}")

    scores = _validate_metrics(
        artifact,
        task=task,
        group=group,
        fold=fold,
        seed=seed,
        support=support,
        expected_sessions=expected_sessions,
    )
    return scores, {
        "artifact": str(artifact.resolve()),
        "source_artifact": str(source_artifact),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": checkpoint_sha,
        "audit": ordered_audits,
        "normalization": normalization,
    }


def _same_t4_statistics(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in ("train_sessions", "mean", "std"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--m1-prefix", default=TASK_CONTRACT["m1"]["prefix"])
    parser.add_argument("--m2-prefix", default=TASK_CONTRACT["m2"]["prefix"])
    parser.add_argument("--m1-frozen-aggregate", type=Path)
    parser.add_argument("--m2-frozen-aggregate", type=Path)
    args = parser.parse_args()

    frozen_result_dir = args.out.parent.parent / "native_mua_t4_v1"
    frozen_paths = {
        "m1": args.m1_frozen_aggregate or frozen_result_dir / "aggregate_m1.json",
        "m2": args.m2_frozen_aggregate or frozen_result_dir / "aggregate_m2.json",
    }
    frozen = {task: _load_json(path) for task, path in frozen_paths.items()}
    prefixes = {"m1": args.m1_prefix, "m2": args.m2_prefix}
    task_results: dict[str, Any] = {}

    for task in ("m1", "m2"):
        prefix = prefixes[task]
        data: dict[tuple[str, int, int], dict[str, float]] = {}
        records: dict[tuple[str, int, int], dict[str, Any]] = {}
        group_cell_means: dict[str, dict[str, float]] = {group: {} for group in GROUPS}
        artifact_manifest: dict[str, dict[str, str]] = {group: {} for group in GROUPS}

        for group in GROUPS:
            for fold, seed in CELLS:
                cell = f"fold{fold}_seed{seed}"
                pattern = f"{prefix}_{group}_{task}_f{fold}_s{seed}_*"
                hits = sorted(
                    path
                    for path in args.root.glob(pattern)
                    if (path / "heldout_t4_provenance.json").is_file()
                )
                if len(hits) != 1:
                    raise ValueError(
                        f"{task}/{group}/{cell}: expected one provenance-complete "
                        f"artifact for {pattern}, found {hits}"
                    )
                scores, record = _validate_cell(
                    hits[0],
                    frozen_aggregate=frozen[task],
                    task=task,
                    group=group,
                    fold=fold,
                    seed=seed,
                )
                data[(group, fold, seed)] = scores
                records[(group, fold, seed)] = record
                group_cell_means[group][cell] = float(statistics.fmean(scores.values()))
                artifact_manifest[group][cell] = record["artifact"]

        for fold, seed in CELLS:
            t4 = records[("t4", fold, seed)]
            ts4 = records[("ts4", fold, seed)]
            if not _same_t4_statistics(t4["normalization"], ts4["normalization"]):
                raise ValueError(
                    f"T4/TS4 train-only statistics differ for {task}/fold{fold}_seed{seed}"
                )
            canonical_audit = [
                {
                    key: entry[key]
                    for key in (
                        "session",
                        "prefix_trials",
                        "directional_trials",
                        "rank",
                        "condition_number",
                    )
                }
                for entry in t4["audit"]
            ]
            for group in ("f0", "ts4"):
                comparison = [
                    {
                        key: entry[key]
                        for key in (
                            "session",
                            "prefix_trials",
                            "directional_trials",
                            "rank",
                            "condition_number",
                        )
                    }
                    for entry in records[(group, fold, seed)]["audit"]
                ]
                if comparison != canonical_audit:
                    raise ValueError(
                        f"Calibration audit differs across controls for "
                        f"{task}/fold{fold}_seed{seed}"
                    )

        paired_deltas: dict[str, Any] = {}
        for control, name in (("f0", "T4_minus_F0"), ("ts4", "T4_minus_TS4")):
            per_cell_session: dict[str, dict[str, float]] = {}
            all_deltas: list[float] = []
            for fold, seed in CELLS:
                cell = f"fold{fold}_seed{seed}"
                deltas = {
                    session: data[("t4", fold, seed)][session]
                    - data[(control, fold, seed)][session]
                    for session in TASK_CONTRACT[task]["sessions"]
                }
                per_cell_session[cell] = deltas
                all_deltas.extend(deltas.values())
            cell_means = {
                cell: float(statistics.fmean(values.values()))
                for cell, values in per_cell_session.items()
            }
            paired_deltas[name] = {
                "per_cell_per_session": per_cell_session,
                "per_cell_mean_delta_r2": cell_means,
                "mean_delta_r2": float(statistics.fmean(all_deltas)),
                "positive_cell_count": sum(value > 0.0 for value in cell_means.values()),
                "positive_session_count": sum(value > 0.0 for value in all_deltas),
                "session_count": len(all_deltas),
            }

        task_results[task] = {
            "support_policy": f"chronological_first_{TASK_CONTRACT[task]['support']}",
            "heldout_calibration_sessions": list(TASK_CONTRACT[task]["sessions"]),
            "artifact_manifest": artifact_manifest,
            "group_per_cell_mean_r2": group_cell_means,
            "group_overall_mean_r2": {
                group: float(statistics.fmean(cell_means.values()))
                for group, cell_means in group_cell_means.items()
            },
            "paired_deltas": paired_deltas,
        }

    output = {
        "schema_version": 2,
        "purpose": "strict_native_mua_local_heldout_calibration_testonly_t4",
        "evaluation_scope": (
            "Local held-out-calib NWB replay matching the existing SPINT/B3 path; "
            "not a hidden EvalAI query/test-set result."
        ),
        "uses_backward_gradients_during_calibration": False,
        "prefix_manifest": prefixes,
        "frozen_source_aggregates": {
            task: str(path.resolve()) for task, path in frozen_paths.items()
        },
        "cells": [f"fold{fold}_seed{seed}" for fold, seed in CELLS],
        "tasks": task_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
