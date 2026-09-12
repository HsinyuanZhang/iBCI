#!/usr/bin/env python3
"""Algebraically restate sealed local RIFT curves on FAIR V2's two R² faces.

This reader never loads a decoder checkpoint into a model and never executes a
network forward.  It combines per-recording R² from sealed historical receipts
with the matching current FAIR V2 evaluation targets only to obtain the two
SST denominators and recover SSE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
EXTERNAL = HERE.parent
ROOT = EXTERNAL.parent
WS = ROOT.parent
for p in (EXTERNAL, ROOT / "learnable_recency_v1/scripts", ROOT / "learnable_recency_v1/src", WS / "btransform_unified_v1/src", WS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from fair_v2 import data
from fair_v2.metrics import array_sha

RESULTS = ROOT / "learnable_recency_v1/results"
SPECS = {
    "m1": {"receipt": RESULTS / "m1_projadd_learned_slope_default_s42/score_receipt.json", "curve": "ema_by_epoch", "epochs": 24, "fixed": 24, "old_metric": "legacy_flattened", "selected": lambda x: int(x["selection"]["epoch"]), "checkpoint": lambda x, e: x["checkpoint_sha256_by_epoch"][str(e)]},
    "m2": {"receipt": RESULTS / "selection_m2_projadd_learned_slope_default_ext6_s42/score_receipt.json", "curve": "ema_by_epoch", "epochs": 24, "fixed": 24, "old_metric": "legacy_flattened", "selected": lambda x: int(x["selection"]["epoch"]), "checkpoint": lambda x, e: x["ema_by_epoch"][str(e)]["checkpoint_sha256"]},
    "h1": {"receipt": RESULTS / "h1_learned_slope_default_s42/ho_m3_selection.json", "curve": "curve", "epochs": 32, "fixed": 32, "old_metric": "standard_grouped", "selected": lambda x: int(x["selected"]["epoch"]), "checkpoint": lambda _x, e: _sha_file(RESULTS / f"h1_learned_slope_default_s42/epoch_{e:03d}.pt")},
}


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON object required")
    return value


def _finite(value: Any, name: str) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name}: finite number required")
    return x


def _denominators(y: np.ndarray) -> tuple[float, float]:
    a = np.asarray(y, np.float64)
    if a.ndim != 2 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("target must be finite [N,O], N >= 2")
    standard = float(np.square(a - a.mean(axis=0)).sum())
    legacy = float(np.square(a - a.mean()).sum())
    # Conversion would silently manufacture a result if either face is
    # undefined, so both faces are explicitly required for every recording.
    if standard <= 0 or legacy <= 0 or np.any(np.square(a - a.mean(axis=0)).sum(axis=0) <= 0):
        raise ValueError("each recording/group must have positive per-output, standard, and legacy SST")
    return standard, legacy


def _h1_grouped(items: Mapping[str, dict]) -> dict[str, dict[str, Any]]:
    from falcon_challenge.config import FalconConfig, FalconTask
    grouped: dict[str, list[tuple[str, dict]]] = {}
    for session, item in items.items():
        basename = Path(item["support_provenance"]["raw_nwb"]).stem
        group = FalconConfig(FalconTask.h1).hash_dataset(basename).split("_set_")[0]
        grouped.setdefault(group, []).append((session, item))
    if set(grouped) != {f"S{i}" for i in range(6, 13)}:
        raise ValueError(f"H1 group roster drift: {sorted(grouped)}")
    out = {}
    for group, rows in grouped.items():
        y = np.concatenate([np.asarray(item["Y"], np.float32) for _, item in rows])
        out[group] = {"Y": y, "recordings": [session for session, _ in rows], "recording_window_counts": {session: int(len(item["Y"])) for session, item in rows}}
    return out


def _faces(y: np.ndarray, old_r2: float, old_metric: str) -> dict[str, float]:
    standard_sst, legacy_sst = _denominators(y)
    if old_metric == "legacy_flattened":
        sse = (1.0 - old_r2) * legacy_sst
    elif old_metric == "standard_grouped":
        sse = (1.0 - old_r2) * standard_sst
    else:
        raise ValueError(old_metric)
    return {"standard_variance_weighted_r2": float(1.0 - sse / standard_sst), "legacy_flattened_r2": float(1.0 - sse / legacy_sst), "recovered_sse": float(sse), "standard_sst": standard_sst, "legacy_sst": legacy_sst}


def _curve_row(receipt: dict, task: str, epoch: int) -> dict:
    if task == "h1":
        rows = receipt["curve"]
        found = [row for row in rows if int(row["epoch"]) == epoch]
        if len(found) != 1:
            raise ValueError(f"H1 epoch {epoch}: curve row missing/ambiguous")
        return found[0]
    row = receipt["ema_by_epoch"].get(str(epoch))
    if not isinstance(row, dict):
        raise ValueError(f"{task} epoch {epoch}: curve row missing")
    return row


def _old_scores(row: dict, task: str) -> dict[str, float]:
    key = "per_session_r2" if task == "h1" else "per_session"
    values = row.get(key)
    if not isinstance(values, Mapping):
        raise ValueError(f"{task}: per-session score map absent")
    if task == "h1":
        return {str(session): _finite(score, f"{task}/{session}") for session, score in values.items()}
    return {str(session): _finite(score["r2"], f"{task}/{session}") for session, score in values.items()}


def _evaluate_epoch(task: str, receipt: dict, targets: dict[str, dict], epoch: int) -> dict:
    spec = SPECS[task]
    curve = _curve_row(receipt, task, epoch)
    old = _old_scores(curve, task)
    if set(old) != set(targets):
        raise ValueError(f"{task} epoch {epoch}: sealed/current roster mismatch: {sorted(old)} vs {sorted(targets)}")
    per = {session: _faces(targets[session]["Y"], old[session], spec["old_metric"]) for session in sorted(targets)}
    counts = {session: int(len(targets[session]["Y"])) for session in sorted(targets)}
    if task != "h1":
        old_counts = {s: int(curve["per_session"][s]["window_count"]) for s in old}
        if counts != old_counts or sum(counts.values()) != int(curve["n_windows"]):
            raise ValueError(f"{task} epoch {epoch}: sealed/current window-count drift")
    return {"epoch": epoch, "checkpoint_sha256": spec["checkpoint"](receipt, epoch), "per_recording_or_group": per,
            "recording_or_group_window_counts": counts, "n_windows": sum(counts.values()),
            "standard_equal_session_or_group_mean": float(np.mean([x["standard_variance_weighted_r2"] for x in per.values()])),
            "legacy_equal_session_or_group_mean": float(np.mean([x["legacy_flattened_r2"] for x in per.values()])),
            "old_sealed_metric": spec["old_metric"], "old_equal_mean": float(np.mean(list(old.values())))}


def build() -> dict[str, Any]:
    # This is intentionally the only target-data read.  Call it only once the
    # new FAIR baseline's source selection has been sealed by the parent flow.
    output: dict[str, Any] = {"schema": "fair_v2_rift_reference_algebraic_restatement_v1", "network_forward_executed": False,
        "method": "sealed per-recording R2 + current matching-Y SST algebra", "tasks": {}}
    for task, spec in SPECS.items():
        receipt_path = spec["receipt"]
        sealed = _read(receipt_path)
        loaded = data.load_task(task, include_evaluation=True)
        evaluation = loaded["evaluation"]
        targets = _h1_grouped(evaluation) if task == "h1" else {session: {"Y": np.asarray(item["Y"], np.float32), "recordings": [session], "recording_window_counts": {session: int(len(item["Y"]))}} for session, item in evaluation.items()}
        fixed_epoch, historical_epoch = spec["fixed"], spec["selected"](sealed)
        all_epochs = {str(epoch): _evaluate_epoch(task, sealed, targets, epoch) for epoch in range(1, spec["epochs"] + 1)}
        fixed = dict(all_epochs[str(fixed_epoch)])
        fixed.update({"epoch_rule": "retrospective target-independent fixed-final reporting rule", "target_labels_used_for_epoch_selection": False,
                      "historical_curve_had_visible_local_target_labels": True})
        historical = dict(all_epochs[str(historical_epoch)])
        historical.update({"epoch_rule": "historical local-HO earliest-max selector from sealed receipt", "target_labels_used_for_epoch_selection": True})
        output["tasks"][task] = {"sealed_score_receipt": str(receipt_path), "sealed_score_receipt_sha256": _sha_file(receipt_path),
            "current_target_y_sha256": {session: array_sha(row["Y"]) for session, row in targets.items()},
            "current_target_recording_groups": {session: {"recordings": row["recordings"], "recording_window_counts": row["recording_window_counts"]} for session, row in targets.items()},
            "fixed_final": fixed, "historical_ho_selected": historical,
            "metric_specific_curve_best_epoch_diagnostics": {
                "standard": max(range(1, spec["epochs"] + 1), key=lambda e: (all_epochs[str(e)]["standard_equal_session_or_group_mean"], -e)),
                "legacy": max(range(1, spec["epochs"] + 1), key=lambda e: (all_epochs[str(e)]["legacy_equal_session_or_group_mean"], -e)),
            }, "all_epoch_same_epoch_faces": all_epochs}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results/rift_reference_v2/receipt.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = build()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({task: {rule: row[rule]["standard_equal_session_or_group_mean"] for rule in ("fixed_final", "historical_ho_selected")} for task, row in result["tasks"].items()}, indent=2))


if __name__ == "__main__":
    main()
