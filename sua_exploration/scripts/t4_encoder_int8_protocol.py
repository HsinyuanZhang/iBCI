#!/usr/bin/env python3
"""Fail-closed protocol binding for the selected SUA T4@50 INT8 experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)
ACTIVITY_BUDGET = 30
T4_LABEL_BUDGET = 50
EVALUATION_START_TRIAL = 50
MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
TEACHER_SHA256 = "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"protocol path escapes repository: {value}") from exc
    return resolved


def _load_hashed(row: dict[str, Any], *, role: str) -> tuple[Path, dict[str, Any]]:
    path = _path(str(row.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"{role}: {path}")
    if sha256_file(path) != row.get("sha256"):
        raise ValueError(f"{role} SHA-256 drifted")
    return path, json.loads(path.read_text(encoding="utf-8"))


def validate_source_fp32(source: dict[str, Any]) -> dict[str, float]:
    if source.get("formal_test_files_opened") is not False:
        raise ValueError("source FP32 aggregate must leave formal test unopened")
    contrasts = source.get("contrasts") or {}
    rows = {
        "t4_minus_b0": contrasts.get("t4_vs_original_spint_b0") or {},
        "t4_minus_ts4": contrasts.get("t4_vs_shuffled_label_ts4") or {},
    }
    deltas: dict[str, float] = {}
    for name, row in rows.items():
        if row.get("passes_all_gates") is not True:
            raise ValueError(f"source FP32 {name} did not pass every gate")
        gates = row.get("gates") or {}
        if not gates or not all(value is True for value in gates.values()):
            raise ValueError(f"source FP32 {name} gate receipt is incomplete")
        deltas[name] = float(row["mean_paired_delta_r2"])
        if deltas[name] < 0.03:
            raise ValueError(f"source FP32 {name} is below +0.03")
    protocol = source.get("protocol") or {}
    if (
        protocol.get("same_trial_count_and_prefix_for_all_arms") is not True
        or protocol.get("evaluation_backward_gradients") is not False
        or protocol.get("scored_epoch_window") != list(range(5, 13))
        or protocol.get("seeds") != list(SEEDS)
        or len(protocol.get("sessions") or []) != 6
    ):
        raise ValueError("source FP32 protocol receipt failed")
    return deltas


def validate_selection(
    selection_path: Path,
    *,
    source_fp32_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    selection_path = selection_path.expanduser().resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "status": "selected_and_authorized_for_encoder_int8",
        "selected_architecture": "coupled_ordinary_B3S_T4",
        "activity_calibration_n": ACTIVITY_BUDGET,
        "t4_label_feature_pool_n": T4_LABEL_BUDGET,
        "evaluation_start_trial": EVALUATION_START_TRIAL,
        "selection_mode": "chronological_first",
        "seeds": list(SEEDS),
        "strict_manifest_sha256": MANIFEST_SHA256,
        "teacher_sha256": TEACHER_SHA256,
        "formal_test_files_opened": False,
        "authorized_scope": "T4_B3S_identity_encoder_W8A8_INT32_plus_FP32_decoder",
        "source_fp32_evidence_scope": "M30_three_seed_T4_mechanism_eligibility_only",
        "source_fp32_is_not_the_selected_m50_checkpoint_comparison": True,
    }
    failed = [key for key, value in expected.items() if selection.get(key) != value]
    if failed:
        raise ValueError(f"final architecture selection receipt failed: {failed}")

    manifest_path = _path(str(selection.get("strict_manifest_path", "")))
    if not manifest_path.is_file() or sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("final architecture selection manifest receipt drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest.get("session_splits") or {}
    if [len(splits.get(name) or []) for name in ("train", "val", "test")] != [27, 6, 6]:
        raise ValueError("final architecture selection manifest split drifted")

    source_row = selection.get("source_fp32_aggregate") or {}
    source_path, source = _load_hashed(source_row, role="source FP32 aggregate")
    if source_fp32_path is not None and source_path != source_fp32_path.resolve():
        raise ValueError("selection/source FP32 path mismatch")
    deltas = validate_source_fp32(source)

    evidence = selection.get("candidate_dispositions") or []
    if not evidence:
        raise ValueError("selection receipt lacks candidate dispositions")
    for index, row in enumerate(evidence):
        _load_hashed(row, role=f"candidate disposition {index}")
        if row.get("disposition") not in {
            "ineffective",
            "early_stopped_by_predeclared_kill_signal",
            "failed_noninferiority",
        }:
            raise ValueError(f"candidate disposition {index} is not terminal")

    selected = selection.get("selected_seed_artifacts") or {}
    if set(selected) != {str(seed) for seed in SEEDS}:
        raise ValueError("selection receipt must bind exactly seeds 42/43/44")
    for seed in SEEDS:
        row = selected[str(seed)]
        metadata_path, metadata = _load_hashed(
            row.get("run_metadata") or {}, role=f"seed {seed} run metadata"
        )
        checkpoint_path = _path(str((row.get("checkpoint") or {}).get("path", "")))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if sha256_file(checkpoint_path) != (row.get("checkpoint") or {}).get("sha256"):
            raise ValueError(f"seed {seed} checkpoint SHA-256 drifted")
        result_path, result = _load_hashed(
            row.get("validation_result") or {}, role=f"seed {seed} validation result"
        )
        side = metadata.get("side_features") or {}
        training = metadata.get("training") or {}
        metadata_failed = {
            "seed": metadata.get("seed") == seed,
            "variant": metadata.get("variant") == "B3S",
            "side_group": side.get("group") == "t4",
            "side_pool": side.get("pool_size") == T4_LABEL_BUDGET,
            "activity_budget": training.get("calibration_n_trials") == ACTIVITY_BUDGET,
            "manifest": metadata.get("train_val_manifest_sha256") == MANIFEST_SHA256,
            "teacher": metadata.get("teacher_sha256") == TEACHER_SHA256,
            "formal_sealed": metadata.get("held_out_test_evaluated") is False,
            "checkpoint_location": checkpoint_path == metadata_path.parent / "epoch_ckpts" / "epoch_011.ckpt",
        }
        if not all(metadata_failed.values()):
            raise ValueError(
                f"seed {seed} selected metadata failed: "
                f"{[key for key, value in metadata_failed.items() if not value]}"
            )
        protocol = result.get("protocol") or {}
        result_failed = {
            "seed": result.get("seed") == seed,
            "variant": result.get("variant") == "B3S",
            "run_dir": Path(result.get("run_dir", "")).resolve() == metadata_path.parent,
            "metadata_hash": result.get("run_metadata_sha256") == row["run_metadata"]["sha256"],
            "activity_budget": protocol.get("train_activity_calibration_n") == ACTIVITY_BUDGET,
            "forward_activity_budget": protocol.get("evaluation_forward_calibration_n") == ACTIVITY_BUDGET,
            "label_budget": protocol.get("label_feature_calibration_n") == T4_LABEL_BUDGET,
            "pool": protocol.get("pool_size") == T4_LABEL_BUDGET,
            "mode": protocol.get("selection_mode") == "first",
            "epoch_window": result.get("epoch_list") == list(range(5, 13)),
            "formal_sealed": result.get("no_test_files_evaluated") is True,
        }
        if not all(result_failed.values()):
            raise ValueError(
                f"seed {seed} selected validation result failed: "
                f"{[key for key, value in result_failed.items() if not value]}"
            )
        if result_path.parent.name != "sua_t4_confidence_film_v1":
            raise ValueError(f"seed {seed} validation result namespace drifted")
    return selection, source, deltas


def selected_seed_entry(selection: dict[str, Any], seed: int) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"unexpected seed: {seed}")
    return selection["selected_seed_artifacts"][str(seed)]
