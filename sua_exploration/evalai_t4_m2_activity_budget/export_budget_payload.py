#!/usr/bin/env python3
"""Export official-compatible M2 cached identities for M4/M10 activity30."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"
ALLOWED_BUDGETS = (4, 10, 30)
ACTIVITY_BUDGET = 30
CHANNELS = 96
WINDOW_SIZE = 50
RIDGE_LAMBDA = 0.1


class ExportError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExportError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def support_indices(dataset: Any, session: str, budget: int) -> np.ndarray:
    require(budget in ALLOWED_BUDGETS, "unsupported label budget")
    if budget != 4:
        return np.arange(budget, dtype=np.int64)
    from sua_exploration.mc_maze.d_optimal_calibration_design import (
        greedy_forward_d_optimal_indices,
    )

    angles = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)
    require(angles.size >= ACTIVITY_BUDGET, f"{session}: fewer than 30 cue rows")
    candidates = np.flatnonzero(np.isfinite(angles[:ACTIVITY_BUDGET])).astype(np.int64)
    require(candidates.size >= 4, f"{session}: fewer than four finite first-30 cues")
    local = greedy_forward_d_optimal_indices(angles[candidates], 4)
    selected = np.sort(candidates[local]).astype(np.int64, copy=False)
    require(selected.size == 4 and np.unique(selected).size == 4, "M4 D-opt selection drift")
    return np.ascontiguousarray(selected)


def build_identity_inputs(dataset: Any, session: str, budget: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.physical import _ridge_side

    selected = support_indices(dataset, session, budget)
    calibration = np.asarray(dataset.calib_trialized_neural_features[session], dtype=np.float32)
    require(calibration.ndim == 3 and calibration.shape[0] >= ACTIVITY_BUDGET,
            f"{session}: calibration activity shape drift")
    activity = np.ascontiguousarray(calibration[:ACTIVITY_BUDGET], dtype=np.float32)
    require(activity.shape == (ACTIVITY_BUDGET, 100, CHANNELS) and np.isfinite(activity).all(),
            f"{session}: activity30 shape/nonfinite drift")
    side, evidence = _ridge_side(dataset, session, selected)
    require(side.shape == (CHANNELS, 4) and np.isfinite(side).all(), f"{session}: side drift")
    return activity, side, {
        **evidence,
        "activity_budget": ACTIVITY_BUDGET,
        "activity_sha256": sha256_array(activity),
        "selection_disclosure": (
            "D-opt reads finite direction metadata in first30 candidate pool"
            if budget == 4
            else "chronological first-M"
        ),
    }


def default_output(budget: int) -> Path:
    return ARTIFACT_ROOT / f"t4_m2_seed42_ridge_m{budget}_activity30_identity.pkl"


def export_payload(output: Path, *, budget: int) -> dict[str, object]:
    require(budget in ALLOWED_BUDGETS, "unsupported label budget")
    require(not output.exists(), f"refusing to overwrite {output}")
    from sua_exploration.evalai_t4_m2.export_t4_payload import (
        EXPECTED_CHECKPOINT_SHA256,
        EXPECTED_NORMALIZATION_SHA256,
        EXPECTED_TEACHER_SHA256,
        calibration_file_map,
        load_frozen_model_and_data,
        manual_decode,
    )

    model, data_module, task_config, metadata = load_frozen_model_and_data()
    require(metadata["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, "T4 checkpoint drift")
    require(metadata["teacher_checkpoint_sha256"] == EXPECTED_TEACHER_SHA256, "teacher drift")
    require(metadata["normalization_sha256"] == EXPECTED_NORMALIZATION_SHA256, "normalizer drift")
    student = model.student.cpu().eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    session_to_tag = calibration_file_map(
        Path(__file__).resolve().parents[2] / "SPINT-main/data/000953",
        task_config,
    )
    identities: dict[str, np.ndarray] = {}
    records: dict[str, object] = {}
    max_direct = 0.0
    max_decoder = 0.0
    import torch

    for dataset in (data_module.train_dataset, data_module.val_heldout_dataset):
        require(dataset is not None, "official export requires both M2 session sets")
        for session in sorted(dataset.calib_trialized_neural_features):
            require(session in session_to_tag, f"{session}: missing official dataset tag")
            activity, side, evidence = build_identity_inputs(dataset, session, budget)
            support = torch.from_numpy(activity).unsqueeze(0)
            side_tensor = torch.from_numpy(side).unsqueeze(0)
            with torch.inference_mode():
                identity = student.compute_identity(support, side_features=side_tensor)
                neural = torch.from_numpy(
                    np.asarray(dataset.neural_data[session][50:100], dtype=np.float32)
                ).unsqueeze(0)
                direct, _ = student(neural, calib_trials=support, side_features=side_tensor)
                cached, _ = student(neural, identity=identity)
                decoder = manual_decode(student.decoder, neural, identity)
            direct_error = float((direct - cached).abs().max().item())
            decoder_error = float((direct - decoder).abs().max().item())
            require(direct_error == 0.0 and decoder_error == 0.0,
                    f"{session}: cached identity is not bit-exact")
            max_direct = max(max_direct, direct_error)
            max_decoder = max(max_decoder, decoder_error)
            identity_np = np.ascontiguousarray(identity.squeeze(0).numpy(), dtype=np.float32)
            require(identity_np.shape == (CHANNELS, WINDOW_SIZE) and np.isfinite(identity_np).all(),
                    f"{session}: identity drift")
            tag = session_to_tag[session]
            require(tag not in identities, "duplicate official dataset tag")
            identities[tag] = identity_np
            records[session] = {
                "dataset_tag": tag,
                "label_budget": budget,
                "activity_budget": ACTIVITY_BUDGET,
                "activity_shape": list(activity.shape),
                "activity_sha256": sha256_array(activity),
                "side_sha256": sha256_array(side),
                "identity_sha256": sha256_array(identity_np),
                "side_evidence": evidence,
                "direct_vs_cached_max_abs": direct_error,
                "direct_vs_decoder_max_abs": decoder_error,
            }
    require(len(identities) == 13 and set(records) == set(session_to_tag), "official session coverage drift")
    decoder = student.decoder.cpu().eval()
    payload = {
        "schema_version": "e8_t4_m2_cached_identity_v1",
        "task": task_config.task,
        "decoder": decoder,
        "identity_by_dataset_tag": identities,
        "window_size": WINDOW_SIZE,
        "behavior_scaling_factor": 5.0,
        "smooth_observations": False,
        "metadata": {
            **metadata,
            "screen_id": "m2_ridge_t4_activity30_official_budget_v1",
            "arm": f"ridge_m{budget}_activity30",
            "seed": 42,
            "label_budget": budget,
            "activity_budget": ACTIVITY_BUDGET,
            "normalized_lambda": RIDGE_LAMBDA,
            "calibration_uses_target_labels": True,
            "m4_candidate_cue_metadata_count": 30 if budget == 4 else budget,
            "online_state": "cached_E[N,50]",
            "online_backward_pass": False,
            "session_records": records,
            "max_direct_vs_cached_identity_abs": max_direct,
            "max_direct_vs_decoder_only_abs": max_decoder,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {
        "schema_version": "m2_ridge_t4_activity30_official_payload_receipt_v1",
        "status": "EXPORTED_NOT_SUBMITTED",
        "budget": budget,
        "activity_budget": ACTIVITY_BUDGET,
        "payload_path": str(output),
        "payload_bytes": output.stat().st_size,
        "payload_sha256": sha256_file(output),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "teacher_checkpoint_sha256": metadata["teacher_checkpoint_sha256"],
        "normalization_sha256": metadata["normalization_sha256"],
        "session_count": len(identities),
        "max_direct_vs_cached_identity_abs": max_direct,
        "max_direct_vs_decoder_only_abs": max_decoder,
        "session_records": records,
    }
    receipt_path = output.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--budget", type=int, choices=ALLOWED_BUDGETS, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or default_output(args.budget)
    if not args.execute:
        print(json.dumps({
            "schema": "m2_ridge_t4_activity30_official_payload_dry_v1",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_TORCH_NO_WRITE_NO_SUBMISSION",
            "budget": args.budget,
            "activity_budget": ACTIVITY_BUDGET,
            "output": str(output),
        }, sort_keys=True))
        return
    print(json.dumps(export_payload(output, budget=args.budget), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
