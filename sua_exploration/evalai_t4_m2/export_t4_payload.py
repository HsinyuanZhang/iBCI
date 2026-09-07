#!/usr/bin/env python3
"""Export the frozen M2 T4 candidate as decoder + cached session identities.

The exporter is deliberately read-only outside its own ``artifacts`` directory.
It reconstructs the selected Lightning checkpoint, uses the exact first-33
calibration policy and train-only T4 normalization, and proves that cached
identity deployment is numerically identical to the training-time T4 path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
STREAMING_ROOT = ROOT / "streaming_calibration_exp"
RUN_ROOT = (
    STREAMING_ROOT
    / "outputs/streaming_calibration"
    / "m2_spint_t4_mainline_fp32_v1_t4_m2_s42_20260730_131806"
)
CHECKPOINT = RUN_ROOT / "checkpoints/best.ckpt"
RESOLVED_CONFIG = RUN_ROOT / "resolved_config.yaml"
CHECKPOINT_MANIFEST = RUN_ROOT / "checkpoint_manifest.json"
SPLIT_MANIFEST = RUN_ROOT / "split_manifest.json"
TEACHER_CHECKPOINT = (
    ROOT
    / "SPINT-main/logs/train/runs/2026-07-07-16-05-16/checkpoints/best_ckpt/epoch_034.ckpt"
)
M2_DATA_DIR = ROOT / "SPINT-main/data/000953"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "artifacts/t4_m2_seed42_identity.pkl"
EXPECTED_CHECKPOINT_SHA256 = (
    "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
)
EXPECTED_NORMALIZATION_SHA256 = (
    "d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e"
)
EXPECTED_TEACHER_SHA256 = (
    "fbcb9914561c4664fa0f8d0b1791e67505841d3ac470ea7ad68d54e408ca13ec"
)
EXPECTED_SESSION_COUNT = 13
CALIBRATION_N_TRIALS = 33
SESSION_PATTERN = re.compile(r"(ses-\d{4}-\d{2}-\d{2}-Run\d+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def manual_decode(decoder: torch.nn.Module, neural: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
    source = neural.permute(0, 2, 1) + identity
    source = decoder.fc_in(source)
    query = decoder.fc_in(decoder.rep).to(source)
    transformed, _ = decoder.transformer(query.repeat(source.shape[0], 1, 1), source)
    return decoder.fc_out(transformed).permute(0, 2, 1)


def calibration_file_map(data_dir: Path, task_config: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in sorted(data_dir.rglob("*calib*.nwb")):
        match = SESSION_PATTERN.search(path.name)
        if match is None:
            raise RuntimeError(f"Cannot derive session name from calibration file {path}")
        session_name = match.group(1)
        dataset_tag = task_config.hash_dataset(path.stem)
        if session_name in mapping and mapping[session_name] != dataset_tag:
            raise RuntimeError(f"Conflicting dataset tags for {session_name}")
        mapping[session_name] = dataset_tag
    return mapping


def load_frozen_model_and_data() -> tuple[Any, Any, Any, dict[str, Any]]:
    # ``src`` must resolve to streaming_calibration_exp for checkpoint
    # reconstruction. The exported decoder itself is compatible with the SPINT
    # base image's unchanged SpintModel class.
    sys.path.insert(0, str(STREAMING_ROOT))
    from falcon_challenge.config import FalconConfig, FalconTask
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    for required in (
        CHECKPOINT,
        RESOLVED_CONFIG,
        CHECKPOINT_MANIFEST,
        SPLIT_MANIFEST,
        TEACHER_CHECKPOINT,
        M2_DATA_DIR,
    ):
        if not required.is_file():
            if not required.is_dir():
                raise FileNotFoundError(required)
    checkpoint_sha = sha256_file(CHECKPOINT)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Frozen T4 checkpoint drift: {checkpoint_sha} != {EXPECTED_CHECKPOINT_SHA256}"
        )
    checkpoint_manifest = json.loads(CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))
    if checkpoint_manifest.get("artifact_checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("Checkpoint manifest does not bind the selected artifact")

    split_manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    normalization = split_manifest.get("native_t4_normalization", {})
    if normalization.get("feature_group") != "t4":
        raise RuntimeError("Frozen split manifest is not the aligned T4 arm")
    if normalization.get("sha256") != EXPECTED_NORMALIZATION_SHA256:
        raise RuntimeError("Frozen train-only T4 normalization receipt drifted")

    config = OmegaConf.load(RESOLVED_CONFIG)
    if str(config.data.task).lower() != "m2" or int(config.data.calibration_n_trials) != 33:
        raise RuntimeError("Resolved config is not the frozen M2 first-33 candidate")
    if int(config.seed) != 42 or str(config.data.side_feature_group).lower() != "t4":
        raise RuntimeError("Resolved config is not canonical seed-42 aligned T4")
    if bool(config.data.random_calibration):
        raise RuntimeError("Submission calibration must be chronological and deterministic")
    teacher_sha = sha256_file(TEACHER_CHECKPOINT)
    if teacher_sha != EXPECTED_TEACHER_SHA256:
        raise RuntimeError(
            f"Frozen teacher checkpoint drift: {teacher_sha} != {EXPECTED_TEACHER_SHA256}"
        )
    # The resolved Hydra receipt stores these two paths relative to the original
    # streaming_calibration_exp working directory. Bind them explicitly so this
    # isolated exporter cannot accidentally resolve through its own directory.
    config.model.teacher_ckpt_path = str(TEACHER_CHECKPOINT)
    config.data.data_dir = str(M2_DATA_DIR)

    model = instantiate(config.model)
    model.setup("test")
    # This is a trusted, locally generated checkpoint already bound by SHA-256.
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    if model.student is None or model.student.decoder_mode != "coupled":
        raise RuntimeError("Frozen candidate did not construct the coupled T4 student")
    if not model.student._decoder_frozen:
        raise RuntimeError("Frozen candidate unexpectedly has a trainable decoder")

    data_module = instantiate(config.data)
    data_module.prepare_data()
    data_module.setup("test")
    actual_normalization = data_module.native_t4_normalization
    np.testing.assert_array_equal(
        np.asarray(actual_normalization["mean"], dtype=np.float32),
        np.asarray(normalization["mean"], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(actual_normalization["std"], dtype=np.float32),
        np.asarray(normalization["std"], dtype=np.float32),
    )

    task_config = FalconConfig(task=FalconTask.m2)
    metadata = {
        "checkpoint_sha256": checkpoint_sha,
        "teacher_checkpoint_sha256": teacher_sha,
        "normalization_sha256": normalization["sha256"],
        "normalization_mean": normalization["mean"],
        "normalization_std": normalization["std"],
        "train_sessions": normalization["train_sessions"],
        "resolved_config_sha256": sha256_file(RESOLVED_CONFIG),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
    }
    return model, data_module, task_config, metadata


def export_payload(output: Path, force: bool = False) -> dict[str, Any]:
    if output.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing payload: {output}")
    model, data_module, task_config, metadata = load_frozen_model_and_data()
    student = model.student
    assert student is not None

    session_to_tag = calibration_file_map(
        M2_DATA_DIR, task_config
    )
    datasets = (data_module.train_dataset, data_module.val_heldout_dataset)
    session_records: dict[str, Any] = {}
    identity_by_tag: dict[str, np.ndarray] = {}
    max_direct_vs_identity = 0.0
    max_direct_vs_decoder = 0.0

    for dataset in datasets:
        if dataset is None:
            raise RuntimeError("Frozen T4 export requires both held-in and held-out calibration sets")
        for session_name in sorted(dataset.calib_trialized_neural_features):
            if session_name not in session_to_tag:
                raise RuntimeError(f"No FALCON dataset tag for {session_name}")
            dataset_tag = session_to_tag[session_name]
            if dataset_tag in identity_by_tag:
                raise RuntimeError(f"Duplicate FALCON dataset tag {dataset_tag}")
            calibration_np = np.asarray(
                dataset.calib_trialized_neural_features[session_name][
                    :CALIBRATION_N_TRIALS
                ],
                dtype=np.float32,
            )
            if calibration_np.shape != (33, 100, 96):
                raise RuntimeError(
                    f"Unexpected first-33 calibration shape for {session_name}: "
                    f"{calibration_np.shape}"
                )
            side_np = np.asarray(
                dataset._native_t4_side_features(
                    session_name, 0, CALIBRATION_N_TRIALS
                ),
                dtype=np.float32,
            )
            if side_np.shape != (96, 4):
                raise RuntimeError(f"Unexpected T4 side-feature shape for {session_name}")

            calibration = torch.from_numpy(calibration_np).unsqueeze(0)
            side = torch.from_numpy(side_np).unsqueeze(0)
            with torch.inference_mode():
                identity = student.compute_identity(
                    calibration, side_features=side
                )
                # One deterministic 50-bin window per session is enough to
                # prove the algebraic cached-identity reduction.
                neural = torch.from_numpy(
                    np.asarray(dataset.neural_data[session_name][50:100], dtype=np.float32)
                ).unsqueeze(0)
                direct, _ = student(
                    neural, calib_trials=calibration, side_features=side
                )
                cached, _ = student(neural, identity=identity)
                decoder_only = manual_decode(student.decoder, neural, identity)
            direct_vs_identity = float((direct - cached).abs().max().item())
            direct_vs_decoder = float((direct - decoder_only).abs().max().item())
            max_direct_vs_identity = max(max_direct_vs_identity, direct_vs_identity)
            max_direct_vs_decoder = max(max_direct_vs_decoder, direct_vs_decoder)
            if direct_vs_identity != 0.0 or direct_vs_decoder != 0.0:
                raise RuntimeError(
                    f"Cached T4 deployment is not bit-exact for {session_name}: "
                    f"identity={direct_vs_identity}, decoder={direct_vs_decoder}"
                )

            identity_np = np.ascontiguousarray(
                identity.squeeze(0).cpu().numpy(), dtype=np.float32
            )
            if identity_np.shape != (96, 50) or not np.isfinite(identity_np).all():
                raise RuntimeError(f"Invalid cached identity for {session_name}")
            identity_by_tag[dataset_tag] = identity_np
            session_records[session_name] = {
                "dataset_tag": dataset_tag,
                "calibration_shape": list(calibration_np.shape),
                "calibration_sha256": sha256_array(calibration_np),
                "t4_side_shape": list(side_np.shape),
                "t4_side_sha256": sha256_array(side_np),
                "identity_shape": list(identity_np.shape),
                "identity_sha256": sha256_array(identity_np),
                "direct_vs_cached_identity_max_abs": direct_vs_identity,
                "direct_vs_decoder_only_max_abs": direct_vs_decoder,
            }

    if len(identity_by_tag) != EXPECTED_SESSION_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SESSION_COUNT} M2 session identities, got {len(identity_by_tag)}"
        )
    if set(session_to_tag) != set(session_records):
        missing = sorted(set(session_to_tag).symmetric_difference(session_records))
        raise RuntimeError(f"Calibration/session coverage mismatch: {missing}")

    decoder = student.decoder.cpu().eval()
    for parameter in decoder.parameters():
        parameter.requires_grad = False
    payload = {
        "schema_version": "e8_t4_m2_cached_identity_v1",
        "task": task_config.task,
        "decoder": decoder,
        "identity_by_dataset_tag": identity_by_tag,
        "window_size": int(decoder.window_size),
        "behavior_scaling_factor": 5.0,
        "smooth_observations": False,
        "metadata": {
            **metadata,
            "screen_id": "m2_spint_t4_mainline_fp32_v1",
            "arm": "t4",
            "seed": 42,
            "best_epoch": 2,
            "selection_metric": "val_heldin/r2_mean",
            "selection_metric_value": 0.6472744345664978,
            "calibration_selection": "chronological_first_33",
            "calibration_uses_target_labels": True,
            "online_state": "cached_E[N,50]",
            "online_backward_pass": False,
            "session_records": session_records,
            "max_direct_vs_cached_identity_abs": max_direct_vs_identity,
            "max_direct_vs_decoder_only_abs": max_direct_vs_decoder,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    receipt = {
        "schema_version": "e8_t4_m2_export_receipt_v1",
        "payload_path": str(output),
        "payload_bytes": output.stat().st_size,
        "payload_sha256": sha256_file(output),
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "calibration_n_trials": CALIBRATION_N_TRIALS,
        "session_count": len(identity_by_tag),
        "dataset_tags": sorted(identity_by_tag),
        "max_direct_vs_cached_identity_abs": max_direct_vs_identity,
        "max_direct_vs_decoder_only_abs": max_direct_vs_decoder,
        "session_records": session_records,
    }
    receipt_path = output.with_suffix(".receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    receipt = export_payload(args.output.resolve(), force=args.force)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
