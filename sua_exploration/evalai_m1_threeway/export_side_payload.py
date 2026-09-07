#!/usr/bin/env python3
"""Export frozen M1 T4/D4 candidates as decoder plus cached session identities.

The exporter uses only public calibration NWBs.  It binds the selected LOSO
checkpoint, its train-session-only side-feature normalization, the common
epoch-19 M1 decoder, and all seven public M1 calibration session tags.  The
official runtime receives only the frozen decoder and cached E[N,50] tensors.
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
DATA_DIR = ROOT / "SPINT-main/data/000941"
TEACHER_CHECKPOINT = (
    ROOT
    / "SPINT-main/logs/train/runs/2026-07-21-19-11-01/checkpoints/best_ckpt/epoch_019.ckpt"
)
EXPECTED_TEACHER_SHA256 = "c81a2bbd860452e6186a9ecf55c0b747da61baef4fae3212f61521be68cc5ac2"
EXPECTED_SESSION_COUNT = 7
CALIBRATION_N_TRIALS = 10
CHANNELS = 64
TRIAL_BINS = 1024
IDENTITY_DIM = 100
SESSION_PATTERN = re.compile(r"(ses-\d{8})")

ARM_SPECS = {
    "t4": {
        "run": "m1_clean_selection_v1_t4_m1_f1_s42_20260801_192033",
        "checkpoint_sha256": "b5cc6d28d9cb17782a234c3b744f827795c6bfaccb766855bebd351efd3529fb",
        "normalization_key": "native_t4_normalization",
        "normalization_sha256": "da187173a51fdfa2b59b52d886dae661a5a3bb8a1f1a312d0b4aa81035bebc0f",
        "best_epoch": 6,
        "selection_metric_value": 0.7444179058074951,
        "labels": "trials.tgt_loc target direction",
    },
    "d4": {
        "run": "m1_d4_pilot_v1_d4_m1_f1_s42_20260802_120434",
        "checkpoint_sha256": "28530a7cd61f08edc9cf8f41cf5c7391f6150de72fc196ab3e38814bb698b11d",
        "normalization_key": "native_d4_normalization",
        "normalization_sha256": "ba03ba1099868861f5348e8d1246285284cc238d29360539c0fbab3adb7741d8",
        "best_epoch": 1,
        "selection_metric_value": None,
        "labels": "trials.obj_id categorical label",
    },
}


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


def manual_decode(
    decoder: torch.nn.Module, neural: torch.Tensor, identity: torch.Tensor
) -> torch.Tensor:
    source = neural.permute(0, 2, 1) + identity
    source = decoder.fc_in(source)
    query = decoder.fc_in(decoder.rep).to(source)
    transformed, _ = decoder.transformer(
        query.repeat(source.shape[0], 1, 1), source
    )
    return decoder.fc_out(transformed).permute(0, 2, 1)


def calibration_file_map(data_dir: Path, task_config: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in sorted(data_dir.rglob("*calib*.nwb")):
        match = SESSION_PATTERN.search(path.name)
        if match is None:
            raise RuntimeError(f"Cannot derive session name from {path}")
        session_name = match.group(1)
        tag = task_config.hash_dataset(path.stem)
        if session_name in mapping and mapping[session_name] != tag:
            raise RuntimeError(f"Conflicting dataset tags for {session_name}")
        mapping[session_name] = tag
    return mapping


def load_candidate(arm: str) -> tuple[Any, Any, Any, dict[str, Any]]:
    spec = ARM_SPECS[arm]
    run_root = STREAMING_ROOT / "outputs/streaming_calibration" / spec["run"]
    checkpoint = run_root / "checkpoints/best.ckpt"
    resolved_config = run_root / "resolved_config.yaml"
    split_manifest_path = run_root / "split_manifest.json"
    for required in (
        checkpoint,
        resolved_config,
        split_manifest_path,
        TEACHER_CHECKPOINT,
        DATA_DIR,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    if sha256_file(checkpoint) != spec["checkpoint_sha256"]:
        raise RuntimeError(f"Frozen {arm.upper()} checkpoint drift")
    if sha256_file(TEACHER_CHECKPOINT) != EXPECTED_TEACHER_SHA256:
        raise RuntimeError("Frozen M1 teacher checkpoint drift")

    split_manifest = json.loads(split_manifest_path.read_text())
    normalization = split_manifest.get(spec["normalization_key"], {})
    if normalization.get("feature_group") != arm:
        raise RuntimeError(f"{arm.upper()} normalization arm mismatch")
    if normalization.get("sha256") != spec["normalization_sha256"]:
        raise RuntimeError(f"Frozen {arm.upper()} normalization drift")

    sys.path.insert(0, str(STREAMING_ROOT))
    from falcon_challenge.config import FalconConfig, FalconTask
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(resolved_config)
    if (
        str(config.data.task).lower() != "m1"
        or int(config.data.calibration_n_trials) != CALIBRATION_N_TRIALS
        or bool(config.data.random_calibration)
        or str(config.data.side_feature_group).lower() != arm
        or int(config.seed) != 42
        or int(config.data.loso_fold) != 1
    ):
        raise RuntimeError(f"Resolved config is not frozen M1 {arm.upper()} fold1/seed42")
    config.model.teacher_ckpt_path = str(TEACHER_CHECKPOINT)
    config.data.data_dir = str(DATA_DIR)
    # Export needs identities for the three public held-out calibration sessions.
    # This changes only which public calibration files are loaded at test setup;
    # the normalization remains fitted on the original three LOSO train sessions.
    config.data.include_heldout_in_test = True
    config.data.include_heldout_in_fit = False
    config.data.query_start_trial = 0
    config.data.num_workers = 0
    config.data.pin_memory = False

    model = instantiate(config.model)
    model.setup("test")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    if model.student is None or model.student.decoder_mode != "coupled":
        raise RuntimeError("Frozen candidate is not a coupled SPINT student")
    if not model.student._decoder_frozen:
        raise RuntimeError("Frozen decoder is unexpectedly trainable")

    data_module = instantiate(config.data)
    data_module.prepare_data()
    data_module.setup("test")
    actual = getattr(data_module, spec["normalization_key"])
    np.testing.assert_array_equal(
        np.asarray(actual["mean"], dtype=np.float32),
        np.asarray(normalization["mean"], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(actual["std"], dtype=np.float32),
        np.asarray(normalization["std"], dtype=np.float32),
    )
    task_config = FalconConfig(task=FalconTask.m1)
    metadata = {
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "teacher_checkpoint_sha256": EXPECTED_TEACHER_SHA256,
        "normalization_sha256": spec["normalization_sha256"],
        "normalization_mean": normalization["mean"],
        "normalization_std": normalization["std"],
        "normalization_train_sessions": normalization["train_sessions"],
        "resolved_config_sha256": sha256_file(resolved_config),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "run_root": str(run_root.relative_to(ROOT)),
    }
    return model, data_module, task_config, metadata


def export_payload(arm: str, output: Path, force: bool = False) -> dict[str, Any]:
    if output.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {output}")
    spec = ARM_SPECS[arm]
    model, data_module, task_config, metadata = load_candidate(arm)
    student = model.student
    assert student is not None
    session_to_tag = calibration_file_map(DATA_DIR, task_config)
    datasets = (
        data_module.train_dataset,
        data_module.val_heldin_dataset,
        data_module.val_heldout_dataset,
    )
    identity_by_tag: dict[str, np.ndarray] = {}
    session_records: dict[str, Any] = {}
    max_direct_vs_cached = 0.0
    max_direct_vs_decoder = 0.0
    side_method = f"_native_{arm}_side_features"

    for dataset in datasets:
        if dataset is None:
            raise RuntimeError("Export requires train, LOSO validation, and held-out calibration sets")
        for session_name in sorted(dataset.calib_trialized_neural_features):
            if session_name not in session_to_tag:
                raise RuntimeError(f"No FALCON tag for {session_name}")
            tag = session_to_tag[session_name]
            if tag in identity_by_tag:
                raise RuntimeError(f"Duplicate session/tag during export: {session_name}")
            calibration_np = np.asarray(
                dataset.calib_trialized_neural_features[session_name][:CALIBRATION_N_TRIALS],
                dtype=np.float32,
            )
            if calibration_np.shape != (CALIBRATION_N_TRIALS, TRIAL_BINS, CHANNELS):
                raise RuntimeError(
                    f"Unexpected calibration shape for {session_name}: {calibration_np.shape}"
                )
            side_np = np.asarray(
                getattr(dataset, side_method)(session_name, 0, CALIBRATION_N_TRIALS),
                dtype=np.float32,
            )
            if side_np.shape != (CHANNELS, 4):
                raise RuntimeError(f"Unexpected {arm.upper()} side shape for {session_name}")
            calibration = torch.from_numpy(calibration_np).unsqueeze(0)
            side = torch.from_numpy(side_np).unsqueeze(0)
            neural = torch.from_numpy(
                np.asarray(dataset.neural_data[session_name][100:200], dtype=np.float32)
            ).unsqueeze(0)
            if neural.shape != (1, 100, CHANNELS):
                raise RuntimeError(f"Unexpected audit neural window for {session_name}: {neural.shape}")
            with torch.inference_mode():
                identity = student.compute_identity(calibration, side_features=side)
                direct, _ = student(neural, calib_trials=calibration, side_features=side)
                cached, _ = student(neural, identity=identity)
                decoder_only = manual_decode(student.decoder, neural, identity)
            delta_cached = float((direct - cached).abs().max().item())
            delta_decoder = float((direct - decoder_only).abs().max().item())
            max_direct_vs_cached = max(max_direct_vs_cached, delta_cached)
            max_direct_vs_decoder = max(max_direct_vs_decoder, delta_decoder)
            if delta_cached != 0.0 or delta_decoder != 0.0:
                raise RuntimeError(
                    f"Cached deployment is not bit-exact for {session_name}: "
                    f"cached={delta_cached}, decoder={delta_decoder}"
                )
            identity_np = np.ascontiguousarray(identity.squeeze(0).numpy(), dtype=np.float32)
            if identity_np.shape != (CHANNELS, IDENTITY_DIM) or not np.isfinite(identity_np).all():
                raise RuntimeError(f"Invalid identity for {session_name}: {identity_np.shape}")
            identity_by_tag[tag] = identity_np
            session_records[session_name] = {
                "dataset_tag": tag,
                "calibration_sha256": sha256_array(calibration_np),
                f"{arm}_side_sha256": sha256_array(side_np),
                "identity_sha256": sha256_array(identity_np),
                "identity_shape": list(identity_np.shape),
                "direct_vs_cached_identity_max_abs": delta_cached,
                "direct_vs_decoder_only_max_abs": delta_decoder,
            }

    if len(identity_by_tag) != EXPECTED_SESSION_COUNT:
        raise RuntimeError(f"Expected 7 M1 sessions, got {len(identity_by_tag)}")
    if set(session_to_tag) != set(session_records):
        raise RuntimeError(
            "Calibration/session coverage mismatch: "
            f"{sorted(set(session_to_tag).symmetric_difference(session_records))}"
        )

    decoder = student.decoder.cpu().eval()
    for parameter in decoder.parameters():
        parameter.requires_grad = False
    payload = {
        "schema_version": "evalai_m1_cached_identity_v1",
        "task": task_config.task,
        "arm": arm,
        "decoder": decoder,
        "identity_by_dataset_tag": identity_by_tag,
        "window_size": int(decoder.window_size),
        "behavior_scaling_factor": 1.0,
        "smooth_observations": False,
        "metadata": {
            **metadata,
            "screen_id": "m1_clean_selection_v1" if arm == "t4" else "m1_d4_pilot_v1",
            "fold": 1,
            "seed": 42,
            "best_epoch": spec["best_epoch"],
            "selection_metric": "val_heldin/r2_mean",
            "selection_metric_value": spec["selection_metric_value"],
            "calibration_selection": "chronological_first_10",
            "calibration_label_source": spec["labels"],
            "calibration_uses_labels": True,
            "query_labels_used": False,
            "online_state": "cached_E[N,100]",
            "online_backward_pass": False,
            "session_records": session_records,
            "max_direct_vs_cached_identity_abs": max_direct_vs_cached,
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
        "schema_version": "evalai_m1_side_export_receipt_v1",
        "arm": arm,
        "payload_path": str(output),
        "payload_bytes": output.stat().st_size,
        "payload_sha256": sha256_file(output),
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "teacher_checkpoint_sha256": EXPECTED_TEACHER_SHA256,
        "calibration_n_trials": CALIBRATION_N_TRIALS,
        "session_count": len(identity_by_tag),
        "dataset_tags": sorted(identity_by_tag),
        "max_direct_vs_cached_identity_abs": max_direct_vs_cached,
        "max_direct_vs_decoder_only_abs": max_direct_vs_decoder,
        "session_records": session_records,
    }
    receipt_path = output.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARM_SPECS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    receipt = export_payload(args.arm, args.output.resolve(), force=args.force)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
