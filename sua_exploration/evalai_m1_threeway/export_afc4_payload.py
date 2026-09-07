#!/usr/bin/env python3
"""Export a clean all-source M1 AFC4 Full/B4 checkpoint for EvalAI.

The exporter is intentionally offline and explicit.  It accepts only the four
held-in calibration NWBs as the PCA/normalizer source and the three exact
held-out calibration NWBs as M10 support.  It never discovers or opens a
future/query/evaluation file, and it verifies direct-vs-cached deployment
equivalence before writing a payload.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
STREAMING_ROOT = ROOT / "streaming_calibration_exp"
if str(STREAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMING_ROOT))

SESSION_PATTERN = re.compile(r"(ses-\d{8})")
SOURCE_SESSIONS = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
HELDOUT_SESSIONS = ("ses-20121004", "ses-20121017", "ses-20121024")
CALIBRATION_N_TRIALS = 10
CHANNELS = 64
TRIAL_BINS = 1024
IDENTITY_DIM = 100
EXPECTED_SESSION_COUNT = 7


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


def _session_from_path(path: Path) -> str:
    match = SESSION_PATTERN.search(path.name)
    if match is None:
        raise RuntimeError(f"cannot derive M1 session from {path}")
    return match.group(1)


def calibration_file_map(data_dir: Path) -> OrderedDict[str, Path]:
    """Resolve exactly seven public calibration NWBs without a recursive search."""
    source_dir = data_dir / "sub-MonkeyL-held-in-calib"
    target_dir = data_dir / "sub-MonkeyL-held-out-calib"
    if not source_dir.is_dir() or not target_dir.is_dir():
        raise FileNotFoundError(f"expected public calibration directories below {data_dir}")
    source = sorted(source_dir.glob("*.nwb"))
    target = sorted(target_dir.glob("*.nwb"))
    if len(source) != len(SOURCE_SESSIONS) or len(target) != len(HELDOUT_SESSIONS):
        raise RuntimeError(
            "public M1 calibration coverage changed: "
            f"held-in={len(source)}, held-out={len(target)}"
        )
    mapping: OrderedDict[str, Path] = OrderedDict()
    for expected, paths in ((SOURCE_SESSIONS, source), (HELDOUT_SESSIONS, target)):
        for path in paths:
            session = _session_from_path(path)
            if session not in expected:
                raise RuntimeError(f"unexpected calibration session {session}: {path}")
            if session in mapping:
                raise RuntimeError(f"duplicate calibration session {session}")
            mapping[session] = path.resolve()
    if tuple(mapping) != SOURCE_SESSIONS + HELDOUT_SESSIONS:
        raise RuntimeError(f"unexpected calibration ordering: {tuple(mapping)}")
    return mapping


def _load_student(
    checkpoint: Path,
    resolved_config: Path,
    teacher_checkpoint: Path,
    data_dir: Path,
    arm: str,
) -> tuple[Any, Any]:
    if arm not in {"full", "b4"}:
        raise ValueError("AFC4 official payload supports only full and b4")
    for required in (checkpoint, resolved_config, teacher_checkpoint, data_dir):
        if not required.exists():
            raise FileNotFoundError(required)
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(resolved_config)
    required = {
        "task": "m1",
        "validation_protocol": "all_source",
        "calibration_n_trials": CALIBRATION_N_TRIALS,
        "random_calibration": False,
        "include_heldout_in_fit": False,
        "include_heldout_in_test": False,
        "query_start_trial": 0,
        "heldin_query_start_trial": 0,
        "heldin_query_end_trial": None,
    }
    if str(config.data.task).lower() != required["task"]:
        raise RuntimeError("resolved config is not native M1")
    if str(config.data.validation_protocol).lower() != required["validation_protocol"]:
        raise RuntimeError("resolved config is not all-source")
    for key in (
        "calibration_n_trials",
        "random_calibration",
        "include_heldout_in_fit",
        "include_heldout_in_test",
        "query_start_trial",
        "heldin_query_start_trial",
        "heldin_query_end_trial",
    ):
        if config.data[key] != required[key]:
            raise RuntimeError(f"resolved data contract drifted for {key}: {config.data[key]}")
    if int(config.trainer.max_epochs) != 12 or str(config.model.loss_mode) != "task_only":
        raise RuntimeError("resolved config is not fixed-12 task-only final student")
    if str(config.data.afc4_arm).lower() != arm:
        raise RuntimeError(f"resolved config arm mismatch: {config.data.afc4_arm}")

    config.model.teacher_ckpt_path = str(teacher_checkpoint.resolve())
    config.data.data_dir = str(data_dir.resolve())
    config.data.num_workers = 0
    config.data.pin_memory = False
    model = instantiate(config.model)
    # setup('fit') only builds the frozen common teacher/streaming student; the
    # exporter does not instantiate or run a datamodule lifecycle.
    model.setup("fit")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not isinstance(state.get("state_dict"), dict):
        raise RuntimeError("final checkpoint is not a Lightning state_dict checkpoint")
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    student = model.student
    if student is None or student.decoder_mode != "coupled":
        raise RuntimeError("final candidate is not a coupled SPINT student")
    if not student._decoder_frozen:
        raise RuntimeError("final candidate decoder is unexpectedly trainable")
    return model, config


def export_payload(
    *,
    arm: str,
    checkpoint: Path,
    resolved_config: Path,
    teacher_checkpoint: Path,
    data_dir: Path,
    output: Path,
    seed: int = 42,
    force: bool = False,
) -> dict[str, Any]:
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {output}")
    mapping = calibration_file_map(data_dir.resolve())
    source_paths = OrderedDict((name, mapping[name]) for name in SOURCE_SESSIONS)
    target_paths = [mapping[name] for name in HELDOUT_SESSIONS]

    from falcon_challenge.config import FalconConfig, FalconTask
    from src.data.falcon_m1_afc4_package import M1AFC4PackagePlan, load_public_m1_calibration_dataset

    model, config = _load_student(
        checkpoint.resolve(), resolved_config.resolve(), teacher_checkpoint.resolve(), data_dir.resolve(), arm
    )
    student = model.student
    assert student is not None
    plan = M1AFC4PackagePlan(source_paths, shuffle_seed=int(seed))
    for path in target_paths:
        plan.add_target(path)
    calibration_paths = list(source_paths.values()) + target_paths
    dataset = load_public_m1_calibration_dataset(calibration_paths)
    task_config = FalconConfig(task=FalconTask.m1)

    session_to_tag = OrderedDict(
        (name, task_config.hash_dataset(path.stem)) for name, path in mapping.items()
    )
    identity_by_tag: dict[str, np.ndarray] = {}
    session_records: dict[str, Any] = {}
    max_direct_vs_cached = 0.0
    max_direct_vs_decoder = 0.0
    with torch.inference_mode():
        for session_name, path in mapping.items():
            if session_name not in dataset.calib_trialized_neural_features:
                raise RuntimeError(f"calibration trialization omitted {session_name}")
            calibration_np = np.asarray(
                dataset.calib_trialized_neural_features[session_name][:CALIBRATION_N_TRIALS],
                dtype=np.float32,
            )
            if calibration_np.shape != (CALIBRATION_N_TRIALS, TRIAL_BINS, CHANNELS):
                raise RuntimeError(f"unexpected M10 calibration shape for {session_name}: {calibration_np.shape}")
            side_np = np.asarray(plan.normalized(session_name, arm=arm), dtype=np.float32)
            if side_np.shape != (CHANNELS, 4):
                raise RuntimeError(f"unexpected {arm.upper()} side shape for {session_name}: {side_np.shape}")
            calibration = torch.from_numpy(calibration_np).unsqueeze(0)
            side = torch.from_numpy(side_np).unsqueeze(0)
            neural_np = np.asarray(dataset.neural_data[session_name][100:200], dtype=np.float32)
            if neural_np.shape != (100, CHANNELS):
                raise RuntimeError(f"unexpected deployment window for {session_name}: {neural_np.shape}")
            neural = torch.from_numpy(neural_np).unsqueeze(0)
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
                    f"cached deployment is not exact for {session_name}: "
                    f"cached={delta_cached}, decoder={delta_decoder}"
                )
            identity_np = np.ascontiguousarray(identity.squeeze(0).numpy(), dtype=np.float32)
            if identity_np.shape != (CHANNELS, IDENTITY_DIM) or not np.isfinite(identity_np).all():
                raise RuntimeError(f"invalid identity for {session_name}: {identity_np.shape}")
            tag = session_to_tag[session_name]
            identity_by_tag[tag] = identity_np
            session_records[session_name] = {
                "dataset_tag": tag,
                "file": str(path),
                "file_sha256": sha256_file(path),
                "calibration_sha256": sha256_array(calibration_np),
                f"{arm}_side_sha256": sha256_array(side_np),
                "identity_sha256": sha256_array(identity_np),
                "identity_shape": list(identity_np.shape),
                "direct_vs_cached_identity_max_abs": delta_cached,
                "direct_vs_decoder_only_max_abs": delta_decoder,
                "future_query_values_read": False,
            }
    if len(identity_by_tag) != EXPECTED_SESSION_COUNT:
        raise RuntimeError(f"expected seven M1 calibration tags, got {len(identity_by_tag)}")

    decoder = student.decoder.cpu().eval()
    for parameter in decoder.parameters():
        parameter.requires_grad = False
    plan_receipt = plan.receipt(arm=arm)
    payload = {
        "schema_version": "evalai_m1_afc4_cached_identity_v1",
        "task": task_config.task,
        "arm": arm,
        "decoder": decoder,
        "identity_by_dataset_tag": identity_by_tag,
        "dataset_tags": sorted(identity_by_tag),
        "window_size": int(decoder.window_size),
        "behavior_scaling_factor": 1.0,
        "smooth_observations": False,
        "metadata": {
            "checkpoint_sha256": sha256_file(checkpoint),
            "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
            "resolved_config_sha256": sha256_file(resolved_config),
            "data_source_file_sha256": {
                name: sha256_file(path) for name, path in mapping.items()
            },
            "calibration_selection": "chronological_first_10",
            "calibration_n_trials": CALIBRATION_N_TRIALS,
            "target_query_values_read": False,
            "online_state": "cached_E[N,100]",
            "online_backward_pass": False,
            "arms_share_source_basis": True,
            "afc4_plan_receipt": plan_receipt,
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
        "schema_version": "evalai_m1_afc4_export_receipt_v1",
        "arm": arm,
        "payload_path": str(output),
        "payload_bytes": output.stat().st_size,
        "payload_sha256": sha256_file(output),
        "checkpoint_sha256": sha256_file(checkpoint),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "calibration_n_trials": CALIBRATION_N_TRIALS,
        "source_sessions": list(SOURCE_SESSIONS),
        "target_sessions": list(HELDOUT_SESSIONS),
        "session_count": len(identity_by_tag),
        "dataset_tags": sorted(identity_by_tag),
        "max_direct_vs_cached_identity_abs": max_direct_vs_cached,
        "max_direct_vs_decoder_only_abs": max_direct_vs_decoder,
        "target_query_values_read": False,
        "session_records": session_records,
    }
    output.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("full", "b4"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "SPINT-main/data/000941")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    receipt = export_payload(
        arm=args.arm,
        checkpoint=args.checkpoint,
        resolved_config=args.resolved_config,
        teacher_checkpoint=args.teacher_checkpoint,
        data_dir=args.data_dir,
        output=args.output.resolve(),
        seed=args.seed,
        force=args.force,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
