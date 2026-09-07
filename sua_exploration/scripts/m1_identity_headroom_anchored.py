#!/usr/bin/env python3
"""Anchored, forward-only M1 identity-headroom diagnostic.

The script first reproduces the sealed strict post-M10 held-in score.  It only
evaluates the literal-zero identity intervention when that anchor is within
the predeclared tolerance.  No optimizer, gradient, backward pass, formal
held-out data, or minival values are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torchmetrics import R2Score


WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = WORKSPACE / "streaming_calibration_exp"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


SEALED_R2 = 0.6622881293296814
ANCHOR_TOLERANCE = 1.0e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    centered = truth - truth.mean(axis=0, keepdims=True)
    denominator = float(np.square(centered).sum())
    if denominator <= 0.0:
        raise ValueError("M1 R2 denominator is not positive")
    return float(1.0 - np.square(truth - estimate).sum() / denominator)


def _predict(
    model: Any, dataset: Any, *, device: torch.device, zero_identity: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    # Match the sealed Lightning evaluator's batch size and torchmetrics state
    # accumulation for the anchor.  The paired diagnostic also retains a
    # float64 direct R2 below so the numerical convention is explicit.
    # The sealed endpoint uses SessionBatchSampler, which drops each
    # session-local remainder.  With the fold-0 query this evaluates 26,496 of
    # 26,517 eligible windows (the final 21 are intentionally absent).
    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=0)
    streaming_r2 = R2Score(multioutput="variance_weighted").to(device)
    prediction_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    session_rows: list[str] = []
    with torch.inference_mode():
        for neural, target, calibration, sessions, side in loader:
            neural = neural.to(device=device, dtype=torch.float32)
            calibration = calibration.to(device=device, dtype=torch.float32)
            side = side.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            identity = model.student.compute_identity(calibration, side_features=side)
            if zero_identity:
                identity = torch.zeros_like(identity)
            prediction, _ = model.student(neural, identity=identity)
            if model._decode_last_timestep_only:
                prediction, target = prediction[:, -1:, :], target[:, -1:, :]
            if model._predict_scaled_behavior:
                prediction = prediction / model._behavior_scaling_factor
            prediction_rows.append(prediction[:, -1, :].cpu().numpy())
            target_rows.append(target[:, -1, :].cpu().numpy())
            session_rows.extend(str(value) for value in sessions)
            streaming_r2.update(prediction[:, -1, :], target[:, -1, :])
    return (
        np.concatenate(target_rows), np.concatenate(prediction_rows), session_rows,
        float(streaming_r2.compute().cpu()),
    )


def _paired_metrics(
    target: np.ndarray, full: np.ndarray, zero: np.ndarray, session_rows: list[str],
) -> dict[str, Any]:
    per_session: dict[str, Any] = {}
    for session in sorted(set(session_rows)):
        mask = np.asarray([value == session for value in session_rows], dtype=bool)
        per_session[session] = {
            "samples": int(mask.sum()),
            "r2_full": _r2(target[mask], full[mask]),
            "r2_zero_identity": _r2(target[mask], zero[mask]),
        }
        per_session[session]["identity_limitedness"] = (
            per_session[session]["r2_full"] - per_session[session]["r2_zero_identity"]
        )
    return {
        "samples": int(target.shape[0]),
        "r2_full": _r2(target, full),
        "r2_zero_identity": _r2(target, zero),
        "identity_limitedness": _r2(target, full) - _r2(target, zero),
        "per_session": per_session,
    }


def run(
    *, config_path: Path, checkpoint_path: Path, output_path: Path, device_name: str,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    cfg = OmegaConf.load(config_path)
    # The resolved Hydra file retains the original run's project-relative
    # paths.  Bind the same canonical dataset explicitly so invocation from
    # the workspace root cannot silently point one directory too high.
    cfg.data.data_dir = str((WORKSPACE / "SPINT-main/data/000941").resolve())
    if str(cfg.data.task).lower() != "m1" or int(cfg.data.heldin_query_start_trial) != 10:
        raise ValueError("diagnostic requires the sealed M1 strict post-M10 config")
    if bool(cfg.data.include_heldout_in_fit) or bool(cfg.data.include_heldout_in_test):
        raise ValueError("formal held-out paths must remain disabled")
    if bool(cfg.model.predict_scaled_behavior) or float(cfg.model.behavior_scaling_factor) != 1.0:
        raise ValueError("sealed M1 anchor has no behavior rescaling")

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("test")
    model = hydra.utils.instantiate(cfg.model)
    model.setup("test")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    device = torch.device(device_name)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = _state_sha256(model)
    target, full, sessions, reproduced_r2 = _predict(
        model, datamodule.val_heldin_dataset, device=device, zero_identity=False,
    )
    difference = float(reproduced_r2 - SEALED_R2)
    anchor_pass = abs(difference) <= ANCHOR_TOLERANCE
    metrics: dict[str, Any] = {"r2_full": reproduced_r2}
    if anchor_pass:
        zero_target, zero, zero_sessions, _zero_streaming_r2 = _predict(
            model, datamodule.val_heldin_dataset, device=device, zero_identity=True,
        )
        if zero_sessions != sessions or not np.array_equal(zero_target, target):
            raise RuntimeError("zero-identity pass changed endpoint ordering or targets")
        metrics = _paired_metrics(target, full, zero, sessions)
    after = _state_sha256(model)
    result = {
        "schema": "m1_identity_headroom_anchored_forward_only_v1",
        "status": "PASS_ANCHOR_AND_ZERO_IDENTITY_SCORED" if anchor_pass else "ANCHOR_NOT_REPRODUCED",
        "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path)},
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "endpoint": {
            "scope": "held-in-calib source LOSO fold0, strict post-support trials [10,210)",
            "support_trials": [0, 10],
            "query_trials": [10, 210],
            "formal_heldout_opened": False,
            "minival_values_used": False,
        },
        "anchor": {
            "sealed_r2": SEALED_R2,
            "reproduced_r2": reproduced_r2,
            "difference": difference,
            "absolute_tolerance_frozen_before_run": ANCHOR_TOLERANCE,
            "pass": anchor_pass,
        },
        "metrics": metrics,
        "metric_conventions": {
            "anchor": (
                "torchmetrics.R2Score(multioutput=variance_weighted), batch_size=32, "
                "session-local incomplete batch dropped exactly as sealed SessionBatchSampler"
            ),
            "paired_diagnostic": "float64 direct variance-weighted R2",
            "device": str(device),
        },
        "model_state": {"before": before, "after": after, "unchanged": before == after},
        "deployment_updates": {"optimizer_steps": 0, "backward_steps": 0},
        "interpretation": "same-checkpoint forward-only diagnostic, non-routing",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_run = PROJECT / "logs/train/runs/2026-08-06-17-21-50-020564_rid-m1_afc4_emg_full_fold0_q3_dev12_f0_s42"
    parser.add_argument("--config", type=Path, default=default_run / ".hydra/config.yaml")
    parser.add_argument("--checkpoint", type=Path, default=default_run / "checkpoints/best_ckpt/epoch_011.ckpt")
    parser.add_argument(
        "--output", type=Path,
        default=WORKSPACE / "sua_exploration/results/m1_identity_headroom_anchored_v2/result.json",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Inference device. Use the original CUDA numerical path for an exact sealed-score replay.",
    )
    args = parser.parse_args()
    print(json.dumps(run(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device_name=args.device,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
