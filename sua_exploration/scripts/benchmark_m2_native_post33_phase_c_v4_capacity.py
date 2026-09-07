#!/usr/bin/env python3
"""Exact-shape synthetic Phase-C capacity benchmark; no dataset or scorer import."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    return {"canonical_path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _measure(
    function: Callable[[], None], *, device: torch.device,
    backward: bool, optimizer_step: bool, batch_size: int,
) -> dict[str, object]:
    for _ in range(5):
        function()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(20):
        torch.cuda.synchronize(device)
        start = time.perf_counter_ns()
        function()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter_ns() - start) / 1.0e6)
    return {
        "batch_size": batch_size,
        "warmup_repeats": 5,
        "timed_repeats": 20,
        "cuda_synchronize_before_and_after": True,
        "backward_executed": backward,
        "optimizer_step_executed": optimizer_step,
        "zero_grad_executed": optimizer_step,
        "timing_ms": {
            "median": statistics.median(samples),
            "p95": _percentile(samples, 0.95),
            "samples": samples,
        },
        "peak_device_memory_bytes": {
            "allocated": int(torch.cuda.max_memory_allocated(device)),
            "reserved": int(torch.cuda.max_memory_reserved(device)),
        },
    }


def _decoder() -> torch.nn.Module:
    from src.models.components.spint import SpintModel
    model = SpintModel(
        model_dim=512, num_covariates=2, window_size=50, num_heads=64,
        num_layers=1, num_id_layers=3, use_learnable_id=True,
        learnable_id_type="mlp", learnable_rep=True, dropout_rate=0.0,
        dynamic_dropout=True, dynamic_dropout_low=0.0,
        dynamic_dropout_high=1.0, tf_drop_rate=0.1,
        readin_layer_type="mlp",
    )
    model.fc_id_in(torch.zeros(1, 100))
    return model


def _spint_identity(model: torch.nn.Module, calibration: torch.Tensor) -> torch.Tensor:
    trials = calibration.permute(0, 1, 3, 2)
    return model.fc_id_out(model.fc_id_in(trials).mean(dim=1))


def _spint_decode(model: torch.nn.Module, neural: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
    src = neural.permute(0, 2, 1) + identity
    src = model.fc_in(src)
    rep = model.fc_in(model.rep).to(src)
    transformed, _ = model.transformer(rep.repeat(src.shape[0], 1, 1), src)
    return model.fc_out(transformed).permute(0, 2, 1)


def _spint_workloads(device: torch.device) -> dict[str, dict[str, object]]:
    model = _decoder().to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=5.0e-5, weight_decay=0.0)
    calibration = torch.randn(32, 33, 100, 96, device=device)
    neural = torch.randn(32, 50, 96, device=device)
    target = torch.randn(32, 1, 2, device=device)

    def train_step() -> None:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(neural, calib_trialized_neural_features=calibration)[:, -1:, :] / 5.0
        F.mse_loss(prediction, target).backward()
        optimizer.step()

    def validation() -> None:
        model.eval()
        with torch.no_grad():
            model(neural, calib_trialized_neural_features=calibration)

    calibration_one = calibration[:1]
    neural_one = neural[:1]

    def support() -> None:
        model.eval()
        with torch.no_grad():
            _spint_identity(model, calibration_one)

    model.eval()
    with torch.no_grad():
        cached = _spint_identity(model, calibration_one)

    def inference() -> None:
        with torch.no_grad():
            _spint_decode(model, neural_one, cached)

    return {
        "source_train_forward_backward": _measure(train_step, device=device, backward=True, optimizer_step=True, batch_size=32),
        "source_validation_forward": _measure(validation, device=device, backward=False, optimizer_step=False, batch_size=32),
        "support_calibration_finalize": _measure(support, device=device, backward=False, optimizer_step=False, batch_size=1),
        "cached_identity_streaming_inference": _measure(inference, device=device, backward=False, optimizer_step=False, batch_size=1),
    }


def _t4_workloads(device: torch.device) -> dict[str, dict[str, object]]:
    from src.data.falcon_t4_features import t4_from_trial_sums
    from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder
    from src.models.components.streaming_spint import StreamingSpintModel

    decoder = _decoder().to(device=device, dtype=torch.float32)
    encoder = SideFeatureEarlyPoolEncoder(
        trial_length=100, window_size=50, hidden_dim=64, side_dim=4,
        electrode_embed_dim=0, num_electrodes=0, num_post_layers=3,
    ).to(device=device, dtype=torch.float32)
    student = StreamingSpintModel(decoder=decoder, id_encoder=encoder).to(device)
    student.freeze_decoder()
    teacher = _decoder().to(device=device, dtype=torch.float32).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.Adam(student.trainable_encoder_parameters(), lr=1.0e-4, weight_decay=0.0)
    calibration = torch.randn(32, 33, 100, 96, device=device)
    neural = torch.randn(32, 50, 96, device=device)
    side = torch.randn(32, 96, 4, device=device)
    target = torch.randn(32, 1, 2, device=device)

    def train_step() -> None:
        student.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, identity = student(neural, calib_trials=calibration, side_features=side)
        prediction = prediction[:, -1:, :] / 5.0
        with torch.no_grad():
            teacher_prediction = teacher(neural, calib_trialized_neural_features=calibration)[:, -1:, :] / 5.0
            teacher_identity = _spint_identity(teacher, calibration)
        loss = (
            F.mse_loss(prediction, target)
            + F.mse_loss(prediction, teacher_prediction)
            + 0.1 * F.mse_loss(identity, teacher_identity)
        )
        loss.backward()
        optimizer.step()

    def validation() -> None:
        student.eval()
        with torch.no_grad():
            student(neural, calib_trials=calibration, side_features=side)
            teacher(neural, calib_trialized_neural_features=calibration)
            _spint_identity(teacher, calibration)

    calibration_one = calibration[:1]
    neural_one = neural[:1]
    angles = np.linspace(-np.pi, np.pi, 33, endpoint=False, dtype=np.float64)
    sums = np.abs(np.random.RandomState(20260804).normal(size=(33, 96))) * 100.0
    lengths = np.full(33, 100.0, dtype=np.float64)

    def support() -> None:
        fitted = t4_from_trial_sums(sums, lengths, angles, source="synthetic-capacity")
        fitted_tensor = torch.as_tensor(fitted, device=device).unsqueeze(0)
        student.eval()
        with torch.no_grad():
            student.compute_identity(calibration_one, side_features=fitted_tensor)

    student.eval()
    fitted = t4_from_trial_sums(sums, lengths, angles, source="synthetic-capacity")
    fitted_tensor = torch.as_tensor(fitted, device=device).unsqueeze(0)
    with torch.no_grad():
        cached = student.compute_identity(calibration_one, side_features=fitted_tensor)

    def inference() -> None:
        with torch.no_grad():
            student.decode_with_identity(neural_one, cached)

    return {
        "source_train_forward_backward": _measure(train_step, device=device, backward=True, optimizer_step=True, batch_size=32),
        "source_validation_forward": _measure(validation, device=device, backward=False, optimizer_step=False, batch_size=32),
        "support_calibration_finalize": _measure(support, device=device, backward=False, optimizer_step=False, batch_size=1),
        "cached_identity_streaming_inference": _measure(inference, device=device, backward=False, optimizer_step=False, batch_size=1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("spint", "t4"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = ROOT / ("SPINT-main" if args.arm == "spint" else "streaming_calibration_exp")
    sys.path.insert(0, str(project))
    torch.manual_seed(20260804)
    np.random.seed(20260804)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("capacity benchmark requires a named CUDA device")
    torch.cuda.set_device(device)
    workloads = _spint_workloads(device) if args.arm == "spint" else _t4_workloads(device)
    properties = torch.cuda.get_device_properties(device)
    payload = {
        "schema": "m2_post33_phase_c_synthetic_capacity_benchmark_v4",
        "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
        "phase_id": "PHASE_C_V4",
        "arm": args.arm,
        "synthetic_capacity_only": True,
        "production_latency_claim_permitted": False,
        "source_data_loaded": False,
        "outer_data_loaded": False,
        "scorer_imported": False,
        "formal_data_accessed": False,
        "dtype": "torch.float32",
        "batch_size": 32,
        "reference_shapes": {
            "neural": [32, 50, 96],
            "calibration": [32, 33, 100, 96],
            "side_features": ([32, 96, 4] if args.arm == "t4" else None),
        },
        "device": {
            "requested": args.device,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "workloads": workloads,
        "source_bindings": {
            "model_config": _metadata(
                ROOT / ("SPINT-main/configs/model/falcon_m2_post33_confirm_v4.yaml" if args.arm == "spint" else "streaming_calibration_exp/configs/model/streaming_b3s_t4_post33_exact_v4.yaml")
            ),
            "decoder_source": _metadata(project / "src/models/components/spint.py"),
            "benchmark": _metadata(Path(__file__)),
            **({
                "streaming_model_source": _metadata(project / "src/models/components/streaming_spint.py"),
                "encoder_source": _metadata(project / "src/models/components/streaming_encoders.py"),
                "t4_estimator_source": _metadata(project / "src/data/falcon_t4_features.py"),
            } if args.arm == "t4" else {}),
        },
    }
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
