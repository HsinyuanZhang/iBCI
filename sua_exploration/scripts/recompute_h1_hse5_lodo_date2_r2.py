#!/usr/bin/env python3
"""Independent batch-29 prediction-level audit for date-2 H-SE5 terminal R².

The implementation deliberately does not import the terminal evaluator.  It
recreates the locked target inputs from the immutable source authority and
uses a different state digest and R² routine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys
from typing import Any, Mapping


WORKSPACE = Path(__file__).resolve().parents[2]
SPINT = WORKSPACE / "SPINT-main"
for directory in (WORKSPACE, SPINT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from src.data.h1_sparse_event_endpoint_dated import (
    H1SparseEventDatedDataModule,
    build_dated_sparse_target_dataset,
)
from src.data.h1_sparse_event_source_snapshot_dated import load_snapshot
SCHEMA = "h1_hse5_lodo_date2_terminal_evaluation_v1"
CHECKPOINT_SCHEMA = "h1_sparse_event_endpoint_dated_h32_terminal_checkpoint_v1"
OUTER_DATE = "19250108"
QUERY_SHA = "b0cd153750cb484af1237b7af1861600aca69a9b201244c22d140b42b1da7f6e"
TARGETS = ("ses-19250108T110520", "ses-19250108T111022", "ses-19250108T111455")
TARGET_SAMPLES = {"ses-19250108T110520": 8330, "ses-19250108T111022": 2508, "ses-19250108T111455": 2269}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def model_digest(state: Mapping[str, Any]) -> str:
    """Independent state hashing scheme (names, shapes, dtype, raw tensor bytes)."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def independent_r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Equivalent pooled multioutput R² without calling the terminal routine."""
    observed = np.asarray(y, dtype=np.float64)
    estimated = np.asarray(y_hat, dtype=np.float64)
    need(observed.ndim == 2 and observed.shape == estimated.shape, "invalid prediction/target shapes")
    means = np.add.reduce(observed, axis=0, dtype=np.float64) / float(observed.shape[0])
    error_sum = np.add.reduce(np.ravel((estimated - observed) * (estimated - observed)), dtype=np.float64)
    variance_sum = np.add.reduce(np.ravel((observed - means) * (observed - means)), dtype=np.float64)
    need(np.isfinite(error_sum) and np.isfinite(variance_sum) and variance_sum > 0.0, "R2 undefined")
    return float((variance_sum - error_sum) / variance_sum)


def file_sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_model(checkpoint_path: Path, expected: Mapping[str, Any], device: torch.device):
    checkpoint_path = checkpoint_path.resolve()
    need(file_sha(checkpoint_path) == expected["sha256"], "checkpoint file SHA differs from terminal receipt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("h1_sparse_event_endpoint_dated")
    need(
        isinstance(metadata, Mapping)
        and metadata == expected["metadata"]
        and metadata.get("schema") == CHECKPOINT_SCHEMA
        and checkpoint.get("epoch") == 49,
        "checkpoint metadata/epoch differs from terminal receipt",
    )
    config_path = checkpoint_path.parents[2] / ".hydra/config.yaml"
    need(config_path.is_file() and file_sha(config_path) == metadata["config_sha256"], "checkpoint config binding drift")
    config = OmegaConf.load(config_path)
    model = hydra.utils.instantiate(config.model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def score(checkpoint_path: Path, expected: Mapping[str, Any], dataset: Any, device: torch.device, batch_size: int) -> dict[str, Any]:
    model = load_model(checkpoint_path, expected, device)
    before = model_digest(model.state_dict())
    predicted: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    recording_names: list[str] = []
    batch_sizes: list[int] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    with torch.no_grad():
        for neural, target, identity, names, carrier in loader:
            output = model(
                neural.to(device=device, dtype=torch.float32),
                calib_trialized_neural_features=identity.to(device=device, dtype=torch.float32),
                carrier=carrier.to(device=device, dtype=torch.float32),
            )
            if model.hparams.decode_last_timestep_only:
                output, target = output[:, -1:, :], target[:, -1:, :]
            if model.hparams.predict_scaled_behavior:
                output = output / model.hparams.behavior_scaling_factor
            predicted.append(output[:, -1].cpu().numpy())
            observed.append(target[:, -1].cpu().numpy())
            recording_names.extend(str(name) for name in names)
            batch_sizes.append(int(output.shape[0]))
    after = model_digest(model.state_dict())
    need(before == after, "model state changed during independent target scoring")
    prediction = np.concatenate(predicted, axis=0)
    truth = np.concatenate(observed, axis=0)
    names = np.asarray(recording_names)
    need(prediction.shape == truth.shape == (13107, 7), "prediction array shape drift")
    per_session: dict[str, Any] = {}
    for session in TARGETS:
        use = names == session
        need(int(use.sum()) == TARGET_SAMPLES[session], f"{session}: sample count drift")
        per_session[session] = {"samples": int(use.sum()), "r2": independent_r2(truth[use], prediction[use])}
    return {
        "pooled_r2": independent_r2(truth, prediction), "samples": int(len(dataset)),
        "batches": len(batch_sizes), "last_batch_size": batch_sizes[-1], "batch_size": batch_size,
        "per_session": per_session, "state_sha256_before": before, "state_sha256_after": after,
        "state_immutable": True, "query_window_indices_sha256": dataset.window_indices_sha256,
    }


def compare(observed: Mapping[str, Any], terminal: Mapping[str, Any], label: str, tolerance: float) -> None:
    need(observed["samples"] == terminal["samples"] == 13107, f"{label}: sample mismatch")
    need(abs(float(observed["pooled_r2"]) - float(terminal["pooled_r2"])) <= tolerance, f"{label}: pooled R2 mismatch")
    for session in TARGETS:
        need(observed["per_session"][session]["samples"] == terminal["per_session"][session]["samples"],
             f"{label}/{session}: sample mismatch")
        need(abs(float(observed["per_session"][session]["r2"]) - float(terminal["per_session"][session]["r2"])) <= tolerance,
             f"{label}/{session}: R2 mismatch")


def run(args: argparse.Namespace) -> dict[str, Any]:
    receipt_path = args.receipt.resolve()
    terminal = json.loads(receipt_path.read_text(encoding="utf-8"))
    need(terminal.get("schema") == SCHEMA and terminal.get("outer_date") == OUTER_DATE, "not a date-2 terminal receipt")
    need(stat.S_IMODE(receipt_path.stat().st_mode) == 0o444, "terminal receipt is not immutable")
    snapshot = load_snapshot(Path(terminal["source_snapshot"]["receipt"]))
    source_item = terminal["source_snapshot"]
    need(snapshot.receipt_sha256 == source_item["receipt_sha256"] and snapshot.snapshot_sha256 == source_item["snapshot_sha256"]
         and snapshot.manifest_sha256 == source_item["manifest_sha256"], "source snapshot binding drift")
    source = H1SparseEventDatedDataModule(
        task="h1", data_dir=str(args.data_dir.resolve()), cache_dir=str(args.source_cache_dir.resolve()),
        fold_date=OUTER_DATE, source_snapshot_receipt=str(snapshot.receipt_path),
    )
    source.setup("fit")
    need(source.pilot_manifest_sha256 == source_item["training_source_manifest_sha256"],
         "independent source module manifest differs from terminal binding")
    dataset = build_dated_sparse_target_dataset(data_dir=args.data_dir, source_module=source)
    need(tuple(dataset.target_sessions) == TARGETS and len(dataset) == 13107 and dataset.window_indices_sha256 == QUERY_SHA,
         "independent target query contract mismatch")
    device = torch.device(args.device)
    full = score(Path(terminal["checkpoints"]["full"]["path"]), terminal["checkpoints"]["full"], dataset.with_intervention("full"), device, args.batch_size)
    zero = score(Path(terminal["checkpoints"]["zero5"]["path"]), terminal["checkpoints"]["zero5"], dataset.with_intervention("full"), device, args.batch_size)
    compare(full, terminal["metrics"]["full_same_checkpoint_interventions"]["full"], "Full", args.tolerance)
    compare(zero, terminal["metrics"]["independently_trained_zero5"], "independent Zero5", args.tolerance)
    return {
        "schema": "h1_hse5_lodo_date2_independent_prediction_recompute_v1", "status": "PASS",
        "terminal_receipt": str(receipt_path), "terminal_receipt_sha256": file_sha(receipt_path),
        "independent_batch_size": args.batch_size, "absolute_r2_tolerance": args.tolerance,
        "full": full, "independently_trained_zero5": zero,
        "full_minus_independently_trained_zero5": full["pooled_r2"] - zero["pooled_r2"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--data-dir", type=Path, default=SPINT / "data/000954")
    parser.add_argument("--source-cache-dir", type=Path, default=SPINT / "pilot_artifacts/h1_hse5_lodo_19250108/shared_source_cache")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=29)
    parser.add_argument("--tolerance", type=float, default=2e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    need(args.batch_size == 29, "independent audit is fixed to batch size 29")
    need(0.0 < args.tolerance <= 1e-4, "invalid R2 tolerance")
    body = run(args)
    text = json.dumps(body, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        need(not output.exists(), f"refusing to overwrite independent audit: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        output.chmod(0o444)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
