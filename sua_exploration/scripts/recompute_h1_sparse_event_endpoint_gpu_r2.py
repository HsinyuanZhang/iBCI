#!/usr/bin/env python3
"""Independently recompute terminal H-SE5/Zero5 prediction-level R2.

The terminal evaluator uses batch size 32.  This audit deliberately uses a
different fixed batch size and an independent model-state digest/R2 routine.
It evaluates only the correct H-SE5 input and the independently trained Zero5
arm; intervention arithmetic is covered by the structural receipt verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SPINT_ROOT = ROOT / "SPINT-main"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SPINT_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINT_ROOT))

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from src.data.h1_sparse_event_endpoint import H1SparseEventDataModule, build_sparse_target_dataset
from src.data.h1_sparse_event_source_snapshot import apply_snapshot_to_source_module, load_snapshot


QUERY_SHA = "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"
TARGETS = ("ses-19250101T111740", "ses-19250101T112404")


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def state_digest(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    need(truth.shape == prediction.shape and truth.ndim == 2, "R2 array shape mismatch")
    residual = float(np.square(truth - prediction).sum(dtype=np.float64))
    centered = truth - truth.mean(axis=0, keepdims=True, dtype=np.float64)
    total = float(np.square(centered).sum(dtype=np.float64))
    need(np.isfinite(residual) and np.isfinite(total) and total > 0, "R2 undefined")
    return 1.0 - residual / total


def instantiate_model(checkpoint_path: Path, device: torch.device):
    config_path = checkpoint_path.parents[2] / ".hydra/config.yaml"
    need(config_path.is_file(), f"missing resolved config {config_path}")
    config = OmegaConf.load(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    need(checkpoint.get("epoch") == 49, f"nonterminal checkpoint {checkpoint_path}")
    model = hydra.utils.instantiate(config.model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def score(checkpoint_path: Path, dataset, device: torch.device, batch_size: int) -> dict[str, Any]:
    model = instantiate_model(checkpoint_path, device)
    before = state_digest(model.state_dict())
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    sessions: list[str] = []
    batch_sizes: list[int] = []
    with torch.inference_mode():
        for neural, target, identity, names, carrier in DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
        ):
            output = model(
                neural.to(device=device, dtype=torch.float32),
                calib_trialized_neural_features=identity.to(device=device, dtype=torch.float32),
                carrier=carrier.to(device=device, dtype=torch.float32),
            )
            if model.hparams.decode_last_timestep_only:
                output = output[:, -1:, :]
                target = target[:, -1:, :]
            if model.hparams.predict_scaled_behavior:
                output = output / model.hparams.behavior_scaling_factor
            predictions.append(output[:, -1].cpu().numpy())
            truths.append(target[:, -1].cpu().numpy())
            sessions.extend(str(value) for value in names)
            batch_sizes.append(int(output.shape[0]))
    after = state_digest(model.state_dict())
    need(before == after, "model state mutated during independent scoring")
    prediction = np.concatenate(predictions)
    truth = np.concatenate(truths)
    need(prediction.shape == truth.shape == (len(dataset), 7), "prediction array shape mismatch")
    per_session: dict[str, Any] = {}
    session_array = np.asarray(sessions)
    for session in TARGETS:
        keep = session_array == session
        per_session[session] = {"samples": int(keep.sum()), "r2": r2(truth[keep], prediction[keep])}
    result = {
        "pooled_r2": r2(truth, prediction),
        "samples": int(len(dataset)),
        "batches": len(batch_sizes),
        "last_batch_size": batch_sizes[-1],
        "batch_size": batch_size,
        "per_session": per_session,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "state_immutable": True,
        "query_window_indices_sha256": dataset.window_indices_sha256,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def compare(observed: Mapping[str, Any], expected: Mapping[str, Any], label: str, tolerance: float) -> None:
    need(observed["samples"] == expected["samples"], f"{label}: pooled sample mismatch")
    need(abs(float(observed["pooled_r2"]) - float(expected["pooled_r2"])) <= tolerance,
         f"{label}: pooled R2 mismatch")
    for session in TARGETS:
        need(observed["per_session"][session]["samples"] == expected["per_session"][session]["samples"],
             f"{label}/{session}: sample mismatch")
        need(abs(float(observed["per_session"][session]["r2"]) -
                 float(expected["per_session"][session]["r2"])) <= tolerance,
             f"{label}/{session}: R2 mismatch")


def run(args: argparse.Namespace) -> dict[str, Any]:
    receipt_path = args.receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    need(receipt.get("schema") == "h1_sparse_event_endpoint_hse5_fold0_terminal_v1",
         "terminal receipt schema mismatch")
    device = torch.device(args.device)
    source = H1SparseEventDataModule(
        task="h1",
        data_dir=str(args.data_dir.resolve()),
        cache_dir=str(SPINT_ROOT / "pilot_artifacts/h1_sparse_event_endpoint/shared_source_cache"),
    )
    source.setup("fit")
    snapshot_entry = receipt.get("source_snapshot")
    need(isinstance(snapshot_entry, Mapping), "terminal receipt source snapshot binding missing")
    snapshot = load_snapshot(Path(snapshot_entry["receipt"]))
    need(snapshot.receipt_sha256 == snapshot_entry["receipt_sha256"] and
         snapshot.snapshot_sha256 == snapshot_entry["snapshot_sha256"] and
         snapshot.manifest_sha256 == receipt["checkpoints"]["full"]["metadata"]["source_manifest_sha256"],
         "independent source snapshot binding drift")
    apply_snapshot_to_source_module(source, snapshot)
    dataset = build_sparse_target_dataset(data_dir=args.data_dir, source_module=source)
    need(dataset.window_indices_sha256 == QUERY_SHA and len(dataset) == 8965,
         "independent target query contract mismatch")
    full_path = Path(receipt["checkpoints"]["full"]["path"]).resolve()
    zero_path = Path(receipt["checkpoints"]["zero"]["path"]).resolve()
    full = score(full_path, dataset.with_intervention("full"), device, args.batch_size)
    zero = score(zero_path, dataset.with_intervention("full"), device, args.batch_size)
    expected = receipt["metrics"]
    compare(full, expected["hse5_same_checkpoint_interventions"]["full"], "H-SE5", args.tolerance)
    compare(zero, expected["separately_trained_zero5"], "Zero5", args.tolerance)
    return {
        "status": "PASS",
        "receipt": str(receipt_path),
        "device": str(device),
        "independent_batch_size": args.batch_size,
        "absolute_r2_tolerance": args.tolerance,
        "source_snapshot_receipt_sha256": snapshot.receipt_sha256,
        "source_snapshot_sha256": snapshot.snapshot_sha256,
        "full": full,
        "zero": zero,
        "full_minus_zero": full["pooled_r2"] - zero["pooled_r2"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--data-dir", type=Path, default=SPINT_ROOT / "data/000954")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=29)
    parser.add_argument("--tolerance", type=float, default=2.0e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    need(args.batch_size > 0 and args.batch_size != 32, "independent batch size must be positive and differ from 32")
    need(0 < args.tolerance <= 1.0e-4, "invalid R2 tolerance")
    result = run(args)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        need(not output.exists(), f"refusing to overwrite audit artifact: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
        output.chmod(0o444)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
