#!/usr/bin/env python3
"""Independently recompute Context Full and common H-SE5 Zero prediction R2.

No terminal evaluator code is imported.  This audit reopens the immutable v3
Context source snapshot, reconstructs the strict target query, and uses a
batch size other than the evaluator's 32 (29 by default).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SPINT = ROOT / "SPINT-main"
if str(SPINT) not in sys.path:
    sys.path.insert(0, str(SPINT))

import numpy as np
import torch


SCHEMA = "h1_context_event_carrier_m4_fold0_terminal_v2"
QUERY_SHA = "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"
TARGETS = ("ses-19250101T111740", "ses-19250101T112404")
TARGET_SAMPLES = {"ses-19250101T111740": 6735, "ses-19250101T112404": 2230}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_digest(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8")); digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth64, prediction64 = np.asarray(truth, np.float64), np.asarray(prediction, np.float64)
    need(truth64.shape == prediction64.shape and truth64.ndim == 2, "R2 shape mismatch")
    residual = float(np.square(truth64 - prediction64).sum(dtype=np.float64))
    centered = truth64 - truth64.mean(axis=0, keepdims=True, dtype=np.float64)
    total = float(np.square(centered).sum(dtype=np.float64))
    need(np.isfinite(residual) and np.isfinite(total) and total > 0, "R2 undefined")
    return 1.0 - residual / total


def checkpoint_config(entry: Mapping[str, Any]) -> Path:
    if "resolved_hydra_config" in entry:
        config = Path(entry["resolved_hydra_config"]).resolve()
        need(config.is_file() and sha256(config) == entry.get("resolved_hydra_config_sha256"), "receipt config SHA mismatch")
        return config
    checkpoint = Path(entry["path"]).resolve()
    config = checkpoint.parents[2] / ".hydra/config.yaml"
    need(config.is_file(), f"missing resolved config {config}")
    return config


def instantiate(entry: Mapping[str, Any], device: torch.device):
    import hydra
    from omegaconf import OmegaConf
    checkpoint_path = Path(entry["path"]).resolve()
    need(checkpoint_path.is_file() and sha256(checkpoint_path) == entry.get("sha256"), "checkpoint SHA mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("h1_sparse_event_endpoint")
    need(checkpoint.get("epoch") == 49 and metadata == entry.get("metadata"), "terminal checkpoint metadata mismatch")
    config = checkpoint_config(entry)
    need(sha256(config) == metadata.get("config_sha256"), "checkpoint/config SHA mismatch")
    model = hydra.utils.instantiate(OmegaConf.load(config).model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def score(entry: Mapping[str, Any], dataset: Any, device: torch.device, batch_size: int) -> dict[str, Any]:
    from torch.utils.data import DataLoader
    model = instantiate(entry, device)
    before = state_digest(model.state_dict())
    predictions: list[np.ndarray] = []; targets: list[np.ndarray] = []; sessions: list[str] = []; sizes: list[int] = []
    with torch.inference_mode():
        for neural, target, identity, names, carrier in DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0):
            output = model(neural.to(device=device, dtype=torch.float32),
                           calib_trialized_neural_features=identity.to(device=device, dtype=torch.float32),
                           carrier=carrier.to(device=device, dtype=torch.float32))
            if model.hparams.decode_last_timestep_only:
                output, target = output[:, -1:, :], target[:, -1:, :]
            if model.hparams.predict_scaled_behavior:
                output = output / model.hparams.behavior_scaling_factor
            predictions.append(output[:, -1].cpu().numpy()); targets.append(target[:, -1].cpu().numpy())
            sessions.extend(str(name) for name in names); sizes.append(int(output.shape[0]))
    after = state_digest(model.state_dict())
    need(before == after, "model state mutated during independent scoring")
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    need(prediction.shape == target.shape == (8965, 7) and len(sessions) == 8965,
         "independent prediction/query sample shape mismatch")
    session_array = np.asarray(sessions)
    per_session = {name: {"samples": int((session_array == name).sum()),
                          "r2": r2(target[session_array == name], prediction[session_array == name])}
                   for name in TARGETS}
    need({name: row["samples"] for name, row in per_session.items()} == TARGET_SAMPLES,
         "independent per-session query counts mismatch")
    return {"pooled_r2": r2(target, prediction), "samples": 8965, "batches": len(sizes),
            "last_batch_size": sizes[-1], "batch_size": batch_size, "per_session": per_session,
            "state_sha256_before": before, "state_sha256_after": after, "state_immutable": True,
            "query_window_indices_sha256": dataset.window_indices_sha256}


def compare(observed: Mapping[str, Any], expected: Mapping[str, Any], label: str, tolerance: float) -> None:
    need(observed["query_window_indices_sha256"] == expected["query_window_indices_sha256"] == QUERY_SHA,
         f"{label}: query SHA mismatch")
    need(observed["samples"] == expected["samples"] == 8965 and observed["state_immutable"] is True,
         f"{label}: state/sample mismatch")
    need(abs(float(observed["pooled_r2"]) - float(expected["pooled_r2"])) <= tolerance,
         f"{label}: pooled R2 mismatch")
    for name in TARGETS:
        need(observed["per_session"][name]["samples"] == expected["per_session"][name]["samples"] == TARGET_SAMPLES[name],
             f"{label}/{name}: sample mismatch")
        need(abs(float(observed["per_session"][name]["r2"]) - float(expected["per_session"][name]["r2"])) <= tolerance,
             f"{label}/{name}: R2 mismatch")


def run(args: argparse.Namespace) -> dict[str, Any]:
    receipt_path = args.receipt.resolve()
    need(receipt_path.is_file() and not receipt_path.is_symlink(), f"missing terminal receipt {receipt_path}")
    need(stat.S_IMODE(receipt_path.stat().st_mode) == 0o444, "terminal receipt is not immutable mode 0444")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    need(receipt.get("schema") == SCHEMA, "terminal receipt schema mismatch")
    from src.data.h1_context_event_carrier import H1ContextEventDataModule, build_context_target_dataset
    from src.data.h1_context_event_source_snapshot import load_snapshot
    snapshot_entry = receipt.get("source_snapshot", {})
    snapshot_receipt = Path(snapshot_entry.get("receipt", "")).resolve()
    need(snapshot_receipt.name == "H1_CONTEXT_SER_Q4_FOLD0_SOURCE_v3.json" and sha256(snapshot_receipt) == snapshot_entry.get("sha256"),
         "immutable v3 source snapshot receipt mismatch")
    snapshot = load_snapshot(snapshot_receipt)
    preflight_entry = receipt.get("common_zero_parity_preflight", {})
    preflight = Path(preflight_entry.get("path", "")).resolve()
    need(preflight.is_file() and not preflight.is_symlink() and stat.S_IMODE(preflight.stat().st_mode) == 0o444,
         "v3 preflight is not immutable mode 0444")
    need(preflight.name == "H1_CONTEXT_SER_Q4_M4_FOLD0_PREFLIGHT_v3.json" and sha256(preflight) == preflight_entry.get("sha256"),
         "v3 preflight mismatch")
    preflight_body = json.loads(preflight.read_text(encoding="utf-8"))
    need(preflight_body["source_snapshot"]["receipt_sha256"] == snapshot["receipt_sha256"] and
         preflight_body["source_manifest_sha256"] == snapshot["metadata"]["manifest_sha256"],
         "preflight/v3 snapshot manifest mismatch")
    source = H1ContextEventDataModule(task="h1", data_dir=str(args.data_dir.resolve()),
        cache_dir=str(SPINT / "pilot_artifacts/h1_context_event_carrier/shared_source_cache"),
        source_snapshot_receipt=str(snapshot_receipt))
    source.setup("fit")
    need(source.pilot_manifest_sha256 == snapshot["metadata"]["manifest_sha256"] == receipt["source_manifest_sha256"],
         "immutable v3 context source reconstruction mismatch")
    need(receipt["checkpoints"]["full"]["metadata"]["source_manifest_sha256"] == source.pilot_manifest_sha256,
         "Context Full checkpoint/source manifest mismatch")
    need(receipt["checkpoints"]["common_hse5_zero5"]["metadata"] == preflight_body["checkpoints"]["sealed_hse5_zero"]["metadata"],
         "common H-SE5 Zero checkpoint/preflight metadata mismatch")
    dataset = build_context_target_dataset(data_dir=args.data_dir, source_module=source)
    need(len(dataset) == 8965 and dataset.window_indices_sha256 == QUERY_SHA,
         "exact 8965-query Context target contract mismatch")
    device = torch.device(args.device)
    full = score(receipt["checkpoints"]["full"], dataset.with_intervention("full"), device, args.batch_size)
    zero = score(receipt["checkpoints"]["common_hse5_zero5"], dataset.with_intervention("full"), device, args.batch_size)
    compare(full, receipt["metrics"]["context_same_checkpoint_interventions"]["full"], "Context Full", args.tolerance)
    compare(zero, receipt["metrics"]["common_independent_hse5_zero5"], "common H-SE5 Zero", args.tolerance)
    return {"status": "PASS", "receipt": str(receipt_path), "device": str(device),
            "independent_batch_size": args.batch_size, "absolute_r2_tolerance": args.tolerance,
            "v3_snapshot_receipt_sha256": snapshot["receipt_sha256"], "v3_snapshot_sha256": snapshot["snapshot_sha256"],
            "v3_preflight_sha256": sha256(preflight), "full": full, "common_hse5_zero5": zero,
            "full_minus_common_hse5_zero5": full["pooled_r2"] - zero["pooled_r2"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, help="immutable Context Full terminal receipt")
    parser.add_argument("--data-dir", type=Path, default=SPINT / "data/000954")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=29)
    parser.add_argument("--tolerance", type=float, default=2.0e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    need(args.batch_size > 0 and args.batch_size != 32, "independent batch size must differ from evaluator batch 32")
    need(0 < args.tolerance <= 1.0e-4, "invalid R2 tolerance")
    result = run(args)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve(); need(not output.exists(), f"refusing to overwrite audit artifact: {output}")
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(encoded, encoding="utf-8"); output.chmod(0o444)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
