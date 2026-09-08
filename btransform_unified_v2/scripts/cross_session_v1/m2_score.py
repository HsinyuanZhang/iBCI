#!/usr/bin/env python3
"""Auditable post-training score wrapper for the sealed cross-session M2 run.

This deliberately leaves the active trainer untouched.  It validates a completed
source-selected run, verifies that the selected package is byte-for-tensor equal
to its declared epoch package, then scores that package and the mandatory e24
package while retaining each target-session prediction row and its cache row ID.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT.parent / "btransform_unified_v1" / "src", ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from btransform_unified_v1.r2 import variance_weighted_r2
from scripts.cross_session_v1 import m2_data, m2_train
from scripts.rift_v1 import m2_concat_train as base
from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan
from tfpd_exploration.src.m2_dual_track_v1 import sampler


SCHEMA = "cross_session_m2_ext4_score_audit_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def package(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise RuntimeError(f"missing EMA package: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not all(isinstance(key, str) and isinstance(tensor, torch.Tensor)
                                               for key, tensor in value.items()):
        raise RuntimeError(f"{path}: EMA package is not a string-to-tensor mapping")
    return value


def exact_package_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], *, label: str) -> None:
    if set(left) != set(right):
        raise RuntimeError(f"{label}: EMA package keys differ")
    unequal = [name for name in sorted(left) if left[name].dtype != right[name].dtype
               or tuple(left[name].shape) != tuple(right[name].shape) or not torch.equal(left[name], right[name])]
    if unequal:
        raise RuntimeError(f"{label}: EMA package tensor content differs at {unequal[0]}")


def read_and_validate_run(run: Path, arm: str, seed: int) -> tuple[dict[str, Any], dict[str, Any], int, dict[str, Any]]:
    meta_path, receipt_path = run / "run_meta.json", run / "train_receipt.json"
    if not meta_path.is_file() or not receipt_path.is_file():
        raise RuntimeError("score requires run_meta.json and completed train_receipt.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for label, value in (("run metadata", meta), ("train receipt", receipt)):
        if value.get("cell") != m2_train.CELL or value.get("arm") != arm or int(value.get("seed", -1)) != seed:
            raise RuntimeError(f"{label} cell/arm/seed mismatch")
    if meta.get("schema") != "cross_session_m2_train_v1" or meta.get("status") != "FORMAL":
        raise RuntimeError("run metadata schema/status mismatch")
    if receipt.get("schema") != "cross_session_m2_train_receipt_v1" or receipt.get("status") != "COMPLETED":
        raise RuntimeError("train receipt schema/status mismatch")
    if int(meta.get("epochs", -1)) != m2_train.EPOCHS or int(receipt.get("epochs", -1)) != m2_train.EPOCHS:
        raise RuntimeError("run metadata/receipt epoch contract mismatch")
    if int(meta.get("updates_per_epoch", -1)) < 1 or receipt.get("updates_per_epoch") != meta.get("updates_per_epoch"):
        raise RuntimeError("run metadata/receipt update-count disagreement")
    if meta.get("encoder_init") != "random":
        raise RuntimeError("run metadata encoder initialization mismatch")
    current_sources = m2_train.source_hashes()
    if meta.get("source_hashes") != current_sources or receipt.get("source_hashes") != meta.get("source_hashes"):
        raise RuntimeError("run/receipt/current source hash disagreement")
    if receipt.get("cache_hashes") != meta.get("cache_hashes") or set(meta.get("cache_hashes", {})) != {"source_train"}:
        raise RuntimeError("run metadata and receipt source-cache disagreement")
    if receipt.get("target_query_labels_used_for_selection") is not False or receipt.get("target_query_labels_used_for_gradients") is not False:
        raise RuntimeError("receipt does not establish source-only training and selection")
    curve = receipt.get("source_trial_val_ema_by_epoch")
    if not isinstance(curve, dict) or set(curve) != {str(epoch) for epoch in range(1, m2_train.EPOCHS + 1)}:
        raise RuntimeError("source validation EMA curve is incomplete")
    selection = receipt.get("selection")
    if not isinstance(selection, dict):
        raise RuntimeError("missing source validation selection")
    epoch = int(selection.get("epoch", -1))
    if not 1 <= epoch <= m2_train.EPOCHS or str(epoch) not in curve:
        raise RuntimeError("selected source validation epoch is invalid")
    chosen_metric = float(curve[str(epoch)].get("equal_session_mean", float("nan")))
    if not math.isfinite(chosen_metric) or float(selection.get("equal_session_mean", float("nan"))) != chosen_metric:
        raise RuntimeError("selection metric is absent, nonfinite, or inconsistent with its curve row")
    return meta, receipt, epoch, curve


def validate_query_banks(dual: Mapping[str, Any]) -> None:
    expected = dict(old_plan.EXT4_EXPECTED_WINDOWS)
    if set(dual) != set(expected):
        raise RuntimeError("EXT4 session roster drift")
    for session, count in expected.items():
        bank = dual[session]
        starts = np.asarray(bank.eligible_starts, dtype=np.int64)
        if starts.ndim != 1 or len(starts) != int(count) or not len(starts) or np.any(np.diff(starts) <= 0):
            raise RuntimeError(f"{session}: EXT4 query row mapping drift")
        if np.any(starts < 0) or np.any(starts + 50 > len(bank.X_store)):
            raise RuntimeError(f"{session}: EXT4 query window bounds drift")
        if np.asarray(bank.target_store).shape != (len(starts), 2):
            raise RuntimeError(f"{session}: EXT4 target/query row geometry drift")


def score_with_artifacts(model: torch.nn.Module, dual: Mapping[str, Any], banks: Mapping[str, Any], pads: Mapping[str, int],
                         device: torch.device, artifact_root: Path) -> dict[str, Any]:
    model.eval()
    rows: dict[str, Any] = {}
    pooled_prediction: list[np.ndarray] = []
    pooled_target: list[np.ndarray] = []
    for session in sorted(dual):
        dual_bank = dual[session]
        prediction_rows: list[np.ndarray] = []
        target_rows: list[np.ndarray] = []
        start_rows: list[np.ndarray] = []
        for offset in range(0, len(dual_bank.eligible_starts), m2_train.BATCH):
            indices = np.arange(offset, min(offset + m2_train.BATCH, len(dual_bank.eligible_starts)), dtype=np.int64)
            batch = sampler.batch_from_indices(dual_bank, indices, device=device, target_space=old_plan.SCORING_TARGET_SPACE)
            with torch.inference_mode():
                raw = model(batch.X, banks[session], input_valid_mask=m2_train.valid(batch, "ext4", pads, device))
            prediction_rows.append(np.ascontiguousarray(raw.float().cpu().numpy() / old_plan.BEHAVIOR_SCALE, dtype=np.float32))
            target_rows.append(np.ascontiguousarray(batch.last_target.float().cpu().numpy(), dtype=np.float32))
            start_rows.append(np.asarray(dual_bank.eligible_starts[indices], dtype=np.int64))
        prediction = np.ascontiguousarray(np.concatenate(prediction_rows), dtype=np.float32)
        target = np.ascontiguousarray(np.concatenate(target_rows), dtype=np.float32)
        starts = np.ascontiguousarray(np.concatenate(start_rows), dtype=np.int64)
        if prediction.shape != target.shape or prediction.shape != (len(dual_bank.eligible_starts), 2) or not np.array_equal(starts, np.asarray(dual_bank.eligible_starts, dtype=np.int64)):
            raise RuntimeError(f"{session}: scored prediction rows are not exactly query-row aligned")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise RuntimeError(f"{session}: nonfinite EXT4 prediction or target")
        artifact = artifact_root / f"{session}.npz"
        atomic_npz(artifact, prediction=prediction, target=target, eligible_starts=starts)
        rows[session] = {"r2": float(variance_weighted_r2(target, prediction)), "window_count": int(len(target)),
                         "prediction_sha256": array_sha(prediction), "target_sha256": array_sha(target),
                         "eligible_starts_sha256": array_sha(starts), "artifact": str(artifact), "artifact_sha256": sha(artifact)}
        pooled_prediction.append(prediction)
        pooled_target.append(target)
    return {"per_session": rows, "equal_session_mean": float(np.mean([rows[session]["r2"] for session in sorted(rows)])),
            "pooled_r2": float(variance_weighted_r2(np.concatenate(pooled_target), np.concatenate(pooled_prediction))),
            "n_windows": int(sum(row["window_count"] for row in rows.values())), "partial": False}


def score(args: argparse.Namespace) -> dict[str, Any]:
    run = args.dest.resolve()
    meta, receipt, selected_epoch, _curve = read_and_validate_run(run, args.arm, args.seed)
    selected_path = run / "selected_source_trial_val_ema.pt"
    epoch_path = run / f"ema_epoch_{selected_epoch:03d}.pt"
    e24_path = run / "ema_epoch_024.pt"
    selected_before, e24_before = sha(selected_path), sha(e24_path)
    exact_package_equal(package(selected_path), package(epoch_path), label="selected versus declared source-validation epoch")
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    ext_dual, ext_banks = m2_data.ext4_query(device)
    validate_query_banks(ext_dual)
    pads = base._surface_padding("ext4", ext_banks)
    model = m2_train.model_for(args.arm, args.seed, device, ext_banks, "ext4", str(meta["encoder_init"]))
    selected_load = m2_train.load_ema_package(model, selected_path, device)
    selected_report = score_with_artifacts(model, ext_dual, ext_banks, pads, device, run / "score_arrays" / "selected_source_trial_val")
    e24_load = m2_train.load_ema_package(model, e24_path, device)
    e24_report = score_with_artifacts(model, ext_dual, ext_banks, pads, device, run / "score_arrays" / "e24")
    packages = {"selected_source_trial_val": {"path": str(selected_path), "sha256_before_score": selected_before,
                "sha256_after_score": sha(selected_path), "declared_source_validation_epoch": selected_epoch,
                "exact_tensor_equal_to_declared_epoch_package": True, "declared_epoch_path": str(epoch_path),
                "declared_epoch_sha256": sha(epoch_path)},
                "e24": {"path": str(e24_path), "sha256_before_score": e24_before, "sha256_after_score": sha(e24_path),
                        "declared_epoch": 24}}
    if packages["selected_source_trial_val"]["sha256_before_score"] != packages["selected_source_trial_val"]["sha256_after_score"] or packages["e24"]["sha256_before_score"] != packages["e24"]["sha256_after_score"]:
        raise RuntimeError("EMA package bytes changed during score")
    out = {"schema": SCHEMA, "status": "COMPLETED", "cell": m2_train.CELL, "arm": args.arm, "seed": args.seed,
           "source_trial_validation_selection": receipt["selection"], "ema_packages": packages,
           "selected_ema_load": selected_load, "selected_ema_ext4": selected_report,
           "e24_ema_load": e24_load, "predeclared_e24_ext4_sensitivity": e24_report,
           "target_query_labels_used_for_selection": False, "target_query_labels_used_for_gradients": False,
           "official_test_used": False, "evalai_opened": False,
           "ext4_cache_hashes": m2_train.cache_hashes("ext4", ext_banks, args.arm), "source_hashes": meta["source_hashes"],
           "score_wrapper_sha256": sha(Path(__file__).resolve())}
    atomic_json(run / "score_receipt.json", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="auditable sealed cross-session M2 EXT4 scorer")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--arm", choices=m2_train.ARMS, required=True)
    parser.add_argument("--seed", choices=(42, 43, 44), type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu threads must be positive")
    print(json.dumps(score(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
