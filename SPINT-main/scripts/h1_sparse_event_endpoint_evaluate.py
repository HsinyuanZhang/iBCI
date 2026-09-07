#!/usr/bin/env python3
"""Evaluate the fixed H-SE5/Zero5 fold-0 pair and same-checkpoint controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from src.data.h1_m4_eb_pilot import H1_M4_FOLD0_TARGET
from src.data.h1_sparse_event_endpoint import (
    H1SparseEventDataModule,
    build_sparse_target_dataset,
)
from src.data.h1_sparse_event_source_snapshot import (
    apply_snapshot_to_source_module,
    load_snapshot,
)
from src.h1_m4_eb_normalized_v2_contract import assert_state_immutable, sha256_file, state_hash, write_immutable_json
from src.models.h1_sparse_event_module import CHECKPOINT_SCHEMA


SCHEMA = "h1_sparse_event_endpoint_hse5_fold0_terminal_v1"
SEALED_HS_R2 = 0.4968330503240682
SEALED_HC_R2 = 0.52551078006707
SEALED_HC0_R2 = 0.48661562
SEALED_REFERENCE = PROJECT_ROOT / "pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/H1_CARRIERID_H32_FOLD0_TERMINAL_GATE_FLOAT64_R2.json"
EXPECTED_QUERY_SHA = "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"
DEFAULT_SOURCE_SNAPSHOT_RECEIPT = (
    PROJECT_ROOT / "pilot_artifacts/h1_sparse_event_endpoint/source_snapshot/H1_SE5_FOLD0_SOURCE_v1.json"
)


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path, config_path: Path, arm: str) -> tuple[dict[str, Any], Any, Mapping[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _need(isinstance(checkpoint, dict) and "state_dict" in checkpoint, f"H-SE5 {arm} checkpoint invalid")
    _need(int(checkpoint.get("epoch", -1)) == 49 and int(checkpoint.get("global_step", 0)) > 0,
          f"H-SE5 {arm} is not terminal epoch 49")
    metadata = checkpoint.get("h1_sparse_event_endpoint")
    _need(isinstance(metadata, dict) and metadata.get("schema") == CHECKPOINT_SCHEMA, f"H-SE5 {arm} metadata missing")
    expected = {
        "fold_date": "19250101", "arm": arm, "checkpoint_epoch_zero_based": 49,
        "epochs_completed": 50, "selected_by": "fixed_terminal_epoch_no_validation_selection",
        "carrier_dim": 5, "carrier_hidden_dim": 32, "carrier_trial_length": 1024,
        "target_session_optimizer_steps": 0, "target_session_backward_steps": 0,
    }
    for key, value in expected.items():
        _need(metadata.get(key) == value, f"H-SE5 {arm} metadata drift at {key}")
    _need(sha256_file(config_path) == metadata.get("config_sha256"), f"H-SE5 {arm} config SHA mismatch")
    config = OmegaConf.load(config_path)
    _need(config.pilot.arm == arm and int(config.model.net.carrier_dim) == 5,
          f"H-SE5 {arm} resolved config mismatch")
    return checkpoint, config, metadata


def _pair(full: Mapping[str, Any], zero: Mapping[str, Any]) -> dict[str, str]:
    fields = (
        "fold_date", "source_manifest_sha256", "normalizer_sha256", "source_cache_sha256",
        "normalized_cache_sha256", "basis_sha256", "source_hashes_sha256", "initial_state_sha256",
    )
    for field in fields:
        _need(full.get(field) == zero.get(field), f"H-SE5 pair mismatch at {field}")
    return {field: str(full[field]) for field in fields}


def _instantiate(config: Any, checkpoint: Mapping[str, Any], device: torch.device):
    model = hydra.utils.instantiate(config.model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device); model.eval()
    return model


def _r2(truth: np.ndarray, estimate: np.ndarray) -> float:
    truth64, estimate64 = np.asarray(truth, np.float64), np.asarray(estimate, np.float64)
    sse = float(np.square(truth64 - estimate64).sum())
    tss = float(np.square(truth64 - truth64.mean(axis=0, keepdims=True)).sum())
    _need(np.isfinite(sse) and np.isfinite(tss) and tss > 0, "H-SE5 R2 undefined")
    return 1.0 - sse / tss


def _evaluate(model, dataset, device: torch.device, label: str) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0)
    predictions: list[np.ndarray] = []; targets: list[np.ndarray] = []; sessions: list[str] = []; sizes: list[int] = []
    before = state_hash(model.state_dict())
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
            predictions.append(output[:, -1].cpu().numpy()); targets.append(target[:, -1].cpu().numpy())
            sessions.extend(list(names)); sizes.append(int(output.shape[0]))
    after = state_hash(model.state_dict())
    assert_state_immutable(before, after, label)
    _need(sum(sizes) == len(dataset) and sizes[-1] == (len(dataset) % 32 or 32), f"{label}: sample omission")
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    per_session = {}
    for name in H1_M4_FOLD0_TARGET:
        indices = np.asarray([item == name for item in sessions], dtype=bool)
        per_session[name] = {"samples": int(indices.sum()), "r2": _r2(target[indices], prediction[indices])}
    return {
        "pooled_r2": _r2(target, prediction), "samples": len(dataset), "per_session": per_session,
        "batches": len(sizes), "last_batch_size": sizes[-1], "state_sha256_before": before,
        "state_sha256_after": after, "state_immutable": before == after,
        "query_window_indices_sha256": dataset.window_indices_sha256,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    # A fresh construction is retained as a source-data/code audit, but the
    # target carrier uses the immutable source snapshot whose manifest is
    # exactly bound by both checkpoints.  LAPACK/SVD last-bit differences may
    # change a fresh basis across process state; no alternate basis is allowed
    # to reach target scoring.
    source = H1SparseEventDataModule(
        task="h1", data_dir=str(args.data_dir.resolve()),
        cache_dir=str((PROJECT_ROOT / "pilot_artifacts/h1_sparse_event_endpoint/shared_source_cache").resolve()),
    )
    source.setup("fit")
    fresh_source_manifest_sha256 = source.pilot_manifest_sha256
    fresh_basis_sha256 = source.basis.basis_sha256
    snapshot = load_snapshot(args.source_snapshot_receipt)
    apply_snapshot_to_source_module(source, snapshot)
    full_ckpt, full_config, full_meta = _load(args.full_checkpoint, args.full_config, "full")
    zero_ckpt, zero_config, zero_meta = _load(args.zero_checkpoint, args.zero_config, "zero")
    pair = _pair(full_meta, zero_meta)
    # Validate all source assets before opening target sessions.
    for metadata in (full_meta, zero_meta):
        print(json.dumps({
            "checkpoint_source_manifest_sha256": metadata["source_manifest_sha256"],
            "reconstructed_source_manifest_sha256": source.pilot_manifest_sha256,
            "checkpoint_normalizer_sha256": metadata["normalizer_sha256"],
            "reconstructed_normalizer_sha256": source.normalizer.normalizer_sha256,
            "reconstructed_s_src": source.normalizer.s_src,
            "reconstructed_basis_sha256": source.basis.basis_sha256,
            "reconstructed_basis_array_sha256": source.basis.manifest()["array_sha256"],
            "fresh_reconstruction_carrier_cache_sha256": source.carrier_cache.manifest["cache_sha256"],
            "snapshot_carrier_cache_sha256": source.pilot_manifest()["carrier_cache_sha256"],
        }, sort_keys=True), flush=True)
        _need(metadata["source_manifest_sha256"] == source.pilot_manifest_sha256,
              "H-SE5 checkpoint/source manifest mismatch: "
              f"checkpoint={metadata['source_manifest_sha256']} "
              f"reconstructed={source.pilot_manifest_sha256}")
        _need(metadata["normalizer_sha256"] == source.normalizer.normalizer_sha256,
              "H-SE5 checkpoint/source normalizer mismatch")
    full_model = _instantiate(full_config, full_ckpt, device)
    zero_model = _instantiate(zero_config, zero_ckpt, device)
    del full_ckpt, zero_ckpt
    target = build_sparse_target_dataset(data_dir=args.data_dir, source_module=source)
    _need(target.window_indices_sha256 == EXPECTED_QUERY_SHA, "H-SE5 query pool differs from sealed H-S/H-C pool")
    variants = {name: target.with_intervention(name) for name in target.INTERVENTIONS}
    full_scores = {name: _evaluate(full_model, dataset, device, f"H-SE5/{name}") for name, dataset in variants.items()}
    zero_score = _evaluate(zero_model, variants["full"], device, "H-SE5-Z5")
    candidate = full_scores["full"]["pooled_r2"]
    margins = {
        "hse5_minus_separately_trained_zero5": candidate - zero_score["pooled_r2"],
        "hse5_minus_same_checkpoint_zero5": candidate - full_scores["zero"]["pooled_r2"],
        "hse5_minus_same_checkpoint_row_shuffle": candidate - full_scores["row"]["pooled_r2"],
        "hse5_minus_same_checkpoint_label_shuffle": candidate - full_scores["label"]["pooled_r2"],
        "hse5_minus_sealed_hs": candidate - SEALED_HS_R2,
        "hse5_minus_sealed_dense_hc": candidate - SEALED_HC_R2,
    }
    clauses = {
        "beats_separately_trained_zero5": margins["hse5_minus_separately_trained_zero5"] > 0,
        "beats_same_checkpoint_zero5": margins["hse5_minus_same_checkpoint_zero5"] > 0,
        "beats_same_checkpoint_row_shuffle": margins["hse5_minus_same_checkpoint_row_shuffle"] > 0,
        "beats_same_checkpoint_label_shuffle": margins["hse5_minus_same_checkpoint_label_shuffle"] > 0,
    }
    reference_sha = sha256_file(SEALED_REFERENCE)
    receipt = {
        "schema": SCHEMA,
        "status": "PASS_HSE5_FIRST_CELL" if all(clauses.values()) else "STOP_HSE5_FIRST_CELL",
        "fold_date": "19250101", "seed": 42, "support_trials": 4,
        "evaluation_device": str(device), "checkpoint_binding_completed_before_target_open": True,
        "checkpoints": {
            "full": {"path": str(args.full_checkpoint), "sha256": sha256_file(args.full_checkpoint), "metadata": full_meta},
            "zero": {"path": str(args.zero_checkpoint), "sha256": sha256_file(args.zero_checkpoint), "metadata": zero_meta},
            "pair_binding": pair,
        },
        "source_manifest": source.pilot_manifest(), "source_manifest_sha256": source.pilot_manifest_sha256,
        "source_snapshot": {
            "receipt": str(snapshot.receipt_path), "receipt_sha256": snapshot.receipt_sha256,
            "snapshot": str(snapshot.snapshot_path), "snapshot_sha256": snapshot.snapshot_sha256,
            "manifest_sha256": snapshot.manifest_sha256,
            "basis_sha256": snapshot.basis.basis_sha256,
            "normalizer_sha256": snapshot.normalizer.normalizer_sha256,
            "fresh_reconstruction_manifest_sha256": fresh_source_manifest_sha256,
            "fresh_reconstruction_basis_sha256": fresh_basis_sha256,
            "fresh_reconstruction_carrier_cache_sha256": source.carrier_cache.manifest["cache_sha256"],
            "reason": "hash-bound source snapshot avoids process-state-dependent SVD last-bit drift",
        },
        "evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "target": {
            "sessions": list(H1_M4_FOLD0_TARGET), "query_window_indices_sha256": target.window_indices_sha256,
            "support_and_carrier_hashes": target.support_and_carrier_hashes(), "post_four_trial_query": True,
        },
        "metrics": {"hse5_same_checkpoint_interventions": full_scores, "separately_trained_zero5": zero_score},
        "sealed_matched_references": {
            "receipt": str(SEALED_REFERENCE), "receipt_sha256": reference_sha,
            "query_window_indices_sha256": EXPECTED_QUERY_SHA,
            "h_s_r2": SEALED_HS_R2, "dense_h_c_r2": SEALED_HC_R2, "dense_h_c0_r2_rounded": SEALED_HC0_R2,
        },
        "gate": {"clauses": clauses, "pass": all(clauses.values()), "margins": margins},
        "scope": {
            "public_held_in_calibration_nwbs_opened": 13, "minival_opened": False,
            "heldout_opened": False, "formal_opened": False, "evalai_opened": False,
            "target_optimizer_steps": 0, "target_backward_steps": 0,
        },
        "interpretation": {
            "development_only": True,
            "m3_not_implied_by_m4": True,
            "same_checkpoint_controls_are_mechanism_diagnostics_not_separately_trained_nulls": True,
        },
    }
    output, digest = write_immutable_json(args.output, receipt)
    return {**receipt, "receipt_path": str(output), "receipt_sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/000954")
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--zero-checkpoint", type=Path, required=True)
    parser.add_argument("--full-config", type=Path, required=True)
    parser.add_argument("--zero-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-snapshot-receipt", type=Path, default=DEFAULT_SOURCE_SNAPSHOT_RECEIPT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    result = evaluate(args)
    print(json.dumps({"status": result["status"], "receipt": result["receipt_path"], "sha256": result["receipt_sha256"]}, indent=2))
    return 0 if result["gate"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
