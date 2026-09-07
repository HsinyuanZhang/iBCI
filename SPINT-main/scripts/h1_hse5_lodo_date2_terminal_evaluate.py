#!/usr/bin/env python3
"""Terminal-only H-SE5 date-2 scorer with matched target-side controls.

This script is deliberately separate from the sealed fold-0 evaluator.  It
opens the three ``19250108`` public held-in recordings only after binding both
terminal checkpoints to one immutable source snapshot.  No optimizer,
backward pass, or Lightning Trainer is constructed for target scoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for directory in (WORKSPACE_ROOT, PROJECT_ROOT):
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
from src.h1_m4_eb_normalized_v2_contract import (
    assert_state_immutable,
    sha256_file,
    state_hash,
    write_immutable_json,
)
from src.models.h1_sparse_event_dated_module import CHECKPOINT_SCHEMA, PROTOCOL


SCHEMA = "h1_hse5_lodo_date2_terminal_evaluation_v1"
OUTER_DATE = "19250108"
TARGETS = (
    "ses-19250108T110520",
    "ses-19250108T111022",
    "ses-19250108T111455",
)
TARGET_SAMPLES = {
    "ses-19250108T110520": 8330,
    "ses-19250108T111022": 2508,
    "ses-19250108T111455": 2269,
}
QUERY_SHA = "b0cd153750cb484af1237b7af1861600aca69a9b201244c22d140b42b1da7f6e"
DEFAULT_SNAPSHOT = (
    PROJECT_ROOT
    / "pilot_artifacts/h1_hse5_lodo_19250108/H1_HSE5_LODO_19250108_SOURCE_v1.json"
)
PAIR_FIELDS = (
    "fold_date",
    "protocol",
    "source_manifest_sha256",
    "normalizer_sha256",
    "source_cache_sha256",
    "normalized_cache_sha256",
    "basis_sha256",
    "source_hashes_sha256",
    "initial_state_sha256",
    "source_snapshot_receipt_sha256",
    "source_snapshot_sha256",
)


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Float64 multi-output pooled R²; intentionally simple and auditable."""
    truth64 = np.asarray(truth, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    _need(truth64.shape == prediction64.shape and truth64.ndim == 2, "invalid R2 arrays")
    residual = float(np.sum((truth64 - prediction64) ** 2, dtype=np.float64))
    centered = truth64 - np.mean(truth64, axis=0, keepdims=True, dtype=np.float64)
    total = float(np.sum(centered**2, dtype=np.float64))
    _need(np.isfinite(residual) and np.isfinite(total) and total > 0.0, "undefined R2")
    return 1.0 - residual / total


def _checkpoint(
    checkpoint_path: Path,
    config_path: Path,
    arm: str,
    snapshot: Any,
    training_source_manifest_sha256: str,
) -> tuple[dict[str, Any], Any, Mapping[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _need(isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), Mapping),
          f"{arm}: invalid checkpoint")
    _need(int(checkpoint.get("epoch", -1)) == 49 and int(checkpoint.get("global_step", 0)) > 0,
          f"{arm}: requires terminal epoch-49 checkpoint")
    metadata = checkpoint.get("h1_sparse_event_endpoint_dated")
    _need(isinstance(metadata, Mapping), f"{arm}: dated H-SE5 metadata missing")
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "protocol": PROTOCOL,
        "fold_date": OUTER_DATE,
        "arm": arm,
        "checkpoint_epoch_zero_based": 49,
        "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_validation_selection",
        "carrier_mode": "correct_sparse_endpoint" if arm == "full" else "literal_zero5_at_model_boundary",
        "carrier_dim": 5,
        "carrier_hidden_dim": 32,
        "carrier_trial_length": 1024,
        "carrier_identity_parameters": 58172,
        "target_session_optimizer_steps": 0,
        "target_session_backward_steps": 0,
        "source_snapshot_receipt_sha256": snapshot.receipt_sha256,
        "source_snapshot_sha256": snapshot.snapshot_sha256,
        # The snapshot's manifest is source-only.  The training DataModule
        # adds a self-describing immutable-snapshot binding before hashing its
        # runtime source manifest; that latter digest is what the checkpoint
        # correctly records.
        "source_manifest_sha256": training_source_manifest_sha256,
        "normalizer_sha256": snapshot.normalizer.normalizer_sha256,
        "basis_sha256": snapshot.basis.basis_sha256,
    }
    for key, value in expected.items():
        _need(metadata.get(key) == value, f"{arm}: checkpoint metadata drift at {key}")
    _need(sha256_file(config_path) == metadata.get("config_sha256"), f"{arm}: config SHA mismatch")
    config = OmegaConf.load(config_path)
    _need(
        str(config.pilot.fold_date) == OUTER_DATE
        and str(config.pilot.arm) == arm
        and bool(config.pilot.zero_carrier) == (arm == "zero"),
        f"{arm}: resolved config pilot mismatch",
    )
    return checkpoint, config, metadata


def _pair_binding(full: Mapping[str, Any], zero: Mapping[str, Any]) -> dict[str, str]:
    for field in PAIR_FIELDS:
        _need(full.get(field) == zero.get(field), f"Full/Zero5 checkpoint mismatch at {field}")
    return {field: str(full[field]) for field in PAIR_FIELDS}


def _instantiate(config: Any, checkpoint: Mapping[str, Any], device: torch.device):
    model = hydra.utils.instantiate(config.model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _score(model: Any, dataset: Any, device: torch.device, label: str) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0)
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    names_seen: list[str] = []
    sizes: list[int] = []
    before = state_hash(model.state_dict())
    with torch.inference_mode():
        for neural, target, identity, names, carrier in loader:
            prediction = model(
                neural.to(device=device, dtype=torch.float32),
                calib_trialized_neural_features=identity.to(device=device, dtype=torch.float32),
                carrier=carrier.to(device=device, dtype=torch.float32),
            )
            if model.hparams.decode_last_timestep_only:
                prediction, target = prediction[:, -1:, :], target[:, -1:, :]
            if model.hparams.predict_scaled_behavior:
                prediction = prediction / model.hparams.behavior_scaling_factor
            predictions.append(prediction[:, -1].cpu().numpy())
            truths.append(target[:, -1].cpu().numpy())
            names_seen.extend(str(name) for name in names)
            sizes.append(int(prediction.shape[0]))
    after = state_hash(model.state_dict())
    assert_state_immutable(before, after, label)
    _need(sum(sizes) == len(dataset) == 13107 and len(sizes) == 410 and sizes[-1] == 19,
          f"{label}: target batch accounting mismatch")
    prediction_array = np.concatenate(predictions, axis=0)
    truth_array = np.concatenate(truths, axis=0)
    session_array = np.asarray(names_seen)
    per_session: dict[str, Any] = {}
    for session in TARGETS:
        select = session_array == session
        _need(int(select.sum()) == TARGET_SAMPLES[session], f"{label}/{session}: query count mismatch")
        per_session[session] = {
            "samples": int(select.sum()),
            "r2": _r2(truth_array[select], prediction_array[select]),
        }
    return {
        "pooled_r2": _r2(truth_array, prediction_array),
        "samples": int(len(dataset)),
        "batches": len(sizes),
        "last_batch_size": sizes[-1],
        "batch_size": 32,
        "per_session": per_session,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "state_immutable": before == after,
        "query_window_indices_sha256": dataset.window_indices_sha256,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = load_snapshot(args.source_snapshot_receipt)
    _need(snapshot.basis.outer_date == OUTER_DATE and tuple(snapshot.cache.source_sessions) == tuple(snapshot.manifest["source_sessions"]),
          "source snapshot is not the date-2 authority")
    # This source-only construction verifies the snapshot against source NWBs.
    # Target records become reachable only in the following isolated builder.
    source = H1SparseEventDatedDataModule(
        task="h1", data_dir=str(args.data_dir.resolve()), cache_dir=str(args.source_cache_dir.resolve()),
        fold_date=OUTER_DATE, source_snapshot_receipt=str(snapshot.receipt_path),
    )
    source.setup("fit")
    source_manifest = source.pilot_manifest()
    source_snapshot_binding = source_manifest.get("source_snapshot")
    _need(
        isinstance(source_snapshot_binding, Mapping)
        and source_snapshot_binding.get("receipt_sha256") == snapshot.receipt_sha256
        and source_snapshot_binding.get("snapshot_sha256") == snapshot.snapshot_sha256,
        "source module did not consume the immutable date-2 snapshot authority",
    )
    _need(source.normalizer.normalizer_sha256 == snapshot.normalizer.normalizer_sha256, "source normalizer drift")
    full_checkpoint, full_config, full_meta = _checkpoint(
        args.full_checkpoint, args.full_config, "full", snapshot, source.pilot_manifest_sha256,
    )
    zero_checkpoint, zero_config, zero_meta = _checkpoint(
        args.zero_checkpoint, args.zero_config, "zero", snapshot, source.pilot_manifest_sha256,
    )
    pair = _pair_binding(full_meta, zero_meta)
    target = build_dated_sparse_target_dataset(data_dir=args.data_dir, source_module=source)
    _need(tuple(target.target_sessions) == TARGETS and target.window_indices_sha256 == QUERY_SHA and len(target) == 13107,
          "date-2 target roster/query window contract drift")
    variants = {name: target.with_intervention(name) for name in target.INTERVENTIONS}
    device = torch.device(args.device)
    full_model = _instantiate(full_config, full_checkpoint, device)
    zero_model = _instantiate(zero_config, zero_checkpoint, device)
    del full_checkpoint, zero_checkpoint
    full_scores = {name: _score(full_model, dataset, device, f"H-SE5-date2/{name}") for name, dataset in variants.items()}
    independent_zero = _score(zero_model, variants["full"], device, "H-SE5-date2/independent-Zero5")
    full_score = full_scores["full"]
    margins = {
        "full_minus_independently_trained_zero5_pooled": full_score["pooled_r2"] - independent_zero["pooled_r2"],
        "full_minus_same_checkpoint_zero5_pooled": full_score["pooled_r2"] - full_scores["zero"]["pooled_r2"],
        "full_minus_same_checkpoint_row_shuffle_pooled": full_score["pooled_r2"] - full_scores["row"]["pooled_r2"],
        "full_minus_same_checkpoint_endpoint_label_shuffle_pooled": full_score["pooled_r2"] - full_scores["label"]["pooled_r2"],
    }
    per_recording_delta = {
        name: full_score["per_session"][name]["r2"] - independent_zero["per_session"][name]["r2"]
        for name in TARGETS
    }
    core = {
        "full_minus_independently_trained_zero5_pooled_positive": margins["full_minus_independently_trained_zero5_pooled"] > 0.0,
        "all_three_recording_full_minus_independently_trained_zero5_positive": all(value > 0.0 for value in per_recording_delta.values()),
    }
    diagnostics = {
        "full_minus_same_checkpoint_zero5_pooled_positive": margins["full_minus_same_checkpoint_zero5_pooled"] > 0.0,
        "full_minus_same_checkpoint_row_shuffle_pooled_positive": margins["full_minus_same_checkpoint_row_shuffle_pooled"] > 0.0,
        "full_minus_same_checkpoint_endpoint_label_shuffle_pooled_positive": margins["full_minus_same_checkpoint_endpoint_label_shuffle_pooled"] > 0.0,
    }
    core_pass, diagnostic_pass = all(core.values()), all(diagnostics.values())
    receipt = {
        "schema": SCHEMA,
        "status": "PASS_HSE5_LODO_DATE2_CORE_GATE" if core_pass else "STOP_HSE5_LODO_DATE2_CORE_GATE",
        "outer_date": OUTER_DATE,
        "seed": 42,
        "support_trials": 4,
        "evaluation_device": str(device),
        "checkpoint_binding_completed_before_target_open": True,
        "checkpoints": {
            "full": {"path": str(args.full_checkpoint.resolve()), "sha256": sha256_file(args.full_checkpoint), "metadata": dict(full_meta)},
            "zero5": {"path": str(args.zero_checkpoint.resolve()), "sha256": sha256_file(args.zero_checkpoint), "metadata": dict(zero_meta)},
            "pair_binding": pair,
        },
        "source_snapshot": {
            "receipt": str(snapshot.receipt_path), "receipt_sha256": snapshot.receipt_sha256,
            "snapshot": str(snapshot.snapshot_path), "snapshot_sha256": snapshot.snapshot_sha256,
            "manifest_sha256": snapshot.manifest_sha256, "basis_sha256": snapshot.basis.basis_sha256,
            "normalizer_sha256": snapshot.normalizer.normalizer_sha256,
            "training_source_manifest_sha256": source.pilot_manifest_sha256,
            "training_authority": "single_immutable_date2_source_snapshot",
        },
        "evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "target": {
            "sessions": list(TARGETS), "query_window_indices_sha256": target.window_indices_sha256,
            "support_and_carrier_hashes": target.support_and_carrier_hashes(), "post_four_trial_query": True,
            "target_samples": TARGET_SAMPLES,
        },
        "metrics": {"full_same_checkpoint_interventions": full_scores, "independently_trained_zero5": independent_zero},
        "gate": {
            "predeclared_core": core, "same_checkpoint_diagnostics": diagnostics,
            "per_recording_full_minus_independently_trained_zero5": per_recording_delta,
            "margins": margins, "core_pass": core_pass, "diagnostic_pass": diagnostic_pass,
        },
        "scope": {
            "public_heldin_calibration_nwbs_opened": 13, "target_optimizer_steps": 0,
            "target_backward_steps": 0, "minival_opened": False, "heldout_opened": False,
            "formal_opened": False, "evalai_opened": False,
        },
        "interpretation": {
            "development_only": True,
            "same_checkpoint_controls_are_mechanism_diagnostics_not_independently_trained_nulls": True,
            "date2_core_gate_is_required_before_two_date_sparse_h1_claim": True,
        },
    }
    output, digest = write_immutable_json(args.output, receipt)
    return {**receipt, "receipt_path": str(output), "receipt_sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--zero-checkpoint", type=Path, required=True)
    parser.add_argument("--full-config", type=Path, required=True)
    parser.add_argument("--zero-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/000954")
    parser.add_argument("--source-cache-dir", type=Path, default=PROJECT_ROOT / "pilot_artifacts/h1_hse5_lodo_19250108/shared_source_cache")
    parser.add_argument("--source-snapshot-receipt", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    result = evaluate(args)
    print(json.dumps({"status": result["status"], "receipt": result["receipt_path"], "sha256": result["receipt_sha256"]}, indent=2))
    return 0 if result["gate"]["core_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
