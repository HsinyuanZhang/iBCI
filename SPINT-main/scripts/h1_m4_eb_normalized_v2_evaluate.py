#!/usr/bin/env python3
"""Fail-closed terminal evaluator for normalized V2.

Both real epoch-49 V2 checkpoints and their metadata are bound before the
first target NWB is opened.  The five V1 gate clauses are intentionally kept
unchanged; normalization is a numerical-contract repair, not a new target
selection criterion.
"""
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

from src.data.h1_m4_eb_normalized_v2 import (
    H1M4EBNormalizedV2DataModule,
    H1M4EBNormalizedV2StrictTargetDataset,
)
from src.data.h1_m4_eb_pilot import H1_M4_FOLD0_TARGET, load_target_records, validate_target_receipt_binding
from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    assert_state_immutable,
    load_and_validate_terminal_checkpoint,
    sha256_file,
    state_hash,
    validate_paired_checkpoint_bindings,
    write_immutable_json,
)


def _validate_resolved_config(config_path: Path, arm: str) -> Any:
    config = OmegaConf.load(config_path)
    required = {"seed": 42, "train": True, "test": False, "ckpt_path": None}
    for key, expected in required.items():
        if config.get(key) != expected:
            raise NormalizedV2ContractError(f"{arm} V2 resolved config drift at {key}")
    if (
        config.pilot.arm != arm
        or bool(config.pilot.train_residual) != (arm == "joint")
        or str(config.pilot.fold_date) != "19250101"
        or int(config.data.calibration_n_trials) != 4
        or int(config.data.window_size) != 700
        or int(config.data.batch_size) != 32
        or int(config.trainer.max_epochs) != 50
        or int(config.trainer.min_epochs) != 50
        or int(config.trainer.limit_val_batches) != 0
        or float(config.model.optimizer.lr) != 5.0e-5
        or float(config.data.normalizer_floor) != 1.0e-12
    ):
        raise NormalizedV2ContractError(f"{arm} V2 resolved config violates fixed contract")
    return config


def _instantiate_model(config: Any, checkpoint: Mapping[str, Any], device: torch.device):
    model = hydra.utils.instantiate(config.model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _evaluate_model(model, dataset, device: torch.device) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sessions: list[str] = []
    batch_sizes: list[int] = []
    before = state_hash(model.state_dict())
    with torch.no_grad():
        for neural, target, identity, session_name, carrier in loader:
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
            predictions.append(output[:, -1, :].cpu().numpy())
            targets.append(target[:, -1, :].cpu().numpy())
            sessions.extend(list(session_name))
            batch_sizes.append(int(output.shape[0]))
    after = state_hash(model.state_dict())
    assert_state_immutable(before, after, f"{model.pilot_arm}/{dataset.intervention}")
    if not batch_sizes or sum(batch_sizes) != len(dataset):
        raise NormalizedV2ContractError("V2 terminal evaluator omitted target samples")
    if batch_sizes[-1] != (len(dataset) % 32 or 32):
        raise NormalizedV2ContractError("V2 evaluator did not preserve final remainder batch")
    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    truth = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    sse = float(np.square(truth - estimate).sum())
    centered = truth - truth.mean(axis=0, keepdims=True)
    tss = float(np.square(centered).sum())
    if not np.isfinite(sse) or not np.isfinite(tss) or tss <= 0:
        raise NormalizedV2ContractError("V2 variance-weighted R2 is undefined")
    counts = {name: sessions.count(name) for name in H1_M4_FOLD0_TARGET}
    if set(sessions) != set(H1_M4_FOLD0_TARGET) or sum(counts.values()) != len(dataset):
        raise NormalizedV2ContractError("V2 terminal target concatenation omitted a fold-0 recording")
    return {
        "r2": float(1.0 - sse / tss),
        "sse": sse,
        "tss": tss,
        "samples": len(dataset),
        "session_samples": counts,
        "batches": len(batch_sizes),
        "last_batch_size": batch_sizes[-1],
        "model_state_sha256_before": before,
        "model_state_sha256_after": after,
        "state_immutable": before == after,
        "query_window_indices_sha256": dataset.window_indices_sha256,
    }


def evaluate_terminal_pilot(
    *,
    data_dir: str | Path,
    raw_receipt_path: str | Path,
    eb_receipt_path: str | Path,
    shared_cache_dir: str | Path,
    base_checkpoint_path: str | Path,
    joint_checkpoint_path: str | Path,
    base_config_path: str | Path,
    joint_config_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    if device not in {"cuda", "cpu"}:
        raise ValueError("normalized V2 evaluator device must be cuda or cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("normalized V2 evaluator requested cuda but CUDA is unavailable")
    evaluation_device = torch.device(device)
    base_config_path = Path(base_config_path).resolve()
    joint_config_path = Path(joint_config_path).resolve()
    # Config and checkpoint validation is deliberately before source/target
    # data access, with target access later still than source manifest binding.
    base_config = _validate_resolved_config(base_config_path, "base")
    joint_config = _validate_resolved_config(joint_config_path, "joint")
    base_checkpoint, base_meta = load_and_validate_terminal_checkpoint(
        base_checkpoint_path, base_config_path, expected_arm="base"
    )
    joint_checkpoint, joint_meta = load_and_validate_terminal_checkpoint(
        joint_checkpoint_path, joint_config_path, expected_arm="joint"
    )
    paired = validate_paired_checkpoint_bindings(base_meta, joint_meta)

    source_module = H1M4EBNormalizedV2DataModule(
        task="h1",
        data_dir=str(Path(data_dir).resolve()),
        raw_receipt_path=str(Path(raw_receipt_path).resolve()),
        eb_receipt_path=str(Path(eb_receipt_path).resolve()),
        cache_dir=str(Path(shared_cache_dir).resolve()),
    )
    source_module.setup("fit")
    manifest = source_module.pilot_manifest()
    if source_module.pilot_manifest_sha256 != paired["source_manifest_sha256"]:
        raise NormalizedV2ContractError("runtime V2 source manifest does not match both checkpoints")
    if source_module.normalizer.normalizer_sha256 != paired["normalizer_sha256"]:
        raise NormalizedV2ContractError("runtime V2 normalizer does not match both checkpoints")
    if manifest["carrier_cache_sha256"] != paired["source_cache_sha256"]:
        raise NormalizedV2ContractError("runtime V2 raw source cache does not match checkpoints")
    if manifest["normalized_cache_sha256"] != paired["normalized_cache_sha256"]:
        raise NormalizedV2ContractError("runtime V2 normalized source cache does not match checkpoints")

    base_model = _instantiate_model(base_config, base_checkpoint, evaluation_device)
    joint_model = _instantiate_model(joint_config, joint_checkpoint, evaluation_device)
    del base_checkpoint, joint_checkpoint

    # This is the first target-data access in the evaluator.
    target_records = load_target_records(data_dir)
    validate_target_receipt_binding(target_records, source_module.plan, raw_receipt_path, eb_receipt_path)
    full = H1M4EBNormalizedV2StrictTargetDataset(target_records, source_module.plan, source_module.normalizer, "full")
    variants = {name: full.with_intervention(name) for name in full.INTERVENTIONS}
    support_hashes = full.support_and_carrier_hashes()
    for name, dataset in variants.items():
        if dataset.window_indices_sha256 != full.window_indices_sha256 or dataset.support_and_carrier_hashes() != support_hashes:
            raise NormalizedV2ContractError(f"V2 {name} intervention changed support/query order")

    base_result = _evaluate_model(base_model, variants["full"], evaluation_device)
    joint_results = {name: _evaluate_model(joint_model, variants[name], evaluation_device) for name in variants}
    full_r2 = joint_results["full"]["r2"]
    clauses = {
        "full_gt_zero": full_r2 > 0.0,
        "full_minus_base_gt_zero": full_r2 - base_result["r2"] > 0.0,
        "full_minus_zero_gt_zero": full_r2 - joint_results["zero"]["r2"] > 0.0,
        "full_minus_row_gt_zero": full_r2 - joint_results["row"]["r2"] > 0.0,
        "full_minus_label_gt_zero": full_r2 - joint_results["label"]["r2"] > 0.0,
    }
    gate_pass = all(clauses.values())
    receipt = {
        "schema": "h1_m4_eb_normalized_v2_fold0_terminal_gate_v1",
        "status": "PASS_H1_M4_EB_NORMALIZED_V2_EXPANSION_AUTHORIZED_ONLY" if gate_pass else "STOP_H1_M4_EB_NORMALIZED_V2_TERMINAL_GATE_FAILED",
        "claim_status": "normalized V2 engineering repair; never a decoding or EB attachment claim",
        "fold_date": "19250101",
        "evaluation_device": str(evaluation_device),
        "checkpoint_binding_completed_before_target_open": True,
        "checkpoints": {
            "base": {"path": str(Path(base_checkpoint_path).resolve()), "sha256": sha256_file(base_checkpoint_path), "config_sha256": sha256_file(base_config_path), "metadata": base_meta},
            "joint": {"path": str(Path(joint_checkpoint_path).resolve()), "sha256": sha256_file(joint_checkpoint_path), "config_sha256": sha256_file(joint_config_path), "metadata": joint_meta},
            "paired": paired,
        },
        "source_manifest": manifest,
        "source_manifest_sha256": source_module.pilot_manifest_sha256,
        "normalizer": source_module.normalizer.manifest,
        "target": {
            "sessions": list(H1_M4_FOLD0_TARGET),
            "files": {name: target_records[name].input_sha256 for name in H1_M4_FOLD0_TARGET},
            "normalized_support_and_carrier_hashes": support_hashes,
            "strict_query_window_indices_sha256": full.window_indices_sha256,
            "all_query_histories_start_at_or_after_fifth_trial": True,
            "recordings_concatenated_before_variance_weighted_r2": True,
            "remainder_preserved": True,
        },
        "metrics": {"matched_base_full": base_result, "joint": joint_results},
        "gate": {
            "clauses": clauses,
            "margins": {
                "full": full_r2,
                "full_minus_base": full_r2 - base_result["r2"],
                "full_minus_zero": full_r2 - joint_results["zero"]["r2"],
                "full_minus_row": full_r2 - joint_results["row"]["r2"],
                "full_minus_label": full_r2 - joint_results["label"]["r2"],
            },
            "pass": gate_pass,
            "one_failure_means_stop": True,
        },
        "data_scope": {"opened": "13 public held-in-calib NWBs only (11 source, then 2 target)", "minival_opened": False, "heldout_opened": False, "evalai_opened": False, "formal_heldout_opened": False},
    }
    output, receipt_sha = write_immutable_json(output_path, receipt)
    receipt["receipt_path"] = str(output)
    receipt["receipt_sha256"] = receipt_sha
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "data/000954")
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--eb-receipt", type=Path, required=True)
    parser.add_argument("--shared-cache-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--joint-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    result = evaluate_terminal_pilot(
        data_dir=args.data_dir,
        raw_receipt_path=args.raw_receipt,
        eb_receipt_path=args.eb_receipt,
        shared_cache_dir=args.shared_cache_dir,
        base_checkpoint_path=args.base_checkpoint,
        joint_checkpoint_path=args.joint_checkpoint,
        base_config_path=args.base_config,
        joint_config_path=args.joint_config,
        output_path=args.output,
        device=args.device,
    )
    print(json.dumps({"status": result["status"], "receipt": result["receipt_path"], "sha256": result["receipt_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()

