#!/usr/bin/env python3
"""Terminal held-in-calibration evaluator for the fixed-h32 CarrierID pilot.

This is the only script permitted to open the two public fold-0 target
recordings.  It first binds all three epoch-49 checkpoints and their resolved
configs, then constructs the source-only normalized cache, and only then opens
the target.  Formal/minival/held-out/EvalAI access is absent by construction.
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

from src.data.h1_m4_eb_normalized_v2 import H1M4EBNormalizedV2DataModule, H1M4EBNormalizedV2StrictTargetDataset
from src.data.h1_m4_eb_pilot import H1_M4_FOLD0_TARGET, load_target_records, validate_target_receipt_binding
from src.h1_m4_eb_normalized_v2_contract import (
    NORMALIZER_FLOOR,
    NORMALIZER_FORMULA,
    NormalizedV2ContractError,
    assert_state_immutable,
    load_and_validate_terminal_checkpoint,
    sha256_file,
    state_hash,
    write_immutable_json,
)
from src.models.components.h1_carrierid_spint import (
    H1_CARRIERID_PARAMETERS,
    H1_CARRIERID_WHOLE_MODEL_PARAMETERS,
    H1_SPINT_ID_PARAMETERS,
    H1_SPINT_WHOLE_MODEL_PARAMETERS,
)
from src.models.h1_carrierid_module import H1_CARRIERID_CHECKPOINT_SCHEMA


TERMINAL_SCHEMA = "h1_carrierid_h32_fold0_terminal_gate_v1"
MATCHED_HS_SCORE = 0.4968330503


def _validate_carrierid_config(path: Path, arm: str) -> Any:
    config = OmegaConf.load(path)
    fixed = {"seed": 42, "train": True, "test": False, "ckpt_path": None}
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise NormalizedV2ContractError(f"CarrierID {arm} resolved config drift at {key}")
    if (
        config.pilot.arm != arm
        or bool(config.pilot.zero_carrier) != (arm == "zero")
        or str(config.pilot.fold_date) != "19250101"
        or int(config.data.calibration_n_trials) != 4
        or int(config.data.max_trial_length) != 1024
        or int(config.data.window_size) != 700
        or int(config.data.batch_size) != 32
        or int(config.trainer.max_epochs) != 50
        or int(config.trainer.min_epochs) != 50
        or int(config.trainer.limit_val_batches) != 0
        or str(config.trainer.precision) != "32-true"
        or float(config.model.optimizer.lr) != 5.0e-5
        or float(config.data.normalizer_floor) != NORMALIZER_FLOOR
        or int(config.model.net.carrier_hidden_dim) != 32
        or int(config.model.net.carrier_dim) != 4
        or int(config.model.net.carrier_trial_length) != 1024
    ):
        raise NormalizedV2ContractError(f"CarrierID {arm} resolved config violates fixed h32 contract")
    return config


def _load_carrierid_checkpoint(path: Path, config_path: Path, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file() or not config_path.is_file():
        raise FileNotFoundError(path if not path.is_file() else config_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise NormalizedV2ContractError("CarrierID checkpoint is not a Lightning state_dict checkpoint")
    if int(checkpoint.get("epoch", -1)) != 49 or int(checkpoint.get("global_step", 0)) <= 0:
        raise NormalizedV2ContractError("CarrierID checkpoint must be real fixed terminal epoch=49")
    meta = checkpoint.get("h1_carrierid")
    if not isinstance(meta, dict):
        raise NormalizedV2ContractError("CarrierID checkpoint lacks h1_carrierid metadata")
    required = {
        "schema": H1_CARRIERID_CHECKPOINT_SCHEMA,
        "fold_date": "19250101",
        "arm": arm,
        "checkpoint_epoch_zero_based": 49,
        "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_selection",
        "normalizer_formula": NORMALIZER_FORMULA,
        "normalizer_floor": NORMALIZER_FLOOR,
        "carrier_hidden_dim": 32,
        "carrier_dim": 4,
        "carrier_trial_length": 1024,
        "carrierid_parameters": H1_CARRIERID_PARAMETERS,
        "whole_model_parameters": H1_CARRIERID_WHOLE_MODEL_PARAMETERS,
        "spint_identity_parameters": H1_SPINT_ID_PARAMETERS,
        "spint_whole_model_parameters": H1_SPINT_WHOLE_MODEL_PARAMETERS,
        "deployment_target_optimizer_steps": 0,
        "deployment_target_backward_steps": 0,
        "carrier_mode": "full" if arm == "full" else "literal_zero_at_model_boundary",
    }
    for key, expected in required.items():
        if meta.get(key) != expected:
            raise NormalizedV2ContractError(f"CarrierID {arm} checkpoint metadata drift at {key}")
    if sha256_file(config_path) != meta.get("config_sha256"):
        raise NormalizedV2ContractError(f"CarrierID {arm} config SHA does not bind checkpoint")
    return checkpoint, meta


def _validate_candidate_pair(full: Mapping[str, Any], zero: Mapping[str, Any]) -> dict[str, str]:
    fields = (
        "fold_date", "source_manifest_sha256", "normalizer_sha256", "source_cache_sha256",
        "normalized_cache_sha256", "source_hashes_sha256", "initial_state_sha256",
    )
    for field in fields:
        if full.get(field) != zero.get(field):
            raise NormalizedV2ContractError(f"CarrierID H-C/H-C0 binding mismatch at {field}")
    return {field: str(full[field]) for field in fields}


def _instantiate(config: Any, checkpoint: Mapping[str, Any], device: torch.device):
    model = hydra.utils.instantiate(config.model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _r2(truth: np.ndarray, estimate: np.ndarray) -> float:
    # Match the established normalized-V2 evaluator exactly: model outputs
    # are float32, but the pooled SSE/TSS accumulation is float64.  Without
    # this explicit conversion NumPy accumulates float32 here and changes the
    # last few reported digits (the model predictions themselves are
    # unchanged).
    truth64 = np.asarray(truth, dtype=np.float64)
    estimate64 = np.asarray(estimate, dtype=np.float64)
    sse = float(np.square(truth64 - estimate64).sum())
    centered = truth64 - truth64.mean(axis=0, keepdims=True)
    tss = float(np.square(centered).sum())
    if not np.isfinite(sse) or not np.isfinite(tss) or tss <= 0.0:
        raise NormalizedV2ContractError("CarrierID R2 is undefined")
    return float(1.0 - sse / tss)


def _evaluate(model, dataset, device: torch.device, label: str) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sessions: list[str] = []
    sizes: list[int] = []
    before = state_hash(model.state_dict())
    with torch.no_grad():
        for neural, target, identity, session_name, carrier in loader:
            output = model(
                neural.to(device=device, dtype=torch.float32),
                calib_trialized_neural_features=identity.to(device=device, dtype=torch.float32),
                carrier=carrier.to(device=device, dtype=torch.float32),
            )
            if model.hparams.decode_last_timestep_only:
                output, target = output[:, -1:, :], target[:, -1:, :]
            if model.hparams.predict_scaled_behavior:
                output = output / model.hparams.behavior_scaling_factor
            predictions.append(output[:, -1, :].cpu().numpy())
            targets.append(target[:, -1, :].cpu().numpy())
            sessions.extend(list(session_name))
            sizes.append(int(output.shape[0]))
    after = state_hash(model.state_dict())
    assert_state_immutable(before, after, f"CarrierID {label}")
    if not sizes or sum(sizes) != len(dataset) or sizes[-1] != (len(dataset) % 32 or 32):
        raise NormalizedV2ContractError("CarrierID target evaluator omitted remainder or samples")
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    if set(sessions) != set(H1_M4_FOLD0_TARGET):
        raise NormalizedV2ContractError("CarrierID evaluator omitted a fold-0 target recording")
    per_session: dict[str, Any] = {}
    for session in H1_M4_FOLD0_TARGET:
        indices = np.asarray([name == session for name in sessions], dtype=bool)
        per_session[session] = {"samples": int(indices.sum()), "r2": _r2(target[indices], prediction[indices])}
    return {
        "pooled_r2": _r2(target, prediction),
        "r2_accumulator_dtype": "float64",
        "samples": len(dataset),
        "session_samples": {name: sessions.count(name) for name in H1_M4_FOLD0_TARGET},
        "per_session": per_session,
        "batches": len(sizes), "last_batch_size": sizes[-1],
        "state_sha256_before": before, "state_sha256_after": after, "state_immutable": before == after,
        "query_window_indices_sha256": dataset.window_indices_sha256,
    }


def evaluate_terminal_pilot(
    *,
    data_dir: str | Path,
    raw_receipt_path: str | Path,
    eb_receipt_path: str | Path,
    shared_cache_dir: str | Path,
    full_checkpoint_path: str | Path,
    zero_checkpoint_path: str | Path,
    full_config_path: str | Path,
    zero_config_path: str | Path,
    spint_checkpoint_path: str | Path,
    spint_config_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    if device not in {"cuda", "cpu"}:
        raise ValueError("CarrierID evaluator device must be cuda or cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CarrierID evaluator requested CUDA but CUDA is unavailable")
    evaluation_device = torch.device(device)
    full_cfg_path, zero_cfg_path, spint_cfg_path = map(Path, (full_config_path, zero_config_path, spint_config_path))
    full_config = _validate_carrierid_config(full_cfg_path, "full")
    zero_config = _validate_carrierid_config(zero_cfg_path, "zero")
    # All checkpoint/config binding occurs before source or target data access.
    full_checkpoint, full_meta = _load_carrierid_checkpoint(Path(full_checkpoint_path), full_cfg_path, "full")
    zero_checkpoint, zero_meta = _load_carrierid_checkpoint(Path(zero_checkpoint_path), zero_cfg_path, "zero")
    pair = _validate_candidate_pair(full_meta, zero_meta)
    spint_config = OmegaConf.load(spint_cfg_path)
    spint_checkpoint, spint_meta = load_and_validate_terminal_checkpoint(
        spint_checkpoint_path, spint_cfg_path, expected_arm="base"
    )
    for field in ("fold_date", "normalizer_sha256", "source_cache_sha256", "normalized_cache_sha256", "source_hashes_sha256"):
        if full_meta.get(field) != spint_meta.get(field):
            raise NormalizedV2ContractError(f"CarrierID H-S reference differs from H-C source binding at {field}")

    source = H1M4EBNormalizedV2DataModule(
        task="h1", data_dir=str(Path(data_dir).resolve()), raw_receipt_path=str(Path(raw_receipt_path).resolve()),
        eb_receipt_path=str(Path(eb_receipt_path).resolve()), cache_dir=str(Path(shared_cache_dir).resolve()),
    )
    source.setup("fit")
    manifest = source.pilot_manifest()
    for meta in (full_meta, zero_meta):
        if meta["source_manifest_sha256"] != source.pilot_manifest_sha256:
            raise NormalizedV2ContractError("CarrierID runtime source manifest does not bind checkpoint")
        if meta["normalizer_sha256"] != source.normalizer.normalizer_sha256:
            raise NormalizedV2ContractError("CarrierID runtime normalizer does not bind checkpoint")

    full_model = _instantiate(full_config, full_checkpoint, evaluation_device)
    zero_model = _instantiate(zero_config, zero_checkpoint, evaluation_device)
    spint_model = _instantiate(spint_config, spint_checkpoint, evaluation_device)
    del full_checkpoint, zero_checkpoint, spint_checkpoint

    # First and only target data access.
    target_records = load_target_records(data_dir)
    validate_target_receipt_binding(target_records, source.plan, raw_receipt_path, eb_receipt_path)
    full_dataset = H1M4EBNormalizedV2StrictTargetDataset(target_records, source.plan, source.normalizer, "full")
    variants = {name: full_dataset.with_intervention(name) for name in full_dataset.INTERVENTIONS}
    support_hashes = full_dataset.support_and_carrier_hashes()
    for name, dataset in variants.items():
        if dataset.window_indices_sha256 != full_dataset.window_indices_sha256 or dataset.support_and_carrier_hashes() != support_hashes:
            raise NormalizedV2ContractError(f"CarrierID intervention {name} changed support/query identity")

    hs = _evaluate(spint_model, variants["full"], evaluation_device, "H-S")
    hc = {name: _evaluate(full_model, dataset, evaluation_device, f"H-C/{name}") for name, dataset in variants.items()}
    hc0 = _evaluate(zero_model, variants["full"], evaluation_device, "H-C0")
    full_r2 = hc["full"]["pooled_r2"]
    clauses = {
        "h_c_gt_matched_h_s": full_r2 > hs["pooled_r2"],
        "h_c_gt_separately_trained_h_c0": full_r2 > hc0["pooled_r2"],
        "h_c_gt_same_checkpoint_zero": full_r2 > hc["zero"]["pooled_r2"],
        "h_c_gt_same_checkpoint_row": full_r2 > hc["row"]["pooled_r2"],
        "h_c_gt_same_checkpoint_label": full_r2 > hc["label"]["pooled_r2"],
    }
    receipt = {
        "schema": TERMINAL_SCHEMA,
        "status": "PASS_H1_CARRIERID_H32_EXPANSION_AUTHORIZED" if all(clauses.values()) else "STOP_H1_CARRIERID_H32_TERMINAL_GATE_FAILED",
        "claim_status": "fixed h32 parameter-efficient session-identity encoder pilot; target-session optimizer/backpropagation steps are zero for H-S, H-C, and H-C0",
        "metric_contract": {
            "prediction_dtype": "float32",
            "r2_sse_tss_accumulator_dtype": "float64",
            "matches_h1_m4_eb_normalized_v2_evaluator": True,
        },
        "fold_date": "19250101", "evaluation_device": str(evaluation_device),
        "checkpoint_binding_completed_before_target_open": True,
        "checkpoints": {
            "h_s_matched_spint": {"path": str(Path(spint_checkpoint_path).resolve()), "sha256": sha256_file(spint_checkpoint_path), "config_sha256": sha256_file(spint_cfg_path), "metadata": spint_meta, "previous_reported_pooled_r2": MATCHED_HS_SCORE},
            "h_c_full": {"path": str(Path(full_checkpoint_path).resolve()), "sha256": sha256_file(full_checkpoint_path), "config_sha256": sha256_file(full_cfg_path), "metadata": full_meta},
            "h_c0_separate_literal_zero": {"path": str(Path(zero_checkpoint_path).resolve()), "sha256": sha256_file(zero_checkpoint_path), "config_sha256": sha256_file(zero_cfg_path), "metadata": zero_meta},
            "h_c_pair_binding": pair,
        },
        "source_manifest": manifest, "source_manifest_sha256": source.pilot_manifest_sha256,
        "target": {
            "sessions": list(H1_M4_FOLD0_TARGET), "files": {name: target_records[name].input_sha256 for name in H1_M4_FOLD0_TARGET},
            "strict_query_window_indices_sha256": full_dataset.window_indices_sha256,
            "support_and_carrier_hashes": support_hashes, "all_query_histories_start_at_or_after_fifth_trial": True,
            "pooled_recordings_before_variance_weighted_r2": True, "remainder_preserved": True,
        },
        "metrics": {"h_s": hs, "h_c_interventions": hc, "h_c0": hc0},
        "parameter_accounting": {
            "static_session_identity_encoder_parameters": {"h_s_spint": H1_SPINT_ID_PARAMETERS, "h_c_h32": H1_CARRIERID_PARAMETERS, "h_s_to_h_c_ratio": H1_SPINT_ID_PARAMETERS / H1_CARRIERID_PARAMETERS},
            "whole_model_parameters": {"h_s_spint": H1_SPINT_WHOLE_MODEL_PARAMETERS, "h_c_h32": H1_CARRIERID_WHOLE_MODEL_PARAMETERS, "h_s_to_h_c_ratio": H1_SPINT_WHOLE_MODEL_PARAMETERS / H1_CARRIERID_WHOLE_MODEL_PARAMETERS},
            "target_session_updated_parameters": {"h_s": 0, "h_c": 0, "h_c0": 0},
            "target_session_optimizer_steps": {"h_s": 0, "h_c": 0, "h_c0": 0},
            "target_session_backward_steps": {"h_s": 0, "h_c": 0, "h_c0": 0},
        },
        "gate": {
            "clauses": clauses, "pass": all(clauses.values()), "one_failure_means_stop_no_RS_LS_date_or_formal_expansion": True,
            "margins": {
                "h_c_minus_h_s": full_r2 - hs["pooled_r2"], "h_c_minus_h_c0": full_r2 - hc0["pooled_r2"],
                "h_c_minus_same_checkpoint_zero": full_r2 - hc["zero"]["pooled_r2"],
                "h_c_minus_same_checkpoint_row": full_r2 - hc["row"]["pooled_r2"],
                "h_c_minus_same_checkpoint_label": full_r2 - hc["label"]["pooled_r2"],
            },
            "secondary_noninferiority_only": {"margin": -0.03, "h_c_minus_h_s_at_least_margin": full_r2 - hs["pooled_r2"] >= -0.03, "cannot_substitute_for_superiority": True},
        },
        "data_scope": {"opened": "13 public held-in-calib NWBs only (11 source then 2 target)", "minival_opened": False, "heldout_opened": False, "formal_heldout_opened": False, "evalai_opened": False},
    }
    output, digest = write_immutable_json(output_path, receipt)
    return {**receipt, "receipt_path": str(output), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/000954")
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--eb-receipt", type=Path, required=True)
    parser.add_argument("--shared-cache-dir", type=Path, required=True)
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--zero-checkpoint", type=Path, required=True)
    parser.add_argument("--full-config", type=Path, required=True)
    parser.add_argument("--zero-config", type=Path, required=True)
    parser.add_argument("--spint-checkpoint", type=Path, required=True)
    parser.add_argument("--spint-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    result = evaluate_terminal_pilot(
        data_dir=args.data_dir, raw_receipt_path=args.raw_receipt, eb_receipt_path=args.eb_receipt,
        shared_cache_dir=args.shared_cache_dir, full_checkpoint_path=args.full_checkpoint,
        zero_checkpoint_path=args.zero_checkpoint, full_config_path=args.full_config,
        zero_config_path=args.zero_config, spint_checkpoint_path=args.spint_checkpoint,
        spint_config_path=args.spint_config, output_path=args.output, device=args.device,
    )
    print(json.dumps({"status": result["status"], "receipt": result["receipt_path"], "sha256": result["receipt_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
