#!/usr/bin/env python3
"""Additive seed-43 H-S/H-C/H-C0 terminal evaluator.

The predecessor ``h1_carrierid_evaluate.py`` is sealed for the seed-42 pilot
and is not modified by this script.  This evaluator uses the same strict
target-window/R² logic, but binds seed-43 configs, checkpoints, source cache,
and the new seed-43 calibration schedule through
``h1_carrierid_seed43_terminal_preflight`` before importing or opening the two
fold-0 target recordings.  Running this module is intentionally a separate
root-authorized action; the no-target preflight is the safe first stage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    assert_immutable_receipt,
    assert_state_immutable,
    sha256_file,
    state_hash,
    write_immutable_json,
)
from src.data.h1_m4_eb_normalized_v2 import H1M4EBNormalizedV2StrictTargetDataset
from src.data.h1_m4_eb_pilot import H1_M4_FOLD0_TARGET, load_target_records, validate_target_receipt_binding
from src.models.components.h1_carrierid_spint import (
    H1_CARRIERID_PARAMETERS,
    H1_CARRIERID_WHOLE_MODEL_PARAMETERS,
    H1_SPINT_ID_PARAMETERS,
    H1_SPINT_WHOLE_MODEL_PARAMETERS,
)
from scripts.h1_carrierid_seed43_terminal_preflight import (
    PREFLIGHT_SCHEMA,
    PREFLIGHT_STATUS,
    _source_closure,
    bind_seed43_terminal_inputs,
)


TERMINAL_SCHEMA = "h1_carrierid_h32_seed43_fold0_terminal_gate_v1"


def _instantiate(config: Any, checkpoint: Mapping[str, Any], device: torch.device):
    model = hydra.utils.instantiate(config.model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _r2(truth: np.ndarray, estimate: np.ndarray) -> float:
    truth64 = np.asarray(truth, dtype=np.float64)
    estimate64 = np.asarray(estimate, dtype=np.float64)
    sse = float(np.square(truth64 - estimate64).sum())
    centered = truth64 - truth64.mean(axis=0, keepdims=True)
    tss = float(np.square(centered).sum())
    if not np.isfinite(sse) or not np.isfinite(tss) or tss <= 0.0:
        raise NormalizedV2ContractError("seed43 R2 is undefined")
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
    assert_state_immutable(before, after, f"seed43 CarrierID {label}")
    if not sizes or sum(sizes) != len(dataset) or sizes[-1] != (len(dataset) % 32 or 32):
        raise NormalizedV2ContractError(f"seed43 {label}: target remainder was omitted")
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    if set(sessions) != set(H1_M4_FOLD0_TARGET):
        raise NormalizedV2ContractError(f"seed43 {label}: target session set drift")
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
        "batches": len(sizes),
        "last_batch_size": sizes[-1],
        "state_sha256_before": before,
        "state_sha256_after": after,
        "state_immutable": before == after,
        "query_window_indices_sha256": dataset.window_indices_sha256,
    }


def _validate_no_target_receipt(path: str | Path, bound: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on the immutable, current-input no-target binding receipt."""

    receipt_path = Path(path).resolve()
    receipt = assert_immutable_receipt(receipt_path, PREFLIGHT_STATUS)
    if receipt.get("schema") != PREFLIGHT_SCHEMA:
        raise NormalizedV2ContractError("seed43 no-target receipt schema drift")
    scope = receipt.get("scope", {})
    for key, expected in {
        "target_recordings_opened": 0,
        "target_recordings_enumerated": 0,
        "minival_opened_or_enumerated": False,
        "formal_heldout_opened_or_enumerated": False,
        "evalai_opened_or_enumerated": False,
        "final_evaluation_run": False,
    }.items():
        if scope.get(key) != expected:
            raise NormalizedV2ContractError(f"seed43 no-target receipt scope drift at {key}")
    if receipt.get("launch", {}).get("final_evaluator_authorized") is not False:
        raise NormalizedV2ContractError("seed43 no-target receipt unexpectedly authorizes evaluation")
    expected_closure = _source_closure()
    if receipt.get("source_closure") != expected_closure:
        raise NormalizedV2ContractError("seed43 no-target receipt source closure drift")
    if receipt.get("sealed_seed42_evaluator_sha256") != expected_closure["scripts/h1_carrierid_evaluate.py"]:
        raise NormalizedV2ContractError("sealed seed42 evaluator hash drift")
    checkpoint_receipts = receipt.get("checkpoints", {})
    for arm in ("hs", "hc", "hc0"):
        current = bound["arms"][arm]
        recorded = checkpoint_receipts.get(arm)
        if recorded is None:
            raise NormalizedV2ContractError(f"seed43 no-target receipt lacks {arm} checkpoint binding")
        if recorded.get("path") != current["path"] or recorded.get("sha256") != sha256_file(current["path"]):
            raise NormalizedV2ContractError(f"seed43 no-target {arm} checkpoint SHA/path drift")
        if recorded.get("config_path") != current["config_path"] or recorded.get("config_sha256") != sha256_file(current["config_path"]):
            raise NormalizedV2ContractError(f"seed43 no-target {arm} config SHA/path drift")
    source = receipt.get("source", {})
    if source.get("manifest_sha256") != bound["source_manifest_sha256"]:
        raise NormalizedV2ContractError("seed43 no-target source manifest SHA drift")
    if source.get("calibration_schedule_sha256") != bound["source_manifest"]["calibration_schedule_sha256"]:
        raise NormalizedV2ContractError("seed43 no-target source schedule SHA drift")
    if source.get("seed42_schedule_sha256") != bound["preflight"]["seed42_comparison"]["seed42_calibration_schedule_sha256"]:
        raise NormalizedV2ContractError("seed43 no-target seed42 comparison schedule drift")
    if receipt.get("initialization_binding") != bound.get("initialization_binding"):
        raise NormalizedV2ContractError("seed43 no-target initialization-binding drift")
    return receipt


def evaluate_terminal_seed43(
    *,
    source_preflight_path: str | Path,
    no_target_preflight_path: str | Path,
    data_dir: str | Path,
    raw_receipt_path: str | Path,
    eb_receipt_path: str | Path,
    shared_cache_dir: str | Path,
    hs_checkpoint_path: str | Path,
    hc_checkpoint_path: str | Path,
    hc0_checkpoint_path: str | Path,
    hs_config_path: str | Path,
    hc_config_path: str | Path,
    hc0_config_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    if device not in {"cuda", "cpu"}:
        raise ValueError("seed43 evaluator device must be cuda or cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("seed43 evaluator requested CUDA but CUDA is unavailable")
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable seed43 terminal evaluator receipt: {output}")
    bound = bind_seed43_terminal_inputs(
        preflight_path=source_preflight_path,
        data_dir=data_dir,
        raw_receipt_path=raw_receipt_path,
        eb_receipt_path=eb_receipt_path,
        shared_cache_dir=shared_cache_dir,
        checkpoints={"hs": hs_checkpoint_path, "hc": hc_checkpoint_path, "hc0": hc0_checkpoint_path},
        configs={"hs": hs_config_path, "hc": hc_config_path, "hc0": hc0_config_path},
    )
    no_target_path = Path(no_target_preflight_path).resolve()
    if not no_target_path.is_file():
        raise FileNotFoundError(no_target_path)
    no_target_receipt = _validate_no_target_receipt(no_target_path, bound)
    evaluation_device = torch.device(device)
    hs_model = _instantiate(bound["configs"]["hs"], bound["arms"]["hs"]["checkpoint"], evaluation_device)
    hc_model = _instantiate(bound["configs"]["hc"], bound["arms"]["hc"]["checkpoint"], evaluation_device)
    hc0_model = _instantiate(bound["configs"]["hc0"], bound["arms"]["hc0"]["checkpoint"], evaluation_device)
    del bound["arms"]["hs"]["checkpoint"], bound["arms"]["hc"]["checkpoint"], bound["arms"]["hc0"]["checkpoint"]

    # This is intentionally the first and only target access.  All checkpoint,
    # config, source-cache, and no-target preflight bindings above complete first.
    target_records = load_target_records(data_dir)
    validate_target_receipt_binding(target_records, bound["source"].plan, raw_receipt_path, eb_receipt_path)
    full_dataset = H1M4EBNormalizedV2StrictTargetDataset(
        target_records, bound["source"].plan, bound["source"].normalizer, "full"
    )
    variants = {name: full_dataset.with_intervention(name) for name in full_dataset.INTERVENTIONS}
    support_hashes = full_dataset.support_and_carrier_hashes()
    for name, dataset in variants.items():
        if dataset.window_indices_sha256 != full_dataset.window_indices_sha256 or dataset.support_and_carrier_hashes() != support_hashes:
            raise NormalizedV2ContractError(f"seed43 intervention {name} changed support/query identity")

    hs = _evaluate(hs_model, variants["full"], evaluation_device, "H-S")
    hc = {name: _evaluate(hc_model, dataset, evaluation_device, f"H-C/{name}") for name, dataset in variants.items()}
    hc0 = _evaluate(hc0_model, variants["full"], evaluation_device, "H-C0")
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
        "status": "PASS_H1_CARRIERID_H32_SEED43_TERMINAL_EVALUATED" if all(clauses.values()) else "STOP_H1_CARRIERID_H32_SEED43_TERMINAL_GATE_FAILED",
        "claim_status": "additive fixed h32 seed-43 source-only session-identity encoder pilot; target-session optimizer/backpropagation steps are zero for H-S, H-C, and H-C0",
        "metric_contract": {
            "prediction_dtype": "float32",
            "r2_sse_tss_accumulator_dtype": "float64",
            "matches_h1_m4_eb_normalized_v2_evaluator": True,
        },
        "fold_date": "19250101",
        "seed": 43,
        "evaluation_device": str(evaluation_device),
        "checkpoint_binding_completed_before_target_open": True,
        "no_target_preflight": {
            "path": str(no_target_path),
            "sha256": sha256_file(no_target_path),
            "status": no_target_receipt["status"],
        },
        "checkpoints": {
            "h_s_matched_spint": {"path": bound["arms"]["hs"]["path"], "sha256": sha256_file(bound["arms"]["hs"]["path"]), "config_sha256": sha256_file(bound["arms"]["hs"]["config_path"]), "metadata": bound["arms"]["hs"]["metadata"]},
            "h_c_full": {"path": bound["arms"]["hc"]["path"], "sha256": sha256_file(bound["arms"]["hc"]["path"]), "config_sha256": sha256_file(bound["arms"]["hc"]["config_path"]), "metadata": bound["arms"]["hc"]["metadata"]},
            "h_c0_separate_literal_zero": {"path": bound["arms"]["hc0"]["path"], "sha256": sha256_file(bound["arms"]["hc0"]["path"]), "config_sha256": sha256_file(bound["arms"]["hc0"]["config_path"]), "metadata": bound["arms"]["hc0"]["metadata"]},
        },
        "source": {
            "manifest": bound["source_manifest"],
            "manifest_sha256": bound["source_manifest_sha256"],
            "calibration_schedule_sha256": bound["source_manifest"]["calibration_schedule_sha256"],
            "seed42_schedule_sha256": bound["preflight"]["seed42_comparison"]["seed42_calibration_schedule_sha256"],
            "schedule_is_new_for_seed43": True,
        },
        "target": {
            "sessions": list(H1_M4_FOLD0_TARGET),
            "files": {name: target_records[name].input_sha256 for name in H1_M4_FOLD0_TARGET},
            "strict_query_window_indices_sha256": full_dataset.window_indices_sha256,
            "support_and_carrier_hashes": support_hashes,
            "all_query_histories_start_at_or_after_fifth_trial": True,
            "pooled_recordings_before_variance_weighted_r2": True,
            "remainder_preserved": True,
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
            "clauses": clauses,
            "pass": all(clauses.values()),
            "one_failure_means_stop": True,
            "margins": {
                "h_c_minus_h_s": full_r2 - hs["pooled_r2"],
                "h_c_minus_h_c0": full_r2 - hc0["pooled_r2"],
                "h_c_minus_same_checkpoint_zero": full_r2 - hc["zero"]["pooled_r2"],
                "h_c_minus_same_checkpoint_row": full_r2 - hc["row"]["pooled_r2"],
                "h_c_minus_same_checkpoint_label": full_r2 - hc["label"]["pooled_r2"],
            },
        },
        "data_scope": {"opened": "13 public held-in-calib NWBs only (11 source then 2 target)", "minival_opened": False, "heldout_opened": False, "formal_heldout_opened": False, "evalai_opened": False},
    }
    output, digest = write_immutable_json(output_path, receipt)
    return {**receipt, "receipt_path": str(output), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-preflight", required=True, type=Path)
    parser.add_argument("--no-target-preflight", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/000954")
    parser.add_argument("--raw-receipt", required=True, type=Path)
    parser.add_argument("--eb-receipt", required=True, type=Path)
    parser.add_argument("--shared-cache-dir", required=True, type=Path)
    for arm in ("hs", "hc", "hc0"):
        parser.add_argument(f"--{arm}-checkpoint", required=True, type=Path)
        parser.add_argument(f"--{arm}-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    result = evaluate_terminal_seed43(
        source_preflight_path=args.source_preflight,
        no_target_preflight_path=args.no_target_preflight,
        data_dir=args.data_dir,
        raw_receipt_path=args.raw_receipt,
        eb_receipt_path=args.eb_receipt,
        shared_cache_dir=args.shared_cache_dir,
        hs_checkpoint_path=args.hs_checkpoint,
        hc_checkpoint_path=args.hc_checkpoint,
        hc0_checkpoint_path=args.hc0_checkpoint,
        hs_config_path=args.hs_config,
        hc_config_path=args.hc_config,
        hc0_config_path=args.hc0_config,
        output_path=args.output,
        device=args.device,
    )
    print(json.dumps({"status": result["status"], "receipt": result["receipt_path"], "sha256": result["receipt_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
