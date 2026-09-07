#!/usr/bin/env python3
"""One-shot H-C re-evaluation on the D-S4e/D-Q4e ``bab4b904`` window pool.

This is deliberately a *development diagnostic*, not a replacement H1
endpoint.  The two historical H1 target datasets enumerate the same strict
post-support window list but encode its hash differently: the sealed H-C
evaluator reports binary ``665fe535...`` and D-S4e/D-Q4e reports canonical-JSON
``bab4b904...``.  This script asserts list equality rather than inferring it
from counts/boundaries, then compares a frozen H-C checkpoint on the D-S4e
dataset representation.  It does not authorize EST4, CI64, model selection,
or a paper result.

Target data are inaccessible unless the explicit command-line acknowledgement
is supplied.  Before that boundary, this script binds: (1) the immutable
float64 H-C terminal gate and its one permitted full checkpoint/config, and
(2) the immutable D-S4e/D-Q4e terminal receipt and its target-file/window
binding.  It then reconstructs two source-only objects: the ordinary H-C
source normalizer/carrier path, and the exposure source plan that defines the
``bab4b904`` target-dataset convention.  The target carrier uses the former,
never the D-S4e 6.8396e-6 paired normalizer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.h1_carrierid_distribution_exposure import H1CarrierIdDistributionExposureDataModule
from src.data.h1_carrierid_distribution_target import H1CarrierIdDistributionStrictTargetDataset
from src.data.h1_m4_eb_normalized_v2 import (
    H1M4EBNormalizedV2DataModule,
    H1M4EBNormalizedV2StrictTargetDataset,
)
from src.data.h1_m4_eb_pilot import H1_M4_FOLD0_TARGET
from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    assert_immutable_receipt,
    sha256_file,
    write_immutable_json,
)
from scripts.h1_carrierid_evaluate import (
    TERMINAL_SCHEMA as H_C_GATE_SCHEMA,
    _evaluate,
    _load_carrierid_checkpoint,
    _validate_carrierid_config,
)
from scripts.h1_carrierid_distribution_exposure_terminal_evaluate import (
    TERMINAL_SCHEMA as EXPOSURE_TERMINAL_SCHEMA,
    TERMINAL_STATUS as EXPOSURE_TERMINAL_STATUS,
)


H_C_FLOAT64_GATE = (
    ROOT / "pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/"
    "H1_CARRIERID_H32_FOLD0_TERMINAL_GATE_FLOAT64_R2.json"
)
H_C_FULL_CHECKPOINT = (
    ROOT / "pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/full/"
    "checkpoints/fixed_epoch50/epoch_049.ckpt"
)
H_C_FULL_CONFIG = ROOT / "pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/full/.hydra/config.yaml"
D_S4E_D_Q4E_TERMINAL = (
    ROOT / "pilot_artifacts/h1_carrierid_distribution_exposure/"
    "H1_CARRIERID_H32_FRESH_D_S4E_D_Q4E_TERMINAL_EVALUATION_v1.json"
)
D_S4E_CONFIG = (
    ROOT / "pilot_artifacts/h1_carrierid_distribution_exposure/gpu_runs/"
    "d_s4e_d_q4e_s42_v1/s4e/.hydra/config.yaml"
)
OUTPUT = (
    ROOT / "pilot_artifacts/h1_carrierid_quality/"
    "H1_CARRIERID_HC_FULL_ON_BAB4B904_DEVELOPMENT_DIAGNOSTIC_v1.json"
)

H_C_GATE_STATUS = "PASS_H1_CARRIERID_H32_EXPANSION_AUTHORIZED"
EXPECTED_BAB4_QUERY_HASH = "bab4b904c16f91a017159e6296efb04a252fb346359748b7aa6b9f942137596f"
EXPECTED_665_QUERY_HASH = "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"
DIAGNOSTIC_SCHEMA = "h1_carrierid_hc_full_bab4b904_development_diagnostic_v1"
DIAGNOSTIC_STATUS = "PASS_H1_CARRIERID_HC_FULL_BAB4B904_OPENED_FOLD0_DEVELOPMENT_DIAGNOSTIC"
CLAIM_BOUNDARY = "OPENED_FOLD0_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_PAPER_ENDPOINT_NOT_MODEL_SELECTION"


def _require_metric_contract(metrics: Mapping[str, Any], *, expected_query_hash: str, label: str) -> None:
    """Reject a metric that is not a frozen float64, remainder-preserving result."""

    if metrics.get("r2_accumulator_dtype") != "float64":
        raise NormalizedV2ContractError(f"{label}: R2 SSE/TSS must accumulate in float64")
    if metrics.get("query_window_indices_sha256") != expected_query_hash:
        raise NormalizedV2ContractError(f"{label}: strict query-window hash mismatch")
    if metrics.get("state_immutable") is not True:
        raise NormalizedV2ContractError(f"{label}: model state changed during evaluation")
    if metrics.get("state_sha256_before") != metrics.get("state_sha256_after"):
        raise NormalizedV2ContractError(f"{label}: state hash differs before/after evaluation")
    sessions = metrics.get("per_session", {})
    if set(sessions) != set(H1_M4_FOLD0_TARGET):
        raise NormalizedV2ContractError(f"{label}: metric omitted or added a target recording")
    if int(metrics.get("samples", -1)) != sum(int(sessions[name].get("samples", -1)) for name in H1_M4_FOLD0_TARGET):
        raise NormalizedV2ContractError(f"{label}: pooled sample accounting mismatch")
    if int(metrics.get("last_batch_size", 0)) <= 0:
        raise NormalizedV2ContractError(f"{label}: target remainder was not recorded")


def _metric_delta(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the paired R2 deltas; values are never used for routing."""

    sessions = tuple(H1_M4_FOLD0_TARGET)
    if set(candidate.get("per_session", {})) != set(sessions) or set(reference.get("per_session", {})) != set(sessions):
        raise NormalizedV2ContractError("cannot form a per-session delta with differing session sets")
    return {
        "pooled_r2": float(candidate["pooled_r2"]) - float(reference["pooled_r2"]),
        "per_session_r2": {
            name: float(candidate["per_session"][name]["r2"]) - float(reference["per_session"][name]["r2"])
            for name in sessions
        },
        "interpretation_limit": "paired development diagnostic delta; not a selection, EST4, CI64, or paper-endpoint statistic",
    }


def _assert_same_window_identity_and_carrier(
    ordinary_full: Any, exposure_s4: Any,
) -> dict[str, Any]:
    """Prove that only the window-hash encoding differs between target views.

    The ordinary H-C target class serializes ``(session, start)`` entries into
    a binary SHA-256; the D-S4e target class canonically JSON-serializes the
    same entries.  Counts and first boundaries are insufficient evidence, so
    require exact ordered-list equality as well as bitwise equality of the
    shared identity and ordinary-support carrier tensors.
    """

    if list(ordinary_full.window_indices) != list(exposure_s4.window_indices):
        raise NormalizedV2ContractError("ordinary H-C and D-S4e target window-index lists differ")
    if ordinary_full.window_indices_sha256 != EXPECTED_665_QUERY_HASH:
        raise NormalizedV2ContractError("ordinary H-C target window hash is not 665fe535")
    if exposure_s4.window_indices_sha256 != EXPECTED_BAB4_QUERY_HASH:
        raise NormalizedV2ContractError("D-S4e target window hash is not bab4b904")
    identities: dict[str, bool] = {}
    carriers: dict[str, bool] = {}
    for name in H1_M4_FOLD0_TARGET:
        ordinary_support = ordinary_full.support[name]
        exposure_support = exposure_s4.support[name]
        identity_equal = bool(np.array_equal(ordinary_support.identity, exposure_support.identity))
        carrier_equal = bool(np.array_equal(ordinary_support.carriers["full"], exposure_support.s4_carrier))
        if not identity_equal:
            raise NormalizedV2ContractError(f"{name}: ordinary H-C/D-S4e identities are not bitwise equal")
        if not carrier_equal:
            raise NormalizedV2ContractError(f"{name}: ordinary H-C/D-S4e S4 carriers are not bitwise equal")
        identities[name] = identity_equal
        carriers[name] = carrier_equal
    return {
        "window_index_lists_exactly_equal": True,
        "hash_encoding_difference_only": {
            "ordinary_h_c_binary_window_hash": ordinary_full.window_indices_sha256,
            "d_s4e_canonical_json_window_hash": exposure_s4.window_indices_sha256,
        },
        "identity_arrays_bitwise_equal": identities,
        "ordinary_full_vs_exposure_s4_carrier_arrays_bitwise_equal": carriers,
    }


def _require_hc_gate_binding(gate: Mapping[str, Any]) -> Mapping[str, Any]:
    if gate.get("schema") != H_C_GATE_SCHEMA or gate.get("status") != H_C_GATE_STATUS:
        raise NormalizedV2ContractError("requires the passing immutable H-C float64 terminal gate")
    metric_contract = gate.get("metric_contract", {})
    if metric_contract.get("r2_sse_tss_accumulator_dtype") != "float64":
        raise NormalizedV2ContractError("H-C float64 gate does not bind float64 R2")
    checkpoint = gate.get("checkpoints", {}).get("h_c_full", {})
    if Path(str(checkpoint.get("path", ""))).resolve() != H_C_FULL_CHECKPOINT.resolve():
        raise NormalizedV2ContractError("H-C float64 gate binds a different full checkpoint")
    if checkpoint.get("sha256") != sha256_file(H_C_FULL_CHECKPOINT):
        raise NormalizedV2ContractError("H-C full checkpoint bytes differ from float64 gate")
    if checkpoint.get("config_sha256") != sha256_file(H_C_FULL_CONFIG):
        raise NormalizedV2ContractError("H-C full config bytes differ from float64 gate")
    metadata = checkpoint.get("metadata", {})
    if metadata.get("arm") != "full" or metadata.get("deployment_target_optimizer_steps") != 0 or metadata.get("deployment_target_backward_steps") != 0:
        raise NormalizedV2ContractError("H-C gate lacks the frozen full/no-target-update binding")
    old = gate.get("metrics", {}).get("h_c_interventions", {}).get("full", {})
    _require_metric_contract(old, expected_query_hash=EXPECTED_665_QUERY_HASH, label="sealed H-C on 665fe535")
    return old


def _require_exposure_terminal_binding(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    if receipt.get("schema") != EXPOSURE_TERMINAL_SCHEMA or receipt.get("status") != EXPOSURE_TERMINAL_STATUS:
        raise NormalizedV2ContractError("requires the immutable D-S4e/D-Q4e terminal receipt")
    target = receipt.get("target", {})
    if tuple(target.get("sessions", ())) != tuple(H1_M4_FOLD0_TARGET):
        raise NormalizedV2ContractError("D-S4e receipt has an unexpected target-session scope")
    if target.get("strict_query_window_indices_sha256") != EXPECTED_BAB4_QUERY_HASH:
        raise NormalizedV2ContractError("D-S4e receipt does not bind the bab4b904 window pool")
    files = target.get("files", {})
    if set(files) != set(H1_M4_FOLD0_TARGET) or not all(isinstance(files[name], str) and len(files[name]) == 64 for name in H1_M4_FOLD0_TARGET):
        raise NormalizedV2ContractError("D-S4e receipt has invalid target-file hashes")
    ds4e = receipt.get("metrics", {}).get("d_s4e", {})
    _require_metric_contract(ds4e, expected_query_hash=EXPECTED_BAB4_QUERY_HASH, label="D-S4e on bab4b904")
    updates = receipt.get("target_updates", {})
    if updates.get("optimizer_steps") != 0 or updates.get("backward_steps") != 0 or updates.get("model_state_unchanged") is not True:
        raise NormalizedV2ContractError("D-S4e terminal receipt has non-forward-only target updates")
    return ds4e


def _rebuild_hc_ordinary_source(cfg: Any) -> H1M4EBNormalizedV2DataModule:
    """Rebuild the ordinary H-C source carrier/normalizer only; no target access."""

    data = cfg.data
    source = H1M4EBNormalizedV2DataModule(
        task=str(data.task), data_dir=str(data.data_dir), raw_receipt_path=str(data.raw_receipt_path),
        eb_receipt_path=str(data.eb_receipt_path), cache_dir=str(data.cache_dir),
        batch_size=int(data.batch_size), window_size=int(data.window_size),
        calibration_n_trials=int(data.calibration_n_trials), max_trial_length=int(data.max_trial_length),
        random_calibration=bool(data.random_calibration), smooth_calibration=bool(data.smooth_calibration),
        interpolate_trials=bool(data.interpolate_trials), interpolate_trials_kind=str(data.interpolate_trials_kind),
        num_workers=int(data.num_workers), pin_memory=False, seed=int(data.seed),
        fixed_epochs=int(data.fixed_epochs), normalizer_floor=float(data.normalizer_floor),
    )
    source.setup("fit")
    return source


def _rebuild_exposure_window_source(cfg: Any) -> H1CarrierIdDistributionExposureDataModule:
    """Rebuild only the D-S4e exposure source plan; it must not supply normalization."""

    data = cfg.data
    source = H1CarrierIdDistributionExposureDataModule(
        task=str(data.task), data_dir=str(data.data_dir), raw_receipt_path=str(data.raw_receipt_path),
        eb_receipt_path=str(data.eb_receipt_path), cache_dir=str(data.cache_dir),
        carrier_distribution_arm="s4", batch_size=int(data.batch_size), window_size=int(data.window_size),
        calibration_n_trials=int(data.calibration_n_trials), max_trial_length=int(data.max_trial_length),
        num_workers=int(data.num_workers), pin_memory=False, seed=int(data.seed), fixed_epochs=int(data.fixed_epochs),
        samples_per_epoch=int(data.samples_per_epoch),
    )
    source.setup("fit")
    return source


def _instantiate_hc_frozen(cfg: Any, checkpoint: Mapping[str, Any], device: torch.device):
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise NormalizedV2ContractError("H-C diagnostic model failed to enter frozen eval mode")
    return model


def run_opened_fold0_development_diagnostic(*, device: str = "cuda") -> dict[str, Any]:
    """Run the sole permitted target-opening operation after all bindings hold."""

    if device not in {"cpu", "cuda"}:
        raise ValueError("diagnostic device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("diagnostic requested CUDA but CUDA is unavailable")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite immutable development diagnostic: {OUTPUT}")

    # All of this section is receipt/checkpoint/source-only.  It precedes the
    # one target loader below by construction.
    hc_gate = assert_immutable_receipt(H_C_FLOAT64_GATE, H_C_GATE_STATUS)
    old_hc_metrics = _require_hc_gate_binding(hc_gate)
    exposure_terminal = assert_immutable_receipt(D_S4E_D_Q4E_TERMINAL, EXPOSURE_TERMINAL_STATUS)
    ds4e_metrics = _require_exposure_terminal_binding(exposure_terminal)
    if hc_gate.get("target", {}).get("files") != exposure_terminal.get("target", {}).get("files"):
        raise NormalizedV2ContractError("sealed H-C and D-S4e receipts bind different target file bytes")
    if exposure_terminal.get("checkpoints", {}).get("d_s4e", {}).get("config_sha256") != sha256_file(D_S4E_CONFIG):
        raise NormalizedV2ContractError("local D-S4e source config bytes differ from its immutable terminal receipt")

    hc_cfg = _validate_carrierid_config(H_C_FULL_CONFIG, "full")
    hc_checkpoint, hc_meta = _load_carrierid_checkpoint(H_C_FULL_CHECKPOINT, H_C_FULL_CONFIG, "full")
    ordinary_source = _rebuild_hc_ordinary_source(hc_cfg)
    if hc_meta.get("source_manifest_sha256") != ordinary_source.pilot_manifest_sha256:
        raise NormalizedV2ContractError("ordinary H-C source manifest differs from sealed checkpoint")
    if hc_meta.get("normalizer_sha256") != ordinary_source.normalizer.normalizer_sha256:
        raise NormalizedV2ContractError("ordinary H-C source normalizer differs from sealed checkpoint")
    if float(hc_meta.get("s_src", float("nan"))) != float(ordinary_source.normalizer.s_src):
        raise NormalizedV2ContractError("ordinary H-C source scalar differs from sealed checkpoint")

    # Reconstruct only the locally sealed D-S4e source config whose digest is
    # bound by the immutable terminal receipt above.  The target receives none
    # of the corresponding paired-normalizer values.
    exposure_config = OmegaConf.load(D_S4E_CONFIG)
    exposure_source = _rebuild_exposure_window_source(exposure_config)
    if ordinary_source.plan.transform_sha256 != exposure_source.plan.transform_sha256:
        raise NormalizedV2ContractError("ordinary H-C and D-S4e exposure frozen plans differ")
    if ordinary_source.normalizer.normalizer_sha256 == exposure_source.normalizer.normalizer_sha256:
        raise NormalizedV2ContractError("diagnostic must not silently substitute the D-S4e paired normalizer")

    evaluation_device = torch.device(device)
    hc_model = _instantiate_hc_frozen(hc_cfg, hc_checkpoint, evaluation_device)
    del hc_checkpoint

    # This is intentionally the first and only target-data access in the
    # runtime.  Do not move this import or call above the source bindings.
    from src.data.h1_m4_eb_pilot import load_target_records, validate_target_receipt_binding

    target_records = load_target_records(hc_cfg.data.data_dir)
    validate_target_receipt_binding(
        target_records, ordinary_source.plan, hc_cfg.data.raw_receipt_path, hc_cfg.data.eb_receipt_path
    )
    target_files = {name: target_records[name].input_sha256 for name in H1_M4_FOLD0_TARGET}
    if target_files != exposure_terminal["target"]["files"]:
        raise NormalizedV2ContractError("opened target files differ from the immutable D-S4e receipt")
    # Construct both target views under one *ordinary* H-C normalizer.  Their
    # hashes are intentionally different encodings; exact list/tensor checks
    # below prove this is not a change in query pool, identity, or carrier.
    ordinary_full_dataset = H1M4EBNormalizedV2StrictTargetDataset(
        target_records, ordinary_source.plan, ordinary_source.normalizer, "full"
    )
    # ``exposure_source.plan`` establishes the D-S4e/D-Q4e target-dataset
    # hash convention, while the *ordinary* H-C normalizer creates the S4
    # carrier.  The exposure paired normalizer is never passed here.
    exposure_s4_dataset = H1CarrierIdDistributionStrictTargetDataset(
        target_records, exposure_source.plan, ordinary_source.normalizer, "s4"
    )
    same_window_proof = _assert_same_window_identity_and_carrier(ordinary_full_dataset, exposure_s4_dataset)
    hc_bab4_metrics = _evaluate(
        hc_model, exposure_s4_dataset, evaluation_device, "H-C/full/ordinary-normalizer on D-S4e hash convention"
    )
    _require_metric_contract(hc_bab4_metrics, expected_query_hash=EXPECTED_BAB4_QUERY_HASH, label="H-C full on bab4b904")

    receipt = {
        "schema": DIAGNOSTIC_SCHEMA,
        "status": DIAGNOSTIC_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "routing_prohibitions": {
            "EST4": "FORBIDDEN: this opened fold-0 diagnostic cannot route EST4",
            "CI64": "FORBIDDEN: this opened fold-0 diagnostic cannot route CI64",
            "paper_endpoint": "FORBIDDEN: not a paper endpoint or held-out claim",
        },
        "checkpoint_binding_completed_before_target_open": True,
        "h_c_float64_terminal_gate": {"path": str(H_C_FLOAT64_GATE), "sha256": sha256_file(H_C_FLOAT64_GATE)},
        "d_s4e_d_q4e_exposure_terminal": {"path": str(D_S4E_D_Q4E_TERMINAL), "sha256": sha256_file(D_S4E_D_Q4E_TERMINAL)},
        "d_s4e_source_config": {"path": str(D_S4E_CONFIG), "sha256": sha256_file(D_S4E_CONFIG)},
        "checkpoint": {
            "path": str(H_C_FULL_CHECKPOINT), "sha256": sha256_file(H_C_FULL_CHECKPOINT),
            "config_path": str(H_C_FULL_CONFIG), "config_sha256": sha256_file(H_C_FULL_CONFIG),
            "frozen_eval_only": True,
        },
        "source": {
            "ordinary_h_c": {
                "manifest_sha256": ordinary_source.pilot_manifest_sha256,
                "normalizer": ordinary_source.normalizer.manifest,
                "normalizer_sha256": ordinary_source.normalizer.normalizer_sha256,
                "s_src": float(ordinary_source.normalizer.s_src),
                "source_recordings_opened_before_target": len(ordinary_source.records),
                "ordinary_carrier_policy": "sealed H-C ordinary first-four-trial support carrier",
            },
            "exposure_window_reconstruction": {
                "manifest_sha256": exposure_source.pilot_manifest_sha256,
                "normalizer": exposure_source.normalizer.manifest,
                "source_recordings_opened_before_target": len(exposure_source.records),
                "window_policy": "D-S4e/D-Q4e strict-target dataset convention only; its paired normalizer is forbidden",
            },
            "frozen_plan_transform_sha256": ordinary_source.plan.transform_sha256,
            "ordinary_normalizer_differs_from_exposure_normalizer": True,
        },
        "target": {
            "sessions": list(H1_M4_FOLD0_TARGET), "files": target_files,
            "strict_query_window_indices_sha256": exposure_s4_dataset.window_indices_sha256,
            "ordinary_h_c_binary_window_indices_sha256": ordinary_full_dataset.window_indices_sha256,
            "window_pool": "exact same ordered strict post-support list; only binary-vs-canonical-JSON hash encoding differs",
            "carrier": "ordinary H-C first-four-trial support carrier normalized by ordinary H-C source scalar",
            "not_used": "D-S4e/D-Q4e 6.8395867936067665e-06 paired normalizer",
            **same_window_proof,
        },
        "metrics": {
            "h_c_full_on_bab4b904": hc_bab4_metrics,
            "sealed_h_c_full_on_665fe535": old_hc_metrics,
            "d_s4e_on_bab4b904": ds4e_metrics,
            "h_c_bab4b904_minus_h_c_665fe535": _metric_delta(hc_bab4_metrics, old_hc_metrics),
            "h_c_bab4b904_minus_d_s4e_bab4b904": _metric_delta(hc_bab4_metrics, ds4e_metrics),
        },
        "target_updates": {"optimizer_steps": 0, "backward_steps": 0, "model_state_unchanged": True},
        "data_scope": {
            "opened": "11 ordinary H-C source plus 11 D-S4e exposure-source reconstructions then exactly 2 public fold-0 target NWBs",
            "formal_heldout_opened": False, "minival_opened": False, "evalai_opened": False,
        },
    }
    path, digest = write_immutable_json(OUTPUT, receipt)
    return {"status": DIAGNOSTIC_STATUS, "receipt_path": str(path), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-opened-fold0-development-diagnostic", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if not args.execute_opened_fold0_development_diagnostic:
        parser.error(
            "refusing target access: pass --execute-opened-fold0-development-diagnostic only after root approval"
        )
    print(json.dumps(run_opened_fold0_development_diagnostic(device=args.device), sort_keys=True))


if __name__ == "__main__":
    main()
