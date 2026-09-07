#!/usr/bin/env python3
"""Source-only CPU preflight for exposure-matched H1 D-S4e/D-Q4e.

This is a non-launcher.  It opens exactly the public eleven source NWBs to
construct the two paired source DataModules and one CPU forward per arm.  It
never constructs a Trainer or CUDA context and never imports/enumerates target,
minival, formal-heldout, or EvalAI data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.h1_carrierid_distribution_exposure import H1CarrierIdDistributionExposureDataModule
from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    array_sha256,
    canonical_sha256,
    sha256_file,
    state_hash,
    write_immutable_json,
)
from src.models.components.h1_carrierid_spint import H1CarrierIdSpint
from src.models.h1_carrierid_distribution_exposure_module import H1CarrierIdDistributionExposureLitModule


PREFLIGHT_SCHEMA = "h1_carrierid_h32_fresh_distribution_exposure_source_cpu_preflight_v1"
PREFLIGHT_STATUS = "PASS_H1_CARRIERID_H32_FRESH_EXPOSURE_MATCHED_D_S4E_D_Q4E_SOURCE_CPU_PREFLIGHT_NONLAUNCH"
RAW = ROOT.parent / "sua_exploration/results/h1_m4_population_decoder_carrier_date_lodo_v1/H1_M4_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json"
EB = ROOT.parent / "sua_exploration/results/h1_m4_empirical_bayes_confidence_carrier_date_lodo_v1/H1_M4_EMPIRICAL_BAYES_CONFIDENCE_CARRIER_CPU_RECEIPT.json"
H_C_GATE = ROOT / "pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/H1_CARRIERID_H32_FOLD0_TERMINAL_GATE.json"
H_C_INITIAL_STATE = "8208b6ebec793e424e1b95cd1cbefdd9c2d3dceea9ee9d01c598a007446e90eb"
SEALED_H_C_POOLED_R2 = 0.5255107931
VALIDITY_MIN_D_S4E_R2 = 0.4955107931
DELTA_GATE = 0.03


def _seed() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def _new(arm: str) -> H1CarrierIdDistributionExposureLitModule:
    return H1CarrierIdDistributionExposureLitModule(
        task="h1", pilot_arm=arm, fold_date="19250101", decode_last_timestep_only=True,
        predict_scaled_behavior=True, behavior_scaling_factor=20.0, optimizer=torch.optim.Adam,
        scheduler=None, compile=False, clean_teacher=True, scheduler_monitor="val_heldin/r2_mean",
        net=H1CarrierIdSpint(
            carrier_hidden_dim=32, carrier_dim=4, carrier_trial_length=1024, zero_carrier=False,
            model_dim=1024, num_covariates=7, window_size=700, num_heads=64, num_layers=1,
            num_id_layers=3, use_learnable_id=True, learnable_id_type="mlp", learnable_rep=True,
            dropout_rate=0.0, dynamic_dropout=True, dynamic_dropout_low=0.0, dynamic_dropout_high=1.0,
            tf_drop_rate=0.1, readin_layer_type="mlp",
        ),
    )


def _first(datamodule: H1CarrierIdDistributionExposureDataModule):
    datamodule.train_batch_sampler.reset_epoch()
    return next(iter(datamodule.train_dataloader()))


def _tensor_sha(value: torch.Tensor) -> str:
    return array_sha256(value.detach().cpu().contiguous().numpy())


def _carrier_distance(s4: np.ndarray, q4: np.ndarray) -> dict[str, Any]:
    left, right = np.asarray(s4, np.float64), np.asarray(q4, np.float64)
    flattened = np.sum(left * right, axis=(1, 2)) / (
        np.linalg.norm(left.reshape(left.shape[0], -1), axis=1)
        * np.linalg.norm(right.reshape(right.shape[0], -1), axis=1)
    )
    relative = np.linalg.norm((q4 - s4).reshape(left.shape[0], -1), axis=1) / np.maximum(
        np.linalg.norm(left.reshape(left.shape[0], -1), axis=1), 1.0e-12
    )
    if not np.isfinite(flattened).all() or not np.isfinite(relative).all():
        raise NormalizedV2ContractError("exposure carrier-distance audit is nonfinite")
    return {
        "entry_count": int(left.shape[0]),
        "flattened_cosine_mean": float(flattened.mean()),
        "flattened_cosine_median": float(np.median(flattened)),
        "relative_frobenius_change_mean": float(relative.mean()),
        "relative_frobenius_change_median": float(np.median(relative)),
        "interpretation_limit": "Different carrier tensors do not establish an accuracy effect.",
    }


def frozen_terminal_branch_gates() -> dict[str, Any]:
    """Predeclare the validity and interpretation rule before target access."""

    return {
        "sealed_h_c_pooled_r2": SEALED_H_C_POOLED_R2,
        "validity": {
            "all_required": True,
            "d_s4e_pooled_r2_min": VALIDITY_MIN_D_S4E_R2,
            "formula": "0.5255107931 - 0.03",
            "per_target_recording_requirement": "both D-S4e R2 values must be strictly > 0",
            "if_fail": "invalid_exposure_repair__q4e_minus_s4e_not_interpretable",
        },
        "only_after_validity_pass": {
            "estimator": {
                "all_required": True,
                "pooled_q4e_minus_s4e_min": DELTA_GATE,
                "per_target_recording_requirement": "both Q4e-S4e deltas must be strictly > 0",
                "decision": "estimator-limited evidence",
            },
            "consumer": {
                "all_required": True,
                "absolute_pooled_q4e_minus_s4e_strictly_below": DELTA_GATE,
                "no_session_sign_reversal_definition": "delta_session_0 * delta_session_1 >= 0",
                "decision": "consumer-limited evidence",
            },
            "otherwise": "unresolved",
        },
        "decision_order": ["validity", "estimator", "consumer", "unresolved"],
        "scope_limit": "D-Q4e is a query-local leakage diagnostic, never a deployable target-time arm or paper-main selection endpoint.",
    }


def run(*, cache_dir: Path, output: Path) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise NormalizedV2ContractError("exposure CPU preflight requires CUDA_VISIBLE_DEVICES unset")
    if torch.cuda.is_initialized():
        raise NormalizedV2ContractError("exposure CPU preflight refuses an initialized CUDA context")
    if not (ROOT / "data/000954/sub-HumanPitt-held-in-calib").is_dir():
        raise FileNotFoundError("public H1 held-in-calib source directory missing")
    data: dict[str, H1CarrierIdDistributionExposureDataModule] = {}
    batches: dict[str, Any] = {}
    for arm in ("s4", "q4"):
        datamodule = H1CarrierIdDistributionExposureDataModule(
            task="h1", data_dir=str(ROOT / "data/000954"), raw_receipt_path=str(RAW),
            eb_receipt_path=str(EB), cache_dir=str(cache_dir / "shared_source_cache"),
            carrier_distribution_arm=arm,
        )
        datamodule.setup("fit")
        data[arm] = datamodule
        batches[arm] = _first(datamodule)
    s4, q4 = data["s4"], data["q4"]
    shared = (
        "source_schedule_sha256", "common_query_samples_sha256", "identity_schedule_sha256",
        "batch_order_sha256", "carrier_cache_sha256", "normalized_cache_sha256", "normalizer_sha256",
        "transform_sha256", "source_sessions", "files", "schedule_count", "sample_count",
        "batches_per_epoch", "scheduled_samples_per_epoch", "exposure_sample_plan",
    )
    for field in shared:
        if s4.pilot_manifest()[field] != q4.pilot_manifest()[field]:
            raise NormalizedV2ContractError(f"D-S4e/D-Q4e shared source contract mismatch at {field}")
    manifest = s4.pilot_manifest()
    if manifest["schedule_count"] != 72 or manifest["sample_count"] != 115_520:
        raise NormalizedV2ContractError("exposure plan did not retain 72 schedules / 115520 samples")
    if manifest["batches_per_epoch"] != 3_610 or manifest["scheduled_samples_per_epoch"] != 115_520:
        raise NormalizedV2ContractError("exposure plan did not match H-C batch/sample throughput")
    exposure = manifest["exposure_sample_plan"]
    if exposure["selected_windows_per_schedule_min"] <= 32:
        raise NormalizedV2ContractError("exposure plan regressed to the fixed-32 window diagnostic")
    # Fields 0-3 (neural, target, identity, session) are exact pair matches;
    # carrier is the sole actual arm difference.
    for index in (0, 1, 2):
        if not torch.equal(batches["s4"][index], batches["q4"][index]):
            raise NormalizedV2ContractError(f"D-S4e/D-Q4e first batch mismatch at tensor {index}")
    if list(batches["s4"][3]) != list(batches["q4"][3]):
        raise NormalizedV2ContractError("D-S4e/D-Q4e first-batch session order mismatch")
    if torch.equal(batches["s4"][4], batches["q4"][4]):
        raise NormalizedV2ContractError("D-S4e/D-Q4e carrier is not the unique differing tensor")
    first_indices = s4.train_batch_sampler.batches[0]
    s4_meta = [s4.train_dataset.sample_metadata(index) for index in first_indices]
    q4_meta = [q4.train_dataset.sample_metadata(index) for index in first_indices]
    for left, right in zip(s4_meta, q4_meta):
        if {key: value for key, value in left.items() if key != "arm"} != {key: value for key, value in right.items() if key != "arm"}:
            raise NormalizedV2ContractError("D-S4e/D-Q4e first-batch schedule/window differs")
    _seed(); model_s4 = _new("s4")
    _seed(); model_q4 = _new("q4")
    common_state = state_hash(model_s4.state_dict())
    if common_state != state_hash(model_q4.state_dict()):
        raise NormalizedV2ContractError("D-S4e/D-Q4e initialization differs")
    if not H_C_GATE.is_file():
        raise FileNotFoundError("sealed H-C terminal gate required for initial-state binding")
    sealed_gate = json.loads(H_C_GATE.read_text(encoding="utf-8"))
    sealed_state = sealed_gate["checkpoints"]["h_c_full"]["metadata"]["initial_state_sha256"]
    if sealed_state != H_C_INITIAL_STATE or common_state != sealed_state:
        raise NormalizedV2ContractError("exposure D-S4e/D-Q4e initial state differs from sealed H-C")
    model_s4.eval(); model_q4.eval()
    with torch.no_grad():
        prediction_s4 = model_s4(batches["s4"][0], batches["s4"][2], batches["s4"][4])
        prediction_q4 = model_q4(batches["q4"][0], batches["q4"][2], batches["q4"][4])
    if not torch.equal(prediction_s4, prediction_q4):
        raise NormalizedV2ContractError("literal-zero carrier columns do not give initial forward parity")
    if torch.cuda.is_initialized():
        raise NormalizedV2ContractError("exposure CPU preflight unexpectedly initialized CUDA")
    source_files = [
        "src/data/h1_carrierid_distribution_exposure.py",
        "src/models/h1_carrierid_distribution_exposure_module.py",
        "scripts/h1_carrierid_distribution_exposure_preflight.py",
        "configs/data/falcon_h1_carrierid_distribution_exposure.yaml",
        "configs/model/falcon_h1_carrierid_distribution_exposure.yaml",
        "configs/experiment/h1_carrierid_distribution_exposure.yaml",
        "configs/experiment/h1_carrierid_distribution_exposure_s4.yaml",
        "configs/experiment/h1_carrierid_distribution_exposure_q4.yaml",
        "src/data/h1_carrierid_distribution.py",
        "src/models/h1_carrierid_distribution_module.py",
        "src/data/h1_m4_eb_pilot.py",
        "src/models/h1_carrierid_module.py",
        "src/models/components/h1_carrierid_spint.py",
        "src/h1_m4_eb_normalized_v2_contract.py",
    ]
    source_sha = {name: sha256_file(ROOT / name) for name in source_files}
    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "scope": {
            "opened": "exactly 11 public held-in-calib fold-0 source NWBs",
            "source_recordings_opened": 11,
            "target_recordings_opened": 0,
            "target_recordings_enumerated": 0,
            "minival_opened_or_enumerated": False,
            "heldout_opened_or_enumerated": False,
            "formal_heldout_opened_or_enumerated": False,
            "evalai_opened_or_enumerated": False,
            "cuda_constructed_or_launched": False,
            "trainer_constructed": False,
            "checkpoint_created": False,
        },
        "fixed_protocol": {
            "arms": ["D-S4e standard-support", "D-Q4e query-local leakage diagnostic"],
            "architecture": "H1CarrierID h=32,d=4", "seed": 42, "epochs": 50,
            "optimizer": "Adam", "learning_rate": 5.0e-5, "precision": "32-true",
            "fixed_terminal_epoch": 49, "support_trials": 4, "common_contiguous_block_trials": 8,
            "schedule_count": 72, "samples_per_epoch": 115_520, "batches_per_epoch": 3_610,
            "s4_carrier": "t..t+3", "q4_carrier": "t+4..t+7",
            "q4_scope": "source analogue of deliberate query-label leakage; LEAKAGE_DIAGNOSTIC_ONLY",
        },
        "shared_source_binding": {field: manifest[field] for field in shared},
        "normalizer": manifest["normalizer"],
        "source_carrier_distance": _carrier_distance(s4.train_dataset.s4_normalized, s4.train_dataset.q4_normalized),
        "first_source_batch": {
            "neural_shape": list(batches["s4"][0].shape),
            "target_shape": list(batches["s4"][1].shape),
            "identity_shape": list(batches["s4"][2].shape),
            "carrier_shape": list(batches["s4"][4].shape),
            "neural_sha256": _tensor_sha(batches["s4"][0]),
            "target_sha256": _tensor_sha(batches["s4"][1]),
            "identity_sha256": _tensor_sha(batches["s4"][2]),
            "session_order": list(batches["s4"][3]),
            "window_schedule_sha256": canonical_sha256(s4_meta),
            "s4_carrier_sha256": _tensor_sha(batches["s4"][4]),
            "q4_carrier_sha256": _tensor_sha(batches["q4"][4]),
            "sole_arm_input_difference": "carrier",
        },
        "initialization": {
            "complete_state_sha256": common_state,
            "s4e_q4e_state_equal": True,
            "sealed_h_c_initial_state_sha256": sealed_state,
            "sealed_h_c_terminal_gate_path": str(H_C_GATE),
            "sealed_h_c_terminal_gate_sha256": hashlib.sha256(H_C_GATE.read_bytes()).hexdigest(),
            "s4e_q4e_equal_sealed_h_c_initial_state": True,
            "initial_forward_bit_identical": True,
            "prediction_sha256": _tensor_sha(prediction_s4),
        },
        "frozen_validity_and_branch_gates": frozen_terminal_branch_gates(),
        "source_cache_dir": str((cache_dir / "shared_source_cache").resolve()),
        "source_sha256": source_sha,
        "launch": {
            "launch_authorized": False,
            "future_target_comparison_label": "LEAKAGE_DIAGNOSTIC_ONLY_NOT_FOR_SELECTION_OR_PAPER_MAIN_RESULT",
        },
    }
    path, digest = write_immutable_json(output, receipt)
    return {"status": PREFLIGHT_STATUS, "receipt_path": str(path), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(cache_dir=args.cache_dir, output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
