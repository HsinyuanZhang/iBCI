#!/usr/bin/env python3
"""CPU-only source-closure preflight for separately-trained H1 CarrierID RS/LS."""
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

from src.data.h1_carrierid_shuffle import H1CarrierIdShuffleDataModule
from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError, array_sha256, sha256_file, state_hash, write_immutable_json,
)
from src.models.h1_carrierid_shuffle_module import H1CarrierIdShuffleLitModule
from src.models.components.h1_carrierid_spint import H1CarrierIdSpint

PREFLIGHT_SCHEMA = "h1_carrierid_h32_rs_ls_source_cpu_preflight_v1"
PREFLIGHT_STATUS = "PASS_H1_CARRIERID_H32_RS_LS_REAL_SOURCE_CPU_PREFLIGHT_NONLAUNCH"
RAW = ROOT.parent / "sua_exploration/results/h1_m4_population_decoder_carrier_date_lodo_v1/H1_M4_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json"
EB = ROOT.parent / "sua_exploration/results/h1_m4_empirical_bayes_confidence_carrier_date_lodo_v1/H1_M4_EMPIRICAL_BAYES_CONFIDENCE_CARRIER_CPU_RECEIPT.json"
HC_GATE = ROOT / "pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/H1_CARRIERID_H32_FOLD0_TERMINAL_GATE.json"
HC_INITIAL_STATE = "8208b6ebec793e424e1b95cd1cbefdd9c2d3dceea9ee9d01c598a007446e90eb"


def _seed() -> None:
    random.seed(42); np.random.seed(42); torch.manual_seed(42)


def _new(arm: str) -> H1CarrierIdShuffleLitModule:
    return H1CarrierIdShuffleLitModule(
        task="h1", pilot_arm=arm, fold_date="19250101", decode_last_timestep_only=True,
        predict_scaled_behavior=True, behavior_scaling_factor=20.0, optimizer=torch.optim.Adam,
        scheduler=None, compile=False, clean_teacher=True, scheduler_monitor="val_heldin/r2_mean",
        net=H1CarrierIdSpint(carrier_hidden_dim=32, carrier_dim=4, carrier_trial_length=1024,
            zero_carrier=False, model_dim=1024, num_covariates=7, window_size=700, num_heads=64,
            num_layers=1, num_id_layers=3, use_learnable_id=True, learnable_id_type="mlp",
            learnable_rep=True, dropout_rate=0.0, dynamic_dropout=True, dynamic_dropout_low=0.0,
            dynamic_dropout_high=1.0, tf_drop_rate=0.1, readin_layer_type="mlp"),
    )


def _first(dm: H1CarrierIdShuffleDataModule):
    dm.train_batch_sampler.reset_epoch()
    return next(iter(dm.train_dataloader()))


def run(*, cache_dir: Path, output: Path) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise NormalizedV2ContractError("H1 RS/LS preflight requires CUDA_VISIBLE_DEVICES unset")
    if not (ROOT / "data/000954/sub-HumanPitt-held-in-calib").is_dir():
        raise FileNotFoundError("public H1 held-in-calib source directory missing")
    data: dict[str, H1CarrierIdShuffleDataModule] = {}
    batches: dict[str, Any] = {}
    for arm in ("rs", "ls"):
        dm = H1CarrierIdShuffleDataModule(task="h1", data_dir=str(ROOT / "data/000954"),
            raw_receipt_path=str(RAW), eb_receipt_path=str(EB), cache_dir=str(cache_dir / arm),
            carrier_intervention=arm)
        dm.setup("fit"); data[arm] = dm; batches[arm] = _first(dm)
    rs, ls = data["rs"], data["ls"]
    for field in ("source_window_indices_sha256", "batch_order_sha256", "calibration_schedule_sha256",
                  "carrier_cache_sha256", "normalized_cache_sha256", "normalizer_sha256"):
        if rs.pilot_manifest()[field] != ls.pilot_manifest()[field]:
            raise NormalizedV2ContractError(f"RS/LS source closure mismatch at {field}")
    for index in (0, 1, 2):
        if not torch.equal(batches["rs"][index], batches["ls"][index]):
            raise NormalizedV2ContractError(f"RS/LS source batch mismatch at field {index}")
    if list(batches["rs"][3]) != list(batches["ls"][3]) or torch.equal(batches["rs"][4], batches["ls"][4]):
        raise NormalizedV2ContractError("RS/LS control intervention/source ordering audit failed")
    _seed(); model_rs = _new("rs")
    _seed(); model_ls = _new("ls")
    if state_hash(model_rs.state_dict()) != state_hash(model_ls.state_dict()):
        raise NormalizedV2ContractError("RS/LS initial model state differs")
    if not HC_GATE.is_file():
        raise FileNotFoundError("existing H-C terminal gate required for initial-state binding")
    existing_gate = json.loads(HC_GATE.read_text(encoding="utf-8"))
    full_state = existing_gate["checkpoints"]["h_c_full"]["metadata"]["initial_state_sha256"]
    if full_state != HC_INITIAL_STATE or state_hash(model_rs.state_dict()) != full_state:
        raise NormalizedV2ContractError("RS/LS initial state does not equal sealed H-C initial state")
    model_rs.eval(); model_ls.eval()
    with torch.no_grad():
        out_rs = model_rs(batches["rs"][0], batches["rs"][2], batches["rs"][4])
        out_ls = model_ls(batches["ls"][0], batches["ls"][2], batches["ls"][4])
    # Their carrier inputs differ but initial carrier columns are literal zero,
    # so the same initialization must be functionally identical.
    if not torch.equal(out_rs, out_ls):
        raise NormalizedV2ContractError("RS/LS initial forward is not bit-identical")
    files = [
        "src/data/h1_carrierid_shuffle.py", "src/models/h1_carrierid_shuffle_module.py",
        "scripts/h1_carrierid_shuffle_preflight.py", "scripts/h1_carrierid_shuffle_launcher.py",
        "configs/data/falcon_h1_carrierid_shuffle.yaml", "configs/model/falcon_h1_carrierid_shuffle.yaml",
        "configs/experiment/h1_carrierid_shuffle.yaml", "configs/experiment/h1_carrierid_rs.yaml",
        "configs/experiment/h1_carrierid_ls.yaml", "configs/callbacks/h1_carrierid_terminal.yaml",
        "tests/test_h1_carrierid_shuffle_contract.py",
        "src/data/h1_m4_eb_normalized_v2.py", "src/data/h1_m4_eb_pilot.py",
        "src/data/h1_m4_eb_fold0_datamodule.py", "src/models/h1_carrierid_module.py",
        "src/models/components/h1_carrierid_spint.py", "src/h1_m4_eb_normalized_v2_contract.py",
    ]
    source_sha = {item: sha256_file(ROOT / item) for item in files}
    receipt = {
        "schema": PREFLIGHT_SCHEMA, "status": PREFLIGHT_STATUS,
        "scope": {"opened": "exactly 11 public held-in-calib source NWBs", "target_recordings_opened": 0,
            "target_recordings_enumerated": 0, "minival_opened_or_enumerated": False,
            "heldout_opened_or_enumerated": False, "formal_heldout_opened_or_enumerated": False,
            "evalai_opened_or_enumerated": False, "cuda_constructed_or_launched": False,
            "checkpoint_created": False},
        "fixed_protocol": {"fold_date": "19250101", "arms": ["H-RS", "H-LS"], "seed": 42,
            "support_trials": 4, "epochs": 50, "precision": "32-true", "optimizer": "Adam",
            "learning_rate": 5.0e-5, "weight_decay": 0.0, "fixed_terminal_epoch": 49},
        "shared_source_binding": {key: rs.pilot_manifest()[key] for key in (
            "source_window_indices_sha256", "batch_order_sha256", "calibration_schedule_sha256",
            "carrier_cache_sha256", "normalized_cache_sha256", "normalizer_sha256", "transform_sha256")},
        "intervention": {"rs": "complete deterministic source carrier row shuffle", "ls": "source velocity-label rotation then frozen carrier refit",
            "same_normalizer_scalar": True, "source_carrier_inputs_differ": True},
        "effective_source_carriers": {arm: {
            "sha256": data[arm].pilot_manifest()["effective_source_carriers_sha256"],
            "shape": data[arm].pilot_manifest()["effective_source_carriers_shape"],
            "count": data[arm].pilot_manifest()["effective_source_carriers_count"],
            "nonidentity_all": data[arm].pilot_manifest()["effective_source_carriers_nonidentity_all"],
        } for arm in ("rs", "ls")},
        "first_source_batch": {"neural_shape": list(batches["rs"][0].shape), "identity_shape": list(batches["rs"][2].shape),
            "carrier_shape": list(batches["rs"][4].shape), "rs_carrier_sha256": array_sha256(batches["rs"][4]),
            "ls_carrier_sha256": array_sha256(batches["ls"][4]), "session_order": list(batches["rs"][3])},
        "initialization": {"complete_state_sha256": state_hash(model_rs.state_dict()), "rs_ls_state_equal": True,
            "sealed_h_c_initial_state_sha256": full_state, "sealed_h_c_terminal_gate_path": str(HC_GATE),
            "sealed_h_c_terminal_gate_sha256": hashlib.sha256(HC_GATE.read_bytes()).hexdigest(),
            "rs_ls_equal_sealed_h_c_initial_state": True,
            "initial_forward_bit_identical": True, "prediction_sha256": array_sha256(out_rs)},
        "source_sha256": source_sha,
        "launch": {"launch_authorized": False, "required_before_execute": "immutable receipt plus exact source-closure recheck"},
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
