#!/usr/bin/env python3
"""Source-only terminal checker for exposure-matched H1 D-S4e/D-Q4e.

No target loader is imported in this module.  Both real 50-epoch checkpoints
must prove the H-C-matched 115,520-sample / 3,610-batch source schedule before
a separate evaluator is permitted to open either target recording.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.h1_carrierid_distribution_exposure import H1CarrierIdDistributionExposureDataModule
from src.h1_m4_eb_normalized_v2_contract import (
    NORMALIZER_FORMULA,
    NormalizedV2ContractError,
    assert_immutable_receipt,
    canonical_sha256,
    sha256_file,
    write_immutable_json,
)
from src.models.components.h1_carrierid_spint import H1_CARRIERID_PARAMETERS, H1_CARRIERID_WHOLE_MODEL_PARAMETERS
from scripts.h1_carrierid_distribution_exposure_preflight import PREFLIGHT_SCHEMA as SOURCE_PREFLIGHT_SCHEMA
from scripts.h1_carrierid_distribution_exposure_preflight import PREFLIGHT_STATUS as SOURCE_PREFLIGHT_STATUS


CHECKER_SCHEMA = "h1_carrierid_h32_fresh_distribution_exposure_terminal_checkpoint_pair_v1"
CHECKER_STATUS = "PASS_H1_CARRIERID_H32_FRESH_D_S4E_D_Q4E_SOURCE_TERMINAL_CHECKPOINT_PAIR_NONLAUNCH"
TERMINAL_PREFLIGHT_SCHEMA = "h1_carrierid_h32_fresh_distribution_exposure_terminal_evaluator_preflight_v1"
TERMINAL_PREFLIGHT_STATUS = "PASS_H1_CARRIERID_H32_FRESH_D_S4E_D_Q4E_TERMINAL_EVALUATOR_SOURCE_CLOSURE_NONLAUNCH"
CHECKPOINT_SCHEMA = "h1_carrierid_h32_fresh_distribution_exposure_terminal_checkpoint_v1"
EXPECTED_GLOBAL_STEP = 180_500
ARM_METADATA = {
    "s4": {"arm": "D-S4E", "carrier_mode": "standard_support_t_to_t_plus_3"},
    "q4": {"arm": "D-Q4E", "carrier_mode": "source_query_local_t_plus_4_to_t_plus_7_LEAKAGE_DIAGNOSTIC_ONLY"},
}


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise NormalizedV2ContractError(f"{label} must be a SHA-256")
    return value


def _verify_source_closure(expected: Mapping[str, Any]) -> dict[str, str]:
    if not expected:
        raise NormalizedV2ContractError("exposure terminal preflight has no source closure")
    output: dict[str, str] = {}
    for relative, digest in expected.items():
        path = (ROOT / str(relative)).resolve()
        if not path.is_file() or path.parent == ROOT.parent:
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != digest:
            raise NormalizedV2ContractError(f"exposure terminal source closure drift at {relative}")
        output[str(relative)] = actual
    return output


def _load_config(path: Path, arm: str) -> Any:
    cfg = OmegaConf.load(path)
    fixed = {
        "protocol_id": "h1_carrierid_h32_fresh_distribution_exposure_source_only_v1",
        "seed": 42, "train": True, "test": False, "ckpt_path": None,
    }
    if any(cfg.get(key) != value for key, value in fixed.items()):
        raise NormalizedV2ContractError(f"D-{arm.upper()}E config base contract drift")
    checks = (
        str(cfg.pilot.fold_date) == "19250101", str(cfg.pilot.arm) == arm,
        bool(cfg.pilot.leakage_diagnostic_only) is True, int(cfg.pilot.schedule_count) == 72,
        int(cfg.pilot.samples_per_epoch) == 115_520, int(cfg.pilot.batches_per_epoch) == 3_610,
        int(cfg.pilot.fixed_terminal_epochs) == 50,
        str(cfg.data._target_) == "src.data.h1_carrierid_distribution_exposure.H1CarrierIdDistributionExposureDataModule",
        str(cfg.model._target_) == "src.models.h1_carrierid_distribution_exposure_module.H1CarrierIdDistributionExposureLitModule",
        str(cfg.data.carrier_distribution_arm) == arm, int(cfg.data.batch_size) == 32,
        int(cfg.data.window_size) == 700, int(cfg.data.calibration_n_trials) == 4,
        int(cfg.data.max_trial_length) == 1024, int(cfg.data.fixed_epochs) == 50,
        int(cfg.data.samples_per_epoch) == 115_520, int(cfg.trainer.max_epochs) == 50,
        int(cfg.trainer.min_epochs) == 50, int(cfg.trainer.limit_val_batches) == 0,
        int(cfg.trainer.num_sanity_val_steps) == 0, str(cfg.trainer.precision) == "32-true",
        float(cfg.model.optimizer.lr) == 5.0e-5, float(cfg.model.optimizer.weight_decay) == 0.0,
        bool(cfg.model.net.zero_carrier) is False, int(cfg.model.net.carrier_hidden_dim) == 32,
        int(cfg.model.net.carrier_dim) == 4, int(cfg.model.net.carrier_trial_length) == 1024,
    )
    if not all(checks):
        raise NormalizedV2ContractError(f"D-{arm.upper()}E resolved config violates frozen exposure contract")
    callback = cfg.callbacks.get("fixed_epoch50")
    if callback is None or callback.get("monitor") is not None or int(callback.get("every_n_epochs", -1)) != 50:
        raise NormalizedV2ContractError("D-S4e/D-Q4e need unselected fixed epoch-49 callback")
    return cfg


def _require_finite_state_dict(state_dict: Mapping[str, Any], arm: str) -> None:
    # A finite metadata receipt is not sufficient: reject a checkpoint that
    # contains a NaN/Inf parameter or buffer before it can reach target data.
    # State dictionaries for this model are entirely tensors; treating a
    # non-tensor item as a contract failure keeps this check fail-closed.
    for tensor_name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise NormalizedV2ContractError(f"D-{arm.upper()} state_dict has non-tensor {tensor_name!r}")
        if not bool(torch.isfinite(tensor).all().item()):
            raise NormalizedV2ContractError(f"D-{arm.upper()} state_dict has nonfinite tensor {tensor_name!r}")


def _load_checkpoint(path: Path, config_path: Path, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise NormalizedV2ContractError("exposure terminal input is not a Lightning checkpoint")
    _require_finite_state_dict(checkpoint["state_dict"], arm)
    if int(checkpoint.get("epoch", -1)) != 49 or int(checkpoint.get("global_step", -1)) != EXPECTED_GLOBAL_STEP:
        raise NormalizedV2ContractError("D-S4e/D-Q4e require epoch49/global_step180500 exactly")
    meta = checkpoint.get("h1_carrierid")
    if not isinstance(meta, dict):
        raise NormalizedV2ContractError("exposure terminal checkpoint lacks h1_carrierid metadata")
    required = {
        "schema": CHECKPOINT_SCHEMA, "fold_date": "19250101", "checkpoint_epoch_zero_based": 49,
        "epochs_completed": 50, "selected_by": "fixed_terminal_epoch_no_selection",
        "config_sha256": sha256_file(config_path), "normalizer_formula": NORMALIZER_FORMULA,
        "carrier_hidden_dim": 32, "carrier_dim": 4, "carrier_trial_length": 1024,
        "carrierid_parameters": H1_CARRIERID_PARAMETERS, "whole_model_parameters": H1_CARRIERID_WHOLE_MODEL_PARAMETERS,
        "exposure_matched_to_h_c": True, "leakage_diagnostic_only": True,
        "not_for_selection_or_paper_main_result": True, "deployment_target_optimizer_steps": 0,
        "deployment_target_backward_steps": 0, **ARM_METADATA[arm],
    }
    for key, value in required.items():
        if meta.get(key) != value:
            raise NormalizedV2ContractError(f"D-{arm.upper()}E metadata drift at {key}")
    for key in (
        "source_manifest_sha256", "normalizer_sha256", "source_cache_sha256", "normalized_cache_sha256",
        "source_hashes_sha256", "initial_state_sha256", "source_schedule_sha256",
        "common_query_samples_sha256", "identity_schedule_sha256", "batch_order_sha256",
        "s4_effective_carriers_sha256", "q4_effective_carriers_sha256",
    ):
        _require_sha(meta.get(key), f"D-{arm.upper()}E metadata.{key}")
    plan = meta.get("exposure_sample_plan")
    if not isinstance(plan, Mapping) or plan.get("target_samples_per_epoch") != 115_520 or plan.get("target_batches_per_epoch") != 3_610:
        raise NormalizedV2ContractError("D-S4e/D-Q4e checkpoint lost matched exposure plan")
    if meta.get("q4_label_scope") != "t+4..t+7 source query-local leakage diagnostic only":
        raise NormalizedV2ContractError("exposure checkpoint lost Q4 leakage disclosure")
    return checkpoint, meta


def _validate_pair(s4: Mapping[str, Any], q4: Mapping[str, Any]) -> dict[str, Any]:
    shared = (
        "fold_date", "normalizer_sha256", "source_cache_sha256", "normalized_cache_sha256",
        "source_hashes_sha256", "initial_state_sha256", "source_schedule_sha256",
        "common_query_samples_sha256", "identity_schedule_sha256", "batch_order_sha256",
        "s4_effective_carriers_sha256", "q4_effective_carriers_sha256", "exposure_sample_plan",
    )
    for key in shared:
        if s4.get(key) != q4.get(key):
            raise NormalizedV2ContractError(f"D-S4e/D-Q4e terminal pair mismatch at {key}")
    if s4.get("source_manifest_sha256") == q4.get("source_manifest_sha256"):
        raise NormalizedV2ContractError("D-S4e/D-Q4e arm-specific source manifests unexpectedly identical")
    return {key: s4[key] for key in shared}


def _source_hashes(manifest: Mapping[str, Any]) -> str:
    return canonical_sha256({
        "source_sessions": manifest.get("source_sessions"), "files": manifest.get("files"),
        "carrier_cache_sha256": manifest.get("carrier_cache_sha256"),
        "normalized_cache_sha256": manifest.get("normalized_cache_sha256"),
    })


def _rebuild_source(cfg: Any, metadata: Mapping[str, Any], arm: str) -> dict[str, Any]:
    data = cfg.data
    source = H1CarrierIdDistributionExposureDataModule(
        task=str(data.task), data_dir=str(data.data_dir), raw_receipt_path=str(data.raw_receipt_path),
        eb_receipt_path=str(data.eb_receipt_path), cache_dir=str(data.cache_dir),
        carrier_distribution_arm=str(data.carrier_distribution_arm), batch_size=int(data.batch_size),
        window_size=int(data.window_size), calibration_n_trials=int(data.calibration_n_trials),
        max_trial_length=int(data.max_trial_length), num_workers=int(data.num_workers),
        pin_memory=bool(data.pin_memory), seed=int(data.seed), fixed_epochs=int(data.fixed_epochs),
        samples_per_epoch=int(data.samples_per_epoch),
    )
    source.setup("fit")
    manifest = source.pilot_manifest()
    expected = {
        "source_manifest_sha256": source.pilot_manifest_sha256,
        "normalizer_sha256": source.normalizer.normalizer_sha256,
        "source_cache_sha256": manifest["carrier_cache_sha256"],
        "normalized_cache_sha256": manifest["normalized_cache_sha256"],
        "source_hashes_sha256": _source_hashes(manifest),
        "source_schedule_sha256": manifest["source_schedule_sha256"],
        "common_query_samples_sha256": manifest["common_query_samples_sha256"],
        "identity_schedule_sha256": manifest["identity_schedule_sha256"],
        "batch_order_sha256": manifest["batch_order_sha256"],
        "s4_effective_carriers_sha256": manifest["s4_effective_carriers_sha256"],
        "q4_effective_carriers_sha256": manifest["q4_effective_carriers_sha256"],
        "exposure_sample_plan": manifest["exposure_sample_plan"],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise NormalizedV2ContractError(f"D-{arm.upper()}E checkpoint/runtime source mismatch at {key}")
    if len(source.records) != 11 or manifest.get("target_nwb_opened_during_training_setup") is not False:
        raise NormalizedV2ContractError("exposure checker source reconstruction target-scope violation")
    return {
        "source_manifest_sha256": source.pilot_manifest_sha256, "normalizer": source.normalizer.manifest,
        "manifest": manifest, "effective_source_carriers_sha256": manifest["effective_source_carriers_sha256"],
        "source_recordings_opened": len(source.records),
    }


def check_terminal_pair(*, s4_checkpoint_path: str | Path, s4_config_path: str | Path,
                        q4_checkpoint_path: str | Path, q4_config_path: str | Path,
                        source_preflight_path: str | Path, terminal_preflight_path: str | Path,
                        output_path: str | Path) -> dict[str, Any]:
    terminal = assert_immutable_receipt(terminal_preflight_path, TERMINAL_PREFLIGHT_STATUS)
    if terminal.get("schema") != TERMINAL_PREFLIGHT_SCHEMA:
        raise NormalizedV2ContractError("exposure terminal-preflight schema drift")
    closure = _verify_source_closure(terminal.get("source_sha256", {}))
    source_pf = assert_immutable_receipt(source_preflight_path, SOURCE_PREFLIGHT_STATUS)
    if source_pf.get("schema") != SOURCE_PREFLIGHT_SCHEMA:
        raise NormalizedV2ContractError("exposure source-preflight schema drift")
    scope = source_pf.get("scope", {})
    if scope.get("source_recordings_opened") != 11 or scope.get("target_recordings_opened") != 0:
        raise NormalizedV2ContractError("exposure source preflight does not prove source-only closure")
    s4_cfg_path, q4_cfg_path = Path(s4_config_path).resolve(), Path(q4_config_path).resolve()
    s4_cfg, q4_cfg = _load_config(s4_cfg_path, "s4"), _load_config(q4_cfg_path, "q4")
    _s4_checkpoint, s4_meta = _load_checkpoint(Path(s4_checkpoint_path).resolve(), s4_cfg_path, "s4")
    _q4_checkpoint, q4_meta = _load_checkpoint(Path(q4_checkpoint_path).resolve(), q4_cfg_path, "q4")
    shared = _validate_pair(s4_meta, q4_meta)
    s4_runtime, q4_runtime = _rebuild_source(s4_cfg, s4_meta, "s4"), _rebuild_source(q4_cfg, q4_meta, "q4")
    if s4_runtime["normalizer"] != q4_runtime["normalizer"]:
        raise NormalizedV2ContractError("D-S4e/D-Q4e runtime paired normalizer differs")
    if s4_runtime["effective_source_carriers_sha256"] == q4_runtime["effective_source_carriers_sha256"]:
        raise NormalizedV2ContractError("D-S4e/D-Q4e runtime carriers unexpectedly identical")
    receipt = {
        "schema": CHECKER_SCHEMA, "status": CHECKER_STATUS,
        "scope": {"opened": "exactly 11 source NWBs per arm reconstruction", "target_recordings_opened": 0,
                  "target_recordings_enumerated": 0, "minival_or_formal_or_evalai_opened": False,
                  "cuda_constructed_or_launched": False, "trainer_constructed": False, "checkpoint_created": False},
        "terminal_preflight": {"path": str(Path(terminal_preflight_path).resolve()), "sha256": sha256_file(terminal_preflight_path)},
        "source_preflight": {"path": str(Path(source_preflight_path).resolve()), "sha256": sha256_file(source_preflight_path)},
        "source_closure": closure,
        "checkpoints": {
            "d_s4e": {"path": str(Path(s4_checkpoint_path).resolve()), "sha256": sha256_file(s4_checkpoint_path),
                       "config_path": str(s4_cfg_path), "config_sha256": sha256_file(s4_cfg_path), "metadata": s4_meta},
            "d_q4e": {"path": str(Path(q4_checkpoint_path).resolve()), "sha256": sha256_file(q4_checkpoint_path),
                       "config_path": str(q4_cfg_path), "config_sha256": sha256_file(q4_cfg_path), "metadata": q4_meta},
            "shared_pair_binding": shared,
        },
        "runtime_source": {"d_s4e": s4_runtime, "d_q4e": q4_runtime},
        "target_gate": {"both_epoch49_global_step180500_checkpoints_validated_before_target_open": True,
                        "future_target_evaluation_may_run_once_only_after_this_receipt": True,
                        "label": "LEAKAGE_DIAGNOSTIC_ONLY_NOT_FOR_SELECTION_OR_PAPER_MAIN_RESULT"},
    }
    path, digest = write_immutable_json(output_path, receipt)
    return {"status": CHECKER_STATUS, "receipt_path": str(path), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s4-checkpoint", required=True, type=Path); parser.add_argument("--s4-config", required=True, type=Path)
    parser.add_argument("--q4-checkpoint", required=True, type=Path); parser.add_argument("--q4-config", required=True, type=Path)
    parser.add_argument("--source-preflight", required=True, type=Path); parser.add_argument("--terminal-preflight", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path); args = parser.parse_args()
    print(json.dumps(check_terminal_pair(s4_checkpoint_path=args.s4_checkpoint, s4_config_path=args.s4_config,
        q4_checkpoint_path=args.q4_checkpoint, q4_config_path=args.q4_config, source_preflight_path=args.source_preflight,
        terminal_preflight_path=args.terminal_preflight, output_path=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
