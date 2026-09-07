#!/usr/bin/env python3
"""Source-only epoch-49 pair checker for H1 fresh D-S4/D-Q4 retraining.

This checker is deliberately separate from both the original distribution
training path and the target evaluator.  It never imports target loaders.  A
future evaluator may open target recordings only after this checker has
produced an immutable passing receipt for *both* arms.
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

from src.data.h1_carrierid_distribution import H1CarrierIdDistributionDataModule
from src.h1_m4_eb_normalized_v2_contract import (
    NORMALIZER_FORMULA,
    NormalizedV2ContractError,
    assert_immutable_receipt,
    canonical_sha256,
    sha256_file,
    write_immutable_json,
)
from src.models.components.h1_carrierid_spint import (
    H1_CARRIERID_PARAMETERS,
    H1_CARRIERID_WHOLE_MODEL_PARAMETERS,
)
from scripts.h1_carrierid_distribution_preflight import PREFLIGHT_SCHEMA as SOURCE_PREFLIGHT_SCHEMA
from scripts.h1_carrierid_distribution_preflight import PREFLIGHT_STATUS as SOURCE_PREFLIGHT_STATUS


CHECKER_SCHEMA = "h1_carrierid_h32_fresh_distribution_terminal_checkpoint_pair_v1"
CHECKER_STATUS = "PASS_H1_CARRIERID_H32_FRESH_D_S4_D_Q4_SOURCE_TERMINAL_CHECKPOINT_PAIR_NONLAUNCH"
TERMINAL_PREFLIGHT_SCHEMA = "h1_carrierid_h32_fresh_distribution_terminal_evaluator_preflight_v1"
TERMINAL_PREFLIGHT_STATUS = "PASS_H1_CARRIERID_H32_FRESH_D_S4_D_Q4_TERMINAL_EVALUATOR_SOURCE_CLOSURE_NONLAUNCH"
CHECKPOINT_SCHEMA = "h1_carrierid_h32_fresh_distribution_terminal_checkpoint_v1"
ARM_METADATA = {
    "s4": {"arm": "D-S4", "carrier_mode": "standard_support_t_to_t_plus_3"},
    "q4": {
        "arm": "D-Q4",
        "carrier_mode": "source_query_local_t_plus_4_to_t_plus_7_LEAKAGE_DIAGNOSTIC_ONLY",
    },
}


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise NormalizedV2ContractError(f"{label} must be a SHA-256")
    return value


def _verify_source_closure(expected: Mapping[str, Any]) -> dict[str, str]:
    if not expected:
        raise NormalizedV2ContractError("fresh distribution terminal preflight has no source closure")
    observed: dict[str, str] = {}
    for relative, expected_sha in expected.items():
        path = (ROOT / str(relative)).resolve()
        if not path.is_file() or path.parent == ROOT.parent:
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected_sha:
            raise NormalizedV2ContractError(f"fresh distribution terminal source closure drift at {relative}")
        observed[str(relative)] = actual
    return observed


def _load_and_validate_config(path: Path, arm: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    cfg = OmegaConf.load(path)
    fixed = {
        "protocol_id": "h1_carrierid_h32_fresh_distribution_source_only_v1",
        "seed": 42,
        "train": True,
        "test": False,
        "ckpt_path": None,
    }
    for field, value in fixed.items():
        if cfg.get(field) != value:
            raise NormalizedV2ContractError(f"D-{arm.upper()} config drift at {field}")
    if (
        str(cfg.pilot.fold_date) != "19250101"
        or str(cfg.pilot.arm) != arm
        or bool(cfg.pilot.leakage_diagnostic_only) is not True
        or int(cfg.pilot.fixed_terminal_epochs) != 50
        or int(cfg.pilot.windows_per_schedule) != 32
        or str(cfg.data._target_) != "src.data.h1_carrierid_distribution.H1CarrierIdDistributionDataModule"
        or str(cfg.model._target_) != "src.models.h1_carrierid_distribution_module.H1CarrierIdDistributionLitModule"
        or str(cfg.data.carrier_distribution_arm) != arm
        or int(cfg.data.batch_size) != 32
        or int(cfg.data.window_size) != 700
        or int(cfg.data.calibration_n_trials) != 4
        or int(cfg.data.max_trial_length) != 1024
        or int(cfg.data.fixed_epochs) != 50
        or int(cfg.data.windows_per_schedule) != 32
        or int(cfg.trainer.max_epochs) != 50
        or int(cfg.trainer.min_epochs) != 50
        or int(cfg.trainer.limit_val_batches) != 0
        or int(cfg.trainer.num_sanity_val_steps) != 0
        or str(cfg.trainer.precision) != "32-true"
        or float(cfg.model.optimizer.lr) != 5.0e-5
        or float(cfg.model.optimizer.weight_decay) != 0.0
        or bool(cfg.model.net.zero_carrier) is not False
        or int(cfg.model.net.carrier_hidden_dim) != 32
        or int(cfg.model.net.carrier_dim) != 4
        or int(cfg.model.net.carrier_trial_length) != 1024
    ):
        raise NormalizedV2ContractError(f"D-{arm.upper()} resolved config violates frozen matched contract")
    callback = cfg.callbacks.get("fixed_epoch50")
    if callback is None or callback.get("monitor") is not None or int(callback.get("every_n_epochs", -1)) != 50:
        raise NormalizedV2ContractError("D-S4/D-Q4 must use unselected fixed epoch-49 checkpoint callback")
    return cfg


def _load_and_validate_checkpoint(path: Path, config_path: Path, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise NormalizedV2ContractError(f"D-{arm.upper()} is not a Lightning state-dict checkpoint")
    if int(checkpoint.get("epoch", -1)) != 49 or int(checkpoint.get("global_step", 0)) <= 0:
        raise NormalizedV2ContractError(f"D-{arm.upper()} must be real fixed epoch=49 after source training")
    metadata = checkpoint.get("h1_carrierid")
    if not isinstance(metadata, dict):
        raise NormalizedV2ContractError(f"D-{arm.upper()} checkpoint lacks h1_carrierid metadata")
    required = {
        "schema": CHECKPOINT_SCHEMA,
        "fold_date": "19250101",
        "checkpoint_epoch_zero_based": 49,
        "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_selection",
        "config_sha256": sha256_file(config_path),
        "normalizer_formula": NORMALIZER_FORMULA,
        "carrier_hidden_dim": 32,
        "carrier_dim": 4,
        "carrier_trial_length": 1024,
        "carrierid_parameters": H1_CARRIERID_PARAMETERS,
        "whole_model_parameters": H1_CARRIERID_WHOLE_MODEL_PARAMETERS,
        "leakage_diagnostic_only": True,
        "not_for_selection_or_paper_main_result": True,
        "deployment_target_optimizer_steps": 0,
        "deployment_target_backward_steps": 0,
        **ARM_METADATA[arm],
    }
    for field, expected in required.items():
        if metadata.get(field) != expected:
            raise NormalizedV2ContractError(f"D-{arm.upper()} terminal checkpoint metadata drift at {field}")
    for field in (
        "source_manifest_sha256", "normalizer_sha256", "source_cache_sha256", "normalized_cache_sha256",
        "source_hashes_sha256", "initial_state_sha256", "source_schedule_sha256",
        "common_query_samples_sha256", "identity_schedule_sha256", "batch_order_sha256",
        "s4_effective_carriers_sha256", "q4_effective_carriers_sha256",
    ):
        _require_sha(metadata.get(field), f"D-{arm.upper()} metadata.{field}")
    if metadata.get("q4_label_scope") != "t+4..t+7 source analogue of deliberate query-label leakage":
        raise NormalizedV2ContractError("D-S4/D-Q4 terminal checkpoint lost frozen Q4 label-scope disclosure")
    return checkpoint, metadata


def _validate_pair(s4: Mapping[str, Any], q4: Mapping[str, Any]) -> dict[str, str]:
    shared = (
        "fold_date", "normalizer_sha256", "source_cache_sha256", "normalized_cache_sha256",
        "source_hashes_sha256", "initial_state_sha256", "source_schedule_sha256",
        "common_query_samples_sha256", "identity_schedule_sha256", "batch_order_sha256",
        "s4_effective_carriers_sha256", "q4_effective_carriers_sha256",
    )
    for field in shared:
        if s4.get(field) != q4.get(field):
            raise NormalizedV2ContractError(f"D-S4/D-Q4 matched terminal checkpoint mismatch at {field}")
    if s4.get("source_manifest_sha256") == q4.get("source_manifest_sha256"):
        raise NormalizedV2ContractError("D-S4/D-Q4 arm-specific source manifests unexpectedly identical")
    return {field: str(s4[field]) for field in shared}


def _source_hashes_from_manifest(manifest: Mapping[str, Any]) -> str:
    # Mirror H1CarrierIdLitModule._bind_initial_state byte-for-byte.  The
    # additive distribution manifest names its per-source listing
    # ``source_files``; the inherited module currently binds the historical
    # ``files`` key (therefore ``None``) inside this derived digest.  The full
    # arm-specific manifest SHA above still binds the actual source listing.
    return canonical_sha256({
        "source_sessions": manifest.get("source_sessions"), "files": manifest.get("files"),
        "carrier_cache_sha256": manifest.get("carrier_cache_sha256"),
        "normalized_cache_sha256": manifest.get("normalized_cache_sha256"),
    })


def _rebuild_runtime_source(cfg: Any, metadata: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Reconstruct and bind one real 11-source arm; target code is absent."""

    data = cfg.data
    source = H1CarrierIdDistributionDataModule(
        task=str(data.task), data_dir=str(data.data_dir), raw_receipt_path=str(data.raw_receipt_path),
        eb_receipt_path=str(data.eb_receipt_path), cache_dir=str(data.cache_dir),
        carrier_distribution_arm=str(data.carrier_distribution_arm), batch_size=int(data.batch_size),
        window_size=int(data.window_size), calibration_n_trials=int(data.calibration_n_trials),
        max_trial_length=int(data.max_trial_length), num_workers=int(data.num_workers),
        pin_memory=bool(data.pin_memory), seed=int(data.seed), fixed_epochs=int(data.fixed_epochs),
        windows_per_schedule=int(data.windows_per_schedule),
    )
    source.setup("fit")
    manifest = source.pilot_manifest()
    required = {
        "source_manifest_sha256": source.pilot_manifest_sha256,
        "normalizer_sha256": source.normalizer.normalizer_sha256,
        "source_cache_sha256": manifest["carrier_cache_sha256"],
        "normalized_cache_sha256": manifest["normalized_cache_sha256"],
        "source_hashes_sha256": _source_hashes_from_manifest(manifest),
        "source_schedule_sha256": manifest["source_schedule_sha256"],
        "common_query_samples_sha256": manifest["common_query_samples_sha256"],
        "identity_schedule_sha256": manifest["identity_schedule_sha256"],
        "batch_order_sha256": manifest["batch_order_sha256"],
        "s4_effective_carriers_sha256": manifest["s4_effective_carriers_sha256"],
        "q4_effective_carriers_sha256": manifest["q4_effective_carriers_sha256"],
    }
    for field, expected in required.items():
        if metadata.get(field) != expected:
            raise NormalizedV2ContractError(f"D-{arm.upper()} checkpoint/runtime source mismatch at {field}")
    if len(source.records) != 11 or manifest.get("target_nwb_opened_during_training_setup") is not False:
        raise NormalizedV2ContractError("D-S4/D-Q4 checker source reconstruction target-scope violation")
    return {
        "source_manifest_sha256": source.pilot_manifest_sha256,
        "normalizer": source.normalizer.manifest,
        "manifest": manifest,
        "effective_source_carriers_sha256": manifest["effective_source_carriers_sha256"],
        "source_recordings_opened": len(source.records),
    }


def check_terminal_pair(
    *,
    s4_checkpoint_path: str | Path,
    s4_config_path: str | Path,
    q4_checkpoint_path: str | Path,
    q4_config_path: str | Path,
    source_preflight_path: str | Path,
    terminal_preflight_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate both source-trained epoch-49 arms before any target access."""

    terminal_preflight = assert_immutable_receipt(terminal_preflight_path, TERMINAL_PREFLIGHT_STATUS)
    if terminal_preflight.get("schema") != TERMINAL_PREFLIGHT_SCHEMA:
        raise NormalizedV2ContractError("fresh distribution terminal preflight schema drift")
    source_closure = _verify_source_closure(terminal_preflight.get("source_sha256", {}))
    source_preflight = assert_immutable_receipt(source_preflight_path, SOURCE_PREFLIGHT_STATUS)
    if source_preflight.get("schema") != SOURCE_PREFLIGHT_SCHEMA:
        raise NormalizedV2ContractError("fresh distribution source CPU preflight schema drift")
    scope = source_preflight.get("scope", {})
    if scope.get("source_recordings_opened") != 11 or scope.get("target_recordings_opened") != 0:
        raise NormalizedV2ContractError("source preflight does not prove exact source-only closure")
    s4_cfg_path, q4_cfg_path = Path(s4_config_path).resolve(), Path(q4_config_path).resolve()
    s4_cfg = _load_and_validate_config(s4_cfg_path, "s4")
    q4_cfg = _load_and_validate_config(q4_cfg_path, "q4")
    s4_checkpoint, s4_meta = _load_and_validate_checkpoint(Path(s4_checkpoint_path).resolve(), s4_cfg_path, "s4")
    q4_checkpoint, q4_meta = _load_and_validate_checkpoint(Path(q4_checkpoint_path).resolve(), q4_cfg_path, "q4")
    del s4_checkpoint, q4_checkpoint
    shared = _validate_pair(s4_meta, q4_meta)
    s4_runtime = _rebuild_runtime_source(s4_cfg, s4_meta, "s4")
    q4_runtime = _rebuild_runtime_source(q4_cfg, q4_meta, "q4")
    if s4_runtime["normalizer"] != q4_runtime["normalizer"]:
        raise NormalizedV2ContractError("D-S4/D-Q4 runtime paired normalizer differs")
    if s4_runtime["effective_source_carriers_sha256"] == q4_runtime["effective_source_carriers_sha256"]:
        raise NormalizedV2ContractError("D-S4/D-Q4 runtime effective carrier input unexpectedly identical")
    receipt = {
        "schema": CHECKER_SCHEMA, "status": CHECKER_STATUS,
        "scope": {
            "opened": "exactly 11 public held-in-calib source NWBs for each arm reconstruction",
            "target_recordings_opened": 0, "target_recordings_enumerated": 0,
            "minival_or_formal_or_evalai_opened": False, "cuda_constructed_or_launched": False,
            "trainer_constructed": False, "checkpoint_created": False,
        },
        "terminal_preflight": {"path": str(Path(terminal_preflight_path).resolve()), "sha256": sha256_file(terminal_preflight_path)},
        "source_preflight": {"path": str(Path(source_preflight_path).resolve()), "sha256": sha256_file(source_preflight_path)},
        "source_closure": source_closure,
        "checkpoints": {
            "d_s4": {"path": str(Path(s4_checkpoint_path).resolve()), "sha256": sha256_file(s4_checkpoint_path),
                     "config_path": str(s4_cfg_path), "config_sha256": sha256_file(s4_cfg_path), "metadata": s4_meta},
            "d_q4": {"path": str(Path(q4_checkpoint_path).resolve()), "sha256": sha256_file(q4_checkpoint_path),
                     "config_path": str(q4_cfg_path), "config_sha256": sha256_file(q4_cfg_path), "metadata": q4_meta},
            "shared_pair_binding": shared,
        },
        "runtime_source": {"d_s4": s4_runtime, "d_q4": q4_runtime},
        "target_gate": {
            "both_fixed_epoch49_checkpoints_validated_before_target_open": True,
            "future_target_evaluation_may_run_once_only_after_this_receipt": True,
            "label": "LEAKAGE_DIAGNOSTIC_ONLY_NOT_FOR_SELECTION_OR_PAPER_MAIN_RESULT",
        },
    }
    path, digest = write_immutable_json(output_path, receipt)
    return {"status": CHECKER_STATUS, "receipt_path": str(path), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s4-checkpoint", required=True, type=Path)
    parser.add_argument("--s4-config", required=True, type=Path)
    parser.add_argument("--q4-checkpoint", required=True, type=Path)
    parser.add_argument("--q4-config", required=True, type=Path)
    parser.add_argument("--source-preflight", required=True, type=Path)
    parser.add_argument("--terminal-preflight", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(check_terminal_pair(
        s4_checkpoint_path=args.s4_checkpoint, s4_config_path=args.s4_config,
        q4_checkpoint_path=args.q4_checkpoint, q4_config_path=args.q4_config,
        source_preflight_path=args.source_preflight, terminal_preflight_path=args.terminal_preflight,
        output_path=args.output,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
