#!/usr/bin/env python3
"""Strict, terminal-only target evaluator for H-C, H-RS and H-LS.

This is evaluation code, not a trainer or a selection program.  It refuses to
open a target recording until the immutable source-closure receipt and all
three fixed epoch-49 checkpoints have been bound.  The three predeclared
target views are deliberately asymmetric: H-C receives ``full`` support,
H-RS receives only ``row`` support, and H-LS receives only ``label`` support.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
from omegaconf import OmegaConf
import torch

from src.data.h1_m4_eb_normalized_v2 import (
    H1M4EBNormalizedV2DataModule,
    H1M4EBNormalizedV2StrictTargetDataset,
)
from src.data.h1_carrierid_shuffle import H1CarrierIdShuffleDataModule
from src.data.h1_m4_eb_pilot import (
    H1_M4_FOLD0_TARGET,
    load_target_records,
    validate_target_receipt_binding,
)
from src.h1_m4_eb_normalized_v2_contract import (
    NORMALIZER_FLOOR,
    NORMALIZER_FORMULA,
    NormalizedV2ContractError,
    assert_immutable_receipt,
    canonical_sha256,
    sha256_file,
    write_immutable_json,
)
from src.models.h1_carrierid_module import H1_CARRIERID_CHECKPOINT_SCHEMA
from scripts.h1_carrierid_evaluate import _evaluate
from scripts.h1_carrierid_paired_launcher import _verify_source_closure
from scripts.h1_carrierid_shuffle_preflight import (
    PREFLIGHT_SCHEMA as SHUFFLE_SOURCE_PREFLIGHT_SCHEMA,
    PREFLIGHT_STATUS as SHUFFLE_SOURCE_PREFLIGHT_STATUS,
)


PREFLIGHT_STATUS = "PASS_H1_CARRIERID_RS_LS_TERMINAL_EVALUATOR_SOURCE_CLOSURE_NONLAUNCH"
PREFLIGHT_SCHEMA = "h1_carrierid_h32_rs_ls_terminal_evaluator_preflight_v1"
TERMINAL_SCHEMA = "h1_carrierid_h32_rs_ls_terminal_eval_v1"
SHUFFLE_CHECKPOINT_SCHEMA = "h1_carrierid_h32_shuffle_terminal_checkpoint_v1"
ARM_TO_TARGET_VARIANT = {"full": "full", "rs": "row", "ls": "label"}


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise NormalizedV2ContractError(f"{label} must be a 64-hex SHA-256")
    return value


def _load_and_validate_config(path: Path, arm: str) -> Any:
    """Validate the resolved, saved run config rather than a mutable template."""

    if not path.is_file():
        raise FileNotFoundError(path)
    config = OmegaConf.load(path)
    expected_data_target = (
        "src.data.h1_m4_eb_normalized_v2.H1M4EBNormalizedV2DataModule"
        if arm == "full"
        else "src.data.h1_carrierid_shuffle.H1CarrierIdShuffleDataModule"
    )
    expected_model_target = (
        "src.models.h1_carrierid_module.H1CarrierIdLitModule"
        if arm == "full"
        else "src.models.h1_carrierid_shuffle_module.H1CarrierIdShuffleLitModule"
    )
    expected_protocol = (
        "h1_carrierid_h32_fold0_source_only_v1"
        if arm == "full"
        else "h1_carrierid_h32_rs_ls_source_only_v1"
    )
    fixed = {
        "protocol_id": expected_protocol,
        "seed": 42,
        "train": True,
        "test": False,
        "ckpt_path": None,
    }
    for field, expected in fixed.items():
        if config.get(field) != expected:
            raise NormalizedV2ContractError(f"H-{arm} resolved config drift at {field}")
    if (
        str(config.pilot.fold_date) != "19250101"
        or str(config.pilot.arm) != arm
        or int(config.pilot.calibration_n_trials) != 4
        or int(config.pilot.batch_size) != 32
        or int(config.pilot.fixed_terminal_epochs) != 50
        or bool(config.pilot.no_checkpoint_selection) is not True
        or str(config.data._target_) != expected_data_target
        or str(config.model._target_) != expected_model_target
        or int(config.data.calibration_n_trials) != 4
        or int(config.data.batch_size) != 32
        or int(config.data.window_size) != 700
        or int(config.data.max_trial_length) != 1024
        or int(config.data.seed) != 42
        or float(config.data.normalizer_floor) != NORMALIZER_FLOOR
        or int(config.trainer.max_epochs) != 50
        or int(config.trainer.min_epochs) != 50
        or int(config.trainer.limit_val_batches) != 0
        or int(config.trainer.num_sanity_val_steps) != 0
        or str(config.trainer.precision) != "32-true"
        or float(config.model.optimizer.lr) != 5.0e-5
        or float(config.model.optimizer.weight_decay) != 0.0
        or bool(config.model.net.zero_carrier) is not False
        or int(config.model.net.carrier_hidden_dim) != 32
        or int(config.model.net.carrier_dim) != 4
        or int(config.model.net.carrier_trial_length) != 1024
    ):
        raise NormalizedV2ContractError(f"H-{arm} resolved config violates the frozen h32 contract")
    if arm in {"rs", "ls"} and str(config.data.carrier_intervention) != arm:
        raise NormalizedV2ContractError(f"H-{arm} source data intervention differs from model arm")
    return config


def _load_and_validate_checkpoint(path: Path, config_path: Path, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind a real terminal state dict to its resolved config and arm semantics."""

    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise NormalizedV2ContractError(f"H-{arm} checkpoint is not a Lightning state-dict checkpoint")
    if int(checkpoint.get("epoch", -1)) != 49 or int(checkpoint.get("global_step", 0)) <= 0:
        raise NormalizedV2ContractError(f"H-{arm} must be the real fixed terminal epoch 49")
    metadata = checkpoint.get("h1_carrierid")
    if not isinstance(metadata, dict):
        raise NormalizedV2ContractError(f"H-{arm} checkpoint lacks h1_carrierid metadata")

    expected_schema = H1_CARRIERID_CHECKPOINT_SCHEMA if arm == "full" else SHUFFLE_CHECKPOINT_SCHEMA
    expected_mode = "full" if arm == "full" else f"source_{arm}_shuffle_training_input"
    required = {
        "schema": expected_schema,
        "fold_date": "19250101",
        "arm": arm,
        "checkpoint_epoch_zero_based": 49,
        "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_selection",
        "config_sha256": sha256_file(config_path),
        "normalizer_formula": NORMALIZER_FORMULA,
        "normalizer_floor": NORMALIZER_FLOOR,
        "carrier_mode": expected_mode,
        "carrier_hidden_dim": 32,
        "carrier_dim": 4,
        "carrier_trial_length": 1024,
        "deployment_target_optimizer_steps": 0,
        "deployment_target_backward_steps": 0,
    }
    for field, expected in required.items():
        if metadata.get(field) != expected:
            raise NormalizedV2ContractError(f"H-{arm} checkpoint metadata drift at {field}")
    for field in (
        "source_manifest_sha256", "normalizer_sha256", "source_cache_sha256",
        "normalized_cache_sha256", "source_hashes_sha256", "initial_state_sha256",
    ):
        _require_sha256(metadata.get(field), f"H-{arm} metadata.{field}")
    if arm in {"rs", "ls"}:
        if (
            metadata.get("carrier_intervention") != arm
            or _require_sha256(metadata.get("effective_source_carriers_sha256"), f"H-{arm} effective carrier") is None
            or metadata.get("effective_source_carriers_shape") != [116, 176, 4]
            or metadata.get("effective_source_carriers_count") != 116
            or metadata.get("effective_source_carriers_nonidentity_all") is not True
        ):
            raise NormalizedV2ContractError(f"H-{arm} checkpoint lacks its effective-source-carrier binding")
    return checkpoint, metadata


def _validate_shared_source_binding(full: Mapping[str, Any], candidate: Mapping[str, Any], arm: str) -> dict[str, str]:
    """Bind source quantities that must match across full/RS/LS training.

    ``source_manifest_sha256`` is intentionally excluded: RS/LS append their
    own effective shuffled-carrier fields to an otherwise shared source
    manifest.  That distinction is separately checked against the arm's own
    cache and immutable RS/LS CPU preflight below.
    """

    fields = (
        "fold_date", "normalizer_sha256", "source_cache_sha256", "normalized_cache_sha256",
        "source_hashes_sha256", "initial_state_sha256",
    )
    for field in fields:
        if full.get(field) != candidate.get(field):
            raise NormalizedV2ContractError(f"H-C/H-{arm} source binding mismatch at {field}")
    return {field: str(full[field]) for field in fields}


def _source_hashes_from_manifest(manifest: Mapping[str, Any]) -> str:
    """Mirror the checkpoint source-hash construction exactly."""

    return canonical_sha256({
        "source_sessions": manifest.get("source_sessions"),
        "files": manifest.get("files"),
        "carrier_cache_sha256": manifest.get("carrier_cache_sha256"),
        "normalized_cache_sha256": manifest.get("normalized_cache_sha256"),
    })


def _validate_checkpoint_against_runtime_source(
    metadata: Mapping[str, Any], source: Any, arm: str
) -> dict[str, Any]:
    """Verify each checkpoint against the manifest made by its own data arm."""

    manifest = source.pilot_manifest()
    required = {
        "source_manifest_sha256": source.pilot_manifest_sha256,
        "normalizer_sha256": source.normalizer.normalizer_sha256,
        "source_cache_sha256": manifest["carrier_cache_sha256"],
        "normalized_cache_sha256": manifest["normalized_cache_sha256"],
        "source_hashes_sha256": _source_hashes_from_manifest(manifest),
    }
    for field, expected in required.items():
        if metadata.get(field) != expected:
            raise NormalizedV2ContractError(f"H-{arm} runtime arm-specific source binding drift at {field}")
    return {
        "source_manifest_sha256": source.pilot_manifest_sha256,
        "normalizer_sha256": source.normalizer.normalizer_sha256,
        "source_cache_sha256": manifest["carrier_cache_sha256"],
        "normalized_cache_sha256": manifest["normalized_cache_sha256"],
        "source_hashes_sha256": required["source_hashes_sha256"],
    }


def _shuffle_source_from_resolved_config(config: Any) -> H1CarrierIdShuffleDataModule:
    """Reconstruct one control arm's *source-only* cache from its saved config."""

    data = config.data
    source = H1CarrierIdShuffleDataModule(
        task=str(data.task),
        data_dir=str(data.data_dir),
        raw_receipt_path=str(data.raw_receipt_path),
        eb_receipt_path=str(data.eb_receipt_path),
        cache_dir=str(data.cache_dir),
        carrier_intervention=str(data.carrier_intervention),
        batch_size=int(data.batch_size),
        window_size=int(data.window_size),
        calibration_n_trials=int(data.calibration_n_trials),
        max_trial_length=int(data.max_trial_length),
        random_calibration=bool(data.random_calibration),
        smooth_calibration=bool(data.smooth_calibration),
        interpolate_trials=bool(data.interpolate_trials),
        interpolate_trials_kind=str(data.interpolate_trials_kind),
        num_workers=int(data.num_workers),
        pin_memory=bool(data.pin_memory),
        seed=int(data.seed),
        fixed_epochs=int(data.fixed_epochs),
        normalizer_floor=float(data.normalizer_floor),
    )
    source.setup("fit")
    return source


def _validate_control_effective_carrier(
    *,
    metadata: Mapping[str, Any],
    runtime_source: H1CarrierIdShuffleDataModule,
    source_preflight: Mapping[str, Any],
    arm: str,
) -> dict[str, Any]:
    """Tie each RS/LS effective carrier to both its cache and frozen preflight."""

    manifest = runtime_source.pilot_manifest()
    runtime = {
        "sha256": manifest.get("effective_source_carriers_sha256"),
        "shape": manifest.get("effective_source_carriers_shape"),
        "count": manifest.get("effective_source_carriers_count"),
        "nonidentity_all": manifest.get("effective_source_carriers_nonidentity_all"),
    }
    expected = source_preflight.get("effective_source_carriers", {}).get(arm)
    if not isinstance(expected, Mapping):
        raise NormalizedV2ContractError(f"RS/LS source preflight lacks effective carrier for {arm}")
    checkpoint = {
        "sha256": metadata.get("effective_source_carriers_sha256"),
        "shape": metadata.get("effective_source_carriers_shape"),
        "count": metadata.get("effective_source_carriers_count"),
        "nonidentity_all": metadata.get("effective_source_carriers_nonidentity_all"),
    }
    if runtime != dict(expected) or checkpoint != dict(expected):
        raise NormalizedV2ContractError(f"H-{arm.upper()} effective carrier differs across checkpoint/cache/preflight")
    for field in (
        "source_window_indices_sha256", "batch_order_sha256", "calibration_schedule_sha256",
        "carrier_cache_sha256", "normalized_cache_sha256", "normalizer_sha256",
    ):
        if source_preflight.get("shared_source_binding", {}).get(field) != manifest.get(field):
            raise NormalizedV2ContractError(f"H-{arm.upper()} source preflight/cache mismatch at {field}")
    return runtime


def _instantiate(config: Any, checkpoint: Mapping[str, Any], device: torch.device):
    model = hydra.utils.instantiate(config.model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _paired_delta(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pooled": float(candidate["pooled_r2"] - reference["pooled_r2"]),
        "per_session": {
            session: float(candidate["per_session"][session]["r2"] - reference["per_session"][session]["r2"])
            for session in H1_M4_FOLD0_TARGET
        },
    }


def evaluate_terminal(
    *,
    data_dir: str | Path,
    raw_receipt_path: str | Path,
    eb_receipt_path: str | Path,
    shared_cache_dir: str | Path,
    hc_checkpoint_path: str | Path,
    hc_config_path: str | Path,
    rs_checkpoint_path: str | Path,
    rs_config_path: str | Path,
    ls_checkpoint_path: str | Path,
    ls_config_path: str | Path,
    rs_ls_source_preflight_path: str | Path,
    evaluator_preflight_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """Perform the one permitted target evaluation after all source checks bind."""

    if device not in {"cuda", "cpu"}:
        raise ValueError("terminal evaluator device must be 'cuda' or 'cpu'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("terminal evaluator requested CUDA but CUDA is unavailable")
    preflight_path = Path(evaluator_preflight_path).resolve()
    preflight = assert_immutable_receipt(preflight_path, PREFLIGHT_STATUS)
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("scope", {}).get("target_opened") is not False
        or preflight.get("launch", {}).get("authorized") is not False
    ):
        raise NormalizedV2ContractError("terminal evaluator requires the immutable no-target preflight")
    # Code/config byte closure is verified before *any* checkpoint/source/target data access.
    source_closure = _verify_source_closure(ROOT, preflight.get("source_sha256", {}))
    rs_ls_source_preflight_path = Path(rs_ls_source_preflight_path).resolve()
    rs_ls_source_preflight = assert_immutable_receipt(rs_ls_source_preflight_path, SHUFFLE_SOURCE_PREFLIGHT_STATUS)
    if rs_ls_source_preflight.get("schema") != SHUFFLE_SOURCE_PREFLIGHT_SCHEMA:
        raise NormalizedV2ContractError("H-RS/H-LS evaluator requires the immutable RS/LS source CPU preflight")

    hc_cfg_path, rs_cfg_path, ls_cfg_path = map(Path, (hc_config_path, rs_config_path, ls_config_path))
    hc_config = _load_and_validate_config(hc_cfg_path, "full")
    rs_config = _load_and_validate_config(rs_cfg_path, "rs")
    ls_config = _load_and_validate_config(ls_cfg_path, "ls")
    hc_checkpoint, hc_metadata = _load_and_validate_checkpoint(Path(hc_checkpoint_path), hc_cfg_path, "full")
    rs_checkpoint, rs_metadata = _load_and_validate_checkpoint(Path(rs_checkpoint_path), rs_cfg_path, "rs")
    ls_checkpoint, ls_metadata = _load_and_validate_checkpoint(Path(ls_checkpoint_path), ls_cfg_path, "ls")
    shared_rs = _validate_shared_source_binding(hc_metadata, rs_metadata, "RS")
    shared_ls = _validate_shared_source_binding(hc_metadata, ls_metadata, "LS")

    # Still source-only.  H-C uses the canonical full cache.  RS/LS must be
    # recreated with their own source-intervention manifests rather than
    # falsely compared to H-C's (legitimately different) manifest SHA.
    source = H1M4EBNormalizedV2DataModule(
        task="h1",
        data_dir=str(Path(data_dir).resolve()),
        raw_receipt_path=str(Path(raw_receipt_path).resolve()),
        eb_receipt_path=str(Path(eb_receipt_path).resolve()),
        cache_dir=str(Path(shared_cache_dir).resolve()),
    )
    source.setup("fit")
    hc_runtime_source = _validate_checkpoint_against_runtime_source(hc_metadata, source, "C")
    rs_source = _shuffle_source_from_resolved_config(rs_config)
    ls_source = _shuffle_source_from_resolved_config(ls_config)
    rs_runtime_source = _validate_checkpoint_against_runtime_source(rs_metadata, rs_source, "RS")
    ls_runtime_source = _validate_checkpoint_against_runtime_source(ls_metadata, ls_source, "LS")
    rs_effective = _validate_control_effective_carrier(
        metadata=rs_metadata, runtime_source=rs_source, source_preflight=rs_ls_source_preflight, arm="rs",
    )
    ls_effective = _validate_control_effective_carrier(
        metadata=ls_metadata, runtime_source=ls_source, source_preflight=rs_ls_source_preflight, arm="ls",
    )

    evaluation_device = torch.device(device)
    hc_model = _instantiate(hc_config, hc_checkpoint, evaluation_device)
    rs_model = _instantiate(rs_config, rs_checkpoint, evaluation_device)
    ls_model = _instantiate(ls_config, ls_checkpoint, evaluation_device)
    del hc_checkpoint, rs_checkpoint, ls_checkpoint

    # First and only target data access.
    target = load_target_records(data_dir)
    validate_target_receipt_binding(target, source.plan, raw_receipt_path, eb_receipt_path)
    full_dataset = H1M4EBNormalizedV2StrictTargetDataset(target, source.plan, source.normalizer, "full")
    datasets = {
        "h_c": full_dataset.with_intervention(ARM_TO_TARGET_VARIANT["full"]),
        "h_rs": full_dataset.with_intervention(ARM_TO_TARGET_VARIANT["rs"]),
        "h_ls": full_dataset.with_intervention(ARM_TO_TARGET_VARIANT["ls"]),
    }
    support_hashes = full_dataset.support_and_carrier_hashes()
    canonical_window_hash = full_dataset.window_indices_sha256
    for name, dataset in datasets.items():
        if dataset.window_indices_sha256 != canonical_window_hash:
            raise NormalizedV2ContractError(f"{name} target intervention changed the query windows")
        if dataset.support_and_carrier_hashes() != support_hashes:
            raise NormalizedV2ContractError(f"{name} target intervention changed support/query identity")

    metrics = {
        "h_c": _evaluate(hc_model, datasets["h_c"], evaluation_device, "H-C/full"),
        "h_rs": _evaluate(rs_model, datasets["h_rs"], evaluation_device, "H-RS/row"),
        "h_ls": _evaluate(ls_model, datasets["h_ls"], evaluation_device, "H-LS/label"),
    }
    for arm, value in metrics.items():
        if value["r2_accumulator_dtype"] != "float64" or not value["state_immutable"]:
            raise NormalizedV2ContractError(f"{arm} evaluator did not meet float64/immutable-state contract")
        if value["query_window_indices_sha256"] != canonical_window_hash:
            raise NormalizedV2ContractError(f"{arm} evaluator reported a query-window mismatch")

    body = {
        "schema": TERMINAL_SCHEMA,
        "status": "PASS_H1_CARRIERID_RS_LS_TERMINAL_EVALUATED",
        "metric_contract": {
            "prediction_dtype": "float32",
            "r2_sse_tss_accumulator_dtype": "float64",
            "eval_no_grad": True,
            "state_hash_before_after": True,
            "checkpoint_selection": "fixed_epoch_49_after_50_source_epochs",
        },
        "binding_order": [
            "immutable_no_target_evaluator_preflight", "source_closure", "configs_and_checkpoints",
            "source_normalizer", "models", "target_once", "forward_only_evaluation",
        ],
        "preflight": {"path": str(preflight_path), "sha256": sha256_file(preflight_path), "source_closure": source_closure},
        "rs_ls_source_preflight": {
            "path": str(rs_ls_source_preflight_path),
            "sha256": sha256_file(rs_ls_source_preflight_path),
            "effective_source_carriers": {"rs": rs_effective, "ls": ls_effective},
        },
        "checkpoints": {
            name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path),
                   "config_path": str(Path(config).resolve()), "config_sha256": sha256_file(config), "metadata": metadata}
            for name, path, config, metadata in (
                ("h_c", hc_checkpoint_path, hc_config_path, hc_metadata),
                ("h_rs", rs_checkpoint_path, rs_config_path, rs_metadata),
                ("h_ls", ls_checkpoint_path, ls_config_path, ls_metadata),
            )
        },
        "shared_source_bindings": {"h_c_h_rs": shared_rs, "h_c_h_ls": shared_ls},
        "source": {
            "h_c_runtime_manifest": hc_runtime_source,
            "h_rs_runtime_manifest": rs_runtime_source,
            "h_ls_runtime_manifest": ls_runtime_source,
            "source_manifest_sha256_intentionally_arm_specific": True,
        },
        "target": {
            "sessions": list(H1_M4_FOLD0_TARGET),
            "files_sha256": {session: target[session].input_sha256 for session in H1_M4_FOLD0_TARGET},
            "support_and_carrier_hashes": support_hashes,
            "query_window_indices_sha256": canonical_window_hash,
            "predeclared_variants": {"h_c": "full", "h_rs": "row", "h_ls": "label"},
        },
        "metrics": {
            **metrics,
            "h_rs_minus_h_c": _paired_delta(metrics["h_rs"], metrics["h_c"]),
            "h_ls_minus_h_c": _paired_delta(metrics["h_ls"], metrics["h_c"]),
        },
        "data_scope": {"minival": False, "formal": False, "evalai": False, "target_optimizer_steps": 0, "target_backward_steps": 0},
    }
    path, digest = write_immutable_json(output_path, body)
    return {"status": body["status"], "receipt_path": str(path), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--raw-receipt-path", required=True, type=Path)
    parser.add_argument("--eb-receipt-path", required=True, type=Path)
    parser.add_argument("--shared-cache-dir", required=True, type=Path)
    parser.add_argument("--hc-checkpoint-path", required=True, type=Path)
    parser.add_argument("--hc-config-path", required=True, type=Path)
    parser.add_argument("--rs-checkpoint-path", required=True, type=Path)
    parser.add_argument("--rs-config-path", required=True, type=Path)
    parser.add_argument("--ls-checkpoint-path", required=True, type=Path)
    parser.add_argument("--ls-config-path", required=True, type=Path)
    parser.add_argument("--rs-ls-source-preflight-path", required=True, type=Path)
    parser.add_argument("--evaluator-preflight-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    print(json.dumps(evaluate_terminal(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
