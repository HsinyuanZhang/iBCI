#!/usr/bin/env python3
"""CPU-only, no-data preflight for C2's sampling-objective factorial.

Composes Hydra configuration only.  Never imports a model or data module,
initializes CUDA/Torch, opens a checkpoint, scans NWB, or launches training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import hydra
from omegaconf import OmegaConf

PROJECT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.metrics import c2_m2_sampling as core


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise core.C2ContractError(message)


def _resolved_config(experiment: str, *, seed: int, fold: int) -> Mapping[str, Any]:
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str((PROJECT / "configs").resolve())):
        cfg = hydra.compose(
            config_name="train",
            overrides=[
                f"experiment={experiment}",
                f"seed={seed}",
                f"data.loso_fold={fold}",
                "train=true",
                "test=false",
            ],
        )
    value = OmegaConf.to_container(cfg, resolve=False)
    _need(isinstance(value, Mapping), f"{experiment}: Hydra did not compose an object")
    return value


def _validate_composed_template(sampling: str, carrier: str) -> dict[str, Any]:
    spec = core.CellSpec(sampling, carrier, fold=0, seed=42)
    experiment = core.expected_config_name(sampling, carrier)
    cfg = _resolved_config(experiment, seed=spec.seed, fold=spec.fold)
    data = cfg.get("data")
    model = cfg.get("model")
    trainer = cfg.get("trainer")
    callbacks = cfg.get("callbacks")
    _need(isinstance(data, Mapping) and isinstance(model, Mapping), f"{spec.key}: missing data/model")
    _need(isinstance(trainer, Mapping) and isinstance(callbacks, Mapping), f"{spec.key}: missing trainer/callbacks")
    for key, expected in {
        "task": "m2",
        "validation_protocol": "loso",
        "calibration_n_trials": 33,
        "random_calibration": False,
        "window_size": 50,
        "include_heldout_in_fit": False,
        "include_heldout_in_test": False,
        "side_feature_group": "t4",
        "reshuffle_train_sampler_each_epoch": False,
        "balance_session_batches": spec.balance_session_batches,
    }.items():
        _need(data.get(key) == expected, f"{spec.key}: data.{key} drift")
    _need("window_budget_per_session" not in data, f"{spec.key}: window_budget must remain unrouted")
    _need(model.get("variant") == "B3S", f"{spec.key}: variant drift")
    _need(model.get("loss_mode") == core.LOSS_MODE, f"{spec.key}: loss_mode drift")
    _need(float(model.get("lambda_y")) == core.LAMBDA_Y, f"{spec.key}: lambda_y drift")
    _need(float(model.get("lambda_E")) == core.LAMBDA_E, f"{spec.key}: lambda_E drift")
    _need("r2" not in str(model.get("loss_mode")).lower(), f"{spec.key}: R2-native loss is forbidden")
    _need(cfg.get("no_early_stopping") is True, f"{spec.key}: early stopping must be disabled")
    _need(cfg.get("train") is True and cfg.get("test") is False, f"{spec.key}: train/test policy drift")
    _need(int(trainer.get("max_epochs", -1)) == 12, f"{spec.key}: max_epochs must be 12")
    _need("best_checkpoint" not in callbacks and "early_stopping" not in callbacks, f"{spec.key}: best-val forbidden")
    periodic = callbacks.get("periodic_checkpoint")
    _need(isinstance(periodic, Mapping), f"{spec.key}: every-epoch checkpoint callback missing")
    _need(periodic.get("every_n_epochs") == 1 and periodic.get("save_top_k") == -1, f"{spec.key}: retain every epoch")
    if carrier == "t4":
        _need(data.get("_target_") == "src.data.falcon_datamodule.FalconDataModule", f"{spec.key}: T4 data target drift")
    else:
        _need(
            data.get("_target_") == "src.data.c2_m2_matched_z4_datamodule.C2M2MatchedZ4DataModule",
            f"{spec.key}: Z4 must use the C2 matched wrapper",
        )
        _need(data.get("c2_z4_mask_after_standardization") is True, f"{spec.key}: Z4 mask policy drift")
    science = core.science_config_projection(cfg)
    return {
        "sampling": sampling,
        "carrier": carrier,
        "experiment": experiment,
        "science_config": science,
        "science_config_sha256": core.sha256_payload(science),
        "data_target": data.get("_target_"),
        "balance_session_batches": data.get("balance_session_batches"),
        "loss": {"mode": model.get("loss_mode"), "lambda_y": model.get("lambda_y"), "lambda_E": model.get("lambda_E")},
    }


def build_preflight(root: Path) -> dict[str, Any]:
    bindings = core.source_bindings(root)
    templates = [
        _validate_composed_template(sampling, carrier)
        for sampling in core.SAMPLINGS
        for carrier in core.CARRIERS
    ]
    cells = [
        {
            "cell": cell.key,
            "experiment": core.expected_config_name(cell.sampling, cell.carrier),
            "sampling": cell.sampling,
            "carrier": cell.carrier,
            "fold": cell.fold,
            "seed": cell.seed,
            "balance_session_batches": cell.balance_session_batches,
        }
        for cell in core.expected_cells()
    ]
    return {
        "schema_version": core.SCHEMA_VERSION,
        "receipt_kind": "c2_m2_sampling_cpu_preflight",
        "screen_id": core.SCREEN_ID,
        "status": "CPU_PREFLIGHT_READY_GPU_NOT_AUTHORIZED",
        "operations": {
            "torch_imported": False,
            "cuda_initialized": False,
            "nwb_opened": False,
            "checkpoint_deserialized": False,
            "model_forward_called": False,
            "training_started": False,
            "formal_test_or_external_heldout_opened": False,
            "sealed_formal_test_sessions_opened": False,
        },
        "scope": {
            "task": "FALCON M2 development internal LOSO only",
            "formal_test_or_external_heldout_forbidden": True,
            "sealed_formal_test_sessions": sorted(core.SEALED_FORMAL_TEST_SESSIONS),
            "sua_a2_sampler_claim": False,
            "validation_sampler_balanced": False,
            "window_budget_per_session_used": False,
            "r2_native_loss": False,
        },
        "existing_lever": dict(core.EXISTING_LEVER),
        "factorial": {
            "sampling": list(core.SAMPLINGS),
            "carriers": list(core.CARRIERS),
            "seeds": list(core.SEEDS),
            "folds": list(core.FOLDS),
            "cells": len(cells),
            "primary_estimand": "unweighted session-mean R2: equal_session(T4) - legacy(T4)",
            "anti_generic_estimand": "delta_T4 - delta_Z4",
            "epoch_window": list(core.EPOCH_WINDOW),
            "loss_mode": core.LOSS_MODE,
        },
        "templates": templates,
        "cells": cells,
        "implementation_bindings": bindings,
        "implementation_bindings_sha256": core.sha256_payload(bindings),
        "gpu_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new immutable receipt path")
    parser.add_argument("--dry-run", action="store_true", help="print the payload digest without writing")
    args = parser.parse_args()
    payload = build_preflight(REPO_ROOT)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "payload_sha256": core.sha256_payload(payload),
                    "cells": len(payload["cells"]),
                    "gpu_authorized": False,
                },
                indent=2,
            )
        )
        return
    digest = core.write_immutable_json(args.out, payload)
    print(json.dumps({"receipt": str(args.out.resolve()), "sha256": digest, "status": payload["status"]}, indent=2))


if __name__ == "__main__":
    main()
