#!/usr/bin/env python3
"""CPU-only, no-data preflight for B1's staged matched factorial.

This script composes Hydra configuration only.  It never imports a model or
data module, initializes CUDA/Torch, opens a checkpoint, scans NWB, or launches
training.  It writes a single immutable plan receipt with the predeclared
Stage-P/Stage-F lattice and implementation bindings.
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

from src.metrics import b1_m2_factorial as core


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise core.B1ContractError(message)


def _resolved_config(experiment: str, *, seed: int, fold: int) -> Mapping[str, Any]:
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str((PROJECT / "configs").resolve())):
        cfg = hydra.compose(
            config_name="train",
            overrides=[
                f"experiment={experiment}", f"seed={seed}", f"data.loso_fold={fold}",
                # The immutable preflight must compose exactly the execution
                # semantics printed by the B1 runner.  In particular,
                # train.py's default `test=true` would silently enable its
                # best-checkpoint route after fit.
                "train=true", "test=false",
            ],
        )
    # Composition happens outside a Hydra job, so `${hydra:runtime...}` paths
    # are intentionally left unresolved.  The B1 policy fields checked below
    # contain no runtime interpolation and are therefore safe to inspect
    # without importing/initializing a trainer.
    value = OmegaConf.to_container(cfg, resolve=False)
    _need(isinstance(value, Mapping), f"{experiment}: Hydra did not compose an object")
    return value


def _validate_composed_cell(spec: core.CellSpec) -> dict[str, Any]:
    experiment = core.expected_config_name(spec.carrier, spec.loss_mode)
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
        "loso_fold": spec.fold,
        "calibration_n_trials": 33,
        "random_calibration": False,
        "smooth_calibration": False,
        "window_size": 50,
        "max_trial_length": 100,
        "use_intertrials": True,
        "interpolate_trials": True,
        "interpolate_trials_kind": "cubic",
        "include_heldout_in_fit": False,
        "include_heldout_in_test": False,
        "query_start_trial": 0,
        "heldin_query_start_trial": 0,
        "heldin_query_end_trial": None,
        "side_feature_group": "t4",
    }.items():
        _need(data.get(key) == expected, f"{spec.key}: data.{key} drift")
    for key, expected in {
        "variant": "B3S",
        "side_dim": 4,
        "hidden_dim": 64,
        "freeze_decoder": True,
        "loss_mode": spec.loss_mode,
    }.items():
        _need(model.get(key) == expected, f"{spec.key}: model.{key} drift")
    expected_lambda = (1.0, 0.0) if spec.loss_mode == "task_plus_y" else (1.0, 0.1)
    _need(float(model.get("lambda_y")) == expected_lambda[0], f"{spec.key}: lambda_y drift")
    _need(float(model.get("lambda_E")) == expected_lambda[1], f"{spec.key}: lambda_E drift")
    _need(cfg.get("no_early_stopping") is True, f"{spec.key}: early stopping must be disabled")
    _need(cfg.get("train") is True and cfg.get("test") is False, f"{spec.key}: train/test execution policy drift")
    _need(int(trainer.get("max_epochs", -1)) == 12, f"{spec.key}: max_epochs must be 12")
    _need("best_checkpoint" not in callbacks and "early_stopping" not in callbacks,
          f"{spec.key}: best-val checkpoint selection is forbidden")
    periodic = callbacks.get("periodic_checkpoint")
    _need(isinstance(periodic, Mapping), f"{spec.key}: every-epoch checkpoint callback missing")
    _need(periodic.get("every_n_epochs") == 1 and periodic.get("save_top_k") == -1,
          f"{spec.key}: every epoch must be retained")
    if spec.carrier == "t4":
        _need(data.get("_target_") == "src.data.falcon_datamodule.FalconDataModule", f"{spec.key}: T4 data target drift")
    else:
        _need(data.get("_target_") == "src.data.b1_m2_matched_z4_datamodule.B1M2MatchedZ4DataModule",
              f"{spec.key}: Z4 must use post-standardization matched wrapper")
        _need(data.get("b1_z4_mask_after_standardization") is True, f"{spec.key}: Z4 mask policy drift")
    # This is deliberately static: no dataset/checkpoint opened.  It captures
    # the complete resolved config needed to detect later binding drift.
    science = core.science_config_projection(cfg)
    return {
        "cell": spec.key,
        "experiment": experiment,
        "science_config": science,
        "science_config_sha256": core.sha256_payload(science),
        # Retained for forensic comparison only.  The scorer binds the full
        # science projection above, not this opaque whole-config hash.
        "resolved_config_sha256": core.sha256_payload(cfg),
        "data_target": data.get("_target_"),
        "loss": {"mode": model.get("loss_mode"), "lambda_y": model.get("lambda_y"), "lambda_E": model.get("lambda_E")},
    }


def build_preflight(root: Path) -> dict[str, Any]:
    bindings = core.source_bindings(root)
    stage_p = [_validate_composed_cell(cell) for cell in core.cells_for_stage("P")]
    stage_f = [_validate_composed_cell(cell) for cell in core.cells_for_stage("F")]
    expansion_manifest = {
        "stage": "F",
        "predeclared_folds": list(core.STAGE_F_FOLDS),
        "expected_cell_keys": [cell.key for cell in core.cells_for_stage("F")],
        "trigger": {
            "stage_p_mean_interaction_at_least": core.STAGE_P_PRACTICAL_INTERACTION,
            "stage_p_all_three_seed_interactions_positive": True,
        },
        "post_stage_p_selection_forbidden": ["fold", "seed", "carrier", "loss_mode", "epoch_window"],
    }
    return {
        "schema_version": core.SCHEMA_VERSION,
        "receipt_kind": "b1_m2_factorial_cpu_preflight",
        "screen_id": core.SCREEN_ID,
        "status": "CPU_PREFLIGHT_READY_GPU_NOT_AUTHORIZED",
        "operations": {
            "torch_imported": False,
            "cuda_initialized": False,
            "nwb_opened": False,
            "teacher_checkpoint_bytes_hashed": True,
            "checkpoint_deserialized": False,
            "model_forward_called": False,
            "training_started": False,
            "formal_test_or_external_heldout_opened": False,
        },
        "scope": {
            "task": "FALCON M2 development internal LOSO only",
            "formal_test_or_external_heldout_forbidden": True,
            "target_session_backward_updates": False,
            "target_session_decoder_weight_updates": False,
            "support_direction_labels_used_for_carrier": True,
            "query_behavior_loaded_for_validation_scoring": True,
            "query_behavior_used_for_gradient_updates": False,
            "query_behavior_used_for_carrier_fit": False,
            "query_behavior_used_for_normalizer_fit": False,
            "query_behavior_used_for_checkpoint_selection": False,
            "query_behavior_stage_p_used_for_stage_f_continuation_gate": True,
            "query_behavior_stage_f_used_for_further_continuation_gate": False,
            "external_heldout_opened": False,
            "teacher_provenance": {
                "during_B1_target_weight_updates": False,
                "legacy_frozen_teacher_pretraining_included_B1_validation_session": True,
                "clean_teacher_target_exclusion": False,
                "interpretation": "internal_development_only",
            },
        },
        "factorial": {
            "arms": ["T4×task_plus_y", "Z4×task_plus_y", "T4×task_plus_y_plus_E", "Z4×task_plus_y_plus_E"],
            "primary_estimand": "(T4-Z4)_task_plus_y - (T4-Z4)_task_plus_y_plus_E",
            "epoch_window": list(core.EPOCH_WINDOW),
            "stage_p": {"folds": list(core.STAGE_P_FOLDS), "cells": len(stage_p), "inference": "descriptive 3-seed consistency only"},
            "stage_f": {
                "folds": list(core.STAGE_F_FOLDS), "cells": len(stage_f),
                "gate": {"mean_interaction_at_least": core.STAGE_P_PRACTICAL_INTERACTION, "all_three_seeds_positive": True},
                "if_gate_fails": "STOP_B1_NO_STAGE_F",
                "inference": "three predeclared Stage-F LOSO sessions × three seeds confirmatory primary only after Stage-P routing pass",
            },
        },
        "compute_and_reuse_audit": {
            "stage_p_fresh_cells_required": 12,
            "stage_f_predeclared_fresh_cells_if_gated": 36,
            "full_program_fresh_cells_if_gated": 48,
            "wall_time_estimate": "NOT_ESTIMATED_NO_B1_BENCHMARK; do not invent a GPU-hour claim",
            "exact_checkpoint_reuse": {
                "admitted_cells": [],
                "status": "NO_LEGAL_REUSE_IDENTIFIED",
                "rule": (
                    "A checkpoint is reusable only after scorer proof of exact full science-config, "
                    "split/query/normalizer, seed/fold/factor, teacher-byte, source-binding, and fixed-epoch retention identity."
                ),
            },
        },
        "carrier_provenance": {
            "T4": "ordinary standardized [a,c,m,b] from calibration-trial direction labels; source-only normalizer",
            "Z4": "same T4 fit and same normalizer, then zeros_like on standardized [N,4] before model consumption",
            "both_arms_target_session_direction_labels_used_for_carrier": True,
            "z4_model_visible_carrier": "all_zero_mask_after_standardized_t4",
        },
        "implementation_bindings": bindings,
        "implementation_bindings_sha256": core.sha256_payload(bindings),
        "stage_p_cells": stage_p,
        "stage_f_cells": stage_f,
        "stage_f_expansion_manifest": {
            **expansion_manifest,
            "sha256": core.sha256_payload(expansion_manifest),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new immutable receipt path")
    parser.add_argument("--dry-run", action="store_true", help="print the payload digest without writing")
    args = parser.parse_args()
    payload = build_preflight(REPO_ROOT)
    core.validate_preflight_payload(payload)
    if args.dry_run:
        print(json.dumps({"status": payload["status"], "payload_sha256": core.sha256_payload(payload), "stage_p_cells": len(payload["stage_p_cells"]), "stage_f_cells": len(payload["stage_f_cells"])}, indent=2))
        return
    digest = core.write_immutable_json(args.out, payload)
    print(json.dumps({"receipt": str(args.out.resolve()), "sha256": digest, "status": payload["status"]}, indent=2))


if __name__ == "__main__":
    main()
