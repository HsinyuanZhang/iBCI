#!/usr/bin/env python3
"""No-target binding preflight for the additive H1 seed-43 terminal evaluator.

This module is deliberately source-only.  It validates the three fixed epoch-49
seed-43 checkpoints and their resolved configs, reconstructs the seed-43 source
cache/schedule, and writes an immutable binding receipt.  It never imports the
fold-0 target loader and never opens a target recording.  The companion
``h1_carrierid_seed43_evaluate.py`` reuses the binding helpers and opens the
target only after this complete binding step has succeeded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import hydra
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.h1_carrierid_seed43 import H1CarrierIdSeed43DataModule
from src.h1_m4_eb_normalized_v2_contract import (
    NORMALIZER_FLOOR,
    NORMALIZER_FORMULA,
    NormalizedV2ContractError,
    assert_immutable_receipt,
    canonical_sha256,
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
from src.h1_m4_eb_normalized_v2_contract import V2_CHECKPOINT_SCHEMA


PREFLIGHT_SCHEMA = "h1_carrierid_h32_seed43_three_arm_terminal_preflight_v2"
PREFLIGHT_STATUS = "PASS_H1_CARRIERID_H32_SEED43_THREE_ARM_TERMINAL_PREFLIGHT_NO_TARGET"
SEED43_SOURCE_PRELIGHT_STATUS = "PASS_H1_CARRIERID_H32_SEED43_THREE_ARM_REAL_SOURCE_CPU_PREFLIGHT_NONLAUNCH"

ARM_TO_META_KEY = {
    "hs": "h1_m4_eb_normalized_v2",
    "hc": "h1_carrierid",
    "hc0": "h1_carrierid",
}
ARM_TO_PILOT_ARM = {"hs": "base", "hc": "full", "hc0": "zero"}
ARM_TO_CONFIG_TARGET = {
    "hs": "src.models.h1_m4_eb_normalized_v2_module.H1M4EBNormalizedV2PilotLitModule",
    "hc": "src.models.h1_carrierid_module.H1CarrierIdLitModule",
    "hc0": "src.models.h1_carrierid_module.H1CarrierIdLitModule",
}
ARM_TO_CARRIER_MODE = {
    "hs": None,
    "hc": "full",
    "hc0": "literal_zero_at_model_boundary",
}

_COMMON_META_FIELDS = (
    "fold_date",
    "source_manifest_sha256",
    "normalizer_sha256",
    "source_cache_sha256",
    "normalized_cache_sha256",
    "source_hashes_sha256",
    "normalizer_formula",
    "normalizer_floor",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NormalizedV2ContractError(message)


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256",
    )
    return value


def initialization_binding_from_source_preflight(
    preflight: Mapping[str, Any], loaded: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind initial-state provenance without making a false H-S equality claim.

    H-C/H-C0 have fully materialized parameters at construction, so their
    source-preflight post-forward hashes must equal the train-time initial
    hashes written into their checkpoints.  H-S contains a ``LazyLinear``
    identity projection.  Its training module materializes that layer in
    ``on_train_batch_start`` after Trainer/DataLoader setup, whereas the CPU
    source preflight materializes it via a direct eval forward after resetting
    the seed.  Those RNG histories are not equivalent.  A mismatch is therefore
    neither evidence of target leakage nor an admissible reason to weaken the
    H-C/H-C0 equality gate.

    The H-S relation is recorded as explicitly non-comparable, while config,
    source manifest/cache/normalizer, fixed epoch/global-step, state finiteness,
    and the two carrier-arm equality bindings remain strict elsewhere.
    """

    initialization = preflight.get("initialization")
    first_batch = preflight.get("first_source_batch")
    closure = preflight.get("source_sha256")
    _require(isinstance(initialization, Mapping), "seed43 source preflight lacks initialization evidence")
    _require(isinstance(first_batch, Mapping), "seed43 source preflight lacks first source batch evidence")
    _require(isinstance(closure, Mapping), "seed43 source preflight lacks source closure")
    preflight_states = initialization.get("state_sha256")
    predictions = initialization.get("prediction_sha256")
    _require(isinstance(preflight_states, Mapping) and isinstance(predictions, Mapping), "seed43 source preflight initialization hashes malformed")

    result: dict[str, Any] = {}
    for arm in ("hc", "hc0"):
        recorded = _require_sha256(preflight_states.get(arm), f"source preflight {arm} initial state")
        checkpoint = _require_sha256(
            loaded[arm]["metadata"].get("initial_state_sha256"), f"{arm} checkpoint initial state"
        )
        _require(recorded == checkpoint, f"{arm}: initial state does not bind source preflight")
        result[arm] = {
            "mode": "STRICT_PREFLIGHT_POST_FORWARD_EQUALS_CHECKPOINT_TRAIN_INITIAL_STATE",
            "preflight_state_sha256": recorded,
            "checkpoint_initial_state_sha256": checkpoint,
            "equal": True,
        }

    _require(
        result["hc"]["checkpoint_initial_state_sha256"] == result["hc0"]["checkpoint_initial_state_sha256"],
        "seed43 H-C/H-C0 initial state mismatch",
    )

    hs_preflight = _require_sha256(preflight_states.get("hs"), "source preflight hs post-forward state")
    hs_prediction = _require_sha256(predictions.get("hs"), "source preflight hs post-forward prediction")
    hs_checkpoint = _require_sha256(
        loaded["hs"]["metadata"].get("initial_state_sha256"), "hs checkpoint train-time initial state"
    )
    hs_identity = _require_sha256(first_batch.get("identity_sha256"), "source preflight hs first-batch identity")
    hs_module = _require_sha256(
        closure.get("src/models/h1_m4_eb_normalized_v2_module.py"), "source preflight H-S lazy-materialization module"
    )
    result["hs"] = {
        "mode": "NONCOMPARABLE_LAZY_TRAIN_TIME_MATERIALIZATION",
        "preflight_post_forward_state_sha256": hs_preflight,
        "checkpoint_train_time_initial_state_sha256": hs_checkpoint,
        "hashes_equal": hs_preflight == hs_checkpoint,
        "preflight_direct_materialization": {
            "method": "direct eval-mode first source forward after seed reset",
            "first_source_identity_sha256": hs_identity,
            "post_forward_prediction_sha256": hs_prediction,
        },
        "training_materialization": {
            "method": "H1M4EBNormalizedV2PilotLitModule.on_train_batch_start lazy.initialize_parameters",
            "source_module_sha256": hs_module,
            "why_hash_equality_is_not_a_gate": (
                "the training hook executes after Trainer/DataLoader setup, so its RNG history is not the "
                "preflight direct-forward RNG history"
            ),
        },
        "hard_gates_retained": [
            "resolved_config_sha256", "source_manifest_cache_normalizer_binding", "fixed_epoch49_global_step180500",
            "finite_checkpoint_state", "source_only_scope",
        ],
    }
    return result


def _checkpoint_state_finite(checkpoint: Mapping[str, Any], arm: str) -> None:
    state = checkpoint.get("state_dict")
    _require(isinstance(state, Mapping), f"{arm}: checkpoint state_dict is missing")
    tensors = [value for value in state.values() if torch.is_tensor(value)]
    _require(bool(tensors), f"{arm}: checkpoint has no tensor state")
    _require(all(bool(torch.isfinite(value).all()) for value in tensors), f"{arm}: non-finite checkpoint state")


def validate_seed43_config(path: str | Path, arm: str) -> Any:
    """Validate one resolved seed-43 Hydra config without touching data."""

    if arm not in ARM_TO_META_KEY:
        raise ValueError(arm)
    config_path = Path(path)
    _require(config_path.is_file(), f"{arm}: resolved config is missing: {config_path}")
    config = OmegaConf.load(config_path)
    expected = {
        "protocol_id": "h1_carrierid_h32_seed43_source_only_v1",
        "seed": 43,
        "train": True,
        "test": False,
        "ckpt_path": None,
        "logger": False,
    }
    for key, value in expected.items():
        _require(config.get(key) == value, f"{arm}: config drift at {key}")
    _require(str(config.data._target_) == "src.data.h1_carrierid_seed43.H1CarrierIdSeed43DataModule", f"{arm}: data module drift")
    data_fields = {
        "task": "h1", "batch_size": 32, "window_size": 700,
        "calibration_n_trials": 4, "max_trial_length": 1024,
        "random_calibration": True, "smooth_calibration": False,
        "interpolate_trials": True, "interpolate_trials_kind": "cubic",
        "num_workers": 0, "seed": 43, "fixed_epochs": 50,
        "normalizer_floor": NORMALIZER_FLOOR,
    }
    for key, value in data_fields.items():
        _require(config.data.get(key) == value, f"{arm}: data.{key} drift")
    pilot = config.pilot
    _require(str(pilot.fold_date) == "19250101", f"{arm}: pilot fold date drift")
    _require(str(pilot.arm) == ARM_TO_PILOT_ARM[arm], f"{arm}: pilot arm drift")
    _require(int(pilot.calibration_n_trials) == 4, f"{arm}: pilot support-trial drift")
    _require(int(pilot.fixed_terminal_epochs) == 50 and bool(pilot.no_checkpoint_selection), f"{arm}: selection drift")
    if arm == "hs":
        _require(bool(pilot.train_residual) is False, "hs: residual arm drift")
    else:
        _require(bool(pilot.zero_carrier) is (arm == "hc0"), f"{arm}: zero-carrier drift")
    trainer = config.trainer
    for key, value in {"max_epochs": 50, "min_epochs": 50, "limit_val_batches": 0, "num_sanity_val_steps": 0}.items():
        _require(int(trainer.get(key)) == value, f"{arm}: trainer.{key} drift")
    _require(str(trainer.precision) == "32-true", f"{arm}: trainer precision drift")
    _require(float(config.model.optimizer.lr) == 5.0e-5 and float(config.model.optimizer.weight_decay) == 0.0, f"{arm}: optimizer drift")
    _require(str(config.model._target_) == ARM_TO_CONFIG_TARGET[arm], f"{arm}: model target drift")
    if arm != "hs":
        net = config.model.net
        for key, value in {"carrier_hidden_dim": 32, "carrier_dim": 4, "carrier_trial_length": 1024}.items():
            _require(int(net.get(key)) == value, f"{arm}: model.net.{key} drift")
    return config


def load_seed43_checkpoint(path: str | Path, config_path: str | Path, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate one CPU checkpoint; no data or CUDA access occurs."""

    if arm not in ARM_TO_META_KEY:
        raise ValueError(arm)
    checkpoint_path, resolved_config_path = Path(path), Path(config_path)
    _require(checkpoint_path.is_file(), f"{arm}: checkpoint is missing: {checkpoint_path}")
    config = validate_seed43_config(resolved_config_path, arm)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, dict) and "state_dict" in checkpoint, f"{arm}: invalid Lightning checkpoint")
    _require(int(checkpoint.get("epoch", -1)) == 49, f"{arm}: terminal checkpoint epoch must be 49")
    # Every arm has 3610 source batches/epoch and exactly 50 fixed epochs.
    # Binding a merely-positive global step could silently admit a truncated
    # or resumed run, so the terminal evaluator requires the exact count.
    _require(int(checkpoint.get("global_step", 0)) == 180500, f"{arm}: terminal checkpoint global_step must be 180500")
    _checkpoint_state_finite(checkpoint, arm)
    meta = checkpoint.get(ARM_TO_META_KEY[arm])
    _require(isinstance(meta, dict), f"{arm}: checkpoint metadata key is missing")
    common = {
        "fold_date": "19250101",
        "checkpoint_epoch_zero_based": 49,
        "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_selection",
        "normalizer_formula": NORMALIZER_FORMULA,
        "normalizer_floor": NORMALIZER_FLOOR,
    }
    for key, value in common.items():
        _require(meta.get(key) == value, f"{arm}: checkpoint metadata drift at {key}")
    _require(meta.get("arm") == ARM_TO_PILOT_ARM[arm], f"{arm}: checkpoint arm drift")
    if arm == "hs":
        for key, value in {
            "schema": V2_CHECKPOINT_SCHEMA,
            "residual_trainable": False,
            "base_residual_literal_zero": True,
        }.items():
            _require(meta.get(key) == value, f"hs: checkpoint metadata drift at {key}")
    else:
        for key, value in {
            "deployment_target_optimizer_steps": 0,
            "deployment_target_backward_steps": 0,
        }.items():
            _require(meta.get(key) == value, f"{arm}: checkpoint metadata drift at {key}")
        for key, value in {
            "schema": "h1_carrierid_h32_fold0_terminal_checkpoint_v1",
            "carrier_mode": ARM_TO_CARRIER_MODE[arm],
            "carrier_hidden_dim": 32,
            "carrier_dim": 4,
            "carrier_trial_length": 1024,
            "carrierid_parameters": H1_CARRIERID_PARAMETERS,
            "whole_model_parameters": H1_CARRIERID_WHOLE_MODEL_PARAMETERS,
            "spint_identity_parameters": H1_SPINT_ID_PARAMETERS,
            "spint_whole_model_parameters": H1_SPINT_WHOLE_MODEL_PARAMETERS,
            "first_post_carrier_columns_literal_zero_at_init": True,
        }.items():
            _require(meta.get(key) == value, f"{arm}: checkpoint metadata drift at {key}")
    _require(sha256_file(resolved_config_path) == meta.get("config_sha256"), f"{arm}: config SHA does not bind checkpoint")
    _require_sha256(meta.get("initial_state_sha256"), f"{arm}: checkpoint initial state")
    return checkpoint, dict(meta)


def _source_closure() -> dict[str, str]:
    """Hash the additive evaluator and every source/model primitive it binds."""

    files = [
        "scripts/h1_carrierid_evaluate.py",  # sealed predecessor, hash-only binding
        "scripts/h1_carrierid_seed43_preflight.py",
        "scripts/h1_carrierid_seed43_terminal_preflight.py",
        "scripts/h1_carrierid_seed43_evaluate.py",
        "src/data/h1_carrierid_seed43.py",
        "src/data/h1_m4_eb_pilot.py",
        "src/data/h1_m4_eb_normalized_v2.py",
        "src/h1_m4_eb_normalized_v2_contract.py",
        "src/models/falcon_module.py",
        "src/models/h1_m4_eb_normalized_v2_module.py",
        "src/models/h1_carrierid_module.py",
        "src/models/components/h1_m4_eb_normalized_v2_residual_spint.py",
        "src/models/components/h1_carrierid_spint.py",
        "configs/data/falcon_h1_carrierid_seed43.yaml",
        "configs/model/falcon_h1_m4_eb_normalized_v2.yaml",
        "configs/model/falcon_h1_carrierid.yaml",
        "configs/callbacks/h1_m4_eb_normalized_v2_terminal.yaml",
        "configs/callbacks/h1_carrierid_terminal.yaml",
        "configs/experiment/h1_carrierid_hs_seed43.yaml",
        "configs/experiment/h1_carrierid_hc_seed43.yaml",
        "configs/experiment/h1_carrierid_hc0_seed43.yaml",
    ]
    return {relative: sha256_file(ROOT / relative) for relative in files}


def bind_seed43_terminal_inputs(
    *,
    preflight_path: str | Path,
    data_dir: str | Path,
    raw_receipt_path: str | Path,
    eb_receipt_path: str | Path,
    shared_cache_dir: str | Path,
    checkpoints: Mapping[str, str | Path],
    configs: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Perform the complete no-target binding and return reusable inputs."""

    receipt_path = Path(preflight_path).resolve()
    preflight = assert_immutable_receipt(receipt_path, SEED43_SOURCE_PRELIGHT_STATUS)
    _require(preflight.get("schema") == "h1_carrierid_h32_seed43_three_arm_source_cpu_preflight_v1", "wrong seed43 source preflight schema")
    _require(preflight.get("fixed_protocol", {}).get("requested_arms") == ["hs", "hc", "hc0"], "source preflight must cover all three arms")
    _require(preflight.get("scope", {}).get("target_recordings_opened") == 0, "seed43 source preflight opened a target")
    _require(preflight.get("scope", {}).get("target_recordings_enumerated") == 0, "seed43 source preflight enumerated a target")
    binding = preflight["source_binding"]
    expected_arms = ("hs", "hc", "hc0")
    _require(set(checkpoints) == set(expected_arms) and set(configs) == set(expected_arms), "seed43 terminal binding requires H-S/H-C/H-C0")
    loaded: dict[str, dict[str, Any]] = {}
    configs_loaded: dict[str, Any] = {}
    for arm in expected_arms:
        configs_loaded[arm] = validate_seed43_config(configs[arm], arm)
        checkpoint, meta = load_seed43_checkpoint(checkpoints[arm], configs[arm], arm)
        loaded[arm] = {"checkpoint": checkpoint, "metadata": meta, "path": str(Path(checkpoints[arm]).resolve()), "config_path": str(Path(configs[arm]).resolve())}
        for field, expected in {
            "normalizer_sha256": binding["normalizer_sha256"],
            "source_cache_sha256": binding["carrier_cache_sha256"],
            "normalized_cache_sha256": binding["normalized_cache_sha256"],
            "normalizer_formula": NORMALIZER_FORMULA,
            "normalizer_floor": NORMALIZER_FLOOR,
        }.items():
            _require(meta.get(field) == expected, f"{arm}: source binding drift at {field}")
    for field in _COMMON_META_FIELDS:
        values = {loaded[arm]["metadata"].get(field) for arm in expected_arms}
        _require(len(values) == 1, f"seed43 arm binding mismatch at {field}")
    initialization_binding = initialization_binding_from_source_preflight(preflight, loaded)

    source = H1CarrierIdSeed43DataModule(
        task="h1",
        data_dir=str(Path(data_dir).resolve()),
        raw_receipt_path=str(Path(raw_receipt_path).resolve()),
        eb_receipt_path=str(Path(eb_receipt_path).resolve()),
        cache_dir=str(Path(shared_cache_dir).resolve()),
    )
    source.setup("fit")
    manifest = source.pilot_manifest()
    _require(manifest["seed"] == 43 and manifest["fold_date"] == "19250101", "runtime source manifest seed/fold drift")
    _require(manifest["target_nwb_opened_during_training_setup"] is False, "runtime source setup opened a target NWB")
    _require(manifest["minival_or_heldout_enumerated"] is False, "runtime source setup enumerated held-out data")
    _require(manifest["calibration_schedule_sha256"] == binding["calibration_schedule_sha256"], "runtime seed43 schedule does not bind preflight")
    _require(manifest["calibration_schedule_sha256"] != preflight["seed42_comparison"]["seed42_calibration_schedule_sha256"], "seed43 schedule unexpectedly equals seed42")
    _require(source.pilot_manifest_sha256 == loaded["hs"]["metadata"]["source_manifest_sha256"], "runtime source manifest does not bind checkpoints")
    for arm in expected_arms:
        meta = loaded[arm]["metadata"]
        _require(meta["source_manifest_sha256"] == source.pilot_manifest_sha256, f"{arm}: source manifest mismatch")
    return {
        "source": source,
        "source_manifest": manifest,
        "source_manifest_sha256": source.pilot_manifest_sha256,
        "preflight": preflight,
        "preflight_path": str(receipt_path),
        "preflight_sha256": sha256_file(receipt_path),
        "configs": configs_loaded,
        "arms": loaded,
        "initialization_binding": initialization_binding,
    }


def run_preflight(
    *,
    preflight_path: str | Path,
    data_dir: str | Path,
    raw_receipt_path: str | Path,
    eb_receipt_path: str | Path,
    shared_cache_dir: str | Path,
    checkpoints: Mapping[str, str | Path],
    configs: Mapping[str, str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    output = Path(output_path).resolve()
    _require(not output.exists(), f"refusing to overwrite immutable seed43 terminal preflight: {output}")
    bound = bind_seed43_terminal_inputs(
        preflight_path=preflight_path, data_dir=data_dir, raw_receipt_path=raw_receipt_path,
        eb_receipt_path=eb_receipt_path, shared_cache_dir=shared_cache_dir,
        checkpoints=checkpoints, configs=configs,
    )
    checkpoint_payload = {
        arm: {
            "path": loaded["path"],
            "sha256": sha256_file(loaded["path"]),
            "config_path": loaded["config_path"],
            "config_sha256": sha256_file(loaded["config_path"]),
            "metadata": loaded["metadata"],
        }
        for arm, loaded in bound["arms"].items()
    }
    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "scope": {
            "opened": "exactly 11 public held-in-calibration source NWBs",
            "target_recordings_opened": 0,
            "target_recordings_enumerated": 0,
            "minival_opened_or_enumerated": False,
            "formal_heldout_opened_or_enumerated": False,
            "evalai_opened_or_enumerated": False,
            "cuda_constructed_or_launched": False,
            "trainer_launched": False,
            "checkpoint_selected": False,
            "final_evaluation_run": False,
        },
        "fixed_protocol": {
            "fold_date": "19250101", "seed": 43, "support_trials": 4,
            "epochs": 50, "fixed_terminal_epoch": 49, "selection": "none_fixed_terminal_epoch_only",
            "precision": "32-true", "learning_rate": 5.0e-5,
        },
        "source": {
            "preflight_path": bound["preflight_path"],
            "preflight_sha256": bound["preflight_sha256"],
            "manifest": bound["source_manifest"],
            "manifest_sha256": bound["source_manifest_sha256"],
            "calibration_schedule_sha256": bound["source_manifest"]["calibration_schedule_sha256"],
            "seed42_schedule_sha256": bound["preflight"]["seed42_comparison"]["seed42_calibration_schedule_sha256"],
            "schedule_is_new_for_seed43": True,
        },
        "checkpoints": checkpoint_payload,
        "initialization_binding": bound["initialization_binding"],
        "source_closure": _source_closure(),
        "sealed_seed42_evaluator_sha256": sha256_file(ROOT / "scripts/h1_carrierid_evaluate.py"),
        "launch": {
            "final_evaluator_authorized": False,
            "required_before_target_open": "root audit of this immutable no-target binding receipt",
        },
    }
    path, digest = write_immutable_json(output, payload)
    return {"status": PREFLIGHT_STATUS, "receipt_path": str(path), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-preflight", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/000954")
    parser.add_argument("--raw-receipt", required=True, type=Path)
    parser.add_argument("--eb-receipt", required=True, type=Path)
    parser.add_argument("--shared-cache-dir", required=True, type=Path)
    for arm in ("hs", "hc", "hc0"):
        parser.add_argument(f"--{arm}-checkpoint", required=True, type=Path)
        parser.add_argument(f"--{arm}-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_preflight(
        preflight_path=args.source_preflight, data_dir=args.data_dir,
        raw_receipt_path=args.raw_receipt, eb_receipt_path=args.eb_receipt,
        shared_cache_dir=args.shared_cache_dir,
        checkpoints={arm: getattr(args, f"{arm}_checkpoint") for arm in ("hs", "hc", "hc0")},
        configs={arm: getattr(args, f"{arm}_config") for arm in ("hs", "hc", "hc0") for _ in [0]},
        output_path=args.output,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
