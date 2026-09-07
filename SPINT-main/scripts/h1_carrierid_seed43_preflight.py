#!/usr/bin/env python3
"""Source-only CPU preflight for the fixed H-S/H-C/H-C0 seed-43 replication.

The only data API exercised here is the fold-0 *source* loader.  The two
fold-0 target recordings are not loaded, and this script has no minival,
formal-test, EvalAI, trainer, checkpoint-selection, or CUDA path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Iterable

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.h1_carrierid_seed43 import H1CarrierIdSeed43DataModule
from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    array_sha256,
    sha256_file,
    state_hash,
    write_immutable_json,
)


PREFLIGHT_SCHEMA = "h1_carrierid_h32_seed43_three_arm_source_cpu_preflight_v1"
PREFLIGHT_STATUS = "PASS_H1_CARRIERID_H32_SEED43_THREE_ARM_REAL_SOURCE_CPU_PREFLIGHT_NONLAUNCH"
ARM_TO_EXPERIMENT = {
    "hs": "h1_carrierid_hs_seed43",
    "hc": "h1_carrierid_hc_seed43",
    "hc0": "h1_carrierid_hc0_seed43",
}
ARM_TO_MODEL = {
    "hs": "src.models.h1_m4_eb_normalized_v2_module.H1M4EBNormalizedV2PilotLitModule",
    "hc": "src.models.h1_carrierid_module.H1CarrierIdLitModule",
    "hc0": "src.models.h1_carrierid_module.H1CarrierIdLitModule",
}
RAW_RECEIPT = ROOT.parent / "sua_exploration/results/h1_m4_population_decoder_carrier_date_lodo_v1/H1_M4_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json"
EB_RECEIPT = ROOT.parent / "sua_exploration/results/h1_m4_empirical_bayes_confidence_carrier_date_lodo_v1/H1_M4_EMPIRICAL_BAYES_CONFIDENCE_CARRIER_CPU_RECEIPT.json"
SEED42_SCHEDULE_MANIFEST = ROOT / "pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/shared_source_cache/full/fold0_source_schedule.manifest.json"


def _seed43() -> None:
    random.seed(43)
    np.random.seed(43)
    torch.manual_seed(43)


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compose(experiment: str) -> Any:
    """Compose only; ``--cfg job`` has no data-loader or Trainer side effect."""

    command = [
        sys.executable,
        str(ROOT / "src/train.py"),
        f"experiment={experiment}",
        "seed=43",
        "test=false",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "--cfg",
        "job",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"seed43 Hydra composition failed for {experiment}:\n{result.stdout}\n{result.stderr}")
    config = OmegaConf.create(result.stdout)
    if not isinstance(OmegaConf.to_container(config, resolve=False), dict):
        raise NormalizedV2ContractError(f"seed43 Hydra composition did not return a mapping for {experiment}")
    return config


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NormalizedV2ContractError(message)


def _audit_config(config: Any, arm: str) -> None:
    _require(config.get("protocol_id") == "h1_carrierid_h32_seed43_source_only_v1", f"{arm}: protocol drift")
    _require(int(config.seed) == 43 and bool(config.train) and not bool(config.test), f"{arm}: seed/train/test drift")
    _require(config.ckpt_path is None and config.logger is False, f"{arm}: checkpoint/logger drift")
    _require(str(config.data._target_) == "src.data.h1_carrierid_seed43.H1CarrierIdSeed43DataModule", f"{arm}: data module drift")
    for field, expected in {
        "task": "h1", "batch_size": 32, "window_size": 700, "calibration_n_trials": 4,
        "max_trial_length": 1024, "random_calibration": True, "smooth_calibration": False,
        "interpolate_trials": True, "interpolate_trials_kind": "cubic", "num_workers": 0,
        "seed": 43, "fixed_epochs": 50, "normalizer_floor": 1.0e-12,
    }.items():
        _require(config.data.get(field) == expected, f"{arm}: data.{field} drift")
    for field, expected in {
        "fold_date": "19250101", "calibration_n_trials": 4, "fixed_terminal_epochs": 50,
        "no_checkpoint_selection": True,
    }.items():
        _require(config.pilot.get(field) == expected, f"{arm}: pilot.{field} drift")
    _require(int(config.trainer.max_epochs) == 50 and int(config.trainer.min_epochs) == 50, f"{arm}: epoch budget drift")
    _require(int(config.trainer.limit_val_batches) == 0 and int(config.trainer.num_sanity_val_steps) == 0, f"{arm}: validation drift")
    _require(str(config.trainer.precision) == "32-true", f"{arm}: precision drift")
    _require(str(config.model._target_) == ARM_TO_MODEL[arm], f"{arm}: model type drift")
    _require(float(config.model.optimizer.lr) == 5.0e-5 and float(config.model.optimizer.weight_decay) == 0.0, f"{arm}: optimizer drift")

    if arm == "hs":
        _require(config.pilot.arm == "base" and config.pilot.train_residual is False, "hs: base residual contract drift")
    elif arm == "hc":
        _require(config.pilot.arm == "full" and config.pilot.zero_carrier is False, "hc: full carrier contract drift")
    elif arm == "hc0":
        _require(config.pilot.arm == "zero" and config.pilot.zero_carrier is True, "hc0: literal-zero contract drift")
    else:  # pragma: no cover - guarded by CLI parser
        raise ValueError(arm)


def _source_datamodule(cache_dir: Path) -> H1CarrierIdSeed43DataModule:
    data = H1CarrierIdSeed43DataModule(
        task="h1",
        data_dir=str(ROOT / "data/000954"),
        raw_receipt_path=str(RAW_RECEIPT),
        eb_receipt_path=str(EB_RECEIPT),
        cache_dir=str(cache_dir),
    )
    data.setup("fit")
    return data


def _initial_forward(config: Any, batch: tuple[Any, ...]) -> tuple[Any, str, str]:
    """Materialize a source-only model and prove its first forward is read-only."""

    _seed43()
    model = hydra.utils.instantiate(config.model)
    model.eval()
    neural, _target, identity, _session, carrier = batch
    with torch.no_grad():
        # Materialize any lazy identity layer before hashing the initialized
        # model.  This is initialization, not a train/eval data-dependent
        # update; all arms receive the identical first source batch.
        output = model(
            neural.to(dtype=torch.float32),
            calib_trialized_neural_features=identity.to(dtype=torch.float32),
            carrier=carrier.to(dtype=torch.float32),
        )
    before = state_hash(model.state_dict())
    with torch.no_grad():
        repeated = model(
            neural.to(dtype=torch.float32),
            calib_trialized_neural_features=identity.to(dtype=torch.float32),
            carrier=carrier.to(dtype=torch.float32),
        )
    after = state_hash(model.state_dict())
    if before != after:
        raise NormalizedV2ContractError("source-only initial forward mutated the model state")
    if not torch.equal(output, repeated):
        raise NormalizedV2ContractError("source-only initial forward is not deterministic in eval mode")
    return model, before, array_sha256(output.detach().cpu().numpy())


def _source_binding(manifest: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "seed", "fold_date", "source_sessions", "target_sessions_not_opened", "files",
        "raw_receipt_sha256", "eb_receipt_sha256", "transform_sha256", "carrier_cache_sha256",
        "normalized_cache_sha256", "normalizer_sha256", "source_window_indices_sha256",
        "batch_order_sha256", "calibration_schedule_sha256", "batches_per_epoch",
        "scheduled_samples_per_epoch", "epochs", "calibration_n_trials", "window_size",
        "target_nwb_opened_during_training_setup", "minival_or_heldout_enumerated",
    )
    return {field: manifest[field] for field in fields}


def _source_closure(arms: Iterable[str]) -> dict[str, str]:
    """Hash only the bytes necessary for the requested arm subset.

    A remote H-C0-only preflight deliberately does not require local H-S/H-C
    source files; its closure is nevertheless complete for its own training
    arm and is rechecked before any remote launch.
    """

    requested = set(arms)
    files: list[str] = [
        "scripts/h1_carrierid_seed43_preflight.py",
        "scripts/h1_carrierid_seed43_launcher.py",
        "src/data/h1_carrierid_seed43.py",
        "src/data/h1_m4_eb_pilot.py",
        "src/data/h1_m4_eb_normalized_v2.py",
        "src/h1_m4_eb_normalized_v2_contract.py",
        "src/models/falcon_module.py",
        "src/models/components/spint.py",
        "configs/data/falcon_h1_carrierid_seed43.yaml",
    ]
    if "hs" in requested:
        files.extend([
            "src/models/h1_m4_eb_normalized_v2_module.py",
            "src/models/components/h1_m4_eb_normalized_v2_residual_spint.py",
            "configs/model/falcon_h1_m4_eb_normalized_v2.yaml",
            "configs/callbacks/h1_m4_eb_normalized_v2_terminal.yaml",
            "configs/experiment/h1_carrierid_hs_seed43.yaml",
        ])
    if {"hc", "hc0"} & requested:
        files.extend([
            "src/models/h1_carrierid_module.py",
            "src/models/components/h1_carrierid_spint.py",
            "configs/model/falcon_h1_carrierid.yaml",
            "configs/callbacks/h1_carrierid_terminal.yaml",
        ])
    if "hc" in requested:
        files.append("configs/experiment/h1_carrierid_hc_seed43.yaml")
    if "hc0" in requested:
        files.append("configs/experiment/h1_carrierid_hc0_seed43.yaml")
    return {relative: sha256_file(ROOT / relative) for relative in files}


def run(*, cache_dir: Path, output_path: Path, arms: Iterable[str]) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise NormalizedV2ContractError("seed43 source preflight requires CUDA_VISIBLE_DEVICES unset")
    requested = tuple(arms)
    if not requested or any(arm not in ARM_TO_EXPERIMENT for arm in requested):
        raise ValueError(f"--arms must be a nonempty subset of {sorted(ARM_TO_EXPERIMENT)}")
    if len(set(requested)) != len(requested):
        raise ValueError("--arms may not repeat an arm")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable seed43 preflight: {output_path}")

    configs = {arm: _compose(ARM_TO_EXPERIMENT[arm]) for arm in requested}
    for arm, config in configs.items():
        _audit_config(config, arm)

    source = _source_datamodule(cache_dir.resolve())
    manifest = source.pilot_manifest()
    _require(manifest["target_nwb_opened_during_training_setup"] is False, "seed43 source preflight opened a target NWB")
    _require(manifest["minival_or_heldout_enumerated"] is False, "seed43 source preflight enumerated minival/heldout")
    _require(manifest["seed"] == 43 and manifest["epochs"] == 50, "seed43 source manifest drift")
    if not SEED42_SCHEDULE_MANIFEST.is_file():
        raise FileNotFoundError(f"sealed seed42 schedule manifest missing: {SEED42_SCHEDULE_MANIFEST}")
    seed42_schedule = json.loads(SEED42_SCHEDULE_MANIFEST.read_text(encoding="utf-8"))["calibration_schedule_sha256"]
    _require(manifest["calibration_schedule_sha256"] != seed42_schedule, "seed43 schedule unexpectedly equals the sealed seed42 schedule")

    source.train_batch_sampler.reset_epoch()
    batch = next(iter(source.train_dataloader()))
    models: dict[str, Any] = {}
    initial_states: dict[str, str] = {}
    initial_prediction_sha256: dict[str, str] = {}
    for arm, config in configs.items():
        model, initial_states[arm], initial_prediction_sha256[arm] = _initial_forward(config, batch)
        models[arm] = model
    hc_hc0_equal: bool | None = None
    hc_hc0_forward_equal: bool | None = None
    if {"hc", "hc0"}.issubset(models):
        hc_hc0_equal = initial_states["hc"] == initial_states["hc0"]
        hc_hc0_forward_equal = initial_prediction_sha256["hc"] == initial_prediction_sha256["hc0"]
        _require(hc_hc0_equal and hc_hc0_forward_equal, "H-C/H-C0 must begin as identical functions at seed43")

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
            "checkpoint_created": False,
            "checkpoint_selected": False,
        },
        "fixed_protocol": {
            "fold_date": "19250101",
            "requested_arms": list(requested),
            "seed": 43,
            "support_trials": 4,
            "epochs": 50,
            "precision": "32-true",
            "optimizer": "Adam",
            "learning_rate": 5.0e-5,
            "weight_decay": 0.0,
            "fixed_terminal_epoch": 49,
            "selection": "none_fixed_terminal_epoch_only",
        },
        "source_binding": _source_binding(manifest),
        "seed42_comparison": {
            "sealed_seed42_schedule_manifest": str(SEED42_SCHEDULE_MANIFEST),
            "seed42_calibration_schedule_sha256": seed42_schedule,
            "seed43_calibration_schedule_sha256": manifest["calibration_schedule_sha256"],
            "schedule_is_new_for_seed43": True,
        },
        "first_source_batch": {
            "neural_shape": list(batch[0].shape),
            "identity_shape": list(batch[2].shape),
            "carrier_shape": list(batch[4].shape),
            "neural_sha256": array_sha256(batch[0].numpy()),
            "identity_sha256": array_sha256(batch[2].numpy()),
            "carrier_sha256": array_sha256(batch[4].numpy()),
            "session_order": list(batch[3]),
        },
        "initialization": {
            "state_sha256": initial_states,
            "prediction_sha256": initial_prediction_sha256,
            "h_c_h_c0_state_equal": hc_hc0_equal,
            "h_c_h_c0_initial_forward_equal": hc_hc0_forward_equal,
        },
        "source_sha256": _source_closure(requested),
        "launch": {
            "launch_authorized": False,
            "required_before_execute": "root audit of immutable local and remote CPU preflights plus terminal H-RS/H-LS evaluator review",
        },
    }
    path, digest = write_immutable_json(output_path, payload)
    return {"status": PREFLIGHT_STATUS, "receipt_path": str(path), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARM_TO_EXPERIMENT), default=("hs", "hc", "hc0"))
    args = parser.parse_args()
    print(json.dumps(run(cache_dir=args.cache_dir, output_path=args.output_path, arms=args.arms), sort_keys=True))


if __name__ == "__main__":
    main()
