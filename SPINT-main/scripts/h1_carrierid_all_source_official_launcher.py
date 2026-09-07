#!/usr/bin/env python3
"""CPU-only asset and launch preparation for H1 all-source CarrierID.

This file deliberately has no subprocess, Trainer, CUDA, checkpoint, EvalAI,
or target-query execution path.  A later GPU operator must consume the emitted
immutable launch receipt.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

import hydra
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.h1_carrierid_all_source_official import (
    ALL_SOURCE_ASSET_SCHEMA,
    ALL_SOURCE_EPOCHS,
    ALL_SOURCE_SEED,
    H1_HELDIN_SESSIONS,
    load_all_source_assets,
    prepare_all_source_assets,
)
from src.h1_m4_cce_contract import canonical_sha256, sha256_file, write_immutable_json


LAUNCH_SCHEMA = "h1_carrierid_all_public_source_official_launch_receipt_v1"
LAUNCH_STATUS = "PASS_ALL_SOURCE_HC_PREPARED_NOT_LAUNCHED"
EXPERIMENT = "h1_carrierid_all_source_official"


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def compose_candidate(asset_manifest: Path) -> Any:
    with hydra.initialize_config_dir(version_base=None, config_dir=str((ROOT / "configs").resolve())):
        return hydra.compose(
            config_name="train",
            overrides=[f"experiment={EXPERIMENT}", f"official_candidate.asset_manifest_path={asset_manifest.resolve()}"],
        )


def validate_candidate_config(config: Any, *, asset_manifest: Path) -> dict[str, Any]:
    assets = load_all_source_assets(asset_manifest)
    raw = OmegaConf.to_container(config, resolve=False)
    _need(isinstance(raw, Mapping), "composed all-source config is malformed")
    data, model, trainer, callbacks = raw["data"], raw["model"], raw["trainer"], raw["callbacks"]
    _need(raw["train"] is True and raw["test"] is False and raw["ckpt_path"] in (None, "", "null"),
          "candidate must be fresh train-only")
    _need(int(raw["seed"]) == ALL_SOURCE_SEED, "candidate seed drift")
    _need(data["_target_"] == "src.data.h1_carrierid_all_source_official.H1CarrierIdAllSourceDataModule",
          "candidate does not use isolated all-source DataModule")
    _need(model["_target_"] == "src.models.h1_carrierid_all_source_official_module.H1CarrierIdAllSourceLitModule",
          "candidate does not use isolated all-source training wrapper")
    _need(model["net"]["_target_"] == "src.models.components.h1_carrierid_spint.H1CarrierIdSpint",
          "candidate no longer reuses H1CarrierIdSpint")
    _need(str(data["asset_manifest_path"]) == "${official_candidate.asset_manifest_path}",
          "data config lost the receipt-bound asset interpolation")
    _need(str(raw["official_candidate"]["asset_manifest_path"]) == str(asset_manifest.resolve()),
          "candidate config is not bound to the supplied asset manifest")
    _need(int(data["calibration_n_trials"]) == 4 and int(data["window_size"]) == 700 and
          int(data["max_trial_length"]) == 1024 and int(data["fixed_epochs"]) == ALL_SOURCE_EPOCHS,
          "candidate M=4/window/epoch contract drift")
    _need(int(trainer["max_epochs"]) == int(trainer["min_epochs"]) == ALL_SOURCE_EPOCHS,
          "candidate is not fixed to 50 source epochs")
    _need(int(trainer["limit_val_batches"]) == 0 and int(trainer["num_sanity_val_steps"]) == 0,
          "candidate unexpectedly enables validation")
    checkpoint = callbacks.get("fixed_epoch50")
    _need(isinstance(checkpoint, Mapping) and checkpoint.get("monitor") is None and
          int(checkpoint.get("every_n_epochs")) == 50 and checkpoint.get("save_last") is False,
          "candidate checkpoint is not fixed terminal e49")
    official = raw["official_candidate"]
    _need(official["formal_test_labels_opened"] == 0 and official["target_optimizer_steps"] == 0 and
          official["target_backward_steps"] == 0 and official["evalai_submission_authorized"] is False,
          "candidate config violates formal/target/EvalAI boundary")
    return {
        "config_sha256": canonical_sha256(raw),
        "asset_manifest_path": str(assets.manifest_path),
        "asset_manifest_sha256": assets.manifest_sha256,
        "transform_sha256": assets.plan.transform_sha256,
        "normalizer_sha256": assets.normalizer.normalizer_sha256,
        "source_sessions": list(H1_HELDIN_SESSIONS),
        "model_target": model["net"]["_target_"],
        "data_target": data["_target_"],
        "wrapper_target": model["_target_"],
        "fresh_seed": ALL_SOURCE_SEED,
        "epochs": ALL_SOURCE_EPOCHS,
        "checkpoint_epoch_zero_based": 49,
    }


def prepare_launch(*, asset_manifest: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite all-source launch receipt: {output}")
    row = validate_candidate_config(compose_candidate(asset_manifest), asset_manifest=asset_manifest)
    command = [
        sys.executable,
        str(ROOT / "src/train.py"),
        f"experiment={EXPERIMENT}",
        f"official_candidate.asset_manifest_path={asset_manifest.resolve()}",
        "ckpt_path=null",
        "test=false",
    ]
    body = {
        "schema": LAUNCH_SCHEMA,
        "status": LAUNCH_STATUS,
        "mode": "prepare_only_no_subprocess_no_trainer_no_cuda_no_evalai",
        "candidate": row,
        "planned_command": command,
        "scope": {
            "public_held_in_calibration_recordings": 13,
            "minival_recordings_opened": 0,
            "formal_test_labels_opened": 0,
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
            "trainer_constructed_or_launched": False,
            "cuda_constructed_or_launched": False,
            "evalai_submission_authorized": False,
        },
        "code_sha256": {
            "launcher": sha256_file(Path(__file__).resolve()),
            "data": sha256_file(ROOT / "src/data/h1_carrierid_all_source_official.py"),
            "model": sha256_file(ROOT / "src/models/h1_carrierid_all_source_official_module.py"),
            "experiment": sha256_file(ROOT / "configs/experiment/h1_carrierid_all_source_official.yaml"),
            "data_config": sha256_file(ROOT / "configs/data/falcon_h1_carrierid_all_source_official.yaml"),
            "model_config": sha256_file(ROOT / "configs/model/falcon_h1_carrierid_all_source_official.yaml"),
        },
    }
    write_immutable_json(output, body)
    return {**body, "output": str(output.resolve()), "output_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    assets = subparsers.add_parser("prepare-assets")
    assets.add_argument("--data-dir", type=Path, required=True)
    assets.add_argument("--output-dir", type=Path, required=True)
    launch = subparsers.add_parser("prepare-launch")
    launch.add_argument("--asset-manifest", type=Path, required=True)
    launch.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-assets":
        result = prepare_all_source_assets(data_dir=args.data_dir, output_dir=args.output_dir)
    else:
        result = prepare_launch(asset_manifest=args.asset_manifest, output=args.output)
    print(result)


if __name__ == "__main__":
    main()
