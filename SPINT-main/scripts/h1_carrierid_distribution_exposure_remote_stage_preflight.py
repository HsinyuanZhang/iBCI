#!/usr/bin/env python3
"""No-launch, no-data staging receipt for remote 5070Ti H1 D-Q4e.

This script is designed to execute inside an isolated staging copy of
``SPINT-main``.  It only reads immutable receipts/caches and composes Hydra
configuration.  It neither builds a DataModule nor opens an NWB, target,
minival, formal-heldout, or EvalAI resource; it also refuses an initialized
CUDA context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any

import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    assert_immutable_receipt,
    sha256_file,
    write_immutable_json,
)
from scripts.h1_carrierid_distribution_exposure_preflight import PREFLIGHT_SCHEMA, PREFLIGHT_STATUS


STAGE_SCHEMA = "h1_carrierid_h32_fresh_distribution_exposure_remote_5070ti_stage_preflight_v1"
STAGE_STATUS = "PASS_H1_CARRIERID_H32_FRESH_D_Q4E_REMOTE_5070TI_STAGE_COMPOSITION_CACHE_CLOSURE_NONLAUNCH"


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_source_closure(expected: dict[str, str]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, digest in sorted(expected.items()):
        path = (ROOT / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != digest:
            raise NormalizedV2ContractError(f"remote exposure staging closure drift at {relative}")
        observed[relative] = actual
    return observed


def _verify_immutable_cache(*, cache_dir: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    manifest_path = cache_dir / "h1_fresh_distribution_exposure_pair_cache.manifest.json"
    array_path = cache_dir / "h1_fresh_distribution_exposure_pair_cache.npz"
    for path in (manifest_path, array_path):
        if not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise FileNotFoundError(f"required immutable exposure cache missing/mutable: {path}")
    manifest = _json_object(manifest_path)
    binding = receipt["shared_source_binding"]
    if manifest.get("manifest_sha256") != binding.get("carrier_cache_sha256"):
        raise NormalizedV2ContractError("remote cache manifest differs from local immutable preflight")
    if manifest.get("sample_plan") != binding.get("exposure_sample_plan"):
        raise NormalizedV2ContractError("remote cache sample plan differs from local immutable preflight")
    if manifest.get("source_schedule_sha256") != binding.get("source_schedule_sha256"):
        raise NormalizedV2ContractError("remote cache schedule differs from local immutable preflight")
    return {
        "directory": str(cache_dir),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "array_path": str(array_path),
        "array_file_sha256": sha256_file(array_path),
        "manifest_sha256": manifest["manifest_sha256"],
    }


def _compose_q4e(*, data_root: Path, raw_receipt: Path, eb_receipt: Path, cache_dir: Path):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=h1_carrierid_distribution_exposure_q4",
                f"paths.root_dir={ROOT}",
                f"paths.work_dir={ROOT}",
                f"paths.data_dir={data_root}",
                f"data.raw_receipt_path={raw_receipt}",
                f"data.eb_receipt_path={eb_receipt}",
                f"pilot.shared_cache_dir={cache_dir}",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
                "test=false",
            ],
        )
    expected = {
        "data_target": "src.data.h1_carrierid_distribution_exposure.H1CarrierIdDistributionExposureDataModule",
        "model_target": "src.models.h1_carrierid_distribution_exposure_module.H1CarrierIdDistributionExposureLitModule",
        "arm": "q4",
        "batch_size": 32,
        "samples_per_epoch": 115_520,
        "epochs": 50,
        "lr": 5.0e-5,
    }
    observed = {
        "data_target": str(cfg.data._target_),
        "model_target": str(cfg.model._target_),
        "arm": str(cfg.pilot.arm),
        "batch_size": int(cfg.data.batch_size),
        "samples_per_epoch": int(cfg.data.samples_per_epoch),
        "epochs": int(cfg.trainer.max_epochs),
        "min_epochs": int(cfg.trainer.min_epochs),
        "seed": int(cfg.seed),
        "lr": float(cfg.model.optimizer.lr),
        "test": bool(cfg.test),
        "data_dir": str(cfg.data.data_dir),
        "raw_receipt_path": str(cfg.data.raw_receipt_path),
        "eb_receipt_path": str(cfg.data.eb_receipt_path),
        "cache_dir": str(cfg.data.cache_dir),
    }
    if any(observed[key] != value for key, value in expected.items()):
        raise NormalizedV2ContractError(f"D-Q4e remote Hydra composition drift: {observed}")
    if observed["min_epochs"] != 50 or observed["seed"] != 42 or observed["test"]:
        raise NormalizedV2ContractError("D-Q4e remote fit epoch/seed/test contract drift")
    return observed


def q4e_command(*, python: Path, data_root: Path, raw_receipt: Path, eb_receipt: Path,
                cache_dir: Path, output_root: Path) -> list[str]:
    output_dir = output_root / "q4e"
    if output_dir.exists():
        raise FileExistsError(f"remote D-Q4e output must be fresh: {output_dir}")
    return [
        str(python), str(ROOT / "src/train.py"), "experiment=h1_carrierid_distribution_exposure_q4",
        f"hydra.run.dir={output_dir}", f"paths.root_dir={ROOT}", f"paths.work_dir={ROOT}",
        f"paths.data_dir={data_root}", f"data.raw_receipt_path={raw_receipt}",
        f"data.eb_receipt_path={eb_receipt}", f"pilot.shared_cache_dir={cache_dir}",
        "trainer.accelerator=gpu", "trainer.devices=1", "trainer.max_epochs=50", "trainer.min_epochs=50",
        "trainer.precision=32-true", "model.optimizer.lr=5e-5", "seed=42", "test=false", "ckpt_path=null",
    ]


def run(*, source_preflight: Path, cache_dir: Path, data_root: Path, raw_receipt: Path,
        eb_receipt: Path, output_root: Path, python: Path, output: Path) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "") or torch.cuda.is_initialized():
        raise NormalizedV2ContractError("remote D-Q4e staging preflight requires no visible/initialized CUDA")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(python)
    # The source root is deliberately only a path check.  No file name is
    # enumerated and no NWB is opened during this staging phase.
    if not (data_root / "000954" / "sub-HumanPitt-held-in-calib").is_dir():
        raise FileNotFoundError("remote source-only H1 data root missing")
    for path in (raw_receipt, eb_receipt):
        if not path.is_file():
            raise FileNotFoundError(path)
    preflight_path = source_preflight.resolve()
    preflight = assert_immutable_receipt(preflight_path, PREFLIGHT_STATUS)
    if preflight.get("schema") != PREFLIGHT_SCHEMA:
        raise NormalizedV2ContractError("remote stage needs the exact local exposure source preflight schema")
    closure = _verify_source_closure(dict(preflight["source_sha256"]))
    cache = _verify_immutable_cache(cache_dir=cache_dir.resolve(), receipt=preflight)
    composition = _compose_q4e(
        data_root=data_root.resolve(), raw_receipt=raw_receipt.resolve(),
        eb_receipt=eb_receipt.resolve(), cache_dir=cache_dir.resolve(),
    )
    command = q4e_command(
        python=python.resolve(), data_root=data_root.resolve(), raw_receipt=raw_receipt.resolve(),
        eb_receipt=eb_receipt.resolve(), cache_dir=cache_dir.resolve(), output_root=output_root.resolve(),
    )
    payload = {
        "schema": STAGE_SCHEMA,
        "status": STAGE_STATUS,
        "scope": {
            "source_nwb_opened": 0,
            "target_opened_or_enumerated": False,
            "minival_opened_or_enumerated": False,
            "formal_heldout_opened_or_enumerated": False,
            "evalai_opened_or_enumerated": False,
            "datamodule_constructed": False,
            "trainer_constructed": False,
            "cuda_constructed_or_launched": False,
            "gpu_queued": False,
        },
        "local_exposure_preflight": {
            "path": str(preflight_path), "sha256": sha256_file(preflight_path), "status": preflight["status"],
        },
        "source_closure": closure,
        "stage_preflight_script_sha256": sha256_file(Path(__file__).resolve()),
        "immutable_exposure_cache": cache,
        "absolute_read_only_inputs": {
            "data_root": str(data_root.resolve()),
            "raw_source_receipt": str(raw_receipt.resolve()),
            "eb_source_receipt": str(eb_receipt.resolve()),
            "write_policy": "the future command writes only below output_root/staging; these paths are references only",
        },
        "q4e_cpu_composition": composition,
        "future_q4e_command_not_executed": " ".join(command),
        "future_q4e_output_root": str((output_root.resolve() / "q4e")),
        "terminal_checkpoint_requirement": {
            "path": "q4e/checkpoints/fixed_epoch50/epoch_049.ckpt",
            "schema": "h1_carrierid_h32_fresh_distribution_exposure_terminal_checkpoint_v1",
            "arm": "D-Q4E",
            "fixed_epoch": 49,
            "no_target_before_pair_checker": True,
        },
        "launch_authorized": False,
    }
    path, digest = write_immutable_json(output, payload)
    return {"status": STAGE_STATUS, "receipt_path": str(path), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-preflight", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--raw-receipt", required=True, type=Path)
    parser.add_argument("--eb-receipt", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
