#!/usr/bin/env python3
"""Print, but never execute, H1 exposure-matched D-S4e/D-Q4e fit commands.

The launcher is deliberately data-free: it reads only the immutable source CPU
preflight receipt, source code hashes, and immutable source cache manifest.
There is no execute switch, no subprocess import, no DataModule construction,
and no target/evaluation interface in this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.h1_m4_eb_normalized_v2_contract import assert_immutable_receipt, sha256_file
from scripts.h1_carrierid_distribution_exposure_preflight import PREFLIGHT_SCHEMA, PREFLIGHT_STATUS


TERMINAL_SCHEMA = "h1_carrierid_h32_fresh_distribution_exposure_terminal_checkpoint_v1"


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _verify_source_closure(expected: dict[str, str]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, digest in sorted(expected.items()):
        path = (ROOT / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(f"exposure source closure drift at {relative}")
        observed[relative] = actual
    return observed


def _verify_cache(receipt: dict[str, Any]) -> dict[str, Any]:
    cache_dir = Path(str(receipt["source_cache_dir"])).resolve()
    manifest_path = cache_dir / "h1_fresh_distribution_exposure_pair_cache.manifest.json"
    cache_path = cache_dir / "h1_fresh_distribution_exposure_pair_cache.npz"
    for path in (manifest_path, cache_path):
        if not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise FileNotFoundError(f"immutable exposure source cache missing or mutable: {path}")
    manifest = _json_object(manifest_path)
    binding = receipt["shared_source_binding"]
    if manifest.get("manifest_sha256") != binding["carrier_cache_sha256"]:
        raise ValueError("exposure source cache manifest does not bind the preflight receipt")
    plan = manifest.get("sample_plan")
    if plan != binding["exposure_sample_plan"]:
        raise ValueError("exposure cache sample plan does not bind the preflight receipt")
    if manifest.get("source_schedule_sha256") != binding["source_schedule_sha256"]:
        raise ValueError("exposure cache schedule differs from immutable preflight")
    return {
        "directory": str(cache_dir),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "array_path": str(cache_path),
        "array_file_sha256": sha256_file(cache_path),
        "manifest_sha256": manifest["manifest_sha256"],
    }


def commands(*, output_root: Path, source_cache_dir: Path, python: Path) -> dict[str, list[str]]:
    """Return the exact two manual fit commands; never execute them."""

    result: dict[str, list[str]] = {}
    for arm, suffix in (("s4", "s4e"), ("q4", "q4e")):
        output_dir = output_root / suffix
        if output_dir.exists():
            raise FileExistsError(f"refusing a non-fresh exposure fit output root: {output_dir}")
        result[suffix] = [
            str(python), str(ROOT / "src/train.py"),
            f"experiment=h1_carrierid_distribution_exposure_{arm}",
            f"hydra.run.dir={output_dir}",
            f"paths.root_dir={ROOT}", f"paths.work_dir={ROOT}", f"paths.data_dir={ROOT / 'data'}",
            f"pilot.shared_cache_dir={source_cache_dir}",
            "trainer.accelerator=gpu", "trainer.devices=1", "trainer.max_epochs=50", "trainer.min_epochs=50",
            "trainer.precision=32-true", "model.optimizer.lr=5e-5", "seed=42", "test=false", "ckpt_path=null",
        ]
    return result


def terminal_checker_requirements() -> dict[str, Any]:
    """Static requirements a later source-only checkpoint checker must enforce."""

    return {
        "checkpoint_paths": {
            "d_s4e": "s4e/checkpoints/fixed_epoch50/epoch_049.ckpt",
            "d_q4e": "q4e/checkpoints/fixed_epoch50/epoch_049.ckpt",
        },
        "each_checkpoint": {
            "metadata_schema": TERMINAL_SCHEMA,
            "checkpoint_epoch_zero_based": 49,
            "epochs_completed": 50,
            "selected_by": "fixed_terminal_epoch_no_selection",
            "arm_values": {"d_s4e": "D-S4E", "d_q4e": "D-Q4E"},
            "must_bind": [
                "config_sha256", "source_manifest_sha256", "source_cache_sha256",
                "normalized_cache_sha256", "normalizer_sha256", "initial_state_sha256",
                "source_schedule_sha256", "common_query_samples_sha256",
                "identity_schedule_sha256", "batch_order_sha256", "exposure_sample_plan",
            ],
        },
        "paired_equal": [
            "initial_state_sha256", "source_schedule_sha256", "common_query_samples_sha256",
            "identity_schedule_sha256", "batch_order_sha256", "source_cache_sha256",
            "normalized_cache_sha256", "normalizer_sha256", "exposure_sample_plan",
        ],
        "paired_different_only": ["arm", "carrier_mode", "effective_source_carriers_sha256"],
        "forbidden": [
            "checkpoint selection by validation/target metric",
            "target/minival/formal/EvalAI load before source terminal audit",
            "optimizer or backward during later target forward-only evaluation",
        ],
        "target_gate": "only a future separate evaluator may open target, after both source terminal checkpoints pass this checker",
    }


def prepare_plan(*, output_root: Path, preflight_receipt: Path, python_bin: Path,
                 s4_gpu: str = "0", q4_gpu: str = "1") -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise ValueError("plan-only launcher requires CUDA_VISIBLE_DEVICES unset")
    receipt_path = preflight_receipt.resolve()
    python = python_bin.resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(python)
    receipt = assert_immutable_receipt(receipt_path, PREFLIGHT_STATUS)
    if receipt.get("schema") != PREFLIGHT_SCHEMA or receipt.get("launch", {}).get("launch_authorized") is not False:
        raise ValueError("exposure launcher requires the immutable nonlaunch CPU preflight")
    binding = receipt["shared_source_binding"]
    if (binding.get("schedule_count"), binding.get("sample_count"), binding.get("batches_per_epoch"),
            binding.get("scheduled_samples_per_epoch")) != (72, 115_520, 3_610, 115_520):
        raise ValueError("immutable preflight does not have the H-C matched exposure contract")
    cache = _verify_cache(receipt)
    built = commands(output_root=output_root.resolve(), source_cache_dir=Path(cache["directory"]), python=python)
    return {
        "schema": "h1_carrierid_h32_fresh_distribution_exposure_plan_only_launcher_v1",
        "mode": "prepare_print_only_no_execution_implemented",
        "launch_authorized": False,
        "launched": False,
        "data_opened": False,
        "target_opened": False,
        "cuda_constructed_or_launched": False,
        "preflight": {
            "path": str(receipt_path), "sha256": sha256_file(receipt_path), "status": receipt["status"],
        },
        "source_closure": _verify_source_closure(dict(receipt["source_sha256"])),
        "launcher_source_sha256": sha256_file(Path(__file__).resolve()),
        "source_cache": cache,
        "future_manual_gpu_assignment_only": {"d_s4e": str(s4_gpu), "d_q4e": str(q4_gpu)},
        "fixed_contract": {
            "architecture": "H1CarrierID h=32,d=4", "seed": 42, "epochs": 50,
            "optimizer": "Adam", "learning_rate": 5.0e-5, "fixed_terminal_epoch": 49,
            "schedules": 72, "samples_per_epoch": 115_520, "batches_per_epoch": 3_610,
            "same_initialization_batch_order_source_sessions_query_windows": True,
            "s4e": "standard support carrier t..t+3",
            "q4e": "query-local t+4..t+7 leakage diagnostic only",
            "target_or_evalai_in_this_plan": False,
        },
        "terminal_checkpoint_checker_requirements": terminal_checker_requirements(),
        "commands_not_executed": {arm: shlex.join(command) for arm, command in built.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--preflight-receipt", required=True, type=Path)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--s4-gpu", default="0")
    parser.add_argument("--q4-gpu", default="1")
    args = parser.parse_args()
    print(json.dumps(
        prepare_plan(output_root=args.output_root, preflight_receipt=args.preflight_receipt,
                     python_bin=args.python_bin, s4_gpu=args.s4_gpu, q4_gpu=args.q4_gpu),
        indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()
