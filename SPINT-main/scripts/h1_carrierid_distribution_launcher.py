#!/usr/bin/env python3
"""Print, but never execute, the matched D-S4/D-Q4 H1 training commands.

The absence of an execute switch is intentional.  A later reviewed operator
must decide whether the immutable source-only preflight warrants a GPU run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.h1_m4_eb_normalized_v2_contract import assert_immutable_receipt, sha256_file
from scripts.h1_carrierid_distribution_preflight import PREFLIGHT_SCHEMA, PREFLIGHT_STATUS


def _verify_source_closure(expected: dict[str, str]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, digest in expected.items():
        path = (ROOT / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(f"fresh distribution source closure drift at {relative}")
        observed[relative] = actual
    return observed


def commands(*, output_root: Path, source_cache_dir: Path, python: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for arm in ("s4", "q4"):
        result[arm] = [
            str(python), str(ROOT / "src/train.py"), f"experiment=h1_carrierid_distribution_{arm}",
            f"hydra.run.dir={output_root / arm}", f"paths.root_dir={ROOT}", f"paths.work_dir={ROOT}",
            f"paths.data_dir={ROOT / 'data'}", f"pilot.shared_cache_dir={source_cache_dir}",
            "trainer.accelerator=gpu", "trainer.devices=1", "trainer.max_epochs=50", "trainer.min_epochs=50",
            "trainer.precision=32-true", "model.optimizer.lr=5e-5", "seed=42", "test=false", "ckpt_path=null",
        ]
    return result


def prepare_plan(*, output_root: Path, preflight_receipt: Path, python_bin: Path,
                 s4_gpu: str = "0", q4_gpu: str = "1") -> dict[str, Any]:
    receipt_path = preflight_receipt.resolve()
    python = python_bin.resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(python)
    receipt = assert_immutable_receipt(receipt_path, PREFLIGHT_STATUS)
    if receipt.get("schema") != PREFLIGHT_SCHEMA or receipt.get("launch", {}).get("launch_authorized") is not False:
        raise ValueError("fresh distribution launcher requires immutable nonlaunch CPU preflight")
    source_cache = Path(receipt["source_cache_dir"]).resolve()
    if not source_cache.is_dir():
        raise FileNotFoundError("immutable fresh distribution source cache missing")
    built = commands(output_root=output_root.resolve(), source_cache_dir=source_cache, python=python)
    return {
        "schema": "h1_carrierid_h32_fresh_distribution_plan_only_launcher_v1",
        "mode": "prepare_print_only_no_execution_implemented",
        "launch_authorized": False,
        "launched": False,
        "preflight": {"path": str(receipt_path), "sha256": sha256_file(receipt_path), "status": receipt["status"]},
        "source_closure": _verify_source_closure(dict(receipt["source_sha256"])),
        "source_cache_dir": str(source_cache),
        "gpu_assignment_for_future_manual_review_only": {"d_s4": str(s4_gpu), "d_q4": str(q4_gpu)},
        "fixed_contract": {
            "architecture": "H1CarrierID h=32,d=4", "seed": 42, "epochs": 50,
            "optimizer": "Adam", "learning_rate": 5e-5, "fixed_terminal_epoch": 49,
            "same_initialization_batch_order_source_sessions_query_windows": True,
            "s4": "standard support carrier t..t+3",
            "q4": "query-local/oracle-distribution carrier t+4..t+7; LEAKAGE_DIAGNOSTIC_ONLY",
            "target_or_evalai_in_this_plan": False,
        },
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
    print(json.dumps(prepare_plan(output_root=args.output_root, preflight_receipt=args.preflight_receipt,
                                  python_bin=args.python_bin, s4_gpu=args.s4_gpu, q4_gpu=args.q4_gpu),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
