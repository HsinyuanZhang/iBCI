#!/usr/bin/env python3
"""Prepare/launch the separately-trained source-only H1 CarrierID RS/LS arms."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.h1_m4_eb_normalized_v2_contract import assert_immutable_receipt, sha256_file
from scripts.h1_carrierid_paired_launcher import _gpu_is_idle, _tmux_session_name, _verify_source_closure
from scripts.h1_carrierid_shuffle_preflight import PREFLIGHT_SCHEMA, PREFLIGHT_STATUS


def commands(output_root: Path, python: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for arm in ("rs", "ls"):
        result[arm] = [str(python), str(ROOT / "src/train.py"), f"experiment=h1_carrierid_{arm}",
            f"hydra.run.dir={output_root / arm}", f"paths.root_dir={ROOT}", f"paths.work_dir={ROOT}",
            f"paths.data_dir={ROOT / 'data'}", f"pilot.shared_cache_dir={output_root / 'shared_source_cache' / arm}",
            "trainer.accelerator=gpu", "trainer.devices=1", "trainer.max_epochs=50", "trainer.min_epochs=50",
            "trainer.precision=32-true", "model.optimizer.lr=5e-5", "seed=42", "test=false", "ckpt_path=null"]
    return result


def prepare_or_launch(*, output_root: Path, preflight_receipt: Path, python_bin: Path,
                      execute: bool = False, rs_gpu: str = "0", ls_gpu: str = "1") -> dict[str, Any]:
    output = output_root.resolve(); receipt_path = preflight_receipt.resolve(); python = python_bin.resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(python)
    receipt = assert_immutable_receipt(receipt_path, PREFLIGHT_STATUS)
    if receipt.get("schema") != PREFLIGHT_SCHEMA or receipt.get("launch", {}).get("launch_authorized") is not False:
        raise ValueError("H1 RS/LS launcher requires an immutable nonlaunch preflight")
    closure = _verify_source_closure(ROOT, receipt.get("source_sha256", {}))
    plan = {"schema": "h1_carrierid_h32_rs_ls_paired_launcher_plan_v1", "mode": "execute" if execute else "prepare_print_only",
        "preflight": {"path": str(receipt_path), "sha256": sha256_file(receipt_path), "status": receipt["status"]},
        "source_closure": closure, "output_root": str(output), "commands": {}, "gpu_assignment": {"rs": str(rs_gpu), "ls": str(ls_gpu)},
        "fixed_contract": {"arms": ["H-RS source row-shuffle", "H-LS source label-shuffle"], "seed": 42,
            "support_trials": 4, "epochs": 50, "precision": "32-true", "optimizer": "Adam", "lr": 5e-5,
            "target_data_opened": False, "additive_eb_residual_reused": False}, "launched": False}
    built = commands(output, python); plan["commands"] = {arm: shlex.join(command) for arm, command in built.items()}
    if not execute:
        return plan
    if shutil.which("tmux") is None or output.exists():
        raise RuntimeError("H1 RS/LS execution requires tmux and a fresh output root")
    idle = {"rs": _gpu_is_idle(str(rs_gpu)), "ls": _gpu_is_idle(str(ls_gpu))}
    if not all(idle.values()):
        raise RuntimeError(f"H1 RS/LS refuses busy GPU(s): {[arm for arm, ready in idle.items() if not ready]}")
    sessions = {arm: _tmux_session_name(output, arm) for arm in built}
    for session in sessions.values():
        if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
            raise RuntimeError(f"tmux session exists: {session}")
    output.mkdir(parents=True)
    launched: dict[str, Any] = {}
    try:
        for arm, gpu in (("rs", str(rs_gpu)), ("ls", str(ls_gpu))):
            run = output / arm; run.mkdir(); log = run / "launcher.stdout.log"; session = sessions[arm]
            command = f"exec env CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} {shlex.join(built[arm])} > {shlex.quote(str(log))} 2>&1"
            subprocess.run(["tmux", "new-session", "-d", "-s", session, command], check=True)
            launched[arm] = {"tmux_session": session, "gpu": gpu, "log": str(log)}
    except Exception:
        for item in launched.values():
            subprocess.run(["tmux", "kill-session", "-t", item["tmux_session"]], check=False)
        raise
    plan["launched"] = True; plan["launches"] = launched
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--preflight-receipt", required=True, type=Path)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--rs-gpu", default="0"); parser.add_argument("--ls-gpu", default="1")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_or_launch(output_root=args.output_root, preflight_receipt=args.preflight_receipt,
        python_bin=args.python_bin, execute=args.execute, rs_gpu=args.rs_gpu, ls_gpu=args.ls_gpu), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
