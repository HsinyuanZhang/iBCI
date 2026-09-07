#!/usr/bin/env python3
"""Prepare or launch the paired H1 CarrierID h=32 source-training arms.

The launcher is deliberately asymmetric with a conventional experiment sweep:
there are exactly two pre-registered arms, H-C and H-C0, one seed, one source
schedule and one fixed terminal epoch.  It creates an independent tmux server
session per GPU arm only after an immutable source-only preflight and an exact
source-closure hash recheck.  It never opens target data itself.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    assert_immutable_receipt,
    sha256_file,
)
from scripts.h1_carrierid_preflight import PREFLIGHT_SCHEMA, PREFLIGHT_STATUS


def _verify_source_closure(project_root: Path, expected: Mapping[str, str]) -> dict[str, Any]:
    if not expected:
        raise ValueError("CarrierID source closure is empty")
    verified: dict[str, str] = {}
    for relative, digest in sorted(expected.items()):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"CarrierID source closure path escapes project root: {relative}")
        candidate = (project_root / path).resolve()
        try:
            candidate.relative_to(project_root)
        except ValueError as exc:
            raise ValueError(f"CarrierID source closure path escapes project root: {relative}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        observed = sha256_file(candidate)
        if observed != digest:
            raise ValueError(f"CarrierID source closure SHA-256 drift at {relative}")
        verified[relative] = observed
    return {"verified": True, "file_count": len(verified), "sha256": verified}


def build_commands(*, project_root: Path, output_root: Path, python_bin: Path) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    for arm in ("full", "zero"):
        run_dir = output_root / arm
        commands[arm] = [
            str(python_bin),
            str(project_root / "src/train.py"),
            f"experiment=h1_carrierid_{arm}",
            f"hydra.run.dir={run_dir}",
            f"paths.root_dir={project_root}",
            f"paths.work_dir={project_root}",
            f"paths.data_dir={project_root / 'data'}",
            f"pilot.shared_cache_dir={output_root / 'shared_source_cache' / arm}",
            "trainer.accelerator=gpu",
            "trainer.devices=1",
            "trainer.max_epochs=50",
            "trainer.min_epochs=50",
            "trainer.precision=32-true",
            "model.optimizer.lr=5e-5",
            "seed=42",
            "test=false",
            "ckpt_path=null",
        ]
    return commands


def _tmux_session_name(output_root: Path, arm: str) -> str:
    # The output-root basename makes the session names collision-resistant
    # while retaining the arm/scope in ``tmux ls`` for human monitoring.
    safe = "".join(ch if ch.isalnum() else "_" for ch in output_root.name)
    return f"h1_carrierid_{safe}_{arm}"


def _gpu_is_idle(gpu: str) -> bool:
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader", "-i", str(gpu)],
        check=False, capture_output=True, text=True,
    )
    if query.returncode != 0:
        raise RuntimeError(f"cannot inspect GPU {gpu} with nvidia-smi")
    return not any(line.strip() for line in query.stdout.splitlines())


def prepare_or_launch(
    *,
    project_root: str | Path,
    output_root: str | Path,
    python_bin: str | Path,
    preflight_receipt: str | Path,
    execute: bool = False,
    full_gpu: str = "0",
    zero_gpu: str = "1",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output_root).resolve()
    python = Path(python_bin).resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(f"CarrierID Python executable unavailable: {python}")
    receipt_path = Path(preflight_receipt).resolve()
    preflight = assert_immutable_receipt(receipt_path, PREFLIGHT_STATUS)
    if preflight.get("schema") != PREFLIGHT_SCHEMA or preflight.get("launch", {}).get("launch_authorized") is not False:
        raise NormalizedV2ContractError("CarrierID requires a nonlaunch source-only preflight receipt")
    closure = _verify_source_closure(root, preflight.get("source_sha256", {}))
    commands = build_commands(project_root=root, output_root=output, python_bin=python)
    result: dict[str, Any] = {
        "schema": "h1_carrierid_h32_paired_launcher_plan_v1",
        "mode": "execute" if execute else "prepare_print_only",
        "preflight": {"path": str(receipt_path), "sha256": sha256_file(receipt_path), "status": preflight["status"]},
        "source_closure": closure,
        "output_root": str(output),
        "commands": {arm: shlex.join(command) for arm, command in commands.items()},
        "gpu_assignment": {"full": str(full_gpu), "zero": str(zero_gpu)},
        "tmux_sessions": {arm: _tmux_session_name(output, arm) for arm in commands},
        "fixed_contract": {
            "arms": ["H-C full", "H-C0 literal zero at model boundary"],
            "seed": 42, "epochs": 50, "precision": "32-true", "optimizer": "Adam", "lr": 5.0e-5,
            "target_data_opened": False, "target_optimizer_or_backward_steps": 0,
        },
        "launched": False,
    }
    if not execute:
        return result
    if shutil.which("tmux") is None:
        raise RuntimeError("CarrierID GPU execution requires tmux for independent arm monitoring")
    if output.exists() and any((output / arm).exists() for arm in commands):
        raise FileExistsError("CarrierID refuses to reuse an existing full/zero run directory")
    idle = {"full": _gpu_is_idle(str(full_gpu)), "zero": _gpu_is_idle(str(zero_gpu))}
    if not all(idle.values()):
        busy = [arm for arm, available in idle.items() if not available]
        raise RuntimeError(f"CarrierID refuses to disturb nonidle GPU arm(s): {busy}")
    for arm in commands:
        session = _tmux_session_name(output, arm)
        if subprocess.run(["tmux", "has-session", "-t", session], check=False, capture_output=True).returncode == 0:
            raise RuntimeError(f"CarrierID tmux session already exists: {session}")
    output.mkdir(parents=True, exist_ok=False)
    launched: dict[str, Any] = {}
    try:
        for arm, gpu in (("full", str(full_gpu)), ("zero", str(zero_gpu))):
            run_dir = output / arm
            run_dir.mkdir(parents=True, exist_ok=False)
            log = run_dir / "launcher.stdout.log"
            command = f"exec env CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} {shlex.join(commands[arm])} > {shlex.quote(str(log))} 2>&1"
            session = _tmux_session_name(output, arm)
            subprocess.run(["tmux", "new-session", "-d", "-s", session, command], check=True)
            pane = subprocess.run(
                ["tmux", "list-panes", "-t", session, "-F", "#{pane_pid}"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            launched[arm] = {"tmux_session": session, "pane_pid": int(pane), "gpu": gpu, "log": str(log)}
    except Exception:
        for arm in launched:
            subprocess.run(["tmux", "kill-session", "-t", launched[arm]["tmux_session"]], check=False)
        raise
    result["launched"] = True
    result["launches"] = launched
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--full-gpu", default="0")
    parser.add_argument("--zero-gpu", default="1")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_or_launch(
        project_root=args.project_root, output_root=args.output_root, python_bin=args.python_bin,
        preflight_receipt=args.preflight_receipt, execute=args.execute, full_gpu=args.full_gpu, zero_gpu=args.zero_gpu,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
