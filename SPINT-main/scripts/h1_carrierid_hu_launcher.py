#!/usr/bin/env python3
"""Prepare or launch the H-U fold-0 H32 source-training cell.

GPU launch requires all three of: --execute, --i-have-authorization, and
ROOT_GO_AND_GPU_AUTHORIZED=1.  This script never sets those gates itself.
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h1_carrierid_paired_launcher import _gpu_is_idle, _tmux_session_name
from src.h1_m4_eb_normalized_v2_contract import sha256_file


AUTHORIZED_FLAG = "--i-have-authorization"
AUTHORIZED_ENV = "ROOT_GO_AND_GPU_AUTHORIZED"
EXPECTED_PYTHON = Path("/home/xinyuan/miniconda3/envs/spint/bin/python")
EXPECTED_TORCH_PREFIX = Path("/home/xinyuan/miniconda3/envs/spint")
FAILED_OUTPUT_ROOT_V1 = ROOT / "pilot_artifacts/h1_carrierid_hu/gpu_runs/h32_fold0_hu_v1"
RECOMMENDED_FRESH_OUTPUT_ROOT = ROOT / "pilot_artifacts/h1_carrierid_hu/gpu_runs/h32_fold0_hu_v2_envfix"


def isolated_environment_command(*, command: list[str], gpu: str) -> list[str]:
    """Explicit environment boundary that survives creation of a tmux server."""

    return [
        "env",
        "PYTHONNOUSERSITE=1",
        "PYTHONPATH=",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        *command,
    ]


def package_identity_probe_command(*, python_bin: Path, gpu: str) -> list[str]:
    probe = (
        "import json,site,sys,torch;"
        "ok=torch.cuda.is_available();"
        "print(json.dumps({'sys_executable':sys.executable,'torch_file':torch.__file__,"
        "'torch_version':torch.__version__,'cuda_available':ok,"
        "'cuda_device_count':torch.cuda.device_count(),"
        "'cuda_device_name':torch.cuda.get_device_name(0) if ok else None,"
        "'enable_user_site':site.ENABLE_USER_SITE},sort_keys=True))"
    )
    return isolated_environment_command(command=[str(python_bin), "-c", probe], gpu=gpu)


def validate_package_identity(identity: dict[str, Any], *, python_bin: Path) -> None:
    expected_python = str(python_bin)
    torch_file = Path(str(identity.get("torch_file", ""))).resolve()
    reported_python = str(identity.get("sys_executable", ""))
    if reported_python != expected_python and not (
        Path(reported_python).is_file() and os.path.samefile(reported_python, python_bin)
    ):
        raise RuntimeError("tmux child sys.executable escaped the requested conda environment")
    if EXPECTED_TORCH_PREFIX not in (torch_file, *torch_file.parents):
        raise RuntimeError("tmux child torch escaped the requested conda environment")
    if identity.get("enable_user_site") is not False:
        raise RuntimeError("tmux child unexpectedly enabled the user site")
    if not isinstance(identity.get("torch_version"), str) or not identity["torch_version"]:
        raise RuntimeError("tmux child did not report a torch version")


def run_tmux_package_identity_preflight(
    *, python_bin: Path, gpu: str, work_dir: Path, timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Run a no-data/no-training package/device probe in a real tmux child."""

    if shutil.which("tmux") is None:
        raise RuntimeError("H-U environment preflight requires tmux")
    work = work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    output = work / "tmux_child_package_identity.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite tmux identity probe {output}")
    session = _tmux_session_name(work, "hu_env_probe")
    probe = package_identity_probe_command(python_bin=python_bin, gpu=gpu)
    wrapped = f"{shlex.join(probe)} > {shlex.quote(str(output))}"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, wrapped], check=True)
    import time
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if output.is_file() and output.stat().st_size:
            break
        time.sleep(0.05)
    else:
        subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
        raise RuntimeError("timed out waiting for tmux child environment probe")
    identity = json.loads(output.read_text(encoding="utf-8"))
    validate_package_identity(identity, python_bin=python_bin)
    identity.update({
        "schema": "h1_carrierid_hu_tmux_child_package_identity_v1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": "",
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "probe_only_no_training_or_data": True,
    })
    output.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return identity


def build_command(*, output_root: Path, python_bin: Path) -> list[str]:
    run_dir = output_root / "hu"
    return [
        str(python_bin),
        str(ROOT / "src/train.py"),
        "experiment=h1_carrierid_hu",
        f"hydra.run.dir={run_dir}",
        f"paths.root_dir={ROOT}",
        f"paths.work_dir={ROOT}",
        f"paths.data_dir={ROOT / 'data'}",
        f"pilot.shared_cache_dir={output_root / 'shared_source_cache' / 'hu'}",
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


def prepare_or_launch(
    *,
    output_root: Path,
    python_bin: Path,
    execute: bool = False,
    i_have_authorization: bool = False,
    gpu: str = "0",
) -> dict[str, Any]:
    python = python_bin.expanduser().absolute()
    output = output_root.resolve()
    command = build_command(output_root=output, python_bin=python)
    isolated_command = isolated_environment_command(command=command, gpu=str(gpu))
    env_ok = os.environ.get(AUTHORIZED_ENV) == "1"
    authorized = bool(execute) and bool(i_have_authorization) and env_ok
    plan = {
        "schema": "h1_carrierid_h32_hu_launcher_plan_v1",
        "mode": "execute" if execute else "prepare_print_only",
        "launch_authorized": authorized,
        "output_root": str(output),
        "command": shlex.join(command),
        "tmux_effective_command": shlex.join(isolated_command),
        "effective_environment": {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "python_bin": str(python),
            "python_executable_resolved": str(python.resolve()),
            "expected_torch_prefix": str(EXPECTED_TORCH_PREFIX),
        },
        "package_identity_preflight_required_before_gpu_launch": True,
        "failed_output_root_preserved": str(FAILED_OUTPUT_ROOT_V1),
        "recommended_fresh_output_root": str(RECOMMENDED_FRESH_OUTPUT_ROOT),
        "gpu_assignment": {"hu": str(gpu)},
        "cells": 1,
        "expected_wall_clock": {
            "source": "H-C full arm log h32_fold0_v1/full: 2026-08-07 16:20:26 to 19:22:32",
            "hours_approx": 3.03,
            "note": "H-U uses the same H32 consumer, 50 epochs, seed 42, fold-0; expect ~3 hours plus a short terminal eval",
        },
        "fixed_contract": {
            "arms": ["H-U label-free 4-D descriptor"],
            "primary_comparators": ["H-C", "H-C0"],
            "secondary_system_level_only": ["H-S", "H-LS"],
            "fold_date": "19250101",
            "seed": 42,
            "epochs": 50,
            "precision": "32-true",
            "optimizer": "Adam",
            "lr": 5.0e-5,
            "carrier_dim": 4,
            "carrier_hidden_dim": 32,
            "window_size": 700,
            "num_covariates": 7,
            "model_dim": 1024,
            "num_neurons": 176,
            "target_data_opened": False,
            "target_optimizer_or_backward_steps": 0,
        },
        "authorization_gates": {
            "execute_flag": bool(execute),
            "i_have_authorization_flag": bool(i_have_authorization),
            "ROOT_GO_AND_GPU_AUTHORIZED": env_ok,
        },
        "preregistration_v2": str(ROOT / "pilot_artifacts/h1_carrierid_hu/H1_CARRIERID_HU_PREREGISTRATION_v2.json"),
        "launched": False,
        "source_files": {
            relative: sha256_file(ROOT / relative)
            for relative in (
                "src/data/h1_carrierid_hu_features.py",
                "src/data/h1_carrierid_hu.py",
                "src/models/h1_carrierid_hu_module.py",
                "configs/experiment/h1_carrierid_hu.yaml",
                "scripts/h1_carrierid_hu_launcher.py",
            )
        },
    }
    if not execute:
        return plan
    if not authorized:
        raise RuntimeError(
            "H-U GPU launch is gated closed.  --execute requires both "
            f"{AUTHORIZED_FLAG} and {AUTHORIZED_ENV}=1.  This launcher never "
            "sets those gates.  GPU spend is the owner's decision."
        )
    if shutil.which("tmux") is None:
        raise RuntimeError("H-U GPU execution requires tmux")
    if output.exists():
        raise FileExistsError(f"H-U refuses to reuse an existing output root: {output}")
    if not _gpu_is_idle(str(gpu)):
        raise RuntimeError(f"H-U refuses to disturb a busy GPU: {gpu}")
    identity = run_tmux_package_identity_preflight(
        python_bin=python, gpu=str(gpu), work_dir=output.parent / f".{output.name}.env_preflight"
    )
    if identity.get("cuda_available") is not True or int(identity.get("cuda_device_count", 0)) != 1:
        raise RuntimeError("H-U tmux child package identity passed but requested CUDA device is unavailable")
    session = _tmux_session_name(output, "hu")
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise RuntimeError(f"tmux session already exists: {session}")
    output.mkdir(parents=True)
    run = output / "hu"
    run.mkdir()
    log = run / "launcher.stdout.log"
    wrapped = f"exec {shlex.join(isolated_command)} > {shlex.quote(str(log))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, wrapped], check=True)
    plan["launched"] = True
    plan["effective_package_identity"] = identity
    plan["launches"] = {"hu": {"tmux_session": session, "gpu": str(gpu), "log": str(log)}}
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RECOMMENDED_FRESH_OUTPUT_ROOT,
    )
    parser.add_argument("--python-bin", type=Path, default=Path("/home/xinyuan/miniconda3/envs/spint/bin/python"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(AUTHORIZED_FLAG, dest="i_have_authorization", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_or_launch(
                output_root=args.output_root,
                python_bin=args.python_bin,
                execute=args.execute,
                i_have_authorization=args.i_have_authorization,
                gpu=args.gpu,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
