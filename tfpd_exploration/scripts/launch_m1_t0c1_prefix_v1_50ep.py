#!/usr/bin/env python3
"""Root-reviewed stage launcher for the 50-epoch M1 T0/C1 lane.

Physical GPU 1 only.  Stages t0 and c1 may share GPU 1 with each other;
smoke, probe, and phase3 still require an exclusive card.  Every pre-flight
check below must pass before the stage driver is imported, because importing
the driver is what pulls in Torch and eventually initializes CUDA.

Usage (the envelope is mandatory, see DESIGN §2.9):

    cd /home/xinyuan/Work_host/SPINT
    CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
    PYTHONPATH=/home/xinyuan/Work_host/SPINT \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/xinyuan/miniconda3/envs/spint/bin/python \
        tfpd_exploration/scripts/launch_m1_t0c1_prefix_v1_50ep.py \
        --stage smoke --i-am-root-reviewer

This launcher never touches physical GPU 0: the only nvidia-smi it runs is
``--id 1``, and the lane's own device profiler (not the frozen one) is likewise
pinned to ``--id ${CUDA_VISIBLE_DEVICES}``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

from tfpd_exploration.src.m1_t0c1_prefix_v1_50ep import plan

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = Path(__file__).resolve()
STAGES = ("smoke", "t0", "c1", "probe", "phase3")

EXPECTED_TORCH = "2.5.1.post303"
EXPECTED_CUDA = "11.8"
EXPECTED_CUDNN = 90300
EXPECTED_DEVICE_NAME = "NVIDIA GeForce RTX 3090"
EXPECTED_TOTAL_MEMORY_BYTES = 25438126080
GPU1_UUID = "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"
GPU0_UUID_REFUSED = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"

REQUIRED_ENV = {
    "CUDA_VISIBLE_DEVICES": "1",
    "PYTHONNOUSERSITE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


class PreflightError(RuntimeError):
    """Refuse to start a stage when the envelope or the card is wrong."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def check_environment() -> dict[str, object]:
    for key, value in REQUIRED_ENV.items():
        actual = os.environ.get(key)
        _require(actual == value, f"env {key}={actual!r}, required {value!r} (see DESIGN 2.9)")
    _require(Path.cwd() == REPO_ROOT, f"cwd must be {REPO_ROOT}, got {Path.cwd()}")
    return {"env": dict(REQUIRED_ENV), "cwd": str(REPO_ROOT)}


def _nvidia_smi(*args: str) -> str:
    """Query physical GPU 1 only.  Never a bare nvidia-smi, never --id 0."""
    _require("--id" in args and args[list(args).index("--id") + 1] == "1",
             "nvidia-smi must be pinned to --id 1")
    return subprocess.run(
        ["nvidia-smi", *args],
        check=True, text=True, capture_output=True, timeout=15,
    ).stdout


def parse_used_memory_mib(raw: str) -> int:
    text = str(raw).strip()
    _require(bool(text) and "N/A" not in text.upper(),
             f"used_memory unparseable: {raw!r}")
    token = text.replace(",", "").split()[0]
    try:
        value = int(token)
    except ValueError as error:
        raise PreflightError(f"used_memory unparseable: {raw!r}") from error
    _require(value >= 0, f"used_memory negative: {raw!r}")
    return value


def parse_compute_apps_csv(stdout: str) -> list[tuple[int, int]]:
    apps: list[tuple[int, int]] = []
    for line in str(stdout).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "," not in stripped:
            raise PreflightError(f"compute-app csv unparseable: {stripped!r}")
        pid_raw, memory_raw = stripped.split(",", 1)
        try:
            pid = int(pid_raw.strip())
        except ValueError as error:
            raise PreflightError(f"compute-app pid unparseable: {stripped!r}") from error
        _require(pid > 0, f"compute-app pid not positive: {pid}")
        apps.append((pid, parse_used_memory_mib(memory_raw)))
    return apps


def argv_from_cmdline_bytes(body: bytes) -> list[str]:
    _require(isinstance(body, (bytes, bytearray)) and bytes(body),
             "empty cmdline")
    argv = [part.decode("utf-8", "surrogateescape")
            for part in bytes(body).split(b"\0") if part]
    _require(bool(argv), "empty cmdline")
    return argv


def read_proc_cmdline(pid: int) -> list[str]:
    path = Path(f"/proc/{int(pid)}/cmdline")
    try:
        body = path.read_bytes()
    except OSError as error:
        raise PreflightError(
            f"compute-app pid {pid} cannot be resolved to a cmdline: {error}"
        ) from error
    if not body:
        raise PreflightError(f"compute-app pid {pid} empty cmdline")
    return argv_from_cmdline_bytes(body)


def _resolve_cmdline(
    pid: int, cmdline_of: Callable[[int], Sequence[str]],
) -> list[str]:
    try:
        argv = list(cmdline_of(int(pid)))
    except PreflightError:
        raise
    except Exception as error:
        raise PreflightError(
            f"compute-app pid {pid} cannot be resolved to a cmdline: {error}"
        ) from error
    if not argv:
        raise PreflightError(f"compute-app pid {pid} empty cmdline")
    return [str(part) for part in argv]


def argv_invokes_launcher(argv: Sequence[str], launcher: Path) -> bool:
    target = Path(launcher).resolve()
    for arg in argv:
        if str(arg).startswith("-"):
            continue
        try:
            resolved = Path(arg).resolve()
        except (OSError, RuntimeError):
            continue
        if resolved == target:
            return True
    return False


def stage_from_argv(argv: Sequence[str]) -> str | None:
    parts = [str(part) for part in argv]
    for index, arg in enumerate(parts):
        if arg == "--stage" and index + 1 < len(parts):
            return parts[index + 1]
        if arg.startswith("--stage="):
            return arg.split("=", 1)[1]
    return None


def classify_compute_app(
    argv: Sequence[str], *, starting_stage: str, launcher: Path,
) -> tuple[str, str | None]:
    """Return (kind, sibling_stage) with kind in {permitted_sibling, same_stage, foreign}."""
    invoked = argv_invokes_launcher(argv, launcher)
    sibling_stage = stage_from_argv(argv) if invoked else None
    if not invoked or sibling_stage not in plan.CONCURRENT_ARM_STAGES:
        return "foreign", sibling_stage
    if starting_stage not in plan.CONCURRENT_ARM_STAGES:
        return "foreign", sibling_stage
    if sibling_stage == starting_stage:
        return "same_stage", sibling_stage
    return "permitted_sibling", sibling_stage


def evaluate_gpu1_concurrency(
    stage: str,
    *,
    compute_apps: Sequence[tuple[int, int]],
    used_memory_mib: int,
    cmdline_of: Callable[[int], Sequence[str]],
    launcher: Path,
) -> dict[str, object]:
    """Allowlist the other 50ep arm; fail closed on everything else.

    CPU-only tests inject ``cmdline_of`` and never call nvidia-smi or /proc.
    """
    _require(stage in STAGES, f"unknown stage: {stage!r}")
    limit = plan.MAX_GPU1_USED_MEMORY_MIB_BEFORE_START
    if int(used_memory_mib) > limit:
        raise PreflightError(
            f"GPU 1 used memory {used_memory_mib} MiB exceeds {limit} MiB headroom guard"
        )
    apps = [(int(pid), int(used)) for pid, used in compute_apps]
    if stage in plan.EXCLUSIVE_CARD_STAGES:
        _require(
            not apps,
            f"stage {stage} requires an exclusive card (zero compute apps on GPU 1); "
            f"saw {apps}",
        )
        return {
            "concurrency_mode": "exclusive_card",
            "sibling_arm_compute_apps": [],
        }

    siblings: list[dict[str, object]] = []
    for pid, app_mib in apps:
        argv = _resolve_cmdline(pid, cmdline_of)
        kind, sibling_stage = classify_compute_app(
            argv, starting_stage=stage, launcher=Path(launcher),
        )
        if kind == "foreign":
            raise PreflightError(
                f"GPU 1 has a foreign compute process pid={pid} argv={list(argv)!r}"
            )
        if kind == "same_stage":
            raise PreflightError(
                f"GPU 1 already has stage {stage} (pid={pid}); refusing double-launch"
            )
        siblings.append({
            "pid": pid,
            "stage": sibling_stage,
            "used_memory_mib": app_mib,
        })
    if len(siblings) > 1:
        raise PreflightError(f"more than one sibling arm on GPU 1: {siblings}")
    return {
        "concurrency_mode": "sibling_arm_allowed",
        "sibling_arm_compute_apps": siblings,
    }


def check_gpu1(stage: str) -> dict[str, object]:
    """Query ONLY physical GPU 1.  Never a bare nvidia-smi, never --id 0."""
    identity = _nvidia_smi(
        "--query-gpu=uuid,pci.bus_id,name,memory.used",
        "--format=csv,noheader",
        "--id", "1",
    ).strip()
    parts = [part.strip() for part in identity.split(",")]
    _require(len(parts) == 4, f"GPU 1 identity csv drift: {identity!r}")
    uuid, pci, name, memory_raw = parts
    _require(uuid == GPU1_UUID, f"GPU 1 uuid drift: {uuid}")
    _require(uuid != GPU0_UUID_REFUSED, "refused: resolved device is physical GPU 0")
    _require(name == EXPECTED_DEVICE_NAME, f"GPU 1 name drift: {name}")
    used_memory_mib = parse_used_memory_mib(memory_raw)

    apps_csv = _nvidia_smi(
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader",
        "--id", "1",
    )
    classified = evaluate_gpu1_concurrency(
        stage,
        compute_apps=parse_compute_apps_csv(apps_csv),
        used_memory_mib=used_memory_mib,
        cmdline_of=read_proc_cmdline,
        launcher=LAUNCHER_PATH,
    )
    return {
        "uuid": uuid,
        "pci_bus_id": pci,
        "name": name,
        "foreign_compute_apps": [],
        "used_memory_mib": used_memory_mib,
        **classified,
    }


def check_torch() -> dict[str, object]:
    import torch

    _require(str(torch.__version__) == EXPECTED_TORCH,
             f"torch {torch.__version__}, required {EXPECTED_TORCH}; is PYTHONNOUSERSITE=1 set?")
    _require(str(torch.version.cuda) == EXPECTED_CUDA, f"cuda {torch.version.cuda}")
    _require(int(torch.backends.cudnn.version()) == EXPECTED_CUDNN,
             f"cudnn {torch.backends.cudnn.version()}")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
             "exactly one visible CUDA device is required")
    props = torch.cuda.get_device_properties(0)
    _require(props.name == EXPECTED_DEVICE_NAME, f"device {props.name}")
    _require(int(props.total_memory) == EXPECTED_TOTAL_MEMORY_BYTES,
             f"total_memory {props.total_memory}")
    return {
        "torch": str(torch.__version__), "cuda": str(torch.version.cuda),
        "cudnn": int(torch.backends.cudnn.version()), "device": props.name,
        "total_memory_bytes": int(props.total_memory),
    }


def sibling_stage_from_preflight(payload: Mapping[str, object]) -> str | None:
    siblings = payload.get("sibling_arm_compute_apps")
    if not siblings:
        return None
    _require(isinstance(siblings, list) and len(siblings) == 1,
             "preflight sibling list drifted")
    stage = siblings[0]["stage"]  # type: ignore[index]
    _require(stage in plan.CONCURRENT_ARM_STAGES, "preflight sibling stage drifted")
    return str(stage)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--arm", choices=("t0", "c1"), default=None,
                        help="required when --stage is t0 or c1 (defaults to the stage name)")
    parser.add_argument("--i-am-root-reviewer", action="store_true",
                        help="explicit acknowledgement: GPU 1 under the 50ep envelope "
                             "(t0/c1 may share the card; smoke/probe/phase3 exclusive)")
    parser.add_argument("--preflight-only", action="store_true",
                        help="run every check and exit without touching a result root")
    arguments = parser.parse_args(argv)

    if not arguments.i_am_root_reviewer and not arguments.preflight_only:
        parser.error("stage execution requires --i-am-root-reviewer")

    preflight: dict[str, object] = {"stage": arguments.stage}
    try:
        preflight["environment"] = check_environment()
        gpu1 = check_gpu1(arguments.stage)
        preflight["gpu1"] = {
            "uuid": gpu1["uuid"],
            "pci_bus_id": gpu1["pci_bus_id"],
            "name": gpu1["name"],
            "foreign_compute_apps": gpu1["foreign_compute_apps"],
        }
        preflight["concurrency_mode"] = gpu1["concurrency_mode"]
        preflight["sibling_arm_compute_apps"] = gpu1["sibling_arm_compute_apps"]
        preflight["torch"] = check_torch()
    except PreflightError as error:
        print(json.dumps({"status": "PREFLIGHT_REFUSED", "reason": str(error)},
                         sort_keys=True, indent=1))
        return 2

    print(json.dumps({"status": "PREFLIGHT_OK", **preflight}, sort_keys=True, indent=1))
    if arguments.preflight_only:
        return 0

    from tfpd_exploration.src.m1_t0c1_prefix_v1_50ep import driver

    stage = arguments.stage
    kwargs = {"source_root": REPO_ROOT, "device": "cuda:0"}
    if stage == "smoke":
        terminal_sha, failure_sha = driver.execute_smoke(REPO_ROOT, **kwargs)
    elif stage in ("t0", "c1"):
        arm = arguments.arm or stage
        _require(arm == stage, f"--arm {arm} contradicts --stage {stage}")
        terminal_sha, failure_sha = driver.execute_arm(
            REPO_ROOT, arm=arm,
            concurrent_sibling_stage=sibling_stage_from_preflight(preflight),
            **kwargs,
        )
    elif stage == "probe":
        terminal_sha, failure_sha = driver.execute_probe(REPO_ROOT, **kwargs)
    else:
        terminal_sha, failure_sha = driver.execute_phase3(REPO_ROOT, **kwargs)

    result = {
        "stage": stage,
        "status": "TERMINAL" if terminal_sha else "FAILED",
        "terminal_sha256": terminal_sha,
        "failure_sha256": failure_sha,
    }
    print(json.dumps(result, sort_keys=True, indent=1))
    return 0 if terminal_sha else 1


if __name__ == "__main__":
    raise SystemExit(main())
