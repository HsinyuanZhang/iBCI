"""Target-GPU idle / UUID bind for fold-local Stage-1."""
from __future__ import annotations

from pathlib import Path
import subprocess
import time
from typing import Callable

from . import plan


class GpuError(RuntimeError):
    """Fail closed for GPU occupancy or UUID drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GpuError(message)


def _default_cmdline(arguments: list[str]) -> str:
    completed = subprocess.run(arguments, check=True, text=True, capture_output=True, timeout=30)
    return completed.stdout


def _default_pid_cmdline(pid: str) -> str:
    path = Path(f"/proc/{int(pid)}/cmdline")
    return path.read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")


def query_gpu_rows(cmdline_runner: Callable[[list[str]], str] | None = None) -> list[dict[str, str]]:
    runner = cmdline_runner or _default_cmdline
    text = runner(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                   "--format=csv,noheader,nounits"])
    rows: list[dict[str, str]] = []
    for line in (line.strip() for line in text.strip().splitlines() if line.strip()):
        parts = [part.strip() for part in line.split(",")]
        _require(len(parts) == 4, "nvidia-smi row drift")
        rows.append({"index": parts[0], "uuid": parts[1],
                     "memory_used_mib": parts[2], "utilization_gpu": parts[3]})
    _require(bool(rows), "nvidia-smi returned no GPUs")
    return rows


def query_compute_apps(cmdline_runner: Callable[[list[str]], str] | None = None) -> list[dict[str, str]]:
    runner = cmdline_runner or _default_cmdline
    text = runner(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                   "--format=csv,noheader,nounits"])
    rows: list[dict[str, str]] = []
    for line in (line.strip() for line in text.strip().splitlines() if line.strip()):
        parts = [part.strip() for part in line.split(",")]
        _require(len(parts) == 4, "nvidia-smi compute-app row drift")
        rows.append({"gpu_uuid": parts[0], "pid": parts[1],
                     "process_name": parts[2], "used_memory_mib": parts[3]})
    return rows


def expected_uuid(gpu_index: int) -> str:
    if gpu_index == plan.GPU0_INDEX:
        return plan.GPU0_UUID
    if gpu_index == plan.GPU1_INDEX:
        return plan.GPU1_UUID
    raise GpuError(f"unsupported gpu index {gpu_index}")


def classify_occupants(
    apps: list[dict[str, str]],
    target_uuid: str,
    *,
    pid_cmdline: Callable[[str], str] | None = None,
    own_tokens: tuple[str, ...] = plan.OWN_TOKENS,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    reader = pid_cmdline or _default_pid_cmdline
    own: list[dict[str, str]] = []
    foreign: list[dict[str, str]] = []
    for app in apps:
        if app["gpu_uuid"] != target_uuid:
            continue
        try:
            command = reader(app["pid"])
        except OSError as exc:
            raise GpuError(f"cannot read cmdline for pid {app['pid']}: {exc}") from exc
        row = {**app, "cmdline": command}
        if any(token in command for token in own_tokens):
            own.append(row)
        else:
            foreign.append(row)
    return own, foreign


def assert_target_gpu_launchable(
    gpu_index: int,
    cmdline_runner: Callable[[list[str]], str] | None = None,
    *,
    pack: bool = False,
    pid_cmdline: Callable[[str], str] | None = None,
    pack_limit: int = plan.PACK_LIMIT,
) -> dict[str, object]:
    """Refuse foreign jobs. Packing allows up to pack_limit same-route arms."""
    _require(type(gpu_index) is int and gpu_index in (plan.GPU0_INDEX, plan.GPU1_INDEX), "gpu_index")
    _require(type(pack) is bool, "pack flag")
    _require(type(pack_limit) is int and 1 <= pack_limit <= plan.FULL_QUERY_PACK_LIMIT, "pack_limit")
    rows = query_gpu_rows(cmdline_runner)
    target = next((row for row in rows if row["index"] == str(gpu_index)), None)
    _require(target is not None, f"GPU {gpu_index} absent")
    _require(target["uuid"] == expected_uuid(gpu_index), "GPU UUID drift")
    utilization = float(target["utilization_gpu"])
    memory = float(target["memory_used_mib"])
    apps = query_compute_apps(cmdline_runner)
    target_apps = [app for app in apps if app["gpu_uuid"] == target["uuid"]]
    if pack:
        own, foreign = classify_occupants(target_apps, target["uuid"], pid_cmdline=pid_cmdline)
        _require(not foreign, f"GPU {gpu_index} has foreign compute apps: {[app['pid'] for app in foreign]}")
        _require(len(own) < pack_limit, f"GPU {gpu_index} pack limit {pack_limit} already filled")
        if gpu_index == plan.GPU0_INDEX:
            _require(plan.GPU0_ALLOWED, "GPU 0 is not allowed")
        occupants = own
    else:
        _require(utilization <= 0.0, f"GPU {gpu_index} util {utilization}%")
        if gpu_index == plan.GPU1_INDEX:
            _require(memory < 100.0, f"GPU {gpu_index} memory {memory} MiB")
        else:
            _require(plan.GPU0_ALLOWED, "GPU 0 is not allowed")
        _require(not target_apps, f"GPU {gpu_index} has compute apps: {[app['pid'] for app in target_apps]}")
        occupants = []
    return {
        "gpu_index": int(gpu_index),
        "target_uuid": target["uuid"],
        "target_memory_used_mib": target["memory_used_mib"],
        "target_utilization_gpu": target["utilization_gpu"],
        "target_compute_apps": [dict(app) for app in occupants],
        "own_occupants": len(occupants),
        "pack": pack,
        "pack_limit": pack_limit,
        "display_memory_allowed": gpu_index == plan.GPU0_INDEX,
        "checked_at_epoch_seconds": time.time(),
        "other_cards": [dict(row) for row in rows if row["index"] != str(gpu_index)],
    }


def require_launch(
    gpu_authorized: bool,
    gpu_index: int,
    cmdline_runner: Callable[[list[str]], str] | None = None,
    *,
    pack: bool = False,
    pid_cmdline: Callable[[str], str] | None = None,
    pack_limit: int = plan.PACK_LIMIT,
) -> dict[str, object]:
    _require(bool(gpu_authorized) is True, "Stage-1 GPU requires --gpu-authorized")
    return assert_target_gpu_launchable(
        gpu_index, cmdline_runner, pack=pack, pid_cmdline=pid_cmdline,
        pack_limit=pack_limit,
    )
