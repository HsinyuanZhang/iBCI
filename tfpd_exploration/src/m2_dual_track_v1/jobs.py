"""Append-only job ledger and GPU lease. Coordinator-owned."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import plan

LEDGER_NAME = "jobs.jsonl"
LEASE_NAME = "gpu_lease.json"
GPU_UUID_BY_INDEX = {
    0: plan.GPU0_UUID,
    1: plan.GPU1_UUID,
}


def visible_gpu_index() -> int:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    plan.require(raw is not None and str(raw).strip() != "", "CUDA_VISIBLE_DEVICES required")
    parts = [part.strip() for part in str(raw).split(",") if part.strip() != ""]
    plan.require(len(parts) == 1, f"exactly one visible GPU, got {raw!r}")
    index = int(parts[0])
    plan.require(index in GPU_UUID_BY_INDEX, f"unknown physical GPU index {index}")
    return index


def visible_gpu_uuid() -> str:
    return GPU_UUID_BY_INDEX[visible_gpu_index()]


@dataclass
class JobRecord:
    run_id: str
    owner: str
    arm: str
    status: str
    gpu_uuid: str | None = None
    pid: int | None = None
    started_unix: float | None = None
    budget_gpu_hours: float = plan.JOB_GPU_HOUR_LIMIT
    config_hash: str = ""
    log_path: str = ""
    ckpt_path: str = ""
    note: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


def _run_root(run_root: Path | None = None) -> Path:
    root = Path(run_root) if run_root is not None else plan.active_run_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "jobs").mkdir(exist_ok=True)
    return root


def ledger_path(run_root: Path | None = None) -> Path:
    return _run_root(run_root) / LEDGER_NAME


def lease_path(run_root: Path | None = None) -> Path:
    return _run_root(run_root) / LEASE_NAME


def append_job(record: JobRecord, run_root: Path | None = None) -> None:
    path = ledger_path(run_root)
    payload = asdict(record)
    payload["logged_at"] = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def default_lease() -> dict[str, Any]:
    return {
        "schema": plan.SCHEMA,
        "updated_unix": time.time(),
        "gpus": {
            plan.GPU0_UUID: {
                "index": 0,
                "default_owner": plan.GPU0_DEFAULT_OWNER,
                "lessee": None,
                "run_id": None,
            },
            plan.GPU1_UUID: {
                "index": 1,
                "default_owner": plan.GPU1_DEFAULT_OWNER,
                "lessee": None,
                "run_id": None,
            },
        },
    }


def load_lease(run_root: Path | None = None) -> dict[str, Any]:
    path = lease_path(run_root)
    if not path.exists():
        lease = default_lease()
        save_lease(lease, run_root)
        return lease
    return json.loads(path.read_text(encoding="utf-8"))


def save_lease(lease: dict[str, Any], run_root: Path | None = None) -> None:
    path = lease_path(run_root)
    lease["updated_unix"] = time.time()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(lease, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def acquire_lease(
    *,
    gpu_uuid: str,
    lessee: str,
    run_id: str,
    run_root: Path | None = None,
) -> dict[str, Any]:
    lease = load_lease(run_root)
    slot = lease["gpus"][gpu_uuid]
    current = slot.get("lessee")
    if current not in {None, lessee}:
        raise plan.DualTrackError(f"GPU {gpu_uuid} leased to {current}, refused {lessee}")
    slot["lessee"] = lessee
    slot["run_id"] = run_id
    save_lease(lease, run_root)
    append_job(
        JobRecord(
            run_id=run_id,
            owner=lessee,
            arm=lessee,
            status="LEASED",
            gpu_uuid=gpu_uuid,
            note="coordinator lease acquire",
        ),
        run_root,
    )
    return lease


def release_lease(
    *,
    gpu_uuid: str,
    lessee: str,
    run_root: Path | None = None,
) -> dict[str, Any]:
    lease = load_lease(run_root)
    slot = lease["gpus"][gpu_uuid]
    if slot.get("lessee") not in {None, lessee}:
        raise plan.DualTrackError(f"cannot release {gpu_uuid}: owned by {slot.get('lessee')}")
    slot["lessee"] = None
    slot["run_id"] = None
    save_lease(lease, run_root)
    append_job(
        JobRecord(run_id="lease", owner=lessee, arm=lessee, status="RELEASED", gpu_uuid=gpu_uuid),
        run_root,
    )
    return lease
