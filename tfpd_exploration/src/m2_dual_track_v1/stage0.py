"""Stage 0 aggregator. Coordinator-owned; calls owner hooks when present."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import jobs, plan
from .monitor import snapshot

WORKORDER_SHA256 = "d0925809f26b36ce4955b3b7b1799a5b617e6bf765468a7dceed7db92de02b89"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha(state: dict) -> str:
    import numpy as np
    import torch

    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        tensor = state[key]
        if not hasattr(tensor, "detach"):
            raise plan.DualTrackError(f"non-tensor head leaf: {key}")
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        header = json.dumps(
            {"dtype": str(array.dtype), "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        leaf = hashlib.sha256()
        leaf.update(header)
        leaf.update(array.tobytes(order="C"))
        digest.update(leaf.hexdigest().encode())
    return digest.hexdigest()


def check_authority() -> dict[str, Any]:
    root = plan.repo_root()
    ckpt = root / plan.CHAMPION_CKPT_RELATIVE
    head = root / plan.SELECTED_HEAD_RELATIVE
    film = root / plan.FILM_STATES_RELATIVE
    split = root / plan.M33_SPLIT_MANIFEST_RELATIVE
    workorder = root / plan.WORKORDER_RELATIVE
    rows = {
        "champion_ckpt_exists": ckpt.exists(),
        "champion_ckpt_sha256": _sha256(ckpt) if ckpt.exists() else None,
        "champion_ckpt_match": ckpt.exists() and _sha256(ckpt) == plan.CHAMPION_CKPT_SHA256,
        "selected_head_exists": head.exists(),
        "film_states_exists": film.exists(),
        "split_manifest_exists": split.exists(),
        "workorder_sha256": _sha256(workorder) if workorder.exists() else None,
        "workorder_match": workorder.exists() and _sha256(workorder) == WORKORDER_SHA256,
    }
    if head.exists():
        import torch

        payload = torch.load(head, map_location="cpu", weights_only=False)
        rows["selected_head_state_sha256"] = _state_sha(payload["state_dict"])
        rows["selected_head_state_match"] = (
            rows["selected_head_state_sha256"] == plan.SELECTED_HEAD_STATE_SHA256
        )
        rows["selected_meta"] = {
            "seed": payload.get("selection", {}).get("seed"),
            "epoch_one_based": payload.get("selection", {}).get("epoch_one_based"),
        }
    if film.exists():
        import torch

        states = torch.load(film, map_location="cpu", weights_only=False)
        rows["film_states_keys"] = sorted(str(key) for key in states)
        rows["film_has_p0"] = "p0" in states
    rows["pass"] = bool(
        rows.get("champion_ckpt_match")
        and rows.get("selected_head_state_match")
        and rows.get("selected_head_exists")
        and rows.get("film_has_p0")
        and rows.get("split_manifest_exists")
        and rows.get("workorder_match")
    )
    return rows


def _optional_hook(module_name: str, attr: str) -> Callable[..., dict[str, Any]] | None:
    try:
        module = __import__(f"tfpd_exploration.src.m2_dual_track_v1.{module_name}", fromlist=[attr])
    except ImportError:
        return None
    return getattr(module, attr, None)


def run_stage0(*, subset: str = "all", device: str = "cpu") -> int:
    root = plan.active_run_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "stage0").mkdir(exist_ok=True)
    report: dict[str, Any] = {
        "schema": plan.SCHEMA,
        "contract_version": plan.CONTRACT_VERSION,
        "subset": subset,
        "device": device,
        "started": datetime.now(timezone.utc).isoformat(),
        "hardware": snapshot(),
    }
    jobs.append_job(
        jobs.JobRecord(run_id="stage0", owner="coordinator", arm="STAGE0", status="STARTED", note=subset),
        root,
    )
    if subset in {"all", "authority"}:
        report["authority"] = check_authority()
    if subset in {"all", "a"}:
        hook = _optional_hook("calibration_memory", "stage0_a")
        report["a"] = hook(device=device) if hook else {"status": "NOT_IMPLEMENTED"}
    if subset in {"all", "b"}:
        hook = _optional_hook("ssm_backend", "stage0_b")
        report["b"] = hook(device=device) if hook else {"status": "NOT_IMPLEMENTED"}
    if subset in {"all", "data"}:
        hook = _optional_hook("data", "stage0_data")
        report["data"] = hook() if hook else {"status": "NOT_IMPLEMENTED"}
    report["finished"] = datetime.now(timezone.utc).isoformat()
    dest = root / "stage0" / f"stage0_{subset}.json"
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if subset == "all":
        (root / "stage0.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ok = True
    if "authority" in report:
        ok = ok and bool(report["authority"].get("pass"))
    for key in ("a", "b", "data"):
        block = report.get(key)
        if isinstance(block, dict) and "pass" in block:
            ok = ok and bool(block["pass"])
    jobs.append_job(
        jobs.JobRecord(
            run_id="stage0",
            owner="coordinator",
            arm="STAGE0",
            status="PASS" if ok else "PARTIAL",
            note=subset,
        ),
        root,
    )
    print(json.dumps({"ok": ok, "path": str(dest)}, indent=2))
    return 0 if ok or subset != "all" else 1
