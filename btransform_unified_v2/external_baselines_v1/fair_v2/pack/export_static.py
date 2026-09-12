#!/usr/bin/env python3
"""Export SHA-locked static-RIFT + diag-z/CORAL payloads. Neural-only maps."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np
import torch

PACK = Path(__file__).resolve().parent
FAIR = PACK.parent
ROOT = FAIR.parent
WS = ROOT.parent
for path in (PACK, ROOT, WS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fair_v2 import data
sys.path.insert(0, str(FAIR))
from static_controls import EXPECTED, fit, model_for, predict, transform
from common import (
    LOCAL_SCORES,
    RESULTS,
    SELECTION_BUDGET,
    STATIC_CKPTS,
    STATIC_EXPORT_TOLERANCE,
    TASKS,
    UNIFIED_V2,
    WORKSPACE,
    sha256_file,
)

from falcon_challenge.config import FalconConfig, FalconTask

V1_SRC = WORKSPACE / "btransform_unified_v1/src"
V2_SRC = UNIFIED_V2 / "src"
LEARNABLE_SRC = UNIFIED_V2 / "learnable_recency_v1/src/learnable_recency_v1"
SKIP_PKG = {"joint_m2_model.py", "m2_mechanism_model.py", "joint_m1_model.py"}
SAFE_INIT = '''"""Container-safe RIFT decoder package. Joint/mechanism modules are omitted."""

from .config import RiftTemporalConfig
from .model import RiftDecoder
from .streaming import RiftStreamDecoder
from .temporal import RiftTemporal, RiftTemporalState
from .cpu_temporal import CpuRiftTemporalRuntime

__all__ = [
    "CpuRiftTemporalRuntime",
    "RiftDecoder",
    "RiftStreamDecoder",
    "RiftTemporal",
    "RiftTemporalConfig",
    "RiftTemporalState",
]
'''
LEARNABLE_SAFE_INIT = '''"""Container-safe learnable recency package."""

from .config import LearnableRecencyConfig, config_from_run_meta
from .cpu_temporal import CpuLearnableRecencyRuntime
from .static_model import StaticLearnableRiftDecoder
from .temporal import LearnableRecencyTemporal
from .wrap import LearnableRiftDecoder

__all__ = [
    "CpuLearnableRecencyRuntime",
    "LearnableRecencyConfig",
    "LearnableRecencyTemporal",
    "LearnableRiftDecoder",
    "StaticLearnableRiftDecoder",
    "config_from_run_meta",
]
'''

SCHEMA = "fair_v2_static_rift_frontend_v1"
ARMS = {"static_rift_diag_z": "diag_z", "static_rift_coral": "coral"}
STATIC_PRED_ARM = {"static_rift_diag_z": "diag_z", "static_rift_coral": "coral"}


def copy_pkg(dest: Path) -> Path:
    pkg = dest / "artifacts" / "pkg"
    if pkg.exists():
        shutil.rmtree(pkg)
    for src, name in (
        (V1_SRC / "btransform_unified_v1", "btransform_unified_v1"),
        (V2_SRC / "btransform_unified_v2", "btransform_unified_v2"),
    ):
        dst = pkg / name
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.rglob("*.py"):
            if item.name in SKIP_PKG:
                continue
            target = dst / item.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
    (pkg / "btransform_unified_v2" / "__init__.py").write_text(SAFE_INIT)
    lr = pkg / "learnable_recency_v1"
    lr.mkdir(parents=True, exist_ok=True)
    for name in ("config.py", "temporal.py", "cpu_temporal.py", "static_model.py"):
        shutil.copy2(LEARNABLE_SRC / name, lr / name)
    shutil.copy2(PACK / "wrap_safe.py", lr / "wrap.py")
    (lr / "__init__.py").write_text(LEARNABLE_SAFE_INIT)
    return pkg


def serialize_transform(fitted: dict) -> dict:
    arm = fitted["arm"]
    if arm == "diag_z":
        return {
            "arm": "diag_z",
            "mt": np.ascontiguousarray(fitted["mt"], np.float64),
            "st": np.ascontiguousarray(fitted["st"], np.float64),
            "ms": np.ascontiguousarray(fitted["ms"], np.float64),
            "ss": np.ascontiguousarray(fitted["ss"], np.float64),
        }
    if arm == "coral":
        return {
            "arm": "coral",
            "mt": np.ascontiguousarray(fitted["mt"], np.float64),
            "ms": np.ascontiguousarray(fitted["ms"], np.float64),
            "A": np.ascontiguousarray(fitted["A"], np.float64),
            "shrinkage": 0.1,
            "ridge": 0.001,
        }
    raise ValueError(arm)


def raw_basename(item: dict[str, Any], task: str, session: str, role: str) -> str:
    raw = item.get("support_provenance", {}).get("raw_nwb")
    if raw:
        return Path(raw).stem
    if task == "m2":
        token = "held-in" if role == "heldin" else "held-out"
        return f"sub-MonkeyN-held-{token}-calib_{session}_behavior+ecephys"
    raise ValueError(f"missing raw_nwb for {task}/{session}")


def build_payload(task: str, pack_arm: str, dest: Path) -> dict[str, Any]:
    frontend = ARMS[pack_arm]
    dest.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    ckpt = STATIC_CKPTS[task]
    if sha256_file(ckpt["path"]) != ckpt["sha256"] or sha256_file(ckpt["path"]) != EXPECTED[task]:
        raise RuntimeError(f"{task}: frozen checkpoint hash drift")
    artifact = json.loads((RESULTS / f"static_{task}_v2" / "artifact_receipt.json").read_text())
    if artifact["checkpoint_sha256"] != ckpt["sha256"]:
        raise RuntimeError(f"{task}: artifact receipt checkpoint mismatch")
    copy_pkg(dest)
    loaded = data.load_task(task, include_evaluation=True)
    model, meta = model_for(task)
    config = FalconConfig(task=getattr(FalconTask, task))
    transforms = {}
    sessions = {}
    checks = {}
    for role, group in (("heldin", loaded["train"]), ("heldout", loaded["evaluation"])):
        for session, item in group.items():
            basename = raw_basename(item, task, session, role)
            tag = str(config.hash_dataset(basename))
            fitted = fit(loaded["train"], item, frontend)
            transforms[tag] = serialize_transform(fitted)
            sessions[tag] = {
                "source_session": session,
                "raw_basename": basename,
                "role": role,
                "support_n": int(len(item["support"])),
                "target_labels_used": False,
            }
            if role != "heldout":
                continue
            raw = transform(np.asarray(item["X"])[int(item["pad"]) :], fitted)
            pred = np.asarray(predict(task, model, item, raw), np.float32)
            stored = np.load(RESULTS / f"static_{task}_v2" / f"{frontend}_{session}_pred.npy")
            err = float(np.max(np.abs(pred.astype(np.float64) - stored.astype(np.float64))))
            checks[session] = {
                "kind": "offline_windowed_vs_sealed",
                "n": int(len(pred)),
                "max_abs_error": err,
                "tolerance": STATIC_EXPORT_TOLERANCE,
                "pass": err <= STATIC_EXPORT_TOLERANCE,
            }
    failed = {k: v for k, v in checks.items() if not v["pass"]}
    if failed:
        raise RuntimeError(f"{task}/{pack_arm}: sealed replay mismatch {failed}")
    if len(transforms) != TASKS[task]["roster"]:
        raise RuntimeError(f"{task}: roster {len(transforms)} != {TASKS[task]['roster']}")
    payload = {
        "schema": SCHEMA,
        "task": task,
        "frontend": frontend,
        "pack_arm": pack_arm,
        "context_bins": TASKS[task]["context"],
        "identity": "static",
        "identity_interface": "static_identity_table",
        "proj_dim": 16,
        "tier": meta["learnable_config"]["tier"] if "learnable_config" in meta else json.loads(
            (ckpt["path"].parent / "run_meta.json").read_text()
        )["learnable_config"]["tier"],
        "ladder": "default",
        "seed": 42,
        "behavior_scaling_factor": TASKS[task]["behavior_scale"],
        "checkpoint": str(ckpt["path"]),
        "checkpoint_sha256": ckpt["sha256"],
        "artifact_receipt_sha256": sha256_file(RESULTS / f"static_{task}_v2" / "artifact_receipt.json"),
        "learnable_config": dict(
            json.loads((ckpt["path"].parent / "run_meta.json").read_text())["learnable_config"]
        ),
        "official_dataset_tags": sorted(transforms),
        "session_transforms": transforms,
        "sessions": sessions,
        "ema_state_dict": {k: v.detach().cpu().float().clone() for k, v in model.state_dict().items()},
        "coral_settings": {"shrinkage": 0.1, "ridge": 0.001, "kind": "diagonal_covariance_fixed_in_advance"},
        "selection_budget_disclosure": SELECTION_BUDGET,
        "local_public_calibration_scores": LOCAL_SCORES[(task, pack_arm)],
        "no_wf_smoothing": True,
        "same_network_weights_as_sibling": True,
    }
    pkl = dest / "artifacts" / "decoder.pkl"
    with pkl.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    audit = {
        "schema": "fair_v2_static_export_audit_v1",
        "task": task,
        "method": pack_arm,
        "frontend": frontend,
        "payload_sha256": sha256_file(pkl),
        "payload_bytes": pkl.stat().st_size,
        "checkpoint_sha256": ckpt["sha256"],
        "tag_count": len(transforms),
        "tags": sorted(transforms),
        "checks": checks,
        "max_abs_error": max(x["max_abs_error"] for x in checks.values()),
        "elapsed_seconds": time.monotonic() - started,
    }
    (dest / "export_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n")
    manifest = {
        "schema": SCHEMA,
        "task": task,
        "method": pack_arm,
        "frontend": frontend,
        "channels": TASKS[task]["channels"],
        "outputs": TASKS[task]["outputs"],
        "history": TASKS[task]["context"],
        "max_batch": TASKS[task]["max_batch"],
        "sessions": sessions,
        "payload": {
            "file": "artifacts/decoder.pkl",
            "sha256": audit["payload_sha256"],
            "contains_raw_data": False,
            "contains_target_y": False,
            "numeric_frontend_only": True,
        },
        "checkpoint_sha256": ckpt["sha256"],
        "local_public_calibration_scores": LOCAL_SCORES[(task, pack_arm)],
        "selection_budget_disclosure": SELECTION_BUDGET,
        "expected_prediction_fields": {
            "task": task,
            "channels": TASKS[task]["channels"],
            "outputs": TASKS[task]["outputs"],
            "history": TASKS[task]["context"],
        },
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("m1", "m2", "h1"), required=True)
    parser.add_argument("--method", choices=tuple(ARMS), required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_payload(args.task, args.method, args.dest), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
