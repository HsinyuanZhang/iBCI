"""Bind the unique P parent without launching a GPU pilot."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import contracts
from . import plan


class ParentAuditError(RuntimeError):
    """Fail closed for parent lineage."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ParentAuditError(message)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(repo_root: Path) -> dict[str, Any]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    repo_root = Path(repo_root)
    s_fix = repo_root / plan.S_FIX_EPOCH011_RELATIVE
    teacher = repo_root / plan.FOLD0_SOURCE_TEACHER_RELATIVE
    allsource = repo_root / plan.ALLSOURCE_TEACHER_RELATIVE
    z_fix = repo_root / plan.Z_FIX_EPOCH011_RELATIVE
    _require(s_fix.is_file(), "S-Fix epoch_011 missing")
    _require(plan.sha256_file(s_fix) == plan.S_FIX_EPOCH011_SHA256, "S-Fix bytes drifted")
    _require(teacher.is_file(), "fold0 source teacher missing")
    _require(plan.sha256_file(teacher) == plan.FOLD0_SOURCE_TEACHER_SHA256, "teacher bytes drifted")
    _require(plan.sha256_file(allsource) == plan.ALLSOURCE_TEACHER_SHA256, "all-source bytes drifted")
    _require(plan.sha256_file(z_fix) == plan.Z_FIX_EPOCH011_SHA256, "Z-Fix bytes drifted")

    import torch

    payload = torch.load(s_fix, map_location="cpu", weights_only=False)
    _require(isinstance(payload, dict), "S-Fix is not a Lightning mapping")
    state = payload.get("state_dict") or {}
    _require("student.carrier_projection_weight" in state, "S-Fix lacks independent P")
    weight = state["student.carrier_projection_weight"]
    _require(tuple(weight.shape) == (1024, 4), f"P shape {tuple(weight.shape)}")
    hparams = payload.get("hyper_parameters") or {}
    data = payload.get("datamodule_hyper_parameters") or {}
    teacher_path = Path(str(hparams.get("teacher_ckpt_path")))
    _require(teacher_path.resolve() == teacher.resolve(), "S-Fix teacher path is not the bound fold0 teacher")
    sources = [str(name) for name in (data.get("source_session_names") or ())]
    _require(tuple(sources) == plan.M1_FOLD0_SOURCES, f"S-Fix sources {sources}")
    _require(plan.M1_FOLD0_TARGET not in sources, "S-Fix sources include the outer target")
    _require(int(data.get("heldin_query_start_trial")) == plan.M1_QUERY_START, "S-Fix query start")
    _require(int(data.get("heldin_query_end_trial")) == plan.M1_QUERY_STOP_EXCLUSIVE, "S-Fix query stop")

    manifest_path = repo_root / plan.FOLD0_SOURCE_TEACHER_MANIFEST_RELATIVE
    _require(manifest_path.is_file(), "teacher manifest missing")
    manifest = _load_json(manifest_path)
    train_sessions = [str(name) for name in manifest.get("train_sessions") or ()]
    _require(manifest.get("outer_left_out") == plan.M1_FOLD0_TARGET, "teacher outer_left_out")
    _require(plan.M1_FOLD0_TARGET not in train_sessions, "teacher trained on 20120924")
    _require(manifest.get("source_only") is True, "teacher not marked source_only")
    _require(manifest.get("heldout_opened") is False, "teacher opened held-out")

    missing_full = [
        key
        for key in ("sampler", "normalizer", "basis", "torch_rng", "numpy_rng", "python_rng")
        if key not in payload
    ]
    return {
        "schema": "cross_dataset_functional_calibration_parent_audit_v1",
        "parent_bytes": plan.S_FIX_EPOCH011_SHA256,
        "parent_relative": plan.S_FIX_EPOCH011_RELATIVE,
        "parent_epoch": int(payload.get("epoch")),
        "parent_global_step": int(payload.get("global_step")),
        "teacher_bytes": plan.FOLD0_SOURCE_TEACHER_SHA256,
        "teacher_relative": plan.FOLD0_SOURCE_TEACHER_RELATIVE,
        "teacher_train_sessions": train_sessions,
        "outer_left_out": str(manifest.get("outer_left_out")),
        "s_fix_query": [int(data["heldin_query_start_trial"]), int(data["heldin_query_end_trial"])],
        "teacher_query_start_trial_hparams": 0,
        "teacher_query_caveat": (
            "fold0 teacher hparams use query_start_trial=0 on source sessions; "
            "that is source-calibration exposure, not 20120924 file leakage. "
            "P training still uses post-M10 windows."
        ),
        "injection": "student.carrier_projection_weight [1024,4] post-fc_in",
        "optimizer_in_parent": bool(payload.get("optimizer_states")),
        "optimizer_resume_authorized": False,
        "optimizer_note": "S-Fix is Adam wd=0; P freeze is AdamW wd=1e-2. Fresh optimizer.",
        "missing_full_ckpt_fields": missing_full,
        "full_ckpt_schema": False,
        "allsource_forbidden": True,
        "allsource_bytes": plan.ALLSOURCE_TEACHER_SHA256,
        "z_fix_rejected": True,
        "z_fix_bytes": plan.Z_FIX_EPOCH011_SHA256,
        "claim": "CLEAN_OUTER_SESSION_FILE_EXCLUSION",
        "gpu_eligible": False,
        "named_revision": plan.P_REVISION_RELATIVE,
        "unresolved_before": list(contracts.UNRESOLVED_P_OPERATORS),
    }
