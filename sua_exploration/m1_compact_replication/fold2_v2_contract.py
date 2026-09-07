#!/usr/bin/env python3
"""Append-only, not-launched M1 fold-2 source-only v2 contract.

This file prepares only a dry proposal and staged receipt.  It intentionally
does not contain an executor or evaluator: fold 2 cannot be launched until a
separate fold-1 terminal gate is explicitly recorded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

from sua_exploration.m1_compact_replication import fold1_runner as base
from sua_exploration.m1_compact_replication import fold1_v2_contract as f1

ROOT = base.ROOT
V1_PROPOSAL = base.PROPOSAL
V1_PROPOSAL_SHA = base.EXPECTED_PROPOSAL_SHA
V2_PROPOSAL = ROOT / "sua_exploration/m1_compact_replication/proposals/M1_COMPACT_B3S_F2_S42_GPU_PROPOSAL_v2.json"
V2_RECEIPT = ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F2_S42_STAGED_RECEIPT_v2.json"
V2_SCHEMA = "m1_compact_b3s_f2_s42_gpu_proposal_v2"
V2_STATUS = "PASS_M1_COMPACT_B3S_F2_S42_PROPOSAL_PREPARED_NOT_LAUNCHED"
RECEIPT_SCHEMA = "m1_compact_b3s_f2_s42_staged_v2"
RECEIPT_STATUS = "PASS_M1_COMPACT_B3S_F2_S42_STAGED_WAITING_FOLD1_NOT_LAUNCHED"
TARGET = "ses-20120927"
SOURCES = ("ses-20120924", "ses-20120926", "ses-20120928")
PYTHON = f1.PYTHON
SOURCE_ONLY_TARGET = f1.SOURCE_ONLY_TARGET
TEACHER = ROOT / "streaming_calibration_exp/logs/m1_afc4_source_decoder_fold2_remote/runs/remote_fold2_source_epoch019/checkpoints/best_ckpt/epoch_019.ckpt"
TEACHER_MANIFEST = ROOT / "streaming_calibration_exp/logs/m1_afc4_source_decoder_fold2_remote/runs/remote_fold2_source_epoch019/source_only_decoder_manifest.json"
EXPECTED_TEACHER_SHA = "925fba67a6a4338ee6e399751e80d53c5e74a22a7dd3ae0329d9c8da9e0291e2"
EXPECTED_MANIFEST_SHA = "4dc53581280281326257c158a977b329195d0335443c658899f83a9a4410e4bb"


def _immutable(path: Path, body: dict[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body["canonical_content_sha256"] = base.canonical(body)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp = Path(name)
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
        tmp.chmod(0o444)
        tmp.replace(path)
        return base.sha(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _v1() -> dict[str, Any]:
    if base.sha(V1_PROPOSAL) != V1_PROPOSAL_SHA:
        raise RuntimeError("v1 proposal SHA drift")
    return base.read(V1_PROPOSAL, "v1 proposal")


def validate_teacher() -> dict[str, Any]:
    if base.sha(TEACHER) != EXPECTED_TEACHER_SHA or base.sha(TEACHER_MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("fold2 teacher SHA drift")
    manifest = base.read(TEACHER_MANIFEST, "fold2 teacher manifest")
    if manifest.get("task") != "m1" or manifest.get("outer_fold") != 2 or manifest.get("outer_left_out") != TARGET:
        raise RuntimeError("fold2 teacher scope drift")
    if manifest.get("source_only") is not True or manifest.get("target_backpropagation") is not False:
        raise RuntimeError("fold2 teacher source-only drift")
    if tuple(manifest.get("train_sessions", ())) != SOURCES or manifest.get("validation_sessions") != []:
        raise RuntimeError("fold2 teacher source list drift")
    if TARGET in manifest.get("source_files", {}):
        raise RuntimeError("fold2 teacher target source file present")
    source_files = {}
    for session in SOURCES:
        spec = manifest.get("source_files", {}).get(session, {})
        path = Path(str(spec.get("path", "")))
        if not path.is_file() or base.sha(path) != spec.get("sha256"):
            raise RuntimeError(f"fold2 teacher source hash drift: {session}")
        source_files[session] = {"path": str(path.resolve()), "sha256": str(spec["sha256"]), "bytes": path.stat().st_size}
    import torch
    payload = torch.load(TEACHER, map_location="cpu", weights_only=False)
    if payload.get("epoch") != 19:
        raise RuntimeError("fold2 teacher epoch drift")
    return {"checkpoint": {"path": str(TEACHER.resolve()), "sha256": EXPECTED_TEACHER_SHA, "epoch": 19}, "manifest": {"path": str(TEACHER_MANIFEST.resolve()), "sha256": EXPECTED_MANIFEST_SHA}, "source_files": source_files, "outer_target": TARGET, "source_sessions": list(SOURCES), "source_only": True}


def source_only_inventory() -> dict[str, Any]:
    code = {rel: base.sha(ROOT / rel) for rel in f1.SOURCE_CODE_FILES}
    data = ROOT / "SPINT-main/data/000941"
    directory = data / "sub-MonkeyL-held-in-calib"
    source_data = {}
    for session in SOURCES:
        suffix = session.removeprefix("ses-")
        path = directory / f"sub-MonkeyL-held-in-calib_ses-{suffix}_behavior+ecephys.nwb"
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing/symlinked fold2 source: {session}")
        source_data[path.relative_to(data).as_posix()] = base.sha(path)
    return {"code_files_sha256": code, "data_root": str(data.resolve()), "source_sessions": list(SOURCES), "source_data_files_sha256": source_data, "source_data_file_count": len(source_data), "target_session_excluded": TARGET, "target_data_hashed_by_prelaunch": False}


def build_proposal() -> dict[str, Any]:
    source = _v1()
    cells = [cell for cell in source["folds1_2_cross_session"]["cells"] if cell.get("fold") == 2]
    cells.sort(key=lambda cell: 0 if cell.get("arm") == "b0" else 1)
    if [cell.get("arm") for cell in cells] != ["b0", "b3s_zero4"]:
        raise RuntimeError("v1 fold2 order drift")
    v2_cells = []
    for cell in cells:
        argv = list(cell["argv"])
        argv[argv.index("test=true")] = "test=false"
        argv.insert(argv.index("test=false"), f"data._target_={SOURCE_ONLY_TARGET}")
        argv.insert(argv.index("test=false") + 1, "optimized_metric=null")
        v2_cells.append({**cell, "argv": argv})
    return {"schema": V2_SCHEMA, "status": V2_STATUS, "parent_v1": {"path": str(V1_PROPOSAL.resolve()), "sha256": V1_PROPOSAL_SHA}, "scope": {"task": "m1", "fold": 2, "seed": 42, "outer_target": TARGET, "source_sessions": list(SOURCES), "support_trials": [0, 10], "query_trials": [10, 210], "formal_or_minival_or_heldout": False, "target_backward_steps": 0, "target_optimizer_steps": 0, "target_checkpoint_selection": False}, "cells": v2_cells, "execution": {"fixed_order": ["b0", "b3s_zero4"], "train_commands_test_false": True, "source_only_fit_data_module": SOURCE_ONLY_TARGET, "target_is_opened_only_by_independent_evaluator_after_both_terminal_checkpoints": True, "intermediate_target_metric_read": False, "launched": False, "fold1_gate_required_before_launch": True}, "gate": {"fold1_gate_required": True, "fold1_gate_path": str(ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F1_S42_GATE_v2.json"), "metric": "B3S-Zero4 minus B0 pooled variance-weighted R2", "threshold": -0.03, "fold2_launch_authorized": False}}


def _validate_proposal(value: dict[str, Any]) -> None:
    if value.get("schema") != V2_SCHEMA or value.get("status") != V2_STATUS:
        raise RuntimeError("fold2 proposal schema/status drift")
    if value.get("scope", {}).get("source_sessions") != list(SOURCES) or value.get("scope", {}).get("outer_target") != TARGET:
        raise RuntimeError("fold2 proposal scope drift")
    cells = value.get("cells", [])
    if [cell.get("arm") for cell in cells] != ["b0", "b3s_zero4"]:
        raise RuntimeError("fold2 proposal order drift")
    expected_exp = ["m1_version_b_hs_continuation", "m1_version_b_c0"]
    expected_run = ["m1_compact_b0_f2_s42_fresh_e11", "m1_compact_b3s_zero4_f2_s42_fresh_e11"]
    for i, cell in enumerate(cells):
        argv = cell["argv"]
        required = [f"experiment={expected_exp[i]}", f"run_id={expected_run[i]}", "data.loso_fold=2", "data.source_session_names=[ses-20120924,ses-20120926,ses-20120928]", "seed=42", "trainer.min_epochs=12", "trainer.max_epochs=12", "trainer.limit_val_batches=0", "trainer.num_sanity_val_steps=0", f"data._target_={SOURCE_ONLY_TARGET}", "test=false", "optimized_metric=null", "model.teacher_ckpt_path=" + str(TEACHER.resolve()), "ckpt_path=null"]
        if len(argv) != 16 or any(argv.count(x) != 1 for x in required) or "test=true" in argv or argv[0] != PYTHON or argv[1] != str((ROOT / "streaming_calibration_exp/src/train.py").resolve()):
            raise RuntimeError(f"fold2 exact argv drift: {cell.get('arm')}")


def build_receipt() -> dict[str, Any]:
    proposal = build_proposal()
    if not V2_PROPOSAL.exists():
        _immutable(V2_PROPOSAL, proposal)
    stored = f1._read_immutable(V2_PROPOSAL, "fold2 proposal", V2_SCHEMA, V2_STATUS)
    _validate_proposal(stored)
    if base.canonical(proposal) != stored["canonical_content_sha256"]:
        raise RuntimeError("fold2 proposal append-only drift")
    teacher = validate_teacher()
    inventory = source_only_inventory()
    commands = {"b0": stored["cells"][0]["argv"], "b3s_zero4": stored["cells"][1]["argv"]}
    artifact = ROOT / "outputs/streaming_calibration"
    for run in ("m1_compact_b0_f2_s42_fresh_e11", "m1_compact_b3s_zero4_f2_s42_fresh_e11"):
        if base.prior_artifact_exists(artifact, run):
            raise RuntimeError(f"prior fold2 artifact exists: {run}")
    body = {"schema": RECEIPT_SCHEMA, "status": RECEIPT_STATUS, "proposal": {"path": str(V2_PROPOSAL.resolve()), "sha256": base.sha(V2_PROPOSAL), "schema": V2_SCHEMA}, "parent_v1_proposal": {"path": str(V1_PROPOSAL.resolve()), "sha256": V1_PROPOSAL_SHA}, "teacher": teacher, "scope": stored["scope"], "commands": commands, "inventory": inventory, "execution": stored["execution"], "target_policy": {"train_test_flag": False, "target_opened_by_training": False, "fit_data_module_source_only": True, "target_opened_by_evaluator_after_both_terminal": True, "intermediate_target_metric_read": False}, "fold1_gate": {"required": True, "available": False, "path": stored["gate"]["fold1_gate_path"], "launch_authorized": False}}
    if V2_RECEIPT.exists():
        existing = f1._read_immutable(V2_RECEIPT, "fold2 staged receipt", RECEIPT_SCHEMA, RECEIPT_STATUS)
        if base.canonical(body) != existing["canonical_content_sha256"]:
            raise RuntimeError("existing fold2 receipt differs")
        return existing
    _immutable(V2_RECEIPT, body)
    return f1._read_immutable(V2_RECEIPT, "fold2 staged receipt", RECEIPT_SCHEMA, RECEIPT_STATUS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal", action="store_true")
    parser.add_argument("--receipt", action="store_true")
    args = parser.parse_args()
    if args.proposal == args.receipt:
        raise SystemExit("choose exactly one --proposal/--receipt")
    body = build_proposal() if args.proposal else build_receipt()
    path = V2_PROPOSAL if args.proposal else V2_RECEIPT
    if args.proposal and not path.exists():
        _immutable(path, body)
    print(json.dumps({"path": str(path.resolve()), "sha256": base.sha(path), "status": body["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
