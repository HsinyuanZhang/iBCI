#!/usr/bin/env python3
"""Append-only v2 fold-1 contract: source fit only, target opened once later."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

from sua_exploration.m1_compact_replication import fold1_runner as v1


ROOT = v1.ROOT
V1_PROPOSAL = v1.PROPOSAL
V1_PROPOSAL_SHA = v1.EXPECTED_PROPOSAL_SHA
V2_PROPOSAL = ROOT / "sua_exploration/m1_compact_replication/proposals/M1_COMPACT_B3S_F1_S42_GPU_PROPOSAL_v2.json"
V2_RECEIPT = ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F1_S42_STAGED_RECEIPT_v2.json"
EQUIVALENCE_RECEIPT = ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F1_S42_SOURCE_FIT_EQUIVALENCE_v1.json"
EQUIVALENCE_BINDING = ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F1_S42_V2_EQUIVALENCE_BINDING_v1.json"
V2_SCHEMA = "m1_compact_b3s_f1_s42_gpu_proposal_v2"
V2_STATUS = "PASS_M1_COMPACT_B3S_F1_S42_PROPOSAL_PREPARED_NOT_LAUNCHED"
RECEIPT_SCHEMA = "m1_compact_b3s_f1_s42_staged_v2"
RECEIPT_STATUS = "PASS_M1_COMPACT_B3S_F1_S42_STAGED_NOT_LAUNCHED"
SOURCE_ONLY_TARGET = (
    "src.data.m1_version_b_source_loso_datamodule."
    "M1VersionBSourceOnlyFitDataModule"
)
SOURCE_SESSIONS = ("ses-20120924", "ses-20120927", "ses-20120928")
TARGET_SESSION = "ses-20120926"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
SOURCE_CODE_FILES = (
    "streaming_calibration_exp/src/train.py",
    "streaming_calibration_exp/src/data/m1_version_b_source_loso_datamodule.py",
    "streaming_calibration_exp/src/models/streaming_calibration_module.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/configs/experiment/m1_version_b_hs_continuation.yaml",
    "streaming_calibration_exp/configs/experiment/m1_version_b_c0.yaml",
    "streaming_calibration_exp/configs/data/m1_version_b_source_loso.yaml",
    "streaming_calibration_exp/configs/model/m1_version_b_b0.yaml",
    "streaming_calibration_exp/configs/model/m1_version_b_b3s.yaml",
    "streaming_calibration_exp/configs/callbacks/m1_version_b_fixed_epoch.yaml",
)


def _immutable(path: Path, body: dict[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body["canonical_content_sha256"] = v1.canonical(body)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(name)
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
            handle.flush()
        temporary.chmod(0o444)
        temporary.replace(path)
        return v1.sha(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _v1() -> dict[str, Any]:
    if v1.sha(V1_PROPOSAL) != V1_PROPOSAL_SHA:
        raise RuntimeError("v1 proposal SHA drift")
    value = v1.read(V1_PROPOSAL, "v1 proposal")
    if value.get("status") != "PASS_M1_COMPACT_B3S_PROPOSALS_PREPARED_NOT_LAUNCHED":
        raise RuntimeError("v1 proposal status drift")
    return value


def _validate_v2_proposal(value: dict[str, Any]) -> None:
    """Validate the append-only proposal, including source-only target class."""
    if value.get("schema") != V2_SCHEMA or value.get("status") != V2_STATUS:
        raise RuntimeError("v2 proposal schema/status drift")
    if value.get("parent_v1") != {"path": str(V1_PROPOSAL.resolve()), "sha256": V1_PROPOSAL_SHA}:
        raise RuntimeError("v2 parent proposal binding drift")
    scope = value.get("scope", {})
    if scope.get("source_sessions") != list(SOURCE_SESSIONS):
        raise RuntimeError("v2 source session scope drift")
    if scope.get("outer_target") != TARGET_SESSION:
        raise RuntimeError("v2 target scope drift")
    cells = value.get("cells", [])
    if [cell.get("arm") for cell in cells] != ["b0", "b3s_zero4"]:
        raise RuntimeError("v2 cell order drift")
    expected_experiments = ["m1_version_b_hs_continuation", "m1_version_b_c0"]
    expected_runs = [
        "m1_compact_b0_f1_s42_fresh_e11",
        "m1_compact_b3s_zero4_f1_s42_fresh_e11",
    ]
    for index, cell in enumerate(cells):
        argv = list(cell.get("argv", []))
        if len(argv) != 16:
            raise RuntimeError(f"v2 argv length drift for {cell.get('arm')}: {len(argv)}")
        required = {
            "experiment": f"experiment={expected_experiments[index]}",
            "run_id": f"run_id={expected_runs[index]}",
            "fold": "data.loso_fold=1",
            "sources": "data.source_session_names=[ses-20120924,ses-20120927,ses-20120928]",
            "seed": "seed=42",
            "min_epochs": "trainer.min_epochs=12",
            "max_epochs": "trainer.max_epochs=12",
            "limit_val": "trainer.limit_val_batches=0",
            "sanity": "trainer.num_sanity_val_steps=0",
            "source_only_target": f"data._target_={SOURCE_ONLY_TARGET}",
            "train_only": "test=false",
            "no_optimized_metric": "optimized_metric=null",
            "teacher": "model.teacher_ckpt_path=" + str(v1.TEACHER.resolve()),
            "null_ckpt": "ckpt_path=null",
        }
        for label, token in required.items():
            if argv.count(token) != 1:
                raise RuntimeError(f"v2 {label} binding drift for {cell.get('arm')}")
        if "test=true" in argv:
            raise RuntimeError(f"v2 test=true remains for {cell.get('arm')}")
        if argv[0] != PYTHON or argv[1] != str((v1.ROOT / "streaming_calibration_exp/src/train.py").resolve()):
            raise RuntimeError("v2 train entrypoint drift")
        if cell.get("fold") != 1 or cell.get("seed") != 42:
            raise RuntimeError("v2 cell fold/seed drift")
    execution = value.get("execution", {})
    if execution.get("train_commands_test_false") is not True:
        raise RuntimeError("v2 execution test policy drift")
    if execution.get("target_is_opened_only_by_independent_evaluator_after_both_terminal_checkpoints") is not True:
        raise RuntimeError("v2 target-opening policy drift")


def _read_immutable(path: Path, label: str, schema: str, status: str) -> dict[str, Any]:
    value = v1.read(path, label)
    if path.stat().st_mode & 0o777 != 0o444:
        raise RuntimeError(f"{label} is not read-only")
    if value.get("schema") != schema or value.get("status") != status:
        raise RuntimeError(f"{label} schema/status drift")
    if value.get("canonical_content_sha256") != v1.canonical(
        {k: v for k, v in value.items() if k != "canonical_content_sha256"}
    ):
        raise RuntimeError(f"{label} canonical hash drift")
    return value


def source_only_inventory() -> dict[str, Any]:
    """Hash only code and the three declared source NWBs.

    The target NWB is intentionally absent.  This receipt is created before
    GPU training, so even hashing the target here would violate the source-only
    fit boundary; the independent evaluator performs its own target read only
    after both terminal checkpoints exist.
    """
    code = {rel: v1.sha(ROOT / rel) for rel in SOURCE_CODE_FILES}
    data = ROOT / "SPINT-main/data/000941"
    if not data.is_dir():
        raise RuntimeError("M1 data root missing")
    source_data: dict[str, str] = {}
    directory = data / "sub-MonkeyL-held-in-calib"
    for session in SOURCE_SESSIONS:
        suffix = session.removeprefix("ses-")
        path = directory / f"sub-MonkeyL-held-in-calib_ses-{suffix}_behavior+ecephys.nwb"
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing/symlinked source NWB: {session}")
        source_data[path.relative_to(data).as_posix()] = v1.sha(path)
    return {
        "code_files_sha256": code,
        "data_root": str(data.resolve()),
        "source_sessions": list(SOURCE_SESSIONS),
        "source_data_files_sha256": source_data,
        "source_data_file_count": len(source_data),
        "target_session_excluded": TARGET_SESSION,
        "target_data_hashed_by_prelaunch": False,
    }


def validate_equivalence_binding(receipt_path: Path = V2_RECEIPT) -> dict[str, Any]:
    binding = _read_immutable(EQUIVALENCE_BINDING, "v2 equivalence binding", "m1_compact_b3s_f1_s42_v2_equivalence_binding_v1", "PASS_M1_COMPACT_B3S_F1_S42_V2_EQUIVALENCE_BOUND")
    equivalence = _read_immutable(EQUIVALENCE_RECEIPT, "source-fit equivalence", "m1_compact_b3s_f1_s42_source_fit_equivalence_v1", "PASS_M1_COMPACT_B3S_F1_S42_SOURCE_FIT_EQUIVALENT")
    if binding.get("parent_v2_receipt") != {"path": str(receipt_path.resolve()), "sha256": v1.sha(receipt_path)}:
        raise RuntimeError("equivalence binding parent receipt drift")
    if binding.get("equivalence_receipt") != {"path": str(EQUIVALENCE_RECEIPT.resolve()), "sha256": v1.sha(EQUIVALENCE_RECEIPT)}:
        raise RuntimeError("equivalence binding receipt drift")
    if not binding.get("equivalence", {}).get("train_dataset_equal") or not binding.get("equivalence", {}).get("sampler_equal") or not binding.get("equivalence", {}).get("zero4_side_exact"):
        raise RuntimeError("source-fit equivalence gate did not pass")
    if binding.get("equivalence", {}).get("target_nwb_opened") is not False:
        raise RuntimeError("source-fit equivalence opened target NWB")
    return {"path": str(EQUIVALENCE_BINDING.resolve()), "sha256": v1.sha(EQUIVALENCE_BINDING), "equivalence": binding["equivalence"], "source_fit_receipt": {"path": str(EQUIVALENCE_RECEIPT.resolve()), "sha256": v1.sha(EQUIVALENCE_RECEIPT)}, "source_fit_canonical": equivalence["canonical_content_sha256"]}


def build_proposal() -> dict[str, Any]:
    source = _v1()
    cells = [cell for cell in source["folds1_2_cross_session"]["cells"] if cell.get("fold") == 1]
    cells.sort(key=lambda cell: (0 if cell.get("arm") == "b0" else 1))
    if [cell.get("arm") for cell in cells] != ["b0", "b3s_zero4"]:
        raise RuntimeError("v1 fold1 cell order drift")
    v2_cells = []
    for cell in cells:
        command = list(cell["argv"])
        if "test=true" not in command:
            raise RuntimeError("v1 fold1 command does not contain explicit test=true to replace")
        command[command.index("test=true")] = "test=false"
        # The source-only subclass is an explicit data-path binding.  Merely
        # setting test=false leaves the legacy setup('fit') path free to
        # prepare the outer target; this override removes that ambiguity.
        command.insert(command.index("test=false"), f"data._target_={SOURCE_ONLY_TARGET}")
        command.insert(command.index("test=false") + 1, "optimized_metric=null")
        v2_cells.append({**cell, "argv": command})
    body = {
        "schema": V2_SCHEMA,
        "status": V2_STATUS,
        "parent_v1": {"path": str(V1_PROPOSAL.resolve()), "sha256": V1_PROPOSAL_SHA},
        "scope": {"task": "m1", "fold": 1, "seed": 42, "outer_target": TARGET_SESSION, "source_sessions": list(SOURCE_SESSIONS), "support_trials": [0, 10], "query_trials": [10, 210], "formal_or_minival_or_heldout": False, "target_backward_steps": 0, "target_optimizer_steps": 0, "target_checkpoint_selection": False},
        "cells": v2_cells,
        "execution": {"fixed_order": ["b0", "b3s_zero4"], "train_commands_test_false": True, "source_only_fit_data_module": SOURCE_ONLY_TARGET, "target_is_opened_only_by_independent_evaluator_after_both_terminal_checkpoints": True, "intermediate_target_metric_read": False, "launched": False},
        "gate": {"metric": "B3S-Zero4 minus B0 pooled variance-weighted R2", "threshold": -0.03, "stop_if_delta_below_threshold": True, "fold2_authorized_only_after_fold1_pass": True},
    }
    return body


def build_receipt() -> dict[str, Any]:
    proposal = build_proposal()
    if not V2_PROPOSAL.exists():
        _immutable(V2_PROPOSAL, proposal)
    on_disk_proposal = _read_immutable(V2_PROPOSAL, "v2 proposal", V2_SCHEMA, V2_STATUS)
    _validate_v2_proposal(on_disk_proposal)
    if v1.canonical(proposal) != on_disk_proposal["canonical_content_sha256"]:
        raise RuntimeError("v2 proposal differs from append-only source")
    gate = v1.validate_gate()
    teacher = v1.validate_teacher()
    preflight = v1.validate_preflight()
    inventory = source_only_inventory()
    commands = {"b0": on_disk_proposal["cells"][0]["argv"], "b3s_zero4": on_disk_proposal["cells"][1]["argv"]}
    if any("test=true" in command or "test=false" not in command for command in commands.values()):
        raise RuntimeError("v2 command is not test=false")
    artifact = ROOT / "outputs/streaming_calibration"
    for base in ("m1_compact_b0_f1_s42_fresh_e11", "m1_compact_b3s_zero4_f1_s42_fresh_e11"):
        if v1.prior_artifact_exists(artifact, base):
            raise RuntimeError(f"prior fold1 artifact exists: {base}")
    body = {
        "schema": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "proposal": {"path": str(V2_PROPOSAL.resolve()), "sha256": v1.sha(V2_PROPOSAL), "schema": V2_SCHEMA},
        "parent_v1_proposal": {"path": str(V1_PROPOSAL.resolve()), "sha256": V1_PROPOSAL_SHA},
        "gate": gate,
        "teacher": teacher,
        "preflight": preflight,
        "scope": proposal["scope"],
        "commands": commands,
        "inventory": inventory,
        "execution": on_disk_proposal["execution"],
        "target_policy": {"train_test_flag": False, "target_opened_by_training": False, "fit_data_module_source_only": True, "target_opened_by_evaluator_after_both_terminal": True, "intermediate_target_metric_read": False},
    }
    if V2_RECEIPT.exists():
        existing = _read_immutable(V2_RECEIPT, "v2 staged receipt", RECEIPT_SCHEMA, RECEIPT_STATUS)
        if v1.canonical(body) != existing["canonical_content_sha256"]:
            raise RuntimeError("existing v2 receipt differs; refusing overwrite")
        return existing
    _immutable(V2_RECEIPT, body)
    return _read_immutable(V2_RECEIPT, "v2 staged receipt", RECEIPT_SCHEMA, RECEIPT_STATUS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", action="store_true")
    parser.add_argument("--receipt", action="store_true")
    args = parser.parse_args()
    if args.proposal == args.receipt:
        raise SystemExit("choose exactly one --proposal/--receipt")
    if args.proposal:
        body = build_proposal()
        if not V2_PROPOSAL.exists():
            digest = _immutable(V2_PROPOSAL, body)
        else:
            stored = _read_immutable(V2_PROPOSAL, "v2 proposal", V2_SCHEMA, V2_STATUS)
            _validate_v2_proposal(stored)
            if v1.canonical(body) != stored["canonical_content_sha256"]:
                raise RuntimeError("existing v2 proposal differs; refusing overwrite")
            digest = v1.sha(V2_PROPOSAL)
        print(json.dumps({"path": str(V2_PROPOSAL.resolve()), "sha256": digest, "status": V2_STATUS}, sort_keys=True))
    else:
        body = build_receipt()
        print(json.dumps({"path": str(V2_RECEIPT.resolve()), "sha256": v1.sha(V2_RECEIPT), "status": body["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
