#!/usr/bin/env python3
"""Fail-closed continuation and receipt-only aggregation for RT R-RS/R-LS.

The two control families use the already sealed R-C ``afc4_vel`` matrix as
their scientific comparator.  They deliberately do *not* wait for the
independent B2 R-S program.  A later GPU invocation is limited to one
``(arm, fold)`` cell, checks that the requested GPU has no active R-S writer,
uses an output root outside every active Stage-R R-S root, and revalidates the
clean nested-LOSO contract before and after the inherited fit/evaluate worker.

Default ``--preview`` is receipt-only/read-only.  It never imports a data
module, opens an NWB, builds a Trainer, starts CUDA, or writes a file.  Raw
remote cells can be inspected in this mode, but they are explicitly *unsealed*
until ``--copy-import`` copies their JSON/config receipts into a new immutable
evidence root.  A final ``--aggregate`` accepts only 15 imported immutable
cells; it cannot turn a partial preview into a claim.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
CONTROL_ARMS = ("afc4_rs", "afc4_ls")
FOLDS = tuple(range(15))
SEED = 42
M = 24
QUERY_START = 24
WINDOW_SIZE = 50
R_C_SCHEMA = "rt_seed42_clean_nested_loso_aggregate_v1"
R_C_STATUS = "PASS_RT_SEALED"
R_C_SEAL_SCHEMA = "rt_seed42_clean_nested_loso_seal_marker_v1"
IMPORT_SCHEMA = "rt_clean_nested_loso_control_copy_import_v1"
IMPORT_STATUS = "PASS_RT_CLEAN_CONTROL_RECEIPT_COPY_IMPORTED"
AGGREGATE_SCHEMA = "rt_clean_nested_loso_control_paired_aggregate_v1"
AGGREGATE_STATUS = "PASS_RT_CLEAN_CONTROL_ALL_15_PAIRED"
PREVIEW_STATUS = "READ_ONLY_RT_CONTROL_PARTIAL_PREVIEW_NOT_PAPER_CLAIM"
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED_BY_ARM = {"afc4_rs": 202608081, "afc4_ls": 202608082}
DEFAULT_R_C_AGGREGATE = WORKSPACE / "sua_exploration/results/k4_rt_loso_v1/rt_seed42_clean_nested_aggregate.json"
DEFAULT_R_C_SEAL = WORKSPACE / "sua_exploration/results/k4_rt_loso_v1/rt_seed42_clean_nested_seal.marker"
DEFAULT_WORK_ROOT = PROJECT / "outputs/rt_controls_continuation"


class ControlContinuationError(RuntimeError):
    """A control receipt or launch-safety invariant failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ControlContinuationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    _need(path.is_file(), f"missing JSON receipt: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ControlContinuationError(f"invalid JSON receipt: {path}") from error
    _need(isinstance(value, dict), f"JSON receipt must be an object: {path}")
    return value


def _immutable(path: Path, label: str) -> None:
    _need(path.is_file(), f"{label} missing: {path}")
    _need(stat.S_IMODE(path.stat().st_mode) == 0o444, f"{label} must be mode 0444: {path}")


def _write_immutable_json(path: Path, body: Mapping[str, Any]) -> str:
    """Exclusive atomic-ish receipt write; never replace prior evidence."""

    _need(not path.exists(), f"refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(body), indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    path.chmod(0o444)
    return _sha256(path)


def _finite(value: Any, label: str) -> float:
    _need(isinstance(value, (int, float)) and math.isfinite(float(value)), f"{label} must be finite")
    return float(value)


@dataclass(frozen=True)
class RcReference:
    rows: Mapping[int, Mapping[str, Any]]
    aggregate_path: Path
    aggregate_sha256: str
    seal_path: Path
    seal_sha256: str


@dataclass(frozen=True)
class CellPaths:
    cell: Path
    outer: Path
    selection: Path
    split: Path
    config: Path
    terminal: Path
    imported: Path


def cell_paths(control_root: Path, arm: str, fold: int) -> CellPaths:
    _need(arm in CONTROL_ARMS, f"unsupported control arm: {arm}")
    _need(fold in FOLDS, f"fold must be in [0,14], got {fold}")
    cell = control_root.resolve() / arm / f"fold_{fold:02d}" / f"seed_{SEED}"
    return CellPaths(
        cell=cell,
        outer=cell / "outer_target_eval.json",
        selection=cell / "fit/rt_nested_selection_receipt.json",
        split=cell / "fit/split_manifest.json",
        config=cell / "fit/.hydra/config.yaml",
        terminal=cell / "cell_terminal.json",
        imported=cell / "control_copy_import.json",
    )


def _load_rc_reference(*, aggregate_path: Path, seal_path: Path) -> RcReference:
    """Read the frozen R-C aggregate only; no RT dataset is imported."""

    aggregate_path, seal_path = aggregate_path.resolve(), seal_path.resolve()
    _immutable(aggregate_path, "frozen R-C aggregate")
    _immutable(seal_path, "frozen R-C seal")
    aggregate, seal = _json(aggregate_path), _json(seal_path)
    _need(aggregate.get("schema") == R_C_SCHEMA and aggregate.get("status") == R_C_STATUS,
          "R-C aggregate schema/status drift")
    _need(seal.get("schema") == R_C_SEAL_SCHEMA and seal.get("status") == R_C_STATUS,
          "R-C seal schema/status drift")
    _need(seal.get("aggregate_sha256") == _sha256(aggregate_path), "R-C seal aggregate SHA mismatch")
    _need(aggregate.get("task") == "rt" and aggregate.get("seed") == SEED,
          "R-C aggregate task/seed drift")
    audits = aggregate.get("audits")
    _need(isinstance(audits, Mapping) and audits.get("exact_main_grid") is True,
          "R-C aggregate lacks exact main-grid audit")
    for field in (
        "all_model_state_unchanged", "all_optimizer_absent", "all_target_backpropagation_false",
        "all_target_loaded_during_fit_false", "all_target_query_labels_read_during_fit_false",
        "all_target_query_labels_used_for_calibration_false",
        "all_target_query_labels_used_for_normalization_false",
        "all_target_query_labels_used_for_checkpoint_selection_false",
        "all_target_query_labels_used_for_scoring_only_true",
    ):
        _need(audits.get(field) is True, f"R-C frozen audit failed: {field}")
    rows: dict[int, Mapping[str, Any]] = {}
    for row in aggregate.get("cells", []):
        if not isinstance(row, Mapping) or row.get("arm") != "afc4_vel":
            continue
        fold = row.get("fold")
        _need(isinstance(fold, int) and fold in FOLDS and fold not in rows,
              "R-C aggregate lacks one unique afc4_vel row per fold")
        _need(row.get("query_start_trial") == QUERY_START and row.get("window_size") == WINDOW_SIZE,
              f"R-C fold {fold} M24/q24/window contract drift")
        _need(isinstance(row.get("target_session"), str) and isinstance(row.get("inner_validation_session"), str),
              f"R-C fold {fold} session binding absent")
        _need(row["target_session"] != row["inner_validation_session"],
              f"R-C fold {fold} inner validation reuses target")
        _finite(row.get("r2_variance_weighted"), f"R-C fold {fold} R2")
        rows[fold] = row
    _need(tuple(sorted(rows)) == FOLDS, "R-C aggregate does not provide all 15 afc4_vel rows")
    return RcReference(rows=rows, aggregate_path=aggregate_path, aggregate_sha256=_sha256(aggregate_path),
                       seal_path=seal_path, seal_sha256=_sha256(seal_path))


def _validate_selection(selection: Mapping[str, Any], *, arm: str, fold: int) -> None:
    expected = {
        "schema": "rt_clean_nested_loso_selection_receipt_v1",
        "status": "PASS_FIT_INNER_SELECTION_ONLY",
        "arm": arm,
        "outer_loso_fold": fold,
        "seed": SEED,
        "selected_by_metric": "val_heldin/r2_mean",
        "selected_metric_scope": "inner_validation_session_only",
        "formal_heldout_opened": False,
        "outer_target_loaded_during_fit": False,
        "outer_target_query_labels_read_during_fit": False,
    }
    for field, value in expected.items():
        _need(selection.get(field) == value, f"{arm} fold {fold} selection drift: {field}")
    for field in ("best_model_sha256", "config_sha256", "split_manifest_sha256"):
        value = selection.get(field)
        _need(isinstance(value, str) and len(value) == 64, f"{arm} fold {fold} selection SHA missing: {field}")


def _validate_split(split: Mapping[str, Any], *, arm: str, fold: int, rc: Mapping[str, Any]) -> None:
    _need(split.get("task") == "rt" and split.get("development_only") is True,
          f"{arm} fold {fold} split task/scope drift")
    _need(split.get("validation_protocol") == "nested_loso", f"{arm} fold {fold} not nested LOSO")
    _need(split.get("outer_loso_fold") == fold and split.get("loso_fold") == fold,
          f"{arm} fold {fold} split fold drift")
    _need(split.get("requested_side_feature_group") == arm, f"{arm} fold {fold} split arm drift")
    split_arm = split.get("arm")
    _need(isinstance(split_arm, Mapping) and split_arm.get("canonical_arm") == arm,
          f"{arm} fold {fold} canonical arm drift")
    _need(split.get("target_session") == rc.get("target_session") and
          split.get("inner_validation_session") == rc.get("inner_validation_session"),
          f"{arm} fold {fold} does not match frozen R-C target/inner-validation pair")
    protocol, calibration, query, nested = (split.get(key) for key in ("protocol", "calibration", "query", "nested_selection"))
    _need(all(isinstance(value, Mapping) for value in (protocol, calibration, query, nested)),
          f"{arm} fold {fold} split sections absent")
    _need(protocol.get("signal_type") == "sorted_SUA" and protocol.get("decode_target") == "2D cursor velocity",
          f"{arm} fold {fold} signal/target drift")
    _need(calibration.get("budget_trials") == M and calibration.get("trial_index_range") == [0, M] and
          calibration.get("target_calibration_optimizer_steps") == 0,
          f"{arm} fold {fold} M24 calibration contract drift")
    _need(query.get("query_start_trial") == QUERY_START and query.get("window_size_bins") == WINDOW_SIZE and
          query.get("full_window_after_support_required") is True,
          f"{arm} fold {fold} q24/window contract drift")
    for field, value in (
        ("clean", True), ("outer_target_loaded_during_fit", False),
        ("outer_target_query_labels_read_during_fit", False),
        ("inner_validation_only_for_checkpoint_selection", True),
        ("checkpoint_metric", "val_heldin/r2_mean"),
        ("checkpoint_metric_scope", "inner_validation_session_only"),
    ):
        _need(nested.get(field) == value, f"{arm} fold {fold} nested-selection drift: {field}")
    normalizer = split.get("source_only_normalizer")
    _need(isinstance(normalizer, Mapping) and normalizer.get("fit_scope") == "inner_train_sessions_only" and
          normalizer.get("feature_group") == arm and normalizer.get("excluded_outer_target_session") == rc.get("target_session"),
          f"{arm} fold {fold} source-only normalizer contract drift")


def _validate_outer(outer: Mapping[str, Any], *, arm: str, fold: int, rc: Mapping[str, Any]) -> float:
    expected = {
        "schema": "rt_clean_nested_loso_outer_eval_v1",
        "status": "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP",
        "arm": arm,
        "outer_loso_fold": fold,
        "seed": SEED,
        "query_start_trial": QUERY_START,
        "window_size": WINDOW_SIZE,
        "outer_target_session": rc.get("target_session"),
        "query_windows_evaluated": rc.get("query_windows_evaluated"),
        "normalizer_fit_scope": "inner_train_sessions_only",
        "target_backpropagation": False,
        "optimizer_present": False,
        "model_training_mode": False,
        "model_state_unchanged": True,
        "target_query_labels_used_for_calibration": False,
        "target_query_labels_used_for_normalization": False,
        "target_query_labels_used_for_checkpoint_selection": False,
        "target_query_labels_used_for_scoring_only": True,
        "target_support_calibration_labels_used": True,
        "target_support_calibration_velocity_used": True,
    }
    for field, value in expected.items():
        _need(outer.get(field) == value, f"{arm} fold {fold} outer contract drift: {field}")
    _need(outer.get("model_state_sha256_before") == outer.get("model_state_sha256_after"),
          f"{arm} fold {fold} target eval changed model state")
    return _finite(outer.get("r2_variance_weighted"), f"{arm} fold {fold} R2")


def validate_control_cell(*, control_root: Path, arm: str, fold: int, reference: RcReference,
                          require_import: bool) -> dict[str, Any]:
    """Validate one receipt bundle without importing a data loader or checkpoint."""

    paths = cell_paths(control_root, arm, fold)
    required = (paths.outer, paths.selection, paths.split, paths.config, paths.terminal)
    exists = [path.exists() for path in required]
    if not any(exists):
        return {"state": "missing", "arm": arm, "fold": fold, "paths": {"cell": str(paths.cell)}}
    _need(all(exists), f"{arm} fold {fold} is partial, not comparable: {[str(path) for path, item in zip(required, exists) if not item]}")
    outer, selection, split = _json(paths.outer), _json(paths.selection), _json(paths.split)
    _validate_selection(selection, arm=arm, fold=fold)
    _need(_sha256(paths.config) == selection.get("config_sha256"), f"{arm} fold {fold} config SHA mismatch")
    _need(_sha256(paths.split) == selection.get("split_manifest_sha256"), f"{arm} fold {fold} split SHA mismatch")
    _need(outer.get("checkpoint_sha256") == selection.get("best_model_sha256"),
          f"{arm} fold {fold} selected checkpoint SHA mismatch")
    _validate_split(split, arm=arm, fold=fold, rc=reference.rows[fold])
    r2 = _validate_outer(outer, arm=arm, fold=fold, rc=reference.rows[fold])
    if paths.terminal.exists():
        terminal = _json(paths.terminal)
        _need(terminal.get("status") == "PASS" and terminal.get("arm") == arm and terminal.get("fold") == fold and
              terminal.get("seed") == SEED, f"{arm} fold {fold} cell terminal drift")
    imported_sha: str | None = None
    if require_import:
        _immutable(paths.terminal, f"{arm} fold {fold} imported cell terminal")
        terminal = _json(paths.terminal)
        _need(terminal.get("status") == "PASS" and terminal.get("arm") == arm and terminal.get("fold") == fold and
              terminal.get("seed") == SEED, f"{arm} fold {fold} imported cell terminal drift")
        _immutable(paths.imported, f"{arm} fold {fold} copy-import receipt")
        imported = _json(paths.imported)
        _need(imported.get("schema") == IMPORT_SCHEMA and imported.get("status") == IMPORT_STATUS and
              imported.get("arm") == arm and imported.get("fold") == fold and imported.get("seed") == SEED,
              f"{arm} fold {fold} copy-import receipt drift")
        files = imported.get("imported_files")
        _need(isinstance(files, Mapping), f"{arm} fold {fold} import file map missing")
        for name, path in (("outer", paths.outer), ("selection", paths.selection), ("split", paths.split),
                           ("config", paths.config), ("cell_terminal", paths.terminal)):
            item = files.get(name)
            _need(isinstance(item, Mapping) and item.get("sha256") == _sha256(path),
                  f"{arm} fold {fold} immutable import SHA mismatch: {name}")
        imported_sha = _sha256(paths.imported)
    file_map = {name: {"path": str(path), "sha256": _sha256(path)} for name, path in (
        ("outer", paths.outer), ("selection", paths.selection), ("split", paths.split), ("config", paths.config),
    )}
    if paths.terminal.exists():
        file_map["cell_terminal"] = {"path": str(paths.terminal), "sha256": _sha256(paths.terminal)}
    return {
        "state": "complete", "arm": arm, "fold": fold, "r2": r2,
        "rc_r2": float(reference.rows[fold]["r2_variance_weighted"]),
        "rc_minus_control": float(reference.rows[fold]["r2_variance_weighted"]) - r2,
        "files": file_map, "copy_import_sha256": imported_sha,
    }


def inventory(*, control_root: Path, aggregate_path: Path, seal_path: Path, require_import: bool) -> dict[str, Any]:
    reference = _load_rc_reference(aggregate_path=aggregate_path, seal_path=seal_path)
    arms: dict[str, Any] = {}
    for arm in CONTROL_ARMS:
        rows: list[dict[str, Any]] = []
        for fold in FOLDS:
            try:
                rows.append(validate_control_cell(control_root=control_root, arm=arm, fold=fold,
                                                  reference=reference, require_import=require_import))
            except ControlContinuationError as error:
                rows.append({"state": "incomparable", "arm": arm, "fold": fold, "reason": str(error)})
        complete = [row for row in rows if row["state"] == "complete"]
        arms[arm] = {
            "rows": rows,
            "complete_folds": [row["fold"] for row in complete],
            "missing_folds": [row["fold"] for row in rows if row["state"] == "missing"],
            "incomparable_folds": [row["fold"] for row in rows if row["state"] == "incomparable"],
            "completed_count": len(complete),
            "mean_rc_minus_control": None if not complete else sum(row["rc_minus_control"] for row in complete) / len(complete),
            "positive_rc_minus_control_folds": sum(row["rc_minus_control"] > 0.0 for row in complete),
        }
    return {
        "schema": "rt_clean_nested_loso_control_inventory_v1",
        "status": PREVIEW_STATUS,
        "scope": "receipt_only_no_nwb_no_trainer_no_cuda",
        "control_root": str(control_root.resolve()),
        "require_import": require_import,
        "r_c_reference": {
            "aggregate": {"path": str(reference.aggregate_path), "sha256": reference.aggregate_sha256},
            "seal": {"path": str(reference.seal_path), "sha256": reference.seal_sha256},
        },
        "arms": arms,
        "final_aggregate_eligible": all(arms[arm]["completed_count"] == len(FOLDS) and not arms[arm]["incomparable_folds"]
                                        for arm in CONTROL_ARMS) and require_import,
    }


def copy_import(*, source_root: Path, import_root: Path, arm: str, fold: int,
                aggregate_path: Path, seal_path: Path) -> dict[str, Any]:
    """Copy a validated raw cell into a new immutable evidence location.

    The source 0644 remote records are never chmodded, modified, or moved.
    Their original paths and byte SHA-256 values are recorded in the imported
    terminal receipt.
    """

    src, dst = cell_paths(source_root, arm, fold), cell_paths(import_root, arm, fold)
    _need(src.terminal.is_file(), f"cannot import {arm} fold {fold} without a valid cell terminal")
    reference = _load_rc_reference(aggregate_path=aggregate_path, seal_path=seal_path)
    source = validate_control_cell(control_root=source_root, arm=arm, fold=fold,
                                   reference=reference, require_import=False)
    _need(source["state"] == "complete", f"cannot import non-complete {arm} fold {fold}")
    source_terminal = _json(src.terminal)
    _need(source_terminal.get("status") == "PASS" and source_terminal.get("arm") == arm and
          source_terminal.get("fold") == fold and source_terminal.get("seed") == SEED,
          f"cannot import invalid {arm} fold {fold} cell terminal")
    _need(not dst.cell.exists(), f"refusing to overwrite import destination: {dst.cell}")
    dst.cell.mkdir(parents=True, exist_ok=False)
    copied: dict[str, dict[str, str]] = {}
    for name, source_path, destination_path in (
        ("outer", src.outer, dst.outer), ("selection", src.selection, dst.selection),
        ("split", src.split, dst.split), ("config", src.config, dst.config),
    ):
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        source_sha = _sha256(source_path)
        shutil.copyfile(source_path, destination_path)
        _need(_sha256(destination_path) == source_sha, f"copy SHA mismatch for {name}")
        destination_path.chmod(0o444)
        copied[name] = {"source_path": str(source_path), "source_sha256": source_sha,
                        "path": str(destination_path), "sha256": _sha256(destination_path)}
    shutil.copyfile(src.terminal, dst.terminal)
    _need(_sha256(src.terminal) == _sha256(dst.terminal), "copy SHA mismatch for cell terminal")
    dst.terminal.chmod(0o444)
    copied["cell_terminal"] = {"source_path": str(src.terminal), "source_sha256": _sha256(src.terminal),
                               "path": str(dst.terminal), "sha256": _sha256(dst.terminal)}
    receipt = {
        "schema": IMPORT_SCHEMA, "status": IMPORT_STATUS, "arm": arm, "fold": fold, "seed": SEED,
        "source_root": str(source_root.resolve()), "source_cell": str(src.cell),
        "imported_cell": str(dst.cell), "imported_files": copied,
        "r_c_reference": {"aggregate_sha256": reference.aggregate_sha256, "seal_sha256": reference.seal_sha256},
        "scope": "receipt_copy_only_no_nwb_no_trainer_no_cuda_no_source_mutation",
    }
    receipt_sha = _write_immutable_json(dst.imported, receipt)
    validated = validate_control_cell(control_root=import_root, arm=arm, fold=fold,
                                      reference=reference, require_import=True)
    return {"status": IMPORT_STATUS, "receipt": str(dst.imported), "receipt_sha256": receipt_sha,
            "validated": validated}


def _same_fold_rt_writer_commands(commands: Sequence[str], *, fold: int) -> list[str]:
    token = f"data.loso_fold={fold}"
    return [command for command in commands if "rt_clean_nested" in command.lower() and token in command]


def _ensure_launch_safe(*, gpu_commands: Sequence[str], all_rt_commands: Sequence[str], fold: int, gpu: int) -> None:
    """Fail closed on *any* requested-GPU compute owner or same-fold RT writer.

    ``gpu_commands`` is scoped by nvidia-smi to the requested GPU.  It is not
    filtered to R-S: an unrelated or unreadable PID is equally unsafe.  The
    global process scan is intentionally narrower and only blocks an exact
    same-fold RT writer, allowing GPU0 controls alongside R-S on GPU1.
    """

    _need(not gpu_commands, f"requested GPU {gpu} is occupied by compute PID(s): {list(gpu_commands)}")
    collisions = _same_fold_rt_writer_commands(all_rt_commands, fold=fold)
    _need(not collisions, f"active same-fold RT writer blocks control fold {fold}: {collisions}")


def _active_compute_commands_on_gpu(gpu: int) -> tuple[list[str], list[str]]:
    """Return requested-GPU and global RT command lines; fail closed if unobservable."""

    try:
        gpu_table = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True,
        )
        uuid_by_index = {int(line.split(",", 1)[0].strip()): line.split(",", 1)[1].strip()
                         for line in gpu_table.splitlines() if "," in line}
        _need(gpu in uuid_by_index, f"requested GPU index is not visible to nvidia-smi: {gpu}")
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ControlContinuationError("cannot verify GPU writer safety; refusing execution") from error
    pids = []
    for line in apps.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[1] == uuid_by_index[gpu] and fields[0].isdigit():
            pids.append(int(fields[0]))
    commands: list[str] = []
    for pid in pids:
        try:
            commands.append(Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8"))
        except OSError:
            commands.append(f"<unreadable-gpu-pid:{pid}>")
    # A same-fold collision can be CPU-side before CUDA initialisation, so scan
    # process arguments in addition to the GPU ownership table.
    try:
        ps_table = subprocess.check_output(["ps", "-eo", "args="], text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ControlContinuationError("cannot verify same-fold writer safety; refusing execution") from error
    commands.extend(line for line in ps_table.splitlines() if "rt_clean_nested" in line)
    return commands[:len(pids)], [line for line in ps_table.splitlines() if "rt_clean_nested" in line]


def _ensure_independent_output_root(run_root: Path) -> Path:
    run_root = run_root.resolve()
    active_roots = (
        PROJECT / "outputs/rt_stage_r_b2_local3090",
        PROJECT / "outputs/rt_stage_r_b2_imported_remote",
        PROJECT / "outputs/rt_stage_r_b2_remote",
    )
    for active in active_roots:
        try:
            run_root.relative_to(active.resolve())
        except ValueError:
            continue
        raise ControlContinuationError(f"control continuation output may not touch active R-S root: {run_root}")
    return run_root


def build_execution_plan(*, work_root: Path, arm: str, fold: int, gpu: int,
                         aggregate_path: Path, seal_path: Path) -> dict[str, Any]:
    """Target-free plan construction; it does not inspect process/GPU state."""

    reference = _load_rc_reference(aggregate_path=aggregate_path, seal_path=seal_path)
    work_root = _ensure_independent_output_root(work_root)
    paths = cell_paths(work_root, arm, fold)
    _need(not paths.cell.exists(), f"refusing overwrite/resume for control cell: {paths.cell}")
    try:
        from scripts.run_rt_clean_nested_loso import _eval_command, _train_command
    except ModuleNotFoundError:  # pragma: no cover - direct script execution
        from run_rt_clean_nested_loso import _eval_command, _train_command
    train_args = argparse.Namespace(fold=fold, arm=arm, seed=SEED, accelerator="gpu", devices=1)
    train = _train_command(train_args)
    train.extend([f"hydra.run.dir={paths.cell / 'fit'}", f"paths.root_dir={PROJECT}",
                  f"paths.log_dir={work_root / '_hydra_logs'}", f"paths.artifact_dir={work_root / '_artifacts'}"])
    _need("experiment=rt_clean_nested_loso_m24" in train and f"data.side_feature_group={arm}" in train and
          f"data.loso_fold={fold}" in train and f"data.outer_loso_fold={fold}" in train and "seed=42" in train and
          "test=false" in train, "continuation train command lost clean nested control bindings")
    selected_checkpoint_placeholder = paths.cell / "fit/checkpoints/<selected_by_inner_receipt>.ckpt"
    eval_args = argparse.Namespace(config=paths.config, checkpoint=selected_checkpoint_placeholder, split_manifest=paths.split,
                                   selection_receipt=paths.selection, output=paths.outer, outer_fold=fold, device="cuda")
    eval_preview = _eval_command(eval_args)
    return {
        "schema": "rt_clean_nested_loso_control_continuation_plan_v1",
        "status": "PLAN_ONLY_NO_GPU_NO_NWB_NO_TRAINER",
        "arm": arm, "fold": fold, "seed": SEED, "gpu": int(gpu), "work_root": str(work_root),
        "cell": str(paths.cell),
        "r_c_reference": {"aggregate_sha256": reference.aggregate_sha256, "seal_sha256": reference.seal_sha256,
                            "target_session": reference.rows[fold]["target_session"],
                            "inner_validation_session": reference.rows[fold]["inner_validation_session"],
                            "query_windows_evaluated": reference.rows[fold]["query_windows_evaluated"]},
        "train_command": train,
        "outer_eval_contract": {"command_preview": eval_preview, "one_shot": True, "target_backpropagation": False},
        "launch_gate": "R-C aggregate+seal valid; independent output root; requested GPU has no compute PID; no global same-fold RT writer",
    }


def execute_one(*, work_root: Path, arm: str, fold: int, gpu: int,
                aggregate_path: Path, seal_path: Path) -> dict[str, Any]:
    """Explicit one-cell execution path.  Never called by preview/import/aggregate."""

    plan = build_execution_plan(work_root=work_root, arm=arm, fold=fold, gpu=gpu,
                                aggregate_path=aggregate_path, seal_path=seal_path)
    gpu_commands, all_rt_commands = _active_compute_commands_on_gpu(gpu)
    _ensure_launch_safe(gpu_commands=gpu_commands, all_rt_commands=all_rt_commands, fold=fold, gpu=gpu)
    try:
        from scripts.run_rt_clean_nested_loso_remote_missing import run_cell
    except ModuleNotFoundError:  # pragma: no cover - direct script execution
        from run_rt_clean_nested_loso_remote_missing import run_cell
    status = run_cell(project=PROJECT, run_root=Path(plan["work_root"]), arm=arm, fold=fold, seed=SEED, gpu=gpu)
    _need(status == "passed", f"control worker did not pass: {status}")
    reference = _load_rc_reference(aggregate_path=aggregate_path, seal_path=seal_path)
    validated = validate_control_cell(control_root=Path(plan["work_root"]), arm=arm, fold=fold,
                                      reference=reference, require_import=False)
    _need(validated["state"] == "complete", "control worker returned pass but receipt validation failed")
    return {"status": "PASS_RT_CONTROL_ONE_CELL", "plan": plan, "validated": validated}


def run_queue(*, work_root: Path, arm: str, folds: Sequence[int], gpu: int,
              aggregate_path: Path, seal_path: Path) -> dict[str, Any]:
    """Run only the explicitly listed control cells, serially and fail-stop.

    This function creates no background process.  The caller owns the tmux
    pane.  Each call to :func:`execute_one` independently rechecks the GPU and
    same-fold process safety immediately before its cell begins.
    """

    requested = tuple(int(fold) for fold in folds)
    _need(bool(requested) and len(set(requested)) == len(requested) and all(fold in FOLDS for fold in requested),
          "--run-queue folds must be a nonempty unique subset of [0,14]")
    _need(arm in CONTROL_ARMS, f"unsupported control arm: {arm}")
    result: list[dict[str, Any]] = []
    for fold in requested:
        # No exception handler: any cell failure stops the external queue pane
        # before another listed fold is considered.
        result.append(execute_one(work_root=work_root, arm=arm, fold=fold, gpu=gpu,
                                  aggregate_path=aggregate_path, seal_path=seal_path))
    return {"status": "PASS_RT_CONTROL_EXPLICIT_SERIAL_QUEUE", "arm": arm, "folds": list(requested),
            "seed": SEED, "gpu": gpu, "cells": result,
            "scope": "explicit_cli_folds_only_no_cross_arm_no_background"}


def _exact_two_sided_sign_test(values: Sequence[float]) -> dict[str, Any]:
    signs = [1 if value > 0.0 else -1 if value < 0.0 else 0 for value in values]
    positive, negative, zero = signs.count(1), signs.count(-1), signs.count(0)
    nonzero = positive + negative
    tail = sum(math.comb(nonzero, index) for index in range(min(positive, negative) + 1)) / float(2 ** nonzero) if nonzero else 1.0
    return {"test": "exact_two_sided_binomial_sign_test", "positive": positive, "negative": negative,
            "zero": zero, "nonzero_pairs": nonzero, "p_value": min(1.0, 2.0 * tail),
            "ties": "excluded_from_sign_test"}


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    _need(bool(sorted_values) and 0.0 <= probability <= 1.0, "invalid bootstrap quantile input")
    position = (len(sorted_values) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(sorted_values[low])
    fraction = position - low
    return float(sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction)


def _paired_bootstrap_ci(values: Sequence[float], *, seed: int, draws: int = BOOTSTRAP_DRAWS) -> dict[str, Any]:
    _need(bool(values) and draws > 0, "paired bootstrap requires finite nonempty values")
    numbers = tuple(_finite(value, "paired bootstrap delta") for value in values)
    rng = random.Random(seed)
    count = len(numbers)
    means = sorted(sum(numbers[rng.randrange(count)] for _ in range(count)) / count for _ in range(draws))
    return {"resampling_unit": "outer_LOSO_fold", "draws": draws, "rng_seed": seed,
            "quantile_method": "linear_order_statistic", "lower_95": _quantile(means, 0.025),
            "upper_95": _quantile(means, 0.975)}


def aggregate(*, import_root: Path, aggregate_path: Path, seal_path: Path, output: Path) -> dict[str, Any]:
    """Seal one full 15-fold imported control matrix; partial inputs are refused."""

    _need(not output.exists(), f"refusing to overwrite aggregate: {output}")
    result = inventory(control_root=import_root, aggregate_path=aggregate_path, seal_path=seal_path, require_import=True)
    _need(result["final_aggregate_eligible"] is True, "all 15 immutable imported cells per arm are required")
    reference = _load_rc_reference(aggregate_path=aggregate_path, seal_path=seal_path)
    arms: dict[str, Any] = {}
    for arm, summary in result["arms"].items():
        rows = summary["rows"]
        deltas = [row["rc_minus_control"] for row in rows]
        sorted_deltas = sorted(deltas)
        arms[arm] = {
            "mean_control_r2": sum(row["r2"] for row in rows) / len(rows),
            "mean_rc_minus_control": sum(deltas) / len(deltas),
            "median_rc_minus_control": _quantile(sorted_deltas, 0.5),
            "positive_rc_minus_control_folds": sum(value > 0.0 for value in deltas),
            "exact_two_sided_sign_test": _exact_two_sided_sign_test(deltas),
            "paired_fold_bootstrap_95": _paired_bootstrap_ci(deltas, seed=BOOTSTRAP_SEED_BY_ARM[arm]),
            "rows": rows,
        }
    body = {
        "schema": AGGREGATE_SCHEMA, "status": AGGREGATE_STATUS, "seed": SEED, "folds": list(FOLDS),
        "scope": "receipt_only_aggregate_no_nwb_no_trainer_no_cuda", "r_c_reference": result["r_c_reference"],
        "import_root": str(import_root.resolve()), "arms": arms,
        "interpretation": "R-C minus R-RS tests descriptor-row attachment; R-C minus R-LS tests velocity-label pairing.",
    }
    digest = _write_immutable_json(output, body)
    return {"status": AGGREGATE_STATUS, "output": str(output), "sha256": digest,
            "r_c_aggregate_sha256": reference.aggregate_sha256}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="default: receipt-only partial inventory")
    mode.add_argument("--copy-import", action="store_true", help="copy one verified raw cell into a new immutable evidence root")
    mode.add_argument("--execute", action="store_true", help="explicitly run exactly one new control cell")
    mode.add_argument("--run-queue", action="store_true", help="explicitly run only listed missing folds for one arm, serial fail-stop")
    mode.add_argument("--aggregate", action="store_true", help="seal a full imported 15-fold control matrix")
    parser.add_argument("--control-root", type=Path, default=DEFAULT_WORK_ROOT,
                        help="raw control cell root for preview; work root for --execute")
    parser.add_argument("--source-root", type=Path, help="raw source root for --copy-import")
    parser.add_argument("--import-root", type=Path, help="immutable copied evidence root")
    parser.add_argument("--output", type=Path, help="new immutable aggregate output")
    parser.add_argument("--arm", choices=CONTROL_ARMS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--folds", type=int, choices=FOLDS, nargs="+", help="explicit queue folds for --run-queue")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--rc-aggregate", type=Path, default=DEFAULT_R_C_AGGREGATE)
    parser.add_argument("--rc-seal", type=Path, default=DEFAULT_R_C_SEAL)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.copy_import:
        _need(args.source_root is not None and args.import_root is not None and args.arm is not None and args.fold is not None,
              "--copy-import requires --source-root --import-root --arm --fold")
        result = copy_import(source_root=args.source_root, import_root=args.import_root, arm=args.arm, fold=args.fold,
                             aggregate_path=args.rc_aggregate, seal_path=args.rc_seal)
    elif args.execute:
        _need(args.arm is not None and args.fold is not None, "--execute requires --arm --fold")
        result = execute_one(work_root=args.control_root, arm=args.arm, fold=args.fold, gpu=args.gpu,
                             aggregate_path=args.rc_aggregate, seal_path=args.rc_seal)
    elif args.run_queue:
        _need(args.arm is not None and args.folds is not None, "--run-queue requires --arm --folds")
        result = run_queue(work_root=args.control_root, arm=args.arm, folds=args.folds, gpu=args.gpu,
                           aggregate_path=args.rc_aggregate, seal_path=args.rc_seal)
    elif args.aggregate:
        _need(args.import_root is not None and args.output is not None, "--aggregate requires --import-root --output")
        result = aggregate(import_root=args.import_root, aggregate_path=args.rc_aggregate, seal_path=args.rc_seal,
                           output=args.output)
    else:
        result = inventory(control_root=args.control_root, aggregate_path=args.rc_aggregate,
                           seal_path=args.rc_seal, require_import=False)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
