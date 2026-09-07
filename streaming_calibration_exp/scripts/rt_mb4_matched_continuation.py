#!/usr/bin/env python3
"""Versioned, fail-closed RT Full-vs-MB4 matched continuation and finalizer.

This module is deliberately a new orchestration surface.  It does not extend
the active RT runner allow-list, modify the DataModule/model/configuration, or
reuse any R-RS/R-LS cell directory.  It constructs the already-existing clean
nested-LOSO Hydra binding directly for ``afc4_mb4=[0,0,||W||,b]`` and later
accepts only receipt bundles matching the sealed Full (R-C ``afc4_vel``)
matrix.

Default ``--plan`` is CPU-only and never opens NWB, creates a Trainer, starts
CUDA, or writes a cell root.  ``--run-partition`` exists only as a future
explicit worker entrypoint; it is not called by planning/import/finalization.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shlex
import shutil
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
ARM = "afc4_mb4"
FULL_ARM = "afc4_vel"
FOLDS = tuple(range(15))
SEED = 42
M = 24
QUERY_START = 24
WINDOW_SIZE = 50
GPU1_ASC_FOLDS = tuple(range(0, 8))
GPU0_DESC_FOLDS = tuple(range(14, 7, -1))
PARTITIONS = {
    "gpu1_after_asc": {"gpu": 1, "folds": GPU1_ASC_FOLDS, "predecessor_partition": "asc"},
    "gpu0_after_desc": {"gpu": 0, "folds": GPU0_DESC_FOLDS, "predecessor_partition": "desc"},
}
R_C_SCHEMA = "rt_seed42_clean_nested_loso_aggregate_v1"
R_C_STATUS = "PASS_RT_SEALED"
R_C_SEAL_SCHEMA = "rt_seed42_clean_nested_loso_seal_marker_v1"
IMPORT_SCHEMA = "rt_mb4_matched_copy_import_v1"
IMPORT_STATUS = "PASS_RT_MB4_MATCHED_COPY_IMPORTED"
AGGREGATE_SCHEMA = "rt_mb4_matched_full_minus_mb4_aggregate_v1"
AGGREGATE_STATUS = "PASS_RT_FULL_MINUS_MB4_ALL_15_PAIRED"
PREDECESSOR_SCHEMA = "rt_controls_partition_clean_terminal_v1"
PREDECESSOR_STATUS = "PASS_RT_CONTROLS_PARTITION_CLEAN_TERMINAL"
PREDECESSOR_FOLDS = {
    # These are the control cells actually owned by the two existing queues,
    # rather than the MB4 folds that subsequently occupy each GPU.
    "asc": tuple(range(5, 11)),
    "desc": tuple(range(0, 5)),
}
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 202608082
DEFAULT_R_C_AGGREGATE = WORKSPACE / "sua_exploration/results/k4_rt_loso_v1/rt_seed42_clean_nested_aggregate.json"
DEFAULT_R_C_SEAL = WORKSPACE / "sua_exploration/results/k4_rt_loso_v1/rt_seed42_clean_nested_seal.marker"
DEFAULT_WORK_ROOT = PROJECT / "outputs/rt_mb4_matched_clean_nested_v1"
DEFAULT_IMPORT_ROOT = WORKSPACE / "sua_exploration/results/rt_mb4_matched_clean_nested_v1/imported_cells"
DEFAULT_PLAN_OUTPUT = WORKSPACE / "sua_exploration/results/rt_mb4_matched_clean_nested_v1/RT_MB4_MATCHED_CONTINUATION_PLAN_v4.json"
DEFAULT_CONTROL_ROOT = PROJECT / "outputs/rt_controls_continuation_v1"
DEFAULT_PREDECESSOR_ROOT = WORKSPACE / "sua_exploration/results/rt_mb4_matched_clean_nested_v1/predecessor_terminals"
DEFAULT_FINAL_OUTPUT = WORKSPACE / "sua_exploration/results/rt_mb4_matched_clean_nested_v1/RT_MB4_MATCHED_FULL15_AGGREGATE_v1.json"
TRAIN_ENTRY = PROJECT / "src/train.py"
EVAL_ENTRY = PROJECT / "src/rt_clean_nested_loso_eval.py"


class Mb4ContinuationError(RuntimeError):
    """A frozen MB4 continuation/receipt invariant failed."""


def _controls_module() -> Any:
    """Load the existing control validator without importing any data/model code.

    The MB4 program must not independently reimplement the R-RS/R-LS receipt
    contract.  Loading its small orchestration module by path works both for
    ``python scripts/...`` and for no-NWB unit tests, and never opens data.
    """

    source = PROJECT / "scripts/rt_controls_continuation.py"
    _need(source.is_file(), f"RT controls validator is missing: {source}")
    spec = importlib.util.spec_from_file_location("rt_controls_continuation_for_mb4", source)
    _need(spec is not None and spec.loader is not None, "cannot load RT controls validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise Mb4ContinuationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    _need(path.is_file(), f"missing JSON receipt: {path}")
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise Mb4ContinuationError(f"invalid JSON receipt: {path}") from error
    _need(isinstance(body, dict), f"receipt must be a JSON object: {path}")
    return body


def _immutable(path: Path, label: str) -> None:
    _need(path.is_file(), f"{label} is missing: {path}")
    _need(stat.S_IMODE(path.stat().st_mode) == 0o444, f"{label} must be mode 0444: {path}")


def _write_immutable(path: Path, body: Mapping[str, Any]) -> str:
    _need(not path.exists(), f"refusing to overwrite immutable evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(body), indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)
    _immutable(path, "new immutable evidence")
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
    log: Path


def cell_paths(root: Path, fold: int) -> CellPaths:
    _need(fold in FOLDS, f"fold must be in [0,14], got {fold}")
    cell = root.resolve() / ARM / f"fold_{fold:02d}" / f"seed_{SEED}"
    return CellPaths(
        cell=cell,
        outer=cell / "outer_target_eval.json",
        selection=cell / "fit/rt_nested_selection_receipt.json",
        split=cell / "fit/split_manifest.json",
        config=cell / "fit/.hydra/config.yaml",
        terminal=cell / "cell_terminal.json",
        imported=cell / "mb4_copy_import.json",
        log=cell / "worker.log",
    )


def load_rc_reference(*, aggregate_path: Path, seal_path: Path) -> RcReference:
    aggregate_path, seal_path = aggregate_path.resolve(), seal_path.resolve()
    _immutable(aggregate_path, "sealed Full R-C aggregate")
    _immutable(seal_path, "sealed Full R-C marker")
    aggregate, seal = _json(aggregate_path), _json(seal_path)
    _need(aggregate.get("schema") == R_C_SCHEMA and aggregate.get("status") == R_C_STATUS,
          "Full R-C aggregate schema/status drift")
    _need(seal.get("schema") == R_C_SEAL_SCHEMA and seal.get("status") == R_C_STATUS and
          seal.get("aggregate_sha256") == _sha256(aggregate_path), "Full R-C seal drift")
    _need(aggregate.get("task") == "rt" and aggregate.get("seed") == SEED, "Full R-C task/seed drift")
    audits = aggregate.get("audits")
    _need(isinstance(audits, Mapping) and audits.get("exact_main_grid") is True,
          "Full R-C does not prove the exact main grid")
    for key in (
        "all_model_state_unchanged", "all_optimizer_absent", "all_target_backpropagation_false",
        "all_target_loaded_during_fit_false", "all_target_query_labels_read_during_fit_false",
        "all_target_query_labels_used_for_calibration_false",
        "all_target_query_labels_used_for_normalization_false",
        "all_target_query_labels_used_for_checkpoint_selection_false",
        "all_target_query_labels_used_for_scoring_only_true",
    ):
        _need(audits.get(key) is True, f"Full R-C audit failed: {key}")
    rows: dict[int, Mapping[str, Any]] = {}
    for row in aggregate.get("cells", []):
        if not isinstance(row, Mapping) or row.get("arm") != FULL_ARM:
            continue
        fold = row.get("fold")
        _need(isinstance(fold, int) and fold in FOLDS and fold not in rows,
              "Full R-C lacks exactly one afc4_vel cell per fold")
        _need(row.get("query_start_trial") == QUERY_START and row.get("window_size") == WINDOW_SIZE,
              f"Full R-C fold {fold} q24/window50 drift")
        _need(isinstance(row.get("target_session"), str) and isinstance(row.get("inner_validation_session"), str) and
              row["target_session"] != row["inner_validation_session"], f"Full R-C fold {fold} session binding drift")
        _finite(row.get("query_windows_evaluated"), f"Full R-C fold {fold} query window count")
        _finite(row.get("r2_variance_weighted"), f"Full R-C fold {fold} R2")
        rows[fold] = row
    _need(tuple(sorted(rows)) == FOLDS, "Full R-C lacks the complete 15-fold comparator")
    return RcReference(rows=rows, aggregate_path=aggregate_path, aggregate_sha256=_sha256(aggregate_path),
                       seal_path=seal_path, seal_sha256=_sha256(seal_path))


def _validate_selection(selection: Mapping[str, Any], *, fold: int) -> None:
    expected = {
        "schema": "rt_clean_nested_loso_selection_receipt_v1", "status": "PASS_FIT_INNER_SELECTION_ONLY",
        "arm": ARM, "outer_loso_fold": fold, "seed": SEED,
        "selected_by_metric": "val_heldin/r2_mean", "selected_metric_scope": "inner_validation_session_only",
        "formal_heldout_opened": False, "outer_target_loaded_during_fit": False,
        "outer_target_query_labels_read_during_fit": False,
    }
    for key, value in expected.items():
        _need(selection.get(key) == value, f"MB4 fold {fold} selection drift: {key}")
    for key in ("best_model_sha256", "config_sha256", "split_manifest_sha256"):
        _need(isinstance(selection.get(key), str) and len(selection[key]) == 64,
              f"MB4 fold {fold} selection SHA absent: {key}")


def _validate_split(split: Mapping[str, Any], *, fold: int, rc: Mapping[str, Any]) -> None:
    _need(split.get("task") == "rt" and split.get("development_only") is True and
          split.get("validation_protocol") == "nested_loso", f"MB4 fold {fold} split scope drift")
    _need(split.get("outer_loso_fold") == fold and split.get("loso_fold") == fold and
          split.get("requested_side_feature_group") == ARM, f"MB4 fold {fold} split arm/fold drift")
    arm = split.get("arm")
    _need(isinstance(arm, Mapping) and arm.get("canonical_arm") == ARM, f"MB4 fold {fold} canonical arm drift")
    _need(split.get("target_session") == rc.get("target_session") and
          split.get("inner_validation_session") == rc.get("inner_validation_session"),
          f"MB4 fold {fold} target/inner session does not match sealed Full")
    protocol, calibration, query, nested, normalizer = (
        split.get(key) for key in ("protocol", "calibration", "query", "nested_selection", "source_only_normalizer")
    )
    _need(all(isinstance(value, Mapping) for value in (protocol, calibration, query, nested, normalizer)),
          f"MB4 fold {fold} split sections absent")
    _need(protocol.get("signal_type") == "sorted_SUA" and protocol.get("decode_target") == "2D cursor velocity",
          f"MB4 fold {fold} signal/target drift")
    _need(calibration.get("budget_trials") == M and calibration.get("trial_index_range") == [0, M] and
          calibration.get("target_calibration_optimizer_steps") == 0, f"MB4 fold {fold} M24/no-BP calibration drift")
    _need(query.get("query_start_trial") == QUERY_START and query.get("window_size_bins") == WINDOW_SIZE and
          query.get("full_window_after_support_required") is True, f"MB4 fold {fold} q24/window50 drift")
    for key, value in (
        ("clean", True), ("outer_target_loaded_during_fit", False),
        ("outer_target_query_labels_read_during_fit", False),
        ("inner_validation_only_for_checkpoint_selection", True),
        ("checkpoint_metric", "val_heldin/r2_mean"), ("checkpoint_metric_scope", "inner_validation_session_only"),
    ):
        _need(nested.get(key) == value, f"MB4 fold {fold} nested selection drift: {key}")
    _need(normalizer.get("fit_scope") == "inner_train_sessions_only" and normalizer.get("feature_group") == ARM and
          normalizer.get("excluded_outer_target_session") == rc.get("target_session"),
          f"MB4 fold {fold} source-only normalizer drift")


def _validate_outer(outer: Mapping[str, Any], *, fold: int, rc: Mapping[str, Any]) -> float:
    expected = {
        "schema": "rt_clean_nested_loso_outer_eval_v1", "status": "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP",
        "arm": ARM, "outer_loso_fold": fold, "seed": SEED, "query_start_trial": QUERY_START,
        "window_size": WINDOW_SIZE, "outer_target_session": rc.get("target_session"),
        "query_windows_evaluated": rc.get("query_windows_evaluated"),
        "normalizer_fit_scope": "inner_train_sessions_only", "target_backpropagation": False,
        "optimizer_present": False, "model_training_mode": False, "model_state_unchanged": True,
        "target_query_labels_used_for_calibration": False, "target_query_labels_used_for_normalization": False,
        "target_query_labels_used_for_checkpoint_selection": False, "target_query_labels_used_for_scoring_only": True,
        "target_support_calibration_labels_used": True, "target_support_calibration_velocity_used": True,
    }
    for key, value in expected.items():
        _need(outer.get(key) == value, f"MB4 fold {fold} outer contract drift: {key}")
    _need(outer.get("model_state_sha256_before") == outer.get("model_state_sha256_after"),
          f"MB4 fold {fold} target pass changed model state")
    return _finite(outer.get("r2_variance_weighted"), f"MB4 fold {fold} R2")


def _accounting(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only verbatim accounting evidence that a receipt already carries."""

    for key in ("model_accounting", "cost_accounting", "accounting"):
        value = receipt.get(key)
        if isinstance(value, Mapping):
            return {"status": "verbatim_validated_receipt_field", "field": key, "value": dict(value)}
    return {"status": "not_present_in_validated_receipt__no_parameter_mac_state_claim"}


def validate_cell(*, root: Path, fold: int, reference: RcReference, require_import: bool,
                  require_terminal: bool = True) -> dict[str, Any]:
    paths = cell_paths(root, fold)
    required = (paths.outer, paths.selection, paths.split, paths.config) + ((paths.terminal,) if require_terminal else ())
    exists = [path.exists() for path in required]
    if not any(exists):
        return {"state": "missing", "fold": fold, "cell": str(paths.cell)}
    _need(all(exists), f"MB4 fold {fold} receipt bundle is partial, not comparable")
    outer, selection, split = (_json(paths.outer), _json(paths.selection), _json(paths.split))
    _validate_selection(selection, fold=fold)
    _need(_sha256(paths.config) == selection.get("config_sha256"), f"MB4 fold {fold} config SHA mismatch")
    _need(_sha256(paths.split) == selection.get("split_manifest_sha256"), f"MB4 fold {fold} split SHA mismatch")
    _need(outer.get("checkpoint_sha256") == selection.get("best_model_sha256"), f"MB4 fold {fold} selected checkpoint SHA mismatch")
    _validate_split(split, fold=fold, rc=reference.rows[fold])
    r2 = _validate_outer(outer, fold=fold, rc=reference.rows[fold])
    if require_terminal:
        terminal = _json(paths.terminal)
        _need(terminal.get("status") == "PASS" and terminal.get("arm") == ARM and terminal.get("fold") == fold and
              terminal.get("seed") == SEED, f"MB4 fold {fold} terminal drift")
    imported_sha = None
    if require_import:
        _immutable(paths.terminal, f"MB4 fold {fold} imported terminal")
        _immutable(paths.imported, f"MB4 fold {fold} import receipt")
        imported = _json(paths.imported)
        _need(imported.get("schema") == IMPORT_SCHEMA and imported.get("status") == IMPORT_STATUS and
              imported.get("arm") == ARM and imported.get("fold") == fold and imported.get("seed") == SEED,
              f"MB4 fold {fold} import receipt drift")
        files = imported.get("imported_files")
        _need(isinstance(files, Mapping), f"MB4 fold {fold} import file map missing")
        for name, path in (("outer", paths.outer), ("selection", paths.selection), ("split", paths.split),
                           ("config", paths.config), ("terminal", paths.terminal)):
            item = files.get(name)
            _need(isinstance(item, Mapping) and item.get("sha256") == _sha256(path),
                  f"MB4 fold {fold} immutable import SHA mismatch: {name}")
        imported_sha = _sha256(paths.imported)
    return {
        "state": "complete", "fold": fold, "mb4_r2": r2,
        "full_r2": float(reference.rows[fold]["r2_variance_weighted"]),
        "full_minus_mb4": float(reference.rows[fold]["r2_variance_weighted"]) - r2,
        "target_session": rc_value(reference.rows[fold], "target_session"),
        "inner_validation_session": rc_value(reference.rows[fold], "inner_validation_session"),
        "query_windows_evaluated": int(reference.rows[fold]["query_windows_evaluated"]),
        "mb4_accounting": _accounting(outer), "full_accounting": _accounting(reference.rows[fold]),
        "copy_import_sha256": imported_sha,
    }


def rc_value(row: Mapping[str, Any], key: str) -> Any:
    return row[key]


def inventory(*, root: Path, aggregate_path: Path, seal_path: Path, require_import: bool) -> dict[str, Any]:
    reference = load_rc_reference(aggregate_path=aggregate_path, seal_path=seal_path)
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        try:
            rows.append(validate_cell(root=root, fold=fold, reference=reference, require_import=require_import))
        except Mb4ContinuationError as error:
            rows.append({"state": "incomparable", "fold": fold, "reason": str(error)})
    complete = [row for row in rows if row["state"] == "complete"]
    return {
        "schema": "rt_mb4_matched_inventory_v1", "status": "READ_ONLY_MB4_PARTIAL_PREVIEW_NOT_A_CLAIM",
        "scope": "receipt_only_no_nwb_no_trainer_no_cuda", "root": str(root.resolve()), "require_import": require_import,
        "r_c_reference": {"aggregate_sha256": reference.aggregate_sha256, "seal_sha256": reference.seal_sha256},
        "rows": rows, "complete_folds": [row["fold"] for row in complete],
        "missing_folds": [row["fold"] for row in rows if row["state"] == "missing"],
        "incomparable_folds": [row["fold"] for row in rows if row["state"] == "incomparable"],
        "finalize_eligible": require_import and len(complete) == len(FOLDS) and not any(row["state"] == "incomparable" for row in rows),
    }


def copy_import(*, source_root: Path, import_root: Path, fold: int, aggregate_path: Path, seal_path: Path) -> dict[str, Any]:
    reference = load_rc_reference(aggregate_path=aggregate_path, seal_path=seal_path)
    source = validate_cell(root=source_root, fold=fold, reference=reference, require_import=False)
    _need(source["state"] == "complete", f"cannot import non-complete MB4 fold {fold}")
    src, dst = cell_paths(source_root, fold), cell_paths(import_root, fold)
    _need(not dst.cell.exists(), f"refusing duplicate/overwrite import destination: {dst.cell}")
    dst.cell.mkdir(parents=True, exist_ok=False)
    copied: dict[str, dict[str, str]] = {}
    for name, origin, destination in (("outer", src.outer, dst.outer), ("selection", src.selection, dst.selection),
                                      ("split", src.split, dst.split), ("config", src.config, dst.config),
                                      ("terminal", src.terminal, dst.terminal)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        origin_sha = _sha256(origin)
        shutil.copyfile(origin, destination)
        _need(_sha256(destination) == origin_sha, f"MB4 fold {fold} copy SHA mismatch: {name}")
        destination.chmod(0o444)
        copied[name] = {"source_path": str(origin), "source_sha256": origin_sha,
                        "path": str(destination), "sha256": _sha256(destination)}
    receipt = {
        "schema": IMPORT_SCHEMA, "status": IMPORT_STATUS, "arm": ARM, "fold": fold, "seed": SEED,
        "source_root": str(source_root.resolve()), "source_cell": str(src.cell), "imported_cell": str(dst.cell),
        "imported_files": copied, "r_c_reference": {"aggregate_sha256": reference.aggregate_sha256, "seal_sha256": reference.seal_sha256},
        "scope": "copy_only_no_nwb_no_trainer_no_cuda_no_source_mutation",
    }
    receipt_sha = _write_immutable(dst.imported, receipt)
    validated = validate_cell(root=import_root, fold=fold, reference=reference, require_import=True)
    return {"status": IMPORT_STATUS, "fold": fold, "receipt_sha256": receipt_sha, "validated": validated}


def _sign_test(values: Sequence[float]) -> dict[str, Any]:
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    zero = len(values) - positive - negative
    nonzero = positive + negative
    tail = 1.0 if nonzero == 0 else sum(math.comb(nonzero, item) for item in range(min(positive, negative) + 1)) / (2 ** nonzero)
    return {"test": "exact_two_sided_binomial_sign_test", "positive": positive, "negative": negative,
            "zero": zero, "nonzero_pairs": nonzero, "p_value": min(1.0, 2.0 * tail), "ties": "excluded"}


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    _need(bool(sorted_values) and 0.0 <= probability <= 1.0, "invalid quantile input")
    position = (len(sorted_values) - 1) * probability
    left, right = math.floor(position), math.ceil(position)
    if left == right:
        return float(sorted_values[left])
    return float(sorted_values[left] * (right - position) + sorted_values[right] * (position - left))


def _bootstrap(values: Sequence[float]) -> dict[str, Any]:
    numbers = tuple(_finite(value, "bootstrap delta") for value in values)
    _need(bool(numbers), "paired bootstrap requires non-empty deltas")
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(sum(numbers[rng.randrange(len(numbers))] for _ in numbers) / len(numbers) for _ in range(BOOTSTRAP_DRAWS))
    return {"resampling_unit": "outer_LOSO_fold", "draws": BOOTSTRAP_DRAWS, "rng_seed": BOOTSTRAP_SEED,
            "quantile_method": "linear_order_statistic", "lower_95": _quantile(means, .025), "upper_95": _quantile(means, .975)}


def finalize(*, import_root: Path, aggregate_path: Path, seal_path: Path, output: Path) -> dict[str, Any]:
    _need(not output.exists(), f"refusing to overwrite MB4 final aggregate: {output}")
    report = inventory(root=import_root, aggregate_path=aggregate_path, seal_path=seal_path, require_import=True)
    _need(report["finalize_eligible"] is True, "MB4 finalizer requires exactly 15 complete immutable imported folds")
    rows = report["rows"]
    delta = [row["full_minus_mb4"] for row in rows]
    body = {
        "schema": AGGREGATE_SCHEMA, "status": AGGREGATE_STATUS, "arm": ARM, "full_comparator_arm": FULL_ARM,
        "seed": SEED, "folds": list(FOLDS), "scope": "receipt_only_no_nwb_no_trainer_no_cuda",
        "r_c_reference": report["r_c_reference"], "import_root": str(import_root.resolve()), "rows": rows,
        "full_minus_mb4": {"mean": sum(delta) / len(delta), "median": _quantile(sorted(delta), .5),
                             "positive_folds": sum(value > 0.0 for value in delta),
                             "exact_two_sided_sign_test": _sign_test(delta), "paired_fold_bootstrap_95": _bootstrap(delta)},
        "accounting": {
            "full": [row["full_accounting"] for row in rows], "mb4": [row["mb4_accounting"] for row in rows],
            "interpretation": "Only accounting fields verbatim present in a validated receipt are reproduced; absent fields make no parameter/MAC/state claim.",
        },
        "interpretation": "Full minus MB4 attributes signed [w_x,w_y] beyond aligned [||W||,b]. All signed fold deltas, including negative folds, are retained.",
    }
    digest = _write_immutable(output, body)
    return {"status": AGGREGATE_STATUS, "output": str(output), "sha256": digest}


def copy_import_all_and_finalize(*, source_root: Path, import_root: Path, aggregate_path: Path, seal_path: Path,
                                 output: Path) -> dict[str, Any]:
    """Copy one prevalidated 15-fold MB4 matrix, then write its sole aggregate.

    The initial inventory is mandatory before the first immutable copy.  Thus a
    missing/partial/incomparable raw cell leaves both the requested import root
    and aggregate absent.  A later filesystem race still stops at the affected
    fold and cannot create a paired claim from the partial import root.
    """

    _need(not import_root.exists(), f"all-fold import root must be fresh: {import_root}")
    _need(not output.exists(), f"refusing to overwrite MB4 final aggregate: {output}")
    preflight = inventory(root=source_root, aggregate_path=aggregate_path, seal_path=seal_path, require_import=False)
    _need(preflight["complete_folds"] == list(FOLDS) and not preflight["missing_folds"] and
          not preflight["incomparable_folds"], "MB4 all-fold import requires exactly 15 complete raw receipt bundles")
    imports = [
        copy_import(source_root=source_root, import_root=import_root, fold=fold,
                    aggregate_path=aggregate_path, seal_path=seal_path)
        for fold in FOLDS
    ]
    aggregate = finalize(import_root=import_root, aggregate_path=aggregate_path, seal_path=seal_path, output=output)
    return {"status": "PASS_RT_MB4_COPY_IMPORT_ALL_AND_FINALIZE", "source_root": str(source_root.resolve()),
            "import_root": str(import_root.resolve()), "folds": list(FOLDS), "imports": imports,
            "aggregate": aggregate,
            "scope": "receipt_only_no_nwb_no_trainer_no_cuda_all_15_required_no_delta_gate"}


def expected_predecessor_cells(partition: str) -> list[dict[str, Any]]:
    """Return the frozen R-RS/R-LS receipt matrix for one control lane."""

    _need(partition in PREDECESSOR_FOLDS, f"unknown controls predecessor partition: {partition}")
    return [
        {"arm": arm, "fold": fold, "seed": SEED}
        for arm in ("afc4_rs", "afc4_ls")
        for fold in PREDECESSOR_FOLDS[partition]
    ]


def _validate_predecessor_cell_row(row: Mapping[str, Any], *, expected: Mapping[str, Any]) -> None:
    _need(row.get("state") == "complete", f"predecessor {expected['arm']} fold {expected['fold']} is not complete")
    for key in ("arm", "fold"):
        _need(row.get(key) == expected[key], f"predecessor row drift: {key}")
    files = row.get("files")
    _need(isinstance(files, Mapping), f"predecessor {expected['arm']} fold {expected['fold']} file SHA map absent")
    for key in ("outer", "selection", "split", "config", "cell_terminal"):
        value = files.get(key)
        _need(isinstance(value, Mapping) and isinstance(value.get("path"), str) and
              isinstance(value.get("sha256"), str) and len(value["sha256"]) == 64,
              f"predecessor {expected['arm']} fold {expected['fold']} SHA absent: {key}")


def seal_predecessor(*, partition: str, control_root: Path, aggregate_path: Path, seal_path: Path,
                     output: Path) -> dict[str, Any]:
    """Seal the exact raw R-RS/R-LS queue prefix needed by one MB4 lane.

    This is deliberately receipt-only: the existing control validator is used
    for every expected arm/fold, all source receipt hashes are recorded, and
    no mutation is made below ``control_root``.  A missing, partial, failed, or
    invalid control cell refuses to create a terminal, so an MB4 worker cannot
    mistake a vanished tmux session for a successful predecessor.
    """

    _need(partition in PREDECESSOR_FOLDS, f"unknown controls predecessor partition: {partition}")
    controls = _controls_module()
    _need(tuple(controls.CONTROL_ARMS) == ("afc4_rs", "afc4_ls"), "control arm contract drift")
    _need(tuple(controls.FOLDS) == FOLDS and controls.SEED == SEED, "control fold/seed contract drift")
    reference = controls._load_rc_reference(aggregate_path=aggregate_path, seal_path=seal_path)
    expected = expected_predecessor_cells(partition)
    validated_cells: list[dict[str, Any]] = []
    for item in expected:
        row = controls.validate_control_cell(
            control_root=control_root, arm=str(item["arm"]), fold=int(item["fold"]),
            reference=reference, require_import=False,
        )
        _validate_predecessor_cell_row(row, expected=item)
        validated_cells.append(dict(row))

    body = {
        "schema": PREDECESSOR_SCHEMA,
        "status": PREDECESSOR_STATUS,
        "partition": partition,
        "all_cells_passed": True,
        "failed_cells": [],
        "control_root": str(control_root.resolve()),
        "expected_cells": expected,
        "validated_cells": validated_cells,
        "r_c_reference": {
            "aggregate_path": str(reference.aggregate_path),
            "aggregate_sha256": reference.aggregate_sha256,
            "seal_path": str(reference.seal_path),
            "seal_sha256": reference.seal_sha256,
        },
        "controls_validator": {
            "path": str((PROJECT / "scripts/rt_controls_continuation.py").resolve()),
            "sha256": _sha256(PROJECT / "scripts/rt_controls_continuation.py"),
        },
        "scope": "receipt_only_no_nwb_no_trainer_no_cuda_no_control_root_mutation",
    }
    digest = _write_immutable(output, body)
    return {"status": PREDECESSOR_STATUS, "partition": partition, "output": str(output), "sha256": digest,
            "expected_cell_count": len(expected)}


def validate_predecessor_terminal(path: Path, *, expected_partition: str) -> dict[str, Any]:
    _need(expected_partition in PREDECESSOR_FOLDS, f"unknown expected controls predecessor partition: {expected_partition}")
    _immutable(path, "controls predecessor terminal")
    terminal = _json(path)
    _need(terminal.get("schema") == PREDECESSOR_SCHEMA and terminal.get("status") == PREDECESSOR_STATUS,
          f"predecessor is not a clean controls terminal: {path}")
    _need(terminal.get("partition") == expected_partition and terminal.get("all_cells_passed") is True and
          terminal.get("failed_cells") == [], f"predecessor terminal is not clean for {expected_partition}: {path}")
    expected = expected_predecessor_cells(expected_partition)
    _need(terminal.get("expected_cells") == expected, f"predecessor terminal expected-cell matrix drift: {path}")
    rows = terminal.get("validated_cells")
    _need(isinstance(rows, list) and len(rows) == len(expected), f"predecessor terminal validated-cell count drift: {path}")
    for item, row in zip(expected, rows):
        _need(isinstance(row, Mapping), f"predecessor terminal non-object validated row: {path}")
        _validate_predecessor_cell_row(row, expected=item)
    reference = terminal.get("r_c_reference")
    _need(isinstance(reference, Mapping) and all(isinstance(reference.get(key), str) and len(reference[key]) == 64
                                                  for key in ("aggregate_sha256", "seal_sha256")),
          f"predecessor terminal R-C binding drift: {path}")
    return terminal


def same_fold_rt_writer_commands(commands: Sequence[str], *, fold: int) -> list[str]:
    """Find exact RT writer bindings; never use a fold-number substring.

    ``data.loso_fold=1`` must not collide with ``data.loso_fold=10``.  A
    command is relevant only when it is an RT-clean invocation *and* carries
    one of the two exact Hydra fold assignment tokens for the requested fold.
    """

    expected = {f"data.loso_fold={fold}", f"data.outer_loso_fold={fold}"}
    matches: list[str] = []
    for command in commands:
        try:
            tokens = shlex.split(command)
        except ValueError:
            # An unparsable active command must be treated as unsafe if it is
            # plausibly an RT clean writer; it may never be quietly ignored.
            if "rt_clean_nested" in command.lower():
                matches.append(command)
            continue
        if "rt_clean_nested" not in " ".join(tokens).lower():
            continue
        if expected.intersection(tokens):
            matches.append(command)
    return matches


def ensure_launch_safe(*, gpu_commands: Sequence[str], all_rt_commands: Sequence[str], fold: int, gpu: int) -> None:
    _need(not gpu_commands, f"requested GPU {gpu} has active compute owner(s): {list(gpu_commands)}")
    collisions = same_fold_rt_writer_commands(all_rt_commands, fold=fold)
    _need(not collisions, f"active exact-same-fold RT writer blocks MB4 fold {fold}: {collisions}")


def active_compute_commands_on_gpu(gpu: int) -> tuple[list[str], list[str]]:
    """Return selected-GPU owners and all RT clean command lines; fail closed."""

    try:
        table = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
        )
        uuid_by_index = {int(line.split(",", 1)[0].strip()): line.split(",", 1)[1].strip()
                         for line in table.splitlines() if "," in line}
        _need(gpu in uuid_by_index, f"GPU index {gpu} is not visible")
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], text=True
        )
        ps_table = subprocess.check_output(["ps", "-eo", "args="], text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise Mb4ContinuationError("cannot safely inspect GPU/RT writers") from error
    owners: list[str] = []
    for line in apps.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and fields[1] == uuid_by_index[gpu]:
            try:
                owners.append(Path(f"/proc/{int(fields[0])}/cmdline").read_bytes().replace(b"\0", b" ").decode())
            except OSError:
                owners.append(f"<unreadable-gpu-pid:{fields[0]}>")
    return owners, [line for line in ps_table.splitlines() if "rt_clean_nested" in line.lower()]


def _ensure_isolated_root(root: Path) -> Path:
    resolved = root.resolve()
    _need(resolved != PROJECT.resolve(), "MB4 root cannot be the project root")
    for forbidden in (
        PROJECT / "outputs/rt_controls_split_gpu1_after_asc", PROJECT / "outputs/rt_controls_split_gpu0_after_desc",
        PROJECT / "outputs/rt_stage_r_b2_local3090", PROJECT / "outputs/rt_stage_r_b2_remote",
    ):
        try:
            resolved.relative_to(forbidden.resolve())
        except ValueError:
            continue
        raise Mb4ContinuationError(f"MB4 root may not overlap an active controls root: {resolved}")
    return resolved


def _prepare_partition_root(root: Path) -> None:
    """Create/read the one shared MB4 root without allowing a foreign root."""

    marker = root / "MB4_CONTINUATION_ROOT_v1.json"
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"schema": "rt_mb4_continuation_root_v1", "arm": ARM, "seed": SEED}, sort_keys=True) + "\n")
        return
    _need(marker.is_file(), f"existing MB4 work root lacks this program's marker: {root}")
    body = _json(marker)
    _need(body == {"schema": "rt_mb4_continuation_root_v1", "arm": ARM, "seed": SEED},
          f"existing MB4 work root marker drift: {root}")


def train_command(*, root: Path, fold: int, gpu: int) -> list[str]:
    paths = cell_paths(root, fold)
    return [
        sys.executable, str(TRAIN_ENTRY), "experiment=rt_clean_nested_loso_m24",
        f"run_id=rt_clean_nested_loso_m24_{ARM}", f"data.loso_fold={fold}", f"data.outer_loso_fold={fold}",
        f"data.side_feature_group={ARM}", f"seed={SEED}", "test=false", "trainer.accelerator=gpu", "trainer.devices=1",
        f"hydra.run.dir={paths.cell / 'fit'}", f"paths.root_dir={PROJECT}",
        f"paths.log_dir={root / '_hydra_logs'}", f"paths.artifact_dir={root / '_artifacts'}",
    ]


def eval_command(*, root: Path, fold: int, checkpoint: Path | str) -> list[str]:
    paths = cell_paths(root, fold)
    return [sys.executable, str(EVAL_ENTRY), "--config", str(paths.config.resolve()), "--checkpoint", str(Path(checkpoint).resolve()),
            "--split-manifest", str(paths.split.resolve()), "--selection-receipt", str(paths.selection.resolve()),
            "--output", str(paths.outer.resolve()), "--outer-fold", str(fold), "--device", "cuda"]


def build_plan(*, work_root: Path, aggregate_path: Path, seal_path: Path,
               gpu1_predecessor_terminal: Path, gpu0_predecessor_terminal: Path,
               control_root: Path = DEFAULT_CONTROL_ROOT, import_root: Path = DEFAULT_IMPORT_ROOT,
               final_output: Path = DEFAULT_FINAL_OUTPUT) -> dict[str, Any]:
    reference = load_rc_reference(aggregate_path=aggregate_path, seal_path=seal_path)
    root = _ensure_isolated_root(work_root)
    _need(not root.exists(), f"fresh MB4 work root must not exist: {root}")
    all_folds = set(GPU1_ASC_FOLDS) | set(GPU0_DESC_FOLDS)
    _need(not (set(GPU1_ASC_FOLDS) & set(GPU0_DESC_FOLDS)) and all_folds == set(FOLDS), "MB4 static partitions are not an exact disjoint 15-fold cover")
    partitions: dict[str, Any] = {}
    predecessor_by_partition = {"gpu1_after_asc": gpu1_predecessor_terminal, "gpu0_after_desc": gpu0_predecessor_terminal}
    for name, spec in PARTITIONS.items():
        folds, gpu = tuple(spec["folds"]), int(spec["gpu"])
        cells = []
        for fold in folds:
            paths = cell_paths(root, fold)
            cells.append({
                "fold": fold, "target_session": reference.rows[fold]["target_session"],
                "inner_validation_session": reference.rows[fold]["inner_validation_session"],
                "query_windows_evaluated": reference.rows[fold]["query_windows_evaluated"],
                "cell": str(paths.cell), "train_command": train_command(root=root, fold=fold, gpu=gpu),
                "eval_command_template": eval_command(root=root, fold=fold, checkpoint=paths.cell / "fit/checkpoints/<inner-selected>.ckpt"),
            })
        partitions[name] = {"gpu": gpu, "folds": list(folds), "predecessor_terminal": str(predecessor_by_partition[name]),
                            "required_predecessor_partition": spec["predecessor_partition"], "cells": cells}
    def _worker_command(name: str, gpu: int, predecessor: Path) -> str:
        inner = " ".join([
            "cd", shlex.quote(str(PROJECT)), "&&", f"CUDA_VISIBLE_DEVICES={gpu}", shlex.quote(sys.executable),
            "scripts/rt_mb4_matched_continuation.py", "--run-partition", name, "--work-root", shlex.quote(str(root)),
            "--predecessor-terminal", shlex.quote(str(predecessor)), "--rc-aggregate", shlex.quote(str(aggregate_path)),
            "--rc-seal", shlex.quote(str(seal_path)),
        ])
        return inner

    def _watcher_template(name: str, gpu: int, controls_partition: str, controls_session: str,
                          predecessor: Path) -> str:
        # This intentionally waits for the *current controls tmux session* to
        # disappear, seals raw R-RS/R-LS receipts, and only then enters the
        # independently guarded MB4 worker.  It does not wait for the other
        # controls lane, so GPU1 can begin once its own asc lane is clean.
        inner = " ".join([
            "set -euo pipefail;", "cd", shlex.quote(str(PROJECT)), ";",
            "while", "tmux", "has-session", "-t", shlex.quote(controls_session), "2>/dev/null;", "do", "sleep", "15;", "done;",
            shlex.quote(sys.executable), "scripts/rt_mb4_matched_continuation.py", "--seal-predecessor", controls_partition,
            "--control-root", shlex.quote(str(control_root)), "--predecessor-output", shlex.quote(str(predecessor)),
            "--rc-aggregate", shlex.quote(str(aggregate_path)), "--rc-seal", shlex.quote(str(seal_path)), ";",
            _worker_command(name, gpu, predecessor),
        ])
        return "tmux new-session -d -s " + shlex.quote(f"rt_mb4_gpu{gpu}_after_controls") + " " + shlex.quote(inner)

    def _finalizer_template() -> str:
        workers = ("rt_mb4_gpu1_after_controls", "rt_mb4_gpu0_after_controls")
        # The starter must run this only after both worker watcher sessions are
        # visible.  The initial check prevents a never-started worker from
        # looking like a clean exit; after that, all-fold receipt preflight is
        # the authority for completion and failures.
        worker_words = " ".join(shlex.quote(worker) for worker in workers)
        inner = " ".join([
            "set -euo pipefail;", "cd", shlex.quote(str(PROJECT)), ";",
            "for", "session", "in", worker_words, ";", "do", "tmux", "has-session", "-t", '"$session"', "2>/dev/null", "||",
            "{", "echo", '"required MB4 worker session was never observed: $session"', ">&2;", "exit", "2;", "};", "done;",
            "while", "tmux", "has-session", "-t", shlex.quote(workers[0]), "2>/dev/null", "||", "tmux", "has-session", "-t", shlex.quote(workers[1]), "2>/dev/null;",
            "do", "sleep", "15;", "done;",
            shlex.quote(sys.executable), "scripts/rt_mb4_matched_continuation.py", "--copy-import-all-and-finalize",
            "--work-root", shlex.quote(str(root)), "--import-root", shlex.quote(str(import_root)),
            "--aggregate-output", shlex.quote(str(final_output)), "--rc-aggregate", shlex.quote(str(aggregate_path)),
            "--rc-seal", shlex.quote(str(seal_path)),
        ])
        return "tmux new-session -d -s rt_mb4_import_finalize_after_workers " + shlex.quote(inner)

    return {
        "schema": "rt_mb4_matched_continuation_plan_v4", "status": "PLAN_ONLY_NO_GPU_NO_NWB_NO_TRAINER",
        "arm": ARM, "full_comparator_arm": FULL_ARM, "seed": SEED, "work_root": str(root),
        "r_c_reference": {"aggregate_sha256": reference.aggregate_sha256, "seal_sha256": reference.seal_sha256,
                            "query_start_trial": QUERY_START, "window_size": WINDOW_SIZE},
        "protocol": {"task": "RT clean nested LOSO", "support": "chronological M24", "fit": "fresh source training; inner validation checkpoint only",
                     "outer": "one-shot target evaluation; no target BP; source-only normalizer", "stop_rule": "execute all assigned folds; do not inspect signed effect to stop"},
        "partitions": partitions,
        "predecessor_sealing": {
            "control_root": str(control_root.resolve()),
            "asc_expected_control_cells": expected_predecessor_cells("asc"),
            "desc_expected_control_cells": expected_predecessor_cells("desc"),
            "rule": "A terminal is created only after the existing controls validator passes every listed raw R-RS/R-LS receipt; missing, partial, failed, or SHA-invalid cells fail closed.",
        },
        "watcher_templates": {
            "gpu1_after_asc": _watcher_template("gpu1_after_asc", 1, "asc", "rt_controls_split_gpu1_after_asc", gpu1_predecessor_terminal),
            "gpu0_after_desc": _watcher_template("gpu0_after_desc", 0, "desc", "rt_controls_split_gpu0_after_desc", gpu0_predecessor_terminal),
        },
        "finalizer": {
            "session": "rt_mb4_import_finalize_after_workers",
            "requires_worker_sessions_observed": ["rt_mb4_gpu1_after_controls", "rt_mb4_gpu0_after_controls"],
            "waits_for_worker_sessions_to_disappear": True,
            "import_root": str(import_root.resolve()),
            "aggregate_output": str(final_output.resolve()),
            "rule": "Copies only after a fresh 15/15 raw-receipt preflight, then finalizes once. Missing, partial, failed, incomparable, or duplicate evidence stops without a signed-delta gate.",
            "tmux_template": _finalizer_template(),
        },
    }


def _run_logged(command: Sequence[str], *, cwd: Path, env: Mapping[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(list(command), cwd=cwd, env=dict(env), stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"[exit={completed.returncode}]\n")
        handle.flush()
    _need(completed.returncode == 0, f"MB4 subprocess failed with exit={completed.returncode}: {command[0]}")


def _write_terminal(path: Path, body: Mapping[str, Any]) -> None:
    _need(not path.exists(), f"refusing to overwrite MB4 cell terminal: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(body), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execute_partition(*, partition: str, work_root: Path, predecessor_terminal: Path,
                      aggregate_path: Path, seal_path: Path) -> dict[str, Any]:
    """Future explicit worker path. It is never called by --plan/import/finalize."""

    _need(partition in PARTITIONS, f"unknown MB4 partition: {partition}")
    spec = PARTITIONS[partition]
    _need(str(spec["gpu"]) == str(os.environ.get("CUDA_VISIBLE_DEVICES")),
          "run-partition CUDA_VISIBLE_DEVICES must exactly equal its frozen physical GPU")
    predecessor = validate_predecessor_terminal(predecessor_terminal, expected_partition=str(spec["predecessor_partition"]))
    reference = load_rc_reference(aggregate_path=aggregate_path, seal_path=seal_path)
    predecessor_reference = predecessor["r_c_reference"]
    _need(predecessor_reference.get("aggregate_sha256") == reference.aggregate_sha256 and
          predecessor_reference.get("seal_sha256") == reference.seal_sha256,
          "controls predecessor R-C comparator SHA does not match this MB4 invocation")
    root = _ensure_isolated_root(work_root)
    _prepare_partition_root(root)
    completed: list[int] = []
    for fold in spec["folds"]:
        paths = cell_paths(root, int(fold))
        cell_created = False
        try:
            _need(not paths.cell.exists(), f"MB4 cell already exists: {paths.cell}")
            gpu_commands, all_rt_commands = active_compute_commands_on_gpu(int(spec["gpu"]))
            ensure_launch_safe(gpu_commands=gpu_commands, all_rt_commands=all_rt_commands, fold=int(fold), gpu=int(spec["gpu"]))
            # The GPU-owner and exact-fold writer check is intentionally before
            # directory creation.  A transient occupied-GPU gate failure must
            # leave no partial MB4 cell that would block a later clean start.
            paths.cell.mkdir(parents=True, exist_ok=False)
            cell_created = True
            env = dict(os.environ)
            _run_logged(train_command(root=root, fold=int(fold), gpu=int(spec["gpu"])), cwd=PROJECT, env=env, log=paths.log)
            selection = _json(paths.selection)
            _validate_selection(selection, fold=int(fold))
            checkpoint = Path(str(selection.get("best_model_path", "")))
            _need(checkpoint.is_file(), f"MB4 fold {fold} selected checkpoint is absent")
            _run_logged(eval_command(root=root, fold=int(fold), checkpoint=checkpoint), cwd=PROJECT, env=env, log=paths.log)
            validate_cell(root=root, fold=int(fold), reference=reference, require_import=False, require_terminal=False)
            _write_terminal(paths.terminal, {"status": "PASS", "arm": ARM, "fold": int(fold), "seed": SEED})
            completed.append(int(fold))
        except Exception as error:
            if cell_created and paths.cell.exists() and not paths.terminal.exists():
                _write_terminal(paths.terminal, {"status": "FAILED", "arm": ARM, "fold": int(fold), "seed": SEED, "error": repr(error)})
            raise
    return {"status": "PASS_RT_MB4_STATIC_PARTITION", "partition": partition, "folds": completed}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run-partition", choices=tuple(PARTITIONS))
    mode.add_argument("--seal-predecessor", choices=tuple(PREDECESSOR_FOLDS))
    mode.add_argument("--copy-import", action="store_true")
    mode.add_argument("--copy-import-all-and-finalize", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--preview", action="store_true")
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--import-root", type=Path, default=DEFAULT_IMPORT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLAN_OUTPUT)
    parser.add_argument("--aggregate-output", type=Path)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--predecessor-terminal", type=Path)
    parser.add_argument("--predecessor-output", type=Path)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--gpu1-predecessor-terminal", type=Path,
                        default=DEFAULT_PREDECESSOR_ROOT / "rt_controls_asc_clean_terminal_v1.json")
    parser.add_argument("--gpu0-predecessor-terminal", type=Path,
                        default=DEFAULT_PREDECESSOR_ROOT / "rt_controls_desc_clean_terminal_v1.json")
    parser.add_argument("--rc-aggregate", type=Path, default=DEFAULT_R_C_AGGREGATE)
    parser.add_argument("--rc-seal", type=Path, default=DEFAULT_R_C_SEAL)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.run_partition:
        _need(args.predecessor_terminal is not None, "--run-partition requires --predecessor-terminal")
        result = execute_partition(partition=args.run_partition, work_root=args.work_root,
                                   predecessor_terminal=args.predecessor_terminal, aggregate_path=args.rc_aggregate, seal_path=args.rc_seal)
    elif args.seal_predecessor:
        _need(args.predecessor_output is not None, "--seal-predecessor requires --predecessor-output")
        result = seal_predecessor(partition=args.seal_predecessor, control_root=args.control_root,
                                  aggregate_path=args.rc_aggregate, seal_path=args.rc_seal,
                                  output=args.predecessor_output)
    elif args.copy_import:
        _need(args.fold is not None, "--copy-import requires --fold")
        result = copy_import(source_root=args.work_root, import_root=args.import_root, fold=args.fold,
                             aggregate_path=args.rc_aggregate, seal_path=args.rc_seal)
    elif args.copy_import_all_and_finalize:
        _need(args.aggregate_output is not None, "--copy-import-all-and-finalize requires --aggregate-output")
        result = copy_import_all_and_finalize(source_root=args.work_root, import_root=args.import_root,
                                              aggregate_path=args.rc_aggregate, seal_path=args.rc_seal,
                                              output=args.aggregate_output)
    elif args.finalize:
        result = finalize(import_root=args.import_root, aggregate_path=args.rc_aggregate, seal_path=args.rc_seal, output=args.output)
    elif args.preview:
        result = inventory(root=args.work_root, aggregate_path=args.rc_aggregate, seal_path=args.rc_seal, require_import=False)
    else:
        result = build_plan(work_root=args.work_root, aggregate_path=args.rc_aggregate, seal_path=args.rc_seal,
                            gpu1_predecessor_terminal=args.gpu1_predecessor_terminal,
                            gpu0_predecessor_terminal=args.gpu0_predecessor_terminal,
                            control_root=args.control_root, import_root=args.import_root,
                            final_output=args.aggregate_output or DEFAULT_FINAL_OUTPUT)
        if args.plan:
            digest = _write_immutable(args.output, result)
            result = {"status": result["status"], "plan": str(args.output), "sha256": digest,
                      "watcher_templates": result["watcher_templates"]}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
