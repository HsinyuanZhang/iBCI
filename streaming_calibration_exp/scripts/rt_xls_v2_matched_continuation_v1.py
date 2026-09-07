#!/usr/bin/env python3
"""Fail-closed RT XLSv2 15-fold supervisor, copy-importer, and finalizer.

Planning, inventory, import, and finalization are receipt-only.  The only
GPU-capable entry point is explicit ``--run-partition``; it trains a fresh
source model for every fold and then invokes the separate XLSv2 one-shot outer
evaluator.  No R-C checkpoint or common inverse/alignment map is accepted.
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
import shlex
import shutil
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
ARM = "afc4_xls_v2"
FULL_ARM = "afc4_vel"
FOLDS = tuple(range(15))
SEED = 42
M = 24
QUERY_START = 24
WINDOW_SIZE = 50
GPU1_ASC_FOLDS = tuple(range(0, 8))
GPU0_DESC_FOLDS = tuple(range(14, 7, -1))
PARTITIONS = {
    "gpu1_asc": {"gpu": 1, "folds": GPU1_ASC_FOLDS},
    "gpu0_desc": {"gpu": 0, "folds": GPU0_DESC_FOLDS},
}
MB4_SCHEMA = "rt_mb4_matched_full_minus_mb4_aggregate_v1"
MB4_STATUS = "PASS_RT_FULL_MINUS_MB4_ALL_15_PAIRED"
MB4_SHA256 = "152b76c149ae6c0cdc75c7320a3ba80863bac5f56abfaea87e58286262d6768a"
AUDIT_SCHEMA = "rt_afc4_ls_null_strength_support_audit_v2"
AUDIT_STATUS = "PASS_CPU_SUPPORT_ONLY_RT_AFC4_LS_NULL_STRENGTH_AUDIT_V2"
AUDIT_SHA256 = "ad2468ca04c3ed2542c7c37b0f1d27c17d4e517943fe64a7fa34e6cf9636c899"
OUTER_SCHEMA = "rt_clean_nested_loso_xls_v2_outer_eval_v1"
OUTER_STATUS = "PASS_ONE_SHOT_XLS_V2_OUTER_TARGET_NO_BACKPROP"
IMPORT_SCHEMA = "rt_xls_v2_matched_copy_import_v1"
IMPORT_STATUS = "PASS_RT_XLS_V2_MATCHED_COPY_IMPORTED"
AGGREGATE_SCHEMA = "rt_xls_v2_rc_minus_xls_v2_aggregate_v1"
AGGREGATE_STATUS = "PASS_RT_RC_MINUS_XLS_V2_ALL_15_PAIRED"
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 202608083
DEFAULT_WORK_ROOT = PROJECT / "outputs/rt_xls_v2_matched_clean_nested_v1"
DEFAULT_IMPORT_ROOT = WORKSPACE / "sua_exploration/results/rt_xls_v2_matched_clean_nested_v1/imported_cells"
DEFAULT_OUTPUT = WORKSPACE / "sua_exploration/results/rt_xls_v2_matched_clean_nested_v1/RT_XLS_V2_RC_MINUS_XLSV2_ALL15_v1.json"
DEFAULT_PLAN_OUTPUT = WORKSPACE / "sua_exploration/results/rt_xls_v2_matched_clean_nested_v1/RT_XLS_V2_LAUNCH_PLAN_v1.json"
DEFAULT_MB4 = WORKSPACE / "sua_exploration/results/rt_mb4_matched_clean_nested_v1/RT_MB4_MATCHED_FULL15_AGGREGATE_v1.json"
DEFAULT_AUDIT = WORKSPACE / "sua_exploration/results/rt_afc4_ls_null_strength_audit_v2/RT_AFC4_LS_NULL_STRENGTH_SUPPORT_AUDIT_v2.json"
TRAIN_ENTRY = PROJECT / "src/train.py"
EVAL_ENTRY = PROJECT / "src/rt_clean_nested_loso_xls_v2_eval.py"
EXPERIMENT = "rt_joint_afc4_xls_v2_m24_loso"


class XlsV2ContinuationError(RuntimeError):
    """A frozen XLSv2 orchestration or receipt invariant failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise XlsV2ContinuationError(message)


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
        raise XlsV2ContinuationError(f"invalid JSON receipt: {path}") from error
    _need(isinstance(value, dict), f"receipt must be a JSON object: {path}")
    return value


def _immutable(path: Path, label: str) -> None:
    _need(path.is_file(), f"{label} is missing: {path}")
    _need(stat.S_IMODE(path.stat().st_mode) == 0o444, f"{label} must be mode 0444: {path}")


def _write_immutable(path: Path, body: Mapping[str, Any]) -> str:
    _need(not path.exists(), f"refusing to overwrite immutable XLSv2 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(body), indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)
    _immutable(path, "new XLSv2 evidence")
    return _sha256(path)


def _finite(value: Any, label: str) -> float:
    _need(isinstance(value, (int, float)) and math.isfinite(float(value)), f"{label} must be finite")
    return float(value)


@dataclass(frozen=True)
class References:
    mb4_path: Path
    mb4_sha256: str
    rows: Mapping[int, Mapping[str, Any]]
    audit_path: Path
    audit_sha256: str
    permutation_sha_by_session: Mapping[str, str]


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
        imported=cell / "xls_v2_copy_import.json",
        log=cell / "worker.log",
    )


def load_references(*, mb4_aggregate: Path, xls_v2_audit: Path) -> References:
    mb4, audit = mb4_aggregate.resolve(), xls_v2_audit.resolve()
    _immutable(mb4, "MB4 Full15 prerequisite")
    _immutable(audit, "XLSv2 support audit")
    _need(_sha256(mb4) == MB4_SHA256, "MB4 Full15 SHA drift")
    _need(_sha256(audit) == AUDIT_SHA256, "XLSv2 support audit SHA drift")
    mb4_body, audit_body = _json(mb4), _json(audit)
    _need(mb4_body.get("schema") == MB4_SCHEMA and mb4_body.get("status") == MB4_STATUS, "MB4 schema/status drift")
    _need(mb4_body.get("seed") == SEED and mb4_body.get("full_comparator_arm") == FULL_ARM, "MB4 comparator drift")
    _need(audit_body.get("schema") == AUDIT_SCHEMA and audit_body.get("status") == AUDIT_STATUS, "XLSv2 audit schema/status drift")
    rows: dict[int, Mapping[str, Any]] = {}
    for row in mb4_body.get("rows", []):
        _need(isinstance(row, Mapping), "MB4 row is not an object")
        fold = row.get("fold")
        _need(isinstance(fold, int) and fold in FOLDS and fold not in rows, "MB4 does not contain one row per fold")
        _need(row.get("state") == "complete", f"MB4 fold {fold} is not complete")
        _need(isinstance(row.get("target_session"), str) and isinstance(row.get("inner_validation_session"), str), f"MB4 fold {fold} session binding absent")
        _finite(row.get("full_r2"), f"MB4 fold {fold} Full R2")
        _finite(row.get("query_windows_evaluated"), f"MB4 fold {fold} query count")
        rows[fold] = row
    _need(tuple(sorted(rows)) == FOLDS, "MB4 prerequisite lacks complete folds 0..14")
    permutations: dict[str, str] = {}
    for row in audit_body.get("fold_rows", []):
        _need(isinstance(row, Mapping) and isinstance(row.get("session_name"), str), "XLSv2 audit session row invalid")
        null = row.get("v2_random_cross_reach_null")
        _need(isinstance(null, Mapping), "XLSv2 audit null row absent")
        value = null.get("permutation_sha256")
        _need(isinstance(value, str) and len(value) == 64, "XLSv2 audit permutation SHA absent")
        _need(row["session_name"] not in permutations, "duplicate XLSv2 audit session")
        permutations[str(row["session_name"])] = value
    _need(len(permutations) == 15, "XLSv2 audit must contain exactly 15 sessions")
    _need(set(permutations) == {str(rows[fold]["target_session"]) for fold in FOLDS}, "MB4 and XLSv2 audit session sets disagree")
    return References(mb4, MB4_SHA256, rows, audit, AUDIT_SHA256, permutations)


def train_command(*, root: Path, fold: int, gpu: int) -> list[str]:
    paths = cell_paths(root, fold)
    return [
        sys.executable,
        str(TRAIN_ENTRY),
        f"experiment={EXPERIMENT}",
        f"run_id=rt_clean_nested_loso_m24_{ARM}",
        f"data.loso_fold={fold}",
        f"data.outer_loso_fold={fold}",
        f"data.side_feature_group={ARM}",
        f"data.xls_v2_support_audit_path={DEFAULT_AUDIT}",
        f"seed={SEED}",
        "test=false",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        f"hydra.run.dir={paths.cell / 'fit'}",
        f"paths.root_dir={PROJECT}",
        f"paths.log_dir={root / '_hydra_logs'}",
        f"paths.artifact_dir={root / '_artifacts'}",
    ]


def eval_command(*, root: Path, fold: int, checkpoint: Path | str) -> list[str]:
    paths = cell_paths(root, fold)
    return [
        sys.executable,
        str(EVAL_ENTRY),
        "--config", str(paths.config.resolve()),
        "--checkpoint", str(Path(checkpoint).resolve()),
        "--split-manifest", str(paths.split.resolve()),
        "--selection-receipt", str(paths.selection.resolve()),
        "--output", str(paths.outer.resolve()),
        "--outer-fold", str(fold),
        "--device", "cuda",
    ]


def _validate_selection(selection: Mapping[str, Any], *, fold: int) -> None:
    expected = {
        "schema": "rt_clean_nested_loso_selection_receipt_v1",
        "status": "PASS_FIT_INNER_SELECTION_ONLY",
        "arm": ARM,
        "outer_loso_fold": fold,
        "seed": SEED,
        "selected_by_metric": "val_heldin/r2_mean",
        "selected_metric_scope": "inner_validation_session_only",
        "formal_heldout_opened": False,
        "outer_target_loaded_during_fit": False,
        "outer_target_query_labels_read_during_fit": False,
    }
    for key, value in expected.items():
        _need(selection.get(key) == value, f"XLSv2 fold {fold} selection drift: {key}")
    for key in ("best_model_sha256", "config_sha256", "split_manifest_sha256"):
        _need(isinstance(selection.get(key), str) and len(selection[key]) == 64, f"XLSv2 fold {fold} selection SHA absent: {key}")


def _validate_split(split: Mapping[str, Any], *, fold: int, references: References) -> None:
    comparator = references.rows[fold]
    _need(split.get("task") == "rt" and split.get("development_only") is True, f"XLSv2 fold {fold} scope drift")
    _need(split.get("validation_protocol") == "nested_loso", f"XLSv2 fold {fold} validation protocol drift")
    _need(split.get("outer_loso_fold") == fold and split.get("loso_fold") == fold, f"XLSv2 fold {fold} fold drift")
    _need(split.get("requested_side_feature_group") == ARM, f"XLSv2 fold {fold} arm drift")
    arm = split.get("arm")
    _need(isinstance(arm, Mapping) and arm.get("canonical_arm") == ARM, f"XLSv2 fold {fold} canonical arm drift")
    _need(split.get("target_session") == comparator.get("target_session"), f"XLSv2 fold {fold} target session mismatch")
    _need(split.get("inner_validation_session") == comparator.get("inner_validation_session"), f"XLSv2 fold {fold} inner validation mismatch")
    calibration, query, nested, normalizer = (split.get(key) for key in ("calibration", "query", "nested_selection", "source_only_normalizer"))
    _need(all(isinstance(item, Mapping) for item in (calibration, query, nested, normalizer)), f"XLSv2 fold {fold} split sections absent")
    _need(calibration.get("budget_trials") == M and calibration.get("trial_index_range") == [0, M] and calibration.get("target_calibration_optimizer_steps") == 0, f"XLSv2 fold {fold} M24/no-BP drift")
    _need(query.get("query_start_trial") == QUERY_START and query.get("window_size_bins") == WINDOW_SIZE and query.get("full_window_after_support_required") is True, f"XLSv2 fold {fold} q24/window50 drift")
    for key, value in (("clean", True), ("outer_target_loaded_during_fit", False), ("outer_target_query_labels_read_during_fit", False), ("inner_validation_only_for_checkpoint_selection", True), ("checkpoint_metric", "val_heldin/r2_mean"), ("checkpoint_metric_scope", "inner_validation_session_only")):
        _need(nested.get(key) == value, f"XLSv2 fold {fold} nested selection drift: {key}")
    inner_train = list(split.get("inner_train_sessions", []))
    _need(len(inner_train) == 13, f"XLSv2 fold {fold} inner train count drift")
    _need(normalizer.get("fit_scope") == "inner_train_sessions_only" and normalizer.get("feature_group") == ARM, f"XLSv2 fold {fold} normalizer scope drift")
    _need(list(normalizer.get("fit_sessions", [])) == inner_train, f"XLSv2 fold {fold} normalizer sessions drift")
    _need(normalizer.get("excluded_outer_target_session") == comparator.get("target_session"), f"XLSv2 fold {fold} normalizer target exclusion drift")
    binding = split.get("xls_v2_support_audit")
    _need(isinstance(binding, Mapping) and binding.get("sha256") == references.audit_sha256, f"XLSv2 fold {fold} support audit binding drift")
    _need(binding.get("query_labels_available_to_generator") is False and binding.get("common_inverse_or_alignment_map") is False, f"XLSv2 fold {fold} leakage/alignment drift")
    per_session = binding.get("per_session_permutation_sha256")
    expected_fit = set(inner_train) | {str(split.get("inner_validation_session"))}
    _need(isinstance(per_session, Mapping) and set(per_session) == expected_fit, f"XLSv2 fold {fold} fit permutation scope drift")
    _need(comparator.get("target_session") not in per_session, f"XLSv2 fold {fold} target entered fit permutation map")
    for session in expected_fit:
        _need(per_session.get(session) == references.permutation_sha_by_session.get(session), f"XLSv2 fold {fold} permutation SHA drift: {session}")


def _validate_outer(outer: Mapping[str, Any], *, fold: int, references: References) -> float:
    comparator = references.rows[fold]
    expected = {
        "schema": OUTER_SCHEMA,
        "status": OUTER_STATUS,
        "arm": ARM,
        "outer_loso_fold": fold,
        "seed": SEED,
        "outer_target_session": comparator.get("target_session"),
        "query_start_trial": QUERY_START,
        "window_size": WINDOW_SIZE,
        "query_windows_evaluated": comparator.get("query_windows_evaluated"),
        "normalizer_fit_scope": "inner_train_sessions_only",
        "xls_v2_support_audit_sha256": references.audit_sha256,
        "xls_v2_target_permutation_sha256": references.permutation_sha_by_session[str(comparator.get("target_session"))],
        "target_backpropagation": False,
        "optimizer_present": False,
        "model_training_mode": False,
        "model_state_unchanged": True,
        "target_query_labels_used_for_calibration": False,
        "target_query_labels_used_for_normalization": False,
        "target_query_labels_used_for_checkpoint_selection": False,
        "target_query_labels_used_for_scoring_only": True,
        "query_labels_available_to_xls_v2_generator": False,
        "common_inverse_or_alignment_map": False,
    }
    for key, value in expected.items():
        _need(outer.get(key) == value, f"XLSv2 fold {fold} outer drift: {key}")
    _need(outer.get("model_state_sha256_before") == outer.get("model_state_sha256_after"), f"XLSv2 fold {fold} model state changed")
    return _finite(outer.get("r2_variance_weighted"), f"XLSv2 fold {fold} R2")


def validate_cell(*, root: Path, fold: int, references: References, require_import: bool, require_terminal: bool = True) -> dict[str, Any]:
    paths = cell_paths(root, fold)
    required = (paths.outer, paths.selection, paths.split, paths.config) + ((paths.terminal,) if require_terminal else ())
    exists = [path.exists() for path in required]
    if not any(exists):
        return {"state": "missing", "fold": fold, "cell": str(paths.cell)}
    _need(all(exists), f"XLSv2 fold {fold} receipt bundle is partial")
    outer, selection, split = _json(paths.outer), _json(paths.selection), _json(paths.split)
    _validate_selection(selection, fold=fold)
    _need(_sha256(paths.config) == selection.get("config_sha256"), f"XLSv2 fold {fold} config SHA mismatch")
    _need(_sha256(paths.split) == selection.get("split_manifest_sha256"), f"XLSv2 fold {fold} split SHA mismatch")
    _need(outer.get("checkpoint_sha256") == selection.get("best_model_sha256"), f"XLSv2 fold {fold} checkpoint SHA mismatch")
    _validate_split(split, fold=fold, references=references)
    score = _validate_outer(outer, fold=fold, references=references)
    if require_terminal:
        terminal = _json(paths.terminal)
        _need(terminal == {"status": "PASS", "arm": ARM, "fold": fold, "seed": SEED}, f"XLSv2 fold {fold} terminal drift")
    import_sha = None
    if require_import:
        _immutable(paths.imported, f"XLSv2 fold {fold} import receipt")
        imported = _json(paths.imported)
        _need(imported.get("schema") == IMPORT_SCHEMA and imported.get("status") == IMPORT_STATUS, f"XLSv2 fold {fold} import receipt drift")
        files = imported.get("imported_files")
        _need(isinstance(files, Mapping), f"XLSv2 fold {fold} import map absent")
        for name, path in (("outer", paths.outer), ("selection", paths.selection), ("split", paths.split), ("config", paths.config), ("terminal", paths.terminal)):
            item = files.get(name)
            _need(isinstance(item, Mapping) and item.get("sha256") == _sha256(path), f"XLSv2 fold {fold} imported SHA mismatch: {name}")
        import_sha = _sha256(paths.imported)
    full_r2 = float(references.rows[fold]["full_r2"])
    return {
        "state": "complete",
        "fold": fold,
        "target_session": references.rows[fold]["target_session"],
        "inner_validation_session": references.rows[fold]["inner_validation_session"],
        "query_windows_evaluated": references.rows[fold]["query_windows_evaluated"],
        "r_c_r2": full_r2,
        "xls_v2_r2": score,
        "r_c_minus_xls_v2": full_r2 - score,
        "xls_v2_target_permutation_sha256": references.permutation_sha_by_session[str(references.rows[fold]["target_session"])],
        "copy_import_sha256": import_sha,
    }


def inventory(*, root: Path, references: References, require_import: bool) -> dict[str, Any]:
    rows = []
    for fold in FOLDS:
        try:
            rows.append(validate_cell(root=root, fold=fold, references=references, require_import=require_import))
        except XlsV2ContinuationError as error:
            rows.append({"state": "incomparable", "fold": fold, "reason": str(error)})
    complete = [row for row in rows if row["state"] == "complete"]
    return {
        "schema": "rt_xls_v2_matched_inventory_v1",
        "status": "READ_ONLY_XLS_V2_PARTIAL_PREVIEW_NOT_A_CLAIM",
        "scope": "receipt_only_no_nwb_no_trainer_no_cuda",
        "rows": rows,
        "complete_folds": [row["fold"] for row in complete],
        "missing_folds": [row["fold"] for row in rows if row["state"] == "missing"],
        "incomparable_folds": [row["fold"] for row in rows if row["state"] == "incomparable"],
        "finalize_eligible": require_import and len(complete) == len(FOLDS) and not any(row["state"] == "incomparable" for row in rows),
    }


def copy_import(*, source_root: Path, import_root: Path, fold: int, references: References) -> dict[str, Any]:
    source = validate_cell(root=source_root, fold=fold, references=references, require_import=False)
    _need(source["state"] == "complete", f"cannot import incomplete XLSv2 fold {fold}")
    src, dst = cell_paths(source_root, fold), cell_paths(import_root, fold)
    _need(not dst.cell.exists(), f"refusing duplicate XLSv2 import: {dst.cell}")
    dst.cell.mkdir(parents=True, exist_ok=False)
    copied: dict[str, Any] = {}
    for name, origin, destination in (("outer", src.outer, dst.outer), ("selection", src.selection, dst.selection), ("split", src.split, dst.split), ("config", src.config, dst.config), ("terminal", src.terminal, dst.terminal)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = _sha256(origin)
        shutil.copyfile(origin, destination)
        _need(_sha256(destination) == digest, f"XLSv2 fold {fold} copy mismatch: {name}")
        destination.chmod(0o444)
        copied[name] = {"source_path": str(origin), "path": str(destination), "sha256": digest}
    body = {
        "schema": IMPORT_SCHEMA,
        "status": IMPORT_STATUS,
        "arm": ARM,
        "fold": fold,
        "seed": SEED,
        "mb4_aggregate_sha256": references.mb4_sha256,
        "xls_v2_support_audit_sha256": references.audit_sha256,
        "imported_files": copied,
        "scope": "copy_only_no_nwb_no_trainer_no_cuda_no_source_mutation",
    }
    digest = _write_immutable(dst.imported, body)
    validate_cell(root=import_root, fold=fold, references=references, require_import=True)
    return {"status": IMPORT_STATUS, "fold": fold, "receipt_sha256": digest}


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    _need(bool(ordered), "quantile input is empty")
    position = (len(ordered) - 1) * probability
    left, right = math.floor(position), math.ceil(position)
    return ordered[left] if left == right else ordered[left] * (right - position) + ordered[right] * (position - left)


def _sign_test(values: Sequence[float]) -> dict[str, Any]:
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    zero = len(values) - positive - negative
    nonzero = positive + negative
    tail = 1.0 if nonzero == 0 else sum(math.comb(nonzero, item) for item in range(min(positive, negative) + 1)) / (2 ** nonzero)
    return {"test": "exact_two_sided_binomial_sign_test", "positive": positive, "negative": negative, "zero": zero, "nonzero_pairs": nonzero, "p_value": min(1.0, 2.0 * tail), "ties": "excluded"}


def _bootstrap(values: Sequence[float]) -> dict[str, Any]:
    numbers = tuple(float(value) for value in values)
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(sum(numbers[rng.randrange(len(numbers))] for _ in numbers) / len(numbers) for _ in range(BOOTSTRAP_DRAWS))
    return {"resampling_unit": "outer_LOSO_fold", "draws": BOOTSTRAP_DRAWS, "rng_seed": BOOTSTRAP_SEED, "quantile_method": "linear_order_statistic", "lower_95": _quantile(means, 0.025), "upper_95": _quantile(means, 0.975)}


def finalize(*, import_root: Path, output: Path, references: References) -> dict[str, Any]:
    _need(not output.exists(), f"refusing to overwrite XLSv2 aggregate: {output}")
    report = inventory(root=import_root, references=references, require_import=True)
    _need(report["finalize_eligible"] is True, "XLSv2 finalizer requires all 15 immutable imported folds")
    rows = report["rows"]
    deltas = [float(row["r_c_minus_xls_v2"]) for row in rows]
    body = {
        "schema": AGGREGATE_SCHEMA,
        "status": AGGREGATE_STATUS,
        "arm": ARM,
        "reference_arm": FULL_ARM,
        "seed": SEED,
        "folds": list(FOLDS),
        "scope": "receipt_only_no_nwb_no_trainer_no_cuda_all_signed_deltas_retained",
        "mb4_prerequisite": {"path": str(references.mb4_path), "sha256": references.mb4_sha256},
        "xls_v2_support_audit": {"path": str(references.audit_path), "sha256": references.audit_sha256},
        "rows": rows,
        "r_c_minus_xls_v2": {
            "mean": sum(deltas) / len(deltas),
            "median": _quantile(deltas, 0.5),
            "positive_folds": sum(value > 0 for value in deltas),
            "negative_folds": sum(value < 0 for value in deltas),
            "exact_two_sided_sign_test": _sign_test(deltas),
            "paired_fold_bootstrap_95": _bootstrap(deltas),
        },
        "interpretation": "R-C minus XLSv2 tests whether correct support neural-velocity pairing matters beyond an audited strong pairing null. Every signed fold is retained.",
    }
    digest = _write_immutable(output, body)
    return {"status": AGGREGATE_STATUS, "output": str(output), "sha256": digest}


def copy_import_all_and_finalize(*, source_root: Path, import_root: Path, output: Path, references: References) -> dict[str, Any]:
    _need(not import_root.exists(), f"XLSv2 import root must be fresh: {import_root}")
    _need(not output.exists(), f"XLSv2 aggregate already exists: {output}")
    preview = inventory(root=source_root, references=references, require_import=False)
    _need(preview["complete_folds"] == list(FOLDS) and not preview["missing_folds"] and not preview["incomparable_folds"], "XLSv2 all-fold import requires exactly 15 complete raw cells")
    imports = [copy_import(source_root=source_root, import_root=import_root, fold=fold, references=references) for fold in FOLDS]
    aggregate = finalize(import_root=import_root, output=output, references=references)
    return {"status": "PASS_RT_XLS_V2_COPY_IMPORT_ALL_AND_FINALIZE", "folds": list(FOLDS), "imports": imports, "aggregate": aggregate}


def _ensure_root(root: Path) -> Path:
    resolved = root.resolve()
    _need(resolved != PROJECT.resolve() and resolved != WORKSPACE.resolve(), "XLSv2 work root is too broad")
    forbidden = ("rt_mb4_matched_clean_nested_v1", "rt_controls_continuation", "k4_rt_loso_v1")
    _need(not any(token in str(resolved) for token in forbidden), f"XLSv2 root overlaps sealed RT work: {resolved}")
    return resolved


def _prepare_root(root: Path) -> None:
    marker = root / "XLS_V2_CONTINUATION_ROOT_v1.json"
    body = {"schema": "rt_xls_v2_continuation_root_v1", "arm": ARM, "seed": SEED}
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
        marker.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
        return
    _need(marker.is_file() and _json(marker) == body, f"existing XLSv2 root marker drift: {root}")


def active_compute_commands_on_gpu(gpu: int) -> tuple[list[str], list[str]]:
    try:
        table = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
        uuids = {int(line.split(",", 1)[0].strip()): line.split(",", 1)[1].strip() for line in table.splitlines() if "," in line}
        _need(gpu in uuids, f"physical GPU {gpu} is unavailable")
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], text=True)
        ps_table = subprocess.check_output(["ps", "-eo", "args="], text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise XlsV2ContinuationError("cannot inspect GPU/RT writers safely") from error
    owners = []
    for line in apps.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and fields[1] == uuids[gpu]:
            try:
                owners.append(Path(f"/proc/{int(fields[0])}/cmdline").read_bytes().replace(b"\0", b" ").decode())
            except OSError:
                owners.append(f"<unreadable-gpu-pid:{fields[0]}>")
    rt = [line for line in ps_table.splitlines() if "rt_clean_nested" in line.lower()]
    return owners, rt


def _same_fold_writers(commands: Sequence[str], fold: int) -> list[str]:
    tokens = (f"data.loso_fold={fold}", f"data.outer_loso_fold={fold}")
    return [command for command in commands if any(token in command.split() for token in tokens)]


def _run_logged(command: Sequence[str], *, env: Mapping[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(shlex.quote(part) for part in command) + "\n")
        handle.flush()
        result = subprocess.run(list(command), cwd=PROJECT, env=dict(env), stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"[exit={result.returncode}]\n")
    _need(result.returncode == 0, f"XLSv2 subprocess failed with exit={result.returncode}")


def execute_partition(*, partition: str, work_root: Path, references: References) -> dict[str, Any]:
    _need(partition in PARTITIONS, f"unknown XLSv2 partition: {partition}")
    spec = PARTITIONS[partition]
    gpu = int(spec["gpu"])
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu), "XLSv2 worker CUDA_VISIBLE_DEVICES must equal its physical GPU")
    root = _ensure_root(work_root)
    _prepare_root(root)
    completed = []
    for fold in spec["folds"]:
        paths = cell_paths(root, int(fold))
        created = False
        try:
            _need(not paths.cell.exists(), f"XLSv2 cell already exists: {paths.cell}")
            owners, rt_writers = active_compute_commands_on_gpu(gpu)
            _need(not owners, f"physical GPU {gpu} has active compute owners: {owners}")
            _need(not _same_fold_writers(rt_writers, int(fold)), f"same-fold RT writer collision for fold {fold}")
            paths.cell.mkdir(parents=True, exist_ok=False)
            created = True
            env = dict(os.environ)
            _run_logged(train_command(root=root, fold=int(fold), gpu=gpu), env=env, log=paths.log)
            selection = _json(paths.selection)
            _validate_selection(selection, fold=int(fold))
            checkpoint = Path(str(selection.get("best_model_path", "")))
            _need(checkpoint.is_file(), f"XLSv2 fold {fold} selected checkpoint absent")
            _run_logged(eval_command(root=root, fold=int(fold), checkpoint=checkpoint), env=env, log=paths.log)
            validate_cell(root=root, fold=int(fold), references=references, require_import=False, require_terminal=False)
            paths.terminal.write_text(json.dumps({"status": "PASS", "arm": ARM, "fold": int(fold), "seed": SEED}, sort_keys=True) + "\n", encoding="utf-8")
            completed.append(int(fold))
        except Exception as error:
            if created and not paths.terminal.exists():
                paths.terminal.write_text(json.dumps({"status": "FAILED", "arm": ARM, "fold": int(fold), "seed": SEED, "error": repr(error)}, sort_keys=True) + "\n", encoding="utf-8")
            raise
    return {"status": "PASS_RT_XLS_V2_STATIC_PARTITION", "partition": partition, "folds": completed}


def build_plan(*, work_root: Path, import_root: Path, output: Path, references: References) -> dict[str, Any]:
    root = _ensure_root(work_root)
    _need(not root.exists(), f"fresh XLSv2 work root must not exist: {root}")
    _need(not import_root.exists(), f"fresh XLSv2 import root must not exist: {import_root}")
    _need(not output.exists(), f"fresh XLSv2 aggregate must not exist: {output}")
    _need(set(GPU1_ASC_FOLDS).isdisjoint(GPU0_DESC_FOLDS) and set(GPU1_ASC_FOLDS) | set(GPU0_DESC_FOLDS) == set(FOLDS), "XLSv2 static partitions are not an exact disjoint cover")
    partitions = {}
    worker_commands = {}
    for name, spec in PARTITIONS.items():
        gpu = int(spec["gpu"])
        cells = []
        for fold in spec["folds"]:
            paths = cell_paths(root, int(fold))
            cells.append({
                "fold": int(fold),
                "target_session": references.rows[int(fold)]["target_session"],
                "inner_validation_session": references.rows[int(fold)]["inner_validation_session"],
                "query_windows_evaluated": references.rows[int(fold)]["query_windows_evaluated"],
                "target_permutation_sha256": references.permutation_sha_by_session[str(references.rows[int(fold)]["target_session"])],
                "train_command": train_command(root=root, fold=int(fold), gpu=gpu),
                "eval_command_template": eval_command(root=root, fold=int(fold), checkpoint=paths.cell / "fit/checkpoints/<inner-selected>.ckpt"),
            })
        inner = " ".join([
            "set -euo pipefail;", "cd", shlex.quote(str(PROJECT)), ";",
            f"CUDA_VISIBLE_DEVICES={gpu}", shlex.quote(sys.executable),
            "scripts/rt_xls_v2_matched_continuation_v1.py", "--run-partition", name,
            "--work-root", shlex.quote(str(root)), "--mb4-aggregate", shlex.quote(str(references.mb4_path)),
            "--xls-v2-audit", shlex.quote(str(references.audit_path)),
        ])
        session = f"rt_xls_v2_{name}"
        worker_commands[name] = "tmux new-session -d -s " + shlex.quote(session) + " " + shlex.quote(inner)
        partitions[name] = {"physical_gpu": gpu, "folds": list(spec["folds"]), "cells": cells, "tmux_session": session}
    workers = ["rt_xls_v2_gpu1_asc", "rt_xls_v2_gpu0_desc"]
    final_inner = " ".join([
        "set -euo pipefail;", "cd", shlex.quote(str(PROJECT)), ";",
        "for session in", " ".join(shlex.quote(item) for item in workers), "; do",
        "tmux has-session -t", '"$session"', "2>/dev/null || { echo", '"required XLSv2 worker session was never observed: $session"', ">&2; exit 2; }; done;",
        "while tmux has-session -t", shlex.quote(workers[0]), "2>/dev/null || tmux has-session -t", shlex.quote(workers[1]), "2>/dev/null; do sleep 15; done;",
        shlex.quote(sys.executable), "scripts/rt_xls_v2_matched_continuation_v1.py", "--copy-import-all-and-finalize",
        "--work-root", shlex.quote(str(root)), "--import-root", shlex.quote(str(import_root)), "--aggregate-output", shlex.quote(str(output)),
        "--mb4-aggregate", shlex.quote(str(references.mb4_path)), "--xls-v2-audit", shlex.quote(str(references.audit_path)),
    ])
    finalizer = "tmux new-session -d -s rt_xls_v2_import_finalize_after_workers " + shlex.quote(final_inner)
    code_paths = [
        PROJECT / "src/data/afc4_xls_v2.py",
        PROJECT / "src/data/afc4_xls_v2_adapter.py",
        PROJECT / "src/data/falcon_k4_features.py",
        PROJECT / "src/data/falcon_datamodule.py",
        PROJECT / "src/data/rt_nested_loso_datamodule.py",
        PROJECT / "src/rt_clean_nested_loso_xls_v2_eval.py",
        PROJECT / "configs/experiment/rt_joint_afc4_xls_v2_m24_loso.yaml",
        PROJECT / "scripts/rt_xls_v2_matched_continuation_v1.py",
    ]
    return {
        "schema": "rt_xls_v2_matched_launch_plan_v1",
        "status": "PLAN_ONLY_NO_GPU_NO_NWB_NO_TRAINER_NO_TMUX",
        "arm": ARM,
        "reference_arm": FULL_ARM,
        "seed": SEED,
        "protocol": {
            "support": "chronological trials [0,24)", "query": "q24/window50/full-window-disjoint",
            "fit": "fresh source training per fold; 13 inner train and inner-validation-only checkpoint selection",
            "outer": "one-shot; no backprop/optimizer; state invariant; query labels scoring only",
            "reuse_r_c_checkpoint": False, "common_inverse_or_alignment_map": False,
            "stop_rule": "all 15 folds; retain negative results; no score-dependent stopping",
        },
        "roots": {"work": str(root), "import": str(import_root.resolve()), "aggregate": str(output.resolve())},
        "references": {
            "mb4_aggregate": {"path": str(references.mb4_path), "sha256": references.mb4_sha256},
            "xls_v2_support_audit": {"path": str(references.audit_path), "sha256": references.audit_sha256, "per_session_permutation_sha256": dict(references.permutation_sha_by_session)},
        },
        "partitions": partitions,
        "worker_launch_commands": worker_commands,
        "finalizer": {"tmux_session": "rt_xls_v2_import_finalize_after_workers", "requires_workers_observed": workers, "launch_command": finalizer},
        "code_sha256": {str(path.relative_to(WORKSPACE)): _sha256(path) for path in code_paths},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--run-partition", choices=tuple(PARTITIONS))
    modes.add_argument("--preview", action="store_true")
    modes.add_argument("--copy-import", action="store_true")
    modes.add_argument("--copy-import-all-and-finalize", action="store_true")
    modes.add_argument("--finalize", action="store_true")
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--import-root", type=Path, default=DEFAULT_IMPORT_ROOT)
    parser.add_argument("--aggregate-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--mb4-aggregate", type=Path, default=DEFAULT_MB4)
    parser.add_argument("--xls-v2-audit", type=Path, default=DEFAULT_AUDIT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    references = load_references(mb4_aggregate=args.mb4_aggregate, xls_v2_audit=args.xls_v2_audit)
    if args.run_partition:
        result = execute_partition(partition=args.run_partition, work_root=args.work_root, references=references)
    elif args.preview:
        result = inventory(root=args.work_root, references=references, require_import=False)
    elif args.copy_import:
        _need(args.fold is not None, "--copy-import requires --fold")
        result = copy_import(source_root=args.work_root, import_root=args.import_root, fold=args.fold, references=references)
    elif args.copy_import_all_and_finalize:
        result = copy_import_all_and_finalize(source_root=args.work_root, import_root=args.import_root, output=args.aggregate_output, references=references)
    elif args.finalize:
        result = finalize(import_root=args.import_root, output=args.aggregate_output, references=references)
    else:
        result = build_plan(work_root=args.work_root, import_root=args.import_root, output=args.aggregate_output, references=references)
        if args.plan and args.plan_output is not None:
            digest = _write_immutable(args.plan_output, result)
            result = {"status": result["status"], "plan": str(args.plan_output), "sha256": digest, "worker_launch_commands": result["worker_launch_commands"], "finalizer": result["finalizer"]}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

