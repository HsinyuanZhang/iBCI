#!/usr/bin/env python3
"""CPU-only paired aggregation for RT Stage-R R-C versus R-S(B2,D1024).

The reference R-C matrix is the immutable 15-fold clean nested-LOSO aggregate.
R-S is the separately retrained B2/D1024, no-side-carrier path: imported fold
0, locally run folds 1--2, and the folds 3--14 serial-supervisor cells.  This
tool reads JSON/configuration receipts only.  It never imports an RT data
module, opens an NWB, starts a Trainer, creates CUDA, or evaluates a model.

There are two deliberately separate modes:

* default aggregate mode requires *all* 15 validated pairs and writes one
  immutable receipt; and
* ``--preview`` is read-only and may show already-complete pairs, but is
  explicitly not a formal aggregate or paper claim (even if all inputs happen
  to be present).
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
import stat
import sys
from typing import Any, Mapping, Sequence


WORKSPACE = Path(__file__).resolve().parents[2]
FOLDS = tuple(range(15))
SEED = 42
M = 24
QUERY_START = 24
WINDOW_SIZE = 50
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 20260808
R_C_IDENTITY_PARAMETERS = 18_290
R_S_B2_D1024_IDENTITY_PARAMETERS = 4_353_074

SCHEMA = "rt_stage_r_rc_vs_rs_b2_d1024_paired_aggregate_v1"
STATUS = "PASS_RT_STAGE_R_RC_VS_RS_B2_D1024_ALL_15_PAIRED"
PREVIEW_SCHEMA = "rt_stage_r_rc_vs_rs_b2_d1024_partial_preview_v1"
PREVIEW_STATUS = "READ_ONLY_PARTIAL_PREVIEW_NOT_PAPER_CLAIM"

DEFAULT_R_C_AGGREGATE = WORKSPACE / "sua_exploration/results/k4_rt_loso_v1/rt_seed42_clean_nested_aggregate.json"
DEFAULT_R_C_SEAL = WORKSPACE / "sua_exploration/results/k4_rt_loso_v1/rt_seed42_clean_nested_seal.marker"
DEFAULT_FOLD0_COMPARISON = WORKSPACE / "sua_exploration/results/k4_rt_loso_v1/RT_STAGE_R_D1024_FOLD0_COMPARISON_v1.json"
DEFAULT_RS_FOLD0_ROOT = WORKSPACE / "streaming_calibration_exp/outputs/rt_stage_r_b2_imported_remote/fold_00/seed_42"
DEFAULT_RS_LOCAL_ROOT = WORKSPACE / "streaming_calibration_exp/outputs/rt_stage_r_b2_local3090/gpu_runs_zero4_v2/b2_d1024_zero4"
DEFAULT_RS_SUPERVISOR_ROOT = WORKSPACE / "streaming_calibration_exp/outputs/rt_stage_r_b2_local3090/supervisor_folds03_14_v1"


class AggregateError(RuntimeError):
    """Raised when a source receipt is absent, altered, or incomparable."""


@dataclass(frozen=True)
class RsPaths:
    outer: Path
    selection: Path
    split: Path
    config: Path
    terminal: Path | None
    source: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AggregateError(f"expected JSON object: {path}")
    return value


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise AggregateError(message)


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        raise AggregateError(f"{label} must be finite")
    return float(value)


def _immutable(path: Path, label: str) -> None:
    _need(path.is_file(), f"{label} is missing: {path}")
    _need(stat.S_IMODE(path.stat().st_mode) == 0o444, f"{label} must be immutable mode 0444: {path}")


def default_rs_paths(
    *, fold0_root: Path = DEFAULT_RS_FOLD0_ROOT, local_root: Path = DEFAULT_RS_LOCAL_ROOT,
    supervisor_root: Path = DEFAULT_RS_SUPERVISOR_ROOT,
) -> dict[int, RsPaths]:
    """Return the one legal receipt location for every B2/D1024 fold."""

    result: dict[int, RsPaths] = {}
    result[0] = RsPaths(
        outer=fold0_root / "outer_target_eval.json",
        selection=fold0_root / "rt_nested_selection_receipt.json",
        split=fold0_root / "split_manifest.json",
        config=fold0_root / "config.yaml",
        terminal=None,
        source="imported_remote_fold0",
    )
    for fold in (1, 2):
        cell = local_root / f"fold_{fold:02d}" / "seed_42"
        result[fold] = RsPaths(
            outer=cell / "outer_target_eval.json",
            selection=cell / "fit/rt_nested_selection_receipt.json",
            split=cell / "fit/split_manifest.json",
            config=cell / "fit/.hydra/config.yaml",
            terminal=None,
            source="local_3090_fold1_2",
        )
    for fold in range(3, 15):
        cell = supervisor_root / "cells/b2_d1024_zero4" / f"fold_{fold:02d}" / "seed_42"
        result[fold] = RsPaths(
            outer=cell / "outer_target_eval.json",
            selection=cell / "fit/rt_nested_selection_receipt.json",
            split=cell / "fit/split_manifest.json",
            config=cell / "fit/.hydra/config.yaml",
            terminal=cell / "cell_terminal.json",
            source="local_3090_supervisor_fold3_14",
        )
    return result


def _validate_baseline(aggregate_path: Path, seal_path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Bind and validate all 15 immutable R-C cells without opening their NWBs."""

    _immutable(aggregate_path, "frozen R-C aggregate")
    _immutable(seal_path, "frozen R-C aggregate seal")
    aggregate, seal = _json(aggregate_path), _json(seal_path)
    _need(seal.get("schema") == "rt_seed42_clean_nested_loso_seal_marker_v1", "R-C seal schema drift")
    _need(seal.get("status") == "PASS_RT_SEALED", "R-C seal status is not passing")
    _need(seal.get("aggregate_sha256") == _sha256(aggregate_path), "R-C aggregate hash differs from sealed marker")
    _need(tuple(seal.get("folds", ())) == FOLDS and seal.get("seed") == SEED, "R-C seal fold/seed drift")
    _need(aggregate.get("task") == "rt" and aggregate.get("seed") == SEED, "R-C aggregate task/seed drift")
    audits = aggregate.get("audits")
    _need(isinstance(audits, Mapping), "R-C aggregate audits are missing")
    for field in (
        "exact_main_grid", "all_model_state_unchanged", "all_optimizer_absent",
        "all_target_backpropagation_false", "all_target_loaded_during_fit_false",
        "all_target_query_labels_read_during_fit_false",
        "all_target_query_labels_used_for_calibration_false",
        "all_target_query_labels_used_for_normalization_false",
        "all_target_query_labels_used_for_checkpoint_selection_false",
        "all_target_query_labels_used_for_scoring_only_true",
    ):
        _need(audits.get(field) is True, f"R-C aggregate audit failed: {field}")
    expected_session = audits.get("expected_target_sessions_by_fold")
    expected_windows = audits.get("query_windows_evaluated_by_arm_fold", {}).get("afc4_vel")
    _need(isinstance(expected_session, Mapping) and isinstance(expected_windows, Mapping), "R-C expected fold maps missing")
    cells = aggregate.get("cells")
    _need(isinstance(cells, list), "R-C aggregate cells missing")
    rows: dict[int, dict[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping) or cell.get("arm") != "afc4_vel":
            continue
        fold = cell.get("fold")
        _need(isinstance(fold, int) and fold in FOLDS and fold not in rows, "R-C cells are not one-per-fold")
        _need(cell.get("target_session") == expected_session.get(str(fold)), f"R-C fold {fold} target session drift")
        _need(cell.get("inner_validation_session") != cell.get("target_session"), f"R-C fold {fold} reuses outer target as inner validation")
        _need(cell.get("query_start_trial") == QUERY_START and cell.get("window_size") == WINDOW_SIZE, f"R-C fold {fold} M24/query contract drift")
        _need(cell.get("query_windows_evaluated") == expected_windows.get(str(fold)), f"R-C fold {fold} query-window audit drift")
        for field, expected in (
            ("model_state_unchanged", True), ("optimizer_present", False), ("model_training_mode", False),
            ("target_backpropagation", False), ("target_query_labels_used_for_calibration", False),
            ("target_query_labels_used_for_normalization", False),
            ("target_query_labels_used_for_checkpoint_selection", False),
            ("target_query_labels_used_for_scoring_only", True),
            ("target_support_calibration_labels_used", True), ("target_support_calibration_velocity_used", True),
        ):
            _need(cell.get(field) == expected, f"R-C fold {fold} forward-only contract drift: {field}")
        provenance = cell.get("provenance", {}).get("outer_target_eval", {})
        _need(isinstance(provenance, Mapping) and provenance.get("status") == "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP", f"R-C fold {fold} outer receipt provenance missing")
        _need(_is_sha(provenance.get("sha256")), f"R-C fold {fold} outer receipt SHA missing")
        _finite(cell.get("r2_variance_weighted"), f"R-C fold {fold} R2")
        rows[fold] = dict(cell)
    _need(tuple(sorted(rows)) == FOLDS, "frozen R-C aggregate does not contain exactly 15 afc4_vel cells")
    return rows, {"aggregate": {"path": str(aggregate_path), "sha256": _sha256(aggregate_path)},
                  "seal": {"path": str(seal_path), "sha256": _sha256(seal_path)}}


def _validate_selection(selection: Mapping[str, Any], *, fold: int) -> None:
    required = {
        "schema": "rt_clean_nested_loso_selection_receipt_v1",
        "status": "PASS_FIT_INNER_SELECTION_ONLY",
        "arm": "zero4", "seed": SEED, "outer_loso_fold": fold,
        "selected_by_metric": "val_heldin/r2_mean",
        "selected_metric_scope": "inner_validation_session_only",
        "formal_heldout_opened": False,
        "outer_target_loaded_during_fit": False,
        "outer_target_query_labels_read_during_fit": False,
    }
    for field, expected in required.items():
        _need(selection.get(field) == expected, f"R-S fold {fold} selection contract drift: {field}")
    for field in ("best_model_sha256", "config_sha256", "split_manifest_sha256"):
        _need(_is_sha(selection.get(field)), f"R-S fold {fold} selection lacks SHA: {field}")


def _validate_config(path: Path, selection: Mapping[str, Any], *, fold: int) -> None:
    _need(path.is_file() and _sha256(path) == selection.get("config_sha256"), f"R-S fold {fold} config SHA mismatch")
    text = path.read_text(encoding="utf-8")
    for fragment in (
        "run_id: rt_clean_nested_loso_m24_b2_d1024_zero4", "calibration_n_trials: 24",
        "query_start_trial:", "side_feature_group: zero4", "freeze_decoder: false",
        "loss_mode: task_only", "id_hidden_dim: 1024", "variant: B2",
    ):
        _need(fragment in text, f"R-S fold {fold} B2/D1024 config lacks {fragment!r}")


def _validate_split(split: Mapping[str, Any], selection: Mapping[str, Any], *, rc: Mapping[str, Any], fold: int) -> None:
    _need(split.get("task") == "rt" and split.get("development_only") is True, f"R-S fold {fold} split task/scope drift")
    _need(split.get("validation_protocol") == "nested_loso", f"R-S fold {fold} split protocol drift")
    _need(split.get("outer_loso_fold") == fold and split.get("loso_fold") == fold, f"R-S fold {fold} split fold drift")
    _need(split.get("target_session") == rc["target_session"], f"R-S fold {fold} target session does not pair R-C")
    _need(split.get("inner_validation_session") == rc["inner_validation_session"], f"R-S fold {fold} inner validation session does not pair R-C")
    _need(split.get("requested_side_feature_group") == "zero4", f"R-S fold {fold} split arm drift")
    arm = split.get("arm", {})
    _need(isinstance(arm, Mapping) and arm.get("canonical_arm") == "zero4", f"R-S fold {fold} canonical arm drift")
    protocol, calibration, query, nested = (split.get(name) for name in ("protocol", "calibration", "query", "nested_selection"))
    _need(isinstance(protocol, Mapping) and isinstance(calibration, Mapping) and isinstance(query, Mapping) and isinstance(nested, Mapping), f"R-S fold {fold} split sections missing")
    _need(protocol.get("decode_target") == "2D cursor velocity", f"R-S fold {fold} target type drift")
    _need(calibration.get("budget_trials") == M and calibration.get("trial_index_range") == [0, M], f"R-S fold {fold} M24 split drift")
    _need(calibration.get("target_calibration_optimizer_steps") == 0, f"R-S fold {fold} calibration used target optimizer")
    _need(query.get("query_start_trial") == QUERY_START and query.get("window_size_bins") == WINDOW_SIZE, f"R-S fold {fold} query contract drift")
    for field, expected in (
        ("clean", True), ("outer_target_loaded_during_fit", False),
        ("outer_target_query_labels_read_during_fit", False),
        ("inner_validation_only_for_checkpoint_selection", True),
        ("checkpoint_metric", "val_heldin/r2_mean"),
        ("checkpoint_metric_scope", "inner_validation_session_only"),
    ):
        _need(nested.get(field) == expected, f"R-S fold {fold} split forward-only contract drift: {field}")
    _need(selection.get("split_manifest_sha256") is not None, f"R-S fold {fold} selection split binding absent")


def _validate_outer(outer: Mapping[str, Any], *, rc: Mapping[str, Any], fold: int) -> None:
    required = {
        "schema": "rt_clean_nested_loso_outer_eval_v1", "status": "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP",
        "arm": "zero4", "outer_loso_fold": fold, "seed": SEED,
        "query_start_trial": QUERY_START, "window_size": WINDOW_SIZE,
        "target_backpropagation": False, "optimizer_present": False, "model_training_mode": False,
        "model_state_unchanged": True, "target_query_labels_used_for_calibration": False,
        "target_query_labels_used_for_normalization": False,
        "target_query_labels_used_for_checkpoint_selection": False,
        "target_query_labels_used_for_scoring_only": True,
        "target_support_calibration_labels_used": True, "target_support_calibration_velocity_used": True,
    }
    for field, expected in required.items():
        _need(outer.get(field) == expected, f"R-S fold {fold} outer forward-only contract drift: {field}")
    _need(outer.get("outer_target_session") == rc["target_session"], f"R-S fold {fold} target session does not pair R-C")
    _need(outer.get("query_windows_evaluated") == rc["query_windows_evaluated"], f"R-S fold {fold} query windows do not pair R-C")
    _need(outer.get("model_state_sha256_before") == outer.get("model_state_sha256_after"), f"R-S fold {fold} outer forward changed model state")
    _need(_is_sha(outer.get("checkpoint_sha256")), f"R-S fold {fold} outer checkpoint SHA missing")
    _finite(outer.get("r2_variance_weighted"), f"R-S fold {fold} R2")


def _validate_supervisor_terminal(path: Path | None, *, fold: int) -> dict[str, Any] | None:
    if path is None:
        return None
    terminal = _json(path)
    _need(
        terminal.get("schema") in {
            "rt_stage_r_b2_d1024_fold_terminal_v1",
            "rt_stage_r_b2_d1024_fold_terminal_v2",
        },
        f"R-S fold {fold} supervisor terminal schema drift",
    )
    _need(terminal.get("status") == "PASS_CPU_ONE_SHOT_OUTER_EVAL", f"R-S fold {fold} supervisor did not finish outer evaluation")
    _need(terminal.get("fold") == fold and terminal.get("seed") == SEED and terminal.get("arm") == "zero4", f"R-S fold {fold} supervisor terminal identity drift")
    _need(terminal.get("formal_heldout_opened") is False, f"R-S fold {fold} supervisor opened formal heldout")
    return terminal


def _validate_rs_fold(paths: RsPaths, *, rc: Mapping[str, Any], fold: int) -> dict[str, Any]:
    for label, path in (("outer", paths.outer), ("selection", paths.selection), ("split", paths.split), ("config", paths.config)):
        _need(path.is_file(), f"R-S fold {fold} required {label} receipt is missing: {path}")
    outer, selection, split = _json(paths.outer), _json(paths.selection), _json(paths.split)
    _validate_selection(selection, fold=fold)
    _validate_config(paths.config, selection, fold=fold)
    _need(_sha256(paths.split) == selection.get("split_manifest_sha256"), f"R-S fold {fold} split SHA mismatch")
    _validate_split(split, selection, rc=rc, fold=fold)
    _validate_outer(outer, rc=rc, fold=fold)
    _need(outer.get("checkpoint_sha256") == selection.get("best_model_sha256"), f"R-S fold {fold} selected checkpoint SHA mismatch")
    terminal = _validate_supervisor_terminal(paths.terminal, fold=fold)
    files = {
        "outer": {"path": str(paths.outer), "sha256": _sha256(paths.outer)},
        "selection": {"path": str(paths.selection), "sha256": _sha256(paths.selection)},
        "split": {"path": str(paths.split), "sha256": _sha256(paths.split)},
        "config": {"path": str(paths.config), "sha256": _sha256(paths.config)},
    }
    if paths.terminal is not None:
        files["supervisor_terminal"] = {"path": str(paths.terminal), "sha256": _sha256(paths.terminal)}
    return {
        "r2": float(outer["r2_variance_weighted"]), "source": paths.source, "files": files,
        "selected_epoch": selection.get("selected_epoch"), "selected_global_step": selection.get("selected_global_step"),
        "terminal": terminal,
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    _need(bool(sorted_values) and 0.0 <= probability <= 1.0, "invalid bootstrap quantile input")
    position = (len(sorted_values) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(sorted_values[low])
    fraction = position - low
    return float(sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction)


def bootstrap_mean_ci(values: Sequence[float], *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    _need(len(values) > 0 and draws > 0, "bootstrap requires nonempty finite values and positive draws")
    numbers = tuple(_finite(value, "bootstrap delta") for value in values)
    rng = random.Random(seed)
    count = len(numbers)
    means = sorted(sum(numbers[rng.randrange(count)] for _ in range(count)) / count for _ in range(draws))
    return {
        "resampling_unit": "outer_LOSO_fold",
        "draws": draws, "rng_seed": seed, "quantile_method": "linear_order_statistic",
        "lower_95": _quantile(means, 0.025), "upper_95": _quantile(means, 0.975),
    }


def exact_two_sided_sign_test(values: Sequence[float]) -> dict[str, Any]:
    signs = [1 if _finite(value, "sign-test delta") > 0.0 else -1 if value < 0.0 else 0 for value in values]
    positive, negative, zero = signs.count(1), signs.count(-1), signs.count(0)
    n = positive + negative
    tail = sum(math.comb(n, index) for index in range(min(positive, negative) + 1)) / float(2 ** n) if n else 1.0
    return {
        "test": "exact_two_sided_binomial_sign_test", "positive": positive, "negative": negative,
        "zero": zero, "nonzero_pairs": n, "p_value": min(1.0, 2.0 * tail),
        "ties": "excluded_from_sign_test",
    }


def _validate_cost(comparison_path: Path, *, rc0: float, rs0: float) -> dict[str, Any]:
    _immutable(comparison_path, "Stage-R fold0 comparison")
    comparison = _json(comparison_path)
    _need(comparison.get("schema") == "rt_stage_r_d1024_fold0_paired_comparison_v1", "fold0 cost comparison schema drift")
    arms = comparison.get("arms", {})
    rc, rs = arms.get("r_c_b3s_continuous_velocity_carrier", {}), arms.get("r_s_b2_d1024_spint_scale_identity", {})
    _need(isinstance(rc, Mapping) and isinstance(rs, Mapping), "fold0 cost comparison arms missing")
    _need(rc.get("identity_encoder_parameters") == R_C_IDENTITY_PARAMETERS, "R-C parameter count drift")
    _need(rs.get("identity_encoder_parameters") == R_S_B2_D1024_IDENTITY_PARAMETERS, "R-S B2/D1024 parameter count drift")
    _need(abs(_finite(rc.get("r2_variance_weighted"), "fold0 R-C reference") - rc0) <= 1.0e-12, "fold0 R-C comparison does not bind frozen aggregate")
    _need(abs(_finite(rs.get("r2_variance_weighted"), "fold0 R-S reference") - rs0) <= 1.0e-12, "fold0 R-S comparison does not bind imported receipt")
    ratio = R_S_B2_D1024_IDENTITY_PARAMETERS / R_C_IDENTITY_PARAMETERS
    _need(abs(_finite(comparison.get("paired_deltas", {}).get("r_s_d1024_to_r_c_identity_parameter_ratio"), "fold0 parameter ratio") - ratio) <= 1.0e-12, "fold0 parameter ratio drift")
    return {
        "r_c_identity_encoder_parameters": R_C_IDENTITY_PARAMETERS,
        "r_s_b2_d1024_identity_encoder_parameters": R_S_B2_D1024_IDENTITY_PARAMETERS,
        "r_s_to_r_c_ratio": ratio,
        "comparison_path": str(comparison_path), "comparison_sha256": _sha256(comparison_path),
    }


def _paired_rows(rc_rows: Mapping[int, Mapping[str, Any]], rs_paths: Mapping[int, RsPaths], *, require_all: bool) -> tuple[list[dict[str, Any]], list[int]]:
    present: list[dict[str, Any]] = []
    missing: list[int] = []
    for fold in FOLDS:
        paths = rs_paths[fold]
        files = (paths.outer, paths.selection, paths.split, paths.config) + (() if paths.terminal is None else (paths.terminal,))
        if not all(path.is_file() for path in files):
            missing.append(fold)
            continue
        rs = _validate_rs_fold(paths, rc=rc_rows[fold], fold=fold)
        delta = float(rc_rows[fold]["r2_variance_weighted"] - rs["r2"])
        present.append({
            "fold": fold, "target_session": rc_rows[fold]["target_session"],
            "inner_validation_session": rc_rows[fold]["inner_validation_session"],
            "query_windows_evaluated": rc_rows[fold]["query_windows_evaluated"],
            "r_c_r2": float(rc_rows[fold]["r2_variance_weighted"]), "r_s_b2_d1024_r2": rs["r2"],
            "r_c_minus_r_s_b2_d1024": delta, "r_s_source": rs["source"],
            "r_s_files": rs["files"],
            "r_c_outer_receipt": dict(rc_rows[fold]["provenance"]["outer_target_eval"]),
        })
    if require_all:
        _need(not missing and len(present) == len(FOLDS), f"formal aggregate requires exactly 15/15 validated pairs; missing folds={missing}")
    return present, missing


def _statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["r_c_minus_r_s_b2_d1024"]) for row in rows]
    ordered = sorted(deltas)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    return {
        "n_pairs": len(deltas), "mean_r_c_minus_r_s_b2_d1024": sum(deltas) / len(deltas),
        "median_r_c_minus_r_s_b2_d1024": median,
        "positive_fold_count": sum(value > 0.0 for value in deltas),
        "exact_sign_test": exact_two_sided_sign_test(deltas),
        "paired_fold_bootstrap_95": bootstrap_mean_ci(deltas),
    }


def preview(*, rc_aggregate: Path, rc_seal: Path, rs_paths: Mapping[int, RsPaths]) -> dict[str, Any]:
    """Read completed pairs only; deliberately creates no receipt or claim."""

    rc_rows, binding = _validate_baseline(rc_aggregate, rc_seal)
    rows, missing = _paired_rows(rc_rows, rs_paths, require_all=False)
    return {
        "schema": PREVIEW_SCHEMA, "status": PREVIEW_STATUS,
        "formal_aggregate_written": False, "paper_claim": False,
        "reason": "preview may omit folds and is not an authorized aggregate endpoint",
        "validated_pair_count": len(rows), "required_pair_count": len(FOLDS), "missing_folds": missing,
        "r_c_binding": binding, "validated_rows": rows,
        "statistics_if_any": None if not rows else _statistics(rows),
        "data_scope": {"nwb_opened": False, "cuda_constructed": False, "trainer_constructed": False},
    }


def aggregate(*, rc_aggregate: Path, rc_seal: Path, fold0_comparison: Path,
              rs_paths: Mapping[int, RsPaths], output: Path) -> dict[str, Any]:
    """Create an immutable formal receipt only after 15/15 strict pairing."""

    _need(not output.exists(), f"refusing to overwrite formal RT aggregate: {output}")
    rc_rows, binding = _validate_baseline(rc_aggregate, rc_seal)
    rows, missing = _paired_rows(rc_rows, rs_paths, require_all=True)
    _need(not missing, "formal RT aggregate cannot have missing folds")
    cost = _validate_cost(fold0_comparison, rc0=rows[0]["r_c_r2"], rs0=rows[0]["r_s_b2_d1024_r2"])
    payload = {
        "schema": SCHEMA, "status": STATUS, "development_only": True,
        "formal_heldout_opened": False, "paper_claim": "development RT nested-LOSO paired evidence only; not a formal-heldout claim",
        "pairing": {"fold_count": len(rows), "required_folds": list(FOLDS), "seed": SEED,
                    "calibration_trials": M, "query_start_trial": QUERY_START, "window_size_bins": WINDOW_SIZE,
                    "comparison": "R-C (continuous velocity carrier) minus R-S (B2/D1024 zero4 no-side-carrier)"},
        "r_c_binding": binding, "cost": cost, "rows": rows, "statistics": _statistics(rows),
        "data_scope": {"nwb_opened": False, "cuda_constructed": False, "trainer_constructed": False},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    output.chmod(0o444)
    return {"status": STATUS, "receipt_path": str(output), "receipt_sha256": _sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="read-only partial diagnostic; never writes a receipt")
    parser.add_argument("--rc-aggregate", type=Path, default=DEFAULT_R_C_AGGREGATE)
    parser.add_argument("--rc-seal", type=Path, default=DEFAULT_R_C_SEAL)
    parser.add_argument("--fold0-comparison", type=Path, default=DEFAULT_FOLD0_COMPARISON)
    parser.add_argument("--rs-fold0-root", type=Path, default=DEFAULT_RS_FOLD0_ROOT)
    parser.add_argument("--rs-local-root", type=Path, default=DEFAULT_RS_LOCAL_ROOT)
    parser.add_argument("--rs-supervisor-root", type=Path, default=DEFAULT_RS_SUPERVISOR_ROOT)
    parser.add_argument("--output", type=Path, help="required in formal 15/15 aggregate mode")
    args = parser.parse_args()
    rs_paths = default_rs_paths(fold0_root=args.rs_fold0_root, local_root=args.rs_local_root, supervisor_root=args.rs_supervisor_root)
    if args.preview:
        print(json.dumps(preview(rc_aggregate=args.rc_aggregate, rc_seal=args.rc_seal, rs_paths=rs_paths), indent=2, sort_keys=True))
        return
    if args.output is None:
        parser.error("--output is required unless --preview is selected")
    print(json.dumps(aggregate(rc_aggregate=args.rc_aggregate, rc_seal=args.rc_seal,
                               fold0_comparison=args.fold0_comparison, rs_paths=rs_paths,
                               output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
