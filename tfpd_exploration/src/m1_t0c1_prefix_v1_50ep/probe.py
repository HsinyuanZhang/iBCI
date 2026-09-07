"""Overfitting probe for the 50-epoch pair (spec §4 / §4.0 / §4.0.1 / §4.1).

Three surfaces: ``train_fit`` (reporting), ``val_heldout`` (SELECTION),
``test_fold`` (reporting).  Selection and the governing verdict read only
``val_heldout``.  Session datasets are opened once and reused across all
(arm, checkpoint) combinations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tfpd_exploration.src.m1_t0c1_prefix_v1.phase3 import (
    SUPPORT_TRIALS,
    UNITS,
    open_session_dataset,
    score_static,
)

from . import plan
from .phase3 import load_arm_binding, strict_load_epoch_checkpoint


class ProbeError(RuntimeError):
    """Fail closed for the 50-epoch overfitting probe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def _checkpoint_float_map(value: Mapping[object, object], label: str) -> dict[int, float]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    converted: dict[int, float] = {}
    for key, raw in value.items():
        _require(type(key) is int, f"{label} keys must be Python ints, saw {key!r}")
        converted[int(key)] = float(raw)
    _require(set(converted) == set(plan.CHECKPOINT_EPOCH_INDICES),
             f"{label} keys must be exactly {plan.CHECKPOINT_EPOCH_INDICES}, saw {sorted(converted)}")
    return converted


def _per_session_by_epoch(
    value: Mapping[object, object], label: str,
) -> dict[int, dict[str, float]]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    converted: dict[int, dict[str, float]] = {}
    for key, raw in value.items():
        _require(type(key) is int, f"{label} keys must be Python ints, saw {key!r}")
        _require(isinstance(raw, Mapping) and raw, f"{label}[{key!r}] must be a non-empty mapping")
        converted[int(key)] = {str(session_id): float(metric) for session_id, metric in raw.items()}
    _require(set(converted) == set(plan.CHECKPOINT_EPOCH_INDICES),
             f"{label} keys must be exactly {plan.CHECKPOINT_EPOCH_INDICES}, saw {sorted(converted)}")
    return converted


def across_session_sd(per_session: Mapping[str, float]) -> float:
    values = [float(item) for item in per_session.values()]
    _require(values, "across-session SD needs at least one session")
    return float(np.std(values))


def overfitting_verdict(
    mean_by_epoch: Mapping[int, float],
    *,
    surface: str,
    per_session_by_epoch: Mapping[int, Mapping[str, float]] | None = None,
    spread_reference_sd_by_epoch: Mapping[int, float] | None = None,
) -> dict[str, object]:
    """Frozen spec §4 verdict.  ``drop == 0.005`` is INCONCLUSIVE (strict ``>``).

    Evaluated on ``val_heldout`` (selection surface) or on ``test_fold`` as a
    labelled diagnostic.  ``drop_within_session_spread`` compares the drop to
    the val_heldout across-session SD at the argmax epoch (spec §4).
    """
    _require(surface in {plan.SELECTION_SURFACE, "test_fold"},
             "overfitting_verdict surface must be 'val_heldout' or 'test_fold' "
             f"(reporting-only surfaces are forbidden), saw {surface!r}")
    means = _checkpoint_float_map(mean_by_epoch, f"{surface}_by_epoch")
    epoch50 = plan.EPOCH_50_INDEX
    ordered = list(plan.CHECKPOINT_EPOCH_INDICES)
    max_value = max(means[index] for index in ordered)
    argmax_epoch = next(index for index in ordered if means[index] == max_value)
    drop = float(max_value - means[epoch50])
    if argmax_epoch == epoch50:
        verdict = plan.OVERFITTING_VERDICT_ABSENT
    elif drop > plan.OVERFITTING_DROP_THRESHOLD:
        verdict = plan.OVERFITTING_VERDICT_PRESENT
    else:
        verdict = plan.OVERFITTING_VERDICT_INCONCLUSIVE
    per_session_out: dict[str, dict[str, float]] = {}
    sd_by_epoch: dict[int, float] = {}
    if per_session_by_epoch is not None:
        parsed = _per_session_by_epoch(per_session_by_epoch, f"{surface}_per_session_by_epoch")
        for epoch_index in ordered:
            per_session_out[str(epoch_index)] = dict(parsed[epoch_index])
            sd_by_epoch[epoch_index] = across_session_sd(parsed[epoch_index])
    reference_sd: dict[int, float] = {}
    if spread_reference_sd_by_epoch is not None:
        _require(isinstance(spread_reference_sd_by_epoch, Mapping),
                 "spread_reference_sd_by_epoch must be a mapping")
        for key, raw in spread_reference_sd_by_epoch.items():
            _require(type(key) is int, "spread SD keys must be Python ints")
            reference_sd[int(key)] = float(raw)
        _require(set(reference_sd) == set(plan.CHECKPOINT_EPOCH_INDICES),
                 "spread_reference_sd_by_epoch keys must be exactly the checkpoint indices")
    elif sd_by_epoch:
        reference_sd = dict(sd_by_epoch)
    sd_at_argmax: float | None = (
        float(reference_sd[argmax_epoch]) if argmax_epoch in reference_sd else None
    )
    drop_within = bool(sd_at_argmax is not None and drop < sd_at_argmax)
    return {
        "schema": "m1_t0c1_prefix_v1_50ep_overfitting_verdict_v1",
        "surface": surface,
        "verdict": verdict,
        "argmax_epoch_index": argmax_epoch,
        "argmax_epoch_1based": argmax_epoch + 1,
        "interior_argmax": argmax_epoch != epoch50,
        "mean_at_argmax": means[argmax_epoch],
        "mean_at_epoch50": means[epoch50],
        "heldout_at_argmax": means[argmax_epoch],
        "heldout_at_epoch50": means[epoch50],
        "drop_from_argmax_to_epoch50": drop,
        "threshold": plan.OVERFITTING_DROP_THRESHOLD,
        "comparison": "strict > for OVERFITTING_PRESENT",
        "per_session_by_epoch": per_session_out,
        "across_session_sd_by_epoch": {str(index): sd_by_epoch[index] for index in sd_by_epoch},
        "spread_reference_sd_at_argmax": sd_at_argmax,
        "spread_reference_epoch_index": argmax_epoch,
        "spread_reference_epoch_1based": argmax_epoch + 1,
        "drop_within_session_spread": drop_within,
        "descriptive_not_inferential": True,
    }


def select_epoch(
    val_heldout_by_epoch: Mapping[int, float],
    *,
    surface: str,
    sessions: Sequence[str],
) -> int:
    """Frozen spec §4.1: val_heldout argmax, smallest-index tie-break, mode=max.

    ``surface`` is required and must equal ``"val_heldout"``.  ``sessions`` is
    required and must be exactly ``VAL_HELDOUT_SESSIONS``.  Omitting either,
    passing ``None``, or handing train_fit/test_fold is a crash, not a silent
    leak.  Keys must be exactly ``{9, 19, 29, 39, 49}``.
    """
    _require(surface == plan.SELECTION_SURFACE,
             "select_epoch surface must be 'val_heldout'; "
             f"train_fit/test_fold (and any other surface) are forbidden, saw {surface!r}")
    _require(sessions is not None,
             "select_epoch sessions is mandatory; omitting it would let a mislabeled "
             "curve select under surface='val_heldout'")
    try:
        plan.assert_selection_sessions(sessions)
    except plan.M1T0C150EpPlanError as error:
        raise ProbeError(str(error)) from error
    curve = _checkpoint_float_map(val_heldout_by_epoch, "val_heldout_by_epoch")
    max_value = max(curve[index] for index in plan.CHECKPOINT_EPOCH_INDICES)
    selected = next(index for index in plan.CHECKPOINT_EPOCH_INDICES if curve[index] == max_value)
    return int(selected)


def equal_session_mean(per_session: Mapping[str, float], sessions: tuple[str, ...]) -> float:
    _require(tuple(sessions) and all(session_id in per_session for session_id in sessions),
             "probe equal-session mean is missing a required session")
    return float(np.mean([float(per_session[session_id]) for session_id in sessions]))


def equal_session_sd(per_session: Mapping[str, float], sessions: tuple[str, ...]) -> float:
    _require(tuple(sessions) and all(session_id in per_session for session_id in sessions),
             "probe equal-session SD is missing a required session")
    return float(np.std([float(per_session[session_id]) for session_id in sessions]))


def val_heldout_digest_receipt(session_id: str, observed_sha256: str) -> dict[str, str]:
    """Pair the frozen expectation with the digest measured from opened bytes."""
    _require(session_id in plan.VAL_HELDOUT_BODY_SHA256, f"not a val_heldout session: {session_id!r}")
    frozen = plan.VAL_HELDOUT_BODY_SHA256[session_id]
    _require(isinstance(observed_sha256, str) and len(observed_sha256) == 64,
             f"observed val_heldout digest must be a 64-hex sha256, saw {observed_sha256!r}")
    _require(observed_sha256 == frozen,
             f"observed val_heldout digest drifted from frozen for {session_id}: "
             f"{observed_sha256} != {frozen}")
    return {"body_sha256": frozen, "observed_body_sha256": observed_sha256}


def val_heldout_opened_digest_table(
    opened: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    """Frozen vs observed body digests for the three val_heldout sessions."""
    table: dict[str, dict[str, str]] = {}
    for session_id in plan.VAL_HELDOUT_SESSIONS:
        _require(session_id in opened, f"val_heldout session was not opened: {session_id}")
        record = opened[session_id]
        _require(isinstance(record, Mapping)
                 and "body_sha256" in record and "observed_body_sha256" in record,
                 f"opened val_heldout record is missing digest keys: {session_id}")
        table[session_id] = val_heldout_digest_receipt(
            session_id, str(record["observed_body_sha256"]),
        )
        _require(table[session_id]["body_sha256"] == str(record["body_sha256"]),
                 f"frozen body_sha256 key drifted for {session_id}")
    return table


def open_val_heldout_session(root: Path, source_root: Path, session_id: str) -> dict[str, Any]:
    """Open one FALCON held-out-calib session under the §4.0.1 access law.

    Path and body digest come from the frozen evaluation-only law.  Dataset
    construction is the same runtime/recipe path as
    ``m1_t0c1_prefix_v1.phase3.open_session_dataset``; the forward law is not
    forked (scoring still goes through ``score_static``).
    """
    from tfpd_exploration.src.cross_session_worst_group_v1 import source_reader

    _require(session_id in plan.VAL_HELDOUT_SESSIONS, f"not a val_heldout session: {session_id!r}")
    relative = plan.assert_evaluation_only_path(plan.val_heldout_relative_path(session_id))
    target = Path(source_root) / relative
    resolved = str(target)
    for token in plan.VAL_HELDOUT_FORBIDDEN_TOKENS:
        _require(token not in resolved, f"val_heldout opened path contains forbidden token {token!r}: {resolved}")
    _require(plan.FOLD_TARGET_EXCLUDED_FROM_SELECTION not in resolved,
             "20120924 must not appear in an opened val_heldout path")
    body = target.read_bytes()
    observed_digest = plan.sha256_bytes(bytes(body))
    try:
        plan.assert_val_heldout_body_digest(session_id, body)
    except plan.M1T0C150EpPlanError as error:
        raise ProbeError(str(error)) from error
    digest_pair = val_heldout_digest_receipt(session_id, observed_digest)
    runtime = source_reader.load_native_m1_runtime(Path(root))
    recipe = source_reader.NATIVE_M1_READER_RECIPE
    datamodule = runtime.falcon_datamodule_type(**recipe.datamodule_kwargs(data_dir=Path(source_root)))
    record = datamodule.prepare_session_data(
        target, runtime.task,
        standardize_covariates=recipe.standardize_covariates,
        covariates_mean=None, covariates_std=None,
        use_intertrials=recipe.use_intertrials,
        include_trial_targets=False, include_trial_obj_ids=False,
    )
    dataset = runtime.falcon_dataset_type(
        sessions_dict={session_id: record}, calib_sessions_dict={session_id: record},
        **recipe.dataset_kwargs(),
    )
    trials = np.ascontiguousarray(
        dataset.calib_trialized_neural_features[session_id], dtype=np.float32)
    _require(trials.ndim == 3 and trials.shape[1:] == (1024, UNITS) and trials.shape[0] >= SUPPORT_TRIALS,
             f"val_heldout trialized activity topology drift: {session_id}")
    return {
        "dataset": dataset,
        "trials": trials,
        "session_id": session_id,
        "relative_path": relative,
        "access_law": "evaluation_only",
        "training_use": False,
        "labels_used_for": "metric only",
        "body_sha256": digest_pair["body_sha256"],
        "observed_body_sha256": digest_pair["observed_body_sha256"],
    }


def open_probe_session(root: Path, source_root: Path, session_id: str) -> dict[str, Any]:
    """Open one of the seven probe sessions; val_heldout uses the sibling loader."""
    if session_id in plan.VAL_HELDOUT_SESSIONS:
        return open_val_heldout_session(Path(root), Path(source_root), session_id)
    _require(session_id in plan.SCORE_ORDER, f"probe session is not in the 7-session grid: {session_id!r}")
    _require(session_id != plan.FOLD_TARGET_EXCLUDED_FROM_SELECTION or session_id in plan.TEST_FOLD_SESSIONS,
             "fold target used outside the test_fold surface")
    return open_session_dataset(Path(root), Path(source_root), session_id)


def build_curve(cells: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Per arm, per checkpoint: train_fit / val_heldout / test_fold surfaces."""
    arms: dict[str, dict[str, object]] = {}
    for arm in plan.ARMS:
        by_epoch: dict[str, object] = {}
        for epoch_index in plan.CHECKPOINT_EPOCH_INDICES:
            per_session: dict[str, float] = {}
            for session_id in plan.PROBE_SESSIONS:
                key = f"{arm}_epoch_{epoch_index:02d}_{session_id}"
                _require(key in cells, f"probe curve cell absent: {key}")
                per_session[session_id] = float(cells[key]["governing_r2"])
            val_heldout_per_session = {
                session_id: per_session[session_id] for session_id in plan.VAL_HELDOUT_SESSIONS
            }
            train_fit_per_session = {
                session_id: per_session[session_id] for session_id in plan.TRAIN_FIT_SESSIONS
            }
            by_epoch[str(epoch_index)] = {
                "epoch_index_0based": epoch_index,
                "epoch_index_1based": epoch_index + 1,
                "per_session": per_session,
                "train_fit_equal_session_mean": equal_session_mean(
                    per_session, plan.TRAIN_FIT_SESSIONS),
                "train_fit_equal_session_sd": equal_session_sd(
                    per_session, plan.TRAIN_FIT_SESSIONS),
                "train_fit_per_session": train_fit_per_session,
                "val_heldout_equal_session_mean": equal_session_mean(
                    per_session, plan.VAL_HELDOUT_SESSIONS),
                "val_heldout_equal_session_sd": equal_session_sd(
                    per_session, plan.VAL_HELDOUT_SESSIONS),
                "val_heldout_per_session": val_heldout_per_session,
                "test_fold": float(per_session[plan.TEST_FOLD_SESSIONS[0]]),
                "test_fold_session": plan.TEST_FOLD_SESSIONS[0],
            }
        arms[arm] = by_epoch
    return {
        "schema": "m1_t0c1_prefix_v1_50ep_probe_curve_v1",
        "cell": plan.CELL,
        "phase": plan.PHASE,
        "deployment": plan.PROBE_DEPLOYMENT,
        "checkpoint_epoch_indices": list(plan.CHECKPOINT_EPOCH_INDICES),
        "probe_sessions": list(plan.PROBE_SESSIONS),
        "train_fit_sessions": list(plan.TRAIN_FIT_SESSIONS),
        "val_heldout_sessions": list(plan.VAL_HELDOUT_SESSIONS),
        "test_fold_sessions": list(plan.TEST_FOLD_SESSIONS),
        "selection_surface": plan.SELECTION_SURFACE,
        "arms": arms,
    }


def _field_by_epoch(curve: Mapping[str, object], arm: str, field: str) -> dict[int, float]:
    _require(arm in plan.ARMS, "probe arm drift")
    rows = dict(curve["arms"][arm])  # type: ignore[arg-type]
    return {int(index): float(rows[str(index)][field])  # type: ignore[index]
            for index in plan.CHECKPOINT_EPOCH_INDICES}


def train_fit_by_epoch_from_curve(curve: Mapping[str, object], arm: str) -> dict[int, float]:
    return _field_by_epoch(curve, arm, "train_fit_equal_session_mean")


def val_heldout_by_epoch_from_curve(curve: Mapping[str, object], arm: str) -> dict[int, float]:
    return _field_by_epoch(curve, arm, "val_heldout_equal_session_mean")


def test_fold_by_epoch_from_curve(curve: Mapping[str, object], arm: str) -> dict[int, float]:
    return _field_by_epoch(curve, arm, "test_fold")


def val_heldout_per_session_by_epoch_from_curve(
    curve: Mapping[str, object], arm: str,
) -> dict[int, dict[str, float]]:
    _require(arm in plan.ARMS, "probe arm drift")
    rows = dict(curve["arms"][arm])  # type: ignore[arg-type]
    return {
        int(index): {str(session_id): float(value)
                      for session_id, value in dict(rows[str(index)]["val_heldout_per_session"]).items()}  # type: ignore[index]
        for index in plan.CHECKPOINT_EPOCH_INDICES
    }


def val_heldout_sd_by_epoch_from_curve(curve: Mapping[str, object], arm: str) -> dict[int, float]:
    return _field_by_epoch(curve, arm, "val_heldout_equal_session_sd")


def test_fold_per_session_by_epoch_from_curve(
    curve: Mapping[str, object], arm: str,
) -> dict[int, dict[str, float]]:
    fold = plan.TEST_FOLD_SESSIONS[0]
    means = test_fold_by_epoch_from_curve(curve, arm)
    return {index: {fold: value} for index, value in means.items()}


def load_selection(root: Path) -> dict[str, int]:
    """Read this lane's probe selection (fail closed)."""
    directory = Path(root).absolute() / plan.PROBE_ROOT_RELATIVE
    _require(directory.is_dir(), "probe selection root absent")
    body = (directory / "selection.json").read_bytes()
    sidecar = (directory / "selection.json.sha256").read_text(encoding="ascii")
    digest = plan.sha256_bytes(body)
    _require(sidecar == f"{digest}  selection.json\n", "probe selection sidecar drift")
    payload = json.loads(body.decode("utf-8"))
    _require(payload.get("selection_surface") == plan.SELECTION_SURFACE,
             "probe selection_surface drifted from val_heldout")
    selected = payload.get("selected_epoch_index_by_arm")
    _require(isinstance(selected, Mapping) and set(selected) == set(plan.ARMS),
             "probe selection arm topology drift")
    result = {arm: int(selected[arm]) for arm in plan.ARMS}
    _require(all(index in plan.CHECKPOINT_EPOCH_INDICES for index in result.values()),
             "probe selected epoch is not a sealed checkpoint index")
    return result


def run_probe(root: Path, *, source_root: Path, device: str) -> dict[str, object]:
    """Score 2 x 5 x 7 static_m10 cells; each of the 7 sessions is opened once."""
    opened = {
        session_id: open_probe_session(Path(root), Path(source_root), session_id)
        for session_id in plan.PROBE_SESSIONS
    }
    _require(set(opened) == set(plan.PROBE_SESSIONS) and len(opened) == 7,
             "probe opened-session topology drifted from the 7-session grid")
    digest_table = val_heldout_opened_digest_table(opened)
    opened_sessions = {
        "schema": "m1_t0c1_prefix_v1_50ep_opened_sessions_v1",
        "access_law_sha256": plan.VAL_HELDOUT_ACCESS_LAW["law_sha256"],
        "frozen_val_heldout_body_sha256": dict(plan.VAL_HELDOUT_BODY_SHA256),
        "observed_val_heldout_body_sha256": {
            session_id: digest_table[session_id]["observed_body_sha256"]
            for session_id in plan.VAL_HELDOUT_SESSIONS
        },
        "sessions": digest_table,
    }
    bindings = {arm: load_arm_binding(Path(root), arm) for arm in plan.ARMS}
    cells: dict[str, dict[str, object]] = {}
    for arm in plan.ARMS:
        binding = bindings[arm]
        for epoch_index in plan.CHECKPOINT_EPOCH_INDICES:
            filename = plan.epoch_checkpoint_filename(epoch_index)
            expected_state = str(binding.checkpoints[epoch_index]["state_sha256"])
            model, state_sha = strict_load_epoch_checkpoint(
                Path(root), binding.root_relative, filename, expected_state, device,
            )
            for session_id in plan.PROBE_SESSIONS:
                cell = dict(score_static(model, opened[session_id], device=device))
                key = f"{arm}_epoch_{epoch_index:02d}_{session_id}"
                surface = (
                    plan.SELECTION_SURFACE if session_id in plan.VAL_HELDOUT_SESSIONS
                    else ("test_fold" if session_id in plan.TEST_FOLD_SESSIONS else "train_fit")
                )
                cell.update({
                    "schema": "m1_t0c1_prefix_v1_50ep_probe_cell_v1",
                    "arm": arm, "epoch_index": epoch_index,
                    "epoch_index_1based": epoch_index + 1,
                    "session_id": session_id, "deployment": plan.PROBE_DEPLOYMENT,
                    "surface": surface,
                    "arm_checkpoint_state_sha256": state_sha,
                    "arm_terminal_sha256": binding.terminal_sha256,
                })
                cells[key] = cell
    curve = build_curve(cells)
    verdict_by_arm: dict[str, dict[str, object]] = {}
    selected_by_arm: dict[str, int] = {}
    for arm in plan.ARMS:
        val_means = val_heldout_by_epoch_from_curve(curve, arm)
        val_per_session = val_heldout_per_session_by_epoch_from_curve(curve, arm)
        val_sd = val_heldout_sd_by_epoch_from_curve(curve, arm)
        fold_means = test_fold_by_epoch_from_curve(curve, arm)
        verdict_by_arm[arm] = {
            "val_heldout": overfitting_verdict(
                val_means, surface=plan.SELECTION_SURFACE,
                per_session_by_epoch=val_per_session,
            ),
            "test_fold": overfitting_verdict(
                fold_means, surface="test_fold",
                per_session_by_epoch=test_fold_per_session_by_epoch_from_curve(curve, arm),
                spread_reference_sd_by_epoch=val_sd,
            ),
        }
        selected_by_arm[arm] = select_epoch(
            val_means, surface=plan.SELECTION_SURFACE, sessions=plan.VAL_HELDOUT_SESSIONS,
        )
    selection = {
        "schema": "m1_t0c1_prefix_v1_50ep_selection_v1",
        "rule": dict(plan.SELECTION_RULE),
        "mode": "max",
        "selection_surface": plan.SELECTION_SURFACE,
        "val_heldout_sessions": list(plan.VAL_HELDOUT_SESSIONS),
        "selected_epoch_index_by_arm": dict(selected_by_arm),
        "selected_epoch_1based_by_arm": {arm: index + 1 for arm, index in selected_by_arm.items()},
        "train_fit_read_for_selection": False,
        "test_fold_read_for_selection": False,
        "held_out_fold_read_for_selection": False,
    }
    return {
        "cells": cells,
        "curve": curve,
        "verdict_by_arm": verdict_by_arm,
        "selection": selection,
        "opened_sessions": opened_sessions,
        "bindings": {arm: binding.terminal_sha256 for arm, binding in bindings.items()},
        "val_heldout_access_law_sha256": plan.VAL_HELDOUT_ACCESS_LAW["law_sha256"],
        "frozen_val_heldout_body_sha256": dict(plan.VAL_HELDOUT_BODY_SHA256),
        "observed_val_heldout_body_sha256": opened_sessions["observed_val_heldout_body_sha256"],
    }


__all__ = (
    "ProbeError", "overfitting_verdict", "select_epoch", "build_curve",
    "equal_session_mean", "run_probe", "load_selection",
    "open_val_heldout_session", "open_probe_session",
    "val_heldout_digest_receipt", "val_heldout_opened_digest_table",
    "train_fit_by_epoch_from_curve", "val_heldout_by_epoch_from_curve",
    "test_fold_by_epoch_from_curve",
)
