#!/usr/bin/env python3
"""Fail-closed CPU-only audit of the proposed M1 O4/D4 descriptor.

This is deliberately an estimand/semantics audit, not a decoder experiment.  It
uses only local M1 calibration NWB files, never imports torch, never starts a
trainer, and refuses to overwrite an output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from falcon_challenge.config import FalconTask
from falcon_challenge.dataloaders import load_nwb
from pynwb import NWBHDF5IO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "SPINT-main" / "data" / "000941"
DEFAULT_OUT = ROOT / "sua_exploration" / "results" / "m1_d4_semantics_v2"
SPLITS = ("held-in-calib", "held-out-calib")
PREFIXES = (10, 40, 90, 200)
REQUIRED = ("obj_id", "tgt_obj", "tgt_loc", "condition_id")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(value: Any) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, np.ndarray):
        return [strict_json(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def session_of(path: Path) -> str:
    return path.name.split("_ses-")[1].split("_behavior")[0]


def files_for(split: str) -> list[Path]:
    return sorted((DATA / f"sub-MonkeyL-{split}").glob("*.nwb"))


def key(value: Any) -> str:
    """Stable level representation, also safe for float NWB metadata."""
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".12g")
    return str(value)


def load_trials(path: Path):
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        trials = io.read().trials.to_dataframe()
    missing = [column for column in REQUIRED if column not in trials.columns]
    if missing:
        raise ValueError(f"{path} lacks required trial columns: {missing}")
    return trials


def joint_table(trials) -> list[dict[str, Any]]:
    columns = list(REQUIRED)
    counts = trials.groupby(columns, dropna=False).size().reset_index(name="n_trials")
    rows = []
    for _, row in counts.iterrows():
        rows.append({column: key(row[column]) for column in columns} | {"n_trials": int(row["n_trials"])})
    return sorted(rows, key=lambda row: tuple(row[column] for column in columns))


def level_counts(trials, column: str) -> list[dict[str, Any]]:
    values = trials[column].value_counts(dropna=False)
    return [
        {"level": key(level), "n_trials": int(count)}
        for level, count in sorted(values.items(), key=lambda item: key(item[0]))
    ]


def deterministic_mapping(trials, source: str, target: str) -> bool:
    if len(trials) == 0:
        return False
    return bool(all(group[target].nunique(dropna=False) == 1 for _, group in trials.groupby(source, dropna=False)))


def design_diagnostic(x: np.ndarray) -> dict[str, Any]:
    """Report an explicit non-finite-safe design diagnosis for a prefix."""
    rank = int(np.linalg.matrix_rank(x))
    columns = int(x.shape[1])
    if rank < columns:
        return {"status": "rank_deficient", "rank": rank, "n_columns": columns,
                "condition_number": None, "reason": "rank_deficient"}
    value = float(np.linalg.cond(x))
    return {"status": "ok", "rank": rank, "n_columns": columns,
            "condition_number": value if math.isfinite(value) else None,
            "reason": None if math.isfinite(value) else "nonfinite_condition_number"}


def prefix_coverage(path: Path, split: str) -> dict[str, Any]:
    trials = load_trials(path)
    entry: dict[str, Any] = {
        "split": split,
        "session": session_of(path),
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "n_trials": int(len(trials)),
        "prefixes": {},
    }
    for support in PREFIXES:
        if support > len(trials):
            entry["prefixes"][str(support)] = {
                "status": "unavailable", "reason": "prefix_exceeds_file_length"
            }
            continue
        prefix = trials.iloc[:support]
        cosine = cosine_design(prefix)
        categorical_direction, _ = categorical(prefix["tgt_loc"])
        categorical_obj, _ = categorical(prefix["obj_id"])
        entry["prefixes"][str(support)] = {
            "status": "available",
            "n_trials": int(len(prefix)),
            "level_counts": {column: level_counts(prefix, column) for column in REQUIRED},
            "joint_table": joint_table(prefix),
            "obj_id_to_tgt_loc_deterministic": deterministic_mapping(prefix, "obj_id", "tgt_loc"),
            "tgt_loc_to_obj_id_deterministic": deterministic_mapping(prefix, "tgt_loc", "obj_id"),
            "tgt_obj_constant": bool(prefix["tgt_obj"].nunique(dropna=False) == 1),
            "design_diagnostics": {
                "cosine_direction": design_diagnostic(cosine),
                "categorical_direction": design_diagnostic(categorical_direction),
                "categorical_obj_id_D4": design_diagnostic(categorical_obj),
            },
        }
    return entry


def movement_window_emg_rates(path: Path) -> tuple[Any, np.ndarray]:
    """Same movement-window EMG trial means used by the prior M1 mechanism audit."""
    trials = load_trials(path)
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        raw = io.read().acquisition["preprocessed_emg"]
        first = raw.get_timeseries(list(raw.time_series.keys())[0])
        timestamps = np.asarray(first.timestamps[:], dtype=np.float64)
    _, emg, trial_change, eval_mask = load_nwb(path, FalconTask.m1)
    starts = np.flatnonzero(trial_change)
    if len(starts) != len(trials):
        raise ValueError(f"trial alignment mismatch in {path}")
    rows = []
    for index, start in enumerate(starts):
        onset = float(trials.iloc[index]["move_onset_time"])
        stop = float(trials.iloc[index]["stop_time"])
        if not math.isfinite(onset):
            onset = float(trials.iloc[index]["start_time"])
        left = int(np.searchsorted(timestamps, onset, side="left"))
        right = int(np.searchsorted(timestamps, stop, side="right"))
        segment = emg[left:right]
        mask = eval_mask[left:right]
        if mask.any():
            segment = segment[mask]
        if segment.size == 0:
            segment = emg[start : start + max(right - left, 1)]
        rows.append(segment.mean(axis=0))
    return trials, np.vstack(rows).astype(np.float64)


def movement_window_neural_rates(path: Path) -> np.ndarray:
    """Raw-spike trial/unit rate in [move_onset, stop), without GPU or binning.

    Native-MUA rows can concatenate recording segments with time resets, so their
    ``spike_times`` are not globally sorted.  Use explicit boolean interval counts,
    matching audit_m1_t4_mechanism.py; searchsorted is invalid for these rows.
    """
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        trials = nwb.trials.to_dataframe()
        units = nwb.units.to_dataframe()
    rates = np.zeros((len(trials), len(units)), dtype=np.float64)
    spike_times = [np.asarray(unit.spike_times, dtype=np.float64) for _, unit in units.iterrows()]
    for trial_index in range(len(trials)):
        onset = float(trials.iloc[trial_index]["move_onset_time"])
        stop = float(trials.iloc[trial_index]["stop_time"])
        duration = max(stop - onset, 1.0e-12)
        for unit_index, times in enumerate(spike_times):
            rates[trial_index, unit_index] = int(np.sum((times >= onset) & (times < stop)))
        rates[trial_index] /= duration
    return rates


def angles(trials) -> np.ndarray:
    return np.deg2rad(np.asarray(trials["tgt_loc"], dtype=np.float64))


def categorical(values, levels: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    values_key = [key(value) for value in values]
    used = sorted(set(values_key)) if levels is None else list(levels)
    if not used:
        return np.ones((len(values_key), 1), dtype=np.float64), used
    # Intercept plus K-1 contrasts.  An unseen query level maps to reference.
    design = np.ones((len(values_key), len(used)), dtype=np.float64)
    for column, level in enumerate(used[1:], start=1):
        design[:, column] = np.asarray([item == level for item in values_key], dtype=np.float64)
    return design, used


def cosine_design(trials) -> np.ndarray:
    theta = angles(trials)
    return np.column_stack([np.ones(len(trials)), np.cos(theta), np.sin(theta)])


def joint_category(trials, fields: tuple[str, ...]) -> list[str]:
    return ["|".join(key(row[field]) for field in fields) for _, row in trials.iterrows()]


def matrix_for(model: str, trials, fit_levels: dict[str, list[str]] | None = None) -> tuple[np.ndarray, dict[str, list[str]]]:
    fit_levels = {} if fit_levels is None else fit_levels
    if model == "cosine_direction":
        return cosine_design(trials), {}
    if model == "categorical_direction":
        values = [key(value) for value in trials["tgt_loc"]]
        x, lv = categorical(values, fit_levels.get("direction"))
        return x, {"direction": lv}
    if model in ("tgt_obj", "obj_id", "condition_id_descriptive"):
        field = "condition_id" if model == "condition_id_descriptive" else model
        x, lv = categorical(trials[field], fit_levels.get(field))
        return x, {field: lv}
    if model in ("categorical_direction_plus_tgt_obj", "categorical_direction_plus_obj_id"):
        field = "tgt_obj" if model.endswith("tgt_obj") else "obj_id"
        direction, direction_levels = categorical(trials["tgt_loc"], fit_levels.get("direction"))
        factor, factor_levels = categorical(trials[field], fit_levels.get(field))
        # This is additive: one shared intercept, direction contrasts, factor
        # contrasts.  It is therefore properly nested over categorical direction.
        return np.column_stack([direction, factor[:, 1:]]), {"direction": direction_levels, field: factor_levels}
    if model == "direction_tgt_obj_obj_id_interaction_descriptive":
        labels = joint_category(trials, ("tgt_loc", "tgt_obj", "obj_id"))
        x, lv = categorical(labels, fit_levels.get("interaction"))
        return x, {"interaction": lv}
    raise ValueError(f"unknown model {model}")


def unseen_query_levels(model: str, trials, fit_levels: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    """Expose query categories encoded as the reference rather than hiding them."""
    fields: tuple[str, ...]
    if model == "categorical_direction":
        fields = ("direction",)
        observed = {"direction": set(map(key, trials["tgt_loc"]))}
    elif model in ("tgt_obj", "obj_id", "condition_id_descriptive"):
        field = "condition_id" if model == "condition_id_descriptive" else model
        fields, observed = (field,), {field: set(map(key, trials[field]))}
    elif model in ("categorical_direction_plus_tgt_obj", "categorical_direction_plus_obj_id"):
        field = "tgt_obj" if model.endswith("tgt_obj") else "obj_id"
        fields, observed = ("direction", field), {"direction": set(map(key, trials["tgt_loc"])), field: set(map(key, trials[field]))}
    elif model == "direction_tgt_obj_obj_id_interaction_descriptive":
        fields, observed = ("interaction",), {"interaction": set(joint_category(trials, ("tgt_loc", "tgt_obj", "obj_id")))}
    else:
        return {}
    return {field: {"n_unseen": len(observed[field] - set(fit_levels.get(field, []))),
                    "levels": sorted(observed[field] - set(fit_levels.get(field, [])))} for field in fields}


MODELS = (
    "cosine_direction", "categorical_direction", "tgt_obj", "obj_id",
    "categorical_direction_plus_tgt_obj", "categorical_direction_plus_obj_id",
    "direction_tgt_obj_obj_id_interaction_descriptive", "condition_id_descriptive",
)


def ols_metrics(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> dict[str, Any]:
    rank = int(np.linalg.matrix_rank(train_x))
    if rank < train_x.shape[1] or train_x.shape[0] <= rank:
        return {"status": "undefined", "reason": "rank_deficient_or_no_residual_df", "rank": rank,
                "n_columns": int(train_x.shape[1]), "r2": None, "rss": None, "tss": None}
    coef, _, _, _ = np.linalg.lstsq(train_x, train_y, rcond=None)
    predicted = test_x @ coef
    rss = float(np.square(test_y - predicted).sum())
    baseline = train_y.mean(axis=0, keepdims=True)
    tss = float(np.square(test_y - baseline).sum())
    if tss <= 0.0:
        return {"status": "undefined", "reason": "zero_test_tss", "rank": rank, "n_columns": int(train_x.shape[1]), "r2": None, "rss": rss, "tss": tss}
    return {"status": "ok", "rank": rank, "n_columns": int(train_x.shape[1]), "r2": 1.0 - rss / tss, "rss": rss, "tss": tss}


def insample_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    return ols_metrics(x, y, x, y)


def expanding_block_cv(trials, y: np.ndarray, model: str, n_blocks: int = 5) -> dict[str, Any]:
    boundaries = np.linspace(0, len(trials), n_blocks + 1, dtype=int)
    folds = []
    total_rss = total_tss = 0.0
    for fold in range(1, n_blocks):
        train_end, test_end = int(boundaries[fold]), int(boundaries[fold + 1])
        train_trials, test_trials = trials.iloc[:train_end], trials.iloc[train_end:test_end]
        train_x, levels = matrix_for(model, train_trials)
        test_x, _ = matrix_for(model, test_trials, levels)
        metric = ols_metrics(train_x, y[:train_end], test_x, y[train_end:test_end])
        metric |= {"fold": fold, "train_trial_range": [0, train_end], "test_trial_range": [train_end, test_end],
                   "unseen_query_levels": unseen_query_levels(model, test_trials, levels)}
        folds.append(metric)
        if metric["status"] == "ok":
            total_rss += float(metric["rss"])
            total_tss += float(metric["tss"])
    if total_tss <= 0.0:
        return {"status": "undefined", "reason": "no_defined_folds", "folds": folds, "pooled_r2": None}
    return {"status": "ok", "folds": folds, "pooled_r2": 1.0 - total_rss / total_tss,
            "defined_folds": int(sum(fold["status"] == "ok" for fold in folds))}


def model_audit_for_target(paths: list[Path], loader: Callable[[Path], tuple[Any, np.ndarray]], target_name: str) -> dict[str, Any]:
    sessions = []
    aggregate: dict[str, list[float]] = {model: [] for model in MODELS}
    for path in paths:
        trials, y = loader(path)
        per_model = {}
        for model in MODELS:
            x, _ = matrix_for(model, trials)
            fit = insample_metrics(x, y)
            cv = expanding_block_cv(trials, y, model)
            per_model[model] = {"in_sample": fit, "chronological_expanding_block_cv": cv}
            if cv["status"] == "ok":
                aggregate[model].append(float(cv["pooled_r2"]))
        sessions.append({"session": session_of(path), "n_trials": int(len(trials)), "n_outputs": int(y.shape[1]), "models": per_model})
    incremental = {}
    for name, full, base in (
        ("categorical_direction_minus_cosine_direction", "categorical_direction", "cosine_direction"),
        ("tgt_obj_given_categorical_direction", "categorical_direction_plus_tgt_obj", "categorical_direction"),
        ("obj_id_given_categorical_direction", "categorical_direction_plus_obj_id", "categorical_direction"),
    ):
        deltas = []
        for session in sessions:
            a = session["models"][full]["chronological_expanding_block_cv"]
            b = session["models"][base]["chronological_expanding_block_cv"]
            if a["status"] == "ok" and b["status"] == "ok":
                deltas.append(float(a["pooled_r2"] - b["pooled_r2"]))
        incremental[name] = {"per_session": deltas, "mean": float(np.mean(deltas)) if deltas else None, "n_defined": len(deltas)}
    return {"target": target_name, "sessions": sessions,
            "mean_session_cv_r2": {model: (float(np.mean(values)) if values else None) for model, values in aggregate.items()},
            "nested_incremental_cv_r2": incremental}


def r2_for_indices(pred: np.ndarray, actual: np.ndarray, train_y: np.ndarray) -> dict[str, Any]:
    if len(actual) == 0:
        return {"status": "unavailable", "reason": "no_query_trials", "r2": None, "n_trials": 0}
    rss = float(np.square(actual - pred).sum())
    tss = float(np.square(actual - train_y.mean(axis=0, keepdims=True)).sum())
    return {"status": "ok" if tss > 0.0 else "undefined", "reason": None if tss > 0.0 else "zero_tss",
            "r2": None if tss <= 0.0 else 1.0 - rss / tss, "n_trials": int(len(actual)), "rss": rss, "tss": tss}


def support_prediction(paths: list[Path]) -> dict[str, Any]:
    sessions = []
    summary: dict[str, dict[str, list[float]]] = {str(m): {"deltas": [], "t4": [], "d4": []} for m in (10, 40)}
    for path in paths:
        trials = load_trials(path)
        neural = movement_window_neural_rates(path)
        session = {"session": session_of(path), "n_trials": int(len(trials)), "supports": {}}
        for support in (10, 40):
            if support >= len(trials):
                session["supports"][str(support)] = {"status": "unavailable", "reason": "support_exhausts_session"}
                continue
            train_trials, query_trials = trials.iloc[:support], trials.iloc[support:]
            train_y, query_y = neural[:support], neural[support:]
            t4_x, _ = matrix_for("cosine_direction", train_trials)
            d4_x, d4_levels = matrix_for("obj_id", train_trials)
            t4q_x, _ = matrix_for("cosine_direction", query_trials)
            d4q_x, _ = matrix_for("obj_id", query_trials, d4_levels)
            t4_fit = ols_metrics(t4_x, train_y, t4q_x, query_y)
            d4_fit = ols_metrics(d4_x, train_y, d4q_x, query_y)
            support_joint = set(joint_category(train_trials, ("obj_id", "tgt_loc", "tgt_obj")))
            query_joint = np.asarray(joint_category(query_trials, ("obj_id", "tgt_loc", "tgt_obj")))
            matched = np.asarray([value in support_joint for value in query_joint], dtype=bool)
            arm = {"status": "ok", "support_trials": support,
                   "d4_definition": "raw categorical obj_id profile (intercept plus K-1 categories); semantic object claim separately gated",
                   "query_combination_match_definition": "exact (obj_id, tgt_loc, tgt_obj) tuple observed in support",
                   "query_combination_matched_n": int(matched.sum()), "query_all_future_n": int(len(query_trials)),
                   "label_shift": {field: {"support_levels": sorted(set(map(key, train_trials[field]))), "query_levels": sorted(set(map(key, query_trials[field])))} for field in REQUIRED},
                   "t4_all_future": t4_fit, "d4_all_future": d4_fit}
            # OLS metric returns prediction only internally; refit for the matched metric.
            for name, x, qx in (("t4", t4_x, t4q_x), ("d4", d4_x, d4q_x)):
                if np.linalg.matrix_rank(x) < x.shape[1]:
                    arm[f"{name}_matched"] = {"status": "undefined", "reason": "rank_deficient_support_fit", "r2": None, "n_trials": int(matched.sum())}
                else:
                    coef, _, _, _ = np.linalg.lstsq(x, train_y, rcond=None)
                    arm[f"{name}_matched"] = r2_for_indices((qx @ coef)[matched], query_y[matched], train_y)
            if t4_fit["status"] == "ok" and d4_fit["status"] == "ok":
                arm["d4_minus_t4_all_future"] = float(d4_fit["r2"] - t4_fit["r2"])
                summary[str(support)]["deltas"].append(arm["d4_minus_t4_all_future"])
                summary[str(support)]["t4"].append(float(t4_fit["r2"]))
                summary[str(support)]["d4"].append(float(d4_fit["r2"]))
            else:
                arm["d4_minus_t4_all_future"] = None
            session["supports"][str(support)] = arm
        sessions.append(session)
    final = {}
    for support, values in summary.items():
        ds = values["deltas"]
        final[support] = {"n_defined": len(ds), "t4_mean_r2": float(np.mean(values["t4"])) if ds else None,
                          "d4_mean_r2": float(np.mean(values["d4"])) if ds else None,
                          "d4_minus_t4_mean": float(np.mean(ds)) if ds else None,
                          "d4_minus_t4_positive_sessions": int(sum(value > 0.0 for value in ds)),
                          "d4_minus_t4_per_session": ds}
    return {"rate_estimand": "raw-spike movement-window trial rates; this is not decoder R2",
            "future_label_oracle": "Each later trial's observed tgt_loc/obj_id is used to form the T4/D4 design row. This is an offline descriptor-identifiability diagnostic, not an online decoder predictor: the deployed decoder does not receive future query labels.",
            "sessions": sessions, "summary": final}


def build_audit() -> dict[str, Any]:
    heldin, heldout = files_for("held-in-calib"), files_for("held-out-calib")
    if len(heldin) != 4 or len(heldout) != 3:
        raise ValueError(f"expected 4 held-in and 3 held-out files; got {len(heldin)}, {len(heldout)}")
    coverage = [prefix_coverage(path, split) for split, paths in (("held-in-calib", heldin), ("held-out-calib", heldout)) for path in paths]
    heldout_m10 = [entry["prefixes"]["10"] for entry in coverage if entry["split"] == "held-out-calib"]
    object_identifiable = bool(all(
        entry["status"] == "available"
        and len(entry["level_counts"]["tgt_obj"]) >= 2
        and not entry["obj_id_to_tgt_loc_deterministic"]
        for entry in heldout_m10
    ))
    emg = model_audit_for_target(heldin, movement_window_emg_rates, "movement-window EMG means")
    neural_loader = lambda path: (load_trials(path), movement_window_neural_rates(path))
    neural = model_audit_for_target(heldin, neural_loader, "movement-window raw-spike neural rates")
    prediction = support_prediction(heldin)
    main = prediction["summary"]["10"]
    all_m10_defined = main["n_defined"] == 4
    d4_eligible = bool(all_m10_defined and main["d4_minus_t4_mean"] is not None and main["d4_minus_t4_mean"] > 0.0 and main["d4_minus_t4_positive_sessions"] >= 3)
    semantic_status = "rejected_as_object_descriptor" if not object_identifiable else "not_rejected_by_deployment_semantics_gate"
    eligibility = ("eligible_for_minimal_gpu_pilot" if d4_eligible else
                   "indeterminate_insufficient_defined_m10_sessions" if not all_m10_defined else
                   "stop_cpu_gate_not_met")
    return {
        "schema_version": "m1_d4_semantics_v2",
        "purpose": "M1 O4/object-semantic and D4/categorical-profile pre-GPU audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"cpu_only": True, "no_training": True, "no_evalai": True, "prefixes": list(PREFIXES),
                     "chronological_order_preserved": True, "generation_command": "python3 sua_exploration/scripts/audit_m1_d4_semantics.py",
                     "environment": {"numpy": np.__version__},
                     "input_split_counts": {"held_in": len(heldin), "held_out": len(heldout)}},
        "input_files": {entry["path"]: entry["sha256"] for entry in coverage},
        "prefix_coverage": coverage,
        "held_out_m10_exact_joint_tables": [{"session": entry["session"], "joint_table": entry["prefixes"]["10"]["joint_table"]} for entry in coverage if entry["split"] == "held-out-calib"],
        "semantic_gate": {"definition": "all held-out M10 prefixes have >=2 tgt_obj levels AND obj_id is not deterministically mapped to tgt_loc", "object_factor_identifiable_at_deployment_m10": object_identifiable, "decision": semantic_status},
        "trial_mean_model_audit": {"emg": emg, "neural": neural,
            "cv_definition": "five chronological blocks; each scored fold trains only on preceding blocks. Undefined rank-deficient folds are retained, not repaired."},
        "deployment_support_neural_prediction": prediction,
        "predeclared_decision": {
            "Gate_S": {"status": "pass" if object_identifiable else "fail", "consequence": semantic_status},
            "Gate_P": {"rule": "all four held-in M10 all-later sessions defined; mean(D4-T4)>0; >=3/4 deltas strictly positive", "status": "pass" if d4_eligible else "indeterminate" if not all_m10_defined else "fail", "observed": main},
            "Gate_G": {"status": "root_review_required" if d4_eligible else "not_authorized", "reason": "CPU protocol may not launch GPU work"},
            "D4_categorical_profile_gpu_eligibility": eligibility},
        "limitations": [
            "O4/D4 support-to-later neural-rate prediction is an offline per-channel rate diagnostic, not decoder R2 and not an EvalAI metric.",
            "The support-to-future diagnostic uses observed future trial labels to select a T4/D4 design row (a later-label oracle); it is not an online-deployment predictor because a deployed decoder receives no future query labels.",
            "Gate P intentionally has no practical-effect threshold. A technical pass from tiny strictly-positive rate-prediction deltas is not a deployment-effect claim and does not authorize GPU work; Gate G remains root-review-required.",
            "At held-out M10, object identity is rejected as an identifiable framing when target object is constant or obj_id is collinear with target location.",
            "Categorical obj_id may still be called a D4/profile descriptor only; it must not be called an object factor without a passed semantic gate.",
            "CV is chronological but early-prefix category coverage can make flexible categorical models rank-deficient; those folds are reported undefined rather than imputed.",
            "M=90 and M=200 change condition coverage as well as sample size; they are not a pure label-count ablation."
        ],
    }


def write_readme(out: Path, audit: dict[str, Any]) -> None:
    gate_p = audit["predeclared_decision"]["Gate_P"]
    observed = gate_p["observed"]
    text = f"""# M1 D4 semantic audit\n\nCPU-only, fail-closed audit. `audit.json` contains input SHA256 values, prefix coverage, semantic identifiability, chronological trial-mean model comparisons, and a neural-rate diagnostic. It does **not** train a decoder or report decoder/EvalAI R².\n\nA failed object semantic gate rejects the O4 *object* framing; it does not itself prove a categorical per-channel profile ineffective. D4 GPU eligibility is separately predeclared in `audit.json`.\n\n## Gate interpretation\n\nGate P's frozen rule has **no practical-effect threshold**: it only requires a strictly positive mean and three positive sessions. In this run its status is `{gate_p['status']}` with `D4−T4={observed['d4_minus_t4_mean']:.9g}` and neural-rate R² values `D4={observed['d4_mean_r2']:.9g}`, `T4={observed['t4_mean_r2']:.9g}`. A technical Gate-P pass at this scale is not evidence of a meaningful deployment effect and does **not** authorize a GPU experiment. Gate G remains root-review-required.\n"""
    (out / "README.md").write_text(text, encoding="utf-8")


def run(out: Path = DEFAULT_OUT) -> Path:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {out}")
    out.mkdir(parents=True, exist_ok=False)
    try:
        audit = strict_json(build_audit())
        written = out / "audit.json"
        with written.open("w", encoding="utf-8") as handle:
            json.dump(audit, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        (out / "audit.sha256").write_text(f"{sha256(written)}  audit.json\n", encoding="utf-8")
        write_readme(out, audit)
        return written
    except Exception:
        # No partial result may masquerade as a completed audit.
        for child in out.iterdir():
            child.unlink()
        out.rmdir()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(run(args.out))


if __name__ == "__main__":
    main()
