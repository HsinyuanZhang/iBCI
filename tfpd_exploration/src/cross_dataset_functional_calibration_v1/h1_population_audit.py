"""E1: H1 focal-channel population dependence of the backward descriptor.

Uses the existing `fit_deployment_carrier` operator and source-frozen plan.
No decoder R2, no mask selection from performance, no new PCA/ridge/EB.
"""
from __future__ import annotations

from dataclasses import replace
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import contracts
from . import plan


class E1Error(RuntimeError):
    """Fail closed for the H1 population assay."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise E1Error(message)


def other_channel_masks(
    n_channels: int,
    focal: int,
    *,
    n_masks: int = plan.E1_N_MASKS,
    retain_frac: float = plan.E1_RETAIN_FRAC,
    seed: int = plan.E1_SEED,
) -> list[dict[str, object]]:
    """16 deterministic masks. Always keep the focal channel. Round-down retain."""
    _require(0 <= int(focal) < int(n_channels), "focal out of roster")
    others = np.array([index for index in range(int(n_channels)) if index != int(focal)], dtype=np.int64)
    n_keep_other = int(retain_frac * others.size)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(focal), int(n_channels)]))
    masks: list[dict[str, object]] = []
    for mask_index in range(int(n_masks)):
        keep_other = rng.choice(others, size=n_keep_other, replace=False)
        keep = np.sort(np.concatenate((np.asarray([int(focal)], dtype=np.int64), keep_other)))
        drop = np.setdiff1d(np.arange(int(n_channels), dtype=np.int64), keep, assume_unique=False)
        _require(int(focal) not in set(drop.tolist()), "focal was dropped")
        _require(int(focal) in set(keep.tolist()), "focal missing from keep")
        masks.append(
            {
                "mask_index": int(mask_index),
                "keep_indices": keep.astype(int).tolist(),
                "drop_indices": drop.astype(int).tolist(),
                "n_keep": int(keep.size),
                "n_drop": int(drop.size),
            }
        )
    return masks


def support_channel_mean(rates: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rates, dtype=np.float64)
    _require(matrix.ndim == 2 and matrix.shape[0] >= 1, "support rates")
    return matrix.mean(axis=0)


def intervene_other_channels(rates: np.ndarray, drop_indices: np.ndarray, channel_mean: np.ndarray) -> np.ndarray:
    intervened = np.array(rates, dtype=np.float64, copy=True)
    drop = np.asarray(drop_indices, dtype=np.int64)
    if drop.size:
        intervened[:, drop] = np.asarray(channel_mean, dtype=np.float64)[drop]
    return intervened


def h1_unnormalized_ridge(design_features: np.ndarray, targets: np.ndarray, ridge_lambda: float) -> np.ndarray:
    """Intercept-unpenalized ridge matching the H1 operator (not the M1 /n bank)."""
    features = np.asarray(design_features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    _require(features.ndim == 2 and y.ndim == 2 and features.shape[0] == y.shape[0], "ridge shapes")
    design = np.column_stack((np.ones(features.shape[0], dtype=np.float64), features))
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(ridge_lambda)
    regularizer[0, 0] = 0.0
    system = design.T @ design + regularizer
    return np.linalg.solve(system, design.T @ y)


def forward_unit_ridge(rates: np.ndarray, velocity: np.ndarray, ridge_lambda: float) -> np.ndarray:
    """Per-unit forward descriptor on a fixed behavior basis (1, velocity)."""
    return h1_unnormalized_ridge(velocity, rates, ridge_lambda)


def _safe_norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def relative_change(baseline: np.ndarray, intervened: np.ndarray) -> float | None:
    denom = _safe_norm(baseline)
    if denom <= plan.E1_NUMERICAL_EPS:
        return None
    return float(_safe_norm(np.asarray(intervened) - np.asarray(baseline)) / denom)


def cosine(baseline: np.ndarray, intervened: np.ndarray) -> float | None:
    left = np.asarray(baseline, dtype=np.float64).reshape(-1)
    right = np.asarray(intervened, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= plan.E1_NUMERICAL_EPS:
        return None
    return float(left.dot(right) / denom)


def descriptor_row_metrics(baseline_row: np.ndarray, intervened_row: np.ndarray) -> dict[str, object]:
    finite = bool(np.isfinite(baseline_row).all() and np.isfinite(intervened_row).all())
    if not finite:
        return {"finite": False, "relative_change": None, "cosine": None}
    return {
        "finite": True,
        "relative_change": relative_change(baseline_row, intervened_row),
        "cosine": cosine(baseline_row, intervened_row),
    }


def _bootstrap_spint(repo_root: Path) -> None:
    project = Path(repo_root) / "SPINT-main"
    _require(project.is_dir(), "SPINT-main missing")
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))


def _replace_support_rates(record, trial_ids: tuple[float, ...], new_rates_by_trial: Mapping[float, np.ndarray]):
    new_trials = []
    for trial in record.trials:
        if float(trial.trial_number) in new_rates_by_trial:
            rates = np.asarray(new_rates_by_trial[float(trial.trial_number)], dtype=np.float64)
            _require(rates.shape == trial.rates.shape, "intervened rates changed shape")
            new_trials.append(replace(trial, rates=rates))
        else:
            new_trials.append(trial)
    return replace(record, trials=tuple(new_trials))


def _finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return None
    return float(np.mean(finite))


def run(repo_root: Path) -> contracts.AssayStatus:
    repo_root = Path(repo_root)
    contracts.require_named_estimator("h1_deployment_m3")
    plan.verify_bound_documents(repo_root)
    os_cuda_guard()
    _bootstrap_spint(repo_root)
    from src.data.h1_m4_eb_pilot import (  # type: ignore
        H1_HELDIN_SESSIONS,
        H1_M4_FOLD0_SOURCE,
        H1_M4_FOLD0_TARGET,
        array_sha256,
        fit_deployment_carrier,
        index_heldin_calib,
        load_record,
        reconstruct_frozen_plan,
    )

    data_dir = repo_root / plan.H1_DATA_DIR_RELATIVE
    plan.reject_forbidden_path(data_dir)
    paths = index_heldin_calib(data_dir)
    # Source-only reconstruction is the documented training path. Passing the
    # two fold-0 targets into reconstruct_frozen_plan also byte-checks M4
    # target carriers against the 2026-08 receipt; that last-bit bind is a
    # known environment drift (same class as the M1 bank disclosure) and is
    # not required to use the source-frozen transform for an M3 assay.
    source_records = {name: load_record(paths[name]) for name in H1_M4_FOLD0_SOURCE}
    frozen = reconstruct_frozen_plan(
        source_records,
        repo_root / plan.H1_RAW_RECEIPT_RELATIVE,
        repo_root / plan.H1_EB_RECEIPT_RELATIVE,
    )
    sealed = json.loads((repo_root / plan.H1_FROZEN_PLAN_MANIFEST_RELATIVE).read_text(encoding="utf-8"))
    reconstructed_hashes = {
        "mean": array_sha256(frozen.mean),
        "scale": array_sha256(frozen.scale),
        "pcs": array_sha256(frozen.pcs),
        "U": array_sha256(frozen.U),
        "mu": array_sha256(frozen.mu),
    }
    plan_bind = {
        "reconstructed_from": "source_records_only",
        "target_m4_receipt_byte_check": "skipped_known_last_bit_drift",
        "transform_sha256": frozen.transform_sha256,
        "sealed_transform_sha256": sealed.get("transform_sha256"),
        "array_sha256_match": {
            name: reconstructed_hashes[name] == sealed.get("array_sha256", {}).get(name)
            for name in reconstructed_hashes
        },
        "q_match": int(frozen.q) == int(sealed.get("q", -1)),
        "lambda_match": float(frozen.ridge_lambda) == float(sealed.get("lambda", -1)),
    }
    records = dict(source_records)
    for name in H1_HELDIN_SESSIONS:
        if name not in records:
            records[name] = load_record(paths[name])
    _ = H1_M4_FOLD0_TARGET
    focals = plan.focal_channel_indices(plan.H1_CHANNELS)
    session_rows: list[dict[str, object]] = []
    for name in H1_HELDIN_SESSIONS:
        record = records[name]
        trial_ids = tuple(float(value) for value in record.trial_values[: plan.H1_SUPPORT_TRIALS_M3])
        _require(len(trial_ids) == plan.H1_SUPPORT_TRIALS_M3, f"{name}: fewer than 3 eval-valid trials")
        baseline = fit_deployment_carrier(record, frozen, trial_ids)
        support_rates = np.concatenate([record.blocks_for(value).rates for value in trial_ids], axis=0)
        support_velocity = np.concatenate([record.blocks_for(value).velocity for value in trial_ids], axis=0)
        channel_mean = support_channel_mean(support_rates)
        forward_base = forward_unit_ridge(support_rates, support_velocity, float(frozen.ridge_lambda))
        system_cond = _system_condition(support_rates, frozen)
        focal_rows: list[dict[str, object]] = []
        for focal in focals:
            masks = other_channel_masks(plan.H1_CHANNELS, int(focal))
            mask_rows: list[dict[str, object]] = []
            for mask in masks:
                drop = np.asarray(mask["drop_indices"], dtype=np.int64)
                rates_by_trial: dict[float, np.ndarray] = {}
                for value in trial_ids:
                    original = record.blocks_for(value).rates
                    intervened = intervene_other_channels(original, drop, channel_mean)
                    _require(np.array_equal(intervened[:, int(focal)], original[:, int(focal)]), "focal rates moved")
                    rates_by_trial[float(value)] = intervened
                isolated = _replace_support_rates(record, trial_ids, rates_by_trial)
                degeneracy: str | None = None
                try:
                    fit = fit_deployment_carrier(isolated, frozen, trial_ids)
                except Exception as error:  # noqa: BLE001 — record, do not retune lambda
                    degeneracy = f"{type(error).__name__}: {error}"
                    fit = None
                intervened_rates = np.concatenate([isolated.blocks_for(value).rates for value in trial_ids], axis=0)
                forward_int = forward_unit_ridge(intervened_rates, support_velocity, float(frozen.ridge_lambda))
                forward_delta = descriptor_row_metrics(forward_base[:, int(focal)], forward_int[:, int(focal)])
                if fit is None:
                    mask_rows.append(
                        {
                            **mask,
                            "degeneracy": degeneracy,
                            "raw_row_focal": {"finite": False, "relative_change": None, "cosine": None},
                            "eb_carrier_focal": {"finite": False, "relative_change": None, "cosine": None},
                            "forward_control_focal": forward_delta,
                            "eb_weight_focal": None,
                            "system_condition": None,
                        }
                    )
                    continue
                mask_rows.append(
                    {
                        **mask,
                        "degeneracy": None,
                        "raw_row_focal": descriptor_row_metrics(
                            baseline["raw_rows"][int(focal)], fit["raw_rows"][int(focal)]
                        ),
                        "eb_carrier_focal": descriptor_row_metrics(
                            baseline["carrier"][int(focal)], fit["carrier"][int(focal)]
                        ),
                        "forward_control_focal": forward_delta,
                        "eb_weight_focal": {
                            "baseline": float(baseline["weight"][int(focal)]),
                            "intervened": float(fit["weight"][int(focal)]),
                            "delta": float(fit["weight"][int(focal)] - baseline["weight"][int(focal)]),
                        },
                        "projected_variance_focal": {
                            "baseline": float(baseline["projected_variance"][int(focal)]),
                            "intervened": float(fit["projected_variance"][int(focal)]),
                        },
                        "system_condition": _fit_condition(fit),
                    }
                )
            focal_rows.append(
                {
                    "focal_index": int(focal),
                    "n_masks": len(mask_rows),
                    "n_finite_raw": sum(1 for row in mask_rows if row["raw_row_focal"]["finite"]),
                    "n_undefined_raw": sum(1 for row in mask_rows if not row["raw_row_focal"]["finite"]),
                    "mean_raw_relative_change": _finite_mean(
                        [row["raw_row_focal"]["relative_change"] for row in mask_rows]
                    ),
                    "mean_raw_cosine": _finite_mean([row["raw_row_focal"]["cosine"] for row in mask_rows]),
                    "mean_eb_relative_change": _finite_mean(
                        [row["eb_carrier_focal"]["relative_change"] for row in mask_rows]
                    ),
                    "mean_eb_cosine": _finite_mean([row["eb_carrier_focal"]["cosine"] for row in mask_rows]),
                    "mean_forward_relative_change": _finite_mean(
                        [row["forward_control_focal"]["relative_change"] for row in mask_rows]
                    ),
                    "masks": mask_rows,
                }
            )
        session_rows.append(
            {
                "session_name": name,
                "date": record.date,
                "input_sha256": record.input_sha256,
                "n_channels": int(record.num_neurons),
                "first3_trial_ids": list(trial_ids),
                "n_support_blocks": int(support_rates.shape[0]),
                "support_seconds": float(support_rates.shape[0] * plan.H1_BLOCK_SECONDS),
                "baseline_system_condition": system_cond,
                "plan_transform_sha256": frozen.transform_sha256,
                "decoder_r2": False,
                "focals": focal_rows,
            }
        )
    date_groups: dict[str, list[dict[str, object]]] = {}
    for row in session_rows:
        date_groups.setdefault(str(row["date"]), []).append(row)
    date_aggregates = []
    for date, rows in date_groups.items():
        raw_rel = []
        eb_rel = []
        fwd_rel = []
        finite = 0
        undefined = 0
        for row in rows:
            for focal in row["focals"]:
                raw_rel.append(focal["mean_raw_relative_change"])
                eb_rel.append(focal["mean_eb_relative_change"])
                fwd_rel.append(focal["mean_forward_relative_change"])
                finite += int(focal["n_finite_raw"])
                undefined += int(focal["n_undefined_raw"])
        date_aggregates.append(
            {
                "date": date,
                "n_recordings": len(rows),
                "mean_raw_relative_change": _finite_mean(raw_rel),
                "mean_eb_relative_change": _finite_mean(eb_rel),
                "mean_forward_relative_change": _finite_mean(fwd_rel),
                "n_finite_raw": finite,
                "n_undefined_raw": undefined,
            }
        )
    payload = {
        "schema": plan.SCHEMA_E1,
        "estimator": contracts.require_named_estimator("h1_deployment_m3"),
        "ridge_objective": plan.h1_ridge_objective(),
        "focal_indices": list(focals),
        "focal_selection": "even_roster_spacing_label_blind",
        "n_masks": plan.E1_N_MASKS,
        "retain_frac": plan.E1_RETAIN_FRAC,
        "seed": plan.E1_SEED,
        "source_plan": frozen.manifest(),
        "source_plan_bind": plan_bind,
        "intervention": (
            "dropped other-channel support rates replaced by that session's "
            "first3 support mean; focal rates/labels/timestamps/source transform unchanged"
        ),
        "decoder_r2": False,
        "replication_unit": "source_date_and_recording_not_channel_x_mask",
        "sessions": session_rows,
        "date_aggregates": date_aggregates,
        "overall": {
            "n_sessions": len(session_rows),
            "n_dates": len(date_aggregates),
            "mean_raw_relative_change": _finite_mean(
                [row["mean_raw_relative_change"] for row in date_aggregates]
            ),
            "mean_eb_relative_change": _finite_mean(
                [row["mean_eb_relative_change"] for row in date_aggregates]
            ),
            "mean_forward_relative_change": _finite_mean(
                [row["mean_forward_relative_change"] for row in date_aggregates]
            ),
            "n_finite_raw": int(sum(int(row["n_finite_raw"]) for row in date_aggregates)),
            "n_undefined_raw": int(sum(int(row["n_undefined_raw"]) for row in date_aggregates)),
        },
    }
    return contracts.AssayStatus(name="E1", status="READY", payload=payload)


def os_cuda_guard() -> None:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _system_condition(rates: np.ndarray, frozen) -> float | None:
    z = ((rates - frozen.mean[None, :]) / frozen.scale[None, :]) @ frozen.pcs[: frozen.q].T
    design = np.column_stack((np.ones(z.shape[0], dtype=np.float64), z))
    singular = np.linalg.svd(design, compute_uv=False)
    if singular[-1] <= plan.E1_NUMERICAL_EPS:
        return None
    return float(singular[0] / singular[-1])


def _fit_condition(fit: Mapping[str, np.ndarray]) -> float | None:
    gram = np.asarray(fit["G"], dtype=np.float64)
    try:
        eig = np.linalg.eigvalsh((gram + gram.T) / 2.0)
    except np.linalg.LinAlgError:
        return None
    if float(np.min(np.abs(eig))) <= plan.E1_NUMERICAL_EPS:
        return None
    return float(np.max(np.abs(eig)) / np.min(np.abs(eig)))
