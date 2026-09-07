"""Exact ridge-100 correction for the H1 PCF8 source precursor.

This v2 module preserves v1's raw-OLS audit as a historical diagnostic and
implements the actual existing L-A forward-estimator regularizer: raw
``[1, v_1, ..., v_7]`` ridge with lambda=100 on W only and an unpenalized
intercept.  It is source-only and contains no decoder path.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

try:  # package import for tests
    from .pcf8_precursor import (
        CHANNELS, DEAD_CHANNEL, DESCRIPTOR_DIM, MACHINE_SCALE_FLOOR, NATIVE_PHASES,
        VELOCITY_DIM, BlockTable, Pcf8PrecursorError, _cosine, _need, _per_column_comparison,
        _quantiles, _phase_for_blocks, _design_summary, pcf8_arm_schema, pcf8_compute_contract,
        per_column_source_scale, table_for_trials,
    )
except ImportError:  # direct script import
    from pcf8_precursor import (  # type: ignore
        CHANNELS, DEAD_CHANNEL, DESCRIPTOR_DIM, MACHINE_SCALE_FLOOR, NATIVE_PHASES,
        VELOCITY_DIM, BlockTable, Pcf8PrecursorError, _cosine, _need, _per_column_comparison,
        _quantiles, _phase_for_blocks, _design_summary, pcf8_arm_schema, pcf8_compute_contract,
        per_column_source_scale, table_for_trials,
    )


RIDGE_LAMBDA = 100.0
V2_SCHEMA = "h1_pcf8_ridge100_source_constructibility_v2"


class Pcf8Ridge100Error(Pcf8PrecursorError):
    """Fail-closed ridge-100 PCF8 construction error."""


def _fixed_point_free_offset(session_name: str, budget: int, blocks: int) -> int:
    _need(blocks >= 2, "LS null needs at least two support rows")
    digest = hashlib.sha256(f"PCF8-LS-v2|{session_name}|M={budget}|B={blocks}".encode("utf-8")).digest()
    return 1 + (int.from_bytes(digest[:8], "big") % (blocks - 1))


def fixed_point_free_velocity_null(velocity: np.ndarray, *, session_name: str, budget: int) -> tuple[np.ndarray, int]:
    velocity = np.asarray(velocity, dtype=np.float64)
    _need(velocity.ndim == 2 and velocity.shape[1] == VELOCITY_DIM, "LS velocity must be B-by-7")
    offset = _fixed_point_free_offset(session_name, budget, velocity.shape[0])
    permutation = (np.arange(velocity.shape[0], dtype=np.int64) + offset) % velocity.shape[0]
    _need(not np.any(permutation == np.arange(velocity.shape[0])), "LS permutation unexpectedly has a fixed point")
    return velocity[permutation], offset


def _ridge_system(velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    velocity = np.asarray(velocity, dtype=np.float64)
    design = np.column_stack((np.ones(velocity.shape[0]), velocity))
    _need(design.shape[1] == DESCRIPTOR_DIM and design.shape[0] >= DESCRIPTOR_DIM, "ridge design must be B-by-8")
    design_info = _design_summary(velocity)
    _need(design_info["rank"] == DESCRIPTOR_DIM, "ridge100 PCF8 design is rank deficient")
    penalty = np.eye(DESCRIPTOR_DIM, dtype=np.float64) * RIDGE_LAMBDA
    penalty[0, 0] = 0.0
    system = design.T @ design + penalty
    eigenvalues = np.linalg.eigvalsh(system)
    _need(eigenvalues[0] > 0.0 and np.isfinite(eigenvalues).all(), "ridge100 normal system is not positive definite")
    response = np.linalg.solve(system, design.T @ design)
    hat_trace = float(np.trace(np.linalg.solve(system, design.T @ design)))
    design_info["ridge100"] = {
        "lambda": RIDGE_LAMBDA,
        "intercept_penalty": 0.0,
        "W_penalty": RIDGE_LAMBDA,
        "regularized_system_eigenvalues": [float(item) for item in eigenvalues],
        "regularized_system_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "effective_degrees_of_freedom_trace_hat": hat_trace,
        "coefficient_response_diagonal": [float(item) for item in np.diag(response)],
        "meaning": "response diagonal and trace are exact ridge shrinkage diagnostics in the native raw coordinate system; no lambda selection/grid is performed.",
    }
    return design, system, penalty, design_info


def fit_forward_pcf8_ridge100(rates: np.ndarray, velocity: np.ndarray) -> dict[str, Any]:
    rates = np.asarray(rates, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    _need(rates.ndim == 2 and rates.shape[1] == CHANNELS, "rates must be B-by-176")
    _need(velocity.shape == (rates.shape[0], VELOCITY_DIM), "velocity must align with rates")
    design, system, _penalty, design_info = _ridge_system(velocity)
    coefficient = np.linalg.solve(system, design.T @ rates)
    degenerate = tuple(int(index) for index in np.flatnonzero(np.max(np.abs(rates - rates[0:1]), axis=0) <= 1e-14))
    coefficient[:, list(degenerate)] = 0.0
    descriptor = np.concatenate((coefficient[1:].T, coefficient[:1].T), axis=1)
    _need(descriptor.shape == (CHANNELS, DESCRIPTOR_DIM) and np.isfinite(descriptor).all(), "ridge100 descriptor invalid")
    return {"descriptor": descriptor, "coefficient": coefficient, "degenerate_channels": degenerate, "design": design_info}


def _comparison(a: Mapping[str, Any], b: Mapping[str, Any], scale: np.ndarray | None = None) -> dict[str, Any]:
    da, db = np.asarray(a["descriptor"], dtype=np.float64), np.asarray(b["descriptor"], dtype=np.float64)
    active = np.asarray([index for index in range(CHANNELS) if index not in set(a["degenerate_channels"]) | set(b["degenerate_channels"])], dtype=np.int64)
    if scale is None:
        sa, sb = da, db
    else:
        scale = np.asarray(scale, dtype=np.float64)
        _need(scale.shape == (DESCRIPTOR_DIM,) and np.all(scale > 0), "invalid source scale")
        sa, sb = da / scale[None, :], db / scale[None, :]
    columns = [
        {"name": f"W{column}", "cosine_across_channels": _cosine(sa[active, column], sb[active, column])}
        for column in range(VELOCITY_DIM)
    ]
    columns.append({"name": "b", "cosine_across_channels": _cosine(sa[active, 7], sb[active, 7])})
    rows = [_cosine(sa[channel, :VELOCITY_DIM], sb[channel, :VELOCITY_DIM]) for channel in active]
    return {"active_channels": int(active.size), "columns": columns,
            "per_channel_7D_W_cosine": _quantiles(np.asarray([np.nan if value is None else value for value in rows]))}


def _later_encoding_delta(reference: BlockTable, fit: Mapping[str, Any], support_rates: np.ndarray) -> dict[str, Any]:
    rates, velocity = reference.rates, reference.velocity
    design = np.column_stack((np.ones(rates.shape[0]), velocity))
    pred_full = design @ np.asarray(fit["coefficient"], dtype=np.float64)
    pred_b = np.broadcast_to(np.asarray(support_rates, dtype=np.float64).mean(axis=0, keepdims=True), rates.shape)
    centered = rates - rates.mean(axis=0, keepdims=True)
    tss = np.square(centered).sum(axis=0)
    valid = tss > 1e-12
    r2_full = np.full(CHANNELS, np.nan)
    r2_b = np.full(CHANNELS, np.nan)
    r2_full[valid] = 1.0 - np.square(rates[:, valid] - pred_full[:, valid]).sum(axis=0) / tss[valid]
    r2_b[valid] = 1.0 - np.square(rates[:, valid] - pred_b[:, valid]).sum(axis=0) / tss[valid]
    for channel in fit["degenerate_channels"]:
        r2_full[channel] = np.nan
        r2_b[channel] = np.nan
    delta = r2_full - r2_b
    return {"r2_full": r2_full, "r2_intercept_only": r2_b, "delta_full_minus_b": delta,
            "summary": _quantiles(delta), "finite_channels": int(np.isfinite(delta).sum())}


def _phase_stability_ridge(support: BlockTable, reference: BlockTable, global_fit: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {"source": "native_epochs_at_100ms_block_center", "phases": {}}
    for phase in NATIVE_PHASES:
        sidx = np.asarray([index for index, value in enumerate(support.phases) if value == phase], dtype=np.int64)
        ridx = np.asarray([index for index, value in enumerate(reference.phases) if value == phase], dtype=np.int64)
        row: dict[str, Any] = {"support_blocks": int(sidx.size), "reference_blocks": int(ridx.size)}
        if sidx.size >= DESCRIPTOR_DIM and ridx.size >= DESCRIPTOR_DIM:
            try:
                a, b = fit_forward_pcf8_ridge100(support.rates[sidx], support.velocity[sidx]), fit_forward_pcf8_ridge100(reference.rates[ridx], reference.velocity[ridx])
                compare = _comparison(a, b)
                row.update({"status": "FIT", "support_regularized_condition": a["design"]["ridge100"]["regularized_system_condition_number"],
                            "reference_regularized_condition": b["design"]["ridge100"]["regularized_system_condition_number"],
                            "support_reference_W_column_cosines": [item["cosine_across_channels"] for item in compare["columns"][:7]],
                            "phase_vs_global_support_W_column_cosines": [_cosine(a["descriptor"][:, col], global_fit["descriptor"][:, col]) for col in range(7)]})
            except Pcf8PrecursorError as error:
                row.update({"status": "RANK_OR_CONSTRUCTIBILITY_FAIL", "reason": str(error)})
        else:
            row["status"] = "INSUFFICIENT_BLOCKS_FOR_Bx8"
        output["phases"][phase] = row
    return output


def _budget_row_ridge(record: Any, *, budget: int, source_scale: np.ndarray | None = None) -> tuple[dict[str, Any], np.ndarray]:
    _need(len(record.trials) >= budget + 1, f"{record.session_name}: no M={budget} later-trial boundary")
    support_trials, later_trials = record.trials[:budget], record.trials[budget:]
    support, reference = table_for_trials(record, support_trials), table_for_trials(record, later_trials)
    fit = fit_forward_pcf8_ridge100(support.rates, support.velocity)
    reference_fit = fit_forward_pcf8_ridge100(reference.rates, reference.velocity)
    raw_compare = _comparison(fit, reference_fit)
    normalized_compare = _comparison(fit, reference_fit, source_scale) if source_scale is not None else None
    ls_velocity, offset = fixed_point_free_velocity_null(support.velocity, session_name=record.session_name, budget=budget)
    ls_fit = fit_forward_pcf8_ridge100(support.rates, ls_velocity)
    full_delta = _later_encoding_delta(reference, fit, support.rates)
    ls_delta = _later_encoding_delta(reference, ls_fit, support.rates)
    delta_advantage = full_delta["delta_full_minus_b"] - ls_delta["delta_full_minus_b"]
    w_norm = np.linalg.norm(fit["descriptor"][:, :7], axis=1)
    active = np.asarray([index for index in range(CHANNELS) if index not in set(fit["degenerate_channels"])], dtype=np.int64)
    return ({
        "support_trials": [float(item.trial_number) for item in support_trials],
        "later_disjoint_reference_trials": [float(item.trial_number) for item in later_trials],
        "support_blocks": int(support.rates.shape[0]), "reference_blocks": int(reference.rates.shape[0]),
        "support_fit": {"design": fit["design"], "degenerate_channels": list(fit["degenerate_channels"])},
        "reference_fit": {"design": reference_fit["design"], "degenerate_channels": list(reference_fit["degenerate_channels"])},
        "support_later_raw": raw_compare,
        "support_later_after_source_scale_only": normalized_compare,
        "W_norm": {"summary": _quantiles(w_norm[active]), "near_zero_threshold": MACHINE_SCALE_FLOOR,
                   "near_zero_count": int(np.sum(w_norm[active] <= MACHINE_SCALE_FLOOR)), "active_channels": int(active.size)},
        "later_trial_encoding": {
            "full_minus_intercept_only": full_delta["summary"],
            "LS_minus_intercept_only": ls_delta["summary"],
            "full_minus_LS": _quantiles(delta_advantage),
            "LS_velocity_null": {"method": "cyclic fixed-point-free permutation of full 7D velocity rows across concatenated support blocks", "offset": offset},
        },
        "phase_conditioned_stability": _phase_stability_ridge(support, reference, fit),
    }, fit["descriptor"])


def _aggregate_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_row = [row["support_later_after_source_scale_only"]["per_channel_7D_W_cosine"]["median"] for row in rows]
    columns = [
        float(np.median([row["support_later_after_source_scale_only"]["columns"][index]["cosine_across_channels"] for row in rows]))
        for index in range(7)
    ]
    full_delta = [row["later_trial_encoding"]["full_minus_intercept_only"]["median"] for row in rows]
    advantage = [row["later_trial_encoding"]["full_minus_LS"]["median"] for row in rows]
    values = {
        "normalized_7D_W_median_cosine": float(np.median(normalized_row)),
        "median_column_cosines": columns,
        "columns_at_least_0p40": int(sum(value >= .40 for value in columns)),
        "later_trial_median_delta_r2_full_minus_b": float(np.median(full_delta)),
        "later_trial_median_delta_r2_full_minus_LS": float(np.median(advantage)),
    }
    gates = {
        "normalized_7D_W_median_cosine_ge_0p50": values["normalized_7D_W_median_cosine"] >= .50,
        "at_least_5_of_7_columns_median_ge_0p40": values["columns_at_least_0p40"] >= 5,
        "later_trial_delta_r2_full_minus_b_positive": values["later_trial_median_delta_r2_full_minus_b"] > 0.0,
        "later_trial_delta_r2_full_minus_LS_ge_0p01": values["later_trial_median_delta_r2_full_minus_LS"] >= .01,
    }
    return {"aggregate_values": values, "predeclared_gates": gates, "all_pass": bool(all(gates.values())),
            "aggregation": "median across the 11 per-recording medians; this is source constructibility only, not decoder R2."}


def pcf8_ridge100_arm_schema() -> dict[str, dict[str, str]]:
    arms = {name: dict(value) for name, value in pcf8_arm_schema().items()}
    arms["PCF8-FULL"]["carrier"] = "ridge100 [W0,W1,W2,W3,W4,W5,W6,b] from exact active M chronological support trials"
    arms["PCF8-B8"]["carrier"] = "[0,0,0,0,0,0,0,b_rate_only], with b_rate_only=mean support rate; separately fit/trained"
    arms["PCF8-LS"]["carrier"] = "ridge100 refit after deterministic fixed-point-free cyclic permutation of concatenated 100-ms full-7D velocity rows across exact active M support trials (M4 development; M3 organizer-held); neural rows unchanged"
    return arms


def audit_ridge100_source_records(records: Mapping[str, Any], expected_names: Sequence[str]) -> dict[str, Any]:
    names = tuple(expected_names)
    _need(tuple(records) == names and len(names) == 11, "requires exact ordered 11-source-record scope")
    # First pass fits only chronological support and establishes source-only
    # scale.  No reference values enter the normalizer.
    support_descriptors: dict[str, dict[int, np.ndarray]] = {name: {} for name in names}
    for name in names:
        record = records[name]
        _need(record.session_name == name and record.neural.shape[1] == CHANNELS and tuple(record.trial_values) == tuple(sorted(record.trial_values)),
              f"{name}: source identity/boundary drift")
        for budget in (4, 3):
            support = table_for_trials(record, record.trials[:budget])
            support_descriptors[name][budget] = fit_forward_pcf8_ridge100(support.rates, support.velocity)["descriptor"]
    scales = {budget: per_column_source_scale([support_descriptors[name][budget] for name in names]) for budget in (4, 3)}
    per_recording: dict[str, Any] = {}
    per_budget_rows: dict[int, list[dict[str, Any]]] = {4: [], 3: []}
    for name in names:
        record = records[name]
        m4, _ = _budget_row_ridge(record, budget=4, source_scale=np.asarray(scales[4]["applied_scale"]))
        m3, _ = _budget_row_ridge(record, budget=3, source_scale=np.asarray(scales[3]["applied_scale"]))
        per_recording[name] = {"input_sha256": str(record.input_sha256), "M4": m4, "M3": m3}
        per_budget_rows[4].append(m4); per_budget_rows[3].append(m3)
    gates = {"M4": _aggregate_gate(per_budget_rows[4]), "M3": _aggregate_gate(per_budget_rows[3])}
    return {
        "schema": V2_SCHEMA,
        "status": "PASS_CPU_CONSTRUCTIBILITY_FOR_SEPARATE_REVIEW" if gates["M4"]["all_pass"] and gates["M3"]["all_pass"] else "FAIL_PREDECLARED_SOURCE_CONSTRUCTIBILITY_GATES__NO_GPU_AUTHORIZATION",
        "scope": {"source_recordings_opened": 11, "target_recordings_opened": 0, "target_decoder_r2_computed": False, "gpu_used": False, "target_backward_steps": 0},
        "v1_relation": "v1 raw OLS audit is preserved immutable as a diagnostic; v2 is the only decision-bearing ridge100 correction.",
        "fit_contract": {"form": "rate_i=b_i+sum_j W_ij*v_j+epsilon", "design": "[ones,native_100ms_mean_velocity_7d] => Bx8", "lag_bins": 0,
            "ridge": {"lambda": RIDGE_LAMBDA, "intercept_penalized": False, "W_penalized": True},
            "preprocessing": "raw contiguous eval-valid five-bin spike sum / 0.1s; same-block mean native seven-dimensional velocity; no PCA/whitening/standardization/lambda grid"},
        "boundary": "exact 11 H1_M4_FOLD0_SOURCE recordings; M4 first4 versus later trial5+; ancillary organizer-held M3 first3 versus later trial4+; all references same-recording and disjoint.",
        "per_recording": per_recording,
        "source_per_column_scale": {"M4": scales[4], "M3": scales[3]},
        "gates": gates,
        "future_arm_schema": pcf8_ridge100_arm_schema(),
        "state_compute_contract": pcf8_compute_contract(),
        "interpretation_guard": "All gates are source constructibility gates. They do not predict decoder R2 or authorize GPU automatically; a failure closes PCF8.",
    }
