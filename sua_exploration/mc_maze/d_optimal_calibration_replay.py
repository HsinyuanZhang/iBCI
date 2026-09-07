"""Receipt builders and fail-closed aggregation for B9 D-optimal replay."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mc_maze.decoder_attention_diagnostic import assert_no_sealed_sessions
from mc_maze.d_optimal_calibration_design import (
    CANONICAL_DIRECTIONS_RAD,
    DEFAULT_BUDGET_LADDER,
    DEFAULT_CANDIDATE_POOL_K,
    FROZEN_ALGORITHM_PARAMETERS,
    FROZEN_PRIMARY_GATE_BUDGETS,
    FROZEN_PRIMARY_GATE_MIN_MEDIAN_DELTA,
    design_matrix_from_thetas,
    design_metrics,
    evaluate_carrier_fidelity,
    fit_carriers_from_selected_trials,
    index_span,
    random_subset_indices,
    select_calibration_trials,
    spans_match,
    split_half_carrier_reproducibility,
)

SCHEMA_VERSION = "d_optimal_calibration_replay_v2"

REPLAY_ARMS = (
    "chronological_first_m",
    "d_optimal_prefix_k",
    "random_m_prefix_k",
    "random_m_span_matched_k",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_synthetic_session_payload(
    session_name: str,
    *,
    seed: int = 42,
    units: int = 8,
    candidate_pool_k: int = DEFAULT_CANDIDATE_POOL_K,
    post_pool_trials: int = 10,
    nonstationary: bool = False,
) -> dict[str, Any]:
    """Synthetic pool; optional post-pool tail and planted nonstationarity."""
    assert_no_sealed_sessions([session_name])
    rng = np.random.Generator(np.random.PCG64(seed))
    total_trials = candidate_pool_k + post_pool_trials
    base_dirs = np.arange(8, dtype=np.int64)
    direction_indices = np.tile(base_dirs, int(np.ceil(total_trials / 8)))[:total_trials]
    rng.shuffle(direction_indices)
    thetas = np.array(
        [CANONICAL_DIRECTIONS_RAD[int(index)] for index in direction_indices],
        dtype=np.float64,
    )
    trial_times = np.arange(total_trials, dtype=np.float64)
    trial_rates = np.zeros((total_trials, units), dtype=np.float64)
    midpoint = candidate_pool_k // 2
    for unit in range(units):
        a_early = float(rng.normal(3.0, 0.2))
        c_early = float(rng.normal(1.0, 0.2))
        b_early = float(rng.normal(5.0, 0.2)) + unit * 0.1
        a_late = float(rng.normal(-2.5, 0.2))
        c_late = float(rng.normal(-0.8, 0.2))
        b_late = float(rng.normal(7.0, 0.2)) + unit * 0.1
        for trial in range(total_trials):
            theta = float(thetas[trial])
            if not nonstationary or trial < midpoint:
                a, c, b = a_early, c_early, b_early
            else:
                a, c, b = a_late, c_late, b_late
            trial_rates[trial, unit] = b + a * math.cos(theta) + c * math.sin(theta)
    return {
        "session_name": session_name,
        "candidate_pool_k": candidate_pool_k,
        "post_pool_trials": post_pool_trials,
        "direction_indices": direction_indices.tolist(),
        "thetas_rad": thetas.tolist(),
        "trial_times": trial_times.tolist(),
        "trial_rates": trial_rates.tolist(),
        "units": units,
        "nonstationary": bool(nonstationary),
    }


def build_phase_flip_session_payload(
    session_name: str,
    *,
    seed: int = 1,
    units: int = 4,
    candidate_pool_k: int = DEFAULT_CANDIDATE_POOL_K,
    post_pool_trials: int = 0,
) -> dict[str, Any]:
    """Deterministic early/late opposite-tuning plant for nonstationary fidelity tests."""
    assert_no_sealed_sessions([session_name])
    rng = np.random.Generator(np.random.PCG64(seed))
    total_trials = candidate_pool_k + post_pool_trials
    direction_indices = np.tile(np.arange(8, dtype=np.int64), int(np.ceil(total_trials / 8)))[:total_trials]
    rng.shuffle(direction_indices)
    thetas = np.array(
        [CANONICAL_DIRECTIONS_RAD[int(index)] for index in direction_indices],
        dtype=np.float64,
    )
    trial_times = np.arange(total_trials, dtype=np.float64)
    trial_rates = np.zeros((total_trials, units), dtype=np.float64)
    midpoint = candidate_pool_k // 2
    a1, c1, b1 = 3.0, 1.0, 5.0
    a2, c2, b2 = -2.5, -0.8, 7.0
    for unit in range(units):
        for trial in range(total_trials):
            theta = float(thetas[trial])
            if trial < midpoint:
                a, c, b = a1, c1, b1 + unit * 0.1
            else:
                a, c, b = a2, c2, b2 + unit * 0.1
            trial_rates[trial, unit] = b + a * math.cos(theta) + c * math.sin(theta)
    return {
        "session_name": session_name,
        "candidate_pool_k": candidate_pool_k,
        "post_pool_trials": post_pool_trials,
        "direction_indices": direction_indices.tolist(),
        "thetas_rad": thetas.tolist(),
        "trial_times": trial_times.tolist(),
        "trial_rates": trial_rates.tolist(),
        "units": units,
        "nonstationary": True,
        "fixture": "phase_flip_planted",
    }


def _arm_row(
    *,
    arm: str,
    candidate_thetas: np.ndarray,
    thetas_rad: np.ndarray,
    direction_indices: np.ndarray,
    trial_rates: np.ndarray,
    trial_times: np.ndarray | None,
    pool_k: int,
    post_pool_indices: np.ndarray,
    budget_m: int,
    session_name: str,
    candidate_trials: list[dict[str, Any]],
    seed: int = 42,
    span_match_target: int | None = None,
) -> dict[str, Any]:
    selection = select_calibration_trials(
        candidate_thetas,
        budget_m,
        arm,
        seed=seed,
        session_name=session_name,
        candidate_trials_for_leakage_check=candidate_trials,
        span_match_target=span_match_target,
        trial_times=trial_times,
    )
    selected = np.asarray(selection.selected_indices, dtype=np.int64)
    fidelity = evaluate_carrier_fidelity(
        trial_rates,
        direction_indices,
        thetas_rad,
        selected,
        pool_size=pool_k,
        post_pool_indices=post_pool_indices if post_pool_indices.size else None,
    )
    split_half = (
        split_half_carrier_reproducibility(trial_rates[:pool_k], direction_indices[:pool_k], selected)
        if selected.size >= 4
        else {"median_cosine_ac": None}
    )
    return {
        "selection": selection.as_dict(),
        "carrier_fidelity": fidelity,
        "split_half": split_half,
        "implementation_sanity_checks": {
            "design_metrics": dict(selection.design_metrics),
            "label": "sanity_only_not_primary_gate",
        },
    }


def replay_session(
    session_payload: Mapping[str, Any],
    *,
    budgets: Sequence[int] = DEFAULT_BUDGET_LADDER,
    random_seeds: Sequence[int] = (42, 43, 44),
) -> dict[str, Any]:
    session_name = str(session_payload["session_name"])
    assert_no_sealed_sessions([session_name])
    thetas = np.asarray(session_payload["thetas_rad"], dtype=np.float64)
    direction_indices = np.asarray(session_payload["direction_indices"], dtype=np.int64)
    trial_rates = np.asarray(session_payload["trial_rates"], dtype=np.float64)
    trial_times = np.asarray(session_payload.get("trial_times", np.arange(thetas.size)), dtype=np.float64)
    pool_k = int(session_payload.get("candidate_pool_k", thetas.size))
    post_pool_trials = int(session_payload.get("post_pool_trials", 0))
    _require(thetas.size >= pool_k + post_pool_trials, "payload shorter than declared pool + post-pool")
    candidate_thetas = thetas[:pool_k]
    candidate_trials = [{"target_dir": float(theta)} for theta in candidate_thetas]
    post_pool_indices = np.arange(pool_k, pool_k + post_pool_trials, dtype=np.int64)

    per_budget: dict[str, Any] = {}
    for budget_m in budgets:
        _require(pool_k >= budget_m, f"pool K={pool_k} < budget M={budget_m}")
        chrono_row = _arm_row(
            arm="chronological_first_m",
            candidate_thetas=candidate_thetas,
            thetas_rad=thetas,
            direction_indices=direction_indices,
            trial_rates=trial_rates,
            trial_times=trial_times[:pool_k],
            pool_k=pool_k,
            post_pool_indices=post_pool_indices,
            budget_m=budget_m,
            session_name=session_name,
            candidate_trials=candidate_trials,
        )
        d_opt_row = _arm_row(
            arm="d_optimal_prefix_k",
            candidate_thetas=candidate_thetas,
            thetas_rad=thetas,
            direction_indices=direction_indices,
            trial_rates=trial_rates,
            trial_times=trial_times[:pool_k],
            pool_k=pool_k,
            post_pool_indices=post_pool_indices,
            budget_m=budget_m,
            session_name=session_name,
            candidate_trials=candidate_trials,
        )
        d_opt_span = int(d_opt_row["selection"]["temporal_coverage"]["index_span"])
        chrono_span = int(chrono_row["selection"]["temporal_coverage"]["index_span"])

        random_rows: list[dict[str, Any]] = []
        for seed in random_seeds:
            selected = random_subset_indices(pool_k, budget_m, seed)
            random_rows.append(
                {
                    "seed": int(seed),
                    "design_metrics": design_metrics(
                        design_matrix_from_thetas(candidate_thetas[selected])
                    ),
                    "index_span": index_span(selected),
                }
            )
        span_rows: list[dict[str, Any]] = []
        for seed in random_seeds:
            span_rows.append(
                _arm_row(
                    arm="random_m_span_matched_k",
                    candidate_thetas=candidate_thetas,
                    thetas_rad=thetas,
                    direction_indices=direction_indices,
                    trial_rates=trial_rates,
                    trial_times=trial_times[:pool_k],
                    pool_k=pool_k,
                    post_pool_indices=post_pool_indices,
                    budget_m=budget_m,
                    session_name=session_name,
                    candidate_trials=candidate_trials,
                    seed=seed,
                    span_match_target=d_opt_span,
                )
            )

        d_opt_fidelity = d_opt_row["carrier_fidelity"]["carrier_fidelity_score"]
        chrono_fidelity = chrono_row["carrier_fidelity"]["carrier_fidelity_score"]
        fidelity_delta = None
        if d_opt_fidelity is not None and chrono_fidelity is not None:
            fidelity_delta = float(d_opt_fidelity - chrono_fidelity)

        span_fidelity_scores = [
            row["carrier_fidelity"]["carrier_fidelity_score"]
            for row in span_rows
            if row["carrier_fidelity"]["carrier_fidelity_score"] is not None
        ]
        mean_span_fidelity = float(np.mean(span_fidelity_scores)) if span_fidelity_scores else None

        d_opt_det = d_opt_row["implementation_sanity_checks"]["design_metrics"]["det_xtx"]
        chrono_det = chrono_row["implementation_sanity_checks"]["design_metrics"]["det_xtx"]
        random_det_mean = float(np.mean([row["design_metrics"]["det_xtx"] for row in random_rows]))

        per_budget[f"M{budget_m}"] = {
            "chronological_first_m": chrono_row,
            "d_optimal_prefix_k": d_opt_row,
            "random_m_prefix_k": {
                "seeds": random_rows,
                "mean_det_xtx": random_det_mean,
                "implementation_sanity_checks": {
                    "label": "sanity_only_not_primary_gate",
                    "mean_det_xtx": random_det_mean,
                },
            },
            "random_m_span_matched_k": {
                "seeds": span_rows,
                "target_index_span": d_opt_span,
                "chronological_index_span": chrono_span,
                "mean_carrier_fidelity_score": mean_span_fidelity,
            },
            "primary_endpoint": {
                "carrier_fidelity_score_d_optimal": d_opt_fidelity,
                "carrier_fidelity_score_chronological": chrono_fidelity,
                "paired_delta_d_optimal_minus_chronological": fidelity_delta,
                "median_cosine_ac_delta": (
                    float(d_opt_row["carrier_fidelity"]["median_cosine_ac_vs_reference"])
                    - float(chrono_row["carrier_fidelity"]["median_cosine_ac_vs_reference"])
                    if d_opt_row["carrier_fidelity"]["median_cosine_ac_vs_reference"] is not None
                    and chrono_row["carrier_fidelity"]["median_cosine_ac_vs_reference"] is not None
                    else None
                ),
            },
            "implementation_sanity_checks": {
                "label": "sanity_only_not_primary_gate",
                "det_xtx_gain_d_optimal_minus_chronological": float(d_opt_det - chrono_det),
                "d_optimal_beats_mean_random_det_xtx": bool(d_opt_det >= random_det_mean),
                "condition_ratio_chronological_over_d_optimal": float(
                    chrono_row["implementation_sanity_checks"]["design_metrics"]["design_condition"]
                    / max(
                        d_opt_row["implementation_sanity_checks"]["design_metrics"]["design_condition"],
                        1.0e-12,
                    )
                ),
            },
            "coverage_audit": {
                "chronological_temporal_coverage": chrono_row["selection"]["temporal_coverage"],
                "d_optimal_temporal_coverage": d_opt_row["selection"]["temporal_coverage"],
            },
        }

    return {
        "session_name": session_name,
        "candidate_pool_k": pool_k,
        "post_pool_trials": post_pool_trials,
        "units": int(trial_rates.shape[1]),
        "per_budget": per_budget,
    }


def build_receipt(
    session_payloads: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    session_names = [str(item["session_name"]) for item in session_payloads]
    assert_no_sealed_sessions(session_names)
    sessions = [replay_session(item) for item in session_payloads]
    primary_deltas: dict[str, list[float]] = {f"M{m}": [] for m in FROZEN_PRIMARY_GATE_BUDGETS}
    for session_row in sessions:
        for budget_m in FROZEN_PRIMARY_GATE_BUDGETS:
            key = f"M{budget_m}"
            delta = session_row["per_budget"][key]["primary_endpoint"]["paired_delta_d_optimal_minus_chronological"]
            if delta is not None:
                primary_deltas[key].append(float(delta))

    primary_gate_summary = {
        "endpoint": "carrier_fidelity_score_delta_d_optimal_minus_chronological",
        "minimum_median_session_delta": FROZEN_PRIMARY_GATE_MIN_MEDIAN_DELTA,
        "per_budget_median_delta": {
            key: float(np.median(values)) if values else None for key, values in primary_deltas.items()
        },
        "passes_at_m10_and_m15": bool(
            all(
                primary_deltas.get(f"M{m}")
                and float(np.median(primary_deltas[f"M{m}"])) >= FROZEN_PRIMARY_GATE_MIN_MEDIAN_DELTA
                for m in FROZEN_PRIMARY_GATE_BUDGETS
            )
        ),
    }

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "device": device,
        "cuda_visible_devices": "",
        "sealed_test_sessions_opened": False,
        "sessions": session_names,
        "resolved_seed": int(seed),
        "algorithm": dict(FROZEN_ALGORITHM_PARAMETERS),
        "candidate_pool_k": DEFAULT_CANDIDATE_POOL_K,
        "budget_ladder": list(DEFAULT_BUDGET_LADDER),
        "session_results": sessions,
        "primary_gate_summary": primary_gate_summary,
        "leakage_assertion": {
            "selector_uses_label_geometry_only": True,
            "forbidden_trial_keys_enforced": True,
            "encoded_in_code": "mc_maze.d_optimal_calibration_design.assert_leakage_boundary",
        },
        "endpoint_scope": "carrier_estimation_quality_only_not_decoder_r2",
        "gpu_authorized": False,
        "training_authorized": False,
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    receipt["generated_at"] = datetime.now(timezone.utc).isoformat()
    return receipt


def validate_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed aggregator for a single replay receipt."""
    _require(receipt.get("schema_version") == SCHEMA_VERSION, "schema_version drift")
    _require(receipt.get("sealed_test_sessions_opened") is False, "sealed sessions opened")
    _require(receipt.get("device") == "cpu", "non-CPU device in receipt")
    _require(receipt.get("gpu_authorized") is False, "GPU run claimed in receipt")
    leakage = receipt.get("leakage_assertion") or {}
    _require(
        leakage.get("selector_uses_label_geometry_only") is True
        and leakage.get("forbidden_trial_keys_enforced") is True,
        "leakage assertion missing or false",
    )
    algorithm = receipt.get("algorithm") or {}
    for key, value in FROZEN_ALGORITHM_PARAMETERS.items():
        _require(algorithm.get(key) == value, f"algorithm parameter drift: {key}")
    ladder = receipt.get("budget_ladder")
    _require(list(ladder) == list(DEFAULT_BUDGET_LADDER), "incomplete or drifted budget ladder")
    sessions = receipt.get("sessions") or []
    assert_no_sealed_sessions(sessions)
    session_results = receipt.get("session_results") or []
    _require(len(session_results) == len(sessions), "session_results incomplete")
    _require(receipt.get("primary_gate_summary") is not None, "missing primary_gate_summary")

    for session_row in session_results:
        per_budget = session_row.get("per_budget") or {}
        for budget_key in [f"M{m}" for m in DEFAULT_BUDGET_LADDER]:
            _require(budget_key in per_budget, f"missing budget arm {budget_key}")
            for arm in REPLAY_ARMS:
                _require(arm in per_budget[budget_key], f"missing arm {arm} at {budget_key}")
            for arm in ("chronological_first_m", "d_optimal_prefix_k"):
                selection = per_budget[budget_key][arm]["selection"]
                _require(selection.get("leakage_assertion_passed") is True, "leakage flag false")
                _require(selection.get("temporal_coverage") is not None, "missing temporal_coverage")
                _require(selection.get("selected_indices") is not None, "missing selected_indices")
            primary = per_budget[budget_key]["primary_endpoint"]
            _require(primary.get("paired_delta_d_optimal_minus_chronological") is not None, "missing primary delta")
            span_target = per_budget[budget_key]["random_m_span_matched_k"]["target_index_span"]
            for span_row in per_budget[budget_key]["random_m_span_matched_k"]["seeds"]:
                selected_span = int(span_row["selection"]["temporal_coverage"]["index_span"])
                _require(
                    spans_match(selected_span, span_target),
                    "span-matched null outside tolerance",
                )

    return {
        "status": "valid",
        "sessions": len(sessions),
        "budgets": list(DEFAULT_BUDGET_LADDER),
        "primary_gate_summary": receipt.get("primary_gate_summary"),
    }


def write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
