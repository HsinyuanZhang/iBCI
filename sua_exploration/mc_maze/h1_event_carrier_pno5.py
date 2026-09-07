"""CPU-only H1 PNO5: support-only population/trial nuisance orthogonalization.

The frozen deployable carrier is exactly ``[w1,w2,w3,w4,b]``.  On a target
support block only, a scalar nuisance is estimated for each event as its
trial's mean population log-rate, centred over support events.  That shared
offset is subtracted with fixed coefficient one before the ordinary closed-form
H-SE5 q=4 ridge fit.  The nuisance is *not* estimated at query time: later
raw log-rates are used only as scoring observations, and predictions are the
five-dimensional carrier alone.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as hse5


SCHEMA = "h1_event_carrier_pno5_source_screen_v1"
PROTOCOL = "h1_sparse_event_population_trial_nuisance_orthogonalization_20260812_v1"
RANK = 4
CARRIER_DIM = 5
SUPPORT_BUDGETS: tuple[int, ...] = (3, 4)
MATERIAL_MEAN = 0.02
MATERIAL_MEDIAN = 0.01
MINIMUM_POSITIVE = 10

# Immutable predeclaration.  There is intentionally no penalty/grid: the sole
# fit is H-SE5's literal ridge=3 after a fixed-coefficient support projection.
PREDECLARATION: Mapping[str, Any] = {
    "model": "Y_event,channel = alpha_channel + nuisance_event + Z_event @ W_channel",
    "nuisance": "support-only trial-mean population log-rate, zero-mean over support; coefficient fixed to one",
    "carrier_order": ["w1", "w2", "w3", "w4", "b"],
    "selection": "no grid or penalty selection; no outer future arm may influence model choice",
    "query_rule": "raw later-event log-rate scoring only; no query population mean, labels, nuisance estimate, or future fit",
}


def _need(condition: bool, message: str) -> None:
    v1._need(condition, message)


def source_names_for_outer(outer_date: str) -> tuple[str, ...]:
    _need(outer_date in v1.H1_DATES, f"PNO5 unknown outer date {outer_date}")
    return tuple(name for name in v1.H1_HELDIN_SESSIONS if v1.session_date(name) != outer_date)


def fit_endpoint_map(sessions: Mapping[str, v1.EventSession], *, outer_date: str) -> hse5.EndpointBasisV2:
    """The exact source-date-excluded H-SE5 endpoint map."""
    _need(all(v1.session_date(name) != outer_date for name in source_names_for_outer(outer_date)), "PNO5 outer leak")
    return hse5.fit_source_all_event_basis(sessions, outer_date=outer_date)


def support_trial_population_nuisance(events: Sequence[v1.MovementEvent], response: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Identify the support-only shared trial offset with a zero-mean gauge.

    The scalar event population mean is first measured across the fixed 176
    channels and is then averaged inside each support trial.  Thus all events
    from a trial receive one shared effect.  Centre it over events so it cannot
    alter the carrier intercept.  No endpoint, later event, or query statistic
    occurs in this calculation.
    """
    y = np.asarray(response, dtype=np.float64)
    _need(y.ndim == 2 and y.shape == (len(events), v1.EXPECTED_NEURONS) and len(events) > 0,
          "PNO5 nuisance response/event shape drift")
    populations = y.mean(axis=1)
    trial_indices = np.asarray([event.trial_index for event in events], dtype=np.int64)
    nuisance = np.empty(len(events), dtype=np.float64)
    for trial in np.unique(trial_indices):
        mask = trial_indices == trial
        nuisance[mask] = populations[mask].mean()
    nuisance -= nuisance.mean()
    _need(np.isfinite(nuisance).all() and abs(float(nuisance.mean())) <= 1.0e-12, "PNO5 nuisance gauge drift")
    return nuisance, {
        "estimator": "support_trial_mean_of_event_population_log_rate_centered",
        "support_only": True, "uses_endpoint_labels": False, "uses_query_statistics": False,
        "support_trial_count": int(np.unique(trial_indices).size),
        "event_population_mean_sha256": v1.array_sha256(populations),
        "nuisance_sha256": v1.array_sha256(nuisance), "nuisance_mean": float(nuisance.mean()),
    }


def _nuisance_order(session: str, budget: int, count: int) -> tuple[np.ndarray, dict[str, Any]]:
    key = f"{v1.SHUFFLE_NAMESPACE}:pno5-nuisance-row:{session}:M{budget}"
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little"))
    identity = np.arange(count, dtype=np.int64); order = identity.copy()
    for _ in range(100):
        rng.shuffle(order)
        if not np.any(order == identity):
            break
    _need(not np.any(order == identity), "PNO5 nuisance shuffle fixed point")
    return order, {"namespace": "h1-pno5-nuisance-row-shuffle-v1", "order_sha256": v1.array_sha256(order), "fixed_points": 0}


def fit_pno5_carrier(z: np.ndarray, response: np.ndarray, nuisance: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Closed-form ridge after the fixed support-only common-offset projection."""
    y = np.asarray(response, dtype=np.float64); n = np.asarray(nuisance, dtype=np.float64)
    _need(z.shape == (y.shape[0], RANK) and y.shape[1] == v1.EXPECTED_NEURONS and n.shape == (y.shape[0],),
          "PNO5 fit shape drift")
    adjusted = y - n[:, None]
    carrier = hse5.fit_carrier_arrays(z, adjusted)
    _need(carrier.shape == (v1.EXPECTED_NEURONS, CARRIER_DIM) and np.isfinite(carrier).all(), "PNO5 carrier drift")
    return carrier, {"estimator": "closed_form_hse5_ridge_on_support_nuisance_residual", "ridge_lambda": hse5.RIDGE_LAMBDA,
                     "target_optimizer_steps": 0, "target_backward_steps": 0, "carrier_sha256": v1.array_sha256(carrier)}


def intercept_carrier(response: np.ndarray) -> np.ndarray:
    y = np.asarray(response, dtype=np.float64)
    _need(y.ndim == 2 and y.shape[1] == v1.EXPECTED_NEURONS, "PNO5 intercept shape")
    return np.column_stack((np.zeros((v1.EXPECTED_NEURONS, RANK)), y.mean(axis=0)))


def nuisance_only_carrier(response: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    """No-pairing control. Query nuisance is forbidden, so deployment is b only."""
    y = np.asarray(response, dtype=np.float64); n = np.asarray(nuisance, dtype=np.float64)
    _need(n.shape == (y.shape[0],), "PNO5 nuisance-only shape")
    # n has a zero support mean: this is intentionally equivalent to intercept.
    return np.column_stack((np.zeros((v1.EXPECTED_NEURONS, RANK)), (y - n[:, None]).mean(axis=0)))


def predict(carrier: np.ndarray, z: np.ndarray) -> np.ndarray:
    value = np.asarray(carrier, dtype=np.float64); latent = np.asarray(z, dtype=np.float64)
    _need(value.shape == (v1.EXPECTED_NEURONS, CARRIER_DIM) and latent.ndim == 2 and latent.shape[1] == RANK,
          "PNO5 prediction must be [176,5] by [events,4]")
    return latent @ value[:, :RANK].T + value[:, RANK][None, :]


def _arrays(session: v1.EventSession, basis: hse5.EndpointBasisV2, budget: int) -> tuple[Any, ...]:
    support, later = session.events_before(budget), session.events_after(budget)
    _need(len(support) >= 8 and len(later) >= 4, f"{session.session_name}: PNO5 sparse event count")
    zs, ys = hse5.event_arrays(support, basis)
    zl, yl = hse5.event_arrays(later, basis)
    return support, later, zs, ys, zl, yl


def hse5_baseline_session(session: v1.EventSession, basis: hse5.EndpointBasisV2, *, budget: int) -> dict[str, Any]:
    support, _later, zs, ys, zl, yl = _arrays(session, basis, budget)
    correct = hse5.fit_carrier_arrays(zs, ys)
    order, _ = v1.within_trial_label_shuffle(support, session=session.session_name, budget=budget)
    label = hse5.fit_carrier_arrays(zs[order], ys)
    intercept = intercept_carrier(ys)
    scores = [v1.r2_by_channel(yl, hse5.predict(item, zl)) for item in (correct, label, intercept)]
    defined = np.logical_and.reduce([np.isfinite(score) for score in scores]); _need(np.any(defined), "PNO5 HSE5 undefined")
    return {"median_r2_correct": float(np.median(scores[0][defined])),
            "median_delta_label_shuffle": float(np.median((scores[0] - scores[1])[defined])),
            "median_delta_intercept": float(np.median((scores[0] - scores[2])[defined])),
            "carrier_sha256": v1.array_sha256(correct)}


def evaluate_pno5_session(session: v1.EventSession, basis: hse5.EndpointBasisV2, *, budget: int) -> dict[str, Any]:
    _need(basis.outer_date == session.date, "PNO5 outer basis/session mismatch")
    support, later, zs, ys, zl, yl = _arrays(session, basis, budget)
    nuisance, nuisance_manifest = support_trial_population_nuisance(support, ys)
    correct, correct_fit = fit_pno5_carrier(zs, ys, nuisance)
    label_order, label_manifest = v1.within_trial_label_shuffle(support, session=session.session_name, budget=budget)
    label, label_fit = fit_pno5_carrier(zs[label_order], ys, nuisance)
    nuisance_order, nuisance_shuffle = _nuisance_order(session.session_name, budget, len(support))
    nuisance_row, nuisance_fit = fit_pno5_carrier(zs, ys, nuisance[nuisance_order])
    row_carrier, attachment_manifest = hse5.row_shuffle(correct, session=session.session_name, budget=budget)
    intercept = intercept_carrier(ys); nuisance_only = nuisance_only_carrier(ys, nuisance)
    carriers = (correct, label, nuisance_row, row_carrier, intercept, nuisance_only)
    score = [v1.r2_by_channel(yl, predict(item, zl)) for item in carriers]
    defined = np.logical_and.reduce([np.isfinite(item) for item in score]); _need(np.any(defined), "PNO5 undefined forward channels")
    median = lambda values: float(np.median(values[defined]))
    return {
        "session": session.session_name, "date": session.date, "budget": budget, "support_events": len(support), "later_events": len(later),
        "design_rank": int(np.linalg.matrix_rank(np.column_stack((np.ones(len(zs)), zs)))), "carrier_dim": CARRIER_DIM,
        "defined_channels": int(defined.sum()), "median_r2_correct": median(score[0]),
        "median_delta_label_shuffle": median(score[0] - score[1]), "median_delta_nuisance_row_shuffle": median(score[0] - score[2]),
        "median_delta_carrier_attachment_shuffle": median(score[0] - score[3]), "median_delta_intercept": median(score[0] - score[4]),
        "median_delta_nuisance_only": median(score[0] - score[5]), "correct_carrier_sha256": v1.array_sha256(correct),
        "label_carrier_sha256": v1.array_sha256(label), "nuisance_row_carrier_sha256": v1.array_sha256(nuisance_row),
        "attachment_carrier_sha256": v1.array_sha256(row_carrier), "intercept_carrier_sha256": v1.array_sha256(intercept),
        "nuisance_only_carrier_sha256": v1.array_sha256(nuisance_only), "correct_fit": correct_fit, "label_fit": label_fit,
        "nuisance_row_fit": nuisance_fit, "nuisance": nuisance_manifest, "label_shuffle": label_manifest,
        "nuisance_row_shuffle": nuisance_shuffle, "carrier_attachment_shuffle": attachment_manifest,
        "nuisance_only": {"endpoint_pairing_used": False, "query_nuisance_estimated": False,
                          "equals_intercept_due_to_zero_mean_support_gauge": bool(np.array_equal(intercept, nuisance_only))},
        "outer_future_used_only_for_raw_log_rate_scoring": True, "query_population_mean_used": False,
        "query_labels_used": False, "query_nuisance_estimated": False,
    }


def _summary(rows: Mapping[str, Mapping[str, Any]], field: str) -> dict[str, Any]:
    return v1.paired_summary([(name, row[field]) for name, row in rows.items()])


def _positive(summary: Mapping[str, Any]) -> bool:
    return bool(summary["defined_sessions"] == 13 and summary["mean"] > 0 and summary["median"] > 0
                and summary["positive"] >= MINIMUM_POSITIVE and summary["leave_largest_absolute_out_mean"] > 0)


def aggregate(rows: Mapping[str, Mapping[str, Any]], baseline: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    augmented = {name: {**row, "delta_vs_hse5": row["median_r2_correct"] - baseline[name]["median_r2_correct"]} for name, row in rows.items()}
    fields = ("median_r2_correct", "delta_vs_hse5", "median_delta_label_shuffle", "median_delta_nuisance_row_shuffle",
              "median_delta_carrier_attachment_shuffle", "median_delta_intercept", "median_delta_nuisance_only")
    names = ("correct_r2", "correct_minus_hse5", "correct_minus_label_shuffle", "correct_minus_nuisance_row_shuffle",
             "correct_minus_carrier_attachment_shuffle", "correct_minus_intercept", "correct_minus_nuisance_only")
    return {name: _summary(augmented, field) for name, field in zip(names, fields, strict=True)}


def gate(value: Mapping[str, Any]) -> dict[str, Any]:
    gain = value["correct_minus_hse5"]
    material = bool(_positive(gain) and gain["mean"] >= MATERIAL_MEAN and gain["median"] >= MATERIAL_MEDIAN)
    required = ("correct_minus_label_shuffle", "correct_minus_nuisance_only", "correct_minus_intercept",
                "correct_minus_nuisance_row_shuffle", "correct_minus_carrier_attachment_shuffle")
    controls = {key: _positive(value[key]) for key in required}
    return {"passed": bool(material and all(controls.values())), "material_gain_vs_hse5": material, **controls,
            "thresholds": {"mean_delta_vs_hse5": MATERIAL_MEAN, "median_delta_vs_hse5": MATERIAL_MEDIAN,
                           "minimum_positive_sessions": MINIMUM_POSITIVE, "leave_largest_absolute_out_mean_positive": True}}


def run_screen(sessions: Mapping[str, v1.EventSession]) -> dict[str, Any]:
    _need(tuple(sessions) == v1.H1_HELDIN_SESSIONS, "PNO5 exact 13 session allowlist/order")
    budgets: dict[str, Any] = {}
    for budget in SUPPORT_BUDGETS:
        maps = {date: fit_endpoint_map(sessions, outer_date=date) for date in v1.H1_DATES}
        rows, baseline = {}, {}
        for name in v1.H1_HELDIN_SESSIONS:
            date = v1.session_date(name)
            rows[name] = evaluate_pno5_session(sessions[name], maps[date], budget=budget)
            baseline[name] = hse5_baseline_session(sessions[name], maps[date], budget=budget)
        combined = aggregate(rows, baseline)
        budgets[f"M{budget}"] = {"budget_trials": budget, "endpoint_map_by_outer_date": {date: maps[date].manifest() for date in v1.H1_DATES},
                                  "hse5_baseline_sessions": baseline, "pno5_sessions": rows, "aggregate": combined, "gate": gate(combined)}
    passed = all(budgets[f"M{budget}"]["gate"]["passed"] for budget in SUPPORT_BUDGETS)
    return {"schema": SCHEMA, "protocol": PROTOCOL, "candidate_matrix_predeclared_before_data_run": True,
            "predeclaration": dict(PREDECLARATION), "frozen_constants": {"rank": RANK, "carrier_dim": CARRIER_DIM,
            "carrier_order": list(PREDECLARATION["carrier_order"]), "support_budgets": list(SUPPORT_BUDGETS),
            "hse5_ridge_lambda": hse5.RIDGE_LAMBDA, "model_grid": [], "outer_date_selection": "none"}, "budgets": budgets,
            "status": "PASS_CPU_PNO5_MATERIAL" if passed else "STOP_CPU_PNO5_NOT_MATERIAL", "gpu_authorized_by_this_screen": False,
            "scope": {"public_held_in_calibration_nwbs_opened": 13, "minival_nwbs_opened": 0, "held_out_nwbs_opened": 0,
                      "formal_test_labels_opened": 0, "dense_velocity_opened": False, "target_session_optimizer_steps": 0,
                      "target_session_backward_steps": 0, "decoder_constructed": False, "trainer_constructed": False, "cuda_used": False,
                      "query_population_mean_used": False, "query_labels_used": False, "query_future_fit_used": False},
            "interpretation": {"nuisance_identifiable_only_under_fixed_unit_shared_offset_and_zero_mean_gauge": True,
                               "query_nuisance_is_forbidden_so_nuisance_only_deploys_as_intercept": True,
                               "fail_closed_if_support_trial_population_nuisance_is_nonfinite": True}}
