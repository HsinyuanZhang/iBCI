"""H1 event-count dose-response sweep for the sealed H-SE5 carrier."""
from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2


SCHEMA = "h1_event_budget_sweep_v1"
PROTOCOL = "h1_event_budget_sweep_20260812_v1"
MODULE_NAME = "h1_event_budget_sweep"
SHUFFLE_NAMESPACE = "h1-event-budget-sweep-v1"
BUDGET_TRIALS = 4
EVENT_COUNTS: tuple[int | str, ...] = (8, 10, 12, 14, 16, 20, "all")
HSE5_TOLERANCE = 1.0e-10


def _k_value(k: int | str, support_count: int) -> int:
    return support_count if k == "all" else int(k)


def shuffle_mode(events: Sequence[v1.MovementEvent]) -> str:
    counts = Counter(event.trial_index for event in events)
    if all(count >= 2 for count in counts.values()):
        return "within_trial"
    return "global_cyclic"


def global_cyclic_shuffle_order(
    event_count: int, *, session: str, budget: int, k: int | str,
) -> tuple[np.ndarray, dict[str, Any]]:
    v1._need(event_count >= 2, "global cyclic shuffle requires at least two events")
    key = f"{SHUFFLE_NAMESPACE}:{MODULE_NAME}:{session}:M{budget}:k{k}"
    shift = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little") % (event_count - 1) + 1
    order = np.roll(np.arange(event_count, dtype=np.int64), shift)
    v1._need(not np.any(order == np.arange(event_count)), "global cyclic shuffle has fixed points")
    return order, {
        "namespace": SHUFFLE_NAMESPACE,
        "session": session,
        "budget": budget,
        "k": k,
        "shift": int(shift),
        "order_sha256": v1.array_sha256(order),
        "fixed_points": 0,
    }


def shuffled_latent(
    z: np.ndarray,
    events: Sequence[v1.MovementEvent],
    *,
    session: str,
    budget: int,
    k: int | str,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    mode = shuffle_mode(events)
    if mode == "within_trial":
        order, manifest = v1.within_trial_label_shuffle(events, session=session, budget=budget)
        return z[order], mode, manifest
    order, manifest = global_cyclic_shuffle_order(len(events), session=session, budget=budget, k=k)
    return z[order], mode, manifest


def median_carrier_weight_cosine(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    w_ref = np.asarray(reference, dtype=np.float64)[:, : v2.LATENT_DIM]
    w_cand = np.asarray(candidate, dtype=np.float64)[:, : v2.LATENT_DIM]
    n_ref = np.linalg.norm(w_ref, axis=1)
    n_cand = np.linalg.norm(w_cand, axis=1)
    defined = (n_ref > v1.NORM_FLOOR) & (n_cand > v1.NORM_FLOOR)
    if not np.any(defined):
        return None
    cosines = np.sum(w_ref[defined] * w_cand[defined], axis=1) / (n_ref[defined] * n_cand[defined])
    return float(np.median(cosines))


def fit_support_carrier(
    events: Sequence[v1.MovementEvent],
    basis: v2.EndpointBasisV2,
    *,
    session: str,
    budget: int,
    k: int | str,
    shuffled: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    z, response = v2.event_arrays(events, basis)
    design = np.column_stack((np.ones(z.shape[0]), z))
    fit_manifest: dict[str, Any] = {
        "support_events": len(events),
        "design_rank": int(np.linalg.matrix_rank(design)),
        "design_condition_number": float(np.linalg.cond(design)),
        "shuffle_mode": None,
        "shuffle": None,
    }
    if shuffled:
        z, fit_manifest["shuffle_mode"], fit_manifest["shuffle"] = shuffled_latent(
            z, events, session=session, budget=budget, k=k,
        )
    carrier = v2.fit_carrier_arrays(z, response)
    fit_manifest["carrier_sha256"] = v1.array_sha256(carrier)
    return carrier, fit_manifest


def evaluate_session_at_k(
    session: v1.EventSession,
    basis: v2.EndpointBasisV2,
    k: int | str,
    *,
    reference_carrier: np.ndarray,
    budget: int = BUDGET_TRIALS,
) -> dict[str, Any]:
    support = v2.select_trial_range(session, start_index=0, budget=budget)
    later = tuple(event for event in session.events if event.trial_index >= budget)
    requested = _k_value(k, len(support))
    if len(support) < requested:
        return {
            "status": "undefined_insufficient_events",
            "requested_k": k,
            "support_events_available": len(support),
        }
    if len(later) < 4:
        return {
            "status": "undefined_event_count",
            "support_events": len(support),
            "later_events": len(later),
            "requested_k": k,
        }
    subset = support if k == "all" else support[:requested]
    try:
        correct, correct_fit = fit_support_carrier(
            subset, basis, session=session.session_name, budget=budget, k=k, shuffled=False,
        )
        shuffled_carrier, shuffled_fit = fit_support_carrier(
            subset, basis, session=session.session_name, budget=budget, k=k, shuffled=True,
        )
    except v1.SparseEventEndpointError as error:
        return {
            "status": "undefined_fit",
            "requested_k": k,
            "support_events": len(subset),
            "reason": str(error),
        }
    z_later, observed = v2.event_arrays(later, basis)
    support_mean = np.mean(np.stack([event.log_rates for event in subset]), axis=0)
    r_correct = v1.r2_by_channel(observed, v2.predict(correct, z_later))
    r_shuffle = v1.r2_by_channel(observed, v2.predict(shuffled_carrier, z_later))
    r_intercept = v1.r2_by_channel(observed, np.broadcast_to(support_mean, observed.shape))
    defined = np.isfinite(r_correct) & np.isfinite(r_shuffle) & np.isfinite(r_intercept)
    fidelity = median_carrier_weight_cosine(reference_carrier, correct)
    return {
        "status": "defined" if np.any(defined) else "undefined_channel_variance",
        "requested_k": k,
        "support_events": len(subset),
        "later_events": len(later),
        "defined_channels": int(np.sum(defined)),
        "design_rank": correct_fit["design_rank"],
        "design_condition_number": correct_fit["design_condition_number"],
        "shuffle_mode": shuffled_fit["shuffle_mode"],
        "correct_fit": correct_fit,
        "shuffled_fit": shuffled_fit,
        "median_r2_correct": float(np.median(r_correct[defined])) if np.any(defined) else None,
        "median_delta_shuffle": float(np.median((r_correct - r_shuffle)[defined])) if np.any(defined) else None,
        "median_delta_intercept": float(np.median((r_correct - r_intercept)[defined])) if np.any(defined) else None,
        "carrier_fidelity": fidelity,
    }


def reproduces_hse5(
    sessions: Mapping[str, v1.EventSession],
    bases: Mapping[str, v2.EndpointBasisV2],
    all_cells: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float = HSE5_TOLERANCE,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    shuffle_mode_mismatches: list[str] = []
    for name in v1.H1_HELDIN_SESSIONS:
        expected = v2.forward_transfer(sessions[name], bases[v1.session_date(name)], budget=BUDGET_TRIALS)
        observed = all_cells[name]
        v1._need(observed.get("status") == "defined", f"H-SE5 reproduction requires defined all cell for {name}")
        v1._need(expected.get("status") == "defined", f"H-SE5 forward transfer undefined for {name}")
        observed_mode = observed.get("shuffle_mode")
        expected_mode = "within_trial"
        compare_shuffle = observed_mode == expected_mode
        if not compare_shuffle:
            shuffle_mode_mismatches.append(name)
        fields = {
            "median_r2_correct": "median_r2_correct",
            "median_delta_intercept": "median_delta_intercept",
        }
        if compare_shuffle:
            fields["median_delta_shuffle"] = "median_delta_shuffle"
        for observed_key, expected_key in fields.items():
            delta = abs(float(observed[observed_key]) - float(expected[expected_key]))
            comparisons.append({
                "session": name,
                "field": observed_key,
                "absolute_difference": delta,
                "within_tolerance": delta <= tolerance,
            })
            v1._need(delta <= tolerance, f"H-SE5 reproduction failed for {name}.{observed_key}: {delta}")
    maximum = max(row["absolute_difference"] for row in comparisons)
    return {
        "passed": True,
        "tolerance": tolerance,
        "maximum_absolute_difference": maximum,
        "comparisons": comparisons,
        "shuffle_mode_mismatches": shuffle_mode_mismatches,
        "shuffle_fields_skipped": bool(shuffle_mode_mismatches),
    }


def run_screen(sessions: Mapping[str, v1.EventSession]) -> dict[str, Any]:
    v1._need(tuple(sessions) == v1.H1_HELDIN_SESSIONS, "event budget sweep session allowlist/order drift")
    bases = {date: v2.fit_source_all_event_basis(sessions, outer_date=date) for date in v1.H1_DATES}
    per_session: dict[str, dict[str, Any]] = {}
    reference_carriers: dict[str, np.ndarray] = {}
    for name in v1.H1_HELDIN_SESSIONS:
        session = sessions[name]
        basis = bases[session.date]
        support = v2.select_trial_range(session, start_index=0, budget=BUDGET_TRIALS)
        v1._need(len(support) >= 8, f"session {name} has fewer than eight M4 support events")
        reference_carrier, _ = fit_support_carrier(
            support, basis, session=name, budget=BUDGET_TRIALS, k="all", shuffled=False,
        )
        reference_carriers[name] = reference_carrier
        per_session[name] = {}
        for k in EVENT_COUNTS:
            per_session[name][str(k)] = evaluate_session_at_k(
                session, basis, k, reference_carrier=reference_carrier,
            )
    all_cells = {name: per_session[name]["all"] for name in v1.H1_HELDIN_SESSIONS}
    reproduction = reproduces_hse5(sessions, bases, all_cells)
    aggregates_by_k: dict[str, Any] = {}
    for k in EVENT_COUNTS:
        key = str(k)
        rows_intercept = []
        rows_shuffle = []
        rows_fidelity = []
        shuffle_mode_counts: Counter[str] = Counter()
        for name in v1.H1_HELDIN_SESSIONS:
            cell = per_session[name][key]
            if cell.get("status") == "defined":
                rows_intercept.append((name, cell.get("median_delta_intercept")))
                rows_shuffle.append((name, cell.get("median_delta_shuffle")))
                rows_fidelity.append((name, cell.get("carrier_fidelity")))
                shuffle_mode_counts[str(cell.get("shuffle_mode"))] += 1
        aggregates_by_k[key] = {
            "defined_sessions": sum(1 for name in v1.H1_HELDIN_SESSIONS if per_session[name][key].get("status") == "defined"),
            "median_delta_intercept": v1.paired_summary(rows_intercept),
            "median_delta_shuffle": v1.paired_summary(rows_shuffle),
            "carrier_fidelity": v1.paired_summary(rows_fidelity),
            "shuffle_mode_counts": dict(sorted(shuffle_mode_counts.items())),
        }
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "budget_trials": BUDGET_TRIALS,
        "event_counts": list(EVENT_COUNTS),
        "basis_by_outer_date": {date: bases[date].manifest() for date in v1.H1_DATES},
        "sessions": per_session,
        "aggregates_by_k": aggregates_by_k,
        "hse5_reproduction": reproduction,
        "scope": {
            "cuda_used": False,
            "decoder_constructed": False,
            "dense_velocity_series_opened": False,
            "public_held_in_calibration_nwbs_opened": 13,
        },
    }
