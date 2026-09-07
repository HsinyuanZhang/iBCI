"""H1 trial-aligned support/evaluation decomposition for the sealed H-SE5 carrier."""
from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2


SCHEMA = "h1_trial_aligned_sweep_v1"
PROTOCOL = "h1_trial_aligned_sweep_20260812_v1"
MODULE_NAME = "h1_trial_aligned_sweep"
SHUFFLE_NAMESPACE = "h1-trial-aligned-sweep-v1"
CELLS: tuple[tuple[int, int], ...] = ((1, 4), (2, 4), (3, 4), (4, 4), (3, 3))
SEALED_TOLERANCE = 1.0e-10
DECOMPOSITION_TOLERANCE = 1.0e-12
TOTAL_NEAR_ZERO = 1.0e-12
MIN_SUPPORT_EVENTS = 8
MIN_EVAL_EVENTS = 4


def cell_key(m: int, e: int) -> str:
    return f"{m}_{e}"


def support_events(session: v1.EventSession, m: int) -> tuple[v1.MovementEvent, ...]:
    return tuple(event for event in session.events if event.trial_index < m)


def eval_events(session: v1.EventSession, e: int) -> tuple[v1.MovementEvent, ...]:
    return tuple(event for event in session.events if event.trial_index >= e)


def shuffle_mode(events: Sequence[v1.MovementEvent]) -> str:
    counts = Counter(event.trial_index for event in events)
    if all(count >= 2 for count in counts.values()):
        return "within_trial"
    return "global_cyclic"


def global_cyclic_shuffle_order(
    event_count: int, *, session: str, m: int, e: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    v1._need(event_count >= 2, "global cyclic shuffle requires at least two events")
    key = f"{SHUFFLE_NAMESPACE}:{MODULE_NAME}:{session}:M{m}:E{e}"
    shift = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little") % (event_count - 1) + 1
    order = np.roll(np.arange(event_count, dtype=np.int64), shift)
    v1._need(not np.any(order == np.arange(event_count)), "global cyclic shuffle has fixed points")
    return order, {
        "namespace": SHUFFLE_NAMESPACE,
        "session": session,
        "m": m,
        "e": e,
        "shift": int(shift),
        "order_sha256": v1.array_sha256(order),
        "fixed_points": 0,
    }


def shuffled_latent(
    z: np.ndarray,
    events: Sequence[v1.MovementEvent],
    *,
    session: str,
    m: int,
    e: int,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    mode = shuffle_mode(events)
    if mode == "within_trial":
        order, manifest = v1.within_trial_label_shuffle(events, session=session, budget=m)
        return z[order], mode, manifest
    order, manifest = global_cyclic_shuffle_order(len(events), session=session, m=m, e=e)
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
    m: int,
    e: int,
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
            z, events, session=session, m=m, e=e,
        )
    carrier = v2.fit_carrier_arrays(z, response)
    fit_manifest["carrier_sha256"] = v1.array_sha256(carrier)
    return carrier, fit_manifest


def undefined_cell(
    status: str,
    *,
    support_count: int,
    eval_count: int,
    design_rank: int | None = None,
    design_condition_number: float | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "support_events": support_count,
        "eval_events": eval_count,
        "design_rank": design_rank,
        "design_condition_number": design_condition_number,
        "shuffle_mode": None,
        "median_r2_correct": None,
        "median_delta_intercept": None,
        "median_delta_shuffle": None,
        "carrier_fidelity": None,
    }


def evaluate_cell(
    session: v1.EventSession,
    basis: v2.EndpointBasisV2,
    m: int,
    e: int,
    *,
    reference_carrier: np.ndarray,
) -> dict[str, Any]:
    support = support_events(session, m)
    later = eval_events(session, e)
    support_count = len(support)
    eval_count = len(later)
    if support_count < MIN_SUPPORT_EVENTS:
        return undefined_cell(
            "undefined_insufficient_support_events",
            support_count=support_count,
            eval_count=eval_count,
        )
    if eval_count < MIN_EVAL_EVENTS:
        return undefined_cell(
            "undefined_insufficient_eval_events",
            support_count=support_count,
            eval_count=eval_count,
        )
    z_support, _ = v2.event_arrays(support, basis)
    design = np.column_stack((np.ones(z_support.shape[0]), z_support))
    design_rank = int(np.linalg.matrix_rank(design))
    design_condition_number = float(np.linalg.cond(design))
    if design_rank < v2.CARRIER_DIM:
        return undefined_cell(
            "undefined_rank_deficient",
            support_count=support_count,
            eval_count=eval_count,
            design_rank=design_rank,
            design_condition_number=design_condition_number,
        )
    try:
        correct, correct_fit = fit_support_carrier(
            support, basis, session=session.session_name, m=m, e=e, shuffled=False,
        )
        shuffled_carrier, shuffled_fit = fit_support_carrier(
            support, basis, session=session.session_name, m=m, e=e, shuffled=True,
        )
    except v1.SparseEventEndpointError as error:
        return undefined_cell(
            "undefined_fit",
            support_count=support_count,
            eval_count=eval_count,
            design_rank=design_rank,
            design_condition_number=design_condition_number,
        ) | {"reason": str(error)}
    z_later, observed = v2.event_arrays(later, basis)
    support_mean = np.mean(np.stack([event.log_rates for event in support]), axis=0)
    r_correct = v1.r2_by_channel(observed, v2.predict(correct, z_later))
    r_shuffle = v1.r2_by_channel(observed, v2.predict(shuffled_carrier, z_later))
    r_intercept = v1.r2_by_channel(observed, np.broadcast_to(support_mean, observed.shape))
    defined = np.isfinite(r_correct) & np.isfinite(r_shuffle) & np.isfinite(r_intercept)
    fidelity = median_carrier_weight_cosine(reference_carrier, correct)
    return {
        "status": "defined" if np.any(defined) else "undefined_channel_variance",
        "support_events": support_count,
        "eval_events": eval_count,
        "design_rank": correct_fit["design_rank"],
        "design_condition_number": correct_fit["design_condition_number"],
        "shuffle_mode": shuffled_fit["shuffle_mode"],
        "median_r2_correct": float(np.median(r_correct[defined])) if np.any(defined) else None,
        "median_delta_shuffle": float(np.median((r_correct - r_shuffle)[defined])) if np.any(defined) else None,
        "median_delta_intercept": float(np.median((r_correct - r_intercept)[defined])) if np.any(defined) else None,
        "carrier_fidelity": fidelity,
    }


def trial_contiguity_holds(session: v1.EventSession) -> bool:
    for m in (2, 3, 4):
        support = support_events(session, m)
        if support != session.events[: len(support)]:
            return False
    return True


def session_decomposition(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    c33 = cells[cell_key(3, 3)]
    c34 = cells[cell_key(3, 4)]
    c44 = cells[cell_key(4, 4)]
    v1._need(c33.get("status") == "defined", "decomposition requires defined Cell(3,3)")
    v1._need(c34.get("status") == "defined", "decomposition requires defined Cell(3,4)")
    v1._need(c44.get("status") == "defined", "decomposition requires defined Cell(4,4)")
    intercept_33 = float(c33["median_delta_intercept"])
    intercept_34 = float(c34["median_delta_intercept"])
    intercept_44 = float(c44["median_delta_intercept"])
    eval_effect = intercept_33 - intercept_34
    support_effect = intercept_34 - intercept_44
    total = intercept_33 - intercept_44
    residual = abs(total - (eval_effect + support_effect))
    v1._need(residual <= DECOMPOSITION_TOLERANCE, f"decomposition identity failed: residual={residual}")
    return {
        "eval_effect": eval_effect,
        "support_effect": support_effect,
        "total": total,
        "identity_residual": residual,
        "identity_holds": True,
    }


def attribution_fractions(
    total_summary: Mapping[str, Any],
    eval_summary: Mapping[str, Any],
    support_summary: Mapping[str, Any],
) -> dict[str, float | None]:
    total_mean = total_summary.get("mean")
    if total_mean is None or not np.isfinite(float(total_mean)) or abs(float(total_mean)) < TOTAL_NEAR_ZERO:
        return {"eval_fraction": None, "support_fraction": None}
    eval_mean = float(eval_summary["mean"])
    support_mean = float(support_summary["mean"])
    return {
        "eval_fraction": eval_mean / float(total_mean),
        "support_fraction": support_mean / float(total_mean),
    }


def reproduces_sealed_forward_transfer(
    sessions: Mapping[str, v1.EventSession],
    bases: Mapping[str, v2.EndpointBasisV2],
    per_session_cells: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float = SEALED_TOLERANCE,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for name in v1.H1_HELDIN_SESSIONS:
        basis = bases[v1.session_date(name)]
        session = sessions[name]
        for budget, key in ((3, cell_key(3, 3)), (4, cell_key(4, 4))):
            expected = v2.forward_transfer(session, basis, budget=budget)
            observed = per_session_cells[name][key]
            v1._need(observed.get("status") == "defined", f"sealed reproduction requires defined {key} for {name}")
            v1._need(expected.get("status") == "defined", f"forward_transfer undefined for {name} budget={budget}")
            for field in ("median_r2_correct", "median_delta_intercept", "median_delta_shuffle"):
                delta = abs(float(observed[field]) - float(expected[field]))
                comparisons.append({
                    "session": name,
                    "cell": key,
                    "field": field,
                    "absolute_difference": delta,
                    "within_tolerance": delta <= tolerance,
                })
                v1._need(delta <= tolerance, f"sealed reproduction failed for {name}.{key}.{field}: {delta}")
    maximum = max(row["absolute_difference"] for row in comparisons)
    return {
        "passed": True,
        "tolerance": tolerance,
        "maximum_absolute_difference": maximum,
        "comparisons": comparisons,
    }


def aggregate_cells(per_session: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for m, e in CELLS:
        key = cell_key(m, e)
        rows_intercept: list[tuple[str, float | None]] = []
        rows_fidelity: list[tuple[str, float | None]] = []
        shuffle_mode_counts: Counter[str] = Counter()
        for name in v1.H1_HELDIN_SESSIONS:
            cell = per_session[name][key]
            if cell.get("status") == "defined":
                rows_intercept.append((name, cell.get("median_delta_intercept")))
                rows_fidelity.append((name, cell.get("carrier_fidelity")))
                shuffle_mode_counts[str(cell.get("shuffle_mode"))] += 1
        aggregates[key] = {
            "m": m,
            "e": e,
            "defined_sessions": sum(
                1 for name in v1.H1_HELDIN_SESSIONS if per_session[name][key].get("status") == "defined"
            ),
            "median_delta_intercept": v1.paired_summary(rows_intercept),
            "carrier_fidelity": v1.paired_summary(rows_fidelity),
            "shuffle_mode_counts": dict(sorted(shuffle_mode_counts.items())),
        }
    return aggregates


def run_screen(sessions: Mapping[str, v1.EventSession]) -> dict[str, Any]:
    v1._need(tuple(sessions) == v1.H1_HELDIN_SESSIONS, "trial aligned sweep session allowlist/order drift")
    bases = {date: v2.fit_source_all_event_basis(sessions, outer_date=date) for date in v1.H1_DATES}
    per_session: dict[str, dict[str, Any]] = {}
    reference_carriers: dict[str, np.ndarray] = {}
    trial_contiguity: dict[str, bool] = {}
    decompositions: dict[str, Any] = {}
    for name in v1.H1_HELDIN_SESSIONS:
        session = sessions[name]
        basis = bases[session.date]
        trial_contiguity[name] = trial_contiguity_holds(session)
        reference_carrier, _ = fit_support_carrier(
            support_events(session, 4),
            basis,
            session=name,
            m=4,
            e=4,
            shuffled=False,
        )
        reference_carriers[name] = reference_carrier
        per_session[name] = {}
        for m, e in CELLS:
            per_session[name][cell_key(m, e)] = evaluate_cell(
                session, basis, m, e, reference_carrier=reference_carrier,
            )
        decompositions[name] = session_decomposition(per_session[name])
    sealed_reproduction = reproduces_sealed_forward_transfer(sessions, bases, per_session)
    decomposition_rows = {
        "eval_effect": [(name, decompositions[name]["eval_effect"]) for name in v1.H1_HELDIN_SESSIONS],
        "support_effect": [(name, decompositions[name]["support_effect"]) for name in v1.H1_HELDIN_SESSIONS],
        "total": [(name, decompositions[name]["total"]) for name in v1.H1_HELDIN_SESSIONS],
    }
    aggregated_decomposition = {
        key: v1.paired_summary(rows) for key, rows in decomposition_rows.items()
    }
    attribution = attribution_fractions(
        aggregated_decomposition["total"],
        aggregated_decomposition["eval_effect"],
        aggregated_decomposition["support_effect"],
    )
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "cells": [{"m": m, "e": e, "key": cell_key(m, e)} for m, e in CELLS],
        "basis_by_outer_date": {date: bases[date].manifest() for date in v1.H1_DATES},
        "sessions": per_session,
        "aggregates_by_cell": aggregate_cells(per_session),
        "decomposition": {
            "per_session": decompositions,
            "aggregated": aggregated_decomposition,
            "attribution_fractions": attribution,
            "identity_tolerance": DECOMPOSITION_TOLERANCE,
        },
        "integrity_checks": {
            "sealed_forward_transfer_reproduction": sealed_reproduction,
            "trial_contiguity": {
                "per_session": trial_contiguity,
                "all_sessions": all(trial_contiguity.values()),
            },
        },
        "scope": {
            "cuda_used": False,
            "decoder_constructed": False,
            "dense_velocity_series_opened": False,
            "public_held_in_calibration_nwbs_opened": 13,
        },
    }
