"""V2 low-rank H1 endpoint carrier: q=4 plus intercept, ridge lambda=3."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1


SCHEMA = "h1_sparse_event_endpoint_v2_source_audit_v1"
PROTOCOL = "h1_sparse_event_endpoint_q4_ridge3_20260811_v2"
LATENT_DIM = 4
CARRIER_DIM = 5
RIDGE_LAMBDA = 3.0
RETAINED_MINIMUM = 0.65
RETAINED_MEDIAN = 0.70
POSITIVE_SESSIONS = 10


@dataclass(frozen=True)
class EndpointBasisV2:
    outer_date: str
    source_sessions: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    score_scale: np.ndarray
    explained_variance_ratio: np.ndarray
    retained_variance: float
    source_event_count: int
    basis_sha256: str

    def transform(self, displacement: np.ndarray) -> np.ndarray:
        values = np.asarray(displacement, dtype=np.float64)
        v1._need(values.ndim == 2 and values.shape[1] == v1.POSITION_DIM, f"V2 endpoint matrix shape {values.shape}")
        return ((values - self.mean[None, :]) / self.scale[None, :]) @ self.components.T / self.score_scale[None, :]

    def manifest(self) -> dict[str, Any]:
        return {
            "outer_date": self.outer_date,
            "source_sessions": list(self.source_sessions),
            "source_event_count": self.source_event_count,
            "retained_variance": self.retained_variance,
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "array_sha256": {
                "mean": v1.array_sha256(self.mean),
                "scale": v1.array_sha256(self.scale),
                "components": v1.array_sha256(self.components),
                "score_scale": v1.array_sha256(self.score_scale),
            },
            "basis_sha256": self.basis_sha256,
        }


def fit_source_all_event_basis(
    sessions: Mapping[str, v1.EventSession], *, outer_date: str,
) -> EndpointBasisV2:
    v1._need(outer_date in v1.H1_DATES, "invalid V2 outer date")
    source_names = tuple(name for name in v1.H1_HELDIN_SESSIONS if v1.session_date(name) != outer_date)
    v1._need(set(source_names).issubset(sessions), "V2 basis is missing source sessions")
    pooled = np.stack([
        event.displacement for name in source_names for event in sessions[name].events
    ]).astype(np.float64)
    mean = pooled.mean(axis=0)
    scale = np.maximum(pooled.std(axis=0), v1.SCALE_FLOOR)
    standardized = (pooled - mean[None, :]) / scale[None, :]
    _u, singular, right = np.linalg.svd(standardized, full_matrices=False)
    v1._need(np.linalg.matrix_rank(standardized) == v1.POSITION_DIM, "V2 source endpoints are rank deficient")
    components = v1._canonicalize_component_signs(right[:LATENT_DIM])
    score_scale = np.maximum((standardized @ components.T).std(axis=0), v1.SCALE_FLOOR)
    ratio = np.square(singular) / np.square(singular).sum()
    body = {
        "protocol": PROTOCOL,
        "outer_date": outer_date,
        "source_sessions": list(source_names),
        "source_event_count": int(pooled.shape[0]),
        "mean": v1.array_sha256(mean),
        "scale": v1.array_sha256(scale),
        "components": v1.array_sha256(components),
        "score_scale": v1.array_sha256(score_scale),
    }
    return EndpointBasisV2(
        outer_date=outer_date,
        source_sessions=source_names,
        mean=np.asarray(mean, np.float64),
        scale=np.asarray(scale, np.float64),
        components=np.asarray(components, np.float64),
        score_scale=np.asarray(score_scale, np.float64),
        explained_variance_ratio=np.asarray(ratio, np.float64),
        retained_variance=float(ratio[:LATENT_DIM].sum()),
        source_event_count=int(pooled.shape[0]),
        basis_sha256=v1.canonical_sha256(body),
    )


def select_trial_range(
    session: v1.EventSession, *, start_index: int, budget: int,
) -> tuple[v1.MovementEvent, ...]:
    v1._need(budget in v1.SUPPORT_BUDGETS and 0 <= start_index <= len(session.trial_values) - budget,
             "invalid V2 support trial range")
    return tuple(event for event in session.events if start_index <= event.trial_index < start_index + budget)


def event_arrays(
    events: Sequence[v1.MovementEvent], basis: EndpointBasisV2,
) -> tuple[np.ndarray, np.ndarray]:
    v1._need(bool(events), "V2 event array is empty")
    displacement = np.stack([event.displacement for event in events]).astype(np.float64)
    response = np.stack([event.log_rates for event in events]).astype(np.float64)
    return basis.transform(displacement), response


def fit_carrier_arrays(z: np.ndarray, response: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    v1._need(z.ndim == 2 and z.shape[1] == LATENT_DIM and z.shape[0] >= 8, f"V2 latent design shape {z.shape}")
    v1._need(response.shape == (z.shape[0], v1.EXPECTED_NEURONS), f"V2 response shape {response.shape}")
    design = np.column_stack((np.ones(z.shape[0]), z))
    v1._need(np.linalg.matrix_rank(design) == CARRIER_DIM, "V2 support design is rank deficient")
    penalty = np.diag([0.0] + [1.0] * LATENT_DIM) * (z.shape[0] * RIDGE_LAMBDA)
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ response)
    v1._need(np.isfinite(coefficient).all(), "V2 coefficient is nonfinite")
    return np.column_stack((coefficient[1:].T, coefficient[0]))


def fit_session_range(
    session: v1.EventSession,
    basis: EndpointBasisV2,
    *,
    start_index: int,
    budget: int,
    shuffled_labels: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    events = select_trial_range(session, start_index=start_index, budget=budget)
    z, response = event_arrays(events, basis)
    shuffle = None
    if shuffled_labels:
        # Rebase trial indices solely for the deterministic namespace. The V1
        # shuffler groups by actual trial index and remains fixed-point-free.
        order, shuffle = v1.within_trial_label_shuffle(
            events, session=session.session_name, budget=budget,
        )
        z = z[order]
    carrier = fit_carrier_arrays(z, response)
    return carrier, {
        "start_index": start_index,
        "budget": budget,
        "support_events": len(events),
        "design_rank": int(np.linalg.matrix_rank(np.column_stack((np.ones(len(z)), z)))),
        "carrier_sha256": v1.array_sha256(carrier),
        "shuffle": shuffle,
    }


def predict(carrier: np.ndarray, z: np.ndarray) -> np.ndarray:
    values = np.asarray(carrier, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    v1._need(values.shape == (v1.EXPECTED_NEURONS, CARRIER_DIM), f"V2 carrier shape {values.shape}")
    return z @ values[:, :LATENT_DIM].T + values[:, LATENT_DIM][None, :]


def row_shuffle(carrier: np.ndarray, *, session: str, budget: int) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(carrier, dtype=np.float64)
    v1._need(values.shape == (v1.EXPECTED_NEURONS, CARRIER_DIM), "V2 row shuffle expects [176,5]")
    key = f"{v1.SHUFFLE_NAMESPACE}:v2-row:{session}:M{budget}"
    seed = int.from_bytes(__import__("hashlib").sha256(key.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    identity = np.arange(v1.EXPECTED_NEURONS)
    order = identity.copy()
    for _ in range(100):
        rng.shuffle(order)
        if not np.any(order == identity):
            break
    v1._need(not np.any(order == identity), "V2 row shuffle has fixed points")
    return values[order], {"order_sha256": v1.array_sha256(order), "fixed_points": 0}


def forward_transfer(
    session: v1.EventSession, basis: EndpointBasisV2, *, budget: int,
) -> dict[str, Any]:
    support = select_trial_range(session, start_index=0, budget=budget)
    later = tuple(event for event in session.events if event.trial_index >= budget)
    if len(support) < 8 or len(later) < 4:
        return {"status": "undefined_event_count", "support_events": len(support), "later_events": len(later)}
    correct, correct_fit = fit_session_range(session, basis, start_index=0, budget=budget)
    shuffled, shuffled_fit = fit_session_range(
        session, basis, start_index=0, budget=budget, shuffled_labels=True,
    )
    z, observed = event_arrays(later, basis)
    support_mean = np.mean(np.stack([event.log_rates for event in support]), axis=0)
    r_correct = v1.r2_by_channel(observed, predict(correct, z))
    r_shuffle = v1.r2_by_channel(observed, predict(shuffled, z))
    r_intercept = v1.r2_by_channel(observed, np.broadcast_to(support_mean, observed.shape))
    defined = np.isfinite(r_correct) & np.isfinite(r_shuffle) & np.isfinite(r_intercept)
    return {
        "status": "defined" if np.any(defined) else "undefined_channel_variance",
        "support_events": len(support),
        "later_events": len(later),
        "defined_channels": int(np.sum(defined)),
        "correct_fit": correct_fit,
        "shuffled_fit": shuffled_fit,
        "median_r2_correct": float(np.median(r_correct[defined])) if np.any(defined) else None,
        "median_delta_shuffle": float(np.median((r_correct - r_shuffle)[defined])) if np.any(defined) else None,
        "median_delta_intercept": float(np.median((r_correct - r_intercept)[defined])) if np.any(defined) else None,
    }


def evaluate_gate(
    *, session_rows: Mapping[str, Mapping[str, Any]], basis_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    defined = len(session_rows) == 13 and all(
        row["support_events"] >= 8 and row["design_rank"] == CARRIER_DIM
        and row["forward"]["status"] == "defined"
        for row in session_rows.values()
    )
    retained = np.asarray([float(row["retained_variance"]) for row in basis_rows])
    basis_gate = retained.size == len(v1.H1_DATES) and np.min(retained) >= RETAINED_MINIMUM and np.median(retained) >= RETAINED_MEDIAN
    shuffle = v1.paired_summary([(name, row["forward"].get("median_delta_shuffle")) for name, row in session_rows.items()])
    intercept = v1.paired_summary([(name, row["forward"].get("median_delta_intercept")) for name, row in session_rows.items()])
    def functional(summary: Mapping[str, Any]) -> bool:
        return bool(
            summary["defined_sessions"] == 13 and summary["mean"] > 0 and summary["median"] > 0
            and summary["positive"] >= POSITIVE_SESSIONS
            and summary["leave_largest_absolute_out_mean"] > 0
        )
    passed = bool(defined and basis_gate and functional(shuffle) and functional(intercept))
    return {
        "passed": passed,
        "defined_carriers_all_13": defined,
        "basis_retained_variance_gate": bool(basis_gate),
        "correct_minus_shuffle_gate": functional(shuffle),
        "correct_minus_intercept_gate": functional(intercept),
        "basis_retained_variance": {"minimum": float(np.min(retained)), "median": float(np.median(retained))},
        "correct_minus_shuffle": shuffle,
        "correct_minus_intercept": intercept,
    }

