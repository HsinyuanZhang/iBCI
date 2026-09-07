"""CPU-only H1 NLE5: a source-learned nonlinear sparse-event label embedding.

The deployable target object is strictly one closed-form ``[w1,w2,w3,w4,b]``
row per channel.  Its four covariates are a frozen source-only RFF/ridge/SVD
embedding of sparse event annotations (tag, start time, endpoint, duration).
No query label or target population statistic is used in fitting the map.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as hse5

SCHEMA = "h1_event_carrier_nle5_source_screen_v1"
PROTOCOL = "h1_event_carrier_nonlinear_label_embedding_20260812_v1"
RANK, CARRIER_DIM = 4, 5
SUPPORT_BUDGETS = (3, 4)
RFF_WIDTH, RFF_SEED, SOURCE_RIDGE_LAMBDA, TARGET_RIDGE_LAMBDA = 16, 190271, 1.0, 3.0
MATERIAL_MEAN, MATERIAL_MEDIAN, MINIMUM_POSITIVE = .02, .01, 10
SCALE_FLOOR = 1.e-8


def _need(value: bool, message: str) -> None:
    v1._need(value, message)


def source_names_for_outer(outer_date: str) -> tuple[str, ...]:
    _need(outer_date in v1.H1_DATES, "NLE5 unknown outer date")
    return tuple(n for n in v1.H1_HELDIN_SESSIONS if v1.session_date(n) != outer_date)


def _raw(events: Sequence[v1.MovementEvent], tags: Sequence[str] | None = None) -> np.ndarray:
    """Sparse-only annotation rows; the continuous start time is source normalized."""
    _need(bool(events), "NLE5 empty event rows")
    chosen = tuple(e.tag for e in events) if tags is None else tuple(tags)
    _need(len(chosen) == len(events) and set(chosen).issubset(v1.MOVEMENT_TAGS), "NLE5 tag rows drift")
    endpoint = np.stack([e.displacement for e in events]).astype(np.float64)
    start = np.asarray([e.start_time for e in events], dtype=np.float64)[:, None]
    duration = np.log(np.asarray([e.duration_seconds for e in events], dtype=np.float64))[:, None]
    onehot = np.asarray([[float(t == candidate) for candidate in v1.MOVEMENT_TAGS] for t in chosen], dtype=np.float64)
    output = np.column_stack((endpoint, start, duration, onehot))
    _need(output.shape == (len(events), 17) and np.isfinite(output).all(), "NLE5 raw annotation shape")
    return output


def _canonical_columns(value: np.ndarray) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64).copy()
    for col in range(out.shape[1]):
        pivot = int(np.argmax(np.abs(out[:, col])))
        if out[pivot, col] < 0:
            out[:, col] *= -1
    return out


@dataclass(frozen=True)
class NonlinearMap:
    outer_date: str
    source_sessions: tuple[str, ...]
    raw_mean: np.ndarray
    raw_scale: np.ndarray
    rff_weight: np.ndarray
    rff_phase: np.ndarray
    projection: np.ndarray
    latent_scale: np.ndarray
    source_event_count: int
    map_sha256: str

    def _rff(self, events: Sequence[v1.MovementEvent], tags: Sequence[str] | None = None) -> np.ndarray:
        x = (_raw(events, tags) - self.raw_mean) / self.raw_scale
        value = np.sqrt(2.0 / RFF_WIDTH) * np.cos(x @ self.rff_weight + self.rff_phase)
        _need(value.shape == (len(events), RFF_WIDTH) and np.isfinite(value).all(), "NLE5 RFF drift")
        return value

    def transform(self, events: Sequence[v1.MovementEvent], tags: Sequence[str] | None = None) -> np.ndarray:
        value = self._rff(events, tags) @ self.projection / self.latent_scale
        _need(value.shape == (len(events), RANK) and np.isfinite(value).all(), "NLE5 latent drift")
        return value

    def manifest(self) -> dict[str, Any]:
        return {"outer_date": self.outer_date, "source_sessions": list(self.source_sessions),
                "source_event_count": self.source_event_count, "input_annotation_order":
                ["delta_tx_ty_tz_rx_g1_g2_g3", "start_time", "log_duration", "tag_onehot:" + ",".join(v1.MOVEMENT_TAGS)],
                "rff_width": RFF_WIDTH, "rff_seed": RFF_SEED, "output_rank": RANK,
                "array_sha256": {k: v1.array_sha256(getattr(self, k)) for k in
                ("raw_mean", "raw_scale", "rff_weight", "rff_phase", "projection", "latent_scale")},
                "map_sha256": self.map_sha256}


def fit_nonlinear_map(sessions: Mapping[str, v1.EventSession], *, outer_date: str,
                      source_names: Sequence[str] | None = None) -> NonlinearMap:
    names = tuple(source_names or source_names_for_outer(outer_date))
    _need(names and len(names) == len(set(names)) and set(names).issubset(sessions), "NLE5 source allowlist invalid")
    _need(all(v1.session_date(n) != outer_date for n in names), "NLE5 outer date entered map")
    events = tuple(e for n in names for e in sessions[n].events)
    raw = _raw(events); raw_mean = raw.mean(0); raw_scale = np.maximum(raw.std(0), SCALE_FLOOR)
    # Constant absent source tags are explicitly rejected: a tag embedding must be identified.
    _need(np.all(raw_scale[-len(v1.MOVEMENT_TAGS):] > SCALE_FLOOR), "NLE5 source tag support deficient")
    rng = np.random.default_rng(RFF_SEED)
    weight = rng.normal(0., 1., size=(raw.shape[1], RFF_WIDTH)); phase = rng.uniform(0., 2*np.pi, size=RFF_WIDTH)
    rff = np.sqrt(2. / RFF_WIDTH) * np.cos(((raw - raw_mean) / raw_scale) @ weight + phase)
    y = np.stack([e.log_rates for e in events]).astype(np.float64)
    y_scale = np.maximum(y.std(0), SCALE_FLOOR)
    coefficient = np.linalg.solve(rff.T @ rff + len(rff) * SOURCE_RIDGE_LAMBDA * np.eye(RFF_WIDTH),
                                  rff.T @ ((y - y.mean(0)) / y_scale))
    left, _singular, _right = np.linalg.svd(coefficient, full_matrices=False)
    projection = _canonical_columns(left[:, :RANK]); latent_scale = np.maximum((rff @ projection).std(0), SCALE_FLOOR)
    body = {"protocol": PROTOCOL, "outer_date": outer_date, "source_sessions": list(names),
            "raw_mean": v1.array_sha256(raw_mean), "raw_scale": v1.array_sha256(raw_scale),
            "rff_weight": v1.array_sha256(weight), "rff_phase": v1.array_sha256(phase),
            "projection": v1.array_sha256(projection), "latent_scale": v1.array_sha256(latent_scale)}
    return NonlinearMap(outer_date, names, raw_mean, raw_scale, weight, phase, projection, latent_scale,
                        len(events), v1.canonical_sha256(body))


def fit_carrier(z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    z, y = np.asarray(z, np.float64), np.asarray(y, np.float64)
    _need(z.ndim == 2 and z.shape[1] == RANK and z.shape[0] >= CARRIER_DIM, "NLE5 target design shape")
    _need(y.shape == (len(z), v1.EXPECTED_NEURONS) and np.isfinite(y).all(), "NLE5 target response shape")
    x = np.c_[np.ones(len(z)), z]
    _need(np.linalg.matrix_rank(x) == CARRIER_DIM, "NLE5 target design rank deficient")
    penalty = np.diag([0.] + [1.] * RANK) * (len(z) * TARGET_RIDGE_LAMBDA)
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    carrier = np.c_[beta[1:].T, beta[0]]
    _need(carrier.shape == (v1.EXPECTED_NEURONS, CARRIER_DIM) and np.isfinite(carrier).all(), "NLE5 carrier drift")
    return carrier, {"target_optimizer_steps": 0, "target_backward_steps": 0, "target_fit": "closed_form_ridge",
                     "design_rank": int(np.linalg.matrix_rank(x)), "carrier_sha256": v1.array_sha256(carrier)}


def predict(carrier: np.ndarray, z: np.ndarray) -> np.ndarray:
    _need(np.asarray(carrier).shape == (v1.EXPECTED_NEURONS, CARRIER_DIM), "NLE5 carrier width")
    return np.asarray(z, np.float64) @ np.asarray(carrier, np.float64)[:, :RANK].T + carrier[:, RANK]


def _fixed_permutation(size: int, namespace: str) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "little"))
    order = np.arange(size)
    for _ in range(100):
        rng.shuffle(order)
        if not np.any(order == np.arange(size)): break
    _need(not np.any(order == np.arange(size)), "NLE5 permutation fixed point")
    return order, {"namespace": namespace, "fixed_points": 0, "order_sha256": v1.array_sha256(order)}


def _rotation() -> np.ndarray:
    # fixed orthogonal mixing; transformation must preserve fitted predictions exactly
    q, r = np.linalg.qr(np.arange(1., 17.).reshape(4, 4) + np.eye(4) * .37)
    return q @ np.diag(np.sign(np.diag(r)))


def _arrays(session: v1.EventSession, mapping: NonlinearMap, budget: int) -> tuple[tuple[v1.MovementEvent, ...], tuple[v1.MovementEvent, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support, later = session.events_before(budget), session.events_after(budget)
    _need(len(support) >= CARRIER_DIM and len(later) >= 4, "NLE5 insufficient support/query")
    return support, later, mapping.transform(support), np.stack([e.log_rates for e in support]), mapping.transform(later), np.stack([e.log_rates for e in later])


def evaluate_session(session: v1.EventSession, mapping: NonlinearMap, *, budget: int) -> dict[str, Any]:
    _need(mapping.outer_date == session.date, "NLE5 outer map/session mismatch")
    support, later, z, y, zl, yl = _arrays(session, mapping, budget)
    correct, fit = fit_carrier(z, y)
    label_order, label_manifest = _fixed_permutation(len(z), f"{PROTOCOL}:endpoint-label:{session.session_name}:M{budget}")
    label, label_fit = fit_carrier(z[label_order], y)
    tag_order, tag_manifest = _fixed_permutation(len(z), f"{PROTOCOL}:tag:{session.session_name}:M{budget}")
    tag_z = mapping.transform(support, [e.tag for e in np.asarray(support, object)[tag_order]])
    tag, tag_fit = fit_carrier(tag_z, y)
    row_order, row_manifest = _fixed_permutation(v1.EXPECTED_NEURONS, f"{PROTOCOL}:row:{session.session_name}:M{budget}")
    row, row_fit = fit_carrier(z, y); row = row[row_order]
    inter = np.c_[np.zeros((v1.EXPECTED_NEURONS, RANK)), y.mean(0)]
    rotation = _rotation(); rotated, _ = fit_carrier(z @ rotation, y)
    rotation_error = float(np.max(np.abs(predict(correct, zl) - predict(rotated, zl @ rotation))))
    _need(rotation_error <= 1.e-11, "NLE5 coordinate rotation invariance failed")
    scores = {"correct": v1.r2_by_channel(yl, predict(correct, zl)), "label": v1.r2_by_channel(yl, predict(label, zl)),
              "tag": v1.r2_by_channel(yl, predict(tag, zl)), "row": v1.r2_by_channel(yl, predict(row, zl)),
              "intercept": v1.r2_by_channel(yl, predict(inter, zl))}
    defined = np.logical_and.reduce([np.isfinite(x) for x in scores.values()]); _need(np.any(defined), "NLE5 no finite channels")
    med = lambda x: float(np.median(x[defined]))
    return {"session": session.session_name, "date": session.date, "budget": budget, "support_events": len(support), "later_events": len(later),
            "carrier_dim": CARRIER_DIM, "defined_channels": int(defined.sum()), "target_design_rank": fit["design_rank"],
            "median_r2_correct": med(scores["correct"]), "median_r2_label_shuffle": med(scores["label"]), "median_r2_tag_shuffle": med(scores["tag"]),
            "median_r2_row_shuffle": med(scores["row"]), "median_r2_intercept": med(scores["intercept"]),
            "median_delta_label_shuffle": med(scores["correct"]-scores["label"]), "median_delta_tag_shuffle": med(scores["correct"]-scores["tag"]),
            "median_delta_row_shuffle": med(scores["correct"]-scores["row"]), "median_delta_intercept": med(scores["correct"]-scores["intercept"]),
            "rotation_invariance_max_abs_error": rotation_error, "rotation_sha256": v1.array_sha256(rotation),
            "correct_fit": fit, "label_fit": label_fit, "tag_fit": tag_fit, "row_fit": row_fit,
            "correct_carrier_sha256": v1.array_sha256(correct), "label_carrier_sha256": v1.array_sha256(label),
            "tag_carrier_sha256": v1.array_sha256(tag), "row_carrier_sha256": v1.array_sha256(row),
            "label_shuffle": label_manifest, "tag_shuffle": tag_manifest, "row_shuffle": row_manifest,
            "outer_future_used_only_for_scoring": True}


def hse5_baseline_session(session: v1.EventSession, basis: hse5.EndpointBasisV2, *, budget: int) -> dict[str, Any]:
    support, later = session.events_before(budget), session.events_after(budget)
    zs, ys = hse5.event_arrays(support, basis); zl, yl = hse5.event_arrays(later, basis)
    carrier = hse5.fit_carrier_arrays(zs, ys)
    order, _ = v1.within_trial_label_shuffle(support, session=session.session_name, budget=budget)
    label = hse5.fit_carrier_arrays(zs[order], ys)
    rc, rl = v1.r2_by_channel(yl, hse5.predict(carrier, zl)), v1.r2_by_channel(yl, hse5.predict(label, zl))
    ri = v1.r2_by_channel(yl, np.broadcast_to(ys.mean(0), yl.shape)); good = np.isfinite(rc)&np.isfinite(rl)&np.isfinite(ri)
    return {"median_r2_correct": float(np.median(rc[good])), "median_delta_label_shuffle": float(np.median((rc-rl)[good])),
            "median_delta_intercept": float(np.median((rc-ri)[good])), "carrier_sha256": v1.array_sha256(carrier)}


def _summary(rows: Mapping[str, Mapping[str, Any]], field: str) -> dict[str, Any]:
    return v1.paired_summary([(n, row[field]) for n, row in rows.items()])


def _positive(value: Mapping[str, Any]) -> bool:
    return bool(value["defined_sessions"] == 13 and value["mean"] > 0 and value["median"] > 0 and value["positive"] >= MINIMUM_POSITIVE and value["leave_largest_absolute_out_mean"] > 0)


def aggregate(rows: Mapping[str, Mapping[str, Any]], hse: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    merged = {n: {**row, "delta_vs_hse5": row["median_r2_correct"] - hse[n]["median_r2_correct"]} for n,row in rows.items()}
    return {"correct_r2": _summary(merged, "median_r2_correct"), "correct_minus_hse5": _summary(merged, "delta_vs_hse5"),
            "correct_minus_label_shuffle": _summary(merged, "median_delta_label_shuffle"), "correct_minus_tag_shuffle": _summary(merged, "median_delta_tag_shuffle"),
            "correct_minus_row_shuffle": _summary(merged, "median_delta_row_shuffle"), "correct_minus_intercept": _summary(merged, "median_delta_intercept")}


def gate(a: Mapping[str, Any]) -> dict[str, Any]:
    material = _positive(a["correct_minus_hse5"]) and a["correct_minus_hse5"]["mean"] >= MATERIAL_MEAN and a["correct_minus_hse5"]["median"] >= MATERIAL_MEDIAN
    controls = {f: _positive(a[f]) for f in ("correct_minus_label_shuffle", "correct_minus_tag_shuffle", "correct_minus_row_shuffle", "correct_minus_intercept")}
    return {"passed": bool(material and all(controls.values())), "material_gain_vs_hse5": bool(material), **controls,
            "thresholds": {"mean_delta_vs_hse5": MATERIAL_MEAN, "median_delta_vs_hse5": MATERIAL_MEDIAN, "minimum_positive_sessions": MINIMUM_POSITIVE, "leave_largest_absolute_out_mean_positive": True}}


def run_screen(sessions: Mapping[str, v1.EventSession]) -> dict[str, Any]:
    _need(tuple(sessions) == v1.H1_HELDIN_SESSIONS, "NLE5 session allowlist/order drift")
    maps = {d: fit_nonlinear_map(sessions, outer_date=d) for d in v1.H1_DATES}
    bases = {d: hse5.fit_source_all_event_basis(sessions, outer_date=d) for d in v1.H1_DATES}
    budgets: dict[str, Any] = {}
    for budget in SUPPORT_BUDGETS:
        rows, baseline = {}, {}
        for name in v1.H1_HELDIN_SESSIONS:
            date = v1.session_date(name); rows[name] = evaluate_session(sessions[name], maps[date], budget=budget)
            baseline[name] = hse5_baseline_session(sessions[name], bases[date], budget=budget)
        total = aggregate(rows, baseline)
        budgets[f"M{budget}"] = {"budget_trials": budget, "nonlinear_map_by_outer_date": {d: maps[d].manifest() for d in v1.H1_DATES},
                                   "hse5_basis_by_outer_date": {d: bases[d].manifest() for d in v1.H1_DATES}, "hse5_baseline_sessions": baseline,
                                   "nle5_sessions": rows, "aggregate": total, "gate": gate(total)}
    passed = all(budgets[f"M{x}"]["gate"]["passed"] for x in SUPPORT_BUDGETS)
    return {"schema": SCHEMA, "protocol": PROTOCOL, "candidate_matrix_predeclared_before_data_run": True,
            "frozen_constants": {"carrier_dim": CARRIER_DIM, "embedding_dim": RANK, "support_budgets": list(SUPPORT_BUDGETS), "rff_width": RFF_WIDTH,
            "rff_seed": RFF_SEED, "source_ridge_lambda": SOURCE_RIDGE_LAMBDA, "target_ridge_lambda": TARGET_RIDGE_LAMBDA,
            "single_model_no_outer_expanded_grid": True, "selection": "none; one predeclared model"}, "budgets": budgets,
            "status": "PASS_CPU_NLE5_MATERIAL" if passed else "STOP_CPU_NLE5_NOT_MATERIAL", "gpu_authorized_by_this_screen": False,
            "scope": {"public_held_in_calibration_nwbs_opened": 13, "minival_nwbs_opened": 0, "held_out_nwbs_opened": 0, "formal_test_labels_opened": 0,
            "dense_velocity_opened": False, "within_event_position_trajectory_opened": False, "target_session_optimizer_steps": 0, "target_session_backward_steps": 0,
            "decoder_constructed": False, "trainer_constructed": False, "cuda_used": False},
            "interpretation": {"source_only_nonlinear_map": True, "target_deployable_object_is_support_only_closed_form_5d_carrier": True,
            "outer_date_excluded_from_map_training": True, "query_labels_used_only_for_scoring": True}}
