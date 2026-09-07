"""QC2F5: fixed degree-two source support-to-future correction for H1.

This CPU-only candidate preserves the exact C2F5 source examples, H-SE5
carrier, quality statistics, outer-date LODO, and raw later-event log-rate
scorer.  Its sole new degree of freedom is a channel-shared polynomial ridge
map, applied to standardized support/quality inputs:

``B_corrected = B_support + F2([B_support, quality])``.

No target future value is fitted, no channel identity is supplied, and no
optimizer/backward pass is used on a target.  The matching controls include a
separately trained source future-teacher pairing shuffle that preserves the
teacher marginal while destroying support-to-teacher pairing.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from sua_exploration.mc_maze import h1_calibration_future_correction_c2f5 as linear
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2


SCHEMA = "h1_calibration_future_quadratic_c2f5_source_screen_v1"
PROTOCOL = "h1_qc2f5_fixed_degree2_residual_operator_source_lodo_20260812_v1"
PREDECLARATION_SCHEMA = "h1_qc2f5_predeclaration_v1"
RANK = linear.RANK
CARRIER_DIM = linear.CARRIER_DIM
QUALITY_NAMES = linear.QUALITY_NAMES
QUALITY_DIM = linear.QUALITY_DIM
SUPPORT_BUDGETS = linear.SUPPORT_BUDGETS
C2F_RIDGE_GRID = linear.C2F_RIDGE_GRID
SELECTION_TIE_TOLERANCE = linear.SELECTION_TIE_TOLERANCE
PRIMARY_INPUT_KIND = linear.PRIMARY_INPUT_KIND
QUALITY_ONLY_INPUT_KIND = linear.QUALITY_ONLY_INPUT_KIND
TEACHER_SHUFFLE_INPUT_KIND = "carrier_plus_quality_source_teacher_pairing_shuffled"
MATERIAL_MEAN_DELTA = linear.MATERIAL_MEAN_DELTA
MATERIAL_MEDIAN_DELTA = linear.MATERIAL_MEDIAN_DELTA
MIN_POSITIVE_SESSIONS = linear.MIN_POSITIVE_SESSIONS
SCALE_FLOOR = linear.SCALE_FLOOR
RESIDUAL_SCALE_FLOOR = linear.RESIDUAL_SCALE_FLOOR
SourceExamples = linear.SourceExamples


def _need(condition: bool, message: str) -> None:
    v1._need(condition, message)


def polynomial_feature_names(raw_names: Sequence[str]) -> tuple[str, ...]:
    """Frozen F2 ordering: linear, squares, then lexicographic crosses."""

    names = tuple(raw_names)
    _need(bool(names) and len(names) == len(set(names)), "QC2F5 raw feature names invalid")
    return (
        tuple(f"linear:{name}" for name in names)
        + tuple(f"square:{name}^2" for name in names)
        + tuple(f"cross:{names[left]}*{names[right]}" for left in range(len(names))
                for right in range(left + 1, len(names)))
    )


def degree2_features(standardized: np.ndarray) -> np.ndarray:
    """Return exactly [linear, squares, pairwise crosses] without an intercept."""

    values = np.asarray(standardized, dtype=np.float64)
    _need(values.ndim == 2 and values.shape[1] > 0 and np.isfinite(values).all(),
          "QC2F5 polynomial input invalid")
    count = values.shape[1]
    pieces = [values, np.square(values)]
    if count > 1:
        pieces.append(np.column_stack([values[:, left] * values[:, right]
                                       for left in range(count) for right in range(left + 1, count)]))
    result = np.concatenate(pieces, axis=1)
    expected = count + count + count * (count - 1) // 2
    _need(result.shape == (values.shape[0], expected) and np.isfinite(result).all(),
          "QC2F5 polynomial feature width/nonfinite drift")
    return result


def _raw_feature_names(input_kind: str) -> tuple[str, ...]:
    if input_kind in (PRIMARY_INPUT_KIND, TEACHER_SHUFFLE_INPUT_KIND):
        return ("B_support_w1", "B_support_w2", "B_support_w3", "B_support_w4", "B_support_b") + QUALITY_NAMES
    if input_kind == QUALITY_ONLY_INPUT_KIND:
        return QUALITY_NAMES
    raise v1.SparseEventEndpointError(f"unknown QC2F5 input kind {input_kind}")


def _raw_features(carrier: np.ndarray, quality: np.ndarray, *, input_kind: str) -> np.ndarray:
    b = np.asarray(carrier, dtype=np.float64)
    q = np.asarray(quality, dtype=np.float64)
    _need(b.shape == (v1.EXPECTED_NEURONS, CARRIER_DIM), "QC2F5 carrier input drift")
    _need(q.shape == (v1.EXPECTED_NEURONS, QUALITY_DIM), "QC2F5 quality input drift")
    if input_kind in (PRIMARY_INPUT_KIND, TEACHER_SHUFFLE_INPUT_KIND):
        return np.concatenate((b, q), axis=1)
    if input_kind == QUALITY_ONLY_INPUT_KIND:
        return q
    raise v1.SparseEventEndpointError(f"unknown QC2F5 input kind {input_kind}")


def _example_rows(examples: SourceExamples, input_kind: str) -> tuple[np.ndarray, np.ndarray]:
    actual_kind = PRIMARY_INPUT_KIND if input_kind == TEACHER_SHUFFLE_INPUT_KIND else input_kind
    return examples.rows(actual_kind)


@dataclass(frozen=True)
class QuadraticOperator:
    """Channel-shared F2 with standardization before fixed polynomial expansion."""

    input_kind: str
    ridge_lambda: float
    raw_feature_mean: np.ndarray
    raw_feature_scale: np.ndarray
    residual_mean: np.ndarray
    residual_scale: np.ndarray
    standardized_coefficients: np.ndarray
    source_sessions: tuple[str, ...]
    outer_date: str
    budget: int
    operator_sha256: str

    @property
    def raw_input_dim(self) -> int:
        return int(self.raw_feature_mean.size)

    @property
    def polynomial_feature_dim(self) -> int:
        return self.raw_input_dim * (self.raw_input_dim + 3) // 2

    def predict_residual(self, carrier: np.ndarray, quality: np.ndarray) -> np.ndarray:
        raw = _raw_features(carrier, quality, input_kind=self.input_kind)
        _need(raw.shape[1] == self.raw_input_dim, "QC2F5 raw feature width drift")
        phi = degree2_features((raw - self.raw_feature_mean) / self.raw_feature_scale)
        design = np.column_stack((np.ones(len(phi), dtype=np.float64), phi))
        _need(design.shape[1] == self.standardized_coefficients.shape[0], "QC2F5 F2 design width drift")
        residual = design @ self.standardized_coefficients
        residual = residual * self.residual_scale + self.residual_mean
        _need(residual.shape == (v1.EXPECTED_NEURONS, CARRIER_DIM) and np.isfinite(residual).all(),
              "QC2F5 residual invalid")
        return residual

    def apply(self, carrier: np.ndarray, quality: np.ndarray) -> np.ndarray:
        corrected = np.asarray(carrier, dtype=np.float64) + self.predict_residual(carrier, quality)
        _need(corrected.shape == (v1.EXPECTED_NEURONS, CARRIER_DIM) and np.isfinite(corrected).all(),
              "QC2F5 corrected carrier invalid")
        return corrected

    def manifest(self) -> dict[str, Any]:
        raw_names = _raw_feature_names(self.input_kind)
        polynomial_names = polynomial_feature_names(raw_names)
        return {
            "family": "fixed_degree2_standardized_polynomial_ridge",
            "input_kind": self.input_kind,
            "ridge_lambda": self.ridge_lambda,
            "raw_input_dim": self.raw_input_dim,
            "polynomial_feature_dim": self.polynomial_feature_dim,
            "design_width_with_intercept": self.standardized_coefficients.shape[0],
            "raw_feature_order": list(raw_names),
            "polynomial_feature_order": list(polynomial_names),
            "quadratic_terms": "linear + squares + pairwise_cross_terms; no channel index",
            "carrier_dim": CARRIER_DIM,
            "quality_dim": QUALITY_DIM,
            "source_sessions": list(self.source_sessions),
            "outer_date": self.outer_date,
            "budget": self.budget,
            "has_channel_index_feature": False,
            "target_session_later_labels_enter_operator_fit": False,
            "array_sha256": {
                "raw_feature_mean": v1.array_sha256(self.raw_feature_mean),
                "raw_feature_scale": v1.array_sha256(self.raw_feature_scale),
                "residual_mean": v1.array_sha256(self.residual_mean),
                "residual_scale": v1.array_sha256(self.residual_scale),
                "standardized_coefficients": v1.array_sha256(self.standardized_coefficients),
            },
            "operator_sha256": self.operator_sha256,
        }


def fit_quadratic_operator(examples: SourceExamples, *, input_kind: str, ridge_lambda: float) -> QuadraticOperator:
    """Fit the frozen F2 ridge map only from supplied source rows."""

    _need(ridge_lambda in C2F_RIDGE_GRID, "QC2F5 lambda outside frozen grid")
    raw, targets = _example_rows(examples, input_kind)
    _need(raw.ndim == 2 and targets.shape == (raw.shape[0], CARRIER_DIM), "QC2F5 source row shape drift")
    raw_mean = raw.mean(axis=0)
    raw_scale = np.maximum(raw.std(axis=0), SCALE_FLOOR)
    residual_mean = targets.mean(axis=0)
    residual_scale = np.maximum(targets.std(axis=0), RESIDUAL_SCALE_FLOOR)
    phi = degree2_features((raw - raw_mean) / raw_scale)
    y = (targets - residual_mean) / residual_scale
    design = np.column_stack((np.ones(len(phi), dtype=np.float64), phi))
    penalty = np.diag([0.0] + [1.0] * phi.shape[1]) * (len(phi) * float(ridge_lambda))
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    _need(np.isfinite(coefficient).all(), "QC2F5 F2 coefficient nonfinite")
    body = {
        "protocol": PROTOCOL, "input_kind": input_kind, "ridge_lambda": ridge_lambda,
        "outer_date": examples.outer_date, "budget": examples.budget,
        "source_sessions": list(examples.session_names),
        "raw_feature_mean": v1.array_sha256(raw_mean), "raw_feature_scale": v1.array_sha256(raw_scale),
        "residual_mean": v1.array_sha256(residual_mean), "residual_scale": v1.array_sha256(residual_scale),
        "standardized_coefficients": v1.array_sha256(coefficient),
    }
    return QuadraticOperator(input_kind, float(ridge_lambda), np.asarray(raw_mean), np.asarray(raw_scale),
                             np.asarray(residual_mean), np.asarray(residual_scale), np.asarray(coefficient),
                             examples.session_names, examples.outer_date, examples.budget,
                             v1.canonical_sha256(body))


def source_teacher_pairing_shuffle(examples: SourceExamples) -> tuple[SourceExamples, dict[str, Any]]:
    """Fixed-point-free permutation of complete source teacher rows, marginal fixed."""

    total = examples.future_teachers.shape[0] * examples.future_teachers.shape[1]
    key = f"{v1.SHUFFLE_NAMESPACE}:qc2f5-source-teacher:{examples.outer_date}:M{examples.budget}"
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
    order = np.arange(total, dtype=np.int64)
    for _ in range(100):
        rng.shuffle(order)
        if not np.any(order == np.arange(total)):
            break
    _need(not np.any(order == np.arange(total)), "QC2F5 teacher pairing shuffle fixed point")
    teachers = examples.future_teachers.reshape(total, CARRIER_DIM)[order].reshape(examples.future_teachers.shape)
    return SourceExamples(examples.session_names, examples.outer_date, examples.budget,
                          examples.support_carriers, examples.quality, teachers, examples.manifests), {
        "namespace": "h1-qc2f5-source-teacher-pairing-shuffle-v1",
        "fixed_points": 0, "order_sha256": v1.array_sha256(order),
        "teacher_marginal_sha256_before": v1.array_sha256(np.sort(examples.future_teachers.reshape(total, CARRIER_DIM), axis=0)),
        "teacher_marginal_sha256_after": v1.array_sha256(np.sort(teachers.reshape(total, CARRIER_DIM), axis=0)),
    }


def _source_inner_validation(sessions: Mapping[str, v1.EventSession], basis: v2.EndpointBasisV2, *, outer_date: str,
                             budget: int, ridge_lambda: float) -> dict[str, Any]:
    dates = tuple(date for date in v1.H1_DATES if date != outer_date)
    rows: list[dict[str, Any]] = []
    for inner_date in dates:
        train = tuple(name for name in v1.H1_HELDIN_SESSIONS if v1.session_date(name) not in {outer_date, inner_date})
        valid = tuple(name for name in v1.H1_HELDIN_SESSIONS if v1.session_date(name) == inner_date)
        examples = linear.build_source_examples(sessions, basis, source_names=train, outer_date=outer_date, budget=budget)
        operator = fit_quadratic_operator(examples, input_kind=PRIMARY_INPUT_KIND, ridge_lambda=ridge_lambda)
        for name in valid:
            support, quality, _ = linear.fit_support_sufficient_statistics(sessions[name], basis, budget=budget)
            later = linear._events_after(sessions[name], budget)
            z_later, observed = v2.event_arrays(later, basis)
            rows.append({"session": name, "inner_validation_date": inner_date, "outer_date": outer_date,
                         "operator_source_dates": sorted({v1.session_date(value) for value in train}),
                         "median_later_event_raw_log_rate_r2": linear._median_channel_r2(observed, operator.apply(support, quality), z_later),
                         "operator_sha256": operator.operator_sha256})
    values = np.asarray([row["median_later_event_raw_log_rate_r2"] for row in rows])
    _need(values.size == len(linear._source_names_for_outer(outer_date)) and np.isfinite(values).all(),
          "QC2F5 inner LODO coverage drift")
    return {"ridge_lambda": ridge_lambda, "outer_date": outer_date, "inner_source_dates": list(dates),
            "selection_metric": "mean_per_recording_median_channel_later_event_raw_log_rate_r2",
            "rows": rows, "mean_score": float(values.mean()), "median_score": float(np.median(values)),
            "defined_recordings": int(values.size), "outer_date_visible_to_operator_training": False}


def select_operator_lambda(sessions: Mapping[str, v1.EventSession], basis: v2.EndpointBasisV2, *, outer_date: str,
                           budget: int) -> tuple[float, list[dict[str, Any]]]:
    rows = [_source_inner_validation(sessions, basis, outer_date=outer_date, budget=budget, ridge_lambda=value)
            for value in C2F_RIDGE_GRID]
    return _select_lambda_from_source_scores(rows), rows


def _select_lambda_from_source_scores(rows: Sequence[Mapping[str, Any]]) -> float:
    """Frozen QC2F5 tie rule: prefer the stronger ridge among score ties."""

    _need(len(rows) == len(C2F_RIDGE_GRID), "QC2F5 lambda grid coverage drift")
    _need(tuple(float(row["ridge_lambda"]) for row in rows) == C2F_RIDGE_GRID,
          "QC2F5 lambda grid order drift")
    best_score = max(float(row["mean_score"]) for row in rows)
    return max(float(row["ridge_lambda"]) for row in rows
               if float(row["mean_score"]) >= best_score - SELECTION_TIE_TOLERANCE)


def evaluate_outer_session(session: v1.EventSession, basis: v2.EndpointBasisV2, *, budget: int,
                           primary: QuadraticOperator, quality_only: QuadraticOperator,
                           teacher_shuffled: QuadraticOperator, linear_operator: linear.C2FOperator) -> dict[str, Any]:
    _need(v1.session_date(session.session_name) == basis.outer_date, "QC2F5 target/basis outer-date mismatch")
    for operator in (primary, quality_only, teacher_shuffled):
        _need(not any(v1.session_date(name) == session.date for name in operator.source_sessions),
              "QC2F5 outer target date leaked into F2")
    support, quality, support_fit = linear.fit_support_sufficient_statistics(session, basis, budget=budget)
    label_support, label_quality, label_fit = linear.fit_support_sufficient_statistics(session, basis, budget=budget, shuffled_labels=True)
    corrected = primary.apply(support, quality)
    label_corrected = primary.apply(label_support, label_quality)
    quality_corrected = quality_only.apply(support, quality)
    teacher_corrected = teacher_shuffled.apply(support, quality)
    linear_corrected = linear_operator.apply(support, quality)
    row_corrected, row_manifest = linear.row_shuffle_carrier(corrected, session=session.session_name, budget=budget)
    later = linear._events_after(session, budget)
    z_later, observed = v2.event_arrays(later, basis)
    support_mean = np.mean(np.stack([event.log_rates for event in linear._events_before(session, budget)]), axis=0)
    carriers = {"hse5": support, "qc2f5": corrected, "linear_c2f5": linear_corrected,
                "label_shuffled_qc2f5": label_corrected, "row_shuffled_qc2f5": row_corrected,
                "quality_only_qc2f5": quality_corrected, "source_teacher_shuffled_qc2f5": teacher_corrected}
    scores = {name: v1.r2_by_channel(observed, v2.predict(value, z_later)) for name, value in carriers.items()}
    scores["intercept_only"] = v1.r2_by_channel(observed, np.broadcast_to(support_mean, observed.shape))
    defined = np.logical_and.reduce([np.isfinite(value) for value in scores.values()])
    _need(np.any(defined), f"{session.session_name}: QC2F5 no jointly defined channels")
    medians = {name: float(np.median(value[defined])) for name, value in scores.items()}
    return {"status": "defined", "session": session.session_name, "date": session.date, "budget": budget,
            "support_events": len(linear._events_before(session, budget)), "later_events": len(later),
            "defined_channels": int(defined.sum()), "basis_sha256": basis.basis_sha256,
            "raw_hse5_support_carrier_sha256": v1.array_sha256(support), "corrected_carrier_sha256": v1.array_sha256(corrected),
            "support_fit": support_fit, "label_shuffled_support_fit": label_fit, "row_shuffle": row_manifest,
            "primary_operator_sha256": primary.operator_sha256, "quality_only_operator_sha256": quality_only.operator_sha256,
            "teacher_shuffle_operator_sha256": teacher_shuffled.operator_sha256, "linear_c2f5_operator_sha256": linear_operator.operator_sha256,
            **{f"median_r2_{name}": value for name, value in medians.items()},
            "delta_qc2f5_minus_hse5": medians["qc2f5"] - medians["hse5"],
            "delta_qc2f5_minus_linear_c2f5": medians["qc2f5"] - medians["linear_c2f5"],
            "delta_qc2f5_minus_label_shuffled": medians["qc2f5"] - medians["label_shuffled_qc2f5"],
            "delta_qc2f5_minus_row_shuffled": medians["qc2f5"] - medians["row_shuffled_qc2f5"],
            "delta_qc2f5_minus_quality_only": medians["qc2f5"] - medians["quality_only_qc2f5"],
            "delta_qc2f5_minus_source_teacher_shuffled": medians["qc2f5"] - medians["source_teacher_shuffled_qc2f5"],
            "delta_qc2f5_minus_intercept": medians["qc2f5"] - medians["intercept_only"],
            "target_session_later_labels_enter_operator_fit": False}


def _summary(rows: Mapping[str, Mapping[str, Any]], field: str) -> dict[str, Any]:
    return v1.paired_summary([(name, row[field]) for name, row in rows.items()])


def budget_gate(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    hse = _summary(rows, "delta_qc2f5_minus_hse5")
    material = bool(linear._positive(hse) and hse["mean"] >= MATERIAL_MEAN_DELTA and hse["median"] >= MATERIAL_MEDIAN_DELTA)
    required = {"correct_minus_linear_c2f5": _summary(rows, "delta_qc2f5_minus_linear_c2f5"),
                "correct_minus_label_shuffled": _summary(rows, "delta_qc2f5_minus_label_shuffled"),
                "correct_minus_source_teacher_shuffled": _summary(rows, "delta_qc2f5_minus_source_teacher_shuffled"),
                "correct_minus_quality_only": _summary(rows, "delta_qc2f5_minus_quality_only"),
                "correct_minus_intercept": _summary(rows, "delta_qc2f5_minus_intercept")}
    audits = {"correct_minus_row_shuffled": _summary(rows, "delta_qc2f5_minus_row_shuffled")}
    control_pass = {key: linear._positive(value) for key, value in required.items()}
    return {"passed": bool(material and all(control_pass.values())), "material_gain_vs_hse5": material,
            "controls": control_pass, "audit_controls": {key: linear._positive(value) for key, value in audits.items()},
            "thresholds": {"mean_delta_vs_hse5": MATERIAL_MEAN_DELTA, "median_delta_vs_hse5": MATERIAL_MEDIAN_DELTA,
                           "minimum_positive_sessions": MIN_POSITIVE_SESSIONS, "leave_largest_absolute_out_mean_positive": True,
                           "required_controls": list(required)}, "correct_minus_hse5": hse, **required, **audits}


def _compare_linear_receipt(observed: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    diffs: list[dict[str, Any]] = []
    for budget in SUPPORT_BUDGETS:
        for name in v1.H1_HELDIN_SESSIONS:
            own = observed["budgets"][f"M{budget}"]["sessions"][name]
            ref = reference["budgets"][f"M{budget}"]["sessions"][name]
            for field in ("median_r2_hse5", "median_r2_c2f5", "raw_hse5_label_delta", "raw_hse5_intercept_delta"):
                diffs.append({"budget": budget, "session": name, "field": field,
                              "absolute_difference": abs(float(own[field]) - float(ref[field]))})
        for field in ("correct_minus_hse5", "correct_minus_label_shuffled", "correct_minus_quality_only", "correct_minus_intercept"):
            own = observed["budgets"][f"M{budget}"]["gate"][field]
            ref = reference["budgets"][f"M{budget}"]["gate"][field]
            for value in ("mean", "median", "leave_largest_absolute_out_mean"):
                diffs.append({"budget": budget, "session": "aggregate", "field": f"{field}:{value}",
                              "absolute_difference": abs(float(own[value]) - float(ref[value]))})
    maximum = max(item["absolute_difference"] for item in diffs)
    return {"comparisons": len(diffs), "absolute_tolerance": 1.0e-10,
            "maximum_absolute_difference": maximum, "passed": bool(maximum <= 1.0e-10), "differences": diffs}


def run_screen(sessions: Mapping[str, v1.EventSession], *, hse5_reference: Mapping[str, Any],
               linear_c2f5_reference: Mapping[str, Any]) -> dict[str, Any]:
    """Run fixed QC2F5 after exact independently recomputed C2F5 reproduction."""

    linear_recomputed = linear.run_screen(sessions, hse5_reference=hse5_reference)
    linear_reproduction = _compare_linear_receipt(linear_recomputed, linear_c2f5_reference)
    _need(linear_reproduction["passed"], f"QC2F5 fail closed: linear C2F5 reproduction {linear_reproduction['maximum_absolute_difference']}")
    budgets: dict[str, Any] = {}
    for budget in SUPPORT_BUDGETS:
        rows: dict[str, dict[str, Any]] = {}
        details: dict[str, Any] = {}
        for outer_date in v1.H1_DATES:
            basis = linear._basis_for_outer(sessions, outer_date=outer_date)
            selected, selection_rows = select_operator_lambda(sessions, basis, outer_date=outer_date, budget=budget)
            names = linear._source_names_for_outer(outer_date)
            examples = linear.build_source_examples(sessions, basis, source_names=names, outer_date=outer_date, budget=budget)
            shuffled_examples, shuffle_manifest = source_teacher_pairing_shuffle(examples)
            primary = fit_quadratic_operator(examples, input_kind=PRIMARY_INPUT_KIND, ridge_lambda=selected)
            quality = fit_quadratic_operator(examples, input_kind=QUALITY_ONLY_INPUT_KIND, ridge_lambda=selected)
            teacher = fit_quadratic_operator(shuffled_examples, input_kind=TEACHER_SHUFFLE_INPUT_KIND, ridge_lambda=selected)
            linear_selected, linear_selection_rows = linear.select_operator_lambda(sessions, basis, outer_date=outer_date, budget=budget)
            linear_operator = linear.fit_operator(examples, input_kind=PRIMARY_INPUT_KIND, ridge_lambda=linear_selected)
            details[outer_date] = {"basis": basis.manifest(), "operator_lambda_selection": {"grid": list(C2F_RIDGE_GRID),
                                   "tie_break": "smallest_lambda_within_1e-12_of_best_mean_source_date_LODO_score",
                                   "selected_lambda": selected, "source_date_lodo_rows": selection_rows},
                                   "linear_c2f5_lambda_selection": {"selected_lambda": linear_selected, "source_date_lodo_rows": linear_selection_rows},
                                   "primary_operator": primary.manifest(), "quality_only_operator": quality.manifest(),
                                   "source_teacher_pairing_shuffle": shuffle_manifest, "source_teacher_shuffle_operator": teacher.manifest(),
                                   "linear_c2f5_operator": linear_operator.manifest(),
                                   "teacher_headroom_and_learnability_diagnostic": linear._teacher_diagnostic(examples)}
            for name in v1.H1_HELDIN_SESSIONS:
                if v1.session_date(name) == outer_date:
                    rows[name] = evaluate_outer_session(sessions[name], basis, budget=budget, primary=primary, quality_only=quality,
                                                        teacher_shuffled=teacher, linear_operator=linear_operator)
        _need(tuple(rows) == v1.H1_HELDIN_SESSIONS, "QC2F5 outer coverage/order drift")
        # Raw H-SE5 controls are copied from the independently recomputed exact C2F5 path.
        for name in v1.H1_HELDIN_SESSIONS:
            source = linear_recomputed["budgets"][f"M{budget}"]["sessions"][name]
            rows[name]["raw_hse5_label_delta"] = source["raw_hse5_label_delta"]
            rows[name]["raw_hse5_intercept_delta"] = source["raw_hse5_intercept_delta"]
        baseline = linear._baseline_reproduction(rows, hse5_reference, budget=budget)
        _need(baseline["passed"], f"QC2F5 fail closed: H-SE5 reproduction M{budget}")
        budgets[f"M{budget}"] = {"budget_trials": budget, "outer_date_details": details, "sessions": rows,
                                  "raw_hse5_v2_reproduction": baseline, "gate": budget_gate(rows)}
    passing = all(budgets[f"M{budget}"]["gate"]["passed"] for budget in SUPPORT_BUDGETS)
    return {"schema": SCHEMA, "protocol": PROTOCOL, "candidate_matrix_predeclared_before_data_run": True,
            "frozen_constants": {"rank": RANK, "carrier_dim": CARRIER_DIM, "quality_names": list(QUALITY_NAMES),
                                 "quality_dim": QUALITY_DIM, "support_budgets": list(SUPPORT_BUDGETS),
                                 "operator_ridge_grid": list(C2F_RIDGE_GRID), "primary_operator": "B_corrected = B_support + F2([B_support, quality])",
                                 "quality_only_control": "B_support + F2(quality), with no B_support supplied to F2",
                                 "source_teacher_shuffle_control": "F2 trained after fixed source future-teacher pairing shuffle",
                                 "linear_c2f5_control": "independently recomputed original affine C2F5",
                                 "polynomial": "standardize raw inputs; linear + squares + pairwise crosses; no channel index",
                                 "lambda_selection": "outer-date-excluded source-date LODO; strongest ridge on tie",
                                 "target_session_optimizer_steps": 0, "target_session_backward_steps": 0, "channel_index_feature": False},
            "budgets": budgets, "linear_c2f5_exact_reproduction": linear_reproduction,
            "passing_candidate": passing, "status": "PASS_CPU_QC2F5_MATERIAL" if passing else "STOP_CPU_QC2F5_NOT_MATERIAL",
            "gpu_authorized_by_this_screen": False,
            "scope": {"native_position_endpoints_per_event": 2, "native_event_timestamps_per_event": 2,
                      "dense_velocity_opened": False, "within_event_position_trajectory_opened": False,
                      "target_session_later_labels_enter_operator_fit": False, "target_session_optimizer_steps": 0,
                      "target_session_backward_steps": 0}}
