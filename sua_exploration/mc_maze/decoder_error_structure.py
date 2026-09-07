"""Paired decoder error-structure decomposition (CPU, forward-only scaffolding).

Stratifies paired carrier-vs-control query-window scores into pre-registered
axes.  All stratum edge rules are frozen constants in this module — they are
never recomputed from the windows being scored.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "decoder_error_structure_v1"

TUNING_NUM_DIRECTIONS = 8
CANONICAL_DIRECTIONS_RAD: tuple[float, ...] = tuple(
    -3.0 * math.pi / 4.0 + k * (math.pi / 4.0) for k in range(TUNING_NUM_DIRECTIONS)
)


def _nearest_canonical_direction_index(target_dir_rad: float) -> int:
    directions = np.asarray(CANONICAL_DIRECTIONS_RAD, dtype=np.float64)
    wrapped = (directions - target_dir_rad + math.pi) % (2.0 * math.pi) - math.pi
    return int(np.argmin(np.abs(wrapped)))

SEALED_FORMAL_TEST_SESSIONS: frozenset[str] = frozenset(
    {
        "sub-C_ses-CO-20151113",
        "sub-C_ses-CO-20151116",
        "sub-C_ses-CO-20151117",
        "sub-C_ses-CO-20151119",
        "sub-C_ses-CO-20151120",
        "sub-C_ses-CO-20151201",
    }
)

# Frozen stratum edge rules (protocol section 2).  Do not tune after seeing results.
SPEED_NORM_EDGES = (0.0, 0.35, 0.70, 1.05, float("inf"))
TUNING_M_EDGES = (0.0, 0.05, 0.15, float("inf"))
DESIGN_CONDITION_LOW_THRESHOLD = 10.0
PHASE_BIN_COUNT = 3

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42

FDR_Q_THRESHOLD = 0.10
NOMINAL_ALPHA_UNCORRECTED = 0.05
MIN_WINDOWS_FOR_MULTIPLICITY = 4
MIN_SESSIONS_FOR_MULTIPLICITY = 2
SIGNFLIP_PERMUTATIONS = 5000
SIGNFLIP_SEED = 42

STRATUM_AXES: tuple[str, ...] = (
    "target_direction",
    "movement_speed",
    "within_trial_phase",
    "active_tuning_strength",
    "calibration_design_coverage",
)

DIRECTION_STRATUM_LABELS: tuple[str, ...] = tuple(
    f"dir_{index}" for index in range(TUNING_NUM_DIRECTIONS)
)
SPEED_STRATUM_LABELS: tuple[str, ...] = ("speed_q1", "speed_q2", "speed_q3", "speed_q4")
PHASE_STRATUM_LABELS: tuple[str, ...] = ("phase_early", "phase_middle", "phase_late")
TUNING_STRATUM_LABELS: tuple[str, ...] = (
    "tuning_low",
    "tuning_mid",
    "tuning_high",
)
DESIGN_DIRECTION_COUNT_LABELS: tuple[str, ...] = (
    "design_dirs_2",
    "design_dirs_3_4",
    "design_dirs_5_6",
    "design_dirs_7_8",
)
DESIGN_CONDITION_LABELS: tuple[str, ...] = (
    "design_cond_rank_deficient",
    "design_cond_finite_low",
    "design_cond_finite_high",
)

FROZEN_STRATUM_DEFINITIONS: dict[str, Any] = {
    "target_direction": {
        "kind": "circular_8",
        "labels": list(DIRECTION_STRATUM_LABELS),
        "canonical_directions_rad": list(CANONICAL_DIRECTIONS_RAD),
        "assignment": "nearest canonical center-out direction to window target velocity",
    },
    "movement_speed": {
        "kind": "fixed_norm_edges",
        "labels": list(SPEED_STRATUM_LABELS),
        "edges": list(SPEED_NORM_EDGES),
        "assignment": "L2 norm of target velocity vector in standardized coordinates",
    },
    "within_trial_phase": {
        "kind": "equal_thirds",
        "labels": list(PHASE_STRATUM_LABELS),
        "bin_count": PHASE_BIN_COUNT,
        "assignment": "relative bin index within scored trial window",
    },
    "active_tuning_strength": {
        "kind": "fixed_m_edges",
        "labels": list(TUNING_STRATUM_LABELS),
        "edges": list(TUNING_M_EDGES),
        "assignment": "activity-weighted mean modulation depth m_i of units in window",
    },
    "calibration_design_coverage": {
        "kind": "session_categorical",
        "direction_count_labels": list(DESIGN_DIRECTION_COUNT_LABELS),
        "condition_labels": list(DESIGN_CONDITION_LABELS),
        "condition_low_threshold": DESIGN_CONDITION_LOW_THRESHOLD,
        "combined_labels": [
            f"{direction}_{condition}"
            for direction in DESIGN_DIRECTION_COUNT_LABELS
            for condition in DESIGN_CONDITION_LABELS
        ],
        "assignment": "session-level calibration pool metadata only",
    },
}

PRIMARY_SCORE = "common_denominator_mse"
SECONDARY_SCORE = "within_stratum_r2"


@dataclass(frozen=True)
class QueryWindowRecord:
  session_name: str
  trial_index: int
  bin_index: int
  bins_in_trial: int
  target: np.ndarray
  pred_carrier: np.ndarray
  pred_control: np.ndarray
  unit_modulation_m: np.ndarray
  unit_activity: np.ndarray
  session_n_directions: int
  session_design_rank: int
  session_design_condition: float
  target_dir_rad: float | None = None

  def identity(self) -> dict[str, Any]:
      return {
          "session_name": self.session_name,
          "trial_index": int(self.trial_index),
          "bin_index": int(self.bin_index),
      }

  def identity_sha256(self) -> str:
      payload = json.dumps(self.identity(), sort_keys=True, separators=(",", ":"))
      return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_sessions_allowed(session_names: Sequence[str]) -> None:
    blocked = sorted({name for name in session_names if name in SEALED_FORMAL_TEST_SESSIONS})
    if blocked:
        raise ValueError(
            "refusing sealed formal-test sessions: " + ", ".join(blocked)
        )


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def paired_bootstrap_interval(
    deltas: Sequence[float],
    *,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap requires nonempty finite deltas")
    rng = np.random.default_rng(seed)
    means = np.empty(n_samples, dtype=np.float64)
    size = values.size
    for index in range(n_samples):
        sample = values[rng.integers(0, size, size=size)]
        means[index] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_ci_excludes_zero(ci: Sequence[float]) -> bool:
    low, high = (float(ci[0]), float(ci[1]))
    return (low > 0.0 and high > 0.0) or (low < 0.0 and high < 0.0)


def signflip_pvalue_two_sided(
    deltas: Sequence[float],
    *,
    n_permutations: int = SIGNFLIP_PERMUTATIONS,
    seed: int = SIGNFLIP_SEED,
) -> float:
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        return float("nan")
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_permutations):
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=values.size)
        if abs(float(np.mean(values * signs))) >= observed:
            exceed += 1
    return float((exceed + 1) / (n_permutations + 1))


def benjamini_hochberg_adjusted(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    tested = values.size
    if tested == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted_sorted = np.empty(tested, dtype=np.float64)
    previous = 1.0
    for index in range(tested - 1, -1, -1):
        rank = index + 1
        current = min(previous, ranked[index] * tested / rank)
        adjusted_sorted[index] = current
        previous = current
    adjusted = np.empty(tested, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted


def stratum_eligible_for_multiplicity(row: Mapping[str, Any]) -> tuple[bool, str | None]:
    n_windows = int(row.get("n_windows", 0))
    if n_windows == 0:
        return False, "empty"
    if n_windows < MIN_WINDOWS_FOR_MULTIPLICITY:
        return False, "too_few_windows"
    n_sessions = int(row.get("n_sessions", 0))
    if n_sessions < MIN_SESSIONS_FOR_MULTIPLICITY:
        return False, "too_few_sessions"
    deltas = row.get("window_primary_deltas") or []
    if len(deltas) < MIN_WINDOWS_FOR_MULTIPLICITY:
        return False, "too_few_windows"
    return True, None


def apply_axis_family_multiplicity_correction(
    axis: str,
    strata: Mapping[str, Mapping[str, Any]],
    *,
    fdr_q: float = FDR_Q_THRESHOLD,
    nominal_alpha: float = NOMINAL_ALPHA_UNCORRECTED,
) -> dict[str, Any]:
    family_size_total = len(strata)
    excluded_counts: dict[str, int] = {"empty": 0, "too_few_windows": 0, "too_few_sessions": 0}
    eligible_labels: list[str] = []
    raw_p_values: list[float] = []
    per_stratum: dict[str, Any] = {}

    for label in sorted(strata):
        row = strata[label]
        eligible, reason = stratum_eligible_for_multiplicity(row)
        if not eligible:
            if reason == "empty":
                excluded_counts["empty"] += 1
            elif reason == "too_few_sessions":
                excluded_counts["too_few_sessions"] += 1
            else:
                excluded_counts["too_few_windows"] += 1
            ci = row.get("paired_bootstrap_95_ci_primary")
            per_stratum[label] = {
                "family": axis,
                "family_size_total": family_size_total,
                "family_size_tested": 0,
                "eligible_for_correction": False,
                "exclusion_reason": reason,
                "raw_p_value_two_sided": None,
                "paired_bootstrap_95_ci_primary": ci,
                "bh_adjusted_p_value": None,
                "ci_excludes_zero": bootstrap_ci_excludes_zero(ci) if ci is not None else False,
                "uncorrected_significant": False,
                "survives_fdr": False,
                "discovery": False,
            }
            continue

        deltas = [float(value) for value in row["window_primary_deltas"]]
        p_value = signflip_pvalue_two_sided(deltas)
        ci = row.get("paired_bootstrap_95_ci_primary") or list(paired_bootstrap_interval(deltas))
        ci_excludes_zero = bootstrap_ci_excludes_zero(ci)
        eligible_labels.append(label)
        raw_p_values.append(p_value)

    family_size_tested = len(eligible_labels)
    adjusted = (
        benjamini_hochberg_adjusted(raw_p_values)
        if family_size_tested > 0
        else np.asarray([], dtype=np.float64)
    )

    for index, label in enumerate(eligible_labels):
        p_value = float(raw_p_values[index])
        adjusted_p = float(adjusted[index])
        ci = strata[label].get("paired_bootstrap_95_ci_primary")
        ci_excludes_zero = bootstrap_ci_excludes_zero(ci) if ci is not None else False
        uncorrected_significant = bool(p_value < nominal_alpha and ci_excludes_zero)
        survives_fdr = bool(adjusted_p <= fdr_q and ci_excludes_zero)
        per_stratum[label] = {
            "family": axis,
            "family_size_total": family_size_total,
            "family_size_tested": family_size_tested,
            "eligible_for_correction": True,
            "exclusion_reason": None,
            "raw_p_value_two_sided": p_value,
            "paired_bootstrap_95_ci_primary": ci,
            "bh_adjusted_p_value": adjusted_p,
            "ci_excludes_zero": ci_excludes_zero,
            "uncorrected_significant": uncorrected_significant,
            "survives_fdr": survives_fdr,
            "discovery": survives_fdr,
        }

    for label in sorted(strata):
        if label in per_stratum and per_stratum[label]["eligible_for_correction"]:
            per_stratum[label]["family_size_tested"] = family_size_tested

    return {
        "method": "benjamini_hochberg",
        "fdr_q_threshold": fdr_q,
        "nominal_alpha_uncorrected": nominal_alpha,
        "family": axis,
        "family_size_total": family_size_total,
        "family_size_tested": family_size_tested,
        "excluded_counts": excluded_counts,
        "strata": per_stratum,
    }


def attach_multiplicity_correction(axes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for axis in STRATUM_AXES:
        axis_body = axes[axis]
        strata = axis_body.get("strata") or {}
        correction = apply_axis_family_multiplicity_correction(axis, strata)
        axis_body["multiplicity_correction"] = correction
        for label, row in correction["strata"].items():
            if label in strata:
                strata[label]["multiplicity"] = row
    return axes


def derive_target_direction_rad(target: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if target.size < 2:
        raise ValueError("target velocity must have at least two dimensions")
    return float(math.atan2(target[1], target[0]))


def assign_target_direction_label(target: np.ndarray, *, target_dir_rad: float | None = None) -> str:
    angle = target_dir_rad if target_dir_rad is not None else derive_target_direction_rad(target)
    index = _nearest_canonical_direction_index(angle)
    return DIRECTION_STRATUM_LABELS[index]


def assign_speed_label(target: np.ndarray) -> str:
    norm = float(np.linalg.norm(np.asarray(target, dtype=np.float64).reshape(-1)))
    for label, low, high in zip(
        SPEED_STRATUM_LABELS,
        SPEED_NORM_EDGES[:-1],
        SPEED_NORM_EDGES[1:],
    ):
        if low <= norm < high:
            return label
    return SPEED_STRATUM_LABELS[-1]


def assign_phase_label(bin_index: int, bins_in_trial: int) -> str:
    if bins_in_trial <= 0:
        raise ValueError("bins_in_trial must be positive")
    relative = float(bin_index) / float(bins_in_trial)
    if relative < 1.0 / PHASE_BIN_COUNT:
        return PHASE_STRATUM_LABELS[0]
    if relative < 2.0 / PHASE_BIN_COUNT:
        return PHASE_STRATUM_LABELS[1]
    return PHASE_STRATUM_LABELS[2]


def assign_tuning_strength_label(unit_modulation_m: np.ndarray, unit_activity: np.ndarray) -> str:
    modulation = np.asarray(unit_modulation_m, dtype=np.float64).reshape(-1)
    activity = np.asarray(unit_activity, dtype=np.float64).reshape(-1)
    if modulation.shape != activity.shape:
        raise ValueError("unit_modulation_m and unit_activity must match shape")
    weight = np.maximum(activity, 0.0)
    total = float(weight.sum())
    if total <= 1.0e-12:
        mean_m = 0.0
    else:
        mean_m = float(np.dot(weight, modulation) / total)
    for label, low, high in zip(
        TUNING_STRATUM_LABELS,
        TUNING_M_EDGES[:-1],
        TUNING_M_EDGES[1:],
    ):
        if low <= mean_m < high:
            return label
    return TUNING_STRATUM_LABELS[-1]


def assign_design_direction_count_label(session_n_directions: int) -> str:
    if session_n_directions <= 2:
        return DESIGN_DIRECTION_COUNT_LABELS[0]
    if session_n_directions <= 4:
        return DESIGN_DIRECTION_COUNT_LABELS[1]
    if session_n_directions <= 6:
        return DESIGN_DIRECTION_COUNT_LABELS[2]
    return DESIGN_DIRECTION_COUNT_LABELS[3]


def assign_design_condition_label(session_design_rank: int, session_design_condition: float) -> str:
    if session_design_rank < 3 or not math.isfinite(session_design_condition):
        return DESIGN_CONDITION_LABELS[0]
    if session_design_condition < DESIGN_CONDITION_LOW_THRESHOLD:
        return DESIGN_CONDITION_LABELS[1]
    return DESIGN_CONDITION_LABELS[2]


def assign_design_coverage_label(record: QueryWindowRecord) -> str:
    direction_label = assign_design_direction_count_label(record.session_n_directions)
    condition_label = assign_design_condition_label(
        record.session_design_rank,
        record.session_design_condition,
    )
    return f"{direction_label}_{condition_label}"


def assign_all_strata(record: QueryWindowRecord) -> dict[str, str]:
    return {
        "target_direction": assign_target_direction_label(
            record.target,
            target_dir_rad=record.target_dir_rad,
        ),
        "movement_speed": assign_speed_label(record.target),
        "within_trial_phase": assign_phase_label(record.bin_index, record.bins_in_trial),
        "active_tuning_strength": assign_tuning_strength_label(
            record.unit_modulation_m,
            record.unit_activity,
        ),
        "calibration_design_coverage": assign_design_coverage_label(record),
    }


def variance_weighted_r2(targets: np.ndarray, predictions: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.shape != predictions.shape or targets.ndim != 2:
        raise ValueError("targets and predictions must share shape [N, D]")
    if targets.shape[0] == 0:
        return float("nan")
    centered = targets - targets.mean(axis=0, keepdims=True)
    ss_tot = float(np.sum(centered * centered))
    if ss_tot <= 1.0e-12:
        return float("nan")
    residual = targets - predictions
    ss_res = float(np.sum(residual * residual))
    return 1.0 - ss_res / ss_tot


def common_denominator_mse(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    global_target_mean: np.ndarray,
) -> float:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    mean = np.asarray(global_target_mean, dtype=np.float64).reshape(1, -1)
    ss_tot = float(np.sum((targets - mean) ** 2))
    if ss_tot <= 1.0e-12:
        return float("nan")
    ss_res = float(np.sum((targets - predictions) ** 2))
    return ss_res / ss_tot


def decompose_error_bias_variance(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    errors = predictions - targets
    mean_error = errors.mean(axis=0)
    bias_squared = float(np.dot(mean_error, mean_error))
    centered = errors - mean_error
    variance = float(np.mean(np.sum(centered * centered, axis=1)))
    mse = float(np.mean(np.sum(errors * errors, axis=1)))
    return {
        "mse": mse,
        "bias_squared": bias_squared,
        "variance": variance,
    }


def circular_direction_dependence(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    target_dirs_rad: Sequence[float] | None = None,
) -> float:
    """Return 1 - R, where R is the mean resultant length of error angles.

    Error angles are measured in the target-velocity frame (circular statistic).
    Values near 0 mean isotropic errors; values near 1 mean strongly direction-aligned
    error structure.
    """
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.shape != predictions.shape or targets.ndim != 2 or targets.shape[1] < 2:
        raise ValueError("expected matching [N, >=2] targets and predictions")
    errors = predictions - targets
    if target_dirs_rad is None:
        angles = np.arctan2(targets[:, 1], targets[:, 0])
    else:
        angles = np.asarray(target_dirs_rad, dtype=np.float64).reshape(-1)
        if angles.shape[0] != targets.shape[0]:
            raise ValueError("target_dirs_rad length must match number of windows")
    error_angles = np.arctan2(errors[:, 1], errors[:, 0]) - angles
    unit_vectors = np.exp(1j * error_angles)
    resultant = np.abs(np.mean(unit_vectors))
    return float(1.0 - resultant)


def _axis_labels(axis: str) -> tuple[str, ...]:
    if axis == "target_direction":
        return DIRECTION_STRATUM_LABELS
    if axis == "movement_speed":
        return SPEED_STRATUM_LABELS
    if axis == "within_trial_phase":
        return PHASE_STRATUM_LABELS
    if axis == "active_tuning_strength":
        return TUNING_STRATUM_LABELS
    if axis == "calibration_design_coverage":
        return tuple(FROZEN_STRATUM_DEFINITIONS["calibration_design_coverage"]["combined_labels"])
    raise ValueError(f"unknown axis {axis!r}")


def _summarize_arm(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    global_target_mean: np.ndarray,
    target_dirs_rad: Sequence[float] | None,
) -> dict[str, Any]:
    residual = decompose_error_bias_variance(targets, predictions)
    return {
        "within_stratum_r2": variance_weighted_r2(targets, predictions),
        "common_denominator_mse": common_denominator_mse(
            targets,
            predictions,
            global_target_mean=global_target_mean,
        ),
        "residual": residual,
        "circular_direction_dependence": circular_direction_dependence(
            targets,
            predictions,
            target_dirs_rad=target_dirs_rad,
        ),
    }


def summarize_axis_stratum(
    records: Sequence[QueryWindowRecord],
    axis: str,
    label: str,
    *,
    global_target_mean: np.ndarray,
) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if assign_all_strata(record)[axis] == label
    ]
    if not selected:
        return {
            "label": label,
            "n_windows": 0,
            "n_sessions": 0,
            "carrier": None,
            "control": None,
            "paired_delta_r2": None,
            "paired_delta_common_denominator_mse": None,
            "paired_delta_primary": None,
            "paired_bootstrap_95_ci_primary": None,
            "window_primary_deltas": [],
            "sessions_positive_primary_delta": 0,
            "residual_delta": None,
            "circular_direction_dependence_delta": None,
        }

    targets = np.stack([record.target for record in selected], axis=0)
    carrier = np.stack([record.pred_carrier for record in selected], axis=0)
    control = np.stack([record.pred_control for record in selected], axis=0)
    dirs = [
        record.target_dir_rad
        if record.target_dir_rad is not None
        else derive_target_direction_rad(record.target)
        for record in selected
    ]

    carrier_summary = _summarize_arm(
        targets,
        carrier,
        global_target_mean=global_target_mean,
        target_dirs_rad=dirs,
    )
    control_summary = _summarize_arm(
        targets,
        control,
        global_target_mean=global_target_mean,
        target_dirs_rad=dirs,
    )
    paired_delta_r2 = carrier_summary["within_stratum_r2"] - control_summary["within_stratum_r2"]
    paired_delta_mse = (
        carrier_summary["common_denominator_mse"] - control_summary["common_denominator_mse"]
    )
    window_primary_deltas = []
    sessions: dict[str, list[float]] = {}
    for record in selected:
        target = record.target.reshape(1, -1)
        carrier_mse = common_denominator_mse(
            target,
            record.pred_carrier.reshape(1, -1),
            global_target_mean=global_target_mean,
        )
        control_mse = common_denominator_mse(
            target,
            record.pred_control.reshape(1, -1),
            global_target_mean=global_target_mean,
        )
        delta = carrier_mse - control_mse
        window_primary_deltas.append(delta)
        sessions.setdefault(record.session_name, []).append(delta)

    ci = paired_bootstrap_interval(window_primary_deltas)
    sessions_positive = sum(
        1 for values in sessions.values() if float(np.mean(values)) < 0.0
    )
    residual_delta = {
        key: carrier_summary["residual"][key] - control_summary["residual"][key]
        for key in ("mse", "bias_squared", "variance")
    }
    return {
        "label": label,
        "n_windows": len(selected),
        "n_sessions": len(sessions),
        "carrier": carrier_summary,
        "control": control_summary,
        "paired_delta_r2": paired_delta_r2,
        "paired_delta_common_denominator_mse": paired_delta_mse,
        "paired_delta_primary": paired_delta_mse,
        "paired_bootstrap_95_ci_primary": list(ci),
        "window_primary_deltas": [float(value) for value in window_primary_deltas],
        "sessions_positive_primary_delta": sessions_positive,
        "residual_delta": residual_delta,
        "circular_direction_dependence_delta": (
            carrier_summary["circular_direction_dependence"]
            - control_summary["circular_direction_dependence"]
        ),
    }


def uniform_gain_diagnostics(
    axis_results: Mapping[str, Mapping[str, Any]],
    *,
    delta_key: str = "paired_delta_primary",
) -> dict[str, Any]:
    deltas = []
    for row in axis_results.values():
        if row.get("n_windows", 0) == 0:
            continue
        if delta_key == "relative_mse_improvement":
            control = row.get("control") or {}
            carrier = row.get("carrier") or {}
            control_mse = (control.get("residual") or {}).get("mse")
            carrier_mse = (carrier.get("residual") or {}).get("mse")
            if control_mse is None or carrier_mse is None or control_mse <= 0:
                continue
            deltas.append((float(control_mse) - float(carrier_mse)) / float(control_mse))
        else:
            value = row.get(delta_key)
            if value is not None:
                deltas.append(float(value))
    if not deltas:
        return {
            "n_strata_with_data": 0,
            "coefficient_of_variation": None,
            "range": None,
            "uniform_gain_flag": None,
        }
    array = np.asarray(deltas, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=0))
    coefficient = float(std / abs(mean)) if abs(mean) > 1.0e-12 else float("inf")
    return {
        "n_strata_with_data": int(array.size),
        "coefficient_of_variation": coefficient,
        "range": [float(array.min()), float(array.max())],
        "uniform_gain_flag": bool(coefficient < 0.25),
    }


def decompose_records(
    records: Sequence[QueryWindowRecord],
    *,
    carrier_checkpoint_sha256: str,
    control_checkpoint_sha256: str,
    query_window_identities: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("at least one query window record is required")
    session_names = sorted({record.session_name for record in records})
    assert_sessions_allowed(session_names)

    targets = np.stack([record.target for record in records], axis=0)
    global_target_mean = targets.mean(axis=0)
    identities = query_window_identities or [record.identity() for record in records]
    identity_hashes = [
        hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for identity in identities
    ]

    per_axis: dict[str, dict[str, Any]] = {}
    for axis in STRATUM_AXES:
        axis_rows = {
            label: summarize_axis_stratum(
                records,
                axis,
                label,
                global_target_mean=global_target_mean,
            )
            for label in _axis_labels(axis)
        }
        per_axis[axis] = {
            "strata": axis_rows,
            "uniform_gain_diagnostics": uniform_gain_diagnostics(
                axis_rows,
                delta_key=(
                    "relative_mse_improvement"
                    if axis == "target_direction"
                    else "paired_delta_primary"
                ),
            ),
        }

    overall_carrier = _summarize_arm(
        targets,
        np.stack([record.pred_carrier for record in records], axis=0),
        global_target_mean=global_target_mean,
        target_dirs_rad=[
            record.target_dir_rad
            if record.target_dir_rad is not None
            else derive_target_direction_rad(record.target)
            for record in records
        ],
    )
    overall_control = _summarize_arm(
        targets,
        np.stack([record.pred_control for record in records], axis=0),
        global_target_mean=global_target_mean,
        target_dirs_rad=[
            record.target_dir_rad
            if record.target_dir_rad is not None
            else derive_target_direction_rad(record.target)
            for record in records
        ],
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "sealed_test_sessions_opened": False,
        "carrier_checkpoint_sha256": carrier_checkpoint_sha256,
        "control_checkpoint_sha256": control_checkpoint_sha256,
        "sessions": session_names,
        "query_window_identities": identities,
        "query_window_identity_sha256": identity_hashes,
        "stratum_definitions": FROZEN_STRATUM_DEFINITIONS,
        "primary_score": PRIMARY_SCORE,
        "secondary_score": SECONDARY_SCORE,
        "global_target_mean": global_target_mean.tolist(),
        "overall": {
            "n_windows": len(records),
            "carrier": overall_carrier,
            "control": overall_control,
            "paired_delta_primary": (
                overall_carrier["common_denominator_mse"]
                - overall_control["common_denominator_mse"]
            ),
            "paired_delta_r2": (
                overall_carrier["within_stratum_r2"] - overall_control["within_stratum_r2"]
            ),
        },
        "axes": attach_multiplicity_correction(per_axis),
    }


def records_from_bundle_rows(rows: Sequence[Mapping[str, Any]]) -> list[QueryWindowRecord]:
    output: list[QueryWindowRecord] = []
    for row in rows:
        output.append(
            QueryWindowRecord(
                session_name=str(row["session_name"]),
                trial_index=int(row["trial_index"]),
                bin_index=int(row["bin_index"]),
                bins_in_trial=int(row["bins_in_trial"]),
                target=np.asarray(row["target"], dtype=np.float64),
                pred_carrier=np.asarray(row["pred_carrier"], dtype=np.float64),
                pred_control=np.asarray(row["pred_control"], dtype=np.float64),
                unit_modulation_m=np.asarray(row["unit_modulation_m"], dtype=np.float64),
                unit_activity=np.asarray(row["unit_activity"], dtype=np.float64),
                session_n_directions=int(row["session_n_directions"]),
                session_design_rank=int(row["session_design_rank"]),
                session_design_condition=float(row["session_design_condition"]),
                target_dir_rad=(
                    None if row.get("target_dir_rad") is None else float(row["target_dir_rad"])
                ),
            )
        )
    return output


def build_planted_direction_effect_fixture(
    *,
    seed: int,
    affected_directions: Sequence[int],
    carrier_gain: float = 0.8,
    noise_scale: float = 0.5,
    windows_per_direction: int = 24,
) -> list[QueryWindowRecord]:
    rng = np.random.default_rng(seed)
    records: list[QueryWindowRecord] = []
    for direction_index in range(TUNING_NUM_DIRECTIONS):
        angle = CANONICAL_DIRECTIONS_RAD[direction_index]
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
        for window_index in range(windows_per_direction):
            norm = 0.5 + 0.1 * window_index
            target = direction * norm
            noise = rng.normal(0.0, noise_scale, size=2)
            pred_control = target + noise
            if direction_index in affected_directions:
                pred_carrier = target + noise * (1.0 - carrier_gain)
            else:
                pred_carrier = pred_control.copy()
            records.append(
                QueryWindowRecord(
                    session_name=f"synthetic_session_{window_index % 3}",
                    trial_index=window_index,
                    bin_index=window_index % 9,
                    bins_in_trial=9,
                    target=target,
                    pred_carrier=pred_carrier,
                    pred_control=pred_control,
                    unit_modulation_m=np.full(4, 0.2 if direction_index in affected_directions else 0.01),
                    unit_activity=np.array([1.0, 0.5, 0.2, 0.1]),
                    session_n_directions=8,
                    session_design_rank=3,
                    session_design_condition=4.0,
                    target_dir_rad=angle,
                )
            )
    return records


def build_uniform_effect_fixture(*, seed: int, improvement: float = 0.35) -> list[QueryWindowRecord]:
    rng = np.random.default_rng(seed)
    records: list[QueryWindowRecord] = []
    for direction_index in range(TUNING_NUM_DIRECTIONS):
        angle = CANONICAL_DIRECTIONS_RAD[direction_index]
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
        for window_index in range(12):
            target = direction * (0.4 + 0.05 * window_index)
            noise = rng.normal(0.0, 1.0, size=2)
            pred_control = target + noise
            pred_carrier = target + noise * (1.0 - improvement)
            records.append(
                QueryWindowRecord(
                    session_name=f"synthetic_uniform_{window_index % 3}",
                    trial_index=window_index,
                    bin_index=window_index % 6,
                    bins_in_trial=6,
                    target=target,
                    pred_carrier=pred_carrier,
                    pred_control=pred_control,
                    unit_modulation_m=np.full(3, 0.1),
                    unit_activity=np.array([1.0, 0.8, 0.2]),
                    session_n_directions=6,
                    session_design_rank=3,
                    session_design_condition=8.0,
                    target_dir_rad=angle,
                )
            )
    return records


def build_all_null_direction_fixture(
    *,
    seed: int,
    windows_per_direction: int = 32,
) -> list[QueryWindowRecord]:
    """Carrier identical to control — all direction strata are null."""
    rng = np.random.default_rng(seed)
    records: list[QueryWindowRecord] = []
    for direction_index in range(TUNING_NUM_DIRECTIONS):
        angle = CANONICAL_DIRECTIONS_RAD[direction_index]
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
        for window_index in range(windows_per_direction):
            norm = 0.5 + 0.08 * window_index
            target = direction * norm
            noise = rng.normal(0.0, 0.6, size=2)
            prediction = target + noise
            records.append(
                QueryWindowRecord(
                    session_name=f"null_session_{window_index % 3}",
                    trial_index=window_index,
                    bin_index=window_index % 9,
                    bins_in_trial=9,
                    target=target,
                    pred_carrier=prediction.copy(),
                    pred_control=prediction.copy(),
                    unit_modulation_m=np.full(4, 0.05),
                    unit_activity=np.array([1.0, 0.5, 0.2, 0.1]),
                    session_n_directions=8,
                    session_design_rank=3,
                    session_design_condition=4.0,
                    target_dir_rad=angle,
                )
            )
    return records


def build_wrap_boundary_fixture() -> list[QueryWindowRecord]:
    angles = (-math.pi + 0.01, math.pi - 0.01)
    records: list[QueryWindowRecord] = []
    for index, angle in enumerate(angles):
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
        target = direction * 0.8
        records.append(
            QueryWindowRecord(
                session_name="wrap_session",
                trial_index=index,
                bin_index=0,
                bins_in_trial=3,
                target=target,
                pred_carrier=target + np.array([0.05, -0.02]),
                pred_control=target + np.array([0.10, -0.04]),
                unit_modulation_m=np.array([0.2]),
                unit_activity=np.array([1.0]),
                session_n_directions=8,
                session_design_rank=3,
                session_design_condition=3.0,
                target_dir_rad=angle,
            )
        )
    return records
