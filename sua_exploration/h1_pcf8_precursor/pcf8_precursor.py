"""Source-only constructibility audit for H-PCF8 = [W_i(7), b_i].

This is deliberately a forward *encoding* fit, not the existing H1 inverse
population carrier.  The fit is ``rate_i = b_i + velocity(7) @ W_i`` on
unprojected native 100-ms block rates and 100-ms mean H1 velocities.  It has
an eight-column design [1, v_1, ..., v_7], not the inverse B-by-17 design.
No decoder, target recording, R2, GPU, optimizer, or target backward pass is
available in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CHANNELS = 176
VELOCITY_DIM = 7
DESCRIPTOR_DIM = 8
SUPPORT_TRIALS = 4
DEAD_CHANNEL = 66
MACHINE_SCALE_FLOOR = 1e-12
NATIVE_PHASES = ("Reach", "Orient", "SnapTo", "Shape", "Grasp", "Carry", "Orient2", "Release")


class Pcf8PrecursorError(ValueError):
    """Fail-closed source boundary or numerical construction violation."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise Pcf8PrecursorError(message)


@dataclass(frozen=True)
class BlockTable:
    rates: np.ndarray       # [B,176], raw spikes / 0.1 s
    velocity: np.ndarray    # [B,7], native H1 velocity block mean
    phases: tuple[str | None, ...]
    trial_values: tuple[float, ...]

    def __post_init__(self) -> None:
        _need(self.rates.ndim == 2 and self.rates.shape[1] == CHANNELS, "rate table shape drift")
        _need(self.velocity.shape == (self.rates.shape[0], VELOCITY_DIM), "velocity table shape drift")
        _need(len(self.phases) == self.rates.shape[0], "phase row alignment drift")
        _need(np.isfinite(self.rates).all() and np.isfinite(self.velocity).all(), "non-finite source block")


@dataclass(frozen=True)
class ForwardFit:
    descriptor: np.ndarray  # [176,8] = [W(7), b]
    coefficients: np.ndarray  # [8,176], intercept then W rows
    degenerate_channels: tuple[int, ...]
    design: dict[str, Any]


def _json_number(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    value = np.asarray(values, dtype=np.float64)
    finite = value[np.isfinite(value)]
    if not finite.size:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None}
    return {key: float(item) for key, item in zip(("min", "q25", "median", "q75", "max"), np.quantile(finite, (0.0, .25, .5, .75, 1.0)))}


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    av, bv = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return None if denominator <= 1e-15 else float(np.dot(av, bv) / denominator)


def _design_summary(velocity: np.ndarray) -> dict[str, Any]:
    velocity = np.asarray(velocity, dtype=np.float64)
    _need(velocity.ndim == 2 and velocity.shape[1] == VELOCITY_DIM and velocity.shape[0] >= DESCRIPTOR_DIM,
          "PCF8 needs finite B-by-7 velocity with B >= 8")
    design = np.column_stack((np.ones(velocity.shape[0]), velocity))
    singular = np.linalg.svd(design, compute_uv=False)
    raw_condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else float("inf")
    centered = velocity - velocity.mean(axis=0, keepdims=True)
    velocity_singular = np.linalg.svd(centered, compute_uv=False)
    column_std = velocity.std(axis=0, ddof=0)
    return {
        "shape": [int(design.shape[0]), int(design.shape[1])],
        "rank": int(np.linalg.matrix_rank(design)),
        "raw_singular_values": [float(item) for item in singular],
        "raw_condition_number": _json_number(raw_condition),
        "velocity_centered_singular_values": [float(item) for item in velocity_singular],
        "velocity_column_std_native_units": [float(item) for item in column_std],
        "velocity_centered_component_fraction": [float(item * item / np.square(velocity_singular).sum()) for item in velocity_singular],
        "interpretation": "Raw design uses native loader velocity units without PCA, whitening, centering, or scale transform. Centered spectrum is a conditioning diagnostic only.",
    }


def fit_forward_pcf8(rates: np.ndarray, velocity: np.ndarray) -> ForwardFit:
    """Closed-form OLS per-channel forward fit; preserve literal zero rows."""
    rates = np.asarray(rates, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    _need(rates.ndim == 2 and rates.shape[1] == CHANNELS, "rates must be [B,176]")
    _need(velocity.shape == (rates.shape[0], VELOCITY_DIM), "velocity must align B-by-7 with rates")
    design_summary = _design_summary(velocity)
    _need(design_summary["rank"] == DESCRIPTOR_DIM, "PCF8 forward design is not full rank")
    coefficient, _, _, _ = np.linalg.lstsq(np.column_stack((np.ones(rates.shape[0]), velocity)), rates, rcond=None)
    degenerate = tuple(int(index) for index in np.flatnonzero(np.max(np.abs(rates - rates[0:1]), axis=0) <= 1e-14))
    # A silent/degenerated rate has no estimable affine functional identity.
    # Set its complete descriptor exactly to literal zero, including b.
    coefficient[:, list(degenerate)] = 0.0
    descriptor = np.concatenate((coefficient[1:].T, coefficient[:1].T), axis=1)
    _need(descriptor.shape == (CHANNELS, DESCRIPTOR_DIM) and np.isfinite(descriptor).all(), "PCF8 descriptor invalid")
    return ForwardFit(descriptor, coefficient, degenerate, design_summary)


def _phase_for_blocks(record: Any, selected_trials: Sequence[Any]) -> tuple[str | None, ...]:
    """Reconstruct native epoch membership for exact 100-ms block centers."""
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment error
        raise Pcf8PrecursorError("h5py is required to reconstruct native H1 phase metadata") from error
    path = Path(record.path).resolve()
    with h5py.File(path, "r") as handle:
        kin = handle["acquisition/OpenLoopKinematics"]
        if "timestamps" in kin:
            timestamps = np.asarray(kin["timestamps"][:], dtype=np.float64).reshape(-1)
        elif "starting_time" in kin:
            starting = kin["starting_time"]
            timestamps = float(starting[()]) + np.arange(record.neural.shape[0], dtype=np.float64) * float(starting.attrs["rate"])
        else:
            raise Pcf8PrecursorError(f"{record.session_name}: no reconstructible kinematics clock")
        epochs = handle["intervals/epochs"]
        starts = np.asarray(epochs["start_time"][:], dtype=np.float64).reshape(-1)
        stops = np.asarray(epochs["stop_time"][:], dtype=np.float64).reshape(-1)
        tags = tuple(value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in epochs["tags"][:])
    _need(timestamps.shape == record.trial_num.shape and starts.shape == stops.shape == (len(tags),),
          f"{record.session_name}: phase clock/epoch shape mismatch")
    selected: list[str | None] = []
    for trial in selected_trials:
        for block in trial.block_indices:
            center = float(timestamps[np.asarray(block, dtype=np.int64)].mean())
            hits = [tag for start, stop, tag in zip(starts, stops, tags) if tag in NATIVE_PHASES and start <= center < stop]
            selected.append(hits[0] if len(hits) == 1 else None)
    return tuple(selected)


def table_for_trials(record: Any, trials: Sequence[Any]) -> BlockTable:
    values = tuple(float(item.trial_number) for item in trials)
    _need(bool(trials), f"{record.session_name}: no selected trials")
    rates = np.concatenate([np.asarray(item.rates, dtype=np.float64) for item in trials], axis=0)
    velocity = np.concatenate([np.asarray(item.velocity, dtype=np.float64) for item in trials], axis=0)
    phases = _phase_for_blocks(record, trials)
    return BlockTable(rates=rates, velocity=velocity, phases=phases, trial_values=values)


def _per_column_comparison(a: ForwardFit, b: ForwardFit) -> dict[str, Any]:
    active = np.asarray([index for index in range(CHANNELS) if index not in set(a.degenerate_channels) | set(b.degenerate_channels)], dtype=np.int64)
    result: dict[str, Any] = {"active_channels": int(active.size), "columns": []}
    names = [f"W{index}" for index in range(VELOCITY_DIM)] + ["b"]
    for column, name in enumerate(names):
        result["columns"].append({
            "name": name,
            "support_reference_cosine_across_channels": _cosine(a.descriptor[active, column], b.descriptor[active, column]),
            "support_abs_magnitude": _quantiles(np.abs(a.descriptor[active, column])),
            "reference_abs_magnitude": _quantiles(np.abs(b.descriptor[active, column])),
        })
    row_cosines = [_cosine(a.descriptor[channel, :VELOCITY_DIM], b.descriptor[channel, :VELOCITY_DIM]) for channel in active]
    result["per_channel_W_cosine"] = _quantiles(np.asarray([np.nan if value is None else value for value in row_cosines]))
    return result


def _phase_stability(support: BlockTable, reference: BlockTable, global_support_fit: ForwardFit) -> dict[str, Any]:
    output: dict[str, Any] = {"source": "native_epochs_at_100ms_block_center", "phases": {}}
    for phase in NATIVE_PHASES:
        sidx = np.asarray([index for index, label in enumerate(support.phases) if label == phase], dtype=np.int64)
        ridx = np.asarray([index for index, label in enumerate(reference.phases) if label == phase], dtype=np.int64)
        row: dict[str, Any] = {"support_blocks": int(sidx.size), "reference_blocks": int(ridx.size)}
        if sidx.size >= DESCRIPTOR_DIM and ridx.size >= DESCRIPTOR_DIM:
            try:
                first = fit_forward_pcf8(support.rates[sidx], support.velocity[sidx])
                second = fit_forward_pcf8(reference.rates[ridx], reference.velocity[ridx])
                compare = _per_column_comparison(first, second)
                row.update({
                    "status": "FIT",
                    "support_design_rank": first.design["rank"],
                    "reference_design_rank": second.design["rank"],
                    "support_raw_condition_number": first.design["raw_condition_number"],
                    "reference_raw_condition_number": second.design["raw_condition_number"],
                    "support_reference_column_cosines": [item["support_reference_cosine_across_channels"] for item in compare["columns"]],
                    "phase_vs_global_support_W_column_cosines": [
                        _cosine(first.descriptor[:, column], global_support_fit.descriptor[:, column])
                        for column in range(VELOCITY_DIM)
                    ],
                })
            except Pcf8PrecursorError as error:
                row.update({"status": "RANK_OR_CONSTRUCTIBILITY_FAIL", "reason": str(error)})
        else:
            row.update({"status": "INSUFFICIENT_BLOCKS_FOR_Bx8"})
        output["phases"][phase] = row
    return output


def per_column_source_scale(descriptors: Iterable[np.ndarray], *, floor: float = MACHINE_SCALE_FLOOR) -> dict[str, Any]:
    values = np.asarray(tuple(np.asarray(item, dtype=np.float64) for item in descriptors), dtype=np.float64)
    _need(values.ndim == 3 and values.shape[1:] == (CHANNELS, DESCRIPTOR_DIM), "source descriptors must be [records,176,8]")
    _need(np.isfinite(values).all() and floor > 0.0, "invalid descriptor or scale floor")
    raw_sd = values.reshape(-1, DESCRIPTOR_DIM).std(axis=0, ddof=0)
    applied = np.maximum(raw_sd, floor)
    normalized = values / applied[None, None, :]
    zero_rows = np.all(values == 0.0, axis=2)
    _need(np.array_equal(normalized[zero_rows], np.zeros((int(zero_rows.sum()), DESCRIPTOR_DIM))),
          "non-centering scale failed literal-zero preservation")
    return {
        "formula": "descriptor / max(source_per_column_population_sd, machine_floor); no centering/subtraction",
        "raw_source_sd": [float(item) for item in raw_sd],
        "applied_scale": [float(item) for item in applied],
        "machine_floor": floor,
        "floor_required_by_observed_zero_or_subfloor_column": bool(np.any(raw_sd <= floor)),
        "columns_using_machine_floor": [int(item) for item in np.flatnonzero(raw_sd <= floor)],
        "floor_selection_status": "machine_safety_constant_only; any larger floor chosen after these source metrics would be SOURCE_SELECTED_NOT_PREREGISTERED_TARGET_EVIDENCE",
        "literal_zero_rows_preserved_exactly": True,
    }


def pcf8_arm_schema() -> dict[str, dict[str, str]]:
    return {
        "PCF8-FULL": {"carrier": "[W0,W1,W2,W3,W4,W5,W6,b]", "training": "separately trained", "role": "proposed forward functional carrier"},
        "PCF8-B8": {"carrier": "[0,0,0,0,0,0,0,b]", "training": "separately trained", "role": "rate-only width-8 control"},
        "PCF8-Z8": {"carrier": "literal zeros [0x8] at model boundary", "training": "separately trained", "role": "carrier-absence width-8 control"},
        "PCF8-RS": {"carrier": "complete descriptor-row permutation, nonidentity and deterministic per source/target record", "training": "separately trained", "role": "correct channel-attachment control"},
        "PCF8-LS": {"carrier": "refit PCF8 after one deterministic fixed-point-free permutation of concatenated 100-ms seven-dimensional velocity rows across the four support trials; neural rates unchanged", "training": "separately trained", "role": "strong wrong neural-kinematic pairing control"},
        "PCF8-FULL@RS/LS": {"carrier": "row shuffle or label-shuffle supplied to frozen PCF8-FULL checkpoint", "training": "no retraining", "role": "same-checkpoint attachment/pairing diagnostic only, never a trained endpoint control"},
        "HC4P8": {"carrier": "fresh current labeled H-C descriptor padded as [HC0,HC1,HC2,HC3,0,0,0,0]", "training": "separately trained", "role": "fresh width-8 parameter-matched labeled operational reference, not PCF8-FULL"},
    }


def pcf8_compute_contract() -> dict[str, Any]:
    b_cases = {"minimum": 558, "representative": 627, "maximum": 696}
    return {
        "carrier_state": {"cached_descriptor_floats": CHANNELS * DESCRIPTOR_DIM, "cached_descriptor_fp32_bytes": CHANNELS * DESCRIPTOR_DIM * 4},
        "calibration_minimal_streaming_sufficient_statistics": {"xtx_floats": DESCRIPTOR_DIM * DESCRIPTOR_DIM, "xty_floats": DESCRIPTOR_DIM * CHANNELS,
            "total_floats": DESCRIPTOR_DIM * DESCRIPTOR_DIM + DESCRIPTOR_DIM * CHANNELS,
            "total_fp32_bytes": (DESCRIPTOR_DIM * DESCRIPTOR_DIM + DESCRIPTOR_DIM * CHANNELS) * 4},
        "optional_raw_calibration_buffer_fp32_M4_documented_range": {name: {"blocks": b, "floats": b * (CHANNELS + VELOCITY_DIM), "bytes": b * (CHANNELS + VELOCITY_DIM) * 4} for name, b in b_cases.items()},
        "optional_raw_calibration_buffer_fp32_general": {"formula_floats": "B*(176+7)", "formula_bytes": "4*B*(176+7)", "M3_note": "M3 block counts are reported empirically per source recording by this audit; do not reuse M4 range as an organizer-held M3 claim."},
        "calibration_compute": {"accumulate_XtX": "O(B*8^2)", "accumulate_XtY": "O(B*8*176)", "factor_and_solve": "O(8^3 + 8^2*176)", "per_column_scale": "O(176*8)", "target_backpropagation": 0},
        "consumer_parameter_delta_if_current_HC_h32_topology_is_widened_4_to_8": {"changed_layer": "carrier_post_pool[0]: Linear(32+carrier_dim,32)", "delta_parameters": 128,
            "current_HC_carrier_parameters": 58140, "proposed_width8_carrier_parameters": 58268,
            "current_HC_whole_model_parameters": 10947836, "proposed_width8_whole_model_parameters": 10947964},
        "width_hygiene": "SUA T8 evidence cannot establish no H1 width effect; H1 PCF8 requires HC4P8 and all width-8 controls trained in the same H1 topology.",
    }


def _weak_direction_reliability(fit: ForwardFit, comparison: Mapping[str, Any]) -> dict[str, Any]:
    columns = list(comparison["columns"])
    return {
        "definition": "descriptive only: native velocity column SD, centered-spectrum fraction, and support-vs-later-reference coefficient cosine. No reliability cutoff is pre-registered or inferred from decoder R2.",
        "directions": [
            {
                "direction": f"W{index}",
                "native_velocity_sd": fit.design["velocity_column_std_native_units"][index],
                "support_centered_spectrum_fraction": fit.design["velocity_centered_component_fraction"][index],
                "support_later_reference_coefficient_cosine": columns[index]["support_reference_cosine_across_channels"],
            }
            for index in range(VELOCITY_DIM)
        ],
    }


def _budget_row(record: Any, *, budget: int, role: str) -> tuple[dict[str, Any], np.ndarray]:
    _need(len(record.trials) >= budget + 1, f"{record.session_name}: M={budget} support/later-reference boundary unavailable")
    support_trials, reference_trials = record.trials[:budget], record.trials[budget:]
    support = table_for_trials(record, support_trials)
    reference = table_for_trials(record, reference_trials)
    # M=3 intentionally uses an unequal 1-versus-2 chronological support
    # split; padding, copied trials, and trial reuse are prohibited.
    split = budget // 2
    first_trials, second_trials = support_trials[:split], support_trials[split:]
    _need(bool(first_trials) and bool(second_trials), "chronological support split unavailable")
    first_half = table_for_trials(record, first_trials)
    second_half = table_for_trials(record, second_trials)
    full_fit, reference_fit = fit_forward_pcf8(support.rates, support.velocity), fit_forward_pcf8(reference.rates, reference.velocity)
    first_fit, second_fit = fit_forward_pcf8(first_half.rates, first_half.velocity), fit_forward_pcf8(second_half.rates, second_half.velocity)
    comparison = _per_column_comparison(full_fit, reference_fit)
    return ({
        "role": role,
        "support_trials": [float(item.trial_number) for item in support_trials],
        "reference_trials_later_disjoint": [float(item.trial_number) for item in reference_trials],
        "support_blocks": int(support.rates.shape[0]), "reference_blocks": int(reference.rates.shape[0]),
        "support_fit": {"design": full_fit.design, "degenerate_channels": list(full_fit.degenerate_channels)},
        "reference_fit": {"design": reference_fit.design, "degenerate_channels": list(reference_fit.degenerate_channels)},
        "support_vs_later_reference": comparison,
        "weak_direction_reliability": _weak_direction_reliability(full_fit, comparison),
        "chronological_support_split": {
            "first_trials": [float(item.trial_number) for item in first_trials],
            "second_trials": [float(item.trial_number) for item in second_trials],
            "stability": _per_column_comparison(first_fit, second_fit),
        },
        "phase_conditioned_stability": _phase_stability(support, reference, full_fit),
        "phase_coverage": {phase: {"support_blocks": int(sum(value == phase for value in support.phases)), "reference_blocks": int(sum(value == phase for value in reference.phases))} for phase in NATIVE_PHASES},
        "unassigned_or_mixed_phase_blocks": {"support": int(sum(value is None for value in support.phases)), "reference": int(sum(value is None for value in reference.phases))},
    }, full_fit.descriptor)


def audit_source_records(records: Mapping[str, Any], expected_names: Sequence[str]) -> dict[str, Any]:
    """Run the 11-record, source-only PCF8 precursor without any decoder metric."""
    names = tuple(expected_names)
    _need(tuple(records) == names and len(names) == 11, "requires exact ordered 11-source-record scope")
    per_recording: dict[str, Any] = {}
    descriptors_m4: list[np.ndarray] = []
    descriptors_m3: list[np.ndarray] = []
    for name in names:
        record = records[name]
        _need(record.session_name == name and record.neural.shape[1] == CHANNELS, f"{name}: source identity width/session drift")
        _need(tuple(record.trial_values) == tuple(sorted(record.trial_values)), f"{name}: TrialNum order is not chronological")
        m4, descriptor_m4 = _budget_row(record, budget=4, role="primary_development_M4")
        m3, descriptor_m3 = _budget_row(record, budget=3, role="ancillary_organizer_held_M3")
        descriptors_m4.append(descriptor_m4)
        descriptors_m3.append(descriptor_m3)
        per_recording[name] = {
            "input_sha256": str(record.input_sha256),
            "M4": m4,
            "M3": m3,
        }
    return {
        "schema": "h1_pcf8_source_constructibility_precursor_v1",
        "status": "SOURCE_CPU_AUDIT_COMPLETE__NO_GPU_AUTHORIZATION",
        "scope": {"source_recordings_opened": 11, "target_recordings_opened": 0, "target_decoder_r2_computed": False, "gpu_used": False, "target_backward_steps": 0},
        "fit_contract": {"form": "rate_i = b_i + sum_j W_ij * velocity_j + epsilon", "design": "[ones, native_100ms_mean_velocity_7d] => Bx8", "lag_bins": 0,
            "rate_preprocessing": "raw neural spikes summed over each contiguous eval-valid five-bin block and divided by BLOCK_SECONDS=0.1", "kinematic_preprocessing": "TrialBlocks.velocity: arithmetic mean of the corresponding five native H1 velocity bins; no PCA, whitening, centering, standardization, or unit conversion in the fit"},
        "source_boundary": "exact H1_M4_FOLD0_SOURCE 11 recordings; primary M4 support=first four chronological eval-valid TrialNum trials/reference=all later disjoint legal trials; ancillary organizer-held M3 support=first three trials/reference=all later disjoint legal trials beginning at fourth",
        "per_recording": per_recording,
        "source_per_column_scale": {"M4": per_column_source_scale(descriptors_m4), "M3": per_column_source_scale(descriptors_m3)},
        "future_arm_schema": pcf8_arm_schema(),
        "state_compute_contract": pcf8_compute_contract(),
        "interpretation_guard": "This precursor reports constructibility/conditioning/reliability only. It contains no overlap-residual-to-decoder-R2 prediction and authorizes no GPU cell.",
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
