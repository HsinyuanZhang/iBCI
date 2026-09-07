"""V5 schema bridge for independently loaded scorer-adapter parity.

This module is imported only after v5 detached authorization and nonce claim. It
retains raw chronology returned by the score-only datamodule owner, then makes a
separate C1-builder copy where only exactly integral start/stop values are
converted to Python ints. It never recomputes bin coordinates, trial selection,
T4 rows, or post-50 valid starts.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
SCHEMA_BRIDGE_RULE = (
    "assert finite exact-integer int64-safe start/stop; cast only start/stop "
    "on an independent copy for the existing C1 calibration builder"
)


class SchemaBridgeError(RuntimeError):
    """Raw owner chronology cannot be safely presented to the C1 builder."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _raw_trial_copy(trial: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_scalar(value) for key, value in trial.items()}


def _checked_slice_index(value: Any, *, field: str, trial_position: int) -> int:
    if isinstance(value, bool):
        raise SchemaBridgeError(
            f"trial[{trial_position}] {field} must be numeric scalar, got {type(value).__name__}"
        )
    if isinstance(value, (int, np.integer)):
        # Do not route true integers through float: values above 2**53 would
        # lose precision, and a legal INT64_MAX could be spuriously rejected.
        integer = int(value)
    elif isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise SchemaBridgeError(f"trial[{trial_position}] {field} must be finite")
        if not numeric.is_integer():
            raise SchemaBridgeError(
                f"trial[{trial_position}] {field} must be exactly integral, got {numeric!r}"
            )
        integer = int(numeric)
    else:
        raise SchemaBridgeError(
            f"trial[{trial_position}] {field} must be numeric scalar, got {type(value).__name__}"
        )
    if integer < INT64_MIN or integer > INT64_MAX:
        raise SchemaBridgeError(
            f"trial[{trial_position}] {field} is outside signed int64 range"
        )
    checked = np.int64(integer)
    return int(checked)


def bridge_owner_chronology_for_c1_builder(
    raw_trials: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Preserve raw owner chronology and return an independent builder-safe copy.

    The caller list and its mapping objects are never modified. Only start and
    stop on the builder copy are cast after every value passes
    finite/exact-integer/int64-safe checks.
    """
    # Validate all owner slice coordinates before serializing the raw evidence.
    # In particular, NaN/inf must fail as schema values, rather than leaking a
    # JSON ValueError from the evidence serializer.  This validation is
    # read-only: the original owner mappings are still untouched when their
    # before-state evidence is captured below.
    checked_coordinates: list[tuple[int, int]] = []
    for position, trial in enumerate(raw_trials):
        if "start" not in trial or "stop" not in trial:
            raise SchemaBridgeError(f"trial[{position}] lacks start/stop")
        checked_coordinates.append(
            (
                _checked_slice_index(trial["start"], field="start", trial_position=position),
                _checked_slice_index(trial["stop"], field="stop", trial_position=position),
            )
        )
    raw_before = [_raw_trial_copy(trial) for trial in raw_trials]
    try:
        raw_before_sha256 = hashlib.sha256(_canonical_bytes(raw_before)).hexdigest()
    except (TypeError, ValueError) as exc:
        raise SchemaBridgeError("raw owner chronology is not canonical-JSON evidence") from exc
    bridged: list[dict[str, Any]] = []
    conversion_rows: list[dict[str, Any]] = []
    for position, trial in enumerate(raw_trials):
        builder_trial = dict(trial)
        raw_start, raw_stop = trial["start"], trial["stop"]
        builder_trial["start"], builder_trial["stop"] = checked_coordinates[position]
        if set(builder_trial) != set(trial):
            raise SchemaBridgeError(f"trial[{position}] builder key set changed")
        for key, original_value in trial.items():
            if key not in {"start", "stop"} and builder_trial[key] != original_value:
                raise SchemaBridgeError(
                    f"trial[{position}] builder changed non-slice field {key!r}"
                )
        bridged.append(builder_trial)
        conversion_rows.append(
            {
                "position": position,
                "raw_start_type": type(raw_start).__name__,
                "raw_stop_type": type(raw_stop).__name__,
                "raw_start": _json_scalar(raw_start),
                "raw_stop": _json_scalar(raw_stop),
                "builder_start": builder_trial["start"],
                "builder_stop": builder_trial["stop"],
            }
        )
    raw_after = [_raw_trial_copy(trial) for trial in raw_trials]
    try:
        raw_after_sha256 = hashlib.sha256(_canonical_bytes(raw_after)).hexdigest()
    except (TypeError, ValueError) as exc:
        raise SchemaBridgeError("raw owner chronology changed to non-canonical evidence") from exc
    if raw_after != raw_before or raw_after_sha256 != raw_before_sha256:
        raise SchemaBridgeError("raw owner chronology evidence mutated during bridge")
    trace = {
        "schema_bridge_rule": SCHEMA_BRIDGE_RULE,
        "raw_owner_chronology": raw_before,
        "raw_owner_chronology_sha256_before": raw_before_sha256,
        "raw_owner_chronology_sha256_after": raw_after_sha256,
        "builder_chronology_sha256": hashlib.sha256(
            _canonical_bytes([_raw_trial_copy(trial) for trial in bridged])
        ).hexdigest(),
        "start_stop_conversions": conversion_rows,
        "only_start_stop_cast": True,
        "bin_recomputation": False,
        "trial_selection_changed_by_schema_bridge": False,
        "t4_changed_by_schema_bridge": False,
        "query_valid_starts_changed_by_schema_bridge": False,
    }
    return raw_before, bridged, trace


def prepare_score_only_adapter_fixture_v5(
    root: Path,
    *,
    owners: Mapping[str, Any],
    v3: Any,
) -> tuple[Any, dict[str, Any]]:
    """Use score-only owners, applying only the verified start/stop bridge."""
    behavior_mean, behavior_std = v3._load_normalizer(
        root, v3.NORMALIZERS["behavior"], label="behavior"
    )
    side_mean, side_std = v3._load_normalizer(
        root, v3.NORMALIZERS["t4"], label="T4"
    )
    nwb_path = v3._pin_path(root, v3.DEV_SESSION, label="C1 consumed development NWB")
    record = owners["load_dandi688_session"](
        nwb_path,
        bin_size_ms=v3.BIN_SIZE_MS,
        window_size=v3.HISTORY_BINS,
        calibration_n_trials=v3.SUPPORT_TRIALS,
        max_trial_length=v3.TRIAL_LENGTH_BINS,
        pad_value=v3.PAD_VALUE,
        interpolate_trials=True,
        behavior_mean=behavior_mean,
        behavior_std=behavior_std,
        trial_result_filter="R",
        exclude_calibration_trials_from_windows=True,
        cache_dir=None,
        signal_view=v3.VIEW,
    )
    raw_owner_trials = owners["list_datamodule_rewarded_trials"](
        nwb_path,
        bin_size_ms=v3.BIN_SIZE_MS,
        window_size=v3.HISTORY_BINS,
        trial_result_filter="R",
    )
    raw_evidence, builder_trials, bridge_trace = bridge_owner_chronology_for_c1_builder(
        raw_owner_trials
    )
    indices = owners["selection"].select_calibration_trial_indices(
        builder_trials, v3.IDENTITY_TRIALS, v3.SUPPORT_TRIALS, "first"
    )
    c1 = owners["c1"]
    rebuild_record = {
        "neural": record.neural,
        "trials": builder_trials,
        "n_units": int(record.neural.shape[1]),
    }
    rebuilt_calibration = c1.build_calib_trials_for_indices(
        rebuild_record, indices, v3.IDENTITY_TRIALS
    )
    features, _metadata = owners["load_unit_side_features"](
        nwb_path,
        feature_group="t4",
        pool_size=v3.SUPPORT_TRIALS,
        mean=side_mean,
        std=side_std,
        cache_dir=None,
        permutation_seed=None,
        bin_size_ms=v3.BIN_SIZE_MS,
        window_size=v3.HISTORY_BINS,
        trial_result_filter="R",
        signal_view=v3.VIEW,
    )
    v3.require(features.shape == (record.neural.shape[1], 4), "adapter T4 shape drift")
    dataset = owners["MCMazeSessionDataset"](
        neural_data=record.neural,
        behavior_data=record.behavior,
        valid_starts=record.valid_starts,
        calib_trials=rebuilt_calibration,
        window_size=v3.HISTORY_BINS,
        session_name=record.name,
        side_features=features,
        electrode_ids=None,
    )
    v3.require(len(dataset) > 0, "adapter post-50 dataset is empty")
    fixture = v3.IndependentlyPreparedFixture(
        label="score_only_adapter_v5_schema_bridge",
        dataset=dataset,
        chronology=v3._canonical_trial_chronology(builder_trials),
        selection_indices=list(indices),
        neural=np.asarray(record.neural),
        behavior=np.asarray(record.behavior),
        calibration=np.asarray(rebuilt_calibration),
        valid_starts=np.asarray(record.valid_starts),
        side_features=v3._as_numpy(getattr(dataset, "side_features", None)),
        electrode_ids=v3._as_numpy(getattr(dataset, "electrode_ids", None)),
        first_batch=v3._first_batch_observation(dataset, owners),
    )
    bridge_trace.update(
        {
            "raw_owner_trial_count": len(raw_evidence),
            "c1_builder_trial_count": len(builder_trials),
            "selection_indices": list(indices),
            "support_trials": v3.SUPPORT_TRIALS,
            "identity_trials": v3.IDENTITY_TRIALS,
            "query_boundary": "owner-loader valid_starts after first 50 rewarded trials",
        }
    )
    return fixture, bridge_trace


def concrete_parity_after_future_authorization_v5(root: Path) -> dict[str, Any]:
    """Run corrected bridge then reuse v3 reference, observer, and forward."""
    from sua_exploration.mc_maze import subm_co_scorer_adapter_parity_v3 as v3

    root = root.resolve()
    v3._verify_owner_source_pins(root)
    owners = v3._runtime_owners()
    device = v3._configure_cpu_determinism(owners)
    v3.require(device.type == "cpu", "CPU-only execution required")
    reference = v3.prepare_c1_reference_fixture(root, owners=owners)
    adapter, bridge_trace = prepare_score_only_adapter_fixture_v5(root, owners=owners, v3=v3)
    observer = v3.SharedAdapterParityObserver()
    shared_input_trace = observer.compare_inputs(reference, adapter)
    checkpoint = v3._pin_path(root, v3.CHECKPOINT, label="fixed C1 checkpoint")
    teacher = v3._pin_path(root, v3.TEACHER, label="fixed C1 teacher checkpoint")
    model = owners["model"].load_frozen_model(
        checkpoint, teacher, "B3S", device, identity_mode="calibrated"
    )
    reference_result = v3._forward_and_observe(
        model, reference.dataset, owners=owners, observer=observer
    )
    adapter_result = v3._forward_and_observe(
        model, adapter.dataset, owners=owners, observer=observer
    )
    v3.compare_prediction_target_exact(reference_result, adapter_result)
    return {
        "status": "PARITY_CONFIRMED_CONSUMED_SUBC_DEV_SESSION_PENDING_SEPARATE_ROOT_REVIEW",
        "input_trace": {
            "shared_observer": shared_input_trace,
            "adapter_schema_bridge": bridge_trace,
        },
        "reference": {
            key: value
            for key, value in reference_result.items()
            if key not in {"prediction", "target"}
        },
        "adapter": {
            key: value
            for key, value in adapter_result.items()
            if key not in {"prediction", "target"}
        },
        "observer": asdict(observer),
        "external_subm_scoring_performed": False,
    }
