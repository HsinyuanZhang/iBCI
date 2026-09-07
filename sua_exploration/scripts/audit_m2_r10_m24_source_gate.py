#!/usr/bin/env python3
"""Guarded M2 R10@M24 source-only CPU gate.

The prelaunch path performs only synthetic/static work. Native source files are
opened solely after a hash-bound root review and exact confirmation phrase.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUA = ROOT / "sua_exploration"
SCE = ROOT / "streaming_calibration_exp"
for path in (SUA, SCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mc_maze.m2_r10_m24_source_gate import (  # noqa: E402
    BIN_SECONDS,
    CHANNELS,
    LAGS,
    RANDOM_SCHEDULES,
    RIDGE,
    SEED,
    TARGET_WIDTH,
    THINNING_REPEATS,
    UINT32_MAX,
    WIDTH,
    future_autocorr,
    l10,
    mc,
    null_features,
    paired,
    r10,
    schedule_receipt,
    score_loso,
    split_reliability,
    streaming_receipt,
)

MANIFEST = SUA / "manifests/m2_r10_m24_source_v1_manifest.json"
PROTOCOL = SUA / "docs/M2_R10_M24_SOURCE_VALIDATION_PROTOCOL.md"
PRIOR = SUA / "results/m2_carrier_stability_factorial_v1/protocol_receipt.json"
PURE = SUA / "mc_maze/m2_r10_m24_source_gate.py"
DATA_MODULE = ROOT / "streaming_calibration_exp/src/data/falcon_datamodule.py"
PRE = SUA / "results/m2_r10_m24_source_prelaunch_v5"
OUT = SUA / "results/m2_r10_m24_source_gate_v2"
REVIEW = "root_approved_m2_r10_m24_source_v2"
CONFIRM = "I_CONFIRM_EXACT_SEVEN_M2_SOURCE_NWBS_CPU_ONLY_R10_M24_V2"
PRIOR_SHA256 = "b094e3f7674dd75571947c262d63879639c8b41ffe99f6725b4e1764e1e0e573"
DEADLINE_SECONDS = 86_400.0


class GateDeadline(TimeoutError):
    """Raised when the frozen CPU budget is exceeded."""

    def __init__(self, stage: str):
        super().__init__(f"24-hour CPU deadline exceeded at {stage}")
        self.stage = stage


class SourceGuardFailure(PermissionError):
    """Authorization failed before any source loader could be called."""

    def __init__(self, cause: Exception):
        super().__init__(str(cause))
        self.cause_type = type(cause).__name__
        self.stage = "pre_source_guard"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_sha(path: Path) -> str | None:
    try:
        return sha(path)
    except (OSError, ValueError):
        return None


def array_sha(value: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(value).astype(dtype, copy=False))
    return hashlib.sha256(array.tobytes()).hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return json_safe(vars(value))
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def manifest() -> dict[str, Any]:
    specification = json.loads(MANIFEST.read_text())
    prior = json.loads(PRIOR.read_text())
    if specification.get("schema_version") != "m2_r10_m24_source_manifest_v1":
        raise ValueError("manifest schema mismatch")
    if sha(PRIOR) != PRIOR_SHA256:
        raise ValueError("prior frozen input receipt mismatch")
    rows = specification.get("sessions")
    prior_rows = prior.get("inputs")
    if not isinstance(rows, list) or len(rows) != 7:
        raise ValueError("manifest requires seven sessions")
    if not isinstance(prior_rows, list) or len(prior_rows) != 7:
        raise ValueError("prior receipt requires seven sessions")
    seen: set[str] = set()
    prohibited = ("held-out", "minival", "test", "formal", "report", "evalai")
    for row, prior_row in zip(rows, prior_rows):
        session = str(row.get("session", ""))
        relative_path = str(row.get("relative_path", ""))
        prior_path = str(prior_row.get("path", ""))
        if (
            not session
            or session in seen
            or session != str(prior_row.get("session", ""))
            or str(row.get("sha256", "")) != str(prior_row.get("sha256", ""))
            or not prior_path.endswith(relative_path)
            or not relative_path.startswith(
                "SPINT-main/data/000953/sub-MonkeyN-held-in-calib/"
            )
            or any(token in relative_path.lower() for token in prohibited)
        ):
            raise ValueError("exact seven-source list mismatch or prohibited scope")
        seen.add(session)
    return specification


def synthetic() -> dict[str, tuple[np.ndarray, ...]]:
    base_time = np.arange(1024, dtype=np.uint16)[:, None]
    base_channel = 3 * np.arange(CHANNELS, dtype=np.uint16)[None, :]
    return {
        row["session"]: tuple(
            ((base_time + base_channel + session_index + trial_index) % 7).astype(np.uint8)
            for trial_index in range(24)
        )
        for session_index, row in enumerate(manifest()["sessions"])
    }


def bench() -> dict[str, Any]:
    support = synthetic()
    names = tuple(support)
    features = {name: r10(trials) for name, trials in support.items()}
    targets = {name: future_autocorr(trials) for name, trials in support.items()}

    start = time.perf_counter()
    score_loso(features, targets)
    canonical_seconds = time.perf_counter() - start

    start = time.perf_counter()
    score_loso(null_features(features, 1), targets)
    one_null_seconds = time.perf_counter() - start

    start = time.perf_counter()
    split_reliability(support[names[0]], names[0], repeats=8)
    reliability_seconds = time.perf_counter() - start

    start = time.perf_counter()
    plan = schedule_receipt(names, CHANNELS)
    schedule_seconds = time.perf_counter() - start

    projection = (
        canonical_seconds
        + RANDOM_SCHEDULES * one_null_seconds
        + 7 * reliability_seconds * (THINNING_REPEATS / 8)
        + schedule_seconds
    )
    return {
        "fixed_cpu_workers": 1,
        "canonical_seconds": canonical_seconds,
        "one_null_seconds": one_null_seconds,
        "reliability_8_repeats_one_session_seconds": reliability_seconds,
        "full_schedule_receipt_seconds": schedule_seconds,
        "formal_schedule_plan_sha256": plan["full_plan_sha256"],
        "formal_schedule_complete": plan["all_vectors_complete_permutations"],
        "projection_seconds": projection,
        "within_24h": projection <= DEADLINE_SECONDS,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def receipt() -> dict[str, Any]:
    specification = manifest()
    benchmark = bench()
    if not benchmark["within_24h"]:
        raise RuntimeError("r10_null_budget_infeasible_stop")
    return {
        "schema_version": "m2_r10_m24_source_prelaunch_v5",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "no-NWB root-review-only M2 R10 M24 CPU gate prelaunch",
        "manifest": {
            "path": str(MANIFEST.relative_to(ROOT)),
            "sha256": sha(MANIFEST),
            "sessions": specification["sessions"],
        },
        "prior_input_receipt": {
            "path": str(PRIOR.relative_to(ROOT)),
            "sha256": sha(PRIOR),
        },
        "protocol": {"path": str(PROTOCOL.relative_to(ROOT)), "sha256": sha(PROTOCOL)},
        "code": {
            "runner": sha(Path(__file__)),
            "pure": sha(PURE),
            "production_falcon_dataset": sha(DATA_MODULE),
        },
        "endpoint": {
            "support": [0, 24],
            "future": [24, "end"],
            "lags": list(LAGS),
            "ridge": RIDGE,
            "outer_loso": 7,
            "target": "future trial-demeaned within-trial autocorrelation only scorer",
            "trialization": "NWB trials start/stop geometry on acquisition timestamps; production FalconDataset",
            "behavior_values_read": False,
            "nwb_eval_mask_read": False,
        },
        "formal_null": {
            "seed_namespace": SEED,
            "distribution": "independent uniform full S_96 per session",
            "random_schedules": RANDOM_SCHEDULES,
            "fixed_points_identity_draws_and_duplicates_retained": True,
            "full_plan_sha256": benchmark["formal_schedule_plan_sha256"],
        },
        "state": streaming_receipt(CHANNELS),
        "benchmark": benchmark,
        "hard_exclusions": {
            "nwb_opened": False,
            "cuda": False,
            "held_out": False,
            "decoder": False,
            "gpu": False,
            "evalai": False,
        },
        "guard": {
            "authorization": REVIEW,
            "confirmation": CONFIRM,
            "fresh_output": True,
        },
        "status": "pending_root_review_no_nwb_opened",
    }


def _write_json_artifact(output: Path, name: str, value: Mapping[str, Any]) -> Path:
    if output.exists():
        raise FileExistsError(output)
    payload = json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n"
    output.mkdir(parents=True)
    path = output / name
    path.write_text(payload)
    (output / f"{Path(name).stem}.sha256").write_text(f"{sha(path)}  {name}\n")
    return path


def write_prelaunch(output: Path) -> Path:
    return _write_json_artifact(output, "prelaunch_receipt.json", receipt())


def _expected_hashes() -> dict[str, str]:
    return {
        "protocol_sha256": sha(PROTOCOL),
        "manifest_sha256": sha(MANIFEST),
        "prior_input_receipt_sha256": sha(PRIOR),
        "falcon_dataset_sha256": sha(DATA_MODULE),
        "runner_sha256": sha(Path(__file__)),
        "pure_sha256": sha(PURE),
    }


def _formal_schedule_contract() -> dict[str, Any]:
    names = tuple(row["session"] for row in manifest()["sessions"])
    plan = schedule_receipt(names, CHANNELS)
    if not plan["all_vectors_complete_permutations"]:
        raise RuntimeError("formal schedule construction is not bijective")
    return {
        "seed_namespace": SEED,
        "distribution": "independent uniform full S_96 per session",
        "random_schedules": RANDOM_SCHEDULES,
        "fixed_points_identity_draws_and_duplicates_retained": True,
        "full_plan_sha256": plan["full_plan_sha256"],
    }


def check(review: Path, prelaunch: Path) -> dict[str, str]:
    root_review = json.loads(review.read_text())
    prelaunch_receipt = json.loads(prelaunch.read_text())
    expected = _expected_hashes()
    if root_review.get("authorization") != REVIEW:
        raise PermissionError("root authorization token mismatch")
    if root_review.get("prelaunch_receipt_sha256") != sha(prelaunch):
        raise PermissionError("root authorization/prelaunch mismatch")
    for key, expected_value in expected.items():
        if root_review.get(key) != expected_value:
            raise PermissionError(f"root review binding mismatch: {key}")
    if prelaunch_receipt.get("schema_version") != "m2_r10_m24_source_prelaunch_v5":
        raise PermissionError("prelaunch schema mismatch")
    if prelaunch_receipt.get("status") != "pending_root_review_no_nwb_opened":
        raise PermissionError("prelaunch status mismatch")
    formal_contract = _formal_schedule_contract()
    if prelaunch_receipt.get("formal_null") != formal_contract:
        raise PermissionError("stale or malformed formal-null prelaunch contract")
    if root_review.get("formal_schedule_plan_sha256") != formal_contract["full_plan_sha256"]:
        raise PermissionError("root review formal schedule binding mismatch")
    if prelaunch_receipt.get("protocol", {}).get("sha256") != expected["protocol_sha256"]:
        raise PermissionError("stale prelaunch protocol")
    if prelaunch_receipt.get("manifest", {}).get("sha256") != expected["manifest_sha256"]:
        raise PermissionError("stale prelaunch manifest")
    if (
        prelaunch_receipt.get("prior_input_receipt", {}).get("sha256")
        != expected["prior_input_receipt_sha256"]
    ):
        raise PermissionError("stale prelaunch prior receipt")
    code = prelaunch_receipt.get("code", {})
    if code.get("runner") != expected["runner_sha256"]:
        raise PermissionError("stale prelaunch runner")
    if code.get("pure") != expected["pure_sha256"]:
        raise PermissionError("stale prelaunch pure module")
    if code.get("production_falcon_dataset") != expected["falcon_dataset_sha256"]:
        raise PermissionError("stale prelaunch FalconDataset")
    hard_exclusions = prelaunch_receipt.get("hard_exclusions", {})
    if any(hard_exclusions.get(key) is not False for key in ("nwb_opened", "cuda", "held_out", "decoder", "gpu", "evalai")):
        raise PermissionError("prelaunch exclusion contract mismatch")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise PermissionError("CUDA is not disabled")
    return expected


def _geometry_trial_mask(
    timestamps: np.ndarray, trial_starts: np.ndarray, trial_stops: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Construct trial inclusion from time geometry only, never from eval_mask/behavior values."""
    time = np.asarray(timestamps, dtype=np.float64)
    starts = np.asarray(trial_starts, dtype=np.float64)
    stops = np.asarray(trial_stops, dtype=np.float64)
    if (
        time.ndim != 1
        or starts.ndim != 1
        or stops.ndim != 1
        or starts.shape != stops.shape
        or len(starts) < 1
        or not np.isfinite(time).all()
        or not np.isfinite(starts).all()
        or not np.isfinite(stops).all()
        or np.any(np.diff(time) <= 0)
        or np.any(np.diff(starts) <= 0)
        or np.any(stops <= starts)
        or np.any(starts[1:] < stops[:-1])
    ):
        raise ValueError("invalid timestamp/trial geometry")
    start_indices = np.searchsorted(time, starts, side="left")
    stop_indices = np.searchsorted(time, stops, side="left")
    if (
        np.any(start_indices < 0)
        or np.any(stop_indices > len(time))
        or np.any(start_indices >= stop_indices)
        or len(np.unique(start_indices)) != len(start_indices)
    ):
        raise ValueError("trial geometry cannot be represented on neural bins")
    trial_change = np.zeros(len(time), dtype=bool)
    inclusion = np.zeros(len(time), dtype=bool)
    for start, stop in zip(start_indices, stop_indices):
        trial_change[int(start)] = True
        inclusion[int(start) : int(stop)] = True
    receipt = {
        "trial_count": int(len(starts)),
        "start_index_sha256": array_sha(start_indices, "<i8"),
        "stop_index_sha256": array_sha(stop_indices, "<i8"),
        "geometry_mask_sha256": array_sha(inclusion, "u1"),
        "geometry_mask_true_bins": int(inclusion.sum()),
    }
    return trial_change, inclusion, receipt


def _load_m2_neural_geometry(
    path: Path, _task: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Read only units, the M2 time base, and trial start/stop geometry.

    It intentionally never reads finger-velocity values or acquisition/eval_mask.
    """
    import pandas as pd
    from pynwb import NWBHDF5IO
    from falcon_challenge.dataloaders import bin_units

    with NWBHDF5IO(str(path), "r") as io:
        nwbfile = io.read()
        # Build the minimal interface expected by the official binning helper;
        # do not materialize electrode, waveform, SNR, or other Units columns.
        units = pd.DataFrame(
            {
                "spike_times": [
                    np.asarray(nwbfile.units.get_unit_spike_times(index), dtype=np.float64)
                    for index in range(len(nwbfile.units.id))
                ]
            }
        )
        velocity_container = nwbfile.acquisition["finger_vel"]
        labels = list(velocity_container.time_series)
        if not labels:
            raise ValueError("M2 time-base acquisition has no time series")
        reference_timestamps = np.asarray(
            velocity_container.get_timeseries(labels[0]).timestamps[:], dtype=np.float64
        )
        for label in labels[1:]:
            candidate = np.asarray(
                velocity_container.get_timeseries(label).timestamps[:], dtype=np.float64
            )
            if not np.array_equal(candidate, reference_timestamps):
                raise ValueError("M2 acquisition time bases differ")
        # Access only time geometry; do not materialize result/target/reward columns.
        trial_starts = np.asarray(nwbfile.trials["start_time"].data[:], dtype=np.float64)
        trial_stops = np.asarray(nwbfile.trials["stop_time"].data[:], dtype=np.float64)
        neural = bin_units(
            units,
            bin_size_s=BIN_SECONDS,
            bin_timestamps=reference_timestamps,
            is_timestamp_bin_start=True,
        )
    trial_change, geometry_mask, geometry_receipt = _geometry_trial_mask(
        reference_timestamps, trial_starts, trial_stops
    )
    placeholder_covariates = np.zeros((len(reference_timestamps), 1), dtype=np.float32)
    return neural, placeholder_covariates, trial_change, geometry_mask, geometry_receipt


def _load_after_review(
    specification: Mapping[str, Any],
    *,
    load_nwb_fn: Callable[..., tuple[Any, Any, Any, Any]] | None = None,
    dataset_cls: type | None = None,
    falcon_task: Any | None = None,
) -> tuple[dict[str, tuple[np.ndarray, ...]], dict[str, dict[str, Any]]]:
    """Deferred exact-path production trializer with injectable test dependencies."""
    production_geometry_loader = load_nwb_fn is None
    if load_nwb_fn is None:
        load_nwb_fn = _load_m2_neural_geometry
    if dataset_cls is None:
        from src.data.falcon_datamodule import FalconDataset
        dataset_cls = FalconDataset
    if falcon_task is None:
        falcon_task = "m2_neural_geometry_only"

    rows = specification.get("sessions")
    if not isinstance(rows, list) or len(rows) != 7:
        raise ValueError("loader requires seven manifest sessions")
    source: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
    for row in rows:
        path = ROOT / row["relative_path"]
        if not path.is_file() or sha(path) != row["sha256"]:
            raise ValueError("source file/hash failure")
        loaded = load_nwb_fn(path, falcon_task)
        if len(loaded) == 5:
            (
                neural,
                placeholder_or_discarded_covariates,
                trial_change,
                geometry_mask,
                geometry_receipt,
            ) = loaded
        elif len(loaded) == 4:
            neural, placeholder_or_discarded_covariates, trial_change, geometry_mask = loaded
            geometry_receipt = {
                "test_dependency_injection": True,
                "geometry_mask_sha256": array_sha(np.asarray(geometry_mask, dtype=bool), "u1"),
            }
        else:
            raise ValueError("geometry loader return contract")
        neural_array = np.asarray(neural)
        covariate_array = np.asarray(placeholder_or_discarded_covariates)
        trial_change_array = np.asarray(trial_change, dtype=bool)
        geometry_mask_array = np.asarray(geometry_mask, dtype=bool)
        if (
            neural_array.ndim != 2
            or neural_array.shape[1] != CHANNELS
            or covariate_array.ndim != 2
            or covariate_array.shape[0] != neural_array.shape[0]
            or trial_change_array.shape != (neural_array.shape[0],)
            or geometry_mask_array.shape != (neural_array.shape[0],)
        ):
            raise ValueError("raw loader shape contract")
        if (
            not np.isfinite(neural_array).all()
            or np.any(neural_array < 0)
            or not np.equal(neural_array, np.floor(neural_array)).all()
            or np.any(neural_array > UINT32_MAX)
        ):
            raise ValueError("raw neural integer-count contract")
        # Official FALCON binning may return uint8. Production FalconDataset
        # pads calibration trials with -1, so convert only after validating
        # exact counts; otherwise NumPy correctly rejects -1 for uint8.
        signed_neural = neural_array.astype(np.int64)
        source[row["session"]] = {
            "neural": signed_neural,
            "covariates": np.zeros_like(covariate_array),
            "trial_change": trial_change_array,
            "eval_mask": geometry_mask_array,
        }
        del placeholder_or_discarded_covariates, covariate_array

    dataset = dataset_cls(
        sessions_dict=source,
        calib_sessions_dict=source,
        window_size=50,
        split="m2_r10_source_only",
        calibration_n_trials=24,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=1024,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        remove_still_times=False,
        remove_calib_still_times=False,
        use_calib_active_segments=False,
        interpolate_trials=False,
        interpolate_trials_kind="linear",
        pad_value=-1.0,
        side_feature_group="none",
        query_start_trial=0,
        allow_empty_query_sessions=True,
    )

    output: dict[str, tuple[np.ndarray, ...]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for row in rows:
        session = row["session"]
        padded = np.asarray(dataset.calib_trialized_neural[session])
        raw_lengths = np.asarray(dataset.calib_trial_lengths[session])
        production_sums = np.asarray(dataset.calib_trial_spike_sums[session])
        if (
            raw_lengths.ndim != 1
            or not np.isfinite(raw_lengths).all()
            or not np.equal(raw_lengths, np.floor(raw_lengths)).all()
        ):
            raise ValueError("production trial-length integer contract")
        lengths = raw_lengths.astype(np.int64)
        if (
            padded.ndim != 3
            or padded.shape[0] != lengths.shape[0]
            or padded.shape[2] != CHANNELS
            or len(lengths) <= 24
            or np.any(lengths <= 0)
            or np.any(lengths >= 1024)
            or production_sums.shape != (len(lengths), CHANNELS)
        ):
            raise ValueError("production raw trial shape or truncation contract")
        trials: list[np.ndarray] = []
        for trial_index, length_value in enumerate(lengths):
            length = int(length_value)
            if length > padded.shape[1]:
                raise ValueError("reported trial length exceeds padded tensor")
            prefix = padded[trial_index, :length]
            tail = padded[trial_index, length:]
            if (
                not np.isfinite(prefix).all()
                or np.any(prefix < 0)
                or not np.equal(prefix, np.floor(prefix)).all()
                or (tail.size and not np.array_equal(tail, np.full_like(tail, -1)))
            ):
                raise ValueError("raw prefix or padding contract")
            prefix_integer = prefix.astype(np.int64)
            prefix_sum = prefix_integer.sum(axis=0, dtype=np.uint64)
            production_sum = production_sums[trial_index]
            if (
                not np.isfinite(production_sum).all()
                or np.any(production_sum < 0)
                or not np.equal(production_sum, np.floor(production_sum)).all()
            ):
                raise ValueError("production spike-sum integer contract")
            rounded_production = production_sum.astype(np.uint64)
            if np.any(prefix_sum > UINT32_MAX) or not np.array_equal(prefix_sum, rounded_production):
                raise ValueError("prefix sum, production sum, or uint32 contract")
            trials.append(prefix_integer)
        output[session] = tuple(trials)
        receipts[session] = {
            "source_path": str(row["relative_path"]),
            "source_sha256": str(row["sha256"]),
            "trial_count": int(len(lengths)),
            "trial_lengths": lengths.tolist(),
            "all_integer_prefixes": True,
            "all_padding_tails_exact_minus_one": True,
            "all_prefix_sums_match_production": True,
            "all_trial_totals_fit_uint32": True,
            "no_trial_length_reaches_1024": True,
            "behavior_covariates_discarded_before_dataset": True,
            "behavior_values_read_by_production_loader": False,
            "nwb_eval_mask_read_by_production_loader": False,
            "trial_inclusion_mask_source": "NWB trials start/stop on finger_vel timestamps",
            "production_geometry_only_loader": production_geometry_loader,
            "signed_neural_dtype_before_falcon_dataset": str(signed_neural.dtype),
            "geometry_receipt": geometry_receipt,
            "channels": CHANNELS,
        }
    return output, receipts


class StageTimer:
    def __init__(self, deadline_seconds: float):
        self.started = time.perf_counter()
        self.previous = self.started
        self.deadline_seconds = deadline_seconds
        self.stages: dict[str, dict[str, float]] = {}

    def mark(self, stage: str) -> float:
        now = time.perf_counter()
        cumulative = now - self.started
        self.stages[stage] = {
            "duration": now - self.previous,
            "cumulative": cumulative,
        }
        self.previous = now
        if cumulative > self.deadline_seconds:
            raise GateDeadline(stage)
        return cumulative

    def check(self, stage: str) -> float:
        cumulative = time.perf_counter() - self.started
        if cumulative > self.deadline_seconds:
            raise GateDeadline(stage)
        return cumulative

    def total(self) -> float:
        return time.perf_counter() - self.started


def _bindings(prelaunch: Path, review: Path) -> dict[str, str | None]:
    return {
        "protocol": safe_sha(PROTOCOL),
        "manifest": safe_sha(MANIFEST),
        "prior_input_receipt": safe_sha(PRIOR),
        "falcon_dataset": safe_sha(DATA_MODULE),
        "runner": safe_sha(Path(__file__)),
        "pure": safe_sha(PURE),
        "prelaunch": safe_sha(prelaunch),
        "root_review": safe_sha(review),
    }


def _target_receipt(target: Mapping[str, np.ndarray]) -> dict[str, Any]:
    values = np.asarray(target["target"], dtype=np.float64)
    mask = np.asarray(target["mask"], dtype=bool)
    variance = np.asarray(target["variance"], dtype=np.float64)
    exposure = np.asarray(target["pair_exposure"], dtype=np.int64)
    return {
        "target_shape": list(values.shape),
        "mask_shape": list(mask.shape),
        "pair_exposure": exposure,
        "defined_fractions": np.asarray(target["defined_row_fraction_by_lag"]),
        "defined_row_count": int(mask.all(axis=1).sum()),
        "mask": mask,
        "mask_sha256": array_sha(mask, "u1"),
        "target_sha256": array_sha(values, "<f8"),
        "pair_exposure_sha256": array_sha(exposure, "<i8"),
        "future_variance_min": float(variance.min()),
        "future_variance_max": float(variance.max()),
        "mask_equals_target_finiteness": bool(np.array_equal(mask, np.isfinite(values))),
        "whole_row_mask": bool(
            np.array_equal(mask, np.repeat(mask.all(axis=1)[:, None], TARGET_WIDTH, axis=1))
        ),
    }


def _write_source_result(output: Path, result: Mapping[str, Any]) -> Path:
    return _write_json_artifact(output, "source_gate.json", result)


def _finish(
    output: Path,
    result: dict[str, Any],
    timer: StageTimer,
    prelaunch: Path,
    review: Path,
) -> Path:
    timer.check("pre_final_guard")
    check(review, prelaunch)
    result["binding"] = _bindings(prelaunch, review)
    result["runtime"] = {"stages": timer.stages, "total_seconds": timer.total()}
    return _write_source_result(output, result)


def run(
    output: Path,
    prelaunch: Path,
    review: Path,
    *,
    deadline_seconds: float = DEADLINE_SECONDS,
    loader: Callable[..., tuple[dict[str, tuple[np.ndarray, ...]], dict[str, dict[str, Any]]]] = _load_after_review,
) -> Path:
    if output.exists():
        raise FileExistsError(output)
    timer = StageTimer(deadline_seconds)
    # The callable API itself owns the source-opening guard; callers cannot
    # bypass authorization by invoking run()/execute_with_terminal directly.
    try:
        check(review, prelaunch)
    except Exception as error:
        raise SourceGuardFailure(error) from error
    timer.mark("pre_source_guard")
    specification = manifest()
    raw, raw_receipts = loader(specification)
    timer.mark("source_load")
    names = tuple(sorted(raw))
    if names != tuple(sorted(row["session"] for row in specification["sessions"])):
        raise ValueError("loaded session set differs from manifest")
    support = {session: trials[:24] for session, trials in raw.items()}
    features = {session: r10(trials) for session, trials in support.items()}
    controls = {session: l10(trials) for session, trials in support.items()}
    timer.mark("support_features")
    targets = {session: future_autocorr(trials[24:]) for session, trials in raw.items()}
    target_receipts = {session: _target_receipt(targets[session]) for session in names}
    timer.mark("future_target")

    r10_scores = score_loso(features, targets)
    l10_scores = score_loso(controls, targets)
    comparison = paired(
        [float(row["r2"]) for row in r10_scores],
        [float(row["r2"]) for row in l10_scores],
    )
    timer.mark("aligned_screen")

    base_result: dict[str, Any] = {
        "schema_version": "m2_r10_m24_source_gate_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "binding": _bindings(prelaunch, review),
        "input_sessions": raw_receipts,
        "target_receipts": target_receipts,
        "state": streaming_receipt(CHANNELS),
        "canonical": {"R10": r10_scores, "L10": l10_scores, "content": comparison},
        "contracts": {
            "source_only": True,
            "held_out": False,
            "gpu": False,
            "decoder": False,
            "evalai": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "behavior_covariates_discarded_before_dataset": all(
                receipt["behavior_covariates_discarded_before_dataset"]
                for receipt in raw_receipts.values()
            ),
            "behavior_values_read_by_production_loader": any(
                receipt["behavior_values_read_by_production_loader"]
                for receipt in raw_receipts.values()
            ),
            "nwb_eval_mask_read_by_production_loader": any(
                receipt["nwb_eval_mask_read_by_production_loader"]
                for receipt in raw_receipts.values()
            ),
            "geometry_only_trial_inclusion": all(
                receipt["production_geometry_only_loader"]
                for receipt in raw_receipts.values()
            ),
            "seven_manifest_sessions": len(names) == 7,
            "raw_receipts_valid": True,
            "target_shapes_masks_and_finiteness_valid": all(
                receipt["target_shape"] == [CHANNELS, TARGET_WIDTH]
                and receipt["mask_shape"] == [CHANNELS, TARGET_WIDTH]
                and receipt["mask_equals_target_finiteness"]
                and receipt["whole_row_mask"]
                and min(receipt["defined_fractions"]) >= 0.90
                for receipt in target_receipts.values()
            ),
            "operational_seven_delta_vector": len(comparison["delta_by_session"]) == 7,
            "valid_execution": True,
        },
        "stage": "aligned_only",
    }

    try:
        reliability = {
            session: split_reliability(trials, session) for session, trials in support.items()
        }
    except ValueError as error:
        timer.mark("reliability_failed")
        base_result["reliability_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        base_result["contracts"]["measurement_reliable"] = False
        base_result["decision"] = "r10_measurement_unreliable_stop"
        return _finish(output, base_result, timer, prelaunch, review)
    timer.mark("reliability")
    base_result["reliability"] = reliability
    measurement_reliable = all(
        item["minimum_defined_row_fraction"] >= 0.90
        and item["quantiles_2p5_50_97p5"][0] >= 0.50
        and np.isfinite(item["median_standardized_row_cosine"]).all()
        and np.isfinite(item["per_coordinate_pearson"]).all()
        for item in reliability.values()
    )
    base_result["contracts"]["measurement_reliable"] = measurement_reliable
    if not measurement_reliable:
        base_result["decision"] = "r10_measurement_unreliable_stop"
        return _finish(output, base_result, timer, prelaunch, review)

    deltas = np.asarray(comparison["delta_by_session"], dtype=np.float64)
    if not np.all(deltas > 0) or comparison["mean"] < 0.030:
        base_result["decision"] = "r10_low_order_control_not_beaten_stop"
        return _finish(output, base_result, timer, prelaunch, review)
    interval = comparison["operational_interval_95"]
    if interval[0] <= 0 or comparison["operational_mde80"] > 0.030:
        base_result["decision"] = "r10_precision_insufficient_stop"
        return _finish(output, base_result, timer, prelaunch, review)

    null_h: list[float] = []
    per_schedule_folds: list[list[dict[str, Any]]] = []
    for replicate in range(1, RANDOM_SCHEDULES + 1):
        if replicate == 1 or replicate % 32 == 0:
            timer.check(f"attachment_null_{replicate}")
        folds = score_loso(null_features(features, replicate), targets)
        per_schedule_folds.append(folds)
        null_h.append(float(np.mean([float(row["bounded_u"]) for row in folds])))
    timer.mark("attachment_null_scores")

    observed_h = float(np.mean([float(row["bounded_u"]) for row in r10_scores]))
    randomization = mc(observed_h, null_h)
    session_above_median = 0
    for observed in r10_scores:
        session = observed["left_out_session"]
        null_session_values = [
            next(
                float(row["bounded_u"])
                for row in schedule_folds
                if row["left_out_session"] == session
            )
            for schedule_folds in per_schedule_folds
        ]
        if float(observed["bounded_u"]) > float(np.median(null_session_values)):
            session_above_median += 1

    score_payload = json.dumps(
        json_safe(per_schedule_folds), sort_keys=True, separators=(",", ":")
    ).encode()
    timer.check("pre_schedule_serialization")
    schedule = schedule_receipt(names, CHANNELS)
    timer.mark("schedule_receipt")
    prelaunch_plan = json.loads(prelaunch.read_text())["formal_null"]["full_plan_sha256"]
    if schedule["full_plan_sha256"] != prelaunch_plan:
        raise ValueError("formal schedule differs from prelaunch plan")

    base_result["stage"] = "attachment_complete"
    base_result["attachment"] = {
        "H": observed_h,
        "random_H": null_h,
        "per_schedule_folds": per_schedule_folds,
        "mc": randomization,
        "session_above_median_count": session_above_median,
        "schedule": schedule,
        "per_fold_scores_sha256": hashlib.sha256(score_payload).hexdigest(),
        "random_H_sha256": array_sha(np.asarray(null_h), "<f8"),
    }
    passes_attachment = (
        randomization["p_attach"] < 0.025
        and randomization["mc_upper_97p5"] < 0.025
        and observed_h > float(np.median(null_h))
        and session_above_median >= 6
    )
    base_result["contracts"]["formal_schedule_matches_prelaunch"] = True
    base_result["contracts"]["attachment_gate_passed"] = passes_attachment
    base_result["decision"] = (
        "r10_m2_source_pass_for_separate_review"
        if passes_attachment
        else "r10_row_attachment_not_distinguishable_stop"
    )
    return _finish(output, base_result, timer, prelaunch, review)


def execute_with_terminal(
    output: Path,
    prelaunch: Path,
    review: Path,
    *,
    deadline_seconds: float = DEADLINE_SECONDS,
    loader: Callable[..., tuple[dict[str, tuple[np.ndarray, ...]], dict[str, dict[str, Any]]]] = _load_after_review,
) -> Path:
    """Run once and persist a terminal receipt for every post-authorization failure."""
    started = time.perf_counter()
    try:
        return run(
            output,
            prelaunch,
            review,
            deadline_seconds=deadline_seconds,
            loader=loader,
        )
    except FileExistsError:
        raise
    except Exception as error:
        if output.exists():
            raise
        deadline_failure = isinstance(error, GateDeadline)
        guard_failure = isinstance(error, SourceGuardFailure)
        terminal = {
            "schema_version": "m2_r10_m24_source_gate_v2",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "binding": _bindings(prelaunch, review),
            "stage": "terminal_error",
            "decision": (
                "r10_null_budget_infeasible_stop"
                if deadline_failure
                else "r10_invalid_execution_stop"
            ),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "stage": getattr(error, "stage", None),
            },
            "contracts": {
                "source_only": True,
                "held_out": False,
                "gpu": False,
                "decoder": False,
                "evalai": False,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "valid_execution": False,
                "source_loader_called": not guard_failure,
            },
            "runtime": {"total_seconds": time.perf_counter() - started},
        }
        return _write_source_result(output, terminal)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-prelaunch", action="store_true")
    parser.add_argument("--prelaunch-output", type=Path, default=PRE)
    parser.add_argument("--execute-source-audit", action="store_true")
    parser.add_argument("--reviewed-prelaunch", type=Path)
    parser.add_argument("--source-only-confirmation")
    parser.add_argument("--output", type=Path, default=OUT)
    arguments = parser.parse_args()
    if arguments.write_prelaunch and arguments.execute_source_audit:
        raise ValueError("prelaunch and execution require separate invocations")
    if arguments.write_prelaunch:
        print(write_prelaunch(arguments.prelaunch_output))
        return
    if not arguments.execute_source_audit:
        raise RuntimeError("refusing implicit source access")
    prelaunch = arguments.prelaunch_output / "prelaunch_receipt.json"
    if arguments.reviewed_prelaunch is None or arguments.source_only_confirmation != CONFIRM:
        raise PermissionError("source execution guard missing")
    print(execute_with_terminal(arguments.output, prelaunch, arguments.reviewed_prelaunch))


if __name__ == "__main__":
    main()
