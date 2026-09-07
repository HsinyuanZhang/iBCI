#!/usr/bin/env python3
"""Audit whether RT exposes per-reach target metadata without behavior traces.

This is the CPU-only Stage 0 specified by
``HANDOFF_SIMPLE_LABEL_MAINLINE_20260810.md``.  It reads NWB metadata and the
cursor-position timestamp axis, writes only to a new isolated result directory,
and never imports torch or creates a CUDA context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from pynwb import NWBHDF5IO


SCHEMA = "rt_simple_label_stage0_metadata_v1"
EXPECTED_SESSION_COUNT = 15
TRIAL_TARGET_NAMES = {
    "target_position",
    "target_pos",
    "target_xy",
    "goal_position",
    "goal_pos",
    "goal_xy",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _flatten_finite(values: Any) -> np.ndarray:
    items: list[np.ndarray] = []
    for value in values:
        try:
            array = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            continue
        items.append(array[np.isfinite(array)])
    if not items:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(items)


def _column_schema(frame: Any, name: str) -> dict[str, Any]:
    series = frame[name]
    first = next((value for value in series if value is not None), None)
    result: dict[str, Any] = {
        "pandas_dtype": str(series.dtype),
        "cell_python_type": type(first).__name__ if first is not None else None,
    }
    if first is not None:
        array = np.asarray(first)
        result["cell_array_dtype"] = str(array.dtype)
        result["cell_shape"] = [int(value) for value in array.shape]
    return result


def _nearest_timestamp_distances(timestamps: np.ndarray, query: np.ndarray) -> np.ndarray:
    right = np.searchsorted(timestamps, query, side="left")
    right = np.clip(right, 0, timestamps.size - 1)
    left = np.clip(right - 1, 0, timestamps.size - 1)
    return np.minimum(np.abs(timestamps[right] - query), np.abs(timestamps[left] - query))


def audit_session(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        target_or_goal_paths: list[str] = []

        def collect(name: str) -> None:
            lower = name.lower()
            if "target" in lower or "goal" in lower:
                target_or_goal_paths.append(name)

        handle.visit(collect)

    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        if nwb.trials is None:
            raise ValueError(f"{path.name}: missing trials table")
        trials = nwb.trials.to_dataframe()
        columns = list(nwb.trials.colnames)
        column_schema = {name: _column_schema(trials, name) for name in columns}

        target_dir = _flatten_finite(trials["target_dir"]) if "target_dir" in columns else np.empty(0)
        target_id = _flatten_finite(trials["target_id"]) if "target_id" in columns else np.empty(0)
        target_dir_unique = np.unique(np.round(target_dir, decimals=12))
        target_id_unique = np.unique(np.round(target_id, decimals=12))
        direct_target_columns = sorted(TRIAL_TARGET_NAMES.intersection(columns))

        if "go_cue_time_array" not in columns:
            raise ValueError(f"{path.name}: missing go_cue_time_array")
        cue_values = _flatten_finite(trials["go_cue_time_array"])
        if cue_values.size == 0:
            raise ValueError(f"{path.name}: no finite go cues")

        try:
            cursor_pos = nwb.processing["behavior"]["Position"].spatial_series["cursor_pos"]
        except KeyError as error:
            raise ValueError(f"{path.name}: missing behavior/Position/cursor_pos") from error
        if cursor_pos.timestamps is None:
            raise ValueError(f"{path.name}: cursor_pos lacks explicit timestamps")
        position_timestamps = np.asarray(cursor_pos.timestamps[:], dtype=np.float64).reshape(-1)
        position_data = np.asarray(cursor_pos.data[:])
        if position_data.ndim != 2 or position_data.shape[1] != 2:
            raise ValueError(f"{path.name}: cursor_pos is not [time,2]")
        if position_timestamps.shape != (position_data.shape[0],):
            raise ValueError(f"{path.name}: cursor_pos timestamp/data mismatch")
        if not np.isfinite(position_timestamps).all() or np.any(np.diff(position_timestamps) <= 0):
            raise ValueError(f"{path.name}: invalid cursor_pos timestamps")
        distances = _nearest_timestamp_distances(position_timestamps, cue_values)
        cues_in_range = (cue_values >= position_timestamps[0]) & (cue_values <= position_timestamps[-1])

    no_nondegenerate_target_metadata = bool(
        not direct_target_columns
        and target_dir_unique.size <= 1
        and target_id_unique.size <= 1
        and not any(
            name not in {
                "intervals/trials/num_targets",
                "intervals/trials/target_dir",
                "intervals/trials/target_id",
                "intervals/trials/target_size",
            }
            for name in target_or_goal_paths
        )
    )
    return {
        "session": path.name.removeprefix("sub-C_").removesuffix("_behavior+ecephys.nwb"),
        "nwb_path": str(path.resolve()),
        "nwb_size_bytes": int(path.stat().st_size),
        "trial_count": int(len(trials)),
        "trial_columns": columns,
        "trial_column_dtypes": column_schema,
        "target_or_goal_hdf5_paths": sorted(target_or_goal_paths),
        "direct_per_trial_target_position_columns": direct_target_columns,
        "target_id": {
            "finite_count": int(target_id.size),
            "unique_finite_values": target_id_unique.tolist(),
            "degenerate_or_absent": bool(target_id_unique.size <= 1),
        },
        "target_dir": {
            "finite_count": int(target_dir.size),
            "unique_finite_values_rad": target_dir_unique.tolist(),
            "degenerate_or_absent": bool(target_dir_unique.size <= 1),
        },
        "go_cue": {
            "recorded_trial_field": "go_cue_time_array",
            "finite_cue_count": int(cue_values.size),
            "all_finite_cues_inside_cursor_position_range": bool(np.all(cues_in_range)),
        },
        "reach_boundaries": {
            "recorded_fields": ["start_time", "stop_time", "go_cue_time_array", "num_targets"],
            "reach_start": "recorded go_cue_time_array entry",
            "reach_end": "derived as next go cue, or trial stop for the final reach",
            "all_reach_segment_boundaries_are_recorded": False,
        },
        "cursor_position": {
            "path": "processing/behavior/Position/cursor_pos",
            "shape": [int(value) for value in position_data.shape],
            "data_dtype": str(position_data.dtype),
            "timestamp_dtype": str(position_timestamps.dtype),
            "unit": str(cursor_pos.unit),
            "reference_frame": str(cursor_pos.reference_frame),
            "median_sample_period_s": float(np.median(np.diff(position_timestamps))),
            "nearest_go_cue_sample_median_abs_dt_s": float(np.median(distances)),
            "nearest_go_cue_sample_max_abs_dt_s": float(np.max(distances)),
            "one_nearest_or_interpolated_sample_can_define_reach_origin": bool(np.all(cues_in_range)),
            "one_sample_cannot_define_target_or_direction": True,
        },
        "per_reach_direction_without_behavior_trace_available": not no_nondegenerate_target_metadata,
        "stage0_session_outcome": (
            "NONDEGENERATE_TARGET_METADATA_PRESENT"
            if not no_nondegenerate_target_metadata
            else "NO_NONDEGENERATE_TARGET_METADATA"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("sua_exploration/data/dandi_000688/sub-C"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sua_exploration/results/rt_simple_label_v1/stage0"),
    )
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be the empty string")
    caps = {
        name: os.environ.get(name)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    if any(value not in {"1", "2"} for value in caps.values()):
        raise RuntimeError(f"CPU thread caps must each be 1 or 2, got {caps}")

    paths = sorted(args.data_root.resolve().glob("sub-C_ses-RT-*_behavior+ecephys.nwb"))
    if len(paths) != EXPECTED_SESSION_COUNT:
        raise RuntimeError(f"expected {EXPECTED_SESSION_COUNT} RT sessions, found {len(paths)}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)

    rows = [audit_session(path) for path in paths]
    all_no_target = all(row["stage0_session_outcome"] == "NO_NONDEGENERATE_TARGET_METADATA" for row in rows)
    branch = (
        "B_NO_PER_REACH_TARGET_FIELD_DIRECTION_REQUIRES_BEHAVIOR_TRACE_ANNOTATION_COST_CLAIM_VOID"
        if all_no_target
        else "A_PER_REACH_TARGET_METADATA_EXISTS_ANNOTATION_COST_CLAIM_AVAILABLE"
    )
    payload = {
        "schema": SCHEMA,
        "status": "STAGE0_COMPLETE_STOP_BEFORE_STAGE1",
        "protocol": {
            "path": str(
                Path("sua_exploration/docs/HANDOFF_SIMPLE_LABEL_MAINLINE_20260810.md").resolve()
            ),
            "sha256": sha256_file(
                Path("sua_exploration/docs/HANDOFF_SIMPLE_LABEL_MAINLINE_20260810.md")
            ),
        },
        "prior_opened_scope_reference": {
            "path": str(
                Path(
                    "sua_exploration/results/k4_rt_loso_v1/RT_STAGE_R_D1024_FULL15_AGGREGATE_v1.json"
                ).resolve()
            ),
            "sha256": sha256_file(
                Path(
                    "sua_exploration/results/k4_rt_loso_v1/RT_STAGE_R_D1024_FULL15_AGGREGATE_v1.json"
                )
            ),
        },
        "compute": {
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "thread_caps": caps,
            "torch_imported": False,
            "gpu_context_created": False,
        },
        "non_interference": {
            "processes_signalled": False,
            "gpu_jobs_started": False,
            "watched_directories_written": False,
            "existing_artifacts_modified": False,
            "output_isolated_under": str(output_dir),
        },
        "session_count": len(rows),
        "stage0_read_rule_branch": branch,
        "accuracy_claim_remains_available": True,
        "annotation_cost_claim_available": not all_no_target,
        "interpretation": (
            "No RT session stores a nondegenerate per-reach target position, target index, or direction. "
            "target_id and target_dir are constant annotations. The recorded go cues identify reach "
            "starts, and sparse cursor-position samples can identify origins, but a target or displacement "
            "still has to be obtained from the behavior trace. Stage 1 may test accuracy constructibility, "
            "but it cannot support a two-coordinate-per-reach annotation-cost claim."
            if all_no_target
            else "At least one session exposes nondegenerate target metadata; inspect per-session rows."
        ),
        "sessions": rows,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    receipt = output_dir / "RT_SIMPLE_LABEL_STAGE0_METADATA_RECEIPT_v1.json"
    receipt.write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(receipt, 0o444)
    print(receipt)


if __name__ == "__main__":
    main()
