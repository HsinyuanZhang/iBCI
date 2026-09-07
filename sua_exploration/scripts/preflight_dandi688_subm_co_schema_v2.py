#!/usr/bin/env python3
"""Append-only v2 adapter for the score-blind sub-M CO schema preflight.

This successor preserves the v1 parser-adapter receipt as an incident record.
It changes only two schema adapters: PyNWB's one-row pandas DataFrame electrode
regions are resolved by their sole index, and abort-only ``target_dir`` NaNs are
diagnostic rather than a rewarded-trial failure.  All frozen content gates,
asset IDs, hash checks, 20-ms / 50-bin loader semantics, and model-score bans
remain fail-closed.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from pynwb import NWBHDF5IO


REPO_ROOT = Path(__file__).resolve().parents[2]
V1_SOURCE = REPO_ROOT / "sua_exploration/scripts/preflight_dandi688_subm_co_schema_v1.py"
V1_SOURCE_SHA256 = "3b2a0d8e402ab3dc0fd2f238a35f5e12c70e2dd2f49984be86b78c94a3536a50"
V1_RECEIPT = REPO_ROOT / "sua_exploration/results/dandi_000688_subm_co_schema_preflight_v1/receipt.json"
DEFAULT_RECEIPT = REPO_ROOT / "sua_exploration/results/dandi_000688_subm_co_schema_preflight_v2/receipt.json"


def _load_v1_module() -> Any:
    if not V1_SOURCE.is_file():
        raise RuntimeError(f"v1 source is missing: {V1_SOURCE}")
    spec = importlib.util.spec_from_file_location("subm_preflight_v1_base", V1_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct v1 preflight module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_v1_module()
_base_audit_static_model_boundary = base.audit_static_model_boundary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _failure(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def resolve_one_dataframe_electrode_id(reference: Any, electrode_ids: set[int]) -> int | None:
    """Resolve a PyNWB materialized DynamicTableRegion without a scalar shortcut."""
    try:
        index = np.asarray(getattr(reference, "index", None), dtype=np.int64)
    except (TypeError, ValueError):
        return None
    if index.ndim != 1 or index.size != 1:
        return None
    value = int(index[0])
    return value if value in electrode_ids else None


def finite_target_required_for_usable_rewarded(result: Any, target_dir: float | None) -> bool:
    """The frozen loader filters result == 'R' before T4 consumes target_dir."""
    return result != "R" or target_dir is not None


def _strictly_chronological(rows: list[dict[str, Any]]) -> bool:
    return all(
        float(right["start_time"]) > float(left["start_time"])
        and float(right["stop_time"]) > float(left["stop_time"])
        for left, right in zip(rows, rows[1:])
    )


def _balance(target_dirs: np.ndarray) -> dict[str, Any]:
    values, counts = np.unique(np.round(target_dirs.astype(np.float64), 12), return_counts=True)
    return {
        "condition_values_target_dir_rad": [float(value) for value in values],
        "counts": [int(value) for value in counts],
        "n_conditions": int(values.size),
        "min_count": int(counts.min()),
        "max_count": int(counts.max()),
        "exclusion_threshold_applied": False,
    }


def inspect_verified_nwb_v2(path: Path, asset: dict[str, Any]) -> dict[str, Any]:
    """Open a manifest-hash-verified NWB once and perform only schema checks."""
    failures: list[dict[str, str]] = []
    observed: dict[str, Any] = {}
    dm: dict[str, Any] = {
        "bin_size_ms": base.BIN_SIZE_MS,
        "window_size_bins": base.WINDOW_SIZE_BINS,
        "reward_filter": "result == 'R'",
        "selection_semantics": "exact list_datamodule_rewarded_trials searchsorted/clip/duration filter",
    }
    pseudo: dict[str, Any] = {"mapping_valid": False, "t4_refit_performed": False}
    t_min: float | None = None
    t_max: float | None = None
    try:
        with NWBHDF5IO(str(path), "r") as io:
            nwb = io.read()
            subject = getattr(getattr(nwb, "subject", None), "subject_id", None)
            session_id = getattr(nwb, "session_id", None)
            description = getattr(nwb, "session_description", None)
            observed["identity"] = {
                "subject_id": subject,
                "session_id": session_id,
                "session_description": description,
            }
            expected_public = asset["public_metadata"]
            if subject != expected_public["participant_identifier"]:
                failures.append(_failure("subject_identity", f"expected M, got {subject!r}"))
            if session_id != asset["session_id"]:
                failures.append(_failure("session_identity", f"expected {asset['session_id']!r}, got {session_id!r}"))
            if description != expected_public["session_description"] or "center-out" not in str(description).lower():
                failures.append(_failure("task_identity", "NWB description does not match frozen center-out identity"))

            units_df = None
            if nwb.units is None:
                failures.append(_failure("units_table", "NWB Units table is absent"))
            else:
                try:
                    units_df = nwb.units.to_dataframe()
                except Exception as exc:
                    failures.append(_failure("units_table", f"cannot materialize Units: {type(exc).__name__}: {exc}"))
            if units_df is not None:
                unit_count = int(len(units_df))
                observed["unit_count"] = unit_count
                if not 0 < unit_count < 100:
                    failures.append(_failure("unit_count", f"requires 0 < N < 100, got {unit_count}"))
                if "spike_times" not in units_df.columns:
                    failures.append(_failure("spike_times", "Units lacks spike_times"))
                else:
                    invalid_spike_rows: list[int] = []
                    total_events = 0
                    nonempty_units = 0
                    for row_index, values in enumerate(units_df["spike_times"]):
                        try:
                            spikes = np.asarray(values, dtype=np.float64)
                        except (TypeError, ValueError):
                            invalid_spike_rows.append(row_index)
                            continue
                        if spikes.ndim != 1 or not np.all(np.isfinite(spikes)) or (spikes.size > 1 and np.any(np.diff(spikes) < 0)):
                            invalid_spike_rows.append(row_index)
                            continue
                        total_events += int(spikes.size)
                        if spikes.size:
                            nonempty_units += 1
                            local_min, local_max = float(spikes.min()), float(spikes.max())
                            t_min = local_min if t_min is None else min(t_min, local_min)
                            t_max = local_max if t_max is None else max(t_max, local_max)
                    observed["event_level_spike_times"] = {
                        "units_checked": unit_count,
                        "invalid_unit_rows": invalid_spike_rows,
                        "total_events": total_events,
                        "nonempty_units": nonempty_units,
                    }
                    if invalid_spike_rows:
                        failures.append(_failure("spike_times", f"invalid finite/monotonic event rows={invalid_spike_rows}"))
                    if t_min is None or t_max is None or not t_max > t_min:
                        failures.append(_failure("spike_time_span", "no nondegenerate event-time span for 20-ms bins"))

                if "electrodes" not in units_df.columns or nwb.electrodes is None or len(nwb.electrodes) == 0:
                    failures.append(_failure("electrode_reference", "Units/electrode table lacks resolvable electrode references"))
                else:
                    valid_ids = {int(value) for value in np.asarray(nwb.electrodes.to_dataframe().index, dtype=np.int64)}
                    mapping = [resolve_one_dataframe_electrode_id(value, valid_ids) for value in units_df["electrodes"]]
                    invalid = [index for index, value in enumerate(mapping) if value is None]
                    resolved = [int(value) for value in mapping if value is not None]
                    pseudo = {
                        "mapping_valid": not invalid and len(resolved) == unit_count,
                        "source_unit_count": unit_count,
                        "electrode_table_row_count": int(len(nwb.electrodes)),
                        "invalid_unit_rows": invalid,
                        "unique_electrode_channel_count": int(len(set(resolved))),
                        "pooling_rule": "deterministic np.unique electrode IDs then sum raw unit binned activity; preflight performs no T4 refit",
                        "t4_refit_performed": False,
                    }
                    if invalid:
                        failures.append(_failure("electrode_reference", f"requires exactly one in-range electrode ID per unit; invalid rows={invalid}"))

            try:
                velocity = nwb.processing["behavior"]["Velocity"].time_series["cursor_vel"]
                data = np.asarray(velocity.data[:], dtype=np.float64)
                times = np.asarray(velocity.timestamps[:], dtype=np.float64)
                unit = re.sub(r"\s+", "", str(getattr(velocity, "unit", "")).strip().lower())
                observed["cursor_vel"] = {
                    "shape": [int(value) for value in data.shape],
                    "timestamp_count": int(times.size),
                    "unit": str(getattr(velocity, "unit", "")),
                    "finite_data": bool(np.all(np.isfinite(data))),
                    "finite_timestamps": bool(np.all(np.isfinite(times))),
                    "strictly_increasing_timestamps": bool(times.ndim == 1 and times.size > 1 and np.all(np.diff(times) > 0)),
                }
                if data.ndim != 2 or data.shape[1] != 2:
                    failures.append(_failure("cursor_vel_shape", f"requires [samples,2], got {data.shape}"))
                if times.ndim != 1 or times.size != data.shape[0] or times.size < 2 or not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
                    failures.append(_failure("cursor_vel_timestamps", "requires one finite strictly increasing timestamp per sample"))
                if not np.all(np.isfinite(data)):
                    failures.append(_failure("cursor_vel_finite", "cursor_vel values must be finite"))
                if unit != "cm/s":
                    failures.append(_failure("cursor_vel_unit", f"requires cm/s, got {getattr(velocity, 'unit', None)!r}"))
            except Exception as exc:
                failures.append(_failure("cursor_vel", f"missing/unreadable behavior/Velocity/cursor_vel: {type(exc).__name__}: {exc}"))

            try:
                trials = nwb.intervals["trials"].to_dataframe()
            except Exception as exc:
                trials = None
                failures.append(_failure("trials_table", f"missing/unreadable trials: {type(exc).__name__}: {exc}"))
            if trials is not None:
                needed = {"start_time", "stop_time", "result", "target_dir"}
                missing = sorted(needed - set(trials.columns))
                dm["trial_table_row_count"] = int(len(trials))
                dm["required_columns_present"] = not missing
                if missing:
                    failures.append(_failure("trials_columns", f"missing {missing}"))
                else:
                    rows: list[dict[str, Any]] = []
                    invalid_rows: list[int] = []
                    abort_nan_rows: list[int] = []
                    for ordinal, (raw_index, row) in enumerate(trials.iterrows()):
                        start, stop, target = _float_or_none(row["start_time"]), _float_or_none(row["stop_time"]), _float_or_none(row["target_dir"])
                        result = row["result"]
                        try:
                            raw_trial_index = int(raw_index)
                        except (TypeError, ValueError):
                            raw_trial_index = None
                        if start is None or stop is None or start >= stop or not isinstance(result, str) or not result or raw_trial_index is None:
                            invalid_rows.append(ordinal)
                            continue
                        if result == "A" and target is None:
                            abort_nan_rows.append(raw_trial_index)
                        rows.append({"trial_index": raw_trial_index, "start_time": start, "stop_time": stop, "result": result, "target_dir": target})
                    dm["all_trial_time_and_result_fields_valid"] = not invalid_rows and len(rows) == len(trials)
                    dm["invalid_trial_row_ordinals"] = invalid_rows
                    dm["abort_nan_target_dir_raw_trial_indices_diagnostic"] = abort_nan_rows
                    dm["trial_rows_strictly_chronological"] = _strictly_chronological(rows)
                    if invalid_rows or len(rows) != len(trials):
                        failures.append(_failure("trials_fields", f"all rows require finite start/stop and nonempty result; invalid rows={invalid_rows}"))
                    if not dm["trial_rows_strictly_chronological"]:
                        failures.append(_failure("trials_chronology", "trial rows are not strictly chronological"))
                    if t_min is None or t_max is None or not t_max > t_min:
                        dm["exact_usable_rewarded_trial_count"] = None
                        failures.append(_failure("datamodule_feasibility", "cannot build bin edges from Units event span"))
                    else:
                        edges = np.arange(t_min, t_max + base.BIN_SIZE_S, base.BIN_SIZE_S)
                        n_bins = len(edges) - 1
                        usable: list[dict[str, Any]] = []
                        for row in rows:
                            if row["result"] != "R":
                                continue
                            start_bin = max(0, int(np.searchsorted(edges, row["start_time"])))
                            stop_bin = min(n_bins, int(np.searchsorted(edges, row["stop_time"])))
                            if stop_bin - start_bin >= base.WINDOW_SIZE_BINS:
                                usable.append({**row, "start_bin": start_bin, "stop_bin": stop_bin})
                        missing_rewarded_targets = [int(row["trial_index"]) for row in usable if not finite_target_required_for_usable_rewarded(row["result"], row["target_dir"])]
                        query_windows = sum(int(row["stop_bin"] - row["start_bin"] - base.WINDOW_SIZE_BINS + 1) for row in usable[base.CALIBRATION_POOL_TRIALS:])
                        dm.update({
                            "spike_time_bin_start": float(edges[0]),
                            "spike_time_bin_count": int(n_bins),
                            "exact_usable_rewarded_trial_count": int(len(usable)),
                            "usable_rewarded_trials_strictly_chronological": _strictly_chronological(usable),
                            "usable_rewarded_trials_have_finite_target_dir": not missing_rewarded_targets,
                            "usable_rewarded_missing_target_dir_raw_trial_indices": missing_rewarded_targets,
                            "first_50_raw_trial_indices": [int(row["trial_index"]) for row in usable[:base.CALIBRATION_POOL_TRIALS]],
                            "complete_query_trials_strictly_after_first_50": int(len(usable[base.CALIBRATION_POOL_TRIALS:])),
                            "complete_query_window_count_strictly_after_first_50": int(query_windows),
                        })
                        if not dm["usable_rewarded_trials_strictly_chronological"]:
                            failures.append(_failure("rewarded_trials_chronology", "usable rewarded rows are not strictly chronological"))
                        if len(usable) < base.CALIBRATION_POOL_TRIALS:
                            failures.append(_failure("usable_rewarded_trial_count", f"requires >=50, got {len(usable)}"))
                        if missing_rewarded_targets:
                            failures.append(_failure("trials_target_dir", f"usable rewarded target_dir nonfinite at raw rows={missing_rewarded_targets}"))
                        if len(usable) >= base.CALIBRATION_POOL_TRIALS and not missing_rewarded_targets:
                            directions = np.asarray([row["target_dir"] for row in usable[:base.CALIBRATION_POOL_TRIALS]], dtype=np.float64)
                            design = np.column_stack((np.cos(directions), np.sin(directions), np.ones(base.CALIBRATION_POOL_TRIALS)))
                            singular_values = np.linalg.svd(design, compute_uv=False)
                            rank = int(np.linalg.matrix_rank(design))
                            dm["first_50_target_design"] = {"columns": ["cos(target_dir)", "sin(target_dir)", "intercept"], "rank": rank, "condition_number": float(np.linalg.cond(design)), "singular_values": [float(value) for value in singular_values], "balance": _balance(directions)}
                            if rank != 3:
                                failures.append(_failure("first_50_target_design_rank", f"requires rank 3, got {rank}"))
                        if query_windows < 1:
                            failures.append(_failure("post50_query_window", "no complete query window remains strictly after rewarded trial 50"))
    except Exception as exc:
        failures.append(_failure("nwb_open_or_schema", f"{type(exc).__name__}: {exc}"))
    return {
        "asset_id": asset["asset_id"], "frozen_path": asset["path"], "session_id": asset["session_id"],
        "disposition": "ELIGIBLE" if not failures else "INELIGIBLE_SCHEMA_OR_FEASIBILITY",
        "eligible": not failures, "failure_reasons": failures, "observed_schema": observed,
        "datamodule_feasibility": dm, "pseudo_mua": pseudo, "score_blind": True,
        "checkpoint_loaded": False, "model_forward_calls": 0, "behavior_scores_computed": 0,
    }


def audit_static_model_boundary_v2() -> dict[str, Any]:
    """Audit both v2 entrypoint and hash-qualified model-free implementation."""
    implementation = _base_audit_static_model_boundary()
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__).resolve()))
    imported = []
    called = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Call):
            called.append(node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else "")
    forbidden_imports = sorted(item for item in imported if any(word in item.lower() for word in ("torch", "lightning", "tensorflow", "torchmetrics", "sklearn")))
    forbidden_calls = sorted(item for item in called if item in {"forward", "predict", "backward", "step", "zero_grad", "load_state_dict", "load_from_checkpoint", "evaluate"})
    base.require(not forbidden_imports, f"forbidden v2 imports: {forbidden_imports}")
    base.require(not forbidden_calls, f"forbidden v2 calls: {forbidden_calls}")
    implementation_source = implementation.pop("preflight_source")
    implementation["preflight_source"] = {"path": str(Path(__file__).resolve().relative_to(REPO_ROOT)), "sha256": _sha256(Path(__file__).resolve()), "bytes": Path(__file__).stat().st_size}
    implementation["implementation_source"] = implementation_source
    implementation["entrypoint_static_forbidden_imports"] = forbidden_imports
    implementation["entrypoint_static_forbidden_calls"] = forbidden_calls
    return implementation


_base_build_receipt = base.build_receipt


def build_receipt_v2(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _base_build_receipt(*args, **kwargs)
    base.require(V1_RECEIPT.is_file(), f"v1 incident receipt missing: {V1_RECEIPT}")
    payload["schema_version"] = 2
    payload["receipt_kind"] = "dandi_000688_subm_co_score_blind_schema_preflight_v2"
    payload["supersedes"] = {
        "v1_receipt": {"path": str(V1_RECEIPT.relative_to(REPO_ROOT)), "sha256": _sha256(V1_RECEIPT), "bytes": V1_RECEIPT.stat().st_size},
        "v1_source_expected_sha256": V1_SOURCE_SHA256,
        "v1_source_observed_sha256": _sha256(V1_SOURCE),
        "v1_disposition": "FAILED_SCORE_BLIND_PARSER_ADAPTER_V1_NOT_ENDPOINT_EVIDENCE",
        "correction_scope": [
            "resolve materialized one-row electrode DataFrame by sole in-range electrode table ID",
            "require finite target_dir for exact usable rewarded rows; report abort-only NaNs diagnostically",
        ],
        "other_frozen_gates_weakened": False,
    }
    return payload


def main() -> None:
    base.require(_sha256(V1_SOURCE) == V1_SOURCE_SHA256, "v1 source SHA drift; v2 refuses to run")
    base.DEFAULT_RECEIPT = DEFAULT_RECEIPT
    base.inspect_verified_nwb = inspect_verified_nwb_v2
    base.audit_static_model_boundary = audit_static_model_boundary_v2
    base.build_receipt = build_receipt_v2
    base.main()


if __name__ == "__main__":
    try:
        main()
    except base.PreflightError as exc:
        raise SystemExit(f"PRECHECK_FAIL_CLOSED: {exc}") from exc
