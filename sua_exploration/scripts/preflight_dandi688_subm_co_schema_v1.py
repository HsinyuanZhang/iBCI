#!/usr/bin/env python3
"""Fail-closed, score-blind DANDI 000688 sub-M CO schema preflight.

This is deliberately *not* an external evaluator.  It downloads only the
22 asset IDs frozen in the immutable v2 manifest, verifies each file's exact
byte count and SHA-256 before opening it, and inspects only NWB schema and
datamodule feasibility.  It neither imports torch nor opens a checkpoint, and
it has no prediction, metric, optimizer, or GPU path.

Downloads are resumable ``.part`` files in an isolated directory.  A part file
is never opened.  A completed file is atomically promoted only after the exact
manifest size and SHA-256 both match.  The final receipt is published once via
an atomic hard link; an existing receipt is never modified or replaced.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pynwb import NWBHDF5IO


REPO_ROOT = Path(__file__).resolve().parents[2]
SCOPE_ID = "dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2"
MANIFEST_PATH = (
    REPO_ROOT
    / "sua_exploration/manifests/dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2.json"
)
EXPECTED_MANIFEST_SHA256 = "68503c7b2985182f821a0c896be68bd4a2f957304f2487fa3a9947b740689c55"
DEFAULT_DOWNLOAD_ROOT = (
    REPO_ROOT / "sua_exploration/cache/dandi_000688_subm_co_schema_preflight_assets"
)
DEFAULT_RECEIPT = (
    REPO_ROOT
    / "sua_exploration/results/dandi_000688_subm_co_schema_preflight_v1/receipt.json"
)
LOADER_SEMANTICS_SOURCE = REPO_ROOT / "sua_exploration/mc_maze/multisession_datamodule.py"

EXPECTED_ASSET_COUNT = 22
EXPECTED_TOTAL_BYTES = 2_312_360_648
BIN_SIZE_MS = 20
BIN_SIZE_S = BIN_SIZE_MS / 1000.0
WINDOW_SIZE_BINS = 50
CALIBRATION_POOL_TRIALS = 50
MIN_ELIGIBLE_SESSIONS = 6
DOWNLOAD_URL_TEMPLATE = "https://api.dandiarchive.org/api/assets/{asset_id}/download/"
CO_PATH_RE = re.compile(
    r"^sub-M/sub-M_ses-CO-(?P<date>[0-9]{8})_behavior\+ecephys\.nwb$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PreflightError(RuntimeError):
    """A fail-closed preflight error that must never lead to file opening."""


class DownloadInterrupted(PreflightError):
    """A resumable download failure; the verified-complete file was not opened."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def asset_download_url(asset_id: str) -> str:
    """Return the official DANDI URL from a frozen asset ID, never a path lookup."""
    require(
        re.fullmatch(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}", asset_id)
        is not None,
        f"invalid frozen asset ID: {asset_id!r}",
    )
    return DOWNLOAD_URL_TEMPLATE.format(asset_id=asset_id)


def load_and_validate_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    require(path.is_file(), f"immutable v2 manifest is missing: {path}")
    observed_sha = sha256_file(path)
    require(
        observed_sha == EXPECTED_MANIFEST_SHA256,
        "immutable v2 manifest SHA-256 mismatch: "
        f"expected {EXPECTED_MANIFEST_SHA256}, got {observed_sha}",
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot parse immutable v2 manifest: {exc}") from exc
    require(isinstance(payload, dict), "immutable v2 manifest root is not an object")
    require(payload.get("scope_id") == SCOPE_ID, "v2 scope ID drift")
    require(payload.get("schema_version") == 2, "v2 schema version drift")
    assets = payload.get("selected_assets")
    require(isinstance(assets, list), "v2 selected_assets is not a list")
    require(len(assets) == EXPECTED_ASSET_COUNT, "frozen sub-M CO asset count drift")
    require(sum(int(row.get("size", -1)) for row in assets) == EXPECTED_TOTAL_BYTES, "frozen byte total drift")

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for row in assets:
        require(isinstance(row, dict), "selected asset row is not an object")
        asset_id = row.get("asset_id")
        asset_path = row.get("path")
        expected_sha = row.get("sha256")
        size = row.get("size")
        session_id = row.get("session_id")
        match = CO_PATH_RE.fullmatch(str(asset_path))
        require(match is not None, f"asset is outside frozen sub-M CO scope: {asset_path!r}")
        require(isinstance(asset_id, str) and asset_id not in seen_ids, "duplicate/invalid asset ID")
        require(isinstance(asset_path, str) and asset_path not in seen_paths, "duplicate/invalid asset path")
        require(isinstance(size, int) and size > 0, f"invalid size for {asset_path}")
        require(isinstance(expected_sha, str) and SHA256_RE.fullmatch(expected_sha) is not None, f"invalid SHA for {asset_path}")
        require(
            session_id == f"sub-M_ses-CO-{match.group('date')}",
            f"frozen session identity drift for {asset_path}",
        )
        public = row.get("public_metadata")
        require(isinstance(public, dict), f"public metadata missing for {asset_path}")
        require(public.get("participant_identifier") == "M", f"frozen subject identity drift for {asset_path}")
        description = public.get("session_description")
        require(isinstance(description, str) and "center-out" in description.lower(), f"frozen task identity drift for {asset_path}")
        seen_ids.add(asset_id)
        seen_paths.add(asset_path)

    return payload, {
        "path": str(path.relative_to(REPO_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": observed_sha,
        "scope_id": SCOPE_ID,
        "selected_asset_count": len(assets),
        "selected_bytes": sum(int(row["size"]) for row in assets),
    }


def audit_static_model_boundary() -> dict[str, Any]:
    """Prove this source has no model, checkpoint, or metric import/call path."""
    source_path = Path(__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(source_path))
    except SyntaxError as exc:  # pragma: no cover - source cannot run if this occurs
        raise PreflightError(f"cannot audit own source: {exc}") from exc

    imports: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)

    forbidden_import_fragments = ("torch", "lightning", "sklearn", "torchmetrics", "tensorflow")
    forbidden_calls = {
        "backward",
        "step",
        "zero_grad",
        "load_from_checkpoint",
        "load_state_dict",
        "predict",
        "forward",
        "evaluate",
    }
    bad_imports = sorted(
        item for item in imports if any(fragment in item.lower() for fragment in forbidden_import_fragments)
    )
    bad_calls = sorted(item for item in calls if item in forbidden_calls)
    require(not bad_imports, f"forbidden model/metric import in preflight source: {bad_imports}")
    require(not bad_calls, f"forbidden model/metric call in preflight source: {bad_calls}")
    require(LOADER_SEMANTICS_SOURCE.is_file(), "reference datamodule source missing")
    return {
        "preflight_source": {
            "path": str(source_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(source_path),
            "bytes": source_path.stat().st_size,
        },
        "safe_loader_semantics_reference": {
            "path": str(LOADER_SEMANTICS_SOURCE.relative_to(REPO_ROOT)),
            "sha256": sha256_file(LOADER_SEMANTICS_SOURCE),
            "used_semantics": (
                "list_datamodule_rewarded_trials: result == 'R', bin-edge searchsorted, "
                "clip to the spike-time span, and retain duration >= 50 bins"
            ),
            "imported": False,
            "reason_not_imported": "the datamodule imports torch; this score-blind preflight does not",
        },
        "static_forbidden_imports": bad_imports,
        "static_forbidden_calls": bad_calls,
        "checkpoint_files_opened": 0,
        "model_forward_calls": 0,
        "prediction_calls": 0,
        "behavior_score_calls": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "weight_updates": 0,
        "gpu_used": False,
        "sub_c_endpoint_opened": False,
    }


def completed_path(download_root: Path, asset_id: str) -> Path:
    return download_root / "assets" / f"{asset_id}.nwb"


def verify_complete_asset(path: Path, asset: dict[str, Any]) -> dict[str, Any]:
    """Verify exact bytes and hash; callers may open only after this returns."""
    expected_size = int(asset["size"])
    expected_sha = str(asset["sha256"])
    require(path.is_file(), f"downloaded file missing for {asset['asset_id']}: {path}")
    observed_size = path.stat().st_size
    require(
        observed_size == expected_size,
        f"downloaded size mismatch for {asset['asset_id']}: expected {expected_size}, got {observed_size}",
    )
    observed_sha = sha256_file(path)
    require(
        observed_sha == expected_sha,
        f"downloaded SHA-256 mismatch for {asset['asset_id']}: expected {expected_sha}, got {observed_sha}",
    )
    return {
        "asset_id": asset["asset_id"],
        "path": str(path.resolve()),
        "bytes": observed_size,
        "sha256": observed_sha,
        "size_and_sha256_verified_before_nwb_open": True,
    }


def _content_range_starts_at(value: str | None, offset: int) -> bool:
    if value is None:
        return False
    return re.fullmatch(rf"bytes {offset}-\d+/\d+", value.strip()) is not None


def download_one_asset(download_root: Path, asset: dict[str, Any], timeout_seconds: float) -> Path:
    """Resume one manifest-bound download and atomically promote only a verified file."""
    asset_id = str(asset["asset_id"])
    final_path = completed_path(download_root, asset_id)
    part_path = final_path.with_suffix(".nwb.part")
    expected_size = int(asset["size"])

    if final_path.exists():
        # A final-name collision is never overwritten.  A valid one is safely reusable.
        verify_complete_asset(final_path, asset)
        return final_path
    require(not final_path.is_symlink(), f"refusing symlink final path: {final_path}")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if part_path.exists():
        require(part_path.is_file() and not part_path.is_symlink(), f"invalid partial file: {part_path}")
        current_size = part_path.stat().st_size
        require(
            current_size <= expected_size,
            f"partial file exceeds manifest size and will not be modified: {part_path}",
        )
    else:
        current_size = 0

    # A prior interruption can have completed the bytes but not reached promotion.
    if current_size == expected_size:
        verified = verify_complete_asset(part_path, asset)
        os.replace(part_path, final_path)
        require(verified["sha256"] == asset["sha256"], "impossible post-verification hash drift")
        return final_path

    url = asset_download_url(asset_id)
    headers = {
        # DANDI's download view negotiates the redirect with a broad media type;
        # narrowing this to application/octet-stream produces HTTP 406 before any
        # content transfer on the archive API.
        "Accept": "*/*",
        "User-Agent": f"SPINT-{SCOPE_ID}-score-blind-schema-preflight/1",
    }
    if current_size:
        headers["Range"] = f"bytes={current_size}-"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", response.getcode())
            if current_size:
                require(status == 206, f"download server did not honor resume range for {asset_id}: HTTP {status}")
                require(
                    _content_range_starts_at(response.headers.get("Content-Range"), current_size),
                    f"download server returned unsafe Content-Range for {asset_id}",
                )
            else:
                require(status in {200, 206}, f"unexpected download HTTP status for {asset_id}: {status}")
                if status == 206:
                    require(
                        _content_range_starts_at(response.headers.get("Content-Range"), 0),
                        f"download server returned unsafe initial Content-Range for {asset_id}",
                    )
            with part_path.open("ab") as handle:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
    except PreflightError:
        raise
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DownloadInterrupted(
            f"download interrupted for frozen asset {asset_id}; partial file remains resumable at {part_path}: {exc}"
        ) from exc

    observed_size = part_path.stat().st_size
    require(
        observed_size == expected_size,
        f"download ended with incomplete/oversized bytes for {asset_id}: expected {expected_size}, got {observed_size}",
    )
    verify_complete_asset(part_path, asset)
    # The target did not previously exist and this rename is same-directory atomic.
    os.replace(part_path, final_path)
    verify_complete_asset(final_path, asset)
    return final_path


def _as_scalar_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _strictly_chronological(rows: Iterable[dict[str, Any]]) -> bool:
    previous_start: float | None = None
    previous_stop: float | None = None
    for row in rows:
        start = float(row["start_time"])
        stop = float(row["stop_time"])
        if previous_start is not None and (start <= previous_start or stop <= previous_stop):
            return False
        previous_start, previous_stop = start, stop
    return True


def _normalise_unit(unit: Any) -> str:
    return re.sub(r"\s+", "", str(unit).strip().lower())


def _balance_summary(target_dirs: np.ndarray) -> dict[str, Any]:
    rounded = np.round(target_dirs.astype(np.float64), decimals=12)
    values, counts = np.unique(rounded, return_counts=True)
    return {
        "condition_values_target_dir_rad": [float(value) for value in values],
        "counts": [int(value) for value in counts],
        "min_count": int(counts.min()),
        "max_count": int(counts.max()),
        "n_conditions": int(values.size),
        "exclusion_threshold_applied": False,
    }


def _schema_failure(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def inspect_verified_nwb(path: Path, asset: dict[str, Any]) -> dict[str, Any]:
    """Inspect schema only.  This function assumes an exact hash check already passed."""
    failures: list[dict[str, str]] = []
    observed: dict[str, Any] = {}
    pseudo_mua: dict[str, Any] = {"mapping_valid": False}
    datamodule: dict[str, Any] = {
        "bin_size_ms": BIN_SIZE_MS,
        "window_size_bins": WINDOW_SIZE_BINS,
        "reward_filter": "result == 'R'",
        "selection_semantics": "exact searchsorted/clip/duration semantics of list_datamodule_rewarded_trials",
    }

    try:
        with NWBHDF5IO(str(path), "r") as io:
            nwb = io.read()
            subject = getattr(nwb, "subject", None)
            observed_subject = getattr(subject, "subject_id", None) if subject is not None else None
            observed_session = getattr(nwb, "session_id", None)
            observed_description = getattr(nwb, "session_description", None)
            expected_public = asset["public_metadata"]
            observed["identity"] = {
                "subject_id": observed_subject,
                "session_id": observed_session,
                "session_description": observed_description,
            }
            if observed_subject != expected_public["participant_identifier"]:
                failures.append(_schema_failure("subject_identity", f"expected subject M, got {observed_subject!r}"))
            if observed_session != asset["session_id"]:
                failures.append(
                    _schema_failure(
                        "session_identity",
                        f"expected {asset['session_id']!r}, got {observed_session!r}",
                    )
                )
            expected_description = expected_public["session_description"]
            if observed_description != expected_description or "center-out" not in str(observed_description).lower():
                failures.append(
                    _schema_failure(
                        "task_identity",
                        "NWB session_description does not exactly match frozen center-out metadata",
                    )
                )

            units_df = None
            spike_t_min: float | None = None
            spike_t_max: float | None = None
            if nwb.units is None:
                failures.append(_schema_failure("units_table", "NWB Units table is absent"))
            else:
                try:
                    units_df = nwb.units.to_dataframe()
                except Exception as exc:  # schema read failure, not a model path
                    failures.append(_schema_failure("units_table", f"cannot materialize Units table: {type(exc).__name__}: {exc}"))

            if units_df is not None:
                unit_count = len(units_df)
                observed["unit_count"] = int(unit_count)
                if not (0 < unit_count < 100):
                    failures.append(_schema_failure("unit_count", f"requires 0 < N < 100, got N={unit_count}"))
                if "spike_times" not in units_df.columns:
                    failures.append(_schema_failure("spike_times", "Units table has no event-level spike_times column"))
                else:
                    invalid_spikes = 0
                    event_counts: list[int] = []
                    for unit_row, values in enumerate(units_df["spike_times"]):
                        try:
                            spike_times = np.asarray(values, dtype=np.float64)
                        except (TypeError, ValueError):
                            invalid_spikes += 1
                            continue
                        if spike_times.ndim != 1 or not np.all(np.isfinite(spike_times)):
                            invalid_spikes += 1
                            continue
                        if spike_times.size and np.any(spike_times[1:] < spike_times[:-1]):
                            invalid_spikes += 1
                            continue
                        event_counts.append(int(spike_times.size))
                        if spike_times.size:
                            local_min = float(spike_times.min())
                            local_max = float(spike_times.max())
                            spike_t_min = local_min if spike_t_min is None else min(spike_t_min, local_min)
                            spike_t_max = local_max if spike_t_max is None else max(spike_t_max, local_max)
                    observed["event_level_spike_times"] = {
                        "units_checked": int(unit_count),
                        "invalid_unit_rows": int(invalid_spikes),
                        "total_events": int(sum(event_counts)),
                        "nonempty_units": int(sum(count > 0 for count in event_counts)),
                    }
                    if invalid_spikes:
                        failures.append(
                            _schema_failure(
                                "spike_times",
                                f"{invalid_spikes} Units rows lack finite monotonic one-dimensional event times",
                            )
                        )
                    if spike_t_min is None or spike_t_max is None or not spike_t_max > spike_t_min:
                        failures.append(
                            _schema_failure(
                                "spike_time_span",
                                "no nondegenerate finite event-time span for datamodule bin edges",
                            )
                        )

                electrode_count = len(nwb.electrodes) if nwb.electrodes is not None else 0
                invalid_electrodes: list[int] = []
                electrode_ids: list[int] = []
                if "electrodes" not in units_df.columns:
                    failures.append(_schema_failure("electrode_reference", "Units table has no electrodes reference column"))
                elif electrode_count <= 0:
                    failures.append(_schema_failure("electrode_reference", "NWB electrode table is absent/empty"))
                else:
                    for unit_row, reference in enumerate(units_df["electrodes"]):
                        index = getattr(reference, "index", None)
                        table = getattr(reference, "table", None)
                        try:
                            indices = np.asarray(index, dtype=np.int64)
                        except (TypeError, ValueError):
                            invalid_electrodes.append(unit_row)
                            continue
                        if (
                            index is None
                            or indices.ndim != 1
                            or indices.size != 1
                            or table is None
                            or int(indices[0]) < 0
                            or int(indices[0]) >= electrode_count
                        ):
                            invalid_electrodes.append(unit_row)
                            continue
                        electrode_ids.append(int(indices[0]))
                    pseudo_mua = {
                        "mapping_valid": not invalid_electrodes and len(electrode_ids) == len(units_df),
                        "source_unit_count": int(len(units_df)),
                        "electrode_table_row_count": int(electrode_count),
                        "invalid_unit_rows": invalid_electrodes,
                        "unique_electrode_channel_count": int(len(set(electrode_ids))),
                        "pooling_rule": "deterministic np.unique electrode IDs then sum raw unit binned activity; no T4 refit performed in preflight",
                        "t4_refit_performed": False,
                    }
                    if invalid_electrodes:
                        failures.append(
                            _schema_failure(
                                "electrode_reference",
                                "exactly one resolvable in-range electrode reference is required per unit; "
                                f"invalid rows={invalid_electrodes}",
                            )
                        )

            # Cursor velocity is inspected independently so an Units failure does not hide it.
            try:
                vel_series = nwb.processing["behavior"]["Velocity"].time_series["cursor_vel"]
                cursor_vel = np.asarray(vel_series.data[:], dtype=np.float64)
                timestamps = np.asarray(vel_series.timestamps[:], dtype=np.float64)
                velocity_unit = _normalise_unit(getattr(vel_series, "unit", ""))
                observed["cursor_vel"] = {
                    "shape": [int(value) for value in cursor_vel.shape],
                    "timestamp_count": int(timestamps.size),
                    "unit": str(getattr(vel_series, "unit", "")),
                    "finite_data": bool(np.all(np.isfinite(cursor_vel))),
                    "finite_timestamps": bool(np.all(np.isfinite(timestamps))),
                    "strictly_increasing_timestamps": bool(
                        timestamps.ndim == 1 and timestamps.size > 1 and np.all(np.diff(timestamps) > 0)
                    ),
                }
                if cursor_vel.ndim != 2 or cursor_vel.shape[1] != 2:
                    failures.append(_schema_failure("cursor_vel_shape", f"requires [samples,2], got {cursor_vel.shape}"))
                if timestamps.ndim != 1 or timestamps.size != cursor_vel.shape[0]:
                    failures.append(_schema_failure("cursor_vel_timestamps", "timestamps are not one per cursor_vel sample"))
                if not np.all(np.isfinite(cursor_vel)) or not np.all(np.isfinite(timestamps)):
                    failures.append(_schema_failure("cursor_vel_finite", "cursor_vel values/timestamps must be finite"))
                if timestamps.ndim != 1 or timestamps.size < 2 or not np.all(np.diff(timestamps) > 0):
                    failures.append(_schema_failure("cursor_vel_timestamps", "timestamps must be strictly increasing"))
                if velocity_unit != "cm/s":
                    failures.append(
                        _schema_failure(
                            "cursor_vel_unit",
                            f"cursor_vel must use cm/s, got {getattr(vel_series, 'unit', None)!r}",
                        )
                    )
            except Exception as exc:
                failures.append(
                    _schema_failure("cursor_vel", f"missing/unreadable behavior/Velocity/cursor_vel: {type(exc).__name__}: {exc}")
                )

            # This is the exact datamodule rewarded trial filter, augmented only with
            # schema checks and audit fields.  No behavior values are transformed or scored.
            try:
                trials_df = nwb.intervals["trials"].to_dataframe()
            except Exception as exc:
                trials_df = None
                failures.append(_schema_failure("trials_table", f"missing/unreadable trials table: {type(exc).__name__}: {exc}"))
            if trials_df is not None:
                required_columns = {"start_time", "stop_time", "result", "target_dir"}
                missing_columns = sorted(required_columns - set(trials_df.columns))
                datamodule["trial_table_row_count"] = int(len(trials_df))
                datamodule["required_columns_present"] = not missing_columns
                if missing_columns:
                    failures.append(_schema_failure("trials_columns", f"missing required columns: {missing_columns}"))
                else:
                    trial_rows: list[dict[str, Any]] = []
                    invalid_trials: list[int] = []
                    for ordinal, (raw_index, trial) in enumerate(trials_df.iterrows()):
                        start = _as_scalar_float(trial["start_time"])
                        stop = _as_scalar_float(trial["stop_time"])
                        target_dir = _as_scalar_float(trial["target_dir"])
                        result = trial["result"]
                        if start is None or stop is None or target_dir is None or start >= stop or not isinstance(result, str) or not result:
                            invalid_trials.append(int(ordinal))
                            continue
                        try:
                            original_index = int(raw_index)
                        except (TypeError, ValueError):
                            invalid_trials.append(int(ordinal))
                            continue
                        trial_rows.append(
                            {
                                "trial_index": original_index,
                                "start_time": start,
                                "stop_time": stop,
                                "result": result,
                                "target_dir": target_dir,
                            }
                        )
                    datamodule["all_trial_fields_finite_and_valid"] = not invalid_trials
                    datamodule["invalid_trial_row_ordinals"] = invalid_trials
                    datamodule["trial_rows_strictly_chronological"] = _strictly_chronological(trial_rows)
                    if invalid_trials or len(trial_rows) != len(trials_df):
                        failures.append(
                            _schema_failure(
                                "trials_fields",
                                f"all trial rows need finite start/stop/target_dir and nonempty result; invalid rows={invalid_trials}",
                            )
                        )
                    if not datamodule["trial_rows_strictly_chronological"]:
                        failures.append(_schema_failure("trials_chronology", "trial rows are not strictly chronological"))

                    if spike_t_min is None or spike_t_max is None or not spike_t_max > spike_t_min:
                        datamodule["exact_usable_rewarded_trial_count"] = None
                        failures.append(
                            _schema_failure(
                                "datamodule_feasibility",
                                "cannot construct 20-ms bin edges from the verified Units event-time span",
                            )
                        )
                    else:
                        bin_edges = np.arange(spike_t_min, spike_t_max + BIN_SIZE_S, BIN_SIZE_S)
                        num_bins = len(bin_edges) - 1
                        usable_rewarded: list[dict[str, Any]] = []
                        # Reuse the loader's loop exactly: result == 'R', searchsorted,
                        # clip to [0,num_bins], and duration >= WINDOW_SIZE_BINS.
                        for trial in trial_rows:
                            if trial["result"] != "R":
                                continue
                            start_bin = int(np.searchsorted(bin_edges, trial["start_time"]))
                            stop_bin = int(np.searchsorted(bin_edges, trial["stop_time"]))
                            start_bin = max(0, start_bin)
                            stop_bin = min(num_bins, stop_bin)
                            if stop_bin - start_bin >= WINDOW_SIZE_BINS:
                                usable_rewarded.append(
                                    {
                                        **trial,
                                        "start_bin": start_bin,
                                        "stop_bin": stop_bin,
                                    }
                                )
                        chronological_rewarded = _strictly_chronological(usable_rewarded)
                        query_windows = sum(
                            int(row["stop_bin"] - row["start_bin"] - WINDOW_SIZE_BINS + 1)
                            for row in usable_rewarded[CALIBRATION_POOL_TRIALS:]
                        )
                        datamodule.update(
                            {
                                "spike_time_bin_start": float(bin_edges[0]),
                                "spike_time_bin_count": int(num_bins),
                                "exact_usable_rewarded_trial_count": int(len(usable_rewarded)),
                                "usable_rewarded_trials_strictly_chronological": chronological_rewarded,
                                "first_50_raw_trial_indices": [
                                    int(row["trial_index"]) for row in usable_rewarded[:CALIBRATION_POOL_TRIALS]
                                ],
                                "complete_query_trials_strictly_after_first_50": int(
                                    len(usable_rewarded[CALIBRATION_POOL_TRIALS:])
                                ),
                                "complete_query_window_count_strictly_after_first_50": int(query_windows),
                            }
                        )
                        if not chronological_rewarded:
                            failures.append(
                                _schema_failure("rewarded_trials_chronology", "datamodule-usable rewarded trials are not strictly chronological")
                            )
                        if len(usable_rewarded) < CALIBRATION_POOL_TRIALS:
                            failures.append(
                                _schema_failure(
                                    "usable_rewarded_trial_count",
                                    f"requires at least {CALIBRATION_POOL_TRIALS}, got {len(usable_rewarded)}",
                                )
                            )
                        if len(usable_rewarded) >= CALIBRATION_POOL_TRIALS:
                            first_50_dirs = np.asarray(
                                [row["target_dir"] for row in usable_rewarded[:CALIBRATION_POOL_TRIALS]], dtype=np.float64
                            )
                            design = np.column_stack(
                                (np.cos(first_50_dirs), np.sin(first_50_dirs), np.ones(CALIBRATION_POOL_TRIALS))
                            )
                            singular_values = np.linalg.svd(design, compute_uv=False)
                            rank = int(np.linalg.matrix_rank(design))
                            condition_number = float(np.linalg.cond(design))
                            datamodule["first_50_target_design"] = {
                                "columns": ["cos(target_dir)", "sin(target_dir)", "intercept"],
                                "rank": rank,
                                "condition_number": condition_number,
                                "singular_values": [float(value) for value in singular_values],
                                "balance": _balance_summary(first_50_dirs),
                            }
                            if rank != 3:
                                failures.append(
                                    _schema_failure("first_50_target_design_rank", f"requires rank 3, got {rank}")
                                )
                        if query_windows < 1:
                            failures.append(
                                _schema_failure(
                                    "post50_query_window",
                                    "no complete 50-bin query window remains strictly after chronological rewarded trial 50",
                                )
                            )
    except Exception as exc:
        failures.append(_schema_failure("nwb_open_or_schema", f"{type(exc).__name__}: {exc}"))

    return {
        "asset_id": asset["asset_id"],
        "frozen_path": asset["path"],
        "session_id": asset["session_id"],
        "disposition": "ELIGIBLE" if not failures else "INELIGIBLE_SCHEMA_OR_FEASIBILITY",
        "eligible": not failures,
        "failure_reasons": failures,
        "observed_schema": observed,
        "datamodule_feasibility": datamodule,
        "pseudo_mua": pseudo_mua,
        "score_blind": True,
        "checkpoint_loaded": False,
        "model_forward_calls": 0,
        "behavior_scores_computed": 0,
    }


def write_once_atomic(path: Path, body: bytes) -> None:
    """Publish exactly once without replacing a receipt or an existing inode."""
    path = path.resolve()
    require(not os.path.lexists(path), f"write-once receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PreflightError(f"write-once receipt collision: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def build_receipt(
    manifest_binding: dict[str, Any],
    boundary_audit: dict[str, Any],
    download_root: Path,
    verified_downloads: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    require(len(verified_downloads) == EXPECTED_ASSET_COUNT, "receipt requires all 22 downloaded/hash-verified assets")
    require(len(ledger) == EXPECTED_ASSET_COUNT, "receipt requires all 22 dispositions")
    require(sum(int(row["bytes"]) for row in verified_downloads) == EXPECTED_TOTAL_BYTES, "receipt downloaded byte total drift")
    eligible_ids = [row["asset_id"] for row in ledger if row["eligible"]]
    ineligible = [row for row in ledger if not row["eligible"]]
    n_eligible = len(eligible_ids)
    endpoint_disposition = (
        "ENDPOINT_NO_GO_INSUFFICIENT_ELIGIBLE_SESSIONS"
        if n_eligible < MIN_ELIGIBLE_SESSIONS
        else "PREFLIGHT_COMPLETE_SEPARATE_SCORE_AUTHORIZATION_REQUIRED"
    )
    return {
        "schema_version": 1,
        "receipt_kind": "dandi_000688_subm_co_score_blind_schema_preflight",
        "scope_id": SCOPE_ID,
        "status": "COMPLETE_SCORE_BLIND_SCHEMA_PREFLIGHT",
        "immutable_v2_manifest": manifest_binding,
        "scope_enforcement": {
            "asset_count": EXPECTED_ASSET_COUNT,
            "downloaded_bytes": EXPECTED_TOTAL_BYTES,
            "download_root": str(download_root.resolve()),
            "download_rule": "exactly the 22 frozen v2 asset IDs via official DANDI /api/assets/{asset_id}/download/ URLs",
            "rt_assets_downloaded": 0,
            "sub_c_endpoint_opened": False,
            "asset_path_discovery_performed": False,
        },
        "protocol": {
            "bin_size_ms": BIN_SIZE_MS,
            "window_size_bins": WINDOW_SIZE_BINS,
            "calibration_first_rewarded_trials": CALIBRATION_POOL_TRIALS,
            "query_rule": "complete 50-bin windows strictly after chronological rewarded trial 50",
            "unit_count_rule": "0 < N < 100",
            "pseudo_mua_rule": "one valid electrode reference per unit, deterministic within-session electrode pooling only",
        },
        "model_score_boundary_audit": boundary_audit,
        "verified_downloads": verified_downloads,
        "asset_disposition_ledger": ledger,
        "eligible_session_ids": eligible_ids,
        "eligible_session_count": n_eligible,
        "ineligible_session_count": len(ineligible),
        "ineligible_failure_code_counts": {
            code: sum(1 for row in ineligible if any(reason["code"] == code for reason in row["failure_reasons"]))
            for code in sorted({reason["code"] for row in ineligible for reason in row["failure_reasons"]})
        },
        "future_endpoint_disposition": endpoint_disposition,
        "checkpoint_load_permitted_by_this_receipt": False,
        "if_n_less_than_6": "NO-GO is issued before any future checkpoint load, prediction, or score",
        "preflight_performed_no_scoring": True,
        "append_only": {
            "receipt_write_policy": "atomic hard-link publication; existing receipt is never replaced",
            "source_freezes_modified": False,
            "all_22_assets_remain_in_ledger": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--download-root", type=Path, default=DEFAULT_DOWNLOAD_ROOT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--verify-contract",
        action="store_true",
        help="verify only the immutable manifest and source-level score/model boundary; no download, NWB open, or receipt write",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.timeout_seconds > 0, "timeout must be positive")
    manifest, manifest_binding = load_and_validate_manifest(args.manifest)
    boundary_audit = audit_static_model_boundary()
    if args.verify_contract:
        print(json.dumps({"manifest": manifest_binding, "model_score_boundary_audit": boundary_audit}, indent=2, sort_keys=True))
        return

    receipt_path = args.receipt.resolve()
    require(not os.path.lexists(receipt_path), f"write-once receipt already exists: {receipt_path}")
    download_root = args.download_root.resolve()
    # The selected assets list is the sole source of download IDs and paths.
    verified_downloads: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for asset in manifest["selected_assets"]:
        local_path = download_one_asset(download_root, asset, args.timeout_seconds)
        verification = verify_complete_asset(local_path, asset)
        verified_downloads.append(verification)
        # This is intentionally the first and only NWB open for this asset.
        ledger.append(inspect_verified_nwb(local_path, asset))

    receipt = build_receipt(manifest_binding, boundary_audit, download_root, verified_downloads, ledger)
    write_once_atomic(receipt_path, canonical_bytes(receipt))
    print(receipt_path)


if __name__ == "__main__":
    try:
        main()
    except PreflightError as exc:
        raise SystemExit(f"PRECHECK_FAIL_CLOSED: {exc}") from exc
