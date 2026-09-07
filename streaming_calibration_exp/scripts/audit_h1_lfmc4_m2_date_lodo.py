#!/usr/bin/env python3
"""Immutable CPU-only H1 LFMC4 M=2 date-LODO feasibility audit.

This program opens exactly the 13 public H1 held-in calibration NWBs.  It
refuses minival/query, held-out/formal/EvalAI paths, models, and target
optimizers.  Each outer date receives a new source-only bounded kinematic
basis; the target consumes only its first two chronological trials through
closed-form exposure-weighted spike/latent cross moments.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src import h1_lfmc4 as lfmc4  # noqa: E402


PROTOCOL_VERSION = "h1_lfmc4_m2_date_lodo_cpu_audit_v1"
NULL_REPLICATES = 31
NULL_SEED = 20260807
COSINE_THRESHOLD = 0.50
COVERAGE_THRESHOLD = 0.90
MIN_PASSING_DATES = 4
NORMALIZER_SCALE_FLOOR = 1.0e-6
EXPECTED_CHANNELS = 176
H1_HELDIN_SESSIONS = (
    "ses-19250101T111740", "ses-19250101T112404", "ses-19250108T110520",
    "ses-19250108T111022", "ses-19250108T111455", "ses-19250113T120811",
    "ses-19250113T121303", "ses-19250115T110633", "ses-19250115T111328",
    "ses-19250119T113543", "ses-19250119T114045", "ses-19250120T115044",
    "ses-19250120T115537",
)
H1_DATES = ("19250101", "19250108", "19250113", "19250115", "19250119", "19250120")


class AuditError(lfmc4.LFMC4Error):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _date_from_session(session: str) -> str:
    if not session.startswith("ses-") or len(session) < 12 or session[12] != "T":
        raise AuditError(f"invalid H1 session name {session!r}")
    date = session[4:12]
    if not date.isdigit():
        raise AuditError(f"invalid H1 date in {session!r}")
    return date


def _session_name_from_path(path: Path) -> str:
    marker = "_ses-"
    if marker not in path.name:
        raise AuditError(f"cannot derive session from {path.name!r}")
    return "ses-" + path.stem.split(marker, 1)[1].split("_behavior", 1)[0]


def _require_data_root(root: Path) -> None:
    lower = str(root.resolve()).lower().rstrip("/") + "/"
    if "spint-main/data/000954/" not in lower or any(x in lower for x in ("held-out", "heldout", "formal", "evalai", "test")):
        raise AuditError(f"LFMC4 accepts only SPINT-main/data/000954, got {root}")
    if not root.is_dir():
        raise FileNotFoundError(root)


def _require_heldin_calib(path: Path) -> None:
    lower = str(path.resolve()).lower()
    if "sub-humanpitt-held-in-calib" not in lower or any(x in lower for x in ("held-out", "heldout", "formal", "evalai", "test")):
        raise AuditError(f"LFMC4 refuses non-held-in-calib path {path}")
    if not path.is_file():
        raise FileNotFoundError(path)


def index_h1_heldin_calib(data_dir: Path) -> dict[str, Path]:
    _require_data_root(data_dir)
    paths = {_session_name_from_path(p): p.resolve() for p in sorted((data_dir / "sub-HumanPitt-held-in-calib").glob("*.nwb"))}
    if set(paths) != set(H1_HELDIN_SESSIONS):
        raise AuditError("LFMC4 requires exactly the 13 named public held-in-calib NWBs")
    for path in paths.values():
        _require_heldin_calib(path)
    return {name: paths[name] for name in H1_HELDIN_SESSIONS}


def _first_two_trials(trial_num: np.ndarray, eval_mask: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(trial_num, dtype=np.float64).reshape(-1)
    valid = np.flatnonzero(np.asarray(eval_mask, dtype=bool).reshape(-1) & np.isfinite(labels))
    if not valid.size:
        raise AuditError("no finite eval-valid TrialNum bins")
    values = labels[valid]
    if np.any(np.diff(values) < 0):
        raise AuditError("TrialNum is not chronological")
    ordered: list[float] = []
    for value in values:
        if not ordered or value != ordered[-1]:
            ordered.append(float(value))
        if len(ordered) == 2:
            return ordered[0], ordered[1]
    raise AuditError("M=2 requires two chronological eval-valid trials")


def _chronological_trial_values(trial_num: np.ndarray, eval_mask: np.ndarray) -> tuple[float, ...]:
    labels = np.asarray(trial_num, dtype=np.float64).reshape(-1)
    valid = np.flatnonzero(np.asarray(eval_mask, dtype=bool).reshape(-1) & np.isfinite(labels))
    if not valid.size:
        raise AuditError("no finite eval-valid TrialNum bins")
    values = labels[valid]
    if np.any(np.diff(values) < 0):
        raise AuditError("TrialNum is not chronological")
    output: list[float] = []
    for value in values:
        if not output or value != output[-1]:
            output.append(float(value))
    return tuple(output)


def _runs(indices: np.ndarray) -> tuple[np.ndarray, ...]:
    if not indices.size:
        return ()
    cuts = np.flatnonzero(np.diff(indices) != 1) + 1
    return tuple(np.asarray(x, dtype=np.int64) for x in np.split(indices, cuts))


def _raw_trial(*, trial_number: float, neural: np.ndarray, velocity: np.ndarray, eval_mask: np.ndarray, trial_num: np.ndarray) -> lfmc4.RawTrial:
    belongs = np.asarray(trial_num == trial_number, dtype=bool)
    finite = np.isfinite(neural).all(axis=1) & np.isfinite(velocity).all(axis=1)
    legal = belongs & np.asarray(eval_mask, dtype=bool) & finite
    chunks = _runs(np.flatnonzero(legal))
    if not chunks:
        raise AuditError(f"trial {trial_number} has no legal native bins")
    indices = np.concatenate(chunks)
    return lfmc4.RawTrial(
        trial_number=float(trial_number), counts=np.asarray(neural[indices], dtype=np.float64),
        velocity=np.asarray(velocity[indices], dtype=np.float64),
        exposure=np.full(indices.size, lfmc4.BIN_SECONDS, dtype=np.float64),
        run_lengths=tuple(int(chunk.size) for chunk in chunks),
        audit={
            "trial_number": float(trial_number), "raw_trial_bins": int(belongs.sum()),
            "finite_eval_valid_bins": int(legal.sum()), "contiguous_finite_eval_valid_runs": int(len(chunks)),
            "native_bin_ms": 20, "activity_threshold_used": None,
            "block_crosses_trial_boundary": False, "all_finite_bins_only": True,
        },
    )


def record_from_arrays(*, session_name: str, neural: np.ndarray, velocity: np.ndarray, eval_mask: np.ndarray, trial_num: np.ndarray, path: str | None = None, input_sha256: str | None = None) -> lfmc4.LFMC4Record:
    neural, velocity = np.asarray(neural, dtype=np.float64), np.asarray(velocity, dtype=np.float64)
    if neural.ndim != 2 or velocity.shape != (neural.shape[0], lfmc4.VELOCITY_DIM) or neural.shape[1] < 2:
        raise AuditError("raw H1 neural/velocity shapes are invalid")
    mask, labels = np.asarray(eval_mask, dtype=bool).reshape(-1), np.asarray(trial_num, dtype=np.float64).reshape(-1)
    if mask.shape != (neural.shape[0],) or labels.shape != mask.shape:
        raise AuditError("raw H1 mask/TrialNum shapes are invalid")
    one, two = _first_two_trials(labels, mask)
    source_trials: list[lfmc4.RawTrial] = []
    for trial_number in _chronological_trial_values(labels, mask):
        try:
            source_trials.append(_raw_trial(trial_number=trial_number, neural=neural, velocity=velocity, eval_mask=mask, trial_num=labels))
        except AuditError:
            # A source "legal trial" is one with at least one finite, eval-valid
            # native bin.  Target first-two validity remains fail-closed below.
            continue
    by_number = {trial.trial_number: trial for trial in source_trials}
    if one not in by_number or two not in by_number:
        raise AuditError("one of the target chronological first two trials has no legal finite bins")
    return lfmc4.LFMC4Record(str(session_name), _date_from_session(str(session_name)), path, input_sha256, (by_number[one], by_number[two]), tuple(source_trials))


def load_h1_record(path: Path) -> lfmc4.LFMC4Record:
    _require_heldin_calib(path)
    try:
        from falcon_challenge.config import FalconTask
        from falcon_challenge.dataloaders import load_nwb
        from pynwb import NWBHDF5IO
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("LFMC4 needs the SPINT falcon/pynwb runtime") from error
    neural, velocity, _trial_change, eval_mask = load_nwb(path, FalconTask.h1)
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        if "TrialNum" not in nwb.acquisition:
            raise AuditError(f"{path}: TrialNum is missing")
        trial_num = np.asarray(nwb.acquisition["TrialNum"].data[:], dtype=np.float64)
    result = record_from_arrays(session_name=_session_name_from_path(path), neural=neural, velocity=velocity, eval_mask=eval_mask, trial_num=trial_num, path=str(path.resolve()), input_sha256=sha256_file(path))
    if result.channels != EXPECTED_CHANNELS:
        raise AuditError(f"{result.session_name}: expected {EXPECTED_CHANNELS} channels, got {result.channels}")
    return result


def partitions(records: Mapping[str, lfmc4.LFMC4Record]) -> tuple[dict[str, Any], ...]:
    by_date: dict[str, list[str]] = {}
    for name, record in records.items():
        if name != record.session_name:
            raise AuditError("record map key/session mismatch")
        by_date.setdefault(record.date, []).append(name)
    if tuple(sorted(by_date)) != H1_DATES or len(by_date) != 6:
        raise AuditError("LFMC4 requires six date groups")
    return tuple({"outer_date": outer, "source_dates": tuple(date for date in sorted(by_date) if date != outer), "source_sessions": tuple(name for date in sorted(by_date) if date != outer for name in sorted(by_date[date])), "target_sessions": tuple(sorted(by_date[outer]))} for outer in sorted(by_date))


def source_plan(records: Mapping[str, lfmc4.LFMC4Record], outer_date: str) -> dict[str, Any]:
    part = next((row for row in partitions(records) if row["outer_date"] == outer_date), None)
    if part is None:
        raise AuditError(f"unknown outer date {outer_date}")
    source = tuple(records[name] for name in part["source_sessions"])
    if any(record.date == outer_date for record in source):
        raise AuditError("outer date leaked into source basis")
    basis = lfmc4.fit_source_basis(source)
    source_trial_pair_counts = {x: {"legal_source_trials": len(records[x].source_trials), "contiguous_m2_pairs": max(0, len(records[x].source_trials) - 1)} for x in part["source_sessions"]}
    if any(item["contiguous_m2_pairs"] <= 0 for item in source_trial_pair_counts.values()):
        raise AuditError("every source record must expose at least one legal contiguous M=2 pair")
    body = {"outer_date": outer_date, "source_dates": list(part["source_dates"]), "source_sessions": list(part["source_sessions"]), "source_input_hashes": {x: records[x].input_sha256 for x in part["source_sessions"]}, "source_trial_pair_counts": source_trial_pair_counts, "basis_state_sha256": basis.source_state_sha256}
    return {**body, "basis": basis, "plan_sha256": lfmc4.sha256_bytes(lfmc4.canonical_json(body).encode())}


def _summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)[np.isfinite(values)]
    return {"count": int(finite.size), "mean": None if not finite.size else float(finite.mean()), "median": None if not finite.size else float(np.median(finite)), "min": None if not finite.size else float(finite.min()), "max": None if not finite.size else float(finite.max())}


def _rotation_seed(*, session: str, trial: float, replicate: int) -> int:
    body = f"{NULL_SEED}|{session}|{trial:.17g}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(body).digest()[:8], "big")


def _record_metrics(record: lfmc4.LFMC4Record, plan: Mapping[str, Any]) -> dict[str, Any]:
    basis: lfmc4.FrozenBasis = plan["basis"]
    first, second = record.trials
    m1, m2 = lfmc4.trial_moment(first, basis), lfmc4.trial_moment(second, basis)
    cosine = lfmc4.cosine_by_channel(m1["moment"], m2["moment"])
    gain = lfmc4.symmetric_transfer_gain(record, basis)
    # Coverage is the *same-channel* intersection of the two primary
    # quantities.  It is deliberately not a product of marginal fractions:
    # a cosine missing on channel i and a gain missing on channel j are two
    # invalid channels, not one probabilistic event.  The same intersection is
    # used for every rotation-null median.
    valid = np.isfinite(cosine) & np.isfinite(gain)
    nulls: list[dict[str, Any]] = []
    for replicate in range(NULL_REPLICATES):
        rotated_one, offsets_one = lfmc4.rotated_velocity(first, offset_seed=_rotation_seed(session=record.session_name, trial=first.trial_number, replicate=replicate))
        rotated_two, offsets_two = lfmc4.rotated_velocity(second, offset_seed=_rotation_seed(session=record.session_name, trial=second.trial_number, replicate=replicate))
        null_gain = lfmc4.symmetric_transfer_gain(record, basis, fit_velocity_one=rotated_one, fit_velocity_two=rotated_two)
        null_valid = valid & np.isfinite(null_gain)
        # Do not silently change the channel set under a null.  A null with a
        # missing channel in the correct intersection leaves this record
        # undefined instead of gaining a favourable subset.
        same_intersection = bool(np.array_equal(null_valid, valid))
        nulls.append({"replicate": replicate, "trial1_rotation_offsets": list(offsets_one), "trial2_rotation_offsets": list(offsets_two), "median_symmetric_gain": float(np.median(null_gain[valid])) if same_intersection and valid.any() else None, "defined_channels": int(null_valid.sum()), "same_correct_channel_intersection": same_intersection})
    null_values = np.asarray([row["median_symmetric_gain"] for row in nulls], dtype=np.float64)
    coverage = float(valid.mean())
    valid_hash = lfmc4.sha256_bytes(np.ascontiguousarray(valid.astype(np.uint8)).tobytes(order="C"))
    cosine_summary, gain_summary = _summary(cosine[valid]), _summary(gain[valid])
    return {
        "status": "defined" if valid.any() and np.isfinite(null_values).all() else "undefined",
        "session_name": record.session_name, "date": record.date, "input_nwb_sha256": record.input_sha256,
        "source_plan_sha256": plan["plan_sha256"],
        "support": {"trial_count": 2, "trial_numbers": [first.trial_number, second.trial_number], "native_bin_ms": 20, "activity_threshold_used": None, "query_labels_read": False, "trial_1": dict(first.audit), "trial_2": dict(second.audit)},
        "valid_channel_mask": {"rule": "finite correct cosine AND finite correct symmetric gain; the same mask is used by correct medians and every null median", "valid_channels": int(valid.sum()), "total_channels": int(valid.size), "mask_sha256": valid_hash},
        "split_trial_moment_cosine": {"metric": "per-channel cosine of trial moments", "median": cosine_summary["median"], "mean": cosine_summary["mean"], "defined_channels": int(valid.sum()), "total_channels": int(cosine.size), "channel_mask": "valid_channel_mask"},
        "correct_symmetric_held_trial_gain": {"metric": "fit-trial rate mean + fit-trial moment with source-only latent whitening", "median": gain_summary["median"], "mean": gain_summary["mean"], "defined_channels": int(valid.sum()), "total_channels": int(gain.size), "channel_mask": "valid_channel_mask", "target_normal_equations_solved": False},
        "label_rotation_null": {"replicates": NULL_REPLICATES, "seed": NULL_SEED, "scope": "nonzero deterministic rotations within each valid run of each fit trial; evaluation labels correct", "replicate_rows": nulls, "q95": float(np.quantile(null_values, .95)) if np.isfinite(null_values).all() else None},
        "coverage_fraction": coverage,
        "coverage_rule": "mean(isfinite(split_trial_moment_cosine) AND isfinite(correct_symmetric_held_trial_gain)) over all channels",
        "combined_m2_descriptor_sha256": lfmc4.array_hash(lfmc4.combine_trials(first, second, basis)),
    }


def _source_normalizer(records: Sequence[lfmc4.LFMC4Record], basis: lfmc4.FrozenBasis) -> dict[str, Any]:
    by_date: dict[str, list[lfmc4.LFMC4Record]] = {}
    for record in records:
        by_date.setdefault(record.date, []).append(record)
    if len(by_date) != 5:
        raise AuditError("source normalizer requires exactly five source dates")
    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    pair_counts: dict[str, int] = {}
    for date in sorted(by_date):
        date_records = by_date[date]
        for record in date_records:
            pairs = tuple(zip(record.source_trials[:-1], record.source_trials[1:]))
            if not pairs:
                raise AuditError(f"{record.session_name}: source normalizer has no contiguous M=2 pair")
            pair_counts[record.session_name] = len(pairs)
            for first, second in pairs:
                descriptor = lfmc4.combine_trials(first, second, basis)
                values.append(descriptor)
                # Date and recording each receive equal total mass; within a
                # recording every legal contiguous M=2 pair receives equal
                # mass, then every channel row within that descriptor does.
                weight = 1.0 / (len(by_date) * len(date_records) * len(pairs) * descriptor.shape[0])
                weights.append(np.full(descriptor.shape[0], weight, dtype=np.float64))
    rows, row_weights = np.concatenate(values, axis=0), np.concatenate(weights, axis=0)
    if not np.isclose(row_weights.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise AuditError("source normalizer equal weights do not sum to one")
    mean = np.sum(row_weights[:, None] * rows, axis=0)
    raw_std = np.sqrt(np.sum(row_weights[:, None] * (rows - mean[None, :]) ** 2, axis=0))
    body = {"field_names": ["cross_moment_0", "cross_moment_1", "cross_moment_2", "cross_moment_3"], "mean": mean.tolist(), "raw_std": raw_std.tolist(), "scale_after_floor": np.maximum(raw_std, NORMALIZER_SCALE_FLOOR).tolist(), "scale_floor": NORMALIZER_SCALE_FLOOR, "fit_rows": int(rows.shape[0]), "source_sessions": [x.session_name for x in records], "source_contiguous_m2_pair_counts": pair_counts, "fit_scope": "outer-source-dates-only", "weighting": "equal-date/equal-recording/equal-contiguous-M2-pair/equal-channel-row", "used_by_cpu_gate": False}
    return {**body, "normalizer_sha256": lfmc4.sha256_bytes(lfmc4.canonical_json(body).encode())}


def _date_row(*, record_rows: Sequence[Mapping[str, Any]], plan: Mapping[str, Any], normalizer: Mapping[str, Any]) -> dict[str, Any]:
    total = len(record_rows)
    defined = [row for row in record_rows if row["status"] == "defined"]
    def extract(row: Mapping[str, Any], *keys: str) -> float:
        obj: Any = row
        for key in keys:
            obj = obj[key]
        return float(obj)
    cosine = [extract(row, "split_trial_moment_cosine", "median") for row in defined]
    gain = [extract(row, "correct_symmetric_held_trial_gain", "median") for row in defined]
    coverage = [float(row["coverage_fraction"]) for row in defined]
    null_values: list[float] = []
    if len(defined) == total:
        for replica in range(NULL_REPLICATES):
            null_values.append(float(np.mean([row["label_rotation_null"]["replicate_rows"][replica]["median_symmetric_gain"] for row in defined])))
    status = "defined" if len(defined) == total and len(null_values) == NULL_REPLICATES else "undefined"
    correct_gain = float(np.mean(gain)) if len(gain) == total else None
    q95 = float(np.quantile(null_values, .95)) if len(null_values) == NULL_REPLICATES else None
    return {"status": status, "date": plan["outer_date"], "recordings_total": total, "recordings_defined": len(defined), "recording_sessions": [row["session_name"] for row in record_rows], "source_plan": {key: value for key, value in plan.items() if key != "basis"}, "basis": plan["basis"].as_dict(), "source_only_normalizer": dict(normalizer), "date_coverage_fraction": float(np.mean(coverage)) if len(coverage) == total else None, "date_split_trial_moment_cosine": float(np.mean(cosine)) if len(cosine) == total else None, "date_correct_symmetric_cross_trial_gain": correct_gain, "date_null_gain_replicates": null_values, "date_null_gain_q95": q95, "date_correct_minus_null_q95": None if correct_gain is None or q95 is None else correct_gain-q95, "aggregation": "equal-recording mean of record-level channel-median metrics", "recording_results": [dict(row) for row in record_rows]}


def cpu_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 6:
        raise AuditError("CPU gate requires exactly six dates")
    defined = all(row.get("status") == "defined" for row in rows)
    metric = lambda key: [row.get(key) if isinstance(row.get(key), (int, float)) and np.isfinite(float(row[key])) else None for row in rows]
    coverage, cosine, gain, margin = (metric(x) for x in ("date_coverage_fraction", "date_split_trial_moment_cosine", "date_correct_symmetric_cross_trial_gain", "date_correct_minus_null_q95"))
    metrics_defined = all(value is not None for series in (coverage, cosine, gain, margin) for value in series)
    counts = {"coverage": sum(x >= COVERAGE_THRESHOLD for x in coverage if x is not None), "cosine": sum(x >= COSINE_THRESHOLD for x in cosine if x is not None), "gain": sum(x > 0 for x in gain if x is not None), "null": sum(x > 0 for x in margin if x is not None)}
    passes = {key: bool(defined and metrics_defined and value >= MIN_PASSING_DATES) for key, value in counts.items()}
    passed = all(passes.values())
    return {"status": "PASS_LFMC4_FEASIBILITY" if passed else "STOP_LFMC4_FEASIBILITY_FAILED", "all_six_dates_defined": bool(defined and metrics_defined), "required_passing_dates_per_clause": MIN_PASSING_DATES, "subgates": {"coverage_at_least_090": {"threshold": COVERAGE_THRESHOLD, "comparison": ">=", "dates_passing": counts["coverage"], "pass": passes["coverage"]}, "split_trial_moment_cosine": {"threshold": COSINE_THRESHOLD, "comparison": ">=", "dates_passing": counts["cosine"], "pass": passes["cosine"]}, "correct_symmetric_held_trial_rate_prediction_gain": {"threshold": 0.0, "comparison": ">", "dates_passing": counts["gain"], "pass": passes["gain"], "target_normal_equations_solved": False}, "correct_gain_above_31_rotation_null_q95": {"threshold": 0.0, "comparison": ">", "dates_passing": counts["null"], "pass": passes["null"], "evaluation_labels_remain_correct": True}}, "conjunction": "all_six_dates_defined AND coverage>=0.90 AND split_trial_cosine>=0.5 AND correct_gain>0 AND correct_gain>null_q95, each in >=4/6 dates", "pass": passed, "gpu_authorized_by_this_gate": False}


def run_audit(records: Mapping[str, lfmc4.LFMC4Record]) -> dict[str, Any]:
    date_rows: list[dict[str, Any]] = []
    for part in partitions(records):
        try:
            plan = source_plan(records, part["outer_date"])
            normalizer = _source_normalizer([records[name] for name in part["source_sessions"]], plan["basis"])
            target_rows: list[dict[str, Any]] = []
            for name in part["target_sessions"]:
                try:
                    target_rows.append(_record_metrics(records[name], plan))
                except (AuditError, lfmc4.LFMC4Error, np.linalg.LinAlgError) as error:
                    target_rows.append({"status": "undefined", "session_name": name, "date": part["outer_date"], "reason": str(error)})
            date_rows.append(_date_row(record_rows=target_rows, plan=plan, normalizer=normalizer))
        except (AuditError, lfmc4.LFMC4Error, np.linalg.LinAlgError) as error:
            date_rows.append({"status": "undefined", "date": part["outer_date"], "recordings_total": len(part["target_sessions"]), "recordings_defined": 0, "recording_sessions": list(part["target_sessions"]), "reason": str(error)})
    gate = cpu_gate(date_rows)
    constants = {"protocol_version": PROTOCOL_VERSION, "null_replicates": NULL_REPLICATES, "null_seed": NULL_SEED, "coverage_threshold": COVERAGE_THRESHOLD, "cosine_threshold": COSINE_THRESHOLD, "minimum_passing_dates": MIN_PASSING_DATES, "basis": lfmc4.frozen_training_constants()}
    return {"schema": "h1_lfmc4_m2_date_lodo_cpu_feasibility_v1", "protocol_version": PROTOCOL_VERSION, "task": "h1", "status": "PASS_CPU_LFMC4_FEASIBILITY__NO_GPU_AUTHORIZATION" if gate["pass"] else "STOP_CPU_LFMC4_FEASIBILITY_FAILED__NO_GPU_AUTHORIZATION", "scope": {"opened_data": "exactly 13 public held-in-calib NWBs", "formal_heldout_opened": False, "minival_or_query_opened": False, "evalai_opened": False, "decoder_constructed": False, "optimizer_constructed": "source_only_basis_cpu_adam", "gpu_constructed": False, "target_backpropagation": False, "target_optimizer_steps": 0}, "carrier_contract": {"width": 4, "basis": "source-only bounded 7->16->4 tanh autoencoder", "target_descriptor": "native-20ms exposure-weighted spike/source-whitened-latent cross moment", "target_ridge": "forbidden", "target_lag_selection": "forbidden", "target_normal_equations": "forbidden", "fit_trial_predictor": "fit-trial rate mean + fit-trial moment; source-only latent whitening"}, "source_only_selection_contract": {"outer_unit": "calendar date", "source_dates_per_fold": 5, "target_date_used_for_basis_or_normalizer": False, "validation_or_outer_selection": False, "outer_mutation_isolation": "synthetic contract test mutates outer first-two and later trial values and verifies source plan hash plus target first-two gate inputs are unchanged"}, "input_nwb_sha256": {name: records[name].input_sha256 for name in sorted(records)}, "source_hashes": {"audit_script_sha256": sha256_file(Path(__file__).resolve()), "lfmc4_module_sha256": sha256_file(Path(lfmc4.__file__).resolve()), "constants_sha256": lfmc4.sha256_bytes(lfmc4.canonical_json(constants).encode())}, "frozen_basis_constants": lfmc4.frozen_training_constants(), "date_lodo": date_rows, "cpu_gate": gate, "decision": {"gpu_authorized_by_this_receipt": False, "if_fails": "stop LFMC4 H1 route; do not tune basis, source steps, threshold, null, descriptor width, target estimator, or fusion", "if_passes": "this CPU receipt alone does not authorize GPU; a separate reviewed exact-SPINT protocol is required"}}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "SPINT-main" / "data" / "000954")
    parser.add_argument("--output", type=Path, default=root / "sua_exploration" / "results" / "h1_lfmc4_m2_date_lodo_v1" / "cpu_feasibility_receipt.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError("CPU-only LFMC4 audit requires CUDA_VISIBLE_DEVICES unset or empty")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable LFMC4 receipt: {output}")
    paths = index_h1_heldin_calib(args.data_dir)
    receipt = run_audit({name: load_h1_record(path) for name, path in paths.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    output.chmod(0o444)
    print(json.dumps({"status": receipt["status"], "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
