"""Strict outer-date target view for H1 CarrierID date-LODO Phase-2.

This module is intentionally isolated from every source-training import path.
It can be imported by a future terminal evaluator only after both source e49
checkpoints have passed the paired terminal checker.  The target dataset uses
only the first four chronological trials as labelled calibration support and
evaluates 700-bin windows beginning at the fifth trial, so no query history
crosses the support boundary.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import stat
from typing import Any, Mapping

import numpy as np
from torch.utils.data import Dataset

from src.data.h1_carrierid_date_lodo_source import SourceRmsNormalizer, target_sessions_for_date, validate_source_bundle_manifest
from src.data.h1_m4_eb_pilot import (
    EXPECTED_NEURONS,
    MAX_TRIAL_LENGTH,
    SUPPORT_TRIALS,
    WINDOW,
    H1PilotRecord,
    PilotDataError,
    fit_frozen_carrier,
    index_heldin_calib,
    interpolate_trial_identity,
    load_record,
)
from src.h1_m4_cce_contract import NORMALIZER_FLOOR, array_sha256, canonical_sha256, sha256_file


TARGET_SCHEMA = "h1_carrierid_date_lodo_phase2_strict_target_view_v1"


class DateLodoTargetError(ValueError):
    """A strict date-LODO target boundary or source-artifact check failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise DateLodoTargetError(message)


def _immutable(path: Path) -> bool:
    return path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o444


@dataclass(frozen=True)
class DateLodoFrozenPlan:
    """The persisted Phase-1 estimator arrays needed for an outer-date fit."""

    outer_date: str
    source_sessions: tuple[str, ...]
    source_input_sha256: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    pcs: np.ndarray
    q: int
    ridge_lambda: float
    U: np.ndarray
    mu: np.ndarray
    tau2: float


def load_target_dependencies(source_manifest_path: str | Path, *, outer_date: str) -> tuple[DateLodoFrozenPlan, SourceRmsNormalizer, dict[str, Any]]:
    """Load only immutable Phase-1 artifacts; no recording bytes are opened."""

    manifest_path = Path(source_manifest_path).resolve()
    _need(_immutable(manifest_path), f"source manifest must be immutable 0444: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _need(isinstance(manifest, dict), "source manifest must be an object")
    validate_source_bundle_manifest(manifest, outer_date=str(outer_date))
    plan_row = manifest.get("frozen_plan")
    normalizer_row = manifest.get("normalizer")
    _need(isinstance(plan_row, Mapping) and isinstance(normalizer_row, Mapping), "source manifest lacks frozen plan/normalizer")
    plan_manifest_path = Path(str(plan_row.get("manifest_path", ""))).resolve()
    plan_arrays_path = plan_manifest_path.with_name("frozen_m4_plan.npz")
    _need(_immutable(plan_manifest_path) and _immutable(plan_arrays_path), "frozen plan artifacts must be immutable")
    _need(sha256_file(plan_manifest_path) == plan_row.get("manifest_sha256"), "frozen plan manifest SHA drift")
    plan_body = json.loads(plan_manifest_path.read_text(encoding="utf-8"))
    _need(plan_body.get("outer_date") == str(outer_date) and tuple(plan_body.get("source_sessions", ())) == tuple(manifest["source_sessions"]),
          "frozen plan date/source partition drift")
    source_rows = manifest.get("source_files")
    _need(isinstance(source_rows, list), "source manifest lacks source file hashes")
    source_hashes = tuple(str(row.get("sha256", "")) for row in source_rows)
    _need(tuple(plan_body.get("source_input_sha256", ())) == source_hashes,
          "frozen plan source input hashes drift")
    for field in ("transform_sha256", "raw_receipt_sha256", "eb_receipt_sha256"):
        _need(plan_body.get(field) == plan_row.get(field), f"frozen plan {field} drift")
    expected_hashes, expected_shapes = plan_body.get("array_sha256"), plan_body.get("array_shape")
    _need(isinstance(expected_hashes, Mapping) and isinstance(expected_shapes, Mapping)
          and set(expected_hashes) == set(expected_shapes) == {"mean", "scale", "pcs", "U", "mu"},
          "frozen plan lacks complete immutable array authority")
    with np.load(plan_arrays_path, allow_pickle=False) as values:
        keys = {"mean", "scale", "pcs", "q", "lambda", "U", "mu", "tau2"}
        _need(set(values.files) == keys, f"unexpected frozen plan arrays: {values.files}")
        arrays = {name: np.asarray(values[name], dtype=np.float64) for name in ("mean", "scale", "pcs", "U", "mu")}
        for name, array in arrays.items():
            _need(list(array.shape) == list(expected_shapes[name]) and array_sha256(array) == expected_hashes[name],
                  f"frozen plan {name} array shape/SHA drift")
        q, ridge_lambda, tau2 = int(values["q"]), float(values["lambda"]), float(values["tau2"])
        _need(q == int(plan_body.get("q", -1)) and ridge_lambda == float(plan_body.get("lambda", float("nan")))
              and tau2 == float(plan_body.get("tau2", float("nan"))), "frozen plan scalar q/lambda/tau2 drift")
        plan = DateLodoFrozenPlan(
            outer_date=str(outer_date), source_sessions=tuple(str(item) for item in plan_body["source_sessions"]),
            source_input_sha256=tuple(str(item) for item in plan_body["source_input_sha256"]),
            mean=arrays["mean"], scale=arrays["scale"], pcs=arrays["pcs"], q=q,
            ridge_lambda=ridge_lambda, U=arrays["U"], mu=arrays["mu"], tau2=tau2,
        )
    _need(plan.mean.shape == plan.scale.shape == (EXPECTED_NEURONS,) and plan.pcs.ndim == 2 and plan.U.shape == (7, 4)
          and plan.U.shape == tuple(expected_shapes["U"])
          and plan.mu.shape == (4,) and 0 < plan.q <= plan.pcs.shape[0] and np.isfinite(plan.mean).all()
          and np.isfinite(plan.scale).all() and np.all(plan.scale > 0.0) and np.isfinite(plan.U).all()
          and np.isfinite(plan.mu).all() and np.isfinite(plan.tau2) and plan.tau2 > 0.0,
          "frozen plan has invalid finite/shape contract")
    normalizer_path = Path(str(normalizer_row.get("manifest_path", ""))).resolve()
    _need(_immutable(normalizer_path) and sha256_file(normalizer_path) == normalizer_row.get("manifest_file_sha256"),
          "normalizer manifest is missing, mutable, or hash-drifted")
    normalizer_body = json.loads(normalizer_path.read_text(encoding="utf-8"))
    normalizer = SourceRmsNormalizer(
        s_src=float(normalizer_body["s_src"]), source_cache_sha256=str(normalizer_body["source_cache_sha256"]),
        entries=int(normalizer_body["entries"]), rows=int(normalizer_body["rows"]), dims=int(normalizer_body["dims"]),
        normalizer_sha256=str(normalizer_body["normalizer_sha256"]),
    )
    _need(normalizer.normalizer_sha256 == normalizer_row.get("normalizer_sha256") and normalizer.rows == EXPECTED_NEURONS
          and normalizer.dims == 4 and np.isfinite(normalizer.s_src) and normalizer.s_src >= 0.0
          and normalizer.denominator >= NORMALIZER_FLOOR, "source normalizer binding drift")
    return plan, normalizer, manifest


def load_outer_date_target_records(data_dir: str | Path, *, outer_date: str) -> dict[str, H1PilotRecord]:
    """The first byte-opening operation in the date-LODO terminal route."""

    paths = index_heldin_calib(data_dir)
    target_sessions = tuple(target_sessions_for_date(str(outer_date)))
    records = {name: load_record(paths[name]) for name in target_sessions}
    _need(tuple(records) == target_sessions and all(record.date == str(outer_date) for record in records.values()),
          "outer-date target loader partition drift")
    return records


def _first_valid_bin(record: H1PilotRecord, trial_value: float) -> int:
    indices = np.flatnonzero(record.eval_mask & np.isfinite(record.trial_num) & (record.trial_num == float(trial_value)))
    _need(indices.size > 0, f"{record.session_name}: fifth trial has no eval-valid bin")
    return int(indices[0])


@dataclass(frozen=True)
class DateLodoTargetSupport:
    session_name: str
    support_trials: tuple[float, float, float, float]
    fifth_trial: float
    query_first_bin: int
    identity: np.ndarray
    normalized_carrier: np.ndarray
    support_sha256: str
    carrier_sha256: str


class H1CarrierIdDateLodoStrictTargetDataset(Dataset):
    """All returned 700-bin targets lie entirely after chronological M=4 support."""

    def __init__(self, records: Mapping[str, H1PilotRecord], plan: DateLodoFrozenPlan,
                 normalizer: SourceRmsNormalizer, *, outer_date: str) -> None:
        expected = tuple(target_sessions_for_date(str(outer_date)))
        _need(tuple(records) == expected, "strict target records/order differ from the declared outer date")
        self.records = {name: records[name] for name in expected}
        self.plan, self.normalizer, self.outer_date = plan, normalizer, str(outer_date)
        self.support: dict[str, DateLodoTargetSupport] = {}
        self.window_indices: list[tuple[str, int]] = []
        for name in expected:
            record = self.records[name]
            _need(len(record.trial_values) >= SUPPORT_TRIALS + 1, f"{name}: fewer than five chronological trials")
            values = tuple(float(value) for value in record.trial_values[:SUPPORT_TRIALS])
            fifth = float(record.trial_values[SUPPORT_TRIALS])
            _need(all(record.blocks_for(value).rates.shape[0] >= 2 for value in values), f"{name}: M=4 support is underspecified")
            boundary = _first_valid_bin(record, fifth)
            identity = np.stack([interpolate_trial_identity(record, value) for value in values], axis=0).astype(np.float32)
            carrier = normalizer.normalize(fit_frozen_carrier(record, plan, values)["carrier"]).astype(np.float32)
            _need(identity.shape == (SUPPORT_TRIALS, MAX_TRIAL_LENGTH, EXPECTED_NEURONS)
                  and carrier.shape == (EXPECTED_NEURONS, 4) and np.isfinite(identity).all() and np.isfinite(carrier).all(),
                  f"{name}: target identity/carrier finite-shape drift")
            digest = hashlib.sha256()
            digest.update(np.asarray(values, dtype=np.float64).tobytes())
            digest.update(identity.tobytes())
            for value in values:
                trial = record.blocks_for(value)
                digest.update(trial.rates.tobytes()); digest.update(trial.velocity.tobytes()); digest.update(trial.block_indices.tobytes())
            support = DateLodoTargetSupport(name, values, fifth, boundary, identity, carrier, digest.hexdigest(),
                                            hashlib.sha256(np.ascontiguousarray(carrier).tobytes()).hexdigest())
            self.support[name] = support
            for start in range(boundary, record.neural.shape[0] - WINDOW + 1):
                if bool(record.eval_mask[start + WINDOW - 1]):
                    self.window_indices.append((name, int(start)))
        _need(bool(self.window_indices), "strict outer-date target has no post-support windows")
        self.window_indices_sha256 = canonical_sha256(self.window_indices)

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, index: int):
        session, start = self.window_indices[int(index)]
        record, support = self.records[session], self.support[session]
        end = start + WINDOW
        _need(start >= support.query_first_bin and end <= record.neural.shape[0] and bool(record.eval_mask[end - 1]),
              "strict target window crossed support boundary or invalid eval bin")
        return (np.asarray(record.neural[start:end], dtype=np.float32), np.asarray(record.velocity[start:end], dtype=np.float32),
                support.identity, session, support.normalized_carrier)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": TARGET_SCHEMA, "outer_date": self.outer_date, "sessions": list(self.records),
            "window_indices_sha256": self.window_indices_sha256, "samples": len(self.window_indices),
            "support": {name: {"support_trials": list(item.support_trials), "fifth_trial": item.fifth_trial,
                                 "query_first_bin": item.query_first_bin, "support_sha256": item.support_sha256,
                                 "normalized_carrier_sha256": item.carrier_sha256,
                                 "identity_sha256": array_sha256(item.identity)} for name, item in self.support.items()},
            "all_query_histories_start_at_or_after_fifth_trial": True,
        }
