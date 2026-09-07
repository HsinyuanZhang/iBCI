"""All-public-source data contract for the H1 CarrierID official candidate.

This module is intentionally independent of the date-LODO training wrappers.
It fits one CarrierID transform from all thirteen public ``held-in-calib``
recordings, freezes that transform in an immutable asset bundle, and exposes a
train-only Lightning DataModule.  It has no minival, held-out query, formal
test, EvalAI, validation, test, or prediction loader.

The same immutable transform is later consumed by the isolated packaging path
to derive a four-dimensional carrier from each *calibration* recording.  A
calibration carrier may use the calibration kinematics, but a query/test file
is rejected before any NWB loader is called.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import random
import stat
from typing import Any, Iterable, Mapping, Sequence
import uuid

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler

from src.data.h1_m4_eb_pilot import (
    EXPECTED_NEURONS,
    H1_HELDIN_SESSIONS,
    H1PilotRecord,
    MAX_TRIAL_LENGTH,
    SUPPORT_TRIALS,
    VELOCITY_DIM,
    array_sha256,
    carrier_sha256,
    fit_frozen_carrier,
    index_heldin_calib,
    interpolate_trial_identity,
    load_record,
    session_date,
    session_from_path,
)
from src.h1_m4_cce_contract import NORMALIZER_FLOOR, NORMALIZER_FORMULA, canonical_sha256, sha256_file


ALL_SOURCE_ASSET_SCHEMA = "h1_carrierid_all_public_source_assets_v1"
ALL_SOURCE_DATA_SCHEMA = "h1_carrierid_all_public_source_training_binding_v1"
ALL_SOURCE_PROTOCOL = "h1_carrierid_all_public_heldin_m4_q16_ridge100_v1"
ALL_SOURCE_Q = 16
ALL_SOURCE_RIDGE = 100.0
ALL_SOURCE_SEED = 42
ALL_SOURCE_EPOCHS = 50
ALL_SOURCE_BATCH = 32
ALL_SOURCE_WINDOW = 700
ALL_SOURCE_ASSET_FILE = "h1_carrierid_all_source_assets.npz"


class H1CarrierIdAllSourceError(ValueError):
    """A source, estimator, asset, or deployment-calibration invariant failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise H1CarrierIdAllSourceError(message)


def _immutable(path: Path) -> bool:
    return path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o444


def _publish_once(path: Path, payload: bytes) -> str:
    output = path.resolve()
    if output.exists():
        raise FileExistsError(f"all-source CarrierID refuses to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    _need(_immutable(output), f"all-source asset is not immutable mode 0444: {output}")
    return sha256_file(output)


@dataclass(frozen=True)
class AllSourceCarrierPlan:
    """The exact quantities consumed by ``fit_frozen_carrier``."""

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
    transform_sha256: str

    # Compatibility fields used only for provenance manifests in older helper
    # code.  The all-source plan is fitted directly and does not impersonate a
    # date-LODO receipt.
    outer_date: str = "ALL_PUBLIC_HELDIN"
    raw_plan_sha256: str = ""
    raw_receipt_sha256: str = ""
    eb_receipt_sha256: str = ""

    def manifest(self) -> dict[str, Any]:
        arrays = {"mean": self.mean, "scale": self.scale, "pcs": self.pcs, "U": self.U, "mu": self.mu}
        return {
            "protocol": ALL_SOURCE_PROTOCOL,
            "source_sessions": list(self.source_sessions),
            "source_input_sha256": list(self.source_input_sha256),
            "q": self.q,
            "lambda": self.ridge_lambda,
            "tau2": self.tau2,
            "array_shape": {name: list(value.shape) for name, value in arrays.items()},
            "array_sha256": {name: array_sha256(value) for name, value in arrays.items()},
            "transform_sha256": self.transform_sha256,
        }


@dataclass(frozen=True)
class AllSourceNormalizer:
    s_src: float
    source_cache_sha256: str
    entries: int
    rows: int
    dims: int
    normalizer_sha256: str

    @property
    def denominator(self) -> float:
        return max(float(self.s_src), NORMALIZER_FLOOR)

    def normalize(self, carrier: np.ndarray) -> np.ndarray:
        values = np.asarray(carrier, dtype=np.float64)
        _need(values.ndim >= 2 and values.shape[-1] == 4, f"carrier must be [...,4], got {values.shape}")
        _need(np.isfinite(values).all(), "carrier contains nonfinite values")
        return np.asarray(values / self.denominator, dtype=np.float64)

    def manifest(self) -> dict[str, Any]:
        return {
            "formula": NORMALIZER_FORMULA,
            "floor": NORMALIZER_FLOOR,
            "s_src": self.s_src,
            "denominator": self.denominator,
            "source_cache_sha256": self.source_cache_sha256,
            "entries": self.entries,
            "rows": self.rows,
            "dims": self.dims,
            "normalizer_sha256": self.normalizer_sha256,
        }


@dataclass(frozen=True)
class AllSourceAssets:
    plan: AllSourceCarrierPlan
    normalizer: AllSourceNormalizer
    manifest_path: Path
    manifest_sha256: str
    arrays_path: Path
    arrays_sha256: str


def _validate_records(records: Mapping[str, H1PilotRecord]) -> tuple[H1PilotRecord, ...]:
    _need(tuple(records) == H1_HELDIN_SESSIONS, "all-source records must be the exact ordered 13 held-in sessions")
    ordered = tuple(records[name] for name in H1_HELDIN_SESSIONS)
    _need(all(record.session_name == name for name, record in zip(H1_HELDIN_SESSIONS, ordered)), "session key drift")
    _need(all(record.num_neurons == EXPECTED_NEURONS for record in ordered), "H1 channel count drift")
    return ordered


def _support_rates(record: H1PilotRecord, values: Sequence[float]) -> np.ndarray:
    trials = [record.blocks_for(float(value)) for value in values]
    _need(len(trials) == SUPPORT_TRIALS and all(trial.rates.shape[0] >= 2 for trial in trials),
          f"{record.session_name}: earliest M=4 support is underspecified")
    return np.concatenate([trial.rates for trial in trials], axis=0)


def _canonicalize_columns(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def _fit_raw_rows(record: H1PilotRecord, plan: AllSourceCarrierPlan, values: Sequence[float]) -> np.ndarray:
    trials = [record.blocks_for(float(value)) for value in values]
    rates = np.concatenate([trial.rates for trial in trials], axis=0)
    labels = np.concatenate([trial.velocity for trial in trials], axis=0)
    z = ((rates - plan.mean[None, :]) / plan.scale[None, :]) @ plan.pcs[: plan.q].T
    design = np.column_stack((np.ones(z.shape[0]), z))
    regularizer = np.eye(design.shape[1], dtype=np.float64) * plan.ridge_lambda
    regularizer[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + regularizer, design.T @ labels)
    return (plan.pcs[: plan.q].T @ beta[1:]) / plan.scale[:, None]


def fit_all_source_plan(records: Mapping[str, H1PilotRecord]) -> AllSourceCarrierPlan:
    """Fit q=16/ridge=100/PCA/output-basis/EB prior from all public sources."""

    ordered = _validate_records(records)
    input_sha = tuple(record.input_sha256 for record in ordered)
    pooled_rates = np.concatenate([_support_rates(record, record.trial_values[:SUPPORT_TRIALS]) for record in ordered])
    mean = pooled_rates.mean(axis=0)
    scale = np.maximum(pooled_rates.std(axis=0), 1.0e-6)
    _u, _s, right = np.linalg.svd((pooled_rates - mean[None, :]) / scale[None, :], full_matrices=False)
    pcs = _canonicalize_columns(np.asarray(right[:ALL_SOURCE_Q], dtype=np.float64).T).T
    provisional = AllSourceCarrierPlan(
        H1_HELDIN_SESSIONS, input_sha, mean, scale, pcs, ALL_SOURCE_Q, ALL_SOURCE_RIDGE,
        np.zeros((VELOCITY_DIM, 4), dtype=np.float64), np.zeros(4, dtype=np.float64), 1.0, "",
    )
    pooled_rows = np.concatenate(
        [_fit_raw_rows(record, provisional, record.trial_values[:SUPPORT_TRIALS]) for record in ordered], axis=0
    )
    _u2, _s2, output_right = np.linalg.svd(pooled_rows, full_matrices=False)
    output_basis = _canonicalize_columns(np.asarray(output_right[:4].T, dtype=np.float64))
    source_carriers = pooled_rows @ output_basis
    mu = source_carriers.mean(axis=0)
    tau2 = float(np.square(source_carriers - mu[None, :]).sum() / (source_carriers.shape[0] * 4))
    _need(math.isfinite(tau2) and tau2 > 0.0, "all-source EB prior variance is undefined")
    body = {
        "protocol": ALL_SOURCE_PROTOCOL,
        "sessions": list(H1_HELDIN_SESSIONS),
        "input_sha256": list(input_sha),
        "q": ALL_SOURCE_Q,
        "lambda": ALL_SOURCE_RIDGE,
        "mean": array_sha256(mean),
        "scale": array_sha256(scale),
        "pcs": array_sha256(pcs),
        "U": array_sha256(output_basis),
        "mu": array_sha256(mu),
        "tau2": tau2,
    }
    transform_sha = canonical_sha256(body)
    return AllSourceCarrierPlan(
        H1_HELDIN_SESSIONS, input_sha, np.asarray(mean, np.float64), np.asarray(scale, np.float64), pcs,
        ALL_SOURCE_Q, ALL_SOURCE_RIDGE, output_basis, np.asarray(mu, np.float64), tau2, transform_sha,
        raw_plan_sha256=transform_sha,
    )


def legal_support_starts(record: H1PilotRecord) -> tuple[int, ...]:
    starts: list[int] = []
    for start in range(len(record.trial_values) - SUPPORT_TRIALS + 1):
        values = record.trial_values[start : start + SUPPORT_TRIALS]
        if all(record.blocks_for(value).rates.shape[0] >= 2 for value in values):
            for value in values:
                record.eval_trial_neural(value)
            starts.append(start)
    _need(bool(starts), f"{record.session_name}: no legal contiguous M=4 calibration support")
    return tuple(starts)


@dataclass(frozen=True)
class AllSourceCarrierEntry:
    session_name: str
    start_index: int
    trial_values: tuple[float, float, float, float]
    carrier: np.ndarray
    carrier_sha256: str


class AllSourceCarrierCache:
    def __init__(self, entries: Iterable[AllSourceCarrierEntry]) -> None:
        self.entries = tuple(entries)
        self._by_key = {(entry.session_name, entry.start_index): entry for entry in self.entries}
        _need(bool(self.entries) and len(self.entries) == len(self._by_key), "all-source cache is empty or duplicated")
        self.starts_by_session = {
            name: tuple(entry.start_index for entry in self.entries if entry.session_name == name)
            for name in H1_HELDIN_SESSIONS
        }
        _need(all(self.starts_by_session.values()), "all-source cache omits a held-in session")
        rows = [
            {"session": entry.session_name, "start": entry.start_index, "trials": list(entry.trial_values),
             "carrier_sha256": entry.carrier_sha256}
            for entry in self.entries
        ]
        self.cache_sha256 = canonical_sha256({"protocol": ALL_SOURCE_PROTOCOL, "entries": rows})

    def get(self, session: str, start: int) -> AllSourceCarrierEntry:
        try:
            return self._by_key[(str(session), int(start))]
        except KeyError as error:
            raise H1CarrierIdAllSourceError(f"uncached support block {session}:{start}") from error


def build_all_source_cache(
    records: Mapping[str, H1PilotRecord], plan: AllSourceCarrierPlan,
) -> tuple[AllSourceCarrierCache, AllSourceNormalizer]:
    _validate_records(records)
    entries: list[AllSourceCarrierEntry] = []
    for name in H1_HELDIN_SESSIONS:
        record = records[name]
        for start in legal_support_starts(record):
            values = tuple(float(value) for value in record.trial_values[start : start + SUPPORT_TRIALS])
            carrier = np.asarray(fit_frozen_carrier(record, plan, values)["carrier"], dtype=np.float64)
            entries.append(AllSourceCarrierEntry(name, start, values, carrier, carrier_sha256(carrier)))
    cache = AllSourceCarrierCache(entries)
    stacked = np.stack([entry.carrier for entry in entries], axis=0)
    scalar = float(np.sqrt(np.mean(np.square(stacked, dtype=np.float64), dtype=np.float64)))
    _need(math.isfinite(scalar) and scalar >= 0.0, "all-source normalizer is undefined")
    normalizer_body = {
        "formula": NORMALIZER_FORMULA, "floor": NORMALIZER_FLOOR, "s_src": scalar,
        "source_cache_sha256": cache.cache_sha256, "entries": int(stacked.shape[0]),
        "rows": int(stacked.shape[1]), "dims": int(stacked.shape[2]),
    }
    normalizer = AllSourceNormalizer(
        scalar, cache.cache_sha256, int(stacked.shape[0]), int(stacked.shape[1]), int(stacked.shape[2]),
        canonical_sha256(normalizer_body),
    )
    return cache, normalizer


def prepare_all_source_assets(*, data_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Create an immutable transform/normalizer bundle from exactly 13 held-in NWBs."""

    directory = Path(output_dir).resolve()
    manifest_path = directory / "H1_CARRIERID_ALL_SOURCE_ASSET_PREFLIGHT_v1.json"
    arrays_path = directory / ALL_SOURCE_ASSET_FILE
    if manifest_path.exists() or arrays_path.exists():
        raise FileExistsError("all-source asset preflight is one-shot and refuses partial/existing output")
    paths = index_heldin_calib(data_dir)
    records = {name: load_record(paths[name]) for name in H1_HELDIN_SESSIONS}
    plan = fit_all_source_plan(records)
    cache, normalizer = build_all_source_cache(records, plan)
    buffer = io.BytesIO()
    np.savez(
        buffer, mean=plan.mean, scale=plan.scale, pcs=plan.pcs, U=plan.U, mu=plan.mu,
        tau2=np.asarray(plan.tau2, np.float64), q=np.asarray(plan.q, np.int64),
        ridge_lambda=np.asarray(plan.ridge_lambda, np.float64), s_src=np.asarray(normalizer.s_src, np.float64),
    )
    arrays_sha = _publish_once(arrays_path, buffer.getvalue())
    body = {
        "schema": ALL_SOURCE_ASSET_SCHEMA,
        "status": "PASS_ALL_PUBLIC_HELDIN_ASSETS_FROZEN_NO_GPU_NO_FORMAL",
        "protocol": ALL_SOURCE_PROTOCOL,
        "source_sessions": list(H1_HELDIN_SESSIONS),
        "source_files": [
            {"session": name, "path": str(paths[name].resolve()), "sha256": records[name].input_sha256}
            for name in H1_HELDIN_SESSIONS
        ],
        "plan": plan.manifest(),
        "normalizer": normalizer.manifest(),
        "asset_file": ALL_SOURCE_ASSET_FILE,
        "asset_file_sha256": arrays_sha,
        "carrier_cache_entries": len(cache.entries),
        "scope": {
            "held_in_calibration_recordings_opened": 13,
            "minival_recordings_opened": 0,
            "held_out_query_recordings_opened": 0,
            "formal_test_labels_opened": 0,
            "evalai_accessed": False,
            "trainer_constructed": False,
            "cuda_used": False,
        },
    }
    manifest_sha = _publish_once(
        manifest_path, (json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    )
    return {**body, "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha}


def load_all_source_assets(manifest_path: str | Path) -> AllSourceAssets:
    path = Path(manifest_path).resolve()
    _need(_immutable(path), f"all-source asset manifest must be immutable mode 0444: {path}")
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise H1CarrierIdAllSourceError(f"invalid all-source asset manifest: {path}") from error
    _need(body.get("schema") == ALL_SOURCE_ASSET_SCHEMA, "all-source asset schema mismatch")
    _need(body.get("status") == "PASS_ALL_PUBLIC_HELDIN_ASSETS_FROZEN_NO_GPU_NO_FORMAL", "asset preflight did not pass")
    _need(tuple(body.get("source_sessions", ())) == H1_HELDIN_SESSIONS, "asset source roster drift")
    scope = body.get("scope", {})
    _need(scope.get("formal_test_labels_opened") == 0 and scope.get("evalai_accessed") is False,
          "asset manifest violates formal/EvalAI boundary")
    arrays_path = (path.parent / str(body.get("asset_file", ""))).resolve()
    _need(arrays_path.parent == path.parent and _immutable(arrays_path), "asset NPZ escapes manifest or is mutable")
    arrays_sha = sha256_file(arrays_path)
    _need(arrays_sha == body.get("asset_file_sha256"), "asset NPZ SHA drift")
    with np.load(arrays_path, allow_pickle=False) as archive:
        expected = {"mean", "scale", "pcs", "U", "mu", "tau2", "q", "ridge_lambda", "s_src"}
        _need(set(archive.files) == expected, f"asset NPZ key drift: {archive.files}")
        mean, scale, pcs = np.asarray(archive["mean"], np.float64), np.asarray(archive["scale"], np.float64), np.asarray(archive["pcs"], np.float64)
        U, mu = np.asarray(archive["U"], np.float64), np.asarray(archive["mu"], np.float64)
        tau2, q = float(archive["tau2"]), int(archive["q"])
        ridge, s_src = float(archive["ridge_lambda"]), float(archive["s_src"])
    _need(mean.shape == scale.shape == (EXPECTED_NEURONS,), "asset mean/scale shape drift")
    _need(pcs.shape == (ALL_SOURCE_Q, EXPECTED_NEURONS) and U.shape == (VELOCITY_DIM, 4) and mu.shape == (4,),
          "asset PCA/output basis shape drift")
    _need(q == ALL_SOURCE_Q and ridge == ALL_SOURCE_RIDGE and tau2 > 0.0 and s_src >= 0.0, "asset scalar drift")
    plan_body = body.get("plan", {})
    arrays = {"mean": mean, "scale": scale, "pcs": pcs, "U": U, "mu": mu}
    _need(all(array_sha256(value) == plan_body.get("array_sha256", {}).get(name) for name, value in arrays.items()),
          "asset plan array SHA drift")
    source_rows = body.get("source_files", [])
    _need(isinstance(source_rows, list) and len(source_rows) == 13, "asset source-file rows drift")
    source_hashes = tuple(str(row.get("sha256", "")) for row in source_rows)
    plan = AllSourceCarrierPlan(
        H1_HELDIN_SESSIONS, source_hashes, mean, scale, pcs, q, ridge, U, mu, tau2,
        str(plan_body.get("transform_sha256", "")), raw_plan_sha256=str(plan_body.get("transform_sha256", "")),
    )
    _need(plan.manifest() == plan_body, "asset plan manifest no longer matches its arrays")
    normalizer_body = body.get("normalizer", {})
    normalizer = AllSourceNormalizer(
        s_src, str(normalizer_body.get("source_cache_sha256", "")), int(normalizer_body.get("entries", 0)),
        int(normalizer_body.get("rows", 0)), int(normalizer_body.get("dims", 0)),
        str(normalizer_body.get("normalizer_sha256", "")),
    )
    _need(normalizer.manifest() == normalizer_body, "asset normalizer manifest drift")
    return AllSourceAssets(plan, normalizer, path, sha256_file(path), arrays_path, arrays_sha)


def assert_deployment_calibration_path(path: str | Path) -> Path:
    """Allow explicit calibration NWBs, while rejecting every query/label endpoint."""

    candidate = Path(path).resolve()
    lower = str(candidate).lower()
    _need(candidate.suffix.lower() == ".nwb" and "calib" in candidate.name.lower(),
          f"CarrierID packaging accepts explicit calibration NWBs only: {candidate}")
    forbidden = ("minival", "test_ecephys", "formal_query", "held-out-query", "heldout-query", "private_query")
    _need(not any(token in lower for token in forbidden), f"query/test path rejected before loader: {candidate}")
    _need(candidate.is_file(), f"calibration NWB is missing: {candidate}")
    return candidate


def load_deployment_calibration_record(path: str | Path) -> H1PilotRecord:
    """Load one explicit H1 calibration recording, including its permitted calibration targets."""

    resolved = assert_deployment_calibration_path(path)
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb
    from pynwb import NWBHDF5IO

    neural, velocity, trial_change, eval_mask = load_nwb(resolved, FalconTask.h1)
    with NWBHDF5IO(str(resolved), "r", load_namespaces=True) as handle:
        nwb = handle.read()
        _need("TrialNum" in nwb.acquisition, f"{resolved}: calibration TrialNum is missing")
        trial_num = np.asarray(nwb.acquisition["TrialNum"].data[:], dtype=np.float64)
    spikes64, velocity64 = np.asarray(neural, np.float64), np.asarray(velocity, np.float64)
    spikes, targets = spikes64.astype(np.float32), velocity64.astype(np.float32)
    changes, mask = np.asarray(trial_change, bool).reshape(-1), np.asarray(eval_mask, bool).reshape(-1)
    _need(spikes.ndim == 2 and spikes.shape[1] == EXPECTED_NEURONS, f"invalid H1 calibration neural shape {spikes.shape}")
    _need(targets.ndim == 2 and targets.shape[1] == VELOCITY_DIM, f"invalid H1 calibration target shape {targets.shape}")
    _need(len({spikes.shape[0], targets.shape[0], changes.size, mask.size, trial_num.size}) == 1,
          "deployment calibration arrays are misaligned")
    ordered = trial_num[mask & np.isfinite(trial_num)]
    _need(ordered.size > 0 and np.all(np.diff(ordered) >= 0.0), "calibration TrialNum is not chronological")
    values: list[float] = []
    for value in ordered.tolist():
        if not values or float(value) != values[-1]:
            values.append(float(value))
    _need(len(values) >= SUPPORT_TRIALS, "deployment calibration has fewer than four trials")
    # Reuse the exact 100-ms block constructor through the held-in record type.
    from src.data.h1_m4_eb_pilot import _trial_blocks
    trials = tuple(_trial_blocks(value, spikes64, velocity64, mask, trial_num) for value in values)
    name = session_from_path(resolved)
    return H1PilotRecord(
        name, session_date(name), resolved, sha256_file(resolved), spikes, targets, changes, mask, trial_num,
        tuple(values), trials,
    )


class H1AllSourceDataset(Dataset):
    def __init__(self, records: Mapping[str, H1PilotRecord], cache: AllSourceCarrierCache, normalizer: AllSourceNormalizer):
        self.records, self.cache, self.normalizer = records, cache, normalizer
        self.neural_data: dict[str, np.ndarray] = {}
        self.target_data: dict[str, np.ndarray] = {}
        self.eval_mask: dict[str, np.ndarray] = {}
        self.window_indices: list[tuple[str, int]] = []
        self._identity: dict[tuple[str, float], np.ndarray] = {}
        prehistory = ALL_SOURCE_WINDOW - 1
        for name in H1_HELDIN_SESSIONS:
            record = records[name]
            self.neural_data[name] = np.pad(record.neural, ((prehistory, 0), (0, 0)), constant_values=0.0)
            self.target_data[name] = np.pad(record.velocity, ((prehistory, 0), (0, 0)), constant_values=0.0)
            self.eval_mask[name] = np.pad(record.eval_mask, (prehistory, 0), constant_values=False)
            for start in cache.starts_by_session[name]:
                for value in cache.get(name, start).trial_values:
                    self._identity.setdefault((name, value), interpolate_trial_identity(record, value))
            for start in range(self.neural_data[name].shape[0] - ALL_SOURCE_WINDOW + 1):
                if self.eval_mask[name][start + ALL_SOURCE_WINDOW - 1]:
                    self.window_indices.append((name, start))
        _need(bool(self.window_indices), "all-source dataset contains no eval-valid windows")

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, request: tuple[int, int]):
        _need(isinstance(request, tuple) and len(request) == 2, "all-source samples require (window,support-start)")
        index, support_start = int(request[0]), int(request[1])
        session, start = self.window_indices[index]
        entry = self.cache.get(session, support_start)
        identity = np.stack([self._identity[(session, value)] for value in entry.trial_values], axis=0)
        return (
            self.neural_data[session][start : start + ALL_SOURCE_WINDOW],
            self.target_data[session][start : start + ALL_SOURCE_WINDOW],
            identity,
            session,
            self.normalizer.normalize(entry.carrier).astype(np.float32),
        )


class H1AllSourceSchedule(Sampler[list[tuple[int, int]]]):
    def __init__(self, dataset: H1AllSourceDataset) -> None:
        grouped = {name: [] for name in H1_HELDIN_SESSIONS}
        for index, (session, _start) in enumerate(dataset.window_indices):
            grouped[session].append(index)
        batches: list[list[int]] = []
        for name in H1_HELDIN_SESSIONS:
            indices = random.Random(ALL_SOURCE_SEED).sample(grouped[name], len(grouped[name]))
            batches.extend(
                indices[offset : offset + ALL_SOURCE_BATCH]
                for offset in range(0, len(indices), ALL_SOURCE_BATCH)
                if len(indices[offset : offset + ALL_SOURCE_BATCH]) == ALL_SOURCE_BATCH
            )
        self.dataset = dataset
        self.batches = tuple(tuple(batch) for batch in random.Random(ALL_SOURCE_SEED).sample(batches, len(batches)))
        flat = [index for batch in self.batches for index in batch]
        flat_sessions = np.asarray([dataset.window_indices[index][0] for index in flat], dtype=object)
        schedule = np.empty((ALL_SOURCE_EPOCHS, len(flat)), dtype=np.int16)
        for name in H1_HELDIN_SESSIONS:
            positions = np.flatnonzero(flat_sessions == name)
            legal = np.asarray(dataset.cache.starts_by_session[name], dtype=np.int16)
            token = hashlib.sha256(f"{ALL_SOURCE_PROTOCOL}|{ALL_SOURCE_SEED}|{name}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(token[:8], "big"))
            schedule[:, positions] = legal[rng.integers(0, len(legal), size=(ALL_SOURCE_EPOCHS, len(positions)))]
        self.flat = tuple(flat)
        self.schedule = schedule
        self.batch_order_sha256 = array_sha256(np.asarray(flat, np.int64))
        self.schedule_sha256 = array_sha256(schedule)
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        if self._epoch >= ALL_SOURCE_EPOCHS:
            raise RuntimeError("all-source fixed schedule exhausted after epoch 49")
        starts = self.schedule[self._epoch]
        offset = 0
        for batch in self.batches:
            selected = starts[offset : offset + len(batch)]
            offset += len(batch)
            yield [(index, int(start)) for index, start in zip(batch, selected)]
        self._epoch += 1


class H1CarrierIdAllSourceDataModule(pl.LightningDataModule):
    """Train-only all-public-held-in DataModule for the official H-C candidate."""

    def __init__(
        self, *, task: str, data_dir: str, asset_manifest_path: str, batch_size: int = ALL_SOURCE_BATCH,
        window_size: int = ALL_SOURCE_WINDOW, calibration_n_trials: int = SUPPORT_TRIALS,
        max_trial_length: int = MAX_TRIAL_LENGTH, seed: int = ALL_SOURCE_SEED,
        fixed_epochs: int = ALL_SOURCE_EPOCHS, num_workers: int = 0, pin_memory: bool = False,
    ) -> None:
        super().__init__()
        fixed = {
            "task": str(task).lower() == "h1", "batch": int(batch_size) == ALL_SOURCE_BATCH,
            "window": int(window_size) == ALL_SOURCE_WINDOW, "support": int(calibration_n_trials) == SUPPORT_TRIALS,
            "trial_length": int(max_trial_length) == MAX_TRIAL_LENGTH, "seed": int(seed) == ALL_SOURCE_SEED,
            "epochs": int(fixed_epochs) == ALL_SOURCE_EPOCHS, "workers": int(num_workers) == 0,
        }
        _need(all(fixed.values()), f"all-source fixed data contract violated: {fixed}")
        self.save_hyperparameters(logger=False)
        self._setup_done = False

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit"}:
            raise RuntimeError("all-source CarrierID DataModule implements fit only")
        if self._setup_done:
            return
        assets = load_all_source_assets(self.hparams.asset_manifest_path)
        paths = index_heldin_calib(self.hparams.data_dir)
        _need(
            tuple(sha256_file(paths[name]) for name in H1_HELDIN_SESSIONS) == assets.plan.source_input_sha256,
            "current held-in files differ from frozen all-source assets",
        )
        records = {name: load_record(paths[name]) for name in H1_HELDIN_SESSIONS}
        cache, observed_normalizer = build_all_source_cache(records, assets.plan)
        _need(observed_normalizer.manifest() == assets.normalizer.manifest(), "reconstructed all-source normalizer drift")
        dataset = H1AllSourceDataset(records, cache, assets.normalizer)
        schedule = H1AllSourceSchedule(dataset)
        self.assets, self.records, self.cache = assets, records, cache
        self.train_dataset, self.train_batch_sampler = dataset, schedule
        self._setup_done = True

    def train_dataloader(self) -> DataLoader:
        if not self._setup_done:
            raise RuntimeError("call setup('fit') before all-source train_dataloader")
        return DataLoader(self.train_dataset, batch_sampler=self.train_batch_sampler, num_workers=0,
                          pin_memory=bool(self.hparams.pin_memory))

    def val_dataloader(self) -> list[Any]:
        return []

    def test_dataloader(self):
        raise RuntimeError("all-source official candidate has no local test loader")

    def predict_dataloader(self):
        raise RuntimeError("all-source official candidate has no formal/query loader")

    def all_source_manifest(self) -> dict[str, Any]:
        if not self._setup_done:
            raise RuntimeError("all-source binding is not set up")
        return {
            "schema": ALL_SOURCE_DATA_SCHEMA,
            "asset_manifest_path": str(self.assets.manifest_path),
            "asset_manifest_sha256": self.assets.manifest_sha256,
            "source_sessions": list(H1_HELDIN_SESSIONS),
            "source_input_sha256": list(self.assets.plan.source_input_sha256),
            "carrier_transform_sha256": self.assets.plan.transform_sha256,
            "normalizer_sha256": self.assets.normalizer.normalizer_sha256,
            "batch_order_sha256": self.train_batch_sampler.batch_order_sha256,
            "schedule_sha256": self.train_batch_sampler.schedule_sha256,
            "epochs": ALL_SOURCE_EPOCHS,
            "formal_test_labels_opened": 0,
            "minival_recordings_opened": 0,
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
            "warm_start_forbidden": True,
        }
