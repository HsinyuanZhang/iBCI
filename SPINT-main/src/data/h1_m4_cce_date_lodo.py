"""Five-date, source-only H1 M=4 CCE data path.

The module deliberately does not inherit the V2/fold-0 DataModule.  Its
source/target partition is reconstructed from the requested outer date for
each run.  Training opens only that fold's source recordings; target public
held-in-calib recordings are opened exclusively by the terminal evaluator
after both paired checkpoints have been validated.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import lightning.pytorch as pl
import numpy as np
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, Dataset, Sampler

from src.data.h1_m4_eb_pilot import (
    BLOCK_BINS,
    EPS,
    EXPECTED_NEURONS,
    H1_HELDIN_SESSIONS,
    MAX_TRIAL_LENGTH,
    ROTATION_SEED,
    TrialBlocks,
    H1PilotRecord,
    PilotDataError,
    array_sha256,
    carrier_sha256,
    fit_frozen_carrier,
    index_heldin_calib,
    interpolate_trial_identity,
    load_record,
    session_date,
)
from src.h1_m4_cce_contract import (
    CONFIRMATORY_DATES,
    FIXED_EPOCHS,
    FIXED_SEED,
    NORMALIZER_FLOOR,
    NORMALIZER_FORMULA,
    SUPPORT_TRIALS,
    WINDOW_SIZE,
    CCEContractError,
    SourceScalarNormalizer,
    assert_confirmatory_date,
    canonical_sha256,
    fit_source_scalar_normalizer,
    immutable_mode_0444,
    reject_nonpublic_heldin_scope,
    sha256_file,
)


CCE_PLAN_SCHEMA = "h1_m4_cce_date_lodo_frozen_eb_plan_v1"
CCE_CACHE_SCHEMA = "h1_m4_cce_date_lodo_all_source_carrier_cache_v1"
CCE_NORMALIZED_CACHE_SCHEMA = "h1_m4_cce_date_lodo_normalized_source_carrier_cache_v1"
CCE_SCHEDULE_SCHEMA = "h1_m4_cce_date_lodo_shared_50epoch_schedule_v1"
CCE_SOURCE_MANIFEST_SCHEMA = "h1_m4_cce_date_lodo_source_training_manifest_v1"


def target_sessions_for_date(outer_date: str) -> tuple[str, ...]:
    outer_date = assert_confirmatory_date(outer_date)
    targets = tuple(name for name in H1_HELDIN_SESSIONS if session_date(name) == outer_date)
    if not targets:
        raise CCEContractError(f"no H1 held-in recordings for CCE date {outer_date}")
    return targets


def source_sessions_for_date(outer_date: str) -> tuple[str, ...]:
    outer_date = assert_confirmatory_date(outer_date)
    targets = set(target_sessions_for_date(outer_date))
    source = tuple(name for name in H1_HELDIN_SESSIONS if name not in targets)
    if len(source) + len(targets) != len(H1_HELDIN_SESSIONS) or any(session_date(name) == outer_date for name in source):
        raise CCEContractError("CCE source/target date partition is invalid")
    return source


def _write_json_once(path: Path, body: Mapping[str, Any]) -> str:
    encoded = (json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if path.exists():
        if not immutable_mode_0444(path) or path.read_bytes() != encoded:
            raise CCEContractError(f"existing CCE artifact drift: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
        path.chmod(0o444)
    return sha256_file(path)


def _write_npz_once(path: Path, **arrays: np.ndarray) -> str:
    if path.exists():
        if not immutable_mode_0444(path):
            raise CCEContractError(f"existing CCE array artifact is mutable: {path}")
        with np.load(path, allow_pickle=False) as prior:
            if set(prior.files) != set(arrays) or any(not np.array_equal(prior[name], value) for name, value in arrays.items()):
                raise CCEContractError(f"existing CCE array artifact drift: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        path.chmod(0o444)
    return sha256_file(path)


def _write_npy_once(path: Path, array: np.ndarray) -> str:
    if path.exists():
        if not immutable_mode_0444(path) or not np.array_equal(np.load(path, allow_pickle=False), array):
            raise CCEContractError(f"existing CCE schedule drift: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array, allow_pickle=False)
        path.chmod(0o444)
    return sha256_file(path)


def load_source_records_for_date(data_dir: str | Path, outer_date: str) -> dict[str, H1PilotRecord]:
    outer_date = assert_confirmatory_date(outer_date)
    reject_nonpublic_heldin_scope(data_dir)
    paths = index_heldin_calib(data_dir)
    source = source_sessions_for_date(outer_date)
    records = {name: load_record(paths[name]) for name in source}
    if tuple(records) != source or any(record.date == outer_date for record in records.values()):
        raise CCEContractError("CCE target date leaked into source loader")
    return records


def load_target_records_for_date(data_dir: str | Path, outer_date: str) -> dict[str, H1PilotRecord]:
    outer_date = assert_confirmatory_date(outer_date)
    reject_nonpublic_heldin_scope(data_dir)
    paths = index_heldin_calib(data_dir)
    targets = target_sessions_for_date(outer_date)
    records = {name: load_record(paths[name]) for name in targets}
    if tuple(records) != targets or any(record.date != outer_date for record in records.values()):
        raise CCEContractError("CCE non-target date leaked into target loader")
    return records


@dataclass(frozen=True)
class CCEFrozenPlan:
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
    raw_plan_sha256: str
    raw_receipt_sha256: str
    eb_receipt_sha256: str
    transform_sha256: str

    def manifest(self) -> dict[str, Any]:
        arrays = {"mean": self.mean, "scale": self.scale, "pcs": self.pcs, "U": self.U, "mu": self.mu}
        return {
            "schema": CCE_PLAN_SCHEMA,
            "outer_date": self.outer_date,
            "source_sessions": list(self.source_sessions),
            "source_input_sha256": list(self.source_input_sha256),
            "q": self.q,
            "lambda": self.ridge_lambda,
            "tau2": self.tau2,
            "raw_plan_sha256": self.raw_plan_sha256,
            "raw_receipt_sha256": self.raw_receipt_sha256,
            "eb_receipt_sha256": self.eb_receipt_sha256,
            "array_sha256": {name: array_sha256(value) for name, value in arrays.items()},
            "array_shape": {name: list(value.shape) for name, value in arrays.items()},
            "transform_sha256": self.transform_sha256,
        }


def _receipt_date_row(receipt: Mapping[str, Any], outer_date: str) -> Mapping[str, Any]:
    candidates = [row for row in receipt.get("date_lodo", []) if row.get("date") == outer_date]
    if len(candidates) != 1 or not isinstance(candidates[0].get("plan"), Mapping):
        raise CCEContractError(f"receipt lacks one valid LODO plan for {outer_date}")
    return candidates[0]


def _support_rates(record: H1PilotRecord, values: Sequence[float]) -> np.ndarray:
    trials = [record.blocks_for(value) for value in values]
    if len(trials) != SUPPORT_TRIALS or any(trial.rates.shape[0] < 2 for trial in trials):
        raise CCEContractError(f"{record.session_name}: illegal earliest contiguous M=4 source support")
    return np.concatenate([trial.rates for trial in trials], axis=0)


def _project(rates: np.ndarray, plan: CCEFrozenPlan) -> np.ndarray:
    return ((rates - plan.mean[None, :]) / plan.scale[None, :]) @ plan.pcs[: plan.q].T


def _fit_without_eb(record: H1PilotRecord, plan: CCEFrozenPlan, values: Sequence[float]) -> dict[str, np.ndarray]:
    trials = [record.blocks_for(value) for value in values]
    rates = np.concatenate([trial.rates for trial in trials])
    labels = np.concatenate([trial.velocity for trial in trials])
    z = _project(rates, plan)
    design = np.column_stack((np.ones(len(z)), z))
    regularizer = np.eye(design.shape[1]) * plan.ridge_lambda
    regularizer[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + regularizer, design.T @ labels)
    raw_rows = (plan.pcs[: plan.q].T @ beta[1:]) / plan.scale[:, None]
    return {"beta": beta, "raw_rows": raw_rows}


def reconstruct_plan_for_date(
    records: Mapping[str, H1PilotRecord],
    outer_date: str,
    raw_receipt_path: str | Path,
    eb_receipt_path: str | Path,
) -> CCEFrozenPlan:
    """Reconstruct one source-only date-LODO transform and EB prior.

    The pre-existing raw/EB receipts fix q, ridge lambda, source partition, and
    EB prior.  This function recomputes all source arrays from the exact source
    NWBs and fails if the receipt binding is not reproduced.
    """

    outer_date = assert_confirmatory_date(outer_date)
    raw_path, eb_path = Path(raw_receipt_path).resolve(), Path(eb_receipt_path).resolve()
    reject_nonpublic_heldin_scope(raw_path)
    reject_nonpublic_heldin_scope(eb_path)
    if not immutable_mode_0444(raw_path) or not immutable_mode_0444(eb_path):
        raise CCEContractError("CCE raw/EB source receipts must be immutable mode 0444")
    raw_sha, eb_sha = sha256_file(raw_path), sha256_file(eb_path)
    raw_receipt = json.loads(raw_path.read_text(encoding="utf-8"))
    eb_receipt = json.loads(eb_path.read_text(encoding="utf-8"))
    raw_row, eb_row = _receipt_date_row(raw_receipt, outer_date), _receipt_date_row(eb_receipt, outer_date)
    raw_plan, eb_plan = raw_row["plan"], eb_row["plan"]
    expected_source = source_sessions_for_date(outer_date)
    source = tuple(raw_plan.get("source_sessions", ()))
    if source != expected_source or tuple(eb_plan.get("source_sessions", ())) != source:
        raise CCEContractError("CCE receipt source partition differs from date LODO contract")
    if set(records) != set(source) or tuple(records) != source:
        raise CCEContractError("CCE reconstructed plan was not supplied exactly the source recordings")
    hashes = tuple(records[name].input_sha256 for name in source)
    if hashes != tuple(raw_plan.get("source_input_sha256", ())):
        raise CCEContractError("CCE source NWB hashes differ from immutable raw receipt")
    if tuple(eb_plan.get("source_input_sha256", hashes)) != hashes:
        raise CCEContractError("CCE source NWB hashes differ from immutable EB receipt")
    q, ridge_lambda = int(raw_plan["q"]), float(raw_plan["lambda"])
    if q <= 0 or q > 16 or ridge_lambda <= 0 or int(eb_plan["q"]) != q or float(eb_plan["lambda"]) != ridge_lambda:
        raise CCEContractError("CCE raw/EB estimator settings drift")
    source_rates = np.concatenate([_support_rates(records[name], records[name].trial_values[:4]) for name in source])
    mean = source_rates.mean(axis=0)
    scale = np.maximum(source_rates.std(axis=0), 1.0e-6)
    _, _, pcs_all = np.linalg.svd((source_rates - mean[None, :]) / scale[None, :], full_matrices=False)
    pcs = np.asarray(pcs_all[:16], dtype=np.float64)
    provisional = CCEFrozenPlan(
        outer_date, source, hashes, mean, scale, pcs, q, ridge_lambda,
        np.empty((7, 4)), np.zeros(4), 1.0, str(raw_plan["plan_sha256"]), raw_sha, eb_sha, "",
    )
    pooled_rows = np.concatenate(
        [_fit_without_eb(records[name], provisional, records[name].trial_values[:4])["raw_rows"] for name in source], axis=0
    )
    _, _, right_vectors = np.linalg.svd(pooled_rows, full_matrices=False)
    U = np.asarray(right_vectors[:4].T, dtype=np.float64)
    source_carriers = pooled_rows @ U
    mu = source_carriers.mean(axis=0)
    tau2 = float(np.square(source_carriers - mu[None, :]).sum() / (source_carriers.shape[0] * 4))
    if not np.allclose(mu, np.asarray(eb_plan["prior_mu"], dtype=np.float64), rtol=1e-10, atol=1e-18):
        raise CCEContractError("CCE reconstructed EB prior mean differs from receipt")
    if not np.isclose(tau2, float(eb_plan["prior_tau2"]), rtol=1e-10, atol=1e-20):
        raise CCEContractError("CCE reconstructed EB prior variance differs from receipt")
    transform_body = {
        "outer_date": outer_date, "source_sessions": list(source), "source_input_sha256": list(hashes),
        "q": q, "lambda": ridge_lambda, "raw_plan_sha256": str(raw_plan["plan_sha256"]),
        "raw_receipt_sha256": raw_sha, "eb_receipt_sha256": eb_sha,
        "mean_sha256": array_sha256(mean), "scale_sha256": array_sha256(scale), "pcs_sha256": array_sha256(pcs),
        "U_sha256": array_sha256(U), "mu_sha256": array_sha256(mu), "tau2": tau2,
    }
    return CCEFrozenPlan(
        outer_date, source, hashes, np.asarray(mean, np.float64), np.asarray(scale, np.float64), pcs, q, ridge_lambda,
        U, np.asarray(mu, np.float64), tau2, str(raw_plan["plan_sha256"]), raw_sha, eb_sha, canonical_sha256(transform_body),
    )


def persist_frozen_plan(plan: CCEFrozenPlan, cache_dir: str | Path) -> dict[str, Any]:
    directory = Path(cache_dir).resolve()
    arrays_path = directory / f"{plan.outer_date}_frozen_eb_plan.npz"
    manifest_path = directory / f"{plan.outer_date}_frozen_eb_plan.manifest.json"
    manifest = plan.manifest()
    arrays_sha = _write_npz_once(
        arrays_path, mean=plan.mean, scale=plan.scale, pcs=plan.pcs, q=np.asarray(plan.q, np.int64),
        **{"lambda": np.asarray(plan.ridge_lambda, np.float64)}, U=plan.U, mu=plan.mu, tau2=np.asarray(plan.tau2, np.float64),
    )
    manifest_sha = _write_json_once(manifest_path, manifest)
    return {**manifest, "arrays_file_sha256": arrays_sha, "manifest_file_sha256": manifest_sha}


@dataclass(frozen=True)
class CCECarrierCacheEntry:
    session_name: str
    start_index: int
    trial_values: tuple[float, float, float, float]
    carrier: np.ndarray
    carrier_sha256: str


class CCECarrierCache:
    def __init__(self, entries: Iterable[CCECarrierCacheEntry], manifest: Mapping[str, Any], source_sessions: Sequence[str]):
        self.entries = tuple(entries)
        self.manifest = dict(manifest)
        self.source_sessions = tuple(source_sessions)
        self._by_key = {(entry.session_name, entry.start_index): entry for entry in self.entries}
        self.starts_by_session = {
            name: tuple(entry.start_index for entry in self.entries if entry.session_name == name) for name in self.source_sessions
        }
        if len(self._by_key) != len(self.entries) or not self.entries or any(not starts for starts in self.starts_by_session.values()):
            raise CCEContractError("CCE cache has duplicate or missing legal source M=4 entries")

    def get(self, session_name: str, start_index: int) -> CCECarrierCacheEntry:
        try:
            return self._by_key[(session_name, int(start_index))]
        except KeyError as exc:
            raise CCEContractError(f"CCE uncached M=4 support block {session_name}:{start_index}") from exc


def legal_contiguous_starts(record: H1PilotRecord) -> tuple[int, ...]:
    starts: list[int] = []
    for start in range(len(record.trial_values) - SUPPORT_TRIALS + 1):
        values = record.trial_values[start : start + SUPPORT_TRIALS]
        if all(record.blocks_for(value).rates.shape[0] >= 2 for value in values):
            for value in values:
                record.eval_trial_neural(value)
            starts.append(start)
    if not starts:
        raise CCEContractError(f"{record.session_name}: no legal contiguous source M=4 supports")
    return tuple(starts)


def build_carrier_cache(records: Mapping[str, H1PilotRecord], plan: CCEFrozenPlan, cache_dir: str | Path) -> CCECarrierCache:
    directory = Path(cache_dir).resolve()
    rows: list[dict[str, Any]] = []
    carriers: list[np.ndarray] = []
    for name in plan.source_sessions:
        record = records[name]
        for start in legal_contiguous_starts(record):
            values = tuple(float(value) for value in record.trial_values[start : start + SUPPORT_TRIALS])
            carrier = fit_frozen_carrier(record, plan, values)["carrier"]
            rows.append({"session": name, "start_index": start, "trial_values": list(values), "carrier_sha256": carrier_sha256(carrier)})
            carriers.append(np.asarray(carrier, dtype=np.float64))
    stacked = np.stack(carriers, axis=0)
    if stacked.ndim != 3 or stacked.shape[1:] != (EXPECTED_NEURONS, 4):
        raise CCEContractError(f"CCE source cache shape drift: {stacked.shape}")
    body = {
        "schema": CCE_CACHE_SCHEMA, "outer_date": plan.outer_date, "source_sessions": list(plan.source_sessions),
        "transform_sha256": plan.transform_sha256, "carrier_dtype": str(stacked.dtype), "carrier_shape": list(stacked.shape), "entries": rows,
    }
    body["cache_sha256"] = canonical_sha256(body)
    arrays_path = directory / f"{plan.outer_date}_all_source_m4_carriers.npz"
    manifest_path = directory / f"{plan.outer_date}_all_source_m4_carriers.manifest.json"
    _write_npz_once(arrays_path, carriers=stacked)
    _write_json_once(manifest_path, body)
    entries = tuple(
        CCECarrierCacheEntry(row["session"], int(row["start_index"]), tuple(float(v) for v in row["trial_values"]), stacked[i], row["carrier_sha256"])
        for i, row in enumerate(rows)
    )
    return CCECarrierCache(entries, body, plan.source_sessions)


def persist_normalized_cache(
    cache: CCECarrierCache, normalizer: SourceScalarNormalizer, cache_dir: str | Path, outer_date: str,
) -> dict[str, Any]:
    raw = np.stack([entry.carrier for entry in cache.entries], axis=0).astype(np.float64)
    normalized = normalizer.normalize(raw)
    directory = Path(cache_dir).resolve()
    arrays_path = directory / f"{outer_date}_normalized_source_carriers.npz"
    manifest_path = directory / f"{outer_date}_normalized_source_carriers.manifest.json"
    arrays_sha = _write_npz_once(arrays_path, carriers=normalized)
    body = {
        "schema": CCE_NORMALIZED_CACHE_SCHEMA, "outer_date": outer_date, "formula": NORMALIZER_FORMULA,
        "normalizer_floor": NORMALIZER_FLOOR, "normalizer_sha256": normalizer.normalizer_sha256,
        "source_cache_sha256": cache.manifest["cache_sha256"], "raw_shape": list(raw.shape), "normalized_shape": list(normalized.shape),
        "raw_carriers_sha256": array_sha256(raw), "normalized_carriers_sha256": array_sha256(normalized), "arrays_file_sha256": arrays_sha,
    }
    body["manifest_sha256"] = canonical_sha256(body)
    return {**body, "manifest_file_sha256": _write_json_once(manifest_path, body)}


def _window_hash(indices: Sequence[tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for session, start in indices:
        digest.update(session.encode("ascii"))
        digest.update(int(start).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


class H1M4CCESourceDataset(Dataset):
    def __init__(self, records: Mapping[str, H1PilotRecord], cache: CCECarrierCache, normalizer: SourceScalarNormalizer):
        self.records = {name: records[name] for name in cache.source_sessions}
        self.cache, self.normalizer = cache, normalizer
        self.window_indices: list[tuple[str, int]] = []
        self.neural_data: dict[str, np.ndarray] = {}
        self.target_data: dict[str, np.ndarray] = {}
        self.eval_mask: dict[str, np.ndarray] = {}
        self._trial_identity: dict[tuple[str, float], np.ndarray] = {}
        prehistory = WINDOW_SIZE - 1
        for name in cache.source_sessions:
            record = self.records[name]
            self.neural_data[name] = np.pad(record.neural, ((prehistory, 0), (0, 0)), constant_values=0.0)
            self.target_data[name] = np.pad(record.velocity, ((prehistory, 0), (0, 0)), constant_values=0.0)
            self.eval_mask[name] = np.pad(record.eval_mask, (prehistory, 0), constant_values=False)
            for support_start in cache.starts_by_session[name]:
                for value in cache.get(name, support_start).trial_values:
                    self._trial_identity.setdefault((name, float(value)), interpolate_trial_identity(record, value))
            self.window_indices.extend(
                (name, start) for start in range(self.neural_data[name].shape[0] - WINDOW_SIZE + 1) if self.eval_mask[name][start + WINDOW_SIZE - 1]
            )
        if not self.window_indices:
            raise CCEContractError("CCE source dataset has no eval-valid windows")
        self.window_indices_sha256 = _window_hash(self.window_indices)

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, request: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
        if not isinstance(request, tuple) or len(request) != 2:
            raise CCEContractError("CCE source samples require predeclared (window, M4-start) requests")
        index, calibration_start = (int(request[0]), int(request[1]))
        session, start = self.window_indices[index]
        entry = self.cache.get(session, calibration_start)
        identity = np.stack([self._trial_identity[(session, value)] for value in entry.trial_values], axis=0)
        expected = tuple(self.records[session].trial_values[calibration_start : calibration_start + SUPPORT_TRIALS])
        if entry.trial_values != expected:
            raise CCEContractError("CCE source identity/carrier support alignment drift")
        return (
            self.neural_data[session][start : start + WINDOW_SIZE], self.target_data[session][start : start + WINDOW_SIZE],
            identity, session, self.normalizer.normalize(entry.carrier).astype(np.float32),
        )


class H1M4CCEPairedBatchSampler(Sampler[list[tuple[int, int]]]):
    """A source schedule fixed by outer date, seed, and legal source supports."""

    def __init__(self, dataset: H1M4CCESourceDataset, outer_date: str, batch_size: int = 32, seed: int = FIXED_SEED, max_epochs: int = FIXED_EPOCHS):
        outer_date = assert_confirmatory_date(outer_date)
        if batch_size != 32 or seed != FIXED_SEED or max_epochs != FIXED_EPOCHS:
            raise CCEContractError("CCE source schedule fixes batch=32, seed=42, epochs=50")
        self.dataset, self.outer_date, self.batch_size, self.seed, self.max_epochs = dataset, outer_date, batch_size, seed, max_epochs
        grouped: dict[str, list[int]] = {name: [] for name in dataset.cache.source_sessions}
        for index, (name, _start) in enumerate(dataset.window_indices):
            grouped[name].append(index)
        batches: list[list[int]] = []
        for name in dataset.cache.source_sessions:
            token = int.from_bytes(hashlib.sha256(f"{seed}|{outer_date}|batch|{name}".encode()).digest()[:8], "big")
            permutation = random.Random(token).sample(grouped[name], len(grouped[name]))
            batches.extend(permutation[offset : offset + batch_size] for offset in range(0, len(permutation), batch_size) if len(permutation[offset : offset + batch_size]) == batch_size)
        shuffle_token = int.from_bytes(hashlib.sha256(f"{seed}|{outer_date}|batches".encode()).digest()[:8], "big")
        self.batches = random.Random(shuffle_token).sample(batches, len(batches))
        self.flat_indices = np.asarray([index for batch in self.batches for index in batch], dtype=np.int64)
        self.batch_order_sha256 = array_sha256(self.flat_indices)
        schedule = np.empty((max_epochs, len(self.flat_indices)), dtype=np.int16)
        session_vector = np.asarray([dataset.window_indices[int(index)][0] for index in self.flat_indices], dtype=object)
        for name in dataset.cache.source_sessions:
            positions = np.flatnonzero(session_vector == name)
            legal = np.asarray(dataset.cache.starts_by_session[name], dtype=np.int16)
            token = hashlib.sha256(f"{seed}|{outer_date}|m4-schedule|{name}".encode()).digest()
            draws = np.random.default_rng(int.from_bytes(token[:8], "big")).integers(0, len(legal), size=(max_epochs, len(positions)))
            schedule[:, positions] = legal[draws]
        self.schedule, self.schedule_sha256, self._epoch = schedule, array_sha256(schedule), 0

    def reset_epoch(self) -> None:
        self._epoch = 0

    def __iter__(self):
        if self._epoch >= self.max_epochs:
            raise RuntimeError("CCE fixed 50-epoch schedule exhausted")
        schedule = self.schedule[self._epoch]
        offset = 0
        for batch in self.batches:
            starts = schedule[offset : offset + len(batch)]
            offset += len(batch)
            yield [(int(index), int(start)) for index, start in zip(batch, starts)]
        self._epoch += 1

    def __len__(self) -> int:
        return len(self.batches)


def persist_schedule(sampler: H1M4CCEPairedBatchSampler, cache_dir: str | Path) -> dict[str, Any]:
    directory = Path(cache_dir).resolve()
    array_path = directory / f"{sampler.outer_date}_source_calibration_schedule.npy"
    schedule_sha = _write_npy_once(array_path, sampler.schedule)
    body = {
        "schema": CCE_SCHEDULE_SCHEMA, "outer_date": sampler.outer_date, "seed": sampler.seed, "epochs": sampler.max_epochs,
        "batch_size": sampler.batch_size, "batches_per_epoch": len(sampler.batches), "scheduled_samples_per_epoch": int(sampler.flat_indices.size),
        "source_window_indices_sha256": sampler.dataset.window_indices_sha256, "batch_order_sha256": sampler.batch_order_sha256,
        "calibration_schedule_sha256": sampler.schedule_sha256, "carrier_cache_sha256": sampler.dataset.cache.manifest["cache_sha256"],
        "selection": "date-seeded dedicated RNG selecting one legal contiguous M4 support per scheduled sample",
    }
    return {**body, "schedule_file_sha256": schedule_sha, "manifest_file_sha256": _write_json_once(directory / f"{sampler.outer_date}_source_schedule.manifest.json", body)}


def _complete_row_shuffle(carrier: np.ndarray, session_name: str, outer_date: str) -> np.ndarray:
    values = np.asarray(carrier, dtype=np.float64)
    token = hashlib.sha256(f"{ROTATION_SEED}|cce-row|{session_name}|{outer_date}".encode()).digest()
    perm = np.random.default_rng(int.from_bytes(token[:8], "big")).permutation(len(values))
    if np.array_equal(perm, np.arange(len(values))):
        perm = np.roll(perm, 1)
    result = values[perm]
    if np.array_equal(result, values):
        raise CCEContractError("CCE row intervention was identity")
    return result


def _label_rotated_carrier(record: H1PilotRecord, plan: CCEFrozenPlan, values: Sequence[float]) -> np.ndarray:
    override: dict[float, np.ndarray] = {}
    for value in values:
        trial = record.blocks_for(value)
        if trial.velocity.shape[0] < 2:
            raise CCEContractError("CCE label rotation needs at least two support blocks")
        token = hashlib.sha256(f"{ROTATION_SEED}|cce-label|{record.session_name}|{trial.trial_number}".encode()).digest()
        shift = 1 + int.from_bytes(token[:8], "big") % (trial.velocity.shape[0] - 1)
        override[float(value)] = np.roll(trial.velocity, shift, axis=0)
    return fit_frozen_carrier(record, plan, values, labels_override=override)["carrier"]


@dataclass(frozen=True)
class CCETargetSupport:
    session_name: str
    trial_values: tuple[float, float, float, float]
    fifth_trial: float
    query_first_bin: int
    identity: np.ndarray
    carriers: Mapping[str, np.ndarray]
    support_sha256: str
    carrier_sha256: Mapping[str, str]


class H1M4CCEStrictTargetDataset(Dataset):
    INTERVENTIONS = ("full", "zero", "row", "label")

    def __init__(self, records: Mapping[str, H1PilotRecord], plan: CCEFrozenPlan, normalizer: SourceScalarNormalizer, intervention: str = "full"):
        if intervention not in self.INTERVENTIONS:
            raise CCEContractError(f"unknown CCE intervention {intervention}")
        self.records = {name: records[name] for name in target_sessions_for_date(plan.outer_date)}
        self.plan, self.normalizer, self.intervention = plan, normalizer, intervention
        self.support: dict[str, CCETargetSupport] = {}
        self.window_indices: list[tuple[str, int]] = []
        for name, record in self.records.items():
            values = tuple(float(value) for value in record.trial_values[:SUPPORT_TRIALS])
            fifth = float(record.trial_values[SUPPORT_TRIALS])
            bins = np.flatnonzero(record.eval_mask & np.isfinite(record.trial_num) & (record.trial_num == fifth))
            if bins.size == 0:
                raise CCEContractError(f"{name}: no first query bin for trial5")
            boundary = int(bins[0])
            raw_full = fit_frozen_carrier(record, plan, values)["carrier"]
            raw = {"full": raw_full, "zero": np.zeros_like(raw_full), "row": _complete_row_shuffle(raw_full, name, plan.outer_date), "label": _label_rotated_carrier(record, plan, values)}
            carriers = {key: (np.zeros_like(value) if key == "zero" else normalizer.normalize(value)) for key, value in raw.items()}
            if np.array_equal(carriers["full"], carriers["zero"]) or np.array_equal(carriers["full"], carriers["row"]) or np.array_equal(carriers["full"], carriers["label"]):
                raise CCEContractError("CCE target intervention is numerically identity")
            identity = np.stack([interpolate_trial_identity(record, value) for value in values], axis=0)
            digest = hashlib.sha256()
            digest.update(np.asarray(values, np.float64).tobytes()); digest.update(identity.tobytes())
            for value in values:
                trial = record.blocks_for(value)
                digest.update(trial.rates.tobytes()); digest.update(trial.velocity.tobytes()); digest.update(trial.block_indices.tobytes())
            hashes = {key: carrier_sha256(value) for key, value in carriers.items()}
            self.support[name] = CCETargetSupport(name, values, fifth, boundary, identity, carriers, digest.hexdigest(), hashes)
            for start in range(boundary, record.neural.shape[0] - WINDOW_SIZE + 1):
                output = start + WINDOW_SIZE - 1
                if record.eval_mask[output]:
                    self.window_indices.append((name, start))
        if not self.window_indices:
            raise CCEContractError("CCE target has no trial5+ 700-bin query windows")
        self.window_indices_sha256 = _window_hash(self.window_indices)

    def with_intervention(self, intervention: str) -> "H1M4CCEStrictTargetDataset":
        if intervention not in self.INTERVENTIONS:
            raise CCEContractError(intervention)
        clone = object.__new__(type(self))
        clone.records, clone.plan, clone.normalizer, clone.intervention = self.records, self.plan, self.normalizer, intervention
        clone.support, clone.window_indices, clone.window_indices_sha256 = self.support, self.window_indices, self.window_indices_sha256
        return clone

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
        session, start = self.window_indices[int(index)]
        record, support = self.records[session], self.support[session]
        if start < support.query_first_bin or not record.eval_mask[start + WINDOW_SIZE - 1]:
            raise CCEContractError("CCE query history crosses M4 support/trial5 boundary")
        return (record.neural[start : start + WINDOW_SIZE], record.velocity[start : start + WINDOW_SIZE], support.identity, session, np.asarray(support.carriers[self.intervention], np.float32))

    def support_and_carrier_hashes(self) -> dict[str, Any]:
        return {name: {"trial_values": list(item.trial_values), "fifth_trial": item.fifth_trial, "query_first_bin": item.query_first_bin, "support_sha256": item.support_sha256, "carrier_sha256": dict(item.carrier_sha256)} for name, item in self.support.items()}


def validate_target_receipt_binding(records: Mapping[str, H1PilotRecord], plan: CCEFrozenPlan, raw_receipt_path: str | Path, eb_receipt_path: str | Path) -> None:
    """Bind target's first-four carrier computation only after checkpoint checks."""

    raw_path, eb_path = Path(raw_receipt_path).resolve(), Path(eb_receipt_path).resolve()
    if sha256_file(raw_path) != plan.raw_receipt_sha256 or sha256_file(eb_path) != plan.eb_receipt_sha256:
        raise CCEContractError("CCE immutable raw/EB receipts changed before target binding")
    raw_row = _receipt_date_row(json.loads(raw_path.read_text(encoding="utf-8")), plan.outer_date)
    eb_row = _receipt_date_row(json.loads(eb_path.read_text(encoding="utf-8")), plan.outer_date)
    raw_by_name = {row["session_name"]: row for row in raw_row["records"]}
    eb_by_name = {row["session_name"]: row for row in eb_row["records"]}
    for name in target_sessions_for_date(plan.outer_date):
        values = records[name].trial_values[:SUPPORT_TRIALS]
        fit = fit_frozen_carrier(records[name], plan, values)
        if list(values) != raw_by_name[name]["support_trial_numbers"]:
            raise CCEContractError(f"{name}: CCE target M4 support differs from raw receipt")
        if carrier_sha256(fit["raw_carrier"]) != raw_by_name[name]["carrier_sha256"]:
            raise CCEContractError(f"{name}: CCE raw target carrier differs from receipt")
        if carrier_sha256(fit["carrier"]) != eb_by_name[name]["carrier_sha256"]:
            raise CCEContractError(f"{name}: CCE EB target carrier differs from receipt")


class H1M4CCEDataModule(pl.LightningDataModule):
    """Fold-generic source-only datamodule used by each paired 50-epoch run."""

    def __init__(self, *, task: str, data_dir: str, raw_receipt_path: str, eb_receipt_path: str, cache_dir: str, outer_date: str, batch_size: int = 32, window_size: int = 700, calibration_n_trials: int = 4, max_trial_length: int = 1024, random_calibration: bool = True, smooth_calibration: bool = False, interpolate_trials: bool = True, interpolate_trials_kind: str = "cubic", num_workers: int = 0, pin_memory: bool = False, seed: int = 42, fixed_epochs: int = 50, normalizer_floor: float = NORMALIZER_FLOOR) -> None:
        super().__init__()
        fixed = {"task": str(task).lower() == "h1", "outer_date": str(outer_date) in CONFIRMATORY_DATES, "batch": int(batch_size) == 32, "window": int(window_size) == WINDOW_SIZE, "m": int(calibration_n_trials) == SUPPORT_TRIALS, "max": int(max_trial_length) == MAX_TRIAL_LENGTH, "random": bool(random_calibration), "smooth": not bool(smooth_calibration), "interpolate": bool(interpolate_trials), "kind": str(interpolate_trials_kind) == "cubic", "workers": int(num_workers) == 0, "seed": int(seed) == FIXED_SEED, "epochs": int(fixed_epochs) == FIXED_EPOCHS, "floor": float(normalizer_floor) == NORMALIZER_FLOOR}
        if not all(fixed.values()):
            raise CCEContractError(f"CCE fixed data contract violated: {fixed}")
        self.save_hyperparameters(logger=False); self.batch_size_per_device = 32; self._setup_done = False

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit"}:
            raise RuntimeError("CCE permits source fit only; target evaluation is isolated")
        if self._setup_done:
            return
        if self.trainer is not None and self.trainer.world_size != 1:
            raise CCEContractError("CCE fixes one device and one source schedule per outer date")
        date = assert_confirmatory_date(self.hparams.outer_date)
        records = load_source_records_for_date(self.hparams.data_dir, date)
        plan = reconstruct_plan_for_date(records, date, self.hparams.raw_receipt_path, self.hparams.eb_receipt_path)
        root = Path(self.hparams.cache_dir).resolve()
        plan_manifest = persist_frozen_plan(plan, root)
        carrier_cache = build_carrier_cache(records, plan, root)
        raw = np.stack([entry.carrier for entry in carrier_cache.entries], axis=0)
        normalizer = fit_source_scalar_normalizer(raw, carrier_cache.manifest["cache_sha256"])
        normalized_manifest = persist_normalized_cache(carrier_cache, normalizer, root, date)
        dataset = H1M4CCESourceDataset(records, carrier_cache, normalizer)
        sampler = H1M4CCEPairedBatchSampler(dataset, date)
        schedule_manifest = persist_schedule(sampler, root)
        files = [{"role": "source_heldin_calib", "session": name, "date": records[name].date, "sha256": records[name].input_sha256, "size_bytes": records[name].path.stat().st_size} for name in plan.source_sessions]
        manifest = {"schema": CCE_SOURCE_MANIFEST_SCHEMA, "outer_date": date, "source_sessions": list(plan.source_sessions), "target_sessions_not_opened": list(target_sessions_for_date(date)), "files": files, "raw_receipt_sha256": plan.raw_receipt_sha256, "eb_receipt_sha256": plan.eb_receipt_sha256, "transform_sha256": plan.transform_sha256, "transform_array_sha256": plan_manifest["array_sha256"], "carrier_cache_sha256": carrier_cache.manifest["cache_sha256"], "normalized_cache_sha256": normalized_manifest["normalized_carriers_sha256"], "normalizer_sha256": normalizer.normalizer_sha256, "source_window_indices_sha256": dataset.window_indices_sha256, "batch_order_sha256": sampler.batch_order_sha256, "calibration_schedule_sha256": sampler.schedule_sha256, "source_schedule_manifest_sha256": schedule_manifest["manifest_file_sha256"], "batches_per_epoch": len(sampler), "scheduled_samples_per_epoch": int(sampler.flat_indices.size), "epochs": FIXED_EPOCHS, "calibration_n_trials": SUPPORT_TRIALS, "window_size": WINDOW_SIZE, "target_nwb_opened_during_training_setup": False, "minival_or_heldout_enumerated": False, "source_only_normalizer_scope": True, "normalization_intervention_policy": "Full/row/label raw then divide same source scalar; Zero literal post-normalization"}
        self.records, self.plan, self.plan_manifest, self.carrier_cache, self.normalizer = records, plan, plan_manifest, carrier_cache, normalizer
        self.normalized_cache_manifest, self.train_dataset, self.train_batch_sampler = normalized_manifest, dataset, sampler
        self._manifest, self._manifest_sha256, self._setup_done = manifest, canonical_sha256(manifest), True

    def train_dataloader(self):
        if not self._setup_done: raise RuntimeError("call setup('fit') before CCE train_dataloader")
        return DataLoader(self.train_dataset, batch_sampler=self.train_batch_sampler, num_workers=0, pin_memory=bool(self.hparams.pin_memory))

    def val_dataloader(self): return []
    def test_dataloader(self): raise RuntimeError("CCE formal test loader is forbidden")
    def predict_dataloader(self): raise RuntimeError("CCE target prediction is evaluator-only")
    def pilot_manifest(self) -> dict[str, Any]:
        if not self._setup_done: raise RuntimeError("CCE setup has not completed")
        return dict(self._manifest)
    @property
    def pilot_manifest_sha256(self) -> str:
        if not self._setup_done: raise RuntimeError("CCE setup has not completed")
        return self._manifest_sha256
