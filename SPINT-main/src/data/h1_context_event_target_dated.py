"""Date-parameterised Context event target dataset and LODO source helpers.

Generalises the sealed fold-0 ``H1ContextStrictTargetDataset`` without editing
sealed modules.  Source-side dating mirrors ``build_context_source_assets`` so
Stage A preflight can audit leave-one-date-out pools before any GPU launch.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from torch.utils.data import Dataset, Sampler

from sua_exploration.mc_maze import h1_event_carrier_design_screen as design
from sua_exploration.mc_maze import h1_sparse_event_endpoint as event_v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as event_v2
from src.data.h1_context_event_carrier import (
    CARRIER_DIM,
    CONTEXT_PROTOCOL,
    ContextCarrierEntry,
    ContextScalarNormalizer,
    ContextTargetSupport,
    M4_BUDGET,
    NORMALIZER_FLOOR,
    RIDGE_LAMBDA,
    SCHEMA,
    _fit,
)
from src.data.h1_m4_eb_pilot import (
    H1_M4_FOLD0_TARGET,
    MAX_TRIAL_LENGTH,
    PilotDataError,
    WINDOW,
    _window_manifest_hash,
    index_heldin_calib,
    interpolate_identity,
    interpolate_trial_identity,
    legal_contiguous_starts,
    load_record,
)

FIXED_SCREEN_REL = "sua_exploration/results/h1_event_carrier_design_screen_v1/source_screen.json"
FIXED_SCREEN_SHA = "74bbc01490432794546e7ca2fd4242fbed6f2a7ebdd65786f56035eaa49bfeb3"
FIXED_FOLD0_MAP_SHA = "50c0c55969e6898e00846302f97a6637ac78b0af6715f33de428a9f3d845e525"
FIXED_FOLD0_ARRAYS = {
    "active_mask": "1b109aa95e5bee6519b035de92dbdddad5b8660ab014f429105a934b4b101223",
    "feature_mean": "cd50c588dc3992f1c07ba65a56553261641d7dcb794d4a6b331e7c8fe2f1b521",
    "feature_scale": "def0b8f639903128e42d775fb635f440d1259aaf9e0be2911030b25f332de6dc",
    "projection": "5f9a3f485ee4be33d915f4db587c67b3e8b88eabca76be34ef41a0fbbc6625df",
    "latent_scale": "164472716c91002aa80cca18ecb79f44e53f5b384757d720596d524695ef8618",
}
ROOT = Path(__file__).resolve().parents[3]
FIXED_SCREEN = ROOT / FIXED_SCREEN_REL


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise PilotDataError(message)


def _screen_path() -> Path:
    return FIXED_SCREEN


def verify_sealed_screen_receipt() -> None:
    _need(event_v1.sha256_file(_screen_path()) == FIXED_SCREEN_SHA, "fixed CPU screen receipt SHA drift")


def sealed_map_manifest(outer_date: str) -> dict[str, Any]:
    verify_sealed_screen_receipt()
    _need(outer_date in event_v1.H1_DATES, f"outer date {outer_date!r} not in H1_DATES")
    return json.loads(_screen_path().read_text())["basis_by_candidate_and_outer_date"]["ser_context_q4"][outer_date]


def lodo_source_sessions(outer_date: str) -> tuple[str, ...]:
    _need(outer_date in event_v1.H1_DATES, f"outer date {outer_date!r} not in H1_DATES")
    return tuple(name for name in event_v1.H1_HELDIN_SESSIONS if event_v1.session_date(name) != outer_date)


def lodo_target_sessions(outer_date: str) -> tuple[str, ...]:
    _need(outer_date in event_v1.H1_DATES, f"outer date {outer_date!r} not in H1_DATES")
    return tuple(name for name in event_v1.H1_HELDIN_SESSIONS if event_v1.session_date(name) == outer_date)


def pure_cpu_fit_and_bind_map(data_dir: Path, outer_date: str) -> design.LatentMap:
    """Run before importing Lightning/Torch or any ``src.data`` training module."""
    verify_sealed_screen_receipt()
    _need(outer_date in event_v1.H1_DATES, f"outer date {outer_date!r} not in H1_DATES")
    indexed = event_v1.index_heldin_calib(data_dir)
    source_names = lodo_source_sessions(outer_date)
    sessions = {
        name: design.load_context_session(event_v1.load_event_session(indexed[name]))
        for name in source_names
    }
    candidate = next(item for item in design.CANDIDATES if item.name == "ser_context_q4")
    mapping = design.fit_latent_map(sessions, outer_date=outer_date, candidate=candidate)
    fixed = sealed_map_manifest(outer_date)
    _need(
        mapping.map_sha256 == fixed["map_sha256"] and mapping.manifest() == fixed,
        f"pure CPU pre-Torch SVD differs from sealed receipt for outer_date={outer_date}",
    )
    if outer_date == "19250101":
        _need(
            mapping.map_sha256 == FIXED_FOLD0_MAP_SHA
            and mapping.manifest()["array_sha256"] == FIXED_FOLD0_ARRAYS,
            "fold-0 map does not match sealed FIXED_MAP_SHA / FIXED_ARRAYS",
        )
    return mapping


def load_dated_source_records(data_dir: str | Path, outer_date: str) -> dict[str, Any]:
    paths = index_heldin_calib(data_dir)
    source_names = lodo_source_sessions(outer_date)
    records = {name: load_record(paths[name]) for name in source_names}
    if any(event_v1.session_date(name) == outer_date for name in records):
        raise PilotDataError("target date leaked into dated source record loader")
    return records


def load_dated_target_records(data_dir: str | Path, outer_date: str) -> dict[str, Any]:
    paths = index_heldin_calib(data_dir)
    target_names = lodo_target_sessions(outer_date)
    records = {name: load_record(paths[name]) for name in target_names}
    if any(event_v1.session_date(name) != outer_date for name in records):
        raise PilotDataError("non-target date leaked into dated target loader")
    return records


class DatedContextCarrierCache:
    def __init__(
        self,
        entries: Iterable[ContextCarrierEntry],
        latent_map: design.LatentMap,
        source_sessions: Sequence[str],
    ) -> None:
        self.entries = tuple(entries)
        self._by_key = {(item.session_name, item.start_index): item for item in self.entries}
        self.source_sessions = tuple(source_sessions)
        self.starts_by_session = {
            name: tuple(item.start_index for item in self.entries if item.session_name == name)
            for name in self.source_sessions
        }
        _need(
            len(self.entries) == len(self._by_key) and all(self.starts_by_session.values()),
            "dated context cache must contain all legal source ranges",
        )
        body = {
            "schema": "h1_context_event_carrier_m4_dated_cache_v1",
            "context_map_sha256": latent_map.map_sha256,
            "candidate": "ser_context_q4",
            "carrier_dim": CARRIER_DIM,
            "ridge_lambda": RIDGE_LAMBDA,
            "outer_date": latent_map.outer_date,
            "source_sessions": list(self.source_sessions),
            "entries": [
                {
                    "session": x.session_name,
                    "start_index": x.start_index,
                    "trial_values": list(x.trial_values),
                    "carrier_sha256": x.carrier_sha256,
                }
                for x in self.entries
            ],
        }
        body["cache_sha256"] = event_v1.canonical_sha256(body)
        self.manifest = body

    def get(self, session_name: str, start_index: int) -> ContextCarrierEntry:
        try:
            return self._by_key[(str(session_name), int(start_index))]
        except KeyError as exc:
            raise PilotDataError(f"uncached dated context support {session_name}:{start_index}") from exc


def _dated_normalizer(cache: DatedContextCarrierCache) -> ContextScalarNormalizer:
    values = np.stack([x.carrier for x in cache.entries])
    scalar = float(np.sqrt(np.mean(np.square(values, dtype=np.float64), dtype=np.float64)))
    _need(np.isfinite(scalar) and scalar > 0, "dated context source normalizer undefined")
    body = {
        "formula": "s_src=sqrt(mean(source_context_cache^2)); carrier_norm=carrier/max(s_src,1e-12)",
        "s_src": scalar,
        "source_cache_sha256": cache.manifest["cache_sha256"],
        "shape": list(values.shape),
    }
    return ContextScalarNormalizer(scalar, cache.manifest["cache_sha256"], event_v1.canonical_sha256(body))


def build_context_source_assets_dated(
    data_dir: str | Path,
    outer_date: str,
    *,
    frozen_map: design.LatentMap | None = None,
) -> tuple[Any, dict[str, design.ContextSession], design.LatentMap, DatedContextCarrierCache, ContextScalarNormalizer]:
    source_names = lodo_source_sessions(outer_date)
    records = load_dated_source_records(data_dir, outer_date)
    indexed = index_heldin_calib(data_dir)
    sessions = {
        name: design.load_context_session(event_v1.load_event_session(indexed[name]))
        for name in source_names
    }
    latent_map = frozen_map or design.fit_latent_map(
        sessions,
        outer_date=outer_date,
        candidate=next(item for item in design.CANDIDATES if item.name == "ser_context_q4"),
    )
    _need(
        latent_map.outer_date == outer_date
        and latent_map.candidate.name == "ser_context_q4"
        and latent_map.candidate.rank == 4,
        "dated context source map contract drift",
    )
    entries: list[ContextCarrierEntry] = []
    for name in source_names:
        record = records[name]
        session = sessions[name]
        _need(tuple(record.trial_values) == tuple(session.base.trial_values), f"{name}: trial alignment drift")
        for start in legal_contiguous_starts(record):
            support = design.select_range(session, start=start, budget=M4_BUDGET)
            carrier = _fit(support, latent_map)
            entries.append(
                ContextCarrierEntry(
                    name,
                    start,
                    tuple(float(x) for x in record.trial_values[start : start + 4]),
                    carrier,
                    event_v1.array_sha256(carrier),
                )
            )
    cache = DatedContextCarrierCache(entries, latent_map, source_names)
    return records, sessions, latent_map, cache, _dated_normalizer(cache)


def build_context_manifest_dated(
    *,
    outer_date: str,
    records: Mapping[str, Any],
    latent_map: design.LatentMap,
    cache: DatedContextCarrierCache,
    normalizer: ContextScalarNormalizer,
    dataset: "H1ContextDatedSourceDataset",
    sampler: "H1ContextDatedBatchSampler",
    target_sessions: Sequence[str],
) -> dict[str, Any]:
    source_names = lodo_source_sessions(outer_date)
    return {
        "schema": SCHEMA,
        "protocol": CONTEXT_PROTOCOL,
        "fold_date": outer_date,
        "source_sessions": list(source_names),
        "target_sessions_not_opened": list(target_sessions),
        "files": [{"session": n, "nwb_sha256": records[n].input_sha256} for n in source_names],
        "candidate": "ser_context_q4",
        "raw_label": "endpoint_delta(7)+endpoint_midpoint(7)+native_tag_one_hot",
        "source_map": latent_map.manifest(),
        "basis": {"basis_sha256": latent_map.map_sha256, "kind": "source_encoding_context_map"},
        "carrier_cache_sha256": cache.manifest["cache_sha256"],
        "normalized_cache_sha256": event_v1.canonical_sha256(
            {"cache": cache.manifest["cache_sha256"], "normalizer": normalizer.normalizer_sha256}
        ),
        "normalizer": normalizer.manifest,
        "normalizer_sha256": normalizer.normalizer_sha256,
        "source_window_indices_sha256": dataset.window_indices_sha256,
        "batch_order_sha256": sampler.batch_order_sha256,
        "calibration_schedule_sha256": sampler.schedule_sha256,
        "carrier_dim": CARRIER_DIM,
        "side_dim": CARRIER_DIM,
        "calibration_n_trials": 4,
        "fixed_epochs": 50,
        "target_carrier_ridge_lambda": RIDGE_LAMBDA,
        "target_session_optimizer_steps": 0,
        "target_session_backward_steps": 0,
        "deployment_carrier_dense_velocity_opened": False,
        "target_nwb_opened_during_training_setup": False,
        "source_snapshot": {"training_authority": "immutable_context_source_snapshot"},
    }


class H1ContextDatedSourceDataset(Dataset):
    """Parameterised source windows with tied identity/carrier block selection."""

    def __init__(
        self,
        records: Mapping[str, Any],
        cache: DatedContextCarrierCache,
        normalizer: ContextScalarNormalizer,
    ) -> None:
        self.source_sessions = cache.source_sessions
        self.records = {name: records[name] for name in self.source_sessions}
        self.cache = cache
        self.normalizer = normalizer
        self.neural_data: dict[str, np.ndarray] = {}
        self.covariate_data: dict[str, np.ndarray] = {}
        self.eval_mask: dict[str, np.ndarray] = {}
        self.window_indices: list[tuple[str, int]] = []
        self._trial_identity: dict[tuple[str, float], np.ndarray] = {}
        prehistory = WINDOW - 1
        for name in self.source_sessions:
            record = self.records[name]
            self.neural_data[name] = np.pad(record.neural, ((prehistory, 0), (0, 0)), constant_values=0.0)
            self.covariate_data[name] = np.pad(record.velocity, ((prehistory, 0), (0, 0)), constant_values=0.0)
            self.eval_mask[name] = np.pad(record.eval_mask, (prehistory, 0), constant_values=False)
            for start in cache.starts_by_session[name]:
                entry = cache.get(name, start)
                for value in entry.trial_values:
                    key = (name, float(value))
                    if key not in self._trial_identity:
                        self._trial_identity[key] = interpolate_trial_identity(record, value)
            for start in range(self.neural_data[name].shape[0] - WINDOW + 1):
                if self.eval_mask[name][start + WINDOW - 1]:
                    self.window_indices.append((name, start))
        if not self.window_indices:
            raise PilotDataError("dated source dataset contains no eval-valid windows")
        self.window_indices_sha256 = _window_manifest_hash(self.window_indices)

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, request: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
        if not isinstance(request, tuple) or len(request) != 2:
            raise PilotDataError("source samples require one predeclared (window, calibration-start) schedule entry")
        index, calibration_start = int(request[0]), int(request[1])
        session, start = self.window_indices[index]
        end = start + WINDOW
        entry = self.cache.get(session, calibration_start)
        identity = np.stack(
            [self._trial_identity[(session, float(value))] for value in entry.trial_values],
            axis=0,
        )
        if tuple(entry.trial_values) != tuple(
            self.records[session].trial_values[calibration_start : calibration_start + 4]
        ):
            raise PilotDataError("identity/carrier TrialNum alignment drift")
        neural, covariate = self.neural_data[session][start:end], self.covariate_data[session][start:end]
        carrier = self.normalizer.normalize(entry.carrier).astype(np.float32)
        return neural, covariate, identity, session, carrier


class H1ContextDatedBatchSampler(Sampler[list[tuple[int, int]]]):
    """Fixed 50-epoch batch/order/calibration schedule for a dated source pool."""

    def __init__(
        self,
        dataset: H1ContextDatedSourceDataset,
        source_sessions: Sequence[str],
        batch_size: int = 32,
        seed: int = 42,
        max_epochs: int = 50,
    ) -> None:
        if batch_size != 32 or seed != 42 or max_epochs != 50:
            raise PilotDataError("dated paired pilot fixes batch=32, seed=42, epochs=50")
        self.dataset = dataset
        self.source_sessions = tuple(source_sessions)
        self.batch_size = batch_size
        self.seed = seed
        self.max_epochs = max_epochs
        grouped: dict[str, list[int]] = {name: [] for name in self.source_sessions}
        for index, (session, _start) in enumerate(dataset.window_indices):
            grouped[session].append(index)
        batches: list[list[int]] = []
        for name in self.source_sessions:
            indices = random.Random(seed).sample(grouped[name], len(grouped[name]))
            batches.extend(
                indices[offset : offset + batch_size]
                for offset in range(0, len(indices), batch_size)
                if len(indices[offset : offset + batch_size]) == batch_size
            )
        self.batches = random.Random(seed).sample(batches, len(batches))
        self.flat_indices = np.asarray([index for batch in self.batches for index in batch], dtype=np.int64)
        self.batch_order_sha256 = event_v1.array_sha256(self.flat_indices)
        schedule = np.empty((max_epochs, len(self.flat_indices)), dtype=np.int16)
        flat_sessions = np.asarray([dataset.window_indices[int(index)][0] for index in self.flat_indices], dtype=object)
        for name in self.source_sessions:
            positions = np.flatnonzero(flat_sessions == name)
            legal = np.asarray(dataset.cache.starts_by_session[name], dtype=np.int16)
            token = hashlib.sha256(f"{seed}|m4-schedule|{name}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(token[:8], "big"))
            draws = rng.integers(0, len(legal), size=(max_epochs, len(positions)))
            schedule[:, positions] = legal[draws]
        self.schedule = schedule
        self.schedule_sha256 = event_v1.array_sha256(schedule)
        self._epoch = 0

    def __iter__(self):
        epoch = self._epoch
        self._epoch += 1
        if epoch >= self.max_epochs:
            return iter(())
        return iter([(int(index), int(self.schedule[epoch, position])) for position, index in enumerate(self.flat_indices)])

    def __len__(self) -> int:
        return len(self.flat_indices)


class H1ContextStrictTargetDatasetDated(Dataset):
    INTERVENTIONS = ("full", "zero", "row", "label", "tag", "midpoint")

    def __init__(
        self,
        records: Mapping[str, Any],
        sessions: Mapping[str, design.ContextSession],
        latent_map: design.LatentMap,
        normalizer: ContextScalarNormalizer,
        target_sessions: Sequence[str],
        intervention: str = "full",
    ) -> None:
        _need(intervention in self.INTERVENTIONS, "unknown context intervention")
        self.target_sessions = tuple(target_sessions)
        self.records = {n: records[n] for n in self.target_sessions}
        self.sessions = {n: sessions[n] for n in self.target_sessions}
        self.latent_map = latent_map
        self.normalizer = normalizer
        self.intervention = intervention
        self.support: dict[str, ContextTargetSupport] = {}
        self.window_indices: list[tuple[str, int]] = []
        for name in self.target_sessions:
            record, session = self.records[name], self.sessions[name]
            trial_values = tuple(float(x) for x in record.trial_values[:4])
            fifth = float(record.trial_values[4])
            bins = np.flatnonzero(record.eval_mask & np.isfinite(record.trial_num) & (record.trial_num == fifth))
            _need(bins.size > 0, f"{name}: no fifth trial")
            support = design.select_range(session, start=0, budget=4)
            full_raw = _fit(support, latent_map)
            endpoint_order, label_manifest = event_v1.within_trial_label_shuffle(
                tuple(x.base for x in support), session=name, budget=4
            )
            label_raw = _fit(
                tuple(support[int(i)] for i in endpoint_order),
                latent_map,
                response_events=support,
            )
            tag_order, tag_manifest = design.within_trial_tag_shuffle(support, session=name, budget=4)
            tag_raw = _fit(support, latent_map, tags=tuple(support[int(i)].base.tag for i in tag_order))
            no_midpoint = tuple(
                replace(event, midpoint_state=np.zeros(7, dtype=np.float64)) for event in support
            )
            midpoint_raw = _fit(no_midpoint, latent_map, response_events=support)
            row_raw, row_manifest = event_v2.row_shuffle(full_raw, session=name, budget=4)
            full = normalizer.normalize(full_raw)
            carriers = {
                "full": full,
                "zero": np.zeros_like(full),
                "row": normalizer.normalize(row_raw),
                "label": normalizer.normalize(label_raw),
                "tag": normalizer.normalize(tag_raw),
                "midpoint": normalizer.normalize(midpoint_raw),
            }
            _need(
                all(not np.array_equal(full, carriers[key]) for key in ("zero", "row", "label", "tag", "midpoint")),
                f"{name}: collapsed control",
            )
            boundary = int(bins[0])
            self.support[name] = ContextTargetSupport(
                interpolate_identity(record, trial_values),
                carriers,
                {k: event_v1.array_sha256(np.asarray(v, np.float64)) for k, v in carriers.items()},
                trial_values,
                fifth,
                boundary,
                {
                    "endpoint_label_shuffle": label_manifest,
                    "tag_shuffle": tag_manifest,
                    "row_shuffle": row_manifest,
                    "midpoint_ablation": "support event endpoint midpoints set to exact zero before frozen source map",
                },
            )
            for start in range(boundary, record.neural.shape[0] - WINDOW + 1):
                if record.eval_mask[start + WINDOW - 1]:
                    self.window_indices.append((name, start))
        _need(bool(self.window_indices), "dated context target has no query windows")
        self.window_indices_sha256 = _window_manifest_hash(self.window_indices)

    def with_intervention(self, intervention: str) -> "H1ContextStrictTargetDatasetDated":
        clone = object.__new__(type(self))
        clone.records = self.records
        clone.sessions = self.sessions
        clone.latent_map = self.latent_map
        clone.normalizer = self.normalizer
        clone.target_sessions = self.target_sessions
        clone.intervention = intervention
        clone.support = self.support
        clone.window_indices = self.window_indices
        clone.window_indices_sha256 = self.window_indices_sha256
        return clone

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
        name, start = self.window_indices[int(index)]
        record, item, end = self.records[name], self.support[name], start + WINDOW
        _need(start >= item.query_first_bin and record.eval_mask[end - 1], "context target query boundary violation")
        return (
            record.neural[start:end],
            record.velocity[start:end],
            item.identity,
            name,
            np.asarray(item.carriers[self.intervention], np.float32),
        )

    def support_and_carrier_hashes(self) -> dict[str, Any]:
        return {
            n: {
                "trial_values": list(x.trial_values),
                "fifth_trial": x.fifth_trial,
                "query_first_bin": x.query_first_bin,
                "carrier_sha256": dict(x.carrier_sha256),
                "controls": dict(x.control_manifest),
            }
            for n, x in self.support.items()
        }


def build_context_target_dataset_dated(
    *,
    data_dir: str | Path,
    source_module: Any,
    outer_date: str,
) -> H1ContextStrictTargetDatasetDated:
    _need(getattr(source_module, "_setup_done", False), "dated context source must be set up before target")
    target_names = lodo_target_sessions(outer_date)
    records = load_dated_target_records(data_dir, outer_date)
    indexed = index_heldin_calib(data_dir)
    sessions = {
        n: design.load_context_session(event_v1.load_event_session(indexed[n])) for n in target_names
    }
    return H1ContextStrictTargetDatasetDated(
        records,
        sessions,
        source_module.latent_map,
        source_module.normalizer,
        target_names,
    )


def compare_fold0_target_dataset_equivalence(
    *,
    data_dir: str | Path,
    source_module: Any,
) -> dict[str, Any]:
    """Structured comparison of dated fold-0 target dataset vs the sealed class."""
    from src.data.h1_context_event_carrier import build_context_target_dataset, H1ContextStrictTargetDataset

    sealed = build_context_target_dataset(data_dir=data_dir, source_module=source_module)
    dated = build_context_target_dataset_dated(data_dir=data_dir, source_module=source_module, outer_date="19250101")
    sealed_order = tuple(H1_M4_FOLD0_TARGET)
    dated_order = dated.target_sessions
    comparisons: dict[str, Any] = {
        "target_session_order_match": sealed_order == dated_order,
        "sealed_target_order": list(sealed_order),
        "dated_target_order": list(dated_order),
        "window_count_match": len(sealed) == len(dated),
        "sealed_window_count": len(sealed),
        "dated_window_count": len(dated),
        "window_indices_sha256_match": sealed.window_indices_sha256 == dated.window_indices_sha256,
        "sealed_window_indices_sha256": sealed.window_indices_sha256,
        "dated_window_indices_sha256": dated.window_indices_sha256,
        "per_session_carrier_sha256": {},
    }
    all_carriers_match = True
    for name in sealed_order:
        sealed_hashes = sealed.support_and_carrier_hashes()[name]["carrier_sha256"]
        dated_hashes = dated.support_and_carrier_hashes()[name]["carrier_sha256"]
        session_match = sealed_hashes == dated_hashes
        comparisons["per_session_carrier_sha256"][name] = {
            "match": session_match,
            "sealed": sealed_hashes,
            "dated": dated_hashes,
        }
        if not session_match:
            all_carriers_match = False
    comparisons["all_carrier_sha256_match"] = all_carriers_match
    comparisons["equivalent"] = (
        comparisons["target_session_order_match"]
        and comparisons["window_count_match"]
        and comparisons["window_indices_sha256_match"]
        and all_carriers_match
    )
    return comparisons
