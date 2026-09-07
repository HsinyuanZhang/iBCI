"""Isolated GPU data contract for the source-screened ``ser_context_q4`` arm.

The carrier is deliberately the same five-wide consumer interface as H-SE5,
but its four slope coordinates are a *source-only* source-encoding map of
``[endpoint_delta(7), endpoint_midpoint(7), native_tag_one_hot]``.  It never
uses dense velocity to form a carrier: target labels are native event
``log_rates`` and the only target fit is the closed-form ridge (lambda=3).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset

from sua_exploration.mc_maze import h1_event_carrier_design_screen as design
from sua_exploration.mc_maze import h1_sparse_event_endpoint as event_v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as event_v2
from src.data.h1_m4_eb_pilot import (
    FOLD0_DATE, H1_M4_FOLD0_SOURCE, H1_M4_FOLD0_TARGET, H1M4EBPairedBatchSampler,
    H1M4EBSourceDataset, MAX_TRIAL_LENGTH, PilotDataError, WINDOW, _window_manifest_hash,
    index_heldin_calib, interpolate_identity, legal_contiguous_starts, load_source_records,
    load_target_records,
)

SCHEMA = "h1_context_event_carrier_fold0_source_training_v1"
CONTEXT_PROTOCOL = "h1_ser_context_q4_m4_fold0_v1"
M4_BUDGET, CARRIER_DIM, RIDGE_LAMBDA = 4, 5, 3.0
NORMALIZER_FLOOR = 1.0e-12


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise PilotDataError(message)


@dataclass(frozen=True)
class ContextCarrierEntry:
    session_name: str
    start_index: int
    trial_values: tuple[float, float, float, float]
    carrier: np.ndarray
    carrier_sha256: str


class ContextCarrierCache:
    def __init__(self, entries: Iterable[ContextCarrierEntry], latent_map: design.LatentMap) -> None:
        self.entries = tuple(entries)
        self._by_key = {(item.session_name, item.start_index): item for item in self.entries}
        self.starts_by_session = {name: tuple(item.start_index for item in self.entries if item.session_name == name)
                                  for name in H1_M4_FOLD0_SOURCE}
        _need(len(self.entries) == len(self._by_key) == 116 and all(self.starts_by_session.values()),
              "context cache must contain all 116 legal source ranges")
        body = {"schema": "h1_context_event_carrier_m4_fold0_cache_v1", "context_map_sha256": latent_map.map_sha256,
                "candidate": "ser_context_q4", "carrier_dim": CARRIER_DIM, "ridge_lambda": RIDGE_LAMBDA,
                "entries": [{"session": x.session_name, "start_index": x.start_index,
                             "trial_values": list(x.trial_values), "carrier_sha256": x.carrier_sha256}
                            for x in self.entries]}
        body["cache_sha256"] = event_v1.canonical_sha256(body)
        self.manifest = body

    def get(self, session_name: str, start_index: int) -> ContextCarrierEntry:
        try:
            return self._by_key[(str(session_name), int(start_index))]
        except KeyError as exc:
            raise PilotDataError(f"uncached context support {session_name}:{start_index}") from exc


@dataclass(frozen=True)
class ContextScalarNormalizer:
    s_src: float
    source_cache_sha256: str
    normalizer_sha256: str

    @property
    def denominator(self) -> float:
        return max(float(self.s_src), NORMALIZER_FLOOR)

    def normalize(self, carrier: np.ndarray) -> np.ndarray:
        value = np.asarray(carrier, np.float64)
        _need(value.shape[-1] == CARRIER_DIM and np.isfinite(value).all(), "invalid context carrier")
        return value / self.denominator

    @property
    def manifest(self) -> dict[str, Any]:
        return {"formula": "s_src=sqrt(mean(source_context_cache^2)); carrier_norm=carrier/max(s_src,1e-12)",
                "s_src": self.s_src, "denominator": self.denominator,
                "source_cache_sha256": self.source_cache_sha256, "normalizer_sha256": self.normalizer_sha256}


def _normalizer(cache: ContextCarrierCache) -> ContextScalarNormalizer:
    values = np.stack([x.carrier for x in cache.entries])
    scalar = float(np.sqrt(np.mean(np.square(values, dtype=np.float64), dtype=np.float64)))
    _need(np.isfinite(scalar) and scalar > 0, "context source normalizer undefined")
    body = {"formula": "s_src=sqrt(mean(source_context_cache^2)); carrier_norm=carrier/max(s_src,1e-12)",
            "s_src": scalar, "source_cache_sha256": cache.manifest["cache_sha256"], "shape": list(values.shape)}
    return ContextScalarNormalizer(scalar, cache.manifest["cache_sha256"], event_v1.canonical_sha256(body))


def build_context_manifest(*, records, latent_map: design.LatentMap, cache: ContextCarrierCache,
                           normalizer: ContextScalarNormalizer, dataset, sampler) -> dict[str, Any]:
    """Canonical training manifest, shared by the snapshot builder and DataModule."""
    return {"schema": SCHEMA, "protocol": CONTEXT_PROTOCOL, "fold_date": FOLD0_DATE,
            "source_sessions": list(H1_M4_FOLD0_SOURCE), "target_sessions_not_opened": list(H1_M4_FOLD0_TARGET),
            "files": [{"session": n, "nwb_sha256": records[n].input_sha256} for n in H1_M4_FOLD0_SOURCE],
            "candidate": "ser_context_q4", "raw_label": "endpoint_delta(7)+endpoint_midpoint(7)+native_tag_one_hot",
            "source_map": latent_map.manifest(), "basis": {"basis_sha256": latent_map.map_sha256, "kind": "source_encoding_context_map"},
            "carrier_cache_sha256": cache.manifest["cache_sha256"],
            "normalized_cache_sha256": event_v1.canonical_sha256({"cache": cache.manifest["cache_sha256"], "normalizer": normalizer.normalizer_sha256}),
            "normalizer": normalizer.manifest, "normalizer_sha256": normalizer.normalizer_sha256,
            "source_window_indices_sha256": dataset.window_indices_sha256, "batch_order_sha256": sampler.batch_order_sha256,
            "calibration_schedule_sha256": sampler.schedule_sha256, "carrier_dim": CARRIER_DIM, "side_dim": CARRIER_DIM,
            "calibration_n_trials": 4, "fixed_epochs": 50, "target_carrier_ridge_lambda": RIDGE_LAMBDA,
            "target_session_optimizer_steps": 0, "target_session_backward_steps": 0,
            "deployment_carrier_dense_velocity_opened": False, "target_nwb_opened_during_training_setup": False,
            "source_snapshot": {"training_authority": "immutable_context_source_snapshot"}}


def _fit(events: tuple[design.ContextEvent, ...], latent_map: design.LatentMap, *,
         response_events: tuple[design.ContextEvent, ...] | None = None,
         tags: tuple[str, ...] | None = None) -> np.ndarray:
    latent = latent_map.transform(events, tag_overrides=tags)
    response_events = events if response_events is None else response_events
    _need(len(response_events) == len(events), "context feature/response association length drift")
    response = np.stack([event.base.log_rates for event in response_events]).astype(np.float64)
    durations = np.asarray([event.base.duration_seconds for event in response_events], np.float64)
    carrier = design.fit_target_carrier(latent, response, durations=durations, weighted=False)
    _need(carrier.shape == (event_v1.EXPECTED_NEURONS, CARRIER_DIM), "context ridge output drift")
    return carrier


def build_context_source_assets(data_dir: str | Path, *, frozen_map: design.LatentMap | None = None):
    records = load_source_records(data_dir); indexed = index_heldin_calib(data_dir)
    sessions = {name: design.load_context_session(event_v1.load_event_session(indexed[name])) for name in H1_M4_FOLD0_SOURCE}
    latent_map = frozen_map or design.fit_latent_map(sessions, outer_date=FOLD0_DATE,
        candidate=next(x for x in design.CANDIDATES if x.name == "ser_context_q4"))
    _need(latent_map.outer_date == FOLD0_DATE and latent_map.candidate.name == "ser_context_q4"
          and latent_map.candidate.rank == 4, "context source map contract drift")
    entries: list[ContextCarrierEntry] = []
    for name in H1_M4_FOLD0_SOURCE:
        record = records[name]; session = sessions[name]
        _need(tuple(record.trial_values) == tuple(session.base.trial_values), f"{name}: trial alignment drift")
        for start in legal_contiguous_starts(record):
            support = design.select_range(session, start=start, budget=M4_BUDGET)
            carrier = _fit(support, latent_map)
            entries.append(ContextCarrierEntry(name, start, tuple(float(x) for x in record.trial_values[start:start + 4]),
                                               carrier, event_v1.array_sha256(carrier)))
    cache = ContextCarrierCache(entries, latent_map)
    return records, sessions, latent_map, cache, _normalizer(cache)


class H1ContextSourceDataset(H1M4EBSourceDataset):
    def __init__(self, records, cache: ContextCarrierCache, normalizer: ContextScalarNormalizer) -> None:
        super().__init__(records, cache)  # type: ignore[arg-type]
        self.normalizer = normalizer

    def __getitem__(self, request):
        neural, target, identity, session, carrier = super().__getitem__(request)
        return neural, target, identity, session, self.normalizer.normalize(carrier).astype(np.float32)


class H1ContextEventDataModule(pl.LightningDataModule):
    """Source-only M4 training data; target access is rejected until evaluator use."""
    def __init__(self, task: str, data_dir: str, cache_dir: str, batch_size: int = 32, window_size: int = 700,
                 calibration_n_trials: int = 4, max_trial_length: int = 1024, num_workers: int = 0,
                 pin_memory: bool = False, seed: int = 42, fixed_epochs: int = 50,
                 source_snapshot_receipt: str | None = None, allow_live_source_map_for_snapshot: bool = False) -> None:
        super().__init__()
        _need(str(task).lower() == "h1" and batch_size == 32 and window_size == WINDOW and calibration_n_trials == 4
              and max_trial_length == MAX_TRIAL_LENGTH and num_workers == 0 and seed == 42 and fixed_epochs == 50,
              "context M4 fixed source contract violated")
        _need(bool(source_snapshot_receipt) or bool(allow_live_source_map_for_snapshot),
              "training requires an immutable context source snapshot receipt")
        self.save_hyperparameters(logger=False); self.batch_size_per_device = 32; self._setup_done = False

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit"): raise RuntimeError("context arm permits source fit only")
        if self._setup_done: return
        if self.trainer is not None and self.trainer.world_size != 1: raise PilotDataError("context arm fixes one GPU")
        # Snapshot creation is the sole explicitly marked live-SVD path.  Every
        # trainable configuration must load the immutable map before rebuilding
        # its 116 source carriers, normalizer, and manifest.
        if self.hparams.source_snapshot_receipt:
            from src.data.h1_context_event_source_snapshot import load_snapshot
            frozen = load_snapshot(self.hparams.source_snapshot_receipt)
            records, sessions, latent_map, cache, normalizer = build_context_source_assets(
                self.hparams.data_dir, frozen_map=frozen["latent_map"])
            # Recompute all 116 map-derived ranges above as an audit, then use
            # the serialized, hash-bound cache for training so Torch/LAPACK
            # process state can never alter a last bit reaching GPU input.
            raw = frozen["arrays"]["source_carriers"]
            entries = tuple(ContextCarrierEntry(str(meta["session"]), int(meta["start_index"]),
                tuple(float(v) for v in meta["trial_values"]), np.asarray(raw[i], np.float64), str(meta["carrier_sha256"]))
                for i, meta in enumerate(frozen["metadata"]["cache_entries"]))
            snap_cache = ContextCarrierCache(entries, latent_map)
            _need(snap_cache.manifest == frozen["metadata"]["cache_manifest"], "serialized context cache manifest drift")
            cache = snap_cache
            n = frozen["metadata"]["normalizer"]
            normalizer = ContextScalarNormalizer(float(n["s_src"]), str(n["source_cache_sha256"]), str(n["normalizer_sha256"]))
            snapshot_manifest = frozen["manifest"]
            snapshot_binding = {"training_authority": "immutable_context_source_snapshot"}
        else:
            records, sessions, latent_map, cache, normalizer = build_context_source_assets(self.hparams.data_dir)
            snapshot_manifest, snapshot_binding = None, {"training_authority": "immutable_context_source_snapshot"}
        dataset = H1ContextSourceDataset(records, cache, normalizer)
        sampler = H1M4EBPairedBatchSampler(dataset, batch_size=32, seed=42, max_epochs=50, cache_dir=None)
        manifest = build_context_manifest(records=records, latent_map=latent_map, cache=cache, normalizer=normalizer,
                                          dataset=dataset, sampler=sampler)
        if snapshot_manifest is not None:
            # This proves the snapshot map, map-derived cache, normalizer and
            # complete manifest reach training without a process-local SVD.
            expected = dict(snapshot_manifest)
            observed_sha, expected_sha = event_v1.canonical_sha256(manifest), event_v1.canonical_sha256(expected)
            _need(observed_sha == expected_sha,
                  f"snapshot-rebuilt context manifest differs from immutable snapshot ({observed_sha} != {expected_sha})")
            manifest = expected
        self.records, self.context_sessions, self.latent_map, self.carrier_cache, self.normalizer = records, sessions, latent_map, cache, normalizer
        self.train_dataset, self.train_batch_sampler, self._manifest = dataset, sampler, manifest
        self._manifest_sha256 = event_v1.canonical_sha256(manifest); self._setup_done = True

    def train_dataloader(self):
        if not self._setup_done: raise RuntimeError("call setup('fit')")
        return DataLoader(self.train_dataset, batch_sampler=self.train_batch_sampler, num_workers=0, pin_memory=bool(self.hparams.pin_memory))
    def val_dataloader(self): return []
    def test_dataloader(self): raise RuntimeError("context target evaluation is isolated")
    def predict_dataloader(self): raise RuntimeError("context target evaluation is isolated")
    def pilot_manifest(self):
        if not self._setup_done: raise RuntimeError("context DataModule not set up")
        return dict(self._manifest)
    @property
    def pilot_manifest_sha256(self):
        if not self._setup_done: raise RuntimeError("context DataModule not set up")
        return self._manifest_sha256


@dataclass(frozen=True)
class ContextTargetSupport:
    identity: np.ndarray; carriers: Mapping[str, np.ndarray]; carrier_sha256: Mapping[str, str]
    trial_values: tuple[float, float, float, float]; fifth_trial: float; query_first_bin: int; control_manifest: Mapping[str, Any]


class H1ContextStrictTargetDataset(Dataset):
    INTERVENTIONS = ("full", "zero", "row", "label", "tag", "midpoint")
    def __init__(self, records, sessions, latent_map, normalizer, intervention: str = "full") -> None:
        _need(intervention in self.INTERVENTIONS, "unknown context intervention")
        self.records = {n: records[n] for n in H1_M4_FOLD0_TARGET}; self.sessions = {n: sessions[n] for n in H1_M4_FOLD0_TARGET}
        self.latent_map, self.normalizer, self.intervention, self.support, self.window_indices = latent_map, normalizer, intervention, {}, []
        for name in H1_M4_FOLD0_TARGET:
            record, session = self.records[name], self.sessions[name]
            trial_values = tuple(float(x) for x in record.trial_values[:4]); fifth = float(record.trial_values[4])
            bins = np.flatnonzero(record.eval_mask & np.isfinite(record.trial_num) & (record.trial_num == fifth)); _need(bins.size > 0, f"{name}: no fifth trial")
            support = design.select_range(session, start=0, budget=4); full_raw = _fit(support, latent_map)
            endpoint_order, label_manifest = event_v1.within_trial_label_shuffle(tuple(x.base for x in support), session=name, budget=4)
            label_raw = _fit(tuple(support[int(i)] for i in endpoint_order), latent_map, response_events=support)
            tag_order, tag_manifest = design.within_trial_tag_shuffle(support, session=name, budget=4)
            tag_raw = _fit(support, latent_map, tags=tuple(support[int(i)].base.tag for i in tag_order))
            no_midpoint = tuple(replace(event, midpoint_state=np.zeros(7, dtype=np.float64)) for event in support)
            midpoint_raw = _fit(no_midpoint, latent_map, response_events=support)
            row_raw, row_manifest = event_v2.row_shuffle(full_raw, session=name, budget=4)
            full = normalizer.normalize(full_raw)
            carriers = {"full": full, "zero": np.zeros_like(full), "row": normalizer.normalize(row_raw),
                        "label": normalizer.normalize(label_raw), "tag": normalizer.normalize(tag_raw),
                        "midpoint": normalizer.normalize(midpoint_raw)}
            _need(all(not np.array_equal(full, carriers[key]) for key in ("zero", "row", "label", "tag", "midpoint")), f"{name}: collapsed control")
            boundary = int(bins[0]); self.support[name] = ContextTargetSupport(interpolate_identity(record, trial_values), carriers,
                {k: event_v1.array_sha256(np.asarray(v, np.float64)) for k, v in carriers.items()}, trial_values, fifth, boundary,
                {"endpoint_label_shuffle": label_manifest, "tag_shuffle": tag_manifest, "row_shuffle": row_manifest,
                 "midpoint_ablation": "support event endpoint midpoints set to exact zero before frozen source map"})
            for start in range(boundary, record.neural.shape[0] - WINDOW + 1):
                if record.eval_mask[start + WINDOW - 1]: self.window_indices.append((name, start))
        _need(bool(self.window_indices), "context target has no query windows"); self.window_indices_sha256 = _window_manifest_hash(self.window_indices)
    def with_intervention(self, intervention: str):
        clone = object.__new__(type(self)); clone.records, clone.sessions, clone.latent_map, clone.normalizer = self.records, self.sessions, self.latent_map, self.normalizer
        clone.intervention, clone.support, clone.window_indices, clone.window_indices_sha256 = intervention, self.support, self.window_indices, self.window_indices_sha256; return clone
    def __len__(self): return len(self.window_indices)
    def __getitem__(self, index):
        name, start = self.window_indices[int(index)]; record, item, end = self.records[name], self.support[name], start + WINDOW
        _need(start >= item.query_first_bin and record.eval_mask[end - 1], "context target query boundary violation")
        return record.neural[start:end], record.velocity[start:end], item.identity, name, np.asarray(item.carriers[self.intervention], np.float32)
    def support_and_carrier_hashes(self):
        return {n: {"trial_values": list(x.trial_values), "fifth_trial": x.fifth_trial, "query_first_bin": x.query_first_bin,
                    "carrier_sha256": dict(x.carrier_sha256), "controls": dict(x.control_manifest)} for n, x in self.support.items()}


def build_context_target_dataset(*, data_dir: str | Path, source_module: H1ContextEventDataModule) -> H1ContextStrictTargetDataset:
    _need(source_module._setup_done, "context source must be set up before target")
    records, indexed = load_target_records(data_dir), index_heldin_calib(data_dir)
    sessions = {n: design.load_context_session(event_v1.load_event_session(indexed[n])) for n in H1_M4_FOLD0_TARGET}
    return H1ContextStrictTargetDataset(records, sessions, source_module.latent_map, source_module.normalizer)
