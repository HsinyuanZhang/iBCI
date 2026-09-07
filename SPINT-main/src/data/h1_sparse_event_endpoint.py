"""Fold-0 source training and strict target data for H-SE5.

The neural windows, four-trial identity tensors, fixed seed-42 sampler, and
post-four-trial query boundary are inherited from the matched H1 CarrierID
pilot.  Only the carrier cache is replaced by the sparse native endpoint
estimator frozen in ``H1_SPARSE_EVENT_ENDPOINT_V2_ADDENDUM_20260811.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sua_exploration.mc_maze import h1_sparse_event_endpoint as event_v1  # noqa: E402
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as event_v2  # noqa: E402
from src.data.h1_m4_eb_pilot import (  # noqa: E402
    FOLD0_DATE,
    H1_M4_FOLD0_SOURCE,
    H1_M4_FOLD0_TARGET,
    H1M4EBPairedBatchSampler,
    H1M4EBSourceDataset,
    MAX_TRIAL_LENGTH,
    PilotDataError,
    WINDOW,
    _window_manifest_hash,
    index_heldin_calib,
    interpolate_identity,
    legal_contiguous_starts,
    load_source_records,
    load_target_records,
)


SCHEMA = "h1_sparse_event_endpoint_fold0_source_training_v1"
M4_BUDGET = 4
SE5_DIM = 5
NORMALIZER_FLOOR = 1.0e-12


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise PilotDataError(message)


@dataclass(frozen=True)
class SparseCarrierEntry:
    session_name: str
    start_index: int
    trial_values: tuple[float, float, float, float]
    carrier: np.ndarray
    carrier_sha256: str


class SparseCarrierCache:
    def __init__(self, entries: Iterable[SparseCarrierEntry], *, basis: event_v2.EndpointBasisV2) -> None:
        self.entries = tuple(entries)
        self._by_key = {(entry.session_name, entry.start_index): entry for entry in self.entries}
        self.starts_by_session = {
            name: tuple(entry.start_index for entry in self.entries if entry.session_name == name)
            for name in H1_M4_FOLD0_SOURCE
        }
        _need(len(self.entries) == len(self._by_key) == 116, f"H-SE5 cache expected 116 entries, got {len(self.entries)}")
        _need(all(self.starts_by_session.values()), "H-SE5 cache omitted a source session")
        rows = [
            {
                "session": entry.session_name,
                "start_index": entry.start_index,
                "trial_values": list(entry.trial_values),
                "carrier_sha256": entry.carrier_sha256,
            }
            for entry in self.entries
        ]
        body = {
            "schema": "h1_sparse_event_endpoint_m4_fold0_cache_v1",
            "basis_sha256": basis.basis_sha256,
            "carrier_dim": SE5_DIM,
            "ridge_lambda": event_v2.RIDGE_LAMBDA,
            "entries": rows,
        }
        body["cache_sha256"] = event_v1.canonical_sha256(body)
        self.manifest = body

    def get(self, session_name: str, start_index: int) -> SparseCarrierEntry:
        try:
            return self._by_key[(str(session_name), int(start_index))]
        except KeyError as error:
            raise PilotDataError(f"H-SE5 uncached source support {session_name}:{start_index}") from error


@dataclass(frozen=True)
class SparseScalarNormalizer:
    s_src: float
    source_cache_sha256: str
    normalizer_sha256: str

    @property
    def denominator(self) -> float:
        return max(self.s_src, NORMALIZER_FLOOR)

    def normalize(self, carrier: np.ndarray) -> np.ndarray:
        values = np.asarray(carrier, dtype=np.float64)
        _need(values.shape[-1] == SE5_DIM and np.isfinite(values).all(), f"invalid H-SE5 carrier {values.shape}")
        return values / self.denominator

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "formula": "s_src=sqrt(mean(source_cache^2)); carrier_norm=carrier/max(s_src,1e-12)",
            "s_src": self.s_src,
            "denominator": self.denominator,
            "source_cache_sha256": self.source_cache_sha256,
            "normalizer_sha256": self.normalizer_sha256,
        }


def build_sparse_source_assets(
    *, data_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, event_v1.EventSession], event_v2.EndpointBasisV2, SparseCarrierCache, SparseScalarNormalizer]:
    records = load_source_records(data_dir)
    indexed = index_heldin_calib(data_dir)
    event_sessions = {
        name: event_v1.load_event_session(indexed[name])
        for name in H1_M4_FOLD0_SOURCE
    }
    basis = event_v2.fit_source_all_event_basis(event_sessions, outer_date=FOLD0_DATE)
    entries: list[SparseCarrierEntry] = []
    for name in H1_M4_FOLD0_SOURCE:
        record = records[name]
        event_session = event_sessions[name]
        _need(tuple(record.trial_values) == tuple(event_session.trial_values), f"{name}: SPINT/event TrialNum order drift")
        for start in legal_contiguous_starts(record):
            trial_values = tuple(float(value) for value in record.trial_values[start : start + M4_BUDGET])
            carrier, fit = event_v2.fit_session_range(
                event_session, basis, start_index=start, budget=M4_BUDGET,
            )
            _need(carrier.shape == (event_v1.EXPECTED_NEURONS, SE5_DIM), "H-SE5 source carrier shape drift")
            entries.append(SparseCarrierEntry(
                session_name=name,
                start_index=start,
                trial_values=trial_values,  # type: ignore[arg-type]
                carrier=carrier,
                carrier_sha256=fit["carrier_sha256"],
            ))
    cache = SparseCarrierCache(entries, basis=basis)
    stacked = np.stack([entry.carrier for entry in cache.entries])
    scalar = float(np.sqrt(np.mean(np.square(stacked, dtype=np.float64), dtype=np.float64)))
    _need(np.isfinite(scalar) and scalar > 0, "H-SE5 source scalar normalizer is undefined")
    normalizer_body = {
        "formula": "s_src=sqrt(mean(source_cache^2)); carrier_norm=carrier/max(s_src,1e-12)",
        "s_src": scalar,
        "source_cache_sha256": cache.manifest["cache_sha256"],
        "shape": list(stacked.shape),
    }
    normalizer = SparseScalarNormalizer(
        s_src=scalar,
        source_cache_sha256=cache.manifest["cache_sha256"],
        normalizer_sha256=event_v1.canonical_sha256(normalizer_body),
    )
    return records, event_sessions, basis, cache, normalizer


class H1SparseEventSourceDataset(H1M4EBSourceDataset):
    def __init__(self, records: Mapping[str, Any], cache: SparseCarrierCache, normalizer: SparseScalarNormalizer) -> None:
        super().__init__(records, cache)  # type: ignore[arg-type]
        self.normalizer = normalizer

    def __getitem__(self, request):
        neural, target, identity, session, raw = super().__getitem__(request)
        normalized = self.normalizer.normalize(raw).astype(np.float32)
        _need(normalized.shape == (event_v1.EXPECTED_NEURONS, SE5_DIM), "H-SE5 normalized carrier shape drift")
        return neural, target, identity, session, normalized


class H1SparseEventDataModule(pl.LightningDataModule):
    """Source-only fold-0 H-SE5 DataModule with the matched M4 schedule."""

    def __init__(
        self,
        task: str,
        data_dir: str,
        cache_dir: str,
        batch_size: int = 32,
        window_size: int = 700,
        calibration_n_trials: int = 4,
        max_trial_length: int = 1024,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        fixed_epochs: int = 50,
    ) -> None:
        super().__init__()
        fixed = {
            "task": str(task).lower() == "h1",
            "batch_size": int(batch_size) == 32,
            "window_size": int(window_size) == WINDOW,
            "calibration_n_trials": int(calibration_n_trials) == M4_BUDGET,
            "max_trial_length": int(max_trial_length) == MAX_TRIAL_LENGTH,
            "num_workers": int(num_workers) == 0,
            "seed": int(seed) == 42,
            "fixed_epochs": int(fixed_epochs) == 50,
        }
        _need(all(fixed.values()), f"H-SE5 fixed source contract violated: {fixed}")
        self.save_hyperparameters(logger=False)
        self.batch_size_per_device = 32
        self._setup_done = False

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit"}:
            raise RuntimeError("H-SE5 DataModule permits source fit only")
        if self._setup_done:
            return
        if self.trainer is not None and self.trainer.world_size != 1:
            raise PilotDataError("H-SE5 pilot fixes one device per arm")
        records, event_sessions, basis, cache, normalizer = build_sparse_source_assets(data_dir=self.hparams.data_dir)
        dataset = H1SparseEventSourceDataset(records, cache, normalizer)
        sampler = H1M4EBPairedBatchSampler(dataset, batch_size=32, seed=42, max_epochs=50, cache_dir=None)
        manifest = {
            "schema": SCHEMA,
            "fold_date": FOLD0_DATE,
            "source_sessions": list(H1_M4_FOLD0_SOURCE),
            "target_sessions_not_opened": list(H1_M4_FOLD0_TARGET),
            "files": [
                {"session": name, "nwb_sha256": records[name].input_sha256}
                for name in H1_M4_FOLD0_SOURCE
            ],
            "event_parser_sha256": event_v1.sha256_file(Path(event_v1.__file__)),
            "event_estimator_sha256": event_v1.sha256_file(Path(event_v2.__file__)),
            "basis": basis.manifest(),
            "carrier_cache_sha256": cache.manifest["cache_sha256"],
            "normalized_cache_sha256": event_v1.canonical_sha256({
                "cache": cache.manifest["cache_sha256"], "normalizer": normalizer.normalizer_sha256,
            }),
            "normalizer": normalizer.manifest,
            "normalizer_sha256": normalizer.normalizer_sha256,
            "source_window_indices_sha256": dataset.window_indices_sha256,
            "batch_order_sha256": sampler.batch_order_sha256,
            "calibration_schedule_sha256": sampler.schedule_sha256,
            "carrier_dim": SE5_DIM,
            "calibration_n_trials": M4_BUDGET,
            "fixed_epochs": 50,
            "deployment_carrier_dense_velocity_opened": False,
            "offline_decoder_training_behavior_targets_opened": True,
            "target_nwb_opened_during_training_setup": False,
        }
        self.records = records
        self.event_sessions = event_sessions
        self.basis = basis
        self.carrier_cache = cache
        self.normalizer = normalizer
        self.train_dataset = dataset
        self.train_batch_sampler = sampler
        self._manifest = manifest
        self._manifest_sha256 = event_v1.canonical_sha256(manifest)
        self._setup_done = True

    def train_dataloader(self):
        if not self._setup_done:
            raise RuntimeError("call setup('fit') before H-SE5 train_dataloader")
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=0,
            pin_memory=bool(self.hparams.pin_memory),
        )

    def val_dataloader(self):
        return []

    def test_dataloader(self):
        raise RuntimeError("H-SE5 target evaluation is isolated")

    def predict_dataloader(self):
        raise RuntimeError("H-SE5 target evaluation is isolated")

    def pilot_manifest(self) -> dict[str, Any]:
        if not self._setup_done:
            raise RuntimeError("H-SE5 DataModule is not set up")
        return dict(self._manifest)

    @property
    def pilot_manifest_sha256(self) -> str:
        if not self._setup_done:
            raise RuntimeError("H-SE5 DataModule is not set up")
        return self._manifest_sha256


@dataclass(frozen=True)
class SparseTargetSupport:
    session_name: str
    trial_values: tuple[float, float, float, float]
    fifth_trial: float
    query_first_bin: int
    identity: np.ndarray
    carriers: Mapping[str, np.ndarray]
    carrier_sha256: Mapping[str, str]


class H1SparseEventStrictTargetDataset(Dataset):
    INTERVENTIONS = ("full", "zero", "row", "label")

    def __init__(
        self,
        records: Mapping[str, Any],
        event_sessions: Mapping[str, event_v1.EventSession],
        basis: event_v2.EndpointBasisV2,
        normalizer: SparseScalarNormalizer,
        intervention: str = "full",
    ) -> None:
        _need(intervention in self.INTERVENTIONS, f"unknown H-SE5 target intervention {intervention}")
        self.records = {name: records[name] for name in H1_M4_FOLD0_TARGET}
        self.event_sessions = {name: event_sessions[name] for name in H1_M4_FOLD0_TARGET}
        self.basis = basis
        self.normalizer = normalizer
        self.intervention = intervention
        self.support: dict[str, SparseTargetSupport] = {}
        self.window_indices: list[tuple[str, int]] = []
        for name in H1_M4_FOLD0_TARGET:
            record = self.records[name]
            events = self.event_sessions[name]
            _need(tuple(record.trial_values) == tuple(events.trial_values), f"{name}: target trial order drift")
            trial_values = tuple(float(value) for value in record.trial_values[:M4_BUDGET])
            fifth = float(record.trial_values[M4_BUDGET])
            fifth_bins = np.flatnonzero(record.eval_mask & np.isfinite(record.trial_num) & (record.trial_num == fifth))
            _need(fifth_bins.size > 0, f"{name}: missing fifth-trial boundary")
            boundary = int(fifth_bins[0])
            identity = interpolate_identity(record, trial_values)
            full_raw, _ = event_v2.fit_session_range(events, basis, start_index=0, budget=M4_BUDGET)
            label_raw, _ = event_v2.fit_session_range(
                events, basis, start_index=0, budget=M4_BUDGET, shuffled_labels=True,
            )
            row_raw, _ = event_v2.row_shuffle(full_raw, session=name, budget=M4_BUDGET)
            full = normalizer.normalize(full_raw)
            carriers = {
                "full": full,
                "zero": np.zeros_like(full),
                "row": normalizer.normalize(row_raw),
                "label": normalizer.normalize(label_raw),
            }
            _need(not np.array_equal(carriers["full"], carriers["zero"]), "H-SE5 Full collapsed to Zero5")
            _need(not np.array_equal(carriers["full"], carriers["row"]), "H-SE5 row shuffle is identity")
            _need(not np.array_equal(carriers["full"], carriers["label"]), "H-SE5 label shuffle is identity")
            hashes = {key: event_v1.array_sha256(np.asarray(value, np.float64)) for key, value in carriers.items()}
            self.support[name] = SparseTargetSupport(
                session_name=name,
                trial_values=trial_values,  # type: ignore[arg-type]
                fifth_trial=fifth,
                query_first_bin=boundary,
                identity=identity,
                carriers=carriers,
                carrier_sha256=hashes,
            )
            for start in range(boundary, record.neural.shape[0] - WINDOW + 1):
                if record.eval_mask[start + WINDOW - 1]:
                    self.window_indices.append((name, start))
        _need(bool(self.window_indices), "H-SE5 strict target has no query windows")
        self.window_indices_sha256 = _window_manifest_hash(self.window_indices)

    def with_intervention(self, intervention: str) -> "H1SparseEventStrictTargetDataset":
        _need(intervention in self.INTERVENTIONS, f"unknown H-SE5 intervention {intervention}")
        clone = object.__new__(type(self))
        clone.records = self.records
        clone.event_sessions = self.event_sessions
        clone.basis = self.basis
        clone.normalizer = self.normalizer
        clone.intervention = intervention
        clone.support = self.support
        clone.window_indices = self.window_indices
        clone.window_indices_sha256 = self.window_indices_sha256
        return clone

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, index: int):
        session, start = self.window_indices[int(index)]
        end = start + WINDOW
        record = self.records[session]
        support = self.support[session]
        _need(start >= support.query_first_bin and record.eval_mask[end - 1], "H-SE5 query boundary violation")
        return (
            record.neural[start:end],
            record.velocity[start:end],
            support.identity,
            session,
            np.asarray(support.carriers[self.intervention], dtype=np.float32),
        )

    def support_and_carrier_hashes(self) -> dict[str, Any]:
        return {
            name: {
                "trial_values": list(value.trial_values),
                "fifth_trial": value.fifth_trial,
                "query_first_bin": value.query_first_bin,
                "carrier_sha256": dict(value.carrier_sha256),
            }
            for name, value in self.support.items()
        }


def build_sparse_target_dataset(
    *, data_dir: str | Path, source_module: H1SparseEventDataModule,
) -> H1SparseEventStrictTargetDataset:
    _need(source_module._setup_done, "H-SE5 source module must be set up before target access")
    records = load_target_records(data_dir)
    indexed = index_heldin_calib(data_dir)
    events = {name: event_v1.load_event_session(indexed[name]) for name in H1_M4_FOLD0_TARGET}
    return H1SparseEventStrictTargetDataset(records, events, source_module.basis, source_module.normalizer)

