"""Source-only H1 CarrierID training controls with corrupted carrier content.

The Full/Zero H1 CarrierID artifacts remain untouched.  This module reuses the
immutable normalized-V2 source cache and its fixed 50-epoch sampler, changing
only the carrier returned to the *source training* batch:

* ``rs``: deterministic complete neuron-row permutation of each carrier;
* ``ls``: deterministic within-support-trial velocity-label rotation followed
  by a fresh frozen carrier fit.

Neither control imports, enumerates, or opens an H1 target/minival/formal
recording.  Target evaluation is intentionally not implemented here.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.data.h1_m4_eb_normalized_v2 import (
    H1M4EBNormalizedV2DataModule,
    H1M4EBNormalizedV2SourceDataset,
)
from src.data.h1_m4_eb_pilot import (
    H1M4EBPairedBatchSampler,
    PilotDataError,
    complete_row_shuffle,
    label_rotation_carrier,
)
from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    array_sha256,
    canonical_sha256,
)


SHUFFLE_ARMS = ("rs", "ls")


class H1CarrierIdShuffleSourceDataset(H1M4EBNormalizedV2SourceDataset):
    """Normalized source dataset whose carrier is deterministically corrupted."""

    def __init__(self, *args: Any, plan: Any, intervention: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if intervention not in SHUFFLE_ARMS:
            raise NormalizedV2ContractError(f"unknown H1 CarrierID shuffle arm: {intervention!r}")
        self.plan = plan
        self.intervention = intervention
        effective: dict[tuple[str, int], np.ndarray] = {}
        full_values: list[np.ndarray] = []
        effective_values: list[np.ndarray] = []
        # Label re-fits are deliberately paid once for the 116 legal source
        # supports, never inside the millions of train-dataloader __getitem__
        # calls.  Cache order is immutable and recorded in the source manifest.
        for entry in self.cache.entries:
            key = (str(entry.session_name), int(entry.start_index))
            raw = np.asarray(entry.carrier, dtype=np.float64)
            if intervention == "rs":
                changed = complete_row_shuffle(raw, key[0])
            else:
                changed = label_rotation_carrier(self.records[key[0]], self.plan, entry.trial_values)
            normalized = self.normalizer.normalize(np.asarray(changed, dtype=np.float64))
            full = self.normalizer.normalize(raw)
            if normalized.shape != raw.shape or not np.isfinite(normalized).all() or np.array_equal(normalized, full):
                raise PilotDataError("H1 CarrierID all-entry source shuffle is invalid or an identity intervention")
            effective[key] = normalized.astype(np.float32)
            full_values.append(full)
            effective_values.append(normalized)
        if len(effective) != 116:
            raise NormalizedV2ContractError(f"H1 CarrierID expected 116 effective source carriers, got {len(effective)}")
        full_stack = np.stack(full_values, axis=0)
        effective_stack = np.stack(effective_values, axis=0)
        self._effective = effective
        self.effective_source_carriers_sha256 = array_sha256(effective_stack)
        self.effective_source_carriers_shape = list(effective_stack.shape)
        self.effective_source_carriers_count = len(effective)
        self.effective_source_carriers_nonidentity_all = not np.array_equal(full_stack, effective_stack)

    def _intervened_raw_carrier(self, request: tuple[int, int], session: str) -> np.ndarray:
        _index, calibration_start = (int(request[0]), int(request[1]))
        try:
            return self._effective[(session, calibration_start)]
        except KeyError as exc:
            raise PilotDataError("H1 CarrierID source request is not in effective carrier cache") from exc

    def __getitem__(self, request: tuple[int, int]):
        neural, target, identity, session, _normalized_full = super().__getitem__(request)
        shuffled = self._intervened_raw_carrier(request, session)
        if shuffled.shape != (self.records[session].num_neurons, 4) or not np.isfinite(shuffled).all():
            raise PilotDataError("H1 CarrierID shuffled normalized carrier shape/finite check failed")
        return neural, target, identity, session, shuffled


class H1CarrierIdShuffleDataModule(H1M4EBNormalizedV2DataModule):
    """Strict source-only V2 DataModule for separately-trained RS/LS arms."""

    def __init__(self, *, carrier_intervention: str, **kwargs: Any) -> None:
        if carrier_intervention not in SHUFFLE_ARMS:
            raise NormalizedV2ContractError("H1 CarrierID shuffle requires carrier_intervention in {'rs','ls'}")
        super().__init__(**kwargs)
        self.carrier_intervention = carrier_intervention

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit"}:
            raise RuntimeError("H1 CarrierID shuffle DataModule permits source fit only")
        if self._setup_done:
            return
        super().setup(stage)
        original_dataset = self.train_dataset
        shuffled_dataset = H1CarrierIdShuffleSourceDataset(
            self.records, self.carrier_cache, self.normalizer,
            plan=self.plan, intervention=self.carrier_intervention,
        )
        shuffled_sampler = H1M4EBPairedBatchSampler(
            shuffled_dataset, batch_size=32, seed=42, max_epochs=50, cache_dir=None,
        )
        if (
            shuffled_dataset.window_indices_sha256 != original_dataset.window_indices_sha256
            or not np.array_equal(shuffled_sampler.schedule, self.train_batch_sampler.schedule)
            or shuffled_sampler.batch_order_sha256 != self.train_batch_sampler.batch_order_sha256
            or shuffled_sampler.schedule_sha256 != self.train_batch_sampler.schedule_sha256
        ):
            raise NormalizedV2ContractError("H1 CarrierID shuffle changed the source window/order/schedule")
        # Bind the explicit source-only intervention to the manifest while
        # retaining the exact shared raw/normalized cache and schedule hashes.
        manifest = dict(self._manifest)
        manifest.update({
            "schema": "h1_carrierid_h32_shuffle_source_training_manifest_v1",
            "carrier_intervention": self.carrier_intervention,
            "source_carrier_intervention_only": True,
            "target_nwb_opened_during_training_setup": False,
            "minival_or_heldout_enumerated": False,
            "effective_source_carriers_sha256": shuffled_dataset.effective_source_carriers_sha256,
            "effective_source_carriers_shape": shuffled_dataset.effective_source_carriers_shape,
            "effective_source_carriers_count": shuffled_dataset.effective_source_carriers_count,
            "effective_source_carriers_nonidentity_all": shuffled_dataset.effective_source_carriers_nonidentity_all,
        })
        # Exercise a real source request, record only non-sensitive digests,
        # and fail closed if the requested corruption accidentally collapses.
        request = (0, int(shuffled_sampler.schedule[0, 0]))
        raw = original_dataset.normalized_carrier(request)
        changed = shuffled_dataset[request][4]
        if np.array_equal(raw, changed):
            raise NormalizedV2ContractError("H1 CarrierID shuffle first source carrier is unchanged")
        manifest["first_source_carrier_audit"] = {
            "full_normalized_sha256": array_sha256(raw),
            "intervened_normalized_sha256": array_sha256(changed),
            "nonidentity": True,
        }
        self.train_dataset = shuffled_dataset
        self.train_batch_sampler = shuffled_sampler
        self._manifest = manifest
        self._manifest_sha256 = canonical_sha256(manifest)

    def test_dataloader(self):
        raise RuntimeError("H1 CarrierID shuffle formal test loader is forbidden")

    def predict_dataloader(self):
        raise RuntimeError("H1 CarrierID shuffle target prediction is forbidden")
