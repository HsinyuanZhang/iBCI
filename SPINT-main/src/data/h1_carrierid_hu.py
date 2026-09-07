"""H-U source DataModule: same H32 fold-0 harness, label-free 4-D identity.

Reuses the immutable H-C producer, window schedule, and source-only RMS
formula.  The only change at the model boundary is the 4-D vector handed to
the compact CarrierID consumer.  Existing H-C / H-LS / H-S / H-C0 paths are
not imported for mutation and are not rewritten.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.data.h1_carrierid_hu_features import (
    CARRIER_WIDTH,
    DECLARED_DEAD_CHANNELS,
    FEATURE_NAMES,
    H1_NUM_NEURONS,
    VERSION,
    hu_from_record,
)
from src.data.h1_m4_eb_normalized_v2 import (
    H1M4EBNormalizedV2DataModule,
    H1M4EBNormalizedV2SourceDataset,
    H1M4EBNormalizedV2StrictTargetDataset,
)
from src.data.h1_m4_eb_pilot import (
    EXPECTED_NEURONS,
    H1M4EBPairedBatchSampler,
    PilotDataError,
    carrier_sha256,
)
from src.h1_m4_eb_normalized_v2_contract import (
    NORMALIZER_FLOOR,
    NORMALIZER_FORMULA,
    NormalizedV2ContractError,
    array_sha256,
    canonical_sha256,
    fit_source_scalar_normalizer,
)


if EXPECTED_NEURONS != H1_NUM_NEURONS:
    raise NormalizedV2ContractError("H-U H1_NUM_NEURONS drifted from EXPECTED_NEURONS")


HU_SOURCE_AUTHORITY_RECEIPT_SHA256 = "fcb0cf843f351715677f14e3cf80acdb613fc5a59fb96bfcf7efc23836b16fb8"
HU_SOURCE_AUTHORITY_DIRECTORY_NAME = "source_authority_v1"


class H1CarrierIdHuSourceDataset(H1M4EBNormalizedV2SourceDataset):
    """Normalized source dataset whose carrier is the label-free H-U descriptor."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        raw_values: list[np.ndarray] = []
        normalized_map: dict[tuple[str, int], np.ndarray] = {}
        silent_union: set[int] = set()
        for entry in self.cache.entries:
            record = self.records[str(entry.session_name)]
            raw, audit = hu_from_record(record, entry.trial_values)
            if raw.shape != (record.num_neurons, CARRIER_WIDTH):
                raise NormalizedV2ContractError(f"H-U raw shape drift: {raw.shape}")
            if audit.used_velocity or audit.used_behaviour_labels:
                raise NormalizedV2ContractError("H-U audit claims a behaviour read")
            silent_union.update(audit.silent_channels)
            raw_values.append(np.asarray(raw, dtype=np.float64))
        stacked = np.stack(raw_values, axis=0)
        if stacked.shape != (116, H1_NUM_NEURONS, CARRIER_WIDTH):
            raise NormalizedV2ContractError(f"H-U source stack shape drift: {stacked.shape}")
        self.hu_raw_sha256 = array_sha256(stacked)
        self.hu_normalizer = fit_source_scalar_normalizer(stacked, self.hu_raw_sha256)
        for entry, raw in zip(self.cache.entries, raw_values):
            normalized = self.hu_normalizer.normalize(raw)
            hc = self.normalizer.normalize(np.asarray(entry.carrier, dtype=np.float64))
            if np.array_equal(normalized, hc):
                raise PilotDataError("H-U collapsed onto the H-C normalized carrier")
            normalized_map[(str(entry.session_name), int(entry.start_index))] = normalized.astype(np.float32)
        self._hu = normalized_map
        ordered = np.stack(
            [normalized_map[(str(entry.session_name), int(entry.start_index))] for entry in self.cache.entries],
            axis=0,
        ).astype(np.float64)
        self.effective_source_carriers_sha256 = array_sha256(ordered)
        self.effective_source_carriers_shape = [116, H1_NUM_NEURONS, CARRIER_WIDTH]
        self.effective_source_carriers_count = 116
        self.effective_source_carriers_nonidentity_all = True
        self.hu_silent_channels = tuple(sorted(silent_union))
        self.hu_feature_names = FEATURE_NAMES
        self.hu_version = VERSION

    def __getitem__(self, request: tuple[int, int]):
        neural, target, identity, session, _hc = super().__getitem__(request)
        _window_index, calibration_start = int(request[0]), int(request[1])
        try:
            hu = self._hu[(session, calibration_start)]
        except KeyError as exc:
            raise PilotDataError("H-U source request is not in the precomputed support cache") from exc
        if hu.shape != (self.records[session].num_neurons, CARRIER_WIDTH) or not np.isfinite(hu).all():
            raise PilotDataError("H-U normalized carrier shape/finite check failed")
        return neural, target, identity, session, hu


class H1CarrierIdHuDataModule(H1M4EBNormalizedV2DataModule):
    """Source-only V2 DataModule with H-U at the carrier boundary."""

    def __init__(
        self,
        *,
        source_authority_dir: str,
        source_authority_receipt_sha256: str,
        **kwargs: Any,
    ) -> None:
        authority = Path(source_authority_dir).resolve()
        if authority.name != HU_SOURCE_AUTHORITY_DIRECTORY_NAME:
            raise NormalizedV2ContractError("H-U source authority directory name drift")
        if source_authority_receipt_sha256 != HU_SOURCE_AUTHORITY_RECEIPT_SHA256:
            raise NormalizedV2ContractError("H-U source authority receipt SHA drift")
        super().__init__(**kwargs)
        # The generic normalized-V2 path stays unchanged for all existing
        # H-C/H-C0 users.  Only H-U opts its isolated V1 source builder into
        # the already-frozen matched source authority.
        self._v1.hparams.source_authority_dir = str(authority)
        self._v1.hparams.source_authority_receipt_sha256 = source_authority_receipt_sha256
        self.hu_source_authority_dir = authority
        self.hu_source_authority_receipt_sha256 = source_authority_receipt_sha256
        self.carrier_intervention = "hu"

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit"}:
            raise RuntimeError("H-U DataModule permits source fit only")
        if self._setup_done:
            return
        super().setup(stage)
        original = self.train_dataset
        hc_normalizer_sha = str(self.normalizer.normalizer_sha256)
        hc_cache_sha = str(self.carrier_cache.manifest["cache_sha256"])
        hu_dataset = H1CarrierIdHuSourceDataset(self.records, self.carrier_cache, self.normalizer)
        hu_sampler = H1M4EBPairedBatchSampler(
            hu_dataset, batch_size=32, seed=42, max_epochs=50, cache_dir=None,
        )
        if (
            hu_dataset.window_indices_sha256 != original.window_indices_sha256
            or not np.array_equal(hu_sampler.schedule, self.train_batch_sampler.schedule)
            or hu_sampler.batch_order_sha256 != self.train_batch_sampler.batch_order_sha256
            or hu_sampler.schedule_sha256 != self.train_batch_sampler.schedule_sha256
        ):
            raise NormalizedV2ContractError("H-U changed the source window/order/schedule")
        if hc_cache_sha != "88261cc03532b605da1790e8669760d4d47e2f87d2db1428060445541638b0af":
            raise NormalizedV2ContractError(
                f"H-U rebuilt H-C source cache hash drifted: {hc_cache_sha}"
            )
        self.hu_normalizer = hu_dataset.hu_normalizer
        self.normalizer = hu_dataset.hu_normalizer
        manifest = dict(self._manifest)
        manifest.update(
            {
                "schema": "h1_carrierid_h32_hu_source_training_manifest_v1",
                "carrier_intervention": "hu",
                "hu_version": VERSION,
                "hu_feature_names": list(FEATURE_NAMES),
                "hu_declared_dead_channels": sorted(DECLARED_DEAD_CHANNELS),
                "hu_silent_channels_union": list(hu_dataset.hu_silent_channels),
                "hu_raw_sha256": hu_dataset.hu_raw_sha256,
                "hu_normalizer_sha256": hu_dataset.hu_normalizer.normalizer_sha256,
                "hu_s_src": float(hu_dataset.hu_normalizer.s_src),
                "hc_normalizer_sha256_before_hu_replace": hc_normalizer_sha,
                "source_carrier_intervention_only": True,
                "label_free": True,
                "used_principal_components": False,
                "normalizer_formula": NORMALIZER_FORMULA,
                "normalizer_floor": NORMALIZER_FLOOR,
                "per_dim_normalization": False,
                "effective_source_carriers_sha256": hu_dataset.effective_source_carriers_sha256,
                "effective_source_carriers_shape": hu_dataset.effective_source_carriers_shape,
                "effective_source_carriers_count": hu_dataset.effective_source_carriers_count,
                "effective_source_carriers_nonidentity_all": True,
                "target_nwb_opened_during_training_setup": False,
                "minival_or_heldout_enumerated": False,
                "source_authority": self._v1.pilot_manifest().get("source_authority"),
            }
        )
        authority = manifest["source_authority"]
        if (
            not isinstance(authority, Mapping)
            or authority.get("receipt_sha256") != HU_SOURCE_AUTHORITY_RECEIPT_SHA256
            or authority.get("carrier_cache_sha256")
            != "88261cc03532b605da1790e8669760d4d47e2f87d2db1428060445541638b0af"
            or authority.get("transform_sha256")
            != "92c932d059d93aaffae3d676fe247b46971232027fe906cb4b28add93ddee66a"
            or authority.get("target_nwb_opened") is not False
        ):
            raise NormalizedV2ContractError("H-U matched source authority binding drift")
        request = (0, int(hu_sampler.schedule[0, 0]))
        hc_carrier = original.normalized_carrier(request)
        hu_carrier = hu_dataset[request][4]
        if np.array_equal(hc_carrier, hu_carrier):
            raise NormalizedV2ContractError("H-U first source carrier is identical to H-C")
        manifest["first_source_carrier_audit"] = {
            "hc_normalized_sha256": array_sha256(hc_carrier),
            "hu_normalized_sha256": array_sha256(np.asarray(hu_carrier, dtype=np.float64)),
            "nonidentity": True,
        }
        self.train_dataset = hu_dataset
        self.train_batch_sampler = hu_sampler
        self._manifest = manifest
        self._manifest_sha256 = canonical_sha256(manifest)

    def test_dataloader(self):
        raise RuntimeError("H-U formal test loader is forbidden")

    def predict_dataloader(self):
        raise RuntimeError("H-U target prediction is evaluator-only")


def bind_hu_target_full_carrier(
    target: H1M4EBNormalizedV2StrictTargetDataset,
    records: Mapping[str, Any],
    hu_normalizer: Any,
) -> H1M4EBNormalizedV2StrictTargetDataset:
    """Replace the target 'full' carrier with source-normalized H-U.

    Query windows, identity interpolation, and the H-C row/label/zero views
    stay as constructed.  Evaluation of H-U uses intervention ``full`` after
    this replacement.
    """

    for name, support in tuple(target.support.items()):
        record = records[name]
        raw, audit = hu_from_record(record, support.trial_values)
        if audit.used_velocity or audit.used_behaviour_labels:
            raise NormalizedV2ContractError("H-U target adapter read behaviour")
        normalized = hu_normalizer.normalize(raw)
        carriers = dict(support.carriers)
        carriers["full"] = normalized
        hashes = {key: carrier_sha256(np.asarray(value)) for key, value in carriers.items()}
        target.support[name] = replace(support, carriers=carriers, carrier_sha256=hashes)
    target.normalized_support_sha256 = canonical_sha256(target.support_and_carrier_hashes())
    target.hu_bound = True
    return target
