"""Stage B source-side carrier corruption for separately trained Context-LS/RS arms.

Corruption is applied only after the sealed fold-0 snapshot manifest validates
against the clean rebuilt source assets.  Training therefore consumes the audited
clean map/cache binding and then one deterministic transform.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np

from sua_exploration.mc_maze import h1_event_carrier_design_screen as design
from sua_exploration.mc_maze import h1_sparse_event_endpoint as event_v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as event_v2
from src.data.h1_context_event_carrier import (
    CARRIER_DIM,
    ContextCarrierCache,
    ContextCarrierEntry,
    H1ContextEventDataModule,
    H1ContextSourceDataset,
    M4_BUDGET,
    PilotDataError,
    _fit,
    _normalizer,
    build_context_manifest,
)

CORRUPTION_MODES = ("row", "label")
CORRUPTION_SCHEMA = "h1_ctxv2_stage_b_source_corruption_v1"


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise PilotDataError(message)


def _rows_multiset(carrier: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(sorted(tuple(float(v) for v in row) for row in np.asarray(carrier, np.float64)))


def _block_record(
    *,
    session: str,
    start_index: int,
    clean_entry: ContextCarrierEntry,
    corrupted_entry: ContextCarrierEntry,
    permutation_sha256: str,
    fixed_points: int,
) -> dict[str, Any]:
    return {
        "session": session,
        "start_index": int(start_index),
        "clean_carrier_sha256": clean_entry.carrier_sha256,
        "corrupted_carrier_sha256": corrupted_entry.carrier_sha256,
        "permutation_sha256": permutation_sha256,
        "fixed_points": int(fixed_points),
    }


def build_corruption_manifest(mode: Literal["row", "label"], blocks: list[dict[str, Any]]) -> dict[str, Any]:
    fixed_points = [int(block["fixed_points"]) for block in blocks]
    body: dict[str, Any] = {
        "schema": CORRUPTION_SCHEMA,
        "mode": mode,
        "block_count": len(blocks),
        "blocks": blocks,
        "fixed_point_counts": {"total": int(sum(fixed_points)), "per_block": fixed_points},
        "per_block_permutation_sha256": [str(block["permutation_sha256"]) for block in blocks],
    }
    body["corruption_manifest_sha256"] = event_v1.canonical_sha256(body)
    return body


def corrupt_row_cache(clean_cache: ContextCarrierCache, latent_map: design.LatentMap) -> tuple[ContextCarrierCache, dict[str, Any]]:
    entries: list[ContextCarrierEntry] = []
    blocks: list[dict[str, Any]] = []
    for entry in clean_cache.entries:
        corrupted_arr, row_manifest = event_v2.row_shuffle(
            entry.carrier, session=entry.session_name, budget=M4_BUDGET,
        )
        fixed_points = int(np.sum([
            np.allclose(entry.carrier[index], corrupted_arr[index], rtol=0.0, atol=0.0)
            for index in range(corrupted_arr.shape[0])
        ]))
        _need(fixed_points == 0, f"row corruption retained fixed points at {entry.session_name}:{entry.start_index}")
        _need(
            _rows_multiset(entry.carrier) == _rows_multiset(corrupted_arr),
            f"row corruption must preserve row multiset at {entry.session_name}:{entry.start_index}",
        )
        _need(
            not np.array_equal(entry.carrier, corrupted_arr),
            f"row corruption collapsed at {entry.session_name}:{entry.start_index}",
        )
        corrupted_entry = ContextCarrierEntry(
            entry.session_name,
            entry.start_index,
            entry.trial_values,
            corrupted_arr,
            event_v1.array_sha256(corrupted_arr),
        )
        entries.append(corrupted_entry)
        blocks.append(_block_record(
            session=entry.session_name,
            start_index=entry.start_index,
            clean_entry=entry,
            corrupted_entry=corrupted_entry,
            permutation_sha256=str(row_manifest["order_sha256"]),
            fixed_points=int(row_manifest["fixed_points"]),
        ))
    cache = ContextCarrierCache(entries, latent_map)
    return cache, build_corruption_manifest("row", blocks)


def corrupt_label_cache(
    clean_cache: ContextCarrierCache,
    sessions: dict[str, design.ContextSession],
    latent_map: design.LatentMap,
) -> tuple[ContextCarrierCache, dict[str, Any]]:
    entries: list[ContextCarrierEntry] = []
    blocks: list[dict[str, Any]] = []
    for entry in clean_cache.entries:
        session = sessions[entry.session_name]
        support = design.select_range(session, start=entry.start_index, budget=M4_BUDGET)
        endpoint_order, shuffle_manifest = event_v1.within_trial_label_shuffle(
            tuple(event.base for event in support),
            session=entry.session_name,
            budget=M4_BUDGET,
        )
        corrupted_arr = _fit(
            tuple(support[int(index)] for index in endpoint_order),
            latent_map,
            response_events=support,
        )
        _need(
            not np.array_equal(entry.carrier, corrupted_arr),
            f"label corruption collapsed at {entry.session_name}:{entry.start_index}",
        )
        _need(
            _rows_multiset(entry.carrier) != _rows_multiset(corrupted_arr),
            f"label corruption must refit rather than permute rows at {entry.session_name}:{entry.start_index}",
        )
        corrupted_entry = ContextCarrierEntry(
            entry.session_name,
            entry.start_index,
            entry.trial_values,
            corrupted_arr,
            event_v1.array_sha256(corrupted_arr),
        )
        entries.append(corrupted_entry)
        blocks.append(_block_record(
            session=entry.session_name,
            start_index=entry.start_index,
            clean_entry=entry,
            corrupted_entry=corrupted_entry,
            permutation_sha256=str(shuffle_manifest["order_sha256"]),
            fixed_points=int(shuffle_manifest["fixed_points"]),
        ))
    cache = ContextCarrierCache(entries, latent_map)
    return cache, build_corruption_manifest("label", blocks)


def apply_source_corruption(
    *,
    mode: Literal["row", "label"],
    clean_cache: ContextCarrierCache,
    latent_map: design.LatentMap,
    sessions: dict[str, design.ContextSession] | None = None,
) -> tuple[ContextCarrierCache, dict[str, Any]]:
    _need(mode in CORRUPTION_MODES, f"unknown corruption mode {mode!r}")
    if mode == "row":
        return corrupt_row_cache(clean_cache, latent_map)
    _need(sessions is not None, "label corruption requires source sessions")
    return corrupt_label_cache(clean_cache, sessions, latent_map)


class H1ContextCorruptedSourceEventDataModule(H1ContextEventDataModule):
    """Fold-0 Context source training with post-validation carrier corruption."""

    def __init__(self, *, corruption_mode: str, **kwargs: Any) -> None:
        _need(corruption_mode in CORRUPTION_MODES, f"invalid corruption_mode {corruption_mode!r}")
        self._corruption_mode = str(corruption_mode)
        super().__init__(**kwargs)

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)
        clean_cache = self.carrier_cache
        clean_normalizer = self.normalizer
        clean_manifest_sha256 = self._manifest_sha256

        corrupted_cache, corruption_manifest = apply_source_corruption(
            mode=self._corruption_mode,  # type: ignore[arg-type]
            clean_cache=clean_cache,
            latent_map=self.latent_map,
            sessions=self.context_sessions,
        )
        corrupted_normalizer = _normalizer(corrupted_cache)

        self.carrier_cache = corrupted_cache
        self.normalizer = corrupted_normalizer
        self.train_dataset.cache = corrupted_cache
        self.train_dataset.normalizer = corrupted_normalizer

        manifest = build_context_manifest(
            records=self.records,
            latent_map=self.latent_map,
            cache=corrupted_cache,
            normalizer=corrupted_normalizer,
            dataset=self.train_dataset,
            sampler=self.train_batch_sampler,
        )
        manifest["clean_snapshot_validation"] = {
            "validated_clean_manifest_sha256": clean_manifest_sha256,
            "clean_carrier_cache_sha256": clean_cache.manifest["cache_sha256"],
            "clean_normalizer_sha256": clean_normalizer.normalizer_sha256,
            "source_snapshot_receipt": str(self.hparams.source_snapshot_receipt),
        }
        manifest["source_corruption"] = corruption_manifest
        manifest["source_snapshot"] = {
            "training_authority": "immutable_context_source_snapshot_then_stage_b_corruption",
            "corruption_mode": self._corruption_mode,
        }

        self._corruption_manifest = corruption_manifest
        self._manifest = manifest
        self._manifest_sha256 = event_v1.canonical_sha256(manifest)

    @property
    def corruption_manifest(self) -> dict[str, Any]:
        if not self._setup_done:
            raise RuntimeError("corrupted context DataModule not set up")
        return dict(self._corruption_manifest)
