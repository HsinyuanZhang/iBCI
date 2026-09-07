"""Route-owned source replay using the exact POOLED/G00m linear authority.

This module deliberately imports neither Torch nor PIT at import time.  The
future executor may call it only after its immutable attempt has been
published, after a single PIT source construction has supplied the DataModule
and source raw-session mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from . import plan
from .pools import PoolState, causal_pool_state, requested_pool_size


class SourceReplayError(RuntimeError):
    pass


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise SourceReplayError(message)


def _activity_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(f"{array.dtype}|{array.shape}".encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _carrier_hz_sha(value: np.ndarray) -> str:
    """Carrier audit digest preserving the reviewed float64 fit precision."""
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256(); digest.update(f"{array.dtype}|{array.shape}".encode("ascii")); digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceCoordinate:
    """One post-first30 supervised source window under an M30-ready state."""

    session: str
    window_start: int
    query_trial_index: int
    state_m30: PoolState
    selected_support4_carrier_hz_sha256: str
    normalized_side_sha256: str
    normalizer_sha256: str


@dataclass(frozen=True)
class SourceSessionMaterial:
    session: str
    support_indices: tuple[int, ...]
    selected_support4_carrier_hz: np.ndarray
    raw_m30_hz_audit: np.ndarray
    normalized_side: np.ndarray
    normalizer_sha256: str
    activities: np.ndarray
    query_rows: tuple[Mapping[str, Any], ...]
    coordinates: tuple[SourceCoordinate, ...]


class OrderedRawActivityPool:
    """Ordered raw G00m activity authority for FIXED30 or UNCAPPED replay.

    The pool preserves support rows permanently, appends only completed query
    rows, and materializes a raw `[M,100,N]` stack for the *same*
    `adapter.forward_batch` call used by APFG training.  It intentionally does
    not maintain a numerically rearranged sufficient statistic.
    """

    def __init__(self, *, support_trial_ids: Sequence[int], support_activities: Sequence[np.ndarray],
                 capacity: int | None) -> None:
        self.support_trial_ids = tuple(int(value) for value in support_trial_ids)
        self._support = tuple(np.ascontiguousarray(np.asarray(value, dtype=np.float32)) for value in support_activities)
        _require(len(self.support_trial_ids) == plan.SUPPORT_COUNT == len(self._support),
                 "APFG raw pool support cardinality drift")
        _require(len(set(self.support_trial_ids)) == len(self.support_trial_ids), "APFG raw pool duplicate support")
        shape = self._support[0].shape
        _require(len(shape) == 2 and shape[0] == plan.TRIAL_LENGTH and shape[1] > 0
                 and all(item.shape == shape for item in self._support), "APFG raw pool support shape drift")
        _require(capacity is None or int(capacity) == 30, "APFG only FIXED30 or UNCAPPED is authorized")
        self.capacity = capacity
        self._query: list[tuple[int, np.ndarray]] = []
        self.evictions = 0

    @property
    def member_trial_ids(self) -> tuple[int, ...]:
        return self.support_trial_ids + tuple(item[0] for item in self._query)

    @property
    def count(self) -> int:
        return len(self.member_trial_ids)

    def stack(self) -> np.ndarray:
        result = np.stack((*self._support, *(item[1] for item in self._query)), axis=0)
        result.setflags(write=False)
        return result

    def commit_completed(self, *, trial_id: int, activity: np.ndarray) -> None:
        identifier = int(trial_id)
        _require(identifier not in self.member_trial_ids, "APFG raw pool duplicate/current/future trial commit")
        value = np.ascontiguousarray(np.asarray(activity, dtype=np.float32))
        _require(value.shape == self._support[0].shape and np.isfinite(value).all(), "APFG raw pool activity shape/finite drift")
        self._query.append((identifier, value))
        if self.capacity is not None and self.count > self.capacity:
            _require(bool(self._query), "APFG FIXED30 would evict support")
            self._query.pop(0)
            self.evictions += 1
        _require(all(item in self.member_trial_ids for item in self.support_trial_ids), "APFG support was evicted")


def materialize_pooled_g00m_source_session(*, dataset: Any, raw_sessions: Mapping[str, Any],
                                            session: str) -> SourceSessionMaterial:
    """Build only source coordinates whose M30 pool is causally available.

    The historical G00m helper deliberately provides the linear raw activity
    reconstruction and selected-T4 side fit.  The PIT cubic calibration array
    is neither read nor accepted here.
    """
    from tfpd_exploration.src.cdm_p1_m2_local_v1 import replay as g_replay

    _require(session in raw_sessions, f"{session}: source raw-session authority absent")
    views = g_replay._g_session_views(raw_sessions, session=session)
    support = g_replay.g_support_material(session=session, views=views)
    query_rows = tuple(g_replay.g_query_rows(ds=dataset, session=session, views=views))
    selected = tuple(int(value) for value in np.asarray(support["selected"], dtype=np.int64))
    _require(len(selected) == plan.SUPPORT_COUNT and all(0 <= value < 30 for value in selected),
             f"{session}: G00m D-opt-k4 support drift")
    activities = np.ascontiguousarray(np.asarray(support["activities"], dtype=np.float32))
    _require(activities.ndim == 3 and activities.shape[0] >= 30, f"{session}: G00m activity topology drift")
    # ``m4_activity_only`` does *not* consume the receipt's full-M30 ridge.
    # Build its exact fixed-ridge carrier from the selected D-opt support-4.
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import physical as cdm_physical
    memory = cdm_physical._build_memory(system=cdm_physical.plan.SYSTEM_ACTIVITY, budget=4, selected=np.asarray(selected),
                                        support_rates_hz30=np.asarray(support["support_rates_hz30"]),
                                        theta_first30=np.asarray(support["theta30"]), support_b3s=support["support_b3s"],
                                        channel_ids=np.asarray(support["channels"]), valid_mask=np.asarray(support["valid_mask"]))
    _activity, carrier_hz = memory.prediction_inputs()
    carrier_hz = np.ascontiguousarray(np.asarray(carrier_hz, dtype=np.float64))
    _require(carrier_hz.ndim == 2 and carrier_hz.shape[1] == plan.SIDE_DIM and np.isfinite(carrier_hz).all(),
             f"{session}: selected support4 fixed-ridge carrier drift")
    raw_m30_audit = np.ascontiguousarray(np.asarray(support["raw_m30_hz"], dtype=np.float32))
    carrier_sha = _carrier_hz_sha(carrier_hz)
    mean = np.ascontiguousarray(np.asarray(dataset.side_feature_mean, dtype=np.float32))
    std = np.ascontiguousarray(np.asarray(dataset.side_feature_std, dtype=np.float32))
    _require(mean.shape == std.shape == (plan.SIDE_DIM,) and np.all(std > 0.0),
             f"{session}: source T4 normalizer drift")
    normalizer_sha = _activity_sha(np.stack((mean, std), axis=0))
    model_count = np.ascontiguousarray(carrier_hz * np.float64(0.020), dtype=np.float32)
    normalized_side = np.ascontiguousarray((model_count - mean) / std, dtype=np.float32)
    _require(np.isfinite(normalized_side).all(), f"{session}: normalized selected support4 side drift")
    normalized_side_sha = _activity_sha(normalized_side)
    coordinates: list[SourceCoordinate] = []
    for row in query_rows:
        position = int(row["position"])
        metric_starts = np.asarray(row["metric_starts"], dtype=np.int64)
        # The M30 controller needs 26 prior non-support trials beginning at
        # trial 30, hence position >= 56.  A current row is never committed
        # until after all its windows are decoded.
        if position < 30 + plan.NON_SUPPORT_COMPLETIONS[30]:
            continue
        completed = tuple(range(30, position))
        for start in metric_starts:
            endpoint = int(start) + plan.WINDOW_BINS - 1
            _require(int(row["padded_start"]) <= endpoint < int(row["padded_stop"]),
                     f"{session}: source query endpoint escaped its completed trial")
            state = causal_pool_state(
                session_id=session, requested_size=30, support_trial_ids=selected,
                completed_non_support_trial_ids=completed, query_trial_index=position,
                query_endpoint=endpoint,
                ordered_activity_sha256=tuple(_activity_sha(activities[index]) for index in (selected + completed[-26:])),
                selected_support4_carrier_hz_sha256=carrier_sha, normalized_side_sha256=normalized_side_sha,
                normalizer_sha256=normalizer_sha,
            )
            coordinates.append(SourceCoordinate(session=session, window_start=int(start), query_trial_index=position,
                                                state_m30=state, selected_support4_carrier_hz_sha256=carrier_sha,
                                                normalized_side_sha256=normalized_side_sha,
                                                normalizer_sha256=normalizer_sha))
    _require(bool(coordinates), f"{session}: no M30-causal source coordinates")
    return SourceSessionMaterial(session=session, support_indices=selected, selected_support4_carrier_hz=carrier_hz,
                                 raw_m30_hz_audit=raw_m30_audit,
                                 normalized_side=normalized_side, normalizer_sha256=normalizer_sha,
                                 activities=activities, query_rows=query_rows, coordinates=tuple(coordinates))


def canonical_m30_source_batches(material: SourceSessionMaterial, *, batch_size: int = plan.SOURCE_BATCH_MAX_MEMBERS
                                 ) -> tuple[tuple[SourceCoordinate, ...], ...]:
    """Partition lexical coordinates by exact M30 identity state, never pad.

    The controller ordinal must never depend on a SHA lexical ordering: source
    exposure is frozen as ``(session, query_trial, window_start)``.  A query
    trial has exactly one M30-ready identity, so chunking this ordered stream
    cannot join different identity states.
    """
    _require(batch_size == plan.SOURCE_BATCH_MAX_MEMBERS, "APFG source batch-size law drift")
    ordered = sorted(material.coordinates, key=lambda item: (item.session, item.query_trial_index, item.window_start))
    batches: list[tuple[SourceCoordinate, ...]] = []
    offset = 0
    while offset < len(ordered):
        anchor = ordered[offset]
        stop = offset + 1
        while (stop < len(ordered) and ordered[stop].session == anchor.session
               and ordered[stop].query_trial_index == anchor.query_trial_index
               and ordered[stop].state_m30.identity_digest() == anchor.state_m30.identity_digest()):
            stop += 1
        values = ordered[offset:stop]
        for chunk_offset in range(0, len(values), batch_size):
            chunk = tuple(values[chunk_offset: chunk_offset + batch_size])
            _require(chunk and len(chunk) <= batch_size, "APFG canonical source batch partition drift")
            _require(all(value.session == anchor.session and value.query_trial_index == anchor.query_trial_index
                         and value.state_m30.identity_digest() == anchor.state_m30.identity_digest() for value in chunk),
                     "APFG batch crossed session/query/pool state")
            batches.append(chunk)
        offset = stop
    return tuple(batches)


def pool_for_controller_coordinate(*, material: SourceSessionMaterial, coordinate: SourceCoordinate,
                                  epoch_one_indexed: int, canonical_batch_ordinal: int) -> PoolState:
    """Derive the current 4/10/30 state from its already M30-eligible source row."""
    requested = requested_pool_size(epoch_one_indexed, canonical_batch_ordinal)
    completed = tuple(range(30, coordinate.query_trial_index))
    needed = plan.NON_SUPPORT_COMPLETIONS[requested]
    members = material.support_indices + (completed[-needed:] if needed else ())
    return causal_pool_state(
        session_id=coordinate.session, requested_size=requested, support_trial_ids=material.support_indices,
        completed_non_support_trial_ids=completed, query_trial_index=coordinate.query_trial_index,
        query_endpoint=coordinate.state_m30.query_endpoint,
        ordered_activity_sha256=tuple(_activity_sha(material.activities[index]) for index in members),
        selected_support4_carrier_hz_sha256=coordinate.selected_support4_carrier_hz_sha256,
        normalized_side_sha256=coordinate.normalized_side_sha256,
        normalizer_sha256=coordinate.normalizer_sha256,
    )


def source_authority_summary(materials: Sequence[SourceSessionMaterial]) -> dict[str, object]:
    """Receipt-safe source authority; activity tensors never escape this body."""
    _require(len(materials) == plan.SOURCE_SESSION_COUNT, "APFG source roster must contain exactly seven sessions")
    sessions = tuple(sorted(item.session for item in materials))
    _require(len(set(sessions)) == len(sessions), "APFG source material duplicates session")
    ordered_coordinates = [(item.session, coordinate.query_trial_index, coordinate.window_start,
                            coordinate.state_m30.digest())
                           for item in sorted(materials, key=lambda item: item.session)
                           for coordinate in canonical_m30_source_batches(item)
                           for coordinate in coordinate]
    ordered_batches = [(item.session, [coordinate.state_m30.digest() for coordinate in batch],
                        [coordinate.window_start for coordinate in batch])
                       for item in sorted(materials, key=lambda item: item.session)
                       for batch in canonical_m30_source_batches(item)]
    digest = lambda value: hashlib.sha256(json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    return {"activity_authority": plan.ACTIVITY_AUTHORITY, "sessions": list(sessions),
            "m30_causal_coordinate_count": int(sum(len(item.coordinates) for item in materials)),
            "ordered_coordinate_sha256": digest(ordered_coordinates),
            "ordered_batch_sha256": digest(ordered_batches),
            "session": {
                item.session: {"support_indices": list(item.support_indices),
                               "selected_support4_carrier_hz_float64_sha256": _carrier_hz_sha(item.selected_support4_carrier_hz),
                               "selected_support4_carrier_model_count_per_bin_float32_sha256": _activity_sha(
                                   np.ascontiguousarray(item.selected_support4_carrier_hz * np.float64(0.020), dtype=np.float32)),
                               "raw_m30_hz_audit_sha256": _activity_sha(item.raw_m30_hz_audit),
                               "normalized_side_sha256": _activity_sha(item.normalized_side),
                               "normalizer_sha256": item.normalizer_sha256,
                               "m30_causal_coordinate_count": len(item.coordinates)}
                for item in sorted(materials, key=lambda item: item.session)
            }}


def material_from_g00m_runtime_record(record: Mapping[str, Any]) -> SourceSessionMaterial:
    """Adapt one already materialized locked scorer input without re-reading data."""
    runtime = record.get("_runtime")
    _require(isinstance(runtime, Mapping), "APFG target record runtime authority missing")
    support = runtime.get("support")
    dataset = runtime.get("dataset")
    rows = tuple(runtime.get("query_rows", ()))
    _require(isinstance(support, Mapping) and dataset is not None and rows, "APFG target G00m runtime components missing")
    selected = tuple(int(value) for value in np.asarray(support["selected"], dtype=np.int64))
    activities = np.ascontiguousarray(np.asarray(support["activities"], dtype=np.float32))
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import physical as cdm_physical
    memory = cdm_physical._build_memory(system=cdm_physical.plan.SYSTEM_ACTIVITY, budget=4, selected=np.asarray(selected),
                                        support_rates_hz30=np.asarray(support["support_rates_hz30"]), theta_first30=np.asarray(support["theta30"]),
                                        support_b3s=support["support_b3s"], channel_ids=np.asarray(support["channels"]), valid_mask=np.asarray(support["valid_mask"]))
    _activity, carrier = memory.prediction_inputs(); carrier = np.ascontiguousarray(np.asarray(carrier, dtype=np.float64))
    raw_m30_audit = np.ascontiguousarray(np.asarray(support["raw_m30_hz"], dtype=np.float32))
    mean = np.asarray(dataset.side_feature_mean, dtype=np.float32); std = np.asarray(dataset.side_feature_std, dtype=np.float32)
    normalizer = _activity_sha(np.stack((mean, std), axis=0))
    side = np.ascontiguousarray((np.ascontiguousarray(carrier * np.float64(0.020), dtype=np.float32) - mean) / std, dtype=np.float32)
    return SourceSessionMaterial(session=str(record["session"]), support_indices=selected, selected_support4_carrier_hz=carrier,
                                 raw_m30_hz_audit=raw_m30_audit, normalized_side=side, normalizer_sha256=normalizer, activities=activities,
                                 query_rows=rows, coordinates=())


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core as pseudo_core
    return float(pseudo_core.variance_weighted_r2(target, prediction))


def rollout_apfg_raw_pool(*, torch: Any, model: Any, material: SourceSessionMaterial, dataset: Any,
                           law: str, device: Any, batch_size: int,
                           branch_cache: dict[str, tuple[Any, Any]] | None = None) -> dict[str, object]:
    """One decode-before-commit G00m-linear raw-pool rollout for APFG.

    This is deliberately independent of the old post-fusion identity pools:
    APFG needs both native pre-fusion and post-fusion identities from the same
    ordered raw activity stack.  At zero gate, the adapter's direct native
    branch preserves the selected-POOLED B3S arithmetic.
    """
    _require(law in ("FIXED30", "UNCAPPED"), "APFG replay law drift")
    _require(not model.training, "APFG replay model must remain eval")
    adapter = model.student.id_encoder
    _require(getattr(adapter, "variant", None) == "B3S_APFG", "APFG replay adapter missing")
    support = [material.activities[index] for index in material.support_indices]
    pool = OrderedRawActivityPool(support_trial_ids=material.support_indices, support_activities=support,
                                  capacity=30 if law == "FIXED30" else None)
    neural = np.asarray(dataset.neural_data[material.session], dtype=np.float32)
    behavior = np.asarray(dataset.covariate_data[material.session], dtype=np.float32)
    side = torch.from_numpy(material.normalized_side).unsqueeze(0).to(device)
    prediction_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    start_parts: list[np.ndarray] = []
    trace: list[dict[str, object]] = []
    cache_hits = 0; cache_misses = 0
    for row in material.query_rows:
        metric_starts = np.asarray(row["metric_starts"], dtype=np.int64)
        if metric_starts.size:
            indices = metric_starts[:, None] + np.arange(plan.WINDOW_BINS, dtype=np.int64)[None, :]
            windows = np.ascontiguousarray(neural[indices], dtype=np.float32)
            pieces: list[np.ndarray] = []
            with torch.inference_mode():
                calibration = torch.from_numpy(np.ascontiguousarray(pool.stack())).unsqueeze(0).to(device)
                member_digest = hashlib.sha256(repr((material.session, law, pool.member_trial_ids,
                                                      _activity_sha(pool.stack()), material.normalizer_sha256)).encode("utf-8")).hexdigest()
                # The +0 sentinel must call native arithmetic exactly.  A
                # learned alpha may reuse immutable pre/post branches because
                # alpha is the only mutable parameter.
                exact_zero = bool(getattr(adapter, "_alpha_training_enabled", False) is False
                                  and float(adapter.alpha.detach().cpu()) == 0.0
                                  and not bool(torch.signbit(adapter.alpha.detach()).item()))
                if exact_zero:
                    identity = adapter.native.forward_batch(calibration, side_features=side)
                else:
                    cached = branch_cache.get(member_digest) if branch_cache is not None else None
                    if cached is None:
                        native = adapter.native.forward_batch(calibration, side_features=side).detach()
                        post = adapter._postfusion_identity(calibration, side).detach()
                        cached = (native, post)
                        cache_misses += 1
                        if branch_cache is not None:
                            branch_cache[member_digest] = cached
                    else:
                        cache_hits += 1
                    identity = cached[0] + torch.tanh(adapter.alpha) * (cached[1] - cached[0])
                for offset in range(0, windows.shape[0], int(batch_size)):
                    item = torch.from_numpy(windows[offset: offset + int(batch_size)]).to(device)
                    output = model.student.decode_with_identity(item, identity)
                    _require(bool(torch.isfinite(output).all()), "APFG raw-pool decode nonfinite")
                    pieces.append(output[:, -1, :].detach().cpu().numpy().astype(np.float32, copy=False) / 5.0)
            prediction_parts.append(np.ascontiguousarray(np.concatenate(pieces), dtype=np.float32))
            target_parts.append(np.ascontiguousarray(behavior[metric_starts + plan.WINDOW_BINS - 1], dtype=np.float32))
            start_parts.append(metric_starts)
        before = pool.count
        pool.commit_completed(trial_id=int(row["position"]), activity=np.asarray(row["activity"], dtype=np.float32))
        trace.append({"trial_id": str(row["trial_id"]), "position": int(row["position"]),
                      "members_before": before, "members_after": pool.count, "evictions": pool.evictions,
                      "activity_sha256": _activity_sha(np.asarray(row["activity"], dtype=np.float32))})
    prediction = np.ascontiguousarray(np.concatenate(prediction_parts), dtype=np.float32)
    target = np.ascontiguousarray(np.concatenate(target_parts), dtype=np.float32)
    starts = np.ascontiguousarray(np.concatenate(start_parts), dtype=np.int64)
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core as pseudo_core
    return {"session": material.session, "memory_law": law, "activity_authority": plan.ACTIVITY_AUTHORITY,
            "prediction": prediction, "target": target, "starts": starts, "r2": _r2(target, prediction),
            "window_count": int(starts.size), "prediction_sha256": _activity_sha(prediction),
            "target_sha256": _activity_sha(target), "query_starts_sha256": _activity_sha(starts),
            "governed_prediction_sha256": str(pseudo_core.array_sha256(prediction)),
            "governed_target_sha256": str(pseudo_core.array_sha256(target)),
            "governed_query_starts_sha256": str(pseudo_core.array_sha256(starts)),
            "commits": len(trace), "evictions": pool.evictions, "final_members": pool.count,
            "causal_trace_sha256": hashlib.sha256(repr(trace).encode("utf-8")).hexdigest(),
            "branch_cache_hits": cache_hits, "branch_cache_misses": cache_misses}


__all__ = ("SourceReplayError", "SourceCoordinate", "SourceSessionMaterial",
           "OrderedRawActivityPool",
           "materialize_pooled_g00m_source_session", "canonical_m30_source_batches",
           "pool_for_controller_coordinate", "source_authority_summary", "material_from_g00m_runtime_record",
           "rollout_apfg_raw_pool")
