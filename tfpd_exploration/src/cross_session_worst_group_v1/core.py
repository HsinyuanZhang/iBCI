"""CPU/synthetic-only CS-WG episode and complete-objective primitives.

This module contains no M1 model, parameter, inference fork, checkpoint, data
loader, root, or CUDA policy.  A future route-owned trainer may pass the exact
already-materialized M1 graph to :class:`RouteOwnedMixedSessionTrainingStep`,
but this package never defines or mutates that graph.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from . import plan


class CSWGCoreError(RuntimeError):
    """Fail closed for source-only strata, episode, or objective drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CSWGCoreError(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_digest(value: Any) -> str:
    """Stable finite-array digest used for source-only authority evidence."""
    array = np.ascontiguousarray(np.asarray(value))
    _require(array.ndim >= 1 and np.isfinite(array).all(), "CS-WG array digest needs a finite non-scalar array")
    header = _json_bytes({"dtype": str(array.dtype), "shape": list(array.shape)})
    return _sha(header + array.tobytes())


def _stable_index(*parts: object, modulo: int) -> int:
    _require(type(modulo) is int and modulo > 0, "CS-WG deterministic index modulo drift")
    return int.from_bytes(hashlib.sha256(_json_bytes(list(parts))).digest()[:8], "big") % modulo


@dataclass(frozen=True, order=True)
class TaskStratum:
    """One deterministic source-only final-bin task stratum."""

    active_flag: bool
    raw_output_norm_quantile: int
    dominant_coordinate: int

    def __post_init__(self) -> None:
        _require(type(self.active_flag) is bool
                 and type(self.raw_output_norm_quantile) is int
                 and 0 <= self.raw_output_norm_quantile < plan.NORM_QUANTILE_COUNT
                 and type(self.dominant_coordinate) is int
                 and 0 <= self.dominant_coordinate < plan.M1_RAW_BEHAVIOR_OUTPUTS,
                 "CS-WG task-stratum topology drift")

    def payload(self) -> dict[str, object]:
        return {
            "active_flag": self.active_flag,
            "raw_output_norm_quantile": self.raw_output_norm_quantile,
            "dominant_coordinate": self.dominant_coordinate,
        }


@dataclass(frozen=True)
class SourceOnlyFinalBinLabels:
    """Typed raw final-bin labels that may be used only for source strata."""

    session_id: str
    raw_final_outputs: Any = field(repr=False, compare=False)
    valid_final_bin: Any = field(repr=False, compare=False)
    provenance: str = "SOURCE_TRAINING_FINAL_BIN_LABELS_ONLY"

    def __post_init__(self) -> None:
        output = np.ascontiguousarray(np.asarray(self.raw_final_outputs, dtype=np.float32))
        valid = np.ascontiguousarray(np.asarray(self.valid_final_bin, dtype=np.bool_))
        _require(
            isinstance(self.session_id, str) and self.session_id in plan.HELD_IN_SOURCE_SESSIONS
            and self.provenance == "SOURCE_TRAINING_FINAL_BIN_LABELS_ONLY"
            and output.ndim == 2 and output.shape[1] == plan.M1_RAW_BEHAVIOR_OUTPUTS
            and valid.shape == (output.shape[0],) and output.shape[0] > 0
            and np.isfinite(output).all() and bool(valid.any()),
            "CS-WG source-only final-bin label capability drift",
        )
        output.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "raw_final_outputs", output)
        object.__setattr__(self, "valid_final_bin", valid)

    @property
    def digest(self) -> str:
        return _sha(_json_bytes({
            "session": self.session_id,
            "provenance": self.provenance,
            "raw_final_outputs": array_digest(self.raw_final_outputs),
            "valid_final_bin": array_digest(self.valid_final_bin),
        }))

    @property
    def valid_indices(self) -> np.ndarray:
        return np.flatnonzero(self.valid_final_bin).astype(np.int64, copy=False)


class _SourceAuthoritySeal:
    pass


_SOURCE_AUTHORITY_SEAL = _SourceAuthoritySeal()


@dataclass(frozen=True)
class SourceStratumAuthority:
    """Frozen quantile fit derived exclusively from typed source labels."""

    source_sessions: tuple[str, ...]
    norm_quantile_count: int
    raw_output_norm_boundaries: tuple[float, ...]
    source_label_digests: Mapping[str, str]
    fit_method: str = "pooled_source_valid_final_bin_l2_norm_numpy_quantile_linear"
    active_rule: str = "raw_output_l2_norm_gt_0"
    dominant_coordinate_rule: str = "argmax_abs_raw_output_lowest_index_tiebreak"
    _seal: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        sessions = tuple(self.source_sessions)
        digests = dict(self.source_label_digests)
        _require(self._seal is _SOURCE_AUTHORITY_SEAL
                 and 2 <= len(sessions) <= 4
                 and len(set(sessions)) == len(sessions)
                 and all(session in plan.HELD_IN_SOURCE_SESSIONS for session in sessions)
                 and type(self.norm_quantile_count) is int
                 and self.norm_quantile_count == plan.NORM_QUANTILE_COUNT
                 and len(self.raw_output_norm_boundaries) == self.norm_quantile_count - 1
                 and all(isinstance(value, float) and math.isfinite(value) and value >= 0.0
                         for value in self.raw_output_norm_boundaries)
                 and tuple(digests) == sessions
                 and all(isinstance(value, str) and len(value) == 64 for value in digests.values())
                 and self.fit_method == "pooled_source_valid_final_bin_l2_norm_numpy_quantile_linear"
                 and self.active_rule == "raw_output_l2_norm_gt_0"
                 and self.dominant_coordinate_rule == "argmax_abs_raw_output_lowest_index_tiebreak",
                 "CS-WG source-only stratum authority drift")
        _require(tuple(sorted(self.raw_output_norm_boundaries)) == self.raw_output_norm_boundaries,
                 "CS-WG quantile boundary order drift")
        object.__setattr__(self, "source_sessions", sessions)
        object.__setattr__(self, "source_label_digests", MappingProxyType(digests))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_source_stratum_authority_v1",
            "source_sessions": list(self.source_sessions),
            "norm_quantile_count": self.norm_quantile_count,
            "raw_output_norm_boundaries": list(self.raw_output_norm_boundaries),
            "source_label_digests": dict(self.source_label_digests),
            "fit_method": self.fit_method,
            "active_rule": self.active_rule,
            "dominant_coordinate_rule": self.dominant_coordinate_rule,
            "source_only": True,
            "target_labels_used": False,
            "normalizer_or_quantile_refit_on_target": False,
        }

    @property
    def sha256(self) -> str:
        return _sha(_json_bytes(self.payload()))


def fit_source_stratum_authority(
    source_labels: Mapping[str, SourceOnlyFinalBinLabels],
    *,
    source_sessions: Sequence[str],
    norm_quantile_count: int = plan.NORM_QUANTILE_COUNT,
) -> SourceStratumAuthority:
    """Fit frozen norm boundaries from the exact source-training label union.

    The function intentionally has no target/caller-boundary parameter.  A
    raw ndarray, a target-shaped object, a missing source session, or a
    source-label digest substitution cannot enter this typed authority.
    """
    sessions = tuple(source_sessions)
    _require(2 <= len(sessions) <= 4 and len(set(sessions)) == len(sessions)
             and all(session in plan.HELD_IN_SOURCE_SESSIONS for session in sessions),
             "CS-WG source authority source-session topology drift")
    _require(type(norm_quantile_count) is int and norm_quantile_count == plan.NORM_QUANTILE_COUNT,
             "CS-WG norm quantile count is not the frozen Stage-0 literal")
    _require(tuple(source_labels) == sessions, "CS-WG source label mapping order/session drift")
    tables = tuple(source_labels[session] for session in sessions)
    _require(all(isinstance(table, SourceOnlyFinalBinLabels) and table.session_id == session
                 for session, table in zip(sessions, tables, strict=True)),
             "CS-WG rejects target, raw, or caller-supplied label capabilities")
    norms: list[np.ndarray] = []
    for table in tables:
        values = np.asarray(table.raw_final_outputs[table.valid_final_bin], dtype=np.float64)
        _require(values.ndim == 2 and values.shape[0] > 0 and np.isfinite(values).all(),
                 "CS-WG source valid final-bin labels drift")
        norms.append(np.linalg.norm(values, axis=1))
    pooled = np.ascontiguousarray(np.concatenate(norms, axis=0), dtype=np.float64)
    _require(pooled.size > 0 and np.isfinite(pooled).all(), "CS-WG source quantile pool drift")
    boundaries = np.quantile(
        pooled,
        np.arange(1, norm_quantile_count, dtype=np.float64) / float(norm_quantile_count),
        method="linear",
    )
    return SourceStratumAuthority(
        source_sessions=sessions,
        norm_quantile_count=norm_quantile_count,
        raw_output_norm_boundaries=tuple(float(value) for value in boundaries.tolist()),
        source_label_digests={session: table.digest for session, table in zip(sessions, tables, strict=True)},
        _seal=_SOURCE_AUTHORITY_SEAL,
    )


def fit_run_spec_source_stratum_authority(
    run_spec: plan.CSWGRunSpec,
    source_labels: Mapping[str, SourceOnlyFinalBinLabels],
) -> SourceStratumAuthority:
    """Fit strata only from the exact source side of one typed run spec.

    ``SourceOnlyFinalBinLabels`` intentionally names a physical held-in M1
    session rather than guessing whether that session is an outer target.  The
    run-spec binding is therefore mandatory in a later physical route: it is
    what prevents a held-in outer target's labels from being silently folded
    into a three-session quantile fit.
    """
    _require(isinstance(run_spec, plan.CSWGRunSpec)
             and tuple(source_labels) == run_spec.source_sessions
             and (run_spec.outer_target_session is None
                  or run_spec.outer_target_session not in source_labels),
             "CS-WG target/caller labels cannot enter a run-spec source stratum authority")
    authority = fit_source_stratum_authority(
        source_labels,
        source_sessions=run_spec.source_sessions,
        norm_quantile_count=run_spec.norm_quantile_count,
    )
    _require(authority.source_sessions == run_spec.source_sessions,
             "CS-WG run-spec/source authority session binding drift")
    return authority


@dataclass(frozen=True)
class AssignedSourceStrata:
    """All and only valid source final-bin targets with frozen stratum IDs."""

    session_id: str
    authority_sha256: str
    sample_indices: tuple[int, ...]
    strata: tuple[TaskStratum, ...]

    def __post_init__(self) -> None:
        _require(isinstance(self.session_id, str) and len(self.authority_sha256) == 64
                 and len(self.sample_indices) == len(self.strata) > 0
                 and all(type(index) is int and index >= 0 for index in self.sample_indices)
                 and len(set(self.sample_indices)) == len(self.sample_indices)
                 and all(isinstance(item, TaskStratum) for item in self.strata),
                 "CS-WG assigned source stratum topology drift")

    def payload(self) -> dict[str, object]:
        return {
            "session": self.session_id,
            "authority_sha256": self.authority_sha256,
            "sample_indices": list(self.sample_indices),
            "strata": [item.payload() for item in self.strata],
            "source_only": True,
        }


def assign_source_task_strata(
    labels: SourceOnlyFinalBinLabels, authority: SourceStratumAuthority,
) -> AssignedSourceStrata:
    _require(isinstance(labels, SourceOnlyFinalBinLabels) and isinstance(authority, SourceStratumAuthority)
             and labels.session_id in authority.source_sessions
             and authority.source_label_digests.get(labels.session_id) == labels.digest,
             "CS-WG task-stratum assignment requires exact frozen source authority")
    indices = labels.valid_indices
    values = np.asarray(labels.raw_final_outputs[indices], dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    boundaries = np.asarray(authority.raw_output_norm_boundaries, dtype=np.float64)
    quantiles = np.minimum(
        np.searchsorted(boundaries, norms, side="right"), authority.norm_quantile_count - 1,
    ).astype(np.int64, copy=False)
    dominant = np.argmax(np.abs(values), axis=1).astype(np.int64, copy=False)
    active = norms > 0.0
    strata = tuple(
        TaskStratum(bool(flag), int(quantile), int(coordinate))
        for flag, quantile, coordinate in zip(active, quantiles, dominant, strict=True)
    )
    return AssignedSourceStrata(labels.session_id, authority.sha256,
                                tuple(int(item) for item in indices.tolist()), strata)


@dataclass(frozen=True)
class SourceEpisodeRow:
    """One source-only row with the exact frozen Falcon M1 forward inputs.

    The future route-owned training step calls
    ``FalconLitModule.forward(x=..., calib_trialized_neural_features=...)``.
    It must never translate these values into an invented ``neural``/
    ``side_features`` API or share one calibration tensor across mixed-session
    rows.
    """

    session_id: str
    sample_index: int
    sample_id: str
    stratum: TaskStratum
    model_inputs: Mapping[str, Any] = field(repr=False, compare=False)
    raw_final_target: Any = field(repr=False, compare=False)
    final_bin_valid: bool = True
    source_only: bool = True

    def __post_init__(self) -> None:
        target = np.ascontiguousarray(np.asarray(self.raw_final_target, dtype=np.float32))
        inputs = dict(self.model_inputs) if isinstance(self.model_inputs, Mapping) else {}
        exact_input_keys = tuple(sorted(inputs)) == tuple(sorted(plan.M1_FORWARD_INPUT_KEYS))
        if exact_input_keys:
            x = np.ascontiguousarray(np.asarray(inputs["x"], dtype=np.float32))
            calibration = np.ascontiguousarray(
                np.asarray(inputs["calib_trialized_neural_features"], dtype=np.float32),
            )
        else:
            # Keep malformed legacy/caller mappings inside the typed
            # fail-closed error surface rather than accidentally coercing a
            # missing value through NumPy.
            x = np.empty((0,), dtype=np.float32)
            calibration = np.empty((0,), dtype=np.float32)
        _require(
            isinstance(self.session_id, str) and self.session_id in plan.HELD_IN_SOURCE_SESSIONS
            and type(self.sample_index) is int and self.sample_index >= 0
            and isinstance(self.sample_id, str) and self.sample_id
            and isinstance(self.stratum, TaskStratum)
            and isinstance(self.model_inputs, Mapping)
            and exact_input_keys
            and x.shape == (plan.M1_WINDOW_SIZE, plan.M1_UNIT_COUNT)
            and calibration.shape == plan.M1_CALIBRATION_SHAPE_PER_ROW
            and np.isfinite(x).all() and np.isfinite(calibration).all()
            and target.shape == (plan.M1_RAW_BEHAVIOR_OUTPUTS,) and np.isfinite(target).all()
            and self.final_bin_valid is True and self.source_only is True,
            "CS-WG source episode row exact Falcon-M1 source/final-bin/input authority drift",
        )
        x.setflags(write=False)
        calibration.setflags(write=False)
        target.setflags(write=False)
        object.__setattr__(self, "raw_final_target", target)
        object.__setattr__(
            self,
            "model_inputs",
            MappingProxyType({"x": x, "calib_trialized_neural_features": calibration}),
        )

    def input_digests(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, value in self.model_inputs.items():
            _require(isinstance(key, str) and key, "CS-WG model input key drift")
            result[key] = array_digest(value)
        return result


@dataclass(frozen=True)
class SessionStratumPool:
    """One source session's exact typed rows, indexed by frozen strata."""

    session_id: str
    authority_sha256: str
    rows: tuple[SourceEpisodeRow, ...]

    def __post_init__(self) -> None:
        _require(isinstance(self.session_id, str) and len(self.authority_sha256) == 64
                 and len(self.rows) > 0 and all(isinstance(row, SourceEpisodeRow)
                                                  and row.session_id == self.session_id
                                                  for row in self.rows)
                 and len({row.sample_id for row in self.rows}) == len(self.rows),
                 "CS-WG session stratum pool topology drift")

    @property
    def strata(self) -> tuple[TaskStratum, ...]:
        return tuple(sorted({row.stratum for row in self.rows}))

    def candidates(self, stratum: TaskStratum) -> tuple[SourceEpisodeRow, ...]:
        return tuple(row for row in self.rows if row.stratum == stratum)


def build_session_stratum_pool(
    assignment: AssignedSourceStrata,
    *,
    rows_by_sample_index: Mapping[int, SourceEpisodeRow],
) -> SessionStratumPool:
    _require(isinstance(assignment, AssignedSourceStrata)
             and tuple(rows_by_sample_index) == assignment.sample_indices,
             "CS-WG episode rows must be supplied in assigned valid-source order")
    rows: list[SourceEpisodeRow] = []
    for sample_index, stratum in zip(assignment.sample_indices, assignment.strata, strict=True):
        row = rows_by_sample_index[sample_index]
        _require(isinstance(row, SourceEpisodeRow)
                 and row.session_id == assignment.session_id
                 and row.sample_index == sample_index
                 and row.stratum == stratum,
                 "CS-WG episode row/assigned stratum drift")
        rows.append(row)
    return SessionStratumPool(assignment.session_id, assignment.authority_sha256, tuple(rows))


@dataclass(frozen=True)
class EpisodeMicrobatch:
    session_id: str
    rows: tuple[SourceEpisodeRow, ...]

    def __post_init__(self) -> None:
        _require(isinstance(self.session_id, str) and len(self.rows) > 0
                 and all(row.session_id == self.session_id and row.final_bin_valid and row.source_only
                         for row in self.rows)
                 and len({row.sample_id for row in self.rows}) == len(self.rows),
                 "CS-WG episode microbatch source/final-bin/duplicate drift")


@dataclass(frozen=True)
class BalancedEpisode:
    """A total-B32 multi-session episode with equalized represented strata."""

    quota: plan.EpisodeQuota
    microbatches: tuple[EpisodeMicrobatch, ...]
    represented_strata: tuple[TaskStratum, ...]

    def __post_init__(self) -> None:
        _require(len(self.microbatches) == len(self.quota.sessions) >= 2
                 and tuple(batch.session_id for batch in self.microbatches) == self.quota.sessions
                 and tuple(len(batch.rows) for batch in self.microbatches) == self.quota.counts
                 and sum(len(batch.rows) for batch in self.microbatches) == plan.TOTAL_BATCH_SIZE
                 and len(self.represented_strata) > 0
                 and tuple(sorted(self.represented_strata)) == self.represented_strata,
                 "CS-WG balanced episode total-B32/session topology drift")
        represented = set(self.represented_strata)
        for microbatch in self.microbatches:
            counts = [sum(row.stratum == stratum for row in microbatch.rows) for stratum in self.represented_strata]
            _require(all(value > 0 for value in counts) and max(counts) - min(counts) <= 1
                     and {row.stratum for row in microbatch.rows}.issubset(represented),
                     "CS-WG equal represented task-stratum weighting drift")

    @property
    def session_ids(self) -> tuple[str, ...]:
        return self.quota.sessions

    @property
    def all_rows(self) -> tuple[SourceEpisodeRow, ...]:
        return tuple(row for batch in self.microbatches for row in batch.rows)

    @property
    def digest(self) -> str:
        return _sha(_json_bytes({
            "quota": self.quota.payload(),
            "represented_strata": [item.payload() for item in self.represented_strata],
            "rows": [
                {"session": row.session_id, "sample_index": row.sample_index,
                 "sample_id": row.sample_id, "stratum": row.stratum.payload(),
                 "input_digests": row.input_digests(), "target": array_digest(row.raw_final_target)}
                for row in self.all_rows
            ],
        }))


def build_balanced_episode(
    pools: Mapping[str, SessionStratumPool], *, quota: plan.EpisodeQuota,
) -> BalancedEpisode:
    """Deterministically balance represented source task strata without global RNG.

    Every participating session must expose the same stratum set.  For a
    microbatch smaller than the full common set, a deterministic rotating
    subset is represented; every represented stratum appears equally (or
    differs by one only when the frozen 10/11 quota cannot divide it).
    """
    _require(tuple(pools) == quota.sessions, "CS-WG pool order must exactly follow quota session order")
    selected = tuple(pools[session] for session in quota.sessions)
    _require(all(isinstance(pool, SessionStratumPool) for pool in selected)
             and len({pool.authority_sha256 for pool in selected}) == 1,
             "CS-WG episode pools need one exact source stratum authority")
    common = selected[0].strata
    _require(common and all(pool.strata == common for pool in selected[1:]),
             "CS-WG missing task stratum has no declared deterministic fallback")
    represented_count = min(min(quota.counts), len(common))
    _require(represented_count > 0, "CS-WG episode has no representable source task stratum")
    start = quota.step_index % len(common)
    represented = tuple(common[(start + offset) % len(common)] for offset in range(represented_count))
    represented = tuple(sorted(represented))
    batches: list[EpisodeMicrobatch] = []
    for session, count, pool in zip(quota.sessions, quota.counts, selected, strict=True):
        base, remainder = divmod(count, represented_count)
        # Rotating the one-extra allocations is deterministic and local; no
        # session gets a persistent stratum privilege merely because of order.
        extra_start = _stable_index("CSWG-extra", quota.step_index, session, modulo=represented_count)
        rows: list[SourceEpisodeRow] = []
        for position, stratum in enumerate(represented):
            need = base + int((position - extra_start) % represented_count < remainder)
            candidates = tuple(sorted(pool.candidates(stratum), key=lambda row: (row.sample_index, row.sample_id)))
            _require(len(candidates) >= need,
                     "CS-WG source stratum lacks enough rows for exact balanced episode")
            offset = _stable_index("CSWG-row", quota.step_index, session, stratum.payload(), modulo=len(candidates))
            chosen = tuple(candidates[(offset + item) % len(candidates)] for item in range(need))
            _require(len({row.sample_id for row in chosen}) == len(chosen),
                     "CS-WG deterministic episode would duplicate a source window within one microbatch")
            rows.extend(chosen)
        batches.append(EpisodeMicrobatch(session, tuple(rows)))
    return BalancedEpisode(quota, tuple(batches), represented)


@dataclass(frozen=True)
class ConcatCompatibilityAuthority:
    """Pre-forward proof that no unit padding/masking/coercion is necessary."""

    sessions: tuple[str, ...]
    input_specs: tuple[tuple[str, str, tuple[int, ...]], ...]
    source_only: bool = True
    unit_padding_masks_or_shape_coercion_forbidden: bool = True

    def __post_init__(self) -> None:
        _require(len(self.sessions) >= 2 and len(set(self.sessions)) == len(self.sessions)
                 and len(self.input_specs) > 0 and self.source_only is True
                 and self.unit_padding_masks_or_shape_coercion_forbidden is True
                 and all(isinstance(key, str) and key and isinstance(dtype, str)
                         and isinstance(shape, tuple) and all(type(item) is int and item > 0 for item in shape)
                         for key, dtype, shape in self.input_specs),
                 "CS-WG concat-compatibility authority drift")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_concat_compatibility_v1",
            "sessions": list(self.sessions),
            "input_specs": [
                {"key": key, "dtype": dtype, "per_row_shape": list(shape)}
                for key, dtype, shape in self.input_specs
            ],
            "physical_preflight_exact_concat_compatibility": True,
            "unit_padding_masks_or_shape_coercion_forbidden": True,
            "per_row_calibration_and_session_ownership": True,
            "source_only": True,
        }

    @property
    def sha256(self) -> str:
        return _sha(_json_bytes(self.payload()))


def _input_spec_for_row(row: SourceEpisodeRow) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    result: list[tuple[str, str, tuple[int, ...]]] = []
    for key in sorted(row.model_inputs):
        array = np.asarray(row.model_inputs[key])
        _require(array.ndim >= 1 and np.isfinite(array).all(),
                 "CS-WG model input must be finite non-scalar per-row data")
        result.append((key, str(array.dtype), tuple(int(item) for item in array.shape)))
    return tuple(result)


def derive_concat_compatibility_authority(episode: BalancedEpisode) -> ConcatCompatibilityAuthority:
    """Require exact trailing shapes/dtypes for all rows before one forward."""
    expected: tuple[tuple[str, str, tuple[int, ...]], ...] | None = None
    for row in episode.all_rows:
        observed = _input_spec_for_row(row)
        if expected is None:
            expected = observed
        _require(observed == expected,
                 "CS-WG heterogeneous per-session model/calibration shape cannot be concatenated without forbidden coercion")
    _require(expected is not None, "CS-WG concat compatibility has no rows")
    return ConcatCompatibilityAuthority(episode.session_ids, expected)


@dataclass(frozen=True)
class SessionSlice:
    session_id: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        _require(isinstance(self.session_id, str) and type(self.start) is int and type(self.stop) is int
                 and 0 <= self.start < self.stop, "CS-WG session slice drift")


@dataclass(frozen=True)
class ConcatenatedEpisode:
    """One exact B32 input handed to exactly one unchanged M1 forward."""

    episode: BalancedEpisode
    compatibility: ConcatCompatibilityAuthority
    model_inputs: Mapping[str, Any] = field(repr=False, compare=False)
    raw_final_targets: Any = field(repr=False, compare=False)
    final_bin_valid: Any = field(repr=False, compare=False)
    session_slices: tuple[SessionSlice, ...]
    row_ownership: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        _require(self.compatibility.sessions == self.episode.session_ids
                 and len(self.session_slices) == len(self.episode.session_ids)
                 and tuple(item.session_id for item in self.session_slices) == self.episode.session_ids
                 and self.session_slices[0].start == 0
                 and self.session_slices[-1].stop == plan.TOTAL_BATCH_SIZE
                 and all(left.stop == right.start for left, right in zip(self.session_slices, self.session_slices[1:]))
                 and len(self.row_ownership) == plan.TOTAL_BATCH_SIZE,
                 "CS-WG concatenated episode ownership/slice topology drift")

    @property
    def digest(self) -> str:
        return _sha(_json_bytes({
            "episode": self.episode.digest,
            "compatibility": self.compatibility.sha256,
            "slices": [item.__dict__ for item in self.session_slices],
            "row_ownership": [dict(item) for item in self.row_ownership],
        }))


def concatenate_episode_for_one_forward(
    episode: BalancedEpisode, *, compatibility: ConcatCompatibilityAuthority,
) -> ConcatenatedEpisode:
    """Concatenate only exact-shape rows; no padding, masks, or session mixing loss."""
    import torch

    observed = derive_concat_compatibility_authority(episode)
    _require(observed.payload() == compatibility.payload(),
             "CS-WG physical preflight concat-compatibility authority drift")
    input_rows: dict[str, list[Any]] = {key: [] for key, _dtype, _shape in compatibility.input_specs}
    targets: list[Any] = []
    valid: list[bool] = []
    ownership: list[Mapping[str, object]] = []
    slices: list[SessionSlice] = []
    offset = 0
    for microbatch in episode.microbatches:
        start = offset
        for row in microbatch.rows:
            for key in input_rows:
                # The immutable typed capability deliberately exposes
                # read-only NumPy buffers.  Copying into this route-owned
                # concatenation prevents Torch from receiving a writable view
                # of source authority memory; it does not pad, reshape, or
                # substitute any per-row M1/calibration value.
                input_rows[key].append(torch.as_tensor(np.array(row.model_inputs[key], copy=True)))
            targets.append(torch.as_tensor(np.array(row.raw_final_target, dtype=np.float32, copy=True)))
            valid.append(bool(row.final_bin_valid))
            ownership.append({
                "session_id": row.session_id,
                "sample_id": row.sample_id,
                "sample_index": row.sample_index,
                "stratum": row.stratum.payload(),
                "input_digests": row.input_digests(),
                "calibration_tensor_owned_per_row": "calib_trialized_neural_features" in row.model_inputs,
            })
            offset += 1
        slices.append(SessionSlice(microbatch.session_id, start, offset))
    _require(offset == plan.TOTAL_BATCH_SIZE and all(valid),
             "CS-WG concatenated episode lost exact total-B32 valid final-bin rows")
    inputs = {key: torch.stack(values, dim=0) for key, values in input_rows.items()}
    target_tensor = torch.stack(targets, dim=0)
    valid_tensor = torch.as_tensor(valid, dtype=torch.bool)
    _require(target_tensor.shape == (plan.TOTAL_BATCH_SIZE, plan.M1_RAW_BEHAVIOR_OUTPUTS)
             and valid_tensor.shape == (plan.TOTAL_BATCH_SIZE,),
             "CS-WG concatenated raw final-bin target shape drift")
    return ConcatenatedEpisode(episode, compatibility, inputs, target_tensor, valid_tensor,
                               tuple(slices), tuple(ownership))


@dataclass(frozen=True)
class ForwardOwnership:
    """One actual concatenated forward and its non-forgeable in-process origin."""

    concatenated: ConcatenatedEpisode = field(repr=False, compare=False)
    raw_outputs: Any = field(repr=False, compare=False)
    forward_call_count: int
    _origin: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require(self.forward_call_count == 1 and self._origin is not None,
                 "CS-WG must have exactly one concatenated forward ownership origin")

    @property
    def token(self) -> str:
        return _sha(_json_bytes({"episode": self.concatenated.digest, "origin": id(self._origin)}))


def run_one_concatenated_forward(model: Any, concatenated: ConcatenatedEpisode) -> ForwardOwnership:
    """Call a supplied unchanged M1-compatible graph exactly once on B32 input."""
    _require(callable(model), "CS-WG route-owned step requires a callable already-built M1 graph")
    output = model(**dict(concatenated.model_inputs))
    if isinstance(output, tuple):
        output = output[0]
    try:
        import torch
    except ImportError as error:  # pragma: no cover - tests install CPU Torch
        raise CSWGCoreError("CS-WG numerical objective requires Torch only at execution-helper call time") from error
    _require(torch.is_tensor(output)
             and output.shape == (plan.TOTAL_BATCH_SIZE, plan.M1_WINDOW_SIZE, plan.M1_RAW_BEHAVIOR_OUTPUTS),
             "CS-WG unchanged M1 forward must emit exact [B32,W100,raw16] output")
    _require(bool(torch.isfinite(output).all()), "CS-WG M1 raw forward output is nonfinite")
    return ForwardOwnership(concatenated, output, 1, object())


class _CurrentLossSeal:
    pass


_CURRENT_LOSS_SEAL = _CurrentLossSeal()


@dataclass(frozen=True)
class CurrentSessionLoss:
    """A differentiable loss tied to the current one-forward ownership object."""

    session_id: str
    value: Any = field(repr=False, compare=False)
    ownership: ForwardOwnership = field(repr=False, compare=False)
    row_count: int
    equalized_task_state: bool
    _seal: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        import torch

        _require(self._seal is _CURRENT_LOSS_SEAL
                 and isinstance(self.ownership, ForwardOwnership)
                 and isinstance(self.session_id, str)
                 and type(self.row_count) is int and self.row_count > 0
                 and self.equalized_task_state is True
                 and torch.is_tensor(self.value) and self.value.ndim == 0
                 and self.value.requires_grad and bool(torch.isfinite(self.value)),
                 "CS-WG session loss must be current, scalar, differentiable, finite, and task-equalized")


@dataclass(frozen=True)
class CSWGObjective:
    mean_loss: Any = field(repr=False, compare=False)
    centered_robust_term: Any = field(repr=False, compare=False)
    loss: Any = field(repr=False, compare=False)
    session_losses: tuple[CurrentSessionLoss, ...] = field(repr=False, compare=False)
    lambda_: float
    tau: float

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_complete_objective_v1",
            "sessions": [item.session_id for item in self.session_losses],
            "row_counts": [item.row_count for item in self.session_losses],
            "lambda": self.lambda_,
            "tau": self.tau,
            "complete_centered_smooth_worst_group": True,
            "single_current_concatenated_forward": True,
            "source_only": True,
        }


def compute_current_session_losses(ownership: ForwardOwnership) -> tuple[CurrentSessionLoss, ...]:
    """Compute exact raw, final-bin-only dense MSE for every represented session."""
    import torch

    concatenated = ownership.concatenated
    output = ownership.raw_outputs
    _require(torch.is_tensor(output) and output.requires_grad,
             "CS-WG current M1 forward output must remain differentiable")
    _require(bool(concatenated.final_bin_valid.all()),
             "CS-WG robust optimization may use only valid final-bin targets")
    final_predictions = output[:, -1, :]
    _require(final_predictions.shape == concatenated.raw_final_targets.shape
             == (plan.TOTAL_BATCH_SIZE, plan.M1_RAW_BEHAVIOR_OUTPUTS),
             "CS-WG final-bin-only raw output/target topology drift")
    losses: list[CurrentSessionLoss] = []
    for segment in concatenated.session_slices:
        prediction = final_predictions[segment.start:segment.stop]
        target = concatenated.raw_final_targets[segment.start:segment.stop].to(
            device=prediction.device, dtype=prediction.dtype,
        )
        # This is dense MSE across all raw coordinates and every valid final
        # bin in that session microbatch; no target normalizer/refit enters.
        value = ((prediction - target) ** 2).mean()
        losses.append(CurrentSessionLoss(
            segment.session_id, value, ownership, segment.stop - segment.start, True, _CURRENT_LOSS_SEAL,
        ))
    return tuple(losses)


def complete_centered_smooth_worst_group_objective(
    session_losses: Sequence[CurrentSessionLoss], *, ownership: ForwardOwnership,
    lambda_: float, tau: float,
) -> CSWGObjective:
    """Compute the full differentiable, centered CS-WG objective once per step."""
    import torch

    _require(isinstance(ownership, ForwardOwnership)
             and 0.0 <= float(lambda_) <= 1.0 and float(tau) > 0.0,
             "CS-WG complete objective lambda/tau range drift")
    rows = tuple(session_losses)
    _require(len(rows) >= 2 and all(isinstance(row, CurrentSessionLoss)
                                    and row.ownership is ownership
                                    and row._seal is _CURRENT_LOSS_SEAL for row in rows),
             "CS-WG rejects single-session, stale, detached, or foreign loss tables")
    _require(len({row.session_id for row in rows}) == len(rows)
             and set(row.session_id for row in rows) == set(item.session_id for item in ownership.concatenated.session_slices),
             "CS-WG current loss/session forward ownership drift")
    # Canonical session ordering makes the objective invariant to a caller's
    # episode mapping order while retaining all differentiable contributions.
    ordered = tuple(sorted(rows, key=lambda row: row.session_id))
    values = torch.stack([row.value for row in ordered])
    mean_loss = values.mean()
    if all(torch.equal(item.value.detach(), ordered[0].value.detach()) for item in ordered[1:]):
        # Avoid a tiny floating logsumexp residual: centered robust is exactly
        # zero when all session losses are equal, as the contract requires.
        robust = mean_loss - mean_loss
    else:
        robust = float(tau) * (torch.logsumexp((values - mean_loss) / float(tau), dim=0)
                                - math.log(len(ordered)))
    loss = mean_loss if float(lambda_) == 0.0 else mean_loss + float(lambda_) * robust
    _require(bool(torch.isfinite(loss)) and bool(torch.isfinite(robust)),
             "CS-WG complete objective became nonfinite")
    return CSWGObjective(mean_loss, robust, loss, ordered, float(lambda_), float(tau))


class RouteOwnedMixedSessionTrainingStep:
    """A minimal future trainer seam that bypasses M1's single-session step only.

    It never pads/collates heterogeneous units, never changes M1's forward,
    and never makes three independent B32 forwards.  The caller must furnish
    the preflighted concat authority and the exact already-built baseline M1
    callable; this helper owns only episode concatenation and loss segmentation.
    """

    def __init__(
        self, *, model: Any, compatibility: ConcatCompatibilityAuthority, run_spec: plan.CSWGRunSpec,
    ) -> None:
        _require(callable(model) and isinstance(compatibility, ConcatCompatibilityAuthority)
                 and isinstance(run_spec, plan.CSWGRunSpec)
                 and compatibility.sessions == run_spec.source_sessions,
                 "CS-WG route-owned training-step construction drift")
        self._model = model
        self._compatibility = compatibility
        self._run_spec = run_spec

    def run(self, episode: BalancedEpisode, *, lambda_: float, tau: float) -> CSWGObjective:
        _require(episode.session_ids == self._compatibility.sessions,
                 "CS-WG route-owned training step received a non-preflighted session episode")
        _require(episode.session_ids == self._run_spec.source_sessions
                 and (self._run_spec.outer_target_session is None
                      or self._run_spec.outer_target_session not in episode.session_ids)
                 and float(lambda_) == float(self._run_spec.lambda_)
                 and float(tau) == float(self._run_spec.tau),
                 "CS-WG route-owned step run-spec/target/objective binding drift")
        concatenated = concatenate_episode_for_one_forward(episode, compatibility=self._compatibility)
        ownership = run_one_concatenated_forward(self._model, concatenated)
        losses = compute_current_session_losses(ownership)
        return complete_centered_smooth_worst_group_objective(
            losses, ownership=ownership, lambda_=lambda_, tau=tau,
        )


def stage0_trainable_parameters() -> tuple[object, ...]:
    """Stage 0 owns no trainable parameter or inference-module surface."""
    return ()


def stage0_inference_modules() -> tuple[object, ...]:
    """Stage 0 intentionally exposes no replacement/forked M1 graph."""
    return ()
