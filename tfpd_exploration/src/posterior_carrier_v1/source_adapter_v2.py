"""Phase-B smoke-v2 source adapter with same-row direction recovery.

The frozen v1 source adapter is intentionally left untouched: its immutable
remote receipt records an honest ``prepare`` failure when one of the already
selected chronological prefix rows has ``target_dir=NaN``.  This additive v2
adapter preserves the v1 trial-selection/count/exposure semantics and changes
only how that *same labelled row* supplies its direction:

* finite ``target_dir`` values are retained verbatim;
* a missing value may be recovered only from that row's finite
  ``target_corners=[x0,y0,x1,y1]`` centre;
* the centre must be an 8-way canonical target with a strict, unique snap;
* the exact audited one-row fallback topology is bound before any model step.

Nothing in this module opens a source file at import time.  The physical
constructor is deferred behind the Phase-B-v2 reviewed lifecycle; the pure
recovery primitive is deliberately CPU/synthetic-test friendly.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from . import core, phase_b, source_adapter


class SourceAdapterV2Error(source_adapter.SourceAdapterError):
    """Fail-closed same-prefix theta-recovery error."""


THETA_RECOVERY_SCHEMA = "posterior_carrier_same_prefix_theta_recovery_v2"
THETA_FALLBACK_TOPOLOGY_SCHEMA = "posterior_carrier_theta_fallback_topology_v2"
THETA_SEMANTICS_V2 = (
    "chronological_labelled_rewarded_trial_target_direction_radians__"
    "same_prefix_target_corners_xyxy_center_unique_canonical_8way_snap_v2"
)
FALLBACK_SOURCE = "same_prefix_trial_target_corners_xyxy_center"
NATIVE_SOURCE = "native_target_dir"
SNAPPED_SOURCE = "same_prefix_target_corners_canonical_snap"
CANONICAL_DIRECTIONS_RAD = tuple(-3.0 * math.pi / 4.0 + index * math.pi / 4.0 for index in range(8))
# A target is 45 degrees away from its next possible direction.  pi/64 is a
# deliberately strict 2.8125-degree acceptance band; it accepts the audited
# 0.002787-rad centre error without admitting a boundary/ambiguous target.
THETA_SNAP_TOLERANCE_RAD = math.pi / 64.0
TARGET_CENTER_RADIUS = 8.0
TARGET_CENTER_RADIUS_TOLERANCE = 1e-3
TARGET_SQUARE_TOLERANCE = 1e-5

EXPECTED_FALLBACK_TOPOLOGY = {
    "schema": THETA_FALLBACK_TOPOLOGY_SCHEMA,
    "fallback_rows": [
        {"session_id": "sub-C_ses-CO-20150313", "trial_index": 33},
    ],
}


def _json_bytes(value: object) -> bytes:
    return phase_b.canonical_json_bytes(value)


def _json_sha(value: object) -> str:
    return phase_b.sha256_bytes(_json_bytes(value))


# This literal is intentionally duplicated from the canonical body rather
# than merely derived at validation time, so a changed digest domain cannot
# silently turn into a new authority.
EXPECTED_FALLBACK_TOPOLOGY_SHA256 = "949431bc77bb484cd2a84e2469d2131bccfb3207655166b6921f41bad8e737c8"

if _json_sha(EXPECTED_FALLBACK_TOPOLOGY) != EXPECTED_FALLBACK_TOPOLOGY_SHA256:
    raise RuntimeError("posterior carrier v2 fallback-topology literal/digest mismatch")


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise SourceAdapterV2Error(f"{name} must be a finite numeric scalar")
    try:
        result = float(value)  # numpy scalar values are valid physical inputs.
    except (TypeError, ValueError) as error:
        raise SourceAdapterV2Error(f"{name} must be a finite numeric scalar") from error
    if not math.isfinite(result):
        raise SourceAdapterV2Error(f"{name} must be finite")
    return result


def _circular_distance(left: float, right: float) -> float:
    return abs((left - right + math.pi) % (2.0 * math.pi) - math.pi)


def _target_corner_geometry(raw: object) -> tuple[tuple[float, float, float, float], float, float, float]:
    """Validate one literal ``[x0,y0,x1,y1]`` target rectangle.

    The recovery deliberately does not infer a direction from a trial's
    position in the sequence or search neighbouring/later trial rows.  Its
    only permitted label-derived input is this four-value field on the exact
    selected row.
    """
    if isinstance(raw, (str, bytes)):
        raise SourceAdapterV2Error("same-prefix target_corners must be a numeric length-4 row")
    try:
        values = tuple(_finite_float(value, name="target_corners value") for value in raw)  # type: ignore[arg-type]
    except TypeError as error:
        raise SourceAdapterV2Error("same-prefix target_corners must be a numeric length-4 row") from error
    if len(values) != 4:
        raise SourceAdapterV2Error("same-prefix target_corners must have exactly four values")
    x0, y0, x1, y1 = values
    width, height = abs(x1 - x0), abs(y1 - y0)
    if width <= 0.0 or height <= 0.0 or abs(width - height) > TARGET_SQUARE_TOLERANCE:
        raise SourceAdapterV2Error("same-prefix target_corners rectangle geometry drift")
    centre_x, centre_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    radius = math.hypot(centre_x, centre_y)
    if abs(radius - TARGET_CENTER_RADIUS) > TARGET_CENTER_RADIUS_TOLERANCE:
        raise SourceAdapterV2Error("same-prefix target centre radius drift")
    if radius <= 0.0:
        raise SourceAdapterV2Error("same-prefix target centre has no direction")
    return values, centre_x, centre_y, radius


def _snap_canonical_direction(angle: float) -> tuple[int, float, float, float]:
    """Return a strictly unique canonical 8-way snap or fail closed."""
    distances = sorted(
        (_circular_distance(angle, candidate), index, candidate)
        for index, candidate in enumerate(CANONICAL_DIRECTIONS_RAD)
    )
    nearest_distance, nearest_index, nearest_angle = distances[0]
    second_distance = distances[1][0]
    if nearest_distance > THETA_SNAP_TOLERANCE_RAD:
        raise SourceAdapterV2Error("same-prefix target centre is too far from a canonical 8-way direction")
    # The second-nearest distance proof makes a boundary tie an explicit
    # error even if a caller changes the tolerance later.
    if second_distance <= THETA_SNAP_TOLERANCE_RAD or second_distance <= nearest_distance:
        raise SourceAdapterV2Error("same-prefix target centre canonical direction is not unique")
    return nearest_index, nearest_angle, nearest_distance, second_distance


def _prefix_row_ids(session: str, prefix: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    if len(prefix) != 30:
        raise SourceAdapterV2Error("same-prefix theta recovery requires exactly 30 selected trials")
    trial_indices: list[int] = []
    for item in prefix:
        if not isinstance(item, Mapping) or type(item.get("trial_index")) is not int:
            raise SourceAdapterV2Error("same-prefix theta recovery trial-index schema drift")
        trial_indices.append(int(item["trial_index"]))
    if len(set(trial_indices)) != len(trial_indices):
        raise SourceAdapterV2Error("same-prefix theta recovery duplicate original trial index")
    return tuple(f"{session}:trial:{index}" for index in trial_indices)


def recover_same_prefix_theta(
    *,
    session: str,
    prefix: Sequence[Mapping[str, object]],
    target_corners_by_trial_index: Mapping[int, object],
) -> tuple[torch.Tensor, tuple[str, ...], dict[str, object]]:
    """Recover only missing directions from the exact existing prefix rows.

    ``target_corners_by_trial_index`` may physically originate from the whole
    trials table, but lookup is strictly by the already-selected original row
    index.  In particular, a valid direction on a later row can never repair a
    missing selected row.
    """
    if not isinstance(session, str) or not session:
        raise SourceAdapterV2Error("same-prefix theta recovery session identifier drift")
    if not isinstance(target_corners_by_trial_index, Mapping):
        raise SourceAdapterV2Error("same-prefix target-corners table must be a mapping")
    row_ids = _prefix_row_ids(session, prefix)
    theta: list[float] = []
    sources: list[str] = []
    fallback_rows: list[dict[str, object]] = []
    for position, item in enumerate(prefix):
        trial_index = int(item["trial_index"])
        raw_direction = item.get("target_dir")
        try:
            direction = _finite_float(raw_direction, name="target_dir")
        except SourceAdapterV2Error:
            if trial_index not in target_corners_by_trial_index:
                raise SourceAdapterV2Error(
                    "same-prefix target_corners missing for the selected unlabeled trial; later-row substitution is forbidden"
                )
            corners, centre_x, centre_y, radius = _target_corner_geometry(target_corners_by_trial_index[trial_index])
            derived_angle = math.atan2(centre_y, centre_x)
            canonical_index, canonical_angle, snap_error, second_error = _snap_canonical_direction(derived_angle)
            geometry = {
                "target_corners_xyxy": list(corners),
                "target_corners_sha256": _json_sha(list(corners)),
                "centre_xy": [centre_x, centre_y],
                "centre_radius": radius,
                "expected_radius": TARGET_CENTER_RADIUS,
                "radius_tolerance": TARGET_CENTER_RADIUS_TOLERANCE,
                "square_tolerance": TARGET_SQUARE_TOLERANCE,
            }
            fallback_rows.append({
                "prefix_position": position,
                "trial_index": trial_index,
                "source": FALLBACK_SOURCE,
                "geometry": geometry,
                "derived_centre_angle_rad": derived_angle,
                "canonical_direction_index": canonical_index,
                "canonical_theta_rad": canonical_angle,
                "snap_error_rad": snap_error,
                "second_nearest_error_rad": second_error,
                "snap_tolerance_rad": THETA_SNAP_TOLERANCE_RAD,
            })
            theta.append(canonical_angle)
            sources.append(SNAPPED_SOURCE)
        else:
            theta.append(direction)
            sources.append(NATIVE_SOURCE)
    theta_tensor = torch.tensor(theta, dtype=torch.float64)
    design = torch.stack((torch.cos(theta_tensor), torch.sin(theta_tensor), torch.ones_like(theta_tensor)), dim=1)
    rank = int(torch.linalg.matrix_rank(design).item())
    # `torch.linalg.cond` supplies a receipt diagnostic, while exact full rank
    # is the required fail-closed condition for the three posterior columns.
    condition = float(torch.linalg.cond(design).item())
    if rank != 3 or not math.isfinite(condition):
        raise SourceAdapterV2Error("same-prefix theta design matrix is not finite full rank")
    evidence = {
        "schema": THETA_RECOVERY_SCHEMA,
        "session_id": session,
        "prefix_row_ids": list(row_ids),
        "prefix_rows_sha256": _json_sha(list(row_ids)),
        "theta_m30_sha256": core.tensor_digest(theta_tensor),
        "theta_sources_by_prefix_position": sources,
        "fallback_rows": fallback_rows,
        "fallback_count": len(fallback_rows),
        "no_later_row_substitution": True,
        "design_matrix_rank": rank,
        "design_matrix_condition": condition,
    }
    evidence["body_sha256"] = _json_sha(evidence)
    return theta_tensor, row_ids, evidence


def validate_theta_recovery_evidence(
    value: Mapping[str, object],
    *,
    session: str,
    prefix_row_ids: Sequence[str],
    theta_sha256: str,
) -> dict[str, object]:
    """Validate persisted recovery evidence without reopening source labels."""
    required = {
        "schema", "session_id", "prefix_row_ids", "prefix_rows_sha256", "theta_m30_sha256",
        "theta_sources_by_prefix_position", "fallback_rows", "fallback_count", "no_later_row_substitution",
        "design_matrix_rank", "design_matrix_condition", "body_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise SourceAdapterV2Error("theta-recovery evidence schema drift")
    payload = dict(value)
    body = {key: payload[key] for key in required - {"body_sha256"}}
    if payload["body_sha256"] != _json_sha(body):
        raise SourceAdapterV2Error("theta-recovery evidence body digest drift")
    expected_rows = list(prefix_row_ids)
    if (payload["schema"] != THETA_RECOVERY_SCHEMA or payload["session_id"] != session
            or payload["prefix_row_ids"] != expected_rows
            or payload["prefix_rows_sha256"] != _json_sha(expected_rows)
            or payload["theta_m30_sha256"] != theta_sha256
            or payload["no_later_row_substitution"] is not True
            or payload["design_matrix_rank"] != 3
            or not isinstance(payload["design_matrix_condition"], (float, int))
            or not math.isfinite(float(payload["design_matrix_condition"]))):
        raise SourceAdapterV2Error("theta-recovery evidence binding drift")
    sources = payload["theta_sources_by_prefix_position"]
    fallback_rows = payload["fallback_rows"]
    if (not isinstance(sources, list) or len(sources) != 30
            or any(item not in {NATIVE_SOURCE, SNAPPED_SOURCE} for item in sources)
            or not isinstance(fallback_rows, list)
            or payload["fallback_count"] != len(fallback_rows)
            or sum(item == SNAPPED_SOURCE for item in sources) != len(fallback_rows)):
        raise SourceAdapterV2Error("theta-recovery source/fallback topology drift")
    seen_positions: set[int] = set()
    for fallback in fallback_rows:
        expected = {
            "prefix_position", "trial_index", "source", "geometry", "derived_centre_angle_rad",
            "canonical_direction_index", "canonical_theta_rad", "snap_error_rad",
            "second_nearest_error_rad", "snap_tolerance_rad",
        }
        if not isinstance(fallback, Mapping) or set(fallback) != expected:
            raise SourceAdapterV2Error("theta-recovery fallback row schema drift")
        position = fallback["prefix_position"]
        if (type(position) is not int or not 0 <= position < 30 or position in seen_positions
                or sources[position] != SNAPPED_SOURCE or type(fallback["trial_index"]) is not int
                or fallback["source"] != FALLBACK_SOURCE
                or fallback["snap_tolerance_rad"] != THETA_SNAP_TOLERANCE_RAD):
            raise SourceAdapterV2Error("theta-recovery fallback row binding drift")
        # The fallback's original NWB row must be the row already fixed at
        # this exact prefix position.  A syntactically valid recovery body
        # therefore cannot move a geometry-derived label to another selected
        # row (or claim an unselected/later trial) by merely recomputing its
        # body digest.
        if expected_rows[position] != f"{session}:trial:{fallback['trial_index']}":
            raise SourceAdapterV2Error("theta-recovery fallback trial/prefix-row binding drift")
        seen_positions.add(position)
        geometry = fallback["geometry"]
        geometry_keys = {
            "target_corners_xyxy", "target_corners_sha256", "centre_xy", "centre_radius",
            "expected_radius", "radius_tolerance", "square_tolerance",
        }
        if (not isinstance(geometry, Mapping) or set(geometry) != geometry_keys
                or geometry["expected_radius"] != TARGET_CENTER_RADIUS
                or geometry["radius_tolerance"] != TARGET_CENTER_RADIUS_TOLERANCE
                or geometry["square_tolerance"] != TARGET_SQUARE_TOLERANCE):
            raise SourceAdapterV2Error("theta-recovery geometry authority drift")
        corners, centre_x, centre_y, radius = _target_corner_geometry(geometry["target_corners_xyxy"])
        if (geometry["target_corners_sha256"] != _json_sha(list(corners))
                or geometry["centre_xy"] != [centre_x, centre_y]
                or geometry["centre_radius"] != radius):
            raise SourceAdapterV2Error("theta-recovery geometry digest/value drift")
        derived = _finite_float(fallback["derived_centre_angle_rad"], name="derived centre angle")
        if derived != math.atan2(centre_y, centre_x):
            raise SourceAdapterV2Error("theta-recovery centre-angle drift")
        index, canonical, error, second = _snap_canonical_direction(derived)
        if (fallback["canonical_direction_index"] != index
                or fallback["canonical_theta_rad"] != canonical
                or fallback["snap_error_rad"] != error
                or fallback["second_nearest_error_rad"] != second):
            raise SourceAdapterV2Error("theta-recovery canonical-snap drift")
    return payload


def theta_fallback_topology(
    *,
    roster: Sequence[str],
    recovery_by_session: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate all per-session recovery facts and bind the audited topology."""
    strict_roster = phase_b._immutable_roster(roster, expected_count=phase_b.SOURCE_SESSION_COUNT)
    if set(recovery_by_session) != set(strict_roster):
        raise SourceAdapterV2Error("theta-recovery strict-27 roster topology drift")
    rows: list[dict[str, object]] = []
    for session in strict_roster:
        evidence = recovery_by_session[session]
        fallbacks = evidence.get("fallback_rows") if isinstance(evidence, Mapping) else None
        if not isinstance(fallbacks, list):
            raise SourceAdapterV2Error("theta-recovery fallback aggregation schema drift")
        for fallback in fallbacks:
            if not isinstance(fallback, Mapping) or type(fallback.get("trial_index")) is not int:
                raise SourceAdapterV2Error("theta-recovery fallback aggregation row drift")
            rows.append({"session_id": session, "trial_index": int(fallback["trial_index"])})
    body = {"schema": THETA_FALLBACK_TOPOLOGY_SCHEMA, "fallback_rows": rows}
    digest = _json_sha(body)
    if body != EXPECTED_FALLBACK_TOPOLOGY or digest != EXPECTED_FALLBACK_TOPOLOGY_SHA256:
        raise SourceAdapterV2Error("unexpected same-prefix theta fallback topology")
    return {**body, "body_sha256": digest}


def validate_theta_fallback_topology(value: Mapping[str, object]) -> dict[str, object]:
    required = {"schema", "fallback_rows", "body_sha256"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise SourceAdapterV2Error("theta-fallback topology schema drift")
    body = {"schema": value["schema"], "fallback_rows": value["fallback_rows"]}
    if (body != EXPECTED_FALLBACK_TOPOLOGY or value["body_sha256"] != EXPECTED_FALLBACK_TOPOLOGY_SHA256
            or value["body_sha256"] != _json_sha(body)):
        raise SourceAdapterV2Error("theta-fallback topology digest/binding drift")
    return dict(value)


def _source_counts_exposure_theta_v2(
    *,
    path: Path,
    session: str,
    expected_units: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...], dict[str, object]]:
    """Read counts and recover theta from the same selected prefix rows only."""
    import numpy as np
    from pynwb import NWBHDF5IO
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials

    trials = list_datamodule_rewarded_trials(path, bin_size_ms=20, window_size=50, trial_result_filter="R")
    if len(trials) < 30:
        raise SourceAdapterV2Error("source session lacks 30 chronological rewarded trials")
    prefix: list[Mapping[str, object]] = [dict(item) for item in trials[:30]]
    row_ids = _prefix_row_ids(session, prefix)
    starts = np.asarray([_finite_float(item.get("start_time"), name="source prefix start_time") for item in prefix], dtype=np.float64)
    stops = np.asarray([_finite_float(item.get("stop_time"), name="source prefix stop_time") for item in prefix], dtype=np.float64)
    if np.any(stops <= starts):
        raise SourceAdapterV2Error("source prefix exposure is nonpositive")
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        if nwb.units is None or "trials" not in nwb.intervals:
            raise SourceAdapterV2Error("source NWB lacks units or trials")
        units = nwb.units.to_dataframe()
        if len(units) != expected_units:
            raise SourceAdapterV2Error("source NWB unit count drift")
        trials_df = nwb.intervals["trials"].to_dataframe()
        corners_by_trial_index: dict[int, object] = {}
        # Crucially, this loop ranges only over the prior datamodule-selected
        # prefix IDs.  It neither re-filters nor scans for a later substitute.
        for item in prefix:
            trial_index = int(item["trial_index"])
            if trial_index not in trials_df.index:
                raise SourceAdapterV2Error("selected source prefix trial is absent from NWB trials table")
            row = trials_df.loc[trial_index]
            if getattr(row, "ndim", 1) != 1:
                raise SourceAdapterV2Error("source trials table original-index ambiguity")
            corners_by_trial_index[trial_index] = row.get("target_corners")
        theta, _recovered_ids, recovery = recover_same_prefix_theta(
            session=session, prefix=prefix, target_corners_by_trial_index=corners_by_trial_index,
        )
        if _recovered_ids != row_ids:
            raise SourceAdapterV2Error("same-prefix row identity drift during theta recovery")
        counts = np.empty((expected_units, 30), dtype=np.int64)
        for index in range(expected_units):
            spikes = np.asarray(units.iloc[index]["spike_times"], dtype=np.float64)
            if spikes.size and not np.all(spikes[:-1] <= spikes[1:]):
                raise SourceAdapterV2Error("source unit spike ordering drift")
            counts[index] = np.searchsorted(spikes, stops, side="left") - np.searchsorted(spikes, starts, side="left")
    return torch.from_numpy(counts), torch.from_numpy(stops - starts), theta, row_ids, recovery


@dataclass
class PhysicalPosteriorSourceAdapterV2(source_adapter.PhysicalPosteriorSourceAdapter):
    """V1-compatible physical adapter carrying explicit theta-recovery proof."""

    theta_recovery_by_session: Mapping[str, Mapping[str, object]]


def wrap_v1_authority_with_theta_recovery(
    *,
    adapter: PhysicalPosteriorSourceAdapterV2,
    nested_v1_authority: Mapping[str, object],
    v2_closure: Mapping[str, object],
) -> dict[str, object]:
    """Attach v2 recovery evidence to an already complete v1 authority.

    The physical backend enriches the nested v1 payload with runtime device,
    optimizer, credibility, and timing evidence first.  Keeping that payload
    nested lets the v2 lifecycle invoke the frozen v1 semantic validator
    before it accepts the only new field in this successor.
    """
    if not isinstance(adapter, PhysicalPosteriorSourceAdapterV2) or not isinstance(nested_v1_authority, Mapping):
        raise SourceAdapterV2Error("v2 authority wrapper type drift")
    recovery: dict[str, object] = {}
    for session in adapter.roster:
        input_payload = adapter.session_inputs[session].payload()
        recovery[session] = validate_theta_recovery_evidence(
            adapter.theta_recovery_by_session[session], session=session,
            prefix_row_ids=input_payload["prefix_row_ids"], theta_sha256=input_payload["theta_m30_sha256"],
        )
    topology = theta_fallback_topology(roster=adapter.roster, recovery_by_session=recovery)
    return {
        "schema": "posterior_carrier_source_authority_v2",
        "cell": core.CELL,
        "v1_compatible_authority": dict(nested_v1_authority),
        "theta_recovery_by_session": recovery,
        "theta_fallback_topology": topology,
        "theta_semantics": THETA_SEMANTICS_V2,
        "closure": dict(v2_closure),
    }


def build_physical_source_adapter_v2(
    root: Path,
    *,
    source_data: phase_b.SourceDataRootCapability,
    num_workers: int = 4,
    on_source_opened: Callable[[], None] | None = None,
) -> PhysicalPosteriorSourceAdapterV2:
    """Deferred strict-source v2 builder; no use in dry/public paths."""
    started = time.monotonic()
    source_adapter._prepend_sua_package(root)
    from mc_maze.multisession_datamodule import SessionBatchSampler

    stage_authority = source_adapter._load_strict27_stage_authority(Path(root).absolute(), source_data=source_data)
    authority = stage_authority.payload()
    dm, a2, roster, train_files = source_adapter._construct_train_only_datamodule(
        root=Path(root).absolute(), authority=authority, source_data=source_data,
        num_workers=num_workers, on_source_opened=on_source_opened,
    )
    source_inputs: dict[str, source_adapter.PosteriorSessionInput] = {}
    recovery_by_session: dict[str, Mapping[str, object]] = {}
    for session, path in zip(roster, train_files, strict=True):
        row = authority["source_lineage"]["rows_by_session"][session]
        source_adapter._verify_source_lineage_file(path, row)
        record = dm.train_dataset.sessions[session]
        expected_units = int(row["unit_count"])
        if (record.source_unit_count != expected_units or record.neural.shape[1] != expected_units
                or record.channel_ids is None
                or not torch.equal(torch.as_tensor(record.channel_ids), torch.arange(expected_units))):
            raise SourceAdapterV2Error("source dataset/unit order drift")
        counts, exposure, theta, row_ids, recovery = _source_counts_exposure_theta_v2(
            path=path, session=session, expected_units=expected_units,
        )
        raw_t4, unit_order_sha, row_order_proof = source_adapter._raw_m30_t4_and_unit_order(
            path=path, session=session, expected_units=expected_units,
            record_source_unit_count=int(record.source_unit_count), record_channel_ids=record.channel_ids,
        )
        source_adapter._verify_source_lineage_file(path, row)
        source_inputs[session] = source_adapter.PosteriorSessionInput(
            session_id=session, raw_m30_t4=raw_t4, counts_m30=counts,
            exposure_m30=exposure, theta_m30=theta, prefix_row_ids=row_ids,
            source_path_sha256=str(row["sha256"]), unit_order_sha256=unit_order_sha,
            raw_t4_row_order_proof=row_order_proof,
        )
        recovery_by_session[session] = recovery
    # Exact one-row topology is checked before posterior construction/model
    # setup; unexpected missing labels cannot become a silent science factor.
    theta_fallback_topology(roster=roster, recovery_by_session=recovery_by_session)
    prior, bank, _posteriors = source_adapter.build_source_posterior_bank(roster=roster, inputs=source_inputs, seed=42)
    behavior_sha = a2.normalizer_value_sha256(*dm._behavior_stats)
    if behavior_sha != authority["behavior_semantic_sha256"]:
        raise SourceAdapterV2Error("strict source behaviour normalizer drift")
    sampler = SessionBatchSampler(dm.train_dataset, batch_size=phase_b.SOURCE_BATCH_SIZE, shuffle=True, seed=42)
    if len(sampler) != phase_b.SOURCE_STEPS_PER_EPOCH:
        raise SourceAdapterV2Error("strict source B32 batch topology drift")
    stage_authority.revalidate()
    return PhysicalPosteriorSourceAdapterV2(
        dataset=dm.train_dataset, sampler=sampler, roster=roster, bank=bank, prior=prior,
        session_inputs=source_inputs, source_authority_metadata=authority,
        behavior_normalizer_semantic_sha256=behavior_sha,
        preparation_seconds=time.monotonic() - started, train_files=train_files,
        theta_recovery_by_session=recovery_by_session,
    )
