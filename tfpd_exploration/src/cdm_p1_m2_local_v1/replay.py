"""The LOCAL-M2 replay: the F carrier axis and the G full stack.

Everything the P1 law needs is ORCHESTRATED from the frozen packages:

* ``direction_estimator.measure_trial``  -- the deployable pseudo-direction
  measurement (law/weight/binding per row spec), verbatim;
* ``stage_p_gate.evaluate_block`` / ``EvidenceBankP`` -- the three-factor
  commit gate and the append-only evidence bank, verbatim;
* ``trust_region.active_carrier``        -- the anchored trust-region movement;
* ``stage_p_replay.enumerate_gate_grid`` / ``enumerate_mass_grid`` /
  ``select_vector`` / ``CellHyperparameters`` / ``ROW_SPECS`` -- the sealed
  selection grids, tie-break law and row specs, verbatim;
* the sealed local screens' own executors -- the comparator's carrier/query
  laws for the F family and ``m2_precision_cdm_v2_screen_v1``'s activity-CDM
  laws for the G family -- so the anchors are same-law by construction.

This module contributes only the M2 surface bindings: the trial partition of
the query streams, the evidence stream boundary, the velocity unit law, and
the ``M2SupportAnchor``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

try:
    from src.causal_dual_memory_cell_d_v1 import core as cdm_core
    from src.support_anchored_t4_stage_o_v1 import trust_region
    from src.support_anchored_t4_stage_p_v1 import direction_estimator
    from src.support_anchored_t4_stage_p_v1 import gate as stage_p_gate
    from src.support_anchored_t4_stage_p_v1 import replay as stage_p_replay
except ModuleNotFoundError as error:
    if not (error.name == "src" or str(error.name).startswith("src.")):
        raise
    from tfpd_exploration.src.causal_dual_memory_cell_d_v1 import core as cdm_core
    from tfpd_exploration.src.support_anchored_t4_stage_o_v1 import trust_region
    from tfpd_exploration.src.support_anchored_t4_stage_p_v1 import direction_estimator
    from tfpd_exploration.src.support_anchored_t4_stage_p_v1 import gate as stage_p_gate
    from tfpd_exploration.src.support_anchored_t4_stage_p_v1 import replay as stage_p_replay

from . import plan
from .anchor import M2SupportAnchor, build_groups


class M2LocalReplayError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M2LocalReplayError(message)


def _array_digest(value: np.ndarray) -> str:
    return cdm_core.array_digest(np.ascontiguousarray(value))


# ---------------------------------------------------------------------------
# Frozen session material (the F family).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryTrial:
    """One completed trial's padded interval and its scored/measured windows."""

    position: int
    start_bin: int
    stop_bin: int
    metric_starts: np.ndarray
    causal_starts: np.ndarray

    @property
    def evidence_eligible(self) -> bool:
        return int(self.position) >= plan.EVIDENCE_START_POSITION

    @property
    def trial_id(self) -> str:
        return f"m2local:trial:{int(self.position)}"


@dataclass(frozen=True)
class SessionMaterial:
    session: str
    surface: str
    dataset: Any
    starts: np.ndarray
    targets: np.ndarray
    trials: tuple[QueryTrial, ...]
    channel_ids: np.ndarray

    @property
    def neural(self) -> np.ndarray:
        return np.asarray(self.dataset.neural_data[self.session], dtype=np.float32)


def surface_starts(dataset: Any, session: str, surface: str) -> np.ndarray:
    """The sealed surface law: post-30 common windows, or the official query."""
    from src.m2_t4_activity_budget_screen_v1.core import (
        select_common_post30_window_starts,
    )

    starts = np.asarray(
        [start for name, start in dataset.window_indices if name == session],
        dtype=np.int64,
    )
    _require(starts.size > 0 and np.all(np.diff(starts) > 0), f"{session}: query order drift")
    if surface in ("within_post30", "external_post30_local"):
        starts = select_common_post30_window_starts(
            starts, np.asarray(dataset.trial_start_indices[session], dtype=np.int64)
        )
    else:
        _require(surface == "external_official_query", f"unknown surface {surface}")
    return np.ascontiguousarray(starts, dtype=np.int64)


def build_session_material(*, dataset: Any, session: str, surface: str) -> SessionMaterial:
    starts = surface_starts(dataset, session, surface)
    endpoints = np.ascontiguousarray(starts + plan.WINDOW_SIZE - 1, dtype=np.int64)
    targets = np.ascontiguousarray(
        np.asarray(dataset.covariate_data[session], dtype=np.float32)[endpoints],
        dtype=np.float32,
    )
    _require(targets.shape == (starts.size, 2) and np.isfinite(targets).all(),
             f"{session}/{surface}: target topology drift")
    trial_starts = np.asarray(dataset.trial_start_indices[session], dtype=np.int64)
    _require(trial_starts.size > plan.ACTIVITY_HORIZON,
             f"{session}: lacks a trial-{plan.ACTIVITY_HORIZON} boundary")
    total_bins = int(np.asarray(dataset.neural_data[session]).shape[0])
    assignment = np.searchsorted(trial_starts, endpoints, side="right") - 1
    _require(int(assignment.min()) >= 0, f"{session}/{surface}: window before the first trial")
    trials: list[QueryTrial] = []
    for position in range(trial_starts.size):
        start_bin = int(trial_starts[position])
        stop_bin = int(trial_starts[position + 1]) if position + 1 < trial_starts.size else total_bins
        metric = np.ascontiguousarray(starts[assignment == position], dtype=np.int64)
        causal = np.arange(start_bin, stop_bin - plan.WINDOW_SIZE + 1, dtype=np.int64)
        trials.append(QueryTrial(
            position=position, start_bin=start_bin, stop_bin=stop_bin,
            metric_starts=metric, causal_starts=np.ascontiguousarray(causal, dtype=np.int64),
        ))
    joined = np.concatenate([trial.metric_starts for trial in trials]) if trials \
        else np.asarray([], dtype=np.int64)
    _require(np.array_equal(np.sort(joined), starts),
             f"{session}/{surface}: query trials do not partition the surface windows")
    return SessionMaterial(
        session=session, surface=surface, dataset=dataset, starts=starts,
        targets=targets, trials=tuple(trials),
        channel_ids=np.arange(int(np.asarray(dataset.neural_data[session]).shape[1]), dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# The sealed F-family carrier law (bit-anchored to the same-query rows).
# ---------------------------------------------------------------------------


def support_indices(dataset: Any, session: str, budget: int) -> np.ndarray:
    """The sealed support-selection law (D-opt four / chronological first-M)."""
    angles = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)
    _require(angles.size >= plan.ACTIVITY_HORIZON, f"{session} lacks first-30 target metadata")
    if budget != plan.M4:
        selected = np.arange(budget, dtype=np.int64)
        _require(int(np.isfinite(angles[selected]).sum()) >= 3,
                 f"{session} M{budget} has fewer than three directional trials")
        return np.ascontiguousarray(selected)
    from sua_exploration.mc_maze.d_optimal_calibration_design import (
        greedy_forward_d_optimal_indices,
    )

    candidates = np.flatnonzero(np.isfinite(angles[: plan.ACTIVITY_HORIZON])).astype(np.int64)
    _require(candidates.size >= budget, f"{session} lacks four directional first-30 candidates")
    local = greedy_forward_d_optimal_indices(angles[candidates], budget)
    selected = np.sort(candidates[local]).astype(np.int64, copy=False)
    _require(selected.size == budget and int(selected.max()) < plan.ACTIVITY_HORIZON,
             f"{session} M4 D-opt selection drift")
    return np.ascontiguousarray(selected)


def sealed_carrier(dataset: Any, session: str, budget: int) -> dict[str, Any]:
    """The sealed ``ridge_static_m{budget}`` carrier law, verbatim."""
    from src.calibration_budget_comparators_v1 import fit_ridge_t4

    selected = support_indices(dataset, session, budget)
    sums = np.asarray(dataset.calib_trial_spike_sums[session][selected], dtype=np.float64)
    lengths = np.asarray(dataset.calib_trial_lengths[session][selected], dtype=np.float64)
    angles = np.asarray(dataset.calib_trial_target_angles[session][selected], dtype=np.float64)
    usable = np.isfinite(angles)
    _require(int(usable.sum()) >= 3,
             f"{session} M{budget} has fewer than three directional trials")
    rates = np.ascontiguousarray(sums[usable] / lengths[usable, None], dtype=np.float64)
    theta = np.ascontiguousarray(angles[usable], dtype=np.float64)
    raw, evidence = fit_ridge_t4(rates, theta, normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA)
    mean = np.asarray(dataset.side_feature_mean, dtype=np.float32)
    std = np.asarray(dataset.side_feature_std, dtype=np.float32)
    _require(mean.shape == std.shape == (4,) and np.all(std > 0), "T4 normalizer drift")
    side = np.ascontiguousarray((raw - mean) / std, dtype=np.float32)
    _require(side.shape == (plan.CHANNELS, 4) and np.isfinite(side).all(),
             "normalized sealed carrier drift")
    calibration = np.asarray(dataset.calib_trialized_neural_features[session], dtype=np.float32)
    activity = np.ascontiguousarray(calibration[selected], dtype=np.float32)
    return {
        "budget": int(budget),
        "selected": selected,
        "theta": theta,
        "rates": rates,
        "raw_t4": np.ascontiguousarray(raw, dtype=np.float32),
        "side": side,
        "activity": activity,
        "fit_evidence": {
            key: evidence[key] for key in (
                "normalized_lambda", "design_rank", "design_condition", "gcv",
                "raw_t4_sha256",
            )
        },
        "selected_sha256": _array_digest(selected),
        "theta_sha256": _array_digest(theta),
        "rates_sha256": _array_digest(rates),
        "raw_t4_sha256": _array_digest(raw),
        "side_sha256": _array_digest(side),
        "activity_sha256": _array_digest(activity),
    }


# ---------------------------------------------------------------------------
# The frozen decode path (identity cache keyed by activity/side bytes).
# ---------------------------------------------------------------------------


class M2Decoder:
    """Cached-identity decode over the frozen M2 student, deterministic settings."""

    def __init__(self, torch: Any, student: Any, device: Any, *, batch_size: int) -> None:
        self.torch = torch
        self.student = student
        self.device = device
        self.batch_size = int(batch_size)
        self._identity_cache: dict[tuple[bytes, bytes], Any] = {}

    def identity(self, activity: np.ndarray, side: np.ndarray,
                 keep: Optional[np.ndarray] = None) -> Any:
        channels = np.arange(activity.shape[2], dtype=np.int64) if keep is None \
            else np.asarray(keep, dtype=np.int64)
        activity_np = activity if keep is None else np.ascontiguousarray(activity[:, :, channels])
        side_np = side if keep is None else np.ascontiguousarray(side[channels])
        key = (np.ascontiguousarray(activity_np, dtype=np.float32).tobytes(),
               np.ascontiguousarray(side_np, dtype=np.float32).tobytes())
        cached = self._identity_cache.get(key)
        if cached is not None:
            return cached
        calibration = self.torch.from_numpy(
            np.ascontiguousarray(activity_np, dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        side_tensor = self.torch.from_numpy(
            np.ascontiguousarray(side_np, dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            identity = self.student.compute_identity(calibration, side_features=side_tensor)
        _require(tuple(identity.shape) == (1, side_np.shape[0], plan.WINDOW_SIZE),
                 "cached identity shape drift")
        self._identity_cache[key] = identity
        return identity

    def decode(self, neural: np.ndarray, identity: Any) -> np.ndarray:
        values = np.ascontiguousarray(neural, dtype=np.float32)
        _require(values.ndim == 3 and values.shape[1] == plan.WINDOW_SIZE,
                 "decode window shape drift")
        outputs: list[np.ndarray] = []
        with self.torch.inference_mode():
            for offset in range(0, values.shape[0], self.batch_size):
                chunk = values[offset : offset + self.batch_size]
                tensor = self.torch.from_numpy(chunk).to(self.device)
                prediction, _aux = self.student(tensor, identity=identity)
                _require(bool(self.torch.isfinite(prediction).all().item()),
                         "decode produced nonfinite output")
                outputs.append(
                    prediction[:, -1, :].detach().cpu().numpy().astype(np.float32, copy=False)
                    / plan.BEHAVIOR_SCALE
                )
        return np.ascontiguousarray(np.concatenate(outputs, axis=0), dtype=np.float32)

    def clear_cache(self) -> None:
        self._identity_cache.clear()

    def windows(self, neural: np.ndarray, starts: np.ndarray,
                keep: Optional[np.ndarray] = None) -> np.ndarray:
        starts = np.asarray(starts, dtype=np.int64)
        indices = starts[:, None] + np.arange(plan.WINDOW_SIZE, dtype=np.int64)[None, :]
        block = np.ascontiguousarray(neural[indices], dtype=np.float32)
        if keep is None:
            return block
        return np.ascontiguousarray(block[:, :, np.asarray(keep, dtype=np.int64)], dtype=np.float32)


def _side_from_raw(raw_t4: np.ndarray, dataset: Any) -> np.ndarray:
    mean = np.asarray(dataset.side_feature_mean, dtype=np.float32)
    std = np.asarray(dataset.side_feature_std, dtype=np.float32)
    side = np.ascontiguousarray(
        (np.asarray(raw_t4, dtype=np.float32) - mean) / std, dtype=np.float32,
    )
    _require(np.isfinite(side).all(), "normalized active carrier became nonfinite")
    return side


def _cm_s(prediction: np.ndarray) -> np.ndarray:
    """The pre-registered velocity unit law: covariate m/s view -> cm/s."""
    return np.ascontiguousarray(
        np.asarray(prediction, dtype=np.float64) * float(plan.VELOCITY_UNIT_LAW["view_scale"]),
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# The per-trial P1 loop (F family).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellSpec:
    name: str
    carrier_law: str
    online: bool


CELL_SPECS_F = (
    CellSpec("F00m_anchor", "frozen_t4_sealed_batching", False),
    CellSpec("F00m", "frozen_t4_per_trial", False),
    CellSpec("F01m", "p1_online", True),
)


def _trial_views(
    decoder: M2Decoder, material: SessionMaterial, trial: QueryTrial,
    *, activity: np.ndarray, side: np.ndarray, groups: cdm_core.ComplementaryGroups,
) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    """The four held-group cm/s views of one completed trial (causal windows)."""
    views = []
    neural = material.neural
    for group in range(cdm_core.GROUP_COUNT):
        keep = np.flatnonzero(~np.asarray(groups.held_mask(group), dtype=bool)).astype(np.int64)
        held_identity = decoder.identity(activity, side, keep=keep)
        prediction = decoder.decode(decoder.windows(neural, trial.causal_starts, keep=keep), held_identity)
        validity = cdm_core.VelocityValidityEvidence(
            valid_mask=np.ones(trial.causal_starts.size, dtype=np.bool_),
            session_id=material.session,
            trial_id=trial.trial_id,
            prediction_interval_start_bin=int(trial.causal_starts[0] + plan.WINDOW_SIZE - 1),
            prediction_interval_stop_bin=int(trial.causal_starts[-1] + plan.WINDOW_SIZE),
        )
        views.append(cdm_core.CompletedVelocityPrediction(_cm_s(prediction), validity))
    return tuple(views), {}


def rollout_session_f(
    decoder: M2Decoder,
    material: SessionMaterial,
    *,
    budget: int,
    spec: CellSpec,
    anchor: M2SupportAnchor,
    carrier: Mapping[str, Any],
    hp: Optional[stage_p_replay.CellHyperparameters],
    config: cdm_core.CDMDConfig,
    row_spec: Optional[stage_p_replay.RowSpec] = None,
) -> dict[str, Any]:
    """Run one F cell over one session; F00m_anchor uses the sealed batching."""
    activity = np.asarray(carrier["activity"], dtype=np.float32)
    _require(activity.shape[0] == int(carrier["budget"]),
             "activity budget must equal the carrier budget (the sealed static law)")
    targets = material.targets
    side = np.asarray(carrier["side"], dtype=np.float32)
    raw_support = np.asarray(carrier["raw_t4"], dtype=np.float32)
    neural = material.neural
    if spec.name == "F00m_anchor":
        identity = decoder.identity(activity, side)
        prediction = decoder.decode(decoder.windows(neural, material.starts), identity)
        return {
            "cell": spec.name, "budget": int(budget),
            "prediction": prediction, "targets": targets,
            "receipts": [], "committed_rows": 0,
            "identity_sha256": _array_digest(
                identity.detach().cpu().numpy().astype(np.float32, copy=False)),
            "side_sha256": _array_digest(side),
            "initial_carrier_sha256": _array_digest(raw_support),
            "final_carrier_sha256": _array_digest(raw_support),
            "bank_digest_chain": [], "measurement_reasons": {}, "gate_reasons": {},
            "movements": [], "d2_values": [], "committed_label_counts": [0] * cdm_core.GROUP_COUNT,
            "structural_noop": True,
        }
    _require(spec.name in ("F00m", "F01m"), f"unknown cell {spec.name}")
    online = spec.online
    _require(online == (hp is not None),
             "the online cell binds hyperparameters and the frozen cell does not")
    groups = anchor.groups
    bank = stage_p_gate.EvidenceBankP.empty()
    bank_digest_chain = [bank.digest]
    active_raw = raw_support
    identity = decoder.identity(activity, side)
    identity_sha = _array_digest(identity.detach().cpu().numpy().astype(np.float32, copy=False))
    side_sha = _array_digest(side)
    expected_carrier_sha = _array_digest(raw_support)
    predictions: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    measurement_reasons: dict[str, int] = {}
    gate_reasons: dict[str, int] = {}
    movements: list[float] = []
    d2_values: list[float] = []
    committed_label_counts = [0] * cdm_core.GROUP_COUNT
    thresholds = hp.thresholds if hp is not None else None
    row = row_spec if online else None
    channel_sha = cdm_core.channel_order_digest(material.channel_ids)
    for trial in material.trials:
        if trial.metric_starts.size:
            predictions.append(
                decoder.decode(decoder.windows(neural, trial.metric_starts), identity)
            )
        route_reason = "frozen_carrier"
        measurement_digest = None
        moved = False
        decision: Optional[stage_p_gate.BlockDecision] = None
        if online and trial.evidence_eligible:
            if trial.causal_starts.size == 0:
                route_reason = "measurement_rejected:no_complete_window_in_trial"
                measurement_reasons[route_reason] = measurement_reasons.get(route_reason, 0) + 1
            else:
                _require(row is not None and thresholds is not None, "online rollout needs its row spec")
                views, _evidence = _trial_views(
                    decoder, material, trial, activity=activity, side=side, groups=groups,
                )
                measurement = direction_estimator.measure_trial(
                    views, config=config, law=row.law, weight_law=row.weight_law,
                    binding=row.binding, tau_d=thresholds.tau_d,
                )
                measurement_digest = measurement.digest
                counts = np.ascontiguousarray(neural[trial.start_bin : trial.stop_bin], dtype=np.float32)
                scalar_rates = np.ascontiguousarray(
                    counts.mean(axis=0, dtype=np.float64), dtype=np.float64,
                )
                facts = stage_p_gate.TrialFacts(
                    session_id=material.session,
                    trial_id=f"{material.session}:{trial.trial_id}",
                    chronology_position=int(trial.position),
                    scalar_rates=scalar_rates,
                    rate_sha256=_array_digest(scalar_rates),
                    native_counts_sha256=_array_digest(counts),
                    channel_order_sha256=channel_sha,
                    valid_mask_sha256=_array_digest(groups.valid_mask),
                )
                if not measurement.accepted:
                    route_reason = "measurement_rejected:" + str(measurement.reason)
                    measurement_reasons[route_reason] = measurement_reasons.get(route_reason, 0) + 1
                else:
                    decision = stage_p_gate.evaluate_block(
                        anchor, bank, measurement.per_group, facts=facts,
                        measurement_sha256=measurement.digest,
                        rho_M=float(hp.rho_M), c_M=hp.c_M, thresholds=thresholds,
                        block_index=len(bank),
                    )
                    if decision.committed:
                        bank = bank.with_row(decision.row)
                        outcome = trust_region.active_carrier(
                            anchor, decision.coefficients, alpha_M=float(hp.alpha_M), c_M=hp.c_M,
                        )
                        movements.append(outcome.movement_frobenius)
                        d2_values.append(outcome.d2_unprojected)
                        for group, label in enumerate(decision.labels):
                            if label is not None:
                                committed_label_counts[group] += 1
                        if float(hp.alpha_M) == 0.0:
                            _require(
                                np.array_equal(np.asarray(outcome.active_t4, dtype=np.float32), raw_support),
                                "alpha_M = 0 must return the sealed support carrier itself",
                            )
                            route_reason = "block_committed_zero_movement"
                        else:
                            active_raw = np.asarray(outcome.active_t4, dtype=np.float32)
                            side = _side_from_raw(active_raw, material.dataset)
                            identity = decoder.identity(activity, side)
                            identity_sha = _array_digest(
                                identity.detach().cpu().numpy().astype(np.float32, copy=False))
                            side_sha = _array_digest(side)
                            moved = True
                            route_reason = "block_committed"
                    else:
                        route_reason = str(decision.block_reason)
                        gate_reasons[route_reason] = gate_reasons.get(route_reason, 0) + 1
        final_carrier_sha = _array_digest(active_raw)
        receipts.append({
            "trial_id": f"{material.session}:{trial.trial_id}",
            "position": int(trial.position),
            "metric_windows": int(trial.metric_starts.size),
            "causal_windows": int(trial.causal_starts.size),
            "evidence_eligible": bool(trial.evidence_eligible),
            "acc": bool(moved),
            "reason": route_reason,
            "bd": bank.digest,
            "bn": len(bank),
            "cb": expected_carrier_sha,
            "ca": final_carrier_sha,
            "measurement_sha256": measurement_digest,
            "d2": None if decision is None else decision.d2_unprojected,
        })
        expected_carrier_sha = final_carrier_sha
        bank_digest_chain.append(bank.digest)
    _require(bool(predictions), "a session surface must contain at least one window")
    prediction = np.ascontiguousarray(np.concatenate(predictions, axis=0), dtype=np.float32)
    _require(prediction.shape == targets.shape, "per-trial decode/target shape drift")
    return {
        "cell": spec.name, "budget": int(budget),
        "prediction": prediction, "targets": targets,
        "receipts": receipts, "committed_rows": len(bank),
        "identity_sha256": identity_sha,
        "side_sha256": side_sha,
        "initial_carrier_sha256": _array_digest(raw_support),
        "final_carrier_sha256": _array_digest(active_raw),
        "bank_digest_chain": bank_digest_chain,
        "measurement_reasons": measurement_reasons,
        "gate_reasons": gate_reasons,
        "movements": movements, "d2_values": d2_values,
        "committed_label_counts": committed_label_counts,
        "structural_noop": bool(
            (not online) or (hp is not None and float(hp.alpha_M) == 0.0)
        ),
    }


# ---------------------------------------------------------------------------
# The M2 source-fold selection (the sealed stage-P selection law mirrored).
# ---------------------------------------------------------------------------


def _selection_score(
    decoder: M2Decoder, material: SessionMaterial, *, budget: int,
    spec: stage_p_replay.RowSpec, hp: stage_p_replay.CellHyperparameters,
    anchor: M2SupportAnchor, carrier: Mapping[str, Any],
    config: cdm_core.CDMDConfig,
) -> tuple[float, dict[str, Any]]:
    rollout = rollout_session_f(
        decoder, material, budget=budget,
        spec=CellSpec("F01m", "p1_online", True),
        anchor=anchor, carrier=carrier, hp=hp, config=config, row_spec=spec,
    )
    from src.m2_same_query_comparator_v1.core import variance_weighted_r2

    score = float(variance_weighted_r2(rollout["targets"], rollout["prediction"]))
    return score, {
        "bank_final_sha256": rollout["bank_digest_chain"][-1],
        "committed_rows": rollout["committed_rows"],
        "committed_label_counts": list(rollout["committed_label_counts"]),
        "d2_values": [float(item) for item in rollout["d2_values"]],
    }


def _mass_key(value: float) -> str:
    return str(float(value)).replace(".", "p")


def selection_pass(
    decoder: M2Decoder,
    materials: Sequence[SessionMaterial],
    anchors: Mapping[str, M2SupportAnchor],
    carriers: Mapping[str, Mapping[str, Any]],
    *,
    budget: int,
    started: float,
) -> dict[str, Any]:
    """The pre-registered factored within-7 selection, mirrored from stage P."""
    spec = stage_p_replay.ROW_SPECS["P2"]
    stage1: list[dict[str, Any]] = []
    for thresholds in stage_p_replay.enumerate_gate_grid():
        hp = stage_p_replay.CellHyperparameters(
            rho_M=1.0, alpha_M=0.5, c_M=None, thresholds=thresholds,
        )
        per_session: dict[str, float] = {}
        for material in materials:
            score, _detail = _selection_score(
                decoder, material, budget=budget, spec=spec, hp=hp,
                anchor=anchors[material.session], carrier=carriers[material.session],
                config=cdm_core.CDMDConfig(support_budget_m=int(budget)),
            )
            per_session[material.session] = score
            _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                     "the M2 hard timeout fired during selection stage 1")
        stage1.append({
            "vector": thresholds.payload(),
            "fixed": {"rho_M": 1.0, "alpha_M": 0.5},
            "per_session": per_session,
            "mean_r2": float(sum(per_session.values()) / len(per_session)),
        })
    best_stage1 = stage_p_replay.select_vector(stage1)
    vector = best_stage1["vector"]
    selected_thresholds = stage_p_gate.GateThresholds(
        tau_d=float(vector["tau_d_rad"]),
        r_max=int(vector["r_max_repetition_per_direction"]),
        d_min=int(vector["d_min_distinct_directions"]),
        max_mass_relative=float(vector["max_pseudo_mass_relative_to_support_rows"]),
    )
    stage2: list[dict[str, Any]] = []
    stage2_details: dict[str, dict[str, Any]] = {}
    for rho_M, alpha_M in stage_p_replay.enumerate_mass_grid():
        hp = stage_p_replay.CellHyperparameters(
            rho_M=float(rho_M), alpha_M=float(alpha_M), c_M=None,
            thresholds=selected_thresholds,
        )
        per_session: dict[str, float] = {}
        details: dict[str, Any] = {}
        for material in materials:
            score, detail = _selection_score(
                decoder, material, budget=budget, spec=spec, hp=hp,
                anchor=anchors[material.session], carrier=carriers[material.session],
                config=cdm_core.CDMDConfig(support_budget_m=int(budget)),
            )
            per_session[material.session] = score
            details[material.session] = detail
            _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                     "the M2 hard timeout fired during selection stage 2")
        stage2.append({
            "vector": {"rho_M": float(rho_M), "alpha_M": float(alpha_M),
                       "thresholds": selected_thresholds.payload()},
            "per_session": per_session,
            "mean_r2": float(sum(per_session.values()) / len(per_session)),
        })
        stage2_details[f"rho{_mass_key(rho_M)}_alpha{_mass_key(alpha_M)}"] = details
    best_stage2 = stage_p_replay.select_vector(stage2)
    winner_key = (
        f"rho{_mass_key(best_stage2['vector']['rho_M'])}"
        f"_alpha{_mass_key(best_stage2['vector']['alpha_M'])}"
    )
    winning = stage_p_replay.CellHyperparameters(
        rho_M=float(best_stage2["vector"]["rho_M"]),
        alpha_M=float(best_stage2["vector"]["alpha_M"]),
        c_M=None, thresholds=selected_thresholds,
    )
    pooled_d2: list[float] = []
    for material in materials:
        pooled_d2.extend(stage2_details[winner_key][material.session]["d2_values"])
    calibration: dict[str, Any] = {
        "law": "within-7 median of the per-commit unprojected D2 (the stage-O law)",
    }
    if pooled_d2:
        winning = stage_p_replay.CellHyperparameters(
            rho_M=winning.rho_M, alpha_M=winning.alpha_M,
            c_M=trust_region.calibrate_c_M(pooled_d2), thresholds=selected_thresholds,
        )
        calibration.update({
            "c_M": float(winning.c_M),
            "n_d2_values": len(pooled_d2),
            "d2_min": float(np.min(pooled_d2)),
            "d2_median": float(np.median(pooled_d2)),
            "d2_max": float(np.max(pooled_d2)),
        })
    else:
        calibration["note"] = "no committed block under the winning configuration; c_M unset"
    return {
        "law": dict(plan.SELECTION_LAW),
        "budget": int(budget),
        "objective_cell": "P2",
        "stage1_grid_size": len(stage1),
        "stage2_grid_size": len(stage2),
        "stage1_scored": stage1,
        "stage2_scored": stage2,
        "stage1_selected": best_stage1["vector"],
        "stage2_selected": best_stage2["vector"],
        "selected": winning.payload(),
        "c_M_calibration": calibration,
        "tie_break": "first maximum in the pre-registered enumeration order",
        "note": (
            "selection rollouts run SLIM (scores, bank digests and D2 values only); "
            "the governing cells re-derive every row with full receipts"
        ),
    }


# ---------------------------------------------------------------------------
# The G family: the sealed Native-M2 CDM activity memory + the P1 carrier.
# ---------------------------------------------------------------------------


def _g_session_views(
    raw_sessions: Mapping[str, Any], session: str,
) -> dict[str, Any]:
    """The sealed CDM screen's per-session raw views, orchestrated verbatim."""
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import physical as cdm_physical

    return cdm_physical._native_trial_views(raw_sessions[session], session=session)


def g_support_material(
    *, session: str, views: Mapping[str, Any],
) -> dict[str, Any]:
    """The sealed CDM screen's M4 support binding for one session.

    The B3S rows are typed with the SAME ``causal_dual_memory_cell_d_v1``
    module object the sealed CDM screen consumes (the ``tfpd_exploration.src``
    spelling), because ``cdm.ActivityMemory`` narrows by isinstance and the two
    import spellings of one file are distinct module objects.
    """
    from tfpd_exploration.src.causal_dual_memory_cell_d_v1 import core as tfpd_cdm_core
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import physical as cdm_physical
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core as pseudo_core

    _neural, raw_starts, theta30, support_rates_hz30, activities = views
    selected = pseudo_core.select_m4_support(theta30)
    raw_m30_hz, valid_mask = cdm_physical._raw_fixed_ridge_m30(support_rates_hz30, theta30)
    channels = np.arange(activities.shape[2], dtype=np.int64)
    channel_sha = tfpd_cdm_core.channel_order_digest(channels)
    support_b3s = tuple(
        tfpd_cdm_core.B3SInterpolatedSpikeCountTrial(
            activity=np.ascontiguousarray(activities[index], dtype=np.float32),
            session_id=session, trial_id=f"{session}:trial:{index}",
            channel_order_sha256=channel_sha,
        )
        for index in range(plan.ACTIVITY_HORIZON)
    )
    return {
        "selected": selected,
        "raw_m30_hz": raw_m30_hz,
        "valid_mask": valid_mask,
        "support_rates_hz30": support_rates_hz30,
        "theta30": theta30,
        "activities": activities,
        "raw_starts": raw_starts,
        "support_b3s": support_b3s,
        "channels": channels,
    }


def g_query_rows(*, ds: Any, session: str, views: Mapping[str, Any]) -> tuple[Any, ...]:
    """The sealed CDM screen's query-trial partition, orchestrated verbatim."""
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import physical as cdm_physical

    raw_neural, raw_starts, _theta, _rates, activities = views
    return cdm_physical._query_trial_rows(
        ds, session, raw_neural=raw_neural,
        raw_starts=raw_starts, activities=activities,
    )


def rollout_g00m(
    *, torch: Any, model: Any, ds: Any, session: str, views: Mapping[str, Any],
    support: Mapping[str, Any], query_rows: Sequence[Mapping[str, Any]],
    side_mean: np.ndarray, side_std: np.ndarray, device: Any, batch_size: int,
) -> dict[str, Any]:
    """G00m: the sealed ``m4_activity_only`` law, called verbatim."""
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import physical as cdm_physical

    memory = cdm_physical._build_memory(
        system="activity_only", budget=plan.BUDGET_G, selected=support["selected"],
        support_rates_hz30=support["support_rates_hz30"], theta_first30=support["theta30"],
        support_b3s=support["support_b3s"], channel_ids=support["channels"],
        valid_mask=support["valid_mask"],
    )
    return cdm_physical._score_system(
        torch=torch, model=model, dataset=ds, session=session,
        memory=memory, query_rows=query_rows, side_mean=side_mean, side_std=side_std,
        raw_neural=views[0], device=device, batch_size=batch_size,
    )


def rollout_g01m(
    *, torch: Any, model: Any, ds: Any, session: str, views: Mapping[str, Any],
    support: Mapping[str, Any], query_rows: Sequence[Mapping[str, Any]],
    side_mean: np.ndarray, side_std: np.ndarray, device: Any, batch_size: int,
    anchor: Any, hp: stage_p_replay.CellHyperparameters, row_spec: stage_p_replay.RowSpec,
) -> dict[str, Any]:
    """G01m: the activity FIFO advances every completed trial (the sealed V3
    law) while the carrier moves only through the P1 three-factor gate."""
    from tfpd_exploration.src.causal_dual_memory_cell_d_v1 import core as tfpd_cdm_core
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import physical as cdm_physical
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core as pseudo_core
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import physical as shared_physical

    raw_neural, _raw_starts, _theta, _rates, _activities = views
    neural = np.asarray(ds.neural_data[session], dtype=np.float32)
    behavior = np.asarray(ds.covariate_data[session], dtype=np.float32)
    activity_memory = pseudo_core.make_activity_memory(
        support_trials=tuple(support["support_b3s"][index] for index in support["selected"]),
        channel_ids=support["channels"], budget=plan.BUDGET_G,
    )
    carrier, _posterior = pseudo_core.fit_initial_carrier(
        support_rates=support["support_rates_hz30"][support["selected"]],
        direction_indices=pseudo_core.canonical_direction_indices(
            support["theta30"][support["selected"]]),
        channel_ids=support["channels"], valid_mask=support["valid_mask"],
    )
    _require(
        np.array_equal(np.asarray(carrier.active_t4), np.asarray(anchor.support_t4)),
        "the G anchor must bind the sealed initial carrier exactly",
    )
    config = cdm_core.CDMDConfig(support_budget_m=plan.BUDGET_G)
    bank = stage_p_gate.EvidenceBankP.empty()
    bank_digest_chain = [bank.digest]
    carrier_hz = np.asarray(carrier.active_t4, dtype=np.float64)
    expected_sha = _array_digest(carrier_hz)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    starts_joined: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    measurement_reasons: dict[str, int] = {}
    gate_reasons: dict[str, int] = {}
    movements: list[float] = []
    d2_values: list[float] = []
    committed_label_counts = [0] * cdm_core.GROUP_COUNT
    channel_sha = cdm_core.channel_order_digest(support["channels"])
    thresholds = hp.thresholds
    for row in query_rows:
        metric_starts = np.asarray(row["metric_starts"], dtype=np.int64)
        causal_starts = np.asarray(row["causal_starts"], dtype=np.int64)
        activity_stack = activity_memory.stack()
        if metric_starts.size:
            prediction = shared_physical._predict(
                torch=torch, model=model,
                neural_windows=cdm_physical._windows(neural, metric_starts),
                activity=activity_stack,
                raw_t4=np.ascontiguousarray(carrier_hz * plan.BIN_SECONDS, dtype=np.float32),
                side_mean=side_mean, side_std=side_std, device=device, batch_size=batch_size,
            )
            predictions.append(prediction)
            targets.append(np.ascontiguousarray(
                behavior[metric_starts + plan.WINDOW_SIZE - 1], dtype=np.float32))
            starts_joined.append(metric_starts)
        b3s, native, validity = cdm_physical._trial_capabilities(
            session=session, row=row, raw_neural=raw_neural, channel_ids=support["channels"],
        )
        route_reason = "no_complete_window_in_trial"
        decision: Optional[stage_p_gate.BlockDecision] = None
        if causal_starts.size:
            _require(isinstance(validity, tfpd_cdm_core.VelocityValidityEvidence),
                     "nonempty causal trial lacks validity evidence")
            held = cdm_physical._held_predictions(
                torch=torch, model=model,
                neural_windows=cdm_physical._windows(neural, causal_starts),
                activity=activity_stack, carrier_hz=carrier_hz,
                groups=carrier.groups, validity=validity,
                side_mean=side_mean, side_std=side_std, device=device, batch_size=batch_size,
            )
            views_cm = tuple(
                tfpd_cdm_core.CompletedVelocityPrediction(_cm_s(view.velocity), view.validity)
                for view in held
            )
            measurement = direction_estimator.measure_trial(
                views_cm, config=config, law=row_spec.law, weight_law=row_spec.weight_law,
                binding=row_spec.binding, tau_d=thresholds.tau_d,
            )
            scalar_rates = np.ascontiguousarray(
                np.asarray(native.counts, dtype=np.float64).mean(axis=0) / plan.BIN_SECONDS,
                dtype=np.float64,
            )
            facts = stage_p_gate.TrialFacts(
                session_id=session,
                trial_id=str(row["trial_id"]),
                chronology_position=int(row["position"]),
                scalar_rates=scalar_rates,
                rate_sha256=_array_digest(scalar_rates),
                native_counts_sha256=_array_digest(np.asarray(native.counts, dtype=np.float32)),
                channel_order_sha256=channel_sha,
                valid_mask_sha256=_array_digest(carrier.groups.valid_mask),
            )
            if not measurement.accepted:
                route_reason = "measurement_rejected:" + str(measurement.reason)
                measurement_reasons[route_reason] = measurement_reasons.get(route_reason, 0) + 1
            else:
                decision = stage_p_gate.evaluate_block(
                    anchor, bank, measurement.per_group, facts=facts,
                    measurement_sha256=measurement.digest,
                    rho_M=float(hp.rho_M), c_M=hp.c_M, thresholds=thresholds,
                    block_index=len(bank),
                )
                if decision.committed:
                    bank = bank.with_row(decision.row)
                    outcome = trust_region.active_carrier(
                        anchor, decision.coefficients, alpha_M=float(hp.alpha_M), c_M=hp.c_M,
                    )
                    movements.append(outcome.movement_frobenius)
                    d2_values.append(outcome.d2_unprojected)
                    for group, label in enumerate(decision.labels):
                        if label is not None:
                            committed_label_counts[group] += 1
                    if float(hp.alpha_M) == 0.0:
                        route_reason = "block_committed_zero_movement"
                    else:
                        carrier_hz = np.asarray(outcome.active_t4, dtype=np.float64)
                        route_reason = "block_committed"
                else:
                    route_reason = str(decision.block_reason)
                    gate_reasons[route_reason] = gate_reasons.get(route_reason, 0) + 1
        else:
            _require(validity is None, "empty causal trial synthesized validity")
        activity_memory = activity_memory.after_completed_trial(b3s)
        final_sha = _array_digest(carrier_hz)
        receipts.append({
            "trial_id": str(row["trial_id"]),
            "position": int(row["position"]),
            "metric_windows": int(metric_starts.size),
            "causal_windows": int(causal_starts.size),
            "activity_fifo_advanced": True,
            "acc": route_reason == "block_committed",
            "reason": route_reason,
            "bd": bank.digest,
            "bn": len(bank),
            "cb": expected_sha,
            "ca": final_sha,
            "d2": None if decision is None else decision.d2_unprojected,
        })
        expected_sha = final_sha
        bank_digest_chain.append(bank.digest)
    prediction = np.ascontiguousarray(np.concatenate(predictions, axis=0), dtype=np.float32)
    target = np.ascontiguousarray(np.concatenate(targets, axis=0), dtype=np.float32)
    starts = np.ascontiguousarray(np.concatenate(starts_joined), dtype=np.int64)
    expected = surface_starts(ds, session, "external_post30_local")
    _require(np.array_equal(starts, expected), "G01m scored query differs from the post30 authority")
    return {
        "cell": "G01m", "budget": plan.BUDGET_G,
        "prediction": prediction, "targets": target,
        "receipts": receipts, "committed_rows": len(bank),
        "initial_carrier_sha256": _array_digest(np.asarray(carrier.active_t4, dtype=np.float64)),
        "final_carrier_sha256": _array_digest(carrier_hz),
        "bank_digest_chain": bank_digest_chain,
        "measurement_reasons": measurement_reasons,
        "gate_reasons": gate_reasons,
        "movements": movements, "d2_values": d2_values,
        "committed_label_counts": committed_label_counts,
        "window_count": int(starts.size),
        "query_starts_sha256": _array_digest(starts),
        "target_sha256": _array_digest(target),
        "prediction_sha256": _array_digest(prediction),
    }


def build_g_anchor(support: Mapping[str, Any]) -> Any:
    """The stage-O canonical anchor over the sealed CDM-screen carrier."""
    from src.support_anchored_t4_stage_o_v1 import anchor as stage_o_anchor
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core as pseudo_core

    carrier, _posterior = pseudo_core.fit_initial_carrier(
        support_rates=support["support_rates_hz30"][support["selected"]],
        direction_indices=pseudo_core.canonical_direction_indices(
            support["theta30"][support["selected"]]),
        channel_ids=support["channels"], valid_mask=support["valid_mask"],
    )
    return stage_o_anchor.SupportAnchor.from_labeled_support(
        groups=carrier.groups,
        support_trial_rates=np.ascontiguousarray(
            support["support_rates_hz30"][support["selected"]], dtype=np.float64),
        support_direction_indices=pseudo_core.canonical_direction_indices(
            support["theta30"][support["selected"]]),
        support_t4=carrier.active_t4,
    ), carrier
