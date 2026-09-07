"""The SUA replay: F00s/F01s on the frozen matched-scorer runtime.

Everything the P1 law needs is ORCHESTRATED from the frozen packages:

* ``direction_estimator.measure_trial``  -- the deployable pseudo-direction
  measurement (law/weight/binding per row spec), verbatim;
* ``stage_p_gate.evaluate_block`` / ``EvidenceBankP`` -- the three-factor
  commit gate and the append-only evidence bank, verbatim;
* ``trust_region.active_carrier``        -- the anchored trust-region movement;
* ``stage_p_replay.enumerate_gate_grid`` / ``enumerate_mass_grid`` /
  ``select_vector`` / ``CellHyperparameters`` / ``ROW_SPECS`` -- the sealed
  selection grids, tie-break law and row specs, verbatim.

This module contributes only the SUA surface bindings: the sealed matched-
scorer runtime (V9 ``shared_t4`` checkpoints, per-view normalizers, the V3R2
session loader), the trial partition of the post-50 query stream, the sealed
M4/M30 ridge carrier laws, and the ``SUASupportAnchor``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from src.causal_dual_memory_cell_d_v1 import core as cdm_core
from src.support_anchored_t4_stage_o_v1 import trust_region
from src.support_anchored_t4_stage_p_v1 import direction_estimator
from src.support_anchored_t4_stage_p_v1 import gate as stage_p_gate
from src.support_anchored_t4_stage_p_v1 import replay as stage_p_replay

from . import plan
from .anchor import SUASupportAnchor, build_groups


class SuaReplayError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SuaReplayError(message)


def _array_digest(value: np.ndarray) -> str:
    return cdm_core.array_digest(np.ascontiguousarray(value))


# ---------------------------------------------------------------------------
# Frozen session material (shared across seeds and cells).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryTrial:
    position: int
    start_bin: int
    stop_bin: int
    window_starts: np.ndarray
    rate_column: int


@dataclass(frozen=True)
class SessionMaterial:
    session_id: str
    record: Any
    activity: np.ndarray
    side_mean: np.ndarray
    side_std: np.ndarray
    behavior_mean: np.ndarray
    behavior_std: np.ndarray
    trials: tuple[Mapping[str, Any], ...]
    query_trials: tuple[QueryTrial, ...]
    rates_hz: np.ndarray
    channel_ids: np.ndarray

    @property
    def targets(self) -> np.ndarray:
        starts = np.concatenate([trial.window_starts for trial in self.query_trials]) \
            if self.query_trials else np.asarray([], dtype=np.int64)
        indices = starts + plan.HISTORY_BINS - 1
        return np.ascontiguousarray(self.record.behavior[indices], dtype=np.float32)

    @property
    def query_starts(self) -> np.ndarray:
        return np.concatenate([trial.window_starts for trial in self.query_trials]) \
            if self.query_trials else np.asarray([], dtype=np.int64)


def build_session_material(
    *, repo_root: Path, nwb_path: Path, owners: Mapping[str, Any],
    behavior_mean: np.ndarray, behavior_std: np.ndarray,
    side_mean: np.ndarray, side_std: np.ndarray,
) -> SessionMaterial:
    from sua_exploration.mc_maze import subm_co_scorer_adapter_parity_v3 as parity_v3
    from sua_exploration.mc_maze import subm_co_three_arm_v9_runtime as runtime
    from sua_exploration.mc_maze.unit_side_features import _pool_trial_rate_matrix

    record, rebuilt, _evidence = runtime._build_base(
        repo_root=repo_root, nwb_path=nwb_path, view=plan.VIEW,
        behavior_mean=behavior_mean, behavior_std=behavior_std, owners=owners,
    )
    trials = owners["list_datamodule_rewarded_trials"](
        nwb_path, bin_size_ms=parity_v3.BIN_SIZE_MS, window_size=parity_v3.HISTORY_BINS,
        trial_result_filter="R",
    )
    _require(len(trials) > plan.SUPPORT_TRIALS, "session lacks a post-50 query stream")
    # The loader's query windows are exactly the whole-window starts inside
    # each rewarded trial after the 50-trial calibration pool; rebuild that
    # partition and require it to be the record's authority.
    query_trials: list[QueryTrial] = []
    for position in range(plan.SUPPORT_TRIALS, len(trials)):
        trial = trials[position]
        start, stop = int(trial["start"]), int(trial["stop"])
        _require(stop > start + plan.HISTORY_BINS - 1, "query trial shorter than one window")
        window_starts = np.arange(start, stop - plan.HISTORY_BINS + 1, dtype=np.int64)
        query_trials.append(QueryTrial(
            position=position, start_bin=start, stop_bin=stop,
            window_starts=window_starts, rate_column=position,
        ))
    _require(bool(query_trials), "session has an empty post-50 query stream")
    rebuilt_starts = np.concatenate([trial.window_starts for trial in query_trials])
    _require(np.array_equal(rebuilt_starts, np.asarray(record.valid_starts, dtype=np.int64)),
             "the rebuilt query partition differs from the runtime's valid_starts authority")
    rates_hz, unit_count = _pool_trial_rate_matrix(nwb_path, trials)
    _require(unit_count == int(record.neural.shape[1]), "rate-matrix channel order drift")
    return SessionMaterial(
        session_id=str(record.name),
        record=record,
        activity=np.ascontiguousarray(rebuilt, dtype=np.float32),
        side_mean=np.asarray(side_mean, dtype=np.float32),
        side_std=np.asarray(side_std, dtype=np.float32),
        behavior_mean=np.asarray(behavior_mean, dtype=np.float32),
        behavior_std=np.asarray(behavior_std, dtype=np.float32),
        trials=tuple(trials),
        query_trials=tuple(query_trials),
        rates_hz=rates_hz,
        channel_ids=np.asarray(record.channel_ids, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# The sealed carrier laws.
# ---------------------------------------------------------------------------


def m4_support(theta_first30: np.ndarray) -> np.ndarray:
    from src.sua_paired_activity_budget_screen_v1 import core as paired_core

    return paired_core.select_m4_support(theta_first30)


def m30_support(theta_first30: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta_first30, dtype=np.float64)
    _require(theta.shape == (plan.ACTIVITY_HORIZON,), "M30 cue pool must be first-30")
    finite = np.flatnonzero(np.isfinite(theta)).astype(np.int64)
    _require(finite.size >= 3, "M30 support has fewer than three directional trials")
    return np.ascontiguousarray(finite)


def sealed_carrier(
    material: SessionMaterial, budget: int,
) -> dict[str, Any]:
    """The sealed M4 ridge (D-opt four) / M30 ridge (first-30 finite) carrier."""
    from src.calibration_budget_comparators_v1 import fit_ridge_t4

    theta_first30 = np.asarray([
        float(trial["target_dir"]) if trial.get("target_dir") is not None else np.nan
        for trial in material.trials[: plan.ACTIVITY_HORIZON]
    ], dtype=np.float64)
    _require(theta_first30.shape == (plan.ACTIVITY_HORIZON,), "first-30 cue pool drift")
    if budget == plan.M4:
        selected = m4_support(theta_first30)
    elif budget == plan.M30:
        selected = m30_support(theta_first30)
    else:
        raise SuaReplayError(f"unsupported carrier budget {budget}")
    theta = np.ascontiguousarray(theta_first30[selected], dtype=np.float64)
    _require(np.isfinite(theta).all(), "selected support includes a non-directional trial")
    rates = np.ascontiguousarray(material.rates_hz[:, selected].T, dtype=np.float64)
    raw, fit = fit_ridge_t4(rates, theta, normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA)
    side = np.ascontiguousarray(
        (raw - material.side_mean) / material.side_std, dtype=np.float32,
    )
    _require(side.shape == (material.record.neural.shape[1], 4) and np.isfinite(side).all(),
             "normalized sealed carrier drift")
    return {
        "budget": int(budget),
        "selected": selected,
        "theta": theta,
        "rates": rates,
        "raw_t4": np.ascontiguousarray(raw, dtype=np.float32),
        "side": side,
        "fit_evidence": {
            key: fit[key] for key in ("normalized_lambda", "design_rank", "design_condition", "gcv")
        },
        "selected_sha256": _array_digest(selected),
        "theta_sha256": _array_digest(theta),
        "rates_sha256": _array_digest(rates),
        "raw_t4_sha256": _array_digest(raw),
        "side_sha256": _array_digest(side),
    }


# ---------------------------------------------------------------------------
# The frozen decode path (identity cache keyed by side bytes and channels).
# ---------------------------------------------------------------------------


class SuaDecoder:
    """Cached-identity decode over one frozen model, deterministic settings."""

    def __init__(self, torch: Any, model: Any, device: Any, *, batch_size: int) -> None:
        self.torch = torch
        self.model = model
        self.device = device
        self.batch_size = int(batch_size)
        self.student = model.student
        self._identity_cache: dict[tuple[bytes, bytes], Any] = {}
        for parameter in model.parameters():
            _require(parameter.requires_grad is False, "frozen evaluator exposes trainable parameters")

    def identity(self, activity: np.ndarray, side: np.ndarray, keep: Optional[np.ndarray] = None) -> Any:
        channels = np.arange(activity.shape[2], dtype=np.int64) if keep is None else np.asarray(keep, dtype=np.int64)
        activity_np = activity if keep is None else np.ascontiguousarray(activity[:, :, channels])
        side_np = side if keep is None else np.ascontiguousarray(side[channels])
        key = (activity_np.tobytes(), side_np.tobytes())
        cached = self._identity_cache.get(key)
        if cached is not None:
            return cached
        calibration = self.torch.from_numpy(np.ascontiguousarray(activity_np, dtype=np.float32)).unsqueeze(0).to(self.device)
        side_tensor = self.torch.from_numpy(np.ascontiguousarray(side_np, dtype=np.float32)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            identity = self.student.compute_identity(calibration, side_features=side_tensor)
        _require(tuple(identity.shape) == (1, side_np.shape[0], plan.HISTORY_BINS),
                 "cached identity shape drift")
        self._identity_cache[key] = identity
        return identity

    def decode(self, neural: np.ndarray, identity: Any) -> np.ndarray:
        values = np.ascontiguousarray(neural, dtype=np.float32)
        _require(values.ndim == 3 and values.shape[1] == plan.HISTORY_BINS, "decode window shape drift")
        outputs: list[np.ndarray] = []
        with self.torch.inference_mode():
            for offset in range(0, values.shape[0], self.batch_size):
                chunk = values[offset : offset + self.batch_size]
                tensor = self.torch.from_numpy(chunk).to(self.device)
                raw = self.student.decode_with_identity(tensor, identity)
                _require(bool(self.torch.isfinite(raw).all().item()), "decode produced nonfinite output")
                outputs.append(
                    raw[:, -1, :].detach().cpu().numpy().astype(np.float32, copy=False)
                    / plan.BEHAVIOR_SCALE
                )
        return np.ascontiguousarray(np.concatenate(outputs, axis=0), dtype=np.float32)

    def clear_cache(self) -> None:
        self._identity_cache.clear()

    def windows(self, material: SessionMaterial, starts: np.ndarray, keep: Optional[np.ndarray] = None) -> np.ndarray:
        indices = np.asarray(starts, dtype=np.int64)[:, None] + np.arange(plan.HISTORY_BINS, dtype=np.int64)[None, :]
        neural = material.record.neural[indices]
        if keep is None:
            return np.ascontiguousarray(neural, dtype=np.float32)
        return np.ascontiguousarray(neural[:, :, np.asarray(keep, dtype=np.int64)], dtype=np.float32)

    def physical_velocity(
        self, prediction: np.ndarray, material: SessionMaterial,
    ) -> np.ndarray:
        standardized = np.asarray(prediction, dtype=np.float64)
        restored = standardized * material.behavior_std.astype(np.float64)[None, :] \
            + material.behavior_mean.astype(np.float64)[None, :]
        _require(np.isfinite(restored).all(), "physical velocity restoration became nonfinite")
        return np.ascontiguousarray(restored, dtype=np.float64)


# ---------------------------------------------------------------------------
# The per-trial P1 loop.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellSpec:
    name: str
    carrier_law: str
    online: bool


CELL_SPECS = (
    CellSpec("F00s_anchor", "frozen_t4_sealed_batching", False),
    CellSpec("F00s", "frozen_t4_per_trial", False),
    CellSpec("F01s", "p1_online", True),
)


def _side_from_raw(raw_t4: np.ndarray, material: SessionMaterial) -> np.ndarray:
    side = np.ascontiguousarray(
        (np.asarray(raw_t4, dtype=np.float32) - material.side_mean) / material.side_std,
        dtype=np.float32,
    )
    _require(np.isfinite(side).all(), "normalized active carrier became nonfinite")
    return side


def _trial_views(
    decoder: SuaDecoder, material: SessionMaterial, trial: QueryTrial,
    activity: np.ndarray, side: np.ndarray, groups: cdm_core.ComplementaryGroups,
) -> tuple[tuple[Any, ...], int]:
    """The four held-group physical-velocity views of one completed trial."""
    views = []
    chunks = 0
    for group in range(cdm_core.GROUP_COUNT):
        keep = np.flatnonzero(~np.asarray(groups.held_mask(group), dtype=bool)).astype(np.int64)
        identity = decoder.identity(activity, side, keep=keep)
        prediction = decoder.decode(decoder.windows(material, trial.window_starts, keep=keep), identity)
        chunks += int(np.ceil(trial.window_starts.size / decoder.batch_size))
        velocity = decoder.physical_velocity(prediction, material)
        validity = cdm_core.VelocityValidityEvidence(
            valid_mask=np.ones(trial.window_starts.size, dtype=np.bool_),
            session_id=material.session_id,
            trial_id=f"{material.session_id}:trial:{trial.position}",
            prediction_interval_start_bin=int(trial.window_starts[0] + plan.HISTORY_BINS - 1),
            prediction_interval_stop_bin=int(trial.window_starts[-1] + plan.HISTORY_BINS),
        )
        views.append(cdm_core.CompletedVelocityPrediction(velocity, validity))
    return tuple(views), chunks


def rollout_session(
    decoder: SuaDecoder,
    material: SessionMaterial,
    *,
    budget: int,
    spec: CellSpec,
    anchor: SUASupportAnchor,
    carrier: Mapping[str, Any],
    hp: Optional[stage_p_replay.CellHyperparameters],
    config: cdm_core.CDMDConfig,
    row_spec: Optional[stage_p_replay.RowSpec] = None,
    slim: bool = False,
) -> dict[str, Any]:
    """Run one cell over one session; F00s_anchor uses the sealed whole-session batching."""
    activity = material.activity
    _require(activity.shape[0] == plan.ACTIVITY_HORIZON, "activity calibration must be first-30")
    targets = material.targets
    side = np.asarray(carrier["side"], dtype=np.float32)
    raw_support = np.asarray(carrier["raw_t4"], dtype=np.float32)
    if spec.name == "F00s_anchor":
        identity = decoder.identity(activity, side)
        prediction = decoder.decode(decoder.windows(material, material.query_starts), identity)
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
    _require(spec.name in ("F00s", "F01s"), f"unknown cell {spec.name}")
    online = spec.online
    _require(online == (hp is not None), "the online cell binds hyperparameters and the frozen cell does not")
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
    for trial in material.query_trials:
        windows = decoder.windows(material, trial.window_starts)
        prediction = decoder.decode(windows, identity)
        predictions.append(prediction)
        route_reason = "frozen_carrier"
        measurement_digest = None
        moved = False
        if online:
            _require(row is not None and thresholds is not None, "online rollout needs its row spec")
            views, _chunks = _trial_views(
                decoder, material, trial, activity=activity, side=side, groups=groups,
            )
            measurement = direction_estimator.measure_trial(
                views, config=config, law=row.law, weight_law=row.weight_law,
                binding=row.binding, tau_d=thresholds.tau_d,
            )
            measurement_digest = measurement.digest
            counts = np.ascontiguousarray(
                material.record.neural[trial.start_bin : trial.stop_bin], dtype=np.float32,
            )
            scalar_rates = np.ascontiguousarray(
                material.rates_hz[:, trial.rate_column], dtype=np.float64,
            )
            facts = stage_p_gate.TrialFacts(
                session_id=material.session_id,
                trial_id=f"{material.session_id}:trial:{trial.position}",
                chronology_position=int(trial.position),
                scalar_rates=scalar_rates,
                rate_sha256=_array_digest(scalar_rates),
                native_counts_sha256=_array_digest(counts),
                channel_order_sha256=cdm_core.channel_order_digest(material.channel_ids),
                valid_mask_sha256=_array_digest(groups.valid_mask),
            )
            decision: Optional[stage_p_gate.BlockDecision] = None
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
                        side = _side_from_raw(active_raw, material)
                        identity = decoder.identity(activity, side)
                        identity_sha = _array_digest(identity.detach().cpu().numpy().astype(np.float32, copy=False))
                        side_sha = _array_digest(side)
                        moved = True
                        route_reason = "block_committed"
                else:
                    route_reason = str(decision.block_reason)
                    gate_reasons[route_reason] = gate_reasons.get(route_reason, 0) + 1
            final_carrier_sha = _array_digest(active_raw)
            receipts.append({
                "trial_id": facts.trial_id,
                "position": int(trial.position),
                "windows": int(trial.window_starts.size),
                "acc": bool(moved),
                "reason": route_reason,
                "bd": bank.digest,
                "bn": len(bank),
                "cb": expected_carrier_sha,
                "ca": final_carrier_sha,
                "measurement_sha256": measurement_digest,
                "d2": None if decision is None else decision.d2_unprojected,
                **({} if slim else {"bank_digest_chain_length": len(bank_digest_chain)}),
            })
            expected_carrier_sha = final_carrier_sha
            bank_digest_chain.append(bank.digest)
        else:
            receipts.append({
                "trial_id": f"{material.session_id}:trial:{trial.position}",
                "position": int(trial.position),
                "windows": int(trial.window_starts.size),
                "acc": False,
                "reason": route_reason,
                "cb": expected_carrier_sha,
                "ca": expected_carrier_sha,
            })
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
# The SUA source-fold selection (the sealed stage-P selection law mirrored).
# ---------------------------------------------------------------------------


def _selection_score(
    decoder: SuaDecoder, material: SessionMaterial, *, budget: int,
    spec: stage_p_replay.RowSpec, hp: stage_p_replay.CellHyperparameters,
    anchor: SUASupportAnchor, carrier: Mapping[str, Any],
    config: cdm_core.CDMDConfig,
) -> tuple[float, dict[str, Any]]:
    rollout = rollout_session(
        decoder, material, budget=budget,
        spec=CellSpec("F01s", "p1_online", True),
        anchor=anchor, carrier=carrier, hp=hp, config=config, row_spec=spec, slim=True,
    )
    from src.sua_paired_activity_budget_screen_v1.core import variance_weighted_r2

    score = float(variance_weighted_r2(rollout["targets"], rollout["prediction"]))
    return score, {
        "bank_final_sha256": rollout["bank_digest_chain"][-1],
        "committed_rows": rollout["committed_rows"],
        "committed_label_counts": list(rollout["committed_label_counts"]),
        "d2_values": [float(item) for item in rollout["d2_values"]],
    }


def selection_pass(
    decoder: SuaDecoder,
    materials: Sequence[SessionMaterial],
    anchors: Mapping[str, SUASupportAnchor],
    carriers: Mapping[str, Mapping[str, Any]],
    *,
    budget: int,
    started: float,
) -> dict[str, Any]:
    """The pre-registered factored within-6 selection, mirrored from stage P."""
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
                anchor=anchors[material.session_id], carrier=carriers[material.session_id],
                config=cdm_core.CDMDConfig(support_budget_m=int(budget)),
            )
            per_session[material.session_id] = score
            _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                     "the SUA hard timeout fired during selection stage 1")
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
                anchor=anchors[material.session_id], carrier=carriers[material.session_id],
                config=cdm_core.CDMDConfig(support_budget_m=int(budget)),
            )
            per_session[material.session_id] = score
            details[material.session_id] = detail
            _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                     "the SUA hard timeout fired during selection stage 2")
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
        pooled_d2.extend(stage2_details[winner_key][material.session_id]["d2_values"])
    calibration: dict[str, Any] = {
        "law": "within-6 median of the per-commit unprojected D2 (the stage-O law)",
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


def _mass_key(value: float) -> str:
    return str(float(value)).replace(".", "p")
