"""Physical driver of the M2 Stage-O oracle carrier headroom test.

CPU-only, inference-only, zero training, one launch.  The receipt law mirrors
the governing run and the sealed alpha diagnostic: ``attempt.json`` (reserved
before any data or model access), ``replay.json`` (the whole grid), and
``terminal.json`` (the composed verdict), each published O_EXCL 0444 with a
sha256 sidecar.  Nothing under any frozen result root or frozen package is
created or modified; the governing machinery is reused BY IMPORT only.

The O0m baseline is the sealed F00m law replayed through the frozen runtime;
the O2m oracle loop is route-owned and orchestrates the frozen Stage-O laws
(:class:`support_anchored_t4_stage_o_v1.block_refit.EvidenceRow` /
``EvidenceBank`` / ``refit_from_anchor`` and
:func:`support_anchored_t4_stage_o_v1.trust_region.active_carrier`) over the
frozen M2 anchor, with the TRUE direction measured by the frozen
``pseudo_direction_from_velocity`` law on the query stream's own covariates.
"""

from __future__ import annotations

import json
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from src.causal_dual_memory_cell_d_v1 import core as cdm_core
from src.cdm_p1_m2_local_v1 import gates as governing_gates
from src.cdm_p1_m2_local_v1 import physical as governing_physical
from src.cdm_p1_m2_local_v1 import replay as governing_replay
from src.cdm_p1_m2_local_v1.anchor import M2SupportAnchor, build_groups
from src.support_anchored_t4_stage_o_v1 import block_refit as stage_o_block_refit
from src.support_anchored_t4_stage_o_v1 import trust_region

from . import plan


class M2CarrierOracleError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M2CarrierOracleError(message)


_sha256_file = governing_physical._sha256_file
_verify_sidecar = governing_physical._verify_sidecar
_publish = governing_physical._publish
_bind_namespaces = governing_physical._bind_namespaces
_load_sealed_same_query_rows = governing_physical._load_sealed_same_query_rows
_array_digest = governing_replay._array_digest
_side_from_raw = governing_replay._side_from_raw
_cm_s = governing_replay._cm_s


def _validate_environment() -> None:
    """The CPU isolation law: no GPU may even be visible to this process."""
    _require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "",
             "this run is CPU-only: CUDA_VISIBLE_DEVICES must be empty "
             "(GPU 0 and GPU 1 are owned by other live routes)")
    _require(os.environ.get("PYTHONNOUSERSITE") == "1",
             "PYTHONNOUSERSITE=1 is required (the sealed environment law)")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        _require(os.environ.get(variable) == str(int(plan.Torch_THREADS)),
                 f"{variable}={plan.Torch_THREADS} is required (the CPU "
                 f"thread cap that keeps the neighbouring GPU training "
                 f"routes unaffected)")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Sealed-foundation readers.
# ---------------------------------------------------------------------------


def _load_sealed_governing(repo_root: Path) -> dict[str, Any]:
    root = repo_root / plan.GOVERNING_ROOT_RELATIVE
    replay_digest = _verify_sidecar(root / "replay.json")
    terminal_digest = _verify_sidecar(root / "terminal.json")
    _verify_sidecar(root / "attempt.json")
    _require(replay_digest == plan.GOVERNING_REPLAY_SHA256,
             "the sealed governing replay receipt drifted")
    _require(terminal_digest == plan.GOVERNING_TERMINAL_SHA256,
             "the sealed governing terminal receipt drifted")
    _require(_sha256_file(root / "attempt.json") == plan.GOVERNING_ATTEMPT_SHA256,
             "the sealed governing attempt receipt drifted")
    _require(_sha256_file(repo_root / plan.AUDIT_RELATIVE) == plan.AUDIT_SHA256,
             "the sealed M2 prerequisite audit drifted")
    replay_payload = json.loads((root / "replay.json").read_text(encoding="utf-8"))
    terminal_payload = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    _require(replay_payload.get("status") == "REPLAY_COMPLETE"
             and terminal_payload.get("status") == "TERMINAL",
             "the governing run is not sealed terminal")
    return {"replay": replay_payload, "terminal": terminal_payload}


def _sealed_matrix_rows(sealed: Mapping[str, Any], cell: str
                        ) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {
        (surface, str(budget), session): row
        for surface, by_budget in sealed["replay"]["matrix"].items()
        for budget, by_cell in by_budget.items()
        for session, row in by_cell[cell].items()
    }


# ---------------------------------------------------------------------------
# The read-only Stage-O anchor adapter (disclosed deviation).
# ---------------------------------------------------------------------------


class StageOAnchorAdapter:
    """A read-only duck-typed view of :class:`M2SupportAnchor`.

    The frozen Stage-O ``block_refit.refit_from_anchor`` reads
    ``anchor.statistics.counts[group]`` for its NON-GOVERNING coverage rows;
    the M2 anchor carries the same canonical support counts as
    ``per_group[g].counts``.  This adapter supplies exactly that view and
    delegates everything else to the immutable M2 anchor.  It holds no state
    of its own and edits nothing.
    """

    def __init__(self, m2_anchor: M2SupportAnchor) -> None:
        self._anchor = m2_anchor
        self.groups = m2_anchor.groups
        self.per_group = m2_anchor.per_group
        self.support_t4 = m2_anchor.support_t4
        self.statistics = SimpleNamespace(
            counts=tuple(np.asarray(item.counts, dtype=np.int64)
                         for item in m2_anchor.per_group),
        )

    def coefficients(self, t4: np.ndarray) -> tuple[np.ndarray, ...]:
        return self._anchor.coefficients(t4)

    def rebuild_t4(self, coefficient_blocks: Sequence[np.ndarray]) -> np.ndarray:
        return self._anchor.rebuild_t4(coefficient_blocks)

    @property
    def digest(self) -> str:
        return self._anchor.digest

    @property
    def a0_b0_digest(self) -> str:
        return self._anchor.a0_b0_digest


# ---------------------------------------------------------------------------
# The frozen true-direction law (target-label leakage, diagnostic only).
# ---------------------------------------------------------------------------


def true_direction_for_trial(
    covariates: np.ndarray, trial: governing_replay.QueryTrial, *,
    config: cdm_core.CDMDConfig,
) -> tuple[cdm_core.PseudoDirection, Mapping[str, Any]]:
    """The TRUE completed-trial direction of one query trial (the frozen law).

    The query stream's own covariate rows (``finger_vel``, SI m/s) at the
    endpoints of the trial's complete causal windows are restored to the
    frozen cm/s view (the governing VELOCITY_UNIT_LAW) and run through the
    frozen ``pseudo_direction_from_velocity`` law under the session carrier's
    own config -- the exact law the decoded measurement uses, on the exact
    same integration domain, but on the TRUE rows instead of the decoded
    ones.  This reads the target labels: diagnostic leakage only.
    """
    starts = np.asarray(trial.causal_starts, dtype=np.int64)
    _require(starts.size >= 1, "the true-direction law needs a complete causal window")
    endpoints = np.ascontiguousarray(starts + governing_replay.plan.WINDOW_SIZE - 1,
                                     dtype=np.int64)
    rows = np.ascontiguousarray(
        np.asarray(covariates, dtype=np.float32)[endpoints], dtype=np.float32,
    )
    _require(rows.ndim == 2 and rows.shape == (starts.size, 2)
             and np.isfinite(rows).all(), "true covariate rows topology drift")
    validity = cdm_core.VelocityValidityEvidence(
        valid_mask=np.ones(starts.size, dtype=np.bool_),
        session_id=f"{trial.trial_id}:true",
        trial_id=trial.trial_id,
        prediction_interval_start_bin=int(starts[0] + governing_replay.plan.WINDOW_SIZE - 1),
        prediction_interval_stop_bin=int(starts[-1] + governing_replay.plan.WINDOW_SIZE),
    )
    view = cdm_core.CompletedVelocityPrediction(_cm_s(rows), validity)
    direction = cdm_core.pseudo_direction_from_velocity(
        view.velocity, view.validity.valid_mask, config=config,
    )
    meta = {"true_rows": int(starts.size)}
    return direction, meta


def _direction_payload(trial: governing_replay.QueryTrial,
                       direction: cdm_core.PseudoDirection,
                       meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trial_id": f"{trial.trial_id}",
        "position": int(trial.position),
        "accepted": bool(direction.accepted),
        "reason": None if direction.reason is None else direction.reason.value,
        "theta_raw_rad": direction.theta_raw_rad,
        "theta_index": None if direction.theta_index is None else int(direction.theta_index),
        "theta_canonical_rad": direction.theta_canonical_rad,
        "canonical_distance_rad": direction.canonical_distance_rad,
        "movement_bins": int(direction.movement_bins),
        "displacement_norm": direction.displacement_norm,
        "mean_speed": direction.mean_speed,
        "true_rows": int(meta["true_rows"]),
        "law": "covariate_rows_at_causal_endpoints_x100_pseudo_direction_from_velocity",
    }


def _direction_digest(payload: Mapping[str, Any]) -> str:
    return cdm_core.sha256_bytes(cdm_core.canonical_json_bytes(dict(payload)))


# ---------------------------------------------------------------------------
# The O2m oracle loop (always-commit + trust-region projection only).
# ---------------------------------------------------------------------------


def rollout_session_oracle(
    decoder: governing_replay.M2Decoder,
    material: governing_replay.SessionMaterial,
    *,
    budget: int,
    anchor: M2SupportAnchor,
    carrier: Mapping[str, Any],
    alpha_M: float,
    c_M: Optional[float],
    config: cdm_core.CDMDConfig,
) -> dict[str, Any]:
    """Run one O2m cell over one session under the Stage-O amendment law.

    Mirrors the frozen ``rollout_session_f`` F01m loop (same activity state,
    same decode path, same receipt binding) with exactly two replacements:
    the measurement is the TRUE direction (no group forwards), and the commit
    is the Stage-O anchored block refit with ALWAYS-COMMIT + trust-region
    projection only.  ``alpha_M = 0`` is the exact no-op identity.
    """
    activity = np.asarray(carrier["activity"], dtype=np.float32)
    _require(activity.shape[0] == int(carrier["budget"]),
             "activity budget must equal the carrier budget (the sealed static law)")
    targets = material.targets
    side = np.asarray(carrier["side"], dtype=np.float32)
    raw_support = np.asarray(carrier["raw_t4"], dtype=np.float32)
    neural = material.neural
    covariates = np.asarray(material.dataset.covariate_data[material.session],
                            dtype=np.float32)
    groups = anchor.groups
    stage_o_anchor = StageOAnchorAdapter(anchor)
    channel_sha = cdm_core.channel_order_digest(material.channel_ids)
    valid_mask_sha = _array_digest(groups.valid_mask)
    rho_M = float(plan.RHO_M)

    bank = stage_o_block_refit.EvidenceBank.empty()
    bank_digest_chain = [bank.digest]
    active_raw = raw_support
    identity = decoder.identity(activity, side)
    identity_sha = _array_digest(identity.detach().cpu().numpy().astype(np.float32, copy=False))
    side_sha = _array_digest(side)
    expected_carrier_sha = _array_digest(raw_support)
    predictions: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    measurement_reasons: dict[str, int] = {}
    movements: list[float] = []
    d2_values: list[float] = []
    projected_count = 0
    for trial in material.trials:
        if trial.metric_starts.size:
            predictions.append(
                decoder.decode(decoder.windows(neural, trial.metric_starts), identity)
            )
        route_reason = "frozen_carrier"
        measurement_digest: Optional[str] = None
        moved = False
        outcome: Optional[trust_region.TrustRegionOutcome] = None
        row_digest: Optional[str] = None
        if trial.evidence_eligible:
            if trial.causal_starts.size == 0:
                route_reason = "measurement_rejected:no_complete_window_in_trial"
                measurement_reasons[route_reason] = measurement_reasons.get(route_reason, 0) + 1
            else:
                direction, meta = true_direction_for_trial(
                    covariates, trial, config=config,
                )
                payload = _direction_payload(trial, direction, meta)
                measurement_digest = _direction_digest(payload)
                if not direction.accepted:
                    route_reason = "measurement_rejected:" + str(payload["reason"])
                    measurement_reasons[route_reason] = measurement_reasons.get(route_reason, 0) + 1
                else:
                    counts = np.ascontiguousarray(
                        neural[trial.start_bin : trial.stop_bin], dtype=np.float32)
                    scalar_rates = np.ascontiguousarray(
                        counts.mean(axis=0, dtype=np.float64), dtype=np.float64,
                    )
                    index = int(direction.theta_index)  # type: ignore[arg-type]
                    row = stage_o_block_refit.EvidenceRow(
                        session_id=material.session,
                        trial_id=f"{material.session}:{trial.trial_id}",
                        chronology_position=int(trial.position),
                        block_index=len(bank),
                        direction_indices=tuple(index for _ in range(cdm_core.GROUP_COUNT)),
                        theta_raw_rad=tuple(float(direction.theta_raw_rad)  # type: ignore[arg-type]
                                            for _ in range(cdm_core.GROUP_COUNT)),
                        canonical_distance_rad=tuple(
                            float(direction.canonical_distance_rad)  # type: ignore[arg-type]
                            for _ in range(cdm_core.GROUP_COUNT)),
                        movement_bins=tuple(int(direction.movement_bins)  # type: ignore[arg-type]
                                            for _ in range(cdm_core.GROUP_COUNT)),
                        displacement_norm=tuple(
                            float(direction.displacement_norm)  # type: ignore[arg-type]
                            for _ in range(cdm_core.GROUP_COUNT)),
                        mean_speed=tuple(
                            float(direction.mean_speed)  # type: ignore[arg-type]
                            for _ in range(cdm_core.GROUP_COUNT)),
                        scalar_rates=scalar_rates,
                        rate_sha256=_array_digest(scalar_rates),
                        native_counts_sha256=_array_digest(counts),
                        channel_order_sha256=channel_sha,
                        valid_mask_sha256=valid_mask_sha,
                    )
                    row_digest = row.digest
                    bank = bank.with_row(row)
                    refit = stage_o_block_refit.refit_from_anchor(
                        stage_o_anchor, bank, rho_M=rho_M,
                    )
                    outcome = trust_region.active_carrier(
                        stage_o_anchor, refit.coefficients,
                        alpha_M=float(alpha_M), c_M=c_M,
                    )
                    movements.append(outcome.movement_frobenius)
                    d2_values.append(outcome.d2_unprojected)
                    projected_count += 1 if outcome.projected else 0
                    if float(alpha_M) == 0.0:
                        _require(
                            np.array_equal(
                                np.asarray(outcome.active_t4, dtype=np.float32), raw_support),
                            "alpha_M = 0 must return the sealed support carrier itself",
                        )
                        route_reason = "always_commit_zero_movement"
                    else:
                        active_raw = np.asarray(outcome.active_t4, dtype=np.float32)
                        side = _side_from_raw(active_raw, material.dataset)
                        identity = decoder.identity(activity, side)
                        identity_sha = _array_digest(
                            identity.detach().cpu().numpy().astype(np.float32, copy=False))
                        side_sha = _array_digest(side)
                        moved = True
                        route_reason = "always_commit"
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
            "evidence_row_sha256": row_digest,
            "d2": None if outcome is None else decision_d2(outcome),
        })
        expected_carrier_sha = final_carrier_sha
        bank_digest_chain.append(bank.digest)
    _require(bool(predictions), "a session surface must contain at least one window")
    prediction = np.ascontiguousarray(np.concatenate(predictions, axis=0), dtype=np.float32)
    _require(prediction.shape == targets.shape, "per-trial decode/target shape drift")
    return {
        "cell": "O2m", "budget": int(budget),
        "prediction": prediction, "targets": targets,
        "receipts": receipts, "committed_rows": len(bank),
        "identity_sha256": identity_sha,
        "side_sha256": side_sha,
        "initial_carrier_sha256": _array_digest(raw_support),
        "final_carrier_sha256": _array_digest(active_raw),
        "bank_digest_chain": bank_digest_chain,
        "measurement_reasons": measurement_reasons,
        "direction_rows_count": len(bank),
        "movements": movements, "d2_values": d2_values,
        "projected_count": int(projected_count),
        "committed_label_counts": [
            sum(1 for row in bank.rows
                if int(row.direction_indices[group]) >= 0)
            for group in range(cdm_core.GROUP_COUNT)
        ],
        "structural_noop": bool(float(alpha_M) == 0.0),
    }


def decision_d2(outcome: trust_region.TrustRegionOutcome) -> float:
    return float(outcome.d2_unprojected)


def _causality_chain(receipts: Sequence[Mapping[str, Any]]) -> bool:
    previous_ca: Optional[str] = None
    for receipt in receipts:
        if previous_ca is not None and receipt["cb"] != previous_ca:
            return False
        previous_ca = receipt["ca"]
    return True


# ---------------------------------------------------------------------------
# Census and verdict (pre-registered; unit-tested on synthetic input).
# ---------------------------------------------------------------------------


def census_from_receipts(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The per-session oracle evidence-stream census of one rollout."""
    eligible = sum(1 for item in receipts if item["evidence_eligible"])
    with_windows = sum(1 for item in receipts
                       if item["evidence_eligible"] and int(item["causal_windows"]) > 0)
    accepted = sum(1 for item in receipts if str(item["reason"]).startswith("always_commit"))
    reasons: dict[str, int] = {}
    for item in receipts:
        reason = str(item["reason"])
        if reason.startswith("measurement_rejected:"):
            key = reason.split("measurement_rejected:", 1)[1]
            reasons[key] = reasons.get(key, 0) + 1
    return {
        "evidence_eligible_trials": int(eligible),
        "with_complete_window_trials": int(with_windows),
        "accepted_true_directions": int(accepted),
        "rejection_reasons": dict(sorted(reasons.items())),
        "oracle_acceptance_rate": (float(accepted) / float(with_windows)
                                   if with_windows else None),
    }


def pool_census(per_session: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    keys = ("evidence_eligible_trials", "with_complete_window_trials",
            "accepted_true_directions")
    pooled = {key: int(sum(int(row[key]) for row in per_session.values()))
              for key in keys}
    reasons: dict[str, int] = {}
    for row in per_session.values():
        for key, value in row["rejection_reasons"].items():
            reasons[key] = reasons.get(key, 0) + int(value)
    pooled["rejection_reasons"] = dict(sorted(reasons.items()))
    pooled["oracle_acceptance_rate"] = (
        float(pooled["accepted_true_directions"]) / float(pooled["with_complete_window_trials"])
        if pooled["with_complete_window_trials"] else None
    )
    return pooled


def decoded_census_from_sealed(sealed_rows: Mapping[str, Mapping[str, Any]],
                               surface: str, budget: int) -> dict[str, Any]:
    """The sealed decoded-direction census, READ from the governing receipt."""
    per_session: dict[str, Mapping[str, Any]] = {}
    for (row_surface, row_budget, session), row in sealed_rows.items():
        if row_surface != surface or int(row_budget) != int(budget):
            continue
        receipts = row["receipts"]
        eligible = sum(1 for item in receipts if item["evidence_eligible"])
        with_windows = sum(1 for item in receipts
                           if item["evidence_eligible"] and int(item["causal_windows"]) > 0)
        per_session[session] = {
            "evidence_eligible_trials": int(eligible),
            "with_complete_window_trials": int(with_windows),
            "committed_rows": int(row["committed_rows"]),
            "measurement_reasons": dict(row["measurement_reasons"]),
        }
    pooled_eligible = sum(r["evidence_eligible_trials"] for r in per_session.values())
    pooled_windows = sum(r["with_complete_window_trials"] for r in per_session.values())
    pooled_committed = sum(r["committed_rows"] for r in per_session.values())
    reasons: dict[str, int] = {}
    for r in per_session.values():
        for key, value in r["measurement_reasons"].items():
            reasons[key] = reasons.get(key, 0) + int(value)
    low_displacement = int(reasons.get("measurement_rejected:low_displacement", 0))
    return {
        "source": "the sealed cdm_p1_m2_local_v1 replay.json F01m rows (read, never recomputed)",
        "per_session": dict(sorted(per_session.items())),
        "pooled": {
            "evidence_eligible_trials": pooled_eligible,
            "with_complete_window_trials": pooled_windows,
            "committed_rows": pooled_committed,
            "measurement_reasons": dict(sorted(reasons.items())),
            "decoded_acceptance_rate": (float(pooled_committed) / float(pooled_windows)
                                        if pooled_windows else None),
            "low_displacement_share_of_measurement_attempts": (
                float(low_displacement) / float(pooled_windows) if pooled_windows else None),
        },
    }


def evaluate_verdict(*, external_deltas: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """The pre-registered oracle-headroom verdict.

    ``external_deltas`` maps budget -> the paired delta payload of the PRIMARY
    O2m row (alpha_M = 0.5) versus O0m on the held-out surface.  GO requires
    some budget with ``equal_session_mean_delta >= +0.01`` AND at least
    ``4/6`` strictly positive held-out sessions at that budget.
    """
    _require(set(map(int, external_deltas)) == set(plan.BUDGETS),
             "the verdict needs both held-out budgets")
    epsilon = float(plan.VERDICT_LAW["boundary_epsilon"])
    rows: dict[str, Any] = {}
    go_budgets: list[int] = []
    for budget in sorted(int(item) for item in external_deltas):
        delta = external_deltas[budget]
        mean_delta = float(delta["equal_session_mean_delta"])
        positive = int(delta["positive_sessions"])
        total = int(delta["session_count"])
        within_band = abs(mean_delta - 0.01) <= epsilon
        meets_delta = bool(mean_delta >= 0.01 or (within_band and mean_delta >= 0.01 - epsilon))
        meets_breadth = bool(positive >= int(plan.HELD_OUT_BREADTH_MIN))
        go = bool(meets_delta and meets_breadth)
        if go:
            go_budgets.append(budget)
        rows[f"m{budget}"] = {
            "equal_session_mean_delta": mean_delta,
            "delta_threshold": 0.01,
            "meets_delta": meets_delta,
            "within_epsilon_band_of_threshold": bool(within_band),
            "positive_sessions": positive,
            "session_count": total,
            "breadth_threshold": int(plan.HELD_OUT_BREADTH_MIN),
            "meets_breadth": meets_breadth,
            "go": go,
        }
    verdict = plan.VERDICT_GO if go_budgets else plan.VERDICT_NULL
    return {
        "law": dict(plan.VERDICT_LAW),
        "rows": rows,
        "go_budgets": [f"m{item}" for item in go_budgets],
        "verdict": verdict,
        "verdict_string_pre_registered": True,
    }


def classify_wall(*, pooled_external_accepted: Mapping[int, int]) -> dict[str, Any]:
    """The pre-registered NULL-branch wall classifier."""
    per_budget = {int(budget): int(value) for budget, value in pooled_external_accepted.items()}
    _require(set(per_budget) == set(plan.BUDGETS), "the wall classifier needs both budgets")
    scarcity = all(value < 24 for value in per_budget.values())
    return {
        "law": dict(plan.WALL_CLASSIFIER),
        "pooled_external_accepted_true_directions": {
            f"m{budget}": value for budget, value in sorted(per_budget.items())
        },
        "scarcity_threshold_per_budget": 24,
        "wall": ("commit_point_scarcity_wall" if scarcity
                 else "no_information_in_true_directions"),
    }


def _leakage_labels(cell: str) -> dict[str, bool]:
    if cell.startswith("O2m"):
        return dict(plan.LEAKAGE_LAW["o2m_rows"])
    return dict(plan.LEAKAGE_LAW["o0m_rows"])


# ---------------------------------------------------------------------------
# The execute stage.
# ---------------------------------------------------------------------------


def execute(repo_root: Path, *, batch_size: int = plan.BATCH_SIZE) -> dict[str, Any]:
    _validate_environment()
    repo_root = Path(repo_root).absolute()
    root = plan.result_root(repo_root)
    _require(not (root / "replay.json").exists(), "the replay receipt already exists")
    _require(not (root / "terminal.json").exists(), "the terminal receipt already exists")
    _require((root / "attempt.json").exists(), "reserve the attempt receipt first")
    attempt_digest = _verify_sidecar(root / "attempt.json")
    attempt = json.loads((root / "attempt.json").read_text(encoding="utf-8"))
    for relative, digest in attempt["predecessor_sha256s"].items():
        _require(_sha256_file(repo_root / relative) == digest,
                 f"an immutable predecessor drifted: {relative}")
    for relative, digest in attempt["owned_sha256s"].items():
        _require(_sha256_file(repo_root / relative) == digest,
                 f"an owned module drifted: {relative}")

    sealed = _load_sealed_governing(repo_root)
    sealed_f00m = _sealed_matrix_rows(sealed, "F00m")
    sealed_f01m = _sealed_matrix_rows(sealed, "F01m")

    _bind_namespaces(repo_root)
    from src.m2_same_query_comparator_v1.core import (
        array_sha256 as sealed_digest,
    )
    from src.m2_same_query_comparator_v1.core import variance_weighted_r2

    import torch

    _require(not torch.cuda.is_available(),
             "CUDA must not be visible to this CPU-only run")
    torch.set_num_threads(int(plan.Torch_THREADS))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    started = time.monotonic()

    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data

    model, data_module, _task_config, metadata = load_frozen_model_and_data()
    _require(metadata["checkpoint_sha256"] == governing_replay.plan.T4_CHECKPOINT_SHA256,
             "T4 checkpoint drift")
    _require(metadata["teacher_checkpoint_sha256"] == governing_replay.plan.SPINT_CHECKPOINT_SHA256,
             "SPINT teacher checkpoint drift")
    _require(metadata["normalization_sha256"] == governing_replay.plan.NORMALIZATION_SHA256,
             "T4 normalization drift")
    student = model.student.to(device).eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    decoder = governing_replay.M2Decoder(torch, student, device, batch_size=batch_size)

    surfaces = {
        "within_post30": data_module.train_dataset,
        "external_official_query": data_module.val_heldout_dataset,
    }
    expected_counts = {
        "within_post30": plan.EXPECTED_WITHIN_SESSIONS,
        "external_official_query": plan.EXPECTED_EXTERNAL_SESSIONS,
    }
    sealed_rows = _load_sealed_same_query_rows(repo_root)

    materials: dict[str, dict[str, governing_replay.SessionMaterial]] = {}
    carriers: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    anchors: dict[str, dict[str, dict[int, M2SupportAnchor]]] = {}
    binding_evidence: dict[str, Any] = {}
    for surface, dataset in surfaces.items():
        _require(dataset is not None, f"{surface}: dataset missing")
        sessions = sorted(dataset.calib_trialized_neural_features)
        _require(len(sessions) == expected_counts[surface], f"{surface}: session count drift")
        materials[surface] = {}
        carriers[surface] = {}
        anchors[surface] = {}
        for session in sessions:
            materials[surface][session] = governing_replay.build_session_material(
                dataset=dataset, session=session, surface=surface,
            )
            carriers[surface][session] = {}
            anchors[surface][session] = {}
            for budget in plan.ALL_BUDGETS:
                carrier = governing_replay.sealed_carrier(dataset, session, budget)
                groups = build_groups(carrier["raw_t4"],
                                      materials[surface][session].channel_ids)
                anchor = M2SupportAnchor.from_labeled_support(
                    groups=groups,
                    support_trial_rates=carrier["rates"],
                    support_angles_rad=carrier["theta"].tolist(),
                    support_t4=carrier["raw_t4"],
                )
                _require(anchor.support_coefficient_parity["rebuilt_rows_bitwise_equal"],
                         f"anchor zero-evidence fallback drifted ({surface}/{session} m{budget})")
                sealed_carrier_row = sealed_rows[(surface, session, budget)]
                _require(carrier["selected"].tolist()
                         == sealed_carrier_row["side_evidence"]["selected_indices"],
                         f"selected support drift {surface}/{session} m{budget}")
                _require(sealed_digest(carrier["raw_t4"])
                         == sealed_carrier_row["side_evidence"]["raw_t4_sha256"],
                         f"raw carrier digest drift {surface}/{session} m{budget}")
                _require(sealed_digest(carrier["side"])
                         == sealed_carrier_row["side_evidence"]["normalized_t4_sha256"],
                         f"normalized carrier digest drift {surface}/{session} m{budget}")
                _require(sealed_digest(carrier["activity"])
                         == sealed_carrier_row["activity_sha256"],
                         f"activity digest drift {surface}/{session} m{budget}")
                empty_parity = stage_o_block_refit.empty_bank_refit_parity(
                    StageOAnchorAdapter(anchor), rho_M=float(plan.RHO_M),
                )
                _require(max(empty_parity["max_abs_coefficient_difference_by_group"])
                         <= 1.0e-6,
                         f"Stage-O empty-bank parity drifted ({surface}/{session} m{budget})")
                carriers[surface][session][budget] = carrier
                anchors[surface][session][budget] = anchor
                binding_evidence[f"{surface}|{session}|m{budget}"] = {
                    "selected": carrier["selected"].tolist(),
                    "raw_t4_sha256": carrier["raw_t4_sha256"],
                    "side_sha256": carrier["side_sha256"],
                    "activity_sha256": carrier["activity_sha256"],
                    "anchor_payload_digest": anchor.digest,
                    "anchor_parity": anchor.support_coefficient_parity,
                    "empty_bank_parity_max_abs": max(
                        empty_parity["max_abs_coefficient_difference_by_group"]),
                }
        print(f"[{time.monotonic() - started:8.1f}s] bound {surface}: "
              f"{len(materials[surface])} sessions", flush=True)

    # -- O0m: the sealed F00m law verbatim -----------------------------------
    o0m_rows: dict[str, dict[int, dict[str, dict[str, Any]]]] = {
        surface: {budget: {} for budget in plan.ALL_BUDGETS} for surface in plan.SURFACES
    }
    for surface in plan.SURFACES:
        for budget in plan.ALL_BUDGETS:
            for session in sorted(materials[surface]):
                rollout = governing_replay.rollout_session_f(
                    decoder, materials[surface][session], budget=int(budget),
                    spec=governing_replay.CellSpec("F00m", "frozen_t4_per_trial", False),
                    anchor=anchors[surface][session][budget],
                    carrier=carriers[surface][session][budget],
                    hp=None, config=cdm_core.CDMDConfig(support_budget_m=int(budget)),
                )
                r2 = float(variance_weighted_r2(rollout["targets"], rollout["prediction"]))
                o0m_rows[surface][budget][session] = {
                    "surface": surface, "session": session, "cell": "O0m",
                    "budget": int(budget),
                    "window_count": int(rollout["prediction"].shape[0]),
                    "query_starts_sha256": sealed_digest(materials[surface][session].starts),
                    "target_sha256": sealed_digest(rollout["targets"]),
                    "prediction_sha256": sealed_digest(rollout["prediction"]),
                    "initial_carrier_sha256": rollout["initial_carrier_sha256"],
                    "final_carrier_sha256": rollout["final_carrier_sha256"],
                    "r2": r2,
                    "zero_carrier_movement_every_trial": True,
                    "causality_chain_verified": bool(
                        _causality_chain(rollout["receipts"])),
                    **_leakage_labels("O0m"),
                }
                _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                         "the CPU hard timeout fired during the O0m family")
            decoder.clear_cache()
        print(f"[{time.monotonic() - started:8.1f}s] O0m {surface} done "
              f"(budgets {list(plan.ALL_BUDGETS)})", flush=True)

    # -- the O0m anchors against the sealed F00m rows -------------------------
    o0m_anchor_checks: dict[str, Any] = {}
    for surface in plan.SURFACES:
        for budget in plan.ALL_BUDGETS:
            for session in sorted(materials[surface]):
                mine = o0m_rows[surface][budget][session]
                sealed_row = sealed_f00m[(surface, str(budget), session)]
                checks = {
                    "window_count_equal": mine["window_count"] == int(sealed_row["window_count"]),
                    "query_starts_sha256_equal": (
                        mine["query_starts_sha256"] == sealed_row["query_starts_sha256"]),
                    "target_sha256_equal": (
                        mine["target_sha256"] == sealed_row["target_sha256"]),
                    "abs_delta_r2_vs_sealed_f00m": abs(mine["r2"] - float(sealed_row["r2"])),
                    "tolerance": float(plan.ANCHOR_TOLERANCE_R2),
                }
                checks["pass"] = bool(
                    checks["window_count_equal"]
                    and checks["query_starts_sha256_equal"]
                    and checks["target_sha256_equal"]
                    and checks["abs_delta_r2_vs_sealed_f00m"] <= float(plan.ANCHOR_TOLERANCE_R2)
                )
                o0m_anchor_checks[f"{surface}|{session}|m{budget}"] = checks
                _require(checks["pass"],
                         f"the O0m anchor failed at {surface}/{session}/m{budget}")
    print(f"[{time.monotonic() - started:8.1f}s] O0m anchors all pass "
          f"(max |delta| = {max(item['abs_delta_r2_vs_sealed_f00m'] for item in o0m_anchor_checks.values()):.3e})",
          flush=True)

    # -- the c_M calibration pass (within-7 ONLY; doubles as the no-op anchor) -
    calibration_rows: dict[int, dict[str, dict[str, Any]]] = {
        budget: {} for budget in plan.BUDGETS
    }
    c_M_by_budget: dict[int, float] = {}
    for budget in plan.BUDGETS:
        pooled_d2: list[float] = []
        for session in sorted(materials[plan.SELECTION_SURFACE]):
            rollout = rollout_session_oracle(
                decoder, materials[plan.SELECTION_SURFACE][session], budget=int(budget),
                anchor=anchors[plan.SELECTION_SURFACE][session][budget],
                carrier=carriers[plan.SELECTION_SURFACE][session][budget],
                alpha_M=0.0, c_M=None,
                config=cdm_core.CDMDConfig(support_budget_m=int(budget)),
            )
            r2 = float(variance_weighted_r2(rollout["targets"], rollout["prediction"]))
            zero_movement = all(item["cb"] == item["ca"] for item in rollout["receipts"])
            o0_digest = o0m_rows[plan.SELECTION_SURFACE][budget][session]["prediction_sha256"]
            calibration_rows[budget][session] = {
                "r2": r2,
                "prediction_sha256": sealed_digest(rollout["prediction"]),
                "o0m_prediction_sha256": o0_digest,
                "prediction_bitwise_equal_o0m": bool(
                    sealed_digest(rollout["prediction"]) == o0_digest),
                "committed_rows": int(rollout["committed_rows"]),
                "zero_movement_every_trial": bool(zero_movement),
                "causality_chain_verified": bool(_causality_chain(rollout["receipts"])),
                "census": census_from_receipts(rollout["receipts"]),
                "d2_values": [float(item) for item in rollout["d2_values"]],
            }
            _require(calibration_rows[budget][session]["prediction_bitwise_equal_o0m"],
                     f"the alpha=0 calibration no-op law failed ({plan.SELECTION_SURFACE}/{session}/m{budget})")
            _require(zero_movement, "the alpha=0 calibration moved the carrier")
            pooled_d2.extend(float(item) for item in rollout["d2_values"])
            _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                     "the CPU hard timeout fired during calibration")
            decoder.clear_cache()
        _require(pooled_d2, f"m{budget}: the calibration collected no commits")
        c_M = trust_region.calibrate_c_M(pooled_d2)
        c_M_by_budget[int(budget)] = float(c_M)
        print(f"[{time.monotonic() - started:8.1f}s] c_M m{budget} calibrated = {c_M:.6f} "
              f"({len(pooled_d2)} pooled within commits)", flush=True)

    # -- O2m: the oracle anchored block refit --------------------------------
    o2m_rows: dict[str, dict[int, dict[float, dict[str, dict[str, Any]]]]] = {
        surface: {budget: {} for budget in plan.BUDGETS} for surface in plan.SURFACES
    }
    o2m_r2: dict[str, dict[int, dict[float, dict[str, float]]]] = {
        surface: {budget: {} for budget in plan.BUDGETS} for surface in plan.SURFACES
    }
    for surface in plan.SURFACES:
        for budget in plan.BUDGETS:
            for alpha in plan.ALPHA_M_ALL:
                for session in sorted(materials[surface]):
                    rollout = rollout_session_oracle(
                        decoder, materials[surface][session], budget=int(budget),
                        anchor=anchors[surface][session][budget],
                        carrier=carriers[surface][session][budget],
                        alpha_M=float(alpha), c_M=float(c_M_by_budget[int(budget)]),
                        config=cdm_core.CDMDConfig(support_budget_m=int(budget)),
                    )
                    r2 = float(variance_weighted_r2(rollout["targets"], rollout["prediction"]))
                    receipts = rollout["receipts"]
                    o2m_rows[surface][budget].setdefault(float(alpha), {})[session] = {
                        "surface": surface, "session": session, "cell": "O2m",
                        "budget": int(budget), "alpha_M": float(alpha),
                        "hyperparameters": {
                            "rho_M": float(plan.RHO_M),
                            "alpha_M": float(alpha),
                            "c_M": float(c_M_by_budget[int(budget)]),
                            "block_size_completed_trials": int(
                                plan.BLOCK_SIZE_COMPLETED_TRIALS),
                            "evidence_weight_w": float(plan.EVIDENCE_WEIGHT_W),
                            "commit_law": "always_commit_trust_region_projection_only",
                            "governs": bool(float(alpha) == float(plan.ALPHA_M_PRIMARY)),
                        },
                        "window_count": int(rollout["prediction"].shape[0]),
                        "query_starts_sha256": sealed_digest(materials[surface][session].starts),
                        "target_sha256": sealed_digest(rollout["targets"]),
                        "prediction_sha256": sealed_digest(rollout["prediction"]),
                        "initial_carrier_sha256": rollout["initial_carrier_sha256"],
                        "final_carrier_sha256": rollout["final_carrier_sha256"],
                        "committed_rows": int(rollout["committed_rows"]),
                        "committed_label_counts": list(rollout["committed_label_counts"]),
                        "projected_count": int(rollout["projected_count"]),
                        "measurement_reasons": dict(rollout["measurement_reasons"]),
                        "movements_count": len(rollout["movements"]),
                        "max_movement": (float(np.max(rollout["movements"]))
                                         if rollout["movements"] else 0.0),
                        "median_movement": (float(np.median(rollout["movements"]))
                                            if rollout["movements"] else 0.0),
                        "d2_values": [float(item) for item in rollout["d2_values"]],
                        "bank_digest_chain_length": len(rollout["bank_digest_chain"]),
                        "bank_final_sha256": rollout["bank_digest_chain"][-1],
                        "census": census_from_receipts(receipts),
                        "r2": r2,
                        "causality_chain_verified": bool(_causality_chain(receipts)),
                        "receipts": receipts,
                        **_leakage_labels("O2m"),
                    }
                    o2m_r2[surface][budget].setdefault(float(alpha), {})[session] = r2
                    _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                             "the CPU hard timeout fired during the O2m family")
                decoder.clear_cache()
            print(f"[{time.monotonic() - started:8.1f}s] O2m {surface} m{budget} done "
                  f"(alphas {list(plan.ALPHA_M_ALL)})", flush=True)

    # -- the M30 no-op row ----------------------------------------------------
    m30_rows: dict[str, dict[str, dict[str, Any]]] = {
        surface: {} for surface in plan.SURFACES
    }
    for surface in plan.SURFACES:
        for session in sorted(materials[surface]):
            rollout = rollout_session_oracle(
                decoder, materials[surface][session], budget=int(plan.M30),
                anchor=anchors[surface][session][int(plan.M30)],
                carrier=carriers[surface][session][int(plan.M30)],
                alpha_M=float(plan.ALPHA_M_M30_NOOP), c_M=None,
                config=cdm_core.CDMDConfig(support_budget_m=int(plan.M30)),
            )
            r2 = float(variance_weighted_r2(rollout["targets"], rollout["prediction"]))
            zero_movement = all(item["cb"] == item["ca"] for item in rollout["receipts"])
            o0_digest = o0m_rows[surface][plan.M30][session]["prediction_sha256"]
            prediction_digest = sealed_digest(rollout["prediction"])
            _require(prediction_digest == o0_digest,
                     f"the M30 no-op law failed ({surface}/{session})")
            _require(zero_movement, f"the M30 no-op row moved the carrier ({surface}/{session})")
            m30_rows[surface][session] = {
                "surface": surface, "session": session, "cell": "O2m_m30_noop",
                "budget": int(plan.M30), "alpha_M": float(plan.ALPHA_M_M30_NOOP),
                "r2": r2,
                "prediction_sha256": prediction_digest,
                "o0m_prediction_sha256": o0_digest,
                "prediction_bitwise_equal_o0m": True,
                "committed_rows": int(rollout["committed_rows"]),
                "zero_movement_every_trial": True,
                "causality_chain_verified": bool(_causality_chain(rollout["receipts"])),
                "census": census_from_receipts(rollout["receipts"]),
                **_leakage_labels("O2m_m30_noop"),
            }
            _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                     "the CPU hard timeout fired during the M30 no-op row")
        decoder.clear_cache()
        print(f"[{time.monotonic() - started:8.1f}s] O2m m30 no-op {surface} done", flush=True)

    # -- summaries, deltas, census, verdict ------------------------------------
    o0m_summaries = {
        surface: {
            budget: {
                "equal_session_mean": governing_gates.equal_session_mean(
                    {name: row["r2"] for name, row in o0m_rows[surface][budget].items()}),
                "per_session_r2": {
                    name: float(row["r2"])
                    for name, row in sorted(o0m_rows[surface][budget].items())},
            }
            for budget in plan.ALL_BUDGETS
        }
        for surface in plan.SURFACES
    }
    o2m_summaries = {
        surface: {
            budget: {
                float(alpha): {
                    "equal_session_mean": governing_gates.equal_session_mean(
                        o2m_r2[surface][budget][float(alpha)]),
                    "per_session_r2": dict(sorted(o2m_r2[surface][budget][float(alpha)].items())),
                }
                for alpha in plan.ALPHA_M_ALL
            }
            for budget in plan.BUDGETS
        }
        for surface in plan.SURFACES
    }
    deltas = {
        surface: {
            budget: {
                float(alpha): governing_gates.paired_delta(
                    o2m_r2[surface][budget][float(alpha)],
                    {name: row["r2"] for name, row in o0m_rows[surface][budget].items()})
                for alpha in plan.ALPHA_M_ALL
            }
            for budget in plan.BUDGETS
        }
        for surface in plan.SURFACES
    }
    primary = float(plan.ALPHA_M_PRIMARY)
    external_deltas = {
        budget: deltas[plan.HELD_OUT_SURFACE][budget][primary]
        for budget in plan.BUDGETS
    }
    verdict = evaluate_verdict(external_deltas=external_deltas)

    oracle_census = {
        surface: {
            budget: {
                "primary_alpha": primary,
                "per_session": {
                    session: o2m_rows[surface][budget][primary][session]["census"]
                    for session in sorted(o2m_rows[surface][budget][primary])
                },
                "pooled": pool_census({
                    session: o2m_rows[surface][budget][primary][session]["census"]
                    for session in sorted(o2m_rows[surface][budget][primary])
                }),
            }
            for budget in plan.BUDGETS
        }
        for surface in plan.SURFACES
    }
    decoded_census = {
        surface: {
            budget: decoded_census_from_sealed(sealed_f01m, surface, int(budget))
            for budget in plan.BUDGETS
        }
        for surface in plan.SURFACES
    }
    wall = classify_wall(pooled_external_accepted={
        budget: oracle_census[plan.HELD_OUT_SURFACE][budget]["pooled"]["accepted_true_directions"]
        for budget in plan.BUDGETS
    })
    if verdict["verdict"] != plan.VERDICT_NULL:
        wall = {
            "law": dict(plan.WALL_CLASSIFIER),
            "not_classified_because": (
                "the verdict is not the NULL branch; the census rows are still reported"
            ),
            "pooled_external_accepted_true_directions": wall[
                "pooled_external_accepted_true_directions"],
        }

    alpha_sensitivity = {
        surface: {
            budget: {
                float(alpha): {
                    "equal_session_mean": o2m_summaries[surface][budget][float(alpha)]["equal_session_mean"],
                    "equal_session_mean_delta_vs_o0m":
                        deltas[surface][budget][float(alpha)]["equal_session_mean_delta"],
                    "positive_sessions":
                        deltas[surface][budget][float(alpha)]["positive_sessions"],
                }
                for alpha in plan.ALPHA_M_ALL
            }
            for budget in plan.BUDGETS
        }
        for surface in plan.SURFACES
    }

    anchors_payload = {
        "o0m_vs_sealed_f00m_rows": o0m_anchor_checks,
        "o0m_all_pass": all(item["pass"] for item in o0m_anchor_checks.values()),
        "o2m_alpha0_noop": {
            f"m{budget}|{session}": {
                "prediction_bitwise_equal_o0m": row["prediction_bitwise_equal_o0m"],
                "zero_movement_every_trial": row["zero_movement_every_trial"],
                "causality_chain_verified": row["causality_chain_verified"],
            }
            for budget in plan.BUDGETS
            for session, row in calibration_rows[budget].items()
        },
        "o2m_alpha0_all_pass": all(
            row["prediction_bitwise_equal_o0m"] and row["zero_movement_every_trial"]
            for budget in plan.BUDGETS for row in calibration_rows[budget].values()
        ),
        "m30_noop_all_pass": all(
            row["prediction_bitwise_equal_o0m"] and row["zero_movement_every_trial"]
            for surface in plan.SURFACES for row in m30_rows[surface].values()
        ),
        "empty_bank_parity_all_within_quantization_floor": all(
            item["empty_bank_parity_max_abs"] <= 1.0e-6
            for item in binding_evidence.values()
        ),
        "tolerance_law": {
            "per_session_abs_delta_r2_max_allowed": float(plan.ANCHOR_TOLERANCE_R2),
            "reason": (
                "CPU float-reduction order cannot bitwise-match GPU receipts; "
                "window counts, query starts and targets remain bit-exact and "
                "the alpha=0 no-op laws are exact by construction"
            ),
        },
    }

    replay_payload = {
        "schema": f"{plan.SCHEMA}_replay_v1",
        "stage": "stage_o_oracle_headroom_inference_only_cpu",
        "status": "REPLAY_COMPLETE",
        "attempt_sha256": attempt_digest,
        "authority": plan.AUTHORITY,
        "foundation": {
            "governing_root": plan.GOVERNING_ROOT_RELATIVE,
            "governing_attempt_sha256": plan.GOVERNING_ATTEMPT_SHA256,
            "governing_replay_sha256": plan.GOVERNING_REPLAY_SHA256,
            "governing_terminal_sha256": plan.GOVERNING_TERMINAL_SHA256,
            "audit_relative": plan.AUDIT_RELATIVE,
            "audit_sha256": plan.AUDIT_SHA256,
            "audit_verdict": governing_replay.plan.AUDIT_VERDICT,
            "machinery": {
                "anchor": "src.cdm_p1_m2_local_v1.anchor.M2SupportAnchor (verbatim, via the read-only StageOAnchorAdapter)",
                "block_refit": "src.support_anchored_t4_stage_o_v1.block_refit (verbatim)",
                "trust_region": "src.support_anchored_t4_stage_o_v1.trust_region.active_carrier (verbatim)",
                "direction_law": "src.causal_dual_memory_cell_d_v1.core.pseudo_direction_from_velocity (verbatim)",
                "f00m_runtime": "src.cdm_p1_m2_local_v1.replay.rollout_session_f (verbatim)",
            },
        },
        "environment": {
            **plan.ENVIRONMENT_LAW,
            "batch_size": int(batch_size),
            "seeds": [42],
            "cpu_count_threads": int(plan.Torch_THREADS),
        },
        "cells": dict(plan.CELLS),
        "true_direction_law": dict(plan.TRUE_DIRECTION_LAW),
        "o2m_commit_law": dict(plan.O2M_COMMIT_LAW),
        "c_m_calibration": {
            "law": dict(plan.C_M_CALIBRATION),
            "c_M_by_budget": {f"m{budget}": float(value)
                              for budget, value in sorted(c_M_by_budget.items())},
            "calibration_rows": {
                f"m{budget}": rows for budget, rows in calibration_rows.items()
            },
        },
        "carrier_support_binding": binding_evidence,
        "o0m_rows": o0m_rows,
        "o2m_rows": o2m_rows,
        "m30_noop_rows": m30_rows,
        "summaries": {"O0m": o0m_summaries, "O2m": o2m_summaries},
        "deltas_vs_o0m": deltas,
        "alpha_sensitivity": alpha_sensitivity,
        "census": {
            "law": dict(plan.CENSUS_LAW),
            "oracle": oracle_census,
            "decoded_sealed": decoded_census,
        },
        "anchors": anchors_payload,
        "leakage_law": dict(plan.LEAKAGE_LAW),
        "verdict": verdict,
        "wall": wall,
        "deviations": list(plan.DEVIATIONS),
        "target_optimizer_backward_update": 0,
        "target_backward_calls": 0,
        "target_parameter_update_calls": 0,
        "decoder_training": False,
        "model_or_checkpoint_updated": False,
        "wall_seconds": float(time.monotonic() - started),
        "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "started_utc": attempt.get("started_utc"),
        "finished_utc": _utc_now(),
    }
    _require(replay_payload["anchors"]["o0m_all_pass"],
             "the O0m reproduction anchor failed at least one row")
    _require(replay_payload["anchors"]["o2m_alpha0_all_pass"],
             "the alpha=0 no-op law failed at least one row")
    _require(replay_payload["anchors"]["m30_noop_all_pass"],
             "the M30 no-op law failed at least one row")
    _require(replay_payload["anchors"]["empty_bank_parity_all_within_quantization_floor"],
             "the Stage-O empty-bank parity failed")
    replay_digest = _publish(root / "replay.json", replay_payload)
    terminal_payload = {
        "schema": f"{plan.SCHEMA}_terminal_v1",
        "status": "TERMINAL",
        "scope": "m2_stage_o_oracle_carrier_headroom_inference_only_cpu",
        "replay_sha256": replay_digest,
        "attempt_sha256": attempt_digest,
        "authority": plan.AUTHORITY,
        "leakage_statement": plan.LEAKAGE_LAW["receipt_statement"],
        "can_this_run_select_or_promote": False,
        "checkpoint_selection_eligible": False,
        "deployment_eligible": False,
        "post_hoc_diagnostic": True,
        "foundation": replay_payload["foundation"],
        "true_direction_law": dict(plan.TRUE_DIRECTION_LAW),
        "c_M_by_budget": replay_payload["c_m_calibration"]["c_M_by_budget"],
        "equal_session_means": replay_payload["summaries"],
        "primary_deltas_external": {
            f"m{budget}": external_deltas[budget] for budget in plan.BUDGETS
        },
        "alpha_sensitivity": alpha_sensitivity,
        "m30_noop": {
            surface: {
                "equal_session_mean": governing_gates.equal_session_mean(
                    {name: row["r2"] for name, row in m30_rows[surface].items()}),
                "all_rows_bitwise_equal_o0m": all(
                    row["prediction_bitwise_equal_o0m"]
                    for row in m30_rows[surface].values()),
                "committed_rows_total": int(sum(
                    row["committed_rows"] for row in m30_rows[surface].values())),
            }
            for surface in plan.SURFACES
        },
        "census": replay_payload["census"],
        "anchors": {
            "o0m_all_pass": anchors_payload["o0m_all_pass"],
            "o2m_alpha0_all_pass": anchors_payload["o2m_alpha0_all_pass"],
            "m30_noop_all_pass": anchors_payload["m30_noop_all_pass"],
            "empty_bank_parity_all_within_quantization_floor": anchors_payload[
                "empty_bank_parity_all_within_quantization_floor"],
            "carrier_support_binding_all_pass": True,
        },
        "verdict": verdict,
        "wall": wall,
        "deviations": list(plan.DEVIATIONS),
        "environment": replay_payload["environment"],
        "wall_seconds": replay_payload["wall_seconds"],
        "finished_utc": replay_payload["finished_utc"],
    }
    _publish(root / "terminal.json", terminal_payload)
    return {
        "status": terminal_payload["status"],
        "replay_sha256": replay_digest,
        "verdict": verdict["verdict"],
        "go_budgets": verdict["go_budgets"],
        "wall": wall.get("wall"),
        "primary_deltas_external": {
            f"m{budget}": external_deltas[budget]["equal_session_mean_delta"]
            for budget in plan.BUDGETS
        },
        "c_M_by_budget": {f"m{budget}": float(value)
                          for budget, value in sorted(c_M_by_budget.items())},
        "anchors_all_pass": (
            anchors_payload["o0m_all_pass"]
            and anchors_payload["o2m_alpha0_all_pass"]
            and anchors_payload["m30_noop_all_pass"]
            and anchors_payload["empty_bank_parity_all_within_quantization_floor"]
        ),
        "wall_seconds": replay_payload["wall_seconds"],
    }
