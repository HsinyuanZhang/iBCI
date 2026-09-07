"""P2' oracle-policy decomposition matrix over the frozen V8 CDM evaluator.

Every matrix row shares, by construction of this module:

* the exact query windows, trial chronology, parser, normalizer and sealed
  Cell-D SWA checkpoint of the accepted V8 evaluator (inherited unchanged);
* the M-trial deployment recipe initial carrier (D-opt@M4 / chronological@M10
  plus the frozen fixed ridge 0.1) built by the inherited ``_initial_memory``;
* the activity-only CAUSAL activity trajectory -- the B3S FIFO advances by the
  frozen independent-activity law on every completed trial regardless of the
  carrier action, which is what makes the accept/reject counterfactual clean;
* the CDM carrier update machinery itself: ``observe_completed_trial``,
  ``propose_pseudo_trial`` (the frozen B8 design/departure checks) and
  ``commit_independent`` are the imported state machine, never reimplemented;
* the one pre-registered output filter (causal EMA alpha=0.25, trial-local,
  trial-boundary reset) for every row except the raw anchor A0.

Rows differ ONLY in (i) the pseudo-direction construction fed to ``observe``
and (ii) the carrier-action rule applied to the resulting pending proposal.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from src.causal_dual_memory_cell_d_activity_only_quick_v1 import physical as aq1physical
from src.causal_dual_memory_cell_d_score_v1 import physical as v1physical
from src.causal_dual_memory_cell_d_score_v1 import plan as v1plan
from src.causal_dual_memory_cell_d_score_v1 import score as v1score
from src.causal_dual_memory_cell_d_score_v5 import physical as v5physical
from src.causal_dual_memory_cell_d_v1 import core
from src.precision_aware_causal_dual_memory_cell_d_score_v1 import score as predecessor_score
from src.precision_aware_causal_dual_memory_cell_d_v2 import transition as precision_transition

from . import filters, plan, policy


class P2PrimePhysicalError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P2PrimePhysicalError(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")).encode("utf-8") + b"\n"


def _publish(path: Path, payload: Mapping[str, object]) -> str:
    body = _json_bytes(payload)
    digest = hashlib.sha256(body).hexdigest()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        os.write(descriptor, body)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    sidecar = path.with_name(path.name + ".sha256")
    descriptor = os.open(sidecar, flags, 0o444)
    try:
        os.write(descriptor, f"{digest}  {path.name}\n".encode("ascii"))
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_bytes())


# ---------------------------------------------------------------------------
# Rollout specification.
# ---------------------------------------------------------------------------


ROW_KINDS = ("activity", "policy", "oracle", "substudy", "m30_oracle")


class RolloutSpec:
    """One matrix row's rollout law (data-only; the loop below interprets)."""

    def __init__(
        self,
        *,
        row_id: str,
        kind: str,
        construction: str = "raw",
        precision_rule: bool = False,
        horizon_H: int = 5,
        constructions: Sequence[str] = (),
    ) -> None:
        _require(kind in ROW_KINDS, f"unknown P2' rollout kind: {kind}")
        _require(not precision_rule or kind == "policy", "the precision rule is a policy-row gate")
        self.row_id = row_id
        self.kind = kind
        self.construction = construction
        self.precision_rule = precision_rule
        self.horizon_H = int(horizon_H)
        self.constructions = tuple(constructions)
        if kind == "substudy":
            _require(len(self.constructions) >= 1, "substudy needs at least one construction")


# ---------------------------------------------------------------------------
# The runtime.
# ---------------------------------------------------------------------------


class P2PrimeOracleMatrixRuntime(aq1physical.ActivityOnlyRuntime):
    """V8 evaluator plus the P2' rollout kinds; inference only, never trained."""

    def __init__(self, *, root: Path, selected_device_profile: Mapping[str, object]) -> None:
        super().__init__(root=Path(root), selected_device_profile=selected_device_profile)
        self._memory_mode = "cdm"
        self._horizon_double_pass_proofs = 0
        self._horizon_single_pass_forwards = 0
        self._padded_true_rows = 0

    # -- memory construction: the row decides which imported memory runs ----

    def _make_dual_memory(self, *, core: Any, activity: Any, carrier: Any) -> Any:
        # Instance-method override of the inherited static factory hook: the
        # row context set by the rollout selects the imported memory class.
        if self._memory_mode == "activity_only":
            memory = aq1physical.ActivityOnlyMemory(activity=activity, carrier=carrier)
            _require(type(memory) is aq1physical.ActivityOnlyMemory, "P2' activity memory drift")
            return memory
        memory = core.IndependentActivityCausalDualMemory(activity=activity, carrier=carrier)
        _require(
            type(memory) is core.IndependentActivityCausalDualMemory,
            "P2' carrier rows require the exact independent activity memory",
        )
        return memory

    def _commit_dispatch(self, *, memory: Any, pending: Any, trial_id: str) -> v1physical.CompletedTrialTransition:
        if type(memory) is aq1physical.ActivityOnlyMemory:
            return aq1physical.ActivityOnlyRuntime._commit_completed_transition(
                self, memory=memory, pending=pending, trial_id=trial_id,
            )
        return v5physical.V5ReviewedCDMScoreRuntime._commit_completed_transition(
            self, memory=memory, pending=pending, trial_id=trial_id,
        )

    def _group_predictions_dispatch(
        self, *, trial: v1physical.EvaluationTrial, memory: Any,
    ) -> tuple[tuple[Any, ...], list[Mapping[str, object]]]:
        """Route to the frozen four-group pseudo forward for carrier rows."""
        if type(memory) is aq1physical.ActivityOnlyMemory:
            return aq1physical.ActivityOnlyRuntime._group_predictions(trial=trial, memory=memory)
        return v1physical.ReviewedCDMScoreRuntime._group_predictions(self, trial=trial, memory=memory)

    # -- light horizon forward ---------------------------------------------

    def _forward_horizon_once(
        self, *, neural_windows: Any, activity_stack: Any, normalized_t4: Any,
        attest_repeat: bool = False,
    ) -> np.ndarray:
        """One eval/no-grad pass; bitwise-deterministic twin of ``_forward_full``.

        Both horizon branches and the reused committed-branch step-1
        prediction run through this single path with identical batching.  A
        first-call double-pass attestation, per-trial digest parity asserts,
        periodic bitwise recompute checks and the A0/C0/P receipt anchors pin
        cross-call determinism.
        """
        state = self._require_state()
        torch, np, pop, arm = (
            state.modules["torch"], state.modules["np"], state.modules["pop_robust"], state.modules["arm_common"],
        )
        neural_cpu = np.ascontiguousarray(np.asarray(neural_windows), dtype=np.float32)
        stack_cpu = np.ascontiguousarray(np.asarray(activity_stack), dtype=np.float32)
        side_cpu = np.ascontiguousarray(np.asarray(normalized_t4), dtype=np.float32)
        if (
            neural_cpu.ndim != 3 or neural_cpu.shape[1] != v1plan.WINDOW_BINS
            or stack_cpu.ndim != 3 or not 1 <= stack_cpu.shape[0] <= 30 or stack_cpu.shape[1] != 100
            or side_cpu.shape != (neural_cpu.shape[2], 4)
            or stack_cpu.shape[2] != neural_cpu.shape[2]
        ):
            raise P2PrimePhysicalError("horizon forward input shape drift")
        neural = torch.as_tensor(neural_cpu, dtype=torch.float32, device=state.device)
        calibration = torch.as_tensor(stack_cpu, dtype=torch.float32, device=state.device)
        calibration = calibration.unsqueeze(0).expand(neural.shape[0], -1, -1, -1)
        side = torch.as_tensor(side_cpu, dtype=torch.float32, device=state.device)
        side = side.unsqueeze(0).expand(neural.shape[0], -1, -1)
        before = arm.state_sha256(state.model)
        chunks: list[Any] = []
        with pop.dynamic_dropout_recorder() as recorder:
            with torch.no_grad():
                for start in range(0, int(neural.shape[0]), v1plan.EVAL_BATCH_SIZE):
                    stop = min(start + v1plan.EVAL_BATCH_SIZE, int(neural.shape[0]))
                    first, _ = state.model(
                        neural[start:stop], calib_trials=calibration[start:stop], side_features=side[start:stop],
                    )
                    if attest_repeat:
                        second, _ = state.model(
                            neural[start:stop], calib_trials=calibration[start:stop], side_features=side[start:stop],
                        )
                        if not torch.equal(first, second):
                            raise P2PrimePhysicalError("horizon forward repeat determinism drift")
                    chunks.append(first.detach())
        output = torch.cat(chunks, dim=0)
        if (
            not bool(torch.isfinite(output).all().item())
            or recorder.get("uniform_calls") != 0
            or recorder.get("dropout_calls") != []
        ):
            raise P2PrimePhysicalError("horizon forward finite/dropout purity drift")
        after = arm.state_sha256(state.model)
        if before != after:
            raise P2PrimePhysicalError("sealed Cell-D model state mutated during horizon forward")
        self._horizon_single_pass_forwards += 1
        if attest_repeat:
            self._horizon_double_pass_proofs += 1
        state.forward_chunks += len(chunks)
        return output[:, v1plan.GOVERNING_BIN, :].detach().cpu().numpy()

    # -- session scaffolding --------------------------------------------------

    def _session_target_views(
        self, session: v1physical.PreparedEvaluationSession, budget: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray], float]:
        """Per-trial governing targets + validity masks + the session SST."""
        np_module = self._require_state().modules["np"]
        per_trial_targets: list[np.ndarray] = []
        per_trial_valid: list[np.ndarray] = []
        for trial_id in session.query_trial_ids[budget]:
            trial = session.trials_by_id[trial_id]
            target = np_module.ascontiguousarray(
                np_module.asarray(session.behavior[trial.endpoint_bins]), dtype=np.float32,
            )
            valid = np_module.ascontiguousarray(np_module.all(target != -1.0, axis=1), dtype=np.uint8)
            per_trial_targets.append(target)
            per_trial_valid.append(valid)
        full_target, full_mask, valid_count = self._governing_target_mask_authority(
            np_module, behavior=session.behavior, query_trial_ids=session.query_trial_ids[budget],
            trials_by_id=session.trials_by_id,
        )
        joined_target = np_module.concatenate(per_trial_targets, axis=0)
        joined_mask = np_module.concatenate(per_trial_valid, axis=0)
        expected = session.input_record_payload
        key = str(budget)

        def _array_sha(value: Any) -> str:
            array = np_module.ascontiguousarray(np_module.asarray(value))
            return hashlib.sha256(
                v1plan.canonical_json_bytes({
                    "dtype": str(array.dtype),
                    "shape": [int(item) for item in array.shape],
                    "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                })
            ).hexdigest()

        if (
            int(joined_mask.sum()) != valid_count
            or int(joined_mask.sum()) != int(expected["valid_last_bin_count_by_budget"][key])
            or _array_sha(joined_target) != expected["target_last_bin_sha256_by_budget"][key]
            or not np_module.array_equal(joined_mask, np_module.asarray(full_mask))
        ):
            raise P2PrimePhysicalError("P2' per-trial target view differs from the immutable input authority")
        sst = policy.session_sst(
            [t[v.astype(bool)] for t, v in zip(per_trial_targets, per_trial_valid)]
        )
        _require(math.isfinite(sst) and sst > 0.0, "session SST degenerate")
        return per_trial_targets, per_trial_valid, sst

    def _true_velocity_rows(
        self, session: v1physical.PreparedEvaluationSession, trial: v1physical.EvaluationTrial,
    ) -> np.ndarray:
        rows = np.asarray(session.behavior[trial.endpoint_bins], dtype=np.float64)
        restored, padded = filters.true_physical_velocity_one_trial(
            rows,
            behavior_mean=v1plan.SEALED_BEHAVIOR_MEAN,
            behavior_std=v1plan.SEALED_BEHAVIOR_STD,
        )
        if int(np.sum(padded)) > 0:
            self._padded_true_rows += int(np.sum(padded))
        return restored

    def _materialize_subset(
        self, *, authority: v1score.FixedEvaluationAuthority, per_surface: int,
    ) -> None:
        """Smoke-only subset materialization (disclosed, never governing)."""
        state = self._require_state()
        _require(not state.sessions, "smoke subset materialization may run exactly once")
        _require(per_surface >= 1, "smoke subset needs at least one session per surface")
        for group in (authority.within, authority.external):
            for asset in group[:per_surface]:
                prepared = self._parse_session(asset=asset)
                state.sessions[(asset.surface, asset.session)] = prepared

    # -- metrics ---------------------------------------------------------------

    def _house_raw_r2(
        self,
        per_trial_raw: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
    ) -> float:
        """The historical float32 house R^2 on the RAW governing predictions."""
        state = self._require_state()
        torch = state.modules["torch"]
        prediction = torch.from_numpy(np.ascontiguousarray(np.concatenate(
            [np.asarray(item, dtype=np.float32)[m.astype(bool)] for item, m in zip(per_trial_raw, masks)],
            axis=0,
        )))
        target = torch.from_numpy(np.ascontiguousarray(np.concatenate(
            [np.asarray(item, dtype=np.float32)[m.astype(bool)] for item, m in zip(targets, masks)],
            axis=0,
        )))
        return float(state.modules["matched_metric"].session_r2(prediction, target))

    def _matrix_r2(
        self,
        per_trial_prediction: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
    ) -> tuple[float, np.ndarray]:
        """The matrix R^2: one float64 path shared by every row."""
        state = self._require_state()
        torch = state.modules["torch"]
        joined = np.ascontiguousarray(np.concatenate(
            [np.asarray(item, dtype=np.float64)[m.astype(bool)] for item, m in zip(per_trial_prediction, masks)],
            axis=0,
        ))
        target = np.ascontiguousarray(np.concatenate(
            [np.asarray(item, dtype=np.float64)[m.astype(bool)] for item, m in zip(targets, masks)],
            axis=0,
        ))
        r2 = float(state.modules["matched_metric"].session_r2(
            torch.from_numpy(joined), torch.from_numpy(target),
        ))
        return r2, joined

    @staticmethod
    def _raw_prediction_digest(
        per_trial_raw: Sequence[np.ndarray], masks: Sequence[np.ndarray],
    ) -> str:
        joined = np.ascontiguousarray(np.concatenate(
            [np.asarray(item, dtype=np.float32)[m.astype(bool)] for item, m in zip(per_trial_raw, masks)],
            axis=0,
        ))
        return hashlib.sha256(joined.tobytes()).hexdigest()

    # -- the horizon counterfactual -------------------------------------------

    def _horizon_counterfactual(
        self,
        *,
        query_trials: Sequence[v1physical.EvaluationTrial],
        decision_index: int,
        accept_state: core.DualMemoryState,
        reject_state: core.DualMemoryState,
        parity: Mapping[str, object],
        targets: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        sst: float,
        horizon_H: int,
        keep_steps: Sequence[int] = (1,),
    ) -> dict[str, object]:
        """Both branches over trials j+1..j+H from the SAME parent state.

        The activity memory advances through the completed trials by the
        frozen independent-activity law identically on both branches; the
        carrier is frozen per branch after the action at trial j.  No further
        carrier action happens inside a horizon.  ``decision_index`` is the
        index of trial j, and the loop starts at ``decision_index + 1``, so
        trial j can never appear in its own utility window.
        """
        _require(parity["activity_stack_bitwise_equal"] is True, "horizon branches differ in activity")
        activity = accept_state.activity
        carrier_accept = accept_state.carrier
        carrier_reject = reject_state.carrier
        accept_losses: list[float] = []
        reject_losses: list[float] = []
        accept_predictions: dict[int, np.ndarray] = {}
        reject_predictions: dict[int, np.ndarray] = {}
        accept_raw_step1: Optional[np.ndarray] = None
        reject_raw_step1: Optional[np.ndarray] = None
        step_activity_digests: list[str] = []
        horizon = 0
        for step in range(1, horizon_H + 1):
            index = decision_index + step
            if index >= len(query_trials):
                break
            trial = query_trials[index]
            stack = activity.stack()
            step_activity_digests.append(core.array_digest(stack))
            raw_accept = self._forward_horizon_once(
                neural_windows=trial.neural_windows, activity_stack=stack,
                normalized_t4=self._normalized_side(carrier_accept.active_t4).detach().cpu().numpy(),
                attest_repeat=(self._horizon_single_pass_forwards == 0),
            )
            raw_reject = self._forward_horizon_once(
                neural_windows=trial.neural_windows, activity_stack=stack,
                normalized_t4=self._normalized_side(carrier_reject.active_t4).detach().cpu().numpy(),
            )
            selection = masks[index].astype(bool)
            target_rows = np.asarray(targets[index], dtype=np.float64)[selection]
            for branch_raw, losses, store in (
                (raw_accept, accept_losses, accept_predictions),
                (raw_reject, reject_losses, reject_predictions),
            ):
                filtered = filters.apply_output_filter_one_trial(branch_raw)
                losses.append(policy.trial_nsse(filtered[selection], target_rows, sst=sst))
                if step in keep_steps:
                    store[step] = filtered
            if step == 1:
                accept_raw_step1 = raw_accept
                reject_raw_step1 = raw_reject
            horizon = step
            activity = activity.after_completed_trial(trial.b3s_activity)
        if horizon == 0:
            # A decision at the final trial has no future horizon: the action
            # cannot affect any scored trial, so no utility is defined.
            return {
                "utility": None,
                "accept_losses": [],
                "reject_losses": [],
                "accept_predictions": {},
                "reject_predictions": {},
                "accept_raw_step1": None,
                "reject_raw_step1": None,
                "step_activity_digests": [],
                "carrier_accept_sha256": carrier_accept.digest,
                "carrier_reject_sha256": carrier_reject.digest,
            }
        utility = policy.horizon_utility(reject_losses=reject_losses, accept_losses=accept_losses)
        return {
            "utility": utility,
            "accept_losses": accept_losses,
            "reject_losses": reject_losses,
            "accept_predictions": accept_predictions,
            "reject_predictions": reject_predictions,
            "accept_raw_step1": accept_raw_step1,
            "reject_raw_step1": reject_raw_step1,
            "step_activity_digests": step_activity_digests,
            "carrier_accept_sha256": carrier_accept.digest,
            "carrier_reject_sha256": carrier_reject.digest,
        }

    # -- row session rollouts ---------------------------------------------------

    def _bump_trial_counters(self) -> None:
        state = self._require_state()
        state.completed_trials += 1
        if state.measurement_started is None:
            state.measurement_started = time.monotonic()

    def _governing_forward(
        self, *, trial: v1physical.EvaluationTrial, inputs: Any,
    ) -> np.ndarray:
        side = self._normalized_side(inputs.active_t4).detach().cpu().numpy()
        full, _evidence = self._forward_full(
            neural_windows=trial.neural_windows, activity_stack=inputs.activity_trials,
            normalized_t4=side,
        )
        return full[:, v1plan.GOVERNING_BIN, :].detach().cpu().numpy()

    def _compact_transition(self, item: v1physical.CompletedTrialTransition) -> dict[str, object]:
        return {
            "trial_id": item.trial_id,
            "activity_transition_committed": item.activity_transition_committed,
            "activity_fifo_changed": item.activity_fifo_changed,
            "carrier_transition_committed": item.carrier_transition_committed,
            "carrier_rejection_reason_or_null": item.carrier_rejection_reason,
            "carrier_before_sha256": item.carrier_before_sha256,
            "carrier_after_sha256": item.carrier_after_sha256,
        }

    def _row_session_payload(
        self,
        *,
        row_id: str,
        session: v1physical.PreparedEvaluationSession,
        budget: int,
        per_trial_prediction: Sequence[np.ndarray],
        per_trial_raw: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        transitions: Sequence[v1physical.CompletedTrialTransition],
        initial: Mapping[str, object],
        extra: Optional[Mapping[str, object]] = None,
    ) -> dict[str, object]:
        matrix_r2, joined = self._matrix_r2(per_trial_prediction, targets, masks)
        payload: dict[str, object] = {
            "schema": "learned_gate_p2prime_row_session_v1",
            "row": row_id,
            "budget": int(budget),
            "surface": session.surface,
            "session": session.session,
            "n_windows": int(joined.shape[0]),
            "matrix_r2": matrix_r2,
            "house_raw_r2": self._house_raw_r2(per_trial_raw, targets, masks),
            "prediction_sha256_raw": self._raw_prediction_digest(per_trial_raw, masks),
            "filtered_prediction_sha256": hashlib.sha256(joined.tobytes()).hexdigest(),
            "output_filter": "raw" if row_id == "A0" else plan.OUTPUT_FILTER["family"] + f"_a{plan.OUTPUT_FILTER['alpha']}",
            "carrier_transitions_committed": sum(
                1 for item in transitions if item.carrier_transition_committed
            ),
            "activity_transitions_committed": sum(
                1 for item in transitions if item.activity_transition_committed
            ),
            "carrier_rejection_counts": _reason_counts(
                item.carrier_rejection_reason for item in transitions
            ),
            "initial_carrier_sha256": str(initial["initial_carrier_sha256"]),
            "initial_activity_sha256": str(initial["initial_activity_sha256"]),
            "final_carrier_sha256": transitions[-1].carrier_after_sha256 if transitions else None,
            "transitions": [self._compact_transition(item) for item in transitions],
        }
        if extra:
            payload.update(extra)
        return payload

    def _rollout_activity(
        self, *, session: v1physical.PreparedEvaluationSession, budget: int,
    ) -> dict[str, object]:
        """The A0/A1 shared rollout: the exact sealed activity-only law."""
        self._memory_mode = "activity_only"
        memory, initial = self._initial_memory(session=session, budget=budget)
        targets, masks, _sst = self._session_target_views(session, budget)
        query_ids = list(session.query_trial_ids[budget])
        per_trial_raw: list[np.ndarray] = []
        transitions: list[v1physical.CompletedTrialTransition] = []
        for trial_id in query_ids:
            trial = session.trials_by_id[trial_id]
            inputs = memory.read_prediction_inputs()
            per_trial_raw.append(self._governing_forward(trial=trial, inputs=inputs))
            group_predictions, _group_evidence = aq1physical.ActivityOnlyRuntime._group_predictions(
                trial=trial, memory=memory,
            )
            pending = memory.observe_completed_trial(
                b3s_trial_activity=trial.b3s_activity, carrier_trial_counts=trial.native_counts,
                complementary_predictions=group_predictions,
            )
            transition = self._commit_dispatch(memory=memory, pending=pending, trial_id=trial_id)
            transitions.append(transition)
            if budget == 30 and transition.activity_after_sha256 != transition.activity_before_sha256:
                raise P2PrimePhysicalError("M30 activity FIFO mutated despite literal capacity zero")
            self._bump_trial_counters()
        _require(
            all(item.carrier_transition_committed is False for item in transitions)
            and all(item.carrier_before_sha256 == item.carrier_after_sha256 for item in transitions)
            and all(item.activity_transition_committed for item in transitions)
            and all(item.carrier_rejection_reason == "insufficient_evidence" for item in transitions),
            "P2' activity rollout drifted from the sealed activity-only law",
        )
        per_trial_filtered = [
            filters.apply_output_filter_one_trial(item) for item in per_trial_raw
        ]
        state_digests = [
            {
                "activity_sha256": item.activity_after_sha256,
                "carrier_sha256": item.carrier_after_sha256,
            }
            for item in transitions
        ]
        rows = {
            "A0": self._row_session_payload(
                row_id="A0", session=session, budget=budget, per_trial_prediction=per_trial_raw,
                per_trial_raw=per_trial_raw, targets=targets, masks=masks, transitions=transitions,
                initial=initial,
            ),
            "A1": self._row_session_payload(
                row_id="A1", session=session, budget=budget, per_trial_prediction=per_trial_filtered,
                per_trial_raw=per_trial_raw, targets=targets, masks=masks, transitions=transitions,
                initial=initial,
            ),
        }
        return {
            "rows": rows,
            "per_trial_raw": per_trial_raw,
            "per_trial_filtered": per_trial_filtered,
            "transitions": transitions,
            "initial": initial,
            "initial_activity_sha256": transitions[0].activity_before_sha256,
            "initial_carrier_sha256": transitions[0].carrier_before_sha256,
            "post_commit_state_digests": state_digests,
        }

    def _bind_precision(
        self, *, session: v1physical.PreparedEvaluationSession, budget: int, memory: Any,
    ) -> dict[str, object]:
        np_module = self._require_state().modules["np"]
        support_ids = tuple(session.support_trial_ids[budget])
        rates = np_module.ascontiguousarray(
            np_module.stack([session.support_rates[item] for item in support_ids]), dtype=np_module.float64,
        )
        directions = np_module.ascontiguousarray(
            np_module.asarray([session.support_direction_indices[item] for item in support_ids], dtype=np_module.int64),
        )
        posterior = precision_transition.SupportConditionalPosterior.from_support_only_fixed_ridge(
            support_rates=rates,
            support_direction_indices=directions,
            valid_mask=memory.state.carrier.groups.valid_mask,
            groups_sha256=memory.state.carrier.groups.digest,
        )
        wrapper = precision_transition.PrecisionAwareIndependentActivityV2(memory=memory, posterior=posterior)
        return {"posterior": posterior, "wrapper": wrapper}

    def _rollout_policy(
        self, *, session: v1physical.PreparedEvaluationSession, budget: int, spec: RolloutSpec,
    ) -> dict[str, object]:
        """C0/C1/P: forward -> group pseudo -> construction -> commit law."""
        _require(budget in plan.DEPLOYMENT_BUDGETS, "policy rows run only on the deployment budgets")
        self._memory_mode = "cdm"
        memory, initial = self._initial_memory(session=session, budget=budget)
        targets, masks, _sst = self._session_target_views(session, budget)
        precision_binding = None
        precision_decisions: list[Optional[Mapping[str, object]]] = []
        if spec.precision_rule:
            precision_binding = self._bind_precision(session=session, budget=budget, memory=memory)
        query_ids = list(session.query_trial_ids[budget])
        per_trial_raw: list[np.ndarray] = []
        transitions: list[v1physical.CompletedTrialTransition] = []
        for trial_id in query_ids:
            trial = session.trials_by_id[trial_id]
            inputs = memory.read_prediction_inputs()
            per_trial_raw.append(self._governing_forward(trial=trial, inputs=inputs))
            group_predictions, _group_evidence = self._group_predictions_dispatch(trial=trial, memory=memory)
            views, _meta = policy.build_construction_predictions(
                group_predictions=group_predictions, construction=spec.construction,
                behavior_rows=session.behavior[trial.endpoint_bins]
                if spec.construction == "true" else None,
                behavior_mean=v1plan.SEALED_BEHAVIOR_MEAN, behavior_std=v1plan.SEALED_BEHAVIOR_STD,
            )
            pending = memory.observe_completed_trial(
                b3s_trial_activity=trial.b3s_activity, carrier_trial_counts=trial.native_counts,
                complementary_predictions=views,
            )
            if spec.precision_rule and pending.carrier_transition_accepted:
                decision = precision_transition.decide_against_frozen_support(
                    precision_binding["posterior"],
                    frozen_initial_active_t4=precision_binding["wrapper"]._frozen_initial_active_t4,
                    proposed_active_t4=pending.carrier_proposal.candidate.active_t4,
                )
                precision_decisions.append(dict(decision.payload()))
                if not decision.accepted:
                    pending = precision_binding["wrapper"]._precision_rejected_pending(pending)
            elif spec.precision_rule:
                precision_decisions.append(None)
            transition = self._commit_dispatch(memory=memory, pending=pending, trial_id=trial_id)
            transitions.append(transition)
            self._bump_trial_counters()
        per_trial_filtered = [filters.apply_output_filter_one_trial(item) for item in per_trial_raw]
        extra: dict[str, object] = {"construction": spec.construction}
        if spec.precision_rule:
            extra["precision_v2_rule"] = plan.ROW_SPECS["P"]["carrier_action"]
            extra["precision_v2_accepted_decisions"] = sum(
                1 for item in precision_decisions if item is not None and bool(item.get("accepted"))
            )
            extra["precision_v2_decision_count"] = sum(
                1 for item in precision_decisions if item is not None
            )
        row = self._row_session_payload(
            row_id=spec.row_id, session=session, budget=budget, per_trial_prediction=per_trial_filtered,
            per_trial_raw=per_trial_raw, targets=targets, masks=masks, transitions=transitions,
            initial=initial, extra=extra,
        )
        return {"rows": {spec.row_id: row}, "per_trial_raw": per_trial_raw, "initial": initial}

    def _rollout_oracle(
        self, *, session: v1physical.PreparedEvaluationSession, budget: int, spec: RolloutSpec,
    ) -> dict[str, object]:
        """O0/O1/O2 (coherent greedy oracle) and the M30 oracle-only no-op."""
        self._memory_mode = "cdm"
        memory, initial = self._initial_memory(session=session, budget=budget)
        targets, masks, sst = self._session_target_views(session, budget)
        query_ids = list(session.query_trial_ids[budget])
        query_trials = [session.trials_by_id[item] for item in query_ids]
        per_trial_raw: list[Optional[np.ndarray]] = [None] * len(query_ids)
        transitions: list[v1physical.CompletedTrialTransition] = []
        decision_records: list[dict[str, object]] = []
        initial_carrier_digest = memory.state.carrier.digest
        governing_cache: Optional[dict[str, object]] = None
        for index, trial in enumerate(query_trials):
            inputs = memory.read_prediction_inputs()
            if governing_cache is not None:
                _require(
                    governing_cache["activity_digest"] == memory.state.activity.digest
                    and governing_cache["carrier_digest"] == memory.state.carrier.digest,
                    "reused committed-branch step-1 parent digests drifted from the live memory",
                )
                per_trial_raw[index] = governing_cache["raw"]
                if index % 13 == 0:
                    recomputed = self._governing_forward(trial=trial, inputs=inputs)
                    _require(
                        np.array_equal(recomputed, per_trial_raw[index]),
                        "reused committed-branch step-1 prediction is not bitwise the governing forward",
                    )
            else:
                per_trial_raw[index] = self._governing_forward(trial=trial, inputs=inputs)
            group_predictions, _group_evidence = self._group_predictions_dispatch(trial=trial, memory=memory)
            views, _meta = policy.build_construction_predictions(
                group_predictions=group_predictions, construction=spec.construction,
                behavior_rows=session.behavior[trial.endpoint_bins]
                if spec.construction == "true" else None,
                behavior_mean=v1plan.SEALED_BEHAVIOR_MEAN, behavior_std=v1plan.SEALED_BEHAVIOR_STD,
            )
            pending = memory.observe_completed_trial(
                b3s_trial_activity=trial.b3s_activity, carrier_trial_counts=trial.native_counts,
                complementary_predictions=views,
            )
            record: dict[str, object] = {
                "trial_id": query_ids[index],
                "proposal_accepted_by_b8": bool(pending.carrier_transition_accepted),
                "action": "reject",
                "u_j": None,
                "horizon_trials": 0,
            }
            if pending.carrier_transition_accepted:
                accept_state, reject_state, parity = policy.branch_states_for_counterfactual(
                    memory, pending,
                    independent_activity=memory.state.activity.after_completed_trial(
                        trial.b3s_activity
                    ),
                )
                horizon = self._horizon_counterfactual(
                    query_trials=query_trials, decision_index=index, accept_state=accept_state,
                    reject_state=reject_state, parity=parity, targets=targets, masks=masks, sst=sst,
                    horizon_H=spec.horizon_H,
                )
                utility = horizon["utility"]
                record["parity"] = dict(parity)
                if utility is None:
                    record["u_j"] = None
                    record["horizon_trials"] = 0
                    record["action"] = "reject_no_future_horizon"
                    pending = policy.oracle_rejected_pending(memory, pending)
                    governing_cache = None
                    transition = self._commit_dispatch(
                        memory=memory, pending=pending, trial_id=query_ids[index],
                    )
                    _require(
                        transition.carrier_after_sha256 == memory.state.carrier.digest
                        and transition.activity_after_sha256 == pending.candidate_state.activity.digest,
                        "oracle no-horizon commit digest disagrees with the decided branch",
                    )
                    transitions.append(transition)
                    decision_records.append(record)
                    self._bump_trial_counters()
                    continue
                record["u_j"] = utility["u_j"]
                record["horizon_trials"] = utility["horizon_trials"]
                decide = policy.oracle_decision(utility)
                if spec.kind == "m30_oracle":
                    record["action"] = "accept_would_have" if decide else "reject"
                    record["m30_noop_enforced"] = True
                    pending = policy.oracle_rejected_pending(memory, pending)
                    governing_cache = None
                elif decide:
                    record["action"] = "accept"
                    governing_cache = {
                        "raw": horizon["accept_raw_step1"],
                        "activity_digest": accept_state.activity.digest,
                        "carrier_digest": accept_state.carrier.digest,
                        "step_activity_digest": horizon["step_activity_digests"][0],
                    }
                else:
                    record["action"] = "reject"
                    pending = policy.oracle_rejected_pending(memory, pending)
                    governing_cache = {
                        "raw": horizon["reject_raw_step1"],
                        "activity_digest": reject_state.activity.digest,
                        "carrier_digest": reject_state.carrier.digest,
                        "step_activity_digest": horizon["step_activity_digests"][0],
                    }
            else:
                record["b8_rejection_reason"] = (
                    pending.carrier_rejection_reason.value
                    if pending.carrier_rejection_reason is not None else None
                )
                governing_cache = None
            expected_carrier = memory.state.carrier.digest if not (
                pending.carrier_transition_accepted
            ) else pending.candidate_state.carrier.digest
            expected_activity = (
                pending.candidate_state.activity.digest
                if pending.candidate_state is not None else memory.state.activity.digest
            )
            transition = self._commit_dispatch(memory=memory, pending=pending, trial_id=query_ids[index])
            _require(
                transition.carrier_after_sha256 == expected_carrier
                and transition.activity_after_sha256 == expected_activity,
                "oracle commit carrier/activity digest disagrees with the decided branch",
            )
            transitions.append(transition)
            decision_records.append(record)
            self._bump_trial_counters()
        _require(all(item is not None for item in per_trial_raw), "oracle rollout left a governing forward unset")
        raw_list: list[np.ndarray] = [np.asarray(item) for item in per_trial_raw]
        per_trial_filtered = [filters.apply_output_filter_one_trial(item) for item in raw_list]
        extra: dict[str, object] = {
            "construction": spec.construction,
            "oracle_level": "COHERENT_GREEDY_ORACLE",
            "horizon_H": spec.horizon_H,
            "oracle_decisions": decision_records,
            "oracle_accept_count": sum(1 for item in decision_records if item["action"] == "accept"),
            "oracle_decision_count": sum(
                1 for item in decision_records if item["proposal_accepted_by_b8"]
            ),
            "mean_u_j_among_proposals": _mean_or_none([
                float(item["u_j"]) for item in decision_records if item["u_j"] is not None
            ]),
        }
        if spec.kind == "m30_oracle":
            final_digest = memory.state.carrier.digest
            extra["m30_carrier_bitwise_noop"] = final_digest == initial_carrier_digest
            extra["m30_carrier_constant_every_trial"] = all(
                item.carrier_before_sha256 == initial_carrier_digest
                and item.carrier_after_sha256 == initial_carrier_digest
                for item in transitions
            )
            extra["m30_would_accept_count"] = sum(
                1 for item in decision_records if item["action"] == "accept_would_have"
            )
            _require(
                extra["m30_carrier_bitwise_noop"] and extra["m30_carrier_constant_every_trial"],
                "M30 oracle diagnostic mutated the carrier",
            )
        row = self._row_session_payload(
            row_id=spec.row_id, session=session, budget=budget, per_trial_prediction=per_trial_filtered,
            per_trial_raw=raw_list, targets=targets, masks=masks, transitions=transitions,
            initial=initial, extra=extra,
        )
        return {"rows": {spec.row_id: row}, "initial": initial}

    def _rollout_substudy(
        self,
        *,
        session: v1physical.PreparedEvaluationSession,
        budget: int,
        constructions: Sequence[str],
        horizon_H: int,
        activity_rollout: Mapping[str, object],
    ) -> dict[str, object]:
        """The section 10.4 sub-study plus the NONCOHERENT ceiling evidence.

        The line never commits a carrier: it is the A1 trajectory with the CDM
        proposal stream computed per trial, which is exactly the parent
        trajectory the noncoherent one-step-switch ceiling rides.  The reject
        branch of every counterfactual is the A-line prediction itself.
        """
        self._memory_mode = "cdm"
        memory, initial = self._initial_memory(session=session, budget=budget)
        targets, masks, sst = self._session_target_views(session, budget)
        query_ids = list(session.query_trial_ids[budget])
        query_trials = [session.trials_by_id[item] for item in query_ids]
        a_raw = activity_rollout["per_trial_raw"]
        a_filtered = activity_rollout["per_trial_filtered"]
        a_nsse = [
            policy.trial_nsse(
                np.asarray(a_filtered[index], dtype=np.float64)[masks[index].astype(bool)],
                np.asarray(targets[index], dtype=np.float64)[masks[index].astype(bool)],
                sst=sst,
            )
            for index in range(len(query_ids))
        ]
        direction_rows: dict[str, list[dict[str, object]]] = {name: [] for name in constructions}
        utilities: dict[str, list[float]] = {name: [] for name in constructions}
        horizon_records: dict[str, list[dict[str, object]]] = {name: [] for name in constructions}
        for index, trial in enumerate(query_trials):
            inputs = memory.read_prediction_inputs()
            parent = (
                activity_rollout["post_commit_state_digests"][index - 1] if index >= 1
                else {
                    "activity_sha256": activity_rollout["initial_activity_sha256"],
                    "carrier_sha256": activity_rollout["initial_carrier_sha256"],
                }
            )
            _require(
                memory.state.activity.digest == parent["activity_sha256"]
                and memory.state.carrier.digest == parent["carrier_sha256"],
                "sub-study line drifted from the activity-only parent trajectory",
            )
            group_predictions, _group_evidence = self._group_predictions_dispatch(trial=trial, memory=memory)
            true_reference = policy.true_direction_payload(
                self._true_velocity_rows(session, trial), trial.velocity_validity,
                config=memory.state.carrier.config,
            )
            pendings: dict[str, Any] = {}
            for construction in constructions:
                views, _meta = policy.build_construction_predictions(
                    group_predictions=group_predictions, construction=construction,
                    behavior_rows=session.behavior[trial.endpoint_bins]
                    if construction == "true" else None,
                    behavior_mean=v1plan.SEALED_BEHAVIOR_MEAN, behavior_std=v1plan.SEALED_BEHAVIOR_STD,
                )
                pendings[construction] = memory.observe_completed_trial(
                    b3s_trial_activity=trial.b3s_activity, carrier_trial_counts=trial.native_counts,
                    complementary_predictions=views,
                )
                direction_rows[construction].append(
                    _direction_error_row(pendings[construction].pseudo_directions, true_reference)
                )
            for construction in constructions:
                pending = pendings[construction]
                record: dict[str, object] = {
                    "trial_id": query_ids[index], "proposal_accepted_by_b8": bool(pending.carrier_transition_accepted),
                }
                if pending.carrier_transition_accepted:
                    accept_state, reject_state, parity = policy.branch_states_for_counterfactual(
                    memory, pending,
                    independent_activity=memory.state.activity.after_completed_trial(
                        trial.b3s_activity
                    ),
                )
                    _require(
                        reject_state.carrier.digest == memory.state.carrier.digest,
                        "sub-study reject branch must keep the never-committing carrier",
                    )
                    accept_losses, reject_losses, accept_predictions = self._substudy_accept_horizon(
                        query_trials=query_trials, decision_index=index, accept_state=accept_state,
                        targets=targets, masks=masks, sst=sst, horizon_H=horizon_H, a_nsse=a_nsse,
                    )
                    record["parity"] = dict(parity)
                    if accept_losses:
                        utility = policy.horizon_utility(
                            reject_losses=reject_losses, accept_losses=accept_losses,
                        )
                        utilities[construction].append(float(utility["u_j"]))
                        record["u_j"] = utility["u_j"]
                        record["accept_losses"] = accept_losses
                        record["reject_losses"] = reject_losses
                        record["step1_accept_prediction"] = accept_predictions.get(1)
                        record["step_predictions"] = accept_predictions
                    else:
                        record["u_j"] = None
                horizon_records[construction].append(record)
            rejected = pendings[constructions[0]]
            if rejected.carrier_transition_accepted:
                rejected = policy.oracle_rejected_pending(memory, rejected)
            transition = self._commit_dispatch(
                memory=memory, pending=rejected, trial_id=query_ids[index],
            )
            _require(
                transition.carrier_before_sha256 == transition.carrier_after_sha256,
                "sub-study line committed a carrier transition",
            )
            self._bump_trial_counters()
        result: dict[str, object] = {
            "session": session.session,
            "budget": budget,
            "surface": session.surface,
            "direction_rows": {
                name: _aggregate_direction_errors(rows) for name, rows in direction_rows.items()
            },
            "utilities": {name: _summary_floats(values) for name, values in utilities.items()},
        }
        for construction in constructions:
            result[f"noncoherent_{construction}"] = self._noncoherent_session_rows(
                horizon_records=horizon_records[construction], a_filtered=a_filtered,
                a_nsse=a_nsse, targets=targets, masks=masks, horizon_H=horizon_H,
            )
        return result

    def _substudy_accept_horizon(
        self,
        *,
        query_trials: Sequence[v1physical.EvaluationTrial],
        decision_index: int,
        accept_state: core.DualMemoryState,
        targets: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        sst: float,
        horizon_H: int,
        a_nsse: Sequence[float],
    ) -> tuple[list[float], list[float], dict[int, np.ndarray]]:
        """Accept-branch horizon; the reject branch is the A-line losses."""
        activity = accept_state.activity
        carrier = accept_state.carrier
        accept_losses: list[float] = []
        reject_losses: list[float] = []
        predictions: dict[int, np.ndarray] = {}
        for step in range(1, horizon_H + 1):
            index = decision_index + step
            if index >= len(query_trials):
                break
            trial = query_trials[index]
            stack = activity.stack()
            raw = self._forward_horizon_once(
                neural_windows=trial.neural_windows, activity_stack=stack,
                normalized_t4=self._normalized_side(carrier.active_t4).detach().cpu().numpy(),
                attest_repeat=(self._horizon_single_pass_forwards == 0),
            )
            filtered = filters.apply_output_filter_one_trial(raw)
            selection = masks[index].astype(bool)
            accept_losses.append(policy.trial_nsse(
                filtered[selection], np.asarray(targets[index], dtype=np.float64)[selection], sst=sst,
            ))
            reject_losses.append(float(a_nsse[index]))
            predictions[step] = filtered
            activity = activity.after_completed_trial(trial.b3s_activity)
        return accept_losses, reject_losses, predictions

    def _noncoherent_session_rows(
        self,
        *,
        horizon_records: Sequence[Mapping[str, object]],
        a_filtered: Sequence[np.ndarray],
        a_nsse: Sequence[float],
        targets: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        horizon_H: int,
    ) -> dict[str, object]:
        """Assemble the noncoherent one-step-switch and loose ceilings."""
        n = len(a_filtered)
        one_step: list[np.ndarray] = []
        loose: list[np.ndarray] = []
        chosen_from_accept = 0
        for index in range(n):
            best_prediction = np.asarray(a_filtered[index], dtype=np.float64)
            best_loss = float(a_nsse[index])
            parent = horizon_records[index - 1] if index >= 1 else None
            if parent is not None and parent.get("step1_accept_prediction") is not None:
                accept_prediction = np.asarray(parent["step1_accept_prediction"], dtype=np.float64)
                accept_loss = float(parent["accept_losses"][0])  # type: ignore[index]
                if accept_loss < best_loss:
                    best_prediction = accept_prediction
                    best_loss = accept_loss
                    chosen_from_accept += 1
            one_step.append(best_prediction)
        for index in range(n):
            best_prediction = np.asarray(a_filtered[index], dtype=np.float64)
            best_loss = float(a_nsse[index])
            for offset in range(1, min(horizon_H, index) + 1):
                record = horizon_records[index - offset]
                steps = record.get("step_predictions")
                if not isinstance(steps, dict):
                    continue
                candidate = steps.get(offset)
                losses = record.get("accept_losses")
                if candidate is None or not isinstance(losses, list) or offset > len(losses):
                    continue
                if float(losses[offset - 1]) < best_loss:  # type: ignore[index]
                    best_prediction = np.asarray(candidate, dtype=np.float64)
                    best_loss = float(losses[offset - 1])  # type: ignore[index]
            loose.append(best_prediction)
        one_step_r2, _joined = self._matrix_r2(one_step, targets, masks)
        loose_r2, _joined = self._matrix_r2(loose, targets, masks)
        return {
            "NONCOHERENT_ONE_STEP_SWITCH_CEILING_r2": one_step_r2,
            "NONCOHERENT_LOOSE_MIN_OVER_PARENTS_r2": loose_r2,
            "one_step_trials_taken_from_accept_branch": chosen_from_accept,
            "status": "loose diagnostic ceiling, not an executable policy",
        }


def _reason_counts(reasons: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reason in reasons:
        if reason is None:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _summary_floats(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "positive": int((array > 0.0).sum()),
    }


def _direction_error_row(
    pseudo_directions: Sequence[core.PseudoDirection], true_reference: Mapping[str, object],
) -> dict[str, object]:
    true_accepted = bool(true_reference.get("accepted"))
    errors: list[float] = []
    mismatch = 0
    groups = 0
    for pseudo in pseudo_directions:
        if not pseudo.accepted or not true_accepted or pseudo.theta_raw_rad is None:
            continue
        groups += 1
        errors.append(core.circular_distance(float(pseudo.theta_raw_rad), float(true_reference["theta_raw_rad"])))
        if int(pseudo.theta_index) != int(true_reference["theta_index"]):
            mismatch += 1
    return {
        "true_accepted": true_accepted,
        "construction_accepted_groups": sum(1 for item in pseudo_directions if item.accepted),
        "groups_compared": groups,
        "mean_circular_error_rad": float(sum(errors) / len(errors)) if errors else None,
        "snapped_mismatch_groups": mismatch,
    }


def _aggregate_direction_errors(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    errors = [float(item["mean_circular_error_rad"]) for item in rows if item["mean_circular_error_rad"] is not None]
    mismatch = sum(int(item["snapped_mismatch_groups"]) for item in rows)
    compared = sum(int(item["groups_compared"]) for item in rows)
    return {
        "leakage_diagnostic_label": True,
        "trials": len(rows),
        "construction_accepted_trials": sum(
            1 for item in rows if int(item["construction_accepted_groups"]) > 0
        ),
        "true_accepted_trials": sum(1 for item in rows if bool(item["true_accepted"])),
        "groups_compared": compared,
        "mean_circular_error_rad": float(sum(errors) / len(errors)) if errors else None,
        "snapped_mismatch_rate": float(mismatch / compared) if compared else None,
    }


# ---------------------------------------------------------------------------
# Summaries, contrasts, anchors, winner selection.
# ---------------------------------------------------------------------------


def _slim_session(row: Mapping[str, object]) -> dict[str, object]:
    keep = (
        "row", "budget", "surface", "session", "n_windows", "matrix_r2", "house_raw_r2",
        "prediction_sha256_raw", "filtered_prediction_sha256", "output_filter",
        "carrier_transitions_committed", "activity_transitions_committed",
        "carrier_rejection_counts", "initial_carrier_sha256", "initial_activity_sha256",
        "final_carrier_sha256", "construction",
    )
    result = {key: row[key] for key in keep if key in row}
    if "oracle_decisions" in row:
        result["oracle_decisions"] = [
            {
                "trial_id": item["trial_id"],
                "proposal_accepted_by_b8": item["proposal_accepted_by_b8"],
                "action": item["action"],
                "u_j": item["u_j"],
                "horizon_trials": item["horizon_trials"],
            }
            for item in row["oracle_decisions"]
        ]
        for key in (
            "oracle_level", "horizon_H", "oracle_accept_count", "oracle_decision_count",
            "mean_u_j_among_proposals",
        ):
            if key in row:
                result[key] = row[key]
    if "m30_carrier_bitwise_noop" in row:
        for key in (
            "m30_carrier_bitwise_noop", "m30_carrier_constant_every_trial", "m30_would_accept_count",
        ):
            if key in row:
                result[key] = row[key]
    if "precision_v2_rule" in row:
        for key in ("precision_v2_accepted_decisions", "precision_v2_decision_count"):
            if key in row:
                result[key] = row[key]
    return result


def _mean(values: Sequence[float]) -> float:
    _require(bool(values), "mean of empty sequence")
    return float(sum(values) / len(values))


def _paired_contrast(
    row_sessions: Sequence[Mapping[str, object]], baseline_sessions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    from src.tfpd_lane.matched_scorer import paired_session_stats

    _require(
        tuple(item["session"] for item in row_sessions) == tuple(item["session"] for item in baseline_sessions),
        "paired contrast session roster drift",
    )
    deltas = [
        float(row["matrix_r2"]) - float(base["matrix_r2"])
        for row, base in zip(row_sessions, baseline_sessions, strict=True)
    ]
    return paired_session_stats(deltas)


def summarize_cells(cells: Mapping[str, Mapping[str, Mapping[str, Mapping[str, object]]]]) -> dict[str, object]:
    """cells[row][budget][surface] -> {"sessions": [...]} from one stage."""
    matrix: dict[str, object] = {}
    for row, by_budget in cells.items():
        for budget, by_surface in by_budget.items():
            for surface, payload in by_surface.items():
                sessions = payload["sessions"]
                matrix.setdefault(f"m{budget}", {}).setdefault(surface, {})[row] = {
                    "mean_r2": _mean([float(item["matrix_r2"]) for item in sessions]),
                    "mean_house_raw_r2": _mean([float(item["house_raw_r2"]) for item in sessions]),
                    "sessions": [_slim_session(item) for item in sessions],
                }
    for _budget, by_surface in matrix.items():
        for _surface, rows in by_surface.items():
            for row, baseline, _role in plan.CONTRASTS:
                if row in rows and baseline in rows:
                    rows[f"{row}_minus_{baseline}"] = _paired_contrast(
                        rows[row]["sessions"], rows[baseline]["sessions"],
                    )
    return matrix


def kill_matrix_view(matrix: Mapping[str, object]) -> dict[str, object]:
    """The row-mean view ``policy.kill_criterion_verdicts`` consumes."""
    view: dict[str, object] = {}
    for budget, by_surface in matrix.items():
        surface_map: dict[str, object] = {}
        for surface, rows in by_surface.items():
            surface_map[surface] = {
                row: {"mean_r2": payload["mean_r2"]}
                for row, payload in rows.items()
                if isinstance(payload, dict) and "mean_r2" in payload
            }
        view[budget] = surface_map
    return view


def _anchor_sessions(
    *, payload: Mapping[str, object], budget: int, surface: str, system: str,
    collection: str = "cells",
) -> dict[str, Mapping[str, object]]:
    summary = payload["budget_summaries"][str(budget)]
    if collection == "cells":
        matches = [
            cell for cell in summary["cells"]
            if cell["budget"] == budget and cell["surface"] == surface and cell["system"] == system
        ]
    else:
        matches = [
            cell for cell in summary[collection]
            if cell["budget"] == budget and cell["surface"] == surface
        ]
    _require(len(matches) == 1, "anchor cell topology drift")
    return {row["session"]: row for row in matches[0]["sessions"]}


def anchor_row_sessions(
    row_sessions: Sequence[Mapping[str, object]], anchor: Mapping[str, Mapping[str, object]], *, label: str,
) -> dict[str, object]:
    per_session = []
    all_exact = True
    for row in row_sessions:
        reference = anchor.get(str(row["session"]))
        if reference is None:
            raise P2PrimePhysicalError(f"anchor session absent for {label}: {row['session']}")
        reference_r2 = float(reference.get("governing_r2", reference.get("r2")))
        digest_match = str(row["prediction_sha256_raw"]) == str(reference["prediction_sha256"])
        r2_gap = abs(float(row["house_raw_r2"]) - reference_r2)
        exact = digest_match and r2_gap == 0.0
        all_exact = all_exact and exact
        per_session.append({
            "session": row["session"],
            "digest_match": digest_match,
            "house_raw_r2": float(row["house_raw_r2"]),
            "anchor_r2": reference_r2,
            "r2_gap": r2_gap,
            "exact_match": exact,
        })
    return {
        "label": label,
        "all_sessions_exact": bool(all_exact),
        "tolerance_pass": all(item["r2_gap"] <= 1.0e-9 for item in per_session),
        "sessions": per_session,
    }


def select_substudy_winner(substudy: Mapping[str, object]) -> dict[str, object]:
    """Apply the pre-registered sub-study winner rule (M4 external governs)."""
    from src.tfpd_lane.matched_scorer import paired_session_stats

    eligible = list(plan.SUB_STUDY_WINNER_RULE["eligible"])
    m4 = substudy_sessions(substudy, budget=4, surface="external")
    _require(bool(m4), "sub-study winner rule found no M4 external sessions")
    per_construction_mean: dict[str, float] = {}
    direction_error: dict[str, Optional[float]] = {}
    for construction in plan.PSEUDO_CONSTRUCTIONS:
        means = [
            float(session["utilities"][construction]["mean"])
            for session in m4 if session["utilities"].get(construction, {}).get("count", 0) > 0
        ]
        errors = [
            float(session["direction_rows"][construction]["mean_circular_error_rad"])
            for session in m4
            if session["direction_rows"].get(construction, {}).get("mean_circular_error_rad") is not None
        ]
        per_construction_mean[construction] = _mean(means) if means else float("nan")
        direction_error[construction] = _mean(errors) if errors else None
    left, right = eligible
    paired = [
        float(session["utilities"][left]["mean"]) - float(session["utilities"][right]["mean"])
        for session in m4
        if session["utilities"].get(left, {}).get("count", 0) > 0
        and session["utilities"].get(right, {}).get("count", 0) > 0
    ]
    stats = paired_session_stats(paired) if paired else {"bootstrap_95_interval": [None, None]}
    interval = stats["bootstrap_95_interval"]
    tie = bool(interval[0] is not None and interval[0] <= 0.0 <= interval[1])
    if tie:
        error_left = direction_error[left]
        error_right = direction_error[right]
        if (
            error_left is not None and error_right is not None
            and abs(error_left - error_right) > 1.0e-12
        ):
            winner = left if error_left < error_right else right
            rule = "tiebreak_1_direction_error"
        else:
            winner = "smoothed_causal" if "smoothed_causal" in eligible else eligible[0]
            rule = "tiebreak_2_prefer_causal_clock"
    else:
        winner = left if per_construction_mean[left] > per_construction_mean[right] else right
        rule = "primary_mean_u_j_m4_external"
    return {
        "winner": winner,
        "rule_applied": rule,
        "m4_external_mean_u_j": dict(per_construction_mean),
        "m4_external_mean_direction_error_rad": dict(direction_error),
        "paired_difference_bootstrap": stats,
        "eligible": eligible,
        "raw_mean_u_j_reference": per_construction_mean.get("raw"),
        "disclosure": plan.SUB_STUDY_WINNER_RULE["disclosure"],
    }


def substudy_sessions(
    substudy: Mapping[str, object], *, budget: int, surface: str,
) -> list[Mapping[str, object]]:
    return [
        item for item in substudy["sessions"]
        if int(item["budget"]) == budget and item["surface"] == surface
    ]


def compose_reading(stage_a_matrix: Mapping[str, object]) -> dict[str, object]:
    reading: dict[str, object] = {"pre_registered_question": plan.COMPOSE_READING["question"]}
    rows: dict[str, object] = {}
    for budget in plan.BUDGETS:
        key = f"m{budget}"
        surface_map = stage_a_matrix.get(key, {})
        for surface in plan.SURFACES:
            rows_map = surface_map.get(surface, {})
            if "A1_minus_A0" not in rows_map:
                continue
            static = plan.COMPOSE_READING["static_comparators_causal_ema_alpha_025_external"].get(str(budget))
            if surface == "external" and static is not None:
                delta = rows_map["A1_minus_A0"]
                comparable = abs(float(delta["mean"]) - static) <= 0.01
                rows[f"m{budget}:{surface}"] = {
                    "cdm_a1_minus_a0_mean": float(delta["mean"]),
                    "static_causal_ema_alpha025_mean": static,
                    "axes_compose": bool(float(delta["mean"]) >= static - 0.01),
                    "comparable_within_0p01": bool(comparable),
                }
            else:
                rows[f"m{budget}:{surface}"] = {
                    "cdm_a1_minus_a0_mean": float(rows_map["A1_minus_A0"]["mean"]),
                }
    reading["rows"] = rows
    reading["static_comparators_note"] = plan.COMPOSE_READING["static_comparators_note"]
    reading["reading_rule"] = plan.COMPOSE_READING["reading_rule"]
    return reading


# ---------------------------------------------------------------------------
# Stage driver.
# ---------------------------------------------------------------------------


def validate_environment(*, gpu_index: int) -> dict[str, str]:
    _require(gpu_index in (0, 1), "P2' GPU index must be 0 or 1")
    expected = {
        **plan.EXACT_DATA_ROOT_ENV,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(gpu_index),
    }
    for name, value in expected.items():
        _require(os.environ.get(name) == value, f"P2' environment drift: {name}")
    return expected


def _runtime_for(base: Path, *, gpu_index: int) -> tuple[P2PrimeOracleMatrixRuntime, Mapping[str, object], Any]:
    environment = validate_environment(gpu_index=gpu_index)
    profile = v1plan.COMPATIBLE_DEVICE_PROFILES[f"gpu{gpu_index}"]
    predecessor_score.validate_completed_v8_predecessor(base)
    identity = predecessor_score.build_reviewed_identity(base, selected_device_profile=profile)
    runtime = P2PrimeOracleMatrixRuntime(root=base, selected_device_profile=profile)
    return runtime, {"environment": environment, "identity_sha256": identity.sha256}, identity


def _attempt_digest(base: Path, output: Path) -> str:
    body = _read_json(output / "attempt.json")
    digest = hashlib.sha256((output / "attempt.json").read_bytes()).hexdigest()
    sidecar = (output / "attempt.json.sha256").read_text(encoding="ascii").strip()
    _require(sidecar == f"{digest}  attempt.json", "P2' attempt sidecar drift")
    plan.validate_pre_registration(body["pre_registration"])
    _require(
        body["cell"] == plan.CELL and body["status"] == "ATTEMPT_RESERVED",
        "P2' attempt cell/status drift",
    )
    _require(
        body["owned_sha256s"] == plan.owned_sha256s(base),
        "P2' owned module bytes drifted after the attempt was reserved",
    )
    return digest


def _ordered_session_keys(runtime: P2PrimeOracleMatrixRuntime, surface: str) -> tuple[tuple[str, str], ...]:
    state = runtime._require_state()
    keys = tuple(key for key, item in state.sessions.items() if key[0] == surface)
    _require(bool(keys), "P2' stage materialized no sessions for a surface")
    return keys


def _sealed_activity_anchor(sealed_activity: Mapping[str, object], *, budget: int, surface: str) -> dict[str, Mapping[str, object]]:
    matches = [
        cell["sessions"] for cell in sealed_activity["cells"]
        if cell["budget"] == budget and cell["surface"] == surface
    ]
    _require(len(matches) == 1, "sealed activity-only anchor cell topology drift")
    return {row["session"]: row for row in matches[0]}


def _stage_a(
    runtime: P2PrimeOracleMatrixRuntime,
    *,
    budgets: Sequence[int],
    identity_sha256: str,
) -> dict[str, object]:
    cells: dict[str, dict[int, dict[str, dict[str, object]]]] = {"A0": {}, "A1": {}}
    for budget in budgets:
        for surface in plan.SURFACES:
            rows = {"A0": [], "A1": []}
            for key in _ordered_session_keys(runtime, surface):
                session = runtime._require_state().sessions[key]
                rollout = runtime._rollout_activity(session=session, budget=budget)
                rows["A0"].append(rollout["rows"]["A0"])
                rows["A1"].append(rollout["rows"]["A1"])
            cells["A0"].setdefault(budget, {})[surface] = {"sessions": rows["A0"]}
            cells["A1"].setdefault(budget, {})[surface] = {"sessions": rows["A1"]}
    matrix = summarize_cells(cells)
    base = runtime.root
    anchors: dict[str, object] = {}
    sealed_activity = _read_json(base / plan.ACTIVITY_ONLY_V2_RESULT_RELATIVE / "result.json")
    for budget in (4, 10):
        if budget not in budgets:
            continue
        for surface in plan.SURFACES:
            anchors[f"A0_m{budget}_{surface}"] = anchor_row_sessions(
                matrix[f"m{budget}"][surface]["A0"]["sessions"],
                _sealed_activity_anchor(sealed_activity, budget=budget, surface=surface),
                label=f"A0 vs sealed activity-only m{budget} {surface}",
            )
    if 30 in budgets:
        v8_score = _read_json(base / plan.V8_SCORE_RELATIVE)
        for surface in plan.SURFACES:
            anchors[f"A0_m30_{surface}"] = anchor_row_sessions(
                matrix["m30"][surface]["A0"]["sessions"],
                _anchor_sessions(payload=v8_score, budget=30, surface=surface, system=v1plan.SYSTEM_SEALED),
                label=f"A0 vs sealed V8 static m30 {surface}",
            )
    return {
        "schema": "learned_gate_p2prime_stage_a_v1",
        "stage": "a_rows_activity_only",
        "identity_sha256": identity_sha256,
        "budgets": list(budgets),
        "matrix": matrix,
        "anchors": anchors,
        "compose_reading": compose_reading(matrix),
        "resources": dict(runtime._resources()),
        "horizon_forward_count": runtime._horizon_single_pass_forwards,
        "target_optimizer_backward_update": 0,
    }


def _stage_substudy(
    runtime: P2PrimeOracleMatrixRuntime,
    *,
    budgets: Sequence[int],
    horizon_H: int,
    identity_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    constructions = tuple(plan.PSEUDO_CONSTRUCTIONS)
    sessions: list[dict[str, object]] = []
    for budget in budgets:
        for surface in plan.SURFACES:
            for key in _ordered_session_keys(runtime, surface):
                session = runtime._require_state().sessions[key]
                activity_rollout = runtime._rollout_activity(session=session, budget=budget)
                record = runtime._rollout_substudy(
                    session=session, budget=budget, constructions=constructions,
                    horizon_H=horizon_H, activity_rollout=activity_rollout,
                )
                record["A0_matrix_r2"] = float(activity_rollout["rows"]["A0"]["matrix_r2"])
                record["A1_matrix_r2"] = float(activity_rollout["rows"]["A1"]["matrix_r2"])
                record["A1_prediction_sha256"] = activity_rollout["rows"]["A1"]["prediction_sha256_raw"]
                sessions.append(record)
    substudy = {
        "schema": "learned_gate_p2prime_stage_substudy_v1",
        "stage": "pseudo_direction_substudy",
        "identity_sha256": identity_sha256,
        "budgets": list(budgets),
        "horizon_H": horizon_H,
        "constructions": list(constructions),
        "sessions": sessions,
        "leakage_diagnostic_label": True,
    }
    winner = select_substudy_winner(substudy)
    substudy["winner_selection"] = winner
    return substudy, winner


def _stage_cop(
    runtime: P2PrimeOracleMatrixRuntime,
    *,
    budgets: Sequence[int],
    horizon_H: int,
    winner_construction: str,
    identity_sha256: str,
    sensitivity: bool,
    activity_matrix: Optional[Mapping[str, object]] = None,
) -> dict[str, object]:
    cells: dict[str, dict[int, dict[str, dict[str, object]]]] = {}
    if activity_matrix is not None:
        # The A0/A1 rows share the exact same session rosters; merging them
        # here is what lets the paired contrasts O1-A1 / P-A1 and the kill
        # criteria see one matrix.
        for budget in budgets:
            surface_map = activity_matrix.get(f"m{budget}", {})
            for surface in plan.SURFACES:
                rows = surface_map.get(surface, {})
                for row in ("A0", "A1"):
                    if row in rows:
                        cells.setdefault(row, {}).setdefault(budget, {})[surface] = {
                            "sessions": rows[row]["sessions"],
                        }
    specs = [
        RolloutSpec(row_id="C0", kind="policy", construction="raw"),
        RolloutSpec(row_id="C1", kind="policy", construction=winner_construction),
        RolloutSpec(row_id="P", kind="policy", construction="raw", precision_rule=True),
        RolloutSpec(row_id="O0", kind="oracle", construction="raw", horizon_H=horizon_H),
        RolloutSpec(row_id="O1", kind="oracle", construction=winner_construction, horizon_H=horizon_H),
        RolloutSpec(row_id="O2", kind="oracle", construction="true", horizon_H=horizon_H),
    ]
    if sensitivity:
        specs.append(
            RolloutSpec(row_id="O1_H10", kind="oracle", construction=winner_construction, horizon_H=10)
        )
    for spec in specs:
        budgets_for_spec = tuple(budgets) if spec.row_id != "O1_H10" else (4,)
        for budget in budgets_for_spec:
            for surface in plan.SURFACES:
                rows: list[Mapping[str, object]] = []
                for key in _ordered_session_keys(runtime, surface):
                    session = runtime._require_state().sessions[key]
                    if spec.kind == "policy":
                        rollout = runtime._rollout_policy(session=session, budget=budget, spec=spec)
                    else:
                        rollout = runtime._rollout_oracle(session=session, budget=budget, spec=spec)
                    rows.append(rollout["rows"][spec.row_id])
                cells.setdefault(spec.row_id, {}).setdefault(budget, {})[surface] = {"sessions": rows}
    matrix = summarize_cells(cells)
    base = runtime.root
    anchors: dict[str, object] = {}
    v8_score = _read_json(base / plan.V8_SCORE_RELATIVE)
    precision_score_payload = _read_json(base / plan.PRECISION_V2_SCORE_RELATIVE)
    for budget in budgets:
        for surface in plan.SURFACES:
            if "C0" in cells:
                anchors[f"C0_m{budget}_{surface}"] = anchor_row_sessions(
                    matrix[f"m{budget}"][surface]["C0"]["sessions"],
                    _anchor_sessions(payload=v8_score, budget=budget, surface=surface, system=v1plan.SYSTEM_CDMD),
                    label=f"C0 vs sealed V8 full CDM m{budget} {surface}",
                )
            if "P" in cells:
                anchors[f"P_m{budget}_{surface}"] = anchor_row_sessions(
                    matrix[f"m{budget}"][surface]["P"]["sessions"],
                    _anchor_sessions(
                        payload=precision_score_payload, budget=budget, surface=surface,
                        system="precision_v2", collection="precision_cells",
                    ),
                    label=f"P raw vs sealed Precision V2 m{budget} {surface}",
                )
    verdicts = policy.kill_criterion_verdicts(kill_matrix_view(matrix))
    return {
        "schema": "learned_gate_p2prime_stage_cop_v1",
        "stage": "c_o_p_rows",
        "identity_sha256": identity_sha256,
        "budgets": list(budgets),
        "horizon_H": horizon_H,
        "winner_construction": winner_construction,
        "matrix": matrix,
        "anchors": anchors,
        "kill_criterion_verdicts": verdicts,
        "resources": dict(runtime._resources()),
        "horizon_forward_count": runtime._horizon_single_pass_forwards,
        "target_optimizer_backward_update": 0,
    }


def _stage_m30(
    runtime: P2PrimeOracleMatrixRuntime,
    *,
    winner_construction: str,
    horizon_H: int,
    identity_sha256: str,
    a0_m30_reference: Mapping[str, object],
) -> dict[str, object]:
    spec = RolloutSpec(row_id="O1", kind="m30_oracle", construction=winner_construction, horizon_H=horizon_H)
    rows: list[Mapping[str, object]] = []
    for surface in plan.SURFACES:
        for key in _ordered_session_keys(runtime, surface):
            session = runtime._require_state().sessions[key]
            rollout = runtime._rollout_oracle(session=session, budget=30, spec=spec)
            rows.append(rollout["rows"]["O1"])
    cells: dict[str, dict[int, dict[str, dict[str, object]]]] = {
        "O1_M30_DIAGNOSTIC": {30: {
            surface: {"sessions": [row for row in rows if row["surface"] == surface]}
            for surface in plan.SURFACES
        }},
    }
    matrix = summarize_cells(cells)
    bitwise: dict[str, object] = {}
    for row in rows:
        reference = a0_m30_reference.get(str(row["session"]))
        if reference is not None:
            bitwise[str(row["session"])] = {
                "prediction_sha256_matches_A0_m30": str(row["prediction_sha256_raw"]) == str(reference),
            }
    return {
        "schema": "learned_gate_p2prime_stage_m30_v1",
        "stage": "m30_oracle_only_noop_diagnostic",
        "identity_sha256": identity_sha256,
        "winner_construction": winner_construction,
        "horizon_H": horizon_H,
        "matrix": matrix,
        "a0_m30_bitwise_reference": bitwise,
        "m30_carrier_bitwise_noop_all_sessions": all(
            bool(row.get("m30_carrier_bitwise_noop")) for row in rows
        ),
        "oracle_only_disclosure": (
            "M30 keeps the deployment no-op law: every proposal is rejected, the carrier state "
            "digest is verified constant every trial, and the diagnostic records what the "
            "coherent oracle would have accepted on a long reliable prefix."
        ),
        "resources": dict(runtime._resources()),
        "horizon_forward_count": runtime._horizon_single_pass_forwards,
        "target_optimizer_backward_update": 0,
    }


def run_stage(
    root: Path,
    *,
    gpu_index: int,
    stage: str,
    smoke: bool = False,
) -> Mapping[str, object]:
    """One P2' stage; every stage revalidates the pinned attempt contract."""
    base = Path(root).absolute()
    output = base / (plan.SMOKE_RESULT_ROOT_RELATIVE if smoke else plan.RESULT_ROOT_RELATIVE)
    _require(stage in ("attempt", "a", "substudy", "cop", "m30", "finalize", "smoke"),
             f"unknown P2' stage: {stage}")
    started = time.monotonic()
    if stage == "attempt":
        _require(not (output / "attempt.json").exists(), "P2' attempt already reserved")
        validate_environment(gpu_index=gpu_index)
        predecessor_score.validate_completed_v8_predecessor(base)
        profile = v1plan.COMPATIBLE_DEVICE_PROFILES[f"gpu{gpu_index}"]
        identity = predecessor_score.build_reviewed_identity(base, selected_device_profile=profile)
        _require(not output.exists(), "P2' result root already exists")
        output.mkdir(mode=0o755, parents=False, exist_ok=False)
        attempt = {
            "schema": "learned_gate_p2prime_attempt_v1",
            "status": "ATTEMPT_RESERVED",
            "cell": plan.CELL,
            "scope": "smoke" if smoke else "non_governing_inference_level_diagnostic",
            "design_authority": plan.DESIGN_RELATIVE,
            "identity_sha256": identity.sha256,
            "owned_sha256s": plan.owned_sha256s(base),
            "environment": validate_environment(gpu_index=gpu_index),
            "selected_device_profile": dict(profile),
            "sealed_cell_d_swa_sha256": plan.SEALED_CELL_D_SWA_SHA256,
            "pre_registration": plan.pre_registration_payload(),
            "rows": list(plan.ROWS),
            "budgets": list(plan.BUDGETS),
            "surfaces": list(plan.SURFACES),
            "smoke": bool(smoke),
            "smoke_parameters": dict(plan.SMOKE_PARAMETERS) if smoke else None,
            "inference_only": True,
            "target_optimizer_backward_update": 0,
        }
        attempt_sha = _publish(output / "attempt.json", attempt)
        return {"stage": "attempt", "attempt_sha256": attempt_sha, "result_root": str(output)}

    attempt_sha = _attempt_digest(base, output)
    budgets = (4,) if smoke else plan.DEPLOYMENT_BUDGETS
    stage_budgets_a = (4, 30) if smoke else plan.BUDGETS
    horizon_H = plan.SMOKE_PARAMETERS["horizon_H"] if smoke else int(plan.UTILITY["primary_horizon_H"])
    winner_construction = plan.SMOKE_PARAMETERS["c1_o1_construction"] if smoke else "smoothed_causal"

    if stage == "smoke":
        runtime, _meta, identity = _runtime_for(base, gpu_index=gpu_index)
        try:
            runtime.prepare(identity=identity)
            fixed = v1score.derive_fixed_evaluation_authority(base)
            runtime._materialize_subset(
                authority=fixed, per_surface=plan.SMOKE_PARAMETERS["sessions_per_surface"],
            )
            stage_a = _stage_a(runtime, budgets=(4, 30), identity_sha256=identity.sha256)
            substudy, _winner = _stage_substudy(
                runtime, budgets=(4,), horizon_H=horizon_H, identity_sha256=identity.sha256,
            )
            stage_cop = _stage_cop(
                runtime, budgets=(4,), horizon_H=horizon_H,
                winner_construction=winner_construction, identity_sha256=identity.sha256,
                sensitivity=False, activity_matrix=stage_a["matrix"],
            )
            reference: dict[str, object] = {}
            for surface in plan.SURFACES:
                for row in stage_a["matrix"]["m30"][surface]["A0"]["sessions"]:
                    reference[str(row["session"])] = row["prediction_sha256_raw"]
            stage_m30 = _stage_m30(
                runtime, winner_construction=winner_construction, horizon_H=horizon_H,
                identity_sha256=identity.sha256, a0_m30_reference=reference,
            )
            payload = {
                "schema": "learned_gate_p2prime_smoke_v1",
                "status": "TERMINAL_SMOKE",
                "scope": "non_governing_pipeline_smoke",
                "attempt_sha256": attempt_sha,
                "stage_a": stage_a,
                "stage_substudy": substudy,
                "stage_cop": stage_cop,
                "stage_m30": stage_m30,
                "smoke_parameters": dict(plan.SMOKE_PARAMETERS),
                "wall_seconds": float(time.monotonic() - started),
                "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                "model_or_checkpoint_updated": False,
                "target_optimizer_backward_update": 0,
            }
            sha = _publish(output / "smoke.json", payload)
            os.chmod(output, 0o555)
            return {"stage": "smoke", "smoke_sha256": sha, "verdict": "SMOKE_COMPLETE"}
        finally:
            runtime.close()

    if stage == "a":
        runtime, _meta, identity = _runtime_for(base, gpu_index=gpu_index)
        try:
            runtime.prepare(identity=identity)
            fixed = v1score.derive_fixed_evaluation_authority(base)
            runtime.materialize_inputs(identity=identity, authority=fixed)
            payload = _stage_a(runtime, budgets=stage_budgets_a, identity_sha256=identity.sha256)
        finally:
            runtime.close()
    elif stage == "substudy":
        runtime, _meta, identity = _runtime_for(base, gpu_index=gpu_index)
        try:
            runtime.prepare(identity=identity)
            fixed = v1score.derive_fixed_evaluation_authority(base)
            runtime.materialize_inputs(identity=identity, authority=fixed)
            payload, _winner = _stage_substudy(
                runtime, budgets=budgets, horizon_H=horizon_H, identity_sha256=identity.sha256,
            )
        finally:
            runtime.close()
    elif stage == "cop":
        substudy = _read_json(output / "stage_substudy.json")
        winner_construction = str(substudy["winner_selection"]["winner"])
        stage_a_matrix = _read_json(output / "stage_a.json")["matrix"]
        runtime, _meta, identity = _runtime_for(base, gpu_index=gpu_index)
        try:
            runtime.prepare(identity=identity)
            fixed = v1score.derive_fixed_evaluation_authority(base)
            runtime.materialize_inputs(identity=identity, authority=fixed)
            payload = _stage_cop(
                runtime, budgets=budgets, horizon_H=horizon_H,
                winner_construction=winner_construction, identity_sha256=identity.sha256,
                sensitivity=not smoke, activity_matrix=stage_a_matrix,
            )
        finally:
            runtime.close()
    elif stage == "m30":
        substudy = _read_json(output / "stage_substudy.json")
        winner_construction = str(substudy["winner_selection"]["winner"])
        stage_a = _read_json(output / "stage_a.json")
        reference = {}
        for surface in plan.SURFACES:
            for row in stage_a["matrix"]["m30"][surface]["A0"]["sessions"]:
                reference[str(row["session"])] = row["prediction_sha256_raw"]
        runtime, _meta, identity = _runtime_for(base, gpu_index=gpu_index)
        try:
            runtime.prepare(identity=identity)
            fixed = v1score.derive_fixed_evaluation_authority(base)
            runtime.materialize_inputs(identity=identity, authority=fixed)
            payload = _stage_m30(
                runtime, winner_construction=winner_construction, horizon_H=horizon_H,
                identity_sha256=identity.sha256, a0_m30_reference=reference,
            )
        finally:
            runtime.close()
    else:  # finalize
        stage_a = _read_json(output / "stage_a.json")
        substudy = _read_json(output / "stage_substudy.json")
        stage_cop = _read_json(output / "stage_cop.json")
        stage_m30 = _read_json(output / "stage_m30.json")
        verdicts = stage_cop["kill_criterion_verdicts"]
        winner_construction = str(substudy["winner_selection"]["winner"])
        noncoherent: dict[str, object] = {}
        for budget in plan.DEPLOYMENT_BUDGETS:
            rows_for_budget: dict[str, object] = {}
            for row_id, construction in (("O0", "raw"), ("O1", winner_construction), ("O2", "true")):
                rows_for_budget[row_id] = {
                    surface: _mean([
                        float(item[f"noncoherent_{construction}"]["NONCOHERENT_ONE_STEP_SWITCH_CEILING_r2"])
                        for item in substudy_sessions(substudy, budget=budget, surface=surface)
                    ])
                    for surface in plan.SURFACES
                }
            noncoherent[f"m{budget}"] = rows_for_budget
        result = {
            "schema": "learned_gate_p2prime_result_v1",
            "status": "TERMINAL",
            "scope": "non_governing_inference_level_diagnostic",
            "attempt_sha256": attempt_sha,
            "sealed_cell_d_swa_sha256": plan.SEALED_CELL_D_SWA_SHA256,
            "winner_construction": winner_construction,
            "matrix": stage_cop["matrix"],
            "activity_rows_matrix": stage_a["matrix"],
            "oracle_levels": {
                "NONCOHERENT_ONE_STEP_SWITCH_CEILING": noncoherent,
                "COHERENT_GREEDY_ORACLE": "matrix rows O0/O1/O2 (and O1_H10 sensitivity)",
                "noncoherent_status": plan.ORACLE_LEVELS["NONCOHERENT_ONE_STEP_SWITCH_CEILING"]["status"],
                "coherent_status": plan.ORACLE_LEVELS["COHERENT_GREEDY_ORACLE"]["status"],
            },
            "substudy": {
                "winner": winner_construction,
                "winner_selection": substudy["winner_selection"],
                "per_session": [
                    {
                        "session": item["session"], "budget": item["budget"], "surface": item["surface"],
                        "direction_rows": item["direction_rows"], "utilities": item["utilities"],
                        "noncoherent": {
                            name: {
                                key: value for key, value in item[f"noncoherent_{name}"].items()
                                if key != "status"
                            }
                            for name in plan.PSEUDO_CONSTRUCTIONS
                        },
                    }
                    for item in substudy["sessions"]
                ],
            },
            "contrast_roles": {f"{row}_minus_{baseline}": role for row, baseline, role in plan.CONTRASTS},
            "kill_criterion_verdicts": verdicts,
            "compose_reading": stage_a["compose_reading"],
            "anchors": {**stage_a["anchors"], **stage_cop["anchors"]},
            "m30_noop_verification": {
                "carrier_bitwise_noop_all_sessions": stage_m30["m30_carrier_bitwise_noop_all_sessions"],
                "a0_m30_bitwise_reference": stage_m30["a0_m30_bitwise_reference"],
                "oracle_only_disclosure": stage_m30["oracle_only_disclosure"],
            },
            "disclosures": {
                "leakage_diagnostic_rows": [
                    "O0", "O1", "O2", "O1_M30_DIAGNOSTIC", "substudy direction errors and utilities",
                ],
                "output_filter_pre_registration": plan.OUTPUT_FILTER["disclosure"],
                "zero_target_optimizer_backward": True,
                "model_or_checkpoint_updated": False,
                "beta_distance_labels_used": False,
                "h10_sensitivity": "O1_H10 rows in the stage_cop matrix (budget 4 only)",
                "anchors_note": (
                    "A0 anchors to the sealed activity-only receipt; C0 anchors to the sealed V8 "
                    "full-CDM receipt; P anchors (raw, unfiltered) to the sealed Precision V2 "
                    "receipt; A0@M30 anchors to the sealed V8 static M30 cells."
                ),
            },
            "target_optimizer_backward_update": 0,
        }
        result_sha = _publish(output / "result.json", result)
        fired = [
            name for name, item in verdicts.items()
            if isinstance(item, dict) and item.get("fired") and name != "KC5_EXTERNAL_MATCHED_SCORE_AUTHORIZATION"
        ]
        terminal = {
            "schema": "learned_gate_p2prime_terminal_v1",
            "status": "TERMINAL",
            "attempt_sha256": attempt_sha,
            "result_sha256": result_sha,
            "verdict": verdicts["verdict"],
            "kill_criteria_fired": fired,
            "kc4_status": verdicts["KC4_HEADROOM_NOT_IDENTIFIABLE"]["status"],
            "target_optimizer_backward_update": 0,
        }
        terminal_sha = _publish(output / "terminal.json", terminal)
        os.chmod(output, 0o555)
        return {
            "stage": "finalize", "result_sha256": result_sha, "terminal_sha256": terminal_sha,
            "verdict": verdicts["verdict"],
        }

    payload["attempt_sha256"] = attempt_sha
    payload["wall_seconds"] = float(time.monotonic() - started)
    payload["rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    payload["model_or_checkpoint_updated"] = False
    payload["target_optimizer_backward_update"] = 0
    stage_sha = _publish(output / f"stage_{stage}.json", payload)
    return {"stage": stage, "stage_sha256": stage_sha, "result_root": str(output)}

