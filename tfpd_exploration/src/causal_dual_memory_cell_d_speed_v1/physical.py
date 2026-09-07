"""Opt-in O1/O2 runtime composition over V8 and Precision-V2.

Nothing in this module changes V8/Precision lifecycle, parser, transition,
metric, authority, or model bytes.  The mixin replaces only two expensive
evaluation forward helpers after the already-reviewed runtime has reached its
physical prepare boundary.
"""
from __future__ import annotations

import hashlib
import importlib
import pickle
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.causal_dual_memory_cell_d_score_v1 import physical as v1physical
from src.causal_dual_memory_cell_d_score_v8 import physical as v8physical
from src.precision_aware_causal_dual_memory_cell_d_score_v1 import physical as precisionphysical

from . import plan
from .accelerator import IdentityAccelerationError, IdentityCacheAccelerator


class PhysicalCDMDSpeedError(RuntimeError):
    """Fail closed for an acceleration seam or invariant drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhysicalCDMDSpeedError(message)


def _rng_digest(torch: Any) -> str:
    """Hash Python/NumPy/Torch state without mutating any of those domains."""
    digest = hashlib.sha256()
    digest.update(pickle.dumps(random.getstate(), protocol=4))
    digest.update(pickle.dumps(np.random.get_state(), protocol=4))
    digest.update(bytes(torch.get_rng_state().detach().cpu().contiguous().numpy()))
    # This code is reached only after the inherited score runtime has already
    # initialized its reviewed CUDA device.  Do not initialize CUDA merely to
    # collect a proof at the no-data/static boundary.
    if bool(torch.cuda.is_initialized()):
        for state in torch.cuda.get_rng_state_all():
            digest.update(bytes(state.detach().cpu().contiguous().numpy()))
    return digest.hexdigest()


class SpeedAcceleratedRuntimeMixin:
    """MRO-safe O1/O2 mixin for V8 and Precision V2 runtime subclasses.

    It wraps ``_score_session`` rather than reproducing its causal loop.  The
    inherited implementation still chooses every input, calls exactly one
    observe/proposal/commit path, and computes the governing metric.  This
    mixin only provides cached full/held forwards for the original hook calls.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._speed_accelerator = IdentityCacheAccelerator()
        self._speed_active_session: tuple[str, str, int, str] | None = None
        self._speed_next_query_index = 0
        self._speed_query_trial_ids: tuple[str, ...] = ()
        self._speed_session_evidence: dict[tuple[str, str, int, str], Mapping[str, object]] = {}

    def _score_session(self, *, session: Any, budget: int, system: str) -> Any:
        query_ids = tuple(session.query_trial_ids[budget])
        key = (str(session.surface), str(session.session), int(budget), str(system))
        _require(self._speed_active_session is None, "accelerated session nesting drift")
        self._speed_active_session = key
        self._speed_next_query_index = 0
        self._speed_query_trial_ids = tuple(str(value) for value in query_ids)
        _require(len(self._speed_query_trial_ids) == len(query_ids) and all(self._speed_query_trial_ids),
                 "accelerated query-trial identifier topology drift")
        self._speed_accelerator.begin_session(total_query_trials=len(query_ids))
        try:
            result = super()._score_session(session=session, budget=budget, system=system)
            _require(self._speed_next_query_index == len(query_ids),
                     "accelerated full-forward/query cardinality drift")
            evidence = self._speed_accelerator.payload()
            self._speed_session_evidence[key] = evidence
            return result
        except IdentityAccelerationError as error:
            raise PhysicalCDMDSpeedError(str(error)) from error
        finally:
            self._speed_active_session = None
            self._speed_query_trial_ids = ()

    def _make_session_score(self, **kwargs: Any) -> Any:
        """Replace inherited duplicate-forward accounting with observed counts.

        The V1 loop calculates group count as ``2 * logical_chunks`` because
        its historical helper always duplicated every chunk.  O2 deliberately
        no longer has that equivalence, so leave the evidence type/codec to the
        inherited successor but pass the real model-call totals.
        """
        system = kwargs.get("system")
        if system == v1physical.plan.SYSTEM_CDMD:
            self._speed_accelerator.repeat_audit.require_complete_coverage(require_held_group_0=True)
        elif system == v1physical.plan.SYSTEM_SEALED:
            self._speed_accelerator.repeat_audit.require_complete_coverage(require_held_group_0=False)
        else:
            raise PhysicalCDMDSpeedError("accelerated system repeat-audit topology drift")
        payload = self._speed_accelerator.payload()
        by_path = payload.get("actual_model_forward_count_by_path")
        _require(isinstance(by_path, Mapping), "accelerated forward accounting payload drift")
        full = by_path.get("full", 0)
        groups = sum(value for key, value in by_path.items() if isinstance(key, str) and key.startswith("held_group_"))
        _require(type(full) is int and full >= 0 and type(groups) is int and groups >= 0,
                 "accelerated actual forward accounting type drift")
        adjusted = dict(kwargs)
        adjusted["full_system_forward_count"] = full
        adjusted["group_forward_count"] = groups
        return super()._make_session_score(**adjusted)

    def speed_evidence_for_session(
        self, *, surface: str, session: str, budget: int, system: str,
    ) -> Mapping[str, object]:
        """Return receipt-ready compact speed proof after an inherited score."""
        key = (surface, session, budget, system)
        value = self._speed_session_evidence.get(key)
        _require(isinstance(value, Mapping), "accelerated session evidence is absent")
        return dict(value)

    def _begin_full_query_coordinate(self) -> int:
        _require(self._speed_active_session is not None, "accelerated full forward outside a session")
        query_index = self._speed_next_query_index
        _require(0 <= query_index < len(self._speed_query_trial_ids),
                 "accelerated full-forward query identifier drift")
        self._speed_accelerator.begin_query(
            query_index, query_trial_id=self._speed_query_trial_ids[query_index],
        )
        self._speed_next_query_index += 1
        return query_index

    def _forward_full(
        self, *, neural_windows: Any, activity_stack: Any, normalized_t4: Any,
    ) -> tuple[Any, Mapping[str, object]]:
        """Exact full evaluator path with B=1 identity and sampled repeats."""
        state = self._require_state()
        torch, np_module, pop, arm = (
            state.modules["torch"], state.modules["np"], state.modules["pop_robust"], state.modules["arm_common"],
        )
        query_index = self._begin_full_query_coordinate()
        neural_cpu = np_module.ascontiguousarray(np_module.asarray(neural_windows), dtype=np_module.float32)
        stack_cpu = np_module.ascontiguousarray(np_module.asarray(activity_stack), dtype=np_module.float32)
        side_cpu = np_module.ascontiguousarray(np_module.asarray(normalized_t4), dtype=np_module.float32)
        if (
            neural_cpu.ndim != 3 or neural_cpu.shape[1] != v1physical.plan.WINDOW_BINS
            or stack_cpu.ndim != 3 or not 1 <= stack_cpu.shape[0] <= 30 or stack_cpu.shape[1] != 100
            or side_cpu.shape != (neural_cpu.shape[2], 4) or stack_cpu.shape[2] != neural_cpu.shape[2]
        ):
            raise PhysicalCDMDSpeedError("accelerated full variable-prefix input shape drift")
        neural = torch.as_tensor(neural_cpu, dtype=torch.float32, device=state.device)
        stack = torch.as_tensor(stack_cpu, dtype=torch.float32, device=state.device)
        side = torch.as_tensor(side_cpu, dtype=torch.float32, device=state.device)
        _require(not bool(state.model.training), "accelerated sealed Cell-D model must be eval mode")
        before_model = arm.state_sha256(state.model)
        rng_before = _rng_digest(torch)
        with pop.dynamic_dropout_recorder() as recorder:
            result = self._speed_accelerator.forward(
                model=state.model, neural_windows=neural, activity_stack=stack, normalized_t4=side,
                path="full",
            )
        rng_after = _rng_digest(torch)
        after_model = arm.state_sha256(state.model)
        if (
            before_model != after_model or rng_before != rng_after
            or recorder.get("uniform_calls") != 0 or recorder.get("dropout_calls") != []
        ):
            raise PhysicalCDMDSpeedError("accelerated full eval/state/RNG/dropout purity drift")
        state.forward_chunks += int(result.actual_model_forward_count)
        state.full_system_forwards += int(result.actual_model_forward_count)
        evidence = {
            "model_state_before_sha256": before_model,
            "model_state_after_sha256": after_model,
            "prediction_sha256": result.prediction_sha256,
            "full_system_forward_count": int(result.actual_model_forward_count),
            "logical_forward_chunk_count": int(result.logical_chunk_count),
            "identity_encoder_forward_count": int(result.identity_encoder_forward_count),
            "identity_cache_key_sha256": result.cache_key_sha256,
            "identity_sha256": result.identity_sha256,
            "dropout_calls": 0,
            "rng_before_sha256": rng_before,
            "rng_after_sha256": rng_after,
            "rng_unchanged": True,
            "repeated_outputs_bitwise_equal": True,
            "prefix_length": int(stack_cpu.shape[0]),
            "query_index": query_index,
            "logical_eval_batch_size": plan.LOGICAL_EVAL_BATCH_SIZE,
        }
        return result.prediction, evidence

    def _accelerated_held_forward(
        self,
        *,
        trial: Any,
        activity_stack: Any,
        normalized_t4: Any,
        held_mask: Any,
        expected_prefix_length: int,
        expected_prefix_activity_sha256: str,
        group: int,
    ) -> tuple[Any, Mapping[str, object]]:
        """Use V8-authenticated slicing, then only replace eager identity work."""
        state = self._require_state()
        torch, np_module, pop, arm = (
            state.modules["torch"], state.modules["np"], state.modules["pop_robust"], state.modules["arm_common"],
        )
        helper = state.source_physical_helpers
        slicer = vars(helper).get("physically_slice_variable_prefix_held_units")
        _require(callable(slicer), "accelerated V8 helper lacks authenticated held slicer")
        stack_cpu = np_module.ascontiguousarray(np_module.asarray(activity_stack), dtype=np_module.float32)
        neural_cpu = np_module.ascontiguousarray(np_module.asarray(trial.neural_windows), dtype=np_module.float32)
        side_cpu = np_module.ascontiguousarray(np_module.asarray(normalized_t4), dtype=np_module.float32)
        held_cpu = np_module.ascontiguousarray(np_module.asarray(held_mask), dtype=np_module.bool_)
        sliced = slicer(
            neural_cpu, stack_cpu, side_cpu, held_cpu,
            expected_prefix_length=expected_prefix_length,
            full_prefix_activity_sha256=expected_prefix_activity_sha256,
        )
        neural = torch.as_tensor(sliced.neural_windows, dtype=torch.float32, device=state.device)
        stack = torch.as_tensor(sliced.b3s_activity_stack, dtype=torch.float32, device=state.device)
        side = torch.as_tensor(sliced.normalized_t4, dtype=torch.float32, device=state.device)
        _require(int(stack.shape[0]) == expected_prefix_length and int(neural.shape[0]) >= 1,
                 "accelerated held variable-prefix topology drift")
        _require(not bool(state.model.training), "accelerated held sealed Cell-D model must be eval mode")
        before_model = arm.state_sha256(state.model)
        rng_before = _rng_digest(torch)
        with pop.dynamic_dropout_recorder() as recorder:
            result = self._speed_accelerator.forward(
                model=state.model, neural_windows=neural, activity_stack=stack, normalized_t4=side,
                path=f"held_group_{group}", held_mask=held_cpu,
            )
        rng_after = _rng_digest(torch)
        after_model = arm.state_sha256(state.model)
        if (
            before_model != after_model or rng_before != rng_after
            or recorder.get("uniform_calls") != 0 or recorder.get("dropout_calls") != []
        ):
            raise PhysicalCDMDSpeedError("accelerated held eval/state/RNG/dropout purity drift")
        velocity = result.prediction[:, -1, :].detach().cpu().numpy().astype(np_module.float64, copy=False)
        _require(velocity.shape == (trial.velocity_validity.valid_mask.size, 2),
                 "accelerated held last-bin/validity topology drift")
        source_physical = importlib.import_module("src.causal_dual_memory_cell_d_v1.physical")
        restored = source_physical.restore_physical_velocity(
            velocity, behavior_mean=state.executor_state.behavior_normalizer.mean,
            behavior_std=state.executor_state.behavior_normalizer.std,
        )
        completed = state.modules["core"].CompletedVelocityPrediction(restored, trial.velocity_validity)
        proof = {
            "model_state_before_sha256": before_model,
            "model_state_after_sha256": after_model,
            "rng_before": rng_before,
            "rng_after": rng_after,
            "rng_unchanged": True,
            "dynamic_dropout_calls_before": 0,
            "dynamic_dropout_calls_after": 0,
            "dynamic_dropout_unchanged": True,
            "repeated_outputs_bitwise_equal": True,
            "prefix_length": expected_prefix_length,
            "forward_chunk_count": int(result.logical_chunk_count),
            "actual_model_forward_count": int(result.actual_model_forward_count),
            "identity_encoder_forward_count": int(result.identity_encoder_forward_count),
            "identity_cache_key_sha256": result.cache_key_sha256,
            "identity_sha256": result.identity_sha256,
            "max_endpoints_per_forward_chunk": plan.LOGICAL_EVAL_BATCH_SIZE,
            "prefix_activity_sha256": sliced.prefix_activity_sha256,
            "retained_channel_indices": [int(value) for value in sliced.retained_channel_indices],
            "held_channel_indices": [int(value) for value in sliced.held_channel_indices],
            "logical_eval_batch_size": plan.LOGICAL_EVAL_BATCH_SIZE,
        }
        return completed, proof

    def _group_predictions(self, *, trial: Any, memory: Any) -> tuple[tuple[Any, ...], list[Mapping[str, object]]]:
        """Retain the V1 group topology, replacing only the redundant encoder work."""
        state = self._require_state()
        inputs = memory.read_prediction_inputs()
        expected_prefix = int(inputs.activity_trials.shape[0])
        if expected_prefix != int(memory.state.carrier.config.support_budget_m) + int(memory.state.activity.query_count):
            raise PhysicalCDMDSpeedError("accelerated causal activity prefix count/state drift")
        _require(self._speed_active_session is not None and self._speed_next_query_index > 0,
                 "accelerated group forward must follow its full query forward")
        normalized = self._normalized_side(inputs.active_t4).detach().cpu().numpy()
        activity_digest = state.source_physical_helpers._variable_prefix_array_digest(inputs.activity_trials)
        predictions: list[Any] = []
        evidence: list[Mapping[str, object]] = []
        actual_forwards = 0
        for group in range(v1physical.plan.GROUP_COUNT):
            prediction, proof = self._accelerated_held_forward(
                trial=trial, activity_stack=inputs.activity_trials, normalized_t4=normalized,
                held_mask=memory.state.carrier.groups.held_mask(group),
                expected_prefix_length=expected_prefix, expected_prefix_activity_sha256=activity_digest,
                group=group,
            )
            predictions.append(prediction)
            evidence.append(proof)
            actual_forwards += int(proof["actual_model_forward_count"])
        _require(len(predictions) == v1physical.plan.GROUP_COUNT,
                 "accelerated CDM-D must finalize exactly four held groups")
        state.forward_chunks += actual_forwards
        return tuple(predictions), evidence


class SpeedV8ReviewedCDMScoreRuntime(SpeedAcceleratedRuntimeMixin, v8physical.V8ReviewedCDMScoreRuntime):
    """V8 evaluator with only O1/O2 physical-forward replacement."""


class SpeedPrecisionV2ReviewedCDMScoreRuntime(
    SpeedAcceleratedRuntimeMixin, precisionphysical.PrecisionV2ReviewedCDMScoreRuntime,
):
    """Precision-V2 evaluator retaining its exact posterior commit veto hook."""


def build_speed_v8_runtime(*, root: Path, selected_device_profile: Mapping[str, object]) -> SpeedV8ReviewedCDMScoreRuntime:
    """Construct an unprepared V8 composition object; no I/O or CUDA action."""
    return SpeedV8ReviewedCDMScoreRuntime(root=Path(root), selected_device_profile=selected_device_profile)


def build_speed_precision_v2_runtime(
    *, root: Path, selected_device_profile: Mapping[str, object],
) -> SpeedPrecisionV2ReviewedCDMScoreRuntime:
    """Construct an unprepared Precision-V2 composition object; no I/O/CUDA."""
    return SpeedPrecisionV2ReviewedCDMScoreRuntime(root=Path(root), selected_device_profile=selected_device_profile)
