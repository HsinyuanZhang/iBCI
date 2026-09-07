"""Deferred GPU0 composition of the V2 environment gate and Stage-0 runtime.

No parser, checkpoint loader, target metric, causal transition, or model
forward is copied here.  The sole new physical seam is a typed runtime factory
that supplies the accepted O1/O2 ``SpeedPrecisionV2ReviewedCDMScoreRuntime``
to the existing Precision-V2 evaluator behind its corrected source-root gate.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Mapping

from src.causal_dual_memory_cell_d_score_v1 import plan as base_plan
from src.causal_dual_memory_cell_d_score_v1 import physical as v1_physical
from src.causal_dual_memory_cell_d_speed_v1 import physical as speed_stage0_physical
from src.causal_dual_memory_cell_d_speed_v1 import accelerator as speed_stage0_accelerator
from src.precision_aware_causal_dual_memory_cell_d_score_v1 import physical as precision_physical
from src.precision_aware_causal_dual_memory_cell_d_score_v2 import physical as precision_v2_physical
from src.precision_aware_causal_dual_memory_cell_d_score_v2 import plan as precision_v2_plan
from src.precision_aware_causal_dual_memory_cell_d_score_v2 import score as precision_v2_score

from . import plan, score


class PhysicalSpeedSmokeError(score.SpeedSmokeError):
    """Fail closed for the narrow GPU0 V2-wrapper/runtime-factory seam."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhysicalSpeedSmokeError(message)


class _PhysicalBatchIdentityCacheAccelerator(speed_stage0_accelerator.IdentityCacheAccelerator):
    """Route-local B1024/B512 decoder loop over the reviewed O1 cache.

    The Stage-0 accelerator deliberately freezes its own B128 route.  This
    narrow subclass retains its exact cache-key, B=1 identity, repeat-audit,
    and accounting primitives while changing only the physical decoder chunk
    size for the current numerical-equivalence smoke.  It owns no parser,
    transition, model state, metric, or lifecycle behavior.
    """

    def __init__(self, *, physical_eval_batch_size: int) -> None:
        _require(
            physical_eval_batch_size in plan.OPTIMIZED_PHYSICAL_EVAL_BATCH_CANDIDATES,
            "speed-smoke optimized physical batch candidate drift",
        )
        super().__init__(logical_batch_size=plan.LOGICAL_EVAL_BATCH_SIZE)
        self.physical_eval_batch_size = int(physical_eval_batch_size)

    def forward(
        self,
        *, model: Any, neural_windows: Any, activity_stack: Any, normalized_t4: Any,
        path: str, held_mask: Any | None = None,
    ) -> speed_stage0_accelerator.AcceleratedForwardResult:
        """Decode physical chunks while retaining B128 logical accounting."""
        import torch

        _require(
            getattr(neural_windows, "ndim", None) == 3 and int(neural_windows.shape[0]) >= 1,
            "speed-smoke optimized neural windows topology drift",
        )
        entry, cache_miss = self._entry(
            model=model, path=path, activity_stack=activity_stack,
            normalized_t4=normalized_t4, held_mask=held_mask,
        )
        _require(
            int(neural_windows.shape[2]) == int(entry.identity.shape[1]),
            "speed-smoke optimized neural/identity unit topology drift",
        )
        chunks: list[Any] = []
        digest = hashlib.sha256()
        actual_forwards = 0
        for start in range(0, int(neural_windows.shape[0]), self.physical_eval_batch_size):
            stop = min(start + self.physical_eval_batch_size, int(neural_windows.shape[0]))
            with torch.no_grad():
                first_result = model(neural_windows[start:stop], identity=entry.identity)
            _require(
                isinstance(first_result, tuple) and len(first_result) == 2,
                "speed-smoke optimized direct identity forward topology drift",
            )
            first, returned_identity = first_result
            _require(
                torch.equal(returned_identity, entry.identity) and bool(torch.isfinite(first).all().item()),
                "speed-smoke optimized identity/prediction finiteness drift",
            )
            actual_forwards += 1
            reasons = self.repeat_audit.reasons_for(
                path=path, cache_key_sha256=entry.key_sha256, chunk_start=start,
            )
            if reasons:
                with torch.no_grad():
                    second_result = model(neural_windows[start:stop], identity=entry.identity)
                _require(
                    isinstance(second_result, tuple) and len(second_result) == 2,
                    "speed-smoke optimized repeated identity forward topology drift",
                )
                second, second_identity = second_result
                _require(
                    torch.equal(first, second) and torch.equal(second_identity, entry.identity),
                    "speed-smoke optimized sampled repeat drift",
                )
                actual_forwards += 1
                self.repeat_audit.record(
                    path=path, cache_key_sha256=entry.key_sha256, chunk_start=start,
                    repeated_prediction_sha256=speed_stage0_accelerator.canonical_array_sha256(
                        first, label="repeat_prediction",
                    ),
                    outputs_bitwise_equal=True, reasons=reasons,
                )
            detached = first.detach().cpu().contiguous()
            digest.update(detached.numpy().tobytes())
            chunks.append(detached)
        logical_chunks = (
            int(neural_windows.shape[0]) + plan.LOGICAL_EVAL_BATCH_SIZE - 1
        ) // plan.LOGICAL_EVAL_BATCH_SIZE
        self._actual_model_forwards += actual_forwards
        self._actual_model_forwards_by_path[path] = (
            self._actual_model_forwards_by_path.get(path, 0) + actual_forwards
        )
        self._logical_chunks_by_path[path] = self._logical_chunks_by_path.get(path, 0) + logical_chunks
        return speed_stage0_accelerator.AcceleratedForwardResult(
            prediction=torch.cat(chunks, dim=0), prediction_sha256=digest.hexdigest(),
            logical_chunk_count=logical_chunks, actual_model_forward_count=actual_forwards,
            identity_encoder_forward_count=1 if cache_miss else 0,
            cache_key_sha256=entry.key_sha256, identity_sha256=entry.identity_sha256,
            repeated_outputs_bitwise_equal=True,
        )

    def payload(self) -> dict[str, object]:
        return {
            **super().payload(),
            "physical_eval_batch_size": self.physical_eval_batch_size,
            "logical_eval_batch_size": plan.LOGICAL_EVAL_BATCH_SIZE,
        }


class _ComparativeSpeedPrecisionV2Runtime(speed_stage0_physical.SpeedPrecisionV2ReviewedCDMScoreRuntime):
    """One eager V2 anchor followed by one O1/O2 numerical-comparison run."""

    def __init__(self, *, root: Path, selected_device_profile: Mapping[str, object]) -> None:
        super().__init__(root=Path(root), selected_device_profile=selected_device_profile)
        self._comparison_mode: str | None = None
        self._comparison_full_predictions: dict[str, list[Any]] = {"baseline": [], "optimized": []}

    def _forward_full(
        self, *, neural_windows: Any, activity_stack: Any, normalized_t4: Any,
    ) -> tuple[Any, Mapping[str, object]]:
        if self._comparison_mode == "baseline":
            prediction, proof = v1_physical.ReviewedCDMScoreRuntime._forward_full(
                self, neural_windows=neural_windows, activity_stack=activity_stack, normalized_t4=normalized_t4,
            )
        elif self._comparison_mode == "optimized":
            prediction, proof = super()._forward_full(
                neural_windows=neural_windows, activity_stack=activity_stack, normalized_t4=normalized_t4,
            )
        else:
            raise PhysicalSpeedSmokeError("speed-smoke full forward outside comparative mode")
        self._comparison_full_predictions[self._comparison_mode].append(prediction.detach().cpu().contiguous())
        return prediction, proof

    def _group_predictions(self, *, trial: Any, memory: Any) -> tuple[tuple[Any, ...], list[Mapping[str, object]]]:
        if self._comparison_mode == "baseline":
            return v1_physical.ReviewedCDMScoreRuntime._group_predictions(self, trial=trial, memory=memory)
        if self._comparison_mode == "optimized":
            return super()._group_predictions(trial=trial, memory=memory)
        raise PhysicalSpeedSmokeError("speed-smoke held-group forward outside comparative mode")

    def _make_session_score(self, **kwargs: Any) -> Any:
        """Keep the eager anchor on its inherited, non-accelerated codec path."""
        if self._comparison_mode == "baseline":
            return precision_physical.PrecisionV2ReviewedCDMScoreRuntime._make_session_score(self, **kwargs)
        if self._comparison_mode == "optimized":
            return super()._make_session_score(**kwargs)
        raise PhysicalSpeedSmokeError("speed-smoke session codec outside comparative mode")

    def _clear_precision_session_bindings(self) -> None:
        # Precision-V2 binds one support posterior to each fresh causal memory.
        # A baseline comparison is intentionally independent of the optimized
        # rerun, so neither may retain the other's local memory binding.
        self._precision_by_memory.clear()
        self._precision_by_session.clear()

    @staticmethod
    def _is_cuda_oom(error: BaseException, torch: Any) -> bool:
        oom_type = getattr(torch, "OutOfMemoryError", ())
        return isinstance(error, oom_type) or "out of memory" in str(error).lower()

    def _run_comparison_pass(self, *, mode: str, session: Any, budget: int, physical_batch_size: int) -> tuple[Any, float]:
        state = self._require_state()
        torch = state.modules["torch"]
        _require(mode in {"baseline", "optimized"}, "speed-smoke comparison mode drift")
        if mode == "baseline":
            _require(
                physical_batch_size == plan.BASELINE_PHYSICAL_EVAL_BATCH_SIZE
                and base_plan.EVAL_BATCH_SIZE == plan.BASELINE_PHYSICAL_EVAL_BATCH_SIZE,
                "speed-smoke eager baseline must retain exact physical B128",
            )
        else:
            _require(
                physical_batch_size in plan.OPTIMIZED_PHYSICAL_EVAL_BATCH_CANDIDATES,
                "speed-smoke optimized physical batch candidate drift",
            )
        self._comparison_mode = mode
        self._comparison_full_predictions[mode].clear()
        if mode == "optimized":
            self._speed_accelerator = _PhysicalBatchIdentityCacheAccelerator(
                physical_eval_batch_size=physical_batch_size,
            )
        torch.cuda.synchronize(0)
        started = time.monotonic()
        raised: BaseException | None = None
        try:
            if mode == "baseline":
                result = v1_physical.ReviewedCDMScoreRuntime._score_session(
                    self, session=session, budget=budget, system=base_plan.SYSTEM_CDMD,
                )
            else:
                result = super()._score_session(
                    session=session, budget=budget, system=base_plan.SYSTEM_CDMD,
                )
        except BaseException as error:
            # A CUDA OOM must reach the caller intact so its predeclared
            # B1024 -> B512 -> B128 fallback can run.  In particular, do not
            # let a best-effort post-error synchronize replace its type/text.
            raised = error
            raise
        finally:
            try:
                torch.cuda.synchronize(0)
            except BaseException:
                if raised is None:
                    raise
            finally:
                self._comparison_mode = None
        elapsed = float(time.monotonic() - started)
        _require(elapsed > 0.0, "speed-smoke comparison wall-clock evidence drift")
        return result, elapsed

    def score_eager_baseline_and_optimized(
        self, *, session: Any, budget: int,
    ) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
        """Run exactly one eager anchor, then O1/O2 with deterministic OOM fallback."""
        state = self._require_state()
        torch = state.modules["torch"]
        _require(budget == 4, "speed-smoke permits only the fixed M4 external row")
        self._clear_precision_session_bindings()
        baseline_object, baseline_wall = self._run_comparison_pass(
            mode="baseline", session=session, budget=budget,
            physical_batch_size=plan.BASELINE_PHYSICAL_EVAL_BATCH_SIZE,
        )
        baseline = baseline_object.payload(budget=budget, query_trial_ids=session.query_trial_ids[budget])
        baseline_predictions = tuple(self._comparison_full_predictions["baseline"])
        _require(baseline_predictions, "speed-smoke baseline predictions are absent")
        self._clear_precision_session_bindings()
        attempts: list[dict[str, object]] = []
        optimized_object: Any | None = None
        optimized_wall = 0.0
        selected_batch: int | None = None
        for candidate in plan.OPTIMIZED_PHYSICAL_EVAL_BATCH_CANDIDATES:
            try:
                optimized_object, optimized_wall = self._run_comparison_pass(
                    mode="optimized", session=session, budget=budget, physical_batch_size=candidate,
                )
                selected_batch = candidate
                attempts.append({"physical_eval_batch_size": candidate, "outcome": "selected"})
                break
            except BaseException as error:
                if not self._is_cuda_oom(error, torch):
                    raise
                attempts.append({"physical_eval_batch_size": candidate, "outcome": "cuda_oom_fallback"})
                self._clear_precision_session_bindings()
                self._comparison_full_predictions["optimized"].clear()
                self._comparison_mode = None
                torch.cuda.empty_cache()
        _require(optimized_object is not None and selected_batch is not None,
                 "speed-smoke all optimized physical batch candidates exhausted")
        optimized = optimized_object.payload(budget=budget, query_trial_ids=session.query_trial_ids[budget])
        optimized_predictions = tuple(self._comparison_full_predictions["optimized"])
        _require(len(baseline_predictions) == len(optimized_predictions) and optimized_predictions,
                 "speed-smoke baseline/optimized query prediction topology drift")
        baseline_tensor = torch.cat(list(baseline_predictions), dim=0)
        optimized_tensor = torch.cat(list(optimized_predictions), dim=0)
        _require(tuple(baseline_tensor.shape) == tuple(optimized_tensor.shape),
                 "speed-smoke baseline/optimized prediction shape drift")
        max_abs = float(torch.max(torch.abs(baseline_tensor - optimized_tensor)).item())
        baseline_transitions = baseline.get("transition_records")
        optimized_transitions = optimized.get("transition_records")
        _require(
            isinstance(baseline_transitions, list) and baseline_transitions == optimized_transitions,
            "speed-smoke optimized causal transition sequence drift",
        )
        stage0_evidence = self.speed_evidence_for_session(
            surface=session.surface, session=session.session, budget=budget, system=base_plan.SYSTEM_CDMD,
        )
        speed_evidence = {
            "schema": "precision_aware_cdmd_speed_smoke_o1_o2_evidence_v2",
            "stage0_o1_o2_engine_evidence": dict(stage0_evidence),
            "physical_eval_batch_size": selected_batch,
            "physical_eval_batch_candidates": list(plan.OPTIMIZED_PHYSICAL_EVAL_BATCH_CANDIDATES),
            "fallback_attempts": attempts,
        }
        comparison = {
            "schema": "precision_aware_cdmd_speed_smoke_numerical_comparison_v1",
            "baseline_physical_eval_batch_size": plan.BASELINE_PHYSICAL_EVAL_BATCH_SIZE,
            "optimized_physical_eval_batch_size": selected_batch,
            "baseline_prediction_sha256": baseline["prediction_sha256"],
            "optimized_prediction_sha256": optimized["prediction_sha256"],
            "max_abs_prediction_error": max_abs,
            "prediction_max_abs_tolerance": plan.PREDICTION_MAX_ABSOLUTE_TOLERANCE,
            "baseline_governing_r2": baseline["governing_r2"],
            "optimized_governing_r2": optimized["governing_r2"],
            "r2_abs_error": abs(float(baseline["governing_r2"]) - float(optimized["governing_r2"])),
            "r2_absolute_tolerance": plan.R2_ABSOLUTE_TOLERANCE,
            "baseline_transition_records_sha256": score._digest(baseline_transitions),
            "optimized_transition_records_sha256": score._digest(optimized_transitions),
            "transition_sequence_exact": True,
            "baseline_wall_seconds": baseline_wall,
            "optimized_wall_seconds": optimized_wall,
            "speedup_ratio": baseline_wall / optimized_wall,
            "baseline_full_system_forward_count": baseline["full_system_forward_count"],
            "baseline_group_forward_count": baseline["group_forward_count"],
            "optimized_full_system_forward_count": optimized["full_system_forward_count"],
            "optimized_group_forward_count": optimized["group_forward_count"],
            "smoke_min_speedup_ratio_exclusive": plan.SMOKE_MIN_SPEEDUP_RATIO,
            "recommended_speedup_ratio": plan.RECOMMENDED_SPEEDUP_RATIO,
        }
        self._comparison_full_predictions["baseline"].clear()
        self._comparison_full_predictions["optimized"].clear()
        return baseline, optimized, comparison, speed_evidence


def _build_v2_environment_identity(root: Path, identity: score.SpeedSmokeIdentity) -> precision_v2_plan.ScoreIdentity:
    """Create only the typed V2 environment view needed by its reviewed gate.

    This does *not* substitute current V2 bytes for the accepted V2 result:
    ``identity.completed_precision_v2_binding`` separately and exactly binds
    the historical GPU1 completion.  The V2 typed view is current GPU0 launch
    infrastructure only, and its closure is authenticated by this successor's
    explicit closure before it can reach a physical boundary.
    """
    runtime = identity.runtime_identity()
    try:
        failed_v1 = precision_v2_score.validate_v1_failed_predecessor(Path(root))
        result = precision_v2_plan.ScoreIdentity(
            closure=runtime.closure,
            v8_binding=runtime.v8_binding,
            selected_device_profile=plan.GPU0_PROFILE,
            successor_closure=precision_v2_plan.implementation_closure(Path(root)).payload(),
            v1_failed_predecessor_binding=failed_v1.payload(),
        )
    except (precision_v2_score.PrecisionMatchedScoreV2Error, precision_v2_plan.PrecisionMatchedScoreV2PlanError) as error:
        raise PhysicalSpeedSmokeError("speed-smoke V2 environment identity reconstruction failed") from error
    payload = result.payload()
    if payload.get("selected_device_profile") != plan.GPU0_PROFILE:
        raise PhysicalSpeedSmokeError("speed-smoke V2 environment bridge selected profile drift")
    return result


class _V2EnvironmentSpeedDelegate(precision_v2_physical.PhysicalPrecisionMatchedScoreV2Backend):
    """Keep V2's environment wrapper; change only its runtime factory bridge."""

    def __init__(self, *, root: Path, identity: score.SpeedSmokeIdentity) -> None:
        self._speed_identity = identity
        self._runtime_identity = identity.runtime_identity()
        compatibility = _build_v2_environment_identity(Path(root), identity)
        super().__init__(root=Path(root), identity=compatibility)
        # Base V2 constructed a standard Precision evaluator.  Replacing the
        # one factory-bearing delegate is the authorized composition seam;
        # all V2 source-root checks remain inherited from this class.
        self._delegate = precision_physical.PhysicalPrecisionMatchedScoreBackend(
            root=Path(root), selected_device_profile=plan.GPU0_PROFILE,
            runtime_factory=lambda item_root, item_profile: _ComparativeSpeedPrecisionV2Runtime(
                root=item_root, selected_device_profile=item_profile,
            ),
        )

    def _speed_gate(self, *, root: Path, identity: score.SpeedSmokeIdentity) -> None:
        _require(identity is self._speed_identity, "speed-smoke physical identity object drift")
        _require(Path(root).absolute() == self._root, "speed-smoke physical root drift")
        # This is the exact accepted V2 current-process environment gate.  It
        # checks CVD/PCI and canonical SUBC/SUBM strings against os.environ,
        # then descriptor-validates its V1 failed-history bridge without any
        # NWB/checkpoint/CUDA action.
        self._gate(root=self._root, identity=self._identity)
        observed = score.validate_completed_precision_v2(self._root)
        if observed.payload() != identity.payload()["completed_precision_v2_binding"]:
            raise PhysicalSpeedSmokeError("speed-smoke held completed V2 binding drift")
        plan.validate_gpu0_profile(self._runtime_identity.payload().get("selected_device_profile"))

    def preflight(self, *, root: Path, identity: score.SpeedSmokeIdentity) -> Mapping[str, object]:
        self._speed_gate(root=Path(root), identity=identity)
        return self._delegate.preflight(root=Path(root), identity=self._runtime_identity)

    def prepare(self, *, root: Path, identity: score.SpeedSmokeIdentity) -> Any:
        self._speed_gate(root=Path(root), identity=identity)
        runtime = self._delegate.prepare(root=Path(root), identity=self._runtime_identity)
        _require(isinstance(runtime, speed_stage0_physical.SpeedPrecisionV2ReviewedCDMScoreRuntime),
                 "speed-smoke runtime-factory type drift")
        return runtime

    def materialize_inputs(
        self, runtime: Any, *, identity: score.SpeedSmokeIdentity, evaluation_authority: Any,
    ) -> Any:
        self._speed_gate(root=self._root, identity=identity)
        return self._delegate.materialize_inputs(
            runtime, identity=self._runtime_identity, evaluation_authority=evaluation_authority,
        )

    def revalidate(self, runtime: Any, *, root: Path, identity: score.SpeedSmokeIdentity) -> None:
        self._speed_gate(root=Path(root), identity=identity)
        self._delegate.revalidate(runtime, root=Path(root), identity=self._runtime_identity)
        self._speed_gate(root=Path(root), identity=identity)

    def failure_progress(self, runtime: Any | None) -> Mapping[str, object]:
        return self._delegate.failure_progress(runtime)

    def close(self, runtime: Any | None) -> None:
        self._delegate.close(runtime)


class PhysicalPrecisionSpeedSmokeBackend:
    """Lifecycle-facing one-row eager-versus-O1/O2 adapter."""

    def __init__(self, *, root: Path, identity: score.SpeedSmokeIdentity) -> None:
        self._root = Path(root).absolute()
        self._identity = identity
        self._delegate = _V2EnvironmentSpeedDelegate(root=self._root, identity=identity)
        self._runtime: Any | None = None

    def preflight(self, *, root: Path, identity: score.SpeedSmokeIdentity) -> Mapping[str, object]:
        return self._delegate.preflight(root=Path(root), identity=identity)

    def prepare(self, *, root: Path, identity: score.SpeedSmokeIdentity) -> Any:
        if self._runtime is not None:
            raise PhysicalSpeedSmokeError("speed-smoke runtime prepared more than once")
        runtime = self._delegate.prepare(root=Path(root), identity=identity)
        self._runtime = runtime
        return runtime

    def materialize_inputs(self, runtime: Any, *, identity: score.SpeedSmokeIdentity, evaluation_authority: Any) -> Any:
        if runtime is not self._runtime:
            raise PhysicalSpeedSmokeError("speed-smoke materialize runtime identity drift")
        return self._delegate.materialize_inputs(runtime, identity=identity, evaluation_authority=evaluation_authority)

    def score_budget(
        self, runtime: Any, *, budget: int, input_authority_sha256: str, identity: score.SpeedSmokeIdentity,
    ) -> tuple[Mapping[str, object], ...]:
        if runtime is not self._runtime:
            raise PhysicalSpeedSmokeError("speed-smoke budget runtime identity drift")
        expected = [item for item in plan.SMOKE_ROWS if item.budget == budget]
        if len(expected) != 1:
            raise PhysicalSpeedSmokeError("speed-smoke budget is outside exact bounded matrix")
        witness = expected[0]
        state = runtime._require_state()
        session = state.sessions.get((witness.surface, witness.session))
        if session is None:
            raise PhysicalSpeedSmokeError("speed-smoke selected materialized session is absent")
        _require(isinstance(runtime, _ComparativeSpeedPrecisionV2Runtime),
                 "speed-smoke comparative runtime-factory type drift")
        try:
            baseline_row, optimized_row, comparison, speed_evidence = runtime.score_eager_baseline_and_optimized(
                session=session, budget=budget,
            )
            resources = runtime._resources()
        except Exception as error:
            raise PhysicalSpeedSmokeError("speed-smoke eager/optimized row reconstruction failed") from error
        return (score.build_speed_cell(
            identity=identity, input_authority_sha256=input_authority_sha256,
            baseline_session_row=baseline_row, optimized_session_row=optimized_row,
            comparison=comparison, speed_evidence=speed_evidence, resources=resources, witness=witness,
        ),)

    def revalidate(self, runtime: Any, *, root: Path, identity: score.SpeedSmokeIdentity) -> None:
        if runtime is not self._runtime:
            raise PhysicalSpeedSmokeError("speed-smoke final runtime identity drift")
        self._delegate.revalidate(runtime, root=Path(root), identity=identity)

    def failure_progress(self, runtime: Any | None) -> Mapping[str, object]:
        return self._delegate.failure_progress(runtime)

    def close(self, runtime: Any | None) -> None:
        try:
            self._delegate.close(runtime)
        finally:
            self._runtime = None


def build_reviewed_physical_backend(
    *, root: Path, identity: score.SpeedSmokeIdentity,
) -> PhysicalPrecisionSpeedSmokeBackend:
    """Construct composition objects only; no data/checkpoint/CUDA work occurs."""
    if not isinstance(identity, score.SpeedSmokeIdentity):
        raise PhysicalSpeedSmokeError("speed-smoke physical identity type drift")
    plan.validate_gpu0_profile(identity.payload().get("selected_device_profile"))
    return PhysicalPrecisionSpeedSmokeBackend(root=Path(root), identity=identity)


def execute_reviewed_physical_speed_smoke(
    root: Path, *, identity: score.SpeedSmokeIdentity, capability: object,
    environ: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Future root-only execution path; public CLI never calls this.

    The lexical GPU0/source-root gate is intentionally before the durable
    authority reload, backend construction, reserve, attempt, checkpoint, or
    evaluation input action.  The V2 wrapper then rechecks the actual process
    at every physical boundary.
    """
    score._validate_mutating_launch_environment(environ)
    score.validate_completed_precision_v2(Path(root))
    backend = build_reviewed_physical_backend(root=Path(root), identity=identity)
    preflight, authorization, pre_sha, auth_sha = score.load_durable_authority(Path(root), identity=identity)
    checked = backend.preflight(root=Path(root), identity=identity)
    _require(isinstance(checked, Mapping) and all(checked.get(key) is False for key in (
        "target_paths_resolved", "target_opened", "checkpoint_opened", "cuda_initialized",
    )), "speed-smoke backend preflight is not target/model/CUDA-free")
    artifact = score.reserve_score_artifact(Path(root), identity=identity, capability=capability, environ=environ)
    return score.run_authorized_speed_smoke_lifecycle(
        Path(root), identity=identity, capability=capability, backend=backend, artifact=artifact,
        official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
        preflight=preflight, authorization=authorization, environ=environ,
    )
