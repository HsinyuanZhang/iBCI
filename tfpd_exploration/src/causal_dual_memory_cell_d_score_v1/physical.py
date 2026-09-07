"""Deferred, descriptor-safe physical seam for CDM-D matched scoring.

This module deliberately has no Torch, NumPy, NWB, CUDA, cache, or result
root import at module import time.  It is reachable only through the opaque
capability/lifecycle in :mod:`score`, after ``attempt.json`` is durable.  The
implementation composes (rather than copies) the reviewed held-FD evaluator
substrate and the accepted CDM-D variable-prefix Cell-D forward seam.

The two systems share one sealed Cell-D SWA and each materialized input.  The
only system difference is the causal dual-memory state used for the CDM-D
side tensor and B3S activity prefix.  Target behavior is retained exclusively
for the last-bin metric; it never crosses the typed pseudo-update boundary.
"""
from __future__ import annotations

import hashlib
import importlib
import math
import os
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import plan, score


class PhysicalCDMDScoreError(score.ScoreError):
    """Fail closed for a deferred model, parser, chronology, or device drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhysicalCDMDScoreError(message)


def _sha(value: object, label: str) -> str:
    try:
        return plan.require_sha(value, label)
    except plan.PlanError as error:
        raise PhysicalCDMDScoreError(str(error)) from error


def _array_sha(np: Any, value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(
        plan.canonical_json_bytes({
            "dtype": str(array.dtype),
            "shape": [int(item) for item in array.shape],
            "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        })
    ).hexdigest()


def _object_sha(value: object) -> str:
    return hashlib.sha256(plan.canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class RawT4AxisProof:
    """The sole M30 raw-T4/SUA unit-order and valid-mask authority."""

    budget: int
    raw_t4_sha256: str
    channel_order_sha256: str
    valid_mask_sha256: str
    source_unit_count: int
    modulation_eps: float

    def payload(self) -> dict[str, object]:
        if (
            self.budget != 30 or type(self.source_unit_count) is not int
            or self.source_unit_count < 4 or not math.isfinite(float(self.modulation_eps))
            or float(self.modulation_eps) < 0.0
        ):
            raise PhysicalCDMDScoreError("raw M30 T4 axis proof budget/unit count drift")
        return {
            "schema": "causal_dual_memory_cell_d_score_raw_t4_axis_v1",
            "budget": self.budget,
            "raw_t4_sha256": _sha(self.raw_t4_sha256, "raw T4 SHA"),
            "channel_order_sha256": _sha(self.channel_order_sha256, "raw T4 channel-order SHA"),
            "valid_mask_sha256": _sha(self.valid_mask_sha256, "raw T4 valid-mask SHA"),
            "source_unit_count": self.source_unit_count,
            "feature_group": "t4",
            "signal_view": "sua",
            "channel_ids_are_exact_int64_arange": True,
            "validity_rule": "raw_t4_modulation_m_gt_modulation_eps",
            "modulation_eps": float(self.modulation_eps),
            "raw_before_normalization": True,
        }


@dataclass(frozen=True)
class EvaluationTrial:
    """Typed target-local dual-view capability for one rewarded trial.

    The values are constructed only after a held asset is copied to a private
    no-cache parser snapshot.  No behavior target appears in this object.
    """

    trial_id: str
    chronology_position: int
    b3s_activity: Any = field(repr=False, compare=False)
    native_counts: Any = field(repr=False, compare=False)
    velocity_validity: Any = field(repr=False, compare=False)
    neural_windows: Any = field(repr=False, compare=False)
    endpoint_bins: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class CompletedTrialTransition:
    """Route-neutral facts from one completed-query state transition.

    V1 maps its historical coupled commit onto this shape.  A successor may
    override only the commit hook to report independent activity/carrier
    outcomes, while the parser, forward, causal query ordering, and metric
    loop remain shared.
    """

    trial_id: str
    activity_transition_committed: bool
    activity_fifo_changed: bool
    carrier_transition_committed: bool
    activity_rejection_reason: str | None
    carrier_rejection_reason: str | None
    state_before_sha256: str
    state_after_sha256: str
    activity_before_sha256: str
    activity_after_sha256: str
    carrier_before_sha256: str
    carrier_after_sha256: str

    def payload(self) -> dict[str, object]:
        if not isinstance(self.trial_id, str) or not self.trial_id:
            raise PhysicalCDMDScoreError("completed-trial transition trial-ID drift")
        for value in (
            self.activity_transition_committed, self.activity_fifo_changed,
            self.carrier_transition_committed,
        ):
            if type(value) is not bool:
                raise PhysicalCDMDScoreError("completed-trial transition boolean drift")
        for label, value in (
            ("state before", self.state_before_sha256), ("state after", self.state_after_sha256),
            ("activity before", self.activity_before_sha256), ("activity after", self.activity_after_sha256),
            ("carrier before", self.carrier_before_sha256), ("carrier after", self.carrier_after_sha256),
        ):
            _sha(value, f"completed-trial transition {label} SHA")
        for label, value in (("activity", self.activity_rejection_reason), ("carrier", self.carrier_rejection_reason)):
            if value is not None and (not isinstance(value, str) or not value):
                raise PhysicalCDMDScoreError(f"completed-trial transition {label} rejection drift")
        return {
            "trial_id": self.trial_id,
            "activity_transition_committed": self.activity_transition_committed,
            "activity_fifo_changed": self.activity_fifo_changed,
            "carrier_transition_committed": self.carrier_transition_committed,
            "activity_rejection_reason_or_null": self.activity_rejection_reason,
            "carrier_rejection_reason_or_null": self.carrier_rejection_reason,
            "state_before_sha256": self.state_before_sha256,
            "state_after_sha256": self.state_after_sha256,
            "activity_before_sha256": self.activity_before_sha256,
            "activity_after_sha256": self.activity_after_sha256,
            "carrier_before_sha256": self.carrier_before_sha256,
            "carrier_after_sha256": self.carrier_after_sha256,
        }


@dataclass
class PreparedEvaluationSession:
    """One held, parsed session shared by every system/budget cell."""

    surface: str
    session: str
    held: Any
    neural: Any
    behavior: Any
    channel_ids: Any
    ordered_trial_ids: tuple[str, ...]
    trials_by_id: Mapping[str, EvaluationTrial]
    support_trial_ids: Mapping[int, tuple[str, ...]]
    query_trial_ids: Mapping[int, tuple[str, ...]]
    support_rates: Mapping[str, Any]
    support_direction_indices: Mapping[str, int]
    raw_m30_valid_mask: Any
    raw_m30_axis_proof: RawT4AxisProof
    theta_recovery: Mapping[str, object]
    theta_recovery_sha256: str
    input_record: score.InputRecord
    input_record_payload: Mapping[str, object]
    input_record_sha256: str


@dataclass
class _RuntimeState:
    modules: Mapping[str, Any]
    substrate: Any
    source_execute: Any
    source_physical: Any
    # The executor module and the physical-helper module are historically the
    # same V1 module.  A typed successor wrapper may retain a distinct
    # executor while explicitly supplying an authenticated V1 helper module.
    # Keeping this value in the runtime state makes the distinction durable at
    # every inherited helper call rather than relying on wrapper re-exports.
    source_physical_helpers: Any
    executor: Any
    executor_state: Any
    model: Any
    device_profile: Mapping[str, object]
    device: Any
    sealed_material: Any
    sealed_load_proof: Mapping[str, object]
    sessions: dict[tuple[str, str], PreparedEvaluationSession] = field(default_factory=dict)
    held_roots: list[Any] = field(default_factory=list)
    held_assets: list[Any] = field(default_factory=list)
    forward_chunks: int = 0
    full_system_forwards: int = 0
    completed_trials: int = 0
    measurement_started: float | None = None
    closed: bool = False


class RuntimeProtocol(Protocol):
    def prepare(self, *, identity: plan.ScoreIdentity) -> None: ...
    def materialize_inputs(self, *, identity: plan.ScoreIdentity,
                           authority: score.FixedEvaluationAuthority) -> score.InputAuthority: ...
    def score_budget(self, *, budget: int, input_authority_sha256: str,
                     identity: plan.ScoreIdentity) -> Sequence[score.CellEvidence]: ...
    def revalidate(self, *, identity: plan.ScoreIdentity) -> None: ...
    def failure_progress(self) -> Mapping[str, object]: ...
    def close(self) -> None: ...


def _prepend_runtime_roots(root: Path) -> None:
    """Install only explicit closure roots after a durable attempt exists."""
    base = Path(root).absolute()
    for item in (
        base / "tfpd_exploration",
        base / "sua_exploration",
        base / "streaming_calibration_exp",
    ):
        rendered = str(item)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)


def _runtime_modules(root: Path) -> Mapping[str, Any]:
    """Load all Torch/NWB-capable dependencies only in physical ``prepare``."""
    _prepend_runtime_roots(root)
    base = Path(root).absolute()
    try:
        torch = importlib.import_module("torch")
        np = importlib.import_module("numpy")
        return {
            "torch": torch,
            "np": np,
            "core": importlib.import_module("src.causal_dual_memory_cell_d_v1.core"),
            "source_adapter_v2": importlib.import_module("src.posterior_carrier_v1.source_adapter_v2"),
            "multisession": importlib.import_module("mc_maze.multisession_datamodule"),
            "unit_side": importlib.import_module("mc_maze.unit_side_features"),
            "d_optimal": importlib.import_module("mc_maze.d_optimal_calibration_design"),
            "pop_robust": score._load_exact_module(
                "_cdmd_score_pop_robust", base / "tfpd_exploration/src/tfpd_lane/pop_robust.py",
            ),
            "arm_common": score._load_exact_module(
                "_cdmd_score_arm_common", base / "tfpd_exploration/src/tfpd_lane/arm_common.py",
            ),
            "matched_metric": score._load_exact_module(
                "_cdmd_score_matched_metric", base / "tfpd_exploration/src/tfpd_lane/matched_scorer.py",
            ),
        }
    except Exception as error:
        raise PhysicalCDMDScoreError("CDM-D score runtime dependency import failed") from error


class ReviewedCDMScoreRuntime:
    """Small physical composition over reviewed parser and Cell-D forward seams."""

    def __init__(self, *, root: Path, selected_device_profile: Mapping[str, object]) -> None:
        self.root = Path(root).absolute()
        self.profile = plan.validate_compatible_device_profile(selected_device_profile)
        self.state: _RuntimeState | None = None
        self._within_opened = False
        self._external_opened = False
        self._checkpoint_opened = False
        self._cuda_initialized = False

    def _require_state(self) -> _RuntimeState:
        state = self.state
        _require(state is not None and not state.closed, "CDM-D score physical runtime is unavailable")
        return state

    def _validate_environment(self) -> None:
        _require(os.environ.get("CUDA_VISIBLE_DEVICES") == self.profile["cuda_visible_devices"],
                 "CDM-D score CUDA_VISIBLE_DEVICES drift")
        _require(os.environ.get("CUDA_DEVICE_ORDER") == self.profile["cuda_device_order"],
                 "CDM-D score CUDA_DEVICE_ORDER drift")

    def _source_execution_runtime_modules(self) -> tuple[Any, Any]:
        """Return the historical closure-bound source-execution modules.

        The V5 successor overrides this narrow loader seam with its accepted
        V5 executor.  No caller can select it through environment or a mapping.
        """
        return (
            importlib.import_module("src.causal_dual_memory_cell_d_v1.source_execute"),
            importlib.import_module("src.causal_dual_memory_cell_d_v1.source_execute_physical"),
        )

    def _build_source_executor(self, source_physical: Any) -> Any:
        return source_physical.ConcreteCellDFourGroupExecutor(root=self.root)

    def _source_physical_helper_module(self, source_physical: Any) -> Any:
        """Return V1's historical physical helper module.

        The default deliberately preserves the original one-module topology:
        the selected executor module owns the variable-prefix digest and both
        static Cell-D helper methods.  Successor routes may override this
        typed hook only after authenticating a closure-bound helper dependency;
        this is not a permissive fallback or wrapper attribute probe.
        """
        return source_physical

    def _build_runtime_flags(self, source_execute: Any) -> Any:
        """Construct the historical V1 source-execution runtime flags.

        The default deliberately retains the historical direct-module law:
        V1's selected source-execution module owns ``RuntimeFlags`` at its
        top level.  A typed successor wrapper may override this one factory
        only after it has closure-authenticated the wrapper's actual V1
        dependency.  This is not a permissive ``hasattr`` fallback.
        """
        return source_execute.RuntimeFlags(stage="score_prepare")

    @staticmethod
    def _make_dual_memory(*, core: Any, activity: Any, carrier: Any) -> Any:
        """Historical V1 coupled-memory constructor; successors override only this hook."""
        return core.CausalDualMemory(activity=activity, carrier=carrier)

    @staticmethod
    def _reason_value(value: object | None) -> str | None:
        if value is None:
            return None
        candidate = getattr(value, "value", value)
        return candidate if isinstance(candidate, str) and candidate else "unknown"

    def _commit_completed_transition(self, *, memory: Any, pending: Any, trial_id: str) -> CompletedTrialTransition:
        """Map V1's coupled commit onto the generic transition evidence shape."""
        before_state = memory.state.digest
        before_activity = memory.state.activity.digest
        before_carrier = memory.state.carrier.digest
        outcome = memory.commit(pending)
        committed = bool(outcome.committed)
        after_state = memory.state.digest
        after_activity = memory.state.activity.digest
        after_carrier = memory.state.carrier.digest
        reason = None if committed else self._reason_value(getattr(outcome, "reason", None))
        return CompletedTrialTransition(
            trial_id=trial_id,
            activity_transition_committed=committed,
            activity_fifo_changed=before_activity != after_activity,
            carrier_transition_committed=committed,
            activity_rejection_reason=reason,
            carrier_rejection_reason=reason,
            state_before_sha256=before_state, state_after_sha256=after_state,
            activity_before_sha256=before_activity, activity_after_sha256=after_activity,
            carrier_before_sha256=before_carrier, carrier_after_sha256=after_carrier,
        )

    def prepare(self, *, identity: plan.ScoreIdentity) -> None:
        """Strict-load the sealed SWA only after the lifecycle wrote attempt."""
        if identity.payload()["selected_device_profile"] != self.profile:
            raise PhysicalCDMDScoreError("score identity/physical selected-device profile drift")
        self._validate_environment()
        modules = _runtime_modules(self.root)
        self._cuda_initialized = bool(modules["torch"].cuda.is_initialized())
        substrate = score._load_exact_module(
            "_cdmd_score_held_fd_substrate",
            self.root / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
        )
        source_execute, source_physical = self._source_execution_runtime_modules()
        source_physical_helpers = self._source_physical_helper_module(source_physical)
        self._checkpoint_opened = True
        try:
            sealed = substrate.load_sealed_cell_d_material(self.root)
        except Exception as error:
            raise PhysicalCDMDScoreError("sealed Cell-D terminal/SWA/baseline descriptor validation failed") from error
        if (
            sealed.terminal_sha256 != plan.SEALED_CELL_D_TERMINAL_SHA256
            or sealed.swa_sha256 != plan.SEALED_CELL_D_SWA_SHA256
            or sealed.baseline_sha256 != plan.SEALED_CELL_D_BASELINE_SHA256
        ):
            raise PhysicalCDMDScoreError("sealed Cell-D immutable material binding drift")
        flags = self._build_runtime_flags(source_execute)
        executor = self._build_source_executor(source_physical)
        try:
            load = executor.load_strict_sealed_swa(
                root=self.root, swa_bytes=sealed.swa_body, selected_device=self.profile, flags=flags,
            )
            executor_state = executor._require_runtime()  # reviewed route-local private composition seam
        except Exception:
            self._cuda_initialized = bool(modules["torch"].cuda.is_initialized())
            executor.close()
            raise
        runtime_environment = load.get("runtime_environment") if isinstance(load, Mapping) else None
        expected_environment = {
            **self.profile,
            "visible_devices": 1,
            "attested": True,
            "torch_cuda_matmul_allow_tf32": False,
            "torch_cudnn_allow_tf32": False,
        }
        if runtime_environment != expected_environment:
            executor.close()
            raise PhysicalCDMDScoreError("CDM-D score sealed model runtime attestation drift")
        self.state = _RuntimeState(
            modules=modules, substrate=substrate, source_execute=source_execute,
            source_physical=source_physical, source_physical_helpers=source_physical_helpers,
            executor=executor, executor_state=executor_state,
            model=executor_state.model, device_profile=dict(self.profile), device=executor_state.device,
            sealed_material=sealed, sealed_load_proof=dict(load.get("sealed_swa_load_proof", {})),
        )
        self._cuda_initialized = bool(modules["torch"].cuda.is_initialized())

    def _held_root_for_surface(self, surface: str) -> Any:
        state = self._require_state()
        variable = "SUBC_DATA_ROOT" if surface == plan.WITHIN else "SUBM_DATA_ROOT"
        expected = Path(os.environ.get(variable, "")).absolute()
        for existing in state.held_roots:
            if existing.root == expected:
                return existing
        try:
            result = state.substrate.HeldDataRoot.from_environment(variable)
        except Exception as error:
            raise PhysicalCDMDScoreError(f"CDM-D score cannot hold {surface} data root") from error
        state.held_roots.append(result)
        return result

    @staticmethod
    def _trial_id(session: str, item: Mapping[str, object]) -> str:
        index = item.get("trial_index")
        if type(index) is not int:
            raise PhysicalCDMDScoreError("rewarded chronology original trial-index drift")
        return f"{session}:trial:{index}"

    @staticmethod
    def _trial_interval(item: Mapping[str, object]) -> tuple[int, int]:
        start, stop = item.get("start"), item.get("stop")
        if isinstance(start, bool) or isinstance(stop, bool):
            raise PhysicalCDMDScoreError("rewarded chronology interval type drift")
        try:
            start_i, stop_i = int(start), int(stop)
        except (TypeError, ValueError) as error:
            raise PhysicalCDMDScoreError("rewarded chronology interval conversion drift") from error
        if start_i < 0 or stop_i <= start_i or stop_i - start_i < plan.WINDOW_BINS:
            raise PhysicalCDMDScoreError("rewarded chronology interval/window drift")
        return start_i, stop_i

    @staticmethod
    def _exact_duration_rates(np: Any, counts_by_unit_trial: Any, exposure: Any) -> Any:
        counts = np.ascontiguousarray(np.asarray(counts_by_unit_trial, dtype=np.float64))
        seconds = np.ascontiguousarray(np.asarray(exposure, dtype=np.float64))
        if (
            counts.ndim != 2 or seconds.ndim != 1 or counts.shape[1] != seconds.size
            or not bool(np.isfinite(counts).all()) or not bool((counts >= 0).all())
            or not bool(np.isfinite(seconds).all()) or not bool((seconds > 0).all())
        ):
            raise PhysicalCDMDScoreError("exact-duration support count/exposure authority drift")
        result = np.ascontiguousarray(counts.T / seconds[:, None], dtype=np.float64)
        if not bool(np.isfinite(result).all()):
            raise PhysicalCDMDScoreError("exact-duration support rate construction drift")
        result.setflags(write=False)
        return result

    def _raw_m30_t4_axis(self, *, snapshot: Any, session: str, record: Any) -> tuple[Any, RawT4AxisProof]:
        state = self._require_state()
        np = state.modules["np"]
        unit_side = state.modules["unit_side"]
        neural = np.asarray(record.neural)
        channels = np.ascontiguousarray(np.asarray(record.channel_ids), dtype=np.int64)
        units = int(neural.shape[1])
        if (
            int(getattr(record, "source_unit_count", -1)) != units
            or channels.shape != (units,)
            or not np.array_equal(channels, np.arange(units, dtype=np.int64))
        ):
            raise PhysicalCDMDScoreError("raw T4 exact SUA channel-order precondition drift")
        channel_sha = state.modules["core"].channel_order_digest(channels)
        try:
            values, metadata = unit_side.compute_unit_side_features_uncached(
                snapshot.path, feature_group="t4", pool_size=30, bin_size_ms=20,
                window_size=50, trial_result_filter="R", signal_view="sua",
            )
        except Exception as error:
            raise PhysicalCDMDScoreError("no-cache raw M30 T4 parser failed") from error
        value = np.ascontiguousarray(np.asarray(values), dtype=np.float32)
        if (
            value.shape != (units, 4) or not bool(np.isfinite(value).all())
            or getattr(metadata, "feature_group", None) != "t4"
            or getattr(metadata, "pool_size", None) != 30
            or getattr(metadata, "signal_view", "sua") != "sua"
        ):
            raise PhysicalCDMDScoreError("raw M30 T4/SUA axis proof drift")
        modulation_eps = getattr(unit_side, "MODULATION_EPS", None)
        if isinstance(modulation_eps, bool) or not isinstance(modulation_eps, (int, float)):
            raise PhysicalCDMDScoreError("raw M30 T4 modulation epsilon authority is unavailable")
        valid = np.ascontiguousarray(value[:, 2] > float(modulation_eps), dtype=np.bool_)
        if int(valid.sum()) < plan.GROUP_COUNT:
            raise PhysicalCDMDScoreError("raw M30 T4 has insufficient valid units")
        return valid, RawT4AxisProof(
            budget=30, raw_t4_sha256=_array_sha(np, value), channel_order_sha256=channel_sha,
            valid_mask_sha256=_array_sha(np, valid), source_unit_count=units,
            modulation_eps=float(modulation_eps),
        )

    @staticmethod
    def _governing_target_mask_authority(
        np: Any, *, behavior: Any, query_trial_ids: Sequence[str], trials_by_id: Mapping[str, EvaluationTrial],
    ) -> tuple[Any, Any, int]:
        """Bind the exact all-valid governing query target and bitmap.

        The frozen matched evaluators construct ``record.valid_starts`` so
        every governed final bin is valid.  Accepting and filtering a later
        invalid row here would silently change both the query surface and the
        historical Cell-D parity target.  We still bind the all-one bitmap so
        both systems prove their identical query-level mask rather than only
        a target digest.
        """
        targets: list[Any] = []
        masks: list[Any] = []
        for trial_id in query_trial_ids:
            trial = trials_by_id[trial_id]
            target = np.ascontiguousarray(np.asarray(behavior[trial.endpoint_bins]), dtype=np.float32)
            if target.ndim != 2 or target.shape[1] != 2 or target.shape[0] != trial.endpoint_bins.size:
                raise PhysicalCDMDScoreError("governing last-bin target trial topology drift")
            valid = np.ascontiguousarray(np.all(target != -1.0, axis=1), dtype=np.uint8)
            targets.append(target)
            masks.append(valid)
        if not targets:
            raise PhysicalCDMDScoreError("governing target authority has no query trials")
        target_joined = np.ascontiguousarray(np.concatenate(targets, axis=0), dtype=np.float32)
        mask_joined = np.ascontiguousarray(np.concatenate(masks, axis=0), dtype=np.uint8)
        count = int(mask_joined.sum())
        if count != int(mask_joined.size):
            raise PhysicalCDMDScoreError("governing target authority contains invalid governed last-bin rows")
        return target_joined, mask_joined, count

    def _make_trial(
        self, *, session: str, trial_id: str, position: int, start: int, stop: int,
        neural: Any, channels: Any, window_starts: Any,
    ) -> EvaluationTrial:
        state = self._require_state()
        np, core = state.modules["np"], state.modules["core"]
        multi = state.modules["multisession"]
        units = int(neural.shape[1])
        activity = multi._build_calib_trials(
            neural, [{"start": start, "stop": stop}], 1, 100, units, -1.0, True,
        )[0]
        starts = np.ascontiguousarray(np.asarray(window_starts), dtype=np.int64)
        if (
            starts.ndim != 1 or starts.size <= 0 or int(starts.min()) < start
            or int(starts.max()) + plan.WINDOW_BINS > stop
        ):
            raise PhysicalCDMDScoreError("governing valid-start/trial interval topology drift")
        endpoints = np.ascontiguousarray(starts + (plan.WINDOW_BINS - 1), dtype=np.int64)
        windows = np.ascontiguousarray(
            np.stack([neural[index - (plan.WINDOW_BINS - 1):index + 1] for index in endpoints], axis=0),
            dtype=np.float32,
        )
        channel_sha = core.channel_order_digest(channels)
        b3s = core.B3SInterpolatedSpikeCountTrial(
            activity=np.ascontiguousarray(activity, dtype=np.float32), session_id=session, trial_id=trial_id,
            channel_order_sha256=channel_sha,
        )
        native = core.NativeRewardedTrialSpikeCounts(
            counts=np.ascontiguousarray(neural[start:stop]), session_id=session, trial_id=trial_id,
            channel_order_sha256=channel_sha, rewarded_interval_start_bin=start, rewarded_interval_stop_bin=stop,
        )
        # The pseudo-update validity is purposefully target-free.  Metric
        # padding/validity is applied separately below at governed bin 49.
        validity = core.VelocityValidityEvidence(
            valid_mask=np.ones(endpoints.shape, dtype=np.bool_), session_id=session, trial_id=trial_id,
            prediction_interval_start_bin=int(endpoints[0]), prediction_interval_stop_bin=int(endpoints[-1]) + 1,
        )
        return EvaluationTrial(
            trial_id=trial_id, chronology_position=position, b3s_activity=b3s,
            native_counts=native, velocity_validity=validity, neural_windows=windows, endpoint_bins=endpoints,
        )

    def _parse_session(self, *, asset: score.EvaluationAsset) -> PreparedEvaluationSession:
        state = self._require_state()
        np, torch = state.modules["np"], state.modules["torch"]
        multi = state.modules["multisession"]
        adapter_v2 = state.modules["source_adapter_v2"]
        d_optimal = state.modules["d_optimal"]
        core = state.modules["core"]
        data_root = self._held_root_for_surface(asset.surface)
        if asset.surface == plan.WITHIN:
            self._within_opened = True
        elif asset.surface == plan.EXTERNAL:
            self._external_opened = True
        else:  # defensive: EvaluationAsset would already reject this surface.
            raise PhysicalCDMDScoreError("evaluation asset surface is outside the fixed score contract")
        try:
            held = data_root.open_asset(
                relative=Path(asset.frozen_path).name, expected_bytes=asset.bytes,
                expected_sha256=asset.sha256, surface=asset.surface, session=asset.session,
            )
        except Exception as error:
            raise PhysicalCDMDScoreError("evaluation asset descriptor/size/SHA validation failed") from error
        state.held_assets.append(held)
        snapshot = held.private_snapshot()
        try:
            record = multi.load_dandi688_session(
                snapshot.path, bin_size_ms=20, window_size=50, calibration_n_trials=30,
                max_trial_length=100, pad_value=-1.0, interpolate_trials=True,
                behavior_mean=np.asarray(plan.SEALED_BEHAVIOR_MEAN, dtype=np.float32),
                behavior_std=np.asarray(plan.SEALED_BEHAVIOR_STD, dtype=np.float32),
                trial_result_filter="R", exclude_calibration_trials_from_windows=True,
                cache_dir=None, signal_view="sua",
            )
            trials = tuple(dict(item) for item in multi.list_datamodule_rewarded_trials(
                snapshot.path, bin_size_ms=20, window_size=50, trial_result_filter="R",
            ))
            if len(trials) <= 30:
                raise PhysicalCDMDScoreError("evaluation session lacks a post-support causal query chronology")
            neural = np.ascontiguousarray(np.asarray(record.neural), dtype=np.float32)
            behavior = np.ascontiguousarray(np.asarray(record.behavior), dtype=np.float32)
            channels = np.ascontiguousarray(np.asarray(record.channel_ids), dtype=np.int64)
            if (
                getattr(record, "name", None) != asset.session or getattr(record, "signal_view", None) != "sua"
                or neural.ndim != 2 or behavior.shape != (neural.shape[0], 2)
                or channels.shape != (neural.shape[1],)
                or not np.array_equal(channels, np.arange(neural.shape[1], dtype=np.int64))
                or not bool(np.isfinite(neural).all()) or not bool(np.isfinite(behavior).all())
            ):
                raise PhysicalCDMDScoreError("evaluation parser neural/behavior/SUA channel contract drift")
            raw_m30_valid_mask, raw_m30_proof = self._raw_m30_t4_axis(
                snapshot=snapshot, session=asset.session, record=record,
            )
            # This generic same-prefix helper reads only the selected first 30
            # rewarded rows.  A missing target_dir can use *that row's*
            # target_corners; no source fallback topology and no later trial
            # can affect target evaluation.
            counts, exposure, theta, prefix_ids, theta_recovery = adapter_v2._source_counts_exposure_theta_v2(
                path=snapshot.path, session=asset.session, expected_units=int(neural.shape[1]),
            )
            if tuple(prefix_ids) != tuple(self._trial_id(asset.session, item) for item in trials[:30]):
                raise PhysicalCDMDScoreError("target same-prefix theta trial-ID order drift")
            adapter_v2.validate_theta_recovery_evidence(
                theta_recovery, session=asset.session, prefix_row_ids=prefix_ids,
                theta_sha256=state.modules["source_adapter_v2"].core.tensor_digest(theta),
            )
            snapshot.reverify()
        finally:
            snapshot.close()
        held.reverify()
        exact_rates = self._exact_duration_rates(
            np, counts.detach().cpu().numpy(), exposure.detach().cpu().numpy(),
        )
        theta_values = np.ascontiguousarray(theta.detach().cpu().numpy(), dtype=np.float64)
        if theta_values.shape != (30,) or not bool(np.isfinite(theta_values).all()):
            raise PhysicalCDMDScoreError("target same-prefix theta numeric authority drift")
        try:
            m4_indices = tuple(sorted(int(item) for item in d_optimal.greedy_forward_d_optimal_indices(theta_values, 4).tolist()))
        except Exception as error:
            raise PhysicalCDMDScoreError("target M4 D-opt support derivation failed") from error
        if len(m4_indices) != 4 or len(set(m4_indices)) != 4:
            raise PhysicalCDMDScoreError("target M4 D-opt support cardinality drift")
        trials_by_id: dict[str, EvaluationTrial] = {}
        ordered: list[str] = []
        valid_starts = np.ascontiguousarray(np.asarray(record.valid_starts), dtype=np.int64)
        intervals = tuple(self._trial_interval(row) for row in trials)
        expected_valid_starts = np.ascontiguousarray(
            np.concatenate([
                np.arange(start, stop - plan.WINDOW_BINS + 1, dtype=np.int64)
                for start, stop in intervals[30:]
            ]),
            dtype=np.int64,
        )
        if valid_starts.ndim != 1 or not np.array_equal(valid_starts, expected_valid_starts):
            raise PhysicalCDMDScoreError(
                "governing parser valid-starts differ from exact post-first30 calibration query surface"
            )
        for position, row in enumerate(trials):
            trial_id = self._trial_id(asset.session, row)
            start, stop = intervals[position]
            if stop > neural.shape[0]:
                raise PhysicalCDMDScoreError("rewarded chronology extends beyond parsed neural bins")
            ordered.append(trial_id)
            if position < 30:
                # Support rows are never evaluated, but the typed native/B3S
                # capability still needs a well-formed local window tensor.
                starts = np.arange(start, stop - plan.WINDOW_BINS + 1, dtype=np.int64)
            else:
                starts = valid_starts[(valid_starts >= start) & (valid_starts + plan.WINDOW_BINS <= stop)]
            trials_by_id[trial_id] = self._make_trial(
                session=asset.session, trial_id=trial_id, position=position, start=start, stop=stop,
                neural=neural, channels=channels, window_starts=starts,
            )
        if len(set(ordered)) != len(ordered) or tuple(ordered[:30]) != tuple(prefix_ids):
            raise PhysicalCDMDScoreError("target chronological trial topology/prefix binding drift")
        calib = np.ascontiguousarray(np.asarray(record.calib_trials), dtype=np.float32)
        if calib.shape != (30, 100, neural.shape[1]) or any(
            not np.array_equal(calib[index], trials_by_id[ordered[index]].b3s_activity.activity)
            for index in range(30)
        ):
            raise PhysicalCDMDScoreError("sealed first30 calibration/B3S support reconstruction drift")
        support_indices = {4: m4_indices, 10: tuple(range(10)), 30: tuple(range(30))}
        support_trial_ids = {budget: tuple(ordered[index] for index in support_indices[budget]) for budget in plan.BUDGETS}
        # All three initial carriers are constructed from a selected subset of
        # the sealed first-30 support pool.  Evaluating a non-selected early
        # row would let a later first-30 support label affect an earlier query
        # prediction for M4/M10.  The causal deployment surface is therefore
        # the common chronological post-support pool only.
        query_trial_ids = {budget: tuple(ordered[30:]) for budget in plan.BUDGETS}
        if any(not query_trial_ids[budget] for budget in plan.BUDGETS):
            raise PhysicalCDMDScoreError("target budget query partition is empty")
        support_rates = {prefix_ids[index]: exact_rates[index] for index in range(30)}
        support_directions = {
            prefix_ids[index]: int(core.nearest_canonical_direction(float(theta_values[index]))[0])
            for index in range(30)
        }
        target_sha: dict[str, str] = {}
        mask_sha: dict[str, str] = {}
        counts_by_budget: dict[str, int] = {}
        for budget in plan.BUDGETS:
            target_joined, mask_joined, valid_count = self._governing_target_mask_authority(
                np, behavior=behavior, query_trial_ids=query_trial_ids[budget], trials_by_id=trials_by_id,
            )
            target_sha[str(budget)] = _array_sha(np, target_joined)
            mask_sha[str(budget)] = _array_sha(np, mask_joined)
            counts_by_budget[str(budget)] = valid_count
        theta_body = theta_recovery.get("body_sha256") if isinstance(theta_recovery, Mapping) else None
        theta_sha = _sha(theta_body, "target same-prefix theta recovery body SHA")
        input_record = score.InputRecord(
            surface=asset.surface, session=asset.session, asset_id=asset.asset_id,
            frozen_path=asset.frozen_path, asset_bytes=asset.bytes, asset_sha256=asset.sha256,
            chronological_trial_ids=tuple(ordered),
            support_trial_ids_by_budget={str(key): value for key, value in support_trial_ids.items()},
            query_trial_ids_by_budget={str(key): value for key, value in query_trial_ids.items()},
            neural_sha256=_array_sha(np, neural),
            calibration_sha256=_array_sha(np, np.asarray(record.calib_trials, dtype=np.float32)),
            target_last_bin_sha256_by_budget=target_sha,
            valid_last_bin_mask_sha256_by_budget=mask_sha,
            valid_last_bin_count_by_budget=counts_by_budget,
            raw_m30_t4_axis_proof=raw_m30_proof.payload(),
            theta_recovery_sha256=theta_sha,
        )
        record_payload = input_record.payload()
        return PreparedEvaluationSession(
            surface=asset.surface, session=asset.session, held=held, neural=neural, behavior=behavior,
            channel_ids=channels, ordered_trial_ids=tuple(ordered), trials_by_id=trials_by_id,
            support_trial_ids=support_trial_ids, query_trial_ids=query_trial_ids,
            support_rates=support_rates, support_direction_indices=support_directions,
            raw_m30_valid_mask=raw_m30_valid_mask, raw_m30_axis_proof=raw_m30_proof,
            theta_recovery=theta_recovery,
            theta_recovery_sha256=theta_sha, input_record=input_record, input_record_payload=record_payload,
            input_record_sha256=_object_sha(record_payload),
        )

    def materialize_inputs(self, *, identity: plan.ScoreIdentity,
                           authority: score.FixedEvaluationAuthority) -> score.InputAuthority:
        state = self._require_state()
        if state.sessions:
            raise PhysicalCDMDScoreError("evaluation input materialization may run exactly once")
        records: list[score.InputRecord] = []
        for asset in (*authority.within, *authority.external):
            prepared = self._parse_session(asset=asset)
            state.sessions[(asset.surface, asset.session)] = prepared
            records.append(prepared.input_record)
        fixed_sha = _object_sha(authority.payload())
        result = score.InputAuthority(tuple(records), fixed_sha)
        result.payload(identity=identity)
        return result

    def _initial_memory(self, *, session: PreparedEvaluationSession, budget: int) -> tuple[Any, Mapping[str, object]]:
        state = self._require_state()
        np, core = state.modules["np"], state.modules["core"]
        support_ids = session.support_trial_ids[budget]
        rates = np.ascontiguousarray(np.stack([session.support_rates[item] for item in support_ids]), dtype=np.float64)
        directions = np.ascontiguousarray(
            np.asarray([session.support_direction_indices[item] for item in support_ids], dtype=np.int64),
        )
        if rates.shape != (budget, session.channel_ids.size) or directions.shape != (budget,):
            raise PhysicalCDMDScoreError(f"M{budget} support exact-duration rate/label topology drift")
        initial = core.fit_carriers_from_trial_table(
            rates, directions, mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
            normalized_lambda=core.RIDGE_NORMALIZED_LAMBDA,
        )
        parity = core.assert_production_parity(
            initial, rates, directions, mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
        )
        raw_axis = session.raw_m30_axis_proof
        # The group topology is one M30 raw-T4 modulation *mask* authority
        # shared by every budget.  It can only exclude undefined rows; among
        # valid rows ``CarrierMemory`` derives each budget's four groups from
        # the support-only fixed-ridge carrier below.  No raw M4/M10 value is
        # computed or consumed, and no raw M30 value determines a group.
        valid = np.ascontiguousarray(np.asarray(session.raw_m30_valid_mask), dtype=np.bool_)
        if _array_sha(np, valid) != raw_axis.valid_mask_sha256:
            raise PhysicalCDMDScoreError("shared raw-M30 T4 valid-mask authority drift")
        if not bool(np.isfinite(initial[valid]).all()):
            raise PhysicalCDMDScoreError("budget valid initial fixed-ridge carrier is nonfinite")
        config = core.CDMDConfig(
            support_budget_m=budget, active_fit_mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
        )
        carrier = core.CarrierMemory.from_support_trials(
            initial_raw_t4=initial, channel_ids=session.channel_ids, support_trial_rates=rates,
            support_direction_indices=directions, config=config, valid_mask=valid,
        )
        activity = core.ActivityMemory.initialize(
            [session.trials_by_id[item].b3s_activity for item in support_ids],
            channel_ids=session.channel_ids, fifo_capacity=int(config.activity_fifo_capacity),
        )
        memory = self._make_dual_memory(core=core, activity=activity, carrier=carrier)
        evidence = {
            "budget": budget,
            "support_trial_ids": list(support_ids),
            "initial_support_rate_domain": "raw_spike_counts_exact_half_open_trial_interval_divided_by_exact_seconds",
            "online_update_rate_domain": "mean_native_20ms_binned_counts_over_full_rewarded_interval_divided_by_0.020",
            "initial_carrier_recipe": core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL.value,
            "initial_carrier_parity": parity,
            "initial_carrier_sha256": core.array_digest(initial),
            "group_assignment_sha256": carrier.groups.digest,
            "group_assignment_rule": "budget_specific_fixed_ridge_initial_carrier_among_m30_valid_rows",
            "group_valid_mask_sha256": core.array_digest(carrier.groups.valid_mask),
            "activity_fifo_capacity": int(config.activity_fifo_capacity),
            "initial_activity_sha256": memory.state.activity.digest,
            "initial_carrier_state_sha256": memory.state.carrier.digest,
            "raw_m30_t4_used_as_initializer": False,
            "raw_m30_t4_used_only_for_channel_and_valid_mask_authority": True,
            "raw_m30_t4_used_for_m4_or_m10": False,
            "raw_m30_t4_axis_proof": raw_axis.payload(),
        }
        return memory, evidence

    def _normalized_side(self, raw_t4: Any) -> Any:
        state = self._require_state()
        return self._normalized_active_t4(state, raw_t4)

    @staticmethod
    def _normalized_active_t4(state: _RuntimeState, raw_t4: Any) -> Any:
        return state.source_physical_helpers.ConcreteCellDFourGroupExecutor._normalized_active_t4(
            state.executor_state, raw_t4,
        )

    @staticmethod
    def _variable_prefix_forward(
        state: _RuntimeState, *, neural_windows: Any, activity_stack: Any, normalized_t4: Any,
        held_mask: Any, validity: Any, expected_prefix_length: int, expected_prefix_activity_sha256: str,
    ) -> tuple[Any, Mapping[str, object]]:
        return state.source_physical_helpers.ConcreteCellDFourGroupExecutor._torch_variable_prefix_forward(
            state.executor_state,
            neural_windows=neural_windows, activity_stack=activity_stack, normalized_t4=normalized_t4,
            held_mask=held_mask, validity=validity, expected_prefix_length=expected_prefix_length,
            expected_prefix_activity_sha256=expected_prefix_activity_sha256,
        )

    def _forward_full(
        self, *, neural_windows: Any, activity_stack: Any, normalized_t4: Any,
    ) -> tuple[Any, Mapping[str, object]]:
        """One eval/no-grad B128 full-system pass plus bitwise repeat proof."""
        state = self._require_state()
        torch, np, pop, arm = (
            state.modules["torch"], state.modules["np"], state.modules["pop_robust"], state.modules["arm_common"],
        )
        neural_cpu = np.ascontiguousarray(np.asarray(neural_windows), dtype=np.float32)
        stack_cpu = np.ascontiguousarray(np.asarray(activity_stack), dtype=np.float32)
        side_cpu = np.ascontiguousarray(np.asarray(normalized_t4), dtype=np.float32)
        if (
            neural_cpu.ndim != 3 or neural_cpu.shape[1] != plan.WINDOW_BINS
            or stack_cpu.ndim != 3 or not 1 <= stack_cpu.shape[0] <= 30 or stack_cpu.shape[1] != 100
            or side_cpu.shape != (neural_cpu.shape[2], 4)
            or stack_cpu.shape[2] != neural_cpu.shape[2]
        ):
            raise PhysicalCDMDScoreError("full Cell-D variable-prefix input shape drift")
        neural = torch.as_tensor(neural_cpu, dtype=torch.float32, device=state.device)
        calibration = torch.as_tensor(stack_cpu, dtype=torch.float32, device=state.device)
        calibration = calibration.unsqueeze(0).expand(neural.shape[0], -1, -1, -1)
        side = torch.as_tensor(side_cpu, dtype=torch.float32, device=state.device)
        side = side.unsqueeze(0).expand(neural.shape[0], -1, -1)
        before = arm.state_sha256(state.model)
        chunks: list[Any] = []
        output_digest = hashlib.sha256()
        actual_forwards = 0
        for start in range(0, int(neural.shape[0]), plan.EVAL_BATCH_SIZE):
            stop = min(start + plan.EVAL_BATCH_SIZE, int(neural.shape[0]))
            with pop.dynamic_dropout_recorder() as recorder:
                with torch.no_grad():
                    first, _ = state.model(neural[start:stop], calib_trials=calibration[start:stop], side_features=side[start:stop])
                    second, _ = state.model(neural[start:stop], calib_trials=calibration[start:stop], side_features=side[start:stop])
            if (
                not torch.equal(first, second) or not bool(torch.isfinite(first).all().item())
                or recorder.get("uniform_calls") != 0 or recorder.get("dropout_calls") != []
            ):
                raise PhysicalCDMDScoreError("full Cell-D eval/repeat/dropout/finite forward drift")
            detached = first.detach().cpu().contiguous()
            output_digest.update(detached.numpy().tobytes())
            chunks.append(detached)
            actual_forwards += 2
        after = arm.state_sha256(state.model)
        if before != after:
            raise PhysicalCDMDScoreError("sealed Cell-D model state mutated during full score forward")
        state.forward_chunks += actual_forwards
        state.full_system_forwards += actual_forwards
        return torch.cat(chunks, dim=0), {
            "model_state_before_sha256": before,
            "model_state_after_sha256": after,
            "prediction_sha256": output_digest.hexdigest(),
            "full_system_forward_count": actual_forwards,
            "dropout_calls": 0,
            "repeated_outputs_bitwise_equal": True,
            "prefix_length": int(stack_cpu.shape[0]),
        }

    def _group_predictions(self, *, trial: EvaluationTrial, memory: Any) -> tuple[tuple[Any, ...], list[Mapping[str, object]]]:
        state = self._require_state()
        inputs = memory.read_prediction_inputs()
        expected_prefix = int(inputs.activity_trials.shape[0])
        if expected_prefix != int(memory.state.carrier.config.support_budget_m) + int(memory.state.activity.query_count):
            raise PhysicalCDMDScoreError("causal activity prefix count/state drift before group pseudo forward")
        normalized = self._normalized_side(inputs.active_t4).detach().cpu().numpy()
        activity_digest = state.source_physical_helpers._variable_prefix_array_digest(inputs.activity_trials)
        predictions: list[Any] = []
        evidence: list[Mapping[str, object]] = []
        for group in range(plan.GROUP_COUNT):
            prediction, proof = self._variable_prefix_forward(
                state,
                neural_windows=trial.neural_windows,
                activity_stack=inputs.activity_trials,
                normalized_t4=normalized,
                held_mask=memory.state.carrier.groups.held_mask(group), validity=trial.velocity_validity,
                expected_prefix_length=expected_prefix, expected_prefix_activity_sha256=activity_digest,
            )
            predictions.append(prediction)
            evidence.append(dict(proof))
        if len(predictions) != plan.GROUP_COUNT:
            raise PhysicalCDMDScoreError("CDM-D must finalize exactly four held-group trajectories")
        state.forward_chunks += 2 * sum(int(item["forward_chunk_count"]) for item in evidence)
        return tuple(predictions), evidence

    def _make_session_score(
        self,
        *,
        session: PreparedEvaluationSession,
        budget: int,
        system: str,
        n_windows: int,
        r2: float,
        prediction_sha256: str,
        model_state_before_sha256: str,
        model_state_after_sha256: str,
        initial: Mapping[str, object],
        full_system_forward_count: int,
        group_forward_count: int,
        transitions: Sequence[CompletedTrialTransition],
        target_last_bin_sha256: str,
        valid_mask_sha256: str,
        sealed_model_load_proof_sha256: str,
    ) -> Any:
        """Encode the historical V1 receipt; V5 overrides only this codec hook."""
        accepted = sum(
            plan.GROUP_COUNT for item in transitions if item.carrier_transition_committed
        )
        rejected: dict[str, int] = {}
        for item in transitions:
            if item.carrier_transition_committed:
                continue
            reason = item.carrier_rejection_reason or "unknown"
            rejected[reason] = rejected.get(reason, 0) + plan.GROUP_COUNT
        return score.SessionScore(
            session=session.session, n_windows=n_windows, r2=r2, prediction_sha256=prediction_sha256,
            input_record_sha256=session.input_record_sha256,
            model_state_before_sha256=model_state_before_sha256,
            model_state_after_sha256=model_state_after_sha256,
            initial_carrier_sha256=str(initial["initial_carrier_sha256"]),
            group_assignment_sha256=str(initial["group_assignment_sha256"]),
            group_valid_mask_sha256=str(initial["group_valid_mask_sha256"]),
            initial_activity_sha256=str(initial["initial_activity_sha256"]),
            support_trial_ids_sha256=_object_sha(list(session.support_trial_ids[budget])),
            raw_m30_t4_axis_proof_sha256=_object_sha(session.raw_m30_axis_proof.payload()),
            sealed_normalizer_sha256=plan.SEALED_OLS_NORMALIZER_SHA256,
            sealed_model_load_proof_sha256=sealed_model_load_proof_sha256,
            target_last_bin_sha256=target_last_bin_sha256, valid_mask_sha256=valid_mask_sha256,
            valid_last_bin_count=n_windows, activity_fifo_capacity=plan.FIFO_CAPACITY[budget],
            accepted_updates=accepted if system == plan.SYSTEM_CDMD else 0,
            rejected_updates=rejected if system == plan.SYSTEM_CDMD else {},
            group_forward_count=group_forward_count if system == plan.SYSTEM_CDMD else 0,
            full_system_forward_count=full_system_forward_count, dropout_calls=0, target_label_state_uses=0,
        )

    def _make_cell_evidence(
        self,
        *, surface: str, budget: int, system: str, input_authority_sha256: str,
        sessions: Sequence[Any], resources: Mapping[str, object],
    ) -> Any:
        """Historical V1 cell codec; successors can provide typed evidence."""
        return score.CellEvidence(
            surface=surface, budget=budget, system=system,
            input_authority_sha256=input_authority_sha256,
            model_swa_sha256=plan.SEALED_CELL_D_SWA_SHA256,
            sessions=tuple(sessions), resources=resources,
        )

    def _score_session(
        self, *, session: PreparedEvaluationSession, budget: int, system: str,
    ) -> Any:
        state = self._require_state()
        np, torch, core = state.modules["np"], state.modules["torch"], state.modules["core"]
        memory, initial = self._initial_memory(session=session, budget=budget)
        initial_inputs = memory.read_prediction_inputs()
        predictions: list[Any] = []
        targets: list[Any] = []
        transitions: list[CompletedTrialTransition] = []
        group_forwards = 0
        full_forwards = 0
        first_state = state.modules["arm_common"].state_sha256(state.model)
        for trial_id in session.query_trial_ids[budget]:
            trial = session.trials_by_id[trial_id]
            inputs = initial_inputs if system == plan.SYSTEM_SEALED else memory.read_prediction_inputs()
            side = self._normalized_side(inputs.active_t4).detach().cpu().numpy()
            full, evidence = self._forward_full(
                neural_windows=trial.neural_windows, activity_stack=inputs.activity_trials, normalized_t4=side,
            )
            full_forwards += int(evidence["full_system_forward_count"])
            endpoint_targets = torch.from_numpy(np.ascontiguousarray(session.behavior[trial.endpoint_bins], dtype=np.float32))
            valid = torch.all(endpoint_targets != -1.0, dim=-1)
            if bool(valid.any().item()):
                predictions.append(full[:, plan.GOVERNING_BIN, :][valid].detach().cpu().contiguous())
                targets.append(endpoint_targets[valid].detach().cpu().contiguous())
            if system == plan.SYSTEM_CDMD:
                group_predictions, group_evidence = self._group_predictions(trial=trial, memory=memory)
                group_forwards += sum(2 * int(row["forward_chunk_count"]) for row in group_evidence)
                pending = memory.observe_completed_trial(
                    b3s_trial_activity=trial.b3s_activity, carrier_trial_counts=trial.native_counts,
                    complementary_predictions=group_predictions,
                )
                transition = self._commit_completed_transition(memory=memory, pending=pending, trial_id=trial_id)
                transitions.append(transition)
                # M30's capacity-zero state is a strict B3S activity no-op;
                # its accepted carrier statistics remain intentionally live.
                if budget == 30 and transition.activity_after_sha256 != transition.activity_before_sha256:
                    raise PhysicalCDMDScoreError("M30 activity FIFO mutated despite literal capacity zero")
                if not transition.carrier_transition_committed and transition.carrier_after_sha256 != transition.carrier_before_sha256:
                    raise PhysicalCDMDScoreError("rejected target pseudo update changed CDM-D carrier state")
            state.completed_trials += 1
            if state.measurement_started is None:
                state.measurement_started = time.monotonic()
        if not predictions:
            raise PhysicalCDMDScoreError(f"M{budget} session has no valid governing endpoint")
        prediction = torch.cat(predictions, dim=0).contiguous()
        target = torch.cat(targets, dim=0).contiguous()
        n_windows = int(prediction.shape[0])
        expected = session.input_record_payload
        key = str(budget)
        full_target, full_mask, valid_count = self._governing_target_mask_authority(
            np, behavior=session.behavior, query_trial_ids=session.query_trial_ids[budget],
            trials_by_id=session.trials_by_id,
        )
        if (
            n_windows != valid_count
            or n_windows != expected["valid_last_bin_count_by_budget"][key]
            or _array_sha(np, full_target) != expected["target_last_bin_sha256_by_budget"][key]
            or _array_sha(np, full_mask) != expected["valid_last_bin_mask_sha256_by_budget"][key]
        ):
            raise PhysicalCDMDScoreError("governing target/mask/count differs from immutable input authority")
        try:
            r2 = float(state.modules["matched_metric"].session_r2(prediction, target))
        except Exception as error:
            raise PhysicalCDMDScoreError("reviewed last-bin variance-weighted R2 failed") from error
        last_state = state.modules["arm_common"].state_sha256(state.model)
        if first_state != last_state:
            raise PhysicalCDMDScoreError("model state drifted across CDM-D session")
        digest = hashlib.sha256(prediction.numpy().tobytes()).hexdigest()
        return self._make_session_score(
            session=session, budget=budget, system=system, n_windows=n_windows, r2=r2,
            prediction_sha256=digest, model_state_before_sha256=first_state,
            model_state_after_sha256=last_state, initial=initial,
            full_system_forward_count=full_forwards, group_forward_count=group_forwards,
            transitions=tuple(transitions),
            target_last_bin_sha256=str(expected["target_last_bin_sha256_by_budget"][key]),
            valid_mask_sha256=str(expected["valid_last_bin_mask_sha256_by_budget"][key]),
            sealed_model_load_proof_sha256=_object_sha(state.sealed_load_proof),
        )

    def _resources(self) -> Mapping[str, object]:
        state = self._require_state()
        torch = state.modules["torch"]
        torch.cuda.synchronize(0)
        wall = 0.0 if state.measurement_started is None else float(time.monotonic() - state.measurement_started)
        if wall <= 0.0 or state.completed_trials <= 0 or state.forward_chunks <= 0:
            raise PhysicalCDMDScoreError("score resource counters lack measured forwards/trials/wall time")
        current_alloc = int(torch.cuda.memory_allocated(0))
        current_reserved = int(torch.cuda.memory_reserved(0))
        peak_alloc = int(torch.cuda.max_memory_allocated(0))
        peak_reserved = int(torch.cuda.max_memory_reserved(0))
        if peak_alloc < current_alloc or peak_reserved < current_reserved:
            raise PhysicalCDMDScoreError("score CUDA current/peak memory ordering drift")
        return {
            "runtime_environment": {**state.device_profile, "visible_devices": 1, "attested": True,
                                    "torch_cuda_matmul_allow_tf32": False, "torch_cudnn_allow_tf32": False},
            "current_cuda_allocated_bytes": current_alloc,
            "current_cuda_reserved_bytes": current_reserved,
            "peak_cuda_allocated_bytes": peak_alloc,
            "peak_cuda_reserved_bytes": peak_reserved,
            "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "wall_seconds": wall,
            "full_and_group_forward_chunks": int(state.forward_chunks),
            "completed_query_trials": int(state.completed_trials),
            "windows_or_trials_per_s": float(state.completed_trials) / wall,
        }

    def score_budget(self, *, budget: int, input_authority_sha256: str,
                     identity: plan.ScoreIdentity) -> Sequence[Any]:
        state = self._require_state()
        _sha(input_authority_sha256, "score input authority SHA")
        if budget not in plan.BUDGETS:
            raise PhysicalCDMDScoreError("score budget is outside frozen M30/M10/M4 topology")
        cells: list[Any] = []
        for surface in plan.SURFACES:
            ordered = tuple(item.session for item in state.sessions.values() if item.surface == surface)
            if len(ordered) != plan.expected_session_count(surface):
                raise PhysicalCDMDScoreError("materialized surface roster cardinality drift")
            for system in plan.SYSTEMS:
                rows = tuple(self._score_session(
                    session=state.sessions[(surface, session)], budget=budget, system=system,
                ) for session in ordered)
                cells.append(self._make_cell_evidence(
                    surface=surface, budget=budget, system=system,
                    input_authority_sha256=input_authority_sha256,
                    sessions=rows, resources=self._resources(),
                ))
        return tuple(cells)

    def revalidate(self, *, identity: plan.ScoreIdentity) -> None:
        state = self._require_state()
        if identity.payload()["selected_device_profile"] != state.device_profile:
            raise PhysicalCDMDScoreError("score final identity/device profile drift")
        for held in state.held_assets:
            held.reverify()
        for root in state.held_roots:
            root.reverify()
        expected_state = state.modules["arm_common"].state_sha256(state.model)
        if not isinstance(expected_state, str) or len(expected_state) != 64:
            raise PhysicalCDMDScoreError("score final sealed model state digest drift")

    def failure_progress(self) -> Mapping[str, object]:
        """Report physical progress without touching a new target/model path."""
        state = self.state
        if state is not None:
            self._cuda_initialized = bool(state.modules["torch"].cuda.is_initialized())
            forward_chunks = int(state.forward_chunks)
            full_forwards = int(state.full_system_forwards)
        else:
            forward_chunks = 0
            full_forwards = 0
        return {
            "within_assets_opened": self._within_opened,
            "external_assets_opened": self._external_opened,
            "checkpoint_opened": self._checkpoint_opened,
            "cuda_initialized": self._cuda_initialized,
            "full_system_forward_count": full_forwards,
            "group_forward_count": max(0, forward_chunks - full_forwards),
        }

    def close(self) -> None:
        state = self.state
        if state is None or state.closed:
            return
        state.closed = True
        try:
            for held in reversed(state.held_assets):
                try:
                    held.close()
                except Exception:
                    pass
            for root in reversed(state.held_roots):
                try:
                    root.close()
                except Exception:
                    pass
        finally:
            state.executor.close()


class PhysicalCDMDMatchedScoreBackend:
    """The lifecycle-facing backend with an injectable, no-data runtime seam."""

    def __init__(
        self,
        *,
        root: Path,
        selected_device_profile: Mapping[str, object],
        runtime_factory: Callable[[Path, Mapping[str, object]], RuntimeProtocol] | None = None,
    ) -> None:
        self.root = Path(root).absolute()
        self.profile = plan.validate_compatible_device_profile(selected_device_profile)
        self._runtime_factory = runtime_factory or (
            lambda root, profile: ReviewedCDMScoreRuntime(root=root, selected_device_profile=profile)
        )
        self._runtime: RuntimeProtocol | None = None
        self._last_runtime: RuntimeProtocol | None = None

    def preflight(self, *, root: Path, identity: plan.ScoreIdentity) -> Mapping[str, object]:
        if Path(root).absolute() != self.root:
            raise PhysicalCDMDScoreError("physical score backend root drift")
        if identity.payload()["selected_device_profile"] != self.profile:
            raise PhysicalCDMDScoreError("physical score backend selected device/profile drift")
        return {
            "target_paths_resolved": False,
            "target_opened": False,
            "checkpoint_opened": False,
            "cuda_initialized": False,
            "source_gate_reloaded": False,
            "normalizer_refit": False,
        }

    def prepare(self, *, root: Path, identity: plan.ScoreIdentity) -> RuntimeProtocol:
        if self._runtime is not None:
            raise PhysicalCDMDScoreError("physical score runtime prepared more than once")
        runtime = self._runtime_factory(Path(root).absolute(), self.profile)
        self._last_runtime = runtime
        runtime.prepare(identity=identity)
        self._runtime = runtime
        return runtime

    def materialize_inputs(self, runtime: RuntimeProtocol, *, identity: plan.ScoreIdentity,
                           evaluation_authority: score.FixedEvaluationAuthority) -> score.InputAuthority:
        if runtime is not self._runtime:
            raise PhysicalCDMDScoreError("physical score input runtime identity drift")
        # Descriptor derive again immediately before any target asset is held;
        # a durable caller payload alone can never choose a target row.
        observed = score.derive_fixed_evaluation_authority(self.root)
        if observed.payload() != evaluation_authority.payload():
            raise PhysicalCDMDScoreError("physical score fixed target authority drift before input resolution")
        return runtime.materialize_inputs(identity=identity, authority=evaluation_authority)

    def score_budget(self, runtime: RuntimeProtocol, *, budget: int, input_authority_sha256: str,
                     identity: plan.ScoreIdentity) -> Sequence[score.CellEvidence]:
        if runtime is not self._runtime:
            raise PhysicalCDMDScoreError("physical score budget runtime identity drift")
        return runtime.score_budget(budget=budget, input_authority_sha256=input_authority_sha256, identity=identity)

    def revalidate(self, runtime: RuntimeProtocol, *, root: Path, identity: plan.ScoreIdentity) -> None:
        if runtime is not self._runtime or Path(root).absolute() != self.root:
            raise PhysicalCDMDScoreError("physical score final runtime/root drift")
        runtime.revalidate(identity=identity)
        # Final descriptor rederivation occurs before the atomic score/terminal
        # group so a mutable manifest/ledger cannot leave a partial science row.
        score.derive_fixed_evaluation_authority(self.root)

    def failure_progress(self, runtime: RuntimeProtocol | None) -> Mapping[str, object]:
        candidate = runtime or self._runtime or self._last_runtime
        if candidate is None:
            return {
                "within_assets_opened": False, "external_assets_opened": False,
                "checkpoint_opened": False, "cuda_initialized": False,
                "full_system_forward_count": 0, "group_forward_count": 0,
            }
        return candidate.failure_progress()

    def close(self, runtime: RuntimeProtocol | None) -> None:
        candidate = runtime or self._runtime or self._last_runtime
        try:
            if candidate is not None:
                candidate.close()
        finally:
            self._runtime = None
            self._last_runtime = None


def build_reviewed_physical_backend(
    *, root: Path, selected_device_profile: Mapping[str, object],
) -> PhysicalCDMDMatchedScoreBackend:
    """Construct the sole production backend without opening data or CUDA."""
    return PhysicalCDMDMatchedScoreBackend(
        root=Path(root), selected_device_profile=plan.validate_compatible_device_profile(selected_device_profile),
    )


def execute_reviewed_physical_score(
    root: Path, *, identity: plan.ScoreIdentity, capability: object,
    environ: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """The sole production launch adapter; public CLI cannot call this.

    All metadata/closure/source-gate checks and the no-data backend preflight
    happen before the fresh score root is reserved.  The injected lifecycle
    then writes its durable attempt before this backend can import Torch,
    strict-load the SWA, hold an evaluation asset, or initialize CUDA.
    """
    profile = plan.validate_compatible_device_profile(identity.payload()["selected_device_profile"])
    score.validate_selected_launch_environment(identity, environ)
    backend = build_reviewed_physical_backend(root=Path(root), selected_device_profile=profile)
    preflight, authorization, pre_sha, auth_sha = score.load_durable_authority(Path(root), identity=identity)
    backend_preflight = backend.preflight(root=Path(root), identity=identity)
    if any(backend_preflight.get(key) is not False for key in (
        "target_paths_resolved", "target_opened", "checkpoint_opened", "cuda_initialized",
    )):
        raise PhysicalCDMDScoreError("reviewed score backend preflight is not target/model/CUDA-free")
    artifact = score.reserve_score_artifact(
        Path(root), identity=identity, capability=capability, environ=environ,
    )
    return score.run_authorized_score_lifecycle(
        Path(root), identity=identity, capability=capability, backend=backend, artifact=artifact,
        official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
        preflight=preflight, authorization=authorization,
    )
