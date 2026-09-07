"""Route-local O1/O2 evaluator acceleration primitives.

The module deliberately owns no parser, result lifecycle, authority, model
weights, or transition decision.  Its only job is to cache an identity that is
already a pure function of an exact trial state and to sample the existing
repeat-forward integrity proof at predeclared coordinates.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from . import plan


class IdentityAccelerationError(RuntimeError):
    """Fail closed for a non-equivalent identity cache or repeat audit."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IdentityAccelerationError(message)


def _contiguous_array_bytes(value: Any, *, label: str) -> tuple[bytes, str, tuple[int, ...]]:
    """Return canonical finite tensor/array bytes without retaining a view.

    This function is used only on active evaluator state after its own
    capability boundary.  It is intentionally not reachable from the public
    dry CLI.
    """
    import numpy as np

    candidate = value.detach() if hasattr(value, "detach") else value
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    array = np.ascontiguousarray(np.asarray(candidate))
    _require(array.ndim >= 1 and bool(np.isfinite(array).all()),
             f"accelerated {label} must be finite and non-scalar")
    return array.tobytes(order="C"), str(array.dtype), tuple(int(item) for item in array.shape)


def canonical_array_sha256(value: Any, *, label: str) -> str:
    body, dtype, shape = _contiguous_array_bytes(value, label=label)
    header = f"{label}|{dtype}|{shape}|".encode("ascii")
    return hashlib.sha256(header + body).hexdigest()


def exact_state_cache_key(
    *, path: str, activity_stack: Any, normalized_t4: Any, held_mask: Any | None,
) -> str:
    """Bind a cache entry to the precise encoder-relevant evaluator state."""
    _require(isinstance(path, str) and path, "accelerated cache path is absent")
    pieces = [
        ("activity", canonical_array_sha256(activity_stack, label="activity")),
        ("normalized_t4", canonical_array_sha256(normalized_t4, label="normalized_t4")),
    ]
    if held_mask is not None:
        pieces.append(("held_mask", canonical_array_sha256(held_mask, label="held_mask")))
    body = "|".join(f"{name}:{digest}" for name, digest in (("path", path), *pieces)).encode("ascii")
    return hashlib.sha256(body).hexdigest()


def _require_identity_fast_path(model: Any) -> None:
    """Reject a model whose encoder gate cannot be replayed by identity alone."""
    _require(callable(getattr(model, "compute_identity", None)),
             "accelerated model lacks compute_identity")
    _require(getattr(model, "decoder_mode", None) == "coupled",
             "accelerated identity cache requires exact coupled decoder mode")
    encoder = getattr(model, "id_encoder", None)
    _require(encoder is not None, "accelerated model identity encoder is absent")
    # StreamingSpintModel.forward consumes this gate only on the eager path.
    # Passing `identity=` would silently lose it, so a positive capability is
    # a hard incompatibility rather than a best-effort optimization.
    _require(not callable(getattr(encoder, "forward_batch_with_gate", None)),
             "accelerated identity cache forbids encoder forward_batch_with_gate semantics")


def _as_batched_calibration(value: Any) -> Any:
    _require(getattr(value, "ndim", None) == 3, "accelerated activity stack must be [M,T,N]")
    return value.unsqueeze(0)


def _as_batched_side(value: Any) -> Any:
    _require(getattr(value, "ndim", None) == 2 and int(value.shape[-1]) == 4,
             "accelerated normalized T4 must be [N,4]")
    return value.unsqueeze(0)


@dataclass(frozen=True)
class IdentityCacheEntry:
    """One B=1 identity and immutable provenance for a trial state."""

    key_sha256: str
    identity: Any
    identity_sha256: str
    activity_sha256: str
    normalized_t4_sha256: str
    held_mask_sha256: str | None

    def payload(self) -> dict[str, object]:
        return {
            "cache_key_sha256": self.key_sha256,
            "identity_sha256": self.identity_sha256,
            "activity_sha256": self.activity_sha256,
            "normalized_t4_sha256": self.normalized_t4_sha256,
            "held_mask_sha256": self.held_mask_sha256,
            "identity_batch_size": 1,
        }


@dataclass(frozen=True)
class RepeatCoordinate:
    """A concrete sampled duplicate-forward coordinate."""

    path: str
    query_index: int
    query_trial_id: str
    chunk_start: int
    cache_key_sha256: str
    repeated_prediction_sha256: str
    outputs_bitwise_equal: bool
    reasons: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        _require(self.path == "full" or self.path in {f"held_group_{group}" for group in range(4)},
                 "accelerated repeat coordinate path drift")
        _require(type(self.query_index) is int and self.query_index >= 0 and self.query_trial_id,
                 "accelerated repeat coordinate trial binding drift")
        _require(type(self.chunk_start) is int and self.chunk_start == 0,
                 "accelerated repeat coordinate must bind a first logical chunk")
        _require(self.reasons and set(self.reasons) <= {
            "canonical_full_first_state", "held_group_0_fixed_mid_session",
        },
                 "accelerated repeat coordinate reason drift")
        _require(type(self.outputs_bitwise_equal) is bool and self.outputs_bitwise_equal,
                 "accelerated repeated outputs are not bitwise equal")
        _require(len(self.cache_key_sha256) == 64 and len(self.repeated_prediction_sha256) == 64,
                 "accelerated repeat coordinate digest drift")
        return {
            "path": self.path,
            "group": (None if self.path == "full" else int(self.path.removeprefix("held_group_"))),
            "query_index": self.query_index,
            "query_trial_id": self.query_trial_id,
            "chunk_start": self.chunk_start,
            "cache_key_sha256": self.cache_key_sha256,
            "repeated_prediction_sha256": self.repeated_prediction_sha256,
            "outputs_bitwise_equal": self.outputs_bitwise_equal,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class AcceleratedForwardResult:
    """Prediction plus exact accounting, but never a retained target."""

    prediction: Any
    prediction_sha256: str
    logical_chunk_count: int
    actual_model_forward_count: int
    identity_encoder_forward_count: int
    cache_key_sha256: str
    identity_sha256: str
    repeated_outputs_bitwise_equal: bool

    def payload(self) -> dict[str, object]:
        return {
            "prediction_sha256": self.prediction_sha256,
            "logical_chunk_count": self.logical_chunk_count,
            "actual_model_forward_count": self.actual_model_forward_count,
            "identity_encoder_forward_count": self.identity_encoder_forward_count,
            "cache_key_sha256": self.cache_key_sha256,
            "identity_sha256": self.identity_sha256,
            "repeated_outputs_bitwise_equal": self.repeated_outputs_bitwise_equal,
            "logical_eval_batch_size": plan.LOGICAL_EVAL_BATCH_SIZE,
        }


@dataclass
class DeterministicRepeatAudit:
    """Record only fixed repeat coordinates for one scored session/system."""

    total_query_trials: int = 0
    current_query_index: int = -1
    current_query_trial_id: str = ""
    _seen_canonical_full_states: set[str] = field(default_factory=set)
    _seen_held_group_0_mid_session: bool = False
    coordinates: list[RepeatCoordinate] = field(default_factory=list)

    def begin_session(self, *, total_query_trials: int) -> None:
        _require(type(total_query_trials) is int and total_query_trials > 0,
                 "accelerated repeat audit needs positive query cardinality")
        self.total_query_trials = total_query_trials
        self.current_query_index = -1
        self.current_query_trial_id = ""
        self._seen_canonical_full_states.clear()
        self._seen_held_group_0_mid_session = False
        self.coordinates.clear()

    def begin_query(self, query_index: int, *, query_trial_id: str) -> None:
        _require(type(query_index) is int and 0 <= query_index < self.total_query_trials,
                 "accelerated repeat audit query coordinate drift")
        _require(isinstance(query_trial_id, str) and query_trial_id,
                 "accelerated repeat audit query trial identifier drift")
        self.current_query_index = query_index
        self.current_query_trial_id = query_trial_id

    @property
    def mid_query_index(self) -> int:
        _require(self.total_query_trials > 0, "accelerated repeat audit session absent")
        return self.total_query_trials // 2

    def reasons_for(self, *, path: str, cache_key_sha256: str, chunk_start: int) -> tuple[str, ...]:
        _require(self.current_query_index >= 0, "accelerated repeat audit query was not entered")
        _require(type(chunk_start) is int and chunk_start >= 0, "accelerated chunk start drift")
        _require(path == "full" or path in {f"held_group_{group}" for group in range(4)},
                 "accelerated repeat audit path drift")
        _require(isinstance(cache_key_sha256, str) and len(cache_key_sha256) == 64,
                 "accelerated repeat audit cache key drift")
        if chunk_start != 0:
            return ()
        reasons: list[str] = []
        # O2 is intentionally *not* a repeat per held group.  It proves the
        # canonical full-system state after each transition, plus exactly one
        # predeclared held-group coordinate for path coverage.  All other
        # logical chunks have one forward only.
        if path == "full" and cache_key_sha256 not in self._seen_canonical_full_states:
            reasons.append("canonical_full_first_state")
        if (
            path == "held_group_0"
            and self.current_query_index == self.mid_query_index
            and not self._seen_held_group_0_mid_session
        ):
            reasons.append("held_group_0_fixed_mid_session")
        return tuple(reasons)

    def record(
        self,
        *,
        path: str,
        cache_key_sha256: str,
        chunk_start: int,
        repeated_prediction_sha256: str,
        outputs_bitwise_equal: bool,
        reasons: Iterable[str],
    ) -> None:
        exact = tuple(reasons)
        if not exact:
            return
        coordinate = RepeatCoordinate(
            path=path, query_index=self.current_query_index, query_trial_id=self.current_query_trial_id,
            chunk_start=chunk_start, cache_key_sha256=cache_key_sha256,
            repeated_prediction_sha256=repeated_prediction_sha256,
            outputs_bitwise_equal=outputs_bitwise_equal, reasons=exact,
        )
        coordinate.payload()
        self.coordinates.append(coordinate)
        if "canonical_full_first_state" in exact:
            self._seen_canonical_full_states.add(cache_key_sha256)
        if "held_group_0_fixed_mid_session" in exact:
            self._seen_held_group_0_mid_session = True

    def require_complete_coverage(self, *, require_held_group_0: bool) -> None:
        """Require the proof topology applicable to the scored system."""
        _require(bool(self._seen_canonical_full_states),
                 "accelerated repeat audit lacks canonical full-state coverage")
        if require_held_group_0:
            _require(self._seen_held_group_0_mid_session,
                     "accelerated repeat audit lacks held_group_0 mid-session coverage")

    def payload(self) -> dict[str, object]:
        _require(self.total_query_trials > 0, "accelerated repeat audit session absent")
        return {
            "schema": "causal_dual_memory_cell_d_speed_v1_repeat_audit_v1",
            "total_query_trials": self.total_query_trials,
            "fixed_mid_query_index": self.mid_query_index,
            "canonical_full_first_chunk_per_exact_state": True,
            "held_group_0_fixed_mid_session_first_chunk": True,
            "coordinates": [item.payload() for item in self.coordinates],
            "repeat_audit_uses_no_target_values": True,
        }


class IdentityCacheAccelerator:
    """One-session cache plus sampled-repeat executor for a sealed eval model."""

    def __init__(self, *, logical_batch_size: int = plan.LOGICAL_EVAL_BATCH_SIZE) -> None:
        _require(logical_batch_size == plan.LOGICAL_EVAL_BATCH_SIZE,
                 "pure-speed successor permits only logical B128")
        self.logical_batch_size = logical_batch_size
        self._entries: dict[str, IdentityCacheEntry] = {}
        self.repeat_audit = DeterministicRepeatAudit()
        self._identity_encoder_forwards = 0
        self._actual_model_forwards = 0
        self._actual_model_forwards_by_path: dict[str, int] = {}
        self._logical_chunks_by_path: dict[str, int] = {}

    def begin_session(self, *, total_query_trials: int) -> None:
        self._entries.clear()
        self._identity_encoder_forwards = 0
        self._actual_model_forwards = 0
        self._actual_model_forwards_by_path.clear()
        self._logical_chunks_by_path.clear()
        self.repeat_audit.begin_session(total_query_trials=total_query_trials)

    def begin_query(self, query_index: int, *, query_trial_id: str) -> None:
        self.repeat_audit.begin_query(query_index, query_trial_id=query_trial_id)

    def _entry(
        self, *, model: Any, path: str, activity_stack: Any, normalized_t4: Any, held_mask: Any | None,
    ) -> tuple[IdentityCacheEntry, bool]:
        _require_identity_fast_path(model)
        key = exact_state_cache_key(
            path=path, activity_stack=activity_stack, normalized_t4=normalized_t4, held_mask=held_mask,
        )
        existing = self._entries.get(key)
        if existing is not None:
            return existing, False
        import torch

        calibration = _as_batched_calibration(activity_stack)
        side = _as_batched_side(normalized_t4)
        with torch.no_grad():
            identity = model.compute_identity(calibration, side_features=side)
        _require(getattr(identity, "ndim", None) == 3 and int(identity.shape[0]) == 1
                 and int(identity.shape[1]) == int(normalized_t4.shape[0]),
                 "accelerated cached identity topology drift")
        _require(bool(torch.isfinite(identity).all().item()), "accelerated cached identity is nonfinite")
        entry = IdentityCacheEntry(
            key_sha256=key,
            identity=identity.detach(),
            identity_sha256=canonical_array_sha256(identity, label="identity"),
            activity_sha256=canonical_array_sha256(activity_stack, label="activity"),
            normalized_t4_sha256=canonical_array_sha256(normalized_t4, label="normalized_t4"),
            held_mask_sha256=(canonical_array_sha256(held_mask, label="held_mask") if held_mask is not None else None),
        )
        self._entries[key] = entry
        self._identity_encoder_forwards += 1
        return entry, True

    def forward(
        self,
        *,
        model: Any,
        neural_windows: Any,
        activity_stack: Any,
        normalized_t4: Any,
        path: str,
        held_mask: Any | None = None,
    ) -> AcceleratedForwardResult:
        """Decode a logical B128 sequence from one cached exact identity."""
        import torch

        _require(getattr(neural_windows, "ndim", None) == 3 and int(neural_windows.shape[0]) >= 1,
                 "accelerated neural windows must be nonempty [B,W,N]")
        entry, cache_miss = self._entry(
            model=model, path=path, activity_stack=activity_stack,
            normalized_t4=normalized_t4, held_mask=held_mask,
        )
        _require(int(neural_windows.shape[2]) == int(entry.identity.shape[1]),
                 "accelerated neural/identity unit topology drift")
        chunks: list[Any] = []
        output_digest = hashlib.sha256()
        actual_forwards = 0
        logical_chunks = 0
        repeat_equal = True
        for start in range(0, int(neural_windows.shape[0]), self.logical_batch_size):
            stop = min(start + self.logical_batch_size, int(neural_windows.shape[0]))
            with torch.no_grad():
                first_result = model(neural_windows[start:stop], identity=entry.identity)
            _require(isinstance(first_result, tuple) and len(first_result) == 2,
                     "accelerated direct identity forward topology drift")
            first, returned_identity = first_result
            _require(torch.equal(returned_identity, entry.identity),
                     "accelerated direct identity forward returned a different identity")
            _require(bool(torch.isfinite(first).all().item()), "accelerated direct identity prediction is nonfinite")
            actual_forwards += 1
            reasons = self.repeat_audit.reasons_for(
                path=path, cache_key_sha256=entry.key_sha256, chunk_start=start,
            )
            if reasons:
                with torch.no_grad():
                    second_result = model(neural_windows[start:stop], identity=entry.identity)
                _require(isinstance(second_result, tuple) and len(second_result) == 2,
                         "accelerated repeated identity forward topology drift")
                second, second_identity = second_result
                _require(torch.equal(first, second) and torch.equal(second_identity, entry.identity),
                         "accelerated sampled repeat identity forward drift")
                actual_forwards += 1
                self.repeat_audit.record(
                    path=path, cache_key_sha256=entry.key_sha256, chunk_start=start,
                    repeated_prediction_sha256=canonical_array_sha256(first, label="repeat_prediction"),
                    outputs_bitwise_equal=True, reasons=reasons,
                )
            detached = first.detach().cpu().contiguous()
            output_digest.update(detached.numpy().tobytes())
            chunks.append(detached)
            logical_chunks += 1
        self._actual_model_forwards += actual_forwards
        self._actual_model_forwards_by_path[path] = (
            self._actual_model_forwards_by_path.get(path, 0) + actual_forwards
        )
        self._logical_chunks_by_path[path] = self._logical_chunks_by_path.get(path, 0) + logical_chunks
        prediction = torch.cat(chunks, dim=0)
        return AcceleratedForwardResult(
            prediction=prediction,
            prediction_sha256=output_digest.hexdigest(),
            logical_chunk_count=logical_chunks,
            actual_model_forward_count=actual_forwards,
            identity_encoder_forward_count=1 if cache_miss else 0,
            cache_key_sha256=entry.key_sha256,
            identity_sha256=entry.identity_sha256,
            repeated_outputs_bitwise_equal=repeat_equal,
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_cell_d_speed_v1_identity_cache_v1",
            "logical_eval_batch_size": self.logical_batch_size,
            "identity_cache_entry_count": len(self._entries),
            "identity_encoder_forward_count": self._identity_encoder_forwards,
            "actual_model_forward_count": self._actual_model_forwards,
            "actual_model_forward_count_by_path": dict(sorted(self._actual_model_forwards_by_path.items())),
            "logical_chunk_count_by_path": dict(sorted(self._logical_chunks_by_path.items())),
            "cache_key_semantics": plan.IDENTITY_CACHE_KEY_SEMANTICS,
            "repeat_audit": self.repeat_audit.payload(),
            "numeric_batch_variants_deferred": list(plan.DEFERRED_NUMERIC_BATCH_VARIANTS),
        }
