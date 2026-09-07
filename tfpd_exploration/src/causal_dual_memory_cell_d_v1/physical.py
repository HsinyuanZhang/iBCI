"""CPU physical-forward seam for the future CDM-D source safety audit.

The future production adapter must provide a sealed Cell-D callable.  This
module makes its immutable input/output obligations testable now: held units
are physically removed from all three unit-aligned inputs, standardized
velocity is restored before direction integration, and repeated eval forwards
must not mutate state or consume a dropout/RNG path.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
from typing import Any, Callable, Iterator, Protocol

import numpy as np

from . import core


WINDOW_BINS = 50


class PhysicalSeamError(ValueError):
    """Fail closed for an invalid physical Cell-D source-forward seam."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhysicalSeamError(message)


def _immutable(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype)).copy()
    array.setflags(write=False)
    return array


def _sha256(value: Any, *, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"{label} must be an exact lowercase SHA256")
    return value


@dataclass(frozen=True)
class RNGStateSnapshot:
    """Injected exact state proof for every RNG domain a live forward can use.

    The physical seam does not import Torch in this no-CUDA module.  A future
    reviewed runtime must nevertheless snapshot the Torch CPU and CUDA RNG
    byte states alongside Python and NumPy; an omitted domain is not evidence
    that it stayed unchanged.
    """

    python_sha256: str
    numpy_sha256: str
    torch_cpu_sha256: str
    torch_cuda_sha256: str

    def __post_init__(self) -> None:
        for label in ("python_sha256", "numpy_sha256", "torch_cpu_sha256", "torch_cuda_sha256"):
            _sha256(getattr(self, label), label=label)

    def payload(self) -> dict[str, str]:
        return {
            "python_sha256": self.python_sha256,
            "numpy_sha256": self.numpy_sha256,
            "torch_cpu_sha256": self.torch_cpu_sha256,
            "torch_cuda_sha256": self.torch_cuda_sha256,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            core.canonical_json_bytes(self.payload())
        ).hexdigest()


@dataclass(frozen=True)
class PhysicalForwardEvidence:
    """Receipt-ready proof emitted by an inspected held-unit eval forward."""

    model_state_before_sha256: str
    model_state_after_sha256: str
    rng_before: RNGStateSnapshot
    rng_after: RNGStateSnapshot
    dynamic_dropout_calls_before: int
    dynamic_dropout_calls_after: int
    repeated_outputs_bitwise_equal: bool

    def __post_init__(self) -> None:
        _sha256(self.model_state_before_sha256, label="model_state_before_sha256")
        _sha256(self.model_state_after_sha256, label="model_state_after_sha256")
        _require(isinstance(self.rng_before, RNGStateSnapshot) and isinstance(self.rng_after, RNGStateSnapshot),
                 "physical forward RNG proof must cover Python/NumPy/Torch/CUDA")
        _require(isinstance(self.dynamic_dropout_calls_before, int)
                 and isinstance(self.dynamic_dropout_calls_after, int)
                 and self.dynamic_dropout_calls_before >= 0 and self.dynamic_dropout_calls_after >= 0,
                 "physical forward dropout counters must be nonnegative integers")
        _require(self.repeated_outputs_bitwise_equal is True,
                 "physical forward evidence requires bitwise repeated outputs")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_physical_forward_evidence_v2",
            "model_state_before_sha256": self.model_state_before_sha256,
            "model_state_after_sha256": self.model_state_after_sha256,
            "rng_before": self.rng_before.payload(),
            "rng_after": self.rng_after.payload(),
            "rng_unchanged": self.rng_before == self.rng_after,
            "dynamic_dropout_calls_before": self.dynamic_dropout_calls_before,
            "dynamic_dropout_calls_after": self.dynamic_dropout_calls_after,
            "dynamic_dropout_unchanged": self.dynamic_dropout_calls_before == self.dynamic_dropout_calls_after,
            "repeated_outputs_bitwise_equal": self.repeated_outputs_bitwise_equal,
        }


@dataclass(frozen=True)
class HeldUnitInputs:
    """The three physical input views after one group has been removed."""

    neural_windows: np.ndarray = field(repr=False, compare=False)
    b3s_activity_stack: np.ndarray = field(repr=False, compare=False)
    normalized_t4: np.ndarray = field(repr=False, compare=False)
    retained_channel_indices: np.ndarray = field(repr=False, compare=False)
    held_channel_indices: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        neural = np.asarray(self.neural_windows)
        b3s = np.asarray(self.b3s_activity_stack)
        t4 = np.asarray(self.normalized_t4)
        retained = np.asarray(self.retained_channel_indices)
        held = np.asarray(self.held_channel_indices)
        _require(neural.ndim == 3 and neural.shape[1] == WINDOW_BINS and neural.shape[2] >= 1,
                 "held neural input must be [P,50,N_remaining]")
        _require(b3s.ndim == 3 and b3s.shape[0] == 30 and b3s.shape[1] == 100 and b3s.shape[2] == neural.shape[2],
                 "held B3S input must be [30,100,N_remaining]")
        _require(t4.shape == (neural.shape[2], 4), "held normalized T4 must be [N_remaining,4]")
        _require(retained.ndim == held.ndim == 1 and np.issubdtype(retained.dtype, np.integer)
                 and np.issubdtype(held.dtype, np.integer), "held channel index dtype drift")
        _require(retained.size == neural.shape[2] and held.size >= 1, "held-unit slicing count drift")
        _require(np.isfinite(neural).all() and np.isfinite(b3s).all() and np.isfinite(t4).all(),
                 "held physical inputs must be finite")
        object.__setattr__(self, "neural_windows", _immutable(neural))
        object.__setattr__(self, "b3s_activity_stack", _immutable(b3s))
        object.__setattr__(self, "normalized_t4", _immutable(t4))
        object.__setattr__(self, "retained_channel_indices", _immutable(retained, dtype=np.int64))
        object.__setattr__(self, "held_channel_indices", _immutable(held, dtype=np.int64))


def physically_slice_held_units(
    neural_windows: Any,
    b3s_activity_stack: Any,
    normalized_t4: Any,
    held_unit_mask: Any,
) -> HeldUnitInputs:
    """Remove held unit rows rather than zeroing retained attention tokens."""
    neural = np.asarray(neural_windows)
    b3s = np.asarray(b3s_activity_stack)
    t4 = np.asarray(normalized_t4)
    held = np.asarray(held_unit_mask)
    _require(neural.ndim == 3 and neural.shape[1] == WINDOW_BINS, "neural windows must be [P,50,N]")
    _require(b3s.ndim == 3 and b3s.shape[:2] == (30, 100), "B3S stack must be [30,100,N]")
    _require(t4.ndim == 2 and t4.shape[1] == 4, "normalized T4 must be [N,4]")
    _require(neural.shape[2] == b3s.shape[2] == t4.shape[0], "unit-aligned physical inputs drift")
    _require(held.dtype == np.bool_ and held.shape == (neural.shape[2],) and bool(held.any()) and bool((~held).any()),
             "held-unit mask must remove a nonempty proper unit subset")
    active = np.flatnonzero(~held).astype(np.int64, copy=False)
    held_indices = np.flatnonzero(held).astype(np.int64, copy=False)
    return HeldUnitInputs(
        neural[:, :, active],
        b3s[:, :, active],
        t4[active],
        active,
        held_indices,
    )


def restore_physical_velocity(standardized_velocity: Any, *, behavior_mean: Any, behavior_std: Any) -> np.ndarray:
    """Undo sealed behavior z-scoring before displacement integration."""
    velocity = np.asarray(standardized_velocity)
    mean = np.asarray(behavior_mean)
    std = np.asarray(behavior_std)
    _require(velocity.ndim == 2 and velocity.shape[1] == 2 and np.issubdtype(velocity.dtype, np.floating)
             and np.isfinite(velocity).all(), "standardized velocity must be finite [P,2]")
    _require(mean.shape == std.shape == (2,) and np.issubdtype(mean.dtype, np.floating)
             and np.issubdtype(std.dtype, np.floating) and np.isfinite(mean).all() and np.isfinite(std).all()
             and np.all(std > 0.0), "sealed behavior mean/std must be finite [2] with positive std")
    result = np.asarray(velocity * std[None, :] + mean[None, :], dtype=np.float64)
    _require(np.isfinite(result).all(), "physical velocity restoration became nonfinite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SealedSourceBehaviorNormalizer:
    """Exact source behavior normalizer required by physical direction audit."""

    mean: np.ndarray = field(repr=False, compare=False)
    std: np.ndarray = field(repr=False, compare=False)
    authority_sha256: str

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean)
        std = np.asarray(self.std)
        _require(mean.shape == std.shape == (2,) and np.issubdtype(mean.dtype, np.floating)
                 and np.issubdtype(std.dtype, np.floating) and np.isfinite(mean).all() and np.isfinite(std).all()
                 and np.all(std > 0.0), "sealed source behavior normalizer must be finite [2]")
        _require(isinstance(self.authority_sha256, str) and len(self.authority_sha256) == 64
                 and all(character in "0123456789abcdef" for character in self.authority_sha256),
                 "sealed source behavior normalizer authority SHA drift")
        object.__setattr__(self, "mean", _immutable(mean, dtype=np.float64))
        object.__setattr__(self, "std", _immutable(std, dtype=np.float64))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_sealed_source_behavior_normalizer_v1",
            "mean": {"sha256": core.array_digest(self.mean), "shape": [2]},
            "std": {"sha256": core.array_digest(self.std), "shape": [2]},
            "authority_sha256": self.authority_sha256,
        }


class CellDLikeSetReader(Protocol):
    """Sealed StreamingSpintModel-compatible no-grad/eval source interface.

    The keyword names and batch layouts intentionally match the actual
    ``StreamingSpintModel.forward(neural, calib_trials=..., side_features=...)``
    seam.  The future physical adapter will provide Torch tensors; synthetic
    tests use NumPy values with the same axes.
    """

    def eval(self) -> None: ...

    def state_digest(self) -> str: ...

    def __call__(self, neural_windows: np.ndarray, *, calib_trials: np.ndarray, side_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class CPUSyntheticCellDSetReader:
    """Small deterministic set reader for synthetic held-slicing tests only."""

    projection: np.ndarray = field(default_factory=lambda: np.asarray([[0.8, -0.2], [0.1, 0.7], [0.3, 0.2], [0.0, 0.4]], dtype=np.float64))
    eval_calls: int = 0
    no_grad_calls: int = 0
    dynamic_dropout_calls: int = 0

    def __post_init__(self) -> None:
        projection = np.asarray(self.projection)
        _require(projection.shape == (4, 2) and np.isfinite(projection).all(), "synthetic reader projection drift")
        self.projection = _immutable(projection)

    def eval(self) -> None:
        self.eval_calls += 1

    @contextmanager
    def no_grad(self) -> Iterator[None]:
        self.no_grad_calls += 1
        yield

    def state_digest(self) -> str:
        return core.sha256_bytes(core.canonical_json_bytes({
            "projection": core.array_digest(self.projection),
            "dynamic_dropout_calls": self.dynamic_dropout_calls,
        }))

    def __call__(self, neural_windows: np.ndarray, *, calib_trials: np.ndarray, side_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        neural = np.asarray(neural_windows)
        b3s = np.asarray(calib_trials)
        t4 = np.asarray(side_features)
        _require(neural.ndim == 3 and neural.shape[1] == WINDOW_BINS,
                 "synthetic reader neural/T4 shape drift")
        _require(b3s.shape == (neural.shape[0], 30, 100, neural.shape[2]), "synthetic reader B3S shape drift")
        _require(t4.shape == (neural.shape[0], neural.shape[2], 4), "synthetic reader side-feature shape drift")
        # A deterministic set aggregation.  No hidden unit slots or zeros for
        # held units remain: the three arrays already have N_remaining columns.
        unit_gain = t4 @ self.projection
        neural_scale = neural.mean(axis=(1, 2), dtype=np.float64)[:, None]
        b3s_scale = np.einsum("bmtn,bnd->bd", b3s, t4[:, :, :2], optimize=True) / float(30 * 100)
        last_bin = neural_scale * unit_gain.mean(axis=1, dtype=np.float64) + b3s_scale
        output = np.repeat(np.asarray(last_bin, dtype=np.float64)[:, None, :], WINDOW_BINS, axis=1)
        _require(np.isfinite(output).all(), "synthetic reader output became nonfinite")
        output.setflags(write=False)
        identity = np.asarray(t4.mean(axis=1, dtype=np.float64), dtype=np.float64)
        identity.setflags(write=False)
        return output, identity


def forward_held_completed_trial(
    model: CellDLikeSetReader,
    *,
    neural_windows: Any,
    b3s_activity_stack: Any,
    normalized_t4: Any,
    held_unit_mask: Any,
    behavior_normalizer: SealedSourceBehaviorNormalizer,
    velocity_validity: core.VelocityValidityEvidence,
    no_grad_context: Callable[[], Iterator[None]],
    rng_state_snapshot: Callable[[], RNGStateSnapshot],
    dropout_counter: Callable[[], int],
    evidence_sink: Callable[[PhysicalForwardEvidence], None] | None = None,
) -> core.CompletedVelocityPrediction:
    """Run an eval/no-grad-compatible held forward and restore physical units.

    Repeated equal outputs alone do not prove evaluation purity: a deterministic
    callable can still advance Python, NumPy, Torch CPU, or CUDA RNG state.
    Therefore the caller must inject exact snapshots for all four domains and
    an observed dropout-call counter.  Missing instrumentation fails closed;
    it is never interpreted as a zero count.
    """
    _require(isinstance(velocity_validity, core.VelocityValidityEvidence), "physical seam needs typed validity evidence")
    _require(isinstance(behavior_normalizer, SealedSourceBehaviorNormalizer),
             "physical source forward requires sealed behavior normalizer capability")
    _require(callable(no_grad_context), "physical source forward requires an injected no-grad context")
    _require(callable(rng_state_snapshot),
             "physical source forward requires injected Python/NumPy/Torch/CUDA RNG proof")
    _require(callable(dropout_counter),
             "physical source forward requires an observed dynamic-dropout counter")
    _require(evidence_sink is None or callable(evidence_sink), "physical source forward evidence sink must be callable")
    sliced = physically_slice_held_units(neural_windows, b3s_activity_stack, normalized_t4, held_unit_mask)
    _require(sliced.neural_windows.shape[0] == velocity_validity.valid_mask.size,
             "velocity validity must match physical forward endpoint count")
    before = model.state_digest()
    _sha256(before, label="model state digest before forward")
    rng_before = rng_state_snapshot()
    _require(isinstance(rng_before, RNGStateSnapshot),
             "physical source forward requires a complete RNG proof from its provider")
    before_dropout = dropout_counter()
    _require(isinstance(before_dropout, int) and before_dropout >= 0,
             "physical source forward dropout counter must return a nonnegative integer")
    model.eval()
    calibration_batch = np.repeat(sliced.b3s_activity_stack[None, :, :, :], sliced.neural_windows.shape[0], axis=0)
    side_batch = np.repeat(sliced.normalized_t4[None, :, :], sliced.neural_windows.shape[0], axis=0)
    with no_grad_context():
        first_output = model(
            sliced.neural_windows,
            calib_trials=calibration_batch,
            side_features=side_batch,
        )
        second_output = model(
            sliced.neural_windows,
            calib_trials=calibration_batch,
            side_features=side_batch,
        )
    after = model.state_digest()
    _sha256(after, label="model state digest after forward")
    rng_after = rng_state_snapshot()
    _require(isinstance(rng_after, RNGStateSnapshot),
             "physical source forward requires a complete post-forward RNG proof")
    after_dropout = dropout_counter()
    _require(isinstance(after_dropout, int) and after_dropout >= 0,
             "physical source forward dropout counter must return a nonnegative integer post-forward")
    _require(isinstance(first_output, tuple) and isinstance(second_output, tuple)
             and len(first_output) == len(second_output) == 2,
             "sealed source model must return (behavior, identity)")
    first_standardized, first_identity = first_output
    second_standardized, second_identity = second_output
    _require(np.array_equal(first_standardized, second_standardized) and np.array_equal(first_identity, second_identity),
             "eval source forward must be bitwise-repeatable")
    _require(before == after, "source model state changed during eval forward")
    _require(rng_before == rng_after,
             "source eval forward consumed Python/NumPy/Torch/CUDA RNG state")
    _require(after_dropout == before_dropout,
             "dynamic dropout may not run during source safety audit eval")
    standardized = np.asarray(first_standardized)
    _require(standardized.ndim == 3 and standardized.shape == (sliced.neural_windows.shape[0], WINDOW_BINS, 2)
             and np.issubdtype(standardized.dtype, np.floating) and np.isfinite(standardized).all(),
             "sealed source model must emit finite [P,50,2] standardized behavior")
    physical = restore_physical_velocity(
        standardized[:, WINDOW_BINS - 1, :],
        behavior_mean=behavior_normalizer.mean,
        behavior_std=behavior_normalizer.std,
    )
    evidence = PhysicalForwardEvidence(
        before, after, rng_before, rng_after,
        before_dropout, after_dropout, True,
    )
    if evidence_sink is not None:
        evidence_sink(evidence)
    return core.CompletedVelocityPrediction(physical, velocity_validity)
