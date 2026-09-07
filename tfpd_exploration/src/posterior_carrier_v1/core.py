"""CPU posterior-carrier primitives for the additive Cell-D Stage-0 route.

Nothing in this module discovers a dataset, opens a cache/checkpoint, creates
an output root, or initializes CUDA.  It consumes explicitly supplied,
source-only count/exposure tensors and a caller-supplied Cell-D graph.  The
future data adapter and trainer remain deliberately out of scope.

The implementation makes the scientific interface explicit:

* raw integer prefix counts plus exposure produce one conjugate 3x3 posterior
  per unit;
* a source-only isotropic directional prior is represented as data, never as a
  learned lookup table;
* posterior samples are generated from a local, domain-separated generator;
* only the [a,c] covariance controls parameter-free attention credibility;
* the Cell-D wrapper preserves the existing complete-token dropout law.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .plan import BUDGETS, CELL, HANDOFF_RELATIVE, HANDOFF_SHA256, PHASE


PRIOR_VARIANCE_FLOOR = 1e-12
CREDIBILITY_FLOOR = 1e-6
POSTERIOR_SCHEMA = "posterior_carrier_conjugate_3x3_v1"
SOURCE_PRIOR_SCHEMA = "posterior_carrier_source_prior_v1"
SOURCE_PRIOR_RECEIPT_SCHEMA = "posterior_carrier_source_prior_receipt_v1"
SESSION_POSTERIOR_RECEIPT_SCHEMA = "posterior_carrier_session_posterior_receipt_v1"
POSTERIOR_SOURCE_NORMALIZER_SCHEMA = "posterior_carrier_source_normalizer_v1"

PACKAGE_INIT_RELATIVE = "tfpd_exploration/src/posterior_carrier_v1/__init__.py"
PLAN_RELATIVE = "tfpd_exploration/src/posterior_carrier_v1/plan.py"
CORE_RELATIVE = "tfpd_exploration/src/posterior_carrier_v1/core.py"
CLI_RELATIVE = "tfpd_exploration/scripts/run_posterior_carrier_stage0.py"
TEST_RELATIVE = "tfpd_exploration/tests/test_posterior_carrier_stage0.py"
STAGE0_CLOSURE_PATHS = (
    HANDOFF_RELATIVE,
    PACKAGE_INIT_RELATIVE,
    PLAN_RELATIVE,
    CORE_RELATIVE,
    CLI_RELATIVE,
    TEST_RELATIVE,
)


class PosteriorCarrierError(ValueError):
    """Raised for a malformed posterior-carrier Stage-0 input or invariant."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise PosteriorCarrierError(f"{name} must be an exact lowercase SHA-256")
    return value


def _require_nonempty_unique_strings(values: Sequence[str], name: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise PosteriorCarrierError(f"{name} must be a nonempty ordered sequence")
    values_tuple = tuple(values)
    if any(not isinstance(item, str) or not item for item in values_tuple) or len(set(values_tuple)) != len(values_tuple):
        raise PosteriorCarrierError(f"{name} must contain unique nonempty strings")
    return values_tuple


def tensor_digest(tensor: Tensor) -> str:
    """Hash dtype, shape, and bytes without mutating the tensor or RNG state."""
    if not isinstance(tensor, Tensor):
        raise PosteriorCarrierError("tensor digest requires a Tensor")
    detached = tensor.detach().cpu().contiguous()
    if detached.is_floating_point():
        # Normalize signed zero for a stable semantic artifact binding.
        detached = detached + 0
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("utf-8"))
    digest.update(str(tuple(detached.shape)).encode("utf-8"))
    if detached.numel():
        digest.update(detached.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _require_finite_floating(tensor: Tensor, name: str, *, ndim: int | None = None) -> None:
    if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
        raise PosteriorCarrierError(f"{name} must be a floating Tensor")
    if ndim is not None and tensor.ndim != ndim:
        raise PosteriorCarrierError(f"{name} must have rank {ndim}, got {tensor.ndim}")
    if tensor.numel() == 0 or not torch.isfinite(tensor).all().item():
        raise PosteriorCarrierError(f"{name} must be nonempty and finite")


def _as_direct_integer_counts(counts: Tensor) -> Tensor:
    """Validate direct count observations without rounding a rate surrogate."""
    if not isinstance(counts, Tensor) or counts.ndim != 2 or counts.shape[0] < 1 or counts.shape[1] < 1:
        raise PosteriorCarrierError("counts must be [units,prefix_trials] with nonempty axes")
    if counts.dtype == torch.bool or counts.is_complex():
        raise PosteriorCarrierError("counts must be direct nonnegative integer observations")
    if counts.is_floating_point():
        if not torch.isfinite(counts).all().item() or not torch.equal(counts, torch.trunc(counts)):
            raise PosteriorCarrierError("floating counts must already be exact integers; rates may not be rounded")
    elif counts.dtype not in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    ):
        raise PosteriorCarrierError("counts must use an integer dtype or exact-integer floating representation")
    if bool((counts < 0).any().item()):
        raise PosteriorCarrierError("counts must be nonnegative")
    return counts


def _coerce_exposure(exposure: Tensor | float, trial_count: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if isinstance(exposure, Tensor):
        if not exposure.is_floating_point() or exposure.is_complex():
            raise PosteriorCarrierError("exposure must be finite positive floating duration(s)")
        candidate = exposure.to(device=device, dtype=dtype)
    elif isinstance(exposure, (float, int)) and not isinstance(exposure, bool):
        candidate = torch.tensor(float(exposure), device=device, dtype=dtype)
    else:
        raise PosteriorCarrierError("exposure must be a scalar or [prefix_trials] floating Tensor")
    if candidate.ndim == 0:
        candidate = candidate.expand(trial_count)
    if candidate.ndim != 1 or candidate.shape[0] != trial_count:
        raise PosteriorCarrierError("exposure must be scalar or exactly [prefix_trials]")
    if not torch.isfinite(candidate).all().item() or not bool((candidate > 0).all().item()):
        raise PosteriorCarrierError("exposure must be finite and strictly positive")
    return candidate


def directional_design(theta: Tensor) -> Tensor:
    """Return X[t]=[cos(theta_t), sin(theta_t), 1] in raw task coordinates."""
    _require_finite_floating(theta, "theta", ndim=1)
    return torch.stack((torch.cos(theta), torch.sin(theta), torch.ones_like(theta)), dim=-1)


def beta_to_raw_t4(beta: Tensor, zero_spike_mask: Tensor | None = None) -> Tensor:
    if not isinstance(beta, Tensor) or beta.ndim != 2 or beta.shape[-1] != 3 or not beta.is_floating_point():
        raise PosteriorCarrierError("beta must be finite floating [units,3]")
    if beta.shape[0] < 1 or not torch.isfinite(beta).all().item():
        raise PosteriorCarrierError("beta must be nonempty and finite")
    result = torch.stack(
        (beta[:, 0], beta[:, 1], torch.linalg.vector_norm(beta[:, :2], dim=-1), beta[:, 2]),
        dim=-1,
    )
    if zero_spike_mask is not None:
        if zero_spike_mask.dtype != torch.bool or zero_spike_mask.shape != (beta.shape[0],):
            raise PosteriorCarrierError("zero_spike_mask must be bool [units]")
        result = torch.where(zero_spike_mask[:, None], torch.zeros_like(result), result)
    return result


@dataclass(frozen=True)
class SourcePrior:
    """Closed source-only prior with an isotropic directional block."""

    source_mean_b: float
    tau_ac2: float
    tau_b2: float
    source_roster: tuple[str, ...]
    raw_m30_t4_sha256: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, (float, int)) and math.isfinite(float(value)) for value in
                   (self.source_mean_b, self.tau_ac2, self.tau_b2)):
            raise PosteriorCarrierError("source-prior moments must be finite scalar values")
        if float(self.tau_ac2) < PRIOR_VARIANCE_FLOOR or float(self.tau_b2) < PRIOR_VARIANCE_FLOOR:
            raise PosteriorCarrierError("source-prior variances must include the exact 1e-12 floor")
        if not isinstance(self.source_roster, tuple):
            raise PosteriorCarrierError("source_roster must be an immutable ordered tuple")
        _require_nonempty_unique_strings(self.source_roster, "source_roster")
        _require_sha(self.raw_m30_t4_sha256, "raw_m30_t4_sha256")

    @classmethod
    def from_raw_m30_t4(cls, raw_t4: Tensor, source_roster: Sequence[str]) -> "SourcePrior":
        """Use exactly the source-27 raw M30 moment formulas from the handoff."""
        _require_finite_floating(raw_t4, "raw_m30_t4", ndim=2)
        if raw_t4.shape[1] != 4:
            raise PosteriorCarrierError("raw_m30_t4 must have shape [units,4]")
        roster = _require_nonempty_unique_strings(source_roster, "source_roster")
        source_mean_b = float(raw_t4[:, 3].mean().item())
        tau_ac2 = max(float((raw_t4[:, 0].square() + raw_t4[:, 1].square()).mean().div(2).item()), PRIOR_VARIANCE_FLOOR)
        tau_b2 = max(float((raw_t4[:, 3] - source_mean_b).square().mean().item()), PRIOR_VARIANCE_FLOOR)
        return cls(
            source_mean_b=source_mean_b,
            tau_ac2=tau_ac2,
            tau_b2=tau_b2,
            source_roster=roster,
            raw_m30_t4_sha256=tensor_digest(raw_t4),
        )

    def mean_tensor(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.tensor((0.0, 0.0, float(self.source_mean_b)), device=device, dtype=dtype)

    def covariance_tensor(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.diag(torch.tensor((float(self.tau_ac2), float(self.tau_ac2), float(self.tau_b2)), device=device, dtype=dtype))

    def precision_tensor(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        # This is analytic for the diagonal S0.  It deliberately avoids a
        # second matrix inverse beyond the one posterior inverse per unit.
        return torch.diag(torch.tensor(
            (1.0 / float(self.tau_ac2), 1.0 / float(self.tau_ac2), 1.0 / float(self.tau_b2)),
            device=device,
            dtype=dtype,
        ))

    def payload(self) -> dict[str, object]:
        roster_digest = sha256_bytes(canonical_json_bytes(list(self.source_roster)))
        return {
            "schema": SOURCE_PRIOR_SCHEMA,
            "mu0": [0.0, 0.0, float(self.source_mean_b)],
            "source_mean_b": float(self.source_mean_b),
            "tau_ac2": float(self.tau_ac2),
            "tau_b2": float(self.tau_b2),
            "directional_prior_mean_exact_zero": True,
            "directional_prior_isotropic": True,
            "variance_floor": PRIOR_VARIANCE_FLOOR,
            "source_roster": list(self.source_roster),
            "source_roster_sha256": roster_digest,
            "raw_m30_t4_sha256": self.raw_m30_t4_sha256,
        }


@dataclass(frozen=True)
class PosteriorFitAudit:
    """Explicit proof that a vectorized batch represents one 3x3 inverse/unit."""

    units: int
    inverse_call_count: int
    inverse_count: int
    inverse_input_shape: tuple[int, int, int]
    design_rank: int
    design_condition_number: float
    count_precision_rule: str = "exposure_squared_over_max_integer_count_1"
    lambda_grid_or_optimizer_used: bool = False

    def __post_init__(self) -> None:
        if self.units < 1 or self.inverse_call_count != 1 or self.inverse_count != self.units:
            raise PosteriorCarrierError("posterior must use exactly one 3x3 inverse per unit")
        if self.inverse_input_shape != (self.units, 3, 3):
            raise PosteriorCarrierError("posterior inverse must be a batched [units,3,3] operation")
        if self.design_rank < 1 or not math.isfinite(self.design_condition_number) or self.design_condition_number <= 0:
            raise PosteriorCarrierError("posterior design audit is invalid")
        if self.lambda_grid_or_optimizer_used:
            raise PosteriorCarrierError("posterior may not use a lambda grid or optimizer")

    def payload(self) -> dict[str, object]:
        return {
            "units": self.units,
            "inverse_call_count": self.inverse_call_count,
            "inverse_count": self.inverse_count,
            "inverse_input_shape": list(self.inverse_input_shape),
            "design_rank": self.design_rank,
            "design_condition_number": self.design_condition_number,
            "count_precision_rule": self.count_precision_rule,
            "lambda_grid_or_optimizer_used": False,
        }


@dataclass(frozen=True)
class PosteriorCarrier:
    """Typed raw-coordinate posterior identity for one source session."""

    mean: Tensor  # [N,3]; zero-spike rows are deliberately emitted as zero.
    covariance: Tensor  # [N,3,3], retained even for all-zero-count units.
    raw_t4: Tensor  # [N,4]
    credibility: Tensor  # [N], based only on covariance[a,c;a,c].
    attention_bias: Tensor  # [N], exactly log(credibility).
    zero_spike_mask: Tensor  # bool [N]
    prior: SourcePrior
    design: Tensor  # [M,3]
    counts_sha256: str
    exposure_sha256: str
    theta_sha256: str
    audit: PosteriorFitAudit

    def __post_init__(self) -> None:
        _require_finite_floating(self.mean, "posterior mean", ndim=2)
        _require_finite_floating(self.covariance, "posterior covariance", ndim=3)
        _require_finite_floating(self.raw_t4, "raw posterior T4", ndim=2)
        _require_finite_floating(self.credibility, "credibility", ndim=1)
        _require_finite_floating(self.attention_bias, "attention bias", ndim=1)
        _require_finite_floating(self.design, "directional design", ndim=2)
        units = self.mean.shape[0]
        if self.mean.shape[1] != 3 or self.covariance.shape != (units, 3, 3) or self.raw_t4.shape != (units, 4):
            raise PosteriorCarrierError("posterior carrier shape drift")
        if self.credibility.shape != (units,) or self.attention_bias.shape != (units,):
            raise PosteriorCarrierError("posterior credibility shape drift")
        if self.zero_spike_mask.dtype != torch.bool or self.zero_spike_mask.shape != (units,):
            raise PosteriorCarrierError("zero_spike_mask must be bool [units]")
        if not torch.allclose(self.covariance, self.covariance.transpose(-1, -2), rtol=0.0, atol=1e-8):
            raise PosteriorCarrierError("posterior covariance must be symmetric")
        try:
            torch.linalg.cholesky(self.covariance)
        except RuntimeError as error:
            raise PosteriorCarrierError("posterior covariance must be positive definite") from error
        if not bool(((self.credibility >= CREDIBILITY_FLOOR) & (self.credibility <= 1.0)).all().item()):
            raise PosteriorCarrierError("credibility must lie in [1e-6,1]")
        if not torch.equal(self.attention_bias, torch.log(self.credibility)):
            raise PosteriorCarrierError("attention bias must be exactly log(credibility)")
        if self.design.shape[1] != 3:
            raise PosteriorCarrierError("directional design must be [prefix_trials,3]")
        if self.zero_spike_mask.any().item():
            zeros = torch.zeros_like(self.raw_t4[self.zero_spike_mask])
            if not torch.equal(self.raw_t4[self.zero_spike_mask], zeros):
                raise PosteriorCarrierError("all-zero-count units must emit exact zero raw T4")
            if not torch.equal(self.credibility[self.zero_spike_mask], torch.full_like(
                    self.credibility[self.zero_spike_mask], CREDIBILITY_FLOOR)):
                raise PosteriorCarrierError("all-zero-count credibility must stay at the clamp floor")
        for label, value in (
            ("counts_sha256", self.counts_sha256),
            ("exposure_sha256", self.exposure_sha256),
            ("theta_sha256", self.theta_sha256),
        ):
            _require_sha(value, label)

    @property
    def unit_count(self) -> int:
        return int(self.mean.shape[0])

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes({
            "schema": POSTERIOR_SCHEMA,
            "mean": tensor_digest(self.mean),
            "covariance": tensor_digest(self.covariance),
            "raw_t4": tensor_digest(self.raw_t4),
            "credibility": tensor_digest(self.credibility),
            "zero_spike_mask": tensor_digest(self.zero_spike_mask),
            "prior": self.prior.payload(),
            "design": tensor_digest(self.design),
            "counts_sha256": self.counts_sha256,
            "exposure_sha256": self.exposure_sha256,
            "theta_sha256": self.theta_sha256,
            "audit": self.audit.payload(),
        }))

    def payload(self) -> dict[str, object]:
        return {
            "schema": POSTERIOR_SCHEMA,
            "unit_count": self.unit_count,
            "posterior_sha256": self.digest(),
            "mean_sha256": tensor_digest(self.mean),
            "covariance_sha256": tensor_digest(self.covariance),
            "raw_t4_sha256": tensor_digest(self.raw_t4),
            "credibility_sha256": tensor_digest(self.credibility),
            "attention_bias_sha256": tensor_digest(self.attention_bias),
            "zero_spike_mask_sha256": tensor_digest(self.zero_spike_mask),
            "counts_sha256": self.counts_sha256,
            "exposure_sha256": self.exposure_sha256,
            "theta_sha256": self.theta_sha256,
            "prior": self.prior.payload(),
            "fit_audit": self.audit.payload(),
        }


def directional_credibility(covariance: Tensor, *, tau_ac2: float, zero_spike_mask: Tensor | None = None) -> Tensor:
    """Compute r from only the rotational [a,c] covariance block.

    The baseline posterior variance is intentionally absent.  B3S already
    carries rate information, so including b here would make attention trust
    mostly a high-rate preference rather than a tuning-identifiability signal.
    """
    _require_finite_floating(covariance, "posterior covariance", ndim=3)
    if covariance.shape[1:] != (3, 3) or not isinstance(tau_ac2, (float, int)) or not math.isfinite(float(tau_ac2)) or float(tau_ac2) < PRIOR_VARIANCE_FLOOR:
        raise PosteriorCarrierError("directional credibility requires [units,3,3] covariance and tau_ac2")
    trace_ac = covariance[:, 0, 0] + covariance[:, 1, 1]
    credibility = (1.0 - trace_ac / (2.0 * float(tau_ac2))).clamp(CREDIBILITY_FLOOR, 1.0)
    if zero_spike_mask is not None:
        if zero_spike_mask.dtype != torch.bool or zero_spike_mask.shape != (covariance.shape[0],):
            raise PosteriorCarrierError("zero_spike_mask must be bool [units] for credibility")
        credibility = torch.where(zero_spike_mask, torch.full_like(credibility, CREDIBILITY_FLOOR), credibility)
    return credibility


def fit_conjugate_posterior(
    *,
    counts: Tensor,
    exposure: Tensor | float,
    theta: Tensor,
    prior: SourcePrior,
    inverse_fn: Callable[[Tensor], Tensor] | None = None,
) -> PosteriorCarrier:
    """Fit the literal count/exposure conjugate posterior for one session.

    ``counts`` is a direct [unit, labelled-prefix-trial] count tensor.  The
    sole posterior inverse is a vectorized [N,3,3] ``torch.linalg.inv`` call:
    mathematically one 3x3 inverse for every unit, with no GCV, lambda search,
    or optimizer.  The diagonal prior precision is written analytically so it
    does not consume a second inverse.
    """
    direct_counts = _as_direct_integer_counts(counts)
    _require_finite_floating(theta, "theta", ndim=1)
    calc_dtype = torch.float64 if direct_counts.dtype == torch.float64 or theta.dtype == torch.float64 else torch.float32
    count_float = direct_counts.to(dtype=calc_dtype)
    if theta.shape[0] != count_float.shape[1] or theta.device != count_float.device:
        raise PosteriorCarrierError("theta must share count device and prefix-trial axis")
    theta_calc = theta.to(dtype=calc_dtype)
    exposures = _coerce_exposure(exposure, count_float.shape[1], device=count_float.device, dtype=calc_dtype)
    design = directional_design(theta_calc)
    if int(torch.linalg.matrix_rank(design).item()) < 3:
        raise PosteriorCarrierError("directional design must have rank 3")
    design_condition = float(torch.linalg.cond(design).item())
    if not math.isfinite(design_condition) or design_condition <= 0:
        raise PosteriorCarrierError("directional design condition number is invalid")

    # Literal Poisson-rate approximation: no rate rounding and no tuned floor.
    rates = count_float / exposures.unsqueeze(0)
    precision_weights = exposures.square().unsqueeze(0) / count_float.clamp_min(1.0)

    s0_precision = prior.precision_tensor(device=count_float.device, dtype=calc_dtype)
    mu0 = prior.mean_tensor(device=count_float.device, dtype=calc_dtype)
    weighted_design = design.unsqueeze(0) * precision_weights.unsqueeze(-1)
    posterior_precision = s0_precision.unsqueeze(0) + torch.matmul(design.transpose(0, 1).unsqueeze(0), weighted_design)
    rhs = torch.matmul(s0_precision, mu0).unsqueeze(0) + torch.matmul(
        design.transpose(0, 1).unsqueeze(0), (precision_weights * rates).unsqueeze(-1)
    ).squeeze(-1)
    inverse = torch.linalg.inv if inverse_fn is None else inverse_fn
    covariance = inverse(posterior_precision)
    if covariance.shape != posterior_precision.shape:
        raise PosteriorCarrierError("posterior inverse must preserve [units,3,3] shape")
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    mean = torch.matmul(covariance, rhs.unsqueeze(-1)).squeeze(-1)
    zero_spike_mask = direct_counts.sum(dim=1).eq(0)
    mean = torch.where(zero_spike_mask[:, None], torch.zeros_like(mean), mean)
    raw_t4 = beta_to_raw_t4(mean, zero_spike_mask)

    # Baseline b deliberately does not appear here.  Only posterior directional
    # uncertainty controls the set-attention credibility bias.
    credibility = directional_credibility(
        covariance,
        tau_ac2=float(prior.tau_ac2),
        zero_spike_mask=zero_spike_mask,
    )
    attention_bias = torch.log(credibility)
    audit = PosteriorFitAudit(
        units=int(count_float.shape[0]),
        inverse_call_count=1,
        inverse_count=int(count_float.shape[0]),
        inverse_input_shape=tuple(int(item) for item in posterior_precision.shape),
        design_rank=int(torch.linalg.matrix_rank(design).item()),
        design_condition_number=design_condition,
    )
    return PosteriorCarrier(
        mean=mean,
        covariance=covariance,
        raw_t4=raw_t4,
        credibility=credibility,
        attention_bias=attention_bias,
        zero_spike_mask=zero_spike_mask,
        prior=prior,
        design=design,
        counts_sha256=tensor_digest(direct_counts),
        exposure_sha256=tensor_digest(exposures),
        theta_sha256=tensor_digest(theta_calc),
        audit=audit,
    )


@dataclass(frozen=True)
class FrozenSourceT4Normalizer:
    """An explicit frozen raw-to-normalized T4 map supplied by a future authority."""

    mean: Tensor
    std: Tensor
    authority_sha256: str

    def __post_init__(self) -> None:
        _require_finite_floating(self.mean, "normalizer mean", ndim=1)
        _require_finite_floating(self.std, "normalizer std", ndim=1)
        if self.mean.shape != (4,) or self.std.shape != (4,) or self.mean.device != self.std.device:
            raise PosteriorCarrierError("T4 normalizer requires same-device [4] mean/std")
        if not bool((self.std > 0).all().item()):
            raise PosteriorCarrierError("T4 normalizer std must be strictly positive")
        _require_sha(self.authority_sha256, "normalizer authority")

    def normalize_raw(self, raw_t4: Tensor) -> Tensor:
        _require_finite_floating(raw_t4, "raw T4", ndim=2)
        if raw_t4.shape[1] != 4 or raw_t4.device != self.mean.device or raw_t4.dtype != self.mean.dtype:
            raise PosteriorCarrierError("raw T4 must be same-device/dtype [units,4] for frozen normalization")
        return (raw_t4 - self.mean) / self.std


@dataclass(frozen=True)
class PosteriorSourceT4Normalizer:
    """Frozen source-only normalizer for the *posterior* carrier distribution.

    The ordinary point-T4 moments are intentionally not reused.  The input is
    exactly the deterministic posterior-mean raw T4 for every strict source
    session at M4, M10, and M30.  Budget-major ordering gives each budget the
    same population weight and makes the raw-row binding unambiguous.
    """

    mean: Tensor
    std: Tensor
    source_roster: tuple[str, ...]
    row_order: str
    row_count: int
    per_budget_row_counts: Mapping[int, int]
    per_budget_raw_rows_sha256: Mapping[int, str]
    raw_rows_sha256: str
    body_sha256: str

    ROW_ORDER = "budget_major_M4_M10_M30_then_strict_roster_then_unit_index"

    def __post_init__(self) -> None:
        _require_finite_floating(self.mean, "posterior normalizer mean", ndim=1)
        _require_finite_floating(self.std, "posterior normalizer std", ndim=1)
        if self.mean.shape != (4,) or self.std.shape != (4,) or self.mean.dtype != torch.float64 or self.std.dtype != torch.float64:
            raise PosteriorCarrierError("posterior normalizer moments must be exact float64 [4]")
        if self.mean.device.type != "cpu" or self.std.device.type != "cpu":
            raise PosteriorCarrierError("posterior normalizer authority moments must be CPU float64")
        if not bool((self.std > 0).all().item()):
            raise PosteriorCarrierError("posterior normalizer standard deviations must be finite and strictly positive")
        if not isinstance(self.source_roster, tuple):
            raise PosteriorCarrierError("posterior normalizer roster must be immutable")
        _require_nonempty_unique_strings(self.source_roster, "posterior normalizer source roster")
        if self.row_order != self.ROW_ORDER or type(self.row_count) is not int or self.row_count < 1:
            raise PosteriorCarrierError("posterior normalizer row-order/count drift")
        if not isinstance(self.per_budget_row_counts, Mapping) or set(self.per_budget_row_counts) != set(BUDGETS):
            raise PosteriorCarrierError("posterior normalizer must bind exactly M4/M10/M30 row counts")
        counts = []
        for budget in BUDGETS:
            count = self.per_budget_row_counts[budget]
            if type(count) is not int or count < 1:
                raise PosteriorCarrierError("posterior normalizer per-budget count drift")
            counts.append(count)
        if len(set(counts)) != 1 or sum(counts) != self.row_count:
            raise PosteriorCarrierError("posterior normalizer budgets must have exactly equal source-row weight")
        if not isinstance(self.per_budget_raw_rows_sha256, Mapping) or set(self.per_budget_raw_rows_sha256) != set(BUDGETS):
            raise PosteriorCarrierError("posterior normalizer per-budget raw digest drift")
        for budget in BUDGETS:
            _require_sha(self.per_budget_raw_rows_sha256[budget], f"posterior normalizer M{budget} raw digest")
        _require_sha(self.raw_rows_sha256, "posterior normalizer raw-row digest")
        _require_sha(self.body_sha256, "posterior normalizer body SHA")
        if self.body_sha256 != sha256_bytes(canonical_json_bytes(self._body_payload())):
            raise PosteriorCarrierError("posterior normalizer body SHA drift")

    @classmethod
    def fit(
        cls,
        *,
        source_roster: Sequence[str],
        raw_mean_t4_by_budget: Mapping[int, Mapping[str, Tensor]],
    ) -> "PosteriorSourceT4Normalizer":
        """Fit float64 population moments from strict source posterior means only.

        ``raw_mean_t4_by_budget[M][session]`` must contain the deterministic
        posterior-mean T4 rows in the source session's original unit order.
        The routine does not synthesize a target row, resample, or infer a
        normalizer from model batches.
        """
        roster = _require_nonempty_unique_strings(source_roster, "posterior normalizer source roster")
        if not isinstance(raw_mean_t4_by_budget, Mapping) or set(raw_mean_t4_by_budget) != set(BUDGETS):
            raise PosteriorCarrierError("posterior normalizer requires exact M4/M10/M30 source mappings")
        ordered_by_budget: dict[int, Tensor] = {}
        per_budget_digest: dict[int, str] = {}
        per_budget_count: dict[int, int] = {}
        for budget in BUDGETS:
            by_session = raw_mean_t4_by_budget[budget]
            if not isinstance(by_session, Mapping) or set(by_session) != set(roster):
                raise PosteriorCarrierError("posterior normalizer source session roster/order drift")
            rows: list[Tensor] = []
            for session in roster:
                raw = by_session[session]
                _require_finite_floating(raw, f"posterior normalizer M{budget} raw {session}", ndim=2)
                if raw.shape[1] != 4 or raw.numel() == 0:
                    raise PosteriorCarrierError("posterior normalizer raw rows must be nonempty [units,4]")
                # Every moment is computed in float64, independent of the
                # eventual Cell-D float32 consumer dtype.
                rows.append(raw.detach().to(device="cpu", dtype=torch.float64).contiguous())
            concatenated = torch.cat(rows, dim=0)
            ordered_by_budget[budget] = concatenated
            per_budget_count[budget] = int(concatenated.shape[0])
            per_budget_digest[budget] = tensor_digest(concatenated)
        raw_rows = torch.cat([ordered_by_budget[budget] for budget in BUDGETS], dim=0)
        mean = raw_rows.mean(dim=0)
        std = raw_rows.sub(mean).square().mean(dim=0).sqrt()
        provisional = {
            "schema": POSTERIOR_SOURCE_NORMALIZER_SCHEMA,
            "source_roster": list(roster),
            "source_roster_sha256": sha256_bytes(canonical_json_bytes(list(roster))),
            "row_order": cls.ROW_ORDER,
            "row_count": int(raw_rows.shape[0]),
            "per_budget_row_counts": {str(budget): per_budget_count[budget] for budget in BUDGETS},
            "per_budget_raw_rows_sha256": {str(budget): per_budget_digest[budget] for budget in BUDGETS},
            "raw_rows_sha256": tensor_digest(raw_rows),
            "mean_float64": [float(value) for value in mean.tolist()],
            "std_float64": [float(value) for value in std.tolist()],
            "ddof": 0,
            "source_only": True,
            "contains_only_deterministic_posterior_means": True,
            "all_zero_raw_rows_retained": True,
        }
        return cls(
            mean=mean,
            std=std,
            source_roster=roster,
            row_order=cls.ROW_ORDER,
            row_count=int(raw_rows.shape[0]),
            per_budget_row_counts=per_budget_count,
            per_budget_raw_rows_sha256=per_budget_digest,
            raw_rows_sha256=tensor_digest(raw_rows),
            body_sha256=sha256_bytes(canonical_json_bytes(provisional)),
        )

    def _body_payload(self) -> dict[str, object]:
        return {
            "schema": POSTERIOR_SOURCE_NORMALIZER_SCHEMA,
            "source_roster": list(self.source_roster),
            "source_roster_sha256": sha256_bytes(canonical_json_bytes(list(self.source_roster))),
            "row_order": self.row_order,
            "row_count": self.row_count,
            "per_budget_row_counts": {str(budget): self.per_budget_row_counts[budget] for budget in BUDGETS},
            "per_budget_raw_rows_sha256": {str(budget): self.per_budget_raw_rows_sha256[budget] for budget in BUDGETS},
            "raw_rows_sha256": self.raw_rows_sha256,
            "mean_float64": [float(value) for value in self.mean.tolist()],
            "std_float64": [float(value) for value in self.std.tolist()],
            "ddof": 0,
            "source_only": True,
            "contains_only_deterministic_posterior_means": True,
            "all_zero_raw_rows_retained": True,
        }

    @property
    def authority_sha256(self) -> str:
        return self.body_sha256

    def payload(self) -> dict[str, object]:
        return {**self._body_payload(), "body_sha256": self.body_sha256}

    def normalize_raw(self, raw_t4: Tensor) -> Tensor:
        _require_finite_floating(raw_t4, "posterior normalizer raw T4", ndim=2)
        if raw_t4.shape[1] != 4:
            raise PosteriorCarrierError("posterior normalizer raw T4 must be [units,4]")
        mean = self.mean.to(device=raw_t4.device, dtype=raw_t4.dtype)
        std = self.std.to(device=raw_t4.device, dtype=raw_t4.dtype)
        return (raw_t4 - mean) / std


SourceT4Normalizer = FrozenSourceT4Normalizer | PosteriorSourceT4Normalizer


@dataclass(frozen=True)
class PosteriorCarrierView:
    """Model-visible carrier that preserves raw-before-normalization lineage."""

    raw_beta: Tensor
    raw_t4: Tensor
    normalized_t4: Tensor
    credibility: Tensor
    zero_spike_mask: Tensor
    sampled: bool
    session_id: str | None
    epoch: int | None
    posterior_sha256: str
    normalizer_authority_sha256: str

    def __post_init__(self) -> None:
        _require_finite_floating(self.raw_beta, "carrier raw beta", ndim=2)
        _require_finite_floating(self.raw_t4, "carrier raw T4", ndim=2)
        _require_finite_floating(self.normalized_t4, "carrier normalized T4", ndim=2)
        _require_finite_floating(self.credibility, "carrier credibility", ndim=1)
        units = self.raw_beta.shape[0]
        if self.raw_beta.shape != (units, 3) or self.raw_t4.shape != (units, 4) or self.normalized_t4.shape != (units, 4):
            raise PosteriorCarrierError("posterior carrier view shape drift")
        if self.credibility.shape != (units,) or self.zero_spike_mask.dtype != torch.bool or self.zero_spike_mask.shape != (units,):
            raise PosteriorCarrierError("posterior carrier view unit-axis drift")
        if type(self.sampled) is not bool:
            raise PosteriorCarrierError("carrier sampled flag must be exact bool")
        if self.sampled:
            if not isinstance(self.session_id, str) or not self.session_id or type(self.epoch) is not int or self.epoch < 0:
                raise PosteriorCarrierError("sampled carrier must bind session and nonnegative epoch")
        elif self.session_id is not None or self.epoch is not None:
            raise PosteriorCarrierError("posterior-mean eval carrier must not claim a sampling key")
        _require_sha(self.posterior_sha256, "posterior SHA")
        _require_sha(self.normalizer_authority_sha256, "normalizer authority")
        if self.zero_spike_mask.any().item() and not torch.equal(
                self.raw_t4[self.zero_spike_mask], torch.zeros_like(self.raw_t4[self.zero_spike_mask])):
            raise PosteriorCarrierError("zero-spike carrier samples must emit exact raw T4 zeros")

    @property
    def unit_count(self) -> int:
        return int(self.raw_t4.shape[0])

    def normalized_for_batch(self, batch_size: int) -> Tensor:
        if type(batch_size) is not int or batch_size < 1:
            raise PosteriorCarrierError("batch_size must be a positive exact int")
        return self.normalized_t4.unsqueeze(0).expand(batch_size, -1, -1)

    def joint_permute(self, permutation: Tensor) -> "PosteriorCarrierView":
        """Permute every unit-indexed carrier field with a neural unit axis."""
        if permutation.ndim != 1 or permutation.dtype != torch.long or permutation.shape[0] != self.unit_count:
            raise PosteriorCarrierError("carrier permutation must be torch.long [units]")
        if permutation.device != self.raw_t4.device:
            raise PosteriorCarrierError("carrier permutation must share the carrier device")
        expected = torch.arange(self.unit_count, device=permutation.device, dtype=torch.long)
        if not torch.equal(torch.sort(permutation).values, expected):
            raise PosteriorCarrierError("carrier permutation must contain every unit exactly once")
        return PosteriorCarrierView(
            raw_beta=self.raw_beta.index_select(0, permutation).detach().clone(),
            raw_t4=self.raw_t4.index_select(0, permutation).detach().clone(),
            normalized_t4=self.normalized_t4.index_select(0, permutation).detach().clone(),
            credibility=self.credibility.index_select(0, permutation).detach().clone(),
            zero_spike_mask=self.zero_spike_mask.index_select(0, permutation).detach().clone(),
            sampled=self.sampled,
            session_id=self.session_id,
            epoch=self.epoch,
            posterior_sha256=self.posterior_sha256,
            normalizer_authority_sha256=self.normalizer_authority_sha256,
        )

    def clone(self) -> "PosteriorCarrierView":
        return PosteriorCarrierView(
            raw_beta=self.raw_beta.detach().clone(),
            raw_t4=self.raw_t4.detach().clone(),
            normalized_t4=self.normalized_t4.detach().clone(),
            credibility=self.credibility.detach().clone(),
            zero_spike_mask=self.zero_spike_mask.detach().clone(),
            sampled=self.sampled,
            session_id=self.session_id,
            epoch=self.epoch,
            posterior_sha256=self.posterior_sha256,
            normalizer_authority_sha256=self.normalizer_authority_sha256,
        )


def posterior_mean_view(carrier: PosteriorCarrier, normalizer: SourceT4Normalizer) -> PosteriorCarrierView:
    """Evaluation path: deterministic posterior mean, with sampling disabled."""
    if carrier.mean.device != normalizer.mean.device or carrier.mean.dtype != normalizer.mean.dtype:
        raise PosteriorCarrierError("carrier and frozen normalizer must share device/dtype")
    raw_t4 = beta_to_raw_t4(carrier.mean, carrier.zero_spike_mask)
    return PosteriorCarrierView(
        raw_beta=carrier.mean.detach().clone(),
        raw_t4=raw_t4.detach().clone(),
        normalized_t4=normalizer.normalize_raw(raw_t4).detach().clone(),
        credibility=carrier.credibility.detach().clone(),
        zero_spike_mask=carrier.zero_spike_mask.detach().clone(),
        sampled=False,
        session_id=None,
        epoch=None,
        posterior_sha256=carrier.digest(),
        normalizer_authority_sha256=normalizer.authority_sha256,
    )


def route_local_seed(*, cell: str, seed: int, session_id: str, epoch: int) -> int:
    if cell != CELL or type(seed) is not int or seed < 0 or not isinstance(session_id, str) or not session_id or type(epoch) is not int or epoch < 0:
        raise PosteriorCarrierError("route-local sampling key is malformed")
    # Canonical JSON prevents accidental string-concatenation collisions.
    body = canonical_json_bytes({"domain": "posterior-carrier-sample-v1", "cell": cell, "seed": seed,
                                 "session_id": session_id, "epoch": epoch})
    return int.from_bytes(hashlib.sha256(body).digest()[:8], byteorder="big", signed=False) & ((1 << 63) - 1)


def _sample_view(
    carrier: PosteriorCarrier,
    normalizer: SourceT4Normalizer,
    *,
    session_id: str,
    epoch: int,
    seed: int,
) -> PosteriorCarrierView:
    if carrier.mean.device != normalizer.mean.device or carrier.mean.dtype != normalizer.mean.dtype:
        raise PosteriorCarrierError("carrier and frozen normalizer must share device/dtype")
    local_generator = torch.Generator(device=carrier.mean.device)
    local_generator.manual_seed(route_local_seed(cell=CELL, seed=seed, session_id=session_id, epoch=epoch))
    # Cholesky consumes no RNG and preserves the retained covariance audit for
    # all-zero-count rows.  Their emitted carrier is then explicitly zeroed.
    factor = torch.linalg.cholesky(carrier.covariance)
    epsilon = torch.randn(carrier.mean.shape, generator=local_generator, device=carrier.mean.device, dtype=carrier.mean.dtype)
    sampled_beta = carrier.mean + torch.matmul(factor, epsilon.unsqueeze(-1)).squeeze(-1)
    sampled_beta = torch.where(carrier.zero_spike_mask[:, None], torch.zeros_like(sampled_beta), sampled_beta)
    raw_t4 = beta_to_raw_t4(sampled_beta, carrier.zero_spike_mask)
    return PosteriorCarrierView(
        raw_beta=sampled_beta.detach().clone(),
        raw_t4=raw_t4.detach().clone(),
        normalized_t4=normalizer.normalize_raw(raw_t4).detach().clone(),
        credibility=carrier.credibility.detach().clone(),
        zero_spike_mask=carrier.zero_spike_mask.detach().clone(),
        sampled=True,
        session_id=session_id,
        epoch=epoch,
        posterior_sha256=carrier.digest(),
        normalizer_authority_sha256=normalizer.authority_sha256,
    )


class SessionStaticPosteriorSampler:
    """Route-local training sampler: one immutable draw per session/epoch."""

    def __init__(self, *, seed: int, normalizer: SourceT4Normalizer, cell: str = CELL) -> None:
        if cell != CELL or type(seed) is not int or seed < 0:
            raise PosteriorCarrierError("sampler requires the exact cell and a nonnegative integer seed")
        self._seed = seed
        self._normalizer = normalizer
        self._cache: dict[tuple[str, int], PosteriorCarrierView] = {}

    @property
    def cached_keys(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self._cache))

    def carrier_for(
        self,
        carrier: PosteriorCarrier,
        *,
        session_id: str,
        epoch: int,
        training: bool,
    ) -> PosteriorCarrierView:
        if type(training) is not bool:
            raise PosteriorCarrierError("training must be an exact bool")
        if not training:
            # Evaluation never invokes a random API or mutates the cache.
            return posterior_mean_view(carrier, self._normalizer)
        if not isinstance(session_id, str) or not session_id or type(epoch) is not int or epoch < 0:
            raise PosteriorCarrierError("session-static sample key is malformed")
        key = (session_id, epoch)
        cached = self._cache.get(key)
        if cached is not None:
            if cached.posterior_sha256 != carrier.digest():
                raise PosteriorCarrierError("same session/epoch cannot silently reuse a different posterior")
            return cached.clone()
        view = _sample_view(carrier, self._normalizer, session_id=session_id, epoch=epoch, seed=self._seed)
        self._cache[key] = view.clone()
        return view


def host_rng_fingerprint() -> dict[str, str | bool]:
    """Audit host RNGs without calling any CUDA API or mutating a RNG state."""
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - the test environment has NumPy.
        raise PosteriorCarrierError("NumPy is required to audit its global RNG state") from error
    return {
        "python_random_sha256": sha256_bytes(pickle.dumps(random.getstate(), protocol=5)),
        "numpy_random_sha256": sha256_bytes(pickle.dumps(np.random.get_state(), protocol=5)),
        "torch_cpu_random_sha256": sha256_bytes(torch.get_rng_state().cpu().numpy().tobytes()),
        # This module never calls torch.cuda or a CUDA global RNG function.
        "cuda_rng_api_reached": False,
    }


def assert_host_rng_unchanged(before: Mapping[str, str | bool], after: Mapping[str, str | bool]) -> None:
    if dict(before) != dict(after):
        raise PosteriorCarrierError("route-local posterior sampling mutated a host global RNG")


def budget_for_epoch(epoch: int, session_index: int) -> int:
    if type(epoch) is not int or epoch < 0 or type(session_index) is not int or session_index < 0:
        raise PosteriorCarrierError("epoch and session_index must be nonnegative exact ints")
    return BUDGETS[(epoch + session_index) % len(BUDGETS)]


def build_budget_schedule(*, epochs: int, session_count: int) -> tuple[tuple[int, ...], ...]:
    if type(epochs) is not int or epochs < 1 or type(session_count) is not int or session_count < 1:
        raise PosteriorCarrierError("schedule epochs/session_count must be positive exact ints")
    return tuple(tuple(budget_for_epoch(epoch, session_index) for session_index in range(session_count)) for epoch in range(epochs))


def budget_schedule_digest(schedule: Sequence[Sequence[int]]) -> str:
    rows = [list(row) for row in schedule]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) or any(value not in BUDGETS for value in row) for row in rows):
        raise PosteriorCarrierError("budget schedule must be rectangular with only M4/M10/M30")
    return sha256_bytes(canonical_json_bytes(rows))


def verify_balanced_48_epoch_schedule(schedule: Sequence[Sequence[int]], *, session_count: int) -> None:
    if len(schedule) != 48 or session_count < 1 or any(len(row) != session_count for row in schedule):
        raise PosteriorCarrierError("headline schedule must be exactly 48 epochs by source-session count")
    for session_index in range(session_count):
        actual = [row[session_index] for row in schedule]
        expected = [budget_for_epoch(epoch, session_index) for epoch in range(48)]
        if actual != expected or any(actual.count(budget) != 16 for budget in BUDGETS):
            raise PosteriorCarrierError("budget schedule must give each session exactly 16 epochs at M4/M10/M30")


def centered_attention_bias(
    credibility: Tensor,
    *,
    batch_size: int,
    num_heads: int,
    query_count: int,
) -> Tensor:
    """Return MHA-compatible [B*H,Q,N] additive logits with exact cancellation.

    The stored scientific quantity remains ``log(r_i)``.  Subtracting one
    unit's shared constant is softmax-equivalent and makes equal credibility
    produce an exact all-zero mask rather than merely a numerically-close one.
    """
    _require_finite_floating(credibility, "credibility")
    if credibility.ndim == 1:
        credibility = credibility.unsqueeze(0).expand(batch_size, -1)
    if credibility.ndim != 2 or credibility.shape[0] != batch_size:
        raise PosteriorCarrierError("credibility must be [units] or [batch,units]")
    if type(batch_size) is not int or batch_size < 1 or type(num_heads) is not int or num_heads < 1 or type(query_count) is not int or query_count < 1:
        raise PosteriorCarrierError("attention dimensions must be positive exact ints")
    if not bool(((credibility >= CREDIBILITY_FLOOR) & (credibility <= 1.0)).all().item()):
        raise PosteriorCarrierError("attention credibility lies outside [1e-6,1]")
    raw = torch.log(credibility)
    centered = raw - raw[:, :1]
    return centered[:, None, :].expand(-1, query_count, -1).repeat_interleave(num_heads, dim=0)


def add_credibility_to_logits(logits: Tensor, credibility: Tensor) -> Tensor:
    """Reference attention operation for synthetic cancellation/monotonicity gates."""
    _require_finite_floating(logits, "attention logits", ndim=4)
    batch_size, num_heads, query_count, units = logits.shape
    bias = centered_attention_bias(
        credibility,
        batch_size=batch_size,
        num_heads=num_heads,
        query_count=query_count,
    ).view(batch_size, num_heads, query_count, units)
    return logits + bias


def _lazy_safe_parameter_signature(module: nn.Module) -> tuple[int, frozenset[int], tuple[str, ...]]:
    from torch.nn.parameter import UninitializedParameter

    live_count = 0
    ids: set[int] = set()
    lazy: list[str] = []
    for name, parameter in module.named_parameters():
        ids.add(id(parameter))
        if isinstance(parameter, UninitializedParameter):
            lazy.append(name)
        else:
            live_count += parameter.numel()
    return live_count, frozenset(ids), tuple(sorted(lazy))


@dataclass(frozen=True)
class CellDPreservationAudit:
    base_live_parameter_count: int
    wrapper_live_parameter_count: int
    wrapper_new_parameter_count: int
    parameter_object_ids_identical: bool
    base_lazy_parameter_names: tuple[str, ...]
    wrapper_lazy_parameter_names: tuple[str, ...]
    dynamic_dropout: bool
    dropout_low: float
    dropout_high: float
    dropout_mask_semantics: str

    def __post_init__(self) -> None:
        if (self.base_live_parameter_count != self.wrapper_live_parameter_count
                or self.wrapper_new_parameter_count != 0
                or not self.parameter_object_ids_identical
                or not self.dynamic_dropout
                or self.dropout_low != 0.0
                or self.dropout_high != 1.0):
            raise PosteriorCarrierError("Cell-D parameter or dynamic-dropout preservation drift")

    def payload(self) -> dict[str, object]:
        return {
            "base_live_parameter_count": self.base_live_parameter_count,
            "wrapper_live_parameter_count": self.wrapper_live_parameter_count,
            "wrapper_new_parameter_count": self.wrapper_new_parameter_count,
            "parameter_object_ids_identical": self.parameter_object_ids_identical,
            "base_lazy_parameter_names": list(self.base_lazy_parameter_names),
            "wrapper_lazy_parameter_names": list(self.wrapper_lazy_parameter_names),
            "dynamic_dropout": self.dynamic_dropout,
            "dropout_low": self.dropout_low,
            "dropout_high": self.dropout_high,
            "dropout_mask_semantics": self.dropout_mask_semantics,
        }


class CellDPosteriorWrapper(nn.Module):
    """Additive Cell-D consumer with a parameter-free unit-attention bias.

    It owns no learned layers.  The supplied coupled Cell-D graph remains the
    only parameter-bearing submodule.  For uniform credibility, the exact
    sealed ``decode_with_identity`` route is delegated directly.  Otherwise
    the wrapper reproduces only that route's coupled-token/dropout sequence and
    adds the mathematically softmax-equivalent centered log-credibility mask to
    the existing unit-axis cross-attention call.
    """

    def __init__(self, cell_d: nn.Module) -> None:
        super().__init__()
        self.cell_d = cell_d
        self._validate_cell_d_surface()

    def _validate_cell_d_surface(self) -> None:
        base = self.cell_d
        decoder = getattr(base, "decoder", None)
        required = (
            "compute_identity", "decode_with_identity", "decoder_mode", "fixed_slot_router",
            "live_activity_gain", "_decoder_frozen",
        )
        if any(not hasattr(base, name) for name in required) or decoder is None:
            raise PosteriorCarrierError("wrapper requires the coupled Cell-D StreamingSpint surface")
        if base.decoder_mode != "coupled" or base.fixed_slot_router is not None or base.live_activity_gain is not None:
            raise PosteriorCarrierError("posterior wrapper only supports the sealed coupled Cell-D route")
        decoder_required = (
            "dynamic_dropout", "dynamic_dropout_low", "dynamic_dropout_high", "dropout_rate", "fc_in",
            "rep", "transformer", "fc_out", "num_heads", "num_covariates", "num_layers",
        )
        if any(not hasattr(decoder, name) for name in decoder_required) or decoder.num_layers != 1:
            raise PosteriorCarrierError("posterior wrapper requires the one-layer sealed Cell-D decoder surface")
        if not bool(decoder.dynamic_dropout) or float(decoder.dynamic_dropout_low) != 0.0 or float(decoder.dynamic_dropout_high) != 1.0:
            raise PosteriorCarrierError("posterior wrapper requires Cell-D U(0,1) dynamic whole-unit dropout")

    def preservation_audit(self) -> CellDPreservationAudit:
        base_count, base_ids, base_lazy = _lazy_safe_parameter_signature(self.cell_d)
        wrapper_count, wrapper_ids, wrapper_lazy_prefixed = _lazy_safe_parameter_signature(self)
        wrapper_lazy = tuple(name.removeprefix("cell_d.") for name in wrapper_lazy_prefixed)
        new_count = len(wrapper_ids - base_ids)
        decoder = self.cell_d.decoder
        return CellDPreservationAudit(
            base_live_parameter_count=base_count,
            wrapper_live_parameter_count=wrapper_count,
            wrapper_new_parameter_count=new_count,
            parameter_object_ids_identical=base_ids == wrapper_ids,
            base_lazy_parameter_names=base_lazy,
            wrapper_lazy_parameter_names=wrapper_lazy,
            dynamic_dropout=bool(decoder.dynamic_dropout),
            dropout_low=float(decoder.dynamic_dropout_low),
            dropout_high=float(decoder.dynamic_dropout_high),
            dropout_mask_semantics="existing_CellD_zero_placeholder_with_inverse_probability_gain_after_complete_fused_token",
        )

    def b3s_m30_activity_prefix(self, calib_trials_m30: Tensor) -> Tensor:
        """Expose the side-feature-independent B3S M30 activity summary for audit."""
        encoder = getattr(self.cell_d, "id_encoder", None)
        pre_pool = getattr(encoder, "pre_pool", None)
        if pre_pool is None:
            raise PosteriorCarrierError("Cell-D B3S pre_pool surface unavailable")
        _require_finite_floating(calib_trials_m30, "M30 B3S calibration trials", ndim=4)
        if calib_trials_m30.shape[1] != 30 or calib_trials_m30.shape[2] != 100:
            raise PosteriorCarrierError("held B3S activity prefix must remain exact M30 x 100")
        # This is the exact side-independent part of B3S's push-trial / mean
        # accumulation: [B,M,T,N] -> [B,M,N,T] -> pre_pool -> mean(M).
        return pre_pool(calib_trials_m30.permute(0, 1, 3, 2)).mean(dim=1)

    @staticmethod
    def _uniform_credibility(credibility: Tensor) -> bool:
        return bool(torch.equal(credibility, credibility[:1].expand_as(credibility)))

    def _decode_with_credibility_bias(self, neural: Tensor, identity: Tensor, credibility: Tensor) -> Tensor:
        """Exact coupled Cell-D flow plus only the additive unit-logit bias."""
        base = self.cell_d
        decoder = base.decoder
        src = neural.permute(0, 2, 1)
        if identity.shape != src.shape:
            raise PosteriorCarrierError("Cell-D identity/neural fused-token shape drift")
        src = src + identity
        if base._decoder_frozen:
            decoder.eval()
        batch_size, unit_count = src.shape[:2]
        gain_mask = torch.ones(batch_size, unit_count, device=src.device, dtype=src.dtype)
        if not base._decoder_frozen:
            if decoder.dynamic_dropout and base.training:
                # Same one Python draw and same [B,N] F.dropout placeholder
                # law as StreamingSpintModel.decode_with_identity.
                probability = random.uniform(decoder.dynamic_dropout_low, decoder.dynamic_dropout_high)
                gain_mask = F.dropout(gain_mask, p=probability, training=True)
            elif decoder.dropout_rate > 0.0 and base.training:
                gain_mask = F.dropout(gain_mask, p=decoder.dropout_rate, training=True)
        src = src * gain_mask.unsqueeze(-1)
        src = decoder.fc_in(src)
        query = decoder.fc_in(decoder.rep).to(src).repeat(batch_size, 1, 1)
        attn_mask = centered_attention_bias(
            credibility.to(device=src.device, dtype=src.dtype),
            batch_size=batch_size,
            num_heads=int(decoder.num_heads),
            query_count=int(decoder.num_covariates),
        )
        transformer_output, _ = decoder.transformer(query, src, attn_mask=attn_mask)
        return decoder.fc_out(transformer_output).permute(0, 2, 1)

    def forward(
        self,
        neural: Tensor,
        *,
        calib_trials_m30: Tensor,
        carrier: PosteriorCarrierView,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(carrier, PosteriorCarrierView):
            raise PosteriorCarrierError("Cell-D posterior wrapper requires a typed normalized posterior carrier")
        _require_finite_floating(neural, "Cell-D neural window", ndim=3)
        if neural.shape[1] != 50 or neural.shape[2] != carrier.unit_count:
            raise PosteriorCarrierError("Cell-D neural/carrier unit or window axis drift")
        if (calib_trials_m30.shape[0] != neural.shape[0] or calib_trials_m30.shape[3] != neural.shape[2]
                or calib_trials_m30.device != neural.device or calib_trials_m30.dtype != neural.dtype):
            raise PosteriorCarrierError("held M30 B3S calibration axis/device/dtype drift")
        side_features = carrier.normalized_for_batch(neural.shape[0]).to(device=neural.device, dtype=neural.dtype)
        identity = self.cell_d.compute_identity(calib_trials_m30, side_features=side_features)
        if self._uniform_credibility(carrier.credibility):
            # Exact semantic and RNG preservation: log(r) is a shared additive
            # constant, so it cancels and the sealed path is literally reused.
            behavior = self.cell_d.decode_with_identity(neural, identity)
        else:
            behavior = self._decode_with_credibility_bias(neural, identity, carrier.credibility)
        return behavior, identity


def stage0_closure(root: Path) -> dict[str, object]:
    """Hash an explicit Stage-0 closure; this is read-only and data-free."""
    root = root.absolute()
    hashes: dict[str, str] = {}
    for relative in STAGE0_CLOSURE_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise PosteriorCarrierError(f"Stage-0 closure path missing or aliased: {relative}")
        hashes[relative] = sha256_bytes(path.read_bytes())
    if hashes[HANDOFF_RELATIVE] != HANDOFF_SHA256:
        raise PosteriorCarrierError("posterior-carrier handoff SHA drift")
    body = canonical_json_bytes({"paths": list(STAGE0_CLOSURE_PATHS), "sha256_by_path": hashes})
    return {"paths": list(STAGE0_CLOSURE_PATHS), "sha256_by_path": hashes, "closure_sha256": sha256_bytes(body)}


def _validate_closure(closure: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(closure, Mapping) or set(closure) != {"paths", "sha256_by_path", "closure_sha256"}:
        raise PosteriorCarrierError("Stage-0 closure schema drift")
    paths = closure.get("paths")
    hashes = closure.get("sha256_by_path")
    if paths != list(STAGE0_CLOSURE_PATHS) or not isinstance(hashes, Mapping) or set(hashes) != set(STAGE0_CLOSURE_PATHS):
        raise PosteriorCarrierError("Stage-0 closure paths drift")
    if any(not _require_sha(hashes[path], f"closure SHA {path}") for path in STAGE0_CLOSURE_PATHS):
        raise PosteriorCarrierError("unreachable")
    if hashes[HANDOFF_RELATIVE] != HANDOFF_SHA256:
        raise PosteriorCarrierError("Stage-0 closure handoff binding drift")
    expected = sha256_bytes(canonical_json_bytes({"paths": list(STAGE0_CLOSURE_PATHS), "sha256_by_path": dict(hashes)}))
    if closure.get("closure_sha256") != expected:
        raise PosteriorCarrierError("Stage-0 closure aggregate drift")
    return {"paths": list(paths), "sha256_by_path": dict(hashes), "closure_sha256": expected}


def build_source_prior_receipt(
    *,
    prior: SourcePrior,
    source_roster: Sequence[str],
    raw_m30_prefix_rows_sha256: str,
    closure: Mapping[str, object],
) -> dict[str, object]:
    roster = _require_nonempty_unique_strings(source_roster, "source_roster")
    if roster != prior.source_roster:
        raise PosteriorCarrierError("source-prior receipt roster must exactly match prior roster")
    _require_sha(raw_m30_prefix_rows_sha256, "raw_m30_prefix_rows_sha256")
    stable_closure = _validate_closure(closure)
    return {
        "schema": SOURCE_PRIOR_RECEIPT_SCHEMA,
        "cell": CELL,
        "phase": PHASE,
        "source_only": True,
        "source_roster": list(roster),
        "source_roster_sha256": sha256_bytes(canonical_json_bytes(list(roster))),
        "raw_m30_t4_sha256": prior.raw_m30_t4_sha256,
        "raw_m30_prefix_rows_sha256": raw_m30_prefix_rows_sha256,
        "prior": prior.payload(),
        "closure": stable_closure,
    }


def validate_source_prior_receipt(value: Mapping[str, object], *, prior: SourcePrior, closure: Mapping[str, object]) -> None:
    expected = {"schema", "cell", "phase", "source_only", "source_roster", "source_roster_sha256", "raw_m30_t4_sha256",
                "raw_m30_prefix_rows_sha256", "prior", "closure"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PosteriorCarrierError("source-prior receipt schema drift")
    if (value.get("schema") != SOURCE_PRIOR_RECEIPT_SCHEMA or value.get("cell") != CELL or value.get("phase") != PHASE
            or value.get("source_only") is not True or value.get("source_roster") != list(prior.source_roster)
            or value.get("source_roster_sha256") != sha256_bytes(canonical_json_bytes(list(prior.source_roster)))
            or value.get("raw_m30_t4_sha256") != prior.raw_m30_t4_sha256 or value.get("prior") != prior.payload()
            or value.get("closure") != _validate_closure(closure)):
        raise PosteriorCarrierError("source-prior receipt binding drift")
    _require_sha(value.get("raw_m30_prefix_rows_sha256"), "raw_m30_prefix_rows_sha256")


def build_session_posterior_receipt(
    *,
    session_id: str,
    session_index: int,
    epoch: int,
    schedule: Sequence[Sequence[int]],
    prefix_row_ids: Sequence[str],
    carrier: PosteriorCarrier,
    source_prior_receipt: Mapping[str, object],
    closure: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(session_id, str) or not session_id or type(session_index) is not int or session_index < 0 or type(epoch) is not int or epoch < 0:
        raise PosteriorCarrierError("session posterior receipt session binding drift")
    prefix_rows = _require_nonempty_unique_strings(prefix_row_ids, "prefix_row_ids")
    if len(schedule) <= epoch or not schedule or session_index >= len(schedule[0]):
        raise PosteriorCarrierError("session posterior receipt schedule bounds drift")
    schedule_digest = budget_schedule_digest(schedule)
    budget = budget_for_epoch(epoch, session_index)
    if schedule[epoch][session_index] != budget or len(prefix_rows) != budget:
        raise PosteriorCarrierError("session posterior receipt exact budget/prefix binding drift")
    stable_closure = _validate_closure(closure)
    validate_source_prior_receipt(source_prior_receipt, prior=carrier.prior, closure=closure)
    return {
        "schema": SESSION_POSTERIOR_RECEIPT_SCHEMA,
        "cell": CELL,
        "phase": PHASE,
        "source_only": True,
        "session_id": session_id,
        "session_index": session_index,
        "epoch": epoch,
        "carrier_budget": budget,
        "prefix_row_ids": list(prefix_rows),
        "prefix_rows_sha256": sha256_bytes(canonical_json_bytes(list(prefix_rows))),
        "budget_schedule_sha256": schedule_digest,
        "design_rank": carrier.audit.design_rank,
        "design_condition_number": carrier.audit.design_condition_number,
        "posterior": carrier.payload(),
        "source_prior_receipt_sha256": sha256_bytes(canonical_json_bytes(source_prior_receipt)),
        "closure": stable_closure,
    }


def validate_session_posterior_receipt(
    value: Mapping[str, object],
    *,
    carrier: PosteriorCarrier,
    schedule: Sequence[Sequence[int]],
    source_prior_receipt: Mapping[str, object],
    closure: Mapping[str, object],
) -> None:
    expected = {"schema", "cell", "phase", "source_only", "session_id", "session_index", "epoch", "carrier_budget",
                "prefix_row_ids", "prefix_rows_sha256", "budget_schedule_sha256", "design_rank", "design_condition_number",
                "posterior", "source_prior_receipt_sha256", "closure"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PosteriorCarrierError("session-posterior receipt schema drift")
    session_index, epoch = value.get("session_index"), value.get("epoch")
    if (not isinstance(value.get("session_id"), str) or not value.get("session_id") or type(session_index) is not int
            or session_index < 0 or type(epoch) is not int or epoch < 0):
        raise PosteriorCarrierError("session-posterior receipt session identity drift")
    rows = value.get("prefix_row_ids")
    if not isinstance(rows, list) or tuple(rows) != _require_nonempty_unique_strings(rows, "prefix_row_ids"):
        raise PosteriorCarrierError("session-posterior receipt prefix row schema drift")
    budget = budget_for_epoch(epoch, session_index)
    if (len(schedule) <= epoch or session_index >= len(schedule[0]) or schedule[epoch][session_index] != budget
            or value.get("carrier_budget") != budget or len(rows) != budget
            or value.get("prefix_rows_sha256") != sha256_bytes(canonical_json_bytes(rows))
            or value.get("budget_schedule_sha256") != budget_schedule_digest(schedule)
            or value.get("design_rank") != carrier.audit.design_rank
            or value.get("design_condition_number") != carrier.audit.design_condition_number
            or value.get("posterior") != carrier.payload()
            or value.get("source_prior_receipt_sha256") != sha256_bytes(canonical_json_bytes(source_prior_receipt))
            or value.get("closure") != _validate_closure(closure)):
        raise PosteriorCarrierError("session-posterior receipt binding drift")
