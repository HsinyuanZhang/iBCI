"""One-scalar posterior-credibility gate on the frozen Cell-D identity path.

This module deliberately receives only a directional credibility vector.  It
does not accept a posterior mean, posterior-normalized carrier, sample, or
attention bias; the ordinary normalized OLS T4 side features remain the sole
Cell-D identity input.  The scalar alpha is the only new trainable parameter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn.parameter import UninitializedParameter

from . import plan


class PIRGError(RuntimeError):
    """Fail closed when the one-factor PIRG contract drifts."""


CREDIBILITY_FLOOR = 1e-6
Z_CLAMP = 4.0
GATE_HALF_RANGE = 0.5


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PIRGError(message)


def _finite_tensor(value: Tensor, label: str, *, dimensions: int | None = None) -> None:
    _require(torch.is_tensor(value), f"{label} must be a tensor")
    if dimensions is not None:
        _require(value.ndim == dimensions, f"{label} dimensionality drift")
    _require(value.is_floating_point(), f"{label} must be floating point")
    _require(bool(torch.isfinite(value).all().item()), f"{label} is nonfinite")


def gate_budget_for(epoch: int, session_index: int) -> int:
    """Exact three-epoch source rotation: every session sees M4/M10/M30 once."""
    return plan.budget_for(epoch, session_index)


def build_gate_schedule(*, roster: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    if not isinstance(roster, (tuple, list)) or not roster or len(set(roster)) != len(roster):
        raise PIRGError("PIRG roster must be an ordered nonempty unique sequence")
    if any(not isinstance(session, str) or not session for session in roster):
        raise PIRGError("PIRG roster session identifier drift")
    rows = tuple(tuple(gate_budget_for(epoch, index) for index in range(len(roster))) for epoch in range(plan.EPOCHS))
    for index in range(len(roster)):
        if set(row[index] for row in rows) != set(plan.BUDGETS):
            raise PIRGError("PIRG three-epoch schedule does not expose every budget exactly once")
    return rows


def centered_log_credibility(credibility: Tensor) -> Tensor:
    """Session-centred, clamped log precision on the unit axis.

    Uniform vectors cancel exactly because each log value is the same tensor
    value and the per-session mean is that same value.  The function supports
    a static ``[N]`` control or an explicitly identical ``[B,N]`` expansion.
    """
    _finite_tensor(credibility, "PIRG credibility")
    _require(credibility.ndim in {1, 2} and credibility.shape[-1] >= 1, "PIRG credibility must be [N] or [B,N]")
    _require(bool(((credibility >= CREDIBILITY_FLOOR) & (credibility <= 1.0)).all().item()),
             "PIRG credibility must lie in [1e-6, 1]")
    log_value = torch.log(credibility)
    centered = log_value - log_value.mean(dim=-1, keepdim=True)
    result = torch.clamp(centered, min=-Z_CLAMP, max=Z_CLAMP)
    _finite_tensor(result, "PIRG centered log credibility", dimensions=credibility.ndim)
    return result


def residual_gate(*, alpha: Tensor, credibility: Tensor) -> tuple[Tensor, Tensor]:
    """Return `(gate, z)` with exact alpha-zero and uniform cancellation.

    Do not special-case alpha zero: the literal arithmetic produces ones in
    the forward pass while retaining the nonzero alpha derivative required to
    learn from nonuniform confidence.
    """
    _finite_tensor(alpha, "PIRG alpha")
    _require(alpha.numel() == 1, "PIRG alpha must be exactly one scalar")
    z = centered_log_credibility(credibility)
    alpha_value = alpha.to(device=z.device, dtype=z.dtype)
    gate = torch.ones_like(z) + GATE_HALF_RANGE * torch.tanh(alpha_value) * torch.tanh(z)
    _finite_tensor(gate, "PIRG residual gate", dimensions=z.ndim)
    _require(bool(((gate >= 0.5) & (gate <= 1.5)).all().item()), "PIRG residual gate left its closed bound")
    return gate, z


def _lazy_safe_signature(module: nn.Module) -> tuple[int, frozenset[int], tuple[str, ...]]:
    live = 0
    identities: set[int] = set()
    lazy: list[str] = []
    for name, parameter in module.named_parameters():
        identities.add(id(parameter))
        if isinstance(parameter, UninitializedParameter):
            lazy.append(name)
        else:
            live += int(parameter.numel())
    return live, frozenset(identities), tuple(sorted(lazy))


@dataclass(frozen=True)
class CellDPreservation:
    base_live_parameters: int
    wrapper_live_parameters: int
    new_trainable_parameter_names: tuple[str, ...]
    new_trainable_parameter_numel: int
    base_parameter_object_ids_preserved: bool
    base_lazy_parameter_names: tuple[str, ...]
    wrapper_lazy_parameter_names: tuple[str, ...]
    dynamic_dropout: bool
    dropout_low: float
    dropout_high: float
    decoder_mode: str

    def payload(self) -> dict[str, object]:
        return {
            "base_live_parameters": self.base_live_parameters,
            "wrapper_live_parameters": self.wrapper_live_parameters,
            "new_trainable_parameter_names": list(self.new_trainable_parameter_names),
            "new_trainable_parameter_numel": self.new_trainable_parameter_numel,
            "base_parameter_object_ids_preserved": self.base_parameter_object_ids_preserved,
            "base_lazy_parameter_names": list(self.base_lazy_parameter_names),
            "wrapper_lazy_parameter_names": list(self.wrapper_lazy_parameter_names),
            "dynamic_dropout": self.dynamic_dropout,
            "dropout_low": self.dropout_low,
            "dropout_high": self.dropout_high,
            "decoder_mode": self.decoder_mode,
        }


class PosteriorIdentityResidualGate(nn.Module):
    """Frozen Cell-D plus one scalar confidence gate on *identity only*."""

    def __init__(self, cell_d: nn.Module) -> None:
        super().__init__()
        self.cell_d = cell_d
        self._validate_cell_d_surface()
        for parameter in self.cell_d.parameters():
            # Cell-D deliberately carries two dead lazy ``fc_id_in``
            # placeholders.  ``requires_grad_`` is illegal on an
            # ``UninitializedParameter`` even though it is never reached by
            # the coupled forward graph.  Assigning the property is the
            # non-materialising PyTorch-supported operation; it preserves the
            # exact lazy topology while keeping it out of the only-alpha
            # optimizer contract.
            if isinstance(parameter, UninitializedParameter):
                parameter.requires_grad = False
            else:
                parameter.requires_grad_(False)
        # Explicit float32 scalar is independent of the frozen Cell-D tensor
        # topology.  It remains the sole optimizer parameter.
        self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self._validate_only_alpha_trainable()

    def _validate_cell_d_surface(self) -> None:
        base, decoder = self.cell_d, getattr(self.cell_d, "decoder", None)
        required = ("compute_identity", "decode_with_identity", "decoder_mode", "fixed_slot_router", "live_activity_gain")
        _require(decoder is not None and all(hasattr(base, key) for key in required),
                 "PIRG requires the coupled sealed Cell-D surface")
        _require(base.decoder_mode == "coupled" and base.fixed_slot_router is None and base.live_activity_gain is None,
                 "PIRG cannot alter Cell-D decoder topology")
        for field in ("dynamic_dropout", "dynamic_dropout_low", "dynamic_dropout_high", "fc_in", "transformer", "fc_out"):
            _require(hasattr(decoder, field), "PIRG Cell-D decoder surface drift")
        _require(bool(decoder.dynamic_dropout) and float(decoder.dynamic_dropout_low) == 0.0
                 and float(decoder.dynamic_dropout_high) == 1.0,
                 "PIRG must retain Cell-D U(0,1) dynamic whole-unit dropout")

    def _validate_only_alpha_trainable(self) -> None:
        trainable = tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)
        _require(trainable == ("alpha",), "PIRG alpha must be the only trainable parameter")
        _require(self.alpha.shape == torch.Size([]) and self.alpha.dtype == torch.float32,
                 "PIRG alpha scalar shape/dtype drift")

    def preservation(self) -> CellDPreservation:
        base_live, base_ids, base_lazy = _lazy_safe_signature(self.cell_d)
        wrapper_live, wrapper_ids, wrapper_lazy_prefixed = _lazy_safe_signature(self)
        wrapper_lazy = tuple(name.removeprefix("cell_d.") for name in wrapper_lazy_prefixed)
        alpha_ids = {id(self.alpha)}
        result = CellDPreservation(
            base_live_parameters=base_live,
            wrapper_live_parameters=wrapper_live,
            new_trainable_parameter_names=tuple(name for name, p in self.named_parameters() if p.requires_grad),
            new_trainable_parameter_numel=int(self.alpha.numel()),
            base_parameter_object_ids_preserved=base_ids.issubset(wrapper_ids) and wrapper_ids - base_ids == alpha_ids,
            base_lazy_parameter_names=base_lazy,
            wrapper_lazy_parameter_names=wrapper_lazy,
            dynamic_dropout=bool(self.cell_d.decoder.dynamic_dropout),
            dropout_low=float(self.cell_d.decoder.dynamic_dropout_low),
            dropout_high=float(self.cell_d.decoder.dynamic_dropout_high),
            decoder_mode=str(self.cell_d.decoder_mode),
        )
        _require(result.wrapper_live_parameters == result.base_live_parameters + 1,
                 "PIRG wrapper has more than one live scalar beyond Cell-D")
        _require(result.new_trainable_parameter_names == ("alpha",) and result.new_trainable_parameter_numel == 1
                 and result.base_parameter_object_ids_preserved and result.base_lazy_parameter_names == result.wrapper_lazy_parameter_names
                 and result.dynamic_dropout and result.dropout_low == 0.0 and result.dropout_high == 1.0
                 and result.decoder_mode == "coupled", "PIRG Cell-D preservation audit drift")
        return result

    @staticmethod
    def _expanded_gate(*, credibility: Tensor, alpha: Tensor, batch_size: int, units: int,
                       device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        gate, z = residual_gate(alpha=alpha, credibility=credibility)
        if gate.ndim == 1:
            _require(gate.shape == (units,), "PIRG static credibility unit count drift")
            gate, z = gate.unsqueeze(0).expand(batch_size, -1), z.unsqueeze(0).expand(batch_size, -1)
        _require(gate.shape == (batch_size, units) and z.shape == (batch_size, units),
                 "PIRG credibility batch/unit axis drift")
        return gate.to(device=device, dtype=dtype), z.to(device=device, dtype=dtype)

    def forward(
        self,
        neural: Tensor,
        *,
        calib_trials_m30: Tensor,
        ordinary_ols_side_features: Tensor,
        directional_credibility: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return behaviour, ungated identity, bounded gate, and centred z.

        The ordinary OLS side tensor is intentionally named and supplied
        separately from credibility.  PIRG has no API capable of replacing it
        with posterior means or a posterior normalizer.
        """
        _finite_tensor(neural, "PIRG neural", dimensions=3)
        _finite_tensor(calib_trials_m30, "PIRG M30 calibration", dimensions=4)
        _finite_tensor(ordinary_ols_side_features, "PIRG ordinary OLS side features")
        batch, window, units = neural.shape
        _require(window == 50 and calib_trials_m30.shape[0] == batch and calib_trials_m30.shape[1] == 30
                 and calib_trials_m30.shape[3] == units, "PIRG held Cell-D neural/calibration axes drift")
        _require(ordinary_ols_side_features.shape == (batch, units, 4),
                 "PIRG ordinary OLS side features must be [B,N,4]")
        _require(ordinary_ols_side_features.device == neural.device and calib_trials_m30.device == neural.device,
                 "PIRG source tensors must share the Cell-D device")
        _require(ordinary_ols_side_features.dtype == neural.dtype and calib_trials_m30.dtype == neural.dtype,
                 "PIRG source tensors must share the Cell-D dtype")
        self._validate_only_alpha_trainable()
        identity = self.cell_d.compute_identity(calib_trials_m30, side_features=ordinary_ols_side_features)
        _finite_tensor(identity, "PIRG Cell-D identity", dimensions=3)
        _require(identity.shape == (batch, units, window), "PIRG Cell-D identity shape drift")
        gate, z = self._expanded_gate(
            credibility=directional_credibility, alpha=self.alpha, batch_size=batch, units=units,
            device=identity.device, dtype=identity.dtype,
        )
        # The activity `neural` is passed unchanged.  The one intended edit is
        # the multiplicative residual on calibration-derived identity tokens.
        behaviour = self.cell_d.decode_with_identity(neural, identity * gate.unsqueeze(-1))
        _finite_tensor(behaviour, "PIRG behaviour", dimensions=3)
        return behaviour, identity, gate, z


@dataclass(frozen=True)
class GateControl:
    session: str
    session_index: int
    epoch: int
    budget: int
    credibility: Tensor
    centered_log_credibility: Tensor
    posterior_sha256: str

    def __post_init__(self) -> None:
        _require(isinstance(self.session, str) and bool(self.session), "PIRG gate-control session drift")
        _require(type(self.session_index) is int and self.session_index >= 0, "PIRG gate-control session-index drift")
        _require(type(self.epoch) is int and 0 <= self.epoch < plan.EPOCHS, "PIRG gate-control epoch drift")
        _require(self.budget == gate_budget_for(self.epoch, self.session_index),
                 "PIRG gate-control budget drift")
        _finite_tensor(self.credibility, "PIRG gate-control credibility", dimensions=1)
        _finite_tensor(self.centered_log_credibility, "PIRG gate-control z", dimensions=1)
        _require(self.credibility.shape == self.centered_log_credibility.shape, "PIRG gate-control unit shape drift")
        _require(isinstance(self.posterior_sha256, str) and len(self.posterior_sha256) == 64,
                 "PIRG gate-control posterior digest drift")

    def gate(self, alpha: Tensor) -> Tensor:
        result, rebuilt_z = residual_gate(alpha=alpha, credibility=self.credibility)
        _require(torch.equal(rebuilt_z, self.centered_log_credibility), "PIRG cached gate z drift")
        return result


@dataclass(frozen=True)
class GateCacheObserver:
    source_sessions: int
    logical_epochs: int
    controls_built: int
    optimizer_batch_requests: int
    optimizer_batch_inverse_calls: int
    posterior_mean_view_builds: int
    posterior_sampling_view_builds: int
    posterior_normalizer_view_builds: int

    def payload(self) -> dict[str, int]:
        return {
            "source_sessions": self.source_sessions,
            "logical_epochs": self.logical_epochs,
            "controls_built": self.controls_built,
            "optimizer_batch_requests": self.optimizer_batch_requests,
            "optimizer_batch_inverse_calls": self.optimizer_batch_inverse_calls,
            "posterior_mean_view_builds": self.posterior_mean_view_builds,
            "posterior_sampling_view_builds": self.posterior_sampling_view_builds,
            "posterior_normalizer_view_builds": self.posterior_normalizer_view_builds,
        }


class CredibilityGateCache:
    """Cache credibility controls at session×epoch boundaries only.

    `posterior_bank` is the audited source adapter's already-fitted 27×3
    bank.  This class calls its lookup method only while prewarming and never
    asks it for a mean view, a sampled view, or a device-normalized carrier.
    """

    def __init__(self, *, roster: Sequence[str], posterior_bank: Any) -> None:
        self.roster = tuple(roster)
        _require(self.roster and len(set(self.roster)) == len(self.roster), "PIRG cache roster drift")
        self._bank = posterior_bank
        _require(hasattr(posterior_bank, "posterior_for"), "PIRG requires audited posterior-bank lookup surface")
        self._cache: dict[tuple[str, int, str], GateControl] = {}
        self._optimizer_batch_requests = 0

    @staticmethod
    def _device_key(device: torch.device | str) -> str:
        return str(torch.device(device))

    def prewarm_epoch(self, *, epoch: int, device: torch.device | str) -> None:
        _require(type(epoch) is int and 0 <= epoch < plan.EPOCHS, "PIRG cache epoch drift")
        destination = torch.device(device)
        for index, session in enumerate(self.roster):
            budget = gate_budget_for(epoch, index)
            posterior = self._bank.posterior_for(session=session, epoch=epoch)
            credibility = getattr(posterior, "credibility", None)
            digest_fn = getattr(posterior, "digest", None)
            _require(torch.is_tensor(credibility) and callable(digest_fn), "PIRG posterior-bank value surface drift")
            # This is the sole route-local device copy; it is before batch
            # iteration and does not invoke posterior fitting/inversion.
            copied = credibility.detach().to(device=destination, copy=True)
            z = centered_log_credibility(copied).detach().clone()
            key = (session, epoch, self._device_key(destination))
            control = GateControl(
                session=session, session_index=index, epoch=epoch, budget=budget, credibility=copied.detach().clone(),
                centered_log_credibility=z, posterior_sha256=str(digest_fn()),
            )
            existing = self._cache.get(key)
            if existing is not None and (
                existing.budget != control.budget or existing.posterior_sha256 != control.posterior_sha256
                or not torch.equal(existing.credibility, control.credibility)
            ):
                raise PIRGError("PIRG session/epoch cache attempted to replace a control")
            self._cache[key] = control

    def control_for_optimizer_batch(self, *, session: str, epoch: int, device: torch.device | str) -> GateControl:
        key = (session, epoch, self._device_key(device))
        control = self._cache.get(key)
        if control is None:
            raise PIRGError("PIRG optimizer batch attempted to construct a credibility control")
        self._optimizer_batch_requests += 1
        return control

    def control_for_receipt_audit(self, *, session: str, epoch: int, device: torch.device | str) -> GateControl:
        """Read an already-built control without pretending it was a batch use.

        Source-authority construction and epoch summary statistics need the
        same immutable control evidence, but neither belongs to the timed
        optimizer loop.  Keeping this path separate makes the receipt counter
        an exact proof that every recorded ``optimizer_batch_request`` was a
        real B32 training batch.
        """
        key = (session, epoch, self._device_key(device))
        control = self._cache.get(key)
        if control is None:
            raise PIRGError("PIRG receipt audit attempted to construct a credibility control")
        return control

    def observer(self) -> GateCacheObserver:
        return GateCacheObserver(
            source_sessions=len(self.roster), logical_epochs=plan.EPOCHS,
            controls_built=len(self._cache), optimizer_batch_requests=self._optimizer_batch_requests,
            optimizer_batch_inverse_calls=0, posterior_mean_view_builds=0,
            posterior_sampling_view_builds=0, posterior_normalizer_view_builds=0,
        )
