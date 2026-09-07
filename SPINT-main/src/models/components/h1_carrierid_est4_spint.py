"""H1-EST4: source-trained differentiable closed-form carrier estimator.

The CarrierID consumer remains the registered h=32/4-D path.  EST4 changes
only how its four carrier values are estimated: it keeps a batched ridge solve
whose right-hand side structurally requires calibration kinematics, while
making the neural projection, kinematic projection, EB prior and penalty
source-trainable.  At deployment the module is still a forward computation and
one linear solve; it owns no learned target-session parameters, optimizer
state, or backward path.  The current implementation is not an end-to-end
streaming operator: its caller still supplies the ordinary `[M,N,1024]`
temporal identity tensor and padded rate/label/mask calibration inputs.

This file deliberately contains no loader, target evaluator, launch routine,
or checkpoint restore logic.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.models.components.h1_carrierid_spint import (
    H1_CARRIERID_DIM,
    H1_CARRIERID_HIDDEN_DIM,
    H1_CARRIERID_PARAMETERS,
    H1_CARRIERID_TRIAL_LENGTH,
    H1_CARRIERID_WINDOW_SIZE,
    H1CarrierIdSpint,
)
from src.h1_m4_cce_contract import state_hash


EST4_Q = 16
EST4_NEURONS = 176
EST4_KINEMATIC_DIM = 7
EST4_DIM = 4
EST4_ADDED_LEARNED_PARAMETERS = EST4_Q * EST4_NEURONS + EST4_KINEMATIC_DIM * EST4_DIM + EST4_DIM + 2
EST4_CARRIERID_PARAMETERS = H1_CARRIERID_PARAMETERS + EST4_ADDED_LEARNED_PARAMETERS
EST4_MIN_LAMBDA = 1.0e-6
EST4_MIN_TAU2 = 1.0e-12
EST4_MAX_BLOCKS_PER_TRIAL = 256
EST4_TEMPORAL_IDENTITY_INPUT_ELEMENTS = 4 * H1_CARRIERID_TRIAL_LENGTH * EST4_NEURONS
EST4_PADDED_RATE_INPUT_ELEMENTS = 4 * EST4_MAX_BLOCKS_PER_TRIAL * EST4_NEURONS
EST4_PADDED_LABEL_INPUT_ELEMENTS = 4 * EST4_MAX_BLOCKS_PER_TRIAL * EST4_KINEMATIC_DIM
EST4_PADDED_MASK_ELEMENTS = 4 * EST4_MAX_BLOCKS_PER_TRIAL


class Est4Error(ValueError):
    """Raised for a malformed frozen source-plan or analytic input."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise Est4Error(message)


def _read_frozen_plan(path: str | Path) -> tuple[dict[str, np.ndarray], Path]:
    """Read only the immutable source-bundle plan required to initialise EST4."""

    candidate = Path(path).resolve()
    _need(candidate.is_file(), f"EST4 frozen source plan is missing: {candidate}")
    try:
        with np.load(candidate, allow_pickle=False) as arrays:
            expected = {"mean", "scale", "pcs", "q", "lambda", "U", "mu", "tau2"}
            _need(set(arrays.files) == expected, "EST4 frozen plan members drift")
            plan = {name: np.asarray(arrays[name], dtype=np.float64) for name in expected}
    except (OSError, ValueError) as error:
        raise Est4Error(f"EST4 cannot read frozen source plan: {candidate}") from error
    _need(int(plan["q"].item()) == EST4_Q, "EST4 fixes q=16")
    _need(plan["mean"].shape == (EST4_NEURONS,) and plan["scale"].shape == (EST4_NEURONS,),
          "EST4 frozen normalization shape drift")
    _need(plan["pcs"].ndim == 2 and plan["pcs"].shape[0] >= EST4_Q and plan["pcs"].shape[1] == EST4_NEURONS,
          "EST4 frozen PCA shape drift")
    _need(plan["U"].shape == (EST4_KINEMATIC_DIM, EST4_DIM), "EST4 fixes U=[7,4]")
    _need(plan["mu"].shape == (EST4_DIM,), "EST4 fixes four-dimensional EB prior")
    _need(np.isfinite(plan["mean"]).all() and np.isfinite(plan["scale"]).all()
          and np.all(plan["scale"] > 0.0), "EST4 source normalization is invalid")
    for name in ("pcs", "U", "mu", "lambda", "tau2"):
        _need(np.isfinite(plan[name]).all(), f"EST4 frozen plan {name} is nonfinite")
    _need(float(plan["lambda"].item()) > 0.0 and float(plan["tau2"].item()) > 0.0,
          "EST4 requires positive frozen ridge/prior initialisation")
    return plan, candidate


class DifferentiableClosedFormEstimator4(nn.Module):
    """The bounded EST4 analytic estimator; labels have no bypass around ridge RHS."""

    def __init__(self, *, frozen_plan_path: str, source_normalizer: float) -> None:
        super().__init__()
        plan, candidate = _read_frozen_plan(frozen_plan_path)
        _need(math.isfinite(float(source_normalizer)) and float(source_normalizer) > 0.0,
              "EST4 source RMS normalizer must be positive and finite")
        self.frozen_plan_path = str(candidate)
        self.register_buffer("mean", torch.as_tensor(plan["mean"], dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(plan["scale"], dtype=torch.float32))
        self.register_buffer("source_normalizer", torch.tensor(float(source_normalizer), dtype=torch.float32))
        self.pcs = nn.Parameter(torch.as_tensor(plan["pcs"][:EST4_Q], dtype=torch.float32))
        self.U = nn.Parameter(torch.as_tensor(plan["U"], dtype=torch.float32))
        self.mu = nn.Parameter(torch.as_tensor(plan["mu"], dtype=torch.float32))
        self.log_lambda = nn.Parameter(torch.tensor(math.log(float(plan["lambda"].item())), dtype=torch.float32))
        self.log_tau2 = nn.Parameter(torch.tensor(math.log(float(plan["tau2"].item())), dtype=torch.float32))
        _need(self.parameter_count() == EST4_ADDED_LEARNED_PARAMETERS, "EST4 parameter accounting drift")

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def deployment_cost(self) -> dict[str, Any]:
        return {
            "estimator": "differentiable_closed_form_ridge_forward_only_at_deployment",
            "learned_source_parameters": self.parameter_count(),
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
            "persistent_learned_target_parameters": 0,
            "target_optimizer_state_elements": 0,
            "current_forward_requires_raw_temporal_identity": True,
            "current_temporal_identity_input_elements": EST4_TEMPORAL_IDENTITY_INPUT_ELEMENTS,
            "current_padded_rate_input_elements": EST4_PADDED_RATE_INPUT_ELEMENTS,
            "current_padded_label_input_elements": EST4_PADDED_LABEL_INPUT_ELEMENTS,
            "current_padded_mask_elements": EST4_PADDED_MASK_ELEMENTS,
            "fully_online_sufficient_statistic_encoder": False,
            "carrier_dim": EST4_DIM,
            "q": EST4_Q,
        }

    def bind_source_normalizer(self, value: float) -> None:
        """Bind the immutable source-only RMS before the first source forward."""

        _need(math.isfinite(float(value)) and float(value) > 0.0, "EST4 source normalizer binding is invalid")
        self.source_normalizer.fill_(float(value))

    def forward(self, rates: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Estimate normalized ``[B,N,4]`` carrier from calibration rates/labels.

        ``labels`` enter the sole ridge right-hand side.  Therefore a caller
        cannot obtain a learned carrier by silently dropping target calibration
        labels or replacing the estimator with a neural-only identity MLP.
        """

        _need(rates.ndim == 4 and labels.ndim == 4 and valid_mask.ndim == 3,
              "EST4 expects rates/labels/mask [B,M,L,*]")
        batch, trials, blocks, neurons = rates.shape
        _need(trials == 4 and neurons == EST4_NEURONS, "EST4 fixes M=4 and N=176")
        _need(tuple(labels.shape) == (batch, trials, blocks, EST4_KINEMATIC_DIM), "EST4 label shape drift")
        _need(tuple(valid_mask.shape) == (batch, trials, blocks), "EST4 validity mask shape drift")
        _need(torch.isfinite(rates).all().item() and torch.isfinite(labels).all().item(), "EST4 calibration tensors are nonfinite")
        flat_rates = rates.reshape(batch, trials * blocks, neurons)
        flat_labels = labels.reshape(batch, trials * blocks, EST4_KINEMATIC_DIM)
        weight = valid_mask.reshape(batch, trials * blocks).to(dtype=flat_rates.dtype)
        _need(torch.all(weight.sum(dim=1) >= EST4_Q + 2).item(), "EST4 support has insufficient valid blocks for q=16 ridge")
        z = torch.matmul((flat_rates - self.mean) / self.scale, self.pcs.t())
        ones = torch.ones((batch, z.shape[1], 1), dtype=z.dtype, device=z.device)
        design = torch.cat((ones, z), dim=-1)
        weighted_design = design * weight.unsqueeze(-1)
        xtx = torch.matmul(design.transpose(1, 2), weighted_design)
        penalty = torch.eye(EST4_Q + 1, dtype=z.dtype, device=z.device).expand(batch, -1, -1).clone()
        penalty[:, 0, 0] = 0.0
        ridge_lambda = torch.exp(self.log_lambda).clamp_min(EST4_MIN_LAMBDA)
        system = xtx + ridge_lambda * penalty
        rhs = torch.matmul(design.transpose(1, 2), flat_labels * weight.unsqueeze(-1))
        try:
            beta = torch.linalg.solve(system, rhs)
        except RuntimeError as error:
            raise Est4Error("EST4 ridge system is singular despite positive ridge penalty") from error
        raw_rows = torch.matmul(self.pcs.t(), beta[:, 1:, :]) / self.scale.view(1, -1, 1)
        raw_carrier = torch.matmul(raw_rows, self.U)

        prediction = torch.matmul(design, beta)
        residual = flat_labels - prediction
        rss = torch.sum(torch.square(residual) * weight.unsqueeze(-1), dim=1)
        inv_system = torch.linalg.inv(system)
        hat_trace = torch.einsum("bij,bji->b", xtx, inv_system)
        degrees = weight.sum(dim=1) - hat_trace
        _need(torch.all(torch.isfinite(degrees) & (degrees > 0.0)).item(), "EST4 residual degrees of freedom invalid")
        sigma2 = rss / degrees.unsqueeze(-1)
        # G is the ridge coefficient covariance factor.  Restrict it to the
        # non-intercept block before mapping each latent dimension back to N.
        g = torch.matmul(torch.matmul(inv_system, xtx), inv_system)
        p = self.pcs.t().unsqueeze(0).expand(batch, -1, -1)
        channel_factor = torch.sum(torch.matmul(p, g[:, 1:, 1:]) * p, dim=-1) / torch.square(self.scale).view(1, -1)
        output_trace = torch.sum(sigma2 * torch.sum(torch.square(self.U), dim=1).view(1, -1), dim=1) / float(EST4_DIM)
        projected_variance = channel_factor * output_trace.unsqueeze(-1)
        tau2 = torch.exp(self.log_tau2).clamp_min(EST4_MIN_TAU2)
        shrinkage = tau2 / (tau2 + projected_variance.clamp_min(0.0))
        carrier = self.mu.view(1, 1, EST4_DIM) + shrinkage.unsqueeze(-1) * (raw_carrier - self.mu.view(1, 1, EST4_DIM))
        _need(torch.isfinite(carrier).all().item(), "EST4 carrier became nonfinite")
        return carrier / self.source_normalizer


class H1CarrierIdEst4Spint(H1CarrierIdSpint):
    """Registered CarrierID consumer plus an optional EST4 carrier producer."""

    def __init__(
        self, *, estimator_mode: str, frozen_plan_path: str | None = None,
        source_normalizer: float | None = None, row_shuffle_output: bool = False,
        **kwargs: Any,
    ) -> None:
        if estimator_mode not in {"baseline", "learned"}:
            raise Est4Error("EST4 estimator_mode must be baseline or learned")
        super().__init__(**kwargs)
        self.estimator_mode = str(estimator_mode)
        self.row_shuffle_output = bool(row_shuffle_output)
        if self.estimator_mode == "learned":
            if not frozen_plan_path or source_normalizer is None:
                raise Est4Error("learned EST4 requires frozen source plan and RMS normalizer")
            self.estimator = DifferentiableClosedFormEstimator4(
                frozen_plan_path=str(frozen_plan_path), source_normalizer=float(source_normalizer),
            )
        else:
            if frozen_plan_path is not None or source_normalizer is not None or self.row_shuffle_output:
                raise Est4Error("baseline B-C/B-LS has no learned estimator or output row shuffle")
            self.estimator = None

    def estimator_parameter_count(self) -> int:
        return 0 if self.estimator is None else self.estimator.parameter_count()

    def est4_parameter_accounting(self) -> dict[str, int]:
        return {
            "carrierid_consumer_parameters": self.carrier_parameter_count(),
            "estimator_parameters": self.estimator_parameter_count(),
            "combined_identity_path_parameters": self.carrier_parameter_count() + self.estimator_parameter_count(),
        }

    def bind_source_normalizer(self, value: float) -> None:
        if self.estimator is None:
            return
        self.estimator.bind_source_normalizer(value)

    def shared_backbone_state_hash(self) -> str:
        """Hash the common SPINT/CarrierID consumer, excluding EST4-only tensors."""

        state = {name: tensor for name, tensor in self.state_dict().items() if not name.startswith("estimator.")}
        return state_hash(state)

    def forward(
        self, src: torch.Tensor, *, calib_trialized_neural_features: torch.Tensor,
        carrier: torch.Tensor, calibration_rates: torch.Tensor | None = None,
        calibration_labels: torch.Tensor | None = None, calibration_mask: torch.Tensor | None = None,
        row_permutation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        effective = carrier
        if self.estimator is not None:
            if calibration_rates is None or calibration_labels is None or calibration_mask is None:
                raise Est4Error("learned EST4 requires calibration rates, labels, and validity mask")
            effective = self.estimator(calibration_rates, calibration_labels, calibration_mask)
            if self.row_shuffle_output:
                if row_permutation is None or row_permutation.ndim != 2 or row_permutation.shape != effective.shape[:2]:
                    raise Est4Error("EST4 L-RS requires one [B,N] output row permutation")
                effective = torch.gather(effective, 1, row_permutation.to(device=effective.device).unsqueeze(-1).expand_as(effective))
        return super().forward(src, calib_trialized_neural_features=calib_trialized_neural_features, carrier=effective)
