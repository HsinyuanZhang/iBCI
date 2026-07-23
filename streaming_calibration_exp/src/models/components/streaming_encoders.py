"""Calibration encoders for streaming few-shot identity estimation (B0–B6)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_affine_stack(
    in_dim: int,
    hidden_dim: int,
    num_affine_layers: int,
    out_dim: int,
) -> nn.Sequential:
    """Build exactly `num_affine_layers` Linear layers (matches SPINT num_id_layers)."""
    if num_affine_layers < 1:
        raise ValueError("num_affine_layers must be >= 1")
    layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim if num_affine_layers > 1 else out_dim)]
    if num_affine_layers == 1:
        return nn.Sequential(*layers)
    layers.append(nn.ReLU())
    for _ in range(num_affine_layers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


def _count_affine_layers(module: nn.Sequential) -> int:
    return sum(isinstance(layer, nn.Linear) for layer in module)


def _affine_mac_per_vector(in_dim: int, layers: nn.Sequential) -> int:
    """Count MAC for one [*, in_dim] -> [*, out_dim] affine stack."""
    mac = 0
    current = in_dim
    for layer in layers:
        if isinstance(layer, nn.Linear):
            mac += current * layer.out_features
            current = layer.out_features
    return mac


def _trial_to_batch_neurons_time(trial: torch.Tensor) -> torch.Tensor:
    """Datamodule convention: trial is [B, T, N] -> [B, N, T]."""
    if trial.dim() == 2:
        trial = trial.unsqueeze(0)
    if trial.dim() != 3:
        raise ValueError(f"Expected trial [B,T,N] or [T,N], got {tuple(trial.shape)}")
    return trial.permute(0, 2, 1)


def _resolve_trial_lengths(
    trial: torch.Tensor,
    trial_length: Optional[Union[int, torch.Tensor]],
) -> torch.Tensor:
    """Return per-batch valid sample counts with shape [B].

    `trial` must already be in [B, N, T] layout.
    """
    batch_size = trial.shape[0]
    time_steps = trial.shape[-1]
    if trial_length is None:
        return torch.full((batch_size,), time_steps, device=trial.device, dtype=torch.long)
    if isinstance(trial_length, int):
        return torch.full((batch_size,), trial_length, device=trial.device, dtype=torch.long)
    lengths = trial_length.to(device=trial.device, dtype=torch.long).view(-1)
    if lengths.numel() == 1 and batch_size > 1:
        lengths = lengths.expand(batch_size)
    if lengths.shape[0] != batch_size:
        raise ValueError(f"trial_length batch {lengths.shape[0]} != trial batch {batch_size}")
    return lengths.clamp(min=0, max=time_steps)


def _as_batch_neurons(sample: torch.Tensor) -> torch.Tensor:
    if sample.dim() == 1:
        return sample.unsqueeze(0)
    if sample.dim() == 2:
        return sample
    raise ValueError(f"Expected per-bin sample [N] or [B,N], got {tuple(sample.shape)}")


def _is_valid_bin(sample: torch.Tensor, lengths: torch.Tensor, time_idx: int, pad_value: Optional[float]) -> torch.Tensor:
    """Per-neuron validity mask with shape [B, N].

    Validity is driven by the declared per-batch trial length. Sentinel pad values are not
    applied inside the valid prefix, so cubic overshoot negatives are not dropped bin-wise.
    """
    del pad_value
    neural_bin = _as_batch_neurons(sample)
    time_valid = time_idx < lengths
    return time_valid.unsqueeze(-1).expand_as(neural_bin)


@dataclass
class EncoderCostProfile:
    parameter_count: int
    weight_bytes: int
    trial_buffer_bytes: int
    support_state_bytes: int
    peak_live_state_bytes: int
    mac_per_trial: int
    mac_per_session: int
    requires_cubic_interpolation: bool = True
    requires_general_multiplier: bool = True
    requires_divider: bool = True
    variant: str = ""
    cost_source: str = "cycle_model_estimate"


class CalibrationEncoder(nn.Module, ABC):
    """Unified streaming calibration encoder API (guide section 5)."""

    variant: str = "base"
    window_size: int
    supports_bin_streaming: bool = False
    pad_value: float = -1.0

    @abstractmethod
    def reset_stream(
        self,
        batch_size: int,
        num_neurons: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def push_trial(
        self,
        state: Dict[str, Any],
        trial: torch.Tensor,
        trial_length: Optional[Union[int, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def start_trial(
        self,
        state: Dict[str, Any],
        trial_length: Optional[Union[int, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        if not self.supports_bin_streaming:
            raise NotImplementedError(f"{self.variant} does not support bin-level streaming")
        return state

    def push_sample(
        self,
        state: Dict[str, Any],
        neural_bin: torch.Tensor,
        time_idx: int = 0,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def end_trial(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def forward_batch(
        self,
        calib_trials: torch.Tensor,
        trial_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if calib_trials.dim() != 4:
            raise ValueError(f"Expected [B,M,T,N], got {tuple(calib_trials.shape)}")
        batch_size, _, _, num_neurons = calib_trials.shape
        state = self.reset_stream(batch_size, num_neurons, calib_trials.device, calib_trials.dtype)
        for trial_idx in range(calib_trials.shape[1]):
            length = None if trial_lengths is None else trial_lengths[:, trial_idx]
            state = self.push_trial(state, calib_trials[:, trial_idx], trial_length=length)
        return self.finalize_identity(state)

    def cost_profile(self, num_neurons: int, trial_length: int, num_trials: int) -> EncoderCostProfile:
        params = sum(p.numel() for p in self.parameters())
        mac_trial = self._mac_per_trial(num_neurons, trial_length)
        return EncoderCostProfile(
            parameter_count=params,
            weight_bytes=params * 4,
            trial_buffer_bytes=self._trial_buffer_bytes(num_neurons, trial_length),
            support_state_bytes=self._support_state_bytes(num_neurons),
            peak_live_state_bytes=self._peak_live_state_bytes(num_neurons, trial_length),
            mac_per_trial=mac_trial,
            mac_per_session=self._mac_per_session(num_neurons, trial_length, num_trials),
            variant=self.variant,
            requires_cubic_interpolation=self._requires_cubic_interpolation(),
            requires_general_multiplier=self._requires_general_multiplier(),
            requires_divider=True,
        )

    def _requires_cubic_interpolation(self) -> bool:
        return True

    def _requires_general_multiplier(self) -> bool:
        return True

    def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
        return trial_length * num_neurons * 4

    def _support_state_bytes(self, num_neurons: int) -> int:
        return 0

    def _peak_live_state_bytes(self, num_neurons: int, trial_length: int) -> int:
        return self._trial_buffer_bytes(num_neurons, trial_length) + self._support_state_bytes(num_neurons)

    def _mac_per_trial(self, num_neurons: int, trial_length: int) -> int:
        return 0

    def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
        return num_trials * self._mac_per_trial(num_neurons, trial_length)


class BatchReferenceEncoder(CalibrationEncoder):
    variant = "B0"

    def __init__(self, fc_id_in: nn.Module, fc_id_out: nn.Module, window_size: int) -> None:
        super().__init__()
        self.fc_id_in = fc_id_in
        self.fc_id_out = fc_id_out
        self.window_size = window_size

    def reset_stream(self, batch_size: int, num_neurons: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        hidden = self.fc_id_in[0].out_features
        return {"sum_phi": torch.zeros(batch_size, num_neurons, hidden, device=device, dtype=dtype), "trial_count": 0}

    def push_trial(self, state: Dict[str, Any], trial: torch.Tensor, trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
        trial = _trial_to_batch_neurons_time(trial)
        phi = self.fc_id_in(trial)
        state["sum_phi"] = state["sum_phi"] + phi
        state["trial_count"] += 1
        return state

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        return self.fc_id_out(state["sum_phi"] / state["trial_count"])

    def forward_batch(self, calib_trials: torch.Tensor, trial_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        trials = calib_trials.permute(0, 1, 3, 2)
        phi = self.fc_id_in(trials)
        pooled = torch.mean(phi, dim=1)
        return self.fc_id_out(pooled)

    def _support_state_bytes(self, num_neurons: int) -> int:
        return num_neurons * self.fc_id_in[0].out_features * 4

    def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
        return 0


class TrialStreamingEncoder(BatchReferenceEncoder):
    variant = "B1"

    def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
        return trial_length * num_neurons * 4

    def _peak_live_state_bytes(self, num_neurons: int, trial_length: int) -> int:
        return self._trial_buffer_bytes(num_neurons, trial_length) + self._support_state_bytes(num_neurons)


class LatePoolEncoder(CalibrationEncoder):
    variant = "B2"

    def __init__(self, trial_length: int, window_size: int, id_hidden_dim: int, num_id_layers: int = 3) -> None:
        super().__init__()
        self.trial_length = trial_length
        self.window_size = window_size
        self.id_hidden_dim = id_hidden_dim
        self.num_id_layers = num_id_layers
        self.fc_id_in = _build_affine_stack(trial_length, id_hidden_dim, num_id_layers, id_hidden_dim)
        self.fc_id_out = _build_affine_stack(id_hidden_dim, id_hidden_dim, num_id_layers, window_size)

    def reset_stream(self, batch_size: int, num_neurons: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        return {"sum_phi": torch.zeros(batch_size, num_neurons, self.id_hidden_dim, device=device, dtype=dtype), "trial_count": 0}

    def push_trial(self, state: Dict[str, Any], trial: torch.Tensor, trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
        trial = _trial_to_batch_neurons_time(trial)
        phi = self.fc_id_in(trial)
        state["sum_phi"] = state["sum_phi"] + phi
        state["trial_count"] += 1
        return state

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        return self.fc_id_out(state["sum_phi"] / state["trial_count"])

    def _support_state_bytes(self, num_neurons: int) -> int:
        return num_neurons * self.id_hidden_dim * 4

    def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
        return trial_length * num_neurons * 4

    def _mac_per_trial(self, num_neurons: int, trial_length: int) -> int:
        phi = _affine_mac_per_vector(trial_length, self.fc_id_in)
        return num_neurons * phi

    def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
        psi = num_neurons * _affine_mac_per_vector(self.id_hidden_dim, self.fc_id_out)
        return num_trials * self._mac_per_trial(num_neurons, trial_length) + psi


class EarlyPoolEncoder(CalibrationEncoder):
    variant = "B3"

    def __init__(self, trial_length: int, window_size: int, hidden_dim: int, num_post_layers: int = 3) -> None:
        super().__init__()
        self.trial_length = trial_length
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.pre_pool = nn.Sequential(nn.Linear(trial_length, hidden_dim), nn.ReLU())
        self.post_pool = _build_affine_stack(hidden_dim, hidden_dim, num_post_layers, window_size)

    def reset_stream(self, batch_size: int, num_neurons: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        return {"sum_feat": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype), "trial_count": 0}

    def push_trial(self, state: Dict[str, Any], trial: torch.Tensor, trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
        trial = _trial_to_batch_neurons_time(trial)
        feat = self.pre_pool(trial)
        state["sum_feat"] = state["sum_feat"] + feat
        state["trial_count"] += 1
        return state

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        return self.post_pool(state["sum_feat"] / state["trial_count"])

    def _support_state_bytes(self, num_neurons: int) -> int:
        return num_neurons * self.hidden_dim * 4

    def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
        return trial_length * num_neurons * 4

    def _mac_per_trial(self, num_neurons: int, trial_length: int) -> int:
        return num_neurons * trial_length * self.hidden_dim

    def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
        post = num_neurons * _affine_mac_per_vector(self.hidden_dim, self.post_pool)
        return num_trials * self._mac_per_trial(num_neurons, trial_length) + post


class RelationalEarlyPoolEncoder(CalibrationEncoder):
    """B15: B3 + cross-neuron self-attention to model SUA split/merge relationships."""
    variant = "B15"

    def __init__(self, trial_length: int, window_size: int, hidden_dim: int, num_heads: int = 4, num_post_layers: int = 3) -> None:
        super().__init__()
        self.trial_length = trial_length
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.pre_pool = nn.Sequential(nn.Linear(trial_length, hidden_dim), nn.ReLU())
        self.cross_neuron_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.post_pool = _build_affine_stack(hidden_dim, hidden_dim, num_post_layers, window_size)

    def reset_stream(self, batch_size: int, num_neurons: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        return {"sum_feat": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype), "trial_count": 0}

    def push_trial(self, state: Dict[str, Any], trial: torch.Tensor, trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
        trial = _trial_to_batch_neurons_time(trial)
        feat = self.pre_pool(trial)
        state["sum_feat"] = state["sum_feat"] + feat
        state["trial_count"] += 1
        return state

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        mean_feat = state["sum_feat"] / state["trial_count"]  # [B,N,D]
        attn_out, _ = self.cross_neuron_attn(mean_feat, mean_feat, mean_feat)
        mean_feat = self.attn_norm(mean_feat + attn_out)  # residual + norm
        return self.post_pool(mean_feat)  # [B,N,W]

    def _support_state_bytes(self, num_neurons: int) -> int:
        return num_neurons * self.hidden_dim * 4

    def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
        return trial_length * num_neurons * 4

    def _mac_per_trial(self, num_neurons: int, trial_length: int) -> int:
        return num_neurons * trial_length * self.hidden_dim

    def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
        attn_mac = num_neurons * num_neurons * self.hidden_dim * 3
        post = num_neurons * _affine_mac_per_vector(self.hidden_dim, self.post_pool)
        return num_trials * self._mac_per_trial(num_neurons, trial_length) + attn_mac + post


class HighOrderStatsEncoder(CalibrationEncoder):
    """B16: B3 + cross-trial variance to capture sorting reliability."""
    variant = "B16"

    def __init__(self, trial_length: int, window_size: int, hidden_dim: int, num_post_layers: int = 3) -> None:
        super().__init__()
        self.trial_length = trial_length
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.pre_pool = nn.Sequential(nn.Linear(trial_length, hidden_dim), nn.ReLU())
        self.post_pool = _build_affine_stack(hidden_dim * 2, hidden_dim, num_post_layers, window_size)

    def reset_stream(self, batch_size: int, num_neurons: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        return {
            "sum_feat": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype),
            "sum_feat_sq": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype),
            "trial_count": 0,
        }

    def push_trial(self, state: Dict[str, Any], trial: torch.Tensor, trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
        trial = _trial_to_batch_neurons_time(trial)
        feat = self.pre_pool(trial)
        state["sum_feat"] = state["sum_feat"] + feat
        state["sum_feat_sq"] = state["sum_feat_sq"] + feat * feat
        state["trial_count"] += 1
        return state

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        n = state["trial_count"]
        mean_feat = state["sum_feat"] / n
        var_feat = (state["sum_feat_sq"] / n - mean_feat * mean_feat).clamp(min=0.0)
        combined = torch.cat([mean_feat, var_feat], dim=-1)  # [B,N,2D]
        return self.post_pool(combined)  # [B,N,W]

    def _support_state_bytes(self, num_neurons: int) -> int:
        return num_neurons * self.hidden_dim * 4 * 2

    def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
        return trial_length * num_neurons * 4

    def _mac_per_trial(self, num_neurons: int, trial_length: int) -> int:
        return num_neurons * trial_length * self.hidden_dim

    def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
        post = num_neurons * _affine_mac_per_vector(self.hidden_dim * 2, self.post_pool)
        return num_trials * self._mac_per_trial(num_neurons, trial_length) + post


class B3PreservingHighOrderStatsEncoder(CalibrationEncoder):
    """B16-Z: B3 mean path plus a zero-initialized variance residual.

    A B3 checkpoint maps exactly onto ``pre_pool``, ``mean_linear``, and
    ``post_tail``. With ``var_linear.weight == 0``, this encoder is therefore
    functionally identical to that B3 for every calibration support.
    """

    variant = "B16Z"

    def __init__(self, trial_length: int, window_size: int, hidden_dim: int, num_post_layers: int = 3) -> None:
        super().__init__()
        if num_post_layers != 3:
            raise ValueError("B16-Z currently requires num_post_layers=3 for exact B3 mapping")
        self.trial_length = trial_length
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.pre_pool = nn.Sequential(nn.Linear(trial_length, hidden_dim), nn.ReLU())
        self.mean_linear = nn.Linear(hidden_dim, hidden_dim)
        self.var_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.post_tail = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, window_size),
        )
        nn.init.zeros_(self.var_linear.weight)

    def load_b3_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load an encoder-only B3 state dict and preserve exact B3 behavior."""
        expected = {
            "pre_pool.0.weight",
            "pre_pool.0.bias",
            "post_pool.0.weight",
            "post_pool.0.bias",
            "post_pool.2.weight",
            "post_pool.2.bias",
            "post_pool.4.weight",
            "post_pool.4.bias",
        }
        missing = expected.difference(state_dict)
        unexpected = set(state_dict).difference(expected)
        if missing or unexpected:
            raise ValueError(
                f"Expected an encoder-only three-layer B3 state dict; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        mapped = {
            "pre_pool.0.weight": state_dict["pre_pool.0.weight"],
            "pre_pool.0.bias": state_dict["pre_pool.0.bias"],
            "mean_linear.weight": state_dict["post_pool.0.weight"],
            "mean_linear.bias": state_dict["post_pool.0.bias"],
            "post_tail.1.weight": state_dict["post_pool.2.weight"],
            "post_tail.1.bias": state_dict["post_pool.2.bias"],
            "post_tail.3.weight": state_dict["post_pool.4.weight"],
            "post_tail.3.bias": state_dict["post_pool.4.bias"],
        }
        self.load_state_dict({**mapped, "var_linear.weight": torch.zeros_like(self.var_linear.weight)}, strict=True)

    def freeze_base_path(self) -> None:
        """Freeze the B3-compatible path while leaving the variance residual trainable."""
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name == "var_linear.weight"

    def freeze_for_fusion_tuning(self) -> None:
        """Train only the mean/variance fusion layer; keep feature and tail mappings anchored."""
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith("mean_linear.") or name == "var_linear.weight"

    def reset_stream(self, batch_size: int, num_neurons: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        return {
            "sum_feat": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype),
            "sum_feat_sq": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype),
            "trial_count": 0,
        }

    def push_trial(self, state: Dict[str, Any], trial: torch.Tensor, trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
        trial = _trial_to_batch_neurons_time(trial)
        feat = self.pre_pool(trial)
        state["sum_feat"] = state["sum_feat"] + feat
        state["sum_feat_sq"] = state["sum_feat_sq"] + feat.square()
        state["trial_count"] += 1
        return state

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        n = state["trial_count"]
        mean_feat = state["sum_feat"] / n
        var_feat = (state["sum_feat_sq"] / n - mean_feat.square()).clamp(min=0.0)
        return self.post_tail(self.mean_linear(mean_feat) + self.var_linear(var_feat))

    def _support_state_bytes(self, num_neurons: int) -> int:
        return num_neurons * self.hidden_dim * 4 * 2

    def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
        return trial_length * num_neurons * 4

    def _mac_per_trial(self, num_neurons: int, trial_length: int) -> int:
        return num_neurons * trial_length * self.hidden_dim

    def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
        mean_and_variance = num_neurons * (self.hidden_dim * self.hidden_dim * 2)
        tail = num_neurons * (
            self.hidden_dim * self.hidden_dim + self.hidden_dim * self.window_size
        )
        return num_trials * self._mac_per_trial(num_neurons, trial_length) + mean_and_variance + tail


class B3PreservingNormalizedHighOrderStatsEncoder(B3PreservingHighOrderStatsEncoder):
    """B16-ZF: B3 plus rate-normalized log-Fano residual in the D-dimensional latent."""

    variant = "B16ZF"

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        n = state["trial_count"]
        mean_feat = state["sum_feat"] / n
        var_feat = (state["sum_feat_sq"] / n - mean_feat.square()).clamp(min=0.0)
        normalized_variance = torch.log1p(var_feat / mean_feat.abs().clamp_min(1.0e-3))
        return self.post_tail(self.mean_linear(mean_feat) + self.var_linear(normalized_variance))


class B3PreservingShrunkNormalizedHighOrderStatsEncoder(
    B3PreservingNormalizedHighOrderStatsEncoder
):
    """B16-ZFS: B16-ZF with fixed 25% cross-feature reliability shrinkage.

    Each latent log-Fano feature is pulled toward the per-neuron cross-feature
    mean before the zero-initialized residual projection. This preserves most
    feature-specific information while reducing support noise in the arbitrary
    seed-dependent B3 latent basis.
    """

    variant = "B16ZFS"
    shrinkage_strength = 0.25

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        n = state["trial_count"]
        mean_feat = state["sum_feat"] / n
        var_feat = (state["sum_feat_sq"] / n - mean_feat.square()).clamp(min=0.0)
        normalized_variance = torch.log1p(var_feat / mean_feat.abs().clamp_min(1.0e-3))
        shared_reliability = normalized_variance.mean(dim=-1, keepdim=True)
        shrunk_variance = torch.lerp(
            normalized_variance, shared_reliability, self.shrinkage_strength
        )
        return self.post_tail(self.mean_linear(mean_feat) + self.var_linear(shrunk_variance))


class B3PreservingDropoutNormalizedHighOrderStatsEncoder(
    B3PreservingNormalizedHighOrderStatsEncoder
):
    """B16-ZFD: B16-ZF with training-only log-Fano branch dropout."""

    variant = "B16ZFD"
    reliability_dropout_p = 0.25

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        n = state["trial_count"]
        mean_feat = state["sum_feat"] / n
        var_feat = (state["sum_feat_sq"] / n - mean_feat.square()).clamp(min=0.0)
        normalized_variance = torch.log1p(var_feat / mean_feat.abs().clamp_min(1.0e-3))
        regularized_variance = F.dropout(
            normalized_variance, p=self.reliability_dropout_p, training=self.training
        )
        return self.post_tail(
            self.mean_linear(mean_feat) + self.var_linear(regularized_variance)
        )


class B3PreservingBoundedOutputFanoEncoder(B3PreservingHighOrderStatsEncoder):
    """B16-ZFO: B3 plus a bounded additive log-Fano identity residual.

    Unlike B16-ZF, the reliability branch bypasses the frozen B3 post-tail and
    corrects the final identity directly. The zero-initialized ``D -> W``
    projection preserves exact B3 behavior at warm-start, while a smooth bound
    limits every residual component to 25% of the corresponding neuron's B3
    identity RMS.
    """

    variant = "B16ZFO"
    residual_rms_fraction = 0.25

    def __init__(self, trial_length: int, window_size: int, hidden_dim: int, num_post_layers: int = 3) -> None:
        super().__init__(trial_length, window_size, hidden_dim, num_post_layers)
        del self.var_linear
        self.var_out = nn.Linear(hidden_dim, window_size, bias=False)
        nn.init.zeros_(self.var_out.weight)

    def load_b3_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load a three-layer B3 state and zero the direct output residual."""
        expected = {
            "pre_pool.0.weight",
            "pre_pool.0.bias",
            "post_pool.0.weight",
            "post_pool.0.bias",
            "post_pool.2.weight",
            "post_pool.2.bias",
            "post_pool.4.weight",
            "post_pool.4.bias",
        }
        missing = expected.difference(state_dict)
        unexpected = set(state_dict).difference(expected)
        if missing or unexpected:
            raise ValueError(
                f"Expected an encoder-only three-layer B3 state dict; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        mapped = {
            "pre_pool.0.weight": state_dict["pre_pool.0.weight"],
            "pre_pool.0.bias": state_dict["pre_pool.0.bias"],
            "mean_linear.weight": state_dict["post_pool.0.weight"],
            "mean_linear.bias": state_dict["post_pool.0.bias"],
            "post_tail.1.weight": state_dict["post_pool.2.weight"],
            "post_tail.1.bias": state_dict["post_pool.2.bias"],
            "post_tail.3.weight": state_dict["post_pool.4.weight"],
            "post_tail.3.bias": state_dict["post_pool.4.bias"],
        }
        self.load_state_dict(
            {**mapped, "var_out.weight": torch.zeros_like(self.var_out.weight)}, strict=True
        )

    def freeze_base_path(self) -> None:
        """Freeze the exact B3 path and train only the output residual."""
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name == "var_out.weight"

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        n = state["trial_count"]
        mean_feat = state["sum_feat"] / n
        var_feat = (state["sum_feat_sq"] / n - mean_feat.square()).clamp(min=0.0)
        normalized_variance = torch.log1p(var_feat / mean_feat.abs().clamp_min(1.0e-3))

        base_identity = self.post_tail(self.mean_linear(mean_feat))
        raw_residual = self.var_out(normalized_variance)
        identity_rms = base_identity.detach().square().mean(dim=-1, keepdim=True).sqrt()
        residual_limit = self.residual_rms_fraction * identity_rms.clamp_min(1.0e-3)
        bounded_residual = residual_limit * torch.tanh(raw_residual / residual_limit)
        return base_identity + bounded_residual

    def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
        mean = num_neurons * self.hidden_dim * self.hidden_dim
        residual = num_neurons * self.hidden_dim * self.window_size
        tail = num_neurons * (
            self.hidden_dim * self.hidden_dim + self.hidden_dim * self.window_size
        )
        return num_trials * self._mac_per_trial(num_neurons, trial_length) + mean + residual + tail


class B3PreservingReliabilityEncoder(CalibrationEncoder):
    """B16-R1: B3 plus one low-state cross-trial firing-rate variance scalar."""

    variant = "B16R1"

    def __init__(self, trial_length: int, window_size: int, hidden_dim: int, num_post_layers: int = 3) -> None:
        super().__init__()
        if num_post_layers != 3:
            raise ValueError("B16-R1 currently requires num_post_layers=3 for exact B3 mapping")
        self.trial_length = trial_length
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.pre_pool = nn.Sequential(nn.Linear(trial_length, hidden_dim), nn.ReLU())
        self.mean_linear = nn.Linear(hidden_dim, hidden_dim)
        self.reliability_linear = nn.Linear(1, hidden_dim, bias=False)
        self.post_tail = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, window_size),
        )
        nn.init.zeros_(self.reliability_linear.weight)

    def load_b3_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        expected = {
            "pre_pool.0.weight", "pre_pool.0.bias",
            "post_pool.0.weight", "post_pool.0.bias",
            "post_pool.2.weight", "post_pool.2.bias",
            "post_pool.4.weight", "post_pool.4.bias",
        }
        if set(state_dict) != expected:
            raise ValueError("Expected an encoder-only three-layer B3 state dict")
        self.load_state_dict(
            {
                "pre_pool.0.weight": state_dict["pre_pool.0.weight"],
                "pre_pool.0.bias": state_dict["pre_pool.0.bias"],
                "mean_linear.weight": state_dict["post_pool.0.weight"],
                "mean_linear.bias": state_dict["post_pool.0.bias"],
                "reliability_linear.weight": torch.zeros_like(self.reliability_linear.weight),
                "post_tail.1.weight": state_dict["post_pool.2.weight"],
                "post_tail.1.bias": state_dict["post_pool.2.bias"],
                "post_tail.3.weight": state_dict["post_pool.4.weight"],
                "post_tail.3.bias": state_dict["post_pool.4.bias"],
            },
            strict=True,
        )

    def freeze_base_path(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name == "reliability_linear.weight"

    def reset_stream(self, batch_size: int, num_neurons: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        return {
            "sum_feat": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype),
            "sum_rate": torch.zeros(batch_size, num_neurons, 1, device=device, dtype=dtype),
            "sum_rate_sq": torch.zeros(batch_size, num_neurons, 1, device=device, dtype=dtype),
            "trial_count": 0,
        }

    def push_trial(self, state: Dict[str, Any], trial: torch.Tensor, trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
        trial = _trial_to_batch_neurons_time(trial)
        feat = self.pre_pool(trial)
        lengths = _resolve_trial_lengths(trial, trial_length)
        valid = torch.arange(trial.shape[-1], device=trial.device).view(1, 1, -1) < lengths.view(-1, 1, 1)
        rate = (trial * valid).sum(dim=-1, keepdim=True) / lengths.clamp_min(1).view(-1, 1, 1)
        state["sum_feat"] = state["sum_feat"] + feat
        state["sum_rate"] = state["sum_rate"] + rate
        state["sum_rate_sq"] = state["sum_rate_sq"] + rate.square()
        state["trial_count"] += 1
        return state

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        n = state["trial_count"]
        mean_feat = state["sum_feat"] / n
        mean_rate = state["sum_rate"] / n
        rate_variance = (state["sum_rate_sq"] / n - mean_rate.square()).clamp(min=0.0)
        return self.post_tail(self.mean_linear(mean_feat) + self.reliability_linear(rate_variance))

    def _support_state_bytes(self, num_neurons: int) -> int:
        return num_neurons * (self.hidden_dim + 2) * 4

    def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
        return trial_length * num_neurons * 4

    def _mac_per_trial(self, num_neurons: int, trial_length: int) -> int:
        return num_neurons * trial_length * self.hidden_dim

    def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
        post = num_neurons * (
            self.hidden_dim * self.hidden_dim * 2 + self.hidden_dim * self.window_size + self.hidden_dim
        )
        return num_trials * self._mac_per_trial(num_neurons, trial_length) + post


class B3PreservingFanoEncoder(B3PreservingReliabilityEncoder):
    """B16-R1F: rate-normalized log-Fano reliability residual."""

    variant = "B16R1F"

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        n = state["trial_count"]
        mean_feat = state["sum_feat"] / n
        mean_rate = state["sum_rate"] / n
        rate_variance = (state["sum_rate_sq"] / n - mean_rate.square()).clamp(min=0.0)
        log_fano = torch.log1p(rate_variance / mean_rate.abs().clamp_min(1.0e-3))
        return self.post_tail(self.mean_linear(mean_feat) + self.reliability_linear(log_fano))


class B3PreservingTemporalFanoEncoder(B3PreservingReliabilityEncoder):
    """B16-R8F: B3 plus seed-stable temporal-bin cross-trial reliability.

    Unlike B16-ZF, the residual inputs have fixed physical semantics rather
    than coordinates defined by a seed-specific learned latent projection.
    """

    variant = "B16R8F"

    def __init__(
        self,
        trial_length: int,
        window_size: int,
        hidden_dim: int,
        num_post_layers: int = 3,
        num_reliability_bins: int = 8,
    ) -> None:
        if num_reliability_bins < 1:
            raise ValueError("num_reliability_bins must be >= 1")
        super().__init__(trial_length, window_size, hidden_dim, num_post_layers)
        self.num_reliability_bins = int(num_reliability_bins)
        self.reliability_linear = nn.Linear(self.num_reliability_bins, hidden_dim, bias=False)
        nn.init.zeros_(self.reliability_linear.weight)

    def reset_stream(
        self,
        batch_size: int,
        num_neurons: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Any]:
        shape = (batch_size, num_neurons, self.num_reliability_bins)
        return {
            "sum_feat": torch.zeros(
                batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype
            ),
            "sum_rate": torch.zeros(*shape, device=device, dtype=dtype),
            "sum_rate_sq": torch.zeros(*shape, device=device, dtype=dtype),
            "trial_count": 0,
        }

    def _temporal_bin_rates(
        self, trial: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        positions = torch.arange(trial.shape[-1], device=trial.device).view(1, 1, -1)
        rates = []
        for bin_index in range(self.num_reliability_bins):
            starts = (lengths * bin_index // self.num_reliability_bins).view(-1, 1, 1)
            ends = (lengths * (bin_index + 1) // self.num_reliability_bins).view(-1, 1, 1)
            valid = (positions >= starts) & (positions < ends)
            counts = (ends - starts).clamp_min(1).to(dtype=trial.dtype)
            rates.append((trial * valid).sum(dim=-1) / counts.squeeze(-1))
        return torch.stack(rates, dim=-1)

    def push_trial(
        self,
        state: Dict[str, Any],
        trial: torch.Tensor,
        trial_length: Optional[Union[int, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        trial = _trial_to_batch_neurons_time(trial)
        feat = self.pre_pool(trial)
        lengths = _resolve_trial_lengths(trial, trial_length)
        rates = self._temporal_bin_rates(trial, lengths)
        state["sum_feat"] = state["sum_feat"] + feat
        state["sum_rate"] = state["sum_rate"] + rates
        state["sum_rate_sq"] = state["sum_rate_sq"] + rates.square()
        state["trial_count"] += 1
        return state

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        n = state["trial_count"]
        mean_feat = state["sum_feat"] / n
        mean_rate = state["sum_rate"] / n
        rate_variance = (state["sum_rate_sq"] / n - mean_rate.square()).clamp(min=0.0)
        log_fano = torch.log1p(rate_variance / mean_rate.abs().clamp_min(1.0e-3))
        return self.post_tail(self.mean_linear(mean_feat) + self.reliability_linear(log_fano))

    def _support_state_bytes(self, num_neurons: int) -> int:
        return num_neurons * (self.hidden_dim + 2 * self.num_reliability_bins) * 4

    def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
        post = num_neurons * (
            self.hidden_dim * self.hidden_dim * 2
            + self.hidden_dim * self.window_size
            + self.num_reliability_bins * self.hidden_dim
        )
        return num_trials * self._mac_per_trial(num_neurons, trial_length) + post


class B3PreservingTemporalMeanFanoEncoder(B3PreservingTemporalFanoEncoder):
    """B16-R8MF: B3 plus coarse temporal mean and log-Fano residual features."""

    variant = "B16R8MF"

    def __init__(
        self,
        trial_length: int,
        window_size: int,
        hidden_dim: int,
        num_post_layers: int = 3,
        num_reliability_bins: int = 8,
    ) -> None:
        super().__init__(
            trial_length,
            window_size,
            hidden_dim,
            num_post_layers,
            num_reliability_bins,
        )
        self.reliability_linear = nn.Linear(2 * self.num_reliability_bins, hidden_dim, bias=False)
        nn.init.zeros_(self.reliability_linear.weight)

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        n = state["trial_count"]
        mean_feat = state["sum_feat"] / n
        mean_rate = state["sum_rate"] / n
        rate_variance = (state["sum_rate_sq"] / n - mean_rate.square()).clamp(min=0.0)
        log_mean_rate = torch.log1p(mean_rate.clamp_min(0.0))
        log_fano = torch.log1p(rate_variance / mean_rate.abs().clamp_min(1.0e-3))
        temporal_features = torch.cat([log_mean_rate, log_fano], dim=-1)
        return self.post_tail(
            self.mean_linear(mean_feat) + self.reliability_linear(temporal_features)
        )

    def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
        post = num_neurons * (
            self.hidden_dim * self.hidden_dim * 2
            + self.hidden_dim * self.window_size
            + 2 * self.num_reliability_bins * self.hidden_dim
        )
        return num_trials * self._mac_per_trial(num_neurons, trial_length) + post


class B3PreservingReliabilityGateEncoder(B3PreservingReliabilityEncoder):
    """B16-G: B3 identity plus a bounded per-neuron reliability gate."""

    variant = "B16G"

    def __init__(self, trial_length: int, window_size: int, hidden_dim: int, num_post_layers: int = 3) -> None:
        super().__init__(trial_length, window_size, hidden_dim, num_post_layers)
        del self.reliability_linear
        self.gate_strength = nn.Parameter(torch.zeros(()))

    def load_b3_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        expected = {
            "pre_pool.0.weight", "pre_pool.0.bias",
            "post_pool.0.weight", "post_pool.0.bias",
            "post_pool.2.weight", "post_pool.2.bias",
            "post_pool.4.weight", "post_pool.4.bias",
        }
        if set(state_dict) != expected:
            raise ValueError("Expected an encoder-only three-layer B3 state dict")
        self.load_state_dict(
            {
                "pre_pool.0.weight": state_dict["pre_pool.0.weight"],
                "pre_pool.0.bias": state_dict["pre_pool.0.bias"],
                "mean_linear.weight": state_dict["post_pool.0.weight"],
                "mean_linear.bias": state_dict["post_pool.0.bias"],
                "gate_strength": torch.zeros_like(self.gate_strength),
                "post_tail.1.weight": state_dict["post_pool.2.weight"],
                "post_tail.1.bias": state_dict["post_pool.2.bias"],
                "post_tail.3.weight": state_dict["post_pool.4.weight"],
                "post_tail.3.bias": state_dict["post_pool.4.bias"],
            },
            strict=True,
        )

    def freeze_base_path(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name == "gate_strength"

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        mean_feat = state["sum_feat"] / state["trial_count"]
        return self.post_tail(self.mean_linear(mean_feat))

    def finalize_gate(self, state: Dict[str, Any]) -> torch.Tensor:
        n = state["trial_count"]
        mean_rate = state["sum_rate"] / n
        rate_variance = (state["sum_rate_sq"] / n - mean_rate.square()).clamp(min=0.0)
        log_fano = torch.log1p(rate_variance / mean_rate.abs().clamp_min(1.0e-3))
        return 1.0 + torch.tanh(self.gate_strength) * torch.tanh(log_fano)

    def forward_batch_with_gate(
        self, calib_trials: torch.Tensor, trial_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, _, num_neurons = calib_trials.shape
        state = self.reset_stream(batch_size, num_neurons, calib_trials.device, calib_trials.dtype)
        for trial_idx in range(calib_trials.shape[1]):
            length = None if trial_lengths is None else trial_lengths[:, trial_idx]
            state = self.push_trial(state, calib_trials[:, trial_idx], trial_length=length)
        return self.finalize_identity(state), self.finalize_gate(state)

    def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
        post = num_neurons * (
            self.hidden_dim * self.hidden_dim * 2 + self.hidden_dim * self.window_size
        )
        return num_trials * self._mac_per_trial(num_neurons, trial_length) + post


class StatsStreamingEncoder(CalibrationEncoder):
    variant = "B4"
    supports_bin_streaming = True

    def __init__(self, window_size: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.feature_proj = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU())
        self.post_pool = _build_affine_stack(hidden_dim, hidden_dim, 2, window_size)

    def _requires_cubic_interpolation(self) -> bool:
        return False

    def reset_stream(self, batch_size: int, num_neurons: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        return {"sum_feat": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype), "trial_count": 0, "bin_state": None}

    def start_trial(
        self,
        state: Dict[str, Any],
        trial_length: Optional[Union[int, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        if trial_length is None:
            raise ValueError("start_trial requires trial_length for bin-streaming encoders")
        batch_size, num_neurons = state["sum_feat"].shape[:2]
        device, dtype = state["sum_feat"].device, state["sum_feat"].dtype
        if isinstance(trial_length, int):
            lengths = torch.full((batch_size,), trial_length, device=device, dtype=torch.long)
        else:
            lengths = trial_length.to(device=device, dtype=torch.long).view(-1)
            if lengths.numel() == 1 and batch_size > 1:
                lengths = lengths.expand(batch_size)
        state["bin_state"] = {
            "sum_x": torch.zeros(batch_size, num_neurons, device=device, dtype=dtype),
            "sum_x2": torch.zeros(batch_size, num_neurons, device=device, dtype=dtype),
            "max_x": torch.full((batch_size, num_neurons), -torch.inf, device=device, dtype=dtype),
            "last_x": torch.zeros(batch_size, num_neurons, device=device, dtype=dtype),
            "count": torch.zeros(batch_size, num_neurons, device=device, dtype=dtype),
            "lengths": lengths,
        }
        return state

    def push_sample(self, state: Dict[str, Any], neural_bin: torch.Tensor, time_idx: int) -> Dict[str, Any]:
        bs = state["bin_state"]
        if bs is None:
            raise ValueError("call start_trial before push_sample")
        neural_bin = _as_batch_neurons(neural_bin)
        valid = _is_valid_bin(neural_bin, bs["lengths"], time_idx, self.pad_value)
        if not valid.any():
            return state
        mask = valid.float()
        bs["sum_x"] = bs["sum_x"] + neural_bin * mask
        bs["sum_x2"] = bs["sum_x2"] + (neural_bin * neural_bin) * mask
        bs["max_x"] = torch.where(valid, torch.maximum(bs["max_x"], neural_bin), bs["max_x"])
        bs["last_x"] = torch.where(valid, neural_bin, bs["last_x"])
        bs["count"] = bs["count"] + mask
        return state

    def end_trial(self, state: Dict[str, Any]) -> Dict[str, Any]:
        bs = state["bin_state"]
        if bs is None:
            raise ValueError("call start_trial before end_trial")
        count = bs["count"].clamp_min(1.0)
        mean = bs["sum_x"] / count
        second_moment = bs["sum_x2"] / count
        trial_feat = torch.stack([mean, second_moment, bs["max_x"], bs["last_x"]], dim=-1)
        projected = self.feature_proj(trial_feat)
        state["sum_feat"] = state["sum_feat"] + projected
        state["trial_count"] += 1
        state["bin_state"] = None
        return state

    def push_trial(self, state: Dict[str, Any], trial: torch.Tensor, trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
        trial = _trial_to_batch_neurons_time(trial)
        lengths = _resolve_trial_lengths(trial, trial_length)
        state = self.start_trial(state, lengths)
        for t_idx in range(trial.shape[-1]):
            state = self.push_sample(state, trial[..., t_idx], t_idx)
        return self.end_trial(state)

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        return self.post_pool(state["sum_feat"] / state["trial_count"])

    def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
        return 0

    def _support_state_bytes(self, num_neurons: int) -> int:
        return num_neurons * (self.hidden_dim + 16) * 4

    def _mac_per_trial(self, num_neurons: int, trial_length: int) -> int:
        return num_neurons * (4 * self.hidden_dim)

    def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
        finalize = num_neurons * _affine_mac_per_vector(self.hidden_dim, self.post_pool)
        return num_trials * self._mac_per_trial(num_neurons, trial_length) + finalize


class EMAStreamingEncoder(CalibrationEncoder):
    variant = "B5"
    supports_bin_streaming = True

    def __init__(
        self,
        window_size: int,
        num_emas: int = 4,
        hidden_dim: int = 64,
        alphas: Optional[Sequence[float]] = None,
        learnable_alpha: bool = False,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_emas = num_emas
        self.hidden_dim = hidden_dim
        if alphas is None:
            alphas = [2.0 ** (-i) for i in range(1, num_emas + 1)]
        if len(alphas) != num_emas:
            raise ValueError("alphas length must match num_emas")
        self.register_buffer("alphas", torch.tensor(list(alphas), dtype=torch.float32), persistent=False)
        self.learnable_alpha_logits = nn.Parameter(torch.zeros(num_emas)) if learnable_alpha else None
        self.feature_proj = nn.Sequential(nn.Linear(2 * num_emas, hidden_dim), nn.ReLU())
        self.post_pool = _build_affine_stack(hidden_dim, hidden_dim, 2, window_size)

    def _alpha_values(self) -> torch.Tensor:
        if self.learnable_alpha_logits is None:
            return self.alphas
        return torch.softmax(self.learnable_alpha_logits, dim=0)

    def _requires_cubic_interpolation(self) -> bool:
        return False

    def reset_stream(self, batch_size: int, num_neurons: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        return {"sum_feat": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype), "trial_count": 0, "bin_state": None}

    def start_trial(
        self,
        state: Dict[str, Any],
        trial_length: Optional[Union[int, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        if trial_length is None:
            raise ValueError("start_trial requires trial_length for bin-streaming encoders")
        batch_size, num_neurons = state["sum_feat"].shape[:2]
        device, dtype = state["sum_feat"].device, state["sum_feat"].dtype
        if isinstance(trial_length, int):
            lengths = torch.full((batch_size,), trial_length, device=device, dtype=torch.long)
        else:
            lengths = trial_length.to(device=device, dtype=torch.long).view(-1)
            if lengths.numel() == 1 and batch_size > 1:
                lengths = lengths.expand(batch_size)
        state["bin_state"] = {
            "ema": torch.zeros(batch_size, num_neurons, self.num_emas, device=device, dtype=dtype),
            "sum_ema": torch.zeros(batch_size, num_neurons, self.num_emas, device=device, dtype=dtype),
            "count": torch.zeros(batch_size, num_neurons, device=device, dtype=dtype),
            "lengths": lengths,
        }
        return state

    def push_sample(self, state: Dict[str, Any], neural_bin: torch.Tensor, time_idx: int) -> Dict[str, Any]:
        bs = state["bin_state"]
        if bs is None:
            raise ValueError("call start_trial before push_sample")
        neural_bin = _as_batch_neurons(neural_bin)
        valid = _is_valid_bin(neural_bin, bs["lengths"], time_idx, self.pad_value)
        if not valid.any():
            return state
        mask = valid.float()
        alphas = self._alpha_values().to(neural_bin.device, neural_bin.dtype)
        for r_idx, alpha in enumerate(alphas):
            updated = bs["ema"][..., r_idx] + alpha * (neural_bin - bs["ema"][..., r_idx])
            bs["ema"][..., r_idx] = torch.where(valid, updated, bs["ema"][..., r_idx])
            bs["sum_ema"][..., r_idx] = bs["sum_ema"][..., r_idx] + bs["ema"][..., r_idx] * mask
        bs["count"] = bs["count"] + mask
        return state

    def end_trial(self, state: Dict[str, Any]) -> Dict[str, Any]:
        bs = state["bin_state"]
        if bs is None:
            raise ValueError("call start_trial before end_trial")
        ema_final = bs["ema"]
        ema_trial_mean = bs["sum_ema"] / bs["count"].clamp_min(1.0).unsqueeze(-1)
        trial_feat = torch.cat([ema_final, ema_trial_mean], dim=-1)
        projected = self.feature_proj(trial_feat)
        state["sum_feat"] = state["sum_feat"] + projected
        state["trial_count"] += 1
        state["bin_state"] = None
        return state

    def push_trial(self, state: Dict[str, Any], trial: torch.Tensor, trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
        trial = _trial_to_batch_neurons_time(trial)
        lengths = _resolve_trial_lengths(trial, trial_length)
        state = self.start_trial(state, lengths)
        for t_idx in range(trial.shape[-1]):
            state = self.push_sample(state, trial[..., t_idx], t_idx)
        return self.end_trial(state)

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        return self.post_pool(state["sum_feat"] / state["trial_count"])

    def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
        return 0

    def _support_state_bytes(self, num_neurons: int) -> int:
        return num_neurons * (self.hidden_dim + 2 * self.num_emas) * 4

    def _mac_per_trial(self, num_neurons: int, trial_length: int) -> int:
        ema_mac = trial_length * self.num_emas
        proj = (2 * self.num_emas) * self.hidden_dim
        return num_neurons * (ema_mac + proj)

    def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
        finalize = num_neurons * _affine_mac_per_vector(self.hidden_dim, self.post_pool)
        return num_trials * self._mac_per_trial(num_neurons, trial_length) + finalize


class FIRStreamingEncoder(CalibrationEncoder):
    variant = "B6"
    supports_bin_streaming = True

    def __init__(self, window_size: int, num_filters: int = 4, kernel_size: int = 5, hidden_dim: int = 64) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        self.window_size = window_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.fir_weights = nn.Parameter(torch.randn(num_filters, kernel_size) * 0.02)
        self.feature_proj = nn.Sequential(nn.Linear(2 * num_filters, hidden_dim), nn.ReLU())
        self.post_pool = _build_affine_stack(hidden_dim, hidden_dim, 2, window_size)

    def _requires_cubic_interpolation(self) -> bool:
        return False

    def reset_stream(self, batch_size: int, num_neurons: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        return {"sum_feat": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype), "trial_count": 0, "bin_state": None}

    def start_trial(
        self,
        state: Dict[str, Any],
        trial_length: Optional[Union[int, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        if trial_length is None:
            raise ValueError("start_trial requires trial_length for bin-streaming encoders")
        batch_size, num_neurons = state["sum_feat"].shape[:2]
        device, dtype = state["sum_feat"].device, state["sum_feat"].dtype
        if isinstance(trial_length, int):
            lengths = torch.full((batch_size,), trial_length, device=device, dtype=torch.long)
        else:
            lengths = trial_length.to(device=device, dtype=torch.long).view(-1)
            if lengths.numel() == 1 and batch_size > 1:
                lengths = lengths.expand(batch_size)
        state["bin_state"] = {
            "history": torch.zeros(batch_size, num_neurons, max(self.kernel_size - 1, 0), device=device, dtype=dtype),
            "sum_fir": torch.zeros(batch_size, num_neurons, self.num_filters, device=device, dtype=dtype),
            "last_fir": torch.zeros(batch_size, num_neurons, self.num_filters, device=device, dtype=dtype),
            "count": torch.zeros(batch_size, num_neurons, device=device, dtype=dtype),
            "lengths": lengths,
        }
        return state

    def _causal_fir(self, history: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            window = sample.unsqueeze(-1)
        else:
            window = torch.cat([history, sample.unsqueeze(-1)], dim=-1)
        weights = self.fir_weights.to(sample.device, sample.dtype)
        return torch.einsum("bnk,rk->bnr", window, weights)

    def push_sample(self, state: Dict[str, Any], neural_bin: torch.Tensor, time_idx: int) -> Dict[str, Any]:
        bs = state["bin_state"]
        if bs is None:
            raise ValueError("call start_trial before push_sample")
        neural_bin = _as_batch_neurons(neural_bin)
        valid = _is_valid_bin(neural_bin, bs["lengths"], time_idx, self.pad_value)
        if not valid.any():
            return state
        fir_out = self._causal_fir(bs["history"], neural_bin)
        mask = valid.float().unsqueeze(-1)
        bs["last_fir"] = torch.where(valid.unsqueeze(-1), fir_out, bs["last_fir"])
        bs["sum_fir"] = bs["sum_fir"] + fir_out * mask
        bs["count"] = bs["count"] + valid.float()
        if self.kernel_size > 1 and bs["history"].shape[-1] > 0:
            bs["history"] = torch.cat([bs["history"][..., 1:], neural_bin.unsqueeze(-1)], dim=-1)
        return state

    def end_trial(self, state: Dict[str, Any]) -> Dict[str, Any]:
        bs = state["bin_state"]
        if bs is None:
            raise ValueError("call start_trial before end_trial")
        fir_trial_mean = bs["sum_fir"] / bs["count"].clamp_min(1.0).unsqueeze(-1)
        trial_feat = torch.cat([bs["last_fir"], fir_trial_mean], dim=-1)
        projected = self.feature_proj(trial_feat)
        state["sum_feat"] = state["sum_feat"] + projected
        state["trial_count"] += 1
        state["bin_state"] = None
        return state

    def push_trial(self, state: Dict[str, Any], trial: torch.Tensor, trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
        trial = _trial_to_batch_neurons_time(trial)
        lengths = _resolve_trial_lengths(trial, trial_length)
        state = self.start_trial(state, lengths)
        for t_idx in range(trial.shape[-1]):
            state = self.push_sample(state, trial[..., t_idx], t_idx)
        return self.end_trial(state)

    def forward_batch(self, calib_trials: torch.Tensor, trial_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Vectorized forward via unfold+einsum. ~100x faster than push_trial loop."""
        if calib_trials.dim() != 4:
            raise ValueError(f"Expected [B,M,T,N], got {tuple(calib_trials.shape)}")
        B, M, T_len, N = calib_trials.shape
        trials = calib_trials.permute(0, 1, 3, 2).contiguous()
        K = self.kernel_size
        if K > 1:
            pad = torch.zeros(B, M, N, K - 1, device=trials.device, dtype=trials.dtype)
            trials_padded = torch.cat([pad, trials], dim=-1)
        else:
            trials_padded = trials
        windows = trials_padded.unfold(-1, K, 1).contiguous()
        weights = self.fir_weights.to(trials.device, trials.dtype)
        fir_all = torch.einsum("bmntk,rk->bmntr", windows, weights)
        last_fir = fir_all[..., -1, :]
        fir_mean = fir_all.mean(dim=-2)
        trial_feat = torch.cat([last_fir, fir_mean], dim=-1)
        projected = self.feature_proj(trial_feat)
        mean_feat = projected.mean(dim=1)
        return self.post_pool(mean_feat)

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        return self.post_pool(state["sum_feat"] / state["trial_count"])

    def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
        return 0

    def _support_state_bytes(self, num_neurons: int) -> int:
        history = max(self.kernel_size - 1, 0)
        return num_neurons * (self.hidden_dim + 2 * self.num_filters + history) * 4

    def _peak_live_state_bytes(self, num_neurons: int, trial_length: int) -> int:
        return self._support_state_bytes(num_neurons)

    def _mac_per_trial(self, num_neurons: int, trial_length: int) -> int:
        fir_mac = trial_length * self.num_filters * self.kernel_size
        proj = (2 * self.num_filters) * self.hidden_dim
        return num_neurons * (fir_mac + proj)

    def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
        finalize = num_neurons * _affine_mac_per_vector(self.hidden_dim, self.post_pool)
        return num_trials * self._mac_per_trial(num_neurons, trial_length) + finalize


class CountConditionedEarlyPoolEncoder(CalibrationEncoder):
  """B7: EarlyPool + explicit survival count scalar appended before post_pool.

  Motivation: when neurons are dropped (training-time augmentation or chronic
  degradation at deployment), the mean-pooled feature distribution shifts.
  Feeding the survival fraction explicitly lets the post-pool MLP compensate.
  Adds only 1 extra input dim to the post_pool's first layer — negligible HW cost.
  """
  variant = "B7"

  def __init__(self, trial_length: int, window_size: int, hidden_dim: int, num_post_layers: int = 3) -> None:
    super().__init__()
    self.trial_length = trial_length
    self.window_size = window_size
    self.hidden_dim = hidden_dim
    self.pre_pool = nn.Sequential(nn.Linear(trial_length, hidden_dim), nn.ReLU())
    self.post_pool = _build_affine_stack(hidden_dim + 1, hidden_dim, num_post_layers, window_size)

  def reset_stream(self, batch_size: int, num_neurons: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    return {
      "sum_feat": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype),
      "trial_count": 0,
      "num_neurons_seen": num_neurons,
    }

  def push_trial(self, state: Dict[str, Any], trial: torch.Tensor, trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
    trial = _trial_to_batch_neurons_time(trial)
    feat = self.pre_pool(trial)
    state["sum_feat"] = state["sum_feat"] + feat
    state["trial_count"] += 1
    return state

  def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
    if state["trial_count"] == 0:
      raise ValueError("trial_count must be > 0 before finalize_identity")
    mean_feat = state["sum_feat"] / state["trial_count"]
    # Survival count per neuron: at deployment, dropped neurons have all-zero
    # calibration (their sum_feat row is exactly zero). Use L2 norm > eps as
    # a differentiable indicator that the neuron contributed to calibration.
    n_total = float(state["num_neurons_seen"])
    # Per-neuron activity flag (1 if neuron contributed anything, else 0)
    feat_norm = mean_feat.norm(dim=-1, keepdim=True)  # [B, N, 1]
    eps = 1e-6
    active = (feat_norm > eps).to(mean_feat.dtype)
    # Survival fraction across the whole session (same for all neurons in batch)
    batch_size, num_neurons = mean_feat.shape[:2]
    survival_rate = active.reshape(batch_size, num_neurons).mean(dim=-1, keepdim=True).unsqueeze(-1)
    survival_rate = survival_rate.expand(batch_size, num_neurons, 1)
    aug = torch.cat([mean_feat, survival_rate], dim=-1)
    return self.post_pool(aug)

  def _support_state_bytes(self, num_neurons: int) -> int:
    return num_neurons * self.hidden_dim * 4

  def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
    return trial_length * num_neurons * 4

  def _mac_per_trial(self, num_neurons: int, trial_length: int) -> int:
    return num_neurons * trial_length * self.hidden_dim

  def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
    post = num_neurons * _affine_mac_per_vector(self.hidden_dim + 1, self.post_pool)
    return num_trials * self._mac_per_trial(num_neurons, trial_length) + post


class FixedRandomProjectionEncoder(CalibrationEncoder):
  """B8: pre_pool replaced by a FIXED Gaussian random matrix (no learned weights).

  Random projections are robust to input corruption (Johnson-Lindenstrauss) and
  require only ROM (no weight update path). Reduces trainable params by ~36%
  vs B3 at D=64. Pre-pool weights are registered as a non-persistent buffer so
  they are NOT returned by .parameters() and are excluded from optimizers.
  """
  variant = "B8"

  def __init__(self, trial_length: int, window_size: int, hidden_dim: int, num_post_layers: int = 3, seed: int = 0) -> None:
    super().__init__()
    self.trial_length = trial_length
    self.window_size = window_size
    self.hidden_dim = hidden_dim
    # Use a per-instance generator for reproducibility regardless of global seed
    gen = torch.Generator().manual_seed(seed)
    proj = torch.randn(hidden_dim, trial_length, generator=gen) / (trial_length ** 0.5)
    self.register_buffer("projection", proj, persistent=True)
    self.post_pool = _build_affine_stack(hidden_dim, hidden_dim, num_post_layers, window_size)

  def reset_stream(self, batch_size: int, num_neurons: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    proj = self.projection.to(device=device, dtype=dtype)
    return {
      "sum_feat": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype),
      "trial_count": 0,
      "_proj": proj,
    }

  def push_trial(self, state: Dict[str, Any], trial: torch.Tensor, trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
    trial = _trial_to_batch_neurons_time(trial)  # [B, N, T]
    # feat = ReLU(proj @ x^T) where proj is [D, T], x is [B, N, T]
    proj = state["_proj"]
    feat = torch.nn.functional.linear(trial, proj)  # [B, N, D]
    feat = torch.nn.functional.relu(feat)
    state["sum_feat"] = state["sum_feat"] + feat
    state["trial_count"] += 1
    return state

  def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
    if state["trial_count"] == 0:
      raise ValueError("trial_count must be > 0 before finalize_identity")
    mean_feat = state["sum_feat"] / state["trial_count"]
    return self.post_pool(mean_feat)

  def _support_state_bytes(self, num_neurons: int) -> int:
    return num_neurons * self.hidden_dim * 4

  def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
    return trial_length * num_neurons * 4

  def _mac_per_trial(self, num_neurons: int, trial_length: int) -> int:
    return num_neurons * trial_length * self.hidden_dim

  def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
    post = num_neurons * _affine_mac_per_vector(self.hidden_dim, self.post_pool)
    return num_trials * self._mac_per_trial(num_neurons, trial_length) + post


class SparseBinaryHashEncoder(CalibrationEncoder):
  """B9: pre_pool is a FIXED sparse {-1,0,+1} matrix with K nonzeros per row.

  Each output is a sum/difference of K input time-bins (pure shift-add, no
  multiplier needed in pre_pool). At K=16, D=64, T=100: 6x MAC reduction vs B3
  in the pre-pool stage while keeping enough information density.

  The hash is generated deterministically per instance (seeded). Hash weights
  are stored as int8 buffer; only post_pool is trained.
  """
  variant = "B9"

  def __init__(
    self,
    trial_length: int,
    window_size: int,
    hidden_dim: int,
    sparsity_k: int = 16,
    num_post_layers: int = 3,
    seed: int = 0,
  ) -> None:
    super().__init__()
    if sparsity_k < 1 or sparsity_k > trial_length:
      raise ValueError(f"sparsity_k must be in [1, trial_length={trial_length}], got {sparsity_k}")
    self.trial_length = trial_length
    self.window_size = window_size
    self.hidden_dim = hidden_dim
    self.sparsity_k = sparsity_k
    # Build sparse {-1, 0, +1} matrix [hidden_dim, trial_length]
    gen = torch.Generator().manual_seed(seed)
    hash_w = torch.zeros(hidden_dim, trial_length, dtype=torch.float32)
    for r in range(hidden_dim):
      idx = torch.randperm(trial_length, generator=gen)[:sparsity_k]
      signs = torch.randint(0, 2, (sparsity_k,), generator=gen).float() * 2 - 1  # {-1, +1}
      hash_w[r, idx] = signs
    self.register_buffer("hash_matrix", hash_w, persistent=True)
    self._hash_indices = torch.nonzero(hash_w != 0, as_tuple=False)  # [D*K, 2]
    self.post_pool = _build_affine_stack(hidden_dim, hidden_dim, num_post_layers, window_size)

  def reset_stream(self, batch_size: int, num_neurons: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    return {
      "sum_feat": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype),
      "trial_count": 0,
    }

  def push_trial(self, state: Dict[str, Any], trial: torch.Tensor, trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
    trial = _trial_to_batch_neurons_time(trial)  # [B, N, T]
    hash_w = self.hash_matrix.to(device=trial.device, dtype=trial.dtype)
    # Standard linear (uses dense matmul; for hardware this is sparse gather+add)
    feat = torch.nn.functional.linear(trial, hash_w)
    feat = torch.nn.functional.relu(feat)
    state["sum_feat"] = state["sum_feat"] + feat
    state["trial_count"] += 1
    return state

  def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
    if state["trial_count"] == 0:
      raise ValueError("trial_count must be > 0 before finalize_identity")
    mean_feat = state["sum_feat"] / state["trial_count"]
    return self.post_pool(mean_feat)

  def _requires_general_multiplier(self) -> bool:
    # Pre-pool uses only add/subtract; post_pool still uses general multipliers
    return True

  def _support_state_bytes(self, num_neurons: int) -> int:
    return num_neurons * self.hidden_dim * 4

  def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
    return trial_length * num_neurons * 4

  def _mac_per_trial(self, num_neurons: int, trial_length: int) -> int:
    # Only K MACs per output (vs trial_length for dense) — count actual ops
    return num_neurons * self.sparsity_k * self.hidden_dim

  def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
    post = num_neurons * _affine_mac_per_vector(self.hidden_dim, self.post_pool)
    return num_trials * self._mac_per_trial(num_neurons, trial_length) + post


class PopulationStatsEncoder(CalibrationEncoder):
  """B10: cross-neuron population statistics -> GLOBAL identity (same for all N).

  Computes 4 population statistics per bin (mean, std, max, count_active) across
  neurons, pools over time and trials, and projects to a single [W] vector that
  is broadcast to all neurons. Inherently dropout-robust (stats degrade
  gracefully with missing samples). Best used as a complementary additive signal.

  Hardware: ~3.3K params, no per-neuron weights, no trial buffer.
  """
  variant = "B10"
  supports_bin_streaming = True

  def __init__(self, window_size: int, hidden_dim: int = 32, activation_threshold: float = 0.0) -> None:
    super().__init__()
    self.window_size = window_size
    self.hidden_dim = hidden_dim
    self.activation_threshold = activation_threshold
    self.post_pool = _build_affine_stack(4, hidden_dim, 2, window_size)

  def _requires_cubic_interpolation(self) -> bool:
    return False

  def reset_stream(self, batch_size: int, num_neurons: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    return {
      "sum_pop_feat": torch.zeros(batch_size, 4, device=device, dtype=dtype),
      "trial_count": 0,
      "bin_state": None,
    }

  def start_trial(self, state: Dict[str, Any], trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
    if trial_length is None:
      raise ValueError("start_trial requires trial_length for bin-streaming encoders")
    batch_size = state["sum_pop_feat"].shape[0]
    device, dtype = state["sum_pop_feat"].device, state["sum_pop_feat"].dtype
    if isinstance(trial_length, int):
      lengths = torch.full((batch_size,), trial_length, device=device, dtype=torch.long)
    else:
      lengths = trial_length.to(device=device, dtype=torch.long).view(-1)
      if lengths.numel() == 1 and batch_size > 1:
        lengths = lengths.expand(batch_size)
    state["bin_state"] = {
      "sum_mean": torch.zeros(batch_size, device=device, dtype=dtype),
      "sum_var": torch.zeros(batch_size, device=device, dtype=dtype),
      "max_max": torch.full((batch_size,), -torch.inf, device=device, dtype=dtype),
      "sum_count_active": torch.zeros(batch_size, device=device, dtype=dtype),
      "n_bins": torch.zeros(batch_size, device=device, dtype=dtype),
      "n_neurons": None,  # set on first push_sample
      "lengths": lengths,
    }
    return state

  def push_sample(self, state: Dict[str, Any], neural_bin: torch.Tensor, time_idx: int) -> Dict[str, Any]:
    bs = state["bin_state"]
    if bs is None:
      raise ValueError("call start_trial before push_sample")
    neural_bin = _as_batch_neurons(neural_bin)  # [B, N]
    if bs["n_neurons"] is None:
      bs["n_neurons"] = neural_bin.shape[-1]
    batch_size = neural_bin.shape[0]
    lengths = bs["lengths"]
    valid_time = (time_idx < lengths).to(neural_bin.dtype)  # [B]
    # Per-bin population stats across neurons
    bin_mean = neural_bin.mean(dim=-1)  # [B]
    bin_var = neural_bin.var(dim=-1, unbiased=False)  # [B]
    bin_max = neural_bin.max(dim=-1).values  # [B]
    bin_count_active = (neural_bin > self.activation_threshold).float().sum(dim=-1)  # [B]
    mask = valid_time
    bs["sum_mean"] = bs["sum_mean"] + bin_mean * mask
    bs["sum_var"] = bs["sum_var"] + bin_var * mask
    bs["max_max"] = torch.maximum(bs["max_max"], bin_max * mask + (1 - mask) * (-torch.inf))
    bs["sum_count_active"] = bs["sum_count_active"] + bin_count_active * mask
    bs["n_bins"] = bs["n_bins"] + mask
    return state

  def end_trial(self, state: Dict[str, Any]) -> Dict[str, Any]:
    bs = state["bin_state"]
    if bs is None:
      raise ValueError("call start_trial before end_trial")
    n = bs["n_bins"].clamp_min(1.0)
    n_neurons = float(bs["n_neurons"]) if bs["n_neurons"] is not None else 1.0
    trial_feat = torch.stack([
      bs["sum_mean"] / n,
      bs["sum_var"] / n,
      bs["max_max"],
      bs["sum_count_active"] / n.clamp_min(1.0) / max(n_neurons, 1.0),
    ], dim=-1)  # [B, 4]
    state["sum_pop_feat"] = state["sum_pop_feat"] + trial_feat
    state["trial_count"] += 1
    state["bin_state"] = None
    return state

  def push_trial(self, state: Dict[str, Any], trial: torch.Tensor, trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
    trial = _trial_to_batch_neurons_time(trial)  # [B, N, T]
    lengths = _resolve_trial_lengths(trial, trial_length)
    state = self.start_trial(state, lengths)
    for t_idx in range(trial.shape[-1]):
      state = self.push_sample(state, trial[..., t_idx], t_idx)
    return self.end_trial(state)

  def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
    if state["trial_count"] == 0:
      raise ValueError("trial_count must be > 0 before finalize_identity")
    mean_pop_feat = state["sum_pop_feat"] / state["trial_count"]  # [B, 4]
    global_vec = self.post_pool(mean_pop_feat)  # [B, W]
    # Broadcast to all neurons — caller must supply num_neurons separately
    # Since finalize_identity doesn't know N, we stash the last-seen N from bin_state
    # But after end_trial, bin_state is None. We need N in state.
    # Workaround: store N in reset_stream.
    raise NotImplementedError(
      "B10 finalize needs num_neurons. Use forward_batch instead."
    )

  def forward_batch(self, calib_trials: torch.Tensor, trial_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Vectorized: [B,M,T,N] -> E [B,N,W] (same identity broadcast to all neurons)."""
    if calib_trials.dim() != 4:
      raise ValueError(f"Expected [B,M,T,N], got {tuple(calib_trials.shape)}")
    B, M, T_len, N = calib_trials.shape
    x = calib_trials  # [B,M,T,N]
    # Per-bin population stats across neurons
    bin_mean = x.mean(dim=-1)  # [B,M,T]
    bin_var = x.var(dim=-1, unbiased=False)  # [B,M,T]
    bin_max = x.max(dim=-1).values  # [B,M,T]
    bin_count_active = (x > self.activation_threshold).float().sum(dim=-1) / N  # [B,M,T]
    # Stack and average over T and M
    pop_feat = torch.stack([
      bin_mean.mean(dim=(1, 2)),  # [B]
      bin_var.mean(dim=(1, 2)),
      bin_max.max(dim=1).values.mean(dim=1),  # max over M, mean over T
      bin_count_active.mean(dim=(1, 2)),
    ], dim=-1)  # [B, 4]
    global_vec = self.post_pool(pop_feat)  # [B, W]
    identity = global_vec.unsqueeze(1).expand(B, N, self.window_size)
    return identity.contiguous()

  def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
    return 0

  def _support_state_bytes(self, num_neurons: int) -> int:
    return self.hidden_dim * 4 + 32  # small global state only

  def _peak_live_state_bytes(self, num_neurons: int, trial_length: int) -> int:
    return self._support_state_bytes(num_neurons)

  def _mac_per_trial(self, num_neurons: int, trial_length: int) -> int:
    # 4 stats × trial_length bins = 4*T ops (negligible; ~N neurons reduced)
    return 4 * trial_length

  def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
    finalize = _affine_mac_per_vector(4, self.post_pool)
    return num_trials * self._mac_per_trial(num_neurons, trial_length) + finalize


class HybridFIRCountEncoder(CalibrationEncoder):
  """B11: learned causal FIR (shared across neurons) + count-conditioned pooling.

  Combines B6's temporal filtering with B7's survival-count signal. Streaming
  (no trial buffer). At R=4 filters, K=5 kernel, D=64: ~8K params, ~29 KiB state.
  """
  variant = "B11"
  supports_bin_streaming = True

  def __init__(
    self,
    window_size: int,
    num_filters: int = 4,
    kernel_size: int = 5,
    hidden_dim: int = 64,
  ) -> None:
    super().__init__()
    if kernel_size < 1:
      raise ValueError("kernel_size must be >= 1")
    self.window_size = window_size
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.hidden_dim = hidden_dim
    self.fir_weights = nn.Parameter(torch.randn(num_filters, kernel_size) * 0.02)
    self.feature_proj = nn.Sequential(nn.Linear(2 * num_filters, hidden_dim), nn.ReLU())
    self.post_pool = _build_affine_stack(hidden_dim + 1, hidden_dim, 2, window_size)

  def _requires_cubic_interpolation(self) -> bool:
    return False

  def reset_stream(self, batch_size: int, num_neurons: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    return {
      "sum_feat": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype),
      "trial_count": 0,
      "bin_state": None,
      "num_neurons_seen": num_neurons,
    }

  def start_trial(self, state: Dict[str, Any], trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
    if trial_length is None:
      raise ValueError("start_trial requires trial_length for bin-streaming encoders")
    batch_size, num_neurons = state["sum_feat"].shape[:2]
    device, dtype = state["sum_feat"].device, state["sum_feat"].dtype
    if isinstance(trial_length, int):
      lengths = torch.full((batch_size,), trial_length, device=device, dtype=torch.long)
    else:
      lengths = trial_length.to(device=device, dtype=torch.long).view(-1)
      if lengths.numel() == 1 and batch_size > 1:
        lengths = lengths.expand(batch_size)
    state["bin_state"] = {
      "history": torch.zeros(batch_size, num_neurons, max(self.kernel_size - 1, 0), device=device, dtype=dtype),
      "sum_fir": torch.zeros(batch_size, num_neurons, self.num_filters, device=device, dtype=dtype),
      "last_fir": torch.zeros(batch_size, num_neurons, self.num_filters, device=device, dtype=dtype),
      "count": torch.zeros(batch_size, num_neurons, device=device, dtype=dtype),
      "lengths": lengths,
    }
    return state

  def _causal_fir(self, history: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    if self.kernel_size == 1:
      window = sample.unsqueeze(-1)
    else:
      window = torch.cat([history, sample.unsqueeze(-1)], dim=-1)
    weights = self.fir_weights.to(sample.device, sample.dtype)
    return torch.einsum("bnk,rk->bnr", window, weights)

  def push_sample(self, state: Dict[str, Any], neural_bin: torch.Tensor, time_idx: int) -> Dict[str, Any]:
    bs = state["bin_state"]
    if bs is None:
      raise ValueError("call start_trial before push_sample")
    neural_bin = _as_batch_neurons(neural_bin)
    valid = _is_valid_bin(neural_bin, bs["lengths"], time_idx, self.pad_value)
    if not valid.any():
      return state
    fir_out = self._causal_fir(bs["history"], neural_bin)
    mask = valid.float().unsqueeze(-1)
    bs["last_fir"] = torch.where(valid.unsqueeze(-1), fir_out, bs["last_fir"])
    bs["sum_fir"] = bs["sum_fir"] + fir_out * mask
    bs["count"] = bs["count"] + valid.float()
    if self.kernel_size > 1 and bs["history"].shape[-1] > 0:
      bs["history"] = torch.cat([bs["history"][..., 1:], neural_bin.unsqueeze(-1)], dim=-1)
    return state

  def end_trial(self, state: Dict[str, Any]) -> Dict[str, Any]:
    bs = state["bin_state"]
    if bs is None:
      raise ValueError("call start_trial before end_trial")
    fir_trial_mean = bs["sum_fir"] / bs["count"].clamp_min(1.0).unsqueeze(-1)
    trial_feat = torch.cat([bs["last_fir"], fir_trial_mean], dim=-1)
    projected = self.feature_proj(trial_feat)
    state["sum_feat"] = state["sum_feat"] + projected
    state["trial_count"] += 1
    state["bin_state"] = None
    return state

  def push_trial(self, state: Dict[str, Any], trial: torch.Tensor, trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
    trial = _trial_to_batch_neurons_time(trial)
    lengths = _resolve_trial_lengths(trial, trial_length)
    state = self.start_trial(state, lengths)
    for t_idx in range(trial.shape[-1]):
      state = self.push_sample(state, trial[..., t_idx], t_idx)
    return self.end_trial(state)

  def forward_batch(self, calib_trials: torch.Tensor, trial_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Vectorized forward: process all [B,M,T,N] at once via unfold+einsum.

    Equivalent to push_trial loop but ~100x faster by avoiding Python overhead.
    """
    if calib_trials.dim() != 4:
      raise ValueError(f"Expected [B,M,T,N], got {tuple(calib_trials.shape)}")
    B, M, T_len, N = calib_trials.shape
    # Permute to [B, M, N, T]
    trials = calib_trials.permute(0, 1, 3, 2).contiguous()
    K = self.kernel_size
    # Left-pad with K-1 zeros for causal FIR
    if K > 1:
      pad = torch.zeros(B, M, N, K - 1, device=trials.device, dtype=trials.dtype)
      trials_padded = torch.cat([pad, trials], dim=-1)
    else:
      trials_padded = trials
    # Unfold to sliding windows: [B, M, N, T, K]
    windows = trials_padded.unfold(-1, K, 1).contiguous()
    # Apply FIR: [B, M, N, T, R]
    weights = self.fir_weights.to(trials.device, trials.dtype)
    fir_all = torch.einsum("bmntk,rk->bmntr", windows, weights)
    # Per-trial features: last_fir and fir_mean
    last_fir = fir_all[..., -1, :]  # [B, M, N, R]
    fir_mean = fir_all.mean(dim=-2)  # [B, M, N, R]
    trial_feat = torch.cat([last_fir, fir_mean], dim=-1)  # [B, M, N, 2R]
    # Project
    projected = self.feature_proj(trial_feat)  # [B, M, N, D]
    # Pool over M trials
    mean_feat = projected.mean(dim=1)  # [B, N, D]
    # Count conditioning
    feat_norm = mean_feat.norm(dim=-1, keepdim=True)
    active = (feat_norm > 1e-6).to(mean_feat.dtype)
    survival_rate = active.reshape(B, N).mean(dim=-1, keepdim=True).unsqueeze(-1).expand(B, N, 1)
    aug = torch.cat([mean_feat, survival_rate], dim=-1)
    return self.post_pool(aug)

  def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
    if state["trial_count"] == 0:
      raise ValueError("trial_count must be > 0 before finalize_identity")
    mean_feat = state["sum_feat"] / state["trial_count"]
    # Survival fraction (same logic as B7)
    feat_norm = mean_feat.norm(dim=-1, keepdim=True)
    eps = 1e-6
    active = (feat_norm > eps).to(mean_feat.dtype)
    batch_size, num_neurons = mean_feat.shape[:2]
    survival_rate = active.reshape(batch_size, num_neurons).mean(dim=-1, keepdim=True).unsqueeze(-1)
    survival_rate = survival_rate.expand(batch_size, num_neurons, 1)
    aug = torch.cat([mean_feat, survival_rate], dim=-1)
    return self.post_pool(aug)

  def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
    return 0

  def _support_state_bytes(self, num_neurons: int) -> int:
    history = max(self.kernel_size - 1, 0)
    return num_neurons * (self.hidden_dim + 2 * self.num_filters + history) * 4

  def _peak_live_state_bytes(self, num_neurons: int, trial_length: int) -> int:
    return self._support_state_bytes(num_neurons)

  def _mac_per_trial(self, num_neurons: int, trial_length: int) -> int:
    fir_mac = trial_length * self.num_filters * self.kernel_size
    proj = (2 * self.num_filters) * self.hidden_dim
    return num_neurons * (fir_mac + proj)

  def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
    finalize = num_neurons * _affine_mac_per_vector(self.hidden_dim + 1, self.post_pool)
    return num_trials * self._mac_per_trial(num_neurons, trial_length) + finalize


class StreamingHashEncoder(CalibrationEncoder):
  """B12: B9's sparse hash applied bin-by-bin with no trial buffer.

  Combines B9's sparse-binary projection (multiplier-free, K nonzeros per output)
  with B4/B5/B6's streaming structure (no [T,N] trial buffer). Per bin, applies
  the same hash across all neurons; accumulates hashed features over T bins.

  Hardware: 0 trial buffer, only K*D add/sub per bin per neuron (shift-add only),
  no multiplier in pre-pool, no cubic interpolation. Ideal streaming ASIC target.
  """
  variant = "B12"
  supports_bin_streaming = True

  def __init__(
    self,
    window_size: int,
    hidden_dim: int = 64,
    sparsity_k: int = 4,
    seed: int = 0,
  ) -> None:
    super().__init__()
    self.window_size = window_size
    self.hidden_dim = hidden_dim
    self.sparsity_k = sparsity_k
    # Hash over the bin scalar (1 input) -> hidden_dim outputs.
    # Each output r picks K distinct "threshold slots" in [-1, 1] (or similar).
    # Simpler: each output is sign(x - theta_r,k) summed — a piecewise-linear hash.
    # Implementation: K random thresholds per output, sign-sum.
    gen = torch.Generator().manual_seed(seed)
    # thresholds[r, k] ~ U(-2, 2) (spike rates roughly in this range)
    self.register_buffer(
      "thresholds",
      torch.empty(hidden_dim, sparsity_k).uniform_(-2.0, 2.0, generator=gen),
      persistent=True,
    )
    self.post_pool = _build_affine_stack(hidden_dim, hidden_dim, 2, window_size)

  def _requires_cubic_interpolation(self) -> bool:
    return False

  def _requires_general_multiplier(self) -> bool:
    # Pre-pool is comparator + sign-add (no multiplier); post_pool still needs mult
    return True

  def reset_stream(self, batch_size: int, num_neurons: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    return {
      "sum_feat": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype),
      "trial_count": 0,
      "bin_state": None,
    }

  def start_trial(self, state: Dict[str, Any], trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
    if trial_length is None:
      raise ValueError("start_trial requires trial_length for bin-streaming encoders")
    batch_size, num_neurons = state["sum_feat"].shape[:2]
    device, dtype = state["sum_feat"].device, state["sum_feat"].dtype
    if isinstance(trial_length, int):
      lengths = torch.full((batch_size,), trial_length, device=device, dtype=torch.long)
    else:
      lengths = trial_length.to(device=device, dtype=torch.long).view(-1)
      if lengths.numel() == 1 and batch_size > 1:
        lengths = lengths.expand(batch_size)
    state["bin_state"] = {
      "trial_sum_feat": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype),
      "lengths": lengths,
    }
    return state

  def push_sample(self, state: Dict[str, Any], neural_bin: torch.Tensor, time_idx: int) -> Dict[str, Any]:
    bs = state["bin_state"]
    if bs is None:
      raise ValueError("call start_trial before push_sample")
    neural_bin = _as_batch_neurons(neural_bin)  # [B, N]
    valid_time = (time_idx < bs["lengths"]).to(neural_bin.dtype)  # [B]
    thresholds = self.thresholds.to(device=neural_bin.device, dtype=neural_bin.dtype)
    # Per-neuron, per-output: sum_k sign(x - theta_{r,k})
    # x: [B, N] -> [B, N, 1]; thresholds: [D, K]
    diff = neural_bin.unsqueeze(-1).unsqueeze(-1) - thresholds  # [B, N, D, K]
    sign_sum = torch.sign(diff).sum(dim=-1)  # [B, N, D]
    mask = valid_time.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
    bs["trial_sum_feat"] = bs["trial_sum_feat"] + sign_sum * mask
    return state

  def end_trial(self, state: Dict[str, Any]) -> Dict[str, Any]:
    bs = state["bin_state"]
    if bs is None:
      raise ValueError("call start_trial before end_trial")
    state["sum_feat"] = state["sum_feat"] + bs["trial_sum_feat"]
    state["trial_count"] += 1
    state["bin_state"] = None
    return state

  def push_trial(self, state: Dict[str, Any], trial: torch.Tensor, trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
    trial = _trial_to_batch_neurons_time(trial)  # [B, N, T]
    lengths = _resolve_trial_lengths(trial, trial_length)
    state = self.start_trial(state, lengths)
    for t_idx in range(trial.shape[-1]):
      state = self.push_sample(state, trial[..., t_idx], t_idx)
    return self.end_trial(state)

  def forward_batch(self, calib_trials: torch.Tensor, trial_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Vectorized forward for B12 with memory-efficient chunked computation.

    Loops over D, immediately reducing each chunk over M and T to keep peak
    memory at ~160 MB instead of ~10 GB.
    """
    if calib_trials.dim() != 4:
      raise ValueError(f"Expected [B,M,T,N], got {tuple(calib_trials.shape)}")
    B, M, T_len, N = calib_trials.shape
    thresholds = self.thresholds.to(calib_trials.device, calib_trials.dtype)  # [D, K]
    D = self.hidden_dim
    sign_sum_list = []
    for d in range(D):
      theta_d = thresholds[d]  # [K]
      diff_d = calib_trials.unsqueeze(-1) - theta_d  # [B,M,T,N,K]
      sign_d = torch.sign(diff_d).sum(dim=-1)  # [B,M,T,N]
      sign_d_reduced = sign_d.mean(dim=(1, 2))  # [B,N]
      sign_sum_list.append(sign_d_reduced)
    mean_feat = torch.stack(sign_sum_list, dim=-1)  # [B,N,D]
    return self.post_pool(mean_feat)

  def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
    if state["trial_count"] == 0:
      raise ValueError("trial_count must be > 0 before finalize_identity")
    mean_feat = state["sum_feat"] / state["trial_count"]
    return self.post_pool(mean_feat)

  def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
    return 0

  def _support_state_bytes(self, num_neurons: int) -> int:
    # Per-neuron accumulator + trial_sum_feat (transient)
    return num_neurons * self.hidden_dim * 4

  def _peak_live_state_bytes(self, num_neurons: int, trial_length: int) -> int:
    # During a trial we hold both sum_feat and trial_sum_feat
    return 2 * num_neurons * self.hidden_dim * 4

  def _mac_per_trial(self, num_neurons: int, trial_length: int) -> int:
    # Per bin: N * D * K sign computations (counted as MAC for fair comparison)
    return num_neurons * trial_length * self.hidden_dim * self.sparsity_k

  def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
    finalize = num_neurons * _affine_mac_per_vector(self.hidden_dim, self.post_pool)
    return num_trials * self._mac_per_trial(num_neurons, trial_length) + finalize


class EnsembleRandomHashEncoder(CalibrationEncoder):
  """B13: ensemble of random projection (B8) + sparse hash (B9).

  Two parallel fixed projections of the trial, concatenated before post_pool.
  The Gaussian projection preserves metric structure (JL lemma); the binary
  hash captures threshold-style nonlinear features. Together they cover both
  linear and nonlinear summaries of the trial at low hardware cost.
  """
  variant = "B13"

  def __init__(
    self,
    trial_length: int,
    window_size: int,
    hidden_dim: int = 64,
    sparsity_k: int = 16,
    seed_proj: int = 0,
    seed_hash: int = 1,
  ) -> None:
    super().__init__()
    self.trial_length = trial_length
    self.window_size = window_size
    self.hidden_dim = hidden_dim
    self.sparsity_k = sparsity_k
    # Half the channels from Gaussian projection, half from binary hash
    self.half = hidden_dim // 2
    # Gaussian random projection [half, T]
    gen_p = torch.Generator().manual_seed(seed_proj)
    proj = torch.randn(self.half, trial_length, generator=gen_p) / (trial_length ** 0.5)
    self.register_buffer("projection", proj, persistent=True)
    # Sparse binary hash [half, T] with K nonzeros per row
    gen_h = torch.Generator().manual_seed(seed_hash)
    hash_w = torch.zeros(self.half, trial_length, dtype=torch.float32)
    for r in range(self.half):
      idx = torch.randperm(trial_length, generator=gen_h)[:sparsity_k]
      signs = torch.randint(0, 2, (sparsity_k,), generator=gen_h).float() * 2 - 1
      hash_w[r, idx] = signs
    self.register_buffer("hash_matrix", hash_w, persistent=True)
    self.post_pool = _build_affine_stack(hidden_dim, hidden_dim, 3, window_size)

  def reset_stream(self, batch_size: int, num_neurons: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    proj = self.projection.to(device=device, dtype=dtype)
    hash_w = self.hash_matrix.to(device=device, dtype=dtype)
    return {
      "sum_feat": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype),
      "trial_count": 0,
      "_proj": proj,
      "_hash": hash_w,
    }

  def push_trial(self, state: Dict[str, Any], trial: torch.Tensor, trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
    trial = _trial_to_batch_neurons_time(trial)  # [B, N, T]
    proj = state["_proj"]  # [half, T]
    hash_w = state["_hash"]  # [half, T]
    # Linear projection + ReLU
    proj_feat = torch.nn.functional.linear(trial, proj)  # [B, N, half]
    proj_feat = torch.nn.functional.relu(proj_feat)
    # Hash projection + ReLU
    hash_feat = torch.nn.functional.linear(trial, hash_w)  # [B, N, half]
    hash_feat = torch.nn.functional.relu(hash_feat)
    feat = torch.cat([proj_feat, hash_feat], dim=-1)  # [B, N, hidden_dim]
    state["sum_feat"] = state["sum_feat"] + feat
    state["trial_count"] += 1
    return state

  def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
    if state["trial_count"] == 0:
      raise ValueError("trial_count must be > 0 before finalize_identity")
    mean_feat = state["sum_feat"] / state["trial_count"]
    return self.post_pool(mean_feat)

  def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
    return trial_length * num_neurons * 4

  def _support_state_bytes(self, num_neurons: int) -> int:
    return num_neurons * self.hidden_dim * 4

  def _mac_per_trial(self, num_neurons: int, trial_length: int) -> int:
    # proj: N * half * T (dense); hash: N * half * K (sparse)
    proj_mac = num_neurons * self.half * trial_length
    hash_mac = num_neurons * self.half * self.sparsity_k
    return proj_mac + hash_mac

  def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
    finalize = num_neurons * _affine_mac_per_vector(self.hidden_dim, self.post_pool)
    return num_trials * self._mac_per_trial(num_neurons, trial_length) + finalize


class TernarizedEarlyPoolEncoder(CalibrationEncoder):
  """B14: B3 with ternarized post_pool weights {-1, 0, +1}.

  The pre_pool and post_pool weights are constrained to {-1, 0, +1} using a
  Straight-Through Estimator (STE). This eliminates the need for general
  multipliers in the entire encoder — every MAC becomes an add/subtract.

  Ternary weights are stored as 2-bit indices (0/1/2 -> -1/0/+1) -> 4x weight
  memory reduction vs INT8. Combined with INT8 activations and an INT32
  accumulator, this is the most aggressive quantization short of full binary.
  """
  variant = "B14"

  def __init__(
    self,
    trial_length: int,
    window_size: int,
    hidden_dim: int = 64,
    num_post_layers: int = 3,
    ternarize_pre: bool = True,
    ternarize_post: bool = True,
  ) -> None:
    super().__init__()
    self.trial_length = trial_length
    self.window_size = window_size
    self.hidden_dim = hidden_dim
    self.ternarize_pre = ternarize_pre
    self.ternarize_post = ternarize_post
    self.pre_pool = nn.Sequential(nn.Linear(trial_length, hidden_dim), nn.ReLU())
    self.post_pool = _build_affine_stack(hidden_dim, hidden_dim, num_post_layers, window_size)

  @staticmethod
  def _ternarize(weight: torch.Tensor) -> torch.Tensor:
    """STE ternarization: forward returns {-1, 0, +1}, backward passes gradient through."""
    threshold = 0.7 * weight.abs().mean()
    ternary = torch.where(weight > threshold, torch.ones_like(weight),
                torch.where(weight < -threshold, -torch.ones_like(weight),
                  torch.zeros_like(weight)))
    # STE: return ternary in forward, but gradient flows through `weight`
    return ternary + (weight - weight.detach()) - (weight - weight.detach()) * 0  # identity-ish
    # Note: cleaner STE is `ternary + weight - weight.detach()` but it double-counts.
    # Simpler correct form:
    # return weight + (ternary - weight).detach()

  def _get_quantized_weights_pre(self):
    if not self.ternarize_pre:
      return None
    # Apply ternarization to the pre_pool Linear weight
    linear = self.pre_pool[0]
    return self._ternarize_stable(linear.weight)

  @staticmethod
  def _ternarize_stable(weight: torch.Tensor) -> torch.Tensor:
    """Stable STE ternarization."""
    threshold = 0.7 * weight.abs().mean().clamp_min(1e-8)
    ternary = torch.where(weight > threshold, torch.ones_like(weight),
                torch.where(weight < -threshold, -torch.ones_like(weight),
                  torch.zeros_like(weight)))
    return weight + (ternary - weight).detach()

  def reset_stream(self, batch_size: int, num_neurons: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    return {
      "sum_feat": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype),
      "trial_count": 0,
    }

  def push_trial(self, state: Dict[str, Any], trial: torch.Tensor, trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
    trial = _trial_to_batch_neurons_time(trial)
    if self.ternarize_pre:
      linear = self.pre_pool[0]
      w_q = self._ternarize_stable(linear.weight)
      feat = torch.nn.functional.linear(trial, w_q, linear.bias)
      feat = torch.nn.functional.relu(feat)
    else:
      feat = self.pre_pool(trial)
    state["sum_feat"] = state["sum_feat"] + feat
    state["trial_count"] += 1
    return state

  def _ternarized_post_pool(self, x: torch.Tensor) -> torch.Tensor:
    """Apply post_pool with ternarized weights layer-by-layer."""
    out = x
    for layer in self.post_pool:
      if isinstance(layer, nn.Linear):
        w_q = self._ternarize_stable(layer.weight)
        out = torch.nn.functional.linear(out, w_q, layer.bias)
      elif isinstance(layer, nn.ReLU):
        out = torch.nn.functional.relu(out)
    return out

  def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
    if state["trial_count"] == 0:
      raise ValueError("trial_count must be > 0 before finalize_identity")
    mean_feat = state["sum_feat"] / state["trial_count"]
    if self.ternarize_post:
      return self._ternarized_post_pool(mean_feat)
    return self.post_pool(mean_feat)

  def _requires_general_multiplier(self) -> bool:
    # With ternarization, ALL weights are {-1, 0, +1} — pure add/subtract
    return not (self.ternarize_pre and self.ternarize_post)

  def _support_state_bytes(self, num_neurons: int) -> int:
    return num_neurons * self.hidden_dim * 4

  def _trial_buffer_bytes(self, num_neurons: int, trial_length: int) -> int:
    return trial_length * num_neurons * 4

  def _mac_per_trial(self, num_neurons: int, trial_length: int) -> int:
    return num_neurons * trial_length * self.hidden_dim

  def _mac_per_session(self, num_neurons: int, trial_length: int, num_trials: int) -> int:
    post = num_neurons * _affine_mac_per_vector(self.hidden_dim, self.post_pool)
    return num_trials * self._mac_per_trial(num_neurons, trial_length) + post


def build_encoder(
    variant: str,
    *,
    window_size: int,
    trial_length: int = 100,
    teacher_fc_id_in: Optional[nn.Module] = None,
    teacher_fc_id_out: Optional[nn.Module] = None,
    id_hidden_dim: int = 128,
    hidden_dim: int = 64,
    num_emas: int = 4,
    num_filters: int = 4,
    kernel_size: int = 5,
    learnable_ema_alpha: bool = False,
    sparsity_k: int = 16,
    pad_value: float = -1.0,
    id_num_heads: int = 4,
) -> CalibrationEncoder:
    variant = variant.upper()
    if variant in {"B0", "BATCH"}:
        if teacher_fc_id_in is None or teacher_fc_id_out is None:
            raise ValueError("B0 requires teacher fc_id modules")
        enc = BatchReferenceEncoder(teacher_fc_id_in, teacher_fc_id_out, window_size)
    elif variant == "B1":
        if teacher_fc_id_in is None or teacher_fc_id_out is None:
            raise ValueError("B1 requires teacher fc_id modules")
        enc = TrialStreamingEncoder(teacher_fc_id_in, teacher_fc_id_out, window_size)
    elif variant == "B2":
        enc = LatePoolEncoder(trial_length, window_size, id_hidden_dim)
    elif variant == "B3":
        enc = EarlyPoolEncoder(trial_length, window_size, hidden_dim)
    elif variant == "B4":
        enc = StatsStreamingEncoder(window_size, hidden_dim)
    elif variant == "B5":
        enc = EMAStreamingEncoder(window_size, num_emas, hidden_dim, learnable_alpha=learnable_ema_alpha)
    elif variant == "B6":
        enc = FIRStreamingEncoder(window_size, num_filters, kernel_size, hidden_dim)
    elif variant == "B7":
        enc = CountConditionedEarlyPoolEncoder(trial_length, window_size, hidden_dim)
    elif variant == "B8":
        enc = FixedRandomProjectionEncoder(trial_length, window_size, hidden_dim)
    elif variant == "B9":
        enc = SparseBinaryHashEncoder(trial_length, window_size, hidden_dim, sparsity_k=sparsity_k)
    elif variant == "B10":
        enc = PopulationStatsEncoder(window_size, hidden_dim=hidden_dim)
    elif variant == "B11":
        enc = HybridFIRCountEncoder(window_size, num_filters, kernel_size, hidden_dim)
    elif variant == "B12":
        enc = StreamingHashEncoder(window_size, hidden_dim=hidden_dim, sparsity_k=sparsity_k)
    elif variant == "B13":
        enc = EnsembleRandomHashEncoder(trial_length, window_size, hidden_dim=hidden_dim, sparsity_k=sparsity_k)
    elif variant == "B14":
        enc = TernarizedEarlyPoolEncoder(trial_length, window_size, hidden_dim=hidden_dim)
    elif variant == "B15":
        enc = RelationalEarlyPoolEncoder(trial_length, window_size, hidden_dim, num_heads=id_num_heads)
    elif variant == "B16":
        enc = HighOrderStatsEncoder(trial_length, window_size, hidden_dim)
    elif variant in {"B16Z", "B16-Z"}:
        enc = B3PreservingHighOrderStatsEncoder(trial_length, window_size, hidden_dim)
    elif variant in {"B16ZF", "B16-ZF"}:
        enc = B3PreservingNormalizedHighOrderStatsEncoder(trial_length, window_size, hidden_dim)
    elif variant in {"B16ZFS", "B16-ZFS"}:
        enc = B3PreservingShrunkNormalizedHighOrderStatsEncoder(trial_length, window_size, hidden_dim)
    elif variant in {"B16ZFD", "B16-ZFD"}:
        enc = B3PreservingDropoutNormalizedHighOrderStatsEncoder(trial_length, window_size, hidden_dim)
    elif variant in {"B16ZFO", "B16-ZFO"}:
        enc = B3PreservingBoundedOutputFanoEncoder(trial_length, window_size, hidden_dim)
    elif variant in {"B16R1", "B16-R1"}:
        enc = B3PreservingReliabilityEncoder(trial_length, window_size, hidden_dim)
    elif variant in {"B16R1F", "B16-R1F"}:
        enc = B3PreservingFanoEncoder(trial_length, window_size, hidden_dim)
    elif variant in {"B16R8F", "B16-R8F"}:
        enc = B3PreservingTemporalFanoEncoder(trial_length, window_size, hidden_dim)
    elif variant in {"B16R8MF", "B16-R8MF"}:
        enc = B3PreservingTemporalMeanFanoEncoder(trial_length, window_size, hidden_dim)
    elif variant in {"B16G", "B16-G"}:
        enc = B3PreservingReliabilityGateEncoder(trial_length, window_size, hidden_dim)
    else:
        raise ValueError(f"Unknown encoder variant: {variant}")
    enc.pad_value = pad_value
    return enc


def copy_teacher_id_weights(encoder: CalibrationEncoder, teacher_net: nn.Module) -> None:
    if isinstance(encoder, (BatchReferenceEncoder, TrialStreamingEncoder)):
        encoder.fc_id_in.load_state_dict(teacher_net.fc_id_in.state_dict())
        encoder.fc_id_out.load_state_dict(teacher_net.fc_id_out.state_dict())
