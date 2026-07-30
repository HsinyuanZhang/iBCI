"""Minimal B3 EarlyPoolEncoder for optional torch cross-check.

Self-contained copy of the B3 topology used by the streaming calibration
experiment. No imports from the original SPINT repository.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


def build_affine_stack(
    in_dim: int,
    hidden_dim: int,
    num_affine_layers: int,
    out_dim: int,
) -> nn.Sequential:
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


def trial_to_batch_neurons_time(trial: torch.Tensor) -> torch.Tensor:
    """trial [B,T,N] or [T,N] -> [B,N,T]."""
    if trial.dim() == 2:
        trial = trial.unsqueeze(0)
    if trial.dim() != 3:
        raise ValueError(f"Expected trial [B,T,N] or [T,N], got {tuple(trial.shape)}")
    return trial.permute(0, 2, 1)


class EarlyPoolEncoder(nn.Module):
    """B3: pre Linear(T->D)+ReLU, mean over trials, post 3-layer MLP -> W."""

    variant = "B3"

    def __init__(
        self,
        trial_length: int,
        window_size: int,
        hidden_dim: int,
        num_post_layers: int = 3,
    ) -> None:
        super().__init__()
        self.trial_length = trial_length
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.pre_pool = nn.Sequential(nn.Linear(trial_length, hidden_dim), nn.ReLU())
        self.post_pool = build_affine_stack(hidden_dim, hidden_dim, num_post_layers, window_size)

    def reset_stream(
        self,
        batch_size: int,
        num_neurons: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Any]:
        return {
            "sum_feat": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype),
            "trial_count": 0,
        }

    def push_trial(
        self,
        state: Dict[str, Any],
        trial: torch.Tensor,
        trial_length: Optional[Union[int, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        del trial_length
        trial = trial_to_batch_neurons_time(trial)
        feat = self.pre_pool(trial)
        state["sum_feat"] = state["sum_feat"] + feat
        state["trial_count"] += 1
        return state

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        return self.post_pool(state["sum_feat"] / state["trial_count"])

    def forward_batch(self, calib_trials: torch.Tensor) -> torch.Tensor:
        """calib_trials: [B, M, T, N] -> E [B, N, W]."""
        if calib_trials.dim() != 4:
            raise ValueError(f"Expected [B,M,T,N], got {tuple(calib_trials.shape)}")
        batch_size, _, _, num_neurons = calib_trials.shape
        state = self.reset_stream(batch_size, num_neurons, calib_trials.device, calib_trials.dtype)
        for trial_idx in range(calib_trials.shape[1]):
            state = self.push_trial(state, calib_trials[:, trial_idx])
        return self.finalize_identity(state)
