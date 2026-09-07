"""Zero-init hold-vs-reach FiLM on frozen B3S+T4. post_pool stays 64+4."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

_STREAMING_ROOT = Path(__file__).resolve().parents[3] / "streaming_calibration_exp"
_STREAMING = str(_STREAMING_ROOT)
if _STREAMING not in sys.path:
    sys.path.insert(0, _STREAMING)

from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder

from . import plan


class HoldContrastFiLMEarlyPoolEncoder(SideFeatureEarlyPoolEncoder):
    """B3S substrate with calib-only hold-vs-reach contrast FiLM on pooled ``h``.

    ``side_features`` is ``[T4(4), contrast(4)]``.  Contrast never enters
    ``post_pool``; T4 still concatenates after FiLM so the geometry remains
    ``[h', T4]`` (64+4).  Zero-init FiLM is bitwise native T4.
    """

    variant = "B3SHC"
    t4_dim = plan.T4_DIM
    contrast_dim = plan.CONTRAST_DIM

    def __init__(
        self,
        trial_length: int = plan.TRIAL_LENGTH,
        window_size: int = plan.WINDOW_SIZE,
        hidden_dim: int = plan.HIDDEN_DIM,
        side_dim: int = plan.SIDE_DIM,
        film_rank: int = plan.FILM_RANK,
        num_post_layers: int = 3,
        film_input: str = "t4_plus_contrast",
    ) -> None:
        if side_dim != self.t4_dim + self.contrast_dim:
            raise ValueError("hold-contrast FiLM requires side_features=[T4(4), contrast(4)]")
        if film_rank <= 0:
            raise ValueError("film_rank must be positive")
        if film_input not in {"t4_plus_contrast", "contrast_only"}:
            raise ValueError("film_input must be t4_plus_contrast or contrast_only")
        super().__init__(
            trial_length,
            window_size,
            hidden_dim,
            side_dim=self.t4_dim,
            electrode_embed_dim=0,
            num_electrodes=0,
            num_post_layers=num_post_layers,
        )
        self.input_side_dim = side_dim
        self.film_rank = int(film_rank)
        self.film_input = film_input
        context_in = self.contrast_dim if film_input == "contrast_only" else self.t4_dim + self.contrast_dim
        self.contrast_context = nn.Sequential(
            nn.Linear(context_in, self.film_rank),
            nn.ReLU(),
        )
        self.contrast_film = nn.Linear(self.film_rank, 2 * hidden_dim)
        with torch.no_grad():
            self.contrast_film.weight.zero_()
            self.contrast_film.bias.zero_()

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before hold-contrast finalization")
        side = state.get("side_features")
        mean_feat = state["sum_feat"] / state["trial_count"]
        expected_shape = (*mean_feat.shape[:2], self.input_side_dim)
        if side is None or tuple(side.shape) != expected_shape:
            raise ValueError(
                f"hold-contrast FiLM requires side_features shape {expected_shape}, "
                f"got {None if side is None else tuple(side.shape)}"
            )
        t4 = side[..., : self.t4_dim]
        film_in = side[..., self.t4_dim :] if self.film_input == "contrast_only" else side
        modulation = self.contrast_film(
            self.contrast_context(film_in)
        )
        gamma, beta = modulation.chunk(2, dim=-1)
        mean_feat = (1.0 + gamma) * mean_feat + beta
        return self.post_pool(torch.cat([mean_feat, t4], dim=-1))

    def load_t4_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        base_keys = set(self.state_dict()) - {
            "contrast_context.0.weight",
            "contrast_context.0.bias",
            "contrast_film.weight",
            "contrast_film.bias",
        }
        if set(state_dict) != base_keys:
            raise ValueError(
                "hold-contrast warm-start must be an exact ordinary B3S/T4 encoder state dict"
            )
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        expected_missing = {
            "contrast_context.0.weight",
            "contrast_context.0.bias",
            "contrast_film.weight",
            "contrast_film.bias",
        }
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError("hold-contrast T4 warm-start violated its exact state mapping")
        if torch.count_nonzero(self.contrast_film.weight).item() or torch.count_nonzero(
            self.contrast_film.bias
        ).item():
            raise RuntimeError("hold-contrast warm-start must leave the zero-init FiLM head at zero")

    def freeze_base_path(self) -> None:
        trainable_prefixes = ("contrast_context.", "contrast_film.")
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith(trainable_prefixes)

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)
