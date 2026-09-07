"""B3S subclass that applies activity-path dropout inside ``finalize_identity``."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from src.models.components.activity_path_dropout import (
  ActivityPathDropoutStrategy,
  NoActivityPathDropout,
  apply_activity_path_mask,
  build_activity_path_dropout,
)
from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder


class ActivityPathDropoutSideFeatureEarlyPoolEncoder(SideFeatureEarlyPoolEncoder):
  """B3S + optional activity-path dropout before side-feature concatenation."""

  variant = "B3S_APD"

  def __init__(
    self,
    trial_length: int,
    window_size: int,
    hidden_dim: int,
    side_dim: int = 0,
    electrode_embed_dim: int = 0,
    num_electrodes: int = 0,
    num_post_layers: int = 3,
    *,
    activity_path_dropout_p: float = 0.0,
  ) -> None:
    super().__init__(
      trial_length,
      window_size,
      hidden_dim,
      side_dim=side_dim,
      electrode_embed_dim=electrode_embed_dim,
      num_electrodes=num_electrodes,
      num_post_layers=num_post_layers,
    )
    self._activity_path_dropout: ActivityPathDropoutStrategy = build_activity_path_dropout(
      p=activity_path_dropout_p
    )

  def set_activity_path_dropout(self, strategy: ActivityPathDropoutStrategy) -> None:
    self._activity_path_dropout = strategy

  def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
    if state["trial_count"] == 0:
      raise ValueError("trial_count must be > 0 before finalize_identity")
    mean_feat = state["sum_feat"] / state["trial_count"]
    if self.training and not isinstance(self._activity_path_dropout, NoActivityPathDropout):
      batch_size = mean_feat.shape[0]
      mask = self._activity_path_dropout.sample_mask(batch_size, mean_feat.device)
      mean_feat = apply_activity_path_mask(mean_feat, mask)
    side_parts: list[torch.Tensor] = []
    side = state.get("side_features")
    if self.side_dim > 0:
      if side is None:
        raise ValueError("B3S requires side_features when side_dim > 0")
      if side.shape[-1] != self.side_dim:
        raise ValueError(
          f"B3S expected side_features with last dim {self.side_dim}, got {tuple(side.shape)}"
        )
      if side.shape[:-1] != mean_feat.shape[:-1]:
        raise ValueError(
          "B3S side_features batch/neuron shape "
          f"{tuple(side.shape[:-1])} does not match pooled features {tuple(mean_feat.shape[:-1])}"
        )
      side_parts.append(side)
    electrode_ids = state.get("electrode_ids")
    if self.electrode_embed_dim > 0:
      if electrode_ids is None:
        raise ValueError("B3S requires electrode_ids when electrode_embed_dim > 0")
      if electrode_ids.shape != mean_feat.shape[:2]:
        raise ValueError(
          "B3S electrode_ids shape "
          f"{tuple(electrode_ids.shape)} does not match pooled features "
          f"{tuple(mean_feat.shape[:2])}"
        )
      assert self.electrode_embed is not None
      side_parts.append(self.electrode_embed(electrode_ids.long()))
    if side_parts:
      mean_feat = torch.cat([mean_feat, *side_parts], dim=-1)
    return self.post_pool(mean_feat)
