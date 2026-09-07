"""Additive B3S wrapper for the bounded misleading-identity swap-v2 scaffold.

No shared factory imports this class.  A dedicated experiment wrapper must opt in
and attach a verified authority.  Clean mode delegates directly to the production
B3S implementation, preserving its exact forward path.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

import torch

from mc_maze.misleading_identity_swap import apply_activity_identity_swap
from mc_maze.misleading_identity_swap_v2_core import (
    VerifiedMatchingAuthority,
    permutation_sha256,
)
from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder


RuntimePhase = Literal[
    "train_clean",
    "train_swap",
    "eval_clean",
    "eval_swapped_diagnostic",
]


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = (
        json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    return hashlib.sha256(header + tensor.numpy().tobytes(order="C")).hexdigest()


class MisleadingIdentitySwapV2Encoder(SideFeatureEarlyPoolEncoder):
    """B3S with a verified activity-only row swap before side concat."""

    variant = "B3S_MISLEADING_IDENTITY_SWAP_V2"

    def __init__(
        self,
        trial_length: int,
        window_size: int,
        hidden_dim: int,
        *,
        authority: VerifiedMatchingAuthority,
        side_dim: int = 0,
        electrode_embed_dim: int = 0,
        num_electrodes: int = 0,
        num_post_layers: int = 3,
    ) -> None:
        super().__init__(
            trial_length=trial_length,
            window_size=window_size,
            hidden_dim=hidden_dim,
            side_dim=side_dim,
            electrode_embed_dim=electrode_embed_dim,
            num_electrodes=num_electrodes,
            num_post_layers=num_post_layers,
        )
        self._swap_v2_authority = authority
        self._runtime_session: str | None = None
        self._runtime_epoch: int | None = None
        self._runtime_phase: RuntimePhase = "eval_clean"
        self.last_swap_trace: dict[str, Any] | None = None

    @classmethod
    def from_parent(
        cls,
        parent: SideFeatureEarlyPoolEncoder,
        *,
        authority: VerifiedMatchingAuthority,
    ) -> "MisleadingIdentitySwapV2Encoder":
        if type(parent) is not SideFeatureEarlyPoolEncoder:
            raise TypeError("swap-v2 wrapper accepts only the ordinary production B3S encoder")
        num_post_layers = sum(isinstance(layer, torch.nn.Linear) for layer in parent.post_pool)
        wrapped = cls(
            trial_length=parent.trial_length,
            window_size=parent.window_size,
            hidden_dim=parent.hidden_dim,
            authority=authority,
            side_dim=parent.side_dim,
            electrode_embed_dim=parent.electrode_embed_dim,
            num_electrodes=parent.num_electrodes,
            num_post_layers=num_post_layers,
        )
        wrapped.load_state_dict(parent.state_dict(), strict=True)
        wrapped.train(parent.training)
        return wrapped

    @property
    def matching_authority_sha256(self) -> str:
        return self._swap_v2_authority.sha256

    def configure_runtime(
        self,
        *,
        session: str,
        lightning_current_epoch: int,
        phase: RuntimePhase,
    ) -> None:
        if not isinstance(session, str) or not session:
            raise ValueError("swap-v2 runtime requires one nonempty session name")
        if phase not in {
            "train_clean",
            "train_swap",
            "eval_clean",
            "eval_swapped_diagnostic",
        }:
            raise ValueError(f"invalid swap-v2 runtime phase: {phase!r}")
        if int(lightning_current_epoch) < 0:
            raise ValueError("swap-v2 runtime epoch must be nonnegative")
        self._runtime_session = session
        self._runtime_epoch = int(lightning_current_epoch)
        self._runtime_phase = phase
        self.last_swap_trace = None

    def _swap_active(self) -> bool:
        return self._runtime_phase in {"train_swap", "eval_swapped_diagnostic"}

    def finalize_identity(self, state: dict[str, Any]) -> torch.Tensor:
        if not self._swap_active():
            # Exact-null guarantee: do not reproduce or approximate the parent
            # implementation in clean mode; execute it directly.
            self.last_swap_trace = {
                "phase": self._runtime_phase,
                "swap_applied": False,
                "authority_sha256": self.matching_authority_sha256,
            }
            return super().finalize_identity(state)

        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0 before finalize_identity")
        if self._runtime_session is None or self._runtime_epoch is None:
            raise RuntimeError("swap-v2 runtime context was not configured")

        mean_feat = state["sum_feat"] / state["trial_count"]
        swap = self._swap_v2_authority.swap_for(
            self._runtime_session, self._runtime_epoch
        )
        if mean_feat.shape[1] != swap.permutation.numel():
            raise ValueError(
                "swap-v2 authority/unit count drift: "
                f"mean_feat has {mean_feat.shape[1]} units, mapping has {swap.permutation.numel()}"
            )
        mean_before_sha = _tensor_sha256(mean_feat)
        mean_feat = apply_activity_identity_swap(mean_feat, swap)
        mean_after_sha = _tensor_sha256(mean_feat)

        # This block intentionally mirrors production B3S only after the activity
        # swap.  Side/carrier tensors are read but never index-selected.
        side_parts: list[torch.Tensor] = []
        side = state.get("side_features")
        side_before_sha = None if side is None else _tensor_sha256(side)
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
        electrode_before_sha = None if electrode_ids is None else _tensor_sha256(electrode_ids)
        if self.electrode_embed_dim > 0:
            if electrode_ids is None:
                raise ValueError("B3S requires electrode_ids when electrode_embed_dim > 0")
            if electrode_ids.shape != mean_feat.shape[:2]:
                raise ValueError("B3S electrode_ids shape does not match pooled features")
            assert self.electrode_embed is not None
            side_parts.append(self.electrode_embed(electrode_ids.long()))
        if side_parts:
            mean_feat = torch.cat([mean_feat, *side_parts], dim=-1)

        self.last_swap_trace = {
            "phase": self._runtime_phase,
            "swap_applied": True,
            "session": self._runtime_session,
            "lightning_current_epoch": self._runtime_epoch,
            "epoch_number_one_based": self._runtime_epoch + 1,
            "authority_sha256": self.matching_authority_sha256,
            "permutation_sha256": permutation_sha256(
                [int(value) for value in swap.permutation.tolist()]
            ),
            "selected_count": swap.selected_count,
            "mean_feat_before_sha256": mean_before_sha,
            "mean_feat_after_sha256": mean_after_sha,
            "visible_side_before_sha256": side_before_sha,
            "visible_side_after_sha256": None if side is None else _tensor_sha256(side),
            "electrode_ids_before_sha256": electrode_before_sha,
            "electrode_ids_after_sha256": None if electrode_ids is None else _tensor_sha256(electrode_ids),
            "hidden_descriptor_forwarded_to_post_pool": False,
        }
        return self.post_pool(mean_feat)
