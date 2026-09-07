"""Frozen C2 calibration encoder / H-C materializer / E0. Never trains identity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .h1_config import C2_CKPT_SHA256, C2_EPOCH15_PATH, H1_TEMPORAL, require


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_module_state(module: nn.Module, state: dict[str, torch.Tensor], prefix: str) -> None:
    mapped = {
        name: state[f"{prefix}{name}"].detach().cpu().clone()
        for name, _ in module.state_dict().items()
    }
    missing = [f"{prefix}{name}" for name in module.state_dict() if f"{prefix}{name}" not in state]
    require(not missing, f"C2 materializer keys missing: {missing}")
    module.load_state_dict(mapped, strict=True)


@dataclass
class FrozenC2Materializer:
    """C2 carrier_pre_pool + carrier_post_pool only. Decoder weights are ignored."""

    pre_pool: nn.Sequential
    post_pool: nn.Sequential
    checkpoint_sha256: str
    e0_is_fused_identity: bool = True

    def to(self, device: torch.device | str) -> "FrozenC2Materializer":
        self.pre_pool.to(device)
        self.post_pool.to(device)
        return self

    def eval(self) -> "FrozenC2Materializer":
        self.pre_pool.eval()
        self.post_pool.eval()
        return self

    @torch.no_grad()
    def activity_signature(self, activity: torch.Tensor) -> torch.Tensor:
        """activity [B,M,T,N] or [M,T,N] -> [B,N,32] pre-pool mean."""
        if activity.ndim == 3:
            activity = activity.unsqueeze(0)
        require(activity.ndim == 4, "activity must be [B,M,T,N]")
        require(activity.shape[2] == 1024, "C2 activity trial length is 1024")
        require(activity.shape[3] == H1_TEMPORAL.n_units, "C2 activity units")
        temporal = activity.permute(0, 1, 3, 2)
        return self.pre_pool(temporal).mean(dim=1)

    @torch.no_grad()
    def fused_identity(self, activity: torch.Tensor, carrier: torch.Tensor) -> torch.Tensor:
        """Return E0 [B,N,700] fused identity (activity + H-C already mixed)."""
        pooled = self.activity_signature(activity)
        if carrier.ndim == 2:
            carrier = carrier.unsqueeze(0)
        require(carrier.shape == (pooled.shape[0], H1_TEMPORAL.n_units, H1_TEMPORAL.hc_dim), "H-C shape")
        joined = torch.cat((pooled, carrier.to(dtype=pooled.dtype, device=pooled.device)), dim=-1)
        identity = self.post_pool(joined)
        require(tuple(identity.shape[1:]) == (H1_TEMPORAL.n_units, H1_TEMPORAL.e0_dim), "E0 fused shape")
        return identity

    @torch.no_grad()
    def materialize_bank(
        self,
        activity: torch.Tensor,
        carrier: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """E0 [N,700] fused identity and T [N,4] H-C. Batch must be 1 for a session bank."""
        e0 = self.fused_identity(activity, carrier)
        require(e0.shape[0] == 1, "session bank expects one session")
        if carrier.ndim == 3:
            hc = carrier[0]
        else:
            hc = carrier
        return e0[0].contiguous(), hc.to(dtype=e0.dtype, device=e0.device).contiguous()


def load_frozen_c2_materializer(
    path: str | Path = C2_EPOCH15_PATH,
    *,
    expected_sha256: str = C2_CKPT_SHA256,
) -> FrozenC2Materializer:
    path = Path(path)
    digest = sha256_file(path)
    require(digest == expected_sha256, f"C2 checkpoint SHA drift: {digest}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(payload.get("schema") == "h1_cal_aug_m3_aware_dual_selection_v2_checkpoint", "C2 schema")
    state = payload["state_dict"]
    decoder_keys = [k for k in state if k.startswith("transformer.") or k.startswith("fc_") or k == "rep"]
    require(len(decoder_keys) > 0, "C2 checkpoint unexpectedly lacks decoder keys to refuse")
    pre = nn.Sequential(nn.Linear(1024, 32), nn.ReLU())
    post = nn.Sequential(
        nn.Linear(32 + 4, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 700),
    )
    _copy_module_state(pre, state, "carrier_pre_pool.")
    _copy_module_state(post, state, "carrier_post_pool.")
    for parameter in list(pre.parameters()) + list(post.parameters()):
        parameter.requires_grad_(False)
    pre.eval()
    post.eval()
    return FrozenC2Materializer(pre_pool=pre, post_pool=post, checkpoint_sha256=digest)
