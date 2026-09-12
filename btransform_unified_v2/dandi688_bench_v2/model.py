"""DANDI000688 B3S-concat calibration plus M2-aligned learned-RIFT decoding.

The RIFT/P16 temporal backbone preserves the current M2 geometry. DANDI uses
its own B3S identity materializer: Full concatenates MOVE--T4 after per-unit
trial pooling and before ``post_pool``; ACT is the side-free B3S control. The
variable padded channel roster is handled explicitly by the set frontend.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from btransform_unified_v2.model import RiftDecoder
from learnable_recency_v1.config import LearnableRecencyConfig, dataset_config
from learnable_recency_v1.wrap import install_temporal, split_optimizer_parameters

MAX_UNITS = 100
E0_DIM = 50
T4_DIM = 4
OUTPUT_DIM = 2
CONTEXT_BINS = 50
SUPPORT_BINS = 100
HIDDEN_DIM = 64


def _affine_stack(input_dim: int, hidden_dim: int, num_layers: int, output_dim: int) -> nn.Sequential:
    """Canonical ``SideFeatureEarlyPoolEncoder`` affine-stack construction."""
    if num_layers < 1:
        raise ValueError("num_post_layers must be positive")
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim if num_layers > 1 else output_dim)]
    if num_layers == 1:
        return nn.Sequential(*layers)
    layers.append(nn.ReLU())
    for _ in range(num_layers - 2):
        layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class B3SIdentityEncoder(nn.Module):
    """Canonical B3S early-pool identity encoder with optional MOVE--T4 side.

    ``support`` has DANDI layout ``[trials,100,units]``.  Each unit's trial
    vector is transformed by ``pre_pool``, averaged over trials, optionally
    concatenated with its four-dimensional MOVE--T4 row, and passed through
    the three-affine-layer ``post_pool`` stack.  Full uses ``side_dim=4``;
    ACT uses ``side_dim=0`` and cannot consume a carrier.
    """
    variant = "B3S"

    def __init__(self, *, side_dim: int, support_bins: int = SUPPORT_BINS,
                 hidden_dim: int = HIDDEN_DIM, e0_dim: int = E0_DIM,
                 num_post_layers: int = 3, seed: int = 42) -> None:
        super().__init__()
        if side_dim not in (0, T4_DIM) or support_bins < 1 or hidden_dim < 1 or e0_dim < 1:
            raise ValueError("B3S dimensions must be positive and side_dim must be 0 or 4")
        self.side_dim, self.support_bins, self.hidden_dim, self.e0_dim = (
            int(side_dim), int(support_bins), int(hidden_dim), int(e0_dim)
        )
        self.num_post_layers = int(num_post_layers)
        # Isolated CPU RNG state preserves the shared M2-RIFT initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 0x423353)
            self.pre_pool = nn.Sequential(nn.Linear(self.support_bins, self.hidden_dim), nn.ReLU())
            self.post_pool = _affine_stack(self.hidden_dim + self.side_dim, self.hidden_dim,
                                           self.num_post_layers, self.e0_dim)
            if self.side_dim:
                assert isinstance(self.post_pool[0], nn.Linear)
                nn.init.zeros_(self.post_pool[0].weight[:, self.hidden_dim:])

    def forward(self, support: Tensor, side: Tensor | None = None) -> Tensor:
        if support.ndim != 3 or support.shape[0] < 1 or support.shape[1] != self.support_bins or support.shape[2] < 1:
            raise ValueError(f"support must be [M,{self.support_bins},N]")
        if not torch.is_floating_point(support) or not bool(torch.isfinite(support).all()):
            raise ValueError("support must be finite floating-point")
        units = support.shape[2]
        pooled = self.pre_pool(support.permute(0, 2, 1)).mean(dim=0)
        if self.side_dim == 0:
            if side is not None:
                raise ValueError("B3S side_dim=0 does not consume a side feature")
            return self.post_pool(pooled)
        if side is None or side.shape != (units, self.side_dim):
            raise ValueError(f"B3S side must be [{units},{self.side_dim}]")
        if not torch.is_floating_point(side) or not bool(torch.isfinite(side).all()):
            raise ValueError("B3S side must be finite floating-point")
        return self.post_pool(torch.cat((pooled, side.to(device=support.device, dtype=support.dtype)), dim=-1))

    def provenance(self, checkpoint_sha256: str) -> dict[str, Any]:
        if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
            raise ValueError("checkpoint_sha256 must be a SHA-256 hex digest")
        return {"schema": "dandi688_v2_b3s_encoder_v1", "checkpoint_sha256": checkpoint_sha256,
                "variant": self.variant, "support_bins": self.support_bins,
                "hidden_dim": self.hidden_dim, "e0_dim": self.e0_dim,
                "side_dim": self.side_dim, "num_post_layers": self.num_post_layers,
                "encoder_line": "concat", "encoder_carrier_fusion": "post_pool_input_concat",
                "encoder_film": False}


class DandiRiftDecoder(RiftDecoder):
    """M2 learned-RIFT/proj_add decoder with explicit direct E0/T4 inputs."""
    def __init__(self, arm: str = "full", n_pad: int = MAX_UNITS, seed: int = 42,
                 encoder: B3SIdentityEncoder | None = None, freeze_encoder: bool = True,
                 recency_cfg: LearnableRecencyConfig | None = None) -> None:
        if arm not in {"full", "activity", "raw_set"}:
            raise ValueError("arm must be full, activity, or raw_set")
        if n_pad < 1:
            raise ValueError("n_pad must be positive")
        recency_cfg = dataset_config("m2", tier="learned_slope", ladder="default") if recency_cfg is None else recency_cfg
        if recency_cfg.context_bins != CONTEXT_BINS or recency_cfg.layers != 4:
            raise ValueError("DANDI v2 requires current M2 R50/D4 learned-recency geometry")
        self.arm = arm
        self.freeze_encoder = bool(freeze_encoder)
        geometry = {"task": "dandi688_v2", "units": int(n_pad), "e0_dim": E0_DIM,
                    "carrier_dim": T4_DIM, "out_dim": OUTPUT_DIM}
        super().__init__(geometry, context_bins=CONTEXT_BINS, bias_mode="recency", seed=int(seed), proj_dim=16)
        install_temporal(self, recency_cfg, int(seed))
        self.temporal.set_attention_backend("local")
        if arm == "full":
            self.encoder = B3SIdentityEncoder(side_dim=T4_DIM, seed=seed) if encoder is None else encoder
            if self.encoder.side_dim != T4_DIM:
                raise ValueError("full arm requires a B3S encoder with side_dim=4")
            if freeze_encoder:
                for parameter in self.encoder.parameters():
                    parameter.requires_grad_(False)
        elif arm == "activity":
            if encoder is not None:
                raise ValueError("activity arm owns its side-free B3S encoder")
            self.encoder = B3SIdentityEncoder(side_dim=0, seed=seed)
        else:
            if encoder is not None:
                raise ValueError("raw_set has no support encoder")
            self.encoder = None
            self.global_g = nn.Parameter(torch.zeros(16))
            # ``e0_proj`` belongs to the inherited P16 frontend but has no
            # path in raw-set, whose identity is directly projected ``g``.
            # Freeze it so the raw arm does not advertise unused capacity.
            assert self.frontend.e0_proj is not None
            for parameter in self.frontend.e0_proj.parameters():
                parameter.requires_grad_(False)

    @property
    def core(self) -> "DandiRiftDecoder":
        """Explicit name for the M2-aligned RIFT/P16 decoder core."""
        return self

    def _identity_batch(self, e0: Tensor, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        shape = (self.units, self.geometry["e0_dim"])
        if e0.ndim == 2:
            if tuple(e0.shape) != shape:
                raise ValueError(f"e0 must be {shape} or [B,{shape[0]},{shape[1]}]")
            e0 = e0.unsqueeze(0).expand(batch, -1, -1)
        elif e0.ndim != 3 or tuple(e0.shape) != (batch, *shape):
            raise ValueError(f"e0 must be {shape} or [B,{shape[0]},{shape[1]}]")
        if not torch.is_floating_point(e0) or not bool(torch.isfinite(e0).all()):
            raise ValueError("e0 must be finite floating-point")
        return e0.to(device=device, dtype=dtype)

    def _carrier_batch(self, carrier: Tensor, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if carrier.ndim == 2:
            if tuple(carrier.shape) != (self.units, T4_DIM):
                raise ValueError(f"carrier must be [{self.units},{T4_DIM}]")
            carrier = carrier.unsqueeze(0).expand(batch, -1, -1)
        elif carrier.ndim != 3 or tuple(carrier.shape) != (batch, self.units, T4_DIM):
            raise ValueError(f"carrier must be [{self.units},{T4_DIM}] or [B,{self.units},{T4_DIM}]")
        if not torch.is_floating_point(carrier) or not bool(torch.isfinite(carrier).all()):
            raise ValueError("carrier must be finite floating-point")
        return carrier.to(device=device, dtype=dtype)

    def forward_scores(self, x: Tensor, e0: Tensor, carrier: Tensor, *, unit_mask: Tensor | None = None,
                       dropout_keep: Tensor | None = None, input_valid_mask: Tensor | None = None) -> Tensor:
        x = self._check_input(x, input_valid_mask)
        if input_valid_mask is not None:
            x = torch.where(input_valid_mask.unsqueeze(-1), x, torch.zeros_like(x))
        keep = self._resolve_direct_keep(unit_mask, dropout_keep, x.shape[0], x.device)
        local = self.frontend.local_conv(x)
        tokens = self._fuse_batched_local(local, self._identity_batch(e0, x.shape[0], x.device, x.dtype),
                                           self._carrier_batch(carrier, x.shape[0], x.device, x.dtype), keep)
        return self.readout(self.final_norm(self.temporal(tokens, input_valid_mask)))

    def _resolve_direct_keep(self, unit_mask: Tensor | None, dropout_keep: Tensor | None, batch: int, device: torch.device) -> Tensor:
        def _as_batch(value: Tensor | None, name: str) -> Tensor:
            if value is None:
                return torch.ones(batch, self.units, dtype=torch.bool, device=device)
            value = value.to(device=device, dtype=torch.bool)
            if value.ndim == 1:
                value = value.unsqueeze(0).expand(batch, -1)
            if tuple(value.shape) != (batch, self.units):
                raise ValueError(f"{name} must be [N] or [B,{self.units}]")
            return value
        # Dropout is an additional removal mask; it may never revive padded
        # or otherwise unobserved rows supplied by unit_mask.
        keep = _as_batch(unit_mask, "unit_mask") & _as_batch(dropout_keep, "dropout_keep")
        if not bool(keep.any(dim=1).all()):
            raise ValueError("unit mask is empty for some batch row")
        return keep.contiguous()

    def forward_identity(self, x: Tensor, e0: Tensor, carrier: Tensor, *, unit_mask: Tensor | None = None,
                dropout_keep: Tensor | None = None, input_valid_mask: Tensor | None = None) -> Tensor:
        return self.forward_scores(x, e0, carrier, unit_mask=unit_mask, dropout_keep=dropout_keep,
                                   input_valid_mask=input_valid_mask)[:, -1]

    def encode(self, activity: Tensor, carrier: Tensor | None = None) -> Tensor:
        """Materialize this arm's identity from its allowed calibration input."""
        if self.arm == "full":
            if carrier is None:
                raise ValueError("full arm requires MOVE-T4 carrier")
            assert isinstance(self.encoder, B3SIdentityEncoder) and self.encoder.side_dim == T4_DIM
            return self.encoder(activity, carrier)
        if self.arm == "activity":
            if carrier is not None:
                raise ValueError("activity arm does not consume a carrier")
            assert isinstance(self.encoder, B3SIdentityEncoder) and self.encoder.side_dim == 0
            return self.encoder(activity)
        raise RuntimeError("raw_set has no calibrated identity")

    def calibrate(self, activity: Tensor, carrier: Tensor | None = None) -> Tensor:
        if self.training:
            raise RuntimeError("calibrate is eval-only")
        with torch.no_grad():
            return self.encode(activity, carrier).detach()

    def forward_full(self, x: Tensor, support: Tensor, t4: Tensor, **kwargs: Any) -> Tensor:
        return self.forward_identity(x, self.encode(support, t4), t4, **kwargs)

    def forward_activity(self, x: Tensor, support: Tensor, **kwargs: Any) -> Tensor:
        assert isinstance(self.encoder, B3SIdentityEncoder) and self.encoder.side_dim == 0
        e0 = self.encoder(support)
        zero = torch.zeros((self.units, T4_DIM), device=x.device, dtype=x.dtype)
        return self.forward_identity(x, e0, zero, **kwargs)

    def forward(self, x: Tensor, *, activity: Tensor | None = None, carrier: Tensor | None = None,
                unit_mask: Tensor | None = None, dropout_keep: Tensor | None = None,
                input_valid_mask: Tensor | None = None) -> Tensor:
        """Run one v2 arm with the public prepared-session interface."""
        if self.arm == "raw_set":
            if activity is not None or carrier is not None:
                raise ValueError("raw_set consumes neither activity support nor carrier")
            return self.forward_raw_set(x, unit_mask=unit_mask, dropout_keep=dropout_keep,
                                        input_valid_mask=input_valid_mask)
        if activity is None:
            raise ValueError(f"{self.arm} arm requires activity[M,100,N]")
        if self.arm == "full":
            if carrier is None:
                raise ValueError("full arm requires carrier[N,4]")
            return self.forward_full(x, activity, carrier, unit_mask=unit_mask, dropout_keep=dropout_keep,
                                     input_valid_mask=input_valid_mask)
        if carrier is not None:
            raise ValueError("activity arm does not consume carrier")
        return self.forward_activity(x, activity, unit_mask=unit_mask, dropout_keep=dropout_keep,
                                     input_valid_mask=input_valid_mask)

    def optimizer_param_groups(self, *, lr: float, weight_decay: float,
                               learned_lr_multiplier: float | None = None) -> list[dict[str, Any]]:
        """AdamW-ready groups; learned recency scalars get the M2 multiplier."""
        if lr <= 0 or weight_decay < 0:
            raise ValueError("lr must be positive and weight_decay nonnegative")
        multiplier = self.learnable_cfg.lr_multiplier if learned_lr_multiplier is None else float(learned_lr_multiplier)
        return split_optimizer_parameters(self, peak_lr=float(lr), weight_decay=float(weight_decay),
                                          lr_multiplier=multiplier)

    def forward_raw_set(self, x: Tensor, *, unit_mask: Tensor | None = None, dropout_keep: Tensor | None = None,
                        input_valid_mask: Tensor | None = None) -> Tensor:
        if self.arm != "raw_set":
            raise RuntimeError("forward_raw_set is only valid for arm='raw_set'")
        x = self._check_input(x, input_valid_mask)
        if input_valid_mask is not None:
            x = torch.where(input_valid_mask.unsqueeze(-1), x, torch.zeros_like(x))
        keep = self._resolve_direct_keep(unit_mask, dropout_keep, x.shape[0], x.device)
        local = self.frontend.local_conv(x)
        if self.frontend.e0_proj is None or self.frontend.e0_proj.out_features != 16:
            raise RuntimeError("raw-set requires P16 proj_add")
        fused = local + self.global_g.to(device=x.device, dtype=local.dtype).view(1, 1, 1, 16)
        zeros = torch.zeros((x.shape[0], self.units, T4_DIM), device=x.device, dtype=local.dtype)
        tokens = self._fuse_batched_local_direct_projected(fused, zeros, keep)
        return self.readout(self.final_norm(self.temporal(tokens, input_valid_mask)))[:, -1]

    def _fuse_batched_local_direct_projected(self, fused: Tensor, carrier: Tensor, keep: Tensor) -> Tensor:
        batch, width, units, channels = fused.shape
        if channels != 16:
            raise RuntimeError("raw-set projected identity expects P16")
        tokens = self.frontend.token_norm(self.frontend.token_mlp(torch.cat((fused, carrier.unsqueeze(1).expand(-1, width, -1, -1)), dim=-1)))
        slots = self.frontend.slot_norm(self.frontend.slots).view(1, 1, 8, 256).expand(batch, width, 8, 256)
        q, k = slots.reshape(batch * width, 8, 256), tokens.reshape(batch * width, units, 256)
        pad = (~keep).unsqueeze(1).expand(batch, width, units).reshape(batch * width, units)
        attended, _ = self.frontend.mha(q, k, k, key_padding_mask=pad, need_weights=False)
        slots_out = q + attended
        slots_out = slots_out + self.frontend.slot_ffn(self.frontend.slot_ffn_norm(slots_out))
        return self.frontend.slot_proj(slots_out.reshape(batch, width, 8 * 256))


class RawSetRiftDecoder(DandiRiftDecoder):
    """Raw-set control: global 16-D identity broadcast to observed rows."""
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(arm="raw_set", **kwargs)



__all__ = ["MAX_UNITS", "E0_DIM", "T4_DIM", "OUTPUT_DIM", "CONTEXT_BINS", "SUPPORT_BINS", "HIDDEN_DIM",
           "B3SIdentityEncoder", "DandiRiftDecoder", "RawSetRiftDecoder"]
