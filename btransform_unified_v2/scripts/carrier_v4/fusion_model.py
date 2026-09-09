"""Matched H1/M1 fusion frontends for carrier v4.

This module imports frozen decoder classes but never modifies them.  ``concat``
and ``proj_add`` use their existing constructors verbatim.  ``proj_add`` at
P32 is expanded from the same P16 reference so its initial function is exactly
P16 while its independent second E0 projection block can learn afterward.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from typing import Any

import torch
from torch import Tensor, nn

from btransform_unified_v2.model import RiftDecoder
from btransform_unified_v2.concat_model import RiftConcatDecoder
from btransform_unified_v2.joint_m1_model import JointM1ConcatDecoder


FUSIONS = ("concat", "proj_add")


def _check_fusion(fusion: str, proj_dim: int) -> None:
    if fusion not in FUSIONS:
        raise ValueError(f"fusion must be one of {FUSIONS}")
    if fusion == "concat" and proj_dim not in (0, 16):
        raise ValueError("concat has no projection dimension; use proj_dim=16")
    if fusion == "proj_add" and proj_dim not in (16, 32):
        raise ValueError("proj_add supports the matched P16 or function-preserving P32 only")


def _linear(module: nn.Module, *, name: str) -> nn.Linear:
    if not isinstance(module, nn.Linear):
        raise RuntimeError(f"{name} must be nn.Linear")
    return module


class RiftDecoderP32(RiftDecoder):
    """P32 expansion initialized to the exact P16 function.

    P16 is built first through the frozen constructor.  The first 16 E0-proj
    rows and all unchanged parameters remain P16-identical.  The new E0-proj
    rows receive independent fixed-seed Xavier weights, while their token-MLP
    input columns begin at zero; hence they affect no initial output but do
    receive nonzero gradients once optimization begins.
    """

    def __init__(self, task: str, *, context_bins: int, bias_mode: str, seed: int) -> None:
        super().__init__(task, context_bins=context_bins, bias_mode=bias_mode, seed=seed, proj_dim=16)
        old_projection = _linear(self.frontend.e0_proj, name="P16 e0_proj")
        old_token = _linear(self.frontend.token_mlp[0], name="P16 token_mlp[0]")
        e0_dim = old_projection.in_features
        if old_projection.out_features != 16 or old_token.in_features != 20:
            raise RuntimeError("P16 frontend geometry drift")
        if old_token.out_features <= 0:
            raise RuntimeError("P16 token MLP output geometry drift")

        # Linear constructors initialize eagerly, so keep their allocation and
        # replacement initialization inside a local RNG domain as well.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 0x50333245)
            expanded_projection = nn.Linear(e0_dim, 32, bias=old_projection.bias is not None)
            expanded_token = nn.Linear(36, old_token.out_features, bias=old_token.bias is not None)
            nn.init.xavier_uniform_(expanded_projection.weight[16:])
        with torch.no_grad():
            expanded_projection.weight[:16].copy_(old_projection.weight)
            if old_projection.bias is not None:
                expanded_projection.bias[:16].copy_(old_projection.bias)
            # New rows are deliberately not copied or symmetrically duplicated.
            if expanded_projection.bias is not None:
                expanded_projection.bias[16:].zero_()

            # P32 fused local is [local+p1 | local+p2 | carrier].  Folding it
            # at initialization gives the P16 effective token map exactly.
            expanded_token.weight[:, :16].copy_(old_token.weight[:, :16])
            expanded_token.weight[:, 16:32].zero_()
            expanded_token.weight[:, 32:].copy_(old_token.weight[:, 16:])
            if old_token.bias is not None:
                expanded_token.bias.copy_(old_token.bias)
        self.frontend.e0_proj = expanded_projection
        self.frontend.token_mlp[0] = expanded_token
        # SharedSetFrontendIdentity and its owning v1 identity both retain
        # dimensional metadata used by static/folded helpers.  Keep all of it
        # truthful after changing the actual first token layer from 20 to 36.
        self.frontend.token_in = 36
        owner = self._frontend_owner
        owner.token_in = 36
        owner.proj_out_dim = 32
        owner.proj_groups = 2
        owner.proj_dim_override = 32
        if isinstance(getattr(owner, "init_meta", None), dict):
            owner.init_meta.update({
                "token_in": 36,
                "proj_out_dim": 32,
                "proj_groups": 2,
                "fused_local_width": 32,
                "proj_dim_override": 32,
                "proj_param_count": int(expanded_projection.weight.numel()),
                "param_count": int(sum(parameter.numel() for parameter in owner.parameters())),
            })
        self.proj_dim = 32


def build_h1(
    fusion: str,
    proj_dim: int,
    context_bins: int,
    bias_mode: str,
    seed: int,
    attention_backend: str,
    device: str | torch.device,
) -> nn.Module:
    """Build an H1 fusion model and set only the existing temporal backend."""
    _check_fusion(fusion, proj_dim)
    if fusion == "concat":
        model: nn.Module = RiftConcatDecoder("h1", context_bins=context_bins, bias_mode=bias_mode, seed=seed)
    elif proj_dim == 16:
        # Direct frozen constructor: byte-identical P16 initialization.
        model = RiftDecoder("h1", context_bins=context_bins, bias_mode=bias_mode, seed=seed, proj_dim=16)
    else:
        model = RiftDecoderP32("h1", context_bins=context_bins, bias_mode=bias_mode, seed=seed)
    model.temporal.set_attention_backend(attention_backend)  # type: ignore[attr-defined]
    return model.to(device)


class JointM1FusionDecoder(JointM1ConcatDecoder):
    """M1 live-B3S decoder with matched concat, P16, or P32 frontend fusion."""

    def __init__(
        self,
        arm: str,
        *,
        fusion: str = "concat",
        proj_dim: int = 16,
        seed: int = 42,
    ) -> None:
        _check_fusion(fusion, proj_dim)
        # Builds and retains the frozen live B3S encoder/session-memory logic.
        super().__init__(arm, seed=seed)
        self.fusion = fusion
        self.fusion_proj_dim = 16 if fusion == "concat" else proj_dim
        if fusion == "concat":
            return
        reference: RiftDecoder
        if proj_dim == 16:
            reference = RiftDecoder("m1", context_bins=100, bias_mode="recency", seed=seed, proj_dim=16)
        else:
            reference = RiftDecoderP32("m1", context_bins=100, bias_mode="recency", seed=seed)
        self._install_reference_frontend(reference)

    def _install_reference_frontend(self, reference: RiftDecoder) -> None:
        """Replace only frozen RIFT frontend/temporal modules; retain live encoder."""
        self.frontend = reference.frontend
        self.final_norm = reference.final_norm
        self.readout = reference.readout
        self.temporal = reference.temporal
        self.temporal_config = reference.temporal_config
        self.bias_mode = reference.bias_mode
        self.seed = reference.seed
        self.proj_dim = reference.proj_dim
        # The inherited concat owner remains the registered home for active v1
        # modules, but its geometry must describe the reference P16/P32
        # frontend rather than the discarded concat frontend.  Several v1
        # helpers inspect this owner directly (including token width and
        # projection grouping), so leaving its concat metadata here creates a
        # false external geometry despite the modules above being correct.
        owner = self._frontend_owner
        reference_owner = reference._frontend_owner
        for name in (
            "identity_mode", "window_override", "geometry", "window", "units",
            "base_e0_dim", "prefix", "pending_prefix", "carrier_dim", "out_dim",
            "l_in", "token_e0_width", "e0_dim", "token_in", "proj_out_dim",
            "proj_groups", "proj_dim_override", "unit_dropout_p", "temporal_layers",
        ):
            setattr(owner, name, copy.deepcopy(getattr(reference_owner, name)))
        owner.init_meta = copy.deepcopy(reference_owner.init_meta)
        owner.frontend = self.frontend
        owner.final_norm = self.final_norm
        owner.readout = self.readout
        owner.temporal = nn.Identity()

    def _fuse_batched_local(self, local: Tensor, e0: Tensor, carrier: Tensor, keep: Tensor) -> Tensor:
        if self.fusion == "concat":
            return RiftConcatDecoder._fuse_batched_local(self, local, e0, carrier, keep)
        return RiftDecoder._fuse_batched_local(self, local, e0, carrier, keep)


def build_static(
    task: str = "m1",
    *,
    fusion: str = "concat",
    proj_dim: int = 16,
    context_bins: int | None = None,
    bias_mode: str = "recency",
    seed: int = 42,
    attention_backend: str = "local",
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Build a non-live reference suitable for matched-initialization audits."""
    _check_fusion(fusion, proj_dim)
    if task == "m1":
        bins = 100 if context_bins is None else context_bins
    elif task == "h1":
        bins = 300 if context_bins is None else context_bins
    else:
        raise ValueError("build_static supports only m1 or h1")
    if fusion == "concat":
        model: nn.Module = RiftConcatDecoder(task, context_bins=bins, bias_mode=bias_mode, seed=seed)
    elif proj_dim == 16:
        model = RiftDecoder(task, context_bins=bins, bias_mode=bias_mode, seed=seed, proj_dim=16)
    else:
        model = RiftDecoderP32(task, context_bins=bins, bias_mode=bias_mode, seed=seed)
    model.temporal.set_attention_backend(attention_backend)  # type: ignore[attr-defined]
    return model.to(device)


def _parameter_bytes(parameter: Tensor) -> bytes:
    return parameter.detach().to(device="cpu").contiguous().numpy().tobytes()


def _fusion_parameter(name: str) -> bool:
    return name.startswith("frontend.e0_proj") or name.startswith("frontend.token_mlp.0") or name.startswith("_frontend_owner.frontend.e0_proj") or name.startswith("_frontend_owner.frontend.token_mlp.0")


def _effective_token_weight(model: nn.Module) -> Tensor:
    """Fold proj-add or concat to [local16 | full-E0 | carrier4] token weights."""
    frontend = model.frontend  # type: ignore[attr-defined]
    token = _linear(frontend.token_mlp[0], name="token_mlp[0]")
    e0_dim = int(model.geometry["e0_dim"])  # type: ignore[attr-defined]
    if token.in_features == 16 + e0_dim + 4:  # concat
        return token.weight.detach().to("cpu")
    projection = _linear(frontend.e0_proj, name="e0_proj")
    groups = projection.out_features // 16
    if projection.out_features != groups * 16 or token.in_features != groups * 16 + 4:
        raise RuntimeError("unsupported proj_add fusion geometry")
    local = torch.zeros_like(token.weight[:, :16])
    folded_e0 = torch.zeros((token.out_features, e0_dim), dtype=token.weight.dtype, device=token.weight.device)
    for group in range(groups):
        group_weight = token.weight[:, group * 16:(group + 1) * 16]
        local = local + group_weight
        folded_e0 = folded_e0 + group_weight @ projection.weight[group * 16:(group + 1) * 16]
    return torch.cat((local, folded_e0, token.weight[:, groups * 16:]), dim=1).detach().to("cpu")


def audit_matched_initialization(
    task: str,
    models: Mapping[str, nn.Module],
    actual_banks: Any | None = None,
    x: Tensor | None = None,
    *,
    input_valid_mask: Tensor | None = None,
) -> dict[str, Any]:
    """Audit parameter identity and optional banked forward equivalence.

    Callers may pass arbitrary named matched models.  When ``actual_banks`` and
    ``x`` are supplied, every model is evaluated with the existing forward API;
    the function does not manufacture a bank or change dropout/mask semantics.
    """
    if task not in ("m1", "h1"):
        raise ValueError("audit task must be m1 or h1")
    if not isinstance(models, Mapping) or len(models) < 2:
        raise ValueError("models must map at least two stable names to modules")
    named = {str(name): model for name, model in models.items()}
    if any(not name or not isinstance(model, nn.Module) for name, model in named.items()):
        raise ValueError("models must have nonempty names and nn.Module values")
    names = tuple(named)
    parameter_counts = {name: sum(parameter.numel() for parameter in model.parameters()) for name, model in named.items()}
    parameter_maps = {name: dict(model.named_parameters()) for name, model in named.items()}
    pairwise: dict[str, Any] = {}
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            left, right = parameter_maps[left_name], parameter_maps[right_name]
            shared = sorted(set(left) & set(right))
            checked = [name for name in shared if not _fusion_parameter(name) and left[name].shape == right[name].shape]
            unequal = [name for name in checked if _parameter_bytes(left[name]) != _parameter_bytes(right[name])]
            folded_left, folded_right = _effective_token_weight(named[left_name]), _effective_token_weight(named[right_name])
            pairwise[f"{left_name}__{right_name}"] = {
                "non_fusion_shared_parameter_count": len(checked),
                "non_fusion_shared_byte_equal": not unequal,
                "non_fusion_unequal_names": unequal,
                "effective_fold_shape_equal": tuple(folded_left.shape) == tuple(folded_right.shape),
                "effective_fold_max_abs": float((folded_left - folded_right).abs().max().item()) if folded_left.shape == folded_right.shape else None,
            }
    report: dict[str, Any] = {"task": task, "parameter_counts": parameter_counts, "pairwise": pairwise, "forward": None}
    if actual_banks is not None or x is not None:
        if actual_banks is None or x is None:
            raise ValueError("actual_banks and x must be supplied together")
        previous_training = {name: model.training for name, model in named.items()}
        outputs: dict[str, Tensor] = {}
        try:
            for name, model in named.items():
                model.eval()
                with torch.no_grad():
                    outputs[name] = model(x, actual_banks, input_valid_mask=input_valid_mask).detach().to("cpu")
        finally:
            for name, model in named.items():
                model.train(previous_training[name])
        anchor = names[0]
        report["forward"] = {
            "anchor": anchor,
            "output_shapes": {name: list(value.shape) for name, value in outputs.items()},
            "max_abs_vs_anchor": {name: float((value - outputs[anchor]).abs().max().item()) for name, value in outputs.items()},
        }
    return report


__all__ = [
    "FUSIONS",
    "RiftDecoderP32",
    "JointM1FusionDecoder",
    "build_h1",
    "build_static",
    "audit_matched_initialization",
]
