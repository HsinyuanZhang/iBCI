"""Independent post-fc_in carrier projection for EMG-rSyn3 Stage-1."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1.plan import CARRIER_DIM


class InjectionError(RuntimeError):
    """Fail closed for independent carrier injection."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InjectionError(message)


def zero_projection_weight(model_dim: int, torch_module) -> torch.nn.Parameter:
    _require(int(model_dim) >= 1, "model_dim")
    return torch_module.nn.Parameter(torch_module.zeros(int(model_dim), int(CARRIER_DIM)))


def apply_post_fc_in(hidden, carrier, weight, mask):
    """u = mask * hidden + P(mask * carrier). Weight is [D, 4]."""
    unit = mask.unsqueeze(-1) if mask.ndim == hidden.ndim - 1 else mask
    return hidden * unit + F.linear(carrier * unit, weight)


def attach_projection(student) -> None:
    """Attach a zero, RNG-free Linear(4, D) onto an already-built StreamingSpintModel."""
    if getattr(student, "carrier_projection_weight", None) is not None:
        return
    model_dim = int(student.decoder.model_dim)
    student.carrier_projection_weight = torch.nn.Parameter(torch.zeros(model_dim, int(CARRIER_DIM)))
    original_decode = student.decode_with_identity
    original_forward = student.forward

    def patched_decode(
        neural,
        identity,
        neuron_gate=None,
        live_gain_features=None,
        live_gain_state=None,
        carrier=None,
    ):
        src = neural.permute(0, 2, 1)
        if live_gain_features is not None and live_gain_state is not None:
            raise ValueError("Pass RT L-D gain features or a cached gain state, not both")
        if live_gain_features is not None:
            if student.live_activity_gain is None:
                raise RuntimeError("RT L-D gain features supplied without an attached gain module")
            src = student.live_activity_gain(src, live_gain_features)
        elif live_gain_state is not None:
            if student.live_activity_gain is None:
                raise RuntimeError("RT L-D gain state supplied without an attached gain module")
            src = student.live_activity_gain.apply_state(src, live_gain_state)
        if neuron_gate is not None:
            src = src * neuron_gate
        if student.fixed_slot_router is None:
            src = src + identity
        else:
            src, _ = student.fixed_slot_router(src, identity)
        if student._decoder_frozen:
            student.decoder.eval()
        batch_size = src.size(0)
        num_neurons = src.size(1)
        dropout_mask = torch.ones(batch_size, num_neurons, device=src.device, dtype=src.dtype)
        if not student._decoder_frozen:
            if student.decoder.dynamic_dropout and student.training:
                import random

                p = random.uniform(student.decoder.dynamic_dropout_low, student.decoder.dynamic_dropout_high)
                dropout_mask = torch.nn.functional.dropout(dropout_mask, p=p, training=True)
            elif student.decoder.dropout_rate > 0.0 and student.training:
                dropout_mask = torch.nn.functional.dropout(
                    dropout_mask, p=student.decoder.dropout_rate, training=True
                )
        src = src * dropout_mask.unsqueeze(-1)
        src = student.decoder.fc_in(src)
        payload = carrier if carrier is not None else getattr(student, "_forward_carrier", None)
        if payload is None:
            raise InjectionError("independent injection requires a carrier tensor")
        if payload.ndim == 2:
            payload = payload.unsqueeze(0)
        if payload.shape[0] == 1 and src.shape[0] > 1:
            payload = payload.expand(src.shape[0], -1, -1)
        if tuple(payload.shape[:2]) != (batch_size, num_neurons) or int(payload.shape[-1]) != int(CARRIER_DIM):
            raise InjectionError(f"carrier shape {tuple(payload.shape)} vs src {tuple(src.shape)}")
        src = apply_post_fc_in(src, payload, student.carrier_projection_weight, dropout_mask)
        rep = student.decoder.fc_in(student.decoder.rep).to(src)
        transformer_output, _ = student.decoder.transformer(rep.repeat(src.size(0), 1, 1), src)
        output = student.decoder.fc_out(transformer_output)
        return output.permute(0, 2, 1)

    def patched_forward(
        neural,
        calib_trials: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        side_features: Optional[torch.Tensor] = None,
        decoder_key_features: Optional[torch.Tensor] = None,
        electrode_ids: Optional[torch.Tensor] = None,
        live_gain_features: Optional[torch.Tensor] = None,
        carrier: Optional[torch.Tensor] = None,
    ):
        student._forward_carrier = carrier
        try:
            if side_features is not None:
                raise InjectionError("B3 independent injection forbids encoder side_features")
            if identity is None:
                if calib_trials is None:
                    raise ValueError("Either calib_trials or identity must be provided")
                identity = student.compute_identity(
                    calib_trials, side_features=None, electrode_ids=electrode_ids,
                )
            behavior = patched_decode(
                neural, identity, live_gain_features=live_gain_features, carrier=carrier,
            )
            return behavior, identity
        finally:
            student._forward_carrier = None

    student.decode_with_identity = patched_decode
    student.forward = patched_forward
    student._parent_decode_with_identity = original_decode
    student._parent_forward = original_forward
