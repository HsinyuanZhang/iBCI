"""Bias-cancelled carrier tokens for a forward-only SetKV diagnostic.

This module is intentionally additive.  It does not modify the production
``StreamingSpintModel`` or register parameters.  The ordinary coupled token
``fc_in(x + E)`` remains present; a second set of carrier-only tokens is built
from already-trained B3S and decoder weights and appended to the K/V set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch


class SetKVDeltaError(RuntimeError):
    """Raised when the diagnostic is applied outside its frozen topology."""


CarrierMode = Literal["aligned", "row_shuffle", "zero"]
DecodeMode = Literal["baseline", "carrier", "duplicate_activity"]


@dataclass(frozen=True)
class SetKVDeltaForward:
    prediction: torch.Tensor
    identity: torch.Tensor
    activity_tokens: torch.Tensor
    appended_tokens: torch.Tensor | None
    attention_scores: tuple[torch.Tensor, ...]
    decode_mode: str
    carrier_mode: str | None
    permutation: torch.Tensor | None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SetKVDeltaError(message)


def _validate_student(student: Any) -> None:
    _require(not bool(student.training), "SetKV-delta forward diagnostic requires eval mode")
    _require(getattr(student, "decoder_mode", None) == "coupled", "SetKV-delta requires coupled decoder")
    _require(getattr(student, "fixed_slot_router", None) is None, "SetKV-delta forbids fixed-slot routing")
    encoder = getattr(student, "id_encoder", None)
    _require(encoder is not None, "student has no identity encoder")
    _require(getattr(encoder, "side_dim", None) == 4, "SetKV-delta requires B3S side_dim=4")
    _require(getattr(encoder, "electrode_embed_dim", 0) == 0, "SetKV-delta pilot forbids electrode embeddings")
    _require(hasattr(encoder, "post_pool") and hasattr(encoder, "hidden_dim"), "identity encoder lacks B3S post-pool API")
    decoder = getattr(student, "decoder", None)
    _require(decoder is not None and hasattr(decoder, "fc_in"), "student has no compatible decoder")


def deterministic_row_permutation(num_units: int, seed: int, *, device: torch.device) -> torch.Tensor:
    _require(num_units > 1, "row-shuffle control requires at least two units")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(num_units, generator=generator)
    if torch.equal(permutation, torch.arange(num_units)):
        permutation = torch.roll(permutation, shifts=1)
    return permutation.to(device=device)


def carrier_identity_contrast(student: Any, side_features: torch.Tensor) -> torch.Tensor:
    """Return ``psi([0,T])-psi([0,0])`` using the trained B3S post-pool.

    Subtracting the zero-side evaluation removes every carrier-independent
    post-pool bias.  For an exactly zero side tensor the result is bitwise zero.
    """
    _validate_student(student)
    _require(side_features.ndim == 3 and side_features.shape[-1] == 4,
             "side_features must have shape [B,N,4]")
    encoder = student.id_encoder
    zero_activity = side_features.new_zeros(
        side_features.shape[0], side_features.shape[1], int(encoder.hidden_dim)
    )
    zeros = torch.zeros_like(side_features)
    carrier_input = torch.cat([zero_activity, side_features], dim=-1)
    zero_input = torch.cat([zero_activity, zeros], dim=-1)
    return encoder.post_pool(carrier_input) - encoder.post_pool(zero_input)


def carrier_hidden_tokens(student: Any, side_features: torch.Tensor) -> torch.Tensor:
    """Return ``fc_in(E_c)-fc_in(0)`` and cancel the decoder read-in bias."""
    contrast = carrier_identity_contrast(student, side_features)
    decoder = student.decoder
    return decoder.fc_in(contrast) - decoder.fc_in(torch.zeros_like(contrast))


def decode_setkv_delta(
    student: Any,
    neural: torch.Tensor,
    calib_trials: torch.Tensor,
    side_features: torch.Tensor,
    *,
    decode_mode: DecodeMode,
    carrier_mode: CarrierMode = "aligned",
    permutation_seed: int | None = None,
) -> SetKVDeltaForward:
    """Run one frozen coupled forward without registering any new parameter."""
    _validate_student(student)
    _require(neural.ndim == 3, "neural must have shape [B,W,N]")
    _require(calib_trials.ndim == 4, "calib_trials must have shape [B,M,T,N]")
    _require(side_features.ndim == 3 and side_features.shape[-1] == 4,
             "side_features must have shape [B,N,4]")
    batch_size, _window, num_units = neural.shape
    _require(calib_trials.shape[0] == batch_size and calib_trials.shape[-1] == num_units,
             "calibration batch/unit dimensions must match neural")
    _require(side_features.shape[:2] == (batch_size, num_units),
             "side-feature batch/unit dimensions must match neural")

    identity = student.compute_identity(calib_trials, side_features=side_features)
    _require(identity.shape == (batch_size, num_units, neural.shape[1]),
             "identity shape does not match coupled decoder contract")
    if decode_mode == "baseline":
        prediction = student.decode_with_identity(neural, identity)
        activity = student.decoder.fc_in(neural.permute(0, 2, 1) + identity)
        return SetKVDeltaForward(
            prediction=prediction,
            identity=identity,
            activity_tokens=activity,
            appended_tokens=None,
            attention_scores=(),
            decode_mode=decode_mode,
            carrier_mode=None,
            permutation=None,
        )

    activity = student.decoder.fc_in(neural.permute(0, 2, 1) + identity)
    permutation = None
    if decode_mode == "duplicate_activity":
        appended = activity
        carrier_mode_out = None
    else:
        _require(decode_mode == "carrier", f"unknown decode mode: {decode_mode}")
        if carrier_mode == "zero":
            appended = carrier_hidden_tokens(student, torch.zeros_like(side_features))
        else:
            _require(carrier_mode in {"aligned", "row_shuffle"}, f"unknown carrier mode: {carrier_mode}")
            # Compute the aligned carrier exactly once.  Re-running nonlinear
            # post_pool after permuting rows can change low floating-point bits
            # because GEMM row order differs; RS4 is instead a byte-exact row
            # permutation of the already-computed carrier tokens.
            appended = carrier_hidden_tokens(student, side_features)
            if carrier_mode == "row_shuffle":
                _require(permutation_seed is not None, "row_shuffle requires a frozen permutation seed")
                permutation = deterministic_row_permutation(
                    side_features.shape[1], int(permutation_seed), device=side_features.device
                )
                appended = appended.index_select(1, permutation)
        carrier_mode_out = carrier_mode

    key_value = torch.cat([activity, appended], dim=1)
    rep = student.decoder.fc_in(student.decoder.rep).to(activity)
    transformed, attention_scores = student.decoder.transformer(
        rep.repeat(batch_size, 1, 1), key_value
    )
    prediction = student.decoder.fc_out(transformed).permute(0, 2, 1)
    return SetKVDeltaForward(
        prediction=prediction,
        identity=identity,
        activity_tokens=activity,
        appended_tokens=appended,
        attention_scores=tuple(attention_scores),
        decode_mode=decode_mode,
        carrier_mode=carrier_mode_out,
        permutation=permutation,
    )


def setkv_delta_cost_receipt(
    *,
    num_units: int,
    num_queries: int,
    model_dim: int,
    element_bytes: int = 4,
) -> dict[str, int | str]:
    """Return the explicit incremental state and dense-MHA MAC accounting.

    K/V projection cost assumes the current uncached ``nn.MultiheadAttention``
    implementation.  It is an engineering count, not measured latency.
    """
    _require(num_units > 0 and num_queries > 0 and model_dim > 0, "cost dimensions must be positive")
    _require(element_bytes > 0, "element_bytes must be positive")
    return {
        "schema": "setkv_delta_uncached_dense_mha_v1",
        "parameter_delta": 0,
        "extra_token_count": int(num_units),
        "persistent_carrier_token_state_bytes": int(num_units * model_dim * element_bytes),
        "extra_kv_projection_macs_per_window": int(2 * num_units * model_dim * model_dim),
        "extra_attention_macs_per_window": int(2 * num_queries * num_units * model_dim),
        "extra_dense_mha_macs_per_window": int(
            2 * num_units * model_dim * model_dim
            + 2 * num_queries * num_units * model_dim
        ),
    }
