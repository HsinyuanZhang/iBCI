"""Route-owned paired training step for PACD V1.

The Cell-D model remains unmodified. The paired operator replays the Python
and Torch RNG states so the anchor and short forwards receive exactly the
same dynamic whole-unit dropout probability and mask, then leaves the global
RNG at the state produced by one ordinary forward.
"""

from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from . import plan


class PACDError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PACDError(message)


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


@dataclass
class RNGSnapshot:
    python_state: object
    torch_cpu_state: torch.Tensor
    torch_cuda_states: tuple[torch.Tensor, ...] | None


def capture_rng() -> RNGSnapshot:
    cuda_states: tuple[torch.Tensor, ...] | None = None
    if torch.cuda.is_initialized():
        cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return RNGSnapshot(
        python_state=random.getstate(),
        torch_cpu_state=torch.random.get_rng_state().clone(),
        torch_cuda_states=cuda_states,
    )


def restore_rng(snapshot: RNGSnapshot) -> None:
    random.setstate(snapshot.python_state)
    torch.random.set_rng_state(snapshot.torch_cpu_state)
    if snapshot.torch_cuda_states is not None:
        _require(torch.cuda.is_initialized(), "CUDA RNG disappeared during paired forward")
        torch.cuda.set_rng_state_all(list(snapshot.torch_cuda_states))


def rng_sha256(snapshot: RNGSnapshot) -> str:
    digest = hashlib.sha256(repr(snapshot.python_state).encode("utf-8"))
    digest.update(snapshot.torch_cpu_state.cpu().numpy().tobytes(order="C"))
    if snapshot.torch_cuda_states is None:
        digest.update(b"NO_CUDA")
    else:
        for state in snapshot.torch_cuda_states:
            digest.update(state.cpu().numpy().tobytes(order="C"))
    return digest.hexdigest()


@contextmanager
def record_unit_dropout_masks() -> Iterator[list[dict[str, object]]]:
    """Passively record only the Cell-D all-ones two-dimensional unit mask.

    Other functional-dropout calls, if present, pass through unchanged and are
    deliberately not classified as the whole-unit mask.
    """

    original = torch.nn.functional.dropout
    records: list[dict[str, object]] = []

    def recording(input, p=0.5, training=True, inplace=False):
        output = original(input, p=p, training=training, inplace=inplace)
        is_unit_mask = (
            bool(training)
            and input.ndim == 2
            and input.numel() > 0
            and bool(torch.all(input == 1).item())
        )
        if is_unit_mask:
            records.append(
                {
                    "p": float(p),
                    "shape": list(input.shape),
                    "mask_sha256": tensor_sha256(output),
                    "retained": int((output != 0).sum().item()),
                    "total": int(output.numel()),
                }
            )
        return output

    torch.nn.functional.dropout = recording
    try:
        yield records
    finally:
        torch.nn.functional.dropout = original


def supervised_valid_loss(
    prediction: torch.Tensor,
    behavior: torch.Tensor,
    *,
    pad_value: float,
) -> tuple[torch.Tensor, int]:
    _require(prediction.shape == behavior.shape, "prediction/behavior shape drift")
    valid = (behavior != pad_value).all(dim=-1)
    valid_count = int(valid.sum().item())
    _require(valid_count > 0, "paired batch has no valid behavior bins")
    diff2 = ((prediction - behavior) ** 2).sum(dim=-1)
    loss = (diff2 * valid).sum() / (valid.sum() * behavior.shape[-1])
    _require(bool(torch.isfinite(loss).item()), "non-finite paired task loss")
    return loss, valid_count


def _grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return 0.0
    value = torch.stack([gradient.detach().norm() for gradient in grads]).pow(2).sum().sqrt()
    _require(bool(torch.isfinite(value).item()), "non-finite gradient norm")
    return float(value.item())


def _materialized_parameters(parameters: list[torch.nn.Parameter]) -> tuple[list[torch.nn.Parameter], dict[str, int]]:
    """Exclude only inactive lazy parameters from evidence-only autograd."""
    from torch.nn.parameter import UninitializedParameter
    materialized = [parameter for parameter in parameters if not isinstance(parameter, UninitializedParameter)]
    return materialized, {"materialized": len(materialized), "skipped_uninitialized_lazy": len(parameters) - len(materialized)}


def _autograd_grad_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> float:
    """Evidence-only branch norm without mutating ``.grad`` or route RNG."""
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    present = [gradient.detach().norm() for gradient in gradients if gradient is not None]
    if not present:
        return 0.0
    value = torch.stack(present).pow(2).sum().sqrt()
    _require(bool(torch.isfinite(value).item()), "non-finite branch gradient norm")
    return float(value.item())


def _autograd_grad_cosine(loss_left: torch.Tensor, loss_right: torch.Tensor,
                          parameters: list[torch.nn.Parameter]) -> float:
    """Evidence-only cosine; intentionally called only at full-run sentinels."""
    if not parameters:
        return 0.0
    left = torch.autograd.grad(loss_left, parameters, retain_graph=True, allow_unused=True)
    right = torch.autograd.grad(loss_right, parameters, retain_graph=True, allow_unused=True)
    numerator = sum((a.detach() * b.detach()).sum() for a, b in zip(left, right) if a is not None and b is not None)
    left_sq = sum((a.detach() ** 2).sum() for a in left if a is not None)
    right_sq = sum((b.detach() ** 2).sum() for b in right if b is not None)
    if float(left_sq.item()) == 0.0 or float(right_sq.item()) == 0.0:
        return 0.0
    value = numerator / (left_sq.sqrt() * right_sq.sqrt())
    _require(bool(torch.isfinite(value).item()), "non-finite branch gradient cosine")
    return float(value.item())


def assert_finite_materialized_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Check finite state without touching inactive lazy parameters.

    Cell-D retains an inactive ``LazyModule`` branch after its ordinary coupled
    forward.  Calling ``numel`` on that ``UninitializedParameter`` is illegal;
    it is neither model state used by this training route nor evidence of a
    non-finite update.  The check therefore covers every materialized
    parameter and records exactly how many inactive lazy parameters were
    intentionally skipped.
    """
    from torch.nn.parameter import UninitializedParameter

    materialized = 0
    skipped_lazy = 0
    for parameter in model.parameters():
        if isinstance(parameter, UninitializedParameter):
            skipped_lazy += 1
            continue
        materialized += 1
        _require(bool(torch.isfinite(parameter.detach()).all().item()), "non-finite model state")
    return {"materialized": materialized, "skipped_uninitialized_lazy": skipped_lazy}


def paired_train_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    neural: torch.Tensor,
    behavior: torch.Tensor,
    calibration: torch.Tensor,
    side_features: torch.Tensor,
    short_m: int,
    pad_value: float,
    encoder_parameters: list[torch.nn.Parameter],
    decoder_parameters: list[torch.nn.Parameter],
    collect_branch_evidence: bool = False,
    zero_encoder_policy: str = "reject",
) -> dict[str, Any]:
    """Run one PACD pair and exactly one optimizer update."""

    _require(short_m in {4, 10, 30}, "short prefix must be M4, M10, or M30")
    _require(
        zero_encoder_policy in {"reject", "accept_if_paired_unit_mask_empty"},
        "unknown zero-encoder policy",
    )
    _require(calibration.ndim >= 3, "calibration tensor rank drift")
    _require(calibration.shape[0] == neural.shape[0], "calibration/query batch drift")
    _require(calibration.shape[1] >= plan.FULL_M, "fewer than 30 calibration trials")
    _require(behavior.shape[0] == neural.shape[0], "behavior/query batch drift")
    _require(side_features.shape[0] == neural.shape[0], "side/query batch drift")
    _require(model.training, "PACD paired step requires training mode")

    full = calibration[:, : plan.FULL_M]
    short = calibration[:, :short_m]
    full_block_before = tensor_sha256(calibration)
    optimizer.zero_grad(set_to_none=True)

    rng_before = capture_rng()
    with record_unit_dropout_masks() as mask_records:
        prediction_full, identity_full = model(
            neural, calib_trials=full, side_features=side_features
        )
        rng_after_anchor_forward = capture_rng()
        loss_full, valid_full = supervised_valid_loss(
            prediction_full, behavior, pad_value=pad_value
        )
        prediction_full_sha = tensor_sha256(prediction_full)
        identity_full_sha = tensor_sha256(identity_full)
        (0.5 * loss_full).backward(retain_graph=collect_branch_evidence)
        rng_after_anchor_backward = capture_rng()
        _require(
            rng_sha256(rng_after_anchor_forward) == rng_sha256(rng_after_anchor_backward),
            "anchor backward unexpectedly consumed RNG",
        )
        anchor_encoder_grad = _grad_norm(encoder_parameters)
        anchor_decoder_grad = _grad_norm(decoder_parameters)

        restore_rng(rng_before)
        try:
            prediction_short, identity_short = model(
                neural, calib_trials=short, side_features=side_features
            )
            rng_after_short_forward = capture_rng()
        finally:
            restore_rng(rng_after_anchor_backward)

        _require(
            rng_sha256(rng_after_short_forward) == rng_sha256(rng_after_anchor_forward),
            "paired short forward did not reproduce the anchor RNG transition",
        )
        loss_short, valid_short = supervised_valid_loss(
            prediction_short, behavior, pad_value=pad_value
        )
        _require(valid_short == valid_full, "paired valid-bin count drift")
        branch_evidence: dict[str, float] = {}
        if collect_branch_evidence:
            # This has real autograd cost, so full training only requests it
            # at four fixed sentinels per epoch.  Default V1/smoke behavior
            # remains byte-for-byte the original single-backward route.
            encoder_materialized, encoder_counts = _materialized_parameters(encoder_parameters)
            decoder_materialized, decoder_counts = _materialized_parameters(decoder_parameters)
            branch_evidence = {
                "short_encoder_grad_norm": _autograd_grad_norm(0.5 * loss_short, encoder_materialized),
                "short_decoder_grad_norm": _autograd_grad_norm(0.5 * loss_short, decoder_materialized),
                "encoder_branch_gradient_cosine": _autograd_grad_cosine(0.5 * loss_full, 0.5 * loss_short, encoder_materialized),
                "decoder_branch_gradient_cosine": _autograd_grad_cosine(0.5 * loss_full, 0.5 * loss_short, decoder_materialized),
                "branch_evidence_parameter_counts": {"encoder": encoder_counts, "decoder": decoder_counts},
            }
        prediction_short_sha = tensor_sha256(prediction_short)
        identity_short_sha = tensor_sha256(identity_short)
        (0.5 * loss_short).backward()
        rng_after_short_backward = capture_rng()
        _require(
            rng_sha256(rng_after_short_backward) == rng_sha256(rng_after_anchor_backward),
            "short backward unexpectedly consumed RNG after paired replay",
        )

    _require(len(mask_records) == 2, "expected exactly two whole-unit dropout calls")
    _require(mask_records[0] == mask_records[1], "paired whole-unit dropout mask drift")
    _require(
        tensor_sha256(calibration) == full_block_before,
        "paired operator mutated the caller calibration tensor",
    )
    if short_m == plan.FULL_M:
        _require(
            prediction_full_sha == prediction_short_sha,
            "P0 full/full predictions differ under identical RNG",
        )
        _require(identity_full_sha == identity_short_sha, "P0 full/full identities differ")

    combined_encoder_grad = _grad_norm(encoder_parameters)
    combined_decoder_grad = _grad_norm(decoder_parameters)
    # A Cell-D dynamic whole-unit dropout realization may mask every unit in
    # the batch.  In that *observable, paired* state the identity encoder is
    # mathematically disconnected while the decoder query/representation path
    # remains trainable.  It is not an optimizer or graph fault.  The default
    # historical policy stays strict; the successor-only policy admits exactly
    # this case and records it below.  No loss/model/RNG/optimizer behavior is
    # changed by this evidence-policy branch.
    all_units_dropped = int(mask_records[0]["retained"]) == 0
    zero_encoder_accepted = False
    if combined_encoder_grad == 0.0:
        _require(
            zero_encoder_policy == "accept_if_paired_unit_mask_empty" and all_units_dropped,
            "encoder gradient is zero",
        )
        zero_encoder_accepted = True
    _require(combined_decoder_grad > 0.0, "decoder gradient is zero")

    optimizer.step()
    rng_after_pair = capture_rng()
    _require(
        rng_sha256(rng_after_pair) == rng_sha256(rng_after_anchor_backward),
        "paired optimizer step changed the one-forward global RNG transition",
    )
    parameter_finiteness = assert_finite_materialized_parameters(model)

    result = {
        "anchor_m": plan.FULL_M,
        "short_m": int(short_m),
        "loss_anchor": float(loss_full.detach().item()),
        "loss_short": float(loss_short.detach().item()),
        "loss_combined": float((0.5 * (loss_full.detach() + loss_short.detach())).item()),
        "valid_bins": valid_full,
        "dropout": dict(mask_records[0]),
        "dropout_pair_equal": True,
        "rng_before_sha256": rng_sha256(rng_before),
        "rng_after_one_forward_sha256": rng_sha256(rng_after_anchor_forward),
        "rng_after_pair_sha256": rng_sha256(rng_after_pair),
        "rng_short_transition_equal": True,
        "rng_pair_transition_equal": True,
        "calibration_full_sha256": full_block_before,
        "calibration_full_after_sha256": tensor_sha256(calibration),
        "calibration_anchor_sha256": tensor_sha256(full),
        "calibration_short_sha256": tensor_sha256(short),
        "prediction_anchor_sha256": prediction_full_sha,
        "prediction_short_sha256": prediction_short_sha,
        "prediction_pair_equal": prediction_full_sha == prediction_short_sha,
        "identity_anchor_sha256": identity_full_sha,
        "identity_short_sha256": identity_short_sha,
        "identity_pair_equal": identity_full_sha == identity_short_sha,
        "anchor_encoder_grad_norm": anchor_encoder_grad,
        "anchor_decoder_grad_norm": anchor_decoder_grad,
        "combined_encoder_grad_norm": combined_encoder_grad,
        "combined_decoder_grad_norm": combined_decoder_grad,
        "optimizer_steps": 1,
        "parameter_finiteness": parameter_finiteness,
        **branch_evidence,
    }
    # Do not alter the default V1/V2 receipt schema.  V3's explicit policy is
    # the only route that needs these explanatory evidence fields.
    if zero_encoder_policy != "reject":
        result["zero_encoder_policy"] = zero_encoder_policy
        result["combined_encoder_zero_accepted"] = zero_encoder_accepted
        result["combined_encoder_zero_reason"] = (
            "all_units_dropped_valid_zero" if zero_encoder_accepted else None
        )
        result["paired_unit_mask_empty"] = all_units_dropped
    return result
