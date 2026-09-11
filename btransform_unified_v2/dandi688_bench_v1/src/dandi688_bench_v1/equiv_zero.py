"""equiv_zero arm math: the param-matched fixed random E0 pathway.

User ruling 2026-09-09 (component ablation, capacity control row): the
no-calibration arms must be accompanied by a network of EQUIVALENT encoder
parameter count that "does not form any identity but simply starts decoding
on the new session" -- paraphrased implementation contract:

  - the full arm's E0 comes from the frozen B3S id_encoder
    (pre_pool activity features -> post_pool(cat(mean, side))).  equiv_zero
    replaces the LABEL-CARRYING sub-pathway ``post_pool`` with a FIXED
    RANDOM projection of exactly the same structure and parameter count
    (Linear-for-Linear mirror; seed = plan.SEED = 42, drawn once, never
    trained, requires_grad False);
  - the random projection acts on the pre_pool ACTIVITY MEAN with a ZERO
    side input: same input width, no behavioral labels, nonzero output
    (activity bytes flow, nothing encodable as identity does);
  - every session gets the SAME random projection applied to its own
    activity mean (deterministic, session-independent weights);
  - the carrier token keeps the t4 direct-token SHAPE but its VALUES are
    zeroed (shape-determined parameter count is unchanged by zeroing);
  - purpose: separate "missing information" from "missing pathway/capacity"
    -- if floor/f_labelfree lose as much as equiv_zero does, the loss is
    pathway/capacity, not information; if equiv_zero ~ t4, the pathway
   itself carries structural value even without encoded identity.

Parameter-parity law (asserted here and re-asserted by the tests):
  1. trainable-parameter count of the trained model is identical between
     the t4 and equiv_zero arms (same model class / geometry -- the bank is
     an input, not model parameters);
  2. the random projection's parameter count is EXACTLY the frozen
     post_pool's (Linear-mirrored structure).

The E0 produced here is baked into the "equiv_zero" variant cache by
scripts/build_vstate_cache.py; the runner-side arm transform then only
zeroes the carrier (arms.ARM_SPECS["equiv_zero"]["e0"] == "identity" -- the
no-identity property lives in the cache bytes, the same law as f_labelfree).

Pure numpy/torch on CPU; deterministic under the frozen seed.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import plan

EQUIV_ZERO_SEED = plan.SEED  # 42: the frozen random-projection seed


def _iter_linear(module):
    """Yield the nn.Linear layers of a post_pool-like module (bare Linear or
    a Sequential stack of them)."""
    from torch import nn

    if isinstance(module, nn.Linear):
        yield module
        return
    for layer in module:
        if isinstance(layer, nn.Linear):
            yield layer


def _mirror_structure(reference) -> list:
    """Mirror the reference post_pool module list Linear-for-Linear.

    The frozen B3S post_pool is an affine stack of nn.Linear (+ ReLU)
    layers (streaming_encoders._build_affine_stack); a bare nn.Linear
    (single-layer stack) is also accepted.  Anything else fails closed so
    the mirror can never silently drop parameters."""
    from torch import nn

    if isinstance(reference, nn.Linear):
        layers: list[Any] = [reference]
    else:
        layers = list(reference)
    mirrored: list[nn.Module] = []
    for module in layers:
        if isinstance(module, nn.Linear):
            fresh = nn.Linear(
                int(module.in_features), int(module.out_features),
                bias=module.bias is not None,
            )
            mirrored.append(fresh)
        elif isinstance(module, nn.ReLU):
            mirrored.append(nn.ReLU())
        else:
            raise RuntimeError(
                f"equiv_zero mirror: unsupported post_pool layer "
                f"{type(module).__name__}; the param-parity law requires an "
                f"exact Linear/ReLU mirror"
            )
    return mirrored


def param_count(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def weights_sha256(module) -> str:
    """SHA over the module's parameter bytes in definition order (dtype,
    shape, raw bytes per tensor -- plan.array_digest law)."""
    digest_parts = [
        plan.array_digest(p.detach().cpu().numpy())
        for p in module.parameters()
    ]
    return plan.obj_sha256({"tensors": digest_parts})


def build_fixed_projection(student: Any, seed: int = EQUIV_ZERO_SEED):
    """Build the fixed random replacement of the frozen post_pool pathway.

    Deterministic: the weights are drawn exactly once under
    torch.manual_seed(seed) (global RNG state saved/restored around the
    draw so callers see no RNG side effects).  All parameters are frozen
    (requires_grad False).  Fails closed unless the parameter count and the
    Linear geometry match the reference exactly."""
    import torch
    from torch import nn

    reference = student.id_encoder.post_pool
    state = torch.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        mirrored = _mirror_structure(reference)
    finally:
        torch.set_rng_state(state)
    projection = nn.Sequential(*mirrored)
    for parameter in projection.parameters():
        parameter.requires_grad_(False)
    ref_params = param_count(reference)
    new_params = param_count(projection)
    if ref_params != new_params:
        raise RuntimeError(
            f"equiv_zero param-parity violated: random projection has "
            f"{new_params} params, frozen post_pool has {ref_params}"
        )
    ref_linears = [
        (int(m.in_features), int(m.out_features))
        for m in _iter_linear(reference)
    ]
    new_linears = [
        (int(m.in_features), int(m.out_features))
        for m in _iter_linear(projection)
    ]
    if ref_linears != new_linears:
        raise RuntimeError(
            f"equiv_zero geometry drift: {new_linears} vs frozen {ref_linears}"
        )
    return projection


def projection_receipt(student: Any, projection, seed: int = EQUIV_ZERO_SEED) -> dict[str, Any]:
    """Receipt block of the fixed projection (baked into the cache contract
    and every downstream preflight/train/score receipt)."""
    reference = student.id_encoder.post_pool
    return {
        "name": "equiv_zero_fixed_random_projection",
        "recipe": "user ruling 2026-09-09 (component ablation): same-"
                  "parameter-count random replacement of the label-carrying "
                  "post_pool; acts on the pre_pool activity mean with a "
                  "zero side; drawn once, never trained, no identity",
        "seed": int(seed),
        "weights_sha256": weights_sha256(projection),
        "param_count_random": param_count(projection),
        "param_count_reference": param_count(reference),
        "param_parity": param_count(projection) == param_count(reference),
        "linear_geometry": [
            [int(m.in_features), int(m.out_features)]
            for m in _iter_linear(projection)
        ],
        "input": "cat(pre_pool(calib).mean(1), zero side [units, 4])",
        "trainable": False,
    }


def e0_from_projection(
    projection,
    student: Any,
    calib_trials: np.ndarray,
    n_pad: int,
    e0_dim: int = plan.E0_DIM,
) -> np.ndarray:
    """E0 slot content of the equiv_zero arm: the fixed random projection
    applied to the pre_pool activity mean with a zero side, padded to
    [n_pad, e0_dim] exactly like vstate.remelt_e0.  Nonzero by construction
    (asserted): activity bytes flow through the random pathway."""
    import torch

    calib_trials = np.asarray(calib_trials)
    units = int(calib_trials.shape[-1])
    if int(n_pad) < units:
        raise ValueError(f"n_pad {n_pad} smaller than real units {units}")
    cal = torch.from_numpy(np.ascontiguousarray(calib_trials, dtype=np.float32)).unsqueeze(0)
    with torch.inference_mode():
        pooled = student.id_encoder.pre_pool(cal.permute(0, 1, 3, 2)).mean(1)
        side = torch.zeros(
            pooled.shape[0], pooled.shape[1], plan.CARRIER_DIM,
            dtype=pooled.dtype,
        )
        e = projection(torch.cat((pooled, side), -1))[0].cpu().numpy()
    if e.shape != (units, e0_dim):
        raise RuntimeError(
            f"equiv_zero projection output must be [{units}, {e0_dim}], got {e.shape}"
        )
    if not bool(np.any(e)):
        raise RuntimeError("equiv_zero E0 must be nonzero (activity pathway)")
    e0 = np.zeros((int(n_pad), e0_dim), dtype=np.float32)
    e0[:units] = e
    return e0


__all__ = [
    "EQUIV_ZERO_SEED",
    "build_fixed_projection",
    "projection_receipt",
    "e0_from_projection",
    "param_count",
    "weights_sha256",
]
