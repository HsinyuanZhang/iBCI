"""vstate-688 carrier math (user directive 2026-09-09).

Recipe (M2 vstate4 of PLAN_CARRIER_ITERATION_M2_688_20260909.md section 3.2,
adapted to the 688 rate primitives), verbatim anchor from the user directive:

  "vstate（主臂）：R700/H300 窗口内的 100ms 块（照 688 率原语：spike-time
  searchsorted 半开计数/块时长）；状态 W=[softplus(v_x/rms_x),
  softplus(−v_x/rms_x), softplus(v_y/rms_y), softplus(−v_y/rms_y)]，
  cursor_vel 块均值；rms 只用 train sessions 校准速度拟合；Poisson 标准化
  （Δ=块时长）+ n0=10 块收缩；读出 a=R₊ₓ−R₋ₓ, c=R₊y−R₋y, m=hypot,
  b=meanₖR−R_hold（hold=H300 块的条件响应，保持 δ_b 语义）；列归一
  train-only mean/std。支持预算对齐现有 t4（M10 的同 10 个 trial）"

DISCLOSURE (against the legacy 688 sparse-label discipline): this carrier
consumes DENSE cursor_vel calibration labels of the M10 support trials.
The old 688 dense-null (dense-speed CP-FiLM) does NOT contradict this arm
along any of the three axes that matter: (1) sign -- the null used unsigned
speed quantile profiles, vstate uses signed direction states; (2) seat --
the null stacked the dense profile ON TOP of the intact T4 (additive
overlay), vstate REPLACES T4 in the carrier seat under the matched M10
support; (3) path -- the null reached the model through the CP-FiLM side
modulation, vstate rides the carrier token path.  The same three-axis
note is recorded in the bench README and the DESIGN doc arm table.

All functions here are pure numpy (torch only inside remelt_e0) so the
constructor correctness tests can hand-compute examples on CPU.
"""
from __future__ import annotations

import sys
from typing import Any, Sequence

import numpy as np

from . import plan

# btransform_unified_v2.carrier_profile_v3 is the shared estimator template
# (one implementation for M2/688, per PLAN_CARRIER_ITERATION_M2_688 section 1).
# Its package __init__ pulls the model stack (which needs btransform_unified_v1),
# so both src roots are put on sys.path exactly like the bench runner does.
for _src in (plan.workspace_root() / "btransform_unified_v2" / "src",
             plan.workspace_root() / "btransform_unified_v1" / "src"):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from btransform_unified_v2 import carrier_profile_v3 as v3  # noqa: E402


# --- block construction over a frozen phase window --------------------------
def block_edges(
    window_start: float, window_stop: float,
    block_seconds: float = plan.VSTATE_BLOCK_SECONDS,
) -> np.ndarray:
    """Half-open block edges tiling [window_start, window_stop).

    The window duration must be an exact multiple of the block length (R700
    -> 7 blocks of 100 ms, H300 -> 3); fail closed otherwise."""
    duration = float(window_stop) - float(window_start)
    if duration <= 0.0:
        raise ValueError(f"window must be non-degenerate, got [{window_start}, {window_stop})")
    n_blocks = duration / block_seconds
    if abs(n_blocks - round(n_blocks)) > 1e-9:
        raise ValueError(
            f"window duration {duration}s is not an integer multiple of the "
            f"{block_seconds}s block"
        )
    n_blocks = int(round(n_blocks))
    # linspace pins edges[0] == window_start and edges[-1] == window_stop
    # exactly, so the half-open [edge_k, edge_{k+1}) counting convention is
    # exact at the final edge (start + k*block accumulation can drift past
    # window_stop by 1 ulp and would swallow boundary spikes).
    return np.linspace(float(window_start), float(window_stop), n_blocks + 1)


def block_rate_matrix(
    spike_times_per_unit: Sequence[np.ndarray], edges: np.ndarray,
    block_seconds: float = plan.VSTATE_BLOCK_SECONDS,
) -> np.ndarray:
    """Per-block firing rates [n_blocks, n_units] (Hz).

    688 rate primitive family: spike-time searchsorted half-open counts
    [edge_k, edge_{k+1}) divided by the block duration (the same convention
    as unit_side_features._pool_trial_rate_matrix, at block resolution)."""
    counts = np.empty((len(edges) - 1, len(spike_times_per_unit)), dtype=np.float64)
    for unit, spikes in enumerate(spike_times_per_unit):
        spikes = np.asarray(spikes, dtype=np.float64)
        if spikes.size and not np.all(spikes[:-1] <= spikes[1:]):
            raise ValueError(f"unit {unit}: spike_times must be non-decreasing")
        left = np.searchsorted(spikes, edges[:-1], side="left")
        right = np.searchsorted(spikes, edges[1:], side="left")
        counts[:, unit] = (right - left) / block_seconds
    return counts


def block_velocity_means(
    velocity_times: np.ndarray,
    velocity_values: np.ndarray,
    edges: np.ndarray,
    subbins: int = plan.VSTATE_VELOCITY_SUBBINS,
    bin_seconds: float = plan.VSTATE_VELOCITY_BIN_SECONDS,
) -> np.ndarray:
    """cursor_vel block means [n_blocks, 2].

    Independent 688 implementation aligned to bin centers (multisession
    datamodule's reading, reimplemented here read-only): linear interpolation
    of cursor_vel evaluated at the ``subbins`` equally spaced 20 ms bin
    centers inside each 100 ms block (out-of-range centers read 0.0, the
    datamodule's fill_value), then averaged over the centers -- i.e. the
    mean of the 5 sub-bins inside the block."""
    velocity_times = np.asarray(velocity_times, dtype=np.float64)
    velocity_values = np.asarray(velocity_values, dtype=np.float64)
    if velocity_values.ndim != 2 or velocity_values.shape[1] != 2:
        raise ValueError(f"cursor_vel must be [T, 2], got {velocity_values.shape}")
    if velocity_times.shape != (velocity_values.shape[0],):
        raise ValueError("velocity times/values length mismatch")
    if abs(bin_seconds * subbins - (edges[1] - edges[0])) > 1e-9:
        raise ValueError(
            f"{subbins} x {bin_seconds}s sub-bins must tile the "
            f"{edges[1] - edges[0]}s block exactly"
        )
    centers = (
        edges[:-1, None]
        + (np.arange(subbins, dtype=np.float64)[None, :] + 0.5) * bin_seconds
    ).ravel()
    interp = np.zeros((centers.size, velocity_values.shape[1]), dtype=np.float64)
    inside = (centers >= velocity_times[0]) & (centers <= velocity_times[-1])
    if bool(inside.any()):
        interp[inside] = np.stack(
            [
                np.interp(centers[inside], velocity_times, velocity_values[:, axis])
                for axis in range(velocity_values.shape[1])
            ],
            axis=1,
        )
    return interp.reshape(len(edges) - 1, subbins, -1).mean(axis=1)


# --- rms calibration (train sessions only) ----------------------------------
def fit_velocity_rms(block_velocities: Sequence[np.ndarray]) -> np.ndarray:
    """Per-axis rms of the calibration block velocities, fitted over the
    TRAIN sessions' blocks only (exp1_narrow = 9 train sessions, exp2_full =
    27 -- resolved by the cache builder, enforced by construction)."""
    joined = np.concatenate([np.asarray(v, dtype=np.float64) for v in block_velocities], axis=0)
    if joined.ndim != 2 or joined.shape[0] == 0 or joined.shape[1] != 2:
        raise ValueError(f"block velocities must stack to [n_blocks, 2], got {joined.shape}")
    rms = np.sqrt(np.mean(np.square(joined), axis=0))
    if not np.all(np.isfinite(rms)) or np.any(rms <= 0.0):
        raise RuntimeError(f"velocity rms calibration failed: {rms}")
    return rms


# --- the vstate estimator ----------------------------------------------------
def vstate_carrier_from_blocks(
    r700_rates: np.ndarray,
    r700_vel: np.ndarray,
    h300_rates: np.ndarray,
    h300_vel: np.ndarray,
    rms: np.ndarray,
    n0: float = plan.VSTATE_N0_BLOCKS,
    block_seconds: float = plan.VSTATE_BLOCK_SECONDS,
) -> np.ndarray:
    """Raw vstate carrier rows [n_units, 4] = [a, c, m, b].

    Poisson standardization on the R700 blocks (delta = block duration, the
    same affine then applied to the H300 blocks); signed-state conditional
    responses R / R_hold with n0-block shrinkage; readout
    a = R(+x) - R(-x), c = R(+y) - R(-y), m = hypot(a, c),
    b = mean_k R - mean_k R_hold (hold = H300 blocks' own conditional
    response, preserving the delta_b motion-minus-hold semantics)."""
    r700_rates = np.asarray(r700_rates, dtype=np.float64)
    h300_rates = np.asarray(h300_rates, dtype=np.float64)
    r700_vel = np.asarray(r700_vel, dtype=np.float64)
    h300_vel = np.asarray(h300_vel, dtype=np.float64)
    if r700_rates.ndim != 2 or r700_rates.shape[1] == 0:
        raise ValueError(f"r700_rates must be [blocks, units], got {r700_rates.shape}")
    if h300_rates.shape[1] != r700_rates.shape[1]:
        raise ValueError("r700/h300 unit count mismatch")
    z_r, rate_mean, noise_rate = v3.poisson_standardize(r700_rates, block_seconds)
    z_h = v3.apply_affine(h300_rates, rate_mean, noise_rate)
    w_r = v3.signed_state_weights(r700_vel, rms)
    w_h = v3.signed_state_weights(h300_vel, rms)
    response = v3.conditional_response(z_r, w_r, n0)
    hold = v3.conditional_response(z_h, w_h, n0)
    a = response[:, 0] - response[:, 1]
    c = response[:, 2] - response[:, 3]
    m = np.hypot(a, c)
    b = response.mean(axis=1) - hold.mean(axis=1)
    raw = np.column_stack((a, c, m, b))
    if not np.isfinite(raw).all():
        raise RuntimeError("nonfinite raw vstate carrier")
    return raw


# --- E0 co-variant remelt (same formula as the t4/u1 cache builders) --------
def remelt_e0(
    student: Any,
    calib_trials: np.ndarray,
    side_rows: np.ndarray,
    n_pad: int,
    e0_dim: int = plan.E0_DIM,
) -> np.ndarray:
    """E0 = post_pool(cat(pre_pool(calib_trials).mean, side_rows)) padded to
    [n_pad, e0_dim] -- the same co-variant E0 formula the frozen cache and
    build_carrier_cache._remelt_e0 use (dandi688_train.build_rows family).
    ``side_rows`` is [n_real, CARRIER_DIM]: the normalized vstate carrier
    (vstate/z arms) or an all-zero side (f_labelfree arm)."""
    import torch

    calib_trials = np.asarray(calib_trials)
    units = int(calib_trials.shape[-1])
    if int(n_pad) < units:
        raise ValueError(f"n_pad {n_pad} smaller than real units {units}")
    side_rows = np.asarray(side_rows, dtype=np.float32)
    if side_rows.shape != (units, plan.CARRIER_DIM):
        raise ValueError(
            f"side_rows must be [{units}, {plan.CARRIER_DIM}], got {side_rows.shape}"
        )
    cal = torch.from_numpy(np.ascontiguousarray(calib_trials, dtype=np.float32)).unsqueeze(0)
    side = torch.from_numpy(side_rows).unsqueeze(0)
    with torch.inference_mode():
        pooled = student.id_encoder.pre_pool(cal.permute(0, 1, 3, 2)).mean(1)
        e = student.id_encoder.post_pool(torch.cat((pooled, side), -1))[0].cpu().numpy()
    e0 = np.zeros((int(n_pad), e0_dim), dtype=np.float32)
    e0[:units] = e
    return e0


__all__ = [
    "block_edges",
    "block_rate_matrix",
    "block_velocity_means",
    "fit_velocity_rms",
    "vstate_carrier_from_blocks",
    "remelt_e0",
]
