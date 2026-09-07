"""Pure CP-FiLM descriptor and early/late identity operators."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import plan


class H1CalibrationProfileFiLMError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise H1CalibrationProfileFiLMError(message)


def robust_z(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    require(vector.shape == (plan.UNITS,) and np.isfinite(vector).all(), "profile column geometry/nonfinite drift")
    median = float(np.median(vector))
    mad = float(np.median(np.abs(vector - median)))
    scale = max(1.4826 * mad, 1.0e-6)
    result = np.ascontiguousarray((vector - median) / scale, dtype=np.float32)
    require(np.isfinite(result).all(), "robust-z profile is nonfinite")
    return result, {"median": median, "mad": mad, "scale": scale}


def profile_from_arrays(neural: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    rates = np.asarray(neural, dtype=np.float64)
    labels = np.asarray(velocity, dtype=np.float64)
    require(rates.ndim == 2 and rates.shape[1] == plan.UNITS, "profile neural must be [bins,176]")
    require(labels.ndim == 2 and labels.shape == (rates.shape[0], plan.OUTPUTS), "profile velocity must be [bins,7]")
    require(rates.shape[0] >= 2 * plan.MIN_STATE_BINS and np.isfinite(rates).all() and np.isfinite(labels).all(),
            "profile inputs are underspecified or nonfinite")
    speed = np.linalg.norm(labels, axis=1)
    low_q, high_q = (float(value) for value in np.quantile(
        speed, [plan.LOW_QUANTILE, plan.HIGH_QUANTILE], method="linear"
    ))
    require(np.isfinite(low_q) and np.isfinite(high_q) and high_q > low_q, "profile speed quartiles are degenerate")
    low = speed <= low_q
    high = speed >= high_q
    low_count, high_count = int(low.sum()), int(high.sum())
    require(low_count >= plan.MIN_STATE_BINS and high_count >= plan.MIN_STATE_BINS,
            "profile state has too few bins")
    low_rates, high_rates = rates[low], rates[high]
    low_mean, high_mean = low_rates.mean(axis=0), high_rates.mean(axis=0)
    raw = (
        high_mean - low_mean,
        np.log1p(np.maximum(high_mean, 0.0)) - np.log1p(np.maximum(low_mean, 0.0)),
        low_rates.std(axis=0, ddof=0),
        high_rates.std(axis=0, ddof=0),
    )
    columns: list[np.ndarray] = []
    normalizers: list[dict[str, float]] = []
    for values in raw:
        column, evidence = robust_z(values)
        columns.append(column)
        normalizers.append(evidence)
    full = np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)
    mask = np.asarray(plan.PROFILE_MASK, dtype=np.float32)
    masked = np.ascontiguousarray(full * mask[None, :], dtype=np.float32)
    require(full.shape == masked.shape == (plan.UNITS, plan.PROFILE_DIM), "profile output geometry drift")
    require(np.isfinite(full).all() and np.isfinite(masked).all(), "profile output is nonfinite")
    return masked, {
        "raw_profile": full,
        "low_speed_quantile": low_q,
        "high_speed_quantile": high_q,
        "low_state_bins": low_count,
        "high_state_bins": high_count,
        "total_bins": int(rates.shape[0]),
        "profile_mask": list(plan.PROFILE_MASK),
        "column_normalizers": normalizers,
    }


def profile_from_support(record: Any, support_trials: Sequence[float]) -> tuple[np.ndarray, dict[str, Any]]:
    values = tuple(float(value) for value in support_trials)
    require(len(values) == plan.SUPPORT and len(set(values)) == plan.SUPPORT, "profile requires three unique trials")
    legal = (
        np.asarray(record.eval_mask, dtype=bool)
        & np.isfinite(record.trial_num)
        & np.isin(record.trial_num, np.asarray(values, dtype=np.float64))
        & np.isfinite(record.neural).all(axis=1)
        & np.isfinite(record.velocity).all(axis=1)
    )
    masked, evidence = profile_from_arrays(record.neural[legal], record.velocity[legal])
    return masked, {**evidence, "support_trials": list(values)}


def build_film() -> Any:
    import torch

    module = torch.nn.Sequential(
        torch.nn.Linear(plan.CONTEXT_DIM, plan.FILM_RANK),
        torch.nn.ReLU(),
        torch.nn.Linear(plan.FILM_RANK, 2 * plan.HIDDEN_DIM),
    )
    with torch.no_grad():
        module[2].weight.zero_()
        module[2].bias.zero_()
    require(sum(parameter.numel() for parameter in module.parameters()) == plan.FILM_PARAMETERS,
            "CP-FiLM parameter count drift")
    return module


def film_identity(net: Any, activity: Any, carrier: Any, profile: Any, film: Any, *, late: bool) -> Any:
    import torch

    require(isinstance(activity, torch.Tensor) and tuple(activity.shape[1:]) ==
            (plan.SUPPORT, plan.TRIAL_LENGTH, plan.UNITS), "activity geometry drift")
    require(isinstance(carrier, torch.Tensor) and tuple(carrier.shape) ==
            (activity.shape[0], plan.UNITS, plan.CARRIER_DIM), "carrier geometry drift")
    require(isinstance(profile, torch.Tensor) and tuple(profile.shape) ==
            (activity.shape[0], plan.UNITS, plan.PROFILE_DIM), "profile geometry drift")
    encoded = net.carrier_pre_pool(activity.permute(0, 1, 3, 2))
    effective = torch.zeros_like(carrier) if net.zero_carrier else carrier.to(encoded)
    context = torch.cat((effective, profile.to(encoded)), dim=-1)
    modulation = film(context)
    gamma, beta = modulation.chunk(2, dim=-1)
    if late:
        modulated = (1.0 + gamma[:, None]) * encoded + beta[:, None]
        joined = torch.cat((modulated, effective[:, None].expand(-1, plan.SUPPORT, -1, -1)), dim=-1)
        identity = net.carrier_post_pool(joined).mean(dim=1)
    else:
        mean_feat = encoded.mean(dim=1)
        modulated = (1.0 + gamma) * mean_feat + beta
        identity = net.carrier_post_pool(torch.cat((modulated, effective), dim=-1))
    require(tuple(identity.shape) == (activity.shape[0], plan.UNITS, plan.WINDOW), "FiLM identity geometry drift")
    require(bool(torch.isfinite(identity).all()), "FiLM identity is nonfinite")
    return identity


__all__ = (
    "H1CalibrationProfileFiLMError", "build_film", "film_identity", "profile_from_arrays",
    "profile_from_support", "require", "robust_z",
)
