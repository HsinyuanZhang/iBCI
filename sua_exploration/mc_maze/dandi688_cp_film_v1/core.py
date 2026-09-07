"""Pure profile, FiLM, digest, and decision helpers."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np

from . import plan


class CPFiLMError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CPFiLMError(message)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def robust_z(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    require(vector.size > 0 and np.isfinite(vector).all(), "profile column empty/nonfinite")
    median = float(np.median(vector))
    mad = float(np.median(np.abs(vector - median)))
    scale = max(1.4826 * mad, 1.0e-6)
    result = np.ascontiguousarray((vector - median) / scale, dtype=np.float32)
    require(np.isfinite(result).all(), "profile robust-z nonfinite")
    return result, {"median": median, "mad": mad, "scale": scale}


def profile_from_bins(neural: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    x = np.asarray(neural, dtype=np.float64)
    y = np.asarray(velocity, dtype=np.float64)
    require(x.ndim == 2 and x.shape[0] > 0, "profile neural must be [bins,units]")
    require(y.shape == (x.shape[0], 2), "profile velocity must be [bins,2]")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "profile inputs nonfinite")
    speed = np.linalg.norm(y, axis=1)
    low_q, high_q = (float(v) for v in np.quantile(speed, (0.25, 0.75), method="linear"))
    require(np.isfinite(low_q) and np.isfinite(high_q) and high_q > low_q, "speed quartiles degenerate")
    low = speed <= low_q
    high = speed >= high_q
    require(int(low.sum()) >= 2 and int(high.sum()) >= 2, "profile state has too few bins")
    low_x, high_x = x[low], x[high]
    low_mean, high_mean = low_x.mean(0), high_x.mean(0)
    raw = (
        high_mean - low_mean,
        np.log1p(np.maximum(high_mean, 0.0)) - np.log1p(np.maximum(low_mean, 0.0)),
        low_x.std(0, ddof=0),
        high_x.std(0, ddof=0),
    )
    columns, normalizers = [], []
    for values in raw:
        column, evidence = robust_z(values)
        columns.append(column)
        normalizers.append(evidence)
    full = np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)
    masked = np.ascontiguousarray(full * np.asarray(plan.PROFILE_MASK, dtype=np.float32), dtype=np.float32)
    require(full.shape == masked.shape == (x.shape[1], plan.PROFILE_DIM), "profile geometry drift")
    return masked, {
        "raw_profile": full,
        "low_speed_quantile": low_q,
        "high_speed_quantile": high_q,
        "low_state_bins": int(low.sum()),
        "high_state_bins": int(high.sum()),
        "total_bins": int(x.shape[0]),
        "column_normalizers": normalizers,
        "profile_mask": list(plan.PROFILE_MASK),
    }


def build_film():
    import torch

    module = torch.nn.Sequential(
        torch.nn.Linear(plan.CONTEXT_DIM, plan.FILM_RANK),
        torch.nn.ReLU(),
        torch.nn.Linear(plan.FILM_RANK, 2 * plan.HIDDEN_DIM),
    )
    with torch.no_grad():
        module[2].weight.zero_()
        module[2].bias.zero_()
    require(sum(p.numel() for p in module.parameters()) == plan.FILM_PARAMETERS, "FiLM parameter count drift")
    return module


def film_identity(base_encoder, mean_feature, carrier, profile, film):
    import torch

    require(mean_feature.ndim == 3 and mean_feature.shape[-1] == plan.HIDDEN_DIM, "mean feature geometry drift")
    require(carrier.shape == (*mean_feature.shape[:2], plan.T4_DIM), "carrier geometry drift")
    require(profile.shape == (*mean_feature.shape[:2], plan.PROFILE_DIM), "profile geometry drift")
    modulation = film(torch.cat((carrier, profile.to(carrier)), dim=-1))
    gamma, beta = modulation.chunk(2, dim=-1)
    modulated = (1.0 + gamma) * mean_feature + beta
    identity = base_encoder.post_pool(torch.cat((modulated, carrier), dim=-1))
    require(identity.shape == (*mean_feature.shape[:2], plan.WINDOW), "identity geometry drift")
    require(bool(torch.isfinite(identity).all()), "identity nonfinite")
    return identity


def paired_summary(candidate: Mapping[str, float], native: Mapping[str, float]) -> dict[str, object]:
    require(set(candidate) == set(native) and bool(candidate), "paired session roster mismatch")
    delta = {key: float(candidate[key] - native[key]) for key in sorted(candidate)}
    values = np.asarray(list(delta.values()), dtype=np.float64)
    return {
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "worst_delta": float(values.min()),
        "positive_sessions": int(np.count_nonzero(values > 0.0)),
        "session_count": int(values.size),
        "per_session_delta": delta,
    }


def decide(primary: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    cp10 = primary["CP10@M10"]
    cp30 = primary["CP30@M30"]
    empty = primary["EMPTY@ZERO"]
    shuffle = primary["SHUFFLE10@M10"]

    def noninferior(row):
        return float(row["mean_delta"]) >= plan.NONINFERIOR_MEAN and float(row["worst_delta"]) >= plan.NONINFERIOR_WORST

    def positive(row, *controls):
        return (
            float(row["mean_delta"]) > 0.0
            and int(row["positive_sessions"]) >= plan.POSITIVE_SESSIONS
            and all(float(row["mean_delta"]) > float(control["mean_delta"]) for control in controls)
        )

    return {
        "cp10_noninferior": noninferior(cp10),
        "cp30_noninferior": noninferior(cp30),
        "seed_expansion_authorized": bool(noninferior(cp10) or noninferior(cp30)),
        "cp10_profile_positive": positive(cp10, empty, shuffle),
        "cp30_profile_positive": positive(cp30, empty),
        "cp10_solid": bool(positive(cp10, empty, shuffle) and float(cp10["mean_delta"]) >= plan.SOLID_MEAN and float(empty["mean_delta"]) <= 0 and float(shuffle["mean_delta"]) <= 0),
        "cp30_solid": bool(positive(cp30, empty) and float(cp30["mean_delta"]) >= plan.SOLID_MEAN and float(empty["mean_delta"]) <= 0),
        "formal_test_access": False,
        "evalai_push": False,
    }

