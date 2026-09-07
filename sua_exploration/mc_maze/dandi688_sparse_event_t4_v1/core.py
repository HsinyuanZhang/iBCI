"""Pure sparse-event profile, masking, shuffle, and training-plan helpers."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from . import plan


class SparseEventContractError(RuntimeError):
    """A frozen sparse-event protocol invariant was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SparseEventContractError(message)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)}, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_columns(raw: np.ndarray, *, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    mean, scale = np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    require(raw.ndim == 2 and raw.shape[1] == plan.PROFILE_DIM, "descriptor width drift")
    require(mean.shape == scale.shape == (plan.PROFILE_DIM,), "normalizer width drift")
    safe_scale = np.maximum(scale, plan.NORMALIZER_SCALE_FLOOR)
    result = np.ascontiguousarray((raw - mean) / safe_scale, dtype=np.float32)
    require(np.isfinite(result).all(), "normalized descriptor nonfinite")
    return result


def fit_source_normalizer(source_rows: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    rows = np.concatenate([np.asarray(value, dtype=np.float64) for value in source_rows], axis=0)
    require(rows.ndim == 2 and rows.shape[1] == plan.PROFILE_DIM and rows.shape[0] > 0, "source normalizer geometry")
    require(np.isfinite(rows).all(), "source normalizer nonfinite")
    return {"mean": rows.mean(axis=0), "scale": np.maximum(rows.std(axis=0), plan.NORMALIZER_SCALE_FLOOR)}


def apply_reliability_mask(profile: np.ndarray, mask: Sequence[bool | int | float]) -> np.ndarray:
    profile = np.asarray(profile, dtype=np.float32)
    retained = np.asarray(mask, dtype=np.float32)
    require(profile.ndim == 2 and profile.shape[1] == plan.PROFILE_DIM, "profile geometry drift")
    require(retained.shape == (plan.PROFILE_DIM,), "mask width drift")
    require(bool(np.isin(retained, (0.0, 1.0)).all()), "mask must be binary")
    result = np.ascontiguousarray(profile * retained[None, :], dtype=np.float32)
    # Multiplication keeps a negative signed zero in some NumPy combinations;
    # the protocol requires failed columns to be exact positive zero.
    result[:, retained == 0.0] = 0.0
    return result


def phase_r_profile(profile: np.ndarray, mask: Sequence[bool | int | float]) -> np.ndarray:
    phase = np.asarray(profile, dtype=np.float32).copy()
    phase[:, 3] = 0.0
    return apply_reliability_mask(phase, mask)


def _shuffle_seed(*, session_id: str, view: str, training_seed: int) -> int:
    payload = f"{plan.ROW_SHUFFLE_DOMAIN}:{session_id}:{view}:seed={training_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def deterministic_nonidentity_permutation(*, rows: int, session_id: str, view: str, training_seed: int) -> np.ndarray:
    require(rows >= 2, "row shuffle requires at least two rows")
    permutation = np.random.Generator(np.random.PCG64(_shuffle_seed(session_id=session_id, view=view, training_seed=training_seed))).permutation(rows)
    if np.array_equal(permutation, np.arange(rows)):
        permutation = np.roll(permutation, 1)
    require(not np.array_equal(permutation, np.arange(rows)), "row shuffle identity")
    return permutation.astype(np.int64, copy=False)


def row_shuffle_profile(profile: np.ndarray, *, session_id: str, view: str, training_seed: int) -> tuple[np.ndarray, np.ndarray]:
    profile = np.asarray(profile, dtype=np.float32)
    permutation = deterministic_nonidentity_permutation(rows=profile.shape[0], session_id=session_id, view=view, training_seed=training_seed)
    return np.ascontiguousarray(profile[permutation]), permutation


def build_film():
    """The reviewed rank-8 head, zero anchored at its final linear layer."""
    import torch

    module = torch.nn.Sequential(torch.nn.Linear(plan.CONTEXT_DIM, plan.FILM_RANK), torch.nn.ReLU(), torch.nn.Linear(plan.FILM_RANK, 2 * plan.HIDDEN_DIM))
    with torch.no_grad():
        module[2].weight.zero_()
        module[2].bias.zero_()
    require(sum(p.numel() for p in module.parameters()) == plan.FILM_PARAMETERS, "FiLM parameter count drift")
    return module


def film_identity(base_encoder, mean_feature, carrier, profile, film, *, direct_zero: bool = True):
    """Apply FiLM, with an auditable direct-native zero branch when requested.

    The pre-first-step sentinel and explicit zero controls use ``direct_zero``.
    A trainable head must pass ``direct_zero=False`` so its zero-initialized
    final layer remains on a differentiable path during the first update.
    """
    import torch

    require(carrier.shape == (*mean_feature.shape[:2], plan.T4_DIM), "carrier geometry")
    require(profile.shape == (*mean_feature.shape[:2], plan.PROFILE_DIM), "profile geometry")
    modulation = film(torch.cat((carrier, profile.to(carrier)), dim=-1))
    gamma, beta = modulation.chunk(2, dim=-1)
    exact_zero = bool(torch.equal(gamma, torch.zeros_like(gamma)) and torch.equal(beta, torch.zeros_like(beta)))
    if exact_zero and direct_zero:
        identity = base_encoder.post_pool(torch.cat((mean_feature, carrier), dim=-1))
    else:
        identity = base_encoder.post_pool(torch.cat((((1.0 + gamma) * mean_feature + beta), carrier), dim=-1))
    require(bool(torch.isfinite(identity).all()), "FiLM identity nonfinite")
    return identity, {"direct_native_branch": bool(exact_zero and direct_zero), "modulation_exact_zero": exact_zero}


def stage1_film_opening_decision(semantic: Mapping[str, float], attachment: Mapping[str, float]) -> bool:
    require(set(semantic) == set(attachment) and len(semantic) == 6, "Stage-1 validation roster drift")
    semantic_values = np.asarray(list(semantic.values()), dtype=np.float64)
    attachment_values = np.asarray(list(attachment.values()), dtype=np.float64)
    return bool(semantic_values.mean() > 0.0 and attachment_values.mean() > 0.0 and np.count_nonzero(semantic_values > 0.0) >= 4)
