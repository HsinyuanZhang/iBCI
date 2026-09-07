"""ARTP-P: equal fusion of raw and M3-profiled activity similarities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tfpd_exploration.src.b1_artp_v1.core import (
    ARTPContractError,
    activity_signature,
    reliability_profile,
    stable_softmax,
)
from tfpd_exploration.src.b1_sfcj_v1.constants import (
    N_CHANNELS,
    N_FREQ,
    N_NEURAL_SAMPLES,
    N_SPEC_FRAMES,
)
from tfpd_exploration.src.b1_sfcj_v1.data import rate_view
from tfpd_exploration.src.b1_sfcj_v1.util import sha256_array

from . import plan


@dataclass(frozen=True)
class ARTPProfileConfig:
    tau: float = plan.TAU
    reliability_strength: float = plan.RELIABILITY_STRENGTH
    raw_profile_mix: float = plan.RAW_PROFILE_MIX

    def validate(self) -> None:
        if not np.isfinite(self.tau) or self.tau <= 0.0:
            raise ARTPContractError("tau must be finite and positive")
        if not np.isfinite(self.reliability_strength) or self.reliability_strength < 0.0:
            raise ARTPContractError("reliability strength must be finite and nonnegative")
        if not np.isfinite(self.raw_profile_mix) or not 0.0 <= self.raw_profile_mix <= 1.0:
            raise ARTPContractError("raw/profile mix must be in [0,1]")


def _tx(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (1, N_CHANNELS, N_NEURAL_SAMPLES):
        array = array[0].T
    if array.shape != (N_NEURAL_SAMPLES, N_CHANNELS) or not np.isfinite(array).all():
        raise ARTPContractError(f"invalid neural input {array.shape}")
    return array


def channel_profile_from_rates(reference_rates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rates = np.asarray(reference_rates, dtype=np.float64)
    if rates.shape != (plan.M, 900, N_CHANNELS) or not np.isfinite(rates).all():
        raise ARTPContractError(f"invalid reference rate stack {rates.shape}")
    flat = rates.reshape(-1, N_CHANNELS)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0, ddof=0)
    std = np.where(std < 1e-6, 1.0, std)
    return np.ascontiguousarray(mean), np.ascontiguousarray(std)


def profiled_signature_from_rate(rate: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    value = np.asarray(rate, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if value.shape != (900, N_CHANNELS) or mean.shape != (N_CHANNELS,) or std.shape != (N_CHANNELS,):
        raise ARTPContractError("invalid rate/profile geometry")
    if not np.isfinite(value).all() or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0.0):
        raise ARTPContractError("invalid channel profile")
    normalized = (value - mean) / std
    signature = np.concatenate([normalized.mean(axis=1), normalized.std(axis=1)], axis=0)
    signature -= signature.mean()
    norm = float(np.linalg.norm(signature))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ARTPContractError("profiled signature has zero/non-finite norm")
    return np.ascontiguousarray(signature / norm, dtype=np.float64)


def build_profiled_payload(calibration_trials) -> dict:
    trials = list(calibration_trials)
    if len(trials) != plan.M:
        raise ARTPContractError(f"ARTP-P requires exactly M3, got {len(trials)}")
    rates = np.stack([rate_view(_tx(trial.tx)) for trial in trials]).astype(np.float64)
    mean, std = channel_profile_from_rates(rates)
    raw_signatures = np.stack([activity_signature(trial.tx) for trial in trials])
    profiled_signatures = np.stack(
        [profiled_signature_from_rate(rate, mean, std) for rate in rates]
    )
    spectrograms = np.stack(
        [np.asarray(trial.spectrogram, dtype=np.float64) for trial in trials]
    )
    if spectrograms.shape != (plan.M, N_FREQ, N_SPEC_FRAMES):
        raise ARTPContractError(f"invalid M3 spectrogram stack {spectrograms.shape}")
    median, distances, normalized = reliability_profile(spectrograms)
    arrays = {
        "reference_rates": rates,
        "channel_profile_mean": mean,
        "channel_profile_std": std,
        "reference_raw_signatures": raw_signatures,
        "reference_profiled_signatures": profiled_signatures,
        "reference_spectrograms": spectrograms,
        "median_template": median,
        "reliability_distance": distances,
        "reliability_normalized": normalized,
    }
    return {
        "schema": "b1-artp-profile-payload-v2",
        "m": plan.M,
        **arrays,
        "array_sha256": {name: sha256_array(value) for name, value in arrays.items()},
        "profile_frozen_after_m3": True,
        "query_labels_used": False,
        "model_updates": 0,
    }


def _validated(payload: dict) -> dict[str, np.ndarray]:
    if payload.get("schema") != "b1-artp-profile-payload-v2" or payload.get("m") != plan.M:
        raise ARTPContractError("invalid ARTP-P payload")
    shapes = {
        "reference_rates": (3, 900, 85),
        "channel_profile_mean": (85,),
        "channel_profile_std": (85,),
        "reference_raw_signatures": (3, 1800),
        "reference_profiled_signatures": (3, 1800),
        "reference_spectrograms": (3, 158, 880),
        "median_template": (158, 880),
        "reliability_distance": (3,),
        "reliability_normalized": (3,),
    }
    out = {}
    digests = payload.get("array_sha256", {})
    if set(digests) != set(shapes):
        raise ARTPContractError("payload digest key set mismatch")
    for name, shape in shapes.items():
        value = np.asarray(payload[name], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ARTPContractError(f"invalid payload array {name}")
        if sha256_array(value) != digests[name]:
            raise ARTPContractError(f"payload digest mismatch: {name}")
        out[name] = value
    if np.any(out["channel_profile_std"] <= 0.0):
        raise ARTPContractError("nonpositive channel profile std")
    median, distance, normalized = reliability_profile(out["reference_spectrograms"])
    if not np.array_equal(median, out["median_template"]):
        raise ARTPContractError("median mismatch")
    if not np.array_equal(distance, out["reliability_distance"]):
        raise ARTPContractError("reliability distance mismatch")
    if not np.array_equal(normalized, out["reliability_normalized"]):
        raise ARTPContractError("reliability normalization mismatch")
    return out


def predict_from_profiled_rate(
    query_rate: np.ndarray,
    payload: dict,
    *,
    config: ARTPProfileConfig = ARTPProfileConfig(),
    channel_mean: np.ndarray | None = None,
    channel_std: np.ndarray | None = None,
    use_query_neural: bool = True,
) -> tuple[np.ndarray, dict]:
    config.validate()
    values = _validated(payload)
    query_rate = np.asarray(query_rate, dtype=np.float64)
    if query_rate.shape != (900, 85) or not np.isfinite(query_rate).all():
        raise ARTPContractError("invalid query rate")
    mean = values["channel_profile_mean"] if channel_mean is None else np.asarray(channel_mean)
    std = values["channel_profile_std"] if channel_std is None else np.asarray(channel_std)
    # External profiles are used only by the predeclared causal-growing control;
    # references and query are always transformed under the same profile.
    ref_profiled = np.stack(
        [profiled_signature_from_rate(rate, mean, std) for rate in values["reference_rates"]]
    )
    raw_vector = np.concatenate([query_rate.mean(axis=1), query_rate.std(axis=1)])
    raw_vector -= raw_vector.mean()
    raw_vector /= np.linalg.norm(raw_vector)
    query_profiled = profiled_signature_from_rate(query_rate, mean, std)
    raw_similarity = values["reference_raw_signatures"] @ raw_vector
    profile_similarity = ref_profiled @ query_profiled
    similarity = (
        (1.0 - config.raw_profile_mix) * raw_similarity
        + config.raw_profile_mix * profile_similarity
    )
    neural_logits = similarity / config.tau if use_query_neural else np.zeros(3)
    reliability_logits = -config.reliability_strength * values["reliability_normalized"]
    weights = stable_softmax(neural_logits + reliability_logits)
    prediction = np.tensordot(weights, values["reference_spectrograms"], axes=(0, 0))
    return prediction, {
        "raw_similarity": raw_similarity,
        "profile_similarity": profile_similarity,
        "fused_similarity": similarity,
        "weights": weights,
        "prediction_sha256": sha256_array(prediction),
        "reads_query_neural": bool(use_query_neural),
        "reads_query_target": False,
        "model_updates": 0,
    }


def predict_from_profiled_payload(
    neural_observations: np.ndarray,
    payload: dict,
    *,
    config: ARTPProfileConfig = ARTPProfileConfig(),
    use_query_neural: bool = True,
) -> tuple[np.ndarray, dict]:
    return predict_from_profiled_rate(
        rate_view(_tx(neural_observations)),
        payload,
        config=config,
        use_query_neural=use_query_neural,
    )
