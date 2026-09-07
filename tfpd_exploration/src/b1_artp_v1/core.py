"""Pure NumPy ARTP operator.

The method has two independent pieces of evidence:

* a query-neural population-time activity signature chooses among the three
  released calibration exemplars;
* a label-derived reliability profile suppresses calibration spectrograms
  that are far from the coordinatewise M3 median.

No query target, model update, channel permutation, or hidden boundary state is
used.  The output is a convex combination of the three raw spectrograms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from tfpd_exploration.src.b1_sfcj_v1.constants import (
    N_CHANNELS,
    N_FREQ,
    N_NEURAL_SAMPLES,
    N_SPEC_FRAMES,
)
from tfpd_exploration.src.b1_sfcj_v1.data import rate_view
from tfpd_exploration.src.b1_sfcj_v1.metric import official_metric_from_trials
from tfpd_exploration.src.b1_sfcj_v1.util import sha256_array

from . import plan


class ARTPContractError(ValueError):
    pass


@dataclass(frozen=True)
class ARTPConfig:
    tau: float = plan.TAU
    reliability_strength: float = plan.RELIABILITY_STRENGTH
    mixing: float = plan.MIXING

    def validate(self) -> None:
        if not np.isfinite(self.tau) or self.tau <= 0.0:
            raise ARTPContractError("tau must be finite and positive")
        if not np.isfinite(self.reliability_strength) or self.reliability_strength < 0.0:
            raise ARTPContractError("reliability_strength must be finite and nonnegative")
        if not np.isfinite(self.mixing) or not 0.0 <= self.mixing <= 1.0:
            raise ARTPContractError("mixing must be in [0,1]")


def _as_tx(tx: np.ndarray) -> np.ndarray:
    value = np.asarray(tx)
    if value.shape == (1, N_CHANNELS, N_NEURAL_SAMPLES):
        value = value[0].T
    if value.shape != (N_NEURAL_SAMPLES, N_CHANNELS):
        raise ARTPContractError(
            f"neural input must be [27000,85] or [1,85,27000], got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ARTPContractError("neural input is non-finite")
    return value


def activity_signature(tx: np.ndarray) -> np.ndarray:
    """Return the 1,800-D population-time mean/std signature.

    The 85 channel indices are not permuted or re-identified.  At each 1 ms
    position the signature records the population mean and population standard
    deviation after the sealed 5 ms rate view.  Scalar centering followed by
    L2 normalization makes cosine similarity explicit and deterministic.
    """
    rate = rate_view(_as_tx(tx)).astype(np.float64, copy=False)
    signature = np.concatenate([rate.mean(axis=1), rate.std(axis=1)], axis=0)
    signature = signature - signature.mean()
    norm = float(np.linalg.norm(signature))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ARTPContractError("activity signature has zero/non-finite norm")
    signature = np.ascontiguousarray(signature / norm, dtype=np.float64)
    if signature.shape != (1800,) or not np.isfinite(signature).all():
        raise ARTPContractError("invalid activity signature")
    return signature


def _spectrogram_stack(values: Iterable[np.ndarray]) -> np.ndarray:
    stack = np.stack([np.asarray(value, dtype=np.float64) for value in values], axis=0)
    if stack.shape != (plan.M, N_FREQ, N_SPEC_FRAMES):
        raise ARTPContractError(f"expected three [158,880] spectrograms, got {stack.shape}")
    if not np.isfinite(stack).all() or np.any(stack <= 0.0):
        raise ARTPContractError("spectrogram templates must be finite and strictly positive")
    return stack


def reliability_profile(spectrograms: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return median template, raw distances, and median-normalized distances."""
    stack = _spectrogram_stack(spectrograms)
    median = np.median(stack, axis=0)
    distances = np.asarray(
        [official_metric_from_trials([row], [median])["MSE Mean"] for row in stack],
        dtype=np.float64,
    )
    scale = float(np.median(distances))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ARTPContractError("M3 reliability scale is zero/non-finite")
    normalized = distances / scale
    if not np.isfinite(normalized).all():
        raise ARTPContractError("non-finite reliability profile")
    return median, distances, normalized


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (plan.M,) or not np.isfinite(logits).all():
        raise ARTPContractError("invalid ARTP logits")
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    weights /= weights.sum()
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ARTPContractError("invalid ARTP weights")
    return weights


def build_payload(calibration_trials) -> dict:
    trials = list(calibration_trials)
    if len(trials) != plan.M:
        raise ARTPContractError(f"ARTP requires exactly M3, got {len(trials)}")
    signatures = np.stack([activity_signature(trial.tx) for trial in trials], axis=0)
    spectrograms = _spectrogram_stack([trial.spectrogram for trial in trials])
    median, distances, normalized = reliability_profile(spectrograms)
    return {
        "schema": "b1-artp-payload-v1",
        "m": plan.M,
        "signature_law": plan.SIGNATURE,
        "reference_signatures": signatures,
        "reference_spectrograms": spectrograms,
        "median_template": median,
        "reliability_distance": distances,
        "reliability_normalized": normalized,
        "reference_signature_sha256": sha256_array(signatures),
        "reference_spectrogram_sha256": sha256_array(spectrograms),
        "median_template_sha256": sha256_array(median),
        "query_labels_used": False,
        "model_updates": 0,
    }


def _validate_payload(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if payload.get("schema") != "b1-artp-payload-v1" or int(payload.get("m", -1)) != plan.M:
        raise ARTPContractError("invalid payload schema/M")
    signatures = np.asarray(payload["reference_signatures"], dtype=np.float64)
    spectrograms = _spectrogram_stack(payload["reference_spectrograms"])
    median = np.asarray(payload["median_template"], dtype=np.float64)
    reliability = np.asarray(payload["reliability_normalized"], dtype=np.float64)
    if signatures.shape != (plan.M, 1800) or reliability.shape != (plan.M,):
        raise ARTPContractError("invalid payload arrays")
    if median.shape != (N_FREQ, N_SPEC_FRAMES):
        raise ARTPContractError("invalid median template")
    if sha256_array(signatures) != payload.get("reference_signature_sha256"):
        raise ARTPContractError("reference signature digest mismatch")
    if sha256_array(spectrograms) != payload.get("reference_spectrogram_sha256"):
        raise ARTPContractError("reference spectrogram digest mismatch")
    if sha256_array(median) != payload.get("median_template_sha256"):
        raise ARTPContractError("median template digest mismatch")
    recomputed_median, _, recomputed_reliability = reliability_profile(spectrograms)
    if not np.array_equal(median, recomputed_median):
        raise ARTPContractError("median template does not match references")
    if not np.array_equal(reliability, recomputed_reliability):
        raise ARTPContractError("reliability profile does not match references")
    return signatures, spectrograms, median, reliability


def predict_from_signature(
    query_signature: np.ndarray,
    payload: dict,
    *,
    config: ARTPConfig = ARTPConfig(),
    use_query_neural: bool = True,
) -> tuple[np.ndarray, dict]:
    config.validate()
    signatures, spectrograms, median, reliability = _validate_payload(payload)
    query = np.asarray(query_signature, dtype=np.float64)
    if query.shape != (1800,) or not np.isfinite(query).all():
        raise ARTPContractError("invalid query signature")
    similarities = signatures @ query
    neural_logits = similarities / config.tau if use_query_neural else np.zeros(plan.M, dtype=np.float64)
    logits = neural_logits - config.reliability_strength * reliability
    weights = stable_softmax(logits)
    retrieved = np.tensordot(weights, spectrograms, axes=(0, 0))
    prediction = median + config.mixing * (retrieved - median)
    if prediction.shape != (N_FREQ, N_SPEC_FRAMES) or not np.isfinite(prediction).all():
        raise ARTPContractError("invalid ARTP prediction")
    evidence = {
        "similarities": similarities,
        "neural_logits": neural_logits,
        "reliability_logits": -config.reliability_strength * reliability,
        "weights": weights,
        "prediction_sha256": sha256_array(prediction),
        "reads_query_neural": bool(use_query_neural),
        "reads_query_target": False,
        "model_updates": 0,
    }
    return prediction, evidence


def predict_from_payload(
    neural_observations: np.ndarray,
    payload: dict,
    *,
    config: ARTPConfig = ARTPConfig(),
    use_query_neural: bool = True,
) -> tuple[np.ndarray, dict]:
    return predict_from_signature(
        activity_signature(neural_observations),
        payload,
        config=config,
        use_query_neural=use_query_neural,
    )
