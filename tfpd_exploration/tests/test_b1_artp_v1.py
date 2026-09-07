from __future__ import annotations

import copy

import numpy as np
import pytest

from tfpd_exploration.src.b1_artp_v1.core import (
    ARTPConfig,
    ARTPContractError,
    activity_signature,
    predict_from_signature,
    reliability_profile,
)


def _tx(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.poisson(0.002, size=(27000, 85)).astype(np.float64)


def _payload() -> dict:
    rng = np.random.default_rng(8)
    signatures = np.stack([activity_signature(_tx(i + 1)) for i in range(3)])
    spectrograms = 1.0 + rng.random((3, 158, 880))
    median, distances, normalized = reliability_profile(spectrograms)
    from tfpd_exploration.src.b1_sfcj_v1.util import sha256_array

    return {
        "schema": "b1-artp-payload-v1",
        "m": 3,
        "signature_law": "population-time-mean-std-l2-v1",
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


def test_activity_signature_shape_norm_and_geometry() -> None:
    tx = _tx(1)
    a = activity_signature(tx)
    b = activity_signature(tx.T[None])
    assert a.shape == (1800,)
    assert np.array_equal(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0, rtol=0, atol=1e-12)
    with pytest.raises(ARTPContractError):
        activity_signature(np.zeros((27000, 85)))


def test_reliability_penalizes_acoustic_outlier() -> None:
    base = np.ones((158, 880), dtype=np.float64)
    base[:, 90:790] += np.linspace(0.0, 1.0, 700)[None]
    specs = [base, base + 0.01, base[:, ::-1]]
    _, distance, normalized = reliability_profile(specs)
    assert distance[2] > distance[0]
    assert normalized[2] > normalized[0]


def test_prediction_is_convex_and_query_neural_changes_weights() -> None:
    payload = _payload()
    q = np.asarray(payload["reference_signatures"])[0]
    pred, evidence = predict_from_signature(q, payload)
    assert pred.shape == (158, 880)
    assert np.isclose(evidence["weights"].sum(), 1.0)
    assert np.all(evidence["weights"] >= 0.0)
    expected = np.tensordot(evidence["weights"], payload["reference_spectrograms"], axes=(0, 0))
    assert np.allclose(pred, expected, rtol=0, atol=1e-12)
    _, neural_free = predict_from_signature(q, payload, use_query_neural=False)
    assert not np.array_equal(evidence["weights"], neural_free["weights"])
    assert evidence["reads_query_target"] is False
    assert evidence["model_updates"] == 0


def test_payload_digest_drift_fails_closed() -> None:
    payload = _payload()
    payload = copy.deepcopy(payload)
    payload["reference_signatures"][0, 0] += 1e-12
    with pytest.raises(ARTPContractError, match="digest"):
        predict_from_signature(np.ones(1800), payload)


def test_config_validation() -> None:
    with pytest.raises(ARTPContractError):
        ARTPConfig(tau=0.0).validate()
    with pytest.raises(ARTPContractError):
        ARTPConfig(reliability_strength=-1.0).validate()
    with pytest.raises(ARTPContractError):
        ARTPConfig(mixing=1.01).validate()
