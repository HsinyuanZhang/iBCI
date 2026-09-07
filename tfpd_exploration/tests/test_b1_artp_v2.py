from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pytest

from tfpd_exploration.src.b1_artp_v1.core import ARTPContractError
from tfpd_exploration.src.b1_artp_v2.core import (
    ARTPProfileConfig,
    build_profiled_payload,
    channel_profile_from_rates,
    predict_from_profiled_payload,
    profiled_signature_from_rate,
)
from tfpd_exploration.src.b1_artp_v2.decoder import B1ARTPProfileDecoder
from tfpd_exploration.src.b1_artp_v2.export import deserialize_payloads, serialize_payloads


@dataclass
class Trial:
    tx: np.ndarray
    spectrogram: np.ndarray


def _trials() -> list[Trial]:
    rng = np.random.default_rng(4)
    return [
        Trial(
            tx=rng.poisson(0.002, size=(27000, 85)).astype(np.float64),
            spectrogram=1.0 + rng.random((158, 880)),
        )
        for _ in range(3)
    ]


def test_profile_is_m3_only_and_shapes_are_exact() -> None:
    trials = _trials()
    payload = build_profiled_payload(trials)
    assert payload["reference_rates"].shape == (3, 900, 85)
    assert payload["channel_profile_mean"].shape == (85,)
    assert payload["channel_profile_std"].shape == (85,)
    assert payload["profile_frozen_after_m3"] is True
    assert payload["query_labels_used"] is False


def test_profiled_signature_and_prediction_are_finite() -> None:
    trials = _trials()
    payload = build_profiled_payload(trials)
    signature = profiled_signature_from_rate(
        payload["reference_rates"][0],
        payload["channel_profile_mean"],
        payload["channel_profile_std"],
    )
    assert signature.shape == (1800,)
    assert np.isclose(np.linalg.norm(signature), 1.0, atol=1e-12)
    prediction, evidence = predict_from_profiled_payload(trials[0].tx, payload)
    assert prediction.shape == (158, 880)
    assert np.isfinite(prediction).all()
    assert np.isclose(evidence["weights"].sum(), 1.0)
    assert evidence["reads_query_target"] is False


def test_raw_profile_endpoints_and_query_neural_control() -> None:
    trials = _trials()
    payload = build_profiled_payload(trials)
    _, raw = predict_from_profiled_payload(
        trials[0].tx, payload, config=ARTPProfileConfig(raw_profile_mix=0.0)
    )
    _, profiled = predict_from_profiled_payload(
        trials[0].tx, payload, config=ARTPProfileConfig(raw_profile_mix=1.0)
    )
    _, neural_free = predict_from_profiled_payload(
        trials[0].tx, payload, use_query_neural=False
    )
    assert not np.array_equal(raw["fused_similarity"], profiled["fused_similarity"])
    assert not np.array_equal(raw["weights"], neural_free["weights"])


def test_payload_drift_fails_closed() -> None:
    payload = build_profiled_payload(_trials())
    bad = copy.deepcopy(payload)
    bad["channel_profile_mean"][0] += 1e-10
    with pytest.raises(ARTPContractError, match="digest"):
        predict_from_profiled_payload(_trials()[0].tx, bad)


def test_profile_std_floor() -> None:
    rates = np.zeros((3, 900, 85), dtype=np.float64)
    mean, std = channel_profile_from_rates(rates)
    assert np.array_equal(mean, np.zeros(85))
    assert np.array_equal(std, np.ones(85))


def test_decoder_shape_repeatability_and_no_updates() -> None:
    payload = build_profiled_payload(_trials())
    decoder = B1ARTPProfileDecoder({"20210626": payload})
    decoder.reset(["2021.06.26"])
    neural = _trials()[0].tx.T[None]
    first = decoder.predict(neural)
    second = decoder.predict(neural)
    assert first.shape == (158, 880)
    assert np.array_equal(first, second)
    assert decoder.query_label_access_count == 0
    assert decoder.model_updates == 0


def test_payload_serialization_is_exact() -> None:
    payload = build_profiled_payload(_trials())
    envelope = {date: copy.deepcopy(payload) for date in ("20210626", "20210627", "20210628", "20210630", "20210701", "20210705")}
    restored = deserialize_payloads(serialize_payloads(envelope))
    assert set(restored) == set(envelope)
    for date in restored:
        assert restored[date]["array_sha256"] == envelope[date]["array_sha256"]
