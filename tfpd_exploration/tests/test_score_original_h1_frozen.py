"""Synthetic contracts for the as-shipped Original H1 frozen scorer."""
from __future__ import annotations

import numpy as np
import pytest

from tfpd_exploration.src.family_runtime_v1 import score_original_h1_frozen as scorer


class FakeEngine:
    window_size = 700
    behavior_scaling_factor = 20.0
    smooth_calibration = False

    def __init__(self):
        self.tags = []
        self.observation_buffer = np.zeros((700, 1, 176), np.float32)

    def reset(self, tags):
        self.tags.append(tags)
        self.observation_buffer.fill(0)

    def predict(self, raw):
        self.observation_buffer[:-1] = self.observation_buffer[1:]
        self.observation_buffer[-1] = raw
        return np.full((1, 7), raw[0, 0], np.float32)


def test_replay_all_public_bins_resets_b1_tag_and_selects_only_masked_bins():
    engine = FakeEngine()
    neural = np.zeros((703, 176), np.float32)
    neural[:, 0] = np.arange(703, dtype=np.float32)
    velocity = np.repeat(neural[:, :1], 7, axis=1)
    mask = np.zeros(703, dtype=bool); mask[[0, 699, 700, 702]] = True
    result = scorer.replay_session(engine, neural, velocity, mask, "ses-fixed")
    assert engine.tags == [[scorer.tag_for("ses-fixed")]]
    assert result["public_calls"] == 703
    assert result["scored_count"] == 4
    np.testing.assert_array_equal(result["end"], [0, 699, 700, 702])
    np.testing.assert_array_equal(result["prediction"][:, 0], [0, 699, 700, 702])
    assert engine.observation_buffer[0, 0, 0] == 3
    assert engine.observation_buffer[-1, 0, 0] == 702


def test_replay_rejects_smoothing_bad_public_output_and_geometry():
    neural = np.zeros((2, 176), np.float32)
    velocity = np.zeros((2, 7), np.float32)
    mask = np.ones(2, dtype=bool)
    engine = FakeEngine(); engine.smooth_calibration = True
    with pytest.raises(RuntimeError, match="unsmoothed"):
        scorer.replay_session(engine, neural, velocity, mask, "ses-fixed")
    with pytest.raises(RuntimeError, match="geometry"):
        scorer.replay_session(FakeEngine(), neural.astype(np.float64), velocity, mask, "ses-fixed")
    for value in (np.zeros((1, 7), np.float64), np.zeros((1, 8), np.float32), np.zeros((2, 7), np.float32)):
        with pytest.raises(RuntimeError, match="owning native FP32"):
            scorer.assert_public(value)


def test_constants_bind_the_released_image_payload_cache_and_complete_counts():
    assert scorer.IMAGE.endswith("f719c4228c345f9a1d6aa7c1e10d63ad7b9aa1f551dc95272b0b7a612d61fac6")
    assert scorer.EXPECTED[scorer.PAYLOAD] == "20a1d41a1d82a8037579caa2e4454f56817e02f021dec2132798c7fc57849298"
    assert scorer.CACHE_SHA256 == "51ff9ebfcd10a032f9c173ec426bfb4c421b751502271577582239c51bcc91b4"
    assert scorer.AUTHORITY_SHA256 == "92739d9f20d8184e2a24ddc2e4a00545c0b90170796c33f9d189cf90e8c6e41f"
    assert (scorer.W, scorer.UNITS, scorer.OUTPUTS, scorer.SCALE, scorer.COUNT, scorer.PUBLIC_CALLS) == (700, 176, 7, 20.0, 20325, 20920)


def test_gate_precedes_cache_import_hashes_and_output_creation(monkeypatch, tmp_path):
    output = tmp_path / "new-reference"
    monkeypatch.delenv("ORIGINAL_H1_FROZEN_GO", raising=False)
    monkeypatch.setattr(scorer, "sha", lambda _: pytest.fail("gate hashed immutable input"))
    with pytest.raises(RuntimeError, match="explicit ORIGINAL_H1_FROZEN_GO"):
        scorer.run(output)
    assert not output.exists()


def test_tags_are_exact_and_invalid_sessions_are_refused():
    assert str(scorer.tag_for("ses-19250101T111740")) == "sub-HumanPitt-held-in-minival_ses-19250101T111740"
    for bad in ("", "19250101", "ses-"):
        with pytest.raises(RuntimeError, match="session tag"):
            scorer.tag_for(bad)


def test_direct_native_subset_progress_and_gap_history():
    neural = np.zeros((703, 176), np.float32)
    neural[:, 0] = np.arange(703, dtype=np.float32)
    velocity = np.repeat(neural[:, :1], 7, axis=1)
    mask = np.zeros(703, bool)
    mask[[0, 702]] = True
    indices, progress = [], []
    def direct(history):
        assert history.shape == (1, 700, 176)
        indices.append(int(history[0, -1, 0]))
        return np.full((1, 7), history[0, -1, 0], np.float32)
    result = scorer.replay_session(FakeEngine(), neural, velocity, mask, "ses-fixed", direct, progress.append)
    assert indices == [0, 4, 699, 700, 702]
    assert result["direct_native_count"] == 5 and result["max_native_abs_error"] == 0
    assert progress[-1]["public_calls"] == 703 and progress[-1]["scored_count"] == 2
    with pytest.raises(AssertionError):
        scorer.replay_session(FakeEngine(), neural, velocity, mask, "ses-fixed", lambda _: np.ones((1,7),np.float32))
