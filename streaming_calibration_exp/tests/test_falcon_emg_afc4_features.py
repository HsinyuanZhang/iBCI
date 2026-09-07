from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.falcon_emg_afc4_features import (  # noqa: E402
    AFC4_DIM, AFC4_SUPPORT_TRIALS, M1EMGSupport, SourceFrozenEMGAFC4Plan,
    _fit_basis, _fit_encoding,
    deterministic_afc4_label_derangement, deterministic_afc4_row_permutation,
    load_m1_emg_support,
)


def test_variable_n_complete_row_permutation_is_deterministic_nonidentity() -> None:
    first = deterministic_afc4_row_permutation(37, session_name="ses-a", seed=42)
    second = deterministic_afc4_row_permutation(37, session_name="ses-a", seed=42)
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(37))
    assert np.all(first != np.arange(37))


def test_m10_label_schedule_is_deterministic_complete_derangement() -> None:
    first = deterministic_afc4_label_derangement(session_name="ses-a", seed=42)
    second = deterministic_afc4_label_derangement(session_name="ses-a", seed=42)
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(AFC4_SUPPORT_TRIALS))
    assert np.all(first != np.arange(AFC4_SUPPORT_TRIALS))


def test_source_pca_order_and_sign_are_frozen() -> None:
    rng = np.random.default_rng(7)
    basis = _fit_basis([rng.normal(size=(20, 16)), rng.normal(size=(30, 16))])
    assert basis.components.shape == (3, 16)
    assert np.all(np.diff(basis.singular_values) <= 0.0)
    assert np.all(basis.components[np.arange(3), basis.sign_anchor_indices] >= 0.0)


def test_m10_q3_encoding_rejects_rank_deficiency_and_support_mismatch() -> None:
    rng = np.random.default_rng(3)
    scores = rng.normal(size=(AFC4_SUPPORT_TRIALS, 3))
    rates = rng.normal(size=(AFC4_SUPPORT_TRIALS, 37))
    raw, design = _fit_encoding(scores, rates)
    assert raw.shape == (37, AFC4_DIM)
    assert design.shape == (AFC4_SUPPORT_TRIALS, 4)
    with pytest.raises(ValueError, match="exactly M10"):
        _fit_encoding(scores[:-1], rates[:-1])
    with pytest.raises(ValueError, match="rank 4"):
        _fit_encoding(np.zeros((AFC4_SUPPORT_TRIALS, 3)), rates)


def test_ls4_deranges_only_score_pairing_and_uses_correct_carrier_normalizer() -> None:
    rng = np.random.default_rng(17)
    scores = rng.normal(size=(AFC4_SUPPORT_TRIALS, 3))
    rates = rng.normal(size=(AFC4_SUPPORT_TRIALS, 5))

    class IdentityBasis:
        @staticmethod
        def project(values):
            return np.asarray(values, dtype=np.float64)

    record = M1EMGSupport(
        session_name="ses-a",
        path=Path("synthetic.nwb"),
        full_trial_emg=scores,
        support_emg=scores,
        support_rates=rates,
        n_trials=AFC4_SUPPORT_TRIALS + 1,
        query_checksum="a" * 64,
        source_for_pca=False,
        emg_trials_materialized=AFC4_SUPPORT_TRIALS,
    )
    correct, _ = _fit_encoding(scores, rates)
    plan = object.__new__(SourceFrozenEMGAFC4Plan)
    plan.shuffle_seed = 42
    plan.support = {"ses-a": record}
    plan.basis = IdentityBasis()
    plan.raw_full = {"ses-a": correct}
    plan.raw_ls4 = {}
    plan.target_fit_calls = {"ses-a": 0}
    plan.mean = np.linspace(-0.2, 0.2, AFC4_DIM, dtype=np.float32)
    plan.std = np.linspace(0.7, 1.3, AFC4_DIM, dtype=np.float32)

    observed = plan.normalized("ses-a", arm="ls4")
    order = deterministic_afc4_label_derangement(session_name="ses-a", seed=42)
    expected_raw, _ = _fit_encoding(scores[order], rates)
    expected = ((expected_raw - plan.mean) / plan.std).astype(np.float32)

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
    assert plan.target_fit_calls == {"ses-a": 1}
    assert not np.array_equal(expected_raw, correct)


def test_b4_and_rs4_are_exact_post_normalization_controls() -> None:
    rng = np.random.default_rng(29)
    channels = 11
    raw = rng.normal(size=(channels, AFC4_DIM)).astype(np.float32)
    record = M1EMGSupport(
        session_name="ses-a",
        path=Path("synthetic.nwb"),
        full_trial_emg=rng.normal(size=(AFC4_SUPPORT_TRIALS, 3)),
        support_emg=rng.normal(size=(AFC4_SUPPORT_TRIALS, 3)),
        support_rates=rng.normal(size=(AFC4_SUPPORT_TRIALS, channels)),
        n_trials=AFC4_SUPPORT_TRIALS + 1,
        query_checksum="b" * 64,
        source_for_pca=False,
        emg_trials_materialized=AFC4_SUPPORT_TRIALS,
    )
    plan = object.__new__(SourceFrozenEMGAFC4Plan)
    plan.shuffle_seed = 42
    plan.support = {"ses-a": record}
    plan.raw_full = {"ses-a": raw}
    plan.raw_ls4 = {}
    plan.target_fit_calls = {"ses-a": 0}
    plan.mean = rng.normal(size=AFC4_DIM).astype(np.float32)
    plan.std = rng.uniform(0.5, 1.5, size=AFC4_DIM).astype(np.float32)

    full = plan.normalized("ses-a", arm="full")
    b4 = plan.normalized("ses-a", arm="b4")
    rs4 = plan.normalized("ses-a", arm="rs4")
    order = deterministic_afc4_row_permutation(channels, session_name="ses-a", seed=42)

    np.testing.assert_array_equal(b4[:, :3], np.zeros_like(b4[:, :3]))
    np.testing.assert_array_equal(b4[:, 3], full[:, 3])
    np.testing.assert_array_equal(rs4, full[order])
    assert np.all(order != np.arange(channels))


def test_arm_receipts_distinguish_all_three_mechanism_controls() -> None:
    plan = object.__new__(SourceFrozenEMGAFC4Plan)
    plan.shuffle_seed = 42
    plan.source_session_names = ("ses-a",)
    plan.support = {
        "ses-a": M1EMGSupport(
            "ses-a", Path("synthetic.nwb"), np.ones((10, 3)), np.ones((10, 3)),
            np.ones((10, 5)), 11, "c" * 64, False, 10,
        )
    }
    plan.basis = type("Basis", (), {
        "mean": np.zeros(3), "scale": np.ones(3), "components": np.eye(3),
        "singular_values": np.ones(3), "explained_energy": np.ones(3) / 3,
        "sign_anchor_indices": np.arange(3),
    })()
    plan.mean = np.zeros(4, dtype=np.float32)
    plan.std = np.ones(4, dtype=np.float32)

    b4, rs4, ls4 = (plan.receipt(arm=arm) for arm in ("b4", "rs4", "ls4"))
    assert b4["b4_mask"] == "post_normalization_coordinates_0_to_2_zero"
    assert rs4["rs4"] == "complete_normalized_row_permutation"
    assert "M10 EMG-score rows" in str(ls4["ls4"])
    assert all(np.asarray(ls4["label_derangement_schedule"]["ses-a"]) != np.arange(10))


def test_target_loader_has_an_explicit_m10_only_path_without_generic_neural_loader() -> None:
    source = inspect.getsource(load_m1_emg_support)
    assert "source_for_pca" in source
    assert "emg_trial_count = len(trials) if source_for_pca else AFC4_SUPPORT_TRIALS" in source
    assert "get_unit_spike_times" in source
    assert "load_nwb" not in source
