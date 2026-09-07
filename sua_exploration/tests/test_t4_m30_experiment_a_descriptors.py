from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mc_maze.unit_side_features import (
    _ls4_direction_indices,
    deterministic_nonidentity_row_permutation,
    mask_standardized_t4,
    permute_side_feature_rows_nonidentity,
)


def test_masks_are_width_four_and_operate_in_standardized_space() -> None:
    standardized = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]], dtype=np.float32)
    assert np.array_equal(mask_standardized_t4(standardized, "z4"), np.zeros_like(standardized))
    assert np.array_equal(mask_standardized_t4(standardized, "ac4"), [[1, 2, 0, 0], [-1, -2, 0, 0]])
    assert np.array_equal(mask_standardized_t4(standardized, "mb4"), [[0, 0, 3, 4], [0, 0, -3, -4]])
    assert np.array_equal(mask_standardized_t4(standardized, "b4"), [[0, 0, 0, 4], [0, 0, 0, -4]])


def test_ls4_label_shuffle_is_session_seed_deterministic_and_reorders_only_labels() -> None:
    directions = np.arange(30, dtype=np.int64)
    a = _ls4_direction_indices(directions, session_name="sub-C_ses-CO-20151103", seed=42)
    b = _ls4_direction_indices(directions, session_name="sub-C_ses-CO-20151103", seed=42)
    c = _ls4_direction_indices(directions, session_name="sub-C_ses-CO-20151104", seed=42)
    assert np.array_equal(a, b)
    assert sorted(a.tolist()) == directions.tolist()
    assert not np.array_equal(a, c)


def test_ls4_seed_is_part_of_uncached_metadata_cache_payload() -> None:
    source = (Path(__file__).resolve().parents[1] / "mc_maze/unit_side_features.py").read_text()
    start = source.index("def _compute_tuning_features_uncached")
    end = source.index("def compute_unit_side_features_uncached", start)
    body = source[start:end]
    assert 'cache_payload["label_permutation_seed"] = label_permutation_seed' in body


def test_ls4_training_metadata_describes_raw_refit_before_normalization() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts/train_variant_dandi688.py").read_text()
    assert '"raw_refit_from_label_permuted_directions_before_normalization": True' in source
    assert '"mask_applied_after_standardization": True' in source


def test_component_row_shuffle_is_deterministic_nonidentity_and_moves_complete_rows() -> None:
    standardized = np.arange(24, dtype=np.float32).reshape(6, 4)
    ac4 = mask_standardized_t4(standardized, "ac4")
    first = permute_side_feature_rows_nonidentity(
        ac4, permutation_seed=42, session_name="session-a"
    )
    second = permute_side_feature_rows_nonidentity(
        ac4, permutation_seed=42, session_name="session-a"
    )
    assert np.array_equal(first, second)
    assert not np.array_equal(first, ac4)
    assert sorted(map(tuple, first.tolist())) == sorted(map(tuple, ac4.tolist()))
    assert np.array_equal(first[:, 2:], np.zeros((6, 2), dtype=np.float32))


def test_component_row_shuffle_rejects_singleton_rows() -> None:
    with np.testing.assert_raises_regex(ValueError, "at least two rows"):
        permute_side_feature_rows_nonidentity(
            np.zeros((1, 4), dtype=np.float32),
            permutation_seed=42,
            session_name="session-a",
        )


def test_component_row_shuffle_salts_equal_size_sessions() -> None:
    first = deterministic_nonidentity_row_permutation(
        61, permutation_seed=42, session_name="session-a"
    )
    second = deterministic_nonidentity_row_permutation(
        61, permutation_seed=42, session_name="session-b"
    )
    assert not np.array_equal(first, np.arange(61))
    assert not np.array_equal(second, np.arange(61))
    assert not np.array_equal(first, second)
