"""Tests for held-in validation protocol helpers."""
from __future__ import annotations

import pytest

from src.data.validation_protocol import loso_split, rotation_5_2_split, sorted_session_names
from src.models.falcon_module import DATASET_NAMES


def test_loso_split_covers_all_sessions_once():
    sessions = DATASET_NAMES["m2"]["heldin"]
    for fold in range(len(sessions)):
        train, heldout = loso_split(sessions, fold)
        assert len(train) == 6
        assert heldout in sessions
        assert heldout not in train
        assert sorted_session_names(train + [heldout]) == sorted_session_names(sessions)


def test_rotation_5_2_split_partitions_seven_sessions():
    sessions = DATASET_NAMES["m2"]["heldin"]
    train, val = rotation_5_2_split(sessions, rotation_id=2)
    assert len(train) == 5
    assert len(val) == 2
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(sessions)


def test_loso_split_rejects_invalid_fold():
    sessions = DATASET_NAMES["m2"]["heldin"]
    with pytest.raises(ValueError):
        loso_split(sessions, fold=7)
