"""Focused tests for CTXV2 Stage A CPU infrastructure."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SPINT = REPO / "SPINT-main"
if str(SPINT) not in sys.path:
    sys.path.insert(0, str(SPINT))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sua_exploration.mc_maze import h1_sparse_event_endpoint as event
from src.data.h1_context_event_target_dated import (
    FIXED_FOLD0_ARRAYS,
    FIXED_FOLD0_MAP_SHA,
    H1ContextStrictTargetDatasetDated,
    lodo_source_sessions,
    lodo_target_sessions,
    pure_cpu_fit_and_bind_map,
    compare_fold0_target_dataset_equivalence,
    build_context_source_assets_dated,
    build_context_target_dataset_dated,
    verify_sealed_screen_receipt,
)

DATA = REPO / "SPINT-main/data/000954"
BUILDER = REPO / "SPINT-main/scripts/build_h1_context_event_source_snapshot_dated.py"


@pytest.fixture(scope="module")
def data_dir() -> Path:
    if not DATA.is_dir():
        pytest.skip("public H1 held-in data unavailable")
    return DATA


def test_verify_sealed_screen_receipt() -> None:
    verify_sealed_screen_receipt()


def test_pure_cpu_fold0_map_matches_sealed_constants(data_dir: Path) -> None:
    mapping = pure_cpu_fit_and_bind_map(data_dir, "19250101")
    assert mapping.map_sha256 == FIXED_FOLD0_MAP_SHA
    assert mapping.manifest()["array_sha256"] == FIXED_FOLD0_ARRAYS


def test_pure_cpu_rejects_invalid_outer_date(data_dir: Path) -> None:
    with pytest.raises(Exception):
        pure_cpu_fit_and_bind_map(data_dir, "19990101")


def test_lodo_19250108_source_pool(data_dir: Path) -> None:
    source = lodo_source_sessions("19250108")
    target = lodo_target_sessions("19250108")
    assert len(source) == 10
    assert len(target) == 3
    assert not set(source).intersection(target)
    assert all(event.session_date(name) == "19250108" for name in target)
    assert all(event.session_date(name) != "19250108" for name in source)


def test_lodo_19250101_source_pool(data_dir: Path) -> None:
    source = lodo_source_sessions("19250101")
    assert len(source) == 11
    assert all(event.session_date(name) != "19250101" for name in source)


@pytest.mark.torch
def test_fold0_target_dataset_equivalence(data_dir: Path) -> None:
    pytest.importorskip("lightning.pytorch")
    mapping = pure_cpu_fit_and_bind_map(data_dir, "19250101")
    _, _, latent_map, cache, normalizer = build_context_source_assets_dated(
        data_dir, "19250101", frozen_map=mapping
    )
    source_module = SimpleNamespace(_setup_done=True, latent_map=latent_map, normalizer=normalizer)
    result = compare_fold0_target_dataset_equivalence(data_dir=data_dir, source_module=source_module)
    assert result["equivalent"]


@pytest.mark.torch
def test_dated_target_19250108_has_three_recordings(data_dir: Path) -> None:
    pytest.importorskip("lightning.pytorch")
    mapping = pure_cpu_fit_and_bind_map(data_dir, "19250108")
    _, _, latent_map, cache, normalizer = build_context_source_assets_dated(
        data_dir, "19250108", frozen_map=mapping
    )
    source_module = SimpleNamespace(_setup_done=True, latent_map=latent_map, normalizer=normalizer)
    target = build_context_target_dataset_dated(
        data_dir=data_dir, source_module=source_module, outer_date="19250108"
    )
    assert len(target.target_sessions) == 3
    assert set(target.target_sessions) == set(lodo_target_sessions("19250108"))


def test_collapsed_control_assertion_fires() -> None:
    constant = np.ones((event.EXPECTED_NEURONS, 5), dtype=np.float64)
    mock_event = SimpleNamespace(
        base=SimpleNamespace(tag="reach", log_rates=np.ones(event.EXPECTED_NEURONS), duration_seconds=0.2),
        midpoint_state=np.zeros(7),
    )
    records = {
        "ses-test": SimpleNamespace(
            trial_values=[1.0, 2.0, 3.0, 4.0, 5.0],
            eval_mask=np.array([True, True, True, True, True]),
            trial_num=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            neural=np.zeros((10, event.EXPECTED_NEURONS), dtype=np.float32),
            velocity=np.zeros((10, 7), dtype=np.float32),
            session_name="ses-test",
        )
    }
    sessions = {
        "ses-test": SimpleNamespace(base=SimpleNamespace(trial_values=records["ses-test"].trial_values))
    }
    latent_map = SimpleNamespace()
    normalizer = SimpleNamespace(
        normalize=lambda x: np.asarray(x, np.float64),
        denominator=1.0,
    )
    support = (mock_event, mock_event, mock_event, mock_event)
    identity = np.zeros((4, 1024, event.EXPECTED_NEURONS), dtype=np.float32)

    with patch("src.data.h1_context_event_target_dated.design.select_range", return_value=support):
        with patch("src.data.h1_context_event_target_dated.interpolate_identity", return_value=identity):
            with patch("src.data.h1_context_event_target_dated.replace", lambda event, **kwargs: mock_event):
                with patch("src.data.h1_context_event_target_dated._fit", return_value=constant):
                    with patch(
                        "src.data.h1_context_event_target_dated.event_v1.within_trial_label_shuffle",
                        return_value=(tuple(range(4)), {}),
                    ):
                        with patch(
                            "src.data.h1_context_event_target_dated.design.within_trial_tag_shuffle",
                            return_value=(tuple(range(4)), {}),
                        ):
                            with patch(
                                "src.data.h1_context_event_target_dated.event_v2.row_shuffle",
                                return_value=(constant, {}),
                            ):
                                with pytest.raises(Exception, match="collapsed control"):
                                    H1ContextStrictTargetDatasetDated(
                                        records,
                                        sessions,
                                        latent_map,
                                        normalizer,
                                        ("ses-test",),
                                    )


def test_builder_fold0_self_validation_via_import(data_dir: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("dated_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    mapping = module._pure_cpu_map(data_dir, "19250101")
    assert mapping.map_sha256 == FIXED_FOLD0_MAP_SHA
    assert mapping.manifest()["array_sha256"] == FIXED_FOLD0_ARRAYS
