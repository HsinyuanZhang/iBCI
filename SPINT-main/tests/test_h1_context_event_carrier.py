from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from sua_exploration.mc_maze import h1_event_carrier_design_screen as design
from sua_exploration.mc_maze import h1_sparse_event_endpoint as event

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/000954"

def _context(tag: str, value: float) -> design.ContextEvent:
    base = event.MovementEvent(row_id=0, tag=tag, trial_value=1.0, trial_index=0, start_time=0.0, stop_time=0.2,
        duration_seconds=0.2, eval_bins=10, displacement=np.full(7, value), log_rates=np.full(176, value))
    return design.ContextEvent(base=base, start_state=np.zeros(7), midpoint_state=np.full(7, value / 2))

def test_context_raw_label_is_exact_delta_midpoint_native_tag_onehot() -> None:
    rows = design.raw_features((_context(event.MOVEMENT_TAGS[0], 1.0), _context(event.MOVEMENT_TAGS[1], 2.0)), "context")
    assert rows.shape == (2, 14 + len(event.MOVEMENT_TAGS))
    np.testing.assert_array_equal(rows[0, :7], np.ones(7))
    np.testing.assert_array_equal(rows[1, 7:14], np.ones(7))
    assert rows[0, 14 + 0] == 1.0 and rows[0, 14:].sum() == 1.0
    assert rows[1, 14 + 1] == 1.0 and rows[1, 14:].sum() == 1.0


def test_common_zero_decision_does_not_reject_expected_carrier_manifest_difference() -> None:
    # Import by path avoids making scripts a Python package.
    import importlib.util
    spec = importlib.util.spec_from_file_location("context_preflight", ROOT / "scripts/h1_context_event_carrier_preflight.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.common_zero_reusable({"same_neural_windows": True, "same_schedule": True, "literal_zero_boundary": True})
    assert not module.common_zero_reusable({"same_neural_windows": True, "same_schedule": False, "literal_zero_boundary": True})

@pytest.mark.skipif(not DATA.is_dir(), reason="public H1 held-in data unavailable")
def test_real_source_context_map_has_fixed_schedule_and_no_target_access() -> None:
    pytest.importorskip("lightning.pytorch")
    from src.data.h1_context_event_carrier import H1ContextEventDataModule
    source = H1ContextEventDataModule(task="h1", data_dir=str(DATA), cache_dir=str(ROOT / "pilot_artifacts/h1_context_event_carrier/test_cache"), source_snapshot_receipt=str(ROOT / "pilot_artifacts/h1_context_event_carrier/source_snapshot/H1_CONTEXT_SER_Q4_FOLD0_SOURCE_v3.json"))
    source.setup("fit")
    manifest = source.pilot_manifest()
    assert len(source.carrier_cache.entries) == 116
    assert source.latent_map.candidate.name == "ser_context_q4"
    assert source.latent_map.candidate.rank == 4
    assert manifest["carrier_dim"] == 5 and manifest["target_carrier_ridge_lambda"] == 3.0
    assert manifest["target_nwb_opened_during_training_setup"] is False
    assert manifest["deployment_carrier_dense_velocity_opened"] is False
    batch = next(iter(source.train_dataloader()))
    assert tuple(batch[2].shape) == (32, 4, 1024, 176)
    assert tuple(batch[4].shape) == (32, 176, 5)


def test_training_constructor_rejects_live_svd_without_snapshot() -> None:
    pytest.importorskip("lightning.pytorch")
    from src.data.h1_context_event_carrier import H1ContextEventDataModule
    with pytest.raises(ValueError, match="immutable context source snapshot"):
        H1ContextEventDataModule(task="h1", data_dir=str(DATA), cache_dir=str(ROOT / "pilot_artifacts/h1_context_event_carrier/test_cache"))
