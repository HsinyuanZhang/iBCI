"""No-target contract tests for the fixed three-arm H1 seed-43 package."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.data.h1_carrierid_seed43 import H1M4Seed43BatchSampler


ROOT = Path(__file__).resolve().parents[1]


def _fake_dataset():
    source = (
        "ses-19250108T110520", "ses-19250108T111022", "ses-19250108T111455",
        "ses-19250113T120811", "ses-19250113T121303", "ses-19250115T110633",
        "ses-19250115T111328", "ses-19250119T113543", "ses-19250119T114045",
        "ses-19250120T115044", "ses-19250120T115537",
    )
    windows = [(session, index) for session in source for index in range(32)]
    return SimpleNamespace(
        window_indices=windows,
        window_indices_sha256="seed43-test-window-hash",
        cache=SimpleNamespace(
            starts_by_session={session: [0, 1, 2, 3, 4] for session in source},
            manifest={"cache_sha256": "seed43-test-cache-hash"},
        ),
    )


def test_seed43_sampler_is_deterministic_and_has_one_new_schedule_per_epoch(tmp_path):
    first = H1M4Seed43BatchSampler(_fake_dataset(), tmp_path / "first")
    second = H1M4Seed43BatchSampler(_fake_dataset(), tmp_path / "second")
    assert first.seed == second.seed == 43
    assert first.max_epochs == second.max_epochs == 50
    assert first.batch_size == second.batch_size == 32
    assert first.batch_order_sha256 == second.batch_order_sha256
    assert first.schedule_sha256 == second.schedule_sha256
    assert first.schedule.shape == (50, first.flat_indices.size)
    assert list(iter(first)) == list(iter(second))
    assert first._epoch == second._epoch == 1
    first.reset_epoch()
    second.reset_epoch()
    assert list(iter(first)) == list(iter(second))


def test_seed43_source_module_and_preflight_do_not_import_target_loader_symbols():
    source = (ROOT / "src/data/h1_carrierid_seed43.py").read_text(encoding="utf-8")
    preflight = (ROOT / "scripts/h1_carrierid_seed43_preflight.py").read_text(encoding="utf-8")
    assert "load_target_records" not in source
    assert "load_target_records" not in preflight
    assert "H1M4EBNormalizedV2StrictTargetDataset" not in source
    assert '"target_recordings_opened": 0' in preflight
    assert '"cuda_constructed_or_launched": False' in preflight
    assert '"checkpoint_selected": False' in preflight


def test_hc0_literal_zero_config_and_three_arm_launcher_are_present():
    config = (ROOT / "configs/experiment/h1_carrierid_hc0_seed43.yaml").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts/h1_carrierid_seed43_launcher.py").read_text(encoding="utf-8")
    assert "arm: zero" in config
    assert "zero_carrier: true" in config
    assert "seed: 43" in config
    assert "max_epochs: 50" in config
    assert "h_s_local_gpu0" in launcher
    assert "h_c_local_gpu1" in launcher
    assert "h_c0_remote_gpu0" in launcher
    assert "launch_authorized\": False" in launcher


def test_seed43_preflight_binds_a_new_seed_schedule_and_c_hc0_initial_equality():
    source = (ROOT / "scripts/h1_carrierid_seed43_preflight.py").read_text(encoding="utf-8")
    assert "manifest[\"calibration_schedule_sha256\"] != seed42_schedule" in source
    assert "H-C/H-C0 must begin as identical functions at seed43" in source
    assert '"selection": "none_fixed_terminal_epoch_only"' in source
