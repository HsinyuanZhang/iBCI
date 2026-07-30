"""M3: deterministic epoch-window checkpoint rule for DANDI 000688 SUA runs.

Tests for sua_exploration/scripts/eval_epoch_window_dandi688.py's pure selection/scoring
logic (no GPU, no NWB data, no checkpoints with real weights). See
sua_exploration/docs/CURRENT_RESULTS.md section H.3 for why this replaces argmax
checkpoint selection: train exactly 12 epochs, score = mean of the protocol metric over
the trailing 8 epochs (5..12, a 4-epoch burn-in).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sua_exploration" / "scripts"))

import eval_epoch_window_dandi688 as ew  # noqa: E402


def test_protocol_epochs_hardcoded_to_5_through_12():
    assert ew.PROTOCOL_EPOCHS == (5, 6, 7, 8, 9, 10, 11, 12)
    assert ew.TOTAL_TRAINING_EPOCHS == 12
    assert ew.BURN_IN_EPOCHS == 4
    assert ew.FIXED_SELECTION_MODE == "first"
    assert ew.FIXED_CALIBRATION_N == 30
    assert ew.FIXED_POOL_SIZE == 50


@pytest.mark.parametrize(
    "protocol_epoch,lightning_epoch",
    [(1, 0), (4, 3), (5, 4), (12, 11)],
)
def test_lightning_epoch_index_mapping(protocol_epoch, lightning_epoch):
    assert ew.lightning_epoch_index(protocol_epoch) == lightning_epoch


def test_lightning_epoch_index_rejects_epoch_zero_or_negative():
    with pytest.raises(ValueError):
        ew.lightning_epoch_index(0)
    with pytest.raises(ValueError):
        ew.lightning_epoch_index(-1)


def test_epoch_checkpoint_path_uses_lightning_native_filename(tmp_path):
    path = ew.epoch_checkpoint_path(tmp_path, 5)
    assert path == tmp_path / "epoch_004.ckpt"
    path12 = ew.epoch_checkpoint_path(tmp_path, 12)
    assert path12 == tmp_path / "epoch_011.ckpt"


def _make_epoch_ckpts(run_dir: Path, lightning_epochs: range) -> Path:
    epoch_ckpt_dir = run_dir / "epoch_ckpts"
    epoch_ckpt_dir.mkdir(parents=True)
    for lightning_epoch in lightning_epochs:
        (epoch_ckpt_dir / f"epoch_{lightning_epoch:03d}.ckpt").write_bytes(b"fake")
    return epoch_ckpt_dir


def test_select_epoch_window_checkpoints_picks_exactly_epochs_5_to_12(tmp_path):
    run_dir = tmp_path / "run"
    # Simulate a full 12-epoch run: Lightning-native epoch_000..epoch_011.
    _make_epoch_ckpts(run_dir, range(0, 12))

    selected = ew.select_epoch_window_checkpoints(run_dir)

    assert sorted(selected.keys()) == [5, 6, 7, 8, 9, 10, 11, 12]
    assert len(selected) == 8
    # protocol epoch 5 -> lightning epoch_004, protocol epoch 12 -> lightning epoch_011.
    assert selected[5].name == "epoch_004.ckpt"
    assert selected[12].name == "epoch_011.ckpt"
    # Burn-in epochs 1-4 (lightning epoch_000..epoch_003) must not be selected even
    # though the checkpoint files exist on disk.
    selected_names = {path.name for path in selected.values()}
    assert "epoch_000.ckpt" not in selected_names
    assert "epoch_003.ckpt" not in selected_names


def test_select_epoch_window_checkpoints_raises_if_run_stopped_early(tmp_path):
    run_dir = tmp_path / "run"
    # Early-stopped or truncated run: only epochs 0-7 exist (protocol epochs 1-8), so
    # protocol epochs 9-12 are missing.
    _make_epoch_ckpts(run_dir, range(0, 8))

    with pytest.raises(FileNotFoundError, match="Missing checkpoint"):
        ew.select_epoch_window_checkpoints(run_dir)


def test_select_epoch_window_checkpoints_raises_if_epoch_ckpts_dir_absent(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="epoch_ckpts"):
        ew.select_epoch_window_checkpoints(run_dir)


def test_compute_variant_score_is_unweighted_mean_of_the_8_epoch_window():
    per_epoch_mean_r2 = {5: 0.10, 6: 0.20, 7: 0.30, 8: 0.40, 9: 0.50, 10: 0.60, 11: 0.70, 12: 0.80}
    score = ew.compute_variant_score(per_epoch_mean_r2)
    assert score == pytest.approx(0.45)  # mean(0.1..0.8 step 0.1) == 0.45


def test_compute_variant_score_ignores_ordering_but_not_membership():
    forward = {5: 0.1, 6: 0.2, 7: 0.3, 8: 0.4, 9: 0.5, 10: 0.6, 11: 0.7, 12: 0.8}
    shuffled = {12: 0.8, 5: 0.1, 9: 0.5, 6: 0.2, 11: 0.7, 7: 0.3, 10: 0.6, 8: 0.4}
    assert ew.compute_variant_score(forward) == ew.compute_variant_score(shuffled)


def test_compute_variant_score_rejects_wrong_epoch_set():
    # Missing epoch 12, extra epoch 13: must not silently average over the wrong window.
    wrong = {5: 0.1, 6: 0.2, 7: 0.3, 8: 0.4, 9: 0.5, 10: 0.6, 11: 0.7, 13: 0.9}
    with pytest.raises(ValueError, match="exactly epochs"):
        ew.compute_variant_score(wrong)


def test_compute_variant_score_rejects_burn_in_epochs_included():
    # Someone accidentally passing epochs 1-8 instead of 5-12 must be rejected, not
    # silently scored -- this is the exact bug class M3 exists to prevent.
    burn_in_included = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.5, 6: 0.5, 7: 0.5, 8: 0.5}
    with pytest.raises(ValueError, match="exactly epochs"):
        ew.compute_variant_score(burn_in_included)


def test_compute_variant_score_rejects_partial_window():
    partial = {5: 0.1, 6: 0.2, 7: 0.3}
    with pytest.raises(ValueError, match="exactly epochs"):
        ew.compute_variant_score(partial)


def _valid_metadata(**overrides) -> dict:
    base = {
        "status": "completed",
        "held_out_test_evaluated": False,
        "training": {
            "max_epochs": 12,
            "no_early_stopping": True,
            "checkpoint_every_epoch": True,
        },
    }
    base.update({key: value for key, value in overrides.items() if key != "training"})
    if "training" in overrides:
        base["training"].update(overrides["training"])
    return base


def test_validate_run_metadata_accepts_the_golden_path():
    ew._validate_run_metadata_for_epoch_window(_valid_metadata())


def test_validate_run_metadata_rejects_wrong_max_epochs():
    metadata = _valid_metadata(training={"max_epochs": 20})
    with pytest.raises(ValueError, match="max_epochs"):
        ew._validate_run_metadata_for_epoch_window(metadata)


def test_validate_run_metadata_rejects_early_stopping_enabled():
    metadata = _valid_metadata(training={"no_early_stopping": False})
    with pytest.raises(ValueError, match="no_early_stopping"):
        ew._validate_run_metadata_for_epoch_window(metadata)


def test_validate_run_metadata_rejects_missing_checkpoint_every_epoch():
    metadata = _valid_metadata(training={"checkpoint_every_epoch": False})
    with pytest.raises(ValueError, match="checkpoint_every_epoch"):
        ew._validate_run_metadata_for_epoch_window(metadata)


def test_validate_run_metadata_rejects_incomplete_run():
    metadata = _valid_metadata(status="initialized")
    with pytest.raises(ValueError, match="completed"):
        ew._validate_run_metadata_for_epoch_window(metadata)


def test_validate_run_metadata_rejects_held_out_test_evaluated():
    metadata = _valid_metadata(held_out_test_evaluated=True)
    with pytest.raises(ValueError, match="held_out_test_evaluated"):
        ew._validate_run_metadata_for_epoch_window(metadata)
