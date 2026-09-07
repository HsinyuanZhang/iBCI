"""Synthetic no-data contracts for the target-pcs screen (CPU-only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data import h1_target_pcs_screen as tps
from src.data import h1_lag_screen as ls


def _synthetic_records(n_sessions: int = 4, n_channels: int = 176, seed: int = 42):
    from types import SimpleNamespace

    rng = np.random.default_rng(seed)
    records = {}
    for i in range(n_sessions):
        trials = []
        for t in range(7):
            n = 20
            trials.append(
                SimpleNamespace(
                    trial_number=float(t + 1),
                    rates=rng.uniform(1, 10, (n, n_channels)),
                    velocity=rng.normal(size=(n, 7)),
                    block_indices=np.tile(np.arange(t * 100, t * 100 + n)[:, None], (1, 5)),
                )
            )
        records[f"ses-r{i}"] = SimpleNamespace(
            session_name=f"ses-r{i}",
            date=f"1925010{i + 1}",
            neural=rng.uniform(0, 1, (700, n_channels)),
            velocity=rng.normal(size=(700, 7)),
            eval_mask=np.ones(700, dtype=bool),
            trial_num=np.concatenate([np.full(100, float(t + 1)) for t in range(7)]),
            trials=trials,
        )
    return records


# --------------------------------------------------------------------------- #
# Canonicalization.
# --------------------------------------------------------------------------- #
def test_canonicalize_flips_negative_anchor_positive():
    components = np.array([[0.1, -0.9, 0.2], [-0.8, 0.1, 0.1]])
    canon = tps.canonicalize_components(components)
    for k in range(canon.shape[0]):
        anchor = int(np.argmax(np.abs(canon[k])))
        assert canon[k, anchor] > 0.0


def test_canonicalize_ties_break_by_lowest_index():
    # Two channels tie for largest |loading|; lowest index (0) must anchor the sign.
    components = np.array([[-0.5, 0.5, 0.1]])
    canon = tps.canonicalize_components(components)
    assert canon[0, 0] > 0.0  # index 0 wins the tie under np.argmax and must end up positive


def test_canonicalize_is_idempotent():
    rng = np.random.default_rng(3)
    components = rng.normal(size=(4, 20))
    once = tps.canonicalize_components(components)
    twice = tps.canonicalize_components(once)
    assert np.array_equal(once, twice)


def test_canonicalize_does_not_reorder_rows():
    rng = np.random.default_rng(9)
    components = rng.normal(size=(5, 30))
    canon = tps.canonicalize_components(components)
    assert canon.shape == components.shape
    # Only sign per row may change, magnitudes are identical.
    assert np.allclose(np.abs(canon), np.abs(components))


# --------------------------------------------------------------------------- #
# fit_target_pcs contract.
# --------------------------------------------------------------------------- #
def test_fit_target_pcs_zeros_dead_channel_column():
    records = _synthetic_records()
    plan = ls.build_plan(records)
    pcs_target = tps.fit_target_pcs(records["ses-r0"], plan)
    assert pcs_target.shape == (plan.q, 176)
    assert np.all(pcs_target[:, tps.DEAD_CHANNEL] == 0.0)


def test_fit_target_pcs_rows_are_unit_norm_on_kept_channels():
    records = _synthetic_records()
    plan = ls.build_plan(records)
    pcs_target = tps.fit_target_pcs(records["ses-r0"], plan)
    keep = np.ones(176, dtype=bool)
    keep[tps.DEAD_CHANNEL] = False
    norms = np.linalg.norm(pcs_target[:, keep], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8)


def test_fit_target_pcs_fails_closed_on_too_few_blocks():
    from types import SimpleNamespace

    rng = np.random.default_rng(1)
    # Fix q=16 explicitly rather than depend on the grid search's data-dependent
    # pick, so the "too few blocks" condition is deterministic.
    plan = ls.LagScreenPlan(
        mean=np.zeros(176), scale=np.ones(176), pcs=rng.normal(size=(16, 176)),
        q=16, ridge_lambda=1.0, source_grid_r2=0.0,
    )
    # A support block with fewer rows than plan.q=16 must raise, not silently truncate.
    tiny_trials = [
        SimpleNamespace(
            trial_number=float(t + 1),
            rates=rng.uniform(1, 10, (1, 176)),
            velocity=rng.normal(size=(1, 7)),
            block_indices=np.zeros((1, 5), dtype=np.int64),
        )
        for t in range(4)
    ]
    tiny_record = SimpleNamespace(session_name="ses-tiny", trials=tiny_trials)
    with pytest.raises(tps.TargetPcsScreenError):
        tps.fit_target_pcs(tiny_record, plan)


def test_build_target_fit_carrier_matches_shape_of_production():
    from src.data.h1_content_lever_screen import build_h1_carrier

    records = _synthetic_records()
    plan = ls.build_plan(records)
    # compute_h1_source_U hardcodes the real H1_M4_FOLD0_SOURCE session names,
    # so this shape-only test supplies its own [7,4] U rather than depending
    # on synthetic session names matching the real fold-0 roster.
    source_U = np.random.default_rng(4).normal(size=(7, 4))
    frozen_carrier = build_h1_carrier(records["ses-r0"], plan, source_U)
    target_carrier, pcs_target = tps.build_target_fit_carrier(records["ses-r0"], plan, source_U)
    assert target_carrier.shape == frozen_carrier.shape
    assert pcs_target.shape == (plan.q, 176)
    # The variant must not be byte-identical to the frozen production carrier
    # (a target-fit pcs on random synthetic data essentially never coincides
    # with the pooled-source pcs).
    assert not np.array_equal(target_carrier, frozen_carrier)


# --------------------------------------------------------------------------- #
# Per-column normalizer.
# --------------------------------------------------------------------------- #
def test_per_column_scale_preserves_zero_carrier():
    carriers = {"a": np.array([[1.0, 2.0, 0.0, -1.0], [3.0, -2.0, 0.5, 1.0], [0.0, 0.0, 0.0, 0.0]])}
    # Pad to 176 rows minimum for _exclude_dead_rows to have a valid dead-channel index.
    carriers = {"a": np.tile(carriers["a"], (60, 1))[:176]}
    s_j = tps.fit_per_column_scale(carriers)
    zero_row = np.zeros((1, 4))
    normalized = tps.apply_per_column_scale(zero_row, s_j)
    assert np.all(normalized == 0.0)


def test_per_column_scale_has_no_centring():
    # A constant nonzero column must map to a constant nonzero column (no mean
    # subtraction), and the scale is exactly the source SD of that column
    # (floored), never adjusted by a mean.
    carrier = np.zeros((176, 4))
    carrier[:, 0] = 5.0  # constant column: SD is 0, so the normalizer floors at 1e-6.
    carrier[:, 1] = np.linspace(-1, 1, 176)
    s_j = tps.fit_per_column_scale({"a": carrier})
    assert s_j[0] == pytest.approx(1.0e-6)
    normalized = tps.apply_per_column_scale(carrier, s_j)
    assert np.all(normalized[:, 0] == 5.0 / 1.0e-6)


def test_exclude_dead_rows_drops_exactly_one_row():
    carrier = np.arange(176 * 4).reshape(176, 4).astype(np.float64)
    reduced = tps._exclude_dead_rows(carrier)
    assert reduced.shape == (175, 4)
    assert tps.DEAD_CHANNEL not in reduced[:, 0].tolist()  # crude but the dead row's marker value is gone


# --------------------------------------------------------------------------- #
# Overlap R^2 is scale-invariant to a per-column rescale of the carrier.
# --------------------------------------------------------------------------- #
def test_overlap_residual_r2_is_scale_invariant():
    rng = np.random.default_rng(11)
    carrier = rng.normal(size=(50, 4))
    activity = rng.normal(size=(50, 8))
    raw = tps._overlap_residual_r2_rows(carrier, activity)
    scaled = tps._overlap_residual_r2_rows(carrier * np.array([2.0, -3.0, 0.5, 7.0]), activity)
    assert np.allclose(raw["residual_r2_per_dim"], scaled["residual_r2_per_dim"], atol=1e-10)


# --------------------------------------------------------------------------- #
# Receipt.
# --------------------------------------------------------------------------- #
def test_receipt_refuses_overwrite(tmp_path: Path):
    p = tmp_path / "receipt.json"
    result = {"schema": tps.TARGET_PCS_SCHEMA, "test": True}
    tps.write_receipt(p, result)
    assert p.is_file()
    with pytest.raises(FileExistsError):
        tps.write_receipt(p, result)


def test_run_target_pcs_screen_requires_all_eleven_source_records():
    with pytest.raises(tps.TargetPcsScreenError):
        tps.run_target_pcs_screen({}, hc0_checkpoint="/nonexistent.ckpt")
