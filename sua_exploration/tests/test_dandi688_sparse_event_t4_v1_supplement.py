from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from mc_maze.dandi688_sparse_event_t4_v1.descriptors import (
    materialize_sparse_event_t4,
    refit_from_single_pool,
    single_pool_interval_layout,
)
from mc_maze.dandi688_sparse_event_t4_v1.supplement import _held_immutable_bytes


def _trials() -> list[dict[str, float | int]]:
    return [
        {
            "trial_index": index,
            "start_time": float(index * 4),
            "stop_time": float(index * 4 + 3),
            "target_dir": float(-3 * np.pi / 4 + (index % 8) * np.pi / 4),
            "target_on_time": float(index * 4 + 0.5),
            "go_cue_time": float(index * 4 + 1.5),
        }
        for index in range(110)
    ]


def _pooler(_path: Path, intervals):
    start = np.asarray([row["start_time"] for row in intervals], dtype=np.float64)
    stop = np.asarray([row["stop_time"] for row in intervals], dtype=np.float64)
    duration = stop - start
    trial_index = np.floor(start / 4.0).astype(np.int64)
    angle = -3 * np.pi / 4 + (trial_index % 8) * np.pi / 4
    phase = np.where(np.isclose(duration, 0.3), -2.0, np.where(np.isclose(duration, 0.7), 2.0, 0.0))
    rates = np.vstack(
        (
            8.0 + phase + 2.0 * np.cos(angle),
            7.0 + phase + 3.0 * np.sin(angle),
            6.0 + phase + np.cos(angle) - np.sin(angle),
        )
    )
    return rates, 3


def test_single_pool_candidate_refit_matches_three_authoritative_calls():
    trials = _trials()
    direct = materialize_sparse_event_t4(
        Path("sub-C_ses-CO-20990101.nwb"),
        trial_lister=lambda *_args, **_kwargs: trials,
        rate_pooler=_pooler,
    )
    intervals, slices = single_pool_interval_layout(trials)
    pooled, _ = _pooler(Path("unused.nwb"), intervals)
    whole, post, profile = refit_from_single_pool(pooled, trials, slices, group="candidate")
    assert np.array_equal(whole, direct.whole_t4)
    assert np.array_equal(post, direct.post700_t4)
    assert np.array_equal(profile, direct.raw_profile)
    assert len(intervals) == 300
    validation_intervals, validation_slices = single_pool_interval_layout(trials, groups=("candidate",))
    assert len(validation_intervals) == 30
    assert set(validation_slices) == {"candidate:whole", "candidate:h300", "candidate:r700"}


def test_held_leaf_reader_requires_regular_0444_single_link(tmp_path: Path):
    leaf = tmp_path / "leaf.json"
    leaf.write_bytes(b"{}\n")
    os.chmod(leaf, 0o444)
    assert _held_immutable_bytes(leaf) == b"{}\n"

    writable = tmp_path / "writable.json"
    writable.write_bytes(b"{}\n")
    with pytest.raises(Exception, match="mode drift"):
        _held_immutable_bytes(writable)

    alias = tmp_path / "alias.json"
    alias.symlink_to(leaf)
    with pytest.raises(Exception, match="not regular"):
        _held_immutable_bytes(alias)


def test_zero_init_training_path_preserves_film_gradient():
    import torch

    from mc_maze.dandi688_sparse_event_t4_v1.core import build_film, film_identity

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.post_pool = torch.nn.Linear(68, 50)

    encoder = Encoder()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    film = build_film()
    mean = torch.randn(1, 3, 64)
    carrier = torch.randn(1, 3, 4)
    profile = torch.randn(1, 3, 4)

    direct, receipt = film_identity(encoder, mean, carrier, profile, film)
    expected = encoder.post_pool(torch.cat((mean, carrier), dim=-1))
    assert receipt["direct_native_branch"] is True
    assert torch.equal(direct, expected)

    differentiable, training_receipt = film_identity(
        encoder, mean, carrier, profile, film, direct_zero=False
    )
    assert training_receipt["direct_native_branch"] is False
    differentiable.square().mean().backward()
    assert film[2].bias.grad is not None
    assert torch.count_nonzero(film[2].bias.grad).item() > 0
