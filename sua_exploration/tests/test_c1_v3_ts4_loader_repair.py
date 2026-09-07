"""Direct regression tests for the C1 v3 TS4 input-pipeline repair.

These tests deliberately stub the raw feature computation.  They exercise the
public loader contract without opening a DANDI NWB file, so they are suitable as
prelaunch microfit gates and cannot touch formal SUA data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))

from mc_maze import unit_side_features as side_features  # noqa: E402


_RAW_BY_VIEW = {
    "sua": np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0],
        ],
        dtype=np.float32,
    ),
    "pseudo_mua": np.asarray(
        [
            [20.0, 21.0, 22.0, 23.0],
            [24.0, 25.0, 26.0, 27.0],
            [28.0, 29.0, 30.0, 31.0],
        ],
        dtype=np.float32,
    ),
}
_MEAN = np.asarray([1.5, -2.0, 0.5, 4.0], dtype=np.float32)
_STD = np.asarray([2.0, 3.0, 4.0, 5.0], dtype=np.float32)


def _install_fake_t4_compute(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Patch only raw T4 computation; no on-disk data access is possible."""
    calls: list[dict[str, object]] = []

    def fake_compute(nwb_path: Path, **kwargs: object):
        group = kwargs["feature_group"]
        signal_view = kwargs["signal_view"]
        assert group == "t4"
        assert signal_view in _RAW_BY_VIEW
        calls.append(
            {
                "path": nwb_path.name,
                "feature_group": group,
                "signal_view": signal_view,
            }
        )
        raw = _RAW_BY_VIEW[str(signal_view)]
        metadata = side_features.SideFeatureMetadata(
            feature_group="t4",
            feature_version=1,
            pool_size=int(kwargs["pool_size"]),
            cache_key="synthetic-t4",
            degenerate_unit_count=0,
            zero_spike_unit_count=0,
            single_spike_unit_count=0,
            zero_noise_std_unit_count=0,
            zero_template_max_unit_count=0,
        )
        return raw.copy(), metadata

    monkeypatch.setattr(
        side_features, "compute_unit_side_features_uncached", fake_compute
    )
    return calls


def _load(
    *,
    group: str,
    signal_view: str,
    seed: int | None = None,
) -> np.ndarray:
    values, _ = side_features.load_unit_side_features(
        Path(f"sub-C_ses-CO-{signal_view}_synthetic_behavior+ecephys.nwb"),
        feature_group=group,
        pool_size=50,
        mean=_MEAN,
        std=_STD,
        cache_dir=None,
        permutation_seed=seed,
        signal_view=signal_view,
    )
    return values


def test_v3_direct_loader_resolves_ts4_to_t4_before_known_group_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_t4_compute(monkeypatch)

    values = _load(group="ts4", signal_view="sua", seed=43)

    assert values.shape == _RAW_BY_VIEW["sua"].shape
    assert calls == [
        {
            "path": "sub-C_ses-CO-sua_synthetic_behavior+ecephys.nwb",
            "feature_group": "t4",
            "signal_view": "sua",
        }
    ]


def test_v3_t4_loader_remains_bit_exact_after_control_resolution_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_t4_compute(monkeypatch)

    observed = _load(group="t4", signal_view="sua")
    expected = ((_RAW_BY_VIEW["sua"] - _MEAN) / _STD).astype(np.float32)

    # The repair is only validation ordering.  Ordinary T4 stays byte-for-byte
    # equal to its prior normalized substrate calculation.
    assert np.array_equal(observed, expected)
    assert calls[0]["feature_group"] == "t4"


@pytest.mark.parametrize("signal_view", ("sua", "pseudo_mua"))
def test_v3_ts4_is_seeded_nonidentity_complete_row_shuffle_in_both_views(
    monkeypatch: pytest.MonkeyPatch,
    signal_view: str,
) -> None:
    _install_fake_t4_compute(monkeypatch)

    aligned = _load(group="t4", signal_view=signal_view)
    shuffled = _load(group="ts4", signal_view=signal_view, seed=43)
    repeated = _load(group="ts4", signal_view=signal_view, seed=43)
    alternate_seed = _load(group="ts4", signal_view=signal_view, seed=44)

    seed_43_order = np.random.RandomState(43).permutation(aligned.shape[0])
    seed_44_order = np.random.RandomState(44).permutation(aligned.shape[0])
    assert np.array_equal(shuffled, aligned[seed_43_order])
    assert np.array_equal(repeated, shuffled)
    assert np.array_equal(alternate_seed, aligned[seed_44_order])
    assert shuffled.shape == aligned.shape == _RAW_BY_VIEW[signal_view].shape
    assert not np.array_equal(shuffled, aligned)
    assert not np.array_equal(alternate_seed, shuffled)
    # A TS4 control changes attachment only: complete normalized rows are
    # retained exactly and no T4 cache/normalizer value is recomputed.
    assert np.array_equal(
        np.sort(shuffled, axis=0), np.sort(aligned, axis=0)
    )
