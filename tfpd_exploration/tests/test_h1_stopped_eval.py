"""Read-only local stop-evaluation policy tests, without production forwards."""
import pytest
from tfpd_exploration.src.h1_queryage_family_v1 import stopped_eval as s


def rows(end):
    return {e: {"epoch": e, "selection": {"n_bins": 2908, "r2_concat_float64": .5 - e / 1000}}
            for e in range(13, end + 1)}


def test_latest_common_committed_not_later_one_arm_or_maximum_score():
    assert s.latest_common_epoch({"flat": rows(21), "route": rows(22)}) == 21


def test_missing_epoch_rejected():
    flat = rows(21); del flat[17]
    with pytest.raises(RuntimeError, match="noncontiguous"):
        s.latest_common_epoch({"flat": flat, "route": rows(22)})


def test_nonfinite_or_wrong_surface_rejected():
    for patch in ({"n_bins": 20325}, {"r2_concat_float64": float("nan")}):
        flat = rows(21); flat[21]["selection"].update(patch)
        with pytest.raises(RuntimeError, match="selection"):
            s.latest_common_epoch({"flat": flat, "route": rows(22)})
