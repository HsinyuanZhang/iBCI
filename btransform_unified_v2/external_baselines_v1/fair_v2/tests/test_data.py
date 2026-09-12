"""Integration contracts for FAIR V2's raw-bin data plane."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FAIR = Path(__file__).resolve().parents[1]
if str(FAIR) not in sys.path:
    sys.path.insert(0, str(FAIR))
import data  # noqa: E402


def test_raw_support_is_a_canonical_query_timeline_subset():
    expected = {
        "m1": (4, 3, 99, 64, 3881, 10),
        "m2": (7, 6, 49, 96, 15403, 33),
        "h1": (13, 14, 299, 176, 33613, 3),
    }
    for task, (n_train, n_eval, pad, units, total_y, trials) in expected.items():
        loaded = data.load_task(task, include_evaluation=True)
        assert len(loaded["train"]) == n_train
        assert len(loaded["evaluation"]) == n_eval
        assert loaded["metadata"]["target_windows"] == total_y
        assert sum(len(x["Y"]) for x in loaded["evaluation"].values()) == total_y
        for item in (*loaded["train"].values(), *loaded["evaluation"].values()):
            raw = item["X"][item["pad"] :]
            indices = item["support_indices"]
            assert item["pad"] == pad
            assert raw.shape[1] == units
            assert indices.ndim == 1 and len(indices) > 0
            assert np.all((0 <= indices) & (indices < len(raw)))
            np.testing.assert_array_equal(item["support"], raw[indices])
            assert len(item["support_segments"]) == trials
            assert item["support_provenance"]["trial_ids"] == list(range(trials)) or task == "h1"
