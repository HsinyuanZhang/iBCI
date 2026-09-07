from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from h1_date_lodo_activity_headroom_v1.plan import AUTHORITIES, DATE_ORDER, breadth_decision, dry_plan
from h1_date_lodo_activity_headroom_v1.evaluate import _DateLodoDatasetAdapter


ARMS = ("STATIC_SUPPORT", "ROLLING_FIXED_M", "CAUSAL_GROWING_CAP30", "FULL_SESSION_ORACLE")


def _rows(growing_deltas: list[float]):
    rows = []
    for date, delta in zip(DATE_ORDER, growing_deltas, strict=True):
        results = []
        for arm in ARMS:
            value = 0.4 + (delta if arm == "CAUSAL_GROWING_CAP30" else 0.0)
            results.append({"arm": arm, "equal_recording_mean_r2": value})
        rows.append({"outer_date": date, "results": results})
    return rows


def test_authority_is_exact_five_date_checkpoint_set():
    assert tuple(AUTHORITIES) == DATE_ORDER
    assert len({row.checkpoint_sha256 for row in AUTHORITIES.values()}) == 5
    assert all(len(row.checkpoint_sha256) == len(row.config_sha256) == len(row.terminal_sha256) == 64 for row in AUTHORITIES.values())
    assert all(row.checkpoint_relative.endswith("epoch_049.ckpt") for row in AUTHORITIES.values())


def test_breadth_pass_requires_four_dates_and_mean_point_zero_one():
    passed = breadth_decision(_rows([0.02, 0.02, 0.02, 0.02, -0.001]))
    assert passed["breadth_pass"] is True
    assert passed["positive_growing_dates"] == 4
    assert passed["verdict"].startswith("PASS_")
    too_narrow = breadth_decision(_rows([0.03, 0.03, 0.03, -0.001, -0.001]))
    assert too_narrow["breadth_pass"] is False
    too_small = breadth_decision(_rows([0.002] * 5))
    assert too_small["breadth_pass"] is False


def test_decision_rejects_reorder_or_arm_drift():
    rows = _rows([0.02] * 5)
    with pytest.raises(ValueError, match="authority order"):
        breadth_decision(list(reversed(rows)))
    broken = copy.deepcopy(rows)
    broken[0]["results"].pop()
    with pytest.raises(ValueError, match="four-arm"):
        breadth_decision(broken)


def test_dry_plan_is_inert_and_predeclares_gate():
    payload = dry_plan()
    assert payload["status"] == "DRY_NO_TARGET_NO_CHECKPOINT_LOAD_NO_CUDA_NO_WRITE"
    assert payload["dates"] == list(DATE_ORDER)
    assert payload["decision"] == {"min_positive_growing_dates": 4, "min_equal_date_growing_delta": 0.01}
    assert payload["target_updates"] == 0


def test_date_lodo_adapter_only_renames_the_fixed_carrier_surface():
    carrier = object()
    class Base:
        records = {"session": object()}
        window_indices = [("session", 7)]
        support = {"session": SimpleNamespace(normalized_carrier=carrier)}

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return ("row", index)

    base = Base()
    adapted = _DateLodoDatasetAdapter(base)
    assert adapted.records is base.records
    assert adapted.window_indices is base.window_indices
    assert adapted.support["session"].carriers["full"] is carrier
    assert adapted[0] == ("row", 0)
