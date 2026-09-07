from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
H1_SRC = ROOT / "tfpd_exploration" / "h1_series_20260830" / "src"
SPINT_MAIN = ROOT / "SPINT-main"
for value in (str(H1_SRC), str(SPINT_MAIN)):
    if value not in sys.path:
        sys.path.insert(0, value)

from h1_cross_record_postpool_v1.core import anchored_identity, identity_states, query_member, stream_window
from h1_cross_record_postpool_v1.plan import ARM_ORDER, DATE_ORDER, decide_oof, select_source_arm


def test_source_selection_prefers_post_pool_when_noninferior_and_uses_smallest_gate() -> None:
    values = {name: 0.20 for name in ARM_ORDER}
    values.update(
        {
            "RN-G100": 0.2200,
            "RP-G005": 0.2182,
            "RP-G010": 0.2190,
            "RP-G020": 0.2198,
            "RP-G050": 0.2199,
            "RP-G100": 0.2197,
        }
    )
    selected = select_source_arm(values)
    assert selected["family"] == "RP"
    assert selected["arm"] == "RP-G010"
    assert selected["gate"] == pytest.approx(0.10)


def test_source_selection_keeps_native_when_post_pool_is_inferior() -> None:
    values = {name: 0.20 for name in ARM_ORDER}
    values["RN-G020"] = 0.2300
    values["RN-G050"] = 0.2295
    values["RP-G100"] = 0.2270
    selected = select_source_arm(values)
    assert selected["arm"] == "RN-G020"
    assert selected["gate"] == pytest.approx(0.20)


def test_oof_gate_is_exact_and_date_ordered() -> None:
    rows = [
        {"outer_date": date, "selected_delta_vs_static": delta}
        for date, delta in zip(DATE_ORDER, (0.010, 0.009, 0.006, 0.002, -0.001), strict=True)
    ]
    decision = decide_oof(rows)
    assert decision["pass"] is True
    assert decision["nonnegative_dates"] == 4
    assert decision["worst_date_gain"] == pytest.approx(-0.001)


def test_query_member_and_stream_window_are_causal_and_deterministic() -> None:
    neural = np.arange(800 * 176, dtype=np.float32).reshape(800, 176) / 1000.0
    first = query_member(neural)
    second = query_member(neural.copy())
    assert first.shape == (1024, 176)
    assert np.array_equal(first, second)
    early = stream_window(neural, 3)
    assert early.shape == (700, 176)
    assert np.count_nonzero(early[:-4]) == 0
    assert np.array_equal(early[-4:], neural[:4])
    late = stream_window(neural, 799)
    assert np.array_equal(late, neural[100:800])


def test_post_pool_identity_is_well_formed_and_zero_anchor_is_exact() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(11)
    net = SimpleNamespace(
        zero_carrier=False,
        carrier_pre_pool=torch.nn.Sequential(torch.nn.Linear(1024, 32), torch.nn.ReLU()),
        carrier_post_pool=torch.nn.Sequential(
            torch.nn.Linear(36, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 700),
        ),
    )
    rng = np.random.default_rng(7)
    support = rng.normal(size=(3, 1024, 176)).astype(np.float32)
    chunk = rng.normal(size=(1024, 176)).astype(np.float32)
    carrier = rng.normal(size=(176, 4)).astype(np.float32)
    states = identity_states(net, support, chunk, carrier, device="cpu")
    assert tuple(states) == ("static", "native", "post")
    assert all(tuple(value.shape) == (176, 700) for value in states.values())
    anchored = anchored_identity(states["static"], states["post"], 0.0)
    assert anchored is states["static"]
    moved = anchored_identity(states["static"], states["post"], 0.2)
    expected = states["static"] + 0.2 * (states["post"] - states["static"])
    assert torch.equal(moved, expected)
