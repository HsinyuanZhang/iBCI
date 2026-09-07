"""CPU tests for source-pick sealing and ext-4 report aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tfpd_exploration.src.m2_b_small_stability_v1 import config as cfg
from tfpd_exploration.src.m2_b_small_stability_v1.score import (
    require_both_summaries,
    seal_source_picks,
    select_epoch,
    _route,
    _view_report,
)


def test_select_epoch_uses_earliest_tie() -> None:
    assert select_epoch({3: 0.2, 1: 0.4, 2: 0.4, 4: 0.1}) == 1
    assert select_epoch({8: 0.11, 7: 0.10}) == 8


def test_require_both_summaries_refuses_incomplete(tmp_path: Path) -> None:
    with pytest.raises(cfg.SmallStabilityError):
        require_both_summaries(root=tmp_path)


def _write_finished(root: Path, cell: str, raw_pick: int, ema_pick: int) -> None:
    dest = root / cell.replace("-", "_") / "seed42"
    dest.mkdir(parents=True)
    raw = {str(epoch): 0.1 + 0.01 * epoch for epoch in range(1, 25)}
    ema = {str(epoch): 0.12 + 0.008 * epoch for epoch in range(1, 25)}
    raw[str(raw_pick)] = 0.9
    ema[str(ema_pick)] = 0.91
    (dest / "summary.json").write_text(
        json.dumps(
            {
                "cell": cell,
                "init_sha256": "abc",
                "source_pick_raw": raw_pick,
                "source_pick_ema": ema_pick,
                "epoch_minival_raw": raw,
                "epoch_minival_ema": ema,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_seal_source_picks_primary_is_s1_ema(tmp_path: Path) -> None:
    _write_finished(tmp_path, cfg.CELL_S0, 4, 6)
    _write_finished(tmp_path, cfg.CELL_S1, 5, 9)
    sealed = seal_source_picks(root=tmp_path)
    assert sealed["primary_candidate"] == "S1-SMALL-COS/EMA"
    assert sealed["picks"][cfg.CELL_S1]["EMA"]["epoch"] == 9
    again = seal_source_picks(root=tmp_path)
    assert again["picks"] == sealed["picks"]


def test_view_report_keeps_lastk_and_visible_separate() -> None:
    all_epochs = {
        str(epoch): {"R_session_equal_mean": 0.30 + 0.001 * epoch}
        for epoch in range(1, 25)
    }
    all_epochs["20"]["R_session_equal_mean"] = 0.40
    scan = {"views": {"EMA": {"all_epochs": all_epochs}}}
    report = _view_report(scan, "EMA", source_pick=9)
    assert report["source_pick_epoch"] == 9
    assert report["visible_ext4_pick_epoch"] == 20
    assert report["last4"]["n"] == 4
    assert report["last8"]["n"] == 8
    assert report["endpoint24"] == pytest.approx(0.324)
    assert report["visible_ext4_pick"] != report["source_pick_ext4"]


def test_route_does_not_promote_visible_only() -> None:
    primary = {
        "source_pick_delta": 0.001,
        "last8": {"mean": 0.40},
    }
    per_session = {session: {"r2": 0.4} for session in ("a",)}
    ref = {"a": 0.3}
    assert _route(primary, per_session, ref) == "COMPLETE_NOT_OVER_GATE"
