"""No-target and static-grid tests for the frozen RT R4 expansion."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import rt_r4_expansion as expansion


def test_static_expansion_is_exact_disjoint_balanced_48_cells() -> None:
    assert set(expansion.STATIC_LANES) == {0, 1}
    assert set(expansion.STATIC_LANES[0]).isdisjoint(expansion.STATIC_LANES[1])
    pairs = {pair for rows in expansion.STATIC_LANES.values() for pair in rows}
    assert pairs == {
        (budget, fold)
        for budget in (6, 12) for fold in (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13)
    }
    cells = [row for lane in (0, 1) for row in expansion.lane_cells(lane)]
    assert len(cells) == 48
    assert len({(row["budget"], row["fold"], row["arm"], row["seed"]) for row in cells}) == 48
    for lane in (0, 1):
        lane_rows = expansion.lane_cells(lane)
        assert len(lane_rows) == 24
        assert sum(row["budget"] == 6 for row in lane_rows) == 12
        assert sum(row["budget"] == 12 for row in lane_rows) == 12
        for budget, fold in expansion.STATIC_LANES[lane]:
            assert [row["arm"] for row in lane_rows if row["budget"] == budget and row["fold"] == fold] == [
                "afc4_vel", "afc4_mb4"
            ]


def test_preflight_binds_passing_pilot_and_has_no_launch_scope(tmp_path: Path) -> None:
    output = tmp_path / "preflight.json"
    body = expansion.build_preflight(output=output)
    assert body["status"] == expansion.PREFLIGHT_STATUS
    assert body["pilot_aggregate"]["sha256"] == expansion.PILOT_AGGREGATE_SHA256
    assert body["execution_preflight"]["sha256"] == expansion.EXECUTION_PREFLIGHT_SHA256
    assert body["expansion_cell_count"] == 48
    assert body["expansion_folds"] == list(expansion.EXPANSION_FOLDS)
    assert body["invariants"]["m18_authorized"] is False
    assert body["scope"] == {
        "nwb_files_opened": 0,
        "outer_target_payloads_opened": 0,
        "trainer_constructed": 0,
        "optimizer_constructed": 0,
        "cuda_queried": 0,
        "gpu_processes_started": 0,
        "tmux_sessions_created": 0,
        "commands_executed": 0,
    }


def test_immutable_temp_preflight_validates_code_closure(tmp_path: Path) -> None:
    output = tmp_path / "preflight.json"
    body = expansion.build_preflight(output=output)
    expansion._write_immutable(output, body)
    assert output.stat().st_mode & 0o777 == 0o444
    validated = expansion.validate_preflight(output)
    assert validated["static_dual_3090_lanes"]["0"]["serial_cells"] == expansion.lane_cells(0)
    with pytest.raises(expansion.RtR4ExpansionError, match="overwrite"):
        expansion._write_immutable(output, body)


def test_launch_receipt_is_static_and_result_blind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    preflight = tmp_path / "preflight.json"
    launch = tmp_path / "launch.json"
    expansion._write_immutable(preflight, expansion.build_preflight(output=preflight))
    monkeypatch.setattr(expansion, "_gpu_snapshot", lambda: {
        "0": {"index": 0, "name": "NVIDIA GeForce RTX 3090", "uuid": "GPU-0", "active_compute_owners": []},
        "1": {"index": 1, "name": "NVIDIA GeForce RTX 3090", "uuid": "GPU-1", "active_compute_owners": []},
    })
    body = expansion.build_launch_receipt(output=launch, preflight=preflight)
    assert body["status"] == expansion.LAUNCH_STATUS
    assert body["static_cell_count"] == 48
    assert body["lane_cell_counts"] == {"0": 24, "1": 24}
    assert body["scope"]["result_values_read"] is False
    assert body["scope"]["gpu_processes_started"] == 0
    assert body["prohibitions"]["m18"] is True
    for lane in (0, 1):
        command = body["commands"][str(lane)]
        assert "run-lane" in command
        assert command[command.index("--lane") + 1] == str(lane)
        assert command[command.index("--gpu") + 1] == str(lane)


def test_lane_terminals_are_distinct_from_pilot_lane_terminals(tmp_path: Path) -> None:
    assert expansion.lane_terminal(tmp_path, 0).name == "expansion_lane_0_terminal.json"
    assert expansion.lane_terminal(tmp_path, 1).name == "expansion_lane_1_terminal.json"
    assert expansion.lane_terminal(tmp_path, 0) != tmp_path / "lane_0_terminal.json"


def test_exact_sign_test() -> None:
    assert expansion._sign_test_two_sided(15, 0) == pytest.approx(2.0 / (2 ** 15))
    assert expansion._sign_test_two_sided(0, 15) == pytest.approx(2.0 / (2 ** 15))
    assert expansion._sign_test_two_sided(8, 7) == pytest.approx(1.0)
    assert expansion._sign_test_two_sided(0, 0) == pytest.approx(1.0)


def test_bad_pilot_hash_is_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "pilot.json"
    fake.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
    fake.chmod(0o444)
    with pytest.raises(expansion.RtR4ExpansionError, match="another R4 pilot aggregate"):
        expansion.validate_pilot_aggregate(fake)
