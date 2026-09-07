"""Focused tests for the CTXV2 Stage A dated source fork fidelity gate."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPINT = REPO / "SPINT-main"
if str(SPINT) not in sys.path:
    sys.path.insert(0, str(SPINT))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sua_exploration.mc_maze import h1_sparse_event_endpoint as event
from src.data.h1_context_event_carrier import H1_M4_FOLD0_SOURCE
from src.data.h1_context_event_snapshot_dated import _expected_shapes, sealed_map_manifest
from src.data.h1_context_event_source_dated import build_dated_source_module_for_snapshot
from src.data.h1_context_event_target_dated import (
    lodo_source_sessions,
    lodo_target_sessions,
    pure_cpu_fit_and_bind_map,
)

DATA = SPINT / "data/000954"
LAUNCHER_V2 = SPINT / "scripts/ctxv2_stage_a_launch_v2.sh"
FIDELITY_GATE = SPINT / "scripts/ctxv2_stage_a_fidelity_gate.py"
FOLD0_SUPPORTS = 116


@pytest.fixture(scope="module")
def data_dir() -> Path:
    if not DATA.is_dir():
        pytest.skip("public H1 held-in data unavailable")
    return DATA


def test_fold0_dated_source_matches_sealed_session_list_and_support_count(data_dir: Path) -> None:
    mapping = pure_cpu_fit_and_bind_map(data_dir, "19250101")
    module = build_dated_source_module_for_snapshot(data_dir, "19250101", frozen_map=mapping)
    assert tuple(module.pilot_manifest()["source_sessions"]) == H1_M4_FOLD0_SOURCE
    assert len(module.carrier_cache.entries) == FOLD0_SUPPORTS
    assert lodo_source_sessions("19250101") == H1_M4_FOLD0_SOURCE


def test_snapshot_shape_assertions_reject_wrong_support_count() -> None:
    shapes_ok = _expected_shapes(n_supports=116, n_active=22, n_features=20, rank=4)
    shapes_bad = _expected_shapes(n_supports=109, n_active=22, n_features=20, rank=4)
    assert shapes_ok["source_carriers"] == (116, event.EXPECTED_NEURONS, 5)
    assert shapes_bad["source_carriers"] == (109, event.EXPECTED_NEURONS, 5)
    assert shapes_ok["source_carriers"] != shapes_bad["source_carriers"]


def test_19250108_source_pool_excludes_all_target_recordings(data_dir: Path) -> None:
    source = lodo_source_sessions("19250108")
    target = lodo_target_sessions("19250108")
    assert len(source) == 10
    assert len(target) == 3
    assert not set(source).intersection(target)
    assert all(event.session_date(name) == "19250108" for name in target)
    assert all(event.session_date(name) != "19250108" for name in source)
    mapping = pure_cpu_fit_and_bind_map(data_dir, "19250108")
    module = build_dated_source_module_for_snapshot(data_dir, "19250108", frozen_map=mapping)
    assert len(module.carrier_cache.entries) == 109
    assert module.pilot_manifest()["fold_date"] == "19250108"


def test_launcher_v2_dry_run_prints_three_arms_with_distinct_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/tmux")
    completed = subprocess.run(
        ["bash", "-c", f'''
          tmux() {{
            if [[ "$1" == "has-session" ]]; then
              return 1
            fi
            command tmux "$@"
          }}
          export -f tmux
          bash "{LAUNCHER_V2}" --dry-run
        '''],
        check=False,
        capture_output=True,
        text=True,
        cwd=SPINT,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "CUDA_VISIBLE_DEVICES=0" in output
    assert "CUDA_VISIBLE_DEVICES=1" in output
    assert output.count("CUDA_VISIBLE_DEVICES=0") >= 2
    assert "ctxv2_ctx" in output
    assert "ctxv2_hse5" in output
    assert "ctxv2_zero" in output
    assert "PYTHONPATH=" in output and f"{SPINT}:{REPO}" in output
    assert "--dry-run: no tmux sessions started." in output
    assert "tmux new-session" not in output


def test_launcher_v2_refuses_while_stage_b_tmux_session_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/tmux")
    completed = subprocess.run(
        ["bash", "-c", f'''
          has_session() {{ [[ "$1" == "ctxv2_ls" ]]; }}
          export -f has_session
          tmux() {{
            if [[ "$1" == "has-session" ]]; then
              has_session "$3"
              return $?
            fi
            command tmux "$@"
          }}
          export -f tmux
          bash "{LAUNCHER_V2}" --dry-run
        '''],
        check=False,
        capture_output=True,
        text=True,
        cwd=SPINT,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 5
    assert "ctxv2_ls" in output


@pytest.mark.torch
def test_fold0_manifest_matches_sealed_pin(data_dir: Path) -> None:
    pytest.importorskip("lightning.pytorch")
    mapping = pure_cpu_fit_and_bind_map(data_dir, "19250101")
    module = build_dated_source_module_for_snapshot(data_dir, "19250101", frozen_map=mapping)
    expected_manifest = "c49694d850c426d58c10f3da5271bcb472e9c52d95963d619a7134a48e6adb78"
    assert module.pilot_manifest_sha256 == expected_manifest
    assert module.latent_map.map_sha256 == sealed_map_manifest("19250101")["map_sha256"]
