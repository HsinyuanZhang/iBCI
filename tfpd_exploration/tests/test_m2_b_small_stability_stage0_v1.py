"""Stage0 receipts and CLI refuse-train contracts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tfpd_exploration.src.m2_b_small_stability_v1 import config as cfg
from tfpd_exploration.src.m2_b_small_stability_v1.launch import run_train, train_is_authorized
from tfpd_exploration.src.m2_b_small_stability_v1.stage0 import run_stage0


REPO = Path(__file__).resolve().parents[2]


def test_train_refuses_without_explicit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("M2_SMALL_TRAIN", raising=False)
    assert train_is_authorized() is False
    with pytest.raises(RuntimeError, match="M2_SMALL_TRAIN"):
        run_train(cell=cfg.CELL_S1)


def test_stage0_writes_receipts_and_is_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "RESULT_ROOT", tmp_path)
    monkeypatch.setattr(cfg, "ACTIVE_RUN_ROOT", tmp_path)
    report = run_stage0(root=tmp_path)
    assert report["status"] == "READY"
    assert report["decoder_params"] == 3_543_010
    assert report["temporal_params"] == 2_108_416
    assert report["frontend_plus_readout"] == 1_434_594
    assert (tmp_path / "historical_trajectory_summary.json").is_file()
    assert (tmp_path / "historical_trajectory_summary.csv").is_file()
    assert (tmp_path / "stage0.json").is_file()
    assert (tmp_path / "parent_ref.json").is_file()
    assert (tmp_path / "S0_SMALL_LEGACY" / "seed42").is_dir()
    assert (tmp_path / "S1_SMALL_COS" / "seed42").is_dir()
    gates = report["gates"]
    for key in (
        "authority_bind",
        "param_count",
        "permutation_causal",
        "init_and_dropout_pairing",
        "lr_ema",
        "raw_step_invariant",
        "interrupt_resume",
        "profile_hook_present",
    ):
        assert gates[key]["pass"] is True, key
    assert report["profile_hook"]["implemented"] is True
    assert report["profile_hook"]["launched_on_gpu"] is False
    assert report["primary_candidate"] == "S1-SMALL-COS/EMA"
    assert report["formal_training_launched"] is False
    # parent pointer lives in the NEW root only
    old_jobs = REPO / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/jobs.jsonl"
    assert old_jobs.is_file()
