from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.falcon_m1_all_source_datamodule import (  # noqa: E402
    M1_ALL_SOURCE_SESSIONS,
    M1AllSourceDataModule,
)
from src.data.falcon_m1_all_source_emg_afc4_datamodule import (  # noqa: E402
    M1AllSourceEMGAFC4DataModule,
)


def _hparams(**changes):
    values = dict(
        task="m1",
        calibration_n_trials=10,
        random_calibration=False,
        include_heldout_in_fit=False,
        include_heldout_in_test=False,
        query_start_trial=0,
        heldin_query_start_trial=0,
        heldin_query_end_trial=None,
        validation_protocol="all_source",
        smooth_calibration=False,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_all_source_session_allowlist_is_exact_and_ordered():
    assert M1_ALL_SOURCE_SESSIONS == (
        "ses-20120924",
        "ses-20120926",
        "ses-20120927",
        "ses-20120928",
    )
    with pytest.raises(ValueError, match="exactly"):
        M1AllSourceDataModule(
            task="m1", data_dir=".", source_session_names=["ses-20120924"],
            calibration_n_trials=10, random_calibration=False,
            include_heldout_in_fit=False, include_heldout_in_test=False,
            query_start_trial=0, heldin_query_start_trial=0,
            heldin_query_end_trial=None, side_feature_group="none",
        )


def test_all_source_contract_rejects_query_and_heldout():
    instance = object.__new__(M1AllSourceDataModule)
    instance._hparams = _hparams()
    instance._assert_fit_stage("fit")
    for changes, message in (
        ({"query_start_trial": 1}, "query"),
        ({"heldin_query_start_trial": 10}, "query"),
        ({"include_heldout_in_test": True}, "held-out"),
        ({"calibration_n_trials": 24}, "M10"),
    ):
        bad = object.__new__(M1AllSourceDataModule)
        bad._hparams = _hparams(**changes)
        with pytest.raises(ValueError, match=message):
            bad._assert_fit_stage("fit")


def test_afc4_contract_requires_all_source_and_no_local_query():
    instance = object.__new__(M1AllSourceEMGAFC4DataModule)
    instance._hparams = _hparams()
    instance._assert_afc4_contract("fit")
    for changes, message in (
        ({"validation_protocol": "loso"}, "all_source"),
        ({"heldin_query_start_trial": 10}, "query"),
        ({"smooth_calibration": True}, "unsmoothed"),
    ):
        bad = object.__new__(M1AllSourceEMGAFC4DataModule)
        bad._hparams = _hparams(**changes)
        with pytest.raises(ValueError, match=message):
            bad._assert_afc4_contract("fit")


def test_final_configs_bind_shared_teacher_and_fixed_budgets():
    configs = {
        "full": ROOT / "configs/experiment/m1_afc4_emg_full_all_source_final.yaml",
        "b4": ROOT / "configs/experiment/m1_afc4_emg_b4_all_source_final.yaml",
    }
    for arm, path in configs.items():
        text = path.read_text(encoding="utf-8")
        if arm == "full":
            assert "train: true" in text
            assert "test: false" in text
            assert "teacher_ckpt_path: M1_ALL_SOURCE_TEACHER_CHECKPOINT_REQUIRED" in text
            assert "max_epochs: 12" in text
            assert "loss_mode: task_only" in text
            assert "lambda_y: 0.0" in text
            assert "lambda_E: 0.0" in text
        else:
            assert "- m1_afc4_emg_full_all_source_final" in text
    assert "afc4_arm: full" in configs["full"].read_text(encoding="utf-8")
    assert "afc4_arm: b4" in configs["b4"].read_text(encoding="utf-8")


def test_packaging_runtime_and_docker_entrypoint_are_pinned():
    runtime = (ROOT.parent / "sua_exploration/evalai_m1_threeway/afc4_cached_identity_decoder.py").read_text(encoding="utf-8")
    docker = (ROOT.parent / "sua_exploration/evalai_m1_threeway/Dockerfile.afc4.cached").read_text(encoding="utf-8")
    exporter = (ROOT.parent / "sua_exploration/evalai_m1_threeway/export_afc4_payload.py").read_text(encoding="utf-8")
    assert "evalai_m1_afc4_cached_identity_v1" in runtime
    assert "decode_afc4_cached.py" in docker
    assert "target_query_values_read" in exporter
    assert "heldout" in exporter
