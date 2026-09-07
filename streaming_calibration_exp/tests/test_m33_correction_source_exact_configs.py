"""Configuration-only contract for the M2/M33 correction replay."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "streaming_calibration_exp/configs"
SOURCES = {
    "m2_m33_disjoint_replay_correction_f0": ROOT / "streaming_calibration_exp/outputs/streaming_calibration/native_mua_t4_v1_f0_m2_f1_s42_20260729_140518/resolved_config.yaml",
    "m2_m33_disjoint_replay_correction_t4": ROOT / "streaming_calibration_exp/outputs/streaming_calibration/native_mua_t4_v1_t4_m2_f1_s42_20260729_142047/resolved_config.yaml",
    "m2_m33_disjoint_replay_correction_ts4": ROOT / "streaming_calibration_exp/outputs/streaming_calibration/native_mua_t4_v1_ts4_m2_f1_s42_20260729_143619/resolved_config.yaml",
}


@pytest.mark.parametrize("experiment", sorted(SOURCES))
def test_correction_model_compose_is_exactly_the_frozen_source(experiment: str) -> None:
    source_path = SOURCES[experiment]
    assert source_path.is_file(), f"frozen source artifact missing: {source_path}"
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))["model"]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG)):
        cfg = compose(config_name="train.yaml", overrides=[f"experiment={experiment}"])
    model = OmegaConf.to_container(cfg.model, resolve=False)
    assert model == source
    assert "teacher_receipt_path" not in model
    assert "require_clean_teacher_receipt" not in model


def test_correction_defaults_are_inert_until_the_runner_overrides_test_fields() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG)):
        cfg = compose(config_name="train.yaml", overrides=["experiment=m2_m33_disjoint_replay_correction_t4"])
    assert cfg.data.include_heldout_in_fit is False
    assert cfg.data.include_heldout_in_test is False
    assert cfg.data.query_start_trial == 0
    assert cfg.data.allow_empty_heldout_query is False
