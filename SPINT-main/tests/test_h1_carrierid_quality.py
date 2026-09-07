from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.data.h1_carrierid_quality import (
    confidence_from_m4_fit,
    leave_one_date_out_channel_prior,
    source_scalar_feature_normalizer,
    split_forward_transfer_quality,
)
from src.data.h1_m4_eb_pilot import PilotDataError, TrialBlocks


ROOT = Path(__file__).resolve().parents[1]


class _Record:
    def __init__(self) -> None:
        generator = np.random.RandomState(17)
        self.num_neurons = 3
        self._blocks = {
            float(index): TrialBlocks(
                trial_number=float(index),
                rates=generator.normal(size=(4, self.num_neurons)),
                velocity=generator.normal(size=(4, 7)),
                block_indices=np.arange(20, dtype=np.int64).reshape(4, 5) + (index * 100),
            )
            for index in range(1, 5)
        }

    def blocks_for(self, value: float) -> TrialBlocks:
        return self._blocks[float(value)]


def _plan() -> SimpleNamespace:
    generator = np.random.RandomState(23)
    return SimpleNamespace(
        q=2,
        ridge_lambda=1e-2,
        mean=np.zeros(3, dtype=np.float64),
        pcs=generator.normal(size=(2, 3)),
        scale=np.array([0.7, 1.1, 0.9], dtype=np.float64),
        U=generator.normal(size=(7, 4)),
        mu=np.zeros(4, dtype=np.float64),
    )


def test_split_quality_is_per_channel_support_only_and_finite():
    feature = split_forward_transfer_quality(_Record(), _plan(), (1.0, 2.0, 3.0, 4.0))
    assert feature["fit_trial_values"].tolist() == [1.0, 2.0]
    assert feature["score_trial_values"].tolist() == [3.0, 4.0]
    assert feature["quality"].shape == (3,)
    assert np.isfinite(feature["quality"]).all()
    assert np.all((feature["quality"] >= 0.0) & (feature["quality"] <= 1.0))


def test_split_quality_rejects_non_m4_support_layout():
    with pytest.raises(PilotDataError, match="exactly 4"):
        split_forward_transfer_quality(_Record(), _plan(), (1.0, 2.0, 3.0))


def test_confidence_and_source_scalar_normalizer_fail_closed_on_bad_values():
    assert np.array_equal(confidence_from_m4_fit({"weight": np.array([0.0, 0.5, 1.0])}), np.array([0.0, 0.5, 1.0]))
    with pytest.raises(PilotDataError, match="escaped"):
        confidence_from_m4_fit({"weight": np.array([1.1])})
    with pytest.raises(PilotDataError, match="degenerate"):
        source_scalar_feature_normalizer(np.ones((2, 3)))


def test_source_date_prior_is_lodo_and_has_channel_resolution():
    quality = {
        "a": np.array([0.1, 0.4, 0.9]),
        "b": np.array([0.2, 0.3, 0.8]),
        "c": np.array([0.15, 0.45, 0.85]),
        "d": np.array([0.25, 0.35, 0.75]),
    }
    dates = {"a": "d1", "b": "d1", "c": "d2", "d": "d2"}
    result = leave_one_date_out_channel_prior(quality, dates)
    assert result["source_date_count"] == 2
    assert result["source_session_count"] == 4
    assert result["channels"] == 3
    assert result["per_channel_all_source_prior"].shape == (3,)
    assert set(result["per_date"]) == {"d1", "d2"}


def test_oracle_script_is_explicitly_leakage_only_and_requires_opt_in():
    from scripts.h1_carrierid_oracle_carrier_diagnostic import _query_trial_values, run

    script = (ROOT / "scripts" / "h1_carrierid_oracle_carrier_diagnostic.py").read_text(encoding="utf-8")
    assert "LEAKAGE_DIAGNOSTIC_ONLY" in script
    assert "--leakage-diagnostic-only" in script
    assert "model_selection\": False" in script
    assert "paper_main_result\": False" in script
    assert "same_sealed_h_c_checkpoint_for_support_and_oracle" in script
    record = SimpleNamespace(session_name="synthetic", trial_values=tuple(float(i) for i in range(1, 10)))
    assert _query_trial_values(record) == (5.0, 6.0, 7.0, 8.0)
    # The opt-in guard is evaluated before any source/checkpoint/target path.
    with pytest.raises(PermissionError, match="leakage-diagnostic-only"):
        run(
            data_dir=ROOT / "missing_data",
            raw_receipt=ROOT / "missing_raw.json",
            eb_receipt=ROOT / "missing_eb.json",
            shared_cache_dir=ROOT / "missing_cache",
            full_checkpoint=ROOT / "missing.ckpt",
            full_config=ROOT / "missing_config.yaml",
            sealed_terminal_receipt=ROOT / "missing_terminal.json",
            output=ROOT / "missing_output.json",
            device="cpu",
            leakage_diagnostic_only=False,
        )


def test_quality_preflight_stays_source_only_and_declares_kill_rules():
    script = (ROOT / "scripts" / "h1_carrierid_quality_preflight.py").read_text(encoding="utf-8")
    assert "load_target_records" not in script
    assert "cuda_constructed_or_launched\": False" in script
    assert "frozen_kill_rules" in script
    assert "no_gpu_authorized_by_this_receipt" in script
