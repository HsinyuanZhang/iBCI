from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.falcon_emg_afc4_datamodule import M1EMGAFC4DataModule  # noqa: E402


def contract(**changes):
    values = dict(
        task="m1", validation_protocol="loso", calibration_n_trials=10,
        random_calibration=False, include_heldout_in_fit=False, include_heldout_in_test=False,
        query_start_trial=0, heldin_query_start_trial=10, heldin_query_end_trial=210,
    )
    values.update(changes)
    instance = object.__new__(M1EMGAFC4DataModule)
    instance.__dict__["_hparams"] = SimpleNamespace(**values)
    return instance


def test_contract_accepts_only_strict_m1_source_loso_m10() -> None:
    contract()._assert_contract()
    with pytest.raises(ValueError, match="M1"):
        contract(task="m2")._assert_contract()
    with pytest.raises(ValueError, match="M10"):
        contract(calibration_n_trials=24)._assert_contract()
    with pytest.raises(ValueError, match="strict held-in query start"):
        contract(heldin_query_start_trial=0)._assert_contract()
    with pytest.raises(ValueError, match="held-out"):
        contract(include_heldout_in_test=True)._assert_contract()
