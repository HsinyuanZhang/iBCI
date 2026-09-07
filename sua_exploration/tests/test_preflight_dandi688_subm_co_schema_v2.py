"""Regression fixtures for the v2 score-blind NWB parser adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "sua_exploration/scripts/preflight_dandi688_subm_co_schema_v2.py"


def _module():
    spec = importlib.util.spec_from_file_location("subm_preflight_v2_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_materialized_one_row_electrode_dataframe_is_resolved_by_its_index():
    module = _module()
    # This is the exact representation returned by
    # nwb.units.to_dataframe()['electrodes'].iloc[i] in DANDI 000688.
    resolved = pd.DataFrame({"location": ["Primary Motor Cortex"]}, index=pd.Index([7], name="id"))
    assert module.resolve_one_dataframe_electrode_id(resolved, {3, 7, 11}) == 7
    assert module.resolve_one_dataframe_electrode_id(resolved, {3, 11}) is None
    multi = pd.DataFrame({"location": ["M1", "PMd"]}, index=pd.Index([7, 11], name="id"))
    assert module.resolve_one_dataframe_electrode_id(multi, {7, 11}) is None


def test_abort_nan_target_dir_is_diagnostic_but_rewarded_nan_is_a_gate_failure():
    module = _module()
    parsed_abort_nan = module._float_or_none(np.nan)
    assert parsed_abort_nan is None
    assert module.finite_target_required_for_usable_rewarded("A", parsed_abort_nan)
    assert not module.finite_target_required_for_usable_rewarded("R", parsed_abort_nan)
    assert module.finite_target_required_for_usable_rewarded("R", module._float_or_none(0.0))
