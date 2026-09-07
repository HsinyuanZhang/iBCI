from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tfpd_exploration/src"))

from budget_matched_posterior_cal_aug_v1 import plan
from budget_matched_posterior_cal_aug_v1 import source_audit as audit


def _session(index: int, *, bad_m4: bool = False):
    theta = np.resize(np.asarray([0.0, np.pi / 2, np.pi, -np.pi / 2]), 30)
    if bad_m4:
        theta[:4] = 0.0
    design = np.stack((np.ones(30), np.cos(theta), np.sin(theta)), axis=1)
    beta = np.asarray([[2.0, 1.0, -0.5], [1.0, -0.2, 0.8]])
    rates = beta @ design.T
    rates += (index + 1) * np.linspace(-0.01, 0.01, 30)[None, :]
    ids = tuple(f"s{index}-trial-{position}" for position in range(30))
    return f"session-{index:02d}", rates, theta, ids, {"relative": f"source-{index}.nwb"}


def test_clean_27_source_rows_build_prior_normalizer_and_q() -> None:
    result = audit.build_products([_session(index) for index in range(27)])
    assert result["status"] == "SOURCE_POSTERIOR_AUTHORITY_PASSED"
    assert result["rank_failures"] == []
    assert result["source_prior"]["source_session_count"] == 27
    assert result["source_prior"]["shared_unchanged_across_budgets"] == [30, 10, 4]
    assert result["posterior_normalizer"]["equal_budget_weight"] is True
    assert result["posterior_normalizer"]["per_budget_row_count"] == {
        "4": 54, "10": 54, "30": 54,
    }
    assert len(result["posterior_rows"]) == 27


def test_one_bad_m4_prefix_yields_typed_stop_without_fallback() -> None:
    rows = [_session(index, bad_m4=(index == 3)) for index in range(27)]
    result = audit.build_products(rows)
    assert result["status"] == plan.M4_FAILURE
    assert result["source_prior"] is None
    assert result["posterior_normalizer"] is None
    assert any(row["session"] == "session-03" and row["budget"] == 4 for row in result["rank_failures"])


def test_dry_plan_is_inert_and_parks_gpu() -> None:
    payload = audit.dry_plan()
    assert payload["status"] == "DRY_NO_DATA_NO_GPU_NO_WRITE"
    assert payload["gpu_smoke_authorized"] is False
    assert payload["budgets"] == [4, 10, 30]
    assert payload["strict_source_session_count"] == 27


def test_predecessor_literals_are_completed_clean_pair() -> None:
    assert audit.PREDECESSORS["t0"]["terminal_sha256"].startswith("3071d907")
    assert audit.PREDECESSORS["c1"]["terminal_sha256"].startswith("320e2b99")
    assert audit.PREDECESSORS["t0"]["swa_sha256"].startswith("b4781d71")
    assert audit.PREDECESSORS["c1"]["swa_sha256"].startswith("5cc24676")


def test_attempt_json_encoding_is_finite_canonical() -> None:
    payload = {"z": 1.0, "a": [4, 10, 30]}
    body = audit._json_bytes(payload)
    assert body == b'{"a":[4,10,30],"z":1.0}\n'
    assert json.loads(body) == payload
