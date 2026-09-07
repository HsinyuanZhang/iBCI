from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tfpd_exploration/src"))

from budget_matched_posterior_cal_aug_v1 import source_audit_v2 as v2  # noqa: E402
from budget_matched_posterior_cal_aug_v1 import source_audit_v3 as v3  # noqa: E402


def test_support_id_encodes_missing_direction_as_null_without_guessing() -> None:
    trial = {"start_time": 1.0, "stop_time": 2.0, "target_dir": None}
    digest = v2._support_id("session", 27, trial)
    expected = v2.v1._json_sha({
        "session": "session", "position": 27, "start_time": 1.0,
        "stop_time": 2.0, "target_dir": None,
    })
    assert digest == expected
    assert digest != v2.v1._json_sha({
        "session": "session", "position": 27, "start_time": 1.0,
        "stop_time": 2.0, "target_dir": 0.0,
    })


def test_v2_failure_literals_and_dry_plan_are_exact() -> None:
    assert v3.V2_ATTEMPT_SHA256.startswith("48366fde")
    assert v3.V2_FAILURE_SHA256.startswith("b8e665c3")
    assert v3.V2_CLOSURE_SHA256.startswith("dc6aef27")
    payload = v3.dry_plan()
    assert payload["status"] == "DRY_NO_DATA_NO_GPU_NO_WRITE"
    assert payload["support_id_missing_direction_encoding"].endswith("no_imputation")

