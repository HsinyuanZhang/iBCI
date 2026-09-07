from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "sua_exploration" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from t4_encoder_int8_protocol import validate_selection


SELECTION = (
    ROOT
    / "sua_exploration"
    / "manifests"
    / "sua_t4_final_architecture_selection_v1.json"
)


def test_real_final_selection_receipt_is_complete_and_hash_bound() -> None:
    selection, _source, deltas = validate_selection(SELECTION)
    assert selection["activity_calibration_n"] == 30
    assert selection["t4_label_feature_pool_n"] == 50
    assert selection["evaluation_start_trial"] == 50
    assert deltas["t4_minus_b0"] > 0.03
    assert deltas["t4_minus_ts4"] > 0.03


def test_selection_rejects_candidate_hash_drift(tmp_path: Path) -> None:
    payload = json.loads(SELECTION.read_text())
    payload["candidate_dispositions"][0]["sha256"] = "0" * 64
    bad = tmp_path / "bad_selection.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="candidate disposition 0 SHA-256 drifted"):
        validate_selection(bad)
