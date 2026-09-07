"""Static integrity check for the explicit no-implementation decision receipt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "sua_exploration/results/m2_m24_k4_t4_residual_stop_v1/stop_design_receipt.json"


def test_stop_receipt_binds_frozen_inputs_and_forbids_shortcuts() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["decision"] == "STOP_BEFORE_IMPLEMENTATION"
    assert "q=0" in payload["exact_q0_contract"]["required"]
    forbidden = " ".join(payload["forbidden_downgrades"])
    assert "side_dim=8" in forbidden and "full encoder" in forbidden and "decoder-key" in forbidden
    for record in payload["frozen_inputs"].values():
        path = Path(record["path"])
        assert path.is_file()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record["sha256"]
