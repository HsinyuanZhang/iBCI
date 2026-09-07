#!/usr/bin/env python3
"""Record the H1 EP-FILM EvalAI submission receipt from the push state."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_calibration_profile_film_v3"
STATE_PATH = (
    REPO_ROOT / "tfpd_exploration/h1_series_20260830/artifacts/h1_epfilm_evalai_v1/evalai_push_state.json"
)
CHALLENGE_ID = 2319
PHASE_ID = 4599
TEAM_ID = 41975


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if not state.get("registered") or "submission_id" not in state:
        raise RuntimeError("push state has no registered submission")
    submission_id = int(state["submission_id"])
    status = subprocess.check_output(
        ["evalai", "submission", str(submission_id)], text=True,
    ).strip()
    receipt = {
        "schema": "h1_calibration_profile_film_v3_submission",
        "status": "SUBMITTED_EP_FILM_EVALAI_PRIVATE_ONE_SHOT",
        "challenge_id": CHALLENGE_ID,
        "phase_id": PHASE_ID,
        "participant_team": TEAM_ID,
        "submission_id": submission_id,
        "visibility": "private",
        "image_tag": state["image_tag"],
        "image_id": state["image_id"],
        "payload_sha256": state["payload_sha256"],
        "submitted_image_uri": state["submitted_image_uri"],
        "uploaded_manifest_digest": state["uploaded_manifest_digest"],
        "uuid_tag": state["uuid_tag"],
        "method_name": "H1 EP-FILM cached identity (all-source, sensitivity)",
        "submission_metadata": {
            "IsHeldOutZeroShot": False,
            "IsTestTimeAdaptive": False,
            "IsPretrained": False,
        },
        "cli_submission_status": status,
        "server_response": state.get("server_response"),
        "hidden_test_opened": False,
        "official_score_claimed": False,
    }
    if not args.execute:
        print(json.dumps({k: receipt[k] for k in ("submission_id", "image_id", "cli_submission_status")}, sort_keys=True))
        return 0
    import sys

    sys.path.insert(0, str(REPO_ROOT / "SPINT-main"))
    sys.path.insert(0, "/tmp/ibci-h1/deployment-v1/SPINT-main")
    from src.h1_m4_cce_contract import write_immutable_json

    _, digest = write_immutable_json(RESULT_ROOT / "submission_receipt.json", receipt)
    side = RESULT_ROOT / "submission_receipt.json.sha256"
    with side.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  submission_receipt.json\n")
    os.chmod(side, 0o444)
    print(json.dumps({"submission_receipt_sha256": digest, "submission_id": submission_id}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
