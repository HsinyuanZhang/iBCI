#!/usr/bin/env python3
"""Guarded EvalAI helper for the profile-free MOVE-T4 adapter candidate."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "evalai_m2_means_squeeze_v2/submit_evalai_means_squeeze.py"
spec = importlib.util.spec_from_file_location("m2_push_base", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

IMAGE_ID = "sha256:caa98aefff18364df18bf2b5d4f143b9cb82beb8aaeeec1395faec134055752d"
PAYLOAD_SHA256 = "4e4dae8f7239582a26d44cdb449e674710f28223523dd691dd4f8758b05220e0"
module.CANDIDATE = {
    "arm": "movement_t4_m33_empty_adapter_seed42",
    "image_tag": "spint-t4-m2:movement-t4-empty-s42-4e4dae8f",
    "image_id": IMAGE_ID,
    "payload_sha256": PAYLOAD_SHA256,
    "method_label": "M2 movement-window T4 plus profile-free adapter (M33, bins 5:30, no TTA)",
    "method_name": "M2 MOVE-T4 plus profile-free M33 adapter",
    "method_description": (
        "Frozen SPINT M2 decoder using first-33 activity and a movement-window T4 carrier "
        "estimated from neural bins 100-600 ms after trial onset with the same sparse target "
        "directions. A 1224-parameter zero-initialized T4-conditioned adapter was trained for "
        "12 epochs on seven held-in sessions with its four profile inputs fixed to positive zero. "
        "The deployed payload contains cached identities only and performs no TTA or online update."
    ),
    "budget_disclosure": (
        "33 public calibration trials for activity and sparse-direction T4; profile input is zero; "
        "no dense behavior at deployment and no online updates"
    ),
}
module.state_path = lambda: HERE / "artifacts/evalai_push_state.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-image-id", default="")
    parser.add_argument("--confirm-payload-sha256", default="")
    args = parser.parse_args()
    candidate, client, image, phase, limits, runtime = module.preflight()
    report = {
        "mode": "execute" if args.execute else "read_only_preflight",
        "candidate": candidate["arm"],
        "image_id": image.id,
        "payload_sha256": candidate["payload_sha256"],
        "phase": phase["id"],
        "quota": limits,
        "state_path": str(module.state_path()),
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return
    if args.confirm_image_id != IMAGE_ID or args.confirm_payload_sha256 != PAYLOAD_SHA256:
        raise RuntimeError("execute requires the complete immutable image ID and payload SHA-256")
    state = module._push_image(client, image, runtime)
    completed = module._register(state, runtime)
    print(json.dumps({"submission_id": completed["submission_id"], "image_id": image.id}, sort_keys=True))


if __name__ == "__main__":
    main()
