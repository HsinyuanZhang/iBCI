#!/usr/bin/env python3
"""Submit the immutable M2 MOVE-T4/EMPTY epoch-picked candidate."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "evalai_m2_means_squeeze_v2/submit_evalai_means_squeeze.py"
spec = importlib.util.spec_from_file_location("m2_push_base", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

IMAGE_ID = "sha256:8de56c58939ebd8306954ea7d180aceb7269fd3df28192f11df0dcaa7b60dc7f"
PAYLOAD_SHA256 = "f2f8cd4c046a5880e9716d61981cee4aa0652d33e411212fa7be217cef05b051"
module.CANDIDATE = {
    "arm": "movement_t4_m33_empty_adapter_seed44_epoch08_epochpick",
    "image_tag": "spint-t4-m2:movement-t4-empty-s44-e08-f2f8cd4c",
    "image_id": IMAGE_ID,
    "payload_sha256": PAYLOAD_SHA256,
    "method_label": "M2 MOVE-T4 plus profile-free adapter, visible-held-out epoch pick",
    "method_name": "M2 MOVE-T4 profile-free adapter s44/e08",
    "method_description": (
        "Frozen SPINT M2 decoder with first-33 activity and a 100-600 ms movement-window "
        "T4 carrier. The profile-free zero-initialized adapter checkpoint is seed 44 epoch 8, "
        "selected before EvalAI on six locally visible held-out sessions from a fixed 3-seed by "
        "12-epoch grid. Deployment uses cached identities and performs no online update."
    ),
    "budget_disclosure": (
        "33 public calibration trials for activity and sparse-direction T4; no dense behavior "
        "at deployment, profile input fixed to zero, no test-time updates"
    ),
}
module.state_path = lambda: HERE / "artifacts/evalai_push_state.json"


def main() -> None:
    candidate, client, image, phase, limits, runtime = module.preflight()
    print(
        json.dumps(
            {
                "candidate": candidate["arm"],
                "image_id": image.id,
                "payload_sha256": candidate["payload_sha256"],
                "phase": phase["id"],
                "quota": limits,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    state = module._push_image(client, image, runtime)
    completed = module._register(state, runtime)
    print(json.dumps({"submission_id": completed["submission_id"], "image_id": image.id}, sort_keys=True))


if __name__ == "__main__":
    main()
