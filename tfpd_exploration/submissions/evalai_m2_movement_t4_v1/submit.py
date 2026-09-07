#!/usr/bin/env python3
"""Submit the immutable M2 movement-window T4 cached-identity image."""
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

module.CANDIDATE = {
    "arm": "movement_t4_bins5_30_m33",
    "image_tag": "spint-t4-m2:movement-t4-m33-f3e64950",
    "image_id": "sha256:374426038abfc17785148b17b39b081b09df354c5f44802b756811f8b9c8300e",
    "payload_sha256": "f3e64950b00193949f6993d1a9022ba98fb65b759198427d666483d0c6bc49c1",
    "method_label": "M2 movement-window T4 cached identity (M33, bins 5:30, no FiLM, no TTA)",
    "method_name": "M2 movement-window T4 M33 cached identity",
    "method_description": (
        "Frozen SPINT M2 decoder with per-session identity calibrated offline from the "
        "chronological first 33 public trials. T4 is estimated from neural activity 100-600 ms "
        "after each trial boundary using the same sparse target-direction labels; its normalizer "
        "is fit on seven held-in sessions only. No dense behavior is used to build the deployment "
        "carrier, and runtime uses cached identities with no TTA, optimizer step, or backpropagation."
    ),
    "budget_disclosure": "33 public calibration trials for activity and sparse-direction T4; no online updates",
}
module.state_path = lambda: HERE / "artifacts/evalai_push_state.json"


def main() -> None:
    candidate, client, image, phase, limits, runtime = module.preflight()
    print(json.dumps({"image_id": image.id, "quota": limits, "phase": phase["id"]}, sort_keys=True), flush=True)
    state = module._push_image(client, image, runtime)
    completed = module._register(state, runtime)
    print(json.dumps({"submission_id": completed["submission_id"], "image_id": image.id}, sort_keys=True))


if __name__ == "__main__":
    main()
