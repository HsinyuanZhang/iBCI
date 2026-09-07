#!/usr/bin/env python3
"""Submit the immutable H1 C2 held-out-selected epoch-15 image."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "h1_series_20260830/scripts/push_h1_epfilm_evalai_v1.py"
spec = importlib.util.spec_from_file_location("h1_push_base", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

module.PAYLOAD_SHA256 = "91ef13cc94b9ab865c8f926dcbb6d33e9628bf9cbc9b757665d10e33cfda144a"
module.CHECKPOINT_SHA256 = "ce46267eb220142b8ef1f2e5acf05194650796ac2752594995b40d2ad0950215"
module.FILM_STATE_SHA256 = "107ad4dc17cc7965c33712377b756ee72ab0b6f74b2ae216ae2d70732437fe61"
module.STATE_PATH = HERE / "artifacts/evalai_push_state.json"
module.METHOD_NAME = "H1 C2 M3-aware held-out-selected epoch 15"
module.METHOD_DESCRIPTION = (
    "C2 H1 decoder trained with deterministic M7/M5/M4/M3 prefix cycling and "
    "selected at epoch 15 using the visible held-out calibration/development surface. "
    "Deployment uses the public first three calibration trials, native early-pooled "
    "activity plus carrier identity, an exact identity output map, and no online update."
)


def main() -> None:
    image_tag = "h1-c2-ho:epoch15-91ef13cc"
    candidate = module.load_candidate(image_tag)
    candidate["method_name"] = module.METHOD_NAME
    candidate["method_description"] = module.METHOD_DESCRIPTION
    loaded, client, image, phase, challenge, limits = module.preflight(candidate)
    print(
        json.dumps(
            {"image_id": image.id, "quota": limits, "phase": phase["id"]},
            sort_keys=True,
        ),
        flush=True,
    )
    state = module.push_image(loaded, client, image)
    completed = module.register(loaded, state)
    print(json.dumps({"submission_id": completed["submission_id"], "image_id": image.id}, sort_keys=True))


if __name__ == "__main__":
    main()
