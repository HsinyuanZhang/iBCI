#!/usr/bin/env python3
"""Submit the immutable H1 EP-FiLM/no-readout image."""
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

module.PAYLOAD_SHA256 = "523d3d2e55a8fd4620a3f0a94dea6a99ae6ff53cefe807c3ff809a30b1d2479d"
module.CHECKPOINT_SHA256 = "0f406a8e69fdb57cf6a5480149f04ab3500e7fad849d36db38042edbadb2cd06"
module.FILM_STATE_SHA256 = "b602c2e090f455cb6259fc76fb9a225bfdb9aa6bf40adcd5ccd10b9a0da13b37"
module.STATE_PATH = HERE / "artifacts/evalai_push_state.json"
module.METHOD_NAME = "H1 EP-FiLM M3 cached identity, no readout calibration"
module.METHOD_DESCRIPTION = (
    "Frozen all-source C1/M3 H1 decoder with the sealed 648-parameter early-pooling "
    "FiLM adapter. The harmful MAT7 output readout from submission 581866 is removed "
    "by an exact identity output map. The runtime uses 27 cached public three-trial "
    "calibration payloads, no hidden labels, no optimizer step, and no online update."
)


def main() -> None:
    image_tag = "h1-epfilm-c1:no-readout-v1-523d3d2e"
    candidate = module.load_candidate(image_tag)
    candidate["method_name"] = module.METHOD_NAME
    candidate["method_description"] = module.METHOD_DESCRIPTION
    loaded, client, image, phase, challenge, limits = module.preflight(candidate)
    print(json.dumps({"image_id": image.id, "quota": limits, "phase": phase["id"]}, sort_keys=True), flush=True)
    state = module.push_image(loaded, client, image)
    completed = module.register(loaded, state)
    print(json.dumps({"submission_id": completed["submission_id"], "image_id": image.id}, sort_keys=True))


if __name__ == "__main__":
    main()
