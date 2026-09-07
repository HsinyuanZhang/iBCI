#!/usr/bin/env python
"""GPU1 disposable 20-warmup + 100 paired updates. Refuses if gates failed."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main() -> int:
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from tfpd_exploration.src.two_mainlines_long_v1.calibration import constants as C
    from tfpd_exploration.src.two_mainlines_long_v1.calibration.profile import ProfileError, run_profile

    gate_path = C.OWNED_RESULT_ROOT / "gates" / "p_gates.json"
    if not gate_path.is_file():
        print("gates missing; refuse profile")
        return 2
    summary = json.loads(gate_path.read_text(encoding="utf-8"))
    try:
        receipt = run_profile(summary, device="cuda:0")
    except ProfileError as error:
        print(error)
        return 2
    print(receipt["status"], receipt["completed_steps"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
