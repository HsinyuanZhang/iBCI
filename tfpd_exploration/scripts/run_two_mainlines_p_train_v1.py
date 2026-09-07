#!/usr/bin/env python
"""Formal 12-epoch pair. Refuses unless P1–P5 passed."""
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
    from tfpd_exploration.src.two_mainlines_long_v1.calibration.train_formal import (
        TrainError,
        refuse_if_gates_failed,
        write_blocked_slots,
    )

    gate_path = C.OWNED_RESULT_ROOT / "gates" / "p_gates.json"
    if not gate_path.is_file():
        write_blocked_slots("P1-P5 receipts missing")
        print("gates missing; refuse formal train")
        return 2
    summary = json.loads(gate_path.read_text(encoding="utf-8"))
    try:
        refuse_if_gates_failed(summary)
    except TrainError as error:
        write_blocked_slots(str(error))
        print(error)
        return 2
    print("gates passed; formal trainer is owned but not started from this refusal path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
