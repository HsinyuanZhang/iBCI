#!/usr/bin/env python3
"""Pack stub for M1 concat D4 + BT-EORT. Refuses until pick is sealed.

Does not EvalAI-submit. After pick, port the M2 concat E-ORT recipe
(token cat local16∥E0∥T4; here E0 is 100-d, token_in=120) onto this dest.
Do not reuse P16/P32 proj_add ONNX graphs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PICK = Path(
    "/home/xinyuan/Work_host/SPINT/btransform_unified_v1/results/m1_concat_eort"
)


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    if root is None:
        print("usage: pack_m1_concat_eort_v1.py <train_root>", file=sys.stderr)
        return 2
    pick = root / "depth4" / "epoch_pick.json"
    if not pick.exists():
        print("REFUSED: epoch_pick.json missing; train/score/pick first", file=sys.stderr)
        return 2
    payload = json.loads(pick.read_text())
    if payload.get("register") is True or payload.get("evalai_opened") is True:
        print("REFUSED: pick already opened EvalAI", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "PACK_SKELETON_ONLY",
                "identity": "M1 concat D4 token_in=120; E-ORT after pick; not proj_add",
                "selected_epoch": payload.get("selected", {}).get("epoch"),
                "equal_session_mean": payload.get("selected", {}).get("equal_session_mean"),
                "next": "port tfpd_exploration/submissions/evalai_m2_small_concat_ort_v1 concat frontend + M1 E-ORT session recipe; register:false",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
