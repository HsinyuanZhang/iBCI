"""Seal the pre-registered +8-epoch upgrade trigger for M-F250@1e-4. CPU only."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from btransform_unified_v1 import receipts

F250 = PACKAGE_ROOT / "results/h1_matrix/M_F250_lr1e4_20260906T094633Z"
LAST_K = 4
RANGE_TOL = 0.01
PICK_LATE_FROM = 22
EXTRA_EPOCHS = 8


def last_k_range(series: dict[str, float], k: int = LAST_K) -> dict:
    epochs = sorted(int(e) for e in series)
    tail_e = epochs[-k:]
    vals = [float(series[str(e)]) for e in tail_e]
    return {
        "epochs": tail_e,
        "values": vals,
        "range": float(max(vals) - min(vals)),
        "strictly_increasing": all(vals[i + 1] > vals[i] for i in range(len(vals) - 1)),
        "plateau": float(max(vals) - min(vals)) <= RANGE_TOL,
    }


def main() -> int:
    receipt = json.loads((F250 / "cell_receipt.json").read_text(encoding="utf-8"))
    series = receipt["epoch_curves"]["exam_ema_equal_mean_SEL2_series"]
    pick = int(receipt["sel"]["SEL-2_epoch_pick"])
    tail = last_k_range(series)
    no_plateau = not tail["plateau"]
    late_pick = pick >= PICK_LATE_FROM
    fire = bool(no_plateau or late_pick)
    payload = {
        "schema": "btransform_unified_v1_h1_extend8_trigger",
        "utc": datetime.now(timezone.utc).isoformat(),
        "rule": (
            "pre-registered upgrade: if the e24 SEL-2 curve has no last-4 plateau "
            f"(range > {RANGE_TOL}) OR SEL-2 pick >= {PICK_LATE_FROM}, continue 8 "
            "epochs at the cosine floor (1e-5). Historical 24-ep root stays sealed."
        ),
        "source": str(F250),
        "sel2_pick_epoch": pick,
        "sel2_pick_exam_ema_equal_mean": receipt["sel"]["SEL-2_pick_exam_ema_equal_mean"],
        "last4_exam_ema": tail,
        "no_plateau": no_plateau,
        "late_pick": late_pick,
        "trigger_fires": fire,
        "extra_epochs": EXTRA_EPOCHS,
        "lr_during_extend": 1.0e-5,
        "inherit_to_stage2": (
            "same recipe (proj_add L=250/200, 1e-4, CAL-2/M3, seed 42): after each "
            "stage-2 24-ep arm completes, run the same +8; submit checkpoint becomes "
            "endpoint32 EMA. Stage-2 24-ep jobs already in flight are not killed."
        ),
    }
    dest = PACKAGE_ROOT / "results/h1_matrix/M_F250_lr1e4_extend8_trigger"
    dest.mkdir(parents=True, exist_ok=True)
    receipts.seal_json(dest / "trigger.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if fire else 1


if __name__ == "__main__":
    raise SystemExit(main())
