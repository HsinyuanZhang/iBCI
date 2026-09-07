"""Pick the H1 stage-2 L-winner from sealed cell receipts. CPU only."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from btransform_unified_v1 import receipts  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l250", type=Path, required=True)
    parser.add_argument("--l200", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()

    rows = {}
    for label, root in (("L250", args.l250), ("L200", args.l200)):
        receipt_path = root / "cell_receipt.json"
        train_path = root / "train_receipt.json"
        if not receipt_path.is_file():
            rows[label] = {
                "status": "MISSING_RECEIPT",
                "root": str(root),
                "train_status": (
                    json.loads(train_path.read_text(encoding="utf-8")).get("status")
                    if train_path.is_file()
                    else "NO_TRAIN_RECEIPT"
                ),
            }
            continue
        receipt = _load(receipt_path)
        legal = receipt["legal_checkpoint"]
        rows[label] = {
            "status": "COMPLETED",
            "root": str(root),
            "window": receipt["window"],
            "minival13_ema_equal_session_mean": legal["minival13_ema_equal_session_mean"],
            "minival13_ema_pooled": legal["minival13_ema_pooled"],
            "ckpt": legal["path"],
        }

    complete = {k: v for k, v in rows.items() if v.get("status") == "COMPLETED"}
    if not complete:
        winner = None
        rule = "no completed L arm"
    elif len(complete) == 1:
        winner = next(iter(complete))
        rule = "only completed arm"
    else:
        s250 = float(complete["L250"]["minival13_ema_equal_session_mean"])
        s200 = float(complete["L200"]["minival13_ema_equal_session_mean"])
        if abs(s250 - s200) <= 1e-10:
            winner = "L250"
            rule = "tie -> L=250 (stage-1 lock)"
        elif s200 > s250:
            winner = "L200"
            rule = "higher all-13 minival EMA equal_session_mean at endpoint24"
        else:
            winner = "L250"
            rule = "higher all-13 minival EMA equal_session_mean at endpoint24"

    payload = {
        "schema": "btransform_unified_v1_h1_stage2_winner",
        "utc": datetime.now(timezone.utc).isoformat(),
        "statistic": "all-13 source-minival EMA equal_session_mean at endpoint24",
        "checkpoint_rule": "endpoint24 EMA",
        "tie_break": "L=250",
        "rule": rule,
        "winner": winner,
        "arms": rows,
        "evalai": {
            "register": False,
            "hold_reason": "P32 in flight; user must authorize official H1 push",
            "runtime": "exact-E (frontend window cache + last-layer last-Q)",
        },
    }
    args.dest.mkdir(parents=True, exist_ok=True)
    receipts.seal_json(args.dest / "winner.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if winner else 1


if __name__ == "__main__":
    raise SystemExit(main())
