#!/usr/bin/env python3
"""Fail-closed CLI validator required before any SSC-T4 student launch."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STREAMING_ROOT = ROOT / "streaming_calibration_exp"
if str(STREAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMING_ROOT))

from src.utils.clean_teacher_validation import validate_clean_teacher_receipt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    facts = validate_clean_teacher_receipt(args.checkpoint, args.receipt)
    print(json.dumps({
        "validated": True,
        "checkpoint": facts["checkpoint_path"],
        "checkpoint_sha256": facts["checkpoint_sha256"],
        "receipt": facts["receipt_path"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
