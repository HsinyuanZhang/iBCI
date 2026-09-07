#!/usr/bin/env python3
"""Inert public descriptor for the deferred PACD mixed-lineage scorer."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paired_anchored_calibration_dropout_score_v2_mixed_lineage import smoke


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parsed = parser.parse_args(argv)
    if parsed.execute:
        print("PACD score V2 has no mintable live mixed producer binding", file=sys.stderr)
        return 2
    print(json.dumps(smoke.dry_payload(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
