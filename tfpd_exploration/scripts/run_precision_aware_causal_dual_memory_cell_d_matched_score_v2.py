#!/usr/bin/env python3
"""Public dry-only CLI for the Precision-Aware CDM-D matched-score V2 route."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap() -> Path:
    root = Path(__file__).resolve().parents[2]
    exploration = root / "tfpd_exploration"
    source = exploration / "src"
    for item in (str(source), str(exploration)):
        if item not in sys.path:
            sys.path.insert(0, item)
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Precision-Aware CDM-D V2 environment-successor dry contract")
    parser.add_argument("--dry-run", action="store_true", help="print static contract; this is the only public mode")
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("only --dry-run is public; authority and physical execution require reviewed in-process APIs")
    _bootstrap()
    from src.precision_aware_causal_dual_memory_cell_d_score_v2 import plan

    print(json.dumps(plan.dry_plan(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
