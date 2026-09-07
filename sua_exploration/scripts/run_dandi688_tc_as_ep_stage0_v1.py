#!/usr/bin/env python3
"""Run source-only TC-AS-EP selector/three-axis Stage 0 (no decoder metric/GPU)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "sua_exploration"))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print("INERT_READY_SOURCE_ONLY_NO_GPU_NO_DECODER_METRIC")
        return
    from sua_exploration.mc_maze.dandi688_tc_as_ep_v1.stage0 import (
        execute_selector_stage0,
    )

    result = execute_selector_stage0(args.repo_root, result_root=args.result_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
