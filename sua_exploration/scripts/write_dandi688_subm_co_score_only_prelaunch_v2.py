#!/usr/bin/env python3
"""Write the append-only, deliberately non-authorizing v2 prelaunch bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))

from mc_maze.subm_co_score_only_v2 import ScoreOnlyContractError, write_prelaunch_artifacts_v2  # noqa: E402


DEFAULT_OUTPUT = ROOT / "sua_exploration/results/dandi_000688_subm_co_score_only_prelaunch_v2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = write_prelaunch_artifacts_v2(args.output_dir, args.repo_root)
    except ScoreOnlyContractError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
