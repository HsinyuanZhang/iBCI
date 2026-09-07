#!/usr/bin/env python3
"""Non-authorizing V8 wrapper around the isolated production quarantine CLI."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sua_exploration.scripts.write_dandi688_subm_co_three_arm_score_only_prelaunch_v8 import (  # noqa: E402
    DEFAULT_OUTPUT,
    StaticV8Error,
    load_stored_blocked_prelaunch,
)


PRODUCTION_GUARD = ROOT / "sua_exploration/scripts/run_dandi688_subm_co_three_arm_score_only_v8_production.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "score"), default="dry-run")
    parser.add_argument("--prelaunch-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        load_stored_blocked_prelaunch(args.prelaunch_dir)
    except StaticV8Error as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr); return 2
    mode = "status" if args.mode == "dry-run" else "formal"
    completed = subprocess.run([sys.executable, "-I", str(PRODUCTION_GUARD), "--mode", mode], check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
