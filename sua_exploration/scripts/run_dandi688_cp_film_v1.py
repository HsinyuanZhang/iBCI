#!/usr/bin/env python3
"""Launch the frozen DANDI 000688 CP-FiLM source-only screen."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/home/xinyuan/Work_host/SPINT"))
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print({"status": "INERT_READY", "seed": args.seed, "gpu": "physical GPU0", "formal_test": False})
        return
    from mc_maze.dandi688_cp_film_v1.runner import execute

    print(execute(args.repo_root, args.seed), flush=True)


if __name__ == "__main__":
    main()

