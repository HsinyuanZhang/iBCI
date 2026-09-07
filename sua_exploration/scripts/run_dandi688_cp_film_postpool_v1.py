#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"status": "INERT", "formal_test_files_opened": False}, sort_keys=True))
        return
    from mc_maze.dandi688_cp_film_postpool_v1.runner import execute
    print(execute(args.repo_root, args.seed))


if __name__ == "__main__":
    main()

