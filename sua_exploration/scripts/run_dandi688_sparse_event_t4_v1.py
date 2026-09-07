#!/usr/bin/env python3
"""Explicit CLI for the sparse-event route; it never starts work on import."""
from __future__ import annotations

import argparse
from pathlib import Path

from mc_maze.dandi688_sparse_event_t4_v1.stage0 import execute_stage0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("stage0", "stage0-supplement", "stage1", "stage2"))
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help="required explicit mutation/compute acknowledgement")
    parser.add_argument("--film-admitted", action="store_true")
    parser.add_argument("--estimator-admitted", action="store_true")
    parser.add_argument("--stage1-film-open", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required; this CLI otherwise performs no lifecycle mutation")
    if args.stage == "stage0":
        execute_stage0(args.repo_root, args.result_root)
        return 0
    if args.stage == "stage0-supplement":
        from mc_maze.dandi688_sparse_event_t4_v1.supplement import execute_stage0_supplement

        execute_stage0_supplement(args.repo_root, args.result_root)
        return 0
    parser.error("GPU launch is root-owned; use this route's Python executor only through the frozen launch admission")


if __name__ == "__main__":
    raise SystemExit(main())
