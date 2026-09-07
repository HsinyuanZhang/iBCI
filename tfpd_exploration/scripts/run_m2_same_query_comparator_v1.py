#!/usr/bin/env python3
"""Dry by default; explicitly execute the frozen M2 comparator matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
# A small set of historical SUA modules still use the established top-level
# ``mc_maze`` namespace internally.  Bind both roots here so execution does not
# depend on the caller's shell PYTHONPATH.  Insert in reverse to leave the
# repository root first for all qualified imports.
for _import_root in (SUA_ROOT, REPO_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if not args.execute:
        from tfpd_exploration.src.m2_same_query_comparator_v1.core import CELL_SPECS

        print(json.dumps({
            "schema": "m2_same_query_comparator_v1_dry",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE_NO_LAUNCH",
            "surfaces": ["within_post30", "external_official_query"],
            "cells": [cell.name for cell in CELL_SPECS],
            "gpu_policy": "GPU1_ONLY_AFTER_CURRENT_PIPELINE_TERMINAL",
        }, sort_keys=True))
        return
    from tfpd_exploration.src.m2_same_query_comparator_v1.physical import execute

    payload = execute(REPO_ROOT, gpu_index=args.gpu_index, batch_size=args.batch_size)
    print(json.dumps({
        "status": payload["status"],
        "row_count": payload["row_count"],
        "summaries": payload["summaries"],
        "paired_contrasts": payload["paired_contrasts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
