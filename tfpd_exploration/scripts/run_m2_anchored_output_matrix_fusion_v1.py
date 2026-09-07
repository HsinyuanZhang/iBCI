#!/usr/bin/env python3
"""Inert entry point for the AOF-M V1 source-only route."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfpd_exploration.src.m2_anchored_output_matrix_fusion_v1 import plan


parser = argparse.ArgumentParser()
parser.add_argument("--execute", action="store_true")
parser.add_argument("--repo-root", default="/home/xinyuan/Work_host/SPINT")
parser.add_argument("--reviewed-closure-sha256")
args = parser.parse_args()
if not args.execute:
    print(json.dumps({"schema": plan.SCHEMA, "status": "INERT__GPU0_SOURCE_ONLY_EXECUTE_REQUIRED"}))
else:
    from tfpd_exploration.src.m2_anchored_output_matrix_fusion_v1.driver import execute
    print(execute(Path(args.repo_root), reviewed_closure_sha256=args.reviewed_closure_sha256))
