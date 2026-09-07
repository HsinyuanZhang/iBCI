#!/usr/bin/env python3
"""Inert AOF-S package plan; local build/validation requires explicit flags.

This driver never registers, pushes, or submits anything to EvalAI/ECR.  It
exists so the package has a guarded operator entry point without making a
network mutation possible by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="AOF-S local package plan (network disabled)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps({
        "status": "INERT_LOCAL_ONLY",
        "package": str(PACKAGE),
        "frozen_beta": -0.3799365677508742,
        "network_submission": False,
        "requires_separate_authorization": ["payload build", "local minival", "docker build", "any submission"],
        "dry_run": bool(args.dry_run),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
