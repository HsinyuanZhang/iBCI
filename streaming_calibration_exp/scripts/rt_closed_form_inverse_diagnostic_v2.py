#!/usr/bin/env python3
"""Append-only retry of the CPU-only RT closed-form inverse diagnostic.

v1's sealed preflight remains immutable.  Its execution stopped before writing
any per-session result because the intentionally zero-W sanity control has a
singular normal equation.  This retry binds the corrected analytic zero limit
(``v_hat=0``) while reusing the same no-GPU, no-formal-scope implementation.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "streaming_calibration_exp/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import rt_closed_form_inverse_diagnostic as base  # noqa: E402


RESULT_DIR = ROOT / "sua_exploration/results/rt_closed_form_inverse_diagnostic_v2"
PREFLIGHT = RESULT_DIR / "RT_CLOSED_FORM_INVERSE_CPU_PREFLIGHT_v2.json"
FINAL = RESULT_DIR / "RT_CLOSED_FORM_INVERSE_DIAGNOSTIC_v2.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bind() -> None:
    # The shared implementation is explicitly named/hashes in its own receipt;
    # no v1 receipt is reused or overwritten.
    base.RESULT_DIR = RESULT_DIR
    base.PREFLIGHT = PREFLIGHT
    base.FINAL = FINAL


def prepare() -> None:
    _bind()
    base.prepare()


def execute() -> None:
    _bind()
    base.execute()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.execute:
        parser.error("choose exactly one of --prepare or --execute")
    prepare() if args.prepare else execute()


if __name__ == "__main__":
    main()
