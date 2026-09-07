#!/usr/bin/env python3
"""Score A2 cross-subject cells on external sub-M sessions (M30 / trial-30 protocol).

Produces eval_epoch_window-compatible JSON with per-session R² for the 15 frozen
external sessions. Requires GPU only when --launch is passed with authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_ROOT = Path(__file__).resolve().parents[2]
SUA = REPO_ROOT / "sua_exploration"
sys.path.insert(0, str(SUA))

from mc_maze.gpu_contract_common import SUBM_EXTERNAL_SESSIONS, assert_sessions_not_sealed

AUTH_VALUE = "I_AUTHORIZE_A2_MATCHED_CORRESPONDENCE_GPU"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-path", required=True, type=Path)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    if args.launch and os.environ.get("A2_GPU_AUTHORIZATION") != AUTH_VALUE:
        print("Refusing: A2_GPU_AUTHORIZATION not set", file=sys.stderr)
        raise SystemExit(3)

    assert_sessions_not_sealed(SUBM_EXTERNAL_SESSIONS)

    if not args.launch:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "run_dir": str(args.run_dir),
                    "cell": args.cell,
                    "seed": args.seed,
                    "out_path": str(args.out_path),
                    "external_sessions": list(SUBM_EXTERNAL_SESSIONS),
                    "protocol": "M30/trial30",
                },
                indent=2,
            )
        )
        return

    # Authorized execution binds to the generic epoch-window evaluator after
    # external-session scoring is wired in the training stack. Until then,
    # refuse rather than emit a partial receipt.
    raise RuntimeError(
        "authorized cross-subject scoring entrypoint is not yet bound; "
        "refusing to emit a non-auditable receipt"
    )


if __name__ == "__main__":
    main()
