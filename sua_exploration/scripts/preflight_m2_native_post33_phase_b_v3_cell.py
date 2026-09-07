#!/usr/bin/env python3
"""Claim one Phase-B v3 cell and emit fail-closed Hydra overrides.

This is a score-free preflight, not a training launcher.  It never imports a
scorer, initializes CUDA, opens a formal endpoint, or invokes EvalAI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sua_exploration.mc_maze.m2_native_post33_phase_b_v3 import (
    CellKey,
    PROTOCOL_ID,
    claim_cell_ownership,
    resolve_paired_teacher_from_receipt,
    write_status_exclusive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("spint", "t4"), required=True)
    parser.add_argument("--fold", type=int, choices=range(7), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--paired-spint-receipt", type=Path)
    return parser.parse_args()


def prepare(args: argparse.Namespace) -> dict[str, object]:
    key = CellKey(PROTOCOL_ID, args.arm, args.fold, args.seed)
    paired_receipt: Path | None = None
    if args.arm == "t4":
        if args.paired_spint_receipt is None:
            raise ValueError("T4 preflight requires --paired-spint-receipt")
        paired_receipt = args.paired_spint_receipt.resolve(strict=True)
        # Validate the exact pair now.  The T4 model resolves the same receipt
        # again at construction; no direct teacher path is emitted or accepted.
        resolve_paired_teacher_from_receipt(
            paired_receipt, fold=args.fold, seed=args.seed
        )
    elif args.paired_spint_receipt is not None:
        raise ValueError("SPINT preflight must not receive a paired teacher receipt")

    paths = claim_cell_ownership(
        args.cell_root, key, owner_token=args.owner_token
    )
    write_status_exclusive(
        args.cell_root,
        key,
        state="started",
        owner_token=args.owner_token,
        details={"scope": "score_free_preflight_only", "training_launched": False},
    )
    overrides = [
        f"data.loso_fold={args.fold}",
        f"seed={args.seed}",
        f"cell_owner_token={args.owner_token}",
        f"cell_paths.cell_dir={paths['cell_dir']}",
        f"cell_paths.owner={paths['owner']}",
        f"cell_paths.selector_records={paths['selector_records']}",
        f"cell_paths.checkpoints={paths['checkpoints']}",
    ]
    if paired_receipt is not None:
        overrides.append(f"model.paired_spint_completion_receipt={paired_receipt}")
    return {
        "schema": "m2_post33_phase_b_v3_preflight",
        "protocol_id": PROTOCOL_ID,
        "arm": args.arm,
        "fold": args.fold,
        "seed": args.seed,
        "outer_session": key.outer_session,
        "source_sessions": list(key.source_sessions),
        "paths": {name: str(path) for name, path in paths.items()},
        "hydra_overrides": overrides,
        "teacher_injection": (
            "matching_spint_completion_receipt_only" if args.arm == "t4" else "not_applicable"
        ),
        "execution_scope": {
            "gpu_used": False,
            "training_launched": False,
            "new_endpoint_r2_values_read": 0,
            "formal_sua_paths_resolved": 0,
            "evalai_calls": 0,
        },
    }


def main() -> None:
    print(json.dumps(prepare(parse_args()), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

