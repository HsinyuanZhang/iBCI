#!/usr/bin/env python3
"""Coordinator CLI for m2_dual_track_v1.

Workers implement subcommands by filling owner modules. This file is
coordinator-owned: add flags here, do not grow architecture inside it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("PYTHONNOUSERSITE", "1")


def _apply_env() -> None:
    from tfpd_exploration.src.m2_dual_track_v1 import plan

    for key, value in plan.REQUIRED_ENV.items():
        os.environ.setdefault(key, value)


def cmd_status(_: argparse.Namespace) -> int:
    from tfpd_exploration.src.m2_dual_track_v1 import jobs, plan

    root = plan.active_run_root()
    lease = jobs.load_lease(root)
    print(json.dumps({
        "schema": plan.SCHEMA,
        "contract_version": plan.CONTRACT_VERSION,
        "run_root": str(root),
        "lease": lease,
        "jobs_path": str(jobs.ledger_path(root)),
    }, indent=2, sort_keys=True))
    return 0


def cmd_stage0(args: argparse.Namespace) -> int:
    from tfpd_exploration.src.m2_dual_track_v1 import plan
    from tfpd_exploration.src.m2_dual_track_v1.stage0 import run_stage0

    return int(run_stage0(subset=args.subset, device=args.device))


def cmd_score_ref(args: argparse.Namespace) -> int:
    from tfpd_exploration.src.m2_dual_track_v1.evaluation import score_reference

    return int(score_reference(device=args.device))


def cmd_train(args: argparse.Namespace) -> int:
    from tfpd_exploration.src.m2_dual_track_v1.launch import run_training

    return int(
        run_training(
            arm=args.arm,
            seed=args.seed,
            device=args.device,
            max_epochs=args.epochs,
            run_tag=args.run_tag,
            manifest=args.manifest,
            resume=args.resume or None,
        )
    )


def cmd_write_manifest(args: argparse.Namespace) -> int:
    from tfpd_exploration.src.m2_dual_track_v1.launch import (
        ensure_shared_manifest,
        ensure_shared_manifest_24,
        shared_manifest_24_path,
        shared_manifest_path,
    )

    if int(args.epochs) >= 24:
        manifest = ensure_shared_manifest_24()
        path = shared_manifest_24_path()
    else:
        manifest = ensure_shared_manifest()
        path = shared_manifest_path()
    print(json.dumps({"path": str(path), "digest": manifest["digest"], "epochs": manifest["epochs"], "parent_12_digest": manifest.get("parent_12_digest")}, indent=2))
    return 0


def main() -> int:
    _apply_env()
    parser = argparse.ArgumentParser(prog="run_m2_dual_track_v1")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status")

    p0 = sub.add_parser("stage0")
    p0.add_argument("--subset", default="all", choices=["all", "authority", "a", "b", "data"])
    p0.add_argument("--device", default="cpu")

    pref = sub.add_parser("score-ref")
    pref.add_argument("--device", default="cuda:0")

    ptr = sub.add_parser("train")
    ptr.add_argument("--arm", required=True, choices=["A-QMEM", "B-MAMBA", "B-TRANSFORMER"])
    ptr.add_argument("--seed", type=int, default=42)
    ptr.add_argument("--device", default="cuda:0")
    ptr.add_argument("--epochs", type=int, default=12)
    ptr.add_argument("--run-tag", default="")
    ptr.add_argument("--manifest", default="")
    ptr.add_argument("--resume", default="", help="full epoch_012.pt; continues into a new run-tag dest")

    pa = sub.add_parser("score-a")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--device", default="cuda:0")

    pas = sub.add_parser("score-a-scan")
    pas.add_argument("--device", default="cuda:0")

    pb = sub.add_parser("score-b")
    pb.add_argument("--arm", required=True, choices=["B-MAMBA", "B-TRANSFORMER"])
    pb.add_argument("--seed", type=int, default=42)
    pb.add_argument("--run-tag", default="")
    pb.add_argument("--device", default="cuda:0")

    pman = sub.add_parser("write-manifest")
    pman.add_argument("--epochs", type=int, default=12)

    pbs = sub.add_parser("score-b-scan")
    pbs.add_argument("--arm", required=True, choices=["B-MAMBA", "B-TRANSFORMER"])
    pbs.add_argument("--seed", type=int, default=42)
    pbs.add_argument("--run-tag", default="shuffled_e13_24")
    pbs.add_argument("--device", default="cuda:0")

    prs = sub.add_parser("resume-smoke")
    prs.add_argument("--arm", required=True, choices=["B-MAMBA", "B-TRANSFORMER"])
    prs.add_argument("--seed", type=int, default=42)
    prs.add_argument("--device", default="cuda:0")

    args = parser.parse_args()
    if args.command == "status":
        return cmd_status(args)
    if args.command == "stage0":
        return cmd_stage0(args)
    if args.command == "score-ref":
        return cmd_score_ref(args)
    if args.command == "train":
        if not args.manifest:
            args.manifest = None
        return cmd_train(args)
    if args.command == "score-a":
        from tfpd_exploration.src.m2_dual_track_v1.launch import score_a_qmem

        return int(score_a_qmem(seed=args.seed, device=args.device))
    if args.command == "score-a-scan":
        from tfpd_exploration.src.m2_dual_track_v1.launch import score_a_all_epochs

        return int(score_a_all_epochs(device=args.device))
    if args.command == "score-b":
        from tfpd_exploration.src.m2_dual_track_v1.launch import score_b_decoder

        return int(score_b_decoder(arm=args.arm, seed=args.seed, run_tag=args.run_tag, device=args.device))
    if args.command == "score-b-scan":
        from tfpd_exploration.src.m2_dual_track_v1.launch import score_b_extended

        return int(score_b_extended(arm=args.arm, seed=args.seed, run_tag=args.run_tag))
    if args.command == "write-manifest":
        return cmd_write_manifest(args)
    if args.command == "resume-smoke":
        from tfpd_exploration.src.m2_dual_track_v1.launch import resume_next_step_parity

        return int(resume_next_step_parity(arm=args.arm, seed=args.seed))
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
