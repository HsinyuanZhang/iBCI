#!/usr/bin/env python3
"""Train/score an M2 RIFT concat run against a variant cache (no trainer edits)."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT.parent
for p in (ROOT / "src", ROOT / "scripts" / "rift_v1", WS / "btransform_unified_v1" / "src", WS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _redirect(run_root: Path) -> None:
    rel = str(run_root.resolve().relative_to(WS))
    from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan

    old_plan.ACTIVE_RUN_RELATIVE = rel
    from btransform_unified_v1 import adapters

    adapters._M2_CACHE_ROOT = run_root.resolve() / "cache"
    if not adapters._M2_CACHE_ROOT.is_dir():
        raise RuntimeError(f"variant cache missing: {adapters._M2_CACHE_ROOT}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--stage", choices=("train", "score", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates-smoke", type=int)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    _redirect(args.run_root)
    train = importlib.import_module("m2_concat_train")
    ns = argparse.Namespace(
        dest=args.dest,
        stage=args.stage,
        device=args.device,
        epochs=args.epochs,
        resume=args.resume,
        max_updates_smoke=args.max_updates_smoke,
        cpu_threads=args.cpu_threads,
    )
    result = None
    if args.stage in ("train", "all"):
        result = train.run_train(ns)
    if args.stage in ("score", "all"):
        if result is not None and result["status"] != "TRAIN_COMPLETED":
            raise RuntimeError("formal training did not complete")
        result = train.score_stage(ns)
    import json
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
