#!/usr/bin/env python3
"""M1 proj_add P32 × temporal depth {2,3} fullsession cell.

User 2026-09-07: after P16 depth-2, try P32 on the same M1 fullsession face
and test whether depth 3 (and depth 2) holds quality. Does NOT modify the
sealed P16 depth-2 dest or ``m1_projadd_depth2_series.py`` constants in place;
this wrapper patches the imported module for P32 / d3.

Face: stage-2 ALL 4 held-in (213,336 windows). Recipe: seed 42, AdamW 1e-4,
EMA 0.9995, 24 epochs, batch 32. Identity: B-transformer unified, NOT SPINT.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
for _p in (str(PACKAGE_ROOT / "src"), str(WORKSPACE_ROOT), str(PACKAGE_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from btransform_unified_v1 import m1_projadd as mp
from btransform_unified_v1 import plan
from btransform_unified_v1.identity_variant import BTransformerUnifiedDecoderIdentity
from btransform_unified_v1.model import N_LAYERS

import m1_projadd_depth2_series as d2

PROJ_DIM = 32
TOKEN_IN = 36
TRAIN_DEPTHS = (2, 3)
TRAIN_ENV_FLAG = "BTRANSFORM_M1_P32_DEPTH_TRAIN"
TEMPORAL_BLOCK = d2.TEMPORAL_BLOCK_PARAMS
P32_D4 = int(mp.m1_projadd_param_count(PROJ_DIM))

d2.TRAIN_ENV_FLAG = TRAIN_ENV_FLAG
d2.EXPECTED_PARAMS_BY_DEPTH = {
    2: P32_D4 - 2 * TEMPORAL_BLOCK,
    3: P32_D4 - 1 * TEMPORAL_BLOCK,
    4: P32_D4,
}
d2.DEPTH_CELLS = (2, 3, 4)
d2.DEPTH_DIR = {2: "depth2", 3: "depth3", 4: "depth4"}


def cell_id(depth: int) -> str:
    return f"M1-PROJADD-P32-D{int(depth)}"


def _build_model(depth: int) -> BTransformerUnifiedDecoderIdentity:
    model = BTransformerUnifiedDecoderIdentity(
        mp.m1_projadd_geometry(PROJ_DIM),
        seed=d2.SEED,
        identity_mode="proj_add",
        proj_dim=PROJ_DIM,
        temporal_layers=None if int(depth) == N_LAYERS else int(depth),
    )
    plan.require(model.identity_mode == "proj_add", "identity_mode drift")
    plan.require(model.token_in == TOKEN_IN, f"token_in {model.token_in} != {TOKEN_IN}")
    plan.require(model.init_meta["temporal_layers"] == int(depth), "temporal depth drift")
    plan.require(len(model.temporal.blocks) == int(depth), "temporal block count drift")
    n_params = int(sum(p.numel() for p in model.parameters()))
    plan.require(
        n_params == d2.EXPECTED_PARAMS_BY_DEPTH[int(depth)],
        f"param count {n_params} != {d2.EXPECTED_PARAMS_BY_DEPTH[int(depth)]} at P32 d{depth}",
    )
    return model


d2.cell_id = cell_id
d2._build_model = _build_model


def main() -> int:
    parser = argparse.ArgumentParser(description="M1 proj_add P32 × depth 2/3 fullsession")
    parser.add_argument("--depth", type=int, choices=list(TRAIN_DEPTHS), default=3)
    parser.add_argument("--cuda-visible", default="0", choices=["0", "1"])
    parser.add_argument("--face", choices=("fullsession",), default="fullsession")
    parser.add_argument("--pick-epochs", default="18:24")
    parser.add_argument("--stage", choices=list(d2.STAGES), default="probe")
    parser.add_argument("--peak-lr", type=float, default=d2.PEAK_LR)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dest", type=Path, default=None)
    args = parser.parse_args()
    if args.peak_lr != d2.PEAK_LR:
        print(f"REFUSED: --peak-lr frozen at {d2.PEAK_LR:g}", file=sys.stderr)
        return 2

    d2.CUDA_PIN = str(args.cuda_visible)
    root = args.root
    dest = args.dest if args.dest else root / d2.DEPTH_DIR[args.depth]
    if args.stage in ("probe", "timing"):
        dest = root
    root.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    print(
        f"[p32-depth] cell={cell_id(args.depth)} stage={args.stage} face={args.face} "
        f"root={root} dest={dest} gpu={args.cuda_visible}",
        flush=True,
    )
    print(
        f"[p32-depth] expected params d2/d3/d4="
        f"{d2.EXPECTED_PARAMS_BY_DEPTH[2]}/{d2.EXPECTED_PARAMS_BY_DEPTH[3]}/{d2.EXPECTED_PARAMS_BY_DEPTH[4]}",
        flush=True,
    )

    gpu_stage = args.stage in ("train", "score", "pick", "all")
    if args.stage == "train" and os.environ.get(TRAIN_ENV_FLAG) != "1":
        print(f"REFUSED: training requires {TRAIN_ENV_FLAG}=1", file=sys.stderr)
        return 2
    if gpu_stage and os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.cuda_visible):
        print(f"REFUSED: CUDA_VISIBLE_DEVICES must be {args.cuda_visible!r}", file=sys.stderr)
        return 2

    if gpu_stage or args.stage == "preflight":
        pre = d2.gpu_preflight(dest / f"preflight_gpu_{args.stage}.json", args.depth, args.stage)
        print(f"[p32-depth] preflight ok={pre['ok']} foreign={pre['foreign_pids_on_target']}", flush=True)
        if gpu_stage and not pre["ok"]:
            print(f"[p32-depth] BLOCKED: GPU{args.cuda_visible} not clean", file=sys.stderr)
            return 2

    if args.stage == "preflight":
        return 0
    if args.stage in ("probe", "all"):
        d2.run_probe(root, root, args.face)
    if args.stage == "timing":
        d2.run_timing(root)
    if args.stage in ("train", "all"):
        summary = d2.run_train(root, dest, args.depth, args.peak_lr, args.face)
        print(
            f"[p32-depth] train d{args.depth} done in {summary['elapsed_s']:.0f}s "
            f"updates={summary['global_updates']}",
            flush=True,
        )
    if args.stage in ("score", "all"):
        d2.run_score(root, dest, args.depth, args.face)
    if args.stage == "pick":
        lo_text, hi_text = args.pick_epochs.split(":")
        d2.run_pick(root, dest, args.depth, int(lo_text), int(hi_text))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(3)
