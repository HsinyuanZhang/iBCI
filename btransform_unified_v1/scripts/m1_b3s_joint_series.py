#!/usr/bin/env python3
"""M1 B3S+BT joint cell: trainable B3S (rSyn3 side) + P16 D2 transformer.

Relative to 582045: same P16 CausalPE depth-2 consumer and fullsession recipe.
The only intended change is the identity encoder — frozen B3
``compute_identity(side_features=None)`` becomes a jointly trained B3S that
sees the sealed rSyn3 carrier. rSyn3 itself stays closed-form.

Does not overwrite P16 d2, P32, concat, or CONCAT_W100 dests.
register:false. No pack / EvalAI here.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
for _p in (str(PACKAGE_ROOT / "src"), str(WORKSPACE_ROOT), str(PACKAGE_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from btransform_unified_v1 import m1_projadd as mp
from btransform_unified_v1 import plan
from btransform_unified_v1.m1_b3s_joint import (
    B3SJointDecoder,
    b3s_param_count,
    expected_joint_params,
    init_e0_matches_b3,
)

import m1_projadd_depth2_series as d2

TRAIN_DEPTH = 2
TRAIN_ENV_FLAG = "BTRANSFORM_M1_B3S_JOINT_TRAIN"
DECODER_PARAMS = int(expected_joint_params(16, TRAIN_DEPTH))
B3S_PARAMS = int(b3s_param_count())

d2.TRAIN_ENV_FLAG = TRAIN_ENV_FLAG
d2.EXPECTED_PARAMS_BY_DEPTH = {TRAIN_DEPTH: DECODER_PARAMS}
d2.DEPTH_CELLS = (TRAIN_DEPTH,)
d2.DEPTH_DIR = {TRAIN_DEPTH: "depth2"}


def cell_id(depth: int) -> str:
    if int(depth) != TRAIN_DEPTH:
        raise ValueError(f"M1 B3S+BT joint cell is D2 only, got {depth}")
    return "M1-B3S-P16-D2"


def _build_model(depth: int):
    if int(depth) != TRAIN_DEPTH:
        raise ValueError(f"M1 B3S+BT joint cell is D2 only, got {depth}")
    model = B3SJointDecoder(
        mp.m1_projadd_geometry(16),
        seed=d2.SEED,
        identity_mode="proj_add",
        proj_dim=16,
        temporal_layers=TRAIN_DEPTH,
    )
    plan.require(model.identity_mode == "proj_add", "identity_mode drift")
    plan.require(model.token_in == 20, "token_in drift")
    plan.require(len(model.temporal.blocks) == TRAIN_DEPTH, "temporal block count drift")
    n_params = int(sum(p.numel() for p in model.parameters()))
    plan.require(n_params == DECODER_PARAMS, f"param count {n_params} != {DECODER_PARAMS}")
    b3s_n = int(sum(p.numel() for p in model.b3s.parameters()))
    plan.require(b3s_n == B3S_PARAMS, f"B3S param count {b3s_n} != {B3S_PARAMS}")
    plan.require(all(p.requires_grad for p in model.b3s.parameters()), "B3S not trainable")
    return model


def prepare_trained_model(model, *, banks, device, face="", stage="", calib_by_session=None):
    if not isinstance(model, B3SJointDecoder):
        raise TypeError(f"joint cell expected B3SJointDecoder, got {type(model)}")
    if not calib_by_session:
        raise RuntimeError(f"{stage}: calib_by_session required for live B3S E0")
    model.attach_session_memories(banks, calib_by_session, device)
    return model


d2.cell_id = cell_id
d2._build_model = _build_model
d2.prepare_trained_model = prepare_trained_model


def run_probe(root: Path, dest: Path, face: str) -> dict:
    import numpy as np
    import torch

    if face != "fullsession":
        raise plan.BTransformerUnifiedError("M1 B3S+BT probe is fullsession only")
    t0 = time.monotonic()
    train_ds, sampler = d2.build_fullsession_face()
    banks, bank_report = d2.build_fullsession_banks(train_ds)
    cache_checks = d2.crosscheck_banks_vs_runtime_cache({n: banks[n] for n in d2.TRAIN_SESSIONS})
    calib = mp.calib_trials_from_dataset(train_ds)
    init_match = {
        name: init_e0_matches_b3(calib[name], banks[name].carrier)
        for name in mp.M1_SESSIONS
    }
    for name, row in init_match.items():
        plan.require(row["matched"], f"{name}: untrained B3S E0 != frozen B3 ({row})")
    model = _build_model(TRAIN_DEPTH)
    n_params = int(sum(p.numel() for p in model.parameters()))
    dropout_parity = d2.verify_dropout_parity()
    conv_path = d2.verify_conv_path(model)
    divisor = mp.assert_divisor_identity(model)
    x = torch.from_numpy(
        np.ascontiguousarray(train_ds[0][0], dtype=np.float32).reshape(1, mp.M1_WINDOW, mp.M1_UNITS)
    )
    session0 = train_ds[0][3]
    session0 = session0.decode() if isinstance(session0, bytes) else str(session0)
    causal = model.causal_check(x, banks[session0])
    train_sessions = list(mp.M1_SESSIONS)
    windows = {name: int(sum(1 for n, _ in train_ds.window_indices if n == name)) for name in train_sessions}
    updates_per_epoch = int(len(sampler))
    inventory = {
        "schema": "btransform_unified_v1_m1_b3s_joint_session_inventory",
        "route": "M1 B3S+BT P16 D2 joint 2026-09-07",
        "face": face,
        "utc": d2.utc_iso(),
        "train_protocol": (
            "stage-2 full-session 4 held-in; identity_mode=proj_add P16; "
            "trainable B3S sees rSyn3; transformer jointly trained; same LR 1e-4"
        ),
        "sampler": "SessionBatchSampler(batch=32, shuffle=True, seed=42, balance=False, reshuffle_each_epoch=False)",
        "sampler_batch_sha256": mp.sampler_digest(sampler),
        "sessions": train_sessions,
        "windows_per_session": windows,
        "total_train_windows": int(len(train_ds.window_indices)),
        "updates_per_epoch": updates_per_epoch,
        "warmup_updates": updates_per_epoch * plan.WARMUP_EPOCHS,
        "total_updates": updates_per_epoch * d2.EPOCHS,
        "identity_mode": "proj_add",
        "token_in": 20,
        "b3s_params": B3S_PARAMS,
        "depth_cells": {
            "2": {
                "cell_id": cell_id(2),
                "temporal_layers": 2,
                "decoder_params": n_params,
                "expected_decoder_params": DECODER_PARAMS,
                "frozen_p16_d2_params": 2479408,
            }
        },
        "init_e0_matches_frozen_b3": init_match,
        "banks": bank_report,
        "runtime_cache_crosscheck": cache_checks,
        "self_checks": {
            "dropout_parity_vs_s1": dropout_parity,
            "conv_path": conv_path,
            "divisor_identity": divisor,
            "causal_check": causal,
        },
        "probe_seconds": time.monotonic() - t0,
        "picks": d2._picks_with_runtime(updates_per_epoch, d2.PEAK_LR, TRAIN_DEPTH),
        "register": False,
        "comparison": {
            "official_p16_d2": "582045 HO 0.533 / local HO-calib 0.613",
            "success_bar": "first beat 581982 official 0.574; 0.64 is a different consumer",
        },
    }
    d2._seal(dest / "session_inventory.json", inventory)
    print(
        f"[m1-b3s-joint] probe: windows={inventory['total_train_windows']} "
        f"upd/ep={updates_per_epoch} params={n_params} b3s={B3S_PARAMS}",
        flush=True,
    )
    return inventory


d2.run_probe = run_probe


def main() -> int:
    parser = argparse.ArgumentParser(description="M1 B3S+BT joint P16 D2 cell")
    parser.add_argument("--depth", type=int, choices=[TRAIN_DEPTH], default=TRAIN_DEPTH)
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
        f"[m1-b3s-joint] cell={cell_id(args.depth)} stage={args.stage} face={args.face} "
        f"root={root} dest={dest} gpu={args.cuda_visible}",
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
        print(f"[m1-b3s-joint] preflight ok={pre['ok']} foreign={pre['foreign_pids_on_target']}", flush=True)
        if gpu_stage and not pre["ok"]:
            print(f"[m1-b3s-joint] BLOCKED: GPU{args.cuda_visible} not clean", file=sys.stderr)
            return 2

    if args.stage == "preflight":
        return 0
    if args.stage in ("probe", "all"):
        run_probe(root, root, args.face)
    if args.stage == "timing":
        d2.run_timing(root)
    if args.stage in ("train", "all"):
        summary = d2.run_train(root, dest, args.depth, args.peak_lr, args.face)
        print(f"[m1-b3s-joint] train done in {summary['elapsed_s']:.0f}s", flush=True)
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
