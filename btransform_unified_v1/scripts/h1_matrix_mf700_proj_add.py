"""H1 matrix baseline cell M-F700: L=700 (full window) x identity proj_add.

MATRIX_H1_L_IDENTITY_V1_20260906.md §2 revision (formalized by
REVIEW_MATRIX_H1_L_IDENTITY_V1_20260906, blocking items A/B/C): M-F700 is the
preregistered same-recipe FULL-WINDOW baseline cell of the L axis — with
M-F250 (proj_add at L=250) it forms the same-injection L-axis controlled pair
(same CAL-2/M3, same LODO holdout, same recipe/seed; the ONLY variable is L).
L=700 is the ORIGINAL settled H1 window (not a shortening), so the model
builds with the settled window itself — NO override_window (that control only
ever shortens and keeps refusing 700).

BTransformerUnifiedDecoderIdentity, identity_mode="proj_add": learnable
P = Linear(700 -> 16, bias=False) fuses the fused-E0 identity additively into
the LOCAL conv channels; carrier concat unchanged; rank-16 bottleneck of the
concat first layer; L-independent (no 700-bin template alignment).

Method-selection exam: LODO holdout date 1925-01-20 (2 sessions) — the cell is
trained on the other 11 sessions and scored on the holdout date's minival
eval_mask coordinates (SUBMISSION-PROTOCOL stage 1; stage-2 full retraining is
outside the matrix).

Per-epoch scoring (EMA + RAW views):
  - lodo exam: holdout-date sessions, all minival eval_mask ends (2,952 faces)
  - 2,908 selection face (AUXILIARY only): minival query_starts+699 grid over
    all 13 sessions — the formal12 alignment face; NOT used for epoch picking.

SEL-2 (matrix §0 revision, blocking item B): epoch-pick = earliest max of the
LODO-exam EMA equal_session_mean series (holdout date only; M1 LOSO lesson —
same-source faces mask cross-session drops). SEL-1 endpoint24 + last-4/last-8
on the same exam series; SEL-4 pooled / session-mean dual report everywhere.

CAL-2/M3 (matrix §0 revision, blocking item A): every train/eval visit
consumes the budget-3 frozen bank (bitwise identical to the five-arm cache
bank via the adapter provenance loop). The CAL-1 {7,5,4,3} rotation is
deferred as a post-matrix upgrade cell (adapter stub: NWB re-trialization for
budgets 7/5/4 is not wired); no rotation schedule is consumed or recorded
here.

TRN-6 at L=700: microbatch 8 x accumulation 4 (effective batch 32; formal12
recorded caliber); an on-card memory probe records peak allocs at micro 8/16
for evidence only.

Data: adapters.build_h1_bank(surface, session, budget=3, window=700,
identity_mode="proj_add") — the live Phase-2b path (five-arm windowing at the
full window; targets native; E0 fused [176,700] reproduced bitwise from the
frozen cache). The 2,908 query-grid face is materialized with the adapter's
own _h1_windows on the minival query grid (count gate == 2908).

Discipline: CUDA_VISIBLE_DEVICES pinned to GPU0 (uuid
GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9; refuses if a foreign pid holds
>500 MiB there); GPU1 belongs to the M-F250 session and is never touched;
bf16 autocast training, FP32 eval; 6h budget from train start; receipts
sealed via receipts.seal_json (0444 + sha256 sidecar); historical roots
untouched; no EvalAI.

Usage:
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
  BTRANSFORM_MF700_TRAIN=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v1/scripts/h1_matrix_mf700_proj_add.py --stage all \
  --dest btransform_unified_v1/results/h1_matrix/M_F700_<stamp>

Identity: B-transformer unified series, NOT SPINT.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
for _p in (str(PACKAGE_ROOT / "src"), str(WORKSPACE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from btransform_unified_v1 import adapters, h1_config, plan, receipts  # noqa: E402
from btransform_unified_v1.ema import DecoderEMA  # noqa: E402
from btransform_unified_v1.identity_variant import BTransformerUnifiedDecoderIdentity  # noqa: E402
from btransform_unified_v1.matrix_cells import baseline_cell, cell_by_id, cell_geometry  # noqa: E402
from btransform_unified_v1.model import UNIT_DROPOUT_DOMAIN_META, unit_dropout_seed, whole_unit_dropout  # noqa: E402
from btransform_unified_v1.r2 import session_mean_report, variance_weighted_r2  # noqa: E402
from btransform_unified_v1.scale_bridge import assert_scale_bridge  # noqa: E402
from btransform_unified_v1.schedule import warmup_cosine_lr  # noqa: E402

CELL_ID = "M-F700"
CELL = "M-F700-PROJ-ADD-FULLWINDOW"
WINDOW = h1_config.FULL_WINDOW  # 700 — the ORIGINAL settled window (no override)
TRAIN_ENV_FLAG = "BTRANSFORM_MF700_TRAIN"
SEED = plan.SEED
GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"  # this cell's card
GPU1_UUID = "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"  # M-F250's card — never touched
FOREIGN_MEM_MIB_LIMIT = 500
FIVEARM_PID = "1803132"
BUDGET_SECONDS = 6.0 * 3600.0
EPOCHS = plan.EPOCHS
SCALE = h1_config.TARGET_MULTIPLIER  # 20.0
EFFECTIVE_BATCH = plan.BATCH_SIZE  # 32
EVAL_BATCH = 8  # L=700 fp32 eval: batch 32 needs ~7.5 GiB single allocs (OOM'd
# attempt M_F700_20260906T092204Z at the first epoch-end eval); 8 keeps the
# largest k/v tensors ~1 GiB alongside the training-fragmented cache
SELECTION_FACE_COUNT = 2908  # frozen minival query-grid count (formal12 rule)
N_TRAIN_FACES_EXPECTED = 18935  # LODO train-side face inventory (face_inventory.json)
ARBITRATION_LOG = (
    plan.RESULT_ROOT / "h1_matrix" / "M_F700_arbitration" / "gpu0_arbitration.json"
)

# Authorized restart variant marker (coordinator approval 2026-09-06, after the
# 3e-4 ep10 EARLY_FAIL): the ONLY change is peak LR 3e-4 -> 1e-4; the schedule
# shape stays warmup 597 -> cosine 14328 -> 1e-5 (min_factor 0.1 x 1e-4).
LR_1E4_VARIANT_MARKER = (
    "TRN-1-LR1e4 variant: deviation from matrix §0 peak 3e-4, approved restart "
    "per early_fail（双规则一致，跨 L 平台 0.00695≈0.006911 佐证 recipe 级失败）"
)
LR_AUTHORIZATION_NOTE = (
    "coordinator-approved restart 2026-09-06: M-F700 retry with peak LR 1e-4 "
    "(schedule shape unchanged: warmup 597 -> cosine 14328 -> 1e-5); everything "
    "else identical to the 3e-4 run; a second plateau will NOT be self-retried "
    "(FAIL receipt + await ruling); pairs with M-F250-lr1e4 (GPU1) as the clean "
    "1e-4 L-axis controlled pair"
)

# Reference comparisons (receipt context only, NOT gates):
REFERENCES = {
    "fivearm_l100_finals": {
        "concat700_pooled_r2": 0.0802,
        "joined36_PROXY_pooled_r2": 0.1777,
        "add_tail_pooled_r2": 0.0086,
        "e0_zero_pooled_r2": 0.0202,
        "permute": "not written to live.json (process stopped; ordering already settled)",
        "note": (
            "five-arm L=100 finals (12-ep, 20,325-face minival, pooled R2, "
            "live.json); joined36 there is the PROXY (first 32 dims of fused "
            "E0), NOT the true C2 pre-pool object"
        ),
        "source": "btransform_unified_v1/results/h1_sec6_fivearm_l100_v1/live.json",
    },
    "formal12_l700_full_window": {
        "pooled_r2_20325_complete": 0.278,
        "pooled_r2_2908_selection": 0.320,
        "note": "L=700 full-window formal12, same-data reference (different recipe: 12ep warmup->1e-4 const)",
        "source": "tfpd_exploration/results/decoder_validation_v2/20260905_190000/h1/queryage_family_v1/formal_prefix_split12_v1/receipt.json",
    },
}

PICKS = {
    "CAL": [
        "CAL-2/M3 {fixed budget 3 frozen bank for EVERY train/eval visit; matrix §0 "
        "revision 2026-09-06 (blocking item A) replaces the original CAL-1 global "
        "rotation; adapter stub: NWB re-trialization for budgets {7,5,4} not wired; "
        "CAL-1 (H1's only officially confirmed gain, +0.043) deferred as a "
        "post-matrix upgrade cell on the winning mode+L}",
    ],
    "TRN": [
        "TRN-1 {24ep, AdamW wd 0.01 clip 1.0, warmup 1ep -> 3e-4 -> cosine 3e-5 "
        "(min_factor 0.1), EMA 0.9995, effective batch 32, bf16 autocast "
        "forward+loss with pred.float() before MSE}",
        "TRN-1 update caliber {optimizer steps/epoch measured from THIS run's "
        "LODO-train face count AND per-session batching: sum_s ceil(n_s/32) = 597 "
        "upd/ep -> 14328 total, warmup 597 (matrix §0 revision + M-F250 attempt3 "
        "recorded caliber; do NOT write formal12's 731/ep nor the naive "
        "ceil(18935/32)=592; LR schedule steps on OPTIMIZER steps, not micro "
        "batches)}",
        "TRN-3 {whole-unit dropout p=0.10, training mode only, per-batch CPU "
        "generator, domain m2_small_unit_dropout (S1 unit_dropout_seed "
        "import-or-replicate)}",
        "TRN-5 {P: prefix 0; window = the settled full window 700 itself — NO "
        "override_window (the L control only ever shortens and keeps refusing 700)}",
        "TRN-6 {microbatch 8 x accum 4 fixed (formal12 recorded L=700 caliber); "
        "on-card probe of micro 8/16 recorded in run_meta.microbatch_probe for "
        "evidence only}",
        "TRN-8 {scale: x20 train on native targets; score pred/20 vs native; ratio "
        "bridge MSE(raw,20y)==400*MSE(raw/20,y) rel tol 1e-9}",
        "TRN-9 {health adjudication (coordinator 2026-09-06, updated after "
        "M-F250@3e-4 ep10 FAIL at plateau 0.006911): M-F700@3e-4 doubles as the "
        "L=700 LR control; ruling at ep10 only — CONTINUE iff train_mse <= "
        "0.0065 AND examRAW >= -0.007; EARLY FAIL iff train_mse pinned at "
        "0.0069±0.0001 OR examRAW declining 3 consecutive epochs below -0.01; "
        "EMA washout never rules; restart hypothesis (LR 1e-4, pairing with "
        "M-F250-lr1e-4 retry) recorded pending approval — never self-authorized}",
    ],
    "SEL": [
        "SEL-2 {surface: LODO holdout-date exam (1925-01-20, 2 sessions, 2,952 "
        "faces, trained-never), view: EMA, statistic: equal_session_mean, rule: "
        "earliest max, ties<=1e-10 -> earliest; rule fixed before reading numbers; "
        "the 2,908 same-source face is AUXILIARY ONLY (formal12 alignment) and is "
        "NEVER used for picking — matrix §0 revision (blocking item B, M1 LOSO "
        "lesson)}",
        "SEL-1 {endpoint24 primary + last-4/last-8 auxiliary on the SEL-2 exam "
        "series (equal_session_mean); 2,908 endpoint24 reported alongside as the "
        "formal12-alignment auxiliary}",
        "SEL-4 {pooled and session-mean dual report on every surface; never "
        "subtract across surfaces}",
        "SEL-3 {official surface: zero participation}",
    ],
    "SPD": [
        "none {training-path run; proj_add static-fold parity is covered by the "
        "skeleton tests, not used in training/scoring}",
    ],
    "IDENTITY_USAGE": [
        "proj_add {P = Linear(700->16, bias=False); tokens = token_mlp([local16 + "
        "P(E0)] | carrier4); carrier concat unchanged; rank-16 bottleneck of the "
        "concat first layer}",
        "L-independence declaration {proj_add enters via a session-static 16-d "
        "channel projection added to the local conv channels; there is NO 700-bin "
        "identity TIME template and NO end-aligned tail alignment (the add_tail "
        "alignment question does not apply at any L); the L axis and the usage "
        "axis do not interact through the identity pathway}",
        "naming {receipts use mode names (concat / joined / add_tail / zero / "
        "permute / proj_add), not matrix letters — matrix §2 revision}",
    ],
}

NOTE_SIX_ROWS = {
    "system": (
        "btransform_unified_v1 BTransformerUnifiedDecoderIdentity (H1 matrix "
        "baseline cell M-F700: settled full window L=700 with NO override, prefix "
        "0, identity_mode proj_add — P=Linear(700->16,bias=False) fused into the "
        "local conv channels, token_in 20, seed 42) — NOT SPINT"
    ),
    "consumer": (
        "8-slot + CausalPE4 unified decoder (this series), H1 L x identity "
        "training matrix (MATRIX_H1_L_IDENTITY_V1_20260906 §2 revision); "
        "matrix-cell readout, no official submission"
    ),
    "calibration_object": (
        "E0 [176,700] C2 fused identity, materialized by the frozen C2 e15 "
        "materializer (ckpt sha ce46267e...) over the M3 payload trialized "
        "activity [3,1024,176] — bitwise-reproduces the five-arm cache bank "
        "(adapter provenance loop); H-C carrier [176,4]; CAL-2 fixed M3 (matrix "
        "§0 revision 2026-09-06: CAL-1 rotation deferred as a post-matrix "
        "upgrade cell — adapter stub, budgets {7,5,4} not materializable)"
    ),
    "scoring_surface": (
        "LODO method-selection exam: holdout date 1925-01-20 (2 sessions, 2,952 "
        "minival eval_mask faces) trained-never — the SEL-2 epoch-pick surface "
        "(EMA equal_session_mean earliest max); the 2,908 full-window selection "
        "face (minival query_starts+699, all 13 sessions) is AUXILIARY "
        "(formal12 alignment, never used for picking); EMA+RAW per epoch; pooled "
        "+ session-mean dual report"
    ),
    "scale": (
        "x20 train / score pred/20 (ratio bridge MSE(raw,20y) = 400*MSE(raw/20,y), "
        "rel tol 1e-9, asserted every epoch on the LODO exam)"
    ),
    "single_difference_vs_historical_best": (
        "vs M-F250 (the L-axis partner): exactly ONE differing item — the input "
        "window L (700 vs 250). Same proj_add injection, same CAL-2/M3 budget, "
        "same LODO holdout (1925-01-20), same TRN-1 recipe, same seed 42, same "
        "face coordinates (session, end); parameterization width and total "
        "parameter count are asserted equal (PE is a non-persistent sinusoidal "
        "buffer), so 'only variable L' holds at the parameter level too"
    ),
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _seal(path: Path, payload: Any) -> str:
    return receipts.seal_json(path, payload)


# ---------------------------------------------------------------------------
# GPU0 preflight (pinned; foreign pid > 500 MiB => BLOCKED receipt)
# ---------------------------------------------------------------------------


def _nvidia_smi(args: list[str]) -> str:
    out = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def gpu_preflight(out_path: Path) -> dict[str, Any]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_rows = _nvidia_smi(
        ["--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
    )
    gpus = []
    for line in gpu_rows.splitlines():
        idx, uuid, util, mem_used, mem_total = [t.strip() for t in line.split(",")]
        gpus.append(
            {
                "index": int(idx),
                "uuid": uuid,
                "utilization_pct": float(util),
                "memory_used_mib": float(mem_used),
                "memory_total_mib": float(mem_total),
            }
        )
    app_rows = _nvidia_smi(
        ["--query-compute-apps=gpu_uuid,pid,used_memory,process_name", "--format=csv,noheader,nounits"]
    )
    apps = []
    for line in app_rows.splitlines():
        parts = [t.strip() for t in line.split(",")]
        apps.append(
            {
                "gpu_uuid": parts[0],
                "pid": int(parts[1]),
                "used_mib": float(parts[2]),
                "process_name": ",".join(parts[3:]),
            }
        )
    own_pid = os.getpid()
    foreign = [
        app
        for app in apps
        if app["gpu_uuid"] == GPU0_UUID and app["pid"] != own_pid and app["used_mib"] > FOREIGN_MEM_MIB_LIMIT
    ]
    gpu0 = next(g for g in gpus if g["uuid"] == GPU0_UUID)
    fivearm_proc = Path(f"/proc/{FIVEARM_PID}")
    report = {
        "schema": "btransform_unified_v1_mf700_gpu_preflight",
        "cell": CELL,
        "unix": time.time(),
        "utc": datetime.now(timezone.utc).isoformat(),
        "own_pid": own_pid,
        "cuda_visible_devices": raw,
        "gpus": gpus,
        "compute_apps": apps,
        "gpu0": gpu0,
        "foreign_pids_on_gpu0": foreign,
        "foreign_threshold_mib": FOREIGN_MEM_MIB_LIMIT,
        "fivearm_pid": FIVEARM_PID,
        "fivearm_proc_present": fivearm_proc.exists(),
        "gpu1_touched": False,
        "gpu1_owner": "M-F250 session (parallel L-axis partner); CUDA pinned to 0 here",
        "arbitration_log": str(ARBITRATION_LOG),
        "note": (
            "arbitration PASSED 2026-09-06T08:58:58Z (five-arm pid 1803132 gone, "
            "last artifact write 08:40:20Z; GPU0 453 MiB graphics-only); this "
            "preflight re-verifies at launch time"
        ),
    }
    ok = (
        raw == "0"
        and not foreign
        and torch.cuda.device_count() == 1
        and gpu0["memory_used_mib"] < FOREIGN_MEM_MIB_LIMIT
        and not fivearm_proc.exists()
    )
    report["ok"] = bool(ok)
    _seal(out_path, report)
    return report


# ---------------------------------------------------------------------------
# Data: LODO split + faces via the adapter (budget 3, window 700, proj_add)
# ---------------------------------------------------------------------------


def build_faces() -> dict[str, Any]:
    """Banks + face arrays for train / lodo-exam / 2908-selection surfaces."""
    split = h1_config.lodo_split()
    train_sessions = list(split["train_sessions"])
    holdout_sessions = list(split["holdout_sessions"])

    train_banks, train_X, train_y, train_ids = {}, {}, {}, {}
    for s in train_sessions:
        bank = adapters.build_h1_bank("train", s, budget=3, window=WINDOW, identity_mode="proj_add")
        train_banks[s] = bank
        train_X[s] = bank.X_store
        train_y[s] = bank.target_store
        train_ids[s] = bank.window_ids

    exam_banks, exam_X, exam_y, exam_ids = {}, {}, {}, {}
    for s in holdout_sessions:
        bank = adapters.build_h1_bank("minival", s, budget=3, window=WINDOW, identity_mode="proj_add")
        exam_banks[s] = bank
        exam_X[s] = bank.X_store
        exam_y[s] = bank.target_store
        exam_ids[s] = bank.window_ids

    # 2,908 AUXILIARY selection face: minival query grid (query_starts + 699),
    # all 13 sessions, windowed with the adapter's own generalized _windows.
    cache = adapters._h1_source_cache()
    sel_banks, sel_X, sel_y, sel_ids = {}, {}, {}, {}
    for s in h1_config.H1_ALL_SESSIONS:
        bank = exam_banks.get(s) or train_banks.get(s)
        if bank is None:
            bank = adapters.build_h1_bank("minival", s, budget=3, window=WINDOW, identity_mode="proj_add", limit_windows=1)
            sel_banks[s] = bank
        else:
            sel_banks[s] = bank
        row = cache["minival"][s]
        ends = (np.asarray(row["query_starts"], dtype=np.int64) + h1_config.FULL_WINDOW - 1).astype(np.int64)
        neural = np.ascontiguousarray(row["neural"], dtype=np.float32)
        sel_X[s] = adapters._h1_windows(neural, ends, WINDOW)
        sel_y[s] = np.ascontiguousarray(np.asarray(row["velocity"], dtype=np.float32)[ends])
        sel_ids[s] = ends
    n_sel = int(sum(len(v) for v in sel_ids.values()))
    plan.require(
        n_sel == SELECTION_FACE_COUNT,
        f"selection face count {n_sel} != frozen {SELECTION_FACE_COUNT} (h1_optimized_v2.score selection grid)",
    )

    return {
        "split": split,
        "train": {"sessions": train_sessions, "banks": train_banks, "X": train_X, "y": train_y, "ids": train_ids},
        "exam": {"sessions": holdout_sessions, "banks": exam_banks, "X": exam_X, "y": exam_y, "ids": exam_ids},
        "sel": {"sessions": list(h1_config.H1_ALL_SESSIONS), "banks": sel_banks, "X": sel_X, "y": sel_y, "ids": sel_ids},
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@torch.no_grad()
def _score_faces(
    model: BTransformerUnifiedDecoderIdentity,
    faces: dict[str, Any],
    sessions: list[str],
    device: torch.device,
    *,
    bridge_check: bool = False,
) -> dict[str, Any]:
    """pooled + per-session + equal-mean R2 in native space (pred/20)."""
    model.eval()
    preds, targets, names = [], [], []
    raw_dump, native_dump = [], []
    for s in sessions:
        X, y = faces["X"][s], faces["y"][s]
        bank = faces["banks"][s]
        for off in range(0, len(X), EVAL_BATCH):
            xb = torch.from_numpy(X[off : off + EVAL_BATCH]).to(device)
            with torch.inference_mode():
                raw = model(xb, bank)
            raw_np = raw.detach().cpu().numpy()
            preds.append(raw_np / SCALE)
            targets.append(y[off : off + EVAL_BATCH])
            names.extend([s] * len(raw_np))
            if bridge_check:
                raw_dump.append(raw_np)
                native_dump.append(y[off : off + EVAL_BATCH])
    pred = np.concatenate(preds, axis=0)
    target = np.concatenate(targets, axis=0)
    name_arr = np.asarray(names)
    plan.require(name_arr.size == target.shape[0], "session id count mismatch")
    if bridge_check:
        assert_scale_bridge("h1", np.concatenate(raw_dump, axis=0), np.concatenate(native_dump, axis=0))
    # r2.session_mean_report flattens [n, out_dim] -> n*out_dim points; tile the
    # session ids to match (each face contributes out_dim flattened points).
    # Same fix as the M-F250 partner script (17:11 local) so the L-axis pair is
    # scored under bit-identical conventions.
    report = session_mean_report(target, pred, np.repeat(name_arr, target.shape[1]))
    return {
        "pooled_r2": float(variance_weighted_r2(target, pred)),
        "equal_session_mean": float(report["session_mean_r2"]),
        "per_session_r2": {k: float(v) for k, v in sorted(report["per_session_r2"].items())},
        "n_faces": int(len(target)),
        "r2_convention": (
            "skeleton r2.py: float64 flattened global-mean SStot (pooled) + equal mean of "
            "per-session flattened R2; five-arm/formal12 used per-column mean(0) SStot — "
            "cross-family numbers indicative only"
        ),
    }


def _score_view(
    model: BTransformerUnifiedDecoderIdentity,
    ema: DecoderEMA | None,
    view: str,
    faces: dict[str, Any],
    sessions: list[str],
    device: torch.device,
    *,
    bridge_check: bool = False,
) -> dict[str, Any]:
    """Score RAW or EMA view; RAW parameters always restored afterwards."""
    if device.type == "cuda":
        torch.cuda.empty_cache()  # release training fragmentation before fp32 eval
    named = model.trainable_parameters()
    backup = {name: param.detach().clone() for name, param in named.items()}
    was_training = model.training
    try:
        if view == "EMA":
            plan.require(ema is not None and ema.n_updates > 0, "EMA view requested before any EMA update")
            with torch.no_grad():
                for name, param in named.items():
                    param.copy_(ema.shadow[name].to(device=param.device, dtype=param.dtype))
        report = _score_faces(model, faces, sessions, device, bridge_check=bridge_check)
        report["view"] = view
        return report
    finally:
        with torch.no_grad():
            for name, param in named.items():
                param.copy_(backup[name].to(device=param.device, dtype=param.dtype))
        model.train(was_training)


# ---------------------------------------------------------------------------
# proj_add brief: P norm / sparsity / effective rank
# ---------------------------------------------------------------------------


def proj_brief(model: BTransformerUnifiedDecoderIdentity, ema: DecoderEMA | None) -> dict[str, Any]:
    def _stats(weight: np.ndarray) -> dict[str, Any]:
        sv = np.linalg.svd(weight, compute_uv=False)
        pr = float((sv ** 2).sum() ** 2 / (sv ** 4).sum()) if (sv > 0).any() else 0.0
        absmax = float(np.abs(weight).max())
        return {
            "frobenius_norm": float(np.linalg.norm(weight)),
            "spectral_norm": float(sv[0]) if len(sv) else 0.0,
            "nuclear_norm": float(sv.sum()),
            "effective_rank_participation_ratio": pr,
            "per_output_row_l2": [float(np.linalg.norm(row)) for row in weight],
            "absmax": absmax,
            "share_below_1e-3_of_absmax": float((np.abs(weight) < 1e-3 * absmax).mean()),
            "share_below_1e-2_of_absmax": float((np.abs(weight) < 1e-2 * absmax).mean()),
            "singular_values": [float(v) for v in sv],
        }

    weight_raw = model.frontend.e0_proj.weight.detach().cpu().numpy().astype(np.float64)
    out: dict[str, Any] = {"shape": list(weight_raw.shape), "raw": _stats(weight_raw)}
    if ema is not None and ema.n_updates > 0:
        shadow = ema.shadow.get("frontend.e0_proj.weight")
        if shadow is not None:
            out["ema"] = _stats(shadow.cpu().numpy().astype(np.float64))
    return out


def proj_apply_stats(model, faces: dict[str, Any], sessions: list[str]) -> dict[str, Any]:
    weight = model.frontend.e0_proj.weight.detach()
    per_session = {}
    for s in sessions:
        e0 = torch.from_numpy(faces["banks"][s].E0)
        with torch.no_grad():
            pe0 = weight(e0.to(weight.device, dtype=weight.dtype))
        per_session[s] = {
            "pe0_frobenius": float(np.linalg.norm(pe0.detach().cpu().numpy())),
            "pe0_absmax": float(pe0.detach().abs().max().item()),
        }
    return {"pe0_per_session": per_session}


# ---------------------------------------------------------------------------
# Memory probe (TRN-6 evidence; decision is FIXED at micro 8 x accum 4)
# ---------------------------------------------------------------------------


def memory_probe(model, bank, X, y, device) -> dict[str, Any]:
    """TRN-6 probe: fwd+bwd peak memory/time at micro 8 and 16 (params restored)."""
    results = {}
    was_training = model.training
    model.train()
    named = model.trainable_parameters()
    backup = {name: param.detach().clone() for name, param in named.items()}
    for micro in (8, 16):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        xb = torch.from_numpy(X[:micro]).to(device)
        yb = torch.from_numpy(y[:micro] * SCALE).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
        t0 = time.monotonic()
        try:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(unit_dropout_seed(SEED, 0, 0))
            keep = whole_unit_dropout(
                torch.from_numpy(bank.unit_mask.copy()), p=plan.UNIT_DROPOUT, generator=generator
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(xb, bank, dropout_keep=keep)
                loss = nn.functional.mse_loss(pred.float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.trainable_parameters().values()), plan.GRAD_CLIP)
            opt.step()
            torch.cuda.synchronize()
            results[f"micro{micro}"] = {
                "peak_alloc_mib": torch.cuda.max_memory_allocated() / 2**20,
                "step_seconds": time.monotonic() - t0,
            }
        except torch.cuda.OutOfMemoryError:
            results[f"micro{micro}"] = {"oom": True}
        finally:
            del opt, xb, yb
            model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(backup[name].to(device=param.device, dtype=param.dtype))
    model.train(was_training)
    return results


# ---------------------------------------------------------------------------
# M-F250 L-axis partner: cross-cell consistency + delta reading
# ---------------------------------------------------------------------------


def _mf250_dirs() -> list[Path]:
    root = plan.RESULT_ROOT / "h1_matrix"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("M_F250_"))


def mf250_run_meta() -> tuple[Path | None, dict[str, Any] | None]:
    for d in reversed(_mf250_dirs()):
        meta_path = d / "run_meta.json"
        if meta_path.is_file():
            try:
                return d, json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None, None


def mf250_consistency(my_split: dict[str, Any], my_train_face_counts: dict[str, int] | None = None) -> dict[str, Any]:
    """Same CAL-2/M3, same holdout, same recipe/seed — only variable L."""
    out: dict[str, Any] = {
        "declaration": (
            "M-F700 and M-F250 form the L-axis controlled pair: same proj_add "
            "injection, same CAL-2/M3 fixed budget, same LODO holdout, same "
            "TRN-1 recipe, same seed 42 — 本 cell 与 M-F250 构成 L 轴受控对，"
            "唯一变量 L (700 vs 250)"
        ),
        "deterministic_guarantee": (
            "both cells consume h1_config.lodo_split() (frozen last-date rule) "
            "and adapters.build_h1_bank(budget=3) — identical by construction; "
            "the assertion below additionally checks the partner's sealed "
            "run_meta when it exists"
        ),
    }
    partner_dir, meta = mf250_run_meta()
    out["partner_results_dir"] = str(partner_dir) if partner_dir else None
    if meta is None:
        out["assertion"] = "PENDING_PARTNER_RUN_META (M-F250 not started/sealed yet; deterministic guarantee holds)"
        return out
    checks: dict[str, bool] = {
        "same_holdout_date": meta.get("lodo_split", {}).get("holdout_date") == my_split["holdout_date"],
        "same_holdout_sessions": meta.get("lodo_split", {}).get("holdout_sessions") == my_split["holdout_sessions"],
        "same_train_sessions": meta.get("lodo_split", {}).get("train_sessions") == my_split["train_sessions"],
        "same_identity_mode": meta.get("identity_mode") == "proj_add",
        "same_seed": int(meta.get("seed", -1)) == SEED,
        "same_epochs": int(meta.get("epochs", -1)) == EPOCHS,
    }
    if my_train_face_counts is not None and isinstance(meta.get("train_face_counts"), dict):
        checks["same_train_face_counts"] = {
            k: int(v) for k, v in meta["train_face_counts"].items()
        } == {k: int(v) for k, v in my_train_face_counts.items()}
    # budget evidence: the partner's build path (adapters.build_h1_bank) refuses
    # anything but budget 3 materializable, and both runs sealed CAL-2/M3 metas
    out["partner_checks"] = {k: bool(v) for k, v in checks.items()}
    out["assertion"] = "CONSISTENT" if all(bool(v) for v in checks.values()) else "MISMATCH"
    return out


def _read_epoch_metrics(dest: Path) -> dict[int, dict[str, Any]]:
    metrics: dict[int, dict[str, Any]] = {}
    path = dest / "metrics.jsonl"
    if not path.is_file():
        return metrics
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("event") == "epoch":
                metrics[int(row["epoch"])] = row
    return metrics


def l_axis_delta(my_receipt_numbers: dict[str, Any]) -> dict[str, Any]:
    """ΔR² = R²(700) − R²(250) on the LODO exam under the SAME SEL-2 rule.

    Reads the partner's metrics.jsonl (any schema generation that carries the
    per-epoch lodo_exam rows) and recomputes its pick under THIS cell's rule
    (exam EMA equal_session_mean earliest max) so the pair is compared under
    one selector even if the partner's own receipt predates the rule fix.
    """
    out: dict[str, Any] = {
        "definition": "ΔR² = R²(M-F700, L=700) − R²(M-F250, L=250), same injection/recipe/holdout/seed",
        "primary_statistic": "LODO exam EMA equal_session_mean at each cell's SEL-2 pick epoch",
    }
    partner = None
    for d in reversed(_mf250_dirs()):
        if (d / "metrics.jsonl").is_file():
            partner = d
            break
    if partner is None:
        out["status"] = "PENDING_MF250_SCORES (partner metrics.jsonl not present at receipt time; compute later)"
        return out
    theirs = _read_epoch_metrics(partner)
    if not theirs:
        out["status"] = "PENDING_MF250_SCORES (partner metrics.jsonl has no epoch rows yet)"
        return out
    n_epochs = len(theirs)

    def series(m: dict[int, dict[str, Any]], view_key: str, stat: str) -> dict[int, float]:
        return {
            e: float(m[e][view_key][stat])
            for e in m
            if isinstance(m[e].get(view_key), dict) and m[e][view_key].get(stat) is not None
        }

    def pick_epoch(series_map: dict[int, float]) -> int | None:
        finite = {int(e): float(v) for e, v in series_map.items() if math.isfinite(v)}
        if not finite:
            return None
        best = max(finite.values())
        tied = [e for e, v in finite.items() if abs(best - v) <= 1e-10]
        return min(tied)

    theirs_eq = series(theirs, "lodo_exam_ema", "equal_session_mean")
    theirs_pool = series(theirs, "lodo_exam_ema", "pooled_r2")
    p_theirs = pick_epoch(theirs_eq)
    if p_theirs is None:
        out["status"] = "PENDING_MF250_SCORES (partner exam EMA series empty)"
        return out
    mine_eq = my_receipt_numbers["exam_ema_equal_mean"]
    mine_pool = my_receipt_numbers["exam_ema_pooled"]
    p_mine = my_receipt_numbers["pick"]
    delta_pick_eq = float(mine_eq[p_mine] - theirs_eq[p_theirs])
    delta_pick_pool = float(mine_pool[p_mine] - theirs_pool[p_theirs])
    delta_endpoint_eq = None
    delta_endpoint_pool = None
    if EPOCHS in theirs_eq and EPOCHS in mine_eq:
        delta_endpoint_eq = float(mine_eq[EPOCHS] - theirs_eq[EPOCHS])
        delta_endpoint_pool = float(mine_pool[EPOCHS] - theirs_pool[EPOCHS])

    def band(delta: float | None) -> str | None:
        if delta is None:
            return None
        if delta <= 0.01:
            return (
                "L=250 near lossless (ΔR² ≤ 0.01): the H1 information horizon fits "
                "within 250 bins under the proj_add injection — take the short window"
            )
        if delta <= 0.02:
            return (
                "0.01 < ΔR² ≤ 0.02 (hole rule, matrix §2.1 revision): record the "
                "horizon cost and decide via the L=350 tiebreaker cell — if "
                "ΔR²(350) ≤ 0.01 take 350, else record the horizon and take 350"
            )
        if delta <= 0.05:
            return (
                "0.02 < ΔR² ≤ 0.05: horizon cost recorded — real information lives "
                "beyond 250 bins; weigh against the latency budget"
            )
        return (
            "ΔR² > 0.05: long-tail information SUBSTANTIALLY exists — read with the "
            "five-arm masking evidence (H1 delay resolution moves to SPD-C1/C2 "
            "streaming KV / local-bandwidth)"
        )

    out.update(
        {
            "status": f"COMPUTED (partner epochs available: {n_epochs}/24{'' if n_epochs >= 24 else ' — PARTNER STILL TRAINING, initial reading'})",
            "partner_results_dir": str(partner),
            "partner_pick_epoch_same_rule": p_theirs,
            "mine_pick_epoch": p_mine,
            "partner_exam_ema_equal_mean_at_pick": theirs_eq[p_theirs],
            "mine_exam_ema_equal_mean_at_pick": mine_eq[p_mine],
            "delta_r2_exam_ema_equal_mean_at_pick": delta_pick_eq,
            "delta_r2_exam_ema_pooled_at_pick": delta_pick_pool,
            "delta_r2_exam_ema_equal_mean_endpoint24": delta_endpoint_eq,
            "delta_r2_exam_ema_pooled_endpoint24": delta_endpoint_pool,
            "reading_primary": band(delta_pick_eq),
            "reading_endpoint24": band(delta_endpoint_eq),
            "bands_preregistered": (
                "ΔR² = R²(700) − R²(250): ≤0.01 near-lossless; 0.01–0.02 hole rule "
                "(L=350 tiebreaker, matrix §2.1 revision); 0.02–0.05 horizon cost "
                "recorded; >0.05 long-tail information substantially exists"
            ),
        }
    )
    return out


# ---------------------------------------------------------------------------
# Health adjudication (coordinator protocol, updated 2026-09-06 after
# M-F250@3e-4 failed its ep10 checkpoint — loss plateau 0.006911 never broke
# the 0.0065 gate): TRN-1's 3e-4 peak was never validated on H1 (formal12 used
# 1e-4). M-F700@3e-4 now doubles as the L=700 LR control: if it also plateaus,
# "3e-4 does not fit H1" holds ACROSS L. Ruling at the ep10 checkpoint only:
#   CONTINUE  iff train_mse <= 0.0065 AND examRAW >= -0.007
#   EARLY FAIL iff train_mse pinned at 0.0069 +/- 0.0001 OR examRAW declining
#             3 consecutive epochs below -0.01
# On FAIL: seal the receipt, release GPU0, await restart authorization (likely
# the LR 1e-4 variant, forming a clean 1e-4 L-axis pair with M-F250-lr1e4).
# EMA washout artifacts never rule; no self-authorized hyperparameter restart.
# ---------------------------------------------------------------------------

HEALTH_CHECKPOINT_EPOCH = 10
HEALTH_PLATEAU_TOL = 1e-4
HEALTH_PLATEAU_MIN_EPOCHS = 4
HEALTH_RAW_DECLINE_MIN_EPOCHS = 3
HEALTH_LOSS_BREAK_TARGET = 0.0065
HEALTH_RAW_FLOOR = -0.007
HEALTH_PIN_CENTER = 0.0069
HEALTH_PIN_TOL = 1e-4
HEALTH_RAW_DECLINE_BELOW = -0.01
HEALTH_RESTART_HYPOTHESIS = (
    "restart at LR peak 1e-4 (formal12 caliber) — RECORDED PENDING APPROVAL; "
    "no self-authorized hyperparameter restart in this cell; a 1e-4 M-F700 "
    "would form the clean L-axis pair with the M-F250-lr1e-4 retry"
)


def _health_state(
    epoch: int,
    train_mse_series: dict[int, float],
    exam_raw_series: dict[int, float],
) -> dict[str, Any]:
    epochs = sorted(e for e in train_mse_series if e <= epoch)
    losses = [float(train_mse_series[e]) for e in epochs]
    raws = [float(exam_raw_series[e]) for e in epochs]
    plateau = False
    if len(losses) >= HEALTH_PLATEAU_MIN_EPOCHS:
        tail = losses[-HEALTH_PLATEAU_MIN_EPOCHS:]
        plateau = (max(tail) - min(tail)) <= HEALTH_PLATEAU_TOL
    pinned_at_0069 = False
    if len(losses) >= HEALTH_PLATEAU_MIN_EPOCHS:
        tail = losses[-HEALTH_PLATEAU_MIN_EPOCHS:]
        pinned_at_0069 = all(abs(v - HEALTH_PIN_CENTER) <= HEALTH_PIN_TOL for v in tail)
    decline3 = False
    if len(raws) >= HEALTH_RAW_DECLINE_MIN_EPOCHS:
        tail = raws[-HEALTH_RAW_DECLINE_MIN_EPOCHS:]
        decline3 = all(tail[i + 1] < tail[i] for i in range(len(tail) - 1))
    decline3_below = bool(decline3 and all(v < HEALTH_RAW_DECLINE_BELOW for v in raws[-HEALTH_RAW_DECLINE_MIN_EPOCHS:]))
    return {
        "epoch": epoch,
        "train_mse": losses[-1] if losses else None,
        "exam_raw_pooled": raws[-1] if raws else None,
        "loss_plateau_4ep": plateau,
        "loss_pinned_0p0069": pinned_at_0069,
        "exam_raw_decline_3ep": decline3,
        "exam_raw_decline_3ep_below_minus0p01": decline3_below,
        "fail_triggers_present": bool(pinned_at_0069 or decline3_below),
    }


def health_adjudication(
    epoch: int,
    train_mse_series: dict[int, float],
    exam_raw_series: dict[int, float],
) -> dict[str, Any]:
    state = _health_state(epoch, train_mse_series, exam_raw_series)
    state.update(
        {
            "protocol": (
                "coordinator 2026-09-06 (updated after M-F250@3e-4 ep10 FAIL at "
                "plateau 0.006911): ruling at ep10 only; CONTINUE iff "
                "train_mse <= 0.0065 AND examRAW >= -0.007; EARLY FAIL iff "
                "train_mse pinned at 0.0069+/-0.0001 OR examRAW declining 3 "
                "consecutive epochs below -0.01; EMA washout never rules; no "
                "self-authorized restart; M-F700@3e-4 doubles as the L=700 LR "
                "control (a plateau here makes '3e-4 unfit for H1' hold across L)"
            ),
            "ruling": "NO_RULING_BEFORE_EP10" if epoch < HEALTH_CHECKPOINT_EPOCH else None,
        }
    )
    if epoch < HEALTH_CHECKPOINT_EPOCH:
        return state
    losses = [float(train_mse_series[e]) for e in sorted(train_mse_series) if e <= epoch]
    raws = [float(exam_raw_series[e]) for e in sorted(exam_raw_series) if e <= epoch]
    loss_ok = losses[-1] <= HEALTH_LOSS_BREAK_TARGET
    raw_ok = raws[-1] >= HEALTH_RAW_FLOOR
    if state["fail_triggers_present"]:
        state.update(
            {
                "loss_break_le_0p0065": loss_ok,
                "exam_raw_ge_minus0p007": raw_ok,
                "ruling": "EARLY_FAIL",
                "restart_hypothesis_pending_approval": HEALTH_RESTART_HYPOTHESIS,
            }
        )
    else:
        state.update(
            {
                "loss_break_le_0p0065": loss_ok,
                "exam_raw_ge_minus0p007": raw_ok,
                "ruling": (
                    "CONTINUE (no fail trigger at ep10)"
                    if (loss_ok and raw_ok)
                    else "EARLY_FAIL (continue-conditions not met and no recovery evidence)"
                ),
            }
        )
    return state


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_train(dest: Path, faces: dict[str, Any], micro_override: int | None, lr_peak: float | None = None) -> dict[str, Any]:
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)

    # LR peak: default = the frozen TRN-1 3e-4; the 1e-4 value may ONLY enter
    # through an explicit authorized restart (coordinator approval recorded in
    # run_meta.lr_authorization) — never a quiet edit.
    if lr_peak is None:
        lr_peak = plan.LR_PEAK
    plan.require(
        float(lr_peak) in (3.0e-4, 1.0e-4),
        f"lr_peak {lr_peak} outside the two authorized calibers {{3e-4 TRN-1, 1e-4 formal12 restart variant}}",
    )

    cell = cell_by_id(CELL_ID)
    geometry = cell_geometry(cell)
    model = BTransformerUnifiedDecoderIdentity(
        geometry, seed=SEED, override_prefix=0, identity_mode=cell.identity_mode
    ).to(device)
    meta = model.init_meta
    plan.require(meta["identity_mode"] == "proj_add", "wrong identity mode")
    plan.require(meta["token_in"] == 20 and meta["proj_out_dim"] == 16, "proj_add geometry drift")
    plan.require(
        model.window == WINDOW and model.l_in == WINDOW and model.window_override is None,
        "full-window drift: the settled 700 window must be used with NO override",
    )

    # L-axis controlled-pair assertion: same parameterization count as M-F250
    # (proj_add is frontend-L-independent; the PE is a non-persistent buffer).
    partner_cell = cell_by_id("M-F250")
    partner_model = BTransformerUnifiedDecoderIdentity(
        cell_geometry(partner_cell), seed=SEED, override_prefix=0,
        override_window=partner_cell.L, identity_mode=partner_cell.identity_mode,
    )
    plan.require(
        partner_model.init_meta["param_count"] == meta["param_count"],
        "M-F700/M-F250 parameter-count parity failed — the L-axis pair must differ "
        "only in L (proj_add is L-independent; PE is a sinusoidal buffer)",
    )
    param_parity = {
        "mf700_param_count": int(meta["param_count"]),
        "mf250_param_count": int(partner_model.init_meta["param_count"]),
        "equal": True,
    }
    del partner_model

    # startup sanity: causality on a real holdout window
    exam0_s = faces["exam"]["sessions"][0]
    causal = model.causal_check(torch.from_numpy(faces["exam"]["X"][exam0_s][:2]).to(device), faces["exam"]["banks"][exam0_s])
    plan.require(causal["passed"], "causal_check failed on real data")

    # banks -> device tensors kept per session (E0/carrier are static)
    train_sessions = faces["train"]["sessions"]
    n_train_faces = int(sum(len(faces["train"]["X"][s]) for s in train_sessions))
    plan.require(
        n_train_faces == N_TRAIN_FACES_EXPECTED,
        f"LODO train face count {n_train_faces} != {N_TRAIN_FACES_EXPECTED} (inventory)",
    )

    # TRN-6: FIXED decision micro 8 x accum 4 (formal12 recorded L=700 caliber);
    # the probe below records on-card peaks for evidence only.
    probe_s = train_sessions[0]
    probe = memory_probe(model, faces["train"]["banks"][probe_s], faces["train"]["X"][probe_s], faces["train"]["y"][probe_s], device)
    if micro_override is not None:
        micro = int(micro_override)
        accum = EFFECTIVE_BATCH // micro
    else:
        micro, accum = 8, 4
    plan.require(micro * accum == EFFECTIVE_BATCH, "micro x accum != effective batch 32")

    # update caliber: measured from THIS run's LODO face count AND batching
    # structure — batches never mix sessions, so the true optimizer steps/epoch
    # is sum_s ceil(n_s/32) = 597, NOT ceil(18935/32) = 592 (matrix §0 revision;
    # M-F250 attempt3 recorded caliber 597/ep -> 14328 total; attempt2's
    # lr_total error must not repeat). The LR schedule steps on OPTIMIZER
    # steps (micro 8 x accum 4 -> 4 micro fwd/bwd per schedule step). The
    # formal full-13-session caliber 731/ep is recorded for reference only.
    updates_per_epoch = int(
        sum(math.ceil(len(faces["train"]["X"][s]) / EFFECTIVE_BATCH) for s in train_sessions)
    )
    warmup_updates = updates_per_epoch * plan.WARMUP_EPOCHS
    total_updates = updates_per_epoch * EPOCHS
    formal = plan.recipe_updates("h1", epochs=EPOCHS)
    naive_ceiling = int(math.ceil(n_train_faces / EFFECTIVE_BATCH))
    plan.require(
        naive_ceiling <= updates_per_epoch <= naive_ceiling + len(train_sessions),
        f"per-session update caliber sanity failed: {updates_per_epoch} vs naive {naive_ceiling}",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr_peak), weight_decay=plan.WEIGHT_DECAY,
        betas=(0.9, 0.999), eps=1e-8,
    )
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    rng = np.random.default_rng(SEED)

    split = faces["split"]
    variant_picks = {k: list(v) for k, v in PICKS.items()}
    if float(lr_peak) == 1.0e-4:
        variant_picks["TRN"] = [LR_1E4_VARIANT_MARKER] + variant_picks["TRN"]
    _seal(
        dest / "run_meta.json",
        {
            "schema": "btransform_unified_v1_mf700_run_meta",
            "cell": CELL,
            "cell_id": CELL_ID,
            "lr_peak": float(lr_peak),
            "lr_variant": LR_1E4_VARIANT_MARKER if float(lr_peak) == 1.0e-4 else None,
            "lr_authorization": LR_AUTHORIZATION_NOTE if float(lr_peak) == 1.0e-4 else None,
            "matrix_doc": (
                "MATRIX_H1_L_IDENTITY_V1_20260906.md §2 revision (baseline cell "
                "formalized by REVIEW_MATRIX_H1_L_IDENTITY_V1_20260906 blocking "
                "item C; blocking items A/B: CAL-2/M3 + SEL-2 holdout-exam pick)"
            ),
            "utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "geometry": geometry,
            "window": model.window,
            "l_in": model.l_in,
            "window_override": None,
            "identity_mode": model.identity_mode,
            "init_meta": meta,
            "param_count": meta["param_count"],
            "proj_param_count": meta.get("proj_param_count"),
            "l_axis_param_parity": param_parity,
            "causal_check_startup": causal,
            "lodo_split": split,
            "train_face_counts": {s: int(len(faces["train"]["X"][s])) for s in train_sessions},
            "exam_face_counts": {s: int(len(faces["exam"]["X"][s])) for s in faces["exam"]["sessions"]},
            "selection_face_counts": {s: int(len(faces["sel"]["X"][s])) for s in faces["sel"]["sessions"]},
            "n_train_faces": n_train_faces,
            "cal": {
                "mode": "CAL-2 fixed M3 (matrix §0 revision 2026-09-06, blocking item A)",
                "budget": h1_config.DEPLOY_BUDGET,
                "budgets_available": [h1_config.DEPLOY_BUDGET],
                "cal1_rotation": "not_used — CAL-1 {7,5,4,3} deferred as a post-matrix "
                "upgrade cell (adapter stub: NWB re-trialization for 7/5/4 not wired)",
            },
            "microbatch_probe": probe,
            "microbatch_decision": {
                "micro": micro, "accum": accum, "effective_batch": EFFECTIVE_BATCH,
                "rule": "FIXED micro 8 x accum 4 at L=700 (formal12 recorded caliber); "
                "probe recorded for evidence only",
            },
            "updates_per_epoch": updates_per_epoch,
            "total_updates": total_updates,
            "warmup_updates": warmup_updates,
            "formal_update_caliber_recorded": formal,
            "update_caliber_note": (
                "per-session batching (batches never mix sessions): true optimizer "
                "steps/epoch = sum_s ceil(n_s/32) = 597 (M-F250 attempt3 recorded "
                "caliber; NOT the naive ceil(18935/32)=592 — attempt2's lr_total "
                "error); the LR schedule steps on OPTIMIZER steps (micro 8 x accum 4 "
                "advances the cosine once per 4 micro fwd/bwd); warmup=1 epoch (597 "
                "steps), cosine span=24 epochs (14328 steps); the formal "
                "full-13-session caliber 731 upd/ep is reference only"
            ),
            "unit_dropout_p": plan.UNIT_DROPOUT,
            "unit_dropout_domain_meta": UNIT_DROPOUT_DOMAIN_META,
            "eval_batch": EVAL_BATCH,
            "health_protocol": {
                "checkpoint_epoch": HEALTH_CHECKPOINT_EPOCH,
                "plateau_tol": HEALTH_PLATEAU_TOL,
                "plateau_min_epochs": HEALTH_PLATEAU_MIN_EPOCHS,
                "raw_decline_min_epochs": HEALTH_RAW_DECLINE_MIN_EPOCHS,
                "loss_break_target": HEALTH_LOSS_BREAK_TARGET,
                "restart_hypothesis_pending_approval": HEALTH_RESTART_HYPOTHESIS,
                "note": (
                    "coordinator 2026-09-06 (same as M-F250): TRN-1 3e-4 was never "
                    "validated on H1 (formal12 used 1e-4); ep10 checkpoint ruling"
                ),
            },
            "batch_rule": (
                "five-arm style: per epoch, session order shuffled then faces permuted "
                "within session (np default_rng(42), consumption order recorded); "
                "batches never mix sessions (bank E0 is per-session static)"
            ),
            "scale": {"target_multiplier": SCALE, "scoring": "pred/20 vs native (ratio bridge)"},
            "epochs": EPOCHS,
            "budget_seconds": BUDGET_SECONDS,
            "gpu_uuid": GPU0_UUID,
            "gpu1_owner": "M-F250 session; never touched by this cell",
            "arbitration_log": str(ARBITRATION_LOG),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "source_cache_sha256": h1_config.SOURCE_CACHE_SHA256,
            "c2_payload_sha256": h1_config.C2_M3_PAYLOAD_SHA256,
            "c2_ckpt_sha256": h1_config.C2_CKPT_SHA256,
            "picks": variant_picks,
            "references_not_gates": REFERENCES,
        },
    )

    metrics_path = dest / "metrics.jsonl"
    heartbeat = dest / "heartbeat.json"
    started = time.monotonic()
    deadline = started + BUDGET_SECONDS
    global_step = 0
    ema_series: dict[int, dict[str, Any]] = {}
    raw_series: dict[int, dict[str, Any]] = {}
    exam_ema_series: dict[int, dict[str, Any]] = {}
    exam_raw_series: dict[int, dict[str, Any]] = {}
    train_mse_series: dict[int, float] = {}
    lr_series: dict[int, float] = {}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_t0 = time.monotonic()
        order = list(train_sessions)
        rng.shuffle(order)
        losses: list[float] = []
        batch_id = -1
        for session in order:
            X, y = faces["train"]["X"][session], faces["train"]["y"][session]
            bank = faces["train"]["banks"][session]
            idx = rng.permutation(len(X))
            for offset in range(0, len(idx), EFFECTIVE_BATCH):
                if time.monotonic() >= deadline:
                    _seal(
                        dest / "budget_hit.json",
                        {
                            "schema": "btransform_unified_v1_mf700_budget_hit",
                            "cell": CELL,
                            "epoch": epoch,
                            "global_step": global_step,
                            "budget_seconds": BUDGET_SECONDS,
                            "elapsed_seconds": time.monotonic() - started,
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "note": "6h GPU budget hit before finishing 24 epochs; no rerun authorized",
                        },
                    )
                    raise RuntimeError("M-F700 6h GPU budget hit")
                take = idx[offset : offset + EFFECTIVE_BATCH]
                micro_losses = []
                for m_off in range(0, len(take), micro):
                    m_take = take[m_off : m_off + micro]
                    batch_id += 1
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(unit_dropout_seed(SEED, epoch, batch_id))
                    keep = whole_unit_dropout(
                        torch.from_numpy(bank.unit_mask.copy()), p=plan.UNIT_DROPOUT, generator=generator
                    )
                    xb = torch.from_numpy(X[m_take]).to(device)
                    yb = torch.from_numpy(y[m_take] * SCALE).to(device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred = model(xb, bank, dropout_keep=keep)
                        loss = nn.functional.mse_loss(pred.float(), yb) / accum
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    micro_losses.append(float(loss.detach().cpu()) * accum)
                # schedule steps on OPTIMIZER steps (one per effective batch),
                # never on micro batches — micro 8 x accum 4 must not advance
                # the cosine 4x too fast (M-F250 attempt2 lr_total lesson)
                global_step += 1
                lr = warmup_cosine_lr(
                    global_step, total_steps=total_updates, warmup_steps=warmup_updates,
                    peak=float(lr_peak), min_factor=plan.LR_MIN_FACTOR,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                nn.utils.clip_grad_norm_(list(model.trainable_parameters().values()), plan.GRAD_CLIP)
                optimizer.step()
                ema.update_after_step(model)
                losses.extend(micro_losses)
                if global_step % 200 == 0:
                    _append_jsonl(
                        metrics_path,
                        {
                            "event": "step", "cell": CELL, "epoch": epoch,
                            "global_step": global_step, "loss": micro_losses[-1] if micro_losses else None,
                            "lr": float(lr), "ema_updates": ema.n_updates, "unix": time.time(),
                        },
                    )
                    _write_json(
                        heartbeat,
                        {
                            "cell": CELL, "epoch": epoch, "global_step": global_step,
                            "lr": float(lr), "ema_updates": ema.n_updates, "unix": time.time(),
                            "gpu_uuid": GPU0_UUID,
                        },
                    )

        exam_raw = _score_view(model, ema, "RAW", faces["exam"], faces["exam"]["sessions"], device)
        exam_ema = _score_view(model, ema, "EMA", faces["exam"], faces["exam"]["sessions"], device, bridge_check=True)
        sel_raw = _score_view(model, ema, "RAW", faces["sel"], faces["sel"]["sessions"], device)
        sel_ema = _score_view(model, ema, "EMA", faces["sel"], faces["sel"]["sessions"], device)
        raw_series[epoch] = sel_raw
        ema_series[epoch] = sel_ema
        exam_raw_series[epoch] = exam_raw
        exam_ema_series[epoch] = exam_ema
        train_mse_series[epoch] = float(np.mean(losses)) if losses else float("nan")
        lr_series[epoch] = float(lr)
        row = {
            "event": "epoch", "epoch": epoch, "cell": CELL,
            "train_mse": train_mse_series[epoch],
            "lodo_exam_raw": exam_raw, "lodo_exam_ema": exam_ema,
            "sel2908_raw": sel_raw, "sel2908_ema": sel_ema,
            "lr": float(lr), "seconds": time.monotonic() - epoch_t0,
            "global_step": global_step, "ema_updates": ema.n_updates,
            "unix": time.time(),
        }
        _append_jsonl(metrics_path, row)
        _write_json(heartbeat, row)
        print(
            f"[mf700] ep{epoch} loss={train_mse_series[epoch]:.5f} "
            f"examEMA(eq/pool)={exam_ema['equal_session_mean']:.4f}/{exam_ema['pooled_r2']:.4f} "
            f"examRAW(pool)={exam_raw['pooled_r2']:.4f} "
            f"sel2908EMA={sel_ema['pooled_r2']:.4f} "
            f"({row['seconds']:.0f}s)",
            flush=True,
        )
        # health adjudication (coordinator protocol; ruling only at ep10)
        health = health_adjudication(
            epoch, train_mse_series,
            {e: exam_raw_series[e]["pooled_r2"] for e in exam_raw_series},
        )
        _append_jsonl(dest / "health.jsonl", health)
        print(
            f"[mf700] health ep{epoch}: plateau={health['loss_plateau_4ep']} "
            f"decline={health['exam_raw_decline_3ep']} ruling={health['ruling']}",
            flush=True,
        )
        if health["ruling"] == "EARLY_FAIL":
            _seal(
                dest / "early_fail_receipt.json",
                {
                    "schema": "btransform_unified_v1_mf700_early_fail",
                    "cell": CELL,
                    "cell_id": CELL_ID,
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "epoch_at_fail": epoch,
                    "health_protocol": health["protocol"],
                    "health_table": [
                        json.loads(line)
                        for line in (dest / "health.jsonl").read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ],
                    "train_mse_by_epoch": train_mse_series,
                    "lodo_exam_raw_pooled_by_epoch": {e: exam_raw_series[e]["pooled_r2"] for e in exam_raw_series},
                    "lodo_exam_ema_pooled_by_epoch": {e: exam_ema_series[e]["pooled_r2"] for e in exam_ema_series},
                    "lodo_exam_ema_equal_mean_by_epoch": {e: exam_ema_series[e]["equal_session_mean"] for e in exam_ema_series},
                    "sel2908_ema_pooled_by_epoch": {e: ema_series[e]["pooled_r2"] for e in ema_series},
                    "restart_hypothesis_pending_approval": HEALTH_RESTART_HYPOTHESIS,
                    "note": (
                        "early FAIL per the coordinator health protocol (3e-4 never "
                        "validated on H1); GPU0 released; NO self-authorized restart — "
                        "the LR 1e-4 restart hypothesis is recorded pending approval"
                    ),
                },
            )
            print("[mf700] EARLY_FAIL at ep10 checkpoint — receipt sealed, GPU0 released", flush=True)
            return {
                "schema": "btransform_unified_v1_mf700_train_receipt",
                "status": "EARLY_FAIL",
                "cell": CELL,
                "cell_id": CELL_ID,
                "lr_peak": float(lr_peak),
                "lr_variant": LR_1E4_VARIANT_MARKER if float(lr_peak) == 1.0e-4 else None,
                "epochs_completed": list(range(1, epoch + 1)),
                "train_mse": train_mse_series,
                "lodo_exam_ema_equal_mean_by_epoch": {e: exam_ema_series[e]["equal_session_mean"] for e in exam_ema_series},
                "lodo_exam_raw_by_epoch": {e: exam_raw_series[e]["pooled_r2"] for e in exam_raw_series},
                "restart_hypothesis_pending_approval": HEALTH_RESTART_HYPOTHESIS,
            }
        ckpt = {
            "schema": "btransform_unified_v1_mf700_ckpt",
            "cell": CELL, "epoch": epoch, "global_step": global_step, "seed": SEED,
            "raw_state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "ema": ema.state_dict(), "lr": float(lr),
        }
        torch.save(ckpt, dest / f"epoch_{epoch:03d}.pt")

    plan.require(
        global_step == total_updates,
        f"update accounting drift: optimizer steps {global_step} != scheduled total {total_updates}",
    )

    summary = {
        "schema": "btransform_unified_v1_mf700_train_receipt",
        "status": "COMPLETED",
        "cell": CELL,
        "cell_id": CELL_ID,
        "lr_peak": float(lr_peak),
        "lr_variant": LR_1E4_VARIANT_MARKER if float(lr_peak) == 1.0e-4 else None,
        "seed": SEED,
        "gpu_uuid": GPU0_UUID,
        "epochs_completed": list(range(1, EPOCHS + 1)),
        "global_updates": global_step,
        "ema_updates": ema.n_updates,
        "train_mse": train_mse_series,
        "lr_at_epoch_end": lr_series,
        "lodo_exam_ema_by_epoch": {e: exam_ema_series[e]["pooled_r2"] for e in exam_ema_series},
        "lodo_exam_ema_equal_mean_by_epoch": {e: exam_ema_series[e]["equal_session_mean"] for e in exam_ema_series},
        "lodo_exam_raw_by_epoch": {e: exam_raw_series[e]["pooled_r2"] for e in exam_raw_series},
        "sel2908_ema_by_epoch": {e: ema_series[e]["pooled_r2"] for e in ema_series},
        "sel2908_ema_equal_mean_by_epoch": {e: ema_series[e]["equal_session_mean"] for e in ema_series},
        "sel2908_raw_by_epoch": {e: raw_series[e]["pooled_r2"] for e in raw_series},
        "elapsed_s": time.monotonic() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "picks": PICKS,
    }
    _seal(dest / "train_receipt.json", summary)
    return summary


# ---------------------------------------------------------------------------
# SEL reports + cell receipt
# ---------------------------------------------------------------------------


def _sel2_pick(series: dict[int, float]) -> int:
    finite = {int(e): float(v) for e, v in series.items() if math.isfinite(v)}
    plan.require(bool(finite), "no finite SEL-2 scores")
    best = max(finite.values())
    tied = [e for e, v in finite.items() if abs(best - v) <= 1e-10]
    return min(tied)


def _last_k(series: dict[int, float], lo: int, hi: int) -> dict[str, Any]:
    vals = [float(series[e]) for e in range(lo, hi + 1) if e in series]
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()) if arr.size else None,
        "std": float(arr.std()) if arr.size else None,
        "range": [float(arr.min()), float(arr.max())] if arr.size else None,
        "values": vals,
    }


def run_cell_receipt(dest: Path, faces: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    metrics = _read_epoch_metrics(dest)
    plan.require(bool(metrics), "no epoch rows in metrics.jsonl — cannot build the cell receipt")

    # SEL-2 (matrix §0 revision): earliest max on the LODO-exam EMA
    # equal_session_mean series. The 2,908 face is auxiliary only.
    exam_eq = {e: float(metrics[e]["lodo_exam_ema"]["equal_session_mean"]) for e in metrics}
    exam_pool = {e: float(metrics[e]["lodo_exam_ema"]["pooled_r2"]) for e in metrics}
    exam_raw_pool = {e: float(metrics[e]["lodo_exam_raw"]["pooled_r2"]) for e in metrics}
    exam_raw_eq = {e: float(metrics[e]["lodo_exam_raw"]["equal_session_mean"]) for e in metrics}
    sel_ema = {e: float(metrics[e]["sel2908_ema"]["pooled_r2"]) for e in metrics}
    sel_ema_eq = {e: float(metrics[e]["sel2908_ema"]["equal_session_mean"]) for e in metrics}
    sel_raw = {e: float(metrics[e]["sel2908_raw"]["pooled_r2"]) for e in metrics}

    pick = _sel2_pick(exam_eq)

    # reload the SEL-2 epoch checkpoint for the proj_add P brief + full detail
    ckpt = torch.load(dest / f"epoch_{pick:03d}.pt", map_location=device, weights_only=False)
    cell_obj = cell_by_id(CELL_ID)
    model = BTransformerUnifiedDecoderIdentity(
        cell_geometry(cell_obj), seed=SEED, override_prefix=0,
        identity_mode=cell_obj.identity_mode,
    ).to(device)
    model.load_state_dict(ckpt["raw_state_dict"])
    ema = DecoderEMA.__new__(DecoderEMA)
    ema.decay = plan.EMA_DECAY
    ema.n_updates = int(ckpt["ema"]["n_updates"])
    ema.shadow = {k: v.detach().clone() for k, v in ckpt["ema"]["shadow"].items()}
    p_brief = proj_brief(model, ema)
    p_apply = proj_apply_stats(model, faces["sel"], faces["sel"]["sessions"])
    # EMA view of the pick epoch on both surfaces (full detail rows)
    pick_sel_detail = _score_view(model, ema, "EMA", faces["sel"], faces["sel"]["sessions"], device)
    pick_exam_detail = _score_view(model, ema, "EMA", faces["exam"], faces["exam"]["sessions"], device)
    _write_json(dest / "sel2_pick_detail.json", {"sel2908_ema": pick_sel_detail, "lodo_exam_ema": pick_exam_detail})

    delta_block = l_axis_delta(
        {
            "pick": pick,
            "exam_ema_equal_mean": exam_eq,
            "exam_ema_pooled": exam_pool,
        }
    )

    # LR variant marker (authorized 1e-4 restart) read from the sealed run_meta
    lr_peak_used = None
    run_meta_path = dest / "run_meta.json"
    if run_meta_path.is_file():
        try:
            lr_peak_used = float(json.loads(run_meta_path.read_text(encoding="utf-8")).get("lr_peak", plan.LR_PEAK))
        except Exception:
            lr_peak_used = plan.LR_PEAK
    receipt_picks = {k: list(v) for k, v in PICKS.items()}
    lr_variant_field = None
    if lr_peak_used is not None and abs(lr_peak_used - 1.0e-4) < 1e-12:
        lr_variant_field = LR_1E4_VARIANT_MARKER
        receipt_picks["TRN"] = [LR_1E4_VARIANT_MARKER] + receipt_picks["TRN"]

    receipt = {
        "schema": "btransform_unified_v1_mf700_cell_receipt",
        "cell": CELL,
        "cell_id": CELL_ID,
        "lr_peak": lr_peak_used,
        "lr_variant": lr_variant_field,
        "lr_authorization": LR_AUTHORIZATION_NOTE if lr_variant_field else None,
        "matrix": (
            "H1 L x identity (MATRIX_H1_L_IDENTITY_V1_20260906 §2 revision) — "
            "full-window L-axis baseline cell, proj_add"
        ),
        "utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "gate": "NONE (matrix baseline cell; readings feed the preregistered L-axis interpretation)",
        "references_not_gates": REFERENCES,
        "lodo_split": faces["split"],
        "l_axis_pair": {
            "declaration": (
                "本 cell 与 M-F250 构成 L 轴受控对，唯一变量 L（700 vs 250）：同 proj_add "
                "注入、同 CAL-2/M3、同 LODO holdout、同 TRN-1 配方、同 seed 42"
            ),
            "consistency": mf250_consistency(
                faces["split"],
                {s: int(len(faces["train"]["X"][s])) for s in faces["train"]["sessions"]},
            ),
            "delta_reading": delta_block,
            "this_results_dir": str(dest),
            "partner_results_dir": delta_block.get("partner_results_dir"),
        },
        "sel": {
            "SEL-2_rule": (
                "earliest max on the LODO-exam EMA equal_session_mean series "
                "(holdout date 1925-01-20 only; ties<=1e-10 -> earliest; rule before "
                "numbers; the 2,908 same-source face is auxiliary and never picks)"
            ),
            "SEL-2_epoch_pick": pick,
            "SEL-2_pick_exam_ema_equal_mean": exam_eq[pick],
            "SEL-2_pick_exam_ema_pooled": exam_pool[pick],
            "SEL-2_pick_exam_ema_per_session": metrics[pick]["lodo_exam_ema"]["per_session_r2"],
            "SEL-2_pick_sel2908_ema_pooled": sel_ema[pick],
            "SEL-2_pick_sel2908_ema_equal_mean": sel_ema_eq[pick],
            "SEL-1_endpoint24_exam_ema_equal_mean": exam_eq.get(EPOCHS),
            "SEL-1_endpoint24_exam_ema_pooled": exam_pool.get(EPOCHS),
            "SEL-1_endpoint24_exam_raw_pooled": exam_raw_pool.get(EPOCHS),
            "SEL-1_endpoint24_exam_raw_equal_mean": exam_raw_eq.get(EPOCHS),
            "SEL-1_endpoint24_sel2908_ema_pooled": sel_ema.get(EPOCHS),
            "SEL-1_last4_exam_ema_equal_mean": _last_k(exam_eq, EPOCHS - 3, EPOCHS),
            "SEL-1_last8_exam_ema_equal_mean": _last_k(exam_eq, EPOCHS - 7, EPOCHS),
            "SEL-1_last4_sel2908_ema_pooled": _last_k(sel_ema, EPOCHS - 3, EPOCHS),
            "SEL-4_note": "pooled and session-mean reported for every surface/epoch (metrics.jsonl); never subtract across surfaces",
        },
        "epoch_curves": {
            "lodo_exam_ema_equal_mean": exam_eq,
            "lodo_exam_ema_pooled": exam_pool,
            "lodo_exam_raw_pooled": exam_raw_pool,
            "sel2908_ema_pooled": sel_ema,
            "sel2908_ema_equal_mean": sel_ema_eq,
            "sel2908_raw_pooled": sel_raw,
            "summary_epochs": {
                str(e): {
                    "train_mse": metrics[e]["train_mse"],
                    "lodo_exam_ema_equal_mean": exam_eq.get(e),
                    "lodo_exam_ema_pooled": exam_pool.get(e),
                    "sel2908_ema_pooled": sel_ema.get(e),
                    "lr": metrics[e]["lr"],
                }
                for e in (1, 12, 24) if e in metrics
            },
        },
        "proj_add_brief": {
            "note": "P = frontend.e0_proj.weight [16,700]; RAW = SEL-2 pick epoch RAW, EMA = same epoch EMA shadow",
            **p_brief,
            **p_apply,
        },
        "picks": receipt_picks,
        "note_six_rows": NOTE_SIX_ROWS,
        "artifacts": {
            "run_meta": "run_meta.json (sealed)",
            "train_receipt": "train_receipt.json (sealed)",
            "metrics": "metrics.jsonl",
            "sel2_pick_detail": "sel2_pick_detail.json",
            "checkpoints": "epoch_001.pt..epoch_024.pt",
            "arbitration": str(ARBITRATION_LOG),
        },
    }
    _write_json(dest / "cell_receipt_content.json", receipt)
    _seal(dest / "cell_receipt.json", receipt)
    return receipt


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="H1 matrix baseline cell M-F700 (proj_add, L=700 full window, LODO)")
    parser.add_argument("--stage", choices=["preflight", "build", "train", "receipt", "all"], default="all")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--micro", type=int, default=None, help="override microbatch (default: fixed 8, accum 4)")
    parser.add_argument(
        "--lr-peak", type=float, default=None,
        help="LR peak override; default = frozen TRN-1 3e-4; 1e-4 ONLY under an "
        "explicit coordinator-approved restart (recorded in run_meta.lr_authorization)",
    )
    args = parser.parse_args()

    if args.stage in ("train", "all") and os.environ.get(TRAIN_ENV_FLAG) != "1":
        print(f"REFUSED: training requires {TRAIN_ENV_FLAG}=1", file=sys.stderr)
        return 2
    if args.stage in ("train", "all", "receipt"):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            print("REFUSED: CUDA_VISIBLE_DEVICES must be pinned to '0' (GPU0)", file=sys.stderr)
            return 2

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"[mf700] dest = {args.dest}", flush=True)

    if args.stage in ("preflight", "all"):
        pre = gpu_preflight(args.dest / "preflight_gpu.json")
        print(f"[mf700] preflight ok={pre['ok']} foreign={pre['foreign_pids_on_gpu0']}", flush=True)
        if not pre["ok"]:
            _seal(
                args.dest / "BLOCKED.json",
                {
                    "schema": "btransform_unified_v1_mf700_blocked",
                    "cell": CELL,
                    "reason": "GPU0 not clean at preflight (foreign pid >500 MiB, CUDA_VISIBLE_DEVICES!=0, or five-arm pid still present)",
                    "preflight": pre,
                    "utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            return 2

    faces = None
    if args.stage in ("build", "train", "receipt", "all"):
        t0 = time.monotonic()
        faces = build_faces()
        print(
            f"[mf700] faces built in {time.monotonic()-t0:.0f}s: "
            f"train={sum(len(v) for v in faces['train']['X'].values())} "
            f"exam={sum(len(v) for v in faces['exam']['X'].values())} "
            f"sel={sum(len(v) for v in faces['sel']['X'].values())}",
            flush=True,
        )

    if args.stage in ("train", "all"):
        summary = run_train(args.dest, faces, args.micro, lr_peak=args.lr_peak)
        if summary.get("status") == "EARLY_FAIL":
            print(
                "[mf700] EARLY FAIL under the health protocol — early_fail_receipt.json "
                "sealed; skipping the cell receipt; GPU0 released",
                flush=True,
            )
            return 0
        print(f"[mf700] train done in {summary['elapsed_s']:.0f}s", flush=True)

    if args.stage in ("receipt", "all"):
        if faces is None:
            faces = build_faces()
        receipt = run_cell_receipt(args.dest, faces)
        pick = receipt["sel"]["SEL-2_epoch_pick"]
        print(
            f"[mf700] SEL-2 pick e{pick} "
            f"examEMA(eq)={receipt['sel']['SEL-2_pick_exam_ema_equal_mean']:.4f} "
            f"examEMA(pool)={receipt['sel']['SEL-2_pick_exam_ema_pooled']:.4f} "
            f"sel2908EMA={receipt['sel']['SEL-2_pick_sel2908_ema_pooled']:.4f}",
            flush=True,
        )
        delta = receipt["l_axis_pair"]["delta_reading"]
        if delta.get("delta_r2_exam_ema_equal_mean_at_pick") is not None:
            print(
                f"[mf700] ΔR²(eq,pick)={delta['delta_r2_exam_ema_equal_mean_at_pick']:+.4f} "
                f":: {delta['reading_primary']}",
                flush=True,
            )
        else:
            print(f"[mf700] ΔR² {delta.get('status')}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        dest = None
        for arg_idx, a in enumerate(sys.argv):
            if a == "--dest" and arg_idx + 1 < len(sys.argv):
                dest = Path(sys.argv[arg_idx + 1])
        trace = traceback.format_exc()
        print(trace, file=sys.stderr)
        try:
            from btransform_unified_v1 import plan as _plan, receipts as _receipts

            target = dest or (_plan.RESULT_ROOT / "h1_matrix" / "M_F700_error")
            target.mkdir(parents=True, exist_ok=True)
            _receipts.seal_json(
                target / "error_receipt.json",
                {
                    "schema": "btransform_unified_v1_mf700_error",
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "traceback": trace,
                },
            )
        except Exception:
            pass
        sys.exit(3)
