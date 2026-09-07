"""H1 matrix cell M-F250: L=250 x identity (f) proj_add (user proposal 2026-09-06).

MATRIX_H1_L_IDENTITY_V1_20260906.md cell M-F250 — BTransformerUnifiedDecoderIdentity
with override_window=250, identity_mode="proj_add" (E0 -> 16-d learnable
projection P added channel-wise onto the LOCAL conv channels; carrier concat
unchanged; rank-16 bottleneck of the concat first layer; L-independent).

Method-selection exam: LODO holdout date 1925-01-20 (2 sessions) — the cell is
trained on the other 11 sessions and scored on the holdout date's minival
eval_mask coordinates (SUBMISSION-PROTOCOL stage 1; stage-2 full retraining is
outside the matrix).

Per-epoch scoring (EMA + RAW views):
  - lodo exam: holdout-date sessions (1925-01-20), all minival eval_mask ends
    (2,952 faces) — the SEL-2 epoch-pick surface: earliest max on the EMA
    equal_session_mean series (REVIEW_MATRIX_H1_L_IDENTITY_V1_20260906.md
    blocking item B: same-source 13-session faces are broken selectors)
  - 2,908 selection face: minival query_starts+699 grid over all 13 sessions
    — AUXILIARY reporting aligned with formal12; never used for selection
SEL-2 earliest-max epoch pick on the exam EMA equal_session_mean; SEL-1
endpoint24 + last-4/last-8 auxiliary on the same exam series; SEL-4 pooled /
session-mean dual report on every surface.

CAL-2/M3 ruling (coordinator, post-skeleton-delivery 2026-09-06): CAL-1 budget
rotation is DOWNGRADED to CAL-2 fixed M3 (budget=3). Reason: adapter stub —
build_h1_bank raises NotImplementedError for budgets {7,5,4} (the only located
trialized source = C2 M3 payload, 3 trials/session; NWB re-trialization not
wired; codified in tests). Training consumes the M3 bank throughout — bitwise
identical to the five-arm cache bank (adapter provenance loop). CAL-1 is
deferred as an upgrade cell. The M-F700 cell (same-session agent) uses the
same CAL-2/M3, keeping the L-axis controlled pair internally consistent.

Data: adapters.build_h1_bank(surface, session, budget=3, window=250,
identity_mode="proj_add") — the live Phase-2b path (five-arm windowing
generalized to L=250, targets native, E0 fused [176,700] reproduced bitwise
from the frozen cache). The 2,908 query-grid face is materialized with the
adapter's own _h1_windows on the minival query grid (count gate == 2908).

Discipline: CUDA_VISIBLE_DEVICES pinned to GPU1 (refuses if a foreign pid
holds >500 MiB there); GPU0 untouched; bf16 autocast training, FP32 eval;
6h budget from train start; receipts sealed via receipts.seal_json
(0444 + sha256 sidecar); historical roots untouched; no EvalAI.

Usage:
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 \
  BTRANSFORM_MF250_TRAIN=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v1/scripts/h1_matrix_mf250_proj_add.py --stage all \
  --dest btransform_unified_v1/results/h1_matrix/M_F250_<stamp>

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
from btransform_unified_v1.matrix_cells import cell_by_id, cell_geometry  # noqa: E402
from btransform_unified_v1.model import UNIT_DROPOUT_DOMAIN_META, unit_dropout_seed, whole_unit_dropout  # noqa: E402
from btransform_unified_v1.r2 import session_mean_report, variance_weighted_r2  # noqa: E402
from btransform_unified_v1.scale_bridge import assert_scale_bridge  # noqa: E402
from btransform_unified_v1.schedule import warmup_cosine_lr  # noqa: E402

CELL_ID = "M-F250"
CELL = "M-F250-PROJ-ADD"
TRAIN_ENV_FLAG = "BTRANSFORM_MF250_TRAIN"
SEED = plan.SEED
GPU1_UUID = "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"
FOREIGN_MEM_MIB_LIMIT = 500
BUDGET_SECONDS = 6.0 * 3600.0
EPOCHS = plan.EPOCHS
SCALE = h1_config.TARGET_MULTIPLIER  # 20.0
EFFECTIVE_BATCH = plan.BATCH_SIZE  # 32
EVAL_BATCH = 32
SELECTION_FACE_COUNT = 2908  # frozen minival query-grid count (formal12 rule)

# Reference comparisons (receipt context only, NOT gates):
REFERENCES = {
    "fivearm_l100_midway": {
        "a_concat700_pooled_r2": 0.080,
        "b_joined36_pooled_r2": 0.178,
        "note": "five-arm L=100 midway readings (12-ep run, 20,325-face minival, pooled R2); not final",
        "source": "btransform_unified_v1/results/h1_sec6_fivearm_l100_v1/live.json (midway snapshot)",
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
        "CAL-2 {fixed budget M3 for banks AND deployment; matrix doc §0 revised 2026-09-06: 全矩阵 CAL-2/M3 固定预算 (REVIEW_MATRIX_H1_L_IDENTITY_V1_20260906.md blocking item A — formal matrix-doc revision, not a script-level downgrade)}",
        "CAL-2 REASON {adapter stub: trialization for 7/5/4 not wired (only located trialized source = C2 M3 payload, 3 trials/session); CAL-1 deferred as upgrade cell; M3 bank bitwise-reproduces the five-arm cache bank (provenance loop closed by adapters.build_h1_bank at budget 3)}",
        "CAL-2 PAIR CONSISTENCY {M-F250 (this cell) and M-F700 (same-session agent) both run CAL-2/M3, so the L-axis comparison 250 vs 700 is controlled on calibration budget}",
    ],
    "TRN": [
        "TRN-1 {24ep, AdamW wd 0.01 clip 1.0, warmup 1ep -> 3e-4 -> cosine 3e-5 (min_factor 0.1), EMA 0.9995, effective batch 32, bf16 autocast forward+loss with pred.float() before MSE}",
        "TRN-1 update caliber {updates/epoch = LODO-measured per-session-ceil caliber (sum_s ceil(n_s/32), ~597; batches never mix sessions); NOT the formal12 full-train 731 — recorded for reference only}",
        "TRN-3 {whole-unit dropout p=0.10, training mode only, per-batch CPU generator, domain m2_small_unit_dropout (S1 unit_dropout_seed import-or-replicate)}",
        "TRN-5 {P: 0; window_override 250 (l_in 250)}",
        "TRN-6 {microbatch decided by on-card memory probe (32 direct preferred; 16x2 fallback), recorded in run_meta}",
        "TRN-8 {scale: x20 train on native targets; score pred/20 vs native; ratio bridge MSE(raw,20y)==400*MSE(raw/20,y) rel tol 1e-9}",
    ],
    "SEL": [
        "SEL-2 {surface: LODO holdout-date exam (1925-01-20, 2 sessions, 2,952 faces), statistic: EMA equal_session_mean, rule: earliest max, ties<=1e-10 -> earliest; rule fixed before reading numbers (REVIEW blocking item B: same-source 13-session faces are broken selectors — M1 lesson; sel2908 is NOT used for picking)}",
        "SEL-2 AUXILIARY {2,908 full-window selection face (minival query_starts+699, all 13 sessions) computed and reported every epoch for formal12 alignment; auxiliary only, never used for selection}",
        "SEL-1 {endpoint24 primary + last-4/last-8 auxiliary on the SEL-2 exam series}",
        "SEL-4 {pooled and session-mean dual report on every surface; never subtract across surfaces}",
        "SEL-3 {official surface: zero participation}",
    ],
    "SPD": ["none {training-path run; SPD-A1 proj_add fold parity is covered by the skeleton tests, not used in training/scoring}"],
    "IDENTITY_USAGE": [
        "(f) proj_add {P = Linear(700->16, bias=False); tokens = token_mlp([local16 + P(E0)] | carrier4); carrier concat unchanged}",
        "add_tail irrelevance declaration {proj_add is L-INDEPENDENT: E0 enters via a session-static 16-d channel projection added to the local conv channels; there is NO 700-bin identity TIME template and NO end-aligned tail alignment (the add_tail (c) alignment question does not apply to this cell at any L)}",
    ],
}

NOTE_SIX_ROWS = {
    "system": (
        "btransform_unified_v1 BTransformerUnifiedDecoderIdentity (H1 matrix cell M-F250: "
        "override_window 250, prefix 0, identity_mode proj_add — P=Linear(700->16,bias=False) "
        "fused into the local conv channels, token_in 20, seed 42) — NOT SPINT"
    ),
    "consumer": (
        "8-slot + CausalPE4 unified decoder (this series), H1 L x identity training matrix "
        "(MATRIX_H1_L_IDENTITY_V1_20260906); matrix-cell readout, no official submission"
    ),
    "calibration_object": (
        "E0 [176,700] C2 fused identity, materialized by the frozen C2 e15 materializer "
        "(ckpt sha ce46267e...) over the M3 payload trialized activity [3,1024,176] — "
        "bitwise-reproduces the five-arm cache bank (adapter provenance loop); "
        "H-C carrier [176,4]; CAL-2 fixed M3 (coordinator ruling: CAL-1 {7,5,4,3} "
        "downgraded — trialization for 7/5/4 not wired; deferred as upgrade cell)"
    ),
    "scoring_surface": (
        "LODO method-selection exam: holdout date 1925-01-20 (2 sessions, 2,952 minival "
        "eval_mask faces) trained-never; plus the preregistered 2,908 full-window selection "
        "face (minival query_starts+699, all 13 sessions) for SEL-2 epoch-pick; EMA+RAW per "
        "epoch; pooled + session-mean dual report"
    ),
    "scale": (
        "x20 train / score pred/20 (ratio bridge MSE(raw,20y) = 400*MSE(raw/20,y), rel tol "
        "1e-9, asserted every epoch on the LODO exam)"
    ),
    "single_difference_vs_historical_best": (
        "vs five-arm L=100 arms (a) 0.080 / (b) 0.178 midway pooled: the single differing "
        "rows are 'system' window (100 -> 250) and identity usage (concat/joined -> proj_add "
        "rank-16 bottleneck); recipe differs from formal12 (24ep cosine vs 12ep const-lr) and "
        "is TRN-1 throughout. proj_add is L-independent (no template alignment), so the L "
        "axis and the usage axis do not interact through the identity pathway in this cell. "
        "L-axis consistency: M-F700 (same-session agent) shares CAL-2/M3, seed, recipe, and "
        "LODO exam — 250-vs-700 reads as a controlled pair"
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
# GPU preflight (GPU1 pinned; foreign pid > 500 MiB => BLOCKED receipt)
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
        if app["gpu_uuid"] == GPU1_UUID and app["pid"] != own_pid and app["used_mib"] > FOREIGN_MEM_MIB_LIMIT
    ]
    gpu1 = next(g for g in gpus if g["uuid"] == GPU1_UUID)
    report = {
        "schema": "btransform_unified_v1_mf250_gpu_preflight",
        "cell": CELL,
        "unix": time.time(),
        "utc": datetime.now(timezone.utc).isoformat(),
        "own_pid": own_pid,
        "cuda_visible_devices": raw,
        "gpus": gpus,
        "compute_apps": apps,
        "gpu1": gpu1,
        "foreign_pids_on_gpu1": foreign,
        "foreign_threshold_mib": FOREIGN_MEM_MIB_LIMIT,
        "gpu0_touched": False,
        "note": "coordinator 2026-09-06: GPU1 yielded by M1 (H1 takes priority); GPU0 held by five-arm pid 1803132 (fallback only)",
    }
    ok = raw == "1" and not foreign and torch.cuda.device_count() == 1 and gpu1["memory_used_mib"] < FOREIGN_MEM_MIB_LIMIT
    report["ok"] = bool(ok)
    _seal(out_path, report)
    return report


# ---------------------------------------------------------------------------
# Data: LODO split + faces via the adapter (budget 3, window 250, proj_add)
# ---------------------------------------------------------------------------


def build_faces() -> dict[str, Any]:
    """Banks + face arrays for train / lodo-exam / 2908-selection surfaces."""
    split = h1_config.lodo_split()
    train_sessions = list(split["train_sessions"])
    holdout_sessions = list(split["holdout_sessions"])

    train_banks, train_X, train_y, train_ids = {}, {}, {}, {}
    for s in train_sessions:
        bank = adapters.build_h1_bank("train", s, budget=3, window=250, identity_mode="proj_add")
        train_banks[s] = bank
        train_X[s] = bank.X_store
        train_y[s] = bank.target_store
        train_ids[s] = bank.window_ids

    exam_banks, exam_X, exam_y, exam_ids = {}, {}, {}, {}
    for s in holdout_sessions:
        bank = adapters.build_h1_bank("minival", s, budget=3, window=250, identity_mode="proj_add")
        exam_banks[s] = bank
        exam_X[s] = bank.X_store
        exam_y[s] = bank.target_store
        exam_ids[s] = bank.window_ids

    # 2,908 selection face: minival query grid (query_starts + 699), all 13
    # sessions, windowed with the adapter's own generalized _windows.
    cache = adapters._h1_source_cache()
    sel_banks, sel_X, sel_y, sel_ids = {}, {}, {}, {}
    for s in h1_config.H1_ALL_SESSIONS:
        bank = exam_banks.get(s) or train_banks.get(s)
        if bank is None:
            bank = adapters.build_h1_bank("minival", s, budget=3, window=250, identity_mode="proj_add", limit_windows=1)
            sel_banks[s] = bank
        else:
            sel_banks[s] = bank
        row = cache["minival"][s]
        ends = (np.asarray(row["query_starts"], dtype=np.int64) + h1_config.FULL_WINDOW - 1).astype(np.int64)
        neural = np.ascontiguousarray(row["neural"], dtype=np.float32)
        sel_X[s] = adapters._h1_windows(neural, ends, 250)
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
# proj_add (f) brief: P norm / sparsity / effective rank
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
    # P(E0) per-session Frobenius norms (identity expressiveness across sessions)
    return out


def proj_apply_stats(model, faces: dict[str, Any], sessions: list[str]) -> dict[str, Any]:
    proj_module = model.frontend.e0_proj  # Linear module (callable), not the weight tensor
    per_session = {}
    for s in sessions:
        e0 = torch.from_numpy(faces["banks"][s].E0)
        with torch.no_grad():
            pe0 = proj_module(e0.to(proj_module.weight.device, dtype=proj_module.weight.dtype))
        per_session[s] = {
            "pe0_frobenius": float(np.linalg.norm(pe0.detach().cpu().numpy())),
            "pe0_absmax": float(pe0.detach().abs().max().item()),
        }
    return {"pe0_per_session": per_session}


# ---------------------------------------------------------------------------
# Memory probe (TRN-6 decision: micro 32 direct vs 16x2)
# ---------------------------------------------------------------------------


def memory_probe(model, bank, X, y, device) -> dict[str, Any]:
    """TRN-6 probe: fwd+bwd peak memory/time at micro 32 vs 16 (params restored)."""
    results = {}
    was_training = model.training
    model.train()
    named = model.trainable_parameters()
    backup = {name: param.detach().clone() for name, param in named.items()}
    for micro in (32, 16):
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
# Training
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ep10 health-checkpoint protocol (coordinator 2026-09-06; no post-hoc relax)
# ---------------------------------------------------------------------------


def _ep10_ruling(train_mse: dict[int, float], exam_raw: dict[int, Any]) -> dict[str, Any]:
    """Preregistered ep10 ruling: continue only if BOTH bars met, else fail.

    Continue: train_mse[10] <= 0.0065 AND exam_raw[10] >= -0.007 AND no
    3-consecutive RAW decline within e1..e10. Any miss = early fail (the
    pinned-loss band 0.0068-0.0070 is the specific plateau instance).
    """
    loss10 = float(train_mse[10])
    raw10 = float(exam_raw[10]["pooled_r2"])
    raws = [float(exam_raw[e]["pooled_r2"]) for e in sorted(exam_raw) if e <= 10]
    decline3 = any(raws[i] > raws[i + 1] > raws[i + 2] for i in range(len(raws) - 2))
    loss_ok = loss10 <= 0.0065
    raw_ok = raw10 >= -0.007
    pinned = 0.0068 <= loss10 <= 0.0070
    continue_run = bool(loss_ok and raw_ok and not (decline3 and raw10 < -0.01))
    parts = [
        f"loss10={loss10:.6f} ({'<=0.0065 OK' if loss_ok else 'bar 0.0065 NOT met'}"
        f"{'; pinned-band' if pinned else ''})",
        f"raw10={raw10:.4f} ({'OK' if raw_ok else 'below -0.007'})",
        f"3-consec-decline={decline3}",
    ]
    verdict = "CONTINUE to 24ep (both bars met)" if continue_run else "FAIL: " + ", ".join(parts)
    return {
        "train_mse_ep10": loss10,
        "train_mse_le_0p0065": bool(loss_ok),
        "loss_pinned_band_0.0068_0.0070": bool(pinned),
        "exam_raw_ep10": raw10,
        "exam_raw_ge_minus0p007": bool(raw_ok),
        "raw_3_consecutive_decline": bool(decline3),
        "continue_run": continue_run,
        "verdict": verdict,
    }


def run_train(dest: Path, faces: dict[str, Any], micro_override: int | None, peak_lr: float | None, ep10_protocol: bool) -> dict[str, Any]:
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)

    peak = float(peak_lr) if peak_lr is not None else plan.LR_PEAK
    lr_variant = None
    if peak != plan.LR_PEAK:
        lr_variant = {
            "tag": "TRN-1-LR1e4 variant" if abs(peak - 1e-4) < 1e-12 else f"TRN-1-LR{peak:g} variant",
            "declaration": (
                "TRN-1-LR1e4 variant: deviation from matrix §0 peak 3e-4, approved restart "
                "per FAIL 62d8a17b (loss-plateau early-fail), LR hypothesis: formal12 H1 用 1e-4"
            ),
            "schedule_shape": "warmup 1ep (597 steps) -> cosine 14328 steps; peak %g, floor %g" % (peak, peak * plan.LR_MIN_FACTOR),
            "approved_by": "coordinator 2026-09-06 restart authorization (user '有问题可提前推翻重来' instruction)",
        }

    cell = cell_by_id(CELL_ID)
    geometry = cell_geometry(cell)
    model = BTransformerUnifiedDecoderIdentity(
        geometry, seed=SEED, override_prefix=0, override_window=cell.L, identity_mode=cell.identity_mode
    ).to(device)
    meta = model.init_meta
    plan.require(meta["identity_mode"] == "proj_add" and meta["matrix_letter"] == "f", "wrong identity mode")
    plan.require(meta["token_in"] == 20 and meta["proj_out_dim"] == 16, "proj_add geometry drift")
    plan.require(model.window == 250 and model.l_in == 250, "window override drift")

    # startup sanity: causality on a real holdout window
    exam0_s = faces["exam"]["sessions"][0]
    causal = model.causal_check(torch.from_numpy(faces["exam"]["X"][exam0_s][:2]).to(device), faces["exam"]["banks"][exam0_s])
    plan.require(causal["passed"], "causal_check failed on real data")

    # banks -> device tensors kept per session (E0/carrier are static)
    train_sessions = faces["train"]["sessions"]
    n_train_faces = int(sum(len(faces["train"]["X"][s]) for s in train_sessions))
    plan.require(n_train_faces == 18935, f"LODO train face count {n_train_faces} != 18935 (inventory)")

    # TRN-6 memory probe -> microbatch decision
    probe_s = train_sessions[0]
    probe = memory_probe(model, faces["train"]["banks"][probe_s], faces["train"]["X"][probe_s], faces["train"]["y"][probe_s], device)
    if micro_override is not None:
        micro = int(micro_override)
        accum = EFFECTIVE_BATCH // micro
    elif probe.get("micro32", {}).get("peak_alloc_mib", 1e9) <= 18000:
        micro, accum = 32, 1
    else:
        micro, accum = 16, 2
    plan.require(micro * accum == EFFECTIVE_BATCH, "micro x accum != effective batch 32")

    # update caliber: LODO-measured, per-session-ceil (batches never mix
    # sessions, so an epoch's optimizer steps = sum_s ceil(n_s / 32)).
    updates_per_epoch = int(
        sum(math.ceil(len(faces["train"]["X"][s]) / EFFECTIVE_BATCH) for s in train_sessions)
    )
    warmup_updates = updates_per_epoch * plan.WARMUP_EPOCHS
    total_updates = updates_per_epoch * EPOCHS
    formal = plan.recipe_updates("h1", epochs=EPOCHS)

    # CAL-2 fixed M3 (coordinator ruling; no rotation). CAL-1 deferred.
    cal1_note = (
        "CAL-1 rotation NOT run (coordinator downgrade to CAL-2/M3): adapter stub — "
        "budgets {7,5,4} not materializable (trialization not wired; only located "
        "trialized source = C2 M3 payload, 3 trials/session); CAL-1 deferred as "
        "upgrade cell. Banks = M3 throughout, bitwise-reproducing the five-arm cache."
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=peak, weight_decay=plan.WEIGHT_DECAY,
        betas=(0.9, 0.999), eps=1e-8,
    )
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    rng = np.random.default_rng(SEED)

    split = faces["split"]
    _seal(
        dest / "run_meta.json",
        {
            "schema": "btransform_unified_v1_mf250_run_meta",
            "cell": CELL,
            "cell_id": CELL_ID,
            "matrix_doc": "MATRIX_H1_L_IDENTITY_V1_20260906.md §1 (M-F250, user 2026-09-06 proposal)",
            "utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "geometry": geometry,
            "peak_lr": peak,
            "lr_min": peak * plan.LR_MIN_FACTOR,
            "lr_variant": lr_variant,
            "window": model.window,
            "l_in": model.l_in,
            "identity_mode": model.identity_mode,
            "init_meta": meta,
            "param_count": meta["param_count"],
            "proj_param_count": meta.get("proj_param_count"),
            "causal_check_startup": causal,
            "lodo_split": split,
            "train_face_counts": {s: int(len(faces["train"]["X"][s])) for s in train_sessions},
            "exam_face_counts": {s: int(len(faces["exam"]["X"][s])) for s in faces["exam"]["sessions"]},
            "selection_face_counts": {s: int(len(faces["sel"]["X"][s])) for s in faces["sel"]["sessions"]},
            "n_train_faces": n_train_faces,
            "microbatch_probe": probe,
            "microbatch_decision": {"micro": micro, "accum": accum, "effective_batch": EFFECTIVE_BATCH},
            "updates_per_epoch": updates_per_epoch,
            "total_updates": total_updates,
            "warmup_updates": warmup_updates,
            "formal_update_caliber_recorded": formal,
            "update_caliber_note": (
                "LODO-measured per-session-ceil caliber is authoritative: warmup=1 epoch and "
                "cosine span=24 epochs of THIS run's step count (sum_s ceil(n_s/32) ~ 597 "
                "upd/ep; batches never mix sessions). The formal12 full-train caliber (731 "
                "upd/ep, 17544 total) is recorded for REFERENCE ONLY"
            ),
            "cal_rule": "CAL-2 fixed M3 (coordinator ruling 2026-09-06; no budget rotation)",
            "cal_budget": h1_config.DEPLOY_BUDGET,
            "cal1_downgrade_note": cal1_note,
            "unit_dropout_p": plan.UNIT_DROPOUT,
            "unit_dropout_domain_meta": UNIT_DROPOUT_DOMAIN_META,
            "batch_rule": (
                "five-arm style: per epoch, session order shuffled then faces permuted within "
                "session (np default_rng(42), consumption order recorded); batches never mix "
                "sessions (bank E0 is per-session static)"
            ),
            "scale": {"target_multiplier": SCALE, "scoring": "pred/20 vs native (ratio bridge)"},
            "epochs": EPOCHS,
            "budget_seconds": BUDGET_SECONDS,
            "gpu_uuid": GPU1_UUID,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "source_cache_sha256": h1_config.SOURCE_CACHE_SHA256,
            "c2_payload_sha256": h1_config.C2_M3_PAYLOAD_SHA256,
            "c2_ckpt_sha256": h1_config.C2_CKPT_SHA256,
            "picks": PICKS,
            "references_not_gates": REFERENCES,
        },
    )

    metrics_path = dest / "metrics.jsonl"
    heartbeat = dest / "heartbeat.json"
    (dest / "PROGRESS_10MIN.md").write_text(
        "# M-F250 progress (per-epoch; ep10 health protocol "
        + ("ENFORCED inline" if ep10_protocol else "off")
        + f")\n\nCell: {CELL}, peak_lr={peak:g}, floor={peak * plan.LR_MIN_FACTOR:g}, "
        + (lr_variant["tag"] + "\n" if lr_variant else "TRN-1 base (3e-4)\n")
        + f"GPU1 {GPU1_UUID}, micro {micro}x{accum}, {updates_per_epoch} upd/ep\n\n"
        "| ep | train_mse | examRAW pooled | examRAW eqmean | examEMA pooled | sel2908 RAW (aux) | sel2908 EMA (aux) |\n"
        "|----|-----------|----------------|----------------|----------------|-----------------|------------------|\n",
        encoding="utf-8",
    )
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
                            "schema": "btransform_unified_v1_mf250_budget_hit",
                            "cell": CELL,
                            "epoch": epoch,
                            "global_step": global_step,
                            "budget_seconds": BUDGET_SECONDS,
                            "elapsed_seconds": time.monotonic() - started,
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "note": "6h GPU budget hit before finishing 24 epochs; no rerun authorized",
                        },
                    )
                    raise RuntimeError("M-F250 6h GPU budget hit")
                take = idx[offset : offset + EFFECTIVE_BATCH]
                micro_losses = []
                for m_off in range(0, len(take), micro):
                    m_take = take[m_off : m_off + micro]
                    batch_id += 1
                    global_step += 1
                    lr = warmup_cosine_lr(
                        global_step, total_steps=total_updates, warmup_steps=warmup_updates,
                        peak=peak, min_factor=plan.LR_MIN_FACTOR,
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = lr
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
                            "gpu_uuid": GPU1_UUID,
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
        with (dest / "PROGRESS_10MIN.md").open("a", encoding="utf-8") as prog:
            prog.write(
                f"| {epoch} | {row['train_mse']:.6f} | {exam_raw['pooled_r2']:.4f} | "
                f"{exam_raw['equal_session_mean']:.4f} | {exam_ema['pooled_r2']:.4f} | "
                f"{sel_raw['pooled_r2']:.4f} | {sel_ema['pooled_r2']:.4f} |\n"
            )
        print(
            f"[mf250] ep{epoch} loss={train_mse_series[epoch]:.5f} "
            f"examRAW={exam_raw['pooled_r2']:.4f} "
            f"examEMA={exam_ema['pooled_r2']:.4f}/{exam_ema['equal_session_mean']:.4f} "
            f"selEMA={sel_ema['pooled_r2']:.4f}/{sel_ema['equal_session_mean']:.4f} "
            f"({row['seconds']:.0f}s)",
            flush=True,
        )
        ckpt = {
            "schema": "btransform_unified_v1_mf250_ckpt",
            "cell": CELL, "epoch": epoch, "global_step": global_step, "seed": SEED,
            "raw_state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "ema": ema.state_dict(), "lr": float(lr),
        }
        torch.save(ckpt, dest / f"epoch_{epoch:03d}.pt")

        if ep10_protocol and epoch == 10:
            ruling = _ep10_ruling(train_mse_series, exam_raw_series)
            if not ruling["continue_run"]:
                fail = {
                    "schema": "btransform_unified_v1_mf250_early_fail_receipt",
                    "cell": CELL,
                    "cell_id": CELL_ID,
                    "status": "EARLY_FAIL_AT_EP10_PER_PROTOCOL",
                    "ruling_text": f"M-F250 {lr_variant['tag'] if lr_variant else 'TRN-1 base'} on LODO: early-fail at ep10 per protocol (re-run; no further self-retry)",
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "lr_variant": lr_variant,
                    "protocol": {
                        "source": "coordinator 2026-09-06 early-checkpoint protocol (same as FAIL 62d8a17b run)",
                        "checkpoint_epoch": 10,
                        "continue_requires_both": {"train_mse_le": 0.0065, "exam_raw_ge": -0.007, "no_3_consecutive_decline": True},
                        "fail_triggers_either": ["loss pinned at 0.0069+-0.0001", "examRAW 3-consecutive-epoch decline AND < -0.01"],
                        "no_post_hoc_relaxation": True,
                        "no_further_self_retry": True,
                    },
                    "evaluation_at_ep10": ruling,
                    "epochs_completed": sorted(train_mse_series),
                    "trajectory": {
                        str(e): {
                            "train_mse": train_mse_series[e],
                            "exam_raw_pooled": exam_raw_series[e]["pooled_r2"],
                            "exam_raw_equal_mean": exam_raw_series[e]["equal_session_mean"],
                            "exam_ema_pooled": exam_ema_series[e]["pooled_r2"],
                            "exam_ema_equal_mean": exam_ema_series[e]["equal_session_mean"],
                            "aux_sel2908_raw_pooled": raw_series[e]["pooled_r2"],
                            "aux_sel2908_ema_pooled": ema_series[e]["pooled_r2"],
                            "lr": lr_series[e],
                        }
                        for e in sorted(train_mse_series)
                    },
                    "picks": PICKS,
                    "note_six_rows": NOTE_SIX_ROWS,
                    "gpu_release": {"gpu1_uuid": GPU1_UUID, "released_utc": datetime.now(timezone.utc).isoformat()},
                }
                _seal(dest / "FAIL_receipt.json", fail)
                print(f"[mf250] EARLY-FAIL at ep10 per protocol: {ruling['verdict']}", flush=True)
                return {"status": "early_fail_ep10", "cell": CELL, "ruling": ruling, "epochs_completed": sorted(train_mse_series)}

    plan.require(
        global_step == total_updates,
        f"update-count drift: made {global_step} optimizer steps, schedule assumed {total_updates}",
    )

    summary = {
        "schema": "btransform_unified_v1_mf250_train_receipt",
        "cell": CELL,
        "cell_id": CELL_ID,
        "seed": SEED,
        "gpu_uuid": GPU1_UUID,
        "peak_lr": peak,
        "lr_variant": lr_variant,
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
    metrics: dict[int, dict[str, Any]] = {}
    with (dest / "metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "epoch":
                metrics[int(row["epoch"])] = row

    sel_ema = {e: float(metrics[e]["sel2908_ema"]["pooled_r2"]) for e in metrics}
    sel_raw = {e: float(metrics[e]["sel2908_raw"]["pooled_r2"]) for e in metrics}
    exam_ema = {e: float(metrics[e]["lodo_exam_ema"]["pooled_r2"]) for e in metrics}
    exam_ema_eq = {e: float(metrics[e]["lodo_exam_ema"]["equal_session_mean"]) for e in metrics}
    exam_raw = {e: float(metrics[e]["lodo_exam_raw"]["pooled_r2"]) for e in metrics}
    exam_raw_eq = {e: float(metrics[e]["lodo_exam_raw"]["equal_session_mean"]) for e in metrics}

    # SEL-2 (REVIEW blocking item B): pick on the LODO holdout-date exam, EMA
    # equal_session_mean, earliest max. The 2,908 face stays auxiliary.
    pick = _sel2_pick(exam_ema_eq)

    # reload the SEL-2 epoch checkpoint for the (f)-mode P brief + full detail
    ckpt = torch.load(dest / f"epoch_{pick:03d}.pt", map_location=device, weights_only=False)
    cell_obj = cell_by_id(CELL_ID)
    model = BTransformerUnifiedDecoderIdentity(
        cell_geometry(cell_obj), seed=SEED, override_prefix=0, override_window=cell_obj.L,
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

    receipt = {
        "schema": "btransform_unified_v1_mf250_cell_receipt",
        "cell": CELL,
        "cell_id": CELL_ID,
        "matrix": "H1 L x identity (MATRIX_H1_L_IDENTITY_V1_20260906) — mode (f) proj_add first GPU training",
        "utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "peak_lr": (json.loads((dest / "run_meta.json").read_text()) or {}).get("peak_lr", plan.LR_PEAK),
        "lr_variant": (json.loads((dest / "run_meta.json").read_text()) or {}).get("lr_variant"),
        "gate": "NONE (matrix cell; readings feed the matrix interpretation rules — matrix doc §2 feeding-method reading: (f) vs later (a)/(b))",
        "references_not_gates": REFERENCES,
        "lodo_split": faces["split"],
        "sel": {
            "SEL-2_rule": (
                "earliest max on the LODO holdout-date exam (1925-01-20, 2 sessions) EMA "
                "equal_session_mean series; ties<=1e-10 -> earliest (rule before numbers; "
                "REVIEW_MATRIX_H1_L_IDENTITY_V1_20260906.md blocking item B)"
            ),
            "SEL-2_epoch_pick": pick,
            "SEL-2_pick_exam_ema_equal_mean": exam_ema_eq[pick],
            "SEL-2_pick_exam_ema_pooled": exam_ema[pick],
            "SEL-2_pick_exam_raw_equal_mean": exam_raw_eq.get(pick),
            "SEL-2_auxiliary_sel2908_ema_pooled_at_pick": sel_ema[pick],
            "SEL-1_endpoint24_exam_ema_equal_mean": exam_ema_eq.get(EPOCHS),
            "SEL-1_endpoint24_exam_ema_pooled": exam_ema.get(EPOCHS),
            "SEL-1_endpoint24_exam_raw_equal_mean": exam_raw_eq.get(EPOCHS),
            "SEL-1_endpoint24_auxiliary_sel2908_ema_pooled": sel_ema.get(EPOCHS),
            "SEL-1_last4_exam_ema_equal_mean": _last_k(exam_ema_eq, EPOCHS - 3, EPOCHS),
            "SEL-1_last8_exam_ema_equal_mean": _last_k(exam_ema_eq, EPOCHS - 7, EPOCHS),
            "SEL-4_note": "pooled and session-mean reported for every surface/epoch (metrics.jsonl); never subtract across surfaces",
        },
        "epoch_curves": {
            "exam_ema_equal_mean_SEL2_series": exam_ema_eq,
            "exam_ema_pooled": exam_ema,
            "exam_raw_equal_mean": exam_raw_eq,
            "exam_raw_pooled": exam_raw,
            "auxiliary_sel2908_ema_pooled": sel_ema,
            "auxiliary_sel2908_raw_pooled": sel_raw,
            "auxiliary_sel2908_ema_equal_mean": {e: float(metrics[e]["sel2908_ema"]["equal_session_mean"]) for e in metrics},
            "auxiliary_sel2908_note": (
                "2,908 full-window selection face (minival query grid, all 13 sessions) — "
                "formal12-aligned AUXILIARY reporting; NOT used for epoch selection"
            ),
            "summary_epochs": {
                str(e): {
                    "train_mse": metrics[e]["train_mse"],
                    "exam_ema_equal_mean": exam_ema_eq.get(e),
                    "exam_ema_pooled": exam_ema.get(e),
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
        "picks": PICKS,
        "note_six_rows": NOTE_SIX_ROWS,
        "artifacts": {
            "run_meta": "run_meta.json (sealed)",
            "train_receipt": "train_receipt.json (sealed)",
            "metrics": "metrics.jsonl",
            "sel2_pick_detail": "sel2_pick_detail.json",
            "checkpoints": "epoch_001.pt..epoch_024.pt",
        },
    }
    _write_json(dest / "cell_receipt_content.json", receipt)
    _seal(dest / "cell_receipt.json", receipt)
    return receipt


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="H1 matrix cell M-F250 (proj_add, L=250, LODO)")
    parser.add_argument("--stage", choices=["preflight", "build", "train", "receipt", "all"], default="all")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--micro", type=int, default=None, help="override microbatch (32 or 16)")
    parser.add_argument("--peak-lr", type=float, default=None, help="override peak LR (default plan.LR_PEAK 3e-4); coordinator-approved variants only")
    parser.add_argument("--ep10-protocol", action="store_true", help="enforce the preregistered ep10 health checkpoint inline")
    args = parser.parse_args()

    if args.stage in ("train", "all") and os.environ.get(TRAIN_ENV_FLAG) != "1":
        print(f"REFUSED: training requires {TRAIN_ENV_FLAG}=1", file=sys.stderr)
        return 2
    if args.stage in ("train", "all", "receipt"):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            print("REFUSED: CUDA_VISIBLE_DEVICES must be pinned to '1' (GPU1)", file=sys.stderr)
            return 2

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"[mf250] dest = {args.dest}", flush=True)

    if args.stage in ("preflight", "train", "receipt", "all"):
        pre = gpu_preflight(args.dest / "preflight_gpu.json")
        print(f"[mf250] preflight ok={pre['ok']} foreign={pre['foreign_pids_on_gpu1']}", flush=True)
        if not pre["ok"]:
            _seal(
                args.dest / "BLOCKED.json",
                {
                    "schema": "btransform_unified_v1_mf250_blocked",
                    "cell": CELL,
                    "reason": "GPU1 not clean at preflight (foreign pid >500 MiB or CUDA_VISIBLE_DEVICES!=1)",
                    "preflight": pre,
                    "utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            return 2

    faces = None
    if args.stage in ("build", "train", "receipt", "all"):
        t0 = time.monotonic()
        faces = build_faces()
        _write_json(
            args.dest / "face_inventory.json",
            {
                "built_utc": datetime.now(timezone.utc).isoformat(),
                "seconds": time.monotonic() - t0,
                "lodo_split": faces["split"],
                "train": {s: int(len(v)) for s, v in faces["train"]["X"].items()},
                "exam": {s: int(len(v)) for s, v in faces["exam"]["X"].items()},
                "sel": {s: int(len(v)) for s, v in faces["sel"]["X"].items()},
                "train_total": int(sum(len(v) for v in faces["train"]["X"].values())),
                "exam_total": int(sum(len(v) for v in faces["exam"]["X"].values())),
                "sel_total": int(sum(len(v) for v in faces["sel"]["X"].values())),
                "selection_face_count_gate": SELECTION_FACE_COUNT,
            },
        )
        print(
            f"[mf250] faces built in {time.monotonic()-t0:.0f}s: "
            f"train={sum(len(v) for v in faces['train']['X'].values())} "
            f"exam={sum(len(v) for v in faces['exam']['X'].values())} "
            f"sel={sum(len(v) for v in faces['sel']['X'].values())}",
            flush=True,
        )

    if args.stage in ("train", "all"):
        summary = run_train(args.dest, faces, args.micro, args.peak_lr, args.ep10_protocol)
        if summary.get("status") == "early_fail_ep10":
            print("[mf250] early-fail receipt sealed; NOT proceeding to cell receipt; no self-retry", flush=True)
            return 1
        print(f"[mf250] train done in {summary['elapsed_s']:.0f}s", flush=True)

    if args.stage in ("receipt", "all"):
        if faces is None:
            faces = build_faces()
        receipt = run_cell_receipt(args.dest, faces)
        pick = receipt["sel"]["SEL-2_epoch_pick"]
        print(
            f"[mf250] SEL-2 pick e{pick} "
            f"examEMAeq={receipt['sel']['SEL-2_pick_exam_ema_equal_mean']:.4f} "
            f"examEMApooled={receipt['sel']['SEL-2_pick_exam_ema_pooled']:.4f} "
            f"auxSel2908EMA={receipt['sel']['SEL-2_auxiliary_sel2908_ema_pooled_at_pick']:.4f}",
            flush=True,
        )
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

            target = dest or (_plan.RESULT_ROOT / "h1_matrix" / "M_F250_error")
            target.mkdir(parents=True, exist_ok=True)
            _receipts.seal_json(
                target / "error_receipt.json",
                {
                    "schema": "btransform_unified_v1_mf250_error",
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "traceback": trace,
                },
            )
        except Exception:
            pass
        sys.exit(3)
