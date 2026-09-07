"""H1 matrix cell M-F150: L=150 x identity (f) proj_add (user directive 2026-09-06).

L=150 REINSTATED per user directive after M-F250@1e-4 success (SEL-2 pick 24
exam EMA equal_session_mean 0.5522) and M-F700@1e-4 plateau (EARLY_FAIL ep10,
train_mse pinned 0.00693, examRAW e6 peak -0.0007 declining to -0.0113; receipt
results/h1_matrix/M_F700_lr1e4_20260906T104217Z/early_fail_receipt.json).
Plateau L-dependence reading: 1e-4 breaks the plateau at L=250 (0.00481) but
not at L=700 (0.00693) — L=700 has an optimization difficulty; the next
question is the SHORT end of the L axis.

L=150 is OUTSIDE the registered MATRIX_L {250, 350} (matrix doc revision had
withdrawn the L=150 cells). This cell runs by explicit user directive and is
constructed WITHOUT touching skeleton files:
  - geometry: dict(plan.TASK_GEOMETRY["h1"]) + matrix_e0_dim("proj_add")
    (mirrors h1_config.h1_matrix_geometry minus the MATRIX_L gate);
    override_window=150 is legal for the model (it only requires shortening
    the settled 700 window).
  - banks: adapters.build_h1_bank(budget=3, window=250, proj_add) — an
    ALLOWED call — then X_store re-windowed to 150 via the adapter's own
    adapters._h1_windows (the five-arm end-anchored mechanism). Target
    coordinates (session, end) are FROZEN (train ends = query_starts+699
    filtered by eval_mask; minival ends = eval_mask bins; sel2908 ends =
    query_starts+699), so face COUNTS are L-independent (train 18,935 /
    exam 2,952 / sel 2,908 — asserted against the M-F250-lr1e4 inventory)
    and the re-windowed store is asserted bitwise-equal to the tail-150 of
    the 250-window store (end-anchored truncation parity).

Everything else is item-by-item isomorphic to M-F250@1e-4
(results/h1_matrix/M_F250_lr1e4_20260906T094633Z): identity_mode proj_add,
seed 42, peak LR 1e-4 (warmup 597 -> cosine 14328 -> 1e-5), CAL-2/M3 bank,
LODO split (train 11 sessions / exam 1925-01-20 two sessions), bf16 autocast,
effective batch 32, EMA 0.9995, S1 domain unit dropout p=0.10.

Scoring per epoch (EMA + RAW views): lodo exam (2,952 faces; SEL-2 surface,
earliest max on EMA equal_session_mean) + auxiliary sel2908; SEL-1 endpoint24
+ last-4/last-8; SEL-4 pooled/session-mean dual report.

ep10 health protocol (coordinator 2026-09-06, same as M-F700):
  CONTINUE  iff train_mse <= 0.0065 AND examRAW >= -0.007 (no fail trigger)
  EARLY FAIL iff train_mse pinned 0.0069+/-0.0001 OR examRAW declining 3
             consecutive epochs below -0.01
  healthy reference at ep10: F250@1e-4 loss 0.00481 / examRAW +0.18.

L-axis judgement (preregistered, vs F250@1e-4 = 0.5522):
  exam EMA equal_session_mean >= 0.5522 - 0.01  -> shortest viable window is
      150 (temporal kernel down another 40% vs 250);
  drop > 0.02                                     -> horizon sits between
      150 and 250; lock 250.

Discipline: CUDA_VISIBLE_DEVICES pinned to GPU0 (uuid
GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9; refuses if a foreign pid holds
>500 MiB there); GPU1 belongs to the M2 projadd session and is NEVER
touched; bf16 autocast training, FP32 eval; 4h budget from train start;
receipts sealed via receipts.seal_json (0444 + sha256 sidecar); historical
roots untouched; no EvalAI.

Usage:
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
  BTRANSFORM_MF150_TRAIN=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v1/scripts/h1_matrix_mf150_proj_add.py --stage all \
  --dest btransform_unified_v1/results/h1_matrix/M_F150_lr1e4_<stamp> \
  --peak-lr 1e-4

Identity: B-transformer unified series, NOT SPINT.
"""
from __future__ import annotations

import argparse
import dataclasses
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
from btransform_unified_v1.bank import TaskBank, array_sha256  # noqa: E402
from btransform_unified_v1.ema import DecoderEMA  # noqa: E402
from btransform_unified_v1.identity_variant import BTransformerUnifiedDecoderIdentity  # noqa: E402
from btransform_unified_v1.matrix_cells import cell_by_id, cell_geometry  # noqa: E402
from btransform_unified_v1.model import UNIT_DROPOUT_DOMAIN_META, unit_dropout_seed, whole_unit_dropout  # noqa: E402
from btransform_unified_v1.r2 import session_mean_report, variance_weighted_r2  # noqa: E402
from btransform_unified_v1.scale_bridge import assert_scale_bridge  # noqa: E402
from btransform_unified_v1.schedule import warmup_cosine_lr  # noqa: E402

CELL_ID = "M-F150"
CELL = "M-F150-PROJ-ADD"
WINDOW = 150  # user-directive L (outside registered MATRIX_L {250,350}; see module docstring)
BANK_BUILD_WINDOW = 250  # allowed adapter call; X_store then re-windowed to 150
TRAIN_ENV_FLAG = "BTRANSFORM_MF150_TRAIN"
SEED = plan.SEED
GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"  # this cell's card
GPU1_UUID = "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"  # M2 projadd session — never touched
FOREIGN_MEM_MIB_LIMIT = 500
FIVEARM_PID = "1803132"
ARBITRATION_LOG = (
    plan.RESULT_ROOT / "h1_matrix" / "M_F700_arbitration" / "gpu0_arbitration.json"
)
BUDGET_SECONDS = 4.0 * 3600.0  # user budget 4h (expected ~25min train + scoring)
EPOCHS = plan.EPOCHS
SCALE = h1_config.TARGET_MULTIPLIER  # 20.0
EFFECTIVE_BATCH = plan.BATCH_SIZE  # 32
EVAL_BATCH = 32  # L=150 eval is lighter than the L=250 run that used 32
SELECTION_FACE_COUNT = 2908  # frozen minival query-grid count (formal12 rule)
N_TRAIN_FACES_EXPECTED = 18935  # LODO train-side face inventory (L-independent, frozen ends)
N_EXAM_FACES_EXPECTED = 2952  # holdout-date minival eval_mask face inventory

MF250_LR1E4_PARTNER = "M_F250_lr1e4_20260906T094633Z"  # the successful 1e-4 L=250 run (SEL-2 0.5522)

LR_1E4_VARIANT_MARKER = (
    "TRN-1-LR1e4 variant: deviation from matrix §0 peak 3e-4; 1e-4 caliber "
    "adopted per user directive 2026-09-06 (F250@1e-4 success 0.5522; F700@1e-4 "
    "plateau EARLY_FAIL)"
)
LR_AUTHORIZATION_NOTE = (
    "user directive 2026-09-06: 'F700 效果差已确认，立即下探 L=150' — this cell "
    "pairs with M-F250@1e-4 (0.5522) as the short-window L pair 150 vs 250 under "
    "the identical 1e-4 recipe; L=150 reinstated by the same directive (the "
    "registered matrix had withdrawn L=150 cells)"
)

# Reference comparisons (receipt context only, NOT gates):
REFERENCES = {
    "mf250_lr1e4_partner": {
        "results_dir": f"btransform_unified_v1/results/h1_matrix/{MF250_LR1E4_PARTNER}",
        "sel2_pick_epoch": 24,
        "sel2_pick_exam_ema_equal_mean": 0.552227817651099,
        "sel2_pick_exam_ema_pooled": 0.5492458863348032,
        "endpoint24_exam_ema_equal_mean": 0.552227817651099,
        "train_mse_ep10": 0.00481,
        "exam_raw_ep10_pooled": 0.1795594418965678,
        "note": "L=250 1e-4 run — the short-window L-pair partner (same CAL-2/M3, seed, recipe, LODO)",
    },
    "mf700_lr1e4_early_fail": {
        "results_dir": "btransform_unified_v1/results/h1_matrix/M_F700_lr1e4_20260906T104217Z",
        "epoch_at_fail": 10,
        "train_mse_ep10": 0.006925423310371152,
        "exam_raw_ep6_peak": -0.0007187117472904347,
        "exam_raw_ep10": -0.011269722643552305,
        "note": "L=700 1e-4 EARLY_FAIL: plateau L-dependence — 1e-4 breaks the plateau at L=250 (0.00481) but not at L=700 (0.00693)",
        "source": "early_fail_receipt.json (sealed, sha256 verified)",
    },
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

# L-axis judgement bands for F150 vs F250@1e-4 (preregistered BEFORE training):
L_AXIS_BANDS = {
    "definition": "delta = exam EMA equal_session_mean(M-F150) - 0.552227817651099 (F250@1e-4 SEL-2 pick / endpoint24)",
    "band_keep_150": "delta >= -0.01: shortest viable window reaches 150 (temporal kernel down another 40% vs 250; vs the settled 700 a ~78.6% cut)",
    "band_hole": "-0.02 <= delta < -0.01: mild horizon cost — record and decide by the deployment latency budget (150 still in play)",
    "band_drop": "delta < -0.02 (user '>0.02 显著掉'): horizon sits between 150 and 250 — lock L=250",
    "source": "user directive 2026-09-06 (F150 >= 0.5522-0.01 -> 150; 显著掉 >0.02 -> 视界在 150-250 间, 250 锁定)",
}

PICKS = {
    "CAL": [
        "CAL-2 {fixed budget M3 for banks AND deployment; matrix doc §0 revised 2026-09-06: 全矩阵 CAL-2/M3 固定预算 (REVIEW_MATRIX_H1_L_IDENTITY_V1_20260906.md blocking item A — formal matrix-doc revision, not a script-level downgrade)}",
        "CAL-2 REASON {adapter stub: trialization for 7/5/4 not wired (only located trialized source = C2 M3 payload, 3 trials/session); CAL-1 deferred as upgrade cell; M3 bank bitwise-reproduces the five-arm cache bank (provenance loop closed by adapters.build_h1_bank at budget 3)}",
        "CAL-2 PAIR CONSISTENCY {M-F150 (this cell), M-F250@1e-4 and M-F700@1e-4 all run CAL-2/M3, so the L-axis comparisons 150 vs 250 vs 700 are controlled on calibration budget}",
    ],
    "TRN": [
        "TRN-1 {24ep, AdamW wd 0.01 clip 1.0, warmup 1ep -> peak -> cosine (min_factor 0.1), EMA 0.9995, effective batch 32, bf16 autocast forward+loss with pred.float() before MSE}",
        "TRN-1-LR1e4 variant {peak 1e-4 (warmup 597 -> cosine 14328 -> floor 1e-5); authorized per user directive 2026-09-06 after F250@1e-4 success 0.5522 and F700@1e-4 plateau EARLY_FAIL}",
        "TRN-1 update caliber {updates/epoch = LODO-measured per-session-ceil caliber (sum_s ceil(n_s/32), 597; batches never mix sessions); NOT the formal12 full-train 731 — recorded for reference only}",
        "TRN-3 {whole-unit dropout p=0.10, training mode only, per-batch CPU generator, domain m2_small_unit_dropout (S1 unit_dropout_seed import-or-replicate)}",
        "TRN-5 {P: 0; window_override 150 (l_in 150) — L=150 reinstated per user directive after F250@1e-4 success 0.5522 and F700 plateau; outside registered MATRIX_L {250,350}, run by explicit user directive, skeleton files untouched}",
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
        "btransform_unified_v1 BTransformerUnifiedDecoderIdentity (H1 matrix cell M-F150: "
        "override_window 150, prefix 0, identity_mode proj_add — P=Linear(700->16,bias=False) "
        "fused into the local conv channels, token_in 20, seed 42; L=150 reinstated per user "
        "directive after F250@1e-4 success 0.5522 and F700 plateau) — NOT SPINT"
    ),
    "consumer": (
        "8-slot + CausalPE4 unified decoder (this series), H1 L x identity training matrix "
        "(MATRIX_H1_L_IDENTITY_V1_20260906) short-window probe; matrix-cell readout, no official "
        "submission"
    ),
    "calibration_object": (
        "E0 [176,700] C2 fused identity, materialized by the frozen C2 e15 materializer "
        "(ckpt sha ce46267e...) over the M3 payload trialized activity [3,1024,176] — "
        "bitwise-reproduces the five-arm cache bank (adapter provenance loop); "
        "H-C carrier [176,4]; CAL-2 fixed M3 (coordinator ruling: CAL-1 {7,5,4,3} "
        "downgraded — trialization for 7/5/4 not wired; deferred as upgrade cell); "
        "E0/carrier are session-static and L-INDEPENDENT (identical objects to the F250 run)"
    ),
    "scoring_surface": (
        "LODO method-selection exam: holdout date 1925-01-20 (2 sessions, 2,952 minival "
        "eval_mask faces) trained-never; plus the preregistered 2,908 full-window selection "
        "face (minival query_starts+699, all 13 sessions) for auxiliary reporting; EMA+RAW "
        "per epoch; pooled + session-mean dual report; target coordinates (session,end) "
        "FROZEN — identical to the F250 run (only the input reach shortens 250 -> 150)"
    ),
    "scale": (
        "x20 train / score pred/20 (ratio bridge MSE(raw,20y) = 400*MSE(raw/20,y), rel tol "
        "1e-9, asserted every epoch on the LODO exam)"
    ),
    "single_difference_vs_historical_best": (
        "vs M-F250@1e-4 (SEL-2 0.5522): the single differing row is 'system' window "
        "(250 -> 150); proj_add is L-independent (no template alignment), so the L axis and "
        "the usage axis do not interact through the identity pathway. L-axis consistency: "
        "M-F150 shares CAL-2/M3, seed 42, 1e-4 recipe, and LODO exam with M-F250@1e-4 AND "
        "M-F700@1e-4 — 150-vs-250-vs-700 reads as one controlled L axis under 1e-4"
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
        "schema": "btransform_unified_v1_mf150_gpu_preflight",
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
        "gpu1_owner": "M2 projadd session (parallel agent); CUDA pinned to 0 here; never touched",
        "arbitration_log": str(ARBITRATION_LOG),
        "note": (
            "M-F700 sessions released GPU0 (EARLY_FAIL sealed 2026-09-06T11:41Z); GPU0 "
            "graphics-only residue ~453 MiB is under the 500 MiB foreign bar"
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
# Data: LODO split + faces (adapter banks at 250, X re-windowed to 150)
# ---------------------------------------------------------------------------


def _rebank_150(surface: str, session: str) -> tuple[TaskBank, dict[str, Any]]:
    """build_h1_bank(window=250) then re-window X_store to L=150.

    Ends are recomputed from the cache row with the adapter's own frozen rule
    and asserted identical to the bank's window_ids, then the 150-window store
    is asserted bitwise-equal to the tail-150 of the 250-window store
    (end-anchored truncation parity — exactly what build_h1_bank(window=150)
    would produce had the adapter's MATRIX_L gate allowed it).
    """
    bank = adapters.build_h1_bank(surface, session, budget=3, window=BANK_BUILD_WINDOW, identity_mode="proj_add")
    row = adapters._h1_source_cache()[surface][session]
    neural = np.ascontiguousarray(row["neural"], dtype=np.float32)
    if surface == "train":
        starts = np.asarray(row["query_starts"], dtype=np.int64)
        ends = starts + h1_config.FULL_WINDOW - 1  # P0-4: end = start + 699, frozen
        eval_mask = np.asarray(row["eval_mask"], dtype=np.bool_)
        ends = ends[(ends >= 0) & (ends < eval_mask.shape[0])]
        ends = ends[eval_mask[ends]] if eval_mask.size else ends
    else:
        ends = np.flatnonzero(np.asarray(row["eval_mask"], dtype=np.bool_)).astype(np.int64)
    plan.require(
        np.array_equal(np.asarray(bank.window_ids, dtype=np.int64), ends),
        f"recomputed ends != bank.window_ids for {surface}/{session}",
    )
    x150 = adapters._h1_windows(neural, ends, WINDOW)
    plan.require(
        np.array_equal(x150, bank.X_store[:, -WINDOW:, :]),
        f"tail-150 truncation parity failed for {surface}/{session}",
    )
    meta = dict(bank.calibration_meta)
    meta["window"] = WINDOW
    meta["x_store_sha256"] = array_sha256(x150)
    meta["rewindow_note"] = (
        f"X_store re-windowed 250 -> {WINDOW} by adapters._h1_windows (end-anchored, "
        "left zero-pad; five-arm mechanism) after an allowed build_h1_bank(window=250) "
        "call; target coordinates (session,end) frozen; tail-parity asserted bitwise"
    )
    bank150 = dataclasses.replace(bank, X_store=np.ascontiguousarray(x150, dtype=np.float32), calibration_meta=meta)
    parity = {
        "n_windows": int(len(ends)),
        "ends_sha256": array_sha256(ends.astype(np.int64)),
        "x150_sha256": array_sha256(x150),
        "tail_parity": True,
        "E0_unchanged": True,
    }
    return bank150, parity


def build_faces() -> dict[str, Any]:
    """Banks + face arrays for train / lodo-exam / 2908-selection surfaces."""
    split = h1_config.lodo_split()
    train_sessions = list(split["train_sessions"])
    holdout_sessions = list(split["holdout_sessions"])

    train_banks, train_X, train_y, train_ids, parity_log = {}, {}, {}, {}, {}
    for s in train_sessions:
        bank, parity = _rebank_150("train", s)
        train_banks[s] = bank
        train_X[s] = bank.X_store
        train_y[s] = bank.target_store
        train_ids[s] = bank.window_ids
        parity_log[f"train/{s}"] = parity

    exam_banks, exam_X, exam_y, exam_ids = {}, {}, {}, {}
    for s in holdout_sessions:
        bank, parity = _rebank_150("minival", s)
        exam_banks[s] = bank
        exam_X[s] = bank.X_store
        exam_y[s] = bank.target_store
        exam_ids[s] = bank.window_ids
        parity_log[f"exam/{s}"] = parity

    # 2,908 selection face: minival query grid (query_starts + 699), all 13
    # sessions, windowed with the adapter's own generalized _windows.
    cache = adapters._h1_source_cache()
    sel_banks, sel_X, sel_y, sel_ids = {}, {}, {}, {}
    for s in h1_config.H1_ALL_SESSIONS:
        bank = exam_banks.get(s) or train_banks.get(s)
        if bank is None:
            b, parity = _rebank_150("minival", s)
            sel_banks[s] = b
            parity_log[f"sel/{s}"] = parity
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
    n_exam = int(sum(len(v) for v in exam_ids.values()))
    plan.require(n_exam == N_EXAM_FACES_EXPECTED, f"LODO exam face count {n_exam} != {N_EXAM_FACES_EXPECTED}")

    return {
        "split": split,
        "train": {"sessions": train_sessions, "banks": train_banks, "X": train_X, "y": train_y, "ids": train_ids},
        "exam": {"sessions": holdout_sessions, "banks": exam_banks, "X": exam_X, "y": exam_y, "ids": exam_ids},
        "sel": {"sessions": list(h1_config.H1_ALL_SESSIONS), "banks": sel_banks, "X": sel_X, "y": sel_y, "ids": sel_ids},
        "rewindow_parity": parity_log,
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
# ep10 health adjudication (coordinator protocol 2026-09-06, same as M-F700)
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
HEALTH_PROTOCOL_TEXT = (
    "coordinator 2026-09-06 (same rule set as M-F700/M-F250): ruling at ep10 "
    "only; CONTINUE iff train_mse <= 0.0065 AND examRAW >= -0.007; EARLY FAIL "
    "iff train_mse pinned at 0.0069+/-0.0001 OR examRAW declining 3 "
    "consecutive epochs below -0.01; EMA washout never rules; no "
    "self-authorized restart; healthy reference at ep10: F250@1e-4 loss "
    "0.00481 / examRAW +0.18"
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
            "protocol": HEALTH_PROTOCOL_TEXT,
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

    if lr_peak is None:
        lr_peak = 1.0e-4  # this cell exists AS the 1e-4 short-window probe (user directive)
    plan.require(
        float(lr_peak) in (3.0e-4, 1.0e-4),
        f"lr_peak {lr_peak} outside the two authorized calibers {{3e-4 TRN-1, 1e-4 formal12 variant}}",
    )

    # geometry: h1_matrix_geometry minus the MATRIX_L gate (150 is a user-directive
    # window; the skeleton files stay untouched — see module docstring)
    geometry = dict(plan.TASK_GEOMETRY["h1"])
    geometry["e0_dim"] = h1_config.matrix_e0_dim("proj_add")
    geometry["task"] = f"h1-matrix-L{WINDOW}-proj_add"
    model = BTransformerUnifiedDecoderIdentity(
        geometry, seed=SEED, override_prefix=0, override_window=WINDOW, identity_mode="proj_add"
    ).to(device)
    meta = model.init_meta
    plan.require(meta["identity_mode"] == "proj_add" and meta["matrix_letter"] == "f", "wrong identity mode")
    plan.require(meta["token_in"] == 20 and meta["proj_out_dim"] == 16, "proj_add geometry drift")
    plan.require(
        model.window == WINDOW and model.l_in == WINDOW and model.window_override == WINDOW,
        "window override drift (expected 150)",
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
        "M-F150/M-F250 parameter-count parity failed — the L-axis pair must differ "
        "only in L (proj_add is L-independent; PE is a sinusoidal buffer)",
    )
    param_parity = {
        "mf150_param_count": int(meta["param_count"]),
        "mf250_param_count": int(partner_model.init_meta["param_count"]),
        "equal": True,
    }
    del partner_model

    # startup sanity: causality on a real holdout window
    exam0_s = faces["exam"]["sessions"][0]
    causal = model.causal_check(torch.from_numpy(faces["exam"]["X"][exam0_s][:2]).to(device), faces["exam"]["banks"][exam0_s])
    plan.require(causal["passed"], "causal_check failed on real data")

    train_sessions = faces["train"]["sessions"]
    n_train_faces = int(sum(len(faces["train"]["X"][s]) for s in train_sessions))
    plan.require(
        n_train_faces == N_TRAIN_FACES_EXPECTED,
        f"LODO train face count {n_train_faces} != {N_TRAIN_FACES_EXPECTED} (inventory; ends frozen so L-independent)",
    )

    # TRN-6 memory probe -> microbatch decision (32 direct preferred at L=150)
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
    plan.require(
        updates_per_epoch == 597,
        f"update caliber drift: expected 597/ep (F250@1e-4 caliber), got {updates_per_epoch}",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr_peak), weight_decay=plan.WEIGHT_DECAY,
        betas=(0.9, 0.999), eps=1e-8,
    )
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    rng = np.random.default_rng(SEED)

    split = faces["split"]
    variant_picks = {k: list(v) for k, v in PICKS.items()}
    _seal(
        dest / "run_meta.json",
        {
            "schema": "btransform_unified_v1_mf150_run_meta",
            "cell": CELL,
            "cell_id": CELL_ID,
            "lr_peak": float(lr_peak),
            "lr_variant": LR_1E4_VARIANT_MARKER if float(lr_peak) == 1.0e-4 else None,
            "lr_authorization": LR_AUTHORIZATION_NOTE if float(lr_peak) == 1.0e-4 else None,
            "matrix_doc": (
                "MATRIX_H1_L_IDENTITY_V1_20260906 — L=150 OUTSIDE registered MATRIX_L "
                "{250,350}: reinstated per user directive 2026-09-06 after "
                "F250@1e-4 success 0.5522 and F700@1e-4 plateau EARLY_FAIL"
            ),
            "utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "geometry": geometry,
            "window": model.window,
            "l_in": model.l_in,
            "window_override": model.window_override,
            "identity_mode": model.identity_mode,
            "init_meta": meta,
            "param_count": meta["param_count"],
            "proj_param_count": meta.get("proj_param_count"),
            "l_axis_param_parity": param_parity,
            "l150_construction": {
                "geometry": "dict(plan.TASK_GEOMETRY['h1']) + matrix_e0_dim('proj_add') — h1_matrix_geometry minus the MATRIX_L gate (skeleton untouched)",
                "banks": f"adapters.build_h1_bank(budget=3, window={BANK_BUILD_WINDOW}, proj_add) then X_store re-windowed via adapters._h1_windows to {WINDOW}",
                "ends_rule": "target coordinates (session,end) FROZEN (train ends=query_starts+699 filtered by eval_mask; minival ends=eval_mask bins; sel2908 ends=query_starts+699); counts L-independent",
                "tail_parity": "X150 == X250[:, -150:, :] asserted bitwise per session (end-anchored truncation)",
                "rewindow_parity_log": faces["rewindow_parity"],
            },
            "causal_check_startup": causal,
            "lodo_split": split,
            "train_face_counts": {s: int(len(faces["train"]["X"][s])) for s in train_sessions},
            "exam_face_counts": {s: int(len(faces["exam"]["X"][s])) for s in faces["exam"]["sessions"]},
            "selection_face_counts": {s: int(len(faces["sel"]["X"][s])) for s in faces["sel"]["sessions"]},
            "n_train_faces": n_train_faces,
            "cal": {
                "mode": "CAL-2 fixed M3 (matrix §0 revision 2026-09-06, blocking item A)",
                "budget": h1_config.DEPLOY_BUDGET,
                "cal1_rotation": "not_used — CAL-1 {7,5,4,3} deferred (adapter stub: NWB re-trialization not wired)",
            },
            "microbatch_probe": probe,
            "microbatch_decision": {"micro": micro, "accum": accum, "effective_batch": EFFECTIVE_BATCH},
            "updates_per_epoch": updates_per_epoch,
            "total_updates": total_updates,
            "warmup_updates": warmup_updates,
            "formal_update_caliber_recorded": formal,
            "update_caliber_note": (
                "per-session batching (batches never mix sessions): optimizer steps/epoch "
                "= sum_s ceil(n_s/32) = 597 (identical to the F250@1e-4 caliber — ends "
                "frozen, counts L-independent); warmup=1 epoch (597 steps), cosine span=24 "
                "epochs (14328 steps); the formal full-13-session caliber 731 upd/ep is "
                "reference only"
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
                "raw_floor": HEALTH_RAW_FLOOR,
                "text": HEALTH_PROTOCOL_TEXT,
                "healthy_reference_ep10": {"mf250_lr1e4_loss": 0.00481, "mf250_lr1e4_exam_raw_pooled": 0.1795594418965678},
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
            "gpu1_owner": "M2 projadd session (parallel agent); never touched by this cell",
            "arbitration_log": str(ARBITRATION_LOG),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "source_cache_sha256": h1_config.SOURCE_CACHE_SHA256,
            "c2_payload_sha256": h1_config.C2_M3_PAYLOAD_SHA256,
            "c2_ckpt_sha256": h1_config.C2_CKPT_SHA256,
            "picks": variant_picks,
            "references_not_gates": REFERENCES,
            "l_axis_bands_preregistered": L_AXIS_BANDS,
        },
    )

    metrics_path = dest / "metrics.jsonl"
    heartbeat = dest / "heartbeat.json"
    (dest / "PROGRESS_10MIN.md").write_text(
        f"# M-F150 progress (per-epoch; ep10 health protocol inline)\n\n"
        f"Cell: {CELL}, peak_lr={float(lr_peak):g}, floor={float(lr_peak) * plan.LR_MIN_FACTOR:g}, "
        "TRN-1-LR1e4 variant (user directive 2026-09-06)\n"
        f"GPU0 {GPU0_UUID}, micro {micro}x{accum}, {updates_per_epoch} upd/ep; "
        f"L=150 (reinstated per user directive)\n\n"
        "| ep | train_mse | examRAW pooled | examRAW eqmean | examEMA pooled | sel2908 RAW (aux) | sel2908 EMA (aux) | ruling |\n"
        "|----|-----------|----------------|----------------|----------------|-----------------|------------------|--------|\n",
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
                            "schema": "btransform_unified_v1_mf150_budget_hit",
                            "cell": CELL,
                            "epoch": epoch,
                            "global_step": global_step,
                            "budget_seconds": BUDGET_SECONDS,
                            "elapsed_seconds": time.monotonic() - started,
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "note": "4h GPU budget hit before finishing 24 epochs; no rerun authorized",
                        },
                    )
                    raise RuntimeError("M-F150 4h GPU budget hit")
                take = idx[offset : offset + EFFECTIVE_BATCH]
                micro_losses = []
                for m_off in range(0, len(take), micro):
                    m_take = take[m_off : m_off + micro]
                    batch_id += 1
                    global_step += 1
                    lr = warmup_cosine_lr(
                        global_step, total_steps=total_updates, warmup_steps=warmup_updates,
                        peak=float(lr_peak), min_factor=plan.LR_MIN_FACTOR,
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
        # health adjudication (coordinator protocol; ruling only at ep10)
        health = health_adjudication(
            epoch, train_mse_series,
            {e: exam_raw_series[e]["pooled_r2"] for e in exam_raw_series},
        )
        _append_jsonl(dest / "health.jsonl", health)
        with (dest / "PROGRESS_10MIN.md").open("a", encoding="utf-8") as prog:
            prog.write(
                f"| {epoch} | {row['train_mse']:.6f} | {exam_raw['pooled_r2']:.4f} | "
                f"{exam_raw['equal_session_mean']:.4f} | {exam_ema['pooled_r2']:.4f} | "
                f"{sel_raw['pooled_r2']:.4f} | {sel_ema['pooled_r2']:.4f} | {health['ruling']} |\n"
            )
        print(
            f"[mf150] ep{epoch} loss={train_mse_series[epoch]:.5f} "
            f"examRAW={exam_raw['pooled_r2']:.4f} "
            f"examEMA={exam_ema['pooled_r2']:.4f}/{exam_ema['equal_session_mean']:.4f} "
            f"selEMA={sel_ema['pooled_r2']:.4f} "
            f"health={health['ruling']} "
            f"({row['seconds']:.0f}s)",
            flush=True,
        )
        if health["ruling"] == "EARLY_FAIL":
            _seal(
                dest / "early_fail_receipt.json",
                {
                    "schema": "btransform_unified_v1_mf150_early_fail",
                    "cell": CELL,
                    "cell_id": CELL_ID,
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "epoch_at_fail": epoch,
                    "lr_peak": float(lr_peak),
                    "lr_variant": LR_1E4_VARIANT_MARKER if float(lr_peak) == 1.0e-4 else None,
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
                    "plateau_L_dependence_context": (
                        "1e-4 breaks the plateau at L=250 (0.00481) but not at L=700 "
                        "(0.00693, EARLY_FAIL receipt M_F700_lr1e4_20260906T104217Z); this "
                        "L=150 fail/plateau reading feeds the short-window horizon decision "
                        "(vs F250@1e-4 0.5522: drop >0.02 -> horizon in 150-250, lock 250)"
                    ),
                    "picks": PICKS,
                    "note_six_rows": NOTE_SIX_ROWS,
                    "note": (
                        "early FAIL per the coordinator health protocol; GPU0 released; NO "
                        "self-authorized restart — the L-axis reading is recorded and awaits "
                        "the coordinator ruling"
                    ),
                    "gpu_release": {"gpu0_uuid": GPU0_UUID, "released_utc": datetime.now(timezone.utc).isoformat()},
                },
            )
            print("[mf150] EARLY_FAIL — receipt sealed, GPU0 released", flush=True)
            return {
                "schema": "btransform_unified_v1_mf150_train_receipt",
                "status": "EARLY_FAIL",
                "cell": CELL,
                "cell_id": CELL_ID,
                "lr_peak": float(lr_peak),
                "lr_variant": LR_1E4_VARIANT_MARKER if float(lr_peak) == 1.0e-4 else None,
                "epochs_completed": list(range(1, epoch + 1)),
                "train_mse": train_mse_series,
                "lodo_exam_ema_equal_mean_by_epoch": {e: exam_ema_series[e]["equal_session_mean"] for e in exam_ema_series},
                "lodo_exam_raw_by_epoch": {e: exam_raw_series[e]["pooled_r2"] for e in exam_raw_series},
            }
        ckpt = {
            "schema": "btransform_unified_v1_mf150_ckpt",
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
        "schema": "btransform_unified_v1_mf150_train_receipt",
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
# SEL reports + L-axis delta + cell receipt
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


def _mf250_lr1e4_partner() -> Path | None:
    """The successful 1e-4 L=250 partner run (cell_receipt sealed, 24 epochs)."""
    root = plan.RESULT_ROOT / "h1_matrix"
    preferred = root / MF250_LR1E4_PARTNER
    if (preferred / "cell_receipt.json").is_file():
        return preferred
    for d in sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith("M_F250_lr1e4")),
        reverse=True,
    ):
        if (d / "cell_receipt.json").is_file() and (d / "metrics.jsonl").is_file():
            return d
    return None


def mf250_partner_consistency(my_split: dict[str, Any]) -> dict[str, Any]:
    """Same CAL-2/M3, same holdout, same recipe/seed — only variable L (150 vs 250)."""
    out: dict[str, Any] = {
        "declaration": (
            "M-F150 and M-F250@1e-4 form the short-window L-axis controlled pair: same "
            "proj_add injection, same CAL-2/M3 fixed budget, same LODO holdout, same "
            "1e-4 recipe (warmup 597 -> cosine 14328 -> 1e-5), same seed 42 — 唯一变量 "
            "L (150 vs 250); with M-F700@1e-4 the axis extends to 700"
        ),
    }
    partner = _mf250_lr1e4_partner()
    out["partner_results_dir"] = str(partner) if partner else None
    if partner is None:
        out["assertion"] = "PENDING_PARTNER (M-F250-lr1e4 receipt not found)"
        return out
    meta = json.loads((partner / "run_meta.json").read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "same_holdout_date": meta.get("lodo_split", {}).get("holdout_date") == my_split["holdout_date"],
        "same_holdout_sessions": meta.get("lodo_split", {}).get("holdout_sessions") == my_split["holdout_sessions"],
        "same_train_sessions": meta.get("lodo_split", {}).get("train_sessions") == my_split["train_sessions"],
        "same_identity_mode": meta.get("identity_mode") == "proj_add",
        "same_seed": int(meta.get("seed", -1)) == SEED,
        "same_epochs": int(meta.get("epochs", -1)) == EPOCHS,
        "same_peak_lr_1e4": abs(float(meta.get("peak_lr", 0.0)) - 1.0e-4) < 1e-15,
    }
    out["partner_checks"] = {k: bool(v) for k, v in checks.items()}
    out["assertion"] = "CONSISTENT" if all(bool(v) for v in checks.values()) else "MISMATCH"
    return out


def l_axis_delta_vs_f250(my_receipt_numbers: dict[str, Any]) -> dict[str, Any]:
    """delta = R2(150) - R2(250) on the LODO exam, one SEL-2 rule for both."""
    out: dict[str, Any] = {
        "definition": "ΔR² = R²(M-F150, L=150) − R²(M-F250, L=250), same proj_add/CAL-2/M3/1e-4 recipe/holdout/seed",
        "primary_statistic": "LODO exam EMA equal_session_mean at each cell's SEL-2 pick epoch (earliest max)",
        "bands": L_AXIS_BANDS,
    }
    partner = _mf250_lr1e4_partner()
    if partner is None:
        out["status"] = "PENDING_PARTNER (F250-lr1e4 metrics not found; compute later)"
        return out
    theirs = _read_epoch_metrics(partner)
    if not theirs:
        out["status"] = "PENDING_PARTNER_SCORES"
        return out

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
        out["status"] = "PENDING_PARTNER_SCORES (partner exam EMA series empty)"
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
        if delta >= -0.01:
            return (
                "ΔR² >= -0.01 (user band): shortest viable window reaches 150 — "
                "temporal kernel down another 40% vs 250 (78.6% below the settled 700)"
            )
        if delta >= -0.02:
            return (
                "-0.02 <= ΔR² < -0.01: mild horizon cost recorded — decide by the "
                "deployment latency budget (150 still in play)"
            )
        return (
            "ΔR² < -0.02 (user '>0.02 显著掉'): horizon sits between 150 and 250 — "
            "lock L=250"
        )

    out.update(
        {
            "status": f"COMPUTED (partner epochs: {len(theirs)}/24)",
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
        },
    )
    return out


def run_cell_receipt(dest: Path, faces: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    metrics = _read_epoch_metrics(dest)
    plan.require(bool(metrics), "no epoch rows in metrics.jsonl")

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
    geometry = dict(plan.TASK_GEOMETRY["h1"])
    geometry["e0_dim"] = h1_config.matrix_e0_dim("proj_add")
    geometry["task"] = f"h1-matrix-L{WINDOW}-proj_add"
    model = BTransformerUnifiedDecoderIdentity(
        geometry, seed=SEED, override_prefix=0, override_window=WINDOW, identity_mode="proj_add"
    ).to(device)
    model.load_state_dict(ckpt["raw_state_dict"])
    ema = DecoderEMA.__new__(DecoderEMA)
    ema.decay = plan.EMA_DECAY
    ema.n_updates = int(ckpt["ema"]["n_updates"])
    ema.shadow = {k: v.detach().clone() for k, v in ckpt["ema"]["shadow"].items()}
    p_brief = proj_brief(model, ema)
    p_apply = proj_apply_stats(model, faces["sel"], faces["sel"]["sessions"])
    pick_sel_detail = _score_view(model, ema, "EMA", faces["sel"], faces["sel"]["sessions"], device)
    pick_exam_detail = _score_view(model, ema, "EMA", faces["exam"], faces["exam"]["sessions"], device)
    _write_json(dest / "sel2_pick_detail.json", {"sel2908_ema": pick_sel_detail, "lodo_exam_ema": pick_exam_detail})

    run_meta = json.loads((dest / "run_meta.json").read_text(encoding="utf-8"))
    l_axis = l_axis_delta_vs_f250(
        {"pick": pick, "exam_ema_equal_mean": exam_ema_eq, "exam_ema_pooled": exam_ema}
    )
    partner_consistency = mf250_partner_consistency(faces["split"])

    receipt = {
        "schema": "btransform_unified_v1_mf150_cell_receipt",
        "cell": CELL,
        "cell_id": CELL_ID,
        "matrix": (
            "H1 L x identity (MATRIX_H1_L_IDENTITY_V1_20260906) — mode (f) proj_add, "
            "short-window probe L=150 reinstated per user directive after F250@1e-4 "
            "success 0.5522 and F700@1e-4 plateau EARLY_FAIL"
        ),
        "utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "peak_lr": run_meta.get("peak_lr", 1.0e-4),
        "lr_variant": run_meta.get("lr_variant"),
        "lr_authorization": run_meta.get("lr_authorization"),
        "gate": "NONE (matrix cell; readings feed the matrix interpretation rules — matrix doc §2 feeding-method reading: (f) vs later (a)/(b))",
        "references_not_gates": REFERENCES,
        "lodo_split": faces["split"],
        "l_axis_pair": partner_consistency,
        "l_axis_delta_vs_f250_lr1e4": l_axis,
        "l_axis_judgement_preregistered": L_AXIS_BANDS,
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
                for e in (1, 10, 24) if e in metrics
            },
        },
        "proj_add_brief": {
            "note": "P = frontend.e0_proj.weight [16,700]; RAW = SEL-2 pick epoch RAW, EMA = same epoch EMA shadow; P is L-independent (identical geometry across the L axis)",
            **p_brief,
            **p_apply,
        },
        "picks": PICKS,
        "note_six_rows": NOTE_SIX_ROWS,
        "artifacts": {
            "run_meta": "run_meta.json (sealed)",
            "train_receipt": "train_receipt.json (sealed)",
            "metrics": "metrics.jsonl (+ health.jsonl per-epoch adjudication)",
            "sel2_pick_detail": "sel2_pick_detail.json",
            "checkpoints": f"epoch_001.pt..epoch_{EPOCHS:03d}.pt",
            "face_inventory": "face_inventory.json (L=150 counts L-independent; tail-parity log)",
        },
    }
    _write_json(dest / "cell_receipt_content.json", receipt)
    _seal(dest / "cell_receipt.json", receipt)
    return receipt


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="H1 matrix cell M-F150 (proj_add, L=150 user-directive window, LODO)")
    parser.add_argument("--stage", choices=["preflight", "build", "train", "receipt", "all"], default="all")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--micro", type=int, default=None, help="override microbatch (32 or 16)")
    parser.add_argument(
        "--peak-lr", type=float, default=None,
        help="override peak LR (default 1e-4 per the user directive; 3e-4 allowed for reference only)",
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
    print(f"[mf150] dest = {args.dest}", flush=True)

    if args.stage in ("preflight", "train", "receipt", "all"):
        pre = gpu_preflight(args.dest / "preflight_gpu.json")
        print(f"[mf150] preflight ok={pre['ok']} foreign={pre['foreign_pids_on_gpu0']}", flush=True)
        if not pre["ok"]:
            _seal(
                args.dest / "BLOCKED.json",
                {
                    "schema": "btransform_unified_v1_mf150_blocked",
                    "cell": CELL,
                    "reason": "GPU0 not clean at preflight (foreign pid >500 MiB or CUDA_VISIBLE_DEVICES!=0)",
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
                "window": WINDOW,
                "window_note": (
                    "L=150 via build_h1_bank(window=250) + adapters._h1_windows re-window; "
                    "target coordinates (session,end) frozen -> counts identical to the "
                    "M-F250 inventory (train 18935 / exam 2952 / sel 2908); X150 bitwise "
                    "== X250 tail-150 per session (end-anchored truncation parity)"
                ),
                "lodo_split": faces["split"],
                "train": {s: int(len(v)) for s, v in faces["train"]["X"].items()},
                "exam": {s: int(len(v)) for s, v in faces["exam"]["X"].items()},
                "sel": {s: int(len(v)) for s, v in faces["sel"]["X"].items()},
                "train_total": int(sum(len(v) for v in faces["train"]["X"].values())),
                "exam_total": int(sum(len(v) for v in faces["exam"]["X"].values())),
                "sel_total": int(sum(len(v) for v in faces["sel"]["X"].values())),
                "selection_face_count_gate": SELECTION_FACE_COUNT,
                "rewindow_parity": faces["rewindow_parity"],
            },
        )
        print(
            f"[mf150] faces built in {time.monotonic()-t0:.0f}s: "
            f"train={sum(len(v) for v in faces['train']['X'].values())} "
            f"exam={sum(len(v) for v in faces['exam']['X'].values())} "
            f"sel={sum(len(v) for v in faces['sel']['X'].values())}",
            flush=True,
        )

    if args.stage in ("train", "all"):
        summary = run_train(args.dest, faces, args.micro, args.peak_lr)
        if summary.get("status") == "EARLY_FAIL":
            print("[mf150] early-fail receipt sealed; NOT proceeding to cell receipt; no self-retry", flush=True)
            return 1
        print(f"[mf150] train done in {summary['elapsed_s']:.0f}s", flush=True)

    if args.stage in ("receipt", "all"):
        if faces is None:
            faces = build_faces()
        receipt = run_cell_receipt(args.dest, faces)
        pick = receipt["sel"]["SEL-2_epoch_pick"]
        print(
            f"[mf150] SEL-2 pick e{pick} "
            f"examEMAeq={receipt['sel']['SEL-2_pick_exam_ema_equal_mean']:.4f} "
            f"examEMApooled={receipt['sel']['SEL-2_pick_exam_ema_pooled']:.4f} "
            f"auxSel2908EMA={receipt['sel']['SEL-2_auxiliary_sel2908_ema_pooled_at_pick']:.4f}",
            flush=True,
        )
        delta = receipt["l_axis_delta_vs_f250_lr1e4"]
        print(
            f"[mf150] L-axis vs F250@1e-4: {delta.get('status')} "
            f"delta_pick_eq={delta.get('delta_r2_exam_ema_equal_mean_at_pick')}",
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

            target = dest or (_plan.RESULT_ROOT / "h1_matrix" / "M_F150_error")
            target.mkdir(parents=True, exist_ok=True)
            _receipts.seal_json(
                target / "error_receipt.json",
                {
                    "schema": "btransform_unified_v1_mf150_error",
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "traceback": trace,
                },
            )
        except Exception:
            pass
        sys.exit(3)
