"""P1a — M2 true-data S1 replication on the unified skeleton (workorder §7 P1a).

Cell definition (S1-SMALL-COS replication, btransform_unified_v1):
  - data: frozen dual_track cache 20260905_101500 (source_train 7 sessions /
    101,171 windows; source_minival 7 / 1,011; ext4 4 / 2,069), read-only.
  - model: BTransformerUnifiedDecoder, m2 geometry with P=0 (L_in = window =
    50), seed 42, init via m2_dual_track_v1 initialize_decoder with the
    state-dict key set asserted IDENTICAL to SmallTransformerDecoder (77
    keys, bitwise-equal init values; RNG walk = sorted names x numel).
  - sampling: frozen 24-epoch manifest (digest a95255fa...), dual_track
    sampler.iter_manifest_batches(target_space="decoder_raw" = native x5).
  - training: 24 epochs / 75,960 updates; AdamW (dual_track build_optimizer:
    no-decay = ndim<=1 / bias / norm; wd 0.01; betas (0.9, 0.999); eps 1e-8);
    LR = linear warmup to 3e-4 over the first 3,165 updates then cosine to
    3e-5; grad clip 1.0; bf16 autocast (forward + loss, pred.float() before
    MSE); EMA 0.9995 after every step; whole-unit dropout p=0.10, training
    mode only, per-batch CPU generator seeded by THIS series'
    unit_dropout_seed(42, epoch, batch_id), mask shared across the batch
    (S1 semantics).
  - per-epoch: source-minival RAW + EMA views (variance_weighted_r2 per
    session + equal mean; pred/5 vs native), metrics.jsonl, epoch checkpoint
    (raw + EMA + optimizer states), heartbeat.
  - after training: ext4 full scoring of the EMA view for all 24 epochs +
    RAW view at endpoint24; SEL-2 epoch-pick (earliest max on ext4, EMA);
    SEL-1 endpoint24 + last-4/last-8; SEL-4 pooled/session-mean dual report.
  - gate: EMA endpoint24 equal_session_mean >= 0.42 AND per-session paired
    diff vs S1 (0.5738/0.3914/0.5010/0.3315) >= -0.02. FAIL => sealed FAIL
    receipt + diagnosis; NO hyperparameter rerun.

Discipline: CUDA_VISIBLE_DEVICES pinned to GPU1
(GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86); GPU0 never touched; preflight
refuses when a foreign pid holds >500 MiB on GPU1; 6h GPU budget from train
start (BUDGET_HIT receipt on overrun); receipts sealed via
btransform_unified_v1.receipts.seal_json (0444 + sha256 sidecar).

Usage (all stages, one process):
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 \
  BTRANSFORM_P1A_TRAIN=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v1/scripts/p1a_m2_s1_replication.py --stage all

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

from btransform_unified_v1 import adapters, plan, receipts  # noqa: E402
from btransform_unified_v1.bank import TaskBank, array_sha256  # noqa: E402
from btransform_unified_v1.ema import DecoderEMA  # noqa: E402
from btransform_unified_v1.model import (  # noqa: E402
    BTransformerUnifiedDecoder,
    S1_M2_PARAM_COUNT,
    unit_dropout_seed,
    whole_unit_dropout,
)
from btransform_unified_v1.r2 import variance_weighted_r2  # noqa: E402
from btransform_unified_v1.schedule import warmup_cosine_lr  # noqa: E402

from tfpd_exploration.src.m2_dual_track_v1 import champion as old_champion  # noqa: E402
from tfpd_exploration.src.m2_dual_track_v1 import data as old_data  # noqa: E402
from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan  # noqa: E402
from tfpd_exploration.src.m2_dual_track_v1 import sampler as old_sampler  # noqa: E402
from tfpd_exploration.src.m2_dual_track_v1 import training as old_training  # noqa: E402

CELL = "P1A-M2-S1-REPL"
SEED = 42
GPU1_UUID = "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"
FOREIGN_MEM_MIB_LIMIT = 500
BUDGET_SECONDS = 6.0 * 3600.0
MANIFEST_24_PATH = (
    WORKSPACE_ROOT
    / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/sampler/shuffled_batch_manifest_24.json"
)
MANIFEST_24_DIGEST = "a95255fa339ea06f1e1cc3ef9f53f3d49d15799bf95e4579f672417fd878d79a"
S1_METRICS_PATH = (
    WORKSPACE_ROOT
    / "tfpd_exploration/results/m2_b_small_stability_v1/20260905_123000/S1_SMALL_COS/seed42/metrics.jsonl"
)
S1_EXT4_SCAN_PATH = (
    WORKSPACE_ROOT
    / "tfpd_exploration/results/m2_b_small_stability_v1/20260905_123000/S1_SMALL_COS/seed42/ext4_epoch_scan.json"
)
P1A_GEOMETRY = {
    "task": "m2",
    "window": 50,
    "prefix": 0,
    "units": 96,
    "e0_dim": 50,
    "carrier_dim": 4,
    "out_dim": 2,
}
GATE_EQUAL_MEAN_MIN = 0.42
GATE_SESSION_DIFF_MIN = -0.02

# REF §6 picks for every receipt this run produces.
PICKS = {
    "CAL": [
        "CAL-2 {budget: 33, surface: frozen dual_track cache 20260905_101500 (read-only)}",
        "CAL-3a {identity: post_pool d_e=50 concat}",
        "CAL-4 {carrier twice: in E0 (MOVE-T4 side-in) and in token concat}",
        "CAL-5 {unit order: units DataFrame row order; unit_mask all-true (M2)}",
    ],
    "TRN": [
        "TRN-1 {updates_per_epoch: 3165, warmup_updates: 3165, total_updates: 75960, peak_lr: 3e-4, min_lr: 3e-5, AdamW wd 0.01 betas (0.9,0.999) eps 1e-8, clip 1.0, batch 32}",
        "TRN-1 precision {bf16 autocast: forward+loss, pred.float() before MSE (S1 launch.py semantics)}",
        "TRN-3 {whole-unit dropout p=0.10, training mode only, per-batch CPU generator, domain: btransform_unified_v1_unit_dropout}",
        "TRN-5 {P: 0, L_in: 50}",
        "TRN-8 {scale: x5 decoder_raw; score pred/5 vs native}",
    ],
    "SEL": [
        "SEL-2 {surface: ext4 (local), view: EMA, rule: earliest max, ties<=1e-10 -> earliest}",
        "SEL-1 {endpoint24 primary + last-4/last-8 auxiliary}",
        "SEL-4 {pooled and session-mean dual report}",
        "SEL-3 {official surface: zero participation}",
    ],
    "SPD": ["none {training-path run; SPD-A1 folding exists but is not used in training or scoring}"],
}

NOTE_SIX_ROWS = {
    "system": (
        "btransform_unified_v1 BTransformerUnifiedDecoder (m2 geometry, P=0, "
        f"L_in=50, {S1_M2_PARAM_COUNT:,} params, seed 42, state-dict keys "
        "identical to S1 SmallTransformerDecoder; init bitwise-equal) — NOT SPINT"
    ),
    "consumer": (
        "8-slot + CausalPE4 unified decoder (this series). Historical best on "
        "this surface is the SPINT family — comparisons are 'same scoring "
        "surface, different system' only (P0-1)"
    ),
    "calibration_object": (
        "E0 [96,50] B3S frozen champion (sha 25d7bc72...) via "
        "native_e0_and_u (push_trial/finalize, not batched-mean) + MOVE-T4 "
        "carrier [96,4] (fit_move_t4 -> t4_from_trial_sums, LSQ on "
        "[1,cos,sin]); M33 budget; frozen cache 20260905_101500"
    ),
    "scoring_surface": (
        "ext4 4 sessions 2,069 windows (519/490/425/635, 10-30-R1/R2, 11-18, "
        "11-19); endpoint24 EMA view primary; per-session variance_weighted_r2 "
        "+ equal_session_mean + pooled (SEL-4); batch order = eligible_starts order"
    ),
    "scale": (
        "x5 decoder_raw contract: train on 5x native target, score pred/5 "
        "against native; MSE(raw,5y) = 25*MSE(raw/5,y) (ratio, rel tol 1e-9)"
    ),
    "single_difference_vs_historical_best": (
        "vs S1-SMALL-COS/EMA endpoint24 ext4 0.4495: the single differing row "
        "is 'system' (unified-series implementation of the same 8-slot + "
        "causal-core blueprint); data/manifest/schedule/optimizer/EMA/"
        "dropout-rate are S1-identical — known float-path deviations: per-tap "
        "causal conv vs F.conv1d, single token GEMM identical, dropout mask "
        "domain differs by design (recorded in TRN-3)"
    ),
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _seal(path: Path, payload: Any) -> str:
    return receipts.seal_json(path, payload)


# ---------------------------------------------------------------------------
# GPU preflight (workorder §8: foreign pid + UUID, one card pinned)
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
        "schema": "btransform_unified_v1_p1a_gpu_preflight",
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
        "torch_cuda_visible_count": torch.cuda.device_count() if raw == "1" else None,
        "gpu0_touched": False,
    }
    ok = raw == "1" and not foreign and torch.cuda.device_count() == 1
    report["ok"] = bool(ok)
    _seal(out_path, report)
    return report


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _load_surface_banks(surface: str, device: torch.device) -> dict[str, Any]:
    sessions = list(old_plan.EXT4_SESSIONS if surface == "ext4" else old_plan.HELDIN_SESSIONS)
    return {s: old_data.load_session_bank(surface, s, device=device) for s in sessions}


def _task_banks(surface: str) -> dict[str, TaskBank]:
    sessions = list(old_plan.EXT4_SESSIONS if surface == "ext4" else old_plan.HELDIN_SESSIONS)
    return {s: adapters.build_m2_bank(surface, s, budget=33) for s in sessions}


def _score_surface(
    model: BTransformerUnifiedDecoder,
    dual_banks: dict[str, Any],
    banks: dict[str, TaskBank],
    device: torch.device,
) -> dict[str, Any]:
    """Per-session variance_weighted_r2 (pred/5 vs native) + equal mean.

    Mirrors S1 launch._score_minival: iter_session_batches(batch 32, native
    target space) in eligible_starts order; R^2 over the flattened [S,2]
    block is identical to dual_track contracts.variance_weighted_r2.
    """
    per_session: dict[str, float] = {}
    native_mses: list[float] = []
    model.eval()
    for session, dual_bank in dual_banks.items():
        targets: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        for batch in old_data.iter_session_batches(
            dual_bank,
            batch_size=old_plan.EFFECTIVE_BATCH,
            device=device,
            target_space=old_plan.SCORING_TARGET_SPACE,
        ):
            with torch.inference_mode():
                raw = model(batch.X, banks[session])
            native = raw.detach().cpu().numpy() / old_plan.BEHAVIOR_SCALE
            preds.append(np.ascontiguousarray(native, dtype=np.float32))
            targets.append(batch.last_target.detach().cpu().numpy())
        target = np.concatenate(targets, axis=0)
        pred = np.concatenate(preds, axis=0)
        per_session[session] = float(variance_weighted_r2(target, pred))
        native_mses.append(float(np.mean(np.square(target - pred))))
    return {
        "per_session_r2": dict(sorted(per_session.items())),
        "equal_session_mean": float(np.mean(list(per_session.values()))),
        "native_mse": float(np.mean(native_mses)),
    }


def _score_with_ema(
    model: BTransformerUnifiedDecoder,
    ema: DecoderEMA,
    dual_banks: dict[str, Any],
    banks: dict[str, TaskBank],
    device: torch.device,
) -> dict[str, Any]:
    """Swap the EMA shadow in, score, restore RAW (never aliases)."""
    named = model.trainable_parameters()
    backup = {name: param.detach().clone() for name, param in named.items()}
    was_training = model.training
    try:
        with torch.no_grad():
            for name, param in named.items():
                param.copy_(ema.shadow[name].to(device=param.device, dtype=param.dtype))
        model.eval()
        return _score_surface(model, dual_banks, banks, device)
    finally:
        with torch.no_grad():
            for name, param in named.items():
                param.copy_(backup[name].to(device=param.device, dtype=param.dtype))
        model.train(was_training)


def run_train(dest: Path) -> dict[str, Any]:
    device = torch.device("cuda:0")
    manifest = old_sampler.load_manifest(MANIFEST_24_PATH)
    plan.require(manifest["digest"] == MANIFEST_24_DIGEST, "24-epoch manifest digest mismatch")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    model = BTransformerUnifiedDecoder(P1A_GEOMETRY, seed=SEED).to(device)
    parity = model.assert_s1_state_dict_parity()
    n_params = sum(p.numel() for p in model.parameters())
    plan.require(n_params == S1_M2_PARAM_COUNT, f"param count {n_params}")

    train_dual = _load_surface_banks("source_train", device)
    mini_dual = _load_surface_banks("source_minival", device)
    mini_banks = _task_banks("source_minival")
    train_bank_cache: dict[str, TaskBank] = {}
    updates = old_training.count_updates(train_dual)
    plan.require(updates == 3165, f"updates/epoch {updates} != 3165")

    optimizer = old_training.build_optimizer(
        model.trainable_parameters().items(), lr=plan.LR_PEAK, weight_decay=plan.WEIGHT_DECAY
    )
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    total_updates = plan.recipe_updates("m2", epochs=plan.EPOCHS)["total_updates"]
    warmup_updates = plan.recipe_updates("m2")["warmup_updates"]

    _seal(
        dest / "run_meta.json",
        {
            "schema": "btransform_unified_v1_p1a_run_meta",
            "cell": CELL,
            "utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "geometry": P1A_GEOMETRY,
            "l_in": model.l_in,
            "decoder_params": n_params,
            "s1_param_count": S1_M2_PARAM_COUNT,
            "s1_state_dict_parity": parity,
            "init_meta": model.init_meta,
            "manifest_path": str(MANIFEST_24_PATH),
            "manifest_digest": manifest["digest"],
            "manifest_digest_expected": MANIFEST_24_DIGEST,
            "source_train_windows": {s: int(len(b.eligible_starts)) for s, b in train_dual.items()},
            "source_minival_windows": {s: int(len(b.eligible_starts)) for s, b in mini_dual.items()},
            "updates_per_epoch": updates,
            "total_updates": total_updates,
            "warmup_updates": warmup_updates,
            "lr_peak": plan.LR_PEAK,
            "lr_min": plan.LR_PEAK * plan.LR_MIN_FACTOR,
            "weight_decay": plan.WEIGHT_DECAY,
            "grad_clip": plan.GRAD_CLIP,
            "ema_decay": plan.EMA_DECAY,
            "unit_dropout_p": plan.UNIT_DROPOUT,
            "unit_dropout_domain": "btransform_unified_v1_unit_dropout",
            "precision": "bf16 autocast (forward+loss; pred.float() before MSE)",
            "batch_size": old_plan.EFFECTIVE_BATCH,
            "epochs": plan.EPOCHS,
            "cache_root_readonly": str(adapters._M2_CACHE_ROOT),
            "gpu_uuid": GPU1_UUID,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "picks": PICKS,
        },
    )

    metrics_path = dest / "metrics.jsonl"
    heartbeat = dest / "heartbeat.json"
    started = time.monotonic()
    deadline = started + BUDGET_SECONDS
    global_step = 0
    raw_scores: dict[int, float] = {}
    ema_scores: dict[int, float] = {}
    train_mse_series: dict[int, float] = {}
    lr_series: dict[int, float] = {}

    for epoch in range(1, plan.EPOCHS + 1):
        model.train()
        epoch_t0 = time.monotonic()
        running = 0.0
        n_batches = 0
        for batch_id, batch in enumerate(
            old_sampler.iter_manifest_batches(
                train_dual, manifest, epoch, device=device, target_space=old_plan.TRAINING_TARGET_SPACE
            )
        ):
            if time.monotonic() >= deadline:
                _seal(
                    dest / "budget_hit.json",
                    {
                        "schema": "btransform_unified_v1_p1a_budget_hit",
                        "cell": CELL,
                        "epoch": epoch,
                        "global_step": global_step,
                        "budget_seconds": BUDGET_SECONDS,
                        "elapsed_seconds": time.monotonic() - started,
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "note": "6h GPU budget hit before finishing 24 epochs; no rerun authorized",
                    },
                )
                raise RuntimeError("P1a 6h GPU budget hit")
            if batch.session_id not in train_bank_cache:
                train_bank_cache[batch.session_id] = adapters.build_m2_bank(
                    "source_train", batch.session_id, budget=33
                )
            global_step += 1
            lr = warmup_cosine_lr(
                global_step,
                total_steps=total_updates,
                warmup_steps=warmup_updates,
                peak=plan.LR_PEAK,
                min_factor=plan.LR_MIN_FACTOR,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            generator = torch.Generator(device="cpu")
            generator.manual_seed(unit_dropout_seed(SEED, epoch, batch_id))
            keep = whole_unit_dropout(batch.unit_mask, p=plan.UNIT_DROPOUT, generator=generator)
            bank = train_bank_cache[batch.session_id]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(batch.X, bank, dropout_keep=keep)
                loss = nn.functional.mse_loss(pred.float(), batch.last_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.trainable_parameters().values()), plan.GRAD_CLIP)
            optimizer.step()
            ema.update_after_step(model)
            running += float(loss.detach().cpu())
            n_batches += 1
            if global_step % 500 == 0:
                _append_jsonl(
                    metrics_path,
                    {
                        "event": "step",
                        "cell": CELL,
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss": float(loss.detach().cpu()),
                        "lr": float(lr),
                        "ema_updates": ema.n_updates,
                        "unix": time.time(),
                    },
                )
                _write_json(
                    heartbeat,
                    {
                        "cell": CELL,
                        "epoch": epoch,
                        "global_step": global_step,
                        "lr": float(lr),
                        "loss": float(loss.detach().cpu()),
                        "ema_updates": ema.n_updates,
                        "unix": time.time(),
                        "gpu_uuid": GPU1_UUID,
                    },
                )
        raw = _score_surface(model, mini_dual, mini_banks, device)
        ema_minival = _score_with_ema(model, ema, mini_dual, mini_banks, device)
        raw_scores[epoch] = float(raw["equal_session_mean"])
        ema_scores[epoch] = float(ema_minival["equal_session_mean"])
        train_mse_series[epoch] = running / max(n_batches, 1)
        lr_series[epoch] = float(lr)
        extras = {
            "train_mse": train_mse_series[epoch],
            "minival_raw": raw,
            "minival_ema": ema_minival,
            "seconds": time.monotonic() - epoch_t0,
            "global_step": global_step,
            "lr": float(lr),
            "ema_updates": ema.n_updates,
        }
        ckpt = {
            "schema": "btransform_unified_v1_p1a_ckpt",
            "cell": CELL,
            "epoch": epoch,
            "global_step": global_step,
            "seed": SEED,
            "manifest_digest": manifest["digest"],
            "raw_state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr": float(lr),
            "ema_decay": ema.decay,
        }
        torch.save(ckpt, dest / f"epoch_{epoch:03d}.pt")
        _append_jsonl(metrics_path, {"event": "epoch", "epoch": epoch, "cell": CELL, **extras, "unix": time.time()})
        _write_json(heartbeat, {"event": "epoch", "cell": CELL, "epoch": epoch, **extras, "unix": time.time(), "gpu_uuid": GPU1_UUID})
        # keep the train bank cache warm but bounded (7 sessions max anyway)
    summary = {
        "schema": "btransform_unified_v1_p1a_train_receipt",
        "cell": CELL,
        "seed": SEED,
        "gpu_uuid": GPU1_UUID,
        "epochs_completed": list(range(1, plan.EPOCHS + 1)),
        "global_updates": global_step,
        "ema_updates": ema.n_updates,
        "endpoint24_minival_raw": raw_scores[plan.EPOCHS],
        "endpoint24_minival_ema": ema_scores[plan.EPOCHS],
        "epoch_minival_raw": raw_scores,
        "epoch_minival_ema": ema_scores,
        "train_mse": train_mse_series,
        "lr_at_epoch_end": lr_series,
        "elapsed_s": time.monotonic() - started,
        "s1_state_dict_parity": parity,
        "manifest_digest": manifest["digest"],
        "picks": PICKS,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    _seal(dest / "train_receipt.json", summary)
    return summary


# ---------------------------------------------------------------------------
# ext4 scan (EMA all epochs + RAW endpoint24) and SEL reports
# ---------------------------------------------------------------------------


def _score_ext4(
    model: BTransformerUnifiedDecoder,
    dual_banks: dict[str, Any],
    banks: dict[str, TaskBank],
    device: torch.device,
) -> dict[str, Any]:
    """Full-ext4 scoring: per-session variance_weighted_r2 + equal mean + pooled.

    Mirrors S1 score_ext4_model (batch 32, eligible_starts order, pred/5 vs
    native, per-session prediction digest) and adds the SEL-4 pooled R^2 and
    the TRN-8 pred_std/target_std diagnostic. The pooled figure is computed
    from the SAME per-session predictions — no second inference pass.
    """
    per_session: dict[str, Any] = {}
    r2s: dict[str, float] = {}
    pooled_targets: list[np.ndarray] = []
    pooled_preds: list[np.ndarray] = []
    model.eval()
    for session, dual_bank in dual_banks.items():
        targets: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        for batch in old_data.iter_session_batches(
            dual_bank,
            batch_size=old_plan.EFFECTIVE_BATCH,
            device=device,
            target_space=old_plan.SCORING_TARGET_SPACE,
        ):
            with torch.inference_mode():
                raw = model(batch.X, banks[session])
            preds.append(
                np.ascontiguousarray(raw.detach().cpu().numpy() / old_plan.BEHAVIOR_SCALE, dtype=np.float32)
            )
            targets.append(batch.last_target.detach().cpu().numpy())
        target = np.concatenate(targets, axis=0)
        pred = np.concatenate(preds, axis=0)
        pooled_targets.append(target)
        pooled_preds.append(pred)
        r2 = float(variance_weighted_r2(target, pred))
        r2s[session] = r2
        per_session[session] = {
            "r2": r2,
            "window_count": int(target.shape[0]),
            "prediction_digest": old_champion.array_sha256(pred),
            "pred_std_over_target_std": float(pred.std() / target.std()) if target.std() > 0 else None,
        }
    equal_mean = float(np.mean([r2s[s] for s in sorted(r2s)]))
    pooled = float(
        variance_weighted_r2(np.concatenate(pooled_targets, axis=0), np.concatenate(pooled_preds, axis=0))
    )
    return {
        "per_session": per_session,
        "equal_session_mean": equal_mean,
        "pooled_r2": pooled,
        "session_count": len(per_session),
        "n_windows": int(sum(v["window_count"] for v in per_session.values())),
    }


def _apply_view(model: BTransformerUnifiedDecoder, ckpt: dict[str, Any], view: str) -> None:
    model.load_state_dict(ckpt["raw_state_dict"])
    if view == "RAW":
        return
    plan.require(view == "EMA", f"unknown view {view}")
    shadow = ckpt["ema"]["shadow"]
    named = model.trainable_parameters()
    plan.require(set(named) == set(shadow), "EMA/RAW key mismatch")
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(shadow[name].to(device=param.device, dtype=param.dtype))


def run_ext4(dest: Path, device: torch.device) -> dict[str, Any]:
    dual_banks = _load_surface_banks("ext4", device)
    banks = _task_banks("ext4")
    model = BTransformerUnifiedDecoder(P1A_GEOMETRY, seed=SEED).to(device)
    views: dict[str, Any] = {"EMA": {"all_epochs": {}}, "RAW": {"all_epochs": {}}}
    for epoch in range(1, plan.EPOCHS + 1):
        ckpt = torch.load(dest / f"epoch_{epoch:03d}.pt", map_location=device, weights_only=False)
        _apply_view(model, ckpt, "EMA")
        report = _score_ext4(model, dual_banks, banks, device)
        report.update({"epoch": epoch, "view": "EMA", "ckpt": str(dest / f"epoch_{epoch:03d}.pt")})
        views["EMA"]["all_epochs"][str(epoch)] = report
    ckpt24 = torch.load(dest / "epoch_024.pt", map_location=device, weights_only=False)
    _apply_view(model, ckpt24, "RAW")
    report = _score_ext4(model, dual_banks, banks, device)
    report.update({"epoch": 24, "view": "RAW", "ckpt": str(dest / "epoch_024.pt")})
    views["RAW"]["all_epochs"]["24"] = report
    scan = {
        "schema": "btransform_unified_v1_p1a_ext4_scan",
        "cell": CELL,
        "seed": SEED,
        "surface": "ext4",
        "sessions": list(old_plan.EXT4_SESSIONS),
        "views": views,
        "scoring": {
            "prediction_space": old_plan.TRAINING_TARGET_SPACE,
            "scoring_space": old_plan.SCORING_TARGET_SPACE,
            "divide_by_behavior_scale": True,
            "r2": "variance_weighted per session (flattened [S,2]) + equal_session_mean + pooled (SEL-4)",
            "batch_order": "eligible_starts order, batch 32, tail kept",
        },
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(dest / "ext4_epoch_scan.json", scan)
    _seal(
        dest / "ext4_scan_receipt.json",
        {
            "schema": "btransform_unified_v1_p1a_ext4_scan_receipt",
            "cell": CELL,
            "ema_equal_mean_by_epoch": {
                e: views["EMA"]["all_epochs"][str(e)]["equal_session_mean"]
                for e in range(1, plan.EPOCHS + 1)
            },
            "raw_endpoint24": views["RAW"]["all_epochs"]["24"]["equal_session_mean"],
            "ema_endpoint24_per_session": views["EMA"]["all_epochs"]["24"]["per_session"],
            "picks": PICKS,
            "scan_path": str(dest / "ext4_epoch_scan.json"),
            "utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return scan


# ---------------------------------------------------------------------------
# Gate (P1a): EMA endpoint24 equal mean >= 0.42 AND per-session paired diff >= -0.02
# ---------------------------------------------------------------------------


def _s1_reference() -> dict[str, Any]:
    scan = json.loads(S1_EXT4_SCAN_PATH.read_text(encoding="utf-8"))
    e24 = scan["views"]["EMA"]["all_epochs"]["24"]
    raw24 = scan["views"]["RAW"]["all_epochs"]["24"]
    return {
        "ema_equal_mean": float(e24["R_session_equal_mean"]),
        "ema_per_session": {k: float(v["r2"]) for k, v in e24["per_session"].items()},
        "raw_equal_mean": float(raw24["R_session_equal_mean"]),
        "source": str(S1_EXT4_SCAN_PATH),
    }


def _sel2_pick(series: dict[int, float]) -> int:
    finite = {int(e): float(v) for e, v in series.items() if math.isfinite(v)}
    plan.require(bool(finite), "no finite ext4 scores")
    best = max(finite.values())
    tied = [e for e, v in finite.items() if abs(best - v) <= 1e-10]
    return min(tied)


def _last_k(series: dict[int, float], lo: int, hi: int) -> dict[str, Any]:
    vals = [float(series[e]) for e in range(lo, hi + 1)]
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "range": [float(arr.min()), float(arr.max())],
        "values": vals,
    }


def _fail_diagnosis(dest: Path, ours: dict[int, float], ours_raw: dict[int, float]) -> dict[str, Any]:
    s1_epochs: dict[int, dict[str, float]] = {}
    s1_raw: dict[int, float] = {}
    with S1_METRICS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "epoch":
                s1_epochs[row["epoch"]] = float(row["minival_ema"]["equal_session_mean"])
                s1_raw[row["epoch"]] = float(row["minival_raw"]["equal_session_mean"])
    first_div = None
    for epoch in range(1, plan.EPOCHS + 1):
        if epoch in s1_epochs and abs(ours.get(epoch, float("nan")) - s1_epochs[epoch]) > 0.10:
            first_div = {
                "epoch": epoch,
                "ours_ema": ours.get(epoch),
                "s1_ema": s1_epochs[epoch],
                "ours_raw": ours_raw.get(epoch),
                "s1_raw": s1_raw.get(epoch),
            }
            break
    curve = {
        str(e): {"ours_ema": ours.get(e), "s1_ema": s1_epochs.get(e), "ours_raw": ours_raw.get(e), "s1_raw": s1_raw.get(e)}
        for e in range(1, plan.EPOCHS + 1)
        if e in s1_epochs
    }
    return {
        "schema": "btransform_unified_v1_p1a_fail_diagnosis",
        "comparison_surface": "source_minival equal_session_mean, per epoch (ours vs S1 metrics.jsonl)",
        "first_divergence_epoch_delta_gt_0.10": first_div,
        "per_epoch_curve": curve,
        "checklist": {
            "init_key_set_and_order": "PASS by construction: 77-key state-dict set identical to SmallTransformerDecoder; initialize_decoder RNG walk identical; init values bitwise-equal (recorded in run_meta.s1_state_dict_parity)",
            "param_count": f"PASS: {S1_M2_PARAM_COUNT:,}",
            "manifest_digest": "PASS: a95255fa... verified before training (run_meta)",
            "batch_order_first_batch_window_ids": "SKIPPED: S1 metrics.jsonl records no window_ids; manifest digest equality is the order guarantee",
            "dropout_generator_semantics": "SAME construction (per-batch CPU generator, [1,N] mask shared across batch, restore-lowest-if-empty); DIFFERENT payload domain by design (btransform_unified_v1_unit_dropout vs m2_small_unit_dropout) -> different mask stream than S1; expected trajectory-level divergence",
            "ema_update_count": f"{75960} expected; see train_receipt.ema_updates",
            "lr_trajectory": "ours = warmup_cosine_lr(1-based t) == S1 lr_at_update(S1-SMALL-COS) formula; per-epoch-end lr in metrics.jsonl",
            "x5_target_contract": "PASS: iter_manifest_batches(target_space=decoder_raw) multiplies native by 5; scoring divides pred by 5",
            "bf16_placement": "PASS: autocast wraps forward + mse_loss(pred.float(), target), optimizer step outside autocast (S1 launch.py identical)",
            "known_float_deviation_sources": "per-tap causal conv vs F.conv1d (~1e-7 class); bf16 rounding; trajectory-level divergence expected; gate is statistical (0.42), not bitwise",
        },
        "note": "failure protocol: no hyperparameter rerun authorized",
        "utc": datetime.now(timezone.utc).isoformat(),
    }


def run_gate(dest: Path) -> dict[str, Any]:
    scan = json.loads((dest / "ext4_epoch_scan.json").read_text(encoding="utf-8"))
    ema = {int(e): row["equal_session_mean"] for e, row in scan["views"]["EMA"]["all_epochs"].items()}
    raw24 = scan["views"]["RAW"]["all_epochs"]["24"]
    ref = _s1_reference()
    e24 = scan["views"]["EMA"]["all_epochs"]["24"]
    sel2 = _sel2_pick(ema)
    paired = {
        session: float(e24["per_session"][session]["r2"]) - ref["ema_per_session"][session]
        for session in ref["ema_per_session"]
    }
    equal_mean_ok = e24["equal_session_mean"] >= GATE_EQUAL_MEAN_MIN
    paired_ok = all(v >= GATE_SESSION_DIFF_MIN for v in paired.values())
    passed = bool(equal_mean_ok and paired_ok)
    metrics = {}
    with (dest / "metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "epoch":
                metrics[row["epoch"]] = row
    gate = {
        "schema": "btransform_unified_v1_p1a_gate_receipt",
        "cell": CELL,
        "gate_definition": {
            "statistic": "EMA endpoint24 ext4 equal_session_mean (session-equal mean of variance_weighted_r2)",
            "threshold_equal_mean": GATE_EQUAL_MEAN_MIN,
            "threshold_per_session_paired_diff": GATE_SESSION_DIFF_MIN,
            "reference": "S1-SMALL-COS/EMA endpoint24 ext4 (20260905_123000, ext4_epoch_scan.json)",
        },
        "pass": passed,
        "ours": {
            "ema_endpoint24_equal_session_mean": e24["equal_session_mean"],
            "ema_endpoint24_pooled_r2": e24["pooled_r2"],
            "ema_endpoint24_per_session": {
                s: e24["per_session"][s]["r2"] for s in sorted(e24["per_session"])
            },
            "raw_endpoint24_equal_session_mean": raw24["equal_session_mean"],
        },
        "s1_reference": ref,
        "paired_diff_per_session_ours_minus_s1": paired,
        "checks": {
            "equal_mean_ge_0.42": bool(equal_mean_ok),
            "all_session_paired_diff_ge_-0.02": bool(paired_ok),
        },
        "sel": {
            "SEL-2_ema_epoch_pick": sel2,
            "SEL-2_ema_pick_equal_mean": ema[sel2],
            "SEL-2_rule": "earliest max on ext4 EMA series (ties<=1e-10 -> earliest); rule fixed before reading numbers",
            "SEL-1_last4": _last_k(ema, 21, 24),
            "SEL-1_last8": _last_k(ema, 17, 24),
            "SEL-1_endpoint24": ema[24],
            "SEL-4_note": "pooled_r2 reported alongside session-mean for every epoch (ext4_epoch_scan.json); never subtract across surfaces",
        },
        "minival_curve_summary": {
            str(e): {
                "raw": metrics[e]["minival_raw"]["equal_session_mean"],
                "ema": metrics[e]["minival_ema"]["equal_session_mean"],
                "train_mse": metrics[e]["train_mse"],
                "lr": metrics[e]["lr"],
            }
            for e in (1, 2, 12, 24)
            if e in metrics
        },
        "picks": PICKS,
        "note_six_rows": NOTE_SIX_ROWS,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    if not passed:
        diagnosis = _fail_diagnosis(
            dest,
            {e: metrics[e]["minival_ema"]["equal_session_mean"] for e in metrics},
            {e: metrics[e]["minival_raw"]["equal_session_mean"] for e in metrics},
        )
        gate["fail_diagnosis"] = diagnosis
        _write_json(dest / "fail_diagnosis.json", diagnosis)
    _seal(dest / ("gate_receipt_PASS.json" if passed else "gate_receipt_FAIL.json"), gate)
    return gate


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="P1a M2 S1 replication (btransform_unified_v1)")
    parser.add_argument("--stage", choices=["preflight", "train", "ext4", "gate", "all"], default="all")
    parser.add_argument("--dest", type=Path, default=None, help="existing run dir (default: new UTC-stamped)")
    args = parser.parse_args()

    if args.stage in ("train", "all") and os.environ.get("BTRANSFORM_P1A_TRAIN") != "1":
        print("REFUSED: training requires BTRANSFORM_P1A_TRAIN=1", file=sys.stderr)
        return 2
    if args.stage in ("train", "all", "ext4"):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            print("REFUSED: CUDA_VISIBLE_DEVICES must be pinned to '1' (GPU1)", file=sys.stderr)
            return 2

    dest = args.dest or plan.RESULT_ROOT / "p1a_m2" / utc_stamp()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[p1a] dest = {dest}", flush=True)

    pre = gpu_preflight(dest / f"preflight_gpu_{args.stage}.json")
    print(f"[p1a] preflight({args.stage}) ok={pre['ok']} foreign={pre['foreign_pids_on_gpu1']}", flush=True)
    if not pre["ok"]:
        print("[p1a] BLOCKED: GPU1 not clean; refusing to start", file=sys.stderr)
        return 2

    if args.stage in ("train", "all"):
        summary = run_train(dest)
        print(
            f"[p1a] train done in {summary['elapsed_s']:.0f}s; "
            f"e24 minival ema={summary['endpoint24_minival_ema']:.4f}",
            flush=True,
        )
    if args.stage in ("ext4", "all"):
        device = torch.device("cuda:0")
        t0 = time.monotonic()
        scan = run_ext4(dest, device)
        e24 = scan["views"]["EMA"]["all_epochs"]["24"]["equal_session_mean"]
        print(f"[p1a] ext4 scan done in {time.monotonic()-t0:.0f}s; EMA e24={e24:.4f}", flush=True)
    if args.stage in ("gate", "all"):
        gate = run_gate(dest)
        print(f"[p1a] gate pass={gate['pass']}", flush=True)
        return 0 if gate["pass"] else 1
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

            target = dest or (_plan.RESULT_ROOT / "p1a_m2" / "error")
            target.mkdir(parents=True, exist_ok=True)
            _receipts.seal_json(
                target / "error_receipt.json",
                {
                    "schema": "btransform_unified_v1_p1a_error",
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "traceback": trace,
                },
            )
        except Exception:
            pass
        sys.exit(3)
