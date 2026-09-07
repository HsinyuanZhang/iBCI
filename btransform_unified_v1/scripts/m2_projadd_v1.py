"""M2 + identity_mode=proj_add interface-validation cell (ADDENDUM-M2-PROJADD).

User directive 2026-09-06 (workorder ADDENDUM-M2-PROJADD): M2 unseals ONE
interface-validation cell — the M2 geometry with ``identity_mode="proj_add"``
(matrix letter (f), M-F250 user proposal): bank E0 [96,50] -> learnable
P = Linear(50 -> 16, bias=False) -> P(E0) ADDED channel-wise onto the local
conv channels; tokens = token_mlp([local16 + P(E0)] | carrier4), token_in 20.
Purpose: if proj_add holds on M2 too, the three tasks share one identity
interface ("structural unification" upgraded to "interface unification").

Design: item-by-item ISOMORPHIC to p1a_v2b
(``results/p1a_v2b_m2/20260906_080831/``, concat seed42 bitwise twin of S1,
EMA endpoint24 ext4 equal_session_mean 0.4501). The SINGLE variable is
``identity_mode``: same seed 42, same manifest digest a95255fa..., same
frozen dual_track cache (read-only), same TRN-1 noise-aligned recipe in full
(S1 dropout domain ``unit_dropout_seed`` + CPU generator per batch, F.conv1d
conv forward, bf16 autocast with pred.float() before MSE, x5 contract, EMA
0.9995, warmup 3165 -> cosine 3e-5, 24 epochs, batch 32), same ext4 scoring
of the EMA view for all 24 epochs + RAW endpoint24, same SEL-2 pick surface
(ext4, earliest max; the legal local face for M2) with endpoint24 / last-4 /
last-8 auxiliary reports.

RNG disclosure (preregistered): the mode switch reorders the
``initialize_decoder`` sorted-name RNG walk (token_mlp.0 narrows 70 -> 20
inputs and the new ``frontend.e0_proj.weight`` [16,50] enters the stream), so
the paired diff vs p1a_v2b's 0.4501 is a NEW DRAW (seed band [0.337, 0.449]
from the P1a diagnosis), not a bitwise-controlled ablation. Paired diffs are
REPORTED, never gated.

Gate (preregistered, ADDENDUM-M2-PROJADD): EMA endpoint24 ext4
equal_session_mean >= 0.42. >= 0.45 additionally marks proj_add as the M2
unified-interface CANDIDATE CHAMPION (noted in the receipt; equal to the
p1a_v2b seed42 reading class).

Discipline: CUDA_VISIBLE_DEVICES pinned to GPU1
(GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86); GPU0 never touched; preflight
refuses when a foreign pid holds >500 MiB on GPU1; 6h GPU budget from train
start (BUDGET_HIT receipt on overrun); receipts sealed via
``btransform_unified_v1.receipts.seal_json`` (0444 + sha256 sidecar);
historical roots untouched; model.py stays byte-stable (the variant lives in
identity_variant.py).

Usage (all stages, one process):
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 \
  BTRANSFORM_M2_PROJADD_TRAIN=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v1/scripts/m2_projadd_v1.py --stage all

Identity: B-transformer unified series, NOT SPINT.
"""

from __future__ import annotations

import argparse
import hashlib
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
from btransform_unified_v1.h1_config import PROJ_ADD_OUT_DIM  # noqa: E402
from btransform_unified_v1.identity_variant import BTransformerUnifiedDecoderIdentity  # noqa: E402
from btransform_unified_v1.model import (  # noqa: E402
    CAUSAL_CHECK_TOLERANCE,
    SET_DIM,
    UNIT_DROPOUT_DOMAIN_META,
    S1_M2_PARAM_COUNT,
    unit_dropout_seed,
    whole_unit_dropout,
)
from btransform_unified_v1.r2 import variance_weighted_r2  # noqa: E402
from btransform_unified_v1.schedule import warmup_cosine_lr  # noqa: E402

from tfpd_exploration.src.m2_b_small_stability_v1 import training as s1_training  # noqa: E402
from tfpd_exploration.src.m2_dual_track_v1 import champion as old_champion  # noqa: E402
from tfpd_exploration.src.m2_dual_track_v1 import data as old_data  # noqa: E402
from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan  # noqa: E402
from tfpd_exploration.src.m2_dual_track_v1 import sampler as old_sampler  # noqa: E402
from tfpd_exploration.src.m2_dual_track_v1 import training as old_training  # noqa: E402

CELL = "M2-PROJADD-V1"
IDENTITY_MODE = "proj_add"
TRAIN_ENV_FLAG = "BTRANSFORM_M2_PROJADD_TRAIN"
SEED = 42
GPU1_UUID = "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"
FOREIGN_MEM_MIB_LIMIT = 500
BUDGET_SECONDS = 6.0 * 3600.0
MANIFEST_24_PATH = (
    WORKSPACE_ROOT
    / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/sampler/shuffled_batch_manifest_24.json"
)
MANIFEST_24_DIGEST = "a95255fa339ea06f1e1cc3ef9f53f3d49d15799bf95e4579f672417fd878d79a"
P1A_V2B_ROOT = plan.RESULT_ROOT / "p1a_v2b_m2" / "20260906_080831"
P1A_V2B_GATE = P1A_V2B_ROOT / "gate_receipt_PASS.json"
P1A_V2B_SCAN = P1A_V2B_ROOT / "ext4_epoch_scan.json"
P1A_V2B_E24_EQUAL_MEAN = 0.45008603537715164  # concat seed42 EMA endpoint24 ext4
SEED_BAND = [0.3373, 0.4495]  # S1 seed43 / seed42 endpoints (P1a diagnosis)
GEOMETRY = {  # P1A mapping, byte-identical to p1a_v2b's P1A_GEOMETRY
    "task": "m2",
    "window": 50,
    "prefix": 0,
    "units": 96,
    "e0_dim": 50,
    "carrier_dim": 4,
    "out_dim": 2,
}
GATE_EQUAL_MEAN_MIN = 0.42
CHAMPION_EQUAL_MEAN_MIN = 0.45

# proj_add parameter arithmetic vs the concat m2 build (recorded, not gated):
# token_mlp.0 Linear(70->256) -> Linear(20->256) loses 256*(70-20) weights;
# the rank-16 projection P adds PROJ_ADD_OUT_DIM*50 = 800.
PARAMS_CONCAT = S1_M2_PARAM_COUNT
PARAMS_PROJ_ADD = S1_M2_PARAM_COUNT - SET_DIM * (70 - 20) + PROJ_ADD_OUT_DIM * 50

# REF §6 picks for every receipt this run produces. Identical to p1a_v2b's
# PICKS except the CAL-3 identity row (concat -> proj_add (f)); TRN/SEL/SPD
# rows are carried verbatim so the isomorphism is auditable row by row.
PICKS = {
    "CAL": [
        "CAL-2 {budget: 33, surface: frozen dual_track cache 20260905_101500 (read-only)}",
        "CAL-3f {identity: proj_add — bank E0 [96,50] -> P=Linear(50->16,bias=False) added onto the local conv channels; tokens=token_mlp([local16+P(E0)]|carrier4), token_in=20 (matrix letter (f), M-F250 user proposal)}",
        "CAL-4 {carrier twice: in E0 (MOVE-T4 side-in) and in token concat}",
        "CAL-5 {unit order: units DataFrame row order; unit_mask all-true (M2)}",
    ],
    "TRN": [
        "TRN-1 {updates_per_epoch: 3165, warmup_updates: 3165, total_updates: 75960, peak_lr: 3e-4, min_lr: 3e-5, AdamW wd 0.01 betas (0.9,0.999) eps 1e-8, clip 1.0, batch 32}",
        "TRN-1 precision {bf16 autocast: forward+loss, pred.float() before MSE (S1 launch.py semantics)}",
        "TRN-3 {whole-unit dropout p=0.10, training mode only, per-batch CPU generator, domain: m2_small_unit_dropout (bit-identical to S1 unit_dropout_seed, import-or-replicate)}",
        "TRN-5 {P: 0, L_in: 50}",
        "TRN-8 {scale: x5 decoder_raw; score pred/5 vs native}",
    ],
    "SEL": [
        "SEL-2 {surface: ext4 (local; the legal M2 local face), view: EMA, rule: earliest max, ties<=1e-10 -> earliest}",
        "SEL-1 {endpoint24 primary + last-4/last-8 auxiliary}",
        "SEL-4 {pooled and session-mean dual report}",
        "SEL-3 {official surface: zero participation}",
    ],
    "SPD": ["none {training-path run; SPD-A1 folding exists but is not used in training or scoring}"],
    "SINGLE_VARIABLE": [
        "IDENTITY-MODE {the ONLY variable vs p1a_v2b 20260906_080831: concat (a) -> proj_add (f); seed42 / manifest a95255fa... / TRN-1 noise-aligned recipe / CAL-2 M33 / ext4 face all carried byte-identically}",
        "RNG-DISCLOSURE {mode switch reorders the initialize_decoder sorted-name RNG walk (token_mlp.0 70->20 inputs + new frontend.e0_proj.weight [16,50]) => paired diff vs p1a_v2b 0.4501 is a NEW DRAW (seed band [0.337,0.449]), not a bitwise-controlled ablation}",
    ],
}

NOTE_SIX_ROWS = {
    "system": (
        "btransform_unified_v1 BTransformerUnifiedDecoderIdentity (m2 geometry, "
        f"P=0, L_in=50, {PARAMS_PROJ_ADD:,} params, identity_mode=proj_add: "
        "E0 [96,50] -> P=Linear(50->16,bias=False) added onto local16; "
        "token_in=20; seed 42; dropout domain m2_small_unit_dropout + conv "
        "F.conv1d path both S1-aligned, carried from p1a_v2b) — NOT SPINT"
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
        "[1,cos,sin]); M33 budget; frozen cache 20260905_101500; consumed "
        "through P(E0) (rank-16 bottleneck onto the local channels)"
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
        "vs p1a_v2b concat seed42 EMA endpoint24 ext4 0.4501: the single "
        "differing row is 'system' — and within it the single variable is "
        "identity_mode (concat 50-d static token concat -> proj_add rank-16 "
        "additive projection onto local16). Data/manifest/schedule/optimizer/"
        "EMA/dropout-rate/dropout-domain/conv-path/scoring identical. The mode "
        "switch reorders the init RNG walk (token_mlp.0 70->20 + new "
        "e0_proj.weight), so the paired diff is a NEW DRAW from the seed band "
        "[0.337,0.449] (P1a diagnosis), reported, not gated"
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
# GPU preflight (one card pinned; foreign pid > 500 MiB => BLOCKED receipt)
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
        "schema": "btransform_unified_v1_m2_projadd_gpu_preflight",
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
# Noise-aligned startup self-checks (carried from p1a_v2b verbatim)
# ---------------------------------------------------------------------------


def verify_dropout_parity(train_dual: dict[str, Any]) -> dict[str, Any]:
    """Prove our loop's mask draw equals S1 ``unit_dropout_mask`` bit for bit."""
    probes: list[dict[str, Any]] = []
    session = sorted(train_dual)[0]
    unit_mask = train_dual[session].unit_mask
    if unit_mask.device.type != "cpu":
        unit_mask = unit_mask.detach().cpu()
    for epoch, batch_id in ((1, 0), (1, 1), (2, 3164), (12, 1500), (24, 3164)):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(unit_dropout_seed(SEED, epoch, batch_id))
        ours = whole_unit_dropout(unit_mask, p=plan.UNIT_DROPOUT, generator=generator)
        s1 = s1_training.unit_dropout_mask(unit_mask, seed=SEED, epoch=epoch, batch_id=batch_id)
        probes.append(
            {
                "session": session,
                "epoch": epoch,
                "batch_id": batch_id,
                "bitwise_equal": bool(torch.equal(ours, s1)),
                "ours_shape": list(ours.shape),
                "s1_shape": list(s1.shape),
                "seed_value": int(unit_dropout_seed(SEED, epoch, batch_id)),
            }
        )
    all_equal = all(p["bitwise_equal"] for p in probes)
    plan.require(all_equal, f"dropout parity FAILED vs S1 unit_dropout_mask: {probes}")
    return {
        "schema": "btransform_unified_v1_m2_projadd_dropout_parity",
        "reference": "tfpd_exploration.src.m2_b_small_stability_v1.training.unit_dropout_mask",
        "domain_meta": UNIT_DROPOUT_DOMAIN_META,
        "probes": probes,
        "all_bitwise_equal": True,
    }


def verify_conv_path() -> dict[str, Any]:
    """Record the conv facts: default forward == S1 pad+conv primitive path."""
    model = BTransformerUnifiedDecoderIdentity(GEOMETRY, seed=SEED, identity_mode=IDENTITY_MODE)
    conv = model.frontend.local_conv
    x = torch.randn(2, 16, 7)
    with torch.no_grad():
        y_prim = conv(x)
        y_ref = conv.forward_local_reference(x)
    delta = float((y_prim - y_ref).abs().max())
    plan.require(delta <= CAUSAL_CHECK_TOLERANCE, f"conv primitive vs local reference delta {delta:.3e}")
    return {
        "default_forward": "F.pad(left=kernel-1) + self.conv (nn.Conv1d/F.conv1d primitive; bf16 under autocast) + SiLU",
        "s1_reference": "tfpd_exploration/src/m2_b_small_stability_v1/decoder.py::SharedCausalConv.forward",
        "local_reference_max_abs_delta": delta,
        "causal_check_tolerance": CAUSAL_CHECK_TOLERANCE,
    }


def verify_proj_add_degeneration() -> dict[str, Any]:
    """CPU control: with P zeroed the m2 proj_add build's forward is bitwise
    independent of the bank E0 (zero-mode semantics at the 20-wide build)."""
    model = BTransformerUnifiedDecoderIdentity(GEOMETRY, seed=SEED, identity_mode=IDENTITY_MODE).eval()
    rng = np.random.default_rng(123)
    e0 = rng.standard_normal((96, 50)).astype(np.float32)
    alt = rng.standard_normal((96, 50)).astype(np.float32)
    x = torch.from_numpy(rng.standard_normal((2, 50, 96)).astype(np.float32))
    with torch.no_grad():
        model.frontend.e0_proj.weight.zero_()
        y_a = model.forward_scores(x, _stub_bank(e0))
        y_b = model.forward_scores(x, _stub_bank(alt))
    bitexact = bool(torch.equal(y_a, y_b))
    plan.require(bitexact, "proj_add P-zero degeneration control FAILED")
    return {
        "control": "P=0 -> forward bitwise independent of bank E0 (alt E0 draw)",
        "bitwise_equal": bitexact,
        "note": "mirror of tests/test_btransform_h1_matrix_v1.py::test_m2_projadd_zero_projection_degenerates_to_no_e0",
    }


def _stub_bank(e0: np.ndarray) -> TaskBank:
    rng = np.random.default_rng(7)
    units = e0.shape[0]
    return TaskBank(
        session_id="stub",
        E0=e0,
        carrier=rng.standard_normal((units, 4)).astype(np.float32),
        unit_mask=np.ones(units, dtype=bool),
        X_store=rng.standard_normal((2, 50, units)).astype(np.float32),
        target_store=np.zeros((2, 2), dtype=np.float32),
        window_ids=np.arange(2, dtype=np.int64),
        calibration_meta={
            "shape": tuple(e0.shape),
            "trial_count": 33,
            "estimator": "synthetic_control",
            "array_sha256": array_sha256(e0),
            "budget": 33,
            "synthetic": True,
        },
    )


# ---------------------------------------------------------------------------
# proj_add (f) brief: P norm / effective rank + P(E0) per-session norms
# ---------------------------------------------------------------------------


def proj_brief(model: BTransformerUnifiedDecoderIdentity, ema: DecoderEMA | None) -> dict[str, Any]:
    def _stats(weight: np.ndarray) -> dict[str, Any]:
        sv = np.linalg.svd(weight, compute_uv=False)
        pr = float((sv ** 2).sum() ** 2 / (sv ** 4).sum()) if (sv > 0).any() else 0.0
        return {
            "frobenius_norm": float(np.linalg.norm(weight)),
            "spectral_norm": float(sv[0]) if len(sv) else 0.0,
            "nuclear_norm": float(sv.sum()),
            "effective_rank_participation_ratio": pr,
            "absmax": float(np.abs(weight).max()),
        }

    weight_raw = model.frontend.e0_proj.weight.detach().cpu().numpy().astype(np.float64)
    out: dict[str, Any] = {"shape": list(weight_raw.shape), "raw": _stats(weight_raw)}
    if ema is not None and ema.n_updates > 0:
        shadow = ema.shadow.get("frontend.e0_proj.weight")
        if shadow is not None:
            out["ema"] = _stats(shadow.cpu().numpy().astype(np.float64))
    return out


def proj_apply_stats(
    model: BTransformerUnifiedDecoderIdentity, banks: dict[str, TaskBank]
) -> dict[str, Any]:
    weight = model.frontend.e0_proj.weight.detach()
    per_session: dict[str, Any] = {}
    for session, bank in banks.items():
        e0 = torch.from_numpy(np.ascontiguousarray(bank.E0))
        with torch.no_grad():
            pe0 = weight(e0.to(weight.device, dtype=weight.dtype))
        per_session[session] = {
            "pe0_frobenius": float(np.linalg.norm(pe0.detach().cpu().numpy())),
            "pe0_absmax": float(pe0.detach().abs().max().item()),
            "e0_frobenius": float(np.linalg.norm(bank.E0)),
        }
    return {"pe0_per_session": per_session}


# ---------------------------------------------------------------------------
# Training (loop carried from p1a_v2b; only the model class/mode differs)
# ---------------------------------------------------------------------------


def _load_surface_banks(surface: str, device: torch.device) -> dict[str, Any]:
    sessions = list(old_plan.EXT4_SESSIONS if surface == "ext4" else old_plan.HELDIN_SESSIONS)
    return {s: old_data.load_session_bank(surface, s, device=device) for s in sessions}


def _task_banks(surface: str) -> dict[str, TaskBank]:
    sessions = list(old_plan.EXT4_SESSIONS if surface == "ext4" else old_plan.HELDIN_SESSIONS)
    return {s: adapters.build_m2_bank(surface, s, budget=33) for s in sessions}


def _score_surface(
    model: BTransformerUnifiedDecoderIdentity,
    dual_banks: dict[str, Any],
    banks: dict[str, TaskBank],
    device: torch.device,
) -> dict[str, Any]:
    """Per-session variance_weighted_r2 (pred/5 vs native) + equal mean."""
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
    model: BTransformerUnifiedDecoderIdentity,
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


def _first_batch_fingerprint(manifest: dict[str, Any]) -> dict[str, Any]:
    spec = manifest["batches"]["1"][0]
    indices = list(spec["indices"])
    return {
        "epoch": 1,
        "batch_id": 0,
        "session": spec["session"],
        "n_indices": len(indices),
        "indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
        "first_indices_head": indices[:8],
        "note": "same manifest as p1a_v2b/S1 (digest a95255fa...); batch-order isomorphism guarantee",
    }


def run_train(dest: Path) -> dict[str, Any]:
    device = torch.device("cuda:0")
    manifest = old_sampler.load_manifest(MANIFEST_24_PATH)
    plan.require(manifest["digest"] == MANIFEST_24_DIGEST, "24-epoch manifest digest mismatch")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    model = BTransformerUnifiedDecoderIdentity(GEOMETRY, seed=SEED, identity_mode=IDENTITY_MODE).to(device)
    meta = model.init_meta
    plan.require(meta["identity_mode"] == IDENTITY_MODE and meta["matrix_letter"] == "f", "wrong identity mode")
    plan.require(model.token_in == 20 and model.base_e0_dim == 50 and model.l_in == 50, "proj_add geometry drift")
    plan.require(
        model.frontend.e0_proj is not None
        and tuple(model.frontend.e0_proj.weight.shape) == (PROJ_ADD_OUT_DIM, 50)
        and model.frontend.e0_proj.bias is None,
        "e0_proj P drift",
    )
    n_params = sum(p.numel() for p in model.parameters())
    plan.require(n_params == PARAMS_PROJ_ADD, f"param count {n_params} != proj_add arithmetic {PARAMS_PROJ_ADD}")

    train_dual = _load_surface_banks("source_train", device)
    mini_dual = _load_surface_banks("source_minival", device)
    mini_banks = _task_banks("source_minival")
    train_bank_cache: dict[str, TaskBank] = {}
    updates = old_training.count_updates(train_dual)
    plan.require(updates == 3165, f"updates/epoch {updates} != 3165")

    dropout_parity = verify_dropout_parity(train_dual)
    conv_path = verify_conv_path()
    degeneration = verify_proj_add_degeneration()
    first_batch = _first_batch_fingerprint(manifest)
    p_brief_init = proj_brief(model, None)

    optimizer = old_training.build_optimizer(
        model.trainable_parameters().items(), lr=plan.LR_PEAK, weight_decay=plan.WEIGHT_DECAY
    )
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    total_updates = plan.recipe_updates("m2", epochs=plan.EPOCHS)["total_updates"]
    warmup_updates = plan.recipe_updates("m2")["warmup_updates"]

    _seal(
        dest / "run_meta.json",
        {
            "schema": "btransform_unified_v1_m2_projadd_run_meta",
            "cell": CELL,
            "route": "ADDENDUM-M2-PROJADD (user directive 2026-09-06): M2 + identity_mode=proj_add interface-validation cell",
            "isomorphic_baseline": {
                "root": str(P1A_V2B_ROOT),
                "gate_receipt": str(P1A_V2B_GATE),
                "concat_seed42_ema_endpoint24_ext4_equal_mean": P1A_V2B_E24_EQUAL_MEAN,
                "single_variable": "identity_mode concat -> proj_add",
            },
            "utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "geometry": GEOMETRY,
            "identity_mode": IDENTITY_MODE,
            "l_in": model.l_in,
            "decoder_params": n_params,
            "params_concat_reference": PARAMS_CONCAT,
            "params_arithmetic": "S1_M2_PARAM_COUNT - 256*(70-20) + 16*50",
            "init_meta": meta,
            "p_brief_at_init": p_brief_init,
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
            "unit_dropout_domain": "m2_small_unit_dropout (S1-bit-identical, carried from p1a_v2b)",
            "unit_dropout_domain_meta": UNIT_DROPOUT_DOMAIN_META,
            "dropout_parity_vs_s1": dropout_parity,
            "conv_path": conv_path,
            "proj_add_degeneration_control": degeneration,
            "first_batch_fingerprint": first_batch,
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
                        "schema": "btransform_unified_v1_m2_projadd_budget_hit",
                        "cell": CELL,
                        "epoch": epoch,
                        "global_step": global_step,
                        "budget_seconds": BUDGET_SECONDS,
                        "elapsed_seconds": time.monotonic() - started,
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "note": "6h GPU budget hit before finishing 24 epochs; no rerun authorized",
                    },
                )
                raise RuntimeError("M2-projadd 6h GPU budget hit")
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
            # S1 unit_dropout_mask construction verbatim — CPU generator seeded
            # from (seed, epoch, batch_id) in the S1 payload domain, mask
            # shared across the batch (carried from p1a_v2b).
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
            "schema": "btransform_unified_v1_m2_projadd_ckpt",
            "cell": CELL,
            "epoch": epoch,
            "global_step": global_step,
            "seed": SEED,
            "identity_mode": IDENTITY_MODE,
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

    p_brief_end = proj_brief(model, ema)
    summary = {
        "schema": "btransform_unified_v1_m2_projadd_train_receipt",
        "cell": CELL,
        "route": "ADDENDUM-M2-PROJADD (identity_mode=proj_add; isomorphic to p1a_v2b)",
        "seed": SEED,
        "identity_mode": IDENTITY_MODE,
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
        "p_brief_raw_and_ema_at_e24": p_brief_end,
        "elapsed_s": time.monotonic() - started,
        "decoder_params": n_params,
        "init_meta": meta,
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
    model: BTransformerUnifiedDecoderIdentity,
    dual_banks: dict[str, Any],
    banks: dict[str, TaskBank],
    device: torch.device,
) -> dict[str, Any]:
    """Full-ext4 scoring: per-session variance_weighted_r2 + equal mean + pooled."""
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


def _apply_view(model: BTransformerUnifiedDecoderIdentity, ckpt: dict[str, Any], view: str) -> None:
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
    model = BTransformerUnifiedDecoderIdentity(GEOMETRY, seed=SEED, identity_mode=IDENTITY_MODE).to(device)
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
        "schema": "btransform_unified_v1_m2_projadd_ext4_scan",
        "cell": CELL,
        "seed": SEED,
        "identity_mode": IDENTITY_MODE,
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
            "schema": "btransform_unified_v1_m2_projadd_ext4_scan_receipt",
            "cell": CELL,
            "identity_mode": IDENTITY_MODE,
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
# Gate + same-seed paired table vs p1a_v2b concat seed42 (0.4501)
# ---------------------------------------------------------------------------


def _p1a_v2b_reference() -> dict[str, Any]:
    """Read the sealed p1a_v2b gate receipt (tamper-checked) + raw scan."""
    gate = receipts.read_sealed(P1A_V2B_GATE)
    return {
        "root": str(P1A_V2B_ROOT),
        "gate_receipt_sha256": receipts.sidecar_path(P1A_V2B_GATE).read_text(encoding="utf-8").split()[0],
        "ema_equal_mean": float(gate["ours"]["ema_endpoint24_equal_session_mean"]),
        "ema_per_session": {
            k: float(v) for k, v in gate["ours"]["ema_endpoint24_per_session"].items()
        },
        "raw_equal_mean": float(gate["ours"]["raw_endpoint24_equal_session_mean"]),
        "sel2_pick_epoch": int(gate["sel"]["SEL-2_ema_epoch_pick"]),
        "sel2_pick_equal_mean": float(gate["sel"]["SEL-2_ema_pick_equal_mean"]),
        "train_mse_ours_series": _p1a_v2b_train_mse_series(),
    }


def _p1a_v2b_train_mse_series() -> dict[int, float]:
    series: dict[int, float] = {}
    path = P1A_V2B_ROOT / "metrics.jsonl"
    if not path.is_file():
        return series
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "epoch" and "train_mse" in row:
                series[int(row["epoch"])] = float(row["train_mse"])
    return series


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


def _paired_table(dest: Path, ours_ema: dict[int, dict[str, Any]], ref: dict[str, Any]) -> dict[str, Any]:
    """Same-seed paired comparison vs p1a_v2b concat seed42 (reported only)."""
    e24 = ours_ema[24]
    equal_diff = float(e24["equal_session_mean"]) - ref["ema_equal_mean"]
    per_session = {
        s: {
            "ours": float(e24["per_session"][s]["r2"]),
            "concat": ref["ema_per_session"][s],
            "diff": float(e24["per_session"][s]["r2"]) - float(ref["ema_per_session"][s]),
        }
        for s in sorted(ref["ema_per_session"])
    }
    # paired train_mse curve (same manifest/seed; mode differs) — informative
    theirs = ref["train_mse_ours_series"]
    ours_mse: dict[int, float] = {}
    with (dest / "metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "epoch" and "train_mse" in row:
                ours_mse[int(row["epoch"])] = float(row["train_mse"])
    mse_pairs = {
        str(e): {"ours_projadd": ours_mse.get(e), "p1a_v2b_concat": theirs.get(e)}
        for e in sorted(set(ours_mse) & set(theirs))
    }
    return {
        "baseline": {
            "cell": "P1A-V2B-M2-S1-ALIGN (concat, matrix letter (a))",
            "seed": SEED,
            "root": ref["root"],
            "gate_receipt_sha256": ref["gate_receipt_sha256"],
            "ema_endpoint24_ext4_equal_session_mean": ref["ema_equal_mean"],
            "ema_endpoint24_ext4_per_session": ref["ema_per_session"],
            "sel2_pick_epoch": ref["sel2_pick_epoch"],
            "sel2_pick_equal_mean": ref["sel2_pick_equal_mean"],
        },
        "ours": {
            "cell": CELL,
            "identity_mode": IDENTITY_MODE,
            "ema_endpoint24_ext4_equal_session_mean": float(e24["equal_session_mean"]),
            "ema_endpoint24_ext4_per_session": {
                s: float(e24["per_session"][s]["r2"]) for s in sorted(e24["per_session"])
            },
        },
        "equal_session_mean_diff_ours_minus_concat": equal_diff,
        "per_session_paired_diff_ours_minus_concat": per_session,
        "train_mse_paired_curve_informative": mse_pairs,
        "rng_disclosure": (
            "the mode switch (concat -> proj_add) reorders the initialize_decoder "
            "sorted-name RNG walk (token_mlp.0 Linear 70->20 inputs + new "
            "frontend.e0_proj.weight [16,50]) and changes the forward graph, so "
            "this paired diff is a NEW DRAW, not a bitwise-controlled ablation; "
            f"seed band from the P1a diagnosis: [{SEED_BAND[0]}, {SEED_BAND[1]}] "
            "(S1 seed43 endpoint24 .. S1 seed42 endpoint24 ext4 EMA equal mean). "
            "Dropout mask stream itself is mode-independent (payload domain "
            "m2_small_unit_dropout over (seed, epoch, batch_id)) — recorded "
            "bitwise-equal vs S1 in run_meta.dropout_parity_vs_s1."
        ),
        "gated": False,
    }


def _p_layer_brief(dest: Path, banks: dict[str, TaskBank]) -> dict[str, Any]:
    """P-layer brief at the SEL-2 pick epoch AND endpoint24 (RAW + EMA views).

    CPU only (SVD/norms on [16,50]); the gate stage never touches any GPU.
    """
    device = torch.device("cpu")
    out: dict[str, Any] = {}
    model = BTransformerUnifiedDecoderIdentity(GEOMETRY, seed=SEED, identity_mode=IDENTITY_MODE).to(device)
    for epoch in sorted({24, _sel2_pick_from_scan(dest)}):
        ckpt = torch.load(dest / f"epoch_{epoch:03d}.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["raw_state_dict"])
        ema = DecoderEMA.__new__(DecoderEMA)
        ema.decay = plan.EMA_DECAY
        ema.n_updates = int(ckpt["ema"]["n_updates"])
        ema.shadow = {k: v.detach().clone() for k, v in ckpt["ema"]["shadow"].items()}
        out[f"epoch_{epoch:03d}"] = {
            "proj_weight": proj_brief(model, ema),
            **proj_apply_stats(model, banks),
        }
    return out


def _sel2_pick_from_scan(dest: Path) -> int:
    scan = json.loads((dest / "ext4_epoch_scan.json").read_text(encoding="utf-8"))
    ema = {int(e): row["equal_session_mean"] for e, row in scan["views"]["EMA"]["all_epochs"].items()}
    return _sel2_pick(ema)


def run_gate(dest: Path) -> dict[str, Any]:
    scan = json.loads((dest / "ext4_epoch_scan.json").read_text(encoding="utf-8"))
    ema = {int(e): row["equal_session_mean"] for e, row in scan["views"]["EMA"]["all_epochs"].items()}
    raw24 = scan["views"]["RAW"]["all_epochs"]["24"]
    ref = _p1a_v2b_reference()
    e24 = scan["views"]["EMA"]["all_epochs"]["24"]
    sel2 = _sel2_pick(ema)
    paired = _paired_table(dest, scan["views"]["EMA"]["all_epochs"], ref)
    equal_mean = float(e24["equal_session_mean"])
    equal_mean_ok = equal_mean >= GATE_EQUAL_MEAN_MIN
    champion = bool(equal_mean >= CHAMPION_EQUAL_MEAN_MIN)
    passed = bool(equal_mean_ok)  # single preregistered gate; paired diffs reported only
    metrics = {}
    with (dest / "metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "epoch":
                metrics[row["epoch"]] = row
    device = torch.device("cpu")  # gate stage: CPU only, no GPU touched
    p_brief = _p_layer_brief(dest, _task_banks("ext4"))
    in_band = bool(SEED_BAND[0] - 0.02 <= equal_mean <= SEED_BAND[1] + 0.05)
    gate = {
        "schema": "btransform_unified_v1_m2_projadd_gate_receipt",
        "cell": CELL,
        "route": "ADDENDUM-M2-PROJADD (preregistered)",
        "gate_definition": {
            "statistic": "EMA endpoint24 ext4 equal_session_mean (session-equal mean of variance_weighted_r2)",
            "threshold_equal_mean": GATE_EQUAL_MEAN_MIN,
            "champion_note_threshold": CHAMPION_EQUAL_MEAN_MIN,
            "per_session_paired_diff_vs_p1a_v2b": "REPORTED ONLY, not gated; new-draw disclosure attached",
            "reference": "p1a_v2b concat seed42 gate_receipt_PASS.json (20260906_080831, sealed)",
        },
        "pass": passed,
        "champion_candidate": champion,
        "champion_note": (
            "endpoint24 EMA ext4 equal_session_mean >= 0.45: proj_add becomes the M2 "
            "unified-interface CANDIDATE CHAMPION (same reading class as the concat "
            "seed42 0.4501); formal adoption is a user ruling, this receipt only records it"
            if champion
            else "below 0.45: no champion claim"
        ),
        "ours": {
            "ema_endpoint24_equal_session_mean": equal_mean,
            "ema_endpoint24_pooled_r2": e24["pooled_r2"],
            "ema_endpoint24_per_session": {
                s: e24["per_session"][s]["r2"] for s in sorted(e24["per_session"])
            },
            "raw_endpoint24_equal_session_mean": raw24["equal_session_mean"],
        },
        "paired_vs_p1a_v2b_concat_seed42": paired,
        "seed_band_placement": {
            "band": SEED_BAND,
            "ours_endpoint24": equal_mean,
            "within_loose_band": in_band,
            "note": "band = S1 seed43/seed42 ext4 EMA endpoint24 equal means (P1a diagnosis); loose margin -0.02/+0.05",
        },
        "checks": {
            "equal_mean_ge_0.42": bool(equal_mean_ok),
            "equal_mean_ge_0.45_champion": champion,
        },
        "sel": {
            "SEL-2_ema_epoch_pick": sel2,
            "SEL-2_ema_pick_equal_mean": ema[sel2],
            "SEL-2_rule": "earliest max on ext4 EMA series (ties<=1e-10 -> earliest); ext4 = the legal M2 local face; rule fixed before reading numbers",
            "SEL-1_last4": _last_k(ema, 21, 24),
            "SEL-1_last8": _last_k(ema, 17, 24),
            "SEL-1_endpoint24": ema[24],
            "SEL-4_note": "pooled_r2 reported alongside session-mean for every epoch (ext4_epoch_scan.json); never subtract across surfaces",
        },
        "p_layer_brief": {
            "note": "P = frontend.e0_proj.weight [16,50]; raw/EMA stats + P(E0) per-session Frobenius norms on ext4 banks, at endpoint24 and the SEL-2 pick epoch",
            **p_brief,
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
    _seal(dest / ("gate_receipt_PASS.json" if passed else "gate_receipt_FAIL.json"), gate)
    return gate


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="M2 + proj_add interface-validation cell (btransform_unified_v1)")
    parser.add_argument("--stage", choices=["preflight", "train", "ext4", "gate", "all"], default="all")
    parser.add_argument("--dest", type=Path, default=None, help="existing run dir (default: new UTC-stamped)")
    args = parser.parse_args()

    if args.stage in ("train", "all") and os.environ.get(TRAIN_ENV_FLAG) != "1":
        print(f"REFUSED: training requires {TRAIN_ENV_FLAG}=1", file=sys.stderr)
        return 2
    if args.stage in ("train", "all", "ext4"):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            print("REFUSED: CUDA_VISIBLE_DEVICES must be pinned to '1' (GPU1)", file=sys.stderr)
            return 2

    dest = args.dest or plan.RESULT_ROOT / "m2_projadd" / utc_stamp()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[m2-projadd] dest = {dest}", flush=True)

    pre = gpu_preflight(dest / f"preflight_gpu_{args.stage}.json")
    print(f"[m2-projadd] preflight({args.stage}) ok={pre['ok']} foreign={pre['foreign_pids_on_gpu1']}", flush=True)
    if not pre["ok"]:
        _seal(
            dest / "BLOCKED.json",
            {
                "schema": "btransform_unified_v1_m2_projadd_blocked",
                "cell": CELL,
                "reason": "GPU1 not clean at preflight (foreign pid >500 MiB or CUDA_VISIBLE_DEVICES!=1)",
                "preflight": pre,
                "utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        print("[m2-projadd] BLOCKED: GPU1 not clean; refusing to start", file=sys.stderr)
        return 2

    if args.stage in ("train", "all"):
        summary = run_train(dest)
        print(
            f"[m2-projadd] train done in {summary['elapsed_s']:.0f}s; "
            f"e24 minival ema={summary['endpoint24_minival_ema']:.4f}",
            flush=True,
        )
    if args.stage in ("ext4", "all"):
        device = torch.device("cuda:0")
        t0 = time.monotonic()
        scan = run_ext4(dest, device)
        e24 = scan["views"]["EMA"]["all_epochs"]["24"]["equal_session_mean"]
        print(f"[m2-projadd] ext4 scan done in {time.monotonic()-t0:.0f}s; EMA e24={e24:.4f}", flush=True)
    if args.stage in ("gate", "all"):
        gate = run_gate(dest)
        print(
            f"[m2-projadd] gate pass={gate['pass']} champion={gate['champion_candidate']} "
            f"e24={gate['ours']['ema_endpoint24_equal_session_mean']:.4f}",
            flush=True,
        )
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

            target = dest or (_plan.RESULT_ROOT / "m2_projadd" / "error")
            target.mkdir(parents=True, exist_ok=True)
            _receipts.seal_json(
                target / "error_receipt.json",
                {
                    "schema": "btransform_unified_v1_m2_projadd_error",
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "traceback": trace,
                },
            )
        except Exception:
            pass
        sys.exit(3)
