"""M1 proj_add series training/eval script (ADDENDUM-UNIFIED-ADD, M1 leg).

User directive 2026-09-06: M1 restarts directly on the unified
``identity_mode="proj_add"`` interface — cells ``M1-PROJADD-P16`` and
``M1-PROJADD-P32`` (``--proj-dim``). Blueprint = ``m2_projadd_v1.py`` (stage
layout, receipts, noise-aligned self-checks) + the DEFERRED-locked M1 data
mechanism (``m1_fullsession_submission_v1.py`` /
``m1_family_loso_outer20120924.py``, both read-only). Both M1 mechanism
scripts never executed on GPU; this series replaces them as the M1 leg of the
unified proj_add push.

Training recipe = the M2-verified noise-aligned TRN-1, transferred verbatim
except the peak LR axis:
  - S1-domain whole-unit dropout ``unit_dropout_seed(42, epoch, batch_id)``
    (payload ``m2_small_unit_dropout|...``; route-B bitwise-verified source),
  - F.conv1d primitive conv forward (bf16 autocast evaluates it exactly like
    S1; ``SharedCausalConv`` default path, probed vs the local reference),
  - AdamW wd 0.01, clip 1, warmup 1 epoch -> peak, cosine -> 0.1x peak,
    24 epochs, batch 32, EMA 0.9995, ``pred.float()`` before MSE,
  - ``--peak-lr`` DEFAULT 1e-4 (workorder ADDENDUM-UNIFIED-ADD M1 leg; H1
    lesson: the TRN-1 3e-4 peak was never validated on M1 — a 3e-4 arm must
    be a separately labeled cell),
  - divisor=1: native target direct (TRN-8; no scale bridge).

Checkpoint rule = endpoint24 EMA (preregistered, stage-2 discipline; the
training set is ALL 4 held-in sessions, so NO surface pick is legal after
retrain — ADDENDUM-SUBMISSION-PROTOCOL). Faces: the 31,252 source-minival and
LOSO ses-20120924 (26,496) scores are DIAGNOSTIC side reports only (both are
polluted by the stage-2 session set); the official score is the only
selector-bearing readout. Acceptance accounting (recorded, not gated here):
proj_add must clear Original + 0.03 = 0.839 pooled on the minival face
(Original 0.809 exposed) or the same +0.03 rule on the LOSO face (Original
0.7983, leakage-disclosed).

Discipline (identical to the sibling cells): GPU1 only
(GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86), CUDA_VISIBLE_DEVICES pinned to
'1', preflight refuses on a foreign pid >500 MiB (BLOCKED receipt), GPU0 is
never touched, 6h budget from train start (BUDGET_HIT receipt), receipts
sealed via ``btransform_unified_v1.receipts.seal_json`` (0444 + sha256),
historical roots untouched.

Usage (one-shot per cell, AFTER the GPU is free — nothing is auto-started):
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 \
  BTRANSFORM_M1_PROJADD_TRAIN=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v1/scripts/m1_projadd_series.py --proj-dim 16 --stage all
  # then the second cell:
  ... --proj-dim 32 --stage all

CPU-only readiness probe (no GPU, no training):
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= \
  /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v1/scripts/m1_projadd_series.py --proj-dim 16 --stage probe --dest <dir>

Identity: B-transformer unified series, NOT SPINT.
"""

from __future__ import annotations

import argparse
import copy
import json
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

from btransform_unified_v1 import m1_projadd as mp  # noqa: E402
from btransform_unified_v1 import plan, receipts  # noqa: E402
from btransform_unified_v1.bank import TaskBank  # noqa: E402
from btransform_unified_v1.ema import DecoderEMA  # noqa: E402
from btransform_unified_v1.identity_variant import BTransformerUnifiedDecoderIdentity  # noqa: E402
from btransform_unified_v1.model import (  # noqa: E402
    CAUSAL_CHECK_TOLERANCE,
    UNIT_DROPOUT_DOMAIN_META,
    unit_dropout_seed,
    whole_unit_dropout,
)
from btransform_unified_v1.r2 import session_mean_report  # noqa: E402
from btransform_unified_v1.schedule import warmup_cosine_lr  # noqa: E402

CELL_PREFIX = "M1-PROJADD"
TRAIN_ENV_FLAG = "BTRANSFORM_M1_PROJADD_TRAIN"
SEED = mp.SERIES_SEED
EPOCHS = mp.SERIES_EPOCHS
GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
GPU1_UUID = "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"
IDENTITY_MODE = "proj_add"
CUDA_PIN = "1"


def cell_id(proj_dim: int) -> str:
    if str(IDENTITY_MODE) == "concat":
        return "M1-CONCAT-W100"
    return f"M1-PROJADD-P{int(proj_dim)}"


def _build_model(proj_dim: int) -> BTransformerUnifiedDecoderIdentity:
    if str(IDENTITY_MODE) == "concat":
        return mp.build_m1_concat_model(seed=SEED)
    return mp.build_m1_projadd_model(proj_dim, seed=SEED)


def _expected_params(proj_dim: int) -> int:
    if str(IDENTITY_MODE) == "concat":
        return int(mp.M1_CONCAT_PARAM_COUNT)
    return int(mp.m1_projadd_param_count(proj_dim))


def _pinned_uuid() -> str:
    return GPU0_UUID if str(CUDA_PIN) == "0" else GPU1_UUID
FOREIGN_MEM_MIB_LIMIT = 500
BUDGET_SECONDS = 6.0 * 3600.0
BATCH_SIZE = plan.BATCH_SIZE
STAGES = ("preflight", "probe", "train", "score", "all")

RUNTIME_CACHE_NPZ = (
    WORKSPACE_ROOT
    / "tfpd_exploration/results/decoder_validation_v2/20260905_190000/m1/m1_optimized_v2_source_runtime_cache.npz"
)

PICKS = {
    "CAL": [
        "CAL-2 {budget: 10 (M10), surface: sealed rSyn3-refit-v1 source-only NPZ (read-only) for "
        "ses-20120926/27/28 + sealed-basis encode for ses-20120924 "
        "(m1_family_loso_outer20120924 mechanism; DEFERRED receipt locked)}",
        "CAL-3f {identity: proj_add — bank E0 [64,100] (B3 Sfix e11 student.id_encoder, "
        "compute_identity(side_features=None), rSyn3 NOT in identity) -> P=Linear(100->R,bias=false) "
        "added onto the local conv channels by group broadcast; tokens=token_mlp([local+P_g]...|carrier4), "
        "token_in=R+4 (matrix letter (f))}",
        "CAL-5 {unit order: units DataFrame row order; unit_mask all-true (64)}",
    ],
    "TRN": [
        "TRN-1 {updates_per_epoch: FILLED_AT_RUNTIME, warmup_updates: FILLED_AT_RUNTIME, "
        "total_updates: FILLED_AT_RUNTIME, peak_lr: FILLED_AT_RUNTIME, min_lr: 0.1x peak, "
        "AdamW wd 0.01 betas (0.9,0.999) eps 1e-8, clip 1.0, batch 32, epochs 24}",
        "TRN-1 precision {bf16 autocast: forward+loss, pred.float() before MSE (S1 launch semantics)}",
        "TRN-3 {whole-unit dropout p=0.10, training mode only, per-batch CPU generator, domain: "
        "m2_small_unit_dropout (S1-aligned unit_dropout_seed; route-B verified)}",
        "TRN-5 {P: 0, L_in: 100}",
        "TRN-8 {scale: divisor=1; train and score native target}",
    ],
    "SEL": [
        "SEL-1 {endpoint24 EMA — preregistered, the ONLY rule for the stage-2 build; no surface pick}",
        "SEL-3 {official surface: zero participation; NO LOSO pick after the all-session retrain "
        "(ADDENDUM-SUBMISSION-PROTOCOL); minival/LOSO scores are DIAGNOSTIC side reports}",
    ],
    "SPD": ["none {training-path run; SPD-A1 folding exists (per-session static terms) but is not used in training or scoring}"],
    "SERIES": [
        "SERIES {cells: M1-PROJADD-P16 / M1-PROJADD-P32 (SERIES_PROJ_DIMS=(16,32)); single axis vs "
        "each other = proj_dim; recipe/data/banks identical}",
        "LR-NOTE {" + mp.PEAK_LR_NOTE + "}",
        "ACCEPT {Original minival 0.809 exposed / official 0.649 references; acceptance law = "
        "proj_add cell (picked) >= Original + 0.03 = 0.839 pooled on the minival face or the same "
        "+0.03 rule on the LOSO face vs Original 0.7983 (leakage-disclosed)}",
    ],
}

NOTE_SIX_ROWS = {
    "system": (
        "btransform_unified_v1 BTransformerUnifiedDecoderIdentity (m1 geometry, P=0, L_in=100=W, "
        "N=64, out=16 EMG, identity_mode=proj_add: E0 [64,100] -> P=Linear(100->R,bias=False) added "
        "onto local16 by group broadcast, token_in=R+4, seed 42) — NOT SPINT"
    ),
    "consumer": (
        "B-transformer unified decoder (8-slot + CausalPE4, this series). Consumers of frozen "
        "SPINT-lineage calibration objects (B3 Sfix e11 id_encoder + rSyn3). Comparisons against "
        "the SPINT family are 'same scoring surface, different system' only (P0-1)"
    ),
    "calibration_object": (
        "E0 [64,100] = B3 Sfix e11 student.id_encoder (sha 7976e0b0...) via "
        "compute_identity(side_features=None) on the 10 chronological calib trials (rSyn3 NOT in "
        "identity, P1-10); carrier [64,4] = rSyn3 (sealed rSyn3-refit-v1 NPZ; ses-20120924 encoded "
        "with the sealed basis + source normalizer); M10 budget; per-session banks frozen (CAL-2)"
    ),
    "scoring_surface": (
        "stage-2 framing: trained on ALL 4 local held-in sessions (20120924/26/27/28); 31,252 "
        "source-minival + LOSO 26,496 recorded as DIAGNOSTIC only (polluted by the stage-2 set); "
        "official score pending (submission packaging is NOT this series' action)"
    ),
    "scale": "divisor=1: train and score native target (M1 never divides by 20)",
    "single_difference_vs_historical_best": (
        "vs the M1 fullsession concat build (DEFERRED before any GPU work; never ran): the single "
        "differing row is the identity interface — concat 100-d token concat -> proj_add rank-R "
        "additive projection onto local16. Data/banks/recipe identical; LR axis moves 3e-4 -> 1e-4 "
        "(H1 lesson, declared); vs SPINT Original the comparison is same-face different-system"
    ),
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _sha_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _seal(path: Path, payload: Any) -> str:
    return receipts.seal_json(path, payload)


def _picks_with_runtime(updates_per_epoch: int, peak_lr: float) -> dict[str, Any]:
    total = updates_per_epoch * EPOCHS
    warmup = updates_per_epoch * plan.WARMUP_EPOCHS
    return {
        **PICKS,
        "TRN": [
            PICKS["TRN"][0]
            .replace("updates_per_epoch: FILLED_AT_RUNTIME", f"updates_per_epoch: {updates_per_epoch}")
            .replace("warmup_updates: FILLED_AT_RUNTIME", f"warmup_updates: {warmup}")
            .replace("total_updates: FILLED_AT_RUNTIME", f"total_updates: {total}")
            .replace("peak_lr: FILLED_AT_RUNTIME", f"peak_lr: {peak_lr:g}")
        ] + PICKS["TRN"][1:],
    }


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
    target_uuid = _pinned_uuid()
    foreign = [
        app
        for app in apps
        if app["gpu_uuid"] == target_uuid and app["pid"] != own_pid and app["used_mib"] > FOREIGN_MEM_MIB_LIMIT
    ]
    gpu1 = next(g for g in gpus if g["uuid"] == GPU1_UUID)
    report = {
        "schema": "btransform_unified_v1_m1_projadd_gpu_preflight",
        "cell": cell_id(16),
        "unix": time.time(),
        "utc": datetime.now(timezone.utc).isoformat(),
        "own_pid": own_pid,
        "cuda_visible_devices": raw,
        "cuda_pin": str(CUDA_PIN),
        "gpus": gpus,
        "compute_apps": apps,
        "gpu1": gpu1,
        "foreign_pids_on_gpu1": foreign if str(CUDA_PIN) == "1" else [],
        "foreign_pids_on_target": foreign,
        "foreign_threshold_mib": FOREIGN_MEM_MIB_LIMIT,
        "torch_cuda_visible_count": torch.cuda.device_count() if raw == str(CUDA_PIN) else None,
        "gpu0_touched": str(CUDA_PIN) == "0",
    }
    ok = raw == str(CUDA_PIN) and not foreign and torch.cuda.device_count() == 1
    report["ok"] = bool(ok)
    _seal(out_path, report)
    return report


# ---------------------------------------------------------------------------
# Noise-aligned startup self-checks (carried from the M2 blueprint verbatim)
# ---------------------------------------------------------------------------


def verify_dropout_parity() -> dict[str, Any]:
    """Prove our loop's mask draw equals S1 ``unit_dropout_mask`` bit for bit."""
    from tfpd_exploration.src.m2_b_small_stability_v1 import training as s1_training

    probes: list[dict[str, Any]] = []
    unit_mask = torch.ones(mp.M1_UNITS, dtype=torch.bool)
    for epoch, batch_id in ((1, 0), (1, 1), (2, 100), (12, 5000), (24, 5999)):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(unit_dropout_seed(SEED, epoch, batch_id))
        ours = whole_unit_dropout(unit_mask, p=plan.UNIT_DROPOUT, generator=generator)
        s1 = s1_training.unit_dropout_mask(unit_mask, seed=SEED, epoch=epoch, batch_id=batch_id)
        probes.append(
            {
                "epoch": epoch,
                "batch_id": batch_id,
                "bitwise_equal": bool(torch.equal(ours, s1)),
                "seed_value": int(unit_dropout_seed(SEED, epoch, batch_id)),
            }
        )
    plan.require(all(p["bitwise_equal"] for p in probes), f"dropout parity FAILED vs S1: {probes}")
    return {
        "schema": "btransform_unified_v1_m1_projadd_dropout_parity",
        "reference": "tfpd_exploration.src.m2_b_small_stability_v1.training.unit_dropout_mask",
        "domain_meta": UNIT_DROPOUT_DOMAIN_META,
        "probes": probes,
        "all_bitwise_equal": True,
    }


def verify_conv_path(proj_dim: int) -> dict[str, Any]:
    """Record the conv facts: default forward == S1 pad+conv primitive path."""
    model = mp.build_m1_projadd_model(proj_dim, seed=SEED)
    conv = model.frontend.local_conv
    x = torch.randn(2, 16, 7)
    with torch.no_grad():
        y_prim = conv(x)
        y_ref = conv.forward_local_reference(x)
    delta = float((y_prim - y_ref).abs().max())
    plan.require(delta <= CAUSAL_CHECK_TOLERANCE, f"conv primitive vs local reference delta {delta:.3e}")
    return {
        "default_forward": "F.pad(left=kernel-1) + self.conv (F.conv1d primitive; bf16 under autocast) + SiLU",
        "local_reference_max_abs_delta": delta,
        "causal_check_tolerance": CAUSAL_CHECK_TOLERANCE,
    }


def verify_proj_add_degeneration(proj_dim: int) -> dict[str, Any]:
    """P zeroed -> the M1 proj_add build's forward is bitwise independent of E0."""
    model = mp.build_m1_projadd_model(proj_dim, seed=SEED).eval()
    rng = np.random.default_rng(123)
    e0 = rng.standard_normal((mp.M1_UNITS, mp.M1_E0_DIM)).astype(np.float32)
    alt = rng.standard_normal((mp.M1_UNITS, mp.M1_E0_DIM)).astype(np.float32)
    x = torch.from_numpy(rng.standard_normal((2, mp.M1_WINDOW, mp.M1_UNITS)).astype(np.float32))
    bank_a = _stub_bank(proj_dim, e0)
    bank_b = _stub_bank(proj_dim, alt)
    with torch.no_grad():
        model.frontend.e0_proj.weight.zero_()
        y_a = model.forward_scores(x, bank_a)
        y_b = model.forward_scores(x, bank_b)
    bitexact = bool(torch.equal(y_a, y_b))
    plan.require(bitexact, "proj_add P-zero degeneration control FAILED")
    return {
        "control": "P=0 -> forward bitwise independent of bank E0 (alt E0 draw)",
        "bitwise_equal": bitexact,
    }


def verify_fold_parity(proj_dim: int) -> dict[str, Any]:
    """SPD-A1 folded static path parity (P3 caliber, FP32 max|delta| <= 1e-6)."""
    model = mp.build_m1_projadd_model(proj_dim, seed=SEED).eval()
    rng = np.random.default_rng(11)
    bank = _stub_bank(proj_dim, rng.standard_normal((mp.M1_UNITS, mp.M1_E0_DIM)).astype(np.float32))
    x = torch.from_numpy(rng.standard_normal((2, mp.M1_WINDOW, mp.M1_UNITS)).astype(np.float32))
    static = model.bank_static_term(bank)
    delta = mp.assert_fold_parity(model, bank, x, static)
    return {
        "folded_vs_unfolded_max_abs_delta": delta,
        "tolerance": 1e-6,
        "note": "per-session static term = sum_g P_g @ W_g^T + carrier @ W_carrier^T + b (P(E0) folded)",
    }


def _stub_bank(proj_dim: int, e0: np.ndarray) -> TaskBank:
    from btransform_unified_v1.bank import array_sha256

    rng = np.random.default_rng(7)
    return TaskBank(
        session_id="stub",
        E0=e0,
        carrier=rng.standard_normal((mp.M1_UNITS, 4)).astype(np.float32),
        unit_mask=np.ones(mp.M1_UNITS, dtype=bool),
        X_store=rng.standard_normal((2, mp.M1_WINDOW, mp.M1_UNITS)).astype(np.float32),
        target_store=np.zeros((2, mp.M1_OUT_DIM), dtype=np.float32),
        window_ids=np.arange(2, dtype=np.int64),
        calibration_meta={
            "shape": tuple(e0.shape),
            "trial_count": mp.M10_BUDGET,
            "estimator": "synthetic_control",
            "array_sha256": array_sha256(e0),
            "budget": mp.M10_BUDGET,
            "synthetic": True,
        },
    )


# ---------------------------------------------------------------------------
# Banks for the real run (probe/train/score)
# ---------------------------------------------------------------------------


def build_real_banks(calib_by_session: dict[str, np.ndarray]) -> tuple[dict[str, TaskBank], dict[str, Any]]:
    """All-4-session frozen banks: sealed NPZ carriers + sealed-basis outer."""
    carriers = mp.load_source_carriers()
    outer_carrier, outer_meta = mp.encode_outer_carrier()
    carriers = {**carriers, mp.M1_OUTER_SESSION: outer_carrier}
    banks, report = mp.build_banks(calib_by_session, carriers)
    report["outer_carrier_meta"] = outer_meta
    return banks, report


def crosscheck_banks_vs_runtime_cache(banks: dict[str, TaskBank]) -> dict[str, Any]:
    """Source-session banks must reproduce the sealed runtime cache bitwise."""
    cache = np.load(RUNTIME_CACHE_NPZ, allow_pickle=False)
    checks: dict[str, Any] = {}
    for name in mp.M1_SOURCE_SESSIONS:
        e0_ref = np.asarray(cache[f"bank_e0/{name}"], dtype=np.float32)
        t_ref = np.asarray(cache[f"bank_t/{name}"], dtype=np.float32)
        checks[name] = {
            "e0_bitwise": bool(np.array_equal(banks[name].E0, e0_ref)),
            "e0_max_abs_diff": float(np.max(np.abs(banks[name].E0 - e0_ref))),
            "carrier_bitwise": bool(np.array_equal(banks[name].carrier, t_ref)),
        }
        plan.require(checks[name]["e0_bitwise"], f"{name}: bank E0 drift vs sealed runtime cache")
        plan.require(checks[name]["carrier_bitwise"], f"{name}: bank carrier drift vs sealed runtime cache")
    return checks


# ---------------------------------------------------------------------------
# Stage: probe (CPU-only; seals the session inventory the train stage asserts)
# ---------------------------------------------------------------------------


def run_probe(dest: Path, proj_dim: int, peak_lr: float) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    t0 = time.monotonic()
    dm = mp.build_loso_datamodule()
    dataset, sampler = mp.assemble_training_universe(dm)
    calib = mp.calib_trials_from_dataset(dataset)
    banks, bank_report = build_real_banks(calib)
    cache_checks = crosscheck_banks_vs_runtime_cache(banks)

    model = _build_model(proj_dim)
    n_params = int(sum(p.numel() for p in model.parameters()))
    dropout_parity = verify_dropout_parity()
    if str(IDENTITY_MODE) == "concat":
        conv_path = {"skipped": True, "reason": "concat cell; proj_add-only conv probe skipped"}
        degeneration = {"skipped": True, "reason": "concat cell; proj_add degeneration N/A"}
        fold_parity = {"skipped": True, "reason": "concat cell; proj_add fold N/A"}
    else:
        conv_path = verify_conv_path(proj_dim)
        degeneration = verify_proj_add_degeneration(proj_dim)
        fold_parity = verify_fold_parity(proj_dim)
    divisor = mp.assert_divisor_identity(model)

    # causality self-certification on a synthetic window (real bank)
    x = torch.from_numpy(
        np.ascontiguousarray(dataset[0][0], dtype=np.float32).reshape(1, mp.M1_WINDOW, mp.M1_UNITS)
    )
    causal = model.causal_check(x, banks[dataset.window_indices[0][0]])

    windows = {name: int(sum(1 for n, _ in dataset.window_indices if n == name)) for name in mp.M1_SESSIONS}
    updates_per_epoch = int(len(sampler))
    audit = {}
    for name in mp.M1_SESSIONS:
        a = dict(dataset.query_window_audit[name])
        for key in ("ordered_window_start_sha256", "ordered_target_covariate_evalmask_sha256", "ordered_query_identity_sha256"):
            a.pop(key, None)
        audit[name] = a
    inventory = {
        "schema": "btransform_unified_v1_m1_projadd_session_inventory",
        "cell": cell_id(proj_dim),
        "utc": datetime.now(timezone.utc).isoformat(),
        "protocol": (
            "stage-2 framing (ADDENDUM-SUBMISSION-PROTOCOL): training = ALL 4 local held-in "
            "sessions incl. ses-20120924 (legal stage-2 member); minival/LOSO faces DIAGNOSTIC only"
        ),
        "proj_dim": proj_dim,
        "identity_mode": str(IDENTITY_MODE),
        "peak_lr": peak_lr,
        "lr_note": mp.PEAK_LR_NOTE,
        "geometry": mp.m1_projadd_geometry(proj_dim),
        "l_in": model.l_in,
        "decoder_params": n_params,
        "expected_decoder_params": _expected_params(proj_dim),
        "token_in": model.token_in,
        "init_meta": model.init_meta,
        "sessions": list(mp.M1_SESSIONS),
        "window_audit": audit,
        "windows_per_session": windows,
        "total_windows": int(len(dataset.window_indices)),
        "updates_per_epoch": updates_per_epoch,
        "warmup_updates": updates_per_epoch * plan.WARMUP_EPOCHS,
        "total_updates": updates_per_epoch * EPOCHS,
        "train_law": (
            "FalconDataset split='train': query_start_trial=0, query_end_trial=None, eval_mask at "
            "last bin, pre_history=99 zero pad"
        ),
        "sampler": "SessionBatchSampler(batch=32, shuffle=True, seed=42, balance=False, reshuffle_each_epoch=False)",
        "sampler_batch_sha256": mp.sampler_digest(sampler),
        "banks": bank_report,
        "bank_determinism": (
            "identity E0 computed under pinned torch threads (b3_identity/_PinnedThreads): bank bytes "
            "are independent of the ambient OMP/thread environment and byte-identical to the sealed "
            "m1_optimized_v2 runtime cache (ses-20120926/27/28 crosscheck below)"
        ),
        "runtime_cache_crosscheck": cache_checks,
        "calib_trials_per_session": mp.M10_BUDGET,
        "self_checks": {
            "dropout_parity_vs_s1": dropout_parity,
            "conv_path": conv_path,
            "proj_add_degeneration": degeneration,
            "fold_parity": fold_parity,
            "divisor_identity": divisor,
            "causal_check": causal,
        },
        "probe_seconds": time.monotonic() - t0,
        "references": {
            "deferred_receipt": str(mp.DEFERRED_RECEIPT_PATH),
            "family_flat": str(mp.FAMILY_FLAT_PATH),
            "loso_script_readonly": str(mp.LOSO_SCRIPT_REFERENCE),
            "fullsession_script_readonly": str(mp.FULLSESSION_SCRIPT_REFERENCE),
            "runtime_cache": str(RUNTIME_CACHE_NPZ),
        },
        "picks": _picks_with_runtime(updates_per_epoch, peak_lr),
    }
    _seal(dest / "session_inventory.json", inventory)
    print(
        f"[m1-projadd] probe: windows={inventory['total_windows']} ({windows}) "
        f"updates/epoch={updates_per_epoch} params={n_params}",
        flush=True,
    )
    return inventory


# ---------------------------------------------------------------------------
# Stage: train (GPU1; noise-aligned TRN-1 at --peak-lr, endpoint24 EMA rule)
# ---------------------------------------------------------------------------


def _collate(items: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]):
    """Light collate (fullsession mechanism): drop the per-item calib tensor."""
    neural = torch.from_numpy(
        np.ascontiguousarray(np.stack([np.asarray(it[0], dtype=np.float32) for it in items]), dtype=np.float32)
    )
    target = torch.from_numpy(
        np.ascontiguousarray(np.stack([np.asarray(it[1], dtype=np.float32)[-1] for it in items]), dtype=np.float32)
    )
    sessions = [str(it[3]) for it in items]
    return neural, target, sessions


def run_train(dest: Path, proj_dim: int, peak_lr: float) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)

    inventory = json.loads((dest / "session_inventory.json").read_text(encoding="utf-8"))
    dm = mp.build_loso_datamodule()
    dataset, sampler = mp.assemble_training_universe(dm)
    if mp.sampler_digest(sampler) != inventory["sampler_batch_sha256"]:
        raise RuntimeError("sampler digest drift vs sealed probe inventory")
    if len(sampler) != inventory["updates_per_epoch"]:
        raise RuntimeError("updates_per_epoch drift vs sealed probe inventory")
    banks, _report = build_real_banks(mp.calib_trials_from_dataset(dataset))
    for name in mp.M1_SESSIONS:
        sealed = inventory["banks"][name]
        if banks[name].calibration_meta["array_sha256"] != sealed["e0_sha256"]:
            raise RuntimeError(f"bank E0 digest drift vs sealed probe inventory: {name}")
        if banks[name].calibration_meta["carrier_sha256"] != sealed["carrier_sha256"]:
            raise RuntimeError(f"bank carrier digest drift vs sealed probe inventory: {name}")

    model = _build_model(proj_dim).to(device)
    if int(sum(p.numel() for p in model.parameters())) != inventory["decoder_params"]:
        raise RuntimeError("param count drift vs probe")

    from tfpd_exploration.src.m2_dual_track_v1 import training as dual_training

    optimizer = dual_training.build_optimizer(
        model.trainable_parameters().items(), lr=peak_lr, weight_decay=plan.WEIGHT_DECAY
    )
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    updates_per_epoch = inventory["updates_per_epoch"]
    total_updates = updates_per_epoch * EPOCHS
    warmup_updates = updates_per_epoch * plan.WARMUP_EPOCHS
    picks = _picks_with_runtime(updates_per_epoch, peak_lr)

    _seal(
        dest / "run_meta.json",
        {
            "schema": "btransform_unified_v1_m1_projadd_run_meta",
            "cell": cell_id(proj_dim),
            "route": "ADDENDUM-UNIFIED-ADD M1 leg (proj_add series; DEFERRED receipt mechanisms)",
            "utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "proj_dim": proj_dim,
            "peak_lr": peak_lr,
            "lr_note": mp.PEAK_LR_NOTE,
            "geometry": inventory["geometry"],
            "l_in": model.l_in,
            "token_in": model.token_in,
            "decoder_params": inventory["decoder_params"],
            "init_meta": model.init_meta,
            "unit_dropout_domain_meta": UNIT_DROPOUT_DOMAIN_META,
            "sessions": list(mp.M1_SESSIONS),
            "windows_per_session": inventory["windows_per_session"],
            "total_windows": inventory["total_windows"],
            "updates_per_epoch": updates_per_epoch,
            "warmup_updates": warmup_updates,
            "total_updates": total_updates,
            "ema_horizon_updates": plan.EMA_HORIZON_UPDATES,
            "lr_min": peak_lr * plan.LR_MIN_FACTOR,
            "weight_decay": plan.WEIGHT_DECAY,
            "grad_clip": plan.GRAD_CLIP,
            "ema_decay": plan.EMA_DECAY,
            "unit_dropout_p": plan.UNIT_DROPOUT,
            "precision": "bf16 autocast (forward+loss; pred.float() before MSE)",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "checkpoint_rule": "endpoint24 EMA (preregistered, only rule; no surface pick)",
            "sampler_batch_sha256": inventory["sampler_batch_sha256"],
            "banks": inventory["banks"],
            "gpu_uuid": _pinned_uuid(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "comparison_targets": {
                "original_minival_pooled_exposed": mp.ORIGINAL_MINIVAL_POOLED,
                "official_original": mp.OFFICIAL_ORIGINAL,
                "original_loso_outer_pooled_leakage_disclosed": mp.ORIGINAL_LOSO_OUTER_POOLED,
                "accept_threshold_minival_pooled": mp.ACCEPT_THRESHOLD_MINIVAL,
                "accept_margin": mp.ACCEPT_MARGIN,
            },
            "picks": picks,
        },
    )

    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=_collate, num_workers=0)
    metrics_path = dest / "metrics.jsonl"
    heartbeat = dest / "heartbeat.json"
    started = time.monotonic()
    deadline = started + BUDGET_SECONDS
    global_step = 0
    train_mse_series: dict[int, float] = {}
    lr_series: dict[int, float] = {}
    projection_reported = False

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_t0 = time.monotonic()
        running = 0.0
        n_batches = 0
        for batch_id, (x, y, sessions) in enumerate(loader):
            if time.monotonic() >= deadline:
                _seal(
                    dest / "budget_hit.json",
                    {
                        "schema": "btransform_unified_v1_m1_projadd_budget_hit",
                        "cell": cell_id(proj_dim),
                        "epoch": epoch,
                        "global_step": global_step,
                        "budget_seconds": BUDGET_SECONDS,
                        "elapsed_seconds": time.monotonic() - started,
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "note": "6h GPU budget hit before finishing 24 epochs; no rerun authorized",
                    },
                )
                raise RuntimeError("M1 proj_add 6h GPU budget hit")
            session = sessions[0]
            if any(s != session for s in sessions):
                raise RuntimeError("mixed-session batch from SessionBatchSampler")
            bank = banks[session]
            x = x.to(device, non_blocking=False)
            y = y.to(device, non_blocking=False)
            global_step += 1
            lr = warmup_cosine_lr(
                global_step,
                total_steps=total_updates,
                warmup_steps=warmup_updates,
                peak=peak_lr,
                min_factor=plan.LR_MIN_FACTOR,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            # S1 unit_dropout_mask construction verbatim (CPU generator seeded
            # from (seed, epoch, batch_id) in the S1 payload domain).
            generator = torch.Generator(device="cpu")
            generator.manual_seed(unit_dropout_seed(SEED, epoch, batch_id))
            keep = whole_unit_dropout(bank.unit_mask, p=plan.UNIT_DROPOUT, generator=generator)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(x, bank, dropout_keep=keep)
                loss = nn.functional.mse_loss(pred.float(), y)  # divisor=1: native target
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
                        "cell": cell_id(proj_dim),
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
                        "cell": cell_id(proj_dim),
                        "epoch": epoch,
                        "global_step": global_step,
                        "lr": float(lr),
                        "loss": float(loss.detach().cpu()),
                        "ema_updates": ema.n_updates,
                        "unix": time.time(),
                        "gpu_uuid": _pinned_uuid(),
                    },
                )
            if not projection_reported and global_step == 50:
                per_step = (time.monotonic() - epoch_t0) / 50.0
                projected = per_step * total_updates
                projection = {
                    "schema": "btransform_unified_v1_m1_projadd_budget_projection",
                    "cell": cell_id(proj_dim),
                    "measured_steps": 50,
                    "seconds_per_update": per_step,
                    "projected_total_seconds": projected,
                    "budget_seconds": BUDGET_SECONDS,
                    "within_budget": bool(projected <= 0.92 * BUDGET_SECONDS),
                    "utc": datetime.now(timezone.utc).isoformat(),
                }
                _seal(dest / "budget_projection.json", projection)
                if not projection["within_budget"]:
                    raise RuntimeError(
                        f"projected train time {projected:.0f}s exceeds 92% of the 6h budget; refusing to continue"
                    )
                projection_reported = True
        train_mse_series[epoch] = running / max(n_batches, 1)
        lr_series[epoch] = float(lr)
        extras = {
            "train_mse": train_mse_series[epoch],
            "seconds": time.monotonic() - epoch_t0,
            "global_step": global_step,
            "lr": float(lr),
            "ema_updates": ema.n_updates,
            "updates_this_epoch": n_batches,
        }
        ckpt = {
            "schema": "btransform_unified_v1_m1_projadd_ckpt",
            "cell": cell_id(proj_dim),
            "epoch": epoch,
            "global_step": global_step,
            "seed": SEED,
            "proj_dim": proj_dim,
            "peak_lr": peak_lr,
            "sampler_batch_sha256": inventory["sampler_batch_sha256"],
            "raw_state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr": float(lr),
            "ema_decay": ema.decay,
        }
        torch.save(ckpt, dest / f"epoch_{epoch:03d}.pt")
        _append_jsonl(metrics_path, {"event": "epoch", "epoch": epoch, "cell": cell_id(proj_dim), **extras, "unix": time.time()})
        _write_json(heartbeat, {"event": "epoch", "cell": cell_id(proj_dim), "epoch": epoch, **extras, "unix": time.time(), "gpu_uuid": _pinned_uuid()})

    summary = {
        "schema": "btransform_unified_v1_m1_projadd_train_receipt",
        "cell": cell_id(proj_dim),
        "proj_dim": proj_dim,
        "peak_lr": peak_lr,
        "seed": SEED,
        "gpu_uuid": _pinned_uuid(),
        "sessions": list(mp.M1_SESSIONS),
        "windows_per_session": inventory["windows_per_session"],
        "total_windows": inventory["total_windows"],
        "updates_per_epoch": updates_per_epoch,
        "global_updates": global_step,
        "ema_updates": ema.n_updates,
        "epochs_completed": list(range(1, EPOCHS + 1)),
        "checkpoint_rule": "endpoint24 EMA (preregistered, only rule)",
        "train_mse": train_mse_series,
        "train_mse_summary": {e: train_mse_series.get(e) for e in (1, 12, 24)},
        "lr_at_epoch_end": lr_series,
        "elapsed_s": time.monotonic() - started,
        "sampler_batch_sha256": inventory["sampler_batch_sha256"],
        "picks": picks,
        "note_six_rows": NOTE_SIX_ROWS,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    _seal(dest / "train_receipt.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Stage: score (GPU1; DIAGNOSTIC faces + acceptance accounting)
# ---------------------------------------------------------------------------


def _apply_endpoint24_ema(model: BTransformerUnifiedDecoderIdentity, dest: Path) -> dict[str, Any]:
    ckpt = torch.load(dest / "epoch_024.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["raw_state_dict"], strict=True)
    shadow = ckpt["ema"]["shadow"]
    named = model.trainable_parameters()
    if set(named) != set(shadow):
        raise RuntimeError("EMA/RAW key mismatch")
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(shadow[name].to(device=param.device, dtype=param.dtype))
    model.eval()
    return {"epoch": 24, "ema_updates": int(ckpt["ema"]["n_updates"]), "global_step": int(ckpt["global_step"])}


def _score_source_minival(model, banks, device) -> dict[str, Any]:
    """31,252-window source-minival face (DIAGNOSTIC; chron-80 dev tail)."""
    from torch.utils.data import DataLoader

    from tfpd_exploration.src.m1_optimized_v2 import bank as src_bank, source_dev

    loaded = src_bank.load()
    dm_src = source_dev.build_source_only_datamodule(loaded)
    _train_rows, dev_rows, rows_report = source_dev._split(dm_src)
    dev_inner = copy.copy(dm_src.train_dataset.base)
    dev_inner.window_indices = list(dev_rows)
    dev_ds = source_dev.SourceCarrierDataset(dev_inner, loaded)

    groups = {name: [] for name in mp.M1_SOURCE_SESSIONS}
    for index, (name, _start) in enumerate(dev_inner.window_indices):
        groups[name].append(index)
    batches = []
    for name in mp.M1_SOURCE_SESSIONS:
        for left in range(0, len(groups[name]), 32):
            batches.append(groups[name][left : left + 32])

    preds, targets, sids = [], [], []
    model.eval()
    with torch.inference_mode():
        for ids in batches:
            neural, behavior, _calib, sessions, _carrier = next(
                iter(DataLoader(dev_ds, batch_sampler=[ids], num_workers=0))
            )
            name = sessions[0].decode() if isinstance(sessions[0], bytes) else str(sessions[0])
            if any((s.decode() if isinstance(s, bytes) else str(s)) != name for s in sessions):
                raise RuntimeError("minival batch is not session-pure")
            out = model(neural.float().to(device), banks[name])
            preds.append(out.detach().cpu().numpy().astype(np.float32))
            targets.append(behavior[:, -1, :].numpy().astype(np.float32))
            sids.extend([name] * len(ids))
    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    report = session_mean_report(target, pred, np.repeat(np.asarray(sids), target.shape[1]))
    report["role"] = "DIAGNOSTIC ONLY (polluted by the stage-2 all-session training set)"
    report["split_rows"] = rows_report
    return report


def _score_loso_outer(model, banks, device) -> dict[str, Any]:
    """LOSO ses-20120924 face, 26,496 windows (DIAGNOSTIC; family_flat precedent)."""
    from torch.utils.data import DataLoader

    dm = mp.build_loso_datamodule()
    val = dm.val_heldin_dataset
    if val is None:
        raise RuntimeError("LOSO val_heldin_dataset missing")
    calib10 = np.asarray(val.calib_trialized_neural_features[mp.M1_OUTER_SESSION][:10])
    if calib10.shape[0] < mp.M10_BUDGET:
        raise RuntimeError("outer session has fewer than M10 calib trials")
    # the bank for the outer session must equal the probe-sealed bank
    outer_bank = banks[mp.M1_OUTER_SESSION]
    preds, targets = [], []
    model.eval()
    sampler = dm.val_heldin_batch_sampler
    with torch.inference_mode():
        for ids in sampler:
            neural, target, _calib, sessions = next(iter(DataLoader(val, batch_sampler=[ids])))[:4]
            names = [s.decode() if isinstance(s, bytes) else str(s) for s in sessions]
            if any(n != mp.M1_OUTER_SESSION for n in names):
                raise RuntimeError("non-outer batch in LOSO val")
            out = model(neural.float().to(device), outer_bank)
            preds.append(out.cpu().numpy().astype(np.float32))
            targets.append(target[:, -1, :].numpy().astype(np.float32))
    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    if len(target) != mp.LOSO_OUTER_WINDOWS:
        raise RuntimeError(f"LOSO cardinality drift: {len(target)} != {mp.LOSO_OUTER_WINDOWS}")
    report = session_mean_report(
        target, pred, np.repeat(np.asarray([mp.M1_OUTER_SESSION] * len(target)), target.shape[1])
    )
    report["role"] = "DIAGNOSTIC ONLY (ses-20120924 is IN the stage-2 training set; a LOSO number on a retrained model is meaningless per ADDENDUM-SUBMISSION-PROTOCOL)"
    report["references"] = {
        "original_loso_outer_pooled_leakage_disclosed": mp.ORIGINAL_LOSO_OUTER_POOLED,
        "family_flat_pooled_r2": 0.6579612934315439,
        "family_flat_path": str(mp.FAMILY_FLAT_PATH),
    }
    return report


def run_score(dest: Path, proj_dim: int) -> dict[str, Any]:
    device = torch.device("cuda:0")
    torch.set_num_threads(4)

    inventory = json.loads((dest / "session_inventory.json").read_text(encoding="utf-8"))
    banks, _report = _score_banks(dest)
    for name in mp.M1_SESSIONS:
        sealed = inventory["banks"][name]
        if banks[name].calibration_meta["array_sha256"] != sealed["e0_sha256"]:
            raise RuntimeError(f"bank E0 digest drift vs sealed probe inventory: {name}")
        if banks[name].calibration_meta["carrier_sha256"] != sealed["carrier_sha256"]:
            raise RuntimeError(f"bank carrier digest drift vs sealed probe inventory: {name}")

    model = _build_model(proj_dim).to(device)
    view = _apply_endpoint24_ema(model, dest)

    minival = _score_source_minival(model, banks, device)
    loso = _score_loso_outer(model, banks, device)

    minival_pooled = float(minival["pooled_r2"])
    loso_pooled = float(loso["pooled_r2"])
    score = {
        "schema": "btransform_unified_v1_m1_projadd_score_receipt",
        "cell": cell_id(proj_dim),
        "view": view,
        "proj_dim": proj_dim,
        "faces": {
            "source_minival_31252": minival,
            "loso_outer_20120924_26496": loso,
        },
        "acceptance_accounting": {
            "rule": (
                "proj_add picked cell must SIGNIFICANTLY beat SPINT Original "
                f"(operationalized >= Original + {mp.ACCEPT_MARGIN}); both faces here are "
                "DIAGNOSTIC (stage-2 session set), official score is the only selector-bearing number"
            ),
            "original_minival_pooled_exposed": mp.ORIGINAL_MINIVAL_POOLED,
            "accept_threshold_minival_pooled": mp.ACCEPT_THRESHOLD_MINIVAL,
            "ours_minival_pooled": minival_pooled,
            "ours_minival_above_threshold": bool(minival_pooled >= mp.ACCEPT_THRESHOLD_MINIVAL),
            "original_loso_outer_pooled_leakage_disclosed": mp.ORIGINAL_LOSO_OUTER_POOLED,
            "ours_loso_pooled": loso_pooled,
            "ours_loso_margin_vs_original": float(loso_pooled - mp.ORIGINAL_LOSO_OUTER_POOLED),
            "gated": False,
            "note": mp.ORIGINAL_LOSO_LEAKAGE_NOTE,
        },
        "picks": _picks_with_runtime(inventory["updates_per_epoch"], inventory["peak_lr"]),
        "note_six_rows": NOTE_SIX_ROWS,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    _seal(dest / "score_receipt.json", score)
    print(
        f"[m1-projadd] score: minival pooled={minival_pooled:.4f} (threshold {mp.ACCEPT_THRESHOLD_MINIVAL:.4f}) "
        f"loso pooled={loso_pooled:.4f}",
        flush=True,
    )
    return score


def _score_banks(dest: Path) -> tuple[dict[str, TaskBank], dict[str, Any]]:
    """Banks for the score stage without rebuilding the 4-session universe.

    Source-session calib trials come from the source-only datamodule
    (``materialize_source_banks`` precedent); the outer session's from the
    LOSO val dataset (family_flat precedent). Carriers: sealed NPZ +
    sealed-basis encode, identical to the probe path.
    """
    from tfpd_exploration.src.m1_optimized_v2 import bank as src_bank, data as src_data

    loaded = src_bank.load()
    dm_src = src_data.build_source_only_datamodule(loaded)
    calib = {
        name: np.asarray(dm_src.train_dataset.base.calib_trialized_neural_features[name][: mp.M10_BUDGET])
        for name in mp.M1_SOURCE_SESSIONS
    }
    dm = mp.build_loso_datamodule()
    calib[mp.M1_OUTER_SESSION] = np.asarray(
        dm.val_heldin_dataset.calib_trialized_neural_features[mp.M1_OUTER_SESSION][: mp.M10_BUDGET]
    )
    return build_real_banks(calib)


# ---------------------------------------------------------------------------


def main() -> int:
    global IDENTITY_MODE, CUDA_PIN
    parser = argparse.ArgumentParser(description="M1 proj_add / concat series (btransform_unified_v1)")
    parser.add_argument("--proj-dim", type=int, choices=list(mp.SERIES_PROJ_DIMS), default=16)
    parser.add_argument("--identity-mode", choices=["proj_add", "concat"], default="proj_add")
    parser.add_argument("--cuda-visible", default="1", choices=["0", "1"])
    parser.add_argument("--peak-lr", type=float, default=mp.DEFAULT_PEAK_LR)
    parser.add_argument("--stage", choices=list(STAGES), default="all")
    parser.add_argument("--dest", type=Path, default=None, help="existing run dir (default: new UTC-stamped)")
    args = parser.parse_args()
    IDENTITY_MODE = str(args.identity_mode)
    CUDA_PIN = str(args.cuda_visible)

    if args.peak_lr <= 0:
        print("REFUSED: --peak-lr must be positive", file=sys.stderr)
        return 2
    # Any non-default peak LR is allowed but travels labeled into every
    # receipt (PICKS["SERIES"] LR-NOTE + run_meta/probe "peak_lr" fields);
    # 3e-4 would be a NEW cell class, never the series default (H1 lesson).

    gpu_stage = args.stage in ("train", "score", "all")
    if args.stage == "train" and os.environ.get(TRAIN_ENV_FLAG) != "1":
        print(f"REFUSED: training requires {TRAIN_ENV_FLAG}=1", file=sys.stderr)
        return 2
    if gpu_stage and os.environ.get("CUDA_VISIBLE_DEVICES") != CUDA_PIN:
        print(f"REFUSED: CUDA_VISIBLE_DEVICES must be pinned to {CUDA_PIN!r}", file=sys.stderr)
        return 2

    if args.dest:
        dest = args.dest
    elif IDENTITY_MODE == "concat":
        dest = plan.RESULT_ROOT / "m1_projadd_series" / f"CONCAT_W100_{utc_stamp()}"
    else:
        dest = plan.RESULT_ROOT / "m1_projadd_series" / f"P{args.proj_dim}-{utc_stamp()}"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[m1-projadd] cell={cell_id(args.proj_dim)} stage={args.stage} dest={dest} peak_lr={args.peak_lr:g}", flush=True)

    if gpu_stage or args.stage == "preflight":
        pre = gpu_preflight(dest / f"preflight_gpu_{args.stage}.json")
        print(f"[m1-projadd] preflight({args.stage}) ok={pre['ok']} foreign={pre.get('foreign_pids_on_target')}", flush=True)
        if gpu_stage and not pre["ok"]:
            _seal(
                dest / "BLOCKED.json",
                {
                    "schema": "btransform_unified_v1_m1_projadd_blocked",
                    "cell": cell_id(args.proj_dim),
                    "reason": f"GPU{CUDA_PIN} not clean at preflight (foreign pid >500 MiB or CUDA pin mismatch)",
                    "preflight": pre,
                    "utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(f"[m1-projadd] BLOCKED: GPU{CUDA_PIN} not clean; refusing to start", file=sys.stderr)
            return 2

    if args.stage == "preflight":
        return 0
    if args.stage in ("probe", "all"):
        run_probe(dest, args.proj_dim, args.peak_lr)
    if args.stage in ("train", "all"):
        summary = run_train(dest, args.proj_dim, args.peak_lr)
        print(
            f"[m1-projadd] train done in {summary['elapsed_s']:.0f}s; updates={summary['global_updates']}; "
            f"train_mse e1/e24={summary['train_mse'][1]:.6g}/{summary['train_mse'][24]:.6g}",
            flush=True,
        )
    if args.stage in ("score", "all"):
        score = run_score(dest, args.proj_dim)
        acc = score["acceptance_accounting"]
        print(
            f"[m1-projadd] acceptance: minival {acc['ours_minival_pooled']:.4f} vs threshold "
            f"{acc['accept_threshold_minival_pooled']:.4f} (above={acc['ours_minival_above_threshold']})",
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

            target = dest or (_plan.RESULT_ROOT / "m1_projadd_series" / "error")
            target.mkdir(parents=True, exist_ok=True)
            _receipts.seal_json(
                target / "error_receipt.json",
                {
                    "schema": "btransform_unified_v1_m1_projadd_error",
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "traceback": trace,
                },
            )
        except Exception:
            pass
        sys.exit(3)
