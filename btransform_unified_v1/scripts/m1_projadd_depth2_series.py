"""M1 proj_add temporal depth 4->2 front-row cell (ADDENDUM-DEPTH2-PROMOTED).

User directive 2026-09-07: the workorder
``WORKORDER_M1_PROJADD_RUNTIME_QUALITY_V1_20260907.md`` §7 priority plan
(temporal depth 4->2, proj_add P16 + CausalPE) is PROMOTED from the backlog
gate to a formal front-row experiment, to run on GPU as early as arbitration
allows. This script owns the cell; it does not modify the historical
``m1_projadd_series.py`` (read-only mechanism source) or any other agent's
files.

Design (workorder §7 "最小质量设计" + the promotion addendum):

  - Model: ``BTransformerUnifiedDecoderIdentity`` proj_add P16 with
    ``temporal_layers=2`` (P16, W=100, 8 slots, frontend, in-window sinusoidal
    PE all unchanged; ONLY the temporal block count changes; params
    3,533,616 -> 2,479,408). Public default stays 4 layers.
  - Training face: 3 sessions ses-20120926/27/28, chron-80 train rows of the
    ``m1_optimized_v2`` source_dev split (train windows end before the
    per-session 80% trial cut and start after M10 calib; 122,688 windows per
    the DEFERRED receipt estimate, exact count sealed by the probe).
    ses-20120924 is FULLY held out; the 31,252-window minival dev tail is
    held out. Same face as QueryAge 0.812-family / Original 0.809 minival
    references.
  - Recipe: the M2-verified noise-aligned TRN-1 transferred verbatim from
    ``m1_projadd_series.py`` (S1-domain unit dropout, F.conv1d primitive conv
    under bf16 autocast, EMA 0.9995, AdamW wd 0.01 clip 1, seed 42, peak LR
    1e-4, 24 epochs, batch 32, divisor=1).
  - Control: a MATCHED depth-4 baseline trained on the same face with the
    same recipe — the single variable is depth, nothing else.
  - Checkpoint rule: endpoint24 EMA (preregistered, only rule; no surface
    pick).
  - Faces: source-minival 31,252 (diagnostic) + LOSO ses-20120924 26,496
    (JUDGMENT face — a true holdout for the 3-session training face).
    References: QueryAge-family LOSO 0.65796 / Original LOSO 0.79833
    (leakage-disclosed: Original's training included 20120924).
  - Gate (preregistered): depth-2 minus depth-4 equal-session Delta R^2 >=
    -0.01 on the LOSO face (primary) -> PASS; < -0.01 -> FAIL.
  - Runtime validation (§7): pure-CPU temporal-core timing (B1/B4, threads
    1/4, interleaved reps) confirming the 2-layer temporal cost is roughly
    half the 4-layer cost; sealed into ``temporal_timing.json``.

Discipline (user directive 2026-09-07, coordinator relay — GPU TARGET SWITCHED
TO GPU0): arbitration target is GPU0 (GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9),
CUDA_VISIBLE_DEVICES pinned to '0' (``--cuda-visible``), preflight refuses on
a foreign pid >500 MiB on the TARGET (BLOCKED receipt). GPU0 may carry the
runtime-optimization agent's light reference-generation processes (<500 MiB)
— coexistence is authorized by the user; never evict each other. A foreign
GPU0 process >500 MiB that persists is reported, not killed. GPU1's training
belongs to another task and is NOT waited on. Receipts sealed via
``btransform_unified_v1.receipts.seal_json`` (0444 + sha256), historical
roots untouched. Overall budget 6h wall clock from the first train start
(budget anchor + per-run projection + BUDGET_HIT receipt).

Usage (one cell per invocation, AFTER the target GPU is clean — nothing is
auto-started):
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= \
    python scripts/m1_projadd_depth2_series.py --stage probe --root <dir>
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= \
    python scripts/m1_projadd_depth2_series.py --stage timing --root <dir>
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
    BTRANSFORM_M1_DEPTH2_TRAIN=1 \
    python scripts/m1_projadd_depth2_series.py --stage train --depth 2 --root <dir>
  ... --stage train --depth 4 --root <dir>
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
    python scripts/m1_projadd_depth2_series.py --stage score --depth 2 --root <dir>
  ... --stage score --depth 4 --root <dir>
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= \
    python scripts/m1_projadd_depth2_series.py --stage gate --root <dir>

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
from btransform_unified_v1.bank import TaskBank, array_sha256  # noqa: E402
from btransform_unified_v1.ema import DecoderEMA  # noqa: E402
from btransform_unified_v1.identity_variant import BTransformerUnifiedDecoderIdentity  # noqa: E402
from btransform_unified_v1.model import (  # noqa: E402
    CAUSAL_CHECK_TOLERANCE,
    N_LAYERS,
    UNIT_DROPOUT_DOMAIN_META,
    unit_dropout_seed,
    whole_unit_dropout,
)
from btransform_unified_v1.r2 import session_mean_report  # noqa: E402
from btransform_unified_v1.schedule import warmup_cosine_lr  # noqa: E402

TRAIN_ENV_FLAG = "BTRANSFORM_M1_DEPTH2_TRAIN"
SEED = mp.SERIES_SEED
EPOCHS = mp.SERIES_EPOCHS
PEAK_LR = mp.DEFAULT_PEAK_LR
BATCH_SIZE = plan.BATCH_SIZE
GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
GPU1_UUID = "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"
GPU_UUIDS = {"0": GPU0_UUID, "1": GPU1_UUID}
CUDA_PIN = "0"  # arbitration target switched to GPU0 per user directive 2026-09-07
FOREIGN_MEM_MIB_LIMIT = 500
OVERALL_BUDGET_SECONDS = 6.0 * 3600.0

TRAIN_SESSIONS = mp.M1_SOURCE_SESSIONS  # ses-20120926/27/28 only
LOSO_WINDOWS = mp.LOSO_OUTER_WINDOWS  # 26496
MINIVAL_WINDOWS = mp.SOURCE_MINIVAL_WINDOWS  # 31252
GATE_DELTA_R2 = -0.01  # preregistered §7 折中门
GATE_DELTA_R2_ADJUSTED = -0.03  # user-adjusted threshold 2026-09-07 (-0.01 -> -0.03)
DEPTH_CELLS = (2, 4)
DEPTH_DIR = {2: "depth2", 4: "depth4"}
TEMPORAL_BLOCK_PARAMS = 527104  # frozen per-block count (256-wide, 8 heads, FFN 512)
EXPECTED_PARAMS_BY_DEPTH = {
    4: mp.m1_projadd_param_count(16),  # 3,533,616
    2: mp.m1_projadd_param_count(16) - (N_LAYERS - 2) * TEMPORAL_BLOCK_PARAMS,  # 2,479,408
}
RUNTIME_CACHE_NPZ = (
    WORKSPACE_ROOT
    / "tfpd_exploration/results/decoder_validation_v2/20260905_190000/m1/m1_optimized_v2_source_runtime_cache.npz"
)
RUNTIME_LINE_ROOT = (
    WORKSPACE_ROOT / "btransform_unified_v1/results/m1_projadd_runtime_v1/20260907T020150Z"
)
STAGES = ("preflight", "probe", "timing", "train", "score", "gate", "pick", "packplan", "all")
FACES = ("chron80", "fullsession")

# The M1 analog of the M2 ext6 face: three LOCALLY VISIBLE official
# held-out-calib sessions (t0c1 VAL_HELDOUT_SESSIONS roster + frozen body
# digests; evaluation-only access law). Used ONLY by the pre-authorized
# post-gate epoch pick on the full-session build, disclosed as
# dev-on-official-selected.
HELDOUT_CALIB_SESSIONS = ("20121004", "20121017", "20121024")
HELDOUT_CALIB_BODY_SHA256 = {
    "20121004": "782c1fd090facfb3c50b6a85da8209ca3aea71f66bd4edc13d61fe4a55a3429d",
    "20121017": "8dd22c67500445ec1e0c11980475badbba9e960a05db16b78aaaad5502b3c652",
    "20121024": "bbeb6c7d66e2c2e9b76506021a6cf8800bc19021b6d1fd03c4eb6de71490bafb",
}
HELDOUT_CALIB_DIR_REL = "SPINT-main/data/000941/sub-MonkeyL-held-out-calib"
PICK_EPOCH_RANGE = (18, 24)  # endpoint24 minus up to 6 epochs (tail-shape visibility)
PICK_BATCH = 256  # eval-only batching (session-pure, fixed W; no sampler law on this face)

PICKS = {
    "CAL": [
        "CAL-2 {budget: 10 (M10), surface: sealed rSyn3-refit-v1 source-only NPZ (read-only) for "
        "the 3 TRAIN sessions ses-20120926/27/28; the LOSO ses-20120924 bank is built at score "
        "time only (sealed-basis encode, m1_family_loso_outer20120924 mechanism; the outer query "
        "file is never opened during training)}",
        "CAL-3f {identity: proj_add — bank E0 [64,100] (B3 Sfix e11 student.id_encoder, "
        "compute_identity(side_features=None), rSyn3 NOT in identity) -> P=Linear(100->16,bias=false) "
        "added onto the local conv channels by group broadcast; tokens=token_mlp([local+P]|carrier4), "
        "token_in=20 (matrix letter (f))}",
        "CAL-5 {unit order: units DataFrame row order; unit_mask all-true (64)}",
    ],
    "TRN": [
        "TRN-1-LR1e-4 {updates_per_epoch: FILLED_AT_RUNTIME, warmup_updates: FILLED_AT_RUNTIME, "
        "total_updates: FILLED_AT_RUNTIME, peak_lr: 1e-4 (M1 既定; H1 lesson: 3e-4 unvalidated on "
        "M1), min_lr: 0.1x peak, AdamW wd 0.01 betas (0.9,0.999) eps 1e-8, clip 1.0, batch 32, "
        "epochs 24}",
        "TRN-1 precision {bf16 autocast: forward+loss, pred.float() before MSE (S1 launch semantics)}",
        "TRN-3 {whole-unit dropout p=0.10, training mode only, per-batch CPU generator, domain: "
        "m2_small_unit_dropout (S1-aligned unit_dropout_seed; route-B verified)}",
        "TRN-5 {P: 0, L_in: 100}",
        "TRN-8 {scale: divisor=1; train and score native target}",
        "TRN-FACE {3-session chron-80: ses-20120926/27/28 train rows of the m1_optimized_v2 "
        "chron80 split (start after M10 calib, end before the per-session 80% trial cut); "
        "ses-20120924 fully held out (LOSO judgment face); the 31,252 minival dev tail held out "
        "(diagnostic face) — same face as QueryAge-family/Original minival references}",
    ],
    "SEL": [
        "SEL-1 {endpoint24 EMA — preregistered, the ONLY rule; no surface pick}",
        "SEL-3 {official surface: zero participation; LOSO ses-20120924 26,496 = JUDGMENT face "
        "(true holdout for the 3-session training face); source-minival 31,252 = diagnostic}",
    ],
    "SPD": ["none {training-path run; SPD-A1 folding exists (per-session static terms) but is not used in training or scoring}"],
    "ARCH": [
        "DEPTH {temporal depth 4->2 (CausalPE4-D2): proj_add P16, W=100, 8 slots, frontend, "
        "in-window sinusoidal PE all unchanged; ONLY the temporal block count changes "
        "(params 3,533,616 -> 2,479,408); public default stays 4 layers}",
        "GATE {depth-2 vs matched depth-4 baseline (same face, same recipe, same seed): "
        "equal-session Delta R^2 >= -0.01 (preregistered §7 折中门) on the LOSO face (primary "
        "judgment); user-adjusted threshold 2026-09-07 (-0.01 -> -0.03) is the operative "
        "budget; the paired receipt reports the verdict under BOTH thresholds}",
        "PROMOTED {depth 4->2 promoted from §7 backlog per user directive 2026-09-07 "
        "(ADDENDUM-DEPTH2-PROMOTED in WORKORDER_M1_PROJADD_RUNTIME_QUALITY_V1_20260907)}",
    ],
}

NOTE_SIX_ROWS = {
    "system": (
        "btransform_unified_v1 BTransformerUnifiedDecoderIdentity (m1 geometry, P=0, L_in=100=W, "
        "N=64, out=16 EMG, identity_mode=proj_add P16, token_in=20, temporal_layers=FILLED_AT_RUNTIME, "
        "seed 42) — NOT SPINT"
    ),
    "consumer": (
        "B-transformer unified decoder (8-slot + CausalPE temporal core, this cell). Consumers of "
        "frozen SPINT-lineage calibration objects (B3 Sfix e11 id_encoder + rSyn3). Comparisons "
        "against the SPINT family are 'same scoring surface, different system' only (P0-1)"
    ),
    "calibration_object": (
        "E0 [64,100] = B3 Sfix e11 student.id_encoder (sha 7976e0b0..., source-session provenance, "
        "the family-LOSO mechanism) via compute_identity(side_features=None) on the 10 chronological "
        "calib trials (rSyn3 NOT in identity, P1-10); carrier [64,4] = rSyn3 (sealed "
        "rSyn3-refit-v1 NPZ for 26/27/28; ses-20120924 sealed-basis encode with the source "
        "normalizer); M10 budget; per-session banks frozen (CAL-2). The outer session contributes "
        "only 10 unlabeled calib trials + query windows at score time — decoder-LOSO conditions"
    ),
    "scoring_surface": (
        "trained on 3 sessions (26/27/28) chron-80 train rows; JUDGMENT face = LOSO ses-20120924 "
        "26,496 windows (true holdout; references: QueryAge-family 0.65796, Original 0.79833 "
        "leakage-disclosed); diagnostic face = source-minival 31,252 (chron-80 dev tail; Original "
        "0.809289 exposed, family_flat selected 0.811652)"
    ),
    "scale": "divisor=1: train and score native target (M1 never divides by 20)",
    "single_difference_vs_historical_best": (
        "vs the MATCHED depth-4 baseline cell in the same results root (same chron-80 3-session "
        "face, same TRN-1-LR1e-4 recipe, same seed 42): the single differing row is the temporal "
        "depth 4->2. Vs the historical all-session P16 build the differences are the training face "
        "AND the depth (declared; that build is not the control here)"
    ),
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _picks_with_runtime(updates_per_epoch: int, peak_lr: float, depth: int) -> dict[str, Any]:
    total = updates_per_epoch * EPOCHS
    warmup = updates_per_epoch * plan.WARMUP_EPOCHS
    picks = json.loads(json.dumps(PICKS))
    picks["TRN"][0] = (
        picks["TRN"][0]
        .replace("updates_per_epoch: FILLED_AT_RUNTIME", f"updates_per_epoch: {updates_per_epoch}")
        .replace("warmup_updates: FILLED_AT_RUNTIME", f"warmup_updates: {warmup}")
        .replace("total_updates: FILLED_AT_RUNTIME", f"total_updates: {total}")
        .replace("peak_lr: FILLED_AT_RUNTIME", f"peak_lr: {peak_lr:g}")
    )
    note = json.loads(json.dumps(NOTE_SIX_ROWS))
    note["system"] = note["system"].replace("FILLED_AT_RUNTIME", str(depth))
    return {**picks, "note_six_rows": note}


def cell_id(depth: int) -> str:
    return "M1-PROJADD-P16-D2" if depth == 2 else "M1-PROJADD-P16-D4-BASE"


# ---------------------------------------------------------------------------
# Model builder (depth axis; asserts the full cell contract)
# ---------------------------------------------------------------------------


def prepare_trained_model(
    model: BTransformerUnifiedDecoderIdentity,
    *,
    banks: dict,
    device,
    face: str = "",
    stage: str = "",
    calib_by_session: dict | None = None,
) -> BTransformerUnifiedDecoderIdentity:
    """Hook for cells that attach extra session memory after the decoder build.

    Default is identity. Joint B3S overrides this to register calib + rSyn3
    and switch the decoder onto live E0. Called after ``.to(device)`` and the
    param-count gate, before optimizer / EMA / scoring.
    """
    return model


def _build_model(depth: int) -> BTransformerUnifiedDecoderIdentity:
    """proj_add P16 at a given temporal depth (depth 4 == the settled build)."""
    model = BTransformerUnifiedDecoderIdentity(
        mp.m1_projadd_geometry(16),
        seed=SEED,
        identity_mode="proj_add",
        proj_dim=16,
        temporal_layers=None if int(depth) == N_LAYERS else int(depth),
    )
    plan.require(model.identity_mode == "proj_add", "identity_mode drift")
    plan.require(model.init_meta["matrix_letter"] == "f", "matrix letter drift")
    plan.require(model.l_in == mp.M1_WINDOW and model.prefix == mp.M1_PREFIX, "l_in drift")
    plan.require(model.base_e0_dim == mp.M1_E0_DIM, "bank-side d_e drift")
    plan.require(model.units == mp.M1_UNITS and model.out_dim == mp.M1_OUT_DIM, "units/out drift")
    plan.require(model.token_in == 20, "token_in drift")
    plan.require(model.init_meta["temporal_layers"] == int(depth), "temporal depth drift")
    plan.require(len(model.temporal.blocks) == int(depth), "temporal block count drift")
    plan.require(
        tuple(model.temporal.pe.shape)[0] >= mp.M1_WINDOW,
        "temporal PE shorter than the window",
    )
    n_params = int(sum(p.numel() for p in model.parameters()))
    plan.require(
        n_params == EXPECTED_PARAMS_BY_DEPTH[int(depth)],
        f"param count {n_params} != expected {EXPECTED_PARAMS_BY_DEPTH[int(depth)]} at depth {depth}",
    )
    return model


# ---------------------------------------------------------------------------
# GPU preflight (target GPU pinned; foreign pid > 500 MiB on target => BLOCKED)
# ---------------------------------------------------------------------------


def _nvidia_smi(args: list[str]) -> str:
    out = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def gpu_preflight(out_path: Path, depth: int, stage: str) -> dict[str, Any]:
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
    target_uuid = GPU_UUIDS[CUDA_PIN]
    foreign = [
        app
        for app in apps
        if app["gpu_uuid"] == target_uuid and app["pid"] != own_pid and app["used_mib"] > FOREIGN_MEM_MIB_LIMIT
    ]
    light = [
        app
        for app in apps
        if app["gpu_uuid"] == target_uuid and app["pid"] != own_pid and app["used_mib"] <= FOREIGN_MEM_MIB_LIMIT
    ]
    target_gpu = next(g for g in gpus if g["uuid"] == target_uuid)
    report = {
        "schema": "btransform_unified_v1_m1_projadd_depth2_gpu_preflight",
        "cell": cell_id(depth),
        "stage": stage,
        "unix": time.time(),
        "utc": utc_iso(),
        "own_pid": own_pid,
        "cuda_visible_devices": raw,
        "cuda_pin": str(CUDA_PIN),
        "arbitration": (
            "target switched GPU1 -> GPU0 per user directive 2026-09-07 (coordinator relay); "
            "GPU0 light co-tenants <=500 MiB (runtime-agent reference generation) coexist and "
            "are never evicted; >500 MiB persistent foreigners are reported, not killed"
        ),
        "gpus": gpus,
        "compute_apps": apps,
        f"gpu{CUDA_PIN}": target_gpu,
        "foreign_pids_on_target": foreign,
        "light_coexisting_on_target": light,
        "foreign_threshold_mib": FOREIGN_MEM_MIB_LIMIT,
        "torch_cuda_visible_count": torch.cuda.device_count() if raw == str(CUDA_PIN) else None,
    }
    ok = raw == str(CUDA_PIN) and not foreign and torch.cuda.device_count() == 1
    report["ok"] = bool(ok)
    if out_path.exists():
        _write_json(out_path.with_name(out_path.stem + "_live.json"), report)
    else:
        _seal(out_path, report)
    return report


# ---------------------------------------------------------------------------
# Noise-aligned startup self-checks (carried from the series blueprint)
# ---------------------------------------------------------------------------


def verify_dropout_parity() -> dict[str, Any]:
    """Prove our loop's mask draw equals S1 ``unit_dropout_mask`` bit for bit."""
    from tfpd_exploration.src.m2_b_small_stability_v1 import training as s1_training

    probes: list[dict[str, Any]] = []
    unit_mask = torch.ones(mp.M1_UNITS, dtype=torch.bool)
    for epoch, batch_id in ((1, 0), (1, 1), (2, 100), (12, 5000), (24, 3833)):
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
        "schema": "btransform_unified_v1_m1_projadd_depth2_dropout_parity",
        "reference": "tfpd_exploration.src.m2_b_small_stability_v1.training.unit_dropout_mask",
        "domain_meta": UNIT_DROPOUT_DOMAIN_META,
        "probes": probes,
        "all_bitwise_equal": True,
    }


def verify_conv_path(model: BTransformerUnifiedDecoderIdentity) -> dict[str, Any]:
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


def _stub_bank(e0: np.ndarray) -> TaskBank:
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


def verify_proj_add_degeneration(model: BTransformerUnifiedDecoderIdentity) -> dict[str, Any]:
    model = model.eval()
    rng = np.random.default_rng(123)
    e0 = rng.standard_normal((mp.M1_UNITS, mp.M1_E0_DIM)).astype(np.float32)
    alt = rng.standard_normal((mp.M1_UNITS, mp.M1_E0_DIM)).astype(np.float32)
    x = torch.from_numpy(rng.standard_normal((2, mp.M1_WINDOW, mp.M1_UNITS)).astype(np.float32))
    bank_a = _stub_bank(e0)
    bank_b = _stub_bank(alt)
    with torch.no_grad():
        model.frontend.e0_proj.weight.zero_()
        y_a = model.forward_scores(x, bank_a)
        y_b = model.forward_scores(x, bank_b)
    bitexact = bool(torch.equal(y_a, y_b))
    plan.require(bitexact, "proj_add P-zero degeneration control FAILED")
    return {"control": "P=0 -> forward bitwise independent of bank E0 (alt E0 draw)", "bitwise_equal": bitexact}


def verify_fold_parity(model: BTransformerUnifiedDecoderIdentity) -> dict[str, Any]:
    model = model.eval()
    rng = np.random.default_rng(11)
    bank = _stub_bank(rng.standard_normal((mp.M1_UNITS, mp.M1_E0_DIM)).astype(np.float32))
    x = torch.from_numpy(rng.standard_normal((2, mp.M1_WINDOW, mp.M1_UNITS)).astype(np.float32))
    static = model.bank_static_term(bank)
    delta = mp.assert_fold_parity(model, bank, x, static)
    return {
        "folded_vs_unfolded_max_abs_delta": delta,
        "tolerance": 1e-6,
        "note": "per-session static term = sum_g P_g @ W_g^T + carrier @ W_carrier^T + b (P(E0) folded)",
    }


# ---------------------------------------------------------------------------
# Chron-80 3-session data face + banks
# ---------------------------------------------------------------------------


def _source_dev_modules():
    from tfpd_exploration.src.m1_optimized_v2 import bank as src_bank, source_dev

    return src_bank, source_dev


def build_chron80_face():
    """The 3-session chron-80 TRAIN face (m1_optimized_v2 source_dev mechanism).

    Returns (train_ds, sampler, train_rows, dev_rows, rows_report, loaded).
    The outer session file is never opened on this path (the source-only
    datamodule asserts no target materialization).
    """
    from torch.utils.data import DataLoader  # noqa: F401  (import surface check)

    src_bank, source_dev = _source_dev_modules()
    loaded = src_bank.load()
    dm_src = source_dev.build_source_only_datamodule(loaded)
    train_rows, dev_rows, rows_report = source_dev._split(dm_src)
    plan.require(
        len(dev_rows) == MINIVAL_WINDOWS,
        f"minival cardinality drift: {len(dev_rows)} != {MINIVAL_WINDOWS}",
    )
    inner = copy.copy(dm_src.train_dataset.base)
    inner.window_indices = list(train_rows)
    train_ds = source_dev.SourceCarrierDataset(inner, loaded)
    from src.data.falcon_datamodule import SessionBatchSampler  # streaming path set by mp imports

    mp._streaming_path()
    sampler = SessionBatchSampler(
        train_ds,
        BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        balance_sessions=False,
        reshuffle_each_epoch=False,
    )
    return train_ds, sampler, train_rows, dev_rows, rows_report, loaded


def calib_from_source_dm() -> dict[str, np.ndarray]:
    """First M10 chronological calib trials per SOURCE session (bank input)."""
    src_bank, source_dev = _source_dev_modules()
    loaded = src_bank.load()
    dm_src = source_dev.build_source_only_datamodule(loaded)
    return {
        name: np.asarray(dm_src.train_dataset.base.calib_trialized_neural_features[name][: mp.M10_BUDGET])
        for name in TRAIN_SESSIONS
    }


def build_source_banks(calib: dict[str, np.ndarray]) -> tuple[dict[str, TaskBank], dict[str, Any]]:
    """3-session frozen banks: sealed NPZ carriers + real B3 identity."""
    plan.require(set(calib) == set(TRAIN_SESSIONS), "source bank calib sessions drift")
    carriers = mp.load_source_carriers()
    provider = mp.default_identity_provider()
    banks: dict[str, TaskBank] = {}
    report: dict[str, Any] = {}
    for session in TRAIN_SESSIONS:
        cal = np.asarray(calib[session])
        if cal.shape[0] < mp.M10_BUDGET:
            raise RuntimeError(f"{session}: fewer than {mp.M10_BUDGET} calib trials")
        e0 = provider(cal[: mp.M10_BUDGET])
        banks[session] = mp.make_m1_bank(session, e0, np.asarray(carriers[session]))
        report[session] = {
            "e0_sha256": banks[session].calibration_meta["array_sha256"],
            "carrier_sha256": banks[session].calibration_meta["carrier_sha256"],
            "trial_count": mp.M10_BUDGET,
            "budget": mp.M10_BUDGET,
        }
    return banks, report


def crosscheck_banks_vs_runtime_cache(banks: dict[str, TaskBank]) -> dict[str, Any]:
    """Source-session banks must reproduce the sealed runtime cache bitwise."""
    cache = np.load(RUNTIME_CACHE_NPZ, allow_pickle=False)
    checks: dict[str, Any] = {}
    for name in TRAIN_SESSIONS:
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


def build_outer_bank() -> tuple[TaskBank, dict[str, Any]]:
    """LOSO ses-20120924 bank (score time only): sealed-basis carrier + B3 E0."""
    dm = mp.build_loso_datamodule()
    val = dm.val_heldin_dataset
    if val is None:
        raise RuntimeError("LOSO val_heldin_dataset missing")
    calib10 = np.asarray(val.calib_trialized_neural_features[mp.M1_OUTER_SESSION][: mp.M10_BUDGET])
    if calib10.shape[0] < mp.M10_BUDGET:
        raise RuntimeError("outer session has fewer than M10 calib trials")
    outer_carrier, carrier_meta = mp.encode_outer_carrier()
    provider = mp.default_identity_provider()
    e0 = provider(calib10[: mp.M10_BUDGET])
    bank = mp.make_m1_bank(mp.M1_OUTER_SESSION, e0, outer_carrier)
    meta = {
        "e0_sha256": bank.calibration_meta["array_sha256"],
        "carrier_sha256": bank.calibration_meta["carrier_sha256"],
        "carrier_meta": carrier_meta,
    }
    return bank, meta


# ---------------------------------------------------------------------------
# Full-session face (pre-authorized stage 2: 4 held-in sessions, the series
# universe; ses-20120924 becomes a legal training member per stage-2 framing)
# ---------------------------------------------------------------------------


def build_fullsession_face():
    """The 4-session training universe (m1_projadd_series mechanism, read-only).

    ``mp.assemble_training_universe`` builds the combined FalconDataset under
    the fold-local train-split law (query_start_trial=0, query_end_trial=None,
    eval_mask last bin, 99-bin zero pre-history pad) + SessionBatchSampler(
    batch=32, shuffle=True, seed=42, balance=False, no reshuffle).
    """
    dm = mp.build_loso_datamodule()
    dataset, sampler = mp.assemble_training_universe(dm)
    return dataset, sampler


def build_fullsession_banks(dataset) -> tuple[dict[str, TaskBank], dict[str, Any]]:
    """All-4-session frozen banks (series build_real_banks mechanism)."""
    calib = mp.calib_trials_from_dataset(dataset)
    carriers = mp.load_source_carriers()
    outer_carrier, outer_meta = mp.encode_outer_carrier()
    carriers = {**carriers, mp.M1_OUTER_SESSION: outer_carrier}
    banks, report = mp.build_banks(calib, carriers)
    report["outer_carrier_meta"] = outer_meta
    return banks, report


# ---------------------------------------------------------------------------
# Held-out-calib pick face (M1 ext6 analog; evaluation-only access law)
# ---------------------------------------------------------------------------


def _heldout_calib_path(session: str) -> Path:
    plan.require(session in HELDOUT_CALIB_SESSIONS, f"not a held-out-calib session: {session!r}")
    path = WORKSPACE_ROOT / HELDOUT_CALIB_DIR_REL / f"sub-MonkeyL-held-out-calib_ses-{session}_behavior+ecephys.nwb"
    text = str(path)
    for token in ("minival", "test", "evalai", "formal", "held-out-minival"):
        plan.require(token not in text.lower().replace("held-out-calib", ""), f"forbidden token {token!r} in {text}")
    plan.require("held-out-calib" in text, f"not a held-out-calib path: {text}")
    plan.require("20120924" not in text, "20120924 must not appear in a pick-face path")
    return path


def open_heldout_calib_session(session: str) -> dict[str, Any]:
    """Open one held-out-calib session (t0c1 direct-reader mechanism, SHA-gated).

    ``FalconDataModule`` is constructed for its ``prepare_session_data``
    utility only (``setup`` is NEVER called, so no broad file discovery);
    the dataset is built with ``query_start_trial=0`` (the ext6-style full
    local query law). Labels are read for the METRIC only.
    """
    import hashlib

    mp._streaming_path()
    from falcon_challenge.config import FalconConfig, FalconTask
    from src.data.falcon_datamodule import FalconDataModule

    path = _heldout_calib_path(session)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    plan.require(
        digest == HELDOUT_CALIB_BODY_SHA256[session],
        f"held-out-calib body sha drift for {session}: {digest}",
    )
    dm = FalconDataModule(
        task="m1",
        data_dir=str(WORKSPACE_ROOT / "SPINT-main/data/000941"),
        heldin_session_names=[],
        batch_size=32,
        window_size=mp.M1_WINDOW,
        calibration_n_trials=10,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=1024,
        standardize_covariates=False,
        use_intertrials=True,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        pad_value=-1.0,
        num_workers=0,
        pin_memory=False,
        validation_protocol="loso",
        loso_fold=0,
        include_heldout_in_fit=False,
        include_heldout_in_test=False,
        query_start_trial=0,
        heldin_query_start_trial=0,
        heldin_query_end_trial=None,
        allow_empty_heldout_query=False,
        sampler_seed=SEED,
        balance_session_batches=False,
        reshuffle_train_sampler_each_epoch=False,
        side_feature_group="none",
        side_feature_shuffle_seed=0,
    )
    record = dm.prepare_session_data(
        path,
        FalconConfig(task=FalconTask.m1).task,
        standardize_covariates=False,
        covariates_mean=None,
        covariates_std=None,
        use_intertrials=True,
    )
    from src.data.falcon_datamodule import FalconDataset

    dataset = FalconDataset(
        sessions_dict={session: record},
        calib_sessions_dict={session: record},
        window_size=mp.M1_WINDOW,
        split="train",
        calibration_n_trials=10,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=1024,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        remove_still_times=False,
        remove_calib_still_times=False,
        use_calib_active_segments=False,
        calib_n_active_segments=1,
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        pad_value=-1.0,
        side_feature_group="none",
        side_feature_shuffle_seed=0,
        query_start_trial=0,
    )
    trials = np.ascontiguousarray(
        dataset.calib_trialized_neural_features[session], dtype=np.float32
    )
    plan.require(
        trials.ndim == 3 and trials.shape[1:] == (1024, mp.M1_UNITS) and trials.shape[0] >= mp.M10_BUDGET,
        f"held-out-calib trialized activity topology drift: {session} {trials.shape}",
    )
    return {
        "dataset": dataset,
        "calib10": trials[: mp.M10_BUDGET],
        "session": session,
        "body_sha256": digest,
        "n_windows": int(len(dataset.window_indices)),
    }


def encode_heldout_calib_carrier(session: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Sealed-basis rSyn3 carrier for a held-out-calib session (M10 support only).

    Read-only reuse of the ``m1_b3_allsource_v1`` public-calib mechanism (the
    sanctioned loader for later-day 10-trial held-out-calib files; never reads
    beyond trial M10) + the sealed ``m1_optimized_v2`` basis/normalizer — the
    same lineage as ``mp.encode_outer_carrier``.
    """
    path = _heldout_calib_path(session)
    from tfpd_exploration.src.m1_b3_allsource_v1 import rsyn3_bank
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3
    from tfpd_exploration.src.m1_optimized_v2 import bank as src_bank
    from tfpd_exploration.src.m1_optimized_v2 import plan as m1_plan

    loaded = src_bank.load()
    blob = np.load(m1_plan.BANK_NPZ, allow_pickle=False)
    basis = syn3.SourceBasis(
        kind="nnmf",
        scale=np.asarray(blob["scale"]),
        dictionary=np.asarray(blob["d0"]),
        activations=np.asarray(blob["activations"]),
        order=tuple(int(v) for v in blob["nmf_order"]),
        reconstruction_digest=str(blob["reconstruction_digest"][0]),
        library={"source": "rSyn3-refit-v1.sealed"},
        extra={},
    )
    record = rsyn3_bank.load_public_calib_support(path)
    raw = rsyn3_bank._encode_record(record, basis)
    normalized = np.ascontiguousarray(
        syn3.normalize_carriers(raw, loaded["normalizer_mean"], loaded["normalizer_scale"]),
        dtype=np.float32,
    )
    plan.require(normalized.shape == (mp.M1_UNITS, mp.M1_CARRIER_DIM), "held-out-calib carrier shape drift")
    meta = {
        "raw_carrier_array_digest": syn3.array_digest(np.ascontiguousarray(raw)),
        "normalized_carrier_sha256": array_sha256(normalized),
        "mechanism": (
            "m1_b3_allsource_v1.rsyn3_bank.load_public_calib_support (M10 support only, later-day "
            "calib sanctioned reader) + sealed-basis encode + source normalizer"
        ),
        "query_values_read": False,
    }
    return normalized, meta


# ---------------------------------------------------------------------------
# Stage: probe (CPU-only; seals the face inventory both trains assert)
# ---------------------------------------------------------------------------


def run_probe(root: Path, dest: Path, face: str) -> dict[str, Any]:
    t0 = time.monotonic()
    if face == "chron80":
        train_ds, sampler, train_rows, dev_rows, rows_report, _loaded = build_chron80_face()
        banks, bank_report = build_source_banks(calib_from_source_dm())
        train_sessions = list(TRAIN_SESSIONS)
        split_rows = rows_report
        train_protocol = (
            "3-session chron-80 face: ses-20120926/27/28 train rows (start after M10 calib, end "
            "before the per-session 80% trial cut; m1_optimized_v2 source_dev._split mechanism); "
            "ses-20120924 fully held out (LOSO judgment); minival dev tail held out (diagnostic)"
        )
        minival_dev_windows = int(len(dev_rows))
    elif face == "fullsession":
        train_ds, sampler = build_fullsession_face()
        banks, bank_report = build_fullsession_banks(train_ds)
        train_sessions = list(mp.M1_SESSIONS)
        split_rows = None
        train_protocol = (
            "stage-2 full-session face: ALL 4 held-in sessions ses-20120924/26/27/28 under the "
            "series training-universe law (query_start_trial=0, query_end_trial=None); "
            "minival/LOSO faces become DIAGNOSTIC only (polluted; official score is the only "
            "selector-bearing number)"
        )
        minival_dev_windows = None
    else:
        raise plan.BTransformerUnifiedError(f"unknown face {face!r}")
    cache_checks = crosscheck_banks_vs_runtime_cache({n: banks[n] for n in TRAIN_SESSIONS})

    models = {depth: _build_model(depth) for depth in DEPTH_CELLS}
    params = {depth: int(sum(p.numel() for p in models[depth].parameters())) for depth in DEPTH_CELLS}

    dropout_parity = verify_dropout_parity()
    conv_path = verify_conv_path(models[2])
    degeneration = verify_proj_add_degeneration(copy.deepcopy(models[2]))
    fold_parity = verify_fold_parity(models[2])
    divisor = mp.assert_divisor_identity(models[2])

    x = torch.from_numpy(
        np.ascontiguousarray(train_ds[0][0], dtype=np.float32).reshape(1, mp.M1_WINDOW, mp.M1_UNITS)
    )
    session0 = train_ds[0][3]
    session0 = session0.decode() if isinstance(session0, bytes) else str(session0)
    causal = models[2].causal_check(x, banks[session0])

    windows = {
        name: int(sum(1 for n, _ in train_ds.window_indices if n == name)) for name in train_sessions
    }
    updates_per_epoch = int(len(sampler))
    inventory = {
        "schema": "btransform_unified_v1_m1_projadd_depth2_session_inventory",
        "route": "ADDENDUM-DEPTH2-PROMOTED (user directive 2026-09-07; §7 priority plan -> formal cell)",
        "face": face,
        "utc": utc_iso(),
        "train_protocol": train_protocol,
        "sampler": "SessionBatchSampler(batch=32, shuffle=True, seed=42, balance=False, reshuffle_each_epoch=False)",
        "sampler_batch_sha256": mp.sampler_digest(sampler),
        "sessions": train_sessions,
        "windows_per_session": windows,
        "total_train_windows": int(len(train_ds.window_indices)),
        "updates_per_epoch": updates_per_epoch,
        "warmup_updates": updates_per_epoch * plan.WARMUP_EPOCHS,
        "total_updates": updates_per_epoch * EPOCHS,
        "split_rows": split_rows,
        "minival_dev_windows": minival_dev_windows,
        "train_law": (
            "chron-80: windows start after M10 calib and end < 80% trial cut; eval_mask at last "
            "bin; pre-history zero pad"
            if face == "chron80"
            else "fullsession: query_start_trial=0, query_end_trial=None (full timeline); eval_mask "
            "at last bin; pre-history zero pad"
        ),
        "depth_cells": {
            str(depth): {
                "cell_id": cell_id(depth),
                "temporal_layers": depth,
                "decoder_params": params[depth],
                "expected_decoder_params": EXPECTED_PARAMS_BY_DEPTH[depth],
                "temporal_block_params": TEMPORAL_BLOCK_PARAMS,
            }
            for depth in DEPTH_CELLS
        },
        "banks": bank_report,
        "bank_determinism": (
            "identity E0 computed under pinned torch threads (b3_identity/_PinnedThreads); bank "
            "bytes byte-identical to the sealed m1_optimized_v2 runtime cache (crosscheck below)"
        ),
        "runtime_cache_crosscheck": cache_checks,
        "calib_trials_per_session": mp.M10_BUDGET,
        "self_checks": {
            "dropout_parity_vs_s1": dropout_parity,
            "conv_path": conv_path,
            "proj_add_degeneration": degeneration,
            "fold_parity": fold_parity,
            "divisor_identity": divisor,
            "causal_check_depth2": causal,
        },
        "probe_seconds": time.monotonic() - t0,
        "picks": _picks_with_runtime(updates_per_epoch, PEAK_LR, 2),
    }
    _seal(dest / "session_inventory.json", inventory)
    print(
        f"[depth2] probe: train_windows={inventory['total_train_windows']} ({windows}) "
        f"updates/epoch={updates_per_epoch} params d2/d4={params[2]}/{params[4]}",
        flush=True,
    )
    return inventory


# ---------------------------------------------------------------------------
# Stage: timing (CPU-only temporal-core cost; §7 runtime validation)
# ---------------------------------------------------------------------------


def run_timing(dest: Path) -> dict[str, Any]:
    """Pure-CPU temporal-core timing: depth 2 vs 4, B1/B4, threads 1/4.

    Workorder timing discipline (§1.3/§2): default CPU, official B<=4,
    workers=0, no profiler, threads pinned, repeated interleaved runs (no
    single best pick). The temporal stack alone is timed on a random
    [B, W=100, 256] input — the per-bin cost axis §7 asks about (each predict
    call replays the full window through the temporal core).
    """
    saved_threads = torch.get_num_threads()
    reps, warmup = 21, 3
    configs = [(threads, depth, batch) for threads in (1, 4) for depth in DEPTH_CELLS for batch in (1, 4)]
    models = {depth: _build_model(depth).eval() for depth in DEPTH_CELLS}
    inputs = {batch: torch.randn(batch, mp.M1_WINDOW, 256) for batch in (1, 4)}
    samples: dict[tuple[int, int, int], list[float]] = {cfg: [] for cfg in configs}
    try:
        for threads, depth, batch in configs:  # warmup each config under its pin
            torch.set_num_threads(threads)
            with torch.inference_mode():
                for _ in range(warmup):
                    models[depth].temporal(inputs[batch])
        for _rep in range(reps):  # interleaved reps: one call per config per round
            for cfg in configs:
                threads, depth, batch = cfg
                torch.set_num_threads(threads)
                with torch.inference_mode():
                    t0 = time.perf_counter()
                    models[depth].temporal(inputs[batch])
                    samples[cfg].append((time.perf_counter() - t0) * 1e3)
    finally:
        torch.set_num_threads(saved_threads)

    def _stats(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean_ms": float(arr.mean()),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "min_ms": float(arr.min()),
            "reps": int(arr.size),
        }

    table: dict[str, Any] = {}
    for threads in (1, 4):
        for batch in (1, 4):
            s2 = _stats(samples[(threads, 2, batch)])
            s4 = _stats(samples[(threads, 4, batch)])
            table[f"threads{threads}_B{batch}"] = {
                "depth2": s2,
                "depth4": s4,
                "ratio_d2_over_d4_mean": s2["mean_ms"] / s4["mean_ms"],
                "ratio_d2_over_d4_median": s2["median_ms"] / s4["median_ms"],
            }
    receipt = {
        "schema": "btransform_unified_v1_m1_projadd_depth2_temporal_timing",
        "route": "§7 runtime validation (ADDENDUM-DEPTH2-PROMOTED): untrained seed-42 models, "
        "pure-CPU temporal-core forward, official B<=4, no profiler, interleaved reps",
        "utc": utc_iso(),
        "model_state": "untrained (seed 42 init)",
        "timing_target": "model.temporal(x): PE add + N causal blocks, x [B, 100, 256]",
        "reps_per_config": reps,
        "warmup_per_config": warmup,
        "interleaved": True,
        "cpu_model": _cpu_model(),
        "torch": torch.__version__,
        "table": table,
        "verdict_note": (
            "development-machine CPU evidence for the depth cost axis only (workorder §1.3: never "
            "present local timing as official-CPU evidence); the ~2x order-of-magnitude check is "
            "the §7 gate, final runtime budget stays with the P1/P3 replay discipline"
        ),
    }
    _seal(dest / "temporal_timing.json", receipt)
    b4_t1 = table["threads1_B4"]
    print(
        f"[depth2] timing: B4 threads1 d2={b4_t1['depth2']['median_ms']:.3f}ms "
        f"d4={b4_t1['depth4']['median_ms']:.3f}ms ratio(median)={b4_t1['ratio_d2_over_d4_median']:.3f}",
        flush=True,
    )
    return receipt


def _cpu_model() -> str:
    try:
        return Path("/proc/cpuinfo").read_text(encoding="utf-8").split("model name")[1].splitlines()[0].split(":", 1)[1].strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Budget anchor (6h overall, shared by both cells)
# ---------------------------------------------------------------------------


def _budget_anchor(root: Path) -> dict[str, Any]:
    path = root / "budget_anchor.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "schema": "btransform_unified_v1_m1_projadd_depth2_budget_anchor",
        "first_train_started_utc": utc_iso(),
        "unix": time.time(),
        "budget_seconds": OVERALL_BUDGET_SECONDS,
        "note": "6h overall wall budget from the first train start (user directive 2026-09-07)",
    }
    _seal(path, payload)
    return payload


# ---------------------------------------------------------------------------
# Stage: train (GPU1; noise-aligned TRN-1-LR1e-4, endpoint24 EMA rule)
# ---------------------------------------------------------------------------


def _collate(items):
    """Chron-80 SourceCarrierDataset items: (neural, target, calib, session, carrier)."""
    neural = torch.from_numpy(
        np.ascontiguousarray(np.stack([np.asarray(it[0], dtype=np.float32) for it in items]), dtype=np.float32)
    )
    target = torch.from_numpy(
        np.ascontiguousarray(np.stack([np.asarray(it[1], dtype=np.float32)[-1] for it in items]), dtype=np.float32)
    )
    sessions = [it[3].decode() if isinstance(it[3], bytes) else str(it[3]) for it in items]
    return neural, target, sessions


def _load_inventory(root: Path) -> dict[str, Any]:
    """The face inventory lives at the RUN ROOT (probe/timing dest), not the cell dir."""
    path = root / "session_inventory.json"
    plan.require(
        path.is_file(),
        f"session_inventory.json missing at {path} — run --stage probe first (it seals the "
        "face inventory both train cells assert)",
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_complete_epoch(dest: Path) -> int:
    last = 0
    for epoch in range(1, EPOCHS + 1):
        if not (dest / f"epoch_{epoch:03d}.pt").is_file():
            break
        last = epoch
    return last


def _epoch_series_from_metrics(dest: Path) -> tuple[dict[int, float], dict[int, float]]:
    mse: dict[int, float] = {}
    lr: dict[int, float] = {}
    path = dest / "metrics.jsonl"
    if not path.is_file():
        return mse, lr
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") != "epoch":
            continue
        epoch = int(row["epoch"])
        mse[epoch] = float(row["train_mse"])
        lr[epoch] = float(row["lr"])
    return mse, lr


def run_train(root: Path, dest: Path, depth: int, peak_lr: float, face: str) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)

    inventory = _load_inventory(root)
    if face == "chron80":
        train_ds, sampler, _tr, _dv, _rows, _loaded = build_chron80_face()
        banks, _report = build_source_banks(calib_from_source_dm())
        bank_sessions = list(TRAIN_SESSIONS)
    elif face == "fullsession":
        train_ds, sampler = build_fullsession_face()
        banks, _report = build_fullsession_banks(train_ds)
        bank_sessions = list(mp.M1_SESSIONS)
    else:
        raise plan.BTransformerUnifiedError(f"unknown face {face!r}")
    if mp.sampler_digest(sampler) != inventory["sampler_batch_sha256"]:
        raise RuntimeError("sampler digest drift vs sealed probe inventory")
    if len(sampler) != inventory["updates_per_epoch"]:
        raise RuntimeError("updates_per_epoch drift vs sealed probe inventory")
    for name in bank_sessions:
        sealed = inventory["banks"][name]
        if banks[name].calibration_meta["array_sha256"] != sealed["e0_sha256"]:
            raise RuntimeError(f"bank E0 digest drift vs sealed probe inventory: {name}")
        if banks[name].calibration_meta["carrier_sha256"] != sealed["carrier_sha256"]:
            raise RuntimeError(f"bank carrier digest drift vs sealed probe inventory: {name}")

    model = _build_model(depth).to(device)
    if int(sum(p.numel() for p in model.parameters())) != EXPECTED_PARAMS_BY_DEPTH[depth]:
        raise RuntimeError("param count drift vs expectation")
    train_calib = None
    if face == "fullsession":
        train_calib = mp.calib_trials_from_dataset(train_ds)
    elif face == "chron80":
        train_calib = calib_from_source_dm()
    model = prepare_trained_model(
        model,
        banks=banks,
        device=device,
        face=face,
        stage="train",
        calib_by_session=train_calib,
    )

    from tfpd_exploration.src.m2_dual_track_v1 import training as dual_training

    optimizer = dual_training.build_optimizer(
        model.trainable_parameters().items(), lr=peak_lr, weight_decay=plan.WEIGHT_DECAY
    )
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    updates_per_epoch = inventory["updates_per_epoch"]
    total_updates = updates_per_epoch * EPOCHS
    warmup_updates = updates_per_epoch * plan.WARMUP_EPOCHS
    picks = _picks_with_runtime(updates_per_epoch, peak_lr, depth)

    anchor = _budget_anchor(root)
    deadline_unix = float(anchor["unix"]) + OVERALL_BUDGET_SECONDS
    last_complete = _latest_complete_epoch(dest)
    resume_from = last_complete if last_complete >= 1 and not (dest / "train_receipt.json").exists() else 0
    if resume_from and last_complete < EPOCHS:
        ckpt_path = dest / f"epoch_{last_complete:03d}.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ckpt.get("schema") != "btransform_unified_v1_m1_projadd_depth2_ckpt":
            raise RuntimeError(f"resume ckpt schema drift: {ckpt_path}")
        if int(ckpt.get("epoch", -1)) != last_complete:
            raise RuntimeError(f"resume ckpt epoch {ckpt.get('epoch')} != {last_complete}")
        if int(ckpt.get("global_step", -1)) != last_complete * updates_per_epoch:
            raise RuntimeError(
                f"resume ckpt global_step {ckpt.get('global_step')} != {last_complete * updates_per_epoch}"
            )
        model.load_state_dict(ckpt["raw_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        ema.load_state_dict(ckpt["ema"])
        print(
            f"[depth2] resume {cell_id(depth)} from {ckpt_path.name} "
            f"epoch={last_complete} step={ckpt['global_step']} ema={ema.n_updates}",
            flush=True,
        )
    elif resume_from and last_complete >= EPOCHS:
        print(f"[depth2] all {EPOCHS} epoch ckpts present; sealing train receipt only", flush=True)

    if not (dest / "run_meta.json").exists():
        _seal(
            dest / "run_meta.json",
            {
            "schema": "btransform_unified_v1_m1_projadd_depth2_run_meta",
            "cell": cell_id(depth),
            "route": "ADDENDUM-DEPTH2-PROMOTED (user directive 2026-09-07; §7 priority plan -> formal cell)",
            "reused_artifacts": {
                "session_inventory": str(root / "session_inventory.json"),
                "session_inventory_sha256": (root / "session_inventory.json.sha256").read_text(encoding="utf-8").split()[0]
                if (root / "session_inventory.json.sha256").is_file()
                else None,
                "temporal_timing": str(root / "temporal_timing.json"),
                "note": (
                    "probe (face inventory: sampler digest, bank digests, self-checks) and the "
                    "CPU temporal timing receipt were sealed BEFORE training and are reused "
                    "as-is on restart; not recomputed"
                ),
            },
            "utc": utc_iso(),
            "seed": SEED,
            "temporal_layers": depth,
            "peak_lr": peak_lr,
            "geometry": mp.m1_projadd_geometry(16),
            "l_in": model.l_in,
            "token_in": model.token_in,
            "decoder_params": EXPECTED_PARAMS_BY_DEPTH[depth],
            "temporal_block_params": TEMPORAL_BLOCK_PARAMS,
            "init_meta": model.init_meta,
            "unit_dropout_domain_meta": UNIT_DROPOUT_DOMAIN_META,
            "face": face,
            "sessions": bank_sessions,
            "heldout_sessions": [] if face == "fullsession" else [mp.M1_OUTER_SESSION],
            "windows_per_session": inventory["windows_per_session"],
            "total_windows": inventory["total_train_windows"],
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
            "budget_anchor": anchor,
            "gpu_uuid": GPU_UUIDS[CUDA_PIN],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "picks": picks,
            },
        )

    loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=_collate, num_workers=0)
    metrics_path = dest / "metrics.jsonl"
    heartbeat = dest / "heartbeat.json"
    started = time.monotonic()
    first_epoch = last_complete + 1 if resume_from else 1
    global_step = last_complete * updates_per_epoch if resume_from else 0
    train_mse_series, lr_series = _epoch_series_from_metrics(dest) if resume_from else ({}, {})
    projection_reported = (dest / "budget_projection.json").exists()
    resume_note = None
    if resume_from:
        resume_note = {
            "resumed_from_epoch": last_complete,
            "first_remaining_epoch": first_epoch,
            "reason": "host reboot mid-epoch; leftover epochs continue from last sealed epoch ckpt",
            "in_progress_epoch_discarded": last_complete + 1 if last_complete < EPOCHS else None,
        }

    for epoch in range(first_epoch, EPOCHS + 1):
        model.train()
        epoch_t0 = time.monotonic()
        running = 0.0
        n_batches = 0
        for batch_id, (x, y, sessions) in enumerate(loader):
            if time.time() >= deadline_unix:
                _seal(
                    dest / "budget_hit.json",
                    {
                        "schema": "btransform_unified_v1_m1_projadd_depth2_budget_hit",
                        "cell": cell_id(depth),
                        "epoch": epoch,
                        "global_step": global_step,
                        "budget_seconds": OVERALL_BUDGET_SECONDS,
                        "utc": utc_iso(),
                        "note": "6h overall budget hit before finishing 24 epochs; no rerun authorized",
                    },
                )
                raise RuntimeError("M1 proj_add depth2 cell 6h overall budget hit")
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
                        "cell": cell_id(depth),
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
                        "cell": cell_id(depth),
                        "epoch": epoch,
                        "global_step": global_step,
                        "lr": float(lr),
                        "loss": float(loss.detach().cpu()),
                        "ema_updates": ema.n_updates,
                        "unix": time.time(),
                        "gpu_uuid": GPU_UUIDS[CUDA_PIN],
                    },
                )
            if not projection_reported and global_step == 50:
                per_step = (time.monotonic() - epoch_t0) / 50.0
                projected = per_step * total_updates
                projection = {
                    "schema": "btransform_unified_v1_m1_projadd_depth2_budget_projection",
                    "cell": cell_id(depth),
                    "measured_steps": 50,
                    "seconds_per_update": per_step,
                    "projected_total_seconds": projected,
                    "budget_seconds": OVERALL_BUDGET_SECONDS,
                    "projected_end_unix": time.time() + projected,
                    "deadline_unix": deadline_unix,
                    "within_budget": bool(time.time() + projected <= deadline_unix),
                    "utc": utc_iso(),
                }
                _seal(dest / "budget_projection.json", projection)
                if not projection["within_budget"]:
                    raise RuntimeError(
                        f"projected train end exceeds the 6h overall budget deadline; refusing to continue"
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
            "schema": "btransform_unified_v1_m1_projadd_depth2_ckpt",
            "cell": cell_id(depth),
            "epoch": epoch,
            "global_step": global_step,
            "seed": SEED,
            "temporal_layers": depth,
            "peak_lr": peak_lr,
            "sampler_batch_sha256": inventory["sampler_batch_sha256"],
            "raw_state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr": float(lr),
            "ema_decay": ema.decay,
        }
        ckpt_out = dest / f"epoch_{epoch:03d}.pt"
        if ckpt_out.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint {ckpt_out}")
        torch.save(ckpt, ckpt_out)
        _append_jsonl(metrics_path, {"event": "epoch", "epoch": epoch, "cell": cell_id(depth), **extras, "unix": time.time()})
        _write_json(heartbeat, {"event": "epoch", "cell": cell_id(depth), "epoch": epoch, **extras, "unix": time.time(), "gpu_uuid": GPU_UUIDS[CUDA_PIN]})

    summary = {
        "schema": "btransform_unified_v1_m1_projadd_depth2_train_receipt",
        "cell": cell_id(depth),
        "face": face,
        "temporal_layers": depth,
        "peak_lr": peak_lr,
        "seed": SEED,
        "gpu_uuid": GPU_UUIDS[CUDA_PIN],
        "sessions": bank_sessions,
        "heldout_sessions": [] if face == "fullsession" else [mp.M1_OUTER_SESSION],
        "windows_per_session": inventory["windows_per_session"],
        "total_windows": inventory["total_train_windows"],
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
        "resume": resume_note,
        "picks": picks,
        "note_six_rows": picks["note_six_rows"],
        "finished_utc": utc_iso(),
    }
    _seal(dest / "train_receipt.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Stage: score (GPU1; endpoint24 EMA; minival diagnostic + LOSO judgment)
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


def _score_source_minival(model, banks, device, face: str) -> dict[str, Any]:
    """31,252-window source-minival face (chron-80 dev tail)."""
    from torch.utils.data import DataLoader

    src_bank, source_dev = _source_dev_modules()
    loaded = src_bank.load()
    dm_src = source_dev.build_source_only_datamodule(loaded)
    _train_rows, dev_rows, rows_report = source_dev._split(dm_src)
    dev_inner = copy.copy(dm_src.train_dataset.base)
    dev_inner.window_indices = list(dev_rows)
    dev_ds = source_dev.SourceCarrierDataset(dev_inner, loaded)

    groups = {name: [] for name in TRAIN_SESSIONS}
    for index, (name, _start) in enumerate(dev_inner.window_indices):
        groups[name].append(index)
    batches = []
    for name in TRAIN_SESSIONS:
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
    report["role"] = (
        "DIAGNOSTIC face — POLLUTED (all 4 training sessions are stage-2 members; official score "
        "is the only selector-bearing number)"
        if face == "fullsession"
        else "DIAGNOSTIC face (chron-80 dev tail of the 3 TRAIN sessions; within-session temporal "
        "extrapolation — sessions seen in training, tail bins held out)"
    )
    report["split_rows"] = rows_report
    report["references"] = {
        "original_minival_pooled_exposed": mp.ORIGINAL_MINIVAL_POOLED,
        "family_flat_selected_pooled": 0.811652,
    }
    return report


def _score_loso_outer(model, outer_bank, device, face: str) -> dict[str, Any]:
    """LOSO ses-20120924 face, 26,496 windows — judgment (chron80) / polluted (fullsession)."""
    from torch.utils.data import DataLoader

    dm = mp.build_loso_datamodule()
    val = dm.val_heldin_dataset
    if val is None:
        raise RuntimeError("LOSO val_heldin_dataset missing")
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
    if len(target) != LOSO_WINDOWS:
        raise RuntimeError(f"LOSO cardinality drift: {len(target)} != {LOSO_WINDOWS}")
    report = session_mean_report(
        target, pred, np.repeat(np.asarray([mp.M1_OUTER_SESSION] * len(target)), target.shape[1])
    )
    report["role"] = (
        "DIAGNOSTIC ONLY (ses-20120924 is IN the stage-2 training set; a LOSO number on a "
        "retrained model is meaningless per ADDENDUM-SUBMISSION-PROTOCOL; official score is the "
        "only selector-bearing number)"
        if face == "fullsession"
        else "JUDGMENT face (true holdout for the 3-session training face: ses-20120924 excluded "
        "from training; banks source-frozen — B3 Sfix e11 encoder + sealed rSyn3 basis/normalizer; "
        "the outer session contributed only 10 unlabeled calib trials + query windows at score time)"
    )
    report["references"] = {
        "original_loso_outer_pooled_leakage_disclosed": mp.ORIGINAL_LOSO_OUTER_POOLED,
        "original_leakage_note": mp.ORIGINAL_LOSO_LEAKAGE_NOTE,
        "queryage_family_loso_pooled": 0.6579612934315439,
        "family_flat_path": str(mp.FAMILY_FLAT_PATH),
    }
    return report


def run_score(root: Path, dest: Path, depth: int, face: str = "chron80") -> dict[str, Any]:
    device = torch.device("cuda:0")
    torch.set_num_threads(4)

    inventory = _load_inventory(root)
    if face == "chron80":
        banks, _report = build_source_banks(calib_from_source_dm())
        bank_sessions = list(TRAIN_SESSIONS)
        outer_bank, outer_meta = build_outer_bank()
    elif face == "fullsession":
        _ds, _sampler = build_fullsession_face()
        banks, _report = build_fullsession_banks(_ds)
        bank_sessions = list(mp.M1_SESSIONS)
        outer_bank = banks[mp.M1_OUTER_SESSION]
        outer_meta = {
            "e0_sha256": outer_bank.calibration_meta["array_sha256"],
            "carrier_sha256": outer_bank.calibration_meta["carrier_sha256"],
            "carrier_meta": _report.get("outer_carrier_meta", {}),
            "note": "outer bank from the full-session bank build (identical mechanism)",
        }
    else:
        raise plan.BTransformerUnifiedError(f"unknown face {face!r}")
    for name in bank_sessions:
        sealed = inventory["banks"][name]
        if banks[name].calibration_meta["array_sha256"] != sealed["e0_sha256"]:
            raise RuntimeError(f"bank E0 digest drift vs sealed probe inventory: {name}")
        if banks[name].calibration_meta["carrier_sha256"] != sealed["carrier_sha256"]:
            raise RuntimeError(f"bank carrier digest drift vs sealed probe inventory: {name}")

    model = _build_model(depth).to(device)
    score_calib = None
    if face == "fullsession":
        score_calib = mp.calib_trials_from_dataset(_ds)
    elif face == "chron80":
        score_calib = calib_from_source_dm()
    model = prepare_trained_model(
        model,
        banks=banks,
        device=device,
        face=face,
        stage="score",
        calib_by_session=score_calib,
    )
    view = _apply_endpoint24_ema(model, dest)

    minival = _score_source_minival(model, banks, device, face)
    loso = _score_loso_outer(model, outer_bank, device, face)

    minival_pooled = float(minival["pooled_r2"])
    loso_pooled = float(loso["pooled_r2"])
    score = {
        "schema": "btransform_unified_v1_m1_projadd_depth2_score_receipt",
        "cell": cell_id(depth),
        "face": face,
        "route": "ADDENDUM-DEPTH2-PROMOTED (user directive 2026-09-07)",
        "view": view,
        "temporal_layers": depth,
        "faces": {
            "source_minival_31252": minival,
            "loso_outer_20120924_26496": loso,
        },
        "outer_bank": outer_meta,
        "gate_context": {
            "rule": (
                "depth-2 vs matched depth-4 baseline equal-session Delta R^2 >= -0.01 on the LOSO "
                "face (primary judgment; minival secondary); evaluated by --stage gate after both "
                "cells are scored"
            ),
            "gate_delta_r2": GATE_DELTA_R2,
        },
        "picks": _picks_with_runtime(inventory["updates_per_epoch"], inventory.get("peak_lr", PEAK_LR), depth),
        "utc": utc_iso(),
    }
    _seal(dest / "score_receipt.json", score)
    print(
        f"[depth2] score({depth}): minival pooled={minival_pooled:.4f} loso pooled={loso_pooled:.4f}",
        flush=True,
    )
    return score


# ---------------------------------------------------------------------------
# Stage: gate (paired Delta R^2 table + preregistered judgment)
# ---------------------------------------------------------------------------


def run_gate(root: Path) -> dict[str, Any]:
    from btransform_unified_v1.receipts import read_sealed

    scores = {}
    for depth in DEPTH_CELLS:
        path = root / DEPTH_DIR[depth] / "score_receipt.json"
        scores[depth] = read_sealed(path)

    def _face(depth: int, key: str) -> dict[str, Any]:
        return scores[depth]["faces"][key]

    deltas: dict[str, Any] = {}
    for key, label in (
        ("loso_outer_20120924_26496", "loso_judgment"),
        ("source_minival_31252", "minival_diagnostic"),
    ):
        f2, f4 = _face(2, key), _face(4, key)
        deltas[label] = {
            "depth2_pooled_r2": float(f2["pooled_r2"]),
            "depth4_pooled_r2": float(f4["pooled_r2"]),
            "delta_pooled_r2": float(f2["pooled_r2"]) - float(f4["pooled_r2"]),
            "depth2_session_mean_r2": float(f2["session_mean_r2"]),
            "depth4_session_mean_r2": float(f4["session_mean_r2"]),
            "delta_session_mean_r2": float(f2["session_mean_r2"]) - float(f4["session_mean_r2"]),
            "depth2_per_session_r2": f2["per_session_r2"],
            "depth4_per_session_r2": f4["per_session_r2"],
        }

    primary = deltas["loso_judgment"]["delta_pooled_r2"]
    secondary = deltas["minival_diagnostic"]["delta_pooled_r2"]
    pass_strict = bool(primary >= GATE_DELTA_R2)
    pass_adjusted = bool(primary >= GATE_DELTA_R2_ADJUSTED)
    inventory2 = _load_inventory(root)
    verdict = {
        "schema": "btransform_unified_v1_m1_projadd_depth2_paired_gate",
        "route": "ADDENDUM-DEPTH2-PROMOTED (user directive 2026-09-07); §7 折中门 preregistered",
        "utc": utc_iso(),
        "cells": {
            str(depth): {
                "cell_id": cell_id(depth),
                "score_receipt": str(root / DEPTH_DIR[depth] / "score_receipt.json"),
                "endpoint24_ema": scores[depth]["view"],
                "temporal_layers": depth,
            }
            for depth in DEPTH_CELLS
        },
        "paired_delta_table": deltas,
        "gate": {
            "rule": "depth2 - depth4 pooled R^2 on the LOSO ses-20120924 face (primary); minival secondary",
            "gate_delta_r2_preregistered": GATE_DELTA_R2,
            "gate_delta_r2_operative": GATE_DELTA_R2_ADJUSTED,
            "gate_revision": "user-adjusted threshold 2026-09-07 (-0.01 -> -0.03)",
            "primary_loso_delta": primary,
            "secondary_minival_delta": secondary,
            "passed_strict_-0.01": pass_strict,
            "passed_adjusted_-0.03": pass_adjusted,
            "passed": pass_adjusted,
            "verdict": (
                "PASS — depth-2 quality within the user-adjusted -0.03 budget on the judgment "
                "face (runtime halving potential accepted; proceed to the full-session build + "
                "ext6-style epoch pick + packaging plan per pre-authorized directive)"
                if pass_adjusted
                else "FAIL — depth cost exceeds even the user-adjusted -0.03 budget on the "
                "judgment face; full-session/pick/packaging NOT started; return to the 4-layer "
                "route (§7 W=50 fallback is a separate application)"
            ),
        },
        "references": {
            "original_loso_outer_pooled_leakage_disclosed": mp.ORIGINAL_LOSO_OUTER_POOLED,
            "queryage_family_loso_pooled": 0.6579612934315439,
            "original_minival_pooled_exposed": mp.ORIGINAL_MINIVAL_POOLED,
            "note": (
                "references are same-face different-system rows (P0-1); the gate itself is the "
                "depth-2 vs depth-4 PAIRED comparison, never a reference subtraction"
            ),
        },
        "picks": _picks_with_runtime(int(inventory2["updates_per_epoch"]), PEAK_LR, 2),
    }
    _seal(root / "paired_gate.json", verdict)
    print(
        f"[depth2] gate: LOSO delta={primary:+.4f} (strict {GATE_DELTA_R2}: "
        f"{'PASS' if pass_strict else 'FAIL'}; adjusted {GATE_DELTA_R2_ADJUSTED}: "
        f"{'PASS' if pass_adjusted else 'FAIL'}) minival delta={secondary:+.4f}",
        flush=True,
    )
    return verdict


# ---------------------------------------------------------------------------
# Stage: pick (pre-authorized; full-session build only) — M1 ext6 analog
# ---------------------------------------------------------------------------


def make_pick_bank(session: str, e0: np.ndarray, carrier: np.ndarray) -> TaskBank:
    """TaskBank for a held-out-calib pick session (direct build; M1 geometry).

    ``mp.make_m1_bank`` gates on the four local session names, so the pick
    sessions build the TaskBank directly under the identical geometry/meta
    contract (E0 [64,100], carrier [64,4], all-true mask, M10 budget).
    """
    e0c = np.ascontiguousarray(e0, dtype=np.float32)
    t4 = np.ascontiguousarray(carrier, dtype=np.float32)
    plan.require(e0c.shape == (mp.M1_UNITS, mp.M1_E0_DIM), f"pick E0 shape {e0c.shape}")
    plan.require(t4.shape == (mp.M1_UNITS, mp.M1_CARRIER_DIM), f"pick carrier shape {t4.shape}")
    return TaskBank(
        session_id=f"ses-{session}",
        E0=e0c,
        carrier=t4,
        unit_mask=np.ones(mp.M1_UNITS, dtype=np.bool_),
        X_store=np.zeros((0, mp.M1_WINDOW, mp.M1_UNITS), dtype=np.float32),
        target_store=np.zeros((0, mp.M1_OUT_DIM), dtype=np.float32),
        window_ids=np.zeros(0, dtype=np.int64),
        calibration_meta={
            "shape": tuple(e0c.shape),
            "trial_count": mp.M10_BUDGET,
            "estimator": (
                "B3 Sfix e11 student.id_encoder compute_identity(side_features=None) + rSyn3 "
                "carrier (sealed basis + source normalizer; b3_allsource public-calib reader)"
            ),
            "array_sha256": array_sha256(e0c),
            "budget": mp.M10_BUDGET,
            "surface": "m1-projadd-depth2-pick (official held-out-calib, locally visible)",
            "session": f"ses-{session}",
            "carrier_sha256": array_sha256(t4),
            "carrier_source": "sealed-basis-encode (public calib support)",
            "unit_mask_all_true": True,
            "store_note": "window store empty; pick windows served by the direct FalconDataset",
        },
    )


def run_pick(root: Path, dest: Path, depth: int, epoch_lo: int, epoch_hi: int) -> dict[str, Any]:
    """ext6-style epoch scan on the three LOCALLY VISIBLE official
    held-out-calib sessions (20121004/17/24), t0c1 direct-reader mechanism.

    Pick rule (M2 581973 / ext6, locked before reading numbers): highest
    equal-session mean, then highest worst-session R^2, then earliest epoch.
    DISCLOSURE: this is a dev-on-official-selected caliber — the surface is
    the locally visible official held-out-calib trio, NOT the hidden test;
    the official score remains the only selector-bearing readout.
    """
    from torch.utils.data import DataLoader

    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    plan.require(1 <= epoch_lo <= epoch_hi <= EPOCHS, f"bad epoch range {epoch_lo}:{epoch_hi}")

    provider = mp.default_identity_provider()
    sessions: dict[str, dict[str, Any]] = {}
    banks: dict[str, TaskBank] = {}
    pick_calib: dict[str, Any] = {}
    for session in HELDOUT_CALIB_SESSIONS:
        opened = open_heldout_calib_session(session)
        carrier, carrier_meta = encode_heldout_calib_carrier(session)
        e0 = provider(opened["calib10"])
        bank = make_pick_bank(session, e0, carrier)
        banks[session] = bank
        pick_calib[session] = opened["calib10"]
        pick_calib[bank.session_id] = opened["calib10"]
        opened.pop("calib10")
        opened["bank"] = {
            "e0_sha256": bank.calibration_meta["array_sha256"],
            "carrier_sha256": bank.calibration_meta["carrier_sha256"],
            "carrier_meta": carrier_meta,
            "body_sha256": opened["body_sha256"],
        }
        sessions[session] = opened
    model = _build_model(depth).to(device)
    model = prepare_trained_model(
        model,
        banks=banks,
        device=device,
        face="pick",
        stage="pick",
        calib_by_session=pick_calib,
    )
    rows: list[dict[str, Any]] = []
    for epoch in range(epoch_lo, epoch_hi + 1):
        ckpt_path = dest / f"epoch_{epoch:03d}.pt"
        view = _apply_endpoint24_ema(model, dest) if epoch == EPOCHS else _apply_ema_epoch(model, ckpt_path)
        per_session: dict[str, float] = {}
        counts: dict[str, int] = {}
        model.eval()
        with torch.inference_mode():
            for session, opened in sessions.items():
                dataset = opened["dataset"]
                ids = list(range(len(dataset.window_indices)))
                preds, targets = [], []
                for left in range(0, len(ids), PICK_BATCH):
                    neural, target, _calib, _sess = next(
                        iter(DataLoader(dataset, batch_sampler=[ids[left : left + PICK_BATCH]]))
                    )[:4]
                    out = model(neural.float().to(device), banks[session])
                    preds.append(out.cpu().numpy().astype(np.float32))
                    targets.append(target[:, -1, :].numpy().astype(np.float32))
                pred = np.concatenate(preds)
                targ = np.concatenate(targets)
                per_session[session] = float(
                    1.0
                    - np.square(pred - targ).sum() / np.square(targ - targ.mean(0, keepdims=True)).sum()
                )
                counts[session] = int(targ.shape[0])
        mean = float(np.mean([per_session[s] for s in HELDOUT_CALIB_SESSIONS]))
        worst = min(HELDOUT_CALIB_SESSIONS, key=lambda s: per_session[s])
        row = {
            "epoch": epoch,
            "view": "EMA",
            "ckpt": str(ckpt_path),
            "ckpt_sha256": _sha_file(ckpt_path),
            "equal_session_mean": mean,
            "per_session_r2": {s: per_session[s] for s in HELDOUT_CALIB_SESSIONS},
            "worst_session": worst,
            "worst_session_r2": per_session[worst],
            "window_count": counts,
        }
        rows.append(row)
        _write_json(dest / "pick_partial.json", {"latest": row})
        print(
            f"[depth2] pick e{epoch:02d} eq={mean:.4f} worst={worst} {per_session[worst]:.4f}",
            flush=True,
        )

    def key(row: dict[str, Any]) -> tuple[float, float, int]:
        return (float(row["equal_session_mean"]), float(row["worst_session_r2"]), -int(row["epoch"]))

    selected = max(rows, key=key)
    tied = [
        r
        for r in rows
        if abs(r["equal_session_mean"] - selected["equal_session_mean"]) <= 1e-10
        and abs(r["worst_session_r2"] - selected["worst_session_r2"]) <= 1e-10
    ]
    selected = min(tied, key=lambda r: int(r["epoch"]))
    tail_note = (
        "endpoint24 is the scan max (no late-epoch rise)"
        if selected["epoch"] == epoch_hi
        else f"curve rises into the tail: picked e{selected['epoch']:02d} over endpoint24"
    )
    payload = {
        "schema": "btransform_unified_v1_m1_projadd_depth2_heldoutcalib_epoch_pick",
        "cell": cell_id(depth),
        "face": "fullsession pick (pre-authorized user directive 2026-09-07)",
        "utc": utc_iso(),
        "selection_surface": (
            "three locally visible official held-out-calib sessions (20121004/20121017/20121024), "
            "query_start_trial=0 full local query; t0c1 direct-reader mechanism, body-SHA-gated, "
            "evaluation-only access law"
        ),
        "selection_rule": "highest equal-session mean, then highest worst-session R2, then earliest epoch (M2 581973 / ext6 rule)",
        "disclosure": (
            "dev-on-official-selected caliber: epoch choice is made on the locally visible "
            "official held-out-calib trio, NOT the hidden test; the official score remains the "
            "only selector-bearing readout and no EvalAI submission is made here"
        ),
        "epoch_range": [epoch_lo, epoch_hi],
        "sessions": {
            s: {
                "body_sha256": sessions[s]["body_sha256"],
                "n_windows": sessions[s]["n_windows"],
                "bank": sessions[s]["bank"],
            }
            for s in HELDOUT_CALIB_SESSIONS
        },
        "curve": rows,
        "selected": selected,
        "tail_note": tail_note,
        "evalai_opened": False,
        "hidden_or_test_opened": False,
        "register": False,
    }
    _seal(dest / "epoch_pick.json", payload)
    print(f"[depth2] pick selected e{selected['epoch']:02d} eq={selected['equal_session_mean']:.4f}", flush=True)
    return payload


def _apply_ema_epoch(model: BTransformerUnifiedDecoderIdentity, ckpt_path: Path) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["raw_state_dict"], strict=True)
    shadow = ckpt["ema"]["shadow"]
    named = model.trainable_parameters()
    if set(named) != set(shadow):
        raise RuntimeError("EMA/RAW key mismatch")
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(shadow[name].to(device=param.device, dtype=param.dtype))
    model.eval()
    return {"epoch": int(ckpt["epoch"]), "ema_updates": int(ckpt["ema"]["n_updates"]), "global_step": int(ckpt["global_step"])}


# ---------------------------------------------------------------------------
# Stage: packplan (pre-authorized; packaging PLAN only, register:false)
# ---------------------------------------------------------------------------


def run_packplan(root: Path, dest: Path, depth: int, timing_receipt: Path | None = None) -> dict[str, Any]:
    """Packaging plan receipt for the full-session depth-2 candidate.

    Backend decision rule: use the runtime line's ORT path only if its FP32
    equivalence gate is closed all-pass; otherwise the exact-E optimized
    torch path. Runtime evidence cites this cell's sealed temporal_timing
    receipt (depth-2 temporal core ~0.54x the 4-layer cost) plus the runtime
    line's receipts (read-only). No packaging is executed here; no EvalAI
    contact; register:false.
    """
    pick = json.loads((dest / "epoch_pick.json").read_text(encoding="utf-8"))
    selected = pick["selected"]
    ckpt_path = Path(selected["ckpt"])
    equivalence_path = RUNTIME_LINE_ROOT / "results/equivalence_gate_intra4.json"
    ort_all_pass = None
    if equivalence_path.is_file():
        try:
            gate = json.loads(equivalence_path.read_text(encoding="utf-8"))
            r1 = gate.get("r1_ema_oracle", {})
            r2 = gate.get("r2_frozen_adapter", {})
            ort_all_pass = bool(r1.get("all_pass")) and bool(r2.get("all_pass"))
        except Exception:
            ort_all_pass = None
    backend = (
        "ORT CPUExecutionProvider path (runtime line equivalence gate all-pass)"
        if ort_all_pass
        else "exact-E optimized torch path (ORT equivalence gate not closed all-pass at plan time; "
        "re-evaluate when the runtime line closes it)"
    )
    plan_payload = {
        "schema": "btransform_unified_v1_m1_projadd_depth2_packplan",
        "cell": cell_id(depth),
        "face": "fullsession (stage-2 framing)",
        "utc": utc_iso(),
        "candidate_payload": {
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": selected["ckpt_sha256"],
            "view": "EMA",
            "epoch": selected["epoch"],
            "pick_rule": pick["selection_rule"],
            "pick_disclosure": pick["disclosure"],
        },
        "backend_decision_rule": (
            "if the runtime line's ORT candidate closes its FP32 equivalence gate all-pass "
            "(r1 EMA oracle AND r2 frozen adapter), package via the ORT path; else the exact-E "
            "optimized torch path (m1_exacte_fast precedent)"
        ),
        "backend_selected": backend,
        "ort_equivalence_all_pass_at_plan_time": ort_all_pass,
        "runtime_evidence": {
            "depth2_temporal_timing": str(timing_receipt) if timing_receipt is not None else None,
            "depth2_temporal_note": (
                "pure-CPU temporal core (untrained seed-42 models, B1/B4, threads 1/4, "
                "interleaved reps): B4/threads1 median ratio d2/d4 ~= 0.54 — the structural "
                "~2x order §7 asked to confirm; sealed in the paired chron80 cell root"
            ),
            "runtime_line_root": str(RUNTIME_LINE_ROOT),
            "runtime_line_equivalence": str(equivalence_path),
            "runtime_line_budget": str(RUNTIME_LINE_ROOT / "results/runtime_budget.json"),
            "budget_verdict_quote": (
                "runtime line verdict at plan time: baseline RUNTIME_FAIL — the depth-2 structural "
                "cut is this cell's contribution toward the 5400s engineering budget; final replay "
                "must be redone on the depth-2 full-session candidate before any submission"
            ),
        },
        "checklist_before_submission": [
            "re-run P0/P3 protocol replay on the depth-2 full-session candidate (call inventory, "
            "B1-B4, budget formula) — the depth-2 timing evidence here is the temporal core only",
            "banks/tags identical to the seven public calibration tags (CAL-2 contract)",
            "pack into a NEW directory/tag (never overwrite 582019 artifacts)",
            "manifest register:false; EvalAI submission executed by the USER, never auto",
        ],
        "register": False,
        "evalai_opened": False,
        "submitted": False,
    }
    _seal(dest / "pack_plan.json", plan_payload)
    print(f"[depth2] packplan: backend={backend[:60]}... payload={ckpt_path.name}", flush=True)
    return plan_payload


# ---------------------------------------------------------------------------


def main() -> int:
    global CUDA_PIN
    parser = argparse.ArgumentParser(description="M1 proj_add temporal depth 4->2 front-row cell")
    parser.add_argument("--depth", type=int, choices=list(DEPTH_CELLS), default=2)
    parser.add_argument("--cuda-visible", default=CUDA_PIN, choices=["0", "1"])
    parser.add_argument("--face", choices=list(FACES), default="chron80")
    parser.add_argument("--pick-epochs", default=f"{PICK_EPOCH_RANGE[0]}:{PICK_EPOCH_RANGE[1]}", help="lo:hi inclusive")
    parser.add_argument("--timing-receipt", default=None, help="path to the sealed temporal_timing.json (packplan cite)")
    parser.add_argument("--stage", choices=list(STAGES), default="all")
    parser.add_argument("--peak-lr", type=float, default=PEAK_LR)
    parser.add_argument("--root", type=Path, default=None, help="run root (default: results/m1_projadd_depth2/<face>_<UTC>/)")
    parser.add_argument("--dest", type=Path, default=None, help="override the cell dir (default: <root>/<depth2|depth4>)")
    args = parser.parse_args()
    CUDA_PIN = str(args.cuda_visible)
    if args.peak_lr <= 0 or args.peak_lr != PEAK_LR:
        print(f"REFUSED: --peak-lr is frozen at {PEAK_LR:g} for this cell", file=sys.stderr)
        return 2

    root = args.root if args.root else plan.RESULT_ROOT / "m1_projadd_depth2" / f"{args.face}_{utc_stamp()}"
    dest = args.dest if args.dest else root / DEPTH_DIR[args.depth]
    if args.stage in ("probe", "timing"):
        dest = root  # face-level artifacts live at the root
    root.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    print(
        f"[depth2] cell={cell_id(args.depth)} stage={args.stage} face={args.face} root={root} dest={dest}",
        flush=True,
    )

    gpu_stage = args.stage in ("train", "score", "pick", "all")
    if args.stage == "train" and os.environ.get(TRAIN_ENV_FLAG) != "1":
        print(f"REFUSED: training requires {TRAIN_ENV_FLAG}=1", file=sys.stderr)
        return 2
    if gpu_stage and os.environ.get("CUDA_VISIBLE_DEVICES") != CUDA_PIN:
        print(f"REFUSED: CUDA_VISIBLE_DEVICES must be pinned to {CUDA_PIN!r}", file=sys.stderr)
        return 2

    if gpu_stage or args.stage == "preflight":
        pre = gpu_preflight(dest / f"preflight_gpu_{args.stage}.json", args.depth, args.stage)
        print(f"[depth2] preflight({args.stage}) ok={pre['ok']} foreign={pre['foreign_pids_on_target']}", flush=True)
        if gpu_stage and not pre["ok"]:
            _seal(
                dest / "BLOCKED.json",
                {
                    "schema": "btransform_unified_v1_m1_projadd_depth2_blocked",
                    "cell": cell_id(args.depth),
                    "reason": f"GPU{CUDA_PIN} not clean at preflight (foreign pid >500 MiB or CUDA pin mismatch)",
                    "preflight": pre,
                    "utc": utc_iso(),
                },
            )
            print(f"[depth2] BLOCKED: GPU{CUDA_PIN} not clean; refusing to start", file=sys.stderr)
            return 2

    if args.stage == "preflight":
        return 0
    if args.stage in ("probe", "all"):
        run_probe(root, root, args.face)
    if args.stage in ("timing", "all"):
        run_timing(root)
    if args.stage in ("train", "all"):
        summary = run_train(root, dest, args.depth, args.peak_lr, args.face)
        print(
            f"[depth2] train({args.depth}/{args.face}) done in {summary['elapsed_s']:.0f}s; "
            f"updates={summary['global_updates']}; "
            f"train_mse e1/e24={summary['train_mse'][1]:.6g}/{summary['train_mse'][24]:.6g}",
            flush=True,
        )
    if args.stage in ("score", "all"):
        run_score(root, dest, args.depth, args.face)
    if args.stage == "gate":
        run_gate(root)
    if args.stage == "pick":
        lo_text, hi_text = args.pick_epochs.split(":")
        run_pick(root, dest, args.depth, int(lo_text), int(hi_text))
    if args.stage == "packplan":
        timing = Path(args.timing_receipt) if args.timing_receipt else None
        run_packplan(root, dest, args.depth, timing_receipt=timing)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        dest = None
        for arg_idx, a in enumerate(sys.argv):
            if a in ("--dest", "--root") and arg_idx + 1 < len(sys.argv):
                dest = Path(sys.argv[arg_idx + 1])
        trace = traceback.format_exc()
        print(trace, file=sys.stderr)
        try:
            from btransform_unified_v1 import plan as _plan, receipts as _receipts

            target = dest or (_plan.RESULT_ROOT / "m1_projadd_depth2" / "error")
            target.mkdir(parents=True, exist_ok=True)
            _receipts.seal_json(
                target / "error_receipt.json",
                {
                    "schema": "btransform_unified_v1_m1_projadd_depth2_error",
                    "utc": utc_iso(),
                    "traceback": trace,
                },
            )
        except Exception:
            pass
        sys.exit(3)
