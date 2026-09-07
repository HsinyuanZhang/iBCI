"""M1 FULL-SESSION SUBMISSION BUILD — btransform_unified_v1 (stage-2 protocol).

ADDENDUM-REPRIORITIZE / ADDENDUM-SUBMISSION-PROTOCOL (workorder
WORKORDER_BTRANSFORM_UNIFIED_V1_20260906.md, user ruling 2026-09-06):
second stage of the two-stage protocol. The recipe is frozen; this cell
RETRAINS on ALL locally available M1 held-in sessions — including
ses-20120924, whose presence in the training set is a LEGAL part of this
protocol (it is the stage-2 product by definition) — and produces a sealed
endpoint24-EMA checkpoint plus a handoff contract for the user's submission
agent (EvalAI packaging). This cell performs NO EvalAI action, NO method
selection, and NO LOSO surface read (after full-data retraining a LOSO number
is meaningless; the official score is the only readout).

Composition (all lineage defaults; no unverified options mixed in):
  - data: 4 held-in sessions (ses-20120924/26/27/28), each contributing its
    full-timeline eligible windows under the fold-local train-split law
    (query_start_trial=0, query_end=None, eval_mask at the window's last bin,
    99-bin zero pre-history pad). Mechanism reused read-only from
    tfpd_exploration/src/m1_emg_rsyn3_fold_local_v1 +
    streaming_calibration_exp M1VersionBSourceLOSODataModule (fold 0) and
    btransform_unified_v1/scripts/m1_family_loso_outer20120924.py (another
    agent's script; its data/coordinate mechanism, never modified).
  - banks (CAL-2, M10, per-session frozen): identity E0 [64,100] = B3 Sfix
    e11 ``student.id_encoder`` via compute_identity(side_features=None)
    semantics (EarlyPoolEncoder strict-load, push 10 chronological calib
    trials, finalize; NO decoder weights, NOTE P1-10); carrier [64,4] =
    rSyn3 (sealed rSyn3-refit-v1 source-only NPZ for ses-20120926/27/28;
    ses-20120924 encoded with the SAME sealed basis + source normalizer —
    the m1_family_loso_outer20120924 mechanism).
  - model: BTransformerUnifiedDecoder, m1 geometry with P=0 (L_in=100=W,
    N=64, out=16, d_e=100 concat identity, carrier 4), seed 42.
  - recipe = TRN-1 (the exact set M2 route-B verified bitwise against S1):
    AdamW wd 0.01 clip 1, warmup 1 epoch to 3e-4 then cosine to 3e-5,
    24 epochs, EMA 0.9995, whole-unit dropout p=0.10 with the S1-aligned
    unit_dropout_seed(42, epoch, batch_id) domain, batch 32, bf16 autocast
    around forward+loss with pred.float() before MSE. divisor = 1 (native).
  - checkpoint rule = endpoint24 EMA (preregistered, the ONLY rule; no
    epoch pick on any surface).

Discipline: GPU1 only (UUID GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86),
CUDA_VISIBLE_DEVICES pinned to '1'; preflight refuses when a foreign pid
holds >500 MiB on GPU1 (BLOCKED receipt); GPU0 is never touched. 6h GPU
budget from train start (BUDGET_HIT receipt on overrun). Receipts sealed via
btransform_unified_v1.receipts.seal_json (0444 + sha256 sidecar). Historical
roots untouched; the five-arm/LOSO scripts are read-only.

Usage:
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 \
    BTRANSFORM_M1_FULLSESSION_TRAIN=1 \
    /home/xinyuan/miniconda3/envs/spint/bin/python \
    btransform_unified_v1/scripts/m1_fullsession_submission_v1.py --stage all

Identity: B-transformer unified series, NOT SPINT.
"""

from __future__ import annotations

import argparse
import hashlib
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

from btransform_unified_v1 import adapters, plan, receipts  # noqa: E402
from btransform_unified_v1.bank import TaskBank, array_sha256  # noqa: E402
from btransform_unified_v1.ema import DecoderEMA  # noqa: E402
from btransform_unified_v1.model import (  # noqa: E402
    UNIT_DROPOUT_DOMAIN_META,
    BTransformerUnifiedDecoder,
    unit_dropout_seed,
    whole_unit_dropout,
)
from btransform_unified_v1.r2 import session_mean_report, variance_weighted_r2  # noqa: E402
from btransform_unified_v1.schedule import warmup_cosine_lr  # noqa: E402

from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import plan as fold_plan  # noqa: E402
from tfpd_exploration.src.m1_optimized_v2 import bank as src_bank  # noqa: E402
from tfpd_exploration.src.m1_optimized_v2 import plan as m1_plan  # noqa: E402
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_config import (  # noqa: E402
    M1_TEMPORAL,
    S_FIX_PATH,
    S_FIX_SHA256,
)

CELL = "M1-FULLSESSION-SUBMISSION-V1"
TRAIN_ENV_FLAG = "BTRANSFORM_M1_FULLSESSION_TRAIN"
SEED = 42
GPU1_UUID = "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"
FOREIGN_MEM_MIB_LIMIT = 500
BUDGET_SECONDS = 6.0 * 3600.0
SESSIONS = tuple(fold_plan.SESSIONS)  # (20120924, 20120926, 20120927, 20120928)
OUTER = fold_plan.FOLD0_TARGET_SESSION  # ses-20120924
SOURCES = tuple(m1_plan.SOURCE_SESSIONS)
M10 = int(fold_plan.SUPPORT_TRIALS)
GEOMETRY = {
    "task": "m1",
    "window": 100,
    "prefix": 0,  # L_in = W = 100 (TRN-5 P=0; family/QA lineage consumes the bare W window)
    "units": 64,
    "e0_dim": 100,
    "carrier_dim": 4,
    "out_dim": 16,
    "target_scale": 1.0,  # divisor = 1 (TRN-8, M1 never divides)
}
RUNTIME_CACHE_NPZ = (
    WORKSPACE_ROOT
    / "tfpd_exploration/results/decoder_validation_v2/20260905_190000/m1/m1_optimized_v2_source_runtime_cache.npz"
)
FAMILY_RUN_META = (
    WORKSPACE_ROOT
    / "tfpd_exploration/results/decoder_validation_v2/20260905_190000/m1/family_v1/p1_queryage16_pair_chron80_v2/run_meta.json"
)
LOSO_SCRIPT_REFERENCE = PACKAGE_ROOT / "scripts/m1_family_loso_outer20120924.py"

PICKS = {
    "CAL": [
        "CAL-2 {budget: 10 (M10), surface: sealed rSyn3-refit-v1 source-only NPZ (read-only) for ses-20120926/27/28 + sealed-basis encode for ses-20120924 (m1_family_loso_outer20120924 mechanism)}",
        "CAL-3a {identity: B3 Sfix e11 student.id_encoder post_pool d_e=100 static concat; compute_identity(side_features=None); NO decoder weights (P1-10)}",
        "CAL-5 {unit order: units DataFrame row order; unit_mask all-true (64)}",
    ],
    "TRN": [
        "TRN-1 {updates_per_epoch: FILLED_AT_RUNTIME, warmup_updates: FILLED_AT_RUNTIME, total_updates: FILLED_AT_RUNTIME, peak_lr: 3e-4, min_lr: 3e-5, AdamW wd 0.01 betas (0.9,0.999) eps 1e-8, clip 1.0, batch 32, epochs 24}",
        "TRN-1 precision {bf16 autocast: forward+loss, pred.float() before MSE (S1 launch semantics)}",
        "TRN-3 {whole-unit dropout p=0.10, training mode only, per-batch CPU generator, domain: m2_small_unit_dropout (S1-aligned unit_dropout_seed; route-B verified)}",
        "TRN-5 {P: 0, L_in: 100}",
        "TRN-8 {scale: divisor=1; train and score native target}",
    ],
    "SEL": [
        "SEL-1 {endpoint24 EMA — preregistered, the ONLY rule for this submission build; no epoch pick on any surface}",
        "SEL-3 {official surface: zero participation; NO LOSO surface read after full-data retrain (ADDENDUM-SUBMISSION-PROTOCOL)}",
    ],
    "SPD": ["none {training-path run; SPD-A1 folding exists but is not used in training or scoring}"],
}

NOTE_SIX_ROWS = {
    "system": (
        "btransform_unified_v1 BTransformerUnifiedDecoder (m1 geometry, P=0, L_in=100=W, "
        "N=64, out=16 EMG, d_e=100 concat identity, seed 42; FILLED_PARAM_COUNT params) "
        "— NOT SPINT"
    ),
    "consumer": (
        "B-transformer unified decoder (8-slot + CausalPE4, this series). Consumers of "
        "frozen SPINT-lineage calibration objects (B3 Sfix e11 id_encoder + rSyn3). "
        "Comparisons against the SPINT family are 'same scoring surface, different "
        "system' only (P0-1)"
    ),
    "calibration_object": (
        "E0 [64,100] = B3 Sfix e11 student.id_encoder (sha 7976e0b0...) via "
        "compute_identity(side_features=None) on the 10 chronological calib trials; "
        "carrier [64,4] = rSyn3 (NNMF rank-3 NNDSVDa seed 42 + ridge lambda=1 "
        "intercept-unpenalized + source RMS scale + source carrier normalizer; sealed "
        "rSyn3-refit-v1 NPZ; ses-20120924 encoded with the sealed basis); M10 budget; "
        "per-session banks frozen (CAL-2)"
    ),
    "scoring_surface": (
        "stage-2 submission build: trained on ALL 4 local held-in sessions "
        "(20120924/26/27/28); official EvalAI score pending (user's submission agent "
        "packages it); LOSO surface FORBIDDEN after this retrain (ADDENDUM-"
        "SUBMISSION-PROTOCOL); train-universe R2 recorded as diagnostic only, never "
        "for selection"
    ),
    "scale": "divisor=1: train and score native target (M1 never divides by 20)",
    "single_difference_vs_historical_best": (
        "vs M1 family (3-session chron80 training, source-minival 0.812 / official "
        "QueryAge 0.574): the single differing row is the TRAINING SESSION SET — all "
        "4 held-in sessions incl. ses-20120924 (legal stage-2 product) vs family "
        "3-session; recipe/calibration/bank mechanism identical"
    ),
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _sha_file(path: Path) -> str:
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


def _streaming_path() -> None:
    p = str(WORKSPACE_ROOT / "streaming_calibration_exp")
    if p not in sys.path:
        sys.path.insert(0, p)


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
        "schema": "btransform_unified_v1_m1_fullsession_gpu_preflight",
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
# Data assembly (fold-local mechanism, read-only reuse)
# ---------------------------------------------------------------------------


def _loso_module():
    """Fold-0 LOSO datamodule exactly as m1_family_loso_outer20120924.py builds it."""
    _streaming_path()
    from src.data.m1_version_b_source_loso_datamodule import M1VersionBSourceLOSODataModule

    dm = M1VersionBSourceLOSODataModule(
        task="m1",
        data_dir=str(m1_plan.DATA_DIR),
        source_session_names=list(m1_plan.SOURCE_SESSIONS),
        heldin_session_names=list(m1_plan.SOURCE_SESSIONS),
        batch_size=32,
        window_size=m1_plan.WINDOW,
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
        validation_protocol="loso",
        loso_fold=0,
        include_heldout_in_fit=False,
        include_heldout_in_test=False,
        query_start_trial=0,
        heldin_query_start_trial=10,
        heldin_query_end_trial=210,
        allow_empty_heldout_query=False,
        num_workers=0,
        pin_memory=False,
        sampler_seed=m1_plan.SEED,
        balance_session_batches=False,
        reshuffle_train_sampler_each_epoch=False,
        afc4_arm="none",
    )
    dm.setup("test")
    if dm.outer_left_out != OUTER:
        raise RuntimeError("fold-0 outer session drift")
    return dm


def assemble_training_universe(dm) -> tuple[Any, Any, dict[str, Any]]:
    """Combined 4-session FalconDataset under the train-split law + batch sampler.

    The three source sessions enter exactly as the fold's train split does
    (full-timeline eligible windows); ses-20120924 enters under the SAME law
    (its record rebuilt with the datamodule's own prepare_session_data call).
    Under ADDENDUM-SUBMISSION-PROTOCOL its presence in training is legal
    (stage-2 product).
    """
    from src.data.falcon_datamodule import FalconDataset, SessionBatchSampler

    h = dm.hparams
    task = dm._falcon_task_m1()
    source_records = dm.train_calib_heldin_sessions
    cov_mean = source_records[SOURCES[0]]["covariates_mean"]
    cov_std = source_records[SOURCES[0]]["covariates_std"]
    outer_record = dm.prepare_session_data(
        dm.target_path,
        task,
        standardize_covariates=False,
        covariates_mean=cov_mean,
        covariates_std=cov_std,
        use_intertrials=True,
    )
    combined = {OUTER: outer_record}
    for name in SOURCES:
        combined[name] = source_records[name]
    records = {name: combined[name] for name in SESSIONS}  # fold-plan order
    dataset = FalconDataset(
        sessions_dict=records,
        calib_sessions_dict=records,
        window_size=h.window_size,
        split="train",
        calibration_n_trials=h.calibration_n_trials,
        random_calibration=False,
        smooth_calibration=h.smooth_calibration,
        max_trial_length=h.max_trial_length,
        use_calib_intertrials=h.use_calib_intertrials,
        trial_feature_type=h.trial_feature_type,
        remove_still_times=h.remove_still_times,
        remove_calib_still_times=h.remove_calib_still_times,
        use_calib_active_segments=h.use_calib_active_segments,
        calib_n_active_segments=h.calib_n_active_segments,
        interpolate_trials=h.interpolate_trials,
        interpolate_trials_kind=h.interpolate_trials_kind,
        pad_value=h.pad_value,
        query_start_trial=0,  # train law: full timeline per session
        query_end_trial=None,
        allow_empty_query_sessions=False,
    )
    sampler = SessionBatchSampler(
        dataset,
        32,
        shuffle=True,
        seed=SEED,
        balance_sessions=False,
        reshuffle_each_epoch=False,
    )
    audit = {}
    for name in SESSIONS:
        a = dict(dataset.query_window_audit[name])
        a.pop("ordered_window_start_sha256", None)
        a.pop("ordered_target_covariate_evalmask_sha256", None)
        a.pop("ordered_query_identity_sha256", None)
        audit[name] = a
    info = {
        "sessions": list(SESSIONS),
        "window_audit": audit,
        "windows_per_session": {n: int(audit[n]["eligible_windows"]) for n in SESSIONS},
        "total_windows": int(len(dataset.window_indices)),
        "updates_per_epoch": int(len(sampler)),
        "train_law": "FalconDataset split='train': query_start_trial=0, query_end_trial=None, eval_mask at last bin, pre_history=99 zero pad",
        "sampler": "SessionBatchSampler(batch=32, shuffle=True, seed=42, balance=False, reshuffle_each_epoch=False)",
        "sampler_batch_sha256": M1VersionBSourceLOSODataModuleSamplerDigest(sampler),
        "source_files": {
            **{n: {"path": str(dm.source_paths[n]), "sha256": _sha_file(dm.source_paths[n])} for n in SOURCES},
            OUTER: {"path": str(dm.target_path), "sha256": _sha_file(dm.target_path)},
        },
    }
    return dataset, sampler, info


def M1VersionBSourceLOSODataModuleSamplerDigest(sampler) -> str:
    """Hash the frozen batch order (same digest law as the datamodule's static helper)."""
    digest = hashlib.sha256()
    for batch_index, batch in enumerate(sampler.batched_indices):
        indices = np.asarray(batch, dtype=np.int64).reshape(-1)
        digest.update(np.asarray([batch_index, indices.size], dtype=np.int64).tobytes())
        digest.update(indices.tobytes())
    return digest.hexdigest()


def _collate(items: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]):
    """Light collate: drop the per-item calib tensor (banks are prebuilt)."""
    neural = torch.from_numpy(
        np.ascontiguousarray(np.stack([np.asarray(it[0], dtype=np.float32) for it in items]), dtype=np.float32)
    )
    target = torch.from_numpy(
        np.ascontiguousarray(np.stack([np.asarray(it[1], dtype=np.float32)[-1] for it in items]), dtype=np.float32)
    )
    sessions = [str(it[3]) for it in items]
    return neural, target, sessions


# ---------------------------------------------------------------------------
# Banks (CAL-2 M10: B3 Sfix e11 identity + rSyn3 sealed carrier)
# ---------------------------------------------------------------------------


def _load_b3_id_encoder() -> nn.Module:
    """Strict-load student.id_encoder from Sfix e11 (no Lightning teacher needed).

    Same mechanism as m1_family_loso_outer20120924.py::_family_bank; canonical
    equivalent of m1_optimized_v2/calibration.py::load_frozen_b3 (whose
    Lightning teacher checkpoint is absent in this workspace). Only
    ``student.id_encoder`` keys are consumed — no B3/Sfix decoder weight ever
    enters the new network (NOTE P1-10).
    """
    _streaming_path()
    from src.models.components.streaming_encoders import EarlyPoolEncoder

    if _sha_file(S_FIX_PATH) != S_FIX_SHA256:
        raise RuntimeError("frozen B3 Sfix e11 checkpoint checksum drift")
    payload = torch.load(S_FIX_PATH, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    prefix = "student.id_encoder."
    incoming = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not any(k.startswith("student.decoder.") for k in state):
        raise RuntimeError("B3 Sfix checkpoint lacks decoder provenance keys")
    encoder = EarlyPoolEncoder(trial_length=1024, window_size=100, hidden_dim=64, num_post_layers=3)
    expected = set(encoder.state_dict())
    missing = sorted(expected - set(incoming))
    unexpected = sorted(set(incoming) - expected)
    if not incoming or missing or unexpected:
        raise RuntimeError(f"strict B3 id_encoder keys failed: missing={missing[:5]} unexpected={unexpected[:5]}")
    encoder.load_state_dict(incoming, strict=True)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


@torch.no_grad()
def _b3_identity(encoder: nn.Module, calib_first10: np.ndarray) -> np.ndarray:
    """compute_identity(side_features=None) semantics on the 10 calib trials."""
    calib = torch.as_tensor(np.ascontiguousarray(calib_first10, dtype=np.float32)[None, ...])
    trials = calib[0] if calib.dim() == 4 else calib
    stream = encoder.reset_stream(1, m1_plan.N_UNITS, trials.device, trials.dtype)
    for trial in trials:
        encoder.push_trial(stream, trial.unsqueeze(0))
    identity = encoder.finalize_identity(stream)
    if identity.dim() == 3:
        identity = identity[0]
    if tuple(identity.shape) != (m1_plan.N_UNITS, M1_TEMPORAL.e0_dim):
        raise RuntimeError(f"B3 identity shape drift: {tuple(identity.shape)}")
    return np.ascontiguousarray(identity.detach().numpy(), dtype=np.float32)


def _encode_outer_carrier() -> tuple[np.ndarray, dict[str, Any]]:
    """ses-20120924 rSyn3 carrier with the sealed basis + source normalizer.

    Mechanism reused read-only from m1_family_loso_outer20120924.py::
    _encode_outer_carrier (sealed NPZ basis; narrow target support read).
    """
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import data as fold_data
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.carrier_bank import _encode_session
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as parent_data, plan as parent_plan, syn3

    loaded = src_bank.load()
    path = parent_data.require_source_path(WORKSPACE_ROOT / parent_plan.SOURCE_RELATIVE[OUTER])
    if parent_data.file_sha256(path) != parent_plan.SOURCE_FILE_SHA256[OUTER]:
        raise RuntimeError("outer session source hash drift")
    record = fold_data.load_fold_session(path, role="target")
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
    raw = _encode_session(record, basis)
    normalized = np.ascontiguousarray(
        syn3.normalize_carriers(raw, loaded["normalizer_mean"], loaded["normalizer_scale"]),
        dtype=np.float32,
    )
    meta = {
        "raw_carrier_array_digest": syn3.array_digest(np.ascontiguousarray(raw)),
        "normalized_carrier_sha256": array_sha256(normalized),
        "mechanism": "sealed-basis encode (m1_family_loso_outer20120924._encode_outer_carrier); role=target narrow support read",
    }
    return normalized, meta


def _verify_source_carrier_reencode(loaded: dict[str, Any]) -> dict[str, Any]:
    """Re-encode the 3 source sessions with the sealed basis; must reproduce NPZ raw bytes."""
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import data as fold_data
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.carrier_bank import _encode_session
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as parent_data, plan as parent_plan, syn3

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
    report: dict[str, Any] = {}
    for name in SOURCES:
        path = parent_data.require_source_path(WORKSPACE_ROOT / parent_plan.SOURCE_RELATIVE[name])
        if parent_data.file_sha256(path) != parent_plan.SOURCE_FILE_SHA256[name]:
            raise RuntimeError(f"source hash drift: {name}")
        record = fold_data.load_fold_session(path, role="source")
        raw = _encode_session(record, basis)
        sealed = np.asarray(loaded["raw"][name], dtype=np.float64)
        report[name] = {
            "digest_match": syn3.array_digest(np.ascontiguousarray(raw)) == syn3.array_digest(np.ascontiguousarray(sealed)),
            "max_abs_diff": float(np.max(np.abs(np.asarray(raw, dtype=np.float64) - sealed))),
        }
    report["all_digest_match"] = all(v["digest_match"] for v in report.values() if isinstance(v, dict))
    return report


def build_banks(dataset, *, verify_sources: bool) -> tuple[dict[str, TaskBank], dict[str, Any]]:
    """Per-session frozen banks (E0 + rSyn3 carrier + all-true mask), M10."""
    loaded = src_bank.load()
    encoder = _load_b3_id_encoder()
    outer_carrier, outer_meta = _encode_outer_carrier()
    banks: dict[str, TaskBank] = {}
    bank_report: dict[str, Any] = {}
    for name in SESSIONS:
        calib10 = np.asarray(dataset.calib_trialized_neural_features[name][:M10])
        if calib10.shape[0] < M10:
            raise RuntimeError(f"{name}: fewer than {M10} calib trials")
        e0 = _b3_identity(encoder, calib10)
        if name == OUTER:
            carrier = outer_carrier
        else:
            carrier = np.ascontiguousarray(loaded["normalized"][name], dtype=np.float32)
        if carrier.shape != (m1_plan.N_UNITS, 4):
            raise RuntimeError(f"carrier shape drift for {name}: {carrier.shape}")
        meta = {
            "shape": tuple(e0.shape),
            "trial_count": M10,
            "estimator": (
                "B3 Sfix e11 student.id_encoder compute_identity(side_features=None) "
                "(EarlyPoolEncoder 1024->64 pre-pool, trial-mean, 3-layer affine "
                "64->100 post-pool) + rSyn3 carrier (sealed rSyn3-refit-v1 NPZ"
                + ("; sealed-basis encode for ses-20120924" if name == OUTER else "")
                + ")"
            ),
            "array_sha256": array_sha256(e0),
            "budget": M10,
            "surface": "fullsession-submission-build",
            "session": name,
            "carrier_sha256": array_sha256(carrier),
            "carrier_source": "sealed-basis-encode" if name == OUTER else "rSyn3-refit-v1.source-only.npz normalized",
            "carrier_raw_digest": outer_meta["raw_carrier_array_digest"] if name == OUTER else None,
            "unit_mask_all_true": True,
            "store_note": (
                "window store intentionally empty; training/scoring windows are served "
                "by the fold-local FalconDataset; the bank payload is E0/carrier/unit_mask"
            ),
        }
        banks[name] = TaskBank(
            session_id=name,
            E0=e0,
            carrier=carrier,
            unit_mask=np.ones(m1_plan.N_UNITS, dtype=np.bool_),
            X_store=np.zeros((0, GEOMETRY["window"], m1_plan.N_UNITS), dtype=np.float32),
            target_store=np.zeros((0, GEOMETRY["out_dim"]), dtype=np.float32),
            window_ids=np.zeros(0, dtype=np.int64),
            calibration_meta=meta,
        )
        bank_report[name] = {
            "e0_sha256": meta["array_sha256"],
            "carrier_sha256": meta["carrier_sha256"],
            "trial_count": M10,
            "budget": M10,
        }
    bank_report["outer_carrier_meta"] = outer_meta
    bank_report["sfix"] = {"path": str(S_FIX_PATH), "sha256": S_FIX_SHA256}
    bank_report["npz"] = {"path": str(m1_plan.BANK_NPZ), "sha256": _sha_file(m1_plan.BANK_NPZ)}
    if verify_sources:
        bank_report["source_carrier_reencode"] = _verify_source_carrier_reencode(loaded)
        if not bank_report["source_carrier_reencode"]["all_digest_match"]:
            raise RuntimeError("source carrier re-encode does not reproduce sealed NPZ raw bytes")
    # Cross-check source-session identity arrays against the sealed runtime cache.
    cache = np.load(RUNTIME_CACHE_NPZ, allow_pickle=False)
    e0_checks = {}
    for name in SOURCES:
        ref = np.asarray(cache[f"bank_e0/{name}"], dtype=np.float32)
        e0_checks[name] = {
            "max_abs_diff": float(np.max(np.abs(banks[name].E0 - ref))),
            "bitwise": bool(np.array_equal(banks[name].E0, ref)),
        }
        t_ref = np.asarray(cache[f"bank_t/{name}"], dtype=np.float32)
        e0_checks[name]["carrier_bitwise"] = bool(np.array_equal(banks[name].carrier, t_ref))
    bank_report["runtime_cache_crosscheck"] = e0_checks
    return banks, bank_report


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def run_probe(dest: Path) -> dict[str, Any]:
    t0 = time.monotonic()
    dm = _loso_module()
    dataset, sampler, info = assemble_training_universe(dm)
    banks, bank_report = build_banks(dataset, verify_sources=True)
    model = BTransformerUnifiedDecoder(GEOMETRY, seed=SEED)
    n_params = int(sum(p.numel() for p in model.parameters()))
    inventory = {
        "schema": "btransform_unified_v1_m1_fullsession_session_inventory",
        "cell": CELL,
        "utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "stage-2 submission build (ADDENDUM-SUBMISSION-PROTOCOL): ALL local held-in sessions incl. ses-20120924 (legal training member)",
        "geometry": GEOMETRY,
        "l_in": model.l_in,
        "decoder_params": n_params,
        "state_dict_key_count": len(model.state_dict()),
        "sessions": info["sessions"],
        "window_audit": info["window_audit"],
        "windows_per_session": info["windows_per_session"],
        "total_windows": info["total_windows"],
        "updates_per_epoch": info["updates_per_epoch"],
        "total_updates": info["updates_per_epoch"] * plan.EPOCHS,
        "warmup_updates": info["updates_per_epoch"] * plan.WARMUP_EPOCHS,
        "train_law": info["train_law"],
        "sampler": info["sampler"],
        "sampler_batch_sha256": info["sampler_batch_sha256"],
        "source_files": info["source_files"],
        "banks": bank_report,
        "calib_trials_per_session": M10,
        "probe_seconds": time.monotonic() - t0,
        "references": {
            "fold_plan": "tfpd_exploration/src/m1_emg_rsyn3_fold_local_v1/plan.py",
            "loso_script_readonly": str(LOSO_SCRIPT_REFERENCE),
            "carrier_npz_receipt": str(m1_plan.BANK_RECEIPT),
            "runtime_cache": str(RUNTIME_CACHE_NPZ),
        },
    }
    _seal(dest / "session_inventory.json", inventory)
    return inventory


def run_train(dest: Path) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)

    inventory = json.loads((dest / "session_inventory.json").read_text(encoding="utf-8"))
    dm = _loso_module()
    dataset, sampler, info = assemble_training_universe(dm)
    if info["sampler_batch_sha256"] != inventory["sampler_batch_sha256"]:
        raise RuntimeError("sampler digest drift vs sealed probe inventory")
    if info["updates_per_epoch"] != inventory["updates_per_epoch"]:
        raise RuntimeError("updates_per_epoch drift vs sealed probe inventory")
    banks, bank_report = build_banks(dataset, verify_sources=False)
    for name in SESSIONS:
        sealed = inventory["banks"][name]
        if banks[name].calibration_meta["array_sha256"] != sealed["e0_sha256"]:
            raise RuntimeError(f"bank E0 digest drift vs sealed probe inventory: {name}")
        if banks[name].calibration_meta["carrier_sha256"] != sealed["carrier_sha256"]:
            raise RuntimeError(f"bank carrier digest drift vs sealed probe inventory: {name}")

    model = BTransformerUnifiedDecoder(GEOMETRY, seed=SEED).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    if n_params != inventory["decoder_params"]:
        raise RuntimeError("param count drift vs probe")

    from tfpd_exploration.src.m2_dual_track_v1 import training as dual_training

    optimizer = dual_training.build_optimizer(
        model.trainable_parameters().items(), lr=plan.LR_PEAK, weight_decay=plan.WEIGHT_DECAY
    )
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    updates_per_epoch = info["updates_per_epoch"]
    total_updates = updates_per_epoch * plan.EPOCHS
    warmup_updates = updates_per_epoch * plan.WARMUP_EPOCHS

    picks = {
        **PICKS,
        "TRN": [
            item.replace("FILLED_AT_RUNTIME", str(updates_per_epoch))
            if "updates_per_epoch: FILLED_AT_RUNTIME" in item
            else item.replace("FILLED_AT_RUNTIME", str(total_updates))
            for item in PICKS["TRN"]
        ],
    }
    picks["TRN"][0] = (
        f"TRN-1 {{updates_per_epoch: {updates_per_epoch}, warmup_updates: {warmup_updates}, "
        f"total_updates: {total_updates}, peak_lr: 3e-4, min_lr: 3e-5, AdamW wd 0.01 "
        "betas (0.9,0.999) eps 1e-8, clip 1.0, batch 32, epochs 24}}"
    )

    six_rows = {k: (v.replace("FILLED_PARAM_COUNT", f"{n_params:,}") if k == "system" else v) for k, v in NOTE_SIX_ROWS.items()}

    _seal(
        dest / "run_meta.json",
        {
            "schema": "btransform_unified_v1_m1_fullsession_run_meta",
            "cell": CELL,
            "protocol": "stage-2 submission build (ADDENDUM-REPRIORITIZE / ADDENDUM-SUBMISSION-PROTOCOL)",
            "utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "geometry": GEOMETRY,
            "l_in": model.l_in,
            "decoder_params": n_params,
            "state_dict_key_count": len(model.state_dict()),
            "init_meta": model.init_meta,
            "unit_dropout_domain_meta": UNIT_DROPOUT_DOMAIN_META,
            "sessions": info["sessions"],
            "windows_per_session": info["windows_per_session"],
            "total_windows": info["total_windows"],
            "updates_per_epoch": updates_per_epoch,
            "warmup_updates": warmup_updates,
            "total_updates": total_updates,
            "ema_horizon_updates": plan.EMA_HORIZON_UPDATES,
            "lr_peak": plan.LR_PEAK,
            "lr_min": plan.LR_PEAK * plan.LR_MIN_FACTOR,
            "weight_decay": plan.WEIGHT_DECAY,
            "grad_clip": plan.GRAD_CLIP,
            "ema_decay": plan.EMA_DECAY,
            "unit_dropout_p": plan.UNIT_DROPOUT,
            "precision": "bf16 autocast (forward+loss; pred.float() before MSE)",
            "batch_size": plan.BATCH_SIZE,
            "epochs": plan.EPOCHS,
            "checkpoint_rule": "endpoint24 EMA (preregistered, only rule; no surface pick)",
            "train_law": info["train_law"],
            "sampler": info["sampler"],
            "sampler_batch_sha256": info["sampler_batch_sha256"],
            "source_files": info["source_files"],
            "banks": bank_report,
            "gpu_uuid": GPU1_UUID,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
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

    for epoch in range(1, plan.EPOCHS + 1):
        model.train()
        epoch_t0 = time.monotonic()
        running = 0.0
        n_batches = 0
        for batch_id, (x, y, sessions) in enumerate(loader):
            if time.monotonic() >= deadline:
                _seal(
                    dest / "budget_hit.json",
                    {
                        "schema": "btransform_unified_v1_m1_fullsession_budget_hit",
                        "cell": CELL,
                        "epoch": epoch,
                        "global_step": global_step,
                        "budget_seconds": BUDGET_SECONDS,
                        "elapsed_seconds": time.monotonic() - started,
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "note": "6h GPU budget hit before finishing 24 epochs; no rerun authorized",
                    },
                )
                raise RuntimeError("M1 fullsession 6h GPU budget hit")
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
                peak=plan.LR_PEAK,
                min_factor=plan.LR_MIN_FACTOR,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            generator = torch.Generator(device="cpu")
            generator.manual_seed(unit_dropout_seed(SEED, epoch, batch_id))
            keep = whole_unit_dropout(bank.unit_mask, p=plan.UNIT_DROPOUT, generator=generator)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(x, bank, dropout_keep=keep)
                loss = nn.functional.mse_loss(pred.float(), y)
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
            if not projection_reported and global_step == 50:
                per_step = (time.monotonic() - epoch_t0) / 50.0
                projected = per_step * total_updates
                projection = {
                    "schema": "btransform_unified_v1_m1_fullsession_budget_projection",
                    "cell": CELL,
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
                        f"projected train time {projected:.0f}s exceeds 92% of the 6h budget; refusing to start"
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
            "schema": "btransform_unified_v1_m1_fullsession_ckpt",
            "cell": CELL,
            "epoch": epoch,
            "global_step": global_step,
            "seed": SEED,
            "geometry": GEOMETRY,
            "sampler_batch_sha256": info["sampler_batch_sha256"],
            "raw_state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr": float(lr),
            "ema_decay": ema.decay,
        }
        torch.save(ckpt, dest / f"epoch_{epoch:03d}.pt")
        _append_jsonl(metrics_path, {"event": "epoch", "epoch": epoch, "cell": CELL, **extras, "unix": time.time()})
        _write_json(heartbeat, {"event": "epoch", "cell": CELL, "epoch": epoch, **extras, "unix": time.time(), "gpu_uuid": GPU1_UUID})

    summary = {
        "schema": "btransform_unified_v1_m1_fullsession_train_receipt",
        "cell": CELL,
        "protocol": "stage-2 submission build (ADDENDUM-SUBMISSION-PROTOCOL)",
        "seed": SEED,
        "gpu_uuid": GPU1_UUID,
        "sessions": info["sessions"],
        "windows_per_session": info["windows_per_session"],
        "total_windows": info["total_windows"],
        "updates_per_epoch": updates_per_epoch,
        "global_updates": global_step,
        "ema_updates": ema.n_updates,
        "epochs_completed": list(range(1, plan.EPOCHS + 1)),
        "checkpoint_rule": "endpoint24 EMA (preregistered, only rule)",
        "train_mse": train_mse_series,
        "train_mse_summary": {e: train_mse_series.get(e) for e in (1, 12, 24)},
        "lr_at_epoch_end": lr_series,
        "elapsed_s": time.monotonic() - started,
        "sampler_batch_sha256": info["sampler_batch_sha256"],
        "picks": picks,
        "note_six_rows": six_rows,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    _seal(dest / "train_receipt.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Finalize: endpoint24 EMA checkpoint + diagnostics + handoff contract
# ---------------------------------------------------------------------------


def _score_view(
    model: BTransformerUnifiedDecoder,
    banks: dict[str, TaskBank],
    dataset,
    indices: list[int] | None,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    class _IndexView(torch.utils.data.Dataset):
        def __init__(self, base, ids):
            self.base, self.ids = base, ids

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            return self.base[self.ids[i]]

    view = _IndexView(dataset, indices if indices is not None else list(range(len(dataset))))
    loader = DataLoader(view, batch_size=batch_size, collate_fn=_collate, shuffle=False, num_workers=0)
    preds, targets, sids = [], [], []
    model.eval()
    with torch.inference_mode():
        for x, y, sessions in loader:
            session = sessions[0]
            bank = banks[session]
            out = model(x.to(device), bank)
            preds.append(out.cpu().numpy().astype(np.float32))
            targets.append(y.numpy().astype(np.float32))
            sids.extend(sessions)
    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    report = session_mean_report(target, pred, np.asarray(sids))
    report["pred_std_over_target_std"] = float(pred.std() / target.std())
    report["role"] = "DIAGNOSTIC ONLY (in-training windows; never used for selection)"
    return report


def run_finalize(dest: Path) -> dict[str, Any]:
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    inventory = json.loads((dest / "session_inventory.json").read_text(encoding="utf-8"))
    dm = _loso_module()
    dataset, sampler, info = assemble_training_universe(dm)
    if info["sampler_batch_sha256"] != inventory["sampler_batch_sha256"]:
        raise RuntimeError("sampler digest drift vs sealed probe inventory")
    banks, _bank_report = build_banks(dataset, verify_sources=False)
    for name in SESSIONS:
        sealed = inventory["banks"][name]
        if banks[name].calibration_meta["array_sha256"] != sealed["e0_sha256"]:
            raise RuntimeError(f"bank E0 digest drift vs sealed probe inventory: {name}")

    model = BTransformerUnifiedDecoder(GEOMETRY, seed=SEED).to(device)
    ckpt24 = torch.load(dest / "epoch_024.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt24["raw_state_dict"], strict=True)
    shadow = ckpt24["ema"]["shadow"]
    named = model.trainable_parameters()
    if set(named) != set(shadow):
        raise RuntimeError("EMA/RAW key mismatch")
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(shadow[name].to(device=param.device, dtype=param.dtype))
    model.eval()

    # Diagnostic readouts (in-training windows; never for selection).
    train_diag = _score_view(model, banks, dataset, None, device)
    family = json.loads(FAMILY_RUN_META.read_text(encoding="utf-8"))
    cuts = {s: int(v["cut_padded_bin"]) for s, v in family["split"].items()}
    chron80_ids = [
        i for i, (s, start) in enumerate(dataset.window_indices) if s in cuts and int(start) >= cuts[s]
    ]
    chron80_diag = _score_view(model, banks, dataset, chron80_ids, device)

    endpoint_path = dest / "endpoint24_ema.pt"
    endpoint = {
        "schema": "btransform_unified_v1_m1_fullsession_endpoint24_ema_v1",
        "cell": CELL,
        "view": "EMA endpoint24 (preregistered rule; no surface pick)",
        "epoch": 24,
        "global_updates": ckpt24["global_step"],
        "ema_updates": ckpt24["ema"]["n_updates"],
        "ema_decay": ckpt24["ema_decay"],
        "seed": SEED,
        "geometry": GEOMETRY,
        "l_in": 100,
        "n_units": 64,
        "out_dim": 16,
        "divisor": 1.0,
        "sessions_trained": list(SESSIONS),
        "sampler_batch_sha256": ckpt24["sampler_batch_sha256"],
        "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
    }
    torch.save(endpoint, endpoint_path)
    endpoint_sha = _sha_file(endpoint_path)
    os.chmod(endpoint_path, 0o444)
    (dest / "endpoint24_ema.pt.sha256").write_text(f"{endpoint_sha}  endpoint24_ema.pt\n", encoding="utf-8")

    key_count = len(endpoint["state_dict"])
    n_params = int(sum(v.numel() for v in endpoint["state_dict"].values()))

    handoff = {
        "schema": "btransform_unified_v1_m1_fullsession_handoff_contract_v1",
        "cell": CELL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Handoff contract for the user's submission agent: sealed endpoint24-EMA "
            "checkpoint of the M1 full-session stage-2 build, its inference input "
            "contract, the deployment-side per-session bank construction contract, "
            "and the training manifest/data receipt chain. No EvalAI action is taken "
            "by this cell; no LOSO surface may be read on this artifact."
        ),
        "checkpoint": {
            "path": str(endpoint_path),
            "sha256": endpoint_sha,
            "view": "EMA endpoint24 (preregistered; the only rule)",
            "epoch": 24,
            "seed": 42,
            "state_dict_key_count": key_count,
            "param_count": n_params,
            "dtype": "float32",
            "load": (
                "model = btransform_unified_v1.model.BTransformerUnifiedDecoder("
                "<geometry mapping below>, seed=42); "
                "ckpt = torch.load(path, map_location='cpu', weights_only=False); "
                "model.load_state_dict(ckpt['state_dict'], strict=True); model.eval()"
            ),
            "raw_and_optimizer_view": f"{dest}/epoch_024.pt (raw_state_dict + ema + optimizer)",
        },
        "inference_contract": {
            "window_W": 100,
            "l_in": 100,
            "prefix_P": 0,
            "n_units": 64,
            "out_dim": 16,
            "unit_column_order": "units DataFrame row order (NOTE P2-13); unit_mask all-true",
            "observation": (
                "ascontiguousarray float32 [B, 100, 64]; raw 20-ms-bin spike counts; "
                "reject NaN/non-finite (NOTE P0-5); 99-bin zero pre-history pad law at "
                "session start (same as training)"
            ),
            "output": "16-dim EMG at the LAST bin of the window ([B,16], model.forward)",
            "divisor": 1.0,
            "scale_bridge": "divisor=1: model output IS the EMG prediction in native units; never divide by 20",
            "forward": "out = model(x, bank)  # eval mode; whole-unit dropout is training-mode only",
            "model_class": "btransform_unified_v1.model.BTransformerUnifiedDecoder",
            "geometry": GEOMETRY,
        },
        "bank_contract": {
            "deployment_budget": "M10 (the chronological FIRST 10 calibration trials of the session)",
            "identity": {
                "source": "B3 Sfix epoch 011 checkpoint (SPINT-lineage frozen encoder)",
                "path": str(S_FIX_PATH),
                "sha256": S_FIX_SHA256,
                "extraction": (
                    "m1_optimized_v2/calibration.py::load_frozen_b3 semantics — strict "
                    "student.id_encoder keys ONLY (no B3/Sfix decoder weight may enter "
                    "the network, NOTE P1-10); compute_identity(side_features=None) == "
                    "EarlyPoolEncoder(trial_length=1024, window_size=100, hidden_dim=64, "
                    "num_post_layers=3): per-trial Linear(1024->64)+ReLU, mean over the "
                    "10 trials, then the 3-layer affine stack 64->100"
                ),
                "input": "10 trialized 1024-bin calib spike trains (cubic interpolation law of the fold-local dataset)",
                "output_shape": [64, 100],
            },
            "carrier": {
                "operator": "rSyn3",
                "constants": {
                    "RANK": int(fold_plan.RANK),
                    "RIDGE_LAMBDA": float(fold_plan.RIDGE_LAMBDA),
                    "SUPPORT_TRIALS": int(fold_plan.SUPPORT_TRIALS),
                    "SEED": int(fold_plan.SEED),
                    "nmf_law": {
                        "n_components": 3,
                        "init": "nndsvda",
                        "solver": "cd",
                        "beta_loss": "frobenius",
                        "tol": 1e-5,
                        "max_iter": 1000,
                        "random_state": 42,
                    },
                },
                "sealed_source_basis": {
                    "npz": str(m1_plan.BANK_NPZ),
                    "sha256": _sha_file(m1_plan.BANK_NPZ),
                    "receipt": str(m1_plan.BANK_RECEIPT),
                    "fitted_on": "ses-20120926/27/28 full-session rectified EMG (ReLU -> source RMS scale -> NNMF rank-3, dictionary unit-norm, energy-ordered)",
                },
                "per_session_closed_form": [
                    "1. support = the session's first 10 calibration trials (EMG bins + spike rates)",
                    "2. scale the support EMG bins by the sealed source RMS scale (npz['scale'])",
                    "3. NNLS-project onto the sealed dictionary npz['d0'] (3 rows) -> scores [n_bins, 3]",
                    "4. per unit: ridge of its support firing rate on design [1, w1, w2, w3], "
                    "lambda=1 penalizing ONLY w (intercept unpenalized), closed-form "
                    "solve of (X'X/n + diag(0,1,1,1)) beta = X'r/n",
                    "5. carrier row = [w1, w2, w3, intercept]",
                    "6. normalize: (raw - normalizer_mean) / normalizer_scale with the sealed "
                    "source-session carrier mean/scale (npz['normalizer_mean'/'normalizer_scale'])",
                ],
                "reference_code": (
                    "tfpd_exploration/src/m1_emg_syn3_fcm_v1/syn3.py (project_basis / "
                    "fit_all_units / carrier_from_encoding / normalize_carriers) + "
                    "tfpd_exploration/src/m1_emg_rsyn3_fold_local_v1/carrier_bank.py::_encode_session"
                ),
            },
            "per_session_banks_used_in_training": {
                name: {
                    "e0_sha256": banks[name].calibration_meta["array_sha256"],
                    "carrier_sha256": banks[name].calibration_meta["carrier_sha256"],
                    "trial_count": M10,
                }
                for name in SESSIONS
            },
        },
        "training_manifest": {
            "sessions": list(SESSIONS),
            "windows_per_session": info["windows_per_session"],
            "total_windows": info["total_windows"],
            "updates_per_epoch": info["updates_per_epoch"],
            "epochs": plan.EPOCHS,
            "total_updates": info["updates_per_epoch"] * plan.EPOCHS,
            "recipe": "TRN-1 (AdamW wd 0.01 clip 1; warmup 1 ep -> 3e-4; cosine -> 3e-5; 24 ep; EMA 0.9995; unit dropout 0.10 S1 domain; batch 32; bf16 autocast)",
            "source_files": info["source_files"],
            "data_receipt_chain": {
                "session_inventory": str(dest / "session_inventory.json"),
                "carrier_npz": str(m1_plan.BANK_NPZ),
                "carrier_npz_receipt": str(m1_plan.BANK_RECEIPT),
                "runtime_cache": str(RUNTIME_CACHE_NPZ),
                "loso_script_readonly": str(LOSO_SCRIPT_REFERENCE),
                "fold_plan": "tfpd_exploration/src/m1_emg_rsyn3_fold_local_v1/plan.py",
            },
        },
        "diagnostics_not_for_selection": {
            "train_universe_r2": train_diag,
            "family_chron80_dev_slice_r2": {
                **chron80_diag,
                "n_points": int(len(chron80_ids)),
                "slice_rule": "window start >= family chron80 cut_padded_bin per source session (family run_meta; in-training windows, polluted by stage-2 retrain)",
            },
        },
        "picks": PICKS,
        "note_six_rows": {
            k: (v.replace("FILLED_PARAM_COUNT", f"{n_params:,}") if k == "system" else v)
            for k, v in NOTE_SIX_ROWS.items()
        },
    }
    _seal(dest / "handoff_contract.json", handoff)
    final = {
        "schema": "btransform_unified_v1_m1_fullsession_final_receipt",
        "cell": CELL,
        "status": "COMPLETE",
        "utc": datetime.now(timezone.utc).isoformat(),
        "endpoint24_ema": {
            "path": str(endpoint_path),
            "sha256": endpoint_sha,
            "state_dict_key_count": key_count,
            "param_count": n_params,
        },
        "handoff_contract": str(dest / "handoff_contract.json"),
        "diagnostics_not_for_selection": handoff["diagnostics_not_for_selection"],
        "evalai_action_taken": False,
        "loso_surface_read": False,
    }
    _seal(dest / "final_receipt.json", final)
    return final


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="M1 full-session submission build (btransform_unified_v1, stage-2 protocol)")
    parser.add_argument("--stage", choices=["preflight", "probe", "train", "finalize", "all"], default="all")
    parser.add_argument("--dest", type=Path, default=None, help="existing run dir (default: new UTC-stamped)")
    args = parser.parse_args()

    if args.stage in ("train", "all") and os.environ.get(TRAIN_ENV_FLAG) != "1":
        print(f"REFUSED: training requires {TRAIN_ENV_FLAG}=1", file=sys.stderr)
        return 2
    if args.stage in ("train", "all", "finalize"):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            print("REFUSED: CUDA_VISIBLE_DEVICES must be pinned to '1' (GPU1)", file=sys.stderr)
            return 2

    dest = args.dest or plan.RESULT_ROOT / "m1_fullsession_submission_v1" / utc_stamp()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[m1-fullsession] dest = {dest}", flush=True)

    pre = gpu_preflight(dest / f"preflight_gpu_{args.stage}.json")
    print(f"[m1-fullsession] preflight({args.stage}) ok={pre['ok']} foreign={pre['foreign_pids_on_gpu1']}", flush=True)
    if not pre["ok"]:
        print("[m1-fullsession] BLOCKED: GPU1 not clean; refusing to start", file=sys.stderr)
        return 2

    if args.stage in ("probe", "all"):
        inventory = run_probe(dest)
        print(
            f"[m1-fullsession] probe done: sessions={inventory['sessions']} "
            f"windows={inventory['total_windows']} updates/epoch={inventory['updates_per_epoch']}",
            flush=True,
        )
    if args.stage in ("train", "all"):
        summary = run_train(dest)
        print(
            f"[m1-fullsession] train done in {summary['elapsed_s']:.0f}s; "
            f"updates={summary['global_updates']}; e1/e12/e24 train_mse="
            f"{summary['train_mse'][1]:.6g}/{summary['train_mse'][12]:.6g}/{summary['train_mse'][24]:.6g}",
            flush=True,
        )
    if args.stage in ("finalize", "all"):
        final = run_finalize(dest)
        print(
            f"[m1-fullsession] finalize done: endpoint sha={final['endpoint24_ema']['sha256'][:16]}... "
            f"contract={final['handoff_contract']}",
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

            target = dest or (_plan.RESULT_ROOT / "m1_fullsession_submission_v1" / "error")
            target.mkdir(parents=True, exist_ok=True)
            _receipts.seal_json(
                target / "error_receipt.json",
                {
                    "schema": "btransform_unified_v1_m1_fullsession_error",
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "traceback": trace,
                },
            )
        except Exception:
            pass
        sys.exit(3)
