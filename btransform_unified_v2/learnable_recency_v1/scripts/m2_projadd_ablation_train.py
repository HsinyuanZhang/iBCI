#!/usr/bin/env python3
"""M2 proj_add (P16) learnable-recency identity-ablation trainer.

Mirrors m2_projadd_learnable_train.py (proj_add P16, learned_slope, default
ladder, seed 42) with the recipe byte-frozen and ONLY the identity injection
changed:

  activity_only: E0 kept (B3S activity identity), carrier zeroed.  NOTE: the
                 cached E0 was computed with side=[MOVE-T4, 0]; MOVE-T4 is
                 label-derived, so this arm is a token-channel ablation with
                 the label present in E0, not a label-free ACT arm.
  norm_only:     E0 replaced by z[u] broadcast to e0_dim=50, carrier zeroed,
                 z[u] = (rate_sess[u] - mu_src[u]) / max(sigma_src[u], eps).
                 rate_sess pools the SAME label-free M33 calibration support
                 set the B3S E0 was computed from (cache
                 ``<surface>/<session>/calib_activity.npy`` [33, 100, 96]);
                 mu_src/sigma_src use source_train sessions only.
  activity_only_empty_side: E0 RECOMPUTED by the SAME frozen B3S-derived
                 encoder on the same M33 support set, but with the encoder
                 side input all-zero ``zeros(N, 8)`` — no MOVE-T4 (the
                 label-derived token channel), no contrast.  Labels cannot
                 enter the identity path.  Carrier zeroed.  The encoder is
                 the frozen champion's ``student.id_encoder``: weights are
                 pinned by the canonical p0 film_states + selected EMPTY head
                 (byte-equal E0 to the cache construction; sealed by
                 scripts/rift_v1/prepare_m2_joint_ext6_m33.py).  train and
                 score recompute E0 independently with single-threaded CPU
                 GEMMs and cross-check SHA-256 digests (fail-closed).

Transforms always build NEW banks/arrays; the frozen dual-track cache objects
(npz/memmap) are never mutated.  ``--identity full`` is rejected: that arm is
the already-completed ``m2_projadd_learned_slope_default_s42`` run.

This file does not modify any existing script; it imports the learnable
template for the shared paired-recipe machinery.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
WORKSPACE, V1 = ROOT.parent, ROOT.parent / "btransform_unified_v1"
for path in (PKG / "src", ROOT, ROOT / "src", V1 / "src", V1 / "scripts", WORKSPACE, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from btransform_unified_v1.bank import TaskBank, array_sha256
from btransform_unified_v1 import plan as v1_plan
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout
from btransform_unified_v1.schedule import warmup_cosine_lr
from learnable_recency_v1.config import add_learnable_flags, config_from_args
from learnable_recency_v1.wrap import (
    LearnableRiftDecoder,
    LearnableRiftStreamDecoder,
    apply_group_lrs,
    split_optimizer_parameters,
    trainable_new_parameter_count,
)
from scripts.rift_v1 import m2_train as frozen
from tfpd_exploration.src.m2_dual_track_v1 import champion as old_champion
from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan
from tfpd_exploration.src.m2_dual_track_v1 import sampler as old_sampler
from tfpd_exploration.src.m2_dual_track_v1 import training as old_training

import m2_projadd_learnable_train as template

SEED, CONTEXT, EPOCHS, BATCH, PROJ_DIM = frozen.SEED, frozen.CONTEXT, frozen.EPOCHS, frozen.BATCH, frozen.PROJ_DIM
IDENTITIES = ("full", "activity_only", "norm_only")
ABLATION_IDENTITIES = ("activity_only", "norm_only")
# ``IDENTITIES``/``ABLATION_IDENTITIES`` keep their sealed historical contents.
# The label-free empty-side arm is additive; parser/validation use the *_ARM
# tuples below so the two frozen tuples stay byte-stable for existing seals.
EMPTY_SIDE_IDENTITY = "activity_only_empty_side"
IDENTITY_CHOICES = IDENTITIES + (EMPTY_SIDE_IDENTITY,)
ABLATION_ARMS = ABLATION_IDENTITIES + (EMPTY_SIDE_IDENTITY,)
EMPTY_SIDE_SEMANTICS = "zeros(N,8): no MOVE-T4, no contrast"
EMPTY_SIDE_NUM_POST_LAYERS = 3  # champion.install_empty_film post_pool Linear count (B3S 64+4 geometry)
SIGMA_EPS = 1e-6
SUPPORT_SHAPE = (int(old_plan.SUPPORT_HORIZON), int(old_plan.CALIB_TRIAL_LENGTH), int(old_plan.CHANNELS))  # (33, 100, 96)
SIDE_DIM = int(old_champion.SIDE_DIM)  # 8 = T4_DIM(4) + CONTRAST_DIM(4)
SCHEMA = "m2_rift_projadd_ablation_train_v1"
CHECKPOINT_SCHEMA = "m2_rift_projadd_ablation_epoch_checkpoint_v1"
SMOKE_RECEIPT_SCHEMA = "m2_rift_projadd_ablation_smoke_receipt_v1"
TRAIN_RECEIPT_SCHEMA = "m2_rift_projadd_ablation_train_receipt_v1"
RESULTS = PKG / "results"
SCORE_SCRIPT = HERE / "m2_projadd_ablation_score.py"


def sha(path: Path) -> str:
    return frozen._sha(path)


def atom(path: Path, value: Mapping[str, Any]) -> None:
    frozen._atomic_json(path, value)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    frozen._append_jsonl(path, value)


def source_hashes() -> dict[str, str]:
    """Frozen proj_add trainer seal, learnable sources, the template, and this ablation pair."""
    paths = {Path(name) for name in frozen._source_hashes()}
    paths.update({Path(__file__).resolve(), Path(template.__file__).resolve(), SCORE_SCRIPT.resolve()})
    for path in sorted((PKG / "src/learnable_recency_v1").glob("*.py")):
        paths.add(path)
    return {str(path): sha(path) for path in sorted(paths)}


def float64_sha256(array: np.ndarray) -> str:
    """SHA-256 over C-contiguous float64 bytes (stable across train/score)."""
    data = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(data.tobytes(order="C")).hexdigest()


def pooled_rate_from_activity(activity: np.ndarray) -> np.ndarray:
    """Pooled per-unit support firing rate: total counts / total duration.

    The M33 support set is stored as 33 trials x 100 interpolated bins per
    unit, so the pooled rate is the float64 sum over all support bins divided
    by the bin count.  Every session uses the same grid; this equals
    counts/seconds up to one GLOBAL bin-duration constant, which cancels in
    the per-unit z-score against source statistics.
    """
    arr = np.asarray(activity, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (SUPPORT_SHAPE[1], SUPPORT_SHAPE[2]) or arr.shape[0] < 1:
        raise RuntimeError(f"calib activity shape drift: {arr.shape} != (*, {SUPPORT_SHAPE[1]}, {SUPPORT_SHAPE[2]})")
    return arr.sum(axis=(0, 1), dtype=np.float64) / float(arr.shape[0] * arr.shape[1])


def load_support_rate(bank: TaskBank) -> np.ndarray:
    """rate_sess for a training-surface bank from its own frozen cache support file.

    Path is provenance-driven: ``calibration_meta['cache_root']/surface/session/
    calib_activity.npy`` — exactly the support set the B3S encoder consumed for
    this bank's E0 (stage0 ``_compute_identity_for_dest`` reads the same file).
    """
    root = Path(str(bank.calibration_meta["cache_root"]))
    surface = str(bank.calibration_meta["surface"])
    path = root / surface / bank.session_id / "calib_activity.npy"
    activity = np.load(path, mmap_mode="r")
    if tuple(int(v) for v in activity.shape) != SUPPORT_SHAPE:
        raise RuntimeError(f"{surface}/{bank.session_id} calib activity shape {activity.shape} != {SUPPORT_SHAPE}")
    return pooled_rate_from_activity(np.asarray(activity))


def source_z_stats(source_rates: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Per-unit source mean/std over source_train session rates (population std)."""
    if not source_rates:
        raise RuntimeError("source rate table is empty")
    matrix = np.stack([np.asarray(source_rates[name], dtype=np.float64) for name in sorted(source_rates)], axis=0)
    if matrix.ndim != 2 or matrix.shape[1] != 96 or matrix.shape[0] != len(old_plan.HELDIN_SESSIONS):
        raise RuntimeError(f"source rate matrix drift: {matrix.shape}")
    mu = matrix.mean(axis=0)
    sigma = matrix.std(axis=0)
    return {
        "source_sessions": sorted(source_rates),
        "mu_src": mu,
        "sigma_src": sigma,
        "mu_sha256": float64_sha256(mu),
        "sigma_sha256": float64_sha256(sigma),
    }


def norm_z(rate: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """z[u] = (rate - mu) / max(sigma, eps); float64 throughout for determinism."""
    z = (np.asarray(rate, dtype=np.float64) - np.asarray(mu, dtype=np.float64)) / np.maximum(
        np.asarray(sigma, dtype=np.float64), SIGMA_EPS
    )
    if not np.isfinite(z).all():
        raise RuntimeError("non-finite norm_only z")
    return z


def norm_e0_from_stats(rate: np.ndarray, mu: np.ndarray, sigma: np.ndarray, *, e0_dim: int, unit_mask: np.ndarray) -> np.ndarray:
    """E0 [N, e0_dim] = z broadcast across columns; padding (masked) rows are zero."""
    z = norm_z(rate, mu, sigma)
    e0 = np.broadcast_to(z[:, None], (z.shape[0], int(e0_dim))).astype(np.float32)
    e0 = np.array(e0, dtype=np.float32, order="C", copy=True)
    e0[~np.asarray(unit_mask, dtype=bool)] = 0.0
    return np.ascontiguousarray(e0, dtype=np.float32)


_EMPTY_SIDE_ENCODER_MEMO: dict[str, tuple[Any, dict[str, Any]]] = {}


def frozen_empty_side_encoder(device: torch.device = torch.device("cpu")) -> tuple[Any, dict[str, Any]]:
    """The frozen champion identity encoder plus its deterministic identity record.

    ``data.py`` stage0 uses ``champion.load_frozen_champion(device).student.id_encoder``.
    Its weights are fully pinned by the canonical p0 film_states + the selected
    EMPTY head: ``install_empty_film`` copies the B3S weights only to be
    immediately overwritten by the strict canonical-p0 overlay, so building
    ``HoldContrastFiLMEarlyPoolEncoder`` directly and overlaying the same two
    state dicts yields the byte-identical encoder
    (scripts/rift_v1/prepare_m2_joint_ext6_m33.py seals this equivalence by
    reproducing the query-cache E0 bit for bit).  The direct construction is
    used because the streaming champion ``best.ckpt`` is not part of this
    machine's frozen artifacts.

    The record travels into run_meta; the score side rebuilds the encoder and
    refuses to run unless its record matches bit for bit.
    """
    key = str(torch.device(device))
    if key not in _EMPTY_SIDE_ENCODER_MEMO:
        from tfpd_exploration.src.m2_hold_film_probe_v1.encoder import HoldContrastFiLMEarlyPoolEncoder

        encoder = HoldContrastFiLMEarlyPoolEncoder(
            int(old_plan.CALIB_TRIAL_LENGTH),
            int(old_plan.WINDOW),
            int(old_plan.HIDDEN_DIM),
            side_dim=SIDE_DIM,
            film_rank=int(old_champion.FILM_RANK),
            num_post_layers=EMPTY_SIDE_NUM_POST_LAYERS,
            film_input="t4_plus_contrast",
        ).to(device)
        head_meta = old_champion.overlay_canonical_p0_and_empty_head(encoder)
        if str(head_meta["head_state_sha256"]) != old_plan.SELECTED_HEAD_STATE_SHA256:
            raise RuntimeError("empty-side encoder EMPTY head digest drift")
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        film_path = WORKSPACE / old_plan.FILM_STATES_RELATIVE
        head_path = WORKSPACE / old_plan.SELECTED_HEAD_RELATIVE
        record = {
            "construction": (
                "HoldContrastFiLMEarlyPoolEncoder(100,50,64,side_dim=8,film_rank=8,"
                "num_post_layers=3,t4_plus_contrast)+canonical_p0+selected_EMPTY_head"
            ),
            "equivalent_to_stage0_encoder": (
                "same weights as champion.load_frozen_champion(device).student.id_encoder "
                "(canonical p0 strict overlay + EMPTY head); sealed byte-equal by "
                "scripts/rift_v1/prepare_m2_joint_ext6_m33.py query-cache E0 reproduction"
            ),
            "champion_cache_key_parts": old_champion.cache_key_parts(),
            "film_states": {"path": str(film_path), "sha256": sha(film_path)},
            "selected_empty_head": {"path": str(head_path), "sha256": sha(head_path)},
            "head_state_sha256": str(head_meta["head_state_sha256"]),
            "eval_mode": True,
            "no_grad": True,
        }
        _EMPTY_SIDE_ENCODER_MEMO[key] = (encoder, record)
    return _EMPTY_SIDE_ENCODER_MEMO[key]


def empty_side_e0(encoder: Any, activity_path: Path) -> np.ndarray:
    """Recompute E0 from the M33 support activity with side = zeros(N, 8).

    Reads ONLY ``calib_activity.npy``; never touches ``T.npy``/angles (the
    label-derived MOVE-T4 path).  Computation is pinned to a single CPU thread
    so the digest is independent of ``--cpu-threads`` at train and score time.
    """
    activity = np.load(activity_path, mmap_mode="r")
    if tuple(int(v) for v in activity.shape) != SUPPORT_SHAPE:
        raise RuntimeError(f"{activity_path}: calib activity shape {activity.shape} != {SUPPORT_SHAPE}")
    trials = torch.from_numpy(np.array(activity, dtype=np.float32, copy=True))
    n_units = SUPPORT_SHAPE[2]
    side = torch.zeros(n_units, SIDE_DIM, dtype=torch.float32)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        e0, _u = old_champion.native_e0_and_u(encoder, trials, side)  # encoder.eval() + no_grad inside
    finally:
        torch.set_num_threads(threads)
    e0_np = np.ascontiguousarray(e0.numpy(), dtype=np.float32)
    if e0_np.shape != (n_units, int(old_plan.IDENTITY_DIM)) or not np.isfinite(e0_np).all():
        raise RuntimeError(f"{activity_path}: empty-side E0 shape/finiteness drift {e0_np.shape}")
    return e0_np


def support_activity_path(bank: TaskBank) -> Path:
    """Provenance-driven M33 support file behind a cache bank's E0 (calib only)."""
    root = Path(str(bank.calibration_meta["cache_root"]))
    surface = str(bank.calibration_meta["surface"])
    return root / surface / bank.session_id / "calib_activity.npy"


def transform_bank(
    bank: TaskBank,
    identity: str,
    *,
    rate: np.ndarray | None = None,
    mu: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    e0: np.ndarray | None = None,
) -> TaskBank:
    """Return a NEW bank with the ablation identity applied; ``bank`` is never mutated.

    E0/carrier are always fresh C-contiguous arrays (the frozen cache may back
    them with shared npz/memmap objects).  X_store/target_store/window_ids are
    observational stores shared read-only with the input bank; unit_mask is
    copied.  ``calibration_meta`` is copied with the ORIGINAL input digests
    intact (the frozen-cache binding) plus an ``identity_ablation`` record.

    ``activity_only_empty_side`` requires ``e0``: the fresh zero-side encoder
    E0 (same shape as ``bank.E0``).  The cached E0 is never consulted for the
    transformed bank — only its digest is recorded for comparison.
    """
    if identity not in ABLATION_ARMS:
        raise ValueError(f"identity must be one of {ABLATION_ARMS}, got {identity!r}")
    record: dict[str, Any] = {
        "identity": identity,
        "input_e0_sha256": str(bank.calibration_meta.get("array_sha256")),
        "input_carrier_sha256": str(bank.calibration_meta.get("carrier_sha256")),
    }
    if identity == "norm_only":
        if rate is None or mu is None or sigma is None:
            raise ValueError("norm_only transform requires rate/mu/sigma")
        z = norm_z(rate, mu, sigma)
        e0 = norm_e0_from_stats(rate, mu, sigma, e0_dim=bank.E0.shape[1], unit_mask=bank.unit_mask)
        record["norm_only"] = {
            "rate_sha256": float64_sha256(rate),
            "z_sha256": float64_sha256(z),
            "mu_sha256": float64_sha256(mu),
            "sigma_sha256": float64_sha256(sigma),
            "sigma_eps": SIGMA_EPS,
        }
    elif identity == EMPTY_SIDE_IDENTITY:
        if e0 is None:
            raise ValueError("activity_only_empty_side transform requires the recomputed zero-side e0")
        e0 = np.array(e0, dtype=np.float32, order="C", copy=True)  # the bank owns its E0
        if e0.shape != tuple(bank.E0.shape) or not np.isfinite(e0).all():
            raise RuntimeError(f"empty-side e0 shape/finiteness drift: {e0.shape} vs {tuple(bank.E0.shape)}")
        record["empty_side"] = {
            "side": EMPTY_SIDE_SEMANTICS,
            "side_dim": SIDE_DIM,
            "e0_sha256": array_sha256(e0),
            "e0_sha256_differs_from_cached": array_sha256(e0) != record["input_e0_sha256"],
        }
    else:
        e0 = np.array(bank.E0, dtype=np.float32, order="C", copy=True)
    carrier = np.zeros(tuple(bank.carrier.shape), dtype=np.float32, order="C")
    meta = dict(bank.calibration_meta)
    record["transformed_e0_sha256"] = array_sha256(e0)
    record["transformed_carrier_sha256"] = array_sha256(carrier)
    meta["identity_ablation"] = record
    return TaskBank(
        session_id=bank.session_id,
        E0=e0,
        carrier=carrier,
        unit_mask=np.array(bank.unit_mask, dtype=np.bool_, copy=True),
        X_store=bank.X_store,
        target_store=bank.target_store,
        window_ids=bank.window_ids,
        calibration_meta=meta,
    )


def identity_ablation_meta(
    identity: str,
    transformed_banks: Mapping[str, Mapping[str, TaskBank]],
    *,
    norm_block: Mapping[str, Any] | None = None,
    empty_side_block: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "identity": identity,
        "definition": (
            "activity_only: E0 unchanged (B3S activity identity), carrier zeroed"
            if identity == "activity_only"
            else "norm_only: E0[u,:] = z[u] broadcast to e0_dim=50, carrier zeroed, "
                 "z = (rate_sess - mu_src) / max(sigma_src, eps)"
            if identity == "norm_only"
            else "activity_only_empty_side: E0 recomputed by the frozen champion encoder "
                 "on the M33 support set with side=zeros(N,8) (no MOVE-T4, no contrast), "
                 "carrier zeroed; labels cannot reach the identity path"
        ),
        "transformed_e0_sha256": {surface: {s: array_sha256(b.E0) for s, b in banks.items()}
                                  for surface, banks in transformed_banks.items()},
        "transformed_carrier_sha256": {surface: {s: array_sha256(b.carrier) for s, b in banks.items()}
                                       for surface, banks in transformed_banks.items()},
    }
    if norm_block is not None:
        base["norm_only"] = dict(norm_block)
    if empty_side_block is not None:
        base["empty_side"] = dict(empty_side_block)
    return base


def load_surface(surface: str, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    return template.load_surface(surface, device)


def cache_hashes(surface: str, dual: Mapping[str, Any], banks: Mapping[str, Any]) -> dict[str, Any]:
    return template.cache_hashes(surface, dual, banks)


def padding(surface: str, banks: Mapping[str, Any]) -> dict[str, int]:
    return template.padding(surface, banks)


def apply_identity(
    identity: str,
    train_banks: Mapping[str, TaskBank],
    mini_banks: Mapping[str, TaskBank],
    ext_banks: Mapping[str, TaskBank],
) -> tuple[dict[str, Mapping[str, TaskBank]], dict[str, Any]]:
    """Transform every loaded surface bank and produce the run-level ablation record."""
    if identity == "activity_only":
        transformed = {
            "source_train": {s: transform_bank(b, identity) for s, b in train_banks.items()},
            "source_minival": {s: transform_bank(b, identity) for s, b in mini_banks.items()},
            "ext4": {s: transform_bank(b, identity) for s, b in ext_banks.items()},
        }
        return transformed, identity_ablation_meta(identity, transformed)

    if identity == EMPTY_SIDE_IDENTITY:
        encoder, encoder_record = frozen_empty_side_encoder(torch.device("cpu"))
        e0_table: dict[str, dict[str, np.ndarray]] = {}
        transformed: dict[str, Mapping[str, TaskBank]] = {}
        for surface, banks in (("source_train", train_banks), ("source_minival", mini_banks), ("ext4", ext_banks)):
            e0_table[surface] = {}
            out: dict[str, TaskBank] = {}
            for session, bank in banks.items():
                e0 = empty_side_e0(encoder, support_activity_path(bank))
                e0_table[surface][session] = e0
                out[session] = transform_bank(bank, identity, e0=e0)
            transformed[surface] = out
        for session in train_banks:
            if float64_sha256(e0_table["source_minival"][session]) != float64_sha256(e0_table["source_train"][session]):
                raise RuntimeError("source_minival empty-side E0 differs from source_train (support-set drift)")
        empty_block = {
            "side": EMPTY_SIDE_SEMANTICS,
            "side_dim": SIDE_DIM,
            "support": "same M33 calibration support set as the cached E0 "
                       "(<dual cache>/<surface>/<session>/calib_activity.npy, 33x100x96); "
                       "T.npy/angles/MOVE-T4 never read",
            "e0_computation": "champion.native_e0_and_u(encoder, trials, zeros(96,8)) on one CPU thread; "
                              "encoder.eval() + torch.no_grad()",
            "encoder": encoder_record,
            "e0_sha256": {surface: {s: float64_sha256(e0) for s, e0 in table.items()}
                          for surface, table in e0_table.items()},
            "cached_e0_sha256": {surface: {s: str(banks[s].calibration_meta.get("array_sha256"))
                                           for s in banks}
                                 for surface, banks in (("source_train", train_banks), ("source_minival", mini_banks), ("ext4", ext_banks))},
            "e0_sha256_differs_from_cached": {
                surface: {s: float64_sha256(e0_table[surface][s]) != str(banks[s].calibration_meta.get("array_sha256"))
                          for s in banks}
                for surface, banks in (("source_train", train_banks), ("source_minival", mini_banks), ("ext4", ext_banks))
            },
            "minival_e0_equals_source_train": True,
            "label_free": "identity path consumes calib neural activity only; "
                          "no MOVE-T4 (label-derived), no angles, no contrast",
        }
        return transformed, identity_ablation_meta(identity, transformed, empty_side_block=empty_block)

    source_rates = {s: load_support_rate(b) for s, b in train_banks.items()}
    stats = source_z_stats(source_rates)
    mini_rates = {s: load_support_rate(b) for s, b in mini_banks.items()}
    ext_rates = {s: load_support_rate(b) for s, b in ext_banks.items()}
    if any(not np.array_equal(source_rates[s], mini_rates[s]) for s in train_banks):
        raise RuntimeError("source_minival support rates differ from source_train (support-set drift)")
    mu, sigma = stats["mu_src"], stats["sigma_src"]
    rates = {"source_train": source_rates, "source_minival": mini_rates, "ext4": ext_rates}
    transformed = {
        surface: {s: transform_bank(b, identity, rate=rates[surface][s], mu=mu, sigma=sigma)
                  for s, b in banks.items()}
        for surface, banks in (("source_train", train_banks), ("source_minival", mini_banks), ("ext4", ext_banks))
    }
    norm_block = {
        "support": "same label-free M33 calibration support set as the B3S E0 "
                   "(<dual cache>/<surface>/<session>/calib_activity.npy, 33x100x96)",
        "rate_formula": "rate_sess[u] = float64 sum of calib activity over all 33x100 support bins / bin count; "
                        "equals total-counts/total-duration up to one global bin-duration constant that cancels in z",
        "sigma_eps": SIGMA_EPS,
        "mu_sha256": stats["mu_sha256"],
        "sigma_sha256": stats["sigma_sha256"],
        "mu_src": [float(v) for v in mu],
        "sigma_src": [float(v) for v in sigma],
        "source_sessions": list(stats["source_sessions"]),
        "rates": {surface: {s: [float(v) for v in rate] for s, rate in table.items()} for surface, table in rates.items()},
        "rate_sha256": {surface: {s: float64_sha256(rate) for s, rate in table.items()} for surface, table in rates.items()},
        "z_sha256": {surface: {s: str(transformed[surface][s].calibration_meta["identity_ablation"]["norm_only"]["z_sha256"])
                               for s in transformed[surface]} for surface in transformed},
        "minival_rate_equals_source_train": True,
        "ho_leakage": "mu_src/sigma_src computed from source_train sessions only; "
                      "source_minival/ext4/EXT6 rates are per-session inputs to z and never enter the statistics",
    }
    return transformed, identity_ablation_meta(identity, transformed, norm_block=norm_block)


def ensure_ablation_identity(identity: str) -> None:
    if identity == "full":
        raise ValueError(
            "--identity full is the already-completed m2_projadd_learned_slope_default_s42 reference; "
            "train it with scripts/m2_projadd_learnable_train.py"
        )
    if identity not in ABLATION_ARMS:
        raise ValueError(f"identity must be one of {ABLATION_ARMS}, got {identity!r}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_ablation_identity(args.identity)
    smoke = args.max_updates_smoke is not None
    recency_cfg = config_from_args(args, "m2")
    ensure_frozen_recipe(args)
    if not smoke and args.epochs != EPOCHS:
        raise ValueError("formal ablation M2 requires exactly 24 epochs")
    device, dest = torch.device(args.device), args.dest.resolve()
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    manifest = old_sampler.load_manifest(frozen.MANIFEST)
    if manifest["digest"] != frozen.MANIFEST_DIGEST or int(manifest["batch_size"]) != BATCH:
        raise RuntimeError("frozen M2 sampler manifest drift")
    if args.resume is None and dest.exists() and any(dest.iterdir()):
        raise FileExistsError("fresh ablation destination must be empty")
    if args.resume is not None and args.resume.resolve().parent != dest:
        raise RuntimeError("ablation resume must belong directly to --dest")
    train_dual, train_banks_orig = load_surface("source_train", device)
    mini_dual, mini_banks_orig = load_surface("source_minival", device)
    ext_dual, ext_banks_orig = load_surface("ext4", device)
    if old_training.count_updates(train_dual) != 3165:
        raise RuntimeError("source train update budget drift")
    transformed, ablation_record = apply_identity(args.identity, train_banks_orig, mini_banks_orig, ext_banks_orig)
    train_banks = dict(transformed["source_train"])
    mini_banks = dict(transformed["source_minival"])
    ext_banks = dict(transformed["ext4"])
    for surface, original in (("source_train", train_banks_orig), ("source_minival", mini_banks_orig), ("ext4", ext_banks_orig)):
        for session, bank in original.items():
            if bank.E0 is transformed[surface][session].E0 or bank.carrier is transformed[surface][session].carrier:
                raise RuntimeError(f"{surface}/{session}: transform must not share identity arrays with the frozen bank")
    caches = {
        "source_train": cache_hashes("source_train", train_dual, train_banks),
        "source_minival": cache_hashes("source_minival", mini_dual, mini_banks),
        "ext4": cache_hashes("ext4", ext_dual, ext_banks),
    }
    reference = template.reference_binding_projadd()
    if caches != reference["frozen_cache_hashes"] or manifest["digest"] != reference["manifest_digest"]:
        raise RuntimeError("ablation input cache/manifest differs from paired proj_add recency reference")
    pads = {
        surface: padding(surface, banks)
        for surface, banks in (("source_train", train_banks), ("source_minival", mini_banks), ("ext4", ext_banks))
    }
    model = template.learnable_decoder(device, recency_cfg)
    init = template.assert_paired_initialization(model, device, recency_cfg)
    ema = DecoderEMA(model, decay=v1_plan.EMA_DECAY)
    groups = split_optimizer_parameters(
        model, peak_lr=v1_plan.LR_PEAK, weight_decay=v1_plan.WEIGHT_DECAY, lr_multiplier=recency_cfg.lr_multiplier
    )
    optimizer = torch.optim.AdamW(groups, lr=v1_plan.LR_PEAK, betas=old_plan.ADAM_BETAS, eps=old_plan.ADAM_EPS)
    cell = f"M2-RIFT-R50-D4-P16-PROJADD-LEARNABLE-ABLATION-{args.identity.upper()}-V1"
    meta = {
        "schema": SCHEMA,
        "status": "SMOKE" if smoke else "FORMAL",
        "cell": cell,
        "seed": SEED,
        "task": "m2",
        "tier": recency_cfg.tier,
        "identity": args.identity,
        "learnable_config": recency_cfg.__dict__,
        "new_parameter_names": init["new_parameter_names"],
        "new_parameter_counts": init["new_parameter_counts"],
        "trainable_new_parameter_count": init["trainable_new_parameter_count"],
        "identity_interface": "proj_add",
        "identity_ablation": ablation_record,
        "proj_dim": PROJ_DIM,
        "context_bins": CONTEXT,
        "layer_windows": list(recency_cfg.temporal_config.windows),
        "depth": recency_cfg.layers,
        "ladder": recency_cfg.ladder_metadata(),
        "width": 256,
        "attention_backend": "local",
        "batch": BATCH,
        "epochs": EPOCHS,
        "updates_per_epoch": 3165,
        "optimizer": {
            "name": "AdamW",
            "weight_decay": v1_plan.WEIGHT_DECAY,
            "new_param_weight_decay": 0.0,
            "new_param_lr_multiplier": recency_cfg.lr_multiplier,
            "betas": list(old_plan.ADAM_BETAS),
            "eps": old_plan.ADAM_EPS,
            "clip": v1_plan.GRAD_CLIP,
        },
        "lr": {"peak": v1_plan.LR_PEAK, "min": v1_plan.LR_PEAK * v1_plan.LR_MIN_FACTOR, "warmup_updates": 3165},
        "ema_decay": v1_plan.EMA_DECAY,
        "unit_dropout": v1_plan.UNIT_DROPOUT,
        "source_hashes": source_hashes(),
        "frozen_cache_hashes": caches,
        "paired_recency_reference": reference,
        "initialization_pairing": dict(init),
        "runtime": {
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_threads": torch.get_num_threads(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    dest.mkdir(parents=True, exist_ok=True)
    if args.resume is None:
        atom(dest / "run_meta.json", meta)
    step, start_epoch = 0, 1
    started = time.monotonic()
    total = EPOCHS * 3165
    smoke_preupdate_parity = smoke_postupdate_parity = None
    last_bias = None
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for batch_index, batch in enumerate(
            old_sampler.iter_manifest_batches(
                train_dual, manifest, epoch, device=device, target_space=old_plan.TRAINING_TARGET_SPACE
            )
        ):
            step += 1
            lr = warmup_cosine_lr(step, total_steps=total, warmup_steps=3165, peak=v1_plan.LR_PEAK, min_factor=v1_plan.LR_MIN_FACTOR)
            apply_group_lrs(optimizer, lr)
            keep_rng = torch.Generator(device="cpu")
            keep_rng.manual_seed(unit_dropout_seed(SEED, epoch, batch_index))
            keep = whole_unit_dropout(batch.unit_mask, p=v1_plan.UNIT_DROPOUT, generator=keep_rng)
            valid = frozen._batch_valid(batch, surface="source_train", padding=pads["source_train"], device=device)
            if smoke and smoke_preupdate_parity is None:
                smoke_preupdate_parity = template.smoke_full_stream_parity(model, batch, train_banks[batch.session_id], valid)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
            with amp:
                pred = model(batch.X, train_banks[batch.session_id], dropout_keep=keep, input_valid_mask=valid)
                loss = nn.functional.mse_loss(pred.float(), batch.last_target)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite ablation loss at epoch={epoch}, batch={batch_index}")
            loss.backward()
            grad = float(nn.utils.clip_grad_norm_(model.parameters(), v1_plan.GRAD_CLIP, error_if_nonfinite=True))
            optimizer.step()
            ema.update_after_step(model)
            if smoke and smoke_postupdate_parity is None:
                smoke_postupdate_parity = template.smoke_full_stream_parity(model, batch, train_banks[batch.session_id], valid)
            losses.append(float(loss.detach().cpu()))
            last_bias = template.bias_snapshot(model, batch, train_banks[batch.session_id], valid)
            if step == 1 or step % 100 == 0:
                atom(
                    dest / "heartbeat.json",
                    {
                        "status": "TRAINING",
                        "event": "step",
                        "identity": args.identity,
                        "epoch": epoch,
                        "global_step": step,
                        "loss": losses[-1],
                        "lr": lr,
                        "grad_norm": grad,
                        "elapsed_seconds": time.monotonic() - started,
                        "bias": last_bias,
                    },
                )
            if smoke and step >= args.max_updates_smoke:
                break
        score_limit = 1 if smoke else None
        raw = frozen._score(model, mini_dual, mini_banks, pads["source_minival"], "source_minival", device, max_batches_per_session=score_limit)
        ema_score = frozen._with_ema(
            model,
            ema,
            lambda: frozen._score(model, mini_dual, mini_banks, pads["source_minival"], "source_minival", device, max_batches_per_session=score_limit),
        )
        ckpt = dest / f"epoch_{epoch:03d}.pt"
        if ckpt.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint {ckpt}")
        torch.save(
            {
                "schema": CHECKPOINT_SCHEMA,
                "cell": cell,
                "epoch": epoch,
                "global_step": step,
                "seed": SEED,
                "tier": recency_cfg.tier,
                "identity": args.identity,
                "identity_interface": "proj_add",
                "proj_dim": PROJ_DIM,
                "new_parameter_names": init["new_parameter_names"],
                "smoke": smoke,
                "raw_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema.state_dict(),
                "rng": frozen._rng_state(device),
                "initialization_pairing": dict(init),
            },
            ckpt,
        )
        elapsed = time.monotonic() - started
        epoch_row = {
            "status": "SMOKE" if smoke else "TRAINING",
            "event": "epoch",
            "identity": args.identity,
            "epoch": epoch,
            "global_step": step,
            "epoch_update_count": len(losses),
            "train_mse": float(np.mean(losses)),
            "ema_updates": ema.n_updates,
            "elapsed_seconds": elapsed,
            "minival_raw": raw,
            "minival_ema": ema_score,
            "bias": last_bias,
        }
        atom(dest / "heartbeat.json", epoch_row)
        append_jsonl(dest / "metrics.jsonl", epoch_row)
        if smoke:
            receipt = {
                "schema": SMOKE_RECEIPT_SCHEMA,
                "status": "COMPLETED",
                "cell": cell,
                "identity": args.identity,
                "tier": recency_cfg.tier,
                "epoch": epoch,
                "global_step": step,
                "finite_train_mse": bool(math.isfinite(float(np.mean(losses)))),
                "finite_loss": bool(math.isfinite(float(np.mean(losses)))),
                "checkpoint": str(ckpt),
                "runtime_seconds": elapsed,
                "full_stream_parity_preupdate": smoke_preupdate_parity,
                "full_stream_parity_postupdate": smoke_postupdate_parity,
                "bias": last_bias,
                "identity_ablation": {
                    "identity": args.identity,
                    "transformed_e0_sha256": ablation_record["transformed_e0_sha256"],
                    "transformed_carrier_sha256": ablation_record["transformed_carrier_sha256"],
                    **({"empty_side": {
                        "side": ablation_record["empty_side"]["side"],
                        "encoder_head_state_sha256": ablation_record["empty_side"]["encoder"]["head_state_sha256"],
                        "encoder_film_states_sha256": ablation_record["empty_side"]["encoder"]["film_states"]["sha256"],
                        "e0_sha256": ablation_record["empty_side"]["e0_sha256"],
                        "cached_e0_sha256": ablation_record["empty_side"]["cached_e0_sha256"],
                        "e0_sha256_differs_from_cached": ablation_record["empty_side"]["e0_sha256_differs_from_cached"],
                    }} if args.identity == EMPTY_SIDE_IDENTITY else {}),
                },
                "new_parameter_names": init["new_parameter_names"],
                "new_parameter_counts": init["new_parameter_counts"],
                "trainable_new_parameter_count": init["trainable_new_parameter_count"],
                "initialization_pairing": init,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }
            atom(dest / "smoke_receipt.json", receipt)
            return receipt
        if len(losses) != 3165 or step != epoch * 3165 or ema.n_updates != step:
            raise RuntimeError("formal ablation epoch update/EMA accounting drift")
    receipt = {
        "schema": TRAIN_RECEIPT_SCHEMA,
        "status": "COMPLETED",
        "cell": cell,
        "identity": args.identity,
        "tier": recency_cfg.tier,
        "epochs": EPOCHS,
        "global_step": step,
        "runtime_seconds": time.monotonic() - started,
        "source_hashes": meta["source_hashes"],
        "frozen_cache_hashes": caches,
        "identity_ablation": {
            "identity": args.identity,
            "transformed_e0_sha256": ablation_record["transformed_e0_sha256"],
            "transformed_carrier_sha256": ablation_record["transformed_carrier_sha256"],
        },
        "reference_binding": reference,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    atom(dest / "train_receipt.json", receipt)
    return receipt


def ensure_frozen_recipe(args: argparse.Namespace) -> None:
    """The ablation keeps the paired FULL recipe verbatim: learned_slope/default/4-layer/seed 42."""
    drifted = []
    if args.tier != "learned_slope":
        drifted.append("--tier learned_slope")
    if args.ladder != "default":
        drifted.append("--ladder default")
    if int(args.layers) != 4:
        drifted.append("--layers 4")
    if float(args.lr_multiplier) != 1.0:
        drifted.append("--lr-multiplier 1.0")
    if args.learn_flat_heads:
        drifted.append("--learn-flat-heads off")
    if args.cable_nw:
        drifted.append("--cable-nw off")
    if args.per_layer is not True:
        drifted.append("--per-layer")
    if args.half_lives is not None:
        drifted.append("no --half-lives override")
    if drifted:
        raise ValueError("identity ablation freezes the paired recipe; expected " + ", ".join(drifted))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_learnable_flags(parser)
    parser.set_defaults(tier="learned_slope", ladder="default")
    parser.add_argument("--config", default="m2")
    parser.add_argument("--identity", choices=IDENTITY_CHOICES, default="full")
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates-smoke", type=int)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.config != "m2":
        parser.error("this runner is the M2 config")
    try:
        ensure_ablation_identity(args.identity)
        ensure_frozen_recipe(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_updates_smoke is not None and args.max_updates_smoke < 1:
        parser.error("smoke counts must be positive")
    if args.dest is None:
        name = f"m2_projadd_{args.identity}_s42"
        args.dest = (RESULTS / "smoke" / name) if args.max_updates_smoke is not None else (RESULTS / name)
    if args.max_updates_smoke is not None and args.device == "cuda:0":
        args.device = "cpu"
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
