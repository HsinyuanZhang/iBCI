#!/usr/bin/env python3
"""H1 R300 learnable-recency IDENTITY-ABLATION trainer (ACTIVITY_ONLY / NORM_ONLY).

New file; does not edit h1_learnable_train.py (the FULL arm) or any signed-state
owner.  Everything except the calibration-bank identity is byte-for-byte the
h1_learnable_train.py recipe: RiftDecoder(task=h1, proj_dim=16) + proj_add,
learned_slope tier, default ladder, seed 42, 32x731 updates, EMA 0.9995,
unit-dropout 0.1, dense attention, HO-M3 grouped-seven selection.

Identity semantics (docs/NORM_ONLY_ZERO_BASELINE_GUIDE_20260910.md sections
2/3, adapted to this proj_add learnable arm):

  activity_only  E0 = C2 materializer(activity prefix, zero side carrier) --
                 the same zero-T rematerialization law as the official
                 h1_signed_state_r300_ablation_v1 ACTIVITY_ONLY arm; T = 0.
  norm_only      E0[u, :] = z[u] broadcast to e0_dim = 700 columns, with
                 z[u] = (rate_sess[u] - mu_src[u]) / max(sigma_src[u], 1e-6);
                 T = 0; no encoder pathway at all.
                 rate_sess is the pooled firing rate (total counts / total
                 bins) over the SAME label-free support tensor the
                 activity-only E0 consumes: per-bank ``block7[:budget]``
                 interpolated trials at train time, the frozen C2 M3 payload
                 activity [3, 1024, 176] at HO score time.
                 mu_src/sigma_src are per-unit mean/std (ddof=0) of the same
                 statistic over the 13 train sessions' earliest-three native
                 eval-valid trials (the M3 support convention this pipeline
                 already uses for its frozen source plans).  They are computed
                 once at train time, sealed into run_meta.json, and only read
                 back at score time -- HO data never touches them.

H1 banks are fixed-width [176] with all-true unit masks, so the guide's
padding-row law is vacuous here; the single source slot with sigma = 0 also
carries rate 0 everywhere (verified on train records and every HO payload
face), so the eps floor leaves its z at exactly 0.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
FLAT = ROOT / "scripts/recency_flat_ablation_v1/h1_flat_train.py"
for path in (PKG / "src", ROOT / "src", ROOT.parent / "btransform_unified_v1/src", ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _prebind_h1_pilot_loader() -> None:
    """Bind ``src`` to SPINT-main before the flat/signed chain imports b2.

    ``btransform_unified_2``'s own package import can bind a foreign top-level
    ``src`` package (streaming_calibration_exp) when it loads first in a shared
    process (pytest collection order).  The signed-state convention keeps
    SPINT-main in front so ``src.data.h1_m4_eb_pilot`` resolves; this prebind
    applies the same law before ``h1_flat_train`` executes.
    """
    if "src.data.h1_m4_eb_pilot" in sys.modules:
        return
    for name in [key for key in sys.modules if key == "src" or key.startswith("src.")]:
        del sys.modules[name]
    spint_main = ROOT.parent / "SPINT-main"
    if str(spint_main) not in sys.path:
        sys.path.insert(0, str(spint_main))
    import src.data.h1_m4_eb_pilot  # noqa: F401  (binding is the point)


_prebind_h1_pilot_loader()

spec = importlib.util.spec_from_file_location("_h1_flat_h1_ablation", FLAT)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot privately load h1_flat_train.py")
h1_flat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h1_flat)
signed, ht = h1_flat.signed, h1_flat.ht
torch, nn = h1_flat.torch, h1_flat.nn

from learnable_recency_v1.config import (  # noqa: E402
    add_learnable_flags,
    config_from_args,
    config_from_run_meta,
    dataset_config,
)
from learnable_recency_v1.wrap import (  # noqa: E402
    LearnableRiftStreamDecoder,
    apply_group_lrs,
    assert_shared_byte_equal,
    install_temporal,
    new_parameter_names,
    recency_bias_snapshot,
    split_optimizer_parameters,
    trainable_new_parameter_count,
)

EPOCHS, UPDATES, SEED, BATCH, MICRO = h1_flat.EPOCHS, h1_flat.UPDATES, h1_flat.SEED, h1_flat.BATCH, h1_flat.MICRO
REF_BANKS = h1_flat.REF_BANKS
RESULTS = PKG / "results"

H1_UNITS = 176
E0_DIM = 700
CARRIER_DIM = 4
SIGMA_FLOOR = 1e-6
SOURCE_SUPPORT_TRIALS = 3
IDENTITIES = ("activity_only", "norm_only")

h1_profiles = signed.h1_profiles
cal1_b2 = signed.cal1_b2
array_sha256 = signed.array_sha256


# ---------------------------------------------------------------------------
# Identity transforms (pure, unit-tested; they never mutate shared objects).
# ---------------------------------------------------------------------------

def pooled_support_rate(activity: np.ndarray) -> np.ndarray:
    """rate_sess[u] = total counts / total bins over the label-free support tensor.

    ``activity`` is [trials, bins, units] of cubic-resampled per-bin counts
    (exactly the tensor the activity-only E0 encoder consumes).  Units are
    counts per resampled bin; the bin-seconds factor is constant and cancels
    inside ``z``.
    """
    calib = np.asarray(activity)
    if calib.ndim != 3 or calib.shape[0] == 0 or calib.shape[1] == 0 or calib.shape[2] == 0:
        raise ValueError(f"support tensor must be non-degenerate [trials,bins,units], got {calib.shape}")
    values = calib.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise RuntimeError("support tensor carries nonfinite values")
    rate = values.sum(axis=(0, 1)) / float(calib.shape[0] * calib.shape[1])
    if not np.isfinite(rate).all():
        raise RuntimeError("pooled support rate is nonfinite")
    return rate


def fit_source_rate_stats(
    source_rates: Mapping[str, np.ndarray] | Sequence[np.ndarray],
    *,
    sigma_floor: float = SIGMA_FLOOR,
) -> dict[str, Any]:
    """Per-unit mu_src/sigma_src over the train sessions' support rates.

    Population std (ddof=0).  The eps floor is applied downstream (inside
    ``norm_only_identity``) so this stays a pure statistic of the source rows.
    """
    rows = [np.asarray(rate, np.float64) for rate in (source_rates.values() if isinstance(source_rates, Mapping) else source_rates)]
    if not rows:
        raise ValueError("source rate rows must be nonempty")
    widths = {int(rate.shape[0]) for rate in rows}
    if len(widths) != 1 or rows[0].ndim != 1:
        raise ValueError(f"H1 source sessions must share one fixed 1-D width, got shapes {[rate.shape for rate in rows]}")
    matrix = np.stack(rows, axis=0)
    if not np.isfinite(matrix).all():
        raise RuntimeError("source rate matrix is nonfinite")
    mu = matrix.mean(axis=0)
    sigma = matrix.std(axis=0, ddof=0)
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise RuntimeError("source rate statistics are nonfinite")
    return {
        "n_source_sessions": int(matrix.shape[0]),
        "mu_src": np.ascontiguousarray(mu),
        "sigma_src": np.ascontiguousarray(sigma),
        "sigma_raw_min": float(sigma.min()),
        "n_sigma_floored": int((sigma < float(sigma_floor)).sum()),
        "sigma_floor": float(sigma_floor),
        "mu_sha256": array_sha256(np.ascontiguousarray(mu)),
        "sigma_sha256": array_sha256(np.ascontiguousarray(sigma)),
    }


def norm_only_identity(
    activity: np.ndarray,
    mu_src: np.ndarray,
    sigma_src: np.ndarray,
    *,
    e0_dim: int = E0_DIM,
    n_units: int = H1_UNITS,
    sigma_floor: float = SIGMA_FLOOR,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """E0[u, :] = z[u] broadcast; carrier all-zero.  No encoder pathway.

    Returns ``(e0 float32 [n_units, e0_dim], carrier float32 [n_units, 4],
    info)``.  ``info`` carries the pooled rate, z, and fail-closed summaries.
    """
    rate = pooled_support_rate(activity)
    if rate.shape != (int(n_units),):
        raise RuntimeError(f"support tensor width {rate.shape[0]} != H1 fixed bank width {int(n_units)}")
    mu = np.asarray(mu_src, np.float64)
    sigma = np.asarray(sigma_src, np.float64)
    if mu.shape != (int(n_units),) or sigma.shape != (int(n_units),):
        raise RuntimeError(f"source statistics must be [{int(n_units)},], got {mu.shape}/{sigma.shape}")
    z = (rate - mu) / np.maximum(sigma, float(sigma_floor))
    if not np.isfinite(z).all():
        raise RuntimeError("NORM_ONLY rate deviation is nonfinite")
    e0 = np.ascontiguousarray(np.repeat(z[:, None], int(e0_dim), axis=1), np.float32)
    carrier = np.zeros((int(n_units), CARRIER_DIM), np.float32)
    info = {
        "rate": rate,
        "z": z,
        "max_abs_z": float(np.abs(z).max()),
        "mean_abs_z": float(np.abs(z).mean()),
        "n_sigma_floored_slots": int((sigma < float(sigma_floor)).sum()),
    }
    return e0, carrier, info


def activity_only_identity(
    activity: np.ndarray,
    materialize: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    n_units: int = H1_UNITS,
) -> tuple[np.ndarray, np.ndarray]:
    """E0 via the frozen C2 encoder with a literal zero side carrier; T = 0.

    ``materialize`` defaults to the b2 materializer at call sites; it is a
    parameter so unit tests can verify the zero-side-input law without the
    frozen checkpoint.  Fail-closed: the returned carrier must be literal
    zero.
    """
    if activity.shape[2] != int(n_units):
        raise RuntimeError(f"support tensor width {activity.shape[2]} != H1 fixed bank width {int(n_units)}")
    zero_carrier = np.zeros((int(n_units), CARRIER_DIM), np.float32)
    e0, carrier = materialize(np.ascontiguousarray(activity, np.float32), zero_carrier)
    e0 = np.ascontiguousarray(e0, np.float32)
    carrier = np.ascontiguousarray(carrier, np.float32)
    if carrier.shape != zero_carrier.shape or not np.array_equal(carrier, np.zeros_like(zero_carrier)):
        raise RuntimeError("ACTIVITY_ONLY materializer changed the literal zero carrier")
    if e0.shape != (int(n_units), E0_DIM) or not np.isfinite(e0).all():
        raise RuntimeError(f"ACTIVITY_ONLY E0 shape/finiteness drift: {e0.shape}")
    return e0, carrier


def replace_bank_identity(
    base: Any,
    e0: np.ndarray,
    carrier: np.ndarray,
    meta_extra: Mapping[str, Any],
) -> Any:
    """Fresh TaskBank with the ablated identity; base objects are never touched.

    ``base`` is a frozen dataclass; ``dataclasses.replace`` builds a new bank
    whose E0/carrier are the freshly materialized arrays above and whose
    calibration_meta is a fresh dict (update on the copy, never on the base).
    """
    meta = dict(base.calibration_meta)
    meta.update(
        {
            "array_sha256": array_sha256(e0),
            "carrier_sha256": array_sha256(carrier),
        }
    )
    meta.update(dict(meta_extra))
    return dataclasses.replace(base, E0=e0, carrier=carrier, calibration_meta=meta)


# ---------------------------------------------------------------------------
# Ablated bank builders (train cal1 + HO face); mirror signed.build_cal1_signed
# / build_ho_signed exactly except for the identity swap.
# ---------------------------------------------------------------------------

def _load_train_records(train_banks: Mapping[str, Any]) -> dict[str, Any]:
    legacy = h1_profiles._legacy()
    paths = legacy.index_heldin_calib(signed.b2.DATA_ROOT)
    return {session: legacy.load_record(paths[session]) for session in train_banks}


def fit_h1_source_rate_stats(records: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Source support rates over each train session's earliest-three trials."""
    legacy = h1_profiles._legacy()
    source_rates: dict[str, np.ndarray] = {}
    for session, record in records.items():
        values = [float(value) for value in record.trial_values[:SOURCE_SUPPORT_TRIALS]]
        activity = np.stack([legacy.interpolate_trial_identity(record, value) for value in values], axis=0)
        source_rates[session] = pooled_support_rate(activity)
    stats = fit_source_rate_stats(source_rates)
    stats["source_sessions"] = [str(name) for name in source_rates]
    stats["support"] = (
        f"earliest {SOURCE_SUPPORT_TRIALS} native eval-valid trials per train session "
        "(M3 convention; the same support family as the HO payload face)"
    )
    stats["rate_units"] = "counts per cubic-resampled bin; the bin-seconds factor cancels inside z"
    return stats, source_rates


def build_cal1_ablation(train_banks: Mapping[str, Any], identity: str) -> dict[str, Any]:
    if identity not in IDENTITIES:
        raise ValueError(f"identity must be one of {IDENTITIES}, got {identity!r}")
    legacy = h1_profiles._legacy()
    inventory = cal1_b2.load_v1_inventory()
    records = _load_train_records(train_banks)
    norm_stats: dict[str, Any] | None = None
    if identity == "norm_only":
        norm_stats, _source_rates = fit_h1_source_rate_stats(records)
    banks: dict[tuple[str, int, int], Any] = {}
    n_trials = {session: len(records[session].trial_values) for session in train_banks}
    starts = {session: tuple(inventory["starts"][session]) for session in train_banks}
    z_diagnostic: dict[str, dict[str, float]] = {}
    for session, base in train_banks.items():
        record = records[session]
        values = [float(value) for value in record.trial_values]
        for start in starts[session]:
            block7 = values[start : start + 7]
            if len(tuple(values[start : start + 4])) < 4:
                continue
            for budget in signed.C2_CYCLE:
                if start + int(budget) > n_trials[session]:
                    continue
                activity = np.stack(
                    [legacy.interpolate_trial_identity(record, value) for value in block7[: int(budget)]],
                    axis=0,
                )
                if identity == "activity_only":
                    e0, carrier = activity_only_identity(activity, signed.b2._materialize_e0)
                    estimator = "C2 activity prefix rematerialized with literal zero side carrier (ACTIVITY_ONLY)"
                else:
                    e0, carrier, info = norm_only_identity(
                        activity, norm_stats["mu_src"], norm_stats["sigma_src"]
                    )
                    estimator = "pooled support-rate z broadcast; no encoder pathway (NORM_ONLY)"
                    z_diagnostic[f"{session}|{start}|{budget}"] = {
                        "max_abs_z": info["max_abs_z"],
                        "mean_abs_z": info["mean_abs_z"],
                    }
                bank = replace_bank_identity(
                    base,
                    e0,
                    carrier,
                    {
                        "identity_ablation": identity,
                        "budget": int(budget),
                        "m7_start": int(start),
                        "trial_count": int(budget),
                        "estimator": estimator,
                    },
                )
                banks[(session, int(start), int(budget))] = bank
        print(f"[identity-ablation cal1] banks {session} n_starts={len(starts[session])}", flush=True)
    return {
        "banks": banks,
        "starts": starts,
        "n_trials": n_trials,
        "s_src": None,
        "plan_q": None,
        "plan_lambda": None,
        "carrier": f"ablation_{identity}",
        "identity": identity,
        "norm_stats": norm_stats,
        "z_diagnostic": z_diagnostic,
    }


def build_ho_ablation(
    context_bins: int,
    identity: str,
    norm_stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    if identity not in IDENTITIES:
        raise ValueError(f"identity must be one of {IDENTITIES}, got {identity!r}")
    if identity == "norm_only" and norm_stats is None:
        raise RuntimeError("norm_only HO face requires the frozen source statistics from run_meta")
    banks = {}
    xs = {}
    vm = {}
    ys = {}
    keys = []
    for session, key in signed.HELDOUT_SESSION_TO_FALCON_KEY:
        path = signed.b2.HO_DIR / f"sub-HumanPitt-held-out-calib_{session}.nwb"
        neural, velocity, _change, eval_mask = load_nwb(path, FalconTask.h1)
        ends = np.flatnonzero(np.asarray(eval_mask, bool)).astype(np.int64)
        x, m = ht.endpoint_context(np.asarray(neural, np.float32), ends, context_bins)
        activity, _full_carrier = signed.adapters._h1_payload_arrays(session)
        if identity == "activity_only":
            e0, carrier = activity_only_identity(activity, signed.b2._materialize_e0)
            estimator = "C2 HO-M3 payload activity rematerialized with literal zero side carrier (ACTIVITY_ONLY)"
        else:
            e0, carrier, _info = norm_only_identity(
                activity, np.asarray(norm_stats["mu_src"], np.float64), np.asarray(norm_stats["sigma_src"], np.float64)
            )
            estimator = "C2 HO-M3 payload pooled-rate z broadcast against frozen source stats (NORM_ONLY)"
        banks[key] = signed.TaskBank(
            session,
            e0,
            carrier,
            np.ones(e0.shape[0], bool),
            x,
            np.asarray(velocity, np.float32)[ends],
            ends,
            {
                "shape": tuple(e0.shape),
                "trial_count": 3,
                "budget": 3,
                "identity_ablation": identity,
                "estimator": estimator,
                "array_sha256": array_sha256(e0),
                "carrier_sha256": array_sha256(carrier),
            },
        )
        xs[key], vm[key], ys[key] = x, m, banks[key].target_store
        keys.append(key)
    return {"context_bins": context_bins, "banks": banks, "X": xs, "valid": vm, "y": ys, "keys": keys}


def load_norm_stats_from_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Read the frozen source statistics back from run_meta (score-time law)."""
    block = meta.get("norm_only_source")
    if not isinstance(block, Mapping):
        raise RuntimeError("run_meta is missing the frozen norm_only source statistics")
    mu = np.ascontiguousarray(np.asarray(block["mu_src"], np.float64))
    sigma = np.ascontiguousarray(np.asarray(block["sigma_src"], np.float64))
    if array_sha256(mu) != block.get("mu_sha256") or array_sha256(sigma) != block.get("sigma_sha256"):
        raise RuntimeError("run_meta norm_only source statistics digest drift")
    return {"mu_src": mu, "sigma_src": sigma, "sigma_floor": float(block.get("sigma_floor", SIGMA_FLOOR))}


def norm_stats_meta_block(stats: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-ready copy of the frozen source statistics (exact float64 mirror)."""
    mu = np.asarray(stats["mu_src"], np.float64)
    sigma = np.asarray(stats["sigma_src"], np.float64)
    return {
        "mu_src": mu.tolist(),
        "sigma_src": sigma.tolist(),
        "mu_sha256": array_sha256(np.ascontiguousarray(mu)),
        "sigma_sha256": array_sha256(np.ascontiguousarray(sigma)),
        "sigma_floor": float(stats["sigma_floor"]),
        "n_sigma_floored": int(stats["n_sigma_floored"]),
        "sigma_raw_min": float(stats["sigma_raw_min"]),
        "n_source_sessions": int(stats["n_source_sessions"]),
        "source_sessions": list(stats["source_sessions"]),
        "support": stats["support"],
        "rate_units": stats["rate_units"],
        "formula": "z[u] = (rate_sess[u] - mu_src[u]) / max(sigma_src[u], sigma_floor); E0 row = z broadcast",
        "score_time_law": "mu/sigma are read back from this block only; HO data never recomputes them",
    }


# ---------------------------------------------------------------------------
# Model / parity / scoring plumbing: verbatim from h1_learnable_train.py.
# ---------------------------------------------------------------------------

def sha_file(path: Path) -> str:
    return h1_flat.sha_file(path)


def atomic(path: Path, value: Any) -> None:
    h1_flat.atomic(path, value)


def learnable_decoder(device: torch.device, recency_cfg, backend: str = "dense"):
    model = ht._decoder("recency", device, 300, backend)
    install_temporal(model, recency_cfg, SEED)
    model.temporal.set_attention_backend(backend)
    return model


def shared_sha(model: Any, names: list[str]) -> str:
    digest = hashlib.sha256()
    named = dict(model.named_parameters())
    for name in sorted(names):
        digest.update(name.encode())
        digest.update(named[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def assert_paired(model: Any, device: torch.device, reference: dict[str, Any], recency_cfg) -> dict[str, Any]:
    stock = ht._decoder("recency", device, 300, "dense")
    if recency_cfg.layers == 4:
        recency_sha = ht._sha_state(stock)
        if recency_sha != reference["reference_initialization_sha256"]:
            raise RuntimeError("fresh recency init SHA differs from current signed-state reference")
        shared = assert_shared_byte_equal(model, stock)
        if shared_sha(model, shared) != recency_sha:
            raise RuntimeError("learnable shared named parameters are not paired to recency")
    else:
        shared_cfg = dataset_config("h1", tier="fixed", layers=recency_cfg.layers, ladder="default")
        install_temporal(stock, shared_cfg, SEED)
        stock.temporal.set_attention_backend("dense")
        shared = assert_shared_byte_equal(model, stock)
        recency_sha = shared_sha(model, shared)
    extra = new_parameter_names(model)
    ladder_cfg = dataset_config(
        "h1", tier="fixed", layers=recency_cfg.layers, half_life_seconds=recency_cfg.half_life_seconds
    )
    ladder_ref = ht._decoder("recency", device, 300, "dense")
    install_temporal(ladder_ref, ladder_cfg, SEED)
    ladder_ref.temporal.set_attention_backend("dense")
    z = torch.randn(1, 32, 256, device=device)
    mask = torch.ones(1, 32, dtype=torch.bool, device=device)
    with torch.inference_mode():
        left = model.temporal(z, mask)
        right = ladder_ref.temporal(z, mask)
    if not torch.allclose(left, right, atol=1e-6, rtol=1e-6):
        raise RuntimeError("H1 learnable/same-ladder init forward drift")
    del stock, ladder_ref
    return {
        "shared_parameter_names": shared,
        "new_parameter_names": extra,
        "new_parameter_counts": {name: int(dict(model.named_parameters())[name].numel()) for name in extra},
        "trainable_new_parameter_count": trainable_new_parameter_count(model),
        "shared_initialization_sha256": recency_sha,
        "init_forward_max_abs": float((left - right).abs().max().cpu()),
        "ladder": recency_cfg.ladder_metadata(),
    }


def runtime_parity(model: Any, train_data: dict[str, Any], cal1: dict[str, Any], device: Any, label: str) -> dict[str, Any]:
    session = train_data["sessions"][0]
    budget = int(signed.prefix_schedule(0, UPDATES)[0])
    starts = signed.cal1_b2.legal_starts(cal1["starts"][session], cal1["n_trials"][session], budget)
    bank = cal1["banks"][(session, signed.pick_m7_start(session, epoch0=0, step=0, starts=starts), budget)]
    xb = torch.from_numpy(train_data["X"][session][:1]).to(device)
    valid = torch.from_numpy(train_data["valid"][session][:1]).to(device)
    was = model.training
    model.eval()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    try:
        model.temporal.set_attention_backend("dense")
        with torch.inference_mode():
            dense = ht._forward(model, xb, bank, None, valid)
        model.temporal.set_attention_backend("local")
        with torch.inference_mode():
            local = ht._forward(model, xb, bank, None, valid)
        stream = LearnableRiftStreamDecoder(model)
        for index in range(xb.shape[1]):
            stream.stream_step(xb[:, index], bank, ["ablation-parity"], valid_mask=valid[:, index])
        stream_last = stream.predict("ablation-parity")
        if not torch.allclose(dense, local, rtol=2e-4, atol=2e-5) or not torch.allclose(local[0], stream_last, rtol=2e-4, atol=2e-5):
            raise RuntimeError(f"{label}: dense/local/stream prefix parity drift")
        return {
            "label": label,
            "prefix_bins": int(xb.shape[1]),
            "valid_bins": int(valid.sum().item()),
            "dense_local_max_abs": float((dense - local).abs().max().cpu()),
            "local_stream_max_abs": float((local[0] - stream_last).abs().max().cpu()),
        }
    finally:
        model.temporal.set_attention_backend("dense")
        model.train(was)
        with torch.no_grad():
            for name, value in before.items():
                dict(model.named_parameters())[name].copy_(value)


def bias_from_prefix(model: Any, train_data: dict[str, Any], device: Any) -> dict[str, Any]:
    session = train_data["sessions"][0]
    xb = torch.from_numpy(train_data["X"][session][:1]).to(device)
    valid = torch.from_numpy(train_data["valid"][session][:1]).to(device)
    was = model.training
    model.eval()
    with torch.inference_mode():
        bank = next(iter(train_data["banks"].values()))
        tokens = model.frontend_tokens(xb, bank)
        stats = recency_bias_snapshot(model.temporal, tokens, valid)
    model.train(was)
    return stats


# ---------------------------------------------------------------------------
# Train / score stages.
# ---------------------------------------------------------------------------

def score(args: Any, recency_cfg) -> dict[str, Any]:
    torch.set_num_threads(2)
    dest = args.dest
    meta = json.loads((dest / "run_meta.json").read_text())
    identity = meta.get("identity")
    if identity != args.identity:
        raise RuntimeError(f"run_meta identity {identity!r} does not match --identity {args.identity!r}")
    reference = h1_flat.validate_reference(args.banks)
    if meta.get("status") != "FORMAL" or meta.get("schema") != "h1_signedstate14_learnable_ablation_v1":
        raise RuntimeError("dest is not a matching formal H1 identity-ablation run")
    if meta.get("banks_receipt_sha256") != sha_file(args.banks / "receipt.json"):
        raise RuntimeError("score-stage banks receipt drift")
    device = torch.device(args.device)
    recency_cfg = config_from_run_meta(meta, "h1")
    model = learnable_decoder(device, recency_cfg, meta.get("attention_backend", "dense"))
    ema = signed.DecoderEMA(model, decay=0.9995)
    norm_stats = load_norm_stats_from_meta(meta) if identity == "norm_only" else None
    ho = build_ho_ablation(300, identity, norm_stats)
    curve = []
    for epoch in range(1, EPOCHS + 1):
        path = dest / f"epoch_{epoch:03d}.pt"
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["raw_state_dict"], strict=True)
        ema.load_state_dict(state["ema"])
        report = signed.score_ho_m3(model, ema, ho, device)
        curve.append(
            {
                "epoch": epoch,
                "epoch_zero_based": epoch - 1,
                signed.HO_SELECTION_METRIC: report["r2_mean"],
                "worst_session_r2": report["worst_session_r2"],
                "session_std_population": report["r2_std_population"],
                "per_session_r2": report["per_session_r2"],
            }
        )
        print(
            f"[identity-ablation ho-m3 {identity}] ep={epoch}/{EPOCHS} mean={report['r2_mean']:.6f} worst={report['worst_session_r2']:.6f}",
            flush=True,
        )
    selected = signed.select_epoch(curve)
    atomic(
        dest / "ho_m3_selection.json",
        {
            "status": "HO_M3_DEVELOPMENT_SELECTION",
            "identity_ablation": identity,
            "selected": selected,
            "curve": curve,
            "selection_rule": "same current HO-M3 grouped-seven, all32, earliest maximum",
        },
    )
    atomic(
        dest / "train_receipt.json",
        {
            "status": "COMPLETED",
            "identity_ablation": identity,
            "epochs": EPOCHS,
            "updates": EPOCHS * UPDATES,
            "selected_epoch": selected["epoch"],
            "official_test_used": False,
            "reference": reference,
        },
    )
    return {"status": "SCORE_COMPLETED", "identity_ablation": identity, "selected_epoch": selected["epoch"]}


def train(args: Any) -> dict[str, Any]:
    recency_cfg = config_from_args(args, "h1")
    reference = h1_flat.validate_reference(args.banks)
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    dest = args.dest
    if dest.exists() and any(dest.iterdir()) and not args.resume:
        raise FileExistsError(f"fresh destination required: {dest}")
    train_data = ht.build_train(300)
    cal1 = build_cal1_ablation(train_data["banks"], args.identity)
    digests = ht._digest_train_contract(train_data, cal1)
    ref = reference["reference_pairing_digests"]
    if digests["endpoint_inventory_sha256"] != ref["endpoint_inventory_sha256"] or digests["valid_mask_inventory_sha256"] != ref["valid_mask_inventory_sha256"]:
        raise RuntimeError("ablation raw endpoint/valid-mask contract differs from current recency reference")
    if digests["bank_roster_sha256"] == ref["bank_roster_sha256"]:
        raise RuntimeError("ablation bank roster digest unexpectedly equals the FULL reference; identity transform did not apply")
    model = learnable_decoder(device, recency_cfg, "dense")
    init = assert_paired(model, device, reference, recency_cfg)
    ema = signed.DecoderEMA(model, decay=0.9995)
    groups = split_optimizer_parameters(model, peak_lr=1e-4, weight_decay=0.01, lr_multiplier=recency_cfg.lr_multiplier)
    opt = torch.optim.AdamW(groups, lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    rng = np.random.default_rng(SEED)
    dest.mkdir(parents=True, exist_ok=True)
    smoke = args.max_updates_smoke is not None
    meta = {
        "schema": "h1_signedstate14_learnable_ablation_v1",
        "cell": f"CURRENT_H1_SIGNEDSTATE14_R300_LEARNABLE_ABLATION_{args.identity.upper()}_S42",
        "status": "SMOKE" if smoke else "FORMAL",
        "smoke": smoke,
        "identity": args.identity,
        "identity_ablation": {
            "e0": (
                "C2 encoder of the day's activity prefix with literal zero side carrier"
                if args.identity == "activity_only"
                else "per-session pooled support firing rate z-broadcast to 700 columns (no encoder)"
            ),
            "carrier": "all-zero [176,4] literal at train and score time",
            "input": "raw neural contexts unchanged",
            "nesting": "NONE < NORM_ONLY < ACTIVITY_ONLY < FULL (guide section 2)",
        },
        "tier": recency_cfg.tier,
        "learnable_config": recency_cfg.__dict__,
        "new_parameter_names": init["new_parameter_names"],
        "new_parameter_counts": init["new_parameter_counts"],
        "trainable_new_parameter_count": init["trainable_new_parameter_count"],
        "context_bins": 300,
        "layer_windows": list(recency_cfg.temporal_config.windows),
        "depth": recency_cfg.layers,
        "ladder": recency_cfg.ladder_metadata(),
        "attention_backend": "dense",
        "epochs": EPOCHS if not smoke else 1,
        "updates_per_epoch": UPDATES,
        "batch": BATCH,
        "microbatch": MICRO,
        "seed": SEED,
        "ema": 0.9995,
        "unit_dropout": 0.1,
        "initialization_pairing": init,
        "banks_receipt_sha256": sha_file(args.banks / "receipt.json"),
        "pairing_digests": digests,
        "pairing_digest_law": (
            "endpoint/valid-mask digests must equal the FULL reference (raw data unchanged); "
            "the bank roster digest must differ (identity transform applied by design)"
        ),
        "reference": reference,
        "official_test_used": False,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.identity == "norm_only":
        meta["norm_only_source"] = norm_stats_meta_block(cal1["norm_stats"])
        z_values = [row["max_abs_z"] for row in cal1["z_diagnostic"].values()]
        meta["norm_only_bank_z_summary"] = {
            "n_banks": len(cal1["z_diagnostic"]),
            "max_abs_z": float(max(z_values)),
            "mean_max_abs_z": float(np.mean(z_values)),
        }
    atomic(dest / "run_meta.json", meta)
    parity_pre = runtime_parity(model, train_data, cal1, device, "pre-train")
    atomic(dest / "runtime_parity_pre.json", parity_pre)
    total = EPOCHS * UPDATES
    step = 0
    started = time.monotonic()
    last_bias = None
    last_loss = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        schedule = signed.prefix_schedule(epoch - 1, UPDATES)
        order = list(train_data["sessions"])
        rng.shuffle(order)
        si = 0
        for session in order:
            index = rng.permutation(len(train_data["X"][session]))
            for off in range(0, len(index), BATCH):
                take = index[off : off + BATCH]
                budget = int(schedule[si])
                starts = signed.cal1_b2.legal_starts(cal1["starts"][session], cal1["n_trials"][session], budget)
                bank = cal1["banks"][(session, signed.pick_m7_start(session, epoch0=epoch - 1, step=si, starts=starts), budget)]
                step += 1
                apply_group_lrs(opt, signed.warmup_cosine_lr(step, total, UPDATES, peak=1e-4, min_factor=0.1))
                keep = signed.whole_unit_dropout(
                    torch.from_numpy(bank.unit_mask.copy()),
                    p=0.1,
                    generator=torch.Generator().manual_seed(signed.unit_dropout_seed(SEED, epoch, si)),
                )
                opt.zero_grad(set_to_none=True)
                batch_loss = 0.0
                for moff in range(0, len(take), MICRO):
                    subset = take[moff : moff + MICRO]
                    xb = torch.from_numpy(train_data["X"][session][subset]).to(device)
                    valid = torch.from_numpy(train_data["valid"][session][subset]).to(device)
                    yb = torch.from_numpy(train_data["y"][session][subset] * signed.SCALE).to(device)
                    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
                    with amp:
                        loss = nn.functional.mse_loss(ht._forward(model, xb, bank, keep, valid).float(), yb)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"nonfinite loss epoch={epoch} step={step}")
                    (loss * (len(subset) / len(take))).backward()
                    batch_loss += float(loss.detach()) * len(subset) / len(take)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                ema.update_after_step(model)
                losses.append(batch_loss)
                last_loss = batch_loss
                si += 1
                if args.max_updates_smoke and step >= args.max_updates_smoke:
                    break
            if args.max_updates_smoke and step >= args.max_updates_smoke:
                break
        last_bias = bias_from_prefix(model, train_data, device)
        ht._checkpoint(
            dest / f"epoch_{epoch:03d}.pt",
            model,
            opt,
            ema,
            epoch,
            step,
            rng,
            variant=f"learnable_{recency_cfg.tier}_{args.identity}",
            context_bins=300,
            attention_backend="dense",
            microbatch=MICRO,
            smoke=smoke,
        )
        row = {
            "event": "epoch",
            "epoch": epoch,
            "global_step": step,
            "train_mse": float(np.mean(losses)),
            "smoke": smoke,
            "bias": last_bias,
        }
        with (dest / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        atomic(dest / "heartbeat.json", row)
        if smoke:
            break
    train_elapsed = time.monotonic() - started
    parity_post = runtime_parity(model, train_data, cal1, device, "post-train")
    atomic(dest / "runtime_parity_post.json", parity_post)
    if smoke:
        receipt = {
            "schema": "h1_signedstate14_learnable_ablation_smoke_receipt_v1",
            "status": "SMOKE_COMPLETED",
            "identity": args.identity,
            "tier": recency_cfg.tier,
            "steps": step,
            "finite_loss": last_loss is not None and bool(np.isfinite(last_loss)),
            "runtime_parity_pre": parity_pre,
            "runtime_parity_post": parity_post,
            "bias": last_bias,
            "new_parameter_names": init["new_parameter_names"],
            "new_parameter_counts": init["new_parameter_counts"],
            "trainable_new_parameter_count": init["trainable_new_parameter_count"],
            "train_elapsed_seconds": train_elapsed,
        }
        if args.identity == "norm_only":
            receipt["norm_only_bank_z_summary"] = meta["norm_only_bank_z_summary"]
        atomic(dest / "smoke_receipt.json", receipt)
        return receipt
    if step != total:
        raise RuntimeError("formal update total drift")
    return score(args, recency_cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_learnable_flags(parser)
    parser.set_defaults(ladder="default")
    parser.add_argument("--identity", choices=IDENTITIES, required=True)
    parser.add_argument("--config", default="h1")
    parser.add_argument("--banks", type=Path, default=REF_BANKS)
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=("train", "score"), default="train")
    parser.add_argument("--max-updates-smoke", type=int)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--resume", type=Path)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.config != "h1":
        parser.error("this runner is the H1 config")
    if args.smoke_steps is not None:
        args.max_updates_smoke = args.smoke_steps
    recency_cfg = config_from_args(args, "h1")
    args.banks = args.banks.resolve()
    if args.dest is None:
        args.dest = (
            (RESULTS / "smoke" / f"h1_{args.identity}_s42").resolve()
            if args.max_updates_smoke
            else (RESULTS / f"h1_{args.identity}_s42").resolve()
        )
    else:
        args.dest = args.dest.resolve()
    if args.max_updates_smoke is not None and args.device == "cuda:0":
        args.device = "cpu"
    payload = score(args, recency_cfg) if args.stage == "score" else train(args)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
