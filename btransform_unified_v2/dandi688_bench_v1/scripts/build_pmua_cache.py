#!/usr/bin/env python3
"""Build the pseudo-MUA (electrode-pooled) prepared-cache variants.

Variants:
  pmua_t4       channel-level T4 carrier + carrier-side E0 remelt (the
                PMUA counterpart of the SUA t4 arm's cache bytes).
  f_labelfree   label-free ACTIVITY-ONLY cell on the pooled signal (user
                directive 2026-09-09 f-series, "不使用任何标签校准"): carrier
                bytes all-zero and E0 remelted with an all-zero side --
                post_pool(cat(pre_pool(pooled calib).mean, zero side)) --
                the PMUA ladder's ACTIVITY_ONLY rung.  The contract's
                ``variant`` field is exactly "f_labelfree" so the bench
                runner's arm binding (plan.ARM_REQUIRED_CACHE_VARIANT)
                accepts the f_labelfree arm on this cache; the pooled-vs-SUA
                distinction lives in the estimator block.  The channel T4
                refit still runs during the build purely as a pooling-
                pipeline diagnostic -- no label-derived byte enters the
                cache.

Frozen PMUA protocol (sua_exploration/docs/PSEUDO_MUA_T4_BRIDGE_48H.md
section 2), adapted to the bench_v1 prepared-cache geometry:

  1. electrode pooling: pmua[e, t] = sum(sua[u, t] for electrode(u) == e)
     over the SAME NWB units table that backs the frozen SUA cache rows
     (multisession_datamodule.pool_spikes_by_electrode /
     electrode_ids_from_units -- the bridge's frozen implementation);
  2. channel-level T4 refit: the closed-form cosine T4 (R700/H300 windows,
     first-M10 candidate support, delta_b = b_R700 - b_H300) is refit on the
     POOLED calibration-trial firing rates via the same
     dandi688_sparse_event_t4_v1 code path that produced the frozen SUA
     carrier bytes (descriptors.refit_from_single_pool /
     materialize_sparse_event_t4 recipe).  Unit-level T4 averaging is
     FORBIDDEN and never performed;
  3. channel-level E0: E0 = post_pool(cat(pre_pool(pooled calib).mean,
     channel carrier side)) -- the same frozen-B3S remelt formula as
     rift_v1 dandi688_train.build_rows, with the encoder unit axis N now
     the electrode count (pre_pool is a per-channel Linear(trial_length),
     so the channel axis is dimension-agnostic);
  4. column normalization train-only: fit_source_normalizer over the
     --norm-protocol train sessions (default exp2_full = the frozen SUA
     cache's own 27-train-session normalizer domain, keeping the PMUA-vs-SUA
     column comparison apples-to-apples; the 2015-only 18-session domain is
     available via --norm-protocol exp1_poverty/exp1_narrow only if the
     protocol names it).

Correctness gates (fail closed, mirroring the bridge's launch gates):
  - PMUA channel count <= SUA unit count, every electrode >= 1 unit,
    every sorted unit maps exactly one electrode;
  - per-time-bin spike-count conservation: pooled column sums == member
    unit column sums exactly (half-open non-overlapping windows);
  - channel T4 finite, shape [n_electrodes, 4];
  - SUA recipe parity gate: the SUA raw profile recomputed here plus a
    train-only column normalizer must reproduce the frozen cache's SUA
    carrier bytes (max |delta| <= 1e-4), proving the NWB unit axis aligns
    with the frozen cache unit axis before any pooling is trusted;
  - behavior/starts/mask-semantics copied byte-identically; neural pooled
    from the frozen cache's own SUA counts (identical time base).

CPU-only by contract: the B3S student is loaded on --device cpu; no CUDA
initialization, no training, no test data (formal-test sessions are never
named outside plan.FORMAL_TEST_SESSIONS rejection).

Writes a prepared_contract.json that still satisfies run_688_bench
.verify_prepared_cache (same schema/manifest/split counts) plus the variant /
electrode-count / unit-to-electrode-mapping provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
WS = PKG_ROOT.parents[1]
for p in (PKG_ROOT / "src", WS / "btransform_unified_v2" / "src",
          WS / "btransform_unified_v1" / "src", WS / "sua_exploration",
          WS / "streaming_calibration_exp", WS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dandi688_bench_v1 import plan, vstate  # noqa: E402

VARIANTS = ("pmua_t4", "f_labelfree")
N_PAD = 91              # frozen Nmax of the contract-v2 cache (all sessions)
SUA_PARITY_MAX_ABS = 1.0e-4  # same tolerance family as build_carrier_cache's
                              # legacy identity gate (normalized carrier bytes)


def _variant_dest(variant: str) -> Path:
    return PKG_ROOT / "results" / ("cache_pmua_t4" if variant == "pmua_t4"
                                   else "cache_pmua_f_labelfree")


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen(cache: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    meta = json.loads((cache / "prepared_contract.json").read_text())
    if meta.get("schema") != plan.PREPARED_CACHE_SCHEMA:
        raise RuntimeError(f"frozen cache schema drift: {meta.get('schema')}")
    rows: dict[str, dict[str, Any]] = {}
    for name, info in meta["sessions"].items():
        z = np.load(cache / "sessions" / f"{name}.npz")
        rows[name] = {
            "split": info["split"],
            "neural": np.asarray(z["neural"]),
            "behavior": np.asarray(z["behavior"]),
            "starts": np.asarray(z["starts"]),
            "e0": np.asarray(z["e0"]),
            "carrier": np.asarray(z["carrier"]),
            "mask": np.asarray(z["mask"]),
        }
    if len(rows) != plan.SPLIT_COUNTS["train"] + plan.SPLIT_COUNTS["val"]:
        raise RuntimeError(f"frozen cache must hold 33 sessions, got {len(rows)}")
    return meta, rows


def _nwb_path(session_id: str) -> Path:
    from mc_maze.dandi688_sparse_event_t4_v1 import plan as se_plan
    return WS / se_plan.DATA_RELATIVE / f"{session_id}_behavior+ecephys.nwb"


# ---------------------------------------------------------------------------
# pure helpers (unit-tested in tests/test_pmua_cache.py)
# ---------------------------------------------------------------------------
def pool_count_array_by_electrode(
    counts: np.ndarray, electrode_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool a per-unit count array along its LAST axis onto electrodes.

    Works for the [T, units] neural axis and the [K, W, units] calib-trial
    axis alike: the leading axes are flattened, the pooling is the bridge's
    frozen np.add.at sum (multisession_datamodule.pool_spikes_by_electrode,
    deterministic np.unique channel order), and the leading geometry is
    restored.  Returns (pooled, channel_ids, inverse) with ``inverse[u]``
    the channel index of unit u."""
    from mc_maze.multisession_datamodule import pool_spikes_by_electrode

    counts = np.asarray(counts)
    electrode_ids = np.asarray(electrode_ids, dtype=np.int64)
    if counts.ndim < 2:
        raise ValueError(f"expected at least 2 axes, got {counts.shape}")
    n_units = counts.shape[-1]
    if electrode_ids.shape != (n_units,):
        raise ValueError(
            f"electrode_ids must hold one id per unit: {electrode_ids.shape} "
            f"for {n_units} units"
        )
    channel_ids, inverse = np.unique(electrode_ids, return_inverse=True)
    flat = counts.reshape(-1, n_units)
    pooled_flat, pooled_channel_ids = pool_spikes_by_electrode(flat, electrode_ids)
    if not np.array_equal(channel_ids, pooled_channel_ids):
        raise RuntimeError("channel id order drift between unique/pool")
    return (
        pooled_flat.reshape(*counts.shape[:-1], channel_ids.size),
        channel_ids,
        inverse,
    )


def assert_pool_conservation(
    pooled: np.ndarray,
    source: np.ndarray,
    inverse: np.ndarray,
    n_channels: int,
) -> None:
    """Per-channel spike-count conservation law of the bridge gates.

    Every channel's total pooled count must equal the sum of its member
    units' totals EXACTLY (half-open non-overlapping windows on one grid)."""
    n_units = source.shape[-1]
    source_totals = np.asarray(source, dtype=np.float64).reshape(-1, n_units).sum(axis=0)
    pooled_totals = np.asarray(pooled, dtype=np.float64).reshape(-1, n_channels).sum(axis=0)
    member_totals = np.bincount(inverse, weights=source_totals, minlength=n_channels)
    if not np.array_equal(pooled_totals, member_totals):
        raise RuntimeError(
            "per-channel pool conservation failed: max_abs="
            f"{np.max(np.abs(pooled_totals - member_totals))}"
        )


def sua_vs_pmua_column_correlation(
    sua_raw: np.ndarray, pmua_raw: np.ndarray, inverse: np.ndarray
) -> dict[str, float]:
    """Column correlation of the channel T4 against its member units' T4.

    For direction columns a/c: Pearson r between the pooled channel value and
    the mean of its member units' values (same-electrode tuning agreement --
    the acceptance diagnostic of the cache build; high r expected because
    pooled counts carry the member units' tuning).  Also reports the mean
    absolute preferred-direction angle discrepancy (degrees) between the
    pooled channel and the member circular mean."""
    n_channels = pmua_raw.shape[0]
    member_mean = np.stack(
        [sua_raw[inverse == channel].mean(axis=0) for channel in range(n_channels)]
    )

    def _pearson(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        xm, ym = x - x.mean(), y - y.mean()
        denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
        if denom <= 0.0:
            return float("nan")
        return float((xm * ym).sum() / denom)

    angle_pooled = np.arctan2(pmua_raw[:, 1], pmua_raw[:, 0])
    # member circular mean weighted 1/unit (unit counts per electrode vary)
    angles = [np.arctan2(sua_raw[inverse == ch, 1], sua_raw[inverse == ch, 0])
              for ch in range(n_channels)]
    weights = [np.full(angles[ch].shape, 1.0 / angles[ch].size) for ch in range(n_channels)]
    member_angle = np.asarray([
        np.arctan2((np.sin(a) * w).sum(), (np.cos(a) * w).sum())
        for a, w in zip(angles, weights)
    ])
    delta = np.abs((angle_pooled - member_angle + np.pi) % (2.0 * np.pi) - np.pi)
    return {
        "pearson_r_col_a": _pearson(pmua_raw[:, 0], member_mean[:, 0]),
        "pearson_r_col_c": _pearson(pmua_raw[:, 1], member_mean[:, 1]),
        "pearson_r_col_m": _pearson(pmua_raw[:, 2], member_mean[:, 2]),
        "mean_abs_angle_delta_deg": float(np.degrees(delta.mean())),
    }


# ---------------------------------------------------------------------------
# worker: one session's NWB-derived T4 profiles (SUA + pooled), read once
# ---------------------------------------------------------------------------
def _session_profiles(session_id: str) -> dict[str, Any]:
    """Compute SUA and electrode-pooled raw T4 profiles for one session.

    One trial listing + one pooled-rate read + one electrode-table read of
    the NWB; the cosine fit then runs on slices of the same rate matrix
    (refit_from_single_pool), so the PMUA profile is EXACTLY the frozen
    bridge's channel-level refit and the SUA profile is exactly the frozen
    cache carrier's pre-normalization recipe."""
    from mc_maze.dandi688_sparse_event_t4_v1.descriptors import (
        _default_electrode_ids,
        _pool_trial_rate_matrix,
        refit_from_single_pool,
        single_pool_interval_layout,
    )
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.dandi688_sparse_event_t4_v1 import plan as se_plan

    nwb_path = _nwb_path(session_id)
    trials = list_datamodule_rewarded_trials(
        nwb_path, bin_size_ms=se_plan.BIN_SIZE_MS, window_size=se_plan.WINDOW_SIZE_BINS,
        trial_result_filter=se_plan.REWARDED_RESULT,
    )
    intervals, slices = single_pool_interval_layout(trials, groups=("candidate",))
    rates, n_units = _pool_trial_rate_matrix(nwb_path, intervals)
    electrode_ids = _default_electrode_ids(nwb_path)
    if electrode_ids.shape != (n_units,):
        raise RuntimeError(
            f"{session_id}: electrode ids {electrode_ids.shape} != units {n_units}"
        )
    _whole_sua, _post_sua, raw_sua = refit_from_single_pool(
        rates, trials, slices, group="candidate", signal_view="sua"
    )
    _whole_pm, _post_pm, raw_pmua = refit_from_single_pool(
        rates, trials, slices, group="candidate",
        signal_view="pseudo_mua", electrode_ids=electrode_ids,
    )
    if raw_pmua.shape[0] > n_units:
        raise RuntimeError(
            f"{session_id}: PMUA channels {raw_pmua.shape[0]} > units {n_units}"
        )
    counts_per_channel = np.bincount(
        np.unique(electrode_ids, return_inverse=True)[1], minlength=raw_pmua.shape[0]
    )
    if int(counts_per_channel.min()) < 1:
        raise RuntimeError(f"{session_id}: electrode with zero units")
    if not (np.isfinite(raw_sua).all() and np.isfinite(raw_pmua).all()):
        raise RuntimeError(f"{session_id}: nonfinite raw T4 profile")
    # first ACTIVITY_SUPPORT(30) rewarded trials' bin spans, the exact trial
    # set the SUA datamodule builds its calib_trials from (same filter loop)
    calib_trial_bins = [
        (int(row["start"]), int(row["stop"])) for row in trials[:30]
    ]
    if len(calib_trial_bins) != 30:
        raise RuntimeError(f"{session_id}: expected 30 calib trials, got {len(calib_trial_bins)}")
    return {
        "session_id": session_id,
        "n_units": int(n_units),
        "n_electrodes": int(raw_pmua.shape[0]),
        "electrode_ids": electrode_ids,
        "raw_sua": raw_sua,
        "raw_pmua": raw_pmua,
        "units_per_electrode": counts_per_channel,
        "calib_trial_bins": calib_trial_bins,
    }


def _worker(session_id: str) -> dict[str, Any]:
    return _session_profiles(session_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=VARIANTS, default="pmua_t4")
    parser.add_argument("--dest", type=Path, default=None,
                        help="destination dir (default results/cache_pmua_t4 "
                             "or results/cache_pmua_f_labelfree per variant)")
    parser.add_argument("--source-cache", type=Path, default=plan.prepared_cache_path())
    parser.add_argument("--norm-protocol", choices=tuple(plan.PROTOCOLS),
                        default="exp2_full",
                        help="train-session set that fits the channel-T4 column "
                             "normalizer (default exp2_full = all 27 train "
                             "sessions, the frozen SUA cache's own domain)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device != "cpu":
        raise RuntimeError("pmua cache build is CPU-only by contract")

    bench_root = (WS / "btransform_unified_v2" / "dandi688_bench_v1").resolve()
    dest = (args.dest if args.dest is not None
            else _variant_dest(args.variant)).resolve()
    if bench_root not in dest.parents:
        raise RuntimeError(f"--dest must live under {bench_root}")
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"destination must be empty: {dest}")

    plan.verify_protocol_definitions()
    norm_train = set(plan.protocol_sessions(args.norm_protocol)["train"])
    frozen_meta, frozen_rows = _load_frozen(args.source_cache)
    train_names = sorted(n for n, r in frozen_rows.items() if n in norm_train)
    if len(train_names) != len(norm_train):
        raise RuntimeError(
            f"norm protocol train sessions missing from frozen cache: "
            f"{len(train_names)}/{len(norm_train)}"
        )
    print(f"variant={args.variant} norm_protocol={args.norm_protocol} "
          f"train={len(train_names)} workers={args.workers}", flush=True)

    # --- cheap SUA datamodule route: per-session records (cached, read-only) -
    # only used for a byte-parity unit-axis gate (record.neural == frozen
    # cache neural columns); the channel calib is rebuilt from pooled counts.
    print("loading SUA datamodule (read-only cache)", flush=True)
    from mc_maze.dandi688_cp_film_v1.data import prepare_datamodule

    dm = prepare_datamodule(WS)
    records: dict[str, Any] = {}
    for split, dataset in (("train", dm.train_dataset), ("val", dm.val_dataset)):
        for name, record in dataset.sessions.items():
            records[name] = record

    # --- per-session NWB profiles (parallel; CPU-only workers) --------------
    started = time.monotonic()
    profiles: dict[str, dict[str, Any]] = {}
    names = sorted(frozen_rows)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for payload in pool.map(_worker, names):
                profiles[payload["session_id"]] = payload
                print(f"  profiles {payload['session_id']} units="
                      f"{payload['n_units']} electrodes={payload['n_electrodes']}",
                      flush=True)
    else:
        for name in names:
            payload = _worker(name)
            profiles[name] = payload
            print(f"  profiles {name} units={payload['n_units']} "
                  f"electrodes={payload['n_electrodes']}", flush=True)
    print(f"NWB profiles done in {time.monotonic() - started:.1f}s", flush=True)

    # --- SUA recipe parity gate (unit-axis alignment + recipe identity) -----
    from mc_maze.dandi688_sparse_event_t4_v1.core import (
        fit_source_normalizer,
        normalize_columns,
    )

    sua_norm = fit_source_normalizer([profiles[n]["raw_sua"] for n in train_names])
    parity_max = 0.0
    for name in names:
        mask = np.asarray(frozen_rows[name]["mask"], dtype=bool)
        n_real = int(mask.sum())
        if profiles[name]["n_units"] != n_real:
            raise RuntimeError(
                f"{name}: NWB units {profiles[name]['n_units']} != frozen mask "
                f"{n_real} (unit-axis alignment broken)"
            )
        got = normalize_columns(profiles[name]["raw_sua"], **sua_norm)
        frozen_carrier = np.asarray(frozen_rows[name]["carrier"], dtype=np.float64)[:n_real]
        max_abs = float(np.max(np.abs(got.astype(np.float64) - frozen_carrier)))
        parity_max = max(parity_max, max_abs)
        if max_abs > SUA_PARITY_MAX_ABS:
            raise RuntimeError(
                f"{name}: SUA recipe parity gate failed max_abs={max_abs}"
            )
    print(f"SUA parity gate PASS (max_abs={parity_max:.3e} <= "
          f"{SUA_PARITY_MAX_ABS:.0e})", flush=True)

    # --- channel-level T4 normalizer (train-only) + correlation report ------
    pmua_norm = fit_source_normalizer([profiles[n]["raw_pmua"] for n in train_names])
    correlation: dict[str, dict[str, float]] = {}
    for name in names:
        prof = profiles[name]
        _, inverse = np.unique(prof["electrode_ids"], return_inverse=True)
        correlation[name] = sua_vs_pmua_column_correlation(
            prof["raw_sua"].astype(np.float64), prof["raw_pmua"].astype(np.float64),
            inverse,
        )

    # --- pooled arrays + channel E0 remelt + write --------------------------
    print("loading frozen B3S student for channel E0 remelt (cpu)", flush=True)
    import torch

    from mc_maze.dandi688_cp_film_v1.runner import _prepare_student

    student = _prepare_student(WS, plan.SEED, torch.device("cpu"))

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "sessions").mkdir(exist_ok=True)
    rows_hashes: dict[str, dict[str, str]] = {}
    sessions_meta: dict[str, dict[str, str]] = {}
    identity_log: dict[str, Any] = {}
    build_started = time.monotonic()
    for name in names:
        row = frozen_rows[name]
        prof = profiles[name]
        mask = np.asarray(row["mask"], dtype=bool)
        n_real = int(mask.sum())
        n_elec = prof["n_electrodes"]
        if n_elec > N_PAD:
            raise RuntimeError(f"{name}: electrodes {n_elec} exceed Nmax {N_PAD}")
        neural = np.asarray(row["neural"])
        sua_record_neural = np.asarray(records[name].neural)
        if (sua_record_neural.shape != (neural.shape[0], n_real)
                or not np.array_equal(sua_record_neural, neural[:, :n_real])):
            raise RuntimeError(
                f"{name}: SUA record.neural != frozen cache neural columns "
                f"(unit-axis byte parity broken)"
            )
        pooled_neural, channel_ids, inverse = pool_count_array_by_electrode(
            neural[:, :n_real], prof["electrode_ids"]
        )
        assert_pool_conservation(pooled_neural, neural[:, :n_real], inverse, n_elec)
        # channel calib trials: the datamodule's own pseudo_mua construction --
        # pool the BINNED COUNTS first, then build the interpolated 30-trial
        # calibration on the pooled channels (_build_calib_trials with the
        # frozen class defaults pad_value=-1.0, interpolate_trials=True);
        # pooling the SUA interpolated calib instead would commute only up to
        # cubic-interp float rounding, so the count-domain route is the
        # byte-faithful one.
        from mc_maze.multisession_datamodule import _build_calib_trials

        calib_trials_info = [
            {"start": start, "stop": stop}
            for start, stop in prof["calib_trial_bins"]
        ]
        pooled_calib = _build_calib_trials(
            pooled_neural, calib_trials_info, 30, 100, n_elec,
            pad_value=-1.0, interpolate_trials=True,
        )
        if pooled_calib.shape != (30, 100, n_elec):
            raise RuntimeError(
                f"{name}: pooled calib geometry {pooled_calib.shape} != "
                f"(30, 100, {n_elec})"
            )
        if not np.isfinite(pooled_calib).all():
            raise RuntimeError(f"{name}: nonfinite pooled calib trials")
        if args.variant == "f_labelfree":
            # label-free ACTIVITY-ONLY cell (f-series, 2026-09-09): zero side
            # into the remelt and zero carrier bytes -- no label-derived byte
            # enters the cache; the T4 refit above was diagnostics only
            carrier_real = np.zeros((n_elec, plan.CARRIER_DIM), dtype=np.float32)
        else:
            carrier_real = normalize_columns(prof["raw_pmua"], **pmua_norm)
        e0 = vstate.remelt_e0(student, pooled_calib, carrier_real, N_PAD)

        neural_pmua = np.zeros((neural.shape[0], N_PAD), dtype=np.float32)
        neural_pmua[:, :n_elec] = pooled_neural
        carrier = np.zeros((N_PAD, plan.CARRIER_DIM), dtype=np.float32)
        carrier[:n_elec] = carrier_real
        mask_pmua = np.zeros(N_PAD, dtype=bool)
        mask_pmua[:n_elec] = True
        packed = {
            "neural": neural_pmua,
            "behavior": row["behavior"],
            "starts": row["starts"],
            "e0": e0,
            "carrier": carrier,
            "mask": mask_pmua,
        }
        for key in ("behavior", "starts"):
            if plan.array_digest(packed[key]) != plan.array_digest(row[key]):
                raise RuntimeError(f"{name}:{key} drifted while copying")
        np.savez_compressed(dest / "sessions" / f"{name}.npz", **packed)
        rows_hashes[name] = {key: plan.array_digest(packed[key]) for key in packed}
        sessions_meta[name] = {"split": row["split"]}
        units_per_elec = prof["units_per_electrode"]
        identity_log[name] = {
            "split": row["split"],
            "n_units": int(n_real),
            "n_electrodes": int(n_elec),
            "singleton_electrodes": int((units_per_elec == 1).sum()),
            "multi_unit_electrodes": int((units_per_elec > 1).sum()),
            "mean_units_per_electrode": float(units_per_elec.mean()),
            "channel_ids_sha256": plan.array_digest(channel_ids),
            "unit_to_electrode_sha256": plan.array_digest(prof["electrode_ids"]),
            "electrode_mapping": {
                "source": "NWB units.electrodes region index (one per unit)",
                "channel_order": "np.unique(electrode_ids) ascending",
            },
            "pooled_neural_total": float(pooled_neural.sum()),
            "sua_neural_total": float(neural[:, :n_real].sum()),
            "conservation_exact": bool(
                float(pooled_neural.sum()) == float(neural[:, :n_real].sum())
            ),
            "sua_vs_pmua_t4_correlation": correlation[name],
        }
        if not identity_log[name]["conservation_exact"]:
            raise RuntimeError(f"{name}: pooled spike conservation not exact")
        print(f"  packed {name} units={n_real} electrodes={n_elec} "
              f"r_a={correlation[name]['pearson_r_col_a']:.3f}", flush=True)

    # --- distribution + size report ------------------------------------------
    def _dist(split: str) -> dict[str, Any]:
        rows = [identity_log[n] for n in names if identity_log[n]["split"] == split]
        units = [r["n_units"] for r in rows]
        elecs = [r["n_electrodes"] for r in rows]
        return {
            "sessions": len(rows),
            "units": {"min": min(units), "max": max(units),
                      "mean": float(np.mean(units))},
            "electrodes": {"min": min(elecs), "max": max(elecs),
                           "mean": float(np.mean(elecs))},
            "channels_per_unit_mean_of_ratios": float(np.mean(
                [r["n_electrodes"] / r["n_units"] for r in rows])),
            "mean_units_per_electrode": float(np.mean(
                [r["mean_units_per_electrode"] for r in rows])),
        }

    r_all = [correlation[n] for n in names]
    size_bytes = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    report = {
        "distribution": {split: _dist(split) for split in ("train", "val")},
        "t4_column_correlation": {
            "pearson_r_col_a": {
                "min": float(np.nanmin([r["pearson_r_col_a"] for r in r_all])),
                "mean": float(np.nanmean([r["pearson_r_col_a"] for r in r_all])),
            },
            "pearson_r_col_c": {
                "min": float(np.nanmin([r["pearson_r_col_c"] for r in r_all])),
                "mean": float(np.nanmean([r["pearson_r_col_c"] for r in r_all])),
            },
            "mean_abs_angle_delta_deg": {
                "max": float(np.nanmax([r["mean_abs_angle_delta_deg"] for r in r_all])),
                "mean": float(np.nanmean([r["mean_abs_angle_delta_deg"] for r in r_all])),
            },
            "reading": "per-session Pearson r of the pooled channel T4 against "
                       "its member units' mean T4 (columns a/c/m, raw pre-"
                       "normalization) plus the pooled-vs-member preferred-"
                       "direction angle discrepancy",
        },
        "cache_size_bytes": size_bytes,
        "cache_size_mib": size_bytes / 1024 ** 2,
    }

    estimator = {
        "name": "pmua_t4_electrode_pooled_channel_t4",
        "recipe": "PSEUDO_MUA_T4_BRIDGE_48H.md section 2 frozen transform: "
                  "electrode pooling of sorted SUA counts, then channel-level "
                  "cosine T4 refit on pooled calibration-trial firing rates "
                  "(R700/H300 windows, first-M10 candidate support, closed-form "
                  "readout, delta_b = b_R700 - b_H300); unit-level T4 averaging "
                  "forbidden and never used",
        "signal_view": "pseudo_mua",
        "pooling": "pmua[e, t] = sum(sua[u, t] for electrode(u) == e); channel "
                   "order = np.unique(electrode_ids) ascending; every unit maps "
                   "exactly one electrode (asserted)",
        "t4_refit": "mc_maze.dandi688_sparse_event_t4_v1.descriptors."
                    "refit_from_single_pool(group='candidate', "
                    "signal_view='pseudo_mua') -- the same code path as the "
                    "frozen SUA carrier bytes",
        "e0_remmelt": "frozen B3S post_pool(cat(pre_pool(pooled calib).mean, "
                      "channel carrier side)), padded to [91, 50]; encoder unit "
                      "axis N = electrode count (pre_pool is per-channel "
                      "Linear(trial_length), channel-agnostic)",
        "norm_protocol": args.norm_protocol,
        "normalizer_domain": {
            "train_sessions": len(train_names),
            "law": "fit_source_normalizer over the norm-protocol train sessions' "
                   "raw channel T4 rows only; val rows never enter any statistic",
        },
        "normalizer_mean": pmua_norm["mean"].tolist(),
        "normalizer_scale": pmua_norm["scale"].tolist(),
        "sua_parity_gate": {
            "law": "recomputed SUA raw profile + train-only column normalizer "
                   "vs the frozen cache SUA carrier bytes (max abs over all 33 "
                   "sessions x real rows)",
            "max_abs": parity_max,
            "tolerance": SUA_PARITY_MAX_ABS,
            "pass": bool(parity_max <= SUA_PARITY_MAX_ABS),
        },
        "conservation_law": "per-channel pooled totals == member unit totals "
                            "exactly (half-open non-overlapping windows)",
        "source_cache": str(args.source_cache),
        "source_cache_contract_sha256": _file_sha256(
            args.source_cache / "prepared_contract.json"),
    }
    if args.variant == "f_labelfree":
        estimator.update({
            "name": "pmua_f_labelfree_zero_side_activity_only",
            "carrier": "all-zero [n_pad, 4]",
            "e0_side": "all-zero [n_electrodes, 4] (ACTIVITY-ONLY identity: "
                       "post_pool(cat(pre_pool(pooled calib).mean, zero side)))",
            "label_disclosure": "no behavioral labels anywhere in the cache "
                                "bytes: carrier all-zero, E0 side all-zero "
                                "(the channel T4 refit above ran only as a "
                                "pooling diagnostic and never enters bytes)",
            "component_role": "ACTIVITY_ONLY rung of the PMUA nested ladder "
                              "(plan.COMPONENT_ABLATION)",
        })
    else:
        estimator["label_disclosure"] = (
            "T4 consumes target_dir labels of the first-M10 rewarded "
            "calibration trials (the frozen SUA t4 arm's own support face); "
            "E0 remelt consumes the frozen B3S student and the same 30-trial "
            "activity calibration window pooled to electrodes"
        )
    contract = {
        "schema": plan.PREPARED_CACHE_SCHEMA,
        "manifest_sha256": plan.MANIFEST_SHA256,
        "split_counts": {"train": plan.SPLIT_COUNTS["train"],
                         "val": plan.SPLIT_COUNTS["val"]},
        "formal_test_used": False,
        "sessions": sessions_meta,
        "rows_hashes": rows_hashes,
        "variant": args.variant,
        "estimator": estimator,
        "electrode_report": report,
        "identity_log": identity_log,
        "implementation_shas": {
            "descriptors_py": _file_sha256(
                WS / "sua_exploration/mc_maze/dandi688_sparse_event_t4_v1/descriptors.py"),
            "se_core_py": _file_sha256(
                WS / "sua_exploration/mc_maze/dandi688_sparse_event_t4_v1/core.py"),
            "multisession_datamodule_py": _file_sha256(
                WS / "sua_exploration/mc_maze/multisession_datamodule.py"),
            "unit_side_features_py": _file_sha256(
                WS / "sua_exploration/mc_maze/unit_side_features.py"),
            "vstate_py": _file_sha256(Path(vstate.__file__)),
        },
        "builder_sha256": _file_sha256(Path(__file__)),
        "built_utc": datetime.now(timezone.utc).isoformat(),
    }
    _dump(dest / "prepared_contract.json", contract)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"wrote {dest} variant={args.variant} "
          f"elapsed_s={time.monotonic() - build_started:.1f}", flush=True)
    del student


if __name__ == "__main__":
    main()
