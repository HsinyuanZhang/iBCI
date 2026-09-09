#!/usr/bin/env python3
"""Build the vstate-688 prepared-cache variants (user directive 2026-09-09).

Variants:
  vstate       signed-velocity-state carrier (100 ms blocks inside the frozen
               R700/H300 windows of the SAME M10 support trials as the t4 arm;
               dense cursor_vel block means as the state labels; rms and the
               column normalizer are fit on the ACTIVE protocol's train
               sessions only) + co-variant E0 remelt
               post_pool(cat(pre_pool(calib).mean, vstate carrier_norm)).
  f_labelfree  label-free floor ("不使用任何标签校准"): carrier all-zero and
               E0 remelted with an all-zero side (ACTIVITY-ONLY identity).
               Protocol-independent (no labels, no normalizer).

z_vstate_srcbank needs NO cache here: it trains on the vstate cache and
swaps exam-session banks to the frozen nearest-date train banks at load
time (run_688_bench.swap_exam_banks_from_train).

Neural/behavior/starts/mask are copied byte-for-byte from the frozen
contract-v2 cache; only carrier and E0 are rewritten.  The written
prepared_contract.json still satisfies bench verify_prepared_cache (same
schema/manifest/split counts) plus the variant/protocol binding consumed
by run_688_bench.assert_cache_variant_for_arm.

CPU-only by contract (the E0 remelt loads the frozen B3S student on --device
cpu); no CUDA initialization, no training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
WS = PKG_ROOT.parents[1]
for p in (PKG_ROOT / "src", WS / "btransform_unified_v2" / "src",
          WS / "btransform_unified_v1" / "src", WS / "sua_exploration", WS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dandi688_bench_v1 import plan, vstate  # noqa: E402
from btransform_unified_v2 import carrier_profile_v3 as v3  # noqa: E402

VARIANTS = ("vstate", "f_labelfree")


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
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
    return meta, rows


def _nwb_path(session_id: str) -> Path:
    from mc_maze.dandi688_sparse_event_t4_v1 import plan as se_plan
    return WS / se_plan.DATA_RELATIVE / f"{session_id}_behavior+ecephys.nwb"


def _session_blocks(nwb_path: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Read the M10 support trials' R700/H300 blocks (rates + cursor_vel block
    means) straight from the NWB file, reusing the t4 arm's exact trial
    selection (_phase_trials, candidate namespace) and phase windows."""
    from pynwb import NWBHDF5IO
    from mc_maze.dandi688_sparse_event_t4_v1.descriptors import _phase_trials, _window_trials
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.dandi688_sparse_event_t4_v1 import plan as se_plan

    trials = list_datamodule_rewarded_trials(
        nwb_path, bin_size_ms=se_plan.BIN_SIZE_MS, window_size=se_plan.WINDOW_SIZE_BINS,
        trial_result_filter=se_plan.REWARDED_RESULT,
    )
    selected, exclusions = _phase_trials(
        trials, support_positions=plan.VSTATE_SUPPORT_POSITIONS,
        namespace=plan.VSTATE_SUPPORT_NAMESPACE,
    )
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        units_df = nwb.units.to_dataframe()
        spikes = [np.asarray(s, dtype=np.float64) for s in units_df["spike_times"].values]
        vel = nwb.processing["behavior"]["Velocity"].time_series["cursor_vel"]
        vel_times = np.asarray(vel.timestamps[:], dtype=np.float64)
        vel_values = np.asarray(vel.data[:], dtype=np.float64)

    blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for phase in ("r700", "h300"):
        rates, vels = [], []
        for window in _window_trials(selected, phase=phase):
            edges = vstate.block_edges(window["start_time"], window["stop_time"])
            rates.append(vstate.block_rate_matrix(spikes, edges))
            vels.append(vstate.block_velocity_means(vel_times, vel_values, edges))
        blocks[phase] = (np.concatenate(rates, axis=0), np.concatenate(vels, axis=0))
    identity = {
        "n_legal_trials": int(len(selected)),
        "exclusions": exclusions,
        "n_source_units": int(len(spikes)),
        "r700_blocks": int(blocks["r700"][0].shape[0]),
        "h300_blocks": int(blocks["h300"][0].shape[0]),
    }
    if identity["r700_blocks"] != len(selected) * round(plan.VSTATE_R700_SECONDS / plan.VSTATE_BLOCK_SECONDS):
        raise RuntimeError(f"{nwb_path.name}: r700 block count drift")
    if identity["h300_blocks"] != len(selected) * round(plan.VSTATE_H300_SECONDS / plan.VSTATE_BLOCK_SECONDS):
        raise RuntimeError(f"{nwb_path.name}: h300 block count drift")
    return blocks, identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), required=True)
    parser.add_argument("--protocol", choices=tuple(plan.PROTOCOLS),
                        default=plan.DEFAULT_PROTOCOL,
                        help="train-session set that calibrates the vstate rms "
                             "and column normalizer (vstate variant only; "
                             "f_labelfree is label-free and protocol-independent)")
    parser.add_argument("--dest", type=Path, help="destination cache dir (required unless --variant all)")
    parser.add_argument("--dest-root", type=Path, default=PKG_ROOT / "results",
                        help="parent dir for cache_<variant> when --variant all")
    parser.add_argument("--source-cache", type=Path, default=plan.prepared_cache_path())
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    dests: dict[str, Path] = {}
    if args.variant == "all":
        dests["vstate"] = (args.dest_root / f"cache_vstate_{args.protocol}").resolve()
        dests["f_labelfree"] = (args.dest_root / "cache_f_labelfree").resolve()
    else:
        if args.dest is None:
            raise ValueError("--dest is required unless --variant all")
        dests[args.variant] = args.dest.resolve()
    bench_root = (WS / "btransform_unified_v2" / "dandi688_bench_v1").resolve()
    for dest in dests.values():
        if bench_root not in dest.parents:
            raise RuntimeError(f"--dest must live under {bench_root}")
        if dest.exists() and any(dest.iterdir()):
            raise FileExistsError(f"destination must be empty: {dest}")

    plan.verify_protocol_definitions()
    train_sessions = plan.protocol_sessions(args.protocol)["train"]
    frozen_meta, frozen_rows = _load_frozen(args.source_cache)
    train_names = [n for n in frozen_rows if n in set(train_sessions)]
    if len(train_names) != len(train_sessions):
        raise RuntimeError(
            f"frozen cache misses protocol train sessions: {len(train_names)}/{len(train_sessions)}"
        )

    from mc_maze.dandi688_sparse_event_t4_v1.production import prepare_source_surface
    from mc_maze.dandi688_cp_film_v1.runner import _prepare_student

    print("prepare_source_surface for calib_trials", flush=True)
    surface = prepare_source_surface(WS, signal_view="sua", reliability_mask=(True, True, True, True))
    print("loading frozen B3S student for E0 remelt", flush=True)
    import torch

    student = _prepare_student(WS, plan.SEED, torch.device(args.device))

    build_started = time.monotonic()
    if "vstate" in variants:
        # pass 1: read every session's M10 blocks; rms uses TRAIN blocks only
        print("reading NWB blocks for all sessions", flush=True)
        blocks_by_session: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        identity_log: dict[str, Any] = {}
        for name in sorted(frozen_rows):
            blocks, identity = _session_blocks(_nwb_path(name))
            blocks_by_session[name] = blocks
            identity["mask_real_units"] = int(np.asarray(frozen_rows[name]["mask"]).sum())
            if identity["n_source_units"] != identity["mask_real_units"]:
                raise RuntimeError(
                    f"{name}: source units {identity['n_source_units']} != mask "
                    f"{identity['mask_real_units']}"
                )
            identity_log[name] = identity
            print(f"  blocks {name} legal_trials={identity['n_legal_trials']} "
                  f"units={identity['n_source_units']}", flush=True)
        rms = vstate.fit_velocity_rms(
            [np.concatenate((blocks_by_session[n]["r700"][1],
                             blocks_by_session[n]["h300"][1]), axis=0)
             for n in train_names]
        )
        print(f"velocity rms (train-only, {args.protocol}): {rms.tolist()}", flush=True)

        raw_rows: dict[str, np.ndarray] = {}
        for name in sorted(frozen_rows):
            r700_rates, r700_vel = blocks_by_session[name]["r700"]
            h300_rates, h300_vel = blocks_by_session[name]["h300"]
            raw_rows[name] = vstate.vstate_carrier_from_blocks(
                r700_rates, r700_vel, h300_rates, h300_vel, rms,
            )
        mean, std = v3.fit_column_normalizer([raw_rows[n] for n in train_names])
        vstate_estimator = {
            "name": "vstate_signed_state_blocks",
            "recipe": "M2 vstate4 (PLAN_CARRIER_ITERATION_M2_688_20260909 §3.2) "
                      "adapted to the 688 rate primitives",
            "protocol": args.protocol,
            "support_namespace": plan.VSTATE_SUPPORT_NAMESPACE,
            "support_positions": list(plan.VSTATE_SUPPORT_POSITIONS),
            "support_note": "same M10 trial set as the t4 arm; only the labels "
                            "swap direction -> velocity",
            "block_seconds": plan.VSTATE_BLOCK_SECONDS,
            "r700_seconds": plan.VSTATE_R700_SECONDS,
            "h300_seconds": plan.VSTATE_H300_SECONDS,
            "n0_blocks": plan.VSTATE_N0_BLOCKS,
            "poisson_duration_s": plan.VSTATE_BLOCK_SECONDS,
            "velocity_subbins": plan.VSTATE_VELOCITY_SUBBINS,
            "velocity_bin_seconds": plan.VSTATE_VELOCITY_BIN_SECONDS,
            "state_columns": list(plan.VSTATE_STATE_COLUMNS),
            "hold": plan.VSTATE_HOLD,
            "velocity_rms": rms.tolist(),
            "rms_train_sessions": len(train_names),
            "normalizer_mean": mean.tolist(),
            "normalizer_std": std.tolist(),
            "label_disclosure": "consumes dense cursor_vel calibration labels of "
                                "the M10 support trials (see bench README for the "
                                "three-axis analysis vs the legacy 688 dense-null)",
            "identity_log": identity_log,
        }
    else:
        vstate_estimator = None

    for variant in variants:
        dest = dests[variant]
        variant_started = time.monotonic()
        rows_hashes: dict[str, dict[str, str]] = {}
        sessions_meta: dict[str, dict[str, str]] = {}
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "sessions").mkdir(exist_ok=True)
        for name, row in frozen_rows.items():
            mask = np.asarray(row["mask"], dtype=bool)
            n_real = int(mask.sum())
            calib = surface.sessions[name].record.calib_trials
            if calib.shape[-1] != n_real:
                raise RuntimeError(f"{name}: calib units {calib.shape[-1]} != mask {n_real}")
            carrier = np.zeros((mask.size, plan.CARRIER_DIM), dtype=np.float32)
            if variant == "vstate":
                carrier[:n_real] = v3.apply_column_normalizer(raw_rows[name], mean, std)
                side = carrier[:n_real]
            else:  # f_labelfree: zero carrier, zero side (ACTIVITY-ONLY identity)
                side = np.zeros((n_real, plan.CARRIER_DIM), dtype=np.float32)
            e0 = vstate.remelt_e0(student, calib, side, mask.size)
            packed = {
                "neural": row["neural"],
                "behavior": row["behavior"],
                "starts": row["starts"],
                "e0": e0,
                "carrier": carrier,
                "mask": row["mask"],
            }
            for key in ("neural", "behavior", "starts", "mask"):
                if plan.array_digest(packed[key]) != plan.array_digest(row[key]):
                    raise RuntimeError(f"{name}:{key} drifted while copying")
            np.savez_compressed(dest / "sessions" / f"{name}.npz", **packed)
            rows_hashes[name] = {key: plan.array_digest(packed[key]) for key in packed}
            sessions_meta[name] = {"split": row["split"]}
            print(f"  {variant} packed {name} units={n_real}", flush=True)

        estimator: dict[str, Any] = {
            "source_cache": str(args.source_cache),
            "source_cache_contract_sha256": _file_sha256(args.source_cache / "prepared_contract.json"),
            "implementation_sha256": _file_sha256(Path(vstate.__file__)),
            "builder_sha256": _file_sha256(Path(__file__)),
        }
        if variant == "vstate":
            estimator = {**vstate_estimator, **estimator}
        else:
            estimator = {
                "name": "activity_only_zero_side",
                "recipe": "user directive 2026-09-09 f-series: 不使用任何标签校准",
                "protocol": None,
                "carrier": "all-zero [n_pad, 4]",
                "e0_side": "all-zero [n_real, 4] (ACTIVITY-ONLY identity: "
                           "post_pool(cat(pre_pool(calib).mean, zero side)))",
                "label_disclosure": "no behavioral labels anywhere: no direction, "
                                    "no velocity, no rms, no column normalizer",
                **estimator,
            }
        contract = {
            "schema": plan.PREPARED_CACHE_SCHEMA,
            "manifest_sha256": plan.MANIFEST_SHA256,
            "split_counts": {"train": plan.SPLIT_COUNTS["train"], "val": plan.SPLIT_COUNTS["val"]},
            "formal_test_used": False,
            "sessions": sessions_meta,
            "rows_hashes": rows_hashes,
            "variant": variant,
            "estimator": estimator,
            "built_utc": datetime.now(timezone.utc).isoformat(),
        }
        _dump(dest / "prepared_contract.json", contract)
        print(f"wrote {dest} variant={variant} "
              f"elapsed_s={time.monotonic() - variant_started:.1f}", flush=True)
    del student
    print(f"total build elapsed_s={time.monotonic() - build_started:.1f}", flush=True)


if __name__ == "__main__":
    main()
