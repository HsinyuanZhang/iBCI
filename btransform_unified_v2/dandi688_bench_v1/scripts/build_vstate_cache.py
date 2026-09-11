#!/usr/bin/env python3
"""Build the vstate-688 prepared-cache variants (user directive 2026-09-09).

Variants (cache build parameters; per-part M2-alignment audit in
btransform_unified_v2/docs/CARRIER_M2_688_ALIGNMENT_MATRIX_20260909.md):
  vstate          main variant: signed-velocity-state carrier on the SAME M10
                  support trials as the t4 arm (candidate namespace), fourth
                  column b = mean_k R  (M2 vstate4-identical, user ruling
                  2026-09-09 final: 最大对应优先).  rms and the column
                  normalizer are fit on the ACTIVE protocol's train sessions
                  only; E0 remelt co-variant
                  post_pool(cat(pre_pool(calib).mean, vstate carrier_norm)).
  vstate_full     alignment-matrix "support" row variant: identical estimator
                  but the support is ALL 30 activity-support trials
                  (reliability_audit namespace), mirroring M2's use of ALL
                  calib trials (M33).  Definition only until built.
  vstate_b_hold   alignment-matrix "readout" row ablation: identical to
                  vstate (same M10 support, same rms, so a/c/m are
                  byte-identical) except b = mean_k R - mean_k R_hold (the
                  original delta_b semantics, hold = the H300 blocks' own
                  signed-state conditional response).  Switchable fallback
                  per the ruling: enable if the main variant's exp1 reading
                  is significantly below t4.
  f_labelfree     label-free floor ("不使用任何标签校准"): carrier all-zero and
                  E0 remelted with an all-zero side (ACTIVITY-ONLY identity).
                  Protocol-independent (no labels, no normalizer).
  norm_only       NORM_ONLY zero-calibration baseline (guide btransform_
                  unified_v2/docs/NORM_ONLY_ZERO_BASELINE_GUIDE_20260910.md
                  sections 2/3/6): carrier all-zero and E0 = the day's
                  label-free per-unit pooled-rate DEVIATION from the source
                  rate distribution, broadcast over e0_dim (NO encoder, NO
                  side input, NO labels; dandi688_bench_v1.norm_only).  The
                  rate statistic is pooled over the SAME label-free support
                  the ACTIVITY_ONLY arm's E0 consumes (the session's
                  calib_trials tensor, ACTIVITY_SUPPORT_N = 30 first rewarded
                  trials); mu_src/sigma_src are fit on the ACTIVE protocol's
                  train sessions only, so this variant IS protocol-bound
                  (unlike f_labelfree/equiv_zero).  The build asserts the E0
                  differs from the f_labelfree cache's E0 (it must: NORM_ONLY
                  carries no encoder pathway) and records the difference.
  equiv_zero      component-ablation capacity control (user clarification
                  2026-09-09): carrier all-zero and E0 = the FIXED RANDOM
                  projection (param-matched to the frozen post_pool pathway,
                  seed 42, never trained) applied to the pre_pool activity
                  mean with a zero side (dandi688_bench_v1.equiv_zero).
                  Protocol-independent (no labels, no normalizer, no rms).

rms fitting domain (decision record in the alignment matrix): each support
face's rms is fit on its OWN movement (R700) block velocities over the
active protocol's train sessions — M2 fits rms on exactly the block
population entering its estimator — and vstate / vstate_b_hold share the
M10 rms, so the b-column ablation isolates exactly the fourth column.

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

from dandi688_bench_v1 import equiv_zero, norm_only, plan, vstate  # noqa: E402
from btransform_unified_v2 import carrier_profile_v3 as v3  # noqa: E402

VARIANTS = ("vstate", "vstate_full", "vstate_b_hold", "f_labelfree",
            "equiv_zero", "norm_only")

B_FORMULAS = {
    "mean_k": "b = mean_k R_u (M2 vstate4-identical fourth column)",
    "hold_diff": "b = mean_k R_u - mean_k R_hold (delta_b, 688-local ablation)",
}


def _variant_spec(variant: str) -> dict[str, Any]:
    """Cache build parameters of one variant (support face + b mode)."""
    if variant == "vstate":
        return {
            "positions": plan.VSTATE_SUPPORT_POSITIONS,
            "namespace": plan.VSTATE_SUPPORT_NAMESPACE,
            "label_budget": len(plan.VSTATE_SUPPORT_POSITIONS),
            "b_mode": plan.VSTATE_B_MODE_MAIN,
        }
    if variant == "vstate_full":
        return {
            "positions": plan.VSTATE_FULL_SUPPORT_POSITIONS,
            "namespace": plan.VSTATE_FULL_SUPPORT_NAMESPACE,
            "label_budget": len(plan.VSTATE_FULL_SUPPORT_POSITIONS),
            "b_mode": plan.VSTATE_B_MODE_MAIN,
        }
    if variant == "vstate_b_hold":
        return {
            "positions": plan.VSTATE_SUPPORT_POSITIONS,
            "namespace": plan.VSTATE_SUPPORT_NAMESPACE,
            "label_budget": len(plan.VSTATE_SUPPORT_POSITIONS),
            "b_mode": "hold_diff",
        }
    if variant == "f_labelfree":
        return {"label_budget": 0, "b_mode": None}
    if variant == "norm_only":
        # NORM_ONLY zero-calibration baseline (guide 2026-09-10): no labels,
        # no carrier, no encoder -- the E0 identity is the broadcast rate
        # deviation, protocol-bound through its source (train) sessions.
        return {
            "label_budget": 0,
            "b_mode": None,
            "normalizer": "source_rate_mu_sigma",
            "source_domain": "protocol_train_sessions",
        }
    if variant == "equiv_zero":
        return {"label_budget": 0, "b_mode": None, "projection_seed": equiv_zero.EQUIV_ZERO_SEED}
    raise ValueError(variant)


def _variant_dest(variant: str, dest_root: Path, protocol: str) -> Path:
    if variant == "f_labelfree":
        return (dest_root / "cache_f_labelfree").resolve()
    if variant == "norm_only":
        # naming law of the label-free-family caches (cache_f_labelfree /
        # cache_equiv_zero); the protocol binding is recorded in the contract
        # and re-checked by run_688_bench.assert_cache_variant_for_arm
        return (dest_root / plan.NORM_ONLY_CACHE_DIRNAME).resolve()
    if variant == "equiv_zero":
        return (dest_root / "cache_equiv_zero").resolve()
    if variant == "vstate":
        return (dest_root / f"cache_vstate_{protocol}").resolve()
    return (dest_root / f"cache_{variant}_{protocol}").resolve()


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
    return meta, rows


def _nwb_path(session_id: str) -> Path:
    from mc_maze.dandi688_sparse_event_t4_v1 import plan as se_plan
    return WS / se_plan.DATA_RELATIVE / f"{session_id}_behavior+ecephys.nwb"


def _session_blocks(
    nwb_path: Path,
    *,
    support_positions: tuple[int, ...],
    namespace: str,
    include_h300: bool,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Read the given support face's R700 (and optionally H300) blocks (rates
    + cursor_vel block means) straight from the NWB file, reusing the t4
    arm's exact trial-selection route (_phase_trials) and phase windows."""
    from pynwb import NWBHDF5IO
    from mc_maze.dandi688_sparse_event_t4_v1.descriptors import _phase_trials, _window_trials
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.dandi688_sparse_event_t4_v1 import plan as se_plan

    trials = list_datamodule_rewarded_trials(
        nwb_path, bin_size_ms=se_plan.BIN_SIZE_MS, window_size=se_plan.WINDOW_SIZE_BINS,
        trial_result_filter=se_plan.REWARDED_RESULT,
    )
    selected, exclusions = _phase_trials(
        trials, support_positions=support_positions, namespace=namespace,
    )
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        units_df = nwb.units.to_dataframe()
        spikes = [np.asarray(s, dtype=np.float64) for s in units_df["spike_times"].values]
        vel = nwb.processing["behavior"]["Velocity"].time_series["cursor_vel"]
        vel_times = np.asarray(vel.timestamps[:], dtype=np.float64)
        vel_values = np.asarray(vel.data[:], dtype=np.float64)

    phases = ("r700", "h300") if include_h300 else ("r700",)
    blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for phase in phases:
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
        "h300_blocks": int(blocks["h300"][0].shape[0]) if "h300" in blocks else None,
    }
    if identity["r700_blocks"] != len(selected) * round(plan.VSTATE_R700_SECONDS / plan.VSTATE_BLOCK_SECONDS):
        raise RuntimeError(f"{nwb_path.name}: r700 block count drift")
    if "h300" in blocks and identity["h300_blocks"] != len(selected) * round(plan.VSTATE_H300_SECONDS / plan.VSTATE_BLOCK_SECONDS):
        raise RuntimeError(f"{nwb_path.name}: h300 block count drift")
    return blocks, identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), required=True)
    parser.add_argument("--protocol", choices=tuple(plan.PROTOCOLS),
                        default=plan.DEFAULT_PROTOCOL,
                        help="train-session set that calibrates the vstate rms "
                             "and column normalizer (vstate-family variants "
                             "only) and the NORM_ONLY source rate "
                             "distribution (norm_only); f_labelfree and "
                             "equiv_zero are label-free and "
                             "protocol-independent")
    parser.add_argument("--dest", type=Path,
                        help="destination cache dir (default: <dest-root>/"
                             "cache_<variant> via the frozen naming rule)")
    parser.add_argument("--dest-root", type=Path, default=PKG_ROOT / "results",
                        help="parent dir for cache_<variant> when --variant all")
    parser.add_argument("--source-cache", type=Path, default=plan.prepared_cache_path())
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reference-cache", type=Path, default=None,
        help="reference variant cache for the norm_only E0-difference "
             "assertion (the NORM_ONLY identity must NOT be the ACTIVITY_ONLY "
             "melt); default: <dest-root>/cache_f_labelfree",
    )
    args = parser.parse_args()

    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    dests: dict[str, Path] = {}
    for name in variants:
        # single variant: explicit --dest wins, else the frozen naming rule
        dests[name] = (
            args.dest.resolve() if (args.variant != "all" and args.dest is not None)
            else _variant_dest(name, args.dest_root, args.protocol)
        )
    bench_root = (WS / "btransform_unified_v2" / "dandi688_bench_v1").resolve()
    for dest in dests.values():
        if bench_root not in dest.parents:
            raise RuntimeError(f"--dest must live under {bench_root}")
        if dest.exists() and any(dest.iterdir()):
            raise FileExistsError(f"destination must be empty: {dest}")
    reference_cache = (
        args.reference_cache if args.reference_cache is not None
        else args.dest_root / "cache_f_labelfree"
    ).resolve()
    if "norm_only" in variants:
        if not (reference_cache / "prepared_contract.json").is_file():
            raise RuntimeError(
                "norm_only needs the f_labelfree reference cache for its "
                f"E0-difference assertion: {reference_cache} has no "
                "prepared_contract.json (pass --reference-cache)"
            )

    plan.verify_protocol_definitions()
    train_sessions = plan.protocol_sessions(args.protocol)["train"]
    frozen_meta, frozen_rows = _load_frozen(args.source_cache)
    train_names = [n for n in frozen_rows if n in set(train_sessions)]
    if len(train_names) != len(train_sessions):
        raise RuntimeError(
            f"frozen cache misses protocol train sessions: {len(train_names)}/{len(train_sessions)}"
        )
    n_pad_values = {int(np.asarray(row["mask"]).size) for row in frozen_rows.values()}
    if len(n_pad_values) != 1:
        raise RuntimeError(f"frozen cache rows disagree on n_pad: {sorted(n_pad_values)}")
    n_pad = n_pad_values.pop()

    from mc_maze.dandi688_sparse_event_t4_v1.production import prepare_source_surface
    from mc_maze.dandi688_cp_film_v1.runner import _prepare_student

    print("prepare_source_surface for calib_trials", flush=True)
    surface = prepare_source_surface(WS, signal_view="sua", reliability_mask=(True, True, True, True))

    # --- NORM_ONLY source rate distribution (guide section 3) --------------
    # rate_sess[u] = pooled firing rate over the SAME label-free support the
    # ACTIVITY_ONLY arm's E0 consumes; mu/sigma = per-slot mean/std of that
    # statistic over the ACTIVE protocol's train sessions.  Frozen once.
    source_rates: dict[str, np.ndarray] = {}
    rate_normalizer: dict[str, Any] | None = None
    if "norm_only" in variants:
        support_sizes = set()
        for name in sorted(frozen_rows):
            calib = np.asarray(surface.sessions[name].record.calib_trials)
            n_real = int(np.asarray(frozen_rows[name]["mask"]).sum())
            if calib.shape[-1] != n_real:
                raise RuntimeError(f"{name}: calib units {calib.shape[-1]} != mask {n_real}")
            if calib.shape[0] != plan.NORM_ONLY_SUPPORT_TRIALS:
                raise RuntimeError(
                    f"{name}: calibration support holds {calib.shape[0]} trials, "
                    f"expected ACTIVITY_SUPPORT_N = {plan.NORM_ONLY_SUPPORT_TRIALS}"
                )
            support_sizes.add(int(calib.shape[1]))
            source_rates[name] = norm_only.pooled_support_rate(calib)
        print(f"norm_only support: {plan.NORM_ONLY_SUPPORT_TRIALS} trials, "
              f"bins={sorted(support_sizes)}, sessions={len(source_rates)}", flush=True)
        rate_normalizer = norm_only.fit_source_rate_normalizer(
            [source_rates[name] for name in train_names], n_pad,
        )
        mu, sigma, defined = (rate_normalizer["mu"], rate_normalizer["sigma"],
                              rate_normalizer["defined"])
        print(f"norm_only source normalizer (train={len(train_names)}): "
              f"defined_slots={int(defined.sum())}/{n_pad} "
              f"mu=[{mu[defined].min():.4f}, {mu[defined].max():.4f}] "
              f"sigma=[{sigma[defined].min():.4f}, {sigma[defined].max():.4f}]",
              flush=True)
        for name in train_names:
            print(f"  source rate {name} mean={source_rates[name].mean():.5f} "
                  f"min={source_rates[name].min():.5f} "
                  f"max={source_rates[name].max():.5f}", flush=True)

    reference_rows: dict[str, np.ndarray] = {}
    if "norm_only" in variants:
        reference_meta, reference_loaded = _load_frozen(reference_cache)
        if reference_meta.get("variant") != "f_labelfree":
            raise RuntimeError(
                f"reference cache {reference_cache} is variant "
                f"{reference_meta.get('variant')!r}, expected 'f_labelfree'"
            )
        reference_rows = {name: np.asarray(row["e0"]) for name, row in reference_loaded.items()}
        del reference_loaded

    # every variant except norm_only melts its E0 through the frozen B3S
    # student (the NORM_ONLY identity is the broadcast rate deviation itself
    # -- no encoder, no side input, so that build stays torch-free)
    needs_student = any(variant != "norm_only" for variant in variants)
    student = None
    if needs_student:
        print("loading frozen B3S student for E0 remelt", flush=True)
        import torch

        student = _prepare_student(WS, plan.SEED, torch.device(args.device))
    else:
        print("no E0 remelt in this build: frozen B3S student not loaded", flush=True)

    build_started = time.monotonic()
    # equiv_zero: ONE fixed random projection (seed 42) shared by every
    # session of the build -- "每个 session 拿到的是同一个随机投影作用于其
    # 自身活动" (user ruling 2026-09-09)
    fixed_projection = (
        equiv_zero.build_fixed_projection(student)
        if "equiv_zero" in variants else None
    )
    vstate_family = [v for v in variants if v in plan.VSTATE_VARIANTS]
    # each DISTINCT support face is read once; H300 blocks only when some
    # selected variant on that face needs them (b_mode="hold_diff")
    face_include_h300: dict[tuple[tuple[int, ...], str], bool] = {}
    for variant in vstate_family:
        spec = _variant_spec(variant)
        key = (spec["positions"], spec["namespace"])
        face_include_h300[key] = face_include_h300.get(key, False) or spec["b_mode"] == "hold_diff"
    faces: dict[tuple[tuple[int, ...], str], dict[str, Any]] = {}
    if vstate_family:
        print("reading NWB blocks for all sessions", flush=True)
        for name in sorted(frozen_rows):
            mask_real_units = int(np.asarray(frozen_rows[name]["mask"]).sum())
            for key, include_h300 in face_include_h300.items():
                blocks, identity = _session_blocks(
                    _nwb_path(name), support_positions=key[0], namespace=key[1],
                    include_h300=include_h300,
                )
                faces.setdefault(key, {"blocks": {}, "identity": {}})
                faces[key]["blocks"][name] = blocks
                faces[key]["identity"][name] = identity
                if identity["n_source_units"] != mask_real_units:
                    raise RuntimeError(
                        f"{name}: source units {identity['n_source_units']} != mask "
                        f"{mask_real_units}"
                    )
                print(f"  blocks {name} face={key[1]}[{len(key[0])}] "
                      f"legal_trials={identity['n_legal_trials']} "
                      f"units={identity['n_source_units']}", flush=True)

    # per-support-face rms (train-only, movement blocks) shared by every
    # variant on that face -> the b-column ablation isolates exactly column 4
    face_rms: dict[tuple[tuple[int, ...], str], np.ndarray] = {}
    for key in faces:
        face_rms[key] = vstate.fit_velocity_rms(
            [faces[key]["blocks"][n]["r700"][1] for n in train_names]
        )
        print(f"velocity rms (train-only, {args.protocol}, face "
              f"{key[1]}[{len(key[0])}]): {face_rms[key].tolist()}", flush=True)

    estimators: dict[str, dict[str, Any]] = {}
    raw_rows_by_variant: dict[str, dict[str, np.ndarray]] = {}
    normalizers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for variant in vstate_family:
        spec = _variant_spec(variant)
        key = (spec["positions"], spec["namespace"])
        rms = face_rms[key]
        raw_rows: dict[str, np.ndarray] = {}
        for name in sorted(frozen_rows):
            r700_rates, r700_vel = faces[key]["blocks"][name]["r700"]
            if spec["b_mode"] == "hold_diff":
                h300_rates, h300_vel = faces[key]["blocks"][name]["h300"]
                raw_rows[name] = vstate.vstate_carrier_from_blocks(
                    r700_rates, r700_vel, h300_rates, h300_vel, rms,
                    b_mode="hold_diff",
                )
            else:
                raw_rows[name] = vstate.vstate_carrier_from_blocks(
                    r700_rates, r700_vel, None, None, rms, b_mode="mean_k",
                )
        raw_rows_by_variant[variant] = raw_rows
        normalizers[variant] = v3.fit_column_normalizer(
            [raw_rows[n] for n in train_names]
        )
        mean, std = normalizers[variant]
        estimators[variant] = {
            "name": "vstate_signed_state_blocks",
            "recipe": "M2 vstate4 (PLAN_CARRIER_ITERATION_M2_688_20260909 "
                      "section 3.2) adapted to the 688 rate primitives; "
                      "per-part alignment audit CARRIER_M2_688_ALIGNMENT_"
                      "MATRIX_20260909",
            "protocol": args.protocol,
            "support_namespace": spec["namespace"],
            "support_positions": list(spec["positions"]),
            "label_budget": int(spec["label_budget"]),
            "support_note": "vstate/vstate_b_hold: same M10 trial set as the "
                            "t4 arm; vstate_full: all 30 activity-support "
                            "trials (M2 all-support semantics)",
            "block_seconds": plan.VSTATE_BLOCK_SECONDS,
            "r700_seconds": plan.VSTATE_R700_SECONDS,
            "h300_seconds": plan.VSTATE_H300_SECONDS,
            "n0_blocks": plan.VSTATE_N0_BLOCKS,
            "poisson_duration_s": plan.VSTATE_BLOCK_SECONDS,
            "velocity_subbins": plan.VSTATE_VELOCITY_SUBBINS,
            "velocity_bin_seconds": plan.VSTATE_VELOCITY_BIN_SECONDS,
            "state_columns": list(plan.VSTATE_STATE_COLUMNS),
            "b_mode": spec["b_mode"],
            "b_formula": B_FORMULAS[spec["b_mode"]],
            "hold": plan.VSTATE_HOLD if spec["b_mode"] == "hold_diff" else None,
            "velocity_rms": rms.tolist(),
            "rms_domain": "r700 movement blocks of this variant's support "
                          "face over the active protocol's train sessions",
            "rms_train_sessions": len(train_names),
            "normalizer_mean": mean.tolist(),
            "normalizer_std": std.tolist(),
            "label_disclosure": "consumes dense cursor_vel calibration labels "
                                "of the support trials (see bench README for "
                                "the three-axis analysis vs the legacy 688 "
                                "dense-null)",
            "identity_log": {
                name: faces[key]["identity"][name] for name in sorted(frozen_rows)
            },
        }

    for variant in variants:
        dest = dests[variant]
        variant_started = time.monotonic()
        rows_hashes: dict[str, dict[str, str]] = {}
        sessions_meta: dict[str, dict[str, str]] = {}
        norm_only_log: dict[str, Any] = {}
        norm_only_e0_diffs: dict[str, Any] = {}
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "sessions").mkdir(exist_ok=True)
        mean, std = normalizers[variant] if variant in vstate_family else (None, None)
        for name, row in frozen_rows.items():
            mask = np.asarray(row["mask"], dtype=bool)
            n_real = int(mask.sum())
            calib = surface.sessions[name].record.calib_trials
            if calib.shape[-1] != n_real:
                raise RuntimeError(f"{name}: calib units {calib.shape[-1]} != mask {n_real}")
            carrier = np.zeros((mask.size, plan.CARRIER_DIM), dtype=np.float32)
            if variant in vstate_family:
                carrier[:n_real] = v3.apply_column_normalizer(
                    raw_rows_by_variant[variant][name], mean, std
                )
                side = carrier[:n_real]
            elif variant == "equiv_zero":
                # capacity control: E0 = the param-matched fixed random
                # projection of the pre_pool activity mean (zero side);
                # carrier stays all-zero
                side = None
            elif variant == "norm_only":
                # NORM_ONLY: carrier all-zero; E0 = the broadcast per-unit rate
                # deviation against the frozen source rate distribution (no
                # encoder, no side input, no labels)
                side = None
            else:  # f_labelfree: zero carrier, zero side (ACTIVITY-ONLY identity)
                side = np.zeros((n_real, plan.CARRIER_DIM), dtype=np.float32)
            if variant == "norm_only":
                e0, info = norm_only.e0_from_rate_deviation(
                    calib, rate_normalizer["mu"], rate_normalizer["sigma"],
                    mask.size, plan.E0_DIM,
                    sigma_floor=plan.NORM_ONLY_SIGMA_FLOOR,
                    source_rates_defined=rate_normalizer["defined"],
                )
                diff = norm_only.e0_difference(e0, reference_rows[name])
                if diff["identical"] or diff["n_differing_rows"] == 0:
                    raise RuntimeError(
                        f"{name}: NORM_ONLY E0 is byte-identical to the "
                        f"f_labelfree (ACTIVITY_ONLY) E0 at {reference_cache}; "
                        "the rung must carry a different identity"
                    )
                norm_only_e0_diffs[name] = diff
                norm_only_log[name] = {
                    "n_real_units": info["n_real_units"],
                    "n_support_trials": info["n_support_trials"],
                    "n_support_bins": info["n_support_bins"],
                    "n_defined_slots": info["n_defined_slots"],
                    "n_undefined_slots": info["n_undefined_slots"],
                    "rate_summary": info["rate_summary"],
                    "z_summary": info["z_summary"],
                    # per-unit pooled rate (JSON mirror; digest of the same
                    # float64 values) -- lets a reader re-derive the E0
                    "rate": [float(value) for value in info["rate"]],
                    "rate_sha256": plan.array_digest(info["rate"]),
                    "e0_sha256": plan.array_digest(e0),
                    "e0_vs_f_labelfree": diff,
                }
            elif variant == "equiv_zero":
                e0 = equiv_zero.e0_from_projection(
                    fixed_projection, student, calib, mask.size
                )
            else:
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
            "implementation_sha256": _file_sha256(
                Path(norm_only.__file__) if variant == "norm_only"
                else Path(vstate.__file__)
            ),
            "builder_sha256": _file_sha256(Path(__file__)),
        }
        if variant in vstate_family:
            estimator = {**estimators[variant], **estimator}
        elif variant == "norm_only":
            estimator = {
                **norm_only.source_normalizer_receipt(
                    rate_normalizer, sorted(train_names),
                ),
                # protocol-bound: the source rate distribution is the ACTIVE
                # protocol's train sessions (enforced by
                # run_688_bench.assert_cache_variant_for_arm)
                "protocol": args.protocol,
                "protocol_train_sessions": list(train_names),
                "reference_cache": str(reference_cache),
                "reference_variant": "f_labelfree",
                "e0_difference_law": (
                    "asserted at build time: the NORM_ONLY E0 must differ from "
                    "the f_labelfree (ACTIVITY_ONLY) E0 -- NORM_ONLY carries no "
                    "encoder pathway, so a byte-identical E0 would mean the "
                    "rung was silently built as ACTIVITY_ONLY"
                ),
                "e0_difference_summary": {
                    "n_sessions": len(norm_only_e0_diffs),
                    "min_max_abs_diff": float(min(
                        d["max_abs_diff"] for d in norm_only_e0_diffs.values())),
                    "max_max_abs_diff": float(max(
                        d["max_abs_diff"] for d in norm_only_e0_diffs.values())),
                    "sessions_differing": int(sum(
                        d["n_differing_rows"] > 0
                        for d in norm_only_e0_diffs.values())),
                },
                "session_log": norm_only_log,
                "ladder_position": norm_only.LADDER_LABEL,
                "component_role": "zero-calibration baseline rung of the "
                                  "component ablation (plan.COMPONENT_ABLATION"
                                  "['ladder']['NORM_ONLY'])",
                **estimator,
            }
        elif variant == "equiv_zero":
            estimator = {
                **equiv_zero.projection_receipt(student, fixed_projection),
                "protocol": None,
                "carrier": "all-zero [n_pad, 4]",
                "e0": "fixed random projection of cat(pre_pool(calib).mean, "
                      "zero side), padded to [n_pad, 50]",
                "label_disclosure": "no behavioral labels anywhere; no "
                                    "trained identity: the projection is "
                                    "drawn at seed 42 and never trained",
                "component_role": "capacity/pathway control of the component "
                                  "ablation (plan.COMPONENT_ABLATION)",
                **estimator,
            }
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
        if variant == "norm_only":
            zs = [log["z_summary"] for log in norm_only_log.values()]
            print(f"  norm_only z range over {len(zs)} sessions: "
                  f"[{min(z['min'] for z in zs):.3f}, "
                  f"{max(z['max'] for z in zs):.3f}] "
                  f"undefined_slots="
                  f"{sorted({log['n_undefined_slots'] for log in norm_only_log.values()})} "
                  f"e0_vs_f_labelfree max_abs_diff=["
                  f"{min(d['max_abs_diff'] for d in norm_only_e0_diffs.values()):.4f}, "
                  f"{max(d['max_abs_diff'] for d in norm_only_e0_diffs.values()):.4f}]",
                  flush=True)
        print(f"wrote {dest} variant={variant} "
              f"elapsed_s={time.monotonic() - variant_started:.1f}", flush=True)
    del student
    print(f"total build elapsed_s={time.monotonic() - build_started:.1f}", flush=True)


if __name__ == "__main__":
    main()
