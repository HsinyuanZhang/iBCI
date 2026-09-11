#!/usr/bin/env python3
"""Build a 688 prepared-cache variant (carrier + remelted E0 only).

Copies neural/behavior/starts/mask from the frozen contract-v2 cache.
Rewrites carrier and E0.  Writes a new prepared_contract.json that still
satisfies bench verify_prepared_cache (same schema/manifest/split counts).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
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

from dandi688_bench_v1 import plan
from btransform_unified_v2 import carrier_profile_v3 as v3

R700_SECONDS = 0.7
H300_SECONDS = 0.3
N0 = 1.0
VARIANTS = ("u1_m10", "u1_m30", "t4_dir16")


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


def _variant_spec(variant: str) -> dict[str, Any]:
    if variant == "u1_m10":
        return {"positions": tuple(range(10)), "namespace": "candidate", "label_budget": 10}
    if variant == "u1_m30":
        return {"positions": tuple(range(30)), "namespace": "reliability_audit", "label_budget": 30}
    if variant == "t4_dir16":
        # C1 (user directive 2026-09-10): same M10 candidate support face as
        # u1_m10/the t4 arm; only the direction design doubles to 16 bins.
        return {"positions": tuple(range(10)), "namespace": "candidate",
                "label_budget": 10, "n_dirs": 16}
    raise ValueError(variant)


def _raw_u1_profile(nwb_path: Path, spec: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    from mc_maze.dandi688_sparse_event_t4_v1.descriptors import (
        _fit_t4,
        _phase_trials,
        _window_trials,
    )
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.unit_side_features import _pool_trial_rate_matrix
    from mc_maze.dandi688_sparse_event_t4_v1 import plan as se_plan

    trials = list_datamodule_rewarded_trials(
        nwb_path, bin_size_ms=se_plan.BIN_SIZE_MS, window_size=se_plan.WINDOW_SIZE_BINS,
        trial_result_filter=se_plan.REWARDED_RESULT,
    )
    selected, exclusions = _phase_trials(
        trials, support_positions=spec["positions"], namespace=spec["namespace"]
    )
    direction = np.asarray([int(row["_direction_index"]) for row in selected], dtype=np.int64)
    rates = {
        phase: _pool_trial_rate_matrix(nwb_path, _window_trials(selected, phase=phase))[0]
        for phase in ("h300", "r700")
    }
    r700 = np.asarray(rates["r700"], dtype=np.float64).T
    h300 = np.asarray(rates["h300"], dtype=np.float64).T
    z_r, rate_mean, noise_rate = v3.poisson_standardize(r700, R700_SECONDS)
    z_h = v3.apply_affine(h300, rate_mean, noise_rate)
    weights = v3.one_hot_directions(direction)
    response = v3.conditional_response(z_r, weights, N0)
    hold = v3.conditional_response(z_h, np.ones((len(selected), 1), dtype=np.float64), N0)
    a, c, m = v3.harmonic_readout(response)
    baseline = response.mean(axis=1) - hold[:, 0]
    raw = v3.stack_t4(a, c, m, baseline)

    identity: dict[str, Any] = {"n_legal": int(len(selected)), "exclusions": exclusions}
    present = sorted(set(direction.tolist()))
    identity["n_dirs"] = int(len(present))
    if present == list(range(8)):
        legacy_r = _fit_t4(rates["r700"], direction)
        legacy_h = _fit_t4(rates["h300"], direction)
        legacy = np.ascontiguousarray(
            np.column_stack((legacy_r[:, 0], legacy_r[:, 1], legacy_r[:, 2],
                             legacy_r[:, 3] - legacy_h[:, 3])),
            dtype=np.float32,
        )
        ident = v3.harmonic_t4_from_rows(
            r700, direction, duration_s=None, n0=0.0,
            baseline=(
                v3.conditional_response(r700, weights, 0.0).mean(axis=1)
                - v3.conditional_response(h300, weights, 0.0).mean(axis=1)
            ),
        )
        max_abs = float(np.max(np.abs(ident.astype(np.float64) - legacy.astype(np.float64))))
        identity["legacy_max_abs"] = max_abs
        if max_abs > 1e-4:
            raise AssertionError(f"{nwb_path.name}: identity gate failed max_abs={max_abs}")
    return raw, identity


def _raw_dir16_profile(nwb_path: Path, spec: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """t4_dir16 (user directive 2026-09-10, plan C1): identical recipe to
    _raw_u1_profile except the direction design doubles to 16 angle bins
    (one-hot weights + 16-entry first-harmonic table, dandi688_bench_v1.dir16).

    Degeneracy law (dir16 module docstring): on this dataset every target_dir
    is exactly an 8-canonical angle landing on an EVEN 16-bin, so R16[:,2k]
    == R8[:,k] exactly; a/c/m halve exactly (2/16 vs 2/8 table mean) and the
    train-fit column normalizer cancels that power-of-two scale.  The fourth
    column genuinely moves: b16 = M8/2 - H while b8 = M8 - H, a shift of
    -H/2 (H = the H300 single-state hold response)."""
    from dandi688_bench_v1 import dir16
    from mc_maze.dandi688_sparse_event_t4_v1.descriptors import (
        _phase_trials,
        _window_trials,
    )
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.unit_side_features import _pool_trial_rate_matrix
    from mc_maze.dandi688_sparse_event_t4_v1 import plan as se_plan

    trials = list_datamodule_rewarded_trials(
        nwb_path, bin_size_ms=se_plan.BIN_SIZE_MS, window_size=se_plan.WINDOW_SIZE_BINS,
        trial_result_filter=se_plan.REWARDED_RESULT,
    )
    selected, exclusions = _phase_trials(
        trials, support_positions=spec["positions"], namespace=spec["namespace"]
    )
    target_dirs = np.asarray([float(row["target_dir"]) for row in selected], dtype=np.float64)
    direction16 = dir16.direction16_indices(target_dirs)
    rates = {
        phase: _pool_trial_rate_matrix(nwb_path, _window_trials(selected, phase=phase))[0]
        for phase in ("h300", "r700")
    }
    r700 = np.asarray(rates["r700"], dtype=np.float64).T
    h300 = np.asarray(rates["h300"], dtype=np.float64).T
    z_r, rate_mean, noise_rate = v3.poisson_standardize(r700, R700_SECONDS)
    z_h = v3.apply_affine(h300, rate_mean, noise_rate)
    weights = dir16.one_hot_directions16(direction16)
    response = v3.conditional_response(z_r, weights, N0)
    hold = v3.conditional_response(z_h, np.ones((len(selected), 1), dtype=np.float64), N0)
    a, c, m = dir16.harmonic_readout16(response)
    baseline = response.mean(axis=1) - hold[:, 0]
    raw = v3.stack_t4(a, c, m, baseline)

    present = sorted(set(direction16.tolist()))
    identity: dict[str, Any] = {
        "n_legal": int(len(selected)),
        "exclusions": exclusions,
        "n_dirs16_present": int(len(present)),
        "odd_bins_occupied": int(len({k for k in present if k % 2 == 1})),
    }
    return raw, identity


def _dir16_degeneracy_audit(
    frozen_rows: dict[str, dict[str, Any]],
    raw16_rows: dict[str, np.ndarray],
    raw8_rows: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, Any]:
    """Honest evidence of what the 16-bin remap changes (see dir16 docstring).

    (1) raw half-scale law: per column max |raw16 - raw8/2| over every real
        row of every session -- expected ~0 for a/c/m, |hold|/2 for column b.
    (2) end-to-end: per column max |normalized dir16 carrier - frozen cache
        carrier| over real rows -- the delta the trained arm actually sees
        relative to the 8-dir t4 bytes."""
    half_dev = np.zeros(plan.CARRIER_DIM, dtype=np.float64)
    end_to_end = np.zeros(plan.CARRIER_DIM, dtype=np.float64)
    for name, row in frozen_rows.items():
        mask = np.asarray(row["mask"], dtype=bool)
        n_real = int(mask.sum())
        raw16, raw8 = raw16_rows[name], raw8_rows[name]
        half_dev = np.maximum(
            half_dev,
            np.abs(raw16[:n_real].astype(np.float64) - raw8[:n_real].astype(np.float64) / 2.0).max(axis=0),
        )
        norm16 = v3.apply_column_normalizer(raw16, mean, std)[:n_real].astype(np.float64)
        frozen = np.asarray(row["carrier"], dtype=np.float64)[:n_real]
        end_to_end = np.maximum(end_to_end, np.abs(norm16 - frozen).max(axis=0))
    return {
        "raw_half_scale_max_abs_per_column": half_dev.tolist(),
        "raw_half_scale_reading": "columns a/c/m expected ~0 (exact power-of-"
                                  "two halving); column b expected |hold|/2 "
                                  "(b16 = M8/2 - H vs b8 = M8 - H)",
        "normalized_vs_frozen_max_abs_per_column": end_to_end.tolist(),
        "normalized_vs_frozen_reading": "end-to-end delta against the frozen "
                                        "8-dir t4 carrier bytes per column",
        "provenance": "dandi688_bench_v1.dir16 degeneracy audit (2026-09-10)",
    }


def _nwb_path(session_id: str) -> Path:
    from mc_maze.dandi688_sparse_event_t4_v1 import plan as se_plan
    return WS / se_plan.DATA_RELATIVE / f"{session_id}_behavior+ecephys.nwb"


def _remelt_e0(student, calib_trials: np.ndarray, carrier_real: np.ndarray, n_pad: int) -> np.ndarray:
    import torch

    units = int(calib_trials.shape[-1])
    cal = torch.from_numpy(np.asarray(calib_trials, dtype=np.float32)).unsqueeze(0)
    c = torch.from_numpy(np.asarray(carrier_real, dtype=np.float32)).unsqueeze(0)
    with torch.inference_mode():
        pooled = student.id_encoder.pre_pool(cal.permute(0, 1, 3, 2)).mean(1)
        e = student.id_encoder.post_pool(torch.cat((pooled, c), -1))[0].cpu().numpy()
    e0 = np.zeros((n_pad, plan.E0_DIM), dtype=np.float32)
    e0[:units] = e
    return e0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), required=True)
    parser.add_argument("--dest", type=Path, help="destination cache dir (required unless --variant all)")
    parser.add_argument("--dest-root", type=Path, default=PKG_ROOT / "results",
                        help="parent dir for cache_<variant> when --variant all")
    parser.add_argument("--source-cache", type=Path, default=plan.prepared_cache_path())
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    dests = {}
    if args.variant == "all":
        for name in variants:
            dests[name] = (args.dest_root / f"cache_{name}").resolve()
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

    frozen_meta, frozen_rows = _load_frozen(args.source_cache)
    train_names = [n for n, r in frozen_rows.items() if r["split"] == "train"]
    if len(train_names) != 27:
        raise RuntimeError(f"expected 27 train sessions, got {len(train_names)}")

    from mc_maze.dandi688_sparse_event_t4_v1.production import prepare_source_surface
    from mc_maze.dandi688_cp_film_v1.runner import _prepare_student

    print("prepare_source_surface for calib_trials", flush=True)
    surface = prepare_source_surface(WS, signal_view="sua", reliability_mask=(True, True, True, True))
    print("loading frozen B3S student for E0 remelt", flush=True)
    student = _prepare_student(WS, plan.SEED, __import__("torch").device(args.device))

    for variant in variants:
        spec = _variant_spec(variant)
        dest = dests[variant]
        raw_rows: dict[str, np.ndarray] = {}
        identity_log: dict[str, Any] = {}
        # dir16 needs the SAME-face raw 8-dir profile as its degeneracy
        # comparator (the half-scale law is proven raw-vs-raw, see dir16.py)
        raw8_comparator: dict[str, np.ndarray] | None = {}
        for name in sorted(frozen_rows):
            if variant == "t4_dir16":
                raw, ident = _raw_dir16_profile(_nwb_path(name), spec)
                raw8, _ = _raw_u1_profile(_nwb_path(name), spec)
                raw8_comparator[name] = raw8
            else:
                raw, ident = _raw_u1_profile(_nwb_path(name), spec)
            raw_rows[name] = raw
            identity_log[name] = ident
            dirs_key = "n_dirs16_present" if variant == "t4_dir16" else "n_dirs"
            print(f"  {variant} raw {name} units={raw.shape[0]} dirs={ident[dirs_key]}", flush=True)

        mean, std = v3.fit_column_normalizer([raw_rows[n] for n in train_names])
        degeneracy_audit = (
            _dir16_degeneracy_audit(frozen_rows, raw_rows, raw8_comparator, mean, std)
            if variant == "t4_dir16" else None
        )
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "sessions").mkdir(exist_ok=True)
        rows_hashes: dict[str, dict[str, str]] = {}
        sessions_meta: dict[str, dict[str, str]] = {}
        for name, row in frozen_rows.items():
            mask = np.asarray(row["mask"], dtype=bool)
            n_real = int(mask.sum())
            carrier = np.zeros((mask.size, plan.CARRIER_DIM), dtype=np.float32)
            carrier[:n_real] = v3.apply_column_normalizer(raw_rows[name], mean, std)
            calib = surface.sessions[name].record.calib_trials
            if calib.shape[-1] != n_real:
                raise RuntimeError(f"{name}: calib units {calib.shape[-1]} != mask {n_real}")
            e0 = _remelt_e0(student, calib, carrier[:n_real], mask.size)
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

        contract = {
            "schema": plan.PREPARED_CACHE_SCHEMA,
            "manifest_sha256": plan.MANIFEST_SHA256,
            "split_counts": {"train": plan.SPLIT_COUNTS["train"], "val": plan.SPLIT_COUNTS["val"]},
            "formal_test_used": False,
            "sessions": sessions_meta,
            "rows_hashes": rows_hashes,
            "variant": variant,
            "estimator": {
                "name": "carrier_profile_v3_u1",
                "poisson_duration_s": R700_SECONDS,
                "n0": N0,
                "label_budget": spec["label_budget"],
                "namespace": spec["namespace"],
                "hold": "single_state_h300_same_affine",
                "normalizer_mean": mean.tolist(),
                "normalizer_std": std.tolist(),
                "source_cache": str(args.source_cache),
                "source_cache_contract_sha256": _file_sha256(args.source_cache / "prepared_contract.json"),
            "implementation_sha256": _file_sha256(Path(v3.__file__)),
            "builder_sha256": _file_sha256(Path(__file__)),
            "identity_log": identity_log,
            **({"n_dirs": 16,
                "dir16_module_sha256": _file_sha256(
                    PKG_ROOT / "src" / "dandi688_bench_v1" / "dir16.py"),
                "direction_mapping": "nearest 16-bin canonical of target_dir "
                                     "(-3pi/4 + k*pi/8); one-hot weights + "
                                     "16-entry first-harmonic closed-form "
                                     "readout; blocks/rate primitives, Poisson "
                                     "affine, n0 and M10 candidate support "
                                     "face all identical to u1_m10",
                "degeneracy_audit": degeneracy_audit}
               if variant == "t4_dir16" else {}),
            },
            "built_utc": datetime.now(timezone.utc).isoformat(),
        }
        _dump(dest / "prepared_contract.json", contract)
        print(f"wrote {dest} variant={variant}", flush=True)
    del student


if __name__ == "__main__":
    main()
