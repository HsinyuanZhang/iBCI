#!/usr/bin/env python3
"""Build the exp2016 extension prepared-cache (2016 remote exam, SUA t4).

User directive 2026-09-10 (coordinator ruling, plan A): train = the SAME 18
sessions of 2015 as exp2015_full; exam = the 14 CO sessions of 2016
(2016-09-09..2016-10-21) that sit OUTSIDE the frozen manifest entirely --
every one carries 188-353 NWB units and the frozen roster law
(multisession_datamodule.discover_nwb_files, max_units_exclusive=100)
excludes any session with >= 100 units, so they were never roster-eligible.
Admission law (ruled deterministic, zero discretion):

  2016 units selection: first-91-by-table-order, Nmax locked to train roster

i.e. each 2016 exam row keeps the FIRST 91 units of the NWB units table
(table order == neural column order of the frozen loader), matching the
frozen Nmax-91 row geometry; per-session firing-rate diagnostics of the
kept-91 vs dropped units are recorded in the contract to quantify any
systematic bias of the truncation.

Row construction laws (2015 train-only normalizer domain, per directive):
  - train rows (18): neural/behavior/starts/mask byte-copied from the frozen
    contract-v2 cache; the T4 carrier is recomputed from the NWB via the
    frozen sparse-event recipe (materialize_sparse_event_t4, candidate M10,
    R700/H300, delta_b) and column-normalized with a normalizer fit on the
    protocol's OWN 18 train sessions only; E0 is remelted with that carrier
    as side (frozen B3S student), so train carrier/E0 bytes DIFFER from the
    frozen cache's 27-session-domain bytes by design (protocol-bound
    normalizer, vstate-cache precedent);
  - a parity gate first proves the pipeline: the recomputed raw profiles of
    all 27 manifest-train sessions plus a 27-domain normalizer must
    reproduce the frozen cache's carrier bytes (max abs <= 1e-4);
  - exam rows (14): built from NWB via the frozen loader
    (load_dandi688_session: 20 ms bins, cursor_vel interpolated at bin
    centers, behavior normalized with the FROZEN 27-train-domain stats --
    train-side statistics only, no 2016 leakage, byte-consistent with the
    train rows' behavior), first 30 rewarded trials -> calib_trials, Q50
    starts from trial 50 on, T4/E0 under the same 18-domain normalizer.

The contract keeps schema/manifest_sha256/split_counts as ATTESTATION of
the unchanged source manifest (27/6) and adds an ``exp2016_extension`` block
naming the actual roster (18 train + 14 exam2016) and the selection law.
CPU-only; no training; no formal-test data (2016 sessions are disjoint from
the frozen formal-test split, asserted at plan level).
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
          WS / "btransform_unified_v1" / "src", WS / "sua_exploration",
          WS / "streaming_calibration_exp", WS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dandi688_bench_v1 import plan, vstate  # noqa: E402

N_PAD = 91                    # frozen Nmax of the contract-v2 cache
SUA_PARITY_MAX_ABS = 1.0e-4   # same tolerance family as build_pmua_cache
EXAM_UNIT_BUDGET = 91         # the ruling: first-91-by-table-order
UNITS_SELECTION_LAW = (
    "2016 units selection: first-91-by-table-order, Nmax locked to train "
    "roster (coordinator ruling 2026-09-10, plan A; deterministic, zero "
    "discretion)"
)


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


def _nwb_path(session_id: str) -> Path:
    from mc_maze.dandi688_sparse_event_t4_v1 import plan as se_plan
    return WS / se_plan.DATA_RELATIVE / f"{session_id}_behavior+ecephys.nwb"


def _load_frozen(cache: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    meta = json.loads((cache / "prepared_contract.json").read_text())
    if meta.get("schema") != plan.PREPARED_CACHE_SCHEMA:
        raise RuntimeError(f"frozen cache schema drift: {meta.get('schema')}")
    rows: dict[str, dict[str, Any]] = {}
    for name in meta["sessions"]:
        z = np.load(cache / "sessions" / f"{name}.npz")
        rows[name] = {
            "split": meta["sessions"][name]["split"],
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


def _raw_profile(session_id: str) -> np.ndarray:
    """Frozen sparse-event SUA T4 raw profile (pre-normalization) of one NWB."""
    from mc_maze.dandi688_sparse_event_t4_v1.descriptors import (
        materialize_sparse_event_t4,
    )

    descriptor = materialize_sparse_event_t4(
        _nwb_path(session_id), signal_view="sua", namespace="candidate"
    )
    return np.asarray(descriptor.raw_profile)


def _trial_bins(session_id: str, n_trials: int) -> list[dict[str, int]]:
    """First-n rewarded trials' (start, stop) bin spans (datamodule filter)."""
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.dandi688_sparse_event_t4_v1 import plan as se_plan

    trials = list_datamodule_rewarded_trials(
        _nwb_path(session_id), bin_size_ms=se_plan.BIN_SIZE_MS,
        window_size=se_plan.WINDOW_SIZE_BINS,
        trial_result_filter=se_plan.REWARDED_RESULT,
    )
    if len(trials) < n_trials:
        raise RuntimeError(f"{session_id}: {len(trials)} rewarded trials < {n_trials}")
    return [{"start": int(row["start"]), "stop": int(row["stop"])}
            for row in trials[:n_trials]]


def _q50_starts(session_id: str) -> np.ndarray:
    from mc_maze.dandi688_sparse_event_t4_v1.production import _q50_starts
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.dandi688_sparse_event_t4_v1 import plan as se_plan

    trials = list_datamodule_rewarded_trials(
        _nwb_path(session_id), bin_size_ms=se_plan.BIN_SIZE_MS,
        window_size=se_plan.WINDOW_SIZE_BINS,
        trial_result_filter=se_plan.REWARDED_RESULT,
    )
    return _q50_starts(trials)


def _frozen_behavior_stats() -> tuple[np.ndarray, np.ndarray]:
    """The datamodule's frozen train-only behavior normalization (27 train
    sessions of the manifest, 20 ms bins) -- recomputed deterministically
    (identical law to Dandi688MultiSessionDataModule.setup)."""
    from mc_maze.multisession_datamodule import fit_behavior_stats

    splits = json.loads(plan.manifest_path().read_text())["session_splits"]
    train_files = [_nwb_path(name) for name in splits["train"]]
    mean, std = fit_behavior_stats(train_files, 20, cache_dir=None)
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def _rate_diagnostics(neural: np.ndarray, n_kept: int, bin_size_s: float) -> dict[str, Any]:
    """Mean firing rate (Hz) per unit over the whole session; kept-vs-dropped
    distribution summary to quantify truncation bias (coordinator ruling
    item 2)."""
    rates = np.asarray(neural, dtype=np.float64).sum(axis=0) / (
        neural.shape[0] * bin_size_s)
    kept, dropped = rates[:n_kept], rates[n_kept:]

    def _summary(values: np.ndarray) -> dict[str, float]:
        return {
            "n": int(values.size),
            "mean_hz": float(np.mean(values)),
            "median_hz": float(np.median(values)),
            "q25_hz": float(np.percentile(values, 25)),
            "q75_hz": float(np.percentile(values, 75)),
            "min_hz": float(np.min(values)),
            "max_hz": float(np.max(values)),
        }

    return {
        "kept_first_91": _summary(kept),
        "dropped_remainder": _summary(dropped),
        "kept_minus_dropped_mean_hz": float(np.mean(kept) - np.mean(dropped)),
        "kept_over_dropped_mean_ratio": (
            float(np.mean(kept) / np.mean(dropped)) if np.mean(dropped) > 0 else None
        ),
        "reading": "positive kept_minus_dropped_mean_hz = the truncation "
                   "keeps LOWER-rate units (dropped units fire more); "
                   "quantifies the systematic bias of first-91 selection",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path,
                        default=PKG_ROOT / "results" / "cache_exp2016")
    parser.add_argument("--source-cache", type=Path, default=plan.prepared_cache_path())
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device != "cpu":
        raise RuntimeError("exp2016 cache build is CPU-only by contract")

    bench_root = (WS / "btransform_unified_v2" / "dandi688_bench_v1").resolve()
    dest = args.dest.resolve()
    if bench_root not in dest.parents:
        raise RuntimeError(f"--dest must live under {bench_root}")
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"destination must be empty: {dest}")

    plan.verify_protocol_definitions()
    train_sessions = plan.protocol_sessions("exp2016")["train"]
    exam_sessions = plan.protocol_sessions("exp2016")["exam"]
    frozen_meta, frozen_rows = _load_frozen(args.source_cache)
    manifest = json.loads(plan.manifest_path().read_text())["session_splits"]
    manifest_train = [n for n in manifest["train"]]
    print(f"exp2016 extension cache: train={len(train_sessions)} "
          f"exam={len(exam_sessions)} selection_law=first-91", flush=True)

    # --- raw T4 profiles of ALL 27 manifest-train sessions (parity + norms) --
    started = time.monotonic()
    raw_sua: dict[str, np.ndarray] = {}
    for name in sorted(manifest_train):
        raw_sua[name] = _raw_profile(name)
        print(f"  raw profile {name} units={raw_sua[name].shape[0]}", flush=True)
    print(f"raw profiles done in {time.monotonic() - started:.1f}s", flush=True)

    from mc_maze.dandi688_sparse_event_t4_v1.core import (
        fit_source_normalizer,
        normalize_columns,
    )

    # parity gate: 27-domain recomputation must reproduce the frozen bytes
    norm27 = fit_source_normalizer([raw_sua[n] for n in manifest_train])
    parity_max = 0.0
    for name in manifest_train:
        mask = np.asarray(frozen_rows[name]["mask"], dtype=bool)
        n_real = int(mask.sum())
        if raw_sua[name].shape[0] != n_real:
            raise RuntimeError(
                f"{name}: NWB units {raw_sua[name].shape[0]} != frozen mask "
                f"{n_real} (unit-axis alignment broken)")
        got = normalize_columns(raw_sua[name], **norm27)
        frozen_carrier = np.asarray(frozen_rows[name]["carrier"], dtype=np.float64)[:n_real]
        parity_max = max(parity_max, float(np.max(np.abs(
            got.astype(np.float64) - frozen_carrier))))
    if parity_max > SUA_PARITY_MAX_ABS:
        raise RuntimeError(f"SUA recipe parity gate failed max_abs={parity_max}")
    print(f"SUA parity gate PASS (max_abs={parity_max:.3e})", flush=True)

    # the protocol-bound 18-session normalizer (2015 train-only law)
    norm18 = fit_source_normalizer([raw_sua[n] for n in train_sessions])

    # --- frozen B3S student for the E0 remelt -------------------------------
    print("loading frozen B3S student for E0 remelt (cpu)", flush=True)
    import torch

    from mc_maze.dandi688_cp_film_v1.runner import _prepare_student

    student = _prepare_student(WS, plan.SEED, torch.device("cpu"))

    # --- frozen behavior stats for the NEW exam rows -------------------------
    print("computing frozen 27-train behavior stats (20 ms bins)", flush=True)
    b_mean, b_std = _frozen_behavior_stats()

    from mc_maze.multisession_datamodule import (
        _build_calib_trials,
        load_dandi688_session,
    )

    rows_hashes: dict[str, dict[str, str]] = {}
    sessions_meta: dict[str, dict[str, str]] = {}
    identity_log: dict[str, Any] = {}
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "sessions").mkdir(exist_ok=True)
    build_started = time.monotonic()

    # --- train rows (18): frozen bytes + 18-domain carrier/E0 ----------------
    for name in train_sessions:
        row = frozen_rows[name]
        mask = np.asarray(row["mask"], dtype=bool)
        n_real = int(mask.sum())
        if n_real > N_PAD:
            raise RuntimeError(f"{name}: real units {n_real} exceed Nmax {N_PAD}")
        calib_bins = _trial_bins(name, 30)
        calib = _build_calib_trials(
            np.asarray(row["neural"])[:, :n_real], calib_bins, 30, 100, n_real,
            pad_value=-1.0, interpolate_trials=True,
        )
        if calib.shape != (30, 100, n_real) or not np.isfinite(calib).all():
            raise RuntimeError(f"{name}: calib geometry/nonfinite failure")
        carrier_real = normalize_columns(raw_sua[name], **norm18).astype(np.float32)
        e0 = vstate.remelt_e0(student, calib, carrier_real, N_PAD)
        carrier = np.zeros((N_PAD, plan.CARRIER_DIM), dtype=np.float32)
        carrier[:n_real] = carrier_real
        packed = {
            "neural": row["neural"], "behavior": row["behavior"],
            "starts": row["starts"], "e0": e0, "carrier": carrier,
            "mask": row["mask"],
        }
        for key in ("neural", "behavior", "starts", "mask"):
            if plan.array_digest(packed[key]) != plan.array_digest(row[key]):
                raise RuntimeError(f"{name}:{key} drifted while copying")
        np.savez_compressed(dest / "sessions" / f"{name}.npz", **packed)
        rows_hashes[name] = {key: plan.array_digest(packed[key]) for key in packed}
        sessions_meta[name] = {"split": "train"}
        identity_log[name] = {
            "role": "train",
            "n_units": n_real,
            "carrier_normalizer_domain": "exp2016 train face (18 sessions)",
            "frozen_bytes_copied": ["neural", "behavior", "starts", "mask"],
            "recomputed_18_domain": ["carrier", "e0"],
        }
        print(f"  packed train {name} units={n_real}", flush=True)

    # --- exam rows (14): NWB-built, first-91 law ------------------------------
    for name in exam_sessions:
        record = load_dandi688_session(
            _nwb_path(name),
            bin_size_ms=20, window_size=plan.WINDOW_BINS,
            calibration_n_trials=30, max_trial_length=100,
            pad_value=-1.0, interpolate_trials=True,
            behavior_mean=b_mean, behavior_std=b_std,
            trial_result_filter="R", signal_view="sua", cache_dir=None,
        )
        n_total = int(record.neural.shape[1])
        if n_total < EXAM_UNIT_BUDGET:
            raise RuntimeError(
                f"{name}: {n_total} units < budget {EXAM_UNIT_BUDGET} "
                f"(selection law undefined below Nmax)")
        if n_total >= 100:
            # the roster-law conflict this cache exists to disclose; expected
            # for every 2016 session (188-353 units)
            pass
        neural = np.asarray(record.neural, dtype=np.float32)[:, :EXAM_UNIT_BUDGET]
        calib = np.asarray(record.calib_trials, dtype=np.float32)[:, :, :EXAM_UNIT_BUDGET]
        behavior = np.asarray(record.behavior, dtype=np.float32)
        starts = _q50_starts(name)
        if not np.isfinite(neural).all() or not np.isfinite(behavior).all():
            raise RuntimeError(f"{name}: nonfinite neural/behavior")
        if not np.isfinite(calib).all() or calib.shape != (30, 100, EXAM_UNIT_BUDGET):
            raise RuntimeError(f"{name}: calib geometry/nonfinite failure")
        raw_full = _raw_profile(name)
        if raw_full.shape[0] != n_total:
            raise RuntimeError(
                f"{name}: descriptor units {raw_full.shape[0]} != loader "
                f"units {n_total} (unit-axis alignment broken)")
        carrier_real = normalize_columns(
            raw_full[:EXAM_UNIT_BUDGET], **norm18).astype(np.float32)
        e0 = vstate.remelt_e0(student, calib, carrier_real, N_PAD)
        carrier = np.zeros((N_PAD, plan.CARRIER_DIM), dtype=np.float32)
        carrier[:EXAM_UNIT_BUDGET] = carrier_real
        mask = np.zeros(N_PAD, dtype=bool)
        mask[:EXAM_UNIT_BUDGET] = True
        packed = {
            "neural": neural, "behavior": behavior, "starts": starts,
            "e0": e0, "carrier": carrier, "mask": mask,
        }
        np.savez_compressed(dest / "sessions" / f"{name}.npz", **packed)
        rows_hashes[name] = {key: plan.array_digest(packed[key]) for key in packed}
        sessions_meta[name] = {"split": "exam2016"}
        identity_log[name] = {
            "role": "exam2016",
            "units_selection_law": UNITS_SELECTION_LAW,
            "n_units_total": n_total,
            "n_units_kept": EXAM_UNIT_BUDGET,
            "n_units_dropped": n_total - EXAM_UNIT_BUDGET,
            "roster_law_conflict": (
                "session carries >= 100 NWB units; the frozen roster law "
                "(discover_nwb_files max_units_exclusive=100) would exclude "
                "it -- admitted under the ruled first-91 selection"),
            "firing_rate_diagnostics": _rate_diagnostics(
                record.neural, EXAM_UNIT_BUDGET, 0.02),
            "n_rewarded_trials_first30_bins_source": "datamodule filter (R)",
            "q50_starts_n": int(len(starts)),
        }
        print(f"  packed exam {name} units={n_total}->kept {EXAM_UNIT_BUDGET} "
              f"q50={len(starts)}", flush=True)

    del student
    size_bytes = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    if size_bytes >= plan.PREPARED_CACHE_MAX_GIB * 1024 ** 3:
        raise RuntimeError(
            f"exp2016 cache exceeds {plan.PREPARED_CACHE_MAX_GIB} GiB "
            f"({size_bytes} bytes)")

    kept_rates = [identity_log[n]["firing_rate_diagnostics"]["kept_first_91"]["mean_hz"]
                  for n in exam_sessions]
    dropped_rates = [identity_log[n]["firing_rate_diagnostics"]["dropped_remainder"]["mean_hz"]
                     for n in exam_sessions]
    contract = {
        "schema": plan.PREPARED_CACHE_SCHEMA,
        "manifest_sha256": plan.MANIFEST_SHA256,
        "split_counts": {"train": plan.SPLIT_COUNTS["train"],
                         "val": plan.SPLIT_COUNTS["val"]},
        "split_counts_semantics": (
            "attestation of the unchanged SOURCE manifest (27/6); this "
            "extension cache itself holds 18 train + 14 exam2016 rows"
        ),
        "formal_test_used": False,
        "sessions": sessions_meta,
        "rows_hashes": rows_hashes,
        "variant": "exp2016_sua_t4",
        "estimator": {
            "name": "exp2016_extension_sua_t4",
            "protocol": "exp2016",
            "recipe": "train rows: frozen cache bytes (neural/behavior/starts/"
                      "mask) + sparse-event T4 carrier and B3S E0 remelt "
                      "recomputed under the 18-session 2015 train-only column "
                      "normalizer; exam rows: NWB-built via the frozen loader "
                      "(20 ms bins, cursor_vel at bin centers, frozen "
                      "27-train-domain behavior stats) under the same "
                      "carrier/E0 law",
            "units_selection_law": UNITS_SELECTION_LAW,
            "normalizer_domain": {
                "train_sessions": len(train_sessions),
                "law": "fit_source_normalizer over the exp2016 train face's "
                       "(= exp2015_full's 18 sessions) raw SUA T4 rows only; "
                       "no exam row enters any statistic",
            },
            "normalizer_mean": norm18["mean"].tolist(),
            "normalizer_scale": norm18["scale"].tolist(),
            "behavior_stats_domain": "frozen manifest 27-train sessions "
                                     "(train-side only; matches the frozen "
                                     "cache rows' behavior bytes)",
            "sua_parity_gate": {
                "law": "recomputed raw profiles of all 27 manifest-train "
                       "sessions + 27-domain normalizer vs the frozen cache "
                       "carrier bytes (max abs over sessions x real rows)",
                "max_abs": parity_max,
                "tolerance": SUA_PARITY_MAX_ABS,
                "pass": bool(parity_max <= SUA_PARITY_MAX_ABS),
            },
            "label_disclosure": "T4 consumes target_dir labels of the "
                                "first-M10 rewarded calibration trials; E0 "
                                "remelt consumes the frozen B3S student and "
                                "the 30-trial activity calibration window",
        },
        "exp2016_extension": {
            "protocol": "exp2016",
            "train_sessions": list(train_sessions),
            "exam_sessions": list(exam_sessions),
            "roster_law_conflict": (
                "every 2016 exam session carries 188-353 NWB units and "
                "violates the frozen <100-unit roster law; admitted under "
                "the ruled first-91-by-table-order selection with "
                "per-session firing-rate diagnostics"
            ),
            "firing_rate_summary": {
                "mean_kept_hz_across_sessions": float(np.mean(kept_rates)),
                "mean_dropped_hz_across_sessions": float(np.mean(dropped_rates)),
                "kept_over_dropped_mean_ratio": (
                    float(np.mean(kept_rates) / np.mean(dropped_rates))
                    if np.mean(dropped_rates) > 0 else None),
            },
            "source_cache": str(args.source_cache),
            "source_cache_contract_sha256": _file_sha256(
                args.source_cache / "prepared_contract.json"),
        },
        "identity_log": identity_log,
        "implementation_shas": {
            "descriptors_py": _file_sha256(
                WS / "sua_exploration/mc_maze/dandi688_sparse_event_t4_v1/descriptors.py"),
            "production_py": _file_sha256(
                WS / "sua_exploration/mc_maze/dandi688_sparse_event_t4_v1/production.py"),
            "se_core_py": _file_sha256(
                WS / "sua_exploration/mc_maze/dandi688_sparse_event_t4_v1/core.py"),
            "multisession_datamodule_py": _file_sha256(
                WS / "sua_exploration/mc_maze/multisession_datamodule.py"),
            "vstate_py": _file_sha256(Path(vstate.__file__)),
        },
        "builder_sha256": _file_sha256(Path(__file__)),
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "cache_size_bytes": size_bytes,
        "cache_size_mib": size_bytes / 1024 ** 2,
    }
    _dump(dest / "prepared_contract.json", contract)
    print(json.dumps(contract["exp2016_extension"]["firing_rate_summary"], indent=2),
          flush=True)
    print(f"wrote {dest} variant=exp2016_sua_t4 "
          f"size_mib={size_bytes / 1024 ** 2:.1f} "
          f"elapsed_s={time.monotonic() - build_started:.1f}", flush=True)


if __name__ == "__main__":
    main()
