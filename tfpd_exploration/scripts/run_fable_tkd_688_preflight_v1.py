#!/usr/bin/env python3
"""FABLE TKD 688 pre-flight: closed-form floor diagnostic (CPU, no training).

Measures whether a small closed-form decoder has ANY floor on the established
688 dev-6 face (the M2 wall was exactly this).  Face mirrored read-only from
the sua_exploration SPINT+T4 mainline: strict 27/6 manifest split, 20 ms
bins, W=50, chronological first-30 rewarded calibration trials, eval windows
from trials[30:] only, train-only behavior standardization, per-session
variance-weighted R^2 over all window bins, equal-session mean.

Baselines (all closed-form, zero gradient):
  PV    classic depth-weighted population vector from the pool-30 T4 cache:
        y(t) = sum_i rho_i (r_i(t) - b_i) [a_i, c_i]  (raw + affine-calibrated
        on the session's own calibration block, the T4 family's label budget)
  RIDGE per-session ridge from per-unit per-bin rates -> standardized
        velocity, fit on the calibration block only, lambda by GCV
        (a literal pooled-27 fixed-N ridge is ill-posed: sessions carry
        different unit counts; disclosed)
  PV-R  pooled-27 shared-feature ridge (the two PV channels + mean rate +
        intercept -> velocity), the well-posed reading of "trained on the 27
        train sessions"

Preregistered fork: best baseline dev-face mean >= 0.15 -> 688 has a floor +
known identity content -> report and STOP (full line needs a planner/user
order).  < 0.15 -> the M2 wall repeats -> decoder-innovation line fully
exhausted on all four datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "sua_exploration")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

SCHEMA = "fable_tkd_688_preflight_v1"
FORK_THRESHOLD = 0.15
BIN_SIZE_MS = 20
WINDOW = 50
CALIB_N = 30
POOL_SIZE = 30
MAX_TRIAL_LENGTH = 100
PAD_VALUE = -1.0
TRIAL_RESULT_FILTER = "R"
SIGNAL_VIEW = "sua"
MANIFEST_RELATIVE = "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
DATA_RELATIVE = "sua_exploration/data/dandi_000688/sub-C"
SEALED_REFS_RELATIVE = "sua_exploration/results/sua_spint_t4_mainline_fp32_v1"
LAMBDA_GRID = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)


class PreFlightError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreFlightError(message)


def receipt_body(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_receipt(path: Path, payload: object, *, exclusive: bool = False) -> str:
    import os
    import tempfile

    path = Path(path)
    if exclusive and path.exists():
        raise PreFlightError(f"refusing to overwrite existing receipt: {path}")
    body = receipt_body(payload)
    digest = hashlib.sha256(body).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_temporary = temporary.with_name(temporary.name + ".sha256")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o444)
        with open(sidecar_temporary, "wb") as handle:
            handle.write(f"{digest}  {path.name}\n".encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        sidecar_temporary.replace(sidecar)
        sidecar.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)
        sidecar_temporary.unlink(missing_ok=True)
    return digest


def sealed_reference() -> dict:
    """SPINT+T4 family sealed dev-6 numbers, read from the aggregate inputs."""
    import numpy as np

    sys.path.insert(0, str(REPO_ROOT / "sua_exploration/scripts"))
    from aggregate_sua_spint_t4_mainline import load_artifact

    out = {}
    for arm in ("b0", "t4", "ts4"):
        per_seed = []
        per_session: dict[str, list[float]] = {}
        for seed in (42, 43, 44):
            path = (REPO_ROOT / SEALED_REFS_RELATIVE / f"{arm}_s{seed}.json")
            sessions, values = load_artifact(path, arm, seed)
            per_seed.append(float(values.mean()))
            for name, value in zip(sessions, values):
                per_session.setdefault(name, []).append(float(value))
        out[arm] = {
            "dev_mean": float(np.mean(per_seed)),
            "per_seed_mean": per_seed,
            "per_session_mean": {k: float(np.mean(v))
                                 for k, v in sorted(per_session.items())},
        }
    out["t4_minus_b0"] = out["t4"]["dev_mean"] - out["b0"]["dev_mean"]
    out["source"] = {
        "aggregate_inputs": SEALED_REFS_RELATIVE + "/{b0,t4,ts4}_s{42,43,44}.json",
        "aggregator": "sua_exploration/scripts/aggregate_sua_spint_t4_mainline.py::load_artifact",
        "law": "mean over scored epochs 5-12, per-session, 3 seeds",
    }
    return out


def session_t4_and_rho(nwb_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pool-30 T4 (the sealed family law) + per-unit fit R^2 (M2 rho law)."""
    import numpy as np
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.unit_side_features import (
        CANONICAL_DIRECTIONS_RAD,
        _fit_cosine_tuning,
        _nearest_canonical_direction_index,
        _pool_trial_rate_matrix,
    )

    pool = list_datamodule_rewarded_trials(
        nwb_path, bin_size_ms=BIN_SIZE_MS, window_size=WINDOW,
        trial_result_filter=TRIAL_RESULT_FILTER,
    )
    require(len(pool) >= POOL_SIZE, f"{nwb_path.name}: pool smaller than 30")
    pool = pool[:POOL_SIZE]
    # The established law maps unlabeled trials to direction index -1 and
    # excludes them from the fit (present_directions) -- mirror verbatim.
    direction = np.asarray(
        [-1 if t.get("target_dir") is None
         else _nearest_canonical_direction_index(t["target_dir"]) for t in pool],
        dtype=np.int64,
    )
    rates, num_units = _pool_trial_rate_matrix(nwb_path, pool)  # [N, 30] Hz
    present = sorted({int(i) for i in direction if i >= 0})
    require(len(present) >= 2, f"{nwb_path.name}: degenerate direction pool")
    theta_by_direction = np.asarray([CANONICAL_DIRECTIONS_RAD[i] for i in present])
    design = np.stack(
        [np.ones_like(theta_by_direction), np.cos(theta_by_direction),
         np.sin(theta_by_direction)], axis=1,
    )
    t4 = np.zeros((num_units, 4), dtype=np.float64)
    rho = np.zeros(num_units, dtype=np.float64)
    for unit in range(num_units):
        per_direction = np.asarray([
            rates[unit][direction == index].mean() for index in present
        ])
        a, c, m, b = _fit_cosine_tuning(theta_by_direction, per_direction)
        t4[unit] = (a, c, m, b)
        # rho: R^2 of the cosine fit on the labelled POOL TRIALS (M2 law).
        valid = direction >= 0
        trial_theta = np.asarray(
            [CANONICAL_DIRECTIONS_RAD[int(i)] for i in direction[valid]],
            dtype=np.float64,
        )
        trial_design = np.stack(
            [np.ones_like(trial_theta), np.cos(trial_theta), np.sin(trial_theta)],
            axis=1,
        )
        observed = rates[unit][valid]
        predicted = trial_design @ np.asarray([b, a, c])
        residual = float(np.square(observed - predicted).sum())
        centered = observed - observed.mean()
        total = float(np.square(centered).sum())
        rho[unit] = 0.0 if total <= 0 else max(0.0, min(1.0, 1.0 - residual / total))
    return t4, rho, {"pool_trials": len(pool), "present_directions": len(present),
                     "num_units": int(num_units)}


def _r2(prediction, target) -> float:
    import numpy as np

    from m1_h1_activity_headroom_v1.core import variance_weighted_r2

    return variance_weighted_r2(
        np.asarray(target, dtype=np.float32), np.asarray(prediction, dtype=np.float32)
    )


def run(repo_root: Path) -> dict:
    import numpy as np
    from mc_maze.multisession_datamodule import (
        _fit_behavior_stats_uncached,
        session_name_from_path,
    )

    sys.path.insert(0, str(REPO_ROOT / "tfpd_exploration/src"))

    data_dir = repo_root / DATA_RELATIVE
    manifest = json.loads((repo_root / MANIFEST_RELATIVE).read_text("utf-8"))
    train_names = list(manifest["session_splits"]["train"])
    val_names = list(manifest["session_splits"]["val"])
    require(len(train_names) == 27 and len(val_names) == 6, "split drift")
    train_files = [data_dir / f"{name}_behavior+ecephys.nwb" for name in train_names]
    for path in train_files:
        require(path.is_file(), f"missing NWB {path}")

    # Import the established session loader (read-only uncached path).
    sys.path.insert(0, str(repo_root / "sua_exploration/scripts"))
    from eval_adaptation_dandi688 import _load_session_with_trials_uncached

    behavior_mean, behavior_std = _fit_behavior_stats_uncached(
        train_files, BIN_SIZE_MS
    )

    def load(name: str) -> dict:
        path = data_dir / f"{name}_behavior+ecephys.nwb"
        require(path.is_file(), f"missing NWB {path}")
        record = _load_session_with_trials_uncached(
            path, BIN_SIZE_MS, WINDOW, CALIB_N, MAX_TRIAL_LENGTH, PAD_VALUE,
            behavior_mean, behavior_std, TRIAL_RESULT_FILTER, SIGNAL_VIEW,
        )
        require(session_name_from_path(path) == name, "session name drift")
        return record

    bin_hz = 1.0 / (BIN_SIZE_MS / 1000.0)

    # --- pooled-27 PV-feature ridge (train sessions; subsampled windows) ---
    pooled_features: list[np.ndarray] = []
    pooled_targets: list[np.ndarray] = []
    for name in train_names:
        record = load(name)
        t4, rho, _info = session_t4_and_rho(data_dir / f"{name}_behavior+ecephys.nwb")
        eval_trials = record["trials"][CALIB_N:]
        starts = []
        for trial in eval_trials:
            starts.extend(range(trial["start"], trial["stop"] - WINDOW + 1, 20))
        if not starts:
            continue
        neural = record["neural"]
        behavior = record["behavior"]
        starts = np.asarray(starts, dtype=np.int64)
        step = max(1, starts.size // 400)
        starts = starts[::step][:400]
        rates = np.stack([neural[s : s + WINDOW].astype(np.float64) * bin_hz
                          for s in starts])           # [n, W, N] Hz
        dev = rates - t4[None, None, :, 3]
        pv_x = (rho[None, None, :] * dev * t4[None, None, :, 0]).sum(axis=2)
        pv_y = (rho[None, None, :] * dev * t4[None, None, :, 1]).sum(axis=2)
        mean_rate = rates.mean(axis=2)
        feats = np.stack([pv_x, pv_y, mean_rate], axis=2).reshape(-1, 3)
        pooled_features.append(feats)
        pooled_targets.append(np.stack(
            [behavior[s : s + WINDOW] for s in starts], axis=0
        ).reshape(-1, 2))
    x_pool = np.concatenate(pooled_features, axis=0)
    y_pool = np.concatenate(pooled_targets, axis=0)
    xm, ym = x_pool.mean(axis=0), y_pool.mean(axis=0)
    xc, yc = x_pool - xm, y_pool - ym
    gram = xc.T @ xc
    moments = xc.T @ yc
    best = None
    for lam in LAMBDA_GRID:
        system = gram + lam * x_pool.shape[0] * np.eye(3)
        beta = np.linalg.solve(system, moments)
        residual = float(np.square(xc @ beta - yc).sum())
        hat = float(np.trace(np.linalg.solve(system, gram)))
        denominator = (x_pool.shape[0] - hat) ** 2
        gcv = residual / denominator if denominator > 0 else float("inf")
        if best is None or gcv < best[0]:
            best = (gcv, lam, beta, ym - xm @ beta)
    _, pooled_lam, pooled_beta, pooled_intercept = best

    # --- dev-6 evaluation ---------------------------------------------------
    rows = []
    per_baseline: dict[str, list[float]] = {"pv_raw": [], "pv_calibrated": [],
                                            "ridge_calib": [], "pv_ridge_pooled": []}
    for name in val_names:
        record = load(name)
        t4, rho, info = session_t4_and_rho(data_dir / f"{name}_behavior+ecephys.nwb")
        eval_trials = record["trials"][CALIB_N:]
        require(eval_trials, f"{name}: no post-calibration trials")
        starts = np.asarray(
            [s for trial in eval_trials
             for s in range(trial["start"], trial["stop"] - WINDOW + 1)],
            dtype=np.int64,
        )
        neural = record["neural"].astype(np.float64)
        behavior = record["behavior"].astype(np.float64)
        rates = np.stack([neural[s : s + WINDOW] * bin_hz for s in starts])
        targets = np.stack([behavior[s : s + WINDOW] for s in starts])
        flat_rates = rates.reshape(-1, rates.shape[-1])          # [n*W, N]
        flat_targets = targets.reshape(-1, 2)
        dev = flat_rates - t4[None, :, 3]
        pv = np.stack([
            (rho[None, :] * dev * t4[None, :, 0]).sum(axis=1),
            (rho[None, :] * dev * t4[None, :, 1]).sum(axis=1),
        ], axis=1)
        scores = {"pv_raw": _r2(pv, flat_targets)}
        # affine calibration on the session's own calibration block (the T4
        # family's label budget: first-30 rewarded trials' bins).
        calib_trials = record["trials"][:CALIB_N]
        calib_starts = np.asarray(
            [s for trial in calib_trials
             for s in range(trial["start"], trial["stop"] - WINDOW + 1, 20)],
            dtype=np.int64,
        )
        calib_rates = np.stack(
            [neural[s : s + WINDOW] * bin_hz for s in calib_starts]
        ).reshape(-1, rates.shape[-1])
        calib_targets = np.stack(
            [behavior[s : s + WINDOW] for s in calib_starts]
        ).reshape(-1, 2)
        calib_dev = calib_rates - t4[None, :, 3]
        calib_pv = np.stack([
            (rho[None, :] * calib_dev * t4[None, :, 0]).sum(axis=1),
            (rho[None, :] * calib_dev * t4[None, :, 1]).sum(axis=1),
        ], axis=1)

        def _affine(fit_x: np.ndarray, fit_y: np.ndarray, x: np.ndarray) -> np.ndarray:
            out = np.empty_like(x)
            for dim in range(x.shape[1]):
                var = fit_x[:, dim].var()
                if var <= 0:
                    out[:, dim] = fit_y[:, dim].mean()
                    continue
                slope = ((fit_x[:, dim] - fit_x[:, dim].mean())
                         * (fit_y[:, dim] - fit_y[:, dim].mean())).mean() / var
                out[:, dim] = (slope * x[:, dim] + fit_y[:, dim].mean()
                               - slope * fit_x[:, dim].mean())
            return out

        scores["pv_calibrated"] = _r2(
            _affine(calib_pv, calib_targets, pv), flat_targets
        )
        # per-session calib ridge (rates -> velocity), GCV lambda
        xm_c, ym_c = calib_rates.mean(axis=0), calib_targets.mean(axis=0)
        xc_c, yc_c = calib_rates - xm_c, calib_targets - ym_c
        gram_c = xc_c.T @ xc_c
        moments_c = xc_c.T @ yc_c
        best_c = None
        for lam in LAMBDA_GRID:
            system = gram_c + lam * xc_c.shape[0] * np.eye(gram_c.shape[0])
            try:
                beta_c = np.linalg.solve(system, moments_c)
            except np.linalg.LinAlgError:
                continue
            residual = float(np.square(xc_c @ beta_c - yc_c).sum())
            hat = float(np.trace(np.linalg.solve(system, gram_c)))
            denominator = (xc_c.shape[0] - hat) ** 2
            gcv = residual / denominator if denominator > 0 else float("inf")
            if best_c is None or gcv < best_c[0]:
                best_c = (gcv, lam, beta_c, ym_c - xm_c @ beta_c)
        _, ridge_lam, ridge_beta, ridge_intercept = best_c
        scores["ridge_calib"] = _r2(
            flat_rates @ ridge_beta + ridge_intercept, flat_targets
        )
        # pooled-27 PV-feature ridge
        pooled = np.stack([pv[:, 0], pv[:, 1], flat_rates.mean(axis=1)], axis=1)
        scores["pv_ridge_pooled"] = _r2(
            pooled @ pooled_beta + pooled_intercept, flat_targets
        )
        for key, value in scores.items():
            per_baseline[key].append(float(value))
        rows.append({
            "session": name, "n_windows": int(starts.size), **info,
            "rho_median": float(np.median(rho)),
            "scores": scores,
        })
        print(f"{name}: " + " ".join(f"{k}={v:+.4f}" for k, v in scores.items()),
              flush=True)

    summary = {key: {"equal_session_mean": float(np.mean(values)),
                     "per_session": dict(zip(val_names, (round(v, 4) for v in values)))}
               for key, values in per_baseline.items()}
    best_mean = max(item["equal_session_mean"] for item in summary.values())
    return {
        "face": {
            "manifest": MANIFEST_RELATIVE,
            "bin_size_ms": BIN_SIZE_MS, "window": WINDOW,
            "calibration_n": CALIB_N, "pool_size": POOL_SIZE,
            "eval_windows": "trials[30:] only, stride 1 within trials",
            "behavior_standardization": "train-only per-dim mean/std",
            "signal_view": SIGNAL_VIEW, "trial_filter": TRIAL_RESULT_FILTER,
            "metric": "variance_weighted_r2 over all window bins, per session",
            "sealed_refs": sealed_reference(),
        },
        "baselines": summary,
        "pooled_pv_ridge_lambda": pooled_lam,
        "best_baseline_mean": float(best_mean),
        "rows": rows,
        "fork_rule": {
            "threshold": FORK_THRESHOLD,
            "best_baseline_mean": float(best_mean),
            "pass": bool(best_mean >= FORK_THRESHOLD),
            "outcome": "FLOOR_PRESENT_REPORT_AND_STOP" if best_mean >= FORK_THRESHOLD
                       else "M2_WALL_REPEATS_LINE_EXHAUSTED_TERMINAL_CLOSURE",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "dry": True, "schema": SCHEMA,
            "steps": ["attempt.json (O_EXCL)", "train-only behavior stats (27 NWB)",
                      "pooled-27 PV-feature ridge", "dev-6 PV/ridge/pooled evaluation",
                      "fork rule: best baseline >= 0.15"],
            "baselines": ["pv_raw", "pv_calibrated", "ridge_calib", "pv_ridge_pooled"],
        }, indent=2))
        return 0

    import time

    started = time.monotonic()
    root = REPO_ROOT / "tfpd_exploration/results/fable_tkd_688_preflight_v1"
    try:
        if not (root / "attempt.json").exists():
            atomic_receipt(root / "attempt.json", {
                "schema": f"{SCHEMA}:attempt",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "closed-form floor diagnostic on the 688 dev-6 face",
                "fork_rule": {"threshold": FORK_THRESHOLD},
            }, exclusive=True)
        result = run(REPO_ROOT)
        terminal = {
            "schema": f"{SCHEMA}:terminal",
            "status": "TERMINAL",
            **result,
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_receipt(root / "terminal.json", terminal, exclusive=True)
        print(json.dumps({
            "best_baseline_mean": result["best_baseline_mean"],
            "baselines": {k: round(v["equal_session_mean"], 4)
                          for k, v in result["baselines"].items()},
            "sealed": {k: round(result["face"]["sealed_refs"][k]["dev_mean"], 4)
                       for k in ("b0", "t4", "ts4")},
            "fork": result["fork_rule"],
        }, indent=2))
        return 0
    except Exception as error:  # noqa: BLE001
        atomic_receipt(root / "failure.json", {
            "schema": f"{SCHEMA}:failure",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
