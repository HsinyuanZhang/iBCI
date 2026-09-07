#!/usr/bin/env python3
"""FABLE TKD H1 pre-flight: parent-decoder diagnostic (CPU, no training).

Rebuilds the H-C carrier's PARENT decoder -- the frozen supervised ridge
(SVD-projected plan, q=16, lambda=100, empirical-Bayes family) that maps
100-ms block rates -> 7-DoF velocity, BEFORE the 4-column carrier
compression -- from the immutable source authority cache, and evaluates it
on the established 5-date LODO target face (same minival windows / targets /
variance-weighted R^2 as the h1 film/postpool machinery).  Zero-gradient,
per-target-date closed form; the plan is fit on fold-0 source sessions only
(date 19250101 sources vs 1925xxxx targets), so deployment is honest LOSO.

Preregistered fork rule: equal-date-mean >= 0.25 -> H1 opens a TKD
generative-read-in line; < 0.25 -> H1 closes and the 688 decision goes to
the user.
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
for entry in (
    str(REPO_ROOT),
    str(REPO_ROOT / "SPINT-main"),
    str(REPO_ROOT / "tfpd_exploration" / "src"),
    str(REPO_ROOT / "tfpd_exploration" / "h1_series_20260830" / "src"),
):
    if entry not in sys.path:
        sys.path.insert(0, entry)

SCHEMA = "fable_tkd_h1_preflight_v1"
DATE_ORDER = ("19250108", "19250113", "19250115", "19250119", "19250120")
FORK_THRESHOLD = 0.25
AUTHORITY_RELATIVE = (
    "SPINT-main/pilot_artifacts/h1_carrierid_hu/source_authority_v1"
)
PLAN_NPZ = "fold0_frozen_eb_plan.npz"
PLAN_MANIFEST = "fold0_frozen_eb_plan.manifest.json"
AUTHORITY_JSON = "H1_CARRIERID_HU_SOURCE_AUTHORITY_v1.json"


class H1PreFlightError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise H1PreFlightError(message)


def receipt_body(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_receipt(path: Path, payload: object, *, exclusive: bool = False) -> str:
    import os
    import tempfile

    path = Path(path)
    if exclusive and path.exists():
        raise H1PreFlightError(f"refusing to overwrite existing receipt: {path}")
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_parent(repo_root: Path) -> dict:
    import numpy as np
    from h1_causal_activity_completion_v1.stage1 import (
        ARTIFACT_RELATIVE as C1_ARTIFACT,
        C1_AUTHORITIES,
        _load_plan,
    )
    from h1_cross_record_postpool_v1.evaluate import _load_minival
    from m1_h1_activity_headroom_v1.core import variance_weighted_r2
    from src.data.h1_carrierid_date_lodo_target import load_outer_date_target_records
    from src.data.h1_m4_eb_pilot import (
        BLOCK_BINS,
        BLOCK_SECONDS,
        _project,
        fit_deployment_carrier,
    )

    data_root = repo_root / "SPINT-main/data/000954"
    per_date = {}
    session_rows = []
    plan_receipts = {}
    for date in DATE_ORDER:
        # The established per-date source plan (honest LODO: the authority
        # asserts target_recordings_opened == 0 for the outer date).
        plan, s_src, plan_receipt = _load_plan(
            repo_root / C1_ARTIFACT / date, C1_AUTHORITIES[date], date
        )
        plan_receipts[date] = {
            "q": plan.q, "lambda": plan.ridge_lambda,
            "plan_sha256": plan_receipt.get("plan_sha256",
                                            plan_receipt.get("plan", {}).get("plan_sha256"))
            if isinstance(plan_receipt, dict) else str(plan_receipt),
        }
        records = load_outer_date_target_records(data_root, outer_date=date)
        values = {}
        for name, record in records.items():
            fit = fit_deployment_carrier(record, plan, record.trial_values[:4])
            beta = fit["beta"]
            minival = _load_minival(data_root, name)
            neural = np.asarray(minival["neural"], dtype=np.float64)
            targets = np.asarray(minival["target"], dtype=np.float32)
            endpoints = np.asarray(minival["endpoints"], dtype=np.int64)
            # Causal 5-bin (100 ms) block rate ending at each endpoint --
            # the parent decoder's native input law; bins before the
            # recording start are zero-padded (the /0.1 denominator is
            # kept, mirroring the M2 psi conv left-pad convention).  Only
            # 2 of 1811 endpoints per session are affected on average.
            rates = np.stack(
                [neural[max(0, e - BLOCK_BINS + 1) : e + 1].sum(axis=0) / BLOCK_SECONDS
                 for e in endpoints]
            )
            z = _project(rates, plan)
            design = np.column_stack((np.ones(z.shape[0]), z))
            prediction = (design @ beta).astype(np.float32)
            target = targets[endpoints]
            r2 = variance_weighted_r2(target, prediction)
            values[name] = float(r2)
            session_rows.append({
                "session": name,
                "date": date,
                "support_trials": fit.get("support_trial_numbers",
                                          list(record.trial_values[:4])),
                "n_endpoints": int(endpoints.size),
                "r2": float(r2),
            })
        per_date[date] = sum(values.values()) / len(values)
    equal_date_mean = sum(per_date.values()) / len(per_date)
    return {
        "plan_law": (
            "per-outer-date frozen source plan from the established C1 "
            "date-LODO authority cache (target_recordings_opened == 0); "
            "parent decoder = deployment ridge beta on the target's own M4 "
            "support (q=16 PC projection, frozen lambda, zero gradient)"
        ),
        "note_hu_authority": (
            "the h1_carrierid_hu source_authority_v1 plan pools the five "
            "LODO dates as sources and was therefore NOT used (would break "
            "date-LOSO for this face); the per-date C1 plans are the honest "
            "law of the established machinery"
        ),
        "plan_receipts": plan_receipts,
        "session_rows": session_rows,
        "per_date_equal_session_mean": {d: float(v) for d, v in per_date.items()},
        "equal_date_mean": float(equal_date_mean),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "dry": True, "schema": SCHEMA,
            "steps": ["attempt.json (O_EXCL)", "rebuild parent ridge from frozen plan",
                      "5-date LODO eval (minival endpoints, causal 100-ms blocks)",
                      "fork rule: equal-date-mean >= 0.25"],
            "date_order": list(DATE_ORDER),
        }, indent=2))
        return 0

    import time

    started = time.monotonic()
    root = REPO_ROOT / "tfpd_exploration/results/fable_tkd_h1_preflight_v1"
    try:
        attempt = {
            "schema": f"{SCHEMA}:attempt",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "law": (
                "parent decoder = the frozen supervised ridge (plan q/lambda "
                "from the immutable H-C source authority cache) mapping "
                "100-ms block rates -> 7-DoF velocity, BEFORE the 4-col "
                "carrier compression; per-target-date closed-form refit on "
                "the target's own M4 support (zero gradient)"
            ),
            "fork_rule": {"threshold": FORK_THRESHOLD,
                          "opens": "TKD generative-read-in line (TKD-M1b pattern)",
                          "closes": "H1 closed; 688 decision to the user"},
        }
        attempt_path = root / "attempt.json"
        if not attempt_path.exists():
            atomic_receipt(attempt_path, attempt, exclusive=True)
        result = evaluate_parent(REPO_ROOT)
        fork_pass = bool(result["equal_date_mean"] >= FORK_THRESHOLD)
        terminal = {
            "schema": f"{SCHEMA}:terminal",
            "status": "TERMINAL",
            **result,
            "context_refs": {"C1_LODO_mean": 0.406, "LP_R3": 0.446,
                             "noise_tolerance": 0.04},
            "fork_rule": {
                "threshold": FORK_THRESHOLD,
                "equal_date_mean": result["equal_date_mean"],
                "pass": fork_pass,
                "outcome": "H1_OPENS_TKD_GENERATIVE_READIN_LINE" if fork_pass
                           else "H1_CLOSED_DECODER_LINE_EXHAUSTED_688_TO_USER",
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_receipt(root / "terminal.json", terminal, exclusive=True)
        print(json.dumps({
            "equal_date_mean": result["equal_date_mean"],
            "per_date": result["per_date_equal_session_mean"],
            "fork_pass": fork_pass,
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
