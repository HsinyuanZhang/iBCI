"""P2 prerequisite: RT go-cue coverage audit (fail-closed).

RT trials must be segmented by ``go_cue_time_array`` into reaches — never whole-
trial averages. Sessions whose usable-trial coverage falls below the predeclared
floor are marked ineligible for k4 RT cells (no silent drop).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from pynwb import NWBHDF5IO

SCHEMA = "carrier_perf_p2_rt_gocue_coverage_v1"
# Predeclared before looking at the full 15-session table (handoff cites 154/187 ≈ 0.82
# on RT-20131009 as the known example). Floor is intentionally below that example so
# the audit classifies rather than secretly tightening after the fact.
MIN_USABLE_FRACTION = 0.50
MIN_FINITE_GOCUES = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_rt_nwbs(data_root: Path) -> list[Path]:
    base = (data_root / "sub-C").resolve()
    paths = sorted(base.glob("sub-C_ses-RT-*_behavior+ecephys.nwb"))
    if not paths:
        raise FileNotFoundError(f"no RT NWBs under {base}")
    return paths


def audit_rt_session(nwb_path: Path) -> dict[str, Any]:
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        trials = nwb.trials.to_dataframe()
    if "go_cue_time_array" not in trials.columns:
        raise ValueError(f"{nwb_path.name}: missing go_cue_time_array")
    n_trials = int(len(trials))
    finite_counts = []
    usable = 0
    for raw in trials["go_cue_time_array"]:
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        n_fin = int(np.isfinite(arr).sum())
        finite_counts.append(n_fin)
        if n_fin >= MIN_FINITE_GOCUES:
            usable += 1
    # target_dir uniqueness (handoff: often a single π/4 value → T4 undefined)
    target_dirs = None
    n_unique_target_dir = None
    if "target_dir" in trials.columns:
        td = np.asarray(trials["target_dir"], dtype=np.float64)
        finite_td = td[np.isfinite(td)]
        n_unique_target_dir = int(np.unique(np.round(finite_td, decimals=9)).size) if finite_td.size else 0
        target_dirs = {
            "n_finite": int(finite_td.size),
            "n_unique_rounded9": n_unique_target_dir,
        }
    frac = usable / n_trials if n_trials else 0.0
    eligible = bool(frac >= MIN_USABLE_FRACTION)
    return {
        "session": nwb_path.name.replace("_behavior+ecephys.nwb", ""),
        "nwb": str(nwb_path),
        "nwb_sha256": sha256_file(nwb_path),
        "n_trials": n_trials,
        "n_usable_ge2_finite_gocue": usable,
        "usable_fraction": frac,
        "finite_gocue_count_mean": float(np.mean(finite_counts)) if finite_counts else float("nan"),
        "finite_gocue_count_min": int(min(finite_counts)) if finite_counts else 0,
        "finite_gocue_count_max": int(max(finite_counts)) if finite_counts else 0,
        "target_dir": target_dirs,
        "t4_undefined_single_target_dir": bool(n_unique_target_dir == 1) if n_unique_target_dir is not None else None,
        "eligible_for_k4_rt": eligible,
        "ineligibility_reason": None if eligible else f"usable_fraction {frac:.4f} < floor {MIN_USABLE_FRACTION}",
    }


def build_rt_gocue_audit(*, data_root: Path) -> dict[str, Any]:
    paths = discover_rt_nwbs(data_root)
    rows = [audit_rt_session(p) for p in paths]
    eligible = [r for r in rows if r["eligible_for_k4_rt"]]
    ineligible = [r for r in rows if not r["eligible_for_k4_rt"]]
    # Fail-closed discipline: ineligible sessions must not be silently omitted later.
    return {
        "schema": SCHEMA,
        "status": "completed_cpu_only",
        "no_gpu": True,
        "min_usable_fraction_floor": MIN_USABLE_FRACTION,
        "min_finite_gocues": MIN_FINITE_GOCUES,
        "n_sessions": len(rows),
        "n_eligible": len(eligible),
        "n_ineligible": len(ineligible),
        "eligible_sessions": [r["session"] for r in eligible],
        "ineligible_sessions": [r["session"] for r in ineligible],
        "all_sessions_have_single_target_dir": all(
            r.get("t4_undefined_single_target_dir") for r in rows
        ),
        "fail_closed_policy": (
            "RT k4 cells may only load eligible_sessions; ineligible_sessions must appear "
            "explicitly in any manifest exclusion list — silent drop forbidden."
        ),
        "rows": rows,
    }


def write_audit(audit: Mapping[str, Any], output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    path = output_dir / "audit.json"
    tmp = output_dir / "audit.json.tmp"

    def _strict(obj: Any) -> Any:
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, dict):
            return {str(k): _strict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_strict(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return _strict(float(obj))
        return obj

    tmp.write_text(json.dumps(_strict(dict(audit)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path
