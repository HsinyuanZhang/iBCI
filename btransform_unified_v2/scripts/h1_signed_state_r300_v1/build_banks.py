#!/usr/bin/env python3
"""Fit official-13 signed-state and rematerialize the public 27-tag C2 banks.

Source plan uses the same 13 held-in sessions as submission 582073.  Each of
the 27 public tags keeps the official first-three calibration trials and the
frozen C2 M3 activity; only T is replaced, then E0 is rematerialized.  This
does not open hidden test, does not train, and does not use last-date 0.564.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import (
    DATA,
    OFFICIAL_HO_M3,
    SOURCE_PAYLOAD,
    WS,
    atomic_json,
    index_heldout_calib,
    jsonable,
    load_public_heldout_m3,
    setup_imports,
    sha_array,
    sha_file,
)

setup_imports()

from btransform_unified_v1 import adapters, h1_config  # noqa: E402
from btransform_unified_v1.c2_protocol import HELDOUT_SESSION_TO_FALCON_KEY  # noqa: E402
import h1_profiles  # noqa: E402


CANDIDATE = "signed_state14"


def load_source_payload() -> dict[str, Any]:
    if not SOURCE_PAYLOAD.is_file():
        raise FileNotFoundError(SOURCE_PAYLOAD)
    return torch.load(SOURCE_PAYLOAD, map_location="cpu", weights_only=False)


def load_plan(receipt: Path) -> Any:
    body = json.loads(receipt.read_text())
    arrays = Path(body["arrays_path"])
    if sha_file(arrays) != body["arrays_sha256"]:
        raise RuntimeError("signed-state plan NPZ hash drift")
    with np.load(arrays) as handle:
        return h1_profiles.ProfilePlan(
            body["candidate"],
            tuple(body["source_sessions"]),
            tuple(body["source_input_sha256"]),
            np.asarray(handle["behavior_rms"], np.float64),
            np.asarray(handle["raw_basis"], np.float64),
            np.asarray(handle["profile_mean"], np.float64),
            np.asarray(handle["profile_scale"], np.float64),
            body["source_raw_sha256"],
            body["source_profile_sha256"],
        )


def materialize(activity: np.ndarray, carrier: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from h1_c2_cal1_b2_l200_p16 import _materialize_e0

    e0, hc = _materialize_e0(activity, carrier)
    if not np.array_equal(hc, carrier):
        raise RuntimeError("C2 direct carrier is not the signed-state T that was supplied")
    return e0, hc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    dest = args.dest.resolve()
    if dest.exists():
        raise FileExistsError(f"refusing nonempty bank dest {dest}")

    sources = tuple(h1_config.H1_ALL_SESSIONS)
    if len(sources) != 13:
        raise RuntimeError("official source roster is not 13 sessions")
    records = h1_profiles._load_records(DATA, sources)
    plan, diagnostics = h1_profiles.fit_source_plan(records, sources, CANDIDATE)
    dest.mkdir(parents=True)
    plan_receipt, plan_arrays = h1_profiles.save_plan(dest / "signed_state14_plan.json", plan, diagnostics)
    if tuple(plan.source_sessions) != sources or plan.candidate != CANDIDATE:
        raise RuntimeError("fitted plan roster/candidate drift")

    payload = load_source_payload()
    rows = payload.get("sessions", {})
    if len(rows) != 27:
        raise RuntimeError(f"official source payload is not 27 tags: {len(rows)}")
    heldin = h1_profiles._legacy().index_heldin_calib(DATA)
    heldout = index_heldout_calib()
    falcon_by_session = {session: key for session, key in HELDOUT_SESSION_TO_FALCON_KEY}
    pilot = h1_profiles._legacy()

    tag_arrays: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {}
    for tag in sorted(rows):
        row = rows[tag]
        session = str(row["session"])
        trials = tuple(float(value) for value in row["calibration_trials"])
        if len(trials) != 3 or len(set(trials)) != 3:
            raise RuntimeError(f"{tag}: official payload is not exact public M3")
        if session in heldin:
            record = records[session] if session in records else h1_profiles._load_records(DATA, (session,))[session]
            surface = "held-in-calib"
            path = heldin[session]
        else:
            if session not in heldout:
                raise RuntimeError(f"{tag}: no public calibration path for {session}")
            record = load_public_heldout_m3(heldout[session], pilot)
            surface = "held-out-calib"
            path = heldout[session]
        if tuple(float(value) for value in record.trial_values[:3]) != trials:
            raise RuntimeError(f"{tag}: first public three trials differ from official payload")
        carrier, raw, info = h1_profiles.deploy_profile(record, plan, trials)
        if carrier.shape != (176, 4) or not np.isfinite(carrier).all() or float(np.max(np.abs(carrier))) == 0.0:
            raise RuntimeError(f"{tag}: signed-state T is degenerate")
        old_t = np.ascontiguousarray(row["carrier"], dtype=np.float32)
        if np.array_equal(np.ascontiguousarray(carrier, np.float32), old_t):
            raise RuntimeError(f"{tag}: signed-state T byte-equals frozen H-C T")
        activity, payload_carrier = adapters._h1_payload_arrays(session)
        if not np.array_equal(payload_carrier, old_t):
            raise RuntimeError(f"{tag}: C2 M3 payload T differs from EP-FiLM source T")
        e0, hc = materialize(activity, np.ascontiguousarray(carrier, np.float32))
        tag_arrays[f"T/{tag}"] = np.ascontiguousarray(carrier, np.float32)
        tag_arrays[f"E0/{tag}"] = e0
        tag_arrays[f"raw/{tag}"] = np.ascontiguousarray(raw, np.float32)
        details[tag] = {
            "session": session,
            "surface": surface,
            "falcon_key": falcon_by_session.get(session),
            "trials": list(trials),
            "path": str(path),
            "path_sha256": sha_file(path),
            "T_sha256": sha_array(tag_arrays[f"T/{tag}"]),
            "E0_sha256": sha_array(e0),
            "old_hc_T_sha256": sha_array(old_t),
            "activity_sha256": sha_array(activity),
            "direct_carrier_byte_equal": bool(np.array_equal(hc, tag_arrays[f"T/{tag}"])),
            **jsonable(info),
        }
        print(f"[signed-state 27] {tag} {session} {surface} T={details[tag]['T_sha256'][:12]}", flush=True)

    if len(details) != 27:
        raise RuntimeError("27-tag bank is incomplete")
    ho_keys = {key for _, key in HELDOUT_SESSION_TO_FALCON_KEY}
    ho_tags = {tag for tag, row in details.items() if row["surface"] == "held-out-calib"}
    if ho_keys - set(details) and ho_keys != ho_tags:
        # Official payload tags are the Falcon keys for held-out rows.
        if not ho_keys.issubset(set(details)):
            raise RuntimeError(f"HO Falcon keys missing from 27-tag bank: {sorted(ho_keys - set(details))}")

    np.savez_compressed(dest / "banks_27.npz", **tag_arrays)
    receipt = {
        "schema": "h1_signed_state_r300_27tag_v1",
        "status": "BUILT",
        "utc": datetime.now(timezone.utc).isoformat(),
        "candidate": CANDIDATE,
        "source_sessions": list(sources),
        "source_count": 13,
        "tag_count": 27,
        "held_in_count": sum(row["surface"] == "held-in-calib" for row in details.values()),
        "held_out_count": sum(row["surface"] == "held-out-calib" for row in details.values()),
        "support": "official first-three public calibration trials; C2 M3 activity unchanged",
        "estimator": "signed_state14 source-fitted on official 13; frozen C2 E0 rematerialization",
        "last_date_0564_used": False,
        "official_test_used": False,
        "selection_surface": "later HO-M3 only; do not pack from last-date transfer",
        "baseline_582073_ho_m3": OFFICIAL_HO_M3,
        "plan_receipt": str(plan_receipt),
        "plan_receipt_sha256": sha_file(plan_receipt),
        "plan_arrays": str(plan_arrays),
        "plan_arrays_sha256": sha_file(plan_arrays),
        "banks_27": str((dest / "banks_27.npz").resolve()),
        "banks_27_sha256": sha_file(dest / "banks_27.npz"),
        "source_payload": str(SOURCE_PAYLOAD),
        "source_payload_sha256": sha_file(SOURCE_PAYLOAD),
        "implementation_sha256": {
            str(path): sha_file(path)
            for path in (Path(__file__).resolve(), HERE_COMMON, Path(h1_profiles.__file__).resolve())
        },
        "tags": details,
    }
    atomic_json(dest / "receipt.json", receipt)
    print(json.dumps({"receipt": str(dest / "receipt.json"), "tags": 27, "candidate": CANDIDATE}, sort_keys=True))
    return 0


HERE_COMMON = Path(__file__).resolve().parent / "common.py"


if __name__ == "__main__":
    raise SystemExit(main())
