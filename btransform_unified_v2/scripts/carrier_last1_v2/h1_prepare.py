"""Prepare the fixed chronological H1 leave-last-one-dates split.

This is an independent program.  It never writes into the historical LODO
tree and it has no single-date ``--fold`` interface.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CROSS = ROOT / "scripts" / "cross_session_v1"
for item in (str(ROOT.parent / "btransform_unified_v1" / "src"),):
    if item not in sys.path:
        sys.path.insert(0, item)
from btransform_unified_v1.h1_config import H1_SESSIONS_BY_DATE

STAGE = __import__("os").environ.get("CARRIER_STAGE", "outer")
if STAGE not in ("inner", "outer"): raise ValueError("CARRIER_STAGE must be inner or outer")
SPLIT_ID = "h1_inner_last1_19250119" if STAGE == "inner" else "h1_outer_last1_19250120"
PREPARE_SCHEMA = "h1_carrier_last1_prepare_v1"
SELECTION_SEAL_SCHEMA = "h1_carrier_last1_selection_seal_v1"
TRAIN_RECEIPT_SCHEMA = "h1_carrier_last1_train_receipt_v1"
SOURCE_DATES = ("1925-01-01", "1925-01-08", "1925-01-13", "1925-01-15") if STAGE == "inner" else ("1925-01-01", "1925-01-08", "1925-01-13", "1925-01-15", "1925-01-19")
TARGET_DATES = ("1925-01-19",) if STAGE == "inner" else ("1925-01-20",)
SOURCE_SESSIONS = tuple(s for d in SOURCE_DATES for s in H1_SESSIONS_BY_DATE[d])
TARGET_SESSIONS = tuple(s for d in TARGET_DATES for s in H1_SESSIONS_BY_DATE[d])


def _sha_array(x: np.ndarray) -> str:
    a=np.ascontiguousarray(x); return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_legacy_prepare():
    """Load the old data helper only in the standalone prepare process.

    Train and audit import this module for split/seal metadata only.  They use
    the streaming-calibration ``src`` package, whereas the frozen preparer
    intentionally requires the old SPINT-main ``src.data`` package.
    """
    spint = ROOT.parent / "SPINT-main"
    for item in (str(ROOT / "src"), str(spint), str(ROOT.parent / "btransform_unified_v1" / "src")):
        if item in sys.path:
            sys.path.remove(item)
        sys.path.insert(0, item)
    spec = importlib.util.spec_from_file_location("carrier_last1_legacy_h1_prepare", CROSS / "h1_prepare.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen cross-session H1 preparation helpers")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    return legacy


def split_contract() -> dict[str, object]:
    source, target = set(SOURCE_SESSIONS), set(TARGET_SESSIONS)
    if source & target or len(source) != (9 if STAGE == "inner" else 11) or len(target) != 2:
        raise RuntimeError("chronological split roster is invalid")
    if not all(date < TARGET_DATES[0] for date in SOURCE_DATES):
        raise RuntimeError("source dates must precede the target period")
    return {"schema": "h1_carrier_last1_split_v1", "split_id": SPLIT_ID,
            "source_dates": list(SOURCE_DATES), "target_dates": list(TARGET_DATES),
            "source_sessions": list(SOURCE_SESSIONS), "target_sessions": list(TARGET_SESSIONS),
            "time_order_disjoint": True,
            "source_count": len(SOURCE_SESSIONS), "target_count": len(TARGET_SESSIONS)}


def matches_split_contract(value: dict) -> bool:
    return all(value.get(key) == expected for key, expected in split_contract().items() if key != "schema")


def _write_manifest(dest: Path, surface: str, authority: dict, records: dict, legacy) -> None:
    contract = split_contract()
    manifest = {**contract, "schema": PREPARE_SCHEMA,
                "surface": surface, "source_authority": authority,
                "records": records,
                "source_authority_files": {
                    "json_sha256": _sha_file(dest.parent / "source" / "source_hc_plan.json"),
                    "arrays_sha256": _sha_file(dest.parent / "source" / "source_hc_plan_arrays.npz"),
                },
                "implementation_sha256": {
                    str(Path(__file__).resolve()): _sha_file(Path(__file__).resolve()),
                    str(Path(legacy.__file__).resolve()): _sha_file(Path(legacy.__file__).resolve()),
                }}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def validate_selection_seals(gate: Path) -> dict[str, object]:
    """Verify the explicitly supplied paired source authority before target IO."""
    if not gate.is_file(): raise RuntimeError("explicit paired source gate missing")
    row=json.loads(gate.read_text())
    if (row.get("schema") != "carrier_last1_v2_paired_audit" or row.get("status") != "PASSED"
            or row.get("stage") != STAGE or row.get("dataset") != "h1"):
        raise RuntimeError("paired source gate identity/status mismatch")
    return row

def prepare(dest: Path, surface: str) -> None:
    contract = split_contract()
    directory = dest / surface
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"refusing to overwrite {directory}")
    if surface == "target" and not (dest / "source" / "manifest.json").is_file():
        raise FileNotFoundError("target preparation requires sealed chronological source authority")
    if surface == "target":
        gate = getattr(prepare, "source_gate", None)
        if gate is None: raise RuntimeError("explicit paired source gate required before target preparation")
        validate_selection_seals(gate)
    directory.mkdir(parents=True, exist_ok=False)
    legacy = _load_legacy_prepare()
    paths = legacy.index_heldin_calib(legacy.DATA)
    chosen = SOURCE_SESSIONS if surface == "source" else TARGET_SESSIONS
    if surface == "source":
        # Only the nine chronological source sessions are opened for this fit.
        records = {name: legacy.load_record(paths[name]) for name in SOURCE_SESSIONS}
        plan, rms, authority = legacy.fresh_plan(SPLIT_ID, records)
        authority = {**authority, **contract, "target_records_opened": 0,
                     "authority_scope": "source dates only; no 1925-01-19/20 records"}
        (directory / "source_hc_plan.json").write_text(json.dumps(authority, indent=2, sort_keys=True) + "\n")
        np.savez_compressed(directory / "source_hc_plan_arrays.npz", mean=plan.mean, scale=plan.scale,
                            pcs=plan.pcs, U=plan.U, mu=plan.mu, tau2=np.float64(plan.tau2),
                            rms=np.float64(rms))
    else:
        plan, rms, authority = legacy.load_frozen_authority(dest / "source")
        if tuple(authority.get("source_sessions", ())) != SOURCE_SESSIONS:
            raise RuntimeError("target preparation rejected non-chronological source authority")
        if authority.get("split_id") != SPLIT_ID or authority.get("target_records_opened") != 0:
            raise RuntimeError("target preparation rejected unsealed source authority")
        records = {name: legacy.load_record(paths[name]) for name in TARGET_SESSIONS}

    result: dict[str, dict] = {}
    for name in chosen:
        record = records[name]
        available = tuple(record.trial_values)
        if len(available) < 6:
            raise RuntimeError(f"{name}: need >= 6 available eval-valid native trials")
        query = available[3:-2] if surface == "source" else available[3:]
        result[name] = legacy.write_session(directory / f"{name}.npz", record, plan, rms, query,
                                            stride=4 if surface == "source" else 1)
        if getattr(prepare, "carrier_pack", None):
            with np.load(prepare.carrier_pack, allow_pickle=False) as pack, np.load(directory / f"{name}.npz", allow_pickle=False) as old:
                key=f"carrier/{name}"
                if key not in pack: raise RuntimeError(f"carrier pack missing {key}")
                payload={k:old[k] for k in old.files}; payload["carrier"]=np.ascontiguousarray(pack[key],np.float32)
            tmp=directory/f"{name}.tmp.npz"; np.savez_compressed(tmp,**payload); tmp.replace(directory/f"{name}.npz")
            result[name]["carrier_sha256"]=_sha_array(payload["carrier"])
        if surface == "source":
            result[name]["validation"] = legacy.write_session(directory / f"{name}.val.npz", record,
                                                                 plan, rms, available[-2:], stride=4)
            if getattr(prepare, "carrier_pack", None):
                valpath=directory/f"{name}.val.npz"
                with np.load(prepare.carrier_pack, allow_pickle=False) as pack, np.load(valpath, allow_pickle=False) as oldval:
                    payload={k:oldval[k] for k in oldval.files}; payload["carrier"]=np.ascontiguousarray(pack[f"carrier/{name}"],np.float32)
                tmp=directory/f"{name}.val.tmp.npz"; np.savez_compressed(tmp,**payload); tmp.replace(valpath)
                result[name]["validation"]["carrier_sha256"]=_sha_array(payload["carrier"])
    _write_manifest(directory, surface, authority, result, legacy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--surface", choices=("source", "target"), required=True)
    parser.add_argument("--carrier-pack", type=Path)
    parser.add_argument("--source-gate", type=Path)
    args = parser.parse_args()
    prepare.carrier_pack = None if args.carrier_pack is None else args.carrier_pack.resolve()
    prepare.source_gate = None if args.source_gate is None else args.source_gate.resolve()
    prepare(args.dest.resolve(), args.surface)


if __name__ == "__main__":
    main()
