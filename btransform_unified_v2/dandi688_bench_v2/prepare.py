"""Materialize the authorized 2015 source/development paired cache; never final data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from . import data, protocol


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    os.replace(temp, path)


def prepare_cache(dest: Path, *, raw_root: Path = protocol.DEFAULT_RAW_ROOT,
                  include_dev: bool = True) -> dict[str, Any]:
    """Build paired raw-derived cache with exact split provenance and no final access."""
    dest = Path(dest).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"destination must be empty: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    roster = [("train", session) for session in protocol.TRAIN_SESSIONS]
    if include_dev:
        roster.extend(("dev", session) for session in protocol.DEV_SESSIONS)
    access_log: list[dict[str, Any]] = []
    sessions: dict[str, Any] = {}
    canonical_hash: str | None = None
    for split, session_id in roster:
        print(f"prepare {len(sessions) + 1}/{len(roster)} {split} {session_id}", flush=True)
        pair = data.load_pair(session_id, raw_root=raw_root,
                              purpose="source" if split == "train" else "development",
                              access_log=access_log)
        # Explicitly repeat the formal M33 OLS admissibility check at prepare time.
        carrier_audit = data.validate_carrier_angles(pair["sua"].carrier_angles)
        receipt: dict[str, Any] = {"split": split, "representations": {}}
        for representation, record in pair.items():
            filename = f"{session_id}.{representation}.npz"
            path = dest / filename
            data.save_session(record, path)
            # Verify each written object before treating the cache as usable.
            restored = data.load_cached_session(path)
            if restored.representation != representation:
                raise RuntimeError("cache representation round-trip drift")
            receipt["representations"][representation] = {
                "file": filename, "sha256": _sha256(path), "channels": int(record.neural.shape[1]),
                "array_sha256": record.metadata["array_sha256"],
            }
        receipt["carrier_angle_audit"] = carrier_audit
        current_hash = hashlib.sha256(json.dumps(pair["sua"].metadata["canonical_electrode_keys"], sort_keys=True).encode()).hexdigest()
        if canonical_hash is None:
            canonical_hash = current_hash
        elif canonical_hash != current_hash:
            raise RuntimeError(f"{session_id}: canonical M1 electrode table drift")
        sessions[session_id] = receipt
        print(f"prepared {session_id}: sua={pair['sua'].neural.shape[1]} pmua={pair['pmua'].neural.shape[1]}", flush=True)
    if any(entry["split"] == "final" for entry in access_log):
        raise RuntimeError("final data access detected during prepare")
    receipt = {
        "schema": "dandi688_bench_v2_prepared_v1",
        "protocol": protocol.protocol_dict(),
        "raw_root": str(Path(raw_root).resolve()),
        "include_dev": include_dev,
        "session_count": len(sessions),
        "sessions": sessions,
        "canonical_m1_table_sha256": canonical_hash,
        "access_log": access_log,
        "final_sessions_opened": 0,
    }
    _write_json(dest / "prepared_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=protocol.DEFAULT_RAW_ROOT)
    parser.add_argument("--source-only", action="store_true", help="write only the 18 source sessions")
    args = parser.parse_args()
    receipt = prepare_cache(args.dest, raw_root=args.raw_root, include_dev=not args.source_only)
    print(json.dumps({"session_count": receipt["session_count"], "final_sessions_opened": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
