#!/usr/bin/env python3
"""Build the CPU-only, score-free audit for the M2 post-33 confirmation.

The audit reads only native-M2 NWB metadata/calibration labels and frozen
structural manifests.  It imports no model, scorer, result aggregator, formal
SUA path, or EvalAI client, and it refuses to overwrite an existing output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUA = ROOT / "sua_exploration"
ENDPOINT_AUDIT = SUA / "results/m2_heldin_postsupport_endpoint_v1/audit.json"
ENDPOINT_AUDIT_SHA256 = "d6940d156d05cd1cbd43bac95220c1d1db370c50d841de8a51d49a53c9ac70c5"
OLD_PROTOCOL = SUA / "results/m2_heldin_postsupport_endpoint_v1/protocol_receipt.json"
OLD_PROTOCOL_SHA256 = "7673d360099775e37b199621956d099a9a2a7bafcbe7b074c3146978300f1c49"
CLEAN_INPUT_MANIFEST = SUA / "results/m2_ssc_t4_v1/clean_teacher_input_manifest.json"
CLEAN_INPUT_MANIFEST_SHA256 = "5969aebef8e5620eb6e68bbe1965ad6656cfbbb695821668d9d1d9c96c16891c"
C1_RECEIPT = SUA / "results/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist/receipt.json"
C1_RECEIPT_SHA256 = "8b17c19515fa0e6cc122233fd20cf7e87a616287fa247e5eacb136d6a56c9d85"
DEFAULT_OUT = SUA / "results/m2_native_t4_spint_post33_confirm_v1_scorefree_audit_20260804"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"SHA-256 drift for {path}: expected {expected}, found {observed}")


def load_protocol_helpers():
    sys.path.insert(0, str(ROOT))
    from sua_exploration.mc_maze import m2_native_post33_confirm_v1 as contract

    return contract


def load_t4_helpers():
    sys.path.insert(0, str(ROOT / "streaming_calibration_exp"))
    from src.data.falcon_t4_features import calibration_target_angles

    return calibration_target_angles


def c1_source_map_closure() -> dict[str, Any]:
    require_hash(C1_RECEIPT, C1_RECEIPT_SHA256)
    receipt = json.loads(C1_RECEIPT.read_text(encoding="utf-8"))
    rows = []
    for relative, expected in sorted(receipt["source_map"].items()):
        path = ROOT / relative
        observed = sha256(path)
        if observed != expected["sha256"]:
            raise ValueError(f"active C1 v3r2 source_map drift at {relative}")
        rows.append({"path": relative, "sha256": observed})
    return {
        "receipt": str(C1_RECEIPT.relative_to(ROOT)),
        "receipt_sha256": C1_RECEIPT_SHA256,
        "source_count": len(rows),
        "all_hashes_match": True,
        "files": rows,
        "files_edited_by_this_program": 0,
    }


def data_closure(*, full_hash: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require_hash(CLEAN_INPUT_MANIFEST, CLEAN_INPUT_MANIFEST_SHA256)
    manifest = json.loads(CLEAN_INPUT_MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for frozen in manifest["files"]:
        path = Path(frozen["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if "held-out" in path.name.lower():
            raise ValueError(f"held-out input found in source manifest: {path}")
        if path.stat().st_size != frozen["size_bytes"]:
            raise ValueError(f"size drift for {path}")
        observed = sha256(path) if full_hash else None
        if observed is not None and observed != frozen["sha256"]:
            raise ValueError(f"data SHA drift for {path}")
        rows.append(
            {
                "role": frozen["role"],
                "session": frozen["session"],
                "path": str(path),
                "size_bytes": frozen["size_bytes"],
                "sha256": frozen["sha256"],
                "sha256_reverified_now": bool(full_hash),
            }
        )
    if len(rows) != 14:
        raise ValueError(f"expected fourteen source inputs, got {len(rows)}")
    roles = {role: sum(row["role"] == role for row in rows) for role in {r["role"] for r in rows}}
    return rows, {
        "path": str(CLEAN_INPUT_MANIFEST.relative_to(ROOT)),
        "sha256": CLEAN_INPUT_MANIFEST_SHA256,
        "role_counts": roles,
        "all_bytes_verified": True,
        "all_sha256_reverified_now": bool(full_hash),
    }


def per_session_endpoint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require_hash(ENDPOINT_AUDIT, ENDPOINT_AUDIT_SHA256)
    endpoint = json.loads(ENDPOINT_AUDIT.read_text(encoding="utf-8"))
    old_windows = endpoint["heldin_postsupport_window_audit"]["per_session"]
    calibration_target_angles = load_t4_helpers()
    output: dict[str, Any] = {}
    calib_rows = [row for row in rows if row["role"] == "heldin_calib"]
    for row in calib_rows:
        path = Path(row["path"])
        session = row["session"]
        short_session = session.removeprefix("ses-")
        angles = np.asarray(calibration_target_angles(path, "m2"), dtype=np.float64)[:33]
        usable = np.isfinite(angles)
        design = np.stack(
            [np.ones(int(usable.sum())), np.cos(angles[usable]), np.sin(angles[usable])], axis=1
        )
        rank = int(np.linalg.matrix_rank(design))
        condition = float(np.linalg.cond(design))
        if len(angles) != 33 or int(usable.sum()) != 16 or rank != 3 or not np.isfinite(condition):
            raise ValueError(
                f"invalid first-33 M2 cosine design for {session}: "
                f"n={len(angles)} directional={usable.sum()} rank={rank} condition={condition}"
            )
        historical = endpoint["heldin_postsupport_endpoint"]["sessions"][short_session]
        window = old_windows[session]["query_window_audit"]
        if window["window_size"] != 50 or window["query_start_trial"] != 33:
            raise ValueError(f"frozen endpoint audit boundary drift for {session}")
        output[session] = {
            "calibration_nwb": {
                "path": str(path),
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            },
            "total_trials": historical["total_calib_trials"],
            "first33_directional_trials": int(usable.sum()),
            "first33_centre_or_rest_trials": int(33 - usable.sum()),
            "cosine_design_rank": rank,
            "cosine_design_condition_2norm": condition,
            "post33_query_trials": int(window["query_trials"]),
            "eligible_windows": int(window["eligible_windows"]),
            "minimum_window_start_padded_bin": int(window["minimum_window_start_padded_bin"]),
            "raw_query_start_bin": int(window["raw_query_start_bin"]),
            "full_50_bin_history_after_boundary": bool(window["full_window_disjoint"]),
        }
    total_windows = sum(row["eligible_windows"] for row in output.values())
    if len(output) != 7 or total_windows != 101_171:
        raise ValueError(f"post33 endpoint count mismatch: sessions={len(output)}, windows={total_windows}")
    return {
        "sessions": output,
        "totals": {
            "sessions": len(output),
            "eligible_windows": total_windows,
            "post33_query_trials": sum(row["post33_query_trials"] for row in output.values()),
            "first33_directional_trials": sum(
                row["first33_directional_trials"] for row in output.values()
            ),
        },
        "frozen_structural_audit": {
            "path": str(ENDPOINT_AUDIT.relative_to(ROOT)),
            "sha256": ENDPOINT_AUDIT_SHA256,
        },
    }


def build(*, full_hash: bool) -> dict[str, Any]:
    require_hash(OLD_PROTOCOL, OLD_PROTOCOL_SHA256)
    contract = load_protocol_helpers()
    rows, manifest = data_closure(full_hash=full_hash)
    folds = {str(fold): contract.fold_roles(fold) for fold in contract.FOLDS}
    outer_sessions = [row["outer_left_out_session"] for row in folds.values()]
    if len(outer_sessions) != 7 or len(set(outer_sessions)) != 7:
        raise ValueError("seven folds do not define seven unique outer sessions")
    return {
        "schema_version": 1,
        "protocol_id": contract.PROTOCOL_ID,
        "status": "PASS_SCORE_FREE_CPU_AUDIT",
        "execution_scope": {
            "gpu_used": False,
            "training_started": False,
            "new_endpoint_r2_values_read": 0,
            "scorer_modules_imported": 0,
            "formal_sua_paths_resolved": 0,
            "formal_sua_files_opened": 0,
            "evalai_calls": 0,
        },
        "supersession": {
            "old_protocol_path": str(OLD_PROTOCOL.relative_to(ROOT)),
            "old_protocol_sha256": OLD_PROTOCOL_SHA256,
            "old_file_modified": False,
            "superseded_scope_only": (
                "ordinary-T4-versus-clean-SPINT effectiveness use and target-session "
                "checkpoint selection; old v1 receipt remains immutable historical evidence"
            ),
        },
        "active_c1_v3r2_closure": c1_source_map_closure(),
        "input_manifest": manifest,
        "input_files": rows,
        "endpoint": per_session_endpoint(rows),
        "folds": folds,
        "matrix": contract.matrix_contract(),
        "forbidden_access": [
            "formal SUA sessions",
            "FALCON held-out-calib/test sessions",
            "new post33 endpoint behavior scores",
            "EvalAI APIs or submission payloads",
        ],
    }


def run(out_dir: Path, *, full_hash: bool) -> Path:
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite score-free audit directory: {out_dir}")
    payload = build(full_hash=full_hash)
    out_dir.mkdir(parents=True)
    path = out_dir / "audit.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "audit.sha256").write_text(f"{sha256(path)}  audit.json\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--reuse-frozen-data-hashes",
        action="store_true",
        help="verify sizes and reuse the already frozen SHA values instead of re-hashing 15 GB",
    )
    args = parser.parse_args()
    path = run(args.out, full_hash=not args.reuse_frozen_data_hashes)
    print(path)
    print(sha256(path))


if __name__ == "__main__":
    main()

