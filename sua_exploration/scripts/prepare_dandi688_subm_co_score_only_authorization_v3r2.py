#!/usr/bin/env python3
"""Prepare, but never sign, a future V3R2 score authorization envelope."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import subm_co_score_only_v3 as auth_v3  # noqa: E402
from sua_exploration.scripts.write_dandi688_subm_co_score_only_prelaunch_v3r2 import (  # noqa: E402
    DEFAULT_OUTPUT,
    StaticScoreV3R2Error,
    canonical_bytes,
    load_stored_prelaunch,
    sha256_file,
)


def _write_exclusive(path: Path, payload: Mapping[str, Any], mode: int = 0o444) -> str:
    raw = canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)
    if stat.S_IMODE(path.stat().st_mode) != mode:
        raise RuntimeError("authorization-preparation file mode failure")
    return hashlib.sha256(raw).hexdigest()


def review_template(prelaunch_dir: Path) -> dict[str, Any]:
    stored = load_stored_prelaunch(prelaunch_dir, ROOT)
    return {
        "schema": "dandi_000688_subm_co_external_score_only_authorization_review_template_v3r2",
        "status": "REVIEW_PENDING_NATIVE_M2_NOT_YET_AUTHORIZED",
        "not_an_executable_authorization": True,
        "do_not_edit_status_to_authorize": True,
        "execution_policy_sha256": stored["execution_policy_sha256"],
        "prelaunch": {
            key: stored[key]
            for key in ("draft_sha256", "receipt_sha256", "seal_sha256", "status")
        },
        "deployment_budget": {
            "activity_identity_trials": 30,
            "t4_fit_pool_trials": 50,
            "query_start": "strictly_after_rewarded_trial_50",
        },
        "matrix": {
            "sessions": 15,
            "views": ["sua", "pseudo_mua"],
            "arms": ["shared_t4", "shared_ts4"],
            "seeds": [42, 43, 44],
            "cells": 180,
        },
        "claim_boundary": "shared_t4_vs_shared_ts4_only_same_dandiset_cross_animal",
        "future_generation_requirements": [
            "Native-M2 completion receipt supplied to the unsigned-envelope generator",
            "fresh V3R2 byte-level root review",
            "fresh output root below the V3R2 output parent",
            "exact external-NWB root",
            "fresh 256-bit nonce and <=15-minute issue window",
            "detached Ed25519 signature created outside the workspace",
        ],
    }


def unsigned_envelope(
    *, stored: Mapping[str, Any], authorization_id: str, output_root: Path,
    external_nwb_root: Path, issued: datetime, validity_seconds: int,
) -> dict[str, Any]:
    if not (1 <= validity_seconds <= auth_v3.MAX_VALIDITY_SECONDS):
        raise ValueError("validity must be within 1..900 seconds")
    policy = stored["execution_policy"]
    output_parent = Path(str(policy["output_parent"])).resolve()
    output = output_root.resolve()
    try:
        output.relative_to(output_parent)
    except ValueError as exc:
        raise ValueError("output root is outside the fixed V3R2 output parent") from exc
    if output == output_parent or output.exists():
        raise ValueError("output root must be a fresh V3R2 child")
    external = external_nwb_root.resolve(strict=True)
    if not external.is_dir() or external.is_symlink():
        raise ValueError("external-NWB root must be an existing canonical directory")
    expires = issued + timedelta(seconds=validity_seconds)
    body = {
        "schema": auth_v3.AUTHORIZATION_SCHEMA,
        "kind": "dandi_000688_subm_co_external_score_only_authorization_v3",
        "status": auth_v3.AUTHORIZATION_STATUS,
        "authorization_id": authorization_id,
        "single_use_nonce": secrets.token_hex(32),
        "issued_at": issued.astimezone(timezone.utc).isoformat(),
        "expires_at": expires.astimezone(timezone.utc).isoformat(),
        "permitted_action": auth_v3.PERMITTED_ACTION,
        "output_root": str(output),
        "external_nwb_root": str(external),
        "execution_policy_sha256": stored["execution_policy_sha256"],
        "bindings": policy["authorization_bindings"],
        "external_subm_scoring_permitted": True,
        "cpu_forward_r2_only": True,
        "normalizer_fitting_permitted": False,
        "optimizer_or_backward_permitted": False,
        "target_updates_permitted": False,
    }
    return {"schema": auth_v3.AUTHORIZATION_ENVELOPE_SCHEMA, "authorization": body}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("template", "unsigned-envelope"), default="template")
    parser.add_argument("--prelaunch-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--native-m2-completion-receipt", type=Path)
    parser.add_argument("--authorization-id")
    parser.add_argument("--score-output-root", type=Path)
    parser.add_argument("--external-nwb-root", type=Path)
    parser.add_argument("--validity-seconds", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "template":
            payload = review_template(args.prelaunch_dir)
            if args.output is None:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                digest = _write_exclusive(args.output.resolve(), payload)
                print(json.dumps({"template": str(args.output.resolve()), "sha256": digest, "status": payload["status"]}, sort_keys=True))
            return 0
        if any(
            value is None
            for value in (
                args.output,
                args.native_m2_completion_receipt,
                args.authorization_id,
                args.score_output_root,
                args.external_nwb_root,
            )
        ):
            raise ValueError(
                "unsigned-envelope requires --output, --native-m2-completion-receipt, --authorization-id, "
                "--score-output-root, and --external-nwb-root"
            )
        completion = args.native_m2_completion_receipt.resolve(strict=True)
        if not completion.is_file() or completion.is_symlink():
            raise ValueError("Native-M2 completion receipt is missing or unsafe")
        stored = load_stored_prelaunch(args.prelaunch_dir, ROOT)
        issued = datetime.now(timezone.utc)
        envelope = unsigned_envelope(
            stored=stored,
            authorization_id=args.authorization_id,
            output_root=args.score_output_root,
            external_nwb_root=args.external_nwb_root,
            issued=issued,
            validity_seconds=args.validity_seconds,
        )
        output = args.output.resolve()
        envelope_sha = _write_exclusive(output, envelope, mode=0o444)
        release = {
            "schema": "dandi_000688_subm_co_v3r2_unsigned_release_receipt_v1",
            "status": "UNSIGNED_NOT_AUTHORIZED",
            "native_m2_completion_receipt": {
                "path": str(completion),
                "sha256": sha256_file(completion),
                "bytes": completion.stat().st_size,
            },
            "unsigned_envelope": {
                "path": str(output),
                "sha256": envelope_sha,
                "bytes": output.stat().st_size,
            },
            "detached_signature_created": False,
            "nonce_claimed": False,
            "external_subm_data_opened": False,
        }
        release_path = output.with_suffix(output.suffix + ".release.json")
        release_sha = _write_exclusive(release_path, release)
        print(
            json.dumps(
                {
                    "status": "UNSIGNED_NOT_AUTHORIZED",
                    "authorization": {"path": str(output), "sha256": envelope_sha},
                    "release": {"path": str(release_path), "sha256": release_sha},
                    "next_action": "independent review, then detached Ed25519 signature outside workspace",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError, RuntimeError, StaticScoreV3R2Error) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
