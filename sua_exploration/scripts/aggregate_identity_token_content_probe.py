#!/usr/bin/env python3
"""Fail-closed A4-v2 aggregate for one paired AC4/Z4 CPU receipt.

The A4 runner intentionally emits both paired arms into one immutable receipt:
they share the raw M=30 carrier target arrays, session roster, normalizers,
and within-training-session null manifest.  This companion aggregate rechecks
all of those bindings before applying the frozen cross-session read rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SUA_ROOT = Path(__file__).resolve().parents[1]
if str(SUA_ROOT) not in sys.path:
    sys.path.insert(0, str(SUA_ROOT))

from mc_maze.identity_token_content_probe import (  # noqa: E402
    ACTIVITY_CALIBRATION_N,
    CARRIER_TARGET_POOL_N,
    PAIRING_PERMUTATION_VERSION,
    PROBE_NAME,
    SCHEMA_VERSION,
    SEALED_TEST_SESSIONS,
    TARGET_NAMES,
    TRIAL_LENGTH,
    WINDOW_SIZE,
    canonical_json_bytes,
    evaluate_paired_gate,
    sha256_bytes,
)


REQUIRED_PROTOCOL = {
    "activity_calibration_n": ACTIVITY_CALIBRATION_N,
    "carrier_target_pool_n": CARRIER_TARGET_POOL_N,
    "same_support_for_identity_and_raw_carrier_target": True,
    "window_size": WINDOW_SIZE,
    "trial_length": TRIAL_LENGTH,
    "cpu_only": True,
    "no_training_no_backward": True,
    "map_location": "cpu",
    "estimator": "session_loso_global_ridge",
    "pairing_permutation_version": PAIRING_PERMUTATION_VERSION,
}

# The aggregate intentionally names every expected value instead of accepting
# whichever values were serialized by a runner.  This protects both the ridge
# fit and the deterministic within-training-session permutation null from a
# post-protocol scientific override.
FROZEN_PROBE_HYPERPARAMETERS = {
    "estimator": "session_loso_global_ridge",
    "feature_standardization": "fit_on_five_training_sessions_only",
    "intercept": "unpenalized",
    "normalized_ridge_lambda": 1.0,
    "random_seed": 42,
    "modulation_eps": 1.0e-6,
    "min_session_units": 15,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_bytes_exclusive(path: Path, payload: bytes, *, label: str) -> None:
    """Publish one immutable aggregate artifact without a check-then-write race."""
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing {label}: {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        os.fchmod(handle.fileno(), 0o444)


def write_aggregate(payload: Mapping[str, Any], output_path: Path) -> None:
    """Write a gate result and its full-file hash as immutable paired artifacts."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_exclusive(
        output_path,
        (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
        label="A4 aggregate",
    )
    sidecar_path = output_path.with_name(output_path.name + ".sha256")
    _write_bytes_exclusive(
        sidecar_path,
        f"{_sha256_file(output_path)}  {output_path.name}\n".encode("ascii"),
        label="A4 aggregate SHA sidecar",
    )


def _read_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read A4 receipt {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"A4 receipt must be a JSON object: {path}")
    return payload


def _verify_receipt_integrity(path: Path, receipt: Mapping[str, Any]) -> None:
    """Bind a receipt to either its canonical body hash or its immutable file sidecar.

    New receipts contain both forms.  Accepting either preserves validation of
    an otherwise valid legacy artifact, but a present integrity binding must
    always verify exactly; a stale or malformed sidecar cannot be ignored.
    """
    verified = False
    embedded_hash = receipt.get("receipt_body_sha256")
    if embedded_hash is not None:
        _require(isinstance(embedded_hash, str), "A4 receipt body SHA is malformed")
        try:
            observed_body_hash = sha256_bytes(
                canonical_json_bytes(receipt, exclude_keys=("receipt_body_sha256",))
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("A4 receipt body cannot be canonicalized for integrity verification") from exc
        _require(embedded_hash == observed_body_hash, "A4 receipt canonical body SHA drift")
        verified = True

    sidecar_path = path.with_name(path.name + ".sha256")
    if sidecar_path.exists():
        try:
            sidecar = sidecar_path.read_text(encoding="ascii")
        except OSError as exc:
            raise ValueError(f"cannot read A4 receipt SHA sidecar {sidecar_path}: {exc}") from exc
        expected_sidecar = f"{_sha256_file(path)}  {path.name}\n"
        _require(sidecar == expected_sidecar, "A4 receipt SHA sidecar drift")
        verified = True

    _require(verified, "A4 receipt lacks a verified canonical body SHA or SHA sidecar")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_checkpoint_row(row: Mapping[str, Any], *, arm: str, epoch_window: list[int]) -> None:
    _require(row.get("arm") == arm, f"{arm}: arm label drift")
    _require(row.get("arm_kind") == {"ac4": "carrier", "z4": "activity_only"}[arm], f"{arm}: arm kind drift")
    metadata = row.get("checkpoint_metadata") or {}
    _require(metadata.get("side_group") == arm, f"{arm}: metadata side group drift")
    _require(metadata.get("pool_size") == ACTIVITY_CALIBRATION_N, f"{arm}: metadata M=30 drift")
    checkpoints = row.get("epoch_checkpoints") or {}
    _require(set(checkpoints) == {str(epoch) for epoch in epoch_window}, f"{arm}: incomplete fixed epoch window")
    for epoch in epoch_window:
        evidence = checkpoints[str(epoch)]
        checkpoint = Path(str(evidence.get("checkpoint_path", ""))).expanduser().resolve()
        _require(checkpoint.is_file(), f"{arm}/epoch{epoch}: checkpoint absent")
        _require(
            _sha256_file(checkpoint) == evidence.get("checkpoint_sha256"),
            f"{arm}/epoch{epoch}: checkpoint SHA drift",
        )
        token_digests = evidence.get("identity_token_digests_by_session")
        _require(isinstance(token_digests, dict) and token_digests, f"{arm}/epoch{epoch}: missing token digests")


def _validate_session_loso(row: Mapping[str, Any], *, arm: str, sessions: list[str]) -> dict[str, Any]:
    loso = row.get("session_loso") or {}
    _require(loso.get("epoch_window") == list(range(5, 13)), f"{arm}: epoch-window drift")
    per_session = loso.get("per_session") or {}
    _require(set(per_session) == set(sessions), f"{arm}: incomplete held-out session matrix")
    manifest = loso.get("pairing_permutation") or {}
    _require(
        manifest.get("version") == PAIRING_PERMUTATION_VERSION,
        f"{arm}: pairing-permutation version drift",
    )
    _require(manifest.get("session_order") == sessions, f"{arm}: permutation session order drift")
    folds = manifest.get("folds") or {}
    _require(set(folds) == set(sessions), f"{arm}: incomplete permutation folds")
    for session in sessions:
        row_session = per_session[session]
        _require(row_session.get("held_out_session") == session, f"{arm}/{session}: held-out label drift")
        _require(
            row_session.get("training_sessions") == [name for name in sessions if name != session],
            f"{arm}/{session}: training roster drift",
        )
        _require(int(row_session.get("n_held_out_units", 0)) >= 15, f"{arm}/{session}: insufficient held-out units")
        _require(int(row_session.get("n_train_units", 0)) > 0, f"{arm}/{session}: empty training units")
        probes = row_session.get("probes") or {}
        null = row_session.get("pairing_permutation_null") or {}
        _require(set(probes) == set(TARGET_NAMES), f"{arm}/{session}: incomplete honest target matrix")
        _require(set(null) == set(TARGET_NAMES), f"{arm}/{session}: incomplete null target matrix")
        for target in TARGET_NAMES:
            _require("r2" in probes[target] and "r2" in null[target], f"{arm}/{session}/{target}: missing R2")
        _require("mean_cosine" in probes["phase_unit"], f"{arm}/{session}: missing phase cosine")
        _require("mean_cosine" in null["phase_unit"], f"{arm}/{session}: missing null phase cosine")
    return loso


def validate_receipt(path: Path) -> dict[str, Any]:
    receipt = _read_receipt(path)
    _verify_receipt_integrity(path, receipt)
    _require(receipt.get("schema_version") == SCHEMA_VERSION, "A4 schema version drift")
    _require(receipt.get("probe_name") == PROBE_NAME, "A4 probe name drift")
    _require(receipt.get("sealed_test_sessions_opened") is False, "A4 opened sealed formal sessions")
    execution = receipt.get("execution_scope") or {}
    _require(execution.get("cpu_only") is True, "A4 receipt is not CPU-only")
    _require(execution.get("cuda_visible_devices") == "", "A4 exposed CUDA")
    _require(execution.get("torch_cuda_available") is False, "A4 ran with CUDA available")
    _require(execution.get("no_training_no_backward") is True, "A4 training/backward guard drift")
    sessions = receipt.get("sessions")
    _require(isinstance(sessions, list) and len(sessions) == 6 and len(set(sessions)) == 6, "A4 requires six unique development sessions")
    _require(not (set(sessions) & set(SEALED_TEST_SESSIONS)), "A4 receipt intersects sealed formal sessions")
    _require(execution.get("opened_nwb_sessions") == sessions, "A4 NWB-open roster drift")
    _require(execution.get("opened_nwb_session_count") == 6, "A4 NWB-open count drift")
    protocol = receipt.get("protocol") or {}
    for key, expected in REQUIRED_PROTOCOL.items():
        _require(protocol.get(key) == expected, f"A4 protocol drift: {key}")
    _require(
        receipt.get("probe_hyperparameters") == FROZEN_PROBE_HYPERPARAMETERS,
        "A4 frozen probe hyperparameters drift",
    )
    binding = receipt.get("checkpoint_binding") or {}
    _require(binding.get("fixed_epoch_window") == list(range(5, 13)), "A4 checkpoint epoch rule drift")
    _require(binding.get("validation_roster") == sessions, "A4 checkpoint validation roster drift")
    _require(len(binding.get("train_roster") or []) == 27, "A4 checkpoint train roster drift")
    shared_inputs = receipt.get("shared_validation_input_digests") or {}
    _require(set(shared_inputs) == set(sessions), "A4 missing shared M30 input digests")
    for session in sessions:
        row = shared_inputs[session]
        _require(isinstance(row, Mapping), f"{session}: malformed shared input digest")
        n_units = row.get("n_units")
        _require(isinstance(n_units, int) and not isinstance(n_units, bool) and n_units > 0, f"{session}: invalid unit count")
        _require(
            row.get("activity_calibration_shape")
            == [ACTIVITY_CALIBRATION_N, TRIAL_LENGTH, n_units],
            f"{session}: activity calibration must be [M,T,N]=[{ACTIVITY_CALIBRATION_N}, {TRIAL_LENGTH}, {n_units}]",
        )
        _require(
            row.get("raw_carrier_target_shape") == [n_units, 4],
            f"{session}: raw carrier must be [N,4]=[{n_units}, 4]",
        )
        _require(row.get("activity_calibration_digest_sha256"), f"{session}: missing activity digest")
        _require(row.get("raw_carrier_target_digest_sha256"), f"{session}: missing carrier digest")
    normalizer = receipt.get("normalizer_provenance") or {}
    side = normalizer.get("side_normalizer") or {}
    _require(side.get("pool_size") == ACTIVITY_CALIBRATION_N, "A4 normalizer M drift")
    _require(side.get("sha256") == binding.get("normalization_sha256"), "A4 normalizer/checkpoint SHA drift")
    _require(side.get("metadata_sha256") == binding.get("normalization_sha256"), "A4 metadata normalizer SHA drift")
    arms = receipt.get("arms") or {}
    _require(set(arms) == {"ac4", "z4"}, "A4 must contain exactly AC4 and Z4")
    _validate_checkpoint_row(arms["ac4"], arm="ac4", epoch_window=list(range(5, 13)))
    _validate_checkpoint_row(arms["z4"], arm="z4", epoch_window=list(range(5, 13)))
    ac4_loso = _validate_session_loso(arms["ac4"], arm="ac4", sessions=sessions)
    z4_loso = _validate_session_loso(arms["z4"], arm="z4", sessions=sessions)
    _require(
        ac4_loso.get("pairing_permutation") == z4_loso.get("pairing_permutation"),
        "AC4/Z4 pairing-permutation target rows or seeds drift",
    )
    # The carrier target rows are shared by construction, but the independent
    # digests make that fact auditable without trusting the runner's prose.
    _require(
        arms["ac4"].get("side_input_digests_by_session") != arms["z4"].get("side_input_digests_by_session"),
        "AC4 and Z4 side inputs unexpectedly identical",
    )
    return receipt


def aggregate(*, receipt_path: Path) -> dict[str, Any]:
    receipt = validate_receipt(receipt_path)
    sessions = receipt["sessions"]
    ac4_loso = receipt["arms"]["ac4"]["session_loso"]
    z4_loso = receipt["arms"]["z4"]["session_loso"]
    verdict = evaluate_paired_gate(
        z4_loso["pooled"],
        ac4_loso["pooled"],
        activity_only_per_session=z4_loso["per_session"],
        carrier_per_session=ac4_loso["per_session"],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "probe_name": PROBE_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_receipt": str(receipt_path.resolve()),
        "source_receipt_sha256": _sha256_file(receipt_path),
        "sessions": sessions,
        "sealed_test_sessions_opened": False,
        "checkpoint_binding": receipt["checkpoint_binding"],
        "protocol": receipt["protocol"],
        "probe_hyperparameters": receipt["probe_hyperparameters"],
        "normalizer_provenance": receipt["normalizer_provenance"],
        "ac4_pooled": ac4_loso["pooled"],
        "z4_pooled": z4_loso["pooled"],
        "pairing_permutation": ac4_loso["pairing_permutation"],
        "gate_verdict": verdict,
        "phase_blindness_gate_pass": bool(verdict["phase_blindness_gate_pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    receipt_path = args.receipt.expanduser().resolve()
    out = args.out.expanduser().resolve()
    payload = aggregate(receipt_path=receipt_path)
    write_aggregate(payload, out)
    print(json.dumps(payload["gate_verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
