"""V7 control-plane tests.

All cryptographic material and checkpoint-like files in this module are
temporary synthetic fixtures.  The tests never open the DANDI external data,
an experiment checkpoint, a formal endpoint, or a production authorization.
They never mint a :class:`VerifiedGrant`.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import io
import os
from pathlib import Path
import stat
import zipfile

import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sua_exploration.mc_maze import subm_co_three_arm_score_only_v7 as core
from sua_exploration.scripts import run_dandi688_subm_co_three_arm_score_only_v7 as runner
from sua_exploration.scripts import write_dandi688_subm_co_three_arm_score_only_prelaunch_v7 as writer


def _write(path: Path, raw: bytes, mode: int = 0o444) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw); os.chmod(path, mode)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "mode": f"0{mode:03o}"}


def _relative_write(root: Path, relative: str, raw: bytes, mode: int = 0o444) -> dict[str, object]:
    pin = _write(root / relative, raw, mode)
    pin["path"] = relative
    return pin


def _roots(tmp_path: Path) -> tuple[core.TrustedRoots, dict[str, Ed25519PrivateKey]]:
    private = {name: Ed25519PrivateKey.generate() for name in ("policy", "checkpoint", "run_auth")}
    dirs = {name: tmp_path / name for name in ("policy", "checkpoints", "run_auth", "outputs", "external", "claims")}
    for directory in dirs.values():
        directory.mkdir()
    pins = {}
    targets = {"policy": "policy", "checkpoint": "checkpoints", "run_auth": "run_auth"}
    for name, location in targets.items():
        raw = private[name].public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        pins[name] = _relative_write(dirs[location], f"keys/{name}.pem", raw)
    payload = {
        "schema": core.TRUSTED_ROOTS_SCHEMA, "root_id": "synthetic-v7-root",
        "policy_root": str(dirs["policy"]), "checkpoint_root": str(dirs["checkpoints"]),
        "run_auth_root": str(dirs["run_auth"]), "output_parent": str(dirs["outputs"]),
        "external_nwb_root": str(dirs["external"]), "claim_root": str(dirs["claims"]),
        "keys": pins,
    }
    return core._install_synthetic_trusted_roots_for_test(payload), private


def _closure_row(roots: core.TrustedRoots, private: Ed25519PrivateKey, *, arm: str = "shared_t4", seed: int = 42) -> dict[str, object]:
    checkpoint = b"synthetic checkpoint only; not an experiment checkpoint\n"
    checkpoint_pin = _relative_write(roots.path("checkpoint_root"), core.checkpoint_path(arm, seed), checkpoint)
    row: dict[str, object] = {
        "arm": arm, "seed": seed, "epoch": 11, "path": checkpoint_pin["path"],
        "sha256": checkpoint_pin["sha256"], "bytes": checkpoint_pin["bytes"], "mode": "0444",
        "status": "INDEPENDENT_V7_TERMINAL_CLOSURE_SIGNED", "closure": None,
    }
    payload = {
        "schema": core.CLOSURE_SCHEMA, "status": core.CLOSURE_STATUS,
        "arm": arm, "seed": seed, "epoch": 11,
        "checkpoint": {key: row[key] for key in ("path", "sha256", "bytes", "mode")},
        "slot_binding_sha256": core.slot_binding(row),
    }
    closure_raw = core.canonical_bytes(payload)
    closure_pin = _relative_write(roots.path("checkpoint_root"), core.closure_relative_path(arm, seed), closure_raw)
    envelope = {
        "schema": core.CLOSURE_ENVELOPE_SCHEMA, "algorithm": "Ed25519",
        "payload_sha256": hashlib.sha256(closure_raw).hexdigest(),
        "signature_b64": base64.b64encode(private.sign(closure_raw)).decode(),
    }
    signature_pin = _relative_write(roots.path("checkpoint_root"), core.closure_signature_relative_path(arm, seed), core.canonical_bytes(envelope))
    row["closure"] = {"kind": "independent_ed25519_checkpoint_closure_v7", "payload": closure_pin, "signature": signature_pin}
    return row


def _cohort() -> list[dict[str, object]]:
    return [
        {
            "asset_id": f"synthetic-{index:02d}", "session_id": f"sub-M_ses-synthetic-{index:02d}",
            "frozen_path": f"sub-M/synthetic-{index:02d}.nwb",
            "nwb_sha256": hashlib.sha256(f"nwb-{index}".encode()).hexdigest(), "nwb_bytes": 100 + index,
        }
        for index in range(core.N)
    ]


def test_checked_in_blocker_has_exact_status_and_no_execution() -> None:
    slots = core.blocked_checkpoint_slots()
    assert [(row["arm"], row["seed"]) for row in slots if row["closure"] is None] == [("shared_zero4", 42), ("shared_zero4", 43), ("shared_zero4", 44)]
    with pytest.raises(core.V7BlockedError, match="BLOCKED_MISSING_ZERO4_TERMINALS"):
        core.refuse_blocked_execution()


def test_blocked_writer_and_runner_seal_only_nonexecutable_temp_package(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "blocked-v7"
    result = writer.write_blocked_prelaunch(output)
    assert result["status"] == "BLOCKED_MISSING_ZERO4_TERMINALS"
    assert result["complete_policy_created"] is False
    assert result["verified_grant_created"] is False
    assert writer.load_stored_blocked_prelaunch(output)["status"] == "BLOCKED_MISSING_ZERO4_TERMINALS"
    assert runner.main(["--mode", "dry-run", "--prelaunch-dir", str(output)]) == 0
    dry = capsys.readouterr().out
    assert '"external_nwb_checkpoint_torch_gpu_r2_complete_policy_grant_allowed": false' in dry
    assert runner.main(["--mode", "score", "--prelaunch-dir", str(output)]) == 2
    assert "BLOCKED_MISSING_ZERO4_TERMINALS" in capsys.readouterr().err


def test_runtime_policy_shape_rejects_embedded_public_key_before_contract(tmp_path: Path) -> None:
    roots, _ = _roots(tmp_path)
    # This is intentionally not a complete policy and is never signed.  The
    # exact-key guard fires before any closure/checkpoint operation.
    policy = {
        "schema": core.POLICY_SCHEMA, "status": core.POLICY_STATUS,
        "trusted_root_id": roots.root_id, "frozen_predecessor_sha256": "a" * 64,
        "cohort": _cohort(), "query_counts": {row["asset_id"]: core.QUERY_WINDOWS_PER_VIEW // core.N for row in _cohort()},
        "reviewed_checkpoint_slots": [], "reviewed_checkpoint_slots_sha256": "b" * 64,
        "source_snapshot_sha256": "c" * 64, "parity_bundle_sha256": "d" * 64,
        "runtime_identity": core.v6.runtime_identity(), "cpu_policy": core.CPU_POLICY,
        "public_key": {"attacker": "must-not-be-a-policy-field"},
    }
    # Fix total only after the exact-key violation: exact schema is the first
    # security boundary and prohibits key/root smuggling categorically.
    with pytest.raises(core.V7AuthorizationError, match="exact schema"):
        core._validate_complete_policy_shape(policy, roots)


def test_trusted_root_key_tamper_is_rejected_at_install(tmp_path: Path) -> None:
    roots, _ = _roots(tmp_path)
    payload = roots.payload
    key_path = Path(payload["policy_root"]) / payload["keys"]["policy"]["path"]
    os.chmod(key_path, 0o600); key_path.write_bytes(b"tampered"); os.chmod(key_path, 0o444)
    with pytest.raises(core.V7LedgerError, match="SHA-256"):
        core._install_synthetic_trusted_roots_for_test(payload)


def test_production_installer_rejects_current_blocked_anchor_and_self_roots(tmp_path: Path) -> None:
    roots, _ = _roots(tmp_path)
    # A caller can construct synthetic material only through the explicitly
    # private fixture helper; formal policy verification rejects it because it
    # was not loaded from the source-pinned formal anchor.
    with pytest.raises(core.V7AuthorizationError, match="pinned trust anchor"):
        core._assert_roots(roots)
    with pytest.raises(core.V7BlockedError, match="BLOCKED_MISSING_ZERO4_TERMINALS"):
        core.install_trusted_roots()


def test_independently_signed_closure_validates_live_synthetic_file_pin(tmp_path: Path) -> None:
    roots, private = _roots(tmp_path)
    row = _closure_row(roots, private["checkpoint"])
    observed = core._verify_complete_closure(row, roots)
    assert observed["arm"] == "shared_t4"
    assert observed["seed"] == 42


@pytest.mark.parametrize("attack", ("checkpoint_mutated", "bad_signature", "closure_symlink"))
def test_closure_attacks_fail_before_any_contract(tmp_path: Path, attack: str) -> None:
    roots, private = _roots(tmp_path)
    row = _closure_row(roots, private["checkpoint"])
    if attack == "checkpoint_mutated":
        checkpoint = roots.path("checkpoint_root") / row["path"]
        os.chmod(checkpoint, 0o600); checkpoint.write_bytes(b"changed after signed closure"); os.chmod(checkpoint, 0o444)
    elif attack == "bad_signature":
        signature = roots.path("checkpoint_root") / row["closure"]["signature"]["path"]
        os.chmod(signature, 0o600); signature.write_bytes(core.canonical_bytes({"schema": core.CLOSURE_ENVELOPE_SCHEMA, "algorithm": "Ed25519", "payload_sha256": "0" * 64, "signature_b64": base64.b64encode(b"x" * 64).decode()})); os.chmod(signature, 0o444)
    else:
        closure = roots.path("checkpoint_root") / row["closure"]["payload"]["path"]
        replacement = closure.with_name("closure-replacement.json")
        closure.rename(replacement); closure.symlink_to(replacement)
    with pytest.raises((core.V7LedgerError, core.V7AuthorizationError)):
        core._verify_complete_closure(row, roots)


def test_canonical_nonce_destination_and_openat_exclusive_write(tmp_path: Path) -> None:
    roots, _ = _roots(tmp_path)
    nonce = "a" * 64
    assert core._nonce_claim_relative_path(nonce) == f"claims/aa/{nonce}.claim.json"
    pin = core._write_exclusive_at(roots.path("claim_root"), core._nonce_claim_relative_path(nonce), core.canonical_bytes({"nonce": nonce}))
    assert pin["path"] == core._nonce_claim_relative_path(nonce)
    assert stat.S_IMODE((roots.path("claim_root") / pin["path"]).stat().st_mode) == 0o444
    with pytest.raises(core.V7LedgerError, match="duplicate"):
        core._write_exclusive_at(roots.path("claim_root"), core._nonce_claim_relative_path(nonce), b"other")
    with pytest.raises(core.V7AuthorizationError, match="nonce"):
        core._nonce_claim_relative_path("not-a-hex-nonce")


def _metric_cases() -> list[tuple[str, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(90210)
    ordinary_target = rng.normal(size=(257, 2)).astype(np.float32)
    ordinary_prediction = (ordinary_target + rng.normal(scale=0.25, size=(257, 2))).astype(np.float32)
    constant_target = np.column_stack((np.linspace(-2, 2, 257), np.full(257, 3.0))).astype(np.float32)
    constant_prediction = constant_target.copy(); constant_prediction[:, 1] = -1.0
    near_target = np.column_stack((np.linspace(-1, 1, 257), np.linspace(0.0, 1.0e-5, 257))).astype(np.float32)
    near_prediction = near_target.copy(); near_prediction[:, 1] += np.float32(1.0)
    perfect_near_target = np.column_stack((np.linspace(-1, 1, 257), np.linspace(0.0, 1.0e-5, 257))).astype(np.float32)
    return [
        ("ordinary", ordinary_prediction, ordinary_target),
        ("constant_nonperfect", constant_prediction, constant_target),
        ("near_constant_nonperfect", near_prediction, near_target),
        ("near_constant_perfect", perfect_near_target.copy(), perfect_near_target),
    ]


def test_frozen_torchmetrics_variance_weighted_golden_parity() -> None:
    torch = pytest.importorskip("torch")
    metrics = pytest.importorskip("torchmetrics.regression")
    for name, prediction, target in _metric_cases():
        observed = core.frozen_torchmetrics_variance_weighted_r2(prediction, target)
        reference = float(metrics.R2Score(multioutput="variance_weighted")(torch.from_numpy(prediction), torch.from_numpy(target)))
        assert np.isfinite(observed), name
        assert abs(observed - reference) <= 1.0e-4, (name, observed, reference)


def test_safe_npz_preparses_npy_and_rejects_duplicate_or_bad_header() -> None:
    prediction = np.ascontiguousarray(np.arange(20, dtype=np.float32).reshape(10, 2))
    target = np.ascontiguousarray(prediction + 1.0)
    raw = core._npz_bytes(prediction, target)
    loaded_prediction, loaded_target = core._safe_npz_from_raw(raw, expected_rows=10)
    np.testing.assert_array_equal(loaded_prediction, prediction)
    np.testing.assert_array_equal(loaded_target, target)
    malformed = io.BytesIO()
    with zipfile.ZipFile(malformed, "w") as archive:
        archive.writestr("prediction.npy", b"not-a-valid-npy")
        archive.writestr("target.npy", b"also-not-a-valid-npy")
    with pytest.raises(core.V7LedgerError):
        core._safe_npz_from_raw(malformed.getvalue(), expected_rows=10)
    duplicate = io.BytesIO()
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("prediction.npy", b"x")
        archive.writestr("prediction.npy", b"x")
        archive.writestr("target.npy", b"x")
    with pytest.raises(core.V7LedgerError):
        core._safe_npz_from_raw(duplicate.getvalue(), expected_rows=10)


def test_module_does_not_import_torch_or_authorize_on_import() -> None:
    source = Path(core.__file__).read_text()
    assert "import torch" not in source
    assert "verify_run_authorization(" in source  # implementation exists
    # No executable code is allowed to call it at module scope.
    assert source.count("return VerifiedGrant(") == 1
