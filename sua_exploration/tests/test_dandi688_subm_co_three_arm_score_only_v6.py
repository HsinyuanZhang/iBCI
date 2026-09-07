from __future__ import annotations

import base64
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sua_exploration.mc_maze import subm_co_three_arm_score_only_v6 as core
from sua_exploration.scripts import run_dandi688_subm_co_three_arm_score_only_v6 as runner
from sua_exploration.scripts import write_dandi688_subm_co_three_arm_score_only_prelaunch_v6 as writer


ROOT = Path(__file__).resolve().parents[2]


def _write_0444(path: Path, raw: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw); os.chmod(path, 0o444)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "mode": "0444"}


def _write_relative_0444(root: Path, relative: str, payload: dict[str, object]) -> dict[str, object]:
    raw = core.canonical_bytes(payload); path = root / relative
    pin = _write_0444(path, raw); pin["path"] = relative
    return pin


def _cohort() -> list[dict[str, object]]:
    return [
        {
            "asset_id": f"synthetic-asset-{index:02d}",
            "session_id": f"sub-M_ses-synthetic-{index:02d}",
            "frozen_path": f"sub-M/synthetic-{index:02d}.nwb",
            "nwb_sha256": hashlib.sha256(f"nwb-{index}".encode()).hexdigest(),
            "nwb_bytes": 1000 + index,
        }
        for index in range(15)
    ]


def _complete_slots(closure_root: Path) -> list[dict[str, object]]:
    slots = copy.deepcopy(core.blocked_checkpoint_slots())
    for row in slots:
        if row["arm"] != "shared_zero4":
            continue
        seed = int(row["seed"]); digest = hashlib.sha256(f"zero4-{seed}".encode()).hexdigest()
        row.update(sha256=digest, bytes=50_000 + seed, mode="0444", status="REVIEWED_FUTURE_SEALED_TERMINAL")
        checkpoint = {key: row[key] for key in ("path", "sha256", "bytes", "mode")}
        receipt_payload = {
            "schema": "terminal_checkpoint_authority_receipt_v6",
            "status": "AUTHORIZED_SEALED_TERMINAL", "arm": row["arm"], "seed": seed,
            "epoch": 11, "checkpoint": checkpoint,
            "slot_binding_sha256": core.slot_binding(row), "issuer": "synthetic-independent-reviewer",
        }
        receipt_pin = _write_relative_0444(closure_root, f"zero4_s{seed}/authority_receipt.json", receipt_payload)
        closure_payload = {
            "schema": "sealed_terminal_checkpoint_closure_v6",
            "status": "SEALED_TERMINAL_CLOSURE_VERIFIED", "arm": row["arm"],
            "seed": seed, "epoch": 11, "checkpoint": checkpoint,
            "slot_binding_sha256": core.slot_binding(row), "authority_receipt": receipt_pin,
        }
        closure_pin = _write_relative_0444(closure_root, f"zero4_s{seed}/closure.json", closure_payload)
        row["closure"] = {
            "kind": "future_zero4_sealed_terminal", **closure_pin,
            "authority_receipt": receipt_pin,
        }
    core.validate_slots(slots, require_complete=True)
    return slots


def _fixture(tmp_path: Path) -> dict[str, object]:
    closure_root = tmp_path / "closures"; closure_root.mkdir()
    slots = _complete_slots(closure_root)
    closure_digest, _ = core.verify_closure_bundle(slots, repository_root=ROOT, closure_root=closure_root)
    output_parent = tmp_path / "outputs"; output_parent.mkdir()
    external_parent = tmp_path / "external"; external_parent.mkdir()
    external_root = external_parent / "subm"; external_root.mkdir()
    claim_root = tmp_path / "claims"
    private = Ed25519PrivateKey.generate()
    public_raw = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    public_path = tmp_path / "auth" / "public.pem"
    public_pin = _write_0444(public_path, public_raw)
    cohort = _cohort(); query = {str(row["asset_id"]): 47_253 for row in cohort}
    policy = {
        "schema": "dandi_000688_subm_three_arm_frozen_policy_v6",
        "frozen_predecessor_sha256": "1" * 64, "cohort": cohort, "query_counts": query,
        "reviewed_checkpoint_slots": slots,
        "reviewed_checkpoint_slots_sha256": core.canonical_sha256(slots),
        "verified_closure_bundle_sha256": closure_digest,
        "source_snapshot_sha256": "2" * 64, "parity_bundle_sha256": "3" * 64,
        "runtime_identity": core.runtime_identity(), "cpu_policy": core.CPU_POLICY,
        "repository_root": str(ROOT), "closure_root": str(closure_root),
        "output_parent": str(output_parent), "external_parent": str(external_parent),
        "claim_root": str(claim_root), "public_key": public_pin,
    }
    contract = core.build_expected_contract(policy)
    return {
        "policy": policy, "contract": contract, "private": private,
        "output_root": output_parent / "run", "external_root": external_root,
        "closure_root": closure_root,
    }


def _auth_files(
    fixture: dict[str, object], tmp_path: Path, *, now: datetime,
    mutate: callable | None = None, invalid_signature: bool = False,
    nonce: str | None = None,
) -> tuple[Path, Path]:
    policy = fixture["policy"]; contract = fixture["contract"]
    authorization = {
        "schema": core.AUTH_SCHEMA, "status": core.AUTH_STATUS,
        "permitted_action": core.AUTH_ACTION,
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(minutes=1)).isoformat(),
        "nonce": nonce or hashlib.sha256(str(tmp_path).encode()).hexdigest(),
        "output_root": str(Path(fixture["output_root"]).absolute()),
        "external_nwb_root": str(Path(fixture["external_root"]).absolute()),
        "cell_count": core.CELL_COUNT,
        "bindings": core.authorization_bindings(policy, contract),
    }
    if mutate is not None:
        mutate(authorization)
    raw = core.canonical_bytes(authorization)
    private = Ed25519PrivateKey.generate() if invalid_signature else fixture["private"]
    signature = private.sign(raw)
    auth_path = tmp_path / "authorization.json"; envelope_path = tmp_path / "signature.json"
    _write_0444(auth_path, raw)
    envelope = {"schema": core.ENVELOPE_SCHEMA, "algorithm": "Ed25519", "authorization_sha256": hashlib.sha256(raw).hexdigest(), "signature_b64": base64.b64encode(signature).decode()}
    _write_0444(envelope_path, core.canonical_bytes(envelope))
    return auth_path, envelope_path


def _grant(fixture: dict[str, object], tmp_path: Path, *, nonce: str | None = None) -> core.VerifiedGrant:
    now = datetime.now(timezone.utc); auth, signature = _auth_files(fixture, tmp_path, now=now, nonce=nonce)
    return core.verify_authorization_and_claim(
        authorization_path=auth, signature_envelope_path=signature,
        output_root=fixture["output_root"], external_nwb_root=fixture["external_root"],
        policy=fixture["policy"], contract=fixture["contract"], now=now,
    )


def _arrays(rows: int, alpha: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    target_1d = (np.arange(rows, dtype=np.int64) & 1).astype(np.float32)
    target = np.ascontiguousarray(np.column_stack((target_1d, target_1d)))
    prediction = np.ascontiguousarray(target * np.float32(alpha))
    return prediction, target


def _publish_one(fixture: dict[str, object], grant: core.VerifiedGrant, key: core.CellKey, alpha: float = 0.9) -> tuple[dict[str, object], dict[str, object]]:
    contract = fixture["contract"]; rows = contract["query_window_count_by_asset_id"][key.asset_id]
    prediction, target = _arrays(rows, alpha)
    artifact = core.publish_prediction_target_artifact(grant, fixture["policy"], contract, key=key, prediction=prediction, target=target)
    cell = core.publish_cell_result(grant, fixture["policy"], contract, key=key, prediction_target_artifact=artifact)
    return artifact, cell


def _rewrite_json(path: Path, payload: dict[str, object]) -> None:
    os.chmod(path, 0o600); path.write_bytes(core.canonical_bytes(payload)); os.chmod(path, 0o444)


def test_blocked_slots_remain_exact_three_zero4_closures() -> None:
    slots = core.blocked_checkpoint_slots()
    assert [(row["arm"], row["seed"]) for row in slots if row["closure"] is None] == [("shared_zero4", 42), ("shared_zero4", 43), ("shared_zero4", 44)]
    with pytest.raises(core.V6BlockedError, match="BLOCKED_MISSING_ZERO4_TERMINALS"):
        core.validate_slots(slots, require_complete=True)
    forged = copy.deepcopy(slots)
    row = next(row for row in forged if row["arm"] == "shared_zero4" and row["seed"] == 42)
    row.update(sha256="f" * 64, bytes=123, mode="0444")
    with pytest.raises(core.V6Error, match="invalid incomplete slot"):
        core.validate_slots(forged, require_complete=False)


def test_ordinary_verified_grant_construction_is_rejected() -> None:
    with pytest.raises(TypeError, match="only be minted"):
        core.VerifiedGrant(proof={})


def test_even_stolen_internal_mint_token_cannot_forge_crypto_proof(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    missing_pin = {"path": str(tmp_path / "missing"), "sha256": "0" * 64, "bytes": 1, "mode": "0444"}
    proof = {
        "schema": core.GRANT_SCHEMA,
        "authorization": missing_pin, "signature_envelope": missing_pin,
        "public_key": missing_pin, "nonce": "4" * 64,
        "nonce_claim": missing_pin,
        "output_root": str(fixture["output_root"]),
        "external_nwb_root": str(fixture["external_root"]),
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(),
        "bindings": core.authorization_bindings(fixture["policy"], fixture["contract"]),
    }
    forged = core.VerifiedGrant(_mint=core._GRANT_MINT_TOKEN, proof=proof)
    with pytest.raises(core.V6LedgerError):
        core.scan_resume_state(forged, fixture["policy"], fixture["contract"])


def test_real_ed25519_verifier_mints_immutable_full_digest_grant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    fixture = _fixture(tmp_path); grant = _grant(fixture, tmp_path)
    assert core.is_sha256(grant.verified_grant_sha256)
    assert grant.proof["bindings"] == core.authorization_bindings(fixture["policy"], fixture["contract"])
    assert stat.S_IMODE(Path(grant.proof["nonce_claim"]["path"]).stat().st_mode) == 0o444
    with pytest.raises(AttributeError, match="immutable"):
        grant.verified_grant_sha256 = "0" * 64


@pytest.mark.parametrize("case", ("bad_signature", "expired", "too_long", "root", "binding"))
def test_signature_time_root_and_binding_attacks_fail_before_grant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    fixture = _fixture(tmp_path); now = datetime.now(timezone.utc)
    def mutate(auth: dict[str, object]) -> None:
        if case == "expired":
            auth["issued_at"] = (now - timedelta(minutes=3)).isoformat(); auth["expires_at"] = (now - timedelta(minutes=2)).isoformat()
        elif case == "too_long":
            auth["issued_at"] = now.isoformat(); auth["expires_at"] = (now + timedelta(minutes=16)).isoformat()
        elif case == "root": auth["output_root"] = str(tmp_path / "wrong")
        elif case == "binding": auth["bindings"] = {**auth["bindings"], "source_snapshot_sha256": "f" * 64}
    auth, sig = _auth_files(fixture, tmp_path, now=now, mutate=mutate, invalid_signature=case == "bad_signature")
    pattern = {"bad_signature": "invalid Ed25519", "expired": "expired", "too_long": "15 minutes", "root": "root binding", "binding": "digest bindings"}[case]
    with pytest.raises(core.V6AuthorizationError, match=pattern):
        core.verify_authorization_and_claim(authorization_path=auth, signature_envelope_path=sig, output_root=fixture["output_root"], external_nwb_root=fixture["external_root"], policy=fixture["policy"], contract=fixture["contract"], now=now)


def test_atomic_nonce_replay_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    fixture = _fixture(tmp_path); now = datetime.now(timezone.utc); nonce = "a" * 64
    auth, sig = _auth_files(fixture, tmp_path, now=now, nonce=nonce)
    kwargs = dict(authorization_path=auth, signature_envelope_path=sig, output_root=fixture["output_root"], external_nwb_root=fixture["external_root"], policy=fixture["policy"], contract=fixture["contract"], now=now)
    core.verify_authorization_and_claim(**kwargs)
    with pytest.raises(core.V6AuthorizationError, match="already claimed"):
        core.verify_authorization_and_claim(**kwargs)


def test_exact_contract_reconstruction_rejects_self_rehash(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path); contract = copy.deepcopy(fixture["contract"])
    contract["score_protocol"]["support_trials"] = 49
    body = dict(contract); body.pop("contract_sha256"); contract["contract_sha256"] = core.canonical_sha256(body)
    with pytest.raises(core.V6AuthorizationError, match="exact reconstruction"):
        core.validate_expected_contract(contract, fixture["policy"], require_complete=True)


@pytest.mark.parametrize("case", ("mutable", "hash", "schema", "authority", "symlink"))
def test_future_zero4_closure_live_verification_attacks(tmp_path: Path, case: str) -> None:
    fixture = _fixture(tmp_path); policy = fixture["policy"]; slots = policy["reviewed_checkpoint_slots"]
    row = next(row for row in slots if row["arm"] == "shared_zero4" and row["seed"] == 42)
    closure_path = Path(policy["closure_root"]) / row["closure"]["path"]
    if case == "mutable": os.chmod(closure_path, 0o644)
    elif case == "hash": row["closure"]["sha256"] = "f" * 64
    elif case == "schema":
        payload = json.loads(closure_path.read_text()); payload["schema"] = "wrong"; _rewrite_json(closure_path, payload); row["closure"]["sha256"] = hashlib.sha256(closure_path.read_bytes()).hexdigest(); row["closure"]["bytes"] = closure_path.stat().st_size
    elif case == "authority":
        receipt = Path(policy["closure_root"]) / row["closure"]["authority_receipt"]["path"]; os.chmod(receipt, 0o644)
    else:
        real = closure_path.with_name("real.json"); closure_path.rename(real); closure_path.symlink_to(real)
    with pytest.raises((core.V6LedgerError, core.V6AuthorizationError)):
        core.verify_closure_bundle(slots, repository_root=ROOT, closure_root=Path(policy["closure_root"]))


def test_npz_shape_dtype_finite_and_exact_members(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    fixture = _fixture(tmp_path); grant = _grant(fixture, tmp_path); contract = fixture["contract"]
    key = core.expected_cell_keys(contract, fixture["policy"])[0]; rows = contract["query_window_count_by_asset_id"][key.asset_id]
    pred, target = _arrays(rows)
    with pytest.raises(core.V6LedgerError, match="shape"):
        core.publish_prediction_target_artifact(grant, fixture["policy"], contract, key=key, prediction=pred[:-1], target=target[:-1])
    with pytest.raises(core.V6LedgerError, match="dtype"):
        core.publish_prediction_target_artifact(grant, fixture["policy"], contract, key=key, prediction=pred.astype(np.float64), target=target.astype(np.float64))
    pred[0, 0] = np.nan
    with pytest.raises(core.V6LedgerError, match="NaN/Inf"):
        core.publish_prediction_target_artifact(grant, fixture["policy"], contract, key=key, prediction=pred, target=target)
    buffer = io.BytesIO(); np.savez_compressed(buffer, prediction=target, target=target, extra=target)
    with pytest.raises(core.V6LedgerError, match="exact prediction/target"):
        core._safe_npz_from_raw(buffer.getvalue(), expected_rows=rows)
    buffer = io.BytesIO(); np.savez_compressed(buffer, prediction=np.asarray([[object()]], dtype=object), target=np.asarray([[object()]], dtype=object))
    with pytest.raises(core.V6LedgerError):
        core._safe_npz_from_raw(buffer.getvalue(), expected_rows=1)
    fortran = np.asfortranarray(target)
    buffer = io.BytesIO(); np.savez_compressed(buffer, prediction=fortran, target=fortran)
    with pytest.raises(core.V6LedgerError, match="C-contiguous"):
        core._safe_npz_from_raw(buffer.getvalue(), expected_rows=rows)


def test_no_caller_r2_and_resume_recomputes_r2_and_grant_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    fixture = _fixture(tmp_path); grant = _grant(fixture, tmp_path); contract = fixture["contract"]
    assert "r2" not in inspect.signature(core.publish_cell_result).parameters
    key = core.expected_cell_keys(contract, fixture["policy"])[0]
    artifact, cell = _publish_one(fixture, grant, key)
    assert artifact["verified_grant_sha256"] == grant.verified_grant_sha256
    cell_path = grant.output_root / cell["path"]; payload = json.loads(cell_path.read_text())
    assert payload["verified_grant_sha256"] == grant.verified_grant_sha256 and payload["r2"] <= 1.0
    payload["r2"] = 1.1; _rewrite_json(cell_path, payload)
    with pytest.raises(core.V6LedgerError, match="exact internal recomputation"):
        core.scan_resume_state(grant, fixture["policy"], contract)


def test_mutable_symlink_duplicate_partial_unknown_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    fixture = _fixture(tmp_path); grant = _grant(fixture, tmp_path); contract = fixture["contract"]
    key = core.expected_cell_keys(contract, fixture["policy"])[0]; rows = contract["query_window_count_by_asset_id"][key.asset_id]; pred, target = _arrays(rows)
    artifact = core.publish_prediction_target_artifact(grant, fixture["policy"], contract, key=key, prediction=pred, target=target)
    with pytest.raises(core.V6LedgerError, match="duplicate"):
        core.publish_prediction_target_artifact(grant, fixture["policy"], contract, key=key, prediction=pred, target=target)
    with pytest.raises(core.V6LedgerError, match="partial/unknown artifact"):
        core.scan_resume_state(grant, fixture["policy"], contract)
    core.publish_cell_result(grant, fixture["policy"], contract, key=key, prediction_target_artifact=artifact)
    artifact_path = grant.output_root / artifact["path"]; os.chmod(artifact_path, 0o644)
    with pytest.raises(core.V6LedgerError, match="mode drift"):
        core.scan_resume_state(grant, fixture["policy"], contract)
    os.chmod(artifact_path, 0o444)
    unknown = grant.output_root / "cells" / "unknown"; unknown.mkdir()
    with pytest.raises(core.V6LedgerError, match="unknown cell directory"):
        core.scan_resume_state(grant, fixture["policy"], contract)
    unknown.rmdir()
    real = artifact_path.with_name("real.npz"); artifact_path.rename(real); artifact_path.symlink_to(real)
    with pytest.raises(core.V6LedgerError, match="safely open|symlink"):
        core.scan_resume_state(grant, fixture["policy"], contract)


def test_full_270_internal_r2_aggregate_and_tamper_recompute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    fixture = _fixture(tmp_path); grant = _grant(fixture, tmp_path); contract = fixture["contract"]
    alpha = {"shared_t4": 0.90, "shared_zero4": 0.75, "shared_ts4": 0.80}
    for key in core.expected_cell_keys(contract, fixture["policy"]):
        _publish_one(fixture, grant, key, alpha=alpha[key.arm])
    aggregate_pin = core.publish_full_aggregate(grant, fixture["policy"], contract)
    aggregate_path = grant.output_root / aggregate_pin["path"]
    aggregate = json.loads(aggregate_path.read_text())
    assert aggregate["verified_grant_sha256"] == grant.verified_grant_sha256
    assert aggregate["verified_cell_count"] == 270 and aggregate["bootstrap_policy"]["replicates"] == 100_000
    assert aggregate["overall_three_arm_claim_pass"] is True
    state = core.scan_resume_state(grant, fixture["policy"], contract)
    assert state["complete_cell_count"] == 270
    aggregate["comparisons"]["t4_minus_zero4"]["views"]["sua"]["grand_paired_mean_r2"] += 0.001
    _rewrite_json(aggregate_path, aggregate)
    with pytest.raises(core.V6LedgerError, match="canonical exact recomputation"):
        core.scan_resume_state(grant, fixture["policy"], contract)


def test_partial_aggregate_without_270_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    fixture = _fixture(tmp_path); grant = _grant(fixture, tmp_path)
    aggregate = grant.output_root / "aggregate"; aggregate.mkdir(parents=True)
    fake = aggregate / "aggregate.json"; fake.write_bytes(core.canonical_bytes({})); os.chmod(fake, 0o444)
    with pytest.raises(core.V6LedgerError, match="without 270"):
        core.scan_resume_state(grant, fixture["policy"], fixture["contract"])


def test_aggregate_api_accepts_no_caller_statistics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    fixture = _fixture(tmp_path); grant = _grant(fixture, tmp_path)
    assert "aggregate" not in inspect.signature(core.publish_full_aggregate).parameters
    with pytest.raises(TypeError):
        core.publish_full_aggregate(grant, fixture["policy"], fixture["contract"], aggregate={})  # type: ignore[call-arg]


def test_unknown_topology_file_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    fixture = _fixture(tmp_path); grant = _grant(fixture, tmp_path)
    grant.output_root.mkdir(); (grant.output_root / "unknown.tmp").write_bytes(b"x")
    with pytest.raises(core.V6LedgerError, match="topology"):
        core.scan_resume_state(grant, fixture["policy"], fixture["contract"])


def test_blocked_writer_dry_run_and_v5_immutability(tmp_path: Path) -> None:
    output = tmp_path / "blocked-v6"; result = writer.write_blocked_prelaunch(output)
    assert result["status"] == "BLOCKED_MISSING_ZERO4_TERMINALS" and result["verified_grant_created"] is False
    stored = writer.load_stored_blocked_prelaunch(output); payload = runner.dry_run_payload(output)
    assert stored["external_capability_created"] is False and payload["checkpoint_nwb_torch_gpu_external_r2_allowed"] is False
    for name, (digest, size) in writer.V5_PINS.items():
        path = writer.V5_ROOT / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest and path.stat().st_size == size and stat.S_IMODE(path.stat().st_mode) == 0o444


def test_blocked_dry_run_does_not_import_torch_or_cryptography(tmp_path: Path) -> None:
    output = tmp_path / "blocked-v6"; writer.write_blocked_prelaunch(output)
    script = (
        "import json,sys; from pathlib import Path; "
        "from sua_exploration.scripts.run_dandi688_subm_co_three_arm_score_only_v6 import dry_run_payload; "
        f"p=dry_run_payload(Path({str(output)!r})); "
        "print(json.dumps({'status':p['status'],'torch':'torch' in sys.modules,'crypto':'cryptography' in sys.modules}))"
    )
    done = subprocess.run([sys.executable, "-c", script], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    assert json.loads(done.stdout) == {"status": "BLOCKED_MISSING_ZERO4_TERMINALS", "torch": False, "crypto": False}
