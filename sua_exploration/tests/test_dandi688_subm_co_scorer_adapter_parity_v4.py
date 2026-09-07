"""Static/preauthorization tests for the v4 CPU parity execution package.

No test imports the v3 helper or Torch/PyNWB/NumPy/model/data owners. A private
Ed25519 key exists only in test process memory; tests write its public component
to pytest's temporary directory and never write a private key to the workspace.
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


ROOT = Path(__file__).resolve().parents[2]
CORE_SOURCE = ROOT / "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_execution_v4.py"
WRITER_SOURCE = ROOT / "sua_exploration/scripts/write_dandi688_subm_co_scorer_adapter_parity_prelaunch_v4.py"
RUNNER_SOURCE = ROOT / "sua_exploration/scripts/run_dandi688_subm_co_scorer_adapter_parity_v4.py"


def _writer():
    spec = importlib.util.spec_from_file_location("subm_parity_v4_writer_test", WRITER_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _core():
    spec = importlib.util.spec_from_file_location("subm_parity_v4_core_test", CORE_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _test_policy(tmp_path: Path, core):
    tmp_path.mkdir(parents=True, exist_ok=True)
    private = Ed25519PrivateKey.generate()
    public = tmp_path / "root-public.pem"
    public.write_bytes(
        private.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    source = tmp_path / "pinned-source.py"
    source.write_text("PINNED = 'v4'\n", encoding="utf-8")
    output_parent = tmp_path / "runs"
    policy = {
        "public_key": {
            "path": str(public.resolve()),
            "sha256": hashlib.sha256(public.read_bytes()).hexdigest(),
            "bytes": public.stat().st_size,
        },
        "source_pins": {
            str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "runtime_identity": core.runtime_identity(),
        "cpu_policy": {
            "device": "cpu",
            "cuda_visible_devices": "",
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "deterministic_algorithms": True,
            "tf32": False,
            "autocast": False,
        },
        "output_parent": str(output_parent.resolve()),
        "claim_root": str((tmp_path / "nonce-claims").resolve()),
        "authorization_bindings": {
            "prelaunch_bundle": {"draft": "test-only"},
            "v3_sources": {"test-only": "test-only"},
            "v3_bundle": {"test-only": "test-only"},
            "fixture_pins": {"test-only": "test-only"},
            "runtime_identity": core.runtime_identity(),
            "cpu_policy": {
                "device": "cpu",
                "cuda_visible_devices": "",
                "torch_num_threads": 1,
                "torch_num_interop_threads": 1,
                "deterministic_algorithms": True,
                "tf32": False,
                "autocast": False,
            },
            "public_key": {
                "path": str(public.resolve()),
                "sha256": hashlib.sha256(public.read_bytes()).hexdigest(),
                "bytes": public.stat().st_size,
            },
            "source_snapshot": {
                str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest(),
            },
        },
    }
    policy["execution_policy_sha256"] = hashlib.sha256(
        core._canonical_bytes(policy)
    ).hexdigest()
    return private, public, source, policy


def _write_signed_authorization(
    tmp_path: Path,
    *,
    core,
    private: Ed25519PrivateKey,
    policy: dict,
    output_root: Path,
    now: datetime,
    issued: datetime | None = None,
    expires: datetime | None = None,
    nonce: str = "1" * 64,
) -> tuple[Path, Path]:
    body = {
        "schema": core.AUTHORIZATION_SCHEMA,
        "kind": "dandi_000688_subc_cpu_parity_execution",
        "status": core.AUTHORIZATION_STATUS,
        "authorization_id": "parity-v4-test",
        "single_use_nonce": nonce,
        "issued_at": (issued or now - timedelta(seconds=5)).isoformat(),
        "expires_at": (expires or now + timedelta(seconds=120)).isoformat(),
        "permitted_action": core.PERMITTED_ACTION,
        "output_root": str(output_root.resolve()),
        "execution_policy_sha256": policy["execution_policy_sha256"],
        "bindings": policy["authorization_bindings"],
        "external_subm_scoring_permitted": False,
        "normalizer_fitting_permitted": False,
        "optimizer_or_backward_permitted": False,
    }
    envelope = {
        "schema": core.AUTHORIZATION_ENVELOPE_SCHEMA,
        "authorization": body,
    }
    raw = core._canonical_bytes(envelope)
    authorization = tmp_path / f"auth-{nonce}.json"
    signature = tmp_path / f"auth-{nonce}.sig"
    authorization.write_bytes(raw)
    signature.write_bytes(base64.b64encode(private.sign(raw)))
    return authorization, signature


def _assert_no_data_access(audit) -> None:
    assert audit.zero_data_access()
    assert audit.checkpoint_files_opened == 0
    assert audit.nwb_files_opened == 0
    assert audit.npz_files_opened == 0
    assert audit.model_forward_calls == 0


def test_static_prelaunch_pins_dedicated_root_key_v3_closure_and_fixed_cpu_runtime(
    tmp_path: Path,
) -> None:
    writer = _writer()
    authority = writer.static_authority(ROOT)
    assert authority["public_key"]["path"].endswith(
        "dandi688_subc_parity_v4_root_ed25519_public.pem"
    )
    assert authority["public_key"]["sha256"] == (
        "a541b5aabc7922251e797da13727b97ceb4ed48de7e95d5b1c4031ba96c602a9"
    )
    assert authority["fixture_pins"]["protocol"] == {
        "view": "sua",
        "support_trials": 50,
        "identity": "first_n30",
        "identity_trials": 30,
        "query": "trials[50:]",
        "loader_batch_size": 128,
        "loader_shuffle": False,
        "loader_num_workers": 0,
    }
    result = writer.write_prelaunch(tmp_path / "v4-prelaunch", ROOT)
    assert result["status"] == "STATIC_V4_CPU_PARITY_EXECUTION_PACKAGE_NOT_AUTHORIZED"
    stored = writer.load_stored_prelaunch(tmp_path / "v4-prelaunch", ROOT)
    assert stored["execution_policy"]["execution_policy_sha256"] == stored["execution_policy_sha256"]
    assert stored["execution_policy"]["cpu_policy"]["torch_num_threads"] == 1
    for path in (tmp_path / "v4-prelaunch").iterdir():
        assert path.stat().st_mode & 0o777 == 0o444


def test_invalid_signature_fails_before_any_data_access_or_nonce_claim(tmp_path: Path) -> None:
    core = _core()
    now = datetime.now(timezone.utc)
    private, _public, _source, policy = _test_policy(tmp_path, core)
    output = tmp_path / "runs" / "result"
    authorization, signature = _write_signed_authorization(
        tmp_path, core=core, private=private, policy=policy, output_root=output, now=now
    )
    signature.write_bytes(base64.b64encode(b"x" * 64))
    audit = core.PreAuthorizationAudit()
    with pytest.raises(core.ParityV4AuthorizationError, match="invalid detached"):
        core._preauthorize_and_claim_with_policy(
            authorization_path=authorization,
            signature_path=signature,
            output_root=output,
            policy=policy,
            now=now,
            audit=audit,
        )
    _assert_no_data_access(audit)
    assert audit.nonce_claim_writes == 0
    assert not Path(policy["claim_root"]).exists()


def test_expired_authorization_fails_before_any_data_access_or_nonce_claim(tmp_path: Path) -> None:
    core = _core()
    now = datetime.now(timezone.utc)
    private, _public, _source, policy = _test_policy(tmp_path, core)
    output = tmp_path / "runs" / "result"
    authorization, signature = _write_signed_authorization(
        tmp_path,
        core=core,
        private=private,
        policy=policy,
        output_root=output,
        now=now,
        issued=now - timedelta(minutes=4),
        expires=now - timedelta(minutes=2),
    )
    audit = core.PreAuthorizationAudit()
    with pytest.raises(core.ParityV4AuthorizationError, match="expired"):
        core._preauthorize_and_claim_with_policy(
            authorization_path=authorization,
            signature_path=signature,
            output_root=output,
            policy=policy,
            now=now,
            audit=audit,
        )
    _assert_no_data_access(audit)
    assert audit.nonce_claim_writes == 0


def test_nonce_replay_is_atomic_and_never_imports_data_runtime(tmp_path: Path) -> None:
    core = _core()
    now = datetime.now(timezone.utc)
    private, _public, _source, policy = _test_policy(tmp_path, core)
    output = tmp_path / "runs" / "result"
    authorization, signature = _write_signed_authorization(
        tmp_path, core=core, private=private, policy=policy, output_root=output, now=now
    )
    first_audit = core.PreAuthorizationAudit()
    grant = core._preauthorize_and_claim_with_policy(
        authorization_path=authorization,
        signature_path=signature,
        output_root=output,
        policy=policy,
        now=now,
        audit=first_audit,
    )
    assert grant.nonce_claim_path.is_file()
    assert grant.nonce_claim_path.stat().st_mode & 0o777 == 0o444
    assert first_audit.nonce_claim_writes == 1
    _assert_no_data_access(first_audit)
    replay_audit = core.PreAuthorizationAudit()
    with pytest.raises(core.ParityV4AuthorizationError, match="already atomically claimed"):
        core._preauthorize_and_claim_with_policy(
            authorization_path=authorization,
            signature_path=signature,
            output_root=output,
            policy=policy,
            now=now,
            audit=replay_audit,
        )
    _assert_no_data_access(replay_audit)
    assert replay_audit.nonce_claim_writes == 0


def test_source_drift_and_output_collision_fail_before_claim_or_data_access(tmp_path: Path) -> None:
    core = _core()
    now = datetime.now(timezone.utc)

    private, _public, source, policy = _test_policy(tmp_path / "drift", core)
    drift_output = tmp_path / "drift" / "runs" / "result"
    authorization, signature = _write_signed_authorization(
        tmp_path / "drift",
        core=core,
        private=private,
        policy=policy,
        output_root=drift_output,
        now=now,
        nonce="2" * 64,
    )
    source.write_text("PINNED = 'changed'\n", encoding="utf-8")
    drift_audit = core.PreAuthorizationAudit()
    with pytest.raises(core.ParityV4AuthorizationError, match="source drift"):
        core._preauthorize_and_claim_with_policy(
            authorization_path=authorization,
            signature_path=signature,
            output_root=drift_output,
            policy=policy,
            now=now,
            audit=drift_audit,
        )
    _assert_no_data_access(drift_audit)
    assert drift_audit.nonce_claim_writes == 0

    private2, _public2, _source2, policy2 = _test_policy(tmp_path / "collision", core)
    collision_output = tmp_path / "collision" / "runs" / "already-exists"
    collision_output.mkdir(parents=True)
    authorization2, signature2 = _write_signed_authorization(
        tmp_path / "collision",
        core=core,
        private=private2,
        policy=policy2,
        output_root=collision_output,
        now=now,
        nonce="3" * 64,
    )
    collision_audit = core.PreAuthorizationAudit()
    with pytest.raises(core.ParityV4AuthorizationError, match="already exists"):
        core._preauthorize_and_claim_with_policy(
            authorization_path=authorization2,
            signature_path=signature2,
            output_root=collision_output,
            policy=policy2,
            now=now,
            audit=collision_audit,
        )
    _assert_no_data_access(collision_audit)
    assert collision_audit.nonce_claim_writes == 0


def test_production_runner_has_no_caller_policy_or_public_key_override_and_core_defers_v3_import() -> None:
    core = CORE_SOURCE.read_text(encoding="utf-8")
    runner = RUNNER_SOURCE.read_text(encoding="utf-8")
    assert "from sua_exploration.mc_maze.subm_co_scorer_adapter_parity_v3 import" not in runner
    assert "execute_from_fixed_prelaunch(" in runner
    assert "--policy" not in runner and "--public-key" not in runner
    assert "load_stored_prelaunch(prelaunch_dir, root)" in core
    assert "from sua_exploration.mc_maze.subm_co_scorer_adapter_parity_v3 import (" in core
