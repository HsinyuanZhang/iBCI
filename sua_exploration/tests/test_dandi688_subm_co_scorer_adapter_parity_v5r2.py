"""No-data tests for V5R2's explicit, fail-closed V5-source closure."""
from __future__ import annotations

import base64
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


ROOT = Path(__file__).resolve().parents[2]
CORE_SOURCE = ROOT / "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_execution_v5r2.py"
WRITER_SOURCE = ROOT / "sua_exploration/scripts/write_dandi688_subm_co_scorer_adapter_parity_prelaunch_v5r2.py"
RUNNER_SOURCE = ROOT / "sua_exploration/scripts/run_dandi688_subm_co_scorer_adapter_parity_v5r2.py"


def _module(name: str, source: Path):
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _writer():
    return _module("subm_parity_v5r2_writer_test", WRITER_SOURCE)


def _core():
    return _module("subm_parity_v5r2_core_test", CORE_SOURCE)


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
    source.write_text("PINNED = 'v5r2'\n", encoding="utf-8")
    public_pin = {
        "path": str(public.resolve()),
        "sha256": hashlib.sha256(public.read_bytes()).hexdigest(),
        "bytes": public.stat().st_size,
    }
    source_pin = {str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest()}
    v5_sources = {f"v5-source-{index}": f"{index:064x}" for index in range(6)}
    bindings = {
        "prelaunch_bundle": {"draft": "test-only"},
        "v3_sources": {"test-only": "test-only"},
        "v3_bundle": {"test-only": "test-only"},
        "v4_sources": {"test-only": "test-only"},
        "v4_bundle": {"test-only": "test-only"},
        "v5_sources": v5_sources,
        "v5_bundle": {"test-only": "test-only"},
        "v5_execution_policy_sha256": "f" * 64,
        "v5r2_sources": {"test-only": "test-only"},
        "v4_incident": {"test-only": "test-only"},
        "fixture_pins": {"test-only": "test-only"},
        "runtime_identity": core.runtime_identity(),
        "cpu_policy": dict(core.CPU_POLICY),
        "public_key": public_pin,
        "source_snapshot": source_pin,
    }
    policy = {
        "public_key": public_pin,
        "source_pins": source_pin,
        "runtime_identity": core.runtime_identity(),
        "cpu_policy": dict(core.CPU_POLICY),
        "output_parent": str((tmp_path / "runs").resolve()),
        "claim_root": str((tmp_path / "nonce-claims").resolve()),
        "authorization_bindings": bindings,
    }
    policy["execution_policy_sha256"] = hashlib.sha256(core._canonical_bytes(policy)).hexdigest()
    return private, policy


def _write_signed_authorization(
    tmp_path: Path,
    *,
    core,
    private: Ed25519PrivateKey,
    policy: dict,
    output_root: Path,
    now: datetime,
    bindings: dict | None = None,
    nonce: str = "2" * 64,
) -> tuple[Path, Path]:
    body = {
        "schema": core.AUTHORIZATION_SCHEMA,
        "kind": "dandi_000688_subc_cpu_explicit_v5_source_closure_parity_execution",
        "status": core.AUTHORIZATION_STATUS,
        "authorization_id": "parity-v5r2-test",
        "single_use_nonce": nonce,
        "issued_at": (now - timedelta(seconds=5)).isoformat(),
        "expires_at": (now + timedelta(seconds=120)).isoformat(),
        "permitted_action": core.PERMITTED_ACTION,
        "output_root": str(output_root.resolve()),
        "execution_policy_sha256": policy["execution_policy_sha256"],
        "bindings": bindings if bindings is not None else policy["authorization_bindings"],
        "external_subm_scoring_permitted": False,
        "normalizer_fitting_permitted": False,
        "optimizer_or_backward_permitted": False,
    }
    raw = core._canonical_bytes(
        {"schema": core.AUTHORIZATION_ENVELOPE_SCHEMA, "authorization": body}
    )
    authorization = tmp_path / f"authorization-{nonce}.json"
    signature = tmp_path / f"authorization-{nonce}.sig"
    authorization.write_bytes(raw)
    signature.write_bytes(base64.b64encode(private.sign(raw)))
    return authorization, signature


def _assert_zero_data_or_runtime(audit) -> None:
    assert audit.zero_data_access()
    assert audit.v5r2_helper_imports == 0
    assert audit.v5_v3_torch_owner_imports == 0
    assert audit.checkpoint_files_opened == 0
    assert audit.nwb_files_opened == 0
    assert audit.npz_files_opened == 0
    assert audit.model_forward_calls == 0


def test_static_authority_returns_and_binds_all_six_v5_paths_with_consistent_snapshot(
    tmp_path: Path,
) -> None:
    writer = _writer()
    authority = writer.static_authority(ROOT)
    assert authority["v5_sources"] == writer.V5_SOURCES
    assert len(authority["v5_sources"]) == 6
    assert len(authority["v5r2_sources"]) == 6
    for relative, digest in authority["v5_sources"].items():
        assert authority["source_snapshot"][str((ROOT / relative).resolve())] == digest
    writer._require_explicit_v5_closure(authority, ROOT)

    output = tmp_path / "v5r2-prelaunch"
    result = writer.write_prelaunch(output, ROOT)
    assert result["status"] == "STATIC_V5R2_EXPLICIT_V5_SOURCE_CLOSURE_PACKAGE_NOT_AUTHORIZED"
    stored = writer.load_stored_prelaunch(output, ROOT)
    bindings = stored["execution_policy"]["authorization_bindings"]
    assert bindings["v5_sources"] == writer.V5_SOURCES
    assert bindings["source_snapshot"] == authority["source_snapshot"]
    assert bindings["v5_execution_policy_sha256"] == writer.V5_EXECUTION_POLICY_SHA256
    for artifact in output.iterdir():
        assert artifact.stat().st_mode & 0o777 == 0o444


@pytest.mark.parametrize("mutation", ("missing", "changed"))
def test_missing_or_changed_v5_source_map_fails_closed_before_stored_authority_is_accepted(
    mutation: str,
) -> None:
    writer = _writer()
    authority = copy.deepcopy(writer.static_authority(ROOT))
    if mutation == "missing":
        authority["v5_sources"].pop(next(iter(authority["v5_sources"])))
    else:
        first = next(iter(authority["v5_sources"]))
        authority["v5_sources"][first] = "0" * 64
    with pytest.raises(writer.StaticParityV5R2Error, match="explicit V5 source map"):
        writer._require_explicit_v5_closure(authority, ROOT)


@pytest.mark.parametrize("mutation", ("missing", "changed"))
def test_missing_or_changed_v5_entry_in_combined_snapshot_fails_closed(mutation: str) -> None:
    writer = _writer()
    authority = copy.deepcopy(writer.static_authority(ROOT))
    first = next(iter(authority["v5_sources"]))
    absolute = str((ROOT / first).resolve())
    if mutation == "missing":
        authority["source_snapshot"].pop(absolute)
    else:
        authority["source_snapshot"][absolute] = "b" * 64
    with pytest.raises(writer.StaticParityV5R2Error, match="combined snapshot V5 mismatch"):
        writer._require_explicit_v5_closure(authority, ROOT)


def test_invalid_signature_has_zero_data_runtime_and_nonce_access(tmp_path: Path) -> None:
    core = _core()
    now = datetime.now(timezone.utc)
    private, policy = _test_policy(tmp_path, core)
    output = tmp_path / "runs" / "result"
    authorization, signature = _write_signed_authorization(
        tmp_path, core=core, private=private, policy=policy, output_root=output, now=now
    )
    signature.write_bytes(base64.b64encode(b"x" * 64))
    audit = core.PreAuthorizationAudit()
    with pytest.raises(core.ParityV5R2AuthorizationError, match="invalid detached"):
        core._preauthorize_and_claim_with_policy(
            authorization_path=authorization, signature_path=signature, output_root=output,
            policy=policy, now=now, audit=audit,
        )
    _assert_zero_data_or_runtime(audit)
    assert audit.nonce_claim_writes == 0
    assert not Path(policy["claim_root"]).exists()


@pytest.mark.parametrize("mutation", ("missing", "changed"))
def test_invalid_explicit_v5_source_binding_has_zero_runtime_or_nonce_access(
    tmp_path: Path, mutation: str
) -> None:
    core = _core()
    now = datetime.now(timezone.utc)
    private, policy = _test_policy(tmp_path, core)
    bindings = copy.deepcopy(policy["authorization_bindings"])
    if mutation == "missing":
        bindings["v5_sources"].pop(next(iter(bindings["v5_sources"])))
    else:
        first = next(iter(bindings["v5_sources"]))
        bindings["v5_sources"][first] = "a" * 64
    output = tmp_path / "runs" / f"result-{mutation}"
    authorization, signature = _write_signed_authorization(
        tmp_path, core=core, private=private, policy=policy, output_root=output,
        now=now, bindings=bindings, nonce=("3" if mutation == "missing" else "4") * 64,
    )
    audit = core.PreAuthorizationAudit()
    with pytest.raises(core.ParityV5R2AuthorizationError, match="binding drift"):
        core._preauthorize_and_claim_with_policy(
            authorization_path=authorization, signature_path=signature, output_root=output,
            policy=policy, now=now, audit=audit,
        )
    _assert_zero_data_or_runtime(audit)
    assert audit.nonce_claim_writes == 0
    assert not Path(policy["claim_root"]).exists()


def test_runner_has_no_caller_policy_or_key_and_core_defers_v5r2_v5_imports() -> None:
    core = CORE_SOURCE.read_text(encoding="utf-8")
    runner = RUNNER_SOURCE.read_text(encoding="utf-8")
    assert "subm_co_scorer_adapter_parity_v5r2 import" not in runner
    assert "subm_co_scorer_adapter_parity_v5 import" not in runner
    assert "execute_from_fixed_prelaunch(" in runner
    assert "--policy" not in runner and "--public-key" not in runner
    assert "load_stored_prelaunch(prelaunch_dir, root)" in core
    assert "from sua_exploration.mc_maze.subm_co_scorer_adapter_parity_v5r2 import (" in core
