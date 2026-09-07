"""Static/no-data tests for V3 score-only's V5R2-parity authorization fence."""
from __future__ import annotations

import base64
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


ROOT = Path(__file__).resolve().parents[2]
CORE_SOURCE = ROOT / "sua_exploration/mc_maze/subm_co_score_only_v3.py"
WRITER_SOURCE = ROOT / "sua_exploration/scripts/write_dandi688_subm_co_score_only_prelaunch_v3.py"
RUNNER_SOURCE = ROOT / "sua_exploration/scripts/run_dandi688_subm_co_score_only_v3.py"
V5R2_RUN_ROOT = ROOT / "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_execution_v5r2_runs/run_20260805_072001_root"


def _module(name: str, source: Path):
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _writer():
    return _module("subm_score_only_v3_writer_test", WRITER_SOURCE)


def _core():
    return _module("subm_score_only_v3_core_test", CORE_SOURCE)


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
    source.write_text("PINNED = 'score-v3'\n", encoding="utf-8")
    public_pin = {
        "path": str(public.resolve()),
        "sha256": hashlib.sha256(public.read_bytes()).hexdigest(),
        "bytes": public.stat().st_size,
    }
    source_pin = {str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest()}
    bindings = {
        "prelaunch_bundle": {"draft": "test-only"},
        "score_only_v2_sources": {"test-only": "test-only"},
        "score_only_v2_bundle": {"test-only": "test-only"},
        "frozen_cohort": [{"asset_id": "test-only"}],
        "frozen_matrix": {"N": 15},
        "v5r2_parity_execution": {"receipt": "test-only"},
        "v3_sources": {"test-only": "test-only"},
        "source_snapshot": source_pin,
        "public_key": public_pin,
        "runtime_identity": core.runtime_identity(),
        "cpu_policy": dict(core.CPU_POLICY),
        "claim_boundary": {"same_window_shared_b0_added": False},
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
    external_nwb_root: Path,
    now: datetime,
    bindings: dict | None = None,
    issued: datetime | None = None,
    expires: datetime | None = None,
    nonce: str = "3" * 64,
) -> tuple[Path, Path]:
    body = {
        "schema": core.AUTHORIZATION_SCHEMA,
        "kind": "dandi_000688_subm_co_external_score_only_authorization_v3",
        "status": core.AUTHORIZATION_STATUS,
        "authorization_id": "score-v3-test",
        "single_use_nonce": nonce,
        "issued_at": (issued or now - timedelta(seconds=5)).isoformat(),
        "expires_at": (expires or now + timedelta(seconds=120)).isoformat(),
        "permitted_action": core.PERMITTED_ACTION,
        "output_root": str(output_root.resolve()),
        "external_nwb_root": str(external_nwb_root.resolve()),
        "execution_policy_sha256": policy["execution_policy_sha256"],
        "bindings": bindings if bindings is not None else policy["authorization_bindings"],
        "external_subm_scoring_permitted": True,
        "cpu_forward_r2_only": True,
        "normalizer_fitting_permitted": False,
        "optimizer_or_backward_permitted": False,
        "target_updates_permitted": False,
    }
    raw = core._canonical_bytes({"schema": core.AUTHORIZATION_ENVELOPE_SCHEMA, "authorization": body})
    authorization = tmp_path / f"authorization-{nonce}.json"
    signature = tmp_path / f"authorization-{nonce}.sig"
    authorization.write_bytes(raw)
    signature.write_bytes(base64.b64encode(private.sign(raw)))
    return authorization, signature


def _assert_zero_data_or_runtime(audit) -> None:
    assert audit.zero_data_or_runtime_access()
    assert audit.v5_bridge_runtime_imports == 0
    assert audit.score_runtime_imports == 0
    assert audit.external_nwb_files_opened == 0
    assert audit.checkpoint_files_opened == 0
    assert audit.normalizer_files_opened == 0
    assert audit.model_forward_calls == 0
    assert audit.r2_computations == 0


def _parity_payloads() -> tuple[dict, dict, dict, dict]:
    return tuple(
        json.loads((V5R2_RUN_ROOT / name).read_text(encoding="utf-8"))
        for name in ("parity_execution_receipt.json", "input_trace.json", "environment.json", "seal.json")
    )


def test_static_authority_pins_semantic_v5r2_success_v2_matrix_and_claim_boundary(tmp_path: Path) -> None:
    writer = _writer()
    authority = writer.static_authority(ROOT)
    parity = authority["v5r2_parity_execution"]
    assert parity["artifacts"]["parity_execution_receipt.json"]["sha256"] == (
        "faace6493fcf939e87bf0f5ad219f75df739434b7941fccc7f654c23d78c63e8"
    )
    assert parity["artifacts"]["input_trace.json"]["sha256"] == (
        "c3309f4d9e60de96517ea9c2cfad5d1f89c28398f3644cbb9a379d17779ae859"
    )
    assert parity["artifacts"]["environment.json"]["sha256"] == (
        "cf2bf2010ba326a3445d409902cb3343c05380ae5b29825d250a997d06345f02"
    )
    assert parity["artifacts"]["seal.json"]["sha256"] == (
        "b5f92c71a0e601fcdbd8866d4d7e1ec882ce9f285936431bfbe27aa7e9a03e18"
    )
    assert parity["r2"] == 0.4910046458244324
    assert parity["prediction_target_exact"] is True
    assert parity["input_exact"] is True
    assert parity["chronology_unchanged"] is True
    assert parity["external_subm_scoring_performed"] is False
    assert authority["score_only_v2"]["sources"] == writer.V2_SOURCES
    assert authority["source_snapshot"][
        str((ROOT / "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v5.py").resolve())
    ] == writer.EXPECTED_V5_BRIDGE_SHA256
    matrix = authority["score_only_v2"]["matrix"]
    assert matrix["N"] == 15
    assert matrix["arms"] == ["shared_t4", "shared_ts4"]
    assert matrix["seeds"] == [42, 43, 44]
    assert matrix["views"] == ["sua", "pseudo_mua"]
    assert authority["claim_boundary"] == {
        "frozen_arms_only": ["shared_t4", "shared_ts4"],
        "supports": "correct-vs-shuffled attachment/content contrast only",
        "absolute_t4_over_spint_claim": "UNSUPPORTED_NO_SHARED_B0_CONTROL",
        "same_window_shared_b0_added": False,
    }

    output = tmp_path / "v3-prelaunch"
    result = writer.write_prelaunch(output, ROOT)
    assert result["status"] == "STATIC_V3_SCORE_ONLY_PACKAGE_NOT_AUTHORIZED"
    stored = writer.load_stored_prelaunch(output, ROOT)
    bindings = stored["execution_policy"]["authorization_bindings"]
    assert bindings["v5r2_parity_execution"] == parity
    assert bindings["frozen_matrix"] == matrix
    for path in output.iterdir():
        assert path.stat().st_mode & 0o777 == 0o444


@pytest.mark.parametrize("mutation", ("r2", "input_exact", "chronology"))
def test_v5r2_semantic_drift_fails_closed(mutation: str) -> None:
    writer = _writer()
    receipt, input_trace, environment, seal = map(copy.deepcopy, _parity_payloads())
    if mutation == "r2":
        receipt["reference"]["r2"] = 0.0
    elif mutation == "input_exact":
        input_trace["shared_observer"]["input_exact"] = False
    else:
        input_trace["adapter_schema_bridge"]["raw_owner_chronology_sha256_after"] = "0" * 64
    with pytest.raises(writer.StaticScoreV3Error):
        writer._verify_v5r2_success_semantics(receipt, input_trace, environment, seal)


def test_invalid_signature_has_zero_external_data_runtime_or_nonce_access(tmp_path: Path) -> None:
    core = _core()
    now = datetime.now(timezone.utc)
    private, policy = _test_policy(tmp_path, core)
    output, external = tmp_path / "runs" / "result", tmp_path / "external-nwb"
    authorization, signature = _write_signed_authorization(
        tmp_path, core=core, private=private, policy=policy, output_root=output,
        external_nwb_root=external, now=now,
    )
    signature.write_bytes(base64.b64encode(b"x" * 64))
    audit = core.PreAuthorizationAudit()
    with pytest.raises(core.ScoreV3AuthorizationError, match="invalid detached"):
        core._preauthorize_and_claim_with_policy(
            authorization_path=authorization, signature_path=signature, output_root=output,
            external_nwb_root=external, policy=policy, now=now, audit=audit,
        )
    _assert_zero_data_or_runtime(audit)
    assert audit.nonce_claim_writes == 0
    assert not Path(policy["claim_root"]).exists()


def test_expired_authorization_has_zero_external_data_runtime_or_nonce_access(tmp_path: Path) -> None:
    core = _core()
    now = datetime.now(timezone.utc)
    private, policy = _test_policy(tmp_path, core)
    output, external = tmp_path / "runs" / "expired", tmp_path / "external-nwb"
    authorization, signature = _write_signed_authorization(
        tmp_path, core=core, private=private, policy=policy, output_root=output,
        external_nwb_root=external, now=now, nonce="4" * 64,
        issued=now - timedelta(minutes=4), expires=now - timedelta(minutes=2),
    )
    audit = core.PreAuthorizationAudit()
    with pytest.raises(core.ScoreV3AuthorizationError, match="expired"):
        core._preauthorize_and_claim_with_policy(
            authorization_path=authorization, signature_path=signature, output_root=output,
            external_nwb_root=external, policy=policy, now=now, audit=audit,
        )
    _assert_zero_data_or_runtime(audit)
    assert audit.nonce_claim_writes == 0
    assert not Path(policy["claim_root"]).exists()


def test_binding_drift_has_zero_external_data_runtime_or_nonce_access(tmp_path: Path) -> None:
    core = _core()
    now = datetime.now(timezone.utc)
    private, policy = _test_policy(tmp_path, core)
    bindings = copy.deepcopy(policy["authorization_bindings"])
    bindings["v5r2_parity_execution"]["receipt"] = "changed"
    output, external = tmp_path / "runs" / "binding", tmp_path / "external-nwb"
    authorization, signature = _write_signed_authorization(
        tmp_path, core=core, private=private, policy=policy, output_root=output,
        external_nwb_root=external, now=now, bindings=bindings, nonce="5" * 64,
    )
    audit = core.PreAuthorizationAudit()
    with pytest.raises(core.ScoreV3AuthorizationError, match="binding drift"):
        core._preauthorize_and_claim_with_policy(
            authorization_path=authorization, signature_path=signature, output_root=output,
            external_nwb_root=external, policy=policy, now=now, audit=audit,
        )
    _assert_zero_data_or_runtime(audit)
    assert audit.nonce_claim_writes == 0
    assert not Path(policy["claim_root"]).exists()


def test_runner_has_fixed_policy_key_boundary_and_v5_bridge_owner_path_only() -> None:
    core = CORE_SOURCE.read_text(encoding="utf-8")
    runner = RUNNER_SOURCE.read_text(encoding="utf-8")
    protocol = (ROOT / "sua_exploration/docs/DANDI_000688_SUBM_CO_SCORE_ONLY_PROTOCOL_V3.md").read_text(encoding="utf-8")
    assert "--policy" not in runner and "--public-key" not in runner
    assert "--external-nwb-root" in runner
    assert "load_stored_prelaunch(prelaunch_dir, root)" in core
    assert "bridge_owner_chronology_for_c1_builder" in core
    assert "load_session_with_trials" not in core
    assert "cpu_forward_r2_only" in core
    assert "same-window shared-B0 control" in protocol
    assert "absolute “T4 over SPINT” claim" in protocol
