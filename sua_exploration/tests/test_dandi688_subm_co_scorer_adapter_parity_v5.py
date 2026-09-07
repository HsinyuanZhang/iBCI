"""Static and schema-bridge tests for the append-only v5 parity package.

These tests never call the data fixture or concrete parity entrypoint.  The
only generated key pair stays in test-process memory; its public component is
written to pytest's temporary directory solely to exercise the pre-import v5
authorization gate.
"""
from __future__ import annotations

import base64
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


ROOT = Path(__file__).resolve().parents[2]
HELPER_SOURCE = ROOT / "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v5.py"
CORE_SOURCE = ROOT / "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_execution_v5.py"
WRITER_SOURCE = ROOT / "sua_exploration/scripts/write_dandi688_subm_co_scorer_adapter_parity_prelaunch_v5.py"
RUNNER_SOURCE = ROOT / "sua_exploration/scripts/run_dandi688_subm_co_scorer_adapter_parity_v5.py"


def _module(name: str, source: Path):
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _helper():
    return _module("subm_parity_v5_helper_test", HELPER_SOURCE)


def _core():
    return _module("subm_parity_v5_core_test", CORE_SOURCE)


def _writer():
    return _module("subm_parity_v5_writer_test", WRITER_SOURCE)


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
    source.write_text("PINNED = 'v5'\n", encoding="utf-8")
    output_parent = tmp_path / "runs"
    public_pin = {
        "path": str(public.resolve()),
        "sha256": hashlib.sha256(public.read_bytes()).hexdigest(),
        "bytes": public.stat().st_size,
    }
    source_pin = {str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest()}
    bindings = {
        "prelaunch_bundle": {"draft": "test-only"},
        "v3_sources": {"test-only": "test-only"},
        "v3_bundle": {"test-only": "test-only"},
        "v4_sources": {"test-only": "test-only"},
        "v4_bundle": {"test-only": "test-only"},
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
        "output_parent": str(output_parent.resolve()),
        "claim_root": str((tmp_path / "nonce-claims").resolve()),
        "authorization_bindings": bindings,
    }
    policy["execution_policy_sha256"] = hashlib.sha256(
        core._canonical_bytes(policy)
    ).hexdigest()
    return private, policy


def _write_signed_authorization(
    tmp_path: Path,
    *,
    core,
    private: Ed25519PrivateKey,
    policy: dict,
    output_root: Path,
    now: datetime,
    nonce: str = "5" * 64,
) -> tuple[Path, Path]:
    body = {
        "schema": core.AUTHORIZATION_SCHEMA,
        "kind": "dandi_000688_subc_cpu_schema_bridge_parity_execution",
        "status": core.AUTHORIZATION_STATUS,
        "authorization_id": "parity-v5-test",
        "single_use_nonce": nonce,
        "issued_at": (now - timedelta(seconds=5)).isoformat(),
        "expires_at": (now + timedelta(seconds=120)).isoformat(),
        "permitted_action": core.PERMITTED_ACTION,
        "output_root": str(output_root.resolve()),
        "execution_policy_sha256": policy["execution_policy_sha256"],
        "bindings": policy["authorization_bindings"],
        "external_subm_scoring_permitted": False,
        "normalizer_fitting_permitted": False,
        "optimizer_or_backward_permitted": False,
    }
    envelope = {"schema": core.AUTHORIZATION_ENVELOPE_SCHEMA, "authorization": body}
    raw = core._canonical_bytes(envelope)
    authorization = tmp_path / "authorization.json"
    signature = tmp_path / "authorization.sig"
    authorization.write_bytes(raw)
    signature.write_bytes(base64.b64encode(private.sign(raw)))
    return authorization, signature


def _assert_zero_data_access(audit) -> None:
    assert audit.zero_data_access()
    assert audit.corrected_helper_imports == 0
    assert audit.v3_torch_owner_imports == 0
    assert audit.checkpoint_files_opened == 0
    assert audit.nwb_files_opened == 0
    assert audit.npz_files_opened == 0
    assert audit.model_forward_calls == 0


def test_integral_float_bridge_builds_independent_int_slice_copy_and_keeps_raw_evidence() -> None:
    helper = _helper()
    raw_trials = [
        {
            "start": 10.0,
            "stop": np.float64(60.0),
            "trial_id": "R-1",
            "metadata": {"result": "R", "epoch": 1},
        },
        {
            "start": np.float64(61.0),
            "stop": 100.0,
            "trial_id": "R-2",
            "metadata": {"result": "R", "epoch": 2},
        },
    ]
    original = copy.deepcopy(raw_trials)

    raw_evidence, builder_trials, trace = helper.bridge_owner_chronology_for_c1_builder(raw_trials)

    assert raw_trials == original
    assert raw_evidence == original
    assert raw_evidence is not raw_trials
    assert trace["raw_owner_chronology_sha256_before"] == trace["raw_owner_chronology_sha256_after"]
    assert trace["only_start_stop_cast"] is True
    assert trace["bin_recomputation"] is False
    assert trace["trial_selection_changed_by_schema_bridge"] is False
    assert trace["t4_changed_by_schema_bridge"] is False
    assert trace["query_valid_starts_changed_by_schema_bridge"] is False
    for raw, builder in zip(raw_trials, builder_trials, strict=True):
        assert builder is not raw
        assert type(builder["start"]) is int
        assert type(builder["stop"]) is int
        assert set(builder) == set(raw)
        for key in raw:
            if key not in {"start", "stop"}:
                assert builder[key] == raw[key]
                assert builder[key] is raw[key]
    assert [row["builder_start"] for row in trace["start_stop_conversions"]] == [10, 61]
    assert [row["builder_stop"] for row in trace["start_stop_conversions"]] == [60, 100]


@pytest.mark.parametrize(
    ("bad_value", "message"),
    [
        (10.5, "exactly integral"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (float("-inf"), "finite"),
        (float(2**63), "outside signed int64 range"),
        (True, "numeric scalar"),
        ("10", "numeric scalar"),
    ],
)
def test_bridge_rejects_non_exact_or_unsafe_slice_coordinates(bad_value, message: str) -> None:
    helper = _helper()
    raw_trials = [{"start": bad_value, "stop": 60.0, "trial_id": "unsafe"}]
    before = copy.deepcopy(raw_trials)

    with pytest.raises(helper.SchemaBridgeError, match=message):
        helper.bridge_owner_chronology_for_c1_builder(raw_trials)

    assert raw_trials == before


def test_true_integer_values_never_take_a_float_precision_path() -> None:
    helper = _helper()
    exact_values = [
        helper.INT64_MIN,
        -(2**53) - 1,
        -(2**53),
        -(2**53) + 1,
        (2**53) - 1,
        2**53,
        (2**53) + 1,
        helper.INT64_MAX,
        np.int64(helper.INT64_MIN),
        np.int64(helper.INT64_MAX),
    ]
    for value in exact_values:
        assert helper._checked_slice_index(value, field="start", trial_position=0) == int(value)
    assert helper._checked_slice_index(float(2**53), field="stop", trial_position=0) == 2**53


def test_static_prelaunch_binds_v3_v4_incident_dedicated_key_and_schema_constraints(
    tmp_path: Path,
) -> None:
    writer = _writer()
    authority = writer.static_authority(ROOT)
    assert authority["public_key"] == {
        "path": str(
            (ROOT / "sua_exploration/configs/dandi688_subc_parity_v4_root_ed25519_public.pem").resolve()
        ),
        "sha256": "a541b5aabc7922251e797da13727b97ceb4ed48de7e95d5b1c4031ba96c602a9",
        "bytes": 113,
    }
    assert authority["v4_incident"] == {
        "path": "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_execution_incident_v4_20260805_0654/receipt.json",
        "sha256": "4c676146c5855f91591aaa13fc1d0b312ebbb51138584bd39c02c2888f47d0bc",
        "status": "FAILED_CLOSED_AFTER_AUTHORIZATION_BEFORE_MODEL_FORWARD_SCHEMA_TYPE_MISMATCH",
    }
    bridge = authority["fixture_pins"]["schema_bridge"]
    assert bridge["only_cast_fields"] == ["start", "stop"]
    assert bridge["finite_exact_integer_int64_safe"] is True
    assert bridge["rebinning"] is False
    assert bridge["selection_changed_by_bridge"] is False
    assert bridge["t4_changed_by_bridge"] is False
    assert bridge["query_valid_starts_changed_by_bridge"] is False

    output = tmp_path / "v5-prelaunch"
    result = writer.write_prelaunch(output, ROOT)
    assert result["status"] == "STATIC_V5_SCHEMA_BRIDGE_CPU_PARITY_EXECUTION_PACKAGE_NOT_AUTHORIZED"
    stored = writer.load_stored_prelaunch(output, ROOT)
    assert stored["execution_policy"]["execution_policy_sha256"] == stored["execution_policy_sha256"]
    assert stored["execution_policy"]["authorization_bindings"]["v4_incident"] == authority["v4_incident"]
    for artifact in output.iterdir():
        assert artifact.stat().st_mode & 0o777 == 0o444


def test_invalid_authorization_has_zero_data_access_and_no_nonce_claim(tmp_path: Path) -> None:
    core = _core()
    now = datetime.now(timezone.utc)
    private, policy = _test_policy(tmp_path, core)
    output = tmp_path / "runs" / "result"
    authorization, signature = _write_signed_authorization(
        tmp_path,
        core=core,
        private=private,
        policy=policy,
        output_root=output,
        now=now,
    )
    signature.write_bytes(base64.b64encode(b"x" * 64))
    audit = core.PreAuthorizationAudit()

    with pytest.raises(core.ParityV5AuthorizationError, match="invalid detached"):
        core._preauthorize_and_claim_with_policy(
            authorization_path=authorization,
            signature_path=signature,
            output_root=output,
            policy=policy,
            now=now,
            audit=audit,
        )

    _assert_zero_data_access(audit)
    assert audit.nonce_claim_writes == 0
    assert audit.source_hash_checks == 0
    assert not Path(policy["claim_root"]).exists()


def test_production_runner_has_no_caller_policy_or_key_and_core_defers_helper_import() -> None:
    core = CORE_SOURCE.read_text(encoding="utf-8")
    runner = RUNNER_SOURCE.read_text(encoding="utf-8")
    assert "subm_co_scorer_adapter_parity_v5 import" not in runner
    assert "execute_from_fixed_prelaunch(" in runner
    assert "--policy" not in runner and "--public-key" not in runner
    assert "load_stored_prelaunch(prelaunch_dir, root)" in core
    assert "from sua_exploration.mc_maze.subm_co_scorer_adapter_parity_v5 import (" in core
