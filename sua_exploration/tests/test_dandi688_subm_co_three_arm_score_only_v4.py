from __future__ import annotations

import base64
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sua_exploration.mc_maze import subm_co_three_arm_score_only_v4 as core
from sua_exploration.scripts import run_dandi688_subm_co_three_arm_score_only_v4 as runner
from sua_exploration.scripts import (
    write_dandi688_subm_co_three_arm_score_only_prelaunch_v4 as writer,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def authority() -> dict[str, object]:
    return writer.static_authority(ROOT)


@pytest.fixture(scope="module")
def contract(authority: dict[str, object]) -> dict[str, object]:
    return authority["contract"]  # type: ignore[return-value]


def _artifact_pin() -> dict[str, object]:
    return {"path": "arrays/example.npz", "sha256": "a" * 64, "bytes": 100, "mode": "0444"}


def _complete_slots() -> list[dict[str, object]]:
    rows = copy.deepcopy(core.checkpoint_slots_v4())
    for row in rows:
        if row["arm"] == "shared_zero4":
            row["sha256"] = str(row["seed"])[-1] * 64
            row["status"] = "SYNTHETIC_COMPLETE_POLICY_TEST_ONLY"
    core.validate_checkpoint_slots(rows, require_complete=True)
    return rows


def _policy(tmp_path: Path) -> dict[str, object]:
    output_parent = tmp_path / "outputs"
    external_parent = tmp_path / "external"
    output_parent.mkdir()
    external_parent.mkdir()
    return {
        "contract_sha256": "1" * 64,
        "cohort_sha256": "2" * 64,
        "checkpoint_slots_sha256": "3" * 64,
        "parity_bundle_sha256": "4" * 64,
        "source_snapshot_sha256": "5" * 64,
        "runtime_identity": core.runtime_identity(),
        "cpu_policy": dict(core.CPU_POLICY),
        "checkpoint_slots": _complete_slots(),
        "output_parent": str(output_parent),
        "external_root_parent": str(external_parent),
        "claim_root": str(tmp_path / "claims"),
    }


def _authorization(
    policy: dict[str, object], *, output_root: Path, external_root: Path,
    now: datetime, nonce: str = "6" * 64,
) -> dict[str, object]:
    return {
        "schema": core.AUTHORIZATION_SCHEMA,
        "status": core.AUTHORIZATION_STATUS,
        "permitted_action": core.PERMITTED_ACTION,
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(minutes=1)).isoformat(),
        "nonce": nonce,
        "output_root": str(output_root.resolve()),
        "external_nwb_root": str(external_root.resolve()),
        "cell_count": 270,
        "bindings": core._authorization_bindings(policy),
    }


def _write_auth_pair(
    tmp_path: Path, authorization: dict[str, object], *, signature_byte: bytes = b"x"
) -> tuple[Path, Path]:
    raw = core.canonical_bytes(authorization)
    auth_path = tmp_path / "authorization.json"
    signature_path = tmp_path / "authorization.sig.json"
    auth_path.write_bytes(raw)
    envelope = {
        "schema": core.AUTHORIZATION_ENVELOPE_SCHEMA,
        "algorithm": "Ed25519",
        "authorization_sha256": hashlib.sha256(raw).hexdigest(),
        "signature_b64": base64.b64encode(signature_byte * 64).decode("ascii"),
    }
    signature_path.write_bytes(core.canonical_bytes(envelope))
    return auth_path, signature_path


def test_contract_is_exact_270_cell_matrix_with_separate_claims(
    contract: dict[str, object],
) -> None:
    assert contract["N"] == 15
    assert contract["views"] == ["sua", "pseudo_mua"]
    assert contract["seeds"] == [42, 43, 44]
    assert contract["arms"] == ["shared_t4", "shared_zero4", "shared_ts4"]
    assert contract["cell_count"] == 270
    assert len(core.expected_cell_keys(contract)) == 270
    assert set(contract["comparison_gates"]) == {"t4_minus_zero4", "t4_minus_ts4"}
    assert contract["comparison_gates"]["t4_minus_zero4"]["grand_paired_mean_minimum_r2"] == 0.03
    assert contract["comparison_gates"]["t4_minus_ts4"]["grand_paired_mean_minimum_r2"] == 0.03
    assert contract["claim_separation"]["cross_claim_substitution"] == "FORBIDDEN"


def test_checkpoint_table_has_nine_slots_and_exact_zero4_blocker(
    authority: dict[str, object],
) -> None:
    slots = authority["checkpoint_slots"]
    assert len(slots) == 9
    missing = authority["missing_checkpoint_slots"]
    assert [(row["arm"], row["seed"]) for row in missing] == [
        ("shared_zero4", 42),
        ("shared_zero4", 43),
        ("shared_zero4", 44),
    ]
    with pytest.raises(core.ThreeArmV4BlockedError, match="BLOCKED_MISSING_ZERO4_TERMINALS"):
        core.validate_checkpoint_slots(slots, require_complete=True)


@pytest.mark.parametrize(
    "field,value",
    (
        ("arm", "shared_bogus"),
        ("seed", 45),
        ("view", "combined"),
        ("session_id", "sub-X_ses-invalid"),
        ("asset_id", "00000000-0000-0000-0000-000000000000"),
    ),
)
def test_wrong_arm_seed_view_or_cohort_cell_fails_closed(
    contract: dict[str, object], field: str, value: object,
) -> None:
    row = core.expected_cell_keys(contract)[0].as_dict()
    row[field] = value
    key = core.CellKey(**row)
    with pytest.raises(core.ThreeArmV4Error, match="wrong arm/seed/view/cohort"):
        core.validate_cell_key(key, contract)


def test_cohort_mutation_breaks_exact_contract_hash(contract: dict[str, object]) -> None:
    mutated = copy.deepcopy(contract)
    mutated["cohort"][0]["session_id"] = "sub-M_ses-mutated"
    with pytest.raises(core.ThreeArmV4Error, match="content hash drift"):
        core.validate_contract_exact(mutated, contract["contract_sha256"])


def test_known_checkpoint_hash_or_path_tamper_fails_closed() -> None:
    slots = copy.deepcopy(core.checkpoint_slots_v4())
    slots[0]["sha256"] = "f" * 64
    with pytest.raises(core.ThreeArmV4Error, match="known checkpoint hash drift"):
        core.validate_checkpoint_slots(slots, require_complete=False)
    slots = copy.deepcopy(core.checkpoint_slots_v4())
    slots[0]["path"] = "wrong.ckpt"
    with pytest.raises(core.ThreeArmV4Error, match="path drift"):
        core.validate_checkpoint_slots(slots, require_complete=False)


def test_nonzero_negative_zero_wrong_shape_and_dtype_zero4_fail_closed() -> None:
    core.validate_direct_zero4_buffer(
        channel_count=2, dtype="float32", shape=[2, 4], raw_bytes=b"\0" * 32
    )
    with pytest.raises(core.ThreeArmV4AuthorizationError, match="nonzero"):
        core.validate_direct_zero4_buffer(
            channel_count=2, dtype="float32", shape=[2, 4], raw_bytes=b"\0" * 31 + b"\1"
        )
    with pytest.raises(core.ThreeArmV4AuthorizationError, match="nonzero"):
        core.validate_direct_zero4_buffer(
            channel_count=1,
            dtype="float32",
            shape=[1, 4],
            raw_bytes=(b"\0\0\0\x80") + b"\0" * 12,
        )
    with pytest.raises(core.ThreeArmV4AuthorizationError, match="dtype"):
        core.validate_direct_zero4_buffer(
            channel_count=2, dtype="float64", shape=[2, 4], raw_bytes=b"\0" * 32
        )
    with pytest.raises(core.ThreeArmV4AuthorizationError, match="shape"):
        core.validate_direct_zero4_buffer(
            channel_count=2, dtype="float32", shape=[2, 3], raw_bytes=b"\0" * 32
        )


@pytest.mark.parametrize("bundle", ("v5r2", "zero4"))
def test_parity_artifact_drift_fails_static_authority_before_runtime(
    monkeypatch: pytest.MonkeyPatch, bundle: str,
) -> None:
    table = writer.V5R2_ARTIFACTS if bundle == "v5r2" else writer.ZERO4_PARITY_ARTIFACTS
    name = next(iter(table))
    monkeypatch.setitem(table[name], "sha256", "0" * 64)
    with pytest.raises(writer.StaticThreeArmV4Error, match="SHA drift"):
        writer.static_authority(ROOT)


def test_adapter_source_drift_fails_static_authority_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = next(iter(writer.ADAPTER_SOURCE_PINS))
    monkeypatch.setitem(writer.ADAPTER_SOURCE_PINS, relative, "0" * 64)
    with pytest.raises(writer.StaticThreeArmV4Error, match="adapter source.*drift"):
        writer.static_authority(ROOT)


def test_current_incomplete_policy_rejects_before_auth_signature_or_runtime_access(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    policy["checkpoint_slots"] = core.checkpoint_slots_v4()
    audit = core.PreAuthorizationAudit()
    with pytest.raises(core.ThreeArmV4BlockedError, match="BLOCKED_MISSING_ZERO4_TERMINALS"):
        core.preauthorize_and_claim_with_policy(
            authorization_path=tmp_path / "must-not-open-auth",
            signature_path=tmp_path / "must-not-open-signature",
            output_root=tmp_path / "outputs" / "run",
            external_nwb_root=tmp_path / "external" / "root",
            policy=policy,
            now=datetime.now(timezone.utc),
            audit=audit,
        )
    assert audit.authorization_files_read == 0
    assert audit.signature_files_read == 0
    assert audit.nonce_claim_writes == 0
    assert audit.data_runtime_zero()


def test_authorization_content_tamper_fails_before_nonce_and_runtime(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    policy = _policy(tmp_path)
    output = tmp_path / "outputs" / "run"
    external = tmp_path / "external" / "root"
    authorization = _authorization(policy, output_root=output, external_root=external, now=now)
    authorization["schema"] = "tampered"
    auth_path, signature_path = _write_auth_pair(tmp_path, authorization)
    audit = core.PreAuthorizationAudit()
    with pytest.raises(core.ThreeArmV4AuthorizationError, match="schema drift"):
        core.preauthorize_and_claim_with_policy(
            authorization_path=auth_path,
            signature_path=signature_path,
            output_root=output,
            external_nwb_root=external,
            policy=policy,
            now=now,
            audit=audit,
            test_verifier=lambda raw, sig: True,
        )
    assert audit.nonce_claim_writes == 0 and audit.data_runtime_zero()


def test_signature_tamper_fails_before_nonce_and_runtime(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    policy = _policy(tmp_path)
    output = tmp_path / "outputs" / "run"
    external = tmp_path / "external" / "root"
    auth_path, signature_path = _write_auth_pair(
        tmp_path, _authorization(policy, output_root=output, external_root=external, now=now)
    )
    audit = core.PreAuthorizationAudit()
    with pytest.raises(core.ThreeArmV4AuthorizationError, match="invalid detached"):
        core.preauthorize_and_claim_with_policy(
            authorization_path=auth_path,
            signature_path=signature_path,
            output_root=output,
            external_nwb_root=external,
            policy=policy,
            now=now,
            audit=audit,
            test_verifier=lambda raw, sig: False,
        )
    assert audit.nonce_claim_writes == 0 and audit.data_runtime_zero()


@pytest.mark.parametrize("mutation", ("nonce", "expiry", "path", "runtime"))
def test_nonce_expiry_path_or_runtime_tamper_fails_before_claim(
    tmp_path: Path, mutation: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    now = datetime.now(timezone.utc)
    policy = _policy(tmp_path)
    output = tmp_path / "outputs" / "run"
    external = tmp_path / "external" / "root"
    authorization = _authorization(policy, output_root=output, external_root=external, now=now)
    if mutation == "nonce":
        authorization["nonce"] = "short"
    elif mutation == "expiry":
        authorization["issued_at"] = (now - timedelta(minutes=3)).isoformat()
        authorization["expires_at"] = (now - timedelta(minutes=2)).isoformat()
    elif mutation == "path":
        authorization["output_root"] = str((tmp_path / "outside").resolve())
    else:
        policy["runtime_identity"] = {"host": "tampered", "python": {}}
        authorization["bindings"] = core._authorization_bindings(policy)
    auth_path, signature_path = _write_auth_pair(tmp_path, authorization)
    audit = core.PreAuthorizationAudit()
    pattern = {
        "nonce": "nonce",
        "expiry": "expired",
        "path": "output-root binding",
        "runtime": "runtime identity drift",
    }[mutation]
    with pytest.raises(core.ThreeArmV4AuthorizationError, match=pattern):
        core.preauthorize_and_claim_with_policy(
            authorization_path=auth_path,
            signature_path=signature_path,
            output_root=output,
            external_nwb_root=external,
            policy=policy,
            now=now,
            audit=audit,
            test_verifier=lambda raw, sig: True,
        )
    assert audit.nonce_claim_writes == 0 and audit.data_runtime_zero()


def test_nonce_claim_is_atomic_and_duplicate_fails(tmp_path: Path) -> None:
    audit = core.PreAuthorizationAudit()
    nonce = "7" * 64
    claim = core.claim_nonce_once(nonce, claim_root=tmp_path / "claims", audit=audit)
    assert claim.is_file() and audit.nonce_claim_writes == 1
    with pytest.raises(core.ThreeArmV4AuthorizationError, match="already claimed"):
        core.claim_nonce_once(nonce, claim_root=tmp_path / "claims", audit=audit)
    assert audit.nonce_claim_writes == 1 and audit.data_runtime_zero()


def test_per_cell_duplicate_resume_partial_and_incomplete_aggregate(
    tmp_path: Path, contract: dict[str, object],
) -> None:
    output = tmp_path / "run"
    first = core.expected_cell_keys(contract)[0]
    core.publish_cell_result(
        output, key=first, contract=contract, r2=0.1,
        prediction_target_artifact=_artifact_pin(),
    )
    state = core.scan_resume_state(output, contract=contract)
    assert state["complete_cell_count"] == 1 and state["missing_cell_count"] == 269
    with pytest.raises(core.ThreeArmV4LedgerError, match="duplicate"):
        core.publish_cell_result(
            output, key=first, contract=contract, r2=0.1,
            prediction_target_artifact=_artifact_pin(),
        )
    with pytest.raises(core.ThreeArmV4LedgerError, match="before all 270"):
        core.publish_full_aggregate(
            output,
            contract=contract,
            aggregate={"t4_minus_zero4": {}, "t4_minus_ts4": {}},
        )
    partial = output / "cells" / "partial.tmp"
    partial.write_text("partial", encoding="utf-8")
    with pytest.raises(core.ThreeArmV4LedgerError, match="partial/unknown"):
        core.scan_resume_state(output, contract=contract)


def test_full_270_cell_aggregate_is_atomic_and_not_repeatable(
    tmp_path: Path, contract: dict[str, object],
) -> None:
    output = tmp_path / "full"
    for index, key in enumerate(core.expected_cell_keys(contract)):
        core.publish_cell_result(
            output, key=key, contract=contract, r2=float(index) / 1000.0,
            prediction_target_artifact=_artifact_pin(),
        )
    state = core.scan_resume_state(output, contract=contract)
    assert state["complete_cell_count"] == 270 and state["missing_cell_count"] == 0
    result = core.publish_full_aggregate(
        output,
        contract=contract,
        aggregate={
            "t4_minus_zero4": {"status": "SYNTHETIC_LEDGER_TEST"},
            "t4_minus_ts4": {"status": "SYNTHETIC_LEDGER_TEST"},
        },
    )
    assert result["mode"] == "0444"
    with pytest.raises(core.ThreeArmV4LedgerError, match="duplicate"):
        core.publish_full_aggregate(
            output,
            contract=contract,
            aggregate={"t4_minus_zero4": {}, "t4_minus_ts4": {}},
        )


def test_blocked_writer_and_dry_run_create_no_executable_capability(tmp_path: Path) -> None:
    prelaunch = tmp_path / "blocked"
    written = writer.write_blocked_prelaunch(prelaunch, ROOT)
    assert written["status"] == "BLOCKED_MISSING_ZERO4_TERMINALS"
    assert written["executable_prelaunch_sealed"] is False
    stored = writer.load_stored_blocked_prelaunch(prelaunch, ROOT)
    payload = runner.dry_run_payload(prelaunch, ROOT)
    assert stored["executable_prelaunch_sealed"] is False
    assert payload["authorization_or_signature_read"] is False
    assert payload["checkpoint_nwb_normalizer_torch_import_allowed"] is False
    assert payload["external_scoring_capability_created"] is False


def test_blocked_score_mode_refuses_before_reading_caller_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    prelaunch = tmp_path / "blocked"
    writer.write_blocked_prelaunch(prelaunch, ROOT)
    auth = tmp_path / "does-not-exist.auth"
    sig = tmp_path / "does-not-exist.sig"
    external = tmp_path / "does-not-exist.external"
    output = tmp_path / "does-not-exist.output"
    rc = runner.main(
        [
            "--mode", "score",
            "--prelaunch-dir", str(prelaunch),
            "--repo-root", str(ROOT),
            "--authorization", str(auth),
            "--signature", str(sig),
            "--external-nwb-root", str(external),
            "--output-root", str(output),
        ]
    )
    assert rc == 2
    assert "BLOCKED_MISSING_ZERO4_TERMINALS" in capsys.readouterr().err
    assert not auth.exists() and not sig.exists() and not external.exists() and not output.exists()


def test_dry_run_subprocess_does_not_import_torch_or_offer_execution(tmp_path: Path) -> None:
    prelaunch = tmp_path / "blocked"
    writer.write_blocked_prelaunch(prelaunch, ROOT)
    script = (
        "import json,sys; "
        "from pathlib import Path; "
        "from sua_exploration.scripts.run_dandi688_subm_co_three_arm_score_only_v4 "
        "import dry_run_payload; "
        f"p=dry_run_payload(Path({str(prelaunch)!r}),Path({str(ROOT)!r})); "
        "print(json.dumps({'payload':p,'torch_imported':'torch' in sys.modules}))"
    )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    observed = json.loads(completed.stdout)
    assert observed["torch_imported"] is False
    assert observed["payload"]["status"] == "BLOCKED_MISSING_ZERO4_TERMINALS"


def test_zero4_runtime_import_is_post_authorization_only_by_source_layout() -> None:
    source = Path(core.__file__).read_text(encoding="utf-8")
    prefix, suffix = source.split("def attach_arm_descriptor_after_authorization", 1)
    assert "paired_view_c1_shared_zero4 import" not in prefix
    assert "from mc_maze.paired_view_c1_shared_zero4 import" in suffix
    assert "import torch" not in prefix
