from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from sua_exploration.mc_maze import subm_co_three_arm_score_only_v5 as core
from sua_exploration.scripts import run_dandi688_subm_co_three_arm_score_only_v5 as runner
from sua_exploration.scripts import write_dandi688_subm_co_three_arm_score_only_prelaunch_v5 as writer


ROOT = Path(__file__).resolve().parents[2]


def _complete_slots() -> list[dict[str, object]]:
    slots = copy.deepcopy(core.checkpoint_slots_v5())
    for row in slots:
        if row["arm"] != "shared_zero4":
            continue
        seed = int(row["seed"])
        digest = hashlib.sha256(f"synthetic-zero4-{seed}".encode()).hexdigest()
        size = 10_000 + seed
        row.update(
            sha256=digest,
            bytes=size,
            mode="0444",
            status="PINNED_FUTURE_SEALED_TERMINAL_CLOSURE",
        )
        row["closure"] = {
            "schema": "sealed_terminal_checkpoint_closure_v5",
            "status": "SEALED_TERMINAL_CLOSURE_VERIFIED",
            "path": f"synthetic/closures/shared_zero4_s{seed}.json",
            "sha256": hashlib.sha256(f"synthetic-closure-{seed}".encode()).hexdigest(),
            "bytes": 500 + seed,
            "mode": "0444",
            "slot_binding_sha256": core._slot_binding(
                str(row["arm"]), seed, int(row["epoch"]), str(row["path"]), digest, size, "0444"
            ),
        }
    core.validate_checkpoint_slots(slots, require_complete=True)
    return slots


def _contract() -> dict[str, object]:
    cohort = [
        {
            "asset_id": f"synthetic-asset-{index:02d}",
            "session_id": f"sub-M_ses-synthetic-{index:02d}",
            "frozen_path": f"sub-M/synthetic-{index:02d}.nwb",
            "nwb_sha256": hashlib.sha256(f"nwb-{index}".encode()).hexdigest(),
            "nwb_bytes": 1000 + index,
        }
        for index in range(15)
    ]
    query_counts = {str(row["asset_id"]): 47_253 for row in cohort}
    return core.build_contract_v5(
        cohort=cohort, query_counts=query_counts, checkpoint_slots=_complete_slots()
    )


def _grant(tmp_path: Path, contract: dict[str, object], *, identity: str = "grant-a") -> core.AuthorizationGrant:
    output_root = (tmp_path / "output").absolute()
    return core.AuthorizationGrant(
        schema=core.GRANT_SCHEMA,
        status=core.GRANT_STATUS,
        permitted_action=core.GRANT_ACTION,
        authorization_sha256=hashlib.sha256(identity.encode()).hexdigest(),
        nonce=hashlib.sha256((identity + "-nonce").encode()).hexdigest(),
        output_root=output_root,
        contract_sha256=str(contract["contract_sha256"]),
        cohort_sha256=str(contract["cohort_sha256"]),
        query_map_sha256=str(contract["query_map_sha256"]),
        checkpoint_slots_sha256=str(contract["checkpoint_slots_sha256"]),
        preimport_authorization_complete=True,
    )


def _publish_one(
    grant: core.AuthorizationGrant, contract: dict[str, object], key: core.CellKey, r2: float = 0.5
) -> tuple[dict[str, object], dict[str, object]]:
    artifact = core.publish_prediction_target_artifact(
        grant, contract, key=key, content=("synthetic:" + repr(key)).encode()
    )
    cell = core.publish_cell_result(
        grant, contract, key=key, r2=r2, prediction_target_artifact=artifact
    )
    return artifact, cell


def _rewrite_immutable_json(path: Path, payload: dict[str, object]) -> None:
    os.chmod(path, 0o600)
    path.write_bytes(core.canonical_bytes(payload))
    os.chmod(path, 0o444)


def test_v5_production_checkpoint_table_has_exact_zero4_closure_blocker() -> None:
    slots = core.checkpoint_slots_v5()
    assert len(slots) == 9
    assert [(row["arm"], row["seed"]) for row in core.missing_checkpoint_slots(slots)] == [
        ("shared_zero4", 42), ("shared_zero4", 43), ("shared_zero4", 44)
    ]
    with pytest.raises(core.ThreeArmV5BlockedError, match="BLOCKED_MISSING_ZERO4_TERMINALS"):
        core.validate_checkpoint_slots(slots, require_complete=True)


def test_contract_requires_external_grant_not_self_rehashed_cohort(tmp_path: Path) -> None:
    contract = _contract()
    grant = _grant(tmp_path, contract)
    core.validate_contract_against_grant(contract, grant)
    mutated = copy.deepcopy(contract)
    mutated["cohort"][0]["session_id"] = "sub-M_ses-self-rehashed"
    mutated["cohort_sha256"] = core.canonical_sha256(mutated["cohort"])
    body = dict(mutated); body.pop("contract_sha256")
    mutated["contract_sha256"] = core.canonical_sha256(body)
    with pytest.raises(core.ThreeArmV5AuthorizationError, match="external verified grant"):
        core.validate_contract_against_grant(mutated, grant)


def test_arbitrary_zero4_hex_without_future_closure_fails() -> None:
    slots = copy.deepcopy(core.checkpoint_slots_v5())
    target = next(row for row in slots if row["arm"] == "shared_zero4" and row["seed"] == 42)
    target["sha256"] = "a" * 64
    target["bytes"] = 123
    target["mode"] = "0444"
    with pytest.raises(core.ThreeArmV5Error, match="exact zero4 closure placeholder"):
        core.validate_checkpoint_slots(slots, require_complete=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_r2_fails_closed(tmp_path: Path, value: float) -> None:
    contract = _contract(); grant = _grant(tmp_path, contract)
    key = core.expected_cell_keys(contract, grant)[0]
    artifact = core.publish_prediction_target_artifact(grant, contract, key=key, content=b"fixture")
    with pytest.raises(core.ThreeArmV5LedgerError, match="finite"):
        core.publish_cell_result(grant, contract, key=key, r2=value, prediction_target_artifact=artifact)


def test_duplicate_and_partial_artifacts_fail_closed(tmp_path: Path) -> None:
    contract = _contract(); grant = _grant(tmp_path, contract)
    key = core.expected_cell_keys(contract, grant)[0]
    artifact = core.publish_prediction_target_artifact(grant, contract, key=key, content=b"fixture")
    with pytest.raises(core.ThreeArmV5LedgerError, match="duplicate"):
        core.publish_prediction_target_artifact(grant, contract, key=key, content=b"fixture")
    with pytest.raises(core.ThreeArmV5LedgerError, match="partial/unknown prediction"):
        core.scan_resume_state(grant, contract)
    core.publish_cell_result(grant, contract, key=key, r2=0.2, prediction_target_artifact=artifact)
    with pytest.raises(core.ThreeArmV5LedgerError, match="duplicate"):
        core.publish_cell_result(grant, contract, key=key, r2=0.2, prediction_target_artifact=artifact)


def test_resume_revalidates_wrong_query_and_mutable_artifact(tmp_path: Path) -> None:
    contract = _contract(); grant = _grant(tmp_path, contract)
    key = core.expected_cell_keys(contract, grant)[0]
    artifact, cell = _publish_one(grant, contract, key)
    cell_path = grant.output_root / str(cell["path"])
    payload = json.loads(cell_path.read_text())
    payload["query_window_count"] += 1
    _rewrite_immutable_json(cell_path, payload)
    with pytest.raises(core.ThreeArmV5LedgerError, match="wrong query"):
        core.scan_resume_state(grant, contract)
    payload["query_window_count"] -= 1
    _rewrite_immutable_json(cell_path, payload)
    artifact_path = grant.output_root / str(artifact["path"])
    os.chmod(artifact_path, 0o644)
    with pytest.raises(core.ThreeArmV5LedgerError, match="mutable prediction"):
        core.scan_resume_state(grant, contract)


def test_resume_rejects_cell_unknown_key_and_unknown_file(tmp_path: Path) -> None:
    contract = _contract(); grant = _grant(tmp_path, contract)
    key = core.expected_cell_keys(contract, grant)[0]
    _, cell = _publish_one(grant, contract, key)
    cell_path = grant.output_root / str(cell["path"])
    payload = json.loads(cell_path.read_text()); payload["caller_statistics"] = {}
    _rewrite_immutable_json(cell_path, payload)
    with pytest.raises(core.ThreeArmV5LedgerError, match="exact schema"):
        core.scan_resume_state(grant, contract)
    payload.pop("caller_statistics")
    _rewrite_immutable_json(cell_path, payload)
    unknown = grant.output_root / "cells" / "partial.tmp"
    unknown.write_bytes(b"partial")
    with pytest.raises(core.ThreeArmV5LedgerError, match="partial/unknown cell"):
        core.scan_resume_state(grant, contract)


def test_unknown_directory_and_partial_aggregate_fail_closed(tmp_path: Path) -> None:
    contract = _contract(); grant = _grant(tmp_path, contract)
    grant.output_root.mkdir()
    (grant.output_root / "cells" / "unknown-empty-directory").mkdir(parents=True)
    with pytest.raises(core.ThreeArmV5LedgerError, match="unknown cell directory"):
        core.scan_resume_state(grant, contract)
    (grant.output_root / "cells" / "unknown-empty-directory").rmdir()
    (grant.output_root / "cells").rmdir()
    (grant.output_root / "aggregate").mkdir()
    with pytest.raises(core.ThreeArmV5LedgerError, match="partial/unknown aggregate"):
        core.scan_resume_state(grant, contract)


def test_symlink_chain_fails_closed(tmp_path: Path) -> None:
    contract = _contract(); grant = _grant(tmp_path, contract)
    grant.output_root.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    (grant.output_root / "cells").symlink_to(outside, target_is_directory=True)
    with pytest.raises(core.ThreeArmV5LedgerError, match="symlink"):
        core.scan_resume_state(grant, contract)


def test_same_authorization_grant_is_required_for_resume(tmp_path: Path) -> None:
    contract = _contract(); grant_a = _grant(tmp_path, contract, identity="a")
    key = core.expected_cell_keys(contract, grant_a)[0]
    _publish_one(grant_a, contract, key)
    grant_b = _grant(tmp_path, contract, identity="b")
    with pytest.raises(core.ThreeArmV5LedgerError, match="authorization grant drift"):
        core.scan_resume_state(grant_b, contract)


def test_aggregate_rejects_partial_matrix_and_has_no_caller_statistics_parameter(tmp_path: Path) -> None:
    contract = _contract(); grant = _grant(tmp_path, contract)
    assert "aggregate" not in inspect.signature(core.publish_full_aggregate).parameters
    with pytest.raises(TypeError):
        core.publish_full_aggregate(grant, contract, aggregate={})  # type: ignore[call-arg]
    with pytest.raises(core.ThreeArmV5LedgerError, match="before all 270"):
        core.publish_full_aggregate(grant, contract)


def test_full_270_aggregate_is_reconstructed_with_fixed_bootstrap(tmp_path: Path) -> None:
    contract = _contract(); grant = _grant(tmp_path, contract)
    keys = core.expected_cell_keys(contract, grant)
    session_order = {row["session_id"]: index for index, row in enumerate(contract["cohort"])}
    seed_order = {seed: index for index, seed in enumerate(core.SEEDS)}
    base = {"shared_t4": 0.52, "shared_zero4": 0.40, "shared_ts4": 0.43}
    for key in keys:
        r2 = base[key.arm] + 0.001 * session_order[key.session_id] + 0.002 * seed_order[key.seed]
        _publish_one(grant, contract, key, r2=r2)
    state = core.public_resume_state(grant, contract)
    assert state["complete_cell_count"] == 270 and state["missing_cell_count"] == 0
    pin = core.publish_full_aggregate(grant, contract)
    aggregate_path = grant.output_root / str(pin["path"])
    aggregate = json.loads(aggregate_path.read_text())
    assert aggregate["statistics_source"] == "reconstructed_from_270_verified_cells"
    assert aggregate["verified_cell_count"] == 270
    assert aggregate["bootstrap_policy"]["replicates"] == 100_000
    assert set(aggregate["comparisons"]) == {"t4_minus_zero4", "t4_minus_ts4"}
    assert aggregate["overall_three_arm_claim_pass"] is True
    for comparison in aggregate["comparisons"].values():
        assert set(comparison["views"]) == {"sua", "pseudo_mua"}
        assert comparison["comparison_pass"] is True
        assert all(view["hierarchical_bootstrap_95"]["lower"] > 0 for view in comparison["views"].values())
    assert stat.S_IMODE(aggregate_path.stat().st_mode) == 0o444
    with pytest.raises(core.ThreeArmV5LedgerError, match="duplicate"):
        core.publish_full_aggregate(grant, contract)


def test_blocked_writer_is_append_only_and_creates_no_grant_or_capability(tmp_path: Path) -> None:
    output = tmp_path / "blocked-v5"
    result = writer.write_blocked_prelaunch(output)
    assert result["status"] == "BLOCKED_MISSING_ZERO4_TERMINALS"
    assert result["executable_prelaunch_sealed"] is False
    stored = writer.load_stored_blocked_prelaunch(output)
    assert stored["external_capability_created"] is False
    payload = runner.dry_run_payload(output)
    assert payload["external_verified_grant_present"] is False
    assert payload["checkpoint_nwb_torch_gpu_r2_allowed"] is False
    with pytest.raises(writer.StaticThreeArmV5Error, match="already exists"):
        writer.write_blocked_prelaunch(output)


def test_dry_run_does_not_import_torch(tmp_path: Path) -> None:
    output = tmp_path / "blocked-v5"
    writer.write_blocked_prelaunch(output)
    script = (
        "import json,sys; from pathlib import Path; "
        "from sua_exploration.scripts.run_dandi688_subm_co_three_arm_score_only_v5 import dry_run_payload; "
        f"p=dry_run_payload(Path({str(output)!r})); "
        "print(json.dumps({'status':p['status'],'torch':'torch' in sys.modules}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    observed = json.loads(completed.stdout)
    assert observed == {"status": "BLOCKED_MISSING_ZERO4_TERMINALS", "torch": False}


def test_v4_artifacts_remain_exactly_unchanged() -> None:
    for name, (digest, size) in writer.V4_PINS.items():
        path = writer.V4_ROOT / name
        assert core.sha256_file(path) == digest
        assert path.stat().st_size == size
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
