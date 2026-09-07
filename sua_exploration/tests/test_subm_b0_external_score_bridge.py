"""Focused, no-target tests for the additive B0 external score bridge."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import subm_b0_external_score_bridge as bridge  # noqa: E402


def _synthetic_score(seed: int, authority: dict) -> dict:
    roster = authority["external_session_order"]
    queries = authority["external_cells"][f"t4_s{seed}"]["payload"]["session_query_receipts"]
    values = {name: 0.01 * (position + 1) + 0.001 * seed for position, name in enumerate(roster)}
    per_epoch = {
        str(epoch): {
            "checkpoint_path": f"/synthetic/epoch_{epoch - 1:03d}.ckpt",
            "checkpoint_sha256": hashlib.sha256(f"{seed}/{epoch}".encode()).hexdigest(),
            "per_session_r2": values,
            "mean_r2": sum(values.values()) / len(values),
        }
        for epoch in bridge.EPOCH_WINDOW
    }
    session_receipts = {}
    for name in roster:
        projection = bridge._query_projection(queries[name])
        session_receipts[name] = {
            **projection,
            "activity_calibration_trial_indices": list(range(30)),
            "target_session_carrier_fit_performed": False,
            "target_direction_labels_used_for_carrier": False,
            "target_velocity_labels_used_for_weight_updates": False,
            "backward_gradients": False,
            "decoder_weight_updates": False,
        }
    mean_by_session = {
        name: sum(per_epoch[str(epoch)]["per_session_r2"][name] for epoch in bridge.EPOCH_WINDOW) / len(bridge.EPOCH_WINDOW)
        for name in roster
    }
    target_authority = bridge.load_subm_target_byte_authority(expected_session_order=roster)
    target_byte_rows = {}
    for name in roster:
        expected = target_authority["sessions"][name]
        identity = {
            "st_dev": 1, "st_ino": len(name), "st_size": expected["expected_bytes"], "st_mode": 0o100644,
            "st_mtime_ns": 123, "st_ctime_ns": 456,
        }
        identity_sha = hashlib.sha256(bridge.canonical_json_bytes(identity)).hexdigest()
        target_byte_rows[name] = {
            **expected,
            "before_nwb_loader": {
                "phase": "before_nwb_loader", "actual_sha256": expected["expected_sha256"],
                "actual_bytes": expected["expected_bytes"], "identity": identity, "identity_sha256": identity_sha,
            },
            "after_nwb_loader": {
                "phase": "after_nwb_loader", "actual_sha256": expected["expected_sha256"],
                "actual_bytes": expected["expected_bytes"], "identity": identity, "identity_sha256": identity_sha,
            },
            "same_bytes_and_identity_before_after_loader": True,
        }
    return {
        "schema": bridge.SCORE_SCHEMA,
        "status": "B0_EXTERNAL_CPU_SCORE_COMPLETE__FIXED_EPOCH_AVERAGE",
        "seed": seed,
        "domain": bridge.EXTERNAL_DOMAIN,
        "variant": "B0",
        "source_topology": "original_SPINT_B0__BatchReferenceEncoder__trainable_copied_fc_id_in_out",
        "implementation_bindings": bridge.implementation_bindings(),
        "target_updates": False,
        "gpu_used": False,
        "formal_subc_test_nwb_opened": False,
        "normalizer_authority": {
            "behavior_normalizer_value_sha256": bridge.EXPECTED_BEHAVIOR_NORMALIZER_VALUE_SHA256,
            "target_domain_normalizer_refit_performed": False,
        },
        "domain_sessions": roster,
        "per_epoch": per_epoch,
        "per_session_mean_r2": mean_by_session,
        "mean_r2": sum(mean_by_session.values()) / len(mean_by_session),
        "target_nwb_byte_authority": {
            **{key: target_authority[key] for key in (
                "a2_official_preflight_path", "a2_official_preflight_sha256", "schema_ledger_path",
                "schema_ledger_sha256", "scope_manifest_path", "scope_manifest_sha256", "external_session_order",
            )},
            "session_bytes": target_byte_rows,
        },
        "session_query_receipts": session_receipts,
    }


def test_source_only_preflight_proves_original_b0_schedule_without_target(tmp_path: Path) -> None:
    payload = bridge.build_preflight(output_root=tmp_path / "fresh")
    assert set(payload) == {
        "schema", "status", "output_root", "a2_external_authority", "b0_source_audit",
        "b0_source_audit_sha256", "implementation_bindings", "behavior_normalizer", "fixed_policy",
        "subm_target_byte_authority",
        "science_scope", "target_data_opened", "target_data_discovered", "formal_subc_test_nwb_opened",
        "gpu_used", "cebra_imported", "score_emitted",
    }
    assert payload["target_data_opened"] is False
    assert payload["target_data_discovered"] is False
    assert payload["formal_subc_test_nwb_opened"] is False
    assert payload["gpu_used"] is False
    assert payload["fixed_policy"]["epoch_window"] == list(range(5, 13))
    assert payload["fixed_policy"]["epoch_mapping"] == "logical epoch e maps to epoch_{e-1:03d}.ckpt"
    for seed in bridge.SEEDS:
        row = payload["b0_source_audit"]["source_runs"][str(seed)]
        assert set(row["source_checkpoint_sha256_bundle"]) == {str(epoch) for epoch in bridge.EPOCH_WINDOW}
        assert row["source_checkpoint_mode"] == "0664"


def test_optional_cpu_topology_audit_does_not_change_authorization_source_digest(tmp_path: Path) -> None:
    light = bridge.build_preflight(output_root=tmp_path / "light", verify_checkpoint_payload=False)
    deep = bridge.build_preflight(output_root=tmp_path / "deep", verify_checkpoint_payload=True)
    assert light["b0_source_audit_sha256"] == deep["b0_source_audit_sha256"]
    topology = deep["b0_source_audit"]["source_runs"]["42"]["checkpoint_payload_audits"]["5"]
    assert topology == {
        "decoder_tensor_count": 31,
        "identity_encoder_tensor_count": 12,
        "topology": "BatchReferenceEncoder__trainable_copied_fc_id_in_out",
    }


def test_a2_authority_is_strict_pair_and_reports_interaction_separately() -> None:
    authority = bridge.load_sealed_a2_external_authority()
    assert authority["terminal_aggregate_sha256"] == bridge.A2_TERMINAL_SHA256
    assert len(authority["external_session_order"]) == 15
    interaction = authority["interaction"]
    assert interaction["mean_interaction"] == pytest.approx(0.23579886074553036)
    assert interaction["definition"].startswith("mean_session_R2(t4 external_subject_M)")


def test_subm_target_byte_authority_binds_a2_preflight_ledger_manifest_and_all_15_sessions() -> None:
    target_authority = bridge.load_subm_target_byte_authority()
    assert target_authority["a2_official_preflight_sha256"] == bridge.A2_OFFICIAL_PREFLIGHT_SHA256
    assert target_authority["schema_ledger_sha256"] == bridge.SUBM_SCHEMA_LEDGER_SHA256
    assert target_authority["scope_manifest_sha256"] == bridge.SUBM_SCOPE_MANIFEST_SHA256
    assert len(target_authority["sessions"]) == 15
    assert tuple(target_authority["sessions"]) == tuple(target_authority["external_session_order"])
    assert target_authority["target_nwb_opened_while_building_authority"] is False


def test_subm_target_authority_rejects_verified_download_tamper_without_opening_target() -> None:
    a2, _ = bridge._strict_pair(bridge.A2_ROOT / "official_cpu_preflight.json", label="test A2 official",
                                expected_sha256=bridge.A2_OFFICIAL_PREFLIGHT_SHA256)
    ledger, _ = bridge._strict_unpaired_json(bridge.SUBM_SCHEMA_LEDGER, label="test sub-M ledger",
                                              expected_sha256=bridge.SUBM_SCHEMA_LEDGER_SHA256)
    manifest, _ = bridge._strict_unpaired_json(bridge.SUBM_SCOPE_MANIFEST, label="test sub-M manifest",
                                                expected_sha256=bridge.SUBM_SCOPE_MANIFEST_SHA256)
    ledger = copy.deepcopy(ledger)
    external_session = "sub-M_ses-CO-20140307"
    asset_id = next(row["asset_id"] for row in manifest["selected_assets"] if row["session_id"] == external_session)
    next(row for row in ledger["verified_downloads"] if row["asset_id"] == asset_id)["sha256"] = "0" * 64
    with pytest.raises(bridge.B0BridgeError, match="verified-download pin drift"):
        bridge._build_subm_target_byte_authority(
            official_preflight=a2, ledger=ledger, manifest=manifest,
            expected_session_order=bridge.load_sealed_a2_external_authority()["external_session_order"],
        )


def test_target_byte_verifier_rejects_synthetic_file_drift_before_nwb_parser(tmp_path: Path) -> None:
    target_path = tmp_path / "sub-M" / "sub-M_ses-CO-20990101_behavior+ecephys.nwb"
    target_path.parent.mkdir()
    target_path.write_bytes(b"verified-target-bytes")
    expected_sha = hashlib.sha256(b"verified-target-bytes").hexdigest()
    authority = {
        "sessions": {
            "sub-M_ses-CO-20990101": {
                "asset_id": "synthetic-asset", "frozen_path": "sub-M/sub-M_ses-CO-20990101_behavior+ecephys.nwb",
                "expected_sha256": expected_sha, "expected_bytes": len(b"verified-target-bytes"),
                "a2_official_nwb_path": str(target_path),
            },
        },
    }
    verified = bridge._verify_target_nwb_before_or_after_loader(
        target_path, session="sub-M_ses-CO-20990101", target_authority=authority, phase="before_nwb_loader",
    )
    assert verified["actual_sha256"] == expected_sha
    target_path.write_bytes(b"drifted-target-bytes")
    with pytest.raises(bridge.B0BridgeError, match="SHA drift"):
        bridge._verify_target_nwb_before_or_after_loader(
            target_path, session="sub-M_ses-CO-20990101", target_authority=authority, phase="after_nwb_loader",
        )


def test_a11_within_b0_and_a2_within_authorities_match_fixed_b0_semantics() -> None:
    a11 = bridge.load_sealed_a11_within_b0_authority()
    a2_within = bridge.load_sealed_a2_within_authority()
    assert a11["sha256"] == bridge.A11_WITHIN_B0_SHA256
    assert a11["mean_r2"] == bridge.A11_WITHIN_B0_MEAN_R2
    assert set(a11["per_seed_mean_r2"]) == {"42", "43", "44"}
    assert a11["normalizer_value_sha256"] == bridge.EXPECTED_BEHAVIOR_NORMALIZER_VALUE_SHA256
    assert a11["within_session_order"] == a2_within["within_session_order"]
    assert set(a2_within["within_cells"]) == {"t4_s42", "t4_s43", "t4_s44", "z4_s42", "z4_s43", "z4_s44"}


def test_a11_within_b0_authority_fails_closed_when_b0_checkpoint_lineage_disagrees() -> None:
    source_audit = bridge.audit_b0_source_bundle(verify_checkpoint_payload=False)
    source_audit["source_runs"]["42"]["source_checkpoint_sha256_bundle"]["5"] = "0" * 64
    with pytest.raises(bridge.B0BridgeError, match="checkpoint lineage drift"):
        bridge.load_sealed_a11_within_b0_authority(source_audit=source_audit)


def test_immutable_preflight_pair_is_required_for_later_score_cells(tmp_path: Path) -> None:
    preflight = bridge.build_preflight(output_root=tmp_path / "initial")
    body = tmp_path / "root_reviewed_preflight.json"
    bridge._write_pair_once(body, preflight)
    loaded, digest = bridge.load_immutable_preflight(body)
    assert digest == hashlib.sha256(bridge.canonical_json_bytes(preflight)).hexdigest()
    assert loaded["b0_source_audit_sha256"] == preflight["b0_source_audit_sha256"]


def test_strict_pair_rejects_sidecar_drift(tmp_path: Path) -> None:
    body = tmp_path / "body.json"
    body.write_text('{"x":1}\n', encoding="utf-8")
    side = tmp_path / "body.json.sha256"
    side.write_text("0" * 64 + "  body.json\n", encoding="ascii")
    os.chmod(body, 0o444); os.chmod(side, 0o444)
    with pytest.raises(bridge.B0BridgeError, match="body/sidecar mismatch"):
        bridge._strict_pair(body, label="synthetic")


def test_pair_writer_rolls_back_body_when_sidecar_collides_after_body_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = tmp_path / "transaction.json"
    sidecar = tmp_path / "transaction.json.sha256"
    original_writer = bridge._write_exclusive_regular
    calls = 0

    def race_sidecar_collision(path: Path, raw: bytes, *, mode: int):
        nonlocal calls
        calls += 1
        if calls == 2:
            sidecar.write_text("external collision\n", encoding="ascii")
            os.chmod(sidecar, 0o444)
        return original_writer(path, raw, mode=mode)

    monkeypatch.setattr(bridge, "_write_exclusive_regular", race_sidecar_collision)
    with pytest.raises(bridge.B0BridgeError, match="transaction failed"):
        bridge._write_pair_once(body, {"x": 1})
    assert not body.exists()
    assert sidecar.read_text(encoding="ascii") == "external collision\n"


def test_score_freshness_rejects_an_orphan_sidecar_before_any_target_path(tmp_path: Path) -> None:
    body = tmp_path / "external_subject_M_b0_s42.json"
    sidecar = body.with_name(f"{body.name}.sha256")
    sidecar.write_text("orphan\n", encoding="ascii")
    with pytest.raises(bridge.B0BridgeError, match="body or sidecar already exists"):
        bridge._require_fresh_score_output_pair(body)
    assert sidecar.exists() and not body.exists()


def test_execute_rejects_orphan_sidecar_before_authorization_or_any_data_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = tmp_path / "external_subject_M_b0_s42.json"
    body.with_name(f"{body.name}.sha256").write_text("orphan\n", encoding="ascii")

    def authorization_must_not_run(*_args, **_kwargs):
        raise AssertionError("authorization/preflight consumer ran after stale output")

    monkeypatch.setattr(bridge, "_authorization_payload", authorization_must_not_run)
    with pytest.raises(bridge.B0BridgeError, match="body or sidecar already exists"):
        bridge.execute_cpu_score(42, output_path=body, preflight={}, authorization_path=tmp_path / "authorization.json")


def test_pair_writer_rejects_symlinked_parent_and_never_writes_through_it(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(bridge.B0BridgeError, match="parent must be a real directory"):
        bridge._write_pair_once(linked_parent / "body.json", {"x": 1})
    assert not list(real_parent.iterdir())


def test_strict_pair_rejects_a_body_symlink_before_any_json_consumer(tmp_path: Path) -> None:
    real_body = tmp_path / "real.json"
    bridge._write_pair_once(real_body, {"x": 1})
    linked_body = tmp_path / "linked.json"
    linked_body.symlink_to(real_body)
    with pytest.raises(bridge.B0BridgeError, match="cannot be opened without following symlinks"):
        bridge._strict_pair(linked_body, label="symlink poison")


def test_strict_pair_rejects_body_rename_after_verified_fd_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = tmp_path / "body.json"
    bridge._write_pair_once(body, {"x": 1})
    replacement = tmp_path / "replacement.json"
    replacement.write_text('{"x":2}\n', encoding="utf-8")
    os.chmod(replacement, 0o444)
    original_read_verified = bridge._read_verified_bytes

    def read_then_rename(path: Path, label: str, **kwargs):
        verified = original_read_verified(path, label, **kwargs)
        if label == "rename poison":
            os.replace(replacement, body)
        return verified

    monkeypatch.setattr(bridge, "_read_verified_bytes", read_then_rename)
    with pytest.raises(bridge.B0BridgeError, match="pathname identity changed after open"):
        bridge._strict_pair(body, label="rename poison")


def test_verified_private_snapshot_survives_mutable_source_swap_and_restore(tmp_path: Path) -> None:
    checkpoint = tmp_path / "mutable_checkpoint.ckpt"
    teacher = tmp_path / "mutable_teacher.ckpt"
    checkpoint.write_bytes(b"checkpoint-original")
    teacher.write_bytes(b"teacher-original")
    os.chmod(checkpoint, 0o664)
    os.chmod(teacher, 0o600)
    verified_checkpoint = bridge._read_verified_bytes(checkpoint, "synthetic mutable checkpoint")
    verified_teacher = bridge._read_verified_bytes(teacher, "synthetic mutable teacher")
    replacement = tmp_path / "checkpoint-replacement.ckpt"
    replacement.write_bytes(b"checkpoint-replaced")
    os.chmod(replacement, 0o664)
    with bridge._private_verified_snapshots({"b0_checkpoint": verified_checkpoint,
                                             "teacher_checkpoint": verified_teacher}) as snapshots:
        os.replace(replacement, checkpoint)
        assert bridge._read_verified_bytes(snapshots["b0_checkpoint"], "private checkpoint snapshot",
                                           expected_sha256=verified_checkpoint.sha256).raw == b"checkpoint-original"
        assert stat.S_IMODE(snapshots["b0_checkpoint"].stat().st_mode) == 0o400
    checkpoint.write_bytes(b"checkpoint-original")
    os.chmod(checkpoint, 0o664)
    assert bridge._read_verified_bytes(checkpoint, "synthetic checkpoint restored").raw == b"checkpoint-original"


def test_actual_source_checkpoint_loader_smoke_proves_snapshot_canonical_state_parity() -> None:
    smoke = bridge.source_only_checkpoint_loader_smoke(seed=42, logical_epoch=5)
    assert smoke["status"] == "B0_SOURCE_ONLY_CHECKPOINT_LOADER_SMOKE_PASSED"
    assert smoke["canonical_loader_state_parity"] is True
    assert smoke["target_data_opened"] is False
    assert smoke["gpu_used"] is False


def test_aggregate_reports_all_predeclared_system_contrasts_and_not_b0_interaction(tmp_path: Path) -> None:
    authority = bridge.load_sealed_a2_external_authority()
    paths: dict[int, Path] = {}
    for seed in bridge.SEEDS:
        body = tmp_path / f"b0_s{seed}.json"
        bridge._write_pair_once(body, _synthetic_score(seed, authority))
        paths[seed] = body
    aggregate = bridge.aggregate_payload(b0_score_pairs=paths)
    assert set(aggregate["contrasts"]) == {"t4_minus_b0", "z4_minus_b0"}
    assert set(aggregate["system_shift_interactions"]) == {
        "t4_minus_b0_external_minus_within", "z4_minus_b0_external_minus_within",
    }
    assert aggregate["a11_within_b0_authority"]["sha256"] == bridge.A11_WITHIN_B0_SHA256
    assert aggregate["sealed_cross_domain_carrier_interaction"]["terminal_aggregate_sha256"] == bridge.A2_TERMINAL_SHA256
    assert "B0 is not included" in aggregate["sealed_cross_domain_carrier_interaction"]["definition"]
    assert "not carrier causal" in aggregate["science_scope"]


def test_score_validation_rejects_live_implementation_binding_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authority = bridge.load_sealed_a2_external_authority()
    payload = _synthetic_score(42, authority)
    replacement = tmp_path / "changed_aggregate_cli.py"
    replacement.write_text("# changed\n", encoding="utf-8")
    monkeypatch.setitem(bridge.IMPLEMENTATION_BINDING_PATHS, "aggregate_cli", replacement)
    with pytest.raises(bridge.B0BridgeError, match="implementation source/path drift"):
        bridge._validate_b0_score(payload, seed=42, a2=authority)


def test_implementation_binding_map_covers_every_b0_execution_and_publication_entrypoint() -> None:
    assert {
        "read_only_preflight_cli", "score_cli", "aggregate_cli", "root_preflight_publisher_cli",
        "root_authorization_publisher_cli",
    }.issubset(bridge.IMPLEMENTATION_BINDING_PATHS)


def test_aggregate_fails_closed_on_query_projection_mismatch(tmp_path: Path) -> None:
    authority = bridge.load_sealed_a2_external_authority()
    paths: dict[int, Path] = {}
    for seed in bridge.SEEDS:
        payload = _synthetic_score(seed, authority)
        if seed == 43:
            payload["session_query_receipts"][authority["external_session_order"][0]]["dataset_query_window_count"] += 1
        body = tmp_path / f"b0_s{seed}.json"
        bridge._write_pair_once(body, payload)
        paths[seed] = body
    with pytest.raises(bridge.B0BridgeError, match="query projection"):
        bridge.aggregate_payload(b0_score_pairs=paths)


def test_public_score_cli_dry_run_is_target_free_and_has_no_gpu_or_train_switch(tmp_path: Path) -> None:
    script = ROOT / "sua_exploration" / "scripts" / "score_subm_b0_external_score_bridge.py"
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([sys.executable, str(script), "--seed", "42", "--output-root", str(tmp_path / "fresh")],
                            cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    assert payload["target_data_opened"] is False and payload["gpu_used"] is False
    help_result = subprocess.run([sys.executable, str(script), "--help"], cwd=ROOT, env=env,
                                 text=True, capture_output=True, check=True)
    assert "--train" not in help_result.stdout and "--gpu" not in help_result.stdout
    assert "--execute" in help_result.stdout
    denied = subprocess.run([sys.executable, str(script), "--seed", "42", "--execute", "--output-root", str(tmp_path / "later")],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert denied.returncode == 2
    assert "--preflight and --authorization" in denied.stderr


def test_root_publishers_fail_before_any_write_without_explicit_root_tokens(tmp_path: Path) -> None:
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = ""
    preflight_writer = ROOT / "sua_exploration" / "scripts" / "write_subm_b0_external_score_bridge_preflight.py"
    auth_writer = ROOT / "sua_exploration" / "scripts" / "authorize_subm_b0_external_score_bridge.py"
    first = subprocess.run([sys.executable, str(preflight_writer), "--out", str(tmp_path / "preflight.json")],
                           cwd=ROOT, env=env, text=True, capture_output=True)
    second = subprocess.run([sys.executable, str(auth_writer), "--preflight", str(tmp_path / "absent.json"),
                            "--out", str(tmp_path / "auth.json")], cwd=ROOT, env=env, text=True, capture_output=True)
    assert first.returncode == second.returncode == 2
    assert "B0_BRIDGE_ROOT_PREFLIGHT_MINT" in first.stderr
    assert "B0_BRIDGE_ROOT_SCORE_AUTHORIZATION" in second.stderr
    assert not list(tmp_path.iterdir())
