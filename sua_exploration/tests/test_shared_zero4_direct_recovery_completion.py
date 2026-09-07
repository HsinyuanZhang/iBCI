from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from sua_exploration.scripts import shared_zero4_direct_recovery_completion as direct
from sua_exploration.scripts import shared_zero4_terminal_completion_bridge as bridge
from sua_exploration.scripts import eval_paired_view_c1_shared_zero4_terminal as evaluator
from sua_exploration.scripts import aggregate_t4_paired_view_c1_three_arm_terminal as aggregator
from sua_exploration.scripts import orchestrate_paired_view_c1_shared_zero4_postcompletion as orchestrator

DEAD_PID = 99_999_999

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value, *, mode: int = 0o444) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(mode)
    return path.resolve()


def _meta(path: Path, *, claimed_mode: str = "0444") -> dict:
    return {
        "canonical_path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
        "mode": claimed_mode,
    }


def _rewrite(path: Path, payload: dict) -> None:
    path.chmod(0o644)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    teacher = _write(tmp_path / "common/teacher.ckpt", b"teacher\n")
    manifest = _write(tmp_path / "common/train_val_manifest.json", {"split": "27/6/6"})
    data_manifest = _write(tmp_path / "common/data_manifest.json", {"data": "sealed"})
    data_dir = tmp_path / "common/data"
    data_dir.mkdir(parents=True)
    sua_cache = tmp_path / "common/cache/sua"
    pseudo_cache = tmp_path / "common/cache/pseudo_mua"
    sua_cache.mkdir(parents=True)
    pseudo_cache.mkdir(parents=True)
    initial_digests = {str(seed): {"sha256": f"initial-{seed}"} for seed in direct.FIXED_SEEDS}
    program = _write(
        tmp_path / "common/program_receipt.json",
        {"program": "shared-zero4", "initial_state_digests": initial_digests},
    )
    monkeypatch.setattr(direct, "EXPECTED_MANIFEST_SHA256", _sha(manifest))

    paths: dict[int, dict[str, Path]] = {}
    for seed in direct.FIXED_SEEDS:
        name = (
            f"t4_paired_view_c1_shared_zero4_source_prelaunch_v4_20260805_shared_zero4_s{seed}"
            if seed in (42, 43)
            else "shared_zero4_seed44_fresh_direct_recovery"
        )
        run = (tmp_path / "checkpoints" / name).resolve()
        initial_payload = {"sha256": f"initial-{seed}", "tensor_count": 3}
        initial = _write(run / "initial_state_digest.json", initial_payload)
        checkpoints = [
            _write(run / "epoch_ckpts" / f"epoch_{index:03d}.ckpt", f"seed{seed}-epoch{index}".encode())
            for index in range(12)
        ]
        side = {
            "group": "shared_zero4_direct_standardized",
            "side_dim": 4,
            "coordinate": "source_only_t4_standardized_coordinate",
            "construction": (
                "np.zeros((N,4), dtype=np.float32) directly in standardized coordinate; "
                "no raw T4 descriptor and no normalizer arithmetic"
            ),
            "target_direction_label_reads_for_descriptor": 0,
            "t4_trial_rate_reads_for_descriptor": 0,
            "target_t4_rate_fit_calls": 0,
            "raw_t4_constructed": False,
            "source_t4_normalizer_arithmetic_performed": False,
        }
        metadata_payload = {
            "schema_version": 2,
            "status": "completed",
            "seed": seed,
            "experiment": "paired_view_c1_shared_zero4_source",
            "training_kind": "shared_paired_view_direct_standardized_zero4",
            "variant": "B3S",
            "task": "CO",
            "signal_view": "paired_sua_pseudo_mua",
            "split_counts": [27, 6, 6],
            "max_units_exclusive": 100,
            "output_dir": str(run),
            "teacher_checkpoint": str(teacher),
            "teacher_sha256": _sha(teacher),
            "data_dir": str(data_dir.resolve()),
            "train_val_manifest": str(manifest),
            "train_val_manifest_sha256": _sha(manifest),
            "data_manifest": str(data_manifest),
            "data_manifest_sha256": _sha(data_manifest),
            "program_receipt": str(program),
            "program_receipt_sha256": _sha(program),
            "held_out_test_evaluated": False,
            "formal_sua_files_opened": False,
            "subm_nwb_files_opened": False,
            "materialized_split_scope": {
                "formal_paths_resolved": False,
                "subm_nwb_files_opened": False,
            },
            "session_files": {"train": [], "val": [], "test": []},
            "no_heldout_backprop_contract": {
                "source_train_sessions": 27,
                "development_heldout_sessions": 6,
                "formal_test_sessions": 6,
                "optimizer_and_backward_scope": "source_train_27_only",
                "development_enters_train_dataloader": False,
                "development_enters_loss": False,
                "development_enters_optimizer": False,
                "development_uses_backward_gradients": False,
                "development_scoring_invoked_by_this_run": False,
                "formal_paths_resolved": False,
                "formal_files_opened": False,
            },
            "view_configs": {
                "sua": {
                    "signal_view": "sua", "cache_dir": str(sua_cache.resolve()),
                    "side_features": side,
                },
                "pseudo_mua": {
                    "signal_view": "pseudo_mua", "cache_dir": str(pseudo_cache.resolve()),
                    "side_features": side,
                },
            },
            "initial_state": {
                "path": str(initial), "sha256": _sha(initial), "digest": initial_payload,
                "matches_prelaunch": True,
            },
            "training": {
                "max_epochs": 12,
                "no_early_stopping": True,
                "checkpoint_every_epoch": True,
                "terminal_checkpoint": "epoch_011.ckpt",
                "checkpoint_selection": "fixed_terminal_epoch_011_no_selection",
                "learning_rate": 1e-4,
                "batch_size": 32,
                "source_activity_calibration_n_trials": 10,
                "future_development_activity_calibration_n_trials": 30,
                "future_development_query_start_trial": 50,
                "future_development_trials_30_49_enter_zero4_identity_or_descriptor": False,
                "loss_mode": "task_only",
                "freeze_decoder": False,
                "shared_optimizer_steps": True,
                "random_calibration": False,
                "development_score_invoked": False,
                "development_score_artifact_paths": [],
            },
            "epoch_checkpoints": [str(path) for path in checkpoints],
            "terminal_checkpoint": str(checkpoints[11]),
            "terminal_checkpoint_sha256": _sha(checkpoints[11]),
        }
        metadata = _write(run / "run_metadata.json", metadata_payload)
        cost = _write(
            run / "post_run_cost_receipt.json",
            {
                "schema_version": 1,
                "status": "completed",
                "run_metadata_path": str(metadata),
                "run_metadata_sha256": _sha(metadata),
                "terminal_checkpoint": str(checkpoints[11]),
                "terminal_checkpoint_sha256": _sha(checkpoints[11]),
                "formal_sua_files_opened": False,
                "subm_nwb_files_opened": False,
                "development_score_invoked": False,
            },
        )
        paths[seed] = {
            "run": run, "metadata": metadata, "cost": cost, "initial": initial,
            "terminal": checkpoints[11], "epoch0": checkpoints[0],
        }
        if seed in (42, 43):
            result = tmp_path / "results"
            initial_copy = _write(result / "initial_state" / f"shared_zero4_s{seed}.json", initial.read_bytes())
            smoke = _write(result / "preflight" / f"seed{seed}_smoke.json", {"seed": seed})
            started = _write(result / "status" / f"shared_zero4_s{seed}.started.json", {"seed": seed})
            nonce = _write(result / "nonce" / f"seed{seed}.json", {"seed": seed})
            closure_dir = result / "closure" / f"shared_zero4_s{seed}"
            copied = []
            for source in (metadata, initial, cost, smoke):
                copy = _write(closure_dir / source.name, source.read_bytes())
                copied.append({"source": str(source), "copy": str(copy), "metadata": _meta(copy)})
            closure = _write(
                closure_dir / "closure_manifest.json",
                {
                    "schema": "t4_paired_view_c1_shared_zero4_remote_v2_closure_v1",
                    "status": "completed_source_only_score_blind",
                    "seed": seed,
                    "files": copied,
                    "terminal_checkpoint": _meta(checkpoints[11], claimed_mode="0664"),
                    "formal_or_subm_endpoint_access": False,
                },
            )
            closure_dir.chmod(0o555)
            completed = _write(
                result / "status" / f"shared_zero4_s{seed}.complete.json",
                {
                    "schema": "t4_paired_view_c1_shared_zero4_remote_v2_completed_v1",
                    "status": "completed_source_only_score_blind",
                    "seed": seed,
                    "started_status": _meta(started),
                    "authorization_nonce_claim": _meta(nonce),
                    "real_cuda_smoke": _meta(smoke),
                    "checkpoint_dir": str(run),
                    "terminal_checkpoint": _meta(checkpoints[11], claimed_mode="0664"),
                    "run_metadata": _meta(metadata, claimed_mode="0644"),
                    "initial_state": _meta(initial_copy),
                    "closure_manifest": _meta(closure),
                    "physical_gpu_uuid": "GPU-fixture",
                    "formal_or_subm_endpoint_access": False,
                },
            )
            paths[seed]["completed"] = completed

    # The real predecessor evidence is the seed42 V2 completed receipt itself,
    # not a V3 failure receipt.  Only its recorded terminal mode is stale.
    incident_checkpoint = paths[42]["terminal"]
    incident_receipt = paths[42]["completed"]
    return paths, incident_receipt, incident_checkpoint


def _build_and_write(tmp_path: Path, paths, incident_receipt, incident_checkpoint) -> Path:
    payload = direct.build_payload(
        seed42_completed=paths[42]["completed"],
        seed43_completed=paths[43]["completed"],
        seed44_run_dir=paths[44]["run"],
        mode_incident_evidence_receipt=incident_receipt,
        mode_incident_checkpoint=incident_checkpoint,
        seed44_trainer_pid=DEAD_PID,
    )
    return direct.write_payload_exclusive(tmp_path / "direct_completion.json", payload)


def test_direct_recovery_receipt_verifies_complete_content_bound_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, incident_receipt, incident_checkpoint = _fixture(tmp_path, monkeypatch)
    receipt = _build_and_write(tmp_path, paths, incident_receipt, incident_checkpoint)
    verified = direct.verify_receipt(receipt)
    assert verified["kind"] == direct.KIND
    assert verified["fixed_seeds"] == [42, 43, 44]
    assert verified["rows"]["44"]["origin"] == "direct_recovery_fresh_run"
    incident = verified["payload"][
        "predecessor_v2_completed_preseal_mode_metadata_incidents"
    ]
    assert incident["classification"] == (
        "predecessor_v2_completed_preseal_mode_metadata_incidents"
    )
    assert set(incident["incidents_by_seed"]) == {"42", "43"}
    for seed in ("42", "43"):
        rows = incident["incidents_by_seed"][seed]["preseal_mode_rows"]
        assert [
            (row["source_kind"], row["json_pointer"], row["recorded_mode"], row["live_mode"])
            for row in rows
        ] == [
            ("v2_completed_receipt", "$.run_metadata", "0644", "0444"),
            ("v2_completed_receipt", "$.terminal_checkpoint", "0664", "0444"),
            ("v2_closure_manifest", "$.terminal_checkpoint", "0664", "0444"),
        ]
    assert incident["incident_evidence_is_a_v3_receipt"] is False
    assert incident["incident_used_as_authorization"] is False
    assert verified["payload"]["score_read_or_evaluated"] is False
    assert bridge.verify_direct_recovery_receipt(receipt)["sha256"] == _sha(receipt)


def test_direct_recovery_kind_integrates_without_opening_any_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, incident_receipt, incident_checkpoint = _fixture(tmp_path, monkeypatch)
    receipt = _build_and_write(tmp_path, paths, incident_receipt, incident_checkpoint)
    _payload, binding = evaluator._load_supported_completion_receipt(
        receipt, program_receipt_sha256="not-used-for-content-bound-direct-kind"
    )
    assert (
        binding["kind"], binding["schema"], binding["status"], binding["sha256"]
    ) == (direct.KIND, direct.SCHEMA, direct.STATUS, _sha(receipt))
    bridge.require_supported_run_binding(
        binding,
        run_dir=paths[44]["run"],
        metadata_path=paths[44]["metadata"],
        terminal_checkpoint=paths[44]["terminal"],
        seed=44,
    )
    count, missing = aggregator._completion_seed_count(receipt)
    assert (count, missing) == (3, [])
    identity = aggregator._completion_identity(receipt)
    assert (
        identity["kind"], identity["schema"], identity["status"], identity["sha256"]
    ) == (direct.KIND, direct.SCHEMA, direct.STATUS, _sha(receipt))
    orchestrated = orchestrator._verify_completion_receipt(receipt)
    assert orchestrated["sha256"] == _sha(receipt)
    assert orchestrator._completion_run_dir(orchestrated, seed=44) == paths[44]["run"]
    controlled = orchestrator._controlled_sources()
    assert "direct_recovery_completion" in controlled
    assert "legacy_paired_view_aggregator" in controlled


def test_synthetic_zero4_artifact_binds_exact_direct_receipt_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, incident_receipt, incident_checkpoint = _fixture(tmp_path, monkeypatch)
    receipt = _build_and_write(tmp_path, paths, incident_receipt, incident_checkpoint)
    sessions = [f"dev_{index}" for index in range(6)]
    artifact = _write(
        tmp_path / "synthetic_zero4_s44_sua.json",
        {
            "schema_version": 1,
            "purpose": "shared_zero4_terminal_fixed_development_evaluation",
            "generated_by": "eval_paired_view_c1_shared_zero4_terminal.py",
            "variant": "B3S", "seed": 44, "task": "CO", "signal_view": "sua",
            "shared_weights": True,
            "side_feature_group": "shared_zero4_direct_standardized",
            "checkpoint_selection_rule": "fixed_terminal_epoch_011_no_selection",
            "checkpoint_epoch_index": 11, "protocol_epoch_number": 12,
            "uses_backward_gradients": False, "development_uses_backward_gradients": False,
            "no_test_files_evaluated": True, "formal_sua_files_opened": False,
            "subm_nwb_files_opened": False,
            "matrix_completion_kind": direct.KIND,
            "matrix_completion_schema": direct.SCHEMA,
            "matrix_completion_status": direct.STATUS,
            "matrix_completion_receipt": str(receipt),
            "matrix_completion_receipt_sha256": _sha(receipt),
            "protocol": {
                "source_activity_calibration_n": 10,
                "development_activity_calibration_n": 30,
                "descriptor_label_pool_n": None,
                "query_start_trial": 50,
                "trials_30_49_enter_zero4_identity_or_descriptor": False,
            },
            "descriptor_access": {
                "target_direction_label_reads_for_descriptor": 0,
                "t4_trial_rate_reads_for_descriptor": 0,
                "target_t4_rate_fit_calls": 0,
                "raw_t4_constructed": False,
                "source_t4_normalizer_arithmetic_performed": False,
                "all_development_records_bitwise_float32_zero": True,
            },
            "session_splits": {"val": sessions},
            "per_session_r2": {name: 0.25 for name in sessions},
            "mean_r2": 0.25, "variant_score": 0.25,
            "checkpoint": str(paths[44]["terminal"]),
            "checkpoint_sha256": _sha(paths[44]["terminal"]),
            "run_metadata_sha256": _sha(paths[44]["metadata"]),
        },
    )
    _values, observed_sessions, evidence = aggregator._load_zero4_view(
        path=artifact, seed=44, view="sua", completion_path=receipt,
        completion_sha256=_sha(receipt), completion_kind=direct.KIND,
        completion_schema=direct.SCHEMA, completion_status=direct.STATUS,
    )
    assert observed_sessions == sessions
    assert (
        evidence["matrix_completion_kind"], evidence["matrix_completion_schema"],
        evidence["matrix_completion_status"], evidence["matrix_completion_receipt_sha256"],
    ) == (direct.KIND, direct.SCHEMA, direct.STATUS, _sha(receipt))
    wrong = json.loads(artifact.read_text())
    wrong["matrix_completion_kind"] = "v3_external_v7_adapter"
    _rewrite(artifact, wrong)
    with pytest.raises(ValueError, match="completion binding drift"):
        aggregator._load_zero4_view(
            path=artifact, seed=44, view="sua", completion_path=receipt,
            completion_sha256=_sha(receipt), completion_kind=direct.KIND,
            completion_schema=direct.SCHEMA, completion_status=direct.STATUS,
        )


def test_partial_receipt_wrong_seed_set_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, incident_receipt, incident_checkpoint = _fixture(tmp_path, monkeypatch)
    receipt = _build_and_write(tmp_path, paths, incident_receipt, incident_checkpoint)
    payload = json.loads(receipt.read_text())
    del payload["terminal_rows"]["43"]
    _rewrite(receipt, payload)
    with pytest.raises(direct.DirectRecoveryError, match="terminal seed set drift"):
        direct.verify_receipt(receipt)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("wrong_seed", "drift"),
        ("wrong_config", "drift"),
        ("wrong_cache", "source content identity"),
        ("score_flag", "drift"),
    ],
)
def test_seed44_metadata_identity_config_or_score_tamper_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, match: str
) -> None:
    paths, incident_receipt, incident_checkpoint = _fixture(tmp_path, monkeypatch)
    metadata = paths[44]["metadata"]
    payload = json.loads(metadata.read_text())
    if mutation == "wrong_seed":
        payload["seed"] = 45
    elif mutation == "wrong_config":
        payload["training"]["batch_size"] = 31
    elif mutation == "wrong_cache":
        alternate = tmp_path / "alternate_cache"
        alternate.mkdir()
        payload["view_configs"]["sua"]["cache_dir"] = str(alternate.resolve())
    else:
        payload["training"]["development_score_invoked"] = True
    _rewrite(metadata, payload)
    if mutation == "wrong_cache":
        cost = paths[44]["cost"]
        cost_payload = json.loads(cost.read_text())
        cost_payload["run_metadata_sha256"] = _sha(metadata)
        _rewrite(cost, cost_payload)
    with pytest.raises(direct.DirectRecoveryError, match=match):
        direct.build_payload(
            seed42_completed=paths[42]["completed"],
            seed43_completed=paths[43]["completed"],
            seed44_run_dir=paths[44]["run"],
            mode_incident_evidence_receipt=incident_receipt,
            mode_incident_checkpoint=incident_checkpoint,
            seed44_trainer_pid=DEAD_PID,
        )


def test_terminal_hash_tamper_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, incident_receipt, incident_checkpoint = _fixture(tmp_path, monkeypatch)
    receipt = _build_and_write(tmp_path, paths, incident_receipt, incident_checkpoint)
    terminal = paths[44]["terminal"]
    terminal.chmod(0o644)
    terminal.write_bytes(b"tampered checkpoint\n")
    terminal.chmod(0o444)
    with pytest.raises(direct.DirectRecoveryError, match="terminal checkpoint.*binding drift"):
        direct.verify_receipt(receipt)


def test_missing_epoch_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, incident_receipt, incident_checkpoint = _fixture(tmp_path, monkeypatch)
    paths[44]["epoch0"].unlink()
    with pytest.raises(direct.DirectRecoveryError, match="epoch0 checkpoint is missing"):
        direct.build_payload(
            seed42_completed=paths[42]["completed"], seed43_completed=paths[43]["completed"],
            seed44_run_dir=paths[44]["run"], mode_incident_evidence_receipt=incident_receipt,
            mode_incident_checkpoint=incident_checkpoint, seed44_trainer_pid=DEAD_PID,
        )


def test_symlinked_seed44_cost_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, incident_receipt, incident_checkpoint = _fixture(tmp_path, monkeypatch)
    cost = paths[44]["cost"]
    replacement = _write(tmp_path / "replacement_cost.json", cost.read_bytes())
    cost.unlink()
    cost.symlink_to(replacement)
    with pytest.raises(direct.DirectRecoveryError, match="must not be a symlink"):
        direct.build_payload(
            seed42_completed=paths[42]["completed"], seed43_completed=paths[43]["completed"],
            seed44_run_dir=paths[44]["run"], mode_incident_evidence_receipt=incident_receipt,
            mode_incident_checkpoint=incident_checkpoint, seed44_trainer_pid=DEAD_PID,
        )


def test_predecessor_v2_incident_must_be_exact_mode_only_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, incident_receipt, incident_checkpoint = _fixture(tmp_path, monkeypatch)
    payload = json.loads(incident_receipt.read_text())
    payload["terminal_checkpoint"]["mode"] = "0444"
    _rewrite(incident_receipt, payload)
    with pytest.raises(direct.DirectRecoveryError, match="recorded mode must be 0664"):
        direct.build_payload(
            seed42_completed=paths[42]["completed"], seed43_completed=paths[43]["completed"],
            seed44_run_dir=paths[44]["run"], mode_incident_evidence_receipt=incident_receipt,
            mode_incident_checkpoint=incident_checkpoint, seed44_trainer_pid=DEAD_PID,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "not-a-v2-completed-schema"),
        ("status", "failed"),
        ("seed", 43),
    ],
)
def test_predecessor_mode_incident_rejects_wrong_schema_status_or_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value,
) -> None:
    paths, incident_receipt, incident_checkpoint = _fixture(tmp_path, monkeypatch)
    payload = json.loads(incident_receipt.read_text())
    payload[field] = value
    _rewrite(incident_receipt, payload)
    with pytest.raises(
        direct.DirectRecoveryError,
        match="must be the seed42 V2 completed receipt",
    ):
        direct.build_payload(
            seed42_completed=paths[42]["completed"],
            seed43_completed=paths[43]["completed"],
            seed44_run_dir=paths[44]["run"],
            mode_incident_evidence_receipt=incident_receipt,
            mode_incident_checkpoint=incident_checkpoint,
            seed44_trainer_pid=DEAD_PID,
        )


def test_missing_seed43_preseal_incident_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, incident_receipt, incident_checkpoint = _fixture(tmp_path, monkeypatch)
    seed43 = paths[43]["completed"]
    payload = json.loads(seed43.read_text())
    payload["terminal_checkpoint"]["mode"] = "0444"
    _rewrite(seed43, payload)
    with pytest.raises(
        direct.DirectRecoveryError,
        match="seed43 predecessor terminal_checkpoint recorded mode must be 0664",
    ):
        _build_and_write(tmp_path, paths, incident_receipt, incident_checkpoint)


@pytest.mark.parametrize(
    ("seed", "field", "wrong_mode", "required_mode"),
    [
        (42, "run_metadata", "0664", "0644"),
        (43, "run_metadata", "0444", "0644"),
        (42, "terminal_checkpoint", "0644", "0664"),
        (43, "terminal_checkpoint", "0444", "0664"),
    ],
)
def test_predecessor_wrong_recorded_modes_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
    field: str,
    wrong_mode: str,
    required_mode: str,
) -> None:
    paths, incident_receipt, incident_checkpoint = _fixture(tmp_path, monkeypatch)
    completed = paths[seed]["completed"]
    payload = json.loads(completed.read_text())
    payload[field]["mode"] = wrong_mode
    _rewrite(completed, payload)
    with pytest.raises(
        direct.DirectRecoveryError,
        match=rf"seed{seed} predecessor {field} recorded mode must be {required_mode}",
    ):
        _build_and_write(tmp_path, paths, incident_receipt, incident_checkpoint)


@pytest.mark.parametrize(("field", "drift"), [("sha256", "f" * 64), ("size_bytes", 1)])
def test_predecessor_path_size_or_sha_drift_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    drift,
) -> None:
    paths, incident_receipt, incident_checkpoint = _fixture(tmp_path, monkeypatch)
    completed = paths[43]["completed"]
    payload = json.loads(completed.read_text())
    payload["run_metadata"][field] = drift
    _rewrite(completed, payload)
    with pytest.raises(
        direct.DirectRecoveryError,
        match="seed43 predecessor run_metadata path/size/SHA drift",
    ):
        _build_and_write(tmp_path, paths, incident_receipt, incident_checkpoint)


def test_predecessor_path_drift_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, incident_receipt, incident_checkpoint = _fixture(tmp_path, monkeypatch)
    completed = paths[43]["completed"]
    payload = json.loads(completed.read_text())
    payload["terminal_checkpoint"]["canonical_path"] = str(paths[42]["terminal"])
    _rewrite(completed, payload)
    with pytest.raises(
        direct.DirectRecoveryError,
        match="seed43 predecessor terminal_checkpoint path/size/SHA drift",
    ):
        _build_and_write(tmp_path, paths, incident_receipt, incident_checkpoint)


def test_unexpected_third_preseal_mode_drift_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, incident_receipt, incident_checkpoint = _fixture(tmp_path, monkeypatch)
    completed = paths[43]["completed"]
    payload = json.loads(completed.read_text())
    payload["initial_state"]["mode"] = "0644"
    _rewrite(completed, payload)
    with pytest.raises(
        direct.DirectRecoveryError,
        match="seed43 V2 terminal chain invalid.*initial state.*binding drift",
    ):
        _build_and_write(tmp_path, paths, incident_receipt, incident_checkpoint)


@pytest.mark.parametrize("wrong_mode", ["0444", "0644"])
def test_missing_or_wrong_closure_terminal_preseal_row_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrong_mode: str,
) -> None:
    paths, incident_receipt, incident_checkpoint = _fixture(tmp_path, monkeypatch)
    completed_path = paths[43]["completed"]
    completed = json.loads(completed_path.read_text())
    closure = Path(completed["closure_manifest"]["canonical_path"])
    closure.parent.chmod(0o755)
    closure_payload = json.loads(closure.read_text())
    closure_payload["terminal_checkpoint"]["mode"] = wrong_mode
    _rewrite(closure, closure_payload)
    closure.parent.chmod(0o555)
    completed["closure_manifest"] = _meta(closure)
    _rewrite(completed_path, completed)
    with pytest.raises(
        direct.DirectRecoveryError,
        match=rf"seed43 predecessor terminal_checkpoint recorded mode must be 0664",
    ):
        _build_and_write(tmp_path, paths, incident_receipt, incident_checkpoint)


def test_default_bridge_still_rejects_stale_closure_terminal_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _incident_receipt, _incident_checkpoint = _fixture(tmp_path, monkeypatch)
    completed = json.loads(paths[42]["completed"].read_text())
    normalized_row = {
        "terminal_checkpoint": _meta(paths[42]["terminal"]),
    }
    with pytest.raises(bridge.TerminalAdapterError, match="metadata binding drift"):
        bridge._verify_closure(
            normalized_row,
            seed=42,
            origin="V2",
            checkpoint=paths[42]["terminal"],
            metadata_path=paths[42]["metadata"],
            closure_path=Path(completed["closure_manifest"]["canonical_path"]),
            initial_path=Path(completed["initial_state"]["canonical_path"]),
        )


def test_v2_closure_tamper_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, incident_receipt, incident_checkpoint = _fixture(tmp_path, monkeypatch)
    completed = json.loads(paths[42]["completed"].read_text())
    closure = Path(completed["closure_manifest"]["canonical_path"])
    closure.parent.chmod(0o755)
    payload = json.loads(closure.read_text())
    payload["seed"] = 99
    _rewrite(closure, payload)
    closure.parent.chmod(0o555)
    with pytest.raises((direct.DirectRecoveryError, bridge.TerminalAdapterError), match="binding drift"):
        direct.build_payload(
            seed42_completed=paths[42]["completed"], seed43_completed=paths[43]["completed"],
            seed44_run_dir=paths[44]["run"], mode_incident_evidence_receipt=incident_receipt,
            mode_incident_checkpoint=incident_checkpoint, seed44_trainer_pid=DEAD_PID,
        )


def test_live_trainer_pid_rejected_before_terminal_content_access(tmp_path: Path) -> None:
    with pytest.raises(direct.DirectRecoveryError, match="is still live"):
        direct.build_payload(
            seed42_completed=tmp_path / "must_not_open_42.json",
            seed43_completed=tmp_path / "must_not_open_43.json",
            seed44_run_dir=tmp_path / "must_not_open_44",
            mode_incident_evidence_receipt=tmp_path / "must_not_open_incident.json",
            mode_incident_checkpoint=tmp_path / "must_not_open_incident.ckpt",
            seed44_trainer_pid=os.getpid(),
        )


def test_seed44_terminal_content_must_be_presealed_by_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, incident_receipt, incident_checkpoint = _fixture(tmp_path, monkeypatch)
    paths[44]["epoch0"].chmod(0o664)
    with pytest.raises(direct.DirectRecoveryError, match="mode must be 0444"):
        direct.build_payload(
            seed42_completed=paths[42]["completed"], seed43_completed=paths[43]["completed"],
            seed44_run_dir=paths[44]["run"], mode_incident_evidence_receipt=incident_receipt,
            mode_incident_checkpoint=incident_checkpoint, seed44_trainer_pid=DEAD_PID,
        )
