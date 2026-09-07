"""Synthetic/adversarial tests for the V3 shared-Z4 post-completion bridge.

No checkpoint is deserialized and no neural data, GPU, formal endpoint, or
sub-M endpoint is touched by this file.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "sua_exploration/scripts"
sys.path.insert(0, str(ROOT / "sua_exploration"))
sys.path.insert(0, str(SCRIPTS))

import shared_zero4_terminal_completion_bridge as bridge
import eval_paired_view_c1_shared_zero4_terminal as evaluator


def _load_aggregator():
    source = SCRIPTS / "aggregate_t4_paired_view_c1_three_arm_terminal.py"
    spec = importlib.util.spec_from_file_location("v3_bridge_aggregator_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


aggregator = _load_aggregator()


def _write(path: Path, payload: Any, *, mode: int = 0o444) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, (dict, list)):
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(str(payload), encoding="utf-8")
    os.chmod(path, mode)
    return path.resolve()


def _meta(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {
        "canonical_path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": bridge.sha256_file(path),
        "mode": "0444",
    }


def _fixture(tmp_path: Path) -> tuple[Path, dict[int, dict[str, Path]]]:
    program = _write(tmp_path / "program.json", {"program": "same-three-seed-source"})
    v2_prelaunch = _write(tmp_path / "prelaunch/v2.json", {"schema": "fixture-v2"})
    v3_prelaunch = _write(tmp_path / "prelaunch/v3.json", {"schema": "fixture-v3"})
    rows: dict[str, Any] = {}
    paths: dict[int, dict[str, Path]] = {}

    for seed in bridge.FIXED_SEEDS:
        run = (tmp_path / "checkpoints" / f"shared_zero4_s{seed}").resolve()
        checkpoint = _write(run / "epoch_ckpts/epoch_011.ckpt", f"ckpt-{seed}".encode())
        initial_source = _write(run / "initial_state_digest.json", {"seed": seed})
        cost = _write(run / "post_run_cost_receipt.json", {"seed": seed, "score": False})
        metadata = _write(
            run / "run_metadata.json",
            {
                "status": "completed",
                "seed": seed,
                "training_kind": "shared_paired_view_direct_standardized_zero4",
                "variant": "B3S",
                "program_receipt": str(program),
                "program_receipt_sha256": bridge.sha256_file(program),
                "terminal_checkpoint": str(checkpoint),
                "terminal_checkpoint_sha256": bridge.sha256_file(checkpoint),
                "held_out_test_evaluated": False,
                "formal_sua_files_opened": False,
                "subm_nwb_files_opened": False,
                "training": {
                    "max_epochs": 12,
                    "terminal_checkpoint": "epoch_011.ckpt",
                    "checkpoint_selection": "fixed_terminal_epoch_011_no_selection",
                    "development_score_invoked": False,
                    "development_score_artifact_paths": [],
                },
            },
        )
        initial = _write(tmp_path / "results/initial_state" / f"shared_zero4_s{seed}.json", initial_source.read_bytes())
        smoke = _write(tmp_path / "results/preflight" / f"seed{seed}_smoke.json", {"seed": seed})
        started = _write(tmp_path / "results/status" / f"shared_zero4_s{seed}.started.json", {"seed": seed})
        nonce = _write(tmp_path / "results/nonce" / f"seed{seed}.json", {"seed": seed})

        closure_dir = tmp_path / "results/closure" / f"shared_zero4_s{seed}"
        copied = []
        for source in (metadata, initial_source, cost, smoke):
            copy = _write(closure_dir / source.name, source.read_bytes())
            copied.append({"source": str(source), "copy": str(copy), "metadata": _meta(copy)})
        origin = "V2" if seed in (42, 43) else "V3_renewal"
        closure_payload: dict[str, Any] = {
            "schema": (
                "t4_paired_view_c1_shared_zero4_remote_v2_closure_v1"
                if origin == "V2"
                else "t4_paired_view_c1_shared_zero4_remote_seed44_renewal_v3_closure_v1"
            ),
            "status": "completed_source_only_score_blind",
            "seed": seed,
            "files": copied,
            "terminal_checkpoint": _meta(checkpoint),
            "student_parameter_count": 123,
            "formal_or_subm_endpoint_access": False,
        }
        if origin == "V3_renewal":
            closure_payload.update(
                {
                    "v3_prelaunch": _meta(v3_prelaunch),
                    "development_score_invoked": False,
                    "score_read_or_evaluated": False,
                }
            )
        closure = _write(closure_dir / "closure_manifest.json", closure_payload)
        os.chmod(closure_dir, 0o555)

        terminal_row = {
            "seed": seed,
            "origin": origin,
            "completed_status": None,
            "run_metadata": _meta(metadata),
            "terminal_checkpoint": _meta(checkpoint),
            "closure_manifest": _meta(closure),
            "initial_state": _meta(initial),
            "authorization_nonce_claim": _meta(nonce),
            "formal_or_subm_endpoint_access": False,
            "score_read_or_evaluated_by_v3": False,
        }
        if origin == "V2":
            completed_payload = {
                "schema": "t4_paired_view_c1_shared_zero4_remote_v2_completed_v1",
                "status": "completed_source_only_score_blind",
                "seed": seed,
                "started_status": _meta(started),
                "authorization_nonce_claim": _meta(nonce),
                "real_cuda_smoke": _meta(smoke),
                "checkpoint_dir": str(run),
                "terminal_checkpoint": _meta(checkpoint),
                "run_metadata": _meta(metadata),
                "initial_state": _meta(initial),
                "closure_manifest": _meta(closure),
                "physical_gpu_uuid": "GPU-fixture",
                "formal_or_subm_endpoint_access": False,
            }
        else:
            launch = _write(tmp_path / "v3/launch.json", {"seed": seed})
            capability = _write(tmp_path / "v3/capability.json", {"seed": seed})
            completed_payload = {
                "schema": "t4_paired_view_c1_shared_zero4_remote_seed44_renewal_v3_completed_v1",
                "status": "completed_source_only_score_blind",
                "seed": seed,
                "v3_prelaunch": _meta(v3_prelaunch),
                "launch_precondition": _meta(launch),
                "seed_capability": _meta(capability),
                "authorization_nonce_claim": _meta(nonce),
                "real_cuda_smoke": _meta(smoke),
                "checkpoint_dir": str(run),
                "terminal_checkpoint": _meta(checkpoint),
                "run_metadata": _meta(metadata),
                "initial_state": _meta(initial),
                "closure_manifest": _meta(closure),
                "physical_gpu_uuid": "GPU-fixture",
                "formal_or_subm_endpoint_access": False,
                "development_score_invoked": False,
            }
        completed = _write(
            tmp_path / "results/status" / f"shared_zero4_s{seed}.complete.json",
            completed_payload,
        )
        terminal_row["completed_status"] = _meta(completed)
        rows[str(seed)] = terminal_row
        paths[seed] = {
            "run": run,
            "checkpoint": checkpoint,
            "metadata": metadata,
            "closure": closure,
            "completed": completed,
        }

    v3_started = _write(tmp_path / "v3/status/seed44.started.json", {"seed": 44})
    v2root_started = _write(tmp_path / "v3/status/seed44.v2root.started.json", {"seed": 44})
    v3_completion = _write(
        tmp_path / "v3/completion/seed44_completed.json",
        {
            "schema": "t4_paired_view_c1_shared_zero4_remote_seed44_renewal_v3_seed44_completion_v1",
            "status": "completed_source_only_score_blind",
            "seed": 44,
            "v3_prelaunch": _meta(v3_prelaunch),
            "v3_started": _meta(v3_started),
            "v2root_started": _meta(v2root_started),
            "v2root_completed": rows["44"]["completed_status"],
            "v2root_closure": rows["44"]["closure_manifest"],
            "terminal_checkpoint": rows["44"]["terminal_checkpoint"],
            "formal_or_subm_endpoint_access": False,
            "development_score_invoked": False,
            "score_read_or_evaluated": False,
        },
    )
    snapshot = {
        "fixed_seed_order": [42, 43, 44],
        "rows": rows,
        "score_read_or_evaluated": False,
    }
    snapshot["sha256"] = bridge.canonical_json_sha256(snapshot)
    adapter = _write(
        tmp_path / "v3/external_v7_adapter/receipt.json",
        {
            "schema": bridge.V3_ADAPTER_SCHEMA,
            "status": bridge.V3_ADAPTER_STATUS,
            "adapter_consumer": "external_v7",
            "v3_prelaunch": _meta(v3_prelaunch),
            "v2_prelaunch": _meta(v2_prelaunch),
            "v3_seed44_completion": _meta(v3_completion),
            "fixed_seed_order": [42, 43, 44],
            "terminal_snapshot": snapshot,
            "terminal_snapshot_sha256": snapshot["sha256"],
            "legacy_v1_finalizer": {
                "invoked": False,
                "compatible": False,
                "reason": "fixture",
                "v1_source_matrix_completion_read": False,
            },
            "consumer_contract": {
                "trust_only_sealed_v2_v3_terminal_and_closure_metadata": True,
                "checkpoint_deserialized": False,
                "metric_or_score_read": False,
                "formal_or_subm_endpoint_access": False,
            },
            "formal_or_subm_endpoint_access": False,
            "development_score_invoked": False,
            "score_read_or_evaluated": False,
        },
    )
    return adapter, paths


def _rewrite_adapter(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    os.chmod(path, 0o644)
    _write(path, payload)


def _refresh_snapshot_digest(payload: dict[str, Any]) -> None:
    snapshot = payload["terminal_snapshot"]
    snapshot.pop("sha256", None)
    snapshot["sha256"] = bridge.canonical_json_sha256(snapshot)
    payload["terminal_snapshot_sha256"] = snapshot["sha256"]


def test_v3_adapter_verifies_and_binds_evaluator_and_aggregator(tmp_path: Path) -> None:
    adapter, paths = _fixture(tmp_path)
    verified = bridge.verify_v3_adapter_receipt(adapter)
    assert verified["fixed_seeds"] == [42, 43, 44]
    assert verified["schema"] == bridge.V3_ADAPTER_SCHEMA
    bridge.require_v3_run_binding(
        verified,
        run_dir=paths[44]["run"],
        metadata_path=paths[44]["metadata"],
        terminal_checkpoint=paths[44]["checkpoint"],
        seed=44,
    )
    payload, evaluator_binding = evaluator._load_supported_completion_receipt(
        adapter, program_receipt_sha256="ignored-for-v3-adapter"
    )
    assert payload["status"] == bridge.V3_ADAPTER_STATUS
    assert evaluator_binding["kind"] == "v3_external_v7_adapter"
    assert aggregator._completion_seed_count(adapter) == (3, [])
    identity = aggregator._completion_identity(adapter)
    assert identity == {
        "kind": "v3_external_v7_adapter",
        "schema": bridge.V3_ADAPTER_SCHEMA,
        "status": bridge.V3_ADAPTER_STATUS,
        "path": str(adapter),
        "sha256": bridge.sha256_file(adapter),
    }


def test_v3_adapter_rejects_swapped_checkpoint_path_even_with_refreshed_digest(tmp_path: Path) -> None:
    adapter, _ = _fixture(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        rows = payload["terminal_snapshot"]["rows"]
        rows["42"]["terminal_checkpoint"] = rows["43"]["terminal_checkpoint"]
        _refresh_snapshot_digest(payload)

    _rewrite_adapter(adapter, mutate)
    with pytest.raises(bridge.TerminalAdapterError, match="completed checkpoint directory drift"):
        bridge.verify_v3_adapter_receipt(adapter)


def test_v3_adapter_rejects_wrong_sha_missing_seed_and_snapshot_tamper(tmp_path: Path) -> None:
    adapter, _ = _fixture(tmp_path)

    def wrong_sha(payload: dict[str, Any]) -> None:
        payload["terminal_snapshot"]["rows"]["42"]["run_metadata"]["sha256"] = "f" * 64
        _refresh_snapshot_digest(payload)

    _rewrite_adapter(adapter, wrong_sha)
    with pytest.raises(bridge.TerminalAdapterError, match="size/SHA/mode binding drift"):
        bridge.verify_v3_adapter_receipt(adapter)

    adapter, _ = _fixture(tmp_path / "missing")

    def missing_seed(payload: dict[str, Any]) -> None:
        payload["terminal_snapshot"]["rows"].pop("43")
        _refresh_snapshot_digest(payload)

    _rewrite_adapter(adapter, missing_seed)
    with pytest.raises(bridge.TerminalAdapterError, match="seed set drift"):
        bridge.verify_v3_adapter_receipt(adapter)

    adapter, _ = _fixture(tmp_path / "tamper")

    def tamper_without_digest(payload: dict[str, Any]) -> None:
        payload["terminal_snapshot"]["rows"]["42"]["origin"] = "V3_renewal"

    _rewrite_adapter(adapter, tamper_without_digest)
    with pytest.raises(bridge.TerminalAdapterError, match="snapshot digest drift"):
        bridge.verify_v3_adapter_receipt(adapter)


def test_v3_adapter_rejects_failure_receipts_and_nonsealed_files(tmp_path: Path) -> None:
    adapter, paths = _fixture(tmp_path)
    _write(paths[42]["completed"].parent / "shared_zero4_s42.failed.json", {"status": "failed"})
    with pytest.raises(bridge.TerminalAdapterError, match="mixed-in failure"):
        bridge.verify_v3_adapter_receipt(adapter)

    adapter, _ = _fixture(tmp_path / "nonsealed")
    os.chmod(adapter, 0o644)
    with pytest.raises(bridge.TerminalAdapterError, match="mode must be 0444"):
        bridge.verify_v3_adapter_receipt(adapter)

    adapter, _ = _fixture(tmp_path / "symlink")
    link = tmp_path / "adapter-link.json"
    link.symlink_to(adapter)
    with pytest.raises(bridge.TerminalAdapterError, match="must not be a symlink"):
        bridge.verify_v3_adapter_receipt(link)

    adapter, paths = _fixture(tmp_path / "closure")
    os.chmod(paths[44]["closure"].parent, 0o755)
    with pytest.raises(bridge.TerminalAdapterError, match="closure directory mode must be 0555"):
        bridge.verify_v3_adapter_receipt(adapter)


def test_v3_adapter_rejects_formal_scope_or_unbound_evaluator_run(tmp_path: Path) -> None:
    adapter, _ = _fixture(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["formal_or_subm_endpoint_access"] = True

    _rewrite_adapter(adapter, mutate)
    with pytest.raises(bridge.TerminalAdapterError, match="identity/scope drift"):
        bridge.verify_v3_adapter_receipt(adapter)

    adapter, paths = _fixture(tmp_path / "binding")
    verified = bridge.verify_v3_adapter_receipt(adapter)
    with pytest.raises(bridge.TerminalAdapterError, match="run directory drift"):
        bridge.require_v3_run_binding(
            verified,
            run_dir=paths[43]["run"],
            metadata_path=paths[42]["metadata"],
            terminal_checkpoint=paths[42]["checkpoint"],
            seed=42,
        )


def test_aggregator_requires_same_v3_receipt_schema_status_path_and_sha(tmp_path: Path) -> None:
    adapter, _ = _fixture(tmp_path)
    artifact = tmp_path / "score.json"
    sessions = [f"dev_{index}" for index in range(6)]
    values = [0.1 + index * 0.01 for index in range(6)]
    score = {
        "schema_version": 1,
        "purpose": "shared_zero4_terminal_fixed_development_evaluation",
        "generated_by": "eval_paired_view_c1_shared_zero4_terminal.py",
        "variant": "B3S",
        "seed": 42,
        "task": "CO",
        "signal_view": "sua",
        "shared_weights": True,
        "side_feature_group": "shared_zero4_direct_standardized",
        "checkpoint_selection_rule": "fixed_terminal_epoch_011_no_selection",
        "checkpoint_epoch_index": 11,
        "protocol_epoch_number": 12,
        "uses_backward_gradients": False,
        "development_uses_backward_gradients": False,
        "no_test_files_evaluated": True,
        "formal_sua_files_opened": False,
        "subm_nwb_files_opened": False,
        "matrix_completion_kind": "v3_external_v7_adapter",
        "matrix_completion_schema": bridge.V3_ADAPTER_SCHEMA,
        "matrix_completion_status": bridge.V3_ADAPTER_STATUS,
        "matrix_completion_receipt": str(adapter),
        "matrix_completion_receipt_sha256": bridge.sha256_file(adapter),
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
        "per_session_r2": dict(zip(sessions, values)),
        "mean_r2": sum(values) / len(values),
        "variant_score": sum(values) / len(values),
        "checkpoint": "fixture/epoch_011.ckpt",
        "checkpoint_sha256": "a" * 64,
        "run_metadata_sha256": "b" * 64,
    }
    _write(artifact, score, mode=0o644)
    loaded, loaded_sessions, _ = aggregator._load_zero4_view(
        path=artifact,
        seed=42,
        view="sua",
        completion_path=adapter,
        completion_sha256=bridge.sha256_file(adapter),
        completion_kind="v3_external_v7_adapter",
        completion_schema=bridge.V3_ADAPTER_SCHEMA,
        completion_status=bridge.V3_ADAPTER_STATUS,
    )
    assert loaded.tolist() == pytest.approx(values)
    assert loaded_sessions == sessions

    corruptions = (
        ("matrix_completion_schema", "swapped-schema"),
        ("matrix_completion_status", "not-completed"),
        ("matrix_completion_receipt", str(tmp_path / "another-receipt.json")),
        ("matrix_completion_receipt_sha256", "f" * 64),
    )
    for field, bad_value in corruptions:
        corrupted = json.loads(json.dumps(score))
        corrupted[field] = bad_value
        _write(artifact, corrupted, mode=0o644)
        with pytest.raises(ValueError, match="completion binding drift"):
            aggregator._load_zero4_view(
                path=artifact,
                seed=42,
                view="sua",
                completion_path=adapter,
                completion_sha256=bridge.sha256_file(adapter),
                completion_kind="v3_external_v7_adapter",
                completion_schema=bridge.V3_ADAPTER_SCHEMA,
                completion_status=bridge.V3_ADAPTER_STATUS,
            )
