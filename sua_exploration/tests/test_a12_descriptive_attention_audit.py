"""Focused synthetic-contract tests for the new A12 metadata boundary.

These tests deliberately use tiny temporary checkpoint *bytes*, not Torch
checkpoint objects.  They never import Torch/PyNWB, open an NWB, or invoke a
decoder forward path.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import stat
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = ROOT / "sua_exploration"
if str(SUA_ROOT) not in sys.path:
    sys.path.insert(0, str(SUA_ROOT))

from mc_maze import a12_descriptive_attention_audit as core  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the deliberate train-then-validation roster insertion order:
    # A12 treats it as provenance, not a set.
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_preflight_module():
    script = SUA_ROOT / "scripts" / "a12_descriptive_attention_preflight.py"
    spec = importlib.util.spec_from_file_location("a12_descriptive_attention_preflight_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_layout(tmp_path: Path) -> core.AuditLayout:
    """Build a six-arm, 48-checkpoint metadata fixture without model objects."""

    train_sessions = [f"fixture_train_{index:02d}" for index in range(27)]
    validation_sessions = list(core.DEFAULT_VALIDATION_SESSIONS)
    formal_sessions = list(core.SEALED_FORMAL_TEST_SESSIONS)
    splits = {
        "train": train_sessions,
        "val": validation_sessions,
        "test": formal_sessions,
    }
    manifest_path = tmp_path / "configs" / "strict_manifest.json"
    _write_json(manifest_path, {"split_counts": [27, 6, 6], "session_splits": splits})
    teacher_path = tmp_path / "checkpoints" / "teacher.ckpt"
    teacher_path.parent.mkdir(parents=True, exist_ok=True)
    teacher_path.write_bytes(b"tiny A12 teacher bytes\n")
    implementation = tmp_path / "fixture_core.py"
    implementation.write_text("# fixture-only implementation binding\n", encoding="utf-8")

    session_order = train_sessions + validation_sessions
    unit_counts = {name: 20 + index for index, name in enumerate(session_order)}
    channel_counts = {name: 30 + index for index, name in enumerate(session_order)}
    session_files = {
        "train": [str(tmp_path / "data" / f"{name}_behavior+ecephys.nwb") for name in train_sessions],
        "val": [str(tmp_path / "data" / f"{name}_behavior+ecephys.nwb") for name in validation_sessions],
        "test": [],
    }
    normalizer_sha = "a" * 64
    manifest_sha = core.sha256_file(manifest_path)
    teacher_sha = core.sha256_file(teacher_path)
    result_paths: dict[tuple[str, int], Path] = {}
    metadata_paths: dict[tuple[str, int], Path] = {}
    result_hashes: dict[tuple[str, int], str] = {}

    for seed in core.SEEDS:
        for arm in ("t4", "z4"):
            run_dir = tmp_path / "runs" / f"fixture_{arm}_s{seed}"
            epoch_dir = run_dir / "epoch_ckpts"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            checkpoints: list[Path] = []
            for epoch_zero_based in range(12):
                checkpoint = epoch_dir / f"epoch_{epoch_zero_based:03d}.ckpt"
                checkpoint.write_bytes(f"fixture {arm} seed={seed} epoch={epoch_zero_based}\n".encode("ascii"))
                checkpoints.append(checkpoint)

            side_features: dict[str, Any] = {
                "group": arm,
                "side_dim": 4,
                "pool_size": core.M30,
                "feature_version": 1,
                "normalization_sha256": normalizer_sha,
                "permutation_seed": None,
            }
            if arm == "z4":
                side_features.update(
                    {
                        "normalization_base_feature_group": "t4",
                        "descriptor_contract": dict(core.EXPECTED_Z4_DESCRIPTOR_CONTRACT),
                    }
                )
            metadata_path = run_dir / "run_metadata.json"
            metadata = {
                "status": "completed",
                "variant": core.EXPECTED_VARIANT,
                "task": core.EXPECTED_TASK,
                "signal_view": core.EXPECTED_SIGNAL_VIEW,
                "split_counts": [27, 6, 6],
                "seed": seed,
                "max_units_exclusive": 100,
                "held_out_test_evaluated": False,
                "teacher_checkpoint": str(teacher_path),
                "teacher_sha256": teacher_sha,
                "train_val_manifest": str(manifest_path),
                "train_val_manifest_sha256": manifest_sha,
                "session_splits": splits,
                "session_files": session_files,
                "session_unit_counts": unit_counts,
                "session_channel_counts": channel_counts,
                "side_features": side_features,
                "training": {
                    "bin_size_ms": core.BIN_SIZE_MS,
                    "calibration_n_trials": core.M30,
                    "window_size": core.WINDOW_SIZE,
                    "trial_length": core.TRIAL_LENGTH,
                    "max_epochs": 12,
                    "no_early_stopping": True,
                    "checkpoint_every_epoch": True,
                    "loss_mode": "task_only",
                    "identity_mode": "calibrated",
                    "deterministic": True,
                    "freeze_decoder": False,
                },
                "validation_protocol": {
                    "calibration_trials": "trials[0:calibration_n_trials]",
                    "evaluation_windows": "trials[calibration_n_trials:] only",
                    "trial_disjoint": True,
                },
                "held_out_evaluation_protocol": {
                    "backward_gradients_on_held_out_sessions": False,
                    "held_out_behavior_labels_used_for_updates": False,
                    "held_out_test_evaluated": False,
                },
                "epoch_checkpoints": [str(checkpoint) for checkpoint in checkpoints],
            }
            if arm == "z4":
                metadata["decoder_architecture"] = {"mode": "coupled", "fixed_slot_count": 0}
            _write_json(metadata_path, metadata)
            metadata_sha = core.sha256_file(metadata_path)

            result_path = tmp_path / "results" / f"{arm}_s{seed}.json"
            per_epoch: dict[str, Any] = {}
            for epoch in core.EPOCH_WINDOW:
                checkpoint = checkpoints[epoch - 1]
                per_epoch[str(epoch)] = {
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": core.sha256_file(checkpoint),
                    "per_session_r2": {session: 0.0 for session in validation_sessions},
                }
            result = {
                "variant": core.EXPECTED_VARIANT,
                "seed": seed,
                "task": core.EXPECTED_TASK,
                "signal_view": core.EXPECTED_SIGNAL_VIEW,
                "split_counts": [27, 6, 6],
                "max_units_exclusive": 100,
                "no_test_files_evaluated": True,
                "uses_backward_gradients": False,
                "uses_behavior_labels_for_weight_updates": False,
                "calibration_trial_selection_uses_behavior_labels": False,
                "calibration_features_use_behavior_labels": True,
                "calibration_feature_label_scope": "chronological_rewarded_trials[0:30]",
                "checkpoint_selection_rule": "pre_declared_fixed_epoch_window_no_argmax",
                "epoch_list": list(core.EPOCH_WINDOW),
                "run_metadata_path": str(metadata_path),
                "run_metadata_sha256": metadata_sha,
                "session_splits": splits,
                "session_unit_counts": unit_counts,
                "teacher_ckpt_sha256": teacher_sha,
                "train_val_manifest_sha256": manifest_sha,
                "protocol": {
                    "calibration_n": core.M30,
                    "train_activity_calibration_n": core.M30,
                    "evaluation_forward_calibration_n": core.M30,
                    "pool_size": core.M30,
                    "epoch_window": list(core.EPOCH_WINDOW),
                    "total_epochs": 12,
                    "selection_mode": "first",
                },
                "per_epoch": per_epoch,
            }
            _write_json(result_path, result)
            result_paths[(arm, seed)] = result_path
            metadata_paths[(arm, seed)] = metadata_path
            result_hashes[(arm, seed)] = core.sha256_file(result_path)

    return core.AuditLayout(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        teacher_path=teacher_path,
        result_paths=result_paths,
        metadata_paths=metadata_paths,
        expected_result_sha256=result_hashes,
        expected_manifest_sha256=manifest_sha,
        expected_teacher_sha256=teacher_sha,
        expected_normalizer_sha256=normalizer_sha,
        implementation_paths={"fixture_core": implementation},
        canonical=False,
    )


def _passed_fixture_payload(tmp_path: Path) -> tuple[core.AuditLayout, dict[str, Any]]:
    layout = _fixture_layout(tmp_path)
    payload = core.build_metadata_preflight(
        layout=layout,
        verify_checkpoint_bytes=True,
        include_implementation_bindings=True,
    )
    # Non-canonical layouts are deliberately labelled dry-run.  The following
    # mutation is solely to exercise receipt validation/write guards with a
    # fully verified synthetic payload; it is never an operational receipt.
    payload["status"] = core.PREFLIGHT_PASS_STATUS
    return layout, payload


def test_temp_metadata_preflight_binds_all_pairs_and_48_checkpoint_shas(tmp_path: Path) -> None:
    layout = _fixture_layout(tmp_path)
    payload = core.build_metadata_preflight(
        layout=layout,
        verify_checkpoint_bytes=True,
        include_implementation_bindings=True,
    )

    assert payload["status"] == core.PREFLIGHT_DRY_RUN_STATUS
    assert payload["checkpoint_bytes_verified"] is True
    assert payload["cpu_forward_batch_contract"] == {
        **core.cpu_forward_batch_contract(),
        "contract_sha256": core.cpu_forward_batch_contract_sha256(),
    }
    assert payload["execution_scope"] == {
        "metadata_only": True,
        "torch_imported": False,
        "checkpoint_deserialized": False,
        "nwb_opened": False,
        "sealed_formal_test_sessions_opened": False,
        "gpu_used": False,
        "forward_probe_run": False,
        "training_run": False,
    }
    assert payload["canonical_scope"]["historical_t4_reference_qualification"].startswith(
        "SHA-qualified historical M30 T4 reference"
    )
    assert sum(
        len(payload["pairs"][str(seed)]["arms"][arm]["result"]["epoch_checkpoints"])
        for seed in core.SEEDS
        for arm in ("t4", "z4")
    ) == 48
    for seed in core.SEEDS:
        assert payload["pairs"][str(seed)]["pairing"]["same_unit_paired"] is True
        for arm in ("t4", "z4"):
            rows = payload["pairs"][str(seed)]["arms"][arm]["result"]["epoch_checkpoints"]
            assert all(row["checkpoint_bytes_verified"] is True for row in rows.values())


def test_metadata_preflight_rejects_z4_normalizer_or_same_unit_drift(tmp_path: Path) -> None:
    layout = _fixture_layout(tmp_path)
    z4_metadata = layout.metadata_paths[("z4", 42)]
    payload = _load_json(z4_metadata)
    payload["side_features"]["normalization_sha256"] = "b" * 64
    _write_json(z4_metadata, payload)
    with pytest.raises(core.A12AuditError, match="normalization_sha256 drift"):
        core.build_metadata_preflight(layout=layout, verify_checkpoint_bytes=False)

    layout = _fixture_layout(tmp_path / "same_unit")
    z4_metadata = layout.metadata_paths[("z4", 42)]
    payload = _load_json(z4_metadata)
    payload["session_unit_counts"][core.DEFAULT_VALIDATION_SESSIONS[0]] += 1
    _write_json(z4_metadata, payload)
    z4_result = layout.result_paths[("z4", 42)]
    result_payload = _load_json(z4_result)
    result_payload["run_metadata_sha256"] = core.sha256_file(z4_metadata)
    result_payload["session_unit_counts"] = payload["session_unit_counts"]
    _write_json(z4_result, result_payload)
    layout = replace(
        layout,
        expected_result_sha256={
            **layout.expected_result_sha256,
            ("z4", 42): core.sha256_file(z4_result),
        },
    )
    with pytest.raises(core.A12AuditError, match="same-unit pairing drift"):
        core.build_metadata_preflight(layout=layout, verify_checkpoint_bytes=False)


def test_immutable_receipt_uses_sidecar_readonly_and_exclusive_write(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    artifact = core.write_immutable_json(output, {"example": "a12"}, label="fixture A12 receipt")
    assert artifact["path"] == str(output.resolve())
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    sidecar = output.with_name(output.name + ".sha256")
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o444
    loaded = core.load_verified_immutable_json(output, label="fixture A12 receipt")
    assert loaded["example"] == "a12"
    with pytest.raises(core.A12AuditError, match="refusing to overwrite"):
        core.write_immutable_json(output, {"example": "replacement"}, label="fixture A12 receipt")


def test_receipt_requires_exact_implementation_binding_key_set(tmp_path: Path) -> None:
    layout, payload = _passed_fixture_payload(tmp_path)
    active = replace(layout, canonical=True)
    core.validate_metadata_preflight_receipt(payload, layout=active)

    missing = copy.deepcopy(payload)
    del missing["implementation_bindings"]["fixture_core"]
    with pytest.raises(core.A12AuditError, match="binding key set drift"):
        core.validate_metadata_preflight_receipt(missing, layout=active)


def test_metadata_preflight_rejects_cpu_forward_batch_contract_drift(tmp_path: Path) -> None:
    layout, payload = _passed_fixture_payload(tmp_path)
    active = replace(layout, canonical=True)
    drifted = copy.deepcopy(payload)
    drifted["cpu_forward_batch_contract"]["cpu_forward_batch_size"] = 32
    with pytest.raises(core.A12AuditError, match="CPU-forward batch contract.cpu_forward_batch_size drift"):
        core.validate_metadata_preflight_receipt(drifted, layout=active)

    drifted = copy.deepcopy(payload)
    drifted["cpu_forward_batch_contract"]["contract_sha256"] = "0" * 64
    with pytest.raises(core.A12AuditError, match="CPU-forward batch contract.contract_sha256 drift"):
        core.validate_metadata_preflight_receipt(drifted, layout=active)


def test_handoff_is_disclosed_but_not_an_executable_binding() -> None:
    layout = core.canonical_layout()
    assert "a12_protocol" in layout.implementation_paths
    assert "a12_handoff_claim" not in layout.implementation_paths
    assert len(layout.result_paths) == 6
    assert len(layout.metadata_paths) == 6


def test_preflight_operational_guards_reject_noncanonical_unverified_and_noncanonical_output(tmp_path: Path) -> None:
    preflight = _load_preflight_module()
    layout, payload = _passed_fixture_payload(tmp_path)

    with pytest.raises(core.A12AuditError, match="canonical artifact layout"):
        preflight.assert_operational_preflight_payload(payload, layout=layout)

    synthetic_canonical = replace(layout, canonical=True)
    unverified = copy.deepcopy(payload)
    unverified["checkpoint_bytes_verified"] = False
    with pytest.raises(core.A12AuditError, match="all 48 checkpoint SHA checks"):
        preflight.assert_operational_preflight_payload(unverified, layout=synthetic_canonical)

    with pytest.raises(core.A12AuditError, match="one canonical output path"):
        preflight.write_operational_receipt(
            payload,
            layout=synthetic_canonical,
            output_path=tmp_path / "not-the-canonical-output.json",
        )


def test_preflight_source_never_imports_torch_or_nwb_and_defaults_to_no_write() -> None:
    source_path = SUA_ROOT / "scripts" / "a12_descriptive_attention_preflight.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(name == "torch" or name.startswith("torch.") for name in imported)
    assert not any(name == "pynwb" or name.startswith("pynwb.") for name in imported)
    assert not any(name.startswith("src.models") for name in imported)

    preflight = _load_preflight_module()
    parsed = preflight.build_parser().parse_args([])
    assert parsed.write_operational_receipt is False
    assert preflight.canonical_receipt_path().name == "official_metadata_preflight.json"
    assert "a12_descriptive_attention_audit_v4" in str(preflight.canonical_receipt_path())
