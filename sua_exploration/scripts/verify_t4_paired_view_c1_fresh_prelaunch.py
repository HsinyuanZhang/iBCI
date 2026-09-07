#!/usr/bin/env python3
"""Verify the immutable fresh paired-view C1 prelaunch on any execution host."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_PRELAUNCH_ID = "t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist"
EXPECTED_SUPERSEDED_RECEIPT_SHA = (
    "b7a3ebc39d195b8b559cf21517826b9a8bb49c06103c1a0fede4adad23b7cab1"
)
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_CELLS = (
    "separate_sua_t4",
    "separate_pseudo_mua_t4",
    "shared_t4",
    "shared_ts4",
)


class PrelaunchVerificationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise PrelaunchVerificationError(message)


def _load(path: Path) -> dict[str, Any]:
    _need(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _need(isinstance(value, dict), f"JSON object required: {path}")
    return value


def verify_prelaunch(
    receipt_path: Path,
    *,
    workspace: Path,
    teacher_path: Path | None = None,
    data_dir: Path | None = None,
    verify_data_content: bool = False,
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    workspace = workspace.expanduser().resolve()
    receipt = _load(receipt_path)
    _need(receipt.get("prelaunch_id") == EXPECTED_PRELAUNCH_ID, "prelaunch id drift")
    _need(
        receipt.get("status") == "prelaunch_only_no_gpu_authorization",
        "receipt is not an immutable prelaunch-only package",
    )
    _need(receipt.get("historical_c1_artifacts_used") is False, "historical C1 bridge forbidden")
    _need(receipt.get("formal_sua_files_opened") is False, "formal access flag drift")
    _need(receipt.get("formal_sua_paths_resolved") is False, "formal path-resolution drift")
    supersedes = receipt.get("supersedes") or {}
    _need(
        supersedes.get("receipt_sha256") == EXPECTED_SUPERSEDED_RECEIPT_SHA
        and supersedes.get("scientific_protocol_changed") is False
        and supersedes.get("model_or_training_changed") is False
        and supersedes.get("data_or_metric_changed") is False
        and supersedes.get("input_pipeline_implementation_repaired") is False
        and supersedes.get("remote_gate_allowlist_repaired") is True
        and supersedes.get("repair_scope")
        == "remote_allowlist_adds_all_three_receipt_pinned_cpu_gate_files"
        and supersedes.get("completed_training_reused") == []
        and supersedes.get("scientific_outputs_reused") == []
        and supersedes.get("partial_started_status_files_preserved") is True,
        "v3r2 remote-gate-allowlist supersession contract drift",
    )
    protocol = receipt.get("frozen_protocol") or {}
    _need(
        (
            protocol.get("source_training_activity_calibration_n"),
            protocol.get("evaluation_forward_calibration_n"),
            protocol.get("t4_label_rate_pool_n"),
            protocol.get("evaluation_pool_n"),
            protocol.get("evaluation_start_trial"),
            protocol.get("total_epochs"),
            protocol.get("epoch_window"),
            protocol.get("seeds"),
            protocol.get("shared_objective"),
            protocol.get("lambda_consistency"),
            protocol.get("shared_backward"),
            protocol.get("decoder_training"),
            protocol.get("development_heldout_use"),
            protocol.get("development_heldout_backprop"),
            protocol.get("formal_test_use"),
        )
        == (
            10,
            30,
            50,
            50,
            50,
            12,
            list(range(5, 13)),
            list(EXPECTED_SEEDS),
            "0.5*L_task(SUA)+0.5*L_task(pseudo_MUA)",
            0.0,
            "sequential_half_weight_backward_then_single_optimizer_step",
            "jointly_trained_offline_on_source_train_27_only",
            "forward_only_fixed_epoch_window_scoring",
            False,
            "sealed_not_resolved_not_opened",
        ),
        "frozen C1 protocol drift",
    )
    cells = receipt.get("fresh_matrix")
    _need(isinstance(cells, list) and len(cells) == 12, "C1 v3 matrix must contain 12 fresh cells")
    observed = {(row.get("cell"), row.get("seed")) for row in cells if isinstance(row, dict)}
    expected = {(cell, seed) for cell in EXPECTED_CELLS for seed in EXPECTED_SEEDS}
    _need(observed == expected, "C1 cell/seed matrix drift")
    all_paths: list[str] = []
    for row in cells:
        _need(isinstance(row, dict), "malformed C1 matrix row")
        cell = str(row.get("cell"))
        seed = int(row.get("seed"))
        expected_artifacts = (
            [f"artifacts/{cell}_s{seed}.json"]
            if cell.startswith("separate_")
            else [
                f"artifacts/{cell}_s{seed}_sua.json",
                f"artifacts/{cell}_s{seed}_pseudo_mua.json",
            ]
        )
        expected_paths = {
            "logical_checkpoint_dir": f"checkpoints/{EXPECTED_PRELAUNCH_ID}_{cell}_s{seed}",
            "logical_status": f"status/{cell}_s{seed}.complete.json",
            "logical_closure_dir": f"closure/{cell}_s{seed}",
            "logical_artifacts": expected_artifacts,
        }
        for key, value in expected_paths.items():
            _need(row.get(key) == value, f"C1 canonical matrix path drift: {cell}/s{seed}/{key}")
        all_paths.extend(
            [
                expected_paths["logical_checkpoint_dir"],
                expected_paths["logical_status"],
                expected_paths["logical_closure_dir"],
                *expected_artifacts,
            ]
        )
    _need(len(all_paths) == len(set(all_paths)), "C1 canonical matrix paths are not unique")
    _need(
        sum(len(row["logical_artifacts"]) for row in cells) == 18,
        "C1 v3 matrix must contain exactly 18 logical view artifacts",
    )

    source_map = receipt.get("source_map")
    _need(isinstance(source_map, dict) and source_map, "source map missing")
    for relative, row in source_map.items():
        _need(isinstance(row, dict), f"malformed source row: {relative}")
        path = (workspace / relative).resolve()
        _need(path.is_file() and path.is_relative_to(workspace), f"source missing/escaped: {relative}")
        _need(sha256_file(path) == row.get("sha256"), f"source hash drift: {relative}")

    strict = receipt.get("strict_loader_manifest") or {}
    strict_path = workspace / "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
    _need(strict_path.is_file(), "strict loader manifest missing")
    _need(sha256_file(strict_path) == strict.get("sha256"), "strict manifest hash drift")

    teacher = receipt.get("teacher") or {}
    selected_teacher = (
        teacher_path.expanduser().resolve()
        if teacher_path is not None
        else (workspace / str(teacher.get("path", ""))).expanduser().resolve()
    )
    _need(selected_teacher.is_file(), f"teacher missing: {selected_teacher}")
    _need(sha256_file(selected_teacher) == teacher.get("sha256"), "teacher hash drift")

    manifest_row = receipt.get("portable_data_manifest") or {}
    manifest_path = receipt_path.parent / str(manifest_row.get("filename", ""))
    _need(manifest_path.is_file(), "portable 33-file manifest missing")
    _need(sha256_file(manifest_path) == manifest_row.get("sha256"), "portable manifest hash drift")
    manifest = _load(manifest_path)
    _need(manifest.get("file_count") == 33, "portable manifest must list exactly 33 files")
    _need(manifest.get("formal_test_file_paths") == [], "formal paths present in data manifest")
    _need(manifest.get("formal_test_file_hashes") == [], "formal hashes present in data manifest")
    inventory = manifest.get("file_inventory") or {}
    _need(
        len(inventory.get("train", [])) == 27 and len(inventory.get("val", [])) == 6,
        "portable inventory is not 27+6",
    )
    if verify_data_content:
        _need(data_dir is not None, "--verify-data-content requires --data-dir")
        selected_data = data_dir.expanduser().resolve()
        for split in ("train", "val"):
            for row in inventory[split]:
                path = (selected_data / str(row["filename"])).resolve()
                _need(path.parent == selected_data and path.is_file(), f"data file missing: {path}")
                _need(path.stat().st_size == row["bytes"], f"data size drift: {path}")
                _need(sha256_file(path) == row["sha256"], f"data hash drift: {path}")

    gates = receipt.get("cpu_gate_receipts") or {}
    data_audit_ref = gates.get("paired_33_session_data_audit") or {}
    data_audit_path = workspace / str(data_audit_ref.get("path", ""))
    _need(data_audit_path.is_file(), "33-session data audit missing")
    _need(sha256_file(data_audit_path) == data_audit_ref.get("sha256"), "data audit drift")
    data_audit = _load(data_audit_path)
    _need(
        data_audit.get("status") == "passed"
        and data_audit.get("session_count") == 33
        and data_audit.get("formal_sua_files_opened") is False
        and data_audit.get("formal_sua_paths_resolved") is False
        and data_audit.get("normalizer_hashes_distinct") is True
        and data_audit.get("max_neural_pool_error") == 0.0
        and float(data_audit.get("max_calibration_pool_error", float("inf"))) <= 1.0e-4
        and float(data_audit.get("max_pooled_rate_t4_error", float("inf"))) <= 2.0e-3,
        "33-session data audit gate failed",
    )

    cost_ref = gates.get("exact_model_parameter_mac_state_audit") or {}
    cost_path = workspace / str(cost_ref.get("path", ""))
    _need(cost_path.is_file(), "model parameter/MAC/state audit missing")
    _need(sha256_file(cost_path) == cost_ref.get("sha256"), "model cost audit drift")
    cost = _load(cost_path)
    parameter = cost.get("exact_parameter_receipt") or {}
    mac = cost.get("configured_mac_receipt") or {}
    state = cost.get("fp32_state_receipt_reference_n64") or {}
    no_backprop = cost.get("no_heldout_backprop_contract") or {}
    _need(
        cost.get("status") == "passed"
        and cost.get("neural_data_files_opened") == 0
        and cost.get("formal_sua_files_opened") is False
        and int(parameter.get("student_parameter_count", 0)) > 0
        and parameter.get("student_parameter_count") == parameter.get("trainable_parameter_count")
        and int(mac.get("encoder_macs_per_activity_calibration_session", 0)) > 0
        and int(mac.get("decoder_macs_per_online_window", 0)) > 0
        and int(state.get("persistent_t4_descriptor_bytes", 0)) > 0
        and int(state.get("persistent_identity_E_bytes", 0)) > 0
        and state.get("deployment_extra_state_shared_vs_one_separate_model_bytes") == 0
        and no_backprop.get("optimizer_and_backward_scope") == "source_train_27_only"
        and no_backprop.get("development_enters_train_dataloader") is False
        and no_backprop.get("development_enters_loss") is False
        and no_backprop.get("development_enters_optimizer") is False
        and no_backprop.get("development_uses_backward_gradients") is False
        and no_backprop.get("formal_paths_resolved") is False
        and no_backprop.get("formal_files_opened") is False,
        "model cost/no-heldout-backprop gate failed",
    )
    _need(cost_ref.get("exact_parameter_receipt") == parameter, "embedded parameter receipt drift")
    _need(cost_ref.get("configured_mac_receipt") == mac, "embedded MAC receipt drift")
    _need(cost_ref.get("fp32_state_receipt_reference_n64") == state, "embedded state receipt drift")
    _need(cost_ref.get("no_heldout_backprop_contract") == no_backprop, "embedded heldout contract drift")

    microfit_ref = gates.get("v3_ts4_real_data_microfit_audit") or {}
    microfit_path = workspace / str(microfit_ref.get("path", ""))
    _need(
        microfit_path.is_file() and microfit_path.resolve().is_relative_to(workspace),
        "v3 TS4 microfit audit missing/escaped",
    )
    _need(
        sha256_file(microfit_path) == microfit_ref.get("sha256"),
        "v3 TS4 microfit audit drift",
    )
    microfit = _load(microfit_path)
    one_step = microfit.get("one_gpu_paired_microfit") or {}
    checkpoint = microfit.get("temporary_checkpoint_round_trip") or {}
    datamodules = microfit.get("ts4_datamodules") or {}
    permutation = microfit.get("permutation_contract") or {}
    permitted = (microfit.get("data_access") or {}).get(
        "strict_train_val_manifest"
    ) or {}
    _need(
        microfit.get("status") == "passed"
        and microfit.get("seed") == 42
        and microfit.get("formal_sua_files_opened") is False
        and microfit.get("formal_sua_paths_resolved") is False
        and microfit.get("held_out_test_evaluated") is False
        and microfit.get("no_r2_or_scorer_output_read") is True
        and permitted.get("permitted_file_count") == 33
        and permitted.get("formal_test_paths_resolved") is False
        and one_step.get("paired_batches_consumed") == 1
        and one_step.get("optimizer_steps") == 1
        and one_step.get("evaluation_or_scoring_invoked") is False
        and datamodules.get("cross_view_normalizers_distinct") is True
        and permutation.get("seeds_checked") == list(EXPECTED_SEEDS)
        and permutation.get("derived_without_reopening_data") is True
        and checkpoint.get("reloaded_student_parameter_count") == 4_613_178
        and checkpoint.get("deleted_after_success") is True
        and checkpoint.get("path_absent_after_cleanup") is True,
        "v3 TS4 real-data microfit gate failed",
    )
    for relative, expected_sha in (microfit.get("source_hashes") or {}).items():
        _need(
            relative in source_map
            and source_map[relative].get("sha256") == expected_sha,
            f"v3 TS4 microfit source drift: {relative}",
        )
    _need(
        microfit_ref.get("status") == "passed"
        and microfit_ref.get("seed") == 42
        and microfit_ref.get("paired_batches_consumed") == 1
        and microfit_ref.get("optimizer_steps") == 1
        and microfit_ref.get("reloaded_student_parameter_count") == 4_613_178
        and microfit_ref.get("formal_sua_files_opened") is False
        and microfit_ref.get("formal_sua_paths_resolved") is False
        and microfit_ref.get("no_r2_or_scorer_output_read") is True,
        "embedded v3 TS4 microfit receipt drift",
    )

    return {
        "receipt_sha256": sha256_file(receipt_path),
        "source_count": len(source_map),
        "cell_count": len(cells),
        "fresh_separate_control_cell_count": sum(
            1 for row in cells if str(row.get("cell", "")).startswith("separate_")
        ),
        "independent_control_cell_count": 0,
        "teacher_sha256": teacher["sha256"],
        "data_manifest_sha256": manifest_row["sha256"],
        "data_content_verified": verify_data_content,
        "formal_sua_files_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--verify-data-content", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_prelaunch(
            args.receipt,
            workspace=args.workspace,
            teacher_path=args.teacher,
            data_dir=args.data_dir,
            verify_data_content=args.verify_data_content,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, PrelaunchVerificationError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "passed", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
