from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
FINALIZER_PATH = ROOT / "sua_exploration/scripts/finalize_t4_paired_view_c1_evidence.py"


def _load_finalizer():
    spec = importlib.util.spec_from_file_location("c1_postrun_finalizer_test", FINALIZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FINALIZER = _load_finalizer()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _runtime(hostname: str, suffix: str) -> dict:
    uuid = f"GPU-fixture-{suffix}"
    return {
        "hostname": hostname,
        "platform": "fixture-linux",
        "python": "fixture-python",
        "cuda_visible_devices": "0",
        "pytorch": "fixture-torch",
        "pytorch_cuda": "fixture-cuda",
        "cuda_available": True,
        "torch_gpus": [
            {
                "logical_index": 0,
                "name": f"fixture GPU {suffix}",
                "uuid": uuid,
                "total_memory_bytes": 24_000_000_000,
            }
        ],
        "nvidia_smi": [f"0, fixture GPU {suffix}, {uuid}, fixture-driver, 24000 MiB"],
    }


def _make_fixture_aggregator(path: Path) -> None:
    source = r'''#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--receipt", type=Path, required=True)
parser.add_argument("--workspace", type=Path, required=True)
parser.add_argument("--result-root", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
receipt_sha = hashlib.sha256(args.receipt.read_bytes()).hexdigest()
payload = {
    "schema_version": 1,
    "status": "completed",
    "receipt_sha256": receipt_sha,
    "seeds": [42, 43, 44],
    "epochs": [5, 6, 7, 8, 9, 10, 11, 12],
    "sessions": ["fixture_s0", "fixture_s1", "fixture_s2", "fixture_s3", "fixture_s4", "fixture_s5"],
    "formal_test_used": False,
    "historical_c1_artifacts_used": False,
    "gates": {
        "sua_noninferiority": True,
        "pseudo_mua_noninferiority": True,
        "sua_correct_content_attachment": True,
        "pseudo_mua_correct_content_attachment": True,
        "cross_view_gap_not_increased": True,
    },
    "primary_contrasts": {
        "shared_t4_minus_separate_t4_sua": {},
        "shared_t4_minus_separate_t4_pseudo_mua": {},
        "shared_t4_minus_shared_ts4_sua": {},
        "shared_t4_minus_shared_ts4_pseudo_mua": {},
        "shared_minus_separate_absolute_cross_view_gap": {},
    },
    "c1_pass": True,
    "decision": "pass_enter_conditional_c2",
}
args.out.parent.mkdir(parents=True, exist_ok=True)
fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
'''
    _write_bytes(path, source.encode("utf-8"))


def _source_map(workspace: Path) -> dict[str, dict[str, str]]:
    aggregator = workspace / FINALIZER.AGGREGATOR_RELATIVE
    _make_fixture_aggregator(aggregator)
    files = [FINALIZER.AGGREGATOR_RELATIVE]
    for index in range(27):
        relative = Path(f"frozen_source/source_{index:02d}.txt")
        _write_bytes(workspace / relative, f"frozen fixture source {index}\n".encode("utf-8"))
        files.append(relative)
    assert len(files) == 28
    return {str(relative): {"sha256": _sha(workspace / relative)} for relative in files}


def _portable_manifest() -> dict:
    return {
        "schema_version": 1,
        "file_count": 33,
        "formal_test_file_paths": [],
        "formal_test_file_hashes": [],
        "file_inventory": {
            "train": [
                {
                    "filename": f"train_{index:02d}.nwb",
                    "bytes": index + 1,
                    "sha256": "a" * 64,
                }
                for index in range(27)
            ],
            "val": [
                {
                    "filename": f"val_{index:02d}.nwb",
                    "bytes": index + 1,
                    "sha256": "b" * 64,
                }
                for index in range(6)
            ],
        },
    }


def _fresh_matrix() -> list[dict]:
    views = {
        "separate_sua_t4": ("sua",),
        "separate_pseudo_mua_t4": ("pseudo_mua",),
        "shared_t4": ("sua", "pseudo_mua"),
        "shared_ts4": ("sua", "pseudo_mua"),
    }
    rows: list[dict] = []
    for cell, signal_views in views.items():
        for seed in (42, 43, 44):
            rows.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "logical_status": f"status/{cell}_s{seed}.complete.json",
                    "logical_closure_dir": f"closure/{cell}_s{seed}",
                    "logical_checkpoint_dir": f"checkpoints/{cell}_s{seed}",
                    "logical_artifacts": [
                        f"artifacts/{cell}_s{seed}_{view}.json"
                        for view in signal_views
                    ],
                }
            )
    assert len(rows) == 12
    return rows


def _copy_record(result_root: Path, source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    return {
        "source": str(source),
        "copy": str(destination),
        "copy_relative": str(destination.relative_to(result_root)),
        "bytes": destination.stat().st_size,
        "sha256": _sha(destination),
    }


def _write_cell(
    *,
    result_root: Path,
    row: dict,
    receipt_sha: str,
    teacher_sha: str,
    strict_sha: str,
    stamp_path: Path,
    runtime: dict,
) -> None:
    cell = row["cell"]
    seed = row["seed"]
    closure_relative = row["logical_closure_dir"]
    closure_dir = result_root / closure_relative
    closure_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = closure_dir / "run_metadata.json"
    _write_json(
        metadata_path,
        {
            "status": "completed",
            "seed": seed,
            "teacher_sha256": teacher_sha,
            "train_val_manifest_sha256": strict_sha,
            "held_out_test_evaluated": False,
        },
    )
    cost_path = closure_dir / "post_run_cost_receipt.json"
    _write_json(
        cost_path,
        {
            "fit_wall_clock_seconds": float(seed + 1),
            "cuda_peak_memory_allocated_bytes": 1_000_000 + seed,
            "cuda_peak_memory_reserved_bytes": 2_000_000 + seed,
        },
    )

    artifact_rows: list[dict] = []
    copied = [
        {
            "source": str(metadata_path),
            "copy": str(metadata_path),
            "copy_relative": str(metadata_path.relative_to(result_root)),
            "bytes": metadata_path.stat().st_size,
            "sha256": _sha(metadata_path),
        },
        {
            "source": str(cost_path),
            "copy": str(cost_path),
            "copy_relative": str(cost_path.relative_to(result_root)),
            "bytes": cost_path.stat().st_size,
            "sha256": _sha(cost_path),
        },
    ]
    for relative in row["logical_artifacts"]:
        artifact = result_root / relative
        _write_json(
            artifact,
            {
                "fixture_only": True,
                "cell": cell,
                "seed": seed,
                "signal_view": Path(relative).stem.rsplit("_", 1)[-1],
            },
        )
        artifact_rows.append(
            {
                "path": str(artifact),
                "relative": relative,
                "sha256": _sha(artifact),
            }
        )
        copied.append(_copy_record(result_root, artifact, closure_dir / artifact.name))

    closure_path = closure_dir / "closure_manifest.json"
    _write_json(
        closure_path,
        {
            "schema_version": 1,
            "status": "completed",
            "cell": cell,
            "seed": seed,
            "program_receipt_sha256": receipt_sha,
            "cell_receipt_sha256": receipt_sha,
            "fresh_control_finalization_only": False,
            "files": copied,
            "runtime_environment": runtime,
            "student_parameter_count": 1234,
            "fp32_student_weight_bytes": 4936,
            "formal_sua_files_opened": False,
        },
    )

    started_path = result_root / f"status/{cell}_s{seed}.started.json"
    _write_json(
        started_path,
        {
            "schema_version": 1,
            "status": "started",
            "cell": cell,
            "seed": seed,
            "receipt_path": "/fixture/prelaunch/receipt.json",
            "program_receipt_sha256": receipt_sha,
            "cell_receipt_path": "/fixture/prelaunch/receipt.json",
            "cell_receipt_sha256": receipt_sha,
            "data_verification_stamp": f"/fixture/stage/{stamp_path.name}",
            "data_verification_stamp_sha256": _sha(stamp_path),
            "runtime_environment": runtime,
            "formal_sua_files_opened": False,
        },
    )
    completed_path = result_root / f"status/{cell}_s{seed}.complete.json"
    _write_json(
        completed_path,
        {
            "schema_version": 1,
            "status": "completed",
            "cell": cell,
            "seed": seed,
            "started_status": str(started_path),
            "started_status_sha256": _sha(started_path),
            "program_receipt_sha256": receipt_sha,
            "cell_receipt_path": "/fixture/prelaunch/receipt.json",
            "cell_receipt_sha256": receipt_sha,
            "checkpoint_dir": f"/fixture/checkpoints/{cell}_s{seed}",
            "metadata_sha256": _sha(metadata_path),
            "cost_sha256": _sha(cost_path),
            "student_parameter_count": 1234,
            "fp32_student_weight_bytes": 4936,
            "artifacts": artifact_rows,
            "closure_manifest": str(closure_path),
            "closure_relative": str(closure_path.relative_to(result_root)),
            "closure_manifest_sha256": _sha(closure_path),
            "formal_sua_files_opened": False,
        },
    )


def _remote_transfer_manifest(result_root: Path, remote_stamp: Path) -> Path:
    cells = ("separate_sua_t4", "separate_pseudo_mua_t4", "shared_t4", "shared_ts4")
    records: list[dict] = []

    def add(path: Path, role: str, cell: str | None = None) -> None:
        relative = str(path.relative_to(result_root))
        record = {
            "role": role,
            "source_relative": f"remote_stage/{relative}",
            "destination_relative": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        if cell is not None:
            record["cell"] = cell
            record["seed"] = 44
        records.append(record)

    for cell in cells:
        add(result_root / f"status/{cell}_s44.started.json", "started_status", cell)
        add(result_root / f"status/{cell}_s44.complete.json", "complete_status", cell)
        closure = result_root / f"closure/{cell}_s44"
        add(closure / "closure_manifest.json", "closure_manifest", cell)
        closure_manifest = json.loads((closure / "closure_manifest.json").read_text(encoding="utf-8"))
        for copy in closure_manifest["files"]:
            add(result_root / copy["copy_relative"], "closure_file", cell)
        completed = json.loads(
            (result_root / f"status/{cell}_s44.complete.json").read_text(encoding="utf-8")
        )
        for artifact in completed["artifacts"]:
            add(result_root / artifact["relative"], "artifact", cell)
    add(remote_stamp, "runtime_verification")
    path = result_root / "transfer/seed44_manifest.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "status": "completed",
            "remote_host": "remote-fixture",
            "source_result_root": "/fixture/remote/result",
            "destination_result_root": str(result_root),
            "files": records,
        },
    )
    return path


def _fixture(tmp_path: Path) -> dict[str, Path | str]:
    workspace = tmp_path / "fixture_workspace"
    result_root = workspace / "results/c1_v3_full_fresh"
    result_root.mkdir(parents=True)
    source_map = _source_map(workspace)
    teacher = workspace / "checkpoints/teacher.ckpt"
    _write_bytes(teacher, b"fixture teacher\n")
    strict = workspace / "sua_exploration/configs/strict.json"
    _write_json(strict, {"fixture": "strict"})
    data_audit = workspace / "audit/data.json"
    _write_json(
        data_audit,
        {
            "status": "passed",
            "session_count": 33,
            "formal_sua_files_opened": False,
            "formal_sua_paths_resolved": False,
        },
    )
    cost_audit = workspace / "audit/cost.json"
    _write_json(
        cost_audit,
        {
            "status": "passed",
            "neural_data_files_opened": 0,
            "formal_sua_files_opened": False,
            "teacher": {"sha256": _sha(teacher)},
        },
    )
    microfit_audit = workspace / "audit/ts4_microfit.json"
    microfit_sources = {
        relative: row["sha256"]
        for relative, row in sorted(source_map.items())[:3]
    }
    _write_json(
        microfit_audit,
        {
            "schema_version": 1,
            "audit": "fixture_v3_ts4_microfit",
            "status": "passed",
            "seed": 42,
            "formal_sua_files_opened": False,
            "formal_sua_paths_resolved": False,
            "held_out_test_evaluated": False,
            "no_r2_or_scorer_output_read": True,
            "source_hashes": microfit_sources,
            "data_access": {
                "strict_train_val_manifest": {
                    "sha256": _sha(strict),
                    "permitted_file_count": 33,
                    "formal_test_paths_resolved": False,
                }
            },
            "ts4_datamodules": {
                "sua": {
                    "normalizer": {
                        "signal_view": "sua",
                        "resolved_raw_group": "t4",
                        "same_view_t4_ts4_equal": True,
                        "normalization_scope": "source_train_27_only",
                    }
                },
                "pseudo_mua": {
                    "normalizer": {
                        "signal_view": "pseudo_mua",
                        "resolved_raw_group": "t4",
                        "same_view_t4_ts4_equal": True,
                        "normalization_scope": "source_train_27_only",
                    }
                },
                "cross_view_normalizers_distinct": True,
            },
            "permutation_contract": {
                "seeds_checked": [42, 43, 44],
                "derived_without_reopening_data": True,
            },
            "one_gpu_paired_microfit": {
                "paired_batches_consumed": 1,
                "optimizer_steps": 1,
                "evaluation_or_scoring_invoked": False,
            },
            "temporary_checkpoint_round_trip": {
                "reloaded_student_parameter_count": 4_613_178,
                "deleted_after_success": True,
                "path_absent_after_cleanup": True,
            },
        },
    )
    receipt_dir = workspace / "receipts/c1_v3_full_fresh"
    manifest_path = receipt_dir / "c1_train_val_33_manifest.json"
    _write_json(manifest_path, _portable_manifest())
    receipt_path = receipt_dir / "receipt.json"
    receipt = {
        "schema_version": 1,
        "prelaunch_id": "fixture_c1_v3_full_fresh_dynamic_id",
        "status": "prelaunch_only_no_gpu_authorization",
        "formal_sua_files_opened": False,
        "formal_sua_paths_resolved": False,
        "historical_c1_artifacts_used": False,
        "frozen_protocol": FINALIZER.EXPECTED_FROZEN_PROTOCOL,
        "aggregation": FINALIZER.EXPECTED_AGGREGATION,
        "source_map": source_map,
        "teacher": {"path": str(teacher), "sha256": _sha(teacher)},
        "strict_loader_manifest": {
            "path": str(strict.relative_to(workspace)),
            "sha256": _sha(strict),
        },
        "portable_data_manifest": {
            "filename": manifest_path.name,
            "sha256": _sha(manifest_path),
            "file_count": 33,
            "formal_paths_resolved": False,
        },
        "cpu_gate_receipts": {
            "paired_33_session_data_audit": {
                "path": str(data_audit.relative_to(workspace)),
                "sha256": _sha(data_audit),
            },
            "exact_model_parameter_mac_state_audit": {
                "path": str(cost_audit.relative_to(workspace)),
                "sha256": _sha(cost_audit),
            },
            "v3_ts4_real_data_microfit_audit": {
                "path": str(microfit_audit.relative_to(workspace)),
                "sha256": _sha(microfit_audit),
                "status": "passed",
                "seed": 42,
                "paired_batches_consumed": 1,
                "optimizer_steps": 1,
                "reloaded_student_parameter_count": 4_613_178,
                "formal_sua_files_opened": False,
                "formal_sua_paths_resolved": False,
                "no_r2_or_scorer_output_read": True,
            },
        },
        "fresh_matrix": _fresh_matrix(),
    }
    _write_json(receipt_path, receipt)
    receipt_sha = _sha(receipt_path)

    local_runtime = _runtime("local-fixture", "local")
    remote_runtime = _runtime("remote-fixture", "remote")
    local_stamp = result_root / "runtime_verification/local/data_fixture.json"
    remote_stamp = result_root / "runtime_verification/remote_seed44/data_fixture.json"
    for stamp_path, runtime in ((local_stamp, local_runtime), (remote_stamp, remote_runtime)):
        _write_json(
            stamp_path,
            {
                "schema_version": 1,
                "status": "passed",
                "hostname": runtime["hostname"],
                "data_manifest_sha256": _sha(manifest_path),
                "verifier_result": {
                    "receipt_sha256": receipt_sha,
                    "source_count": len(source_map),
                    "teacher_sha256": _sha(teacher),
                    "data_manifest_sha256": _sha(manifest_path),
                    "data_content_verified": True,
                    "formal_sua_files_opened": False,
                },
            },
        )
    for row in receipt["fresh_matrix"]:
        _write_cell(
            result_root=result_root,
            row=row,
            receipt_sha=receipt_sha,
            teacher_sha=_sha(teacher),
            strict_sha=_sha(strict),
            stamp_path=remote_stamp if row["seed"] == 44 else local_stamp,
            runtime=remote_runtime if row["seed"] == 44 else local_runtime,
        )
    transfer = _remote_transfer_manifest(result_root, remote_stamp)
    return {
        "workspace": workspace,
        "result_root": result_root,
        "teacher": teacher,
        "receipt": receipt_path,
        "receipt_sha": receipt_sha,
        "transfer": transfer,
        "microfit": microfit_audit,
        "aggregate_out": result_root / "finalization/aggregate.json",
        "finalization_out": result_root / "finalization/publication_readiness.json",
    }


def _finalize(fixture: dict[str, Path | str]) -> dict:
    return FINALIZER.finalize_evidence(
        receipt_path=Path(fixture["receipt"]),
        workspace=Path(fixture["workspace"]),
        result_root=Path(fixture["result_root"]),
        teacher_path=Path(fixture["teacher"]),
        remote_transfer_manifest=Path(fixture["transfer"]),
        aggregate_out=Path(fixture["aggregate_out"]),
        finalization_out=Path(fixture["finalization_out"]),
        python_executable=Path(sys.executable),
        expected_receipt_sha=str(fixture["receipt_sha"]),
    )


def _refresh_microfit_receipt_binding(fixture: dict[str, Path | str]) -> None:
    """Rebind a deliberately mutated synthetic microfit before finalizer entry.

    The expected caller receipt hash changes with this fixture-only mutation.
    Its cell evidence deliberately remains bound to the old hash because every
    negative test below must fail at the microfit gate *before* status/evidence
    validation or aggregation can be reached.
    """
    receipt_path = Path(fixture["receipt"])
    microfit_path = Path(fixture["microfit"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["cpu_gate_receipts"]["v3_ts4_real_data_microfit_audit"]["sha256"] = _sha(
        microfit_path
    )
    _write_json(receipt_path, receipt)
    fixture["receipt_sha"] = _sha(receipt_path)


def test_dynamic_full_fresh_fixture_finalizes_without_old_control_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _finalize(fixture)

    receipt = json.loads(Path(fixture["finalization_out"]).read_text(encoding="utf-8"))
    assert result["c1_pass"] is True
    assert receipt["status"] == "finalized"
    assert receipt["frozen_contract"]["source_map_count"] == 28
    assert receipt["exact_status_closure"]["expected_started"] == 12
    assert receipt["exact_status_closure"]["expected_completed"] == 12
    assert len(receipt["exact_status_closure"]["cells"]) == 12
    assert len(receipt["runtime_host_groups"]) == 2
    assert receipt["remote_seed44_transfer"]["remote_host"] == "remote-fixture"
    assert receipt["frozen_contract"]["v3_ts4_microfit_audit_sha256"] == _sha(
        Path(fixture["microfit"])
    )
    assert receipt["frozen_contract"]["v3_ts4_microfit_source_count"] == 3
    assert "control_receipt_sha256" not in receipt["frozen_contract"]
    assert "absolute_r2" not in receipt
    with pytest.raises(FINALIZER.FinalizationError, match="write-once aggregate already exists"):
        _finalize(fixture)


def test_source_map_drift_fails_before_aggregation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    drifted = Path(fixture["workspace"]) / "frozen_source/source_00.txt"
    _write_bytes(drifted, b"tampered source\n")

    with pytest.raises(FINALIZER.FinalizationError, match="source hash drift"):
        _finalize(fixture)
    assert not Path(fixture["aggregate_out"]).exists()
    assert not Path(fixture["finalization_out"]).exists()


def test_missing_remote_runtime_stamp_manifest_entry_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    transfer_path = Path(fixture["transfer"])
    transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
    transfer["files"] = [
        row for row in transfer["files"] if row["role"] != "runtime_verification"
    ]
    _write_json(transfer_path, transfer)

    with pytest.raises(
        FINALIZER.FinalizationError,
        match="omitted runtime-verification stamp",
    ):
        _finalize(fixture)
    assert not Path(fixture["aggregate_out"]).exists()


def test_extra_status_fails_before_any_aggregation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _write_json(
        Path(fixture["result_root"]) / "status/stale_s99.complete.json",
        {"status": "completed"},
    )

    with pytest.raises(FINALIZER.FinalizationError, match="status set is not exact"):
        _finalize(fixture)
    assert not Path(fixture["aggregate_out"]).exists()


def test_microfit_source_hash_drift_fails_before_any_aggregation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    microfit_path = Path(fixture["microfit"])
    microfit = json.loads(microfit_path.read_text(encoding="utf-8"))
    source = next(iter(microfit["source_hashes"]))
    microfit["source_hashes"][source] = "0" * 64
    _write_json(microfit_path, microfit)
    _refresh_microfit_receipt_binding(fixture)

    with pytest.raises(
        FINALIZER.FinalizationError, match="v3 TS4 microfit source hash drift"
    ):
        _finalize(fixture)
    assert not Path(fixture["aggregate_out"]).exists()
    assert not Path(fixture["finalization_out"]).exists()


def test_microfit_semantic_gate_drift_fails_before_any_aggregation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    microfit_path = Path(fixture["microfit"])
    microfit = json.loads(microfit_path.read_text(encoding="utf-8"))
    microfit["one_gpu_paired_microfit"]["optimizer_steps"] = 2
    _write_json(microfit_path, microfit)
    _refresh_microfit_receipt_binding(fixture)

    with pytest.raises(
        FINALIZER.FinalizationError, match="v3 TS4 real-data microfit gate failed"
    ):
        _finalize(fixture)
    assert not Path(fixture["aggregate_out"]).exists()
    assert not Path(fixture["finalization_out"]).exists()
