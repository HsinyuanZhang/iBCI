from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "aggregate_m1_version_b_pilot.py"
)
SPEC = importlib.util.spec_from_file_location("aggregate_m1_version_b_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


SOURCE_SESSIONS = ["ses-20120926", "ses-20120927", "ses-20120928"]
TARGET_SESSION = "ses-20120924"
ARM_INFO = {
    "hs": ("m1_version_b_hs_continuation", "none", "B0", 0.50),
    "bc0": ("m1_version_b_c0", "zero4", "B3S", 0.53),
    "bc": ("m1_version_b_c", "full", "B3S", 0.57),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _resolved_config(
    experiment: str, arm: str, variant: str, teacher: Path
) -> dict[str, object]:
    model: dict[str, object] = {
        "_target_": "fake.StreamingCalibrationLitModule",
        "variant": variant,
        "teacher_ckpt_path": str(teacher),
        "freeze_decoder": False,
        "loss_mode": "task_only",
        "lambda_y": 0.0,
        "lambda_E": 0.0,
    }
    if variant == "B3S":
        model.update({"hidden_dim": 64, "side_dim": 4})
    return {
        "task_name": experiment,
        "seed": 42,
        "train": True,
        "test": True,
        "no_early_stopping": True,
        "optimized_metric": "test_heldin/r2_mean",
        "data": {
            "task": "m1",
            "loso_fold": 0,
            "source_session_names": SOURCE_SESSIONS,
            "calibration_n_trials": 10,
            "random_calibration": False,
            "heldin_query_start_trial": 10,
            "heldin_query_end_trial": 210,
            "include_heldout_in_fit": False,
            "include_heldout_in_test": False,
            "validation_protocol": "loso",
            "afc4_arm": arm,
            "batch_size": 32,
            "sampler_seed": 42,
            "balance_session_batches": False,
            "reshuffle_train_sampler_each_epoch": False,
        },
        "model": model,
        "trainer": {
            "min_epochs": 12,
            "max_epochs": 12,
            "limit_val_batches": 0,
            "num_sanity_val_steps": 0,
        },
        "callbacks": {
            "early_stopping": None,
            "fixed_last_checkpoint": {
                "monitor": None,
                "save_last": False,
                "save_top_k": -1,
                "every_n_epochs": 12,
            },
        },
    }


def _split_manifest(
    arm: str, source_records: dict[str, dict[str, str]]
) -> dict[str, object]:
    return {
        "schema": "m1_version_b_source_loso_v1",
        "task": "m1",
        "outer_fold": 0,
        "outer_left_out": TARGET_SESSION,
        "train_sessions": SOURCE_SESSIONS,
        "validation_sessions": [TARGET_SESSION],
        "source_only": True,
        "formal": False,
        "evalai": False,
        "heldout_files_opened": False,
        "heldout_values_used": False,
        "minival_files_opened": False,
        "minival_values_used": False,
        "target_query_values_read_by_validation_or_evaluator": True,
        "target_query_values_used_for_optimizer_or_checkpoint_selection": False,
        "carrier_arm": arm,
        "support_trials": [0, 10],
        "query_trials": [10, 210],
        "calibration_n_trials": 10,
        "checkpoint_selection": "fixed_last_epoch_11_train_source_only",
        "target_backpropagation": False,
        "source_files": {name: source_records[name] for name in SOURCE_SESSIONS},
        "target_file": source_records[TARGET_SESSION],
        "query_window_audit": {
            TARGET_SESSION: {
                "total_trials": 300,
                "support_trials": 10,
                "query_start_trial": 10,
                "query_end_trial": 210,
                "query_trials": 200,
                "raw_query_start_bin": 1000,
                "minimum_window_start_padded_bin": 1100,
                "maximum_window_start_padded_bin": 30_000,
                "window_size": 100,
                "eligible_windows": 26_517,
                "full_window_disjoint": True,
                "ineligible_reason": None,
            }
        },
        "train_batch_counts": {name: 100 + index for index, name in enumerate(SOURCE_SESSIONS)},
        "query_batch_count": 828,
    }


def _make_fixture(tmp_path: Path) -> tuple[dict[str, Path], Path]:
    data_dir = tmp_path / "held-in-calib"
    data_dir.mkdir()
    source_records: dict[str, dict[str, str]] = {}
    for session in [TARGET_SESSION, *SOURCE_SESSIONS]:
        path = data_dir / f"sub-MonkeyL-held-in-calib_{session}_behavior+ecephys.nwb"
        path.write_bytes(f"synthetic-data-{session}".encode())
        source_records[session] = {"path": str(path), "sha256": _sha(path)}

    teacher = tmp_path / "teacher.ckpt"
    teacher.write_bytes(b"synthetic trusted teacher")
    teacher_sha = _sha(teacher)
    common_source_manifest = {
        "src/data/m1_version_b_source_loso_datamodule.py": "a" * 64,
        "src/models/streaming_calibration_module.py": "b" * 64,
    }
    run_dirs: dict[str, Path] = {}
    for key, (experiment, arm, variant, score) in ARM_INFO.items():
        run_dir = tmp_path / f"run_{key}"
        (run_dir / "checkpoints").mkdir(parents=True)
        run_dirs[key] = run_dir
        config = _resolved_config(experiment, arm, variant, teacher)
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
        )
        _json(run_dir / "split_manifest.json", _split_manifest(arm, source_records))
        _json(run_dir / "source_manifest.json", common_source_manifest)
        _json(
            run_dir / "teacher_metadata.json",
            {
                "teacher_checkpoint_path": str(teacher),
                "teacher_checkpoint_sha256": teacher_sha,
            },
        )
        checkpoint = run_dir / "checkpoints" / "best.ckpt"
        checkpoint.write_bytes(f"terminal-{key}".encode())
        checkpoint_sha = _sha(checkpoint)
        _json(
            run_dir / "checkpoint_manifest.json",
            {
                "source_checkpoint_path": f"/synthetic/{key}/epoch_epoch=011.ckpt",
                "source_checkpoint_sha256": checkpoint_sha,
                "selected_by_metric": "val_heldin/r2_mean",
                "selected_metric_value": None,
                "artifact_checkpoint_path": str(checkpoint),
                "artifact_checkpoint_sha256": checkpoint_sha,
            },
        )
        _json(
            run_dir / "run_metadata.json",
            {
                "run_id": run_dir.name,
                "variant": variant,
                "seed": 42,
                "fold_id": 0,
                "validation_protocol": "loso",
                "train_sessions": SOURCE_SESSIONS,
                "validation_sessions": [TARGET_SESSION],
                "selected_by_metric": "val_heldin/r2_mean",
                "selected_metric_value": None,
                # Generic run_artifacts.extract_best_epoch does not parse
                # Lightning's auto-inserted epoch_epoch=011 spelling.
                "best_epoch": None,
                "no_early_stopping": True,
                "max_epochs": 12,
            },
        )
        identity = {
            "run_id": run_dir.name,
            "variant": variant,
            "seed": 42,
            "validation_protocol": "loso",
            "fold_id": 0,
            "M": 10,
        }
        _csv(
            run_dir / "metrics_summary.csv",
            [
                {**identity, "split": "test_heldin", "session": "", "R2_variance_weighted": f"{score:.8f}"},
                {**identity, "split": "test_heldout", "session": "", "R2_variance_weighted": ""},
            ],
        )
        _csv(
            run_dir / "metrics_per_session.csv",
            [{**identity, "split": "test_heldin", "session": TARGET_SESSION,
              "R2_variance_weighted": f"{score:.8f}"}],
        )

    decoder_sha = "d" * 64
    initial_sha = "e" * 64
    preflight = tmp_path / "receipt_v2.json"
    _json(
        preflight,
        {
            "schema": "m1_version_b_preflight_v2",
            "scope": {
                "fold": 0,
                "source_sessions": SOURCE_SESSIONS,
                "target_session": TARGET_SESSION,
                "formal_test_opened": False,
                "heldout_files_opened": False,
                "minival_files_opened": False,
                "target_backpropagation": False,
            },
            "source_files": source_records,
            "checks": {"teacher_checkpoint_sha256": teacher_sha},
            "model_invariants": {
                "b3s_c0_initial_encoder_sha256": initial_sha,
                "b3s_c_initial_encoder_sha256": initial_sha,
                "teacher_net_sha256": decoder_sha,
                "decoder_sha256": {
                    experiment: decoder_sha for experiment, _arm, _variant, _score in ARM_INFO.values()
                },
            },
        },
    )
    return run_dirs, preflight


def _aggregate(run_dirs: dict[str, Path], preflight: Path):
    return MODULE.aggregate(run_dirs["hs"], run_dirs["bc0"], run_dirs["bc"], preflight)


def test_aggregate_accepts_only_terminal_paired_cell(tmp_path: Path) -> None:
    run_dirs, preflight = _make_fixture(tmp_path)
    result = _aggregate(run_dirs, preflight)
    assert result["status"] == "valid_terminal_three_arm_result"
    assert result["arms"]["hs"]["R2"] == pytest.approx(0.50)
    assert result["paired_deltas"]["bc_minus_bc0"] == pytest.approx(0.04)
    assert result["paired_deltas"]["bc_minus_hs"] == pytest.approx(0.07)
    assert result["paired_deltas"]["bc0_minus_hs"] == pytest.approx(0.03)
    assert result["primary_gate"]["passed"] is True
    assert result["scope"]["test_sample_count"] == 26_496
    assert result["scope"]["eligible_query_window_count"] == 26_517
    assert result["scope"]["dropped_tail_window_count"] == 21


def test_aggregate_accepts_exact_exported_omegaconf_interpolations(tmp_path: Path) -> None:
    run_dirs, preflight = _make_fixture(tmp_path)
    for run_dir in run_dirs.values():
        path = run_dir / "resolved_config.yaml"
        payload = yaml.safe_load(path.read_text())
        payload["data"]["sampler_seed"] = "${seed}"
        teacher = Path(payload["model"]["teacher_ckpt_path"])
        payload["paths"] = {"root_dir": str(teacher.parent)}
        payload["model"]["teacher_ckpt_path"] = "${paths.root_dir}/" + teacher.name
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    result = _aggregate(run_dirs, preflight)
    assert result["status"] == "valid_terminal_three_arm_result"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("query", "query_pairing_sha256 mismatch"),
        ("query_count", "eligible query window count"),
        ("query_boundary", "query end"),
        ("source_manifest", "same source manifest"),
        ("model", "resolved model configuration"),
        ("limit_val", "limit_val_batches"),
        ("terminal", "terminal source checkpoint epoch"),
        ("checkpoint_sha", "terminal checkpoint on-disk SHA"),
        ("heldout", "heldout_files_opened"),
        ("target_validation", "target_query_values_used_for_optimizer_or_checkpoint_selection"),
        ("teacher_sha", "teacher checkpoint SHA"),
        ("data_sha", "ses-20120924.sha256"),
        ("initial", "initial encoder SHA"),
    ],
)
def test_aggregate_rejects_contract_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    run_dirs, preflight = _make_fixture(tmp_path)
    if mutation == "query":
        path = run_dirs["bc"] / "split_manifest.json"
        payload = json.loads(path.read_text())
        payload["query_window_audit"][TARGET_SESSION]["raw_query_start_bin"] += 1
        _json(path, payload)
    elif mutation == "query_count":
        path = run_dirs["bc"] / "split_manifest.json"
        payload = json.loads(path.read_text())
        payload["query_window_audit"][TARGET_SESSION]["eligible_windows"] = 26_516
        _json(path, payload)
    elif mutation == "query_boundary":
        path = run_dirs["bc"] / "split_manifest.json"
        payload = json.loads(path.read_text())
        payload["query_window_audit"][TARGET_SESSION]["query_end_trial"] = 209
        _json(path, payload)
    elif mutation == "source_manifest":
        path = run_dirs["bc"] / "source_manifest.json"
        payload = json.loads(path.read_text())
        payload["src/models/streaming_calibration_module.py"] = "c" * 64
        _json(path, payload)
    elif mutation == "model":
        path = run_dirs["bc"] / "resolved_config.yaml"
        payload = yaml.safe_load(path.read_text())
        payload["model"]["hidden_dim"] = 63
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    elif mutation == "limit_val":
        path = run_dirs["bc"] / "resolved_config.yaml"
        payload = yaml.safe_load(path.read_text())
        payload["trainer"]["limit_val_batches"] = 1
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    elif mutation == "terminal":
        path = run_dirs["bc"] / "checkpoint_manifest.json"
        payload = json.loads(path.read_text())
        payload["source_checkpoint_path"] = "/synthetic/bc/epoch_epoch=010.ckpt"
        _json(path, payload)
    elif mutation == "checkpoint_sha":
        (run_dirs["bc"] / "checkpoints" / "best.ckpt").write_bytes(b"tampered")
    elif mutation == "heldout":
        path = run_dirs["bc"] / "split_manifest.json"
        payload = json.loads(path.read_text())
        payload["heldout_files_opened"] = True
        _json(path, payload)
    elif mutation == "target_validation":
        path = run_dirs["bc"] / "split_manifest.json"
        payload = json.loads(path.read_text())
        payload["target_query_values_used_for_optimizer_or_checkpoint_selection"] = True
        _json(path, payload)
    elif mutation == "teacher_sha":
        path = run_dirs["bc"] / "teacher_metadata.json"
        payload = json.loads(path.read_text())
        payload["teacher_checkpoint_sha256"] = "0" * 64
        _json(path, payload)
    elif mutation == "data_sha":
        path = run_dirs["bc"] / "split_manifest.json"
        payload = json.loads(path.read_text())
        payload["target_file"]["sha256"] = "0" * 64
        _json(path, payload)
    elif mutation == "initial":
        payload = json.loads(preflight.read_text())
        payload["model_invariants"]["b3s_c_initial_encoder_sha256"] = "f" * 64
        _json(preflight, payload)
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(MODULE.ArtifactContractError, match=message):
        _aggregate(run_dirs, preflight)


def test_cli_refuses_to_overwrite_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dirs, preflight = _make_fixture(tmp_path)
    output = tmp_path / "aggregate.json"
    argv = [
        str(SCRIPT),
        "--hs-run-dir", str(run_dirs["hs"]),
        "--bc0-run-dir", str(run_dirs["bc0"]),
        "--bc-run-dir", str(run_dirs["bc"]),
        "--preflight-receipt", str(preflight),
        "--output", str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert MODULE.main() == 0
    assert output.is_file()
    first = output.read_bytes()
    capsys.readouterr()
    assert MODULE.main() == 2
    captured = capsys.readouterr()
    assert "refusing to overwrite existing output" in captured.err
    assert output.read_bytes() == first
