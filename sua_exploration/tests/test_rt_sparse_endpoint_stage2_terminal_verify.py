"""Focused contracts for the independent RT Stage-2 terminal verifier."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/rt_sparse_endpoint_stage2_terminal_verify.py"


def module():
    spec = importlib.util.spec_from_file_location("rt_stage2_terminal_verify_test", SCRIPT)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config(cell):
    return {
        "run_id": cell["run_id"],
        "seed": 42,
        "train": True,
        "test": False,
        "no_early_stopping": True,
        "optimized_metric": "val_heldin/r2_mean",
        "data": {
            "_target_": "src.data.rt_nested_loso_datamodule.RtNestedLossoDataModule",
            "data_dir": "${paths.root_dir}/../sua_exploration/data/dandi_000688/sub-C/",
            "batch_size": 32,
            "window_size": 50,
            "calibration_n_trials": 24,
            "query_start_trial": "${.calibration_n_trials}",
            "random_calibration": False,
            "smooth_calibration": False,
            "max_trial_length": 100,
            "interpolate_trials": True,
            "interpolate_trials_kind": "cubic",
            "loso_fold": cell["fold"],
            "outer_loso_fold": cell["fold"],
            "side_feature_group": cell["arm"],
            "session_window_budget": 4096,
            "sampler_seed": "${seed}",
        },
        "model": {
            "_target_": "src.models.streaming_calibration_module.StreamingCalibrationLitModule",
            "variant": "B3S",
            "hidden_dim": 64,
            "side_dim": 4,
            "freeze_decoder": False,
            "loss_mode": "task_only",
            "lambda_y": 0.0,
            "lambda_E": 0.0,
            "teacher_ckpt_path": "${paths.teacher_ckpt_path}",
            "optimizer": {
                "_target_": "torch.optim.Adam",
                "_partial_": True,
                "lr": 1.0e-4,
                "weight_decay": 0.0,
            },
            "scheduler": None,
        },
        "trainer": {"max_epochs": 35},
        "paths": {
            "root_dir": ".",
            "teacher_ckpt_path": (
                "${paths.root_dir}/../SPINT-main/logs/train/runs/2026-07-07-16-05-16/"
                "checkpoints/best_ckpt/epoch_034.ckpt"
            ),
        },
    }


def _split(cell, session):
    return {
        "validation_protocol": "nested_loso",
        "formal_heldout_opened": False,
        "requested_side_feature_group": cell["arm"],
        "outer_loso_fold": cell["fold"],
        "target_session": session,
        "nested_selection": {
            "clean": True,
            "outer_target_loaded_during_fit": False,
            "outer_target_query_labels_read_during_fit": False,
            "inner_validation_only_for_checkpoint_selection": True,
            "checkpoint_metric": "val_heldin/r2_mean",
            "checkpoint_metric_scope": "inner_validation_session_only",
        },
        "calibration": {
            "budget_trials": 24,
            "trial_index_range": [0, 24],
            "target_calibration_optimizer_steps": 0,
        },
        "query": {"query_start_trial": 24, "window_size_bins": 50},
    }


def _outer(cell, session, paths, checkpoint_sha, r2):
    state_sha = hashlib.sha256(cell["output_key"].encode()).hexdigest()
    query_start_sha = hashlib.sha256(f"start:{cell['fold']}".encode()).hexdigest()
    target_sha = hashlib.sha256(f"target:{cell['fold']}".encode()).hexdigest()
    combined_sha = hashlib.sha256(f"combined:{cell['fold']}".encode()).hexdigest()
    eligible = 128 + cell["fold"]
    evaluated = eligible - (eligible % 32)
    return {
        "schema": "rt_clean_nested_loso_outer_eval_v1",
        "status": "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP",
        "run_id": cell["run_id"],
        "seed": 42,
        "arm": cell["arm"],
        "outer_loso_fold": cell["fold"],
        "outer_target_session": session,
        "checkpoint_sha256": checkpoint_sha,
        "selection_receipt_path": str(paths["selection_receipt"]),
        "config_path": str(paths["config"]),
        "fit_split_manifest": str(paths["split_manifest"]),
        "query_start_trial": 24,
        "window_size": 50,
        "query_windows_evaluated": evaluated,
        "r2_variance_weighted": r2,
        "target_query_labels_used_for_scoring_only": True,
        "target_query_labels_used_for_calibration": False,
        "target_query_labels_used_for_normalization": False,
        "target_query_labels_used_for_checkpoint_selection": False,
        "target_backpropagation": False,
        "optimizer_present": False,
        "model_training_mode": False,
        "model_state_sha256_before": state_sha,
        "model_state_sha256_after_target_carrier": state_sha,
        "model_state_sha256_after": state_sha,
        "model_state_unchanged": True,
        "model_state_three_point_unchanged": True,
        "matched_query_window_identity": {
            session: {
                "support_trials": 24,
                "query_start_trial": 24,
                "window_size": 50,
                "eligible_windows": eligible,
                "full_window_disjoint": True,
                "ordered_window_start_sha256": query_start_sha,
                "ordered_target_covariate_evalmask_sha256": target_sha,
                "ordered_query_identity_sha256": combined_sha,
            }
        },
    }


class _Dataset:
    def __init__(self, count: int):
        import numpy as np
        self.window_indices = [("target", index) for index in range(count)]
        self.window_size = 50
        self.covariate_data = {"target": np.arange((count + 50) * 2, dtype=np.float32).reshape(-1, 2)}
        self.eval_mask = {"target": np.ones(count + 50, dtype=bool)}


class _Sampler:
    def __init__(self, batches, batch_size=32):
        self._batches = batches
        self.batch_size = batch_size

    def __iter__(self):
        return iter(self._batches)


@pytest.mark.parametrize("eligible,remainder", [(24632, 24), (19310, 14), (29895, 7)])
def test_actual_query_replay_uses_complete_sampler_batches_for_real_rt_remainders(eligible, remainder):
    mod = module()
    dataset = _Dataset(eligible)
    count = eligible - remainder
    sampler = _Sampler([list(range(start, start + 32)) for start in range(0, count, 32)])
    replay = mod.evaluated_query_identity_from_sampler(dataset, sampler)
    assert replay["evaluated_windows"] == count
    assert replay["sampler_batch_count"] == count // 32


def test_actual_query_replay_rejects_same_count_different_ordered_indices():
    mod = module()
    dataset = _Dataset(64)
    left = mod.evaluated_query_identity_from_sampler(dataset, _Sampler([list(range(32)), list(range(32, 64))]))
    right = mod.evaluated_query_identity_from_sampler(dataset, _Sampler([list(range(1, 33)), list(range(0, 32))]))
    assert left["evaluated_windows"] == right["evaluated_windows"] == 64
    assert left["ordered_query_identity_sha256"] != right["ordered_query_identity_sha256"]


def make_terminal_bundle(tmp_path: Path):
    mod = module()
    artifact = tmp_path / "matrix"
    manifest = mod._synthetic_manifest()
    manifest_path = artifact / mod.MANIFEST_NAME
    write_json(manifest_path, manifest)
    manifest_sha = sha(manifest_path)
    records = []
    for cell in mod._expected_cells():
        run = tmp_path / (
            f"2026-08-10-00-00-00-000000_rid-{cell['run_id']}_"
            f"f{cell['fold']}_s42"
        )
        paths = {
            "selection_receipt": run / "rt_nested_selection_receipt.json",
            "config": run / ".hydra/config.yaml",
            "checkpoint": run / "checkpoints/best_ckpt/epoch_003.ckpt",
            "split_manifest": run / "split_manifest.json",
            "outer_receipt": artifact / "outer" / f"{cell['output_key']}.json",
        }
        paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
        paths["checkpoint"].write_bytes(f"fresh-checkpoint:{cell['output_key']}".encode())
        write_json(paths["config"], _config(cell))
        session = f"ses-RT-synthetic-{cell['fold']:02d}"
        write_json(paths["split_manifest"], _split(cell, session))
        selection = {
            "schema": "rt_clean_nested_loso_selection_receipt_v1",
            "status": "PASS_FIT_INNER_SELECTION_ONLY",
            "selection_receipt_path": str(paths["selection_receipt"]),
            "run_id": cell["run_id"],
            "run_dir": str(run),
            "arm": cell["arm"],
            "outer_loso_fold": cell["fold"],
            "seed": 42,
            "selected_by_metric": "val_heldin/r2_mean",
            "selected_metric_scope": "inner_validation_session_only",
            "selected_metric_value": 0.5,
            "selected_epoch": 3,
            "selected_global_step": 10,
            "best_model_path": str(paths["checkpoint"]),
            "best_model_sha256": sha(paths["checkpoint"]),
            "config_path": str(paths["config"]),
            "config_sha256": sha(paths["config"]),
            "split_manifest_path": str(paths["split_manifest"]),
            "split_manifest_sha256": sha(paths["split_manifest"]),
            "formal_heldout_opened": False,
            "outer_target_loaded_during_fit": False,
            "outer_target_query_labels_read_during_fit": False,
        }
        write_json(paths["selection_receipt"], selection)
        r2 = {
            "rt_sparse_endpoint_t4d": 0.40 + cell["fold"] * 0.001,
            "afc4_vel": 0.38 + cell["fold"] * 0.0005,
            "zero4": 0.35 - cell["fold"] * 0.0002,
        }[cell["arm"]]
        outer = _outer(cell, session, paths, sha(paths["checkpoint"]), r2)
        write_json(paths["outer_receipt"], outer)
        closure = {
            "schema": "rt_sparse_endpoint_stage2_cell_closure_v2",
            "matrix_manifest_sha256": manifest_sha,
            "cell": cell,
            "selection_receipt_sha256": sha(paths["selection_receipt"]),
            "config_sha256": sha(paths["config"]),
            "checkpoint_sha256": sha(paths["checkpoint"]),
            "split_manifest_sha256": sha(paths["split_manifest"]),
            "outer_receipt_sha256": sha(paths["outer_receipt"]),
            "artifact_paths": {name: str(path) for name, path in paths.items()},
            "outer_receipt": outer,
        }
        write_json(artifact / "cells" / f"{cell['output_key']}.json", closure)
        records.append(
            mod.validate_cell_closure(
                cell,
                closure,
                matrix_manifest_sha256=manifest_sha,
                resolver=mod.PathResolver(),
            )
        )
    aggregate = mod.recompute_terminal_aggregate(records, manifest_sha256=manifest_sha)
    write_json(artifact / mod.AGGREGATE_NAME, aggregate)
    return mod, artifact


def test_schema_only_preflight_is_pure_and_rejects_empty_results():
    mod = module()
    result = mod.preflight_schema_only()
    assert result["status"] == "PASS_SCHEMA_ONLY_NO_ARTIFACT_OR_TARGET_ACCESS"
    assert result["synthetic_cells"] == 45
    assert result["empty_terminal_result_rejected"] is True
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "NWBHDF5IO" not in source
    assert "subprocess.run" not in source


def test_schema_only_cli_needs_no_artifact_root_or_data(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--preflight-schema-only"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "PASS_SCHEMA_ONLY_NO_ARTIFACT_OR_TARGET_ACCESS"


def test_complete_45_cell_bundle_recomputes_and_matches_aggregate(tmp_path):
    mod, artifact = make_terminal_bundle(tmp_path)
    result = mod.verify_terminal_bundle(artifact, verify_local_bindings=False)
    assert result["status"] == "PASS_INDEPENDENT_TERMINAL_VERIFICATION_READ_ONLY"
    assert result["verified_cells"] == 45
    assert result["verified_folds"] == 15
    assert result["historical_full_used"] is False
    assert result["recomputed"]["t4d_minus_zero4"]["positive"] == 15
    assert result["recomputed"]["t4d_minus_full"]["positive"] == 15


def test_missing_or_duplicate_cell_is_rejected(tmp_path):
    mod, artifact = make_terminal_bundle(tmp_path)
    missing = artifact / "cells/f03_zero4.json"
    missing.unlink()
    with pytest.raises(mod.VerificationError, match="terminal closure missing"):
        mod.verify_terminal_bundle(artifact, verify_local_bindings=False)


def test_aggregate_field_tamper_is_rejected(tmp_path):
    mod, artifact = make_terminal_bundle(tmp_path)
    aggregate_path = artifact / mod.AGGREGATE_NAME
    aggregate = json.loads(aggregate_path.read_text())
    aggregate["t4d_minus_zero4"]["mean"] += 0.001
    write_json(aggregate_path, aggregate)
    with pytest.raises(mod.VerificationError, match="float mismatch"):
        mod.verify_terminal_bundle(artifact, verify_local_bindings=False)


def test_cross_arm_query_target_mask_digest_mismatch_is_rejected(tmp_path):
    mod, artifact = make_terminal_bundle(tmp_path)
    key = "f04_zero4"
    closure_path = artifact / "cells" / f"{key}.json"
    closure = json.loads(closure_path.read_text())
    outer_path = Path(closure["artifact_paths"]["outer_receipt"])
    outer = json.loads(outer_path.read_text())
    session = outer["outer_target_session"]
    outer["matched_query_window_identity"][session]["ordered_target_covariate_evalmask_sha256"] = "f" * 64
    write_json(outer_path, outer)
    closure["outer_receipt"] = outer
    closure["outer_receipt_sha256"] = sha(outer_path)
    write_json(closure_path, closure)
    with pytest.raises(mod.VerificationError, match="ordered evaluated query/target/mask digest mismatch"):
        mod.verify_terminal_bundle(artifact, verify_local_bindings=False)


def test_historical_or_nonfresh_full_selection_is_rejected(tmp_path):
    mod, artifact = make_terminal_bundle(tmp_path)
    key = "f02_afc4_vel"
    closure_path = artifact / "cells" / f"{key}.json"
    closure = json.loads(closure_path.read_text())
    selection_path = Path(closure["artifact_paths"]["selection_receipt"])
    selection = json.loads(selection_path.read_text())
    selection["run_id"] = "historical_full"
    write_json(selection_path, selection)
    closure["selection_receipt_sha256"] = sha(selection_path)
    write_json(closure_path, closure)
    with pytest.raises(mod.VerificationError, match="fresh matrix run"):
        mod.verify_terminal_bundle(artifact, verify_local_bindings=False)


def test_target_state_or_sha_chain_tamper_is_rejected(tmp_path):
    mod, artifact = make_terminal_bundle(tmp_path)
    key = "f01_rt_sparse_endpoint_t4d"
    closure_path = artifact / "cells" / f"{key}.json"
    closure = json.loads(closure_path.read_text())
    outer_path = Path(closure["artifact_paths"]["outer_receipt"])
    outer = json.loads(outer_path.read_text())
    outer["model_state_sha256_after"] = "e" * 64
    write_json(outer_path, outer)
    closure["outer_receipt"] = outer
    closure["outer_receipt_sha256"] = sha(outer_path)
    write_json(closure_path, closure)
    with pytest.raises(mod.VerificationError, match="model state SHA values differ"):
        mod.verify_terminal_bundle(artifact, verify_local_bindings=False)


def test_path_resolver_supports_remote_prefix_rebase(tmp_path):
    mod = module()
    local = tmp_path / "local"
    file = local / "run/checkpoint.ckpt"
    file.parent.mkdir(parents=True)
    file.write_bytes(b"checkpoint")
    resolver = mod.PathResolver([(Path("/remote/work"), local)])
    assert resolver.resolve("/remote/work/run/checkpoint.ckpt") == file.resolve()
    with pytest.raises(mod.VerificationError):
        resolver.resolve("/another/remote/checkpoint.ckpt")


def test_workspace_source_config_teacher_and_nwb_bindings_are_transitive(tmp_path):
    mod = module()
    workspace = tmp_path / "workspace"
    manifest = mod._synthetic_manifest()

    contract = workspace / mod.CONTRACT_REL
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_text("frozen contract\n", encoding="utf-8")
    manifest["contract"] = {"path": str(contract), "sha256": sha(contract)}

    for index, (name, relative) in enumerate(mod.SURFACE_RELS.items()):
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"surface {index} {name}\n", encoding="utf-8")
        manifest["surfaces"][name] = {"path": str(path), "sha256": sha(path)}

    teacher = workspace / mod.TEACHER_REL
    teacher.parent.mkdir(parents=True, exist_ok=True)
    teacher.write_bytes(b"frozen teacher")
    config_digest = mod._tree_digest(
        workspace / mod.CONFIG_TREE_REL, "**/*.yaml", workspace
    )
    source_digest = mod._tree_digest(
        workspace / mod.SOURCE_TREE_REL, "**/*.py", workspace
    )
    manifest["launch_inputs"] = {
        "configs_yaml_tree": {
            "path": str(workspace / mod.CONFIG_TREE_REL),
            "sha256": config_digest,
        },
        "src_py_tree": {
            "path": str(workspace / mod.SOURCE_TREE_REL),
            "sha256": source_digest,
        },
        "teacher_checkpoint": {
            "path": str(teacher),
            "bytes": teacher.stat().st_size,
            "sha256": sha(teacher),
        },
    }

    nwb_root = workspace / mod.DEFAULT_NWB_REL
    nwb_root.mkdir(parents=True, exist_ok=True)
    nwb_rows = []
    for index in range(15):
        path = nwb_root / f"sub-C_ses-RT-{index:08d}_behavior+ecephys.nwb"
        path.write_bytes(f"nwb {index}".encode())
        nwb_rows.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
        )
    manifest["nwb_allowlist"] = nwb_rows

    readiness = {
        "status": "PASS_ROOT_REVIEW_STAGE2_MATRIX_READY_NOT_LAUNCHED",
        "bound_files": {
            "contract": {"path": str(mod.CONTRACT_REL), "sha256": sha(contract)},
            **{
                name: {"path": str(relative), "sha256": manifest["surfaces"][name]["sha256"]}
                for name, relative in mod.SURFACE_RELS.items()
            },
            "configs_yaml_tree": {
                "path": str(mod.CONFIG_TREE_REL),
                "sha256": config_digest,
            },
            "src_py_tree": {
                "path": str(mod.SOURCE_TREE_REL),
                "sha256": source_digest,
            },
            "teacher_checkpoint": {
                "path": str(mod.TEACHER_REL),
                "bytes": teacher.stat().st_size,
                "sha256": sha(teacher),
            },
        },
    }
    readiness_path = workspace / mod.READINESS_REL
    write_json(readiness_path, readiness)
    manifest["root_readiness"] = {
        "path": str(readiness_path),
        "sha256": sha(readiness_path),
    }

    mod.validate_manifest_schema(manifest)
    result = mod.verify_workspace_bindings(
        manifest, workspace_root=workspace, nwb_root=nwb_root
    )
    assert result["nwb_count"] == 15
    assert result["source_tree_sha256"] == source_digest
    assert result["config_tree_sha256"] == config_digest
    assert result["teacher_sha256"] == sha(teacher)


def test_manifest_rejects_seed_arm_or_freshness_drift():
    mod = module()
    for mutation in ("seed", "arm", "fresh"):
        manifest = copy.deepcopy(mod._synthetic_manifest())
        if mutation == "seed":
            manifest["matrix"]["cells"][0]["seed"] = 43
        elif mutation == "arm":
            manifest["matrix"]["arms"] = ["zero4", "afc4_vel", "rt_sparse_endpoint_t4d"]
        else:
            manifest["matrix"]["cells"][0]["fresh_fit"] = False
        with pytest.raises(mod.VerificationError):
            mod.validate_manifest_schema(manifest)
