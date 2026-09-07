"""Receipt-only tests for the RT XLSv2 supervisor and finalizer."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SCRIPT = ROOT / "scripts/rt_xls_v2_matched_continuation_v1.py"
SPEC = importlib.util.spec_from_file_location("rt_xls_v2_matched_continuation_v1_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write(path: Path, body: dict | str, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body if isinstance(body, str) else json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    if immutable:
        path.chmod(0o444)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def references():
    return MODULE.load_references(mb4_aggregate=MODULE.DEFAULT_MB4, xls_v2_audit=MODULE.DEFAULT_AUDIT)


def _complete_cell(root: Path, *, fold: int, references, score: float | None = None) -> None:
    paths = MODULE.cell_paths(root, fold)
    comparator = references.rows[fold]
    target = str(comparator["target_session"])
    inner_validation = str(comparator["inner_validation_session"])
    all_sessions = sorted(references.permutation_sha_by_session)
    inner_train = [name for name in all_sessions if name not in {target, inner_validation}]
    fit_sessions = inner_train + [inner_validation]
    _write(paths.config, "data:\n  side_feature_group: afc4_xls_v2\nmodel:\n  freeze_decoder: false\n")
    split = {
        "task": "rt", "development_only": True, "validation_protocol": "nested_loso",
        "outer_loso_fold": fold, "loso_fold": fold, "requested_side_feature_group": MODULE.ARM,
        "arm": {"canonical_arm": MODULE.ARM}, "target_session": target,
        "inner_validation_session": inner_validation, "inner_train_sessions": inner_train,
        "calibration": {"budget_trials": 24, "trial_index_range": [0, 24], "target_calibration_optimizer_steps": 0},
        "query": {"query_start_trial": 24, "window_size_bins": 50, "full_window_after_support_required": True},
        "nested_selection": {
            "clean": True, "outer_target_loaded_during_fit": False,
            "outer_target_query_labels_read_during_fit": False,
            "inner_validation_only_for_checkpoint_selection": True,
            "checkpoint_metric": "val_heldin/r2_mean", "checkpoint_metric_scope": "inner_validation_session_only",
        },
        "source_only_normalizer": {
            "fit_scope": "inner_train_sessions_only", "feature_group": MODULE.ARM,
            "fit_sessions": inner_train, "excluded_outer_target_session": target,
        },
        "xls_v2_support_audit": {
            "sha256": references.audit_sha256,
            "query_labels_available_to_generator": False,
            "common_inverse_or_alignment_map": False,
            "per_session_permutation_sha256": {
                name: references.permutation_sha_by_session[name] for name in fit_sessions
            },
        },
    }
    _write(paths.split, split)
    checkpoint_sha = "a" * 64
    _write(paths.selection, {
        "schema": "rt_clean_nested_loso_selection_receipt_v1",
        "status": "PASS_FIT_INNER_SELECTION_ONLY", "arm": MODULE.ARM,
        "outer_loso_fold": fold, "seed": 42, "selected_by_metric": "val_heldin/r2_mean",
        "selected_metric_scope": "inner_validation_session_only", "formal_heldout_opened": False,
        "outer_target_loaded_during_fit": False, "outer_target_query_labels_read_during_fit": False,
        "best_model_sha256": checkpoint_sha, "config_sha256": _sha(paths.config),
        "split_manifest_sha256": _sha(paths.split),
    })
    _write(paths.outer, {
        "schema": MODULE.OUTER_SCHEMA, "status": MODULE.OUTER_STATUS, "arm": MODULE.ARM,
        "outer_loso_fold": fold, "seed": 42, "outer_target_session": target,
        "query_start_trial": 24, "window_size": 50,
        "query_windows_evaluated": comparator["query_windows_evaluated"],
        "normalizer_fit_scope": "inner_train_sessions_only",
        "xls_v2_support_audit_sha256": references.audit_sha256,
        "xls_v2_target_permutation_sha256": references.permutation_sha_by_session[target],
        "target_backpropagation": False, "optimizer_present": False,
        "model_training_mode": False, "model_state_unchanged": True,
        "target_query_labels_used_for_calibration": False,
        "target_query_labels_used_for_normalization": False,
        "target_query_labels_used_for_checkpoint_selection": False,
        "target_query_labels_used_for_scoring_only": True,
        "query_labels_available_to_xls_v2_generator": False,
        "common_inverse_or_alignment_map": False,
        "model_state_sha256_before": "b" * 64, "model_state_sha256_after": "b" * 64,
        "checkpoint_sha256": checkpoint_sha,
        "r2_variance_weighted": float(comparator["full_r2"] - 0.1 if score is None else score),
    })
    _write(paths.terminal, {"status": "PASS", "arm": MODULE.ARM, "fold": fold, "seed": 42})


def test_plan_is_static_joint_only_and_does_not_create_roots(tmp_path: Path, references) -> None:
    work, imported, output = tmp_path / "raw", tmp_path / "imported", tmp_path / "aggregate.json"
    plan = MODULE.build_plan(work_root=work, import_root=imported, output=output, references=references)
    assert plan["status"] == "PLAN_ONLY_NO_GPU_NO_NWB_NO_TRAINER_NO_TMUX"
    assert not work.exists() and not imported.exists() and not output.exists()
    assert plan["protocol"]["reuse_r_c_checkpoint"] is False
    assert plan["protocol"]["common_inverse_or_alignment_map"] is False
    left, right = plan["partitions"]["gpu1_asc"], plan["partitions"]["gpu0_desc"]
    assert left["physical_gpu"] == 1 and left["folds"] == list(range(8))
    assert right["physical_gpu"] == 0 and right["folds"] == list(range(14, 7, -1))
    assert set(left["folds"]).isdisjoint(right["folds"])
    for partition in (left, right):
        for cell in partition["cells"]:
            command = cell["train_command"]
            joined = " ".join(command)
            assert "experiment=rt_joint_afc4_xls_v2_m24_loso" in joined
            assert "rt_b3s_afc4_xls_v2" not in joined
            assert "data.side_feature_group=afc4_xls_v2" in joined
            assert "seed=42" in joined and "test=false" in joined
            assert "ckpt" not in joined.lower() and "checkpoint" not in joined.lower()
            assert cell["eval_command_template"][-2:] == ["--device", "cuda"]
    assert plan["finalizer"]["requires_workers_observed"] == ["rt_xls_v2_gpu1_asc", "rt_xls_v2_gpu0_desc"]
    assert "--copy-import-all-and-finalize" in plan["finalizer"]["launch_command"]


def test_validation_rejects_query_target_and_permutation_drift(tmp_path: Path, references) -> None:
    raw = tmp_path / "raw"
    _complete_cell(raw, fold=0, references=references)
    paths = MODULE.cell_paths(raw, 0)
    outer = json.loads(paths.outer.read_text())
    outer["query_windows_evaluated"] += 1
    _write(paths.outer, outer)
    with pytest.raises(MODULE.XlsV2ContinuationError, match="query_windows_evaluated"):
        MODULE.validate_cell(root=raw, fold=0, references=references, require_import=False)

    _complete_cell(raw, fold=1, references=references)
    paths = MODULE.cell_paths(raw, 1)
    split = json.loads(paths.split.read_text())
    session = split["inner_train_sessions"][0]
    split["xls_v2_support_audit"]["per_session_permutation_sha256"][session] = "0" * 64
    _write(paths.split, split)
    selection = json.loads(paths.selection.read_text())
    selection["split_manifest_sha256"] = _sha(paths.split)
    _write(paths.selection, selection)
    with pytest.raises(MODULE.XlsV2ContinuationError, match="permutation SHA drift"):
        MODULE.validate_cell(root=raw, fold=1, references=references, require_import=False)


def test_copy_import_finalizer_requires_15_and_retains_negative_delta(tmp_path: Path, references) -> None:
    raw, imported = tmp_path / "raw", tmp_path / "imported"
    for fold in MODULE.FOLDS:
        score = float(references.rows[fold]["full_r2"] + 0.2) if fold == 4 else None
        _complete_cell(raw, fold=fold, references=references, score=score)
    output = tmp_path / "aggregate.json"
    result = MODULE.copy_import_all_and_finalize(
        source_root=raw, import_root=imported, output=output, references=references,
    )
    assert result["status"] == "PASS_RT_XLS_V2_COPY_IMPORT_ALL_AND_FINALIZE"
    assert output.stat().st_mode & 0o777 == 0o444
    body = json.loads(output.read_text())
    assert body["status"] == MODULE.AGGREGATE_STATUS
    row = next(item for item in body["rows"] if item["fold"] == 4)
    assert row["r_c_minus_xls_v2"] == pytest.approx(-0.2)
    assert body["r_c_minus_xls_v2"]["negative_folds"] == 1
    assert body["r_c_minus_xls_v2"]["exact_two_sided_sign_test"]["negative"] == 1
    assert len(list(imported.glob("afc4_xls_v2/fold_*/seed_42/xls_v2_copy_import.json"))) == 15


def test_gpu_collision_fails_before_cell_creation(tmp_path: Path, references, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(MODULE, "active_compute_commands_on_gpu", lambda gpu: (["busy-owner"], []))
    work = tmp_path / "raw"
    with pytest.raises(MODULE.XlsV2ContinuationError, match="active compute owners"):
        MODULE.execute_partition(partition="gpu1_asc", work_root=work, references=references)
    assert not MODULE.cell_paths(work, 0).cell.exists()


def test_same_fold_detection_is_token_exact() -> None:
    fold10 = "python src/train.py data.loso_fold=10 data.outer_loso_fold=10"
    fold1 = "python src/train.py data.loso_fold=1 data.outer_loso_fold=1"
    assert MODULE._same_fold_writers([fold10], 1) == []
    assert MODULE._same_fold_writers([fold1], 1) == [fold1]

