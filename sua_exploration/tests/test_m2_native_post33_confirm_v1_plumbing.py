"""Focused score-free tests for M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from sua_exploration.mc_maze import m2_native_post33_confirm_v1 as contract


ROOT = Path(__file__).resolve().parents[2]
OLD_AUDIT = ROOT / "sua_exploration/results/m2_heldin_postsupport_endpoint_v1/audit.json"
SCORE_FREE_AUDIT = ROOT / "sua_exploration/results/m2_native_t4_spint_post33_confirm_v1_scorefree_audit_20260804/audit.json"
LIVE_PLUMBING = ROOT / "sua_exploration/results/m2_native_t4_spint_post33_confirm_v1_live_plumbing_20260804/live_plumbing.json"
C1_RECEIPT = ROOT / "sua_exploration/results/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist/receipt.json"
WRITER = ROOT / "sua_exploration/scripts/write_m2_native_post33_confirm_v1_draft_prelaunch.py"
VERIFIER = ROOT / "sua_exploration/scripts/verify_m2_native_post33_confirm_v1_draft.py"


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_contract(**overrides):
    values = {
        "task": "m2",
        "validation_protocol": "loso",
        "calibration_n_trials": 33,
        "heldin_query_start_trial": 33,
        "random_calibration": False,
        "include_heldout_in_fit": False,
        "loso_fold": 0,
        "window_size": 50,
        "heldin_query_end_trial": None,
        "include_heldout_in_test": False,
        "query_start_trial": 0,
    }
    values.update(overrides)
    return values


def test_only_exact_m2_loso_post33_contract_is_accepted() -> None:
    contract.require_exact_endpoint_contract(**exact_contract())


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("task", "m1"),
        ("validation_protocol", "minival"),
        ("calibration_n_trials", 32),
        ("heldin_query_start_trial", 32),
        ("random_calibration", True),
        ("include_heldout_in_fit", True),
        ("loso_fold", 7),
        ("window_size", 49),
        ("heldin_query_end_trial", 200),
        ("include_heldout_in_test", True),
        ("query_start_trial", 33),
    ],
)
def test_every_nearby_endpoint_variant_fails_closed(field: str, bad: object) -> None:
    with pytest.raises(ValueError):
        contract.require_exact_endpoint_contract(**exact_contract(**{field: bad}))


def test_seven_outer_folds_are_unique_and_absent_from_all_source_roles() -> None:
    outer = []
    for fold in range(7):
        roles = contract.fold_roles(fold)
        left_out = roles["outer_left_out_session"]
        outer.append(left_out)
        assert len(roles["source_train_sessions"]) == 6
        assert roles["source_train_sessions"] == roles["source_normalizer_sessions"]
        assert roles["source_train_sessions"] == roles["source_checkpoint_selection_sessions"]
        assert left_out not in roles["source_train_sessions"]
        assert roles["post33_query_sessions"] == [left_out]
        assert roles["outer_counts"] == {
            "train": 0,
            "normalizer": 0,
            "checkpoint_selection": 0,
            "post33_query": 1,
        }
    assert len(set(outer)) == 7
    assert set(outer) == set(contract.FOLDS.values())


def test_matrix_is_exactly_two_arms_three_seeds_and_seven_folds() -> None:
    matrix = contract.matrix_contract()
    assert matrix["arms"] == ["spint", "t4"]
    assert matrix["seeds"] == [42, 43, 44]
    assert matrix["full_terminal_arm_cells"] == 42
    assert matrix["full_paired_deltas"] == 21
    assert matrix["full_trainings"] == {"spint": 21, "t4": 21}
    assert matrix["stage_a"]["terminal_arm_cells"] == 14
    assert matrix["stage_a"]["futility_rule"] == "(mean42 <= -0.03) OR (pos42 <= 1)"
    assert matrix["stage_a"]["untriggered_decision"] == "continue_without_positive_claim"
    assert matrix["stage_b"]["additional_terminal_arm_cells"] == 28
    assert matrix["stage_b"]["automatic_if_stage_a_not_futile"] is True
    assert len(matrix["cell_ids"]) == len(set(matrix["cell_ids"])) == 42
    assert len(contract.effectiveness_gate_contract()) == 6


def test_checkpoint_selection_is_source_only_and_ties_choose_earlier_epoch() -> None:
    records = [
        {
            "epoch": 4,
            "metric_name": "val_heldin/r2_mean",
            "metric_value": 0.5,
            "metric_scope": "outer_train_source_sessions_only",
            "outer_session_window_count": 0,
            "checkpoint_path": "epoch4.ckpt",
        },
        {
            "epoch": 2,
            "metric_name": "val_heldin/r2_mean",
            "metric_value": 0.5,
            "metric_scope": "outer_train_source_sessions_only",
            "outer_session_window_count": 0,
            "checkpoint_path": "epoch2.ckpt",
        },
        {
            "epoch": 1,
            "metric_name": "val_heldin/r2_mean",
            "metric_value": 0.4,
            "metric_scope": "outer_train_source_sessions_only",
            "outer_session_window_count": 0,
            "checkpoint_path": "epoch1.ckpt",
        },
    ]
    assert contract.select_source_checkpoint(records)["checkpoint_path"] == "epoch2.ckpt"
    for update in (
        {"metric_name": "outer_query/r2"},
        {"metric_scope": "outer_left_out_query"},
        {"outer_session_window_count": 1},
    ):
        invalid = dict(records[0])
        invalid.update(update)
        with pytest.raises(ValueError):
            contract.select_source_checkpoint([invalid])


def test_decoder_and_target_calibration_runtime_contracts_fail_closed() -> None:
    decoder = {
        "tensor_count_expected": 31,
        "tensor_count_compared": 31,
        "bit_exact": True,
        "decoder_requires_grad_tensor_count": 0,
        "decoder_updated_tensor_count": 0,
    }
    target = {
        "optimizer_steps": 0,
        "backward_calls": 0,
        "updated_parameter_tensors": 0,
        "fit_kind": "closed_form_cosine_rank3",
        "support_trials": 33,
        "query_trials_used_for_fit": 0,
    }
    contract.validate_decoder_contract(decoder)
    contract.validate_target_calibration_contract(target)
    for key in ("tensor_count_compared", "bit_exact", "decoder_updated_tensor_count"):
        invalid = dict(decoder)
        invalid[key] = 30 if key == "tensor_count_compared" else (False if key == "bit_exact" else 1)
        with pytest.raises(ValueError):
            contract.validate_decoder_contract(invalid)
    for key in ("optimizer_steps", "backward_calls", "updated_parameter_tensors"):
        invalid = dict(target)
        invalid[key] = 1
        with pytest.raises(ValueError):
            contract.validate_target_calibration_contract(invalid)


def test_frozen_structural_audit_recomputes_101171_from_seven_sessions() -> None:
    audit = json.loads(OLD_AUDIT.read_text(encoding="utf-8"))
    sessions = audit["heldin_postsupport_window_audit"]["per_session"]
    assert len(sessions) == 7
    assert sum(row["query_window_audit"]["eligible_windows"] for row in sessions.values()) == 101_171
    for row in sessions.values():
        query = row["query_window_audit"]
        assert query["query_start_trial"] == 33
        assert query["window_size"] == 50
        assert query["full_window_disjoint"] is True
        assert query["minimum_window_start_padded_bin"] - query["raw_query_start_bin"] == 49


def test_versioned_configs_bind_only_specialized_modules_and_exact_endpoint() -> None:
    configs = [
        ROOT / "SPINT-main/configs/data/falcon_m2_post33_confirm_v1.yaml",
        ROOT / "streaming_calibration_exp/configs/data/falcon_m2_post33_confirm_v1.yaml",
    ]
    for path in configs:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "falcon_post33_confirm_v1_datamodule" in cfg["_target_"]
        expected = exact_contract(loso_fold=0)
        for key in (
            "task",
            "validation_protocol",
            "calibration_n_trials",
            "heldin_query_start_trial",
            "random_calibration",
            "include_heldout_in_fit",
            "window_size",
            "heldin_query_end_trial",
            "include_heldout_in_test",
            "query_start_trial",
        ):
            assert cfg[key] == expected[key]
        assert cfg["standardize_covariates"] is False


def test_new_runtime_modules_import_no_scorer_evalai_or_formal_sua_package() -> None:
    paths = [
        ROOT / "SPINT-main/src/data/falcon_post33_confirm_v1_datamodule.py",
        ROOT / "streaming_calibration_exp/src/data/falcon_post33_confirm_v1_datamodule.py",
        ROOT / "sua_exploration/mc_maze/m2_native_post33_confirm_v1.py",
        ROOT / "sua_exploration/scripts/audit_m2_native_post33_confirm_v1_scorefree.py",
    ]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name.lower() for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append((node.module or "").lower())
        assert not any("scor" in name or "evalai" in name for name in imported)
        assert not any("dandi" in name or "sua" in name for name in imported if name != "sua_exploration.mc_maze")


def test_active_c1_v3r2_source_map_still_matches_every_hash() -> None:
    receipt = json.loads(C1_RECEIPT.read_text(encoding="utf-8"))
    import hashlib

    for relative, frozen in receipt["source_map"].items():
        path = ROOT / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == frozen["sha256"]


@pytest.mark.skipif(not SCORE_FREE_AUDIT.is_file(), reason="official score-free audit not written yet")
def test_score_free_audit_has_no_score_scorer_formal_or_evalai_access() -> None:
    audit = json.loads(SCORE_FREE_AUDIT.read_text(encoding="utf-8"))
    assert audit["execution_scope"] == {
        "gpu_used": False,
        "training_started": False,
        "new_endpoint_r2_values_read": 0,
        "scorer_modules_imported": 0,
        "formal_sua_paths_resolved": 0,
        "formal_sua_files_opened": 0,
        "evalai_calls": 0,
    }
    assert audit["endpoint"]["totals"]["eligible_windows"] == 101_171
    assert audit["input_manifest"]["all_sha256_reverified_now"] is True


@pytest.mark.skipif(not LIVE_PLUMBING.is_file(), reason="live all-fold plumbing audit not written yet")
def test_live_both_side_all_fold_plumbing_matches_101171_and_has_no_outer_leak() -> None:
    evidence = json.loads(LIVE_PLUMBING.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_SCORE_FREE_LIVE_PLUMBING"
    assert evidence["probe_count"] == 14
    assert evidence["eligible_window_totals"] == {"spint": 101_171, "t4": 101_171}
    assert evidence["execution_scope"] == {
        "gpu_used": False,
        "training_started": False,
        "optimizer_steps": 0,
        "new_endpoint_r2_values_read": 0,
        "scorer_modules_imported": 0,
        "formal_sua_paths_resolved": 0,
        "evalai_calls": 0,
    }
    for side in ("spint", "t4"):
        assert len(evidence["sides"][side]) == 7
        for record in evidence["sides"][side].values():
            manifest = record["manifest"]
            assert manifest["outer_counts"] == {
                "train": 0,
                "normalizer": 0,
                "checkpoint_selection": 0,
                "post33_query": 1,
            }
            assert list(record["query_audit"]) == [manifest["outer_left_out_session"]]


def test_pinned_input_verifier_rejects_same_size_content_mutation(tmp_path: Path) -> None:
    verifier = load_script(VERIFIER, "m2_post33_hash_mutation_verifier")
    path = tmp_path / "held-in-calib_fixture.nwb"
    original = b"six-byte-fixture"
    mutated = b"six-byte-fixturf"
    assert len(original) == len(mutated)
    path.write_bytes(original)
    row = {
        "role": "heldin_calib",
        "session": "ses-fixture",
        "path": str(path),
        "size_bytes": len(original),
        "sha256": hashlib.sha256(original).hexdigest(),
        "sha256_reverified_now": True,
    }
    assert verifier.verify_pinned_file(row) == path.resolve()
    path.write_bytes(mutated)
    assert path.stat().st_size == row["size_bytes"]
    with pytest.raises(ValueError, match="SHA-256 drift"):
        verifier.verify_pinned_file(row)


def test_pinned_input_verifier_requires_current_reverification_and_valid_role(tmp_path: Path) -> None:
    verifier = load_script(VERIFIER, "m2_post33_hash_metadata_verifier")
    path = tmp_path / "held-in-minival_fixture.nwb"
    payload = b"fixture"
    path.write_bytes(payload)
    base = {
        "role": "heldin_minival",
        "session": "ses-fixture",
        "path": str(path),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "sha256_reverified_now": True,
    }
    verifier.verify_pinned_file(base)
    for update in ({"sha256_reverified_now": False}, {"role": "heldout_test"}):
        invalid = dict(base)
        invalid.update(update)
        with pytest.raises(ValueError):
            verifier.verify_pinned_file(invalid)


@pytest.mark.skipif(not SCORE_FREE_AUDIT.is_file(), reason="official score-free audit not written yet")
def test_full_verifier_rejects_nonzero_audit_or_live_access_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = load_script(WRITER, "m2_post33_scope_writer")
    verifier = load_script(VERIFIER, "m2_post33_scope_verifier")
    monkeypatch.setattr(verifier, "verify_pinned_file", lambda row: Path(row["path"]))

    payload = writer.build(SCORE_FREE_AUDIT)
    bad_audit = json.loads(SCORE_FREE_AUDIT.read_text(encoding="utf-8"))
    bad_audit["execution_scope"]["formal_sua_paths_resolved"] = 1
    bad_audit_path = tmp_path / "bad_audit.json"
    bad_audit_path.write_text(json.dumps(bad_audit), encoding="utf-8")
    audit_receipt = json.loads(json.dumps(payload))
    audit_receipt["score_free_audit"]["path"] = str(bad_audit_path)
    audit_receipt["score_free_audit"]["sha256"] = hashlib.sha256(
        bad_audit_path.read_bytes()
    ).hexdigest()
    audit_receipt_path = tmp_path / "bad_audit_receipt.json"
    audit_receipt_path.write_text(json.dumps(audit_receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="audit execution scope"):
        verifier.verify(audit_receipt_path)

    bad_live = json.loads(LIVE_PLUMBING.read_text(encoding="utf-8"))
    bad_live["execution_scope"]["scorer_modules_imported"] = 1
    bad_live_path = tmp_path / "bad_live.json"
    bad_live_path.write_text(json.dumps(bad_live), encoding="utf-8")
    live_receipt = json.loads(json.dumps(payload))
    live_receipt["live_plumbing"]["path"] = str(bad_live_path)
    live_receipt["live_plumbing"]["sha256"] = hashlib.sha256(
        bad_live_path.read_bytes()
    ).hexdigest()
    live_receipt_path = tmp_path / "bad_live_receipt.json"
    live_receipt_path.write_text(json.dumps(live_receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="live plumbing execution scope"):
        verifier.verify(live_receipt_path)


@pytest.mark.skipif(not SCORE_FREE_AUDIT.is_file(), reason="official score-free audit not written yet")
def test_draft_writer_is_write_once_and_verifier_rejects_authorization_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = load_script(WRITER, "m2_post33_draft_writer")
    verifier = load_script(VERIFIER, "m2_post33_draft_verifier")
    # Full current-file SHA verification is exercised by the official verifier
    # and by the small same-size mutation test above.  Keep this orthogonal
    # write-once/authorization unit test from re-reading roughly 15 GB.
    monkeypatch.setattr(verifier, "verify_pinned_file", lambda row: Path(row["path"]))
    out = tmp_path / "draft.json"
    writer.write(out, SCORE_FREE_AUDIT)
    verified = verifier.verify(out)
    assert verified["status"] == "PASS_DRAFT_V2_HASH_HARDENED_NOT_AUTHORIZED"
    assert verified["gpu_launch_authorized"] is False
    with pytest.raises(FileExistsError):
        writer.write(out, SCORE_FREE_AUDIT)
    mutated = json.loads(out.read_text(encoding="utf-8"))
    mutated["authorization"]["gpu_launch_authorized"] = True
    bad = tmp_path / "mutated.json"
    bad.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError):
        verifier.verify(bad)
