"""Focused fail-closed contracts for the repaired label-free H-U evaluator."""
from __future__ import annotations

from dataclasses import replace
import gc
import io
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import hydra
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from lightning.pytorch import Trainer, seed_everything

from scripts import h1_carrierid_hu_terminal_evaluate as hu_eval
from scripts.h1_carrierid_evaluate import _evaluate
from src.data.h1_m4_eb_pilot import (
    H1PilotRecord,
    TrialBlocks,
    load_immutable_source_authority,
)
from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    array_sha256,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_record(session: str, file_sha: str, *, n_bins: int) -> H1PilotRecord:
    rng = np.random.default_rng(sum(session.encode("ascii")))
    neural = rng.poisson(0.3, size=(n_bins, 176)).astype(np.float32)
    velocity = rng.normal(size=(n_bins, 7)).astype(np.float32)
    eval_mask = np.ones(n_bins, dtype=bool)
    trial_num = np.full(n_bins, 5.0, dtype=np.float64)
    trial_num[:20] = np.repeat(np.arange(1.0, 5.0), 5)
    trial_values = (1.0, 2.0, 3.0, 4.0, 5.0)
    trials = []
    for value in trial_values:
        indices = np.flatnonzero(trial_num == value)
        block = indices[:5]
        trials.append(
            TrialBlocks(
                trial_number=value,
                rates=neural[block].sum(axis=0, keepdims=True).astype(np.float64),
                velocity=velocity[block].mean(axis=0, keepdims=True).astype(np.float64),
                block_indices=block[None, :].astype(np.int64),
            )
        )
    return H1PilotRecord(
        session_name=session,
        date="19250101",
        path=Path(f"/synthetic/{session}.nwb"),
        input_sha256=file_sha,
        neural=neural,
        velocity=velocity,
        trial_change=np.zeros(n_bins, dtype=bool),
        eval_mask=eval_mask,
        trial_num=trial_num,
        trial_values=trial_values,
        trials=tuple(trials),
    )


@pytest.fixture()
def synthetic_lineage():
    names = hu_eval.H1_M4_FOLD0_TARGET
    files = {names[0]: "1" * 64, names[1]: "2" * 64}
    records = {
        names[0]: _synthetic_record(names[0], files[names[0]], n_bins=725),
        names[1]: _synthetic_record(names[1], files[names[1]], n_bins=728),
    }
    lineage = hu_eval.build_target_lineage(records)
    gate = {
        "sessions": list(names),
        "files": dict(files),
        "strict_query_window_indices_sha256": lineage.window_indices_sha256,
        "support_and_carrier_hashes": {
            name: {
                "trial_values": list(lineage.support[name].trial_values),
                "fifth_trial": lineage.support[name].fifth_trial,
                "query_first_bin": lineage.support[name].query_first_bin,
                "support_sha256": lineage.support[name].support_sha256,
                # Deliberately irrelevant poison-shaped historical field.
                "carrier_sha256": {"full": object(), "row": object(), "label": object()},
            }
            for name in names
        },
    }
    return records, lineage, gate, files


def _validate_synthetic(records, lineage, gate, files):
    return hu_eval.validate_target_lineage(
        records,
        lineage,
        gate,
        expected_file_sha=files,
        expected_query_sha=lineage.window_indices_sha256,
        expected_session_samples=lineage.session_samples,
        expected_total_samples=len(lineage.window_indices),
    )


def test_synthetic_target_file_sha_drift_fails_closed(synthetic_lineage):
    records, lineage, gate, files = synthetic_lineage
    name = hu_eval.H1_M4_FOLD0_TARGET[0]
    mutated = dict(records)
    mutated[name] = replace(records[name], input_sha256="f" * 64)
    with pytest.raises(NormalizedV2ContractError, match="NWB input SHA drifted"):
        _validate_synthetic(mutated, lineage, gate, files)


def test_synthetic_target_support_or_boundary_drift_fails_closed(synthetic_lineage):
    records, lineage, gate, files = synthetic_lineage
    name = hu_eval.H1_M4_FOLD0_TARGET[0]
    drifted = json.loads(json.dumps(gate, default=lambda _value: "unused"))
    drifted["support_and_carrier_hashes"][name]["support_sha256"] = "0" * 64
    with pytest.raises(NormalizedV2ContractError, match="support/boundary lineage drifted"):
        _validate_synthetic(records, lineage, drifted, files)


def test_synthetic_target_query_sha_drift_fails_closed(synthetic_lineage):
    records, lineage, gate, files = synthetic_lineage
    drifted = dict(gate)
    drifted["strict_query_window_indices_sha256"] = "0" * 64
    with pytest.raises(NormalizedV2ContractError, match="query SHA drifted"):
        _validate_synthetic(records, lineage, drifted, files)


def test_historical_carrier_hashes_are_not_consulted(synthetic_lineage):
    records, lineage, gate, files = synthetic_lineage
    result = _validate_synthetic(records, lineage, gate, files)
    assert result["historical_carrier_sha256_fields_consulted"] is False
    source = inspect.getsource(hu_eval.validate_target_lineage)
    assert 'gate_value["carrier_sha256"]' not in source
    assert 'gate_value.get("carrier_sha256")' not in source
    evaluator_source = Path(hu_eval.__file__).read_text(encoding="utf-8")
    assert "validate_target_receipt_binding(" not in evaluator_source
    assert "H1M4EBStrictTargetDataset" not in evaluator_source
    assert "skipped_by_design_for_label_free_HU" in evaluator_source


def _valid_checkpoint():
    return {
        "epoch": 49,
        "global_step": 100,
        "state_dict": {},
        "h1_carrierid": {
            "schema": hu_eval.CHECKPOINT_SCHEMA,
            "arm": "hu",
            "fold_date": "19250101",
            "checkpoint_epoch_zero_based": 49,
            "epochs_completed": 50,
            "selected_by": "fixed_terminal_epoch_no_selection",
            "initial_state_sha256": hu_eval.MATCHED_INITIAL_STATE_SHA,
            "carrier_hidden_dim": 32,
            "carrier_dim": 4,
            "carrier_trial_length": 1024,
            "carrier_mode": "source_hu_label_free_descriptor",
            "carrier_intervention": "hu",
            "hu_version": hu_eval.HU_VERSION,
            "hu_feature_names": list(hu_eval.HU_FEATURE_NAMES),
            "effective_source_carriers_shape": [116, 176, 4],
            "effective_source_carriers_count": 116,
            "label_free": True,
            "used_principal_components": False,
            "deployment_target_optimizer_steps": 0,
            "deployment_target_backward_steps": 0,
        },
    }


@pytest.mark.parametrize(
    ("location", "value", "match"),
    [
        ("epoch", 48, "epoch-49"),
        ("schema", "wrong", "metadata drift at schema"),
        ("arm", "full", "metadata drift at arm"),
        ("initial_state_sha256", "0" * 64, "metadata drift at initial_state_sha256"),
        ("deployment_target_optimizer_steps", 1, "metadata drift at deployment_target_optimizer_steps"),
        ("deployment_target_backward_steps", 1, "metadata drift at deployment_target_backward_steps"),
    ],
)
def test_checkpoint_epoch_schema_arm_and_no_target_update_gate(location, value, match):
    checkpoint = _valid_checkpoint()
    if location == "epoch":
        checkpoint[location] = value
    else:
        checkpoint["h1_carrierid"][location] = value
    with pytest.raises(NormalizedV2ContractError, match=match):
        hu_eval._validate_checkpoint(checkpoint)


def test_target_evaluator_has_no_backward_or_optimizer_and_model_eval_is_immutable():
    target_source = inspect.getsource(hu_eval.H1CarrierIdHuStrictTargetDataset)
    evaluator_source = inspect.getsource(hu_eval.evaluate)
    shared_eval_source = inspect.getsource(_evaluate)
    assert ".backward(" not in target_source + evaluator_source
    assert "optimizer.step(" not in target_source + evaluator_source
    assert "torch.no_grad()" in shared_eval_source
    assert "assert_state_immutable" in shared_eval_source
    assert '"target_optimizer_steps": 0' in evaluator_source
    assert '"target_backward_steps": 0' in evaluator_source


def test_v3_addendum_binds_v2_and_only_repairs_execution_procedure():
    path = hu_eval.ADDENDUM_V3
    assert path.is_file()
    body = json.loads(path.read_text(encoding="utf-8"))
    digest = sha256_file(path)
    assert body["schema"] == "h1_carrierid_h32_hu_execution_addendum_v3"
    assert body["supersedes_only"] == "evaluator_execution_procedure"
    assert body["does_not_supersede"] == "v2_scientific_read_rule"
    assert body["bindings"]["preregistration_v2_sha256"] == hu_eval.PREREG_SHA
    assert body["bindings"]["evaluator_sha256_after_repair"] == "bb5c0e6f18109efaca8c1f2a9583c9ca2caa1cdbff1f81cf606f4fea3877ac06"
    assert body["bindings"]["current_m4_producer_sha256"] == hu_eval.CURRENT_M4_PRODUCER_SHA
    assert body["historical_mismatch_evidence"]["mismatch"] is True
    assert body["execution_procedure_repair"]["historical_carrier_reconstruction_check"] == "skipped_by_design_for_label_free_HU"
    assert body["execution_procedure_repair"]["not_claimed_as_passed"] is True
    sidecar = Path(f"{path}.sha256")
    assert sidecar.read_text(encoding="ascii") == f"{digest}  {path.name}\n"
    assert path.stat().st_mode & 0o777 == 0o444
    assert sidecar.stat().st_mode & 0o777 == 0o444


def _valid_resolved_config():
    return OmegaConf.create({
        "seed": 42,
        "train": True,
        "test": False,
        "ckpt_path": None,
        "pilot": {
            "arm": "hu", "fold_date": "19250101", "zero_carrier": False,
            "calibration_n_trials": 4, "batch_size": 32,
            "fixed_terminal_epochs": 50, "no_checkpoint_selection": True,
        },
        "trainer": {
            "max_epochs": 50, "min_epochs": 50, "precision": "32-true",
            "limit_val_batches": 0, "num_sanity_val_steps": 0,
        },
        "data": {
            "_target_": "src.data.h1_carrierid_hu.H1CarrierIdHuDataModule",
            "task": "h1", "batch_size": 32, "window_size": 700,
            "calibration_n_trials": 4, "max_trial_length": 1024,
            "random_calibration": True, "smooth_calibration": False,
            "interpolate_trials": True, "interpolate_trials_kind": "cubic",
            "seed": 42, "fixed_epochs": 50, "normalizer_floor": 1.0e-12,
            "source_authority_dir": str(hu_eval.SOURCE_AUTHORITY),
            "source_authority_receipt_sha256": hu_eval.HU_SOURCE_AUTHORITY_RECEIPT_SHA256,
        },
        "model": {
            "_target_": "src.models.h1_carrierid_hu_module.H1CarrierIdHuLitModule",
            "task": "h1", "pilot_arm": "hu", "fold_date": "19250101",
            "optimizer": {
                "_target_": "torch.optim.Adam", "_partial_": True,
                "lr": 5.0e-5, "weight_decay": 0.0,
            },
            "net": {
                "_target_": "src.models.components.h1_carrierid_spint.H1CarrierIdSpint",
                "carrier_hidden_dim": 32, "carrier_dim": 4,
                "carrier_trial_length": 1024, "zero_carrier": False,
                "model_dim": 1024, "num_covariates": 7, "window_size": 700,
                "num_heads": 64, "num_layers": 1, "num_id_layers": 3,
            },
        },
    })


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("seed", 43),
        ("model.optimizer.lr", 1.0e-4),
        ("trainer.precision", "16-mixed"),
        ("trainer.max_epochs", 49),
        ("data._target_", "wrong.DataModule"),
        ("data.source_authority_dir", "/tmp/wrong-authority"),
        ("data.source_authority_receipt_sha256", "0" * 64),
        ("model._target_", "wrong.Model"),
        ("model.net.model_dim", 512),
        ("model.net.carrier_hidden_dim", 64),
        ("model.net.carrier_dim", 8),
    ],
)
def test_full_resolved_config_drift_fails_closed(key, value):
    config = _valid_resolved_config()
    OmegaConf.update(config, key, value)
    with pytest.raises(NormalizedV2ContractError, match=f"config drift at {key}"):
        hu_eval._validate_resolved_config(config)


def test_full_resolved_config_exact_contract_passes():
    hu_eval._validate_resolved_config(_valid_resolved_config())


def test_source_provenance_hash_shape_count_version_and_features_fail_closed():
    manifest = {
        "carrier_cache_sha256": hu_eval.HC_CACHE_SHA,
        "normalized_cache_sha256": "a" * 64,
        "hu_normalizer_sha256": "b" * 64,
        "hu_raw_sha256": "c" * 64,
        "effective_source_carriers_sha256": hu_eval.HU_AUTHORITATIVE_EFFECTIVE_SHA,
        "effective_source_carriers_shape": [116, 176, 4],
        "effective_source_carriers_count": 116,
        "effective_source_carriers_nonidentity_all": True,
        "hu_version": hu_eval.HU_VERSION,
        "hu_feature_names": list(hu_eval.HU_FEATURE_NAMES),
        "hu_s_src": 1.25,
        "source_authority": {
            "receipt_sha256": hu_eval.HU_SOURCE_AUTHORITY_RECEIPT_SHA256,
            "carrier_cache_sha256": hu_eval.HC_CACHE_SHA,
            "transform_sha256": "92c932d059d93aaffae3d676fe247b46971232027fe906cb4b28add93ddee66a",
            "target_nwb_opened": False,
        },
    }
    dm = SimpleNamespace(
        pilot_manifest=lambda: dict(manifest),
        pilot_manifest_sha256="e" * 64,
        hu_normalizer=SimpleNamespace(normalizer_sha256="b" * 64),
    )
    meta = {
        "source_manifest_sha256": "e" * 64,
        "source_cache_sha256": hu_eval.HC_CACHE_SHA,
        "normalized_cache_sha256": "a" * 64,
        "normalizer_sha256": "b" * 64,
        "hu_raw_sha256": "c" * 64,
        "effective_source_carriers_sha256": hu_eval.HU_AUTHORITATIVE_EFFECTIVE_SHA,
        "effective_source_carriers_shape": [116, 176, 4],
        "effective_source_carriers_count": 116,
        "hu_version": hu_eval.HU_VERSION,
        "hu_feature_names": list(hu_eval.HU_FEATURE_NAMES),
    }
    binding = hu_eval._validate_source_checkpoint_binding(meta, dm)
    assert binding["effective_source_carriers_count"] == 116
    for key, bad in (
        ("hu_raw_sha256", "0" * 64),
        ("effective_source_carriers_sha256", "0" * 64),
        ("effective_source_carriers_shape", [1, 176, 4]),
        ("effective_source_carriers_count", 115),
        ("hu_version", "HU-wrong"),
        ("hu_feature_names", ["wrong"]),
    ):
        mutated = dict(meta)
        mutated[key] = bad
        with pytest.raises(NormalizedV2ContractError, match=f"provenance drift at {key}"):
            hu_eval._validate_source_checkpoint_binding(mutated, dm)


@pytest.mark.parametrize("existing", ["output", "sidecar", "both"])
def test_output_pair_conflicts_fail_without_new_files_or_partials(tmp_path, existing):
    output = tmp_path / "receipt.json"
    sidecar = Path(f"{output}.sha256")
    if existing in {"output", "both"}:
        output.write_bytes(b"preexisting-output")
    if existing in {"sidecar", "both"}:
        sidecar.write_bytes(b"preexisting-sidecar")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(FileExistsError, match="output pair conflict"):
        hu_eval.write_immutable_json_o_excl(output, {"new": True})
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert after == before
    assert not any(".tmp." in name for name in after)


def test_transaction_rolls_back_output_if_sidecar_race_wins(tmp_path, monkeypatch):
    output = tmp_path / "receipt.json"
    sidecar = Path(f"{output}.sha256")
    sidecar.write_bytes(b"racing-sidecar")
    monkeypatch.setattr(hu_eval, "_assert_output_pair_fresh", lambda path: (path.resolve(), Path(f"{path.resolve()}.sha256")))
    with pytest.raises(FileExistsError):
        hu_eval.write_immutable_json_o_excl(output, {"new": True})
    assert not output.exists()
    assert sidecar.read_bytes() == b"racing-sidecar"
    assert not any(".tmp." in path.name for path in tmp_path.iterdir())


def test_evaluate_output_conflict_precedes_checkpoint_and_addendum_reads(tmp_path):
    output = tmp_path / "receipt.json"
    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="before checkpoint/target access"):
        hu_eval.evaluate(
            checkpoint=tmp_path / "missing.ckpt",
            config_path=tmp_path / "missing.yaml",
            cache_dir=tmp_path / "cache",
            output=output,
            execution_addendum_sha256="0" * 64,
        )


def test_v4_addendum_is_preserved_as_immutable_predecessor():
    path = hu_eval.ADDENDUM_V4
    assert path.is_file()
    digest = sha256_file(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    assert digest == hu_eval.ADDENDUM_V4_SHA
    assert body["status"] == "SEALED_BEFORE_ANY_HU_GPU_EXECUTION"
    assert body["scientific_read_rule"] == hu_eval._scientific_read_rule_v4()
    assert body["formal_scope"] == hu_eval._formal_scope_v4()
    assert body["execution_contract"] == hu_eval._execution_contract_v4()
    assert body["bindings"]["execution_addendum_v3_sha256"] == hu_eval.ADDENDUM_V3_SHA
    assert body["supersedes_only"] == "evaluator_execution_procedure"
    assert body["does_not_supersede"] == "v2_scientific_read_rule"


def test_v5_addendum_is_preserved_and_its_superseded_closure_is_not_live():
    path = hu_eval.ADDENDUM_V5
    digest = sha256_file(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    assert digest == hu_eval.ADDENDUM_V5_SHA
    assert body["execution_contract"] == hu_eval._execution_contract_v5()
    assert body["bindings"]["execution_addendum_v4_sha256"] == hu_eval.ADDENDUM_V4_SHA
    assert body["bindings"]["source_authority_receipt_sha256"] == hu_eval.HU_SOURCE_AUTHORITY_RECEIPT_SHA256
    assert body["bindings"]["source_authority_cache_sha256"] == hu_eval.HC_CACHE_SHA
    assert body["bindings"]["runtime_closure_sha256"] != hu_eval._code_sha256()
    with pytest.raises(NormalizedV2ContractError, match="runtime closure hash map drift"):
        hu_eval._validate_addendum_v5(path, trusted_external_sha256=digest)
    with pytest.raises(NormalizedV2ContractError, match="SHA-256 drift"):
        hu_eval._validate_addendum_v5(path, trusted_external_sha256="0" * 64)


def test_v5_runtime_closure_tamper_fails_closed(tmp_path):
    body = json.loads(hu_eval.ADDENDUM_V5.read_text(encoding="utf-8"))
    body["bindings"]["runtime_closure_sha256"]["scripts/h1_carrierid_evaluate.py"] = "0" * 64
    path, digest = hu_eval.write_immutable_json_o_excl(tmp_path / "tampered-v5.json", body)
    with pytest.raises(NormalizedV2ContractError, match="runtime closure hash map drift"):
        hu_eval._validate_addendum_v5(path, trusted_external_sha256=digest)


def test_v6_is_preserved_as_superseded_launch_order_evidence():
    digest = sha256_file(hu_eval.ADDENDUM_V6)
    body = json.loads(hu_eval.ADDENDUM_V6.read_text(encoding="utf-8"))
    assert digest == hu_eval.ADDENDUM_V6_SHA
    assert body["bindings"]["execution_addendum_v5_sha256"] == hu_eval.ADDENDUM_V5_SHA
    assert body["bindings"]["runtime_closure_sha256"] != hu_eval._code_sha256()
    assert body["launch_order_reproducibility"]["train_and_evaluator_actual_paths_match"] is True
    assert body["launch_order_reproducibility"]["no_checkpoint_preflight_is_authoritative"] is False
    assert body["source_authority_loader"]["pathname_hash_then_reopen_forbidden"] is True


def test_v7_is_preserved_as_superseded_exact_pair_equality_evidence():
    digest = sha256_file(hu_eval.ADDENDUM_V7)
    body = json.loads(hu_eval.ADDENDUM_V7.read_text(encoding="utf-8"))
    assert digest == hu_eval.ADDENDUM_V7_SHA
    assert body["bindings"]["execution_addendum_v6_sha256"] == hu_eval.ADDENDUM_V6_SHA
    assert body["bindings"]["runtime_closure_sha256"] != hu_eval._code_sha256()
    rule = body["source_hash_equality_rule"]
    assert rule["checkpoint_metadata_must_equal_evaluator_reconstruction"] is True
    assert rule["hardcode_float64_raw_or_normalizer_hash"] is False
    assert rule["hash_contract_relaxed"] is False


def test_v8_is_preserved_as_superseded_tmux_environment_evidence():
    digest = sha256_file(hu_eval.ADDENDUM_V8)
    body = json.loads(hu_eval.ADDENDUM_V8.read_text(encoding="utf-8"))
    assert digest == hu_eval.ADDENDUM_V8_SHA
    assert body["bindings"]["execution_addendum_v7_sha256"] == hu_eval.ADDENDUM_V7_SHA
    assert body["bindings"]["tmux_environment_preflight_sha256"] == hu_eval.TMUX_ENV_PREFLIGHT_SHA
    assert body["bindings"]["runtime_closure_sha256"] != hu_eval._code_sha256()
    assert body["failed_launch_v1"]["preserved_immutable_history"] is True
    assert body["failed_launch_v1"]["training_started"] is False
    assert body["recommended_fresh_output_root"].endswith("h32_fold0_hu_v2_envfix")


def test_v9_external_anchor_terminal_checkpoint_and_live_closure():
    digest = sha256_file(hu_eval.ADDENDUM_V9)
    snapshot, body = hu_eval._validate_addendum_v9(
        hu_eval.ADDENDUM_V9, trusted_external_sha256=digest
    )
    assert snapshot.sha256 == "520f84e7079042e437fa2a93f3c54f9cb64180b2ee34ee644469765b468f507d"
    assert body["bindings"]["execution_addendum_v8_sha256"] == hu_eval.ADDENDUM_V8_SHA
    assert body["bindings"]["terminal_checkpoint_sha256"] == "36833d4b1260bbbc3f69c4840f8f1bd330e8a6c2a3a1cb05c3a1ec63a74cb599"
    assert body["bindings"]["runtime_closure_sha256"] == hu_eval._code_sha256()
    assert body["evaluator_source_reconstruction_repair"]["equality_relaxed"] is False


@pytest.mark.parametrize(
    ("field", "mutator"),
    [
        ("status", lambda body: body.__setitem__("status", "WRONG")),
        ("scientific_read_rule", lambda body: body["scientific_read_rule"].__setitem__("third_escape_reading_forbidden", False)),
        ("formal_scope", lambda body: body["formal_scope"].__setitem__("development_fold0_only", False)),
        ("execution_contract", lambda body: body["execution_contract"]["resolved_config"].__setitem__("seed", 43)),
    ],
)
def test_v5_exact_status_scientific_scope_and_contract_tamper_fails_closed(tmp_path, field, mutator):
    body = json.loads(hu_eval.ADDENDUM_V5.read_text(encoding="utf-8"))
    mutator(body)
    path, digest = hu_eval.write_immutable_json_o_excl(tmp_path / f"tampered-{field}.json", body)
    with pytest.raises(NormalizedV2ContractError, match=f"exact contract drift at {field}"):
        hu_eval._validate_addendum_v5(path, trusted_external_sha256=digest)


def test_gate_hc_hc0_initial_state_references_fail_closed():
    gate = json.loads(hu_eval.GATE.read_text(encoding="utf-8"))
    hu_eval._validate_gate_matched_initial_state(gate)
    gate["checkpoints"]["h_c0_separate_literal_zero"]["metadata"]["initial_state_sha256"] = "0" * 64
    with pytest.raises(NormalizedV2ContractError, match="h_c0_separate_literal_zero matched initial-state drift"):
        hu_eval._validate_gate_matched_initial_state(gate)


def test_source_authority_receipt_and_files_are_exact_immutable_bindings():
    receipt = hu_eval.SOURCE_AUTHORITY / "H1_CARRIERID_HU_SOURCE_AUTHORITY_v1.json"
    body = json.loads(receipt.read_text(encoding="utf-8"))
    assert sha256_file(receipt) == hu_eval.HU_SOURCE_AUTHORITY_RECEIPT_SHA256
    assert body["bindings"]["h_c_h_c0_source_cache_sha256"] == hu_eval.HC_CACHE_SHA
    assert body["bindings"]["transform_sha256"] == "92c932d059d93aaffae3d676fe247b46971232027fe906cb4b28add93ddee66a"
    assert body["fresh_rebuild_blocker_evidence"]["fresh_carrier_cache_sha256"] == "85e0b0655036674d0b84103c83051a50cae830548be40af7bbe2078a2ab331a2"
    for name, binding in body["files"].items():
        path = hu_eval.SOURCE_AUTHORITY / name
        assert path.stat().st_mode & 0o777 == 0o444
        assert path.stat().st_size == binding["size_bytes"]
        assert sha256_file(path) == binding["sha256"]


def test_source_authority_loader_consumes_open_verified_inode_after_path_poison(tmp_path, monkeypatch):
    """An opened authority file is parsed from that inode, not a reopened path."""

    copied = tmp_path / "source_authority_v1"
    shutil.copytree(hu_eval.SOURCE_AUTHORITY, copied)
    plan_body = json.loads(
        (copied / "fold0_frozen_eb_plan.manifest.json").read_text(encoding="utf-8")
    )
    cache_body = json.loads(
        (copied / "fold0_all_source_m4_carriers.manifest.json").read_text(encoding="utf-8")
    )
    records = {}
    for name, input_sha in zip(plan_body["source_sessions"], plan_body["source_input_sha256"]):
        rows = [row for row in cache_body["entries"] if row["session"] == name]
        trial_values: list[float | None] = [None] * (max(row["start_index"] for row in rows) + 4)
        for row in rows:
            for offset, value in enumerate(row["trial_values"]):
                trial_values[int(row["start_index"]) + offset] = float(value)
        assert all(value is not None for value in trial_values)
        records[name] = SimpleNamespace(
            session_name=name,
            input_sha256=input_sha,
            trial_values=tuple(trial_values),
            blocks_for=lambda _value: SimpleNamespace(rates=np.zeros((2, 1))),
            eval_trial_neural=lambda _value: np.zeros((2, 1)),
        )

    victim = copied / "fold0_frozen_eb_plan.manifest.json"
    original = victim.read_bytes()
    moved = copied / "opened-original-plan.manifest.json"
    real_open = os.open
    poisoned = False

    def open_then_poison(path, flags, *args, **kwargs):
        nonlocal poisoned
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path) == victim and not poisoned:
            poisoned = True
            victim.rename(moved)
            victim.write_bytes(b'{"poison":true}\n')
            victim.chmod(0o444)
        return fd

    monkeypatch.setattr(os, "open", open_then_poison)
    plan, _plan_binding, cache, authority = load_immutable_source_authority(
        records, copied, hu_eval.HU_SOURCE_AUTHORITY_RECEIPT_SHA256
    )
    assert poisoned is True
    assert moved.read_bytes() == original
    assert victim.read_bytes() == b'{"poison":true}\n'
    assert plan.transform_sha256 == "92c932d059d93aaffae3d676fe247b46971232027fe906cb4b28add93ddee66a"
    assert cache.manifest["cache_sha256"] == hu_eval.HC_CACHE_SHA
    assert authority["receipt_sha256"] == hu_eval.HU_SOURCE_AUTHORITY_RECEIPT_SHA256


def _torch_save_bytes(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def test_exact_launcher_hu_datamodule_succeeds_from_fresh_source_cache_without_target_or_gpu(tmp_path):
    # Exact src/train.py ordering starts by seeding, before DataModule/model/
    # Trainer construction.  This is the authoritative training provenance;
    # the older unseeded/no-checkpoint preflight was not launch-equivalent.
    seed_everything(42, workers=True)
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs"), job_name="hu_fresh_source_test"):
        config = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=h1_carrierid_hu",
                f"paths.root_dir={ROOT}",
                f"paths.work_dir={ROOT}",
                f"paths.data_dir={ROOT / 'data'}",
                f"paths.output_dir={tmp_path / 'output'}",
                f"pilot.shared_cache_dir={tmp_path / 'fresh-cache'}",
            ],
        )
    dm = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=50,
        logger=False,
        enable_checkpointing=False,
    )
    dm.trainer = trainer
    dm.setup("fit")
    train_manifest = dm.pilot_manifest()
    assert train_manifest["carrier_cache_sha256"] == hu_eval.HC_CACHE_SHA
    assert train_manifest["transform_sha256"] == "92c932d059d93aaffae3d676fe247b46971232027fe906cb4b28add93ddee66a"
    assert train_manifest["source_authority"]["receipt_sha256"] == hu_eval.HU_SOURCE_AUTHORITY_RECEIPT_SHA256
    assert train_manifest["effective_source_carriers_sha256"] == hu_eval.HU_AUTHORITATIVE_EFFECTIVE_SHA
    assert isinstance(train_manifest["hu_raw_sha256"], str) and len(train_manifest["hu_raw_sha256"]) == 64
    assert isinstance(train_manifest["hu_normalizer_sha256"], str) and len(train_manifest["hu_normalizer_sha256"]) == 64
    assert np.isfinite(train_manifest["hu_s_src"]) and train_manifest["hu_s_src"] > 0.0
    assert train_manifest["effective_source_carriers_shape"] == [116, 176, 4]
    assert train_manifest["effective_source_carriers_count"] == 116
    assert train_manifest["hu_version"] == hu_eval.HU_VERSION
    assert train_manifest["hu_feature_names"] == list(hu_eval.HU_FEATURE_NAMES)
    assert train_manifest["target_nwb_opened_during_training_setup"] is False
    assert train_manifest["minival_or_heldout_enumerated"] is False
    batch = next(iter(dm.train_dataloader()))
    assert batch[0].shape == (32, 700, 176)
    assert batch[4].shape == (32, 176, 4)
    assert model.pilot_arm == "hu"
    assert trainer.accelerator.__class__.__name__ == "CPUAccelerator"
    assert config.trainer.accelerator == "gpu"  # exact launcher config, but no Trainer/GPU was instantiated

    # Build the exact metadata-style source binding written into a training
    # checkpoint, serialize it, then follow the evaluator's decisive order:
    # torch.load(checkpoint bytes) before a fresh live source reconstruction.
    metadata = {
        "source_manifest_sha256": dm.pilot_manifest_sha256,
        "source_cache_sha256": hu_eval.HC_CACHE_SHA,
        "normalized_cache_sha256": train_manifest["normalized_cache_sha256"],
        "normalizer_sha256": train_manifest["hu_normalizer_sha256"],
        "hu_raw_sha256": train_manifest["hu_raw_sha256"],
        "effective_source_carriers_sha256": train_manifest["effective_source_carriers_sha256"],
        "effective_source_carriers_shape": train_manifest["effective_source_carriers_shape"],
        "effective_source_carriers_count": train_manifest["effective_source_carriers_count"],
        "hu_version": train_manifest["hu_version"],
        "hu_feature_names": train_manifest["hu_feature_names"],
    }
    checkpoint_bytes = _torch_save_bytes(
        {"state_dict": {"numerical_state_probe": torch.ones(1)}, "h1_carrierid": metadata}
    )
    del batch, dm, model, trainer
    gc.collect()
    loaded = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    eval_dm = hydra.utils.instantiate(config.data)
    eval_dm.setup("fit")
    binding = hu_eval._validate_source_checkpoint_binding(loaded["h1_carrierid"], eval_dm)
    eval_manifest = eval_dm.pilot_manifest()
    for field, manifest_field in (
        ("source_manifest_sha256", None),
        ("normalizer_sha256", "hu_normalizer_sha256"),
        ("hu_raw_sha256", "hu_raw_sha256"),
        ("effective_source_carriers_sha256", "effective_source_carriers_sha256"),
        ("effective_source_carriers_shape", "effective_source_carriers_shape"),
        ("effective_source_carriers_count", "effective_source_carriers_count"),
        ("hu_version", "hu_version"),
        ("hu_feature_names", "hu_feature_names"),
    ):
        live = eval_dm.pilot_manifest_sha256 if manifest_field is None else eval_manifest[manifest_field]
        assert loaded["h1_carrierid"][field] == live
    assert binding["effective_source_carriers_sha256"] == hu_eval.HU_AUTHORITATIVE_EFFECTIVE_SHA
    assert eval_manifest["target_nwb_opened_during_training_setup"] is False
    assert eval_manifest["minival_or_heldout_enumerated"] is False


def test_checkpoint_metadata_and_evaluator_source_binding_require_same_authoritative_hashes():
    source = inspect.getsource(hu_eval._validate_source_checkpoint_binding)
    for name in (
        "HU_AUTHORITATIVE_EFFECTIVE_SHA",
    ):
        assert name in source
    module_source = (ROOT / "src/models/h1_carrierid_hu_module.py").read_text(encoding="utf-8")
    for field in ("hu_raw_sha256", "effective_source_carriers_sha256"):
        assert field in module_source
    base_source = (ROOT / "src/models/h1_carrierid_module.py").read_text(encoding="utf-8")
    for field in ("source_manifest_sha256", "normalizer_sha256"):
        assert field in base_source


def test_actual_terminal_checkpoint_reconstructs_byte_exact_in_evaluator_training_order(tmp_path):
    """Real epoch-49 checkpoint, source-only; target loader is never called."""

    run = ROOT / "pilot_artifacts/h1_carrierid_hu/gpu_runs/h32_fold0_hu_v2_envfix/hu"
    checkpoint_path = run / "checkpoints/fixed_epoch50/epoch_049.ckpt"
    config_path = run / ".hydra/config.yaml"
    assert sha256_file(checkpoint_path) == "36833d4b1260bbbc3f69c4840f8f1bd330e8a6c2a3a1cb05c3a1ec63a74cb599"
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": ""})
    completed = subprocess.run(
        [
            "/home/xinyuan/miniconda3/envs/spint/bin/python",
            str(ROOT / "scripts/h1_carrierid_hu_terminal_evaluate.py"),
            "--pretarget-source-only",
            "--checkpoint", str(checkpoint_path),
            "--config-path", str(config_path),
            "--cache-dir", str(tmp_path / "source-cache"),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result["checkpoint_epoch"] == 49
    assert result["checkpoint_global_step"] == 180500
    binding = result["source_provenance"]
    order = result["source_reconstruction_order"]
    assert binding["source_manifest_sha256"] == "30e33f2e9527010bba60c8c8f1025d9b9e0ab1fd309ce4717b7b79e14116e153"
    assert binding["hu_raw_sha256"] == "7e62fc22446c6674574fecdaa6fdc255c7a621458aa184de01c659e6de4af996"
    assert binding["hu_normalizer_sha256"] == "850c75da4d429a2c4f924b8409af34e765c3668185dfe57587da997627be9d7f"
    assert binding["effective_source_carriers_sha256"] == hu_eval.HU_AUTHORITATIVE_EFFECTIVE_SHA
    assert order["trainer_fit_called"] is False
    assert order["model_forward_or_backward_called"] is False
    assert order["target_opened"] is False
    assert order["order"][-1] == "datamodule.setup(fit)"
    assert result["target_opened"] is False
    assert result["formal_or_organizer_opened"] is False


def test_pretarget_source_validation_rejects_poison_checkpoint_and_config_before_setup(tmp_path):
    run = ROOT / "pilot_artifacts/h1_carrierid_hu/gpu_runs/h32_fold0_hu_v2_envfix/hu"
    checkpoint = run / "checkpoints/fixed_epoch50/epoch_049.ckpt"
    config = run / ".hydra/config.yaml"
    poison_checkpoint = tmp_path / "poison.ckpt"
    poison_checkpoint.write_bytes(checkpoint.read_bytes() + b"poison")
    with pytest.raises(NormalizedV2ContractError, match="terminal checkpoint binding"):
        hu_eval.pretarget_source_only_validate(
            checkpoint=poison_checkpoint, config_path=config, cache_dir=tmp_path / "cache-a"
        )
    poison_config = tmp_path / "poison.yaml"
    poison_config.write_bytes(config.read_bytes() + b"\n# poison\n")
    with pytest.raises(NormalizedV2ContractError, match="resolved config binding"):
        hu_eval.pretarget_source_only_validate(
            checkpoint=checkpoint, config_path=poison_config, cache_dir=tmp_path / "cache-b"
        )


def test_static_real_target_bindings_are_exact_and_fail_closed():
    assert hu_eval.PREREG_SHA == "6a94ee2ac96885d7462d14e47b8ea454dcf6414f07ddf8f29a43660ea602dd8c"
    assert hu_eval.QUERY_SHA == "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"
    assert hu_eval.EXPECTED_TOTAL_SAMPLES == 8965
    assert hu_eval.EXPECTED_SESSION_SAMPLES == {
        "ses-19250101T111740": 6735,
        "ses-19250101T112404": 2230,
    }
    assert hu_eval.EXPECTED_TARGET_FILE_SHA == {
        "ses-19250101T111740": "f8af652a31228f08b26ff8a7ecfd2a6bd05a345e368da8c98be649229f80adb2",
        "ses-19250101T112404": "b946c4cf49f00c2ea8f0051481765ed090294e3e1e5a035cc5716ad93aa052b2",
    }
