from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.h1_m4_eb_normalized_v2_contract import (
    NORMALIZER_FLOOR,
    NORMALIZER_FORMULA,
    NormalizedV2ContractError,
    canonical_sha256,
    fit_source_scalar_normalizer,
    reject_target_or_heldout_scope,
    validate_intervention_normalization_algebra,
    validate_paired_checkpoint_bindings,
    validate_raw_normalized_roundtrip,
)


ROOT = Path(__file__).resolve().parents[1]


def _source_carriers() -> np.ndarray:
    return np.load(
        ROOT / "pilot_artifacts/h1_m4_eb_fold0/preflight_cache/fold0_all_source_m4_carriers.npz",
        allow_pickle=False,
    )["carriers"]


def test_source_only_scalar_exact_shape_formula_and_rms():
    carriers = _source_carriers()
    normalizer = fit_source_scalar_normalizer(carriers, "a" * 64)
    assert carriers.shape == (116, 176, 4)
    assert normalizer.s_src == pytest.approx(6.8927984976088e-06, rel=0, abs=1e-20)
    assert normalizer.floor == NORMALIZER_FLOOR
    assert normalizer.manifest["formula"] == NORMALIZER_FORMULA
    assert normalizer.normalized_global_rms == pytest.approx(1.0, rel=0, abs=1e-12)


def test_raw_normalized_roundtrip_and_intervention_zero_row_label_algebra():
    carriers = _source_carriers()
    normalizer = fit_source_scalar_normalizer(carriers, "b" * 64)
    evidence = validate_raw_normalized_roundtrip(normalizer, carriers)
    assert evidence["invertible"] and evidence["max_abs_error"] < 1e-18
    algebra = validate_intervention_normalization_algebra(normalizer)
    assert algebra["zero_literal_post_normalization"]
    assert algebra["row_equals_normalized_raw_row"]
    assert algebra["label_nonidentity"]


def test_normalizer_rejects_target_shaped_or_nonfinite_source_cache():
    with pytest.raises(NormalizedV2ContractError):
        fit_source_scalar_normalizer(np.zeros((2, 176, 4)), "c" * 64)
    bad = _source_carriers().copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(NormalizedV2ContractError):
        fit_source_scalar_normalizer(bad, "c" * 64)


def test_pair_binding_is_same_state_and_fails_closed():
    fields = {
        "fold_date": "19250101",
        "source_manifest_sha256": "a" * 64,
        "normalizer_sha256": "b" * 64,
        "source_cache_sha256": "c" * 64,
        "normalized_cache_sha256": "d" * 64,
        "source_hashes_sha256": "f" * 64,
        "initial_state_sha256": "e" * 64,
    }
    base = dict(fields)
    joint = dict(fields)
    joint["arm"] = "joint"
    assert validate_paired_checkpoint_bindings(base, joint)["normalizer_sha256"] == "b" * 64
    joint["normalizer_sha256"] = "f" * 64
    with pytest.raises(NormalizedV2ContractError):
        validate_paired_checkpoint_bindings(base, joint)


def test_v2_configs_are_present_and_named_without_v1_edits():
    paths = [
        ROOT / "configs/data/falcon_h1_m4_eb_normalized_v2.yaml",
        ROOT / "configs/model/falcon_h1_m4_eb_normalized_v2.yaml",
        ROOT / "configs/callbacks/h1_m4_eb_normalized_v2_terminal.yaml",
        ROOT / "configs/experiment/h1_m4_eb_normalized_v2_base.yaml",
        ROOT / "configs/experiment/h1_m4_eb_normalized_v2_joint.yaml",
    ]
    assert all(path.is_file() for path in paths)
    assert all("h1_m4_eb_normalized_v2" in path.name for path in paths)
    assert "normalizer_floor: 1.0e-12" in paths[0].read_text()


def test_source_scope_guard_rejects_target_and_heldout_paths():
    with pytest.raises(NormalizedV2ContractError):
        reject_target_or_heldout_scope(ROOT / "data/000954/sub-HumanPitt-held-in-minival")
    with pytest.raises(NormalizedV2ContractError):
        reject_target_or_heldout_scope(ROOT / "formal_heldout")


def test_evaluator_rejects_invalid_device_before_any_target_access(tmp_path):
    from scripts.h1_m4_eb_normalized_v2_evaluate import evaluate_terminal_pilot

    with pytest.raises(ValueError, match="cuda or cpu"):
        evaluate_terminal_pilot(
            data_dir=ROOT / "data/000954",
            raw_receipt_path=tmp_path / "raw.json",
            eb_receipt_path=tmp_path / "eb.json",
            shared_cache_dir=tmp_path / "cache",
            base_checkpoint_path=tmp_path / "base.ckpt",
            joint_checkpoint_path=tmp_path / "joint.ckpt",
            base_config_path=tmp_path / "base.yaml",
            joint_config_path=tmp_path / "joint.yaml",
            output_path=tmp_path / "must_not_exist.json",
            device="tpu",
        )
    assert not (tmp_path / "must_not_exist.json").exists()


def test_preflight_source_scope_has_no_target_loader_call():
    text = (ROOT / "scripts/h1_m4_eb_normalized_v2_preflight.py").read_text()
    assert "load_target_records" not in text
    assert "target_record" in text


def test_paired_launcher_source_closure_passes_and_fails_closed(tmp_path):
    from scripts.h1_m4_eb_normalized_v2_paired_launcher import _verify_source_closure

    source = tmp_path / "source.py"
    source.write_text("frozen-v2\n", encoding="utf-8")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    evidence = _verify_source_closure(tmp_path, {"source_sha256": {"source.py": expected}})
    assert evidence["verified"] and evidence["file_count"] == 1
    source.write_text("drifted-v2\n", encoding="utf-8")
    with pytest.raises(NormalizedV2ContractError, match="SHA-256 drift"):
        _verify_source_closure(tmp_path, {"source_sha256": {"source.py": expected}})


@pytest.mark.parametrize("arm", ["base", "joint"])
def test_hydra_compose_and_instantiate_v2_arm(arm, tmp_path):
    import hydra
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs"), job_name="h1_m4_eb_normalized_v2_test"):
        config = compose(
            config_name="train.yaml",
            overrides=[
                f"experiment=h1_m4_eb_normalized_v2_{arm}",
                "trainer.accelerator=cpu",
                f"pilot.shared_cache_dir={tmp_path / arm}",
                f"paths.root_dir={ROOT}",
                f"paths.work_dir={ROOT}",
                f"paths.data_dir={ROOT / 'data'}",
                f"paths.output_dir={tmp_path / 'output' / arm}",
                "hydra.run.dir=/tmp/h1_m4_eb_normalized_v2_compose",
            ],
        )
    assert float(config.data.normalizer_floor) == NORMALIZER_FLOOR
    datamodule = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model)
    assert datamodule.__class__.__name__ == "H1M4EBNormalizedV2DataModule"
    assert model.pilot_arm == arm
