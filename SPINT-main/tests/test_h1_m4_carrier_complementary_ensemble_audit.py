from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import py_compile

import numpy as np
import pytest

from src.h1_m4_eb_normalized_v2_contract import NormalizedV2ContractError


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/h1_m4_carrier_complementary_ensemble_audit.py"


def _module():
    spec = importlib.util.spec_from_file_location("h1_m4_carrier_complementary_ensemble_audit_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_half_blend_algebra_and_fixed_alpha():
    audit = _module()
    base = np.array([[1.0, 4.0], [3.0, 2.0]])
    joint = np.array([[5.0, 0.0], [7.0, 6.0]])
    observed = audit.fixed_half_blend(base, joint)
    assert np.array_equal(observed, np.array([[3.0, 2.0], [5.0, 4.0]]))
    assert audit.require_fixed_alpha() == 0.5
    with pytest.raises(audit.CarrierComplementaryAuditError, match="frozen at 0.5"):
        audit.fixed_half_blend(base, joint, alpha=0.5000001)
    with pytest.raises(audit.CarrierComplementaryAuditError, match="frozen at 0.5"):
        audit.require_fixed_alpha(0.0)


def test_pooled_and_per_session_variance_weighted_r2():
    audit = _module()
    first, second = audit.H1_M4_FOLD0_TARGET
    target = np.array([[0.0], [2.0], [100.0], [102.0]])
    prediction = np.array([[0.0], [0.0], [100.0], [100.0]])
    summary = audit.r2_summary(target, prediction, (first, first, second, second))
    # SSE=8; global TSS=10004, whereas separate-session TSS=4.
    assert summary["pooled"]["r2"] == pytest.approx(1.0 - 8.0 / 10004.0)
    assert summary["per_session_variance_weighted"]["r2"] == pytest.approx(-1.0)
    assert summary["per_session"][first]["r2"] == pytest.approx(-1.0)
    assert summary["per_session"][second]["r2"] == pytest.approx(-1.0)


def test_atomic_immutable_receipt_refuses_overwrite(tmp_path):
    audit = _module()
    output = tmp_path / "receipt.json"
    written, digest = audit._write_atomic_immutable_json(output, {"schema": "test", "n": 1})
    assert written == output.resolve()
    assert len(digest) == 64
    assert (written.stat().st_mode & 0o777) == 0o444
    assert json.loads(written.read_text()) == {"n": 1, "schema": "test"}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit._write_atomic_immutable_json(output, {"schema": "different"})
    assert json.loads(written.read_text()) == {"n": 1, "schema": "test"}


def test_scope_guard_rejects_formal_minival_and_evalai_before_any_loader():
    audit = _module()
    for forbidden in ("formal_heldout", "sub-HumanPitt-held-in-minival", "evalai_submission"):
        with pytest.raises(NormalizedV2ContractError):
            audit.assert_audit_scope_safe(ROOT / forbidden / "must_not_open.nwb")


def test_config_import_and_script_compile_without_data_access():
    audit = _module()
    py_compile.compile(str(SCRIPT), doraise=True)
    for arm in ("base", "joint"):
        config = audit._validate_resolved_config(
            ROOT / "pilot_artifacts/h1_m4_eb_normalized_v2/gpu_runs" / arm / ".hydra/config.yaml",
            arm,
        )
        assert config.pilot.arm == arm
        assert int(config.trainer.max_epochs) == 50


def test_reference_terminal_receipt_binds_exact_existing_epoch49_pair_without_data_access():
    audit = _module()
    receipt_path = ROOT / "pilot_artifacts/h1_m4_eb_normalized_v2/gpu_runs/h1_m4_eb_normalized_v2_terminal_gate.json"
    reference = audit._read_reference_terminal_receipt(receipt_path)
    paired = reference["checkpoints"]["paired"]
    base = reference["checkpoints"]["base"]
    joint = reference["checkpoints"]["joint"]
    audit._validate_reference_terminal_binding(
        reference,
        paired,
        base_checkpoint_sha256=base["sha256"],
        joint_checkpoint_sha256=joint["sha256"],
        base_config_sha256=base["config_sha256"],
        joint_config_sha256=joint["config_sha256"],
    )
    changed = json.loads(json.dumps(reference))
    changed["checkpoints"]["joint"]["sha256"] = "0" * 64
    with pytest.raises(audit.CarrierComplementaryAuditError, match="current joint"):
        audit._validate_reference_terminal_binding(
            changed,
            paired,
            base_checkpoint_sha256=base["sha256"],
            joint_checkpoint_sha256=joint["sha256"],
            base_config_sha256=base["config_sha256"],
            joint_config_sha256=joint["config_sha256"],
        )


def test_invalid_device_rejected_before_any_data_or_receipt_access(tmp_path):
    audit = _module()
    with pytest.raises(ValueError, match="device must be cuda or cpu"):
        audit.run_posthoc_headroom_audit(
            data_dir=ROOT / "data/000954",
            raw_receipt_path=tmp_path / "raw.json",
            eb_receipt_path=tmp_path / "eb.json",
            shared_cache_dir=tmp_path / "cache",
            base_checkpoint_path=tmp_path / "base.ckpt",
            joint_checkpoint_path=tmp_path / "joint.ckpt",
            base_config_path=tmp_path / "base.yaml",
            joint_config_path=tmp_path / "joint.yaml",
            reference_terminal_receipt_path=tmp_path / "terminal.json",
            output_path=tmp_path / "not_written.json",
            device="tpu",
        )
    assert not (tmp_path / "not_written.json").exists()
