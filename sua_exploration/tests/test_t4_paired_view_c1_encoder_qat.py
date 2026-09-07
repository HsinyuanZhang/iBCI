from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT / "sua_exploration",
    ROOT / "sua_exploration/scripts",
    ROOT / "software-to-hardware",
    ROOT / "streaming_calibration_exp",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load(name: str, filename: str):
    path = ROOT / "sua_exploration/scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


writer = _load(
    "c1_qat_writer_test", "write_t4_paired_view_c1_encoder_qat_prelaunch.py"
)
trainer = _load(
    "c1_qat_trainer_test", "train_t4_paired_view_c1_encoder_qat.py"
)
aggregate = _load(
    "c1_qat_aggregate_test", "aggregate_t4_paired_view_c1_encoder_qat.py"
)


def _ptq_trigger(**overrides):
    value = {
        "schema_version": 1,
        "scope": "C1 shared B3S/T4 identity encoder W8A8 INT32 + FP32 decoder",
        "method": "uniform_three_seed_ptq",
        "seeds": [42, 43, 44],
        "views": ["sua", "pseudo_mua"],
        "prelaunch_sha256": "a" * 64,
        "decoder_quantized_in_this_program": False,
        "decoder_precision": "FP32",
        "formal_test_files_opened": False,
        "ptq_pass": False,
        "next_step": "trigger_uniform_three_seed_encoder_qat",
        "qat_triggered_by_this_aggregate": True,
        "mixed_ptq_qat_seed_aggregate_forbidden": True,
        "gates": {"accuracy": True, "saturation": False},
    }
    value.update(overrides)
    return value


def test_qat_trigger_requires_frozen_three_seed_ptq_failure() -> None:
    writer.validate_ptq_trigger(_ptq_trigger(), "a" * 64)
    with pytest.raises(ValueError, match="does not authorize QAT"):
        writer.validate_ptq_trigger(_ptq_trigger(ptq_pass=True), "a" * 64)
    with pytest.raises(ValueError, match="failed frozen gate"):
        writer.validate_ptq_trigger(
            _ptq_trigger(gates={"accuracy": True, "saturation": True}), "a" * 64
        )


def _fake_paired_side(max_sua: float, max_pseudo: float):
    def dataset(maximum: float):
        sessions = {
            f"session-{index}": SimpleNamespace(
                side_features=np.asarray([[maximum, -maximum / 2]], dtype=np.float32)
            )
            for index in range(27)
        }
        return SimpleNamespace(sessions=sessions)

    return SimpleNamespace(
        train_dataset=SimpleNamespace(
            sua_dataset=dataset(max_sua), pseudo_mua_dataset=dataset(max_pseudo)
        )
    )


def test_source_side_scale_floor_uses_both_views_and_fixed_margin() -> None:
    floor, maxima = trainer.source_side_scale_floor(_fake_paired_side(3.0, 5.0))
    assert maxima == {"sua": 3.0, "pseudo_mua": 5.0}
    assert floor == pytest.approx(1.10 * 5.0 / 127.0)


def test_mean_scale_projection_is_source_floor_and_bounded() -> None:
    from b3_fake_quant import B3QATScales
    from b3_qat_encoder import QATEarlyPoolEncoder
    from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder

    base = SideFeatureEarlyPoolEncoder(
        trial_length=100,
        window_size=50,
        hidden_dim=64,
        side_dim=4,
        electrode_embed_dim=0,
        num_electrodes=0,
    )
    quant = QATEarlyPoolEncoder(
        base,
        B3QATScales(0.1, 0.1, 0.02, 0.1, 0.1, 0.1),
        num_trials=30,
        learnable_scales=True,
    )
    trainer._project_mean_scale_floor(quant, 0.05)
    assert float(quant.shared_scales.s_mean.value()) >= 0.05 - 1e-7
    with pytest.raises(ValueError, match="exceeds"):
        trainer._project_mean_scale_floor(quant, 0.2)


def test_qat_protocol_separates_integer_codes_from_float_dequant() -> None:
    source = (
        ROOT / "sua_exploration/scripts/train_t4_paired_view_c1_encoder_qat.py"
    ).read_text()
    assert "integer_E_q_code_mismatch_count" in source
    assert "E_dequant_torch_minus_numpy_max_abs" in source
    assert "qat_code_parity_pass" in source
    assert "qat_strict_legacy_pass" in source
    assert '"ptq_frozen_failure_reclassified": False' in source


def test_integer_code_parity_uses_real_golden_engine_output_key() -> None:
    trainer_source = (
        ROOT / "sua_exploration/scripts/train_t4_paired_view_c1_encoder_qat.py"
    ).read_text()
    audit_source = (
        ROOT
        / "sua_exploration/scripts/audit_t4_paired_view_c1_ptq_integer_code_parity.py"
    ).read_text()
    engine_source = (ROOT / "software-to-hardware/b3_quant_engine.py").read_text()
    reference_source = (ROOT / "software-to-hardware/b3_qat_align.py").read_text()
    assert '"E": E_out' in engine_source
    assert 'stages_np["E"]' in reference_source
    assert 'numpy_stages["E"]' in trainer_source
    assert 'stages_np["E"]' in audit_source
    assert 'E_out"]' not in trainer_source
    assert 'E_out"]' not in audit_source


def test_qat_is_uniform_source_only_decoder_fp32_and_fixed_epoch() -> None:
    source = (
        ROOT / "sua_exploration/scripts/train_t4_paired_view_c1_encoder_qat.py"
    ).read_text()
    assert "QAT_EPOCHS = 8" in source
    assert "calibration_n_trials=30" in source
    assert "model.student.freeze_decoder()" in source
    assert '"validation_used_for_training": False' in source
    assert '"validation_used_for_epoch_selection": False' in source
    assert '"formal_test_files_opened": False' in source
    runner = (
        ROOT / "sua_exploration/scripts/run_t4_paired_view_c1_encoder_qat.sh"
    ).read_text()
    assert "--epochs 8" in runner
    assert "QAT prelaunch is absent" in runner


def test_qat_aggregate_keeps_strict_legacy_disposition_separate() -> None:
    source = (
        ROOT / "sua_exploration/scripts/aggregate_t4_paired_view_c1_encoder_qat.py"
    ).read_text()
    assert "strict_qat_pass" in source
    assert "integer_code_deployment_pass_but_legacy_float_exact_gate_remains_failed" in source
    assert '"ptq_frozen_failure_reclassified": False' in source


def test_remote_sync_audit_uses_the_cuda_enabled_spint_interpreter() -> None:
    audit = _load(
        "c1_qat_remote_sync_audit_test",
        "audit_t4_paired_view_c1_encoder_qat_remote_sync.py",
    )
    assert audit.REMOTE_PYTHON == "/home/xinyuan/miniconda3/envs/spint/bin/python"
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert 'REMOTE, REMOTE_PYTHON, "-"' in source
    assert 'get("executable") == REMOTE_PYTHON' in source
