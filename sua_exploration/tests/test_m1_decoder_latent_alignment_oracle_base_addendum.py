"""Contracts for the v2 non-circular M1 decoder-latent oracle evidence."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration" / "scripts" / "audit_m1_decoder_latent_alignment_oracle_base_addendum.py"
PROTOCOL = ROOT / "sua_exploration" / "docs" / "M1_DECODER_LATENT_ALIGNMENT_ORACLE_V2_PROTOCOL.md"


def module():
    spec = importlib.util.spec_from_file_location("m1_decoder_latent_oracle_v2", SCRIPT)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def test_decoder_comparison_requires_tensorwise_exact_equality():
    m = module()
    teacher = {"fc.weight": torch.tensor([[1.0, 2.0]]), "fc.bias": torch.tensor([3.0])}
    same = {key: value.clone() for key, value in teacher.items()}
    assert m.compare_decoder_states(teacher, same)["bit_exact"]
    same["fc.weight"][0, 1] = 9.0
    result = m.compare_decoder_states(teacher, same)
    assert not result["bit_exact"]
    assert result["unequal_tensor_keys"] == ["fc.weight"]


def test_teacher_identity_requires_exact_m10_time_unit_support_shape():
    m = module()
    fc_in = torch.nn.Sequential(torch.nn.Linear(1024, 64), torch.nn.ReLU())
    fc_out = torch.nn.Linear(64, 100)
    with pytest.raises(ValueError, match="expected support"):
        m.teacher_identity(torch.zeros(1, 9, 1024, 64), fc_in, fc_out)
    actual = m.teacher_identity(torch.zeros(1, 10, 1024, 64), fc_in, fc_out)
    assert actual.shape == (1, 64, 100)


def test_protocol_locks_distinct_f0_base_teacher_target_and_query_endpoint():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "Delta*    = E_teacher - E0" in text
    assert "F0/B3 student.id_encoder.forward_batch(C)" in text
    assert "all 31 decoder\nstate tensors" in text
    assert "primary development mechanism endpoint" in text
    assert "[210,end)" in text
    assert "selection paired MDE is no larger than `0.015 R2`" in text
    assert "rate-residualized-condition-only" in text
    assert "**not\northogonal Procrustes**" in text
    assert "### A. Deployable shared reduced-rank map" in text
    assert "### B. Target-support teacher oracle" in text
    assert "must not compute, inspect, or fit to `E_teacher_leftout`" in text
    assert "teacher-assisted, neural-only upper bound" in text
    assert "M10 calibration" in text and "MACs" in text and "floating-point state values" in text


def test_v2_receipt_writer_cannot_be_misread_as_a_fit_launcher():
    source = (ROOT / "sua_exploration/scripts/write_m1_decoder_latent_alignment_oracle_v2_receipt.py").read_text()
    assert '"formal_oracle_status": "blocked_pending_root_review"' in source
    assert '"adapter_fit"' in source and '"R2_scoring"' in source
    assert '"GPU_launch"' in source and '"EvalAI_access_or_submission"' in source
