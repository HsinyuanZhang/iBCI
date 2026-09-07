"""CPU gates for M1-TEMPORAL-v2 / rSyn3-refit-v1. Not a P-carrier repair."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from tfpd_exploration.src.m1_temporal_v2 import plan
from tfpd_exploration.src.m1_temporal_v2.bank import load_source_bank
from tfpd_exploration.src.m1_temporal_v2.cpu_budget import S2_TIMEOUT_SUBMISSION
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import (
    M1TemporalFlatDecoder,
    M1TemporalRouteDecoder,
    prove_zero_gate_equals_flat,
    shared_parameter_max_abs_diff,
)


REPO = Path(__file__).resolve().parents[2]
V2_SRC = REPO / "tfpd_exploration/src/m1_temporal_v2"
FORBIDDEN_IMPORTS = {
    "build_fold0_carrier_bank",
    "load_frozen_m1_materializer",
}


def _python_files() -> list[Path]:
    return sorted(V2_SRC.glob("*.py"))


def test_v2_modules_do_not_call_old_carrier_refit() -> None:
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = {alias.name for alias in node.names}
                assert not (names & FORBIDDEN_IMPORTS), path
            if isinstance(node, ast.Attribute) and node.attr == "build_fold0_carrier_bank":
                raise AssertionError(f"{path} still references build_fold0_carrier_bank")


def test_revision_is_named_and_not_old_stage0() -> None:
    assert plan.REVISION == "M1-TEMPORAL-v2"
    assert plan.CARRIER_NAME == "rSyn3-refit-v1"
    assert plan.OLD_TARGET_M10_DIGEST.startswith("2d1a638e")
    assert S2_TIMEOUT_SUBMISSION == 581938


def test_sealed_bank_load_is_stable_and_not_old_digest() -> None:
    first = load_source_bank()
    second = load_source_bank()
    assert first["revision"] == plan.REVISION
    assert first["npz_sha256"] == second["npz_sha256"]
    assert first["receipt"]["digests"]["d0"] != plan.OLD_D0_DIGEST
    assert first["receipt"]["target_loaded"] is False
    assert first["receipt"]["query_values_read"] is False
    assert set(first["normalized"]) == set(plan.SOURCE_SESSIONS)
    for name in plan.SOURCE_SESSIONS:
        assert first["normalized"][name].shape[1] == 4
        np.testing.assert_array_equal(first["normalized"][name], second["normalized"][name])


def test_source_only_loader_does_not_open_outer_query() -> None:
    from tfpd_exploration.src.m1_temporal_v2.data import build_source_only_datamodule, isolation_receipt

    data = build_source_only_datamodule()
    receipt = isolation_receipt(data)
    assert receipt["target_path_resolved"] is False
    assert receipt["val_heldin_dataset"] is False
    assert receipt["train_sessions"] == list(plan.SOURCE_SESSIONS)
    assert receipt["manifest"]["target_query_values_read_by_fit"] is False
    assert receipt["manifest"]["source_only"] is True
    assert getattr(data, "target_path", None) is None
    assert data.val_heldin_dataset is None


def test_zero_gate_and_shared_init() -> None:
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_config import M1_TEMPORAL
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import M1Bank

    g = torch.Generator().manual_seed(0)
    bank = M1Bank(
        E0=torch.randn(M1_TEMPORAL.n_units, M1_TEMPORAL.e0_dim, generator=g),
        T=torch.randn(M1_TEMPORAL.n_units, M1_TEMPORAL.hc_dim, generator=g),
        unit_mask=torch.ones(M1_TEMPORAL.n_units, dtype=torch.bool),
    )
    flat = M1TemporalFlatDecoder(seed=42)
    route = M1TemporalRouteDecoder(seed=42, flat_template=flat)
    assert shared_parameter_max_abs_diff(flat, route) == 0.0
    prove_zero_gate_equals_flat(flat, route, torch.randn(2, 32, M1_TEMPORAL.n_units), bank)


def test_b3_encoder_loads_strict_after_setup() -> None:
    from tfpd_exploration.src.m1_temporal_v2.calibration import load_frozen_b3_student, sfix_decoder_keys

    student = load_frozen_b3_student()
    assert student.id_encoder is not None
    assert all(not p.requires_grad for p in student.id_encoder.parameters())
    decoder_keys = sfix_decoder_keys()
    encoder_names = {f"student.id_encoder.{name}" for name in student.id_encoder.state_dict()}
    assert decoder_keys.isdisjoint(encoder_names)
