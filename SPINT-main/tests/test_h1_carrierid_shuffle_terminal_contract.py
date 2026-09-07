"""Static, no-target contract tests for the H-C/H-RS/H-LS terminal evaluator."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.h1_m4_eb_normalized_v2_contract import NormalizedV2ContractError
from scripts.h1_carrierid_shuffle_terminal_evaluate import _validate_shared_source_binding


ROOT = Path(__file__).resolve().parents[1]


def test_terminal_preflight_is_static_and_never_imports_target_loading_primitives():
    source = (ROOT / "scripts/h1_carrierid_shuffle_terminal_preflight.py").read_text(encoding="utf-8")
    assert "load_target_records" not in source
    assert "H1M4EBNormalizedV2DataModule" not in source
    assert '"target_opened": False' in source
    assert '"cuda_launched": False' in source


def test_terminal_evaluator_has_predeclared_asymmetric_target_views_and_float64_reference_metric():
    source = (ROOT / "scripts/h1_carrierid_shuffle_terminal_evaluate.py").read_text(encoding="utf-8")
    assert 'ARM_TO_TARGET_VARIANT = {"full": "full", "rs": "row", "ls": "label"}' in source
    assert '"h_rs": full_dataset.with_intervention(ARM_TO_TARGET_VARIANT["rs"])' in source
    assert '"h_ls": full_dataset.with_intervention(ARM_TO_TARGET_VARIANT["ls"])' in source
    assert "from scripts.h1_carrierid_evaluate import _evaluate" in source
    assert '"r2_sse_tss_accumulator_dtype": "float64"' in source
    assert "with_intervention" in source
    assert "target_optimizer_steps" in source
    assert "target_backward_steps" in source


def test_terminal_evaluator_binds_immutable_source_closure_before_target_loader_call():
    source = (ROOT / "scripts/h1_carrierid_shuffle_terminal_evaluate.py").read_text(encoding="utf-8")
    closure_call = source.index("source_closure = _verify_source_closure")
    target_call = source.index("target = load_target_records")
    assert closure_call < target_call
    assert "assert_immutable_receipt(preflight_path, PREFLIGHT_STATUS)" in source
    assert "if int(checkpoint.get(\"epoch\", -1)) != 49" in source
    assert '"selected_by": "fixed_terminal_epoch_no_selection"' in source


def test_rs_ls_intervention_manifest_is_allowed_to_differ_but_shared_source_hashes_are_not():
    """Regression for the exact fail-closed pre-target bug fixed in evaluator v2."""
    common = {
        "fold_date": "19250101",
        "normalizer_sha256": "n" * 64,
        "source_cache_sha256": "r" * 64,
        "normalized_cache_sha256": "z" * 64,
        "source_hashes_sha256": "h" * 64,
        "initial_state_sha256": "i" * 64,
    }
    h_c = {"source_manifest_sha256": "c" * 64, **common}
    h_rs = {"source_manifest_sha256": "s" * 64, **common}
    observed = _validate_shared_source_binding(h_c, h_rs, "RS")
    assert "source_manifest_sha256" not in observed
    assert observed["normalizer_sha256"] == "n" * 64

    broken = dict(h_rs)
    broken["normalizer_sha256"] = "x" * 64
    with pytest.raises(NormalizedV2ContractError, match="normalizer_sha256"):
        _validate_shared_source_binding(h_c, broken, "RS")


def test_terminal_preflight_hashes_evaluator_and_this_contract_test():
    source = (ROOT / "scripts/h1_carrierid_shuffle_terminal_preflight.py").read_text(encoding="utf-8")
    assert "scripts/h1_carrierid_shuffle_terminal_evaluate.py" in source
    assert "tests/test_h1_carrierid_shuffle_terminal_contract.py" in source
    assert "scripts/h1_carrierid_shuffle_preflight.py" in source
