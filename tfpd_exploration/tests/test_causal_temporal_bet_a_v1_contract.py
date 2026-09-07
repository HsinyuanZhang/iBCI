"""Small synthetic adversarial tests for the no-execution Bet A design contract."""

from __future__ import annotations

import importlib.util
import hashlib
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from causal_temporal_bet_a_v1.contract import (  # noqa: E402
    ALTERNATIVES,
    DRY_PLAN_STATUS,
    SEALED_CELL_D_FACTORS,
    ContractViolation,
    dry_plan,
    validate_one_factor_candidate,
)


def candidate() -> dict:
    factors = deepcopy(dry_plan()["sealed_cell_d_factors"])
    factors["temporal_information_flow_operator"] = {
        "proposal_id": "A-CausalMask",
        "proposal_status": "UNFROZEN_DESIGN_ALTERNATIVE_ONLY",
        "changed_factor": "temporal_information_flow_operator",
        "state_reset_semantics": "WINDOW_LOCAL",
        "non_temporal_factor_change_count": 0,
        "implementation_spec_sha256": None,
    }
    return {
        "factors": factors,
        "alternatives": deepcopy(dry_plan()["alternatives"]),
        "disclosures": deepcopy(dry_plan()["required_disclosures"]),
        "implementation": "none",
        "launch": "none",
        "review_state": "UNFROZEN_DESIGN_REVIEW_ONLY",
    }


def test_only_temporal_information_flow_change_is_eligible_for_review() -> None:
    assert validate_one_factor_candidate(candidate()) == ("temporal_information_flow_operator",)


@pytest.mark.parametrize(
    ("factor", "value"),
    [
        ("b3s_calibration_encoder", "normalized T4 only; B3S removal"),
        ("query_slots", "4 slots"),
        ("width_heads_layers_ffn", "width=256; heads=2; layers=1; current Cell D FFN"),
        ("whole_unit_dropout_law", "new dropout law"),
    ],
)
def test_held_cell_d_identity_slots_width_and_dropout_cannot_drift(factor: str, value: str) -> None:
    proposal = candidate()
    proposal["factors"][factor] = value
    with pytest.raises(ContractViolation):
        validate_one_factor_candidate(proposal)


def test_more_than_one_changed_factor_fails_closed() -> None:
    proposal = candidate()
    proposal["factors"]["loss_semantics"] = "last-bin-only MSE"
    with pytest.raises(ContractViolation, match="exactly one scientific factor"):
        validate_one_factor_candidate(proposal)


def test_wrong_single_changed_factor_fails_closed() -> None:
    proposal = candidate()
    proposal["factors"]["temporal_information_flow_operator"]["changed_factor"] = "seed"
    with pytest.raises(ContractViolation, match="changed_factor"):
        validate_one_factor_candidate(proposal)


def test_state_reset_is_required_inside_temporal_contract() -> None:
    proposal = candidate()
    proposal["factors"]["temporal_information_flow_operator"].pop("state_reset_semantics")
    with pytest.raises(ContractViolation, match="schema must be exact"):
        validate_one_factor_candidate(proposal)


def test_temporal_contract_schema_rejects_the_reviewed_hidden_factor_attack() -> None:
    proposal = candidate()
    proposal["factors"]["temporal_information_flow_operator"] = {
        "operator": "causal operator plus hidden dimension 384 and three learned query representatives",
        "state_reset_semantics": "window local",
        "hidden_width": 384,
        "query_count": 3,
    }
    with pytest.raises(ContractViolation, match="schema must be exact"):
        validate_one_factor_candidate(proposal)


@pytest.mark.parametrize(
    "operator",
    [
        "causal map with decoder channels set to 384 and three global tokens",
        "causal map that replaces ReLU with GELU and adds layer normalization",
        "arbitrary temporal prose is forbidden",
    ],
)
def test_temporal_row_rejects_any_arbitrary_prose(operator: str) -> None:
    proposal = candidate()
    proposal["factors"]["temporal_information_flow_operator"]["operator"] = operator
    with pytest.raises(ContractViolation, match="schema must be exact"):
        validate_one_factor_candidate(proposal)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("proposal_id", "SELECTED_DEFAULT", "proposal_id"),
        ("proposal_status", "AUTHORIZED", "proposal_status"),
        ("state_reset_semantics", "UNDECLARED", "state_reset_semantics"),
        ("non_temporal_factor_change_count", 1, "non_temporal_factor_change_count"),
        ("implementation_spec_sha256", "f" * 64, "implementation_spec_sha256"),
    ],
)
def test_temporal_review_schema_rejects_invalid_values(field: str, value: object, match: str) -> None:
    proposal = candidate()
    proposal["factors"]["temporal_information_flow_operator"][field] = value
    with pytest.raises(ContractViolation, match=match):
        validate_one_factor_candidate(proposal)


def test_selected_alternative_and_learned_tables_are_rejected() -> None:
    proposal = candidate()
    proposal["alternatives"]["A-CausalMask"]["selection"] = "selected"
    with pytest.raises(ContractViolation, match="frozen ALTERNATIVES"):
        validate_one_factor_candidate(proposal)
    proposal = candidate()
    proposal["factors"]["teacher_and_tables"] = "one learned unit table; zero learned session tables"
    with pytest.raises(ContractViolation):
        validate_one_factor_candidate(proposal)


def test_alternatives_mapping_is_deeply_frozen_including_description() -> None:
    proposal = candidate()
    proposal["alternatives"]["A-CausalMask"]["description"] = "SELECTED DEFAULT AND AUTHORIZED BY DESCRIPTION"
    with pytest.raises(ContractViolation, match="frozen ALTERNATIVES"):
        validate_one_factor_candidate(proposal)


def test_extra_architecture_selection_and_nonzero_disclosure_are_rejected() -> None:
    proposal = candidate()
    proposal["architecture_selection"] = "A-CausalMask"
    with pytest.raises(ContractViolation, match="unsupported top-level"):
        validate_one_factor_candidate(proposal)
    proposal = candidate()
    proposal["disclosures"]["learned_unit_tables"] = 1
    with pytest.raises(ContractViolation, match="frozen required disclosure contract"):
        validate_one_factor_candidate(proposal)
    proposal = candidate()
    proposal["authorization"] = "AUTHORIZED"
    with pytest.raises(ContractViolation, match="unsupported top-level"):
        validate_one_factor_candidate(proposal)


@pytest.mark.parametrize("review_state", [None, "AUTHORIZED", "SELECTED", "READY_TO_LAUNCH"])
def test_review_state_must_remain_exactly_unfrozen_design_review_only(review_state: str | None) -> None:
    proposal = candidate()
    if review_state is None:
        proposal.pop("review_state")
    else:
        proposal["review_state"] = review_state
    with pytest.raises(ContractViolation, match="review_state"):
        validate_one_factor_candidate(proposal)


@pytest.mark.parametrize("field", ["implementation", "launch"])
def test_operational_top_level_fields_are_required_and_exactly_none(field: str) -> None:
    proposal = candidate()
    proposal.pop(field)
    with pytest.raises(ContractViolation, match=field):
        validate_one_factor_candidate(proposal)
    proposal = candidate()
    proposal[field] = "authorized"
    with pytest.raises(ContractViolation, match=field):
        validate_one_factor_candidate(proposal)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("data_scope", "open target data"),
        ("carrier_fixture", "raw carrier fixture"),
        ("prefix_test", "test final index only"),
        ("prefix_test", "every prefix output equality without state equality"),
        ("target_optimizer_backward_calls", 1),
    ],
)
def test_disclosures_are_frozen_not_merely_nonempty(field: str, value: object) -> None:
    proposal = candidate()
    proposal["disclosures"][field] = value
    with pytest.raises(ContractViolation, match="frozen required disclosure contract"):
        validate_one_factor_candidate(proposal)


@pytest.mark.parametrize("argument", ["--execute", "--launch", "--mint", "--data-path=x", "--gpu=0"])
def test_operational_cli_requests_are_rejected_before_contract_import(argument: str) -> None:
    spec = importlib.util.spec_from_file_location(
        "bet_a_dry_cli", ROOT / "scripts" / "preflight_causal_temporal_bet_a_v1.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract_module = sys.modules.pop("causal_temporal_bet_a_v1.contract", None)
    try:
        with pytest.raises(SystemExit, match="NO_IMPLEMENTATION_NO_DATA_NO_CUDA_NO_WRITE_NO_LAUNCH"):
            module.main([argument])
        assert "causal_temporal_bet_a_v1.contract" not in sys.modules
    finally:
        if contract_module is not None:
            sys.modules["causal_temporal_bet_a_v1.contract"] = contract_module


def _tree_snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_zero_argument_cli_writes_no_bytecode_or_other_files_in_fresh_route(tmp_path: Path) -> None:
    fresh_route = tmp_path / "fresh_route"
    fresh_src = fresh_route / "src"
    fresh_scripts = fresh_route / "scripts"
    shutil.copytree(
        ROOT / "src" / "causal_temporal_bet_a_v1",
        fresh_src / "causal_temporal_bet_a_v1",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    fresh_scripts.mkdir(parents=True)
    fresh_cli = fresh_scripts / "preflight_causal_temporal_bet_a_v1.py"
    shutil.copy2(ROOT / "scripts" / "preflight_causal_temporal_bet_a_v1.py", fresh_cli)
    before = _tree_snapshot(fresh_route)
    environment = dict(os.environ)
    environment.pop("PYTHONDONTWRITEBYTECODE", None)
    environment.pop("PYTHONPYCACHEPREFIX", None)
    completed = subprocess.run(
        [sys.executable, str(fresh_cli)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert DRY_PLAN_STATUS in completed.stdout
    assert _tree_snapshot(fresh_route) == before


def test_dry_plan_is_explicitly_nonoperational_and_has_exact_source_closure() -> None:
    plan = dry_plan()
    assert plan["status"] == DRY_PLAN_STATUS
    assert plan["authorization"] == "none"
    assert plan["data_access"] == "none"
    assert plan["cuda_access"] == "none"
    assert plan["writes"] == "none"
    assert isinstance(plan["source_closure"], tuple)
    assert len(plan["source_closure"]) == 4
    assert {"parameter_count", "mac_estimate", "persistent_state", "latency"}.issubset(plan["required_disclosures"])
    assert all("*" not in path and "rglob" not in path for path in plan["source_closure"])
