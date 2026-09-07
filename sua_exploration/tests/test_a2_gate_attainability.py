"""Read-only gate-attainability coverage for the frozen A2 v2 aggregator.

Does not edit the A2 aggregator, contract, thresholds, receipts, or logs.
Constructs synthetic domain matrices only.  Never opens sealed sub-C test NWBs.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SUA = ROOT / "sua_exploration"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SUA) not in sys.path:
    sys.path.insert(0, str(SUA))
if str(SUA / "tests") not in sys.path:
    sys.path.insert(0, str(SUA / "tests"))

from sua_exploration.mc_maze.gate_attainability import (  # noqa: E402
    assert_gate_can_act,
    assert_wilcoxon_threshold_attainable,
    wilcoxon_threshold_attainability,
)
from sua_exploration.mc_maze import a2_matched_subject_shift_v2_core as core  # noqa: E402
from test_a2_matched_subject_shift_v2 import (  # noqa: E402
    _official_preflight_payload,
    _receipt,
    _write_immutable,
)


AGGREGATOR_PATH = SUA / "scripts" / "aggregate_a2_matched_subject_shift_v2.py"
BOOTSTRAP_DRAWS = 401
BOOTSTRAP_SEED = 20260813
# The constructed all-zero must-fail matrix is a valid Wilcoxon fail path; scipy
# warns while switching from exact to a normal approximation, which the frozen
# aggregator already maps to a non-passing p-value.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Exact p-value calculation does not work if there are zeros:UserWarning"
)


def _aggregator():
    spec = importlib.util.spec_from_file_location("a2_v2_aggregator_attainability", AGGREGATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _exact_mean(values: Mapping[str, float], sessions: tuple[str, ...]) -> float:
    return sum(float(values[session]) for session in sessions) / len(sessions)


def _apply_session_r2(receipt: dict, values: Mapping[str, float]) -> dict:
    sessions = tuple(receipt["domain_sessions"])
    ordered = {session: float(values[session]) for session in sessions}
    epoch_mean = _exact_mean(ordered, sessions)
    for epoch_row in receipt["per_epoch"].values():
        epoch_row["per_session_r2"] = dict(ordered)
        epoch_row["mean_r2"] = epoch_mean
    receipt["per_session_mean_r2"] = dict(ordered)
    receipt["mean_r2"] = epoch_mean
    return receipt


def _write_delta_matrix(
    root: Path,
    *,
    within_delta: np.ndarray,
    external_delta: np.ndarray,
    z4_base: float = 0.20,
) -> Path:
    """Write a valid 12-receipt A2 matrix with controlled T4-Z4 session deltas."""
    within_sessions = core.expected_domain_sessions("within_subject")
    external_sessions = core.expected_domain_sessions("external_subject_M")
    assert within_delta.shape == (len(core.SEEDS), len(within_sessions))
    assert external_delta.shape == (len(core.SEEDS), len(external_sessions))
    root.mkdir(parents=True, exist_ok=True)
    bindings = core.current_implementation_bindings()
    official_sha = _write_immutable(
        core.official_preflight_path(result_root=root),
        _official_preflight_payload(root, bindings=bindings),
    )
    for seed_index, seed in enumerate(core.SEEDS):
        for domain, delta_row, sessions in (
            ("within_subject", within_delta[seed_index], within_sessions),
            ("external_subject_M", external_delta[seed_index], external_sessions),
        ):
            z4_values = {session: z4_base for session in sessions}
            t4_values = {
                session: z4_base + float(delta_row[session_index])
                for session_index, session in enumerate(sessions)
            }
            z4 = _apply_session_r2(
                _receipt("source_z4", seed, domain, official_sha=official_sha, bindings=bindings),
                z4_values,
            )
            t4 = _apply_session_r2(
                _receipt("source_t4", seed, domain, official_sha=official_sha, bindings=bindings),
                t4_values,
            )
            _write_immutable(core.domain_result_path("source_z4", seed, domain, result_root=root), z4)
            _write_immutable(core.domain_result_path("source_t4", seed, domain, result_root=root), t4)
    return root


def _constant_delta(n_session: int, value: float) -> np.ndarray:
    return np.full((len(core.SEEDS), n_session), float(value), dtype=np.float64)


def _ranked_positive_delta(n_session: int, start: float = 0.05) -> np.ndarray:
    row = start + 0.01 * np.arange(n_session, dtype=np.float64)
    return np.broadcast_to(row, (len(core.SEEDS), n_session)).copy()


def _aggregate(result_dir: Path) -> dict:
    return _aggregator().aggregate(
        result_dir=result_dir,
        bootstrap_draws=BOOTSTRAP_DRAWS,
        bootstrap_seed=BOOTSTRAP_SEED,
    )


@pytest.fixture(scope="module")
def a2_must_pass(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("a2_must_pass")
    return _aggregate(
        _write_delta_matrix(
            root,
            within_delta=_ranked_positive_delta(6, start=0.05),
            external_delta=_ranked_positive_delta(15, start=0.25),
        )
    )


@pytest.fixture(scope="module")
def a2_must_fail_null(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("a2_must_fail_null")
    return _aggregate(
        _write_delta_matrix(
            root,
            within_delta=_constant_delta(6, 0.0),
            external_delta=_constant_delta(15, 0.0),
        )
    )


@pytest.fixture(scope="module")
def a2_mean_fail_signs_pass(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("a2_mean_fail")
    return _aggregate(
        _write_delta_matrix(
            root,
            within_delta=_constant_delta(6, 0.010),
            external_delta=_constant_delta(15, 0.020),
        )
    )


@pytest.fixture(scope="module")
def a2_sign_fail_mean_pass(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("a2_sign_fail")
    within = _constant_delta(6, 0.01)
    external = _constant_delta(15, 0.20)
    external[2] = 0.0  # seed 44 interaction = -0.01
    return _aggregate(_write_delta_matrix(root, within_delta=within, external_delta=external))


def test_a2_primary_compound_gate_can_act(a2_must_pass: dict, a2_must_fail_null: dict) -> None:
    assert_gate_can_act(
        name="a2.interaction.passes_all_gates",
        evaluate=lambda payload: payload["interaction"]["passes_all_gates"],
        must_pass=a2_must_pass,
        must_fail=a2_must_fail_null,
    )
    assert a2_must_pass["interaction"]["verdict"]["name"] == "subject_shift_interaction_effective"
    assert a2_must_fail_null["interaction"]["verdict"]["name"] == "interaction_ineffective"


def test_a2_primary_mean_gate_can_act(
    a2_must_pass: dict, a2_mean_fail_signs_pass: dict
) -> None:
    assert_gate_can_act(
        name="a2.interaction.mean_interaction_at_least_0p03",
        evaluate=lambda payload: payload["interaction"]["gates"]["mean_interaction_at_least_0p03"],
        must_pass=a2_must_pass,
        must_fail=a2_mean_fail_signs_pass,
    )
    assert a2_mean_fail_signs_pass["interaction"]["gates"]["all_three_seed_interactions_strictly_positive"] is True


def test_a2_primary_sign_gate_can_act(a2_must_pass: dict, a2_sign_fail_mean_pass: dict) -> None:
    assert_gate_can_act(
        name="a2.interaction.all_three_seed_interactions_strictly_positive",
        evaluate=lambda payload: payload["interaction"]["gates"]["all_three_seed_interactions_strictly_positive"],
        must_pass=a2_must_pass,
        must_fail=a2_sign_fail_mean_pass,
    )
    assert a2_sign_fail_mean_pass["interaction"]["gates"]["mean_interaction_at_least_0p03"] is True


def test_a2_primary_bootstrap_ci_gate_can_act(a2_must_pass: dict, a2_must_fail_null: dict) -> None:
    assert_gate_can_act(
        name="a2.interaction.hierarchical_bootstrap_95ci_lower_positive",
        evaluate=lambda payload: payload["interaction"]["gates"]["hierarchical_bootstrap_95ci_lower_positive"],
        must_pass=a2_must_pass,
        must_fail=a2_must_fail_null,
    )


def test_a2_kill_rule_can_fire(a2_must_pass: dict, a2_must_fail_null: dict) -> None:
    assert_gate_can_act(
        name="a2.interaction.ineffective_kill_rule",
        evaluate=lambda payload: payload["interaction"]["verdict"]["name"] == "interaction_ineffective",
        must_pass=a2_must_fail_null,
        must_fail=a2_must_pass,
    )


def test_a2_secondary_contrasts_can_act(a2_must_pass: dict, a2_must_fail_null: dict) -> None:
    for contrast in ("within_subject_t4_minus_z4", "external_subject_M_t4_minus_z4"):
        assert_gate_can_act(
            name=f"a2.secondary.{contrast}.passes_all_gates",
            evaluate=lambda payload, key=contrast: payload["secondary_contrasts"][key]["passes_all_gates"],
            must_pass=a2_must_pass,
            must_fail=a2_must_fail_null,
        )
        for clause in (
            "mean_paired_delta_at_least_0p03",
            "all_three_seed_means_positive",
            "all_session_means_positive",
            "hierarchical_seed_then_session_bootstrap_95ci_lower_positive",
            "session_paired_exact_wilcoxon_two_sided_le_0p05",
        ):
            assert_gate_can_act(
                name=f"a2.secondary.{contrast}.{clause}",
                evaluate=lambda payload, key=contrast, item=clause: payload["secondary_contrasts"][key]["gates"][item],
                must_pass=a2_must_pass,
                must_fail=a2_must_fail_null,
            )


def test_a2_secondary_wilcoxon_thresholds_are_attainable_at_session_n() -> None:
    assert_wilcoxon_threshold_attainable(6, 0.05, gate_name="a2.secondary.within.wilcoxon")
    assert_wilcoxon_threshold_attainable(15, 0.05, gate_name="a2.secondary.external.wilcoxon")


def test_a2_seed_level_wilcoxon_remains_withdrawn_because_n3_is_unattainable(
    a2_must_pass: dict,
) -> None:
    report = wilcoxon_threshold_attainability(3, 0.05)
    assert report.attainable is False
    assert a2_must_pass["interaction"]["seed_level_wilcoxon"]["computed"] is False
    assert "three seed-level" in a2_must_pass["interaction"]["seed_level_wilcoxon"]["reason"]


def test_a2_pairing_invariant_can_fail_closed(tmp_path: Path) -> None:
    aggregate = _aggregator()
    must_pass_dir = _write_delta_matrix(
        tmp_path / "pair_pass",
        within_delta=_ranked_positive_delta(6, start=0.05),
        external_delta=_ranked_positive_delta(15, start=0.25),
    )
    must_fail_dir = _write_delta_matrix(
        tmp_path / "pair_fail",
        within_delta=_ranked_positive_delta(6, start=0.05),
        external_delta=_ranked_positive_delta(15, start=0.25),
    )
    path = core.domain_result_path("source_t4", 42, "external_subject_M", result_root=must_fail_dir)
    receipt = dict(_receipt("source_t4", 42, "external_subject_M"))
    live = __import__("json").loads(path.read_text(encoding="utf-8"))
    receipt = _apply_session_r2(receipt, live["per_session_mean_r2"])
    for key in (
        "official_preflight_sha256",
        "implementation_bindings",
        "implementation_bindings_sha256",
        "contract_sha256",
    ):
        receipt[key] = live[key]
    broken_bundle = {str(epoch): "b" * 64 for epoch in core.EPOCH_WINDOW}
    receipt["source_checkpoint_sha256_bundle"] = broken_bundle
    receipt["source_run"]["source_checkpoint_sha256_bundle"] = broken_bundle
    digest = core.canonical_json_sha256(broken_bundle)
    receipt["source_checkpoint_sha256_bundle_sha256"] = digest
    receipt["source_run"]["source_checkpoint_sha256_bundle_sha256"] = digest
    for epoch in receipt["per_epoch"]:
        receipt["per_epoch"][epoch]["checkpoint_sha256"] = broken_bundle[epoch]
    path.unlink()
    Path(f"{path}.sha256").unlink()
    _write_immutable(path, receipt)

    def pairing_passed(result_dir: Path) -> bool:
        payload = aggregate.aggregate(
            result_dir=result_dir,
            bootstrap_draws=BOOTSTRAP_DRAWS,
            bootstrap_seed=BOOTSTRAP_SEED,
        )
        return bool(payload["interaction"]["gates"]["receipt_pairing_and_reuse_invariants_pass"])

    assert_gate_can_act(
        name="a2.interaction.receipt_pairing_and_reuse_invariants_pass",
        evaluate=pairing_passed,
        must_pass=must_pass_dir,
        must_fail=must_fail_dir,
        treat_exception_as_fail=(ValueError, core.A2V2ContractError),
    )
