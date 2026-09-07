"""No-data adversarial tests for the additive paired Stage-P executor plan."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_subject_m_stagep_live_executor as live  # noqa: E402
import track_b_v2_subject_m_stagep_runtime as runtime  # noqa: E402


CLI = REPO_ROOT / "cebra_exploration/scripts/run_track_b_v2_subject_m_stagep_live_executor.py"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _ml_runtime_import_state() -> dict[str, bool]:
    """Snapshot optional ML runtimes without assuming pytest starts clean."""
    return {name: name in sys.modules for name in ("torch", "cebra")}


def _assert_ml_runtime_import_state_unchanged(before: dict[str, bool]) -> None:
    assert _ml_runtime_import_state() == before


def _admission(view: str, *, root: str = "root", control: str = "control",
               cost: str = "cost", closure: str = "closure") -> dict:
    return {
        "status": live.LIVE_ADMISSION_STATUS,
        "cell": runtime.StagePCell.from_view(view).as_dict(),
        "target_path_resolution_permitted": False,
        "official_stagep_preflight_body_sha256": _sha(f"official-body-{view}"),
        "official_stagep_preflight_sidecar_sha256": _sha(f"official-sidecar-{view}"),
        "root_authorization_pair": {"body_sha256": _sha(root), "sidecar_sha256": _sha(f"{root}-sidecar")},
        "fixed_runtime_control_pair": {"body_sha256": _sha(control), "sidecar_sha256": _sha(f"{control}-sidecar")},
        "fixed_d8it250_gpu_cost_gate": {
            "status": "FIXED_D8IT250_GPU_COST_RECEIPT_VALID__REQUIRED_BUT_NOT_SUFFICIENT_FOR_TARGET_EXECUTION",
            "canonical_body_sha256": _sha(cost),
        },
        "implementation_closure_sha256": _sha(closure),
    }


def test_static_plan_freezes_paired_order_and_exact_execution_contract() -> None:
    ml_imports_before = _ml_runtime_import_state()
    plan = live.build_unadmitted_paired_execution_plan()
    live.validate_paired_execution_plan(plan=plan)
    assert plan["pair_order"] == ["sua", "pseudo_mua"]
    assert plan["fixed_geometry"] == {"output_dimension": 8, "iterations": 10000}
    assert plan["per_view_execution_contract"]["one_28_session_joint_fit"] is True
    assert plan["per_view_execution_contract"]["target_query_enters_encoder_or_readout_fit"] is False
    assert plan["cross_view_contract"]["pMUA_fit_is_independent_from_SUA_fit"] is True
    assert len(plan["producer_interfaces"]) == 8
    assert plan["all_real_target_gpu_cebra_score_operations_disabled"] is True
    assert live.next_paired_execution_state(completed_state="SUA_TERMINAL_PUBLISHED") == "PMUA_START_PUBLISHED"
    with pytest.raises(live.TrackBV2SubjectMStagePLiveExecutorError, match="no successor"):
        live.next_paired_execution_state(completed_state="PAIRED_COMPLETION_PUBLISHED")
    _assert_ml_runtime_import_state_unchanged(ml_imports_before)


def test_each_view_has_exactly_thirteen_distinct_fresh_outputs_and_excludes_official_preflight() -> None:
    plan = live.build_unadmitted_paired_execution_plan()
    expected_score_roles = {
        f"{route}__{decoder}" for route in runtime.ROUTES for decoder in runtime.DECODERS
    }
    for view in live.PAIRED_VIEW_ORDER:
        topology = plan["cell_topology_by_view"][view]
        paths = live._fresh_output_paths(topology)
        assert len(paths) == 13
        assert len(set(paths)) == 13
        assert Path(topology["official_preflight"]) not in paths
        assert set(topology["scores"]) == expected_score_roles
        assert Path(topology["official_preflight"]).parent == Path(topology["cell_root"])
        assert Path(topology["start"]) in paths
        assert Path(topology["target_materialization"]) in paths
        assert Path(topology["joint_encoder"]) in paths
        assert Path(topology["joint_encoder_checkpoint"]) in paths
        assert Path(topology["joint_embedding_bundle"]) in paths
        assert Path(topology["terminal"]) in paths
        assert Path(topology["completion"]) in paths


def test_capability_is_only_created_by_exact_ordered_admission_builder() -> None:
    calls: list[str] = []

    def builder(*, view: str) -> dict:
        calls.append(view)
        return _admission(view)

    plan = live.build_capability_bound_paired_execution_plan(admission_builder=builder)
    assert calls == ["sua", "pseudo_mua"]
    assert plan["status"] == "CAPABILITY_BOUND_PLAN_VALID__REAL_PRODUCERS_STILL_UNIMPLEMENTED"
    assert set(plan["capability_bindings"]) == {"sua", "pseudo_mua"}
    assert plan["capability_bindings"]["sua"]["root_authorization_body_sha256"] == _sha("root")
    assert "capability" not in inspect.signature(live.build_capability_bound_paired_execution_plan).parameters


def test_tampered_cross_view_authority_or_order_fails_closed() -> None:
    def root_drift(*, view: str) -> dict:
        return _admission(view, root="different-root" if view == "pseudo_mua" else "root")

    with pytest.raises(live.TrackBV2SubjectMStagePLiveExecutorError, match="same root authorization"):
        live.build_capability_bound_paired_execution_plan(admission_builder=root_drift)

    plan = live.build_unadmitted_paired_execution_plan()
    plan["pair_order"] = ["pseudo_mua", "sua"]
    with pytest.raises(live.TrackBV2SubjectMStagePLiveExecutorError, match="SUA then pMUA"):
        live.validate_paired_execution_plan(plan=plan)


def test_query_fit_or_artifact_path_tamper_fails_before_freshness() -> None:
    plan = copy.deepcopy(live.build_unadmitted_paired_execution_plan())
    plan["per_view_execution_contract"]["target_query_enters_encoder_or_readout_fit"] = True
    with pytest.raises(live.TrackBV2SubjectMStagePLiveExecutorError, match="target-query fit"):
        live.require_fresh_paired_execution_outputs(plan=plan)

    plan = copy.deepcopy(live.build_unadmitted_paired_execution_plan())
    plan["cell_topology_by_view"]["sua"]["joint_encoder_checkpoint"] = (
        plan["cell_topology_by_view"]["pseudo_mua"]["joint_encoder_checkpoint"]
    )
    with pytest.raises(live.TrackBV2SubjectMStagePLiveExecutorError, match="output alias/path"):
        live.require_fresh_paired_execution_outputs(plan=plan)


def test_body_or_sidecar_collision_is_rejected_before_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "RESULT_ROOT", tmp_path / "results")
    plan = live.build_unadmitted_paired_execution_plan()
    score = Path(plan["cell_topology_by_view"]["sua"]["scores"][
        "source_only_consumer_mechanism_alignment__linear_ridge"
    ])
    # A sidecar-only collision is enough.
    score.parent.mkdir(parents=True)
    Path(f"{score}.sha256").write_text("collision", encoding="utf-8")
    with pytest.raises(live.TrackBV2SubjectMStagePLiveExecutorError, match="fresh"):
        live.require_fresh_paired_execution_outputs(plan=plan)


def test_reserved_cell_root_rejects_stray_prior_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "RESULT_ROOT", tmp_path / "results")
    plan = live.build_unadmitted_paired_execution_plan()
    root = Path(plan["cell_topology_by_view"]["sua"]["cell_root"])
    root.mkdir(parents=True)
    (root / "old_terminal.json").write_text("not a result", encoding="utf-8")
    with pytest.raises(live.TrackBV2SubjectMStagePLiveExecutorError, match="not fresh/reserved"):
        live.require_fresh_paired_execution_outputs(plan=plan)


def test_existing_cell_root_allows_only_exact_official_preflight_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "RESULT_ROOT", tmp_path / "results")
    plan = live.build_unadmitted_paired_execution_plan()
    for view in live.PAIRED_VIEW_ORDER:
        topology = plan["cell_topology_by_view"][view]
        official = Path(topology["official_preflight"])
        official.parent.mkdir(parents=True, exist_ok=True)
        official.write_text("official body placeholder", encoding="utf-8")
        Path(f"{official}.sha256").write_text("official sidecar placeholder", encoding="utf-8")
    # Freshness is deliberately lexical.  The subsequent exact live admission
    # is the only component allowed to validate the official immutable pair.
    live.require_fresh_paired_execution_outputs(plan=plan)


def test_execute_successful_synthetic_admission_reaches_deliberate_no_target_tripwire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ml_imports_before = _ml_runtime_import_state()
    monkeypatch.setattr(runtime, "RESULT_ROOT", tmp_path / "results")
    calls: list[str] = []

    def builder(*, view: str) -> dict:
        calls.append(view)
        return _admission(view)

    with pytest.raises(live.TrackBV2SubjectMStagePLiveExecutorError, match="deliberately remains disabled"):
        live.refuse_paired_stagep_live_execution(admission_builder=builder)
    assert calls == ["sua", "pseudo_mua"]
    _assert_ml_runtime_import_state_unchanged(ml_imports_before)


def test_missing_authority_fails_before_any_target_provider_or_ml_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ml_imports_before = _ml_runtime_import_state()
    monkeypatch.setattr(runtime, "RESULT_ROOT", tmp_path / "results")
    reached = False

    def absent(*, view: str) -> dict:
        nonlocal reached
        reached = True
        raise live.TrackBV2SubjectMStagePLiveExecutorError("synthetic missing immutable authority")

    with pytest.raises(live.TrackBV2SubjectMStagePLiveExecutorError, match="missing immutable authority"):
        live.refuse_paired_stagep_live_execution(admission_builder=absent)
    assert reached is True
    _assert_ml_runtime_import_state_unchanged(ml_imports_before)


def test_fresh_process_module_import_does_not_load_ml_runtimes() -> None:
    probe = """
import sys
assert 'torch' not in sys.modules
assert 'cebra' not in sys.modules
import track_b_v2_subject_m_stagep_live_executor
assert 'torch' not in sys.modules
assert 'cebra' not in sys.modules
"""
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(SRC)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_cli_default_is_no_data_and_execute_requires_explicit_review_flag() -> None:
    default = subprocess.run([sys.executable, str(CLI)], cwd=REPO_ROOT, text=True,
                             capture_output=True, check=True)
    payload = json.loads(default.stdout)
    assert payload["status"] == live.LIVE_EXECUTOR_STATUS
    refused = subprocess.run([sys.executable, str(CLI), "--execute"], cwd=REPO_ROOT, text=True,
                             capture_output=True, check=False)
    assert refused.returncode != 0
    assert "requires --i-have-independent-root-review" in refused.stderr
