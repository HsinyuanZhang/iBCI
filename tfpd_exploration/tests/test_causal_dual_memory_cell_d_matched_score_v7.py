"""Focused CPU/no-data regressions for the CDM-D V7 RuntimeFlags successor.

All descriptor graphs below are synthetic temporary directories.  These tests
never resolve an evaluation/source asset, open an NWB/checkpoint tensor,
initialize CUDA, inspect a canonical result root, or reserve a canonical V7
root.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
from typing import Mapping

import pytest

from src.causal_dual_memory_cell_d_score_v1 import physical as v1physical
from src.causal_dual_memory_cell_d_score_v1 import plan as v1plan
from src.causal_dual_memory_cell_d_score_v1 import score as v1score
from src.causal_dual_memory_cell_d_score_v5 import plan as v5plan
from src.causal_dual_memory_cell_d_score_v5 import physical as v5physical
from src.causal_dual_memory_cell_d_score_v5 import score as v5score
from src.causal_dual_memory_cell_d_score_v6 import plan as v6plan
from src.causal_dual_memory_cell_d_score_v6 import score as v6score
from src.causal_dual_memory_cell_d_score_v7 import physical, plan, score


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _identity() -> plan.ScoreIdentity:
    rows = {path: _sha(f"v7:{path}") for path in plan.IMPLEMENTATION_PATHS}
    rows[plan.WORKORDER_RELATIVE] = plan.WORKORDER_SHA256
    return plan.ScoreIdentity(plan.ImplementationClosure(rows).payload())


def _v6_identity() -> v6plan.ScoreIdentity:
    rows = {path: _sha(f"v6-predecessor:{path}") for path in v6plan.IMPLEMENTATION_PATHS}
    rows[v6plan.WORKORDER_RELATIVE] = v6plan.WORKORDER_SHA256
    return v6plan.ScoreIdentity(v6plan.ImplementationClosure(rows).payload())


def _source_gate_binding(*, prefix: str = "v7") -> v1score.SourceGateBinding:
    roster = tuple(f"sub-C_ses-CO-{prefix}-synthetic-{index:02d}" for index in range(27))
    bodies = dict(v5plan.SOURCE_GATE_EXPECTED_SHAS)
    bodies.update({
        f"budget_m{budget}__{session}.json": _sha(f"{prefix}:{budget}:{session}")
        for budget in plan.BUDGETS for session in roster
    })
    return v1score.SourceGateBinding(
        directory_device=1, directory_inode=2, body_sha256s=bodies,
        terminal_status=v5plan.SOURCE_GATE_STATUS,
        source_gate_closure_sha256=v5plan.SOURCE_GATE_CLOSURE_SHA256,
        strict_source_roster=roster, contract=v5plan.SOURCE_GATE_CONTRACT,
    )


def _fixed_authority() -> v1score.FixedEvaluationAuthority:
    within = tuple(
        v1score.EvaluationAsset(
            v1plan.WITHIN, f"sub-C_ses-CO-v7-within-{index:02d}", f"within-{index}",
            f"sub-C/sub-C_ses-CO-v7-within-{index:02d}_behavior+ecephys.nwb", 1000 + index,
            _sha(f"within:{index}"),
        )
        for index in range(6)
    )
    external = tuple(
        v1score.EvaluationAsset(
            v1plan.EXTERNAL, f"sub-M_ses-CO-v7-external-{index:02d}", f"external-{index}",
            f"sub-M/sub-M_ses-CO-v7-external-{index:02d}_behavior+ecephys.nwb", 2000 + index,
            _sha(f"external:{index}"),
        )
        for index in range(15)
    )
    return v1score.FixedEvaluationAuthority(
        within=within, external=external,
        fixed_authority_bindings={"strict_manifest": {"sha256": _sha("within-manifest")}},
        within_manifest_binding={"manifest_sha256": _sha("within-paired-view")},
    )


def _raw_axis(session: str) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_score_raw_t4_axis_v1",
        "budget": 30, "raw_t4_sha256": _sha(f"{session}:raw"),
        "channel_order_sha256": _sha(f"{session}:channels"),
        "valid_mask_sha256": _sha(f"{session}:valid"), "source_unit_count": 8,
        "feature_group": "t4", "signal_view": "sua", "channel_ids_are_exact_int64_arange": True,
        "validity_rule": "raw_t4_modulation_m_gt_modulation_eps", "modulation_eps": 1.0e-6,
        "raw_before_normalization": True,
    }


def _record(asset: v1score.EvaluationAsset) -> v1score.InputRecord:
    ordered = tuple(f"{asset.session}:trial:{index:02d}" for index in range(32))
    support = {"4": ordered[:4], "10": ordered[:10], "30": ordered[:30]}
    query = {key: ordered[30:] for key in support}
    return v1score.InputRecord(
        surface=asset.surface, session=asset.session, asset_id=asset.asset_id,
        frozen_path=asset.frozen_path, asset_bytes=asset.bytes, asset_sha256=asset.sha256,
        chronological_trial_ids=ordered, support_trial_ids_by_budget=support, query_trial_ids_by_budget=query,
        neural_sha256=_sha(f"{asset.session}:neural"), calibration_sha256=_sha(f"{asset.session}:calibration"),
        target_last_bin_sha256_by_budget={key: _sha(f"{asset.session}:target:{key}") for key in support},
        valid_last_bin_mask_sha256_by_budget={key: _sha(f"{asset.session}:mask:{key}") for key in support},
        valid_last_bin_count_by_budget={key: 3 for key in support},
        raw_m30_t4_axis_proof=_raw_axis(asset.session), theta_recovery_sha256=_sha(f"{asset.session}:theta"),
    )


def _input_authority() -> tuple[v1score.InputAuthority, v1score.FixedEvaluationAuthority]:
    fixed = _fixed_authority()
    return v1score.InputAuthority(tuple(_record(item) for item in (*fixed.within, *fixed.external)), score._digest(fixed.payload())), fixed


def _resources() -> dict[str, object]:
    return {
        "runtime_environment": {
            **v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"], "visible_devices": 1, "attested": True,
            "torch_cuda_matmul_allow_tf32": False, "torch_cudnn_allow_tf32": False,
        },
        "current_cuda_allocated_bytes": 4, "current_cuda_reserved_bytes": 8,
        "peak_cuda_allocated_bytes": 12, "peak_cuda_reserved_bytes": 16, "rss_bytes": 4096,
        "wall_seconds": 1.0, "full_and_group_forward_chunks": 8, "completed_query_trials": 2,
        "windows_or_trials_per_s": 1.0,
    }


def _transition_rows(*, budget: int, query_ids: tuple[str, ...]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    state_before = _sha(f"M{budget}:state:0")
    activity_before = _sha(f"M{budget}:activity:0")
    carrier_before = _sha(f"M{budget}:carrier:0")
    for index, trial_id in enumerate(query_ids):
        committed = budget != 30 and index == len(query_ids) - 1
        state_after = _sha(f"M{budget}:state:{index + 1}")
        activity_after = activity_before if budget == 30 else _sha(f"M{budget}:activity:{index + 1}")
        carrier_after = _sha(f"M{budget}:carrier:{index + 1}") if committed else carrier_before
        rows.append({
            "trial_id": trial_id, "activity_transition_committed": True,
            "activity_fifo_changed": budget != 30, "carrier_transition_committed": committed,
            "activity_rejection_reason_or_null": None,
            "carrier_rejection_reason_or_null": None if committed else "movement_too_short",
            "state_before_sha256": state_before, "state_after_sha256": state_after,
            "activity_before_sha256": activity_before, "activity_after_sha256": activity_after,
            "carrier_before_sha256": carrier_before, "carrier_after_sha256": carrier_after,
        })
        state_before, activity_before, carrier_before = state_after, activity_after, carrier_after
    return tuple(rows)


def _session(record: v1score.InputRecord, *, budget: int, system: str, r2: float) -> v5score.IndependentSessionScore:
    payload = record.payload()
    key = str(budget)
    query_ids = tuple(payload["query_trial_ids_by_budget"][key])
    transitions = _transition_rows(budget=budget, query_ids=query_ids) if system == plan.SYSTEM_CDMD else ()
    committed = sum(bool(item["carrier_transition_committed"]) for item in transitions)
    rejected = len(transitions) - committed
    base = v1score.SessionScore(
        session=record.session, n_windows=3, r2=r2,
        prediction_sha256=_sha(f"{system}:{record.session}:M{budget}:prediction"),
        input_record_sha256=score._digest(payload), model_state_before_sha256=_sha("sealed"),
        model_state_after_sha256=_sha("sealed"), initial_carrier_sha256=_sha(f"{record.session}:carrier:{budget}"),
        group_assignment_sha256=_sha(f"{record.session}:groups:{budget}"),
        group_valid_mask_sha256=str(payload["raw_m30_t4_axis_proof"]["valid_mask_sha256"]),
        initial_activity_sha256=_sha(f"{record.session}:activity:{budget}"),
        support_trial_ids_sha256=str(payload["support_trial_ids_sha256_by_budget"][key]),
        raw_m30_t4_axis_proof_sha256=score._digest(payload["raw_m30_t4_axis_proof"]),
        sealed_normalizer_sha256=v1plan.SEALED_OLS_NORMALIZER_SHA256,
        sealed_model_load_proof_sha256=_sha("strict-load"),
        target_last_bin_sha256=str(payload["target_last_bin_sha256_by_budget"][key]),
        valid_mask_sha256=str(payload["valid_last_bin_mask_sha256_by_budget"][key]), valid_last_bin_count=3,
        activity_fifo_capacity=plan.FIFO_CAPACITY[budget],
        accepted_updates=committed * plan.GROUP_COUNT if system == plan.SYSTEM_CDMD else 0,
        rejected_updates={"movement_too_short": rejected * plan.GROUP_COUNT} if rejected else {},
        group_forward_count=8 if system == plan.SYSTEM_CDMD else 0, full_system_forward_count=4,
        dropout_calls=0, target_label_state_uses=0,
    )
    return v5score.IndependentSessionScore(
        base=base, transition_records=transitions,
        activity_transition_committed_count=len(transitions),
        activity_fifo_changed_count=sum(bool(item["activity_fifo_changed"]) for item in transitions),
        carrier_transition_committed_count=committed, activity_rejection_counts={},
        carrier_rejection_counts={"movement_too_short": rejected} if rejected else {},
    )


def _cells(authority: v1score.InputAuthority, *, identity: plan.ScoreIdentity, budget: int,
           input_sha: str, delta: float) -> tuple[v5score.CellEvidence, ...]:
    input_payload = authority.payload(identity=identity)
    cells: list[v5score.CellEvidence] = []
    for surface in plan.SURFACES:
        rows = tuple(item for item in authority.records if item.surface == surface)
        sealed = tuple(_session(item, budget=budget, system=plan.SYSTEM_SEALED, r2=0.20) for item in rows)
        cdmd = tuple(_session(item, budget=budget, system=plan.SYSTEM_CDMD, r2=0.20 + delta) for item in rows)
        cells.extend((
            v5score.CellEvidence(surface, budget, plan.SYSTEM_SEALED, input_sha,
                                 v1plan.SEALED_CELL_D_SWA_SHA256, sealed, _resources()),
            v5score.CellEvidence(surface, budget, plan.SYSTEM_CDMD, input_sha,
                                 v1plan.SEALED_CELL_D_SWA_SHA256, cdmd, _resources()),
        ))
    assert len(cells) == 4
    assert all(item.payload(input_payload=input_payload)["budget"] == budget for item in cells)
    return tuple(cells)


class _Artifact:
    topology = score.SCORE_TOPOLOGY

    def __init__(self, events: list[str]) -> None:
        self.items: dict[str, bytes] = {}
        self.events = events

    def publish_json(self, name: str, payload: Mapping[str, object]) -> str:
        if name in self.items:
            raise RuntimeError("synthetic collision")
        body = score._json(payload)
        self.items[name] = body
        self.events.append(f"publish:{name}")
        return hashlib.sha256(body).hexdigest()

    def publish_group(self, bodies: Mapping[str, bytes], *, post_publish=None):
        if any(name in self.items for name in bodies):
            raise RuntimeError("synthetic collision")
        digests = {name: hashlib.sha256(body).hexdigest() for name, body in bodies.items()}
        if post_publish is not None:
            post_publish(bodies, digests)
        self.items.update(bodies)
        self.events.append("publish:score-terminal")
        return digests

    def reload_json(self, name: str, expected_sha256: str | None = None):
        body = self.items[name]
        if expected_sha256 is not None and hashlib.sha256(body).hexdigest() != expected_sha256:
            raise RuntimeError("synthetic digest drift")
        return json.loads(body)

    def reload_pair(self, name: str, expected_sha256: str | None = None) -> bytes:
        body = self.items[name]
        if expected_sha256 is not None and hashlib.sha256(body).hexdigest() != expected_sha256:
            raise RuntimeError("synthetic digest drift")
        return body

    def has_name(self, name: str) -> bool:
        return name in self.items


class _Runtime:
    def __init__(self, authority: v1score.InputAuthority, events: list[str], *, fail_prepare: bool = False) -> None:
        self.authority, self.events, self.calls, self.fail_prepare = authority, events, [], fail_prepare

    def _event(self, value: str) -> None:
        self.calls.append(value)
        self.events.append(value)

    def preflight(self, *, root: Path, identity: plan.ScoreIdentity):
        self._event("preflight")
        return {"target_paths_resolved": False, "target_opened": False, "checkpoint_opened": False, "cuda_initialized": False}

    def prepare(self, *, root: Path, identity: plan.ScoreIdentity):
        self._event("prepare")
        if self.fail_prepare:
            raise RuntimeError("synthetic post-attempt prepare failure")
        return self

    def materialize_inputs(self, runtime, *, identity: plan.ScoreIdentity, evaluation_authority: v1score.FixedEvaluationAuthority):
        self._event("materialize")
        return self.authority

    def score_budget(self, runtime, *, budget: int, input_authority_sha256: str, identity: plan.ScoreIdentity):
        self._event(f"budget{budget}")
        return _cells(self.authority, identity=identity, budget=budget, input_sha=input_authority_sha256,
                      delta=(-0.02 if budget == 30 else 0.06))

    def revalidate(self, runtime, *, root: Path, identity: plan.ScoreIdentity) -> None:
        self._event("revalidate")

    def failure_progress(self, runtime):
        return {
            "within_assets_opened": False, "external_assets_opened": False, "checkpoint_opened": False,
            "cuda_initialized": False, "full_system_forward_count": 0, "group_forward_count": 0,
        }

    def close(self, runtime) -> None:
        self._event("close")


def _write_pair(directory: Path, name: str, payload: Mapping[str, object]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    (directory / name).write_bytes(body)
    (directory / f"{name}.sha256").write_bytes(f"{digest}  {Path(name).name}\n".encode("ascii"))
    os.chmod(directory / name, 0o444)
    os.chmod(directory / f"{name}.sha256", 0o444)
    return digest


def _synthetic_v6_predecessor(tmp_path: Path) -> plan.V6RuntimeFlagsFailureContract:
    identity = _v6_identity()
    binding = _source_gate_binding(prefix="v6-history")
    fixed = _fixed_authority()
    authority_relative, score_relative = v6plan.AUTHORITY_ROOT_RELATIVE, v6plan.SCORE_ROOT_RELATIVE
    authority_root, failed_root = tmp_path / authority_relative, tmp_path / score_relative
    authority_root.mkdir(parents=True)
    failed_root.mkdir(parents=True)
    os.chmod(failed_root, 0o755)
    preflight = v6score.build_target_free_preflight(
        root=tmp_path, identity=identity, source_gate=binding, fixed_authority=fixed,
    )
    pre_sha = _write_pair(authority_root, "official_preflight.json", preflight)
    authorization = v6score.build_root_authorization(official_preflight_sha256=pre_sha, preflight=preflight)
    auth_sha = _write_pair(authority_root, "root_authorization.json", authorization)
    attempt = v6score._attempt_payload(identity, pre_sha, auth_sha)
    attempt_sha = _write_pair(failed_root, "attempt.json", attempt)
    failure = v6score._failure_payload(
        identity, attempt_sha, None, "prepare", AttributeError("synthetic V5 RuntimeFlags absence"),
        {
            "within_assets_opened": False, "external_assets_opened": False, "checkpoint_opened": True,
            "cuda_initialized": False, "full_system_forward_count": 0, "group_forward_count": 0,
        },
    )
    failure_sha = _write_pair(failed_root, "failure.json", failure)
    info = failed_root.stat()
    return plan.V6RuntimeFlagsFailureContract(
        authority_root_relative=authority_relative, preflight_sha256=pre_sha, authorization_sha256=auth_sha,
        identity_sha256=score._digest(identity.payload()),
        implementation_closure_sha256=identity.payload()["closure"]["closure_sha256"],
        score_root_relative=score_relative, score_directory_device=int(info.st_dev), score_directory_inode=int(info.st_ino),
        score_directory_mode=stat.S_IMODE(info.st_mode), attempt_sha256=attempt_sha, failure_sha256=failure_sha,
        launch_log_sha256=_sha("synthetic-v6-log"), failure_stage="prepare", failure_class="AttributeError",
        failure_error_sha256=str(failure["error_sha256"]),
    )


def _real_reserved_artifact(tmp_path: Path):
    module = v1score._equal_session_module(Path.cwd())
    parent = tmp_path / "tfpd_exploration" / "results"
    parent.mkdir(parents=True)
    return module.reserve_artifact_root(parent, "causal_dual_memory_cell_d_matched_score_v7", topology=score.SCORE_TOPOLOGY)


def test_v7_static_plan_cli_closure_and_public_flags_are_inert() -> None:
    dry = plan.dry_plan()
    assert dry["workorder_sha256"] == plan.WORKORDER_SHA256
    assert plan.score_matrix() == v6plan.score_matrix()
    assert dry["score_spec"]["matrix_cell_count"] == 12
    assert dry["v6_runtime_flags_failure"]["exact_json_bodies"] == 2
    assert dry["v6_runtime_flags_failure"]["exact_leaves"] == 4
    closure = plan.implementation_closure(Path.cwd()).payload()
    assert plan.validate_implementation_closure(closure) == closure
    assert "tfpd_exploration/src/causal_dual_memory_cell_d_score_v1/physical.py" in plan.IMPLEMENTATION_PATHS
    source_leaf = "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute.py"
    rows = {path: _sha(f"v7-closure-drift:{path}") for path in plan.IMPLEMENTATION_PATHS}
    rows[plan.WORKORDER_RELATIVE] = plan.WORKORDER_SHA256
    rows[source_leaf] = _sha("forged-runtime-flags-source")
    forged_identity = plan.ScoreIdentity(plan.ImplementationClosure(rows).payload())
    assert plan.implementation_closure(Path.cwd()).payload() != forged_identity.payload()["closure"]
    assert "glob" not in inspect.getsource(plan)
    script = Path("tfpd_exploration/scripts/run_causal_dual_memory_cell_d_matched_score_v7.py").absolute()
    code = (
        "import importlib.util,sys;"
        f"s=importlib.util.spec_from_file_location('candidate',{str(script)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.main([]);"
        "print('TORCH='+str('torch' in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=Path.cwd(), text=True, capture_output=True, check=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
             "PYTHONPATH": ""},
    )
    assert '"no_torch_import":true' in completed.stdout and "TORCH=False" in completed.stdout
    denied = subprocess.run(
        [sys.executable, str(script), "--execute", "--root-reviewed"], cwd=Path.cwd(), text=True,
        capture_output=True, env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    assert denied.returncode != 0 and "opaque in-process root capability" in denied.stderr


def test_v7_runtime_flags_factory_authenticates_v5_wrapper_v1_dependency_without_mutation(tmp_path: Path) -> None:
    from src.causal_dual_memory_cell_d_v1 import source_execute_v5 as wrapper
    from src.causal_dual_memory_cell_d_v1 import source_execute_physical_v5 as source_physical

    assert "RuntimeFlags" not in vars(wrapper)
    with pytest.raises(AttributeError):
        v1physical.ReviewedCDMScoreRuntime._build_runtime_flags(object(), wrapper)
    runtime = physical.V7ReviewedCDMScoreRuntime(
        root=Path.cwd(), selected_device_profile=v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"],
    )
    relative = runtime._V1_FLAGS_RELATIVE
    runtime._v5_runtime_sha256_by_path = {relative: plan._read_regular_no_follow(Path.cwd() / relative)}
    flags = runtime._build_runtime_flags(wrapper)
    assert type(flags) is wrapper.v1.RuntimeFlags and flags.stage == "score_prepare"
    # A faithful no-data seam reaches the inherited V5 executor interface;
    # construction alone does not resolve source data or load a sealed SWA.
    executor = runtime._build_source_executor(source_physical)
    assert type(executor) is source_physical.V5OneShotFinalizedRowExecutor
    backend = physical.build_reviewed_physical_backend(
        root=Path.cwd(), selected_device_profile=v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"],
    )
    assert isinstance(backend, v1physical.PhysicalCDMDMatchedScoreBackend)
    assert type(backend) is physical.PhysicalV7ScoreBackend
    assert type(backend._runtime_factory(Path.cwd(), v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"])) is physical.V7ReviewedCDMScoreRuntime
    assert inspect.getattr_static(wrapper, "v1") is wrapper.v1
    assert "torch" not in sys.modules or sys.modules["torch"].cuda.is_initialized() is False

    wrong_sha = physical.V7ReviewedCDMScoreRuntime(
        root=Path.cwd(), selected_device_profile=v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"],
    )
    wrong_sha._v5_runtime_sha256_by_path = {relative: _sha("wrong-v1-source-execute")}
    with pytest.raises(physical.PhysicalV7ScoreError, match="bytes"):
        wrong_sha._build_runtime_flags(wrapper)

    forged_type = type("RuntimeFlags", (), {"__module__": "forged.flags", "__init__": lambda self, stage: setattr(self, "stage", stage)})
    forged_dependency = SimpleNamespace(
        __name__=runtime._V1_FLAGS_MODULE_NAME,
        __file__=str(Path.cwd() / relative), RuntimeFlags=forged_type,
    )
    forged_wrapper = SimpleNamespace(__name__=wrapper.__name__, v1=forged_dependency)
    with pytest.raises(physical.PhysicalV7ScoreError, match="dependency object"):
        runtime._build_runtime_flags(forged_wrapper)
    wrong_path = physical.V7ReviewedCDMScoreRuntime(
        root=tmp_path / "wrong-root", selected_device_profile=v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"],
    )
    wrong_path._v5_runtime_sha256_by_path = {relative: plan._read_regular_no_follow(Path.cwd() / relative)}
    with pytest.raises(physical.PhysicalV7ScoreError, match="module/path"):
        wrong_path._build_runtime_flags(wrapper)
    source = inspect.getsource(physical)
    assert "sys.modules" not in source and "monkeypatch" not in source
    assert "hasattr" not in source and "except AttributeError" not in source
    assert "validate_v6_runtime_flags_predecessor" in inspect.getsource(physical.PhysicalV7ScoreBackend.revalidate)


def test_v7_faithful_v1_prepare_dispatches_flags_then_v5_executor_before_sealed_load() -> None:
    """Observe the real V1 prepare hook ordering without opening an artifact.

    Calling ``ReviewedCDMScoreRuntime.prepare`` itself would intentionally
    deserialize the sealed SWA before it reaches ``RuntimeFlags``.  This
    no-data observer instead exercises the exact two dynamically dispatched
    hook calls with real V5 wrapper/executor objects and proves, from the
    reviewed V1 method body, that their order is the one used in the physical
    prepare path before ``load_strict_sealed_swa``.
    """
    from src.causal_dual_memory_cell_d_v1 import source_execute_v5 as wrapper
    from src.causal_dual_memory_cell_d_v1 import source_execute_physical_v5 as source_physical

    class PrepareObserver(physical.V7ReviewedCDMScoreRuntime):
        def __init__(self) -> None:
            super().__init__(root=Path.cwd(), selected_device_profile=v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"])
            self.events: list[str] = []
            self._v5_runtime_sha256_by_path = {
                self._V1_FLAGS_RELATIVE: plan._read_regular_no_follow(Path.cwd() / self._V1_FLAGS_RELATIVE),
            }

        def _build_runtime_flags(self, source_execute):
            self.events.append("flags")
            return super()._build_runtime_flags(source_execute)

        def _build_source_executor(self, source_physical):
            self.events.append("executor")
            return super()._build_source_executor(source_physical)

    observer = PrepareObserver()
    flags = observer._build_runtime_flags(wrapper)
    executor = observer._build_source_executor(source_physical)
    assert observer.events == ["flags", "executor"]
    assert type(flags) is wrapper.v1.RuntimeFlags
    assert type(executor) is source_physical.V5OneShotFinalizedRowExecutor
    prepare_source = inspect.getsource(v1physical.ReviewedCDMScoreRuntime.prepare)
    flags_line = prepare_source.index("flags = self._build_runtime_flags(source_execute)")
    executor_line = prepare_source.index("executor = self._build_source_executor(source_physical)")
    sealed_load_line = prepare_source.index("executor.load_strict_sealed_swa")
    assert flags_line < executor_line < sealed_load_line
    assert "source_execute.RuntimeFlags(" not in prepare_source


def test_v7_generic_v6_failure_reader_holds_exact_pairs_and_rejects_replacement(tmp_path: Path) -> None:
    contract = _synthetic_v6_predecessor(tmp_path)
    assert score._read_v6_runtime_flags_failure(tmp_path, contract=contract, require_literal_contract=False) == contract.payload()
    failure_root = tmp_path / contract.score_root_relative
    os.chmod(failure_root / "failure.json", 0o644)
    with pytest.raises(score.V7ScoreError, match="mode"):
        score._read_v6_runtime_flags_failure(tmp_path, contract=contract, require_literal_contract=False)
    os.chmod(failure_root / "failure.json", 0o644)
    os.chmod(failure_root / "failure.json.sha256", 0o644)
    replacement = {"synthetic": "replaced"}
    _write_pair(failure_root, "failure.json", replacement)
    with pytest.raises(score.V7ScoreError, match="body SHA"):
        score._read_v6_runtime_flags_failure(tmp_path, contract=contract, require_literal_contract=False)
    with pytest.raises(score.V7ScoreError, match="contract identity"):
        score._read_v6_runtime_flags_failure(tmp_path, contract=contract, require_literal_contract=True)


@pytest.mark.parametrize("mutation, expected", [
    ("missing", "topology"),
    ("reordered_sidecar", "canonical sidecar"),
    ("symlink", "descriptor-open"),
    ("inode_replace", "identity"),
])
def test_v7_v6_predecessor_leaf_topology_tamper_fails_before_any_v7_output(
    tmp_path: Path, mutation: str, expected: str,
) -> None:
    contract = _synthetic_v6_predecessor(tmp_path)
    failed_root = tmp_path / contract.score_root_relative
    if mutation == "missing":
        os.chmod(failed_root / "attempt.json.sha256", 0o644)
        (failed_root / "attempt.json.sha256").unlink()
    elif mutation == "reordered_sidecar":
        sidecar = failed_root / "attempt.json.sha256"
        os.chmod(sidecar, 0o644)
        body = (failed_root / "attempt.json").read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        sidecar.write_bytes(f"{digest}  failure.json\n".encode("ascii"))
        os.chmod(sidecar, 0o444)
    elif mutation == "symlink":
        body = failed_root / "failure.json"
        os.chmod(body, 0o644)
        body.unlink()
        os.symlink("attempt.json", body)
    elif mutation == "inode_replace":
        old = failed_root.with_name("old-v6-failed")
        failed_root.rename(old)
        failed_root.mkdir()
    else:  # pragma: no cover - parametrization is literal above
        raise AssertionError(mutation)
    with pytest.raises(score.V7ScoreError, match=expected):
        score._read_v6_runtime_flags_failure(tmp_path, contract=contract, require_literal_contract=False)
    assert not (tmp_path / plan.AUTHORITY_ROOT_RELATIVE).exists()
    assert not (tmp_path / plan.SCORE_ROOT_RELATIVE).exists()


def test_v7_complete_twelve_cell_lifecycle_attempt_before_prepare_and_honest_failure(tmp_path: Path) -> None:
    identity = _identity()
    binding = _source_gate_binding()
    authority, fixed = _input_authority()
    preflight = score.build_target_free_preflight(root=tmp_path, identity=identity, source_gate=binding, fixed_authority=fixed)
    pre_sha = score._digest(preflight)
    authorization = score.build_root_authorization(official_preflight_sha256=pre_sha, preflight=preflight)
    auth_sha = score._digest(authorization)
    capability = v1score.issue_execution_capability(
        durable_preflight_sha256=pre_sha, durable_authorization_sha256=auth_sha, identity=identity,
        root_capability=v1score._issue_root_publication_capability(),
    )
    hooks = replace(
        score.V7_LIFECYCLE_HOOKS,
        implementation_closure=lambda _root: identity.payload()["closure"],
        validate_source_gate=lambda _root: binding,
        validate_reserved_score_artifact=lambda _root, _artifact, _identity: None,
    )
    events: list[str] = []
    artifact = _Artifact(events)
    result = v1score.run_profiled_score_lifecycle(
        tmp_path, identity=identity, capability=capability, backend=_Runtime(authority, events), artifact=artifact,
        official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
        preflight=preflight, authorization=authorization, hooks=hooks,
    )
    assert events[:4] == ["preflight", "publish:attempt.json", "prepare", "materialize"]
    assert result["verdict"] == "ADVANCE_SHORT_BUDGET"
    assert {"attempt.json", "input_authority.json", "score.json", "terminal.json"} <= set(artifact.items)
    assert "failure.json" not in artifact.items
    score_body = json.loads(artifact.items["score.json"])
    assert score_body["schema"] == "causal_dual_memory_cell_d_matched_score_v7"
    assert score_body["v6_runtime_flags_failure"] == plan.V6_RUNTIME_FLAGS_FAILURE.payload()
    assert score_body["budget_execution_order"] == [30, 10, 4]
    assert len(score_body["cell_execution_order"]) == 12

    failed_events: list[str] = []
    failed = _Artifact(failed_events)
    with pytest.raises(RuntimeError, match="post-attempt"):
        v1score.run_profiled_score_lifecycle(
            tmp_path, identity=identity, capability=capability,
            backend=_Runtime(authority, failed_events, fail_prepare=True), artifact=failed,
            official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
            preflight=preflight, authorization=authorization, hooks=hooks,
        )
    assert failed_events[:3] == ["preflight", "publish:attempt.json", "prepare"]
    assert "failure.json" in failed.items and "terminal.json" not in failed.items
    failure = json.loads(failed.items["failure.json"])
    assert failure["schema"] == "causal_dual_memory_cell_d_score_failure_v7"
    assert failure["stage"] == "prepare" and failure["checkpoint_opened"] is False


def test_v7_real_held_artifact_validation_rejects_replacement_before_attempt(tmp_path: Path) -> None:
    identity = _identity()
    artifact = _real_reserved_artifact(tmp_path)
    score._validate_reserved_score_artifact(tmp_path, artifact, identity)
    artifact.directory.rename(artifact.directory.with_name("old-root"))
    artifact.directory.mkdir()
    with pytest.raises(score.V7ScoreError, match="identity"):
        score._validate_reserved_score_artifact(tmp_path, artifact, identity)


def test_v7_v1_default_factory_remains_historical_and_v7_is_only_override() -> None:
    source = inspect.getsource(v1physical.ReviewedCDMScoreRuntime._build_runtime_flags)
    assert "source_execute.RuntimeFlags" in source
    assert v5physical.V5ReviewedCDMScoreRuntime._build_runtime_flags is v1physical.ReviewedCDMScoreRuntime._build_runtime_flags
    assert v1physical.ReviewedCDMScoreRuntime._build_runtime_flags is not physical.V7ReviewedCDMScoreRuntime._build_runtime_flags
    assert physical.V7ReviewedCDMScoreRuntime.__mro__[1].__name__ == "V5ReviewedCDMScoreRuntime"
    with pytest.raises(v1score.ScoreError, match="opaque"):
        physical._assert_v7_typed_capability(object(), _identity())


def test_v7_predecessor_validation_is_ordered_before_any_fresh_output_reservation() -> None:
    for function in (
        score.reserve_authority_artifact,
        score.issue_durable_execution_capability,
        score.reserve_score_artifact,
    ):
        source = inspect.getsource(function)
        assert source.index("validate_v6_runtime_flags_predecessor") < source.index("assert_fresh_prospective_root")
