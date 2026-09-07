"""Focused CPU/no-data tests for CDM-D matched-score V6.

Every descriptor graph below is synthetic and lives in ``tmp_path``.  These
tests never resolve an evaluation asset, open an NWB/checkpoint tensor,
initialize CUDA, or touch a canonical authority/result root.
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
from typing import Mapping

import pytest

from src.causal_dual_memory_cell_d_score_v1 import physical as v1physical
from src.causal_dual_memory_cell_d_score_v1 import plan as v1plan
from src.causal_dual_memory_cell_d_score_v1 import score as v1score
from src.causal_dual_memory_cell_d_score_v5 import plan as v5plan
from src.causal_dual_memory_cell_d_score_v5 import score as v5score
from src.causal_dual_memory_cell_d_score_v6 import physical, plan, score


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _identity() -> plan.ScoreIdentity:
    rows = {path: _sha(f"v6:{path}") for path in plan.IMPLEMENTATION_PATHS}
    rows[plan.WORKORDER_RELATIVE] = plan.WORKORDER_SHA256
    return plan.ScoreIdentity(plan.ImplementationClosure(rows).payload())


def _source_gate_binding() -> v1score.SourceGateBinding:
    roster = tuple(f"sub-C_ses-CO-v6-synthetic-{index:02d}" for index in range(27))
    bodies = dict(v5plan.SOURCE_GATE_EXPECTED_SHAS)
    bodies.update({
        f"budget_m{budget}__{session}.json": _sha(f"v6:{budget}:{session}")
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
            v1plan.WITHIN, f"sub-C_ses-CO-v6-within-{index:02d}", f"within-{index}",
            f"sub-C/sub-C_ses-CO-v6-within-{index:02d}_behavior+ecephys.nwb", 1000 + index,
            _sha(f"within:{index}"),
        )
        for index in range(6)
    )
    external = tuple(
        v1score.EvaluationAsset(
            v1plan.EXTERNAL, f"sub-M_ses-CO-v6-external-{index:02d}", f"external-{index}",
            f"sub-M/sub-M_ses-CO-v6-external-{index:02d}_behavior+ecephys.nwb", 2000 + index,
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
    rows = tuple(_record(asset) for asset in (*fixed.within, *fixed.external))
    return v1score.InputAuthority(rows, score._digest(fixed.payload())), fixed


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
    committed = sum(bool(row["carrier_transition_committed"]) for row in transitions)
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
        activity_fifo_changed_count=sum(bool(row["activity_fifo_changed"]) for row in transitions),
        carrier_transition_committed_count=committed, activity_rejection_counts={},
        carrier_rejection_counts={"movement_too_short": rejected} if rejected else {},
    )


def _cells(authority: v1score.InputAuthority, *, identity: plan.ScoreIdentity, budget: int,
           input_sha: str, delta: float) -> tuple[v5score.CellEvidence, ...]:
    input_payload = authority.payload(identity=identity)
    output: list[v5score.CellEvidence] = []
    for surface in plan.SURFACES:
        rows = tuple(item for item in authority.records if item.surface == surface)
        sealed = tuple(_session(item, budget=budget, system=plan.SYSTEM_SEALED, r2=0.20) for item in rows)
        cdmd = tuple(_session(item, budget=budget, system=plan.SYSTEM_CDMD, r2=0.20 + delta) for item in rows)
        output.extend((
            v5score.CellEvidence(surface, budget, plan.SYSTEM_SEALED, input_sha,
                                 v1plan.SEALED_CELL_D_SWA_SHA256, sealed, _resources()),
            v5score.CellEvidence(surface, budget, plan.SYSTEM_CDMD, input_sha,
                                 v1plan.SEALED_CELL_D_SWA_SHA256, cdmd, _resources()),
        ))
    assert all(item.payload(input_payload=input_payload)["budget"] == budget for item in output)
    return tuple(output)


class _Artifact:
    topology = score.SCORE_TOPOLOGY

    def __init__(self, *, event_log: list[str], fail_prepare: bool = False) -> None:
        self.items: dict[str, bytes] = {}
        self.event_log = event_log
        self.fail_prepare = fail_prepare

    def publish_json(self, name: str, payload: Mapping[str, object]) -> str:
        if name in self.items:
            raise RuntimeError("synthetic collision")
        body = score._json(payload)
        self.items[name] = body
        self.event_log.append(f"publish:{name}")
        return hashlib.sha256(body).hexdigest()

    def publish_group(self, bodies: Mapping[str, bytes], *, post_publish=None):
        if any(name in self.items for name in bodies):
            raise RuntimeError("synthetic group collision")
        digests = {name: hashlib.sha256(body).hexdigest() for name, body in bodies.items()}
        if post_publish is not None:
            post_publish(bodies, digests)
        self.items.update(bodies)
        self.event_log.append("publish:score-terminal")
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
    def __init__(self, authority: v1score.InputAuthority, event_log: list[str], *, fail_prepare: bool = False) -> None:
        self.authority, self.event_log, self.calls, self.fail_prepare = authority, event_log, [], fail_prepare

    def _event(self, name: str) -> None:
        self.calls.append(name)
        self.event_log.append(name)

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


def _synthetic_v5_history(tmp_path: Path) -> plan.HistoricalV5ReservationContract:
    authority_relative, score_relative = "synthetic_v5_authority", "synthetic_v5_score"
    authority = tmp_path / authority_relative
    score_root = tmp_path / score_relative
    authority.mkdir()
    score_root.mkdir()
    os.chmod(score_root, 0o755)
    closure_sha = _sha("v5-history-closure")
    identity = {"closure": {"closure_sha256": closure_sha}}
    preflight = {
        "schema": "causal_dual_memory_cell_d_score_target_free_preflight_v5", "status": "PREFLIGHT_ACCEPTED",
        "identity": identity, "authority_root_relative": authority_relative, "score_root_relative": score_relative,
    }
    pre_sha = _write_pair(authority, "official_preflight.json", preflight)
    authorization = {
        "schema": "causal_dual_memory_cell_d_score_root_authorization_v5", "status": "ROOT_AUTHORIZED",
        "official_preflight_sha256": pre_sha, "identity_sha256": score._digest(identity),
        "authority_root_relative": authority_relative, "score_root_relative": score_relative,
    }
    auth_sha = _write_pair(authority, "root_authorization.json", authorization)
    info = score_root.stat()
    return plan.HistoricalV5ReservationContract(
        authority_root_relative=authority_relative, preflight_sha256=pre_sha, authorization_sha256=auth_sha,
        identity_sha256=score._digest(identity), implementation_closure_sha256=closure_sha,
        score_root_relative=score_relative, score_directory_device=int(info.st_dev), score_directory_inode=int(info.st_ino),
        score_directory_mode=stat.S_IMODE(info.st_mode), score_directory_empty=True, attempt_published=False,
        target_or_checkpoint_opened=False, cuda_initialized=False, launch_log_sha256=_sha("v5-history-log"),
    )


def _real_reserved_artifact(tmp_path: Path):
    # Load the real transactional helper once from the workspace.  Its held
    # concrete ArtifactRoot is then used against an isolated temporary root.
    module = v1score._equal_session_module(Path.cwd())
    parent = tmp_path / "tfpd_exploration" / "results"
    parent.mkdir(parents=True)
    artifact = module.reserve_artifact_root(parent, "causal_dual_memory_cell_d_matched_score_v6", topology=score.SCORE_TOPOLOGY)
    return module, artifact


def test_v6_static_plan_cli_closure_and_v1_v5_defaults_are_preserved() -> None:
    dry = plan.dry_plan()
    assert dry["workorder_sha256"] == plan.WORKORDER_SHA256
    assert plan.score_matrix() == v5plan.score_matrix()
    assert dry["score_spec"]["matrix_cell_count"] == 12
    assert dry["v5_reserved_root_history"] == plan.V5_RESERVED_ROOT_HISTORY.payload()
    closure = plan.implementation_closure(Path.cwd()).payload()
    assert plan.validate_implementation_closure(closure) == closure
    assert len(closure["paths"]) == len(plan.IMPLEMENTATION_PATHS)
    assert "tfpd_exploration/src/causal_dual_memory_cell_d_score_v6/physical.py" in plan.IMPLEMENTATION_PATHS
    assert v5plan.SCORE_ROOT_RELATIVE.endswith("matched_score_v5")
    assert v1plan.V1_ROUTE_PROFILE.route == "causal_dual_memory_cell_d_score_v1"
    assert v5score.V5_LIFECYCLE_HOOKS.validate_reserved_score_artifact is None
    assert score.V6_LIFECYCLE_HOOKS.validate_reserved_score_artifact is score._validate_reserved_score_artifact
    source = inspect.getsource(v1score.run_profiled_score_lifecycle)
    assert "validate_reserved_score_artifact is None" in source
    script = Path("tfpd_exploration/scripts/run_causal_dual_memory_cell_d_matched_score_v6.py").absolute()
    code = (
        "import importlib.util,sys;"
        f"s=importlib.util.spec_from_file_location('candidate',{str(script)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.main([]);"
        "print('TORCH='+str('torch' in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=Path.cwd(), text=True, capture_output=True, check=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
             "PYTHONPATH": str(Path.cwd() / "tfpd_exploration")},
    )
    assert '"no_torch_import":true' in completed.stdout and "TORCH=False" in completed.stdout
    denied = subprocess.run([sys.executable, str(script), "--execute"], cwd=Path.cwd(), text=True,
                            capture_output=True, env={**os.environ, "PYTHONNOUSERSITE": "1"})
    assert denied.returncode != 0 and "both --execute" in denied.stderr
    denied_both = subprocess.run(
        [sys.executable, str(script), "--execute", "--root-reviewed"], cwd=Path.cwd(), text=True,
        capture_output=True, env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    assert denied_both.returncode != 0 and "opaque in-process root capability" in denied_both.stderr


def test_v6_synthetic_v5_history_holds_pairs_and_exact_empty_root(tmp_path: Path) -> None:
    contract = _synthetic_v5_history(tmp_path)
    assert score._read_v5_history(tmp_path, contract=contract, require_literal_contract=False) == contract.payload()
    score_root = tmp_path / contract.score_root_relative
    (score_root / "forged").write_text("x")
    with pytest.raises(score.V6ScoreError, match="empty topology"):
        score._read_v5_history(tmp_path, contract=contract, require_literal_contract=False)
    (score_root / "forged").unlink()
    authority = tmp_path / contract.authority_root_relative
    os.chmod(authority / "official_preflight.json", 0o644)
    with pytest.raises(score.V6ScoreError, match="mode"):
        score._read_v5_history(tmp_path, contract=contract, require_literal_contract=False)
    with pytest.raises(score.V6ScoreError, match="contract identity"):
        score._read_v5_history(tmp_path, contract=contract, require_literal_contract=True)
    # Even a self-consistent body/sidecar replacement is not the fixed V5
    # authority pair: the held reader compares the body digest to the literal
    # historical contract before accepting any semantic fields.
    os.chmod(authority / "official_preflight.json", 0o644)
    os.chmod(authority / "official_preflight.json.sha256", 0o644)
    replacement = {
        "schema": "causal_dual_memory_cell_d_score_target_free_preflight_v5", "status": "PREFLIGHT_ACCEPTED",
        "identity": {"closure": {"closure_sha256": _sha("other-closure")}},
        "authority_root_relative": contract.authority_root_relative,
        "score_root_relative": contract.score_root_relative,
    }
    _write_pair(authority, "official_preflight.json", replacement)
    with pytest.raises(score.V6ScoreError, match="body SHA"):
        score._read_v5_history(tmp_path, contract=contract, require_literal_contract=False)


def test_v6_reserved_artifact_requires_actual_held_identity_empty_topology_and_no_replacement(tmp_path: Path) -> None:
    module, artifact = _real_reserved_artifact(tmp_path)
    score._validate_reserved_score_artifact(tmp_path, artifact, _identity())
    class PretendArtifact:
        directory = artifact.directory
        parent = artifact.parent
        topology = artifact.topology
        identity = artifact.identity
        parent_identity = artifact.parent_identity

    with pytest.raises(score.V6ScoreError, match="concrete type"):
        score._validate_reserved_score_artifact(tmp_path, PretendArtifact(), _identity())  # type: ignore[arg-type]
    # A valid own attempt is still forbidden at this boundary: only the
    # lifecycle may publish it after the empty-root validation succeeds.
    artifact.publish_json("attempt.json", {"synthetic": True})
    with pytest.raises(score.V6ScoreError, match="empty"):
        score._validate_reserved_score_artifact(tmp_path, artifact, _identity())

    _, fresh = _real_reserved_artifact(tmp_path / "extra")
    (fresh.directory / "extra.txt").write_text("x")
    with pytest.raises(score.V6ScoreError, match="empty"):
        score._validate_reserved_score_artifact(tmp_path / "extra", fresh, _identity())
    with pytest.raises(v1score.ScoreError, match="path/topology"):
        v1score.validate_reserved_artifact_root(
            tmp_path / "extra", fresh, root_relative=plan.SCORE_ROOT_RELATIVE, topology=("attempt.json",),
        )

    _, replaced = _real_reserved_artifact(tmp_path / "replaced")
    old = replaced.directory.with_name("old-root")
    replaced.directory.rename(old)
    replaced.directory.mkdir()
    with pytest.raises(score.V6ScoreError, match="identity"):
        score._validate_reserved_score_artifact(tmp_path / "replaced", replaced, _identity())

    _, parent_replaced = _real_reserved_artifact(tmp_path / "parent-replaced")
    old_parent = parent_replaced.parent.with_name("old-results")
    parent_replaced.parent.rename(old_parent)
    parent_replaced.parent.mkdir(parents=True)
    parent_replaced.directory.mkdir()
    with pytest.raises(score.V6ScoreError, match="identity"):
        score._validate_reserved_score_artifact(tmp_path / "parent-replaced", parent_replaced, _identity())

    _, symlinked = _real_reserved_artifact(tmp_path / "symlinked")
    target = symlinked.directory.with_name("target")
    symlinked.directory.rename(target)
    os.symlink(target.name, symlinked.directory)
    with pytest.raises(score.V6ScoreError, match="identity"):
        score._validate_reserved_score_artifact(tmp_path / "symlinked", symlinked, _identity())
    assert type(artifact) is module.ArtifactRoot


def test_v6_lifecycle_attempt_precedes_prepare_and_reaches_atomic_terminal_or_honest_failure(tmp_path: Path) -> None:
    identity = _identity()
    binding = _source_gate_binding()
    authority, fixed = _input_authority()
    preflight = score.build_target_free_preflight(
        root=tmp_path, identity=identity, source_gate=binding, fixed_authority=fixed,
    )
    pre_sha = score._digest(preflight)
    authorization = score.build_root_authorization(official_preflight_sha256=pre_sha, preflight=preflight)
    auth_sha = score._digest(authorization)
    root_capability = v1score._issue_root_publication_capability()
    capability = v1score.issue_execution_capability(
        durable_preflight_sha256=pre_sha, durable_authorization_sha256=auth_sha,
        identity=identity, root_capability=root_capability,
    )
    # The production V6 hook is separately exercised against the exact real
    # ArtifactRoot above.  This synthetic lifecycle adapter records the sole
    # required publication ordering without creating a canonical root.
    hooks = replace(
        score.V6_LIFECYCLE_HOOKS,
        implementation_closure=lambda _root: identity.payload()["closure"],
        validate_source_gate=lambda _root: binding,
        validate_reserved_score_artifact=lambda _root, _artifact, _identity: None,
    )
    events: list[str] = []
    runtime = _Runtime(authority, events)
    artifact = _Artifact(event_log=events)
    result = v1score.run_profiled_score_lifecycle(
        tmp_path, identity=identity, capability=capability, backend=runtime, artifact=artifact,
        official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
        preflight=preflight, authorization=authorization, hooks=hooks,
    )
    assert result["verdict"] == "ADVANCE_SHORT_BUDGET"
    assert events[:4] == ["preflight", "publish:attempt.json", "prepare", "materialize"]
    assert events.index("publish:attempt.json") < events.index("prepare") < events.index("materialize")
    assert runtime.calls[:5] == ["preflight", "prepare", "materialize", "budget30", "budget10"]
    assert "budget4" in runtime.calls
    assert {"attempt.json", "input_authority.json", "score.json", "terminal.json"} <= set(artifact.items)
    assert "failure.json" not in artifact.items
    score_payload = json.loads(artifact.items["score.json"])
    assert score_payload["schema"] == "causal_dual_memory_cell_d_matched_score_v6"
    assert score_payload["v5_reserved_root_history"] == plan.V5_RESERVED_ROOT_HISTORY.payload()
    assert score_payload["budget_execution_order"] == [30, 10, 4]

    failed_events: list[str] = []
    failed = _Artifact(event_log=failed_events)
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
    assert failure["schema"] == "causal_dual_memory_cell_d_score_failure_v6"
    assert failure["stage"] == "prepare" and failure["checkpoint_opened"] is False


def test_v6_actual_held_artifact_reaches_terminal_only_after_attempt_then_prepare(tmp_path: Path) -> None:
    """Exercise the new hook with the real transactional ``ArtifactRoot``.

    The runtime's prepare guard is deliberately the only observer needed:
    it proves that the concrete held root has durably gained ``attempt.json``
    before any prepare-reachable model/data/CUDA work can begin.
    """
    identity = _identity()
    binding = _source_gate_binding()
    authority, fixed = _input_authority()
    preflight = score.build_target_free_preflight(
        root=tmp_path, identity=identity, source_gate=binding, fixed_authority=fixed,
    )
    pre_sha = score._digest(preflight)
    authorization = score.build_root_authorization(official_preflight_sha256=pre_sha, preflight=preflight)
    auth_sha = score._digest(authorization)
    capability = v1score.issue_execution_capability(
        durable_preflight_sha256=pre_sha, durable_authorization_sha256=auth_sha, identity=identity,
        root_capability=v1score._issue_root_publication_capability(),
    )
    _module, artifact = _real_reserved_artifact(tmp_path)
    events: list[str] = []

    class GuardRuntime(_Runtime):
        def prepare(self, *, root: Path, identity: plan.ScoreIdentity):
            assert artifact.has_name("attempt.json")
            events.append("prepare-observed-durable-attempt")
            return super().prepare(root=root, identity=identity)

    hooks = replace(
        score.V6_LIFECYCLE_HOOKS,
        implementation_closure=lambda _root: identity.payload()["closure"],
        validate_source_gate=lambda _root: binding,
    )
    result = v1score.run_profiled_score_lifecycle(
        tmp_path, identity=identity, capability=capability, backend=GuardRuntime(authority, events), artifact=artifact,
        official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
        preflight=preflight, authorization=authorization, hooks=hooks,
    )
    terminal = artifact.reload_json("terminal.json", expected_sha256=result["terminal_sha256"])
    assert terminal["attempt_sha256"] == result["attempt_sha256"]
    input_body = artifact.reload_pair("input_authority.json", expected_sha256=result["input_authority_sha256"])
    assert input_body == score._json(artifact.reload_json("input_authority.json"))
    assert events.index("prepare-observed-durable-attempt") < events.index("prepare")
    assert artifact.has_name("attempt.json") and artifact.has_name("score.json") and artifact.has_name("terminal.json")
    assert not artifact.has_name("failure.json")

    # A distinct reserved root verifies the same concrete hook preserves an
    # honest immutable failure when prepare fails after the durable attempt.
    failure_root = tmp_path / "post-attempt-failure"
    failure_preflight = score.build_target_free_preflight(
        root=failure_root, identity=identity, source_gate=binding, fixed_authority=fixed,
    )
    failure_pre_sha = score._digest(failure_preflight)
    failure_authorization = score.build_root_authorization(
        official_preflight_sha256=failure_pre_sha, preflight=failure_preflight,
    )
    failure_auth_sha = score._digest(failure_authorization)
    failure_capability = v1score.issue_execution_capability(
        durable_preflight_sha256=failure_pre_sha, durable_authorization_sha256=failure_auth_sha,
        identity=identity, root_capability=v1score._issue_root_publication_capability(),
    )
    _module, failure_artifact = _real_reserved_artifact(failure_root)
    with pytest.raises(RuntimeError, match="post-attempt"):
        v1score.run_profiled_score_lifecycle(
            failure_root, identity=identity, capability=failure_capability,
            backend=_Runtime(authority, [], fail_prepare=True), artifact=failure_artifact,
            official_preflight_sha256=failure_pre_sha, root_authorization_sha256=failure_auth_sha,
            preflight=failure_preflight, authorization=failure_authorization, hooks=hooks,
        )
    assert failure_artifact.has_name("attempt.json") and failure_artifact.has_name("failure.json")
    assert not failure_artifact.has_name("terminal.json") and not failure_artifact.has_name("score.json")


def test_v6_physical_adapter_is_one_typed_v5_delegation_without_global_mutation() -> None:
    backend = physical.build_reviewed_physical_backend(
        root=Path.cwd(), selected_device_profile=v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"],
    )
    assert isinstance(backend, v1physical.PhysicalCDMDMatchedScoreBackend)
    implementation = inspect.getsource(physical)
    assert "v5physical.build_reviewed_physical_backend" in implementation
    assert "run_authorized_score_lifecycle" in implementation
    assert "sys.modules" not in implementation and "monkeypatch" not in implementation
    with pytest.raises(v1score.ScoreError, match="opaque"):
        physical._assert_v6_typed_capability(object(), _identity())
    if "torch" in sys.modules:
        assert sys.modules["torch"].cuda.is_initialized() is False


def test_v6_plan_prospective_roots_fail_closed_without_reservation(tmp_path: Path) -> None:
    plan.assert_fresh_prospective_roots(tmp_path)
    candidate = tmp_path / plan.SCORE_ROOT_RELATIVE
    candidate.parent.mkdir(parents=True)
    candidate.mkdir()
    with pytest.raises(plan.V6PlanError, match="already exists"):
        plan.assert_fresh_prospective_root(tmp_path, plan.SCORE_ROOT_RELATIVE)
