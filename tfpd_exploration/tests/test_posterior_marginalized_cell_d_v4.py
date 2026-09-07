"""No-data/CPU tests for the additive PMC-D B128 matched-score V4 seam."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "tfpd_exploration"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.posterior_marginalized_cell_d_v4 import matched_score as score
from src.posterior_marginalized_cell_d_v4 import matched_score_physical as physical


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _BaseIdentity:
    within_roster: tuple[str, ...] = tuple(f"within-{index:02d}" for index in range(6))
    external_roster: tuple[str, ...] = tuple(f"external-{index:02d}" for index in range(15))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "synthetic-base-identity-v1", "cell": score.CELL,
            "within": list(self.within_roster), "external": list(self.external_roster),
        }


@dataclass(frozen=True)
class _Closure:
    graph: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        body = {
            "schema": "synthetic-v4-closure-v1", "failed_v3_graph": dict(self.graph),
            "evaluation_batch_size": score.EVAL_BATCH_SIZE,
        }
        return {**body, "closure_sha256": _sha(json.dumps(body, sort_keys=True, separators=(",", ":")))}


def _graph() -> dict[str, object]:
    return {
        "schema": "posterior_marginalized_cell_d_score_v3_failed_graph_v1",
        "root_relative": score.FAILED_V3_ROOT_RELATIVE,
        "body_sha256": dict(physical.FAILED_V3_BODY_SHA256),
        "identity_sha256": physical.FAILED_V3_IDENTITY_SHA256,
        "input_authority_sha256": physical.FAILED_V3_BODY_SHA256["input_authority.json"],
        "failure_stage": "score", "failure_status": "SCORE_FAILED", "terminal_present": False,
        "score_present": False, "input_authority_present": True, "no_target_updates": True,
    }


def _identity() -> score.V4ScoreIdentity:
    return score.V4ScoreIdentity(
        base_identity=_BaseIdentity(), closure=_Closure(_graph()),
        v3_input_authority_sha256=physical.FAILED_V3_BODY_SHA256["input_authority.json"],
    )


def _input_payload(identity: score.V4ScoreIdentity) -> dict[str, object]:
    records = []
    for surface, roster in ((score.WITHIN, identity.base_identity.within_roster),
                            (score.EXTERNAL, identity.base_identity.external_roster)):
        for order, session in enumerate(roster):
            records.append({
                "surface": surface, "session": session, "n_windows": order + 2,
                "input_token_sha256": _sha(f"input:{surface}:{session}"),
            })
    return {
        "schema": "posterior_marginalized_cell_d_matched_score_input_authority_v1",
        "identity_cell": score.CELL, "records": records,
        "shared_materialized_input_pass": True, "cache_read_or_write": False,
        "within_opened": True, "external_opened": True, "formal_opened": False, "target_opened": True,
        "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
        "normalizer_refit": False,
    }


def _evidence(identity: score.V4ScoreIdentity) -> tuple[score.V4CellEvidence, ...]:
    input_payload = _input_payload(identity)
    rows_by_surface = {
        surface: [row for row in input_payload["records"] if row["surface"] == surface]
        for surface in score.SURFACES
    }
    evidence: list[score.V4CellEvidence] = []
    for cell in score.score_matrix():
        rows = []
        for row in rows_by_surface[cell.surface]:
            # PMC gains only at M4 in this synthetic screen; all tests retain
            # a fully paired same-input row order.
            base = 0.20 + 0.01 * int(row["n_windows"])
            value = base + (0.03 if cell.system == score.SYSTEM_PMC and cell.budget == 4 else 0.0)
            rows.append(score.V4SessionScore(
                session=str(row["session"]), n_windows=int(row["n_windows"]), r2=value,
                prediction_sha256=_sha(f"pred:{cell.surface}:{cell.budget}:{cell.system}:{row['session']}"),
                input_record_sha256=_sha(json.dumps(row, sort_keys=True, separators=(",", ":"))),
            ))
        evidence.append(score.V4CellEvidence(
            cell=cell, sessions=tuple(rows),
            model_swa_sha256=_sha(f"swa:{cell.system}"),
            state_before_sha256=_sha(f"state:{cell.system}"), state_after_sha256=_sha(f"state:{cell.system}"),
        ))
    return tuple(evidence)


def _bridge(identity: score.V4ScoreIdentity, *, mismatch: bool = False) -> dict[str, object]:
    rows: dict[str, list[dict[str, object]]] = {}
    first: dict[str, object] | None = None
    for surface, roster in ((score.WITHIN, identity.base_identity.within_roster),
                            (score.EXTERNAL, identity.base_identity.external_roster)):
        local: list[dict[str, object]] = []
        for order, session in enumerate(roster):
            exact = not (mismatch and first is None)
            item = {
                "session": session, "historical_n_windows": order + 2, "live_n_windows": order + 2,
                "historical_r2": 0.1 + order / 100,
                "live_r2": 0.1 + order / 100 + (0.0001 if not exact else 0.0),
                "prediction_sha256": _sha(f"bridge:{surface}:{session}"), "exact_match": exact,
                "mismatch_field": None if exact else "r2",
            }
            local.append(item)
            if not exact and first is None:
                first = {"surface": surface, **item}
        rows[surface] = local
    return {
        "schema": "posterior_marginalized_cell_d_historical_sealed_bridge_v4",
        "evaluation_batch_size": 128, "metric": dict(score.METRIC_CONTRACT),
        "sealed_swa_sha256": _sha("sealed"), "sealed_baseline_receipt_sha256": _sha("baseline"),
        "input_authority_sha256": _sha(json.dumps(_input_payload(identity), sort_keys=True, separators=(",", ":"))),
        "rows": rows, "all_exact": first is None, "first_mismatch": first,
        "interpretation": "historical_absolute_reference_exact" if first is None else
        "historical_absolute_reference_contextual_non_authorizing__same_evaluator_paired_screen_governs",
    }


class _MemoryArtifact:
    def __init__(self, topology=physical.V4_SCORE_TOPOLOGY) -> None:
        self.topology = topology
        self.bodies: dict[str, bytes] = {}
        self.events: list[str] = []

    def publish_group(self, bodies, *, post_publish=None):
        if any(name in self.bodies for name in bodies):
            raise RuntimeError("collision")
        digests = {name: hashlib.sha256(body).hexdigest() for name, body in bodies.items()}
        self.bodies.update(bodies)
        self.events.append("publish:" + ",".join(sorted(bodies)))
        if post_publish is not None:
            post_publish(bodies, digests)
        return digests

    def reload_pair(self, name, expected_sha256=None):
        body = self.bodies[name]
        if expected_sha256 is not None and hashlib.sha256(body).hexdigest() != expected_sha256:
            raise RuntimeError("sha")
        return body

    def reload_json(self, name, expected_sha256=None):
        return json.loads(self.reload_pair(name, expected_sha256))

    def has_name(self, name):
        return name in self.bodies


@dataclass(frozen=True)
class _FakeInput:
    payload_value: Mapping[str, object]

    def payload(self, *, identity):
        return dict(self.payload_value)


class _MockBackend:
    def __init__(self, identity: score.V4ScoreIdentity, *, fail: bool = False) -> None:
        self.identity = identity
        self.fail = fail
        self.cells: list[score.V4ScoreCell] = []
        self.closed = False
        self.prepared = False

    def prepare(self, *, identity):
        assert identity is self.identity
        self.prepared = True

    def materialize_inputs(self, *, identity, assets):
        assert self.prepared and identity is self.identity
        return _FakeInput(_input_payload(identity))

    def score_cell_v4(self, *, cell, input_authority_sha256):
        if self.fail and cell == score.score_matrix()[2]:
            raise RuntimeError("synthetic forward failure")
        self.cells.append(cell)
        return _evidence(self.identity)[len(self.cells) - 1]

    def reverify_after_v4_forwards(self):
        assert tuple(self.cells) == score.score_matrix()

    def historical_bridge_payload(self, *, input_authority_sha256):
        return _bridge(self.identity, mismatch=True)

    def close(self):
        self.closed = True


def _capability(identity: score.V4ScoreIdentity) -> physical.V4ExecutionCapability:
    closure = identity.closure.payload()
    return physical.V4ExecutionCapability(
        identity_sha256=score._digest(score._json(identity.payload())),
        device_profile={
            "cuda_visible_devices": "0", "cuda_device_order": "PCI_BUS_ID", "logical_device": "cuda:0",
            "uuid": "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9", "bdf": "00000000:01:00.0",
            "name": "NVIDIA GeForce RTX 3090", "nvidia_smi_memory_total_mib": 24576,
            "torch_total_memory_bytes": 25435111424, "torch_version": "2.5.1.post303",
            "torch_cuda_version": "11.8", "cudnn_version": 90300,
        },
        closure_sha256=str(closure["closure_sha256"]),
        failed_v3_graph_sha256=score._digest(score._json(_graph())), seal=physical._EXECUTION_SEAL,
        official_preflight_sha256=_sha("synthetic-official-preflight"),
        root_authorization_sha256=_sha("synthetic-root-authorization"),
    )


def test_matrix_is_exactly_b128_m30_m4_eight_cells() -> None:
    matrix = score.score_matrix()
    assert len(matrix) == 8
    assert [(row.surface, row.budget, row.system) for row in matrix] == [
        (surface, budget, system)
        for surface in (score.WITHIN, score.EXTERNAL)
        for budget in (30, 4)
        for system in (score.SYSTEM_PMC, score.SYSTEM_SEALED)
    ]
    assert {row.budget for row in matrix} == {30, 4}
    assert all(row.payload()["eval_batch_size"] == 128 for row in matrix)


def test_actual_v3_failed_graph_metadata_is_held_and_exact() -> None:
    graph = physical.validate_failed_v3_graph(ROOT)
    assert graph.payload() == _graph()
    assert len(graph.input_authority_payload["records"]) == 21


def test_failed_v3_semantic_tamper_is_rejected_before_any_v4_root() -> None:
    graph = physical.validate_failed_v3_graph(ROOT)
    payloads = {
        "preflight.json": {"schema": physical.v1p.PREFLIGHT_SCHEMA, "identity_sha256": physical.FAILED_V3_IDENTITY_SHA256},
        "authorization.json": {"schema": physical.v1p.AUTHORIZATION_SCHEMA, "identity_sha256": physical.FAILED_V3_IDENTITY_SHA256},
        "attempt.json": {"schema": physical.v1p.ATTEMPT_SCHEMA, "identity_sha256": physical.FAILED_V3_IDENTITY_SHA256},
        "input_authority.json": dict(graph.input_authority_payload),
        "failure.json": {
            "schema": physical.v1p.FAILURE_SCHEMA, "identity_sha256": physical.FAILED_V3_IDENTITY_SHA256,
            "stage": "score", "status": "SCORE_FAILED", "terminal_published": False,
            "input_authority_sha256": physical.FAILED_V3_BODY_SHA256["input_authority.json"],
            "attempt_sha256": physical._digest(physical._json({"schema": physical.v1p.ATTEMPT_SCHEMA, "identity_sha256": physical.FAILED_V3_IDENTITY_SHA256})),
            "preflight_sha256": physical._digest(physical._json({"schema": physical.v1p.PREFLIGHT_SCHEMA, "identity_sha256": physical.FAILED_V3_IDENTITY_SHA256})),
            "authorization_sha256": physical._digest(physical._json({"schema": physical.v1p.AUTHORIZATION_SCHEMA, "identity_sha256": physical.FAILED_V3_IDENTITY_SHA256})),
            "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
        },
    }
    payloads["failure.json"]["stage"] = "input"
    with pytest.raises(physical.V4PhysicalScoreError, match="stage/status"):
        physical._validate_failed_v3_payloads(payloads)


def test_input_record_comparator_rejects_one_digest_or_order_change() -> None:
    identity = _identity()
    payload = _input_payload(identity)
    physical.require_v3_input_records_equal(payload, payload)
    altered = json.loads(json.dumps(payload))
    altered["records"][0]["input_token_sha256"] = _sha("different")
    with pytest.raises(physical.V4PhysicalScoreError, match="records"):
        physical.require_v3_input_records_equal(altered, payload)


def test_score_payload_persists_contextual_bridge_without_tolerance_waiver() -> None:
    identity = _identity()
    payload = score.build_score_payload(
        identity=identity, input_payload=_input_payload(identity), evidence=_evidence(identity),
        historical_bridge=_bridge(identity, mismatch=True),
    )
    score.validate_score_payload(payload, identity=identity)
    assert payload["historical_absolute_reference"].startswith("contextual_non_authorizing")
    assert payload["paired_pmc_minus_sealed"]["external_M4_pmc_minus_sealed"]["n_positive"] == 15
    assert payload["paired_pmc_minus_sealed"]["within_M30_pmc_minus_sealed"]["mean"] == 0.0


def test_v4_lifecycle_finishes_paired_screen_even_when_historical_bridge_is_contextual() -> None:
    identity = _identity()
    artifact = _MemoryArtifact()
    backend = _MockBackend(identity)
    terminal = physical.run_v4_score_lifecycle(
        artifact=artifact, identity=identity, capability=_capability(identity), backend=backend,
        assets_factory=lambda: {score.WITHIN: (), score.EXTERNAL: ()},
        final_reverify=lambda: identity.closure.payload(),
    )
    assert terminal["status"] == "TERMINAL"
    assert terminal["historical_reference_contextual"] is True
    assert "score.json" in artifact.bodies and "failure.json" not in artifact.bodies
    assert backend.closed is True
    assert artifact.events[0] == "publish:attempt.json,authorization.json,preflight.json"
    tampered = dict(terminal)
    tampered["score_sha256"] = _sha("wrong-score")
    with pytest.raises(physical.V4PhysicalScoreError, match="score_sha256"):
        physical.validate_v4_terminal(
            tampered, identity=identity,
            expected_hashes={name: hashlib.sha256(body).hexdigest() for name, body in artifact.bodies.items()},
            score_payload=artifact.reload_json("score.json"),
        )


def test_v4_lifecycle_failure_has_no_terminal() -> None:
    identity = _identity()
    artifact = _MemoryArtifact()
    backend = _MockBackend(identity, fail=True)
    with pytest.raises(RuntimeError, match="synthetic forward"):
        physical.run_v4_score_lifecycle(
            artifact=artifact, identity=identity, capability=_capability(identity), backend=backend,
            assets_factory=lambda: {score.WITHIN: (), score.EXTERNAL: ()},
            final_reverify=lambda: identity.closure.payload(),
        )
    assert "failure.json" in artifact.bodies and "terminal.json" not in artifact.bodies
    failure = artifact.reload_json("failure.json")
    assert failure["stage"] == "score" and failure["terminal_published"] is False


def test_target_free_authority_pair_is_distinct_and_binds_v3_graph() -> None:
    identity = _identity()
    artifact = _MemoryArtifact(physical.V4_AUTHORITY_TOPOLOGY)
    profile = _capability(identity).device_profile
    durable = physical.publish_v4_durable_authority(
        artifact=artifact, identity=identity,
        graph=physical.FailedV3Graph(
            body_sha256=dict(physical.FAILED_V3_BODY_SHA256),
            identity_sha256=physical.FAILED_V3_IDENTITY_SHA256,
            input_authority_payload=_input_payload(identity),
        ),
        device_profile=profile, environ=score.LAUNCH_ENVIRONMENT,
    )
    assert set(artifact.bodies) == {"official_preflight.json", "root_authorization.json"}
    assert durable.official_preflight_sha256 == hashlib.sha256(artifact.bodies["official_preflight.json"]).hexdigest()
    preflight = artifact.reload_json("official_preflight.json")
    assert preflight["assets_deferred_until_score_attempt"] is True
    assert preflight["target_opened"] is False


def test_wrong_launch_environment_blocks_authority_publish_and_reservation_before_write() -> None:
    identity = _identity()
    graph = physical.FailedV3Graph(
        body_sha256=dict(physical.FAILED_V3_BODY_SHA256),
        identity_sha256=physical.FAILED_V3_IDENTITY_SHA256,
        input_authority_payload=_input_payload(identity),
    )
    artifact = _MemoryArtifact(physical.V4_AUTHORITY_TOPOLOGY)
    bad_env = {**score.LAUNCH_ENVIRONMENT, "SUBM_DATA_ROOT": "/wrong/subm"}
    with pytest.raises(physical.v3p.V3PhysicalScoreError, match="launch environment"):
        physical.publish_v4_durable_authority(
            artifact=artifact, identity=identity, graph=graph,
            device_profile=_capability(identity).device_profile, environ=bad_env,
        )
    assert artifact.bodies == {}
    # The same gate is intentionally first in reservation: this direct call
    # does not reach graph/provenance/root reservation under wrong env.
    with pytest.raises(physical.v3p.V3PhysicalScoreError, match="launch environment"):
        physical.reserve_v4_authority_root(ROOT, identity=identity, graph=graph, environ=bad_env)
    with pytest.raises(physical.v3p.V3PhysicalScoreError, match="launch environment"):
        physical.reserve_v4_score_root(ROOT, graph=graph, environ=bad_env)


def test_issuer_reloads_durable_authority_and_rebuilds_closure_before_mint() -> None:
    identity = _identity()
    graph = physical.FailedV3Graph(
        body_sha256=dict(physical.FAILED_V3_BODY_SHA256),
        identity_sha256=physical.FAILED_V3_IDENTITY_SHA256,
        input_authority_payload=_input_payload(identity),
    )
    profile = _capability(identity).device_profile
    durable = physical.V4DurableAuthority(
        official_preflight_sha256=_sha("official"), root_authorization_sha256=_sha("authorization"),
        device_profile=profile,
    )
    review = physical.issue_v4_root_review_capability()
    capability = physical.issue_v4_execution_capability(
        ROOT, root_review_capability=review, identity=identity, device_profile=profile,
        durable_authority=durable, environ=score.LAUNCH_ENVIRONMENT,
        graph_loader=lambda _root: graph,
        durable_authority_loader=lambda _root, *, identity, graph: durable,
        closure_rebuilder=lambda _root, _graph: identity.closure,
        producer_validator=lambda _root, *, identity, graph: None,
    )
    capability.verify(identity)
    forged = physical.V4DurableAuthority(
        official_preflight_sha256=_sha("forged"), root_authorization_sha256=durable.root_authorization_sha256,
        device_profile=profile,
    )
    with pytest.raises(physical.V4PhysicalScoreError, match="durable authority differs"):
        physical.issue_v4_execution_capability(
            ROOT, root_review_capability=review, identity=identity, device_profile=profile,
            durable_authority=forged, environ=score.LAUNCH_ENVIRONMENT,
            graph_loader=lambda _root: graph,
            durable_authority_loader=lambda _root, *, identity, graph: durable,
            closure_rebuilder=lambda _root, _graph: identity.closure,
            producer_validator=lambda _root, *, identity, graph: None,
        )

    class _StaleClosure:
        def payload(self):
            return {**identity.closure.payload(), "closure_sha256": _sha("stale-closure")}

    with pytest.raises(physical.V4PhysicalScoreError, match="V2/V3/V4 closure drift"):
        physical.issue_v4_execution_capability(
            ROOT, root_review_capability=review, identity=identity, device_profile=profile,
            durable_authority=durable, environ=score.LAUNCH_ENVIRONMENT,
            graph_loader=lambda _root: graph,
            durable_authority_loader=lambda _root, *, identity, graph: durable,
            closure_rebuilder=lambda _root, _graph: _StaleClosure(),
            producer_validator=lambda _root, *, identity, graph: None,
        )

    stale_input_identity = score.V4ScoreIdentity(
        base_identity=identity.base_identity, closure=identity.closure,
        v3_input_authority_sha256=_sha("stale-v3-input"),
    )
    with pytest.raises(physical.V4PhysicalScoreError, match="input-authority SHA"):
        physical.issue_v4_execution_capability(
            ROOT, root_review_capability=review, identity=stale_input_identity, device_profile=profile,
            durable_authority=durable, environ=score.LAUNCH_ENVIRONMENT,
            graph_loader=lambda _root: graph,
            durable_authority_loader=lambda _root, *, identity, graph: durable,
            closure_rebuilder=lambda _root, _graph: identity.closure,
            producer_validator=lambda _root, *, identity, graph: None,
        )


def test_runtime_rejects_b32_subclass_before_runtime_or_data_access() -> None:
    class _B32Runtime(physical.V4PhysicalRuntime):
        EVAL_BATCH_SIZE = 32

    with pytest.raises(physical.V4PhysicalScoreError, match="B128"):
        _B32Runtime._forward_once(object.__new__(_B32Runtime), system=score.SYSTEM_SEALED, budget=30, model=None, session=None)


def test_physical_seam_is_narrow_and_b128_without_runtime_import() -> None:
    assert issubclass(physical.V4PhysicalRuntime, physical.v2p.V2PhysicalRuntime)
    assert issubclass(physical.V4PhysicalMatchedScoreBackend, physical.v2p.V2PhysicalMatchedScoreBackend)
    source = Path(physical.__file__).read_text(encoding="utf-8")
    assert "EVAL_BATCH_SIZE = score.EVAL_BATCH_SIZE" in source
    assert "range(0, len(starts), self.EVAL_BATCH_SIZE)" in source
    assert "build_spintshape_model(seed=42)" in source
    assert "def _forward_once" in source


def test_fixed_closure_recomputes_and_workorder_sha_is_literal() -> None:
    closure = score.implementation_closure(ROOT, v3_base_closure={"schema": "synthetic-v3"}, failed_v3_graph=_graph())
    payload = closure.payload()
    assert payload["local_sha256_by_path"][score.WORKORDER_RELATIVE] == score.WORKORDER_SHA256
    assert len(payload["local_paths"]) == len(score.V4_LOCAL_CLOSURE)


def test_fresh_root_guard_is_read_only_and_rejects_collision(tmp_path: Path) -> None:
    score.assert_fresh_prospective_roots(tmp_path)
    occupied = tmp_path / score.SCORE_ROOT_RELATIVE
    occupied.parent.mkdir(parents=True)
    occupied.mkdir()
    with pytest.raises(score.ScoreV4Error, match="not fresh"):
        score.assert_fresh_prospective_roots(tmp_path)


def test_public_cli_is_dry_no_torch_and_execution_flags_fail_closed() -> None:
    cli = ROOT / "tfpd_exploration/scripts/run_posterior_marginalized_cell_d_matched_score_v4.py"
    clean_env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "CUDA_VISIBLE_DEVICES"}}
    clean_env["PYTHONNOUSERSITE"] = "1"
    dry = subprocess.run([sys.executable, "-S", str(cli), "--dry-run"], cwd=ROOT, env=clean_env,
                         capture_output=True, text=True, check=False)
    assert dry.returncode == 0, dry.stderr
    assert json.loads(dry.stdout)["evaluation_batch_size"] == 128
    probe = subprocess.run([
        sys.executable, "-S", "-c",
        "import runpy,sys; sys.argv=['route','--dry-run']; "
        "\ntry:\n runpy.run_path(sys.argv[0] if False else 'tfpd_exploration/scripts/run_posterior_marginalized_cell_d_matched_score_v4.py',run_name='__main__')"
        "\nexcept SystemExit:\n pass\nprint('TORCH_LOADED='+str('torch' in sys.modules))",
    ], cwd=ROOT, env=clean_env, capture_output=True, text=True, check=False)
    assert probe.returncode == 0, probe.stderr
    assert "TORCH_LOADED=False" in probe.stdout
    denied = subprocess.run([sys.executable, "-S", str(cli), "--execute", "--root-reviewed"], cwd=ROOT,
                            env=clean_env, capture_output=True, text=True, check=False)
    assert denied.returncode != 0
    assert "capability" in (denied.stderr + denied.stdout).lower()
