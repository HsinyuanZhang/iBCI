"""No-data/no-CUDA tests for the seed43 Phase-E V2 replication wrapper.

The real seed42 V2 terminal is immutable evidence and the seed43 terminal is
still live while this candidate is assembled.  These tests therefore build a
complete synthetic V2 outer graph using the already-tested seed42 V2 mock
lifecycle; they never inspect either live score/training result, an NWB, a
checkpoint tensor, or CUDA.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_score_v2/score_v2.py"
CLI = ROOT / "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed43_score_v2.py"
BASE_V2_TEST = ROOT / "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_seed42_score_v2.py"


def _load(path: Path, package_name: str) -> ModuleType:
    package = ModuleType(package_name)
    package.__path__ = [str(path.parent)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(package_name + ".module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


route = _load(MODULE, "_test_tfsr_seed43_phase_e_v2")
base = _load(BASE_V2_TEST, "_test_tfsr_seed42_phase_e_v2_helpers")
s42v2 = route._seed42_v2()
v1 = s42v2._v1()
s43 = route._seed43_v1()
WITHIN = tuple(sorted(row[0] for row in v1.SEALED_WITHIN_PAIRED_VIEW_ROWS))
EXTERNAL = tuple(f"external-{index:02d}" for index in range(15))


def _sha(value: str | bytes) -> str:
    return route._sha(value.encode("utf-8") if isinstance(value, str) else value)


def _write_pair(directory: Path, name: str, payload: Mapping[str, object]) -> str:
    body = route._json_bytes(payload)
    digest = route._sha(body)
    leaf = directory / name
    leaf.write_bytes(body)
    sidecar = directory / f"{name}.sha256"
    sidecar.write_bytes(f"{digest}  {name}\n".encode("ascii"))
    os.chmod(leaf, 0o444)
    os.chmod(sidecar, 0o444)
    return digest


def _seed42_v2_graph(tmp_path: Path) -> tuple[route.Seed42V2Evidence, route.Seed42V2Expectation, dict[str, dict[str, object]]]:
    """Create a valid synthetic *completed* seed42 V2 graph.

    We deliberately call the independent seed42 V2 mock lifecycle rather
    than hand-writing an outer receipt.  That makes the V2-to-nested-V1
    validation exercised here a real compatibility test, not a schema echo.
    """
    identity, v1_identity = base._v2_identity()
    authorization = base.score.V2Authorization(
        preflight_sha256=identity.v2_authorization["preflight_sha256"],
        root_authorization_sha256=identity.v2_authorization["root_authorization_sha256"],
        preflight={},
        root_authorization={},
        expected_closure=identity.launch_closure,
    )
    artifact = base.v1.reserve_artifact_root(tmp_path, "synthetic-seed42-v2", base.score.SCORE_TOPOLOGY)
    within = base._within_bindings(tmp_path)
    external = base._external_bindings(tmp_path)
    class StopBackend(base._MockBackend):
        """Use the accepted V2 lifecycle with a valid, explicitly STOP delta."""

        def score_tfsr(self, *, surface: str, mode: str, input_authority_sha256: str, flags: Any) -> Any:
            self.calls.append(f"t:{surface}:{mode}")
            flags.record_forward("tfsr", surface, mode)
            # The outer V2 terminal must match the real accepted upstream
            # fact: STOP.  A negative aligned TF-SR versus Cell-D contrast is
            # a fully valid synthetic way to exercise that fixed verdict.
            return base._mode("tfsr", surface, mode, input_authority_sha256, offset=-0.10)

    backend = StopBackend(within, external)
    base.score.run_score_lifecycle_v2(
        artifact=artifact,
        identity=identity,
        execution_capability=base.score._issue_execution_capability(authorization),
        v1_identity=v1_identity,
        within_roster=base.WITHIN,
        external_sessions=base.EXTERNAL,
        within_roster_factory=lambda: within,
        external_roster_factory=lambda: external,
        sealed_cell_d_table=base._sealed_table(),
        a2_pooled={"within": {name: 0.4 for name in base.WITHIN}, "external": {name: 0.2 for name in base.EXTERNAL}},
        backend=backend,
        final_reverify=lambda: identity.launch_closure,
    )
    payloads = {
        name: artifact.reload_json(name, route._sha(artifact.reload_pair(name)))
        for name in ("attempt.json", "input_authority.json", "score.json", "terminal.json")
    }
    expectation = route.Seed42V2Expectation(
        attempt_sha256=route._sha(route._json_bytes(payloads["attempt.json"])),
        input_sha256=route._sha(route._json_bytes(payloads["input_authority.json"])),
        score_sha256=route._sha(route._json_bytes(payloads["score.json"])),
        terminal_sha256=route._sha(route._json_bytes(payloads["terminal.json"])),
    )
    evidence = route.validate_seed42_v2_payload_graph(
        s42v2,
        attempt_payload=payloads["attempt.json"],
        input_payload=payloads["input_authority.json"],
        score_payload=payloads["score.json"],
        terminal_payload=payloads["terminal.json"],
        expectation=expectation,
        score_root_relative="synthetic/seed42-v2",
        score_root_identity=(17, 23),
        current_closure=identity.launch_closure,
    )
    return evidence, expectation, payloads


def _seed43_training() -> Any:
    return s43.Seed43TrainingEvidence(
        terminal_sha256=_sha("seed43-terminal"),
        swa_sha256=_sha("seed43-swa"),
        swa_state_digest=_sha("seed43-state"),
        checkpoint_sha256={str(epoch): _sha(f"seed43-checkpoint-{epoch}") for epoch in (44, 45, 46, 47)},
        launch_closure={"closure_sha256": _sha("seed43-launch-closure")},
        lineage={"seed": 43, "synthetic": True},
        build_disclosure={"build": s43.BUILD_LABEL},
        training_peak_memory_bytes=17,
        cell=s43.CELL,
    )


def _v2_identity(upstream: route.Seed42V2Evidence, training: Any) -> tuple[route.Seed43V2Identity, route.ExecutionCapability]:
    closure = {
        "paths": ["synthetic-seed43-v2"],
        "sha256_by_path": {"synthetic-seed43-v2": _sha("synthetic-seed43-v2")},
        "seed42_v2_closure": {"synthetic": True},
        "closure_sha256": _sha("synthetic-seed43-v2-closure"),
    }
    authorization = route.Seed43V2Authorization(
        preflight_sha256=_sha("seed43-v2-preflight"),
        root_authorization_sha256=_sha("seed43-v2-root-authorization"),
        preflight={"synthetic": True},
        root_authorization={"synthetic": True},
        closure=closure,
    )
    identity = route.Seed43V2Identity(
        training=training,
        upstream=upstream,
        closure=closure,
        authorization=authorization.payload(),
    )
    return identity, route._issue_execution_capability(authorization)


def _within_bindings(tmp_path: Path) -> tuple[Any, ...]:
    rows = v1.sealed_within_assets()
    return tuple(
        v1.EvaluationAssetBinding(
            surface="within",
            asset_id=str(row["asset_id"]),
            session=str(row["session"]),
            local_path=tmp_path / Path(str(row["frozen_path"])).name,
            expected_bytes=int(row["bytes"]),
            expected_sha256=str(row["sha256"]),
            frozen_path=str(row["frozen_path"]),
        )
        for row in rows
    )


def _external_bindings(tmp_path: Path) -> tuple[Any, ...]:
    return tuple(
        v1.ExternalAssetBinding(
            asset_id=f"asset-{index:02d}",
            session=name,
            local_path=tmp_path / f"{name}.nwb",
            expected_bytes=7,
            expected_sha256=_sha(f"asset:{name}"),
            frozen_path=f"sub-M/{name}.nwb",
        )
        for index, name in enumerate(EXTERNAL)
    )


def _input_evidence(within: tuple[Any, ...], external: tuple[Any, ...]) -> Any:
    records: list[Any] = []
    tensor = {"dtype": "torch.float32", "shape": [3, 4], "bytes_sha256": _sha("tensor")}

    def add(session: str, surface: str, asset: Mapping[str, object]) -> None:
        proof = {
            "schema": "tfsr_phase_e_raw_t4_sua_axis_proof_v1",
            "session": session,
            "signal_view": "sua",
            "channel_ids_dtype": "int64",
            "channel_ids_sha256": _sha(f"chan:{session}"),
            "source_unit_count": 3,
            "raw_t4_shape": [3, 4],
            "feature_group": "t4",
            "pool_size": 30,
            "mapping": "raw_t4_row_k_equals_sua_neural_column_k_for_contiguous_channel_ids",
            "closure_bound_functions": [
                "mc_maze.multisession_datamodule.load_dandi688_session",
                "mc_maze.unit_side_features.compute_unit_side_features_uncached",
            ],
        }
        records.append(
            v1.SessionInputAuthority(
                session=session,
                surface=surface,
                asset=asset,
                ordered_unit_digest=_sha(f"units:{session}"),
                unit_count=3,
                raw_t4=tensor,
                normalized_t4=tensor,
                calibration_sha256=_sha(f"cal:{session}"),
                query_start_sha256=_sha(f"start:{session}"),
                neural_sha256=_sha(f"neural:{session}"),
                behavior_sha256=_sha(f"beh:{session}"),
                valid_mask_sha256=_sha(f"mask:{session}"),
                target_last_bin_sha256=_sha(f"target:{session}"),
                last_bin_valid_mask_sha256=_sha(f"last:{session}"),
                last_bin_valid_count=100,
                n_windows=100,
                raw_t4_sua_axis_proof=proof,
                raw_t4_sua_axis_proof_sha256=_sha(v1._json_bytes(proof)),
            )
        )

    for item in within:
        add(item.session, "within", item.payload())
    for item in external:
        # The V2 lifecycle intentionally normalizes frozen UUID-ledger
        # bindings before the shared input-authority validator sees them.
        asset = item.payload() if isinstance(item, v1.EvaluationAssetBinding) else v1._as_evaluation_asset(item).payload()
        add(item.session, "external", asset)
    return v1.InputAuthorityEvidence(
        records=tuple(sorted(records, key=lambda item: (item.surface, item.session))),
        source_normalizer_body_sha256=v1.SOURCE_NORMALIZER_BODY_SHA,
        t4_normalizer_semantic_sha256=v1.T4_NORMALIZER_SEMANTIC_SHA,
        behavior_normalizer_semantic_sha256=v1.BEHAVIOR_NORMALIZER_SEMANTIC_SHA,
        strict_manifest_sha256=next(item.sha256 for item in v1.FIXED_AUTHORITIES if item.name == "strict_manifest"),
        raw_to_normalized_exact=True,
        no_cache_readonly_adapter=True,
    )


def _mode(system: str, surface: str, mode: str, input_sha: str, *, offset: float) -> Any:
    names = WITHIN if surface == "within" else EXTERNAL
    base_value = 0.50 if surface == "within" else 0.30
    sessions = tuple(
        v1.SessionScore(
            session=name,
            n_windows=100 + index,
            governing_r2=base_value + index / 1000.0 + offset,
            full_window_r2=base_value + index / 1000.0 + offset - 0.01,
            output_sha256=_sha(f"out:{system}:{surface}:{mode}:{name}"),
            input_authority_sha256=input_sha,
        )
        for index, name in enumerate(names)
    )
    common: dict[str, object] = {
        "system": system,
        "surface": surface,
        "mode": mode,
        "sessions": sessions,
        "state_before_sha256": _sha(f"state:{system}:{surface}:{mode}"),
        "state_after_sha256": _sha(f"state:{system}:{surface}:{mode}"),
        "eval_mode": True,
        "gradients_none": True,
        "finite_output": True,
        "output_shape": (2, 50, 2),
        "latency_ms": 1.0,
        "peak_memory_bytes": 1,
    }
    if system == "tfsr":
        control: dict[str, object] = {"mode": mode, "post_normalization": True, "b3s_recomputed": True}
        if mode == "zero":
            control["exact_zeros_like"] = True
        if mode == "wrong_pair":
            control.update({"permutation_is_derangement": True, "permutation_sha256": _sha("permutation")})
        common.update(
            {
                "repeat_bitwise_equal": mode == "aligned",
                "eval_no_mask": True,
                "capture_diagnostics": False,
                "b3s_recomputed": True,
                "t4_control": control,
            }
        )
    return v1.ModeEvidence(**common)


class _MockBackend:
    def __init__(self, within: tuple[Any, ...], external: tuple[Any, ...]) -> None:
        self.within = within
        self.external = external
        self.closed = False
        self.manifest = s43.AcceleratedParityManifest()
        self.calls: list[str] = []

    def resolve_inputs(self, *, flags: Any, external_roster: tuple[Any, ...], **_: object) -> Any:
        self.calls.append("resolve")
        flags.within_opened = True
        flags.external_opened = True
        assert all(isinstance(item, v1.EvaluationAssetBinding) for item in external_roster)
        return _input_evidence(self.within, external_roster)

    def score_tfsr(self, *, surface: str, mode: str, input_authority_sha256: str, flags: Any) -> Any:
        self.calls.append(f"tfsr:{surface}:{mode}")
        flags.record_forward("tfsr", surface, mode)
        self.manifest.record(
            surface=surface,
            mode=mode,
            eager_bytes=b"same-synthetic-output",
            accelerated_bytes=b"same-synthetic-output",
            shape=(2, 50, 2),
        )
        return _mode("tfsr", surface, mode, input_authority_sha256, offset=0.03 if mode == "aligned" else -0.02)

    def build_parity_manifest(self, *, flags: Any) -> Mapping[str, object]:
        return self.manifest.payload(flags)

    def reverify_after_forwards(self, **_: object) -> None:
        self.calls.append("reverify")

    def resource_disclosure(self) -> Mapping[str, object]:
        return {
            "schema": "tfsr_seed43_phase_e_replication_resources_v1",
            "runtime": {**dict(v1.FROZEN_SCORE_DEVICE), "torchmetrics_version": "1.5.1"},
            "seed43_training_peak_memory_bytes": 17,
            "score_peak_memory_bytes": 1,
            "aligned_eager_latency_ms": 1.0,
            "tfsr_parameters": 10,
            "build": s43.BUILD_LABEL,
            "cell_d_replayed_upstream_only": True,
            "cell_d_forward_calls": 0,
            "forward_only": True,
            "no_grad": True,
        }

    def close(self) -> None:
        self.closed = True


def test_static_cli_is_no_torch_and_partial_flags_fail_before_any_authority(tmp_path: Path) -> None:
    (tmp_path / "torch.py").write_text("raise RuntimeError('TORCH_IMPORTED')\n")
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    dry = subprocess.run([sys.executable, str(CLI)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert dry.returncode == 0, dry.stderr
    assert json.loads(dry.stdout)["phase"] == route.PHASE
    assert "TORCH_IMPORTED" not in dry.stdout + dry.stderr
    partial = subprocess.run([sys.executable, str(CLI), "--execute"], cwd=ROOT, env=env, text=True, capture_output=True)
    assert partial.returncode != 0
    assert "TORCH_IMPORTED" not in partial.stdout + partial.stderr


def test_seed42_v2_outer_graph_and_nested_v1_payloads_are_independently_validated(tmp_path: Path) -> None:
    evidence, expectation, payloads = _seed42_v2_graph(tmp_path)
    assert evidence.verdict == "STOP"
    assert evidence.binding_payload()["upstream_kind"] == "completed_seed42_phase_e_v2_wrapper"
    assert evidence.input_authority_sha256 == payloads["input_authority.json"]["v1_input_authority_sha256"]
    assert evidence.score_sha256 == payloads["score.json"]["v1_score_sha256"]

    # Change nested evidence while truthfully recomputing every *outer* body
    # SHA.  The nested V1 schema/digest validation must still reject it.
    forged_input = json.loads(json.dumps(payloads["input_authority.json"]))
    forged_input["v1_input_authority"]["cell"] = "forged-cell"
    forged_expectation = route.Seed42V2Expectation(
        expectation.attempt_sha256,
        route._sha(route._json_bytes(forged_input)),
        expectation.score_sha256,
        expectation.terminal_sha256,
    )
    with pytest.raises(route.FailClosedError, match="outer graph|nested V1"):
        route.validate_seed42_v2_payload_graph(
            s42v2,
            attempt_payload=payloads["attempt.json"],
            input_payload=forged_input,
            score_payload=payloads["score.json"],
            terminal_payload=payloads["terminal.json"],
            expectation=forged_expectation,
            score_root_relative="synthetic/seed42-v2",
            score_root_identity=(17, 23),
            current_closure=payloads["terminal.json"]["launch_closure"],
        )


def test_seed42_v2_held_pair_reader_requires_exact_topology_sidecars_and_modes(tmp_path: Path) -> None:
    _evidence, expectation, payloads = _seed42_v2_graph(tmp_path)
    root = tmp_path / s42v2.SCORE_ROOT_RELATIVE
    root.mkdir(parents=True)
    for name, payload in payloads.items():
        _write_pair(root, name, payload)
    loaded, identity = route._read_seed42_v2_pairs(tmp_path, expectation=expectation)
    assert identity == route._directory_identity(root)
    assert set(loaded) == set(payloads)
    extra = root / "failure.json"
    extra.write_text("{}\n")
    os.chmod(extra, 0o444)
    with pytest.raises(route.FailClosedError, match="topology"):
        route._read_seed42_v2_pairs(tmp_path, expectation=expectation)
    os.unlink(extra)
    body = root / "score.json"
    os.chmod(body, 0o644)
    with pytest.raises(route.FailClosedError, match="mode"):
        route._read_seed42_v2_pairs(tmp_path, expectation=expectation)


def test_current_seed42_v2_closure_is_required_not_merely_self_consistent(tmp_path: Path) -> None:
    _evidence, expectation, payloads = _seed42_v2_graph(tmp_path)
    drifted = json.loads(json.dumps(payloads["terminal.json"]["launch_closure"]))
    drifted["closure_sha256"] = _sha("drift")
    with pytest.raises(route.FailClosedError, match="current closure"):
        route.validate_seed42_v2_payload_graph(
            s42v2,
            attempt_payload=payloads["attempt.json"],
            input_payload=payloads["input_authority.json"],
            score_payload=payloads["score.json"],
            terminal_payload=payloads["terminal.json"],
            expectation=expectation,
            score_root_relative="synthetic/seed42-v2",
            score_root_identity=(17, 23),
            current_closure=drifted,
        )


def test_v2_closure_is_explicit_and_detects_missing_or_drifted_leaf() -> None:
    closure = route.phase_e_seed43_v2_closure(ROOT)
    assert route.validate_phase_e_seed43_v2_closure(closure) == closure
    assert route.WORKORDER_RELATIVE in closure["paths"]
    assert s43.SOURCE_RELATIVE in closure["paths"]
    assert "tfpd_exploration/src/tfsr_b3st4_ddrop_seed42_score_v2/score_v2.py" in closure["paths"]
    forged = json.loads(json.dumps(closure))
    forged["sha256_by_path"].pop(route.CLI_RELATIVE)
    with pytest.raises(route.FailClosedError, match="topology"):
        route.validate_phase_e_seed43_v2_closure(forged)
    forged = json.loads(json.dumps(closure))
    forged["sha256_by_path"][route.SOURCE_RELATIVE] = _sha("drift")
    with pytest.raises(route.FailClosedError, match="aggregate"):
        route.validate_phase_e_seed43_v2_closure(forged)


def test_exact_data_roots_are_checked_before_upstream_training_or_reservation() -> None:
    calls: list[str] = []
    with pytest.raises(route.FailClosedError, match="exact canonical"):
        route._prepare_execution(
            ROOT,
            environment={"SUBC_DATA_ROOT": str(Path(route.SUBC_DATA_ROOT).parent), "SUBM_DATA_ROOT": route.SUBM_DATA_ROOT},
            upstream_loader=lambda _root: calls.append("upstream") or None,
            training_loader=lambda _root: calls.append("training") or None,
            authorization_loader=lambda _root, _upstream, _training: calls.append("authorization") or None,
        )
    assert calls == []


def test_incomplete_seed43_training_is_no_go_before_authorization_or_any_root_reservation(tmp_path: Path) -> None:
    upstream, _expectation, _payloads = _seed42_v2_graph(tmp_path)
    calls: list[str] = []
    with pytest.raises(route.FailClosedError, match="incomplete"):
        route._prepare_execution(
            ROOT,
            environment={"SUBC_DATA_ROOT": route.SUBC_DATA_ROOT, "SUBM_DATA_ROOT": route.SUBM_DATA_ROOT},
            upstream_loader=lambda _root: calls.append("upstream") or upstream,
            training_loader=lambda _root: (_ for _ in ()).throw(route.FailClosedError("incomplete seed43 training")),
            authorization_loader=lambda _root, _upstream, _training: calls.append("authorization") or None,
        )
    assert calls == ["upstream"]


def test_physical_backend_is_exact_frozen_seed43_v1_composition_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    sentinel = object()
    monkeypatch.setattr(s43, "build_seed43_physical_backend", lambda **kwargs: calls.append(kwargs) or sentinel)
    result = route.build_physical_backend(
        root=ROOT,
        fixed_authorities={"fixed": object()},
        upstream_training=object(),
        upstream_authorization=object(),
        seed43_training=object(),
    )
    assert result is sentinel
    assert len(calls) == 1
    assert calls[0]["root"] == ROOT
    assert set(calls[0]) == {"root", "fixed_authorities", "upstream_training", "upstream_authorization", "seed43_training"}


def test_complete_mock_v2_lifecycle_keeps_seed42_v2_replay_and_six_tfsr_cells(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upstream, _expectation, _payloads = _seed42_v2_graph(tmp_path)
    training = _seed43_training()
    identity, capability = _v2_identity(upstream, training)
    # The frozen core imports its seed42 V1 scorer lazily.  Bind the same
    # static V1 module used to build the nested V2 evidence; no production
    # global is modified and no physical backend is constructed.
    monkeypatch.setattr(s43, "_seed42_score", lambda: v1)
    monkeypatch.setattr(route, "validate_phase_e_seed43_v2_closure", lambda value: dict(value))
    artifact = v1.reserve_artifact_root(tmp_path, "seed43-v2-success", route.SCORE_TOPOLOGY)
    within = _within_bindings(tmp_path)
    external = _external_bindings(tmp_path)
    backend = _MockBackend(within, external)
    terminal = route.run_seed43_v2_lifecycle(
        artifact=artifact,
        identity=identity,
        execution_capability=capability,
        within_roster=WITHIN,
        external_sessions=EXTERNAL,
        within_assets_factory=lambda: within,
        external_assets_factory=lambda: external,
        backend=backend,
        final_reverify=lambda: identity.closure,
    )
    score_body = artifact.reload_pair("score.json")
    persisted_score = artifact.reload_json("score.json", route._sha(score_body))
    route.validate_score_payload(persisted_score, identity=identity, input_sha=persisted_score["input_replay_sha256"])
    route.validate_terminal_payload(
        terminal,
        identity=identity,
        score=persisted_score,
        expected_score_sha=route._sha(score_body),
    )
    assert terminal["status"] == "REPLICATION_SCORE_COMPLETE"
    assert terminal["seed42_v2_verdict"] == "STOP"
    assert not artifact.has_name("failure.json")
    assert backend.closed is True
    assert len([item for item in backend.calls if item.startswith("tfsr:")]) == 6
    assert len(persisted_score["seed43_v1_core_score"]["accelerated_eager_parity"]["events"]) == 6


def test_target_free_preflight_binds_completed_seed43_evidence_upstream_and_superseded_code_only(tmp_path: Path) -> None:
    upstream, _expectation, _payloads = _seed42_v2_graph(tmp_path)
    training = _seed43_training()
    closure = route.phase_e_seed43_v2_closure(ROOT)
    preflight = route.build_target_free_preflight(ROOT, upstream=upstream, training=training, closure=closure)
    assert route.validate_target_free_preflight(preflight, upstream=upstream, training=training, closure=closure) == preflight
    assert preflight["superseded_seed43_v1_code_evidence"]["role"] == "superseded_code_evidence_only_no_result_required"
    forged = json.loads(json.dumps(preflight))
    forged["seed43_training"]["terminal_sha256"] = _sha("forged")
    with pytest.raises(route.FailClosedError, match="preflight"):
        route.validate_target_free_preflight(forged, upstream=upstream, training=training, closure=closure)
