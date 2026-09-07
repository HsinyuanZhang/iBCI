"""No-data adversarial tests for the additive Phase-E V2 wrapper."""
from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "tfpd_exploration/src/tfsr_b3st4_ddrop_seed42_score_v2/score_v2.py"
CLI = ROOT / "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed42_score_v2.py"


def _load() -> ModuleType:
    package = ModuleType("_test_tfsr_phase_e_v2")
    package.__path__ = [str(MODULE.parent)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(package.__name__ + ".score_v2", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


score = _load()
v1 = score._v1()
WITHIN = tuple(sorted(row[0] for row in v1.SEALED_WITHIN_PAIRED_VIEW_ROWS))
EXTERNAL = tuple(f"external-{index:02d}" for index in range(15))


def _sha(text: str | bytes) -> str:
    return score._sha(text if isinstance(text, bytes) else text.encode("utf-8"))


class _Material:
    def __init__(self, name: str) -> None:
        self.name = name

    def binding(self) -> dict[str, object]:
        return {"relative_path": f"{self.name}.json", "body_sha256": _sha(self.name), "mode": "0444"}


def _v1_context() -> tuple[dict[str, _Material], Any, Any, Any]:
    fixed = {"fixed": _Material("fixed")}
    training = v1.TrainingEvidence(
        terminal_sha256=_sha("terminal"), swa_sha256=_sha("swa"), swa_state_digest=_sha("state"),
        closure={"paths": ["phase-d"], "sha256_by_path": {"phase-d": _sha("phase-d")}, "closure_sha256": _sha("pdc")},
        checkpoint_sha256={str(epoch): _sha(f"checkpoint:{epoch}") for epoch in (44, 45, 46, 47)},
    )
    closure = {"paths": ["v1"], "sha256_by_path": {"v1": _sha("v1")}, "closure_sha256": _sha("v1c")}
    authorization = v1.PhaseEAuthorization(
        preflight_sha256=_sha("pre"), root_authorization_sha256=_sha("auth"),
        preflight={"synthetic": True}, root_authorization={"synthetic": True}, expected_closure=closure,
    )
    identity = score._expected_v1_identity(v1, fixed, training, authorization)
    return fixed, training, authorization, identity


def _write_pair(directory: Path, name: str, payload: dict[str, object]) -> str:
    body = v1._json_bytes(payload)
    digest = score._sha(body)
    leaf = directory / name
    leaf.write_bytes(body)
    (directory / f"{name}.sha256").write_bytes(f"{digest}  {name}\n".encode("ascii"))
    os.chmod(leaf, 0o444)
    os.chmod(directory / f"{name}.sha256", 0o444)
    return digest


def _make_synthetic_v1_failed_root(tmp_path: Path) -> tuple[Path, dict[str, _Material], Any, Any, Any, score.V1FailureExpectation]:
    fixed, training, authorization, identity = _v1_context()
    directory = tmp_path / "v1-failure"; directory.mkdir()
    attempt = v1._attempt_payload(identity)
    failure = {
        "schema": "tfsr_phase_e_score_failure_v1", "cell": v1.CELL, "phase": v1.PHASE,
        "stage": "resolve_inputs",
        "resolved": {"source": False, "within": True, "external": True, "formal": False},
        "opened": {"source": False, "within": False, "external": False, "formal": False},
        "forward_calls": v1.ScoreFlags().payload()["forward_calls"],
        "backward_calls": 0, "optimizer_calls": 0, "terminal_published": False,
        "traceback_sha256": _sha("synthetic trace"),
    }
    attempt_sha = _write_pair(directory, "attempt.json", attempt)
    failure_sha = _write_pair(directory, "failure.json", failure)
    return directory, fixed, training, authorization, identity, score.V1FailureExpectation(attempt_sha, failure_sha)


def _lineage(identity: Any) -> score.V1FailureLineage:
    return score.V1FailureLineage(
        v1_authorization=identity.phase_e_authorization, v1_identity=identity.payload(),
        failed_score_root_relative="synthetic/v1", expectation=score.V1FailureExpectation(_sha("attempt"), _sha("failure")),
        failed_score_root_identity=(1, 2),
    )


def _v2_identity() -> tuple[score.V2ScoreIdentity, Any]:
    _fixed, _training, _authorization, identity = _v1_context()
    lineage = _lineage(identity)
    closure = {"paths": ["v2"], "sha256_by_path": {"v2": _sha("v2")},
               "v1_phase_e_closure": {"synthetic": True}, "closure_sha256": _sha("v2closure")}
    auth = score.V2Authorization(
        preflight_sha256=_sha("v2pre"), root_authorization_sha256=_sha("v2auth"),
        preflight={"synthetic": True}, root_authorization={"synthetic": True}, expected_closure=closure,
    )
    return score.V2ScoreIdentity(identity.payload(), lineage.payload(), auth.payload(), closure), identity


def _within_bindings(tmp_path: Path) -> tuple[Any, ...]:
    rows = v1.sealed_within_assets()
    return tuple(v1.EvaluationAssetBinding(
        surface="within", asset_id=str(row["asset_id"]), session=str(row["session"]),
        local_path=tmp_path / Path(str(row["frozen_path"])).name,
        expected_bytes=int(row["bytes"]), expected_sha256=str(row["sha256"]), frozen_path=str(row["frozen_path"]),
    ) for row in rows)


def _external_bindings(tmp_path: Path) -> tuple[Any, ...]:
    return tuple(v1.ExternalAssetBinding(
        asset_id=f"asset-{index:02d}", session=name, local_path=tmp_path / f"{name}.nwb",
        expected_bytes=7, expected_sha256=_sha(f"asset:{name}"), frozen_path=f"sub-M/{name}.nwb",
    ) for index, name in enumerate(EXTERNAL))


def _session_score(surface: str, session: str, index: int, value: float, input_sha: str) -> Any:
    return v1.SessionScore(session=session, n_windows=100 + index, governing_r2=value, full_window_r2=value - 0.01,
                           output_sha256=_sha(f"out:{surface}:{session}:{value}"), input_authority_sha256=input_sha)


def _mode(system: str, surface: str, mode: str, input_sha: str, *, offset: float) -> Any:
    names = WITHIN if surface == "within" else EXTERNAL
    base = 0.5 if surface == "within" else 0.3
    sessions = tuple(_session_score(surface, name, index, base + index / 1000.0 + offset, input_sha)
                     for index, name in enumerate(names))
    common: dict[str, Any] = {
        "system": system, "surface": surface, "mode": mode, "sessions": sessions,
        "state_before_sha256": _sha(f"state:{system}:{surface}:{mode}"),
        "state_after_sha256": _sha(f"state:{system}:{surface}:{mode}"),
        "eval_mode": True, "gradients_none": True, "finite_output": True,
        "output_shape": (2, 50, 2), "latency_ms": 1.0, "peak_memory_bytes": 1,
    }
    if system == "tfsr":
        control: dict[str, object] = {"mode": mode, "post_normalization": True, "b3s_recomputed": True}
        if mode == "zero": control["exact_zeros_like"] = True
        if mode == "wrong_pair": control.update({"permutation_is_derangement": True, "permutation_sha256": _sha("perm")})
        common.update({"repeat_bitwise_equal": mode == "aligned", "eval_no_mask": True, "capture_diagnostics": False,
                       "b3s_recomputed": True, "t4_control": control})
    return v1.ModeEvidence(**common)


def _input_evidence(within: tuple[Any, ...], external: tuple[Any, ...]) -> Any:
    records: list[Any] = []
    tensor = {"dtype": "torch.float32", "shape": [3, 4], "bytes_sha256": _sha("tensor")}
    def add(session: str, surface: str, asset: Mapping[str, object]) -> None:
        proof = {
            "schema": "tfsr_phase_e_raw_t4_sua_axis_proof_v1", "session": session, "signal_view": "sua",
            "channel_ids_dtype": "int64", "channel_ids_sha256": _sha(f"chan:{session}"), "source_unit_count": 3,
            "raw_t4_shape": [3, 4], "feature_group": "t4", "pool_size": 30,
            "mapping": "raw_t4_row_k_equals_sua_neural_column_k_for_contiguous_channel_ids",
            "closure_bound_functions": ["mc_maze.multisession_datamodule.load_dandi688_session",
                                        "mc_maze.unit_side_features.compute_unit_side_features_uncached"],
        }
        records.append(v1.SessionInputAuthority(
            session=session, surface=surface, asset=asset, ordered_unit_digest=_sha(f"units:{session}"), unit_count=3,
            raw_t4=tensor, normalized_t4=tensor, calibration_sha256=_sha(f"cal:{session}"),
            query_start_sha256=_sha(f"start:{session}"), neural_sha256=_sha(f"neural:{session}"),
            behavior_sha256=_sha(f"beh:{session}"), valid_mask_sha256=_sha(f"mask:{session}"),
            target_last_bin_sha256=_sha(f"target:{session}"), last_bin_valid_mask_sha256=_sha(f"last:{session}"),
            last_bin_valid_count=100, n_windows=100, raw_t4_sua_axis_proof=proof,
            raw_t4_sua_axis_proof_sha256=_sha(v1._json_bytes(proof)),
        ))
    for item in within: add(item.session, "within", item.payload())
    for item in external:
        promoted = v1._as_evaluation_asset(item)
        add(promoted.session, "external", promoted.payload())
    return v1.InputAuthorityEvidence(
        records=tuple(sorted(records, key=lambda item: (item.surface, item.session))),
        source_normalizer_body_sha256=v1.SOURCE_NORMALIZER_BODY_SHA,
        t4_normalizer_semantic_sha256=v1.T4_NORMALIZER_SEMANTIC_SHA,
        behavior_normalizer_semantic_sha256=v1.BEHAVIOR_NORMALIZER_SEMANTIC_SHA,
        strict_manifest_sha256=next(item.sha256 for item in v1.FIXED_AUTHORITIES if item.name == "strict_manifest"),
        raw_to_normalized_exact=True, no_cache_readonly_adapter=True,
    )


def _sealed_table() -> dict[str, object]:
    def block(surface: str, names: tuple[str, ...]) -> dict[str, object]:
        base = 0.5 if surface == "within" else 0.3
        rows = [{"session": name, "n_windows": 100 + index, "r2": base + index / 1000.0}
                for index, name in enumerate(names)]
        return {"per_session": rows, "mean_r2": sum(row["r2"] for row in rows) / len(rows)}
    return {"results": {"D_swa": {"within": block("within", WITHIN), "external": block("external", EXTERNAL)}}}


class _MockBackend:
    def __init__(self, within: tuple[Any, ...], external: tuple[Any, ...]) -> None:
        self.within, self.external, self.closed = within, external, False
        self.calls: list[str] = []

    def resolve_inputs(self, *, flags: Any, **_: Any) -> Any:
        self.calls.append("resolve"); flags.within_opened = flags.external_opened = True
        return _input_evidence(self.within, self.external)

    def score_cell_d(self, *, surface: str, input_authority_sha256: str, flags: Any) -> Any:
        self.calls.append(f"d:{surface}"); flags.record_forward("cell_d", surface, "aligned")
        return _mode("cell_d", surface, "aligned", input_authority_sha256, offset=0.0)

    def score_tfsr(self, *, surface: str, mode: str, input_authority_sha256: str, flags: Any) -> Any:
        self.calls.append(f"t:{surface}:{mode}"); flags.record_forward("tfsr", surface, mode)
        return _mode("tfsr", surface, mode, input_authority_sha256,
                     offset=0.04 if mode == "aligned" else (-0.02 if mode == "zero" else -0.04))

    def reverify_after_forwards(self, **_: Any) -> None:
        self.calls.append("reverify")

    def resource_disclosure(self) -> Mapping[str, object]:
        return {
            "cell_d_parameters": v1.CELL_D_INITIALIZED_TRAINABLE_PARAMETERS,
            "cell_d_parameter_accounting": {
                "schema": v1.CELL_D_PARAMETER_ACCOUNTING_SCHEMA,
                "count_semantics": "initialized_trainable_parameters_only",
                "initialized_trainable_parameters": v1.CELL_D_INITIALIZED_TRAINABLE_PARAMETERS,
                "sealed_initialized_trainable_parameters": v1.CELL_D_INITIALIZED_TRAINABLE_PARAMETERS,
                "uninitialized_lazy_parameter_count": 2, "sealed_uninitialized_lazy_parameter_count": 2,
                "uninitialized_lazy_parameter_keys": list(v1.CELL_D_UNINITIALIZED_LAZY_PARAMETER_KEYS),
                "uninitialized_lazy_parameter_role": v1.CELL_D_UNINITIALIZED_LAZY_PARAMETER_ROLE,
            },
            "tfsr_parameters": 10, "tfsr_analytic_macs": 20, "persistent_state": {"per_window": "zero"},
            "training_peak_memory_bytes": 1, "score_peak_memory_bytes": 1, "aligned_latency_ms": 1.0,
            "runtime": {**v1.FROZEN_SCORE_DEVICE, "torchmetrics_version": "1.5.1"},
        }

    def close(self) -> None:
        self.closed = True


def test_static_dry_cli_is_no_torch_and_no_score_root(tmp_path: Path):
    (tmp_path / "torch.py").write_text("raise RuntimeError('TORCH_IMPORTED')\n")
    env = os.environ.copy(); env.update({"PYTHONPATH": str(tmp_path), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    run = subprocess.run([sys.executable, str(CLI)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["phase"] == score.PHASE
    assert "TORCH_IMPORTED" not in run.stdout + run.stderr
    assert not (ROOT / score.SCORE_ROOT_RELATIVE).exists()


@pytest.mark.parametrize("args", (("--execute",), ("--i-have-phase-e-root-authorization",), ("--bad",)))
def test_partial_flags_fail_before_torch(args: tuple[str, ...], tmp_path: Path):
    (tmp_path / "torch.py").write_text("raise RuntimeError('TORCH_IMPORTED')\n")
    env = os.environ.copy(); env.update({"PYTHONPATH": str(tmp_path), "PYTHONNOUSERSITE": "1"})
    run = subprocess.run([sys.executable, str(CLI), *args], cwd=ROOT, env=env, text=True, capture_output=True)
    assert run.returncode != 0
    assert "TORCH_IMPORTED" not in run.stdout + run.stderr


def test_correct_root_contract_rejects_parent_or_alias_before_any_reservation():
    with pytest.raises(score.V2Error, match="exact canonical"):
        score._correct_data_roots({"SUBC_DATA_ROOT": str(Path(score.SUBC_DATA_ROOT).parent), "SUBM_DATA_ROOT": score.SUBM_DATA_ROOT})
    assert score._correct_data_roots({"SUBC_DATA_ROOT": score.SUBC_DATA_ROOT, "SUBM_DATA_ROOT": score.SUBM_DATA_ROOT}) == (Path(score.SUBC_DATA_ROOT), Path(score.SUBM_DATA_ROOT))


def test_wrong_root_stops_before_any_v1_loader_or_v2_reservation():
    calls: list[str] = []
    with pytest.raises(score.V2Error, match="exact canonical"):
        score._prepare_authorized_execution(
            ROOT,
            environment={"SUBC_DATA_ROOT": str(Path(score.SUBC_DATA_ROOT).parent), "SUBM_DATA_ROOT": score.SUBM_DATA_ROOT},
            fixed_loader=lambda _root: calls.append("fixed") or {},
            training_loader=lambda _root: calls.append("training"),
        )
    assert calls == []


def test_v1_failed_pair_reader_requires_exact_pair_topology_schema_and_sidecars(tmp_path: Path):
    directory, fixed, training, auth, identity, expectation = _make_synthetic_v1_failed_root(tmp_path)
    # Directly test the held-FD reader with synthetic exact SHAs, then the
    # schema/identity semantics behind the production reader.
    attempt, failure, _ = score._read_exact_failed_v1_pairs(directory, attempt_sha256=expectation.attempt_sha256,
                                                              failure_sha256=expectation.failure_sha256)
    score._validate_v1_failure_semantics(v1, attempt, failure, identity)
    extra = directory / "score.json"; extra.write_text("{}"); os.chmod(extra, 0o444)
    with pytest.raises(score.V2Error, match="topology"):
        score._read_exact_failed_v1_pairs(directory, attempt_sha256=expectation.attempt_sha256,
                                           failure_sha256=expectation.failure_sha256)


def test_v1_failed_pair_tamper_or_mode_drift_fails(tmp_path: Path):
    directory, _fixed, _training, _auth, _identity, expectation = _make_synthetic_v1_failed_root(tmp_path)
    sidecar = directory / "failure.json.sha256"
    os.chmod(sidecar, 0o644); sidecar.write_text("forged\n"); os.chmod(sidecar, 0o444)
    with pytest.raises(score.V2Error, match="sidecar"):
        score._read_exact_failed_v1_pairs(directory, attempt_sha256=expectation.attempt_sha256,
                                           failure_sha256=expectation.failure_sha256)


def test_v1_lineage_loader_binds_canonical_v1_root_and_expected_identity(tmp_path: Path):
    directory, fixed, training, auth, identity, expectation = _make_synthetic_v1_failed_root(tmp_path)
    canonical = tmp_path / v1.SCORE_ROOT_RELATIVE
    canonical.parent.mkdir(parents=True)
    directory.rename(canonical)
    lineage = score.validate_v1_failed_lineage(
        tmp_path, fixed_authorities=fixed, training=training, authorization=auth, expectation=expectation,
    )
    assert lineage.v1_identity == identity.payload()
    assert lineage.failed_score_root_relative == v1.SCORE_ROOT_RELATIVE
    # A semantically valid-looking but changed V1 failure body cannot be
    # accepted merely by retaining the old expected digest.
    failure = canonical / "failure.json"
    os.chmod(failure, 0o644); failure.write_text("{}\n"); os.chmod(failure, 0o444)
    with pytest.raises(score.V2Error, match="SHA"):
        score.validate_v1_failed_lineage(
            tmp_path, fixed_authorities=fixed, training=training, authorization=auth, expectation=expectation,
        )


def test_v2_physical_backend_is_exact_v1_inheritance_seam():
    assert issubclass(score.PhysicalMatchedScoreBackendV2, v1.PhysicalMatchedScoreBackend)
    assert score.PhysicalMatchedScoreBackendV2.resolve_inputs is v1.PhysicalMatchedScoreBackend.resolve_inputs
    assert score.PhysicalMatchedScoreBackendV2.score_cell_d is v1.PhysicalMatchedScoreBackend.score_cell_d
    assert score.PhysicalMatchedScoreBackendV2.score_tfsr is v1.PhysicalMatchedScoreBackend.score_tfsr
    assert set(score.PhysicalMatchedScoreBackendV2.__dict__) <= {"__module__", "__doc__"}


def test_complete_mock_v2_lifecycle_is_atomic_and_uses_all_frozen_v1_cells(tmp_path: Path):
    identity, v1_identity = _v2_identity()
    authorization = score.V2Authorization(
        preflight_sha256=identity.v2_authorization["preflight_sha256"],
        root_authorization_sha256=identity.v2_authorization["root_authorization_sha256"],
        preflight={}, root_authorization={}, expected_closure=identity.launch_closure,
    )
    artifact = v1.reserve_artifact_root(tmp_path, "score-v2", score.SCORE_TOPOLOGY)
    within = _within_bindings(tmp_path); external = _external_bindings(tmp_path)
    backend = _MockBackend(within, external)
    terminal = score.run_score_lifecycle_v2(
        artifact=artifact, identity=identity, execution_capability=score._issue_execution_capability(authorization),
        v1_identity=v1_identity, within_roster=WITHIN, external_sessions=EXTERNAL,
        within_roster_factory=lambda: within, external_roster_factory=lambda: external,
        sealed_cell_d_table=_sealed_table(),
        a2_pooled={"within": {name: 0.4 for name in WITHIN}, "external": {name: 0.2 for name in EXTERNAL}},
        backend=backend, final_reverify=lambda: identity.launch_closure,
    )
    assert terminal["status"] == "SCORE_COMPLETE"
    assert backend.closed is True
    assert artifact.has_name("score.json") and artifact.has_name("terminal.json")
    assert not artifact.has_name("failure.json")
    assert backend.calls.count("reverify") == 2
    assert len([call for call in backend.calls if call.startswith("d:")]) == 2
    assert len([call for call in backend.calls if call.startswith("t:")]) == 6


def test_v2_lifecycle_closure_drift_after_forwards_leaves_failure_not_score(tmp_path: Path):
    identity, v1_identity = _v2_identity()
    authorization = score.V2Authorization(
        identity.v2_authorization["preflight_sha256"], identity.v2_authorization["root_authorization_sha256"],
        {}, {}, identity.launch_closure,
    )
    artifact = v1.reserve_artifact_root(tmp_path, "score-v2", score.SCORE_TOPOLOGY)
    within = _within_bindings(tmp_path); external = _external_bindings(tmp_path); backend = _MockBackend(within, external)
    with pytest.raises(score.V2Error, match="launch/final closure"):
        score.run_score_lifecycle_v2(
            artifact=artifact, identity=identity, execution_capability=score._issue_execution_capability(authorization),
            v1_identity=v1_identity, within_roster=WITHIN, external_sessions=EXTERNAL,
            within_roster_factory=lambda: within, external_roster_factory=lambda: external,
            sealed_cell_d_table=_sealed_table(),
            a2_pooled={"within": {name: 0.4 for name in WITHIN}, "external": {name: 0.2 for name in EXTERNAL}},
            backend=backend, final_reverify=lambda: {"closure_sha256": _sha("drift")},
        )
    assert artifact.has_name("attempt.json") and artifact.has_name("input_authority.json")
    assert artifact.has_name("failure.json")
    assert not artifact.has_name("score.json") and not artifact.has_name("terminal.json")


def test_v2_closure_is_explicit_and_detects_missing_or_drifted_leaf():
    closure = score.phase_e_v2_closure(ROOT)
    assert closure["paths"] == list(score.V2_CLOSURE)
    altered = json.loads(json.dumps(closure))
    altered["sha256_by_path"][score.WORKORDER_RELATIVE] = _sha("drift")
    with pytest.raises(score.V2Error, match="workorder"):
        score._validate_v2_closure(altered)


def test_v2_authority_validation_binds_predecessor_and_closure_without_data():
    identity, _v1_identity = _v2_identity()
    # This direct validator is deliberately strict even in the synthetic path:
    # changing a predecessor digest invalidates the preflight before any
    # evaluation-root factory could be invoked.
    lineage = score.V1FailureLineage(
        v1_authorization=identity.v1_identity["phase_e_authorization"], v1_identity=identity.v1_identity,
        failed_score_root_relative="synthetic/v1", expectation=score.V1FailureExpectation(_sha("a"), _sha("f")),
        failed_score_root_identity=(1, 2),
    )
    closure = score.phase_e_v2_closure(ROOT)
    # A production closure validator is intentionally not bypassed here.  The
    # first payload is exact; changing only a predecessor SHA then fails before
    # any V2 authority leaf could be published.
    preflight = {
        "schema": "tfsr_phase_e_v2_target_free_preflight_v1", "status": "PREFLIGHT_ACCEPTED", "cell": score.CELL,
        "phase": score.PHASE, "v1_score_spec": v1.PUBLIC_SPEC.payload(), "v1_failure_lineage": lineage.payload(),
        "v2_closure": closure, "authority_root_relative": score.AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": score.SCORE_ROOT_RELATIVE,
        "deployment_data_roots": {"SUBC_DATA_ROOT": score.SUBC_DATA_ROOT, "SUBM_DATA_ROOT": score.SUBM_DATA_ROOT},
        "semantic_change": "correct_canonical_data_root_binding_only", "target_free": True,
    }
    assert score.validate_target_free_preflight_v2(preflight, lineage=lineage, closure=closure)["target_free"] is True
    malformed = json.loads(json.dumps(preflight))
    malformed["v1_failure_lineage"]["v1_failed_pairs"]["failure_sha256"] = _sha("forged")
    with pytest.raises(score.V2Error, match="schema/binding"):
        score.validate_target_free_preflight_v2(malformed, lineage=lineage, closure=closure)
