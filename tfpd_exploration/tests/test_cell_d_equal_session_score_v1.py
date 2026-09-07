"""No-data/no-CUDA adversarial tests for equal-session matched scoring."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tfpd_exploration"))
from src import cell_d_equal_session_score_v1 as score


WITHIN = tuple(row[0] for row in score.SEALED_WITHIN_PAIRED_VIEW_ROWS)
EXTERNAL = tuple(f"external-{index:02d}" for index in range(score.PUBLIC_SPEC.external_sessions))


def _sha(label: str | bytes) -> str:
    return score._sha(label if isinstance(label, bytes) else label.encode("utf-8"))


def _closure() -> dict[str, object]:
    hashes = {path: _sha(f"synthetic:{path}") for path in score.IMPLEMENTATION_CLOSURE}
    immutable = {"within_paired_view_manifest": dict(score.WITHIN_PAIRED_VIEW_CLOSURE_BINDING)}
    return {
        "schema": "cell_d_equal_session_score_implementation_closure_v1",
        "paths": list(score.IMPLEMENTATION_CLOSURE),
        "sha256_by_path": hashes,
        "immutable_input_bindings": immutable,
        "closure_sha256": _sha(score._json_bytes({
            "sha256_by_path": hashes,
            "immutable_input_bindings": immutable,
        })),
    }


def _fixed_bindings() -> dict[str, dict[str, object]]:
    return {
        spec.name: {
            "relative_path": spec.relative,
            "body_sha256": spec.sha256,
            "mode": format(spec.mode, "04o"),
            "descriptor_identity": [1, 2, 3],
        }
        for spec in score.FIXED_AUTHORITIES
    }


def _identity() -> score.ScoreIdentity:
    return score.ScoreIdentity(
        fixed_authorities=_fixed_bindings(), closure=_closure(),
        baseline_receipt_sha256=score.BASELINE_RECEIPT_SHA256,
        sealed_cell_d_terminal_sha256=score.SEALED_CELL_D_TERMINAL_SHA256,
        sealed_cell_d_swa_sha256=score.SEALED_CELL_D_SWA_SHA256,
        successor_terminal_sha256=score.SUCCESSOR_TERMINAL_SHA256,
        successor_swa_sha256=score.SUCCESSOR_SWA_SHA256,
        successor_swa_state_sha256=score.SUCCESSOR_SWA_STATE_SHA256,
        preflight_sha256=_sha("preflight"), root_authorization_sha256=_sha("authorization"),
    )


def _baseline() -> score.BaselineAuthority:
    return score.BaselineAuthority(
        receipt_sha256=score.BASELINE_RECEIPT_SHA256,
        within=tuple(score.BaselineRow(name, 100 + index, 0.50 + index / 1000.0) for index, name in enumerate(WITHIN)),
        external=tuple(score.BaselineRow(name, 200 + index, 0.30 + index / 1000.0) for index, name in enumerate(EXTERNAL)),
    )


def _assets(surface: str, roster: tuple[str, ...]) -> tuple[score.PreflightAsset, ...]:
    if surface == "within":
        assert roster == WITHIN
        return score.sealed_within_assets()
    return tuple(score.PreflightAsset(
        surface=surface, asset_id=f"{surface}-asset-{index:02d}", session=name,
        frozen_path=f"{surface}/{name}.nwb", expected_bytes=10 + index,
        expected_sha256=_sha(f"{surface}:asset:{name}"),
    ) for index, name in enumerate(roster))


def _synthetic_t4_axis_proof(session: str, *, count: int = 1) -> tuple[dict[str, object], str]:
    proof: dict[str, object] = {
        "schema": "cell_d_equal_session_raw_t4_sua_axis_proof_v1",
        "session": session,
        "signal_view": "sua",
        "channel_ids_dtype": "int64",
        "channel_ids_sha256": _sha(b"\0" * (8 * count)),
        "source_unit_count": count,
        "raw_t4_shape": [count, 4],
        "feature_group": "t4",
        "pool_size": 30,
        "mapping": "raw_t4_row_k_equals_sua_neural_column_k_for_contiguous_channel_ids",
        "closure_bound_functions": [
            "mc_maze.multisession_datamodule.load_dandi688_session",
            "mc_maze.unit_side_features.compute_unit_side_features_uncached",
        ],
    }
    return proof, score._sha(score._json_bytes(proof))


def _input_evidence(within: tuple[score.PreflightAsset, ...], external: tuple[score.PreflightAsset, ...]) -> score.InputAuthorityEvidence:
    records: list[score.SessionInputAuthority] = []
    for asset in (*within, *external):
        t4_axis, t4_axis_sha = _synthetic_t4_axis_proof(asset.session)
        records.append(score.SessionInputAuthority(
            surface=asset.surface, session=asset.session, n_windows=100,
            asset=asset.payload(), ordered_unit_digest=_sha(f"units:{asset.session}"),
            neural_sha256=_sha(f"neural:{asset.session}"), behavior_sha256=_sha(f"behavior:{asset.session}"),
            query_start_sha256=_sha(f"starts:{asset.session}"),
            last_bin_target_sha256=_sha(f"last-target:{asset.session}"),
            last_bin_valid_mask_sha256=_sha(f"last-mask:{asset.session}"),
            last_bin_valid_count=100,
            normalized_t4_sha256=_sha(f"t4:{asset.session}"), raw_t4_sha256=_sha(f"raw-t4:{asset.session}"),
            raw_t4_channel_alignment=t4_axis, raw_t4_channel_alignment_sha256=t4_axis_sha,
            calibration_sha256=_sha(f"cal:{asset.session}"),
            held_descriptor_identity=(1, 2, 3),
        ))
    return score.InputAuthorityEvidence(
        records=tuple(sorted(records, key=lambda item: (item.surface, item.session))),
        source_normalizer_body_sha256=score.SOURCE_NORMALIZER_BODY_SHA256,
        t4_normalizer_semantic_sha256=score.SOURCE_T4_NORMALIZER_SHA256,
        behavior_normalizer_semantic_sha256=score.SOURCE_BEHAVIOR_NORMALIZER_SHA256,
        no_cache_readonly_adapter=True, shared_input_pass=True,
    )


def _mode(
    system: str,
    surface: str,
    baseline: score.BaselineAuthority,
    input_sha: str,
    *,
    offset: float = 0.04,
    d_offset: float = 0.0,
    changed_state: bool = False,
    dropout_disabled: bool = True,
) -> score.ModeEvidence:
    rows = baseline.rows(surface)
    values = tuple(score.SessionScore(
        session=row.session, n_windows=row.n_windows,
        governing_last_bin_r2=row.r2 + (d_offset if system == "cell_d" else offset),
        prediction_sha256=_sha(f"prediction:{system}:{surface}:{row.session}"),
        input_authority_sha256=input_sha,
    ) for row in rows)
    state = _sha(f"state:{system}")
    return score.ModeEvidence(
        system=system, surface=surface, mode="aligned_native", sessions=values,
        state_before_sha256=state, state_after_sha256=(_sha("changed") if changed_state else state),
        eval_mode=True, dropout_disabled=dropout_disabled, gradients_none=True, finite_output=True,
        output_shape=(2, 50, 2), repeat_fixed_batch_bitwise_equal=True, b3s_recomputed=True,
    )


class MockBackend:
    def __init__(
        self,
        *,
        baseline: score.BaselineAuthority,
        artifact: score.ArtifactRoot,
        within: tuple[score.PreflightAsset, ...],
        external: tuple[score.PreflightAsset, ...],
        offset: float = 0.04,
        d_offset: float = 0.0,
        fail_stage: str | None = None,
        updates: bool = False,
    ) -> None:
        self.baseline, self.artifact, self.within, self.external = baseline, artifact, within, external
        self.offset, self.d_offset, self.fail_stage, self.updates = offset, d_offset, fail_stage, updates
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    def resolve_inputs(self, *, flags: score.ScoreFlags, **_: Any) -> score.InputAuthorityEvidence:
        assert self.artifact.has_name("attempt.json"), "target resolution preceded durable attempt"
        self.calls.append(("resolve", "inputs"))
        flags.within_opened = True
        flags.external_opened = True
        if self.updates:
            flags.optimizer_calls = 1
        if self.fail_stage == "resolve":
            raise RuntimeError("synthetic resolve failure")
        return _input_evidence(self.within, self.external)

    def score(self, *, system: str, surface: str, input_authority_sha256: str, flags: score.ScoreFlags) -> score.ModeEvidence:
        self.calls.append((system, surface))
        flags.record_forward(system, surface)
        if self.fail_stage == f"{system}:{surface}":
            raise RuntimeError("synthetic score failure")
        return _mode(
            system, surface, self.baseline, input_authority_sha256,
            offset=self.offset, d_offset=self.d_offset,
        )

    def reverify_after_forwards(self, *, flags: score.ScoreFlags) -> None:
        self.calls.append(("reverify", "forwards"))
        if self.fail_stage == "reverify":
            raise RuntimeError("synthetic post-forward failure")

    def resource_disclosure(self) -> Mapping[str, object]:
        return {
            "device_contract": dict(score.FROZEN_GPU0), "forward_only": True, "no_grad": True,
            "optimizer_steps": 0, "backward_calls": 0, "update_calls": 0, "normalizer_refit": False,
            "formal_or_held_out_opened": False, "same_inputs_for_both_models": True,
            "details": {"synthetic": True},
        }

    def close(self) -> None:
        self.closed = True


def _lifecycle(tmp_path: Path, *, backend_kwargs: Mapping[str, Any] | None = None):
    parent = tmp_path / "results"
    parent.mkdir()
    artifact = score.reserve_artifact_root(parent, "attempt", topology=score.SCORE_TOPOLOGY)
    baseline = _baseline()
    identity = _identity()
    within, external = _assets("within", WITHIN), _assets("external", EXTERNAL)
    backend = MockBackend(
        baseline=baseline, artifact=artifact, within=within, external=external,
        **dict(backend_kwargs or {}),
    )
    return artifact, baseline, identity, within, external, backend


def _run(tmp_path: Path, *, backend_kwargs: Mapping[str, Any] | None = None, final: Mapping[str, object] | None = None):
    artifact, baseline, identity, within, external, backend = _lifecycle(tmp_path, backend_kwargs=backend_kwargs)
    terminal = score.run_score_lifecycle(
        artifact=artifact, identity=identity, execution_capability=score._issue_execution_capability(identity),
        baseline=baseline, within_roster=WITHIN, external_roster=EXTERNAL,
        within_assets=within, external_assets=external, backend=backend,
        final_reverify=lambda: dict(final or identity.closure),
    )
    return terminal, artifact, backend, identity, baseline


def _actual_metadata_materials() -> dict[str, score.AuthorityMaterial]:
    """Read only JSON receipt/manifest bytes; never load a checkpoint tensor."""
    materials: dict[str, score.AuthorityMaterial] = {}
    for index, spec in enumerate(score.FIXED_AUTHORITIES):
        if spec.json_required:
            body = (ROOT / spec.relative).read_bytes()
            value = json.loads(body)
        else:
            body, value = b"metadata-only-binary-not-opened", None
        materials[spec.name] = score.AuthorityMaterial(
            spec=spec, body=body, identity=(1, 10 + index, len(body)), value=value,
        )
    return materials


def test_real_metadata_only_baseline_and_lineage_are_last_bin_authority() -> None:
    materials = _actual_metadata_materials()
    lineage = score.validate_sealed_lineage(materials)
    baseline = lineage["baseline"]
    assert isinstance(baseline, score.BaselineAuthority)
    assert baseline.mean("within") == 0.5696851710478464
    assert baseline.mean("external") == 0.4179362749059995
    assert tuple(row.session for row in baseline.within) == (
        "sub-C_ses-CO-20151103", "sub-C_ses-CO-20151104", "sub-C_ses-CO-20151106",
        "sub-C_ses-CO-20151109", "sub-C_ses-CO-20151110", "sub-C_ses-CO-20151112",
    )
    assert len(lineage["external_roster"]) == 15
    assert lineage["external_roster"] == tuple(row.session for row in baseline.external)


def test_legacy_full_window_receipt_cannot_substitute_for_last_bin_authority() -> None:
    legacy = json.loads((ROOT / "tfpd_exploration/results/pop_robust_v1/matched_score_receipt.json").read_text())
    with pytest.raises(score.FailClosedError):
        score.load_last_bin_baseline(legacy)


def test_public_cli_is_static_dry_and_flag_route_fails_before_execution(tmp_path: Path) -> None:
    script = ROOT / "tfpd_exploration/scripts/run_cell_d_equal_session_seed42_score.py"
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    dry = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert "DRY_NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE" in dry.stdout
    assert not (ROOT / score.AUTHORITY_ROOT_RELATIVE).exists()
    assert not (ROOT / score.SCORE_ROOT_RELATIVE).exists()
    denied = subprocess.run(
        [sys.executable, str(script), "--execute", "--i-have-root-reviewed-equal-session-score-authorization"],
        cwd=ROOT, env=env, text=True, capture_output=True,
    )
    assert denied.returncode != 0
    assert "root-reviewed in-process execution capability required" in denied.stderr


def test_target_free_preflight_and_root_authorization_are_exact() -> None:
    baseline = _baseline()
    within, external = _assets("within", WITHIN), _assets("external", EXTERNAL)
    payload = score.build_target_free_preflight(
        root=ROOT, fixed_authorities=_fixed_bindings(), closure=_closure(), baseline=baseline,
        within_roster=WITHIN, external_roster=EXTERNAL, external_assets=external,
        normalizers=score.public_normalizer_payload(),
    )
    assert score.validate_target_free_preflight(payload, fixed_authorities=_fixed_bindings()) == payload
    authorization = score.build_root_authorization(official_preflight_sha256=_sha(score._json_bytes(payload)), preflight=payload)
    assert score.validate_root_authorization(
        authorization, official_preflight_sha256=_sha(score._json_bytes(payload)), preflight=payload,
    ) == authorization


def test_within_paired_view_manifest_is_the_only_preflight_asset_authority() -> None:
    baseline = _baseline()
    external = _assets("external", EXTERNAL)
    payload = score.build_target_free_preflight(
        root=ROOT, fixed_authorities=_fixed_bindings(), closure=_closure(), baseline=baseline,
        within_roster=WITHIN, external_roster=EXTERNAL, external_assets=external,
        normalizers=score.public_normalizer_payload(),
    )
    assert payload["within_assets"] == [asset.payload() for asset in score.sealed_within_assets()]
    binding = payload["within_paired_view_manifest"]
    assert binding["relative_path"] == score.WITHIN_PAIRED_VIEW_MANIFEST_RELATIVE
    assert binding["body_sha256"] == score.WITHIN_PAIRED_VIEW_MANIFEST_SHA256
    assert binding["mode"] == "0600"
    assert binding["read_once"] is True
    assert tuple(binding["descriptor_identity"])[2] == 9_888
    assert score.validate_target_free_preflight(payload, fixed_authorities=_fixed_bindings()) == payload

    forged_row = json.loads(json.dumps(payload))
    forged_row["within_assets"][0]["sha256"] = _sha("other-within-file")
    with pytest.raises(score.FailClosedError, match="within"):
        score.validate_target_free_preflight(forged_row, fixed_authorities=_fixed_bindings())
    forged_manifest = json.loads(json.dumps(payload))
    forged_manifest["within_paired_view_manifest"]["body_sha256"] = _sha("other-manifest")
    with pytest.raises(score.FailClosedError, match="manifest"):
        score.validate_target_free_preflight(forged_manifest, fixed_authorities=_fixed_bindings())


def test_preflight_builder_has_no_caller_supplied_within_asset_escape_hatch() -> None:
    baseline = _baseline()
    with pytest.raises(TypeError):
        score.build_target_free_preflight(
            root=ROOT, fixed_authorities=_fixed_bindings(), closure=_closure(), baseline=baseline,
            within_roster=WITHIN, external_roster=EXTERNAL,
            within_assets=_assets("within", WITHIN),  # type: ignore[call-arg]
            external_assets=_assets("external", EXTERNAL), normalizers=score.public_normalizer_payload(),
        )


@pytest.mark.parametrize("mutator", [
    lambda payload: payload["normalizers"].__setitem__("behavior_semantic_sha256", _sha("wrong")),
    lambda payload: payload["external_roster"].reverse(),
    lambda payload: payload["metric_contract"].__setitem__("query", "full_window"),
    lambda payload: payload["predecessor_successor"].__setitem__("successor_swa_sha256", _sha("wrong-swa")),
])
def test_preflight_rejects_normalizer_roster_metric_and_lineage_drift(mutator: Any) -> None:
    baseline = _baseline()
    within, external = _assets("within", WITHIN), _assets("external", EXTERNAL)
    payload = score.build_target_free_preflight(
        root=ROOT, fixed_authorities=_fixed_bindings(), closure=_closure(), baseline=baseline,
        within_roster=WITHIN, external_roster=EXTERNAL, external_assets=external,
        normalizers=score.public_normalizer_payload(),
    )
    mutator(payload)
    with pytest.raises(score.FailClosedError):
        score.validate_target_free_preflight(payload, fixed_authorities=_fixed_bindings())


def test_root_authorization_rejects_wrong_preflight_sha() -> None:
    baseline = _baseline()
    within, external = _assets("within", WITHIN), _assets("external", EXTERNAL)
    preflight = score.build_target_free_preflight(
        root=ROOT, fixed_authorities=_fixed_bindings(), closure=_closure(), baseline=baseline,
        within_roster=WITHIN, external_roster=EXTERNAL, external_assets=external,
        normalizers=score.public_normalizer_payload(),
    )
    auth = score.build_root_authorization(official_preflight_sha256=_sha("correct"), preflight=preflight)
    with pytest.raises(score.FailClosedError):
        score.validate_root_authorization(auth, official_preflight_sha256=_sha("other"), preflight=preflight)


def test_lifecycle_success_is_cell_d_parity_blocked_and_transactional(tmp_path: Path) -> None:
    terminal, artifact, backend, identity, baseline = _run(tmp_path)
    assert terminal["status"] == "SCORE_COMPLETE"
    assert terminal["verdict"] == "CLEAR_GO"
    assert artifact.has_name("score.json") and artifact.has_name("terminal.json")
    assert not artifact.has_name("failure.json")
    assert backend.closed is True
    persisted = artifact.reload_json("score.json")
    score.validate_score_payload(
        persisted, identity=identity, baseline=baseline,
        input_authority_sha256=terminal["input_authority_sha256"],
    )
    assert backend.calls[0] == ("resolve", "inputs")
    assert backend.calls[1:3] == [("cell_d", "within"), ("cell_d", "external")]


def test_lifecycle_requires_attempt_before_input_resolution(tmp_path: Path) -> None:
    _terminal, artifact, backend, _identity_value, _baseline_value = _run(tmp_path)
    assert backend.calls[0] == ("resolve", "inputs")
    assert artifact.has_name("attempt.json")


def test_cell_d_baseline_parity_failure_blocks_successor_and_writes_failure(tmp_path: Path) -> None:
    artifact, baseline, identity, within, external, backend = _lifecycle(tmp_path, backend_kwargs={"d_offset": 0.001})
    with pytest.raises(score.FailClosedError, match="parity"):
        score.run_score_lifecycle(
            artifact=artifact, identity=identity, execution_capability=score._issue_execution_capability(identity),
            baseline=baseline, within_roster=WITHIN, external_roster=EXTERNAL,
            within_assets=within, external_assets=external, backend=backend, final_reverify=lambda: identity.closure,
        )
    assert all(system != "equal_session" for system, _surface in backend.calls)
    assert artifact.has_name("failure.json")
    assert not artifact.has_name("score.json") and not artifact.has_name("terminal.json")


@pytest.mark.parametrize(
    ("external_mean", "within_mean", "external_median", "positive", "expected"),
    [
        (0.03, -0.03, 0.001, 9, "CLEAR_GO"),
        (0.029999999, 0.0, 0.01, 15, "HOLD"),
        (-0.0000001, 0.2, 0.2, 15, "STOP"),
        (0.2, -0.0300001, 0.2, 15, "STOP"),
        (0.04, 0.0, 0.0, 15, "HOLD"),
        (0.04, 0.0, 0.2, 8, "HOLD"),
    ],
)
def test_verdict_boundaries(
    external_mean: float, within_mean: float, external_median: float, positive: int, expected: str,
) -> None:
    external = {"mean": external_mean, "median": external_median, "n_positive": positive, "n_total": 15}
    within = {"mean": within_mean, "median": 0.0, "n_positive": 0, "n_total": 6}
    assert score.decide_verdict(external=external, within=within) == expected


def test_state_mutation_dropout_or_gradient_operations_are_rejected() -> None:
    baseline = _baseline()
    with pytest.raises(ValueError, match="state changed"):
        _mode("cell_d", "within", baseline, _sha("input"), changed_state=True)
    with pytest.raises(ValueError, match="eval/dropout"):
        _mode("cell_d", "within", baseline, _sha("input"), dropout_disabled=False)
    flags = score.ScoreFlags(optimizer_calls=1)
    with pytest.raises(score.FailClosedError, match="forbidden"):
        score.build_score_payload(
            identity=_identity(), baseline=baseline, input_authority_sha256=_sha("input"),
            cell_d=(_mode("cell_d", "within", baseline, _sha("input")), _mode("cell_d", "external", baseline, _sha("input"))),
            equal_session=(_mode("equal_session", "within", baseline, _sha("input")), _mode("equal_session", "external", baseline, _sha("input"))),
            resources=MockBackend(baseline=baseline, artifact=None, within=(), external=()).resource_disclosure(), flags=flags,
        )


def test_lifecycle_update_or_resolve_failure_has_no_partial_scientific_score(tmp_path: Path) -> None:
    artifact, baseline, identity, within, external, backend = _lifecycle(tmp_path, backend_kwargs={"updates": True})
    with pytest.raises(score.FailClosedError):
        score.run_score_lifecycle(
            artifact=artifact, identity=identity, execution_capability=score._issue_execution_capability(identity),
            baseline=baseline, within_roster=WITHIN, external_roster=EXTERNAL,
            within_assets=within, external_assets=external, backend=backend, final_reverify=lambda: identity.closure,
        )
    assert artifact.has_name("failure.json")
    assert not artifact.has_name("score.json") and not artifact.has_name("terminal.json")


def test_group_publication_rolls_back_score_and_terminal_together(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    artifact = score.reserve_artifact_root(parent, "group", topology=score.SCORE_TOPOLOGY)
    with pytest.raises(RuntimeError):
        artifact.publish_group(
            {"score.json": b"score", "terminal.json": b"terminal"},
            post_publish=lambda _bodies, _hashes: (_ for _ in ()).throw(RuntimeError("inject post-publish failure")),
        )
    assert not artifact.has_name("score.json")
    assert not artifact.has_name("terminal.json")


def test_post_score_closure_drift_rolls_back_scientific_score_and_records_failure(tmp_path: Path) -> None:
    artifact, baseline, identity, within, external, backend = _lifecycle(tmp_path)
    bad_closure = _closure()
    bad_closure["closure_sha256"] = _sha("drift")
    with pytest.raises(score.FailClosedError, match="closure"):
        score.run_score_lifecycle(
            artifact=artifact, identity=identity, execution_capability=score._issue_execution_capability(identity),
            baseline=baseline, within_roster=WITHIN, external_roster=EXTERNAL,
            within_assets=within, external_assets=external, backend=backend, final_reverify=lambda: bad_closure,
        )
    assert artifact.has_name("failure.json")
    assert not artifact.has_name("score.json") and not artifact.has_name("terminal.json")


def test_score_validator_rejects_wrong_estimator_roster_and_legacy_control(tmp_path: Path) -> None:
    terminal, artifact, _backend, identity, baseline = _run(tmp_path)
    payload = artifact.reload_json("score.json")
    for mutation in (
        lambda item: item["metric_contract"].__setitem__("estimator", "flattened_r2"),
        lambda item: item["cell_d"]["within"]["sessions"].reverse(),
        lambda item: item["controls"].__setitem__("full_window_governing", True),
    ):
        altered = json.loads(json.dumps(payload))
        mutation(altered)
        with pytest.raises(score.FailClosedError):
            score.validate_score_payload(
                altered, identity=identity, baseline=baseline,
                input_authority_sha256=terminal["input_authority_sha256"],
            )


def test_artifact_collision_fails_before_any_leaf(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    artifact = score.reserve_artifact_root(parent, "collision", topology=score.SCORE_TOPOLOGY)
    artifact.publish_json("attempt.json", {"a": 1})
    with pytest.raises(score.FailClosedError, match="collision"):
        artifact.publish_group({"attempt.json": b"again", "score.json": b"score"})
    assert not artifact.has_name("score.json")


def test_public_execute_capability_cannot_be_forged_by_flags() -> None:
    identity = _identity()
    with pytest.raises(score.FailClosedError):
        score._require_execution_capability(None, identity)
    fake = score.ExecutionCapability(identity_digest=_sha("wrong"), _seal=object())
    with pytest.raises(score.FailClosedError):
        score._require_execution_capability(fake, identity)


def _physical_gpu0_attestation() -> dict[str, object]:
    return {
        **score.FROZEN_GPU0,
        "torchmetrics_version": score.TORCHMETRICS_VERSION,
        "no_user_site": True,
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("uuid", "GPU-wrong"),
        ("bdf", "00000000:02:00.0"),
        ("nvidia_smi_memory_total_mib", 24_256),
        ("torch_total_memory_bytes", 24_576 * 1024 * 1024),
        ("torch_version", "2.5.1"),
        ("torch_cuda_version", "12.0"),
        ("cudnn_version", 90_301),
        ("torchmetrics_version", "1.5.0"),
    ],
)
def test_physical_gpu0_attestation_binds_each_authority_exactly(field: str, replacement: object) -> None:
    exact = _physical_gpu0_attestation()
    assert score._validate_physical_gpu0_attestation(exact) == exact
    altered = dict(exact)
    altered[field] = replacement
    with pytest.raises(score.FailClosedError):
        score._validate_physical_gpu0_attestation(altered)


def test_raw_t4_sua_axis_and_last_bin_target_evidence_are_exact_no_data() -> None:
    """Exercise only synthetic arrays, never a target parser or model."""
    import numpy as np
    from types import SimpleNamespace

    asset = score.sealed_within_assets()[0]
    raw = np.arange(12, dtype=np.float32).reshape(3, 4)
    metadata = SimpleNamespace(feature_group="t4", pool_size=30)
    proof_payload = score._raw_t4_sua_axis_proof_payload(
        np, session=asset.session, neural_unit_count=3,
        channel_ids=np.asarray([0, 1, 2], dtype=np.int64), source_unit_count=3,
        raw_t4=raw, metadata=metadata,
    )
    proof = score._sha(score._json_bytes(proof_payload))
    assert len(proof) == 64
    with pytest.raises(score.FailClosedError, match="channel-order"):
        score._prove_t4_row_order_against_sua_channels(
            np, session=asset.session, neural_unit_count=3,
            channel_ids=np.asarray([1, 0, 2], dtype=np.int64), source_unit_count=3,
            raw_t4=raw, metadata=metadata,
        )
    with pytest.raises(score.FailClosedError, match="channel-order"):
        score._prove_t4_row_order_against_sua_channels(
            np, session=asset.session, neural_unit_count=3,
            channel_ids=np.asarray([0, 1, 2], dtype=np.int64), source_unit_count=3,
            raw_t4=raw, metadata=SimpleNamespace(feature_group="z4", pool_size=30),
        )

    behavior = np.arange(120, dtype=np.float32).reshape(60, 2)
    starts = np.asarray([0, 2, 4], dtype=np.int64)
    targets, mask, target_sha, mask_sha, count = score._last_bin_authority_from_behavior(
        np, behavior=behavior, starts=starts,
    )
    authority = score.SessionInputAuthority(
        surface="within", session=asset.session, n_windows=3, asset=asset.payload(),
        ordered_unit_digest=_sha("units"), neural_sha256=_sha("neural"), behavior_sha256=_sha("behavior"),
        query_start_sha256=_sha("starts"), last_bin_target_sha256=target_sha,
        last_bin_valid_mask_sha256=mask_sha, last_bin_valid_count=count,
        normalized_t4_sha256=_sha("norm-t4"), raw_t4_sha256=_sha("raw-t4"),
        raw_t4_channel_alignment=proof_payload, raw_t4_channel_alignment_sha256=proof,
        calibration_sha256=_sha("cal"),
        held_descriptor_identity=(1, 2, 3),
    )
    score._validate_last_bin_authority_arrays(np, target=targets, valid_mask=mask, authority=authority)
    shifted = targets.copy()
    shifted[0, 0] += 1.0
    with pytest.raises(score.FailClosedError, match="last-bin"):
        score._validate_last_bin_authority_arrays(np, target=shifted, valid_mask=mask, authority=authority)
    with pytest.raises(ValueError, match="window/asset"):
        replace(authority, last_bin_valid_count=2)
    forged_proof = dict(proof_payload)
    forged_proof["pool_size"] = 31
    with pytest.raises(ValueError, match="raw T4/SUA"):
        replace(authority, raw_t4_channel_alignment=forged_proof)


def test_input_authority_payload_requires_per_session_last_bin_and_t4_axis_digests() -> None:
    evidence = _input_evidence(_assets("within", WITHIN), _assets("external", EXTERNAL))
    payload = evidence.payload()
    assert score.input_authority_from_payload(payload) == evidence
    for field in (
        "last_bin_target_sha256", "last_bin_valid_mask_sha256", "last_bin_valid_count",
        "raw_t4_channel_alignment_sha256",
    ):
        altered = json.loads(json.dumps(payload))
        del altered["records"][0][field]
        with pytest.raises(score.FailClosedError):
            score.input_authority_from_payload(altered)


def test_physical_held_descriptor_and_private_snapshot_are_no_cache_and_reverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise only a synthetic regular file—never an NWB or a target root."""
    data_root = tmp_path / "synthetic-within"
    data_root.mkdir()
    body = b"synthetic-no-nwb-payload"
    name = "session-0.nwb"
    local = data_root / name
    local.write_bytes(body)
    os.chmod(local, 0o400)
    monkeypatch.setenv("SYNTHETIC_WITHIN_ROOT", str(data_root))
    root_capability = score._ScoreDataRoot.from_environment(
        variable="SYNTHETIC_WITHIN_ROOT", label="synthetic-within",
    )
    asset = score.PreflightAsset(
        surface="within", asset_id="synthetic-asset", session="synthetic-session",
        frozen_path=f"reviewed/subdir/{name}", expected_bytes=len(body), expected_sha256=_sha(body),
    )
    binding = score._ScoreAssetBinding.from_preflight(asset, root_capability)
    assert binding.local_path == local
    held = score._hold_verified_score_asset(binding)
    snapshot = score._private_verified_score_snapshot(held)
    try:
        assert snapshot.path.name == name
        assert snapshot.path.read_bytes() == body
        snapshot.reverify()
        held.reverify()
        assert held.identity[2] == len(body)
    finally:
        snapshot.close()
        held.close()
    assert not snapshot.path.exists()


def test_physical_held_asset_rejects_symlink_and_backend_construction_is_inert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "synthetic-root"
    data_root.mkdir()
    target = tmp_path / "target"
    target.write_bytes(b"body")
    os.chmod(target, 0o400)
    link = data_root / "link.nwb"
    link.symlink_to(target)
    monkeypatch.setenv("SYNTHETIC_ROOT", str(data_root))
    root_capability = score._ScoreDataRoot.from_environment(variable="SYNTHETIC_ROOT", label="synthetic")
    asset = score.PreflightAsset(
        surface="within", asset_id="link", session="synthetic-session", frozen_path="x/link.nwb",
        expected_bytes=4, expected_sha256=_sha(b"body"),
    )
    with pytest.raises(score.FailClosedError, match="regular non-symlink"):
        score._hold_verified_score_asset(score._ScoreAssetBinding.from_preflight(asset, root_capability))

    materials = {
        spec.name: score.AuthorityMaterial(
            spec=spec, body=b"opaque-not-opened", identity=(1, index + 1, 17), value=None,
        )
        for index, spec in enumerate(score.FIXED_AUTHORITIES)
    }
    backend = score.PhysicalMatchedScoreBackend(root=ROOT, fixed_authorities=materials)
    assert backend._runtime is None
    assert backend._models == {}
    backend.close()
    assert backend._closed is True
    assert not (ROOT / score.AUTHORITY_ROOT_RELATIVE).exists()
    assert not (ROOT / score.SCORE_ROOT_RELATIVE).exists()


def test_physical_ordered_unit_digest_and_normalizer_contract_are_not_semantic_placeholders() -> None:
    np = pytest.importorskip("numpy")
    digest = score._physical_ordered_unit_digest(("s:channel:0", "s:channel:1"))
    assert digest != score._physical_ordered_unit_digest(("s:channel:1", "s:channel:0"))
    with pytest.raises(score.FailClosedError):
        score._physical_ordered_unit_digest(("same", "same"))
    t4_mean, t4_std, behavior_mean, behavior_std = score._validate_source_normalizer_numerics(np)
    assert t4_mean.dtype == np.float32 and t4_std.dtype == np.float32
    assert behavior_mean.dtype == np.float32 and behavior_std.dtype == np.float32
    assert np.array_equal(behavior_mean, np.asarray(score.SOURCE_BEHAVIOR_MEAN, dtype=np.float32))


@pytest.mark.parametrize(
    ("authority_name", "mutate"),
    [
        (
            "sealed_cell_d_terminal",
            lambda value: value["swa"].__setitem__("sha256", _sha("wrong-d-swa")),
        ),
        (
            "successor_terminal",
            lambda value: value["terminal_detail"].__setitem__("swa_state_sha256", _sha("wrong-successor-state")),
        ),
        (
            "successor_terminal",
            lambda value: value["run_spec"].__setitem__("checkpoint_epochs", [43, 44, 45, 46]),
        ),
    ],
)
def test_real_metadata_lineage_rejects_predecessor_successor_swa_or_state_drift(
    authority_name: str, mutate: Any,
) -> None:
    materials = _actual_metadata_materials()
    material = materials[authority_name]
    assert material.value is not None
    altered_value = json.loads(json.dumps(material.value))
    mutate(altered_value)
    materials[authority_name] = score.AuthorityMaterial(
        spec=material.spec, body=material.body, identity=material.identity, value=altered_value,
    )
    with pytest.raises(score.FailClosedError):
        score.validate_sealed_lineage(materials)


def test_physical_score_invariants_reject_gradient_eval_and_formal_boundary_drift() -> None:
    baseline = _baseline()
    evidence_payload = _mode("cell_d", "within", baseline, _sha("input")).payload()
    for field in ("eval_mode", "gradients_none", "finite_output", "repeat_fixed_batch_bitwise_equal", "b3s_recomputed"):
        altered = dict(evidence_payload)
        altered[field] = False
        with pytest.raises(score.FailClosedError):
            score.mode_evidence_from_payload(altered)
    flags = score.ScoreFlags(formal_opened=True)
    with pytest.raises(score.FailClosedError, match="forbidden"):
        score._require_no_updates(flags)


def test_no_capability_stops_before_fixed_authority_or_backend_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("public no-capability route touched an authority/backend")

    monkeypatch.setattr(score, "verify_fixed_authorities", _unexpected)
    with pytest.raises(score.FailClosedError, match="in-process execution capability"):
        score.execute_authorized(ROOT, capability=None, backend=None)


def test_actual_explicit_closure_is_non_glob_and_future_roots_remain_fresh() -> None:
    closure = score.implementation_closure(ROOT)
    assert closure["paths"] == list(score.IMPLEMENTATION_CLOSURE)
    assert len(closure["paths"]) == len(set(closure["paths"]))
    assert all("*" not in path and "?" not in path for path in closure["paths"])
    assert closure["immutable_input_bindings"] == {
        "within_paired_view_manifest": score.WITHIN_PAIRED_VIEW_CLOSURE_BINDING,
    }
    assert closure["closure_sha256"] == score._sha(score._json_bytes({
        "sha256_by_path": closure["sha256_by_path"],
        "immutable_input_bindings": closure["immutable_input_bindings"],
    }))
    assert not (ROOT / score.AUTHORITY_ROOT_RELATIVE).exists()
    assert not (ROOT / score.SCORE_ROOT_RELATIVE).exists()
