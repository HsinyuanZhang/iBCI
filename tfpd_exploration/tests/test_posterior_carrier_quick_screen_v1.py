"""Focused no-data/no-CUDA tests for the Posterior Carrier quick screen."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path
import subprocess
import sys
import tempfile
import importlib
import types
from typing import Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tfpd_exploration/src/posterior_carrier_quick_screen_v1/quick_screen.py"
STAGE_SOURCE = ROOT / "tfpd_exploration/src/posterior_carrier_quick_screen_v1/remote_stage.py"
PHYSICAL_SOURCE = ROOT / "tfpd_exploration/src/posterior_carrier_quick_screen_v1/physical.py"
CLI = ROOT / "tfpd_exploration/scripts/run_posterior_carrier_quick_screen.py"


if str(ROOT / "tfpd_exploration") not in sys.path:
    sys.path.insert(0, str(ROOT / "tfpd_exploration"))

quick = importlib.import_module("src.posterior_carrier_quick_screen_v1.quick_screen")
stage = importlib.import_module("src.posterior_carrier_quick_screen_v1.remote_stage")


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _synthetic_upstream_formal_device_contract() -> dict[str, object]:
    """Schema-valid upstream provenance, deliberately distinct from 5070 Ti."""
    return {
        "cuda_visible_devices": "0", "cuda_device_order": "PCI_BUS_ID", "logical_device": "cuda:0",
        "uuid": "GPU-synthetic-formal", "bdf": "00000000:01:00.0", "name": "NVIDIA Formal Authority",
        "nvidia_smi_memory_total_mib": 24_576, "torch_total_memory_bytes": 25_435_111_424,
        "torch_version": "2.5.1.post303", "torch_cuda_version": "11.8", "cudnn_version": 90_300,
    }


def _identity() -> object:
    hashes = {relative: _hash(relative) for relative in quick.IMPLEMENTATION_CLOSURE}
    hashes[quick.WORKORDER_RELATIVE] = quick.WORKORDER_SHA256
    return quick.QuickScreenIdentity(closure=quick.ImplementationClosure(hashes))


def _input(identity: object) -> object:
    records = []
    for surface in quick.SURFACES:
        for index, session in enumerate(quick.validate_identity(identity)["selected_rosters"][surface]):
            records.append(quick.SessionInput(
                surface=surface, session=session, n_windows=4 + index,
                neural_sha256=_hash(f"{surface}:{session}:neural"),
                calibration_m30_sha256=_hash(f"{surface}:{session}:calibration"),
                last_bin_target_sha256=_hash(f"{surface}:{session}:target"),
                last_bin_valid_mask_sha256=_hash(f"{surface}:{session}:mask"),
                last_bin_valid_count=4 + index,
                prefix_row_ids_sha256s={"30": _hash(f"{session}:rows30"), "4": _hash(f"{session}:rows4")},
                point_carrier_sha256s={"30": _hash(f"{session}:point30"), "4": _hash(f"{session}:point4")},
                posterior_carrier_sha256s={"30": _hash(f"{session}:post30"), "4": _hash(f"{session}:post4")},
                materialization_sha256=_hash(f"{surface}:{session}:materialized"),
            ))
    return quick.InputAuthority(records=tuple(records))


class _FakeBackend:
    def __init__(self, identity: object, *, fail_at: tuple[str, str, int] | None = None) -> None:
        self.identity = identity
        self.authority = _input(identity)
        self.fail_at = fail_at
        self.closed = False

    def prepare(self, *, identity: object, flags: object) -> None:
        assert identity is self.identity
        flags.remote_initialized = True

    def resolve_inputs(self, *, identity: object, flags: object) -> object:
        assert identity is self.identity
        flags.within_opened = True
        flags.external_opened = True
        return self.authority

    def score_cell(self, *, cell: object, input_payload: object, flags: object) -> object:
        if self.fail_at == (cell.surface, cell.mode, cell.budget):
            raise RuntimeError("synthetic selected forward failure")
        selected = quick.validate_identity(self.identity)["selected_rosters"][cell.surface]
        records = {row["session"]: row for row in input_payload["records"] if row["surface"] == cell.surface}
        sessions = tuple(
            quick.SessionScore(
                session=session, n_windows=records[session]["n_windows"],
                r2=(0.1 * (index + 1) + (0.03 if cell.mode == quick.POSTERIOR_MODE else 0.0) - (0.02 if cell.budget == 4 else 0.0)),
                prediction_sha256=_hash(f"{cell.surface}:{cell.mode}:{cell.budget}:{session}:prediction"),
                input_record_sha256=quick._digest(quick._json(records[session])),
            )
            for index, session in enumerate(selected)
        )
        return quick.CellEvidence(
            cell=cell,
            model_system="sealed_cell_d_checkpoint" if cell.mode == quick.SEALED_POINT_MODE else "posterior_carrier_full_swa",
            model_swa_sha256=quick.SEALED_CELL_D_SWA_SHA256 if cell.mode == quick.SEALED_POINT_MODE else quick.POSTERIOR_FULL_SWA_SHA256,
            sessions=sessions, input_authority_sha256=quick._digest(quick._json(input_payload)),
            model_state_before_sha256=_hash(f"state:{cell.mode}"), model_state_after_sha256=_hash(f"state:{cell.mode}"),
            eval_mode=True, dropout_disabled=True, gradients_none=True, finite_outputs=True,
            repeated_fixed_batch_bitwise_equal=True, b3s_m30_recomputed=True, no_target_sampling=True,
        )

    def reverify_after_forwards(self, *, identity: object, flags: object) -> object:
        return identity.closure

    def device_attestation(self) -> object:
        return quick.remote_engineering_device_payload()

    def upstream_formal_preflight_device(self) -> object:
        return quick.upstream_formal_preflight_device_payload(_synthetic_upstream_formal_device_contract())

    def posterior_final4_compatibility(self) -> object:
        return quick.posterior_final4_compatibility_payload()

    def close(self) -> None:
        self.closed = True


def _reserve(tmp_path: Path) -> object:
    results = tmp_path / "tfpd_exploration/results"
    results.mkdir(parents=True, exist_ok=True)
    return quick.reserve_result_artifact(tmp_path)


def test_frozen_selection_and_transfer_bytes_are_exact() -> None:
    payload = quick.selected_asset_payload()
    selected = quick.selected_assets_from_payload(payload)
    assert tuple(item.session for item in selected[quick.WITHIN]) == (
        "sub-C_ses-CO-20151103", "sub-C_ses-CO-20151106", "sub-C_ses-CO-20151112",
    )
    assert tuple(item.session for item in selected[quick.EXTERNAL]) == (
        "sub-M_ses-CO-20140307", "sub-M_ses-CO-20150611", "sub-M_ses-CO-20150626",
    )
    assert quick.selected_transfer_bytes() == 392_880_348


def test_selection_rejects_byte_sha_and_index_substitution() -> None:
    payload = quick.selected_asset_payload()
    payload[quick.EXTERNAL][1]["sha256"] = _hash("forged")
    with pytest.raises(quick.QuickScreenError, match="roster index|literal/order"):
        quick.selected_assets_from_payload(payload)
    payload = quick.selected_asset_payload()
    payload[quick.WITHIN][0]["full_roster_index"] = 1
    with pytest.raises(quick.QuickScreenError, match="roster index|literal/order"):
        quick.selected_assets_from_payload(payload)


def test_full_authority_selection_requires_full_rosters_not_three_row_surrogate() -> None:
    payload = quick.selected_asset_payload()
    with pytest.raises(quick.QuickScreenError, match="cardinality"):
        quick.select_from_full_authority_rows(payload)
    full = {quick.WITHIN: [{"asset_id": "x", "session": "x", "frozen_path": "x.nwb", "bytes": 1, "sha256": _hash("x")}] * 6,
            quick.EXTERNAL: [{"asset_id": "y", "session": "y", "frozen_path": "y.nwb", "bytes": 1, "sha256": _hash("y")}] * 15}
    for surface, chosen in payload.items():
        for row in chosen:
            full[surface][row["full_roster_index"]] = dict(row)
    result = quick.select_from_full_authority_rows(full)
    assert result == payload


def test_real_metadata_only_derivation_matches_frozen_three_plus_three_rows() -> None:
    """Read only C1/v2 metadata; no NWB pathname or tensor is opened."""
    derived = stage.derive_selected_assets_from_existing_authorities(ROOT)
    assert derived == quick.selected_asset_payload()


def test_stage_requires_descriptor_derived_c1_and_v2_selection_not_a_caller_three_row_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviewed = quick.selected_asset_payload()
    assert stage._require_descriptor_derived_selection(ROOT, reviewed_selection=reviewed) == reviewed
    # Make the derivation return a different but schema-shaped selection,
    # proving the final comparison is a real authority check rather than just
    # a caller-table schema check.
    other_valid = quick.selected_asset_payload()
    other_valid[quick.EXTERNAL] = list(reversed(other_valid[quick.EXTERNAL]))
    monkeypatch.setattr(stage, "derive_selected_assets_from_existing_authorities", lambda _root: other_valid)
    with pytest.raises(stage.StageError, match="descriptor-derived C1/v2"):
        stage._require_descriptor_derived_selection(ROOT, reviewed_selection=reviewed)


def test_matrix_is_exactly_two_surfaces_by_four_predeclared_cells() -> None:
    assert [item.payload() for item in quick.quick_screen_matrix()] == [
        {"surface": quick.WITHIN, "mode": quick.SEALED_POINT_MODE, "budget": 30},
        {"surface": quick.WITHIN, "mode": quick.POSTERIOR_MODE, "budget": 30},
        {"surface": quick.WITHIN, "mode": quick.SEALED_POINT_MODE, "budget": 4},
        {"surface": quick.WITHIN, "mode": quick.POSTERIOR_MODE, "budget": 4},
        {"surface": quick.EXTERNAL, "mode": quick.SEALED_POINT_MODE, "budget": 30},
        {"surface": quick.EXTERNAL, "mode": quick.POSTERIOR_MODE, "budget": 30},
        {"surface": quick.EXTERNAL, "mode": quick.SEALED_POINT_MODE, "budget": 4},
        {"surface": quick.EXTERNAL, "mode": quick.POSTERIOR_MODE, "budget": 4},
    ]


def test_success_lifecycle_publishes_atomic_non_governing_score_and_terminal(tmp_path: Path) -> None:
    identity = _identity()
    artifact = _reserve(tmp_path)
    backend = _FakeBackend(identity)
    terminal = quick.run_quick_screen_lifecycle(
        artifact=artifact, identity=identity, execution_capability=quick._issue_root_review_capability(identity), backend=backend,
    )
    assert terminal["classification"] == quick.CLASSIFICATION
    assert terminal["no_governing_decision"] is True
    assert not artifact.has_name("failure.json")
    score = artifact.reload_json("score.json", terminal["score_sha256"])
    assert score["status"] == "NON_GOVERNING_QUICK_SCREEN_COMPLETE__NO_GOVERNING_VERDICT"
    assert score["paired_posterior_minus_sealed_point"][quick.EXTERNAL]["posterior_minus_point_m4"]["n_sessions"] == 3
    assert score["device_attestation"] == quick.remote_engineering_device_payload()
    assert score["upstream_formal_preflight_device"] == quick.upstream_formal_preflight_device_payload(
        _synthetic_upstream_formal_device_contract(),
    )
    assert score["posterior_final4_compatibility"] == quick.posterior_final4_compatibility_payload()
    assert backend.closed is True


def test_failure_lifecycle_leaves_failure_not_partial_scientific_score(tmp_path: Path) -> None:
    identity = _identity()
    artifact = _reserve(tmp_path)
    backend = _FakeBackend(identity, fail_at=(quick.WITHIN, quick.POSTERIOR_MODE, 30))
    with pytest.raises(RuntimeError, match="synthetic selected forward failure"):
        quick.run_quick_screen_lifecycle(
            artifact=artifact, identity=identity, execution_capability=quick._issue_root_review_capability(identity), backend=backend,
        )
    assert artifact.has_name("attempt.json") and artifact.has_name("input_authority.json") and artifact.has_name("failure.json")
    assert not artifact.has_name("score.json") and not artifact.has_name("terminal.json")
    failure = artifact.reload_json("failure.json", hashlib.sha256((artifact.directory / "failure.json").read_bytes()).hexdigest())
    quick.validate_failure_payload(failure, identity=identity)


def test_output_collision_and_public_capability_forgery_fail_before_execution(tmp_path: Path) -> None:
    identity = _identity()
    _reserve(tmp_path)
    with pytest.raises(quick.QuickScreenError, match="fresh"):
        _reserve(tmp_path)
    forged = quick.QuickScreenExecutionCapability("0" * 64, "0" * 64, "0" * 64, object())
    with pytest.raises(quick.QuickScreenError, match="capability"):
        quick.execute_authorized(root=tmp_path, identity=identity, execute=True, root_reviewed=True, execution_capability=forged)


def test_score_validation_rejects_wrong_estimator_evidence_and_hidden_governing_label(tmp_path: Path) -> None:
    identity = _identity()
    artifact = _reserve(tmp_path)
    backend = _FakeBackend(identity)
    terminal = quick.run_quick_screen_lifecycle(
        artifact=artifact, identity=identity, execution_capability=quick._issue_root_review_capability(identity), backend=backend,
    )
    score = artifact.reload_json("score.json", terminal["score_sha256"])
    score["metric"]["query"] = "full_window"
    with pytest.raises(quick.QuickScreenError, match="metric"):
        quick.validate_score_payload(score, identity=identity, input_payload=artifact.reload_json(
            "input_authority.json", hashlib.sha256((artifact.directory / "input_authority.json").read_bytes()).hexdigest(),
        ))
    score = artifact.reload_json("score.json", terminal["score_sha256"])
    score["no_governing_decision"] = False
    with pytest.raises(quick.QuickScreenError, match="non-governing"):
        quick.validate_score_payload(score, identity=identity, input_payload=artifact.reload_json(
            "input_authority.json", hashlib.sha256((artifact.directory / "input_authority.json").read_bytes()).hexdigest(),
        ))
    score = artifact.reload_json("score.json", terminal["score_sha256"])
    score["posterior_final4_compatibility"]["final4_exact_recompute"] = "PASS__FORGED"
    with pytest.raises(quick.QuickScreenError, match="final-four compatibility"):
        quick.validate_score_payload(score, identity=identity, input_payload=artifact.reload_json(
            "input_authority.json", hashlib.sha256((artifact.directory / "input_authority.json").read_bytes()).hexdigest(),
        ))


def test_regular_source_reader_rejects_mode_swap_and_body_swap(tmp_path: Path) -> None:
    path = tmp_path / "source.bin"
    path.write_bytes(b"exact")
    os.chmod(path, 0o444)
    digest = hashlib.sha256(b"exact").hexdigest()
    assert stage._read_regular_exact(path, byte_count=5, sha256=digest, mode=0o444) == b"exact"
    os.chmod(path, 0o600)
    with pytest.raises(stage.StageError, match="mode"):
        stage._read_regular_exact(path, byte_count=5, sha256=digest, mode=0o444)
    os.chmod(path, 0o444)
    path.chmod(0o644)
    path.write_bytes(b"wrong")
    os.chmod(path, 0o444)
    with pytest.raises(stage.StageError, match="SHA"):
        stage._read_regular_exact(path, byte_count=5, sha256=digest, mode=0o444)


def test_stage_pair_requires_reviewed_predecessor_digest_not_only_a_self_consistent_sidecar(tmp_path: Path) -> None:
    relative = "immutable/terminal.json"
    body_path = tmp_path / relative
    body_path.parent.mkdir(parents=True)
    original = b'{"status":"original"}'
    original_sha = hashlib.sha256(original).hexdigest()
    body_path.write_bytes(original)
    sidecar = tmp_path / f"{relative}.sha256"
    sidecar.write_bytes(f"{original_sha}  terminal.json\n".encode("ascii"))
    os.chmod(body_path, 0o444)
    os.chmod(sidecar, 0o444)
    pair = stage._pair_files(
        tmp_path, relative, role="synthetic_predecessor", expected_body_sha256=original_sha,
    )
    assert pair[0].sha256 == original_sha

    # A body plus matching sidecar can be internally consistent while still
    # being the wrong predecessor.  The plan must reject it before transfer.
    substituted = b'{"status":"substituted"}'
    substituted_sha = hashlib.sha256(substituted).hexdigest()
    os.chmod(body_path, 0o644)
    os.chmod(sidecar, 0o644)
    body_path.write_bytes(substituted)
    sidecar.write_bytes(f"{substituted_sha}  terminal.json\n".encode("ascii"))
    os.chmod(body_path, 0o444)
    os.chmod(sidecar, 0o444)
    with pytest.raises(stage.StageError, match="reviewed predecessor SHA"):
        stage._pair_files(
            tmp_path, relative, role="synthetic_predecessor", expected_body_sha256=original_sha,
        )


def test_stage_pair_sidecars_use_canonical_basenames_not_full_nested_paths(tmp_path: Path) -> None:
    relative = "nested/result/attempt.json"
    body_path = tmp_path / relative
    body_path.parent.mkdir(parents=True)
    body = b'{"attempt":true}'
    digest = hashlib.sha256(body).hexdigest()
    body_path.write_bytes(body)
    sidecar = tmp_path / f"{relative}.sha256"
    # A full relative destination is not the repository's immutable sidecar
    # grammar.  It must not be silently accepted during remote staging.
    sidecar.write_bytes(f"{digest}  {relative}\n".encode("ascii"))
    os.chmod(body_path, 0o444)
    os.chmod(sidecar, 0o444)
    with pytest.raises(stage.StageError, match="body/sidecar/mode"):
        stage._pair_files(tmp_path, relative, role="posterior_full_mirror", expected_body_sha256=digest)
    os.chmod(sidecar, 0o644)
    sidecar.write_bytes(f"{digest}  attempt.json\n".encode("ascii"))
    os.chmod(sidecar, 0o444)
    pair = stage._pair_files(tmp_path, relative, role="posterior_full_mirror", expected_body_sha256=digest)
    assert pair[0].destination_relative == relative


def test_real_immutable_metadata_pairs_use_basename_sidecars() -> None:
    """Read only JSON receipt bytes, never a checkpoint tensor or NWB body."""
    pairs = (
        (quick.BASE_OFFICIAL_PREFLIGHT_RELATIVE, quick.BASE_OFFICIAL_PREFLIGHT_SHA256, "base_authority_evidence"),
        (f"{quick.FULL_MIRROR_ROOT}/attempt.json", None, "posterior_full_mirror"),
        (quick.SEALED_CELL_D_TERMINAL_RELATIVE, quick.SEALED_CELL_D_TERMINAL_SHA256, "sealed_cell_d_evidence"),
    )
    for relative, expected, role in pairs:
        body, sidecar = stage._pair_files(ROOT, relative, role=role, expected_body_sha256=expected)
        assert body.destination_relative == relative
        assert sidecar.destination_relative == f"{relative}.sha256"
        assert body.source.name == Path(relative).name


def test_stage_source_mode_admission_accepts_real_0664_closure_and_nwb_but_rejects_unreviewed_modes(tmp_path: Path) -> None:
    # Metadata-only lstat: no NWB body is opened.  Both observed source leaf
    # classes are 0664 and must stage to fresh 0444 destinations.
    real_nwb = ROOT / "sua_exploration/data/dandi_000688/sub-C/sub-C_ses-CO-20151103_behavior+ecephys.nwb"
    for source, role, destination in (
        (SOURCE, "code_or_metadata", "tfpd_exploration/src/posterior_carrier_quick_screen_v1/quick_screen.py"),
        (real_nwb, "evaluation_nwb", "evaluation_assets/sub-C/sub-C_ses-CO-20151103_behavior+ecephys.nwb"),
    ):
        info = os.lstat(source)
        assert stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode)
        assert stat.S_IMODE(info.st_mode) == 0o664
        payload = stage.StageFile(
            source, destination, int(info.st_size), _hash(str(source)), role, stat.S_IMODE(info.st_mode),
        ).payload()
        assert payload["source_mode"] == "0664" and payload["destination_mode"] == "0444"

    leaf = tmp_path / "mode.bin"
    leaf.write_bytes(b"x")
    for mode in (0o666, 0o777, 0o640):
        os.chmod(leaf, mode)
        with pytest.raises(stage.StageError, match="source mode"):
            stage.StageFile(leaf, "leaf.bin", 1, _hash(f"mode-{mode}"), "code_or_metadata", mode).payload()
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(stage.StageError, match="regular"):
        stage._regular_identity(directory)
    symlink = tmp_path / "linked.bin"
    symlink.symlink_to(leaf)
    with pytest.raises(stage.StageError, match="regular"):
        stage._regular_identity(symlink)


def test_stage_plan_requires_exact_six_nwb_total_and_no_source_training_nwb(tmp_path: Path) -> None:
    identity = _identity()
    file = tmp_path / "leaf"
    file.write_bytes(b"x")
    os.chmod(file, 0o444)
    plan = stage.RemoteStagePlan(
        identity=quick.validate_identity(identity), stage_root=quick.REMOTE_STAGE_ROOT,
        score_root_relative=quick.REMOTE_SCORE_ROOT_RELATIVE,
        files=(stage.StageFile(file, "evaluation_assets/sub-C/x.nwb", 1, _hash("x"), "evaluation_nwb", 0o444),),
    )
    with pytest.raises(stage.StageError, match="exactly six"):
        plan.payload()
    forbidden = stage.RemoteStagePlan(
        identity=quick.validate_identity(identity), stage_root=quick.REMOTE_STAGE_ROOT,
        score_root_relative=quick.REMOTE_SCORE_ROOT_RELATIVE,
        files=tuple(
            stage.StageFile(file, f"evaluation_assets/sub-C/{index}.nwb", byte_count, digest, "source_training_nwb", 0o444)
            for index, (_surface, _asset, _session, _path, byte_count, digest) in enumerate(
                item for surface in quick.SURFACES for item in quick._SELECTED_LITERALS[surface]
            )
        ),
    )
    with pytest.raises(stage.StageError, match="source-training"):
        forbidden.payload()


class _FakeTransport:
    def __init__(self) -> None:
        self.begun = False
        self.writes: list[str] = []
        self.aborted = False

    def begin_fresh_stage(self, *, stage_root: str, manifest: object) -> None:
        self.begun = True
        self.stage_root = stage_root
        self.manifest = manifest

    def write_regular(self, *, destination_relative: str, body: bytes, sha256: str, mode: int) -> None:
        # A production transport receives only held-FD verified bytes.  This
        # fake intentionally accepts synthetic placeholder bodies.
        assert mode == 0o444
        self.writes.append(destination_relative)

    def finalize_stage(self, *, expected_manifest_sha256: str) -> object:
        return {"stage_root": self.stage_root, "manifest_sha256": expected_manifest_sha256}

    def abort_owned_stage(self) -> None:
        self.aborted = True


def test_stage_capability_is_opaque_and_forged_plan_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    identity = _identity()
    # Use literal evaluation sizes but mock the held-FD reader so this is a
    # transport lifecycle test, not a 393-MB local data test.
    leaves = []
    for surface in quick.SURFACES:
        for index, (_global, _asset, _session, _path, byte_count, digest) in enumerate(quick._SELECTED_LITERALS[surface]):
            path = tmp_path / f"{surface}-{index}.nwb"
            path.write_bytes(b"x")
            os.chmod(path, 0o444)
            subject = "sub-C" if surface == quick.WITHIN else "sub-M"
            frozen_path = quick._SELECTED_LITERALS[surface][index][3]
            leaves.append(stage.StageFile(
                path, f"evaluation_assets/{subject}/{Path(frozen_path).name}", byte_count, digest, "evaluation_nwb", 0o444,
            ))
    plan = stage.RemoteStagePlan(quick.validate_identity(identity), quick.REMOTE_STAGE_ROOT, quick.REMOTE_SCORE_ROOT_RELATIVE, tuple(leaves))
    # ``stage_reviewed_plan`` gets its verified bodies only through this helper;
    # the patch makes the staged lifecycle synthetic while preserving topology.
    monkeypatch.setattr(stage, "_read_regular_exact", lambda _path, *, byte_count, sha256, mode: b"x")
    transport = _FakeTransport()
    with pytest.raises(stage.StageError, match="capability"):
        stage.stage_reviewed_plan(plan=plan, capability=object(), transport=transport)
    result = stage.stage_reviewed_plan(plan=plan, capability=stage._issue_root_review_stage_capability(plan), transport=transport)
    assert result["stage_root"] == quick.REMOTE_STAGE_ROOT
    assert len(transport.writes) == 6 and transport.aborted is False
    assert transport.manifest["fresh_result_parent_relative"] == "tfpd_exploration/results"
    assert transport.manifest["fresh_result_root_relative"] == quick.REMOTE_SCORE_ROOT_RELATIVE
    assert transport.manifest["result_root_created_by_stage"] is False


def test_remote_engineering_device_is_explicitly_not_local_authority() -> None:
    physical = importlib.import_module("src.posterior_carrier_quick_screen_v1.physical")
    device = physical.RemoteEngineeringDevice()
    assert device.payload() == quick.remote_engineering_device_payload()
    assert device.payload()["current_local_authority"] is False
    assert device.payload()["nvml_status"] == "UNAVAILABLE_DRIVER_LIBRARY_MISMATCH"
    assert device.payload()["uuid"] is None and device.payload()["bdf"] is None
    assert device.payload()["nvidia_smi_memory_total_mib"] is None
    with pytest.raises(physical.PhysicalQuickScreenError, match="engineering"):
        physical.RemoteEngineeringDevice(current_local_authority=True).payload()


def _write_immutable_pair(directory: Path, name: str, payload: dict[str, object]) -> str:
    """Create a test-only 0444 canonical pair under one temporary root."""
    body = quick._json(payload)
    digest = hashlib.sha256(body).hexdigest()
    body_path = directory / name
    sidecar_path = directory / f"{name}.sha256"
    body_path.write_bytes(body)
    sidecar_path.write_bytes(f"{digest}  {name}\n".encode("ascii"))
    os.chmod(body_path, 0o444)
    os.chmod(sidecar_path, 0o444)
    return digest


def _synthetic_failed_predecessor(directory: Path, *, version: int) -> dict[str, object]:
    """Schema-faithful test fixture for a held-FD V1/V2 predecessor reader."""
    if version not in (1, 2):
        raise AssertionError("synthetic predecessor version")
    phase = f"POSTERIOR_CARRIER_QUICK_SCREEN_V{version}"
    attempt = {
        "schema": f"posterior_carrier_quick_screen_attempt_v{version}",
        "status": "ATTEMPT_RESERVED_BEFORE_EVALUATION_PATH_RESOLUTION",
        "classification": quick.CLASSIFICATION,
        "cell": quick.CELL,
        "phase": phase,
        "identity": {f"v{version}": "synthetic"},
        "boundaries": {"target_optimizer_steps": 0},
        "evaluation_assets_resolved": False,
        "remote_initialized": False,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
    }
    attempt_sha = _write_immutable_pair(directory, "attempt.json", attempt)
    error_sha = _hash(f"v{version} synthetic physical quick screen error")
    failure = {
        "schema": f"posterior_carrier_quick_screen_failure_v{version}",
        "classification": quick.CLASSIFICATION,
        "cell": quick.CELL,
        "phase": phase,
        "identity": attempt["identity"],
        "stage": "prepare",
        "attempt_sha256": attempt_sha,
        "input_authority_sha256": None,
        "flags": {
            "within_opened": False, "external_opened": False, "remote_initialized": False,
            "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
            "normalizer_refit": False, "target_sampling": False, "h1_opened": False,
            "formal_opened": False, "forward_cells": [],
        },
        "terminal_published": False,
        "error_class": "PhysicalQuickScreenError",
        "error_sha256": error_sha,
        "traceback_sha256": _hash(f"v{version} synthetic traceback"),
    }
    failure_sha = _write_immutable_pair(directory, "failure.json", failure)
    literal = quick.V1FailedPredecessorEvidence() if version == 1 else quick.V2FailedPredecessorEvidence()
    evidence = json.loads(json.dumps(literal.payload()))
    evidence["attempt"]["sha256"] = attempt_sha
    evidence["attempt"]["sidecar"] = f"{attempt_sha}  attempt.json\n"
    evidence["failure"]["sha256"] = failure_sha
    evidence["failure"]["sidecar"] = f"{failure_sha}  failure.json\n"
    evidence["failure"]["error_sha256"] = error_sha
    return evidence


def _synthetic_v1_failed_predecessor(directory: Path) -> dict[str, object]:
    return _synthetic_failed_predecessor(directory, version=1)


def _synthetic_v2_failed_predecessor(directory: Path) -> dict[str, object]:
    return _synthetic_failed_predecessor(directory, version=2)


def test_v3_identity_and_stage_bind_immutable_v1_v2_failures_without_reusing_prior_outputs(tmp_path: Path) -> None:
    identity = _identity()
    payload = quick.validate_identity(identity)
    assert payload["phase"] == "POSTERIOR_CARRIER_QUICK_SCREEN_V3"
    assert payload["v1_failed_predecessor"] == quick.V1FailedPredecessorEvidence().payload()
    assert payload["v2_failed_predecessor"] == quick.V2FailedPredecessorEvidence().payload()
    assert payload["v1_failed_predecessor"]["attempt"]["sidecar"].encode("ascii") == (
        b"7620c853d2b4ff5c2aae359859956bc94bca9c48740294103c453674bd0a8e23  attempt.json\n"
    )
    assert payload["v1_failed_predecessor"]["failure"]["sidecar"].encode("ascii") == (
        b"275111598568502cabc7c68350644b2599108c1930cdb622057c27dd7a5cf965  failure.json\n"
    )
    assert len(payload["v1_failed_predecessor"]["attempt"]["sidecar"].encode("ascii")) == 79
    assert len(payload["v1_failed_predecessor"]["failure"]["sidecar"].encode("ascii")) == 79
    assert payload["v2_failed_predecessor"]["attempt"]["sidecar"].encode("ascii") == (
        b"11c586cc6bf837cf21c4972acd797018ebb40cf48b2be12c0619490f45147383  attempt.json\n"
    )
    assert payload["v2_failed_predecessor"]["failure"]["sidecar"].encode("ascii") == (
        b"d06a6f9afc5ba0015010a9cedf1e96ab872944c6851334a6b67308c851a348be  failure.json\n"
    )
    assert quick.REMOTE_STAGE_ROOT.endswith("_v3")
    assert quick.REMOTE_SCORE_ROOT_RELATIVE.endswith("_v3")
    assert quick.REMOTE_STAGE_ROOT != quick.V1_REMOTE_STAGE_ROOT
    assert quick.REMOTE_SCORE_ROOT_RELATIVE != quick.V1_REMOTE_SCORE_ROOT_RELATIVE
    assert quick.REMOTE_STAGE_ROOT != quick.V2_REMOTE_STAGE_ROOT
    assert quick.REMOTE_SCORE_ROOT_RELATIVE != quick.V2_REMOTE_SCORE_ROOT_RELATIVE

    # Prior leaves do not block the fresh V3 reservation and are not
    # overwritten; only the V3 name is created in this temporary result parent.
    results = tmp_path / "tfpd_exploration/results"
    old_v1 = results / Path(quick.V1_REMOTE_SCORE_ROOT_RELATIVE).name
    old_v2 = results / Path(quick.V2_REMOTE_SCORE_ROOT_RELATIVE).name
    old_v1.mkdir(parents=True)
    old_v2.mkdir(parents=True)
    sentinel_v1 = old_v1 / "failure.json"
    sentinel_v2 = old_v2 / "failure.json"
    sentinel_v1.write_text("immutable-v1", encoding="utf-8")
    sentinel_v2.write_text("immutable-v2", encoding="utf-8")
    artifact = quick.reserve_result_artifact(tmp_path)
    assert artifact.directory.name == Path(quick.REMOTE_SCORE_ROOT_RELATIVE).name
    assert sentinel_v1.read_text(encoding="utf-8") == "immutable-v1"
    assert sentinel_v2.read_text(encoding="utf-8") == "immutable-v2"

    tampered = json.loads(json.dumps(payload))
    tampered["v2_failed_predecessor"]["failure"]["sha256"] = _hash("forged-v2-failure")
    plan = stage.RemoteStagePlan(tampered, quick.REMOTE_STAGE_ROOT, quick.REMOTE_SCORE_ROOT_RELATIVE, ())
    with pytest.raises(stage.StageError, match="V2 failed-predecessor"):
        plan.payload()


def test_held_fd_v1_v2_failure_validation_rejects_extra_topology_before_v3_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    physical = importlib.import_module("src.posterior_carrier_quick_screen_v1.physical")
    base_physical = _real_base_physical_module(physical)
    predecessor_root = tmp_path / "v1"
    predecessor_root.mkdir()
    evidence = _synthetic_v1_failed_predecessor(predecessor_root)
    assert physical._validate_failed_predecessor_directory_with_expected(
        physical=base_physical, directory=predecessor_root, evidence=evidence,
    ) == evidence
    extra = predecessor_root / "input_authority.json"
    extra.write_bytes(b"{}")
    os.chmod(extra, 0o444)
    with pytest.raises(physical.PhysicalQuickScreenError, match="topology"):
        physical._validate_failed_predecessor_directory_with_expected(
            physical=base_physical, directory=predecessor_root, evidence=evidence,
        )

    predecessor_v2_root = tmp_path / "v2"
    predecessor_v2_root.mkdir()
    evidence_v2 = _synthetic_v2_failed_predecessor(predecessor_v2_root)
    assert physical._validate_failed_predecessor_directory_with_expected(
        physical=base_physical, directory=predecessor_v2_root, evidence=evidence_v2,
    ) == evidence_v2

    # The production executor must inspect V1 then V2 before it reserves any
    # V3 root or accepts a V3 capability.
    identity = _identity()
    reserved = False
    capability_checked = False
    observed: list[str] = []

    def pass_v1(**_kwargs: object) -> None:
        observed.append("v1")

    def fail_v2(**_kwargs: object) -> None:
        observed.append("v2")
        raise physical.PhysicalQuickScreenError("synthetic V2 predecessor drift")

    def fail_reserve(_root: Path) -> object:
        nonlocal reserved
        reserved = True
        raise AssertionError("V3 reserve must not occur after V2 failure")

    def fail_capability(**_kwargs: object) -> object:
        nonlocal capability_checked
        capability_checked = True
        raise AssertionError("V3 capability must not be accepted before V1/V2 validation")

    monkeypatch.setattr(physical.QuickScreenPhysicalBackend, "_load_base", lambda _self: (object(), object()))
    monkeypatch.setattr(physical, "validate_v1_failed_predecessor_directory", pass_v1)
    monkeypatch.setattr(physical, "validate_v2_failed_predecessor_directory", fail_v2)
    monkeypatch.setattr(quick, "reserve_result_artifact", fail_reserve)
    monkeypatch.setattr(quick, "_require_capability", fail_capability)
    with pytest.raises(physical.PhysicalQuickScreenError, match="V2 predecessor"):
        physical.execute_reviewed_remote(
            root=tmp_path, identity=identity, engineering_device=physical.RemoteEngineeringDevice(),
            execution_capability=quick._issue_root_review_capability(identity),
        )
    assert reserved is False
    assert capability_checked is False
    assert observed == ["v1", "v2"]


def _fake_torch_only_runtime(
    *,
    torch_version: str = "2.13.0+cu130",
    cuda_version: str = "13.0",
    cudnn_version: int = 92_000,
    name: str = "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
    total_memory_bytes: int = 12_346_195_968,
    capability: tuple[int, int] = (12, 0),
    visible_count: int = 1,
    current_device: int = 0,
    available: bool = True,
) -> object:
    class Properties:
        pass

    properties = Properties()
    properties.name = name
    properties.total_memory = total_memory_bytes

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return available

        @staticmethod
        def device_count() -> int:
            return visible_count

        @staticmethod
        def current_device() -> int:
            return current_device

        @staticmethod
        def get_device_properties(index: int) -> object:
            assert index == 0
            return properties

        @staticmethod
        def get_device_capability(index: int) -> tuple[int, int]:
            assert index == 0
            return capability

    class CuDnn:
        @staticmethod
        def version() -> int:
            return cudnn_version

    return types.SimpleNamespace(
        __version__=torch_version,
        version=types.SimpleNamespace(cuda=cuda_version),
        cuda=Cuda,
        backends=types.SimpleNamespace(cudnn=CuDnn),
    )


def test_manual_variance_weighted_r2_matches_local_1_5_helper_on_random_fixture() -> None:
    """CPU-only regression: the V3 route keeps the frozen metric value."""
    import torch
    import torchmetrics
    from torchmetrics.regression import R2Score

    physical = importlib.import_module("src.posterior_carrier_quick_screen_v1.physical")
    helper = importlib.import_module("src.tfpd_lane.matched_scorer")
    assert str(torchmetrics.__version__) == "1.5.1"
    generator = torch.Generator(device="cpu").manual_seed(42)
    targets = torch.randn((257, 2), generator=generator, dtype=torch.float32)
    predictions = 0.7 * targets + 0.3 * torch.randn((257, 2), generator=generator, dtype=torch.float32)
    manual = physical.manual_variance_weighted_two_coordinate_r2(
        predictions=predictions, targets=targets, torch=torch,
    )
    metric = R2Score(multioutput="variance_weighted")
    metric.update(predictions, targets)
    observed = float(metric.compute())
    frozen_helper = helper.session_r2(predictions, targets)
    assert abs(manual - observed) <= 1e-7
    assert abs(manual - frozen_helper) <= 1e-7
    assert torch.cuda.is_initialized() is False


def test_remote_parity_gate_uses_torchmetrics_1_9_constructor_without_num_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 1.9-shaped constructor rejects legacy kwargs before any input path."""
    import torch

    physical = importlib.import_module("src.posterior_carrier_quick_screen_v1.physical")
    seen: list[dict[str, object]] = []

    class R2Score19:
        def __init__(self, *, multioutput: str) -> None:
            seen.append({"multioutput": multioutput})
            self.predictions = None
            self.targets = None

        def update(self, predictions: object, targets: object) -> None:
            self.predictions, self.targets = predictions, targets

        def compute(self) -> object:
            assert self.predictions is not None and self.targets is not None
            value = physical.manual_variance_weighted_two_coordinate_r2(
                predictions=self.predictions, targets=self.targets, torch=torch,
            )
            return torch.tensor(value, dtype=torch.float32)

    fake_root = types.ModuleType("torchmetrics")
    fake_regression = types.ModuleType("torchmetrics.regression")
    fake_regression.R2Score = R2Score19
    fake_root.regression = fake_regression
    monkeypatch.setitem(sys.modules, "torchmetrics", fake_root)
    monkeypatch.setitem(sys.modules, "torchmetrics.regression", fake_regression)
    base_physical = _real_base_physical_module(physical)
    physical._assert_variance_weighted_r2_parity(
        torch=torch,
        torchmetrics=types.SimpleNamespace(__version__="1.9.0"),
        physical=base_physical,
    )
    assert seen == [{"multioutput": "variance_weighted"}]
    assert torch.cuda.is_initialized() is False


def test_quick_physical_scoring_route_does_not_call_legacy_shared_r2_helper() -> None:
    source = PHYSICAL_SOURCE.read_text(encoding="utf-8")
    assert 'runtime["metric"].session_r2' not in source
    assert "manual_variance_weighted_two_coordinate_r2(" in source


def _real_base_physical_module(physical: object) -> object:
    return physical._load_exact_module(
        ROOT,
        "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
        "_quick_screen_real_base_physical_for_no_cuda_test",
    )


def test_authoritative_full_train_fp32_final4_rule_differs_from_fp64_and_is_bitwise_strict() -> None:
    """The V3 repair must reproduce producer math, not merely accept a state."""
    import torch

    physical = importlib.import_module("src.posterior_carrier_quick_screen_v1.physical")
    base_physical = _real_base_physical_module(physical)
    values = (100_000_000.0, 1.0, -100_000_000.0, 1.0)
    checkpoints = {
        epoch: {"model_state": {"weight": torch.tensor([value], dtype=torch.float32)}}
        for epoch, value in zip((44, 45, 46, 47), values)
    }
    authoritative = physical.authoritative_full_train_fp32_final4_state(
        checkpoints=checkpoints, torch=torch, physical=base_physical,
    )
    fp64 = sum(
        checkpoints[epoch]["model_state"]["weight"].double()
        for epoch in (44, 45, 46, 47)
    ).div(4).to(torch.float32)
    assert torch.equal(authoritative["weight"], torch.tensor([0.25], dtype=torch.float32))
    assert torch.equal(fp64, torch.tensor([0.5], dtype=torch.float32))
    assert not torch.equal(authoritative["weight"], fp64)
    assert torch.cuda.is_initialized() is False


def test_v3_swa_override_accepts_only_known_base_fp64_mismatch_then_requires_fp32_bitwise_state() -> None:
    """No arbitrary base error may be relabelled as a quick-SWA compatibility."""
    import torch

    physical = importlib.import_module("src.posterior_carrier_quick_screen_v1.physical")

    class FakePhysicalScoreError(RuntimeError):
        pass

    class FakeArmCommon:
        @staticmethod
        def state_sha256(state: object) -> str:
            assert isinstance(state, Mapping)
            return quick.POSTERIOR_FULL_SWA_STATE_SHA256

    class Parent:
        def _validate_swa_against_checkpoints(self, **_kwargs: object) -> object:
            raise FakePhysicalScoreError(self._base_error)

        def _load_runtime(self) -> object:
            return {"torch": torch, "arm_common": FakeArmCommon}

        def _safe_weights_only_load(self, _body: bytes, *, label: str) -> object:
            assert label.startswith("full final-four SWA V3")
            return self._payload

        @staticmethod
        def _state_equal(_torch: object, left: Mapping[str, object], right: Mapping[str, object]) -> bool:
            return set(left) == set(right) and all(torch.equal(left[key], right[key]) for key in left)

        @staticmethod
        def _fresh_posterior_wrapper(state: Mapping[str, object], *, label: str, place_on_device: bool) -> object:
            assert label == "full SWA V3 authoritative FP32" and place_on_device is False
            return state

    fake_physical = types.SimpleNamespace(
        PhysicalScoreError=FakePhysicalScoreError,
        PhysicalPosteriorMatchedBackend=Parent,
    )
    cls = physical.build_torch_only_engineering_backend_class(
        physical=fake_physical,
        device=physical.RemoteEngineeringDevice(),
    )
    checkpoints = {
        epoch: {"model_state": {"weight": torch.tensor([value], dtype=torch.float32)}}
        for epoch, value in zip((44, 45, 46, 47), (100_000_000.0, 1.0, -100_000_000.0, 1.0))
    }
    expected = physical.authoritative_full_train_fp32_final4_state(
        checkpoints=checkpoints, torch=torch, physical=fake_physical,
    )
    instance = object.__new__(cls)
    instance._base_error = physical._FROZEN_BASE_FP64_FINAL4_MISMATCH
    instance._payload = {"state": expected, "state_sha256": quick.POSTERIOR_FULL_SWA_STATE_SHA256}
    result = cls._validate_swa_against_checkpoints(
        instance,
        swa_body=b"synthetic", checkpoints=checkpoints, terminal={},
        expected_swa_state_sha256=quick.POSTERIOR_FULL_SWA_STATE_SHA256,
    )
    assert result is instance._payload
    assert instance._quick_screen_posterior_final4_compatibility == quick.posterior_final4_compatibility_payload()

    with pytest.raises(FakePhysicalScoreError, match="state-digest authority drift"):
        cls._validate_swa_against_checkpoints(
            instance,
            swa_body=b"synthetic", checkpoints=checkpoints, terminal={},
            expected_swa_state_sha256="0" * 64,
        )

    # One changed ULP cannot pass merely because the known base FP64 check did
    # not reach the strict-load/digest phase.
    altered = expected["weight"].clone()
    altered[0] = torch.nextafter(altered[0], torch.tensor(float("inf"), dtype=torch.float32))
    instance._payload = {"state": {"weight": altered}, "state_sha256": quick.POSTERIOR_FULL_SWA_STATE_SHA256}
    with pytest.raises(FakePhysicalScoreError, match="authoritative full-train FP32"):
        cls._validate_swa_against_checkpoints(
            instance,
            swa_body=b"synthetic", checkpoints=checkpoints, terminal={},
            expected_swa_state_sha256=quick.POSTERIOR_FULL_SWA_STATE_SHA256,
        )

    # Any different base failure (schema/proof/terminal/checkpoint binding) is
    # re-raised unchanged rather than becoming an optional compatibility path.
    instance._base_error = "full SWA persisted proof binding drift"
    instance._payload = {"state": expected, "state_sha256": quick.POSTERIOR_FULL_SWA_STATE_SHA256}
    with pytest.raises(FakePhysicalScoreError, match="persisted proof binding drift"):
        cls._validate_swa_against_checkpoints(
            instance,
            swa_body=b"synthetic", checkpoints=checkpoints, terminal={},
            expected_swa_state_sha256=quick.POSTERIOR_FULL_SWA_STATE_SHA256,
        )
    assert torch.cuda.is_initialized() is False


def test_torch_only_engineering_subclass_overrides_only_attestation_and_authoritative_swa_and_never_calls_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physical = importlib.import_module("src.posterior_carrier_quick_screen_v1.physical")
    base_physical = _real_base_physical_module(physical)
    cls = physical.build_torch_only_engineering_backend_class(
        physical=base_physical,
        device=physical.RemoteEngineeringDevice(),
    )
    assert cls.__bases__ == (base_physical.PhysicalPosteriorMatchedBackend,)
    assert cls._runtime_device_attestation is not base_physical.PhysicalPosteriorMatchedBackend._runtime_device_attestation
    assert cls._validate_swa_against_checkpoints is not base_physical.PhysicalPosteriorMatchedBackend._validate_swa_against_checkpoints
    # Parsing, strict model loading, carrier computation, batching, forward,
    # scoring, and sealed Cell-D validation remain inherited by object
    # identity—no monkeypatch.  Posterior SWA is the one additional V3 seam.
    for name in (
        "prepare", "_parse_one", "_session_batch", "_forward", "score_cell", "_ensure_models", "_load_runtime",
        "_validate_checkpoint_state", "_fresh_base_cell_d",
    ):
        assert getattr(cls, name) is getattr(base_physical.PhysicalPosteriorMatchedBackend, name)

    parity_calls: list[object] = []
    monkeypatch.setattr(physical, "_assert_variance_weighted_r2_parity", lambda **kwargs: parity_calls.append(kwargs))
    import subprocess as standard_subprocess
    monkeypatch.setattr(
        standard_subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("nvidia-smi/subprocess must not be called")),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    instance = object.__new__(cls)
    # This mapping is formal upstream provenance; the subclass must not
    # rewrite it into an unavailable NVML-shaped remote contract.
    instance._preflight = {"device_contract": _synthetic_upstream_formal_device_contract()}
    observed = cls._runtime_device_attestation(
        instance,
        torch=_fake_torch_only_runtime(),
        torchmetrics=types.SimpleNamespace(__version__="1.9.0"),
    )
    assert observed == quick.remote_engineering_device_payload()
    assert len(parity_calls) == 1


@pytest.mark.parametrize(
    "mutator",
    (
        lambda kwargs, env: kwargs.update(torch_version="2.13.0+cu131"),
        lambda kwargs, env: kwargs.update(cuda_version="13.1"),
        lambda kwargs, env: kwargs.update(cudnn_version=92_001),
        lambda kwargs, env: kwargs.update(name="NVIDIA GeForce RTX 5070 Laptop GPU"),
        lambda kwargs, env: kwargs.update(total_memory_bytes=12_346_195_969),
        lambda kwargs, env: kwargs.update(capability=(12, 1)),
        lambda kwargs, env: kwargs.update(visible_count=2),
        lambda kwargs, env: env.update(CUDA_VISIBLE_DEVICES="1"),
        lambda kwargs, env: env.update(CUDA_DEVICE_ORDER="FASTEST_FIRST"),
        lambda kwargs, env: env.update(torchmetrics_version="1.9.1"),
    ),
)
def test_torch_only_engineering_authority_drifts_fail_before_any_nvidia_path(
    monkeypatch: pytest.MonkeyPatch, mutator: object,
) -> None:
    physical = importlib.import_module("src.posterior_carrier_quick_screen_v1.physical")
    base_physical = _real_base_physical_module(physical)
    cls = physical.build_torch_only_engineering_backend_class(
        physical=base_physical,
        device=physical.RemoteEngineeringDevice(),
    )
    monkeypatch.setattr(physical, "_assert_variance_weighted_r2_parity", lambda **_kwargs: None)
    kwargs: dict[str, object] = {}
    env = {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "torchmetrics_version": "1.9.0"}
    mutator(kwargs, env)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", str(env["CUDA_VISIBLE_DEVICES"]))
    monkeypatch.setenv("CUDA_DEVICE_ORDER", str(env["CUDA_DEVICE_ORDER"]))
    instance = object.__new__(cls)
    instance._preflight = {"device_contract": _synthetic_upstream_formal_device_contract()}
    with pytest.raises(base_physical.PhysicalScoreError, match="authority drift"):
        cls._runtime_device_attestation(
            instance,
            torch=_fake_torch_only_runtime(**kwargs),
            torchmetrics=types.SimpleNamespace(__version__=env["torchmetrics_version"]),
        )


def test_public_cli_is_dry_and_execution_flags_fail_before_any_route() -> None:
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    dry = subprocess.run([sys.executable, str(CLI)], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    payload = json.loads(dry.stdout)
    assert payload["classification"] == quick.CLASSIFICATION
    assert payload["selected_nwb_transfer_bytes"] == 392_880_348
    assert "torch" not in dry.stderr.lower()
    blocked = subprocess.run(
        [sys.executable, str(CLI), "--execute", "--i-have-root-reviewed-posterior-quick-screen-authorization"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert blocked.returncode != 0
    assert "capability" in (blocked.stderr + blocked.stdout).lower()


def test_static_contract_and_public_dry_cli_do_not_import_torch() -> None:
    code = f"""
import importlib.util
import sys
from pathlib import Path
path = Path({str(SOURCE)!r})
spec = importlib.util.spec_from_file_location('quick_screen_static_import_probe', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
assert module.dry_plan()['status'].startswith('DRY_ONLY')
print('static-no-torch')
"""
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    probe = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert probe.stdout.strip() == "static-no-torch"


def test_actual_closure_is_explicit_and_non_glob() -> None:
    closure = quick.implementation_closure(ROOT)
    payload = closure.payload()
    assert tuple(payload["paths"]) == quick.IMPLEMENTATION_CLOSURE
    assert payload["sha256_by_path"][quick.WORKORDER_RELATIVE] == quick.WORKORDER_SHA256
