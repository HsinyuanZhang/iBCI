"""No-data/no-CUDA focused tests for M30 attribution contract and lifecycle."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import pytest

from src.posterior_carrier_m30_attribution_v1 import attribution as a


def _sha(label: str) -> str:
    return a._digest(label.encode("utf-8"))


def _identity() -> a.AttributionIdentity:
    hashes = {path: _sha(f"closure:{path}") for path in a.IMPLEMENTATION_CLOSURE}
    return a.AttributionIdentity(a.ImplementationClosure(hashes))


def _record(surface: str, session: str, index: int) -> dict[str, object]:
    base = f"{surface}:{session}:{index}"
    return {
        "surface": surface,
        "session": session,
        "n_windows": 3 + index,
        "neural_sha256": _sha(base + ":neural"),
        "calibration_m30_sha256": _sha(base + ":calibration"),
        "last_bin_target_sha256": _sha(base + ":target"),
        "last_bin_valid_mask_sha256": _sha(base + ":mask"),
        "last_bin_valid_count": 3 + index,
        "prefix_row_ids_sha256s": {"30": _sha(base + ":prefix30"), "4": _sha(base + ":prefix4")},
        "point_carrier_sha256s": {"30": _sha(base + ":point30"), "4": _sha(base + ":point4")},
        "posterior_carrier_sha256s": {"30": _sha(base + ":posterior30"), "4": _sha(base + ":posterior4")},
        "materialization_sha256": _sha(base + ":materialization"),
    }


def _input(identity: a.AttributionIdentity) -> dict[str, object]:
    rows = []
    index = 0
    for surface in a.SURFACES:
        for session in a.SELECTED_ROSTERS[surface]:
            rows.append(_record(surface, session, index))
            index += 1
    return a.InputReplay(tuple(rows)).payload(identity=identity)


def _v3_validation() -> dict[str, object]:
    return {
        "schema": "posterior_carrier_m30_attribution_v3_replay_validation_v1",
        "v3_predecessor": a.V3Evidence().payload(),
        "v3_identity_closure_sha256": a.V3_CLOSURE_SHA256,
        "v3_input_replayed_exactly": True,
        "v3_score_terminal_chain_valid": True,
        "reused_v3_m30_cell_sha256s": dict(a.V3_REUSED_M30_CELL_SHA256S),
    }


def _evidence_object(cell: a.AttributionCell, input_payload: Mapping[str, object], value: float) -> a.CellEvidence:
    records = {row["session"]: row for row in input_payload["records"] if row["surface"] == cell.surface}
    sessions = tuple(
        a.SessionScore(
            session=session,
            n_windows=records[session]["n_windows"],
            r2=value + offset * 0.01,
            prediction_sha256=_sha(f"prediction:{cell.surface}:{cell.system}:{session}"),
            input_record_sha256=a._digest(a._json(records[session])),
        )
        for offset, session in enumerate(a.SELECTED_ROSTERS[cell.surface])
    )
    raw = {session: _sha(f"raw:{cell.surface}:{cell.system}:{session}") for session in a.SELECTED_ROSTERS[cell.surface]}
    normalized = {session: _sha(f"norm:{cell.surface}:{cell.system}:{session}") for session in a.SELECTED_ROSTERS[cell.surface]}
    credibility = {session: _sha(f"cred:{cell.surface}:{cell.system}:{session}") for session in a.SELECTED_ROSTERS[cell.surface]}
    system = "sealed" if cell.system in {a.SEALED_OLS_POINT, a.SEALED_POSTERIOR_MEAN} else "posterior"
    return a.CellEvidence(
        cell=cell,
        model_swa_sha256=(
            "626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd"
            if system == "sealed" else "def27d8edffc0c6ed17dee80292dd1c1b628ba1d454dc214278caee814cf40fa"
        ),
        sessions=sessions,
        input_replay_sha256=a._digest(a._json(input_payload)),
        model_state_before_sha256=_sha(f"state:{system}"),
        model_state_after_sha256=_sha(f"state:{system}"),
        carrier_raw_sha256s=raw,
        carrier_normalized_sha256s=normalized,
        credibility_sha256s=credibility,
        raw_before_normalization_verified=True,
        eval_mode=True,
        dropout_disabled=True,
        gradients_none=True,
        finite_outputs=True,
        repeated_fixed_batch_bitwise_equal=True,
        b3s_m30_recomputed=True,
        no_target_sampling=True,
        evidence_origin=("V3_REUSED_VALIDATED" if cell.system in a.V3_REUSED_SYSTEMS else "NEW_FORWARD"),
        v3_reused_cell_sha256=(
            a.V3_REUSED_M30_CELL_SHA256S[f"{cell.surface}:{cell.system}"]
            if cell.system in a.V3_REUSED_SYSTEMS else None
        ),
    )


def _evidence(cell: a.AttributionCell, input_payload: Mapping[str, object], value: float) -> dict[str, object]:
    return _evidence_object(cell, input_payload, value).payload(identity=_identity(), input_payload=input_payload)


def _complete_flags() -> a.RuntimeFlags:
    flags = a.RuntimeFlags(within_opened=True, external_opened=True, remote_initialized=True)
    for cell in a.attribution_matrix():
        flags.record_cell(cell)
    return flags


def test_matrix_is_exact_m30_five_systems_two_surfaces() -> None:
    matrix = a.attribution_matrix()
    assert len(matrix) == 10
    assert [(cell.surface, cell.system, cell.budget) for cell in matrix] == [
        (surface, system, 30) for surface in a.SURFACES for system in a.SYSTEMS
    ]
    assert a.NEW_FORWARD_SYSTEMS == {
        a.SEALED_POSTERIOR_MEAN, a.POSTERIOR_OLS_UNIFORM, a.POSTERIOR_MEAN_UNIFORM,
    }
    assert a.V3_REUSED_SYSTEMS == {a.SEALED_OLS_POINT, a.POSTERIOR_MEAN_PRECISION}


def test_input_replay_rejects_roster_order_and_budget_drift() -> None:
    identity = _identity(); payload = _input(identity)
    assert a.validate_input_replay_payload(payload, identity=identity) == payload
    wrong = copy.deepcopy(payload)
    wrong["records"][0], wrong["records"][1] = wrong["records"][1], wrong["records"][0]
    with pytest.raises(a.AttributionError, match="roster/order"):
        a.validate_input_replay_payload(wrong, identity=identity)
    wrong = copy.deepcopy(payload)
    wrong["records"][0]["point_carrier_sha256s"] = {"30": _sha("x"), "10": _sha("y")}
    with pytest.raises(a.AttributionError, match="budget topology"):
        a.validate_input_replay_payload(wrong, identity=identity)


def test_v3_validation_requires_exact_chain_and_four_reused_rows() -> None:
    payload = _v3_validation()
    assert a.validate_v3_validation_payload(payload) == payload
    wrong = copy.deepcopy(payload); wrong["v3_identity_closure_sha256"] = _sha("drift")
    with pytest.raises(a.AttributionError, match="immutable binding"):
        a.validate_v3_validation_payload(wrong)
    wrong = copy.deepcopy(payload); del wrong["reused_v3_m30_cell_sha256s"][f"{a.WITHIN}:{a.SEALED_OLS_POINT}"]
    with pytest.raises(a.AttributionError, match="topology"):
        a.validate_v3_validation_payload(wrong)
    wrong = copy.deepcopy(payload)
    wrong["reused_v3_m30_cell_sha256s"][f"{a.EXTERNAL}:{a.POSTERIOR_MEAN_PRECISION}"] = _sha("forged-v3-cell")
    with pytest.raises(a.AttributionError, match="literal digest"):
        a.validate_v3_validation_payload(wrong)


def test_cell_evidence_rejects_wrong_swa_origin_raw_or_same_input() -> None:
    identity = _identity(); inputs = _input(identity)
    cell = a.AttributionCell(a.WITHIN, a.POSTERIOR_MEAN_UNIFORM)
    payload = _evidence(cell, inputs, 0.1)
    assert a.validate_cell_evidence_payload(payload, identity=identity, input_payload=inputs) == payload
    wrong = copy.deepcopy(payload); wrong["model_swa_sha256"] = _sha("wrong")
    with pytest.raises(a.AttributionError, match="model/SWA"):
        a.validate_cell_evidence_payload(wrong, identity=identity, input_payload=inputs)
    wrong = copy.deepcopy(payload); wrong["evidence_origin"] = "V3_REUSED_VALIDATED"
    with pytest.raises(a.AttributionError, match="origin"):
        a.validate_cell_evidence_payload(wrong, identity=identity, input_payload=inputs)
    wrong = copy.deepcopy(payload); wrong["sessions"][0]["input_record_sha256"] = _sha("wrong")
    with pytest.raises(a.AttributionError, match="same-input"):
        a.validate_cell_evidence_payload(wrong, identity=identity, input_payload=inputs)


def test_reused_evidence_requires_literal_v3_body_reconstruction() -> None:
    """Synthetic rows cannot impersonate immutable V3 A/E evidence.

    We do not fabricate a successful V3 body in a no-data test.  Instead this
    asserts that every copied source field enters the original V3 JSON domain
    and that a route-local body is rejected unless it reproduces the frozen
    remote digest.
    """
    identity = _identity()
    inputs = _input(identity)
    # Begin with a valid new-forward shape, then construct a syntactically
    # valid A row.  It remains invalid because its copied facts are not the
    # immutable V3 facts whose body digest is frozen by the contract.
    body = _evidence(a.AttributionCell(a.WITHIN, a.POSTERIOR_MEAN_UNIFORM), inputs, 0.2)
    body["cell"] = a.AttributionCell(a.WITHIN, a.SEALED_OLS_POINT).payload()
    body["model_system"] = "sealed_cell_d_checkpoint"
    body["model_swa_sha256"] = "626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd"
    body["evidence_origin"] = "V3_REUSED_VALIDATED"
    body["v3_reused_cell_sha256"] = a.V3_REUSED_M30_CELL_SHA256S[f"{a.WITHIN}:{a.SEALED_OLS_POINT}"]
    original = a._reconstruct_original_v3_reused_cell(body)
    original_digest = a._digest(a._json(original))
    assert original_digest != body["v3_reused_cell_sha256"]
    with pytest.raises(a.AttributionError, match="original-body replay"):
        a.validate_cell_evidence_payload(body, identity=identity, input_payload=inputs)

    variants: list[dict[str, object]] = []
    changed_r2 = copy.deepcopy(body); changed_r2["sessions"][0]["r2"] += 0.125; variants.append(changed_r2)
    changed_prediction = copy.deepcopy(body); changed_prediction["sessions"][0]["prediction_sha256"] = _sha("changed-prediction"); variants.append(changed_prediction)
    changed_state = copy.deepcopy(body); changed_state["model_state_before_sha256"] = _sha("changed-state"); changed_state["model_state_after_sha256"] = _sha("changed-state"); variants.append(changed_state)
    changed_model = copy.deepcopy(body); changed_model["model_swa_sha256"] = _sha("changed-model"); variants.append(changed_model)
    for changed in variants:
        assert a._digest(a._json(a._reconstruct_original_v3_reused_cell(changed))) != original_digest
        with pytest.raises(a.AttributionError):
            a.validate_cell_evidence_payload(changed, identity=identity, input_payload=inputs)

    changed_digest = copy.deepcopy(body); changed_digest["v3_reused_cell_sha256"] = _sha("forged-reused-cell")
    with pytest.raises(a.AttributionError, match="literal SHA"):
        a.validate_cell_evidence_payload(changed_digest, identity=identity, input_payload=inputs)


@pytest.mark.parametrize("field", tuple(a.V3_TORCH_ONLY_DEVICE_AUTHORITY))
def test_v3_torch_only_device_authority_rejects_each_literal_drift(field: str) -> None:
    exact = dict(a.V3_TORCH_ONLY_DEVICE_AUTHORITY)
    assert a.validate_v3_torch_only_device_payload(exact) == exact
    wrong = dict(exact)
    value = wrong[field]
    if value is None:
        wrong[field] = "unexpected-non-null"
    elif isinstance(value, bool):
        wrong[field] = not value
    elif isinstance(value, int):
        wrong[field] = value + 1
    elif isinstance(value, list):
        wrong[field] = [value[0] + 1, *value[1:]]
    else:
        wrong[field] = f"{value}__drift"
    with pytest.raises(a.AttributionError, match="Torch-only device authority"):
        a.validate_v3_torch_only_device_payload(wrong)


def test_identity_embeds_exact_v3_torch_only_device_authority() -> None:
    identity = _identity()
    payload = a.validate_identity(identity)
    assert payload["remote_stage"]["device_contract"] == a.V3_TORCH_ONLY_DEVICE_AUTHORITY
    wrong = copy.deepcopy(payload)
    wrong["remote_stage"]["device_contract"]["torchmetrics_version"] = "1.5.1"
    with pytest.raises(a.AttributionError, match="scientific boundary"):
        a.validate_identity(wrong)


def test_descriptive_pairing_and_flag_topology_remain_non_governing() -> None:
    left = [{"session": "a", "r2": 0.3}, {"session": "b", "r2": 0.2}, {"session": "c", "r2": 0.1}]
    right = [{"session": "a", "r2": 0.2}, {"session": "b", "r2": 0.1}, {"session": "c", "r2": 0.0}]
    paired = a._paired(left, right, "synthetic")
    decision = a._attribution_decision(
        sealed_estimator=paired, posterior_estimator=paired, bias=paired, consumer_system=paired,
    )
    assert paired["positive_count"] == 3
    assert decision["classification"] == "DESCRIPTIVE_NON_GOVERNING_NON_INFERENTIAL"
    flags = _complete_flags(); flags.new_forward_cells.pop()
    with pytest.raises(a.AttributionError, match="matrix/runtime topology"):
        a._validate_runtime_flags(flags, complete=True)


def test_lifecycle_failure_does_not_leave_partial_score_group(tmp_path: Path) -> None:
    identity = _identity()

    class Broken:
        closed = False

        def prepare(self, *, identity: a.AttributionIdentity, flags: a.RuntimeFlags) -> None:
            del identity, flags
            raise RuntimeError("synthetic pre-input failure")

        def resolve_inputs(self, **_kwargs: Any) -> a.InputReplay:
            raise AssertionError("must fail before input resolution")

        def score_cell(self, **_kwargs: Any) -> a.CellEvidence:
            raise AssertionError("must fail before scoring")

        def reverify_after_forwards(self, **_kwargs: Any) -> a.ImplementationClosure:
            raise AssertionError("must fail before reverify")

        def device_attestation(self) -> Mapping[str, object]:
            raise AssertionError("must fail before device attestation")

        def v3_validation(self) -> Mapping[str, object]:
            raise AssertionError("must fail before V3 validation")

        def close(self) -> None:
            self.closed = True

    backend = Broken()
    with pytest.raises(RuntimeError, match="synthetic pre-input"):
        a.run_attribution_lifecycle(
            root=tmp_path, identity=identity, backend=backend,
            execution_capability=a._issue_root_review_capability(identity),
        )
    output = tmp_path / a.RESULT_ROOT_RELATIVE
    assert (output / "attempt.json").exists()
    assert (output / "failure.json").exists()
    assert not (output / "score.json").exists()
    assert not (output / "terminal.json").exists()
    assert backend.closed


def test_success_group_rolls_back_both_pairs_if_post_publish_reload_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / Path(a.RESULT_ROOT_RELATIVE).name
    artifact = a._ArtifactRoot(root)
    artifact.reserve()
    original = artifact.reload_json

    def fail_terminal(name: str, expected_sha256: str) -> dict[str, object]:
        if name == "terminal.json":
            raise a.AttributionError("synthetic terminal reload failure")
        return original(name, expected_sha256)

    monkeypatch.setattr(artifact, "reload_json", fail_terminal)
    with pytest.raises(a.AttributionError, match="synthetic terminal"):
        artifact.publish_success_group(score={"score": True}, terminal={"terminal": True})
    assert not (root / "score.json").exists()
    assert not (root / "score.json.sha256").exists()
    assert not (root / "terminal.json").exists()
    assert not (root / "terminal.json.sha256").exists()
    artifact.close()


def test_public_cli_is_static_dry_and_flags_fail_before_execution(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_posterior_carrier_m30_attribution.py"
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""})
    dry = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=environment, capture_output=True, text=True, check=True)
    assert "DRY_ONLY__NO_NETWORK_NO_TORCH" in dry.stdout
    blocked = subprocess.run(
        [sys.executable, str(script), "--execute", "--i-have-root-reviewed-posterior-m30-attribution-authorization"],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
    )
    assert blocked.returncode != 0
    assert "remains dry" in (blocked.stderr + blocked.stdout)


def test_dry_stage_plan_is_pure_and_identity_bound(tmp_path: Path) -> None:
    from src.posterior_carrier_m30_attribution_v1 import remote_stage

    identity = _identity()
    before = set(tmp_path.iterdir())
    plan = remote_stage.dry_stage_plan(root=tmp_path, identity=identity)
    assert plan["staging_policy"]["network_performed"] is False
    assert plan["v3_predecessor"]["score_sha256"] == a.V3_SCORE_SHA256
    assert set(tmp_path.iterdir()) == before


def test_physical_module_static_import_does_not_import_torch() -> None:
    sys.modules.pop("torch", None)
    import src.posterior_carrier_m30_attribution_v1.physical as physical

    assert hasattr(physical, "PhysicalM30AttributionBackend")
    assert "torch" not in sys.modules


def test_raw_ols_capture_happens_before_any_normalizer_with_fake_substrate() -> None:
    """The capture seam retains raw M30 plus the frozen axis proof exactly."""
    import numpy as np
    from src.posterior_carrier_m30_attribution_v1 import physical as attribution_physical

    class FakeError(RuntimeError):
        pass

    class FakeParent:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def _raw_t4_by_budget(self, *, runtime: Mapping[str, Any], snapshot: object, session: str, record: object) -> tuple[dict[int, Any], dict[int, dict[str, object]]]:
            del runtime, snapshot, record
            raw = {30: np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)}
            return raw, {30: {"raw_t4_sha256": FakePhysical._array_digest(raw[30]), "body_sha256": _sha(session)}}

    class FakeV3Physical:
        class RemoteEngineeringDevice:
            pass

        @staticmethod
        def build_torch_only_engineering_backend_class(*, physical: object, device: object) -> type[FakeParent]:
            del physical, device
            return FakeParent

    class FakePhysical:
        PhysicalScoreError = FakeError

        @staticmethod
        def _array_digest(value: Any) -> str:
            return a._digest(np.ascontiguousarray(value, dtype=np.float32).tobytes())

    captured = attribution_physical.build_capturing_v3_backend_class(
        v3_physical=FakeV3Physical, physical=FakePhysical,
    )()
    raw, proof = captured._raw_t4_by_budget(
        runtime={"np": np}, snapshot=object(), session="s1", record=object(),
    )
    assert np.array_equal(raw[30], captured._attribution_raw_ols_m30["s1"])
    assert captured._attribution_raw_axis_proofs["s1"] == proof[30]
    # A later mutation of the producer buffer cannot alter the retained raw
    # carrier used by C; it is a detached contiguous copy, not point_side.
    raw[30][0, 0] = 99.0
    assert captured._attribution_raw_ols_m30["s1"][0, 0] == 1.0


def test_raw_ols_capture_rejects_axis_digest_drift_with_fake_substrate() -> None:
    import numpy as np
    from src.posterior_carrier_m30_attribution_v1 import physical as attribution_physical

    class FakeError(RuntimeError):
        pass

    class Parent:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def _raw_t4_by_budget(self, **_kwargs: Any) -> tuple[dict[int, Any], dict[int, dict[str, object]]]:
            raw = {30: np.zeros((1, 4), dtype=np.float32)}
            return raw, {30: {"raw_t4_sha256": _sha("wrong"), "body_sha256": _sha("body")}}

    class V3:
        class RemoteEngineeringDevice:
            pass

        @staticmethod
        def build_torch_only_engineering_backend_class(**_kwargs: Any) -> type[Parent]:
            return Parent

    class Physical:
        PhysicalScoreError = FakeError

        @staticmethod
        def _array_digest(value: Any) -> str:
            return a._digest(np.ascontiguousarray(value, dtype=np.float32).tobytes())

    backend = attribution_physical.build_capturing_v3_backend_class(v3_physical=V3, physical=Physical)()
    with pytest.raises(FakeError, match="raw OLS"):
        backend._raw_t4_by_budget(runtime={"np": np}, snapshot=object(), session="s", record=object())
