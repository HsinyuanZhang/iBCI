"""No-target adversarial tests for post-cost fixed-geometry controls."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration/src"
VENDOR = ROOT / "cebra_exploration/third_party/cebra"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_actual_cpu_route as route  # noqa: E402
import track_b_v2_post_cost_synthetic_controls as controls  # noqa: E402


RUNNER = ROOT / "cebra_exploration/scripts/run_track_b_v2_post_cost_synthetic_controls.py"
ALIGNMENT = {
    "offset_left": 5,
    "offset_right": 5,
    "valid_slice_start": 5,
    "valid_slice_stop": -5,
}


def _runner(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = f"{SRC}:{VENDOR}"
    return subprocess.run(
        [sys.executable, str(RUNNER), *args], cwd=ROOT, env=env,
        capture_output=True, text=True,
    )


def _rows(*, positive: float = 0.7, ordinary: float = 0.1,
          deranged: float = -0.2) -> list[dict]:
    result = []
    values = {
        "cebra_joint_behavior": positive,
        route.UNALIGNED_ARM: ordinary,
        route.DERANGED_ARM: deranged,
    }
    for seed in controls.SEEDS:
        for arm in controls.ARMS:
            for decoder in controls.DECODERS:
                result.append({
                    "seed": seed,
                    "arm": arm,
                    "decoder": decoder,
                    "target_query_r2": values[arm] + seed * 0.001,
                    "query_neural_or_auxiliary_in_fit": False,
                })
    return result


def _receipt() -> dict:
    rows = _rows()
    closure = {"new_route": {"path": "/immutable", "sha256": "1" * 64, "bytes": 1}}
    body = {
        "schema": controls.SCHEMA_RECEIPT,
        "status": controls.STATUS_RECEIPT,
        "official": False,
        "scientific_result": False,
        "engineering_measurements_only": True,
        "canonical_cost_receipt": {"canonical_body_sha256": controls.EXPECTED_COST_SHA256},
        "fixed_geometry": controls.GEOMETRY.as_dict(),
        "seeds": list(controls.SEEDS),
        "seed_provenance": {
            "requested_seeds": list(controls.SEEDS),
            "same_seed_reused_across_three_arms_for_matched_sensitivity": True,
            "vendored_sklearn_cebra_more_tags_non_deterministic": True,
            "seeding_fully_implemented_by_vendored_cebra": False,
            "bitwise_determinism_claimed": False,
        },
        "arms": list(controls.ARMS),
        "derangement_authority": {
            "nonidentity_derangement": True,
            "cebra_seed_independent": True,
            "support_neural_rows_exact_equal": True,
            "auxiliary_label_multiset_exact_equal": True,
            "permutation_int64_sha256": "2" * 64,
        },
        "measurements": rows,
        "arm_run_count": controls.EXPECTED_ARM_RUNS,
        "decoder_measurement_count": controls.EXPECTED_MEASUREMENTS,
        "cebra_fit_call_count": controls.EXPECTED_FIT_CALLS,
        "measured_distributions_and_threshold_proposal": controls.threshold_proposal(rows),
        "threshold_frozen": False,
        "threshold_authority_minted": False,
        "final_Subject_M_runtime_control_pair_minted": False,
        "final_RT_runtime_control_pair_minted": False,
        "target_execution_authorized": False,
        "implementation_closure_at_launch": closure,
        "implementation_closure_at_final": closure,
        "launch_final_closure_exact_equal": True,
        "target_data_opened": False,
        "formal_data_opened": False,
        "NWB_or_NPZ_opened": False,
    }
    body["receipt_payload_sha256"] = controls._sha_json(body)
    return body


def _resign(body: dict) -> dict:
    body.pop("receipt_payload_sha256", None)
    body["receipt_payload_sha256"] = controls._sha_json(body)
    return body


def test_live_no_data_preflight_binds_exact_cost_pair_and_complete_closure() -> None:
    payload = controls.build_preflight()
    assert payload["status"] == controls.STATUS_PREFLIGHT
    assert payload["canonical_cost_gate"]["canonical_body_sha256"] == controls.EXPECTED_COST_SHA256
    assert payload["fixed_geometry"] == {
        "output_dimension": 8, "source_iterations": 10_000,
        "encoder_geometry_key": "d8-it10000",
    }
    assert payload["expected_cebra_fit_call_count"] == 40
    assert payload["expected_decoder_measurement_count"] == 48
    assert payload["synthetic_arrays_built"] is False
    assert payload["gpu_used"] is payload["target_data_opened"] is False
    closure = payload["implementation_closure"]
    assert closure["vendored_cebra_provenance"]["path"].endswith("third_party/CEBRA_PROVENANCE.txt")
    assert closure["root_frozen_protocol"]["sha256"] == controls.PROTOCOL_SHA256


def test_default_runner_is_no_fit_no_torch_no_data_no_write() -> None:
    assert not os.path.lexists(controls.CANONICAL_OUTPUT)
    assert not os.path.lexists(controls._sidecar(controls.CANONICAL_OUTPUT))
    completed = _runner()
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == controls.STATUS_PREFLIGHT
    assert payload["torch_imported"] is payload["cebra_imported"] is False
    assert payload["synthetic_arrays_built"] is payload["gpu_used"] is False
    assert not os.path.lexists(controls.CANONICAL_OUTPUT)
    assert not os.path.lexists(controls._sidecar(controls.CANONICAL_OUTPUT))


@pytest.mark.parametrize("flag", ("--execute", "--i-have-authorization"))
def test_runner_rejects_single_execution_flag_before_gpu_or_data(flag: str) -> None:
    completed = _runner(flag)
    assert completed.returncode != 0
    assert "require both" in completed.stderr
    assert not os.path.lexists(controls.CANONICAL_OUTPUT)


def test_runner_rejects_gpu_substitution_and_has_no_target_or_geometry_cli() -> None:
    completed = _runner("--physical-gpu", "0")
    assert completed.returncode != 0
    assert "physical GPU1" in completed.stderr
    text = RUNNER.read_text()
    for forbidden in ("--target", "--formal", "--nwb", "--npz", "--seed", "--geometry",
                      "--iterations", "--threshold", "--output"):
        assert forbidden not in text.lower()


def test_body_or_sidecar_conflict_precedes_cost_closure_and_synthetic_data(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for body_exists, sidecar_exists in ((True, False), (False, True), (True, True)):
        body = tmp_path / f"case-{body_exists}-{sidecar_exists}" / "receipt.json"
        body.parent.mkdir()
        if body_exists:
            body.write_text("poison")
        if sidecar_exists:
            controls._sidecar(body).write_text("poison")
        monkeypatch.setattr(controls, "CANONICAL_OUTPUT", body)
        monkeypatch.setattr(
            controls, "validate_cost_pair",
            lambda: (_ for _ in ()).throw(AssertionError("cost must not run")),
        )
        monkeypatch.setattr(
            controls, "synthetic_fold",
            lambda: (_ for _ in ()).throw(AssertionError("data must not build")),
        )
        with pytest.raises(controls.PostCostSyntheticControlError, match="fresh"):
            controls.build_preflight()


def test_cost_and_protocol_or_implementation_drift_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        controls.cost_validator, "inspect_fixed_d8it250_gpu_cost_receipt",
        lambda: {"status": "poison"},
    )
    with pytest.raises(controls.PostCostSyntheticControlError, match="not live-valid"):
        controls.validate_cost_pair()
    monkeypatch.undo()
    original = controls.source._read_regular_file

    def poison(path: Path, *, label: str) -> bytes:
        raw = original(path, label=label)
        return raw + b"drift" if "protocol" in label else raw

    monkeypatch.setattr(controls.source, "_read_regular_file", poison)
    with pytest.raises(controls.PostCostSyntheticControlError, match="protocol SHA drift"):
        controls.implementation_closure()


def test_fixed_derangement_is_seed_independent_nonidentity_and_preserves_neural_multiset() -> None:
    fold = controls.synthetic_fold()
    authority = controls.synthetic_data_authority(fold)
    first, first_auth = route.fixed_derangement(
        fold.held_support_auxiliary.shape[0], authority["synthetic_data_authority_sha256"])
    second, second_auth = route.fixed_derangement(
        fold.held_support_auxiliary.shape[0], authority["synthetic_data_authority_sha256"])
    assert np.array_equal(first, second)
    assert first_auth == second_auth
    assert np.all(first != np.arange(first.size))
    neural, auxiliary, proof = route.apply_deranged_auxiliary(
        support_neural=fold.held_support_neural,
        support_auxiliary=fold.held_support_auxiliary,
        permutation=first,
    )
    assert np.array_equal(neural, fold.held_support_neural)
    assert proof["auxiliary_label_multiset_exact_equal"] is True
    assert np.array_equal(auxiliary, fold.held_support_auxiliary[first])


def test_derangement_rejects_identity_neural_drift_multiset_drift_and_seed_dependence() -> None:
    fold = controls.synthetic_fold()
    n = fold.held_support_auxiliary.shape[0]
    permutation, _ = route.fixed_derangement(n, "authority")
    good_y = fold.held_support_auxiliary[permutation]
    with pytest.raises(route.TrackBV2ActualCpuError, match="identity"):
        route.verify_derangement_arrays(
            neural_before=fold.held_support_neural, neural_after=fold.held_support_neural,
            auxiliary_before=fold.held_support_auxiliary, auxiliary_after=fold.held_support_auxiliary,
            permutation=np.arange(n),
        )
    with pytest.raises(route.TrackBV2ActualCpuError, match="neural"):
        route.verify_derangement_arrays(
            neural_before=fold.held_support_neural, neural_after=fold.held_support_neural + 1,
            auxiliary_before=fold.held_support_auxiliary, auxiliary_after=good_y,
            permutation=permutation,
        )
    bad_y = good_y.copy()
    bad_y[0] += 1
    with pytest.raises(route.TrackBV2ActualCpuError):
        route.verify_derangement_arrays(
            neural_before=fold.held_support_neural, neural_after=fold.held_support_neural,
            auxiliary_before=fold.held_support_auxiliary, auxiliary_after=bad_y,
            permutation=permutation,
        )
    other, _ = route.fixed_derangement(n, "seed-dependent-poison")
    assert not np.array_equal(permutation, other), "adversarial fixture must change the permutation"


def test_offset10_crops_each_block_separately_and_query_never_enters_fit() -> None:
    torch = pytest.importorskip("torch")
    torchmetrics = pytest.importorskip("torchmetrics")
    fold = controls.synthetic_fold()
    run = route.EmbeddingRun(
        peer_fit_embeddings=tuple(np.asarray(x[:, :8], dtype=np.float64) for x in fold.peer_neural),
        target_support_embedding=np.asarray(fold.held_support_neural[:, :8], dtype=np.float64),
        target_query_embedding=np.asarray(fold.held_query_neural[:, :8], dtype=np.float64),
        fit_calls=({"label": "synthetic"},), model_alignment=ALIGNMENT,
    )
    rows, proof = controls.score_embedding_run(
        run=run, fold=fold, torch=torch, torchmetrics=torchmetrics,
    )
    assert len(rows) == 2
    assert [p["valid_embedding_rows"] for p in proof["source_blocks"]] == [86, 86, 86]
    assert proof["held_support_block"]["valid_embedding_rows"] == 38
    assert proof["held_query_block"]["valid_embedding_rows"] == 38
    assert proof["concatenate_then_crop_used"] is False
    assert proof["padded_edge_rows_used"] == 0
    assert all(row["query_neural_or_auxiliary_in_fit"] is False for row in rows)
    expected_fit = np.concatenate([x[5:-5, :8] for x in fold.peer_neural], axis=0)
    assert rows[0]["fit_embedding_float64_sha256"] == route.array_sha256(expected_fit)


def test_short_block_is_rejected_instead_of_crossing_boundaries() -> None:
    with pytest.raises(route.TrackBV2ActualCpuError):
        controls._crop_block(np.zeros((9, 8)), np.zeros((9, 2)), ALIGNMENT, role="short")


def test_threshold_proposal_is_measurement_only_and_never_freezes() -> None:
    proposal = controls.threshold_proposal(_rows())
    assert proposal["threshold_frozen"] is proposal["threshold_authority_minted"] is False
    assert proposal["target_execution_authorized"] is False
    for decoder in controls.DECODERS:
        item = proposal["per_decoder"][decoder]
        assert item["separating_threshold_exists_on_measured_eight_seeds"] is True
        assert item["ordinary_negative_is_diagnostic_not_a_rescue_or_gate"] is True
    overlapping = controls.threshold_proposal(_rows(positive=-0.2, deranged=0.7))
    assert overlapping["conservative_global_candidate"] is None


@pytest.mark.parametrize("mutation", (
    "missing_cell", "fit_count", "query_leak", "threshold_frozen", "target_open",
    "identity", "seed_claim", "closure", "self_sha",
))
def test_receipt_tamper_fails_closed(mutation: str) -> None:
    body = _receipt()
    if mutation == "missing_cell":
        body["measurements"].pop()
    elif mutation == "fit_count":
        body["cebra_fit_call_count"] = 39
    elif mutation == "query_leak":
        body["measurements"][0]["query_neural_or_auxiliary_in_fit"] = True
    elif mutation == "threshold_frozen":
        body["threshold_frozen"] = True
    elif mutation == "target_open":
        body["target_data_opened"] = True
    elif mutation == "identity":
        body["derangement_authority"]["nonidentity_derangement"] = False
    elif mutation == "seed_claim":
        body["seed_provenance"]["bitwise_determinism_claimed"] = True
    elif mutation == "closure":
        body["implementation_closure_at_final"] = {}
    elif mutation == "self_sha":
        body["receipt_payload_sha256"] = "0" * 64
    if mutation != "self_sha":
        _resign(body)
    with pytest.raises(controls.PostCostSyntheticControlError):
        controls.validate_measurement_receipt(body)


def test_valid_measurement_receipt_and_exact_grid() -> None:
    body = _receipt()
    controls.validate_measurement_receipt(body)
    assert len({(r["seed"], r["arm"], r["decoder"]) for r in body["measurements"]}) == 48


def test_route_surface_contains_no_target_nwb_or_final_authority_minter() -> None:
    text = Path(controls.__file__).read_text().lower()
    assert "open_canonical_verified_asset_after_authority" not in text
    assert "pynwb" not in text
    assert "h5py" not in text
    assert "np.load" not in text
    assert "target execution authorized" not in text
    assert "final_subject_m_runtime_control_pair_minted\": true" not in text
