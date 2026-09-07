from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

# Tests must work whether pytest is invoked from repository root or from
# ``sua_exploration``.  Other legacy tests rely on the latter cwd; this new
# finalizer test deliberately does not.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aggregate_sua_t4_factorized_logit import (
    ANCHOR_SHA,
    BASELINE,
    EPOCHS,
    _load_scored_result,
    _metadata_for,
    _validate_common_result,
    TEACHER_SHA,
    rederive_result_scores,
    tensor_state_sha256,
    verify_checkpoint,
)


SESSIONS = tuple(f"validation_{index}" for index in range(6))


def _minimal_result() -> dict:
    per_epoch = {}
    for epoch in EPOCHS:
        scores = {session: 0.4 + 0.01 * index + epoch * 1e-4 for index, session in enumerate(SESSIONS)}
        mean = sum(scores.values()) / len(scores)
        per_epoch[str(epoch)] = {"per_session_r2": scores, "mean_r2": mean}
    return {
        "epoch_list": list(EPOCHS),
        "protocol": {"epoch_window": list(EPOCHS)},
        "per_epoch": per_epoch,
        "per_epoch_mean_r2": {str(epoch): per_epoch[str(epoch)]["mean_r2"] for epoch in EPOCHS},
        "variant_score": sum(per_epoch[str(epoch)]["mean_r2"] for epoch in EPOCHS) / len(EPOCHS),
    }


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(path, state):
    factors = {
        "query_factors": state["student.t4_logit_residual.query_factors"],
        "unit_projection.weight": state["student.t4_logit_residual.unit_projection.weight"],
    }
    torch.save(
        {
            "state_dict": state,
            "t4_logit_residual_receipt": {
                "module": "T4LogitResidualLitModule",
                "residual_mode": "aligned",
                "interaction_mode": "attention_logit",
                "residual_rank": 8,
                "residual_permutation_seed": None,
                "teacher_checkpoint_sha256": TEACHER_SHA,
                "selected_t4_anchor_sha256": ANCHOR_SHA,
                "backbone_frozen": True,
                "cached_state": "per-unit rank factor only",
                "active_factor_sha256": tensor_state_sha256(factors),
                "initial_factor_sha256": "initial",
            },
        },
        path,
    )


def test_rederives_epoch_and_session_means_without_trusting_summary():
    result = _minimal_result()
    sessions, score, session_means = rederive_result_scores(result, label="synthetic")
    assert sessions == SESSIONS
    assert score == pytest.approx(result["variant_score"])
    assert session_means[SESSIONS[-1]] > session_means[SESSIONS[0]]


def test_rejects_result_with_tampered_top_level_score():
    result = _minimal_result()
    result["variant_score"] += 0.01
    with pytest.raises(ValueError, match="variant_score"):
        rederive_result_scores(result, label="synthetic")


def test_checkpoint_audit_requires_exactly_two_factors_and_bitwise_anchor(tmp_path):
    anchor = {"student.decoder.weight": torch.tensor([1.0]), "student.id_encoder.weight": torch.tensor([2.0])}
    state = dict(anchor)
    state["student.t4_logit_residual.query_factors"] = torch.ones(2, 8)
    state["student.t4_logit_residual.unit_projection.weight"] = torch.ones(8, 4)
    checkpoint = tmp_path / "epoch.ckpt"
    _checkpoint(checkpoint, state)
    receipt = verify_checkpoint(checkpoint, expected_sha=_sha(checkpoint), anchor_state=anchor, arm="aligned_logit", epoch=5)
    assert receipt["query_factors_nonzero"] == 16
    assert receipt["unit_projection_nonzero"] == 32

    state["student.decoder.weight"] = torch.tensor([3.0])
    _checkpoint(checkpoint, state)
    with pytest.raises(ValueError, match="inherited anchor tensor drifted"):
        verify_checkpoint(checkpoint, expected_sha=_sha(checkpoint), anchor_state=anchor, arm="aligned_logit", epoch=5)


def test_checkpoint_audit_rejects_a_third_new_student_tensor(tmp_path):
    anchor = {"student.decoder.weight": torch.tensor([1.0])}
    state = dict(anchor)
    state["student.t4_logit_residual.query_factors"] = torch.ones(2, 8)
    state["student.t4_logit_residual.unit_projection.weight"] = torch.ones(8, 4)
    state["student.illegal_new_path"] = torch.ones(1)
    checkpoint = tmp_path / "epoch.ckpt"
    _checkpoint(checkpoint, state)
    with pytest.raises(ValueError, match="only two extra student factor tensors"):
        verify_checkpoint(checkpoint, expected_sha=_sha(checkpoint), anchor_state=anchor, arm="aligned_logit", epoch=5)


@pytest.mark.parametrize(
    ("key", "bad_value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("generated_by", "other_evaluator.py", "generated_by"),
        ("purpose", "argmax_selected", "purpose"),
    ],
)
def test_common_result_rejects_schema_evaluator_and_purpose_drift(key, bad_value, message):
    result = json.loads(BASELINE.read_text(encoding="utf-8"))
    result[key] = bad_value
    with pytest.raises(ValueError, match=message):
        _validate_common_result(result, label="baseline", expected_sessions=None)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda metadata: metadata["session_files"].__setitem__("test", ["formal.nwb"]), "session_files.test"),
        (lambda metadata: metadata["trainer_fit_validation_loader_contract"].__setitem__("formal_test_sessions_loaded_during_fit", True), "fit-loader"),
        (lambda metadata: metadata["held_out_evaluation_protocol"].__setitem__("held_out_test_evaluated", True), "isolation seal"),
    ],
)
def test_metadata_rejects_each_formal_access_seal_drift(tmp_path, mutator, message):
    result = json.loads(BASELINE.read_text(encoding="utf-8"))
    metadata = json.loads(Path(result["run_metadata_path"]).read_text(encoding="utf-8"))
    mutator(metadata)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    result["run_metadata_path"] = str(metadata_path)
    result["run_metadata_sha256"] = _sha(metadata_path)
    with pytest.raises(ValueError, match=message):
        _metadata_for(result, label="baseline")


def test_scored_result_rejects_noncanonical_epoch_checkpoint_path(tmp_path):
    result = json.loads(BASELINE.read_text(encoding="utf-8"))
    result["per_epoch"]["5"]["checkpoint_path"] = str(tmp_path / "misbound.ckpt")
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint path"):
        _load_scored_result(path, arm="t4_continuation", expected_sessions=None, anchor_state=None, preflight=None)
