from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.metrics.run_artifacts import _teacher_exposure_attestation
from src.models.components.streaming_encoders import (
    BatchReferenceEncoder,
    SideFeatureEarlyPoolEncoder,
)
from sua_exploration.m1_compact_replication import fold0_forward_authority as authority
from sua_exploration.m1_compact_replication import prepare_proposals as proposals


class _Sampler:
    def __init__(self, batches):
        self.batched_indices = batches


class _Dataset:
    def __init__(self):
        self.window_indices = [("ses-a", 10), ("ses-a", 11), ("ses-b", 2)]


def test_ordered_sampler_receipt_binds_order_and_semantic_windows():
    dataset = _Dataset()
    first = authority.ordered_sampler_receipt(_Sampler([[0, 1], [2]]), dataset, label="x")
    second = authority.ordered_sampler_receipt(_Sampler([[1, 0], [2]]), dataset, label="x")
    assert first["samples"] == 3
    assert first["ordered_sampler_indices_sha256"] != second["ordered_sampler_indices_sha256"]
    assert first["ordered_window_identity_sha256"] != second["ordered_window_identity_sha256"]
    with pytest.raises(authority.ForwardAuthorityError, match="repeats"):
        authority.ordered_sampler_receipt(_Sampler([[0, 1], [1]]), dataset, label="x")


def test_regression_metrics_use_per_output_sse_tss_and_pooled_definition():
    target = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    prediction = target + np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    receipt = authority.regression_metrics(prediction, target)
    assert receipt["sse_float64_per_output"] == [2.0, 1.0]
    assert receipt["tss_float64_per_output"] == [8.0, 8.0]
    assert receipt["r2_per_output"] == [0.75, 0.875]
    assert receipt["pooled_variance_weighted_r2"] == 0.8125


def test_b0_analytic_mac_is_nonzero_and_matches_affine_algebra():
    encoder = BatchReferenceEncoder(
        nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU()),
        nn.Sequential(nn.Linear(4, 3), nn.ReLU()),
        window_size=3,
    )
    receipt = authority.analytic_identity_mac(
        encoder, num_units=2, calibration_trials=5, trial_length=8,
    )
    # Per unit/trial: 8*4 + 4*4 = 48. Finalize per unit: 4*3 = 12.
    assert receipt["mac_per_trial"] == 96
    assert receipt["mac_per_session"] == 504
    assert receipt["profiler_measurement_collected"] is False


def test_mac_uses_loaded_weight_shape_not_stale_lazy_linear_metadata():
    encoder = BatchReferenceEncoder(
        nn.Sequential(nn.Linear(8, 4), nn.ReLU()),
        nn.Sequential(nn.Linear(4, 3)),
        window_size=3,
    )
    # Archived B0 has exactly this kind of stale metadata after strict loading:
    # the tensor is materialized while ``in_features`` remains zero.
    encoder.fc_id_in[0].in_features = 0
    receipt = authority.analytic_identity_mac(
        encoder, num_units=2, calibration_trials=5, trial_length=8,
    )
    assert receipt["mac_per_trial"] == 64
    assert receipt["mac_per_session"] == 344


def test_zero4_column_pruning_is_exact_for_zero_side_input():
    torch.manual_seed(7)
    encoder = SideFeatureEarlyPoolEncoder(
        trial_length=8, window_size=3, hidden_dim=4, side_dim=4,
    ).eval()
    calibration = torch.randn(2, 5, 8, 3)
    zero = torch.zeros(2, 3, 4)
    expected = encoder.forward_batch(calibration, side_features=zero)
    observed = authority._pruned_zero4_identity(encoder, calibration)
    assert torch.equal(observed, expected)


def test_teacher_exposure_is_unknown_unless_split_owner_attests_it():
    assert _teacher_exposure_attestation({}) == {
        "teacher_seen_validation_session": None,
        "teacher_seen_validation_session_attested": False,
    }
    assert _teacher_exposure_attestation({"teacher_seen_validation_session": False}) == {
        "teacher_seen_validation_session": False,
        "teacher_seen_validation_session_attested": True,
    }
    with pytest.raises(ValueError, match="boolean or null"):
        _teacher_exposure_attestation({"teacher_seen_validation_session": "unknown"})


def _synthetic_resume_checkpoint(path: Path, *, variant: str = "B0") -> None:
    optimizer = {
        "state": {
            0: {
                "step": torch.tensor(float(proposals.EXPECTED_GLOBAL_STEP)),
                "exp_avg": torch.zeros(1),
                "exp_avg_sq": torch.zeros(1),
            },
        },
        "param_groups": [{
            "lr": 1.0e-4, "weight_decay": 0.0, "params": [0],
        }],
    }
    payload = {
        "epoch": proposals.EXPECTED_RESUME_EPOCH,
        "global_step": proposals.EXPECTED_GLOBAL_STEP,
        "pytorch-lightning_version": proposals.EXPECTED_LIGHTNING,
        "state_dict": {"student.weight": torch.ones(1)},
        "hyper_parameters": {"variant": variant},
        "optimizer_states": [optimizer],
        "lr_schedulers": [],
        "loops": {"fit_loop": {
            "epoch_loop.batch_progress": {
                "total": {"completed": proposals.EXPECTED_GLOBAL_STEP},
            },
            "epoch_progress": {"total": {"processed": 12}},
        }},
        "callbacks": {
            "ModelCheckpoint{'monitor': None, 'every_n_epochs': 12}": {},
        },
    }
    torch.save(payload, path)


def test_resume_validator_requires_optimizer_loop_and_callback_state(tmp_path: Path):
    checkpoint = tmp_path / "resume.ckpt"
    _synthetic_resume_checkpoint(checkpoint)
    receipt = proposals.validate_resume_checkpoint(checkpoint, expected_variant="B0")
    assert receipt["strict_resume_possible"] is True
    assert receipt["optimizer"]["optimizer_step"] == proposals.EXPECTED_GLOBAL_STEP

    broken = torch.load(checkpoint, map_location="cpu", weights_only=False)
    broken["lr_schedulers"] = [{"state_dict": {}}]
    torch.save(broken, checkpoint)
    with pytest.raises(proposals.ProposalError, match="scheduler"):
        proposals.validate_resume_checkpoint(checkpoint, expected_variant="B0")


def test_source_only_teacher_validator_rejects_target_in_source_manifest(tmp_path: Path):
    fold = 1
    target, sources = proposals.M1_SESSIONS[fold]
    checkpoint = tmp_path / "teacher.ckpt"
    torch.save({"epoch": 19}, checkpoint)
    manifest_path = tmp_path / "manifest.json"
    base = {
        "task": "m1", "outer_fold": fold, "outer_left_out": target,
        "source_only": True, "target_backpropagation": False,
        "heldout_opened": False, "minival_opened": False,
        "formal": False, "evalai": False,
        "train_sessions": list(sources), "validation_sessions": [],
        "source_files": {name: {"sha256": name} for name in sources},
    }
    manifest_path.write_text(__import__("json").dumps(base), encoding="utf-8")
    spec = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": proposals.sha256_file(checkpoint),
        "manifest": manifest_path,
        "manifest_sha256": proposals.sha256_file(manifest_path),
    }
    assert proposals.validate_source_only_teacher(fold, spec)[
        "outer_target_excluded_from_source_files"
    ] is True

    contaminated = copy.deepcopy(base)
    contaminated["source_files"][target] = {"sha256": "bad"}
    contaminated["train_sessions"].append(target)
    manifest_path.write_text(__import__("json").dumps(contaminated), encoding="utf-8")
    spec["manifest_sha256"] = proposals.sha256_file(manifest_path)
    with pytest.raises(proposals.ProposalError, match="source list|includes target"):
        proposals.validate_source_only_teacher(fold, spec)


def test_real_proposal_is_inert_and_has_four_conditional_cross_fold_cells():
    body = proposals.build_proposals()
    assert body["status"] == proposals.STATUS
    assert body["execution"]["gpu_launched"] is False
    assert body["fold0_e23_continuation"]["gate"]["threshold"] == -0.03
    cells = body["folds1_2_cross_session"]["cells"]
    assert {(cell["fold"], cell["arm"]) for cell in cells} == {
        (1, "b0"), (1, "b3s_zero4"), (2, "b0"), (2, "b3s_zero4"),
    }
    assert all(cell["argv"][-1] == "ckpt_path=null" for cell in cells)
