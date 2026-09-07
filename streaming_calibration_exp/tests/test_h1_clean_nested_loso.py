from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.falcon_h1_afc4_features import (  # noqa: E402
    H1_NUM_NEURONS,
    H1_VELOCITY_DIM,
    H1AFC4SourcePlan,
    record_from_arrays,
)
from src.data.falcon_h1_clean_nested_loso_datamodule import (  # noqa: E402
    H1AFC4Dataset,
    H1CleanNestedLossoDataModule,
    assert_shared_teacher_binding,
    h1_nested_loso_partition,
)
from src.callbacks.h1_nested_selection import (  # noqa: E402
    audit_h1_teacher_student_monitors,
    build_h1_selection_contract,
    h1_inner_validation_monitor,
    validate_h1_checkpoint_monitor,
)
from src.h1_clean_nested_loso_eval import evaluate_h1_outer_predictor  # noqa: E402
from src.models.h1_teacher_module import nonempty_metric_values  # noqa: E402
from src.models.h1_teacher_module import H1TeacherLitModule  # noqa: E402
from src.models.falcon_module import FalconLitModule  # noqa: E402
from src.models.components.spint import SpintModel  # noqa: E402
from torchmetrics.regression import R2Score  # noqa: E402
import torch  # noqa: E402
import lightning  # noqa: E402


def _record(name: str, *, seed: int = 1):
    rng = np.random.default_rng(seed)
    length = 1600
    velocity = rng.normal(size=(length, H1_VELOCITY_DIM)).astype(np.float64)
    neural = rng.normal(size=(length, H1_NUM_NEURONS)).astype(np.float64)
    trial_num = np.ones(length, dtype=np.float64)
    trial_num[850:] = 2.0
    eval_mask = np.ones(length, dtype=bool)
    # Explicit support gaps make the first trial's segment contract visible.
    velocity[120:123] = 0.0
    velocity[420:425] = 0.0
    trial_change = np.zeros(length, dtype=bool)
    trial_change[0] = True
    return record_from_arrays(
        session_name=name,
        split="calib",
        neural=neural,
        covariates=velocity,
        trial_num=trial_num,
        eval_mask=eval_mask,
        trial_change=trial_change,
    )


def test_h1_nested_partition_has_outer12_inner11_and_no_target_overlap():
    sessions = [f"ses-{index:02d}" for index in range(13)]
    split = h1_nested_loso_partition(sessions, 4)
    assert split.outer_target_session == "ses-04"
    assert len(split.outer_source_sessions) == 12
    assert len(split.inner_train_sessions) == 11
    assert split.inner_validation_session == "ses-05"
    assert split.outer_target_session not in split.inner_train_sessions
    assert split.outer_target_session != split.inner_validation_session


def test_h1_dataset_support_and_query_are_full_window_disjoint():
    support = _record("ses-a")
    query = _record("ses-a", seed=2)
    plan = H1AFC4SourcePlan({"ses-a": support})
    side = plan.descriptor_for(support, "afc4_h1q3")[0]
    dataset = H1AFC4Dataset(
        support_records={"ses-a": support},
        query_records={"ses-a": query},
        side_features={"ses-a": side},
        query_mode="calib_post_support",
        window_size=700,
        max_trial_length=1024,
    )
    assert len(dataset) > 0
    neural, behavior, calib, names, side_batch = dataset[0]
    assert neural.shape == (700, H1_NUM_NEURONS)
    assert behavior.shape == (700, H1_VELOCITY_DIM)
    assert calib.shape == (1, 1024, H1_NUM_NEURONS)
    assert names == "ses-a"
    assert side_batch.shape == (H1_NUM_NEURONS, 4)
    assert dataset.query_window_audit["ses-a"]["full_window_disjoint"] is True


def test_h1_teacher_uses_four_tuple_and_student_uses_five_tuple():
    support = _record("ses-a")
    query = _record("ses-a", seed=2)
    plan = H1AFC4SourcePlan({"ses-a": support})
    side = plan.descriptor_for(support, "afc4_h1q3")[0]
    teacher = H1AFC4Dataset(
        support_records={"ses-a": support},
        query_records={"ses-a": query},
        side_features=None,
        query_mode="calib_post_support",
        include_side_features=False,
        window_size=700,
        max_trial_length=1024,
    )
    student = H1AFC4Dataset(
        support_records={"ses-a": support},
        query_records={"ses-a": query},
        side_features={"ses-a": side},
        query_mode="calib_post_support",
        include_side_features=True,
        window_size=700,
        max_trial_length=1024,
    )
    assert len(teacher[0]) == 4
    assert len(student[0]) == 5


def test_h1_teacher_students_bind_exact_inner_validation_monitor_and_source_plan():
    split = h1_nested_loso_partition([f"ses-{index:02d}" for index in range(13)], 4)
    contract_rows = [
        build_h1_selection_contract(split, arm=arm, role=role).as_dict()
        for arm, role in (("teacher", "teacher"), ("afc4_h1q3", "afc4_h1q3"), ("afc4_h1_b4", "afc4_h1_b4"), ("zero4", "zero4"))
    ]
    audit = audit_h1_teacher_student_monitors(
        contract_rows,
        inner_validation_session=split.inner_validation_session,
        outer_target_session=split.outer_target_session,
    )
    assert audit["monitor"] == h1_inner_validation_monitor(split.inner_validation_session)
    assert audit["outer_target_used"] is False
    with pytest.raises(ValueError, match="exact inner-validation"):
        validate_h1_checkpoint_monitor(
            "val_heldin/r2_mean",
            inner_validation_session=split.inner_validation_session,
            outer_target_session=split.outer_target_session,
        )


def test_h1_full_b4_zero_share_teacher_and_source_plan_sha():
    manifests = {
        arm: {"teacher_checkpoint_sha256": "teacher-sha", "source_plan_sha256": "plan-sha"}
        for arm in ("afc4_h1q3", "afc4_h1_b4", "zero4")
    }
    assert_shared_teacher_binding(manifests) == {
        "teacher_checkpoint_sha256": "teacher-sha",
        "source_plan_sha256": "plan-sha",
    }
    manifests["zero4"]["teacher_checkpoint_sha256"] = "different"
    with pytest.raises(ValueError, match="share one"):
        assert_shared_teacher_binding(manifests)


def test_h1_outer_predictor_is_no_grad_and_supports_teacher_student_arities():
    support = _record("ses-a")
    query = _record("ses-a", seed=2)
    plan = H1AFC4SourcePlan({"ses-a": support})
    side = plan.descriptor_for(support, "afc4_h1q3")[0]
    student_dataset = H1AFC4Dataset(
        support_records={"ses-a": support},
        query_records={"ses-a": query},
        side_features={"ses-a": side},
        query_mode="minival",
        include_side_features=True,
        window_size=700,
        max_trial_length=1024,
    )
    teacher_dataset = H1AFC4Dataset(
        support_records={"ses-a": support},
        query_records={"ses-a": query},
        side_features=None,
        query_mode="minival",
        include_side_features=False,
        window_size=700,
        max_trial_length=1024,
    )
    seen_grad: list[bool] = []

    def student_predictor(neural, calib, side_batch):
        seen_grad.append(bool(neural.requires_grad))
        return neural[..., :H1_VELOCITY_DIM]

    def teacher_predictor(neural, calib):
        seen_grad.append(bool(neural.requires_grad))
        return neural[..., :H1_VELOCITY_DIM]

    student_result = evaluate_h1_outer_predictor(
        student_dataset, student_predictor, include_side_features=True, batch_size=32
    )
    teacher_result = evaluate_h1_outer_predictor(
        teacher_dataset, teacher_predictor, include_side_features=False, batch_size=32
    )
    assert np.isfinite(student_result["r2_variance_weighted"])
    assert np.isfinite(teacher_result["r2_variance_weighted"])
    assert seen_grad and not any(seen_grad)


def test_h1_teacher_metric_helper_skips_empty_heldin_and_heldout_slots():
    populated = R2Score(multioutput="variance_weighted")
    populated.update(torch.tensor([[0.0], [1.0], [2.0], [3.0]]), torch.tensor([[0.0], [1.0], [2.0], [2.0]]))
    empty_heldin = R2Score(multioutput="variance_weighted")
    empty_heldout = R2Score(multioutput="variance_weighted")
    heldin = nonempty_metric_values({"inner": populated, "empty": empty_heldin})
    heldout = nonempty_metric_values({"outer": empty_heldout})
    assert [name for name, _value in heldin] == ["inner"]
    assert heldout == []


def test_h1_teacher_checkpoint_restores_through_falcon_student_loader(tmp_path):
    net = SpintModel(
        model_dim=32,
        num_covariates=7,
        window_size=8,
        num_heads=2,
        num_layers=1,
        num_id_layers=1,
        use_learnable_id=True,
        learnable_id_type="mlp",
        learnable_rep=True,
        dropout_rate=0.0,
        dynamic_dropout=False,
        dynamic_dropout_low=0.0,
        dynamic_dropout_high=1.0,
        tf_drop_rate=0.0,
        readin_layer_type="mlp",
    )
    teacher = H1TeacherLitModule(
        task="h1",
        net=net,
        decode_last_timestep_only=True,
        predict_scaled_behavior=True,
        behavior_scaling_factor=20.0,
        optimizer=torch.optim.Adam,
        scheduler=None,
        compile=False,
    )
    checkpoint = tmp_path / "teacher.ckpt"
    torch.save(
        {
            "state_dict": teacher.state_dict(),
            "hyper_parameters": dict(teacher.hparams),
            "pytorch-lightning_version": lightning.__version__,
        },
        checkpoint,
    )
    loaded = FalconLitModule.load_from_checkpoint(str(checkpoint), weights_only=False)
    assert set(loaded.state_dict()) == set(teacher.state_dict())


@pytest.mark.skipif(
    not (ROOT.parent / "SPINT-main/data/000954").exists(),
    reason="local DANDI 000954 data is unavailable",
)
def test_h1_real_fold0_never_opens_outer_target_during_fit(tmp_path):
    dm = H1CleanNestedLossoDataModule(
        data_dir=str(ROOT.parent / "SPINT-main/data/000954"),
        outer_loso_fold=0,
        batch_size=8,
        session_window_budget=512,
        side_feature_group="afc4_h1q3",
        num_workers=0,
        pin_memory=False,
    )
    dm.setup("fit")
    manifest = dm.get_split_manifest()
    assert len(dm.loaded_fit_sessions) == 12
    assert dm.outer_target_loaded is False
    assert dm.outer_target_query_labels_read is False
    assert manifest["outer_target_loaded_during_fit"] is False
    assert manifest["outer_target_query_labels_read_during_fit"] is False
    assert manifest["source_only_basis_or_normalizer"]["basis"]["fit_sessions"] == list(dm.split.inner_train_sessions)
    assert manifest["checkpoint_selection"]["outer_target_used"] is False
    assert dm.split.outer_target_session not in manifest["loaded_fit_sessions"]
    receipt = dm.write_source_only_manifest(tmp_path / "fit_manifest")
    assert receipt["status"] == "PASS_H1_SOURCE_ONLY_MANIFEST_PRE_OPTIMIZER"
    assert receipt["source_only_manifest_written_before_optimizer"] is True
    assert receipt["checkpoint_selection"]["monitor"] == (
        f"val_heldin_{dm.split.inner_validation_session}/r2"
    )
