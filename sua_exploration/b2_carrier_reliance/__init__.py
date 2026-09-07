"""Default-off B2 carrier-reliance contract helpers.

This package contains no trainer, launcher, data reader, or artifact writer.
It freezes the proposed M2 development schedule, staged matrix, and paired
estimand without changing the B1-bound streaming runtime.
"""

from .contract import (
    DEPLOYMENT_NONINFERIORITY_MARGIN,
    EPOCH_WINDOW,
    EVAL_ARMS,
    FOLDS,
    FOLD_TO_SESSION,
    P_PLUS_F,
    PRACTICAL_EFFECT_FLOOR,
    SEEDS,
    STAGE_F,
    STAGE_F_SEEDS,
    STAGE_P,
    STAGE_P_SEEDS,
    TRAIN_ARMS,
    StageFNotAuthorizedError,
    aggregate_descriptive_p_plus_f,
    aggregate_stage_f,
    aggregate_stage_p,
    expected_score_keys,
    expected_training_cells,
    stage_seeds,
    validate_stage_p_result,
)
from .corruption import (
    LOGICAL_EPOCHS,
    TRAINING_SCHEDULE_COUNTS,
    apply_scheduled_training_corruption,
    apply_training_corruption,
    complete_row_derangement,
    derange_finite_angles,
    rs4_row_mapping,
    training_kind_for_epoch,
    training_schedule,
)

__all__ = [
    "DEPLOYMENT_NONINFERIORITY_MARGIN",
    "EPOCH_WINDOW",
    "EVAL_ARMS",
    "FOLDS",
    "FOLD_TO_SESSION",
    "LOGICAL_EPOCHS",
    "P_PLUS_F",
    "PRACTICAL_EFFECT_FLOOR",
    "SEEDS",
    "STAGE_F",
    "STAGE_F_SEEDS",
    "STAGE_P",
    "STAGE_P_SEEDS",
    "TRAIN_ARMS",
    "TRAINING_SCHEDULE_COUNTS",
    "StageFNotAuthorizedError",
    "aggregate_descriptive_p_plus_f",
    "aggregate_stage_f",
    "aggregate_stage_p",
    "apply_scheduled_training_corruption",
    "apply_training_corruption",
    "complete_row_derangement",
    "derange_finite_angles",
    "expected_score_keys",
    "expected_training_cells",
    "rs4_row_mapping",
    "stage_seeds",
    "training_kind_for_epoch",
    "training_schedule",
    "validate_stage_p_result",
]
