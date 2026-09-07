"""Additive same-fold MATCHED_ERM full-training successor for CS-WG M1."""

from .full_train import (
    MATCHED_ERM_FULL_ROOT_RELATIVE,
    build_matched_erm_full_identity,
    execute_reviewed_matched_erm_full_training,
    issue_root_reviewed_matched_erm_full_capability,
    matched_erm_full_training_spec,
)

__all__ = (
    "MATCHED_ERM_FULL_ROOT_RELATIVE", "matched_erm_full_training_spec",
    "build_matched_erm_full_identity", "issue_root_reviewed_matched_erm_full_capability",
    "execute_reviewed_matched_erm_full_training",
)
