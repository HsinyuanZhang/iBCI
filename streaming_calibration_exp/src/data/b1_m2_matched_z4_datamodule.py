"""B1's matched T4-normalized, model-invisible Z4 control.

This module is deliberately an *additive wrapper*, rather than a change to
``FalconDataModule``.  The wrapped data module is configured as ordinary
``side_feature_group=t4``: it reads only calibration-trial direction labels,
fits the ordinary source-session T4 normalizer, computes the standardized
``[a,c,m,b]`` feature, and passes that result to this wrapper.  The wrapper
then replaces exactly that final side-feature tensor with zeros.

Consequently T4 and Z4 have identical label-disclosure, carrier-fit,
normalizer, activity-calibration, query, and tensor-width paths.  Z4 is *not*
the historical ``zero4`` no-label width control.  It is a matched
carrier-content null for the B1 factorial only.

No decoder/model/trainer code is changed here.  This module must never be
used outside the B1 M2 development-only contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.data.falcon_datamodule import FalconDataModule


class _MaskStandardizedT4Dataset:
    """Delegate every dataset attribute but zero only its final side tensor."""

    def __init__(self, base: Any) -> None:
        self._base = base

    def __len__(self) -> int:
        return len(self._base)

    def __getattr__(self, name: str) -> Any:
        # SessionBatchSampler and provenance code consume several FalconDataset
        # attributes (notably ``window_indices``).  Delegating preserves the
        # original activity/query/session geometry exactly.
        return getattr(self._base, name)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        row = self._base[index]
        if not isinstance(row, tuple) or len(row) not in (5, 6):
            raise RuntimeError(
                "B1 Z4 expects ordinary T4 dataset rows with side features "
                "at position 4"
            )
        side = np.asarray(row[4])
        if side.ndim != 2 or side.shape[-1] != 4:
            raise RuntimeError(
                f"B1 Z4 expected standardized T4 side shape [N,4], got {side.shape}"
            )
        # ``zeros_like`` intentionally occurs after the base T4 path has
        # already computed and standardized the descriptor.  Preserve dtype
        # and do not mutate cached base values.
        return (*row[:4], np.zeros_like(side), *row[5:])


class B1M2MatchedZ4DataModule(FalconDataModule):
    """M2-only FalconDataModule whose final standardized T4 tensor is masked."""

    def __init__(self, *args: Any, b1_z4_mask_after_standardization: bool = True, **kwargs: Any) -> None:
        if not b1_z4_mask_after_standardization:
            raise ValueError("B1M2MatchedZ4DataModule requires post-standardization masking")
        if str(kwargs.get("task", "")).lower() != "m2":
            raise ValueError("B1M2MatchedZ4DataModule is frozen to task='m2'")
        if str(kwargs.get("side_feature_group", "")).lower() != "t4":
            raise ValueError(
                "B1 Z4 must construct ordinary T4 before masking; "
                "side_feature_group must be 't4'"
            )
        if bool(kwargs.get("include_heldout_in_fit", False)) or bool(kwargs.get("include_heldout_in_test", False)):
            raise ValueError("B1 Z4 is development-internal only; held-out files must remain disabled")
        # Keep the marker local to this additive subclass: FalconDataModule's
        # constructor intentionally has no B1-specific argument.
        self._b1_z4_mask_after_standardization = True
        # ``FalconDataModule`` records inherited constructor arguments through
        # Lightning's hyperparameter capture.  When this subclass is built by
        # Hydra, that capture observes ``kwargs`` rather than the base class's
        # local ``data_dir = Path(data_dir)`` binding.  Normalize before the
        # inherited constructor so its deliberate ``.rglob`` file-scope checks
        # always receive a ``Path``.  This is a type-only repair: the resolved
        # directory, T4 fit, source normalizer, and post-standardization Z4
        # mask are unchanged.
        if "data_dir" in kwargs:
            kwargs["data_dir"] = Path(kwargs["data_dir"])
        super().__init__(*args, **kwargs)

    @staticmethod
    def _wrap_once(dataset: Any) -> Any:
        return dataset if isinstance(dataset, _MaskStandardizedT4Dataset) else _MaskStandardizedT4Dataset(dataset)

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)
        for attr in ("train_dataset", "val_heldin_dataset", "val_heldout_dataset"):
            dataset = getattr(self, attr, None)
            if dataset is not None:
                setattr(self, attr, self._wrap_once(dataset))

    def get_split_manifest(self) -> dict[str, Any]:
        manifest = super().get_split_manifest()
        manifest["b1_z4_control"] = {
            "carrier_fit_executed": True,
            "calibration_direction_labels_read": True,
            "source_t4_normalizer_fit": True,
            "target_t4_normalizer_refit": False,
            "mask_operation": "zeros_like(standardized_t4_[a,c,m,b])",
            "model_visible_side_feature": "all_zero_[N,4]",
            "query_target_labels_used_for_carrier": False,
            "formal_or_external_heldout_files_opened": False,
        }
        return manifest
