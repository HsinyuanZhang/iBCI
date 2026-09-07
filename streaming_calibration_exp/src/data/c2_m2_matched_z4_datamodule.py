"""C2's matched T4-normalized, model-invisible Z4 control.

Additive wrapper: ordinary ``side_feature_group=t4`` fit and source
normalizer, then ``zeros_like`` on the standardized side tensor.  This module
is C2-only and must not be imported by B1, A1, or C1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.data.falcon_datamodule import FalconDataModule


class _MaskStandardizedT4Dataset:
    def __init__(self, base: Any) -> None:
        self._base = base

    def __len__(self) -> int:
        return len(self._base)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        row = self._base[index]
        if not isinstance(row, tuple) or len(row) not in (5, 6):
            raise RuntimeError("C2 Z4 expects ordinary T4 dataset rows with side features at position 4")
        side = np.asarray(row[4])
        if side.ndim != 2 or side.shape[-1] != 4:
            raise RuntimeError(f"C2 Z4 expected standardized T4 side shape [N,4], got {side.shape}")
        return (*row[:4], np.zeros_like(side), *row[5:])


class C2M2MatchedZ4DataModule(FalconDataModule):
    """M2-only FalconDataModule whose final standardized T4 tensor is masked."""

    def __init__(self, *args: Any, c2_z4_mask_after_standardization: bool = True, **kwargs: Any) -> None:
        if not c2_z4_mask_after_standardization:
            raise ValueError("C2M2MatchedZ4DataModule requires post-standardization masking")
        if str(kwargs.get("task", "")).lower() != "m2":
            raise ValueError("C2M2MatchedZ4DataModule is frozen to task='m2'")
        if str(kwargs.get("side_feature_group", "")).lower() != "t4":
            raise ValueError("C2 Z4 must construct ordinary T4 before masking")
        if bool(kwargs.get("include_heldout_in_fit", False)) or bool(kwargs.get("include_heldout_in_test", False)):
            raise ValueError("C2 Z4 is development-internal only; held-out files must remain disabled")
        self._c2_z4_mask_after_standardization = True
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
        manifest["c2_z4_control"] = {
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
