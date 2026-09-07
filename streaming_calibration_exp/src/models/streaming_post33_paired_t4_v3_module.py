"""T4 Lightning wrapper requiring an exact paired SPINT completion receipt."""
from __future__ import annotations

from typing import Any

from src.models.streaming_calibration_module import StreamingCalibrationLitModule
from src.utils.post33_paired_teacher_v3 import resolve_paired_spint_teacher


class M2Post33PairedT4LitModuleV3(StreamingCalibrationLitModule):
    """No teacher path argument and therefore no global/default fallback."""

    def __init__(
        self,
        *,
        paired_spint_completion_receipt: str,
        loso_fold: int,
        seed: int,
        **kwargs: Any,
    ) -> None:
        if not paired_spint_completion_receipt or paired_spint_completion_receipt == "???":
            raise ValueError("paired_spint_completion_receipt is mandatory")
        if "teacher_ckpt_path" in kwargs:
            raise ValueError("manual teacher_ckpt_path is forbidden; use the matching SPINT receipt")
        teacher = resolve_paired_spint_teacher(
            paired_spint_completion_receipt, loso_fold=loso_fold, seed=seed
        )
        super().__init__(teacher_ckpt_path=str(teacher), **kwargs)
        self.paired_spint_completion_receipt = paired_spint_completion_receipt
        self.protocol_loso_fold = loso_fold
        self.protocol_seed = seed

