"""Dedicated Lightning wrapper for swap-v2; shared production modules stay untouched."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from mc_maze.misleading_identity_swap_v2_core import load_verified_runtime_authority
from src.models.components.misleading_identity_swap_v2_encoder import (
    MisleadingIdentitySwapV2Encoder,
    RuntimePhase,
)
from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder
from src.models.streaming_calibration_module import StreamingCalibrationLitModule


IdentityTrainingMode = Literal["clean", "swap"]
EvaluationInputMode = Literal["clean", "swapped_diagnostic"]


class MisleadingIdentitySwapV2LitModule(StreamingCalibrationLitModule):
    """Opt-in wrapper that configures one B3S mapping before every forward."""

    def __init__(
        self,
        *args: Any,
        matching_authority_path: str,
        matching_authority_kind: Literal["source", "target_diagnostic"] = "source",
        identity_training_mode: IdentityTrainingMode,
        evaluation_input_mode: EvaluationInputMode = "clean",
        **kwargs: Any,
    ) -> None:
        if identity_training_mode not in {"clean", "swap"}:
            raise ValueError("identity_training_mode must be clean or swap")
        if evaluation_input_mode not in {"clean", "swapped_diagnostic"}:
            raise ValueError("evaluation_input_mode must be clean or swapped_diagnostic")
        if kwargs.get("variant") != "B3S":
            raise ValueError("misleading-identity swap-v2 is restricted to ordinary B3S")
        if kwargs.get("loss_mode") != "task_only":
            raise ValueError("misleading-identity swap-v2 requires task_only loss")
        super().__init__(*args, **kwargs)
        self._swap_v2_authority = load_verified_runtime_authority(
            Path(matching_authority_path), expected_kind=matching_authority_kind
        )
        self._swap_v2_authority_kind = matching_authority_kind
        self._swap_v2_identity_training_mode = identity_training_mode
        self._swap_v2_evaluation_input_mode = evaluation_input_mode
        self._swap_v2_explicit_evaluation_epoch: int | None = None

    def setup(self, stage: str) -> None:
        super().setup(stage)
        assert self.student is not None
        encoder = self.student.id_encoder
        if isinstance(encoder, MisleadingIdentitySwapV2Encoder):
            if encoder.matching_authority_sha256 != self._swap_v2_authority.sha256:
                raise RuntimeError("attached swap-v2 encoder authority drift")
            return
        if type(encoder) is not SideFeatureEarlyPoolEncoder:
            raise TypeError("swap-v2 setup requires the ordinary production B3S encoder")
        self.student.id_encoder = MisleadingIdentitySwapV2Encoder.from_parent(
            encoder,
            authority=self._swap_v2_authority,
        )

    def set_evaluation_input_mode(
        self,
        mode: EvaluationInputMode,
        *,
        checkpoint_lightning_current_epoch: int,
    ) -> None:
        if mode not in {"clean", "swapped_diagnostic"}:
            raise ValueError("evaluation input mode drift")
        if checkpoint_lightning_current_epoch < 0 or checkpoint_lightning_current_epoch > 11:
            raise ValueError("evaluation checkpoint epoch must be zero-based 0..11")
        self._swap_v2_evaluation_input_mode = mode
        self._swap_v2_explicit_evaluation_epoch = int(checkpoint_lightning_current_epoch)

    @staticmethod
    def _single_session(batch: Sequence[Any]) -> str:
        if len(batch) < 4:
            raise ValueError("swap-v2 batch lacks session names")
        field = batch[3]
        if isinstance(field, str):
            names = [field]
        elif isinstance(field, Sequence):
            names = [str(value) for value in field]
        else:
            raise ValueError("swap-v2 session field is malformed")
        if not names or len(set(names)) != 1 or not names[0]:
            raise ValueError("swap-v2 requires a single nonempty session before forward")
        return names[0]

    def _runtime_phase_and_epoch(self) -> tuple[RuntimePhase, int]:
        if self.training:
            phase: RuntimePhase = (
                "train_swap"
                if self._swap_v2_identity_training_mode == "swap"
                else "train_clean"
            )
            return phase, int(self.current_epoch)
        phase = (
            "eval_swapped_diagnostic"
            if self._swap_v2_evaluation_input_mode == "swapped_diagnostic"
            else "eval_clean"
        )
        if self._swap_v2_explicit_evaluation_epoch is not None:
            return phase, self._swap_v2_explicit_evaluation_epoch
        # Lightning's source validation *and its pre-fit sanity validation*
        # are internal training-time clean evaluation.  In Lightning 2.4 the
        # latter sets ``sanity_checking`` before ``validating`` becomes true;
        # treating it as stand-alone scoring wrongly demands a checkpoint
        # epoch before epoch zero.  Neither branch may use the swapped
        # diagnostic phase: the diagnostic is only a separately authorised
        # post-training scorer mode.
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and (
            bool(getattr(trainer, "validating", False)) or
            bool(getattr(trainer, "sanity_checking", False))
        ):
            if phase != "eval_clean":
                raise RuntimeError("training-time sanity/validation may not enable swapped diagnostic input")
            epoch = getattr(trainer, "current_epoch", None)
            if epoch is None:
                epoch = self.current_epoch
            return phase, int(epoch)
        raise RuntimeError("stand-alone swap-v2 evaluation requires an explicit checkpoint epoch")

    def model_step(self, batch):
        # This guard intentionally runs before the production forward, unlike the
        # shared module's post-forward training_step assertion.
        session = self._single_session(batch)
        phase, epoch = self._runtime_phase_and_epoch()
        assert self.student is not None
        encoder = self.student.id_encoder
        if not isinstance(encoder, MisleadingIdentitySwapV2Encoder):
            raise RuntimeError("swap-v2 encoder wrapper is not attached")
        encoder.configure_runtime(
            session=session,
            lightning_current_epoch=epoch,
            phase=phase,
        )
        return super().model_step(batch)

    def intervention_receipt(self) -> dict[str, Any]:
        assert self.student is not None
        encoder = self.student.id_encoder
        if not isinstance(encoder, MisleadingIdentitySwapV2Encoder):
            raise RuntimeError("swap-v2 encoder wrapper is not attached")
        return {
            "identity_training_mode": self._swap_v2_identity_training_mode,
            "evaluation_input_mode": self._swap_v2_evaluation_input_mode,
            "matching_authority_sha256": encoder.matching_authority_sha256,
            "matching_authority_canonical_sha256": self._swap_v2_authority.canonical_sha256,
            "matching_authority_path": self._swap_v2_authority.immutable_file_path,
            "matching_authority_kind": self._swap_v2_authority_kind,
            "last_swap_trace": encoder.last_swap_trace,
            "target_optimizer_or_backward_steps": 0,
            "formal_subc_test_nwb_opened": False,
        }
