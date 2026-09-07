"""Lightning module: B3 activity encoder + independent post-fc_in P."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from src.models.streaming_calibration_module import StreamingCalibrationLitModule

from . import injection
from . import plan


class RSyn3IndependentInjectionModule(StreamingCalibrationLitModule):
    """Joint decoder+B3+P training. Encoder never consumes the carrier."""

    def setup(self, stage: str) -> None:
        super().setup(stage)
        assert self.student is not None
        injection.attach_projection(self.student)
        if str(self._variant).upper() != "B3":
            raise RuntimeError("EMG-rSyn3 Stage-1 requires variant B3")
        if int(self._side_dim) != 0:
            raise RuntimeError("EMG-rSyn3 Stage-1 requires side_dim=0")

    def model_step(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
        if len(batch) != 5:
            raise ValueError("rSyn3 dataloader must provide [neural,target,calib,session,carrier]")
        neural, behavior_target, calib, session_name, carrier = batch
        assert self.student is not None
        y_student, e_student = self.student(
            neural,
            calib_trials=calib,
            side_features=None,
            carrier=carrier,
        )
        y_student, behavior_target = self._slice_last_timestep(y_student, behavior_target)
        loss = self.mse_loss(y_student, behavior_target)
        nan = torch.tensor(float("nan"), device=loss.device)
        return {
            "loss": loss,
            "behavior_pred": y_student,
            "behavior_target": behavior_target,
            "session_name": session_name,
            "identity_mse": nan,
            "prediction_distill_mse": nan,
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        assert self.student is not None
        if self._freeze_decoder:
            raise RuntimeError("rSyn3 Stage-1 is frozen to freeze_decoder=false")
        parameters = [parameter for parameter in self.student.parameters() if parameter.requires_grad]
        optimizer = self._optimizer_factory(params=parameters)
        expected = {
            id(parameter) for parameter in self.student.decoder.parameters() if parameter.requires_grad
        } | {
            id(parameter) for parameter in self.student.id_encoder.parameters() if parameter.requires_grad
        } | {id(self.student.carrier_projection_weight)}
        observed = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if expected != observed:
            raise RuntimeError(
                "rSyn3 optimizer must contain decoder+encoder+P exactly once; "
                f"expected={len(expected)}, observed={len(observed)}"
            )
        return {"optimizer": optimizer}


def make_module(repo_root: Path) -> RSyn3IndependentInjectionModule:
    teacher = Path(repo_root) / plan.TEACHER_CKPT_RELATIVE
    if not teacher.is_file():
        raise FileNotFoundError(teacher)
    return RSyn3IndependentInjectionModule(
        task="m1",
        variant="B3",
        teacher_ckpt_path=str(teacher),
        window_size=plan.WINDOW_SIZE,
        trial_length=plan.TRIAL_LENGTH,
        freeze_decoder=False,
        loss_mode="task_only",
        lambda_y=0.0,
        lambda_E=0.0,
        decode_last_timestep_only=True,
        predict_scaled_behavior=False,
        behavior_scaling_factor=1.0,
        hidden_dim=64,
        side_dim=0,
        optimizer=partial(torch.optim.Adam, lr=plan.STAGE1_LR, weight_decay=plan.STAGE1_WEIGHT_DECAY),
        scheduler=None,
    )
