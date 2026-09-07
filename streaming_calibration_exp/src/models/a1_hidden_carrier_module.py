"""Production Lightning integration for the A1 hidden-space carrier.

Only the A1 student is replaced.  The shared streaming SPINT model, teacher,
data path, and parent training/evaluation logic are reused unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, Literal

import torch

from src.models.components.streaming_spint_hidden_carrier_adapter import (
    HiddenSpaceCarrierStreamingSpint,
)
from src.models.streaming_calibration_module import (
    StreamingCalibrationLitModule,
)


class A1HiddenCarrierLitModule(StreamingCalibrationLitModule):
    """Jointly train matched activity identity, decoder, and hidden ``P``.

    The production A1 source family is H/T4 only.  W/T4 and W/Z4 are sealed
    A2 reuse cells, while H/Z4 is their exact structural alias and must not be
    trained as a separate family.
    """

    def __init__(
        self,
        *args: Any,
        a1_attachment_mode: Literal["aligned", "shuffled"] = "aligned",
        a1_attachment_permutation_seed: int | None = None,
        a1_carrier_dim: int = 4,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("task") != "mc_maze":
            raise ValueError("A1 is restricted to the SUA center-out task='mc_maze'")
        if kwargs.get("variant") != "B3S":
            raise ValueError("A1 requires the matched A2 B3S activity identity")
        if int(kwargs.get("side_dim", 0)) != 4 or int(a1_carrier_dim) != 4:
            raise ValueError("A1 is frozen to the four-coordinate T4 carrier")
        if kwargs.get("identity_mode", "calibrated") != "calibrated":
            raise ValueError("A1 requires calibrated activity identity")
        if kwargs.get("decoder_mode", "coupled") != "coupled":
            raise ValueError("A1 requires the ordinary coupled SPINT decoder")
        if int(kwargs.get("fixed_slot_count", 0)) != 0:
            raise ValueError("A1 requires explicit per-unit decoder tokens")
        if kwargs.get("loss_mode", "task_plus_y_plus_E") != "task_only":
            raise ValueError("A1 source training is frozen to loss_mode='task_only'")
        if float(kwargs.get("lambda_y", 1.0)) != 0.0:
            raise ValueError("A1 task-only source training requires lambda_y=0")
        if float(kwargs.get("lambda_E", 0.1)) != 0.0:
            raise ValueError("A1 task-only source training requires lambda_E=0")
        if kwargs.get("identity_mode", "calibrated") != "calibrated":
            raise ValueError("A1 forbids learned-prior identity")
        if bool(kwargs.get("tune_encoder_fusion", False)):
            raise ValueError("A1 does not open a fusion-only optimizer policy")
        if bool(kwargs.get("compile", False)):
            raise ValueError("A1 requires an inspectable uncompiled student")
        if float(kwargs.get("support_prediction_consistency_weight", 0.0)) != 0.0:
            raise ValueError("A1 does not open support-consistency training")
        if float(kwargs.get("ssc_t4_prediction_consistency_weight", 0.0)) != 0.0:
            raise ValueError("A1 does not open SSC-T4 training")
        if a1_attachment_mode != "aligned":
            raise ValueError(
                "A1 source checkpoints must train aligned T4; use the student's "
                "evaluation-only attachment control for same-checkpoint TS4"
            )
        if a1_attachment_permutation_seed is not None:
            raise ValueError("aligned A1 attachment forbids a permutation seed")

        self._a1_attachment_mode = a1_attachment_mode
        self._a1_attachment_permutation_seed = a1_attachment_permutation_seed
        self._a1_carrier_dim = int(a1_carrier_dim)
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(
            {
                "a1_attachment_mode": a1_attachment_mode,
                "a1_attachment_permutation_seed": a1_attachment_permutation_seed,
                "a1_carrier_dim": self._a1_carrier_dim,
                "a1_hidden_map_initialization": "direct_zero_no_rng",
                "a1_hidden_map_bias": None,
            }
        )

    def setup(self, stage: str) -> None:
        if isinstance(self.student, HiddenSpaceCarrierStreamingSpint):
            return
        super().setup(stage)
        if self.student is None:
            raise RuntimeError("parent setup did not construct the A1 substrate")
        substrate = self.student
        was_decoder_frozen = substrate._decoder_frozen
        adapter = HiddenSpaceCarrierStreamingSpint(
            decoder=substrate.decoder,
            id_encoder=substrate.id_encoder,
            add_site="hidden",
            attachment_mode=self._a1_attachment_mode,
            permutation_seed=self._a1_attachment_permutation_seed,
            carrier_dim=self._a1_carrier_dim,
        )
        if was_decoder_frozen:
            adapter.freeze_decoder()
        self.student = adapter

    @staticmethod
    def _validate_optimizer_coverage(
        student: HiddenSpaceCarrierStreamingSpint,
        optimizer: torch.optim.Optimizer,
        *,
        freeze_decoder: bool,
    ) -> None:
        expected_named = [
            (name, parameter)
            for name, parameter in student.named_parameters()
            if parameter.requires_grad
        ]
        if freeze_decoder and any(
            name.startswith("decoder.") for name, _ in expected_named
        ):
            raise RuntimeError("freeze_decoder=true left trainable decoder tensors")
        expected_ids = [id(parameter) for _, parameter in expected_named]
        observed = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        observed_ids = [id(parameter) for parameter in observed]
        missing_ids = set(expected_ids) - set(observed_ids)
        extra_ids = set(observed_ids) - set(expected_ids)
        duplicate_count = len(observed_ids) - len(set(observed_ids))
        if missing_ids or extra_ids or duplicate_count:
            names = {id(parameter): name for name, parameter in expected_named}
            missing = sorted(names[parameter_id] for parameter_id in missing_ids)
            raise RuntimeError(
                "A1 optimizer must contain every trainable student tensor exactly once; "
                f"missing={missing}, unexpected={len(extra_ids)}, "
                f"duplicates={duplicate_count}"
            )
        projection = student.hidden_carrier_map
        if projection is None or id(projection.weight) not in set(observed_ids):
            raise RuntimeError("A1 P is absent from the optimizer")

    def configure_optimizers(self) -> Dict[str, Any]:
        assert isinstance(self.student, HiddenSpaceCarrierStreamingSpint)
        if self._freeze_decoder:
            parameters = list(self.student.trainable_encoder_parameters())
        else:
            parameters = [
                parameter
                for parameter in self.student.parameters()
                if parameter.requires_grad
            ]
        optimizer = self._optimizer_factory(params=parameters)
        self._validate_optimizer_coverage(
            self.student,
            optimizer,
            freeze_decoder=self._freeze_decoder,
        )
        if self._scheduler_factory is not None:
            scheduler = self._scheduler_factory(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_heldin/r2_mean",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
