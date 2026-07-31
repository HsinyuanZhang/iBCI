"""Lightning selector for a baseline-preserving coupled T4 key residual."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch

from src.models.components.streaming_spint_t4_key_residual_adapter import (
    CoupledT4KeyResidualStreamingSpint,
)
from src.models.streaming_calibration_module import (
    StreamingCalibrationLitModule,
)


class T4KeyResidualLitModule(StreamingCalibrationLitModule):
    """Warm-start selected T4 and train only a direct key residual.

    The parent first restores the complete selected ordinary-T4 student
    (decoder and B3S identity encoder). ``setup`` then wraps that exact
    substrate and freezes it. The first policy trains only the new residual;
    the single predeclared optimization policy additionally trains the
    teacher attention output projection.
    """

    def __init__(
        self,
        *,
        residual_mode: Literal["aligned", "shuffled"],
        residual_rank: int = 8,
        residual_permutation_seed: int | None = None,
        residual_training_policy: Literal[
            "residual_only",
            "residual_plus_attention_out",
        ] = "residual_only",
        **kwargs: Any,
    ) -> None:
        if residual_mode not in {"aligned", "shuffled"}:
            raise ValueError(
                "residual_mode must be 'aligned' or 'shuffled'"
            )
        if residual_rank <= 0:
            raise ValueError("residual_rank must be positive")
        if residual_training_policy not in {
            "residual_only",
            "residual_plus_attention_out",
        }:
            raise ValueError(
                "unsupported residual_training_policy"
            )
        if (
            residual_mode == "shuffled"
            and residual_permutation_seed is None
        ):
            raise ValueError(
                "shuffled residual requires a permutation seed"
            )
        if (
            residual_mode == "aligned"
            and residual_permutation_seed is not None
        ):
            raise ValueError(
                "aligned residual forbids a permutation seed"
            )
        if kwargs.get("variant") != "B3S":
            raise ValueError(
                "T4 key residual requires variant='B3S'"
            )
        if int(kwargs.get("side_dim", 0)) != 4:
            raise ValueError(
                "T4 key residual requires side_dim=4"
            )
        if kwargs.get(
            "identity_mode", "calibrated"
        ) != "calibrated":
            raise ValueError(
                "T4 key residual requires calibrated identity"
            )
        if int(kwargs.get("fixed_slot_count", 0)) != 0:
            raise ValueError(
                "T4 key residual forbids fixed slots"
            )
        if kwargs.get("decoder_mode", "coupled") != "coupled":
            raise ValueError(
                "T4 key residual owns decoder selection; "
                "base mode must be coupled"
            )
        if bool(kwargs.get("compile", False)):
            raise ValueError(
                "T4 key residual does not support compile"
            )
        if bool(kwargs.get("freeze_decoder", False)):
            raise ValueError(
                "use residual_training_policy, not freeze_decoder"
            )
        selected_anchor = kwargs.get("encoder_warmstart_path")
        if not selected_anchor:
            raise ValueError(
                "T4 key residual requires a selected full T4 "
                "warm-start checkpoint"
            )

        kwargs["decoder_mode"] = "coupled"
        kwargs["freeze_decoder"] = False
        super().__init__(**kwargs)
        self._residual_mode = residual_mode
        self._residual_rank = int(residual_rank)
        self._residual_permutation_seed = (
            residual_permutation_seed
        )
        self._residual_training_policy = (
            residual_training_policy
        )
        self._pending_key_residual_checkpoint_receipt: (
            dict[str, object] | None
        ) = None
        self.save_hyperparameters({
            "residual_mode": residual_mode,
            "residual_rank": self._residual_rank,
            "residual_permutation_seed": (
                residual_permutation_seed
            ),
            "residual_training_policy": (
                residual_training_policy
            ),
            "residual_initialization": (
                "zero_output_projection"
            ),
        })

    def setup(self, stage: str) -> None:
        if isinstance(
            self.student,
            CoupledT4KeyResidualStreamingSpint,
        ):
            return
        trainer = getattr(self, "_trainer", None)
        if int(getattr(trainer, "world_size", 1)) != 1:
            raise RuntimeError(
                "T4 key residual setup is world_size=1 until "
                "factor broadcast and post-broadcast hashing exist"
            )
        super().setup(stage)
        if self.student is None:
            raise RuntimeError(
                "parent setup did not construct a student"
            )
        substrate = self.student
        adapter = CoupledT4KeyResidualStreamingSpint(
            decoder=substrate.decoder,
            id_encoder=substrate.id_encoder,
            residual_mode=self._residual_mode,
            residual_rank=self._residual_rank,
            residual_permutation_seed=(
                self._residual_permutation_seed
            ),
        )
        adapter.freeze_backbone_for_residual_pilot()
        if (
            self._residual_training_policy
            == "residual_plus_attention_out"
        ):
            for parameter in (
                adapter.decoder.transformer.layers[0]
                .cross_attn.out_proj.parameters()
            ):
                parameter.requires_grad = True
        self.student = adapter

    def active_key_residual_checkpoint_receipt(
        self,
    ) -> dict[str, object]:
        self.setup("fit")
        assert isinstance(
            self.student,
            CoupledT4KeyResidualStreamingSpint,
        )
        initialization = self.student.key_residual_receipt
        teacher_path = self._teacher_ckpt_path
        teacher_sha = (
            self.teacher_sha256(teacher_path)
            if teacher_path
            and Path(teacher_path).is_file()
            else None
        )
        anchor_path = Path(
            self._encoder_warmstart_path
        ).expanduser()
        anchor_sha = (
            self.teacher_sha256(str(anchor_path))
            if anchor_path.is_file()
            else None
        )
        return {
            "schema_version": 1,
            "module": "T4KeyResidualLitModule",
            "residual_mode": self._residual_mode,
            "residual_rank": self._residual_rank,
            "residual_permutation_seed": (
                self._residual_permutation_seed
            ),
            "residual_training_policy": (
                self._residual_training_policy
            ),
            "teacher_checkpoint_sha256": teacher_sha,
            "selected_t4_anchor_path": str(
                anchor_path.resolve()
            ),
            "selected_t4_anchor_sha256": anchor_sha,
            "initial_factor_sha256": initialization[
                "initial_factor_sha256"
            ],
            "active_factor_sha256": initialization[
                "active_factor_sha256"
            ],
            "zero_initialized": initialization[
                "output_projection_zero_initialized"
            ],
            "backbone_frozen": initialization[
                "backbone_frozen_for_residual_pilot"
            ],
        }

    def on_save_checkpoint(
        self, checkpoint: dict[str, Any]
    ) -> None:
        checkpoint["t4_key_residual_receipt"] = (
            self.active_key_residual_checkpoint_receipt()
        )

    def on_load_checkpoint(
        self, checkpoint: dict[str, Any]
    ) -> None:
        receipt = checkpoint.get(
            "t4_key_residual_receipt"
        )
        if not isinstance(receipt, dict):
            raise ValueError(
                "checkpoint is missing the T4 key-residual receipt"
            )
        expected = {
            "residual_mode": self._residual_mode,
            "residual_rank": self._residual_rank,
            "residual_permutation_seed": (
                self._residual_permutation_seed
            ),
            "residual_training_policy": (
                self._residual_training_policy
            ),
        }
        for name, value in expected.items():
            if receipt.get(name) != value:
                raise ValueError(
                    f"checkpoint {name}="
                    f"{receipt.get(name)!r} does not match "
                    f"constructor {value!r}"
                )
        self._pending_key_residual_checkpoint_receipt = (
            dict(receipt)
        )

    def validate_loaded_key_residual_checkpoint_receipt(
        self,
    ) -> None:
        pending = (
            self._pending_key_residual_checkpoint_receipt
        )
        if pending is None:
            return
        active = self.active_key_residual_checkpoint_receipt()
        if active["active_factor_sha256"] != pending.get(
            "active_factor_sha256"
        ):
            raise ValueError(
                "restored active T4 key-residual hash does not "
                "match checkpoint receipt"
            )
        if active["selected_t4_anchor_sha256"] != pending.get(
            "selected_t4_anchor_sha256"
        ):
            raise ValueError(
                "selected T4 anchor hash does not match "
                "checkpoint receipt"
            )
        self._pending_key_residual_checkpoint_receipt = None

    def on_fit_start(self) -> None:
        self.validate_loaded_key_residual_checkpoint_receipt()

    def decoder_key_features(
        self, side_features: torch.Tensor | None
    ) -> None:
        if (
            side_features is None
            or side_features.ndim != 3
            or side_features.shape[-1] != 4
        ):
            raise ValueError(
                "T4 key residual requires aligned T4 with "
                "shape [B,N,4]"
            )
        # The adapter consumes aligned T4 directly and applies any
        # residual-only control internally. Never mutate encoder T4.
        return None

    def decoupled_cost_receipt(
        self,
        *,
        batch_size: int,
        num_neurons: int,
    ) -> dict[str, object]:
        self.setup("fit")
        assert isinstance(
            self.student,
            CoupledT4KeyResidualStreamingSpint,
        )
        return self.student.residual_cost_receipt(
            batch_size=batch_size,
            num_units=num_neurons,
        )

    @property
    def key_residual_initialization_receipt(
        self,
    ) -> dict[str, object]:
        self.setup("fit")
        assert isinstance(
            self.student,
            CoupledT4KeyResidualStreamingSpint,
        )
        return self.student.key_residual_receipt
