"""C1 shared-weight SUA/pseudo-MUA training module.

The C1 hypothesis is deliberately narrow: one ordinary B3S/T4 encoder/decoder
weight set is optimized on matched sorted-SUA and electrode-pooled pseudo-MUA
microbatches.  This module adds no feature fusion path, no view-specific head,
and no consistency loss.  It performs two *sequential* half-weight backwards
passes followed by one optimizer step, rather than retaining both graphs for a
single summed loss.  That is the same gradient for a stateless task loss while
keeping activation residency close to the larger single-view pass.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

from src.models.streaming_calibration_module import StreamingCalibrationLitModule


class PairedViewC1LitModule(StreamingCalibrationLitModule):
    """Shared C1 model with fixed ``0.5 L_SUA + 0.5 L_pseudoMUA`` objective."""

    def __init__(
        self,
        *args: Any,
        lambda_consistency: float = 0.0,
        sua_task_loss_weight: float = 0.5,
        pseudo_mua_task_loss_weight: float = 0.5,
        **kwargs: Any,
    ) -> None:
        if lambda_consistency != 0.0:
            raise ValueError(
                "C1 is lambda_consistency=0 only; consistency is a conditional C2 experiment"
            )
        if (sua_task_loss_weight, pseudo_mua_task_loss_weight) != (0.5, 0.5):
            raise ValueError("C1 fixes exact equal task-loss weights (0.5, 0.5)")
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(
            "lambda_consistency", "sua_task_loss_weight", "pseudo_mua_task_loss_weight"
        )
        self.automatic_optimization = False
        self._c1_lambda_consistency = float(lambda_consistency)
        self._c1_sua_weight = float(sua_task_loss_weight)
        self._c1_pseudo_weight = float(pseudo_mua_task_loss_weight)

    @staticmethod
    def _unpack_pair(batch: Any) -> tuple[Sequence[Any], Sequence[Any]]:
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ValueError("C1 training_step requires (sua_batch, pseudo_mua_batch)")
        sua_batch, pseudo_batch = batch
        if not isinstance(sua_batch, (tuple, list)) or not isinstance(pseudo_batch, (tuple, list)):
            raise ValueError("C1 paired components must be standard batch tuples")
        return sua_batch, pseudo_batch

    @staticmethod
    def _assert_matched_exposure(
        sua_batch: Sequence[Any], pseudo_mua_batch: Sequence[Any]
    ) -> None:
        # Keep this dependency local to avoid importing data code into every
        # legacy streaming model at module import time.
        from mc_maze.paired_view_c1 import validate_pair_batch

        validate_pair_batch(sua_batch, pseudo_mua_batch)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        sua_batch, pseudo_mua_batch = self._unpack_pair(batch)
        self._assert_matched_exposure(sua_batch, pseudo_mua_batch)
        if self._c1_lambda_consistency != 0.0:
            raise RuntimeError("C1 execution attempted a non-zero consistency path")

        optimizer = self.optimizers()
        optimizer.zero_grad()

        # Crucially, backpropagate and free the SUA graph before constructing the
        # pseudo-MUA graph.  There is one shared optimizer and exactly one step.
        sua_out = self.model_step(tuple(sua_batch))
        sua_loss = sua_out["loss"]
        self.manual_backward(self._c1_sua_weight * sua_loss)

        pseudo_out = self.model_step(tuple(pseudo_mua_batch))
        pseudo_loss = pseudo_out["loss"]
        self.manual_backward(self._c1_pseudo_weight * pseudo_loss)
        optimizer.step()

        combined = self._c1_sua_weight * sua_loss.detach() + self._c1_pseudo_weight * pseudo_loss.detach()
        self.train_loss(combined)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/sua_task_loss", sua_loss.detach(),
            on_step=False, on_epoch=True,
        )
        self.log(
            "train/pseudo_mua_task_loss", pseudo_loss.detach(),
            on_step=False, on_epoch=True,
        )
        self.log(
            "train/lambda_consistency", torch.tensor(0.0, device=combined.device),
            on_step=False, on_epoch=True,
        )
        return combined

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """C1 has no in-fit validation selection path.

        Epoch 5--12 fixed external evaluation is the only development scorer;
        this method fails closed if a caller accidentally wires a validation
        loader and might otherwise create a hidden model-selection channel.
        """
        raise RuntimeError(
            "C1 training must not consume a validation loader; use the fixed external "
            "epoch-window evaluator after training"
        )

    def c1_objective_receipt(self) -> dict[str, object]:
        return {
            "formula": "0.5*L_task(SUA)+0.5*L_task(pseudo_MUA)",
            "sua_task_loss_weight": self._c1_sua_weight,
            "pseudo_mua_task_loss_weight": self._c1_pseudo_weight,
            "lambda_consistency": self._c1_lambda_consistency,
            "backward_execution": "sequential_half_weight_backward_then_single_optimizer_step",
            "shared_optimizer": True,
            "view_specific_heads": False,
        }
