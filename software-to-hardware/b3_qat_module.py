"""Lightning module for B3 W8A8 QAT with bit-exact integer forward and 4-path validation."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import MeanMetric
from torchmetrics.regression import R2Score

LossMode = Literal["task_only", "anchor", "task_plus_y", "task_plus_y_plus_E"]


class B3QATLitModule(pl.LightningModule):
    def __init__(
        self,
        *,
        task: str,
        teacher_ckpt_path: str,
        init_student_ckpt_path: str,
        exp_root: str,
        qat_scales: Dict[str, float],
        calib_sessions: Optional[Sequence[np.ndarray]] = None,
        apply_equalization: bool = True,
        learnable_scales: bool = False,
        num_calib_trials: int = 33,
        loss_mode: LossMode = "anchor",
        lambda_y: float = 0.75,
        lambda_E: float = 0.075,
        lambda_weight: float = 1e-4,
        behavior_scaling_factor: float = 5.0,
        lr: float = 1e-5,
        lr_scale: float = 1e-5,
        weight_decay: float = 0.0,
        warmup_epochs: int = 2,
        lsq_scale_grad_mult: float = 1.0 / 127.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["qat_scales", "calib_sessions"])

        exp = Path(exp_root).resolve()
        if str(exp) not in sys.path:
            sys.path.insert(0, str(exp))

        sw_path = Path(__file__).resolve().parent
        if str(sw_path) not in sys.path:
            sys.path.insert(0, str(sw_path))

        from src.models.components.spint import SpintModel
        from src.models.components.streaming_encoders import EarlyPoolEncoder, build_encoder, copy_teacher_id_weights
        from src.models.components.streaming_spint import StreamingSpintModel
        from src.models.falcon_module import DATASET_NAMES, FalconLitModule

        from b3_fake_quant import B3QATScales
        from b3_ptq import apply_equalization_to_early_pool
        from b3_qat_encoder import clone_anchor_encoder, wrap_early_pool_with_qat

        self._behavior_scaling_factor = behavior_scaling_factor
        self._loss_mode = loss_mode
        self._lambda_y = lambda_y
        self._lambda_E = lambda_E
        self._lambda_weight = lambda_weight
        self.mse_loss = nn.MSELoss()

        teacher_module = FalconLitModule.load_from_checkpoint(teacher_ckpt_path, weights_only=False)
        teacher_module.eval()
        for p in teacher_module.parameters():
            p.requires_grad = False
        self.teacher = teacher_module.net

        decoder = SpintModel(
            model_dim=self.teacher.model_dim,
            num_covariates=self.teacher.num_covariates,
            window_size=self.teacher.window_size,
            num_heads=self.teacher.num_heads,
            num_layers=self.teacher.num_layers,
            num_id_layers=self.teacher.num_id_layers,
            use_learnable_id=True,
            learnable_id_type="mlp",
            learnable_rep=True,
            dropout_rate=self.teacher.dropout_rate,
            dynamic_dropout=self.teacher.dynamic_dropout,
            dynamic_dropout_low=self.teacher.dynamic_dropout_low,
            dynamic_dropout_high=self.teacher.dynamic_dropout_high,
            tf_drop_rate=self.teacher.tf_drop_rate,
            readin_layer_type=self.teacher.readin_layer_type,
        )
        decoder.load_state_dict(self.teacher.state_dict(), strict=True)

        id_encoder = build_encoder(
            "B3",
            window_size=self.teacher.window_size,
            trial_length=100,
            hidden_dim=64,
            pad_value=-1.0,
        )
        copy_teacher_id_weights(id_encoder, self.teacher)

        self.student = StreamingSpintModel(decoder=decoder, id_encoder=id_encoder)
        self.student.freeze_decoder()

        ckpt = torch.load(init_student_ckpt_path, map_location="cpu", weights_only=False)
        student_sd = {
            k.replace("student.", "", 1): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("student.")
        }
        missing, unexpected = self.student.load_state_dict(student_sd, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys loading student checkpoint: {unexpected[:8]}")

        base_enc = self.student.id_encoder
        if not isinstance(base_enc, EarlyPoolEncoder):
            raise TypeError(f"Expected EarlyPoolEncoder, got {type(base_enc)}")

        self.anchor_encoder = clone_anchor_encoder(base_enc)

        if apply_equalization and calib_sessions:
            apply_equalization_to_early_pool(base_enc, calib_sessions)

        scales = B3QATScales(
            input=float(qat_scales["input"]),
            pre_out=float(qat_scales["pre_out"]),
            mean=float(qat_scales["mean"]),
            post0_out=float(qat_scales["post0_out"]),
            post1_out=float(qat_scales["post1_out"]),
            E=float(qat_scales["E"]),
        )
        self.qat_encoder = wrap_early_pool_with_qat(
            base_enc,
            scales,
            num_trials=num_calib_trials,
            learnable_scales=learnable_scales,
        )
        self.student.id_encoder = self.qat_encoder

        # L_weight reference: equalized shadow at init (preserves cross-layer scaling).
        self._shadow_weight_reference = {
            name: param.detach().clone()
            for name, param in self._encoder_named_linears(self.qat_encoder)
        }

        heldin_names = DATASET_NAMES[task]["heldin"]
        self.val_r2_anchor = nn.ModuleDict({k: R2Score(multioutput="variance_weighted") for k in heldin_names})
        self.val_r2_shadow = nn.ModuleDict({k: R2Score(multioutput="variance_weighted") for k in heldin_names})
        self.val_r2_fake = nn.ModuleDict({k: R2Score(multioutput="variance_weighted") for k in heldin_names})
        self.val_loss = MeanMetric()
        self.train_loss = MeanMetric()
        self._heldout_calib: Optional[torch.Tensor] = None
        self._best_train_loss = float("inf")

    @staticmethod
    def _encoder_named_linears(encoder: nn.Module) -> List[Tuple[str, torch.Tensor]]:
        from b3_qat_encoder import QATEarlyPoolEncoder

        if isinstance(encoder, QATEarlyPoolEncoder):
            names: List[Tuple[str, torch.Tensor]] = [
                ("pre_linear.weight", encoder.pre_linear.weight),
                ("pre_linear.bias", encoder.pre_linear.bias),
            ]
            for i, layer in enumerate(encoder.post_linears):
                names.append((f"post_linears.{i}.weight", layer.weight))
                names.append((f"post_linears.{i}.bias", layer.bias))
            return names

        from src.models.components.streaming_encoders import EarlyPoolEncoder

        if not isinstance(encoder, EarlyPoolEncoder):
            raise TypeError(encoder)
        names = [("pre_linear.weight", encoder.pre_pool[0].weight), ("pre_linear.bias", encoder.pre_pool[0].bias)]
        idx = 0
        for layer in encoder.post_pool:
            if isinstance(layer, nn.Linear):
                names.append((f"post_linears.{idx}.weight", layer.weight))
                names.append((f"post_linears.{idx}.bias", layer.bias))
                idx += 1
        return names

    def _weight_anchor_penalty(self) -> torch.Tensor:
        """Normalized relative MSE across all encoder tensors (scale-invariant per layer)."""
        rel_terms: List[torch.Tensor] = []
        for name, ref_w in self._shadow_weight_reference.items():
            cur = dict(self._encoder_named_linears(self.qat_encoder))[name]
            ref = ref_w.to(cur.device)
            rel_terms.append(((cur - ref) ** 2 / ref.pow(2).clamp_min(1e-8)).mean())
        return torch.stack(rel_terms).mean()

    def _slice_last(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = pred[:, -1:, :]
        target = target[:, -1:, :]
        return pred / self._behavior_scaling_factor, target

    @torch.no_grad()
    def _anchor_targets(self, calib: torch.Tensor) -> torch.Tensor:
        return self.anchor_encoder.forward_batch(calib)

    def _decode(self, neural: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.student.decode_with_identity(neural, e)

    def _ensure_devices(self, device: torch.device) -> None:
        if next(self.qat_encoder.parameters()).device != device:
            self.qat_encoder.to(device)
        if next(self.anchor_encoder.parameters()).device != device:
            self.anchor_encoder.to(device)

    def _forward_paths(
        self,
        neural: torch.Tensor,
        calib: torch.Tensor,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        self._ensure_devices(calib.device)
        e_shadow = self.qat_encoder.forward_fp32(calib)
        e_fake = self.qat_encoder.forward_batch(calib)
        y_shadow = self._decode(neural, e_shadow)
        y_fake = self._decode(neural, e_fake)
        with torch.no_grad():
            e_anchor = self._anchor_targets(calib)
            y_anchor = self._decode(neural, e_anchor)
        return {
            "anchor": (y_anchor, e_anchor),
            "shadow": (y_shadow, e_shadow),
            "fake": (y_fake, e_fake),
        }

    def _integer_engine_E(self, calib: torch.Tensor) -> torch.Tensor:
        from b3_hw_golden import B3Shapes, B3Weights
        from b3_quant_engine import ABLATION_PRESETS, build_quant_engine_bundle, forward_quant_engine

        calib_np = calib.detach().cpu().numpy()
        if calib_np.ndim == 4:
            calib_np = calib_np[0]
        qe = self.qat_encoder

        def _np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy().astype(np.float32)

        weights = B3Weights(
            pre_w=_np(qe.pre_linear.weight),
            pre_b=_np(qe.pre_linear.bias),
            post0_w=_np(qe.post_linears[0].weight),
            post0_b=_np(qe.post_linears[0].bias),
            post1_w=_np(qe.post_linears[1].weight),
            post1_b=_np(qe.post_linears[1].bias),
            post2_w=_np(qe.post_linears[2].weight),
            post2_b=_np(qe.post_linears[2].bias),
        )
        shapes = B3Shapes(T=calib_np.shape[1], D=64, W=50, N=calib_np.shape[2], M=calib_np.shape[0])
        bundle = build_quant_engine_bundle(
            weights,
            shapes,
            qe.to_frozen_scales([]),
            ABLATION_PRESETS["w8_a8_e8"],
        )
        e = forward_quant_engine(calib_np, bundle)["E_dequant"]
        return torch.from_numpy(e).to(device=calib.device, dtype=calib.dtype)

    def _training_loss(self, neural: torch.Tensor, behavior: torch.Tensor, calib: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._ensure_devices(calib.device)
        e_fake = self.qat_encoder.forward_batch(calib)
        y = self._decode(neural, e_fake)
        y, behavior = self._slice_last(y, behavior)
        loss_task = self.mse_loss(y, behavior)
        components: Dict[str, torch.Tensor] = {"task": loss_task.detach()}

        loss = loss_task
        if self._loss_mode == "anchor":
            with torch.no_grad():
                e_anchor = self._anchor_targets(calib)
                y_anchor = self._decode(neural, e_anchor)
                y_anchor, _ = self._slice_last(y_anchor, behavior)
            loss_y = self.mse_loss(y, y_anchor)
            denom = (e_anchor**2).mean().clamp_min(1e-8)
            loss_e = ((e_fake - e_anchor) ** 2).mean() / denom
            loss_w = self._weight_anchor_penalty()
            loss = loss + self._lambda_y * loss_y + self._lambda_E * loss_e + self._lambda_weight * loss_w
            components.update({
                "y_anchor": loss_y.detach(),
                "E_anchor": loss_e.detach(),
                "weight": loss_w.detach(),
            })
        return loss, components

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        neural, behavior, calib, _session = batch
        loss, components = self._training_loss(neural, behavior, calib)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_epoch=True, prog_bar=True)
        if batch_idx % 10 == 0:
            for key, val in components.items():
                self.log(f"train/loss_{key}", val, prog_bar=(key == "task"))
        if loss.item() < self._best_train_loss:
            self._best_train_loss = loss.item()
        elif loss.item() > self._best_train_loss * 5.0:
            self.log("train/loss_exploded", 1.0, prog_bar=True)
        return loss

    def on_fit_start(self) -> None:
        if getattr(self.trainer.datamodule, "val_heldin_dataset", None) is None:
            return
        from b3_eval_protocol import get_full_calib_pool, load_split_manifest

        manifest = load_split_manifest(Path(self.hparams.init_student_ckpt_path).parent.parent / "split_manifest.json")
        pool = get_full_calib_pool(self.trainer.datamodule.val_heldin_dataset, manifest.heldout_session)
        self._heldout_calib = torch.from_numpy(pool[: self.hparams.num_calib_trials]).unsqueeze(0).to(self.device)

    def validation_step(self, batch, batch_idx: int) -> None:
        neural, behavior, calib, session_name = batch
        sess = session_name[0]
        paths = self._forward_paths(neural, calib)
        for key, (y_pred, _) in paths.items():
            y_pred, target = self._slice_last(y_pred, behavior)
            metric_map = {
                "anchor": self.val_r2_anchor,
                "shadow": self.val_r2_shadow,
                "fake": self.val_r2_fake,
            }
            if sess in metric_map[key]:
                metric_map[key][sess].update(y_pred.flatten(0, 1), target.flatten(0, 1))

    def on_validation_epoch_end(self) -> None:
        path_keys = ("anchor", "shadow", "fake")
        means: Dict[str, List[torch.Tensor]] = {k: [] for k in path_keys}
        for key in path_keys:
            metric_dict = getattr(self, f"val_r2_{key}")
            for name, metric in metric_dict.items():
                if metric.total > 2:
                    r2 = metric.compute()
                    means[key].append(r2)
                    self.log(f"val_{key}/{name}/r2", r2, prog_bar=(key == "fake"))
                metric.reset()
            if means[key]:
                mean_r2 = torch.stack(means[key]).mean()
                self.log(f"val_{key}/r2_mean", mean_r2, prog_bar=(key in {"anchor", "shadow", "fake"}))

        if means["shadow"] and means["anchor"]:
            delta_shadow = torch.stack(means["shadow"]).mean() - torch.stack(means["anchor"]).mean()
            self.log("val_shadow/delta_r2", delta_shadow, prog_bar=True)

        if self._heldout_calib is not None:
            with torch.no_grad():
                e_fake = self.qat_encoder.forward_batch(self._heldout_calib).reshape(-1)
                e_int = self._integer_engine_E(self._heldout_calib).reshape(-1)
                max_abs = (e_fake - e_int).abs().max().item()
                exact = max_abs == 0.0
            self.log("val_align/E_exact", float(exact), prog_bar=True)
            self.log("val_align/E_max_abs", max_abs, prog_bar=True)

            from b3_eval_protocol import collect_session_windows, load_split_manifest, session_r2_with_E

            manifest = load_split_manifest(Path(self.hparams.init_student_ckpt_path).parent.parent / "split_manifest.json")
            val_ds = self.trainer.datamodule.val_heldin_dataset
            batch_size = int(getattr(self.trainer.datamodule.hparams, "batch_size", 32) or 32)
            neural, behavior = collect_session_windows(val_ds, manifest.heldout_session, batch_size=batch_size)
            e_int_np = self._integer_engine_E(self._heldout_calib).squeeze(0).cpu().numpy()
            r2_int = session_r2_with_E(
                self.student,
                neural,
                behavior,
                e_int_np,
                behavior_scale=self._behavior_scaling_factor,
            )
            self.log("val_integer_engine/r2_mean", torch.tensor(r2_int), prog_bar=True)
            if means.get("fake"):
                delta_gap = torch.stack(means["fake"]).mean() - torch.tensor(r2_int)
                self.log("val_fake_integer/r2_gap", delta_gap, prog_bar=True)

            if self.hparams.learnable_scales:
                diag = self.qat_encoder.compute_quant_diagnostics(self._heldout_calib)
                for edge, ratio in diag["scales_relative"].items():
                    self.log(f"scale/{edge}/rel_init", ratio, prog_bar=(edge == "input"))
                for edge, stats in diag.items():
                    if edge in {"scales_relative", "scales_current"}:
                        continue
                    self.log(f"sat/{edge}/rate", stats["saturation_rate"])
                    self.log(f"sat/{edge}/pre_clip_max", stats["pre_clip_max"], prog_bar=(edge == "post1_out"))

    def _split_qat_params(self) -> Tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        scale_params = self.qat_encoder.shared_scales.scale_parameters()
        scale_ids = {id(p) for p in scale_params}
        weight_params = [p for p in self.qat_encoder.parameters() if p.requires_grad and id(p) not in scale_ids]
        return weight_params, scale_params

    def on_before_optimizer_step(self, optimizer) -> None:
        if not self.hparams.learnable_scales:
            total_norm = 0.0
            for p in self.qat_encoder.parameters():
                if p.grad is not None:
                    total_norm += float(p.grad.data.norm(2).item() ** 2)
            self.log("train/grad_norm", total_norm ** 0.5, prog_bar=False)
            return
        _, scale_params = self._split_qat_params()
        for p in scale_params:
            if p.grad is not None:
                p.grad.mul_(self.hparams.lsq_scale_grad_mult)
        w_norm, s_norm = 0.0, 0.0
        weight_params, _ = self._split_qat_params()
        for p in weight_params:
            if p.grad is not None:
                w_norm += float(p.grad.data.norm(2).item() ** 2)
        for p in scale_params:
            if p.grad is not None:
                s_norm += float(p.grad.data.norm(2).item() ** 2)
        self.log("train/grad_norm_weight", w_norm ** 0.5, prog_bar=False)
        self.log("train/grad_norm_scale", s_norm ** 0.5, prog_bar=False)

    def configure_optimizers(self):
        warmup = max(int(self.hparams.warmup_epochs), 0)
        if self.hparams.learnable_scales:
            weight_params, scale_params = self._split_qat_params()
            opt = torch.optim.Adam(
                [
                    {"params": weight_params, "lr": self.hparams.lr, "weight_decay": self.hparams.weight_decay},
                    {"params": scale_params, "lr": self.hparams.lr_scale, "weight_decay": 0.0},
                ],
            )
        else:
            params = [p for p in self.qat_encoder.parameters() if p.requires_grad]
            opt = torch.optim.Adam(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        if warmup <= 0:
            return opt

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return float(epoch + 1) / float(warmup)
            return 1.0

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
