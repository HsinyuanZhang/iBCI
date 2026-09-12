"""Falcon runtime for M1 RIFT R100 proj_add P16 + muscle flat (zero-slope).

Identity is proj_add P16. Recency is per-layer fixed zero slopes (flat
control). The seven sealed banks keep legacy fullsession E0 and replace only
the 64x4 T carrier with official muscle_response16_svd4/global_rms.
Backend is cached PyTorch FP32 via CpuRiftTemporalRuntime.
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
from math import log
from pathlib import Path
from typing import Hashable, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from falcon_challenge.interface import BCIDecoder

from btransform_unified_v1.bank import TaskBank
from btransform_unified_v2.cpu_temporal import CpuRiftTemporalRuntime
from btransform_unified_v2.model import RiftDecoder
from btransform_unified_v2.temporal import RiftTemporal
from learnable_recency_v1.config import LearnableRecencyConfig, config_from_run_meta

PAYLOAD_SCHEMA = "m1_rift_r100_projadd_muscle_flat_e2_cached_v1"
CARRIER_VARIANT = "muscle_response16_svd4/global_rms"
CHANNELS = 64
BEHAVIOR_SCALE = 1.0
EXPECTED_SESSION_COUNT = 7
OFFICIAL_BATCH = 4
CONTEXT_BINS = 100
PROJ_DIM = 16


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda value: torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_payload(path: str | Path) -> dict:
    with open(path, "rb") as handle:
        return CPUUnpickler(handle).load()


def recency_config_from_payload(payload: dict) -> LearnableRecencyConfig:
    raw = payload.get("learnable_config")
    if not isinstance(raw, dict):
        raise ValueError("payload missing learnable_config")
    return config_from_run_meta({"learnable_config": raw}, "m1")


def _architecture_key(config) -> tuple:
    return (config.layers, config.windows, config.width, config.heads, config.ffn_width)


def _slopes_from_config(config, device: torch.device) -> Tensor:
    slopes = [
        0.0 if half is None else log(2.0) * config.bin_seconds / half
        for half in config.half_life_seconds
    ]
    return torch.tensor(slopes, dtype=torch.float32, device=device)


def _adopt_temporal(existing: RiftTemporal, config) -> RiftTemporal:
    if _architecture_key(existing.config) != _architecture_key(config):
        raise ValueError("RIFT temporal architecture drifted versus learnable_config")
    if existing.config.half_life_seconds == config.half_life_seconds and existing.config.bin_seconds == config.bin_seconds:
        return existing
    temporal = RiftTemporal.__new__(RiftTemporal)
    nn.Module.__init__(temporal)
    temporal.config = config
    temporal.add_module("blocks", existing.blocks)
    temporal.register_buffer(
        "recency_slopes",
        _slopes_from_config(config, existing.recency_slopes.device),
        persistent=True,
    )
    temporal._attention_backend = existing._attention_backend
    return temporal


def build_decoder(recency_cfg: LearnableRecencyConfig, seed: int) -> RiftDecoder:
    """Rebuild proj_add + per-layer flat (fixed zero-slope) without wrap.py."""
    if recency_cfg.tier != "fixed" or recency_cfg.ladder != "default" or not recency_cfg.per_layer:
        raise ValueError("this pack is only for per-layer fixed/flat --ladder default")
    if any(half is not None for half in recency_cfg.half_life_seconds):
        raise ValueError("flat pack requires all half-lives None")
    model = RiftDecoder("m1", context_bins=CONTEXT_BINS, bias_mode="recency", seed=int(seed), proj_dim=PROJ_DIM)
    model.temporal = _adopt_temporal(model.temporal, recency_cfg.temporal_config)
    model.temporal_config = recency_cfg.temporal_config
    model.learnable_cfg = recency_cfg
    model.bias_mode = "recency"
    return model


class CpuLearnableRiftRuntime:
    """CpuRiftRuntime frontend with CpuRiftTemporalRuntime (plain zero slopes)."""

    def __init__(
        self,
        decoder: RiftDecoder,
        banks: Sequence[TaskBank],
        stream_ids: Sequence[Hashable],
    ) -> None:
        if decoder.training:
            raise ValueError("CpuLearnableRiftRuntime requires decoder.eval()")
        if not isinstance(decoder.temporal, RiftTemporal):
            raise TypeError("flat decoder.temporal must be a plain RiftTemporal")
        if len(banks) < 1 or len(banks) != len(stream_ids) or len(set(stream_ids)) != len(stream_ids):
            raise ValueError("one unique stream_id and TaskBank per fixed row required")
        self.decoder, self.ids, self.banks = decoder, list(stream_ids), list(banks)
        self.row = {key: index for index, key in enumerate(self.ids)}
        batch = len(banks)
        device = next(decoder.parameters()).device
        decoder._normalise_banks(self.banks, batch)
        self.e0 = torch.stack([torch.from_numpy(bank.E0) for bank in banks]).to(device, torch.float32)
        self.carrier = torch.stack([torch.from_numpy(bank.carrier) for bank in banks]).to(device, torch.float32)
        self.keep = torch.stack([torch.from_numpy(bank.unit_mask) for bank in banks]).to(device, torch.bool)
        self.weight_versions = tuple(parameter._version for parameter in decoder.parameters())
        if not bool(self.keep.any(dim=1).all()):
            raise ValueError("unit mask became empty for some batch row")
        self.raw4 = torch.zeros(batch, 4, decoder.units, device=device)
        self.cached_temporal = CpuRiftTemporalRuntime(decoder.temporal, batch, device)
        self.last: Tensor | None = None

    @torch.inference_mode()
    def advance(
        self,
        observed: Tensor,
        stream_ids: Sequence[Hashable] | None = None,
        *,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        if self.decoder.training:
            raise RuntimeError("decoder switched to train mode")
        if tuple(parameter._version for parameter in self.decoder.parameters()) != self.weight_versions:
            raise RuntimeError("decoder parameters changed after runtime registration")
        if observed.shape != (len(self.ids), self.decoder.units):
            raise ValueError("observed must match fixed [B,N]")
        if valid_mask is None:
            valid_mask = torch.ones(len(self.ids), device=observed.device, dtype=torch.bool)
        if valid_mask.dtype != torch.bool or valid_mask.shape != (len(self.ids),):
            raise ValueError("valid_mask must be bool [B]")
        if stream_ids is not None:
            if set(stream_ids) != set(self.ids) or len(stream_ids) != len(self.ids):
                raise ValueError("stream_ids must be a permutation of registered ids")
            take = torch.tensor([self.row[key] for key in stream_ids], device=observed.device)
            self.raw4 = self.raw4.index_select(0, take)
            self.e0 = self.e0.index_select(0, take)
            self.carrier = self.carrier.index_select(0, take)
            self.keep = self.keep.index_select(0, take)
            self.cached_temporal.reorder(take)
            if self.last is not None:
                self.last = self.last.index_select(0, take)
            self.ids = list(stream_ids)
            self.row = {key: index for index, key in enumerate(self.ids)}
        x = observed.to(self.raw4.device, torch.float32)
        valid = valid_mask.to(self.raw4.device)
        raw5 = torch.cat((self.raw4, x.unsqueeze(1)), dim=1)
        conv = self.decoder.frontend.local_conv
        flat = raw5.permute(0, 2, 1).reshape(x.shape[0] * self.decoder.units, 1, 5)
        local = conv.act(conv.conv(flat)).reshape(x.shape[0], self.decoder.units, 16, 1).permute(0, 3, 1, 2)
        z = self.decoder._fuse_batched_local(local, self.e0, self.carrier, self.keep)[:, 0]
        hidden = self.cached_temporal.step(z, valid)
        self.raw4.copy_(torch.where(valid[:, None, None], raw5[:, 1:], self.raw4))
        score = self.decoder.readout(self.decoder.final_norm(hidden))
        self.last = score if self.last is None else torch.where(valid[:, None], score, self.last)
        return score


def _task_bank(tag: str, row: dict) -> TaskBank:
    e0 = np.ascontiguousarray(row["E0"], dtype=np.float32)
    carrier = np.ascontiguousarray(row["T"], dtype=np.float32)
    mask = np.ascontiguousarray(row["unit_mask"], dtype=np.bool_)
    return TaskBank(
        session_id=str(row.get("session") or tag),
        E0=e0,
        carrier=carrier,
        unit_mask=mask,
        X_store=np.zeros((0, CONTEXT_BINS, e0.shape[0]), dtype=np.float32),
        target_store=np.zeros((0, 16), dtype=np.float32),
        window_ids=np.zeros(0, dtype=np.int64),
        calibration_meta={
            "shape": tuple(e0.shape),
            "trial_count": 10,
            "estimator": "legacy fullsession E0 + sealed muscle_response16_svd4/global_rms T",
            "array_sha256": str(row.get("e0_sha256") or "payload"),
            "budget": 10,
        },
    )


class M1RiftCachedFalconDecoder(BCIDecoder):
    def __init__(self, task_config, model_path: str, batch_size: int = 1):
        super().__init__(task_config=task_config, batch_size=batch_size)
        self.task_config = task_config
        self.batch_size = int(batch_size)
        if self.batch_size > OFFICIAL_BATCH:
            raise ValueError(f"batch_size {self.batch_size} exceeds official M1 max {OFFICIAL_BATCH}")
        payload = load_payload(model_path)
        if payload.get("schema") != PAYLOAD_SCHEMA:
            raise ValueError(f"unsupported payload schema {payload.get('schema')}")
        if int(payload.get("context_bins", -1)) != CONTEXT_BINS:
            raise ValueError("payload context_bins must be 100")
        if payload.get("identity_interface") != "proj_add" or int(payload.get("proj_dim", -1)) != PROJ_DIM:
            raise ValueError("payload must be proj_add P16")
        if payload.get("carrier_variant") != CARRIER_VARIANT:
            raise ValueError("payload carrier_variant must be muscle_response16_svd4/global_rms")
        if payload.get("ladder") != "default" or payload.get("tier") != "fixed":
            raise ValueError("payload must be fixed/flat default-ladder")
        self.bank_by_dataset_tag = payload["bank_by_dataset_tag"]
        if len(self.bank_by_dataset_tag) != EXPECTED_SESSION_COUNT:
            raise ValueError("M1 payload must cover exactly 7 official dataset tags")
        self.behavior_scaling_factor = float(payload["behavior_scaling_factor"])
        if abs(self.behavior_scaling_factor - BEHAVIOR_SCALE) > 1.0e-12:
            raise ValueError("behavior scale must be 1")
        recency_cfg = recency_config_from_payload(payload)
        seed = int(payload.get("seed", 42))
        self.decoder = build_decoder(recency_cfg, seed)
        state = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in payload["ema_state_dict"].items()}
        missing, unexpected = self.decoder.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise ValueError(f"state_dict mismatch missing={missing} unexpected={unexpected}")
        self.decoder.eval()
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        self.device = torch.device("cpu")
        self.decoder = self.decoder.to(self.device)
        self._runtime: CpuLearnableRiftRuntime | None = None
        self._n_predicts = 0
        self.kind = "rift_cached_flat_projadd_muscle"
        self.window_size = CONTEXT_BINS

    @staticmethod
    def _stem(value) -> str:
        return Path(value).stem

    def reset(self, dataset_tags: Iterable[Path] = (Path(""),)):
        hashed_tags = [self.task_config.hash_dataset(self._stem(value)) for value in dataset_tags]
        missing = [tag for tag in hashed_tags if tag not in self.bank_by_dataset_tag]
        if missing:
            raise ValueError(f"Dataset tags {missing} are absent from the M1 RIFT bank payload")
        banks = [_task_bank(tag, self.bank_by_dataset_tag[tag]) for tag in hashed_tags]
        ids = [str(tag) for tag in hashed_tags]
        self._runtime = CpuLearnableRiftRuntime(self.decoder, banks, ids)
        self._n_predicts = 0

    def on_done(self, dones: np.ndarray):
        return None

    def set_batch_size(self, batch_size: int):
        batch_size = int(batch_size)
        if batch_size > OFFICIAL_BATCH:
            raise ValueError(f"batch_size {batch_size} exceeds official M1 max {OFFICIAL_BATCH}")
        self.batch_size = batch_size
        self._runtime = None

    def observe(self, neural_observations: np.ndarray):
        return None

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        if self._runtime is None:
            raise RuntimeError("reset(dataset_tags) must be called before predict")
        observations = np.asarray(neural_observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != self.task_config.n_channels:
            raise ValueError(
                f"Expected neural observations [B,{self.task_config.n_channels}], got {observations.shape}"
            )
        active = observations.shape[0]
        n_reg = len(self._runtime.ids)
        if active > n_reg:
            raise ValueError(f"evaluator batch {active} exceeds reset roster {n_reg}")
        if active < n_reg:
            observations = np.pad(observations, ((0, n_reg - active), (0, 0)))
        valid = torch.zeros(n_reg, dtype=torch.bool)
        valid[:active] = True
        with torch.inference_mode():
            pred = self._runtime.advance(torch.from_numpy(np.ascontiguousarray(observations)), valid_mask=valid)
        native = pred[:active].detach().cpu().numpy() / self.behavior_scaling_factor
        if not np.isfinite(native).all():
            raise RuntimeError("non-finite M1 RIFT prediction")
        self._n_predicts += 1
        return native.astype(np.float32, copy=False)


def smoke(payload_path: str, window_path: str) -> None:
    from falcon_challenge.config import FalconConfig, FalconTask

    bundle = np.load(window_path)
    config = FalconConfig(task=FalconTask.m1)
    decoder = M1RiftCachedFalconDecoder(task_config=config, model_path=payload_path, batch_size=1)
    decoder.reset(dataset_tags=[str(bundle["tag_stem"])])
    window = np.asarray(bundle["window"], dtype=np.float32)
    pred = None
    for row in window:
        pred = decoder.predict(row.reshape(1, -1))
        decoder.on_done(np.ones((1,), dtype=bool))
    expected = np.asarray(bundle["expected"], dtype=np.float32)
    if pred is None or pred.shape != expected.shape or not np.allclose(pred, expected, atol=1.0e-5, rtol=1.0e-5):
        raise RuntimeError(f"container smoke mismatch {pred} vs {expected}")
    print(json.dumps({"status": "CONTAINER_SMOKE_PASS", "pred": pred.tolist(), "kind": decoder.kind}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-payload", default="")
    parser.add_argument("--smoke-window", default="")
    args = parser.parse_args()
    if args.smoke_payload and args.smoke_window:
        smoke(args.smoke_payload, args.smoke_window)
    else:
        raise SystemExit("use --smoke-payload and --smoke-window")
