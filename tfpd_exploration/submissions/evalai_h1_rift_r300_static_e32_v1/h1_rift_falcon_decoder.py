"""Falcon runtime for H1 RIFT R300 static_identity + learned_slope e32.

No TaskBank, no E0, no carrier. Shared static_identity[176,16] is in the EMA
weights. Cached CPU via CpuLearnableRecencyRuntime. Not 582196 / 582241 / 582278.
"""
from __future__ import annotations

import io
import json
import pickle
from pathlib import Path
from typing import Hashable, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor
from falcon_challenge.interface import BCIDecoder

from learnable_recency_v1.config import LearnableRecencyConfig, config_from_run_meta
from learnable_recency_v1.cpu_temporal import CpuLearnableRecencyRuntime
from learnable_recency_v1.static_model import StaticLearnableRiftDecoder
from learnable_recency_v1.temporal import LearnableRecencyTemporal

PAYLOAD_SCHEMA = "h1_rift_r300_static_e32_cached_v1"
CHANNELS = 176
BEHAVIOR_SCALE = 20.0
EXPECTED_SESSION_COUNT = 27
OFFICIAL_BATCH = 8
CONTEXT_BINS = 300
SEED = 42


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
    return config_from_run_meta({"learnable_config": raw}, "h1")


def build_decoder(recency_cfg: LearnableRecencyConfig) -> StaticLearnableRiftDecoder:
    if recency_cfg.tier != "learned_slope" or recency_cfg.ladder != "default" or not recency_cfg.per_layer:
        raise ValueError("this pack is only for per-layer learned_slope --ladder default")
    return StaticLearnableRiftDecoder("h1", recency_cfg, seed=SEED)


class CpuStaticRiftRuntime:
    def __init__(self, decoder: StaticLearnableRiftDecoder, stream_ids: Sequence[Hashable]) -> None:
        if decoder.training:
            raise ValueError("CpuStaticRiftRuntime requires decoder.eval()")
        if not isinstance(decoder.temporal, LearnableRecencyTemporal):
            raise TypeError("decoder.temporal must be LearnableRecencyTemporal")
        if len(stream_ids) < 1 or len(set(stream_ids)) != len(stream_ids):
            raise ValueError("one unique stream_id per fixed row required")
        self.decoder = decoder
        self.ids = list(stream_ids)
        self.row = {key: index for index, key in enumerate(self.ids)}
        batch = len(self.ids)
        device = next(decoder.parameters()).device
        if tuple(decoder.static_identity.shape) != (CHANNELS, 16) or bool(decoder.static_carrier.any()):
            raise ValueError("static identity/carrier invariant drift")
        self.keep = torch.ones(batch, decoder.units, dtype=torch.bool, device=device)
        self.weight_versions = tuple(parameter._version for parameter in decoder.parameters())
        self.raw4 = torch.zeros(batch, 4, decoder.units, device=device)
        self.cached_temporal = CpuLearnableRecencyRuntime(decoder.temporal, batch, device)
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
            self.keep = self.keep.index_select(0, take)
            self.cached_temporal.reorder(take)
            if self.last is not None:
                self.last = self.last.index_select(0, take)
            self.ids = list(stream_ids)
            self.row = {key: index for index, key in enumerate(self.ids)}
        x = observed.to(self.raw4.device, torch.float32)
        valid = valid_mask.to(self.raw4.device)
        raw5 = torch.cat((self.raw4, x.unsqueeze(1)), dim=1)
        z = self.decoder.frontend_last(raw5, self.keep)
        hidden = self.cached_temporal.step(z, valid)
        self.raw4.copy_(torch.where(valid[:, None, None], raw5[:, 1:], self.raw4))
        score = self.decoder.readout(self.decoder.final_norm(hidden))
        self.last = score if self.last is None else torch.where(valid[:, None], score, self.last)
        return score


class H1RiftLearnableFalconDecoder(BCIDecoder):
    def __init__(self, task_config, model_path: str, batch_size: int = 1):
        super().__init__(task_config=task_config, batch_size=batch_size)
        self.task_config = task_config
        self.batch_size = int(batch_size)
        if self.batch_size > OFFICIAL_BATCH:
            raise ValueError(f"batch_size {self.batch_size} exceeds official H1 max {OFFICIAL_BATCH}")
        payload = load_payload(model_path)
        if payload.get("schema") != PAYLOAD_SCHEMA:
            raise ValueError(f"unsupported payload schema {payload.get('schema')}")
        if int(payload.get("context_bins", -1)) != CONTEXT_BINS:
            raise ValueError("payload context_bins must be 300")
        if payload.get("identity") != "static" or payload.get("identity_interface") != "static_identity_table":
            raise ValueError("payload must be static_identity_table")
        if payload.get("tier") != "learned_slope" or payload.get("ladder") != "default":
            raise ValueError("payload must be learned_slope default-ladder")
        self.official_tags = set(payload["official_dataset_tags"])
        if len(self.official_tags) != EXPECTED_SESSION_COUNT:
            raise ValueError("H1 payload must cover exactly 27 official dataset tags")
        self.behavior_scaling_factor = float(payload["behavior_scaling_factor"])
        if abs(self.behavior_scaling_factor - BEHAVIOR_SCALE) > 1.0e-12:
            raise ValueError("behavior scale must be 20")
        recency_cfg = recency_config_from_payload(payload)
        self.decoder = build_decoder(recency_cfg)
        state = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in payload["ema_state_dict"].items()}
        missing, unexpected = self.decoder.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise ValueError(f"state_dict mismatch missing={missing} unexpected={unexpected}")
        if bool(self.decoder.static_carrier.any()):
            raise ValueError("static carrier must stay literal zero")
        self.decoder.eval()
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        self.device = torch.device("cpu")
        self.decoder = self.decoder.to(self.device)
        self._runtime: CpuStaticRiftRuntime | None = None
        self._n_predicts = 0
        self.kind = "rift_cached_static_identity"
        self.window_size = CONTEXT_BINS

    @staticmethod
    def _stem(value) -> str:
        return Path(value).stem

    def reset(self, dataset_tags: Iterable[Path] = (Path(""),)):
        hashed_tags = [self.task_config.hash_dataset(self._stem(value)) for value in dataset_tags]
        missing = [tag for tag in hashed_tags if tag not in self.official_tags]
        if missing:
            raise ValueError(f"Dataset tags {missing} are absent from the H1 static payload")
        self._runtime = CpuStaticRiftRuntime(self.decoder, [str(tag) for tag in hashed_tags])
        self._n_predicts = 0

    def on_done(self, dones: np.ndarray):
        return None

    def set_batch_size(self, batch_size: int):
        batch_size = int(batch_size)
        if batch_size > OFFICIAL_BATCH:
            raise ValueError(f"batch_size {batch_size} exceeds official H1 max {OFFICIAL_BATCH}")
        self.batch_size = batch_size
        self._runtime = None

    def observe(self, neural_observations: np.ndarray):
        return None

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        if self._runtime is None:
            raise RuntimeError("reset(dataset_tags) must be called before predict")
        observations = np.asarray(neural_observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != CHANNELS:
            raise ValueError(f"Expected neural observations [B,{CHANNELS}], got {observations.shape}")
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
            raise RuntimeError("non-finite H1 static prediction")
        self._n_predicts += 1
        return native.astype(np.float32, copy=False)


def smoke(payload_path: str, window_path: str) -> None:
    from falcon_challenge.config import FalconConfig, FalconTask

    bundle = np.load(window_path)
    config = FalconConfig(task=FalconTask.h1)
    decoder = H1RiftLearnableFalconDecoder(task_config=config, model_path=payload_path, batch_size=1)
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-payload", default="")
    parser.add_argument("--smoke-window", default="")
    args = parser.parse_args()
    if args.smoke_payload and args.smoke_window:
        smoke(args.smoke_payload, args.smoke_window)
    else:
        raise SystemExit("use --smoke-payload and --smoke-window")
