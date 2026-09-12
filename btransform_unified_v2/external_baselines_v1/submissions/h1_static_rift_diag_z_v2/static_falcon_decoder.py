"""Falcon runtime for fair_v2 same-capacity static-RIFT + neural frontend.

Same frozen EMA weights as the identity static control. Only the pre-local_conv
affine (diag-z or CORAL) changes. No WF smoothing. Cached CPU.
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

PAYLOAD_SCHEMA = "fair_v2_static_rift_frontend_v1"
TASK_SPEC = {
    "m1": {"channels": 64, "outputs": 16, "context": 100, "batch": 4, "scale": 1.0, "roster": 7},
    "m2": {"channels": 96, "outputs": 2, "context": 50, "batch": 7, "scale": 5.0, "roster": 13},
    "h1": {"channels": 176, "outputs": 7, "context": 300, "batch": 8, "scale": 20.0, "roster": 27},
}


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
    return config_from_run_meta({"learnable_config": raw}, payload["task"])


def apply_frontend(raw: np.ndarray, row: dict) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float64)
    if row["arm"] == "diag_z":
        return ((x - row["mt"]) / row["st"] * row["ss"] + row["ms"]).astype(np.float32)
    if row["arm"] == "coral":
        return ((x - row["mt"]) @ row["A"] + row["ms"]).astype(np.float32)
    raise ValueError("unsupported static frontend " + str(row.get("arm")))


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
        if bool(decoder.static_carrier.any()):
            raise ValueError("static carrier must stay literal zero")
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


class FairV2StaticRiftFalconDecoder(BCIDecoder):
    def __init__(self, task_config, model_path: str, batch_size: int = 1):
        super().__init__(task_config=task_config, batch_size=batch_size)
        self.task_config = task_config
        payload = load_payload(model_path)
        if payload.get("schema") != PAYLOAD_SCHEMA:
            raise ValueError("unsupported payload schema " + str(payload.get("schema")))
        self.task = str(payload["task"])
        spec = TASK_SPEC[self.task]
        self.batch_size = int(batch_size)
        if self.batch_size > spec["batch"]:
            raise ValueError(f"batch_size {self.batch_size} exceeds official {self.task} max {spec['batch']}")
        if int(payload.get("context_bins", -1)) != spec["context"]:
            raise ValueError("payload context_bins mismatch")
        if payload.get("identity") != "static" or payload.get("identity_interface") != "static_identity_table":
            raise ValueError("payload must be static_identity_table")
        if payload.get("frontend") not in {"diag_z", "coral"}:
            raise ValueError("payload frontend must be diag_z or coral")
        self.transforms = payload["session_transforms"]
        if len(self.transforms) != spec["roster"]:
            raise ValueError(f"{self.task} payload must cover exactly {spec['roster']} official dataset tags")
        self.behavior_scaling_factor = float(payload["behavior_scaling_factor"])
        if abs(self.behavior_scaling_factor - spec["scale"]) > 1.0e-12:
            raise ValueError("behavior scale mismatch")
        recency_cfg = recency_config_from_payload(payload)
        seed = int(payload.get("seed", 42))
        self.decoder = StaticLearnableRiftDecoder(self.task, recency_cfg, seed=seed)
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
        self._prefixes: list[str] = []
        self._n_predicts = 0
        self.kind = f"fair_v2_static_rift_{payload['frontend']}_cached"
        self.window_size = spec["context"]
        self._channels = spec["channels"]

    @staticmethod
    def _stem(value) -> str:
        return Path(value).stem

    def reset(self, dataset_tags: Iterable[Path] = (Path(""),)):
        hashed = [self.task_config.hash_dataset(self._stem(value)) for value in dataset_tags]
        missing = [tag for tag in hashed if tag not in self.transforms]
        if missing:
            raise ValueError(f"Dataset tags {missing} are absent from the static frontend payload")
        self._prefixes = [str(tag) for tag in hashed]
        self._runtime = CpuStaticRiftRuntime(self.decoder, self._prefixes)
        self._n_predicts = 0

    def on_done(self, dones: np.ndarray):
        return None

    def set_batch_size(self, batch_size: int):
        spec = TASK_SPEC[self.task]
        batch_size = int(batch_size)
        if batch_size > spec["batch"]:
            raise ValueError(f"batch_size {batch_size} exceeds official max {spec['batch']}")
        self.batch_size = batch_size
        self._runtime = None

    def observe(self, neural_observations: np.ndarray):
        return None

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        if self._runtime is None:
            raise RuntimeError("reset(dataset_tags) must be called before predict")
        observations = np.asarray(neural_observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != self._channels:
            raise ValueError(f"Expected neural observations [B,{self._channels}], got {observations.shape}")
        active = observations.shape[0]
        n_reg = len(self._runtime.ids)
        if active > n_reg:
            raise ValueError(f"evaluator batch {active} exceeds reset roster {n_reg}")
        calibrated = np.zeros((n_reg, self._channels), dtype=np.float32)
        for i in range(active):
            calibrated[i] = apply_frontend(observations[i], self.transforms[self._prefixes[i]])
        valid = torch.zeros(n_reg, dtype=torch.bool)
        valid[:active] = True
        with torch.inference_mode():
            pred = self._runtime.advance(torch.from_numpy(np.ascontiguousarray(calibrated)), valid_mask=valid)
        native = pred[:active].detach().cpu().numpy() / self.behavior_scaling_factor
        if not np.isfinite(native).all():
            raise RuntimeError("non-finite static-RIFT prediction")
        self._n_predicts += 1
        return native.astype(np.float32, copy=False)


def smoke(payload_path: str, window_path: str) -> None:
    from falcon_challenge.config import FalconConfig, FalconTask

    bundle = np.load(window_path, allow_pickle=False)
    task = str(bundle["task"])
    config = FalconConfig(task=getattr(FalconTask, task))
    decoder = FairV2StaticRiftFalconDecoder(task_config=config, model_path=payload_path, batch_size=1)
    decoder.reset(dataset_tags=[str(bundle["tag_stem"])])
    window = np.asarray(bundle["window"], dtype=np.float32)
    pred = None
    for row in window:
        pred = decoder.predict(row.reshape(1, -1))
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
