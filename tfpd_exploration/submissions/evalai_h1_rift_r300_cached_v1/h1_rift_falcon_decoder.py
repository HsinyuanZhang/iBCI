"""Falcon runtime for H1 RIFT R300 recency + cached CPU backend.

Architecture is RIFT, backend is cached PyTorch FP32. Not BT-EORT, not ORT,
not SPINT. Banks are the sealed C2-CAL-1 official 27-tag payload.
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from falcon_challenge.interface import BCIDecoder

from btransform_unified_v1.bank import TaskBank
from btransform_unified_v2.cpu_runtime import CpuRiftRuntime
from btransform_unified_v2.model import RiftDecoder

PAYLOAD_SCHEMA = "h1_rift_r300_cached_falcon_payload_v1"
CHANNELS = 176
BEHAVIOR_SCALE = 20.0
EXPECTED_SESSION_COUNT = 27
OFFICIAL_BATCH = 8
CONTEXT_BINS = 300


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
        target_store=np.zeros((0, 7), dtype=np.float32),
        window_ids=np.zeros(0, dtype=np.int64),
        calibration_meta={
            "shape": tuple(e0.shape),
            "trial_count": 3,
            "estimator": "C2-CAL-1 B2 official H1 bank (sealed 27-tag payload)",
            "array_sha256": str(row.get("e0_sha256") or "payload"),
            "budget": 3,
        },
    )


class H1RiftCachedFalconDecoder(BCIDecoder):
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
        if payload.get("bias_mode") != "recency":
            raise ValueError("payload bias_mode must be recency")
        self.bank_by_dataset_tag = payload["bank_by_dataset_tag"]
        if len(self.bank_by_dataset_tag) != EXPECTED_SESSION_COUNT:
            raise ValueError("H1 payload must cover exactly 27 official dataset tags")
        self.behavior_scaling_factor = float(payload["behavior_scaling_factor"])
        if abs(self.behavior_scaling_factor - BEHAVIOR_SCALE) > 1.0e-12:
            raise ValueError("behavior scale must be 20")
        self.decoder = RiftDecoder("h1", context_bins=CONTEXT_BINS, bias_mode="recency", seed=42, proj_dim=16)
        state = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in payload["ema_state_dict"].items()}
        missing, unexpected = self.decoder.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise ValueError(f"state_dict mismatch missing={missing} unexpected={unexpected}")
        self.decoder.eval()
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        self.device = torch.device("cpu")
        self.decoder = self.decoder.to(self.device)
        self._runtime: CpuRiftRuntime | None = None
        self._n_predicts = 0
        self.kind = "rift_cached"
        self.window_size = CONTEXT_BINS

    @staticmethod
    def _stem(value) -> str:
        return Path(value).stem

    def reset(self, dataset_tags: Iterable[Path] = (Path(""),)):
        hashed_tags = [self.task_config.hash_dataset(self._stem(value)) for value in dataset_tags]
        missing = [tag for tag in hashed_tags if tag not in self.bank_by_dataset_tag]
        if missing:
            raise ValueError(f"Dataset tags {missing} are absent from the H1 RIFT bank payload")
        banks = [_task_bank(tag, self.bank_by_dataset_tag[tag]) for tag in hashed_tags]
        ids = [str(tag) for tag in hashed_tags]
        self._runtime = CpuRiftRuntime(self.decoder, banks, ids, temporal_backend="cached")
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
            raise RuntimeError("non-finite H1 RIFT prediction")
        self._n_predicts += 1
        return native.astype(np.float32, copy=False)


def smoke(payload_path: str, window_path: str) -> None:
    from falcon_challenge.config import FalconConfig, FalconTask

    bundle = np.load(window_path)
    config = FalconConfig(task=FalconTask.h1)
    decoder = H1RiftCachedFalconDecoder(task_config=config, model_path=payload_path, batch_size=1)
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
