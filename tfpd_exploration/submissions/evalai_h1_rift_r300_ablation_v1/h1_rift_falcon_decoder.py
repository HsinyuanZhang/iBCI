"""Falcon runtime for H1 RIFT R300 ACTIVITY_ONLY/NONE ablation + cached CPU backend.

Architecture is RIFT, backend is cached PyTorch FP32. Not BT-EORT, not ORT,
not SPINT. Banks are the sealed 27-tag signed_state14 T with rematerialized C2 E0.
"""
from __future__ import annotations

import argparse
import hashlib
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

PAYLOAD_SCHEMA = "h1_rift_r300_ablation_falcon_payload_v1"
ARMS = ("ACTIVITY_ONLY", "NONE")
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


def validate_payload(payload: dict) -> None:
    if payload.get("schema") != PAYLOAD_SCHEMA or payload.get("task") != "h1" or payload.get("arm") not in ARMS:
        raise ValueError("unsupported H1 ablation payload identity")
    banks = payload.get("bank_by_dataset_tag")
    canonical=payload.get('canonical_dataset_tags'); mapping=payload.get('canonical_tag_to_session')
    if not isinstance(canonical,list) or not isinstance(mapping,dict) or set(canonical)!=set(mapping): raise ValueError('canonical roster metadata missing')
    if not isinstance(banks, dict) or set(banks)!=set(canonical) or len(banks) != EXPECTED_SESSION_COUNT:
        raise ValueError("H1 ablation payload must cover exactly 27 official tags")
    for tag, row in banks.items():
        raw_e0=np.asarray(row.get("E0")); raw_t=np.asarray(row.get("T")); mask=np.ascontiguousarray(row.get("unit_mask"),np.bool_)
        if raw_e0.dtype != np.float32 or raw_t.dtype != np.float32: raise ValueError(f'{tag}: raw E0/T must be float32')
        e0=np.ascontiguousarray(raw_e0); t=np.ascontiguousarray(raw_t)
        if str(row.get('session')) != str(mapping[tag]): raise ValueError(f'{tag}: canonical session mapping mismatch')
        if e0.shape != (176,700) or t.shape != (176,4) or mask.shape != (176,) or e0.dtype != np.float32 or t.dtype != np.float32 or not np.isfinite(e0).all() or not np.isfinite(t).all() or not np.array_equal(t,np.zeros_like(t)):
            raise ValueError(f"{tag}: invalid sealed H1 ablation bank")
        digest=lambda a: hashlib.sha256(np.ascontiguousarray(a).dtype.str.encode()+str(np.ascontiguousarray(a).shape).encode()+np.ascontiguousarray(a).tobytes()).hexdigest()
        if row.get('e0_sha256') != digest(e0) or row.get('t_sha256') != digest(t): raise ValueError(f"{tag}: static E0/T SHA mismatch")
        if payload["arm"] == "NONE" and not np.array_equal(e0,np.zeros_like(e0)):
            raise ValueError(f"{tag}: NONE E0 must be literal zero")

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
            "estimator": "sealed H1 ACTIVITY_ONLY/NONE static identity (27-tag)",
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
        validate_payload(payload)
        if int(payload.get("context_bins", -1)) != CONTEXT_BINS:
            raise ValueError("payload context_bins must be 300")
        if payload.get("bias_mode") != "recency":
            raise ValueError("payload bias_mode must be recency")
        self.arm = str(payload["arm"])
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
    predicted = []
    for row in window:
        predicted.append(decoder.predict(row.reshape(1, -1)).copy())
        decoder.on_done(np.ones((1,), dtype=bool))
    pred = np.stack(predicted)
    expected = np.asarray(bundle["expected"], dtype=np.float32)
    # New fixtures contain each raw-bin output; accept legacy last-bin fixtures only for replay compatibility.
    target = expected if expected.ndim == pred.ndim else expected[None]
    actual = pred if expected.ndim == pred.ndim else pred[-1:]
    if actual.shape != target.shape or not np.allclose(actual, target, atol=1.0e-5, rtol=1.0e-5):
        raise RuntimeError(f"container smoke mismatch {actual} vs {target}")
    print(json.dumps({"status": "CONTAINER_SMOKE_PASS", "pred": pred[-1].tolist(), "kind": decoder.kind}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-payload", default="")
    parser.add_argument("--smoke-window", default="")
    args = parser.parse_args()
    if args.smoke_payload and args.smoke_window:
        smoke(args.smoke_payload, args.smoke_window)
    else:
        raise SystemExit("use --smoke-payload and --smoke-window")
