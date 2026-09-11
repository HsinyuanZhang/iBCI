"""Falcon runtime for the static M2 RIFT-R50 ACTIVITY_ONLY/NONE controls.

It loads the selected EMA of the original RIFT concat decoder, but its only
identity inputs are sealed static E0/T/mask arrays.  FULL identity material,
joint encoders, ORT, and SPINT are not present in this runtime.
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
from btransform_unified_v2.concat_model import RiftConcatDecoder
from btransform_unified_v2.cpu_runtime import CpuRiftRuntime

PAYLOAD_SCHEMA = "m2_rift_r50_ablation_cached_falcon_payload_v1"
CHANNELS = 96
BEHAVIOR_SCALE = 5.0
EXPECTED_TAGS = frozenset((
    "Run1_20201019", "Run2_20201019", "Run1_20201020", "Run2_20201020",
    "Run1_20201027", "Run2_20201027", "Run1_20201028",
    "Run1_20201030", "Run2_20201030", "Run1_20201118", "Run1_20201119",
    "Run1_20201124", "Run2_20201124",
))
EXPECTED_SESSION_COUNT = len(EXPECTED_TAGS)
OFFICIAL_BATCH = 7
CONTEXT_BINS = 50


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda value: torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_payload(path: str | Path) -> dict:
    with open(path, "rb") as handle:
        payload = CPUUnpickler(handle).load()
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dictionary")
    return payload


def _validate_bank(tag: str, row: dict, arm: str) -> None:
    if set(row) - {"session", "E0", "T", "unit_mask", "e0_sha256", "t_sha256"}:
        raise ValueError(f"{tag}: FULL-carrier or unknown bank fields are forbidden")
    e0=np.asarray(row.get("E0")); t=np.asarray(row.get("T")); mask=np.asarray(row.get("unit_mask"))
    if e0.shape != (CHANNELS, CONTEXT_BINS) or t.shape != (CHANNELS, 4) or mask.shape != (CHANNELS,):
        raise ValueError(f"{tag}: expected E0[96,50], T[96,4], mask[96]")
    if e0.dtype != np.float32 or t.dtype != np.float32 or not np.isfinite(e0).all() or not np.isfinite(t).all():
        raise ValueError(f"{tag}: banks must be finite float32")
    digest=lambda x: __import__("hashlib").sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    if row.get("e0_sha256") != digest(e0) or row.get("t_sha256") != digest(t): raise ValueError(f"{tag}: static bank SHA mismatch")
    if arm == "NONE" and (np.any(e0) or np.any(t)): raise ValueError(f"{tag}: NONE must be all zero")
    if arm == "ACTIVITY_ONLY" and np.any(t): raise ValueError(f"{tag}: ACTIVITY_ONLY T must be zero")

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
        target_store=np.zeros((0, 2), dtype=np.float32),
        window_ids=np.zeros(0, dtype=np.int64),
        calibration_meta={
            "shape": tuple(e0.shape),
            "trial_count": 33,
            "estimator": "sealed static M2 ablation bank (E0/T/mask only)",
            "array_sha256": str(row.get("e0_sha256") or "payload"),
            "budget": 33,
        },
    )


class M2RiftCachedFalconDecoder(BCIDecoder):
    def __init__(self, task_config, model_path: str, batch_size: int = 1):
        super().__init__(task_config=task_config, batch_size=batch_size)
        self.task_config = task_config
        self.batch_size = int(batch_size)
        if self.batch_size > OFFICIAL_BATCH:
            raise ValueError(f"batch_size {self.batch_size} exceeds official M2 max {OFFICIAL_BATCH}")
        payload = load_payload(model_path)
        if payload.get("schema") != PAYLOAD_SCHEMA:
            raise ValueError(f"unsupported payload schema {payload.get('schema')}")
        if int(payload.get("context_bins", -1)) != CONTEXT_BINS:
            raise ValueError("payload context_bins must be 50")
        if payload.get("bias_mode") != "recency":
            raise ValueError("payload bias_mode must be recency")
        if payload.get("identity_interface") != "concat":
            raise ValueError("payload identity_interface must bind original concat RIFT decoder")
        self.arm = payload.get("arm")
        if self.arm not in ("ACTIVITY_ONLY", "NONE"): raise ValueError("invalid ablation arm")
        self.bank_by_dataset_tag = payload["bank_by_dataset_tag"]
        if not isinstance(self.bank_by_dataset_tag, dict):
            raise ValueError("bank_by_dataset_tag must be a dictionary")
        if len(self.bank_by_dataset_tag) != EXPECTED_SESSION_COUNT:
            raise ValueError("M2 payload must cover exactly 13 official dataset tags")
        canonical = set()
        for tag, row in self.bank_by_dataset_tag.items():
            if not isinstance(row, dict):
                raise ValueError(f"{tag}: bank row must be a dictionary")
            _validate_bank(str(tag), row, self.arm)
            session = str(row.get("session", ""))
            pieces = session.split("-")
            if len(pieces) != 5 or pieces[0] != "ses" or f"{pieces[-1]}_{''.join(pieces[1:4])}" != str(tag):
                raise ValueError(f"{tag}: noncanonical session/tag mapping")
            canonical.add(str(tag))
        if canonical != EXPECTED_TAGS:
            raise ValueError(
                f"M2 payload tags differ from fixed official roster: "
                f"missing={sorted(EXPECTED_TAGS-canonical)} extra={sorted(canonical-EXPECTED_TAGS)}"
            )
        self.behavior_scaling_factor = float(payload["behavior_scaling_factor"])
        if abs(self.behavior_scaling_factor - BEHAVIOR_SCALE) > 1.0e-12:
            raise ValueError("behavior scale must be 5")
        self.decoder = RiftConcatDecoder("m2", context_bins=CONTEXT_BINS, bias_mode="recency", seed=42)
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
            raise ValueError(f"Dataset tags {missing} are absent from the M2 RIFT bank payload")
        banks = [_task_bank(tag, self.bank_by_dataset_tag[tag]) for tag in hashed_tags]
        ids = [str(tag) for tag in hashed_tags]
        self._runtime = CpuRiftRuntime(self.decoder, banks, ids, temporal_backend="cached")
        self._n_predicts = 0

    def on_done(self, dones: np.ndarray):
        return None

    def set_batch_size(self, batch_size: int):
        batch_size = int(batch_size)
        if batch_size > OFFICIAL_BATCH:
            raise ValueError(f"batch_size {batch_size} exceeds official M2 max {OFFICIAL_BATCH}")
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
            raise RuntimeError("non-finite M2 RIFT prediction")
        self._n_predicts += 1
        return native.astype(np.float32, copy=False)


def smoke(payload_path: str, window_path: str) -> None:
    from falcon_challenge.config import FalconConfig, FalconTask

    bundle = np.load(window_path)
    config = FalconConfig(task=FalconTask.m2)
    decoder = M2RiftCachedFalconDecoder(task_config=config, model_path=payload_path, batch_size=1)
    decoder.reset(dataset_tags=[str(bundle["tag_stem"])])
    window = np.asarray(bundle["window"], dtype=np.float32)
    predictions = []
    for row in window:
        predictions.append(decoder.predict(row.reshape(1, -1)))
        decoder.on_done(np.ones((1,), dtype=bool))
    pred = np.concatenate(predictions, axis=0) if predictions else None
    expected = np.asarray(bundle["expected"], dtype=np.float32)
    # The current fixture is a whole raw 40-bin trace.  Accept the historic
    # one-bin fixture only to make an old locally built image diagnosable.
    compare = pred if pred is not None and pred.shape == expected.shape else (pred[-1] if pred is not None else None)
    if compare is None or compare.shape != expected.shape or not np.allclose(compare, expected, atol=1.0e-5, rtol=1.0e-5):
        raise RuntimeError(f"container smoke mismatch {compare} vs {expected}")
    print(json.dumps({"status": "CONTAINER_SMOKE_PASS", "pred": compare.tolist(), "kind": decoder.kind}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-payload", default="")
    parser.add_argument("--smoke-window", default="")
    args = parser.parse_args()
    if args.smoke_payload and args.smoke_window:
        smoke(args.smoke_payload, args.smoke_window)
    else:
        raise SystemExit("use --smoke-payload and --smoke-window")
