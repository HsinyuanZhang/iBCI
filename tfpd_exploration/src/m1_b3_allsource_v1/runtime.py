"""FALCON runtime for all-source B3 / B3S-rSyn3 cached-identity images.

Mirrors the decoder that produced official M1 submissions 578244-578247 and
the M2 cached-identity image registered as submission 581644: reset selects a
precomputed identity; predict decodes one window; on_done is a no-op.
"""
from __future__ import annotations

import io
import os
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from falcon_challenge.interface import BCIDecoder

PAYLOAD_SCHEMA = "m1_b3_allsource_cached_identity_v1"
EXPECTED_ARMS = ("b3", "b3s_rsyn3", "b3s_rsyn3_freeze_top4", "b3s_rsyn3_acyc_top4")
EXPECTED_SESSION_COUNT = 7
WINDOW_SIZE = 100
CHANNELS = 64
IDENTITY_DIM = 100


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda value: torch.load(
                io.BytesIO(value), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


def _require_payload(payload: Mapping[str, Any], *, expected_arm: str, task: Any) -> None:
    if payload.get("schema_version") != PAYLOAD_SCHEMA:
        raise ValueError("Unsupported cached-identity payload")
    if payload.get("task") != task:
        raise ValueError("payload task does not match evaluator task")
    if expected_arm not in EXPECTED_ARMS:
        raise ValueError(f"unsupported runtime arm {expected_arm!r}")
    if payload.get("arm") != expected_arm:
        raise ValueError(f"payload arm {payload.get('arm')!r} is not {expected_arm!r}")
    identities = payload.get("identity_by_dataset_tag")
    if not isinstance(identities, dict) or len(identities) != EXPECTED_SESSION_COUNT:
        raise ValueError("M1 payload must cover exactly seven public calibration sessions")
    if int(payload.get("window_size", -1)) != WINDOW_SIZE:
        raise ValueError("payload window_size is not 100")


class AllSourceCachedIdentityDecoder(BCIDecoder):
    """Decode with one chronological-M10 identity per session."""

    def __init__(self, task_config, model_path: str, batch_size: int = 1, expected_arm: str | None = None):
        super().__init__(task_config=task_config, batch_size=batch_size)
        self.task_config = task_config
        self.batch_size = int(batch_size)
        arm = expected_arm or os.environ.get("ARM")
        if arm not in EXPECTED_ARMS:
            raise ValueError(f"runtime ARM must be one of {EXPECTED_ARMS}, got {arm!r}")
        with open(model_path, "rb") as handle:
            payload = CPUUnpickler(handle).load()
        _require_payload(payload, expected_arm=arm, task=task_config.task)
        self.arm = arm
        self.decoder = payload["decoder"].eval()
        self.identity_by_dataset_tag = payload["identity_by_dataset_tag"]
        self.window_size = int(payload["window_size"])
        self.behavior_scaling_factor = float(payload["behavior_scaling_factor"])
        self.smooth_observations = bool(payload.get("smooth_observations", False))
        for tag, identity in self.identity_by_dataset_tag.items():
            array = np.asarray(identity)
            if array.shape != (CHANNELS, IDENTITY_DIM):
                raise ValueError(f"identity {tag} has shape {array.shape}")
        max_history = int(240.0 / task_config.bin_size_ms) * 5
        self.raw_history_buffer = np.zeros(
            (max_history, self.batch_size, task_config.n_channels), dtype=np.float32
        )
        self.observation_buffer = np.zeros(
            (self.window_size, self.batch_size, task_config.n_channels), dtype=np.float32
        )
        self.device = torch.device("cpu")
        self.local_identities: list[torch.Tensor] = []

    @staticmethod
    def _stem(value) -> str:
        return Path(value).stem

    def reset(self, dataset_tags: Iterable[Path] = (Path(""),)):
        hashed_tags = [
            self.task_config.hash_dataset(self._stem(value)) for value in dataset_tags
        ]
        missing = [tag for tag in hashed_tags if tag not in self.identity_by_dataset_tag]
        if missing:
            raise ValueError(f"Dataset tags absent from {self.arm} payload: {missing}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder = self.decoder.to(self.device).eval()
        self.local_identities = [
            torch.as_tensor(
                self.identity_by_dataset_tag[tag], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            for tag in hashed_tags
        ]
        self.raw_history_buffer.fill(0.0)
        self.observation_buffer.fill(0.0)

    def on_done(self, dones: np.ndarray):
        return None

    def _decode(self, neural: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        source = neural.permute(0, 2, 1) + identity
        source = self.decoder.fc_in(source)
        query = self.decoder.fc_in(self.decoder.rep).to(source)
        transformed, _ = self.decoder.transformer(
            query.repeat(source.shape[0], 1, 1), source
        )
        return self.decoder.fc_out(transformed).permute(0, 2, 1)

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        if not self.local_identities:
            raise RuntimeError("reset(dataset_tags) must be called before predict")
        self.observe(neural_observations)
        decoder_input = torch.as_tensor(
            self.observation_buffer.copy().transpose(1, 0, 2),
            dtype=torch.float32,
            device=self.device,
        )
        predictions = []
        with torch.inference_mode():
            for index, identity in enumerate(self.local_identities):
                predictions.append(self._decode(decoder_input[index : index + 1], identity))
        prediction = torch.cat(predictions, dim=0)
        return prediction[:, -1, :].cpu().numpy() / self.behavior_scaling_factor

    def observe(self, neural_observations: np.ndarray):
        observations = np.asarray(neural_observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != self.task_config.n_channels:
            raise ValueError(
                f"Expected neural observations [B,{self.task_config.n_channels}], got {observations.shape}"
            )
        if observations.shape[0] > self.batch_size:
            raise ValueError("Evaluator batch exceeds configured decoder batch size")
        if observations.shape[0] < self.batch_size:
            observations = np.pad(
                observations, ((0, self.batch_size - observations.shape[0]), (0, 0))
            )
        self.raw_history_buffer = np.roll(self.raw_history_buffer, -1, axis=0)
        self.raw_history_buffer[-1] = observations
        if self.smooth_observations:
            from third_party.falcon_challenge.filtering import NEURAL_TAU_MS, apply_exponential_filter

            history, batch, channels = self.raw_history_buffer.shape
            flat = self.raw_history_buffer.reshape(history, batch * channels)
            smoothed = apply_exponential_filter(
                flat, tau=NEURAL_TAU_MS, bin_size=self.task_config.bin_size_ms
            )
            latest = smoothed[-1].reshape(batch, channels).astype(np.float32)
        else:
            latest = observations
        self.observation_buffer = np.roll(self.observation_buffer, -1, axis=0)
        self.observation_buffer[-1] = latest
