"""FALCON runtime for the clean all-source M1 AFC4 Full/B4 payloads.

The online evaluator supplies only neural observations and a dataset tag.  All
EMG support fitting and identity construction happen offline in
``export_afc4_payload.py``; this decoder never performs calibration, a query
read, or a backward pass.
"""
from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from falcon_challenge.interface import BCIDecoder

from third_party.falcon_challenge.filtering import NEURAL_TAU_MS, apply_exponential_filter


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda value: torch.load(
                io.BytesIO(value), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


class M1AFC4CachedIdentityDecoder(BCIDecoder):
    """Cached-identity decoder bound to the official M1 Full/B4 schema."""

    def __init__(self, task_config, model_path: str, batch_size: int = 4):
        super().__init__(task_config=task_config, batch_size=batch_size)
        # ``BCIDecoder`` stores the configuration privately as ``_task_config``;
        # the runtime methods below intentionally use the public local alias so
        # reset/observe can hash dataset tags and validate channel dimensions.
        self.task_config = task_config
        with open(model_path, "rb") as handle:
            payload = CPUUnpickler(handle).load()
        if payload.get("schema_version") != "evalai_m1_afc4_cached_identity_v1":
            raise ValueError("Unsupported M1 AFC4 cached-identity payload")
        if payload.get("task") != task_config.task or payload.get("arm") not in {"full", "b4"}:
            raise ValueError("Payload task/arm does not match M1 AFC4 Full/B4 runtime")
        self.arm = str(payload["arm"])
        self.decoder = payload["decoder"].eval()
        self.identity_by_dataset_tag = payload["identity_by_dataset_tag"]
        self.window_size = int(payload["window_size"])
        self.behavior_scaling_factor = float(payload.get("behavior_scaling_factor", 1.0))
        self.smooth_observations = bool(payload.get("smooth_observations", False))
        if set(self.identity_by_dataset_tag) != set(payload.get("dataset_tags", self.identity_by_dataset_tag)):
            raise ValueError("payload dataset tag index is inconsistent")
        if len(self.identity_by_dataset_tag) != 7:
            raise ValueError("M1 AFC4 payload must cover exactly seven public calibration tags")
        for tag, identity in self.identity_by_dataset_tag.items():
            value = np.asarray(identity, dtype=np.float32)
            if value.shape != (task_config.n_channels, 100) or not np.isfinite(value).all():
                raise ValueError(f"invalid cached identity for tag {tag}: {value.shape}")
        max_history = int(NEURAL_TAU_MS / task_config.bin_size_ms) * 5
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
            raise ValueError(f"Dataset tags absent from {self.arm.upper()} payload: {missing}")
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


__all__ = ["M1AFC4CachedIdentityDecoder"]
