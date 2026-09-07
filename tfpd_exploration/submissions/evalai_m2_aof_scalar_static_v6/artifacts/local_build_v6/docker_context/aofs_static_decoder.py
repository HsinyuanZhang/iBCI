"""Frozen AOF-S M2 decoder.

The runtime owns no calibration/trial state.  It holds two immutable identity
maps generated offline from the same act30/D-opt4 inputs and makes *two*
independent frozen decoder calls for every governed prediction.  The separate
``beta=+0`` control is intentionally a direct native return and never touches
the post identity.
"""
from __future__ import annotations

import io
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPINT_MAIN = _REPO_ROOT / "SPINT-main"
if _SPINT_MAIN.is_dir() and str(_SPINT_MAIN) not in sys.path:
    # Host-only import bootstrap.  The pinned container already provides this
    # evaluator dependency and does not use the host path.
    sys.path.append(str(_SPINT_MAIN))
from falcon_challenge.interface import BCIDecoder
from third_party.falcon_challenge.filtering import NEURAL_TAU_MS, apply_exponential_filter

try:  # package import in local validation/tests
    from .laws import BEHAVIOR_SCALING_FACTOR, FROZEN_BETA, exact_positive_zero, validate_identity_map
except ImportError:  # copied flat into the immutable evaluator container
    from laws import BEHAVIOR_SCALING_FACTOR, FROZEN_BETA, exact_positive_zero, validate_identity_map

PAYLOAD_SCHEMA = "e8_t4_m2_aofs_static_decoded_output_v1"
PAYLOAD_ARM = "aofs_static_act30_dopt4"
EXPECTED_SESSION_COUNT = 13


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda value: torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


class AofsStaticM2Decoder(BCIDecoder):
    """Offline-cached scalar decoded-output fusion, with no online state."""

    def __init__(self, task_config, model_path: str, batch_size: int = 1):
        super().__init__(task_config=task_config, batch_size=batch_size)
        self.task_config = task_config
        self.batch_size = int(batch_size)
        with open(model_path, "rb") as handle:
            payload = CPUUnpickler(handle).load()
        if payload.get("schema_version") != PAYLOAD_SCHEMA:
            raise ValueError("unsupported AOF-S payload schema")
        if payload.get("task") != task_config.task:
            raise ValueError("AOF-S payload task mismatch")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("arm") != PAYLOAD_ARM:
            raise ValueError("AOF-S payload arm mismatch")
        if float(metadata.get("frozen_beta", "nan")) != FROZEN_BETA:
            raise ValueError("AOF-S payload beta mismatch")
        if float(payload.get("behavior_scaling_factor", "nan")) != BEHAVIOR_SCALING_FACTOR:
            raise ValueError("AOF-S behavior scale mismatch")
        if int(payload.get("window_size", -1)) != 50:
            raise ValueError("AOF-S window law mismatch")
        native = payload.get("native_identity_by_dataset_tag")
        post = payload.get("post_identity_by_dataset_tag")
        native_digests = validate_identity_map(native, expected_tags=EXPECTED_SESSION_COUNT)
        post_digests = validate_identity_map(post, expected_tags=EXPECTED_SESSION_COUNT)
        if set(native_digests) != set(post_digests):
            raise ValueError("AOF-S native/post tag coverage mismatch")
        self.decoder = payload["decoder"].eval()
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.native_identity_by_dataset_tag = native
        self.post_identity_by_dataset_tag = post
        self.window_size = 50
        self.behavior_scaling_factor = BEHAVIOR_SCALING_FACTOR
        self.smooth_observations = bool(payload.get("smooth_observations", False))
        history = int(NEURAL_TAU_MS / task_config.bin_size_ms) * 5
        self.raw_history_buffer = np.zeros((history, self.batch_size, task_config.n_channels), dtype=np.float32)
        self.observation_buffer = np.zeros((self.window_size, self.batch_size, task_config.n_channels), dtype=np.float32)
        self.device = torch.device("cpu")
        self.local_native: list[torch.Tensor] = []
        self.local_post: list[torch.Tensor] = []
        self._parameter_fingerprint = self.parameter_fingerprint()

    @staticmethod
    def _stem(value) -> str:
        return Path(value).stem

    def parameter_fingerprint(self) -> str:
        import hashlib
        digest = hashlib.sha256()
        with torch.no_grad():
            for parameter in self.decoder.parameters():
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def identity_fingerprints(self) -> dict[str, str]:
        import hashlib
        result: dict[str, str] = {}
        for name, mapping in (("native", self.native_identity_by_dataset_tag), ("post", self.post_identity_by_dataset_tag)):
            digest = hashlib.sha256()
            for tag, value in sorted(mapping.items()):
                array = np.ascontiguousarray(value)
                digest.update(tag.encode("utf-8"))
                digest.update(array.tobytes(order="C"))
            result[name] = digest.hexdigest()
        return result

    def reset(self, dataset_tags: Iterable[Path] = (Path(""),)):
        tags = [self.task_config.hash_dataset(self._stem(value)) for value in dataset_tags]
        if any(tag not in self.native_identity_by_dataset_tag for tag in tags):
            raise ValueError("AOF-S payload lacks one or more requested dataset tags")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder = self.decoder.to(self.device).eval()
        self.local_native = [torch.as_tensor(self.native_identity_by_dataset_tag[tag], dtype=torch.float32, device=self.device).unsqueeze(0) for tag in tags]
        self.local_post = [torch.as_tensor(self.post_identity_by_dataset_tag[tag], dtype=torch.float32, device=self.device).unsqueeze(0) for tag in tags]
        self.raw_history_buffer.fill(0.0)
        self.observation_buffer.fill(0.0)

    def on_done(self, dones: np.ndarray):
        return None

    def _decode_with_identity(self, neural: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        source = neural.permute(0, 2, 1) + identity
        source = self.decoder.fc_in(source)
        query = self.decoder.fc_in(self.decoder.rep).to(source)
        transformed, _ = self.decoder.transformer(query.repeat(source.shape[0], 1, 1), source)
        return self.decoder.fc_out(transformed).permute(0, 2, 1)

    def observe(self, neural_observations: np.ndarray):
        observations = np.asarray(neural_observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != self.task_config.n_channels:
            raise ValueError("AOF-S observation shape mismatch")
        if observations.shape[0] > self.batch_size:
            raise ValueError("AOF-S evaluator batch exceeds configured batch size")
        if observations.shape[0] < self.batch_size:
            observations = np.pad(observations, ((0, self.batch_size - observations.shape[0]), (0, 0)))
        self.raw_history_buffer = np.roll(self.raw_history_buffer, -1, axis=0)
        self.raw_history_buffer[-1] = observations
        if self.smooth_observations:
            history, batch, channels = self.raw_history_buffer.shape
            latest = apply_exponential_filter(self.raw_history_buffer.reshape(history, batch * channels), tau=NEURAL_TAU_MS, bin_size=self.task_config.bin_size_ms)[-1].reshape(batch, channels).astype(np.float32)
        else:
            latest = observations
        self.observation_buffer = np.roll(self.observation_buffer, -1, axis=0)
        self.observation_buffer[-1] = latest

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        if not self.local_native:
            raise RuntimeError("reset(dataset_tags) must be called before predict")
        self.observe(neural_observations)
        windows = torch.as_tensor(self.observation_buffer.copy().transpose(1, 0, 2), dtype=torch.float32, device=self.device)
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for index, native_identity in enumerate(self.local_native):
                # Governing AOF-S path: two *independent* B calls on the same window.
                native = self._decode_with_identity(windows[index:index + 1], native_identity)
                post = self._decode_with_identity(windows[index:index + 1], self.local_post[index])
                # The source-reference arithmetic is intentionally not a torch
                # fused expression: each final-bin output crosses to a
                # contiguous NumPy float32 array, is independently scaled by
                # float32(5), then the literal scalar fusion is evaluated.
                native_np = np.ascontiguousarray(native[:, -1, :].cpu().numpy(), dtype=np.float32)
                post_np = np.ascontiguousarray(post[:, -1, :].cpu().numpy(), dtype=np.float32)
                native_scaled = np.ascontiguousarray(native_np / np.float32(BEHAVIOR_SCALING_FACTOR), dtype=np.float32)
                post_scaled = np.ascontiguousarray(post_np / np.float32(BEHAVIOR_SCALING_FACTOR), dtype=np.float32)
                outputs.append(np.ascontiguousarray(native_scaled + FROZEN_BETA * (post_scaled - native_scaled), dtype=np.float32))
        return np.ascontiguousarray(np.concatenate(outputs, axis=0), dtype=np.float32)

    def predict_with_beta_for_validation(self, neural_observations: np.ndarray, beta: float) -> np.ndarray:
        """Local validation hook; not used by the public evaluator path."""
        if not self.local_native:
            raise RuntimeError("reset(dataset_tags) must be called before predict")
        if float(beta) == 0.0 and not exact_positive_zero(beta):
            raise ValueError("negative zero is not a validation sentinel")
        self.observe(neural_observations)
        windows = torch.as_tensor(self.observation_buffer.copy().transpose(1, 0, 2), dtype=torch.float32, device=self.device)
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for index, native_identity in enumerate(self.local_native):
                native = self._decode_with_identity(windows[index:index + 1], native_identity)
                native_np = np.ascontiguousarray(native[:, -1, :].cpu().numpy(), dtype=np.float32)
                native_scaled = np.ascontiguousarray(native_np / np.float32(BEHAVIOR_SCALING_FACTOR), dtype=np.float32)
                if exact_positive_zero(beta):
                    outputs.append(native_scaled)  # direct native sentinel; no post call / multiply-add
                else:
                    post = self._decode_with_identity(windows[index:index + 1], self.local_post[index])
                    post_np = np.ascontiguousarray(post[:, -1, :].cpu().numpy(), dtype=np.float32)
                    post_scaled = np.ascontiguousarray(post_np / np.float32(BEHAVIOR_SCALING_FACTOR), dtype=np.float32)
                    outputs.append(np.ascontiguousarray(native_scaled + float(beta) * (post_scaled - native_scaled), dtype=np.float32))
        return np.ascontiguousarray(np.concatenate(outputs, axis=0), dtype=np.float32)
